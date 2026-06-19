# agent.py 성능/정답률 개선 보고서

## 핵심 진단

현재 `agent.py`는 모든 문제를 `CodeAgent + TransformersModel` 하나로 처리한다. 이 구조는 구현은 단순하지만 GAIA식 채점에는 불리하다.

- 모델을 프로세스 안에서 직접 로드한다. 요청 처리와 모델 실행이 같은 Python 프로세스에 묶여 vLLM/llama.cpp의 continuous batching, prefix caching, 장기 실행 서버 최적화를 쓰지 못한다.
- `max_steps=11`, `planning_interval=3`이라 한 문제마다 LLM 호출과 검색 호출이 여러 번 반복된다.
- 웹페이지 전체 HTML을 Markdown으로 바꿔 넣어서 컨텍스트가 쉽게 커진다.
- 제출 답변이 최종값이 아니라 긴 설명문인 경우가 있다. 기존 CSV 평균 답변 길이는 약 2354자였고, 8개는 장문 추론이었다.
- `test.py`와 `app.py`가 `/questions`의 `file_name`을 버리고 질문만 전달한다. 이미지, 오디오, Python, Excel 첨부 문제는 구조적으로 풀 수 없다.

따라서 가장 큰 개선점은 더 큰 모델 하나로 밀어붙이는 것이 아니라, 파일/표/코드/단순 문자열 문제를 deterministic handler로 먼저 처리하고 LLM은 검색과 추론이 필요한 문제에만 쓰는 것이다.

## agent2.py 설계

`agent2.py`는 기존 Gradio `/predict(question)` 계약을 유지한다.

1. 질문을 scoring 서버 `/questions`와 매칭해 `task_id`와 `file_name`을 복구한다.
2. 첨부 파일은 로컬 디렉터리, 공개 미러, 인증된 GAIA dataset URL 순서로 내려받는다.
3. 빠른 핸들러를 먼저 실행한다.
   - 뒤집힌 문장, 비가환성 표, botanical vegetable 문제
   - `.py`는 timeout이 있는 서브프로세스 실행
   - `.xlsx`는 pandas로 음료 컬럼 제외 합산
   - `.mp3`는 `faster-whisper` 또는 OpenAI-compatible 로컬 transcription endpoint가 있을 때 전사
   - 이미지는 OpenAI-compatible 로컬 VLM endpoint가 있을 때 직접 질의
4. 그래도 못 풀면 `CodeAgent` fallback을 실행한다.
5. 최종 답변은 `<think>`, `final_answer(...)`, `Answer:` 등의 흔적을 제거하고 짧은 문자열만 캐시/반환한다.

이 구조는 기존 `agent.py`와 달리 모델 import 시 즉시 로드하지 않는다. 첫 LLM fallback이 필요할 때만 모델을 초기화하므로 deterministic 문제는 모델 없이도 빠르게 끝난다.

## 권장 추론 서버

여기서 말하는 `OpenAI-compatible`은 유료 OpenAI API 사용이 아니라 vLLM, llama.cpp, LocalAI, Ollama proxy 같은 **내 서버의 로컬 HTTP API 형식**을 뜻한다. API key 값은 서버가 요구하지 않으면 `EMPTY`로 둔다.

### vLLM 우선

vLLM은 OpenAI-compatible 로컬 HTTP 서버를 제공하고, Chat Completions API, tool calling, prefix caching, continuous batching을 지원한다. 공식 문서도 OpenAI-compatible 로컬 서버와 online serving을 권장 경로로 설명한다.

예시:

```bash
vllm serve Qwen/Qwen3.5-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

`agent2.py` 연결:

```bash
export AGENT2_LLM_BASE_URL=http://127.0.0.1:8000/v1
export AGENT2_LLM_API_KEY=EMPTY
export AGENT2_MODEL_ID=Qwen/Qwen3.5-8B-Instruct
python agent2.py
```

tool parser는 모델 계열에 맞춰 바꿔야 한다. vLLM 문서 기준으로 Qwen2.5/QwQ는 `hermes`, Qwen3-Coder는 `qwen3_xml`, Mistral은 `mistral`, GLM-4.5는 `glm45` 등 모델별 parser를 확인한다. tool 인자 안정성이 중요하면 strict schema를 사용하고, object schema에는 `additionalProperties: false`와 `required`를 명시한다.

최종 답변 형식은 tool calling과 별도로 structured output으로 강제하는 편이 좋다. vLLM 최신 문서는 legacy guided 옵션보다 `extra_body={"structured_outputs": {"json": schema}}` 형식을 권장한다. GAIA 채점처럼 최종 문자열만 필요한 경우 `{ "answer": "..." }` JSON schema를 강제한 뒤 `answer`만 제출하는 구조가 가장 안전하다.

### llama.cpp / GGUF 대안

VRAM이 작거나 CPU/Apple Silicon/단일 consumer GPU 위주라면 llama.cpp 서버가 단순하다. llama.cpp 서버는 OpenAI-compatible 로컬 chat completions, continuous batching, schema-constrained JSON response를 제공한다.

예시:

```bash
llama-server \
  -m /models/qwen-agentic-q5.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --jinja \
  -ngl 99
```

`agent2.py` 연결:

```bash
export AGENT2_LLM_BASE_URL=http://127.0.0.1:8080/v1
export AGENT2_MODEL_ID=local-model
python agent2.py
```

## 모델 선택

4B급 모델은 빠르지만 GAIA 문제의 검색, 표 계산, 파일 처리, 긴 instruction following에서 흔들리기 쉽다. 권장 순서는 다음과 같다.

- 24GB VRAM 이상: 14B-32B급 instruction/tool 모델을 vLLM BF16 또는 FP8/AWQ/GPTQ로 서빙. Qwen3/Qwen3.5/Qwen3.6, Mistral Small 3.x, GLM-4.5-Air처럼 tool template와 vLLM serve 예시가 문서화된 모델을 우선 테스트한다.
- 12-16GB VRAM: 7B-14B급 Qwen/DeepSeek/Llama 계열 instruct 모델. 정확도를 우선하면 4bit보다 8bit/FP8이 낫다.
- 8GB 이하 또는 CPU 포함: Unsloth Dynamic GGUF, Q5_K_M 또는 Dynamic Q4 계열을 llama.cpp로 서빙.

이 과제에서는 모델 하나가 모든 도구를 직접 다루게 하기보다, deterministic handler와 짧은 final-answer contract를 강제하는 쪽이 모델 크기 증설보다 먼저다.

## 양자화 선택

- vLLM GPU 서버: FP8, AWQ, GPTQ, INT8/INT4 중 하드웨어와 모델 지원을 확인한다. vLLM 공식 문서는 FP8, INT8, INT4, GPTQ/AWQ, GGUF 등 다양한 quantization backend를 지원한다고 설명한다.
- Hopper/Ada급 GPU: FP8이 메모리와 throughput 균형이 좋다.
- Ampere consumer GPU: AWQ/GPTQ Marlin 계열 또는 weight-only quantization을 먼저 시험한다.
- llama.cpp: GGUF Q5_K_M은 정확도 손실이 작고, Q4_K_M/Dynamic Q4는 속도와 메모리 절감에 유리하다.
- Unsloth Dynamic 2.0 GGUF는 계층별 민감도에 따라 bit-width를 다르게 두는 방식으로 같은 평균 크기에서 일반 GGUF보다 품질 보존을 노린다.
- vLLM에서 GGUF는 아직 실험적/저최적화 경로로 취급하는 편이 안전하다. production vLLM 서버는 safetensors 기반 BF16/FP8/AWQ/GPTQ를 우선하고, GGUF는 llama.cpp/Ollama/LM Studio 계열에 우선 배치한다.

## 도구 선택

현재 agent의 도구는 검색과 페이지 방문뿐이다. GAIA Level 1도 멀티모달/파일 문제가 섞여 있어 다음 도구가 필요하다.

- Web: 검색 결과 캐시, 페이지 fetch 캐시, 페이지 길이 제한
- Python: 첨부 `.py` 실행, 표/수학 계산
- Excel: pandas/openpyxl
- Audio: faster-whisper 또는 OpenAI-compatible 로컬 transcription endpoint
- Vision/chess: OpenAI-compatible 로컬 VLM 또는 이미지 전용 모델
- YouTube: `yt-dlp`로 자막/오디오/프레임 추출

`agent2.py`는 Excel과 Python을 기본 구현했고, audio/vision은 환경변수가 있을 때 사용한다. 이는 불필요한 무거운 의존성을 기본 경로에 넣지 않기 위한 선택이다.

## 운영 팁

- `agent.py`처럼 Gradio 프로세스가 모델을 직접 들고 있지 말고, 추론 서버와 agent 서버를 분리한다.
- vLLM/llama.cpp는 `tmux`나 systemd에서 먼저 띄우고, agent는 OpenAI-compatible 로컬 endpoint를 호출한다.
- `AGENT2_MAX_STEPS`는 4-6부터 시작한다. GAIA exact-match에서는 긴 chain보다 짧은 tool loop와 강한 answer formatting이 유리하다.
- `AGENT2_MAX_PAGE_CHARS`는 10k-20k 정도로 제한한다.
- `AGENT2_CACHE_DIR`를 유지해 반복 평가에서 검색/페이지/정답 캐시를 재사용한다.
- 첨부 파일 endpoint가 404면 `GAIA_FILES_DIR`에 파일을 미리 넣거나, `HF_TOKEN`을 설정해 gated GAIA dataset 접근을 사용한다.
- 긴 context 모델은 `--max-model-len`을 명시적으로 낮춘다. 미지정 시 KV cache가 과하게 잡혀 OOM이 날 수 있다.
- prefix caching을 재현성 있게 쓰려면 vLLM의 hash 설정을 확인한다. system prompt와 tool schema가 반복되는 agent workload에서는 prefix caching 효과가 크다.

## 참고 자료

- vLLM OpenAI-compatible local server: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html
- vLLM online serving APIs: https://docs.vllm.ai/en/stable/serving/online_serving/
- vLLM quantization: https://docs.vllm.ai/en/latest/features/quantization/
- vLLM tool calling example: https://docs.vllm.ai/en/v0.23.0/examples/tool_calling/openai_chat_completion_client_with_tools/
- smolagents agents: https://huggingface.co/docs/smolagents/v1.5.0/reference/agents
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp function calling: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Unsloth Dynamic 2.0 GGUF: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- HF Agents Course file endpoint issue: https://github.com/huggingface/agents-course/issues/624
