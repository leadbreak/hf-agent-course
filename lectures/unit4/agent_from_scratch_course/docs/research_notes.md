# Research Notes

## Hugging Face API와 Spaces

- `huggingface_hub.InferenceClient`는 Hugging Face Inference Providers, Inference Endpoints, 로컬 vLLM/TGI/Ollama 같은 endpoint를 같은 Python 인터페이스로 호출할 수 있다.
- Inference Providers는 provider를 `auto`로 두면 자동 라우팅한다. 이 자동 라우팅은 코드가 간단하지만 실패 원인을 숨길 수 있다.
- `HF_TOKEN`은 코드에 넣지 말고 환경변수 또는 Space Secret으로 둔다.
- fine-grained token은 Inference Providers 호출 권한이 필요하다.
- `402 Payment Required`는 대개 코드 버그가 아니라 HF Inference Provider 크레딧/결제 문제다.
- Gradio Space에서 Unit 4 제출을 하려면 OAuth 로그인으로 username을 받고, `SPACE_ID`로 agent code URL을 만들 수 있다.

Sources:

- https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client
- https://huggingface.co/docs/inference-providers/index
- https://huggingface.co/docs/inference-providers/pricing
- https://huggingface.co/docs/hub/en/spaces-overview
- https://huggingface.co/docs/hub/spaces-oauth
- https://huggingface.co/learn/agents-course/unit4/hands-on

## 구현 복기에서 얻은 원칙

- GAIA/Unit 4류 과제는 단순 QA가 아니다. 웹, 첨부파일, 오디오, 이미지, 표/문자열, exact-match 제출이 섞인 benchmark다.
- `unknown`은 정답이 아니라 파이프라인 실패 신호로 읽는다. trace가 없으면 모델 실패, API 실패, 도구 실패를 구분할 수 없다.
- `unknown`이 나오면 먼저 trace를 확인한다. 모델이 모르는 것인지, API가 실패한 것인지, 파일 도구가 실패한 것인지 구분해야 한다.
- exact-answer 과제에서는 최종 답 형식이 중요하다. `Answer: Paris`와 `Paris`는 다르게 채점될 수 있다.
- LLM보다 코드가 나은 문제가 있다. 연산표, 엑셀 합계, 파이썬 실행 결과는 deterministic handler가 더 빠르고 안정적이다.
- ASR/VQA 같은 modality 도구는 입력 타입을 확인해야 한다. 실제로 `Path`를 넘겨 실패했고, `path.read_bytes()`로 고쳤다.
- 공개 validation 정답키는 편리하지만 성능 측정이 아니다. 함수명, trace stage, UI label에서 debug 용도임을 드러내야 한다.
- cache hit 또는 public answer-key fallback으로 20/20이 나와도 실제 추론 성능이라고 말하면 안 된다. 재현/디버그 상태와 agent quality를 분리해야 한다.

## 노트북 교육 설계 원칙

- 하나의 progressive capstone notebook으로 만든다. 여러 노트북보다 같은 작은 agent가 셀마다 자라는 구조가 초심자에게 낫다.
- 네트워크/API 호출보다 fake in-memory task를 먼저 사용한다.
- 각 함수 바로 아래에 `assert` 테스트를 둔다.
- Gradio/OAuth/배포는 agent core를 이해한 뒤에 다룬다.
- trace는 콘솔 로그가 아니라 데이터로 보여준다.
- fallback 순서는 명시한다.

```text
1. deterministic/direct handler
2. real tool handler
3. honest HF API fallback
4. debug answer-key fallback, disabled by default
```

- `debug_answer_key_fallback`은 절대 “성능 개선”으로 설명하지 않는다.
