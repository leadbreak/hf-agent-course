# Unit 4 Agent From Scratch Course

이 폴더는 Hugging Face Agents Course Unit 4 과제를 처음 배우는 사람을 위한 실습형 강의 자료입니다.

기존 `Final_Assignment_Template`와 `practice` 구현을 바로 읽으면 기능이 많아서 어렵습니다. 여기서는 같은 문제를 더 작은 순서로 나눕니다.

1. 정답 문자열을 깨끗하게 만드는 법
2. trace로 에이전트가 한 일을 남기는 법
3. LLM 없이 풀 수 있는 문제를 먼저 코드로 푸는 법
4. 첨부파일, HF API, 제출 앱을 나중에 붙이는 법
5. 우리가 겪은 실수, 특히 `unknown`, `402 Payment Required`, ASR bytes 오류, 정답키 fallback 오해를 복기하는 법

## 빠른 시작

```bash
cd /workspace/qscar/hf-agent-course/lectures/unit4/agent_from_scratch_course
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
jupyter notebook notebooks/01_agent_from_scratch.ipynb
```

`HF_TOKEN`이 없어도 앞부분 실습과 deterministic handler 실습은 실행됩니다.

HF API 실습을 하려면:

```bash
export HF_TOKEN=hf_...
export HF_MODEL_ID=openai/gpt-oss-120b:fastest
export HF_PROVIDER=auto
```

## `Use public validation fallback`은 무엇인가?

Unit 4 validation 문제의 공개 정답키를 읽어, 에이전트가 답을 못 냈을 때 그 정답을 대신 넣는 디버그 기능입니다.

체크하지 않으면:

- 에이전트는 deterministic handler, 파일 도구, HF API 같은 실제 풀이 경로만 사용합니다.
- HF API가 `402 Payment Required`로 실패하면 많은 문제가 `unknown`이 될 수 있습니다.
- 이 모드의 점수가 실제 에이전트 성능에 가깝습니다.

체크하면:

- 실제 풀이 경로가 실패한 뒤 공개 validation 정답키를 fallback으로 사용합니다.
- `unknown`은 크게 줄거나 사라질 수 있습니다.
- 하지만 이것은 정답 누수입니다. 학습/디버깅/제출 흐름 확인용이지, 에이전트 성능 측정이 아닙니다.

그래서 이 강의 자료에서는 항상 `debug_answer_key_fallback`이라는 이름으로 부릅니다. 그냥 `fallback`이라고 부르면 실제 추론 기능처럼 착각하기 쉽기 때문입니다.

## 산출물

- `docs/agent_from_scratch_lecture.md`: 초심자용 상세 강의 문서
- `notebooks/01_agent_from_scratch.ipynb`: 직접 실행하는 단계별 노트북
- `src/learning_agent.py`: 노트북에서 import하는 작은 학습용 에이전트
- `tests/test_learning_agent.py`: 핵심 동작 테스트
