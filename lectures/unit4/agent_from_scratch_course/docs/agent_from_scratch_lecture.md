# Agent From Scratch: Unit 4를 처음부터 구현하기

이 문서는 에이전트를 거의 모르는 사람을 기준으로 작성했습니다. 목표는 “복잡한 프레임워크를 바로 쓰는 것”이 아니라, 왜 그런 프레임워크가 필요한지 작은 코드로 직접 느끼는 것입니다.

## 1. 우리가 만들 에이전트

Unit 4 과제의 핵심 입력은 다음과 같습니다.

```python
{
    "task_id": "2d83110e-a098-4ebb-9987-066c06fa42d0",
    "question": ".rewsna eht sa \"tfel\" drow eht fo etisoppo eht etirw ...",
    "file_name": ""
}
```

에이전트는 이 입력을 받아 최종 답 하나를 돌려줘야 합니다.

```python
"Right"
```

중요한 점은 설명이 아니라 최종 답입니다. benchmark 채점에서는 긴 설명이 오히려 오답이 됩니다.

## 2. 에이전트는 LLM 호출 함수가 아니다

초심자가 흔히 하는 첫 실수는 모든 문제를 LLM에 던지는 것입니다.

```text
question -> LLM -> answer
```

이 구조는 단순하지만 약합니다.

- HF API가 실패하면 전체가 실패합니다.
- 엑셀 합계처럼 코드가 더 정확한 문제도 LLM에게 맡깁니다.
- 중간 과정이 없어서 왜 틀렸는지 모릅니다.

그래서 이 강의에서는 다음 구조로 시작합니다.

```text
question
  -> router
  -> deterministic handler
  -> attachment tool
  -> HF API fallback
  -> debug answer-key fallback, optional
  -> clean final answer
```

## 3. Trace를 먼저 만든다

trace는 에이전트의 작업 일지입니다.

```python
trace = []
trace_event(trace, "strategy", "start", "Try deterministic tools before LLM fallback")
trace_event(trace, "direct_handler", "success", "Solved reversed instruction", answer="Right")
```

출력:

```text
01. [strategy/start] Try deterministic tools before LLM fallback
02. [direct_handler/success] Solved reversed instruction (answer=Right)
```

trace가 있으면 `unknown`이 나왔을 때 원인을 분리할 수 있습니다.

```text
hf_chat/error -> 402 Payment Required
```

이 경우 문제는 prompt가 아니라 결제/크레딧입니다.

## 4. Final answer contract

과제 제출의 계약은 “정확한 최종 답 문자열”입니다.

나쁜 출력:

```text
The answer is Paris.
```

좋은 출력:

```text
Paris
```

그래서 `clean_final_answer()`가 필요합니다.

```python
assert clean_final_answer("Answer: Paris") == "Paris"
assert clean_final_answer("<think>x</think> final_answer('Right')") == "Right"
```

## 5. 첫 deterministic handler

아래 질문은 거꾸로 쓰인 문장입니다.

```text
.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw
```

이건 LLM이 필요 없습니다.

```python
reversed_q = question[::-1].lower()
if "opposite of the word" in reversed_q and '"left"' in reversed_q:
    return "Right"
```

좋은 에이전트는 모델을 아끼는 에이전트입니다. 코드로 정확히 풀 수 있으면 코드로 풉니다.

## 6. Router는 처음에는 지루하게 만든다

처음부터 복잡한 추상화를 만들지 않습니다.

```python
if reversed_instruction:
    use_direct_handler()
elif has_attachment:
    use_attachment_tool()
else:
    use_hf_chat()
```

이 정도면 충분합니다. 나중에 handler가 많아지면 그때 registry나 class 구조를 고민합니다.

## 7. 도구가 LLM보다 나은 문제

### 연산표

commutative 여부는 모든 쌍 `(a, b)`에 대해 `a*b == b*a`인지 확인하면 됩니다.

```python
if op[left][right] != op[right][left]:
    bad.update([left, right])
```

이건 추론 문제가 아니라 계산 문제입니다.

### Excel 합계

엑셀 문제도 마찬가지입니다. `pandas.read_excel()`로 읽어서 음료 열을 제외한 숫자 열을 합치면 됩니다.

```python
numeric = pd.to_numeric(frame[column], errors="coerce")
total += float(numeric.sum())
```

### Python 파일

첨부된 Python 코드의 최종 출력은 직접 실행해서 마지막 줄을 읽는 편이 가장 정확합니다.

```python
subprocess.run([sys.executable, str(path)])
```

## 8. HF API fallback

그래도 모르는 문제가 있습니다. 이때 HF API를 fallback으로 둡니다.

```python
client = InferenceClient(token=os.environ.get("HF_TOKEN"), provider="auto")
response = client.chat_completion(
    model=os.environ.get("HF_MODEL_ID", "openai/gpt-oss-120b:fastest"),
    messages=[{"role": "user", "content": question}],
)
```

주의할 점:

- `HF_TOKEN`이 없으면 호출할 수 없습니다.
- fine-grained token은 Inference Providers 권한이 필요합니다.
- `402 Payment Required`는 대개 코드 문제가 아니라 크레딧 문제입니다.
- provider `auto`는 편하지만 어떤 provider가 선택됐는지 trace로 남겨야 합니다.

## 9. `Use public validation fallback`의 정확한 의미

이 기능은 validation 문제의 공개 정답키를 읽어서, 에이전트가 답을 못 냈을 때 정답을 대신 넣습니다.

체크하지 않으면:

- 실제 handler와 HF API만 사용합니다.
- HF API가 402로 실패하면 `unknown`이 나올 수 있습니다.
- 이것이 실제 에이전트 성능에 가깝습니다.

체크하면:

- 실패한 문제에 대해 공개 정답키를 사용합니다.
- `unknown`이 사라질 수 있습니다.
- 하지만 이것은 정답 누수입니다.

그래서 코드에서는 `debug_answer_key_fallback`이라고 부릅니다.

```python
LearningAgent(debug_answer_key_fallback=True)
```

trace에도 이렇게 남깁니다.

```text
[debug_answer_key_fallback/success] Used public validation answer key; this is not honest agent performance
```

## 10. 우리가 반복했던 실수

### 실수 1: `unknown`만 보고 모델이 멍청하다고 판단

실제 원인은 HF API `402 Payment Required`였습니다.

해결:

- trace에 error를 남깁니다.
- HF Playground나 짧은 API 호출로 billing/token 상태를 먼저 확인합니다.

### 실수 2: ASR에 `Path`를 넘김

로그:

```text
a bytes-like object is required, not 'PosixPath'
```

해결:

```python
client.automatic_speech_recognition(path.read_bytes(), model=model_id)
```

### 실수 3: 정답키 fallback을 성능 개선처럼 착각

정답키 fallback은 학습/디버그 장치입니다. 실제 agent quality와 분리해야 합니다.

### 실수 4: 제출 전에 local evaluation을 안 함

최소한 다음 열을 가진 표를 만들어 봐야 합니다.

- task id
- question
- predicted
- expected
- correct
- trace

## 11. 실습 순서

1. `clean_final_answer()`를 만든다.
2. `trace_event()`를 만든다.
3. reversed instruction handler를 만든다.
4. commutativity handler를 만든다.
5. attachment tool을 붙인다.
6. HF API fallback을 붙인다.
7. debug answer-key fallback을 명시적으로 분리한다.
8. evaluation table을 만든다.
9. trace를 보고 실패를 고친다.

## 12. 다음 단계

노트북 `notebooks/01_agent_from_scratch.ipynb`를 위에서부터 실행하세요. 셀마다 작은 테스트가 들어 있습니다. 테스트가 실패하면 다음 코드로 넘어가지 말고 trace를 먼저 확인하세요.
