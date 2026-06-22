import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from gradio_client import Client

QUESTIONS_URL = "https://agents-course-unit4-scoring.hf.space/questions"
SUBMIT_URL = "https://agents-course-unit4-scoring.hf.space/submit"
ANSWER_KEY_URL = "https://huggingface.co/spaces/bstraehle/gaia/resolve/main/files/gaia_validation.jsonl"
DEFAULT_LOCAL_URL = "http://localhost:7860"
DEFAULT_CSV_PATH = "gaia_agent_local_test_results.csv"
DEFAULT_TRACE_DIR = ".agent2_cache/traces"


def cache_key(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_for_compare(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    try:
        number = float(text.replace(",", ""))
        if number.is_integer():
            return str(int(number))
        return f"{number:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    return " ".join(text.lower().split())


def is_correct_answer(predicted: object, actual: object) -> bool:
    return normalize_for_compare(predicted) == normalize_for_compare(actual)


def build_answers_payload(rows: Iterable[dict]) -> list[dict[str, str]]:
    payload = []
    for row in rows:
        payload.append(
            {
                "task_id": str(row["Task ID"]),
                "submitted_answer": str(row.get("Predicted Answer", "unknown")).strip() or "unknown",
            }
        )
    return payload


def load_trace(question: str, trace_dir: str = DEFAULT_TRACE_DIR) -> dict:
    path = Path(trace_dir) / f"{cache_key(question)}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_trace(trace: dict) -> str:
    events = trace.get("events") or []
    if not events:
        return "trace unavailable"

    lines = []
    for idx, event in enumerate(events, 1):
        stage = event.get("stage", "?")
        status = event.get("status", "?")
        message = event.get("message", "")
        details = event.get("details") or {}
        detail_parts = []
        for key in ("tool", "file", "url", "query", "answer", "final_answer", "total", "seconds", "error"):
            if key in details:
                value = str(details[key]).replace("\n", " ")
                if len(value) > 180:
                    value = value[:177] + "..."
                detail_parts.append(f"{key}={value}")
        suffix = f" ({'; '.join(detail_parts)})" if detail_parts else ""
        lines.append(f"{idx:02d}. [{stage}/{status}] {message}{suffix}")
    return "\n".join(lines)


def fetch_questions() -> list[dict]:
    response = requests.get(QUESTIONS_URL, timeout=15)
    response.raise_for_status()
    return response.json()


def load_answer_key(cache_path: str = ".agent2_cache/gaia_validation_answers.jsonl") -> dict[str, str]:
    path = Path(cache_path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(ANSWER_KEY_URL, timeout=30)
        response.raise_for_status()
        text = response.text
        path.write_text(text, encoding="utf-8")

    answer_key = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        task_id = str(item.get("task_id", "")).strip()
        if task_id:
            answer_key[task_id] = str(item.get("Final answer", "")).strip()
    return answer_key


def submit_answers(rows: list[dict], hf_username: str, agent_space_url: str) -> dict:
    request_data = {
        "username": hf_username,
        "agent_code": agent_space_url,
        "answers": build_answers_payload(rows),
    }
    response = requests.post(
        SUBMIT_URL,
        json=request_data,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_local_evaluation(local_agent_url: str, trace_dir: str = DEFAULT_TRACE_DIR, show_trace: bool = True) -> list[dict]:
    print("=" * 80)
    print(f"📡 Connecting to Agent Server via Gradio Client: {local_agent_url}")
    print("=" * 80)

    print("📥 Fetching tasks and local answer key...")
    questions_data = fetch_questions()
    answer_key = load_answer_key()
    print(f"✅ Loaded {len(questions_data)} benchmarks.")
    print(f"✅ Loaded {len(answer_key)} answer-key rows for local scoring.\n")

    client = Client(local_agent_url)
    results_log = []

    for idx, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        actual_answer = answer_key.get(str(task_id), "")
        print(f"[{idx}/{len(questions_data)}] Task ID: {task_id}")
        print(f"📋 Question: {question_text[:90].replace(chr(10), ' ')}...")

        try:
            result = client.predict(question=question_text, api_name="/predict")
            predicted_answer = str(result).strip()
        except Exception as exc:
            print(f"❌ Inference/Tunnel Error: {exc}")
            predicted_answer = "ERROR"

        correct = is_correct_answer(predicted_answer, actual_answer)
        trace = load_trace(question_text, trace_dir)
        trace_text = format_trace(trace)
        status = "✅ CORRECT" if correct else "❌ WRONG"
        print(f"➡️ Predicted: {predicted_answer}")
        print(f"🎯 Actual   : {actual_answer}")
        print(f"📊 Local    : {status}")
        if show_trace:
            print("🧭 Trace:")
            print(trace_text)
        print("-" * 80)

        results_log.append(
            {
                "Task ID": task_id,
                "Question": question_text,
                "Predicted Answer": predicted_answer,
                "Actual Answer": actual_answer,
                "Local Correct": correct,
                "Trace": trace_text,
            }
        )

    return results_log


def print_local_summary(rows: list[dict]) -> None:
    correct_count = sum(1 for row in rows if row["Local Correct"])
    total = len(rows)
    accuracy = (correct_count / total * 100) if total else 0.0
    print("\n" + "=" * 80)
    print(f"📊 Local Score: {accuracy:.1f}% ({correct_count}/{total})")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local GAIA agent evaluation and optionally submit to HF.")
    parser.add_argument("--local-url", default=os.getenv("LOCAL_AGENT_URL", DEFAULT_LOCAL_URL).rstrip("/"))
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--trace-dir", default=os.getenv("AGENT2_TRACE_DIR", DEFAULT_TRACE_DIR))
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--submit", action="store_true", default=os.getenv("SUBMIT_TO_HF", "0") == "1")
    parser.add_argument("--hf-username", default=os.getenv("HF_USERNAME", "QscarKIM"))
    parser.add_argument(
        "--agent-space-url",
        default=os.getenv(
            "AGENT_SPACE_URL",
            "https://huggingface.co/spaces/QscarKIM/Final_Assignment_Template/tree/main",
        ),
    )
    args = parser.parse_args()

    rows = run_local_evaluation(args.local_url, trace_dir=args.trace_dir, show_trace=not args.no_trace)
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 로컬 테스트 런 완료. 결과가 {args.csv} 에 저장되었습니다.")
    print_local_summary(rows)

    if args.submit:
        print(f"\n🚀 Submitting {len(rows)} answers to Hugging Face scoring server...")
        try:
            result = submit_answers(rows, args.hf_username, args.agent_space_url)
            print("🏅 Official submission response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"❌ Official submission failed: {exc}")
    else:
        print("\nℹ️ 공식 제출은 건너뜁니다. 제출하려면 `python test.py --submit` 또는 `SUBMIT_TO_HF=1`을 사용하세요.")


if __name__ == "__main__":
    main()
