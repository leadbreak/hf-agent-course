import os
import requests
import pandas as pd
from gradio_client import Client

QUESTIONS_URL = "https://agents-course-unit4-scoring.hf.space/questions"
DEFAULT_LOCAL_URL = "http://localhost:7860"

def main():
    # 환경 변수에 LOCAL_AGENT_URL이 없으면 로컬 기본 포트 자동 추적
    local_agent_url = os.getenv("LOCAL_AGENT_URL", DEFAULT_LOCAL_URL).rstrip("/")
    
    print("=" * 80)
    print(f"📡 Connecting to Agent Server via Gradio Client: {local_agent_url}")
    print("=" * 80)

    print(f"📥 Fetching tasks from scoring server...")
    try:
        response = requests.get(QUESTIONS_URL, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        print(f"✅ Loaded {len(questions_data)} benchmarks.\n")
    except Exception as e:
        print(f"❌ Failed to fetch questions: {e}")
        return

    # Gradio 내장 프로토콜 추적 파싱 클라이언트 가동
    try:
        client = Client(local_agent_url)
    except Exception as e:
        print(f"❌ Connection Refused to Gradio Server: {e}")
        return

    results_log = []
    for idx, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        print(f"[{idx}/{len(questions_data)}] Task ID: {task_id}")
        print(f"📋 Question: {question_text[:70]}...")

        try:
            # 내부 경로 버전을 타지 않고 객체 지향으로 결과 직결 매핑
            res = client.predict(question=question_text, api_name="/predict")
            predicted_answer = str(res).strip()
            print(f"➡️ Derived Answer: {predicted_answer}")
        except Exception as e:
            print(f"❌ Inference/Tunnel Error: {e}")
            predicted_answer = "ERROR"

        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Predicted Answer": predicted_answer
        })
        print("-" * 80)

    df = pd.DataFrame(results_log)
    df.to_csv("gaia_agent_local_test_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ 로컬 테스트 런 완료. 결과가 gaia_agent_local_test_results.csv 에 저장되었습니다.")

if __name__ == "__main__":
    main()