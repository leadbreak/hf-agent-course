import os
import requests
import pandas as pd

# --- 채점 서버 및 로컬 에이전트 기본값 설정 ---
QUESTIONS_URL = "https://agents-course-unit4-scoring.hf.space/questions"
DEFAULT_LOCAL_URL = "http://localhost:7860"

def main():
    # 환경 변수에서 로컬 에이전트 서버 주소 로드 (없으면 localhost:7860 사용)
    local_agent_base = os.getenv("LOCAL_AGENT_URL", DEFAULT_LOCAL_URL).rstrip("/")
    local_agent_predict_url = f"{local_agent_base}/api/predict"

    print("=" * 80)
    print(f"📡 타겟 에이전트 서버 엔드포인트: {local_agent_predict_url}")
    print("=" * 80)

    # 1. 채점 서버로부터 실시간 GAIA 평가 문항 다운로드
    print(f"📥 채점 서버에서 문항 다운로드 중: {QUESTIONS_URL}")
    try:
        response = requests.get(QUESTIONS_URL, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        print(f"✅ 총 {len(questions_data)}개의 테스트 문항을 확보했습니다.\n")
    except Exception as e:
        print(f"❌ 문항 다운로드 실패: {e}")
        return

    # 2. 로컬 agent.py 서버로 질의 릴레이 루프 가동
    results_log = []
    print("🚀 로컬 에이전트 서버 기반 벤치마크 테스트를 시작합니다.")
    print("-" * 80)

    for idx, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue

        print(f"[{idx}/{len(questions_data)}] Task ID: {task_id}")
        print(f"📋 질문: {question_text[:90]}...")

        try:
            # 구동 중인 agent.py(Gradio 인터페이스) 규격에 맞춰 POST 요청
            res = requests.post(
                local_agent_predict_url,
                json={"data": [question_text]},
                timeout=300  # 에이전트 코드 컴파일 및 서핑 자율 예산 시간 확보
            )
            
            if res.status_code == 200:
                predicted_answer = res.json()["data"][0]
                print(f"➡️ 도출된 정답: {predicted_answer}")
            else:
                print(f"⚠️ 에이전트 서버 응답 실패 (HTTP {res.status_code})")
                predicted_answer = "unknown"
                
        except Exception as e:
            print(f"❌ 에이전트 서버 통신 실패 (Connection Refused 또는 Timeout): {e}")
            predicted_answer = "CONNECTION_ERROR"

        results_log.append({
            "Task ID": task_id,
            "Question": question_text,
            "Predicted Answer": predicted_answer
        })
        print("-" * 80)

    # 3. 결과 분석용 로컬 CSV 파일 덤프
    df = pd.DataFrame(results_log)
    output_csv = "gaia_agent_local_test_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 80)
    print("🎉 에이전트 서버 릴레이 테스트 완료!")
    print(f"💾 결과 데이터가 '{output_csv}' 파일로 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()