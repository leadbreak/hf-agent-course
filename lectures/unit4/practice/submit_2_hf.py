import os
import json
import pandas as pd
import requests

def submit_local_results_to_hf(
    csv_path="gaia_agent_local_test_results.csv",
    hf_username="QscarKIM",           # 본인의 Hugging Face 유저명을 입력하세요
    agent_space_url=f"https://huggingface.co/spaces/QscarKIM/Final_Assignment_Template/tree/main" # 검증용 Space 코드 주소
):
    # 1. 로컬에 저장된 에이전트 결과 CSV 로드
    if not os.path.exists(csv_path):
        print(f"❌ 에러: 채점 대상 파일인 '{csv_path}' 파일이 존재하지 않습니다.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"📋 로컬 파일 '{csv_path}'에서 {len(df)}개의 에이전트 실행 기록을 읽어왔습니다.")

    # 2. 유연한 컬럼 자동 매핑 (task_id 및 prediction 추출)
    id_cols = [c for c in df.columns if 'task' in c.lower() or 'id' in c.lower()]
    pred_cols = [c for c in df.columns if 'pred' in c.lower() or 'answer' in c.lower() or 'output' in c.lower()]

    if not id_cols or not pred_cols:
        print("❌ 에러: CSV에서 'task_id' 또는 에이전트의 'prediction' 컬럼을 찾을 수 없습니다.")
        print(f"현재 컬럼 목록: {list(df.columns)}")
        return

    id_col = id_cols[0]
    pred_col = pred_cols[0]

    # 3. Hugging Face 공식 API 규격에 맞게 페이로드 빌드
    # 포맷 가이드라인: {"task_id": ..., "submitted_answer": ...}
    answers_payload = []
    for _, row in df.iterrows():
        # 빈 값 방어 처리
        submitted_val = str(row[pred_col]).strip() if pd.notna(row[pred_col]) else "unknown"
        
        answers_payload.append({
            "task_id": str(row[id_col]),
            "submitted_answer": submitted_val
        })

    # 전체 요청 본문(Payload) 바인딩
    request_data = {
        "username": hf_username,
        "agent_code": agent_space_url,
        "answers": answers_payload
    }

    # 4. Hugging Face Unit 4 Scoring API 호출
    scoring_url = "https://agents-course-unit4-scoring.hf.space/submit"
    print(f"🚀 Hugging Face 공식 채점 서버로 데이터를 전송합니다... (대상 유저: {hf_username})")
    
    try:
        response = requests.post(
            scoring_url, 
            json=request_data, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # 5. 결과 파싱 및 스코어 리포팅
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("🏅 [Hugging Face Unit 4 Final Score Result]")
            # API 반환 규격에 맞춰 스코어 표출 (기본값 대응)
            score = result.get("score", result.get("total_score", "N/A"))
            total = result.get("total", len(answers_payload)*5)
            
            print(f" • 채점 성공 여부 : SUCCESS")
            print(f" • 최종 획득 점수 : {score} / {total}")
            if "accuracy" in result:
                print(f" • 공식 정확도   : {result['accuracy']}%")
            print("="*60 + "\n")
        else:
            print(f"❌ 채점 서버 응답 실패 (HTTP {response.status_code})")
            print(f"↳ 서버 메세지: {response.text}")

    except Exception as e:
        print(f"❌ API 통신 중 네트워크 에러 발생: {e}")

if __name__ == "__main__":
    # 코드 실행 전 본인의 Hugging Face 정보로 수정하여 가동하십시오.
    submit_local_results_to_hf(
        hf_username="QscarKIM", # 실제 수료증과 연동될 계정명 명시
    )