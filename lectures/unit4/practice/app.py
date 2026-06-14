import os
import gradio as gr
import requests
import pandas as pd

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if not profile:
        return "Hugging Face 로그인이 필요합니다.", None

    # 스페이스 Secrets 환경 변수에서 로컬 에이전트 주소 동적 로드
    local_agent_url = os.getenv("LOCAL_AGENT_URL")
    if not local_agent_url:
        return "Error: LOCAL_AGENT_URL Secret variable is not configured in Space Settings.", None

    # Gradio Interface API 표준 예측 엔드포인트 포맷 보정
    if not local_agent_url.endswith("/api/predict"):
        local_agent_url = local_agent_url.rstrip("/") + "/api/predict"

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 채점 서버로부터 GAIA 문제셋 다운로드
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return f"문항 추출 에러: {e}", None

    results_log = []
    answers_payload = []
    
    print(f"Relaying tasks to remote agent server: {local_agent_url}")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            continue
            
        try:
            # 로컬 agent.py 터널로 추론 요청 송신 (연산 버든 분리)
            res = requests.post(
                local_agent_url, 
                json={"data": [question_text]}, 
                timeout=240  # 로컬 에이전트 자율 서핑 시간을 위해 4분 확보
            )
            submitted_answer = res.json()["data"][0] if res.status_code == 200 else "unknown"
        except Exception as e:
            print(f"Tunneling Error for task {task_id}: {e}")
            submitted_answer = "unknown"

        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})

    if not answers_payload:
        return "Agent produced no answers to submit.", pd.DataFrame(results_log)

    # 채점 최종 서브밋 페이로드 구성
    submission_data = {
        "username": profile.username.strip(), 
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main", 
        "answers": answers_payload
    }
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    
    try:
        sub_res = requests.post(submit_url, json=submission_data, timeout=60)
        sub_res.raise_for_status()
        result_data = sub_res.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        status_message = f"Submission Failed: {e}"
        return status_message, pd.DataFrame(results_log)

# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Remote Lightweight Agent Proxy")
    gr.Markdown(
        """
        **Instructions:**
        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to relay questions to your local compute cluster.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    
    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)