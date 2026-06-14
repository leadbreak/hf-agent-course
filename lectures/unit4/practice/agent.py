import os
import torch
import gradio as gr
from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool

MODEL_ID = "Qwen/Qwen3.5-0.8B"

print(f"📥 Loading lightweight agent engine ({MODEL_ID})...")
local_model = TransformersModel(
    model_id=MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

search_tool = DuckDuckGoSearchTool()

agent_core = CodeAgent(
    tools=[search_tool],
    model=local_model,
    add_base_tools=True,
    planning_interval=0,
    max_steps=5
)

def predict(question: str) -> str:
    print(f"\n[📥 Request Inbound] Task: {question[:60]}...")
    try:
        result = agent_core.run(question)
        return str(result).strip() if result else "unknown"
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return "unknown"

# TypeError 해결: gr.Textbox 내부에 존재하지 않는 name 인자 완전 제거
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.Textbox(label="Answer")
)

if __name__ == "__main__":
    # 내부 큐 프로토콜을 활성화하여 404 라우팅 오류 없이 gradio-client 통신 인터페이스 동기화
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)