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

# gradio_client가 인식할 수 있도록 정석 Interface 컴포넌트 구성
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Question", name="question"),
    outputs=gr.Textbox(label="Answer", name="answer")
)

if __name__ == "__main__":
    # 장기 연산을 위해 queue()를 유지하되, gradio-client를 통해 통신 장벽을 허뭅니다.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)