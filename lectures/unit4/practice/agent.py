import os
import torch
import warnings
import requests
import gradio as gr
from markdownify import markdownify
from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool, Tool

# 전역 노이즈 묵음
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# 🛠️ Wikipedia 403 차단 우회용 커스텀 웹 방문 도구
class SafeVisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage URL and returns its content converted to Markdown text."
    inputs = {"url": {"type": "string", "description": "The target URL to visit."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return markdownify(response.text)
        except Exception as e:
            return f"Error fetching the webpage: {e}"

MODEL_ID = "Qwen/Qwen3.5-4B"
local_model = TransformersModel(model_id=MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)

# 할당된 컴퓨트 디바이스 상태 출력 (1줄 고정)
try:
    print(f"\n🚀 [Compute Device Assigned] {next(local_model.model.parameters()).device} ({MODEL_ID})")
except Exception:
    pass

# 오픈소스 토큰 정렬 가드레일 (출력 없음)
try:
    hf_model = local_model.model
    if hf_model and hasattr(hf_model, "config"):
        eos_id = hf_model.config.eos_token_id
        if isinstance(eos_id, list) and len(eos_id) > 0: eos_id = eos_id[0]
        if eos_id is not None:
            hf_model.config.pad_token_id = eos_id
            if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
                hf_model.generation_config.pad_token_id = eos_id
except Exception:
    pass

# ==============================================================================
# 🤖 [정석 바인딩] 내장 툴 레지스트리 메모리 직접 치환
# ==============================================================================
search_tool = DuckDuckGoSearchTool()
custom_visit_tool = SafeVisitWebpageTool()

agent_core = CodeAgent(
    tools=[search_tool],  # 중복 충돌을 피하기 위해 검색 툴만 선포입
    model=local_model,
    add_base_tools=True,  # 멀티모달성 질문 대응을 위해 내장 툴 풀셋 활성화
    planning_interval=3,  
    max_steps=11           
)

# 인스턴스 샌드박스가 완성된 직후, 구형 순정 툴을 커스텀 우회형 객체로 메모리 맵 스왑
agent_core.tools[custom_visit_tool.name] = custom_visit_tool
# ==============================================================================

def predict(question: str) -> str:
    print(f"\n[📥 Task Received] {question[:60]}...")
    managed_prompt = (
        f"Task Question:\n{question}\n\n"
        f"----- CRITICAL EXECUTING RULES -----\n"
        f"1. You ONLY have two allowed tools: 'web_search' and 'visit_webpage'.\n"
        f"2. Never call 'wikipedia_search(...)'. It does NOT exist.\n"
        f"3. To find Wikipedia data, you MUST use the general search tool: web_search(query='your query here')."
    )
    try:
        result = agent_core.run(managed_prompt)
        return str(result).strip() if result else "unknown"
    except Exception as e:
        print(f"❌ Inference Error: {e}")
        return "unknown"

demo = gr.Interface(fn=predict, inputs=gr.Textbox(label="Question"), outputs=gr.Textbox(label="Answer"))

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)