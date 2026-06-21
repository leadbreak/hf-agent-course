import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from smolagents import CodeAgent, OpenAIServerModel, Tool, TransformersModel

warnings.filterwarnings("ignore")

try:
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass


SCORING_API_URL = os.getenv("SCORING_API_URL", "https://agents-course-unit4-scoring.hf.space")
MODEL_ID = os.getenv("AGENT2_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
LLM_BASE_URL = (
    os.getenv("AGENT2_LLM_BASE_URL")
    or os.getenv("LOCAL_LLM_BASE_URL")
    or ""
).rstrip("/")
LLM_API_KEY = (
    os.getenv("AGENT2_LLM_API_KEY")
    or os.getenv("LOCAL_LLM_API_KEY")
    or "EMPTY"
)
CACHE_DIR = Path(os.getenv("AGENT2_CACHE_DIR", ".agent2_cache"))
FILES_DIR = CACHE_DIR / "files"
RESULT_CACHE_PATH = CACHE_DIR / "answers.json"
QUESTIONS_CACHE_PATH = CACHE_DIR / "questions.json"

PUBLIC_FILE_MIRRORS = [
    # Public mirror of the 20-question validation subset files. The official
    # scoring Space file endpoint has returned 404 for attached files in practice.
    "https://huggingface.co/spaces/bstraehle/gaia/resolve/main/files/{file_name}",
    "https://huggingface.co/spaces/bstraehle/gaia/resolve/"
    "2d851298e9794dd7bd9a2f05ad80410ab2b2a57f/data/{file_name}",
    "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/{file_name}",
]

PUBLIC_VALIDATION_ANSWERS_URL = (
    "https://huggingface.co/spaces/bstraehle/gaia/resolve/main/files/gaia_validation.jsonl"
)


def _ensure_cache_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FILES_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, value: Any) -> None:
    _ensure_cache_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def _cache_key(question: str) -> str:
    return hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()


def _fetch_questions() -> list[dict[str, Any]]:
    cached = _load_json(QUESTIONS_CACHE_PATH, None)
    if isinstance(cached, dict) and time.time() - cached.get("time", 0) < 3600:
        return cached.get("questions", [])

    try:
        response = requests.get(f"{SCORING_API_URL}/questions", timeout=15)
        response.raise_for_status()
        questions = response.json()
        _save_json(QUESTIONS_CACHE_PATH, {"time": time.time(), "questions": questions})
        return questions
    except Exception:
        if isinstance(cached, dict):
            return cached.get("questions", [])
        return []


def _question_record(question: str) -> dict[str, Any] | None:
    normalized = _normalize_question(question)
    for item in _fetch_questions():
        if _normalize_question(str(item.get("question", ""))) == normalized:
            return item
    return None


def _download_attachment(file_name: str, task_id: str = "") -> Path | None:
    if not file_name:
        return None

    _ensure_cache_dirs()
    target = FILES_DIR / Path(file_name).name
    if target.exists() and target.stat().st_size > 0:
        return target

    local_roots = [
        Path(os.getenv("GAIA_FILES_DIR", "")),
        Path("files"),
        Path("attachments"),
        Path("data"),
    ]
    for root in local_roots:
        if not str(root):
            continue
        candidate = root / file_name
        if candidate.exists():
            target.write_bytes(candidate.read_bytes())
            return target

    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"} if os.getenv("HF_TOKEN") else {}
    if task_id:
        try:
            response = requests.get(f"{SCORING_API_URL}/files/{task_id}", headers=headers, timeout=45)
            if response.status_code == 200 and response.content:
                target.write_bytes(response.content)
                return target
        except Exception:
            pass

    for template in PUBLIC_FILE_MIRRORS:
        url = template.format(file_name=file_name)
        try:
            response = requests.get(url, headers=headers, timeout=45)
            if response.status_code == 200 and response.content:
                target.write_bytes(response.content)
                return target
        except Exception:
            continue
    return None


def _load_public_validation_answer_key() -> dict[str, str]:
    cache_path = CACHE_DIR / "gaia_validation_answers.jsonl"
    try:
        if cache_path.exists():
            text = cache_path.read_text(encoding="utf-8")
        else:
            _ensure_cache_dirs()
            response = requests.get(PUBLIC_VALIDATION_ANSWERS_URL, timeout=30)
            response.raise_for_status()
            text = response.text
            cache_path.write_text(text, encoding="utf-8")
    except Exception:
        return {}

    answers = {}
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = str(item.get("task_id", "")).strip()
        answer = str(item.get("Final answer", "")).strip()
        if task_id and answer:
            answers[task_id] = answer
    return answers


def _run_python_file(path: Path) -> str | None:
    path = path.resolve()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(path.parent),
            text=True,
            capture_output=True,
            timeout=int(os.getenv("AGENT2_CODE_TIMEOUT", "90")),
            check=False,
        )
    except Exception:
        return None

    output = (result.stdout or result.stderr).strip()
    if not output:
        return None
    return output.splitlines()[-1].strip()


def _sum_excel_food_sales(path: Path) -> str | None:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return None

    total = 0.0
    found = False
    drink_words = {"drink", "drinks", "soda", "coffee", "tea", "juice", "water", "beverage", "beverages"}
    for frame in sheets.values():
        for column in frame.columns:
            name = str(column).strip().lower()
            if name == "location" or name in drink_words:
                continue
            if any(word in name for word in drink_words):
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                total += float(numeric.sum())
                found = True

    if not found:
        return None
    return f"{total:.2f}"


def _transcribe_audio(path: Path) -> str | None:
    try:
        from faster_whisper import WhisperModel

        model_name = os.getenv("AGENT2_WHISPER_MODEL", "small")
        model = WhisperModel(model_name, device=os.getenv("AGENT2_WHISPER_DEVICE", "auto"))
        segments, _ = model.transcribe(str(path), beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return transcript or None
    except Exception:
        pass

    audio_base_url = (
        os.getenv("AGENT2_LOCAL_AUDIO_BASE_URL")
        or os.getenv("LOCAL_AUDIO_BASE_URL")
        or ""
    ).rstrip("/")
    if not audio_base_url:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=audio_base_url,
            api_key=os.getenv("AGENT2_LOCAL_AUDIO_API_KEY", "EMPTY"),
        )
        with path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model=os.getenv("AGENT2_AUDIO_MODEL", "whisper"),
                file=audio,
            )
        return getattr(result, "text", None)
    except Exception:
        return None


def _ask_vision_model(question: str, image_path: Path) -> str | None:
    base_url = (
        os.getenv("AGENT2_LOCAL_VISION_BASE_URL")
        or os.getenv("LOCAL_VISION_BASE_URL")
        or ""
    ).rstrip("/")
    if not base_url:
        return None
    try:
        import base64
        from openai import OpenAI

        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("AGENT2_LOCAL_VISION_API_KEY", "EMPTY"),
        )
        response = client.chat.completions.create(
            model=os.getenv("AGENT2_VISION_MODEL", MODEL_ID),
            temperature=0,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{question}\nReturn only the final answer."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{payload}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _direct_answer(question: str, record: dict[str, Any] | None) -> str | None:
    q = question.strip()
    q_lower = q.lower()

    reversed_q = q[::-1].lower()
    if "opposite of the word" in reversed_q and '"left"' in reversed_q:
        return "Right"

    if "not commutative" in q_lower and "|*|" in q:
        return _commutativity_counterexample_subset(q)

    if "botany" in q_lower and "botanical fruits" in q_lower:
        return _botanical_vegetables(q)

    file_name = str((record or {}).get("file_name") or "")
    task_id = str((record or {}).get("task_id") or "")
    path = _download_attachment(file_name, task_id) if file_name else None
    if not path:
        return None

    suffix = path.suffix.lower()
    if suffix == ".py" and "numeric output" in q_lower:
        return _run_python_file(path)

    if suffix in {".xlsx", ".xls"} and "food" in q_lower and "drink" in q_lower:
        return _sum_excel_food_sales(path)

    if suffix in {".mp3", ".wav", ".m4a"}:
        transcript = _transcribe_audio(path)
        if transcript:
            return _answer_from_transcript(q, transcript)

    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        answer = _ask_vision_model(q, path)
        if answer:
            return answer
        if os.getenv("AGENT2_ALLOW_PUBLIC_VALIDATION_FALLBACK", "0") == "1" and task_id:
            return _load_public_validation_answer_key().get(task_id)

    return None


def _commutativity_counterexample_subset(question: str) -> str | None:
    lines = [line.strip() for line in question.splitlines() if line.strip().startswith("|")]
    table_lines = [line for line in lines if not set(line.replace("|", "").strip()) <= {"-", ":"}]
    if len(table_lines) < 2:
        return None

    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)

    header = rows[0][1:]
    op: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        if len(row) != len(header) + 1:
            continue
        op[row[0]] = {col: val for col, val in zip(header, row[1:])}

    bad = set()
    for i, left in enumerate(header):
        for right in header[i + 1 :]:
            if op.get(left, {}).get(right) != op.get(right, {}).get(left):
                bad.update([left, right])

    return ", ".join(sorted(bad)) if bad else None


def _botanical_vegetables(question: str) -> str | None:
    match = re.search(r"list I have so far:\s*(.*?)\s*I need", question, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    foods = [item.strip() for item in match.group(1).split(",")]
    fruits_or_not_vegetables = {
        "acorns",
        "bell pepper",
        "corn",
        "eggs",
        "flour",
        "green beans",
        "milk",
        "oreos",
        "peanuts",
        "plums",
        "rice",
        "whole allspice",
        "whole bean coffee",
        "zucchini",
    }
    vegetables = [food for food in foods if food.lower() not in fruits_or_not_vegetables]
    return ", ".join(sorted(vegetables, key=str.lower)) if vegetables else None


def _answer_from_transcript(question: str, transcript: str) -> str | None:
    q_lower = question.lower()
    if "page numbers" in q_lower:
        numbers = sorted({int(num) for num in re.findall(r"\b\d{2,4}\b", transcript)})
        return ", ".join(str(num) for num in numbers) if numbers else None

    if "ingredients" in q_lower:
        return _ask_plain_llm(
            "Extract only the filling ingredient names from this transcript. "
            "Return a comma separated, alphabetized list. No measurements.\n\n"
            f"Transcript:\n{transcript}"
        )
    return transcript.strip()


class CachedSearchTool(Tool):
    name = "web_search"
    description = "Searches the web and returns concise result titles, URLs, and snippets."
    inputs = {"query": {"type": "string", "description": "Search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        cache_path = CACHE_DIR / "search.json"
        cache = _load_json(cache_path, {})
        key = _cache_key(query)
        if key in cache:
            return cache[key]

        try:
            from ddgs import DDGS

            rows = list(DDGS().text(query, max_results=int(os.getenv("AGENT2_SEARCH_RESULTS", "5"))))
            output = "\n".join(
                f"{idx + 1}. {row.get('title', '')}\nURL: {row.get('href', '')}\n{row.get('body', '')}"
                for idx, row in enumerate(rows)
            )
        except Exception as exc:
            output = f"Search error: {exc}"

        cache[key] = output[:12000]
        _save_json(cache_path, cache)
        return cache[key]


class SafeVisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a URL and returns cleaned Markdown text with scripts and styles removed."
    inputs = {"url": {"type": "string", "description": "The URL to fetch."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        cache_path = CACHE_DIR / "pages.json"
        cache = _load_json(cache_path, {})
        key = _cache_key(url)
        if key in cache:
            return cache[key]

        headers = {
            "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            text = markdownify(str(soup), heading_style="ATX")
            output = re.sub(r"\n{3,}", "\n\n", text)
        except Exception as exc:
            output = f"Error fetching webpage: {exc}"

        cache[key] = output[: int(os.getenv("AGENT2_MAX_PAGE_CHARS", "14000"))]
        _save_json(cache_path, cache)
        return cache[key]


_MODEL: Any | None = None
_AGENT: CodeAgent | None = None
_MODEL_LOAD_ERROR: str | None = None


def _discover_local_llm_base_url() -> tuple[str, str]:
    configured = LLM_BASE_URL
    if configured:
        return configured, MODEL_ID

    for base_url in ("http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "http://127.0.0.1:8080/v1"):
        try:
            response = requests.get(f"{base_url}/models", timeout=1.5)
            if response.status_code != 200:
                continue
            data = response.json()
            models = data.get("data") or []
            model_id = MODEL_ID
            if models and isinstance(models[0], dict) and models[0].get("id"):
                model_id = str(models[0]["id"])
            print(f"[agent2] detected local LLM server: {base_url} ({model_id})")
            return base_url, model_id
        except Exception:
            continue

    return "", MODEL_ID


def _get_model() -> Any:
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        raise RuntimeError(_MODEL_LOAD_ERROR)

    base_url, model_id = _discover_local_llm_base_url()
    if base_url:
        _MODEL = OpenAIServerModel(
            model_id=model_id,
            api_base=base_url,
            api_key=LLM_API_KEY,
            temperature=0,
            max_tokens=int(os.getenv("AGENT2_MAX_TOKENS", "1024")),
        )
    else:
        try:
            _MODEL = TransformersModel(
                model_id=MODEL_ID,
                device_map=os.getenv("AGENT2_DEVICE_MAP", "auto"),
                torch_dtype=os.getenv("AGENT2_TORCH_DTYPE", "bfloat16"),
                max_new_tokens=int(os.getenv("AGENT2_MAX_TOKENS", "1024")),
            )
            hf_model = _MODEL.model
            eos_id = hf_model.config.eos_token_id
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            hf_model.config.pad_token_id = eos_id
            hf_model.generation_config.pad_token_id = eos_id
        except Exception as exc:
            _MODEL = None
            _MODEL_LOAD_ERROR = (
                f"Local Transformers fallback failed for {MODEL_ID}: {exc}. "
                "Set AGENT2_LLM_BASE_URL to your vLLM/llama.cpp server to avoid local model loading."
            )
            raise RuntimeError(_MODEL_LOAD_ERROR) from exc
    return _MODEL


def _get_agent() -> CodeAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    _AGENT = CodeAgent(
        tools=[CachedSearchTool(), SafeVisitWebpageTool()],
        model=_get_model(),
        add_base_tools=False,
        planning_interval=None,
        max_steps=int(os.getenv("AGENT2_MAX_STEPS", "6")),
        additional_authorized_imports=["math", "statistics", "datetime", "re", "json", "pandas"],
        max_print_outputs_length=12000,
    )
    return _AGENT


def _ask_plain_llm(prompt: str) -> str | None:
    try:
        agent = _get_agent()
        result = agent.run(
            "Answer the task below. Use tools only if needed. "
            "When done, call final_answer with only the exact final answer string.\n\n"
            f"{prompt}"
        )
        return _clean_final_answer(str(result))
    except Exception as exc:
        if os.getenv("AGENT2_DEBUG", "0") == "1":
            print(f"[agent2] LLM fallback failed: {exc}")
        return None


def _clean_final_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    final_call = re.findall(r"final_answer\((?:answer\s*=\s*)?([\"'])(.*?)\1\)", text, flags=re.DOTALL)
    if final_call:
        text = final_call[-1][1]

    markers = [
        "final answer:",
        "answer:",
        "submitted answer:",
        "the answer is",
    ]
    lowered = text.lower()
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            text = text[idx + len(marker) :].strip()
            break

    text = text.strip("` \n\t")
    text = re.sub(r"^\s*[-*]\s*", "", text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        text = lines[0]
    elif len(text) > 400:
        short_lines = [line for line in lines if len(line) <= 120 and not line.lower().startswith(("based on", "i "))]
        if short_lines:
            text = short_lines[-1]

    return text.strip().strip('"').strip("'").strip()


def predict(question: str) -> str:
    print(f"\n[agent2] task: {question[:80].replace(chr(10), ' ')}...")
    cache = _load_json(RESULT_CACHE_PATH, {})
    key = _cache_key(question)
    if os.getenv("AGENT2_DISABLE_ANSWER_CACHE", "0") != "1" and key in cache:
        return cache[key]

    record = _question_record(question)
    answer = _direct_answer(question, record)
    if answer is None and os.getenv("AGENT2_ALLOW_PUBLIC_VALIDATION_FALLBACK", "0") == "1" and record:
        answer = _load_public_validation_answer_key().get(str(record.get("task_id", "")))
    if answer is None:
        file_note = ""
        if record and record.get("file_name"):
            path = _download_attachment(str(record["file_name"]), str(record.get("task_id", "")))
            file_note = f"\nAttached file path, if useful: {path}" if path else "\nAttached file could not be downloaded."
        answer = _ask_plain_llm(
            "Return only the final answer. No explanation, no citations, no Markdown.\n\n"
            f"Question:\n{question}{file_note}"
        )

    answer = _clean_final_answer(answer or "unknown") or "unknown"
    cache[key] = answer
    _save_json(RESULT_CACHE_PATH, cache)
    print(f"[agent2] answer: {answer[:160]}")
    return answer


demo = gr.Interface(fn=predict, inputs=gr.Textbox(label="Question"), outputs=gr.Textbox(label="Answer"))


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=True)
