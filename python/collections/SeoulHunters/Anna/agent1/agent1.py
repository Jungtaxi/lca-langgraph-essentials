import json
from openai import OpenAI
from langgraph.graph import StateGraph, END
from .schema import TravelPreference
import re
import os

# print("DEBUG:: OPENAI KEY =", os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
너는 여행 성향을 구조화하는 어시스턴트야.

반드시 **순수 JSON만** 출력해라.
- 설명 문장 X
- 마크다운 코드블록 X
- 오직 JSON 객체 하나만 출력.

필드는 다음과 같다:
- target_area: 여행 지역 리스트
- duration: 체류 기간 (2박 3일이면 3일)
- themes: ["미식","쇼핑","문화","야경", ...] 형태
- intensity: 0~100
- companions: ["혼자","친구","연인","가족"] 리스트
- transport: ["도보","대중교통","자차","택시"] 리스트
"""

def _extract_json(text: str) -> str:
    """응답에서 JSON만 추출"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 못 찾음:\n{text}")
    return m.group(0)

def classify_user_input(state: dict) -> dict:
    user_text = state["user_input"]

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
    )

    raw = resp.output_text
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    prefs = TravelPreference(**data)

    return {
        "user_input": user_text,
        "prefs": prefs,
    }

def build_agent1():
    graph = StateGraph(dict)
    graph.add_node("classify_user_input", classify_user_input)
    graph.set_entry_point("classify_user_input")
    graph.add_edge("classify_user_input", END)
    return graph.compile()

