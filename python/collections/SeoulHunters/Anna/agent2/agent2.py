import json
import re
from langgraph.graph import StateGraph, END
from agent1.schema import TravelPreference
from agent1.agent1 import client

SYSTEM_PROMPT = """
너는 여행 코스 설계 어시스턴트야.

입력으로 사용자의 여행 성향(preferences)을 받으면,
'어떤 태그 그룹의 장소를 몇 번 방문하면 좋은지'를 계획해주는 역할을 한다.

반드시 **순수 JSON만** 출력해라.
- 마크다운 코드블록(```json 같은 것) 쓰지 말 것.
- 설명 문장, 텍스트, 주석을 한 글자도 붙이지 말 것.
- 오직 JSON 객체 하나만 출력.

출력 형식 예시:
{
  "tag_plan": [
    { "tag": "음식점", "weight": 0.4, "visits": 4 },
    { "tag": "카페",   "weight": 0.2, "visits": 2 }
  ]
}

규칙:
- weight 합은 1.0에 가깝게.
- visits는 일정일수 + 강도(intensity) 기반으로 합리적인 정수.
- 테마(미식, 야경 등)에 따라 가중치 반영.
"""

def _extract_json(text: str) -> str:
    """응답에서 첫 { .. 마지막 } 사이만 잘라 JSON 문자열로 반환."""
    text = text.strip()
    # ```json ... ``` 같은 코드블록 제거 시도
    if text.startswith("```"):
        # 첫 줄(```json 등) 제거, 마지막 ``` 제거
        lines = text.splitlines()
        # 첫 줄, 마지막 줄 빼고 다시 합치기
        inner = "\n".join(lines[1:-1]).strip()
        text = inner

    # 그래도 앞뒤에 뭔가 있을 수 있으니 { ... }만 추출
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 본문을 찾지 못함. 원본:\n{text}")
    return m.group(0)

def make_tag_plan(state: dict) -> dict:
    prefs: TravelPreference = state["prefs"]

    pref_text = (
        f"여행 위치: {prefs.target_area}\n"
        f"체류 기간: {prefs.duration}일\n"
        f"테마: {prefs.themes}\n"
        f"강도(0=힐링,100=빡셈): {prefs.intensity}\n"
        f"동행자: {prefs.companions}\n"
        f"이동수단: {prefs.transport}\n"
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pref_text},
        ]
    )
    raw = resp.output_text

    try:
        json_str = _extract_json(raw)
        data = json.loads(json_str)
    except Exception as e:
        print("=== TAG_PLAN RAW RESPONSE ===")
        print(raw)
        raise e  # 어디서 깨지는지 보려고 그대로 다시 던지기

    return {
        "prefs": prefs,
        "tag_plan": data["tag_plan"],
    }

def build_agent2():
    graph = StateGraph(dict)

    graph.add_node("make_tag_plan", make_tag_plan)

    graph.set_entry_point("make_tag_plan")
    graph.set_finish_point("make_tag_plan")

    return graph.compile()
