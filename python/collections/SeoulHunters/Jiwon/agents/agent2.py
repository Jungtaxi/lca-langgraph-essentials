import json
from state import AgentState
from .agent1 import client, _extract_json 

SYSTEM2 = """
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

def agent2_node(state: AgentState) -> AgentState:
    prefs = state["prefs"]

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
            {"role": "system", "content": SYSTEM2},
            {"role": "user", "content": pref_text}
        ]
    )
    raw = resp.output_text

    try:
        json_str = _extract_json(raw)
        data = json.loads(json_str)
    except Exception as e:
        print("=== AGENT2 PARSE ERROR RAW ===")
        print(raw)
        raise e

    tag_plan = data.get("tag_plan")

    if tag_plan is None and isinstance(data, list):
        tag_plan = data

    if tag_plan is None:
        tag_plan = data.get("plan", [])

    if tag_plan is None:
        tag_plan = []

    state["tag_plan"] = tag_plan
    return state
