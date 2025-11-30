import json
from typing import Any, Dict, List
from state import AgentState, TravelPreference, Place
from openai import OpenAI 

client = OpenAI()


def _place_dict(p: Any) -> Dict:
    if hasattr(p, "model_dump"):
        return p.model_dump()
    if isinstance(p, dict):
        return p
    return dict(p)


SYSTEM_PROMPT_AGENT4 = """
당신은 여행 일정 플래너의 "메인 장소 추천 에이전트(Agent4)"입니다.

## 역할
- 사용자의 여행 취향(prefs)과 테마별 방문 계획(tag_plan), 후보 장소(place_pool)을 보고
  "메인으로 잡을 만한 장소" 후보를 최대 3개 추천합니다.
- 반드시 JSON 형식으로만 출력하세요.

## 입력 정보 설명

1. prefs
   - 사용자의 기본 여행 취향
   - themes, must_avoid 등 포함될 수 있음

2. tag_plan
   - {"tag": "...", "weight": 0.4} 같은 구조
   - weight 높은 상위 2개 테마만 사용

3. place_pool
   - 실제 방문 가능한 장소 리스트
   - theme, name, address, rating 등 포함 가능

## 추천 규칙

1. 테마 선택
   - tag_plan의 weight 기준 상위 2개
   - 유효하지 않으면 prefs.themes 상위 2개
   - 그것도 없으면 ["맛집", "카페"] 기본값

2. place_pool 필터
   - 위 테마에 해당하는 장소만 선택
   - name + address 기준 중복 제거
   - must_avoid 있으면 가능한 한 제외

3. 우선순위
   - rating 높은 순
   - review_count 높은 순
   - 다양성 약간 고려

4. 개수 제한
   - 최대 3개

## 출력 형식

아래 예시처럼, JSON만 출력하세요.

예시:
{
  "main_place_candidates": [
    {
      "name": "...",
      "theme": "...",
      "road_address": "...",
      "address": "...",
      "reason": "...",
      "extra": {
        "rating": 4.3,
        "review_count": 122,
        "notes": "..."
      }
    }
  ]
}

설명 문장 없이, 위 형식과 같은 JSON만 출력하세요.
main_place_candidate가 이미 있는데 agent4에 들어온 경우에는 마음에 안들어서 돌아온거니까 그걸 제외하고 
다른 main_place_candidate을 골라주세요.
"""

def agent4_suggest_node(state: AgentState) -> AgentState:
    prefs = state["prefs"]
    tag_plan = state.get("tag_plan") or []
    place_pool = state["place_pool"]

    # 직렬화
    if hasattr(prefs, "model_dump"):
        prefs_payload = prefs.model_dump()
    else:
        prefs_payload = prefs

    payload = {
        "prefs": prefs_payload,
        "tag_plan": tag_plan,
        "place_pool": [_place_dict(p) for p in place_pool],
    }

    user_content = json.dumps(payload, ensure_ascii=False)

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_AGENT4},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    raw_output = completion.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_output)
        main_place_candidates = parsed.get("main_place_candidates", [])
        if not isinstance(main_place_candidates, list):
            main_place_candidates = []
    except json.JSONDecodeError:
        main_place_candidates = []

    state["main_place_candidates"] = main_place_candidates
    return state

def agent4_wait_node(state: AgentState) -> AgentState:
    """
    역할:
    - agent4_suggest_node가 main_place_candidates를 채운 뒤
      그래프를 여기서 멈추게 하기 위한 '대기 노드'
    - 여기서는 state를 건드리지 않고 그대로 반환만 함.
    """
    return state