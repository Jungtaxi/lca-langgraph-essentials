import json
from typing import Any, Dict, List
from state import AgentState
from openai import OpenAI

client = OpenAI()  # 이미 어딘가에 있다면 재사용

def path_finder_node(state: AgentState) -> AgentState:
    prefs = state["preferences"]
    tag_plan = state["strategy"]
    place_pool = state["candidates"]
    selected_main_places = state["selected_main_places"]

    duration = prefs.duration or 1
    intensity = prefs.intensity or 50

    ROUTE_SYSTEM_PROMPT ='''
    당신은 사용자의 취향과 장소 목록을 바탕으로 '여행 동선'을 짜는 전문가입니다.

    [입력 설명]
    - prefs: 사용자의 취향 설정. 
    - intensity: 0~100 (낮을수록 느긋, 높을수록 빡센 일정)
    - duration: 여행 일수 (정수)
    - tag_plan: 각 테마별(weight) 비율 정보. 
    - 예: [{"tag": "음식점", "weight": 0.5}, {"tag":"카페","weight":0.3}, {"tag":"쇼핑","weight":0.2}]
    - place_pool: 추천 가능한 장소 목록.
    - 각 원소는 {"id": int, "name": str, "theme": str, "address": str, "road_address": str, ...}
    - theme 예: "음식점", "카페", "쇼핑", "술집" 등
    - selected_main_places: 사용자가 반드시 가고 싶다고 고른 장소 목록.
    - place_pool와 동일한 구조. id를 통해 식별 가능.

    [규칙]
    1. 하루 방문 스팟 수
    - intensity <= 30: 하루 4곳
    - 30 < intensity <= 60: 하루 5곳
    - intensity > 60: 하루 6곳

    2. 테마 비율
    - "음식점", "카페", "쇼핑" 세 가지를 MAIN TAG로 본다.
    - tag_plan의 weight를 사용해 하루 스팟 중 이 세 테마 비율을 최대한 맞춘다.
    - tag_plan이 없거나 weight 합이 0이면 기본 비율은 음식점:0.4, 카페:0.3, 쇼핑:0.3 으로 가정한다.
    - "술집", "맛집" theme는 "음식점"과 동일한 계열로 취급해도 좋다.

    3. 사용자 입력과 main_place_candidates 기반 선택한 selceted_main_places 추출
    
    3. selected_main_places 사용
    - 가능한 한 하루에 하나씩 분배하되, 장소가 모자라면 일부 날은 없어도 된다.
    - 이미 main_place가 포함된 날에도, 나머지 스팟은 위의 테마 비율을 유지하려고 노력한다.
    - 같은 장소를 여러 날에 넣지 않는다.

    4. 동선 구성
    - 같은 테마가 연속해서 너무 많이 나오지 않도록 섞는다. (예: 음식점-음식점-음식점-카페 보다는 음식점-카페-음식점 형태를 선호)
    - 주소(대략적인 동/구 정도)를 참고하여, 한 날 안에서는 가능한 한 비슷한 지역끼리 묶는다.
    - 단, 정확한 거리 계산은 불가능하므로, 텍스트 주소 수준에서 상식적으로 판단한다.

    5. 출력 형식
    - JSON 형식의 객체 하나만 출력한다. 다른 설명 문장은 쓰지 않는다.
    - 최상위 키는 "routes" 이고, 각 원소는 아래 형태이다.

    {
    "routes": [
        {
        "day": 1,
        "schedule": [
            {
            "order": 1,
            "place_id": 10
            },
            {
            "order": 2,
            "place_id": 5
            }
        ]
        },
        ...
    ]
    }

    - order는 1부터 시작하는 방문 순서이다.
    - place_id는 place_pool 안의 "id" 값을 사용한다.
    - duration 일수만큼 day=1..duration 을 모두 포함하려고 시도한다.
    '''

    llm_input = {
        "prefs": {
            "intensity": intensity,
            "duration": duration,
        },
        "tag_plan": tag_plan,
        "place_pool": place_pool,
        "selected_main_places": selected_main_places,
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # 쓰는 모델로 변경
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(llm_input, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # 혹시 망하면 최소한의 fallback (그냥 첫 몇 개씩 자르기 등)
        # 여기서는 state 그대로 리턴하거나, 아주 단순 로직을 넣을 수 있음
        return state

    routes_out = []
    for day_info in data.get("routes", []):
        day_num = day_info.get("day")
        schedule_out = []
        for s in day_info.get("schedule", []):
            place_id = s.get("place_id")
            order = s.get("order", 0)

            # id로 place 찾기
            pl = next((p for p in place_pool if p["id"] == place_id), None)
            if pl is None:
                continue

            schedule_out.append({
                "order": order,
                "place": pl,
            })

        if schedule_out:
            # order 기준으로 정렬 보정
            schedule_out.sort(key=lambda x: x["order"])
            routes_out.append({
                "day": day_num,
                "schedule": schedule_out,
            })

    state["routes"] = routes_out
    return state