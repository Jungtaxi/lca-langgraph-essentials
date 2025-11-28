"""
agent4.py

역할:
- Agent1의 prefs(TravelPreference) + Agent3의 place_pool을 받아서
  '일차별 방문 장소 순서(routes)'를 만들어주는 Agent4.

입력 예시 (main.py):

    app4 = build_agent4()
    result4 = app4({
        "prefs": result1["prefs"],          # TravelPreference
        "place_pool": result3["place_pool"] # List[Place] (Pydantic or dict)
    })
    routes = result4["routes"]

출력 예시:

    {
      "routes": [
        {
          "day": 1,
          "schedule": [
            {"time": "morning", "place": {...} },
            {"time": "lunch",   "place": {...} },
            ...
          ]
        },
        ...
      ]
    }
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# 1. place 를 dict 형태로 통일 (Pydantic / dict / 객체 모두 대응)
# ---------------------------------------------------------
def _place_to_dict(p: Any) -> Dict[str, Any]:
    """Place 객체를 dict 로 통일해서 다루기 위한 헬퍼."""
    if p is None:
        return {}

    # Pydantic 모델인 경우
    if hasattr(p, "model_dump"):
        return p.model_dump()

    # 이미 dict인 경우
    if isinstance(p, dict):
        return dict(p)

    # 그 외에는 __dict__ 사용 (일반 객체)
    if hasattr(p, "__dict__"):
        return dict(p.__dict__)

    # 어떻게 해도 안되면 그대로 감싸기
    return {"value": p}


# ---------------------------------------------------------
# 2. place_pool 을 테마별 버킷으로 분류
# ---------------------------------------------------------
def _bucket_places_by_theme(place_pool: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    place_pool을 theme 기준으로 분류해서
    {"맛집": [...], "카페": [...], "관광": [...], ...} 형태로 반환.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for p in place_pool:
        data = _place_to_dict(p)
        theme = data.get("theme") or "기타"

        if theme not in buckets:
            buckets[theme] = []
        buckets[theme].append(data)

    return buckets


# ---------------------------------------------------------
# 3. intensity(일정 강도) → 하루에 사용할 time slot 결정
# ---------------------------------------------------------
BASE_SLOTS = ["morning", "lunch", "afternoon", "snack", "dinner", "night"]


def _slots_from_intensity(intensity: Optional[int]) -> List[str]:
    """
    0~100 사이의 intensity(일정 강도)에 따라 하루에 몇 개의 슬롯을 쓸지 결정.

    - 0~30  : 여유 일정 (점심, 오후, 저녁) → 3개
    - 31~60 : 보통 일정 (아침, 점심, 오후, 저녁) → 4개
    - 61~100: 빡센 일정 (아침, 점심, 오후, 간식, 저녁, 밤) → 6개
    """
    if intensity is None:
        intensity = 50

    # 클램핑 (0~100 사이로 맞추기)
    try:
        intensity = int(intensity)
    except Exception:
        intensity = 50

    if intensity < 0:
        intensity = 0
    if intensity > 100:
        intensity = 100

    if intensity <= 30:
        # 아주 여유로운 일정
        return ["lunch", "afternoon", "dinner"]
    elif intensity <= 60:
        # 보통
        return ["morning", "lunch", "afternoon", "dinner"]
    else:
        # 빡센 일정
        return BASE_SLOTS[:]  # 전부 사용


# ---------------------------------------------------------
# 4. time slot → 선호 테마 매핑
# ---------------------------------------------------------
SLOT_THEME_PRIORITIES: Dict[str, List[str]] = {
    "morning": ["관광", "카페"],
    "lunch": ["맛집"],
    "afternoon": ["관광", "카페"],
    "snack": ["카페", "디저트"],
    "dinner": ["맛집"],
    "night": ["야경", "전망", "한강", "산책", "관광"],
}


def _pick_place_for_slot(
    slot: str,
    theme_buckets: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """
    특정 time slot 에 대해 우선순위 테마를 기준으로 place 하나 선택.
    없으면 '남은 아무 테마' 중 하나를 사용.
    """
    priorities = SLOT_THEME_PRIORITIES.get(slot, [])

    # 1) 우선순위 테마에서 먼저 찾기
    for theme in priorities:
        bucket = theme_buckets.get(theme, [])
        if bucket:
            return bucket.pop(0)  # FIFO 방식으로 하나 꺼내기

    # 2) 아무 테마나 남아있는 것 중에서 사용 (fallback)
    for theme, bucket in theme_buckets.items():
        if bucket:
            return bucket.pop(0)

    # 3) 정말 아무것도 없으면 None
    return None


# ---------------------------------------------------------
# 5. prefs + place_pool → routes 생성 (핵심 로직)
# ---------------------------------------------------------
# def _build_routes(
#     prefs: Any,
#     place_pool: List[Any],
# ) -> List[Dict[str, Any]]:
#     """
#     Agent1의 prefs + Agent3의 place_pool 을 받아
#     일차별 route 리스트를 생성한다.
#     """

#     # prefs: TravelPreference or dict
#     if hasattr(prefs, "model_dump"):
#         prefs_data: Dict[str, Any] = prefs.model_dump()
#     else:
#         prefs_data = dict(prefs)

#     duration: int = int(prefs_data.get("duration") or 1)
#     intensity: Optional[int] = prefs_data.get("intensity")

#     # 1) 하루에 사용할 time slot 결정
#     slots = _slots_from_intensity(intensity)

#     # 2) place_pool을 theme 기준으로 버킷화
#     theme_buckets = _bucket_places_by_theme(place_pool)

#     # 3) 일차별 route 생성
#     routes: List[Dict[str, Any]] = []

#     for day in range(1, duration + 1):
#         schedule_entries: List[Dict[str, Any]] = []

#         for slot in slots:
#             place = _pick_place_for_slot(slot, theme_buckets)
#             schedule_entries.append(
#                 {
#                     "time": slot,
#                     "place": place,  # dict 또는 None
#                 }
#             )

#         routes.append(
#             {
#                 "day": day,
#                 "schedule": schedule_entries,
#             }
#         )

#     return routes

def _build_routes(
    prefs: Any,
    place_pool: List[Any],
) -> List[Dict[str, Any]]:
    """
    Agent1의 prefs + Agent3의 place_pool 을 받아
    일차별 route 리스트를 생성한다.
    - 하루에 들어가는 장소 수를 가능한 한 '균등 분배' 하도록 조정.
    """

    # prefs: TravelPreference or dict
    if hasattr(prefs, "model_dump"):
        prefs_data: Dict[str, Any] = prefs.model_dump()
    else:
        prefs_data = dict(prefs)

    duration: int = int(prefs_data.get("duration") or 1)
    intensity: Optional[int] = prefs_data.get("intensity")

    # 1) 하루에 사용할 time slot 결정
    slots = _slots_from_intensity(intensity)

    # 2) place_pool을 theme 기준으로 버킷화
    theme_buckets = _bucket_places_by_theme(place_pool)

    # 🔥 3) 전체 place 개수와 하루 최소 개수 계산 (균등 분배용)
    total_places = sum(len(bucket) for bucket in theme_buckets.values())
    if total_places <= 0:
        # 장소가 아예 없으면 전부 비워둔 routes 반환
        return [
            {
                "day": day,
                "schedule": [
                    {"time": slot, "place": None} for slot in slots
                ],
            }
            for day in range(1, duration + 1)
        ]

    # 하루에 최소 몇 개씩은 넣자
    # 예: total_places=12, duration=3 → min_per_day=4
    min_per_day = max(1, total_places // duration)

    routes: List[Dict[str, Any]] = []

    for day in range(1, duration + 1):
        schedule_entries: List[Dict[str, Any]] = []

        # 이 날에 실제로 채울 슬롯 개수
        # (slots 수보다 min_per_day가 클 수 있으니)
        slots_to_fill = min(len(slots), min_per_day)

        for idx, slot in enumerate(slots):
            if idx < slots_to_fill:
                # 이 날에서 "우선 채워야 하는" 슬롯들만 place 할당
                place = _pick_place_for_slot(slot, theme_buckets)
            else:
                # 나머지 슬롯은 비워두기
                place = None

            schedule_entries.append(
                {
                    "time": slot,
                    "place": place,
                }
            )

        routes.append(
            {
                "day": day,
                "schedule": schedule_entries,
            }
        )

    return routes




# ---------------------------------------------------------
# 6. Agent4 빌더 (함수 기반, Agent1/2/3 스타일)
# ---------------------------------------------------------
def build_agent4():
    """
    main.py 예시:

        from agent4.agent4 import build_agent4

        app4 = build_agent4()
        result4 = app4({
            "prefs": result1["prefs"],
            "place_pool": result3["place_pool"],
        })
        routes = result4["routes"]
    """

    def agent4_app(inputs: Dict[str, Any]) -> Dict[str, Any]:
        prefs = inputs.get("prefs")
        place_pool = inputs.get("place_pool")

        if prefs is None:
            raise ValueError("Agent4: 'prefs'가 필요합니다. (TravelPreference)")
        if place_pool is None:
            raise ValueError("Agent4: 'place_pool'이 필요합니다. (Agent3 결과)")

        routes = _build_routes(
            prefs=prefs,
            place_pool=place_pool,
        )

        return {"routes": routes}

    return agent4_app
