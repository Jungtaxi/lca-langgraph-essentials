from typing import Any, Dict, List
from state import AgentState

def _place_dict(p):
    if hasattr(p, "model_dump"):
        return p.model_dump()
    return dict(p)

# 먼저 추천 하나 뱉고 그거랑 가까운 곳들로 동선 구성하기
# Base_slots랑 slot_theme_priority이런거를 agent에 맡기기
BASE_SLOTS = ["morning","lunch","afternoon","snack","dinner","night"]

SLOT_THEME_PRIORITIES = {
    "morning": ["관광", "카페"],
    "lunch": ["맛집"],
    "afternoon": ["관광", "카페"],
    "snack": ["카페"],
    "dinner": ["맛집"],
    "night": ["야경", "관광"],
}

def _slots_from_intensity(i):
    if i <= 30:
        return ["lunch","afternoon","dinner"]
    if i <= 60:
        return ["morning","lunch","afternoon","dinner"]
    return BASE_SLOTS

# ==========================
#  좌표 기반 동선 정렬 유틸
# ==========================

def _coord(p: Dict):
    """네이버 좌표(mapx, mapy)를 숫자로 변환"""
    x_raw = p.get("mapx") or 0
    y_raw = p.get("mapy") or 0
    try:
        x = float(x_raw)
    except Exception:
        x = 0.0
    try:
        y = float(y_raw)
    except Exception:
        y = 0.0
    return x, y


def _sort_places_by_route(places: List[Dict]) -> List[Dict]:
    """
    하루치 장소 리스트를 받아서
    '현재 위치에서 가장 가까운 다음 장소'를 고르는 방식으로 순서 정렬.
    (nearest neighbor heuristic)
    """
    if len(places) <= 1:
        return places

    remaining = places[:]
    current = remaining.pop(0)
    ordered = [current]

    while remaining:
        cx, cy = _coord(current)
        # 현재 위치와의 거리 제곱을 기준으로 가장 가까운 곳 선택
        remaining.sort(
            key=lambda q: (
                (_coord(q)[0] - cx) ** 2 + (_coord(q)[1] - cy) ** 2
            )
        )
        current = remaining.pop(0)
        ordered.append(current)

    return ordered

def _pick_place_for_slot(slot: str, buckets: Dict[str, List[Dict]]) -> Dict | None:
    """
    기존 로직: 슬롯별 테마 우선순위에 따라 버킷에서 하나 꺼내기.
    (buckets는 theme → [place dict, ...])
    """
    priorities = SLOT_THEME_PRIORITIES.get(slot, [])

    # 1) 우선순위 테마에서 먼저 찾기
    for th in priorities:
        if buckets.get(th):
            return buckets[th].pop(0)

    # 2) 아무 테마나 남아 있는 것 중 하나 사용
    for b in buckets.values():
        if b:
            return b.pop(0)

    return None

# ==========================
#   Agent4 노드 (동선 고려)
# ==========================

def agent4_node(state: AgentState) -> AgentState:
    prefs = state["prefs"]
    place_pool = state["place_pool"]

    # 1) theme 기준으로 버킷화
    buckets: Dict[str, List[Dict]] = {}
    for p in place_pool:
        d = _place_dict(p)
        t = d.get("theme", "기타")
        buckets.setdefault(t, []).append(d)

    # 2) intensity/기간에 따른 기본 슬롯 정보
    slots = _slots_from_intensity(prefs.intensity)
    duration = prefs.duration

    routes = []

    for day in range(1, duration + 1):
        # 2-1) 이 날 사용할 장소들 먼저 선택 (slot 순서 기준)
        day_places: List[Dict] = []
        for slot in slots:
            chosen = _pick_place_for_slot(slot, buckets)
            if chosen:
                day_places.append(chosen)

        # 2-2) 선택된 장소들을 '동선 기준'으로 재정렬
        day_places_sorted = _sort_places_by_route(day_places)

        # 2-3) 정렬된 순서를 slots 에 순서대로 배치
        schedule = []
        idx = 0
        for slot in slots:
            place = day_places_sorted[idx] if idx < len(day_places_sorted) else None
            if idx < len(day_places_sorted):
                idx += 1
            schedule.append({"time": slot, "place": place})

        routes.append({"day": day, "schedule": schedule})

    state["routes"] = routes
    return state