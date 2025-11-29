# agents/agent5_route.py

from typing import Any, Dict, List, Optional
from state import AgentState

def _place_dict(p: Any) -> Dict:
    if hasattr(p, "model_dump"):
        return p.model_dump()
    return dict(p)

# 우리가 비율로 다룰 메인 테마들
MAIN_TAGS = ["음식점", "카페", "쇼핑"]

def _build_theme_counts_per_day(tag_plan: List[Dict], intensity: int) -> Dict[str, int]:
    """
    tag_plan의 weight 를 이용해
    '하루에 각 테마별로 몇 군데 갈지'를 계산.
    예: 5스팟/day이면 음식점2, 카페1, 쇼핑2 이런 식.
    """
    # 1) 하루에 몇 스팟 갈지 intensity로 결정
    #    대충: 힐링(<=30): 4, 보통(<=60):5, 빡셈:6
    if intensity is None:
        intensity = 50
    try:
        intensity = int(intensity)
    except Exception:
        intensity = 50

    if intensity <= 30:
        total_stops = 4
    elif intensity <= 60:
        total_stops = 5
    else:
        total_stops = 6

    # 2) tag_plan에서 MAIN_TAGS만 골라서 weight 가져오기
    items: List[Dict] = []
    for item in tag_plan:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag") or item.get("theme")
        weight = item.get("weight", 0)
        if tag in MAIN_TAGS:
            items.append({"tag": tag, "weight": float(weight)})

    # 아무 것도 없으면 기본값
    if not items:
        items = [
            {"tag": "음식점", "weight": 0.4},
            {"tag": "카페",   "weight": 0.3},
            {"tag": "쇼핑",   "weight": 0.3},
        ]

    # weight 정규화
    total_w = sum(x["weight"] for x in items)
    if total_w <= 0:
        for x in items:
            x["weight"] = 1.0
        total_w = float(len(items))
    for x in items:
        x["weight"] /= total_w

    # 3) weight * total_stops 로 개수 배분 (반올림)
    counts: Dict[str, int] = {}
    for i, x in enumerate(items):
        raw = x["weight"] * total_stops
        c = int(round(raw))
        # 최소 1개는 보장 (제일 큰 weight 하나만)
        if c == 0 and i == 0:
            c = 1
        counts[x["tag"]] = c

    # 4) 총합이 total_stops랑 안 맞으면 조정
    curr_sum = sum(counts.values())
    # 너무 많으면 큰 weight부터 줄이기
    while curr_sum > total_stops:
        # weight 높은 순으로 줄여 나감
        # items는 weight 내림차순으로 정렬해서 사용
        items_sorted = sorted(items, key=lambda x: x["weight"], reverse=True)
        for x in items_sorted:
            t = x["tag"]
            if counts[t] > 0 and curr_sum > total_stops:
                counts[t] -= 1
                curr_sum -= 1
    # 너무 적으면 큰 weight부터 늘리기
    while curr_sum < total_stops:
        items_sorted = sorted(items, key=lambda x: x["weight"], reverse=True)
        for x in items_sorted:
            if curr_sum >= total_stops:
                break
            t = x["tag"]
            counts[t] = counts.get(t, 0) + 1
            curr_sum += 1

    return counts  # 예: {"음식점":2, "카페":1, "쇼핑":2}


def _build_day_sequence(counts: Dict[str, int]) -> List[str]:
    """
    예: {"음식점":2, "카페":1, "쇼핑":2} -> ["음식점","쇼핑","카페","음식점","쇼핑"]
    - 가능한 한 같은 테마가 연속되지 않게 구성
    """
    seq: List[str] = []
    # 남은 개수 복사
    remain = counts.copy()

    last_tag: Optional[str] = None

    while sum(remain.values()) > 0:
        # 아직 남아 있는 태그들 중에서
        # 1순위: last_tag와 다르고, 남은 개수가 많은 것
        candidates = [
            (tag, c) for tag, c in remain.items()
            if c > 0 and tag != last_tag
        ]
        if not candidates:
            # 어쩔 수 없이 같은 태그 연속 허용
            candidates = [(tag, c) for tag, c in remain.items() if c > 0]
            if not candidates:
                break

        # 남은 개수가 많은 순으로 정렬해서 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen_tag = candidates[0][0]

        seq.append(chosen_tag)
        remain[chosen_tag] -= 1
        last_tag = chosen_tag

    return seq


def agent5_route_node(state: AgentState) -> AgentState:
    """
    새 버전:
    - 하루를 time-slot(오전/점심/오후)으로 나누지 않고
      '스팟 시퀀스'로만 구성.
    - 예: 식당2, 카페1, 쇼핑2 → [식당,쇼핑,카페,식당,쇼핑] 순서로
      place_pool에서 실제 장소를 뽑아서 채운다.
    """
    prefs = state["prefs"]
    tag_plan = state.get("tag_plan") or []
    place_pool = state["place_pool"]
    selected_main_places = state.get("selected_main_places") or []

    intensity = prefs.intensity or 50
    duration = prefs.duration or 1

    # 1) place_pool을 theme 기준으로 버킷화
    #    (theme 예: "음식점", "카페", "쇼핑", "술집" 등)
    buckets: Dict[str, List[Dict]] = {}
    for p in place_pool:
        d = _place_dict(p)
        theme = d.get("theme") or "기타"
        buckets.setdefault(theme, []).append(d)

    # "술집"도 일단 음식점 계열로 같이 쓰고 싶다면 여기서 매핑 가능
    def map_theme_to_tag(theme: str) -> str:
        if theme in ["술집", "맛집"]:
            return "음식점"
        return theme

    # 2) 하루에 각 태그(음식점/카페/쇼핑) 몇 개씩 갈지 계산
    base_counts = _build_theme_counts_per_day(tag_plan, intensity)
    # 예: {"음식점":2, "카페":1, "쇼핑":2}

    routes: List[Dict] = []
    anchors: List[Dict] = [_place_dict(p) for p in selected_main_places]

    anchor_idx = 0

    for day in range(1, duration + 1):
        # 2-1) 이 날 사용할 counts 복사
        day_counts = base_counts.copy()

        # 2-2) 메인 앵커(선택한 장소)가 있으면 먼저 한 개 박아두기 (보너스 느낌)
        day_places: List[Dict] = []
        if anchor_idx < len(anchors):
            anchor = anchors[anchor_idx]
            anchor_idx += 1
            day_places.append(anchor)
            # 앵커의 theme를 하나 소모했다고 보고 counts에서 1 깎을 수도 있고,
            # "앵커는 보너스"로 두고 counts는 그대로 둘 수도 있음.
            # 여기서는 보너스 취급 (깎지 않음).

        # 2-3) 이 날의 테마 시퀀스 생성 (ex: ["음식점","쇼핑","카페","음식점","쇼핑"])
        seq = _build_day_sequence(day_counts)

        # 2-4) 시퀀스대로 각 theme에서 place 하나씩 뽑기
        for tag in seq:
            # tag("음식점")에 대응하는 theme 버킷에서 꺼내기
            # theme들을 tag 기준으로 매핑
            # ex) theme "술집" -> tag "음식점"
            # 우선 순위: theme가 정확히 tag인 것 먼저, 그다음 alias
            # (간단 버전: tag와 같은 theme만 사용)
            picked: Optional[Dict] = None

            # ① theme == tag 인 버킷
            exact_bucket = buckets.get(tag, [])
            if exact_bucket:
                picked = exact_bucket.pop(0)
            else:
                # ② alias (술집 -> 음식점 등)에서 찾아보고 싶으면 여기 확장
                # 간단히 모든 버킷을 뒤져서 map_theme_to_tag(theme) == tag 인 것 찾기
                for theme, b in buckets.items():
                    if not b:
                        continue
                    if map_theme_to_tag(theme) == tag:
                        picked = b.pop(0)
                        break

            if picked:
                # 앵커 중복 방지
                if day_places and any(
                    (p.get("name") == picked.get("name") and 
                     (p.get("road_address") or p.get("address")) == (picked.get("road_address") or picked.get("address")))
                    for p in day_places
                ):
                    continue
                day_places.append(picked)

        # 2-5) 이 날의 스팟들을 '#1, #2, ...' 순서로 schedule에 넣기
        schedule = []
        for idx, pl in enumerate(day_places, start=1):
            schedule.append({
                "order": idx,   # 번호
                "place": pl,
            })

        routes.append({
            "day": day,
            "schedule": schedule,
        })

    state["routes"] = routes
    return state


# from typing import Any, Dict, List, Optional
# from state import AgentState

# def _place_dict(p: Any) -> Dict:
#     if hasattr(p, "model_dump"):
#         return p.model_dump()
#     return dict(p)

# # ==========================
# #  좌표 기반 동선 정렬 유틸
# # ==========================

# def _coord(p: Dict):
#     x_raw = p.get("mapx") or 0
#     y_raw = p.get("mapy") or 0
#     try:
#         x = float(x_raw)
#     except Exception:
#         x = 0.0
#     try:
#         y = float(y_raw)
#     except Exception:
#         y = 0.0
#     return x, y


# def _sort_places_by_route(places: List[Dict]) -> List[Dict]:
#     """nearest-neighbor 방식으로 동선 정렬"""
#     if len(places) <= 1:
#         return places

#     remaining = places[:]
#     current = remaining.pop(0)
#     ordered = [current]

#     while remaining:
#         cx, cy = _coord(current)
#         remaining.sort(
#             key=lambda q: (
#                 (_coord(q)[0] - cx) ** 2 + (_coord(q)[1] - cy) ** 2
#             )
#         )
#         current = remaining.pop(0)
#         ordered.append(current)

#     return ordered


# # ===================================================
# #   하루 방문 개수 / 식당 횟수 등 간단한 규칙 정의
# # ===================================================

# MEAL_THEMES = {"맛집", "음식점", "술집"}
# CAFE_THEMES = {"카페", "디저트"}

# def _decide_visits_per_day(intensity: int) -> int:
#     """
#     intensity 기준으로 '하루에 몇 군데 돌지' 대략 결정
#     - 0~30  : 3곳
#     - 31~60 : 5곳
#     - 61~100: 7곳
#     """
#     if intensity <= 30:
#         return 3
#     if intensity <= 60:
#         return 5
#     return 7


# def _decide_meal_limit_per_day(prefs) -> int:
#     """
#     하루에 갈 식당(밥집/술집) 최대 횟수.
#     - 기본 2번
#     - prefs.themes 에 '미식'이 있으면 3번까지 허용
#     (정교하게 '아침 안 먹음'을 반영하려면 prefs에 필드를 더 만들어야 함)
#     """
#     base = 2
#     if "미식" in (prefs.themes or []):
#         base = 3
#     return base


# def _pop_next_place(
#     buckets: Dict[str, List[Dict]],
#     used_meals: int,
#     max_meals: int,
# ) -> Optional[Dict]:
#     """
#     테마 버킷에서 다음 장소 하나 꺼내기.
#     - 아직 식당 횟수가 여유 있으면 MEAL_THEMES 우선
#     - 아니면 그 외 테마 우선
#     - 그래도 없으면 남은 거 아무거나
#     """
#     # 1) 식당/술집 우선 (아직 quota 남아있다면)
#     if used_meals < max_meals:
#         for theme in list(buckets.keys()):
#             if theme in MEAL_THEMES and buckets[theme]:
#                 return buckets[theme].pop(0)

#     # 2) 그 외 테마
#     for theme in list(buckets.keys()):
#         if theme not in MEAL_THEMES and buckets[theme]:
#             return buckets[theme].pop(0)

#     # 3) 마지막 fallback: 뭐라도 남아 있으면
#     for theme in list(buckets.keys()):
#         if buckets[theme]:
#             return buckets[theme].pop(0)

#     return None


# def agent5_route_node(state: AgentState) -> AgentState:
#     """
#     역할:
#     - prefs + place_pool + selected_main_places 를 받아
#       '시간 슬롯 없이' 하루 방문 순서만 고려한 routes 생성.
#     - 각 day마다 stop_1, stop_2 ... 형태로 순서만 의미.
#     """
#     prefs = state["prefs"]
#     place_pool = state["place_pool"]
#     selected_main_places = state.get("selected_main_places") or []

#     # 0) intensity / duration
#     intensity = int(getattr(prefs, "intensity", 50) or 50)
#     duration = int(getattr(prefs, "duration", 1) or 1)

#     # 1) 모든 장소 / 앵커를 dict 로 통일
#     all_places: List[Dict] = [_place_dict(p) for p in place_pool]
#     anchors: List[Dict] = [_place_dict(p) for p in selected_main_places]

#     # 2) theme 기준 버킷화
#     buckets: Dict[str, List[Dict]] = {}
#     for p in all_places:
#         t = p.get("theme", "기타")
#         buckets.setdefault(t, []).append(p)

#     visits_per_day = _decide_visits_per_day(intensity)
#     max_meals_per_day = _decide_meal_limit_per_day(prefs)

#     routes = []
#     anchor_idx = 0

#     for day in range(1, duration + 1):
#         day_places: List[Dict] = []

#         # (1) 이 날 앵커(메인 장소) 하나 가져오기 (있으면)
#         day_anchor: Optional[Dict] = None
#         if anchor_idx < len(anchors):
#             day_anchor = anchors[anchor_idx]
#             anchor_idx += 1
#             day_places.append(day_anchor)

#         # 식당 몇 번 썼는지 카운트
#         used_meals = 0
#         if day_anchor and (day_anchor.get("theme") in MEAL_THEMES):
#             used_meals += 1

#         # (2) 나머지 방문지 채우기
#         while len(day_places) < visits_per_day:
#             nxt = _pop_next_place(buckets, used_meals, max_meals_per_day)
#             if not nxt:
#                 break

#             # 앵커와 중복이면 스킵
#             if day_anchor and nxt.get("name") == day_anchor.get("name"):
#                 continue

#             if nxt.get("theme") in MEAL_THEMES:
#                 used_meals += 1

#             day_places.append(nxt)

#         # (3) 동선 기준으로 정렬
#         day_places_sorted = _sort_places_by_route(day_places)

#         # (4) stop_1, stop_2 ... 형태로 schedule 구성
#         schedule = []
#         for idx, place in enumerate(day_places_sorted, start=1):
#             schedule.append(
#                 {
#                     "time": f"stop_{idx}",  # 의미: 'idx번째 방문지'
#                     "place": place,
#                 }
#             )

#         routes.append({"day": day, "schedule": schedule})

#     state["routes"] = routes
#     return state

#-----------------------------------------------------------------------------------------------

# from typing import Any, Dict, List, Optional
# from state import AgentState

# def _place_dict(p: Any) -> Dict:
#     if hasattr(p, "model_dump"):
#         return p.model_dump()
#     return dict(p)

# BASE_SLOTS = ["morning", "lunch", "afternoon", "snack", "dinner", "night"]

# SLOT_THEME_PRIORITIES = {
#     "morning": ["관광", "카페"],
#     "lunch": ["맛집"],
#     "afternoon": ["관광", "카페"],
#     "snack": ["카페"],
#     "dinner": ["맛집"],
#     "night": ["야경", "관광"],
# }

# def _slots_from_intensity(i: int):
#     if i <= 30:
#         return ["lunch", "afternoon", "dinner"]
#     if i <= 60:
#         return ["morning", "lunch", "afternoon", "dinner"]
#     return BASE_SLOTS

# # ==========================
# #  좌표 기반 동선 정렬 유틸
# # ==========================

# def _coord(p: Dict):
#     x_raw = p.get("mapx") or 0
#     y_raw = p.get("mapy") or 0
#     try:
#         x = float(x_raw)
#     except Exception:
#         x = 0.0
#     try:
#         y = float(y_raw)
#     except Exception:
#         y = 0.0
#     return x, y


# def _sort_places_by_route(places: List[Dict]) -> List[Dict]:
#     if len(places) <= 1:
#         return places

#     remaining = places[:]
#     current = remaining.pop(0)
#     ordered = [current]

#     while remaining:
#         cx, cy = _coord(current)
#         remaining.sort(
#             key=lambda q: (
#                 (_coord(q)[0] - cx) ** 2 + (_coord(q)[1] - cy) ** 2
#             )
#         )
#         current = remaining.pop(0)
#         ordered.append(current)

#     return ordered


# def _pick_place_for_slot(slot: str, buckets: Dict[str, List[Dict]]) -> Optional[Dict]:
#     priorities = SLOT_THEME_PRIORITIES.get(slot, [])

#     for th in priorities:
#         if buckets.get(th):
#             return buckets[th].pop(0)

#     for b in buckets.values():
#         if b:
#             return b.pop(0)

#     return None


# def agent5_route_node(state: AgentState) -> AgentState:
#     """
#     역할:
#     - prefs + place_pool (+ selected_main_places)가 주어졌을 때
#       실제 일차별 routes를 생성한다.
#     - selected_main_places 가 있으면 루트 안에 반드시 포함되도록 우선 배치.
#     """
#     prefs = state["prefs"]
#     place_pool = state["place_pool"]
#     selected_main_places = state.get("selected_main_places") or []

#     # 1) place_pool / anchors 를 dict로 통일
#     all_places: List[Dict] = [_place_dict(p) for p in place_pool]
#     anchors: List[Dict] = [_place_dict(p) for p in selected_main_places]

#     # 2) theme 기준 버킷화 (anchors는 우선 별도로 관리)
#     buckets: Dict[str, List[Dict]] = {}
#     for p in all_places:
#         t = p.get("theme", "기타")
#         buckets.setdefault(t, []).append(p)

#     slots = _slots_from_intensity(prefs.intensity)
#     duration = prefs.duration

#     routes = []

#     # 🔥 anchors를 하루에 하나씩 배분한다는 가정 (필요하면 로직 더 바꿀 수 있음)
#     anchor_idx = 0

#     for day in range(1, duration + 1):
#         day_places: List[Dict] = []

#         # 1) 이 날 앵커(메인 장소) 하나 가져오기 (있으면)
#         day_anchor: Optional[Dict] = None
#         if anchor_idx < len(anchors):
#             day_anchor = anchors[anchor_idx]
#             anchor_idx += 1
#             day_places.append(day_anchor)

#         # 2) 나머지는 기존 로직대로 슬롯 기준으로 place 채우기
#         for slot in slots:
#             chosen = _pick_place_for_slot(slot, buckets)
#             if chosen:
#                 # 앵커와 중복이면 패스
#                 if day_anchor and (chosen.get("name") == day_anchor.get("name")):
#                     continue
#                 day_places.append(chosen)

#         # 3) 동선 기준으로 순서 정렬
#         day_places_sorted = _sort_places_by_route(day_places)

#         # 4) 정렬된 순서를 slots 에 순서대로 배치
#         schedule = []
#         idx = 0
#         for slot in slots:
#             place = day_places_sorted[idx] if idx < len(day_places_sorted) else None
#             if idx < len(day_places_sorted):
#                 idx += 1
#             schedule.append({"time": slot, "place": place})

#         routes.append({"day": day, "schedule": schedule})

#     state["routes"] = routes
#     return state