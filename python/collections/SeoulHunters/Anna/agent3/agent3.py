"""
agent3.py - 함수 기반 Agent3

역할:
- Agent1의 prefs(TravelPreference) + Agent2의 tag_plan 을 받아서
  네이버 지역 검색 API로 "장소 풀(place_pool)"을 만들어준다.

사용 예시 (main.py):

    from agent3.agent3 import build_agent3

    app3 = build_agent3()
    result3 = app3({
        "prefs": result1["prefs"],
        "tag_plan": result2["tag_plan"],
    })
    place_pool = result3["place_pool"]
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from naver_local_test import search_local_places

# ---------------------------------------------------------
# 1. Place 스키마 (Agent3의 출력 단위)
# ---------------------------------------------------------
class Place(BaseModel):
    name: str
    category: Optional[str] = None
    address: Optional[str] = None
    road_address: Optional[str] = None
    mapx: Optional[str] = None
    mapy: Optional[str] = None
    link: Optional[str] = None
    telephone: Optional[str] = None # 넣을까 말까 

    theme: str   # 예: "맛집", "카페", "관광"
    area: str    # 예: "종로", "홍대"
    source: str = "naver_local"


# ---------------------------------------------------------
# 2. 테마 → 검색 키워드 매핑
# ---------------------------------------------------------
THEME_KEYWORDS: Dict[str, List[str]] = {
    "맛집": ["맛집", "식당", "맛있는 집"],
    "카페": ["카페", "디저트 카페"],
    "관광": ["관광지", "명소", "유적지", "박물관"],
    "쇼핑": ["쇼핑", "백화점", "아울렛", "시장"],
    # 필요하면 계속 추가
}


# ---------------------------------------------------------
# 3. tag_plan 에서 (theme, area) 쌍 뽑아내기
#    - Agent2의 tag_plan 구조가 어떻게 생겼는지 모르니까
#      dict / str 둘 다 최대한 유연하게 처리
# ---------------------------------------------------------
def extract_theme_area_pairs(
    tag_plan: Any,
    prefs_data: Dict[str, Any],
) -> List[tuple[str, str]]:
    """
    tag_plan에서 (theme, area) 목록을 뽑아낸다.

    tag_plan의 각 원소 예시 (가정):
      - dict 형태:
          {"day": 1, "time_slot": "morning", "area": "홍대", "theme": "카페"}
        또는
          {"day": 1, "tag": "맛집", "location": "종로"}

      - str 형태:
          "맛집", "관광"  (이 경우 area는 prefs.target_area[0] 사용)
    """

    areas_default: List[str] = prefs_data.get("target_area") or ["서울"]
    default_area = areas_default[0]
    pairs: List[tuple[str, str]] = []

    if tag_plan is None:
        tag_plan = []

    for item in tag_plan:
        theme: Optional[str] = None
        area: Optional[str] = None

        # 1) dict 타입인 경우
        if isinstance(item, dict) or hasattr(item, "get"):
            # pydantic 모델일 수도 있으므로 일단 dict로 캐스팅
            if not isinstance(item, dict):
                # model_dump()가 있으면 써보고, 아니면 그대로 사용
                if hasattr(item, "model_dump"):
                    item = item.model_dump()
                else:
                    item = dict(item)

            theme = (
                item.get("theme")
                or item.get("tag")
                or item.get("category")
            )
            area = (
                item.get("area")
                or item.get("location")
                or item.get("region")
            )

        # 2) 문자열인 경우: 그냥 테마라고 가정
        elif isinstance(item, str):
            theme = item

        # theme이 없으면 스킵
        if not theme:
            continue

        # area가 없으면 prefs의 첫 번째 target_area 사용
        if not area:
            area = default_area

        pairs.append((theme, area))

    # 아무것도 안 뽑혔으면 prefs 기반으로 fallback
    if not pairs:
        themes = prefs_data.get("themes") or ["관광", "맛집"]
        pairs = [(t, default_area) for t in themes]

    # 중복 (theme, area) 제거
    uniq = list(dict.fromkeys(pairs))  # insertion order 유지
    return uniq


# ---------------------------------------------------------
# 4. (area, theme) 기준으로 네이버 검색 → Place 리스트 변환
# ---------------------------------------------------------
def search_places_for_theme_area(
    area: str,
    theme: str,
    limit: int,
    sort: str = "random",
) -> List[Place]:
    """
    (area, theme) 조합에 대해 최대 limit 개의 장소를 가져온다.
    키워드 리스트를 순차적으로 돌면서 누적.
    """

    keywords = THEME_KEYWORDS.get(theme, [theme])
    collected: List[Place] = []

    for kw in keywords:
        if len(collected) >= limit:
            break

        remain = limit - len(collected)
        display = min(5, remain)  # 네이버 API 한 번에 5개까지

        query = f"{area} {kw}"
        raw_list = search_local_places(query=query, display=display, sort=sort)

        for item in raw_list:
            place = Place(
                name=item["name"],
                category=item.get("category"),
                address=item.get("address"),
                road_address=item.get("road_address"),
                mapx=item.get("mapx"),
                mapy=item.get("mapy"),
                link=item.get("link"),
                telephone=item.get("telephone"),
                theme=theme,
                area=area,
            )
            collected.append(place)
            if len(collected) >= limit:
                break

    return collected


# ---------------------------------------------------------
# 5. prefs + tag_plan → place_pool 생성 (핵심 로직)
# ---------------------------------------------------------
def build_place_pool_from_plan(
    prefs: Any,
    tag_plan: Any,
    per_slot: int = 3,
) -> List[Place]:
    """
    Agent1의 prefs + Agent2의 tag_plan 을 받아 한 번에 place_pool을 만든다.

    - per_slot:
        tag_plan의 한 "슬롯" (예: day1-morning-맛집) 당 몇 개의 장소를 뽑을지.
        예: per_slot=3, tag_plan에 10개의 슬롯이 있으면
            최대 30개 정도를 목표로 장소를 모음.
    """

    # prefs: TravelPreference or dict
    if hasattr(prefs, "model_dump"):
        prefs_data: Dict[str, Any] = prefs.model_dump()
    else:
        prefs_data = dict(prefs)

    # 1) tag_plan에서 (theme, area) 리스트 뽑기
    theme_area_pairs = extract_theme_area_pairs(tag_plan, prefs_data)

    # 2) 각 (theme, area) 슬롯당 per_slot개 검색
    pool: List[Place] = []
    for theme, area in theme_area_pairs:
        places = search_places_for_theme_area(
            area=area,
            theme=theme,
            limit=per_slot,
            sort="random",
        )
        pool.extend(places)

    # 3) 이름 + 주소 기준으로 전체 중복 제거
    dedup: Dict[tuple, Place] = {}
    for p in pool:
        key = (p.name, p.road_address or p.address or "")
        if key not in dedup:
            dedup[key] = p

    return list(dedup.values())


# ---------------------------------------------------------
# 6. Agent3 빌더 함수 (함수 기반)
# ---------------------------------------------------------
def build_agent3(per_slot: int = 3):
    """
    main.py 에서:

        from agent3.agent3 import build_agent3

        app3 = build_agent3()
        result3 = app3({
            "prefs": result1["prefs"],
            "tag_plan": result2["tag_plan"],
        })

    이런 식으로 사용.
    """
    def agent3_app(inputs: Dict[str, Any]) -> Dict[str, Any]:
        prefs = inputs.get("prefs")
        tag_plan = inputs.get("tag_plan")

        if prefs is None:
            raise ValueError("Agent3: 'prefs'가 필요합니다.")
        if tag_plan is None:
            # tag_plan 이 없어도 동작은 하게 하고 싶다면, 경고만 띄우고 진행해도 됨
            # 여기서는 명시적으로 에러를 던지도록 함
            raise ValueError("Agent3: 'tag_plan'이 필요합니다. (Agent2 결과)")

        place_pool = build_place_pool_from_plan(
            prefs=prefs,
            tag_plan=tag_plan,
            per_slot=per_slot,
        )

        return {"place_pool": place_pool}

    return agent3_app
