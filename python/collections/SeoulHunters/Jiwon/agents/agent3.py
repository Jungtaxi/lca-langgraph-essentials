from typing import List
from state import AgentState, TravelPreference, Place
from tools import search_local_places

THEME_KEYWORDS = {
    "맛집": ["맛집", "식당"],
    "카페": ["카페", "디저트"],
    "관광": ["관광지", "명소"],
    "쇼핑": ["쇼핑", "백화점"],
}

def extract_theme_area_pairs(tag_plan, prefs: TravelPreference):
    default_area = prefs.target_area[0]
    pairs = []
    for item in tag_plan:
        theme = item.get("theme") or item.get("tag")
        area = item.get("area") or default_area
        if theme:
            pairs.append((theme, area))
    return list(dict.fromkeys(pairs))

def agent3_node(state: AgentState) -> AgentState:
    prefs = state["prefs"]
    tag_plan = state["tag_plan"]

    pairs = extract_theme_area_pairs(tag_plan, prefs)
    place_pool: List[Place] = []

    for theme, area in pairs:
        keywords = THEME_KEYWORDS.get(theme, [theme])
        for kw in keywords:
            results = search_local_places(f"{area} {kw}", display=5)
            for r in results:
                place_pool.append(
                    Place(
                        name=r["name"],
                        category=r.get("category"),
                        address=r.get("address"),
                        road_address=r.get("road_address"),
                        mapx=r.get("mapx"),
                        mapy=r.get("mapy"),
                        link=r.get("link"),
                        telephone=r.get("telephone"),
                        theme=theme,
                        area=area,
                    )
                )

    dedup = {}
    for p in place_pool:
        key = (p.name, p.address)
        if key not in dedup:
            dedup[key] = p

    state["place_pool"] = list(dedup.values())
    return state
