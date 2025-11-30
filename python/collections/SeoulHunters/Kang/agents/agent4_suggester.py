from typing import Any, Dict, List
from state import AgentState

def suggester_node(state: AgentState) -> AgentState:
    """
    역할:
    - tag_plan + place_pool 을 기반으로
      '가중치가 높은 테마의 장소 후보' 몇 개를 추천한다.
    - 결과는 state["main_place_candidates"] 에 저장.
    """
    prefs = state["preferences"]
    scored_tags = state["strategy"]
    place_pool = state["candidates"]
    print("=== Suggester Node LOG ===")
    print(f"Preferences: {prefs}")
    print(f"Tag Plan: {scored_tags}")
    print(f"Place Pool Size: {len(place_pool)}")
    print("=========================")
    # 1) tag_plan에서 weight 높은 순으로 상위 몇 개 테마(tag) 추출
    #    tag_plan 원소 예시 가정: {"tag": "맛집", "weight": 0.4, "visits": 4}
    #    혹은 {"theme": "카페", "weight": 0.3, ...}
    # scored_tags: List[Dict] = []
    # for item in tag_plan:
    #     if not isinstance(item, dict):
    #         continue
    #     tag = item.get("tag") or item.get("theme")
    #     weight = item.get("weight", 0)
    #     if tag:
    #         scored_tags.append({"tag": tag, "weight": weight})

    # weight 기준으로 정렬 후 상위 N개만 사용 (예: 2개)
    
    scored_tags.allocations.sort(key=lambda x: x.weight, reverse=True)
    top_tags = [t.tag_name for t in scored_tags.allocations[:2]]
    print(f"Top Tags for Suggestion: {top_tags}")
    # 2) place_pool에서 해당 theme/tag에 해당하는 장소들 필터링
    #    theme 필드가 "맛집", "카페" 등으로 들어있다고 가정.
    candidates: List[Dict] = []
    seen_keys = set()

    # 현재 병목
    for p in place_pool:
        theme = p.tag_name
        if not theme:
            continue
        if theme not in top_tags:
            continue

        key = p.place_name
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(p)

    MAX_CANDIDATES = 3
    main_place_candidates = candidates[:MAX_CANDIDATES]

    # state에 저장 (여기서는 dict 리스트 그대로 두고, 나중에 필요하면 Place로 다시 감싸도 됨)
    return {"main_place_candidates": main_place_candidates}

def agent4_wait_node(state: AgentState) -> AgentState:
    """
    역할:
    - agent4_suggest_node가 main_place_candidates를 채운 뒤
      그래프를 여기서 멈추게 하기 위한 '대기 노드'
    - 여기서는 state를 건드리지 않고 그대로 반환만 함.
    """
    return state