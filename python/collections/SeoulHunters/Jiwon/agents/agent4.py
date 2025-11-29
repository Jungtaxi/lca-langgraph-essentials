from typing import Any, Dict, List
from state import AgentState

def _place_dict(p):
    if hasattr(p, "model_dump"):
        return p.model_dump()
    return dict(p)

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

def agent4_node(state: AgentState) -> AgentState:
    prefs = state["prefs"]
    place_pool = state["place_pool"]

    buckets: Dict[str, List[Dict]] = {}
    for p in place_pool:
        d = _place_dict(p)
        t = d.get("theme","기타")
        buckets.setdefault(t,[]).append(d)

    slots = _slots_from_intensity(prefs.intensity)
    duration = prefs.duration

    routes = []
    for day in range(1, duration+1):
        schedule=[]
        for slot in slots:
            priorities = SLOT_THEME_PRIORITIES.get(slot,[])
            chosen=None
            for th in priorities:
                if buckets.get(th):
                    chosen=buckets[th].pop(0)
                    break
            if not chosen:
                for b in buckets.values():
                    if b:
                        chosen=b.pop(0)
                        break
            schedule.append({"time":slot,"place":chosen})
        routes.append({"day":day,"schedule":schedule})

    state["routes"] = routes
    return state
