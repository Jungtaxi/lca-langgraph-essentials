from langgraph.graph import StateGraph, END
from state import AgentState
from agents.agent1 import agent1_node
from agents.agent2 import agent2_node
from agents.agent3 import agent3_node
from agents.agent4 import agent4_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4", agent4_node)

    graph.set_entry_point("agent1")
    graph.add_edge("agent1","agent2")
    graph.add_edge("agent2","agent3")
    graph.add_edge("agent3","agent4")
    graph.add_edge("agent4", END)

    return graph.compile()

if __name__ == "__main__":
    user_input = input("✈️ 여행 계획 문장을 입력하세요:\n> ")

    app = build_graph()
    result = app.invoke({"user_input": user_input})

    prefs = result["prefs"]
    routes = result["routes"]

    print("\n=== TravelPreference ===")
    print(prefs)

    print("\n=== ROUTES ===")
    for day in routes:
        print(f"\n[Day {day['day']}]")
        for slot in day["schedule"]:
            place = slot["place"]
            slot_ko = {
                "morning":"오전","lunch":"점심","afternoon":"오후",
                "snack":"간식/카페","dinner":"저녁","night":"야간"
            }.get(slot["time"],slot["time"])

            if not place:
                print(f"  - {slot_ko}: (추천 장소 없음)")
                continue

            print(f"  - {slot_ko}: {place['name']}")
            if place.get("theme"):
                print(f"      · 테마/지역: {place['theme']} / {place['area']}")
            if place.get("category"):
                print(f"      · 카테고리: {place['category']}")
            if place.get("road_address"):
                print(f"      · 주소: {place['road_address']}")
