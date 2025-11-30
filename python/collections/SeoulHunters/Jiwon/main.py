from langgraph.graph import StateGraph, END
from state import AgentState
from agents.agent1 import agent1_node
from agents.agent2 import agent2_node
from agents.agent3 import agent3_node
from agents.agent4_suggest import agent4_suggest_node, agent4_wait_node
from agents.agent5_route import agent5_route_node
import time

SLOT_KO = {
    "morning": "오전",
    "lunch": "점심",
    "afternoon": "오후",
    "snack": "간식/카페",
    "dinner": "저녁",
    "night": "야간",
}

def build_graph_step1():
    graph = StateGraph(AgentState)

    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4_suggest", agent4_suggest_node)
    graph.add_node("agent4_wait", agent4_wait_node)

    graph.set_entry_point("agent1")
    
    graph.add_edge("agent1","agent2")
    graph.add_edge("agent2","agent3")
    graph.add_edge("agent3", "agent4_suggest")
    graph.add_edge("agent4_suggest", "agent4_wait")
    graph.add_edge("agent4_wait", END) 

    return graph.compile()

def build_graph_step2():
    g = StateGraph(AgentState)
    g.add_node("agent5_route", agent5_route_node)
    g.set_entry_point("agent5_route")
    g.add_edge("agent5_route", END)
    return g.compile()

if __name__ == "__main__":
    user_input = input("✈️ 여행 계획 문장을 입력하세요:\n> ")
    # 1차 실행: prefs / tag_plan / place_pool / main_place_candidates 까지
    app1 = build_graph_step1()
    # 실행시간 확인 line 49,50,102,103
    # start = time.perf_counter()
    state = app1.invoke({"user_input": user_input})
    
    #-----------------------------------------
    # prefs = state["prefs"]
    # tag_plan = state.get("tag_plan")
    # candidates = state.get("main_place_candidates", [])

    # print("\n=== TravelPreference ===")
    # print(prefs)
    # print("\n=== tag_plan ===")
    # print(tag_plan)

    # # 2) 메인 후보 장소 보여주기
    # print("\n=== 메인 후보 장소 목록 ===")
    # if not candidates:
    #     print("(후보가 없습니다. 바로 전체 루트를 만들게요.)")
    # else:
    #     for i, place in enumerate(candidates, start=1):
    #         # pydantic이면 dict로
    #         if hasattr(place, "model_dump"):
    #             place = place.model_dump()

    #         name = place.get("name", "이름 없음")
    #         theme = place.get("theme", "")
    #         area = place.get("area", "")
    #         addr = place.get("road_address") or place.get("address") or ""

    #         print(f"[{i}] {name}")
    #         if theme or area:
    #             print(f"    · 테마/지역: {theme} / {area}")
    #         if addr:
    #             print(f"    · 주소: {addr}")

    # # 3) 사용자에게 메인 장소 선택 받기
    # selected = []
    # if candidates:
    #     raw = input("\n메인으로 잡고 싶은 장소 번호를 콤마로 입력하세요 (예: 1,3) 또는 그냥 엔터:\n> ").strip()
    #     if raw:
    #         try:
    #             idxs = [int(x) for x in raw.split(",")]
    #             for idx in idxs:
    #                 if 1 <= idx <= len(candidates):
    #                     sel = candidates[idx - 1]
    #                     # pydantic이면 그대로, dict면 그대로 사용
    #                     selected.append(sel)
    #         except ValueError:
    #             print("번호를 제대로 입력하지 않아 선택을 무시하고 진행합니다.")

    # # 4) 선택 결과를 state에 심기
    # state["selected_main_places"] = selected
    #------------------------------------------------
    
    # elapsed = time.perf_counter() - start
    # print(f"실행시간-llm: {elapsed:.2f}초")

    # 5) 2차 실행: agent5_route만 돌려서 최종 routes 생성
    app2 = build_graph_step2()
    final_state = app2.invoke(state)

    routes = final_state["routes"]

    print("\n=== 최종 ROUTES ===")
    for day in routes:
        print(f"\n[Day {day['day']}]")
        for stop in day["schedule"]:
            place = stop["place"]
            idx = stop.get("order", 0)

            if not place:
                print(f"  - #{idx}: (추천 장소 없음)")
                continue

            if hasattr(place, "model_dump"):
                place = place.model_dump()

            name = place.get("name", "이름 없음")
            theme = place.get("theme")
            area = place.get("area")
            category = place.get("category")
            addr = place.get("road_address") or place.get("address")

            print(f"  - #{idx}: {name}")
            if theme or area:
                print(f"      · 테마/지역: {theme} / {area}")
            if category:
                print(f"      · 카테고리: {category}")
            if addr:
                print(f"      · 주소: {addr}")



    # print("\n=== 최종 ROUTES ===")
    # for day in routes:
    #     print(f"\n[Day {day['day']}]")
    #     for idx, slot in enumerate(day["schedule"], start=1):
    #         place = slot["place"]

    #         if not place:
    #             print(f"  - #{idx}: (추천 장소 없음)")
    #             continue

    #         # Pydantic이면 dict 변환
    #         if hasattr(place, "model_dump"):
    #             place = place.model_dump()

    #         name = place.get("name", "이름 없음")
    #         theme = place.get("theme")
    #         area = place.get("area")
    #         category = place.get("category")
    #         addr = place.get("road_address") or place.get("address")

    #         print(f"  - #{idx}: {name}")
    #         if theme or area:
    #             print(f"      · 테마/지역: {theme} / {area}")
    #         if category:
    #             print(f"      · 카테고리: {category}")
    #         if addr:
    #             print(f"      · 주소: {addr}")


#-----------------------------------------------------


    # routes = final_state["routes"]

    # # 6) 최종 루트 출력 (기존 포맷 유지)
    # print("\n=== 최종 ROUTES ===")
    # for day in routes:
    #     print(f"\n[Day {day['day']}]")
    #     for slot in day["schedule"]:
    #         place = slot["place"]
    #         slot_ko = SLOT_KO.get(slot["time"], slot["time"])

    #         if not place:
    #             print(f"  - {slot_ko}: (추천 장소 없음)")
    #             continue

    #         # Pydantic이면 dict 변환
    #         if hasattr(place, "model_dump"):
    #             place = place.model_dump()

    #         name = place.get("name", "이름 없음")
    #         theme = place.get("theme")
    #         area = place.get("area")
    #         category = place.get("category")
    #         addr = place.get("road_address") or place.get("address")

    #         print(f"  - {slot_ko}: {name}")
    #         if theme or area:
    #             print(f"      · 테마/지역: {theme} / {area}")
    #         if category:
    #             print(f"      · 카테고리: {category}")
    #         if addr:
    #             print(f"      · 주소: {addr}")
#--------------------------------------------------
    # app = build_graph()
    # result = app.invoke({"user_input": user_input})

    # prefs = result["prefs"]
    # tag_plan = result["tag_plan"]
    # routes = result["routes"]

    # print("\n=== TravelPreference ===")
    # print(prefs)
    # print("\n=== tag_plan ===")
    # print(tag_plan)
    # print("\n=== ROUTES ===")
    # for day in routes:
    #     print(f"\n[Day {day['day']}]")
    #     for slot in day["schedule"]:
    #         place = slot["place"]
    #         slot_ko = {
    #             "morning":"오전","lunch":"점심","afternoon":"오후",
    #             "snack":"간식/카페","dinner":"저녁","night":"야간"
    #         }.get(slot["time"],slot["time"])

    #         if not place:
    #             print(f"  - {slot_ko}: (추천 장소 없음)")
    #             continue

    #         print(f"  - {slot_ko}: {place['name']}")
    #         if place.get("theme"):
    #             print(f"      · 테마/지역: {place['theme']} / {place['area']}")
    #         if place.get("category"):
    #             print(f"      · 카테고리: {place['category']}")
    #         if place.get("road_address"):
    #             print(f"      · 주소: {place['road_address']}")
