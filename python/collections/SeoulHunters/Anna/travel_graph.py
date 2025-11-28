# travel_graph.py

from typing import TypedDict, Any, List, Dict
from langgraph.graph import StateGraph, END

from agent1.agent1 import build_agent1
from agent2.agent2 import build_agent2
from agent3.agent3 import build_agent3
from agent4.agent4 import build_agent4


# 1) 그래프에서 공유할 상태 정의
class TravelState(TypedDict, total=False):
    user_input: str
    prefs: Any          # TravelPreference (pydantic)
    tag_plan: Any       # Agent2 결과
    place_pool: List[Any]
    routes: Any


# 2) 각 Agent를 LangGraph 노드로 래핑

# Agent1: user_input -> prefs
_app1 = build_agent1()

def agent1_node(state: TravelState) -> TravelState:
    """
    입력:
        state["user_input"]
    출력:
        state["prefs"] (TravelPreference)
    """
    user_input = state.get("user_input")
    # build_agent1 쪽에서 "prefs"까지 채워주는 구조라고 가정
    result1 = _app1.invoke({
        "user_input": user_input,
        "prefs": None,
    })
    state["prefs"] = result1["prefs"]
    return state


# Agent2: prefs -> tag_plan
_app2 = build_agent2()

def agent2_node(state: TravelState) -> TravelState:
    prefs = state["prefs"]
    result2 = _app2.invoke({"prefs": prefs})
    state["tag_plan"] = result2["tag_plan"]
    return state


# Agent3: prefs + tag_plan -> place_pool
_app3 = build_agent3()  # 필요하면 per_slot=5 같은 옵션을 여기서 조절

def agent3_node(state: TravelState) -> TravelState:
    prefs = state["prefs"]
    tag_plan = state["tag_plan"]
    result3 = _app3({
        "prefs": prefs,
        "tag_plan": tag_plan,
    })
    state["place_pool"] = result3["place_pool"]
    return state


# Agent4: prefs + place_pool -> routes
_app4 = build_agent4()

def agent4_node(state: TravelState) -> TravelState:
    prefs = state["prefs"]
    place_pool = state["place_pool"]
    result4 = _app4({
        "prefs": prefs,
        "place_pool": place_pool,
    })
    state["routes"] = result4["routes"]
    return state


# 3) LangGraph 그래프 정의

def build_travel_graph():
    """
    TravelState를 입력받아 Agent1~4를 순서대로 실행하는 LangGraph 앱 생성
    """
    graph = StateGraph(TravelState)

    # 노드 등록
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4", agent4_node)

    # 시작 지점: agent1
    graph.set_entry_point("agent1")

    # 에지 연결: 1 → 2 → 3 → 4 → END
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", "agent4")
    graph.add_edge("agent4", END)

    app = graph.compile()
    return app

build_travel_graph()
