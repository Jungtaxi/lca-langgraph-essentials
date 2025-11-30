import json
from typing import Any, Dict, List, Optional
from state import AgentState, CandidatePlace
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

class RoutesOutput(BaseModel):
    selected_main_places: Optional[List[CandidatePlace]] = Field(
        description="여행 경로에 포함되는 장소"
    )
    routes_text: str = Field(
        description="여행 경로"
    )

def agent5_route_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_user_msg = messages[-1].content

    # LLM 설정
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    
    # 여기에서 진행 할 겁니다.
    # LLM 설정
    structured_llm = llm.with_structured_output(RoutesOutput)
    
    
    prefs = state["preferences"]
    # strategy: 테마 비율 정보 (tag_plan 역할)
    tag_plan = state["strategy"] or []
    # candidates: 실제 방문 후보 장소 리스트
    place_pool = state.get("candidates") or []
    main_place_candidates = state["main_place_candidates"] or []
    selected_main_places = state.get("selected_main_places") or []

    ROUTE_SYSTEM_PROMPT = f"""
    당신은 사용자의 취향과 장소 목록을 바탕으로 '여행 동선'을 짜는 전문가입니다.

    이전 단계(agent4)에서 사용자는 추천 후보 중 마음에 드는 장소를 선택했습니다.
    그 선택 결과는 selected_main_places에 담겨 전달됩니다.
    - 선택이 1개일 수도, 여러 개일 수도, 아예 없을 수도 있습니다.
    - 선택이 없더라도 전체 여행 동선을 자연스럽게 만들어야 합니다.

    아래 입력 정보를 참고하여, 한국어로 사람이 읽기 좋은 여행 일정을 설계하세요.

    [사용자 선호도]
    {json.dumps(prefs.model_dump(), indent=2, ensure_ascii=False)}

    [테마 비율 정보]
    {tag_plan}

    [추천 장소 리스트]
    {main_place_candidates}

    [후보 장소 리스트]
    {place_pool}

    [규칙]
    1. 사용자의 입력에 대해서 추천 장소 리스트와 대조해, 적절한 장소를 반드시 포함한다.

    2. 하루 방문 스팟 수
      - intensity <= 30: 하루 4곳
      - 30 < intensity <= 60: 하루 5곳
      - intensity > 60: 하루 6곳

    3. 테마 비율
      - "음식점", "카페", "쇼핑" 세 가지를 MAIN TAG로 본다.
      - 테마 비율 정보의 weight 비율을 참고하여, 하루 스팟 구성에서 이 비율을 최대한 맞춘다.
      - 테마 비율 정보가 없거나 weight 합이 0이면 기본 비율은
        - 음식점 0.4, 카페 0.3, 쇼핑 0.3 으로 가정한다.
      - "술집", "맛집" theme는 "음식점"과 비슷한 계열로 취급해도 좋다.

    4. 동선 구성
      - "음식점" 테마가 연속해서 너무 많이 나오지 않도록 섞는다.
        예: 음식점-카페-음식점-쇼핑 처럼 구성하는 것을 선호.
      - 주소(대략적인 동/구 정도)를 참고하여, 한 날 안에서는 가능한 한 비슷한 지역끼리 묶는다.
      - x, y (경도와 위도)도 참고하여, 한 날 안에서는 가능한 한 비슷한 지역끼리 묶는다.
      - 정확한 거리 계산은 하지 말고, 텍스트 주소 수준에서 상식적으로 판단한다.

    [출력 형식]
    - selected_main_places: 여행 경로에 포함되는 장소들을 구조에 맞춰 작성한다.
    - routes_text: 포함되는 장소들을 날짜에 맞게 적절히 분배한다.

    routes_text 예시 형식:

    Day 1
    1. 장소이름 (테마, 대략 위치: ○○구 ○○동)  - 간단 설명 및 동선 이유
    2. 장소이름 (테마, 대략 위치: ○○구 ○○동)
    3. ...

    Day 2
    1. 장소이름 (테마, 대략 위치: ○○구 ○○동)
    2. ...

    - 각 Day에는 방문 순서대로 번호를 매긴다.
    - 가능한 한 duration 일수만큼 Day 1 ~ Day N을 모두 채우려고 시도한다.
    - selected_main_places에 해당하는 장소가 일정에 포함되면,
      해당 줄에 "(사용자 선택 스팟)"이라고 명시해 주면 좋다.

    반드시 위 형식을 따르되, 사람 입장에서 읽기 편한 한국어 설명으로 일정을 작성하세요.
    """
    
    result = structured_llm.invoke([
        SystemMessage(content=ROUTE_SYSTEM_PROMPT),
        HumanMessage(content = last_user_msg)
    ])

    print("===agent5_log===")
    print(result)
    
    state["selected_main_places"] = result.selected_main_places
    state["routes_text"] = result.routes_text
    return state