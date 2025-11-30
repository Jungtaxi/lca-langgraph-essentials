from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

# [1] 라우팅 데이터 구조
class RouteDecision(BaseModel):
    next_agent: Literal["planner", "suggester", "path_finder", "general_chat"] = Field(
        description="다음에 실행할 에이전트"
    )
    reason: str = Field(description="판단 이유")

# [2] 상태 요약 헬퍼 (핵심 수정: 장소 이름과 주소까지 포함!)
def get_state_context(state):
    context = []
    
    # 1. 기획 단계 상태
    prefs = state.get("preferences")
    if not prefs:
        context.append("- 여행 계획서: 없음 (아직 시작 안 함)")
    elif not prefs.is_complete:
        context.append(f"- 여행 계획서: 작성 중 (미완성 항목: {prefs.missing_info_question})")
    else:
        context.append(f"- 여행 계획서: 완료됨 (지역: {prefs.target_area}, 테마: {prefs.themes})")
        
    # 2. 후보 추천 상태 (여기가 핵심!)
    candidates = state.get("main_place_candidates")
    if candidates:
        context.append(f"- 장소 추천 상태: 완료됨 ({len(candidates)}개 후보 제시 중)")
        
        # [수정] LLM에게 "이게 우리가 추천한 장소들이다"라고 족보를 줍니다.
        # 이름뿐만 아니라 '주소'도 같이 줘야 주소를 입력했을 때 알아듣습니다.
        place_info = []
        for c in candidates:
            # c가 객체면 .place_name, dict면 ['place_name'] (상황에 맞게 처리)
            name = getattr(c, 'place_name', str(c))
            addr = getattr(c, 'address', '')
            place_info.append(f"'{name}' (주소: {addr})")
            
        places_str = "\n".join(place_info)
        context.append(f"★ [현재 추천된 후보 리스트]:\n{places_str}")
        
    else:
        context.append("- 장소 추천 상태: 아직 안 함 (후보 없음)")
        
    return "\n".join(context)

# [3] Router Node
def router_node(state):
    print("\n🚦 --- [Router] 대화 맥락 & 데이터 기반 라우팅 ---")
    
    messages = state["messages"]
    last_user_msg = messages[-1].content
    
    # 직전 AI 메시지 가져오기 (맥락 파악용)
    last_ai_msg = "없음 (대화 시작)"
    if len(messages) >= 2 and isinstance(messages[-2], AIMessage):
        last_ai_msg = messages[-2].content
        
    # 상태 요약 가져오기
    state_context = get_state_context(state)
    print("=== Agent 0 Router log 'state_context' ===")
    print(f"   📋 현재 상태 요약:\n{state_context}")
    # 모델명 수정 (gpt-4.1-mini -> gpt-4o-mini)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    router_chain = llm.with_structured_output(RouteDecision)
    
    system_prompt = f"""
    당신은 여행 AI 서비스의 지능형 라우터입니다.
    **[현재 추천된 후보 리스트]**와 사용자의 입력을 비교하여 다음 실행 단계를 결정하세요.

    [현재 시스템 상태]
    {state_context}

    [대화 맥락]
    - 직전 AI 발언: "{last_ai_msg}"
    - 사용자 입력: "{last_user_msg}"

    [라우팅 가이드라인]
    1. **path_finder (선택/경로)**:
       - **가장 중요:** 사용자가 **[현재 추천된 후보 리스트]**에 있는 **'장소명'**이나 **'주소'**를 언급한 경우.
       - 입력이 복잡한 JSON 형태라도, 그 안에 있는 텍스트가 후보지의 주소나 이름과 일치하면 선택으로 간주하세요.
       - 예: "한남동 744-5로 갈래" (주소 일치) -> path_finder
       - 예: "1번이랑 3번" (번호 선택) -> path_finder
       
    2. **suggester (재추천)**:
       - 사용자가 추천 목록에 만족하지 못하고 "다른 거", "더 찾아줘"라고 할 때.
       - (조건: 여행 계획서가 완료된 상태여야 함)
       
    3. **planner (기획/수정)**:
       - 여행 지역이나 테마 자체를 변경하고 싶어 할 때.
       - AI가 여행 정보를 묻는 질문을 했을 때의 답변.
       
    4. **general_chat**:
       - 위 상황에 해당하지 않는 단순 잡담.
       - **주의:** 주소나 장소명이 포함되어 있다면 잡담으로 분류하지 말고 path_finder인지 확인하세요.
    """
    
    # LLM 호출
    decision = router_chain.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_msg)
    ])
    
    print(f"   👉 [Router 판단] {decision.next_agent} (이유: {decision.reason})")
    
    # AI 메시지는 굳이 저장 안 해도 됨 (State에만 반영)
    return {"next_step": decision.next_agent}