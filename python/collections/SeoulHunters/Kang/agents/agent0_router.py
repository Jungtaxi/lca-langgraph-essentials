from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# 1. 라우팅 결정을 위한 데이터 구조 정의
class RouteDecision(BaseModel):
    """사용자의 입력에 따라 다음으로 실행할 에이전트를 결정합니다."""
    next_agent: Literal["planner", "suggester", "path_finder", "general_chat"] = Field(
        description="다음에 실행할 에이전트의 이름. 기획/장소변경은 planner, 추천리스트 변경은 suggester, 장소선택/경로는 path_finder"
    )
    reason: str = Field(description="판단 이유")

# [2] 상태 요약 헬퍼 함수 (상태를 LLM이 이해하기 쉬운 텍스트로 변환)
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
        
    # 2. 후보 추천 상태
    candidates = state.get("main_place_candidates")
    if candidates:
        context.append(f"- 장소 추천: 완료됨 ({len(candidates)}개 후보 제시 중)")
    else:
        context.append("- 장소 추천: 아직 안 함")
        
    return "\n".join(context)

# [3] Router Node (Pure LLM Decision)
def router_node(state):
    print("\n🚦 --- [Router] LLM 기반 의도 파악 중 ---")
    
    messages = state["messages"]
    last_user_msg = messages[-1].content
    
    # 현재 상태를 텍스트로 요약
    state_context = get_state_context(state)
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    router_chain = llm.with_structured_output(RouteDecision)
    
    system_prompt = f"""
    당신은 여행 AI 서비스의 지능형 라우터입니다.
    **[현재 대화 상태]**와 **[사용자 입력]**을 종합적으로 고려하여 다음 실행 단계를 결정하세요.

    [현재 대화 상태]
    {state_context}

    [라우팅 가이드라인]
    1. **general_chat**:
       - 여행 계획과 무관한 인사("안녕"), 질문("너 누구니"), 감사 인사("고마워") 등.
       - **중요:** 상태가 미완성이더라도, 사용자의 말이 단순 잡담이면 이쪽으로 보내세요.
       
    2. **planner (기획/수정)**:
       - 여행 정보를 제공하거나("종로로 갈래", "친구랑 가"), 계획을 수정할 때("지역 바꿀래").
       - 현재 '여행 계획서'가 작성 중이라면, 사용자의 답변은 대부분 이쪽입니다.
       
    3. **suggester (재추천)**:
       - 이미 추천된 장소가 마음에 안 들어서 **'다른 곳'**을 찾을 때.
       - "더 찾아줘", "다른 식당 없어?", "목록 다시 뽑아줘".
       - (조건: 여행 계획서가 완료된 상태여야 함)
       
    4. **path_finder (선택/경로)**:
       - 추천된 후보 중에서 **'선택'**하거나 **'루트 생성'**을 요청할 때.
       - "1번이랑 3번 갈래", "여기랑 여기로 결정", "루트 짜줘".
       - (조건: 장소 추천이 완료된 상태여야 함)
    """
    
    # LLM 호출
    decision = router_chain.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_msg)
    ])
    
    print(f"   👉 [Router 판단] 입력: '{last_user_msg}' -> 결정: {decision.next_agent} ({decision.reason})")
    
    return {"next_step": decision.next_agent}