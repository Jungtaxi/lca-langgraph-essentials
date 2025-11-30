import json
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from state import AgentState, CandidatePlace

# --- [스키마 정의] ---

class SuggestionOutput(BaseModel):
    selected_indices: List[int] = Field(
        description="추천할 장소의 인덱스 번호 리스트 (0부터 시작)"
    )
    reasoning: str = Field(
        description="이 장소들을 선정한 이유"
    )

# --- [Node 1] 제안 노드 (Suggest) ---
def agent4_suggest_node(state: AgentState):
    print("\n✨ --- [Agent 4] Phase 1: 후보 장소 제안 (LLM) ---")
    
    prefs = state.get("preferences")
    strategy = state.get("strategy")
    place_pool = state.get("candidates")
    
    if not place_pool:
        print("   ⚠️ 후보군(Pool)이 없습니다.")
        return {}

    # 가중치(Weight) 높은 순으로 정렬 후 상위 15개 추출
    # 이렇게 해야 LLM이 가장 중요한 장소들을 먼저 볼 수 있습니다.
    sorted_pool = sorted(place_pool, key=lambda x: x.weight, reverse=True)
    target_pool = sorted_pool[:15] # 상위 15개만 LLM에게 노출

    # LLM에게 보여줄 요약 정보 생성
    candidate_summary = ""
    for i, p in enumerate(target_pool):
        candidate_summary += f"{i}. [{p.category}] {p.place_name} (W:{p.weight})\n"

    # LLM 설정
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
    structured_llm = llm.with_structured_output(SuggestionOutput)
    
    system_prompt = f"""
    당신은 여행 일정 플래너의 수석 큐레이터입니다.
    제공된 '상위 가중치 장소 리스트' 중에서, 사용자의 취향과 전략에 가장 잘 맞는 **'메인 추천 장소(Anchor)'를 3~5개 엄선**하세요.

    [사용자 선호도]
    {json.dumps(prefs.model_dump(), indent=2, ensure_ascii=False)}
   
    [후보 장소 리스트 (Weight 상위 30개)]
    {candidate_summary}

    [지시사항]
    1. **적합성**: 사용자의 요청사항(예: 조용한 곳, 흑백요리사 등)에 가장 부합하는 곳을 고르세요.
    2. **다양성**: 식당만 5개 고르지 말고, [식당, 카페, 관광지] 등 카테고리를 적절히 섞으세요.
    3. **출력**: 선택한 장소의 **인덱스 번호(Integer)** 리스트를 반환하세요.
    """
    
    try:
        result = structured_llm.invoke([SystemMessage(content=system_prompt)])
        selected_indices = result.selected_indices
    except Exception as e:
        print(f"LLM Error: {e}")
        # 에러나면 그냥 상위 3개 선택
        selected_indices = [0, 1, 2]

    # [중요] 인덱스를 객체로 변환할 때, 반드시 'target_pool'(정렬된 리스트)에서 가져와야 함
    main_candidates = []
    for idx in selected_indices:
        if 0 <= idx < len(target_pool):
            main_candidates.append(target_pool[idx])
            
    # 중복 제거 (장소명 기준)
    unique_candidates = []
    seen = set()
    for c in main_candidates:
        if c.place_name not in seen:
            unique_candidates.append(c)
            seen.add(c.place_name)

    print(f"   ✅ {len(unique_candidates)}개 장소 제안 완료 (From Top 30).")
    return {"main_place_candidates": unique_candidates}