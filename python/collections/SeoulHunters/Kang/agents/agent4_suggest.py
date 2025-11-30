import json
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState

# --- [스키마 정의] ---
class SuggestionOutput(BaseModel):
    selected_indices: List[int] = Field(
        description="추천할 장소의 인덱스 번호 리스트 (0부터 시작)"
    )
    reasoning: str = Field(
        description="이 장소들을 선정한 이유 (사용자 피드백 반영 여부 포함)"
    )

# --- [System Prompt 1: 초기 추천용] ---
SYSTEM_PROMPT_INITIAL = """
당신은 여행 일정 플래너의 수석 큐레이터입니다.
제공된 '후보 장소 리스트' 중에서, 사용자의 취향과 전략에 가장 잘 맞는 **'메인 추천 장소(Anchor)'를 3~5개 엄선**하세요.

[사용자 선호도]
{prefs_json}

[후보 장소 리스트 (Weight 상위)]
{candidate_summary}

[지시사항]
1. **적합성**: 사용자의 요청사항(themes, additional_notes)에 가장 부합하는 곳을 고르세요.
2. **다양성**: [식당, 카페, 관광지] 등 카테고리를 적절히 섞으세요. (단, 맛집 투어라면 식당 위주 가능)
3. **출력**: 선택한 장소의 **인덱스 번호(Integer)** 리스트를 반환하세요.
"""

# --- [System Prompt 2: 재추천(피드백 반영)용] ---
SYSTEM_PROMPT_FEEDBACK = """
당신은 여행 큐레이터입니다. 사용자가 **이전 추천을 거절하고 새로운 요구사항(피드백)**을 제시했습니다.
전체 후보군(Pool)을 다시 검토하여, **사용자의 피드백을 충족하는 새로운 장소**를 찾아내세요.

[사용자 선호도 (기본)]
{prefs_json}

[⛔ 제외할 장소 (이전 추천)]
{excluded_names}

[🗣️ 사용자 피드백 (가장 중요)]
"{user_feedback}"

[후보 장소 리스트 (전체 재검토)]
{candidate_summary}

[지시사항]
1. **피드백 최우선**: 사용자의 피드백(예: "더 조용한 곳", "고기 말고 회", "분위기 좋은 곳")을 최우선 기준으로 삼으세요.
2. **제외 장소 회피**: 위 [제외할 장소]에 있는 곳은 절대 다시 추천하지 마세요.
3. **Weight 무시 가능**: 피드백에 맞는다면 Weight가 다소 낮더라도 선택하세요.
4. **출력**: 선택한 장소의 **인덱스 번호(Integer)** 리스트를 반환하세요.
"""


# --- [Node] Suggester ---
def agent4_suggest_node(state: AgentState):
    print("\n✨ --- [Agent 4] Phase 1: 후보 장소 제안 (Dual Mode) ---")
    
    prefs = state.get("preferences")
    place_pool = state.get("candidates")
    prev_candidates = state.get("main_place_candidates") # 이전 추천 기록
    
    if not place_pool:
        print("   ⚠️ 후보군(Pool)이 없습니다.")
        return {}

    # LLM 설정
    llm = ChatOpenAI(model='gpt-4o', temperature=0.7)
    structured_llm = llm.with_structured_output(SuggestionOutput)

    # --- [모드 결정 및 데이터 준비] ---
    
    # 1. 공통 데이터 준비 (Pool 정렬)
    # 초기 추천: Weight 순 정렬
    # 재추천: 피드백에 따라 달라질 수 있지만, 일단 기본 품질 보장을 위해 Weight 순 정렬 유지
    sorted_pool = sorted(place_pool, key=lambda x: x.weight, reverse=True)
    
    # LLM에게 보여줄 후보 개수 (재추천 시에는 더 넓은 범위를 탐색하도록 설정)
    pool_limit = 50 if prev_candidates else 30 
    target_pool = sorted_pool[:pool_limit]

    # 후보 리스트 텍스트 생성
    candidate_summary = ""
    for i, p in enumerate(target_pool):
        candidate_summary += f"{i}. [{p.category}] {p.place_name} (키워드:{p.keyword}, W:{p.weight})\n"

    # 2. 분기 처리 (Initial vs Feedback)
    
    if prev_candidates:
        # === [Case B: 재추천 모드] ===
        print("   🔄 사용자 피드백 반영 재추천 모드 진입")
        
        last_user_msg = state["messages"][-1].content
        excluded_names = [p.place_name for p in prev_candidates]
        
        # Prompt 2 사용
        prompt = SYSTEM_PROMPT_FEEDBACK.format(
            prefs_json=json.dumps(prefs.model_dump(), indent=2, ensure_ascii=False),
            excluded_names=", ".join(excluded_names),
            user_feedback=last_user_msg,
            candidate_summary=candidate_summary
        )
        
    else:
        # === [Case A: 초기 추천 모드] ===
        print("   🆕 초기 추천 모드 진입")
        
        # Prompt 1 사용
        prompt = SYSTEM_PROMPT_INITIAL.format(
            prefs_json=json.dumps(prefs.model_dump(), indent=2, ensure_ascii=False),
            candidate_summary=candidate_summary
        )

    # 3. LLM 실행
    try:
        result = structured_llm.invoke([SystemMessage(content=prompt)])
        selected_indices = result.selected_indices
        print(f"   🤖 AI Reasoning: {result.reasoning}")
    except Exception as e:
        print(f"LLM Error: {e}")
        selected_indices = [0, 1, 2] # Fallback

    # 4. 인덱스 -> 객체 매핑
    main_candidates = []
    seen = set()
    
    # (재추천일 경우 제외할 이름 목록)
    excluded_names_set = {p.place_name for p in prev_candidates} if prev_candidates else set()

    for idx in selected_indices:
        if 0 <= idx < len(target_pool):
            place = target_pool[idx]
            
            # 중복 및 제외 장소 필터링
            if place.place_name in seen: continue
            if place.place_name in excluded_names_set: continue # LLM이 실수로 또 골랐을 경우 방어
            
            main_candidates.append(place)
            seen.add(place.place_name)

    print(f"   ✅ {len(main_candidates)}개 장소 선정 완료.")
    
    # State 업데이트 (덮어쓰기)
    return {"main_place_candidates": main_candidates}