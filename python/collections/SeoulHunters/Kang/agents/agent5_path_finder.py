import json
import math
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# state.py에서 정의한 클래스들 import
from state import AgentState, CandidatePlace

# [1] LLM 출력용 스키마 (가볍게 이름만 리턴받음)
class RoutePlanOutput(BaseModel):
    ordered_place_names: List[str] = Field(
        description="최적의 동선 순서대로 정렬된 장소 이름 리스트 (사용자 선택 포함 + 부족하면 Pool에서 추가)"
    )
    routes_text: str = Field(
        description="해당 경로에 대한 매력적인 설명 (마크다운 형식)"
    )

# [2] 거리 계산 헬퍼 (단순 유클리드 거리, 정렬용)
def calc_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def agent5_route_node(state: AgentState) -> AgentState:
    print("\n🚗 --- [Agent 5] 최종 경로 생성 및 최적화 ---")
    
    messages = state["messages"]
    last_user_msg = messages[-1].content

    # 1. 데이터 준비
    prefs = state["preferences"]
    place_pool = state.get("candidates") or []           # 전체 수집 데이터 (Agent 3)
    main_candidates = state.get("main_place_candidates") or [] # 제안했던 후보 (Agent 4)
    
    # LLM에게 보여줄 데이터 경량화 (토큰 절약 & 집중력 향상)
    # 전체 Pool을 다 보여주면 너무 많으니, Weight 상위 + 메인 후보만 추림
    combined_pool = {p.place_name: p for p in place_pool + main_candidates} # 중복제거용 Dict
    
    # LLM에게 넘길 텍스트 요약본 생성
    candidates_txt = ""
    for name, p in list(combined_pool.items()): 
        candidates_txt += f"- {name} ({p.category}, 키워드:{p.keyword}, 좌표:{p.y},{p.x})\n"

    # 2. 목표 방문 장소 개수 계산 (Intensity 기반)
    intensity = prefs.intensity or 50
    if intensity <= 30: target_count = 3
    elif intensity <= 60: target_count = 4
    else: target_count = 5
    
    # Duration 고려 (1일 기준이므로 곱하기 1, 만약 N일이면 늘어남)
    # 여기서는 '하루 코스'를 짜는 것으로 가정
    
    # 3. LLM 설정
    llm = ChatOpenAI(model="gpt-4.1", temperature=0) # gpt-4o 추천 (복잡한 추론 필요)
    structured_llm = llm.with_structured_output(RoutePlanOutput)

    # 4. 프롬프트 작성
    system_prompt = f"""
    당신은 여행 동선 설계 전문가입니다.
    사용자의 선택과 전체 후보군을 조합하여 **가장 효율적이고 매력적인 하루 여행 코스**를 짜세요.

    [사용자 프로필]
    - 테마: {prefs.themes}
    - 강도: {intensity} (목표 방문지 수: 약 {target_count}곳)
    - 요청사항: "{prefs.additional_notes}"

    [사용자가 보고 있던 추천 후보 (Agent 4 제안)]
    {", ".join([p.place_name for p in main_candidates])}

    [전체 이용 가능한 장소 풀 (Pool)]
    {candidates_txt}

    [사용자 입력 (선택 사항)]
    "{last_user_msg}"

    [동선 설계 규칙]
    1. **사용자 선택 반영**: 사용자 입력에서 특정 장소를 선택했다면, 그 장소를 **반드시 포함**하고 **우선순위(Anchor)**로 두세요.
    2. **빈자리 채우기**: 선택된 장소가 목표({target_count}개)보다 적다면, '장소 풀'에서 동선(좌표)과 테마 밸런스를 고려해 추가하세요.
       - 동선 효율성: 선택된 장소와 좌표가 가까운 곳 위주로 선택.
       - 테마 밸런스: 식당 -> 카페 -> 관광지 -> 쇼핑 순서 등 지루하지 않게 배치.
    3. **출력**: 방문 순서대로 장소의 **'정확한 이름'**만 리스트에 담으세요.
    """

    # 5. 실행
    try:
        result = structured_llm.invoke([SystemMessage(content=system_prompt)])
    except Exception as e:
        print(f"Error in Agent 5: {e}")
        return state

    # 6. [핵심] LLM이 뱉은 이름(String)을 실제 객체(CandidatePlace)로 복원
    # 이 과정이 있어야 지도에 핀이 찍힙니다.
    final_route_objects = []
    
    print(f"   📍 AI 제안 경로: {result.ordered_place_names}")
    
    for name in result.ordered_place_names:
        # 이름이 유사한 객체를 찾음 (완전 일치 우선, 없으면 포함 여부)
        found = None
        
        # 1차 시도: 완전 일치
        if name in combined_pool:
            found = combined_pool[name]
        
        # 2차 시도: 부분 일치 (LLM이 이름을 약간 줄여서 말했을 경우 대비)
        if not found:
            for real_name, p in combined_pool.items():
                if name in real_name or real_name in name:
                    found = p
                    break
        
        if found:
            final_route_objects.append(found)
        else:
            print(f"   ⚠️ 경고: '{name}'에 해당하는 장소 객체를 찾을 수 없습니다.")

    # 7. State 업데이트
    # selected_main_places에 '순서대로 정렬된 실제 객체 리스트'를 넣습니다.
    # main.py의 create_map_html(is_route=True)가 이걸 보고 선을 그립니다.
    
    print(f"   ✅ 최종 경로 확정: {len(final_route_objects)}개 장소")
    
    # 기존 값을 덮어씁니다 (Agent 5의 결과가 최종 권위)
    state["selected_main_places"] = final_route_objects
    state['routes_text'] = result.routes_text
    # 설명 텍스트는 별도 필드나 messages에 저장 가능하지만, 여기선 로그로만 확인
    # (필요하다면 state에 'final_itinerary_text' 같은 필드 추가)
    
    return {
        "selected_main_places": final_route_objects,
        # "messages": [AIMessage(content=result.routes_text)] # 필요시 주석 해제
    }