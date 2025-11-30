import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import List

# state.py에서 정의한 클래스들 import
from state import AgentState, CandidatePlace, FinalItinerary, DaySchedule, ScheduledPlace

# --- [LLM 출력용 스키마 (이름만 받기)] ---
# CandidatePlace 객체 전체를 LLM이 뱉게 하면 망가지므로, 이름만 받아서 매핑함.
class LLMPlaceRef(BaseModel):
    place_name: str = Field(description="장소의 정확한 이름")
    visit_time: str = Field(description="방문 시간대")
    description: str = Field(description="동선 이유")

class LLMDaySchedule(BaseModel):
    day: int
    places: List[LLMPlaceRef]
    daily_theme: str

class LLMItineraryOutput(BaseModel):
    total_days: int
    schedule: List[LLMDaySchedule]
    overall_review: str


def agent5_route_node(state: AgentState) -> AgentState:
    print("\n🚗 --- [Agent 5] 일자별 상세 여행 경로 생성 ---")
    
    prefs = state["preferences"]
    place_pool = state.get("candidates") or []
    main_candidates = state.get("main_place_candidates") or []
    user_selection_msg = state["messages"][-1].content # 사용자의 선택 ("1번이랑 3번")

    # 1. 데이터 준비 (Mapping용 Dict 생성)
    combined_pool = {p.place_name: p for p in place_pool + main_candidates}
    
    # LLM에게 보여줄 텍스트
    # (메인 후보는 강조, 나머지는 풀로 제공)
    main_txt = ", ".join([f"{p.place_name}({p.category})" for p in main_candidates])
    
    pool_txt = ""
    for i, (name, p) in enumerate(list(combined_pool.items())[:50]): # 너무 많으면 자름
        pool_txt += f"- {name} ({p.category}, 키워드:{p.keyword}, 좌표:{p.y:.3f},{p.x:.3f})\n"

    # 2. 목표 일수 및 스팟 수 계산
    duration = prefs.duration # (int)
    intensity = prefs.intensity
    spots_per_day = 4 if intensity <= 30 else (5 if intensity <= 60 else 6)

    # 3. LLM 설정
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # 복잡한 작업은 gpt-4o 필수
    structured_llm = llm.with_structured_output(LLMItineraryOutput)

    # 4. 프롬프트
    system_prompt = f"""
    당신은 여행 동선 설계 전문가입니다.
    사용자의 선택과 전체 장소 풀을 조합하여 **{duration}일간의 여행 코스**를 작성하세요.

    [사용자 프로필]
    - 여행 기간: {duration}일 (반드시 Day 1 ~ Day {duration}까지 채울 것)
    - 테마: {prefs.themes}
    - 목표 스팟 수: 하루 약 {spots_per_day}곳
    - 요청사항: "{prefs.additional_notes}"

    [사용자가 선택한 후보 (필수 포함)]
    (이전 단계 제안 목록: {main_txt})
    사용자 피드백: "{user_selection_msg}"
    -> 사용자가 선택한 장소는 **반드시** 일정에 포함하고 Anchor로 삼으세요.

    [이용 가능한 전체 장소 풀 (Pool)]
    {pool_txt}

    [작성 규칙]
    1. **일자별 분배**: 장소들의 **좌표(위도, 경도)**를 고려하여, 가까운 곳끼리 같은 날짜에 묶으세요. (동선 효율화)
    2. **순서 배열**: 식사 -> 카페 -> 관광 -> 식사 등 상식적인 순서로 배치하세요.
    3. **빈자리 채우기**: 선택된 장소만으로 부족하면, '장소 풀'에서 적절한 곳을 추가하여 하루 일정을 완성하세요.
    4. **출력**: 장소 이름은 위 리스트에 있는 **정확한 이름**을 사용해야 매핑이 가능합니다.
    """

    # 5. 실행
    try:
        result = structured_llm.invoke([SystemMessage(content=system_prompt)])
    except Exception as e:
        print(f"Error in Agent 5: {e}")
        return state # 에러 시 기존 상태 반환

    # 6. [핵심] LLM 결과를 실제 객체(FinalItinerary)로 변환 (매핑)
    final_schedule = []
    
    for day_plan in result.schedule:
        daily_places = []
        for i, place_ref in enumerate(day_plan.places, 1):
            # 이름으로 실제 객체 찾기
            real_place_obj = None
            
            # 1. 완전 일치
            if place_ref.place_name in combined_pool:
                real_place_obj = combined_pool[place_ref.place_name]
            else:
                # 2. 부분 일치 (유연성)
                for db_name, db_obj in combined_pool.items():
                    if place_ref.place_name in db_name or db_name in place_ref.place_name:
                        real_place_obj = db_obj
                        break
            
            if real_place_obj:
                # 스케줄 객체 생성
                scheduled_p = ScheduledPlace(
                    place=real_place_obj,
                    order=i,
                    visit_time=place_ref.visit_time,
                    description=place_ref.description
                )
                daily_places.append(scheduled_p)
            else:
                print(f"   ⚠️ 경고: '{place_ref.place_name}' 매핑 실패")

        # 하루 일정 완성
        day_schedule = DaySchedule(
            day=day_plan.day,
            places=daily_places,
            daily_theme=day_plan.daily_theme
        )
        final_schedule.append(day_schedule)

    # 최종 결과 객체
    final_itinerary = FinalItinerary(
        total_days=result.total_days,
        schedule=final_schedule,
        overall_review=result.overall_review
    )

    print(f"   ✅ 최종 일정 생성 완료: 총 {len(final_schedule)}일, {sum(len(d.places) for d in final_schedule)}개 장소")
    
    return {
        "final_itinerary": final_itinerary,
        "routes_text": result.overall_review # 간단한 텍스트용
    }