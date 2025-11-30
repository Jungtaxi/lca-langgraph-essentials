import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json 
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, END, START, MessagesState

# Agent가 채워 넣어야 할 데이터 구조 정의
# --- 1. Agent 1 데이터 스키마 (TripPreferences) ---
class TripPreferences(BaseModel):
    # 1. 기간
    duration: int = Field(
        1,
        description="체류 일자. 1박 2일이면 2일을 체류하기 때문에 2를 저장한다. 예: 당일치기: 1, 1박 2일: 2, 2박: 3 등 체류할 기간을 숫자로 명시한다."
    )

    # 2. 여행 지역 (여기에 추가!)
    target_area: Optional[str] = Field(
        default=None,
        description="구체적인 서울의 여행 지역. 예: '종로구', '성수동', '강남', '홍대', '잠실'. 명시되지 않으면 None."
    )

    # 3. 테마 (여러 개일 수 있으니 List로)
    themes: List[str] = Field(
        default_factory=list, 
        description="""
        사용자의 입력에서 유추할 수 있는 **여행 테마 키워드 리스트**.
        정해진 카테고리에 얽매이지 말고, 구체적이고 다양한 테마를 자유롭게 추출하세요.
        
        [작성 규칙]
        1. **명사형 키워드**로 작성할 것. (예: '미식', '쇼핑', '호캉스', '문화', '야경', '힐링', '액티비티', '트렌드', '데이트', '가족여행', 등)
        2. 사용자의 의도를 반영하는 키워드를 **최소 2개 이상** 풍부하게 뽑을 것.
        3. 예: "조용히 책 읽고 싶어" -> ['독서', '힐링', '조용한 카페']
        """
    )

    # 4. 강도 (숫자로 변환)
    intensity: int = Field(
        50,
        description="여행 강도 (0~100). 0은 완전한 휴식, 100은 빡빡한 일정. 강도를 단순히 0, 50, 100으로 나누지 말고, 사용자의 입력을 분석한 뒤 적절한 값으로 반환합니다. 정보가 없으면 50"
    )
    
    # 5. 동행자
    companions: Optional[Literal['혼자', '친구', '연인', '가족(아이)', '가족(부모님)']] = Field(
        None,
        description="동행자 유형. 예: '혼자', '친구', '연인', '가족(아이)', '가족(부모님)'. 정보가 없으면 None"
    )
    
    # 6. 이동 방식
    transport: Optional[Literal['대중교통', '걷기', '자차', '택시']] = Field(
        None,
        description="선호하는 이동수단. 사용자의 입력에서 유추할 수 있는 이동수단을 선택하세요."
    )

    # 7. HITL 필수 필드 (대화 제어용)
    is_complete: bool = Field(
        False,
        description="위 6가지 필수 정보(duration, themes, intensity, companions, transport, target_area)가 모두 채워졌는지 여부."
    )
    
    # 8. 부족한 부분을 채우기 위한 질문지
    missing_info_question: Optional[str] = Field(
        None, 
        description="정보가 부족할 경우 사용자에게 던질 질문. (부족한 항목을 콕 집어서 질문)"
    )

    # 9. 기타 (혹시 모를 추가 요청사항)
    additional_notes: Optional[str] = Field(
        None,
        description="""
        사용자의 여행 계획을 요약한 '종합 브리핑'입니다. 다음 두 가지 내용을 자연스러운 줄글로 통합하여 작성하세요.
        1. [요약]: 추출된 5가지 필수 정보(누구와, 기간, 테마, 강도, 이동수단)를 포함하여 여행의 전체 그림을 서술.
        2. [맥락]: 위 5가지 카테고리에 딱 들어맞지 않는 구체적인 요구사항이나 제약조건(예: '다리가 아파요', '매운 음식 못 먹어요', '특정 유튜버 맛집')도 빠짐없이 포함.
        ※ 정보가 부족하다면, 현재까지 파악된 내용만으로 요약합니다.
        """
    )

    # 10. 사용자의 사용 언어
    language: str = Field(
        None,
        description="사용자가 입력한 언어(한/영/일/중 등)를 감지하세요. 그리고 해당 언어를 영어로 작성합니다.(예: Korean, English, Japanese, Chinese)"
    )

# --- 2. Agent 2 데이터 스키마 (TripPreferences) ---
class CategoryAllocation(BaseModel):
    tag_name: str = Field(
        description="테마에 맞는 태그 (예: 맛집, 카페, 관광지, 옷가게, 신발가게 등)"
    )
    
    keywords: List[str] = Field(
        description="""
        검색 정확도를 높이기 위한 구체적인 키워드 리스트.
        
        [필수 규칙]
        1. **다양성 확보**: 단순히 '맛집' 하나만 쓰지 말고, 구체적인 메뉴나 업종으로 확장해서 **최소 3개** 작성하세요.
           - (Bad) ['연남동 맛집']
           - (Good) ['연남동 맛집', '연남동 일식', '연남동 브런치']
        2. **쇼핑 예시**: ['연남동 쇼핑', '연남동 옷가게', '연남동 빈티지샵'] (O)
        3. **형식**: 무조건 **"{지역명} {명사}"** 형태 유지.
        """
    )

    count: int = Field(description="count는 일정 일수(duration) + 강도(intensity) 기반으로 합리적인 정수.") # 수정

    weight: float = Field(description="검색 우선순위 가중치 (0.0 ~ 1.0). 점수가 높을수록 '여행의 메인 목적'이며, 먼저 검색되어 기준점(Anchor)이 됩니다. 전체 합은 약 1.0이 되어야 함.") # 수정

    reason: str = Field(
        description="해당 category_group_code들을 선택한 이유"
    )

class ItineraryStrategy(BaseModel):
    allocations: List[CategoryAllocation] = Field(
        description="태그별 할당 리스트"
    )

# --- 3. Agent 3 데이터 스키마 (TripPreferences) ---
class CandidatePlace(BaseModel):
    place_name: str = Field(description="장소 이름")
    address: str = Field(description="주소")
    category: str = Field(description="카테고리 이름 (예: 음식점 > 한식)")     
    tag_name: str =Field(description="테마 이름 (예: 맛집, 카페, 관광지 등)")    
    place_url: str =Field(description="장소의 홈페이지 주소")
    x: float   =Field(description="경도 (Longitude) - 계산을 위해 float 변환")      
    y: float    =Field(description="위도 (Latitude)")      
    weight: float=Field(description="중요도 (Agent 2에서 받음)")       
    keyword: str  =Field(description="검색 키워드")     

# [NEW] 개별 장소 스케줄 (방문 순서 포함)
class ScheduledPlace(BaseModel):
    place: CandidatePlace = Field(description="장소 정보 객체")
    order: int = Field(description="그 날의 방문 순서 (1, 2, 3...)")
    visit_time: str = Field(description="추천 방문 시간대 (예: 점심, 오후 2시)")
    description: str = Field(description="이 장소를 이 시간에 배치한 이유")

# [NEW] 하루 일정
class DaySchedule(BaseModel):
    day: int = Field(description="일차 (1, 2, 3...)")
    places: List[ScheduledPlace] = Field(description="해당 날짜에 방문할 장소 리스트 (순서대로)")
    daily_theme: str = Field(description="그 날의 테마 한 줄 요약")

# [NEW] 전체 일정 (Agent 5의 최종 결과물)
class FinalItinerary(BaseModel):
    total_days: int = Field(description="총 여행 일수")
    schedule: List[DaySchedule] = Field(description="일자별 스케줄 리스트")
    overall_review: str = Field(description="전체 여행 코스에 대한 총평")
    
class AgentState(TypedDict):

    next_step: Literal["planner", "suggester", "path_finder", "general_chat"]

    # 1. 대화 기록 (공통)
    messages: Annotated[List[BaseMessage], operator.add]

    # 2. Agent 1의 결과물 (User Preferences)
    preferences: Optional[TripPreferences]

    # 3. Agent 2의 결과물 (Search Strategy)
    strategy: Optional[ItineraryStrategy]   # tag_plan 역할
    
    # 4. Agent 3의 결과물 (Pool)
    candidates : Annotated[List[CandidatePlace], operator.add]

    # 5. Agent 4의 결과물 (Top-3 Candidates)
    main_place_candidates: Optional[List[CandidatePlace]]
    
    # 6. Agent 5의 결과물 (Route Locations)
    # selected_main_places: Optional[List[CandidatePlace]] # <-- 이거 대신 아래꺼 사용
    final_itinerary: Optional[FinalItinerary] # [NEW] 일자별로 구조화된 최종 일정
    
    # 7. 텍스트 설명 (유지하거나 final_itinerary 내부로 통합 가능)
    routes_text: str