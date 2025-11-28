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
        description="여행 기간. 예: '1', '2', '3' 등 체류할 기간을 숫자로 명시한다. 1박 2일이면 2일을 체류하기 때문에 2를 저장한다. 몇 박 묵을지는 궁금하지 않다. 명시되지 않으면 1."
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
        1. **명사형 키워드**로 작성할 것. (예: '맛집 탐방', '빵지순례', '호캉스', '역사 투어', '아이돌 덕질')
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
    transport: Optional[Literal['대중교통', '도보', '차량']] = Field(
        None,
        description="이동 수단 선호. 사용자의 입력에서 유추할 수 있는 **모든** 테마를 전부 선택하세요. 하나만 선택하지 말고, 해당되는 것은 다 포함시키세요. 정보가 없다면 None"
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
        description="사용자가 입력한 언어(한/영/일/중 등)를 감지하세요. 그리고 해당 언어를 영어로 작성합니다. 예: Korean"
    )

class CategoryAllocation(BaseModel):
    category_group_code: Literal[
        'FD6', 'CE7', 'CT1', 'AT4', 'AD5', # 핵심 (식음료, 관광, 숙박)
        'PK6', 'OL7', 'SW8',               # 교통 (주차, 주유, 지하철)
        'MT1', 'CS2'                       # 편의 (마트, 편의점)
        # HP8(병원), PM9(약국) 등은 특수 상황 아니면 제외 (필요시 추가 가능)
    ] = Field(
            description = "Kakao Map API의 카테고리 그룹 코드(MT1:대형마트, CS2:편의점, PS3:어린이집, 유치원, SC4:학교, AC5:학원, PK6:주차장, OL7:주유소, 충전소, SW8:지하철역, BK9:은행, CT1:문화시설, AG2:중개업소, PO3:공공기관, AT4:관광명소, AD5:숙박, FD6:음식점, CE7:카페, HP8:병원, PM9:약국)"
    )
    
    keywords: List[str] = Field(
        description="인터넷 검색할 때 필요한 구체적인 키워드 리스트. Agent 1이 정한 지역명을 포함해서 작성한다."
    )

    count: int = Field(description="필요한 장소의 개수. 하루에 방문할 만한 최대 횟수를 기록한다. (여행 기간에 비례)") # 수정

    weight: int = Field(description="검색 우선순위 가중치 (1~10). 점수가 높을수록 '여행의 메인 목적'이며, 먼저 검색되어 기준점(Anchor)이 됩니다.") # 수정

    reason: str = Field(
        description="해당 category_group_code들을 선택한 이유"
    )

class ItineraryStrategy(BaseModel):
    allocations: List[CategoryAllocation] = Field(
        description="카테고리 코드별 할당 리스트"
    )

class CandidatePlace(BaseModel):
    place_name: str
    address: str
    category: str      # 카테고리 이름 (예: 음식점 > 한식)
    code: str          # 카테고리 코드 (FD6 등)
    place_url: str
    x: float           # 경도 (Longitude) - 계산을 위해 float 변환
    y: float           # 위도 (Latitude)
    weight: int        # 중요도 (Agent 2에서 받음)
    keyword: str       # 검색 키워드

class AgentState(TypedDict):
    # 1. 대화 기록 (공통)
    messages: Annotated[List[BaseMessage], operator.add]

    # 2. Agent 1의 결과물 (User Preferences)
    preferences: Optional[TripPreferences]

    # 3. Agent 2의 결과물 (Search Strategy)
    strategy: Optional[ItineraryStrategy]
    
    # 4. Agent 3의 결과물 (Pool)
    candidates : Optional[CandidatePlace]