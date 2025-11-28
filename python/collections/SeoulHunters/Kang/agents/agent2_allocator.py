import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from state import AgentState, ItineraryStrategy

def allocator_node(state: AgentState):
   print("\n⚖️ --- [Agent 2] Kakao 코드 기반 할당 전략 수립 중 ---")

   # Agent 1의 결과 (TripPreferences)
   preferences = state['preferences']
   target_area = preferences.target_area
   duration = preferences.duration
   # 수정사항
   # 지금은 모든 node에서 llm을 ? 하고 있는데, 이걸 global로 변경할 필요는 있다.
   
   llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
   structured_llm = llm.with_structured_output(ItineraryStrategy)

   system_prompt = f"""
   당신은 치밀한 여행 전략가입니다. 
   Agent 1이 분석한 **여행 지역({target_area})**과 사용자 선호도를 바탕으로, 우선순위(weight)와 수량(count)이 포함된 Kakao Map API 카테고리 코드를 활용한 검색 전략을 수립하세요.
   
   [사용자 선호도]
   {json.dumps(preferences.model_dump(), indent=2, ensure_ascii=False)}
   
   [중요: 검색 타겟 지역]
    **{target_area}**
    
    [전략 수립 로직 (Algorithm)]
    
    1. **수량(Count) 산정**: 여행 기간({duration})에 비례하여 결정하세요.
       - 당일치기: 총 3~5곳 (식당1, 카페1, 관광1 등)
       - 1박 2일: 총 6~8곳 (식당3, 카페2, 관광2, 숙소1 등)
       - 2박 3일: 총 9~12곳
       
    2. **가중치(Weight) 부여 (핵심)**:
       - **Anchor (Weight 8~10)**: 여행 테마(themes)와 가장 밀접한 카테고리.
         (예: '맛집 투어' -> FD6=10, '힐링' -> CE7=10, '관광' -> AT4=10)
         (예: 숙박 여행인 경우 AD5(숙소)는 동선의 기준이 되므로 Weight 9 이상 부여)
       - **Satellite (Weight 1~5)**: 메인 장소 근처에서 가볍게 들를 곳.
         (예: 맛집 갔다가 갈 카페, 관광지 근처 편의점/주차장)

   [Kakao Category Code 매핑 규칙 (여행 관점)]
   
   1. **핵심 여행지 (Main Spots)**
      - **FD6 (음식점)**: '맛집' 테마 필수. 모든 여행의 기본 식사 장소.
      - **CE7 (카페)**: '힐링', '트렌드', '카페' 테마 필수. 휴식이 필요할 때 할당.
      - **AT4 (관광명소)**: '관광', '야경', '체험', '뷰' 테마일 때 할당. (가장 우선순위 높음)
      - **CT1 (문화시설)**: '문화/예술' 테마일 때 할당 (박물관, 미술관, 공연장).
      - **AD5 (숙박)**: 기간({duration})이 당일치기가 아닌 경우(1박 이상) 필수 할당.
      
   2. **쇼핑 및 편의 (Shopping & Convenience)**
      - **MT1 (대형마트)**: '쇼핑' 테마 중에서도 식료품/기념품 구매 혹은 펜션 여행 장보기 필요 시.
      - **CS2 (편의점)**: 구체적인 할당보다는, 숙소 근처나 한강 공원 등에서 '간식' 키워드가 있을 때 보조적으로 사용.
      
   3. **교통 및 이동 (Transport)**
      - **PK6 (주차장)**: 이동수단이 **'차량'**일 때, 주요 관광지(AT4)나 식당(FD6) 근처 주차장 검색용으로 추가 할당.
      - **OL7 (주유소/충전소)**: 이동수단이 **'차량'**이고 장거리 이동 시 고려.
      - **SW8 (지하철역)**: 이동수단이 **'대중교통'**일 때, 여행의 시작점이나 만남의 장소(Anchor)로 활용.
   
   [할당 전략 (Logic)]
   1. **테마 매칭**: 사용자의 `themes` 리스트를 보고 위 코드를 조합하세요. (예: '힐링+맛집' -> CE7 + FD6 집중)
   2. **강도 조절**:
      - 강도 높음(>70): AT4(관광), CT1(문화) 개수 증가 (많이 걷고 구경).
      - 강도 낮음(<30): CE7(카페), FD6(식당) 위주로 동선 최소화.
   3. **키워드 생성**:
      - `keywords` 작성 시 단순 '맛집'이 아니라 `additional_notes`의 맥락을 반영해 구체적으로 적으세요.
      - 한가지 카테고리만 사용해서 작성하세요.
   """
   message = [SystemMessage(content=system_prompt)]

   result = structured_llm.invoke(message)
   print(result)
   return {"strategy": result}