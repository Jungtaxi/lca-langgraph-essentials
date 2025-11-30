import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from state import AgentState, ItineraryStrategy

def allocator_node(state: AgentState):
   print("\n⚖️ --- [Agent 2] 장소 할당 전략 수립 중 ---")

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
   1. **수량(Count) 산정**: count는 여행 기간({preferences.duration}) + 강도({preferences.intensity}) 기반으로 합리적인 정수.
       
   2. **가중치(Weight) 배분 (Sum = 1.0)**:
      모든 카테고리의 weight 합계가 **정확히 1.0**에 가깝도록 비중을 나누세요.
      
      - **Anchor**: 여행의 주 목적.
      - **Satellite**: 보조/편의 시설.
      
      => 합계: 1.0

   [키워드 생성 규칙 (Strict SEO Rules)]
   API 검색 정확도를 위해 아래 규칙을 위반하면 안 됩니다.
   
   1. **지역명 필수 결합**: 모든 키워드 맨 앞에는 반드시 **"{target_area}"** (또는 구체적인 동네 이름)가 와야 합니다.
   2. **형용사/서술어 금지**: '유명한', '현지인 추천', '즐기기 좋은', '분위기 있는' 등 수식어 절대 금지.
   3. **형태**: **"[지역명] + [단순명사]"** 형태의 복합명사만 허용.
      - (Bad) "친구랑 가기 좋은 저녁" -> (Good) "**{target_area} 저녁**" 또는 "**{target_area} 맛집**"
      - (Bad) "{target_area} 힙한 감성 카페" -> (Good) "**{target_area} 카페**"

   [키워드 생성 전략 (Rich Keywords)]
    **단 하나의 대표 키워드만 쓰는 것을 금지합니다.** 카테고리를 세분화하여 다양한 검색어를 생성하세요.
    
    1. **맛집**: '맛집'만 쓰지 말고, 구체적인 메뉴(한식/양식/일식/카페)나 분위기(이자카야/펍)로 확장하세요.
       - 예: ["{target_area} 맛집", "{target_area} 파스타", "{target_area} 흑돼지", "{target_area} 밥집"]
    
    2. **쇼핑**: 무엇을 살 것인가? (옷/신발/소품/기념품)
       - 예: ["{target_area} 쇼핑", "{target_area} 편집샵", "{target_area} 소품샵", "{target_area} 옷가게"]
       
    3. **카페**: 디저트/베이커리/뷰 등으로 세분화.
       - 예: ["{target_area} 카페", "{target_area} 디저트", "{target_area} 베이커리", "{target_area} 루프탑"]
       
    4. **개수**: 각 카테고리당 **최소 3개 이상의 연관 검색어**를 포함해야 합니다.
   """
   message = [SystemMessage(content=system_prompt)]

   result = structured_llm.invoke(message)
   print(result)
   return {"strategy": result,
           "candidates": []
   }