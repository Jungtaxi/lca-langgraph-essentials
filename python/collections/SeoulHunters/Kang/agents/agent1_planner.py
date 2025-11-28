from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import json

from state import AgentState, TripPreferences
def planner_node(state: AgentState):
   print("🤖 --- [Planner Node] 사용자 의도 분석 중. . . ---")
   
   # 2. LLM 설정 (구조화된 출력)
   llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
   structured_llm = llm.with_structured_output(TripPreferences)

   # 현재 상태 가져오기
   current_pref = state.get("preferences", TripPreferences())

   # 시스템 프롬프트: "5가지가 다 모여야 탈출 가능"
   system_prompt = f"""
   당신은 한국어, 영어, 일본어, 중국어에 모두 능통한 '베테랑 여행 플래너'입니다.
   사용자의 입력과 대화 내역을 분석하여 여행 정보를 추출하세요.
   
   [현재 파악된 정보]
   {json.dumps(current_pref.model_dump(), indent=2, ensure_ascii=False)}
   
   [필수 확보 정보 (5-Check)]
   여행 계획을 짜기 위해 다음 5가지 정보는 **반드시** 값이 있어야 합니다.
   1. duration (기간)
   2. target_area (여행 지역 - 예: 종로, 성수, 강남 등)
   3. themes (테마 - 최소 1개 이상)
   4. intensity (여행 강도)
   5. companions (동행자)
   6. transport (이동 수단)

   [언어 처리 규칙 (매우 중요)]
   1. **언어 감지**: 사용자가 입력한 언어(한/영/일/중 등)를 감지하세요.
   2. **데이터 매핑 (Internal)**: 사용자가 외국어로 말하더라도, 추출하는 값은 반드시 스키마에 정의된 **'한국어 표준 값'**으로 변환하여 저장하세요.
      - 예: User "I want to go shopping" -> themes=['쇼핑'] (NOT 'Shopping')
      - 예: User "電車で行きます (전철로 갈래요)" -> transport='대중교통'
   3. **질문 생성 (External)**: `missing_info_question`은 반드시 **사용자가 입력한 언어**로 작성하세요.
      - 예: 사용자가 일본어로 말했으면, 질문도 일본어로 생성.
   
   [지시사항]
   - 위 6가지 중 하나라도 값이 없거나(None), 빈 리스트라면 'is_complete'는 False입니다.
   - 'is_complete'가 False라면, 부족한 정보가 무엇인지 파악하여 사용자에게 자연스럽게 물어보는 질문을 'missing_info_question'에 작성하세요.
   - **특히 '지역(target_area)'이 없다면, "서울의 어느 동네를 보고 싶으신가요? (예: 종로, 성수, 강남)" 하고 물어보세요.**
   - 만약 사용자가 "잘 모르겠어요"라고 하면, 테마를 보고 추천해주지 말고 "추천해드릴까요?"라고 되물어서 지역을 확정 짓도록 유도하세요.
   - **질문은 친절하고 정중한 톤을 유지하세요.**
   - 모든 정보가 채워졌을 때만 'is_complete'를 True로 설정하세요.
   """

   messages = [SystemMessage(content=system_prompt)] + state["messages"]
   
   result = structured_llm.invoke(messages)
   
   return {"preferences": result}