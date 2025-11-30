from typing import List, Set
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import AgentState, CandidatePlace
# [수정] search_kakao 대신 search_local_places import
from tools import search_local_places 

# 검증용 출력 스키마
class Satisfied(BaseModel):
    satisfy: bool = Field(description="조건 충족 여부 (True/False)")

def collector_node_naver(state: AgentState):
    print("\n🏃 --- [Agent 3]장소 수집 및 검증중 NAVER ---")
    
    strategy = state.get('strategy')
    preferences = state.get('preferences')
    
    if not strategy or not preferences:
        print("🚨 전략(Strategy) 또는 선호도(Preferences)가 없습니다.")
        return {}

    # 1. LLM 초기화 (검증용)
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    structured_llm = llm.with_structured_output(Satisfied)

    final_candidates: List[CandidatePlace] = []
    seen_ids: Set[str] = set()
    
    # 2. 가중치 높은 순으로 정렬
    allocations = sorted(
        strategy.allocations, 
        key=lambda x: x.weight, 
        reverse=True
    )

    for alloc in allocations:
        tag_name = alloc.tag_name
        weight = alloc.weight
        target_count = alloc.count
        keywords = alloc.keywords

        if target_count <= 0: continue

        # 검색 한도 (API 페이징 고려 최대 15개)
        search_limit = min(15, target_count)
        
        print(f"   🔎 [Collect] '{tag_name}' (W:{weight}) | 키워드: {keywords[0]} 등... (목표 {search_limit}개)")

        for kw in keywords:
            # [수정] search_local_places 함수 사용
            # (tools.py에 정의된 함수 시그니처에 맞춰 호출)
            places = search_local_places(kw, search_limit)
            # print("==== DEBUG ====")
            # print(tag_name)
            # print("==== KEYWORD ====")
            # print(kw)
            # print("==== PLACES ====")
            # print(places)
            # print(len(places))
            # print("==== END PLACES ====")
            for p in places:
                # print("---- PLACE ----")
                # print(p)
                # API 결과 키값 매핑 (search_local_places의 리턴 형태에 맞춰 조정 필요)
                # 여기서는 Kakao API 표준 키('id', 'place_name' 등)를 가정합니다.
                pid = p.get('title') 
                
                if pid in seen_ids: continue

                # --- [LLM 검증 단계] ---
                system_prompt = f"""
                당신은 검색 결과 검증기입니다.
                사용자가 입력한 **'검색 키워드'**와 API가 반환한 **'장소 정보'**가 논리적으로 일치하는지 O/X로 판단하세요.
                [기준 테마]: {tag_name}
                [기준 키워드]: {kw}
                
                [검색된 장소]
                - 이름: {p.get('title')}
                - 카테고리: {p.get('category')}
                
                [판단 기준]
                1. **카테고리 일치**: 키워드가 '맛집/식당'인데 '편의점', 'PC방', '재료상'이면 False.
                2. **지역 일치**: 키워드에 포함된 지역명(예: 종로)과 장소 위치가 터무니없이 다르면 False.
                3. **폐업/부적합**: 이름에 '폐업', '이전' 등이 포함되어 있으면 False.
                
                적합하면 true, 아니면 false를 반환하세요.
                """
                # print(p.get('title'))
                # print(p.get('category'))
                try:
                    validation = structured_llm.invoke([SystemMessage(content=system_prompt)])
                    # print(validation.satisfy)
                    if not validation.satisfy:
                        continue
                except Exception as e:
                    print(f"      ⚠️ [Error] 검증 중 오류: {e}")
                    # 에러 시 안전하게 통과 또는 스킵 (여기선 통과)

                # --- [수집 성공] ---
                seen_ids.add(pid)
                
                # CandidatePlace 매핑
                place_obj = CandidatePlace(
                    place_name=p.get('title'),
                    address=p.get('address'),
                    category=p.get('category'),
                    tag_name=tag_name,
                    place_url=p.get('link'),
                    x=float(p.get('mapx', 0))/10000000,
                    y=float(p.get('mapy', 0))/10000000,
                    weight=weight,
                    keyword=kw
                )
                final_candidates.append(place_obj)

    print(f"✅ 총 {len(final_candidates)}개의 장소 후보 수집 완료. - NAVER")
    
    return {"candidates": final_candidates}