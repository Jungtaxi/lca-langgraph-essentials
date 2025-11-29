from state import AgentState, CandidatePlace
from tools import search_kakao
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import json

class Satisfied(BaseModel):
    satisfy: bool

def collector_node(state: AgentState):
    print("\n🏃 --- [Agent 3] 메인 장소 후보군(Pool) 대량 수집 중 ---")
    
    strategy = state['strategy']
    preferences = state.get('preferences')
    if not strategy or not preferences: return {}

    days = preferences.duration

    print(f"   📅 여행 기간: {days}일")

    final_candidates = []
    seen_ids = set()
    ANCHOR_WEIGHT_THRESHOLD = 7

    # 1. 가중치 높은 순으로 정렬
    allocations = sorted(
        strategy.allocations,
        key=lambda x: x.weight,
        reverse=True
    )
    
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
    structured_llm = llm.with_structured_output(Satisfied)
    for alloc in allocations:
        tag_name = alloc.tag_name
        weight = alloc.weight
        target_count = alloc.count
        keywords = alloc.keywords

        # 2. 개수가 0이면 패스
        if target_count <= 0:
            continue

        # 3. 목표 수집 개수 설정 (days * count)
        search_limit = target_count
        if search_limit > 15: search_limit = 15

        print(f"   🔎 [Collect] '{tag_name}' (Weight {weight}) | 키워드당 {search_limit}개 검색 시작...")
        for kw in keywords:
            places = search_kakao(kw, search_limit)
            print(kw)
            print(places)
            for p in places:
                print(p)

                system_prompt = f"""
                당신은 검색 결과 검증기입니다.
                사용자가 입력한 **'검색 키워드'**와 API가 반환한 **'장소 정보'**가 논리적으로 일치하는지 O/X로 판단하세요.
                [기준 테마]: {tag_name}
                [기준 키워드]: {kw}
                
                [검색된 장소]
                - 이름: {p['place_name']}
                - 카테고리: {p['category_name']}
                
                [판단 기준]
                1. **카테고리 일치**: 키워드가 '맛집/식당'인데 '편의점', 'PC방', '재료상'이면 False.
                2. **지역 일치**: 키워드에 포함된 지역명(예: 종로)과 장소 위치가 터무니없이 다르면 False.
                3. **폐업/부적합**: 이름에 '폐업', '이전' 등이 포함되어 있으면 False.
                
                일치하면 true, 아니면 false를 반환하세요.
                """
                message = [SystemMessage(content=system_prompt)]

                result = structured_llm.invoke(message)
                if not result.satisfy:
                    print(f"   ⚠️ 장소 '{p['place_name']}'는 기준 미달로 스킵됨.")
                    continue
                
                pid = p['id']
                
                # 중복 제거 로직 (이미 수집한 장소면 스킵)
                if pid in seen_ids: 
                    continue

                seen_ids.add(pid)
                
                place_obj = CandidatePlace(
                    place_name=p['place_name'],
                    address=p['road_address_name'] or p['address_name'],
                    category=p['category_name'],
                    tag_name=tag_name,
                    place_url=p['place_url'],
                    x=float(p['x']),
                    y=float(p['y']),
                    weight=weight,
                    keyword=kw
                )
                final_candidates.append(place_obj)
            
    print(f"✅ 총 {len(final_candidates)}개의 유니크한 장소 후보(Pool) 수집 완료.")

    return {"candidates": final_candidates}