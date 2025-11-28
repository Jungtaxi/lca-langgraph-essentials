from state import AgentState, CandidatePlace
from tools import search_kakao

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

    for alloc in allocations:
        code = alloc.category_group_code
        weight = alloc.weight
        target_count = alloc.count
        keywords = alloc.keywords

        # 1. Satellite(보조 장소) 패스
        if weight < ANCHOR_WEIGHT_THRESHOLD:
            print(f"   ⏭️  [Pass] Weight {weight} (Satellite) -> 루트 확정 후 검색")
            continue

        # 2. 개수가 0이면 패스
        if target_count <= 0:
            continue

        # 3. 목표 수집 개수 설정 (days * count)
        search_limit = days * target_count
        if search_limit > 15: search_limit = 15
        # 하드코딩 수정
        search_limit = 15

        print(f"   🔎 [Collect] '{code}' (Weight {weight}) | 키워드당 {search_limit}개 검색 시작...")
        for kw in keywords:
            places = search_kakao(kw, search_limit, code)
            
            if places:
                print(kw)
                print(places)
                for p in places:
                    pid = p['id']
                    
                    # 중복 제거 로직 (이미 수집한 장소면 스킵)
                    if pid in seen_ids: 
                        continue

                    seen_ids.add(pid)
                    
                    place_obj = CandidatePlace(
                        place_name=p['place_name'],
                        address=p['road_address_name'] or p['address_name'],
                        category=p['category_name'],
                        code=code,
                        place_url=p['place_url'],
                        x=float(p['x']),
                        y=float(p['y']),
                        weight=weight,
                        keyword=kw
                    )
                    final_candidates.append(place_obj)
            else:
                places = search_kakao(kw, search_limit)
                print(kw)
                print(places)
                for p in places:
                    pid = p['id']
                    
                    # 중복 제거 로직 (이미 수집한 장소면 스킵)
                    if pid in seen_ids: 
                        continue

                    seen_ids.add(pid)
                    
                    place_obj = CandidatePlace(
                        place_name=p['place_name'],
                        address=p['road_address_name'] or p['address_name'],
                        category=p['category_name'],
                        code=code,
                        place_url=p['place_url'],
                        x=float(p['x']),
                        y=float(p['y']),
                        weight=weight,
                        keyword=kw
                    )
                    final_candidates.append(place_obj)
                    
    print(f"✅ 총 {len(final_candidates)}개의 유니크한 장소 후보(Pool) 수집 완료.")

    return {"candidates": final_candidates}