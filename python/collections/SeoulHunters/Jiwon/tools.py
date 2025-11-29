import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수를 먼저 설정하세요.")


def clean_html(raw: str) -> str:
    """응답에 섞여있는 <b>태그 같은 HTML 태그 제거"""
    return re.sub(r"<.*?>", "", raw)


def search_local_places(query: str, display: int = 5, start: int = 1, sort: str = "random"):
    """
    네이버 지역 검색 API 호출 함수
    - query: 검색어 (예: '정자역 카페', '판교 맛집')
    - display: 한 번에 가져올 개수 (공식 문서상 최대 5개) 
    - start: 시작 위치
    - sort: 'random' (기본, 정확도순) / 'comment' (리뷰 많은 순)
    """
    url = "https://openapi.naver.com/v1/search/local.json"

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=5)
    resp.raise_for_status()  # 4xx, 5xx 에러 시 예외 발생

    data = resp.json()

    # 필요한 정보만 깔끔하게 정리해서 리턴
    places = []
    for item in data.get("items", []):
        places.append(
            {
                "name": clean_html(item.get("title", "")),
                "category": item.get("category"),
                "address": item.get("address"),
                "road_address": item.get("roadAddress"),
                "mapx": item.get("mapx"),  # 경도 (네이버 자체 좌표계)
                "mapy": item.get("mapy"),  # 위도 (네이버 자체 좌표계)
                "link": item.get("link"),  # 네이버 상세 페이지
                "telephone": item.get("telephone"),
                "raw": item,  # 필요하면 전체 원본도 같이
            }
        )
    return places


# if __name__ == "__main__":
#     query = "종로 유적지"
#     results = search_local_places(query, display=5, sort="random") # sort='comment' -> 리뷰순, random -> 정확도

#     print(f"검색어: {query}")
#     for i, p in enumerate(results, start=1):
#         print(f"\n[{i}] {p['name']}")
#         print(f"   카테고리  : {p['category']}")
#         print(f"   지번주소  : {p['address']}")
#         print(f"   도로명주소: {p['road_address']}")
#         print(f"   전화번호  : {p['telephone']}")
#         print(f"   링크      : {p['link']}")
#         print(f"   좌표(mapx, mapy): {p['mapx']}, {p['mapy']}")
