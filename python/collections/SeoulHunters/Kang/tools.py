import os
import requests
from dotenv import load_dotenv
import re
import html

load_dotenv()

def search_kakao(query, n, sort_type='accuracy', x=None, y=None):
    api_key = os.environ.get("KAKAO_REST_API_KEY")
    if not api_key:
        print("🚨 Error: KAKAO_REST_API_KEY 환경변수가 없습니다.")
        return []
    
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "query": query,
        "size": n,
        "sort": sort_type
    }
    # 거리순 정렬일 경우 중심 좌표 필수
    if sort_type == 'distance' and x and y:
        params['x'] = x
        params['y'] = y
        params['radius'] = 2000 # 반경 2km 이내 (도보/차량 고려)

    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json().get('documents', [])
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return []

def clean_html(text):
    """문자열에서 HTML 태그 제거 및 엔티티(&amp; 등) 변환"""
    if not isinstance(text, str): # 문자열이 아니면(숫자 등) 그냥 반환
        return text
    
    # 태그 제거 (<...>)
    clean_text = re.sub(r"<.*?>", "", text)
    # HTML 엔티티 복원 (예: &amp; -> &)
    clean_text = html.unescape(clean_text)
    
    return clean_text

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
        "X-Naver-Client-Id": os.environ.get("NAVER_CLIENT_ID"),
        "X-Naver-Client-Secret": os.environ.get("NAVER_CLIENT_SECRET"),
    }

    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=5)
        resp.raise_for_status()  # 4xx, 5xx 에러 시 예외 발생
        data = resp.json()
        items = data.get("items", [])
        # 2. 리스트를 돌면서 전처리를 수행합니다.
        cleaned_items = []
        for item in items:
            # 딕셔너리의 모든 값(Value)에 대해 clean_html 적용
            new_item = {}
            for key, value in item.items():
                new_item[key] = clean_html(value)
            cleaned_items.append(new_item)
            
        return cleaned_items
    
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return []