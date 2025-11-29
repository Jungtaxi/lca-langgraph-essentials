import os
import requests
from dotenv import load_dotenv

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
        print(resp.json().get('documents', []))
        resp.raise_for_status()
        return resp.json().get('documents', [])
    except Exception as e:
        print(f"   ❌ API Error: {e}")
        return []