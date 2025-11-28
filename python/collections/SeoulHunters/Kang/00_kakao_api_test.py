import requests
import json

# -----------------------------------------------------------
# [준비물] 여기에 발급받은 'REST API 키'를 넣어주세요
# -----------------------------------------------------------
api_key = "my_api_key"
import requests

searching = '정자역 카페 러브멜로딩'

url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query={}'.format(searching)
headers = {
    "Authorization": "KakaoAK " + api_key
}
places = requests.get(url, headers = headers).json()['documents']
print(json.dumps(places, indent=4, ensure_ascii=False))