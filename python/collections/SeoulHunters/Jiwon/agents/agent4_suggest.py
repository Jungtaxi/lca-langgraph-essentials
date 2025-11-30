import json
import re
from typing import Any, Dict, List
from state import AgentState, TravelPreference, Place
from openai import OpenAI

client = OpenAI()

def _place_dict(p: Any) -> Dict:
    """Place / Pydantic 모델 / dict 등을 전부 dict로 통일"""
    if hasattr(p, "model_dump"):
        return p.model_dump()
    if isinstance(p, dict):
        return p
    return dict(p)

def _extract_json_from_output(text: str) -> str:
    """
    LLM이 ```json ... ``` 형식으로 답하거나,
    앞뒤에 설명이 조금 섞여 있어도 JSON 본문만 뽑아내는 헬퍼.
    """
    text = text.strip()

    # 코드블럭(````, ```json) 제거
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 가장 바깥 { ... } 부분만 추출
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return text


# 1단계: 후보 추천 + 안내 문구 생성용 시스템 프롬프트
SYSTEM_PROMPT_AGENT4_RECOMMEND = """
당신은 여행 일정 플래너의 "메인 장소 추천 에이전트(Agent4-추천단계)"입니다.

## 역할
- 사용자의 여행 취향(prefs), 테마별 방문 계획(tag_plan), 후보 장소(place_pool)를 보고
  "메인으로 잡을 만한 장소" 후보를 최대 3개 선택합니다.
- 그리고 이 후보들을 사용자에게 보여주고, 어떻게 선택하면 될지 안내하는 한국어 문장을 생성합니다.
- 최종 출력은 반드시 JSON 형식 하나만이어야 합니다.

## 입력 정보

입력 JSON은 다음과 같은 구조입니다:
{
  "prefs": { ... },
  "tag_plan": [ ... ],
  "place_pool": [ ... ],
  "prev_main_place_candidates": [ ... ]  // 있을 수도, 없을 수도 있음
}

- prefs: 사용자의 여행 취향 (themes, must_avoid 등 포함 가능)
- tag_plan: {"tag": "...", "weight": 0.4} 같은 구조의 리스트
- place_pool: 테마(theme), 이름(name), 주소(address/road_address) 등이 포함된 장소 리스트
- prev_main_place_candidates:
  - 이미 한 번 추천했던 메인 후보 리스트입니다.
  - 이 값이 비어있지 않다면, 가능한 한 이 장소들은 다시 추천하지 마세요.
  - 사용자가 이전 후보가 마음에 들지 않아 "다른 후보"를 요청한 상황입니다.

## 추천 규칙

1. 테마 선택
   - tag_plan에서 weight가 높은 순으로 정렬하여 상위 2개 테마를 우선 고려합니다.
   - tag 혹은 theme 필드에 적힌 문자열을 테마 이름으로 사용합니다.
   - tag_plan이 비어있거나 유효하지 않다면 prefs.themes 상위 2개를 사용합니다.
   - 이것도 없다면 ["맛집", "카페"]를 기본값으로 사용합니다.

2. 후보 필터링
   - place_pool에서 위 테마 목록에 해당하는 장소만 고려합니다.
   - name + (road_address 또는 address) 조합이 같은 것은 중복으로 간주하고 하나만 남깁니다.
   - prev_main_place_candidates에 이미 포함된 장소는 가능한 한 제외합니다.
   - prefs.must_avoid가 있다면, 이에 명백히 어긋나는 장소는 가능하면 제외합니다.

3. 우선순위
   - rating이 높고, review_count가 많은 장소를 우선합니다 (해당 필드가 있을 경우).
   - 동일 테마 내에서도 분위기/특징이 조금씩 다른 장소를 섞어서 다양성을 줍니다.

4. 개수
   - 최종 main_place_candidates는 1~3개가 되도록 노력합니다.
   - 적절한 후보가 부족하면 1개만 선택해도 되고, 정말 없으면 빈 리스트를 반환할 수 있습니다.

## 사용자 안내 문구(message_for_user)

- main_place_candidates를 바탕으로, 사용자에게 보여 줄 한국어 안내 문장을 만들어야 합니다.

숫자 목록(1., 2., 3.)을 사용해 각 후보를 하나씩 보여주세요.
장소명 / 테마 / 주소 / 간단 설명을 줄바꿈을 포함해 두세 줄로 깔끔하게 정리하세요.

예시 형식:

"메인으로 고려해볼 만한 후보를 골라봤어요.
[1] 카페 OO
   · 테마: 카페
   · 주소: 서울시 마포구 ...
   · 설명: 분위기 좋은 카페입니다.
[2] 식당 XX
   · 테마: 맛집
   · 주소: 서울시 ...
   · 설명: 현지에서 유명한 맛집이에요.
마음에 드는 번호를 쉼표로 입력해 주세요. 예: 1,3
마음에 드는 곳이 없으면 '다 별로야', '그냥 알아서 골라줘'처럼 자유롭게 말해도 돼요."

- 불필요한 긴 문장 없이, 간단 명료하게 구성하세요.

## 출력 형식

아래와 같은 JSON만 출력하세요:

{
  "main_place_candidates": [
    {
      "name": "...",
      "theme": "...",
      "road_address": "...",
      "address": "...",
      "reason": "...",
      "extra": {
        "rating": 4.3,
        "review_count": 122,
        "notes": "..."
      }
    }
  ],
  "message_for_user": "여기에 사용자에게 보여줄 한국어 안내 문장 전체를 넣습니다."
}

- main_place_candidates 배열 길이는 0~3개입니다.
- road_address, address 중 하나만 있으면 나머지는 빈 문자열로 둘 수 있습니다.
- extra는 선택 필드입니다.
- prev_main_place_candidates에 없는 place_pool의 장소만 사용하도록 노력하세요.
- 설명 문장이나 다른 형식의 텍스트를 JSON 바깥에 추가하지 마세요.
"""


# 2단계: 사용자 응답 해석용 시스템 프롬프트
SYSTEM_PROMPT_AGENT4_INTERPRET = """
당신은 여행 일정 플래너의 "메인 장소 선택 해석 에이전트(Agent4-선택단계)"입니다.

## 역할
- main_place_candidates 목록과 사용자 입력(user_reply)을 보고,
  실제로 어떤 장소들을 "selected_main_places"로 선택할지 결정합니다.
- 사용자의 입력이 숫자가 아니어도, 자연어로 된 의도를 최대한 해석해야 합니다.
- 항상 JSON 형식 하나만 출력해야 합니다.

## 입력 정보

입력 JSON 예시는 다음과 같습니다:

{
  "main_place_candidates": [ ... ],  // 1단계에서 만든 후보 리스트
  "user_reply": "1,3"
}

- main_place_candidates: 각 장소는 name, theme, address 등의 필드를 가진 dict입니다.
- user_reply: 사용자가 터미널에 입력한 아무 문자열입니다.
  - 예: "1,3"
  - 예: "1번이랑 3번이요"
  - 예: "첫 번째 것만"
  - 예: "다 별로인데 그냥 진행할게요"
  - 예: "저는 알아서 골라주셔도 좋아요"

## 해석 규칙

1. 숫자/순번으로 선택하는 경우
   - user_reply 안에 "1", "2", "3" 등의 숫자 또는 "1번", "첫 번째" 등의 표현이 있으면
     해당 인덱스를 main_place_candidates에서 골라 selected_main_places에 넣습니다.
   - 인덱스는 1부터 시작한다고 가정합니다.

2. 이름으로 언급하는 경우
   - user_reply에 특정 후보의 name이 들어 있다면 해당 후보를 선택 대상으로 간주합니다.
   - 부분 일치라도 문맥상 분명하면 선택해도 됩니다.

3. "알아서 골라줘" 류의 위임
   - "알아서 골라줘", "추천에 맡길게요", "그냥 적당히 골라줘" 등
     선택을 에이전트에게 위임하는 표현이라면:
     - main_place_candidates 중에서 1~3개를 당신이 판단해서 선택하세요.
     - 이때 사용자가 선호를 언급했다면(예: "카페 위주로 골라줘") 그에 맞게 골라야 합니다.

4. "다 별로", "안 고를래" 등 명시적 거부
   - "다 별로에요", "안 고를래요", "그냥 넘어가고 싶어요"와 같이
     어떤 후보도 메인으로 선택하고 싶지 않다는 의도가 명확하다면:
     - selected_main_places를 빈 리스트([])로 반환하세요.
     - 즉, 메인 장소를 고르지 않고도 다음 단계로 진행할 수 있도록 합니다.

5. 모호한 경우
   - 위 규칙 어느 쪽도 명확하지 않다면:
     - 사용자가 특별히 거부하지 않은 선에서, 후보 중 가장 무난한 1개만 선택하거나,
     - 선택 없이 빈 리스트를 반환해도 됩니다.
   - 중요한 점은, 어떤 경우에도 JSON 형식은 항상 유효해야 합니다.

## 출력 형식

아래 형식의 JSON만 출력하세요:

{
  "selected_main_places": [
    { ... },
    { ... }
  ],
  "mode": "user_chosen" | "auto_chosen" | "skipped"
}

- selected_main_places:
  - main_place_candidates에 들어 있던 객체들 중 일부를 그대로 복사해서 넣어야 합니다.
  - 새로운 장소를 상상해서 만들어내지 마세요.
- mode:
  - "user_chosen": 사용자 입력이 명확히 특정 후보를 선택한 경우
  - "auto_chosen": "알아서 골라줘"처럼 선택을 위임한 경우
  - "skipped": "다 별로", "안 고를래요"처럼 선택을 거부한 경우
"""


def agent4_suggest_node(state: AgentState) -> AgentState:
    """
    - LLM 1번: 메인 후보 추천 + 사용자 안내 문구 생성
    - 사용자 input()
    - LLM 2번: 사용자의 자유로운 답변 해석 → selected_main_places 결정
    """
    # 1) state에서 정보 꺼내기
    prefs = state["prefs"]
    tag_plan = state.get("tag_plan") or []
    place_pool = state["place_pool"]
    prev_candidates = state.get("main_place_candidates") or []

    # TravelPreference 직렬화
    if hasattr(prefs, "model_dump"):
        prefs_payload = prefs.model_dump()
    else:
        prefs_payload = prefs

    # ---------- 1차 LLM: 후보 추천 + 안내 문구 ----------
    payload_recommend = {
        "prefs": prefs_payload,
        "tag_plan": tag_plan,
        "place_pool": [_place_dict(p) for p in place_pool],
        "prev_main_place_candidates": [
            _place_dict(p) for p in prev_candidates
        ] if prev_candidates else [],
    }

    user_content_1 = json.dumps(payload_recommend, ensure_ascii=False)

    completion1 = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_AGENT4_RECOMMEND},
            {"role": "user", "content": user_content_1},
        ],
        temperature=0,
    )

    raw_output1 = completion1.choices[0].message.content or ""
    json_str1 = _extract_json_from_output(raw_output1)

    try:
        data1 = json.loads(json_str1)
    except json.JSONDecodeError:
        data1 = {}

    main_place_candidates = data1.get("main_place_candidates", []) or []
    if not isinstance(main_place_candidates, list):
        main_place_candidates = []

    message_for_user = data1.get("message_for_user") or ""

    # state에 후보 저장 (나중에 agent5나 디버깅에서도 사용할 수 있음)
    state["main_place_candidates"] = main_place_candidates

    # 후보 안내 출력
    print("\n=== 메인 후보 장소 안내 ===")
    if message_for_user:
        print(message_for_user)
    else:
        # fallback: 아주 단순하게라도 보여주기
        if not main_place_candidates:
            print("(추천할 후보가 없습니다. 메인 장소 없이 바로 루트를 만들게요.)")
        else:
            print("메인으로 고려해볼 만한 후보입니다:")
            for i, place in enumerate(main_place_candidates, start=1):
                if hasattr(place, "model_dump"):
                    place = place.model_dump()
                name = place.get("name", "이름 없음")
                theme = place.get("theme", "")
                addr = place.get("road_address") or place.get("address") or ""
                print(f"{i}. {name} / {theme} / {addr}")
            print("번호를 쉼표로 입력하거나, 자유롭게 의견을 말씀해 주세요.")

    # 사용자 입력 받기 (자유로운 내용 가능)
    user_reply = input("\n> ").strip()

    # ---------- 2차 LLM: 사용자 응답 해석 ----------
    payload_interpret = {
        "main_place_candidates": main_place_candidates,
        "user_reply": user_reply,
    }

    user_content_2 = json.dumps(payload_interpret, ensure_ascii=False)

    completion2 = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_AGENT4_INTERPRET},
            {"role": "user", "content": user_content_2},
        ],
        temperature=0,
    )

    raw_output2 = completion2.choices[0].message.content or ""
    json_str2 = _extract_json_from_output(raw_output2)

    try:
        data2 = json.loads(json_str2)
    except json.JSONDecodeError:
        data2 = {}

    selected = data2.get("selected_main_places", []) or []
    if not isinstance(selected, list):
        selected = []

    mode = data2.get("mode", "unknown")

    state["selected_main_places"] = selected

    return state