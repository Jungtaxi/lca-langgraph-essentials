# from agent1.agent1 import build_agent1
# from agent2.agent2 import build_agent2
# from agent3.agent3 import build_agent3
# from agent4.agent4 import build_agent4

# # Agent1 앱
# app1 = build_agent1()
# test_input = {
#     "user_input": "1박 2일로 친구랑 연남동에 가는데, 아침은 안먹고 점심은 맛집! 저녁은 술 파는 곳 가고 싶어, "
#                   "중간중간 카페도 가고 싶고, 쇼핑 위주로 다니고 싶어. 점심과 저녁 사이에는 쇼핑 무조건 2군데 이상 걸어댕길껴.",
#     "prefs": None,
# }
# result1 = app1.invoke(test_input)

# print("user_input:", test_input["user_input"])
# print("=== Agent1 Output ===")
# print(result1["prefs"])   # TravelPreference

# # Agent2 앱
# app2 = build_agent2()
# result2 = app2.invoke({"prefs": result1["prefs"]})

# print("\n=== Agent2 Tag Plan ===")
# for item in result2["tag_plan"]:
#     print(item)

# # Agent3 앱
# # Agent3
# app3 = build_agent3(per_slot=5)
# result3 = app3({
#     "prefs": result1["prefs"],
#     "tag_plan": result2["tag_plan"],
# })

# place_pool = result3["place_pool"]
# # print(place_pool)

# # 4) Agent4
# app4 = build_agent4()
# result4 = app4({
#     "prefs": result1["prefs"],
#     "place_pool": place_pool,
# })

# routes = result4["routes"]
# # print(routes)

# # === 예쁘게 출력하기 ===
# KOR_SLOT_NAME = {
#     "morning": "아침",
#     "lunch": "점심",
#     "afternoon": "오후",
#     "snack": "간식/카페",
#     "dinner": "저녁",
#     "night": "야간/야경",
# }

# print("\n=== 최종 루트 ===")
# for day_plan in routes:
#     day = day_plan.get("day")
#     print(f"\n[Day {day}]")

#     for slot in day_plan.get("schedule", []):
#         slot_name = slot.get("time", "")
#         place = slot.get("place")

#         # 시간대 한글 이름
#         time_kor = KOR_SLOT_NAME.get(slot_name, slot_name)

#         if not place:
#             print(f"  - {time_kor}: (추천 장소 없음)")
#             continue

#         # place는 dict 또는 Pydantic라서 둘 다 대응
#         if hasattr(place, "model_dump"):
#             place = place.model_dump()

#         name = place.get("name", "이름 없음")
#         theme = place.get("theme", "")
#         area = place.get("area", "")
#         category = place.get("category", "")
#         addr = place.get("road_address") or place.get("address") or ""

#         print(f"  - {time_kor}: {name}")
#         if theme or area:
#             print(f"      · 테마/지역: {theme} / {area}")
#         if category:
#             print(f"      · 카테고리: {category}")
#         if addr:
#             print(f"      · 주소: {addr}")


# main.py

from travel_graph import build_travel_graph
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    user_input = (
        "1박 2일로 친구랑 연남동에 가는데, 아침은 안먹고 점심은 맛집! 저녁은 술 파는 곳 가고 싶어, "
        "중간중간 카페도 가고 싶고, 쇼핑 위주로 다니고 싶어. 점심과 저녁 사이에는 쇼핑 무조건 2군데 이상걸어댕길껴."
    )
    travel_app = build_travel_graph()
    result = travel_app.invoke({
        "user_input": user_input,
        # prefs/tag_plan/place_pool/routes 는 그래프가 알아서 채움
    })

    prefs = result.get("prefs")
    routes = result.get("routes")

    print("user_input:", user_input)
    print("=== 최종 TravelPreference ===")
    print(prefs)

    print("\n=== 최종 루트 ===")
    KOR_SLOT_NAME = {
        "morning": "아침",
        "lunch": "점심",
        "afternoon": "오후",
        "snack": "간식/카페",
        "dinner": "저녁",
        "night": "야간/야경",
    }

    for day_plan in routes:
        day = day_plan.get("day")
        print(f"\n[Day {day}]")

        for slot in day_plan.get("schedule", []):
            slot_name = slot.get("time", "")
            place = slot.get("place")

            time_kor = KOR_SLOT_NAME.get(slot_name, slot_name)

            if not place:
                print(f"  - {time_kor}: (추천 장소 없음)")
                continue

            if hasattr(place, "model_dump"):
                place = place.model_dump()

            name = place.get("name", "이름 없음")
            theme = place.get("theme", "")
            area = place.get("area", "")
            category = place.get("category", "")
            addr = place.get("road_address") or place.get("address") or ""

            print(f"  - {time_kor}: {name}")
            if theme or area:
                print(f"      · 테마/지역: {theme} / {area}")
            if category:
                print(f"      · 카테고리: {category}")
            if addr:
                print(f"      · 주소: {addr}")
