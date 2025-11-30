import gradio as gr
import pandas as pd
import uuid
import operator
from typing import Annotated, List, Optional, TypedDict 
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 모듈 import
from state import AgentState, CandidatePlace
from agents.agent0_router import router_node
from agents.agent1_planner import planner_node
from agents.agent2_allocator import allocator_node
from agents.agent3_collector_kakao import collector_node_kakao
from agents.agent3_collector_naver import collector_node_naver
from agents.agent4_suggest import agent4_suggest_node
from agents.agent5_path_finder import agent5_route_node 
import folium
# --- [UI 헬퍼] 번역 및 데이터프레임 변환 ---

# (기존 UI_LABELS, translate_text, translate_dataframe 등은 동일하게 유지)
UI_LABELS = {
    # 1. 기획 (Planner) 관련
    "여행 지역": {"English": "Target Area", "Japanese": "旅行エリア", "Chinese": "旅游区域"},
    "기간": {"English": "Duration", "Japanese": "期間", "Chinese": "期间"},
    "테마": {"English": "Themes", "Japanese": "テーマ", "Chinese": "主题"},
    "강도": {"English": "Intensity", "Japanese": "旅行強度", "Chinese": "强度"},
    "동행자": {"English": "Companions", "Japanese": "同行者", "Chinese": "同行人员"},
    "이동수단": {"English": "Transport", "Japanese": "移動手段", "Chinese": "交通方式"},
    "요약/노트": {"English": "Summary/Note", "Japanese": "要約・ノート", "Chinese": "摘要/备注"},
    
    # 2. 전략 (Strategy) 관련
    "카테고리": {"English": "Category", "Japanese": "カテゴリ", "Chinese": "类别"},
    "가중치": {"English": "Weight", "Japanese": "重要度", "Chinese": "权重"},
    "목표 개수": {"English": "Target Count", "Japanese": "目標数", "Chinese": "目标数量"},
    "검색 키워드": {"English": "Keywords", "Japanese": "検索キーワード", "Chinese": "搜索关键词"},
    "선정 이유": {"English": "Reason", "Japanese": "選定理由", "Chinese": "选定理由"},
    
    # 3. 수집 (Collector) 관련
    "장소명": {"English": "Place Name", "Japanese": "場所名", "Chinese": "地点名称"},
    "키워드": {"English": "Keyword", "Japanese": "キーワード", "Chinese": "关键词"},
    "주소": {"English": "Address", "Japanese": "住所", "Chinese": "地址"},
    "URL": {"English": "Map URL", "Japanese": "地図URL", "Chinese": "地图链接"}
}


def create_map_html(data):
    """
    data: 
      - List[CandidatePlace]: Agent 4 (제안) 단계 -> 단색 마커 표시
      - FinalItinerary: Agent 5 (경로) 단계 -> 일자별 다른 색상 경로 표시
    """
    if not data:
        return "<div style='text-align:center; padding:20px; color:gray;'>지도에 표시할 데이터가 없습니다.</div>"
    
    # 지도 초기 중심 잡기 위한 좌표 수집
    all_lats = []
    all_lngs = []

    # --- [Case 1] Agent 5: 최종 경로 (일자별 색상 구분) ---
    if hasattr(data, 'schedule'): # FinalItinerary 객체인지 확인
        
        # 일자별 색상 팔레트 (Folium 지원 색상)
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'cadetblue']
        
        # 좌표 수집 (중심 잡기용)
        for day in data.schedule:
            for sp in day.places:
                if sp.place.y > 0 and sp.place.x > 0:
                    all_lats.append(sp.place.y)
                    all_lngs.append(sp.place.x)
        
        if not all_lats: return "<div>유효한 좌표가 없습니다.</div>"
        
        # 지도 생성
        avg_lat, avg_lng = sum(all_lats)/len(all_lats), sum(all_lngs)/len(all_lngs)
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13)

        # 일자별 루프
        for idx, day_schedule in enumerate(data.schedule):
            # 색상 선택 (일자별 순환)
            day_color = colors[idx % len(colors)]
            day_coords = [] # 선 그리기용 좌표 리스트
            
            # 장소 루프
            for sp in day_schedule.places:
                place = sp.place
                lat, lng = place.y, place.x
                
                if lat <= 0 or lng <= 0: continue
                
                day_coords.append((lat, lng))
                
                # 팝업 내용
                popup_html = (
                    f"<div style='min-width:150px'>"
                    f"<b style='color:{day_color}'>[Day {day_schedule.day}] {sp.order}. {place.place_name}</b><br>"
                    f"<span style='font-size:12px;'>{place.category}</span><br>"
                    f"<span style='font-size:11px; color:gray'>{sp.visit_time}</span><br>"
                    f"<a href='{place.place_url}' target='_blank'>Kakao Map</a>"
                    f"</div>"
                )
                
                # 마커 추가
                folium.Marker(
                    [lat, lng],
                    popup=popup_html,
                    tooltip=f"Day{day_schedule.day}-{sp.order}. {place.place_name}",
                    icon=folium.Icon(color=day_color, icon='info-sign')
                ).add_to(m)
            
            # [핵심] 일자별 경로 선 그리기
            if len(day_coords) > 1:
                folium.PolyLine(
                    locations=day_coords,
                    color=day_color,
                    weight=5,
                    opacity=0.8,
                    tooltip=f"Day {day_schedule.day} 경로"
                ).add_to(m)
                
        return m._repr_html_()

    # --- [Case 2] Agent 4: 후보 제안 (단색 표시) ---
    elif isinstance(data, list):
        candidates = data
        lats = [c.y for c in candidates if c.y > 0]
        lngs = [c.x for c in candidates if c.x > 0]
        
        if not lats: return "<div>유효한 좌표가 없습니다.</div>"
        
        avg_lat, avg_lng = sum(lats)/len(lats), sum(lngs)/len(lngs)
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13)
        
        for i, c in enumerate(candidates, 1):
            popup_html = f"<div style='width:150px'><b>{i}. {c.place_name}</b><br>{c.category}<br><a href='{c.place_url}' target='_blank'>Kakao Map</a></div>"
            folium.Marker(
                [c.y, c.x], 
                popup=popup_html, 
                tooltip=f"{i}. {c.place_name}",
                icon=folium.Icon(color='blue', icon='star')
            ).add_to(m)
            
        return m._repr_html_()

    else:
        return "<div>지도 데이터 형식이 올바르지 않습니다.</div>"
    
def translate_text(text, target_lang):
    text = str(text)
    if text.startswith("http") or text.startswith("www"): return text
    if target_lang in ["Korean", "한국어"] or not text.strip(): return text
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = f"Translate Korean to {target_lang}. Keep proper nouns/codes. Return only text."
    try:
        res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=text)])
        return res.content
    except: return text

def translate_dataframe(df, target_lang):
    if target_lang in ["Korean", "한국어"] or df.empty: return df
    
    col_map = {
        "항목": "Item", "내용": "Content", "카테고리": "Category", 
        "장소명": "Place Name", "키워드": "Keyword", "주소": "Address",
        "선정 이유": "Reason"
    }
    df = df.rename(columns={k: v for k,v in col_map.items() if k in df.columns})
    
    SKIP = ["URL", "Link", "Place Name", "장소명"]
    target_cols = [c for c in df.columns if df[c].dtype == 'object' and not any(s in c for s in SKIP)]
    
    for col in target_cols:
        df[col] = df[col].apply(lambda x: translate_text(str(x), target_lang))
    return df

def format_prefs_to_df(prefs):
    if not prefs: return pd.DataFrame()
    data = prefs.model_dump()
    display_map = {"target_area": "여행 지역", "themes": "테마", "duration": "기간", "companions": "동행자"}
    table_data = []
    for key, label in display_map.items():
        val = data.get(key)
        if isinstance(val, list): val = ", ".join(val)
        table_data.append({"항목": label, "내용": str(val)})
    return pd.DataFrame(table_data)

def format_strategy_to_df(strategy):
    if not strategy: return pd.DataFrame()
    rows = []
    for alloc in sorted(strategy.allocations, key=lambda x: x.weight, reverse=True):
        rows.append({"카테고리": alloc.tag_name, "키워드": ", ".join(alloc.keywords),  "선정 이유": alloc.reason})
    return pd.DataFrame(rows)


# --- 그래프 조립 ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("planner", planner_node)
workflow.add_node("allocator", allocator_node)
# workflow.add_node("kakao", collector_node_kakao)
workflow.add_node("naver", collector_node_naver)
workflow.add_node("suggester", agent4_suggest_node)
workflow.add_node("path_finder", agent5_route_node) 
# workflow.add_node("scheduler", agent5_schedule_node) # [Future] Agent 5 추가 예정
def get_next_node(state):
    return state["next_step"]

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    get_next_node,
    {
        "planner": "planner",
        "suggester": "suggester",     # 유저가 "술집 보여줘" 하면 여기로
        "path_finder": "path_finder", # 유저가 "1번 갈래" 하면 여기로
        "general_chat": END           # 잡담이면 그냥 답변하고 끝내거나 별도 노드로
    }
)

def check_complete(state: AgentState):
    if state['preferences'].is_complete: return "allocator"
    return END

workflow.add_conditional_edges("planner", check_complete, {"allocator": "allocator", END: END})
# workflow.add_edge("allocator", "kakao")
workflow.add_edge("allocator", "naver")
# workflow.add_edge("kakao", "suggester")
workflow.add_edge("naver", "suggester")

# [중요] Suggester 이후 Agent 5로 바로 가지 않고 일단 END.
# 사용자가 채팅창에서 "여기 여기 갈래"라고 입력하면, 그때 Router가 판단해서 Agent 5로 보내는 구조가 됩니다.

workflow.add_edge("suggester", END) 

app = workflow.compile(checkpointer=MemorySaver())

# --- Gradio 로직 ---
def user_turn(user_message, history):
    if not user_message: return "", history
    history.append({"role": "user", "content": user_message})
    return "", history

def bot_turn(history, thread_id):
    if not thread_id: thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    last_user_msg = history[-1]['content']
    inputs = {"messages": [HumanMessage(content=last_user_msg)]}
    
    accumulated_state = {}
    history.append({"role": "assistant", "content": "🤔 Thinking..."})
    
    detected_language = "Korean"

    # [핵심 수정] 초기값을 루프 밖에서 미리 선언해야 에러가 안 납니다!
    map_html = "<div style='text-align:center; padding:20px; color:gray;'>아직 지도가 생성되지 않았습니다.</div>"
    for output in app.stream(inputs, config=config):
        for node_name, state_update in output.items():
            accumulated_state.update(state_update)
            
            if 'preferences' in accumulated_state and accumulated_state['preferences']:
                if accumulated_state['preferences'].language:
                    detected_language = accumulated_state['preferences'].language

            # --- 로그 메시지 생성 ---
            kor_log = ""
            if node_name == "planner":
                prefs = state_update['preferences']
                if not prefs.is_complete:
                    kor_log = f"❓ **Agent 1:** {prefs.missing_info_question}"
                else:
                    kor_log = f"✅ **Agent 1:** 기획 완료!\n- 지역: {prefs.target_area}\n- 테마: {prefs.themes}"

            elif node_name == "allocator":
                kor_log = f"\n ⬇️\n📊 **Agent 2:** 전략 수립 완료!"

            elif node_name in ["kakao", "naver"]:
                cands = accumulated_state.get('candidates', [])
                source = "Kakao" if node_name == "kakao" else "Naver"
                kor_log = f"\n ⬇️\n🏃 **Agent 3 ({source}):** 수집 중... (현재 누적 {len(cands)}개)"

            # [핵심 수정] Agent 4 결과 출력 (체크박스 제거 -> 채팅창 리스트 출력)
            elif node_name == "suggester":
                main_cands = state_update.get('main_place_candidates', [])
                # Folium 지도 HTML 생성
                map_html = create_map_html(main_cands)
                
                # Markdown 리스트 생성
                list_text = []
                for i, c in enumerate(main_cands, 1):
                    # URL이 있으면 링크 생성, 없으면 텍스트만
                    link_text = f"[지도보기]({c.place_url})" if c.place_url else "(링크없음)"
                    row = f"{i}. **{c.place_name}** ({c.category}) | {c.address} | {link_text}"
                    list_text.append(row)
                
                candidates_str = "\n".join(list_text)
                
                kor_log = (
                    f"\n ⬇️\n✨ **Agent 4:** 후보 장소를 엄선했습니다!\n\n"
                    f"{candidates_str}\n\n"
                    f"💡 **이 중에서 방문하고 싶은 곳을 말씀해 주시면, Agent 5가 최적의 루트를 짜드릴게요!**"
                )
            
            elif node_name=="path_finder":  ### 내가 main에서 agent5넣고 수정해야하는 것
                # [수정] 구조화된 일정 객체(FinalItinerary)를 그대로 가져옴
                final_itinerary = accumulated_state.get('final_itinerary')
                
                if final_itinerary:
                    # [핵심] 객체를 통째로 create_map_html에 넘김 (함수 안에서 타입 체크함)
                    map_html = create_map_html(final_itinerary)
                    
                    # 로그 메시지 생성
                    log_text = f"\n⬇️\n🚗 **Agent 5:** 최종 일정 생성 완료!\n\n**[총평]** {final_itinerary.overall_review}\n"
                    
                    for day in final_itinerary.schedule:
                        # 일자별 테마 표시
                        log_text += f"\n**📅 Day {day.day} - {day.daily_theme}**\n"
                        for sp in day.places:
                            log_text += f"{sp.order}. {sp.place.place_name} ({sp.visit_time})\n"
                            
                    kor_log = log_text
                else:
                    kor_log = "⚠️ 일정 생성 실패."
            
                
            # --- 번역 및 UI 업데이트 ---
            # 링크(Markdown Link)가 깨지지 않도록 주의하며 번역
            # translate_text 함수가 URL을 건드리지 않도록 되어 있으므로 안전함
            final_display_log = translate_text(kor_log, detected_language)
            
            if final_display_log:
                if history[-1]['content'] == "🤔 Thinking...":
                    history[-1]['content'] = final_display_log
                else:
                    history[-1]['content'] += "\n\n" + final_display_log
            
            # --- 데이터프레임 갱신 ---
            curr_pref = accumulated_state.get('preferences')
            curr_strat = accumulated_state.get('strategy')
            
            
            df_p = format_prefs_to_df(curr_pref)
            df_s = format_strategy_to_df(curr_strat)
            
            if detected_language and detected_language not in ["Korean", "한국어"]:
                df_p = translate_dataframe(df_p, detected_language)
                df_s = translate_dataframe(df_s, detected_language)
            # yield에 map_html 추가 (순서 주의)
            yield history, thread_id, df_p, df_s, map_html

    # 최종 상태 한 번 더 yield
    yield history, thread_id, df_p, df_s, map_html

# --- Gradio UI (단순화됨) ---
with gr.Blocks(title="Seoul Hunters") as demo:
    tid_state = gr.State("")
    
    with gr.Row():
        gr.Markdown("# Seoul Hunters")
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Input", placeholder="여행 계획을 이야기해주세요...")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("1. Planner"):
                    df_pref_ui = gr.Dataframe(headers=["항목", "내용"], wrap=True)
                with gr.Tab("2. Strategy"):
                    df_strat_ui = gr.Dataframe(headers=["카테고리", "키워드"], wrap=True)
                
                # [수정] 3번 탭을 '지도 & 제안'으로 통합
                with gr.Tab("3. Map & Suggestion"):
                    # 지도 표시용 HTML 컴포넌트
                    map_output = gr.HTML(label="Interactive Map")

    msg.submit(
        user_turn, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_turn,
        inputs=[chatbot, tid_state],
        outputs=[chatbot, tid_state, df_pref_ui, df_strat_ui, map_output] # outputs 순서 주의!
    )

if __name__ == "__main__":
    demo.launch()