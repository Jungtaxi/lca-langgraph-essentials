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


def create_map_html(places, is_route=False):
    """
    places: 장소 리스트
    is_route: True면 순서대로 선을 연결함 (Agent 5 결과용)
    """
    if not places:
        return "<div style='text-align:center; padding:20px; color:gray;'>지도에 표시할 데이터가 없습니다.</div>"
    
    try:
        # 좌표 유효성 검사 (0,0 제외)
        valid_places = [p for p in places if p.x > 0 and p.y > 0]
        
        if not valid_places:
            return "<div>유효한 좌표가 없습니다.</div>"
        
        # 중심 좌표 계산
        avg_lat = sum(p.y for p in valid_places) / len(valid_places)
        avg_lng = sum(p.x for p in valid_places) / len(valid_places)
        
        m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13)
        
        # 좌표 리스트 (선 그리기용)
        route_coords = []

        for i, p in enumerate(valid_places, 1):
            lat, lng = p.y, p.x
            route_coords.append((lat, lng))
            
            # 마커 색상 (경로 모드일 때: 출발=빨강, 도착=초록, 중간=파랑)
            if is_route:
                if i == 1: color = 'red'       # Start
                elif i == len(valid_places): color = 'green' # End
                else: color = 'blue'
            else:
                color = 'blue' # 일반 제안 모드

            # 팝업 HTML
            popup_html = (
                f"<div style='min-width:150px'>"
                f"<b>{i}. {p.place_name}</b><br>"
                f"<span style='font-size:12px; color:gray'>{p.category}</span><br>"
                f"<a href='{p.place_url}' target='_blank' style='text-decoration:none; color:blue;'>kakao map 🔗</a>"
                f"</div>"
            )
            
            folium.Marker(
                [lat, lng],
                popup=popup_html,
                tooltip=f"{i}. {p.place_name}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)

        # [핵심] 경로 모드일 경우 선 그리기
        if is_route and len(route_coords) > 1:
            folium.PolyLine(
                locations=route_coords,
                color="blue",
                weight=5,
                opacity=0.7,
                tooltip="추천 이동 경로"
            ).add_to(m)

        return m._repr_html_()
        
    except Exception as e:
        return f"<div>Map Error: {str(e)}</div>"
    
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

def format_candidates_to_df(candidates):
    if not candidates: return pd.DataFrame()
    return pd.DataFrame([
        {"장소명": c.place_name, "카테고리": c.category, "키워드": c.keyword, "주소": c.address}
        for c in candidates[:100]
    ])

def format_main_candidates_to_df(candidates):
    if not candidates: return pd.DataFrame()
    data = []
    for c in candidates:
        row = {"장소명": c.place_name, "카테고리": c.category, "주소": c.address, "URL": c.place_url}
        data.append(row)
    return pd.DataFrame(data)


# 4. Conditional Edge 설정


# 5. Graph 연결




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
                map_html = create_map_html(main_cands, is_route=False)
                
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
                # Agent5가 만든 동선 텍스트
                routes_text = state_update.get("routes_text") or accumulated_state.get("routes_text", "")
                
                # 1. 확정된 경로 데이터 가져오기
                # (Agent 5가 state['selected_main_places'] 또는 state['final_route']에 저장했다고 가정)
                final_places = accumulated_state.get('selected_main_places', [])
                
                if final_places:
                    # 2. 지도 업데이트 (is_route=True 로 선 그리기!)
                    map_html = create_map_html(final_places, is_route=True)
                    
                    kor_log = (
                        f"\n⬇️\n🚗 **Agent 5:** 경로 생성 완료!\n"
                        f"{routes_text}"
                    )
                else:
                    kor_log = "⚠️ 경로를 생성할 장소 데이터가 없습니다."
            
                
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
            curr_main_cands = accumulated_state.get('main_place_candidates')
            
            df_p = format_prefs_to_df(curr_pref)
            df_s = format_strategy_to_df(curr_strat)
            df_m = format_main_candidates_to_df(curr_main_cands)
            
            if detected_language and detected_language not in ["Korean", "한국어"]:
                df_p = translate_dataframe(df_p, detected_language)
                df_s = translate_dataframe(df_s, detected_language)
                df_m = translate_dataframe(df_m, detected_language)
            # yield에 map_html 추가 (순서 주의)
            yield history, thread_id, df_p, df_s, df_m, map_html

    # 최종 상태 한 번 더 yield
    yield history, thread_id, df_p, df_s, df_m, map_html

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
                    # 후보 장소 리스트
                    df_main_ui = gr.Dataframe(headers=["장소명", "카테고리", "주소", "URL"], wrap=True)

    msg.submit(
        user_turn, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_turn,
        inputs=[chatbot, tid_state],
        outputs=[chatbot, tid_state, df_pref_ui, df_strat_ui, df_main_ui, map_output] # outputs 순서 주의!
    )

if __name__ == "__main__":
    demo.launch()