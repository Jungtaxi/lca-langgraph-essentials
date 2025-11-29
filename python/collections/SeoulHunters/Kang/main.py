import gradio as gr
import pandas as pd
import uuid
import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 모듈 import
from state import AgentState
from agents.agent1_planner import planner_node
from agents.agent2_allocator import allocator_node
from agents.agent3_collector_kakao import collector_node_kakao
from agents.agent3_collector_naver import collector_node_naver

# --- [추가] 고정 UI 라벨 번역 사전 (속도/정확도 향상) ---
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
# --- [0] 번역기 (Translation Layer) ---
def translate_text(text, target_lang):
    """
    한국어 텍스트를 대상 언어로 번역합니다.
    """
    text = str(text) # 문자열 변환 안전장치
    
    # [수정] URL이거나, 한국어거나, 빈 값이면 번역 스킵
    if text.startswith("http") or text.startswith("www"):
        return text
    if target_lang in ["Korean", "한국어"] or not text.strip():
        return text
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system_prompt = f"""
    You are a professional translator for a travel AI agent.
    Translate the following Korean text into **{target_lang}**.
    
    [Rules]
    1. Keep the tone professional yet friendly.
    2. Do NOT translate proper nouns unless necessary.
    3. Keep emojis as they are.
    4. Return ONLY the translated text.
    """
    
    sys_msg = SystemMessage(content=system_prompt)
    msg = HumanMessage(content=text)
    
    try:
        response = llm.invoke([sys_msg, msg])
        return response.content
    except Exception as e:
        print(f"Translation Error: {e}")
        return text
    
# --- [수정] 데이터프레임 번역 함수 ---
def translate_dataframe(df, target_lang):
    """
    데이터프레임의 컬럼과 내용을 번역합니다. (URL, 장소명 제외)
    """
    if target_lang in ["Korean", "한국어"] or df.empty:
        return df
    
    # 1. 컬럼 번역 (UI 라벨 매핑)
    col_map = {
        "항목": "Item", "내용": "Content",
        "카테고리": "Category", "가중치": "Weight", "목표 개수": "Target Count", 
        "검색 키워드": "Keywords", "선정 이유": "Reason",
        "장소명": "Place Name", "키워드": "Keyword", "주소": "Address"
    }
    
    renamed_cols = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=renamed_cols)

    # 2. 내용 번역 (선별적 번역)
    # 번역하면 안 되는 컬럼명 키워드 (각 언어별 장소명/URL 라벨 포함)
    SKIP_KEYWORDS = [
        "URL", "Link", "Place Name", "장소명", "場所名", "地点名称", 
        "링크", "ID", "Code"
    ]
    
    # 텍스트 컬럼만 대상
    target_cols = [c for c in df.columns if df[c].dtype == 'object']
    
    for col in target_cols:
        # [핵심 수정] 컬럼 이름에 금지어(URL, 장소명 등)가 포함되어 있으면 번역 스킵!
        if any(skip_word in col for skip_word in SKIP_KEYWORDS):
            continue
            
        # 나머지(키워드, 주소, 이유 등)는 번역 진행
        df[col] = df[col].apply(lambda x: translate_cell_value(str(x), "English", target_lang))
        
    return df

def translate_cell_value(text, lang_key, full_target_lang):
    """
    셀 값 하나를 번역하는 헬퍼 함수
    1순위: UI_LABELS 사전 매칭 (빠름)
    2순위: LLM 번역 (느리지만 정확)
    """
    # 1. 사전에 있는 단어인지 확인 (예: '여행 지역'이라는 값이 셀 안에 들어있을 경우)
    if text in UI_LABELS and lang_key in UI_LABELS[text]:
        return UI_LABELS[text][lang_key]
    
    # 2. 사전에 없으면 LLM 번역 (숫자나 짧은 기호는 패스)
    if len(text) < 2 or text.isdigit():
        return text
        
    return translate_text(text, full_target_lang)

# --- [UI 헬퍼] 데이터프레임 변환 ---
def format_prefs_to_df(prefs):
    if not prefs: return pd.DataFrame()
    data = prefs.model_dump()
    display_map = {
        "target_area": "여행 지역", "duration": "기간", "themes": "테마",
        "intensity": "강도", "companions": "동행자", "transport": "이동수단",
        "additional_notes": "요약/노트"
    }
    table_data = []
    for key, label in display_map.items():
        val = data.get(key)
        if isinstance(val, list): val = ", ".join(val)
        table_data.append({"항목": label, "내용": str(val)})
    return pd.DataFrame(table_data)

def format_strategy_to_df(strategy):
    if not strategy: return pd.DataFrame()
    rows = []
    # 가중치 높은 순 정렬
    sorted_allocs = sorted(strategy.allocations, key=lambda x: x.weight, reverse=True)
    for alloc in sorted_allocs:
        rows.append({
            "카테고리": alloc.tag_name, "가중치": alloc.weight, "목표 개수": alloc.count,
            "검색 키워드": ", ".join(alloc.keywords), "선정 이유": alloc.reason
        })
    return pd.DataFrame(rows)

def format_candidates_to_df(candidates):
    if not candidates: return pd.DataFrame()
    
    # [수정] Weight 별 상위 3개씩 필터링 로직
    # 1. Weight 기준으로 그룹화하기 위해 데이터프레임 먼저 생성
    df = pd.DataFrame([c.model_dump() for c in candidates])
    
    if df.empty: return df
    
    # 2. Weight 내림차순 정렬
    df = df.sort_values(by="weight", ascending=False)
    
    # 3. 각 Weight(또는 카테고리 코드) 별로 상위 3개만 남기기
    # (같은 Weight를 가진 그룹 내에서 3개 자르기)
    df_filtered = df.groupby("tag_name").head(3)
    
    # 4. 필요한 컬럼만 선택 (Weight, Category 제거 요청 반영)
    # "장소명", "키워드", "주소", "URL" 정도만 남김
    result_df = df_filtered[["place_name", "keyword", "address", "place_url"]]
    
    # 컬럼명 한글로 변경
    result_df.columns = ["장소명", "키워드", "주소", "URL"]
    
    return result_df

# --- 그래프 조립 (기존과 동일) ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("allocator", allocator_node)
workflow.add_node("collector", collector_node_naver)
workflow.set_entry_point("planner")
def check_complete(state: AgentState):
    if state['preferences'].is_complete: return "allocator"
    return END
workflow.add_conditional_edges("planner", check_complete, {"allocator": "allocator", END: END})
workflow.add_edge("allocator", "collector")
workflow.add_edge("collector", END)
app = workflow.compile(checkpointer=MemorySaver())

# --- [핵심] 유저 입력 처리 ---
def user_turn(user_message, history):
    if not user_message: return "", history
    history.append({"role": "user", "content": user_message})
    return "", history

# --- [핵심] 봇 응답 처리 ---
def bot_turn(history, thread_id):
    if not thread_id: thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    last_user_msg = history[-1]['content']
    inputs = {"messages": [HumanMessage(content=last_user_msg)]}
    
    accumulated_state = {}
    history.append({"role": "assistant", "content": "🤔 Thinking..."})
    
    # [NEW] 감지된 언어 초기값
    detected_language = "Korean"

    for output in app.stream(inputs, config=config):
        for node_name, state_update in output.items():
            accumulated_state.update(state_update)
            
            # [NEW] Agent 1에서 감지된 언어 가져오기
            if 'preferences' in accumulated_state and accumulated_state['preferences']:
                pref_lang = accumulated_state['preferences'].language
                if pref_lang:
                    detected_language = pref_lang
            
            # --- 로그 메시지 생성 (한국어) ---
            kor_log = ""
            if node_name == "planner":
                prefs = state_update['preferences']
                if not prefs.is_complete:
                    kor_log = f"❓ **Agent 1:** {prefs.missing_info_question}"
                else:
                    kor_log = f"✅ **Agent 1:** 기획 완료!\n- 지역: {prefs.target_area}\n- 테마: {prefs.themes}"
            elif node_name == "allocator":
                strategy = state_update['strategy']
                kor_log = f"\n⬇️\n📊 **Agent 2:** 전략 수립 완료!\n\n"
                sorted_allocs = sorted(strategy.allocations, key=lambda x: x.weight, reverse=True)
                for alloc in sorted_allocs[:5]:
                    kor_log += f"- **[{alloc.tag_name}]** (W:{alloc.weight}): {alloc.reason[:30]}...\n"
            elif node_name == "collector":
                cands = state_update.get('candidates', [])
                kor_log = f"\n⬇️\n🏃 **Agent 3:** 수집 완료! ({len(cands)}개)"

            # --- [자동 번역 단계] ---
            # detected_language로 채팅 메시지 번역
            final_display_log = translate_text(kor_log, detected_language)
            
            if final_display_log:
                if history[-1]['content'] == "🤔 Thinking...":
                    history[-1]['content'] = final_display_log
                else:
                    history[-1]['content'] += "\n\n" + final_display_log
            
            # --- 데이터프레임 생성 및 번역 ---
            curr_pref = accumulated_state.get('preferences')
            curr_strat = accumulated_state.get('strategy')
            curr_cands = accumulated_state.get('candidates')
            
            # 1. 한국어 DF 생성
            df_p = format_prefs_to_df(curr_pref)
            df_s = format_strategy_to_df(curr_strat)
            df_c = format_candidates_to_df(curr_cands) # 여기서 상위 3개 필터링 됨
            
            # 2. 감지된 언어로 번역 (한국어가 아닐 때만)
            if detected_language and detected_language not in ["Korean", "한국어"]:
                df_p = translate_dataframe(df_p, detected_language)
                df_s = translate_dataframe(df_s, detected_language)
                df_c = translate_dataframe(df_c, detected_language)

            yield history, thread_id, df_p, df_s, df_c

    # --- 최종 마무리 ---
    pass 

# --- Gradio UI 레이아웃 ---
with gr.Blocks(title="Seoul Mate") as demo:
    tid_state = gr.State("")
    
    with gr.Row():
        # 언어 선택 UI 제거됨 (자동 감지)
        gr.Markdown("# 🇰🇷 Seoul Mate AI Agent")
    
    with gr.Row():
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Input", placeholder="여행 계획을 이야기해주세요... (Start typing in any language)")
        
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("1. Planner"):
                    df_pref_ui = gr.Dataframe(headers=["항목", "내용"], wrap=True)
                with gr.Tab("2. Strategy"):
                    df_strat_ui = gr.Dataframe(headers=["카테고리", "가중치", "목표 개수", "검색 키워드", "선정 이유"], wrap=True)
                with gr.Tab("3. Collector"):
                    # 필요한 컬럼만 표시 (장소명, 키워드, 주소, URL)
                    df_cand_ui = gr.Dataframe(headers=["장소명", "키워드", "주소", "URL"], wrap=True)

    msg.submit(
        user_turn, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_turn,
        inputs=[chatbot, tid_state], # language_radio 인자 제거
        outputs=[chatbot, tid_state, df_pref_ui, df_strat_ui, df_cand_ui]
    )

if __name__ == "__main__":
    demo.launch()