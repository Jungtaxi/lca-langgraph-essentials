import gradio as gr
import pandas as pd
import uuid
import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# 모듈 import
from state import AgentState
from agents.agent1_planner import planner_node
from agents.agent2_allocator import allocator_node
from agents.agent3_collector import collector_node

CATEGORY_CODES = {
    "MT1":"대형마트", 
    "CS2":"편의점", 
    "PS3":"어린이집/유치원",
    "SC4":"학교", 
    "AC5":"학원", 
    "PK6":"주차장", 
    "OL7":"주유소/충전소", 
    "SW8":"지하철역", 
    "BK9":"은행", 
    "CT1":"문화시설", 
    "AG2":"중개업소", 
    "PO3":"공공기관", 
    "AT4":"관광명소", 
    "AD5":"숙박", 
    "FD6":"음식점", 
    "CE7":"카페", 
    "HP8":"병원", 
    "PM9":"약국", 
}


# --- 그래프 조립 ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("allocator", allocator_node)
workflow.add_node("collector", collector_node)

workflow.set_entry_point("planner")

def check_complete(state: AgentState):
    if state['preferences'].is_complete:
        return "allocator"
    return END

workflow.add_conditional_edges("planner", check_complete, {"allocator": "allocator", END: END})
workflow.add_edge("allocator", "collector")
workflow.add_edge("collector", END)

app = workflow.compile(checkpointer=MemorySaver())

# --- [핵심] 유저 입력 처리 (즉시 반영) ---
def user_turn(user_message, history):
    if not user_message:
        return "", history
    # 유저 메시지를 히스토리에 즉시 추가하여 화면에 띄움
    history.append({"role": "user", "content": user_message})
    return "", history

# --- [핵심] 봇 응답 처리 (실시간 스트리밍) ---
def bot_turn(history, thread_id):
    if not thread_id: thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    last_user_msg = history[-1]['content']
    inputs = {"messages": [HumanMessage(content=last_user_msg)]}
    
    accumulated_state = {}
    
    # 봇의 '생각 중...' 메시지
    history.append({"role": "assistant", "content": "🤔 생각 중..."})
    
    for output in app.stream(inputs, config=config):
        for node_name, state_update in output.items():
            accumulated_state.update(state_update)
            
            log_msg = ""
            
            # 1. Agent 1 로그 (기존 동일)
            if node_name == "planner":
                prefs = state_update['preferences']
                if not prefs.is_complete:
                    log_msg = f"❓ **Agent 1 (기획):** 정보가 부족해요.\n\n_{prefs.missing_info_question}_"
                else:
                    log_msg = f"✅ **Agent 1 (기획):** 완료\n- 지역: {prefs.target_area}\n- 테마: {prefs.themes}"

            # 2. Agent 2 로그 (⭐⭐ 여기가 수정되었습니다 ⭐⭐)
            elif node_name == "allocator":
                strategy = state_update['strategy']
                
                # 헤더 작성
                log_msg += f"\n⬇️\n📊 **Agent 2 (전략):** 검색 전략 수립!\n\n"
                
                # 리스트 포맷팅 로직
                details = []
                for alloc in strategy.allocations:
                    
                    cat_name = alloc.tag_name
                    
                    # 한 줄 요약 작성
                    # 예: "- [음식점] (Weight 10): 맛집 테마 반영"
                    line = f"- **[{cat_name}]** (가중치 {alloc.weight}): {alloc.reason}"
                    details.append(line)
                
                # 줄바꿈으로 합치기
                log_msg += "\n".join(details)
                
            # 3. Agent 3 로그 (기존 동일)
            elif node_name == "collector":
                cands = state_update.get('candidates', [])
                log_msg += f"✅ **\n⬇️\n🏃 **Agent 3 (수집):** 장소 수집 끝! 총 {len(cands)}개 발견."

            # --- UI 업데이트 ---
            history[-1]['content'] += log_msg
            
            # (데이터 추출 및 yield 부분은 기존과 동일)
            curr_pref = accumulated_state.get('preferences')
            curr_strat = accumulated_state.get('strategy')
            curr_cands = accumulated_state.get('candidates')
            
            p_json = curr_pref.model_dump() if curr_pref else {}
            s_json = curr_strat.model_dump() if curr_strat else {}
            
            c_df = pd.DataFrame()
            if curr_cands:
                c_df = pd.DataFrame([c.model_dump() for c in curr_cands])
                if not c_df.empty:
                    c_df = c_df[['place_name', 'category', 'weight', 'keyword']]

            yield history, thread_id, p_json, s_json, c_df

    # (최종 마무리 부분 기존과 동일)
    final_prefs = accumulated_state.get('preferences')
    final_cands = accumulated_state.get('candidates')
    
    final_msg = ""
    if final_prefs and not final_prefs.is_complete:
        final_msg = final_prefs.missing_info_question
    elif final_cands:
        
        history.append({"role": "assistant", "content": "🤔 생각 중..."})
        final_msg = f"🎉 **모든 단계 완료!**\n총 {len(final_cands)}개의 후보 장소를 찾았습니다.\n오른쪽 탭에서 상세 내역을 확인하세요."
    else:
        final_msg = history[-1]['content']

    history[-1]['content'] = final_msg
    yield history, thread_id, p_json, s_json, c_df

# --- Gradio UI 레이아웃 ---
with gr.Blocks(title="Seoul Mate") as demo:
    tid_state = gr.State("") # 세션 ID
    
    gr.Markdown("# 🇰🇷 Seoul Mate AI Agent (Live Streaming)")
    
    with gr.Row():
        with gr.Column():
            # type="messages" 제거 (호환성)
            chatbot = gr.Chatbot(height=500) 
            msg = gr.Textbox(label="입력")
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("1. 기획"): json_pref = gr.JSON()
                with gr.Tab("2. 전략"): json_strat = gr.JSON()
                with gr.Tab("3. 수집"): df_cand = gr.Dataframe()

    # [이벤트 체인]
    # 1. 유저 입력 즉시 반영 (user_turn)
    # 2. 이어서 봇 스트리밍 실행 (bot_turn)
    msg.submit(
        user_turn, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_turn,
        inputs=[chatbot, tid_state],
        outputs=[chatbot, tid_state, json_pref, json_strat, df_cand]
    )

if __name__ == "__main__":
    demo.launch()