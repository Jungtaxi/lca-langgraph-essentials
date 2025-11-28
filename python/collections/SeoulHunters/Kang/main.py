import gradio as gr
import uuid
import json
import pandas as pd
from langchain_core.messages import HumanMessage

# --- [UI 헬퍼 함수] 데이터 포맷팅 ---
def format_json(data):
    """Pydantic 모델이나 Dict를 예쁜 JSON 문자열로 변환"""
    if hasattr(data, 'model_dump'):
        return data.model_dump()
    return data

def format_candidates_to_df(candidates):
    """수집된 장소 리스트를 데이터프레임으로 변환 (테이블 표시용)"""
    if not candidates:
        return pd.DataFrame()
    
    data = []
    for c in candidates:
        data.append({
            "장소명": c.place_name,
            "카테고리": c.category,
            "키워드": c.keyword,
            "Weight": c.weight,
            "주소": c.address,
            "URL": c.place_url
        })
    return pd.DataFrame(data)

# --- [핵심 로직] 채팅 처리 함수 ---
def respond(message, history, thread_id):
    """
    Gradio 채팅창에서 유저 입력을 받아 LangGraph를 실행하는 함수
    """
    if not thread_id:
        thread_id = str(uuid.uuid4()) # 세션 ID 생성
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. LangGraph 실행
    # (inputs에 messages만 넣으면 MemorySaver가 알아서 히스토리 관리함)
    inputs = {"messages": [HumanMessage(content=message)]}
    
    # invoke를 통해 그래프 실행 (중간 단계는 생략하고 최종 결과만 받음)
    # stream을 쓰면 좋지만, UI 단순화를 위해 invoke 사용
    result_state = app.invoke(inputs, config=config)
    
    # 2. 결과 추출
    prefs = result_state.get('preferences')
    strategy = result_state.get('strategy')
    candidates = result_state.get('candidates')
    
    # 3. 챗봇 응답 메시지 결정
    bot_message = ""
    if prefs and not prefs.is_complete:
        # 정보가 부족하면 Agent 1의 질문 반환
        bot_message = prefs.missing_info_question
    elif candidates:
        # 후보군 수집까지 끝났다면 결과 요약 반환
        bot_message = f"🎉 **{len(candidates)}개의 장소**를 찾았습니다!\n\n오른쪽 탭에서 상세 정보를 확인해보세요.\n이제 Agent 4가 최적의 경로를 계산할 준비가 되었습니다."
    else:
        # 정보는 다 찼는데 아직 수집 전 (혹은 에러)
        bot_message = "정보 확인이 완료되었습니다. 잠시만 기다려주세요..."

    # 4. 오른쪽 패널 데이터 업데이트
    pref_json = format_json(prefs) if prefs else {}
    strat_json = format_json(strategy) if strategy else {}
    cand_df = format_candidates_to_df(candidates)
    
    return bot_message, thread_id, pref_json, strat_json, cand_df

# --- [Gradio UI 구성] ---
with gr.Blocks(title="Seoul Mate AI", theme=gr.themes.Soft()) as demo:
    # 세션 ID 저장소 (브라우저 새로고침 전까지 유지)
    thread_id_state = gr.State(value="")
    
    gr.Markdown("# 🇰🇷 Seoul Mate: AI 여행 플래너")
    gr.Markdown("서울 여행 계획을 도와드립니다. 친구에게 말하듯이 편하게 이야기해주세요!")
    
    with gr.Row():
        # [왼쪽] 채팅창
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=600, type="messages")
            msg = gr.Textbox(placeholder="예: 친구랑 종로 1박 2일 맛집 투어 갈래", label="입력")
            clear = gr.ClearButton([msg, chatbot])

        # [오른쪽] 내부 상태 대시보드
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📋 1. 기획 (Agent 1)"):
                    gr.Markdown("### 사용자 의도 분석 결과")
                    pref_display = gr.JSON(label="Trip Preferences")
                
                with gr.TabItem("📊 2. 전략 (Agent 2)"):
                    gr.Markdown("### 검색 키워드 및 할당 전략")
                    strat_display = gr.JSON(label="Search Strategy")
                
                with gr.TabItem("📍 3. 수집 (Agent 3)"):
                    gr.Markdown("### 수집된 장소 후보군 (Pool)")
                    cand_display = gr.Dataframe(label="Candidate Places", headers=["장소명", "카테고리", "Weight"], wrap=True)

    # 이벤트 연결
    # msg.submit -> respond 함수 실행 -> [chatbot, thread_id, json패널들] 업데이트
    msg.submit(
        respond, 
        [msg, chatbot, thread_id_state], 
        [chatbot, thread_id_state, pref_display, strat_display, cand_display]
    )
    
    # 챗봇에 응답 추가 (Gradio 최신 버전 방식)
    def update_chat(user_msg, history, bot_msg):
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_msg})
        return history, "" # msg박스 비우기

    msg.submit(
        respond, 
        [msg, chatbot, thread_id_state], 
        [thread_id_state, thread_id_state, pref_display, strat_display, cand_display]
    ).then(
        # 챗봇 메시지 UI 업데이트는 별도로 처리 (봇 응답만 가져와서)
        lambda user, hist, res: update_chat(user, hist, res[0]),
        [msg, chatbot, msg], # msg를 임시로 출력 결과로 사용 (응답 텍스트)
        [chatbot, msg]
    )

    # (주의: 위 이벤트 체인이 복잡하면 아래의 간단한 ChatInterface 스타일로 대체 가능)
    # 하지만 좌우 레이아웃을 위해 Custom Event를 씁니다.
    # 위 체인이 복잡하니, 더 직관적인 '함수 하나가 모든 걸 리턴하는 방식'으로 정리합니다.
    
    def chat_wrapper(message, history, thread_id):
        # 1. 로직 실행
        bot_response, new_thread_id, p_json, s_json, c_df = respond(message, history, thread_id)
        
        # 2. 히스토리 업데이트
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        
        return "", history, new_thread_id, p_json, s_json, c_df

    # 기존 이벤트 덮어쓰기 (가장 깔끔한 방식)
    msg.submit(
        chat_wrapper,
        inputs=[msg, chatbot, thread_id_state],
        outputs=[msg, chatbot, thread_id_state, pref_display, strat_display, cand_display]
    )

if __name__ == "__main__":
    demo.launch()