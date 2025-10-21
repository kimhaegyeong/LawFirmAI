# -*- coding: utf-8 -*-
"""
LawFirmAI - 안정적인 ChatGPT 스타일 인터페이스
디자인 깨짐 문제 해결, JavaScript 오류 수정
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Gradio 및 기타 라이브러리
import gradio as gr
import torch

# 프로젝트 모듈
from source.services.chat_service import ChatService
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/stable_chatgpt_app.log')
    ]
)
logger = logging.getLogger(__name__)

class StableChatGPTStyleLawFirmAI:
    """안정적인 ChatGPT 스타일 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_service = None
        self.is_initialized = False
        
        # 대화 관리
        self.conversations = []
        self.current_conversation_id = None
        self.current_session_id = f"stable_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._initialize_components()
        self._create_new_conversation()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing stable ChatGPT-style components...")
            
            config = Config()
            self.chat_service = ChatService(config)
            
            self.is_initialized = True
            self.logger.info("Stable ChatGPT-style components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.is_initialized = False
    
    def _create_new_conversation(self):
        """새 대화 생성"""
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        conversation = {
            "id": conversation_id,
            "title": "새 대화",
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "last_updated": datetime.now().isoformat()
        }
        
        self.conversations.append(conversation)
        self.current_conversation_id = conversation_id
        
        return conversation_id
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """질의 처리"""
        if not self.is_initialized:
            return {
                "answer": "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "error": "System not initialized",
                "confidence": 0.0
            }
        
        if not query.strip():
            return {
                "answer": "질문을 입력해주세요.",
                "error": "Empty query",
                "confidence": 0.0
            }
        
        start_time = time.time()
        
        try:
            # ChatService를 사용한 처리
            import asyncio
            result = asyncio.run(self.chat_service.process_message(query, session_id=self.current_session_id))
            
            response_time = time.time() - start_time
            
            # 대화에 메시지 추가
            self._add_message_to_conversation("user", query)
            self._add_message_to_conversation("assistant", result.get("response", ""))
            
            return {
                "answer": result.get("response", ""),
                "confidence": result.get("confidence", 0.0),
                "processing_time": response_time,
                "question_type": result.get("question_type", "general"),
                "session_id": self.current_session_id
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Error processing query: {e}")
            
            return {
                "answer": "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "error": str(e),
                "confidence": 0.0,
                "processing_time": response_time
            }
    
    def _add_message_to_conversation(self, role: str, content: str):
        """대화에 메시지 추가"""
        if self.current_conversation_id:
            for conv in self.conversations:
                if conv["id"] == self.current_conversation_id:
                    conv["messages"].append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
                    conv["last_updated"] = datetime.now().isoformat()
                    
                    # 첫 번째 사용자 메시지로 대화 제목 생성
                    if role == "user" and len(conv["messages"]) == 1:
                        conv["title"] = content[:30] + "..." if len(content) > 30 else content
                    break
    
    def get_conversation_list(self) -> List[Dict[str, Any]]:
        """대화 목록 반환"""
        return [
            {
                "id": conv["id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "last_updated": conv["last_updated"],
                "message_count": len(conv["messages"])
            }
            for conv in sorted(self.conversations, key=lambda x: x["last_updated"], reverse=True)
        ]
    
    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """특정 대화의 메시지 반환"""
        for conv in self.conversations:
            if conv["id"] == conversation_id:
                return conv["messages"]
        return []
    
    def switch_conversation(self, conversation_id: str):
        """대화 전환"""
        self.current_conversation_id = conversation_id
        return self.get_conversation_messages(conversation_id)
    
    def delete_conversation(self, conversation_id: str):
        """대화 삭제"""
        self.conversations = [conv for conv in self.conversations if conv["id"] != conversation_id]
        if self.current_conversation_id == conversation_id:
            if self.conversations:
                self.current_conversation_id = self.conversations[0]["id"]
            else:
                self._create_new_conversation()

def create_stable_chatgpt_interface():
    """안정적인 ChatGPT 스타일 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = StableChatGPTStyleLawFirmAI()
    
    # 안정적인 ChatGPT CSS (단순화)
    css = """
    /* 안정적인 ChatGPT 스타일 CSS */
    * {
        box-sizing: border-box;
    }
    
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #212121;
        color: #ececf1;
    }
    
    .gradio-container {
        margin: 0 !important;
        padding: 0 !important;
        background: #212121 !important;
        height: 100vh !important;
        display: flex !important;
        flex-direction: row !important;
        max-width: 100% !important;
    }
    
    /* 사이드바 */
    .gradio-container > div:first-child {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        background: #171717 !important;
        border-right: 1px solid #2f2f2f !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        overflow-y: auto !important;
        flex-shrink: 0 !important;
    }
    
    /* 메인 영역 */
    .gradio-container > div:last-child {
        flex: 1 !important;
        background: #212121 !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* 사이드바 헤더 */
    .gradio-container > div:first-child > div:first-child {
        padding: 16px !important;
        border-bottom: 1px solid #2f2f2f !important;
        background: #171717 !important;
    }
    
    .gradio-container > div:first-child > div:first-child h2 {
        color: #ececf1 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    /* 새 대화 버튼 */
    .gradio-container > div:first-child button {
        width: calc(100% - 32px) !important;
        background: transparent !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
        padding: 12px 16px !important;
        border-radius: 6px !important;
        margin: 16px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
    }
    
    .gradio-container > div:first-child button:hover {
        background: #2f2f2f !important;
    }
    
    /* 대화 목록 */
    .gradio-container > div:first-child > div:last-child {
        flex: 1 !important;
        padding: 8px !important;
        overflow-y: auto !important;
    }
    
    .gradio-container > div:first-child > div:last-child label {
        color: #ececf1 !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        margin: 2px 8px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        display: block !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        transition: background-color 0.2s ease !important;
    }
    
    .gradio-container > div:first-child > div:last-child label:hover {
        background: #2f2f2f !important;
    }
    
    .gradio-container > div:first-child > div:last-child label.selected {
        background: #2f2f2f !important;
    }
    
    /* 채팅 헤더 */
    .gradio-container > div:last-child > div:first-child {
        padding: 16px 24px !important;
        border-bottom: 1px solid #2f2f2f !important;
        background: #212121 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .gradio-container > div:last-child > div:first-child h1 {
        color: #ececf1 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    .gradio-container > div:last-child > div:first-child button {
        background: transparent !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
    }
    
    .gradio-container > div:last-child > div:first-child button:hover {
        background: #2f2f2f !important;
    }
    
    /* 채팅 영역 */
    .gradio-container > div:last-child > div:nth-child(2) {
        flex: 1 !important;
        background: #212121 !important;
        overflow-y: auto !important;
        padding: 0 !important;
    }
    
    /* 채팅 메시지 */
    .gradio-container .message {
        padding: 24px !important;
        border-bottom: 1px solid #2f2f2f !important;
        display: flex !important;
        gap: 16px !important;
        align-items: flex-start !important;
    }
    
    .gradio-container .message .avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        flex-shrink: 0 !important;
    }
    
    .gradio-container .message.user .avatar {
        background: #5436da !important;
        color: white !important;
    }
    
    .gradio-container .message.assistant .avatar {
        background: #10a37f !important;
        color: white !important;
    }
    
    .gradio-container .message .content {
        flex: 1 !important;
        color: #ececf1 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        word-wrap: break-word !important;
    }
    
    /* 입력 영역 */
    .gradio-container > div:last-child > div:last-child {
        padding: 24px !important;
        background: #212121 !important;
        border-top: 1px solid #2f2f2f !important;
    }
    
    .gradio-container > div:last-child > div:last-child > div {
        max-width: 768px !important;
        margin: 0 auto !important;
        position: relative !important;
    }
    
    /* 입력창 */
    .gradio-container textarea {
        width: 100% !important;
        border: 1px solid #2f2f2f !important;
        border-radius: 12px !important;
        padding: 12px 50px 12px 16px !important;
        font-size: 16px !important;
        background: #2f2f2f !important;
        color: #ececf1 !important;
        resize: none !important;
        min-height: 24px !important;
        max-height: 200px !important;
        line-height: 1.5 !important;
        font-family: inherit !important;
    }
    
    .gradio-container textarea:focus {
        outline: none !important;
        border-color: #5436da !important;
        box-shadow: 0 0 0 3px rgba(84,54,218,0.1) !important;
    }
    
    .gradio-container textarea::placeholder {
        color: #8e8ea0 !important;
    }
    
    /* 전송 버튼 */
    .gradio-container > div:last-child > div:last-child button {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 6px !important;
        background: #5436da !important;
        border: none !important;
        color: white !important;
        cursor: pointer !important;
        font-size: 16px !important;
        transition: background-color 0.2s ease !important;
    }
    
    .gradio-container > div:last-child > div:last-child button:hover {
        background: #4c2db8 !important;
    }
    
    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2f2f2f;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4f4f4f;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .gradio-container {
            flex-direction: column !important;
        }
        
        .gradio-container > div:first-child {
            width: 100% !important;
            height: auto !important;
            max-height: 200px !important;
        }
        
        .gradio-container > div:last-child {
            height: calc(100vh - 200px) !important;
        }
        
        .gradio-container textarea {
            font-size: 16px !important;
            padding: 10px 45px 10px 14px !important;
        }
        
        .gradio-container > div:last-child > div:last-child button {
            width: 28px !important;
            height: 28px !important;
            font-size: 14px !important;
        }
    }
    
    /* Gradio 기본 스타일 오버라이드 */
    .gradio-container .gradio-button {
        background: transparent !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
    }
    
    .gradio-container .gradio-button:hover {
        background: #2f2f2f !important;
    }
    
    .gradio-container .gradio-textbox {
        background: #2f2f2f !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
    }
    
    .gradio-container .gradio-textbox:focus {
        border-color: #5436da !important;
    }
    
    .gradio-container .gradio-radio {
        background: transparent !important;
    }
    
    .gradio-container .gradio-radio label {
        color: #ececf1 !important;
    }
    """
    
    with gr.Blocks(
        css=css,
        title="LawFirmAI - 법률 AI 어시스턴트",
        theme=gr.themes.Soft()
    ) as interface:
        
        # 메인 레이아웃
        with gr.Row():
            # 왼쪽 사이드바
            with gr.Column(scale=0):
                # 사이드바 헤더
                gr.HTML("""
                <div style="padding: 16px; border-bottom: 1px solid #2f2f2f; background: #171717;">
                    <h2 style="color: #ececf1; font-size: 16px; font-weight: 600; margin: 0;">⚖️ LawFirmAI</h2>
                </div>
                """)
                
                # 새 대화 버튼
                new_chat_btn = gr.Button("+ 새 대화")
                
                # 대화 목록
                conversation_list = gr.Radio(
                    choices=[],
                    value=None,
                    label="대화 목록",
                    show_label=False
                )
            
            # 메인 채팅 영역
            with gr.Column(scale=1):
                # 채팅 헤더
                chat_title = gr.HTML("<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>새 대화</h1>")
                
                # 채팅 메시지 영역
                chatbot = gr.Chatbot(
                    label="",
                    height=600,
                    show_label=False,
                    type="messages",
                    container=False
                )
                
                # 입력 영역
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="메시지를 입력하세요...",
                        label="",
                        lines=1,
                        max_lines=10,
                        show_label=False
                    )
                    submit_btn = gr.Button("➤")
        
        # 이벤트 핸들러들
        def respond(message, history):
            """응답 생성"""
            if not message.strip():
                return history, ""
            
            # 질의 처리
            result = app.process_query(message)
            
            # 메시지 형식 변환
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result["answer"]})
            
            return history, ""
        
        def create_new_conversation():
            """새 대화 생성"""
            conversation_id = app._create_new_conversation()
            conversation_list = app.get_conversation_list()
            
            choices = [(conv["title"], conv["id"]) for conv in conversation_list]
            
            return choices, conversation_id, [], "새 대화"
        
        def switch_conversation(conversation_id):
            """대화 전환"""
            if not conversation_id:
                return [], "새 대화"
            
            messages = app.switch_conversation(conversation_id)
            conversation_list = app.get_conversation_list()
            
            # 현재 대화 찾기
            current_conv = None
            for conv in conversation_list:
                if conv["id"] == conversation_id:
                    current_conv = conv
                    break
            
            title = current_conv["title"] if current_conv else "대화"
            
            return messages, title
        
        def delete_conversation(conversation_id):
            """대화 삭제"""
            if conversation_id:
                app.delete_conversation(conversation_id)
                conversation_list = app.get_conversation_list()
                choices = [(conv["title"], conv["id"]) for conv in conversation_list]
                
                if conversation_list:
                    current_id = conversation_list[0]["id"]
                    messages = app.switch_conversation(current_id)
                    title = conversation_list[0]["title"]
                else:
                    current_id = None
                    messages = []
                    title = "새 대화"
                
                return choices, current_id, messages, title
            
            return [], None, [], "새 대화"
        
        # 이벤트 연결
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        # 새 대화 버튼
        new_chat_btn.click(
            create_new_conversation,
            outputs=[conversation_list, gr.State(), chatbot, chat_title]
        )
        
        # 대화 목록 선택
        conversation_list.change(
            switch_conversation,
            inputs=[conversation_list],
            outputs=[chatbot, chat_title]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI Stable ChatGPT-style application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_stable_chatgpt_interface()
    
    # 안정적인 실행 설정
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=20,
        show_api=False
    )

if __name__ == "__main__":
    main()
