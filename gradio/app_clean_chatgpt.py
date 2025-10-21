# -*- coding: utf-8 -*-
"""
LawFirmAI - ChatGPT 완전 동일 디자인
복잡한 컴포넌트 제거, 깔끔한 ChatGPT 스타일 구현
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
        logging.FileHandler('logs/clean_chatgpt_app.log')
    ]
)
logger = logging.getLogger(__name__)

class CleanChatGPTStyleLawFirmAI:
    """깔끔한 ChatGPT 스타일 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_service = None
        self.is_initialized = False
        
        # 대화 관리
        self.conversations = []
        self.current_conversation_id = None
        self.current_session_id = f"clean_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._initialize_components()
        self._create_new_conversation()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing clean ChatGPT-style components...")
            
            config = Config()
            self.chat_service = ChatService(config)
            
            self.is_initialized = True
            self.logger.info("Clean ChatGPT-style components initialized successfully")
            
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

def create_clean_chatgpt_interface():
    """깔끔한 ChatGPT 스타일 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = CleanChatGPTStyleLawFirmAI()
    
    # ChatGPT 완전 동일 CSS
    css = """
    /* ChatGPT 완전 동일 디자인 */
    .gradio-container {
        max-width: 100% !important;
        margin: 0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background: #212121 !important;
        min-height: 100vh !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* 메인 컨테이너 강제 설정 */
    .gradio-container > div {
        display: flex !important;
        height: 100vh !important;
        width: 100% !important;
    }
    
    .main-layout {
        display: flex !important;
        height: 100vh !important;
        width: 100% !important;
        flex-direction: row !important;
    }
    
    /* 왼쪽 사이드바 - ChatGPT 동일 */
    .sidebar-chatgpt {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        background: #171717 !important;
        color: #ececf1 !important;
        display: flex !important;
        flex-direction: column !important;
        border-right: 1px solid #2f2f2f !important;
        flex-shrink: 0 !important;
        position: relative !important;
        z-index: 10 !important;
        height: 100vh !important;
        overflow-y: auto !important;
    }
    
    /* 사이드바 컬럼 강제 표시 */
    .gradio-container .gradio-column:first-child {
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
        display: flex !important;
        flex-direction: column !important;
        background: #171717 !important;
        border-right: 1px solid #2f2f2f !important;
        flex-shrink: 0 !important;
    }
    
    .sidebar-header {
        padding: 16px !important;
        border-bottom: 1px solid #2f2f2f !important;
    }
    
    .sidebar-title {
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 0 !important;
        color: #ececf1 !important;
    }
    
    /* 새 대화 버튼 - ChatGPT 동일 */
    .new-chat-btn {
        width: calc(100% - 32px) !important;
        background: transparent !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
        padding: 12px 16px !important;
        border-radius: 6px !important;
        margin: 16px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    
    .new-chat-btn:hover {
        background: #2f2f2f !important;
    }
    
    /* 대화 목록 - ChatGPT 동일 */
    .conversation-list {
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 8px !important;
        max-height: calc(100vh - 200px) !important;
    }
    
    .conversation-item {
        padding: 12px 16px !important;
        margin: 2px 8px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        color: #ececf1 !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .conversation-item:hover {
        background: #2f2f2f !important;
    }
    
    .conversation-item.active {
        background: #2f2f2f !important;
    }
    
    .conversation-title {
        font-weight: 500 !important;
        margin-bottom: 4px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .conversation-meta {
        font-size: 12px !important;
        color: #8e8ea0 !important;
    }
    
    /* 메인 채팅 영역 - ChatGPT 동일 */
    .main-chat-area {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        background: #212121 !important;
        position: relative !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* 메인 채팅 컬럼 강제 설정 */
    .gradio-container .gradio-column:last-child {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        background: #212121 !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* 채팅 헤더 - ChatGPT 동일 */
    .chat-header {
        padding: 16px 24px !important;
        border-bottom: 1px solid #2f2f2f !important;
        background: #212121 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        flex-shrink: 0 !important;
    }
    
    .chat-title {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #ececf1 !important;
        margin: 0 !important;
    }
    
    .chat-actions {
        display: flex !important;
        gap: 8px !important;
    }
    
    .chat-action-btn {
        background: transparent !important;
        border: 1px solid #2f2f2f !important;
        color: #ececf1 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }
    
    .chat-action-btn:hover {
        background: #2f2f2f !important;
    }
    
    /* 채팅 메시지 영역 - ChatGPT 동일 */
    .chat-messages {
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 0 !important;
        background: #212121 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* 메시지 컨테이너 - ChatGPT 동일 */
    .message-container {
        display: flex !important;
        gap: 16px !important;
        align-items: flex-start !important;
        padding: 24px !important;
        border-bottom: 1px solid #2f2f2f !important;
    }
    
    .message-avatar {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        flex-shrink: 0 !important;
        font-weight: 600 !important;
    }
    
    .message-avatar.user {
        background: #5436da !important;
        color: white !important;
    }
    
    .message-avatar.assistant {
        background: #10a37f !important;
        color: white !important;
    }
    
    .message-content {
        flex: 1 !important;
        padding: 0 !important;
        line-height: 1.6 !important;
        font-size: 16px !important;
        color: #ececf1 !important;
        word-wrap: break-word !important;
    }
    
    /* 입력 영역 - ChatGPT 동일 */
    .chat-input-area {
        padding: 24px !important;
        background: #212121 !important;
        border-top: 1px solid #2f2f2f !important;
        flex-shrink: 0 !important;
    }
    
    .input-container {
        max-width: 768px !important;
        margin: 0 auto !important;
        position: relative !important;
    }
    
    .chat-input {
        width: 100% !important;
        border: 1px solid #2f2f2f !important;
        border-radius: 12px !important;
        padding: 12px 50px 12px 16px !important;
        font-size: 16px !important;
        background: #2f2f2f !important;
        resize: none !important;
        min-height: 24px !important;
        max-height: 200px !important;
        line-height: 1.5 !important;
        transition: all 0.2s ease !important;
        font-family: inherit !important;
        color: #ececf1 !important;
    }
    
    .chat-input:focus {
        outline: none !important;
        border-color: #5436da !important;
        box-shadow: 0 0 0 3px rgba(84,54,218,0.1) !important;
    }
    
    .chat-input::placeholder {
        color: #8e8ea0 !important;
    }
    
    .send-button {
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
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        font-size: 16px !important;
    }
    
    .send-button:hover {
        background: #4c2db8 !important;
    }
    
    .send-button:disabled {
        background: #2f2f2f !important;
        cursor: not-allowed !important;
    }
    
    /* 스크롤바 스타일링 - ChatGPT 동일 */
    .conversation-list::-webkit-scrollbar,
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .conversation-list::-webkit-scrollbar-track,
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .conversation-list::-webkit-scrollbar-thumb {
        background: #2f2f2f;
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #2f2f2f;
        border-radius: 3px;
    }
    
    .conversation-list::-webkit-scrollbar-thumb:hover,
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #4f4f4f;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .sidebar-chatgpt {
            position: fixed !important;
            left: -260px !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 1000 !important;
            transition: left 0.3s ease !important;
            box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
        }
        
        .sidebar-chatgpt.open {
            left: 0 !important;
        }
        
        .main-chat-area {
            width: 100% !important;
        }
        
        .chat-header {
            padding: 12px 16px !important;
        }
        
        .message-container {
            padding: 16px !important;
        }
        
        .chat-input-area {
            padding: 16px !important;
        }
        
        .message-container {
            gap: 12px !important;
        }
        
        .message-avatar {
            width: 28px !important;
            height: 28px !important;
            font-size: 14px !important;
        }
        
        .message-content {
            font-size: 14px !important;
        }
        
        .chat-input {
            font-size: 16px !important;
            padding: 10px 45px 10px 14px !important;
        }
        
        .send-button {
            width: 28px !important;
            height: 28px !important;
            font-size: 14px !important;
        }
    }
    
    /* 포커스 표시 개선 */
    button:focus, input:focus, textarea:focus, select:focus {
        outline: 2px solid #5436da !important;
        outline-offset: 2px !important;
    }
    
    /* 터치 디바이스 최적화 */
    @media (hover: none) and (pointer: coarse) {
        .new-chat-btn, .send-button, .chat-action-btn {
            min-height: 44px !important;
            padding: 12px 20px !important;
        }
        
        .chat-input {
            min-height: 44px !important;
        }
        
        .conversation-item {
            min-height: 44px !important;
            display: flex !important;
            align-items: center !important;
        }
    }
    """
    
    # HTML 헤드 (키보드 단축키 지원)
    head_html = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#212121">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-title" content="LawFirmAI">
    <script>
        // 키보드 단축키 지원
        document.addEventListener('keydown', function(e) {
            // Ctrl+N: 새 대화
            if (e.ctrlKey && e.key === 'n') {
                e.preventDefault();
                const newChatBtn = document.querySelector('.new-chat-btn');
                if (newChatBtn) newChatBtn.click();
            }
            
            // Escape: 사이드바 토글 (모바일)
            if (e.key === 'Escape') {
                const sidebar = document.querySelector('.sidebar-chatgpt');
                if (sidebar) {
                    sidebar.classList.remove('open');
                }
            }
        });
    </script>
    """
    
    with gr.Blocks(
        css=css,
        title="LawFirmAI - 법률 AI 어시스턴트",
        head=head_html
    ) as interface:
        
        # 메인 레이아웃
        with gr.Row(elem_classes=["main-layout"]):
            # 왼쪽 사이드바
            with gr.Column(scale=0, elem_classes=["sidebar-chatgpt"]):
                # 사이드바 헤더
                gr.HTML("""
                <div class="sidebar-header">
                    <h2 class="sidebar-title">⚖️ LawFirmAI</h2>
                </div>
                """)
                
                # 새 대화 버튼
                new_chat_btn = gr.Button("+ 새 대화", elem_classes=["new-chat-btn"])
                
                # 대화 목록
                conversation_list = gr.Radio(
                    choices=[],
                    value=None,
                    label="대화 목록",
                    elem_classes=["conversation-list"],
                    show_label=False
                )
            
            # 메인 채팅 영역
            with gr.Column(scale=1, elem_classes=["main-chat-area"]):
                # 채팅 헤더
                with gr.Row(elem_classes=["chat-header"]):
                    chat_title = gr.HTML("<h1 class='chat-title'>새 대화</h1>")
                    with gr.Row(elem_classes=["chat-actions"]):
                        clear_btn = gr.Button("🗑️ 삭제", elem_classes=["chat-action-btn"])
                
                # 채팅 메시지 영역
                chatbot = gr.Chatbot(
                    label="",
                    height=600,
                    show_label=False,
                    type="messages",
                    elem_classes=["chat-messages"],
                    container=False
                )
                
                # 입력 영역
                with gr.Column(elem_classes=["chat-input-area"]):
                    # 입력 컨테이너
                    with gr.Row(elem_classes=["input-container"]):
                        msg = gr.Textbox(
                            placeholder="메시지를 입력하세요...",
                            label="",
                            lines=1,
                            max_lines=10,
                            elem_classes=["chat-input"],
                            show_label=False
                        )
                        submit_btn = gr.Button("➤", elem_classes=["send-button"])
        
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
        
        # 대화 삭제
        clear_btn.click(
            delete_conversation,
            inputs=[conversation_list],
            outputs=[conversation_list, gr.State(), chatbot, chat_title]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI Clean ChatGPT-style application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_clean_chatgpt_interface()
    
    # 깔끔한 실행 설정
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
