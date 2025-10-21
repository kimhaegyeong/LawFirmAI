# -*- coding: utf-8 -*-
"""
LawFirmAI - ChatGPT 스타일 UI/UX 개선
기존 디자인 스타일을 유지하면서 ChatGPT의 구성요소를 적용한 인터페이스
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
        logging.FileHandler('logs/chatgpt_style_app.log')
    ]
)
logger = logging.getLogger(__name__)

class ChatGPTStyleLawFirmAI:
    """ChatGPT 스타일 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_service = None
        self.is_initialized = False
        
        # 대화 관리
        self.conversations = []
        self.current_conversation_id = None
        self.current_session_id = f"chatgpt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 사용자 설정
        self.user_type = "일반인"
        self.interest_areas = []
        
        self._initialize_components()
        self._create_new_conversation()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing ChatGPT-style components...")
            
            config = Config()
            self.chat_service = ChatService(config)
            
            self.is_initialized = True
            self.logger.info("ChatGPT-style components initialized successfully")
            
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
    
    def update_user_settings(self, user_type: str, interest_areas: List[str]):
        """사용자 설정 업데이트"""
        self.user_type = user_type
        self.interest_areas = interest_areas
        self.logger.info(f"User settings updated: {user_type}, {interest_areas}")
        return "설정이 업데이트되었습니다."

def create_chatgpt_style_interface():
    """ChatGPT 스타일 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = ChatGPTStyleLawFirmAI()
    
    # ChatGPT 스타일 CSS
    css = """
    /* ChatGPT 스타일 CSS - 기존 디자인 유지하면서 개선 */
    .gradio-container {
        max-width: 100% !important;
        margin: 0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: #f7f7f8 !important;
        min-height: 100vh !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* 메인 레이아웃 - ChatGPT 스타일 */
    .main-layout {
        display: flex !important;
        height: 100vh !important;
        width: 100% !important;
    }
    
    /* 왼쪽 사이드바 - ChatGPT 스타일 */
    .sidebar-chatgpt {
        width: 260px !important;
        background: #202123 !important;
        color: white !important;
        display: flex !important;
        flex-direction: column !important;
        border-right: 1px solid #4d4d4f !important;
        flex-shrink: 0 !important;
    }
    
    .sidebar-header {
        padding: 20px 16px !important;
        border-bottom: 1px solid #4d4d4f !important;
    }
    
    .sidebar-title {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        color: white !important;
    }
    
    .new-chat-btn {
        width: 100% !important;
        background: transparent !important;
        border: 1px solid #4d4d4f !important;
        color: white !important;
        padding: 12px 16px !important;
        border-radius: 6px !important;
        margin-top: 16px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    .new-chat-btn:hover {
        background: #2a2b32 !important;
    }
    
    /* 대화 목록 */
    .conversation-list {
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 8px !important;
    }
    
    .conversation-item {
        padding: 12px 16px !important;
        margin: 4px 0 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        color: #ececf1 !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
    }
    
    .conversation-item:hover {
        background: #2a2b32 !important;
    }
    
    .conversation-item.active {
        background: #343541 !important;
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
    
    /* 메인 채팅 영역 */
    .main-chat-area {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        background: #ffffff !important;
        position: relative !important;
    }
    
    /* 채팅 헤더 */
    .chat-header {
        padding: 16px 24px !important;
        border-bottom: 1px solid #e5e5e5 !important;
        background: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .chat-title {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin: 0 !important;
    }
    
    .chat-actions {
        display: flex !important;
        gap: 8px !important;
    }
    
    .chat-action-btn {
        background: transparent !important;
        border: 1px solid #d1d5db !important;
        color: #374151 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    .chat-action-btn:hover {
        background: #f3f4f6 !important;
    }
    
    /* 채팅 메시지 영역 */
    .chat-messages {
        flex: 1 !important;
        overflow-y: auto !important;
        padding: 24px !important;
        background: white !important;
    }
    
    .message-container {
        margin-bottom: 24px !important;
        display: flex !important;
        gap: 16px !important;
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
    }
    
    .message-avatar.user {
        background: #007bff !important;
        color: white !important;
    }
    
    .message-avatar.assistant {
        background: #10b981 !important;
        color: white !important;
    }
    
    .message-content {
        flex: 1 !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        background: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        line-height: 1.6 !important;
        font-size: 14px !important;
        color: #2c3e50 !important;
    }
    
    .message-content.user {
        background: #007bff !important;
        color: white !important;
        border-color: #007bff !important;
    }
    
    /* 입력 영역 - ChatGPT 스타일 */
    .chat-input-area {
        padding: 24px !important;
        background: white !important;
        border-top: 1px solid #e5e5e5 !important;
    }
    
    .input-container {
        max-width: 768px !important;
        margin: 0 auto !important;
        position: relative !important;
    }
    
    .chat-input {
        width: 100% !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        padding: 12px 50px 12px 16px !important;
        font-size: 16px !important;
        background: white !important;
        resize: none !important;
        min-height: 24px !important;
        max-height: 200px !important;
        line-height: 1.5 !important;
        transition: all 0.2s ease !important;
    }
    
    .chat-input:focus {
        outline: none !important;
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
    }
    
    .send-button {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 6px !important;
        background: #007bff !important;
        border: none !important;
        color: white !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
    }
    
    .send-button:hover {
        background: #0056b3 !important;
    }
    
    .send-button:disabled {
        background: #d1d5db !important;
        cursor: not-allowed !important;
    }
    
    /* 빠른 질문 버튼들 */
    .quick-questions {
        display: flex !important;
        gap: 8px !important;
        margin-bottom: 16px !important;
        flex-wrap: wrap !important;
    }
    
    .quick-question-btn {
        background: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        color: #495057 !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        cursor: pointer !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }
    
    .quick-question-btn:hover {
        background: #e9ecef !important;
        border-color: #007bff !important;
        color: #007bff !important;
    }
    
    /* 설정 패널 */
    .settings-panel {
        background: #f8f9fa !important;
        border-top: 1px solid #e5e5e5 !important;
        padding: 16px !important;
    }
    
    .settings-content {
        max-width: 768px !important;
        margin: 0 auto !important;
    }
    
    .settings-section {
        margin-bottom: 20px !important;
    }
    
    .settings-label {
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 8px !important;
        font-size: 14px !important;
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
        
        .chat-messages {
            padding: 16px !important;
        }
        
        .chat-input-area {
            padding: 16px !important;
        }
        
        .message-container {
            margin-bottom: 16px !important;
        }
        
        .quick-questions {
            justify-content: center !important;
        }
    }
    
    /* 다크 모드 지원 */
    @media (prefers-color-scheme: dark) {
        .main-chat-area {
            background: #343541 !important;
            color: #ececf1 !important;
        }
        
        .chat-header {
            background: #343541 !important;
            border-color: #4d4d4f !important;
        }
        
        .chat-title {
            color: #ececf1 !important;
        }
        
        .chat-messages {
            background: #343541 !important;
        }
        
        .message-content {
            background: #444654 !important;
            border-color: #4d4d4f !important;
            color: #ececf1 !important;
        }
        
        .chat-input-area {
            background: #343541 !important;
            border-color: #4d4d4f !important;
        }
        
        .chat-input {
            background: #40414f !important;
            border-color: #4d4d4f !important;
            color: #ececf1 !important;
        }
        
        .settings-panel {
            background: #2a2b32 !important;
            border-color: #4d4d4f !important;
        }
    }
    
    /* 스크롤바 스타일링 */
    .conversation-list::-webkit-scrollbar,
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .conversation-list::-webkit-scrollbar-track,
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .conversation-list::-webkit-scrollbar-thumb {
        background: #4d4d4f;
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 3px;
    }
    
    .conversation-list::-webkit-scrollbar-thumb:hover,
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    """
    
    # HTML 헤드
    head_html = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#007bff">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-title" content="LawFirmAI">
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
                
                # 사이드바 하단 설정
                with gr.Accordion("⚙️ 설정", open=False):
                    user_type = gr.Radio(
                        choices=["일반인", "법무팀", "변호사"],
                        value="일반인",
                        label="사용자 유형"
                    )
                    interest_area = gr.Dropdown(
                        choices=["민법", "형법", "상법", "근로기준법", "부동산", "금융"],
                        multiselect=True,
                        label="관심 분야"
                    )
            
            # 메인 채팅 영역
            with gr.Column(scale=1, elem_classes=["main-chat-area"]):
                # 채팅 헤더
                with gr.Row(elem_classes=["chat-header"]):
                    chat_title = gr.HTML("<h1 class='chat-title'>새 대화</h1>")
                    with gr.Row(elem_classes=["chat-actions"]):
                        clear_btn = gr.Button("🗑️ 삭제", elem_classes=["chat-action-btn"])
                        settings_btn = gr.Button("⚙️ 설정", elem_classes=["chat-action-btn"])
                
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
                    # 빠른 질문 버튼들
                    with gr.Row(elem_classes=["quick-questions"]):
                        quick_questions = [
                            gr.Button("📄 계약서 검토", elem_classes=["quick-question-btn"]),
                            gr.Button("🔍 판례 검색", elem_classes=["quick-question-btn"]),
                            gr.Button("📚 법령 조회", elem_classes=["quick-question-btn"]),
                            gr.Button("💼 법률 상담", elem_classes=["quick-question-btn"])
                        ]
                    
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
        
        def update_settings(user_type, interest_areas):
            """사용자 설정 업데이트"""
            return app.update_user_settings(user_type, interest_areas)
        
        def quick_question_handler(question_type):
            """빠른 질문 처리"""
            questions = {
                "계약서 검토": "계약서 작성 시 주의사항은 무엇인가요?",
                "판례 검색": "관련 판례를 검색해주세요",
                "법령 조회": "해당 법령의 내용을 알려주세요",
                "법률 상담": "법률 상담을 받고 싶습니다"
            }
            return questions.get(question_type, "질문을 입력해주세요.")
        
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
        
        # 빠른 질문 버튼들
        for i, quick_btn in enumerate(quick_questions):
            question_types = ["계약서 검토", "판례 검색", "법령 조회", "법률 상담"]
            quick_btn.click(
                lambda x, i=i: quick_question_handler(question_types[i]),
                outputs=[msg]
            )
        
        # 사용자 설정 업데이트
        user_type.change(
            update_settings,
            inputs=[user_type, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
        
        interest_area.change(
            update_settings,
            inputs=[user_type, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI ChatGPT-style application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_chatgpt_style_interface()
    
    # ChatGPT 스타일 실행 설정
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
