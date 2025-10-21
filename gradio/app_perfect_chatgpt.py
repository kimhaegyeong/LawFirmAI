# -*- coding: utf-8 -*-
"""
LawFirmAI - 완전히 안정적인 ChatGPT 스타일 인터페이스
404 오류 완전 해결, JavaScript 오류 제거
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
        logging.FileHandler('logs/perfect_chatgpt_app.log')
    ]
)
logger = logging.getLogger(__name__)

class PerfectChatGPTStyleLawFirmAI:
    """완전히 안정적인 ChatGPT 스타일 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_service = None
        self.is_initialized = False
        
        # 대화 관리
        self.conversations = []
        self.current_conversation_id = None
        self.current_session_id = f"perfect_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._initialize_components()
        self._create_new_conversation()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing perfect ChatGPT-style components...")
            
            config = Config()
            self.chat_service = ChatService(config)
            
            self.is_initialized = True
            self.logger.info("Perfect ChatGPT-style components initialized successfully")
            
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

def create_perfect_chatgpt_interface():
    """완전히 안정적인 ChatGPT 스타일 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = PerfectChatGPTStyleLawFirmAI()
    
    # 현대적이고 세련된 CSS (JavaScript 오류 방지)
    css = """
    /* 현대적이고 세련된 ChatGPT 스타일 CSS */
    * {
        box-sizing: border-box;
    }
    
    body {
        margin: 0;
        padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ececf1;
        overflow: hidden;
    }
    
    .gradio-container {
        margin: 0 !important;
        padding: 0 !important;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        height: 100vh !important;
        display: flex !important;
        flex-direction: row !important;
        max-width: 100% !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* 배경 패턴 효과 */
    .gradio-container::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(84,54,218,0.1) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(16,163,127,0.1) 0%, transparent 50%) !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }
    
    /* 사이드바 - 글래스모피즘 효과 */
    .gradio-container > div:first-child {
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
        background: rgba(23,23,23,0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(47,47,47,0.3) !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        overflow-y: auto !important;
        flex-shrink: 0 !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.3) !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* 메인 영역 */
    .gradio-container > div:last-child {
        flex: 1 !important;
        background: transparent !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        overflow: hidden !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* 채팅 헤더 - 세련된 디자인 */
    .gradio-container > div:last-child > div:first-child {
        padding: 24px 32px !important;
        border-bottom: 1px solid rgba(47,47,47,0.3) !important;
        background: rgba(33,33,33,0.6) !important;
        backdrop-filter: blur(10px) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
    }
    
    .gradio-container > div:last-child > div:first-child h1 {
        color: #ececf1 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* 채팅 메시지 영역 */
    .gradio-container > div:last-child > div:nth-child(2) {
        flex: 1 !important;
        padding: 24px 32px !important;
        overflow-y: auto !important;
        background: transparent !important;
    }
    
    /* 입력 영역 - 플로팅 효과 */
    .gradio-container > div:last-child > div:last-child {
        padding: 24px 32px !important;
        background: rgba(33,33,33,0.6) !important;
        backdrop-filter: blur(10px) !important;
        border-top: 1px solid rgba(47,47,47,0.3) !important;
        position: relative !important;
    }
    
    /* 버튼 스타일 - 현대적인 디자인 */
    .gradio-container button {
        background: linear-gradient(135deg, #2f2f2f 0%, #3f3f3f 100%) !important;
        color: #ececf1 !important;
        border: 1px solid rgba(47,47,47,0.5) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .gradio-container button:hover {
        background: linear-gradient(135deg, #3f3f3f 0%, #4f4f4f 100%) !important;
        border-color: #5436da !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(84,54,218,0.3) !important;
    }
    
    /* 새 대화 버튼 - 프라이머리 스타일 */
    .gradio-container > div:first-child > div:first-child button {
        width: calc(100% - 32px) !important;
        margin: 20px 16px !important;
        background: linear-gradient(135deg, #5436da 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 24px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(84,54,218,0.4) !important;
    }
    
    .gradio-container > div:first-child > div:first-child button:hover {
        background: linear-gradient(135deg, #4c2db8 0%, #6d28d9 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(84,54,218,0.6) !important;
    }
    
    /* 대화 목록 - 카드 스타일 */
    .gradio-container > div:first-child > div:first-child > div:nth-child(3) {
        margin: 0 16px !important;
        background: rgba(47,47,47,0.3) !important;
        border-radius: 16px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(47,47,47,0.2) !important;
    }
    
    .gradio-container > div:first-child > div:first-child > div:nth-child(3) label {
        color: #ececf1 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* 텍스트박스 스타일 - 현대적인 입력 필드 */
    .gradio-container textarea {
        width: 100% !important;
        border: 2px solid rgba(47,47,47,0.3) !important;
        border-radius: 20px !important;
        padding: 16px 60px 16px 20px !important;
        font-size: 16px !important;
        background: rgba(47,47,47,0.4) !important;
        backdrop-filter: blur(10px) !important;
        color: #ececf1 !important;
        resize: none !important;
        min-height: 28px !important;
        max-height: 200px !important;
        line-height: 1.6 !important;
        font-family: inherit !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .gradio-container textarea:focus {
        outline: none !important;
        border-color: #5436da !important;
        box-shadow: 0 0 0 4px rgba(84,54,218,0.2) !important;
        background: rgba(47,47,47,0.6) !important;
        transform: scale(1.02) !important;
    }
    
    .gradio-container textarea::placeholder {
        color: #8e8ea0 !important;
        font-style: italic !important;
    }
    
    /* 전송 버튼 - 플로팅 액션 버튼 */
    .gradio-container > div:last-child > div:last-child button {
        position: absolute !important;
        right: 12px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #5436da 0%, #7c3aed 100%) !important;
        border: none !important;
        color: white !important;
        cursor: pointer !important;
        font-size: 18px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(84,54,218,0.4) !important;
    }
    
    .gradio-container > div:last-child > div:last-child button:hover {
        background: linear-gradient(135deg, #4c2db8 0%, #6d28d9 100%) !important;
        transform: translateY(-50%) scale(1.1) !important;
        box-shadow: 0 8px 25px rgba(84,54,218,0.6) !important;
    }
    
    /* 스크롤바 - 커스텀 디자인 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(47,47,47,0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #5436da 0%, #7c3aed 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #4c2db8 0%, #6d28d9 100%);
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .gradio-container {
            flex-direction: column !important;
        }
        
        .gradio-container > div:first-child {
            width: 100% !important;
            height: auto !important;
            border-right: none !important;
            border-bottom: 1px solid rgba(47,47,47,0.3) !important;
        }
        
        .gradio-container > div:last-child {
            height: calc(100vh - 200px) !important;
        }
        
        .gradio-container > div:first-child button {
            width: calc(100% - 40px) !important;
            margin: 15px auto !important;
        }
        
        .gradio-container .gradio-textbox {
            height: 150px !important;
        }
        
        .gradio-container .gradio-chatbot {
            padding: 15px !important;
        }
        
        .gradio-container .gradio-chatbot .message {
            max-width: 95% !important;
        }
        
        .gradio-container > div:last-child > div:last-child {
            padding: 15px !important;
        }
        
        .gradio-container textarea {
            min-height: 40px !important;
        }
        
        .gradio-container > div:last-child > div:last-child button {
            right: 20px !important;
            width: 36px !important;
            height: 36px !important;
            font-size: 16px !important;
        }
    }
    
    /* 채팅 메시지 스타일 - 현대적인 카드 */
    .gradio-container .message {
        margin: 20px 0 !important;
        padding: 20px !important;
        border-radius: 20px !important;
        max-width: 80% !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    }
    
    .gradio-container .message.user {
        background: linear-gradient(135deg, #5436da 0%, #7c3aed 100%) !important;
        color: white !important;
        margin-left: auto !important;
    }
    
    .gradio-container .message.assistant {
        background: rgba(47,47,47,0.6) !important;
        color: #ececf1 !important;
        margin-right: auto !important;
        border: 1px solid rgba(47,47,47,0.3) !important;
    }
    
    /* 로딩 애니메이션 - 개선된 스피너 */
    .gradio-container .loading {
        display: inline-block !important;
        width: 24px !important;
        height: 24px !important;
        border: 3px solid rgba(47,47,47,0.3) !important;
        border-radius: 50% !important;
        border-top-color: #5436da !important;
        animation: spin 1s cubic-bezier(0.4, 0, 0.2, 1) infinite !important;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* 타이포그래피 - 개선된 폰트 */
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
        color: #ececf1 !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
    }
    
    .gradio-container p {
        color: #8e8ea0 !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }
    
    /* 애니메이션 - 부드러운 전환 */
    .gradio-container * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* 다크 모드 최적화 */
    .gradio-container {
        color-scheme: dark !important;
    }
    
    /* 접근성 - 개선된 포커스 */
    .gradio-container button:focus,
    .gradio-container textarea:focus {
        outline: 3px solid rgba(84,54,218,0.5) !important;
        outline-offset: 2px !important;
    }
    
    /* 성능 최적화 */
    .gradio-container {
        will-change: auto !important;
        contain: layout style paint !important;
    }
    
    /* 브라우저 호환성 */
    .gradio-container {
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }
    
    /* 숨겨진 버튼들 - 오류 해결 */
    .gradio-container .gradio-share-button,
    .gradio-container .gradio-embed-button,
    .gradio-container .gradio-api-button {
        display: none !important;
    }
    
    /* 추가 시각적 효과 */
    .gradio-container button:active {
        transform: scale(0.98) !important;
    }
    
    /* 텍스트 선택 색상 */
    ::selection {
        background: rgba(84,54,218,0.3) !important;
        color: #ececf1 !important;
    }
    
    /* 포커스 링 */
    .gradio-container *:focus-visible {
        outline: 2px solid #5436da !important;
        outline-offset: 2px !important;
    }
    
    /* 폰트 임포트 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    """
    
    # JavaScript 오류 방지를 위한 개선된 HTML 헤드
    head_html = """
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#5436da">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-title" content="LawFirmAI">
    <style>
        /* 추가 안정성 스타일 */
        .gradio-container .gradio-share-button,
        .gradio-container .gradio-embed-button,
        .gradio-container .gradio-api-button {
            display: none !important;
        }
        
        /* JavaScript 오류 방지 */
        .gradio-container script[src*="share-modal"] {
            display: none !important;
        }
        
        /* PWA 관련 오류 방지 */
        .gradio-container link[rel="manifest"] {
            display: none !important;
        }
        
        /* 추가 안정성 */
        .gradio-container .gradio-share-button,
        .gradio-container .gradio-embed-button,
        .gradio-container .gradio-api-button {
            display: none !important;
        }
    </style>
    """
    
    with gr.Blocks(
        css=css,
        title="LawFirmAI - 법률 AI 어시스턴트",
        head=head_html,
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
                conversation_list = gr.Textbox(
                    value="",
                    label="대화 목록",
                    show_label=False,
                    interactive=False,
                    lines=10
                )
                
                # 현재 대화 ID 상태
                current_conversation = gr.State(value=None)
            
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
            try:
                print("DEBUG: 새 대화 생성 시작")
                conversation_id = app._create_new_conversation()
                print(f"DEBUG: 생성된 대화 ID: {conversation_id}")
                
                conversation_list = app.get_conversation_list()
                print(f"DEBUG: 대화 목록 개수: {len(conversation_list)}")
                
                # 대화 목록을 텍스트 형식으로 변환
                conversation_text = "\n".join([f"• {conv['title']}" for conv in conversation_list])
                print(f"DEBUG: 대화 목록 텍스트: {conversation_text}")
                
                result = conversation_text, conversation_id, [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>새 대화</h1>"
                print(f"DEBUG: 반환값: {result}")
                return result
                
            except Exception as e:
                print(f"ERROR: 새 대화 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return "오류 발생", None, [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>오류 발생</h1>"
        
        def switch_conversation(conversation_id):
            """대화 전환"""
            try:
                print(f"DEBUG: 대화 전환 시작 - ID: {conversation_id}")
                
                if not conversation_id:
                    print("DEBUG: 대화 ID가 없음, 새 대화로 설정")
                    return [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>새 대화</h1>"
                
                messages = app.switch_conversation(conversation_id)
                print(f"DEBUG: 전환된 메시지 개수: {len(messages) if messages else 0}")
                
                conversation_list = app.get_conversation_list()
                
                # 현재 대화 찾기
                current_conv = None
                for conv in conversation_list:
                    if conv["id"] == conversation_id:
                        current_conv = conv
                        break
                
                title = current_conv["title"] if current_conv else "대화"
                print(f"DEBUG: 대화 제목: {title}")
                
                result = messages, f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>{title}</h1>"
                print(f"DEBUG: 대화 전환 결과: {result}")
                return result
                
            except Exception as e:
                print(f"ERROR: 대화 전환 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>오류 발생</h1>"
        
        def delete_conversation(conversation_id):
            """대화 삭제"""
            try:
                print(f"DEBUG: 대화 삭제 시작 - ID: {conversation_id}")
                
                if conversation_id:
                    app.delete_conversation(conversation_id)
                    conversation_list = app.get_conversation_list()
                    
                    # 대화 목록을 텍스트 형식으로 변환
                    conversation_text = "\n".join([f"• {conv['title']}" for conv in conversation_list])
                    print(f"DEBUG: 삭제 후 대화 목록: {conversation_text}")
                    
                    if conversation_list:
                        current_id = conversation_list[0]["id"]
                        messages = app.switch_conversation(current_id)
                        title = conversation_list[0]["title"]
                    else:
                        current_id = None
                        messages = []
                        title = "새 대화"
                    
                    result = conversation_text, current_id, messages, f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>{title}</h1>"
                    print(f"DEBUG: 삭제 결과: {result}")
                    return result
                
                return "", None, [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>새 대화</h1>"
                
            except Exception as e:
                print(f"ERROR: 대화 삭제 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return "오류 발생", None, [], f"<h1 style='color: #ececf1; font-size: 18px; font-weight: 600; margin: 0;'>오류 발생</h1>"
        
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
        def debug_create_new_conversation():
            print("DEBUG: 새 대화 버튼 클릭됨")
            result = create_new_conversation()
            print(f"DEBUG: 새 대화 생성 결과: {result}")
            return result
        
        new_chat_btn.click(
            debug_create_new_conversation,
            outputs=[conversation_list, current_conversation, chatbot, chat_title]
        )
        
        # 대화 목록은 이제 읽기 전용이므로 이벤트 핸들러 제거
        # conversation_list.change(
        #     switch_conversation,
        #     inputs=[conversation_list],
        #     outputs=[chatbot, chat_title]
        # )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI Perfect ChatGPT-style application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_perfect_chatgpt_interface()
    
    # 완전히 안정적인 실행 설정 (404 오류 방지)
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=20
    )

if __name__ == "__main__":
    main()
