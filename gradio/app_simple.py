# -*- coding: utf-8 -*-
"""
LawFirmAI - 간소화된 사용자 친화적 인터페이스
일반 사용자를 위한 직관적이고 단순한 Gradio 애플리케이션
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
        logging.FileHandler('logs/simple_app.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleLawFirmAI:
    """간소화된 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chat_service = None
        self.is_initialized = False
        self.current_session_id = f"simple_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 사용자 설정
        self.user_type = "일반인"
        self.interest_areas = []
        
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화 (간소화)"""
        try:
            self.logger.info("Initializing simple components...")
            
            # 필수 컴포넌트만 초기화
            config = Config()
            self.chat_service = ChatService(config)
            
            self.is_initialized = True
            self.logger.info("Simple components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.is_initialized = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """질의 처리 (간소화)"""
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
    
    def update_user_settings(self, user_type: str, interest_areas: List[str]):
        """사용자 설정 업데이트"""
        self.user_type = user_type
        self.interest_areas = interest_areas
        self.logger.info(f"User settings updated: {user_type}, {interest_areas}")
        return "설정이 업데이트되었습니다."

def create_simple_interface():
    """간소화된 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = SimpleLawFirmAI()
    
    # 간소화된 CSS
    css = """
    /* 간소화된 CSS - 이미지 깨짐 방지 */
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: #f8f9fa !important;
        min-height: 100vh !important;
        padding: 20px !important;
    }
    
    /* 헤더 스타일 */
    .simple-header {
        text-align: center !important;
        padding: 30px 20px !important;
        background: white !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    .simple-header h1 {
        font-size: 2.5rem !important;
        color: #2c3e50 !important;
        margin: 0 !important;
        font-weight: 700 !important;
    }
    
    .simple-header p {
        color: #6c757d !important;
        margin: 10px 0 0 0 !important;
        font-size: 1.1rem !important;
    }
    
    /* 채팅 인터페이스 */
    .simple-chatbot {
        border: 2px solid #dee2e6 !important;
        border-radius: 15px !important;
        background: white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        height: 500px !important;
    }
    
    /* 입력 영역 */
    .simple-input {
        border: 2px solid #dee2e6 !important;
        border-radius: 12px !important;
        padding: 15px 20px !important;
        font-size: 16px !important;
        transition: border-color 0.3s ease !important;
        background: white !important;
        resize: none !important;
    }
    
    .simple-input:focus {
        border-color: #007bff !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
    }
    
    /* 버튼 스타일 */
    .submit-btn {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        background: #007bff !important;
        color: white !important;
        padding: 15px 30px !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 16px !important;
    }
    
    .submit-btn:hover {
        background: #0056b3 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3) !important;
    }
    
    /* 빠른 질문 버튼 */
    .quick-btn {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        background: #6c757d !important;
        color: white !important;
        padding: 10px 20px !important;
        border: none !important;
        cursor: pointer !important;
        margin: 5px !important;
        font-size: 14px !important;
    }
    
    .quick-btn:hover {
        background: #545b62 !important;
        transform: translateY(-1px) !important;
    }
    
    /* 사이드바 */
    .simple-sidebar {
        background: white !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* 설정 섹션 */
    .settings-section {
        margin-bottom: 20px !important;
        padding: 15px !important;
        background: #f8f9fa !important;
        border-radius: 10px !important;
    }
    
    .settings-section h3 {
        margin: 0 0 10px 0 !important;
        color: #495057 !important;
        font-size: 1.1rem !important;
    }
    
    /* 도움말 섹션 */
    .help-section {
        margin-top: 20px !important;
        padding: 15px !important;
        background: #e3f2fd !important;
        border-radius: 10px !important;
        border-left: 4px solid #2196f3 !important;
    }
    
    .help-section h3 {
        margin: 0 0 10px 0 !important;
        color: #1976d2 !important;
        font-size: 1.1rem !important;
    }
    
    .help-section ul {
        margin: 0 !important;
        padding-left: 20px !important;
    }
    
    .help-section li {
        margin: 5px 0 !important;
        color: #424242 !important;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        
        .simple-header h1 {
            font-size: 1.8rem !important;
        }
        
        .simple-chatbot {
            height: 400px !important;
        }
        
        .simple-sidebar {
            margin-top: 15px !important;
            padding: 15px !important;
        }
        
        .submit-btn, .quick-btn {
            min-height: 44px !important;
            font-size: 16px !important;
        }
        
        .simple-input {
            font-size: 16px !important; /* iOS 줌 방지 */
        }
    }
    
    /* 다크 모드 지원 */
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        .simple-header, .simple-sidebar {
            background: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        .simple-chatbot {
            background: #2d2d2d !important;
            border-color: #404040 !important;
        }
        
        .simple-input {
            background: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #404040 !important;
        }
        
        .settings-section {
            background: #3d3d3d !important;
        }
        
        .help-section {
            background: #1e3a5f !important;
        }
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
        
        # 헤더
        gr.HTML("""
        <div class="simple-header">
            <h1>⚖️ LawFirmAI</h1>
            <p>법률 질문에 답변해드립니다</p>
        </div>
        """)
        
        # 메인 레이아웃
        with gr.Row():
            # 메인 채팅 영역 (80%)
            with gr.Column(scale=4):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    show_label=False,
                    type="messages",
                    elem_classes=["simple-chatbot"]
                )
                
                # 입력 영역
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="법률 관련 질문을 입력하세요... (예: 계약서 검토, 판례 검색, 법령 해석)",
                        label="",
                        scale=4,
                        lines=2,
                        max_lines=4,
                        elem_classes=["simple-input"]
                    )
                    submit_btn = gr.Button("전송", scale=1, variant="primary", elem_classes=["submit-btn"])
                
                # 빠른 질문 버튼들
                with gr.Row():
                    quick_questions = [
                        gr.Button("📄 계약서 검토", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("🔍 판례 검색", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("📚 법령 조회", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("💼 법률 상담", variant="secondary", elem_classes=["quick-btn"])
                    ]
                
                # 예시 질문
                with gr.Accordion("💡 예시 질문", open=False):
                    gr.Examples(
                        examples=[
                            "계약 해제 조건이 무엇인가요?",
                            "손해배상 관련 최신 판례를 찾아주세요",
                            "불법행위의 법적 근거를 알려주세요",
                            "이혼 절차는 어떻게 진행하나요?",
                            "민법 제750조의 내용이 무엇인가요?"
                        ],
                        inputs=msg,
                        label=""
                    )
            
            # 간소화된 사이드바 (20%)
            with gr.Column(scale=1, elem_classes=["simple-sidebar"]):
                # 사용자 설정
                with gr.Accordion("⚙️ 설정", open=False):
                    user_type = gr.Radio(
                        choices=["일반인", "법무팀", "변호사"],
                        value="일반인",
                        label="사용자 유형",
                        info="선택한 유형에 따라 답변의 전문성이 조정됩니다"
                    )
                    interest_area = gr.Dropdown(
                        choices=["민법", "형법", "상법", "근로기준법", "부동산", "금융"],
                        multiselect=True,
                        label="관심 분야 (선택사항)",
                        info="관심 있는 법률 분야를 선택하세요"
                    )
                
                # 도움말
                with gr.Accordion("❓ 도움말", open=False):
                    gr.HTML("""
                    <div class="help-section">
                        <h3>사용법</h3>
                        <ul>
                            <li>질문을 입력하고 전송 버튼을 클릭하세요</li>
                            <li>빠른 질문 버튼을 활용해보세요</li>
                            <li>구체적인 질문을 하시면 더 정확한 답변을 받을 수 있습니다</li>
                            <li>사용자 유형을 설정하여 맞춤형 답변을 받으세요</li>
                        </ul>
                    </div>
                    """)
                
                # 세션 정보
                with gr.Accordion("📊 세션 정보", open=False):
                    session_info = gr.JSON(label="", show_label=False)
        
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
            
            # 세션 정보 업데이트
            session_data = {
                "세션 ID": result.get("session_id", ""),
                "처리 시간": f"{result.get('processing_time', 0):.2f}초",
                "신뢰도": f"{result.get('confidence', 0):.1%}",
                "질문 유형": result.get("question_type", "일반")
            }
            
            return history, "", session_data
        
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
            outputs=[chatbot, msg, session_info]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, session_info]
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
    logger.info("Starting LawFirmAI Simple application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_simple_interface()
    
    # 간소화된 실행 설정
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
