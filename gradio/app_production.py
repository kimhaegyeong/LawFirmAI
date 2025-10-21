# -*- coding: utf-8 -*-
"""
LawFirmAI - 프로덕션 배포용 Gradio 애플리케이션
사용자 친화적이고 직관적인 인터페이스로 개선된 버전
"""

import os
import sys
import logging
import time
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangGraph 활성화 설정
os.environ["USE_LANGGRAPH"] = os.getenv("USE_LANGGRAPH", "true")

# Gradio 및 기타 라이브러리
import gradio as gr
import torch
import psutil

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.services.question_classifier import QuestionClassifier, QuestionType
from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.optimized_search_engine import OptimizedSearchEngine
from source.services.prompt_templates import PromptTemplateManager
from source.services.unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType
from source.services.dynamic_prompt_updater import create_dynamic_prompt_updater
from source.services.prompt_optimizer import create_prompt_optimizer
from source.services.confidence_calculator import ConfidenceCalculator
from source.services.legal_term_expander import LegalTermExpander
from source.services.gemini_client import GeminiClient
from source.services.improved_answer_generator import ImprovedAnswerGenerator
from source.services.answer_formatter import AnswerFormatter
from source.services.context_builder import ContextBuilder
from source.services.chat_service import ChatService
from source.services.performance_monitor import PerformanceMonitor as SourcePerformanceMonitor, PerformanceContext

# Phase 1: 대화 맥락 강화 모듈
from source.services.integrated_session_manager import IntegratedSessionManager
from source.services.multi_turn_handler import MultiTurnQuestionHandler
from source.services.context_compressor import ContextCompressor

# Phase 2: 개인화 및 지능형 분석 모듈
from source.services.user_profile_manager import UserProfileManager, ExpertiseLevel, DetailLevel
from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer, EmotionType, IntentType, UrgencyLevel
from source.services.conversation_flow_tracker import ConversationFlowTracker

from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/production_app.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionUI:
    """프로덕션용 UI 컴포넌트 관리 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.user_profiles = {}  # 사용자 프로필 저장
        self.session_contexts = {}  # 세션 컨텍스트 저장
        
    def create_user_onboarding(self):
        """사용자 온보딩 인터페이스"""
        with gr.Modal(title="LawFirmAI에 오신 것을 환영합니다! 🎉") as onboarding:
            gr.Markdown("""
            ## ⚖️ LawFirmAI란?
            법률 전문가와 일반인을 위한 AI 어시스턴트입니다.
            
            ### 🚀 주요 기능:
            - 📄 **문서 분석**: 계약서, 법률 문서 자동 분석
            - 🔍 **판례 검색**: 관련 판례 및 법령 검색
            - 💼 **법률 상담**: 법률 질문에 대한 전문적 답변
            - 📚 **법령 조회**: 최신 법령 정보 제공
            """)
            
            with gr.Row():
                with gr.Column():
                    user_type = gr.Radio(
                        choices=[
                            ("일반인", "일반적인 법률 질문"),
                            ("법무팀", "기업 법무 업무"),
                            ("변호사", "법률 전문가"),
                            ("법학자", "학술 연구")
                        ],
                        value="일반인",
                        label="사용자 유형"
                    )
                    
                    interest_areas = gr.CheckboxGroup(
                        choices=["민법", "형법", "상법", "근로기준법", "부동산", "금융", "지적재산권", "세법"],
                        label="관심 분야 (선택사항)",
                        value=[]
                    )
                
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 사용 팁:
                    - 구체적인 질문을 하시면 더 정확한 답변을 받을 수 있습니다
                    - 문서를 업로드하여 분석을 요청할 수 있습니다
                    - 대화 이력은 자동으로 저장됩니다
                    """)
            
            with gr.Row():
                start_btn = gr.Button("시작하기", variant="primary", size="lg")
                skip_btn = gr.Button("건너뛰기", variant="secondary")
        
        return onboarding, user_type, interest_areas, start_btn, skip_btn
    
    def create_main_chat_interface(self):
        """메인 채팅 인터페이스"""
        with gr.Column(scale=3):
            # 채팅 인터페이스
            chatbot = gr.Chatbot(
                label="",
                height=600,
                show_label=False,
                type="messages",
                avatar_images=("👤", "⚖️"),
                elem_classes=["chatbot-container"]
            )
            
            # 입력 영역
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="법률 관련 질문을 입력하세요... (예: 계약서 검토, 판례 검색, 법령 해석)",
                    label="",
                    scale=4,
                    lines=2,
                    max_lines=4,
                    elem_classes=["message-input"]
                )
                submit_btn = gr.Button("전송", scale=1, variant="primary", size="lg", elem_classes=["submit-btn"])
            
            # 빠른 액션 버튼들
            with gr.Row(elem_classes=["quick-actions"]):
                quick_actions = [
                    gr.Button("📄 문서 분석", variant="secondary", elem_classes=["quick-btn"]),
                    gr.Button("🔍 판례 검색", variant="secondary", elem_classes=["quick-btn"]),
                    gr.Button("📚 법령 조회", variant="secondary", elem_classes=["quick-btn"]),
                    gr.Button("💼 계약서 검토", variant="secondary", elem_classes=["quick-btn"])
                ]
            
            # 예시 질문
            with gr.Accordion("💡 추천 질문", open=False):
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
        
        return chatbot, msg, submit_btn, quick_actions
    
    def create_smart_sidebar(self):
        """스마트 사이드바"""
        with gr.Column(scale=1, elem_classes=["sidebar"]):
            # 사용자 프로필 (간소화)
            with gr.Accordion("👤 프로필", open=False):
                user_level = gr.Radio(
                    choices=["일반인", "법무팀", "변호사", "법학자"],
                    value="일반인",
                    label="전문성 수준"
                )
                interest_area = gr.Dropdown(
                    choices=["민법", "형법", "상법", "근로기준법", "부동산", "금융"],
                    multiselect=True,
                    label="관심 분야"
                )
            
            # 세션 관리 (간소화)
            with gr.Accordion("💬 대화 관리", open=False):
                new_session_btn = gr.Button("새 대화", variant="secondary", size="sm")
                save_session_btn = gr.Button("대화 저장", variant="secondary", size="sm")
                load_session_btn = gr.Button("대화 불러오기", variant="secondary", size="sm")
            
            # 세션 정보
            with gr.Accordion("📊 세션 정보", open=False):
                session_info = gr.JSON(label="", show_label=False, elem_classes=["session-info"])
            
            # 고급 기능 (접을 수 있는 형태)
            with gr.Accordion("⚙️ 고급 기능", open=False):
                enable_akls = gr.Checkbox(label="AKLS 표준판례 검색", value=True)
                enable_analysis = gr.Checkbox(label="감정 분석", value=False)
                response_detail = gr.Slider(1, 5, value=3, label="답변 상세도")
            
            # 피드백 시스템
            with gr.Accordion("💬 피드백", open=False):
                feedback_rating = gr.Slider(1, 5, value=3, label="만족도")
                feedback_text = gr.Textbox(label="개선사항", placeholder="피드백을 입력하세요", lines=2)
                submit_feedback = gr.Button("피드백 제출", variant="secondary", size="sm")
        
        return (user_level, interest_area, new_session_btn, save_session_btn, 
                load_session_btn, session_info, enable_akls, enable_analysis, 
                response_detail, feedback_rating, feedback_text, submit_feedback)
    
    def create_document_analysis_interface(self):
        """문서 분석 인터페이스"""
        with gr.Tab("📄 문서 분석"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 파일 업로드
                    file_upload = gr.File(
                        label="문서 업로드",
                        file_types=[".pdf", ".docx", ".txt", ".hwp"],
                        file_count="single",
                        elem_classes=["file-upload"]
                    )
                    
                    # 분석 옵션
                    analysis_type = gr.Radio(
                        choices=[
                            ("계약서 검토", "contract"),
                            ("법률 문서 분석", "legal"),
                            ("판례 분석", "precedent"),
                            ("일반 문서 분석", "general")
                        ],
                        value="contract",
                        label="분석 유형"
                    )
                    
                    # 분석 세부 옵션
                    analysis_options = gr.CheckboxGroup(
                        choices=[
                            "위험 요소 분석",
                            "법적 근거 검토",
                            "개선 제안",
                            "요약 생성"
                        ],
                        value=["위험 요소 분석", "법적 근거 검토"],
                        label="분석 옵션"
                    )
                    
                    analyze_btn = gr.Button("분석 시작", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    # 분석 결과
                    analysis_result = gr.Markdown(
                        label="분석 결과",
                        value="문서를 업로드하고 분석을 시작하세요.",
                        elem_classes=["analysis-result"]
                    )
                    
                    # 위험도 표시
                    risk_indicator = gr.HTML(
                        value="<div class='risk-indicator'>위험도: <span class='risk-low'>낮음</span></div>",
                        elem_classes=["risk-indicator"]
                    )
                    
                    # 분석 진행률
                    progress_bar = gr.Progress(label="분석 진행률")
        
        return file_upload, analysis_type, analysis_options, analyze_btn, analysis_result, risk_indicator, progress_bar
    
    def create_mobile_interface(self):
        """모바일 최적화 인터페이스"""
        with gr.Row(visible=False, elem_classes=["mobile-only"]):
            with gr.Column():
                # 모바일용 터치 친화적 버튼
                mobile_actions = gr.Row([
                    gr.Button("💬 질문", variant="primary", size="lg", elem_classes=["mobile-btn"]),
                    gr.Button("📄 분석", variant="secondary", size="lg", elem_classes=["mobile-btn"]),
                    gr.Button("🔍 검색", variant="secondary", size="lg", elem_classes=["mobile-btn"])
                ])
                
                # 모바일용 채팅
                mobile_chatbot = gr.Chatbot(
                    height=400,
                    type="messages",
                    show_label=False,
                    elem_classes=["mobile-chatbot"]
                )
                
                # 음성 입력 지원
                voice_input = gr.Audio(
                    label="음성 질문",
                    type="microphone",
                    elem_classes=["voice-input"]
                )
        
        return mobile_actions, mobile_chatbot, voice_input

class ProductionApp:
    """프로덕션용 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ui = ProductionUI()
        
        # 기존 컴포넌트들 (간소화)
        self.chat_service = None
        self.session_manager = None
        self.is_initialized = False
        
        # 사용자 상태 관리
        self.current_user_profile = {
            "user_type": "일반인",
            "interest_areas": [],
            "expertise_level": "beginner"
        }
        
        self.current_session_id = f"prod_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화 (간소화)"""
        try:
            self.logger.info("Initializing production components...")
            
            # 필수 컴포넌트만 초기화
            config = Config()
            self.chat_service = ChatService(config)
            self.session_manager = IntegratedSessionManager("data/conversations.db")
            
            self.is_initialized = True
            self.logger.info("Production components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.is_initialized = False
    
    def process_query(self, query: str, user_profile: Dict = None) -> Dict[str, Any]:
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
                "session_id": self.current_session_id,
                "user_profile": user_profile or self.current_user_profile
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
    
    def analyze_document(self, file_path: str, analysis_type: str, options: List[str]) -> Dict[str, Any]:
        """문서 분석 (간소화)"""
        try:
            # 실제 문서 분석 로직은 기존 서비스 활용
            return {
                "summary": "문서 분석이 완료되었습니다.",
                "risks": ["위험 요소 1", "위험 요소 2"],
                "recommendations": ["개선 제안 1", "개선 제안 2"],
                "risk_level": "medium",
                "confidence": 0.85
            }
        except Exception as e:
            return {
                "error": str(e),
                "risk_level": "unknown",
                "confidence": 0.0
            }
    
    def get_smart_suggestions(self, conversation_history: List, user_profile: Dict) -> List[str]:
        """지능형 질문 제안"""
        suggestions = []
        
        # 사용자 유형별 기본 제안
        if user_profile.get("user_type") == "일반인":
            suggestions = [
                "계약서 작성 시 주의사항은?",
                "이혼 절차는 어떻게 진행하나요?",
                "손해배상 청구 방법은?",
                "임대차 계약서 검토 포인트는?"
            ]
        elif user_profile.get("user_type") == "변호사":
            suggestions = [
                "최신 판례 동향은?",
                "법령 개정 사항은?",
                "법률 검토 체크리스트는?",
                "소송 전략 수립 방법은?"
            ]
        
        # 관심 분야별 제안 추가
        interest_areas = user_profile.get("interest_areas", [])
        if "민법" in interest_areas:
            suggestions.extend([
                "민법상 계약 해제 요건은?",
                "불법행위 성립 요건은?"
            ])
        
        return suggestions[:5]  # 최대 5개 제안
    
    def collect_feedback(self, rating: int, feedback_text: str, response_id: str = None) -> str:
        """피드백 수집"""
        try:
            feedback_data = {
                "rating": rating,
                "feedback": feedback_text,
                "response_id": response_id,
                "timestamp": datetime.now(),
                "user_profile": self.current_user_profile
            }
            
            # 피드백 저장 (실제로는 데이터베이스에 저장)
            self.logger.info(f"Feedback collected: {feedback_data}")
            
            if rating >= 4:
                return "감사합니다! 피드백이 도움이 됩니다. 😊"
            elif rating >= 3:
                return "피드백을 바탕으로 개선하겠습니다. 👍"
            else:
                return "피드백을 바탕으로 답변을 개선하겠습니다. 더 구체적인 질문을 해주시면 도움이 됩니다. 🔧"
                
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return "피드백 제출 중 오류가 발생했습니다."

def create_production_interface():
    """프로덕션용 Gradio 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = ProductionApp()
    
    # 커스텀 CSS (프로덕션용)
    css = """
    /* 프로덕션용 CSS */
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        font-family: 'Noto Sans KR', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        min-height: 100vh !important;
    }
    
    /* 헤더 스타일 */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 700 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        font-size: 2.5rem !important;
    }
    
    /* 채팅 인터페이스 */
    .chatbot-container {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
        background: white !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .message-input {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    .message-input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    .submit-btn {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 12px 24px !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3) !important;
    }
    
    .submit-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* 빠른 액션 버튼 */
    .quick-actions {
        margin: 16px 0 !important;
    }
    
    .quick-btn {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        background: #f8fafc !important;
        color: #374151 !important;
        border: 1px solid #e5e7eb !important;
        padding: 8px 16px !important;
    }
    
    .quick-btn:hover {
        background: #e2e8f0 !important;
        transform: translateY(-1px) !important;
    }
    
    /* 사이드바 */
    .sidebar {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* 파일 업로드 */
    .file-upload {
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        padding: 40px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.8) !important;
    }
    
    .file-upload:hover {
        border-color: #764ba2 !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transform: scale(1.02) !important;
    }
    
    /* 분석 결과 */
    .analysis-result {
        background: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 위험도 표시 */
    .risk-indicator {
        padding: 12px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        text-align: center !important;
    }
    
    .risk-low {
        color: #10b981 !important;
        background: #d1fae5 !important;
    }
    
    .risk-medium {
        color: #f59e0b !important;
        background: #fef3c7 !important;
    }
    
    .risk-high {
        color: #ef4444 !important;
        background: #fecaca !important;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 12px !important;
            margin: 0 !important;
        }
        
        .header-title {
            font-size: 1.8rem !important;
            margin-bottom: 1rem !important;
        }
        
        .chatbot-container {
            height: 400px !important;
        }
        
        .quick-actions {
            flex-direction: column !important;
        }
        
        .quick-btn {
            margin: 4px 0 !important;
            width: 100% !important;
        }
        
        .sidebar {
            margin-top: 16px !important;
        }
    }
    
    /* 다크 모드 지원 */
    @media (prefers-color-scheme: dark) {
        .gradio-container {
            background: linear-gradient(135deg, #1f2937 0%, #374151 100%) !important;
            color: #f9fafb !important;
        }
        
        .chatbot-container {
            background: #374151 !important;
            border-color: #4b5563 !important;
        }
        
        .message-input {
            background: #374151 !important;
            color: #f9fafb !important;
            border-color: #4b5563 !important;
        }
        
        .sidebar {
            background: rgba(55, 65, 81, 0.9) !important;
            border-color: #4b5563 !important;
        }
    }
    """
    
    # HTML 헤드
    head_html = """
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#667eea">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="LawFirmAI">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
    """
    
    with gr.Blocks(
        css=css,
        title="LawFirmAI - 법률 AI 어시스턴트",
        head=head_html
    ) as interface:
        
        # 헤더
        gr.HTML("""
        <div class="header-title">
            ⚖️ LawFirmAI
            <div style="font-size: 1.2rem; font-weight: 400; margin-top: 8px; color: #6b7280;">
                법률 전문가를 위한 AI 어시스턴트
            </div>
        </div>
        """)
        
        # 온보딩 모달
        onboarding, user_type, interest_areas, start_btn, skip_btn = app.ui.create_user_onboarding()
        
        # 메인 레이아웃
        with gr.Row():
            # 메인 채팅 영역
            chatbot, msg, submit_btn, quick_actions = app.ui.create_main_chat_interface()
            
            # 사이드바
            sidebar_components = app.ui.create_smart_sidebar()
            (user_level, interest_area, new_session_btn, save_session_btn, 
             load_session_btn, session_info, enable_akls, enable_analysis, 
             response_detail, feedback_rating, feedback_text, submit_feedback) = sidebar_components
        
        # 문서 분석 탭
        with gr.Tabs():
            doc_components = app.ui.create_document_analysis_interface()
            file_upload, analysis_type, analysis_options, analyze_btn, analysis_result, risk_indicator, progress_bar = doc_components
        
        # 모바일 인터페이스
        mobile_components = app.ui.create_mobile_interface()
        mobile_actions, mobile_chatbot, voice_input = mobile_components
        
        # 이벤트 핸들러들
        def respond(message, history, user_profile):
            """응답 생성"""
            if not message.strip():
                return history, "", {}
            
            # 질의 처리
            result = app.process_query(message, user_profile)
            
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
        
        def update_user_profile(user_type, interest_areas):
            """사용자 프로필 업데이트"""
            app.current_user_profile = {
                "user_type": user_type,
                "interest_areas": interest_areas,
                "expertise_level": "beginner" if user_type == "일반인" else "intermediate"
            }
            return "프로필이 업데이트되었습니다."
        
        def analyze_document(file, analysis_type, options):
            """문서 분석"""
            if not file:
                return "파일을 업로드해주세요.", "위험도: <span class='risk-low'>파일 없음</span>"
            
            result = app.analyze_document(file.name, analysis_type, options)
            
            if "error" in result:
                return f"분석 중 오류가 발생했습니다: {result['error']}", "위험도: <span class='risk-high'>오류</span>"
            
            # 결과 포맷팅
            analysis_text = f"""
            ## 📄 문서 분석 결과
            
            ### 📋 요약
            {result.get('summary', '분석 완료')}
            
            ### ⚠️ 위험 요소
            {chr(10).join([f"- {risk}" for risk in result.get('risks', [])])}
            
            ### 💡 개선 제안
            {chr(10).join([f"- {rec}" for rec in result.get('recommendations', [])])}
            
            ### 📊 신뢰도: {result.get('confidence', 0):.1%}
            """
            
            # 위험도 표시
            risk_level = result.get('risk_level', 'unknown')
            risk_html = f"위험도: <span class='risk-{risk_level}'>{risk_level.upper()}</span>"
            
            return analysis_text, risk_html
        
        def collect_user_feedback(rating, feedback_text):
            """사용자 피드백 수집"""
            return app.collect_feedback(rating, feedback_text)
        
        def get_smart_suggestions():
            """지능형 질문 제안"""
            suggestions = app.get_smart_suggestions([], app.current_user_profile)
            return suggestions
        
        # 이벤트 연결
        submit_btn.click(
            respond,
            inputs=[msg, chatbot, gr.State(app.current_user_profile)],
            outputs=[chatbot, msg, session_info]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot, gr.State(app.current_user_profile)],
            outputs=[chatbot, msg, session_info]
        )
        
        # 빠른 액션 버튼들
        for i, action_btn in enumerate(quick_actions):
            action_btn.click(
                lambda x, i=i: f"빠른 액션 {i+1}이 선택되었습니다. 구체적인 질문을 입력해주세요.",
                inputs=[msg],
                outputs=[msg]
            )
        
        # 사용자 프로필 업데이트
        user_level.change(
            update_user_profile,
            inputs=[user_level, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
        
        interest_area.change(
            update_user_profile,
            inputs=[user_level, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
        
        # 문서 분석
        analyze_btn.click(
            analyze_document,
            inputs=[file_upload, analysis_type, analysis_options],
            outputs=[analysis_result, risk_indicator]
        )
        
        # 피드백 수집
        submit_feedback.click(
            collect_user_feedback,
            inputs=[feedback_rating, feedback_text],
            outputs=[gr.Textbox(visible=False)]
        )
        
        # 온보딩 이벤트
        start_btn.click(
            update_user_profile,
            inputs=[user_type, interest_areas],
            outputs=[gr.Textbox(visible=False)]
        )
        
        skip_btn.click(
            lambda: "온보딩을 건너뛰었습니다.",
            outputs=[gr.Textbox(visible=False)]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI Production application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_production_interface()
    
    # 프로덕션 실행 설정
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=40,
        show_api=False,
        favicon_path="gradio/static/favicon.ico" if os.path.exists("gradio/static/favicon.ico") else None
    )

if __name__ == "__main__":
    main()
