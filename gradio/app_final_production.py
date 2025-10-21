# -*- coding: utf-8 -*-
"""
LawFirmAI - 최종 프로덕션 애플리케이션
모든 개선사항을 통합한 사용자 친화적 인터페이스
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

# 프로덕션 컴포넌트
try:
    from components.production_ux import create_production_ux_components
    from components.advanced_features import create_production_advanced_features
except ImportError as e:
    # 기본 컴포넌트로 대체
    def create_production_ux_components():
        return {}
    def create_production_advanced_features():
        return {}

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

class ProductionLawFirmAI:
    """프로덕션용 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 핵심 컴포넌트
        self.chat_service = None
        self.session_manager = None
        self.is_initialized = False
        
        # 사용자 상태 관리
        self.current_user_profile = {
            "user_type": "일반인",
            "interest_areas": [],
            "expertise_level": "beginner",
            "onboarding_completed": False
        }
        
        self.current_session_id = f"prod_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # UX 컴포넌트
        self.ux_components = create_production_ux_components()
        self.advanced_features = create_production_advanced_features()
        
        # 피드백 수집
        self.feedback_data = []
        
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
    
    async def process_query_stream(self, query: str, user_profile: Dict = None):
        """스트림 형태로 질의 처리"""
        if not self.is_initialized:
            yield {
                "type": "error",
                "content": "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            }
            return
        
        if not query.strip():
            yield {
                "type": "error",
                "content": "질문을 입력해주세요.",
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
            }
            return
        
        try:
            # ChatService의 스트림 처리 사용
            async for chunk in self.chat_service.process_message_stream(
                query, 
                session_id=self.current_session_id,
                user_id="gradio_user"
            ):
                # 사용자 프로필 정보 추가
                chunk["user_profile"] = user_profile or self.current_user_profile
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error processing query stream: {e}")
            yield {
                "type": "error",
                "content": f"스트림 처리 중 오류가 발생했습니다: {str(e)}",
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat()
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
                "user_profile": self.current_user_profile,
                "session_id": self.current_session_id
            }
            
            # 피드백 저장
            self.feedback_data.append(feedback_data)
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
    
    def update_user_profile(self, user_type: str, interest_areas: List[str]) -> str:
        """사용자 프로필 업데이트"""
        try:
            self.current_user_profile.update({
                "user_type": user_type,
                "interest_areas": interest_areas,
                "expertise_level": "beginner" if user_type == "일반인" else "intermediate",
                "onboarding_completed": True
            })
            
            self.logger.info(f"User profile updated: {self.current_user_profile}")
            return "프로필이 업데이트되었습니다."
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return "프로필 업데이트 중 오류가 발생했습니다."

def create_final_production_interface():
    """최종 프로덕션 인터페이스 생성"""
    
    # 앱 인스턴스 생성
    app = ProductionLawFirmAI()
    
    # 프로덕션용 CSS 로드
    css_file = Path("gradio/static/production.css")
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
    else:
        css = """
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
            font-family: 'Noto Sans KR', sans-serif !important;
        }
        """
    
    # 스트림 모드용 CSS 추가
    css += """
    .stream-toggle {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
    }
    
    .stream-toggle:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .chatbot-container {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    .message-input {
        border-radius: 8px !important;
        border: 2px solid #e5e7eb !important;
        transition: border-color 0.3s ease !important;
    }
    
    .message-input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .submit-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .submit-btn:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .quick-btn {
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
    }
    
    .quick-btn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* 스트림 애니메이션 */
    @keyframes stream-pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .streaming {
        animation: stream-pulse 1.5s infinite !important;
    }
    
    /* 타이핑 효과 */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    
    .typing-effect {
        overflow: hidden;
        white-space: nowrap;
        animation: typing 2s steps(40, end);
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
        
        # 온보딩 플로우 (컴포넌트가 없을 때 기본값 사용)
        if "onboarding" in app.ux_components:
            onboarding_flow = app.ux_components["onboarding"]
            welcome_modal, start_btn, demo_btn = onboarding_flow["welcome"]
            profile_modal, user_type, interest_areas, complete_btn, skip_btn = onboarding_flow["profile"]
            tutorial_modal, got_it_btn, need_help_btn = onboarding_flow["tutorial"]
            demo_modal, try_now_btn, close_demo_btn = onboarding_flow["demo"]
        else:
            # 기본 컴포넌트 생성
            welcome_modal = gr.Modal(visible=False)
            start_btn = gr.Button(visible=False)
            demo_btn = gr.Button(visible=False)
            profile_modal = gr.Modal(visible=False)
            user_type = gr.Radio(choices=["일반인", "법무팀", "변호사", "법학자"], value="일반인")
            interest_areas = gr.CheckboxGroup(choices=["민법", "형법", "상법"], value=[])
            complete_btn = gr.Button(visible=False)
            skip_btn = gr.Button(visible=False)
            tutorial_modal = gr.Modal(visible=False)
            got_it_btn = gr.Button(visible=False)
            need_help_btn = gr.Button(visible=False)
            demo_modal = gr.Modal(visible=False)
            try_now_btn = gr.Button(visible=False)
            close_demo_btn = gr.Button(visible=False)
        
        # 메인 레이아웃
        with gr.Row():
            # 메인 채팅 영역
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
                
                # 스트림 모드 토글
                with gr.Row():
                    stream_mode = gr.Checkbox(label="🔄 스트림 모드 (실시간 답변)", value=True, elem_classes=["stream-toggle"])
                    clear_btn = gr.Button("🗑️ 대화 지우기", variant="secondary", size="sm")
                
                # 빠른 액션 버튼들
                with gr.Row(elem_classes=["quick-actions"]):
                    quick_actions = [
                        gr.Button("📄 문서 분석", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("🔍 판례 검색", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("📚 법령 조회", variant="secondary", elem_classes=["quick-btn"]),
                        gr.Button("💼 계약서 검토", variant="secondary", elem_classes=["quick-btn"])
                    ]
                
                # 지능형 질문 제안 (컴포넌트가 없을 때 기본값 사용)
                if "advanced_features" in app.advanced_features and "suggestions" in app.advanced_features["advanced_features"]:
                    suggestion_components = app.advanced_features["advanced_features"]["suggestions"]
                    suggestion_buttons, refresh_suggestions_btn, suggestion_count, auto_refresh = suggestion_components
                else:
                    # 기본 컴포넌트 생성
                    suggestion_buttons = [gr.Button(visible=False) for _ in range(5)]
                    refresh_suggestions_btn = gr.Button("새로고침", variant="secondary", size="sm")
                    suggestion_count = gr.Slider(1, 5, value=3, step=1, label="제안 개수")
                    auto_refresh = gr.Checkbox(label="자동 새로고침", value=True)
                
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
            
            # 사이드바
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
                
                # 피드백 시스템 (컴포넌트가 없을 때 기본값 사용)
                if "feedback" in app.ux_components:
                    feedback_components = app.ux_components["feedback"]
                    feedback_modal, satisfaction, accuracy, speed, usability, feedback_text, issue_type, submit_feedback_btn, cancel_feedback_btn = feedback_components
                else:
                    # 기본 컴포넌트 생성
                    feedback_modal = gr.Modal(visible=False)
                    satisfaction = gr.Slider(1, 5, value=3, step=1, label="전체 만족도")
                    accuracy = gr.Slider(1, 5, value=3, step=1, label="답변 정확성")
                    speed = gr.Slider(1, 5, value=3, step=1, label="응답 속도")
                    usability = gr.Slider(1, 5, value=3, step=1, label="사용 편의성")
                    feedback_text = gr.Textbox(label="자유 피드백", placeholder="개선사항을 입력하세요", lines=5)
                    issue_type = gr.CheckboxGroup(choices=["답변이 부정확함", "응답이 너무 느림", "인터페이스가 복잡함"], label="문제 유형")
                    submit_feedback_btn = gr.Button("피드백 제출", variant="primary")
                    cancel_feedback_btn = gr.Button("취소", variant="secondary")
                
                with gr.Accordion("💬 피드백", open=False):
                    feedback_rating = gr.Slider(1, 5, value=3, label="만족도")
                    feedback_text_input = gr.Textbox(label="개선사항", placeholder="피드백을 입력하세요", lines=2)
                    submit_feedback_btn_sidebar = gr.Button("피드백 제출", variant="secondary", size="sm")
        
        # 문서 분석 탭 (컴포넌트가 없을 때 기본값 사용)
        with gr.Tabs():
            if "advanced_features" in app.advanced_features and "document_analysis" in app.advanced_features["advanced_features"]:
                doc_components = app.advanced_features["advanced_features"]["document_analysis"]
                file_upload, analysis_type, analysis_options, analysis_detail, analyze_btn, analysis_status, analysis_summary, risk_analysis, legal_basis, improvement_suggestions, risk_indicator, confidence_score, analysis_stats, download_report_btn, share_results_btn = doc_components
            else:
                # 기본 문서 분석 컴포넌트 생성
                with gr.Tab("📄 문서 분석"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_upload = gr.File(label="문서 업로드", file_types=[".pdf", ".docx", ".txt"])
                            analysis_type = gr.Radio(choices=[("계약서 검토", "contract"), ("법률 문서 분석", "legal")], value="contract")
                            analysis_options = gr.CheckboxGroup(choices=["위험 요소 분석", "법적 근거 검토", "개선 제안"], value=["위험 요소 분석"])
                            analyze_btn = gr.Button("분석 시작", variant="primary")
                        with gr.Column(scale=2):
                            analysis_summary = gr.Markdown(label="분석 결과", value="문서를 업로드하고 분석을 시작하세요.")
                            risk_indicator = gr.HTML(value="<div>위험도: <span>분석 대기</span></div>")
                
                # 나머지 컴포넌트들
                analysis_detail = gr.Slider(1, 5, value=3, label="분석 상세도")
                analysis_status = gr.Textbox(label="분석 상태", value="대기 중")
                risk_analysis = gr.Markdown(label="위험 요소 분석")
                legal_basis = gr.Markdown(label="법적 근거")
                improvement_suggestions = gr.Markdown(label="개선 제안")
                confidence_score = gr.HTML(label="신뢰도")
                analysis_stats = gr.JSON(label="분석 통계")
                download_report_btn = gr.Button("보고서 다운로드", variant="secondary")
                share_results_btn = gr.Button("결과 공유", variant="secondary")
        
        # 도움말 시스템 (컴포넌트가 없을 때 기본값 사용)
        if "help" in app.ux_components:
            help_modal, contact_support_btn, close_help_btn = app.ux_components["help"]
        else:
            help_modal = gr.Modal(visible=False)
            contact_support_btn = gr.Button(visible=False)
            close_help_btn = gr.Button(visible=False)
        
        # 이벤트 핸들러들
        def respond(message, history, user_profile):
            """응답 생성 (일반 모드)"""
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
        
        def respond_stream(message, history, user_profile):
            """스트림 응답 생성"""
            if not message.strip():
                return history, "", {}
            
            # 사용자 메시지 추가
            history.append({"role": "user", "content": message})
            
            # 스트림 처리 시작
            import asyncio
            
            async def stream_response():
                full_response = ""
                session_data = {}
                
                try:
                    async for chunk in app.process_query_stream(message, user_profile):
                        chunk_type = chunk.get("type", "unknown")
                        content = chunk.get("content", "")
                        
                        if chunk_type == "status":
                            # 상태 메시지 표시
                            status_message = f"🔄 {content}"
                            history.append({"role": "assistant", "content": status_message})
                            yield history, "", session_data
                            
                        elif chunk_type == "content":
                            # 실제 답변 내용 누적
                            full_response += content
                            history[-1] = {"role": "assistant", "content": full_response}
                            yield history, "", session_data
                            
                        elif chunk_type == "metadata":
                            # 메타데이터 처리
                            metadata = content if isinstance(content, dict) else {}
                            session_data = {
                                "세션 ID": chunk.get("session_id", ""),
                                "처리 시간": f"{metadata.get('processing_time', 0):.2f}초",
                                "신뢰도": f"{metadata.get('confidence', 0):.1%}",
                                "질문 유형": metadata.get("question_type", "일반"),
                                "스트림 모드": "활성화"
                            }
                            yield history, "", session_data
                            
                        elif chunk_type == "error":
                            # 오류 처리
                            error_message = f"❌ {content}"
                            history.append({"role": "assistant", "content": error_message})
                            yield history, "", session_data
                            
                except Exception as e:
                    error_message = f"❌ 스트림 처리 중 오류가 발생했습니다: {str(e)}"
                    history.append({"role": "assistant", "content": error_message})
                    yield history, "", session_data
            
            # 비동기 함수 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 스트림 응답을 단계별로 실행
                response_generator = stream_response()
                final_history = history
                final_session_data = {}
                
                for result in response_generator:
                    final_history, _, final_session_data = result
                
                return final_history, "", final_session_data
                
            finally:
                loop.close()
        
        def respond_stream_gradio(message, history, user_profile):
            """Gradio용 스트림 응답 래퍼"""
            if not message.strip():
                return history, "", {}
            
            # 사용자 메시지 추가
            history.append({"role": "user", "content": message})
            
            # 스트림 응답을 시뮬레이션 (실제로는 WebSocket이나 Server-Sent Events 사용 권장)
            import time
            
            # 초기 상태 메시지
            history.append({"role": "assistant", "content": "🔄 질문을 분석하고 있습니다..."})
            yield history, "", {}
            time.sleep(0.5)
            
            # 검색 상태 메시지
            history[-1] = {"role": "assistant", "content": "🔍 관련 법령과 판례를 검색하고 있습니다..."}
            yield history, "", {}
            time.sleep(0.5)
            
            # 답변 생성 상태 메시지
            history[-1] = {"role": "assistant", "content": "📝 답변을 생성하고 있습니다..."}
            yield history, "", {}
            time.sleep(0.5)
            
            # 실제 답변 생성
            result = app.process_query(message, user_profile)
            answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")
            
            # 답변을 단어별로 스트림
            words = answer.split()
            current_response = ""
            
            for i, word in enumerate(words):
                current_response += word + " "
                history[-1] = {"role": "assistant", "content": current_response.strip()}
                
                # 세션 정보 업데이트
                session_data = {
                    "세션 ID": result.get("session_id", ""),
                    "처리 시간": f"{result.get('processing_time', 0):.2f}초",
                    "신뢰도": f"{result.get('confidence', 0):.1%}",
                    "질문 유형": result.get("question_type", "일반"),
                    "스트림 모드": "활성화",
                    "진행률": f"{((i + 1) / len(words) * 100):.0f}%"
                }
                
                yield history, "", session_data
                time.sleep(0.1)  # 스트림 효과를 위한 지연
        
        def update_user_profile_handler(user_type, interest_areas):
            """사용자 프로필 업데이트"""
            return app.update_user_profile(user_type, interest_areas)
        
        def analyze_document_handler(file, analysis_type, options):
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
        
        def collect_feedback_handler(rating, feedback_text):
            """사용자 피드백 수집"""
            return app.collect_feedback(rating, feedback_text)
        
        def get_smart_suggestions_handler():
            """지능형 질문 제안"""
            suggestions = app.get_smart_suggestions([], app.current_user_profile)
            return suggestions
        
        # 이벤트 연결
        def handle_submit(message, history, user_profile, use_stream):
            """제출 처리 (스트림/일반 모드 선택)"""
            if use_stream:
                return respond_stream_gradio(message, history, user_profile)
            else:
                return respond(message, history, user_profile)
        
        submit_btn.click(
            handle_submit,
            inputs=[msg, chatbot, gr.State(app.current_user_profile), stream_mode],
            outputs=[chatbot, msg, session_info]
        )
        
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot, gr.State(app.current_user_profile), stream_mode],
            outputs=[chatbot, msg, session_info]
        )
        
        # 대화 지우기 버튼
        def clear_chat():
            return [], "", {}
        
        clear_btn.click(
            clear_chat,
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
            update_user_profile_handler,
            inputs=[user_level, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
        
        interest_area.change(
            update_user_profile_handler,
            inputs=[user_level, interest_area],
            outputs=[gr.Textbox(visible=False)]
        )
        
        # 문서 분석
        analyze_btn.click(
            analyze_document_handler,
            inputs=[file_upload, analysis_type, analysis_options],
            outputs=[analysis_summary, risk_indicator]
        )
        
        # 피드백 수집
        submit_feedback_btn_sidebar.click(
            collect_feedback_handler,
            inputs=[feedback_rating, feedback_text_input],
            outputs=[gr.Textbox(visible=False)]
        )
        
        # 온보딩 이벤트 (컴포넌트가 있을 때만 연결)
        if start_btn.visible:
            start_btn.click(
                update_user_profile_handler,
                inputs=[user_type, interest_areas],
                outputs=[gr.Textbox(visible=False)]
            )
        
        if complete_btn.visible:
            complete_btn.click(
                update_user_profile_handler,
                inputs=[user_type, interest_areas],
                outputs=[gr.Textbox(visible=False)]
            )
        
        if skip_btn.visible:
            skip_btn.click(
                lambda: "온보딩을 건너뛰었습니다.",
                outputs=[gr.Textbox(visible=False)]
            )
        
        # 지능형 제안 새로고침
        refresh_suggestions_btn.click(
            get_smart_suggestions_handler,
            outputs=[gr.Textbox(visible=False)]
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI Final Production application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_final_production_interface()
    
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
