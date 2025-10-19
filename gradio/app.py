# -*- coding: utf-8 -*-
"""
LawFirmAI - HuggingFace Spaces 전용 Gradio 애플리케이션
Phase 2의 모든 개선사항을 통합한 최적화된 버전
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
from source.services.performance_monitor import PerformanceMonitor, PerformanceContext
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/huggingface_spaces_app.log')
    ]
)
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """메모리 최적화 클래스"""
    
    def __init__(self, max_memory_percent: float = 85.0):
        self.max_memory_percent = max_memory_percent
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def memory_efficient_inference(self):
        """메모리 효율적인 추론 컨텍스트"""
        # 추론 전 메모리 정리
        self._cleanup_memory()
        
        try:
            yield
        finally:
            # 추론 후 메모리 정리
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def monitor_memory_usage(self) -> float:
        """메모리 사용량 모니터링"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.max_memory_percent:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
            self._cleanup_memory()
        return memory_percent

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0
    
    def log_request(self, response_time: float, success: bool = True):
        """요청 로깅"""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if not success:
            self.error_count += 1
        
        # 최근 100개 요청만 유지
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        if not self.response_times:
            return {
                "avg_response_time": 0,
                "error_rate": 0,
                "total_requests": 0
            }
        
        return {
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "total_requests": self.total_requests
        }

class HuggingFaceSpacesApp:
    """HuggingFace Spaces 전용 LawFirmAI 애플리케이션"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_optimizer = MemoryOptimizer()
        self.performance_monitor = PerformanceMonitor()
        # 성능 모니터링 시작 (옵션)
        if hasattr(self.performance_monitor, 'start_monitoring'):
            self.performance_monitor.start_monitoring()
        
        # 컴포넌트 초기화
        self.db_manager = None
        self.vector_store = None
        self.question_classifier = None
        self.hybrid_search_engine = None
        self.prompt_template_manager = None
        self.unified_prompt_manager = None
        self.dynamic_prompt_updater = None
        self.prompt_optimizer = None
        self.confidence_calculator = None
        self.legal_term_expander = None
        self.gemini_client = None
        self.improved_answer_generator = None
        self.answer_formatter = None
        self.context_builder = None
        
        # ChatService 초기화 (LangGraph 통합)
        self.chat_service = None
        
        # 초기화 상태
        self.is_initialized = False
        self.initialization_error = None
        
        # HuggingFace Spaces 환경 설정
        self._setup_huggingface_spaces_env()
    
    def _setup_huggingface_spaces_env(self):
        """HuggingFace Spaces 환경 설정"""
        # 환경 변수 설정
        os.environ.setdefault('GRADIO_SERVER_NAME', '0.0.0.0')
        os.environ.setdefault('GRADIO_SERVER_PORT', '7860')
        os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
        
        # 로깅 레벨 설정
        if os.getenv('HUGGINGFACE_SPACES', '').lower() == 'true':
            logging.getLogger().setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)
    
    def initialize_components(self) -> bool:
        """컴포넌트 초기화"""
        try:
            self.logger.info("Initializing LawFirmAI components...")
            start_time = time.time()
            
            with self.memory_optimizer.memory_efficient_inference():
                # 데이터베이스 관리자 초기화
                self.db_manager = DatabaseManager("data/lawfirm.db")
                self.logger.info("Database manager initialized")
                
                # 벡터 스토어 초기화 (판례 데이터용)
                self.vector_store = LegalVectorStore(
                    model_name='jhgan/ko-sroberta-multitask',
                    dimension=768,
                    index_type='flat'
                )
                # 판례 벡터 인덱스 로드
                if not self.vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents'):
                    self.logger.warning("Failed to load precedent vector index, using law index")
                    self.vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta')
                self.logger.info("Vector store initialized")
                
                # 질문 분류기 초기화
                self.question_classifier = QuestionClassifier()
                self.logger.info("Question classifier initialized")
                
                # 하이브리드 검색 엔진 초기화
                self.hybrid_search_engine = HybridSearchEngine()
                self.logger.info("Hybrid search engine initialized")
                
                # 최적화된 검색 엔진 초기화
                self.optimized_search_engine = OptimizedSearchEngine(
                    vector_store=self.vector_store,
                    hybrid_engine=self.hybrid_search_engine,
                    cache_size=1000,
                    cache_ttl=3600
                )
                self.logger.info("Optimized search engine initialized")
                
                # 프롬프트 템플릿 관리자 초기화
                self.prompt_template_manager = PromptTemplateManager()
                self.logger.info("Prompt template manager initialized")
                
                # 통합 프롬프트 관리자
                self.unified_prompt_manager = UnifiedPromptManager()
                self.logger.info("Unified prompt manager initialized")
                
                # 동적 프롬프트 업데이터
                self.dynamic_prompt_updater = create_dynamic_prompt_updater(self.unified_prompt_manager)
                self.logger.info("Dynamic prompt updater initialized")
                
                # 프롬프트 최적화기
                self.prompt_optimizer = create_prompt_optimizer(self.unified_prompt_manager)
                self.logger.info("Prompt optimizer initialized")
                
                # 신뢰도 계산기 초기화
                self.confidence_calculator = ConfidenceCalculator()
                self.logger.info("Confidence calculator initialized")
                
                # 법률 용어 확장기 초기화
                self.legal_term_expander = LegalTermExpander()
                self.logger.info("Legal term expander initialized")
                
                # Gemini 클라이언트 초기화
                if os.getenv('GEMINI_ENABLED', 'true').lower() == 'true':
                    self.gemini_client = GeminiClient()
                    self.logger.info("Gemini client initialized")
                
                # 답변 포맷터 초기화
                self.answer_formatter = AnswerFormatter()
                self.logger.info("Answer formatter initialized")
                
                # 컨텍스트 빌더 초기화
                self.context_builder = ContextBuilder()
                self.logger.info("Context builder initialized")
                
                # 개선된 답변 생성기 초기화
                self.improved_answer_generator = ImprovedAnswerGenerator(
                    gemini_client=self.gemini_client,
                    prompt_template_manager=self.prompt_template_manager,
                    confidence_calculator=self.confidence_calculator,
                    answer_formatter=self.answer_formatter,
                    context_builder=self.context_builder
                )
                self.logger.info("Improved answer generator initialized")
                
                # ChatService 초기화 (LangGraph 통합)
                config = Config()
                self.chat_service = ChatService(config)
                self.logger.info("ChatService initialized with LangGraph integration")
            
            initialization_time = time.time() - start_time
            self.logger.info(f"All components initialized successfully in {initialization_time:.2f} seconds")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """질의 처리"""
        if not self.is_initialized:
            return {
                "answer": "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "error": "System not initialized",
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"}
            }
        
        # 성능 모니터링 컨텍스트
        with PerformanceContext(
            self.performance_monitor, 
            "query_processing",
            {"query_length": len(query), "session_id": session_id}
        ) as perf_ctx:
            try:
                with self.memory_optimizer.memory_efficient_inference():
                    # ChatService를 사용한 처리 (실제 RAG 시스템)
                    if self.chat_service and hasattr(self.chat_service, 'improved_answer_generator') and self.chat_service.improved_answer_generator:
                        import asyncio
                        result = asyncio.run(self.chat_service.process_message(query, session_id=session_id))
                        
                        response_time = time.time() - start_time
                        self.performance_monitor.log_request(response_time, success=True)
                        
                        return {
                            "answer": result.get("response", ""),
                            "confidence": {
                                "confidence": result.get("confidence", 0.0),
                                "reliability_level": "HIGH" if result.get("confidence", 0) > 0.7 else "MEDIUM" if result.get("confidence", 0) > 0.4 else "LOW"
                            },
                            "processing_time": response_time,
                            "memory_usage": self.memory_optimizer.monitor_memory_usage(),
                            "session_id": result.get("session_id"),
                            "query_type": result.get("query_type", ""),
                            "legal_references": result.get("legal_references", []),
                            "processing_steps": result.get("processing_steps", []),
                            "metadata": result.get("metadata", {}),
                            "errors": result.get("errors", [])
                        }
                    else:
                        # 기존 방식으로 폴백
                        return self._process_query_legacy(query, start_time)
                        
            except Exception as e:
                response_time = time.time() - start_time
                self.performance_monitor.log_request(response_time, success=False)
                self.logger.error(f"Error processing query: {e}")
                
                return {
                    "answer": "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                    "error": str(e),
                    "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                    "processing_time": response_time
                }
    
    def _process_query_legacy(self, query: str, start_time: float) -> Dict[str, Any]:
        """기존 방식으로 질의 처리 (폴백)"""
        try:
            with self.memory_optimizer.memory_efficient_inference():
                # 질문 분류
                question_classification = self.question_classifier.classify_question(query)
                
                # 지능형 검색 실행
                search_results = self.hybrid_search_engine.search_with_question_type(
                    query=query,
                    question_type=question_classification,
                    max_results=10
                )
                
                # 답변 생성
                answer_result = self.improved_answer_generator.generate_answer(
                    query=query,
                    question_type=question_classification,
                    context="",
                    sources=search_results,
                    conversation_history=None
                )
                
                response_time = time.time() - start_time
                self.performance_monitor.log_request(response_time, success=True)
                
                return {
                    "answer": answer_result.answer,
                    "question_type": question_classification.question_type.value,
                        "confidence": {
                            "confidence": answer_result.confidence.confidence,
                            "reliability_level": answer_result.confidence.level.value
                        },
                    "processing_time": response_time,
                    "memory_usage": self.memory_optimizer.monitor_memory_usage()
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.log_request(response_time, success=False)
            self.logger.error(f"Error in legacy processing: {e}")
            
            return {
                "answer": "기존 처리 방식에서 오류가 발생했습니다.",
                "error": str(e),
                "confidence": {"confidence": 0, "reliability_level": "VERY_LOW"},
                "processing_time": response_time
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        stats = self.performance_monitor.get_stats()
        memory_usage = self.memory_optimizer.monitor_memory_usage()
        
        status = {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "memory_usage_percent": memory_usage,
            "performance_stats": stats,
            "timestamp": datetime.now().isoformat(),
            "prompt_system": {
                "unified_manager_available": self.unified_prompt_manager is not None,
                "dynamic_updater_available": self.dynamic_prompt_updater is not None,
                "prompt_optimizer_available": self.prompt_optimizer is not None,
                "prompt_analytics": self.unified_prompt_manager.get_prompt_analytics() if self.unified_prompt_manager else {},
                "optimization_recommendations": self.prompt_optimizer.get_optimization_recommendations() if self.prompt_optimizer else []
            }
        }
        
        # ChatService 상태 추가
        if self.chat_service:
            try:
                chat_status = self.chat_service.get_service_status()
                status["chat_service"] = chat_status
            except Exception as e:
                status["chat_service_error"] = str(e)
        
        return status

# 전역 앱 인스턴스
app_instance = HuggingFaceSpacesApp()

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    # 컴포넌트 초기화
    if not app_instance.initialize_components():
        logger.error("Failed to initialize components")
    
    # 커스텀 CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        text-align: left;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    """
    
    # HTML 헤드에 매니페스트 및 메타 태그 추가
    head_html = """
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="LawFirmAI">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """
    
    with gr.Blocks(
        css=css, 
        title="LawFirmAI - 법률 AI 어시스턴트",
        head=head_html
    ) as interface:
        gr.Markdown("""
        # ⚖️ LawFirmAI - 법률 AI 어시스턴트
        
        **Phase 2 완료**: 지능형 질문 분류, 동적 검색 가중치, 구조화된 답변, 신뢰도 시스템
        
        법률 관련 질문에 정확하고 신뢰할 수 있는 답변을 제공합니다.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="법률 AI 어시스턴트",
                    height=500,
                    show_label=True,
                    type="messages"  # 최신 Gradio 형식 사용
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="법률 관련 질문을 입력하세요...",
                        label="질문",
                        scale=4
                    )
                    submit_btn = gr.Button("전송", scale=1, variant="primary")
                
                # 예시 질문
                gr.Examples(
                    examples=[
                        "계약 해제 조건이 무엇인가요?",
                        "손해배상 관련 판례를 찾아주세요",
                        "불법행위의 법적 근거를 알려주세요",
                        "이혼 절차는 어떻게 진행하나요?",
                        "민법 제750조의 내용이 무엇인가요?"
                    ],
                    inputs=msg
                )
            
            with gr.Column(scale=1):
                # 시스템 상태
                status_output = gr.JSON(
                    label="시스템 상태",
                    value=app_instance.get_system_status()
                )
                
                # 신뢰도 정보
                confidence_output = gr.JSON(
                    label="신뢰도 정보"
                )
                
                # 성능 통계
                performance_output = gr.JSON(
                    label="성능 통계"
                )
        
        # 이벤트 핸들러
        def respond(message, history):
            """응답 생성"""
            if not message.strip():
                return history, "", {}
            
            # 질의 처리
            result = app_instance.process_query(message)
            
            # 메시지 형식 변환 (type="messages"에 맞게)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result["answer"]})
            
            # 신뢰도 정보
            confidence_info = {
                "신뢰도": f"{result['confidence']['confidence']:.1%}",
                "수준": result['confidence']['reliability_level'],
                "처리 시간": f"{result.get('processing_time', 0):.2f}초",
                "질문 유형": result.get('question_type', 'Unknown')
            }
            
            return history, "", confidence_info
        
        def update_status():
            """상태 업데이트"""
            return app_instance.get_system_status()
        
        def update_performance():
            """성능 통계 업데이트"""
            return app_instance.performance_monitor.get_stats()
        
        # 이벤트 연결
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, confidence_output]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, confidence_output]
        )
        
        # 상태 업데이트 버튼 추가 (주기적 업데이트 대신 수동 업데이트)
        with gr.Row():
            refresh_status_btn = gr.Button("상태 새로고침", variant="secondary")
            refresh_performance_btn = gr.Button("성능 통계 새로고침", variant="secondary")
        
        # 수동 상태 업데이트 이벤트
        refresh_status_btn.click(
            update_status,
            outputs=status_output
        )
        
        refresh_performance_btn.click(
            update_performance,
            outputs=performance_output
        )
    
        return interface
    
    def get_performance_dashboard(self):
        """성능 모니터링 대시보드"""
        with gr.Blocks(title="성능 모니터링 대시보드") as dashboard:
            gr.Markdown("# LawFirmAI 성능 모니터링 대시보드")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 시스템 상태")
                    system_health = gr.JSON(label="시스템 상태", value=self.performance_monitor.get_system_health())
                    
                    refresh_btn = gr.Button("새로고침", variant="secondary")
                    
                with gr.Column():
                    gr.Markdown("## 성능 요약 (최근 24시간)")
                    performance_summary = gr.JSON(label="성능 요약", value=self.performance_monitor.get_performance_summary())
            
            with gr.Row():
                gr.Markdown("## 성능 메트릭 내보내기")
                export_btn = gr.Button("메트릭 내보내기", variant="primary")
                export_status = gr.Textbox(label="내보내기 상태", interactive=False)
            
            def refresh_data():
                return (
                    self.performance_monitor.get_system_health(),
                    self.performance_monitor.get_performance_summary()
                )
            
            def export_metrics():
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"performance_metrics_{timestamp}.json"
                    self.performance_monitor.export_metrics(filepath)
                    return f"메트릭이 {filepath}에 저장되었습니다."
                except Exception as e:
                    return f"내보내기 실패: {str(e)}"
            
            refresh_btn.click(
                fn=refresh_data,
                outputs=[system_health, performance_summary]
            )
            
            export_btn.click(
                fn=export_metrics,
                outputs=[export_status]
            )
        
        return dashboard

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI HuggingFace Spaces application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # 안정적인 실행 설정
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=40,
        # 공유 기능 완전 비활성화
        show_api=False,
        # 정적 파일 서빙 설정
        favicon_path="gradio/static/favicon.ico" if os.path.exists("gradio/static/favicon.ico") else None
    )

if __name__ == "__main__":
    main()
