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

# Gradio 및 기타 라이브러리
import gradio as gr
import torch
import psutil

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.services.question_classifier import QuestionClassifier, QuestionType
from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.prompt_templates import PromptTemplateManager
from source.services.confidence_calculator import ConfidenceCalculator
from source.services.legal_term_expander import LegalTermExpander
from source.services.ollama_client import OllamaClient
from source.services.improved_answer_generator import ImprovedAnswerGenerator
from source.services.answer_formatter import AnswerFormatter
from source.services.context_builder import ContextBuilder

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
        
        # 컴포넌트 초기화
        self.db_manager = None
        self.vector_store = None
        self.question_classifier = None
        self.hybrid_search_engine = None
        self.prompt_template_manager = None
        self.confidence_calculator = None
        self.legal_term_expander = None
        self.ollama_client = None
        self.improved_answer_generator = None
        self.answer_formatter = None
        self.context_builder = None
        
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
                
                # 벡터 스토어 초기화
                self.vector_store = LegalVectorStore("data/embeddings/ml_enhanced_ko_sroberta")
                self.logger.info("Vector store initialized")
                
                # 질문 분류기 초기화
                self.question_classifier = QuestionClassifier()
                self.logger.info("Question classifier initialized")
                
                # 하이브리드 검색 엔진 초기화
                self.hybrid_search_engine = HybridSearchEngine(
                    db_manager=self.db_manager,
                    vector_store=self.vector_store
                )
                self.logger.info("Hybrid search engine initialized")
                
                # 프롬프트 템플릿 관리자 초기화
                self.prompt_template_manager = PromptTemplateManager()
                self.logger.info("Prompt template manager initialized")
                
                # 신뢰도 계산기 초기화
                self.confidence_calculator = ConfidenceCalculator()
                self.logger.info("Confidence calculator initialized")
                
                # 법률 용어 확장기 초기화
                self.legal_term_expander = LegalTermExpander()
                self.logger.info("Legal term expander initialized")
                
                # Ollama 클라이언트 초기화 (로컬 환경에서만)
                if os.getenv('OLLAMA_ENABLED', 'false').lower() == 'true':
                    self.ollama_client = OllamaClient()
                    self.logger.info("Ollama client initialized")
                
                # 답변 포맷터 초기화
                self.answer_formatter = AnswerFormatter()
                self.logger.info("Answer formatter initialized")
                
                # 컨텍스트 빌더 초기화
                self.context_builder = ContextBuilder()
                self.logger.info("Context builder initialized")
                
                # 개선된 답변 생성기 초기화
                self.improved_answer_generator = ImprovedAnswerGenerator(
                    ollama_client=self.ollama_client,
                    prompt_template_manager=self.prompt_template_manager,
                    confidence_calculator=self.confidence_calculator,
                    answer_formatter=self.answer_formatter,
                    context_builder=self.context_builder
                )
                self.logger.info("Improved answer generator initialized")
            
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
        
        start_time = time.time()
        
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
                        "reliability_level": answer_result.confidence.reliability_level.value
                    },
                    "processing_time": response_time,
                    "memory_usage": self.memory_optimizer.monitor_memory_usage()
                }
                
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        stats = self.performance_monitor.get_stats()
        memory_usage = self.memory_optimizer.monitor_memory_usage()
        
        return {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "memory_usage_percent": memory_usage,
            "performance_stats": stats,
            "timestamp": datetime.now().isoformat()
        }

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
    
    with gr.Blocks(css=css, title="LawFirmAI - 법률 AI 어시스턴트") as interface:
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
                    show_label=True
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
            
            # 응답 추가
            history.append([message, result["answer"]])
            
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
        
        # 주기적 상태 업데이트
        interface.load(
            update_status,
            outputs=status_output,
            every=30
        )
        
        interface.load(
            update_performance,
            outputs=performance_output,
            every=60
        )
    
    return interface

def main():
    """메인 함수"""
    logger.info("Starting LawFirmAI HuggingFace Spaces application...")
    
    # Gradio 인터페이스 생성 및 실행
    interface = create_gradio_interface()
    
    # HuggingFace Spaces 환경에서 실행
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
