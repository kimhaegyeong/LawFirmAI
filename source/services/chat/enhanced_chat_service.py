# -*- coding: utf-8 -*-
"""
Enhanced Chat Service
개선된 채팅 메시지 처리 서비스
"""

import hashlib
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# 상대경로 import
from ...data.database import DatabaseManager
from ...data.vector_store import LegalVectorStore
from ...utils.config import Config
from ...utils.logger import get_logger
from ...utils.memory_manager import get_memory_manager
from ...utils.monitoring.realtime_memory_monitor import (
    get_memory_monitor,
)
from ...utils.weakref_cleanup import get_weakref_registry

# 하이브리드 분류기로 완전 대체됨 - 키워드 시스템 제거 완료
# 모든 키워드 추출 및 도메인 분류 기능은 IntegratedHybridQuestionClassifier에서 처리

# Phase 1: 대화 맥락 강화 모듈
# from .integrated_session_manager import IntegratedSessionManager
# from .multi_turn_handler import MultiTurnQuestionHandler
# from .context_compressor import ContextCompressor

# Phase 2: 개인화 및 지능형 분석 모듈
# from .user_profile_manager import UserProfileManager
# from .emotion_intent_analyzer import EmotionIntentAnalyzer
# from .conversation_flow_tracker import ConversationFlowTracker

# Phase 3: 장기 기억 및 품질 모니터링 모듈
# from .contextual_memory_manager import ContextualMemoryManager
# from .conversation_quality_monitor import ConversationQualityMonitor

# 지능형 응답 스타일 시스템
# from .intelligent_response_style_system import IntelligentResponseStyleSystem, ResponseStyle

# ResponseStyle을 간단히 정의 (테스트용)
class ResponseStyle:
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CONCISE = "concise"
    DETAILED = "detailed"

# 대화형 계약서 작성 모듈
# from .interactive_contract_assistant import InteractiveContractAssistant
# from .contract_query_handler import ContractQueryHandler

# 자연스러운 답변 개선 모듈
# from .conversation_connector import ConversationConnector
# from .emotional_tone_adjuster import EmotionalToneAdjuster
# from .personalized_style_learner import PersonalizedStyleLearner
# from ..realtime_feedback_system import RealtimeFeedbackSystem
# from ..naturalness_evaluator import NaturalnessEvaluator

# 성능 최적화 모듈
# from ..cache_manager import get_cache_manager, cached
# from .optimized_search_engine import OptimizedSearchEngine

# 법률 제한 시스템 모듈 (ML 통합 최신 버전) - 테스트를 위해 주석 처리
# from .ml_integrated_validation_system import MLIntegratedValidationSystem
# from .improved_legal_restriction_system import ImprovedLegalRestrictionSystem, ImprovedRestrictionResult
# from .intent_based_processor import IntentBasedProcessor, ProcessingResult
# from .content_filter_engine import ContentFilterEngine, FilterResult
# from .response_validation_system import ResponseValidationSystem, ValidationResult, ValidationStatus, ValidationLevel
# from .safe_response_generator import SafeResponseGenerator, SafeResponse
# from .legal_compliance_monitor import LegalComplianceMonitor, ComplianceStatus
# from .user_education_system import UserEducationSystem, WarningMessage
# from .multi_stage_validation_system import MultiStageValidationSystem, MultiStageValidationResult

logger = get_logger(__name__)


class EnhancedChatService:
    """향상된 채팅 서비스 클래스"""

    def __init__(self, config: Config):
        """서비스 초기화"""
        # Google Cloud 경고 설정
        self._setup_google_cloud_warnings()

        self.config = config
        self.logger = get_logger(__name__)

        # LangGraph 사용 여부 확인 (활성화)
        self.use_langgraph = True

        # 메모리 관리 시스템 초기화
        self._initialize_memory_management()

        # 사용자 설정 관리자 초기화 (안전한 초기화)
        try:
            from ..user_preference_manager import preference_manager
            self.user_preferences = preference_manager
        except ImportError:
            self.logger.warning("User preference manager를 import할 수 없습니다. 기본값으로 설정합니다.")
            self.user_preferences = None

        # 답변 완성도 검증자 초기화 (안전한 초기화)
        try:
            from ..answer_completion_validator import completion_validator
            self.completion_validator = completion_validator
        except ImportError:
            self.logger.warning("Answer completion validator를 import할 수 없습니다. 기본값으로 설정합니다.")
            self.completion_validator = None

        # 향상된 완성 시스템 초기화 (안전한 초기화)
        try:
            from ..enhanced_completion_system import enhanced_completion_system
            self.enhanced_completion_system = enhanced_completion_system
        except ImportError:
            self.logger.warning("Enhanced completion system을 import할 수 없습니다. 기본값으로 설정합니다.")
            self.enhanced_completion_system = None

        # 핵심 컴포넌트 초기화
        self._initialize_core_components()

        # 법률 제한 시스템 초기화
        self._initialize_legal_restriction_systems()

        # 고급 검색 엔진 초기화
        self._initialize_advanced_search_engines()

        # 현재법 검색 엔진 초기화
        self._initialize_current_law_search_engine()

        # 통합 서비스 초기화 (현재법 검색 엔진 초기화 후)
        self._initialize_unified_services()

        # Phase 시스템 초기화
        self._initialize_phase_systems()

        # 대화형 계약서 어시스턴트 초기화
        self._initialize_interactive_contract_assistant()

        # 자연스러운 대화 개선 시스템 초기화
        self._initialize_natural_conversation_systems()

        # 성능 모니터링 시스템 초기화
        self._initialize_performance_monitoring()

        # 성능 최적화 시스템 초기화
        self._initialize_performance_systems()

        # 품질 향상 시스템 초기화
        self._initialize_quality_enhancement_systems()

        # 향상된 법률 검색 시스템 초기화
        try:
            self.logger.info("🔍 향상된 법률 검색 시스템 초기화 시작...")
            self._initialize_enhanced_law_search()
            self.logger.info("✅ 향상된 법률 검색 시스템 초기화 완료")
        except Exception as e:
            self.logger.error(f"❌ 향상된 법률 검색 시스템 초기화 실패: {e}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")

        # 지능형 응답 스타일 시스템 초기화
        try:
            self.logger.info("🔍 지능형 응답 스타일 시스템 초기화 시작...")
            self._initialize_intelligent_style_system()
            self.logger.info("✅ 지능형 응답 스타일 시스템 초기화 완료")
        except Exception as e:
            self.logger.error(f"❌ 지능형 응답 스타일 시스템 초기화 실패: {e}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")

        # LangGraph 워크플로우 서비스 초기화
        try:
            self.logger.info("🚀 LangGraph 워크플로우 초기화 시작...")
            self._initialize_langgraph_workflow()
            self.logger.info(f"🔍 LangGraph 초기화 완료 - 서비스 상태: {self.langgraph_service is not None}")
        except Exception as e:
            self.logger.error(f"❌ LangGraph 워크플로우 초기화 실패: {e}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")

        # 🆕 LangGraph 초기화 검증 및 상태 로깅
        self._validate_langgraph_initialization()

        self.logger.info("EnhancedChatService 초기화 완료")

    def _validate_langgraph_initialization(self):
        """LangGraph 초기화 상태 검증 및 상세 로깅"""
        self.logger.info("=" * 70)
        self.logger.info("🔍 LangGraph 초기화 상태 검증")
        self.logger.info("=" * 70)

        # 현재 상태 확인
        self.logger.info("📊 현재 상태:")
        self.logger.info(f"   - use_langgraph: {self.use_langgraph}")
        self.logger.info(f"   - langgraph_service: {self.langgraph_service is not None}")

        if self.langgraph_service is not None:
            self.logger.info(f"   - langgraph_service 타입: {type(self.langgraph_service).__name__}")
            self.logger.info(f"   - process_query 메서드: {hasattr(self.langgraph_service, 'process_query')}")

        # 문제가 있는 경우 재시도
        if self.use_langgraph and self.langgraph_service is None:
            self.logger.warning("⚠️ LangGraph 활성화되었으나 서비스가 초기화되지 않음")
            self.logger.info("🔄 LangGraph 재초기화 시도...")

            try:
                self._initialize_langgraph_workflow()

                if self.langgraph_service and self.use_langgraph:
                    self.logger.info("✅ LangGraph 재초기화 성공")
                    self.logger.info(f"   - use_langgraph: {self.use_langgraph}")
                    self.logger.info(f"   - langgraph_service: {self.langgraph_service is not None}")
                else:
                    self.logger.error("❌ LangGraph 재초기화 실패")
                    self.logger.error("💡 해결 방법:")
                    self.logger.error("   1. pip install langgraph langchain-core langchain-community")
                    self.logger.error("   2. .env 파일에 GOOGLE_API_KEY 설정")
                    self.logger.error("   3. 로그를 확인하여 구체적인 오류 원인 파악")

            except Exception as e:
                self.logger.error(f"❌ 재초기화 중 오류 발생: {e}")
                import traceback
                self.logger.error(f"상세 오류: {traceback.format_exc()}")

        # 최종 상태
        if self.langgraph_service and self.use_langgraph:
            self.logger.info("=" * 70)
            self.logger.info("✅ LangGraph 사용 가능 - 워크플로우가 활성화됩니다")
            self.logger.info("=" * 70)
        else:
            self.logger.info("=" * 70)
            self.logger.warning("⚠️ LangGraph 사용 불가 - 기본 RAG 시스템으로 폴백됩니다")
            self.logger.info("=" * 70)

    def _setup_google_cloud_warnings(self):
        """Google Cloud 경고 설정"""
        os.environ['GRPC_DNS_RESOLVER'] = 'native'
        os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
        os.environ['GOOGLE_CLOUD_PROJECT'] = ''
        os.environ['GCLOUD_PROJECT'] = ''
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
        os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GRPC_TRACE'] = ''

        # gRPC 로그 레벨 설정
        import logging
        logging.getLogger('grpc').setLevel(logging.ERROR)
        logging.getLogger('google').setLevel(logging.ERROR)
        logging.getLogger('google.auth').setLevel(logging.ERROR)
        logging.getLogger('google.auth.transport').setLevel(logging.ERROR)
        logging.getLogger('google.auth.transport.grpc').setLevel(logging.ERROR)
        logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)
        logging.getLogger('google.cloud').setLevel(logging.ERROR)
        logging.getLogger('google.api_core').setLevel(logging.ERROR)

    def _initialize_memory_management(self):
        """메모리 관리 시스템 초기화"""
        try:
            # 메모리 매니저 초기화 - 메모리 제한 증가
            self.memory_manager = get_memory_manager(max_memory_mb=2048)  # 1024에서 2048로 증가

            # WeakRef 레지스트리 초기화
            self.weakref_registry = get_weakref_registry()

            # 실시간 메모리 모니터 초기화
            self.memory_monitor = get_memory_monitor()

            # 메모리 알림 콜백 등록
            self.memory_manager.add_alert_callback(self._on_memory_alert)

            # 컴포넌트 추적용 WeakRef 등록 함수
            self._track_component = self._create_component_tracker()

            self.logger.info("메모리 관리 시스템 초기화 완료")

        except Exception as e:
            self.logger.error(f"메모리 관리 시스템 초기화 실패: {e}")
            # 기본값으로 설정
            self.memory_manager = None
            self.weakref_registry = None
            self.memory_monitor = None
            self._track_component = lambda obj, name: None

    def _create_component_tracker(self):
        """컴포넌트 추적 함수 생성"""
        def track_component(obj: Any, name: str) -> str:
            """컴포넌트를 WeakRef로 등록"""
            if self.weakref_registry:
                return self.weakref_registry.register_object(obj, name)
            return name
        return track_component

    def _on_memory_alert(self, alert):
        """메모리 알림 처리"""
        self.logger.warning(f"메모리 알림 [{alert.severity}]: {alert.message}")

        # 메모리 부족 시 강제 정리 (기준 완화)
        if alert.severity in ['medium', 'high', 'critical']:  # medium 추가
            self.logger.info("메모리 정리 시작")
            cleanup_result = self.perform_memory_cleanup()
            self.logger.info(f"메모리 정리 완료: {cleanup_result.get('memory_freed_mb', 0):.1f}MB 해제")

    def perform_memory_cleanup(self):
        """메모리 정리 수행 (고급 최적화 포함)"""
        try:
            import gc
            import os

            import psutil

            # 현재 메모리 사용량 측정
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # 가비지 컬렉션 실행
            collected = gc.collect()
            # 추가 가비지 컬렉션 (3회 반복)
            for _ in range(3):
                collected += gc.collect()

            # 컴포넌트별 메모리 정리
            cleanup_count = 0

            # 모델 매니저 메모리 정리
            if hasattr(self, 'model_manager') and self.model_manager:
                try:
                    if hasattr(self.model_manager, 'clear_cache'):
                        self.model_manager.clear_cache()
                        cleanup_count += 1
                    # 추가: 모델 언로드 시도
                    if hasattr(self.model_manager, 'unload_unused_models'):
                        self.model_manager.unload_unused_models()
                except Exception as e:
                    self.logger.debug(f"Model manager cleanup failed: {e}")

            # 벡터 스토어 메모리 정리
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    if hasattr(self.vector_store, 'clear_cache'):
                        self.vector_store.clear_cache()
                        cleanup_count += 1
                    # 추가: 인덱스 캐시 정리
                    if hasattr(self.vector_store, 'clear_index_cache'):
                        self.vector_store.clear_index_cache()
                except Exception as e:
                    self.logger.debug(f"Vector store cleanup failed: {e}")

            # 답변 생성기 메모리 정리
            if hasattr(self, 'answer_generator') and self.answer_generator:
                try:
                    if hasattr(self.answer_generator, 'clear_cache'):
                        self.answer_generator.clear_cache()
                        cleanup_count += 1
                except Exception as e:
                    self.logger.debug(f"Answer generator cleanup failed: {e}")

            # RAG 서비스 메모리 정리
            if hasattr(self, 'unified_rag_service') and self.unified_rag_service:
                try:
                    if hasattr(self.unified_rag_service, 'clear_cache'):
                        self.unified_rag_service.clear_cache()
                        cleanup_count += 1
                except Exception as e:
                    self.logger.debug(f"RAG service cleanup failed: {e}")

            # 메모리 사용량 재측정
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after

            self.logger.info(f"전체 메모리 정리 완료: {memory_freed:.1f}MB 해제, {collected}개 객체 수집, {cleanup_count}개 컴포넌트 정리")

            return {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_freed_mb': memory_freed,
                'objects_collected': collected,
                'components_cleaned': cleanup_count,
                'success': True
            }

        except Exception as e:
            self.logger.error(f"메모리 정리 실패: {e}")
            return {
                'memory_freed_mb': 0,
                'objects_collected': 0,
                'components_cleaned': 0,
                'success': False,
                'error': str(e)
            }

    def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            # 데이터베이스 매니저
            self.db_manager = DatabaseManager("data/lawfirm.db")
            self._track_component(self.db_manager, "db_manager")

            # 벡터 스토어
            self.vector_store = LegalVectorStore(
                model_name="jhgan/ko-sroberta-multitask",
                dimension=768,
                index_type="flat",
                enable_quantization=True,
                enable_lazy_loading=True,
                memory_threshold_mb=3000
            )
            self._track_component(self.vector_store, "vector_store")

            # 벡터 인덱스 로드 - 개선된 오류 처리 (여러 경로 시도)
            index_paths = [
                "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index",  # 가장 큰 데이터셋
                "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index",  # 판례 데이터
                "data/embeddings/legal_vector_index"  # 기본 데이터
            ]

            index_loaded = False
            for index_path in index_paths:
                try:
                    if self.vector_store.load_index(index_path):
                        self.logger.info(f"벡터 인덱스 로드 성공: {index_path}")
                        index_loaded = True
                        break
                    else:
                        self.logger.warning(f"벡터 인덱스 로드 실패: {index_path}")
                except Exception as e:
                    self.logger.warning(f"벡터 인덱스 로드 오류 {index_path}: {e}")
                    continue

            if not index_loaded:
                self.logger.warning("모든 벡터 인덱스 로드 실패, 키워드 검색으로 대체")
                # 벡터 인덱스가 없어도 서비스는 계속 동작하도록 함
                self.logger.info("벡터 인덱스 없이 서비스 계속 진행")

            # 모델 매니저 (안전한 초기화)
            try:
                from ..optimized_model_manager import OptimizedModelManager
                self.model_manager = OptimizedModelManager()
                self._track_component(self.model_manager, "model_manager")
            except ImportError:
                self.logger.warning("OptimizedModelManager를 import할 수 없습니다. 기본값으로 설정합니다.")
                self.model_manager = None

            # RAG 서비스 (MLEnhancedRAGService를 대체하고 UnifiedRAGService로 통합)
            # self.rag_service = MLEnhancedRAGService(...)

            # 하이브리드 검색 엔진 (안전한 초기화)
            try:
                from ..search.hybrid_search_engine import HybridSearchEngine
                self.hybrid_search_engine = HybridSearchEngine()
                self._track_component(self.hybrid_search_engine, "hybrid_search_engine")
            except ImportError:
                self.logger.warning("HybridSearchEngine을 import할 수 없습니다. 기본값으로 설정합니다.")
                self.hybrid_search_engine = None

            # 질문 분류기 (안전한 초기화)
            try:
                from ..question_classifier import QuestionClassifier
                self.question_classifier = QuestionClassifier()
                self._track_component(self.question_classifier, "question_classifier")
            except ImportError:
                self.logger.warning("QuestionClassifier를 import할 수 없습니다. 기본값으로 설정합니다.")
                self.question_classifier = None

            # 향상된 답변 생성기 (안전한 초기화)
            try:
                self.logger.debug("ImprovedAnswerGenerator import 시도 중...")
                from ..improved_answer_generator import ImprovedAnswerGenerator
                self.logger.debug(f"ImprovedAnswerGenerator import 성공: {ImprovedAnswerGenerator}")

                self.logger.debug("ImprovedAnswerGenerator 인스턴스 생성 시도 중...")
                self.improved_answer_generator = ImprovedAnswerGenerator()
                self.logger.debug("ImprovedAnswerGenerator 인스턴스 생성 성공")

                self._track_component(self.improved_answer_generator, "improved_answer_generator")
                self.logger.info("ImprovedAnswerGenerator 초기화 완료")
            except ImportError as e:
                self.logger.warning(
                    f"ImprovedAnswerGenerator를 import할 수 없습니다 (ImportError). "
                    f"기본값으로 설정합니다. 상세 오류: {type(e).__name__}: {str(e)}"
                )
                self.logger.debug(f"ImportError 상세 정보: {e.__traceback__}")
                self.improved_answer_generator = None
            except Exception as e:
                self.logger.error(
                    f"ImprovedAnswerGenerator 초기화 중 예상치 못한 오류 발생: "
                    f"{type(e).__name__}: {str(e)}"
                )
                self.logger.debug(f"오류 상세 정보: {e.__traceback__}", exc_info=True)
                self.improved_answer_generator = None

            self.logger.info("핵심 컴포넌트 초기화 완료")

        except Exception as e:
            self.logger.error(f"핵심 컴포넌트 초기화 실패: {e}")
            # 기본값으로 설정
            self.db_manager = None
            self.vector_store = None
            self.model_manager = None
            self.rag_service = None
            self.hybrid_search_engine = None
            self.question_classifier = None
            self.improved_answer_generator = None

        # 하이브리드 질문 분류기 초기화 (try 블록 외부에서)
        self._initialize_hybrid_classifier()

    def _initialize_hybrid_classifier(self):
        """하이브리드 질문 분류기 초기화"""
        try:
            self.logger.debug("IntegratedHybridQuestionClassifier import 시도 중...")
            # 하이브리드 분류기 초기화 (안전한 import)
            from ..integrated_hybrid_classifier import (
                IntegratedHybridQuestionClassifier,
            )
            self.logger.debug(f"IntegratedHybridQuestionClassifier import 성공: {IntegratedHybridQuestionClassifier}")

            self.logger.debug("IntegratedHybridQuestionClassifier 인스턴스 생성 시도 중...")
            self.hybrid_classifier = IntegratedHybridQuestionClassifier(
                confidence_threshold=0.7  # 기본 임계값
            )
            self.logger.debug("IntegratedHybridQuestionClassifier 인스턴스 생성 성공")

            self._track_component(self.hybrid_classifier, "hybrid_classifier")

            self.logger.info("✅ 하이브리드 질문 분류기 초기화 완료")

        except ImportError as e:
            self.logger.warning(
                f"IntegratedHybridQuestionClassifier를 import할 수 없습니다 (ImportError). "
                f"기본값으로 설정합니다. 상세 오류: {type(e).__name__}: {str(e)}"
            )
            self.logger.debug(f"ImportError 상세 정보: {e.__traceback__}", exc_info=True)
            self.hybrid_classifier = None
        except Exception as e:
            self.logger.error(
                f"하이브리드 질문 분류기 초기화 중 예상치 못한 오류 발생: "
                f"{type(e).__name__}: {str(e)}"
            )
            self.logger.debug(f"오류 상세 정보: {e.__traceback__}", exc_info=True)
            self.hybrid_classifier = None

    def _initialize_unified_services(self):
        """통합 서비스 초기화"""
        try:
            # 벡터 스토어가 없으면 다시 초기화
            if not self.vector_store:
                from ..data.vector_store import LegalVectorStore
                self.vector_store = LegalVectorStore()
                try:
                    self.vector_store.load_index()
                    self.logger.info("벡터 인덱스 로드 성공")
                except Exception as e:
                    self.logger.warning(f"벡터 인덱스 로드 실패: {e}")

            # 통합 검색 엔진 (안전한 초기화)
            try:
                from ..unified_search_engine import UnifiedSearchEngine
                self.unified_search_engine = UnifiedSearchEngine(
                    vector_store=self.vector_store,
                    current_law_search_engine=self.current_law_search_engine
                )
                self.logger.info("✅ UnifiedSearchEngine 초기화 성공")
            except ImportError as e:
                self.logger.warning(f"UnifiedSearchEngine을 import할 수 없습니다: {e}")
                self.unified_search_engine = None
            except Exception as e:
                self.logger.error(f"UnifiedSearchEngine 초기화 실패: {e}")
                self.unified_search_engine = None

            # 통합 RAG 서비스 (안전한 초기화)
            try:
                self.logger.debug("UnifiedRAGService import 시도 중...")
                from ..unified_rag_service import UnifiedRAGService
                self.logger.debug(f"UnifiedRAGService import 성공: {UnifiedRAGService}")

                self.logger.debug("UnifiedRAGService 인스턴스 생성 시도 중...")
                self.unified_rag_service = UnifiedRAGService(
                    model_manager=self.model_manager,
                    search_engine=self.unified_search_engine,
                    answer_generator=self.improved_answer_generator,
                    question_classifier=self.question_classifier
                )
                self.logger.debug("UnifiedRAGService 인스턴스 생성 성공")

                self.logger.info("✅ UnifiedRAGService 초기화 완료")
            except ImportError as e:
                self.logger.warning(
                    f"UnifiedRAGService를 import할 수 없습니다 (ImportError). "
                    f"기본값으로 설정합니다. 상세 오류: {type(e).__name__}: {str(e)}"
                )
                self.logger.debug(f"ImportError 상세 정보: {e.__traceback__}", exc_info=True)
                self.unified_rag_service = None
            except Exception as e:
                self.logger.error(
                    f"UnifiedRAGService 초기화 중 예상치 못한 오류 발생: "
                    f"{type(e).__name__}: {str(e)}"
                )
                self.logger.debug(f"오류 상세 정보: {e.__traceback__}", exc_info=True)
                self.unified_rag_service = None

            self.logger.info("통합 서비스 초기화 완료")

        except Exception as e:
            self.logger.error(f"통합 서비스 초기화 실패: {e}")
            self.unified_search_engine = None
            self.unified_rag_service = None

    def _initialize_legal_restriction_systems(self):
        """법률 제한 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 법률 제한 시스템을 None으로 설정 (테스트용)
            self.ml_validation_system = None
            self.improved_legal_restriction_system = None
            self.intent_based_processor = None
            self.content_filter_engine = None
            self.response_validation_system = None
            self.safe_response_generator = None
            self.legal_compliance_monitor = None
            self.user_education_system = None
            self.multi_stage_validation_system = None

            self.logger.info("법률 제한 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"법률 제한 시스템 초기화 실패: {e}")
            # 기본값으로 설정
            self.ml_validation_system = None
            self.improved_legal_restriction_system = None
            self.intent_based_processor = None
            self.content_filter_engine = None
            self.response_validation_system = None
            self.safe_response_generator = None
            self.legal_compliance_monitor = None
            self.user_education_system = None
            self.multi_stage_validation_system = None

    def _initialize_advanced_search_engines(self):
        """고급 검색 엔진 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 고급 검색 엔진을 None으로 설정 (테스트용)
            self.optimized_search_engine = None
            self.exact_search_engine = None
            self.semantic_search_engine = None
            self.precedent_search_engine = None

            self.logger.info("고급 검색 엔진 초기화 완료 (테스트 모드 - 모든 엔진 비활성화)")

        except Exception as e:
            self.logger.error(f"고급 검색 엔진 초기화 실패: {e}")
            self.optimized_search_engine = None
            self.exact_search_engine = None
            self.semantic_search_engine = None
            self.precedent_search_engine = None

    def _initialize_current_law_search_engine(self):
        """현재법령 검색 엔진 초기화 - 안전한 초기화"""
        try:
            from ..current_law_search_engine import CurrentLawSearchEngine

            self.current_law_search_engine = CurrentLawSearchEngine(
                db_path="data/lawfirm.db",
                vector_store=self.vector_store
            )

            self.logger.info("현재법령 검색 엔진 초기화 완료")

        except ImportError as e:
            self.logger.warning(f"CurrentLawSearchEngine을 import할 수 없습니다: {e}")
            self.current_law_search_engine = None
        except Exception as e:
            self.logger.error(f"현재법령 검색 엔진 초기화 실패: {e}")
            self.current_law_search_engine = None

    def _initialize_phase_systems(self):
        """Phase 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 Phase 시스템을 None으로 설정 (테스트용)
            self.integrated_session_manager = None
            self.multi_turn_handler = None
            self.context_compressor = None
            self.user_profile_manager = None
            self.emotion_intent_analyzer = None
            self.conversation_flow_tracker = None
            self.contextual_memory_manager = None
            self.conversation_quality_monitor = None

            self.logger.info("Phase 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"Phase 시스템 초기화 실패: {e}")
            # 기본값으로 설정
            self.integrated_session_manager = None
            self.multi_turn_handler = None
            self.context_compressor = None
            self.user_profile_manager = None
            self.emotion_intent_analyzer = None
            self.conversation_flow_tracker = None
            self.contextual_memory_manager = None
            self.conversation_quality_monitor = None

    def _initialize_natural_conversation_systems(self):
        """자연스러운 대화 개선 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 자연스러운 대화 개선 시스템을 None으로 설정 (테스트용)
            self.conversation_connector = None
            self.emotional_tone_adjuster = None
            self.personalized_style_learner = None

            self.logger.info("자연스러운 대화 개선 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"자연스러운 대화 개선 시스템 초기화 실패: {e}")
            # 기본값으로 설정
            self.conversation_connector = None
            self.emotional_tone_adjuster = None
            self.personalized_style_learner = None

    def _initialize_performance_monitoring(self):
        """성능 모니터링 시스템 초기화 - 안전한 초기화"""
        try:
            from ...utils.monitoring.performance_monitor import PerformanceMonitor

            # 성능 모니터 초기화
            self.performance_monitor = PerformanceMonitor(self.config)

            # 메서드 존재 여부 확인
            if hasattr(self.performance_monitor, 'log_response_metrics'):
                self.logger.info("성능 모니터링 시스템 초기화 완료")
            else:
                self.logger.warning("PerformanceMonitor 초기화되었으나 log_response_metrics 메서드가 없습니다")
                self.performance_monitor = None

        except ImportError as e:
            self.logger.warning(f"PerformanceMonitor를 import할 수 없습니다: {e}")
            self.performance_monitor = None
        except Exception as e:
            self.logger.error(f"성능 모니터링 시스템 초기화 실패: {e}")
            self.performance_monitor = None

    def _initialize_interactive_contract_assistant(self):
        """대화형 계약서 어시스턴트 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 대화형 계약서 어시스턴트를 None으로 설정 (테스트용)
            self.interactive_contract_assistant = None
            self.contract_query_handler = None

            self.logger.info("대화형 계약서 어시스턴트 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"대화형 계약서 어시스턴트 초기화 실패: {e}")
            self.interactive_contract_assistant = None
            self.contract_query_handler = None

    def _initialize_performance_systems(self):
        """성능 최적화 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 성능 최적화 시스템을 None으로 설정 (테스트용)
            self.performance_monitor = None
            self.memory_optimizer = None

            self.logger.info("성능 최적화 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"성능 최적화 시스템 초기화 실패: {e}")
            self.performance_monitor = None
            self.memory_optimizer = None

    def _initialize_quality_enhancement_systems(self):
        """품질 향상 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 품질 향상 시스템을 None으로 설정 (테스트용)
            self.answer_quality_enhancer = None
            self.answer_structure_enhancer = None
            self.confidence_calculator = None
            self.prompt_optimizer = None
            self.unified_prompt_manager = None

            self.logger.info("품질 향상 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"품질 향상 시스템 초기화 실패: {e}")
            self.answer_quality_enhancer = None
            self.answer_structure_enhancer = None
            self.confidence_calculator = None
            self.prompt_optimizer = None
            self.unified_prompt_manager = None

    async def process_message(self,
                            message: str,
                            context: Optional[str] = None,
                            session_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> Dict[str, Any]:
        """메시지 처리 메인 메서드"""
        self.logger.info(f"EnhancedChatService.process_message called for: {message}")
        start_time = time.time()

        # 세션 ID와 사용자 ID 생성
        if not session_id:
            session_id = f"session_{int(time.time())}_{hashlib.md5(message.encode()).hexdigest()[:8]}"
        if not user_id:
            user_id = f"user_{int(time.time())}"

        try:
            # 법률 조문 쿼리 우선 처리
            if self._is_law_article_query(message):
                self.logger.info(f"법률 조문 쿼리 감지: {message}")
                return await self._handle_law_article_query(message, user_id, session_id)

            # 계약서 관련 쿼리 우선 처리
            if self.contract_query_handler and self.contract_query_handler.is_contract_related_query(message):
                self.logger.info(f"계약서 관련 쿼리 감지: {message}")
                return await self.contract_query_handler.handle_interactive_contract_query(message, session_id, user_id)

            # 입력 검증 및 전처리
            validation_result = self._validate_and_preprocess_input(message)
            if not validation_result["valid"]:
                return self._create_simple_error_response(
                    validation_result["error"], session_id, user_id, start_time
                )

            # 캐시 확인 - 테스트를 위해 주석 처리
            # cache_key = self._generate_cache_key(message, user_id, context)
            # cached_result = self.cache_manager.get(cache_key) if self.cache_manager else None
            # if cached_result:
            #     cached_result["processing_time"] = time.time() - start_time
            #     cached_result["cached"] = True
            #     return cached_result

            # 쿼리 분석
            query_analysis = await self._analyze_query(message, context, user_id, session_id)
            self.logger.debug(f"process_message에서 query_analysis: {query_analysis}")

            # 법률 제한 검증
            restriction_result = await self._validate_legal_restrictions(
                message, query_analysis, user_id, session_id
            )

            if restriction_result and restriction_result.get("restricted", False):
                return self._create_restricted_response(
                    restriction_result, session_id, user_id, start_time
                )

            # Phase 1: 대화 맥락 강화
            phase1_info = await self._process_phase1_context(message, session_id, user_id)

            # Phase 2: 개인화 및 지능형 분석
            phase2_info = await self._process_phase2_personalization(
                message, session_id, user_id, phase1_info
            )

            # Phase 3: 장기 기억 및 품질 모니터링
            phase3_info = await self._process_phase3_memory_quality(
                message, session_id, user_id, phase1_info, phase2_info
            )

            # 답변 생성 실행
            self.logger.info(f"About to call _generate_enhanced_response for: {message}")
            response_result = await self._generate_enhanced_response(
                message, query_analysis, restriction_result, user_id, session_id,
                phase1_info, phase2_info, phase3_info
            )
            self.logger.info(f"_generate_enhanced_response completed, method: {response_result.get('generation_method', 'unknown')}")

            # response_result가 문자열인 경우 딕셔너리로 변환
            if isinstance(response_result, str):
                self.logger.debug(f"_generate_enhanced_response가 문자열을 반환함: {type(response_result)}")
                response_result = {"response": response_result, "confidence": 0.5, "generation_method": "string_fallback"}

            # 답변 완성도 검증 및 보완 (안전한 처리)
            if response_result.get("response") and self.enhanced_completion_system:
                response_text = response_result["response"]
                if isinstance(response_text, str):
                    try:
                        # 강화된 완성 시스템 사용
                        completion_result = self.enhanced_completion_system.force_complete_answer(
                            response_text, message, query_analysis.get("category", "일반")
                        )

                        if completion_result.was_truncated:
                            self.logger.info(f"답변이 추가로 보완됨. 완성 방법: {completion_result.completion_method}")
                            response_result["response"] = completion_result.completed_answer
                            response_result["completion_improved"] = True
                            response_result["completion_method"] = completion_result.completion_method
                            response_result["completion_confidence"] = completion_result.confidence
                    except Exception as e:
                        self.logger.debug(f"답변 완성도 검증 실패: {e}")
                        # 완성도 검증 실패 시 원본 응답 유지

                    # 예제 추가 기능 제거 (의존성 문제로 비활성화됨)
                    # if self.user_preferences.get_preference("example_preference"):
                    #     enhanced_response = self._add_examples_to_response(
                    #         response_result["response"], message, query_analysis
                    #     )
                    #     if enhanced_response != response_result["response"]:
                    #         response_result["response"] = enhanced_response
                    #         response_result["examples_added"] = True

            # 사용자 선호도 기반 면책 조항 처리 (안전한 처리)
            if self.user_preferences and hasattr(self.user_preferences, 'add_disclaimer_to_response'):
                try:
                    final_response_text = self.user_preferences.add_disclaimer_to_response(
                        response_result["response"], message
                    )
                    response_result["response"] = final_response_text
                except Exception as e:
                    self.logger.debug(f"면책 조항 추가 실패: {e}")
                    # 면책 조항 추가 실패 시 원본 응답 유지
            else:
                # 기본 면책 조항 추가
                if response_result["response"] and not response_result["response"].endswith("."):
                    response_result["response"] += "\n\n※ 이 답변은 일반적인 법률 정보 제공을 목적으로 하며, 구체적인 법률 자문은 변호사와 상담하시기 바랍니다."

            # 처리 시간 추가 (음수 방지)
            processing_time = max(0.0, time.time() - start_time)
            response_result["processing_time"] = processing_time
            response_result["session_id"] = session_id
            response_result["user_id"] = user_id

            # 메모리 정리 (더 적극적으로)
            if processing_time > 3.0:  # 3초 이상 걸린 경우에 메모리 정리
                try:
                    cleanup_result = self.perform_memory_cleanup()
                    if cleanup_result.get('success'):
                        response_result["memory_cleanup"] = {
                            "memory_freed_mb": cleanup_result.get('memory_freed_mb', 0),
                            "objects_collected": cleanup_result.get('objects_collected', 0)
                        }
                except Exception as e:
                    self.logger.warning(f"메모리 정리 실패: {e}")

            # 추가 메모리 정리 (매번 실행)
            try:
                import gc
                collected = gc.collect()
                if collected > 0:
                    self.logger.debug(f"Garbage collection freed {collected} objects")
            except Exception as e:
                self.logger.warning(f"Garbage collection failed: {e}")

            # 캐시 저장 (추가 최적화 - 캐시 시간 증가) - 테스트를 위해 주석 처리
            # if self.cache_manager:
            #     self.cache_manager.set(cache_key, response_result, ttl_seconds=7200)  # 1시간에서 2시간으로 증가

            # 성능 메트릭 로그
            if self.performance_monitor and hasattr(self.performance_monitor, 'log_response_metrics'):
                try:
                    query_type = response_result.get('generation_method', 'unknown')
                    processing_time = response_result.get('processing_time', 0)
                    confidence = response_result.get('confidence', 0)
                    response_length = len(response_result.get('response', ''))

                    self.performance_monitor.log_response_metrics(
                        query_type=query_type,
                        processing_time=processing_time,
                        confidence=confidence,
                        response_length=response_length,
                        success=True
                    )
                except Exception as e:
                    self.logger.error(f"성능 메트릭 로그 실패: {e}")
            elif self.performance_monitor:
                self.logger.warning("PerformanceMonitor가 초기화되었으나 log_response_metrics 메서드가 없습니다")
            else:
                self.logger.debug("PerformanceMonitor가 초기화되지 않았습니다")

            return response_result

        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류 발생: {e}")
            import traceback
            self.logger.error(f"전체 스택: {traceback.format_exc()}")

            # 오류 발생시에도 성능 메트릭 로그
            if self.performance_monitor and hasattr(self.performance_monitor, 'log_response_metrics'):
                try:
                    processing_time = time.time() - start_time
                    self.performance_monitor.log_response_metrics(
                        query_type='error',
                        processing_time=processing_time,
                        confidence=0.0,
                        response_length=0,
                        success=False,
                        error_message=str(e)
                    )
                except Exception as metric_error:
                    self.logger.error(f"성능 메트릭 로그 실패: {metric_error}")
            elif self.performance_monitor:
                self.logger.warning("PerformanceMonitor가 초기화되었으나 log_response_metrics 메서드가 없습니다")
            else:
                self.logger.debug("PerformanceMonitor가 초기화되지 않았습니다")

            return self._create_simple_error_response(
                f"메시지 처리 중 오류가 발생했습니다: {str(e)}",
                session_id, user_id, start_time
            )

    def _validate_and_preprocess_input(self, message: str) -> Dict[str, Any]:
        """입력 검증 및 전처리"""
        if not message or not message.strip():
            return {"valid": False, "error": "메시지가 비어있습니다."}

        if len(message) > 10000:
            return {"valid": False, "error": "메시지가 너무 깁니다."}

        return {"valid": True, "message": message.strip()}

    def _generate_cache_key(self, message: str, user_id: str, context: Optional[str] = None) -> str:
        """캐시 키 생성"""
        key_data = f"{message}_{user_id}_{context or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _create_simple_error_response(self, error_message: str, session_id: str, user_id: str, start_time: float) -> Dict[str, Any]:
        """간단한 오류 응답 생성"""
        return {
            "response": f"죄송합니다. {error_message}",
            "confidence": 0.0,
            "sources": [],
            "processing_time": time.time() - start_time,
            "session_id": session_id,
            "user_id": user_id,
            "error": error_message,
            "generation_method": "error"
        }

    def _create_restricted_response(self, restriction_result: Dict[str, Any], session_id: str, user_id: str, start_time: float) -> Dict[str, Any]:
        """제한된 응답 생성"""
        return {
            "response": restriction_result.get("safe_response", "해당 질문에 대해서는 답변을 제공할 수 없습니다."),
            "confidence": 0.0,
            "sources": [],
            "processing_time": time.time() - start_time,
            "session_id": session_id,
            "user_id": user_id,
            "restricted": True,
            "restriction_reason": restriction_result.get("reason", "법률 제한"),
            "generation_method": "restricted"
        }

    async def _analyze_query(self, message: str, context: Optional[str], user_id: str, session_id: str) -> Dict[str, Any]:
        """질문 분석 - 하이브리드 분류기 우선 사용"""
        try:
            # 하이브리드 분류기 사용 (우선)
            if self.hybrid_classifier:
                try:
                    classification_result = self.hybrid_classifier.classify(message)

                    # 향상된 도메인 분석 수행
                    domain_analysis = self.hybrid_classifier.get_enhanced_domain_analysis(message, classification_result)

                    # 하이브리드 분류기에서 추출된 특징 정보 활용
                    features = classification_result.features or {}

                    # 통합된 결과를 기존 형식으로 변환 (하이브리드 분류기 정보 우선 사용)
                    query_analysis = {
                        "query_type": classification_result.question_type_value,
                        "intent": "unknown",  # 기본값
                        "confidence": classification_result.confidence,
                        "context": context,
                        "keywords": features.get("keywords", []),  # 하이브리드 분류기에서 추출
                        "statute_match": features.get("statute_match"),
                        "statute_law": features.get("statute_law"),
                        "statute_article": features.get("statute_article"),
                        "domain": domain_analysis.get("domain", classification_result.question_type.to_domain()),
                        "domain_confidence": domain_analysis.get("domain_confidence", classification_result.confidence),
                        "domain_scores": domain_analysis.get("domain_scores", {}),
                        "domain_info": domain_analysis.get("domain_info", {}),
                        "timestamp": datetime.now(),
                        "session_id": session_id,
                        "user_id": user_id,
                        "classification_method": classification_result.method,
                        "classification_reasoning": classification_result.reasoning,
                        "law_weight": classification_result.law_weight,
                        "precedent_weight": classification_result.precedent_weight,
                        "features": features,  # 추가 특징 정보
                        "hybrid_analysis": True  # 하이브리드 분석 사용 표시
                    }

                    self.logger.info(f"하이브리드 분류 결과: {classification_result.question_type_value} "
                                   f"(신뢰도: {classification_result.confidence:.3f}, 방법: {classification_result.method}, "
                                   f"도메인: {domain_analysis.get('domain', 'unknown')})")

                    return query_analysis

                except Exception as e:
                    self.logger.warning(f"하이브리드 분류 실패, 기존 분류기 사용: {e}")

            # 폴백: 기존 질문 분류기 사용 (키워드 시스템 의존성 감소)
            if self.question_classifier:
                classification = self.question_classifier.classify_question(message)
                query_type = classification.question_type
                intent = "unknown"
                confidence = classification.confidence
            else:
                query_type = "general"
                intent = "unknown"
                confidence = 0.5

            # 간소화된 폴백 분석 (키워드 시스템 의존성 최소화)
            fallback_analysis = self._perform_fallback_analysis(message, query_type, confidence)

            return {
                "query_type": query_type,
                "intent": intent,
                "confidence": confidence,
                "context": context,
                "keywords": fallback_analysis.get("keywords", []),
                "statute_match": fallback_analysis.get("statute_match"),
                "statute_law": fallback_analysis.get("statute_law"),
                "statute_article": fallback_analysis.get("statute_article"),
                "domain": fallback_analysis.get("domain", "general"),
                "domain_confidence": fallback_analysis.get("domain_confidence", confidence),
                "domain_scores": fallback_analysis.get("domain_scores", {}),
                "timestamp": datetime.now(),
                "session_id": session_id,
                "user_id": user_id,
                "classification_method": "legacy_fallback",
                "classification_reasoning": "하이브리드 분류기 실패로 인한 폴백 사용",
                "hybrid_analysis": False  # 하이브리드 분석 미사용 표시
            }

        except Exception as e:
            self.logger.error(f"쿼리 분석 중 오류: {e}")
            return {
                "query_type": "general",
                "intent": "unknown",
                "confidence": 0.5,
                "context": context,
                "keywords": [],
                "statute_match": None,
                "statute_law": None,
                "statute_article": None,
                "domain": "general",
                "domain_confidence": 0.5,
                "domain_scores": {},
                "timestamp": datetime.now(),
                "session_id": session_id,
                "user_id": user_id,
                "classification_method": "error",
                "classification_reasoning": f"분석 실패: {str(e)}",
                "error": str(e),
                "hybrid_analysis": False
            }

    def _perform_fallback_analysis(self, message: str, query_type: str, confidence: float) -> Dict[str, Any]:
        """간소화된 폴백 분석 (키워드 시스템 의존성 최소화)"""
        try:
            message_lower = message.lower()

            # 기본 키워드만 추출 (최소한의 키워드 시스템 사용)
            basic_keywords = []
            basic_domain_scores = {}

            # 간소화된 도메인 분류 (핵심 키워드만 사용)
            core_keywords = {
                "civil_law": ["민법", "계약", "손해배상", "불법행위"],
                "criminal_law": ["형법", "범죄", "처벌", "형량"],
                "family_law": ["이혼", "상속", "양육권", "재산분할"],
                "commercial_law": ["상법", "회사", "주식", "이사"],
                "labor_law": ["노동법", "근로", "임금", "해고"],
                "real_estate": ["부동산", "매매", "임대차", "등기"],
                "general": ["법률", "법령", "조문", "판례"]
            }

            for domain, keywords in core_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in message_lower:
                        score += 1
                        basic_keywords.append(keyword)
                basic_domain_scores[domain] = score

            # 가장 높은 점수의 도메인 선택
            best_domain = max(basic_domain_scores.items(), key=lambda x: x[1])[0] if basic_domain_scores else "general"
            domain_confidence = min(1.0, basic_domain_scores.get(best_domain, 0) / 4.0)  # 정규화

            # 법률 조문 패턴 검색 (간소화)
            import re
            statute_patterns = [
                r'(민법|형법|상법|노동법|가족법|행정법|헌법)\s*제\s*(\d+)\s*조',
                r'제\s*(\d+)\s*조',
                r'(\d+)\s*조'
            ]

            statute_match = None
            statute_law = None
            statute_article = None

            for pattern in statute_patterns:
                match = re.search(pattern, message)
                if match:
                    statute_match = match
                    groups = match.groups()

                    if len(groups) == 2:
                        statute_law = groups[0]
                        statute_article = groups[1]
                    elif len(groups) == 1:
                        statute_article = groups[0]
                    break

            return {
                "keywords": list(set(basic_keywords)),
                "domain": best_domain,
                "domain_confidence": domain_confidence,
                "domain_scores": basic_domain_scores,
                "statute_match": statute_match.group(0) if statute_match else None,
                "statute_law": statute_law,
                "statute_article": statute_article
            }

        except Exception as e:
            self.logger.error(f"폴백 분석 실패: {e}")
            return {
                "keywords": [],
                "domain": "general",
                "domain_confidence": 0.5,
                "domain_scores": {},
                "statute_match": None,
                "statute_law": None,
                "statute_article": None
            }

    async def _validate_legal_restrictions(self, message: str, query_analysis: Dict[str, Any], user_id: str, session_id: str) -> Dict[str, Any]:
        """법률 제한 검증"""
        try:
            if self.multi_stage_validation_system:
                try:
                    validation_result = await self.multi_stage_validation_system.validate_message(
                        message, query_analysis, user_id, session_id
                    )
                    return {
                        "restricted": validation_result.is_restricted,
                        "reason": validation_result.restriction_reason,
                        "safe_response": validation_result.safe_response,
                        "confidence": validation_result.confidence
                    }
                except AttributeError:
                    self.logger.debug("MultiStageValidationSystem에 validate_message 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"법률 제한 검증 실패: {e}")
            else:
                return {"restricted": False, "reason": None, "safe_response": None, "confidence": 1.0}

        except Exception as e:
            self.logger.error(f"법률 제한 검증 중 오류: {e}")
            return {"restricted": False, "reason": None, "safe_response": None, "confidence": 0.5}

    async def _process_phase1_context(self, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """Phase 1: 대화 맥락 강화"""
        try:
            phase1_info = {
                "session_context": None,
                "multi_turn_context": None,
                "compressed_context": None,
                "enabled": False
            }

            if self.integrated_session_manager:
                try:
                    session_context = await self.integrated_session_manager.get_session_context(session_id)
                    phase1_info["session_context"] = session_context
                    phase1_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("IntegratedSessionManager에 get_session_context 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"세션 컨텍스트 가져오기 실패: {e}")

            if self.multi_turn_handler:
                try:
                    multi_turn_context = await self.multi_turn_handler.process_message(message, session_id)
                    phase1_info["multi_turn_context"] = multi_turn_context
                    phase1_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("MultiTurnQuestionHandler에 process_message 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"다중 턴 처리 실패: {e}")

            if self.context_compressor:
                try:
                    compressed_context = await self.context_compressor.compress_context(message, session_id)
                    phase1_info["compressed_context"] = compressed_context
                except AttributeError:
                    self.logger.debug("ContextCompressor에 compress_context 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"컨텍스트 압축 실패: {e}")
                phase1_info["enabled"] = True

            return phase1_info

        except Exception as e:
            self.logger.error(f"Phase 1 처리 중 오류: {e}")
            return {"enabled": False, "error": str(e)}

    async def _process_phase2_personalization(self, message: str, session_id: str, user_id: str, phase1_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: 개인화 및 지능형 분석"""
        try:
            phase2_info = {
                "user_profile": None,
                "emotion_intent": None,
                "conversation_flow": None,
                "enabled": False
            }

            if self.user_profile_manager:
                try:
                    user_profile = await self.user_profile_manager.get_user_profile(user_id)
                    phase2_info["user_profile"] = user_profile
                    phase2_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("UserProfileManager에 get_user_profile 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"사용자 프로필 가져오기 실패: {e}")

            if self.emotion_intent_analyzer:
                try:
                    emotion_intent = await self.emotion_intent_analyzer.analyze_emotion_intent(message, user_id)
                    phase2_info["emotion_intent"] = emotion_intent
                    phase2_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("EmotionIntentAnalyzer에 analyze_emotion_intent 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"감정 의도 분석 실패: {e}")

            if self.conversation_flow_tracker:
                try:
                    # ConversationTurn 객체 생성
                    from datetime import datetime

                    from .conversation_manager import ConversationTurn
                    turn = ConversationTurn(
                        user_query=message,
                        bot_response="",
                        timestamp=datetime.now(),
                        question_type="general",
                        intent="unknown",
                        entities=[],
                        confidence=0.5
                    )
                    conversation_flow = await self.conversation_flow_tracker.track_conversation_flow(session_id, turn)
                    phase2_info["conversation_flow"] = conversation_flow
                except AttributeError as e:
                    self.logger.debug(f"ConversationFlowTracker 메서드 없음: {e}")
                except Exception as e:
                    self.logger.debug(f"대화 흐름 추적 실패: {e}")
                phase2_info["enabled"] = True

            return phase2_info

        except Exception as e:
            self.logger.error(f"Phase 2 처리 중 오류: {e}")
            return {"enabled": False, "error": str(e)}

    async def _process_phase3_memory_quality(self, message: str, session_id: str, user_id: str, phase1_info: Dict[str, Any], phase2_info: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: 장기 기억 및 품질 모니터링"""
        try:
            phase3_info = {
                "contextual_memory": None,
                "quality_metrics": None,
                "enabled": False
            }

            if self.contextual_memory_manager:
                try:
                    contextual_memory = await self.contextual_memory_manager.manage_contextual_memory(
                        message, session_id, user_id, phase1_info, phase2_info
                    )
                    phase3_info["contextual_memory"] = contextual_memory
                    phase3_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("ContextualMemoryManager에 manage_contextual_memory 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"컨텍스트 메모리 관리 실패: {e}")

            if self.conversation_quality_monitor:
                try:
                    quality_metrics = await self.conversation_quality_monitor.monitor_conversation_quality(
                        message, session_id, user_id
                    )
                    phase3_info["quality_metrics"] = quality_metrics
                    phase3_info["enabled"] = True
                except AttributeError:
                    self.logger.debug("ConversationQualityMonitor에 monitor_conversation_quality 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"대화 품질 모니터링 실패: {e}")

            return phase3_info

        except Exception as e:
            self.logger.error(f"Phase 3 처리 중 오류: {e}")
            return {"enabled": False, "error": str(e)}

    async def _generate_enhanced_response(self, message: str, query_analysis: Dict[str, Any],
                                         restriction_result: Dict[str, Any], user_id: str, session_id: str,
                                         phase1_info: Dict[str, Any], phase2_info: Dict[str, Any], phase3_info: Dict[str, Any]) -> Dict[str, Any]:
        """향상된 답변 생성 - LangGraph 워크플로우 우선 사용"""
        self.logger.info(f"_generate_enhanced_response called for: {message}")
        start_time = time.time()  # start_time 변수 추가
        try:
            # 스타일 분석 및 결정
            detected_style = None
            if self.intelligent_style_system:
                try:
                    detected_style = self.intelligent_style_system.determine_optimal_style(
                        message, query_analysis, session_id
                    )
                    self.logger.info("Detected response style: " + detected_style.value)
                except Exception as e:
                    self.logger.debug(f"Style detection failed: {e}")
                    detected_style = ResponseStyle.FRIENDLY  # 기본값

            # 🔥 1순위: LangGraph 워크플로우 (가장 고도화된 처리) - 강제 활성화
            self.logger.info("🔍 LangGraph 실행 조건 확인:")
            self.logger.info(f"  - use_langgraph: {self.use_langgraph}")
            self.logger.info(f"  - langgraph_service: {self.langgraph_service is not None}")

            if self.use_langgraph:
                # LangGraph 서비스가 None이면 재초기화 시도
                if not self.langgraph_service:
                    self.logger.warning("⚠️ LangGraph 서비스가 None입니다. 재초기화 시도...")
                    self._initialize_langgraph_workflow()

                if self.langgraph_service:
                    try:
                        self.logger.info(f"🚀 LangGraph 워크플로우로 처리 시작: {message}")
                        self.logger.info(f"📊 LangGraph 서비스 상태: {self.langgraph_service is not None}")
                        self.logger.info(f"⚙️ LangGraph 사용 설정: {self.use_langgraph}")

                        # LangGraph 워크플로우 실행
                        self.logger.info("🔍 LangGraph 워크플로우 실행 전 상태 확인:")
                        self.logger.info(f"  - langgraph_service: {self.langgraph_service is not None}")
                        self.logger.info(f"  - use_langgraph: {self.use_langgraph}")
                        self.logger.info(f"  - message: {message}")
                        self.logger.info(f"  - langgraph_service type: {type(self.langgraph_service)}")

                        langgraph_result = await self.langgraph_service.process_query(
                            query=message,
                            context=query_analysis.get("context"),
                            session_id=session_id,
                            user_id=user_id
                        )

                        self.logger.info(f"✅ LangGraph 워크플로우 실행 완료: {langgraph_result is not None}")
                        self.logger.info("🔍 LangGraph 결과 키: " + str(list(langgraph_result.keys()) if langgraph_result else 'None'))
                        self.logger.info(f"🔍 LangGraph 응답 텍스트: {langgraph_result.get('response', 'NOT_FOUND')[:100] if langgraph_result else 'None'}")
                        self.logger.info(f"🔍 LangGraph 전체 결과: {langgraph_result}")

                        if langgraph_result and langgraph_result.get('response'):
                            self.logger.info("🎉 LangGraph에서 유효한 응답을 받았습니다!")
                            return {
                                'response': langgraph_result['response'],
                                'confidence': langgraph_result.get('confidence', 0.8),
                                'sources': langgraph_result.get('sources', []),
                                'workflow_steps': langgraph_result.get('workflow_steps', []),
                                'processing_time': time.time() - start_time,
                                'session_id': session_id,
                                'user_id': user_id,
                                'quality_metrics': langgraph_result.get('quality_metrics', {}),
                                'error_messages': langgraph_result.get('error_messages', []),
                                'intermediate_results': langgraph_result.get('intermediate_results', {}),
                                'langgraph_enabled': True,
                                'generation_method': 'langgraph_workflow'
                            }
                        else:
                            self.logger.warning("⚠️ LangGraph에서 유효한 응답을 받지 못했습니다.")
                            self.logger.warning(f"LangGraph 결과: {langgraph_result}")

                    except Exception as e:
                        self.logger.error(f"❌ LangGraph 워크플로우 실행 실패: {e}")
                        self.logger.error(f"오류 타입: {type(e).__name__}")
                        import traceback
                        self.logger.error(f"상세 오류: {traceback.format_exc()}")
                else:
                    self.logger.error("❌ LangGraph service initialization failed completely")
            else:
                self.logger.warning("⚠️ LangGraph 사용이 비활성화되어 있습니다.")

            # 2순위: 특정 법률 조문 검색 (LangGraph 실패 시)
            statute_law = query_analysis.get("statute_law")
            statute_article = query_analysis.get("statute_article")

            if statute_law and statute_article and self.current_law_search_engine:
                try:
                    self.logger.info(f"Searching specific law article: {statute_law} 제{statute_article}조")
                    specific_result = self.current_law_search_engine.search_by_law_article(
                        statute_law, statute_article
                    )

                    if specific_result and specific_result.article_content:
                        return {
                            "response": specific_result.article_content,
                            "confidence": 0.95,  # 특정 조문 검색은 높은 신뢰도
                            "sources": [{
                                "content": specific_result.article_content,
                                "law_name": specific_result.law_name_korean,
                                "article_number": statute_article,
                                "similarity": 1.0,
                                "source": "specific_article",
                                "type": "current_law"
                            }],
                            "query_analysis": query_analysis,
                            "generation_method": "specific_article",
                            "session_id": session_id,
                            "user_id": user_id
                        }
                    elif specific_result:
                        # 법률 정보는 있지만 조문을 찾지 못한 경우
                        return {
                            "response": f"'{statute_law} 제{statute_article}조'에 대한 정보를 찾았지만, 해당 조문의 전체내용을 가져올 수 없습니다.\n\n찾은 법률 정보:\n- 법령명: {specific_result.law_name_korean}\n- 소관부처: {specific_result.ministry_name}\n- 시행일: {specific_result.effective_date}\n\n더 전체적인 조문 내용이 필요하시면 국가법령정보센터(www.law.go.kr)에서 확인하시기 바랍니다.",
                            "confidence": 0.7,
                            "sources": [{
                                "content": f"법령명: {specific_result.law_name_korean}, 소관부처: {specific_result.ministry_name}",
                                "law_name": specific_result.law_name_korean,
                                "article_number": statute_article,
                                "similarity": 0.8,
                                "source": "law_info_only",
                                "type": "current_law"
                            }],
                            "query_analysis": query_analysis,
                            "generation_method": "law_info_only",
                            "session_id": session_id,
                            "user_id": user_id
                        }
                except Exception as e:
                    self.logger.debug(f"Specific law article search failed: {e}")

            # 3순위: UnifiedSearchEngine 사용 (LangGraph 및 특정 조문 검색 실패 시)
            if self.unified_search_engine:
                try:
                    self.logger.info(f"🔍 UnifiedSearchEngine으로 검색 수행: {message}")

                    # UnifiedSearchEngine으로 검색 수행
                    search_result = await self.unified_search_engine.search(
                        query=message,
                        top_k=5,
                        search_types=['vector', 'exact', 'current_law'],
                        category='all',
                        use_cache=True
                    )

                    self.logger.info(f"✅ UnifiedSearchEngine 검색 완료: {len(search_result.results)}개 결과")

                    if search_result.results:
                        # 검색 결과를 기반으로 답변 생성
                        sources = []
                        for result in search_result.results:
                            sources.append({
                                'content': result.get('content', ''),
                                'score': result.get('score', 0.0),
                                'source': result.get('source', 'unknown'),
                                'metadata': result.get('metadata', {})
                            })

                        # 간단한 답변 생성 (실제 LLM 사용)
                        if self.model_manager and hasattr(self.model_manager, 'generate_response'):
                            try:
                                context_text = "\n".join([f"- {source['content'][:200]}..." for source in sources[:3]])
                                prompt = f"""
다음 법률 문서를 참고하여 질문에 답변해주세요:

질문: {message}

참고 문서:
{context_text}

위 문서를 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.
"""

                                response_text = await self.model_manager.generate_response(prompt)

                                return {
                                    'response': response_text,
                                    'confidence': search_result.confidence,
                                    'sources': sources,
                                    'workflow_steps': ['unified_search_engine'],
                                    'processing_time': time.time() - start_time,
                                    'session_id': session_id,
                                    'user_id': user_id,
                                    'quality_metrics': {'search_results_count': len(sources)},
                                    'error_messages': [],
                                    'intermediate_results': {'search_result': search_result},
                                    'langgraph_enabled': False,
                                    'generation_method': 'unified_search_engine'
                                }
                            except Exception as e:
                                self.logger.warning(f"LLM 응답 생성 실패: {e}")

                        # LLM이 없으면 검색 결과만 반환
                        response_text = f"'{message}'에 대한 검색 결과를 찾았습니다:\n\n"
                        for i, source in enumerate(sources[:3], 1):
                            response_text += f"{i}. {source['content'][:150]}...\n"

                        return {
                            'response': response_text,
                            'confidence': search_result.confidence,
                            'sources': sources,
                            'workflow_steps': ['unified_search_engine'],
                            'processing_time': time.time() - start_time,
                            'session_id': session_id,
                            'user_id': user_id,
                            'quality_metrics': {'search_results_count': len(sources)},
                            'error_messages': [],
                            'intermediate_results': {'search_result': search_result},
                            'langgraph_enabled': False,
                            'generation_method': 'unified_search_engine'
                        }
                    else:
                        self.logger.warning("UnifiedSearchEngine에서 검색 결과를 찾지 못했습니다.")

                except Exception as e:
                    self.logger.error(f"UnifiedSearchEngine 검색 실패: {e}")
                    import traceback
                    self.logger.error(f"상세 오류: {traceback.format_exc()}")

            # 4순위: 기본 RAG 서비스 (UnifiedSearchEngine 실패 시)
            if self.unified_rag_service:
                try:
                    self.logger.info(f"Calling RAG service for query: {message}")
                    rag_response = await self.unified_rag_service.generate_response(
                        query=message,
                        context=query_analysis.get("context"),
                        max_length=800,  # 토큰 제한을 300에서 800으로 증가
                        top_k=3,  # 검색 결과 수를 2에서 3으로 증가 (더 많은 소스 확보)
                        use_cache=True
                    )

                    # 완화된 소스 검증 - 더 관대한 기준 적용
                    if rag_response and rag_response.response and self._has_meaningful_sources_relaxed(rag_response.sources):
                        # 응답 신뢰도 계산
                        confidence = self._calculate_confidence(rag_response.sources, "good")

                        # 스타일 적용된 응답 생성
                        final_response = rag_response.response
                        if detected_style and self.intelligent_style_system:
                            try:
                                final_response = self.intelligent_style_system.generate_adaptive_response(
                                    rag_response.response, message, query_analysis, session_id
                                )
                            except Exception as e:
                                self.logger.debug(f"Style application failed: {e}")

                        return {
                            "response": final_response,
                            "confidence": confidence,
                            "sources": rag_response.sources,
                            "query_analysis": query_analysis,
                            "generation_method": "rag_with_style",
                            "session_id": session_id,
                            "user_id": user_id,
                            "detected_style": detected_style.value if detected_style else "unknown"
                        }
                    else:
                        # 의미 있는 소스가 없으면 안내하고 알려줌
                        return self._create_no_sources_response(message, query_analysis, session_id, user_id)

                except Exception as e:
                    self.logger.debug(f"Simple RAG service failed: {e}")
            else:
                self.logger.warning("unified_rag_service is None, skipping RAG generation")

            # 4순위: 템플릿 기반 답변 (최후 수단)
            template_response = self._generate_improved_template_response(message, query_analysis, detected_style)
            if template_response and template_response.get("response"):
                self.logger.info("Using template-based response as fallback")
                return {
                    "response": template_response["response"],
                    "confidence": template_response.get("confidence", 0.8),
                    "sources": template_response.get("sources", []),
                    "query_analysis": query_analysis,
                    "generation_method": template_response.get("generation_method", "template"),
                    "session_id": session_id,
                    "user_id": user_id,
                    "detected_style": detected_style.value if detected_style else "unknown"
                }

            # 의미 있는 소스가 없으면 안내 답변으로 처리
            return self._create_no_sources_response(message, query_analysis, session_id, user_id)

        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            return self._create_error_response(message, query_analysis, session_id, user_id, str(e))

    def _has_meaningful_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """의미있는 법률 소스가 있는지 확인 - 기준치 강화 적용"""
        if not sources:
            return False

        # 더 엄격한 관련도 임계값 적용
        MIN_RELEVANCE_THRESHOLD = 0.4  # 0.3에서 0.4로 증가
        MIN_CONTENT_LENGTH = 60  # 50에서 60으로 증가

        meaningful_sources = []
        high_relevance_sources = []

        for source in sources:
            relevance_score = source.get("similarity", source.get("score", 0.0))
            content = source.get("content", "")

            # 관련도가 높고 내용이 충분한 소스만 유효한 소스로 판단
            if relevance_score >= MIN_RELEVANCE_THRESHOLD and len(content.strip()) > MIN_CONTENT_LENGTH:
                meaningful_sources.append(source)

                # 매우 높은 관련도 소스 별도 카운트
                if relevance_score >= 0.6:
                    high_relevance_sources.append(source)

        # 최소 1개 이상의 의미있는 소스가 있어야 유효
        if len(meaningful_sources) >= 1:
            # 추가 검증: 법률 관련 콘텐츠인지 확인 (강화된 키워드)
            legal_keywords = ["법률", "조문", "판례", "법령", "규정", "소송", "계약", "권리", "의무",
                           "민법", "형법", "상법", "헌법", "행정", "형사", "민사", "이혼", "상속", "재산분할",
                           "손해배상", "채권", "채무", "불법행위", "임금", "근로", "해고", "임대차", "매매"]
            legal_content_count = 0

            for source in meaningful_sources:
                content = source.get("content", "").lower()
                if any(keyword in content for keyword in legal_keywords):
                    legal_content_count += 1

            # 법률 관련 내용이 1개 이상이고 높은 관련도 소스가 있으면 유효
            return legal_content_count >= 1 and len(high_relevance_sources) >= 1

        return False

    def _has_meaningful_sources_relaxed(self, sources: List[Dict[str, Any]]) -> bool:
        """완화된 의미있는 법률 소스 확인 - 더 관대한 기준 적용"""
        if not sources:
            return False

        # 완화된 관련도 임계값 적용
        MIN_RELEVANCE_THRESHOLD = 0.2  # 0.4에서 0.2로 완화
        MIN_CONTENT_LENGTH = 30  # 60에서 30으로 완화

        meaningful_sources = []
        high_relevance_sources = []

        for source in sources:
            relevance_score = source.get("similarity", source.get("score", 0.0))
            content = source.get("content", "")

            # 완화된 기준으로 소스 검증
            if relevance_score >= MIN_RELEVANCE_THRESHOLD and len(content.strip()) > MIN_CONTENT_LENGTH:
                meaningful_sources.append(source)

                # 높은 관련도 소스 별도 카운트
                if relevance_score >= 0.4:  # 0.6에서 0.4로 완화
                    high_relevance_sources.append(source)

        # 최소 1개 이상의 의미있는 소스가 있으면 유효
        if len(meaningful_sources) >= 1:
            # 법률 관련 콘텐츠 확인 (완화된 키워드)
            legal_keywords = ["법률", "조문", "판례", "법령", "규정", "소송", "계약", "권리", "의무",
                           "민법", "형법", "상법", "헌법", "행정", "형사", "민사", "이혼", "상속", "재산분할",
                           "손해배상", "채권", "채무", "불법행위", "임금", "근로", "해고", "임대차", "매매",
                           "부동산", "가족", "회사", "주식", "이사", "노동", "근로기준법"]
            legal_content_count = 0

            for source in meaningful_sources:
                content = source.get("content", "").lower()
                if any(keyword in content for keyword in legal_keywords):
                    legal_content_count += 1

            # 법률 관련 내용이 있거나 높은 관련도 소스가 있으면 유효
            return legal_content_count >= 1 or len(high_relevance_sources) >= 1

        return False

    def _calculate_confidence(self, sources: List[Dict[str, Any]], response_quality: str = "good") -> float:
        """응답 신뢰도 계산 - 기준치 강화"""
        if not sources:
            return 0.0

        # 기본 신뢰도 (강화)
        base_confidence = 0.3  # 0.25에서 0.3으로 증가

        # 소스 품질에 따른 가중치 (강화된 기준)
        avg_relevance = sum(source.get("similarity", source.get("score", 0.0)) for source in sources) / len(sources)

        # 관련도에 따른 보너스 (강화된 기준)
        if avg_relevance >= 0.7:
            relevance_bonus = 0.4  # 매우 높은 관련도 (0.35에서 0.4로 증가)
        elif avg_relevance >= 0.5:
            relevance_bonus = 0.25  # 중간 관련도 (0.15에서 0.25로 증가)
        elif avg_relevance >= 0.3:
            relevance_bonus = 0.15  # 낮은 관련도 (0.05에서 0.15로 증가)
        else:
            relevance_bonus = 0.05  # 매우 낮은 관련도 (0.0에서 0.05로 증가)

        # 소스 개수에 따른 가중치 (강화된 기준)
        if len(sources) >= 3:
            source_count_bonus = 0.15  # 많은 소스 (0.1에서 0.15로 증가)
        elif len(sources) >= 2:
            source_count_bonus = 0.1  # 중간 소스 (0.05에서 0.1로 증가)
        else:
            source_count_bonus = 0.05  # 적은 소스 (0.0에서 0.05로 증가)

        # 응답 품질에 따른 가중치
        quality_bonus = 0.15 if response_quality == "excellent" else 0.1 if response_quality == "good" else 0.05

        # 최종 신뢰도 계산
        final_confidence = base_confidence + relevance_bonus + source_count_bonus + quality_bonus

        # 0.0 ~ 1.0 범위로 제한
        return max(0.0, min(1.0, final_confidence))

    def _create_no_sources_response(self, message: str, query_analysis: Dict[str, Any], session_id: str, user_id: str) -> Dict[str, Any]:
        """의미 있는 소스가 없을 때의 응답 생성"""
        query_type = query_analysis.get("query_type", "general")

        # 쿼리 타입별 맞춤 메시지
        if query_type == "legal_advice":
            response = f"""죄송합니다. '{message}'에 대한 전체적인 법률 정보를 찾을 수 없습니다.

다음과 같이 질문을 구체화해 주실 수 있습니까?
- 더 전체적인 상황을 알려주세요 (예: "민법 제750조 손해배상 청구권")
- 관련 법령 조문이나 판례가 있다면 함께 알려주세요
- 일반적인 법률 정보를 원하신다면 안내해드릴 수 있습니다

전체적인 법률 자문은 변호사와 직접 상담하시기 바랍니다."""

        elif query_type == "precedent":
            response = f"""죄송합니다. '{message}'에 관련된 판례를 찾을 수 없습니다.

다음과 같이 질문을 구체화해 주실 수 있습니까?
- 판결번호나 사건명을 정확하게 알려주세요
- 더 전체적인 키워드로 검색해주세요
- 관련 법령 조문을 함께 확인해보세요

판례 검색이 어려우시면 대법원 홈페이지나 법률 데이터베이스를 이용해보시기 바랍니다."""

        elif query_type == "law_inquiry":
            response = f"""죄송합니다. '{message}'에 대한 법령 정보를 찾을 수 없습니다.

다음과 같이 질문을 구체화해 주실 수 있습니까?
- 정확한 법령명과 조문번호를 알려주세요 (예: "민법 제750조")
- 법령의 공식 명칭을 확인해주세요
- 관련 키워드를 더 전체적으로 작성해주세요

법령 정보는 국가법령정보센터(www.law.go.kr)에서 확인하실 수 있습니다."""

        else:
            response = f"""죄송합니다. '{message}'에 대한 관련 정보를 찾을 수 없습니다.

다음과 같이 질문을 구체화해 주실 수 있습니까?
- 질문을 더 전체적으로 작성해주세요
- 관련 법령 조문이나 판례를 알려주세요
- 키워드를 더 정확하게 작성해주세요

일반적인 법률 안내나 정보에 대해서는 안내해드릴 수 있습니다."""

        # 검색 제안 생성
        suggestions = self._generate_search_suggestions(message, query_analysis)
        suggestion_text = suggestions[0] if suggestions else "질문을 더 전체적으로 작성해주세요"

        return {
            "response": response,
            "confidence": 0.1,  # 0.0에서 0.1로 증가 - 안내문 제공 시 기본 신뢰도
            "sources": [],
            "query_analysis": query_analysis,
            "generation_method": "no_sources",
            "session_id": session_id,
            "user_id": user_id,
            "no_sources": True,
            "suggestion": suggestion_text
        }

    def _create_error_response(self, message: str, query_analysis: Dict[str, Any], session_id: str, user_id: str, error: str) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            "response": f"죄송합니다. '{message}'에 대한 답변 생성 중 오류가 발생했습니다.\n\n오류: {error}\n\n잠시 후 다시 시도해주세요.",
            "confidence": 0.0,
            "sources": [],
            "query_analysis": query_analysis,
            "generation_method": "error",
            "session_id": session_id,
            "user_id": user_id,
            "error": error
        }

    def _generate_search_suggestions(self, message: str, query_analysis: Dict[str, Any]) -> List[str]:
        """검색 제안 생성 - 간소화된 방법"""
        suggestions = []

        # 정규식으로 키워드 추출 (간소화된 방법)
        import re

        # 법률 도메인 키워드 추출 (확실한 패턴)
        law_patterns = [
            r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법)',
            r'(계약|손해배상|불법행위|채권|채무)',
            r'(이혼|상속|양육권|재산분할|가족)',
            r'(회사|주식|이사|상법|상행위)',
            r'(근로|임금|해고|노동법|근로기준법)',
            r'(부동산|매매|임대차|등기|부동산등기법)',
            r'(법률|법령|조문|판례|법원|법정)'
        ]

        extracted_keywords = []
        for pattern in law_patterns:
            matches = re.findall(pattern, message)
            if matches:
                if isinstance(matches[0], tuple):
                    extracted_keywords.extend([match for match in matches[0] if match])
                else:
                    extracted_keywords.extend(matches)

        # 중복 제거 및 우선순위 정렬
        extracted_keywords = list(set(extracted_keywords))

        # 키워드 우선순위 정렬 (법률명 > 법조문 > 일반 조문)
        priority_keywords = []
        specific_keywords = []
        general_keywords = []

        for keyword in extracted_keywords:
            if keyword in ["민법", "형법", "상법", "노동법", "가족법", "행정법", "헌법", "민사소송법", "형사소송법"]:
                priority_keywords.append(keyword)
            elif keyword in ["계약", "손해배상", "불법행위", "채권", "채무", "이혼", "상속", "양육권", "재산분할", "가족", "회사", "주식", "이사", "상법", "상행위", "근로", "임금", "해고", "노동법", "근로기준법", "부동산", "매매", "임대차", "등기", "부동산등기법"]:
                specific_keywords.append(keyword)
            else:
                general_keywords.append(keyword)

        # 우선순위 키워드 먼저, 그 다음 특정 법조문 키워드, 마지막에 일반 키워드
        extracted_keywords = priority_keywords + specific_keywords + general_keywords

        # 질문 분석에서 키워드 추출 시도 (fallback) - 우선순위 정렬 적용
        if not extracted_keywords:
            keywords = query_analysis.get("keywords", [])
            if keywords:
                # fallback 키워드도 우선순위 정렬 적용
                priority_keywords = []
                specific_keywords = []
                general_keywords = []

                for keyword in keywords:
                    if keyword in ["민법", "형법", "상법", "노동법", "가족법", "행정법", "헌법", "민사소송법", "형사소송법"]:
                        priority_keywords.append(keyword)
                    elif keyword in ["계약", "손해배상", "불법행위", "채권", "채무", "이혼", "상속", "양육권", "재산분할", "가족", "회사", "주식", "이사", "상법", "상행위", "근로", "임금", "해고", "노동법", "근로기준법", "부동산", "매매", "임대차", "등기", "부동산등기법"]:
                        specific_keywords.append(keyword)
                    else:
                        general_keywords.append(keyword)

                # 우선순위 키워드 먼저, 그 다음 특정 법조문 키워드, 마지막에 일반 키워드
                extracted_keywords = priority_keywords + specific_keywords + general_keywords

        # 추출된 키워드로 제안 생성 (이미 우선순위 정렬됨)
        if extracted_keywords:
            main_keyword = extracted_keywords[0]
            suggestions.append(f"'{main_keyword}' 관련 법률 조문을 검색해보세요")
            suggestions.append(f"'{main_keyword}' 판례를 찾아보세요")
            if len(extracted_keywords) > 1:
                suggestions.append(f"'{extracted_keywords[1]}'도 함께 검색해보세요")

        # 질문 유형별 제안
        query_type = query_analysis.get("query_type", "general")
        if query_type == "legal_advice":
            suggestions.extend([
                "법률 조문 상황을 설명해주세요",
                "관련 법령이나 판례를 찾아주세요"
            ])
        elif query_type == "precedent":
            suggestions.extend([
                "판례번호를 정확히 입력해주세요",
                "해당 법조문 키워드로 검색해보세요"
            ])
        elif query_type == "law_inquiry":
            suggestions.extend([
                "정확한 법률명과 조문번호를 입력해주세요",
                "법률의 공식 명칭을 확인해주세요"
            ])

        # 일반적인 제안 (키워드가 없는 경우)
        if not suggestions:
            suggestions.extend([
                "구체적인 법률 조문을 작성해주세요",
                "관련 법령이나 판례를 찾아주세요",
                "키워드를 더 정확히 입력해주세요"
            ])

        return suggestions[:3]  # 최대 3개 제안

    def _generate_improved_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """개선된 템플릿 기반 답변 생성 - 스타일 지원"""
        self.logger.info(f"_generate_improved_template_response called for: {message}")

        # 도메인별 특화 템플릿 답변 생성
        message_lower = message.lower()

        # 계약서 관련 질문 처리
        if any(keyword in message_lower for keyword in ["계약서", "계약", "작성", "체결"]):
            return self._generate_contract_template_response(message, query_analysis, detected_style)

        # 부동산 관련 질문 처리
        elif any(keyword in message_lower for keyword in ["부동산", "매매", "임대차", "등기"]):
            return self._generate_real_estate_template_response(message, query_analysis, detected_style)

        # 가족법 관련 질문 처리
        elif any(keyword in message_lower for keyword in ["이혼", "상속", "양육권", "재산분할"]):
            return self._generate_family_law_template_response(message, query_analysis, detected_style)

        # 법률 조문 관련 질문 처리
        elif query_analysis.get("statute_match"):
            return self._generate_statute_template_response(message, query_analysis, detected_style)

        # 기본 템플릿 답변
        return self._generate_general_template_response(message, query_analysis, detected_style)

    def _generate_contract_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """계약서 관련 템플릿 답변 생성"""
        response = """📋 **계약서 작성을 도와드리겠습니다!**

어떤 종류의 계약서를 작성하시나요?

○ **용역계약** (디자인, 개발, 컨설팅 등)
○ **근로계약** (직원 채용)
○ **부동산계약** (매매, 임대차)
○ **지적재산권계약** (저작권, 특허 등)
○ **제휴계약** (업무 협력)
○ **기타**

## 📝 계약서 작성 기본 원칙

1. **당사자 정보**: 정확한 이름, 주소, 연락처
2. **계약 목적**: 구체적이고 명확한 내용
3. **대금 및 지급**: 금액, 지급 시기, 방법
4. **계약 기간**: 시작일, 종료일, 연장 조건
5. **위약 조항**: 계약 위반 시 손해배상
6. **분쟁 해결**: 조정, 중재, 관할 법원

## ⚠️ 중요 안내
- 계약서는 나중에 해석의 여지가 없도록 명확하게 작성하세요
- 중요한 계약은 변호사 검토를 권장합니다
- 계약 금액이 큰 경우 전문가 상담이 필요합니다

구체적인 계약 유형을 알려주시면 더 자세한 가이드를 제공해드리겠습니다!"""

        return {
            "response": response,
            "confidence": 0.90,
            "generation_method": "contract_template",
            "sources": [],
            "query_analysis": query_analysis
        }

    def _generate_real_estate_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """부동산 관련 템플릿 답변 생성"""
        response = """🏠 **부동산 관련 도움을 드리겠습니다!**

어떤 부동산 관련 질문이 있으신가요?

○ **매매 절차** (부동산 구매/판매)
○ **임대차 계약** (전세, 월세)
○ **등기 절차** (소유권 이전)
○ **부동산 세금** (취득세, 양도세)
○ **부동산 분쟁** (계약 분쟁, 권리 분쟁)
○ **기타**

## 📋 부동산 거래 기본 절차

### 매매 거래
1. **물건 확인** → 등기부등본, 건축물대장 확인
2. **계약 체결** → 매매계약서 작성, 계약금 지급
3. **중도금 지급** → 중도금 지급, 근저당 해지
4. **잔금 지급** → 잔금 지급, 소유권 이전 등기
5. **세금 납부** → 취득세, 등록면허세 납부

### 임대차 계약
1. **물건 확인** → 전세금 확인, 월세금 확인
2. **계약 체결** → 임대차계약서 작성, 보증금 지급
3. **입주** → 전입신고, 확정일자 받기
4. **계약 종료** → 보증금 반환, 원상복구

구체적인 상황을 알려주시면 더 자세한 도움을 드리겠습니다!"""

        return {
            "response": response,
            "confidence": 0.90,
            "generation_method": "real_estate_template",
            "sources": [],
            "query_analysis": query_analysis
        }

    def _generate_family_law_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """가족법 관련 템플릿 답변 생성"""
        response = """👨‍👩‍👧‍👦 **가족법 관련 도움을 드리겠습니다!**

어떤 가족법 관련 질문이 있으신가요?

○ **이혼 절차** (협의이혼, 재판이혼)
○ **상속 문제** (상속분, 유언, 상속포기)
○ **양육권** (자녀 양육권 문제, 양육비)
○ **재산분할** (이혼 시 재산 분할, 위자료)
○ **입양** (입양 절차, 친양자 입양)
○ **기타**

## 📋 주요 가족법 절차

### 이혼 절차
1. **협의이혼**: 이혼 합의서 작성 후 이혼 신고
2. **재판이혼**: 법원에 이혼 소송 제기

### 상속 문제
1. **상속인 확인**: 법정상속인, 유언상속인 확인
2. **상속분 결정**: 상속분 계산, 분할 협의
3. **유산 분할**: 유산목록 작성, 상속재산 분할
4. **유언집행**: 유언집행자 선임, 유언 집행

### 양육권 문제
1. **자녀 양육권**: 친권자 결정, 양육비 결정
2. **양육비 지급**: 양육비 지급, 면접교섭권
3. **친권자변경**: 친권자 변경 신청

구체적인 상황을 알려주시면 더 자세한 도움을 드리겠습니다!"""

        return {
            "response": response,
            "confidence": 0.90,
            "generation_method": "family_law_template",
            "sources": [],
            "query_analysis": query_analysis
        }

    def _generate_statute_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """법률 조문 관련 템플릿 답변 생성"""
        statute_law = query_analysis.get("statute_law")
        statute_article = query_analysis.get("statute_article")

        response = f"""📖 **법률 조문 관련 도움을 드리겠습니다!**

{'**' + statute_law + ' 제' + statute_article + '조**' if statute_law and statute_article else '**해당 법률 조문**'}에 대해 도움을 드리겠습니다.

## 📋 법률 조문 핵심 가이드

### 조문 기본 정보
1. **조문 번호**: 해당 조문
2. **조문 내용**: 조문의 내용
3. **조문 해석**: 법조문의 의미 해석
4. **조문 적용**: 조문의 다른 조문과의 관계

### 핵심 포인트
1. **문언해석**: 조문의 문언의 의미 파악
2. **목적해석**: 조문의 입법 목적 고려
3. **체계해석**: 다른 조문들과의 관계 고려
4. **판례해석**: 관련 판례의 해석 방법

## ⚠️ 법률 판례 및 핵심
- 해당 조문의 주요 판례
- 조문의 핵심 내용
- 의미 있는 사례들

구체적인 법률 조문이나 관련 판례에 대해 더 자세한 도움을 드리겠습니다!"""

        return {
            "response": response,
            "confidence": 0.85,
            "generation_method": "statute_template",
            "sources": [],
            "query_analysis": query_analysis
        }

    def _generate_general_template_response(self, message: str, query_analysis: Dict[str, Any], detected_style: ResponseStyle = None) -> Dict[str, Any]:
        """일반 템플릿 답변 생성"""
        response = """⚖️ **법률 관련 도움을 드리겠습니다!**

어떤 법률 관련 질문이 필요하신가요?

○ **민사법** (계약, 손해배상, 불법행위)
○ **형사법** (범죄, 처벌, 형량)
○ **가족법** (이혼, 상속, 양육권)
○ **상법** (회사, 주식, 이사)
○ **노동법** (근로, 임금, 해고)
○ **부동산법** (매매, 임대차, 등기)
○ **기타**

## 📋 법률 질문 가이드

### 효과적인 질문 작성 방법
1. **법률 조문 상황 설명**: 구체적인 상황, 시간, 장소
2. **관련법령 찾기**: 계약서, 판례, 법원판결 등
3. **정확한 질문**: 구체적인 답변을 원하는 질문
4. **상황별 대응**: 법률적 문제의 상황별 대응

### 법률 정보 활용 팁
- **법령**: 정확한 법률 조문 확인
- **판례**: 관련 판례 해석 방법 파악
- **사례**: 법률 적용 사례들 참고

구체적인 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다!"""

        return {
            "response": response,
            "confidence": 0.80,
            "generation_method": "general_template",
            "sources": [],
            "query_analysis": query_analysis
        }

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 결과를 바탕으로 추천사항 생성"""
        recommendations = []

        try:
            # 메모리 사용량 분석
            memory_usage = analysis_result.get('memory_usage', {})
            if memory_usage.get('usage_percent', 0) > 80:
                recommendations.append({
                    'type': 'warning',
                    'title': '메모리 사용량 경고',
                    'description': f"현재 메모리 사용률: {memory_usage.get('usage_percent', 0):.1f}%",
                    'action': '메모리 정리 권장',
                    'command': 'service.perform_memory_cleanup()'
                })

            # 응답 시간 분석
            response_time = analysis_result.get('response_time', 0)
            if response_time > 10:
                recommendations.append({
                    'type': 'performance',
                    'title': '응답 시간 개선 필요',
                    'description': f"평균 응답 시간: {response_time:.2f}초",
                    'action': '성능 최적화 권장',
                    'command': 'service._optimize_performance()'
                })

            # 컴포넌트 상태 분석
            components = analysis_result.get('components', {})
            for comp_name, comp_info in components.items():
                if comp_info.get('status') == 'error':
                    recommendations.append({
                        'type': 'error',
                        'title': f'{comp_name} 컴포넌트 오류',
                        'description': comp_info.get('error', '알 수 없는 오류'),
                        'action': '컴포넌트 재시작',
                        'command': f'service._restart_component("{comp_name}")'
                    })

            # 기본 추천사항
            if not recommendations:
                recommendations.append({
                    'type': 'info',
                    'title': '시스템 상태 양호',
                    'description': '현재 시스템이 정상적으로 작동하고 있습니다.',
                    'action': '정기적인 모니터링 유지',
                    'command': 'service._cleanup_components()'
                })

        except Exception as e:
            recommendations.append({
                'type': 'error',
                'title': '추천사항 생성 오류',
                'description': str(e),
                'action': '시스템 로그를 확인하세요',
                'command': None
            })

        return recommendations

    def _add_fallback_ending(self, response: str) -> str:
        """답변 마무리 추가"""
        try:
            # 불완전한 문장 패턴 검사
            incomplete_patterns = [
                r'다$', r'그래서$', r'때문$', r'있습니다$', r'됩니다$',
                r'해야 할$', r'전체적으로$', r'특히$', r'또한$',
                r'[가-힣]+며$', r'[가-힣]+고$', r'[가-힣]+면$'
            ]

            import re
            for pattern in incomplete_patterns:
                if re.search(pattern, response.strip()):
                    # 불완전한 부분을 자연스럽게 마무리
                    if response.strip().endswith('다'):
                        return f"{response.strip()} 이렇게 하시면 됩니다."
                    elif response.strip().endswith(('그래서', '때문')):
                        return f"{response.strip()} 등 사항을 고려 하시면 해결방법을 찾으실겁니다."
                    else:
                        return f"{response.strip()} 이렇게 하시면 됩니다."

            # 정상적 마무리가 없는지 검사
            if not response.strip().endswith(('.', '!', '?', '니다.', '습니다.', '요.')):
                return f"{response.strip()} 이렇게 하시면 됩니다."

            return response

        except Exception as e:
            self.logger.error(f"답변 마무리 추가 실패: {e}")
            return response

    # 새로운 법률 검색 및 답변 최적화 메서드들

    def _initialize_enhanced_law_search(self):
        """향상된 법률 검색 시스템 초기화 - 테스트를 위해 비활성화"""
        try:
            # 모든 향상된 법률 검색 시스템을 None으로 설정 (테스트용)
            self.precedent_service = None
            self.enhanced_law_search_engine = None
            self.integrated_law_search = None
            self.adaptive_response_manager = None
            self.progressive_response_system = None

            # 법률 조문 쿼리 패턴 (기본 패턴만 유지)
            self.law_query_patterns = [
                r'(\w+법)\s*제\s*(\d+)조',
                r'제\s*(\d+)조',
                r'(\w+법)\s*(\d+)조',
                r'(\w+법)\s*제\s*(\d+)조\s*제\s*(\d+)항'
            ]

            self.logger.info("향상된 법률 검색 시스템 초기화 완료 (테스트 모드 - 모든 시스템 비활성화)")

        except Exception as e:
            self.logger.error(f"향상된 법률 검색 시스템 초기화 실패: {e}")
            self.precedent_service = None
            self.enhanced_law_search_engine = None
            self.integrated_law_search = None
            self.adaptive_response_manager = None
            self.progressive_response_system = None

    def _initialize_langgraph_workflow(self):
        """LangGraph 워크플로우 서비스 초기화"""
        self.logger.info("🔍 _initialize_langgraph_workflow 메서드 호출됨")
        self.logger.info("=" * 70)
        self.logger.info("🔍 LangGraph 초기화 진단 시작")
        self.logger.info("=" * 70)

        try:
            self.logger.info("🚀 LangGraph 워크플로우 서비스 초기화 시작...")
            self.logger.info("📍 단계 1: 기본 LangGraph 모듈 import 테스트")

            # 먼저 기본 LangGraph 모듈 import 테스트 (강화된 방식)
            try:
                # 여러 방법으로 import 시도
                import os
                import sys

                # 현재 디렉토리를 Python 경로에 추가
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)

                # LangGraph import 시도
                self.logger.info("   → langgraph.graph에서 END, StateGraph import 시도...")
                from langgraph.graph import END, StateGraph
                self.logger.info("✅ 기본 LangGraph 모듈 import 성공")
                self.logger.info(f"   → StateGraph 클래스: {StateGraph}")
                self.logger.info(f"   → END 상수: {END}")

                # langgraph 버전 확인
                try:
                    import langgraph
                    version = getattr(langgraph, '__version__', 'unknown')
                    self.logger.info(f"   → langgraph 버전: {version}")
                except Exception as e:
                    self.logger.debug(f"버전 확인 실패: {e}")

            except ImportError as e:
                self.logger.error(f"❌ 기본 LangGraph 모듈 import 실패: {e}")
                self.logger.error(f"Python 경로: {sys.path[:3]}...")  # 처음 3개만 표시
                self.logger.error(f"현재 작업 디렉토리: {os.getcwd()}")

                # 추가 디버깅 정보
                try:
                    import langgraph
                    self.logger.error(f"langgraph 모듈은 존재: {langgraph}")
                    self.logger.error(f"langgraph 경로: {getattr(langgraph, '__path__', 'No path')}")
                except Exception as debug_e:
                    self.logger.error(f"langgraph 모듈도 없음: {debug_e}")

                self.langgraph_service = None
                self.use_langgraph = False  # LangGraph 사용 불가로 설정
                return

            # 프로젝트 모듈 import
            self.logger.info("📍 단계 2: 프로젝트 LangGraph 모듈 import")
            try:
                self.logger.info("   → langgraph_config import 시도...")
                from ...utils.langgraph_config import langgraph_config
                self.logger.info("   → langgraph_config import 성공")

                self.logger.info("   → IntegratedWorkflowService import 시도...")
                from ..langgraph_workflow.integrated_workflow_service import (
                    IntegratedWorkflowService,
                )
                self.logger.info("✅ 프로젝트 LangGraph 모듈 import 성공")
            except ImportError as e:
                self.logger.error(f"❌ 프로젝트 LangGraph 모듈 import 실패: {e}")
                self.langgraph_service = None
                self.use_langgraph = False  # LangGraph 사용 불가로 설정
                return

            # 설정 검증
            self.logger.info("📍 단계 3: LangGraph 설정 검증")
            config_errors = langgraph_config.validate()
            if config_errors:
                self.logger.warning(f"⚠️ LangGraph 설정 오류: {config_errors}")
            else:
                self.logger.info("✅ 설정 검증 통과")

            # LangGraph 활성화 여부 확인
            self.logger.info(f"   → langgraph_enabled: {langgraph_config.langgraph_enabled}")
            if not langgraph_config.langgraph_enabled:
                self.logger.warning("⚠️ LangGraph가 비활성화되어 있습니다.")
                self.langgraph_service = None
                self.use_langgraph = False  # LangGraph 사용 불가로 설정
                return

            self.logger.info(f"📋 LangGraph 설정: {langgraph_config.to_dict()}")

            # 워크플로우 서비스 초기화
            self.logger.info("📍 단계 4: IntegratedWorkflowService 초기화")
            try:
                self.logger.info("   → IntegratedWorkflowService 인스턴스 생성 중...")
                self.langgraph_service = IntegratedWorkflowService(langgraph_config)
                self.logger.info("🎉 LangGraph 워크플로우 서비스 초기화 완료")
                self.logger.info(f"   → LangGraph 서비스 타입: {type(self.langgraph_service).__name__}")
                self.logger.info(f"   → process_query 메서드 존재: {hasattr(self.langgraph_service, 'process_query')}")
                self.use_langgraph = True  # LangGraph 정상 초기화됨
            except Exception as init_e:
                self.logger.error(f"❌ IntegratedWorkflowService 초기화 실패: {init_e}")
                import traceback
                self.logger.error(f"상세 오류: {traceback.format_exc()}")
                self.langgraph_service = None
                self.use_langgraph = False  # LangGraph 사용 불가로 설정
                return

        except ImportError as e:
            self.logger.error(f"❌ LangGraph 모듈 import 실패: {e}")
            self.logger.error("LangGraph 관련 패키지가 설치되지 않았을 수 있습니다.")
            self.logger.error("다음 명령어로 설치하세요: pip install langgraph langchain-core langchain-community")
            self.langgraph_service = None
            self.use_langgraph = False  # LangGraph 사용 불가로 설정
        except Exception as e:
            self.logger.error(f"❌ LangGraph 워크플로우 서비스 초기화 실패: {e}")
            self.logger.error(f"오류 타입: {type(e).__name__}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            self.langgraph_service = None
            self.use_langgraph = False  # LangGraph 사용 불가로 설정

    def _initialize_intelligent_style_system(self):
        """지능형 응답 스타일 시스템 초기화 - 테스트를 위해 주석 처리"""
        try:
            # self.intelligent_style_system = IntelligentResponseStyleSystem()
            self.intelligent_style_system = None  # 테스트용으로 None 설정
            self.logger.info("지능형 응답 스타일 시스템 초기화 완료 (테스트 모드)")
        except Exception as e:
            self.logger.error(f"지능형 응답 스타일 시스템 초기화 실패: {e}")
            self.intelligent_style_system = None

    def _is_law_article_query(self, query: str) -> bool:
        """법률 조문 쿼리인지 확인"""
        try:
            import re
            for pattern in self.law_query_patterns:
                if re.search(pattern, query):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"법률 조문 쿼리 확인 실패: {e}")
            return False

    async def _handle_law_article_query(self, message: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """법률 조문 쿼리 처리"""
        start_time = time.time()

        try:
            # 🆕 current_law_search_engine 사용 (적극적 활용)
            if self.current_law_search_engine:
                # 법률 조문 추출
                article_info = self._extract_law_article_from_query(message)

                if article_info and article_info.get('law_name') and article_info.get('article_number'):
                    self.logger.info(f"🔍 특정 조문 검색: {article_info['law_name']} 제{article_info['article_number']}조")

                    # 조문 검색
                    search_result = self.current_law_search_engine.search_by_law_article(
                        article_info['law_name'],
                        article_info['article_number']
                    )

                    if search_result and search_result.article_content:
                        return {
                            'response': search_result.article_content,
                            'confidence': 0.95,
                            'sources': [{
                                'content': search_result.article_content,
                                'law_name': search_result.law_name_korean,
                                'article_number': article_info['article_number'],
                                'similarity': 1.0,
                                'source': 'current_law'
                            }],
                            'processing_time': time.time() - start_time,
                            'generation_method': 'law_article',
                            'session_id': session_id,
                            'user_id': user_id
                        }

            # 폴백: integrated_law_search 사용
            if self.integrated_law_search:
                # 법률 조문 검색 실행
                search_result = await self.integrated_law_search.search_law_article(message)

            # 사용자 컨텍스트 분석
            user_context = await self._analyze_user_context(user_id, session_id)

            # 적응형 답변 길이 조정 (법률 해석 제외)
            if self.adaptive_response_manager and "법률 해석:" not in search_result.response:
                optimized_response = self.adaptive_response_manager.adapt_response_length(
                    search_result.response, user_context
                )
            else:
                optimized_response = search_result.response

            # 단계별 답변 생성 (법률 해석 제외)
            if self.progressive_response_system and "법률 해석:" not in optimized_response:
                progressive_response = self.progressive_response_system.generate_progressive_response(
                    optimized_response, user_context.get('response_level', 'standard')
                )
                final_response = progressive_response.response
                additional_options = progressive_response.additional_options
            else:
                final_response = optimized_response
                additional_options = []

            return {
                'response': final_response,
                'confidence': search_result.confidence,
                'sources': search_result.sources,
                'processing_time': time.time() - start_time,
                'generation_method': 'integrated_law_search',
                'restricted': False,
                'context_info': search_result.context_info,
                'additional_options': additional_options,
                'has_more_detail': len(search_result.response) > len(final_response)
            }

        except Exception as e:
            self.logger.error(f"법률 조문 쿼리 처리 실패: {e}")
            return await self._fallback_response(message)

    async def _analyze_user_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """사용자 컨텍스트 분석"""
        try:
            context = {
                'user_id': user_id,
                'session_id': session_id,
                'expertise_level': 'beginner',
                'response_level': 'standard',
                'device_info': {'type': 'desktop'},
                'preferred_length': 1000
            }

            # 사용자 프로필 정보 가져오기
            if hasattr(self, 'user_preferences') and self.user_preferences:
                try:
                    user_profile = self.user_preferences.get_user_profile(user_id)
                    if user_profile:
                        context.update({
                            'expertise_level': user_profile.get('expertise_level', 'beginner'),
                            'response_level': user_profile.get('preferred_detail_level', 'standard'),
                            'device_info': user_profile.get('device_info', {'type': 'desktop'}),
                            'preferred_length': self._get_preferred_length(user_profile)
                        })
                except Exception as e:
                    self.logger.debug(f"사용자 프로필 조회 실패: {e}")

            return context

        except Exception as e:
            self.logger.error(f"사용자 컨텍스트 분석 실패: {e}")
            return {
                'user_id': user_id,
                'session_id': session_id,
                'expertise_level': 'beginner',
                'response_level': 'standard',
                'device_info': {'type': 'desktop'},
                'preferred_length': 1000
            }

    def _get_preferred_length(self, user_profile: Dict[str, Any]) -> int:
        """사용자 프로필에서 선호 길이 계산"""
        try:
            expertise_level = user_profile.get('expertise_level', 'beginner')
            detail_level = user_profile.get('preferred_detail_level', 'medium')
            device_type = user_profile.get('device_info', {}).get('type', 'desktop')

            # 기본 길이 설정 (추가 최적화 - 더 긴 답변 제공)
            base_lengths = {
                'mobile': 800,   # 모바일에서 더 긴 답변 제공
                'desktop': 2000, # 데스크톱에서 더 긴 답변 제공
                'tablet': 1200   # 태블릿에서 더 긴 답변 제공
            }

            base_length = base_lengths.get(device_type, 2000)

            # 전문성 수준에 따른 배율
            expertise_multipliers = {
                'beginner': 0.8,
                'intermediate': 1.0,
                'expert': 1.2,
                'professional': 1.3
            }

            multiplier = expertise_multipliers.get(expertise_level, 1.0)

            # 상세도에 따른 배율
            detail_multipliers = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.3
            }

            detail_multiplier = detail_multipliers.get(detail_level, 1.0)

            return int(base_length * multiplier * detail_multiplier)

        except Exception as e:
            self.logger.error(f"선호 길이 계산 실패: {e}")
            return 1500  # 기본 길이를 1000에서 1500으로 증가

    async def get_expanded_response(self, base_response: str, option_type: str, user_id: str = None) -> str:
        """확장된 답변 생성"""
        try:
            if not self.progressive_response_system:
                return base_response

            # 사용자 컨텍스트 분석
            user_context = await self._analyze_user_context(user_id, None)

            # 확장된 답변 생성
            expanded_response = self.progressive_response_system.generate_expanded_response(
                base_response, option_type, base_response
            )

            # 적응형 길이 조정
            if self.adaptive_response_manager:
                optimized_response = self.adaptive_response_manager.adapt_response_length(
                    expanded_response, user_context
                )
                return optimized_response

            return expanded_response

        except Exception as e:
            self.logger.error(f"확장된 답변 생성 실패: {e}")
            return base_response

    async def _fallback_response(self, message: str) -> Dict[str, Any]:
        """폴백 응답 생성"""
        return {
            'response': f"죄송합니다. '{message}'에 대한 정보를 찾을 수 없습니다. 다른 방식으로 질문해주시겠습니까?",
            'confidence': 0.1,
            'sources': [],
            'processing_time': 0.0,
            'generation_method': 'fallback',
            'restricted': False,
            'session_id': '',
            'user_id': ''
        }

    def get_hybrid_classifier_stats(self) -> Dict[str, Any]:
        """하이브리드 분류기 통계 반환"""
        if self.hybrid_classifier:
            return self.hybrid_classifier.get_stats()
        return {"error": "하이브리드 분류기가 초기화되지 않음"}

    def adjust_classifier_threshold(self, new_threshold: float):
        """하이브리드 분류기 임계값 조정"""
        if self.hybrid_classifier:
            self.hybrid_classifier.adjust_threshold(new_threshold)
            self.logger.info(f"하이브리드 분류기 임계값 조정: {new_threshold}")
        else:
            self.logger.warning("하이브리드 분류기가 초기화되지 않음")

    def train_hybrid_classifier(self, training_data: List[Tuple[str, str]]):
        """하이브리드 분류기 ML 모델 학습"""
        if not self.hybrid_classifier:
            self.logger.error("하이브리드 분류기가 초기화되지 않음")
            return False

        try:
            # 문자열을 UnifiedQuestionType으로 변환
            from .unified_question_types import UnifiedQuestionType
            converted_data = []
            for question, question_type_str in training_data:
                question_type = UnifiedQuestionType.from_string(question_type_str)
                converted_data.append((question, question_type))

            # ML 모델 학습
            self.hybrid_classifier.train_ml_model(converted_data)
            self.logger.info(f"하이브리드 분류기 ML 모델 학습 완료: {len(training_data)}개 데이터")
            return True

        except Exception as e:
            self.logger.error(f"하이브리드 분류기 학습 실패: {e}")
            return False

    def _extract_law_article_from_query(self, message: str) -> Dict[str, Any]:
        """메시지에서 법률 조문 정보 추출"""
        try:
            import re

            # 확장된 법률 조문 패턴
            statute_patterns = {
                'standard': r'([\w가-힣]+법)\s*제\s*(\d+)\s*조',  # 민법 제750조
                'compact': r'([\w가-힣]+법)제(\d+)조',           # 민법제750조
                'with_clause': r'([\w가-힣]+법)\s*제\s*(\d+)\s*조\s*제\s*(\d+)\s*항',  # 민법 제750조 제1항
                'simple': r'제\s*(\d+)\s*조',                      # 제750조
                'number_only': r'(\d+)\s*조'                       # 750조
            }

            for pattern_name, pattern in statute_patterns.items():
                match = re.search(pattern, message)
                if match:
                    groups = match.groups()

                    if pattern_name == 'with_clause' and len(groups) == 3:
                        return {
                            'law_name': groups[0],
                            'article_number': groups[1],
                            'clause_number': groups[2],
                            'pattern_type': pattern_name,
                            'full_match': match.group(0)
                        }
                    elif len(groups) == 2:
                        return {
                            'law_name': groups[0],
                            'article_number': groups[1],
                            'clause_number': None,
                            'pattern_type': pattern_name,
                            'full_match': match.group(0)
                        }
                    elif len(groups) == 1:
                        return {
                            'law_name': None,
                            'article_number': groups[0],
                            'clause_number': None,
                            'pattern_type': pattern_name,
                            'full_match': match.group(0)
                        }

            return None

        except Exception as e:
            self.logger.error(f"Error extracting law article: {e}")
            return None
