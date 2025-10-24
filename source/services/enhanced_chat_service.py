# -*- coding: utf-8 -*-
"""
Enhanced Chat Service
개선된 채팅 메시지 처리 서비스
"""

import os
import time
import asyncio
import hashlib
import gc
import weakref
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.memory_manager import get_memory_manager, MemoryManager
from ..utils.weakref_cleanup import get_weakref_registry, WeakRefRegistry
from ..utils.realtime_memory_monitor import get_memory_monitor, RealTimeMemoryMonitor
from ..utils.advanced_response_processor import advanced_response_processor
from ..utils.quality_validator import quality_validator
from .user_preference_manager import preference_manager, UserPreferences
from .answer_completion_validator import completion_validator, CompletionCheck
from .example_database import example_database, dynamic_generator
from .enhanced_completion_system import enhanced_completion_system, CompletionResult
from .rag_service import MLEnhancedRAGService
from .hybrid_search_engine import HybridSearchEngine
from .improved_answer_generator import ImprovedAnswerGenerator
from .question_classifier import QuestionClassifier
from ..models.model_manager import LegalModelManager
from ..data.vector_store import LegalVectorStore
from ..data.database import DatabaseManager

# 법률 도메인 키워드 정의
LEGAL_DOMAIN_KEYWORDS = {
    "civil_law": {
        "primary": ["민법", "계약", "손해배상", "불법행위", "채권", "채무", "소유권", "물권"],
        "secondary": ["계약서", "위약금", "손해", "배상", "채권자", "채무자", "소유자"],
        "exclude": ["형법", "형사", "범죄", "처벌"]
    },
    "criminal_law": {
        "primary": ["형법", "범죄", "처벌", "형량", "구성요건", "고의", "과실"],
        "secondary": ["사기", "절도", "강도", "살인", "상해", "폭행", "협박"],
        "exclude": ["민법", "계약", "손해배상"]
    },
    "family_law": {
        "primary": ["이혼", "상속", "양육권", "친권", "위자료", "재산분할", "유언"],
        "secondary": ["협의이혼", "재판이혼", "상속인", "상속세", "유산", "양육비"],
        "exclude": ["회사", "상법", "주식"]
    },
    "commercial_law": {
        "primary": ["상법", "회사", "주식", "이사", "주주", "회사설립", "합병"],
        "secondary": ["주식회사", "유한회사", "합명회사", "합자회사", "자본금", "정관"],
        "exclude": ["이혼", "상속", "가족"]
    },
    "labor_law": {
        "primary": ["노동법", "근로", "임금", "해고", "근로시간", "휴게시간", "연차"],
        "secondary": ["근로계약서", "임금체불", "부당해고", "노동위원회", "최저임금"],
        "exclude": ["이혼", "상속", "범죄"]
    },
    "real_estate": {
        "primary": ["부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"],
        "secondary": ["부동산등기법", "매매계약서", "임대차계약서", "등기부등본"],
        "exclude": ["이혼", "상속", "범죄"]
    },
    "general": {
        "primary": ["법률", "법령", "조문", "법원", "판례", "소송"],
        "secondary": ["법적", "법률적", "법적근거", "법적효력"],
        "exclude": []
    }
}

# 키워드 가중치 정의
KEYWORD_WEIGHTS = {
    "primary": 3.0,
    "secondary": 1.0,
    "exclude": -2.0
}

# 도메인 우선순위 정의
DOMAIN_PRIORITY = {
    "civil_law": 1.2,
    "criminal_law": 1.1,
    "family_law": 1.0,
    "commercial_law": 1.0,
    "labor_law": 1.0,
    "real_estate": 1.0,
    "general": 0.8
}

# Phase 1: 대화 맥락 강화 모듈
from .integrated_session_manager import IntegratedSessionManager
from .multi_turn_handler import MultiTurnQuestionHandler
from .context_compressor import ContextCompressor

# Phase 2: 개인화 및 지능형 분석 모듈
from .user_profile_manager import UserProfileManager
from .emotion_intent_analyzer import EmotionIntentAnalyzer
from .conversation_flow_tracker import ConversationFlowTracker

# Phase 3: 장기 기억 및 품질 모니터링 모듈
from .contextual_memory_manager import ContextualMemoryManager
from .conversation_quality_monitor import ConversationQualityMonitor

# 자연스러운 답변 개선 모듈
from .conversation_connector import ConversationConnector
from .emotional_tone_adjuster import EmotionalToneAdjuster
from .personalized_style_learner import PersonalizedStyleLearner
from .realtime_feedback_system import RealtimeFeedbackSystem
from .naturalness_evaluator import NaturalnessEvaluator

# 성능 최적화 모듈
from .cache_manager import get_cache_manager, cached
from .optimized_search_engine import OptimizedSearchEngine

# 법률 제한 시스템 모듈 (ML 통합 최신 버전)
from .ml_integrated_validation_system import MLIntegratedValidationSystem
from .improved_legal_restriction_system import ImprovedLegalRestrictionSystem, ImprovedRestrictionResult
from .intent_based_processor import IntentBasedProcessor, ProcessingResult
from .content_filter_engine import ContentFilterEngine, FilterResult
from .response_validation_system import ResponseValidationSystem, ValidationResult, ValidationStatus, ValidationLevel
from .safe_response_generator import SafeResponseGenerator, SafeResponse
from .legal_compliance_monitor import LegalComplianceMonitor, ComplianceStatus
from .user_education_system import UserEducationSystem, WarningMessage
from .multi_stage_validation_system import MultiStageValidationSystem, MultiStageValidationResult

logger = get_logger(__name__)


class EnhancedChatService:
    """개선된 채팅 서비스 클래스"""
    
    def __init__(self, config: Config):
        """채팅 서비스 초기화"""
        # Google Cloud 관련 경고 방지
        self._setup_google_cloud_warnings()
        
        self.config = config
        self.logger = get_logger(__name__)
        
        # LangGraph 사용 여부 확인 (비활성화)
        self.use_langgraph = False
        
        # 메모리 관리 시스템 초기화
        self._initialize_memory_management()
        
        # 사용자 설정 관리자 초기화
        self.user_preferences = preference_manager
        
        # 답변 완성도 검증기 초기화
        self.completion_validator = completion_validator
        
        # 예시 데이터베이스 초기화
        self.example_database = example_database
        self.dynamic_generator = dynamic_generator
        
        # 강화된 완성 시스템 초기화
        self.enhanced_completion_system = enhanced_completion_system
        
        # 핵심 컴포넌트 초기화
        self._initialize_core_components()
        
        # 법률 제한 시스템 초기화
        self._initialize_legal_restriction_systems()
        
        # 고급 검색 엔진 초기화
        self._initialize_advanced_search_engines()
        
        # 현행법령 전용 검색 엔진 초기화
        self._initialize_current_law_search_engine()
        
        # 통합 서비스 초기화 (현행법령 검색 엔진 초기화 후)
        self._initialize_unified_services()
        
        # Phase 시스템 초기화
        self._initialize_phase_systems()
        
        # 자연스러운 대화 개선 시스템 초기화
        self._initialize_natural_conversation_systems()
        
        # 성능 최적화 시스템 초기화
        self._initialize_performance_systems()
        
        # 품질 개선 시스템 초기화
        self._initialize_quality_enhancement_systems()
        
        # 향상된 조문 검색 시스템 초기화
        self._initialize_enhanced_law_search()
        
        self.logger.info("EnhancedChatService 초기화 완료")
    
    def _setup_google_cloud_warnings(self):
        """Google Cloud 관련 경고 방지"""
        os.environ['GRPC_DNS_RESOLVER'] = 'native'
        os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
        os.environ['GOOGLE_CLOUD_PROJECT'] = ''
        os.environ['GCLOUD_PROJECT'] = ''
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
        os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'
        os.environ['GRPC_VERBOSITY'] = 'ERROR'
        os.environ['GRPC_TRACE'] = ''
        
        # gRPC 로깅 레벨 조정
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
            # 메모리 관리자 초기화
            self.memory_manager = get_memory_manager(max_memory_mb=1024)
            
            # WeakRef 등록소 초기화
            self.weakref_registry = get_weakref_registry()
            
            # 실시간 메모리 모니터 초기화
            self.memory_monitor = get_memory_monitor()
            
            # 메모리 알림 콜백 등록
            self.memory_manager.add_alert_callback(self._on_memory_alert)
            
            # 컴포넌트 추적을 위한 WeakRef 등록 함수
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
            """컴포넌트를 WeakRef로 추적"""
            if self.weakref_registry:
                return self.weakref_registry.register_object(obj, name)
            return name
        return track_component
    
    def _on_memory_alert(self, alert):
        """메모리 알림 처리"""
        self.logger.warning(f"메모리 알림 [{alert.severity}]: {alert.message}")
        
        # 심각한 메모리 부족 시 자동 정리
        if alert.severity in ['high', 'critical']:
            self.logger.info("자동 메모리 정리 실행")
            cleanup_result = self.perform_memory_cleanup()
            self.logger.info(f"자동 정리 완료: {cleanup_result.get('memory_freed_mb', 0):.1f}MB 해제")
    
    def perform_memory_cleanup(self):
        """메모리 정리 수행"""
        try:
            import gc
            import psutil
            import os
            
            # 현재 메모리 사용량 측정
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 가비지 컬렉션 실행
            collected = gc.collect()
            
            # 컴포넌트별 메모리 정리
            cleanup_count = 0
            
            # 모델 매니저 메모리 정리
            if hasattr(self, 'model_manager') and self.model_manager:
                try:
                    if hasattr(self.model_manager, 'clear_cache'):
                        self.model_manager.clear_cache()
                        cleanup_count += 1
                except Exception as e:
                    self.logger.debug(f"Model manager cleanup failed: {e}")
            
            # 벡터 스토어 메모리 정리
            if hasattr(self, 'vector_store') and self.vector_store:
                try:
                    if hasattr(self.vector_store, 'clear_cache'):
                        self.vector_store.clear_cache()
                        cleanup_count += 1
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
            
            # 메모리 사용량 재측정
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            self.logger.info(f"메모리 정리 완료: {memory_freed:.1f}MB 해제, {collected}개 객체 수집, {cleanup_count}개 컴포넌트 정리")
            
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
            # 데이터베이스 관리자
            self.db_manager = DatabaseManager("data/lawfirm.db")
            self._track_component(self.db_manager, "db_manager")
            
            # 벡터 스토어
            self.vector_store = LegalVectorStore()
            self._track_component(self.vector_store, "vector_store")
            # 벡터 인덱스 로드
            try:
                self.vector_store.load_index()
                self.logger.info("벡터 인덱스 로드 성공")
            except Exception as e:
                self.logger.warning(f"벡터 인덱스 로드 실패: {e}")
            
            # 모델 관리자
            from .optimized_model_manager import OptimizedModelManager
            self.model_manager = OptimizedModelManager()
            self._track_component(self.model_manager, "model_manager")
            
            # RAG 서비스 (MLEnhancedRAGService는 제거하고 UnifiedRAGService만 사용)
            # self.rag_service = MLEnhancedRAGService(...)
            
            # 하이브리드 검색 엔진
            self.hybrid_search_engine = HybridSearchEngine()
            self._track_component(self.hybrid_search_engine, "hybrid_search_engine")
            
            # 질문 분류기
            self.question_classifier = QuestionClassifier()
            self._track_component(self.question_classifier, "question_classifier")
            
            # 개선된 답변 생성기
            self.improved_answer_generator = ImprovedAnswerGenerator()
            self._track_component(self.improved_answer_generator, "improved_answer_generator")
            
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
    
    def _initialize_unified_services(self):
        """통합 서비스 초기화"""
        try:
            # 벡터 스토어가 없는 경우 다시 초기화
            if not self.vector_store:
                from ..data.vector_store import LegalVectorStore
                self.vector_store = LegalVectorStore()
                try:
                    self.vector_store.load_index()
                    self.logger.info("벡터 인덱스 재로드 성공")
                except Exception as e:
                    self.logger.warning(f"벡터 인덱스 재로드 실패: {e}")
            
            # 통합 검색 엔진
            from .unified_search_engine import UnifiedSearchEngine
            self.unified_search_engine = UnifiedSearchEngine(
                vector_store=self.vector_store,
                current_law_search_engine=self.current_law_search_engine
            )
            
            # 통합 RAG 서비스
            from .unified_rag_service import UnifiedRAGService
            self.unified_rag_service = UnifiedRAGService(
                model_manager=self.model_manager,
                search_engine=self.unified_search_engine,
                answer_generator=self.improved_answer_generator,
                question_classifier=self.question_classifier
            )
            
            self.logger.info("통합 서비스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"통합 서비스 초기화 실패: {e}")
            self.unified_search_engine = None
            self.unified_rag_service = None
    
    def _initialize_legal_restriction_systems(self):
        """법률 제한 시스템 초기화"""
        try:
            # ML 통합 검증 시스템
            self.ml_validation_system = MLIntegratedValidationSystem()
            
            # 개선된 법률 제한 시스템
            self.improved_legal_restriction_system = ImprovedLegalRestrictionSystem()
            
            # 의도 기반 프로세서
            self.intent_based_processor = IntentBasedProcessor()
            
            # 콘텐츠 필터 엔진
            self.content_filter_engine = ContentFilterEngine()
            
            # 응답 검증 시스템
            self.response_validation_system = ResponseValidationSystem()
            
            # 안전한 응답 생성기
            self.safe_response_generator = SafeResponseGenerator()
            
            # 법률 준수 모니터
            self.legal_compliance_monitor = LegalComplianceMonitor()
            
            # 사용자 교육 시스템
            self.user_education_system = UserEducationSystem()
            
            # 다단계 검증 시스템
            self.multi_stage_validation_system = MultiStageValidationSystem()
            
            self.logger.info("법률 제한 시스템 초기화 완료")
            
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
        """고급 검색 엔진 초기화"""
        try:
            # 정확한 검색 엔진
            from .exact_search_engine import ExactSearchEngine
            self.exact_search_engine = ExactSearchEngine()
            
            # 의미론적 검색 엔진
            from .semantic_search_engine import SemanticSearchEngine
            self.semantic_search_engine = SemanticSearchEngine()
            
            # 최적화된 검색 엔진
            self.optimized_search_engine = OptimizedSearchEngine(
                vector_store=self.vector_store,
                exact_search_engine=self.exact_search_engine,
                semantic_search_engine=self.semantic_search_engine
            )
            
            # 판례 검색 엔진
            from .precedent_search_engine import PrecedentSearchEngine
            self.precedent_search_engine = PrecedentSearchEngine()
            
            self.logger.info("고급 검색 엔진 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"고급 검색 엔진 초기화 실패: {e}")
            self.optimized_search_engine = None
            self.exact_search_engine = None
            self.semantic_search_engine = None
            self.precedent_search_engine = None
    
    def _initialize_current_law_search_engine(self):
        """현행법령 전용 검색 엔진 초기화"""
        try:
            from .current_law_search_engine import CurrentLawSearchEngine
            
            self.current_law_search_engine = CurrentLawSearchEngine(
                db_path="data/lawfirm.db",
                vector_store=self.vector_store
            )
            
            self.logger.info("현행법령 전용 검색 엔진 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"현행법령 전용 검색 엔진 초기화 실패: {e}")
            self.current_law_search_engine = None
    
    def _initialize_phase_systems(self):
        """Phase 시스템 초기화"""
        try:
            # Phase 1: 대화 맥락 강화
            self.integrated_session_manager = IntegratedSessionManager("data/conversations.db")
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.context_compressor = ContextCompressor(self.config)
            
            # Phase 2: 개인화 및 지능형 분석
            self.user_profile_manager = UserProfileManager()
            self.emotion_intent_analyzer = EmotionIntentAnalyzer()
            self.conversation_flow_tracker = ConversationFlowTracker(self.config)
            
            # Phase 3: 장기 기억 및 품질 모니터링
            self.contextual_memory_manager = ContextualMemoryManager()
            self.conversation_quality_monitor = ConversationQualityMonitor()
            
            self.logger.info("Phase 시스템 초기화 완료")
            
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
        """자연스러운 대화 개선 시스템 초기화"""
        try:
            # 대화 연결기
            self.conversation_connector = ConversationConnector()
            
            # 감정 톤 조절기
            self.emotional_tone_adjuster = EmotionalToneAdjuster()
            
            # 개인화된 스타일 학습기
            self.personalized_style_learner = PersonalizedStyleLearner(self.config)
            
            # 실시간 피드백 시스템
            self.realtime_feedback_system = RealtimeFeedbackSystem(self.config)
            
            # 자연스러움 평가기
            self.naturalness_evaluator = NaturalnessEvaluator()
            
            self.logger.info("자연스러운 대화 개선 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"자연스러운 대화 개선 시스템 초기화 실패: {e}")
            # 기본값으로 설정
            self.conversation_connector = None
            self.emotional_tone_adjuster = None
            self.personalized_style_learner = None
            self.realtime_feedback_system = None
            self.naturalness_evaluator = None
    
    def _initialize_performance_systems(self):
        """성능 최적화 시스템 초기화"""
        try:
            # 캐시 관리자
            self.cache_manager = get_cache_manager()
            
            # 성능 모니터
            from .performance_monitor import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor()
            
            # 메모리 최적화기 (모듈이 없으므로 비활성화)
            # from .memory_optimizer import MemoryOptimizer
            # self.memory_optimizer = MemoryOptimizer()
            self.memory_optimizer = None
            
            self.logger.info("성능 최적화 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"성능 최적화 시스템 초기화 실패: {e}")
            self.cache_manager = None
            self.performance_monitor = None
            self.memory_optimizer = None
    
    def _initialize_quality_enhancement_systems(self):
        """품질 개선 시스템 초기화"""
        try:
            # 답변 품질 향상기
            from .answer_quality_enhancer import AnswerQualityEnhancer
            self.answer_quality_enhancer = AnswerQualityEnhancer()
            
            # 답변 구조 향상기
            from .answer_structure_enhancer import AnswerStructureEnhancer
            self.answer_structure_enhancer = AnswerStructureEnhancer()
            
            # 신뢰도 계산기
            from .confidence_calculator import ConfidenceCalculator
            self.confidence_calculator = ConfidenceCalculator()
            
            # 통합 프롬프트 관리자
            from .unified_prompt_manager import UnifiedPromptManager
            self.unified_prompt_manager = UnifiedPromptManager()
            
            # 프롬프트 최적화기
            from .prompt_optimizer import PromptOptimizer
            self.prompt_optimizer = PromptOptimizer(self.unified_prompt_manager)
            
            self.logger.info("품질 개선 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"품질 개선 시스템 초기화 실패: {e}")
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
            # 법률 조문 질문 감지 및 처리
            if self._is_law_article_query(message):
                self.logger.info(f"법률 조문 질문 감지: {message}")
                return await self._handle_law_article_query(message, user_id, session_id)
            
            # 입력 검증 및 전처리
            validation_result = self._validate_and_preprocess_input(message)
            if not validation_result["valid"]:
                return self._create_error_response(
                    validation_result["error"], session_id, user_id, start_time
                )
            
            # 캐시 확인
            cache_key = self._generate_cache_key(message, user_id, context)
            cached_result = self.cache_manager.get(cache_key) if self.cache_manager else None
            if cached_result:
                cached_result["processing_time"] = time.time() - start_time
                cached_result["cached"] = True
                return cached_result
            
            # 질문 분석
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
            
            # 향상된 응답 생성
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
            
            # 답변 완성도 검증 및 개선 (강화된 시스템 적용)
            if response_result.get("response"):
                response_text = response_result["response"]
                if isinstance(response_text, str):
                    # 강화된 완성 시스템 적용
                    completion_result = self.enhanced_completion_system.force_complete_answer(
                        response_text, message, query_analysis.get("category", "일반")
                    )
                    
                    if completion_result.was_truncated:
                        self.logger.info(f"답변이 중간에 끊어짐 감지. 완성 방법: {completion_result.completion_method}")
                        response_result["response"] = completion_result.completed_answer
                        response_result["completion_improved"] = True
                        response_result["completion_method"] = completion_result.completion_method
                        response_result["completion_confidence"] = completion_result.confidence
                    
                    # 예시 추가 (사용자 설정에 따라)
                    if self.user_preferences.get_preference("example_preference"):
                        enhanced_response = self._add_examples_to_response(
                            response_result["response"], message, query_analysis
                        )
                        if enhanced_response != response_result["response"]:
                            response_result["response"] = enhanced_response
                            response_result["examples_added"] = True
            
            # 사용자 설정에 따른 면책 조항 처리
            final_response_text = self.user_preferences.add_disclaimer_to_response(
                response_result["response"], message
            )
            response_result["response"] = final_response_text
            
            # 처리 시간 추가
            response_result["processing_time"] = time.time() - start_time
            response_result["session_id"] = session_id
            response_result["user_id"] = user_id
            
            # 캐시 저장
            if self.cache_manager:
                self.cache_manager.set(cache_key, response_result, ttl_seconds=3600)
            
            return response_result
            
        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류 발생: {e}")
            import traceback
            self.logger.error(f"상세 오류: {traceback.format_exc()}")
            return self._create_error_response(
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
    
    def _create_error_response(self, error_message: str, session_id: str, user_id: str, start_time: float) -> Dict[str, Any]:
        """에러 응답 생성"""
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
        """질문 분석 - 개선된 버전"""
        try:
            # 질문 분류
            if self.question_classifier:
                classification = self.question_classifier.classify_question(message)
                query_type = classification.question_type
                intent = "unknown"  # QuestionClassification에 intent 속성이 없음
                confidence = classification.confidence
            else:
                query_type = "general"
                intent = "unknown"
                confidence = 0.5
            
            # 키워드 추출 개선
            import re
            keywords = []
            
            # 법률 관련 키워드 추출
            law_patterns = [
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법)',
                r'(계약|손해배상|불법행위|소유권|물권|채권|채무)',
                r'(이혼|상속|양육권|친권|위자료|재산분할)',
                r'(회사|주식|이사|주주|회사설립|합병)',
                r'(근로|임금|해고|근로시간|휴게시간|연차)',
                r'(부동산|매매|임대차|등기|소유권이전|전세|월세)'
            ]
            
            for pattern in law_patterns:
                matches = re.findall(pattern, message)
                keywords.extend(matches)
            
            # 법률 조문 패턴 검색 (개선된 정규표현식)
            # 다양한 법률 조문 패턴 지원
            statute_patterns = [
                # 표준 형태: 민법 제750조
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)\s*제\s*(\d+)\s*조',
                # 공백 없는 형태: 민법제750조
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)제\s*(\d+)\s*조',
                # 공백 있는 형태: 민법 750조
                r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법|노동기준법|가족관계등록법)\s+(\d+)\s*조',
                # 단축 형태: 제750조
                r'제\s*(\d+)\s*조',
                # 숫자만: 750조
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
                        # 법률명과 조문번호가 모두 있는 경우
                        statute_law = groups[0]
                        statute_article = groups[1]
                    elif len(groups) == 1:
                        # 조문번호만 있는 경우
                        statute_article = groups[0]
                        # 법률명은 문맥에서 추론하거나 None으로 설정
                        statute_law = None
                    break
            
            return {
                "query_type": query_type,
                "intent": intent,
                "confidence": confidence,
                "context": context,
                "keywords": keywords,  # 추출된 키워드 추가
                "statute_match": statute_match.group(0) if statute_match else None,
                "statute_law": statute_law,
                "statute_article": statute_article,
                "timestamp": datetime.now(),
                "session_id": session_id,
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"질문 분석 중 오류: {e}")
            return {
                "query_type": "general",
                "intent": "unknown",
                "confidence": 0.5,
                "context": context,
                "keywords": [],
                "statute_match": None,
                "statute_law": None,
                "statute_article": None,
                "timestamp": datetime.now(),
                "session_id": session_id,
                "user_id": user_id,
                "error": str(e)
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
                    from .conversation_manager import ConversationTurn
                    from datetime import datetime
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
                    self.logger.debug(f"ConversationFlowTracker 메서드 오류: {e}")
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
        """향상된 답변 생성 - 참고 데이터 기반으로만 답변"""
        self.logger.info(f"_generate_enhanced_response called for: {message}")
        try:
            # 0순위: 특정 법령 조문 검색 (새로 추가)
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
                        # 조문 내용은 없지만 법령은 찾은 경우
                        return {
                            "response": f"'{statute_law} 제{statute_article}조'에 대한 정보를 찾았지만, 해당 조문의 구체적인 내용을 찾을 수 없습니다.\n\n찾은 법령 정보:\n- 법령명: {specific_result.law_name_korean}\n- 소관부처: {specific_result.ministry_name}\n- 시행일: {specific_result.effective_date}\n\n더 구체적인 조문 내용이 필요하시면 국가법령정보센터(www.law.go.kr)에서 확인하시기 바랍니다.",
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
            
            # 1순위: 기본 RAG 서비스 (실제 AI 답변 우선)
            if self.unified_rag_service:
                try:
                    self.logger.info(f"Calling RAG service for query: {message}")
                    rag_response = await self.unified_rag_service.generate_response(
                        query=message,
                        context=query_analysis.get("context"),
                        max_length=300,
                        top_k=3,
                        use_cache=True
                    )
                    
                    # 참고 데이터가 있는지 확인
                    if rag_response and rag_response.response and self._has_meaningful_sources(rag_response.sources):
                        # 동적 신뢰도 계산
                        confidence = self._calculate_confidence(rag_response.sources, "good")
                        
                        return {
                            "response": rag_response.response,
                            "confidence": confidence,
                            "sources": rag_response.sources,
                            "query_analysis": query_analysis,
                            "generation_method": "simple_rag",
                            "session_id": session_id,
                            "user_id": user_id
                        }
                    else:
                        # 참고 데이터가 없으면 솔직하게 알려줌
                        return self._create_no_sources_response(message, query_analysis, session_id, user_id)
                        
                except Exception as e:
                    self.logger.debug(f"Simple RAG service failed: {e}")
            else:
                self.logger.warning("unified_rag_service is None, skipping RAG generation")
            
            # 참고 데이터가 없으면 억지로 답변하지 않음
            return self._create_no_sources_response(message, query_analysis, session_id, user_id)
            
        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            return self._create_error_response(message, query_analysis, session_id, user_id, str(e))
    
    def _has_meaningful_sources(self, sources: List[Dict[str, Any]]) -> bool:
        """의미있는 참고 데이터가 있는지 확인 - 완화된 버전"""
        if not sources:
            return False
        
        # 최소 관련도 임계값 설정 (완화)
        MIN_RELEVANCE_THRESHOLD = 0.3  # 0.4에서 0.3으로 하향
        MIN_CONTENT_LENGTH = 50  # 70에서 50으로 하향
        
        meaningful_sources = []
        for source in sources:
            relevance_score = source.get("similarity", source.get("score", 0.0))
            content = source.get("content", "")
            
            # 관련도가 높고 내용이 충분한 소스만 유효한 것으로 판단
            if relevance_score >= MIN_RELEVANCE_THRESHOLD and len(content.strip()) > MIN_CONTENT_LENGTH:
                meaningful_sources.append(source)
        
        # 최소 1개 이상의 의미있는 소스가 있으면 유효
        if len(meaningful_sources) >= 1:
            return True
        
        # 추가 검증: 실제 법률 관련 내용인지 확인 (확장된 키워드, 완화된 기준)
        if meaningful_sources:
            legal_keywords = ["법률", "조문", "판례", "법원", "법령", "규정", "조항", "법적", "법률적", "계약", "소송", "재판", "민법", "형법", "상법", "이혼", "부동산", "손해배상"]
            legal_content_count = 0
            
            for source in meaningful_sources:
                content = source.get("content", "").lower()
                if any(keyword in content for keyword in legal_keywords):
                    legal_content_count += 1
            
            # 법률 관련 내용이 1개 이상이면 유효 (기존 50%에서 완화)
            return legal_content_count >= 1
        
        return False
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]], response_quality: str = "good") -> float:
        """동적 신뢰도 계산 - 개선된 버전"""
        if not sources:
            return 0.0
        
        # 기본 신뢰도 (완화)
        base_confidence = 0.3  # 0.25에서 0.3으로 상향
        
        # 소스 품질에 따른 조정 (완화된 기준)
        avg_relevance = sum(source.get("similarity", source.get("score", 0.0)) for source in sources) / len(sources)
        
        # 관련도에 따른 보너스 (완화된 기준)
        if avg_relevance >= 0.7:
            relevance_bonus = 0.4  # 높은 관련도 (0.35에서 0.4로 상향)
        elif avg_relevance >= 0.5:
            relevance_bonus = 0.25  # 중간 관련도 (0.15에서 0.25로 상향)
        elif avg_relevance >= 0.3:
            relevance_bonus = 0.15  # 낮은 관련도 (0.05에서 0.15로 상향)
        else:
            relevance_bonus = 0.05  # 매우 낮은 관련도 (0.0에서 0.05로 상향)
        
        # 소스 개수에 따른 조정 (완화된 기준)
        if len(sources) >= 3:
            source_count_bonus = 0.15  # 많은 소스 (0.1에서 0.15로 상향)
        elif len(sources) >= 2:
            source_count_bonus = 0.1  # 적당한 소스 (0.05에서 0.1로 상향)
        else:
            source_count_bonus = 0.05  # 적은 소스 (0.0에서 0.05로 상향)
        
        # 응답 품질에 따른 조정
        quality_bonus = 0.15 if response_quality == "excellent" else 0.1 if response_quality == "good" else 0.05
        
        # 최종 신뢰도 계산
        final_confidence = base_confidence + relevance_bonus + source_count_bonus + quality_bonus
        
        # 0.0 ~ 1.0 범위로 제한
        return max(0.0, min(1.0, final_confidence))
    
    def _create_no_sources_response(self, message: str, query_analysis: Dict[str, Any], session_id: str, user_id: str) -> Dict[str, Any]:
        """참고 데이터가 없을 때의 응답 생성"""
        query_type = query_analysis.get("query_type", "general")
        
        # 질문 유형별 맞춤 메시지
        if query_type == "legal_advice":
            response = f"""죄송합니다. '{message}'에 대한 구체적인 법률 정보를 찾을 수 없습니다.

다음과 같이 도움을 드릴 수 있습니다:
• 더 구체적인 질문을 해주세요 (예: "민법 제750조 불법행위 손해배상")
• 관련 법률 조문이나 판례가 있다면 함께 언급해주세요
• 일반적인 법률 절차에 대해서는 안내해드릴 수 있습니다

구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
        
        elif query_type == "precedent":
            response = f"""죄송합니다. '{message}'와 관련된 판례를 찾을 수 없습니다.

다음과 같이 도움을 드릴 수 있습니다:
• 사건번호나 법원명을 포함해서 질문해주세요
• 더 구체적인 키워드로 검색해주세요
• 관련 법률 조문을 먼저 확인해보세요

판례 검색이 어려우시면 법원 도서관이나 법률 데이터베이스를 이용해보시기 바랍니다."""
        
        elif query_type == "law_inquiry":
            response = f"""죄송합니다. '{message}'에 대한 법률 조문을 찾을 수 없습니다.

다음과 같이 도움을 드릴 수 있습니다:
• 정확한 법률명과 조문번호를 포함해주세요 (예: "민법 제750조")
• 법률의 정식 명칭을 사용해주세요
• 관련 키워드를 더 구체적으로 해주세요

법령 정보는 국가법령정보센터(www.law.go.kr)에서 확인하실 수 있습니다."""
        
        else:
            response = f"""죄송합니다. '{message}'에 대한 관련 정보를 찾을 수 없습니다.

다음과 같이 도움을 드릴 수 있습니다:
• 질문을 더 구체적으로 작성해주세요
• 관련 법률 조문이나 판례를 포함해주세요
• 키워드를 더 명확하게 해주세요

일반적인 법률 상식이나 절차에 대해서는 안내해드릴 수 있습니다."""

        # 검색 제안 생성
        suggestions = self._generate_search_suggestions(message, query_analysis)
        suggestion_text = suggestions[0] if suggestions else "질문을 더 구체적으로 작성해주세요"
        
        return {
            "response": response,
            "confidence": 0.1,  # 0.0에서 0.1로 상향 - 솔직한 응답에 대한 기본 신뢰도
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
        """검색 제안 생성 - 개선된 버전"""
        suggestions = []
        
        # 질문에서 직접 키워드 추출 (개선된 로직)
        import re
        
        # 법률 관련 키워드 패턴 (확장된 버전)
        law_patterns = [
            r'(민법|형법|상법|노동법|가족법|행정법|헌법|민사소송법|형사소송법)',
            r'(계약|손해배상|불법행위|소유권|물권|채권|채무)',
            r'(이혼|상속|양육권|친권|위자료|재산분할)',
            r'(회사|주식|이사|주주|회사설립|합병)',
            r'(근로|임금|해고|근로시간|휴게시간|연차)',
            r'(부동산|매매|임대차|등기|소유권이전|전세|월세)',
            r'(법률|법령|조문|판례|법원|규정|조항|법적|법률적)'  # 추가
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
        
        # 키워드 우선순위 정렬 (법률명 > 구체적 용어 > 일반 용어)
        priority_keywords = []
        specific_keywords = []
        general_keywords = []
        
        for keyword in extracted_keywords:
            if keyword in ["민법", "형법", "상법", "노동법", "가족법", "행정법", "헌법", "민사소송법", "형사소송법"]:
                priority_keywords.append(keyword)
            elif keyword in ["계약", "손해배상", "불법행위", "소유권", "물권", "채권", "채무", "이혼", "상속", "양육권", "친권", "위자료", "재산분할", "회사", "주식", "이사", "주주", "회사설립", "합병", "근로", "임금", "해고", "부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"]:
                specific_keywords.append(keyword)
            else:
                general_keywords.append(keyword)
        
        # 우선순위 키워드 먼저, 그 다음 구체적 키워드, 마지막 일반 키워드
        extracted_keywords = priority_keywords + specific_keywords + general_keywords
        
        # 질문 분류기에서 키워드 가져오기 (fallback) - 우선순위 정렬 적용
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
                    elif keyword in ["계약", "손해배상", "불법행위", "소유권", "물권", "채권", "채무", "이혼", "상속", "양육권", "친권", "위자료", "재산분할", "회사", "주식", "이사", "주주", "회사설립", "합병", "근로", "임금", "해고", "부동산", "매매", "임대차", "등기", "소유권이전", "전세", "월세"]:
                        specific_keywords.append(keyword)
                    else:
                        general_keywords.append(keyword)
                
                # 우선순위 키워드 먼저, 그 다음 구체적 키워드, 마지막 일반 키워드
                extracted_keywords = priority_keywords + specific_keywords + general_keywords
        
        # 추출된 키워드로 제안 생성 (이미 우선순위로 정렬됨)
        if extracted_keywords:
            main_keyword = extracted_keywords[0]
            suggestions.append(f"'{main_keyword}' 관련 법률 조문을 검색해보세요")
            suggestions.append(f"'{main_keyword}' 판례를 찾아보세요")
            if len(extracted_keywords) > 1:
                suggestions.append(f"'{extracted_keywords[1]}'와 함께 검색해보세요")
        
        # 질문 유형별 제안
        query_type = query_analysis.get("query_type", "general")
        if query_type == "legal_advice":
            suggestions.extend([
                "구체적인 상황을 설명해주세요",
                "관련 법률 조문을 함께 언급해주세요"
            ])
        elif query_type == "precedent":
            suggestions.extend([
                "사건번호나 법원명을 포함해주세요",
                "더 구체적인 키워드로 검색해보세요"
            ])
        elif query_type == "law_inquiry":
            suggestions.extend([
                "정확한 법률명과 조문번호를 포함해주세요",
                "법률의 정식 명칭을 사용해주세요"
            ])
        
        # 일반적인 제안 (키워드가 없는 경우)
        if not suggestions:
            suggestions.extend([
                "질문을 더 구체적으로 작성해주세요",
                "관련 법률 조문이나 판례를 포함해주세요",
                "키워드를 더 명확하게 해주세요"
            ])
        
        return suggestions[:3]  # 최대 3개 제안
    
    def _generate_improved_template_response(self, message: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """완전히 자연스러운 답변 생성 - 최고 수준 프롬프트 엔지니어링 적용"""
        self.logger.info(f"_generate_improved_template_response called for: {message}")
        
        # 템플릿 기반 답변을 완전히 제거하고 자연스러운 답변만 생성
        try:
            # Gemini 클라이언트를 사용하여 직접 답변 생성
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            # 완전히 새로운 프롬프트 구조 적용
            prompt = f"""사용자: {message}

위 질문에 대해 마치 친한 변호사와 대화하는 것처럼 자연스럽게 답변하세요.

예시:
사용자: "민법 제750조가 뭐야?"
변호사: "민법 제750조는 불법행위에 관한 조항이에요. 쉽게 말해서 누군가가 고의나 실수로 다른 사람에게 피해를 주면 그 피해를 배상해야 한다는 내용입니다. 불법행위가 성립하려면 네 가지 조건이 모두 맞아야 해요: 첫째, 가해자가 고의나 과실이 있어야 하고, 둘째, 위법한 행위여야 하며, 셋째, 실제로 손해가 발생해야 하고, 넷째, 그 행위와 손해 사이에 인과관계가 있어야 합니다. 이 모든 조건이 충족되면 손해배상 책임이 생겨요."

사용자: "계약서 작성 방법을 알려주세요"
변호사: "계약서는 당사자 간의 약속을 명확히 하는 중요한 문서예요. 작성할 때는 몇 가지 핵심 사항을 꼼꼼히 챙기셔야 합니다. 먼저 계약 당사자들의 정확한 정보(이름, 주소, 연락처)를 기재하고, 계약의 목적과 내용을 구체적으로 명시해야 해요. 예를 들어 부동산 매매라면 매물 정보와 매매 대금, 지급 시기 등을 상세히 적어야 합니다. 또한 계약 기간, 대금 지급 방법, 위약 시 손해배상 조항, 분쟁 해결 방법 등도 포함하는 것이 좋아요. 중요한 것은 나중에 해석의 여지가 없도록 명확하고 구체적으로 작성하는 것입니다."

답변:"""
            
            # 프롬프트 체인 방식 사용 시도
            try:
                from .prompt_chain import prompt_chain_processor
                response = prompt_chain_processor.process_with_chain(message, "")
                
                return {
                    "response": response,
                    "confidence": 0.85,
                    "sources": [],
                    "generation_method": "prompt_chain"
                }
                
            except Exception as e:
                self.logger.error(f"Prompt chain generation failed: {e}")
                # 폴백: 대안 모델 클라이언트 사용
                try:
                    from .alternative_model_client import alternative_model_client
                    response = alternative_model_client.generate_with_fallback(prompt)
                    
                    return {
                        "response": response,
                        "confidence": 0.8,
                        "sources": [],
                        "generation_method": "alternative_model"
                    }
                    
                except Exception as e2:
                    self.logger.error(f"Alternative model generation failed: {e2}")
                    # 최종 폴백: 기본 Gemini 클라이언트 사용
                    gemini_response = gemini_client.generate(prompt)
                    response = gemini_response.response
            
            return {
                "response": response,
                "confidence": 0.8,
                "sources": [],
                "generation_method": "natural_response"
            }
            
        except Exception as e:
            self.logger.error(f"Natural response generation failed: {e}")
            # 폴백: 간단한 답변
            return {
                "response": f"'{message}'에 대한 답변을 준비 중입니다. 잠시만 기다려주세요.",
                "confidence": 0.5,
                "sources": [],
                "generation_method": "fallback"
            }
    
    def _classify_domain_by_weight(self, message: str) -> str:
        """가중치 기반 도메인 분류"""
        message_lower = message.lower()
        domain_scores = {}
        
        for domain, keywords_config in LEGAL_DOMAIN_KEYWORDS.items():
            score = 0
            
            # 주요 키워드 매칭
            for keyword in keywords_config.get("primary", []):
                if keyword in message_lower:
                    score += KEYWORD_WEIGHTS["primary"]
            
            # 보조 키워드 매칭
            for keyword in keywords_config.get("secondary", []):
                if keyword in message_lower:
                    score += KEYWORD_WEIGHTS["secondary"]
            
            # 제외 키워드 매칭
            for keyword in keywords_config.get("exclude", []):
                if keyword in message_lower:
                    score += KEYWORD_WEIGHTS["exclude"]
            
            # 우선순위 적용
            priority = DOMAIN_PRIORITY.get(domain, 1)
            domain_scores[domain] = max(0, score) * priority
        
        # 가장 높은 점수의 도메인 반환
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return "general"
    
    def _extract_law_article_from_query(self, message: str) -> Dict[str, Any]:
        """질문에서 법령 조문 정보 추출"""
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

    def _extract_intent_from_message(self, message: str, domain: str) -> str:
        """메시지에서 의도 추출"""
        message_lower = message.lower()
        
        if domain not in LEGAL_DOMAIN_KEYWORDS:
            return "unknown"
        
        domain_config = LEGAL_DOMAIN_KEYWORDS[domain]
        intent_keywords = domain_config.get("intent_keywords", {})
        
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in message_lower:
                    score += 1
            intent_scores[intent] = score
        
        # 가장 높은 점수의 의도 반환
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        return "default"
    
    def _generate_template_by_intent(self, message: str, domain: str, intent: str) -> Dict[str, Any]:
        """의도에 따른 자연스러운 답변 생성 - 템플릿 완전 제거"""
        try:
            # 템플릿 대신 자연스러운 답변 생성
            from .gemini_client import GeminiClient
            gemini_client = GeminiClient()
            
            prompt = f"""사용자의 질문에 대해 자연스럽고 친근한 답변을 해주세요:

질문: {message}
도메인: {domain}
의도: {intent}

답변 방식:
- 마치 친한 변호사와 대화하는 것처럼 자연스럽게
- 섹션 제목이나 플레이스홀더 사용 금지
- 구체적이고 실용적인 정보 제공
- 하나의 연속된 답변으로 작성

답변:"""
            
            response = gemini_client.generate(prompt)
            
            return {
                "response": response.response,
                "confidence": 0.8,
                "sources": [],
                "generation_method": "natural_intent_based"
            }
            
        except Exception as e:
            self.logger.error(f"Natural intent-based generation failed: {e}")
            return {
                "response": f"'{message}'에 대한 답변을 준비 중입니다. 잠시만 기다려주세요.",
                "confidence": 0.5,
                "sources": [],
                "generation_method": "fallback"
            }
    
    # === 템플릿 메서드들 제거됨 - 자연스러운 답변만 생성 ===
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """분석 결과를 바탕으로 권장사항 생성"""
        recommendations = []
        
        try:
            # 메모리 사용량 분석
            memory_usage = analysis_result.get('memory_usage', {})
            if memory_usage.get('usage_percent', 0) > 80:
                recommendations.append({
                    'type': 'warning',
                    'title': '메모리 사용량 높음',
                    'description': f"현재 메모리 사용률: {memory_usage.get('usage_percent', 0):.1f}%",
                    'action': '메모리 정리 실행',
                    'command': 'service.perform_memory_cleanup()'
                })
            
            # 응답 시간 분석
            response_time = analysis_result.get('response_time', 0)
            if response_time > 10:
                recommendations.append({
                    'type': 'performance',
                    'title': '응답 시간 개선 필요',
                    'description': f"평균 응답 시간: {response_time:.2f}초",
                    'action': '성능 최적화 실행',
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
            
            # 기본 권장사항
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
                'title': '권장사항 생성 실패',
                'description': str(e),
                'action': '시스템 로그를 확인하세요',
                'command': None
            })
        
        return recommendations
    
    def _add_examples_to_response(self, response: str, question: str, query_analysis: Dict[str, Any]) -> str:
        """답변에 예시 추가"""
        try:
            category = query_analysis.get("category", "일반")
            question_type = query_analysis.get("question_type", "일반")
            
            # 카테고리별 예시 키 매핑
            example_key_mapping = {
                "법률조문": "민법_750조",
                "계약서": "계약서_작성", 
                "부동산": "부동산_매매",
                "가족법": "이혼_소송",
                "민사법": "손해배상_청구"
            }
            
            example_key = example_key_mapping.get(category, category)
            
            # 데이터베이스에서 예시 가져오기
            examples = self.example_database.get_examples(example_key, 1)
            
            if examples:
                example = examples[0]
                example_text = f"\n\n예를 들어, {example.situation}의 경우 {example.analysis}가 됩니다."
                
                # 실무 팁이 있으면 추가
                if example.practical_tips:
                    tips_text = "실무 팁: " + ", ".join(example.practical_tips[:3])
                    example_text += f" {tips_text}."
                
                return "" + example_text
            
            else:
                # 동적 예시 생성
                dynamic_example = self.dynamic_generator.generate_example(
                    category, question, question_type
                )
                if dynamic_example and len(dynamic_example) > 50:
                    return "" + f"\n\n{dynamic_example}"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"예시 추가 실패: {e}")
            return ""
    
    def _add_fallback_ending(self, response: str) -> str:
        """폴백 마무리 추가"""
        try:
            # 불완전한 문장 패턴 감지
            incomplete_patterns = [
                r'드$', r'그리고$', r'또한$', r'마지막으로$', r'결론적으로$',
                r'예를 들어$', r'구체적으로$', r'특히$', r'또한$',
                r'[가-힣]+드$', r'[가-힣]+고$', r'[가-힣]+며$'
            ]
            
            import re
            for pattern in incomplete_patterns:
                if re.search(pattern, response.strip()):
                    # 불완전한 부분을 자연스럽게 마무리
                    if response.strip().endswith('드'):
                        return f"{response.strip()} 이렇게 진행하시면 됩니다."
                    elif response.strip().endswith(('그리고', '또한')):
                        return f"{response.strip()} 더 궁금한 점이 있으시면 언제든지 물어보세요."
                    else:
                        return f"{response.strip()} 이렇게 하시면 됩니다."
            
            # 문장이 적절히 끝나지 않은 경우
            if not response.strip().endswith(('.', '!', '?', '니다.', '습니다.', '요.')):
                return f"{response.strip()} 이렇게 진행하시면 됩니다."
            
            return ""
            
        except Exception as e:
            self.logger.error(f"폴백 마무리 추가 실패: {e}")
            return ""
    
    # 새로운 조문 검색 및 답변 최적화 기능들
    
    def _initialize_enhanced_law_search(self):
        """향상된 조문 검색 시스템 초기화"""
        try:
            from .enhanced_law_search_engine import EnhancedLawSearchEngine
            from .law_context_search_engine import LawContextSearchEngine
            from .integrated_law_search_service import IntegratedLawSearchService
            from .adaptive_response_manager import AdaptiveResponseManager
            from .progressive_response_system import ProgressiveResponseSystem
            
            # 통합 조문 검색 서비스 초기화
            self.integrated_law_search = IntegratedLawSearchService(self.config)
            
            # 적응형 답변 관리자 초기화
            self.adaptive_response_manager = AdaptiveResponseManager()
            
            # 단계별 답변 시스템 초기화
            self.progressive_response_system = ProgressiveResponseSystem()
            
            # 법률 조문 질문 패턴
            self.law_query_patterns = [
                r'(\w+법)\s*제\s*(\d+)조',
                r'제\s*(\d+)조',
                r'(\w+법)\s*(\d+)조',
                r'(\w+법)\s*제\s*(\d+)조\s*제\s*(\d+)항'
            ]
            
            self.logger.info("향상된 조문 검색 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"향상된 조문 검색 시스템 초기화 실패: {e}")
            self.integrated_law_search = None
            self.adaptive_response_manager = None
            self.progressive_response_system = None
    
    def _is_law_article_query(self, query: str) -> bool:
        """법률 조문 질문인지 확인"""
        try:
            import re
            for pattern in self.law_query_patterns:
                if re.search(pattern, query):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"법률 조문 질문 확인 실패: {e}")
            return False
    
    async def _handle_law_article_query(self, message: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """법률 조문 질문 처리"""
        start_time = time.time()
        
        try:
            if not self.integrated_law_search:
                return await self._fallback_response(message)
            
            # 통합 조문 검색 실행
            search_result = await self.integrated_law_search.search_law_article(message)
            
            # 사용자 컨텍스트 분석
            user_context = await self._analyze_user_context(user_id, session_id)
            
            # 적응형 답변 길이 조정
            if self.adaptive_response_manager:
                optimized_response = self.adaptive_response_manager.adapt_response_length(
                    search_result.response, user_context
                )
            else:
                optimized_response = search_result.response
            
            # 단계별 답변 생성
            if self.progressive_response_system:
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
            self.logger.error(f"법률 조문 질문 처리 실패: {e}")
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
            
            # 기본 길이 설정
            base_lengths = {
                'mobile': 600,
                'desktop': 1200,
                'tablet': 900
            }
            
            base_length = base_lengths.get(device_type, 1200)
            
            # 전문성 수준에 따른 조정
            expertise_multipliers = {
                'beginner': 0.8,
                'intermediate': 1.0,
                'expert': 1.2,
                'professional': 1.3
            }
            
            multiplier = expertise_multipliers.get(expertise_level, 1.0)
            
            # 상세 수준에 따른 조정
            detail_multipliers = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.3
            }
            
            detail_multiplier = detail_multipliers.get(detail_level, 1.0)
            
            return int(base_length * multiplier * detail_multiplier)
            
        except Exception as e:
            self.logger.error(f"선호 길이 계산 실패: {e}")
            return 1000
    
    async def get_expanded_response(self, base_response: str, option_type: str, user_id: str = None) -> str:
        """확장된 답변 제공"""
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

            # 카테고리별 예시 키 매핑
            example_key_mapping = {
                "법률조문": "민법_750조",
                "계약서": "계약서_작성", 
                "부동산": "부동산_매매",
                "가족법": "이혼_소송",
                "민사법": "손해배상_청구"
            }
            
            example_key = example_key_mapping.get(category, category)
            
            # 데이터베이스에서 예시 가져오기
            examples = self.example_database.get_examples(example_key, 1)
            
            if examples:
                example = examples[0]
                example_text = f"\n\n예를 들어, {example.situation}의 경우 {example.analysis}가 됩니다."
                
                # 실무 팁이 있으면 추가
                if example.practical_tips:
                    tips_text = "실무 팁: " + ", ".join(example.practical_tips[:3])
                    example_text += f" {tips_text}."
                
                return "" + example_text
            
            else:
                # 동적 예시 생성
                dynamic_example = self.dynamic_generator.generate_example(
                    category, question, question_type
                )
                if dynamic_example and len(dynamic_example) > 50:
                    return "" + f"\n\n{dynamic_example}"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"예시 추가 실패: {e}")
            return ""
    
    def _add_fallback_ending(self, response: str) -> str:
        """폴백 마무리 추가"""
        try:
            # 불완전한 문장 패턴 감지
            incomplete_patterns = [
                r'드$', r'그리고$', r'또한$', r'마지막으로$', r'결론적으로$',
                r'예를 들어$', r'구체적으로$', r'특히$', r'또한$',
                r'[가-힣]+드$', r'[가-힣]+고$', r'[가-힣]+며$'
            ]
            
            import re
            for pattern in incomplete_patterns:
                if re.search(pattern, response.strip()):
                    # 불완전한 부분을 자연스럽게 마무리
                    if response.strip().endswith('드'):
                        return f"{response.strip()} 이렇게 진행하시면 됩니다."
                    elif response.strip().endswith(('그리고', '또한')):
                        return f"{response.strip()} 더 궁금한 점이 있으시면 언제든지 물어보세요."
                    else:
                        return f"{response.strip()} 이렇게 하시면 됩니다."
            
            # 문장이 적절히 끝나지 않은 경우
            if not response.strip().endswith(('.', '!', '?', '니다.', '습니다.', '요.')):
                return f"{response.strip()} 이렇게 진행하시면 됩니다."
            
            return ""
            
        except Exception as e:
            self.logger.error(f"폴백 마무리 추가 실패: {e}")
            return ""
    
    # 새로운 조문 검색 및 답변 최적화 기능들
    
    def _initialize_enhanced_law_search(self):
        """향상된 조문 검색 시스템 초기화"""
        try:
            from .enhanced_law_search_engine import EnhancedLawSearchEngine
            from .law_context_search_engine import LawContextSearchEngine
            from .integrated_law_search_service import IntegratedLawSearchService
            from .adaptive_response_manager import AdaptiveResponseManager
            from .progressive_response_system import ProgressiveResponseSystem
            
            # 통합 조문 검색 서비스 초기화
            self.integrated_law_search = IntegratedLawSearchService(self.config)
            
            # 적응형 답변 관리자 초기화
            self.adaptive_response_manager = AdaptiveResponseManager()
            
            # 단계별 답변 시스템 초기화
            self.progressive_response_system = ProgressiveResponseSystem()
            
            # 법률 조문 질문 패턴
            self.law_query_patterns = [
                r'(\w+법)\s*제\s*(\d+)조',
                r'제\s*(\d+)조',
                r'(\w+법)\s*(\d+)조',
                r'(\w+법)\s*제\s*(\d+)조\s*제\s*(\d+)항'
            ]
            
            self.logger.info("향상된 조문 검색 시스템 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"향상된 조문 검색 시스템 초기화 실패: {e}")
            self.integrated_law_search = None
            self.adaptive_response_manager = None
            self.progressive_response_system = None
    
    def _is_law_article_query(self, query: str) -> bool:
        """법률 조문 질문인지 확인"""
        try:
            import re
            for pattern in self.law_query_patterns:
                if re.search(pattern, query):
                    return True
            return False
        except Exception as e:
            self.logger.error(f"법률 조문 질문 확인 실패: {e}")
            return False
    
    async def _handle_law_article_query(self, message: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """법률 조문 질문 처리"""
        start_time = time.time()
        
        try:
            if not self.integrated_law_search:
                return await self._fallback_response(message)
            
            # 통합 조문 검색 실행
            search_result = await self.integrated_law_search.search_law_article(message)
            
            # 사용자 컨텍스트 분석
            user_context = await self._analyze_user_context(user_id, session_id)
            
            # 적응형 답변 길이 조정
            if self.adaptive_response_manager:
                optimized_response = self.adaptive_response_manager.adapt_response_length(
                    search_result.response, user_context
                )
            else:
                optimized_response = search_result.response
            
            # 단계별 답변 생성
            if self.progressive_response_system:
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
            self.logger.error(f"법률 조문 질문 처리 실패: {e}")
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
            
            # 기본 길이 설정
            base_lengths = {
                'mobile': 600,
                'desktop': 1200,
                'tablet': 900
            }
            
            base_length = base_lengths.get(device_type, 1200)
            
            # 전문성 수준에 따른 조정
            expertise_multipliers = {
                'beginner': 0.8,
                'intermediate': 1.0,
                'expert': 1.2,
                'professional': 1.3
            }
            
            multiplier = expertise_multipliers.get(expertise_level, 1.0)
            
            # 상세 수준에 따른 조정
            detail_multipliers = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.3
            }
            
            detail_multiplier = detail_multipliers.get(detail_level, 1.0)
            
            return int(base_length * multiplier * detail_multiplier)
            
        except Exception as e:
            self.logger.error(f"선호 길이 계산 실패: {e}")
            return 1000
    
    async def get_expanded_response(self, base_response: str, option_type: str, user_id: str = None) -> str:
        """확장된 답변 제공"""
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
