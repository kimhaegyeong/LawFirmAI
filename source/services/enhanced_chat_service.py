# -*- coding: utf-8 -*-
"""
Enhanced Chat Service
개선된 채팅 메시지 처리 서비스
"""

import os
import time
import asyncio
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..utils.config import Config
from ..utils.logger import get_logger
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
        
        # 핵심 컴포넌트 초기화
        self._initialize_core_components()
        
        # 통합 서비스 초기화
        self._initialize_unified_services()
        
        # 법률 제한 시스템 초기화
        self._initialize_legal_restriction_systems()
        
        # 고급 검색 엔진 초기화
        self._initialize_advanced_search_engines()
        
        # Phase 시스템 초기화
        self._initialize_phase_systems()
        
        # 자연스러운 대화 개선 시스템 초기화
        self._initialize_natural_conversation_systems()
        
        # 성능 최적화 시스템 초기화
        self._initialize_performance_systems()
        
        # 품질 개선 시스템 초기화
        self._initialize_quality_enhancement_systems()
        
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
    
    def _initialize_core_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            # 데이터베이스 관리자
            self.db_manager = DatabaseManager("data/lawfirm.db")
            
            # 벡터 스토어
            self.vector_store = LegalVectorStore()
            # 벡터 인덱스 로드
            try:
                self.vector_store.load_index()
                self.logger.info("벡터 인덱스 로드 성공")
            except Exception as e:
                self.logger.warning(f"벡터 인덱스 로드 실패: {e}")
            
            # 모델 관리자
            from .optimized_model_manager import OptimizedModelManager
            self.model_manager = OptimizedModelManager()
            
            # RAG 서비스 (MLEnhancedRAGService는 제거하고 UnifiedRAGService만 사용)
            # self.rag_service = MLEnhancedRAGService(...)
            
            # 하이브리드 검색 엔진
            self.hybrid_search_engine = HybridSearchEngine()
            
            # 질문 분류기
            self.question_classifier = QuestionClassifier()
            
            # 개선된 답변 생성기
            self.improved_answer_generator = ImprovedAnswerGenerator()
            
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
                vector_store=self.vector_store
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
    
    def _initialize_phase_systems(self):
        """Phase 시스템 초기화"""
        try:
            # Phase 1: 대화 맥락 강화
            self.integrated_session_manager = IntegratedSessionManager(self.config)
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.context_compressor = ContextCompressor(self.config)
            
            # Phase 2: 개인화 및 지능형 분석
            self.user_profile_manager = UserProfileManager(self.config)
            self.emotion_intent_analyzer = EmotionIntentAnalyzer()
            self.conversation_flow_tracker = ConversationFlowTracker(self.config)
            
            # Phase 3: 장기 기억 및 품질 모니터링
            self.contextual_memory_manager = ContextualMemoryManager(self.config)
            self.conversation_quality_monitor = ConversationQualityMonitor(self.config)
            
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
            
            # 자연스러움 개선 적용
            if response_result.get("response"):
                response_result["response"] = await self._apply_naturalness_improvements(
                    response_result["response"], phase1_info, phase2_info, user_id
                )
            
            # 후처리
            final_result = await self._post_process_response(
                response_result, query_analysis, user_id, session_id
            )
            
            # 처리 시간 추가
            final_result["processing_time"] = time.time() - start_time
            final_result["session_id"] = session_id
            final_result["user_id"] = user_id
            
            # 캐시 저장
            if self.cache_manager:
                self.cache_manager.set(cache_key, final_result, ttl_seconds=3600)
            
            return final_result
            
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
        """질문 분석"""
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
            
            # 법률 조문 패턴 검색 (개선된 정규표현식)
            import re
            
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
                        question_type=query_analysis.get("query_type", "general"),
                        intent=query_analysis.get("intent", "unknown"),
                        entities=query_analysis.get("entities", []),
                        confidence=query_analysis.get("confidence", 0.5)
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
        """향상된 답변 생성"""
        self.logger.info(f"_generate_enhanced_response called for: {message}")
        try:
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
                    
                    if rag_response and rag_response.response:
                        return {
                            "response": rag_response.response,
                            "confidence": 0.7,
                            "sources": rag_response.sources,
                            "query_analysis": query_analysis,
                            "generation_method": "simple_rag",
                            "session_id": session_id,
                            "user_id": user_id
                        }
                except Exception as e:
                    self.logger.debug(f"Simple RAG service failed: {e}")
            else:
                self.logger.warning("unified_rag_service is None, skipping RAG generation")
            
            # 2순위: 고급 시스템들
            # COT 프롬프트 시도
            if self.unified_prompt_manager:
                try:
                    cot_result = await self._generate_with_cot_prompt(message, query_analysis, user_id, session_id)
                    if cot_result:
                        return cot_result
                except Exception as e:
                    self.logger.debug(f"COT generation failed: {e}")
            
            # 개선된 답변 생성기 시도
            improved_result = await self._generate_with_improved_generator(message, query_analysis, user_id, session_id)
            if improved_result:
                return improved_result
            
            # 3순위: 간단한 템플릿 기반 답변 (fallback)
            simple_result = self._generate_improved_template_response(message, query_analysis)
            if simple_result and simple_result.get("confidence", 0) > 0.3:
                return simple_result
            
            # 최종 fallback: 개선된 답변
            fallback_response = self._generate_improved_fallback_response(message, query_analysis)
            return {
                "response": fallback_response,
                "confidence": 0.6,
                "sources": [],
                "query_analysis": query_analysis,
                "generation_method": "improved_fallback",
                "session_id": session_id,
                "user_id": user_id
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced response generation failed: {e}")
            return {
                "response": f"'{message}'에 대한 질문을 받았습니다. 관련 정보를 찾아 답변드리겠습니다.",
                "confidence": 0.5,
                "sources": [],
                "query_analysis": query_analysis,
                "generation_method": "error_fallback",
                "session_id": session_id,
                "user_id": user_id,
                "error": str(e)
            }
    
    def _generate_improved_template_response(self, message: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """개선된 템플릿 기반 답변 생성"""
        self.logger.info(f"_generate_improved_template_response called for: {message}")
        
        # 법률 조문이 있는 경우
        if query_analysis.get("statute_match") or query_analysis.get("statute_law") or query_analysis.get("statute_article"):
            return self._generate_statute_template(message, query_analysis.get("intent", "unknown"))
        
        # 1. ML 기반 분류 우선 시도
        if query_analysis.get("confidence", 0) > 0.7:
            domain = query_analysis.get("query_type", "general")
            intent = query_analysis.get("intent", "unknown")
        else:
            # 2. 가중치 기반 키워드 분류
            domain = self._classify_domain_by_weight(message)
            intent = self._extract_intent_from_message(message, domain)
        
        # 3. 의도 기반 세분화된 템플릿 생성
        return self._generate_template_by_intent(message, domain, intent)
    
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
        """의도에 따른 세분화된 템플릿 생성"""
        
        # 의도별 템플릿 매핑
        intent_mapping = {
            "contract": {
                "creation": self._generate_contract_creation_template,
                "termination": self._generate_contract_termination_template,
                "review": self._generate_contract_review_template,
                "dispute": self._generate_contract_dispute_template,
                "default": self._generate_contract_template
            },
            "civil_law": {
                "litigation": self._generate_civil_litigation_template,
                "statute_of_limitations": self._generate_statute_limitations_template,
                "damages": self._generate_damages_template,
                "debt": self._generate_debt_template,
                "default": self._generate_civil_law_template
            },
            "real_estate": {
                "purchase": self._generate_real_estate_purchase_template,
                "rental": self._generate_real_estate_rental_template,
                "registration": self._generate_real_estate_registration_template,
                "investment": self._generate_real_estate_investment_template,
                "default": self._generate_real_estate_template
            },
            "family_law": {
                "divorce": self._generate_divorce_template,
                "custody": self._generate_custody_template,
                "inheritance": self._generate_inheritance_template,
                "marriage": self._generate_marriage_template,
                "default": self._generate_family_law_template
            },
            "labor_law": {
                "termination": self._generate_labor_termination_template,
                "wage": self._generate_labor_wage_template,
                "working_conditions": self._generate_labor_conditions_template,
                "discrimination": self._generate_labor_discrimination_template,
                "default": self._generate_labor_law_template
            },
            "commercial_law": {
                "incorporation": self._generate_commercial_incorporation_template,
                "management": self._generate_commercial_management_template,
                "securities": self._generate_commercial_securities_template,
                "merger": self._generate_commercial_merger_template,
                "default": self._generate_commercial_law_template
            },
            "criminal_law": {
                "murder": self._generate_criminal_murder_template,
                "theft": self._generate_criminal_theft_template,
                "fraud": self._generate_criminal_fraud_template,
                "assault": self._generate_criminal_assault_template,
                "default": self._generate_criminal_law_template
            },
            "general": {
                "consultation": self._generate_general_consultation_template,
                "information": self._generate_general_information_template,
                "procedure": self._generate_general_procedure_template,
                "default": self._generate_general_template
            }
        }
        
        # 해당 도메인의 의도별 템플릿 함수 가져오기
        domain_mapping = intent_mapping.get(domain, intent_mapping["general"])
        template_func = domain_mapping.get(intent, domain_mapping["default"])
        
        return template_func(message, intent)
    
    # === 세분화된 템플릿 함수들 ===
    
    def _generate_civil_litigation_template(self, message: str, intent: str) -> Dict[str, Any]:
        """민사소송 전용 템플릿"""
        response = """민사소송 절차에 대해 안내드리겠습니다.

**민사소송의 기본 절차**
1. **소장 제출**: 법원에 소장을 제출하여 소송을 시작
2. **소송비용 납부**: 인지대, 송달료 등 소송비용 납부
3. **기일 통지**: 변론기일 통지서 수령
4. **변론**: 원고와 피고가 법정에서 주장과 입증
5. **판결**: 법원의 최종 판결 선고
6. **항소/상고**: 불복 시 상급법원에 항소 또는 상고

**소송 기간**
- 1심: 보통 6개월~1년
- 항소: 6개월~1년
- 상고: 6개월~1년

**소송비용**
- 인지대: 소송목적가액에 따라 차등
- 변호사 선임비용
- 증거조사비용

**주의사항**
- 소송기간은 사안에 따라 달라질 수 있음
- 전문가 상담 권장

구체적인 소송 문제는 변호사와 상담하시기 바랍니다."""
        
        return {
            "response": response,
            "confidence": 0.9,
            "sources": [],
            "generation_method": "template_civil_litigation"
        }
    
    def _generate_statute_limitations_template(self, message: str, intent: str) -> Dict[str, Any]:
        """소멸시효 전용 템플릿"""
        response = """소멸시효에 대해 안내드리겠습니다.

**소멸시효의 기본 원칙**
- 민법 제162조: 일반채권의 소멸시효는 10년
- 민법 제163조: 상사채권의 소멸시효는 5년
- 민법 제164조: 단기소멸시효는 3년 또는 1년

**주요 소멸시효 기간**
**3년 시효:**
- 의료비, 변호사보수 등 전문가 보수
- 교사, 강사, 기술자 등의 보수
- 제조자, 도매상, 소매상의 상품대금

**1년 시효:**
- 숙박료, 음식료, 대차료
- 노무자의 급료, 운송료
- 임대료, 사용료

**시효 중단 사유**
- 청구 (소송 제기, 지급명령 신청 등)
- 압류, 가압류, 가처분
- 승인 (채무 인정)

**시효 완성 효과**
- 채권이 소멸하여 청구할 수 없음
- 시효 완성 후에는 상대방이 시효이익을 포기하지 않는 한 소멸

**실무 팁**
- 시효 완성 전에 법적 조치 필요
- 시효 중단 사유 확인 중요

구체적인 시효 문제는 변호사와 상담하시기 바랍니다."""
        
        return {
            "response": response,
            "confidence": 0.9,
            "sources": [],
            "generation_method": "template_statute_limitations"
        }
    
    def _generate_damages_template(self, message: str, intent: str) -> Dict[str, Any]:
        """손해배상 전용 템플릿"""
        response = """손해배상에 대해 안내드리겠습니다.

**손해배상의 요건**
1. 불법행위 또는 채무불이행
2. 손해의 발생
3. 인과관계
4. 고의 또는 과실

**손해배상 청구 방법**
- 협의를 통한 해결
- 조정을 통한 해결
- 소송을 통한 해결

구체적인 손해배상 문제는 변호사와 상담하시기 바랍니다."""
        
        return {
            "response": response,
            "confidence": 0.8,
            "sources": [],
            "generation_method": "template_damages"
        }
    
    def _generate_debt_template(self, message: str, intent: str) -> Dict[str, Any]:
        """채무채권 전용 템플릿"""
        response = """채무채권에 대해 안내드리겠습니다.

**채무채권의 기본 원칙**
- 민법 제374조: 채권은 채무자로 하여금 일정한 행위를 하게 할 권리
- 민법 제390조: 채무자는 채권의 목적에 좇아 이행할 의무

**채무불이행의 효과**
- 손해배상 청구
- 강제이행 청구
- 계약해지

**채권보전 방법**
- 압류, 가압류
- 가처분
- 채권자대위권

구체적인 채무채권 문제는 변호사와 상담하시기 바랍니다."""
        
        return {
            "response": response,
            "confidence": 0.8,
            "sources": [],
            "generation_method": "template_debt"
        }
    
    # 기존 템플릿 함수들을 기본 템플릿으로 사용
    def _generate_contract_creation_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_contract_template(message, intent)
    
    def _generate_contract_termination_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_contract_template(message, intent)
    
    def _generate_contract_review_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_contract_template(message, intent)
    
    def _generate_contract_dispute_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_contract_template(message, intent)
    
    def _generate_real_estate_purchase_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_real_estate_template(message, intent)
    
    def _generate_real_estate_rental_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_real_estate_template(message, intent)
    
    def _generate_real_estate_registration_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_real_estate_template(message, intent)
    
    def _generate_real_estate_investment_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_real_estate_template(message, intent)
    
    def _generate_divorce_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_family_law_template(message, intent)
    
    def _generate_custody_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_family_law_template(message, intent)
    
    def _generate_inheritance_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_family_law_template(message, intent)
    
    def _generate_marriage_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_family_law_template(message, intent)
    
    def _generate_labor_termination_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_labor_law_template(message, intent)
    
    def _generate_labor_wage_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_labor_law_template(message, intent)
    
    def _generate_labor_conditions_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_labor_law_template(message, intent)
    
    def _generate_labor_discrimination_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_labor_law_template(message, intent)
    
    def _generate_commercial_incorporation_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_commercial_law_template(message, intent)
    
    def _generate_commercial_management_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_commercial_law_template(message, intent)
    
    def _generate_commercial_securities_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_commercial_law_template(message, intent)
    
    def _generate_commercial_merger_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_commercial_law_template(message, intent)
    
    def _generate_criminal_murder_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_criminal_law_template(message, intent)
    
    def _generate_criminal_theft_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_criminal_law_template(message, intent)
    
    def _generate_criminal_fraud_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_criminal_law_template(message, intent)
    
    def _generate_criminal_assault_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_criminal_law_template(message, intent)
    
    def _generate_general_consultation_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_general_template(message, intent)
    
    def _generate_general_information_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_general_template(message, intent)
    
    def _generate_general_procedure_template(self, message: str, intent: str) -> Dict[str, Any]:
        return self._generate_general_template(message, intent)
    
    def _generate_statute_template(self, message: str, intent: str) -> Dict[str, Any]:
        """법률 조문 템플릿 답변"""
        response = f"'{message}'에 대한 법률 조문 정보를 제공해드리겠습니다. 관련 법령의 내용과 해석에 대해 안내드릴 수 있습니다."
        return {
            "response": response,
            "confidence": 0.8,
            "sources": [],
            "generation_method": "template_statute"
        }
    
    def _generate_contract_template(self, message: str, intent: str) -> Dict[str, Any]:
        """계약서 템플릿 답변"""
        if "작성" in message:
            response = """계약서 작성에 대해 안내드리겠습니다.

**계약서 필수 요소**
- 당사자 명시
- 계약 목적과 내용
- 계약 기간
- 대가와 지급 방법
- 위약금 조항

**주의사항**
- 명확한 용어 사용
- 불공정한 조항 배제
- 법적 검토 권장

구체적인 계약서는 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
        else:
            response = f"'{message}'에 대한 계약 관련 질문을 받았습니다. 계약법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_contract"
        }
    
    def _generate_real_estate_template(self, message: str, intent: str) -> Dict[str, Any]:
        """부동산 템플릿 답변"""
        if "매매" in message:
            response = """부동산 매매 절차에 대해 안내드리겠습니다.

**매매 절차**
1. 부동산 확인 및 조사
2. 매매계약서 작성
3. 계약금 지급
4. 잔금 지급 및 인도
5. 소유권 이전 등기

**주의사항**
- 등기부등본 확인
- 권리관계 조사
- 중개업소 확인

구체적인 부동산 거래는 전문가와 상담하시기 바랍니다."""
            confidence = 0.8
        else:
            response = f"'{message}'에 대한 부동산 관련 질문을 받았습니다. 부동산법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_real_estate"
        }
    
    def _generate_family_law_template(self, message: str, intent: str) -> Dict[str, Any]:
        """가족법 템플릿 답변"""
        if "이혼" in message:
            response = """이혼 절차에 대해 안내드리겠습니다.

**이혼 유형별 절차**
1. 협의이혼: 부부 합의 후 가정법원 신고
2. 조정이혼: 가정법원 조정을 통한 합의
3. 재판이혼: 법원 판결을 통한 이혼

**고려사항**
- 자녀 양육권
- 재산 분할
- 위자료

구체적인 이혼 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
        else:
            response = f"'{message}'에 대한 가족법 관련 질문을 받았습니다. 가족법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_family_law"
        }
    
    def _generate_civil_law_template(self, message: str, intent: str) -> Dict[str, Any]:
        """민사법 템플릿 답변"""
        message_lower = message.lower()
        
        if "민사소송" in message or "소송절차" in message:
            response = """민사소송 절차에 대해 안내드리겠습니다.

**민사소송의 기본 절차**
1. **소장 제출**: 법원에 소장을 제출하여 소송을 시작
2. **소송비용 납부**: 인지대, 송달료 등 소송비용 납부
3. **기일 통지**: 변론기일 통지서 수령
4. **변론**: 원고와 피고가 법정에서 주장과 입증
5. **판결**: 법원의 최종 판결 선고
6. **항소/상고**: 불복 시 상급법원에 항소 또는 상고

**소송 기간**
- 1심: 보통 6개월~1년
- 항소: 6개월~1년
- 상고: 6개월~1년

**소송비용**
- 인지대: 소송목적가액에 따라 차등
- 변호사 선임비용
- 증거조사비용

**주의사항**
- 소송기간은 사안에 따라 달라질 수 있음
- 전문가 상담 권장

구체적인 소송 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.9
            
        elif "소멸시효" in message or "시효" in message:
            response = """소멸시효에 대해 안내드리겠습니다.

**소멸시효의 기본 원칙**
- 민법 제162조: 일반채권의 소멸시효는 10년
- 민법 제163조: 상사채권의 소멸시효는 5년
- 민법 제164조: 단기소멸시효는 3년 또는 1년

**주요 소멸시효 기간**
**3년 시효:**
- 의료비, 변호사보수 등 전문가 보수
- 교사, 강사, 기술자 등의 보수
- 제조자, 도매상, 소매상의 상품대금

**1년 시효:**
- 숙박료, 음식료, 대차료
- 노무자의 급료, 운송료
- 임대료, 사용료

**시효 중단 사유**
- 청구 (소송 제기, 지급명령 신청 등)
- 압류, 가압류, 가처분
- 승인 (채무 인정)

**시효 완성 효과**
- 채권이 소멸하여 청구할 수 없음
- 시효 완성 후에는 상대방이 시효이익을 포기하지 않는 한 소멸

**실무 팁**
- 시효 완성 전에 법적 조치 필요
- 시효 중단 사유 확인 중요

구체적인 시효 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.9
            
        elif "손해배상" in message:
            response = """손해배상에 대해 안내드리겠습니다.

**손해배상의 요건**
1. 불법행위 또는 채무불이행
2. 손해의 발생
3. 인과관계
4. 고의 또는 과실

**손해배상 청구 방법**
- 협의를 통한 해결
- 조정을 통한 해결
- 소송을 통한 해결

구체적인 손해배상 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        else:
            response = f"'{message}'에 대한 민사법 관련 질문을 받았습니다. 민사법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_civil_law"
        }
    
    def _generate_labor_law_template(self, message: str, intent: str) -> Dict[str, Any]:
        """노동법 템플릿 답변"""
        if "해고" in message:
            response = """해고에 대해 안내드리겠습니다.

**정당한 해고 사유**
1. 근로자의 귀책사유
2. 경영상 필요
3. 정년 도달

**해고 절차**
- 사전 통지 (30일)
- 해고 사유 명시
- 해고 통지서 교부

**부당해고 구제**
- 부당해고 구제신청
- 복직 명령
- 손해배상 청구

구체적인 해고 문제는 노동위원회나 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
        else:
            response = f"'{message}'에 대한 노동법 관련 질문을 받았습니다. 노동법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_labor_law"
        }
    
    def _generate_commercial_law_template(self, message: str, intent: str) -> Dict[str, Any]:
        """상법 템플릿 답변"""
        if "회사" in message:
            response = """회사에 대해 안내드리겠습니다.

**회사 형태**
1. 주식회사
2. 유한책임회사
3. 합명회사
4. 합자회사

**회사 설립 절차**
1. 정관 작성
2. 주주 모집
3. 설립 등기
4. 사업자 등록

**회사 운영**
- 이사회 구성
- 주주총회 개최
- 재무제표 작성

구체적인 회사 문제는 변호사나 회계사와 상담하시기 바랍니다."""
            confidence = 0.8
        else:
            response = f"'{message}'에 대한 상법 관련 질문을 받았습니다. 상법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_commercial_law"
        }
    
    def _generate_criminal_law_template(self, message: str, intent: str) -> Dict[str, Any]:
        """형사법 템플릿 답변"""
        message_lower = message.lower()
        
        if "살인" in message:
            response = """살인죄의 처벌에 대해 설명드리겠습니다.

**살인죄 처벌**
- 형법 제250조: 사형, 무기 또는 5년 이상의 징역
- 일반살인: 사형, 무기 또는 5년 이상의 징역
- 존속살인: 사형, 무기 또는 7년 이상의 징역

**살인죄의 구성요건**
1. 타인의 생명을 침해하는 행위
2. 살인의 고의 (의도)
3. 결과의 발생

**참고사항**
- 살인미수도 동일한 처벌 대상
- 정신장애 등으로 책임능력이 감소된 경우 형의 감경 가능
- 구체적인 사안은 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        elif "절도" in message:
            response = """절도죄의 처벌에 대해 설명드리겠습니다.

**절도죄 처벌**
- 형법 제329조: 6년 이하의 징역 또는 1천만원 이하의 벌금
- 야간주거침입절도: 1년 이상 10년 이하의 징역
- 특수절도: 1년 이상 10년 이하의 징역

**절도죄의 구성요건**
1. 타인의 재물을 절취하는 행위
2. 불법영득의 의사
3. 재물의 점유 이탈

**참고사항**
- 절도미수도 처벌 대상
- 친족상도례: 친족 간 절도는 고소가 있어야 공소제기
- 구체적인 사안은 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        elif "사기" in message:
            response = """사기죄의 처벌에 대해 설명드리겠습니다.

**사기죄 처벌**
- 형법 제347조: 10년 이하의 징역 또는 2천만원 이하의 벌금
- 컴퓨터 등 사용사기: 10년 이하의 징역 또는 2천만원 이하의 벌금
- 신용카드 등 사용사기: 5년 이하의 징역 또는 1천만원 이하의 벌금

**사기죄의 구성요건**
1. 기망행위 (속임수)
2. 착오 유발
3. 재산상 이익 취득
4. 인과관계

**참고사항**
- 사기미수도 처벌 대상
- 피해자와의 합의는 형의 감경 사유
- 구체적인 사안은 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        else:
            response = f"'{message}'에 대한 형사법 관련 질문을 받았습니다. 형사법의 기본 원칙과 관련 정보를 제공해드릴 수 있습니다."
            confidence = 0.7
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_criminal_law"
        }
    
    def _generate_general_template(self, message: str, intent: str) -> Dict[str, Any]:
        """일반 템플릿 답변"""
        message_lower = message.lower()
        
        if "안녕" in message or "인사" in message:
            response = "안녕하세요! 법률 관련 질문이 있으시면 언제든지 말씀해 주세요. 일반적인 법률 정보를 제공해드릴 수 있습니다."
            confidence = 0.75
            
        elif "도움" in message or "정보" in message:
            response = f"'{message}'에 대한 질문을 받았습니다. 법률 관련 일반적인 정보를 제공해드릴 수 있습니다. 구체적인 질문을 해주시면 더 정확한 정보를 안내해드릴 수 있습니다."
            confidence = 0.65
            
        elif "계약" in message:
            response = """계약에 대한 일반적인 정보를 안내드리겠습니다.

**계약의 기본 원칙**
- 민법 제105조: 계약은 당사자 간의 합의로 성립
- 자유의사에 기한 합의
- 법률이 금지하지 않는 내용이어야 함

**계약의 효력**
- 계약 체결 시 당사자는 계약 내용을 이행할 의무
- 계약 위반 시 손해배상 책임 발생
- 계약 해지 시 정당한 사유 필요

**계약서 작성 시 주의사항**
- 당사자 명시
- 계약 목적 명확히 기술
- 이행 기간 및 방법 명시
- 위약금 조항 포함

구체적인 계약 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        elif "손해" in message or "배상" in message:
            response = """손해배상에 대한 일반적인 정보를 안내드리겠습니다.

**손해배상의 기본 원칙**
- 민법 제750조: 불법행위로 인한 손해배상
- 민법 제390조: 채무불이행으로 인한 손해배상

**손해배상의 요건**
1. 불법행위 또는 채무불이행
2. 손해의 발생
3. 인과관계
4. 고의 또는 과실

**손해의 종류**
- 재산적 손해: 직접적 손해, 이익 상실
- 정신적 손해: 위자료, 정신적 고통

**손해배상 청구 방법**
- 협의를 통한 해결
- 조정을 통한 해결
- 소송을 통한 해결

구체적인 손해배상 문제는 변호사와 상담하시기 바랍니다."""
            confidence = 0.8
            
        else:
            response = f"'{message}'에 대한 질문을 받았습니다. 법률 관련 일반적인 정보를 제공해드릴 수 있습니다. 더 구체적인 정보가 필요하시면 추가 질문을 해주세요."
            confidence = 0.6
        
        return {
            "response": response,
            "confidence": confidence,
            "sources": [],
            "generation_method": "template_general"
        }
    
    async def _generate_with_cot_prompt(self, message: str, query_analysis: Dict[str, Any], user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """COT 프롬프트를 사용한 답변 생성"""
        try:
            if not self.unified_prompt_manager:
                return None
            
            # COT 프롬프트 생성
            cot_prompt = await self.unified_prompt_manager.generate_cot_prompt(
                message=message,
                query_analysis=query_analysis,
                model_type="gemini"
            )
            
            if not cot_prompt:
                return None
            
            # 모델을 사용한 응답 생성
            if self.model_manager:
                response = await self.model_manager.generate_response(cot_prompt)
                if response:
                    # response가 문자열인 경우 딕셔너리로 변환
                    if isinstance(response, str):
                        return {
                            "response": response,
                            "confidence": 0.8,
                            "sources": [],
                            "query_analysis": query_analysis,
                            "generation_method": "cot_prompt",
                            "session_id": session_id,
                            "user_id": user_id
                        }
                    elif isinstance(response, dict):
                        return response
                    else:
                        return {
                            "response": str(response),
                            "confidence": 0.8,
                            "sources": [],
                            "query_analysis": query_analysis,
                            "generation_method": "cot_prompt",
                            "session_id": session_id,
                            "user_id": user_id
                        }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"COT prompt generation failed: {e}")
            return None
    
    async def _generate_with_improved_generator(self, message: str, query_analysis: Dict[str, Any], user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """개선된 답변 생성기를 사용한 답변 생성"""
        try:
            if not self.improved_answer_generator:
                return None
            
            # 질문 분류 정보 생성
            from .question_classifier import QuestionClassification
            classification_details = QuestionClassification(
                question_type=query_analysis.get("query_type", "general"),
                intent=query_analysis.get("intent", "unknown"),
                confidence=query_analysis.get("confidence", 0.5),
                subcategories=[]
            )
            
            # 개선된 답변 생성
            improved_response = await self.improved_answer_generator.generate_answer(
                question=message,
                classification_details=classification_details,
                context=query_analysis.get("context"),
                user_id=user_id
            )
            
            if improved_response:
                # improved_response가 문자열인 경우 딕셔너리로 변환
                if isinstance(improved_response, str):
                    return {
                        "response": improved_response,
                        "confidence": 0.75,
                        "sources": [],
                        "query_analysis": query_analysis,
                        "generation_method": "improved_generator",
                        "session_id": session_id,
                        "user_id": user_id
                    }
                elif isinstance(improved_response, dict):
                    return improved_response
                else:
                    return {
                        "response": str(improved_response),
                        "confidence": 0.75,
                        "sources": [],
                        "query_analysis": query_analysis,
                        "generation_method": "improved_generator",
                        "session_id": session_id,
                        "user_id": user_id
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Improved generator failed: {e}")
            return None
    
    async def _apply_naturalness_improvements(self, response: str, phase1_info: Dict[str, Any], phase2_info: Dict[str, Any], user_id: str) -> str:
        """자연스러움 개선 적용"""
        try:
            improved_response = response
            
            # phase2_info가 문자열인 경우 딕셔너리로 변환
            if isinstance(phase2_info, str):
                self.logger.debug(f"phase2_info가 문자열로 전달됨: {type(phase2_info)}")
                phase2_info = {"emotion_intent": {}}
            elif phase2_info is None:
                self.logger.debug("phase2_info가 None으로 전달됨")
                phase2_info = {"emotion_intent": {}}
            elif not isinstance(phase2_info, dict):
                self.logger.debug(f"phase2_info가 예상치 못한 타입으로 전달됨: {type(phase2_info)}")
                phase2_info = {"emotion_intent": {}}
            
            # 감정 톤 조절
            if self.emotional_tone_adjuster:
                improved_response = await self._adjust_emotional_tone(improved_response, phase2_info)
            
            # 대화 연결
            if self.conversation_connector:
                improved_response = await self._connect_conversation(improved_response, phase1_info)
            
            return improved_response
            
        except Exception as e:
            self.logger.error(f"자연스러움 개선 적용 중 오류: {e}")
            return response
    
    async def _adjust_emotional_tone(self, answer: str, phase2_info: Dict[str, Any]) -> str:
        """감정 톤 조절"""
        try:
            if not self.emotional_tone_adjuster:
                return answer
            
            # phase2_info가 문자열인 경우 딕셔너리로 변환
            if isinstance(phase2_info, str):
                self.logger.debug(f"_adjust_emotional_tone에서 phase2_info가 문자열로 전달됨: {type(phase2_info)}")
                phase2_info = {"emotion_intent": {}}
            elif phase2_info is None:
                self.logger.debug("_adjust_emotional_tone에서 phase2_info가 None으로 전달됨")
                phase2_info = {"emotion_intent": {}}
            elif not isinstance(phase2_info, dict):
                self.logger.debug(f"_adjust_emotional_tone에서 phase2_info가 예상치 못한 타입으로 전달됨: {type(phase2_info)}")
                phase2_info = {"emotion_intent": {}}
            
            emotion_info = phase2_info.get("emotion_intent", {})
            try:
                adjusted_answer = self.emotional_tone_adjuster.adjust_tone(
                    answer, emotion_info.get("emotion_type", "neutral")
                )
                return adjusted_answer
            except Exception as e:
                self.logger.debug(f"감정 톤 조절 실패: {e}")
                return answer
            
        except Exception as e:
            self.logger.error(f"감정 톤 조절 중 오류: {e}")
            return answer
    
    async def _connect_conversation(self, answer: str, phase1_info: Dict[str, Any]) -> str:
        """대화 연결"""
        try:
            if not self.conversation_connector:
                return answer
            
            session_context = phase1_info.get("session_context", {})
            try:
                connected_answer = await self.conversation_connector.connect_conversation(
                    answer, session_context
                )
                return connected_answer
            except AttributeError:
                self.logger.debug("ConversationConnector에 connect_conversation 메서드가 없습니다")
                return answer
            except Exception as e:
                self.logger.debug(f"대화 연결 실패: {e}")
                return answer
            
        except Exception as e:
            self.logger.error(f"대화 연결 중 오류: {e}")
            return answer
    
    def _remove_repetitive_patterns(self, response: str) -> str:
        """반복적인 패턴 제거"""
        try:
            import re
            
            # "### 관련 법령\n관련 법령:" 같은 반복 패턴 제거 (더 강력한 패턴)
            response = re.sub(r'###\s*관련\s*법령\s*\n+\s*관련\s*법령\s*:\s*\n+', '', response)
            response = re.sub(r'###\s*법령\s*해설\s*\n+\s*법령\s*해설\s*:\s*\n+', '', response)
            response = re.sub(r'###\s*적용\s*사례\s*\n+\s*실제\s*적용\s*사례\s*:\s*\n+', '', response)
            
            # "문의하신 내용에 대해", "질문하신 내용에 대해", "관련해서 말씀드리면" 같은 불필요한 서론 제거
            response = re.sub(r'(문의하신|질문하신)\s*내용에\s*대해\s*', '', response)
            response = re.sub(r'관련해서\s*말씀드리면\s*', '', response)
            
            # 중복된 제목 제거 (예: "## 제목\n제목:")
            response = re.sub(r'(###+\s*[^\n]+)\s*\n+\s*\1\s*:', r'\1\n\n', response, flags=re.IGNORECASE)
            
            # 연속된 빈 줄 제거 (3개 이상의 연속 줄바꿈을 2개로)
            response = re.sub(r'\n{3,}', '\n\n', response)
            
            # 응답 시작 부분의 불필요한 섹션 제목 제거
            response = re.sub(r'^###\s*관련\s*법령\s*\n+', '', response)
            
            return response.strip()
        except Exception as e:
            self.logger.debug(f"패턴 제거 중 오류: {e}")
            return response
    
    async def _post_process_response(self, response_result: Dict[str, Any], query_analysis: Dict[str, Any], user_id: str, session_id: str) -> Dict[str, Any]:
        """후처리"""
        try:
            # response_result가 문자열인 경우 딕셔너리로 변환
            if isinstance(response_result, str):
                self.logger.debug(f"response_result가 문자열로 전달됨: {type(response_result)}")
                response_result = {"response": response_result, "confidence": 0.5}
            elif response_result is None:
                self.logger.debug("response_result가 None으로 전달됨")
                response_result = {"response": "죄송합니다. 답변을 생성할 수 없습니다.", "confidence": 0.1}
            elif not isinstance(response_result, dict):
                self.logger.debug(f"response_result가 예상치 못한 타입으로 전달됨: {type(response_result)}")
                response_result = {"response": str(response_result), "confidence": 0.3}
            
            # 반복 패턴 제거
            if response_result.get("response"):
                response_result["response"] = self._remove_repetitive_patterns(response_result["response"])
            
            # 답변 구조화 향상 적용
            if response_result.get("response") and self.answer_structure_enhancer:
                try:
                    question_type = query_analysis.get("query_type", "general")
                    # QuestionType enum을 문자열로 변환
                    if hasattr(question_type, 'value'):
                        question_type = question_type.value
                    elif hasattr(question_type, 'name'):
                        question_type = question_type.name
                    elif not isinstance(question_type, str):
                        question_type = str(question_type)
                    
                    question = query_analysis.get("original_query", "")
                    domain = query_analysis.get("domain", "general")
                    
                    structure_result = self.answer_structure_enhancer.enhance_answer_structure(
                        response_result["response"],
                        question_type,
                        question,
                        domain
                    )
                    
                    if structure_result and not structure_result.get("error"):
                        response_result["response"] = structure_result.get("structured_answer", response_result["response"])
                        response_result["structure_metrics"] = structure_result.get("quality_metrics", {})
                        response_result["structure_analysis"] = structure_result.get("analysis", {})
                        response_result["structure_improvements"] = structure_result.get("improvements", [])
                        
                        # 구조화 품질에 따른 신뢰도 조정
                        overall_score = structure_result.get("quality_metrics", {}).get("overall_score", 0.0)
                        if overall_score > 0.7:
                            response_result["confidence"] = min(0.95, response_result.get("confidence", 0.5) + 0.15)
                        elif overall_score > 0.5:
                            response_result["confidence"] = min(0.9, response_result.get("confidence", 0.5) + 0.1)
                        
                        self.logger.info(f"답변 구조화 향상 완료 - 품질 점수: {overall_score:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"답변 구조화 향상 중 오류: {e}")
            
            # 품질 향상 적용
            if response_result.get("response") and self.answer_quality_enhancer:
                try:
                    enhanced_result = await self.answer_quality_enhancer.enhance_answer(
                        response_result["response"], 
                        query_analysis.get("query_type", "general")
                    )
                    if enhanced_result:
                        response_result["response"] = enhanced_result
                except AttributeError:
                    self.logger.debug("AnswerQualityEnhancer에 enhance_answer 메서드가 없습니다")
                except Exception as e:
                    self.logger.debug(f"답변 품질 향상 실패: {e}")
                    response_result["confidence"] = min(0.9, response_result.get("confidence", 0.5) + 0.1)
            
            # 신뢰도 계산
            if self.confidence_calculator:
                calculated_confidence = self.confidence_calculator.calculate_confidence(
                    response_result["response"],
                    response_result.get("sources", []),
                    query_analysis.get("query_type", "general_question")
                )
                if calculated_confidence:
                    response_result["confidence"] = calculated_confidence.confidence
            
            return response_result
            
        except Exception as e:
            self.logger.error(f"후처리 중 오류: {e}")
            self.logger.error(f"오류 발생 위치: _post_process_response")
            self.logger.error(f"response_result 타입: {type(response_result)}")
            self.logger.error(f"response_result 내용: {response_result}")
            import traceback
            self.logger.error(f"스택 트레이스: {traceback.format_exc()}")
            return response_result
    
    async def enhance_answer_structure(self, answer: str, question_type: str, question: str = "", domain: str = "general") -> Dict[str, Any]:
        """답변 구조화 향상 (공개 메서드)"""
        try:
            if not self.answer_structure_enhancer:
                return {"error": "Answer structure enhancer not initialized"}
            
            # QuestionType enum을 문자열로 변환
            if hasattr(question_type, 'value'):
                question_type = question_type.value
            elif hasattr(question_type, 'name'):
                question_type = question_type.name
            elif not isinstance(question_type, str):
                question_type = str(question_type)
            
            result = self.answer_structure_enhancer.enhance_answer_structure(
                answer, question_type, question, domain
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"답변 구조화 향상 실패: {e}")
            return {"error": str(e)}
    
    async def enhance_answer_with_legal_basis(self, answer: str, question_type: str, query: str = "") -> Dict[str, Any]:
        """법적 근거를 포함한 답변 강화"""
        try:
            if not self.answer_structure_enhancer:
                return {"error": "Answer structure enhancer not initialized"}
            
            # QuestionType enum으로 변환
            from .answer_structure_enhancer import QuestionType
            try:
                qt_enum = QuestionType(question_type.lower())
            except ValueError:
                qt_enum = QuestionType.GENERAL_QUESTION
            
            result = self.answer_structure_enhancer.enhance_answer_with_legal_basis(
                answer, qt_enum, query
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"법적 근거 강화 실패: {e}")
            return {"error": str(e)}
    
    def get_structure_template_info(self, question_type: str) -> Dict[str, Any]:
        """구조화 템플릿 정보 조회"""
        try:
            if not self.answer_structure_enhancer:
                return {"error": "Answer structure enhancer not initialized"}
            
            return self.answer_structure_enhancer.get_template_info(question_type)
            
        except Exception as e:
            self.logger.error(f"템플릿 정보 조회 실패: {e}")
            return {"error": str(e)}
    
    def reload_structure_templates(self):
        """구조화 템플릿 동적 리로드"""
        try:
            if self.answer_structure_enhancer:
                self.answer_structure_enhancer.reload_templates()
                self.logger.info("구조화 템플릿 리로드 완료")
            else:
                self.logger.warning("Answer structure enhancer not initialized")
        except Exception as e:
            self.logger.error(f"템플릿 리로드 실패: {e}")
    
    def _generate_improved_fallback_response(self, message: str, query_analysis: Dict[str, Any]) -> str:
        """개선된 폴백 응답 생성"""
        try:
            domain = query_analysis.get("domain", "general")
            query_type = query_analysis.get("query_type", "general")
            
            # 도메인별 맞춤형 폴백 응답
            domain_responses = {
                "civil_law": f"'{message}'에 대한 민사법 관련 질문으로 이해됩니다. 민법, 계약법, 손해배상 등 민사법 분야의 구체적인 내용에 대해 답변드릴 수 있습니다. 더 자세한 정보가 필요하시면 구체적인 상황을 말씀해 주세요.",
                "criminal_law": f"'{message}'에 대한 형사법 관련 질문으로 이해됩니다. 형법, 범죄 구성요건, 처벌 등 형사법 분야의 내용에 대해 답변드릴 수 있습니다. 구체적인 사안이 있으시면 말씀해 주세요.",
                "family_law": f"'{message}'에 대한 가족법 관련 질문으로 이해됩니다. 이혼, 상속, 양육권 등 가족법 분야의 내용에 대해 답변드릴 수 있습니다. 구체적인 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다.",
                "commercial_law": f"'{message}'에 대한 상법 관련 질문으로 이해됩니다. 회사법, 주식, 이사 등 상법 분야의 내용에 대해 답변드릴 수 있습니다. 구체적인 회사 형태나 상황을 말씀해 주세요.",
                "labor_law": f"'{message}'에 대한 노동법 관련 질문으로 이해됩니다. 근로계약, 임금, 해고 등 노동법 분야의 내용에 대해 답변드릴 수 있습니다. 구체적인 근로 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다.",
                "real_estate": f"'{message}'에 대한 부동산 관련 질문으로 이해됩니다. 부동산 매매, 임대차, 등기 등 부동산 분야의 내용에 대해 답변드릴 수 있습니다. 구체적인 부동산 거래 상황을 말씀해 주세요.",
                "general": f"'{message}'에 대한 법률 질문으로 이해됩니다. 관련 법령과 판례를 바탕으로 답변드릴 수 있습니다. 더 구체적인 상황이나 조건을 알려주시면 더 정확한 답변을 드릴 수 있습니다."
            }
            
            # 질문 유형별 추가 안내
            type_guidance = {
                "contract": "계약서 작성이나 검토에 대한 구체적인 내용을 말씀해 주시면 더 도움이 될 것 같습니다.",
                "procedure": "해당 절차의 구체적인 단계나 필요한 서류에 대해 더 자세히 알려드릴 수 있습니다.",
                "statute": "관련 법령의 구체적인 조문이나 해석에 대해 더 자세히 설명드릴 수 있습니다.",
                "precedent": "관련 판례나 법원의 해석에 대해 더 구체적으로 안내드릴 수 있습니다."
            }
            
            base_response = domain_responses.get(domain, domain_responses["general"])
            additional_guidance = type_guidance.get(query_type, "")
            
            if additional_guidance:
                return f"{base_response}\n\n{additional_guidance}"
            else:
                return base_response
                
        except Exception as e:
            self.logger.error(f"폴백 응답 생성 실패: {e}")
            return f"'{message}'에 대한 질문을 받았습니다. 관련 정보를 찾아 답변드리겠습니다."
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "service_name": "EnhancedChatService",
            "version": "2.0.0",
            "components": {
                "rag_service": self.rag_service is not None,
                "hybrid_search_engine": self.hybrid_search_engine is not None,
                "question_classifier": self.question_classifier is not None,
                "improved_answer_generator": self.improved_answer_generator is not None,
                "unified_search_engine": self.unified_search_engine is not None,
                "unified_rag_service": self.unified_rag_service is not None,
                "ml_validation_system": self.ml_validation_system is not None,
                "improved_legal_restriction_system": self.improved_legal_restriction_system is not None,
                "multi_stage_validation_system": self.multi_stage_validation_system is not None,
                "optimized_search_engine": self.optimized_search_engine is not None,
                "exact_search_engine": self.exact_search_engine is not None,
                "semantic_search_engine": self.semantic_search_engine is not None,
                "precedent_search_engine": self.precedent_search_engine is not None,
                "integrated_session_manager": self.integrated_session_manager is not None,
                "multi_turn_handler": self.multi_turn_handler is not None,
                "context_compressor": self.context_compressor is not None,
                "user_profile_manager": self.user_profile_manager is not None,
                "emotion_intent_analyzer": self.emotion_intent_analyzer is not None,
                "conversation_flow_tracker": self.conversation_flow_tracker is not None,
                "contextual_memory_manager": self.contextual_memory_manager is not None,
                "conversation_quality_monitor": self.conversation_quality_monitor is not None,
                "conversation_connector": self.conversation_connector is not None,
                "emotional_tone_adjuster": self.emotional_tone_adjuster is not None,
                "personalized_style_learner": self.personalized_style_learner is not None,
                "realtime_feedback_system": self.realtime_feedback_system is not None,
                "naturalness_evaluator": self.naturalness_evaluator is not None,
                "cache_manager": self.cache_manager is not None,
                "performance_monitor": self.performance_monitor is not None,
                "memory_optimizer": self.memory_optimizer is not None,
                "answer_quality_enhancer": self.answer_quality_enhancer is not None,
                "answer_structure_enhancer": self.answer_structure_enhancer is not None,
                "confidence_calculator": self.confidence_calculator is not None,
                "prompt_optimizer": self.prompt_optimizer is not None,
                "unified_prompt_manager": self.unified_prompt_manager is not None
            },
            "langgraph_enabled": self.use_langgraph,
            "timestamp": datetime.now()
        }
    
    def clear_cache(self):
        """캐시 클리어"""
        if self.cache_manager:
            self.cache_manager.clear()
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 확인"""
        return {
            "service_status": "active",
            "components_status": self.get_stats()["components"],
            "memory_usage": self.memory_optimizer.get_memory_usage() if self.memory_optimizer else None,
            "performance_metrics": self.performance_monitor.get_metrics() if self.performance_monitor else None,
            "timestamp": datetime.now()
        }
    
    def validate_input(self, message: str) -> bool:
        """입력 검증"""
        if not message or not message.strip():
            return False
        
        if len(message) > 10000:  # Max 10,000 characters
            return False
        
        return True
    
    async def test_service(self, test_message: str = "테스트 질문입니다") -> Dict[str, Any]:
        """서비스 테스트"""
        try:
            result = await self.process_message(test_message)
            
            test_passed = (
                "response" in result and 
                result["response"] and 
                "processing_time" in result
            )
            
            return {
                "test_passed": test_passed,
                "test_message": test_message,
                "result": result,
                "langgraph_enabled": self.use_langgraph
            }
            
        except Exception as e:
            return {
                "test_passed": False,
                "test_message": test_message,
                "error": str(e),
                "langgraph_enabled": self.use_langgraph
            }