# -*- coding: utf-8 -*-
"""
개선된 LangGraph Legal Workflow
답변 품질 향상을 위한 향상된 워크플로우 구현

주요 기능:
- 긴급도 평가 (Urgency Assessment)
- 법률분야 분류 강화 (Legal Field Classification)
- 법령 검증 (Legal Basis Validation)
- 문서 분석 (Document Analysis) - 계약서/고소장 등
- 전문가 라우팅 (Expert Router) - 가족법/기업법/지적재산권
- 멀티턴 대화 처리 (Multi-turn Conversation)
"""

import asyncio
import hashlib
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Global logger 사용
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
# Import with fallback for compatibility
try:
    from lawfirm_langgraph.core.workflow.state.workflow_types import QueryComplexity, RetryCounterManager
except ImportError:
    from core.workflow.state.workflow_types import QueryComplexity, RetryCounterManager

try:
    from lawfirm_langgraph.core.workflow.mixins import (
        StateUtilsMixin,
        QueryUtilsMixin,
        SearchMixin,
        AnswerGenerationMixin,
        DocumentAnalysisMixin,
        ClassificationMixin,
    )
except ImportError:
    from core.workflow.mixins import (
        StateUtilsMixin,
        QueryUtilsMixin,
        SearchMixin,
        AnswerGenerationMixin,
        DocumentAnalysisMixin,
        ClassificationMixin,
    )

from langgraph.graph import StateGraph
from pathlib import Path

try:
    from lawfirm_langgraph.core.agents.handlers.answer_formatter import AnswerFormatterHandler
except ImportError:
    from core.agents.handlers.answer_formatter import AnswerFormatterHandler

try:
    from lawfirm_langgraph.core.generation.generators.answer_generator import AnswerGenerator
except ImportError:
    from core.generation.generators.answer_generator import AnswerGenerator

try:
    from lawfirm_langgraph.core.generation.generators.context_builder import ContextBuilder
except ImportError:
    from core.generation.generators.context_builder import ContextBuilder

try:
    from lawfirm_langgraph.core.processing.extractors import (
        DocumentExtractor,
        QueryExtractor,
    )
except ImportError:
    from core.processing.extractors import (
        DocumentExtractor,
        QueryExtractor,
    )

try:
    from lawfirm_langgraph.core.search.optimizers.keyword_mapper import LegalKeywordMapper
except ImportError:
    from core.search.optimizers.keyword_mapper import LegalKeywordMapper

try:
    from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
except ImportError:
    from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2

try:
    from lawfirm_langgraph.core.shared.wrappers.node_wrappers import with_state_optimization
except ImportError:
    from core.shared.wrappers.node_wrappers import with_state_optimization

try:
    from lawfirm_langgraph.core.agents.optimizers.performance_optimizer import PerformanceOptimizer
except ImportError:
    from core.agents.optimizers.performance_optimizer import PerformanceOptimizer

try:
    from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType, DocumentTypeConfig
except ImportError:
    from core.workflow.constants.document_types import DocumentType, DocumentTypeConfig

try:
    from lawfirm_langgraph.core.workflow.builders.prompt_builders import QueryBuilder
except ImportError:
    from core.workflow.builders.prompt_builders import QueryBuilder

try:
    from lawfirm_langgraph.core.generation.validators.quality_validators import (
        SearchValidator,
    )
except ImportError:
    from core.generation.validators.quality_validators import (
        SearchValidator,
    )

try:
    from lawfirm_langgraph.core.search.optimizers.query_enhancer import QueryEnhancer
except ImportError:
    from core.search.optimizers.query_enhancer import QueryEnhancer

try:
    from lawfirm_langgraph.core.processing.extractors.reasoning_extractor import ReasoningExtractor
except ImportError:
    from core.processing.extractors.reasoning_extractor import ReasoningExtractor

try:
    from lawfirm_langgraph.core.search.handlers.search_handler import SearchHandler
except ImportError:
    from core.search.handlers.search_handler import SearchHandler

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState

try:
    from lawfirm_langgraph.core.workflow.state.state_utils import (
        MAX_DOCUMENT_CONTENT_LENGTH,
        MAX_RETRIEVED_DOCS,
        prune_retrieved_docs,
    )
except ImportError:
    from core.workflow.state.state_utils import (
        MAX_DOCUMENT_CONTENT_LENGTH,
        MAX_RETRIEVED_DOCS,
        prune_retrieved_docs,
    )

try:
    from lawfirm_langgraph.core.workflow.utils.workflow_constants import (
        RetryConfig,
        WorkflowConstants,
    )
except ImportError:
    from core.workflow.utils.workflow_constants import (
        RetryConfig,
        WorkflowConstants,
    )

try:
    from lawfirm_langgraph.core.workflow.routes.classification_routes import ClassificationRoutes
except ImportError:
    from core.workflow.routes.classification_routes import ClassificationRoutes

try:
    from lawfirm_langgraph.core.workflow.routes.search_routes import SearchRoutes
except ImportError:
    from core.workflow.routes.search_routes import SearchRoutes

try:
    from lawfirm_langgraph.core.workflow.routes.answer_routes import AnswerRoutes
except ImportError:
    from core.workflow.routes.answer_routes import AnswerRoutes

try:
    from lawfirm_langgraph.core.workflow.routes.agentic_routes import AgenticRoutes
except ImportError:
    from core.workflow.routes.agentic_routes import AgenticRoutes

try:
    from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
except ImportError:
    from core.workflow.utils.workflow_utils import WorkflowUtils

try:
    from lawfirm_langgraph.core.classification.classifiers.question_classifier import QuestionType
except ImportError:
    from core.classification.classifiers.question_classifier import QuestionType

try:
    from lawfirm_langgraph.core.search.processors.result_merger import ResultMerger, ResultRanker
except ImportError:
    from core.search.processors.result_merger import ResultMerger, ResultRanker

try:
    from lawfirm_langgraph.core.shared.utils.environment import Environment
except ImportError:
    from core.shared.utils.environment import Environment

# 호환성을 위해 기존 WorkflowRoutes도 유지 (점진적 마이그레이션)
try:
    from lawfirm_langgraph.core.workflow.utils.workflow_routes import WorkflowRoutes
except ImportError:
    try:
        from core.workflow.utils.workflow_routes import WorkflowRoutes
    except ImportError:
        from core.workflow.routes.workflow_routes import WorkflowRoutes

# 설정 파일 import (lawfirm_langgraph 구조 우선 시도)
try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    # Fallback: 기존 경로 (호환성 유지)
    try:
        from core.utils.langgraph_config import LangGraphConfig
    except ImportError:
        from core.utils.langgraph_config import LangGraphConfig
try:
    from lawfirm_langgraph.core.processing.integration.term_integration_system import TermIntegrator
except ImportError:
    try:
        from core.processing.integration.term_integration_system import TermIntegrator
    except ImportError:
        # 호환성을 위한 fallback
        from core.processing.integration.term_integration_system import TermIntegrator

try:
    from lawfirm_langgraph.core.services.unified_prompt_manager import (
        LegalDomain,
        ModelType,
        UnifiedPromptManager,
    )
except ImportError:
    # 호환성을 위한 fallback
    try:
        from core.services.unified_prompt_manager import (
            LegalDomain,
            ModelType,
            UnifiedPromptManager,
        )
    except ImportError:
        # 최종 fallback
        from core.services.unified_prompt_manager import (
            LegalDomain,
            ModelType,
            UnifiedPromptManager,
        )


# AnswerStructureEnhancer 통합 (답변 구조화 및 법적 근거 강화)
try:
    from lawfirm_langgraph.core.generation.formatters.answer_structure_enhancer import AnswerStructureEnhancer
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = True
except ImportError:
    try:
        from core.generation.formatters.answer_structure_enhancer import AnswerStructureEnhancer
        ANSWER_STRUCTURE_ENHANCER_AVAILABLE = True
    except ImportError:
        ANSWER_STRUCTURE_ENHANCER_AVAILABLE = False


# 성능 최적화: 정규식 패턴 컴파일 (모듈 레벨)
LAW_PATTERN = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
PRECEDENT_PATTERN = re.compile(r'대법원|법원.*\d{4}[다나마]\d+')

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# lawfirm_langgraph 디렉토리를 sys.path에 추가 (core 모듈 import를 위해)
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))


# Mock observe decorator (Langfuse 제거됨)
def observe(**kwargs):
    def decorator(func):
        return func
    return decorator


# Logger 초기화 (global logger 사용)
logger = get_logger(__name__)

class EnhancedLegalQuestionWorkflow(
    StateUtilsMixin,
    QueryUtilsMixin,
    SearchMixin,
    AnswerGenerationMixin,
    DocumentAnalysisMixin,
    ClassificationMixin
):
    """개선된 법률 질문 처리 워크플로우"""
    
    # 클래스 상수: 쿼리 타입별 기본 키워드
    QUERY_TYPE_KEYWORDS = {
        "precedent_search": ["판례", "사건", "판결", "대법원"],
        "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
        "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
        "procedure_guide": ["절차", "방법", "대응", "소송"],
        "term_explanation": ["의미", "정의", "개념", "해석"]
    }
    
    # 클래스 상수: 법률 분야별 기본 키워드
    FIELD_KEYWORDS = {
        "family": ["가족", "이혼", "양육", "상속", "부부"],
        "civil": ["민사", "계약", "손해배상", "채권", "채무"],
        "criminal": ["형사", "범죄", "처벌", "형량"],
        "labor": ["노동", "근로", "해고", "임금", "근로자"],
        "corporate": ["기업", "회사", "주주", "법인"]
    }
    
    # 키워드 추출 관련 상수
    MAX_FALLBACK_KEYWORDS = 20
    MIN_KEYWORD_LENGTH = 2
    
    # 메타데이터 복사 대상 필드
    METADATA_COPY_FIELDS = [
        "statute_name", "law_name", "article_no", "article_number",
        "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id",
        "chunk_id", "source_id"
    ]

    def __init__(self, config: LangGraphConfig):
        self.config = config

        # 개선: 로거를 명시적으로 초기화 (global logger 사용)
        self.logger = get_logger(__name__)

        # 로거 레벨 설정 (명시적으로 설정)
        self.logger.setLevel(logging.DEBUG)

        # 로거의 propagate 설정 (루트 로거로 전파, 중복 방지를 위해 핸들러 추가하지 않음)
        self.logger.propagate = True

        # 핸들러는 루트 로거에서 관리하므로 여기서 추가하지 않음
        # (루트 로거에 이미 핸들러가 있으면 중복 출력 방지)

        # 통합 프롬프트 관리자 초기화 (우선)
        self.unified_prompt_manager = UnifiedPromptManager()

        # 컴포넌트 초기화
        self.keyword_mapper = LegalKeywordMapper()
        from core.search.optimizers.keyword_extractor import KeywordExtractor
        self.keyword_extractor = KeywordExtractor(use_morphology=True, logger_instance=self.logger)
        self.data_connector = LegalDataConnectorV2()
        self.performance_optimizer = PerformanceOptimizer()
        self.term_integrator = TermIntegrator()
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()
        
        # KoreanStopwordProcessor 초기화 (KoNLPy 기반 불용어 처리)
        try:
            from core.utils.korean_stopword_processor import KoreanStopwordProcessor
            self.stopword_processor = KoreanStopwordProcessor()
            self.logger.debug("KoreanStopwordProcessor initialized successfully")
        except Exception as e:
            self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}, will use fallback method")
            self.stopword_processor = None
        
        # 검색 결과 처리 프로세서 초기화
        from core.search.processors.search_result_processor import SearchResultProcessor
        self.search_result_processor = SearchResultProcessor(
            logger=self.logger,
            result_merger=self.result_merger,
            result_ranker=self.result_ranker
        )
        
        # 워크플로우 문서 처리 프로세서 초기화
        from core.workflow.processors.workflow_document_processor import WorkflowDocumentProcessor
        semantic_search_engine = None
        if hasattr(self, 'search_handler') and self.search_handler:
            semantic_search_engine = getattr(self.search_handler, 'semantic_search_engine', None)
        self.workflow_document_processor = WorkflowDocumentProcessor(
            logger=self.logger,
            query_enhancer=None,  # query_enhancer는 나중에 설정됨
            semantic_search_engine=semantic_search_engine  # 문서 내용 복원을 위해 전달
        )
        
        # 워크플로우 검증기 초기화
        from core.workflow.validators.workflow_validator import WorkflowValidator
        self.workflow_validator = WorkflowValidator(logger=self.logger)
        
        # 워크플로우 프롬프트 빌더 초기화
        from core.workflow.builders.workflow_prompt_builder import WorkflowPromptBuilder
        self.workflow_prompt_builder = WorkflowPromptBuilder(logger=self.logger)
    
        # 재시도 카운터 관리자 초기화
        self.retry_manager = RetryCounterManager(self.logger)

        # 추론 과정 분리 모듈 초기화 
        self.reasoning_extractor = ReasoningExtractor(logger=self.logger)

        # AnswerStructureEnhancer 초기화 (답변 구조화 및 법적 근거 강화)
        if ANSWER_STRUCTURE_ENHANCER_AVAILABLE:
            self.answer_structure_enhancer = AnswerStructureEnhancer()
            self.logger.debug("AnswerStructureEnhancer initialized for answer quality enhancement")
        else:
            self.answer_structure_enhancer = None
            self.logger.debug("AnswerStructureEnhancer not available (optional feature)")

        # AnswerFormatter 초기화 (시각적 포맷팅 - 선택적 기능)
        # 모듈이 존재하지 않으므로 None으로 설정
        self.answer_formatter = None

        # Semantic Search Engine 초기화 (벡터 검색을 위한 - lawfirm_v2_faiss.index 사용)
        try:
            from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
            from core.utils.config import Config
            # lawfirm_v2.db 기반으로 자동으로 ./data/lawfirm_v2_faiss.index 사용
            config = Config()
            db_path = config.database_path
            
            # MLflow 벡터 스토어 설정 확인 (기본값 True)
            use_mlflow_index = getattr(config, 'use_mlflow_index', True)
            mlflow_run_id = getattr(config, 'mlflow_run_id', None)
            
            # MLflow 벡터 스토어 사용 로깅
            if use_mlflow_index:
                self.logger.debug("📌 MLflow 벡터 스토어를 사용합니다 (기본값)")
            
            # LangGraphConfig에서 embedding_model 가져오기 (법률 특화 SBERT 모델 지원)
            model_name = getattr(self.config, 'embedding_model', None)
            
            self.semantic_search = SemanticSearchEngineV2(
                db_path=db_path,
                model_name=model_name,
                use_mlflow_index=use_mlflow_index,
                mlflow_run_id=mlflow_run_id
            )
            
            # ✅ MLflow 모델 감지 결과 확인
            if hasattr(self.semantic_search, 'model_name'):
                self.logger.debug(f"✅ SemanticSearchEngineV2 초기화 완료 - 사용 모델: {self.semantic_search.model_name}")
            
            if hasattr(self.semantic_search, 'diagnose'):
                diagnosis = self.semantic_search.diagnose()
                if diagnosis.get("available"):
                    self.logger.debug(f"SemanticSearchEngineV2 initialized successfully with {db_path}")
                else:
                    self.logger.warning(f"SemanticSearchEngineV2 initialized but not available: {diagnosis.get('issues', [])}")
            else:
                self.logger.debug(f"SemanticSearchEngineV2 initialized successfully with {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SemanticSearchEngineV2: {e}", exc_info=True)
            self.semantic_search = None
            self.logger.error("SemanticSearchEngineV2 is not available. Vector search will be disabled.")

        # 검색 핸들러 초기화 (Phase 2 리팩토링) - semantic_search 초기화 이후
        self.search_handler = SearchHandler(
            semantic_search=self.semantic_search,
            keyword_mapper=self.keyword_mapper,
            data_connector=self.data_connector,
            result_merger=self.result_merger,
            result_ranker=self.result_ranker,
            performance_optimizer=self.performance_optimizer,
            config=self.config,
            logger=self.logger
        )

        # 컨텍스트 빌더 초기화 (Phase 6 리팩토링) - semantic_search 초기화 이후
        self.context_builder = ContextBuilder(
            semantic_search=self.semantic_search,
            config=self.config,
            logger=self.logger
        )

        # MultiTurnQuestionHandler 초기화 (멀티턴 질문 처리)
        try:
            from core.conversation.conversation_manager import ConversationManager
            from core.conversation.multi_turn_handler import MultiTurnQuestionHandler
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.conversation_manager = ConversationManager()
            self.logger.debug("MultiTurnQuestionHandler initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MultiTurnQuestionHandler: {e}")
            self.multi_turn_handler = None
            self.conversation_manager = None

        # AIKeywordGenerator 초기화 (AI 키워드 확장)
        try:
            from core.processing.extractors.ai_keyword_generator import AIKeywordGenerator
            self.ai_keyword_generator = AIKeywordGenerator()
            self.logger.debug("AIKeywordGenerator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AIKeywordGenerator: {e}")
            self.ai_keyword_generator = None

        # EmotionIntentAnalyzer 초기화 (긴급도 평가용)
        try:
            from core.classification.analyzers.emotion_intent_analyzer import EmotionIntentAnalyzer
            self.emotion_analyzer = EmotionIntentAnalyzer()
            self.logger.debug("EmotionIntentAnalyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize EmotionIntentAnalyzer: {e}")
            self.emotion_analyzer = None

        # LegalBasisValidator 초기화 (법령 검증용)
        try:
            from core.generation.validators.legal_basis_validator import LegalBasisValidator
            self.legal_validator = LegalBasisValidator()
            self.logger.debug("LegalBasisValidator initialized")
        except ImportError:
            try:
                # 호환성을 위한 fallback
                from core.services.legal_basis_validator import LegalBasisValidator
                self.legal_validator = LegalBasisValidator()
                self.logger.debug("LegalBasisValidator initialized (from services)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LegalBasisValidator: {e}")
                self.legal_validator = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalBasisValidator: {e}")
            self.legal_validator = None

        # DocumentProcessor 초기화 (문서 분석용)
        try:
            from core.utils.config import Config as UtilsConfig
            try:
                from core.processing.processors.document_processor import LegalDocumentProcessor
            except ImportError:
                # 호환성을 위한 fallback (더 이상 services에 없음)
                LegalDocumentProcessor = None
            utils_config = UtilsConfig()
            self.document_processor = LegalDocumentProcessor(utils_config)
            self.logger.debug("LegalDocumentProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalDocumentProcessor: {e}")
            self.document_processor = None

        # ConfidenceCalculator 초기화 (신뢰도 계산용)
        try:
            from core.generation.validators.confidence_calculator import (
                ConfidenceCalculator,
            )
            self.confidence_calculator = ConfidenceCalculator()
            self.logger.debug("ConfidenceCalculator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ConfidenceCalculator: {e}")
            self.confidence_calculator = None

        # LLMInitializer 초기화
        try:
            from core.workflow.initializers.llm_initializer import LLMInitializer
            self.llm_initializer = LLMInitializer(config=self.config)
            self.logger.debug("LLMInitializer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLMInitializer: {e}")
            self.llm_initializer = None

        # LLM 초기화 (LLMInitializer 사용)
        self.llm = self._initialize_llm()
        
        # LLMInitializer에 main_llm 설정
        if self.llm_initializer:
            self.llm_initializer.main_llm = self.llm
        
        # 빠른 LLM 초기화 (간단한 질문용)
        self.llm_fast = self._initialize_llm_fast()
        
        # UnifiedPromptManager에 llm_fast 전달 (요약 에이전트용)
        if self.unified_prompt_manager:
            self.unified_prompt_manager.llm_fast = self.llm_fast
            self.logger.debug("UnifiedPromptManager.llm_fast set for document summary agent")
        
        # 품질 검증용 LLM 초기화 (별도)
        self.validator_llm = self._initialize_validator_llm()
        
        # 긴 글/코드 생성용 LLM 초기화 (60초 timeout)
        self.llm_long_text = self._initialize_llm_long_text()

        # DocumentAnalysisProcessor 초기화
        try:
            from core.workflow.processors.document_analysis_processor import DocumentAnalysisProcessor
            self.document_analysis_processor = DocumentAnalysisProcessor(
                llm=self.llm,
                logger=self.logger,
                document_processor=self.document_processor,
                llm_fast=self.llm_fast
            )
            self.logger.debug("DocumentAnalysisProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize DocumentAnalysisProcessor: {e}")
            self.document_analysis_processor = None

        # SearchExecutionProcessor 초기화
        try:
            from core.workflow.processors.search_execution_processor import SearchExecutionProcessor
            self.search_execution_processor = SearchExecutionProcessor(
                search_handler=self.search_handler,
                logger=self.logger,
                config=self.config,
                keyword_search_func=self._keyword_search,
                get_state_value_func=self._get_state_value,
                set_state_value_func=self._set_state_value,
                get_query_type_str_func=self._get_query_type_str,
                determine_search_parameters_func=self._determine_search_parameters,
                save_metadata_safely_func=self._save_metadata_safely,
                update_processing_time_func=self._update_processing_time,
                handle_error_func=self._handle_error,
                semantic_search_engine=self.semantic_search  # 타입별 검색용
            )
            self.logger.debug("SearchExecutionProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SearchExecutionProcessor: {e}")
            self.search_execution_processor = None

        # ContextExpansionProcessor 초기화
        try:
            from core.workflow.processors.context_expansion_processor import ContextExpansionProcessor
            self.context_expansion_processor = ContextExpansionProcessor(
                search_handler=self.search_handler,
                logger=self.logger,
                keyword_search_func=self._keyword_search,
                get_state_value_func=self._get_state_value,
                set_state_value_func=self._set_state_value
            )
            self.logger.debug("ContextExpansionProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ContextExpansionProcessor: {e}")
            self.context_expansion_processor = None

        # AnswerQualityValidator 초기화
        try:
            from core.workflow.validators.answer_quality_validator import AnswerQualityValidator
            self.answer_quality_validator = AnswerQualityValidator(
                logger=self.logger,
                validator_llm=self.validator_llm,
                legal_validator=self.legal_validator,
                workflow_validator=self.workflow_validator,
                get_state_value_func=self._get_state_value,
                set_state_value_func=self._set_state_value,
                normalize_answer_func=self._normalize_answer,
                set_answer_safely_func=self._set_answer_safely,
                add_step_func=self._add_step,
                save_metadata_safely_func=self._save_metadata_safely,
                check_has_sources_func=self._check_has_sources
            )
            self.logger.debug("AnswerQualityValidator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AnswerQualityValidator: {e}")
            self.answer_quality_validator = None

        # 그래프 빌더 초기화 (Phase 4: ModularGraphBuilder 사용, 기존 WorkflowGraphBuilder는 폴백)
        self.use_modular_builder = os.getenv("USE_MODULAR_GRAPH_BUILDER", "true").lower() == "true"
        
        if self.use_modular_builder:
            try:
                from core.workflow.builders.modular_graph_builder import ModularGraphBuilder
                from core.workflow.registry.node_registry import NodeRegistry
                from core.workflow.registry.subgraph_registry import SubgraphRegistry
                from core.workflow.edges.classification_edges import ClassificationEdges
                from core.workflow.edges.search_edges import SearchEdges
                from core.workflow.edges.answer_edges import AnswerEdges
                from core.workflow.edges.agentic_edges import AgenticEdges
                
                # 레지스트리 초기화
                node_registry = NodeRegistry(logger_instance=self.logger)
                subgraph_registry = SubgraphRegistry(logger_instance=self.logger)
                
                # 엣지 빌더 초기화
                classification_edges = ClassificationEdges(
                    route_by_complexity_func=self._route_by_complexity,
                    route_by_complexity_with_agentic_func=self._route_by_complexity_with_agentic,
                    should_analyze_document_func=self._should_analyze_document,
                    logger_instance=self.logger
                )
                search_edges = SearchEdges(
                    should_skip_search_adaptive_func=self._should_skip_search_adaptive,
                    should_use_multi_query_agent_func=self._should_use_multi_query_agent,
                    logger_instance=self.logger
                )
                answer_edges = AnswerEdges(
                    should_retry_validation_func=self._should_retry_validation,
                    should_skip_final_node_func=self._should_skip_final_node,
                    logger_instance=self.logger
                )
                agentic_edges = AgenticEdges(
                    route_after_agentic_func=self._route_after_agentic,
                    logger_instance=self.logger
                )
                
                self.modular_graph_builder = ModularGraphBuilder(
                    config=self.config,
                    logger_instance=self.logger,
                    node_registry=node_registry,
                    subgraph_registry=subgraph_registry,
                    classification_edges=classification_edges,
                    search_edges=search_edges,
                    answer_edges=answer_edges,
                    agentic_edges=agentic_edges
                )
                self.workflow_graph_builder = None  # ModularGraphBuilder 사용
                self.logger.debug("ModularGraphBuilder initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ModularGraphBuilder: {e}, falling back to WorkflowGraphBuilder")
                self.use_modular_builder = False
                self.modular_graph_builder = None
        
        if not self.use_modular_builder:
            # 기존 WorkflowGraphBuilder 사용 (폴백)
            try:
                from core.workflow.builders.workflow_graph_builder import WorkflowGraphBuilder
                self.workflow_graph_builder = WorkflowGraphBuilder(
                    config=self.config,
                    logger=self.logger,
                    route_by_complexity_func=self._route_by_complexity,
                    route_by_complexity_with_agentic_func=self._route_by_complexity_with_agentic,
                    route_after_agentic_func=self._route_after_agentic,
                    should_analyze_document_func=self._should_analyze_document,
                    should_skip_search_adaptive_func=self._should_skip_search_adaptive,
                    should_retry_validation_func=self._should_retry_validation,
                    should_skip_final_node_func=self._should_skip_final_node
                )
                self.logger.debug("WorkflowGraphBuilder initialized (fallback)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize WorkflowGraphBuilder: {e}")
                self.workflow_graph_builder = None

        # QueryEnhancer 초기화 (llm, llm_fast, term_integrator, config 필요)
        self.query_enhancer = QueryEnhancer(
            llm=self.llm,
            llm_fast=self.llm_fast,
            term_integrator=self.term_integrator,
            config=self.config,
            logger=self.logger
        )
        
        # workflow_document_processor의 query_enhancer 설정 (Phase 14 리팩토링)
        if hasattr(self, 'workflow_document_processor'):
            self.workflow_document_processor.query_enhancer = self.query_enhancer

        # HybridQueryProcessor 초기화 (HuggingFace + LLM 하이브리드)
        try:
            from core.search.optimizers.hybrid_query_processor import HybridQueryProcessor
            from langchain_google_genai import ChatGoogleGenerativeAI
            from core.workflow.utils.workflow_constants import WorkflowConstants
            
            embedding_model_name = getattr(self.config, 'embedding_model', None)
            
            # prepare_search_query에서 사용할 LLM: gemini-2.5-flash-lite
            search_query_llm = None
            if self.config.llm_provider == "google":
                try:
                    search_query_llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-lite",
                        temperature=WorkflowConstants.TEMPERATURE,
                        max_output_tokens=WorkflowConstants.MAX_OUTPUT_TOKENS,
                        timeout=WorkflowConstants.TIMEOUT_RAG_QA,
                        api_key=self.config.google_api_key
                    )
                    self.logger.debug("✅ Initialized gemini-2.5-flash-lite LLM for prepare_search_query")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to initialize gemini-2.5-flash-lite LLM: {e}, using main LLM")
                    search_query_llm = self.llm
            else:
                search_query_llm = self.llm
            
            self.hybrid_query_processor = HybridQueryProcessor(
                keyword_extractor=self.keyword_extractor,
                term_integrator=self.term_integrator,
                llm=search_query_llm,
                embedding_model_name=embedding_model_name,
                logger=self.logger
            )
            self.logger.debug("✅ HybridQueryProcessor initialized (HuggingFace + LLM hybrid)")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize HybridQueryProcessor: {e}, using QueryEnhancer")
            self.hybrid_query_processor = None

        # 답변 생성 핸들러 초기화 (Phase 5 리팩토링) - LLM 초기화 이후
        # AnswerGenerator에 이미 초기화된 LLM 전달
        self.answer_generator = AnswerGenerator(
            config=self.config,
            langfuse_client=None,  # langfuse_client는 선택사항
            llm=self.llm  # 이미 초기화된 LLM 전달
        )
        
        # DirectAnswerHandler 초기화 (Phase 10 리팩토링) - LLM 초기화 이후
        try:
            from core.agents.handlers.direct_answer_handler import DirectAnswerHandler
            self.direct_answer_handler = DirectAnswerHandler(
                llm=self.llm,
                llm_fast=self.llm_fast,
                logger=self.logger
            )
            self.logger.debug("DirectAnswerHandler initialized")
        except ImportError:
            try:
                from core.generation.generators.direct_answer_handler import DirectAnswerHandler
                self.direct_answer_handler = DirectAnswerHandler(
                    llm=self.llm,
                    llm_fast=self.llm_fast,
                    logger=self.logger
                )
                self.logger.debug("DirectAnswerHandler initialized (from generators)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DirectAnswerHandler: {e}")
                self.direct_answer_handler = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize DirectAnswerHandler: {e}")
            self.direct_answer_handler = None

        # 워크플로우 라우팅 핸들러 초기화 (Phase 5: 새로운 라우팅 클래스들 사용)
        self.classification_routes = ClassificationRoutes(logger_instance=self.logger)
        self.search_routes = SearchRoutes(logger_instance=self.logger)
        self.answer_routes = AnswerRoutes(
            retry_manager=self.retry_manager,
            answer_generator=self.answer_generator,
            logger_instance=self.logger
        )
        self.agentic_routes = AgenticRoutes(logger_instance=self.logger)
        
        # 호환성을 위해 기존 WorkflowRoutes도 유지 (래퍼 메서드용)
        self.workflow_routes = WorkflowRoutes(
            retry_manager=self.retry_manager,
            answer_generator=self.answer_generator,
            ai_keyword_generator=self.ai_keyword_generator,
            logger=self.logger
        )

        # 답변 포맷팅 핸들러 초기화 (Phase 4 리팩토링) - 필요한 의존성 초기화 이후
        self.answer_formatter_handler = AnswerFormatterHandler(
            keyword_mapper=self.keyword_mapper,
            answer_structure_enhancer=self.answer_structure_enhancer,
            answer_formatter=self.answer_formatter,
            confidence_calculator=self.confidence_calculator,
            reasoning_extractor=self.reasoning_extractor,
            answer_generator=self.answer_generator,
            logger=self.logger
        )

        # 복잡도 분류 캐시 초기화
        self._complexity_cache: Dict[str, Tuple[QueryComplexity, bool]] = {}

        # 통합 분류 캐시 초기화 (질문 유형 + 복잡도 동시 분류)
        self._classification_cache: Dict[str, Tuple[QuestionType, float, QueryComplexity, bool]] = {}
        
        # ClassificationHandler 지연 로딩을 위한 초기화
        self._classification_handler = None
        self._classification_handler_initialized = False


        # Agentic AI Tool 시스템 초기화 (Tool Use/Function Calling)
        # lawfirm_langgraph 구조 사용 (core.workflow.tools)
        if self.config.use_agentic_mode:
            try:
                from core.workflow.tools import LEGAL_TOOLS
                self.legal_tools = LEGAL_TOOLS
                self.agentic_agent = None  # 지연 초기화
                self.logger.debug(f"Agentic AI mode enabled with {len(LEGAL_TOOLS)} tools (from core.workflow.tools)")
            except ImportError as e:
                self.logger.warning(f"Failed to import legal tools from core.workflow.tools: {e}. Agentic mode disabled.")
                self.legal_tools = []
                self.agentic_agent = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize Agentic tools: {e}. Agentic mode disabled.")
                self.legal_tools = []
                self.agentic_agent = None
        else:
            self.legal_tools = []
            self.agentic_agent = None

        # WorkflowStatistics 초기화 (통계 보장)
        try:
            from core.workflow.utils.workflow_statistics import WorkflowStatistics
            self.workflow_statistics = WorkflowStatistics(
                enable_statistics=self.config.enable_statistics
            )
            self.stats = self.workflow_statistics.stats
            if self.stats is None and self.config.enable_statistics:
                # 통계가 활성화되었지만 초기화 실패한 경우, 기본 통계 생성
                self.logger.warning("WorkflowStatistics.stats is None, creating fallback stats")
                self.stats = self.workflow_statistics._initialize_statistics()
            self.logger.debug(f"WorkflowStatistics initialized (enabled: {self.config.enable_statistics})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WorkflowStatistics: {e}")
            self.workflow_statistics = None
            # 통계가 활성화된 경우 기본 통계 생성
            if self.config.enable_statistics:
                try:
                    from core.workflow.utils.workflow_statistics import WorkflowStatistics
                    temp_stats = WorkflowStatistics(enable_statistics=True)
                    self.stats = temp_stats._initialize_statistics()
                    self.logger.debug("Created fallback statistics dictionary")
                except Exception:
                    self.stats = {
                        'total_queries': 0,
                        'total_documents_retrieved': 0,
                        'avg_response_time': 0.0,
                        'avg_confidence': 0.0,
                        'total_errors': 0,
                        'llm_complexity_classifications': 0,
                        'complexity_cache_hits': 0,
                        'complexity_cache_misses': 0,
                        'complexity_fallback_count': 0,
                        'avg_complexity_classification_time': 0.0,
                        'unified_classification_calls': 0,
                        'unified_classification_llm_calls': 0,
                        'avg_unified_classification_time': 0.0,
                        'total_unified_classification_time': 0.0
                    }
                    self.logger.warning("Created minimal fallback statistics dictionary")
            else:
                self.stats = None

        # ConversationProcessor 초기화
        try:
            from core.workflow.processors.conversation_processor import ConversationProcessor
            self.conversation_processor = ConversationProcessor(
                logger=self.logger,
                emotion_analyzer=self.emotion_analyzer,
                multi_turn_handler=self.multi_turn_handler,
                conversation_manager=self.conversation_manager,
                llm=self.llm,
                get_state_value_func=self._get_state_value,
                set_state_value_func=self._set_state_value,
                update_processing_time_func=self._update_processing_time,
                handle_error_func=self._handle_error
            )
            self.logger.debug("ConversationProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ConversationProcessor: {e}")
            self.conversation_processor = None

        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.debug("EnhancedLegalQuestionWorkflow initialized with UnifiedPromptManager.")

    def _initialize_llm(self):
        """LLMInitializer.initialize_llm 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_llm()
        return self._create_mock_llm_fallback()

    def _initialize_gemini(self):
        """LLMInitializer.initialize_gemini 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_gemini()
        return self._create_mock_llm_fallback()


    def _create_mock_llm(self):
        """LLMInitializer.create_mock_llm 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.create_mock_llm()
        return self._create_mock_llm_fallback()

    def _initialize_llm_fast(self):
        """LLMInitializer.initialize_llm_fast 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_llm_fast()
        return self.llm

    def _initialize_validator_llm(self):
        """LLMInitializer.initialize_validator_llm 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_validator_llm()
        return self.llm
    
    def _initialize_llm_long_text(self) -> Any:
        """긴 글/코드 생성용 LLM 초기화 (60초 timeout)"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_gemini_long_text()
        return self.llm

    def _create_mock_llm_fallback(self):
        """Mock LLM 생성 (폴백)"""
        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response for: " + prompt
            async def ainvoke(self, prompt):
                return "Mock LLM async response for: " + prompt
        return MockLLM()

    def _build_graph(self) -> StateGraph:
        """그래프 빌더를 사용하여 그래프 구축 (ModularGraphBuilder 우선)"""
        if self.use_modular_builder and self.modular_graph_builder:
            # ModularGraphBuilder 사용
            from core.workflow.nodes.classification_nodes import ClassificationNodes
            from core.workflow.nodes.search_nodes import SearchNodes
            from core.workflow.nodes.document_nodes import DocumentNodes
            from core.workflow.nodes.answer_nodes import AnswerNodes
            from core.workflow.nodes.agentic_nodes import AgenticNodes
            from core.workflow.nodes.ethical_rejection_node import EthicalRejectionNode
            
            # 노드 클래스 인스턴스 생성
            classification_nodes = ClassificationNodes(workflow_instance=self, logger_instance=self.logger)
            search_nodes = SearchNodes(workflow_instance=self, logger_instance=self.logger)
            document_nodes = DocumentNodes(workflow_instance=self, logger_instance=self.logger)
            answer_nodes = AnswerNodes(workflow_instance=self, logger_instance=self.logger)
            agentic_nodes = AgenticNodes(workflow_instance=self, logger_instance=self.logger)
            from core.workflow.nodes.multi_query_search_agent import MultiQuerySearchAgentNode
            multi_query_search_agent = MultiQuerySearchAgentNode(
                workflow_instance=self,
                logger_instance=self.logger
            )
            
            # 노드 레지스트리에 등록
            node_registry = self.modular_graph_builder.node_registry
            node_registry.register("classify_query_and_complexity", classification_nodes.classify_query_and_complexity)
            node_registry.register("direct_answer_node", classification_nodes.direct_answer)
            node_registry.register("classification_parallel", classification_nodes.classification_parallel)
            node_registry.register("assess_urgency", classification_nodes.assess_urgency)
            node_registry.register("resolve_multi_turn", classification_nodes.resolve_multi_turn)
            node_registry.register("route_expert", classification_nodes.route_expert)
            node_registry.register("analyze_document", document_nodes.analyze_document)
            node_registry.register("expand_keywords", search_nodes.expand_keywords)
            node_registry.register("prepare_search_query", search_nodes.prepare_search_query)
            node_registry.register("execute_searches_parallel", search_nodes.execute_searches_parallel)
            node_registry.register("process_search_results_combined", search_nodes.process_search_results_combined)
            node_registry.register("multi_query_search_agent", multi_query_search_agent.execute)
            node_registry.register("prepare_documents_and_terms", document_nodes.prepare_documents_and_terms)
            node_registry.register("generate_and_validate_answer", answer_nodes.generate_and_validate_answer)
            node_registry.register("continue_answer_generation", answer_nodes.continue_answer_generation)
            node_registry.register("ethical_rejection", EthicalRejectionNode.ethical_rejection_node)
            
            # 스트리밍 노드 추가
            if hasattr(self, 'generate_answer_stream'):
                node_registry.register("generate_answer_stream", answer_nodes.generate_answer_stream)
            if hasattr(self, 'generate_answer_final'):
                node_registry.register("generate_answer_final", answer_nodes.generate_answer_final)
            
            # Agentic 노드 추가
            if self.config.use_agentic_mode:
                node_registry.register("agentic_decision", agentic_nodes.agentic_decision_node)
            
            # 서브그래프 등록 (Phase 3: 서브그래프를 메인 그래프에 통합)
            from core.workflow.subgraphs.classification_subgraph import ClassificationSubgraph
            from core.workflow.subgraphs.search_subgraph import SearchSubgraph
            from core.workflow.subgraphs.answer_generation_subgraph import AnswerGenerationSubgraph
            
            subgraph_registry = self.modular_graph_builder.subgraph_registry
            
            # 분류 서브그래프 등록
            classification_subgraph = ClassificationSubgraph(
                workflow_instance=self,
                logger_instance=self.logger
            )
            subgraph_registry.register("classification_subgraph", classification_subgraph.build_subgraph())
            
            # 검색 서브그래프 등록
            search_subgraph = SearchSubgraph(
                workflow_instance=self,
                logger_instance=self.logger
            )
            subgraph_registry.register("search_subgraph", search_subgraph.build_subgraph())
            
            # 답변 생성 서브그래프 등록
            answer_subgraph = AnswerGenerationSubgraph(
                workflow_instance=self,
                logger_instance=self.logger
            )
            subgraph_registry.register("answer_generation_subgraph", answer_subgraph.build_subgraph())
            
            return self.modular_graph_builder.build_graph()
        
        elif self.workflow_graph_builder:
            # 기존 WorkflowGraphBuilder 사용 (폴백)
            node_handlers = {
                "generate_answer_stream": self.generate_answer_stream,
                "generate_answer_final": self.generate_answer_final,
                "classify_query_and_complexity": self.classify_query_and_complexity,
                "direct_answer_node": self.direct_answer_node,
                "classification_parallel": self.classification_parallel,
                "assess_urgency": self.assess_urgency,
                "resolve_multi_turn": self.resolve_multi_turn,
                "route_expert": self.route_expert,
                "analyze_document": self.analyze_document,
                "expand_keywords": self.expand_keywords,
                "prepare_search_query": self.prepare_search_query,
                "execute_searches_parallel": self.execute_searches_parallel,
                "process_search_results_combined": self.process_search_results_combined,
                "prepare_documents_and_terms": self.prepare_documents_and_terms,
                "generate_and_validate_answer": self.generate_and_validate_answer,
                "continue_answer_generation": self.continue_answer_generation,
                "agentic_decision_node": self.agentic_decision_node,
                "multi_query_search_agent": self.multi_query_search_agent_node
            }
            return self.workflow_graph_builder.build_graph(node_handlers)
        
        # 기본 그래프 반환
        from langgraph.graph import StateGraph
        return StateGraph(LegalWorkflowState)

    @with_state_optimization("expand_keywords", enable_reduction=False)
    def expand_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 추출 노드 (HuggingFace 모델 기반)"""
        try:
            start_time = time.time()
            
            # 1. State 초기화 및 보존
            self._preserve_and_update_metadata(state, "expand_keywords")
            
            # 2. 키워드 추출 (이미 있으면 스킵)
            keywords = self._get_state_value(state, "extracted_keywords", [])
            if not keywords:
                keywords = self._extract_keywords_with_fallback(state)
                self._save_keywords_to_state(state, keywords)
                self.logger.debug(
                    f"✅ [HF KEYWORD EXTRACTION] Final extracted {len(keywords)} keywords "
                    f"(HF: {len([k for k in keywords if k])}, mapper: {len([k for k in keywords if k])})"
                )
            
            # 3. AI 키워드 확장 비활성화 (prepare_search_query의 HybridQueryProcessor에서 처리)
            # HybridQueryProcessor의 LegalKeywordExpander가 HuggingFace 모델을 사용하여 키워드 확장을 수행하므로
            # 여기서 LLM 기반 확장은 중복이며, 504 Deadline Exceeded 에러를 방지하기 위해 비활성화
            self.logger.debug("🔍 [KEYWORD EXPANSION] LLM-based expansion disabled (using HybridQueryProcessor in prepare_search_query instead)")

            self._save_metadata_safely(state, "_last_executed_node", "expand_keywords")
            self._update_processing_time(state, start_time)
            self._add_step(state, "키워드 추출", f"키워드 추출 완료: {len(self._get_state_value(state, 'extracted_keywords', []))}개 (HuggingFace 모델 사용)")

        except Exception as e:
            self._handle_error(state, str(e), "키워드 추출 중 오류 발생")

        return state
    
    # ========== 키워드 추출 헬퍼 메서드들 ==========
    
    def _preserve_and_update_metadata(self, state: LegalWorkflowState, node_name: str) -> None:
        """metadata 보존 및 업데이트"""
        preserved_complexity = state.get("metadata", {}).get("query_complexity") if isinstance(state.get("metadata"), dict) else None
        preserved_needs_search = state.get("metadata", {}).get("needs_search") if isinstance(state.get("metadata"), dict) else None

        if "metadata" not in state or not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        state["metadata"] = dict(state["metadata"])
        if preserved_complexity:
            state["metadata"]["query_complexity"] = preserved_complexity
        if preserved_needs_search is not None:
            state["metadata"]["needs_search"] = preserved_needs_search
        state["metadata"]["_last_executed_node"] = node_name

        if "common" not in state or not isinstance(state.get("common"), dict):
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"]["_last_executed_node"] = node_name
    
    def _extract_keywords_with_fallback(self, state: LegalWorkflowState) -> List[str]:
        """키워드 추출 (다단계 폴백)"""
        query = self._get_state_value(state, "query", "")
        query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", "general_question"))
        legal_field = self._get_state_value(state, "legal_field", "")
        
        # 방법 1: HybridQueryProcessor 사용
        extracted_keywords = self._try_extract_with_hybrid_processor(query, query_type_str, legal_field)
        if extracted_keywords:
            mapper_keywords = self._get_mapper_keywords(query, query_type_str)
            return self._merge_and_clean_keywords(extracted_keywords, mapper_keywords)
        
        # 방법 2: Standalone LegalQueryAnalyzer 사용
        extracted_keywords = self._try_extract_with_standalone_analyzer(query, query_type_str, legal_field)
        if extracted_keywords:
            mapper_keywords = self._get_mapper_keywords(query, query_type_str)
            return self._merge_and_clean_keywords(extracted_keywords, mapper_keywords)
        
        # 방법 3: 최종 폴백 (정규식 + KoNLPy)
        extracted_keywords = self._extract_keywords_fallback(query, query_type_str, legal_field)
        mapper_keywords = self._get_mapper_keywords(query, query_type_str)
        return self._merge_and_clean_keywords(extracted_keywords, mapper_keywords)
    
    def _try_extract_with_hybrid_processor(
        self, query: str, query_type_str: str, legal_field: str
    ) -> List[str]:
        """HybridQueryProcessor를 사용한 키워드 추출 시도"""
        if not (self.hybrid_query_processor and hasattr(self.hybrid_query_processor, 'query_analyzer')):
            return []
        
        try:
            self.logger.debug("🔍 [HF KEYWORD EXTRACTION] Using LegalQueryAnalyzer")
            analysis_result = self.hybrid_query_processor.query_analyzer.analyze_query(
                query=query, query_type=query_type_str, legal_field=legal_field
            )
            keywords = self._process_analysis_result(analysis_result)
            self.logger.debug(
                f"✅ [HF KEYWORD EXTRACTION] Extracted {len(keywords)} keywords using LegalQueryAnalyzer "
                f"(core: {len(analysis_result.get('core_keywords', []))}, concepts: {len(analysis_result.get('key_concepts', []))})"
            )
            return keywords
        except Exception as e:
            self.logger.warning(f"⚠️ [HF KEYWORD EXTRACTION] LegalQueryAnalyzer failed: {e}, using fallback method", exc_info=True)
            return []
    
    def _try_extract_with_standalone_analyzer(
        self, query: str, query_type_str: str, legal_field: str
    ) -> List[str]:
        """Standalone LegalQueryAnalyzer를 사용한 키워드 추출 시도"""
        try:
            from core.search.optimizers.legal_query_analyzer import LegalQueryAnalyzer
            embedding_model_name = getattr(self.config, 'embedding_model', None)
            
            analyzer = LegalQueryAnalyzer(
                keyword_extractor=None,
                embedding_model_name=embedding_model_name,
                logger=self.logger
            )
            
            self.logger.debug("🔍 [HF KEYWORD EXTRACTION] Using standalone LegalQueryAnalyzer")
            analysis_result = analyzer.analyze_query(
                query=query, query_type=query_type_str, legal_field=legal_field
            )
            keywords = self._process_analysis_result(analysis_result)
            self.logger.debug(f"✅ [HF KEYWORD EXTRACTION] Extracted {len(keywords)} keywords using standalone LegalQueryAnalyzer")
            return keywords
        except Exception as e:
            self.logger.warning(f"⚠️ [HF KEYWORD EXTRACTION] Standalone analyzer failed: {e}, using simple regex fallback", exc_info=True)
            return []
    
    def _extract_keywords_fallback(
        self, query: str, query_type_str: str, legal_field: str
    ) -> List[str]:
        """최종 폴백: 정규식 기반 키워드 추출 + KoNLPy 불용어 제거"""
        import re
        words = re.findall(r'[가-힣]+', query)
        extracted_keywords = [w for w in words if len(w) >= self.MIN_KEYWORD_LENGTH]
        
        # KoNLPy 기반 불용어 제거
        if hasattr(self, 'stopword_processor') and self.stopword_processor:
            extracted_keywords = self.stopword_processor.filter_stopwords(extracted_keywords)
            self.logger.debug(f"✅ [HF KEYWORD EXTRACTION] KoNLPy로 불용어 제거 완료: {len(extracted_keywords)}개 키워드")
        else:
            # 폴백: 기본 불용어 제거
            basic_stopwords = self._get_basic_stopwords()
            extracted_keywords = [kw for kw in extracted_keywords if kw not in basic_stopwords]
            self.logger.debug(f"⚠️ [HF KEYWORD EXTRACTION] 기본 불용어 제거 사용 (KoreanStopwordProcessor 없음): {len(extracted_keywords)}개 키워드")
        
        # 쿼리 타입/법률 분야 기반 키워드 추가
        extracted_keywords.extend(self._get_query_type_keywords(query_type_str))
        extracted_keywords.extend(self._get_field_keywords(legal_field))
        
        # 정리
        extracted_keywords = list(set(extracted_keywords))
        extracted_keywords = [
            kw for kw in extracted_keywords 
            if kw and len(kw.strip()) >= self.MIN_KEYWORD_LENGTH
        ][:self.MAX_FALLBACK_KEYWORDS]
        
        if extracted_keywords:
            self.logger.debug(f"✅ [HF KEYWORD EXTRACTION] 폴백으로 {len(extracted_keywords)}개 키워드 추출: {extracted_keywords[:5]}")
        
        return extracted_keywords
    
    def _process_analysis_result(self, analysis_result: Dict[str, Any]) -> List[str]:
        """분석 결과에서 키워드 추출 및 정리"""
        extracted_keywords = analysis_result.get("core_keywords", [])
        key_concepts = analysis_result.get("key_concepts", [])
        
        all_keywords = list(set(extracted_keywords + key_concepts))
        return [kw for kw in all_keywords if isinstance(kw, str) and len(kw.strip()) >= self.MIN_KEYWORD_LENGTH]
    
    def _get_mapper_keywords(self, query: str, query_type_str: str) -> List[str]:
        """keyword_mapper를 사용한 키워드 추출"""
        if not self.keyword_mapper:
            return []
        
        try:
            mapper_keywords = self.keyword_mapper.get_keywords_for_question(query, query_type_str)
            return [
                kw for kw in mapper_keywords 
                if isinstance(kw, (str, int, float, tuple)) and kw is not None
            ]
        except Exception as e:
            self.logger.debug(f"keyword_mapper failed: {e}")
            return []
    
    def _merge_and_clean_keywords(
        self, extracted_keywords: List[str], mapper_keywords: List[str]
    ) -> List[str]:
        """키워드 통합 및 정리"""
        keywords = list(set(extracted_keywords + mapper_keywords))
        return [kw for kw in keywords if isinstance(kw, str) and len(kw.strip()) >= self.MIN_KEYWORD_LENGTH]
    
    def _save_keywords_to_state(
        self, state: LegalWorkflowState, keywords: List[str]
    ) -> None:
        """키워드를 state의 여러 위치에 저장 (TASK 5: 검증 강화)"""
        if not keywords:
            self.logger.warning("⚠️ [KEYWORD SAVE] 키워드가 비어있음")
            return
        
        # 여러 위치에 저장 (보장)
        self._set_state_value(state, "extracted_keywords", keywords)
        
        if "search" not in state:
            state["search"] = {}
        if not isinstance(state["search"], dict):
            state["search"] = {}
        state["search"]["extracted_keywords"] = keywords
        
        if "common" not in state:
            state["common"] = {}
        if "search" not in state["common"]:
            state["common"]["search"] = {}
        state["common"]["search"]["extracted_keywords"] = keywords
        
        state["extracted_keywords"] = keywords
        
        # TASK 5: 저장 확인
        saved = self._get_state_value(state, "extracted_keywords", [])
        if not saved:
            self.logger.error("❌ [KEYWORD SAVE] 키워드 저장 실패")
        else:
            self.logger.info(f"✅ [KEYWORD SAVE] 키워드 저장 완료: {len(saved)}개")
    
    def _get_query_type_keywords(self, query_type_str: str) -> List[str]:
        """쿼리 타입별 기본 키워드 반환"""
        return self.QUERY_TYPE_KEYWORDS.get(query_type_str, [])
    
    def _get_field_keywords(self, legal_field: str) -> List[str]:
        """법률 분야별 기본 키워드 반환"""
        return self.FIELD_KEYWORDS.get(legal_field, [])
    
    def _get_basic_stopwords(self) -> Set[str]:
        """기본 불용어 리스트 반환 (KoreanStopwordProcessor 폴백용)"""
        return {
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로',
            '에서', '에게', '한테', '께', '무엇', '어떤', '어떻게', '언제', '어디', '누구',
            '시', '할', '하는', '된', '되는', '이다', '입니다', '있습니다', '합니다', '주세요',
            '알려주세요', '설명해주세요', '알려주시', '설명해주시'
        }

    def _should_retry_generation(self, state: LegalWorkflowState) -> str:
        """AnswerRoutes.should_retry_generation 사용 (호환성 유지)"""
        # 기존 WorkflowRoutes의 메서드 사용 (새로운 AnswerRoutes에는 아직 없음)
        return self.workflow_routes.should_retry_generation(state)

    def _should_skip_final_node(self, state: LegalWorkflowState) -> str:
        """AnswerRoutes.should_skip_final_node 사용"""
        return self.answer_routes.should_skip_final_node(state)

    def _should_retry_validation(self, state: LegalWorkflowState) -> str:
        """AnswerRoutes.should_retry_validation 사용"""
        return self.answer_routes.should_retry_validation(state, answer_generator=self.answer_generator)


    @with_state_optimization("agentic_decision", enable_reduction=False)
    def agentic_decision_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """
        Agentic AI 노드: LLM이 Tool을 자동으로 선택하고 실행
        
        LangChain AgentExecutor를 사용하여 LLM이 질문을 분석하고
        필요한 검색 Tool만 선택적으로 실행합니다.
        """
        try:
            start_time = time.time()
            self._save_metadata_safely(state, "_last_executed_node", "agentic_decision")
            
            # Agentic 모드가 비활성화되어 있으면 기존 플로우로 진행
            if not self.config.use_agentic_mode or not self.legal_tools:
                self.logger.warning("Agentic mode disabled or no tools available. Skipping agentic decision.")
                # 기존 검색 노드로 라우팅하기 위해 빈 검색 결과 설정
                state.setdefault("search", {})["results"] = []
                return state
            
            query = self._get_state_value(state, "query", "")
            if not query:
                query = state.get("input", {}).get("query", "") if state.get("input") else ""
            
            if not query:
                self.logger.error("No query found in state for agentic decision")
                return state
            
            self.logger.debug(f"🤖 [AGENTIC] Processing query with {len(self.legal_tools)} tools: {query[:100]}")
            
            # AgentExecutor 초기화 (지연 초기화)
            if self.agentic_agent is None:
                try:
                    from langchain.agents import AgentExecutor, create_openai_tools_agent
                    try:
                        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
                    except ImportError:
                        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # pyright: ignore[reportMissingImports]
                    
                    # Agent 프롬프트 생성
                    agent_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 전문 법률 AI 어시스턴트입니다.

당신의 역할:
1. 사용자의 법률 질문을 분석하고 필요한 정보를 파악
2. 적절한 도구(Tool)를 선택하여 법률 정보 검색
3. 검색 결과를 바탕으로 정확하고 도움이 되는 답변 제공
4. 법적 근거(판례, 법령)를 명확히 제시

사용 가능한 도구:
- search_precedent: 판례 검색 - 관련 판례 검색
- search_law: 법령 검색 - 법령 조문 검색  
- search_legal_term: 법률 용어 검색 - 법률 용어 정의 검색
- hybrid_search: 통합 검색 - 법령, 판례 등을 종합적으로 검색

중요 원칙:
- 정확한 법적 근거 제시 필수
- 판단 불가능한 경우 전문가 상담 권장
- 사용자에게 명확하고 이해하기 쉬운 답변 제공
- 필요한 도구만 선택적으로 사용 (불필요한 검색은 피함)
"""),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    # Agent 생성
                    agent = create_openai_tools_agent(self.llm, self.legal_tools, agent_prompt)
                    self.agentic_agent = AgentExecutor(
                        agent=agent,
                        tools=self.legal_tools,
                        verbose=True,
                        max_iterations=5,
                        max_execution_time=30,  # 최대 30초
                        handle_parsing_errors=True,
                        return_intermediate_steps=True
                    )
                    self.logger.debug("Agentic agent initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize agentic agent: {e}")
                    # 기존 플로우로 fallback
                    state.setdefault("search", {})["results"] = []
                    return state
            
            # Agent 실행
            try:
                # 대화 맥락 가져오기
                session_id = self._get_state_value(state, "session_id", "")
                chat_history = []
                
                if session_id:
                    conversation_context = self._get_or_create_conversation_context(session_id)
                    if conversation_context:
                        chat_history = self._convert_conversation_context_to_messages(
                            conversation_context, 
                            max_turns=5,
                            current_query=query,
                            use_relevance=True
                        )
                        self.logger.debug(f"Loaded {len(chat_history)} messages from conversation context (relevance-based)")
                
                result = self.agentic_agent.invoke({
                    "input": query,
                    "chat_history": chat_history
                })
                
                # Agent 실행 결과에서 Tool 호출 정보 추출
                tool_calls = []
                search_results = []
                
                if "intermediate_steps" in result:
                    for step in result["intermediate_steps"]:
                        action, observation = step
                        tool_name = action.tool if hasattr(action, 'tool') else str(action)
                        tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
                        
                        tool_calls.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": observation
                        })
                        
                        # 검색 Tool인 경우 결과 파싱
                        if tool_name in ["search_precedent", "search_law", "search_legal_term", "hybrid_search"]:
                            try:
                                import json
                                if isinstance(observation, str):
                                    tool_result = json.loads(observation)
                                    if tool_result.get("success") and tool_result.get("results"):
                                        search_results.extend(tool_result["results"])
                            except Exception as e:
                                self.logger.warning(f"Failed to parse tool result: {e}")
                
                # Agent의 최종 답변
                agent_answer = result.get("output", "")
                
                # 검색 결과가 있으면 state에 저장 (기존 구조와 호환)
                if search_results:
                    # 중복 제거 (같은 문서 ID 기준)
                    seen_ids = set()
                    unique_results = []
                    for result in search_results:
                        doc_id = result.get("metadata", {}).get("id") or result.get("source", "")
                        if doc_id and doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            unique_results.append(result)
                    
                    state.setdefault("search", {})["results"] = unique_results[:10]  # 상위 10개만
                    state.setdefault("search", {})["total_results"] = len(unique_results)
                    self.logger.debug(f"✅ [AGENTIC] Retrieved {len(unique_results)} documents from tool execution")
                else:
                    state.setdefault("search", {})["results"] = []
                    state.setdefault("search", {})["total_results"] = 0
                    self.logger.warning("⚠️ [AGENTIC] No search results from tool execution")
                
                # Tool 호출 정보 저장
                self._save_metadata_safely(state, "agentic_tool_calls", tool_calls)
                self._save_metadata_safely(state, "agentic_answer", agent_answer)
                
                # 검색 결과가 있으면 prepare_documents_and_terms로, 없으면 generate_and_validate_answer로
                # 라우팅은 조건부 엣지에서 처리
                
            except Exception as e:
                self.logger.error(f"Agentic agent execution failed: {e}")
                # 기존 플로우로 fallback
                state.setdefault("search", {})["results"] = []
                return state
            
            processing_time = time.time() - start_time
            self._update_processing_time(state, start_time)
            self._add_step(state, "Agentic 검색", f"{len(search_results)}개 문서 검색, {len(tool_calls)}개 도구 사용")
            
            self.logger.debug(f"✅ [AGENTIC] Completed in {processing_time:.2f}s, {len(search_results)} results, {len(tool_calls)} tools used")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in agentic_decision_node: {e}")
            self._handle_error(state, str(e), "Agentic 검색 중 오류")
            # 에러 시 기존 플로우로 fallback
            state.setdefault("search", {})["results"] = []
            return state

    @with_state_optimization("multi_query_search_agent", enable_reduction=False)
    def multi_query_search_agent_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """
        멀티 질의 + 에이전트 기반 검색 노드
        
        원본 질문을 여러 하위 질문으로 분해하고, 에이전트가 적절한 검색 전략을 선택하여
        PostgreSQL 키워드 검색과 벡터 인덱스 검색을 수행합니다.
        """
        try:
            start_time = time.time()
            self._save_metadata_safely(state, "_last_executed_node", "multi_query_search_agent")
            
            # 멀티 질의 검색 에이전트 인스턴스 가져오기 (지연 초기화)
            if not hasattr(self, '_multi_query_search_agent_instance'):
                from core.workflow.nodes.multi_query_search_agent import MultiQuerySearchAgentNode
                self._multi_query_search_agent_instance = MultiQuerySearchAgentNode(
                    workflow_instance=self,
                    logger_instance=self.logger
                )
            
            # 에이전트 실행
            state = self._multi_query_search_agent_instance.execute(state)
            
            processing_time = time.time() - start_time
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            self._update_processing_time(state, start_time)
            self._add_step(state, "멀티 질의 검색", f"{len(retrieved_docs)}개 문서 검색 완료 (시간: {processing_time:.2f}s)")
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ [MULTI-QUERY-AGENT] Error: {e}", exc_info=True)
            self._handle_error(state, str(e), "멀티 질의 검색 중 오류 발생")
            state.setdefault("search", {})["results"] = []
            state.setdefault("retrieved_docs", [])
            return state

    def _restore_state_data_for_final(self, state: LegalWorkflowState) -> None:
        """최종 노드를 위한 State 데이터 복원"""
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        if not retrieved_docs:
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache and _global_search_results_cache.get("retrieved_docs"):
                    retrieved_docs = _global_search_results_cache["retrieved_docs"]
                    self._set_state_value(state, "retrieved_docs", retrieved_docs)
                    self.logger.debug(f"✅ [FINAL NODE] Restored retrieved_docs from global cache: {len(retrieved_docs)} docs")
            except (ImportError, AttributeError):
                pass
        
        structured_docs = self._get_state_value(state, "structured_documents", [])
        if not structured_docs and retrieved_docs:
            structured_docs = retrieved_docs
            self._set_state_value(state, "structured_documents", structured_docs)
        
        query_type = self._get_state_value(state, "query_type") or (state.get("metadata", {}).get("query_type") if isinstance(state.get("metadata"), dict) else None)
        if not query_type:
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache and _global_search_results_cache.get("query_type"):
                    query_type = _global_search_results_cache["query_type"]
                    metadata = self._get_metadata_safely(state)
                    metadata["query_type"] = query_type
                    self._set_state_value(state, "metadata", metadata)
                    state["query_type"] = query_type
                    self.logger.debug(f"✅ [FINAL NODE] Restored query_type from global cache: {query_type}")
            except (ImportError, AttributeError):
                pass
    
    def _validate_and_handle_regeneration(self, state: LegalWorkflowState) -> bool:
        """품질 검증 및 재생성 처리 (성능 최적화: 불필요한 검증 스킵)"""
        # 성능 최적화: needs_regeneration이 없으면 검증 스킵
        needs_regeneration_from_helper = self._get_state_value(state, "needs_regeneration", False)
        needs_regeneration_from_top = state.get("needs_regeneration", False)
        needs_regeneration_from_metadata = state.get("metadata", {}).get("needs_regeneration", False) if isinstance(state.get("metadata"), dict) else False
        needs_regeneration = needs_regeneration_from_helper or needs_regeneration_from_top or needs_regeneration_from_metadata
        
        # needs_regeneration이 없으면 빠른 경로 (검증 스킵)
        if not needs_regeneration:
            # 이미 검증이 완료되었는지 확인
            quality_check_passed = state.get("metadata", {}).get("quality_check_passed", True) if isinstance(state.get("metadata"), dict) else True
            if quality_check_passed:
                return True
        
        # needs_regeneration이 있거나 quality_check_passed가 False인 경우에만 검증 수행
        quality_check_passed = self._validate_answer_quality_internal(state)
        
        regeneration_reason = (
            self._get_state_value(state, "regeneration_reason") or
            state.get("regeneration_reason") or
            (state.get("metadata", {}).get("regeneration_reason") if isinstance(state.get("metadata"), dict) else None) or
            "unknown"
        )
        
        if needs_regeneration:
            can_retry = self.retry_manager.should_allow_retry(state, "generation")
            retry_counts = self.retry_manager.get_retry_counts(state)
            self.logger.debug(
                f"🔄 [REGENERATION CHECK] needs_regeneration={needs_regeneration}, "
                f"can_retry={can_retry}, reason={regeneration_reason}, "
                f"retry_count={retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES}"
            )
            if can_retry:
                self.logger.warning(
                    f"🔄 [AUTO RETRY] Regeneration needed: {regeneration_reason}. "
                    f"Retrying answer generation "
                    f"(retry count: {retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES})"
                )
                self.retry_manager.increment_retry_count(state, "generation")
                try:
                    state = self.generate_answer_enhanced(state)
                    quality_check_passed = self._validate_answer_quality_internal(state)
                    self._set_state_value(state, "needs_regeneration", False)
                except asyncio.CancelledError:
                    self.logger.warning("⚠️ [REGENERATION] Answer generation was cancelled. Using existing answer.")
                    existing_answer = self._get_state_value(state, "answer", "")
                    if existing_answer and len(str(existing_answer).strip()) > 10:
                        quality_check_passed = True
                        self._set_state_value(state, "needs_regeneration", False)
                    else:
                        quality_check_passed = False
                except Exception as regen_error:
                    self.logger.error(f"⚠️ [REGENERATION] Error during regeneration: {regen_error}", exc_info=True)
                    existing_answer = self._get_state_value(state, "answer", "")
                    if existing_answer and len(str(existing_answer).strip()) > 10:
                        quality_check_passed = True
                        self._set_state_value(state, "needs_regeneration", False)
                    else:
                        quality_check_passed = False
        
        return quality_check_passed
    
    def _handle_format_errors(self, state: LegalWorkflowState, quality_check_passed: bool) -> bool:
        """형식 오류 처리 (성능 최적화: 빠른 정규화 시도)"""
        answer = self._get_state_value(state, "answer", "")
        if not answer or len(answer.strip()) < 10:
            return quality_check_passed
        
        # 성능 최적화: 정규화만 먼저 시도 (비용이 낮음)
        normalized_answer = self._normalize_answer(answer)
        if normalized_answer != answer:
            self._set_answer_safely(state, normalized_answer)
            answer = normalized_answer
        
        # 정규화 후에도 형식 오류가 있는지 확인 (비용이 높은 검증은 마지막에)
        has_format_errors = self._detect_format_errors(answer)
        if has_format_errors and self.retry_manager.should_allow_retry(state, "generation"):
            retry_counts = self.retry_manager.get_retry_counts(state)
            # 성능 최적화: 재시도 횟수가 많으면 정규화만 수행하고 재생성 스킵
            if retry_counts['generation'] >= RetryConfig.MAX_GENERATION_RETRIES - 1:
                self.logger.warning("⚠️ [FORMAT ERRORS] Max retries reached, skipping regeneration")
                return quality_check_passed
            
            self.logger.warning(
                f"🔄 [AUTO RETRY] Format errors detected. Retrying answer generation "
                f"(retry count: {retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES})"
            )
            
            if self._detect_format_errors(normalized_answer):
                self.logger.warning("🔄 [AUTO RETRY] Format errors persist after normalization. Retrying generation.")
                self.retry_manager.increment_retry_count(state, "generation")
                try:
                    state = self.generate_answer_enhanced(state)
                    quality_check_passed = self._validate_answer_quality_internal(state)
                except asyncio.CancelledError:
                    self.logger.warning("⚠️ [FORMAT ERRORS] Answer generation was cancelled. Using normalized answer.")
                    self._set_answer_safely(state, normalized_answer)
                    quality_check_passed = True
                except Exception as format_regen_error:
                    self.logger.error(f"⚠️ [FORMAT ERRORS] Error during regeneration: {format_regen_error}", exc_info=True)
                    self._set_answer_safely(state, normalized_answer)
                    quality_check_passed = True
        
        return quality_check_passed
    
    def _format_and_finalize(self, state: LegalWorkflowState, overall_start_time: float) -> None:
        """포맷팅 및 최종 준비"""
        formatting_start_time = time.time()
        try:
            state = self._format_and_finalize_answer(state)
            self._update_processing_time(state, formatting_start_time)
            
            elapsed = time.time() - overall_start_time
            confidence = state.get("confidence", 0.0)
            self.logger.debug(
                f"✅ [FINAL NODE] 최종 검증 및 포맷팅 완료 ({elapsed:.2f}s), "
                f"confidence: {confidence:.3f}"
            )
        except asyncio.CancelledError:
            self.logger.warning("⚠️ [FORMAT AND FINALIZE] Formatting was cancelled. Using basic format.")
            existing_answer = self._get_state_value(state, "answer", "")
            if existing_answer and len(str(existing_answer).strip()) > 10:
                state["answer"] = self._normalize_answer(str(existing_answer))
                self._prepare_final_response_minimal(state)
            else:
                state["answer"] = self._normalize_answer(state.get("answer", ""))
                self._prepare_final_response_minimal(state)
            self._update_processing_time(state, formatting_start_time)
            # CancelledError는 상위로 전파하지 않고 기본 포맷으로 처리 완료
        except Exception as format_error:
            self.logger.warning(f"Formatting failed: {format_error}, using basic format")
            state["answer"] = self._normalize_answer(state.get("answer", ""))
            self._prepare_final_response_minimal(state)
            self._update_processing_time(state, formatting_start_time)
    
    def _handle_final_node_error(self, state: LegalWorkflowState, error: Exception) -> None:
        """최종 노드 오류 처리"""
        error_msg = str(error)
        
        if isinstance(error, asyncio.CancelledError):
            self.logger.warning("⚠️ [FORMAT_ANSWER] Operation was cancelled. Preserving existing answer.")
        else:
            self.logger.error(f"⚠️ [FORMAT_ANSWER] Exception occurred: {error_msg}", exc_info=True)
            self._handle_error(state, error_msg, "최종 검증 및 포맷팅 중 오류 발생")
        
        if error_msg == 'control' or 'control' in error_msg.lower():
            self.logger.warning("[FORMAT_ANSWER] 'control' error detected. Preserving existing answer if available.")
        
        existing_answer = self._get_state_value(state, "answer", "")
        if isinstance(existing_answer, dict):
            existing_answer = existing_answer.get("text", "") or existing_answer.get("content", "") or str(existing_answer)
        elif not isinstance(existing_answer, str):
            existing_answer = str(existing_answer) if existing_answer else ""
        
        if existing_answer and len(existing_answer.strip()) > 10:
            self._set_answer_safely(state, existing_answer)
            self.logger.debug(f"✅ [FORMAT_ANSWER] Preserved existing answer: length={len(existing_answer)}")
        else:
            query = self._get_state_value(state, "query", "")
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            if retrieved_docs and len(retrieved_docs) > 0:
                minimal_answer = f"질문 '{query}'에 대한 답변을 준비했습니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
            else:
                minimal_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다."
            self._set_answer_safely(state, minimal_answer)
            self.logger.debug(f"⚠️ [FORMAT_ANSWER] Generated minimal answer: length={len(minimal_answer)}")
        
        self._set_state_value(state, "legal_validity_check", True)
        self._save_metadata_safely(state, "quality_score", 0.0, save_to_top_level=True)
        self._save_metadata_safely(state, "quality_check_passed", False, save_to_top_level=True)
    
    @with_state_optimization("generate_answer_stream", enable_reduction=True)
    def generate_answer_stream(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """스트리밍 전용 답변 생성 노드 - 스트리밍만 수행하고 검증/포맷팅은 하지 않음 (콜백 방식 사용)"""
        try:
            start_time = time.time()
            self.logger.debug("📡 [STREAM NODE] 스트리밍 전용 답변 생성 시작 (콜백 방식)")
            
            # 🔥 개선: 검색 결과를 여러 위치에서 복구 (강화된 버전)
            retrieved_docs = self._recover_retrieved_docs_comprehensive(state)
            structured_docs = self._get_state_value(state, "structured_documents", [])
            
            # structured_docs가 없으면 retrieved_docs 사용
            if not structured_docs and retrieved_docs:
                structured_docs = retrieved_docs
                self._set_state_value(state, "structured_documents", structured_docs)
            
            # 🔥 개선: 검색 결과가 없으면 경고 및 복구 시도
            if not retrieved_docs or len(retrieved_docs) == 0:
                self.logger.warning("⚠️ [STREAM NODE] retrieved_docs가 비어있음. 추가 복구 시도 중...")
                retrieved_docs = self._recover_retrieved_docs_comprehensive(state)
                
                if not retrieved_docs or len(retrieved_docs) == 0:
                    self.logger.error("❌ [STREAM NODE] retrieved_docs 복구 실패. 답변 생성이 제한될 수 있습니다.")
            
            # 검색 결과를 명시적으로 state에 저장 (다음 단계를 위해)
            if retrieved_docs:
                self._set_state_value(state, "retrieved_docs", retrieved_docs)
                # 여러 위치에 저장
                if "search" not in state:
                    state["search"] = {}
                state["search"]["retrieved_docs"] = retrieved_docs.copy()
            
            # 중요: query_type 보존
            preserved_query_type = self._get_state_value(state, "query_type") or (state.get("metadata", {}).get("query_type") if isinstance(state.get("metadata"), dict) else None)
            
            # 🔥 개선: state에서 콜백 추출 (스트리밍을 위해 필요)
            callbacks = state.get("_callbacks", [])
            if not callbacks:
                metadata = self._get_metadata_safely(state)
                callbacks = metadata.get("_callbacks", [])
            
            if callbacks:
                self.logger.debug(f"📡 [STREAM NODE] 콜백 {len(callbacks)}개를 state에서 추출하여 전달")
            else:
                self.logger.warning("⚠️ [STREAM NODE] 콜백이 없습니다. 스트리밍이 작동하지 않을 수 있습니다.")
            
            # generate_answer_enhanced 실행 (답변 생성만)
            # 콜백을 전달하여 스트리밍이 작동하도록 함
            state = self.generate_answer_enhanced(state, callbacks=callbacks)
            
            # 보존된 필드 복원 (reduction으로 손실된 경우 대비)
            if retrieved_docs and not self._get_state_value(state, "retrieved_docs"):
                self._set_state_value(state, "retrieved_docs", retrieved_docs)
            if structured_docs and not self._get_state_value(state, "structured_documents"):
                self._set_state_value(state, "structured_documents", structured_docs)
            if preserved_query_type:
                metadata = self._get_metadata_safely(state)
                if "query_type" not in metadata:
                    metadata["query_type"] = preserved_query_type
                self._set_state_value(state, "metadata", metadata)
                # top-level에도 보존
                if "query_type" not in state:
                    state["query_type"] = preserved_query_type
            
            # 성능 최적화: 간단한 검증 및 포맷팅 수행 (generate_answer_final 스킵 가능하도록)
            answer = self._get_state_value(state, "answer", "")
            
            # 개선 10.1: 재생성 로직 적용
            coverage = self._get_state_value(state, "search_quality_evaluation", {}).get("coverage", 0.0)
            citations = self._extract_citations(self._get_state_value(state, "retrieved_docs", []))
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            
            should_regenerate = self._should_regenerate_answer(
                answer=answer or "",
                coverage=coverage,
                citation_count=len(citations),
                retrieved_docs_count=len(retrieved_docs)
            )
            
            if should_regenerate and answer and len(answer.strip()) >= 10:
                # 인용 자동 추가
                legal_references = self._get_state_value(state, "legal_references", [])
                answer = self._enhance_answer_with_citations(
                    answer=answer,
                    retrieved_docs=retrieved_docs,
                    legal_references=legal_references,
                    citations=citations
                )
                self._set_state_value(state, "legal_citations", citations)
                self._set_answer_safely(state, answer)
                self.logger.debug(f"✅ [CITATIONS] {len(citations)}개 인용 추가됨")
            
            if answer and len(answer.strip()) >= 10:
                # 간단한 정규화만 수행 (비용이 낮음)
                normalized_answer = WorkflowUtils.normalize_answer(answer)
                if normalized_answer != answer:
                    self._set_answer_safely(state, normalized_answer)
                
                # 검증 완료 표시 (generate_answer_final 스킵 가능)
                metadata = self._get_metadata_safely(state)
                metadata["streaming_completed"] = True
                metadata["streaming_node_executed"] = True
                metadata["quality_check_passed"] = True  # 간단한 검증 완료 표시
                metadata["skip_final_node"] = True  # 최종 노드 스킵 가능 표시
                self._set_state_value(state, "metadata", metadata)
            else:
                # 답변이 없거나 너무 짧으면 최종 노드에서 처리
                metadata = self._get_metadata_safely(state)
                metadata["streaming_completed"] = True
                metadata["streaming_node_executed"] = True
                metadata["skip_final_node"] = False  # 최종 노드 필요
                self._set_state_value(state, "metadata", metadata)
            
            elapsed = time.time() - start_time
            self.logger.debug(f"📡 [STREAM NODE] 스트리밍 전용 답변 생성 완료 ({elapsed:.2f}s)")
            
        except asyncio.CancelledError:
            self.logger.warning("⚠️ [STREAM NODE] 스트리밍 작업이 취소되었습니다. 기존 답변 보존 시도 중...")
            # 취소된 경우 기존 답변 보존 시도
            existing_answer = self._get_state_value(state, "answer", "")
            if existing_answer and len(str(existing_answer).strip()) > 10:
                self._set_answer_safely(state, existing_answer)
                self.logger.debug(f"✅ [STREAM NODE] 취소 후 기존 답변 보존 완료 (길이: {len(str(existing_answer))}자)")
                # 메타데이터 업데이트
                metadata = self._get_metadata_safely(state)
                metadata["streaming_completed"] = True
                metadata["streaming_cancelled"] = True
                self._set_state_value(state, "metadata", metadata)
            else:
                self.logger.warning("⚠️ [STREAM NODE] 보존할 답변이 없습니다.")
                self._set_answer_safely(state, "")
        except Exception as e:
            self.logger.error(f"❌ [STREAM NODE] 스트리밍 노드 오류: {e}", exc_info=True)
            self._handle_error(state, str(e), "스트리밍 답변 생성 중 오류 발생")
            if "answer" not in state or not state.get("answer"):
                self._set_answer_safely(state, "")
        
        return state

    @with_state_optimization("generate_answer_final", enable_reduction=True)
    def generate_answer_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 검증 및 포맷팅 노드 - 검증과 포맷팅만 수행"""
        try:
            overall_start_time = time.time()
            self.logger.debug("✅ [FINAL NODE] 최종 검증 및 포맷팅 시작")
            
            # 답변이 없으면 먼저 생성 시도
            existing_answer = self._get_state_value(state, "answer", "")
            # 딕셔너리 형태의 answer 처리
            if isinstance(existing_answer, dict):
                existing_answer = existing_answer.get("answer", "") if isinstance(existing_answer, dict) else ""
            # 문자열로 변환
            answer_str = str(existing_answer).strip() if existing_answer else ""
            if not answer_str or len(answer_str) < 10:
                self.logger.warning("⚠️ [FINAL NODE] 답변이 없거나 너무 짧습니다. generate_answer_enhanced 호출 중...")
                state = self.generate_answer_enhanced(state)
                existing_answer = self._get_state_value(state, "answer", "")
                # 딕셔너리 형태의 answer 처리
                if isinstance(existing_answer, dict):
                    existing_answer = existing_answer.get("answer", "") if isinstance(existing_answer, dict) else ""
                answer_str = str(existing_answer).strip() if existing_answer else ""
                if answer_str and len(answer_str) >= 10:
                    self.logger.debug(f"✅ [FINAL NODE] 답변 생성 완료 (길이: {len(answer_str)}자)")
                else:
                    self.logger.warning("⚠️ [FINAL NODE] 답변 생성 후에도 답변이 없습니다.")
            
            try:
                self._restore_state_data_for_final(state)
            except asyncio.CancelledError:
                self.logger.warning("⚠️ [FINAL NODE] State restoration was cancelled.")
                raise
            except Exception as e:
                self.logger.warning(f"⚠️ [FINAL NODE] State restoration failed: {e}")
            
            validation_start_time = time.time()
            try:
                quality_check_passed = self._validate_and_handle_regeneration(state)
            except asyncio.CancelledError:
                self.logger.warning("⚠️ [FINAL NODE] Validation was cancelled. Preserving existing answer.")
                existing_answer = self._get_state_value(state, "answer", "")
                if existing_answer and len(str(existing_answer).strip()) > 10:
                    self._set_answer_safely(state, existing_answer)
                    self.logger.debug(f"✅ [FINAL NODE] Preserved existing answer after cancellation: length={len(str(existing_answer))}")
                    return state
                raise
            
            try:
                quality_check_passed = self._handle_format_errors(state, quality_check_passed)
            except asyncio.CancelledError:
                self.logger.warning("⚠️ [FINAL NODE] Format error handling was cancelled. Preserving existing answer.")
                existing_answer = self._get_state_value(state, "answer", "")
                if existing_answer and len(str(existing_answer).strip()) > 10:
                    self._set_answer_safely(state, existing_answer)
                    self.logger.debug(f"✅ [FINAL NODE] Preserved existing answer after cancellation: length={len(str(existing_answer))}")
                    return state
                raise
            
            self._update_processing_time(state, validation_start_time)
            
            if quality_check_passed:
                try:
                    self._format_and_finalize(state, overall_start_time)
                except asyncio.CancelledError:
                    self.logger.warning("⚠️ [FINAL NODE] Formatting was cancelled. Preserving existing answer.")
                    existing_answer = self._get_state_value(state, "answer", "")
                    if existing_answer and len(str(existing_answer).strip()) > 10:
                        self._set_answer_safely(state, existing_answer)
                        self.logger.debug(f"✅ [FINAL NODE] Preserved existing answer after cancellation: length={len(str(existing_answer))}")
                        return state
                    raise

            self._update_processing_time(state, overall_start_time)

        except asyncio.CancelledError:
            # 최상위 CancelledError 처리 - 이미 처리된 경우가 아니면 여기서 처리
            self.logger.warning("⚠️ [FINAL NODE] Operation was cancelled. Preserving existing answer.")
            existing_answer = self._get_state_value(state, "answer", "")
            if existing_answer and len(str(existing_answer).strip()) > 10:
                self._set_answer_safely(state, existing_answer)
                self.logger.debug(f"✅ [FINAL NODE] Preserved existing answer after cancellation: length={len(str(existing_answer))}")
            else:
                # 답변이 없으면 기본 답변 설정
                self._set_answer_safely(state, "죄송합니다. 작업이 취소되었습니다.")
                self.logger.warning("⚠️ [FINAL NODE] No existing answer to preserve. Set default message.")
            # CancelledError는 다시 발생시키지 않고 상태를 보존한 채로 반환
            # LangGraph가 비동기 실행 중 취소를 처리할 수 있도록 함
        except Exception as e:
            self._handle_final_node_error(state, e)

        return state

    @with_state_optimization("generate_and_validate_answer", enable_reduction=True)
    def generate_and_validate_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 답변 생성, 검증, 포맷팅 및 최종 준비"""
        try:
            overall_start_time = time.time()

            # Part 1: 답변 생성 (generate_answer_enhanced 실행)
            generation_start_time = time.time()

            # 이전에 실행된 노드 확인
            metadata = state.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if "common" in state and isinstance(state.get("common"), dict):
                common_metadata = state["common"].get("metadata", {})
                if isinstance(common_metadata, dict):
                    metadata = {**metadata, **common_metadata}

            last_executed_node = metadata.get("_last_executed_node", "")
            is_retry = (last_executed_node == "validate_answer_quality" or last_executed_node == "generate_and_validate_answer")
            if is_retry:
                if self.retry_manager.should_allow_retry(state, "validation"):
                    self.retry_manager.increment_retry_count(state, "validation")

            # generate_answer_enhanced 실행
            # LangGraph는 노드 내에서 stream() 또는 astream()을 호출하면 자동으로 on_llm_stream 이벤트를 발생시킵니다
            state = self.generate_answer_enhanced(state)

            self._update_processing_time(state, generation_start_time)
            self._save_metadata_safely(state, "_last_executed_node", "generate_and_validate_answer")

            # Part 2: 품질 검증 (validate_answer_quality 로직)
            validation_start_time = time.time()

            quality_check_passed = self._validate_answer_quality_internal(state)
            
            # 재생성 필요 여부 확인 (여러 위치에서 검색)
            needs_regeneration_from_helper = self._get_state_value(state, "needs_regeneration", False)
            needs_regeneration_from_top = state.get("needs_regeneration", False)
            needs_regeneration_from_metadata = state.get("metadata", {}).get("needs_regeneration", False)
            needs_regeneration = needs_regeneration_from_helper or needs_regeneration_from_top or needs_regeneration_from_metadata
            
            # 디버깅: needs_regeneration 값 확인
            self.logger.debug(
                f"🔍 [REGENERATION DEBUG] After validation:\n"
                f"   needs_regeneration_from_helper: {needs_regeneration_from_helper}\n"
                f"   needs_regeneration_from_top: {needs_regeneration_from_top}\n"
                f"   needs_regeneration_from_metadata: {needs_regeneration_from_metadata}\n"
                f"   needs_regeneration (combined): {needs_regeneration}\n"
                f"   state keys: {list(state.keys())[:10]}..."
            )
            regeneration_reason = (
                self._get_state_value(state, "regeneration_reason") or
                state.get("regeneration_reason") or
                state.get("metadata", {}).get("regeneration_reason") or
                "unknown"
            )
            if needs_regeneration:
                retry_counts = self.retry_manager.get_retry_counts(state)
                can_retry = self.retry_manager.should_allow_retry(state, "generation")
                
                self.logger.debug(
                    f"🔄 [REGENERATION CHECK] needs_regeneration={needs_regeneration}, "
                    f"can_retry={can_retry}, reason={regeneration_reason}, "
                    f"retry_count={retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES}"
                )
                
                if can_retry and retry_counts['generation'] < RetryConfig.MAX_GENERATION_RETRIES:
                    # 재시도 전 답변 저장 (비교용)
                    previous_answer = self._get_state_value(state, "answer", "")
                    previous_copy_score = 0.0
                    if previous_answer:
                        previous_result = self._detect_specific_case_copy(previous_answer)
                        previous_copy_score = previous_result.get("copy_score", 0.0)
                    
                    self.logger.warning(
                        f"🔄 [AUTO RETRY] Regeneration needed: {regeneration_reason}. "
                        f"Retrying answer generation "
                        f"(retry count: {retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES}, "
                        f"previous_copy_score: {previous_copy_score:.2f})"
                    )
                    self.retry_manager.increment_retry_count(state, "generation")
                    # 재생성을 위해 generate_answer_enhanced 다시 호출
                    state = self.generate_answer_enhanced(state)
                    
                    # 재시도 후 답변 검증 및 비교
                    new_answer = self._get_state_value(state, "answer", "")
                    if new_answer:
                        new_result = self._detect_specific_case_copy(new_answer)
                        new_copy_score = new_result.get("copy_score", 0.0)
                        
                        # 개선 여부 확인
                        if new_copy_score < previous_copy_score:
                            self.logger.debug(
                                f"✅ [RETRY IMPROVEMENT] Copy score improved: {previous_copy_score:.2f} → {new_copy_score:.2f}"
                            )
                        elif new_copy_score >= previous_copy_score:
                            self.logger.warning(
                                f"⚠️ [RETRY NO IMPROVEMENT] Copy score not improved: {previous_copy_score:.2f} → {new_copy_score:.2f}"
                            )
                    
                    # 재검증
                    quality_check_passed = self._validate_answer_quality_internal(state)
                    # 재생성 플래그 초기화
                    self._set_state_value(state, "needs_regeneration", False)
                else:
                    self.logger.warning(
                        f"⚠️ [REGENERATION SKIP] Cannot retry: retry_count={retry_counts['generation']}, "
                        f"max_retries={RetryConfig.MAX_GENERATION_RETRIES}"
                    )
                    # 재시도 불가 시 재생성 플래그 초기화
                    self._set_state_value(state, "needs_regeneration", False)
            
            # 형식 오류가 감지된 경우 자동 재생성
            has_format_errors = self._detect_format_errors(self._get_state_value(state, "answer", ""))
            if has_format_errors and self.retry_manager.should_allow_retry(state, "generation"):
                self.logger.warning(
                    f"🔄 [AUTO RETRY] Format errors detected. Retrying answer generation "
                    f"(retry count: {self.retry_manager.get_retry_counts(state)['generation']}/{RetryConfig.MAX_GENERATION_RETRIES})"
                )
                # 답변 정규화로 형식 오류 제거 시도
                normalized_answer = self._normalize_answer(self._get_state_value(state, "answer", ""))
                self._set_answer_safely(state, normalized_answer)
                
                # 정규화 후에도 형식 오류가 있으면 재생성
                if self._detect_format_errors(normalized_answer):
                    self.logger.warning("🔄 [AUTO RETRY] Format errors persist after normalization. Retrying generation.")
                    self.retry_manager.increment_retry_count(state, "generation")
                    # 재생성을 위해 generate_answer_enhanced 다시 호출
                    state = self.generate_answer_enhanced(state)
                    # 재검증
                    quality_check_passed = self._validate_answer_quality_internal(state)

            self._update_processing_time(state, validation_start_time)

            # Part 3: 검증 통과 시 포맷팅 및 최종 준비
            if quality_check_passed:
                formatting_start_time = time.time()
                try:
                    state = self._format_and_finalize_answer(state)
                    self._update_processing_time(state, formatting_start_time)

                    elapsed = time.time() - overall_start_time
                    confidence = state.get("confidence", 0.0)
                    self.logger.debug(
                        f"generate_and_validate_answer completed (with formatting) in {elapsed:.2f}s, "
                        f"confidence: {confidence:.3f}"
                    )
                except Exception as format_error:
                    self.logger.warning(f"Formatting failed: {format_error}, using basic format")
                    state["answer"] = self._normalize_answer(state.get("answer", ""))
                    self._prepare_final_response_minimal(state)
                    self._update_processing_time(state, formatting_start_time)

            self._update_processing_time(state, overall_start_time)

        except Exception as e:
            self._handle_error(state, str(e), "답변 생성 및 검증 중 오류 발생")
            # Phase 1/Phase 7: 기본값 설정 - _set_answer_safely 사용
            if "answer" not in state or not state.get("answer"):
                self._set_answer_safely(state, "")
            elif state.get("answer"):
                # answer가 있으면 정규화만 수행
                self._set_answer_safely(state, state["answer"])
            self._set_state_value(state, "legal_validity_check", True)
            self._save_metadata_safely(state, "quality_score", 0.0, save_to_top_level=True)
            self._save_metadata_safely(state, "quality_check_passed", False, save_to_top_level=True)

        return state

    def _validate_answer_quality_internal(self, state: LegalWorkflowState) -> bool:
        """AnswerQualityValidator.validate_answer_quality 래퍼 + TASK 13: 관련 없는 조문 인용 검증"""
        if self.answer_quality_validator:
            result = self.answer_quality_validator.validate_answer_quality(state)
            
            # TASK 13: 관련 없는 조문 인용 검증
            answer = self._get_state_value(state, "answer", "")
            query = self._get_state_value(state, "query", "")
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            
            if answer and query and retrieved_docs:
                citation_validation = self._validate_answer_citations(answer, query, retrieved_docs)
                if not citation_validation.get("is_valid", True):
                    self.logger.warning(
                        f"⚠️ [TASK 13] 관련 없는 조문 인용 감지: "
                        f"unrelated_articles={citation_validation.get('unrelated_articles', [])}, "
                        f"invalid_articles={citation_validation.get('invalid_articles', [])}"
                    )
                    # 관련 없는 조문이 있으면 품질 점수 감소
                    return False
            
            return result
        return True
    
    def _validate_answer_citations(
        self, 
        answer: str, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """답변의 인용 검증 (관련성 확인) - TASK 13"""
        import re
        
        # 질문에서 조문 번호 추출
        question_articles = set()
        article_matches = re.findall(r'제\s*(\d+)\s*조', query)
        question_articles.update(article_matches)
        
        # 답변에서 인용된 조문 추출
        answer_articles = set()
        answer_matches = re.findall(r'제\s*(\d+)\s*조', answer)
        answer_articles.update(answer_matches)
        
        # 관련 없는 조문 감지
        unrelated_articles = answer_articles - question_articles
        
        # 검색된 문서의 조문 번호
        doc_articles = set()
        for doc in retrieved_docs:
            article_no = doc.get("article_no") or doc.get("article_number")
            if article_no:
                article_str = str(article_no).strip().lstrip('0')
                if article_str:
                    doc_articles.add(article_str)
        
        # 문서에 없는 조문 인용 감지
        invalid_articles = answer_articles - doc_articles
        
        return {
            "has_unrelated_articles": len(unrelated_articles) > 0,
            "unrelated_articles": list(unrelated_articles),
            "has_invalid_articles": len(invalid_articles) > 0,
            "invalid_articles": list(invalid_articles),
            "is_valid": len(unrelated_articles) == 0 and len(invalid_articles) == 0
        }
    
    def _detect_format_errors(self, answer: str) -> bool:
        """AnswerQualityValidator.detect_format_errors 래퍼"""
        if self.answer_quality_validator:
            return self.answer_quality_validator.detect_format_errors(answer)
        return False
    
    def _detect_specific_case_copy(self, answer: str) -> Dict[str, Any]:
        """AnswerQualityValidator.detect_specific_case_copy 래퍼"""
        if self.answer_quality_validator:
            return self.answer_quality_validator.detect_specific_case_copy(answer)
        return {
            "has_specific_case": False,
            "case_numbers": [],
            "party_names": [],
            "copy_score": 0.0,
            "needs_regeneration": False
        }
    
    def _check_general_principle_first(self, answer: str) -> Dict[str, Any]:
        """일반 법적 원칙이 먼저 설명되었는지 검증 (WorkflowValidator 사용)"""
        return self.workflow_validator.check_general_principle_first(answer)
    
    def _check_answer_structure(self, answer: str) -> Dict[str, Any]:
        """답변 구조가 올바른지 검증 (WorkflowValidator 사용)"""
        return self.workflow_validator.check_answer_structure(answer)
    
    def _check_has_sources(self, state: LegalWorkflowState, sources: List[Any]) -> bool:
        """소스 존재 여부 확인 (WorkflowValidator 사용)"""
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        legal_references = self._get_state_value(state, "legal_references", [])
        legal_citations = self._get_state_value(state, "legal_citations", [])
        return self.workflow_validator.check_has_sources(
            sources=sources,
            retrieved_docs=retrieved_docs,
            legal_references=legal_references,
            legal_citations=legal_citations
        )

    def _format_and_finalize_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """포맷팅 및 최종 준비 (내부 메서드)"""
        # AnswerFormatterHandler 사용 - format_and_prepare_final 사용하여 메타 정보 분리 포함
        self.logger.debug("[FORMAT_AND_FINALIZE] Calling format_and_prepare_final")
        state = self.answer_formatter_handler.format_and_prepare_final(state)
        self.logger.debug(f"[FORMAT_AND_FINALIZE] format_and_prepare_final completed, legal_references={len(state.get('legal_references', []))}, related_questions={len(state.get('metadata', {}).get('related_questions', []))}")

        # 통계 업데이트
        self.update_statistics(state)

        return state

    def _prepare_final_response_minimal(self, state: LegalWorkflowState) -> None:
        """최소한의 최종 준비 (포맷팅 실패 시)"""
        query_complexity = state.get("metadata", {}).get("query_complexity")
        needs_search = state.get("metadata", {}).get("needs_search", False)
        self.answer_formatter_handler.prepare_final_response_part(state, query_complexity, needs_search)

    # ============================================================================
    # WorkflowUtils 래퍼 메서드
    # ============================================================================
    
    # StateUtilsMixin으로 이동됨

    # ============================================================================
    # 분류 관련 노드들 (Classification Nodes)
    # ============================================================================

    @with_state_optimization("classify_complexity", enable_reduction=False)  # 라우팅에 필요한 값 보존을 위해 reduction 비활성화
    def classify_complexity(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 복잡도를 판단하고 검색 필요 여부 결정 (Adaptive RAG)"""
        try:
            start_time = time.time()

            query = self._get_state_value(state, "query", "")
            if not query:
                # 기본값: 중간 복잡도 (검색 필요)
                self._set_state_value(state, "query_complexity", QueryComplexity.MODERATE.value)
                self._set_state_value(state, "needs_search", True)
                return state

            query_lower = query.lower()

            # 1. 인사말/간단한 질문 체크
            simple_greetings = ["안녕", "고마워", "감사", "도움", "설명", "안녕하세요", "고마워요", "감사합니다"]
            if any(pattern in query_lower for pattern in simple_greetings):
                if len(query) < 20:  # 매우 짧은 인사말
                    complexity = QueryComplexity.SIMPLE
                    needs_search = False
                    self.logger.debug(f"✅ 간단한 질문 감지 (인사말): {query[:50]}...")
                    # classification 그룹과 최상위 레벨 모두에 저장 (라우팅 및 최종 결과 추출을 위해)
                    self._set_state_value(state, "query_complexity", complexity.value)
                    self._set_state_value(state, "needs_search", needs_search)
                    # 최상위 레벨에도 직접 저장
                    if "classification" not in state:
                        state["classification"] = {}
                    state["classification"]["query_complexity"] = complexity.value
                    state["classification"]["needs_search"] = needs_search
                    # 최상위 레벨에도 저장
                    state["query_complexity"] = complexity.value
                    state["needs_search"] = needs_search
                    # common 그룹과 metadata에도 저장 (reducer가 보존하는 그룹)
                    if "common" not in state:
                        state["common"] = {}
                    state["common"]["query_complexity"] = complexity.value
                    state["common"]["needs_search"] = needs_search
                    # metadata에도 저장 (기존 내용 보존)
                    if "metadata" not in state:
                        state["metadata"] = {}
                    elif not isinstance(state.get("metadata"), dict):
                        state["metadata"] = {}
                    # 기존 metadata 내용 보존하면서 query_complexity 추가
                    state["metadata"]["query_complexity"] = complexity.value
                    state["metadata"]["needs_search"] = needs_search

                    # 중요: Global cache에도 저장 (reducer 손실 방지)
                    try:
                        from core.agents import node_wrappers
                        # 모듈 레벨 변수에 직접 접근
                        if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                            node_wrappers._global_search_results_cache = {}
                        # query_complexity 정보 저장
                        node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
                        node_wrappers._global_search_results_cache["needs_search"] = needs_search
                        self.logger.debug(f"[DEBUG] classify_complexity (간단): ✅ Global cache 저장 완료 - complexity={complexity.value}, needs_search={needs_search}")
                    except Exception as e:
                        self.logger.debug(f"[DEBUG] classify_complexity (간단): ❌ Global cache 저장 실패: {e}")
                        import traceback
                        self.logger.debug(f"[DEBUG] classify_complexity (간단): Exception traceback: {traceback.format_exc()}")

                    # 디버깅: 저장 확인
                    # saved_complexity and saved_needs_search are retrieved but not used
                    top_level_complexity = state.get("query_complexity")
                    top_level_needs_search = state.get("needs_search")
                    common_complexity = state.get("common", {}).get("query_complexity")
                    metadata_complexity = state.get("metadata", {}).get("query_complexity")
                    self.logger.debug("[DEBUG] classify_complexity: 저장 완료")
                    self.logger.debug(f"  - 최상위 레벨: complexity={top_level_complexity}, needs_search={top_level_needs_search}")
                    self.logger.debug(f"  - classification 그룹: complexity={state.get('classification', {}).get('query_complexity')}")
                    self.logger.debug(f"  - common 그룹: complexity={common_complexity}")
                    self.logger.debug(f"  - metadata: complexity={metadata_complexity}")

                    processing_time = self._update_processing_time(state, start_time)
                    self._add_step(state, "복잡도 분류", f"간단한 질문 (인사말) - 검색 불필요 (시간: {processing_time:.3f}s)")
                    return state

            # 2. 법률 용어 정의 질문 체크
            definition_keywords = ["뜻", "의미", "정의", "이란", "란 무엇", "무엇인가", "무엇이야", "무엇이냐"]
            if any(pattern in query_lower for pattern in definition_keywords):
                # 단순 정의 질문인지 확인 (길이와 키워드로 판단)
                if len(query) < 30 and any(word in query for word in definition_keywords):
                    # 간단한 정의 질문
                    complexity = QueryComplexity.SIMPLE
                    needs_search = False
                    self.logger.debug(f"✅ 간단한 질문 감지 (용어 정의): {query[:50]}...")
                    self._set_state_value(state, "query_complexity", complexity.value)
                    self._set_state_value(state, "needs_search", needs_search)
                    # Global cache에도 저장
                    try:
                        from core.agents import node_wrappers
                        if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                            node_wrappers._global_search_results_cache = {}
                        node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
                        node_wrappers._global_search_results_cache["needs_search"] = needs_search
                        self.logger.debug("[DEBUG] classify_complexity (용어정의): Global cache 저장 완료")
                    except Exception as e:
                        self.logger.debug(f"[DEBUG] classify_complexity (용어정의): ❌ Global cache 저장 실패: {e}")
                    processing_time = self._update_processing_time(state, start_time)
                    self._add_step(state, "복잡도 분류", f"간단한 질문 (용어 정의) - 검색 불필요 (시간: {processing_time:.3f}s)")
                    return state

            # 3. 특정 조문/법령 질의 (중간 복잡도)
            if ("조" in query or "법" in query or "법령" in query or "법률" in query) and len(query) < 50:
                complexity = QueryComplexity.MODERATE
                needs_search = True
                self.logger.debug(f"📋 중간 복잡도 질문 (법령 조회): {query[:50]}...")

            # 4. 복잡한 질문 (비교, 절차, 사례 분석)
            elif any(keyword in query for keyword in ["비교", "차이", "어떻게", "방법", "절차", "사례", "판례 비교"]):
                complexity = QueryComplexity.COMPLEX
                needs_search = True
                self.logger.debug(f"🔍 복잡한 질문 감지: {query[:50]}...")

            # 5. 기본값 (중간 복잡도)
            else:
                complexity = QueryComplexity.MODERATE
                needs_search = True

            # State에 저장 (classification 그룹과 최상위 레벨 모두에 저장)
            self._set_state_value(state, "query_complexity", complexity.value)
            self._set_state_value(state, "needs_search", needs_search)

            # 최상위 레벨에도 직접 저장 (라우팅 및 최종 결과 추출을 위해)
            if "classification" not in state:
                state["classification"] = {}
            state["classification"]["query_complexity"] = complexity.value
            state["classification"]["needs_search"] = needs_search
            # 최상위 레벨에도 저장
            state["query_complexity"] = complexity.value
            state["needs_search"] = needs_search
            # common 그룹과 metadata에도 저장 (reducer가 보존하는 그룹)
            if "common" not in state:
                state["common"] = {}
            state["common"]["query_complexity"] = complexity.value
            state["common"]["needs_search"] = needs_search
            # metadata에도 저장 (기존 내용 보존)
            if "metadata" not in state:
                state["metadata"] = {}
            elif not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            # 기존 metadata 내용 보존하면서 query_complexity 추가
            state["metadata"]["query_complexity"] = complexity.value
            state["metadata"]["needs_search"] = needs_search

            # 중요: Global cache에도 저장 (reducer 손실 방지)
            try:
                from core.agents import node_wrappers
                # 모듈 레벨 변수에 직접 접근
                if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                    node_wrappers._global_search_results_cache = {}
                # query_complexity 정보 저장
                node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
                node_wrappers._global_search_results_cache["needs_search"] = needs_search
                self.logger.debug(f"[DEBUG] classify_complexity: ✅ Global cache 저장 완료 - complexity={complexity.value}, needs_search={needs_search}")
                self.logger.debug(f"[DEBUG] classify_complexity: Global cache keys={list(node_wrappers._global_search_results_cache.keys())[:10]}")
            except Exception as e:
                self.logger.debug(f"[DEBUG] classify_complexity: ❌ Global cache 저장 실패: {e}")
                import traceback
                self.logger.debug(f"[DEBUG] classify_complexity: Exception traceback: {traceback.format_exc()}")

            # 디버깅: 저장 확인
            top_level_complexity = state.get("query_complexity")
            top_level_needs_search = state.get("needs_search")
            common_complexity = state.get("common", {}).get("query_complexity")
            metadata_complexity = state.get("metadata", {}).get("query_complexity")
            self.logger.debug("[DEBUG] classify_complexity: 저장 완료 (최종)")
            self.logger.debug(f"  - 최상위 레벨: complexity={top_level_complexity}, needs_search={top_level_needs_search}")
            self.logger.debug(f"  - classification 그룹: complexity={state.get('classification', {}).get('query_complexity')}")
            self.logger.debug(f"  - common 그룹: complexity={common_complexity}")
            self.logger.debug(f"  - metadata: complexity={metadata_complexity}")

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(
                state,
                "복잡도 분류",
                f"질문 복잡도: {complexity.value}, 검색 필요: {needs_search} (시간: {processing_time:.3f}s)"
            )

        except Exception as e:
            self._handle_error(state, str(e), "복잡도 분류 중 오류 발생")
            # 기본값: 중간 복잡도 (검색 필요)
            self._set_state_value(state, "query_complexity", QueryComplexity.MODERATE.value)
            self._set_state_value(state, "needs_search", True)

        return state

    @with_state_optimization("direct_answer", enable_reduction=True)
    def direct_answer_node(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """간단한 질문 - 검색 없이 LLM만 사용하고 포맷팅까지 통합 처리"""
        try:
            start_time = time.time()

            query = self._get_state_value(state, "query", "")
            if not query:
                self.logger.warning("direct_answer_node: query가 없습니다")
                return state

            # 빠른 모델 사용 (Flash)
            # llm is retrieved but not used (DirectAnswerHandler uses its own LLM)
            # llm = self.llm_fast if hasattr(self, 'llm_fast') and self.llm_fast else self.llm

            # Phase 10 리팩토링: DirectAnswerHandler 사용
            if not hasattr(self, 'direct_answer_handler') or self.direct_answer_handler is None:
                self.logger.warning("direct_answer_handler not initialized, falling back to search")
                self._set_state_value(state, "needs_search", True)
                return state
            
            # Prompt Chaining을 사용한 직접 답변 생성
            answer = self.direct_answer_handler.generate_direct_answer_with_chain(query)

            # 체인 실패 시 폴백
            if not answer or len(answer.strip()) < 10:
                self.logger.debug("Chain direct answer failed, using fallback")
                answer = self.direct_answer_handler.generate_fallback_answer(query)

                # 최소 길이 체크
                if not answer or len(answer.strip()) < 10:
                    # 폴백: 검색 경로로
                    answer_length = len(answer) if answer else 0
                    self.logger.warning(f"직접 답변이 너무 짧음 (길이: {answer_length}), 검색 경로로 전환")
                    self._set_state_value(state, "needs_search", True)
                    return state

            # 답변 저장 (체인 성공 또는 폴백 성공) - Phase 1: _set_answer_safely 사용
            self._set_answer_safely(state, answer)
            self._set_state_value(state, "sources", [])  # 검색 없음

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(
                state,
                "직접 답변 생성",
                f"검색 없이 직접 답변 생성 완료 (시간: {processing_time:.3f}s)"
            )

            # 포맷팅 및 최종 준비 (통합 처리)
            formatting_start_time = time.time()
            try:
                state = self._format_and_finalize_answer(state)
                self._update_processing_time(state, formatting_start_time)

                total_time = time.time() - start_time
                confidence = state.get("confidence", 0.0)
                self.logger.debug(
                    f"✅ 직접 답변 생성 및 포맷팅 완료 (검색 스킵): {query[:50]}... "
                    f"(총 시간: {total_time:.2f}s, confidence: {confidence:.3f})"
                )
            except Exception as format_error:
                self.logger.warning(f"Direct answer formatting failed: {format_error}, using basic format")
                state["answer"] = self._normalize_answer(state.get("answer", ""))
                self._prepare_final_response_minimal(state)
                self._update_processing_time(state, formatting_start_time)

        except Exception as e:
            self._handle_error(state, str(e), "direct_answer_node 중 오류 발생")
            # 폴백: 검색 경로로
            self._set_state_value(state, "needs_search", True)

        return state

    @with_state_optimization("resolve_multi_turn", enable_reduction=True)
    def resolve_multi_turn(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """ConversationProcessor.resolve_multi_turn 래퍼"""
        if self.conversation_processor:
            result_state = self.conversation_processor.resolve_multi_turn(state)
            is_multi_turn = self._get_state_value(result_state, "is_multi_turn", False)
            if is_multi_turn:
                query = self._get_state_value(result_state, "query", "")
                resolved_query = self._get_state_value(result_state, "search_query", query)
                self._add_step(result_state, "멀티턴 처리", f"멀티턴 질문 해결: '{query}' -> '{resolved_query}'")
            return result_state
        
        self.logger.warning("ConversationProcessor not available, using fallback")
        query = self._get_state_value(state, "query", "")
        self._set_state_value(state, "is_multi_turn", False)
        self._set_state_value(state, "multi_turn_confidence", 1.0)
        self._set_state_value(state, "search_query", query)
        return state

    def _get_or_create_conversation_context(self, session_id: str):
        """대화 맥락 가져오기 또는 생성"""
        try:
            if not self.conversation_manager:
                return None

            # 세션에서 대화 맥락 조회
            # ConversationManager의 sessions 딕셔너리에서 가져오기
            sessions = getattr(self.conversation_manager, 'sessions', {})
            context = sessions.get(session_id)

            return context
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return None

    def _build_conversation_context_dict(self, context):
        """ConversationContext를 딕셔너리로 변환"""
        result = QueryBuilder.build_conversation_context_dict(context)
        if result is None and context is not None:
            self.logger.error("Error building conversation context dict")
        return result

    def _convert_conversation_context_to_messages(
        self, 
        context, 
        max_turns: int = 5,
        current_query: str = "",
        use_relevance: bool = True,
        max_tokens: Optional[int] = None
    ) -> List:
        """
        ConversationContext를 LangChain Message 형식으로 변환
        관련성 기반 또는 토큰 기반으로 턴을 선택할 수 있음
        
        Args:
            context: ConversationContext 객체
            max_turns: 최대 변환할 턴 수
            current_query: 현재 질문 (관련성 계산용)
            use_relevance: 관련성 기반 선택 사용 여부
            max_tokens: 최대 토큰 수 (지정 시 토큰 기반 선택 사용)
            
        Returns:
            List[BaseMessage]: LangChain Message 리스트
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            messages = []
            if not context or not hasattr(context, 'turns') or not context.turns:
                return messages
            
            selected_turns = []
            
            # 토큰 기반 선택 (우선순위 1)
            if max_tokens and max_tokens > 0:
                try:
                    selected_turns = self._prune_conversation_history_by_tokens(
                        context,
                        max_tokens=max_tokens
                    )
                    if selected_turns:
                        self.logger.debug(f"Selected {len(selected_turns)} turns based on token limit ({max_tokens})")
                except Exception as e:
                    self.logger.warning(f"Failed to use token-based selection: {e}, falling back to relevance/recent")
            
            # 관련성 기반 선택 (우선순위 2, 토큰 기반이 실패하거나 사용하지 않는 경우)
            if not selected_turns and use_relevance and current_query and self.conversation_manager:
                try:
                    relevant_context = self.conversation_manager.get_relevant_context(
                        context.session_id,
                        current_query,
                        max_turns=max_turns
                    )
                    
                    if relevant_context and relevant_context.get("relevant_turns"):
                        # 관련 턴에서 실제 ConversationTurn 객체 찾기
                        relevant_turn_data = relevant_context["relevant_turns"]
                        for turn_data in relevant_turn_data:
                            user_query = turn_data.get("user_query", "")
                            bot_response = turn_data.get("bot_response", "")
                            
                            if user_query:
                                messages.append(HumanMessage(content=user_query))
                            if bot_response:
                                # "..." 제거 (get_relevant_context에서 추가된 것)
                                clean_response = bot_response.replace("...", "").strip()
                                messages.append(AIMessage(content=clean_response))
                        
                        self.logger.debug(f"Selected {len(relevant_turn_data)} relevant turns based on relevance")
                        return messages
                except Exception as e:
                    self.logger.warning(f"Failed to use relevance-based selection: {e}, falling back to recent turns")
            
            # 폴백: 최근 턴만 선택 (토큰 기반 선택 결과 사용 또는 최근 턴)
            if not selected_turns:
                selected_turns = context.turns[-max_turns:] if len(context.turns) > max_turns else context.turns
            
            # 선택된 턴을 메시지로 변환
            for turn in selected_turns:
                if hasattr(turn, 'user_query') and turn.user_query:
                    messages.append(HumanMessage(content=turn.user_query))
                if hasattr(turn, 'bot_response') and turn.bot_response:
                    messages.append(AIMessage(content=turn.bot_response))
            
            self.logger.debug(f"Converted {len(selected_turns)} turns to {len(messages)} messages")
            return messages
            
        except ImportError:
            self.logger.warning("langchain_core.messages not available, returning empty list")
            return []
        except Exception as e:
            self.logger.error(f"Error converting conversation context to messages: {e}")
            return []

    def _prune_conversation_history_by_tokens(
        self,
        context,
        max_tokens: int = 2000
    ) -> List:
        """
        토큰 수 기반으로 대화 이력 정리
        
        Args:
            context: ConversationContext 객체
            max_tokens: 최대 토큰 수
            
        Returns:
            List[ConversationTurn]: 선택된 턴 리스트
        """
        try:
            if not context or not hasattr(context, 'turns') or not context.turns:
                return []
            
            selected_turns = []
            total_tokens = 0
            
            def _estimate_tokens(text: str) -> int:
                """간단한 토큰 수 추정 (한글 기준 약 1.5배)"""
                if not text:
                    return 0
                # 한글은 평균적으로 영어보다 토큰 수가 많음
                # 대략적인 추정: 문자 수 / 3 (한글 기준)
                return len(text) // 3
            
            # 최근 턴부터 역순으로 검사 (최신 우선)
            for turn in reversed(context.turns):
                turn_text = ""
                if hasattr(turn, 'user_query') and turn.user_query:
                    user_query = turn.user_query
                    if isinstance(user_query, dict):
                        user_query = user_query.get('content', user_query.get('text', str(user_query)))
                    elif not isinstance(user_query, str):
                        user_query = str(user_query)
                    turn_text += user_query + " "
                if hasattr(turn, 'bot_response') and turn.bot_response:
                    bot_response = turn.bot_response
                    if isinstance(bot_response, dict):
                        bot_response = bot_response.get('content', bot_response.get('text', str(bot_response)))
                    elif not isinstance(bot_response, str):
                        bot_response = str(bot_response)
                    turn_text += bot_response
                
                turn_tokens = _estimate_tokens(turn_text)
                
                if total_tokens + turn_tokens <= max_tokens:
                    selected_turns.insert(0, turn)  # 시간순 유지
                    total_tokens += turn_tokens
                else:
                    # 토큰 제한 초과 시 중단
                    break
            
            self.logger.debug(f"Pruned conversation history: {len(context.turns)} → {len(selected_turns)} turns ({total_tokens} tokens)")
            return selected_turns
            
        except Exception as e:
            self.logger.error(f"Error pruning conversation history by tokens: {e}")
            # 폴백: 최근 5개 턴 반환
            if context and hasattr(context, 'turns') and context.turns:
                return context.turns[-5:]
            return []

    # 직접 답변 생성 래퍼 메서드 (DirectAnswerHandler)
    def _generate_direct_answer_with_chain(self, query: str, llm) -> Optional[str]:
        """DirectAnswerHandler.generate_direct_answer_with_chain 래퍼 (LLM 파라미터 무시)"""
        return self.direct_answer_handler.generate_direct_answer_with_chain(query)

    def _parse_query_type_analysis_response(self, response: str) -> Dict[str, Any]:
        """WorkflowUtils.parse_query_type_analysis_response 래퍼"""
        return WorkflowUtils.parse_query_type_analysis_response(response, self.logger)

    def _parse_quality_validation_response(self, response: str) -> Dict[str, Any]:
        """WorkflowUtils.parse_quality_validation_response 래퍼"""
        return WorkflowUtils.parse_quality_validation_response(response, self.logger)

    # 분류 관련 래퍼 메서드 (ClassificationHandler)
    @property
    def classification_handler(self):
        """
        ClassificationHandler 지연 로딩 Property
        
        LLM과 llm_fast를 필요로 하므로 실제 사용 시점에 초기화하여 Component Initialization 시간 단축
        """
        if not self._classification_handler_initialized:
            try:
                # LLM 초기화 상태 확인
                llm_available = self.llm is not None
                llm_fast_available = self.llm_fast is not None
                
                if not llm_available:
                    # LLM 초기화 재시도 로직 추가
                    self.logger.warning(
                        "ClassificationHandler not available: llm is None. "
                        "Attempting to reinitialize LLM..."
                    )
                    
                    # LLM 재초기화 시도
                    try:
                        # Config is not used (config is accessed via self.config)
                        # from core.utils.config import Config
                        if hasattr(self, 'config') and hasattr(self.config, 'llm'):
                            # config에서 llm 가져오기 시도
                            if hasattr(self.config, 'get_llm'):
                                self.llm = self.config.get_llm()
                                llm_available = self.llm is not None
                                if llm_available:
                                    self.logger.debug("✅ LLM reinitialized successfully from config")
                    except Exception as e:
                        self.logger.debug(f"LLM reinitialization attempt failed: {e}")
                    
                    if not llm_available:
                        self.logger.warning(
                            "ClassificationHandler not available: llm is None. "
                            "Please check LLM configuration and ensure llm/llm_fast are properly initialized. "
                            "Using fallback classification."
                        )
                        self._classification_handler = None
                        self._classification_handler_initialized = True
                        return None
                
                from core.classification.handlers.classification_handler import ClassificationHandler
                
                # ClassificationHandler 초기화
                self._classification_handler = ClassificationHandler(
                    llm=self.llm,
                    llm_fast=self.llm_fast if llm_fast_available else self.llm,
                    stats=getattr(self, 'stats', None),
                    logger=self.logger
                )
                self._classification_handler_initialized = True
                self.logger.debug(
                    f"ClassificationHandler lazy-loaded successfully "
                    f"(llm={'available' if llm_available else 'None'}, "
                    f"llm_fast={'available' if llm_fast_available else 'using llm'})"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to lazy-load ClassificationHandler: {e}. "
                    "Please check LLM configuration and ensure llm/llm_fast are properly initialized."
                )
                self._classification_handler = None
                self._classification_handler_initialized = True
        return self._classification_handler
    
    def _classify_with_llm(self, query: str) -> Tuple[QuestionType, float]:
        """ClassificationHandler.classify_with_llm 래퍼"""
        if not self.classification_handler:
            return self._fallback_classification(query)
        return self.classification_handler.classify_with_llm(query)
    
    def _fallback_classification(self, query: str) -> Tuple[QuestionType, float]:
        """ClassificationHandler.fallback_classification 래퍼"""
        try:
            handler = self.classification_handler
            if handler:
                return handler.fallback_classification(query)
        except (AttributeError, Exception) as e:
            self.logger.debug(f"ClassificationHandler fallback_classification failed: {e}")
        
        # 직접 폴백 분류 수행
        self.logger.warning(
            "ClassificationHandler not available, using direct fallback. "
            "Please check LLM configuration and ensure llm/llm_fast are properly initialized."
        )
        query_lower = query.lower() if query else ""
        if any(k in query_lower for k in ["판례", "사건", "판결"]):
            return QuestionType.PRECEDENT_SEARCH, 0.7
        elif any(k in query_lower for k in ["법률", "조문", "법령", "규정"]):
            return QuestionType.LAW_INQUIRY, 0.7
        elif any(k in query_lower for k in ["절차", "방법", "대응"]):
            return QuestionType.PROCEDURE_GUIDE, 0.7
        else:
            return QuestionType.GENERAL_QUESTION, 0.7


    def _fallback_complexity_classification(self, query: str) -> Tuple[QueryComplexity, bool]:
        """ClassificationHandler.fallback_complexity_classification 래퍼"""
        try:
            handler = self.classification_handler
            if handler:
                return handler.fallback_complexity_classification(query)
        except (AttributeError, Exception) as e:
            self.logger.debug(f"ClassificationHandler fallback_complexity_classification failed: {e}")
        
        # 직접 폴백 복잡도 분류 수행
        self.logger.warning(
            "ClassificationHandler not available, using direct fallback for complexity. "
            "Please check LLM configuration and ensure llm/llm_fast are properly initialized."
        )
        return QueryComplexity.MODERATE, True

    def _parse_complexity_response(self, response: str) -> Optional[Dict[str, Any]]:
        """ClassificationHandler.parse_complexity_response 래퍼"""
        if not self.classification_handler:
            self.logger.warning("ClassificationHandler not available for parse_complexity_response")
            return None
        return self.classification_handler.parse_complexity_response(response)

    def _parse_unified_classification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """ClassificationHandler.parse_unified_classification_response 래퍼"""
        if not self.classification_handler:
            self.logger.warning("ClassificationHandler not available for parse_unified_classification_response")
            return None
        return self.classification_handler.parse_unified_classification_response(response)

    # ============================================================================
    # _classify_query_with_chain 헬퍼 메서드들 (메서드 분해)
    # ============================================================================
    
    def _build_classification_chain_steps(self, query: str) -> List[Dict[str, Any]]:
        """분류 체인 스텝 정의 (WorkflowPromptBuilder 사용)"""
        return self.workflow_prompt_builder.build_classification_chain_steps(
            query=query,
            build_question_type_prompt_func=self.workflow_prompt_builder.build_question_type_prompt,
            build_legal_field_prompt_func=self.workflow_prompt_builder.build_legal_field_prompt,
            build_complexity_prompt_func=self.workflow_prompt_builder.build_complexity_prompt,
            build_search_necessity_prompt_func=self.workflow_prompt_builder.build_search_necessity_prompt
        )
    
    def _extract_chain_results(self, chain_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """체인 실행 결과 추출 (WorkflowUtils 사용)"""
        return WorkflowUtils.extract_chain_results(chain_history)
    
    def _convert_chain_results(
        self,
        question_type_result: Dict[str, Any],
        complexity_result: Dict[str, Any],
        search_necessity_result: Dict[str, Any]
    ) -> Tuple[QuestionType, float, QueryComplexity, bool]:
        """체인 결과를 반환 형식으로 변환 (WorkflowUtils 사용)"""
        return WorkflowUtils.convert_chain_results(
            question_type_result=question_type_result,
            complexity_result=complexity_result,
            search_necessity_result=search_necessity_result
        )

    def _classify_query_with_chain(self, query: str) -> Tuple[QuestionType, float, QueryComplexity, bool]:
        """
        단일 통합 프롬프트로 질문 분류 (최적화: 4회 → 1회 LLM 호출)
        
        기존 체인 방식(4회 LLM 호출) 대신 단일 통합 프롬프트를 사용하여
        질문 유형, 복잡도, 검색 필요성을 한 번에 분류합니다.
        
        Returns:
            Tuple[QuestionType, float, QueryComplexity, bool]: (질문 유형, 신뢰도, 복잡도, 검색 필요 여부)
        """
        try:
            cache_key = f"query_and_complexity:{query}"

            if cache_key in self._classification_cache:
                self.logger.debug(f"Using cached unified classification for: {query[:50]}...")
                if hasattr(self, 'stats'):
                    self.stats['complexity_cache_hits'] = self.stats.get('complexity_cache_hits', 0) + 1
                return self._classification_cache[cache_key]

            if hasattr(self, 'stats'):
                self.stats['complexity_cache_misses'] = self.stats.get('complexity_cache_misses', 0) + 1

            start_time = time.time()

            # 단일 통합 프롬프트 사용 (4회 → 1회 호출로 최적화)
            result_tuple = self.classification_handler.classify_query_and_complexity_with_llm(query)

            elapsed_time = time.time() - start_time

            # 성능 메트릭 업데이트
            if hasattr(self, 'stats'):
                self.stats['unified_classification_calls'] = self.stats.get('unified_classification_calls', 0) + 1
                self.stats['unified_classification_llm_calls'] = self.stats.get('unified_classification_llm_calls', 0) + 1
                current_avg = self.stats.get('avg_unified_classification_time', 0.0)
                count = self.stats.get('unified_classification_calls', 1)
                self.stats['avg_unified_classification_time'] = (current_avg * (count - 1) + elapsed_time) / count
                self.stats['total_unified_classification_time'] = self.stats.get('total_unified_classification_time', 0.0) + elapsed_time

            self.logger.debug(
                f"✅ [UNIFIED CLASSIFICATION] "
                f"question_type={result_tuple[0].value}, complexity={result_tuple[2].value}, "
                f"needs_search={result_tuple[3]}, confidence={result_tuple[1]:.2f}, "
                f"(시간: {elapsed_time:.3f}s, LLM 호출: 1회)"
            )

            # 캐시 크기 제한
            if len(self._classification_cache) >= 200:
                oldest_key = next(iter(self._classification_cache))
                del self._classification_cache[oldest_key]
                self.logger.debug(f"[CACHE] Removed oldest classification cache entry: {oldest_key[:50]}")

            self._classification_cache[cache_key] = result_tuple

            return result_tuple

        except Exception as e:
            # 예외 타입별 원인 분류
            error_type = type(e).__name__
            error_message = str(e)
            
            if "timeout" in error_message.lower() or "timed out" in error_message.lower():
                fallback_reason = "LLM timeout"
            elif "network" in error_message.lower() or "connection" in error_message.lower():
                fallback_reason = "Network error"
            elif "rate limit" in error_message.lower() or "429" in error_message:
                fallback_reason = "Rate limit"
            elif "api" in error_message.lower() or "key" in error_message.lower():
                fallback_reason = "API error"
            else:
                fallback_reason = f"Exception: {error_type}"
            
            self.logger.warning(
                f"Unified classification failed: {fallback_reason} ({error_type}: {error_message}), using fallback"
            )
            if hasattr(self, 'stats') and self.stats:
                self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
                # 폴백 원인 분류
                if 'fallback_reasons' not in self.stats:
                    self.stats['fallback_reasons'] = {}
                self.stats['fallback_reasons'][fallback_reason] = self.stats['fallback_reasons'].get(fallback_reason, 0) + 1
            question_type, confidence = self.classification_handler.fallback_classification(query)
            complexity, needs_search = self._fallback_complexity_classification(query)
            return (question_type, confidence, complexity, needs_search)

    def _parse_question_type_response(self, response: str) -> Dict[str, Any]:
        """질문 유형 분류 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "question_type": "general_question",
                "confidence": 0.7,
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse question type response: {e}")
            return {
                "question_type": "general_question",
                "confidence": 0.7,
                "reasoning": f"파싱 에러: {e}"
            }

    def _parse_legal_field_response(self, response: str) -> Optional[Dict[str, Any]]:
        """법률 분야 추출 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse legal field response: {e}")
            return None

    def _parse_complexity_response(self, response: str) -> Dict[str, Any]:
        """복잡도 평가 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            # 기본값
            return {
                "complexity": "moderate",
                "confidence": 0.7,
                "reasoning": "JSON 파싱 실패"
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse complexity response: {e}")
            return {
                "complexity": "moderate",
                "confidence": 0.7,
                "reasoning": f"파싱 에러: {e}"
            }

    def _parse_search_necessity_response(self, response: str) -> Optional[Dict[str, Any]]:
        """검색 필요성 판단 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse search necessity response: {e}")
            return None

    def _classify_query_and_complexity_with_llm(self, query: str) -> Tuple[QuestionType, float, QueryComplexity, bool]:
        """ClassificationHandler.classify_query_and_complexity_with_llm 래퍼"""
        return self.classification_handler.classify_query_and_complexity_with_llm(query)

    def _classify_complexity_with_llm(self, query: str, query_type: str = "") -> Tuple[QueryComplexity, bool]:
        """ClassificationHandler.classify_complexity_with_llm 래퍼"""
        return self.classification_handler.classify_complexity_with_llm(query, query_type)

    # QueryUtilsMixin으로 이동됨

    # Phase 2 리팩토링: 검색 관련 메서드는 SearchHandler로 이동됨
    # 호환성을 위한 래퍼 메서드

    def _keyword_search(
        self,
        query: str,
        query_type_str: str,
        limit: Optional[int] = None,
        legal_field: str = "",
        extracted_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """향상된 키워드 기반 검색"""
        try:
            category_mapping = self._get_category_mapping()
            categories_to_search = category_mapping.get(query_type_str, ["civil_law"])

            # 지원되는 법률 분야만 매핑 (민사법, 지식재산권법, 행정법, 형사법만)
            field_category_map = {
                "civil": "civil_law",
                "criminal": "criminal_law",
                "intellectual_property": "intellectual_property",
                "administrative": "administrative_law"
            }

            preferred_category = None
            if legal_field and legal_field in field_category_map:
                preferred_category = field_category_map[legal_field]
                if preferred_category in categories_to_search:
                    categories_to_search.remove(preferred_category)
                    categories_to_search.insert(0, preferred_category)

            keyword_results = []
            search_limit = limit if limit is not None else WorkflowConstants.CATEGORY_SEARCH_LIMIT

            # 확장된 키워드를 쿼리에 추가
            enhanced_query = query
            if extracted_keywords and len(extracted_keywords) > 0:
                # 상위 3개 키워드만 추가 (쿼리가 너무 길어지지 않도록)
                safe_keywords = [kw for kw in extracted_keywords[:3] if isinstance(kw, str)]
                if safe_keywords:
                    enhanced_query = f"{query} {' '.join(safe_keywords)}"

            self.logger.debug(f"_keyword_search: Searching {len(categories_to_search)} categories with query='{enhanced_query[:50]}...', limit={search_limit}")

            for category in categories_to_search:
                # 키워드 검색은 항상 FTS5 검색 수행 (force_fts=True)
                self.logger.debug(f"_keyword_search: Searching category={category}")
                category_docs = self.data_connector.search_documents(
                    enhanced_query, category, limit=search_limit, force_fts=True
                )
                self.logger.debug(f"_keyword_search: Category {category} returned {len(category_docs)} documents")

                for doc in category_docs:
                    doc['search_type'] = 'keyword'
                    doc['category'] = category
                    # 카테고리 일치도 점수 추가
                    if preferred_category and category == preferred_category:
                        doc['category_boost'] = 1.2
                    else:
                        doc['category_boost'] = 1.0

                keyword_results.extend(category_docs)
                self.logger.debug(f"Found {len(category_docs)} documents in category: {category}")

            self.logger.debug(f"_keyword_search: Total results={len(keyword_results)}")
            return keyword_results, len(keyword_results)
        except Exception as e:
            self.logger.warning(f"Keyword search failed: {e}")
            return [], 0

    def _merge_and_rerank_search_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        query: str,
        optimized_queries: Dict[str, Any],
        rerank_params: Dict[str, Any]
    ) -> List[Dict]:
        """SearchHandler.merge_and_rerank_search_results 래퍼"""
        return self.search_handler.merge_and_rerank_search_results(
            semantic_results, keyword_results, query, optimized_queries, rerank_params
        )




    def _fallback_search(self, state: LegalWorkflowState) -> None:
        """SearchHandler.fallback_search 래퍼"""
        self.search_handler.fallback_search(state)




    @with_state_optimization("process_legal_terms", enable_reduction=True)
    def process_legal_terms(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """법률 용어 추출 및 통합 (문서 검색 후, 답변 생성 전)"""
        try:
            # 방법 1: 노드 호출 추적 - 실행 기록 남기기
            # 직접 설정하여 상태 최적화로 인한 손실 방지
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"] = dict(state["metadata"])  # 복사본 생성
            state["metadata"]["_last_executed_node"] = "process_legal_terms"
            # common 그룹에도 설정 (nested 구조 지원)
            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "process_legal_terms"

            start_time = time.time()

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            all_terms = self._extract_terms_from_documents(retrieved_docs)
            self.logger.debug(f"추출된 용어 수: {len(all_terms)}")

            # 기존 재시도 카운터 보존 (메타데이터 업데이트 시 손실 방지)
            # 강화된 로깅으로 상태 확인
            existing_metadata_direct = state.get("metadata", {})
            existing_metadata_common = state.get("common", {}).get("metadata", {}) if isinstance(state.get("common"), dict) else {}
            existing_metadata = existing_metadata_direct if isinstance(existing_metadata_direct, dict) else {}
            if isinstance(existing_metadata_common, dict):
                existing_metadata = {**existing_metadata, **existing_metadata_common}

            existing_top_level = state.get("retry_count", 0)

            # 기존 재시도 카운터 저장
            saved_gen_retry = max(
                existing_metadata.get("generation_retry_count", 0),
                existing_top_level
            )
            saved_val_retry = existing_metadata.get("validation_retry_count", 0)

            self.logger.debug(
                f"[METADATA READ] process_legal_terms: gen_retry={saved_gen_retry}, "
                f"val_retry={saved_val_retry}, top_level={existing_top_level}"
            )

            if all_terms:
                representative_terms = self._integrate_and_process_terms(all_terms)
                metadata = dict(existing_metadata)  # 기존 메타데이터 복사
                metadata["extracted_terms"] = representative_terms
                metadata["total_terms_extracted"] = len(all_terms)
                metadata["unique_terms"] = len(representative_terms)
                # 재시도 카운터 보존
                metadata["generation_retry_count"] = saved_gen_retry
                metadata["validation_retry_count"] = saved_val_retry
                metadata["_last_executed_node"] = "process_legal_terms"
                state["metadata"] = metadata
                # common 그룹에도 동기화
                if "common" not in state or not isinstance(state.get("common"), dict):
                    state["common"] = {}
                if "metadata" not in state["common"]:
                    state["common"]["metadata"] = {}
                state["common"]["metadata"].update(metadata)
                # 최상위 레벨에도 저장
                state["retry_count"] = saved_gen_retry

                self.logger.debug(
                    f"[METADATA SAVE] process_legal_terms: gen_retry={saved_gen_retry}, "
                    f"val_retry={saved_val_retry} (preserved)"
                )

                self._set_state_value(state, "metadata", metadata)
                self._add_step(state, "용어 통합 완료", f"용어 통합 완료: {len(representative_terms)}개")
                self.logger.debug(f"통합된 용어 수: {len(representative_terms)}")
            else:
                metadata = dict(existing_metadata)  # 기존 메타데이터 복사
                metadata["extracted_terms"] = []
                # 재시도 카운터 보존
                metadata["generation_retry_count"] = saved_gen_retry
                metadata["validation_retry_count"] = saved_val_retry
                metadata["_last_executed_node"] = "process_legal_terms"
                state["metadata"] = metadata
                # common 그룹에도 동기화
                if "common" not in state or not isinstance(state.get("common"), dict):
                    state["common"] = {}
                if "metadata" not in state["common"]:
                    state["common"]["metadata"] = {}
                state["common"]["metadata"].update(metadata)
                # 최상위 레벨에도 저장
                state["retry_count"] = saved_gen_retry

                self.logger.debug(
                    f"[METADATA SAVE] process_legal_terms (no terms): gen_retry={saved_gen_retry}, "
                    f"val_retry={saved_val_retry} (preserved)"
                )

                self._set_state_value(state, "metadata", metadata)
                self._add_step(state, "용어 추출 없음", "용어 추출 없음 (문서 내용 부족)")

            # 반환 직전 최종 상태 확인
            final_meta_direct = state.get("metadata", {})
            final_meta_common = state.get("common", {}).get("metadata", {}) if isinstance(state.get("common"), dict) else {}
            final_gen_check = final_meta_direct.get("generation_retry_count", 0) if isinstance(final_meta_direct, dict) else 0
            final_gen_common = final_meta_common.get("generation_retry_count", 0) if isinstance(final_meta_common, dict) else 0
            final_top_check = state.get("retry_count", 0)

            self.logger.debug(
                f"[METADATA FINAL] process_legal_terms: gen_retry={max(final_gen_check, final_gen_common, final_top_check)} "
                f"(metadata={final_gen_check}, common={final_gen_common}, top={final_top_check})"
            )

            self._update_processing_time(state, start_time)
        except Exception as e:
            self._handle_error(state, str(e), "법률 용어 처리 중 오류 발생")
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["extracted_terms"] = []
            self._set_state_value(state, "metadata", metadata)
        return state

    def _extract_terms_from_documents(self, docs: List[Dict]) -> List[str]:
        """문서에서 법률 용어 추출"""
        return DocumentExtractor.extract_terms_from_documents(docs)

    def _integrate_and_process_terms(self, all_terms: List[str]) -> List[str]:
        """용어 통합 및 처리"""
        processed_terms = self.term_integrator.integrate_terms(all_terms)
        return [term["representative_term"] for term in processed_terms]

    # ============================================================================
    def _get_metadata_safely(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """metadata를 안전하게 가져오기 (dict 타입 보장)"""
        metadata = self._get_state_value(state, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return metadata

    def _extract_doc_type(self, doc: Dict[str, Any]) -> str:
        """문서에서 타입 추출 (메타데이터 필드 기준만 사용)"""
        # DocumentType.from_metadata를 사용하여 메타데이터 필드 기준으로만 타입 추론
        doc_type_enum = DocumentType.from_metadata(doc)
        return doc_type_enum.value

    def _calculate_type_distribution(self, docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """문서 타입 분포 계산"""
        type_distribution = {}
        for doc in docs:
            doc_type = self._extract_doc_type(doc)
            type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
        return type_distribution

    def _has_precedent_or_decision(self, docs: List[Dict[str, Any]], check_precedent: bool = True, check_decision: bool = True) -> Tuple[bool, bool]:
        """판례 존재 여부 확인 (메타데이터 필드 기준)"""
        has_precedent = False
        has_decision = False  # 레거시 호환성을 위해 유지하지만 항상 False 반환
        
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            
            doc_type_enum = DocumentType.from_metadata(doc)
            doc_type = doc_type_enum.value
            
            # precedent_content는 판례로 인식
            if check_precedent and not has_precedent:
                if doc_type_enum == DocumentType.PRECEDENT_CONTENT:
                    has_precedent = True
                    self.logger.debug(f"🔀 [DIVERSITY] Found precedent document: type={doc_type}")
            
            if not check_precedent or has_precedent:
                break
        
        return has_precedent, has_decision

    def _extract_doc_content(self, doc: Dict[str, Any]) -> str:
        """문서 내용 추출 (강화된 버전)"""
        
        # 1. 기본 필드 확인
        content = doc.get("content") or doc.get("text") or doc.get("content_text")
        
        # 2. metadata에서 확인
        if not content:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                content = metadata.get("content") or metadata.get("text")
        
        # 3. content가 문자열이 아니면 변환 시도
        if content and not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = ""
        
        # 4. 내용이 비어있으면 DB에서 복원 시도
        if not content or len(content.strip()) < 10:
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
            chunk_id = doc.get("chunk_id")
            
            if doc_id or chunk_id:
                try:
                    if hasattr(self, 'search_handler') and self.search_handler:
                        semantic_engine = getattr(self.search_handler, 'semantic_search_engine', None)
                        if semantic_engine and hasattr(semantic_engine, '_ensure_text_content'):
                            restored_content = semantic_engine._ensure_text_content(doc)
                            if restored_content and len(restored_content.strip()) >= 10:
                                content = restored_content
                                doc["content"] = content
                                self.logger.debug(f"✅ [CONTENT RESTORE] 문서 내용 복원 성공: doc_id={doc_id}")
                except Exception as e:
                    self.logger.debug(f"문서 내용 복원 실패: {e}")
        
        # 5. 최종 검증
        if not content or len(content.strip()) < 10:
            self.logger.warning(
                f"⚠️ [CONTENT EXTRACT] 문서 내용 부족: "
                f"doc_id={doc.get('id', 'unknown')}, "
                f"content_len={len(content) if content else 0}, "
                f"keys={list(doc.keys())[:10]}"
            )
        
        return content or ""

    def _normalize_score(self, score: Optional[float], min_val: float = 0.0, max_val: float = 1.0) -> float:
        """점수를 0.0~1.0 범위로 정규화
        
        Args:
            score: 정규화할 점수
            min_val: 원본 점수의 최소값 (예상)
            max_val: 원본 점수의 최대값 (예상)
        
        Returns:
            0.0~1.0 범위의 정규화된 점수
        """
        if score is None:
            return 0.5  # 기본값
        
        score = float(score)
        
        # 이미 0.0~1.0 범위에 있으면 그대로 반환
        if 0.0 <= score <= 1.0:
            return score
        
        # 음수면 0.0으로 클리핑
        if score < 0.0:
            return 0.0
        
        # 1.0보다 크면 정규화 (예: Cross-Encoder 점수가 0~10 범위인 경우)
        if score > 1.0:
            # min-max 정규화
            if max_val > min_val:
                normalized = (score - min_val) / (max_val - min_val)
                # 0.0~1.0 범위로 클리핑
                return max(0.0, min(1.0, normalized))
            else:
                # max_val이 1.0 이하면 그냥 1.0으로 클리핑
                return 1.0
        
        return score
    
    def _normalize_scores_batch(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 문서의 점수를 일괄 정규화
        
        Args:
            docs: 점수를 정규화할 문서 리스트
        
        Returns:
            점수가 정규화된 문서 리스트
        """
        if not docs:
            return docs
        
        # 모든 점수 수집하여 최소/최대값 계산
        all_scores = []
        for doc in docs:
            score = doc.get("relevance_score") or doc.get("similarity") or doc.get("score")
            if score is not None:
                all_scores.append(float(score))
        
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            
            # 점수 범위가 1.0보다 크면 정규화 필요
            if max_score > 1.0 or min_score < 0.0:
                self.logger.debug(
                    f"📊 [SCORE NORMALIZATION] 점수 범위: {min_score:.3f}~{max_score:.3f}, "
                    f"정규화 적용: {len(docs)}개 문서"
                )
                
                for doc in docs:
                    relevance_score = doc.get("relevance_score") or doc.get("similarity") or doc.get("score")
                    if relevance_score is not None:
                        # min-max 정규화
                        if max_score > min_score:
                            normalized = (float(relevance_score) - min_score) / (max_score - min_score)
                            doc["relevance_score"] = max(0.0, min(1.0, normalized))
                            # similarity도 함께 정규화
                            if "similarity" in doc:
                                doc["similarity"] = doc["relevance_score"]
                            # score도 함께 정규화
                            if "score" in doc:
                                doc["score"] = doc["relevance_score"]
        
        return docs

    def _ensure_scores(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서에 점수가 있는지 확인하고 없으면 설정 (0.0~1.0 범위로 정규화)"""
        relevance_score = doc.get("relevance_score") or doc.get("similarity") or doc.get("score")
        final_weighted_score = doc.get("final_weighted_score")
        
        if relevance_score is None or relevance_score == 0.0:
            similarity = doc.get("similarity")
            if similarity is not None:
                relevance_score = float(similarity)
            else:
                # 기본값: 0.5 (중간 점수)
                relevance_score = 0.5
                self.logger.debug(
                    f"⚠️ [SCORE INIT] 점수 없음, 기본값 설정: "
                    f"doc_id={doc.get('id', 'unknown')}, "
                    f"score=0.5"
                )
        
        # 점수 정규화 (0.0~1.0 범위)
        relevance_score = self._normalize_score(relevance_score)
        
        if final_weighted_score is None:
            final_weighted_score = relevance_score
        else:
            # final_weighted_score도 정규화
            final_weighted_score = self._normalize_score(final_weighted_score)
        
        doc["relevance_score"] = float(relevance_score)
        doc["final_weighted_score"] = float(final_weighted_score)
        
        # similarity와 score도 함께 업데이트 (일관성 유지)
        if "similarity" not in doc or doc.get("similarity") != relevance_score:
            doc["similarity"] = float(relevance_score)
        if "score" not in doc or doc.get("score") != relevance_score:
            doc["score"] = float(relevance_score)
        
        return doc
    
    def _normalize_document_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서의 메타데이터를 정규화하고 최상위 필드로 복사"""
        if not isinstance(doc, dict):
            return doc
        
        metadata = doc.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            doc["metadata"] = metadata
        
        # 🔥 CRITICAL: metadata의 source_type을 type으로 변환 (레거시 호환)
        if metadata.get("source_type") and not doc.get("type"):
            doc["type"] = metadata.get("source_type")
            metadata["type"] = metadata.get("source_type")
        
        # metadata의 type을 최상위 필드로 복사
        if metadata.get("type") and not doc.get("type"):
            doc["type"] = metadata.get("type")
        
        # metadata의 법령/판례 관련 필드를 최상위 필드로 복사
        for key in self.METADATA_COPY_FIELDS:
            if metadata.get(key) and not doc.get(key):
                doc[key] = metadata.get(key)
        
        return doc
    
    def _normalize_document_type(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서의 type 필드를 정규화 (메타데이터 보존)"""
        if not isinstance(doc, dict):
            return doc
        
        # type이 없거나 unknown이면 metadata에서 복원
        current_type = doc.get("type", "").lower() if doc.get("type") else ""
        
        if not doc.get("type") or current_type == "unknown":
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                # 우선순위: metadata.type > metadata.source_type > DocumentType.from_metadata
                metadata_type = metadata.get("type") or metadata.get("source_type")
                if metadata_type and metadata_type.lower() != "unknown":
                    doc["type"] = metadata_type
                    # metadata에도 type으로 저장
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    if not isinstance(doc["metadata"], dict):
                        doc["metadata"] = {}
                    doc["metadata"]["type"] = metadata_type
                elif not doc.get("type"):
                    # DocumentType.from_metadata로 추론
                    from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                    doc_type_enum = DocumentType.from_metadata(doc)
                    if doc_type_enum != DocumentType.UNKNOWN:
                        doc["type"] = doc_type_enum.value
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        if not isinstance(doc["metadata"], dict):
                            doc["metadata"] = {}
                        doc["metadata"]["type"] = doc_type_enum.value
        
        return doc
    
    def _normalize_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 리스트의 메타데이터와 type 필드를 정규화"""
        if not docs:
            return docs
        
        normalized_docs = []
        for doc in docs:
            if not isinstance(doc, dict):
                normalized_docs.append(doc)
                continue
            
            # 메타데이터 정규화
            doc = self._normalize_document_metadata(doc)
            # type 필드 정규화
            doc = self._normalize_document_type(doc)
            
            normalized_docs.append(doc)
        
        return normalized_docs
    
    def _extract_citations(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """인용 추출 (강화된 버전 - 더 많은 문서에서 추출)"""
        self.logger.debug(
            f"🔍 [CITATION EXTRACTION] 시작: retrieved_docs={len(retrieved_docs)}개"
        )
        
        citations = []
        seen_citations = set()  # 중복 방지
        
        import re
        law_pattern = re.compile(r'([가-힣]+법)\s*제?\s*(\d+)\s*조')
        precedent_pattern = re.compile(r'([가-힣]+(?:지방)?법원|대법원).*?(\d{4}[다나마]\d+)')
        
        for idx, doc in enumerate(retrieved_docs, 1):
            # type 정보 복구 시도 (여러 위치에서 확인)
            doc_type = doc.get("type", "")
            
            # type이 없거나 "unknown"이면 metadata에서 복구 시도
            if not doc_type or doc_type == "unknown":
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    # 여러 필드에서 타입 확인
                    doc_type = (
                        metadata.get("type") or 
                        metadata.get("document_type") or
                        doc_type
                    )
                    if doc_type and doc_type != "unknown":
                        doc["type"] = doc_type
            
            # 여전히 없으면 DocumentType.from_metadata로 메타데이터 필드 기반 추론
            if not doc_type or doc_type == "unknown":
                doc_type_enum = DocumentType.from_metadata(doc)
                doc_type = doc_type_enum.value
                if doc_type != "unknown":
                    doc["type"] = doc_type
            
            content = doc.get("content", "") or doc.get("text", "") or doc.get("content_text", "")
            
            # 디버깅: 문서의 모든 키 확인
            doc_keys = list(doc.keys()) if isinstance(doc, dict) else []
            self.logger.debug(
                f"🔍 [CITATION DEBUG] 문서 {idx}/{len(retrieved_docs)}: "
                f"keys={doc_keys[:10]}, "
                f"type={doc_type}, "
                f"has_type_field={'type' in doc_keys}, "
                f"metadata_type={doc.get('metadata', {}).get('type') if isinstance(doc.get('metadata'), dict) else 'N/A'}"
            )
            self.logger.debug(
                f"🔍 [CITATION] 문서 {idx}/{len(retrieved_docs)} 처리 중: "
                f"keys={doc_keys[:10]}, "
                f"type={doc_type}, "
                f"content_length={len(content) if content else 0}, "
                f"has_metadata={bool(doc.get('metadata'))}, "
                f"source={doc.get('source', 'N/A')[:50]}"
            )
            
            # DocumentType Enum으로 변환
            doc_type_enum = DocumentType.from_metadata(doc)
            doc_type = doc_type_enum.value
            
            # 1. 법령 조문 인용 (타입 기반)
            if doc_type_enum == DocumentType.STATUTE_ARTICLE:
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: statute_article 타입 감지, 필드 확인 중..."
                )
                # law_name 추출 강화 (여러 위치 확인)
                metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                law_name = (
                    doc.get("statute_name") or 
                    doc.get("law_name") or 
                    metadata.get("statute_name") or
                    metadata.get("law_name") or
                    metadata.get("statute_abbrv")
                )
                
                # article_no 추출 강화 (여러 위치 확인)
                article_no = (
                    doc.get("article_no") or 
                    doc.get("article_number") or 
                    metadata.get("article_no") or
                    metadata.get("article_number")
                )
                
                # 여전히 없으면 source나 content에서 파싱 시도
                if not law_name or not article_no:
                    source = doc.get("source", "")
                    content = doc.get("content", "") or doc.get("text", "")
                    combined_text = f"{source} {content[:500]}"
                    
                    # 법령명과 조문번호 패턴 매칭
                    law_article_match = re.search(r'([가-힣]+법)\s*제\s*(\d+)\s*조', combined_text)
                    if law_article_match:
                        if not law_name:
                            law_name = law_article_match.group(1)
                        if not article_no:
                            article_no = law_article_match.group(2)
                
                # TASK 11: article_no 정규화 개선 (선행 0만 제거, 중간/후행 0은 유지)
                # 데이터베이스 형식:
                #   - "제750조" = "075000" (앞에 0, 뒤에 000) -> "75000" (선행 0만 제거)
                #   - "제75조" = "007500" (앞에 00, 뒤에 00) -> "7500" (선행 0만 제거)
                #   - "제537조" = "053700" (앞에 0, 뒤에 00) -> "53700" (선행 0만 제거)
                # 하지만 실제로는 content에서 확인하여 정확한 번호 사용
                if article_no:
                    article_no_str = str(article_no).strip()
                    original_article_no = article_no_str  # 디버깅용
                    
                    # content에서 실제 조문 번호 확인 (정규화 검증용)
                    content = doc.get("content", "") or doc.get("text", "")
                    content_article_match = None
                    if content:
                        content_article_match = re.search(r'제\s*(\d+)\s*조', content[:200])
                    
                    # TASK 11: 선행 0만 제거 (중간/후행 0은 유지)
                    # 예: '007500' → '7500', '075000' → '75000', '000123' → '123'
                    # 🔥 개선: 6자리 형식 처리 (예: '075000' = 법령ID(3자리) + 조문번호(3자리) 형식일 수 있음)
                    article_no_clean = article_no_str.lstrip('0')
                    
                    # 모두 0이면 '0' 반환
                    if not article_no_clean:
                        article_no = "0"
                    else:
                        # 🔥 개선: 6자리 형식 처리 (075000 → 750)
                        # 데이터베이스에서 6자리 형식으로 저장된 경우 (법령ID 3자리 + 조문번호 3자리)
                        # 예: '075000' → '750', '001234' → '1234'
                        if len(article_no_str) == 6 and article_no_str.isdigit():
                            # 앞의 3자리(법령ID)를 제거하고 뒤의 3자리(조문번호)만 사용
                            # 단, 앞의 3자리가 모두 0이 아니면 조문번호로 간주
                            first_three = article_no_str[:3]
                            last_three = article_no_str[3:]
                            
                            # 앞의 3자리가 모두 0이면 뒤의 3자리를 조문번호로 사용
                            if first_three == '000':
                                last_three_clean = last_three.lstrip('0')
                                if last_three_clean:
                                    article_no_clean = last_three_clean
                                    self.logger.debug(
                                        f"🔍 [CITATION NORMALIZE] 6자리 형식 처리: "
                                        f"'{article_no_str}' -> '{article_no_clean}' (뒤의 3자리 사용)"
                                    )
                                else:
                                    article_no_clean = "0"
                            # 앞의 3자리가 0이 아니면 전체를 조문번호로 간주 (예: '123456' → '123456')
                            # 하지만 일반적으로 조문번호는 4자리 이하이므로, content 기반 검증 사용
                        
                        # 🔥 개선: 정규화된 번호가 범위를 초과하면 content 기반으로 재시도
                        try:
                            clean_num = int(article_no_clean)
                            if clean_num > 9999 and content_article_match:
                                # 범위 초과 시 content에서 추출한 번호 사용
                                content_article = content_article_match.group(1).lstrip('0')
                                if content_article and content_article.isdigit():
                                    content_num = int(content_article)
                                    if 1 <= content_num <= 9999:
                                        self.logger.debug(
                                            f"🔍 [CITATION NORMALIZE] 범위 초과로 content 기반 수정: "
                                            f"'{article_no_clean}' -> '{content_article}' (content: {content[:50]})"
                                        )
                                        article_no_clean = content_article
                        except ValueError:
                            pass
                    
                    if not article_no_clean or article_no_clean == "0":
                        article_no = "0"
                    else:
                        # 개선: content 기반 수정은 신중하게 적용
                        # content가 잘못되어 있을 수 있으므로, 원본 article_no를 우선 사용
                        if content_article_match:
                            content_article = content_article_match.group(1).lstrip('0')
                            if content_article:
                                # content의 조문 번호와 정규화된 번호 비교
                                # 차이가 크면 (예: 7500 vs 75) content가 잘못된 것으로 간주하고 원본 사용
                                try:
                                    clean_num = int(article_no_clean)
                                    content_num = int(content_article)
                                    # 🔥 개선: 10배 차이 체크 강화 (절대값 차이로도 확인)
                                    diff_ratio = max(clean_num, content_num) / min(clean_num, content_num) if min(clean_num, content_num) > 0 else 0
                                    abs_diff = abs(clean_num - content_num)
                                    
                                    # 10배 이상 차이거나, 절대값 차이가 큰 경우 content가 잘못된 것으로 간주
                                    if clean_num > 0 and (diff_ratio >= 10 or (abs_diff >= clean_num * 9)):
                                        # 10배 관계면 content가 잘못된 것으로 간주 (예: 7500 -> 75)
                                        self.logger.debug(
                                            f"🔍 [CITATION NORMALIZE] content 기반 수정 스킵 (10배 차이): "
                                            f"'{article_no_clean}' vs '{content_article}' (ratio={diff_ratio:.1f}, diff={abs_diff}, content: {content[:50]})"
                                        )
                                        article_no = article_no_clean
                                    elif content_article == article_no_clean:
                                        # 일치하면 정규화된 값 사용
                                        article_no = article_no_clean
                                    elif abs_diff < 10:
                                        # 작은 차이(10 미만)면 content 기준 사용 (예: 750 vs 751)
                                        self.logger.debug(
                                            f"🔍 [CITATION NORMALIZE] content 기반 수정 (작은 차이): "
                                            f"'{article_no_clean}' -> '{content_article}' (diff={abs_diff}, content: {content[:50]})"
                                        )
                                        article_no = content_article
                                    else:
                                        # 중간 차이면 원본 정규화된 값 사용 (content 신뢰 불가)
                                        self.logger.debug(
                                            f"🔍 [CITATION NORMALIZE] content 기반 수정 스킵 (중간 차이): "
                                            f"'{article_no_clean}' vs '{content_article}' (diff={abs_diff}, content: {content[:50]})"
                                        )
                                        article_no = article_no_clean
                                except ValueError:
                                    # 숫자 변환 실패 시 정규화된 값 사용
                                    article_no = article_no_clean
                            else:
                                article_no = article_no_clean
                        else:
                            # content에서 확인 불가능하면 선행 0만 제거한 값 사용
                            article_no = article_no_clean
                    
                    # TASK 11: 형식 검증 추가
                    if article_no and article_no.isdigit():
                        try:
                            num = int(article_no)
                            # 1~9999 범위 검증
                            if not (1 <= num <= 9999):
                                self.logger.warning(
                                    f"⚠️ [ARTICLE NO] 조문 번호 범위 초과: {article_no} (1~9999 범위 아님). "
                                    f"원본 사용: {original_article_no}"
                                )
                                article_no = original_article_no
                        except ValueError:
                            self.logger.warning(
                                f"⚠️ [ARTICLE NO] 조문 번호 형식 오류: {article_no}. "
                                f"원본 사용: {original_article_no}"
                            )
                            article_no = original_article_no
                    
                    # 디버깅: 정규화 결과 로깅
                    if original_article_no != article_no:
                        self.logger.debug(
                            f"🔍 [CITATION NORMALIZE] article_no 정규화: "
                            f"'{original_article_no}' -> '{article_no}'"
                        )
                
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: law_name={law_name}, article_no={article_no}"
                )
                
                if law_name and article_no:
                    citation_key = f"{law_name}_{article_no}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "statute",
                            "law_name": law_name,
                            "article_no": article_no,
                            "source": doc.get("source", ""),
                            "doc_id": doc.get("id")
                        })
                        seen_citations.add(citation_key)
                        self.logger.debug(
                            f"✅ [CITATION] 문서 {idx}: 타입 기반 법령 추출 성공 - {law_name} 제{article_no}조"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ [CITATION] 문서 {idx}: statute_article 타입이지만 필드 부족 "
                        f"(law_name={bool(law_name)}, article_no={bool(article_no)})"
                    )
            
            # 2. 판례 인용 (타입 기반)
            elif doc_type_enum == DocumentType.PRECEDENT_CONTENT:
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: precedent_content 타입 감지, 필드 확인 중..."
                )
                case_name = (
                    doc.get("casenames") or 
                    doc.get("case_name") or 
                    doc.get("metadata", {}).get("casenames")
                )
                court = (
                    doc.get("court") or 
                    doc.get("metadata", {}).get("court")
                )
                decision_date = (
                    doc.get("decision_date") or 
                    doc.get("metadata", {}).get("decision_date")
                )
                
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: case_name={case_name}, court={court}, "
                    f"decision_date={decision_date}"
                )
                
                if case_name:
                    citation_key = f"{case_name}_{court}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "precedent",
                            "case_name": case_name,
                            "court": court or "법원",
                            "decision_date": decision_date,
                            "source": doc.get("source", ""),
                            "doc_id": doc.get("id")
                        })
                        seen_citations.add(citation_key)
                        self.logger.debug(
                            f"✅ [CITATION] 문서 {idx}: 타입 기반 판례 추출 성공 - {case_name}"
                        )
                else:
                    # 폴백: case_name이 없을 때 다른 필드에서 추출 시도
                    fallback_case_name = (
                        doc.get("source", "") or
                        doc.get("title", "") or
                        doc.get("metadata", {}).get("title", "") or
                        doc.get("metadata", {}).get("source", "") or
                        ""
                    )
                    
                    # content에서 사건명 패턴 추출 시도
                    if not fallback_case_name:
                        content = doc.get("content", "") or doc.get("text", "")
                        if isinstance(content, str) and content:
                            import re
                            # 판례 패턴: "○○ 사건", "○○ 사건의", "○○ 사건에서" 등
                            case_patterns = [
                                r'([가-힣\s]+사건)',
                                r'([가-힣\s]+건)',
                                r'([가-힣\s]+소송)',
                            ]
                            for pattern in case_patterns:
                                match = re.search(pattern, content[:500])  # 처음 500자만 검색
                                if match:
                                    fallback_case_name = match.group(1).strip()
                                    break
                    
                    if fallback_case_name:
                        citation_key = f"{fallback_case_name}_{court}"
                        if citation_key not in seen_citations:
                            citations.append({
                                "type": "precedent",
                                "case_name": fallback_case_name,
                                "court": court or "법원",
                                "decision_date": decision_date,
                                "source": doc.get("source", ""),
                                "doc_id": doc.get("id"),
                                "fallback": True  # 폴백으로 추출된 경우 표시
                            })
                            seen_citations.add(citation_key)
                            self.logger.debug(
                                f"✅ [CITATION] 문서 {idx}: 폴백으로 case_name 추출 성공 - {fallback_case_name}"
                            )
                    else:
                        self.logger.warning(
                            f"⚠️ [CITATION] 문서 {idx}: precedent_content 타입이지만 case_name 없음 (폴백도 실패)"
                        )
            
            # 3. 개선: 문서 내용에서 직접 법령/판례 패턴 추출 (타입 기반과 독립적으로 수행)
            if content and isinstance(content, str):
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: 패턴 기반 추출 시도 (content_length={len(content)})"
                )
                
                # 법령 조문 패턴 추출 (각 문서에서 최대 3개까지)
                law_matches = law_pattern.findall(content)
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: 법령 패턴 매칭 결과={len(law_matches)}개"
                )
                
                for law_name_match, article_no_match in law_matches[:3]:
                    citation_key = f"{law_name_match}_{article_no_match}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "statute",
                            "law_name": law_name_match,
                            "article_no": article_no_match,
                            "source": doc.get("source", ""),
                            "doc_id": doc.get("id"),
                            "extracted_from": "content_pattern"
                        })
                        seen_citations.add(citation_key)
                        self.logger.debug(
                            f"✅ [CITATION] 문서 {idx}: 패턴 기반 법령 추출 성공 - "
                            f"{law_name_match} 제{article_no_match}조"
                        )
                
                # 판례 패턴 추출 (각 문서에서 최대 2개까지)
                precedent_matches = precedent_pattern.findall(content)
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: 판례 패턴 매칭 결과={len(precedent_matches)}개"
                )
                
                for court_match, case_no_match in precedent_matches[:2]:
                    citation_key = f"{court_match}_{case_no_match}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "precedent",
                            "case_name": f"{court_match} {case_no_match}",
                            "court": court_match,
                            "source": doc.get("source", ""),
                            "doc_id": doc.get("id"),
                            "extracted_from": "content_pattern"
                        })
                        seen_citations.add(citation_key)
                        self.logger.debug(
                            f"✅ [CITATION] 문서 {idx}: 패턴 기반 판례 추출 성공 - "
                            f"{court_match} {case_no_match}"
                        )
            else:
                self.logger.debug(
                    f"⚠️ [CITATION] 문서 {idx}: 내용 없음 또는 문자열이 아님 "
                    f"(content={bool(content)}, type={type(content).__name__})"
                )
        
        # 문서 타입별 통계
        doc_types = {}
        for doc in retrieved_docs:
            doc_type = doc.get("type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        self.logger.debug(
            f"✅ [CITATION EXTRACTION] 완료: {len(citations)}개 citation 추출됨 "
            f"(retrieved_docs: {len(retrieved_docs)}개, "
            f"doc_types: {doc_types})"
        )
        
        if len(citations) == 0:
            self.logger.debug(
                f"⚠️ [CITATION EXTRACTION] Citation 추출 실패: "
                f"retrieved_docs={len(retrieved_docs)}개 중 0개 추출됨. "
                f"문서 타입 분포: {doc_types}"
            )
        
        self.logger.debug(
            f"✅ [CITATION EXTRACTION] 완료: {len(citations)}개 citation 추출됨 "
            f"(retrieved_docs: {len(retrieved_docs)}개, doc_types: {doc_types})"
        )
        
        return citations

    def _enhance_answer_with_citations(
        self,
        answer: str,
        citations: List[Dict[str, Any]]
    ) -> str:
        """답변에 인용 추가"""
        if not citations:
            return answer
        
        citation_section = "\n\n【참고 법령 및 판례】\n"
        
        for idx, citation in enumerate(citations, 1):
            if citation["type"] == "statute":
                citation_section += f"{idx}. {citation['law_name']} 제{citation['article_no']}조\n"
            elif citation["type"] == "precedent":
                citation_section += f"{idx}. {citation['court']} {citation['case_name']}"
                if citation.get("decision_date"):
                    citation_section += f" ({citation['decision_date']})"
                citation_section += "\n"
            elif citation["type"] == "interpretation":
                citation_section += f"{idx}. {citation['org']} 해석례 (ID: {citation['interpretation_id']})\n"
            elif citation["type"] == "decision":
                citation_section += f"{idx}. {citation['org']} 결정례 (ID: {citation['doc_id']})\n"
        
        return answer + citation_section
    
    def _add_citations_simple(
        self,
        answer: str,
        citations: List[Dict[str, Any]]
    ) -> str:
        """답변에 인용 간단 추가 (기존 함수와 호환성을 위한 래퍼)"""
        if not citations:
            return answer
        
        citation_section = "\n\n【참고 법령 및 판례】\n"
        
        for idx, citation in enumerate(citations, 1):
            if citation["type"] == "statute":
                citation_section += f"{idx}. {citation['law_name']} 제{citation['article_no']}조\n"
            elif citation["type"] == "precedent":
                citation_section += f"{idx}. {citation['court']} {citation['case_name']}"
                if citation.get("decision_date"):
                    citation_section += f" ({citation['decision_date']})"
                citation_section += "\n"
            elif citation["type"] == "interpretation":
                citation_section += f"{idx}. {citation['org']} 해석례 (ID: {citation['interpretation_id']})\n"
            elif citation["type"] == "decision":
                citation_section += f"{idx}. {citation['org']} 결정례 (ID: {citation['doc_id']})\n"
        
        return answer + citation_section

    def _should_regenerate_answer(
        self,
        answer: str,
        coverage: float,
        citation_count: int,
        retrieved_docs_count: int
    ) -> bool:
        """답변 재생성 필요 여부 판단 (개선된 버전)"""
        
        if coverage < 0.4:
            self.logger.debug(
                f"🔄 [REGENERATION] Coverage 낮음: {coverage:.2f} < 0.4"
            )
            return True
        
        if citation_count == 0 and retrieved_docs_count > 0:
            self.logger.debug(
                f"🔄 [REGENERATION] 인용 없음: citation_count=0, retrieved_docs={retrieved_docs_count}"
            )
            return True
        
        if len(answer.strip()) < 100:
            self.logger.debug(
                f"🔄 [REGENERATION] 답변 너무 짧음: {len(answer)} < 100"
            )
            return True
        
        error_indicators = ["오류", "에러", "실패", "불가능", "없습니다", "알 수 없"]
        if any(indicator in answer for indicator in error_indicators):
            self.logger.debug(
                "🔄 [REGENERATION] 에러 메시지 포함"
            )
            return True
        
        return False

    @with_state_optimization("generate_answer_enhanced", enable_reduction=True)
    def generate_answer_enhanced(self, state: LegalWorkflowState, callbacks: Optional[List[Any]] = None) -> LegalWorkflowState:
        """개선된 답변 생성 - UnifiedPromptManager 활용
        
        Args:
            state: 워크플로우 상태
            callbacks: 콜백 핸들러 리스트 (스트리밍용, 선택적)
        """
        # 🔥 개선: 검색 결과 복구를 먼저 수행
        self._recover_retrieved_docs_at_start(state)
        
        # 🔥 개선: 검색 결과가 없으면 추가 복구 시도
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        if not retrieved_docs or len(retrieved_docs) == 0:
            self.logger.warning("⚠️ [GENERATE_ANSWER] retrieved_docs가 비어있음. 추가 복구 시도...")
            retrieved_docs = self._recover_retrieved_docs_comprehensive(state)
            
            if retrieved_docs:
                self._set_state_value(state, "retrieved_docs", retrieved_docs)
        
        metadata = self._get_metadata_safely(state)
        
        # 🔥 개선: state에서 콜백 추출 (파라미터로 전달되지 않은 경우)
        if callbacks is None:
            callbacks = state.get("_callbacks", [])
            if not callbacks:
                callbacks = metadata.get("_callbacks", [])
        
        try:
            is_retry, start_time = self._prepare_answer_generation(state)
            query_type = self._restore_query_type(state)
            
            # 🔥 개선: _restore_retrieved_docs 대신 이미 복구된 retrieved_docs 사용
            if not retrieved_docs:
                retrieved_docs = self._restore_retrieved_docs(state)
            
            query = self._get_state_value(state, "query", "")
            
            # 🔥 개선 2: 검색 결과가 0개일 때 빠른 응답 생성 (timeout 방지)
            if not retrieved_docs or len(retrieved_docs) == 0:
                self.logger.warning(
                    f"⚠️ [NO SEARCH RESULTS] retrieved_docs is empty. "
                    f"Generating quick response without document references to avoid timeout. "
                    f"Query: '{query[:50]}...'"
                )
                # 빠른 응답 생성 (LLM 호출 없음 - 타임아웃 방지)
                quick_answer = (
                    f"죄송합니다. '{query[:50]}...'에 대한 관련 법률 문서를 데이터베이스에서 찾지 못했습니다.\n\n"
                    f"일반적인 법률 정보를 바탕으로 답변을 드리면, 해당 질문에 대한 구체적인 조문이나 판례를 "
                    f"인용할 수 없어 정확한 답변을 제공하기 어렵습니다.\n\n"
                    f"더 정확한 답변을 위해 질문을 구체화하시거나, 다른 키워드로 검색해 주시기 바랍니다."
                )
                self._set_answer_safely(state, quick_answer)
                self._set_state_value(state, "retrieved_docs", [])
                self._set_state_value(state, "sources", [])
                self._set_state_value(state, "confidence", 0.3)  # 낮은 신뢰도 표시
                self._update_processing_time(state, start_time)
                
                elapsed_time = time.time() - start_time
                self.logger.debug(
                    f"✅ [QUICK RESPONSE] Generated quick response without LLM call "
                    f"(retrieved_docs empty, {elapsed_time:.2f}초)"
                )
                return state
            
            question_type, domain = WorkflowUtils.get_question_type_and_domain(query_type, query, self.logger)
            model_type = ModelType.GEMINI
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])

            prompt_optimized_context = self._validate_and_generate_prompt_context(
                state, retrieved_docs, query, extracted_keywords, query_type
            )
            
            context_dict = self._build_and_validate_context_dict(
                state, query_type, retrieved_docs, prompt_optimized_context
            )

            context_dict, validation_results, retrieved_docs = self._validate_context_quality_and_expand(
                state, context_dict, query, query_type, extracted_keywords, retrieved_docs
            )

            metadata = self._get_metadata_safely(state)
            metadata["context_validation"] = validation_results

            quality_feedback, base_prompt_type, context_dict = self._prepare_quality_feedback_and_context(
                state, is_retry, context_dict
            )

            context_dict = self._inject_search_results_into_context(state, context_dict, retrieved_docs, query)
            
            # 🔥 개선: context_dict에 retrieved_docs와 structured_documents가 명시적으로 포함되도록 보장
            if isinstance(context_dict, dict) and retrieved_docs:
                if "retrieved_docs" not in context_dict:
                    context_dict["retrieved_docs"] = retrieved_docs
                    context_dict["retrieved_docs_count"] = len(retrieved_docs)
                    self.logger.info(f"✅ [CONTEXT DICT] Added {len(retrieved_docs)} retrieved_docs to context_dict")
                
                # structured_documents가 없으면 생성
                if "structured_documents" not in context_dict or not context_dict.get("structured_documents"):
                    normalized_documents = self._normalize_retrieved_docs_to_structured(retrieved_docs)
                    if normalized_documents:
                        structured_docs = self._create_structured_documents_from_normalized(
                            normalized_documents, retrieved_docs, context_dict, state
                        )
                        context_dict["structured_documents"] = structured_docs
                        context_dict["document_count"] = len(normalized_documents)
                        context_dict["docs_included"] = len(normalized_documents)
                        self.logger.info(f"✅ [CONTEXT DICT] Created structured_documents from {len(normalized_documents)} retrieved_docs")
            
            optimized_prompt, prompt_file, prompt_length, structured_docs_count = self._generate_and_validate_prompt(
                state, context_dict, query, question_type, domain, model_type, base_prompt_type, retrieved_docs
            )

            normalized_response = self._generate_answer_with_cache(
                state, optimized_prompt, query, query_type, context_dict, retrieved_docs, 
                quality_feedback, is_retry, callbacks=callbacks
            )
            
            normalized_response = WorkflowUtils.normalize_answer(normalized_response)
            
            # 빈 응답 검증 및 처리
            if not normalized_response or not isinstance(normalized_response, str) or len(normalized_response.strip()) < 10:
                self.logger.warning(
                    f"⚠️ [GENERATE_ANSWER] Empty or invalid response from _generate_answer_with_cache. "
                    f"Length: {len(normalized_response) if normalized_response else 0}, "
                    f"Type: {type(normalized_response).__name__}. "
                    f"Attempting fallback answer generation..."
                )
                # 폴백 답변 생성 시도
                try:
                    fallback_answer = self.answer_generator.generate_fallback_answer(state)
                    if fallback_answer and len(fallback_answer.strip()) >= 10:
                        normalized_response = fallback_answer
                        self.logger.info(f"✅ [GENERATE_ANSWER] Fallback answer generated: length={len(normalized_response)}")
                    else:
                        self.logger.error("❌ [GENERATE_ANSWER] Fallback answer generation also failed")
                        normalized_response = ""
                except Exception as fallback_error:
                    self.logger.error(f"❌ [GENERATE_ANSWER] Fallback answer generation error: {fallback_error}")
                    normalized_response = ""
            
            if normalized_response and len(normalized_response.strip()) >= 10:
                normalized_response = self._validate_and_enhance_answer(
                    state, normalized_response, query, context_dict, retrieved_docs, 
                    prompt_length, prompt_file, structured_docs_count
                )
            else:
                # 빈 응답인 경우 최소한의 답변 생성
                self.logger.error("❌ [GENERATE_ANSWER] All answer generation attempts failed. Generating minimal answer.")
                query = self._get_state_value(state, "query", "")
                minimal_answer = f"죄송합니다. '{query[:50]}...'에 대한 답변을 생성하는 중 문제가 발생했습니다."
                self._set_answer_safely(state, minimal_answer)
                normalized_response = minimal_answer
            
            metadata["context_dict"] = context_dict
            self._set_state_value(state, "metadata", metadata)

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "답변 생성 완료", "답변 생성 완료")

            # 실행 기록 저장 (재시도 카운터는 RetryCounterManager에서 관리)
            self._save_metadata_safely(state, "_last_executed_node", "generate_answer_enhanced")

            self.logger.debug(f"Enhanced answer generated with UnifiedPromptManager in {processing_time:.2f}s")
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"⚠️ [GENERATE_ANSWER] Exception occurred: {error_msg}", exc_info=True)
            self._handle_error(state, error_msg, "개선된 답변 생성 중 오류 발생")
            
            # 개선: 'control' 오류 등 특정 오류에 대한 추가 처리
            if error_msg == 'control' or 'control' in error_msg.lower():
                self.logger.warning("⚠️ [GENERATE_ANSWER] 'control' error detected. This may indicate a validation or generation control flow issue.")
                # retrieved_docs가 있는 경우 최소한의 답변 생성 시도
                retrieved_docs = state.get("retrieved_docs", [])
                if retrieved_docs and len(retrieved_docs) > 0:
                    query = state.get("query", "")
                    simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
                    self._set_answer_safely(state, simple_answer)
                    self.logger.debug(f"⚠️ [GENERATE_ANSWER] Generated simple fallback answer due to 'control' error: length={len(simple_answer)}")
                    return state
            
            # Phase 1/Phase 7: 폴백 answer 생성 - _set_answer_safely 사용
            try:
                fallback_answer = self.answer_generator.generate_fallback_answer(state)
                if fallback_answer and len(fallback_answer.strip()) > 10:
                    # 최소 길이 보장 (최소 100자)
                    if len(fallback_answer.strip()) < 100:
                        query = state.get("query", "")
                        retrieved_docs = state.get("retrieved_docs", [])
                        if retrieved_docs and len(retrieved_docs) > 0:
                            # retrieved_docs 내용 추가
                            doc_summaries = []
                            for i, doc in enumerate(retrieved_docs[:2], 1):
                                if isinstance(doc, dict):
                                    content = doc.get("content", "") or doc.get("text", "") or doc.get("summary", "")
                                    if content and len(content) > 50:
                                        summary = content[:150].strip()
                                        if len(content) > 150:
                                            summary += "..."
                                        doc_summaries.append(f"{i}. {summary}")
                            
                            if doc_summaries:
                                fallback_answer += "\n\n검색된 문서 내용:\n" + "\n".join(doc_summaries)
                    
                    self._set_answer_safely(state, fallback_answer)
                    self.logger.debug(f"⚠️ [GENERATE_ANSWER] Generated fallback answer: length={len(fallback_answer)}")
                else:
                    # 최종 fallback: retrieved_docs 내용 활용하여 더 긴 답변 생성
                    query = state.get("query", "")
                    retrieved_docs = state.get("retrieved_docs", [])
                    if retrieved_docs and len(retrieved_docs) > 0:
                        doc_summaries = []
                        for i, doc in enumerate(retrieved_docs[:3], 1):
                            if isinstance(doc, dict):
                                content = doc.get("content", "") or doc.get("text", "") or doc.get("summary", "")
                                if content and len(content) > 50:
                                    summary = content[:200].strip()
                                    if len(content) > 200:
                                        summary += "..."
                                    doc_summaries.append(f"{i}. {summary}")
                        
                        if doc_summaries:
                            simple_answer = f"질문 '{query}'에 대한 답변입니다.\n\n"
                            simple_answer += f"검색된 문서 {len(retrieved_docs)}개를 참고하여 다음과 같은 내용을 확인했습니다:\n\n"
                            simple_answer += "\n".join(doc_summaries)
                            simple_answer += "\n\n위 내용을 바탕으로 답변을 제공합니다."
                        else:
                            simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
                    else:
                        simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다."
                    
                    # 최소 길이 보장
                    if len(simple_answer) < 100:
                        simple_answer += " 추가 정보를 확인 중입니다."
                    
                    self._set_answer_safely(state, simple_answer)
                    self.logger.debug(f"⚠️ [GENERATE_ANSWER] Generated minimal fallback answer: length={len(simple_answer)}")
            except Exception as fallback_error:
                self.logger.error(f"⚠️ [GENERATE_ANSWER] Fallback answer generation also failed: {fallback_error}")
                # 최소한의 답변이라도 생성 (최소 100자)
                query = state.get("query", "")
                retrieved_docs = state.get("retrieved_docs", [])
                minimal_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다."
                if retrieved_docs and len(retrieved_docs) > 0:
                    minimal_answer += f" 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성 중입니다."
                if len(minimal_answer) < 100:
                    minimal_answer += " 추가 정보를 확인 중입니다."
                self._set_answer_safely(state, minimal_answer)
        return state

    @with_state_optimization("continue_answer_generation", enable_reduction=True)
    def continue_answer_generation(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """이전 답변의 마지막 부분부터 이어서 답변 생성"""
        metadata = self._get_metadata_safely(state)
        
        try:
            start_time = time.time()
            
            # 이전 답변 가져오기
            previous_answer = self._get_state_value(state, "answer", "")
            if not previous_answer:
                self.logger.warning("이전 답변이 없어 이어서 생성할 수 없습니다.")
                return state
            
            # 이전 답변의 마지막 부분 (컨텍스트로 사용)
            previous_context = previous_answer[-500:] if len(previous_answer) > 500 else previous_answer
            
            # 원본 질문과 컨텍스트 복원
            query = self._get_state_value(state, "query", "")
            query_type = self._restore_query_type(state)
            retrieved_docs = self._restore_retrieved_docs(state)
            prompt_optimized_context = self._get_state_value(state, "prompt_optimized_context", {})
            context_dict = self._build_context_dict(state, query_type, retrieved_docs, prompt_optimized_context)
            
            # 이어서 생성 프롬프트 추가
            continuation_prompt = f"""
이전 답변의 마지막 부분:
{previous_context}

위 답변을 이어서 계속 작성해주세요. 자연스럽게 연결되도록 작성하세요.
"""
            
            # 프롬프트에 continuation_prompt 추가
            if isinstance(context_dict, dict):
                context_dict["continuation_prompt"] = continuation_prompt
                context_dict["previous_answer"] = previous_context
            
            # 답변 생성 (이어서 생성)
            if self.answer_generator:
                # 이어서 생성 프롬프트 직접 생성
                context_text = context_dict.get("context", "") if isinstance(context_dict, dict) else str(context_dict)
                
                optimized_prompt = f"""다음은 법률 질문에 대한 답변의 일부입니다. 이 답변을 자연스럽게 이어서 완성해주세요.

## 질문
{query}

## 이전 답변의 마지막 부분
{previous_context}

## 관련 법률 문서
{context_text[:2000] if len(context_text) > 2000 else context_text}

위 답변을 이어서 계속 작성해주세요. 자연스럽게 연결되도록 작성하고, 관련 법률 문서를 참고하여 답변을 완성해주세요."""
                
                # 이어서 답변 생성
                # LangGraph는 노드 내에서 stream() 또는 astream()을 호출하면 자동으로 on_llm_stream 이벤트를 발생시킵니다
                continued_answer = self.answer_generator.generate_answer_with_chain(
                    optimized_prompt=optimized_prompt,
                    query=query,
                    context_dict=context_dict,
                    quality_feedback=None,
                    is_retry=False
                )
                
                # 이전 답변에 이어서 생성된 답변 추가
                full_answer = previous_answer + "\n\n" + continued_answer
                
                # 상태 업데이트
                self._set_answer_safely(state, full_answer)
                
                # metadata 업데이트
                metadata["continuation_generated"] = True
                metadata["continuation_length"] = len(continued_answer)
                self._set_state_value(state, "metadata", metadata)
                
                processing_time = time.time() - start_time
                self.logger.debug(
                    f"✅ [CONTINUE ANSWER] Continued answer generation: "
                    f"{len(continued_answer)} chars added in {processing_time:.2f}s"
                )
            else:
                self.logger.error("AnswerGenerator가 없어 이어서 답변을 생성할 수 없습니다.")
        
        except Exception as e:
            self.logger.error(f"❌ [CONTINUE ANSWER] Error: {e}", exc_info=True)
            self._handle_error(state, str(e), "이어서 답변 생성 중 오류 발생")
        
        return state

    def _extract_supported_domain_from_query(self, query: str) -> Optional[LegalDomain]:
        """WorkflowUtils.extract_supported_domain_from_query 래퍼"""
        return WorkflowUtils.extract_supported_domain_from_query(query)

    # Phase 5 리팩토링: 답변 생성 관련 메서드는 AnswerGenerator로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _get_quality_feedback_for_retry(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """AnswerGenerator.get_quality_feedback_for_retry 래퍼"""
        return self.answer_generator.get_quality_feedback_for_retry(state)

    def _determine_retry_prompt_type(self, quality_feedback: Dict[str, Any]) -> str:
        """AnswerGenerator.determine_retry_prompt_type 래퍼"""
        return self.answer_generator.determine_retry_prompt_type(quality_feedback)

    def _assess_improvement_potential(
        self,
        quality_score: float,
        quality_checks: Dict[str, bool],
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """AnswerGenerator.assess_improvement_potential 래퍼"""
        result = self.answer_generator.assess_improvement_potential(
            quality_score,
            quality_checks,
            state
        )
        # 호환성을 위해 반환 형식 변환
        return {
            "should_retry": result.get("potential", 0.0) >= 0.3,
            "confidence": result.get("potential", 0.0),
            "best_strategy": result.get("strategy") or "retry_generate",
            "reasons": result.get("reasons", [])
        }

    def _improve_search_query_for_retry(
        self,
        original_query: str,
        quality_feedback: Dict[str, Any],
        state: LegalWorkflowState
    ) -> str:
        """재시도를 위한 검색 쿼리 개선"""
        improvements = []

        # 법령 검증 실패 → 법령 관련 키워드 추가
        failed_checks = quality_feedback.get("failed_checks", [])
        if any("법령" in check or "법" in check for check in failed_checks):
            improvements.append("법령 조항")
            improvements.append("법률 규정")

        # 소스 없음 → 검색 범위 확대
        if any("소스" in check or "출처" in check for check in failed_checks):
            # 키워드 확장
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            if isinstance(extracted_keywords, list) and len(extracted_keywords) > 0:
                # hashable 타입만 추가
                safe_keywords = [kw for kw in extracted_keywords[:3] if isinstance(kw, (str, int, float))]
                improvements.extend([str(kw) for kw in safe_keywords])

        # 답변이 짧음 → 더 구체적인 키워드
        if any("짧" in check for check in failed_checks):
            query_type = self._get_state_value(state, "query_type", "")
            if "계약" in str(query_type) or "계약" in original_query:
                improvements.append("계약서 작성 요건")
            elif "소송" in str(query_type) or "소송" in original_query:
                improvements.append("소송 절차")
            elif "손해" in original_query or "배상" in original_query:
                improvements.append("손해배상 요건")

        if improvements:
            improved_query = f"{original_query} {' '.join(improvements)}"
            return improved_query.strip()

        return original_query


    def _select_high_value_documents(
        self,
        documents: List[Dict],
        query: str,
        min_relevance: float = 0.7,
        max_docs: int = 5
    ) -> List[Dict]:
        """정보 밀도 기반 문서 선택 (WorkflowDocumentProcessor 사용)"""
        return self.workflow_document_processor.select_high_value_documents(
            documents=documents,
            query=query,
            min_relevance=min_relevance,
            max_docs=max_docs
        )

    def _extract_key_insights(
        self,
        documents: List[Dict],
        query: str
    ) -> List[str]:
        """핵심 정보 추출 - 질문과 직접 관련된 핵심 문장 추출"""
        insights = DocumentExtractor.extract_key_insights(documents, query)
        self.logger.debug(f"📝 [KEY INSIGHTS] Extracted {len(insights)} key insights")
        return insights

    def _extract_legal_citations(
        self,
        documents: List[Dict]
    ) -> List[Dict[str, str]]:
        """법률 인용 정보 추출"""
        citations = DocumentExtractor.extract_legal_citations(documents)
        self.logger.debug(f"⚖️ [LEGAL CITATIONS] Extracted {len(citations)} citations")
        return citations

    def _validate_context_quality(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        state: Optional[LegalWorkflowState] = None
    ) -> Dict[str, Any]:
        """ContextExpansionProcessor.validate_context_quality 래퍼 (ContextValidator 사용)"""
        # context 타입 검증 및 변환 (근본적 해결)
        if not isinstance(context, dict):
            if isinstance(context, str):
                self.logger.error("❌ [VALIDATE CONTEXT] context is str, converting to dict")
                context = {"context": context}
            else:
                self.logger.error(f"❌ [VALIDATE CONTEXT] context is not dict (type: {type(context)}), using empty dict")
                context = {}
        
        try:
            if self.context_expansion_processor:
                result = self.context_expansion_processor.validate_context_quality(
                    context, query, query_type, extracted_keywords
                )
            else:
                from core.generation.validators.quality_validators import ContextValidator
                result = ContextValidator.validate_context_quality(
                    context=context,
                    query=query,
                    query_type=query_type,
                    extracted_keywords=extracted_keywords,
                    calculate_relevance_func=None,
                    calculate_coverage_func=None
                )
            
            # 반환값 타입 검증 (근본적 해결)
            if not isinstance(result, dict):
                self.logger.error(f"❌ [VALIDATE CONTEXT] validate_context_quality returned non-dict (type: {type(result)}), using default dict")
                if isinstance(result, str):
                    result = {"error": result, "overall_score": 0.3, "relevance_score": 0.3, "coverage_score": 0.3, "sufficiency_score": 0.3, "needs_expansion": False}
                else:
                    result = {"overall_score": 0.3, "relevance_score": 0.3, "coverage_score": 0.3, "sufficiency_score": 0.3, "needs_expansion": False}
            
            # 검색 결과가 있는 경우 최소 품질 점수 보장
            if state is None:
                state = {}
            retrieved_docs = state.get("retrieved_docs", [])
            semantic_results = state.get("search", {}).get("semantic_results", []) if isinstance(state.get("search"), dict) else []
            keyword_results = state.get("search", {}).get("keyword_results", []) if isinstance(state.get("search"), dict) else []
            
            has_search_results = (retrieved_docs and len(retrieved_docs) > 0) or (semantic_results and len(semantic_results) > 0) or (keyword_results and len(keyword_results) > 0)
            
            if has_search_results:
                # 검색 결과가 있으면 최소 점수 보장
                overall_score = result.get("overall_score", 0.0)
                relevance_score = result.get("relevance_score", 0.0)
                coverage_score = result.get("coverage_score", 0.0)
                sufficiency_score = result.get("sufficiency_score", 0.0)
                
                # 모든 점수가 0.0이면 최소 점수로 설정 (개선: 0.4로 상향)
                if overall_score == 0.0 and relevance_score == 0.0 and coverage_score == 0.0 and sufficiency_score == 0.0:
                    self.logger.warning("⚠️ [VALIDATE CONTEXT] All scores are 0.0 but search results exist. Setting minimum scores.")
                    result["relevance_score"] = 0.4
                    result["coverage_score"] = 0.4
                    result["sufficiency_score"] = 0.4
                    result["overall_score"] = 0.4
                elif overall_score < 0.4:
                    # overall_score가 0.4 미만이면 최소 0.4로 보장 (개선된 품질 평가와 일치)
                    result["overall_score"] = max(0.4, result.get("overall_score", 0.0))
                    if result.get("relevance_score", 0.0) < 0.4:
                        result["relevance_score"] = max(0.4, result.get("relevance_score", 0.0))
                    if result.get("coverage_score", 0.0) < 0.4:
                        result["coverage_score"] = max(0.4, result.get("coverage_score", 0.0))
                    if result.get("sufficiency_score", 0.0) < 0.4:
                        result["sufficiency_score"] = max(0.4, result.get("sufficiency_score", 0.0))
                elif overall_score == 0.0:
                    # overall_score만 0.0이면 재계산
                    result["overall_score"] = max(0.4, (result.get("relevance_score", 0.4) * 0.4 + 
                                               result.get("coverage_score", 0.4) * 0.3 + 
                                               result.get("sufficiency_score", 0.4) * 0.3))
            
            return result
        except Exception as e:
            self.logger.error(f"❌ [VALIDATE CONTEXT] Error in validate_context_quality: {e}", exc_info=True)
            # 예외 발생 시 검색 결과가 있으면 최소 점수 보장
            if state is None:
                state = {}
            retrieved_docs = state.get("retrieved_docs", [])
            semantic_results = state.get("search", {}).get("semantic_results", []) if isinstance(state.get("search"), dict) else []
            keyword_results = state.get("search", {}).get("keyword_results", []) if isinstance(state.get("search"), dict) else []
            
            has_search_results = (retrieved_docs and len(retrieved_docs) > 0) or (semantic_results and len(semantic_results) > 0) or (keyword_results and len(keyword_results) > 0)
            
            if has_search_results:
                # 검색 결과가 있으면 최소 점수 반환
                return {
                    "overall_score": 0.3,
                    "relevance_score": 0.3,
                    "coverage_score": 0.3,
                    "sufficiency_score": 0.3,
                    "needs_expansion": False,
                    "error": str(e)
                }
            else:
                # 검색 결과가 없으면 0.0 반환
                return {
                    "overall_score": 0.0,
                    "relevance_score": 0.0,
                    "coverage_score": 0.0,
                    "sufficiency_score": 0.0,
                    "needs_expansion": False,
                    "error": str(e)
                }

    def _validate_context_quality_original(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> Dict[str, Any]:
        """컨텍스트 적합성 검증 (ContextBuilder 사용)"""
        try:
            relevance_score = self.context_builder.calculate_context_relevance(context, query)
            coverage_score = self.context_builder.calculate_information_coverage(
                context, query, query_type, extracted_keywords
            )
            sufficiency_score = self.context_builder.calculate_context_sufficiency(context, query_type)
            missing_info = self.context_builder.identify_missing_information(
                context, query, query_type, extracted_keywords
            )

            overall_score = (relevance_score * 0.4 + coverage_score * 0.3 + sufficiency_score * 0.3)

            validation_result = {
                "relevance_score": relevance_score,
                "coverage_score": coverage_score,
                "sufficiency_score": sufficiency_score,
                "overall_score": overall_score,
                "missing_information": missing_info,
                "is_sufficient": overall_score >= 0.6,
                "needs_expansion": overall_score < 0.6 or len(missing_info) > 3
            }

            self.logger.debug(
                f"🔍 [CONTEXT VALIDATION] Relevance: {relevance_score:.2f}, "
                f"Coverage: {coverage_score:.2f}, Sufficiency: {sufficiency_score:.2f}, "
                f"Overall: {overall_score:.2f}, Missing: {len(missing_info)} items"
            )

            return validation_result

        except Exception as e:
            self.logger.warning(f"Context validation failed: {e}")
            return {
                "relevance_score": 0.5,
                "coverage_score": 0.5,
                "sufficiency_score": 0.5,
                "overall_score": 0.5,
                "missing_information": [],
                "is_sufficient": True,
                "needs_expansion": False
            }

    def _should_expand_context(
        self,
        validation_results: Dict[str, Any],
        existing_docs: List[Dict[str, Any]]
    ) -> bool:
        """ContextExpansionProcessor.should_expand 래퍼"""
        if self.context_expansion_processor:
            return self.context_expansion_processor.should_expand(validation_results, existing_docs)
        return False

    def _enhance_answer_with_citations(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        legal_references: List[str],
        citations: List[Dict[str, Any]]
    ) -> str:
        """답변에 Citation이 부족하면 자동으로 보강 (성능 최적화)"""
        # 현재 답변의 Citation 개수 확인 (컴파일된 정규식 사용)
        existing_laws = len(LAW_PATTERN.findall(answer))
        existing_precedents = len(PRECEDENT_PATTERN.findall(answer))
        
        # retrieved_docs에서 법령 조문 추출 (개선: 더 많은 문서에서 추출)
        extracted_laws = []
        extracted_precedents = []
        
        for doc in retrieved_docs[:10]:  # 5 -> 10으로 증가
            content = doc.get("content", "") or doc.get("text", "")
            if not content or not isinstance(content, str):
                continue
            
            # 법령 조문 추출 (컴파일된 정규식 사용)
            law_matches = LAW_PATTERN.findall(content)
            for law in law_matches[:5]:  # 3 -> 5로 증가
                if law not in extracted_laws:
                    extracted_laws.append(law)
            
            # 판례 추출 (컴파일된 정규식 사용)
            precedent_matches = PRECEDENT_PATTERN.findall(content)
            for precedent in precedent_matches[:3]:  # 2 -> 3으로 증가
                if precedent not in extracted_precedents:
                    extracted_precedents.append(precedent)
        
        # legal_references와 병합
        if legal_references:
            for ref in legal_references:
                if isinstance(ref, str) and ref not in extracted_laws:
                    extracted_laws.append(ref)
        
        # citations와 병합
        if citations:
            for cit in citations:
                if isinstance(cit, dict):
                    cit_text = cit.get("text", "")
                    if cit_text:
                        if cit.get("type") == "precedent" and cit_text not in extracted_precedents:
                            extracted_precedents.append(cit_text)
                        elif "법" in cit_text and "조" in cit_text and cit_text not in extracted_laws:
                            extracted_laws.append(cit_text)
        
        # 개선: 더 많은 Citation 추가 (retrieved_docs 개수에 비례하여 목표 설정)
        # retrieved_docs가 5개면 최소 3개, 10개면 최소 5개 목표
        if retrieved_docs:
            target_citation_count = max(3, min(len(retrieved_docs), 8))  # 최소 3개, 최대 8개
        else:
            target_citation_count = 3
        
        required_laws = min(max(2, target_citation_count // 2), len(extracted_laws))  # 최소 2개
        required_precedents = min(max(1, target_citation_count - required_laws), len(extracted_precedents))  # 최소 1개
        
        enhanced_answer = answer
        
        # 법령 조문이 부족하면 추가 (개선: 더 적극적으로 추가)
        if existing_laws < required_laws and extracted_laws:
            missing_count = required_laws - existing_laws
            missing_laws = extracted_laws[:missing_count]
            
            # 답변에 이미 Citation 섹션이 있는지 확인
            if "### 관련 법령" in answer or "**관련 법령**" in answer or "[법령:" in answer:
                # 기존 섹션에 추가
                citation_text = ""
                for law in missing_laws:
                    if isinstance(law, str) and law not in answer:
                        citation_text += f"- **[법령: {law}]**: 해당 조문에 따르면...\n"
                if citation_text:
                    enhanced_answer += "\n" + citation_text
            else:
                # 새 섹션 생성
                citation_text = "\n\n### 관련 법령\n"
                for law in missing_laws:
                    if isinstance(law, str):
                        citation_text += f"- **[법령: {law}]**: 해당 조문에 따르면...\n"
                enhanced_answer += citation_text
            
            if missing_laws:
                self.logger.debug(f"🔧 [CITATION ENHANCEMENT] Added {len(missing_laws)} law citations")
        
        # 판례가 부족하면 추가 (개선: 더 적극적으로 추가)
        if existing_precedents < required_precedents and extracted_precedents:
            missing_count = required_precedents - existing_precedents
            missing_precedents = extracted_precedents[:missing_count]
            
            # 답변에 이미 Citation 섹션이 있는지 확인
            if "### 참고 판례" in answer or "**참고 판례**" in answer or "[판례:" in answer:
                # 기존 섹션에 추가
                citation_text = ""
                for precedent in missing_precedents:
                    if isinstance(precedent, str) and precedent not in answer:
                        citation_text += f"- **[판례: {precedent}]**: 해당 판결에 의하면...\n"
                if citation_text:
                    enhanced_answer += "\n" + citation_text
            else:
                # 새 섹션 생성
                citation_text = "\n\n### 참고 판례\n"
                for precedent in missing_precedents:
                    if isinstance(precedent, str):
                        citation_text += f"- **[판례: {precedent}]**: 해당 판결에 의하면...\n"
                enhanced_answer += citation_text
            
            if missing_precedents:
                self.logger.debug(f"🔧 [CITATION ENHANCEMENT] Added {len(missing_precedents)} precedent citations")
        
        return enhanced_answer

    def _build_expanded_query(
        self,
        query: str,
        missing_info: List[str],
        query_type: str
    ) -> str:
        """ContextExpansionProcessor.build_expanded_query 래퍼"""
        if self.context_expansion_processor:
            return self.context_expansion_processor.build_expanded_query(query, missing_info, query_type)
        return query

    def _adaptive_context_expansion(
        self,
        state: LegalWorkflowState,
        validation_results: Dict[str, Any]
    ) -> LegalWorkflowState:
        """ContextExpansionProcessor.expand_context 래퍼"""
        if self.context_expansion_processor:
            return self.context_expansion_processor.expand_context(state, validation_results)
        return state

    # Phase 5 리팩토링: 답변 생성 관련 메서드는 AnswerGenerator로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _validate_answer_uses_context(
        self,
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """AnswerGenerator.validate_answer_uses_context 래퍼"""
        return self.answer_generator.validate_answer_uses_context(
            answer=answer,
            context=context,
            query=query,
            retrieved_docs=retrieved_docs
        )

    def _track_search_to_answer_pipeline(
        self,
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """AnswerGenerator.track_search_to_answer_pipeline 래퍼"""
        return self.answer_generator.track_search_to_answer_pipeline(state)

    # Phase 6 리팩토링: 컨텍스트 빌더 관련 메서드는 ContextBuilder로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _build_context(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """ContextBuilder.build_context 래퍼"""
        return self.context_builder.build_context(state)
    
    def _validate_with_llm(self, answer: str, state: LegalWorkflowState) -> Dict[str, Any]:
        """AnswerQualityValidator.validate_with_llm 래퍼"""
        if self.answer_quality_validator:
            return self.answer_quality_validator.validate_with_llm(answer, state)
        return {}

    def _call_llm_with_retry(self, prompt: str, max_retries: int = WorkflowConstants.MAX_RETRIES, timeout: float = 30.0) -> str:
        """LLM 호출 (재시도 로직 및 타임아웃 포함)"""
        if hasattr(self, 'answer_generator') and self.answer_generator and hasattr(self.answer_generator, 'call_llm_with_retry'):
            return self.answer_generator.call_llm_with_retry(prompt, max_retries)
        
        # Fallback: 직접 LLM 호출
        if not hasattr(self, 'llm') or not self.llm:
            self.logger.error("LLM not available for multi-query generation")
            raise RuntimeError("LLM not available")
        
        import asyncio
        for attempt in range(max_retries):
            try:
                # 비동기 호출이 가능한 경우 타임아웃 적용
                if hasattr(self.llm, 'ainvoke'):
                    try:
                        response = asyncio.run(
                            asyncio.wait_for(
                                self.llm.ainvoke(prompt),
                                timeout=timeout
                            )
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"LLM call timed out after {timeout} seconds")
                else:
                    # 동기 호출의 경우 타임아웃을 시뮬레이션하기 어려우므로 그대로 호출
                    response = self.llm.invoke(prompt)
                
                from core.workflow.utils.workflow_utils import WorkflowUtils
                result = WorkflowUtils.extract_response_content(response)
                return result
            except (TimeoutError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"LLM call timed out (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    wait_time = min(0.1 * (2 ** attempt), 0.5)
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"LLM call timed out after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    wait_time = min(0.1 * (2 ** attempt), 0.5)
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
    
    def _generate_multi_queries_with_llm(
        self,
        query: str,
        query_type: str,
        max_queries: int = 5,
        use_cache: bool = True
    ) -> List[str]:
        """
        LLM을 사용하여 사용자 질문을 여러 개의 검색용 질문으로 재작성 (Multi-Query Retrieval)
        
        Args:
            query: 원본 사용자 질문
            query_type: 질문 유형 (statute, case, decision, interpretation 등)
            max_queries: 생성할 최대 질문 수 (원본 포함)
            use_cache: 캐싱 사용 여부
        
        Returns:
            재작성된 검색용 질문 리스트 (원본 포함)
        """
        if not query or not query.strip():
            return [query] if query else []
        
        # 간단한 메모리 캐시 사용 (클래스 변수로 관리)
        if not hasattr(self.__class__, '_multi_query_cache'):
            self.__class__._multi_query_cache = {}
        
        # 캐시 키 생성
        cache_key = f"multi_query:{query}:{query_type}:{max_queries}"
        
        # 캐시 확인
        if use_cache:
            cached = self.__class__._multi_query_cache.get(cache_key)
            if cached:
                self.logger.debug(f"✅ [MULTI-QUERY] Cache hit for query: '{query[:50]}...'")
                self.logger.debug(f"[MULTI-QUERY] Cache hit for query: '{query[:50]}...'")
                return cached
        
        try:
            self.logger.debug("[MULTI-QUERY] Calling LLM to generate query variations...")
            self.logger.debug(f"🔍 [MULTI-QUERY] Calling LLM to generate query variations for: '{query[:50]}...'")
            
            # 새로운 프롬프트 (법률 전문 질의 재작성)
            num_variations = max_queries - 1  # 원본 제외한 변형 개수
            
            prompt = f"""당신은 법률 분야 전문 질의 재작성(Multi-Query) 생성기입니다.  
지금부터 사용자의 원본 질문을 **서로 다른 관점·법률 용어·쟁점 표현·조문 방식**으로 다양하게 변형해 생성하세요.

아래 규칙을 따르십시오:

[생성 규칙]
1. 원문의 의미는 유지하되, 서로 다른 방식(용어·문장구조·법률 개념)으로 표현할 것
2. 법률 용어(조문, 법률명, 법적 표현 등)를 포함한 변형 1개 이상 생성
3. 실무에서 자주 쓰는 질문 형태로 변형 1개 이상 생성
4. 너무 포괄적이거나 너무 좁은 의미로 변형하지 말 것
5. 한 줄에 하나씩 출력할 것
6. 질문만 출력하고 설명은 금지

[원본 질문]
{query}

[출력 형태]
재작성:
- 질문1
- 질문2
- 질문3
{'- 질문4' if num_variations >= 4 else ''}{'- 질문5' if num_variations >= 5 else ''}

총 {num_variations}개의 변형된 질문을 생성하세요."""

            response = self._call_llm_with_retry(prompt, max_retries=2)
            self.logger.debug(f"[MULTI-QUERY] LLM response received: {len(response)} chars")
            self.logger.debug(f"🔍 [MULTI-QUERY] LLM response: {response[:200]}...")
            
            # 응답에서 질문 추출 (새로운 프롬프트 형식에 맞게)
            queries = []
            skip_patterns = [
                "재작성:", "재작성", "각 줄에", "하나씩", "질문:", "유형:", "원본 질문:",
                "요구사항:", "다음 질문을", "다음 법률 질문을", "출력 형태", "생성 규칙",
                "당신은", "법률 분야", "지금부터", "아래 규칙", "원본 질문", "총", "개의 변형"
            ]
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # "재작성:" 섹션 시작 확인
                if "재작성" in line and ":" in line:
                    continue
                
                # 프롬프트 텍스트 스킵
                if any(pattern in line for pattern in skip_patterns):
                    continue
                
                # "- 질문" 형식 또는 번호 패턴 제거
                if line.startswith('-'):
                    line = line[1:].strip()
                line = line.lstrip('0123456789.-) ')
                
                if line and not line.startswith('#') and len(line) > 5:
                        # 개선: 불필요한 텍스트 제거 (예: "계약 해지 통보 계약" 같은 중복/불필요한 텍스트)
                        # 1. 중복된 단어 제거 (연속된 동일 단어)
                        words = line.split()
                        cleaned_words = []
                        prev_word = None
                        for word in words:
                            if word != prev_word:
                                cleaned_words.append(word)
                            prev_word = word
                        line = ' '.join(cleaned_words)
                        
                        # 2. 불필요한 접미사 제거 (예: "계약 해석", "계약 해지 통보 계약" 등)
                        # 원본 쿼리의 핵심 키워드만 추출하여 불필요한 접미사 제거
                        query_keywords = set(query.split())
                        line_words = line.split()
                        # 원본 쿼리에 없는 불필요한 접미사 제거 (마지막 부분)
                        while line_words and line_words[-1] not in query_keywords and len(line_words) > len(query.split()):
                            # 원본 쿼리와 겹치는 단어가 있는지 확인
                            if any(word in query_keywords for word in line_words[:-1]):
                                line_words.pop()
                            else:
                                break
                        line = ' '.join(line_words)
                        
                        # 3. 쿼리 길이 제한 (너무 긴 쿼리는 검색 실패 가능성 높음)
                        if len(line) > 100:
                            # 핵심 키워드만 추출 (원본 쿼리의 키워드 우선)
                            query_words = query.split()
                            line_words = line.split()
                            # 원본 쿼리 단어를 우선 포함
                            essential_words = [w for w in line_words if w in query_words]
                            # 나머지 단어 추가 (최대 100자)
                            for word in line_words:
                                if word not in essential_words and len(' '.join(essential_words + [word])) <= 100:
                                    essential_words.append(word)
                            line = ' '.join(essential_words[:15])  # 최대 15개 단어
                        
                        if line and len(line) > 5:
                            # 불완전한 질문 제거 (끝이 조사나 불완전한 경우)
                            if line.endswith(('이', '을', '를', '의', '에', '에서', '에게', '한테', '께', '와', '과', '하고', '그리고', '또한', '또는', '및')):
                                # 불완전한 질문으로 보이지만, 법률 용어일 수 있으므로 최소 길이 체크
                                if len(line) >= 10:  # 최소 10자 이상이면 포함
                                    queries.append(line)
                            else:
                                queries.append(line)
            
            # 원본 질문을 첫 번째로 포함
            result_queries = [query] + queries[:max_queries - 1]
            result_queries = result_queries[:max_queries]
            
            # 최소 1개는 보장 (원본)
            if not result_queries:
                result_queries = [query]
            
            # 개선: 캐시 크기 증가 및 LRU 스타일 유지
            if use_cache:
                if len(self.__class__._multi_query_cache) >= 200:  # 100 → 200
                    # 가장 오래된 항목 제거 (FIFO)
                    oldest_key = next(iter(self.__class__._multi_query_cache))
                    del self.__class__._multi_query_cache[oldest_key]
                self.__class__._multi_query_cache[cache_key] = result_queries
            
            self.logger.debug(
                f"✅ [MULTI-QUERY] Generated {len(result_queries)} queries for: '{query[:50]}...' "
                f"(original + {len(result_queries) - 1} variations)"
            )
            self.logger.debug(
                f"[MULTI-QUERY] Generated {len(result_queries)} queries: "
                f"{[q[:30] + '...' if len(q) > 30 else q for q in result_queries]}"
            )
            
            return result_queries
            
        except Exception as e:
            self.logger.warning(
                f"⚠️ [MULTI-QUERY] LLM 기반 질문 재작성 실패: {e}, 원본 질문 사용"
            )
            self.logger.debug(f"[MULTI-QUERY] Error: {e}, using original query")
            return [query]

    def _generate_answer_with_chain(
        self,
        optimized_prompt: str,
        query: str,
        context_dict: Dict[str, Any],
        quality_feedback: Optional[Dict[str, Any]] = None,
        is_retry: bool = False
    ) -> str:
        """AnswerGenerator.generate_answer_with_chain 래퍼"""
        return self.answer_generator.generate_answer_with_chain(
            optimized_prompt=optimized_prompt,
            query=query,
            context_dict=context_dict,
            quality_feedback=quality_feedback,
            is_retry=is_retry
        )

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """WorkflowUtils.parse_validation_response 래퍼"""
        return WorkflowUtils.parse_validation_response(response, self.logger)

    def _parse_improvement_instructions(self, response: str) -> Optional[Dict[str, Any]]:
        """WorkflowUtils.parse_improvement_instructions 래퍼"""
        return WorkflowUtils.parse_improvement_instructions(response, self.logger)

    def _parse_final_validation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """WorkflowUtils.parse_final_validation_response 래퍼"""
        return WorkflowUtils.parse_final_validation_response(response, self.logger)



    # Phase 5 리팩토링: 답변 생성 관련 메서드는 AnswerGenerator로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _generate_fallback_answer(self, state: LegalWorkflowState) -> str:
        """AnswerGenerator.generate_fallback_answer 래퍼"""
        return self.answer_generator.generate_fallback_answer(state)

    @with_state_optimization("format_answer", enable_reduction=True)
    # Phase 4 리팩토링: 답변 포맷팅 관련 메서드는 AnswerFormatterHandler로 이동됨
    # 호환성을 위한 래퍼 메서드
    def format_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AnswerFormatterHandler.format_answer 래퍼"""
        return self.answer_formatter_handler.format_answer(state)

    @with_state_optimization("prepare_final_response", enable_reduction=False)
    def prepare_final_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AnswerFormatterHandler.prepare_final_response 래퍼"""
        return self.answer_formatter_handler.prepare_final_response(state)

    # 호환성을 위한 래퍼 메서드
    def _format_answer_part(self, state: LegalWorkflowState) -> str:
        """AnswerFormatterHandler.format_answer_part 래퍼"""
        return self.answer_formatter_handler.format_answer_part(state)

    def _prepare_final_response_part(
        self,
        state: LegalWorkflowState,
        query_complexity: Optional[str],
        needs_search: bool
    ) -> None:
        """AnswerFormatterHandler.prepare_final_response_part 래퍼"""
        self.answer_formatter_handler.prepare_final_response_part(state, query_complexity, needs_search)

    def _extract_preserved_values(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """AnswerFormatterHandler.extract_preserved_values 래퍼"""
        return self.answer_formatter_handler.extract_preserved_values(state)

    def _preserve_and_store_values(
        self,
        state: LegalWorkflowState,
        query_complexity: Optional[str],
        needs_search: bool
    ) -> None:
        """AnswerFormatterHandler.preserve_and_store_values 래퍼"""
        self.answer_formatter_handler.preserve_and_store_values(state, query_complexity, needs_search)

    def _map_confidence_level(self, confidence: float):
        """AnswerFormatterHandler.map_confidence_level 래퍼"""
        return self.answer_formatter_handler.map_confidence_level(confidence)

    def _calculate_keyword_coverage(
        self,
        state: LegalWorkflowState,
        answer: Union[str, Dict[str, Any], None]
    ) -> float:
        """AnswerFormatterHandler.calculate_keyword_coverage 래퍼"""
        return self.answer_formatter_handler.calculate_keyword_coverage(state, answer)

    def _set_metadata(
        self,
        state: LegalWorkflowState,
        answer: Union[str, Dict[str, Any], None],
        keyword_coverage: float
    ) -> None:
        """AnswerFormatterHandler.set_metadata 래퍼"""
        self.answer_formatter_handler.set_metadata(state, answer, keyword_coverage)

    # 답변 포맷팅 관련 메서드는 AnswerFormatterHandler로 이동됨

    def update_statistics(self, state: LegalWorkflowState):
        """WorkflowStatistics.update_statistics 래퍼"""
        if self.workflow_statistics:
            self.workflow_statistics.update_statistics(state, self.config)
            self.stats = self.workflow_statistics.stats

    def get_statistics(self) -> Dict[str, Any]:
        """WorkflowStatistics.get_statistics 래퍼"""
        if self.workflow_statistics:
            return self.workflow_statistics.get_statistics()
        return {"enabled": False}

    def _extract_legal_field(self, query_type: str, query: str) -> str:
        """법률 분야 추출"""
        return QueryExtractor.extract_legal_field(query_type, query)

    def _map_to_legal_domain(self, legal_field: str) -> str:
        """
        LegalDomain enum으로 매핑 - 지원되는 도메인만

        현재 지원 도메인:
        - 민사법 (CIVIL_LAW)
        - 지식재산권법 (INTELLECTUAL_PROPERTY)
        - 행정법 (ADMINISTRATIVE_LAW)
        - 형사법 (CRIMINAL_LAW)

        이외는 "기타/일반"으로 처리
        """
        # 지원되는 도메인만 매핑
        mapping = {
            "civil": LegalDomain.CIVIL_LAW.value if hasattr(LegalDomain.CIVIL_LAW, 'value') else "민사법",
            "criminal": LegalDomain.CRIMINAL_LAW.value if hasattr(LegalDomain.CRIMINAL_LAW, 'value') else "형사법",
            "intellectual_property": LegalDomain.INTELLECTUAL_PROPERTY.value if hasattr(LegalDomain.INTELLECTUAL_PROPERTY, 'value') else "지적재산권법",
            "administrative": LegalDomain.ADMINISTRATIVE_LAW.value if hasattr(LegalDomain.ADMINISTRATIVE_LAW, 'value') else "행정법",
        }
        return mapping.get(legal_field, "기타/일반")

    @with_state_optimization("classification_parallel", enable_reduction=True)
    def classification_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """분류 작업 병렬 실행 (긴급도 평가 + 멀티턴 처리)"""
        try:
            start_time = time.time()
            from concurrent.futures import ThreadPoolExecutor

            query = self._get_state_value(state, "query", "")
            session_id = self._get_state_value(state, "session_id", "")

            # 병렬 작업 결과 저장
            urgency_level = None
            urgency_reasoning = None
            is_multi_turn = False
            search_query = query

            with ThreadPoolExecutor(max_workers=2) as executor:
                # 병렬 작업 정의
                futures = {
                    'urgency': executor.submit(self._assess_urgency_internal, query),
                    'multi_turn': executor.submit(self._resolve_multi_turn_internal, query, session_id),
                }

                # 결과 수집
                results = {}
                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=10)
                    except Exception as e:
                        self.logger.error(f"{key} 병렬 실행 실패: {e}")
                        results[key] = None

                # State 업데이트
                if results['urgency']:
                    urgency_level, urgency_reasoning = results['urgency']
                    self._set_state_value(state, "urgency_level", urgency_level)
                    self._set_state_value(state, "urgency_reasoning", urgency_reasoning)

                if results['multi_turn']:
                    is_multi_turn, search_query = results['multi_turn']
                    self._set_state_value(state, "is_multi_turn", is_multi_turn)
                    if search_query:
                        self._set_state_value(state, "search_query", search_query)

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(
                state,
                "병렬 분류 완료",
                f"긴급도 평가 및 멀티턴 처리 병렬 완료 (시간: {processing_time:.3f}s)"
            )

            self.logger.debug(
                f"✅ 병렬 분류 완료: 긴급도={urgency_level}, 멀티턴={is_multi_turn} (시간: {processing_time:.3f}s)"
            )

        except Exception as e:
            self._handle_error(state, str(e), "병렬 분류 중 오류 발생")
            # 폴백: 기본값 설정
            self._set_state_value(state, "urgency_level", "medium")
            self._set_state_value(state, "is_multi_turn", False)
            self._set_state_value(state, "search_query", query)

        return state

    def _assess_urgency_internal(self, query: str) -> Tuple[str, str]:
        """긴급도 평가 (내부 로직)"""
        try:
            if self.emotion_analyzer:
                intent_result = self.emotion_analyzer.analyze_intent(query, None)

                # 긴급도 설정
                if intent_result and hasattr(intent_result, 'emergency_level'):
                    if hasattr(intent_result.emergency_level, 'value'):
                        urgency_level = intent_result.emergency_level.value
                    elif hasattr(intent_result.emergency_level, 'lower'):
                        urgency_level = intent_result.emergency_level.lower()
                    else:
                        urgency_level = str(intent_result.emergency_level).lower()

                    urgency_reasoning = getattr(intent_result, 'reasoning', None) or "긴급도 분석 완료"
                    return urgency_level, urgency_reasoning

            # 폴백: 키워드 기반 평가
            urgency_level = self._assess_urgency_fallback(query)
            return urgency_level, "키워드 기반 평가"
        except Exception as e:
            self.logger.error(f"긴급도 평가 내부 로직 실패: {e}")
            return "medium", "오류 발생, 기본값 사용"

    def _resolve_multi_turn_internal(self, query: str, session_id: str) -> Tuple[bool, str]:
        """멀티턴 처리 (내부 로직)"""
        try:
            if not self.multi_turn_handler or not self.conversation_manager:
                return False, query

            conversation_context = self._get_or_create_conversation_context(session_id)

            if conversation_context and conversation_context.turns:
                is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, conversation_context)

                if is_multi_turn:
                    multi_turn_result = self.multi_turn_handler.build_complete_query(query, conversation_context)
                    search_query = multi_turn_result.complete_query if multi_turn_result else query
                    return True, search_query

            return False, query
        except Exception as e:
            self.logger.error(f"멀티턴 처리 내부 로직 실패: {e}")
            return False, query

    @with_state_optimization("assess_urgency", enable_reduction=True)
    def assess_urgency(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """ConversationProcessor.assess_urgency 래퍼"""
        if self.conversation_processor:
            result_state = self.conversation_processor.assess_urgency(state)
            query = self._get_state_value(result_state, "query", "")
            urgency_level = self._get_state_value(result_state, "urgency_level", "medium")
            
            if "기한" in query or "마감" in query or "데드라인" in query:
                self._set_state_value(result_state, "emergency_type", "legal_deadline")
            elif "소송" in query or "재판" in query or "법원" in query:
                self._set_state_value(result_state, "emergency_type", "case_progress")
            else:
                self._set_state_value(result_state, "emergency_type", None)
            
            self._add_step(result_state, "긴급도 평가", f"긴급도: {urgency_level}")
            return result_state
        
        self.logger.warning("ConversationProcessor not available, using fallback")
        self._set_state_value(state, "urgency_level", "medium")
        self._set_state_value(state, "urgency_reasoning", "긴급도 평가 실패")
        return state

    def _assess_urgency_fallback(self, query: str) -> Tuple[str, str]:
        """ConversationProcessor.assess_urgency_fallback 래퍼"""
        if self.conversation_processor:
            return self.conversation_processor.assess_urgency_fallback(query)
        return "medium", "일반 긴급도"

    @with_state_optimization("analyze_document", enable_reduction=True)
    def analyze_document(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """업로드된 문서 분석"""
        try:
            start_time = time.time()

            if self.document_analysis_processor:
                state = self.document_analysis_processor.analyze_document(state)
                self._update_processing_time(state, start_time)
                doc_type = state.get("document_type", "unknown")
                self._add_step(state, "문서 분석", f"{doc_type} 분석 완료")
            else:
                self.logger.warning("DocumentAnalysisProcessor not available, skipping document analysis")
                self._handle_error(state, "DocumentAnalysisProcessor not initialized", "문서 분석 중 오류")

        except Exception as e:
            self._handle_error(state, str(e), "문서 분석 중 오류")

        return state

    def _detect_document_type(self, text: str) -> str:
        """DocumentAnalysisProcessor.detect_document_type 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.detect_document_type(text)
        return "general_legal_document"

    def _analyze_legal_document(self, text: str, doc_type: str) -> Dict[str, Any]:
        """DocumentAnalysisProcessor.analyze_legal_document 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.analyze_legal_document(text, doc_type)
        return {"document_type": doc_type, "key_clauses": [], "issues": [], "summary": "", "recommendations": []}

    def _extract_contract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """계약서 주요 조항 추출 (DocumentExtractor 직접 호출)"""
        return DocumentExtractor.extract_contract_clauses(text)

    def _identify_contract_issues(self, text: str, clauses: List[Dict]) -> List[Dict[str, Any]]:
        """DocumentAnalysisProcessor.identify_contract_issues 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.identify_contract_issues(text, clauses)
        return []

    def _extract_complaint_elements(self, text: str) -> List[Dict[str, Any]]:
        """고소장 요건 추출 (DocumentExtractor 직접 호출)"""
        return DocumentExtractor.extract_complaint_elements(text)

    def _identify_complaint_issues(self, text: str) -> List[Dict[str, Any]]:
        """DocumentAnalysisProcessor.identify_complaint_issues 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.identify_complaint_issues(text)
        return []

    def _analyze_legal_document_with_chain(self, text: str, doc_type: str) -> Dict[str, Any]:
        """DocumentAnalysisProcessor.analyze_legal_document_with_chain 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.analyze_legal_document_with_chain(text, doc_type)
        return {"document_type": doc_type, "key_clauses": [], "issues": [], "summary": "", "recommendations": []}
    

    def _generate_document_summary(self, text: str, doc_type: str, analysis: Dict[str, Any]) -> str:
        """DocumentAnalysisProcessor.generate_document_summary 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.generate_document_summary(text, doc_type, analysis)
        return f"문서 유형: {doc_type}"

    def _generate_document_summary_fallback(self, text: str, doc_type: str, key_clauses: List[Dict], issues: List[Dict]) -> str:
        """DocumentAnalysisProcessor.generate_document_summary_fallback 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.generate_document_summary_fallback(text, doc_type, key_clauses, issues)
        return f"문서 유형: {doc_type}"

    def _parse_document_type_response(self, response: str) -> Dict[str, Any]:
        """ClassificationHandler.parse_document_type_response 래퍼"""
        return self.classification_handler.parse_document_type_response(response)

    def _parse_clause_extraction_response(self, response: str) -> Dict[str, Any]:
        """조항 추출 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # key_clauses가 있는지 확인
                if "key_clauses" in parsed:
                    return parsed

            # 기본값
            return {
                "key_clauses": [],
                "clause_count": 0
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse clause extraction response: {e}")
            return {
                "key_clauses": [],
                "clause_count": 0
            }

    def _parse_issue_identification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # issues가 있는지 확인
                if "issues" in parsed:
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse issue identification response: {e}")
            return None

    def _parse_improvement_recommendations_response(self, response: str) -> Optional[Dict[str, Any]]:
        """개선 권고 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # recommendations가 있는지 확인
                if "recommendations" in parsed:
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse improvement recommendations response: {e}")
            return None

    def _parse_issue_identification_response_with_context(self, response: str, prev_output: Any) -> Optional[Dict[str, Any]]:
        """문제점 식별 응답 파싱 (이전 단계 출력 통합)"""
        try:
            import json
            import re

            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                # issues가 있는지 확인
                if "issues" in parsed:
                    # 이전 단계 결과(key_clauses)도 포함
                    if isinstance(prev_output, dict):
                        parsed["key_clauses"] = prev_output.get("key_clauses", [])
                        parsed["document_type"] = prev_output.get("document_type", "")
                    return parsed

            return None
        except Exception as e:
            self.logger.warning(f"Failed to parse issue identification response: {e}")
            return None

    def _create_document_summary(self, analysis: Dict[str, Any]) -> str:
        """DocumentAnalysisProcessor.create_document_summary 래퍼"""
        if self.document_analysis_processor:
            return self.document_analysis_processor.create_document_summary(analysis)
        return f"## 업로드 문서 분석 ({analysis.get('document_type', 'unknown')})"

    @with_state_optimization("route_expert", enable_reduction=True)
    def route_expert(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """전문가 서브그래프로 라우팅 (Phase 9 리팩토링: WorkflowRoutes 사용)"""
        try:
            start_time = time.time()

            # WorkflowRoutes.route_expert 호출
            state = self.workflow_routes.route_expert(state)

            complexity = state.get("complexity_level", "simple")
            requires_expert = state.get("requires_expert", False)

            self._update_processing_time(state, start_time)
            self._add_step(state, "전문가 라우팅", f"복잡도: {complexity}, 전문가: {requires_expert}")

        except Exception as e:
            self._handle_error(state, str(e), "전문가 라우팅 중 오류")
            state["complexity_level"] = "simple"
            state["requires_expert"] = False
            state["expert_subgraph"] = None

        return state

    # Phase 9 리팩토링: 라우팅 관련 메서드는 WorkflowRoutes로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _assess_complexity(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.assess_complexity 래퍼"""
        return self.workflow_routes.assess_complexity(state)

    def _should_analyze_document(self, state: LegalWorkflowState) -> str:
        """SearchRoutes.should_analyze_document 사용"""
        return self.search_routes.should_analyze_document(state)

    # ============================================================================
    # 개선된 검색 노드들 (노드 분리 및 병렬 실행 지원)
    # ============================================================================
    
    # ============================================================================
    # prepare_search_query 헬퍼 메서드들 (메서드 분해)
    # ============================================================================
    
    def _get_query_info_for_optimization(
        self,
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """쿼리 정보 가져오기 및 검증 (중복 코드 제거)"""
        query = None
        
        if "input" in state and isinstance(state["input"], dict):
            query = state["input"].get("query", "")
        
        if not query or not str(query).strip():
            query = self._get_state_value(state, "query", "")
        
        if not query or not str(query).strip():
            if isinstance(state, dict) and "query" in state:
                query = state["query"]
        
        search_query = self._get_state_value(state, "search_query") or query
        
        if not query or not str(query).strip():
            self.logger.error(f"prepare_search_query: query is empty! State keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            if "input" in state:
                self.logger.error(f"prepare_search_query: state['input'] = {state['input']}")
            return {
                "query": None,
                "search_query": None,
                "query_type_str": "",
                "extracted_keywords": [],
                "legal_field": ""
            }
        
        query_type_raw = self._get_state_value(state, "query_type", "")
        query_type_str = self._get_query_type_str(query_type_raw)
        query_type_str = self._normalize_query_type_for_prompt(query_type_str)
        
        extracted_keywords_raw = self._get_state_value(state, "extracted_keywords", [])
        if not isinstance(extracted_keywords_raw, list):
            self.logger.warning(f"extracted_keywords is not a list: {type(extracted_keywords_raw)}, converting to empty list")
            extracted_keywords = []
        else:
            extracted_keywords = [kw for kw in extracted_keywords_raw if kw and isinstance(kw, str) and len(str(kw).strip()) > 0]
        
        legal_field_raw = self._get_state_value(state, "legal_field", "")
        legal_field = str(legal_field_raw).strip() if legal_field_raw else ""
        
        self.logger.debug(
            f"📋 [PREPARE SEARCH QUERY] Data for query optimization:\n"
            f"   query: '{query[:50]}{'...' if len(query) > 50 else ''}'\n"
            f"   search_query: '{search_query[:50]}{'...' if len(search_query) > 50 else ''}'\n"
            f"   query_type (raw): '{query_type_raw}' → (normalized): '{query_type_str}'\n"
            f"   extracted_keywords: {len(extracted_keywords)} items {extracted_keywords[:5] if extracted_keywords else '[]'}\n"
            f"   legal_field: '{legal_field}'"
        )
        
        return {
            "query": query,
            "search_query": search_query,
            "query_type_str": query_type_str,
            "extracted_keywords": extracted_keywords,
            "legal_field": legal_field
        }
    
    def _optimize_query_with_cache(
        self,
        search_query: str,
        query_type_str: str,
        extracted_keywords: List[str],
        legal_field: str,
        is_retry: bool
    ) -> Tuple[Dict[str, Any], bool]:
        """쿼리 최적화 (캐싱 포함) (중복 코드 제거)"""
        import hashlib
        
        optimized_queries = None
        cache_hit = False
        
        # 쿼리 캐시가 비활성화되지 않은 경우에만 캐시 확인
        if not is_retry and not self.config.disable_query_cache:
            cache_key_parts = [
                search_query,
                query_type_str,
                ",".join(sorted(extracted_keywords)) if extracted_keywords else "",
                legal_field
            ]
            cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
            
            try:
                cached_result = self.performance_optimizer.cache.get_cached_answer(
                    f"query_opt:{cache_key}", query_type_str
                )
                if cached_result and isinstance(cached_result, dict) and "optimized_queries" in cached_result:
                    optimized_queries = cached_result.get("optimized_queries")
                    cache_hit = True
                    self.logger.debug(f"✅ [CACHE HIT] 쿼리 최적화 결과 캐시 히트: {cache_key[:16]}...")
            except Exception as e:
                self.logger.debug(f"캐시 확인 중 오류 (무시): {e}")
        elif self.config.disable_query_cache:
            self.logger.debug("Query cache is disabled, skipping cache check")
        
        if not optimized_queries:
            self.logger.debug(
                f"🔍 [QUERY OPTIMIZATION] Calling _optimize_search_query: "
                f"query='{search_query[:50]}...', query_type={query_type_str}, "
                f"keywords={len(extracted_keywords)}, legal_field={legal_field}"
            )
            optimized_queries = self._optimize_search_query(
                query=search_query,
                query_type=query_type_str,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field
            )
            self.logger.debug(
                f"✅ [QUERY OPTIMIZATION] _optimize_search_query completed: "
                f"llm_enhanced={optimized_queries.get('llm_enhanced', False)}, "
                f"semantic_query_length={len(optimized_queries.get('semantic_query', ''))}"
            )
            
            if not is_retry and not self.config.disable_query_cache:
                try:
                    cache_key_parts = [
                        search_query,
                        query_type_str,
                        ",".join(sorted(extracted_keywords)) if extracted_keywords else "",
                        legal_field
                    ]
                    cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
                    self.performance_optimizer.cache.cache_answer(
                        f"query_opt:{cache_key}",
                        query_type_str,
                        {"optimized_queries": optimized_queries},
                        confidence=1.0,
                        sources=[]
                    )
                    self.logger.debug(f"✅ [CACHE STORE] 쿼리 최적화 결과 캐시 저장: {cache_key[:16]}...")
                except Exception as e:
                    self.logger.debug(f"캐시 저장 중 오류 (무시): {e}")
            elif self.config.disable_query_cache:
                self.logger.debug("Query cache is disabled, not storing result")
        
        return optimized_queries, cache_hit
    
    def _validate_and_fix_optimized_queries(
        self,
        state: LegalWorkflowState,
        optimized_queries: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """최적화된 쿼리 검증 및 수정 (중복 코드 제거)"""
        # Multi-Query 보존
        multi_queries_backup = optimized_queries.get("multi_queries", None)
        
        semantic_query_created = optimized_queries.get("semantic_query", "")
        if not semantic_query_created or not str(semantic_query_created).strip():
            self.logger.warning(f"semantic_query is empty, using base query: '{query[:50]}...'")
            optimized_queries["semantic_query"] = query
            semantic_query_created = query
            self._set_state_value(state, "optimized_queries", optimized_queries)
        
        keyword_queries_created = optimized_queries.get("keyword_queries", [])
        if not keyword_queries_created or len(keyword_queries_created) == 0:
            self.logger.warning("keyword_queries is empty, using base query")
            optimized_queries["keyword_queries"] = [query]
            keyword_queries_created = [query]
            self._set_state_value(state, "optimized_queries", optimized_queries)
        
        # Multi-Query 복원
        if multi_queries_backup:
            optimized_queries["multi_queries"] = multi_queries_backup
        
        return {
            "optimized_queries": optimized_queries,
            "semantic_query_created": semantic_query_created,
            "keyword_queries_created": keyword_queries_created
        }

    @with_state_optimization("prepare_search_query", enable_reduction=False)
    def prepare_search_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 쿼리 준비 및 최적화 전용 노드 (Part 2)"""
        try:
            start_time = time.time()

            preserved = self._preserve_metadata(state, ["query_complexity", "needs_search"])
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"] = dict(state["metadata"])
            state["metadata"].update(preserved)
            state["metadata"]["_last_executed_node"] = "prepare_search_query"
            
            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "prepare_search_query"

            self._ensure_input_group(state)
            query_value, session_id_value = self._restore_query_from_state(state)
            if query_value:
                query_value = self._normalize_query_encoding(query_value)
                state["input"]["query"] = query_value
                if session_id_value:
                    state["input"]["session_id"] = session_id_value

            # 재시도 카운터 관리
            metadata = state.get("metadata", {}) if isinstance(state.get("metadata"), dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}

            # 중요: state.get("common")이 None일 수 있으므로 안전하게 처리
            common_state = state.get("common")
            if common_state and isinstance(common_state, dict):
                common_metadata = common_state.get("metadata", {})
                if isinstance(common_metadata, dict):
                    metadata = {**metadata, **common_metadata}

            last_executed_node = metadata.get("_last_executed_node", "")
            is_retry_from_generation = (last_executed_node == "generate_answer_enhanced")
            is_retry_from_validation = (last_executed_node == "validate_answer_quality")

            # 재시도 카운터 증가
            if is_retry_from_generation:
                if self.retry_manager.should_allow_retry(state, "generation"):
                    self.retry_manager.increment_retry_count(state, "generation")

            if is_retry_from_validation:
                if self.retry_manager.should_allow_retry(state, "validation"):
                    self.retry_manager.increment_retry_count(state, "validation")

            # 재시도 횟수 체크
            retry_counts = self.retry_manager.get_retry_counts(state)
            if retry_counts["total"] >= RetryConfig.MAX_TOTAL_RETRIES:
                self.logger.error("Maximum total retry count reached")
                if not self._get_state_value(state, "answer", ""):
                    query = self._get_state_value(state, "query", "")
                    # Phase 1/Phase 7: 에러 메시지 설정 - _set_answer_safely 사용
                    self._set_answer_safely(state,
                        f"죄송합니다. 질문 '{query}'에 대한 답변을 생성하는데 어려움이 있습니다.")
                return state

            query_info = self._get_query_info_for_optimization(state)
            if not query_info["query"]:
                self._set_answer_safely(state, "죄송합니다. 질문을 이해하지 못했습니다. 다시 질문해주세요.")
                return state
            
            query = query_info["query"]
            search_query = query_info["search_query"]
            query_type_str = query_info["query_type_str"]
            extracted_keywords = query_info["extracted_keywords"]
            legal_field = query_info["legal_field"]

            is_retry = (last_executed_node == "validate_answer_quality")
            complexity = self._get_state_value(state, "complexity_level", "moderate")

            # HybridQueryProcessor 사용 (HuggingFace + LLM 하이브리드)
            if self.hybrid_query_processor:
                try:
                    self.logger.debug("🔍 [HYBRID] Using HybridQueryProcessor for query optimization")
                    optimized_queries, cache_hit_optimization = self.hybrid_query_processor.process_query_hybrid(
                        query=query,
                        search_query=search_query,
                        query_type=query_type_str,
                        extracted_keywords=extracted_keywords,
                        legal_field=legal_field,
                        complexity=complexity,
                        is_retry=is_retry
                    )
                    multi_queries = optimized_queries.get("multi_queries", [search_query])
                    self.logger.debug(
                        "[HYBRID] Query processing completed: "
                        f"semantic_query='{optimized_queries.get('semantic_query', '')[:50]}...', "
                        f"multi_queries={len(multi_queries) if multi_queries else 0}"
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ [HYBRID] HybridQueryProcessor failed: {e}, falling back to QueryEnhancer", exc_info=True)
                    # 폴백: 기존 방식 사용
                    optimized_queries, cache_hit_optimization = self._optimize_query_with_cache(
                        search_query=search_query,
                        query_type_str=query_type_str,
                        extracted_keywords=extracted_keywords,
                        legal_field=legal_field,
                        is_retry=is_retry
                    )
                    multi_queries = self._generate_multi_queries_with_llm(
                        query=search_query,
                        query_type=query_type_str,
                        max_queries=3 if complexity == "simple" else (4 if complexity == "complex" else 3),
                        use_cache=True
                    )
                    if multi_queries and len(multi_queries) > 1:
                        optimized_queries["multi_queries"] = multi_queries
            else:
                # 폴백: 기존 방식 사용
                self.logger.debug("🔍 [FALLBACK] Using QueryEnhancer (HybridQueryProcessor not available)")
                optimized_queries, cache_hit_optimization = self._optimize_query_with_cache(
                    search_query=search_query,
                    query_type_str=query_type_str,
                    extracted_keywords=extracted_keywords,
                    legal_field=legal_field,
                    is_retry=is_retry
                )
                
                # Multi-Query Retrieval 적용 (LLM 기반 질문 재작성)
                multi_queries = None
                self.logger.debug(f"[MULTI-QUERY] Starting multi-query generation for: '{search_query[:50]}...'")
                self.logger.debug(f"🔍 [MULTI-QUERY] Starting multi-query generation for: '{search_query[:50]}...'")
                try:
                    if complexity == "complex":
                        max_queries = 4
                    elif complexity == "moderate":
                        max_queries = 3
                    else:
                        max_queries = 2
                    
                    multi_queries = self._generate_multi_queries_with_llm(
                        query=search_query,
                        query_type=query_type_str,
                        max_queries=max_queries,
                        use_cache=True
                    )
                    self.logger.debug(f"[MULTI-QUERY] Generated {len(multi_queries) if multi_queries else 0} queries")
                    self.logger.debug(f"🔍 [MULTI-QUERY] Generated {len(multi_queries) if multi_queries else 0} queries")
                    
                    if multi_queries and len(multi_queries) > 1:
                        optimized_queries["multi_queries"] = multi_queries
                        if len(multi_queries) > 1:
                            optimized_queries["semantic_query"] = multi_queries[0]
                except Exception as e:
                    self.logger.debug(f"[MULTI-QUERY] Error: {e}")
                    self.logger.warning(f"⚠️ [MULTI-QUERY] Error generating multi-queries: {e}, using original query", exc_info=True)
                    multi_queries = [search_query]

            if is_retry:
                quality_feedback = self.answer_generator.get_quality_feedback_for_retry(state)
                improved_query = self._improve_search_query_for_retry(
                    optimized_queries.get("semantic_query", search_query),
                    quality_feedback,
                    state
                )
                if improved_query != optimized_queries.get("semantic_query", search_query):
                    self.logger.debug(
                        f"🔍 [SEARCH RETRY] Improved query: '{optimized_queries.get('semantic_query', search_query)}' → '{improved_query}'"
                    )
                    optimized_queries["semantic_query"] = improved_query
                    if optimized_queries.get("keyword_queries"):
                        optimized_queries["keyword_queries"][0] = improved_query

            search_params = self._determine_search_parameters(
                query_type=query_type_str,
                query_complexity=len(query),
                keyword_count=len(extracted_keywords),
                is_retry=is_retry
            )

            self._set_state_value(state, "optimized_queries", optimized_queries)
            self._set_state_value(state, "search_params", search_params)
            self._set_state_value(state, "is_retry_search", is_retry)
            self._set_state_value(state, "search_start_time", start_time)

            validated_queries = self._validate_and_fix_optimized_queries(state, optimized_queries, query)
            optimized_queries = validated_queries["optimized_queries"]
            semantic_query_created = validated_queries["semantic_query_created"]
            keyword_queries_created = validated_queries["keyword_queries_created"]
            
            # Multi-Query가 검증 과정에서 손실되지 않도록 보장 (항상 추가)
            if multi_queries and len(multi_queries) > 1:
                optimized_queries["multi_queries"] = multi_queries
                self.logger.debug(f"[MULTI-QUERY] Added multi_queries to optimized_queries after validation: {len(multi_queries)} queries")
                self.logger.debug(f"🔍 [MULTI-QUERY] Added multi_queries to optimized_queries after validation: {len(multi_queries)} queries")
            
            # 검증 후 최종 optimized_queries를 state에 저장
            self.logger.debug(f"[MULTI-QUERY] Saving optimized_queries to state (keys: {list(optimized_queries.keys())}, has_multi_queries: {'multi_queries' in optimized_queries})")
            self._set_state_value(state, "optimized_queries", optimized_queries)
            
            # Global cache에도 저장 (state reduction 대응)
            try:
                from core.shared.wrappers.node_wrappers import _global_search_results_cache
                if _global_search_results_cache is None or not isinstance(_global_search_results_cache, dict):
                    import core.shared.wrappers.node_wrappers as node_wrappers_module
                    node_wrappers_module._global_search_results_cache = {}
                    _global_search_results_cache = node_wrappers_module._global_search_results_cache
                
                if "search" not in _global_search_results_cache:
                    _global_search_results_cache["search"] = {}
                _global_search_results_cache["search"]["optimized_queries"] = optimized_queries.copy()
                self.logger.debug(f"[MULTI-QUERY] Saved optimized_queries to global cache (keys: {list(optimized_queries.keys())})")
            except Exception as e:
                self.logger.debug(f"Failed to save optimized_queries to global cache: {e}")
            
            # 직접 state에 저장 (이중 보장)
            if multi_queries and len(multi_queries) > 1:
                if "optimized_queries" not in state:
                    state["optimized_queries"] = {}
                if not isinstance(state["optimized_queries"], dict):
                    state["optimized_queries"] = {}
                state["optimized_queries"]["multi_queries"] = multi_queries
                self.logger.debug("[MULTI-QUERY] Directly saved multi_queries to state['optimized_queries']")
            self._set_state_value(state, "search_query", semantic_query_created)

            # 캐시 확인 (재시도 시에는 캐시 우회)
            cache_hit = False
            if not is_retry:
                cached_documents = self.performance_optimizer.cache.get_cached_documents(
                    optimized_queries["semantic_query"],
                    query_type_str
                )
                if cached_documents:
                    self._set_state_value(state, "retrieved_docs", cached_documents)
                    self._set_state_value(state, "search_cache_hit", True)
                    cache_hit = True
                    self._add_step(state, "캐시 히트", f"캐시 히트: {len(cached_documents)}개 문서")

            self._set_state_value(state, "search_cache_hit", cache_hit)
            self._save_metadata_safely(state, "_last_executed_node", "prepare_search_query")
            self._update_processing_time(state, start_time)
            self._add_step(state, "검색 쿼리 준비", f"검색 쿼리 준비 완료: {semantic_query_created[:50]}...")

            if cache_hit:
                self.logger.debug(f"✅ [CACHE HIT] 캐시 히트: {len(cached_documents)}개 문서, 검색 스킵")
            else:
                self.logger.debug(
                    f"✅ [PREPARE SEARCH QUERY] "
                    f"semantic_query: '{semantic_query_created[:50]}...', "
                    f"keyword_queries: {len(keyword_queries_created)}개, "
                    f"search_params: k={search_params.get('semantic_k', 'N/A')}"
                )

        except asyncio.CancelledError:
            # 최상위 CancelledError 처리
            self.logger.warning("⚠️ [PREPARE SEARCH QUERY] Operation was cancelled. Preserving existing state.")
            # 기존 상태 보존
            existing_queries = self._get_state_value(state, "optimized_queries")
            if not existing_queries:
                query = self._get_state_value(state, "query", "")
                if query:
                    self._set_state_value(state, "search_query", query)
                    self._set_state_value(state, "optimized_queries", {
                        "semantic_query": query,
                        "keyword_queries": [query]
                    })
            # CancelledError는 다시 발생시키지 않고 상태를 보존한 채로 반환
        except Exception as e:
            self._handle_error(state, str(e), "검색 쿼리 준비 중 오류 발생")

        self._ensure_input_group(state)
        query_value, session_id_value = self._restore_query_from_state(state)
        if query_value:
            state["input"]["query"] = query_value
            if session_id_value:
                state["input"]["session_id"] = session_id_value

        return state

    # Phase 9 리팩토링: 라우팅 관련 메서드는 WorkflowRoutes로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _should_skip_search(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_skip_search 래퍼"""
        return self.workflow_routes.should_skip_search(state)

    def _should_skip_search_adaptive(self, state: LegalWorkflowState) -> str:
        """SearchRoutes.should_skip_search_adaptive 사용"""
        return self.search_routes.should_skip_search_adaptive(state)
    
    def _should_use_multi_query_agent(self, state: LegalWorkflowState) -> str:
        """SearchRoutes.should_use_multi_query_agent 사용"""
        return self.search_routes.should_use_multi_query_agent(state)

    def _route_by_complexity(self, state: LegalWorkflowState) -> str:
        """ClassificationRoutes.route_by_complexity 사용"""
        return self.classification_routes.route_by_complexity(state)
    
    def _route_by_complexity_with_agentic(self, state: LegalWorkflowState) -> str:
        """Agentic 모드용 복잡도 라우팅"""
        return self.classification_routes.route_by_complexity_with_agentic(state)
    
    def _route_after_agentic(self, state: LegalWorkflowState) -> str:
        """Agentic 노드 실행 후 라우팅 (검색 결과 유무에 따라)"""
        return self.agentic_routes.route_after_agentic(state)
    
    # ============================================================================
    # process_search_results_combined 헬퍼 메서드들 (메서드 분해)
    # ============================================================================
    
    def _evaluate_search_quality_internal(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """검색 품질 평가 (WorkflowValidator 사용)"""
        return self.workflow_validator.evaluate_search_quality(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
            query_type_str=query_type_str,
            search_params=search_params,
            evaluate_semantic_func=self._evaluate_semantic_search_quality,
            evaluate_keyword_func=self._evaluate_keyword_search_quality
        )
    
    def _merge_search_results_internal(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str = "",
        query_type: str = "general_question",
        extracted_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """검색 결과 병합 (SearchHandler 사용, 개선 기능 포함)"""
        if self.search_handler:
            return self.search_handler.merge_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
        else:
            # 폴백: SearchResultProcessor 사용
            return self.search_result_processor.merge_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                result_merger=self.result_merger
            )
    
    def _apply_keyword_weights_to_docs(
        self,
        merged_docs: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """키워드 가중치 적용 (SearchResultProcessor 사용)"""
        return self.search_result_processor.apply_keyword_weights_to_docs(
            merged_docs=merged_docs,
                keyword_weights=keyword_weights,
            query=query,
            query_type_str=query_type_str,
            search_params=search_params
        )
    
    def _apply_citation_boost(
        self,
        weighted_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Citation 부스트 적용 (SearchResultProcessor 사용)"""
        return self.search_result_processor.apply_citation_boost(weighted_docs)
    
    def _filter_documents_internal(
        self,
        weighted_docs: List[Dict[str, Any]],
        max_docs: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """문서 필터링 (SearchResultProcessor 사용)"""
        return self.search_result_processor.filter_documents(weighted_docs, max_docs)
    
    def _prepare_search_inputs(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """검색 결과 입력 데이터 준비"""
        semantic_results = self._get_state_value(state, "semantic_results", [])
        keyword_results = self._get_state_value(state, "keyword_results", [])
        semantic_count = self._get_state_value(state, "semantic_count", 0)
        keyword_count = self._get_state_value(state, "keyword_count", 0)
        query = self._get_state_value(state, "query", "")
        query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
        search_params = self._get_state_value(state, "search_params", {})
        extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
        
        # 🔥 디버그: state 직접 확인 (모든 가능한 위치 확인)
        direct_semantic = state.get("semantic_results", []) if isinstance(state, dict) else []
        direct_search_results = state.get("search", {}).get("results", []) if isinstance(state.get("search"), dict) else []
        direct_search_semantic = state.get("search", {}).get("semantic_results", []) if isinstance(state.get("search"), dict) else []
        direct_common_semantic = state.get("common", {}).get("search", {}).get("semantic_results", []) if isinstance(state.get("common", {}).get("search"), dict) else []
        direct_retrieved_docs = state.get("retrieved_docs", []) if isinstance(state, dict) else []
        
        self.logger.info(f"📥 [SEARCH RESULTS] Debug - _get_state_value semantic: {len(semantic_results)}, direct state['semantic_results']: {len(direct_semantic)}, state['search']['results']: {len(direct_search_results)}, state['search']['semantic_results']: {len(direct_search_semantic)}, state['common']['search']['semantic_results']: {len(direct_common_semantic)}, retrieved_docs: {len(direct_retrieved_docs)}")
        
        # 🔥 multi-query 결과 찾기 (여러 위치에서 확인)
        if not semantic_results:
            # 1. state["search"]["semantic_results"] 확인
            if direct_search_semantic:
                semantic_results = direct_search_semantic
                semantic_count = len(direct_search_semantic)
                self.logger.info(f"📥 [SEARCH RESULTS] Found semantic_results in state['search']['semantic_results']: {len(semantic_results)} docs")
            # 2. state["search"]["results"] 확인
            elif direct_search_results:
                # sub_query 필드가 있는 경우 multi-query 결과로 간주
                has_sub_query = any(doc.get("sub_query") or doc.get("multi_query_source") for doc in direct_search_results if isinstance(doc, dict))
                if has_sub_query:
                    semantic_results = direct_search_results
                    semantic_count = len(direct_search_results)
                    self.logger.info(f"📥 [SEARCH RESULTS] Multi-query results found in state['search']['results']: {len(semantic_results)} docs")
                else:
                    # sub_query 필드가 없어도 multi-query 결과일 수 있음
                    semantic_results = direct_search_results
                    semantic_count = len(direct_search_results)
                    self.logger.info(f"📥 [SEARCH RESULTS] Found results in state['search']['results'] (no sub_query): {len(semantic_results)} docs")
            # 3. state["common"]["search"]["semantic_results"] 확인
            elif direct_common_semantic:
                semantic_results = direct_common_semantic
                semantic_count = len(direct_common_semantic)
                self.logger.info(f"📥 [SEARCH RESULTS] Found semantic_results in state['common']['search']['semantic_results']: {len(semantic_results)} docs")
            # 4. state["semantic_results"] 확인
            elif direct_semantic:
                semantic_results = direct_semantic
                semantic_count = len(direct_semantic)
                self.logger.info(f"📥 [SEARCH RESULTS] Found semantic_results in state: {len(semantic_results)} docs")
            # 5. retrieved_docs에서 확인 (multi-query 결과가 여기에만 있을 수 있음)
            elif direct_retrieved_docs:
                # sub_query 필드가 있는 경우 multi-query 결과로 간주
                has_sub_query = any(doc.get("sub_query") or doc.get("multi_query_source") for doc in direct_retrieved_docs if isinstance(doc, dict))
                if has_sub_query:
                    semantic_results = direct_retrieved_docs
                    semantic_count = len(direct_retrieved_docs)
                    self.logger.info(f"📥 [SEARCH RESULTS] Multi-query results found in retrieved_docs: {len(semantic_results)} docs")
        
        self.logger.info(f"📥 [SEARCH RESULTS] 최종 입력 데이터 - semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, semantic_count: {semantic_count}, keyword_count: {keyword_count}")
        
        return {
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "semantic_count": semantic_count,
            "keyword_count": keyword_count,
            "query": query,
            "query_type_str": query_type_str,
            "search_params": search_params,
            "extracted_keywords": extracted_keywords
        }
    
    def _perform_conditional_retry_search(
        self,
        state: LegalWorkflowState,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_count: int,
        keyword_count: int,
        quality_evaluation: Dict[str, Any],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any],
        extracted_keywords: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        """조건부 재검색 수행"""
        semantic_quality = quality_evaluation["semantic_quality"]
        keyword_quality = quality_evaluation["keyword_quality"]
        overall_quality = quality_evaluation["overall_quality"]
        needs_retry = quality_evaluation["needs_retry"]
        
        if needs_retry and overall_quality < 0.6 and semantic_count + keyword_count < 10:
            self.logger.debug(f"검색 품질 낮음 (점수: {overall_quality:.2f}), 재검색 수행...")
            try:
                retry_semantic = []
                retry_keyword = []
                
                if semantic_quality["needs_retry"]:
                    optimized_queries = self._get_state_value(state, "optimized_queries", {})
                    retry_extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
                    retry_semantic = self._execute_semantic_search_internal(
                        optimized_queries, search_params, query, retry_extracted_keywords
                    )[0][:5]
                
                if keyword_quality["needs_retry"]:
                    optimized_queries = self._get_state_value(state, "optimized_queries", {})
                    retry_keyword = self._execute_keyword_search_internal(
                        optimized_queries, search_params, query_type_str,
                        self._get_state_value(state, "legal_field", ""),
                        extracted_keywords, query
                    )[0][:5]
                
                semantic_results.extend(retry_semantic)
                keyword_results.extend(retry_keyword)
                semantic_count += len(retry_semantic)
                keyword_count += len(retry_keyword)
            except Exception as e:
                self.logger.warning(f"재검색 실패: {e}")
        
        return semantic_results, keyword_results, semantic_count, keyword_count
    
    def _consolidate_expanded_query_results(
        self,
        semantic_results: List[Dict[str, Any]],
        original_query: str
    ) -> List[Dict[str, Any]]:
        """
        확장된 쿼리 결과 통합 및 중복 제거 (최소 변경 버전)
        
        Query Expansion으로 생성된 여러 쿼리 결과를:
        1. 쿼리별로 그룹화하여 추적
        2. ID 및 Content Hash 기반 중복 제거
        3. 점수 정규화 및 쿼리별 가중치 적용
        
        Args:
            semantic_results: 확장된 쿼리들의 검색 결과 리스트
            original_query: 원본 쿼리
        
        Returns:
            통합 및 중복 제거된 결과 리스트
        """
        if not semantic_results:
            return semantic_results
        
        try:
            import hashlib
            
            # 1. 쿼리별 그룹화 및 통계 수집
            results_by_query = {}
            for doc in semantic_results:
                # 🔥 개선: 메타데이터 보존 (metadata에서 최상위 필드로 복사)
                if "metadata" not in doc:
                    doc["metadata"] = {}
                if not isinstance(doc["metadata"], dict):
                    doc["metadata"] = {}
                
                metadata = doc["metadata"]
                # metadata에서 최상위 필드로 복사 (우선순위: metadata > 최상위 필드)
                if metadata.get("type") and not doc.get("type"):
                    doc["type"] = metadata.get("type")
                
                # 🔥 개선: type 필드 정규화 (메타데이터 보존)
                # type이 없거나 unknown이면 metadata에서 복원
                current_type = doc.get("type", "").lower()
                if not doc.get("type") or current_type == "unknown":
                    metadata_type = metadata.get("type")
                    if metadata_type and metadata_type != "unknown":
                        doc["type"] = metadata_type
                    elif not doc.get("type"):
                        # DocumentType.from_metadata로 추론
                        from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                        doc_type_enum = DocumentType.from_metadata(doc)
                        if doc_type_enum != DocumentType.UNKNOWN:
                            doc["type"] = doc_type_enum.value
                            metadata["type"] = doc_type_enum.value
                if metadata.get("statute_name") and not doc.get("statute_name"):
                    doc["statute_name"] = metadata.get("statute_name")
                if metadata.get("law_name") and not doc.get("law_name"):
                    doc["law_name"] = metadata.get("law_name")
                if metadata.get("article_no") and not doc.get("article_no"):
                    doc["article_no"] = metadata.get("article_no")
                if metadata.get("case_id") and not doc.get("case_id"):
                    doc["case_id"] = metadata.get("case_id")
                if metadata.get("court") and not doc.get("court"):
                    doc["court"] = metadata.get("court")
                if metadata.get("doc_id") and not doc.get("doc_id"):
                    doc["doc_id"] = metadata.get("doc_id")
                if metadata.get("casenames") and not doc.get("casenames"):
                    doc["casenames"] = metadata.get("casenames")
                if metadata.get("precedent_id") and not doc.get("precedent_id"):
                    doc["precedent_id"] = metadata.get("precedent_id")
                
                # 최상위 필드를 metadata에도 복사 (일관성 유지)
                if doc.get("type") and not metadata.get("type"):
                    metadata["type"] = doc.get("type")
                if doc.get("statute_name") and not metadata.get("statute_name"):
                    metadata["statute_name"] = doc.get("statute_name")
                if doc.get("law_name") and not metadata.get("law_name"):
                    metadata["law_name"] = doc.get("law_name")
                if doc.get("article_no") and not metadata.get("article_no"):
                    metadata["article_no"] = doc.get("article_no")
                if doc.get("case_id") and not metadata.get("case_id"):
                    metadata["case_id"] = doc.get("case_id")
                if doc.get("court") and not metadata.get("court"):
                    metadata["court"] = doc.get("court")
                if doc.get("doc_id") and not metadata.get("doc_id"):
                    metadata["doc_id"] = doc.get("doc_id")
                if doc.get("casenames") and not metadata.get("casenames"):
                    metadata["casenames"] = doc.get("casenames")
                if doc.get("precedent_id") and not metadata.get("precedent_id"):
                    metadata["precedent_id"] = doc.get("precedent_id")
                
                query_id = (
                    doc.get("expanded_query_id") or 
                    doc.get("sub_query") or 
                    doc.get("query_variation") or
                    doc.get("source_query") or
                    doc.get("multi_query_source") or
                    "original"
                )
                if query_id not in results_by_query:
                    results_by_query[query_id] = []
                results_by_query[query_id].append(doc)
            
            # 통계 로깅
            if len(results_by_query) > 1:
                query_stats = {qid: len(results) for qid, results in results_by_query.items()}
                self.logger.info(
                    f"🔄 [MERGE EXPANDED] Found {len(results_by_query)} query sources: {query_stats}"
                )
            
            # 2. 다층 중복 제거
            seen_ids = set()
            seen_content_hashes = {}  # content_hash -> (doc, score)
            consolidated = []
            
            # 원본 쿼리 결과를 먼저 처리 (높은 우선순위)
            for query_id in sorted(results_by_query.keys(), key=lambda x: 0 if x == "original" else 1):
                results = results_by_query[query_id]
                query_weight = 1.0 if query_id == "original" else 0.9
                
                for doc in results:
                    # Layer 1: ID 기반 중복 제거
                    doc_id = (
                        doc.get("id") or 
                        doc.get("doc_id") or 
                        doc.get("document_id") or
                        doc.get("metadata", {}).get("chunk_id") or
                        doc.get("chunk_id")
                    )
                    
                    if doc_id and doc_id in seen_ids:
                        continue
                    
                    # Layer 2: Content Hash 기반 중복 제거
                    content = doc.get("content") or doc.get("text", "")
                    content_hash = None
                    if content:
                        content_hash = hashlib.md5(content[:500].encode('utf-8')).hexdigest()
                    
                    if content_hash:
                        if content_hash in seen_content_hashes:
                            # 중복 발견: 더 높은 점수를 가진 결과로 교체
                            existing_doc, existing_score = seen_content_hashes[content_hash]
                            new_score = doc.get("relevance_score", doc.get("similarity", 0.0)) * query_weight
                            
                            if new_score > existing_score:
                                # 기존 결과 제거하고 새 결과 추가 (성능 최적화: 인덱스 사용)
                                # 🔥 개선: 기존 문서의 메타데이터를 새 문서에 복사 (메타데이터 보존)
                                existing_metadata = existing_doc.get("metadata", {})
                                if isinstance(existing_metadata, dict):
                                    if "metadata" not in doc:
                                        doc["metadata"] = {}
                                    if not isinstance(doc["metadata"], dict):
                                        doc["metadata"] = {}
                                    # 기존 문서의 메타데이터를 새 문서에 복사 (우선순위: 기존 > 새)
                                    for key in ["type", "statute_name", "law_name", "article_no", 
                                               "case_id", "court", "doc_id", "casenames", "precedent_id"]:
                                        if existing_metadata.get(key) and not doc["metadata"].get(key):
                                            doc["metadata"][key] = existing_metadata.get(key)
                                        if existing_doc.get(key) and not doc.get(key):
                                            doc[key] = existing_doc.get(key)
                                
                                existing_idx = None
                                for idx, consolidated_doc in enumerate(consolidated):
                                    if consolidated_doc is existing_doc:
                                        existing_idx = idx
                                        break
                                
                                if existing_idx is not None:
                                    consolidated[existing_idx] = doc
                                else:
                                    consolidated.append(doc)
                                seen_content_hashes[content_hash] = (doc, new_score)
                                if doc_id:
                                    seen_ids.add(doc_id)
                            # 점수가 같거나 낮으면 무시
                            continue
                        else:
                            # 새로운 content hash
                            score = doc.get("relevance_score", doc.get("similarity", 0.0)) * query_weight
                            seen_content_hashes[content_hash] = (doc, score)
                    
                    # 중복이 아니므로 추가
                    if doc_id:
                        seen_ids.add(doc_id)
                    
                    # 쿼리 정보 및 가중치 저장
                    doc["source_query"] = query_id
                    doc["query_weight"] = query_weight
                    original_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                    doc["weighted_score"] = original_score * query_weight
                    
                    consolidated.append(doc)
            
            # 3. 가중치가 적용된 점수 기준 정렬
            consolidated.sort(key=lambda x: x.get("weighted_score", x.get("relevance_score", 0.0)), reverse=True)
            
            # 통계 로깅 (상세 정보 포함)
            removed_count = len(semantic_results) - len(consolidated)
            if removed_count > 0 or len(results_by_query) > 1:
                # 쿼리별 최종 결과 수 계산
                final_by_query = {}
                for doc in consolidated:
                    query_id = doc.get("source_query", "unknown")
                    final_by_query[query_id] = final_by_query.get(query_id, 0) + 1
                
                self.logger.info(
                    f"🔄 [MERGE EXPANDED] Consolidation: {len(semantic_results)} → {len(consolidated)} "
                    f"(removed {removed_count} duplicates, sources: {len(results_by_query)})"
                )
                if len(results_by_query) > 1:
                    self.logger.debug(
                        f"   Query distribution - Before: {query_stats}, After: {final_by_query}"
                    )
            
            return consolidated
            
        except Exception as e:
            self.logger.warning(
                f"⚠️ [MERGE EXPANDED] Error in consolidate_expanded_query_results: {e}, "
                f"returning original results"
            )
            return semantic_results
    
    def _merge_and_rerank_results(
        self,
        state: LegalWorkflowState,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """병합 및 재순위 (개선 기능 포함)"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        # query_type과 extracted_keywords 추출
        query_type = self._get_state_value(state, "query_type", "general_question")
        if isinstance(query_type, dict):
            query_type = query_type.get("type", "general_question")
        extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
        
        # 검색 타입 라벨링 강화: semantic_results와 keyword_results에 search_type 명시적으로 설정
        # 병합 전에 원본 리스트에 search_type을 강제로 설정하여 병합 과정에서 손실 방지
        for doc in semantic_results:
            if not isinstance(doc, dict):
                continue
            # search_type이 없거나 빈 문자열이면 강제로 설정
            if not doc.get("search_type"):
                doc["search_type"] = "semantic"
            # metadata에도 저장하여 보존
            if "metadata" not in doc:
                doc["metadata"] = {}
            if not isinstance(doc["metadata"], dict):
                doc["metadata"] = {}
            doc["metadata"]["original_search_type"] = "semantic"
        
        for doc in keyword_results:
            if not isinstance(doc, dict):
                continue
            # search_type이 없거나 빈 문자열이면 강제로 설정
            if not doc.get("search_type"):
                doc["search_type"] = "keyword"
            # metadata에도 저장하여 보존
            if "metadata" not in doc:
                doc["metadata"] = {}
            if not isinstance(doc["metadata"], dict):
                doc["metadata"] = {}
            doc["metadata"]["original_search_type"] = "keyword"
        
        # 🔥 개선: 메타데이터 및 type 필드 정규화 (중복 코드 제거)
        semantic_results = self._normalize_documents(semantic_results)
        keyword_results = self._normalize_documents(keyword_results)
        
        if self.search_handler and semantic_results and keyword_results:
            # 개선된 merge_search_results 사용
            merged_docs = self.search_handler.merge_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            self.logger.debug(f"🔀 [MERGE] Using improved merge_search_results: {len(merged_docs)} docs")
            
            # 🔥 개선: merge_search_results 반환 후 메타데이터 복원 (원본 문서에서)
            # 원본 문서를 ID와 content 해시로 매핑하여 메타데이터 복원
            original_docs_by_id = {}
            original_docs_by_content = {}
            for doc in semantic_results + keyword_results:
                if isinstance(doc, dict):
                    doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                    if doc_id:
                        original_docs_by_id[doc_id] = doc
                    # content 기반 매칭도 추가
                    content = doc.get("text") or doc.get("content", "")
                    if content:
                        import hashlib
                        content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                        original_docs_by_content[content_hash] = doc
            
            # merged_docs의 메타데이터를 원본 문서에서 복원
            for merged_doc in merged_docs:
                if not isinstance(merged_doc, dict):
                    continue
                
                # ID 기반 매칭 시도
                merged_id = merged_doc.get("id") or merged_doc.get("chunk_id") or merged_doc.get("document_id")
                original_doc = None
                
                if merged_id and merged_id in original_docs_by_id:
                    original_doc = original_docs_by_id[merged_id]
                else:
                    # content 기반 매칭 시도
                    content = merged_doc.get("text") or merged_doc.get("content", "")
                    if content:
                        import hashlib
                        content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                        if content_hash in original_docs_by_content:
                            original_doc = original_docs_by_content[content_hash]
                
                if original_doc:
                    # 🔥 CRITICAL: metadata의 source_type을 type으로 변환 (레거시 호환)
                    original_metadata = original_doc.get("metadata", {})
                    if isinstance(original_metadata, dict):
                        if original_metadata.get("source_type") and not merged_doc.get("type"):
                            merged_doc["type"] = original_metadata.get("source_type")
                            if "metadata" not in merged_doc:
                                merged_doc["metadata"] = {}
                            if not isinstance(merged_doc["metadata"], dict):
                                merged_doc["metadata"] = {}
                            merged_doc["metadata"]["type"] = original_metadata.get("source_type")
                    
                    # 원본 문서의 메타데이터를 merged_doc에 복원
                    # 🔥 개선: unknown 타입도 복원하도록 수정
                    current_type = merged_doc.get("type", "").lower()
                    if original_doc.get("type") and (not merged_doc.get("type") or current_type == "unknown"):
                        merged_doc["type"] = original_doc.get("type")
                    
                    # 법령/판례 관련 필드 복원
                    for key in ["statute_name", "law_name", "article_no", "article_number", 
                               "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id"]:
                        if not merged_doc.get(key) and original_doc.get(key):
                            merged_doc[key] = original_doc.get(key)
                    
                    # metadata에도 복원
                    if "metadata" not in merged_doc:
                        merged_doc["metadata"] = {}
                    if not isinstance(merged_doc["metadata"], dict):
                        merged_doc["metadata"] = {}
                    
                    original_metadata = original_doc.get("metadata", {})
                    if isinstance(original_metadata, dict):
                        # 원본 metadata의 필드를 merged_doc의 metadata에 복원
                        for key in ["type", "statute_name", "law_name", "article_no", 
                                   "article_number", "case_id", "court", "ccourt", "doc_id", 
                                   "casenames", "precedent_id"]:
                            if original_metadata.get(key) and not merged_doc["metadata"].get(key):
                                merged_doc["metadata"][key] = original_metadata.get(key)
        else:
            # 🔥 CRITICAL: _merge_search_results_internal 호출 전에 원본 문서의 타입 정보 백업
            original_docs_by_id = {}
            original_docs_by_content = {}
            for doc in semantic_results + keyword_results:
                if isinstance(doc, dict):
                    doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                    if doc_id:
                        original_docs_by_id[doc_id] = doc
                    # content 기반 매칭도 추가
                    content = doc.get("text") or doc.get("content", "")
                    if content:
                        import hashlib
                        content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                        original_docs_by_content[content_hash] = doc
            
            merged_docs = self._merge_search_results_internal(
                semantic_results, 
                keyword_results,
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            self.logger.debug(f"🔀 [MERGE] Using _merge_search_results_internal: {len(merged_docs)} docs")
            
            # 🔥 CRITICAL: _merge_search_results_internal 호출 후 원본 문서에서 타입 복원
            for merged_doc in merged_docs:
                if not isinstance(merged_doc, dict):
                    continue
                
                # ID 기반 매칭 시도
                merged_id = merged_doc.get("id") or merged_doc.get("chunk_id") or merged_doc.get("document_id")
                original_doc = None
                
                if merged_id and merged_id in original_docs_by_id:
                    original_doc = original_docs_by_id[merged_id]
                else:
                    # content 기반 매칭 시도
                    content = merged_doc.get("text") or merged_doc.get("content", "")
                    if content:
                        import hashlib
                        content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                        if content_hash in original_docs_by_content:
                            original_doc = original_docs_by_content[content_hash]
                
                if original_doc:
                    # 🔥 CRITICAL: 타입 복원 (원본 문서에서)
                    current_type = merged_doc.get("type", "").lower()
                    original_type = original_doc.get("type", "").lower()
                    
                    if original_type and original_type != "unknown":
                        if (not merged_doc.get("type") or current_type == "unknown" or current_type == ""):
                            merged_doc["type"] = original_doc.get("type")
                            self.logger.info(
                                f"🔍 [TYPE RESTORE AFTER MERGE] Doc ID={merged_id}: "
                                f"타입 복원: {current_type} → {original_type}"
                            )
                    
                    # metadata에서도 타입 복원 시도
                    original_metadata = original_doc.get("metadata", {})
                    if isinstance(original_metadata, dict):
                        metadata_type = original_metadata.get("type") or original_metadata.get("source_type")
                        if metadata_type and metadata_type.lower() != "unknown":
                            if (not merged_doc.get("type") or merged_doc.get("type", "").lower() == "unknown"):
                                merged_doc["type"] = metadata_type
                                if "metadata" not in merged_doc:
                                    merged_doc["metadata"] = {}
                                if not isinstance(merged_doc["metadata"], dict):
                                    merged_doc["metadata"] = {}
                                merged_doc["metadata"]["type"] = metadata_type
                    
                    # 법령/판례 관련 필드 복원
                    for key in ["statute_name", "law_name", "article_no", "article_number", 
                               "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id"]:
                        if not merged_doc.get(key) and original_doc.get(key):
                            merged_doc[key] = original_doc.get(key)
                    
                    # metadata에도 복원
                    if "metadata" not in merged_doc:
                        merged_doc["metadata"] = {}
                    if not isinstance(merged_doc["metadata"], dict):
                        merged_doc["metadata"] = {}
                    
                    if isinstance(original_metadata, dict):
                        for key in ["type", "statute_name", "law_name", "article_no", 
                                   "article_number", "case_id", "court", "ccourt", "doc_id", 
                                   "casenames", "precedent_id"]:
                            if original_metadata.get(key) and not merged_doc["metadata"].get(key):
                                merged_doc["metadata"][key] = original_metadata.get(key)
        
        # 병합 후에도 search_type이 없는 문서에 대해 강화된 추론 로직 적용
        for doc in merged_docs:
            if not doc.get("search_type"):
                # 1. metadata에서 원본 search_type 복원 시도
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict) and metadata.get("original_search_type"):
                    doc["search_type"] = metadata["original_search_type"]
                    continue
            
            # 🔥 개선: type 필드 정규화 (메타데이터 보존)
            # type이 없거나 unknown이면 metadata에서 복원
            current_type = doc.get("type", "").lower()
            if not doc.get("type") or current_type == "unknown":
                metadata_type = doc.get("metadata", {}).get("type")
                if metadata_type and metadata_type != "unknown":
                    doc["type"] = metadata_type
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    if not isinstance(doc["metadata"], dict):
                        doc["metadata"] = {}
                    doc["metadata"]["type"] = metadata_type
            
            # 2. type으로 추론
            doc_type = doc.get("type", "").lower()
            
            # 법령/판례 관련 문서는 semantic으로 분류
            if doc_type in ["statute_article", "precedent_content", "statute", "precedent"]:
                doc["search_type"] = "semantic"
            # law_name이나 article_no가 있으면 semantic (법령 조문)
            elif doc.get("law_name") or doc.get("article_no"):
                doc["search_type"] = "semantic"
            # case_number나 court_name이 있으면 semantic (판례)
            elif doc.get("case_number") or doc.get("court_name"):
                doc["search_type"] = "semantic"
            # direct_match가 있으면 database
            elif doc.get("direct_match", False):
                doc["search_type"] = "database"
            else:
                # 기본적으로 keyword로 분류
                doc["search_type"] = "keyword"
            
            # search_type이 있더라도 metadata에 원본 정보 보존
            if "metadata" not in doc:
                doc["metadata"] = {}
            if not isinstance(doc["metadata"], dict):
                doc["metadata"] = {}
            if "original_search_type" not in doc["metadata"]:
                doc["metadata"]["original_search_type"] = doc.get("search_type", "unknown")
        
        # 🔥 개선: merged_docs에 메타데이터 보존 로직 추가
        # merge_search_results 또는 _merge_search_results_internal에서 반환된 문서의 메타데이터 보존
        for doc in merged_docs:
            # metadata 필드 보존 및 보강
            if "metadata" not in doc:
                doc["metadata"] = {}
            if not isinstance(doc["metadata"], dict):
                doc["metadata"] = {}
            
            # 기존 metadata의 정보를 최상위 필드로 복사 (우선순위: metadata > 최상위 필드)
            metadata = doc["metadata"]
            if isinstance(metadata, dict):
                # metadata에서 최상위 필드로 복사
                if metadata.get("type") and not doc.get("type"):
                    doc["type"] = metadata.get("type")
                if metadata.get("statute_name") and not doc.get("statute_name"):
                    doc["statute_name"] = metadata.get("statute_name")
                if metadata.get("law_name") and not doc.get("law_name"):
                    doc["law_name"] = metadata.get("law_name")
                if metadata.get("article_no") and not doc.get("article_no"):
                    doc["article_no"] = metadata.get("article_no")
                if metadata.get("article_number") and not doc.get("article_no"):
                    doc["article_no"] = metadata.get("article_number")
                if metadata.get("case_id") and not doc.get("case_id"):
                    doc["case_id"] = metadata.get("case_id")
                if metadata.get("court") and not doc.get("court"):
                    doc["court"] = metadata.get("court")
                if metadata.get("ccourt") and not doc.get("court"):
                    doc["court"] = metadata.get("ccourt")
                if metadata.get("doc_id") and not doc.get("doc_id"):
                    doc["doc_id"] = metadata.get("doc_id")
                if metadata.get("casenames") and not doc.get("casenames"):
                    doc["casenames"] = metadata.get("casenames")
                if metadata.get("precedent_id") and not doc.get("precedent_id"):
                    doc["precedent_id"] = metadata.get("precedent_id")
            
            # 최상위 필드의 정보를 metadata에 복사 (DocumentType 추론을 위해)
            if doc.get("statute_name") or doc.get("law_name") or doc.get("article_no"):
                doc["metadata"]["statute_name"] = doc.get("statute_name") or doc.get("law_name")
                doc["metadata"]["law_name"] = doc.get("law_name") or doc.get("statute_name")
                doc["metadata"]["article_no"] = doc.get("article_no") or doc.get("article_number")
            
            if doc.get("case_id") or doc.get("court") or doc.get("doc_id") or doc.get("casenames"):
                doc["metadata"]["case_id"] = doc.get("case_id")
                doc["metadata"]["court"] = doc.get("court") or doc.get("ccourt")
                doc["metadata"]["doc_id"] = doc.get("doc_id")
                doc["metadata"]["casenames"] = doc.get("casenames")
                doc["metadata"]["precedent_id"] = doc.get("precedent_id")
            
            # type도 metadata에 복사
            if doc.get("type"):
                doc["metadata"]["type"] = doc.get("type")
        
        if debug_mode:
            doc_structure_stats = {
                "total": len(merged_docs),
                "has_content": 0,
                "has_text": 0,
                "has_both": 0,
                "content_lengths": [],
                "search_types": {}
            }
            for doc in merged_docs[:3]:
                has_content = bool(doc.get("content", ""))
                has_text = bool(doc.get("text", ""))
                content_len = len(doc.get("content", "") or doc.get("text", "") or "")
                search_type = doc.get("search_type", "unknown")
                if has_content:
                    doc_structure_stats["has_content"] += 1
                if has_text:
                    doc_structure_stats["has_text"] += 1
                if has_content and has_text:
                    doc_structure_stats["has_both"] += 1
                doc_structure_stats["content_lengths"].append(content_len)
                doc_structure_stats["search_types"][search_type] = doc_structure_stats["search_types"].get(search_type, 0) + 1
            
            self.logger.debug(f"📋 [SEARCH RESULTS] merged_docs 구조 분석 - Total: {doc_structure_stats['total']}, Has content: {doc_structure_stats['has_content']}, Has text: {doc_structure_stats['has_text']}, Has both: {doc_structure_stats['has_both']}, Avg content length: {sum(doc_structure_stats['content_lengths'])/len(doc_structure_stats['content_lengths']) if doc_structure_stats['content_lengths'] else 0:.1f}, Search types: {doc_structure_stats['search_types']}")
        
        return merged_docs
    
    def _apply_keyword_weights_and_rerank(
        self,
        state: LegalWorkflowState,
        merged_docs: List[Dict[str, Any]],
        query: str,
        query_type_str: str,
        extracted_keywords: List[str],
        search_params: Dict[str, Any],
        overall_quality: float
    ) -> List[Dict[str, Any]]:
        """키워드 가중치 적용 및 재정렬"""
        # 즉시 확인을 위한 print 문 추가 (로그 파일 기록 문제 대비)
        self.logger.debug(f"[RERANK ENTRY] _apply_keyword_weights_and_rerank called: merged_docs={len(merged_docs)}, quality={overall_quality:.2f}")
        self.logger.debug(
            f"🔍 [RERANK ENTRY] _apply_keyword_weights_and_rerank called: "
            f"merged_docs={len(merged_docs)}, query='{query[:50]}...', quality={overall_quality:.2f}"
        )
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        # 🔥 CRITICAL: 메타데이터 복원 (문서 분류 전에 수행)
        # 원본 검색 결과에서 메타데이터를 가져와서 merged_docs에 복원
        semantic_results = self._get_state_value(state, "semantic_results", [])
        keyword_results = self._get_state_value(state, "keyword_results", [])
        original_docs = semantic_results + keyword_results
        
        # 원본 문서를 ID와 content 해시로 매핑
        original_docs_by_id = {}
        original_docs_by_content = {}
        for doc in original_docs:
            if isinstance(doc, dict):
                doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                if doc_id:
                    original_docs_by_id[doc_id] = doc
                # content 기반 매칭도 추가
                content = doc.get("text") or doc.get("content", "")
                if content:
                    import hashlib
                    content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                    original_docs_by_content[content_hash] = doc
        
        # merged_docs의 메타데이터를 원본 문서에서 복원
        restored_count = 0
        for merged_doc in merged_docs:
            if not isinstance(merged_doc, dict):
                continue
            
            # ID 기반 매칭 시도
            merged_id = merged_doc.get("id") or merged_doc.get("chunk_id") or merged_doc.get("document_id")
            original_doc = None
            
            if merged_id and merged_id in original_docs_by_id:
                original_doc = original_docs_by_id[merged_id]
            else:
                # content 기반 매칭 시도
                content = merged_doc.get("text") or merged_doc.get("content", "")
                if content:
                    import hashlib
                    content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                    if content_hash in original_docs_by_content:
                        original_doc = original_docs_by_content[content_hash]
            
            if original_doc:
                # 원본 문서의 메타데이터를 merged_doc에 복원
                # 🔥 개선: unknown 타입도 복원하도록 수정
                current_type = merged_doc.get("type", "").lower()
                if original_doc.get("type") and (not merged_doc.get("type") or current_type == "unknown"):
                    merged_doc["type"] = original_doc.get("type")
                    restored_count += 1
                
                # 법령/판례 관련 필드 복원
                for key in ["statute_name", "law_name", "article_no", "article_number",
                           "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id"]:
                    if not merged_doc.get(key) and original_doc.get(key):
                        merged_doc[key] = original_doc.get(key)
                
                # metadata에도 복원
                if "metadata" not in merged_doc:
                    merged_doc["metadata"] = {}
                if not isinstance(merged_doc["metadata"], dict):
                    merged_doc["metadata"] = {}
                
                original_metadata = original_doc.get("metadata", {})
                if isinstance(original_metadata, dict):
                    for key in ["type", "statute_name", "law_name", "article_no",
                               "article_number", "case_id", "court", "ccourt", "doc_id",
                               "casenames", "precedent_id"]:
                        if original_metadata.get(key) and not merged_doc["metadata"].get(key):
                            merged_doc["metadata"][key] = original_metadata.get(key)
        
        if restored_count > 0:
            self.logger.info(
                f"✅ [METADATA RESTORE] _apply_keyword_weights_and_rerank 시작 시 "
                f"메타데이터 복원: {restored_count}/{len(merged_docs)}개 문서"
            )
        
        # extracted_keywords가 비어있을 때 쿼리에서 추출 시도
        if not extracted_keywords or len(extracted_keywords) == 0:
            self.logger.warning(
                f"⚠️ [KEYWORD WEIGHTS] extracted_keywords가 비어있음. "
                f"쿼리에서 키워드 추출 시도: query='{query[:50]}...'"
            )
            # state에서 다시 확인
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            if not extracted_keywords:
                # search 그룹에서도 확인
                if "search" in state and isinstance(state.get("search"), dict):
                    search_keywords = state["search"].get("extracted_keywords", [])
                    if search_keywords:
                        extracted_keywords = search_keywords
                        self.logger.debug(
                            f"✅ [KEYWORD WEIGHTS] search 그룹에서 {len(extracted_keywords)}개 키워드 발견"
                        )
            
            if not extracted_keywords:
                # 쿼리에서 간단한 키워드 추출
                import re
                korean_words = re.findall(r'[가-힣]+', query)
                extracted_keywords = [w for w in korean_words if len(w) >= 2]
                
                # 불용어 제거 (KoNLPy 기반 KoreanStopwordProcessor 사용)
                if hasattr(self, 'stopword_processor') and self.stopword_processor:
                    extracted_keywords = self.stopword_processor.filter_stopwords(extracted_keywords)
                else:
                    # 폴백: 기본 불용어 제거
                    stopwords = {'은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로',
                                '에서', '에게', '한테', '께', '무엇', '어떤', '어떻게', '언제', '어디', '누구',
                                '시', '할', '하는', '된', '되는', '이다', '입니다', '있습니다', '합니다'}
                    extracted_keywords = [kw for kw in extracted_keywords if kw not in stopwords]
                
                # 쿼리 타입/법률 분야 기반 기본 키워드 추가
                legal_field = self._get_state_value(state, "legal_field", "")
                query_type_keywords = {
                    "precedent_search": ["판례", "사건", "판결", "대법원"],
                    "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
                    "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
                    "procedure_guide": ["절차", "방법", "대응", "소송"],
                    "term_explanation": ["의미", "정의", "개념", "해석"]
                }
                field_keywords = {
                    "family": ["가족", "이혼", "양육", "상속", "부부"],
                    "civil": ["민사", "계약", "손해배상", "채권", "채무"],
                    "criminal": ["형사", "범죄", "처벌", "형량"],
                    "labor": ["노동", "근로", "해고", "임금", "근로자"],
                    "corporate": ["기업", "회사", "주주", "법인"]
                }
                
                if query_type_str in query_type_keywords:
                    extracted_keywords.extend(query_type_keywords[query_type_str])
                if legal_field in field_keywords:
                    extracted_keywords.extend(field_keywords[legal_field])
                
                extracted_keywords = list(set(extracted_keywords))
                extracted_keywords = [kw for kw in extracted_keywords if kw and len(kw.strip()) >= 2]
                
                if extracted_keywords:
                    self.logger.debug(
                        f"✅ [KEYWORD WEIGHTS] 쿼리에서 {len(extracted_keywords)}개 키워드 추출: "
                        f"{extracted_keywords[:5]}..."
                    )
                    # 추출된 키워드를 state에 저장하여 재사용
                    self._set_state_value(state, "extracted_keywords", extracted_keywords)
                    if "search" not in state:
                        state["search"] = {}
                    if not isinstance(state["search"], dict):
                        state["search"] = {}
                    state["search"]["extracted_keywords"] = extracted_keywords
                    state["extracted_keywords"] = extracted_keywords
                else:
                    self.logger.warning(
                        f"⚠️ [KEYWORD WEIGHTS] 키워드 추출 실패: query='{query[:50]}...'"
                    )
        
        keyword_weights = self._calculate_keyword_weights(
            extracted_keywords=extracted_keywords,
            query=query,
            query_type=query_type_str,
            legal_field=self._get_state_value(state, "legal_field", "")
        )
        
        # 🔥 근본적 개선: 통합 reranking이 있으면 원본 점수를 보존하기 위해
        # _apply_keyword_weights_to_docs를 건너뛰고 merged_docs를 직접 전달
        has_integrated_rerank = (
            self.result_ranker and 
            hasattr(self.result_ranker, 'integrated_rerank_pipeline')
        )
        
        # 🔥 근본적 개선: 통합 reranking 사용 여부에 따라 처리 분기
        if has_integrated_rerank:
            # 통합 reranking은 검색 타입별 정규화와 가중치를 내부에서 처리하므로
            # 원본 점수를 보존하기 위해 merged_docs를 직접 사용
            # 검색 타입별로 결과 분리 (원본 점수 보존)
            db_results = []
            vector_results = []
            keyword_results = []
            unknown_results = []  # 분류 실패 문서 추적용
            
            # 🔥 CRITICAL: 문서 분류 전에 타입 정보 백업 (타입 손실 방지)
            doc_type_backup = {}
            for doc in merged_docs:
                if isinstance(doc, dict):
                    doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                    if doc_id:
                        # 🔥 CRITICAL: 타입 정보를 최대한 확보 (최상위 필드, metadata, source_type 모두 확인)
                        doc_type = doc.get("type")
                        metadata = doc.get("metadata", {})
                        if isinstance(metadata, dict):
                            metadata_type = metadata.get("type") or metadata.get("source_type")
                            # 타입이 없거나 unknown이면 metadata에서 복원 시도
                            if not doc_type or doc_type.lower() == "unknown":
                                if metadata_type and metadata_type.lower() != "unknown":
                                    doc_type = metadata_type
                                    doc["type"] = metadata_type  # 최상위 필드에도 복원
                                    self.logger.debug(
                                        f"🔍 [BACKUP TYPE RESTORE] Doc ID={doc_id}: "
                                        f"백업 생성 전 타입 복원: {doc.get('type')} → {metadata_type}"
                                    )
                        
                        doc_type_backup[doc_id] = {
                            "type": doc_type,
                            "metadata_type": metadata.get("type") if isinstance(metadata, dict) else None,
                            "statute_fields": {
                                "statute_name": doc.get("statute_name") or (metadata.get("statute_name") if isinstance(metadata, dict) else None),
                                "law_name": doc.get("law_name") or (metadata.get("law_name") if isinstance(metadata, dict) else None),
                                "article_no": doc.get("article_no") or (metadata.get("article_no") if isinstance(metadata, dict) else None),
                            },
                            "case_fields": {
                                "case_id": doc.get("case_id") or (metadata.get("case_id") if isinstance(metadata, dict) else None),
                                "court": doc.get("court") or (metadata.get("court") if isinstance(metadata, dict) else None),
                                "doc_id": doc.get("doc_id") or (metadata.get("doc_id") if isinstance(metadata, dict) else None),
                            }
                        }
            
            for doc in merged_docs:
                # 원본 점수 보존을 위해 별도 필드에 저장
                if "original_relevance_score" not in doc:
                    doc["original_relevance_score"] = doc.get("relevance_score", 0.0)
                if "original_similarity" not in doc and "similarity" in doc:
                    doc["original_similarity"] = doc.get("similarity")
                
                # 🔥 CRITICAL: 먼저 백업에서 타입 복원 (문서 분류 전에 타입 보존)
                doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                if doc_id and doc_id in doc_type_backup:
                    backup_info = doc_type_backup[doc_id]
                    backup_type = backup_info.get("type")
                    current_type = doc.get("type", "").lower() if doc.get("type") else ""
                    
                    # 백업된 타입이 있고, 현재 타입이 없거나 unknown이거나 백업과 다른 경우 복원
                    if backup_type and backup_type.lower() != "unknown":
                        if (not doc.get("type") or 
                            current_type == "unknown" or 
                            current_type == "" or
                            (current_type != backup_type.lower() and current_type != "")):
                            doc["type"] = backup_type
                            self.logger.info(
                                f"🔍 [TYPE RESTORE FROM BACKUP] Doc ID={doc_id}: "
                                f"타입 복원: {current_type} → {backup_type}"
                            )
                            
                            # 법령/판례 관련 필드도 복원
                            for key, value in backup_info.get("statute_fields", {}).items():
                                if value and not doc.get(key):
                                    doc[key] = value
                            for key, value in backup_info.get("case_fields", {}).items():
                                if value and not doc.get(key):
                                    doc[key] = value
                
                # 🔥 개선: 메타데이터 및 type 필드 정규화 (중복 코드 제거)
                # 🔥 개선: 메타데이터 및 type 필드 정규화 (중복 코드 제거)
                doc = self._normalize_document_metadata(doc)
                doc = self._normalize_document_type(doc)
                
                # 메타데이터에 type 저장 (일관성 유지)
                if doc.get("type") and not doc.get("metadata", {}).get("type"):
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    if not isinstance(doc["metadata"], dict):
                        doc["metadata"] = {}
                    doc["metadata"]["type"] = doc.get("type")
                
                search_type = doc.get("search_type", "").lower()
                classified = False
                
                # 1. DB 검색 결과 (database, db, text2sql, direct_statute)
                if (search_type in ["database", "db", "text2sql", "direct_statute"] or 
                    doc.get("direct_match", False) or
                    doc.get("search_method") == "text2sql"):
                    db_results.append(doc)
                    classified = True
                # 2. 벡터 검색 결과 (semantic, vector)
                elif search_type in ["semantic", "vector"]:
                    vector_results.append(doc)
                    classified = True
                # 3. 키워드 검색 결과 (keyword)
                elif search_type == "keyword":
                    keyword_results.append(doc)
                    classified = True
                
                # 4. search_type이 없거나 알 수 없는 경우, 강화된 추론 로직 적용
                if not classified:
                    # metadata에서 원본 search_type 복원 시도
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict) and metadata.get("original_search_type"):
                        original_type = metadata["original_search_type"].lower()
                        if original_type in ["semantic", "vector"]:
                            vector_results.append(doc)
                            doc["search_type"] = "semantic"  # 복원
                            classified = True
                        elif original_type == "keyword":
                            keyword_results.append(doc)
                            doc["search_type"] = "keyword"  # 복원
                            classified = True
                    
                    # type으로 판단
                    if not classified:
                        # 🔥 CRITICAL: 문서 분류 전에 백업에서 타입 복원 시도
                        doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                        if doc_id and doc_id in doc_type_backup:
                            backup_info = doc_type_backup[doc_id]
                            backup_type = backup_info.get("type")
                            current_type = doc.get("type", "").lower() if doc.get("type") else ""
                            
                            # 백업된 타입이 있고, 현재 타입이 없거나 unknown인 경우 복원
                            if backup_type and backup_type.lower() != "unknown":
                                if (not doc.get("type") or current_type == "unknown" or current_type == ""):
                                    doc["type"] = backup_type
                                    self.logger.info(
                                        f"🔍 [TYPE RESTORE BEFORE CLASSIFY] Doc ID={doc_id}: "
                                        f"문서 분류 전 타입 복원: {current_type} → {backup_type}"
                                    )
                                    
                                    # 법령/판례 관련 필드도 복원
                                    for key, value in backup_info.get("statute_fields", {}).items():
                                        if value and not doc.get(key):
                                            doc[key] = value
                                    for key, value in backup_info.get("case_fields", {}).items():
                                        if value and not doc.get(key):
                                            doc[key] = value
                        
                        doc_type = doc.get("type", "").lower()
                        
                        # 법령/판례 관련 문서는 semantic으로 분류
                        if doc_type in ["statute_article", "precedent_content", "statute", "precedent"]:
                            vector_results.append(doc)
                            doc["search_type"] = "semantic"  # 설정
                            classified = True
                        # law_name이나 article_no가 있으면 semantic (법령 조문)
                        elif doc.get("law_name") or doc.get("article_no"):
                            vector_results.append(doc)
                            doc["search_type"] = "semantic"  # 설정
                            classified = True
                        # case_number나 court_name이 있으면 semantic (판례)
                        elif doc.get("case_number") or doc.get("court_name"):
                            vector_results.append(doc)
                            doc["search_type"] = "semantic"  # 설정
                            classified = True
                        # direct_match가 있으면 database
                        elif doc.get("direct_match", False):
                            db_results.append(doc)
                            doc["search_type"] = "database"  # 설정
                            classified = True
                    
                    # 여전히 분류되지 않으면 keyword로 분류
                    if not classified:
                        keyword_results.append(doc)
                        doc["search_type"] = "keyword"  # 기본값 설정
                        unknown_results.append(doc)  # 추적용
            
            # 분류 결과 로깅
            if unknown_results:
                self.logger.warning(
                    f"⚠️ [SEARCH TYPE] {len(unknown_results)}개 문서가 자동으로 keyword로 분류됨"
                )
            
            self.logger.info(
                f"🔍 [SEARCH TYPE SPLIT] DB={len(db_results)}, Vector={len(vector_results)}, "
                f"Keyword={len(keyword_results)}, Unknown={len(unknown_results)}"
            )
            
            # 🔥 CRITICAL: 문서 분류 후 메타데이터 복원 (원본 문서에서)
            # 분류된 모든 문서에 대해 메타데이터 복원
            all_classified_docs = db_results + vector_results + keyword_results
            for classified_doc in all_classified_docs:
                if not isinstance(classified_doc, dict):
                    continue
                
                # 🔥 개선: 먼저 백업된 타입 정보로 복원 시도 (분류 전 타입 보존)
                classified_id = classified_doc.get("id") or classified_doc.get("chunk_id") or classified_doc.get("document_id")
                if classified_id and classified_id in doc_type_backup:
                    backup_info = doc_type_backup[classified_id]
                    current_type = classified_doc.get("type", "").lower() if classified_doc.get("type") else ""
                    backup_type = backup_info.get("type", "").lower() if backup_info.get("type") else ""
                    
                    # 백업된 타입이 있고, 현재 타입이 없거나 unknown이거나 백업과 다른 경우 복원
                    if backup_type and backup_type != "unknown":
                        if (not classified_doc.get("type") or 
                            current_type == "unknown" or 
                            (current_type != backup_type and current_type != "")):
                            classified_doc["type"] = backup_info.get("type")
                            if "metadata" not in classified_doc:
                                classified_doc["metadata"] = {}
                            if not isinstance(classified_doc["metadata"], dict):
                                classified_doc["metadata"] = {}
                            classified_doc["metadata"]["type"] = backup_info.get("type")
                            
                            # 법령/판례 관련 필드도 복원
                            for key, value in backup_info.get("statute_fields", {}).items():
                                if value and not classified_doc.get(key):
                                    classified_doc[key] = value
                            for key, value in backup_info.get("case_fields", {}).items():
                                if value and not classified_doc.get(key):
                                    classified_doc[key] = value
                            
                            self.logger.debug(
                                f"🔍 [TYPE RESTORE FROM BACKUP] Doc ID={classified_id}: "
                                f"타입 복원: {current_type} → {backup_type}"
                            )
                
                # 원본 문서에서 메타데이터 복원 시도
                original_doc = None
                
                if classified_id and classified_id in original_docs_by_id:
                    original_doc = original_docs_by_id[classified_id]
                else:
                    # content 기반 매칭 시도
                    content = classified_doc.get("text") or classified_doc.get("content", "")
                    if content:
                        import hashlib
                        content_hash = str(hashlib.md5(content[:200].encode('utf-8')).hexdigest())
                        if content_hash in original_docs_by_content:
                            original_doc = original_docs_by_content[content_hash]
                
                if original_doc:
                    # 🔥 CRITICAL: metadata의 source_type을 type으로 변환 (레거시 호환)
                    original_metadata = original_doc.get("metadata", {})
                    if isinstance(original_metadata, dict):
                        if original_metadata.get("source_type") and not classified_doc.get("type"):
                            classified_doc["type"] = original_metadata.get("source_type")
                            if "metadata" not in classified_doc:
                                classified_doc["metadata"] = {}
                            if not isinstance(classified_doc["metadata"], dict):
                                classified_doc["metadata"] = {}
                            classified_doc["metadata"]["type"] = original_metadata.get("source_type")
                    
                    # type 필드 복원
                    # 🔥 개선: 이미 타입이 설정되어 있어도 원본 타입과 다르면 복원
                    current_type = classified_doc.get("type", "").lower()
                    original_type = original_doc.get("type", "").lower() if original_doc.get("type") else ""
                    
                    # 원본 타입이 있고, 현재 타입이 없거나 unknown이거나 원본과 다른 경우 복원
                    if original_type and original_type != "unknown":
                        if (not classified_doc.get("type") or 
                            current_type == "unknown" or 
                            (current_type != original_type and current_type != "")):
                            # 원본 타입으로 복원
                            classified_doc["type"] = original_doc.get("type")
                            self.logger.debug(
                                f"🔍 [TYPE RESTORE AFTER CLASSIFY] Doc ID={classified_doc.get('id', 'unknown')}: "
                                f"타입 복원: {current_type} → {original_type}"
                            )
                    
                    # 법령/판례 관련 필드 복원
                    for key in ["statute_name", "law_name", "article_no", "article_number",
                               "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id",
                               "chunk_id", "source_id"]:
                        if not classified_doc.get(key) and original_doc.get(key):
                            classified_doc[key] = original_doc.get(key)
                    
                    # metadata에도 복원
                    if "metadata" not in classified_doc:
                        classified_doc["metadata"] = {}
                    if not isinstance(classified_doc["metadata"], dict):
                        classified_doc["metadata"] = {}
                    
                    if isinstance(original_metadata, dict):
                        # source_type을 type으로 변환하여 복원
                        original_type = original_metadata.get("type") or original_metadata.get("source_type")
                        if original_type and not classified_doc["metadata"].get("type"):
                            classified_doc["metadata"]["type"] = original_type
                        
                        for key in ["statute_name", "law_name", "article_no", "article_number",
                                   "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id",
                                   "chunk_id", "source_id"]:
                            if original_metadata.get(key) and not classified_doc["metadata"].get(key):
                                classified_doc["metadata"][key] = original_metadata.get(key)
            
            # 통합 reranking 파이프라인 호출 (원본 점수 보존)
            search_quality = self._get_state_value(state, "search_quality", {})
            overall_quality = search_quality.get("overall_quality", 0.7) if isinstance(search_quality, dict) else 0.7
            
            search_params["overall_quality"] = overall_quality
            search_params["document_count"] = len(merged_docs)
            
            # 재랭킹 스킵 조건 (통합 reranking은 더 적극적으로 사용)
            should_skip_rerank = (
                overall_quality >= 0.85 or  # 품질이 매우 높으면 스킵
                (overall_quality >= 0.80 and len(merged_docs) <= 3)  # 품질이 높고 문서 수가 매우 적으면 스킵
            )
            
            self.logger.info(
                f"🔍 [RERANK CHECK] overall_quality={overall_quality:.2f}, "
                f"merged_docs={len(merged_docs)}, should_skip={should_skip_rerank}, "
                f"has_integrated={has_integrated_rerank}"
            )
            
            if not should_skip_rerank:
                top_k = search_params.get("max_results", 20)
                
                # 🔍 로깅: integrated_rerank_pipeline 호출 전 입력 문서의 메타데이터 확인
                input_docs = db_results + vector_results + keyword_results
                
                # 🔥 개선: integrated_rerank_pipeline 호출 전 메타데이터 보존
                # 🔥 CRITICAL: merged_docs에 대해 메타데이터 복원 (백업 정보 활용)
                # 🔥 CRITICAL: input_docs 대신 merged_docs를 사용하여 백업 정보와 일치시킴
                for doc in merged_docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    # 🔥 CRITICAL: metadata의 source_type을 type으로 변환 (레거시 호환)
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict):
                        if metadata.get("source_type") and not doc.get("type"):
                            doc["type"] = metadata.get("source_type")
                            metadata["type"] = metadata.get("source_type")
                    
                    # type 필드 정규화
                    # 🔥 개선: unknown 타입도 복원하도록 수정
                    current_type = doc.get("type", "").lower()
                    if not doc.get("type") or current_type == "unknown":
                        if not isinstance(metadata, dict):
                            metadata = doc.get("metadata", {})
                        if isinstance(metadata, dict):
                            metadata_type = metadata.get("type") or metadata.get("source_type")
                            if metadata_type and metadata_type != "unknown":
                                doc["type"] = metadata_type
                                if "metadata" not in doc:
                                    doc["metadata"] = {}
                                if not isinstance(doc["metadata"], dict):
                                    doc["metadata"] = {}
                                doc["metadata"]["type"] = metadata_type
                            elif not doc.get("type"):
                                # DocumentType.from_metadata로 추론
                                from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                                doc_type_enum = DocumentType.from_metadata(doc)
                                if doc_type_enum != DocumentType.UNKNOWN:
                                    doc["type"] = doc_type_enum.value
                                    if "metadata" not in doc:
                                        doc["metadata"] = {}
                                    if not isinstance(doc["metadata"], dict):
                                        doc["metadata"] = {}
                                    doc["metadata"]["type"] = doc_type_enum.value
                    
                    # 🔥 CRITICAL: content 기반 추론 (메타데이터 완전 손실 시 폴백)
                    # 🔥 개선: 판례 패턴을 우선적으로 확인 (판례가 법률 조문 패턴과 겹칠 수 있음)
                    # 🔥 CRITICAL: 이미 타입이 설정되어 있으면 content 기반 추론하지 않음 (타입 손실 방지)
                    # 🔥 CRITICAL: 백업된 타입이 있으면 content 기반 추론을 건너뜀
                    current_type_after_restore = doc.get("type", "").lower() if doc.get("type") else ""
                    doc_id_for_backup = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                    has_backup_type = doc_id_for_backup and doc_id_for_backup in doc_type_backup and doc_type_backup[doc_id_for_backup].get("type")
                    
                    # content 기반 추론은 타입이 완전히 없거나 unknown이고, 백업도 없을 때만 실행
                    if (not doc.get("type") or current_type_after_restore == "unknown" or current_type_after_restore == "") and not has_backup_type:
                        content = doc.get("content", "") or doc.get("text", "")
                        if content:
                            import re
                            
                            # 판례 패턴 (우선 확인 - 판례가 법률 조문 패턴도 포함할 수 있음)
                            precedent_patterns = [
                                r'【원고',  # 【원고, 피상고인】
                                r'【피고',  # 【피고, 상고인】
                                r'【청구인',  # 【청구인, 재항고인】
                                r'【사건본인',  # 【사건본인】
                                r'대법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',  # 대법원 2023. 9. 27.
                                r'고등법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',
                                r'지방법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',
                                r'선고.*판결',  # 선고 2021다255655 판결
                                r'선고.*결정',  # 선고 2017브10 결정
                                r'원심판결',  # 【원심판결】
                                r'원심결정',  # 【원심결정】
                                r'소송대리인',  # 소송대리인 변호사
                                r'담당변호사',  # 담당변호사 이종희
                                r'사건번호',  # 사건번호
                                r'사건.*\d+',  # 사건 2015르3081
                                r'판결 참조',  # 판결 참조
                                r'판례',  # 판례
                            ]
                            
                            # 법률 조문 패턴 (판례 패턴이 없을 때만 확인)
                            statute_patterns = [
                                r'제\d+조\s*제\d+항',  # 제750조 제1항 (구체적인 조문 형식)
                                r'제\d+조\s*제\d+호',  # 제750조 제1호
                                r'법률.*제\d+조.*제\d+항',  # 법률 제750조 제1항
                                r'민법.*제\d+조.*제\d+항',  # 민법 제750조 제1항
                                r'형법.*제\d+조.*제\d+항',  # 형법 제750조 제1항
                                r'상법.*제\d+조.*제\d+항',  # 상법 제750조 제1항
                            ]
                            
                            # 판례 패턴 우선 확인
                            is_precedent = any(re.search(p, content, re.IGNORECASE) for p in precedent_patterns)
                            
                            # 판례 패턴이 없을 때만 법률 조문 패턴 확인
                            is_statute = False
                            if not is_precedent:
                                is_statute = any(re.search(p, content, re.IGNORECASE) for p in statute_patterns)
                            
                            if is_precedent:
                                doc["type"] = "precedent_content"
                                if "metadata" not in doc:
                                    doc["metadata"] = {}
                                if not isinstance(doc["metadata"], dict):
                                    doc["metadata"] = {}
                                doc["metadata"]["type"] = "precedent_content"
                                self.logger.debug(f"🔍 [CONTENT INFERENCE] 판례로 추론: {doc.get('id', 'unknown')}")
                            elif is_statute:
                                doc["type"] = "statute_article"
                                if "metadata" not in doc:
                                    doc["metadata"] = {}
                                if not isinstance(doc["metadata"], dict):
                                    doc["metadata"] = {}
                                doc["metadata"]["type"] = "statute_article"
                                self.logger.debug(f"🔍 [CONTENT INFERENCE] 법률 조문으로 추론: {doc.get('id', 'unknown')}")
                    
                    # metadata에서 최상위 필드로 복사
                    if not isinstance(metadata, dict):
                        metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict):
                        if metadata.get("type") and not doc.get("type"):
                            doc["type"] = metadata.get("type")
                        
                        # 법령/판례 관련 필드 복원
                        for key in ["statute_name", "law_name", "article_no", "article_number",
                                   "case_id", "court", "ccourt", "doc_id", "casenames", "precedent_id",
                                   "chunk_id", "source_id"]:
                            if metadata.get(key) and not doc.get(key):
                                doc[key] = metadata.get(key)
                
                input_type_count = sum(1 for doc in input_docs if doc.get("type"))
                input_metadata_type_count = sum(
                    1 for doc in input_docs 
                    if isinstance(doc.get("metadata"), dict) and 
                    doc.get("metadata").get("type")
                )
                self.logger.info(
                    f"🔄 [INTEGRATED RERANK] 통합 reranking 시작 (원본 점수 보존): "
                    f"DB={len(db_results)}, Vector={len(vector_results)}, "
                    f"Keyword={len(keyword_results)}, top_k={top_k}, "
                    f"입력 문서 타입 정보: 최상위={input_type_count}개, metadata={input_metadata_type_count}개"
                )
                
                # 샘플 입력 문서 메타데이터 로깅
                if input_docs:
                    sample_input = input_docs[0]
                    self.logger.info(
                        f"🔍 [INTEGRATED RERANK INPUT SAMPLE] "
                        f"type={sample_input.get('type')}, "
                        f"metadata_type={sample_input.get('metadata', {}).get('type') if isinstance(sample_input.get('metadata'), dict) else 'N/A'}, "
                        f"has_statute_fields={bool(sample_input.get('statute_name') or sample_input.get('law_name') or sample_input.get('article_no'))}, "
                        f"has_case_fields={bool(sample_input.get('case_id') or sample_input.get('court') or sample_input.get('doc_id'))}"
                    )
                
                try:
                    weighted_docs = self.result_ranker.integrated_rerank_pipeline(
                        db_results=db_results,
                        vector_results=vector_results,
                        keyword_results=keyword_results,
                        query=query,
                        query_type=query_type_str,
                        extracted_keywords=extracted_keywords,
                        top_k=top_k,
                        search_quality=overall_quality
                    )
                    
                    # 🔥 개선: integrated_rerank_pipeline 내부에서 이미 메타데이터 복원이 완료되었으므로
                    # 추가 복원 로직은 제거 (중복 방지)
                    # 다만, 복원된 메타데이터가 제대로 반환되었는지 확인
                    restored_metadata_count = sum(
                        1 for doc in weighted_docs 
                        if doc.get("type") or 
                           (isinstance(doc.get("metadata"), dict) and 
                            doc.get("metadata").get("type"))
                    )
                    self.logger.info(
                        f"✅ [INTEGRATED RERANK] 통합 reranking 완료: {len(weighted_docs)}개 문서 "
                        f"(메타데이터 복원 확인: {restored_metadata_count}/{len(weighted_docs)}개 문서에 타입 정보 있음)"
                    )
                    
                    # 통합 reranking 완료 후 weighted_docs 반환
                    return weighted_docs
                except Exception as e:
                    self.logger.warning(f"Integrated rerank failed: {e}, falling back to keyword weights", exc_info=True)
                    # 폴백: 키워드 가중치 적용
                    weighted_docs = self._apply_keyword_weights_to_docs(
                        merged_docs=merged_docs,
                        keyword_weights=keyword_weights,
                        query=query,
                        query_type_str=query_type_str,
                        search_params=search_params
                    )
            else:
                # 스킵된 경우 키워드 가중치만 적용
                weighted_docs = self._apply_keyword_weights_to_docs(
                    merged_docs=merged_docs,
                    keyword_weights=keyword_weights,
                    query=query,
                    query_type_str=query_type_str,
                    search_params=search_params
                )
                self.logger.debug(
                    f"⏭️ [RERANK SKIP] 통합 reranking 스킵 (quality: {overall_quality:.2f}, docs: {len(merged_docs)})"
                )
        else:
            # 통합 reranking이 없는 경우 기존 로직 사용
            weighted_docs = self._apply_keyword_weights_to_docs(
                merged_docs=merged_docs,
                keyword_weights=keyword_weights,
                query=query,
                query_type_str=query_type_str,
                search_params=search_params
            )
            
            # 재랭킹 조건 확인
            should_skip_rerank = (
                len(weighted_docs) <= 5 or  # 문서 수가 매우 적으면 스킵
                overall_quality >= 0.80 or  # 품질이 높으면 스킵
                (overall_quality >= 0.70 and len(weighted_docs) <= 10)  # 품질이 중간 이상이고 문서 수가 적으면 스킵
            )
            
            self.logger.info(
                f"🔍 [RERANK CHECK] overall_quality={overall_quality:.2f}, "
                f"weighted_docs={len(weighted_docs)}, should_skip={should_skip_rerank}, "
                f"has_integrated={has_integrated_rerank}, "
                f"has_multi_stage={hasattr(self.result_ranker, 'multi_stage_rerank') if self.result_ranker else False}"
            )
            
            if not should_skip_rerank and self.result_ranker:
                try:
                    search_quality = self._get_state_value(state, "search_quality", {})
                    overall_quality = search_quality.get("overall_quality", 0.7) if isinstance(search_quality, dict) else 0.7
                    
                    search_params["overall_quality"] = overall_quality
                    search_params["document_count"] = len(weighted_docs)
                    
                    # 하위 호환성: multi_stage_rerank 사용
                    if hasattr(self.result_ranker, 'multi_stage_rerank'):
                        self.logger.debug(f"[MULTI-STAGE RERANK] Starting reranking: {len(weighted_docs)} documents, quality={overall_quality:.2f}")
                        self.logger.debug(
                            f"🔄 [MULTI-STAGE RERANK] Starting reranking: {len(weighted_docs)} documents, "
                            f"quality={overall_quality:.2f}, query='{query[:50]}...'"
                        )
                        
                        weighted_docs = self.result_ranker.multi_stage_rerank(
                            documents=weighted_docs,
                            query=query,
                            query_type=query_type_str,
                            extracted_keywords=extracted_keywords,
                            search_quality=overall_quality
                        )
                        
                        self.logger.debug(f"[MULTI-STAGE RERANK] Applied multi-stage reranking: {len(weighted_docs)} documents")
                        self.logger.debug(f"🔄 [MULTI-STAGE RERANK] Applied multi-stage reranking: {len(weighted_docs)} documents")
                    else:
                        self.logger.warning("⚠️ [RERANK SKIP] result_ranker에 reranking 메서드가 없음, using citation boost")
                        weighted_docs = self._apply_citation_boost(weighted_docs)
                except Exception as e:
                    self.logger.warning(f"Rerank failed: {e}, using citation boost", exc_info=True)
                    weighted_docs = self._apply_citation_boost(weighted_docs)
            else:
                if not self.result_ranker:
                    self.logger.warning("⚠️ [RERANK SKIP] result_ranker is None, using citation boost")
                elif not hasattr(self.result_ranker, 'multi_stage_rerank'):
                    self.logger.warning("⚠️ [RERANK SKIP] result_ranker has no multi_stage_rerank method, using citation boost")
                weighted_docs = self._apply_citation_boost(weighted_docs)
        
        # 개선 9.1: Reranking 후 최소 문서 수 보장
        MIN_DOCS_AFTER_RERANK = 5
        if len(weighted_docs) < MIN_DOCS_AFTER_RERANK and len(merged_docs) > len(weighted_docs):
            excluded_docs = [doc for doc in merged_docs if doc not in weighted_docs]
            excluded_docs.sort(
                key=lambda x: x.get("relevance_score", x.get("final_weighted_score", 0.0)),
                reverse=True
            )
            needed = MIN_DOCS_AFTER_RERANK - len(weighted_docs)
            weighted_docs.extend(excluded_docs[:needed])
            self.logger.debug(
                f"✅ [RERANK] 최소 문서 수 보장: {len(weighted_docs)}개 "
                f"(추가: {needed}개)"
            )
        elif should_skip_rerank:
            # 재랭킹 스킵 이유 설명 (개선된 조건 반영)
            if len(weighted_docs) <= 5:
                reason = 'docs <= 5'
            elif overall_quality >= 0.80:
                reason = 'quality >= 0.80'
            elif overall_quality >= 0.70 and len(weighted_docs) <= 10:
                reason = 'quality >= 0.70 and docs <= 10'
            else:
                reason = 'other'
            self.logger.debug(
                f"⏭️ [RERANK SKIP] Skipping multi-stage rerank "
                f"(quality: {overall_quality:.2f}, docs: {len(weighted_docs)}, "
                f"reason: {reason})"
            )
        else:
            if not self.result_ranker:
                self.logger.warning("⚠️ [RERANK SKIP] result_ranker is None, using citation boost")
            elif not hasattr(self.result_ranker, 'multi_stage_rerank'):
                self.logger.warning("⚠️ [RERANK SKIP] result_ranker has no multi_stage_rerank method, using citation boost")
            weighted_docs = self._apply_citation_boost(weighted_docs)
        
        if debug_mode and weighted_docs:
            scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in weighted_docs]
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            self.logger.debug(f"📊 [SEARCH RESULTS] Score distribution after weighting - Total: {len(weighted_docs)}, Min: {min_score:.3f}, Max: {max_score:.3f}, Avg: {avg_score:.3f}")
        
        return weighted_docs
    
    def _filter_and_validate_documents(
        self,
        state: LegalWorkflowState,
        weighted_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        merged_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """필터링 및 검증"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        max_docs_before_filter = self.config.max_retrieved_docs or 20
        
        if weighted_docs:
            type_distribution = self._calculate_type_distribution(weighted_docs)
            self.logger.debug(f"🔀 [DIVERSITY] weighted_docs type distribution before diversity: {type_distribution}")
        
        has_precedent_before, has_decision_before = self._has_precedent_or_decision(weighted_docs)
        self.logger.debug(f"🔀 [DIVERSITY] Before filtering - has_precedent={has_precedent_before}, has_decision={has_decision_before}")
        
        if self.search_handler and len(weighted_docs) > 0:
            diverse_weighted_docs = self.search_handler._ensure_diverse_source_types(
                weighted_docs,
                min(max_docs_before_filter * 3, len(weighted_docs))
            )
            self.logger.debug(f"🔀 [DIVERSITY] Before filtering: {len(weighted_docs)} → {len(diverse_weighted_docs)} docs (ensuring diversity)")
            weighted_docs = diverse_weighted_docs
        
        if debug_mode and weighted_docs:
            sample_doc = weighted_docs[0]
            sample_structure = f"Sample doc keys: {list(sample_doc.keys())}, has content: {'content' in sample_doc}, has text: {'text' in sample_doc}, content type: {type(sample_doc.get('content', 'N/A')).__name__}"
            self.logger.debug(f"🔍 [SEARCH RESULTS] {sample_structure}")
        
        filtered_docs = []
        skipped_content = 0
        skipped_score = 0
        skipped_relevance = 0
        skipped_content_details = []
        
        core_query_keywords = set()
        if query:
            query_words = query.split()
            for word in query_words:
                if len(word) >= 2 and word not in ["시", "의", "와", "과", "는", "은", "이", "가", "을", "를", "에", "에서", "로", "으로"]:
                    core_query_keywords.add(word.lower())
        
        if extracted_keywords:
            for kw in extracted_keywords[:10]:
                if isinstance(kw, str) and len(kw) >= 2:
                    core_query_keywords.add(kw.lower())
        
        # 벡터 점수 계산 (NumPy 벡터 연산으로 최적화)
        try:
            import numpy as np
            
            # 점수 배열 생성 (벡터화)
            scores = np.array([
                doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                for doc in weighted_docs
            ])
            
            # text2sql 문서 마스크 생성 (벡터화)
            is_text2sql_mask = np.array([
                bool(
                    doc.get("search_type") == "text2sql" or
                    doc.get("search_type") == "direct_statute" or
                    doc.get("direct_match", False) or
                    (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
                )
                for doc in weighted_docs
            ], dtype=bool)
            
            # scores를 numpy array로 변환 (타입 안전성 보장)
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores, dtype=float)
            
            # 벡터 점수만 추출 (text2sql 제외)
            # 마스크와 scores의 길이가 일치하는지 확인
            if len(scores) != len(is_text2sql_mask):
                self.logger.warning(
                    f"⚠️ [FILTER] scores length ({len(scores)}) != mask length ({len(is_text2sql_mask)}). "
                    f"Using fallback filtering method."
                )
                vector_scores = [
                    score for i, score in enumerate(scores)
                    if i < len(is_text2sql_mask) and not is_text2sql_mask[i]
                ]
                vector_scores = np.array(vector_scores, dtype=float) if vector_scores else np.array([], dtype=float)
            else:
                vector_scores = scores[~is_text2sql_mask]
            
            if len(vector_scores) > 0:
                avg_score = float(np.mean(vector_scores))
                default_min_score_threshold = max(0.60, min(0.75, avg_score * 0.8))
                if avg_score < 0.70:
                    default_min_score_threshold = max(0.50, avg_score * 0.7)
            else:
                default_min_score_threshold = 0.75
        except ImportError:
            # NumPy가 없으면 기존 방식 사용
            vector_scores = [
                doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                for doc in weighted_docs
                if not (
                    doc.get("search_type") == "text2sql" or
                    doc.get("search_type") == "direct_statute" or
                    doc.get("direct_match", False) or
                    (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
                )
            ]
            if vector_scores:
                avg_score = sum(vector_scores) / len(vector_scores)
                default_min_score_threshold = max(0.60, min(0.75, avg_score * 0.8))
                if avg_score < 0.70:
                    default_min_score_threshold = max(0.50, avg_score * 0.7)
            else:
                default_min_score_threshold = 0.75
        
        # 병렬 처리 (문서가 5개 이상일 때만)
        if len(weighted_docs) >= 5:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # results_lock is defined but not used (may be needed for future thread safety)
            skipped_content_lock = threading.Lock()
            skipped_score_lock = threading.Lock()
            skipped_relevance_lock = threading.Lock()
            
            def process_single_doc(doc):
                nonlocal skipped_content, skipped_score, skipped_relevance, skipped_content_details
                
                if "type" not in doc or not doc.get("type"):
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    if metadata.get("type"):
                        doc["type"] = metadata.get("type")
                
                content = self._extract_doc_content(doc)
                doc_type = self._extract_doc_type(doc)
                if doc.get("type") != doc_type:
                    doc["type"] = doc_type
                
                is_precedent_or_decision = any(keyword in doc_type for keyword in ["precedent", "case", "decision", "판례", "결정"])
                is_statute = any(keyword in doc_type for keyword in ["statute", "article", "법령", "조문"]) or doc_type == "statute_article"
                
                min_content_length = 200  # TASK 3: 최소 내용 길이 기준 상향 (10 → 200)
                if not content or len(content.strip()) < min_content_length:
                    with skipped_content_lock:
                        skipped_content += 1
                        if skipped_content <= 3:
                            skipped_content_details.append({
                                "keys": list(doc.keys()),
                                "content_type": type(doc.get("content", None)).__name__,
                                "text_type": type(doc.get("text", None)).__name__,
                                "content_len": len(str(doc.get("content", ""))),
                                "text_len": len(str(doc.get("text", "")))
                            })
                    return None
                
                if not self._validate_document_metadata(doc):
                    with skipped_relevance_lock:
                        skipped_relevance += 1
                    return None
                
                if not self._validate_document_content_quality(doc, content):
                    with skipped_relevance_lock:
                        skipped_relevance += 1
                    return None
                
                if not self._validate_document_source_reliability(doc):
                    with skipped_relevance_lock:
                        skipped_relevance += 1
                    return None
                
                if core_query_keywords:
                    content_lower = content.lower()
                    has_relevant_keyword = any(kw in content_lower for kw in core_query_keywords if len(kw) > 2)
                    score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                    if not has_relevant_keyword and not (is_precedent_or_decision or is_statute) and score < 0.3:
                        with skipped_relevance_lock:
                            skipped_relevance += 1
                        return None
                
                score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                search_type = doc.get("search_type", "")
                is_text2sql = (
                    search_type == "text2sql" or
                    search_type == "direct_statute" or
                    doc.get("direct_match", False) or
                    (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
                )
                
                # TASK 6: 점수 필터링 (벡터화된 계산 결과 활용) + 법령 조문 예외 처리
                if is_text2sql:
                    min_score_threshold = 0.0
                else:
                    # TASK 6: 관련도 임계값 강화 (최소 0.4)
                    min_score_threshold = max(0.4, default_min_score_threshold)
                
                if score < min_score_threshold:
                    # TASK 6: 낮은 관련도지만 예외 조건 충족 시 포함
                    if self._should_include_statute_despite_low_relevance(doc, query):
                        self.logger.debug(f"✅ [TASK 6] 낮은 관련도지만 예외 조건 충족하여 포함: score={score:.3f}, article_no={doc.get('article_no')}")
                    else:
                        with skipped_score_lock:
                            skipped_score += 1
                        return None
                
                return doc
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_single_doc, doc): doc for doc in weighted_docs}
                for future in as_completed(futures, timeout=30):
                    try:
                        result = future.result(timeout=2)
                        if result:
                            filtered_docs.append(result)
                    except Exception as e:
                        self.logger.warning(f"Document validation failed: {e}")
        else:
            # 문서가 적으면 순차 처리
            for doc in weighted_docs:
                if "type" not in doc or not doc.get("type"):
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    if metadata.get("type"):
                        doc["type"] = metadata.get("type")
                
                content = self._extract_doc_content(doc)
                doc_type = self._extract_doc_type(doc)
                if doc.get("type") != doc_type:
                    doc["type"] = doc_type
                
                is_precedent_or_decision = any(keyword in doc_type for keyword in ["precedent", "case", "decision", "판례", "결정"])
                is_statute = any(keyword in doc_type for keyword in ["statute", "article", "법령", "조문"]) or doc_type == "statute_article"
                
                min_content_length = 200  # TASK 3: 최소 내용 길이 기준 상향 (10 → 200)
                if not content or len(content.strip()) < min_content_length:
                    skipped_content += 1
                    if skipped_content <= 3:
                        skipped_content_details.append({
                            "keys": list(doc.keys()),
                            "content_type": type(doc.get("content", None)).__name__,
                            "text_type": type(doc.get("text", None)).__name__,
                            "content_len": len(str(doc.get("content", ""))),
                            "text_len": len(str(doc.get("text", "")))
                        })
                    continue
                
                if not self._validate_document_metadata(doc):
                    skipped_relevance += 1
                    self.logger.debug(f"🔍 [METADATA VALIDATION] 메타데이터 검증 실패: {doc.get('id', 'unknown')[:50]}")
                    continue
                
                if not self._validate_document_content_quality(doc, content):
                    skipped_relevance += 1
                    self.logger.debug(f"🔍 [CONTENT QUALITY] 내용 품질 검증 실패: {doc.get('id', 'unknown')[:50]}")
                    continue
                
                if not self._validate_document_source_reliability(doc):
                    skipped_relevance += 1
                    self.logger.debug(f"🔍 [SOURCE RELIABILITY] 출처 신뢰도 검증 실패: {doc.get('id', 'unknown')[:50]}")
                    continue
                
                if core_query_keywords:
                    content_lower = content.lower()
                    has_relevant_keyword = any(kw in content_lower for kw in core_query_keywords if len(kw) > 2)
                    score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                    if not has_relevant_keyword and not (is_precedent_or_decision or is_statute) and score < 0.3:
                        skipped_relevance += 1
                        self.logger.debug(f"🔍 [SEARCH FILTERING] Filtered out irrelevant document: {doc.get('id', 'unknown')[:50]} (no relevant keywords, score={score:.3f})")
                        continue
                
                score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                search_type = doc.get("search_type", "")
                is_text2sql = (
                    search_type == "text2sql" or
                    search_type == "direct_statute" or
                    doc.get("direct_match", False) or
                    (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
                )
                
                if is_text2sql:
                    min_score_threshold = 0.0
                else:
                    # TASK 2: 관련도 임계값 강화 (최소 0.4)
                    min_score_threshold = max(0.4, default_min_score_threshold)
                
                if score < min_score_threshold:
                    skipped_score += 1
                    self.logger.debug(f"🔍 [SCORE FILTER] 점수 부족으로 제외: score={score:.3f} < {min_score_threshold}, source={doc.get('source', 'Unknown')[:50]}")
                    continue
                
                # TASK 2: 의미적 관련성 검증 추가
                if not self._check_semantic_relevance(doc, query, extracted_keywords):
                    skipped_relevance += 1
                    self.logger.debug(f"🔍 [SEMANTIC FILTER] 의미적 관련성 부족으로 제외: {doc.get('id', 'unknown')[:50]}")
                    continue
                
                filtered_docs.append(doc)
        
        if debug_mode:
            self.logger.debug(f"📊 [SEARCH RESULTS] Filtering statistics - Merged: {len(merged_docs)}, Weighted: {len(weighted_docs)}, Filtered: {len(filtered_docs)}, Skipped (content): {skipped_content}, Skipped (score): {skipped_score}, Skipped (relevance): {skipped_relevance}")
            if skipped_content > 0 and skipped_content_details:
                self.logger.warning(f"⚠️ [SEARCH RESULTS] Content 필터링 제외 상세 (상위 {len(skipped_content_details)}개): {skipped_content_details}")
        
        return filtered_docs
    
    def _check_semantic_relevance(
        self, 
        doc: Dict[str, Any], 
        query: str, 
        keywords: List[str]
    ) -> bool:
        """문서와 질문 간 의미적 관련성 검증 (TASK 2)"""
        content = doc.get("content", "") or doc.get("text", "")
        if not content:
            return False
        
        # 키워드 매칭 검증
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw in content)
            if keyword_matches < len(keywords) * 0.3:  # 최소 30% 키워드 매칭
                return False
        
        # 법령 조문 번호 매칭 검증 (강화)
        if "제" in query and "조" in query:
            import re
            article_match = re.search(r'제\s*(\d+)\s*조', query)
            if article_match:
                question_article = article_match.group(1).lstrip('0')
                if not question_article:
                    question_article = "0"
                
                doc_article = str(doc.get("article_no", "")).strip()
                # 조문 번호가 일치하지 않으면 관련성 낮음
                if doc_article:
                    # 선행 0 제거하여 비교
                    doc_article_normalized = doc_article.lstrip('0')
                    if not doc_article_normalized:
                        doc_article_normalized = "0"
                    
                    # 정확한 매칭 확인 (10배 차이 체크)
                    try:
                        question_num = int(question_article)
                        doc_num = int(doc_article_normalized)
                        
                        # 정확히 일치하거나, 직접 검색된 조문인 경우만 통과
                        if question_num == doc_num:
                            return True
                        elif doc.get("direct_match", False):
                            # 직접 검색된 조문이지만 번호가 다르면 추가 검증
                            # 10배 차이면 완전히 다른 조문으로 간주
                            if doc_num > 0 and (question_num * 10 == doc_num or doc_num * 10 == question_num):
                                return False
                            # 작은 차이면 통과 (예: 750 vs 7500)
                            return True
                        else:
                            # 직접 검색되지 않았고 번호가 다르면 제외
                            return False
                    except (ValueError, TypeError):
                        # 숫자 변환 실패 시 문자열 비교
                        if question_article != doc_article_normalized:
                            if not doc.get("direct_match", False):
                                return False
        
        return True
    
    def _should_include_statute_despite_low_relevance(
        self, 
        doc: Dict[str, Any], 
        query: str
    ) -> bool:
        """낮은 관련도에도 법령 조문을 포함할지 결정 (TASK 6)"""
        # 직접 검색된 조문만 예외 적용
        if not doc.get("direct_match", False):
            return False
        
        # 질문에서 조문 번호 추출
        import re
        article_match = re.search(r'제\s*(\d+)\s*조', query)
        if article_match:
            question_article = article_match.group(1)
            doc_article = str(doc.get("article_no", "")).strip()
            # 조문 번호가 일치하는 경우만 예외
            if doc_article:
                doc_article_normalized = doc_article.lstrip('0')
                if question_article == doc_article_normalized:
                    return True
        
        return False
    
    def _validate_document_metadata(self, doc: Dict[str, Any]) -> bool:
        """우선순위 2: 메타데이터 검증 강화"""
        # 필수 필드 검증
        has_content = bool(doc.get("content") or doc.get("text"))
        has_source = bool(doc.get("source"))
        has_type = bool(doc.get("type"))
        
        if not has_content:
            return False
        if not has_source:
            return False
        if not has_type:
            return False
        
        # 메타데이터 완전성 검증
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            # metadata가 있으면 최소한의 구조는 있어야 함
            pass
        
        return True
    
    def _validate_document_content_quality(self, doc: Dict[str, Any], content: str) -> bool:
        """우선순위 3: 문서 내용 품질 검증"""
        if not content or len(content.strip()) < 10:
            return False
        
        content_stripped = content.strip()
        
        # 특수 문자만 있는 문서 제외
        import re
        # 의미 있는 문자(한글, 영문, 숫자) 비율 계산
        meaningful_chars = re.findall(r'[가-힣a-zA-Z0-9]', content_stripped)
        total_chars = len(content_stripped)
        if total_chars == 0:
            return False
        
        meaningful_ratio = len(meaningful_chars) / total_chars
        if meaningful_ratio < 0.5:
            # 의미 있는 문자가 50% 미만이면 제외
            return False
        
        # 불완전한 문장 제외 (문장 끝이 없는 경우가 너무 많으면 제외)
        sentence_endings = content_stripped.count('.') + content_stripped.count('。') + content_stripped.count('!') + content_stripped.count('?')
        if len(content_stripped) > 100 and sentence_endings == 0:
            # 100자 이상인데 문장 끝이 없으면 제외
            return False
        
        return True
    
    def _validate_document_source_reliability(self, doc: Dict[str, Any]) -> bool:
        """우선순위 3: 출처 신뢰도 검증"""
        source = doc.get("source", "")
        if not source or len(source.strip()) < 2:
            return False
        
        source_stripped = source.strip()
        
        # 출처 형식 검증
        # 법령명 형식: "민법", "형법" 등
        # 판례 형식: "대법원", "법원" 등이 포함되거나 사건번호 포함
        # 해설 형식: 기관명 포함
        
        # 기본적인 출처 형식 검증
        has_valid_format = (
            any(keyword in source_stripped for keyword in ["법", "법원", "위원회", "부", "청", "원"]) or
            bool(re.match(r'[가-힣]+법', source_stripped)) or
            bool(re.match(r'.*법원.*', source_stripped)) or
            len(source_stripped) >= 3
        )
        
        if not has_valid_format:
            return False
        
        # 메타데이터에서 출처 정보 확인
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            # statute_name, case_name 등이 있으면 더 신뢰할 수 있음
            pass
        
        return True
    
    def _separate_text2sql_and_vector_results(
        self,
        filtered_docs: List[Dict[str, Any]],
        weighted_docs: List[Dict[str, Any]],
        merged_docs: List[Dict[str, Any]]
    ) -> tuple:
        """textToSQL 결과와 벡터 임베딩 결과 분리"""
        text2sql_docs = []
        vector_docs = []
        seen_ids = set()
        
        # 모든 문서 소스에서 textToSQL 결과 추출
        all_docs = filtered_docs + weighted_docs + merged_docs
        
        for doc in all_docs:
            if not isinstance(doc, dict):
                continue
            
            doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            # textToSQL 결과 판별
            search_type = doc.get("search_type", "")
            direct_match = doc.get("direct_match", False)
            is_text2sql = (
                search_type == "text2sql" or
                search_type == "direct_statute" or
                direct_match is True or
                (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
            )
            
            if is_text2sql:
                text2sql_docs.append(doc)
            else:
                # 벡터 임베딩 결과 (semantic search)
                search_type_val = doc.get("search_type", "")
                if search_type_val == "semantic" or search_type_val == "hybrid" or not search_type_val:
                    vector_docs.append(doc)
        
        self.logger.debug(
            f"🔀 [TEXT2SQL SEPARATION] textToSQL: {len(text2sql_docs)}개, "
            f"벡터 임베딩: {len(vector_docs)}개"
        )
        
        return text2sql_docs, vector_docs
    
    def _rerank_vector_results_only(
        self,
        state: LegalWorkflowState,
        vector_docs: List[Dict[str, Any]],
        query: str,
        query_type_str: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """벡터 임베딩 결과만 재랭킹 (관련성 점수 0.75 이상만 포함)"""
        if not vector_docs:
            return []
        
        # 질의와 검색된 문서의 relevance_score 로깅 (모든 문서)
        self.logger.debug(f"📊 [RELEVANCE SCORES] 질의: '{query}'")
        self.logger.debug(f"📊 [RELEVANCE SCORES] 검색된 벡터 문서 수: {len(vector_docs)}개")
        
        # 모든 문서의 점수 수집 및 로깅
        doc_scores = []
        for doc in vector_docs:
            score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            similarity = doc.get("similarity", 0.0)
            keyword_score = doc.get("keyword_match_score", 0.0)
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id") or "unknown"
            # 🔥 개선: doc_id가 int일 수 있으므로 문자열로 변환
            doc_id = str(doc_id) if doc_id != "unknown" else "unknown"
            doc_type = doc.get("type", "unknown")
            source = doc.get("source", "")[:100] or "unknown"
            content_preview = (doc.get("content", "")[:100] or "").replace("\n", " ")
            doc_scores.append((score, similarity, keyword_score, doc_id, doc_type, source, content_preview, doc))
        
        # 점수 분포 통계
        if doc_scores:
            scores_only = [s[0] for s in doc_scores]
            avg_score = sum(scores_only) / len(scores_only)
            max_score = max(scores_only)
            min_score = min(scores_only)
            median_score = sorted(scores_only)[len(scores_only) // 2]
            self.logger.debug(
                f"📊 [SCORE STATS] 평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}, 중앙값={median_score:.3f}"
            )
            
            # 모든 문서의 점수 상세 로깅 (정렬된 순서)
            doc_scores_sorted = sorted(doc_scores, key=lambda x: x[0], reverse=True)
            self.logger.debug(f"📊 [ALL DOCS SCORES] 모든 {len(doc_scores_sorted)}개 문서의 relevance_score:")
            for i, (score, similarity, keyword_score, doc_id, doc_type, source, content_preview, doc) in enumerate(doc_scores_sorted, 1):
                self.logger.debug(
                    f"   {i}. final_score={score:.3f}, similarity={similarity:.3f}, keyword={keyword_score:.3f}, "
                    f"type={doc_type}, id={doc_id[:50]}, source={source}, "
                    f"content_preview={content_preview}"
                )
        
        # 우선순위 1: 벡터 결과 관련성 점수 동적 임계값 계산
        # 기본 임계값 0.75, 하지만 점수 분포에 따라 조정
        base_threshold = 0.75
        if doc_scores:
            scores_only = [s[0] for s in doc_scores]
            avg_score = sum(scores_only) / len(scores_only)
            max_score = max(scores_only)
            min_score = min(scores_only)
            
            # 점수 분포가 낮으면 임계값을 낮춤
            if avg_score < 0.5:
                # 평균 점수가 매우 낮으면 (Cross-Encoder reranking 후 점수 스케일 문제)
                # 평균의 1.2배 또는 최소 0.15로 설정
                min_relevance_threshold = max(0.15, avg_score * 1.2)
            elif avg_score < 0.6:
                # 평균 점수가 낮으면 평균의 1.1배 또는 최소 0.50로 설정
                min_relevance_threshold = max(0.50, avg_score * 1.1)
            elif avg_score < 0.7:
                # 평균 점수가 중간이면 평균의 1.05배 또는 최소 0.60로 설정
                min_relevance_threshold = max(0.60, avg_score * 1.05)
            else:
                # 평균 점수가 높으면 기본 임계값 사용
                min_relevance_threshold = base_threshold
            
            # 최대 점수가 임계값보다 낮으면 임계값을 최대 점수의 0.9배로 조정
            if max_score < min_relevance_threshold:
                min_relevance_threshold = max(0.10, max_score * 0.9)
            
            self.logger.debug(
                f"📊 [FILTER THRESHOLD] 동적 임계값 계산: "
                f"평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}, "
                f"임계값={min_relevance_threshold:.3f} (기본: {base_threshold})"
            )
        else:
            min_relevance_threshold = base_threshold
            self.logger.debug(f"📊 [FILTER THRESHOLD] 임계값: {min_relevance_threshold} (기본값, 문서 없음)")
        filtered_vector_docs = []
        filtered_out_docs = []
        for doc in vector_docs:
            score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            similarity = doc.get("similarity", 0.0)
            keyword_score = doc.get("keyword_match_score", 0.0)
            weighted_keyword = doc.get("weighted_keyword_score", 0.0)
            
            # 필터링 조건: final_score가 임계값 이상이거나, similarity가 높은 경우 예외 허용
            # similarity >= 0.75인 경우는 원본 벡터 검색 점수가 높은 것이므로 통과
            should_include = (
                score >= min_relevance_threshold or
                similarity >= 0.75  # 원본 벡터 검색 점수가 높으면 통과
            )
            
            if should_include:
                filtered_vector_docs.append(doc)
            else:
                filtered_out_docs.append((score, similarity, keyword_score, weighted_keyword, doc))
                # 상세 필터링 이유 로깅
                filter_reason = []
                if similarity >= 0.75:
                    filter_reason.append(f"similarity 높음({similarity:.3f})")
                if keyword_score < 0.1:
                    filter_reason.append(f"keyword 낮음({keyword_score:.3f})")
                if weighted_keyword < 0.1:
                    filter_reason.append(f"weighted_keyword 낮음({weighted_keyword:.3f})")
                
                reason_str = ", ".join(filter_reason) if filter_reason else "점수 부족"
                self.logger.debug(
                    f"🔍 [VECTOR FILTER OUT] 벡터 결과 제외: "
                    f"final_score={score:.3f} < {min_relevance_threshold}, "
                    f"similarity={similarity:.3f}, keyword={keyword_score:.3f}, "
                    f"weighted_keyword={weighted_keyword:.3f}, "
                    f"이유={reason_str}, "
                    f"type={doc.get('type', 'unknown')}, "
                    f"source={doc.get('source', 'Unknown')[:100]}"
                )
        
        # 필터링된 문서들의 점수 통계
        if filtered_out_docs:
            filtered_out_scores = [s[0] for s in filtered_out_docs]  # final_score
            filtered_out_similarities = [s[1] for s in filtered_out_docs]  # similarity
            filtered_out_keywords = [s[2] for s in filtered_out_docs]  # keyword_score
            filtered_out_weighted_keywords = [s[3] for s in filtered_out_docs]  # weighted_keyword
            
            avg_filtered_out = sum(filtered_out_scores) / len(filtered_out_scores)
            max_filtered_out = max(filtered_out_scores)
            min_filtered_out = min(filtered_out_scores)
            avg_similarity = sum(filtered_out_similarities) / len(filtered_out_similarities)
            max_similarity = max(filtered_out_similarities)
            avg_keyword = sum(filtered_out_keywords) / len(filtered_out_keywords) if filtered_out_keywords else 0.0
            avg_weighted_keyword = sum(filtered_out_weighted_keywords) / len(filtered_out_weighted_keywords) if filtered_out_weighted_keywords else 0.0
            
            # similarity가 높은데 final_score가 낮은 문서 수 계산
            high_sim_low_final = sum(1 for s in filtered_out_docs if s[1] >= 0.75 and s[0] < min_relevance_threshold)
            
            self.logger.debug(
                f"📊 [FILTERED OUT STATS] 제외된 문서: {len(filtered_out_docs)}개, "
                f"final_score 평균={avg_filtered_out:.3f}, 최대={max_filtered_out:.3f}, 최소={min_filtered_out:.3f}, "
                f"similarity 평균={avg_similarity:.3f}, 최대={max_similarity:.3f}, "
                f"keyword 평균={avg_keyword:.3f}, weighted_keyword 평균={avg_weighted_keyword:.3f}, "
                f"similarity 높은데 제외된 문서={high_sim_low_final}개"
            )
        
        # 모든 문서가 필터링된 경우 안전장치: 상위 문서들을 통과시킴
        if not filtered_vector_docs and vector_docs:
            self.logger.warning(
                f"⚠️ [VECTOR FILTER] 모든 문서가 필터링됨. "
                f"안전장치 적용: 상위 문서들을 통과시킴 (임계값: {min_relevance_threshold})"
            )
            # 점수 순으로 정렬하여 상위 문서 선택
            sorted_docs = sorted(
                vector_docs,
                key=lambda x: (
                    x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                    x.get("similarity", 0.0),
                    x.get("keyword_match_score", 0.0)
                ),
                reverse=True
            )
            # 최소 3개 또는 전체의 20% 중 작은 값만큼은 통과
            min_docs_to_keep = min(3, max(1, len(vector_docs) // 5))
            filtered_vector_docs = sorted_docs[:min_docs_to_keep]
            self.logger.debug(
                f"🔧 [VECTOR FILTER] 안전장치로 {len(filtered_vector_docs)}개 문서 통과시킴"
            )
        
        if len(filtered_vector_docs) < len(vector_docs):
            self.logger.debug(
                f"🔀 [VECTOR FILTER] 관련성 점수 필터링: "
                f"{len(vector_docs)}개 → {len(filtered_vector_docs)}개 "
                f"(임계값: {min_relevance_threshold:.3f})"
            )
        
        # 벡터 임베딩 결과들을 relevance_score 기준으로 정렬
        reranked = sorted(
            filtered_vector_docs,
            key=lambda x: (
                x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                x.get("similarity", 0.0),
                x.get("keyword_match_score", 0.0)
            ),
            reverse=True
        )
        
        # 최대 개수 제한 (벡터 결과만)
        max_vector_docs = 10
        reranked = reranked[:max_vector_docs]
        
        # 필터링된 문서의 점수 통계
        if filtered_vector_docs:
            filtered_scores = [
                doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                for doc in filtered_vector_docs
            ]
            avg_filtered = sum(filtered_scores) / len(filtered_scores)
            self.logger.debug(
                f"🔀 [VECTOR RERANK] {len(vector_docs)}개 → {len(reranked)}개 재랭킹 완료 "
                f"(필터링 후: {len(filtered_vector_docs)}개, 필터링된 문서 평균 점수: {avg_filtered:.3f})"
            )
        else:
            self.logger.warning(
                f"⚠️ [VECTOR RERANK] 모든 벡터 문서가 필터링됨: "
                f"{len(vector_docs)}개 → 0개 (임계값: {min_relevance_threshold:.3f})"
            )
        
        return reranked
    
    def _combine_text2sql_and_reranked_vector(
        self,
        text2sql_docs: List[Dict[str, Any]],
        reranked_vector_docs: List[Dict[str, Any]],
        query_type_str: str
    ) -> List[Dict[str, Any]]:
        """textToSQL 결과와 재랭킹된 벡터 결과 결합"""
        final_docs = []
        seen_ids = set()
        
        # 1. textToSQL 결과를 최우선으로 추가 (무조건 포함)
        for doc in text2sql_docs:
            doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
            if doc_id not in seen_ids:
                final_docs.append(doc)
                seen_ids.add(doc_id)
                self.logger.debug(f"✅ [TEXT2SQL INCLUSION] textToSQL 결과 포함: {doc_id}")
        
        # 2. 재랭킹된 벡터 결과 추가 (중복 제거)
        for doc in reranked_vector_docs:
            doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
            if doc_id not in seen_ids:
                final_docs.append(doc)
                seen_ids.add(doc_id)
        
        # 최종 개수 제한 (textToSQL 결과는 제외하고 벡터 결과만 제한)
        max_final_docs = 10
        if len(final_docs) > max_final_docs:
            text2sql_count = len(text2sql_docs)
            vector_count = max_final_docs - text2sql_count
            if vector_count > 0:
                final_docs = text2sql_docs + reranked_vector_docs[:vector_count]
            else:
                final_docs = text2sql_docs[:max_final_docs]
        
        # 최종 문서의 relevance_score 로깅
        self.logger.debug(
            f"🔀 [FINAL COMBINE] textToSQL: {len(text2sql_docs)}개, "
            f"벡터: {len([d for d in final_docs if d not in text2sql_docs])}개, "
            f"총: {len(final_docs)}개"
        )
        
        # 최종 문서들의 relevance_score 상세 로깅
        if final_docs:
            self.logger.debug(f"📊 [FINAL DOCS SCORES] 최종 선택된 {len(final_docs)}개 문서의 relevance_score:")
            for i, doc in enumerate(final_docs, 1):
                score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id") or "unknown"
                doc_type = doc.get("type", "unknown")
                source = doc.get("source", "")[:50] or "unknown"
                search_type = doc.get("search_type", "unknown")
                self.logger.debug(
                    f"   {i}. score={score:.3f}, type={doc_type}, search_type={search_type}, "
                    f"id={doc_id[:30]}, source={source}"
                )
        
        return final_docs

    def _ensure_diversity_and_limit(
        self,
        state: LegalWorkflowState,
        filtered_docs: List[Dict[str, Any]],
        weighted_docs: List[Dict[str, Any]],
        merged_docs: List[Dict[str, Any]],
        query: str,
        query_type_str: str,
        semantic_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """다양성 보장 및 최종 문서 제한"""
        max_docs = self.config.max_retrieved_docs or 20
        
        if filtered_docs:
            filtered_type_distribution = self._calculate_type_distribution(filtered_docs)
            self.logger.debug(f"🔀 [DIVERSITY] filtered_docs type distribution: {filtered_type_distribution}")
        
        has_precedent, has_decision = self._has_precedent_or_decision(filtered_docs)
        
        if not has_precedent or not has_decision:
            self.logger.debug(f"🔀 [DIVERSITY] Missing precedent={not has_precedent}, decision={not has_decision}, attempting to restore from weighted_docs (total: {len(weighted_docs)}) and semantic_results (total: {len(semantic_results)})")
            
            if weighted_docs:
                sample_doc = weighted_docs[0] if isinstance(weighted_docs[0], dict) else {}
                self.logger.debug(f"🔀 [DIVERSITY] weighted_docs sample keys: {list(sample_doc.keys())[:10]}")
                self.logger.debug(f"🔀 [DIVERSITY] weighted_docs sample type: {sample_doc.get('type')}, metadata: {type(sample_doc.get('metadata'))}")
            
            restored_count = 0
            precedent_candidates = []
            decision_candidates = []
            
            # weighted_docs에서 후보 수집
            for doc in weighted_docs:
                if not isinstance(doc, dict):
                    continue
                
                doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                already_in_filtered = any(
                    (d.get("id") or d.get("document_id") or d.get("doc_id") or str(d.get("source", ""))) == doc_id
                    for d in filtered_docs
                )
                if already_in_filtered:
                    continue
                
                doc_type = self._extract_doc_type(doc)
                content = self._extract_doc_content(doc)
                
                if not has_precedent and any(keyword in doc_type for keyword in ["precedent", "case", "case_paragraph", "판례"]):
                    if content and len(content.strip()) >= 3:
                        precedent_candidates.append((doc, doc_type, doc_id))
                
                if not has_decision and any(keyword in doc_type for keyword in ["decision", "decision_paragraph", "결정"]):
                    if content and len(content.strip()) >= 3:
                        decision_candidates.append((doc, doc_type, doc_id))
            
            # semantic_results에서도 후보 수집 (개선: keyword_results에서도 수집)
            if semantic_results and (not has_precedent or not has_decision):
                for doc in semantic_results:
                    if not isinstance(doc, dict):
                        continue
                    
                    doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                    already_in_filtered = any(
                        (d.get("id") or d.get("document_id") or d.get("doc_id") or str(d.get("source", ""))) == doc_id
                        for d in filtered_docs
                    )
                    if already_in_filtered:
                        continue
                    
                    # precedent_candidates나 decision_candidates에 이미 있는지 확인
                    already_candidate = any(
                        (c[2] == doc_id) for c in (precedent_candidates + decision_candidates)
                    )
                    if already_candidate:
                        continue
                    
                    doc_type = self._extract_doc_type(doc)
                    content = self._extract_doc_content(doc)
                    
                    if not has_precedent and any(keyword in doc_type for keyword in ["precedent", "case", "case_paragraph", "판례"]):
                        if content and len(content.strip()) >= 3:
                            precedent_candidates.append((doc, doc_type, doc_id))
                    
                    if not has_decision and any(keyword in doc_type for keyword in ["decision", "decision_paragraph", "결정"]):
                        if content and len(content.strip()) >= 3:
                            decision_candidates.append((doc, doc_type, doc_id))
            
            # keyword_results에서도 후보 수집 (개선: 추가)
            keyword_results = self._get_state_value(state, "keyword_results", [])
            if keyword_results and (not has_precedent or not has_decision):
                for doc in keyword_results:
                    if not isinstance(doc, dict):
                        continue
                    
                    doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                    already_in_filtered = any(
                        (d.get("id") or d.get("document_id") or d.get("doc_id") or str(d.get("source", ""))) == doc_id
                        for d in filtered_docs
                    )
                    if already_in_filtered:
                        continue
                    
                    already_candidate = any(
                        (c[2] == doc_id) for c in (precedent_candidates + decision_candidates)
                    )
                    if already_candidate:
                        continue
                    
                    doc_type = self._extract_doc_type(doc)
                    content = self._extract_doc_content(doc)
                    
                    if not has_precedent and any(keyword in doc_type for keyword in ["precedent", "case", "case_paragraph", "판례"]):
                        if content and len(content.strip()) >= 3:
                            precedent_candidates.append((doc, doc_type, doc_id))
                    
                    if not has_decision and any(keyword in doc_type for keyword in ["decision", "decision_paragraph", "결정"]):
                        if content and len(content.strip()) >= 3:
                            decision_candidates.append((doc, doc_type, doc_id))
            
            # 개선: merged_docs에서도 후보 수집 (추가)
            if merged_docs and (not has_precedent or not has_decision):
                for doc in merged_docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                    already_in_filtered = any(
                        (d.get("id") or d.get("document_id") or d.get("doc_id") or str(d.get("source", ""))) == doc_id
                        for d in filtered_docs
                    )
                    if already_in_filtered:
                        continue
                    
                    already_candidate = any(
                        (c[2] == doc_id) for c in (precedent_candidates + decision_candidates)
                    )
                    if already_candidate:
                        continue
                    
                    doc_type = self._extract_doc_type(doc)
                    content = self._extract_doc_content(doc)
                    
                    if not has_precedent and any(keyword in doc_type for keyword in ["precedent", "case", "case_paragraph", "판례"]):
                        if content and len(content.strip()) >= 3:
                            precedent_candidates.append((doc, doc_type, doc_id))
                    
                    if not has_decision and any(keyword in doc_type for keyword in ["decision", "decision_paragraph", "결정"]):
                        if content and len(content.strip()) >= 3:
                            decision_candidates.append((doc, doc_type, doc_id))
            
            # 판례 복원
            if not has_precedent and precedent_candidates:
                precedent_candidates.sort(key=lambda x: len(self._extract_doc_content(x[0]) or ""), reverse=True)
                best_precedent = precedent_candidates[0]
                filtered_docs.append(best_precedent[0])
                has_precedent = True
                restored_count += 1
                self.logger.debug(f"🔀 [DIVERSITY] ✅ Restored precedent document: {best_precedent[1]} (id: {best_precedent[2]})")
            
            # 결정례 복원
            if not has_decision and decision_candidates:
                decision_candidates.sort(key=lambda x: len(self._extract_doc_content(x[0]) or ""), reverse=True)
                best_decision = decision_candidates[0]
                filtered_docs.append(best_decision[0])
                has_decision = True
                restored_count += 1
                self.logger.debug(f"🔀 [DIVERSITY] ✅ Restored decision document: {best_decision[1]} (id: {best_decision[2]})")
            
            if restored_count > 0:
                self.logger.debug(f"🔀 [DIVERSITY] ✅ Restored {restored_count} documents (precedent={has_precedent}, decision={has_decision})")
            else:
                # 개선: 이미 판례/결정례 문서가 있는 경우 경고를 출력하지 않음
                if not has_precedent or not has_decision:
                    # filtered_docs의 타입 분포 확인
                    filtered_type_distribution = self._calculate_type_distribution(filtered_docs)
                    has_case_paragraph = filtered_type_distribution.get("case_paragraph", 0) > 0
                    has_decision_paragraph = filtered_type_distribution.get("decision_paragraph", 0) > 0
                    
                    # case_paragraph는 판례로 인식되므로, 이미 있으면 경고를 출력하지 않음
                    if not has_precedent and has_case_paragraph:
                        self.logger.debug(f"🔀 [DIVERSITY] case_paragraph documents already present ({filtered_type_distribution.get('case_paragraph', 0)}), no need to restore precedent")
                        has_precedent = True  # case_paragraph를 판례로 인식
                    
                    if not has_decision and has_decision_paragraph:
                        self.logger.debug(f"🔀 [DIVERSITY] decision_paragraph documents already present ({filtered_type_distribution.get('decision_paragraph', 0)}), no need to restore decision")
                        has_decision = True  # decision_paragraph를 결정례로 인식
                    
                    # 여전히 판례/결정례가 없는 경우에만 경고 출력
                    if not has_precedent or not has_decision:
                        self.logger.warning(f"🔀 [DIVERSITY] ⚠️ Missing precedent={not has_precedent}, decision={not has_decision} (precedent_candidates: {len(precedent_candidates)}, decision_candidates: {len(decision_candidates)})")
                        if weighted_docs:
                            type_distribution = {}
                            for doc in weighted_docs[:10]:
                                if isinstance(doc, dict):
                                    doc_type = self._extract_doc_type(doc)
                                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                            self.logger.debug(f"🔀 [DIVERSITY] weighted_docs type distribution: {type_distribution}")
                        if semantic_results:
                            type_distribution = {}
                            for doc in semantic_results[:10]:
                                if isinstance(doc, dict):
                                    doc_type = self._extract_doc_type(doc)
                                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                            self.logger.debug(f"🔀 [DIVERSITY] semantic_results type distribution: {type_distribution}")
                else:
                    self.logger.debug("🔀 [DIVERSITY] ✅ Precedent and decision documents already present, no restoration needed")
        
        if self.search_handler and len(filtered_docs) > 0:
            diverse_filtered_docs = self.search_handler._ensure_diverse_source_types(
                filtered_docs,
                min(max_docs * 2, len(filtered_docs))
            )
            
            if diverse_filtered_docs:
                final_type_distribution = self._calculate_type_distribution(diverse_filtered_docs)
                self.logger.debug(f"🔀 [DIVERSITY] final_docs type distribution after diversity: {final_type_distribution}")
            
            final_docs = diverse_filtered_docs[:max_docs]
        else:
            final_docs = filtered_docs[:max_docs]
        
        if not final_docs:
            self.logger.warning(
                f"⚠️ [SEARCH RESULTS] No valid documents found after filtering. "
                f"Query: '{query[:50]}...', Query type: {query_type_str}, "
                f"Total merged: {len(merged_docs)}, Filtered: {len(filtered_docs)}"
            )
            
            if weighted_docs:
                fallback_docs = []
                for doc in weighted_docs[:3]:
                    content = self._extract_doc_content(doc)
                    if content and len(content.strip()) >= 10:
                        score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                        if score >= 0.05:
                            fallback_docs.append(doc)
                
                if fallback_docs:
                    final_docs = fallback_docs
                    self.logger.debug(f"🔄 [FALLBACK] Using {len(final_docs)} lower-scored documents as fallback (original filtered count: 0)")
                else:
                    self.logger.error("❌ [SEARCH RESULTS] No fallback documents available. All documents were filtered out (content too short or score too low).")
            else:
                self.logger.error("❌ [SEARCH RESULTS] No documents available at all. Search may have failed or returned empty results.")
        
        if not final_docs or len(final_docs) == 0:
            self.logger.warning("⚠️ [SEARCH RESULTS] final_docs가 0개입니다. semantic_results에서 변환 시도...")
            if semantic_results and len(semantic_results) > 0:
                converted_docs = []
                for doc in semantic_results[:10]:
                    if isinstance(doc, dict):
                        doc_type = doc.get("type") or (doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None)
                        text_content = doc.get("text", "") or doc.get("content", "") or str(doc.get("metadata", {}).get("text", "")) or str(doc.get("metadata", {}).get("content", ""))
                        converted_doc = {
                            "content": text_content,
                            "text": text_content,
                            "source": doc.get("source", "") or doc.get("title", "Unknown"),
                            "relevance_score": doc.get("relevance_score", 0.5),
                            "search_type": "semantic",
                            "type": doc_type,
                            "metadata": doc.get("metadata", {})
                        }
                        if doc_type == "statute_article":
                            doc_metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                            converted_doc["statute_name"] = (
                                doc.get("statute_name") or
                                doc.get("law_name") or
                                doc_metadata.get("statute_name") or
                                doc_metadata.get("law_name") or
                                doc_metadata.get("abbrv")
                            )
                            converted_doc["law_name"] = converted_doc["statute_name"]
                            converted_doc["article_no"] = (
                                doc.get("article_no") or
                                doc.get("article_number") or
                                doc_metadata.get("article_no") or
                                doc_metadata.get("article_number")
                            )
                            converted_doc["article_number"] = converted_doc["article_no"]
                            converted_doc["clause_no"] = doc.get("clause_no") or doc_metadata.get("clause_no")
                            converted_doc["item_no"] = doc.get("item_no") or doc_metadata.get("item_no")
                        if converted_doc["content"] and len(converted_doc["content"].strip()) >= 10:
                            converted_docs.append(converted_doc)
                
                if converted_docs:
                    final_docs = converted_docs
                    self.logger.debug(f"🔄 [FALLBACK] Converted {len(final_docs)} documents from semantic_results to retrieved_docs (original final_docs count: 0)")
                else:
                    self.logger.error(f"❌ [SEARCH RESULTS] semantic_results에서도 변환 실패 - semantic_results 개수: {len(semantic_results)}")
        
        return final_docs
    
    def _save_final_results_to_state(
        self,
        state: LegalWorkflowState,
        final_docs: List[Dict[str, Any]],
        merged_docs: List[Dict[str, Any]],
        filtered_docs: List[Dict[str, Any]],
        overall_quality: float,
        semantic_count: int,
        keyword_count: int,
        needs_retry: bool,
        start_time: float,
        query: str = "",
        query_type_str: str = "",
        extracted_keywords: List[str] = None
    ) -> None:
        """최종 결과를 State에 저장"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        search_metadata = {
            "total_results": len(merged_docs),
            "filtered_results": len(filtered_docs),
            "final_results": len(final_docs),
            "quality_score": overall_quality,
            "semantic_count": semantic_count,
            "keyword_count": keyword_count,
            "retry_performed": needs_retry,
            "has_results": len(final_docs) > 0,
            "used_fallback": len(final_docs) > 0 and len(filtered_docs) == 0,
            "timestamp": time.time()
        }
        self._set_state_value(state, "search_metadata", search_metadata)
        
        self.logger.debug(f"📊 [SEARCH RESULTS] final_docs 설정 완료 - 개수: {len(final_docs)}")
        
        if debug_mode:
            self.logger.debug(f"💾 [SEARCH RESULTS] State 저장 전 검증 - final_docs 개수: {len(final_docs)}, 타입: {type(final_docs).__name__}")
        
        self._save_search_results_to_state(state, final_docs)
        
        processing_time = self._update_processing_time(state, start_time)
        
        # 배치 로깅 최적화: 여러 로그를 한 번에 처리
        if len(final_docs) > 0:
            final_scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in final_docs]
            min_score = min(final_scores) if final_scores else 0.0
            max_score = max(final_scores) if final_scores else 0.0
            avg_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
            
            # 배치 로깅: 한 번에 모든 정보 출력
            processed_msg = (
                f"✅ [SEARCH RESULTS] Processed {len(final_docs)} documents "
                f"(quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s, "
                f"scores: min={min_score:.3f}, max={max_score:.3f}, avg={avg_score:.3f})"
            )
            self.logger.debug(processed_msg)
            self.logger.debug(processed_msg)
            
            self._add_step(
                state,
                "검색 결과 처리",
                f"검색 결과 처리 완료: {len(final_docs)}개 문서 (품질 점수: {overall_quality:.2f}, 시간: {processing_time:.3f}s)"
            )
            
            # 검색 품질 메트릭 로깅 추가
            if final_docs:
                try:
                    # 검색 품질 메트릭 직접 계산
                    metrics = {
                        "avg_relevance": 0.0,
                        "min_relevance": 0.0,
                        "max_relevance": 0.0,
                        "diversity_score": 0.0,
                        "keyword_coverage": 0.0
                    }
                    
                    # result_ranker의 evaluate_search_quality 사용 (다차원 다양성 통합)
                    if self.result_ranker and hasattr(self.result_ranker, 'evaluate_search_quality'):
                        # 개선: extracted_keywords 로깅 강화 및 폴백 로직
                        if not extracted_keywords:
                            self.logger.warning("⚠️ [KEYWORD COVERAGE] extracted_keywords is empty or None")
                            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
                            if not extracted_keywords and "search" in state and isinstance(state.get("search"), dict):
                                extracted_keywords = state["search"].get("extracted_keywords", [])
                            
                            # 폴백: 쿼리에서 직접 키워드 추출
                            if not extracted_keywords and query:
                                import re
                                korean_words = re.findall(r'[가-힣]+', query)
                                extracted_keywords = [w for w in korean_words if len(w) >= 2]
                                
                                # 불용어 제거 (KoNLPy 기반 KoreanStopwordProcessor 사용)
                                if hasattr(self, 'stopword_processor') and self.stopword_processor:
                                    extracted_keywords = self.stopword_processor.filter_stopwords(extracted_keywords)
                                else:
                                    # 폴백: 기본 불용어 제거
                                    stopwords = {'은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로', 
                                                '에서', '에게', '한테', '께', '에게서', '한테서', '께서', '의', '것', '수', '등', 
                                                '및', '또한', '또', '그리고', '또는', '무엇', '어떤', '어떻게', '언제', '어디', 
                                                '누구', '왜', '시', '할', '하는', '된', '되는', '이다', '입니다', '있습니다', '합니다'}
                                    extracted_keywords = [kw for kw in extracted_keywords if kw not in stopwords]
                                
                                # 쿼리 타입/법률 분야 기반 기본 키워드 추가
                                query_type_keywords = {
                                    "precedent_search": ["판례", "사건", "판결", "대법원"],
                                    "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
                                    "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
                                    "procedure_guide": ["절차", "방법", "대응", "소송"],
                                    "term_explanation": ["의미", "정의", "개념", "해석"]
                                }
                                # legal_field는 state에서 직접 가져오기
                                legal_field = self._get_state_value(state, "legal_field", "")
                                if not legal_field and "search" in state and isinstance(state.get("search"), dict):
                                    search_params_local = state["search"].get("search_params", {})
                                    if isinstance(search_params_local, dict):
                                        legal_field = search_params_local.get("legal_field", "")
                                field_keywords = {
                                    "family": ["가족", "이혼", "양육", "상속", "부부"],
                                    "civil": ["민사", "계약", "손해배상", "채권", "채무"],
                                    "criminal": ["형사", "범죄", "처벌", "형량"],
                                    "labor": ["노동", "근로", "해고", "임금", "근로자"],
                                    "corporate": ["기업", "회사", "주주", "법인"]
                                }
                                
                                if query_type_str in query_type_keywords:
                                    extracted_keywords.extend(query_type_keywords[query_type_str])
                                if legal_field in field_keywords:
                                    extracted_keywords.extend(field_keywords[legal_field])
                                
                                # 중복 제거 및 정리
                                extracted_keywords = list(set(extracted_keywords))
                                extracted_keywords = [kw for kw in extracted_keywords if kw and len(kw.strip()) >= 2]
                                
                                if extracted_keywords:
                                    self.logger.debug(
                                        f"🔍 [KEYWORD COVERAGE] 쿼리에서 키워드 추출: "
                                        f"{len(extracted_keywords)}개 키워드 (query='{query[:50]}...')"
                                    )
                            
                            self.logger.debug(f"🔍 [KEYWORD COVERAGE] 최종 extracted_keywords: {len(extracted_keywords)} keywords")
                        else:
                            self.logger.debug(f"🔍 [KEYWORD COVERAGE] Using extracted_keywords: {len(extracted_keywords)} keywords")
                        
                        metrics = self.result_ranker.evaluate_search_quality(
                            query=query,
                            results=final_docs,
                            query_type=query_type_str,
                            extracted_keywords=extracted_keywords
                        )
                    else:
                        # 폴백: 기본 계산
                        scores = [doc.get("relevance_score", doc.get("final_weighted_score", 0.0)) for doc in final_docs]
                        if scores:
                            metrics["avg_relevance"] = sum(scores) / len(scores)
                            metrics["min_relevance"] = min(scores)
                            metrics["max_relevance"] = max(scores)
                        
                        contents = [doc.get("content", doc.get("text", "")) for doc in final_docs]
                        unique_terms = set()
                        total_terms = 0
                        for content in contents:
                            if isinstance(content, str):
                                terms = content.lower().split()
                                unique_terms.update(terms)
                                total_terms += len(terms)
                        
                        if total_terms > 0:
                            metrics["diversity_score"] = len(unique_terms) / total_terms
                        
                        if extracted_keywords:
                            covered_keywords = set()
                            for doc in final_docs:
                                content = doc.get("content", doc.get("text", "")).lower()
                                if isinstance(content, str):
                                    for keyword in extracted_keywords:
                                        if keyword.lower() in content:
                                            covered_keywords.add(keyword.lower())
                            
                            if extracted_keywords:
                                metrics["keyword_coverage"] = len(covered_keywords) / len(extracted_keywords)
                    
                    metrics_msg = (
                        f"📊 [SEARCH QUALITY METRICS] "
                        f"Avg Relevance: {metrics.get('avg_relevance', 0.0):.3f}, "
                        f"Min: {metrics.get('min_relevance', 0.0):.3f}, "
                        f"Max: {metrics.get('max_relevance', 0.0):.3f}, "
                        f"Diversity: {metrics.get('diversity_score', 0.0):.3f}, "
                        f"Keyword Coverage: {metrics.get('keyword_coverage', 0.0):.3f}"
                    )
                    self.logger.debug(metrics_msg)
                    self.logger.debug(metrics_msg)
                    
                    # MLflow 로깅 추가
                    try:
                        import mlflow
                        from datetime import datetime
                        from core.utils.config import Config
                        
                        # MLflow run 확인 (안전한 방법)
                        active_run = None
                        try:
                            # MLflow 버전에 따라 active_run()이 없을 수 있음
                            if hasattr(mlflow, 'active_run'):
                                active_run = mlflow.active_run()
                            else:
                                # active_run()이 없는 경우, tracking client를 통해 확인
                                try:
                                    client = mlflow.tracking.MlflowClient()
                                    # 현재 run이 있는지 확인 (간접적 방법)
                                    active_run = None  # 명시적으로 None으로 설정
                                except Exception:
                                    active_run = None
                        except (AttributeError, Exception) as e:
                            # active_run()이 없거나 오류 발생 시
                            self.logger.debug(f"MLflow active_run() not available: {e}")
                            active_run = None
                        
                        if active_run is None:
                            # MLflow 설정 확인
                            config = Config()
                            tracking_uri = config.mlflow_tracking_uri
                            if tracking_uri:
                                mlflow.set_tracking_uri(tracking_uri)
                            
                            # Experiment 설정
                            experiment_name = getattr(config, 'mlflow_experiment_name', 'rag_search_quality')
                            mlflow.set_experiment(experiment_name)
                            
                            # Run 시작
                            run_name = f"search_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            mlflow.start_run(run_name=run_name)
                            
                            # Run 확인 (다시 시도)
                            try:
                                if hasattr(mlflow, 'active_run'):
                                    active_run = mlflow.active_run()
                                    if active_run:
                                        run_id = getattr(active_run.info, 'run_id', 'unknown')
                                        self.logger.debug(f"✅ [MLFLOW] Started new MLflow run: {run_name} (run_id: {run_id})")
                            except (AttributeError, Exception) as e:
                                self.logger.debug(f"MLflow active_run() check failed after start_run: {e}")
                                active_run = None
                        
                        if active_run is not None:
                            try:
                                mlflow.log_metrics({
                                    "search_quality_avg_relevance": metrics.get('avg_relevance', 0.0),
                                    "search_quality_min_relevance": metrics.get('min_relevance', 0.0),
                                    "search_quality_max_relevance": metrics.get('max_relevance', 0.0),
                                    "search_quality_diversity": metrics.get('diversity_score', 0.0),
                                    "search_quality_keyword_coverage": metrics.get('keyword_coverage', 0.0),
                                    "search_results_count": len(final_docs),
                                    "search_overall_quality": overall_quality,
                                    "search_semantic_count": semantic_count,
                                    "search_keyword_count": keyword_count,
                                    "search_retry_performed": 1.0 if needs_retry else 0.0
                                })
                                
                                # 파라미터 대신 태그로만 저장 (태그는 변경 가능하므로 중복 로깅 오류 방지)
                                # MLflow 파라미터는 한 번 설정되면 변경할 수 없으므로, 변경 가능한 태그만 사용
                                try:
                                    mlflow.set_tags({
                                        "search_query_type": query_type_str or "",
                                        "search_processing_time": str(processing_time),
                                        "search_query": query[:100] if query else "",
                                        "search_query_length": str(len(query)) if query else "0"
                                    })
                                except Exception as tag_error:
                                    self.logger.warning(f"Failed to log MLflow tags: {tag_error}")
                                
                                run_id = getattr(active_run.info, 'run_id', 'unknown') if hasattr(active_run, 'info') else 'unknown'
                                self.logger.debug(f"✅ [MLFLOW] Search quality metrics logged to MLflow run: {run_id}")
                            except Exception as log_error:
                                self.logger.warning(f"Failed to log to MLflow: {log_error}")
                    except ImportError:
                        self.logger.debug("MLflow not available, skipping metric logging")
                    except Exception as mlflow_error:
                        # MLflow 관련 모든 오류를 안전하게 처리
                        self.logger.warning(f"Failed to log to MLflow: {mlflow_error}")
                    except Exception as e:
                        self.logger.warning(f"Failed to log to MLflow: {e}", exc_info=True)
                except Exception as e:
                    self.logger.warning(f"Failed to log search quality metrics: {e}", exc_info=True)
        else:
            no_docs_msg = f"⚠️ [SEARCH RESULTS] No documents available after processing (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
            self.logger.debug(no_docs_msg)
            self.logger.warning(no_docs_msg)
    
    def _save_search_results_to_state(
        self,
        state: LegalWorkflowState,
        final_docs: List[Dict[str, Any]]
    ) -> None:
        """검색 결과를 State에 저장 (성능 최적화: 한 번만 복사, 배치 저장)"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        # 성능 최적화: final_docs를 한 번만 복사하여 재사용
        final_docs_copy = final_docs.copy() if final_docs else []
        
        # 최상위 레벨에 저장
        self._set_state_value(state, "retrieved_docs", final_docs_copy)
        self._set_state_value(state, "merged_documents", final_docs_copy)
        
        # search 그룹에 저장
        if "search" not in state:
            state["search"] = {}
        state["search"]["retrieved_docs"] = final_docs_copy
        state["search"]["merged_documents"] = final_docs_copy
        
        # common 그룹에 저장
        if "common" not in state:
            state["common"] = {}
        if "search" not in state["common"]:
            state["common"]["search"] = {}
        state["common"]["search"]["retrieved_docs"] = final_docs_copy
        state["common"]["search"]["merged_documents"] = final_docs_copy
        
        # metadata에도 저장 (복구를 위해)
        metadata = self._get_metadata_safely(state)
        if "search" not in metadata:
            metadata["search"] = {}
        metadata["search"]["retrieved_docs"] = final_docs_copy
        metadata["search"]["merged_documents"] = final_docs_copy
        metadata["retrieved_docs"] = final_docs_copy
        self._set_state_value(state, "metadata", metadata)
        
        # global cache에도 저장 (한 번만 복사하여 재사용)
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache is None or not isinstance(_global_search_results_cache, dict):
                import core.shared.wrappers.node_wrappers as node_wrappers_module
                node_wrappers_module._global_search_results_cache = {}
                _global_search_results_cache = node_wrappers_module._global_search_results_cache
            
            # 한 번만 복사하여 여러 위치에 재사용
            _global_search_results_cache["retrieved_docs"] = final_docs_copy
            _global_search_results_cache["merged_documents"] = final_docs_copy
            
            if "search" not in _global_search_results_cache:
                _global_search_results_cache["search"] = {}
            _global_search_results_cache["search"]["retrieved_docs"] = final_docs_copy
            _global_search_results_cache["search"]["merged_documents"] = final_docs_copy
            
            if "common" not in _global_search_results_cache:
                _global_search_results_cache["common"] = {}
            if "search" not in _global_search_results_cache["common"]:
                _global_search_results_cache["common"]["search"] = {}
            _global_search_results_cache["common"]["search"]["retrieved_docs"] = final_docs_copy
            
            # 배치 로깅 (한 번만 로깅)
            if debug_mode:
                saved_retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
                saved_search_group = state.get("search", {}).get("retrieved_docs", [])
                saved_common_group = state.get("common", {}).get("search", {}).get("retrieved_docs", [])
                self.logger.debug(
                    f"✅ [SAVE RESULTS] 저장 완료: {len(final_docs_copy)}개 문서 "
                    f"(최상위: {len(saved_retrieved_docs)}, search: {len(saved_search_group)}, "
                    f"common: {len(saved_common_group)}, 전역 캐시)"
                )
            else:
                self.logger.debug(
                    f"✅ [SAVE RESULTS] 저장 완료: {len(final_docs_copy)}개 문서 "
                    f"(최상위, search, common, 전역 캐시)"
                )
        except (ImportError, AttributeError, TypeError) as e:
            if debug_mode:
                self.logger.debug(f"Could not save to global cache: {e}")

    @with_state_optimization("process_search_results_combined", enable_reduction=True)
    def process_search_results_combined(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)"""
        self.logger.debug("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작")

        try:
            start_time = time.time()

            search_inputs = self._prepare_search_inputs(state)
            semantic_results = search_inputs["semantic_results"]
            keyword_results = search_inputs["keyword_results"]
            semantic_count = search_inputs["semantic_count"]
            keyword_count = search_inputs["keyword_count"]
            query = search_inputs["query"]
            query_type_str = search_inputs["query_type_str"]
            search_params = search_inputs["search_params"]
            extracted_keywords = search_inputs["extracted_keywords"]
            
            # TASK 5: Fallback 키워드 추출 (extracted_keywords가 비어있을 경우)
            if not extracted_keywords:
                self.logger.warning(f"⚠️ [KEYWORD FALLBACK] extracted_keywords가 비어있음. 쿼리에서 직접 추출: {query}")
                legal_field = self._get_state_value(state, "legal_field", "")
                extracted_keywords = self._extract_keywords_fallback(query, query_type_str, legal_field)
                # Fallback으로 추출한 키워드도 state에 저장
                if extracted_keywords:
                    self._save_keywords_to_state(state, extracted_keywords)
                    self.logger.info(f"✅ [KEYWORD FALLBACK] {len(extracted_keywords)}개 키워드 추출 및 저장 완료")
            
            # 🔥 개선: 입력 검색 결과의 metadata에서 최상위 필드로 메타데이터 복사 및 type 정규화
            semantic_results = self._normalize_documents(semantic_results)
            keyword_results = self._normalize_documents(keyword_results)
            
            # 🔍 로깅: 입력 검색 결과의 메타데이터 확인
            self.logger.info(f"🔍 [METADATA TRACE] process_search_results_combined 시작 - semantic={len(semantic_results)}, keyword={len(keyword_results)}")
            if semantic_results:
                sample_semantic = semantic_results[0] if semantic_results else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] Semantic result sample (after metadata copy): "
                    f"type={sample_semantic.get('type')}, "
                    f"has_statute_fields={bool(sample_semantic.get('statute_name') or sample_semantic.get('law_name') or sample_semantic.get('article_no'))}, "
                    f"has_case_fields={bool(sample_semantic.get('case_id') or sample_semantic.get('court') or sample_semantic.get('doc_id'))}, "
                    f"metadata_keys={list(sample_semantic.get('metadata', {}).keys())[:10] if isinstance(sample_semantic.get('metadata'), dict) else 'N/A'}"
                )
            if keyword_results:
                sample_keyword = keyword_results[0] if keyword_results else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] Keyword result sample (after metadata copy): "
                    f"type={sample_keyword.get('type')}, "
                    f"has_statute_fields={bool(sample_keyword.get('statute_name') or sample_keyword.get('law_name') or sample_keyword.get('article_no'))}, "
                    f"has_case_fields={bool(sample_keyword.get('case_id') or sample_keyword.get('court') or sample_keyword.get('doc_id'))}, "
                    f"metadata_keys={list(sample_keyword.get('metadata', {}).keys())[:10] if isinstance(sample_keyword.get('metadata'), dict) else 'N/A'}"
                )

            # 🔥 개선 1: 검색 결과가 0개일 때 즉시 Early Exit (timeout 방지)
            # semantic_count/keyword_count와 실제 리스트 길이 모두 확인
            actual_semantic_count = len(semantic_results) if semantic_results else 0
            actual_keyword_count = len(keyword_results) if keyword_results else 0
            total_count = semantic_count + keyword_count
            total_actual = actual_semantic_count + actual_keyword_count
            
            if total_count == 0 and total_actual == 0:
                self.logger.warning(
                    f"⚠️ [EARLY EXIT] 검색 결과가 없습니다. "
                    f"빠른 응답 생성을 위해 처리 중단: "
                    f"semantic={semantic_count}, keyword={keyword_count}, "
                    f"query='{query[:50]}...'"
                )
                
                # 빈 결과를 state에 저장하고 즉시 반환
                self._set_state_value(state, "retrieved_docs", [])
                self._set_state_value(state, "sources", [])
                self._set_state_value(state, "search_quality_evaluation", {
                    "overall_quality": 0.0,
                    "needs_retry": False,
                    "early_exit": True,
                    "reason": "no_search_results"
                })
                self._update_processing_time(state, start_time)
                
                elapsed_time = time.time() - start_time
                self.logger.debug(f"✅ [EARLY EXIT] 빈 검색 결과 처리 완료 ({elapsed_time:.2f}초)")
                return state

            quality_evaluation = self._evaluate_search_quality_internal(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
                query_type_str=query_type_str,
                search_params=search_params
            )
            overall_quality = quality_evaluation["overall_quality"]
            needs_retry = quality_evaluation["needs_retry"]
            self._set_state_value(state, "search_quality_evaluation", quality_evaluation)

            # 개선: 검색 결과 품질 검증 (우선순위 2)
            # 법조문 조회 쿼리인데 법조문이 없으면 textToSQL 강제 실행
            has_statute_article = False
            all_results = semantic_results + keyword_results
            for doc in all_results:
                doc_type = doc.get("type", "") or ""
                if "statute" in doc_type.lower() or (doc.get("law_name") and doc.get("article_no")):
                    has_statute_article = True
                    break
            
            # 법조문 조회 쿼리인데 법조문이 없으면 textToSQL 강제 실행
            if query_type_str == "law_inquiry" and not has_statute_article:
                self.logger.debug("[SEARCH QUALITY] 법조문 조회 쿼리인데 법조문이 검색되지 않음. textToSQL 강제 실행")
                try:
                    from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
                    data_connector = LegalDataConnectorV2()
                    text2sql_results = data_connector.search_documents(query, limit=5)
                    if text2sql_results:
                        keyword_results.extend(text2sql_results)
                        keyword_count += len(text2sql_results)
                        self.logger.debug(f"[SEARCH QUALITY] textToSQL 강제 실행: {len(text2sql_results)}개 법조문 검색 성공")
                        self.logger.debug(f"✅ [SEARCH QUALITY] textToSQL 강제 실행: {len(text2sql_results)}개 법조문 검색 성공")
                        # 품질 재평가
                        quality_evaluation = self._evaluate_search_quality_internal(
                            semantic_results=semantic_results,
                            keyword_results=keyword_results,
                            query=query,
                            query_type_str=query_type_str,
                            search_params=search_params
                        )
                        overall_quality = quality_evaluation["overall_quality"]
                        needs_retry = quality_evaluation["needs_retry"]
                except Exception as e:
                    self.logger.debug(f"[SEARCH QUALITY] textToSQL 강제 실행 실패: {e}")
                    self.logger.warning(f"⚠️ [SEARCH QUALITY] textToSQL 강제 실행 실패: {e}")

            # 조기 종료 로직: 품질이 충분히 높고 결과 수가 충분하면 추가 처리 생략
            min_quality_for_early_exit = 0.8
            min_results_for_early_exit = 5
            total_results = len(semantic_results) + len(keyword_results)
            
            if overall_quality >= min_quality_for_early_exit and total_results >= min_results_for_early_exit:
                self.logger.debug(
                    f"✅ [EARLY EXIT] Quality sufficient (quality={overall_quality:.2f}, "
                    f"results={total_results}), skipping retry search"
                )
                # 🔥 확장된 쿼리 결과 병합 및 중복 제거 (최소 변경)
                if semantic_results:
                    # 디버그: sub_query 필드 확인
                    has_sub_query = any(
                        doc.get("sub_query") or doc.get("multi_query_source") or doc.get("expanded_query_id")
                        for doc in semantic_results if isinstance(doc, dict)
                    )
                    if has_sub_query:
                        self.logger.debug(f"🔄 [MERGE EXPANDED] Found expanded query results: {len(semantic_results)} docs with sub_query fields")
                    semantic_results = self._consolidate_expanded_query_results(semantic_results, query)
                
                # 🔥 개선: _merge_and_rerank_results 호출 전에 메타데이터 보존
                # semantic_results와 keyword_results의 메타데이터를 최상위 필드로 복사
                for doc in semantic_results + keyword_results:
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    if not isinstance(doc["metadata"], dict):
                        doc["metadata"] = {}
                    
                    metadata = doc["metadata"]
                    # metadata에서 최상위 필드로 복사 (우선순위: metadata > 최상위 필드)
                    if metadata.get("type") and not doc.get("type"):
                        doc["type"] = metadata.get("type")
                    if metadata.get("statute_name") and not doc.get("statute_name"):
                        doc["statute_name"] = metadata.get("statute_name")
                    if metadata.get("law_name") and not doc.get("law_name"):
                        doc["law_name"] = metadata.get("law_name")
                    if metadata.get("article_no") and not doc.get("article_no"):
                        doc["article_no"] = metadata.get("article_no")
                    if metadata.get("case_id") and not doc.get("case_id"):
                        doc["case_id"] = metadata.get("case_id")
                    if metadata.get("court") and not doc.get("court"):
                        doc["court"] = metadata.get("court")
                    if metadata.get("doc_id") and not doc.get("doc_id"):
                        doc["doc_id"] = metadata.get("doc_id")
                    if metadata.get("casenames") and not doc.get("casenames"):
                        doc["casenames"] = metadata.get("casenames")
                    if metadata.get("precedent_id") and not doc.get("precedent_id"):
                        doc["precedent_id"] = metadata.get("precedent_id")
                
                # 재검색 생략하고 바로 병합 진행
                # 🔥 성능 최적화: 확장된 쿼리 결과 병합을 먼저 수행
                if semantic_results:
                    semantic_results = self._consolidate_expanded_query_results(semantic_results, query)
                
                merged_docs = self._merge_and_rerank_results(
                    state, semantic_results, keyword_results, query
                )
                # 🔥 성능 최적화: 이미 병합 완료했으므로 중복 호출 방지 플래그 설정
                already_merged = True
            else:
                semantic_results, keyword_results, semantic_count, keyword_count = self._perform_conditional_retry_search(
                    state, semantic_results, keyword_results, semantic_count, keyword_count,
                    quality_evaluation, query, query_type_str, search_params, extracted_keywords
                )
                already_merged = False

            # 🔥 확장된 쿼리 결과 병합 및 중복 제거 (이미 병합되지 않은 경우만)
            if not already_merged:
                if semantic_results:
                    # 디버그: sub_query 필드 확인
                    has_sub_query = any(
                        doc.get("sub_query") or doc.get("multi_query_source") or doc.get("expanded_query_id")
                        for doc in semantic_results if isinstance(doc, dict)
                    )
                    if has_sub_query:
                        self.logger.debug(f"🔄 [MERGE EXPANDED] Found expanded query results: {len(semantic_results)} docs with sub_query fields")
                    semantic_results = self._consolidate_expanded_query_results(semantic_results, query)

                # 🔥 성능 최적화: 중복 호출 방지 - 이미 병합되지 않은 경우만 호출
                merged_docs = self._merge_and_rerank_results(
                    state, semantic_results, keyword_results, query
                )

            # 개선 3.1: 모든 문서에 점수 보장 및 type 정보 보존 (DocumentType Enum 사용)
            from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
            
            # 🔥 개선: merged_docs에 직접 메타데이터 보존 로직 적용
            # 먼저 모든 문서의 점수를 일괄 정규화
            if merged_docs:
                merged_docs = self._normalize_scores_batch(merged_docs)
            
            for doc in merged_docs:
                # 점수 보장 (정규화된 점수 사용)
                doc = self._ensure_scores(doc)
                
                # metadata 필드 보존 및 보강
                if "metadata" not in doc:
                    doc["metadata"] = {}
                if not isinstance(doc["metadata"], dict):
                    doc["metadata"] = {}
                
                # 🔥 개선: 기존 metadata의 정보를 먼저 최상위 필드로 복사 (일관성 유지)
                metadata = doc["metadata"]
                if isinstance(metadata, dict):
                    # metadata에서 최상위 필드로 복사 (우선순위: metadata > 최상위 필드)
                    if metadata.get("statute_name") and not doc.get("statute_name"):
                        doc["statute_name"] = metadata.get("statute_name")
                    if metadata.get("law_name") and not doc.get("law_name"):
                        doc["law_name"] = metadata.get("law_name")
                    if metadata.get("article_no") and not doc.get("article_no"):
                        doc["article_no"] = metadata.get("article_no")
                    if metadata.get("article_number") and not doc.get("article_no"):
                        doc["article_no"] = metadata.get("article_number")
                    if metadata.get("case_id") and not doc.get("case_id"):
                        doc["case_id"] = metadata.get("case_id")
                    if metadata.get("court") and not doc.get("court"):
                        doc["court"] = metadata.get("court")
                    if metadata.get("ccourt") and not doc.get("court"):
                        doc["court"] = metadata.get("ccourt")
                    if metadata.get("doc_id") and not doc.get("doc_id"):
                        doc["doc_id"] = metadata.get("doc_id")
                    if metadata.get("casenames") and not doc.get("casenames"):
                        doc["casenames"] = metadata.get("casenames")
                    if metadata.get("precedent_id") and not doc.get("precedent_id"):
                        doc["precedent_id"] = metadata.get("precedent_id")
                    # type도 복사
                    if metadata.get("type") and not doc.get("type"):
                        doc["type"] = metadata.get("type")
                
                # 최상위 필드의 정보를 metadata에 복사 (DocumentType 추론을 위해)
                # statute_article 관련 필드
                if doc.get("statute_name") or doc.get("law_name") or doc.get("article_no"):
                    doc["metadata"]["statute_name"] = doc.get("statute_name") or doc.get("law_name")
                    doc["metadata"]["law_name"] = doc.get("law_name") or doc.get("statute_name")
                    doc["metadata"]["article_no"] = doc.get("article_no") or doc.get("article_number")
                
                # precedent_content 관련 필드
                if doc.get("case_id") or doc.get("court") or doc.get("doc_id") or doc.get("casenames"):
                    doc["metadata"]["case_id"] = doc.get("case_id")
                    doc["metadata"]["court"] = doc.get("court") or doc.get("ccourt")
                    doc["metadata"]["doc_id"] = doc.get("doc_id")
                    doc["metadata"]["casenames"] = doc.get("casenames")
                    doc["metadata"]["precedent_id"] = doc.get("precedent_id")
                
                # DocumentType Enum을 사용하여 타입 추출
                doc_type = DocumentType.from_metadata(doc)
                doc_type_str = doc_type.value
                
                # 타입 정보 설정
                doc["type"] = doc_type_str
                # metadata에도 타입 정보 저장 (일관성 유지)
                doc["metadata"]["type"] = doc_type_str

            # 로깅 최적화: DEBUG 레벨로 변경 (성능 향상)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"🔍 [BEFORE RERANK] About to call _apply_keyword_weights_and_rerank: "
                    f"merged_docs={len(merged_docs)}, overall_quality={overall_quality:.2f}"
                )
            
            weighted_docs = self._apply_keyword_weights_and_rerank(
                state, merged_docs, query, query_type_str, extracted_keywords,
                search_params, overall_quality
            )
            
            # 🔍 로깅: 재랭킹 후 메타데이터 확인
            if weighted_docs:
                sample_weighted = weighted_docs[0] if weighted_docs else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] After rerank sample: "
                    f"type={sample_weighted.get('type')}, "
                    f"has_statute_fields={bool(sample_weighted.get('statute_name') or sample_weighted.get('law_name') or sample_weighted.get('article_no'))}, "
                    f"has_case_fields={bool(sample_weighted.get('case_id') or sample_weighted.get('court') or sample_weighted.get('doc_id'))}, "
                    f"metadata_keys={list(sample_weighted.get('metadata', {}).keys())[:10] if isinstance(sample_weighted.get('metadata'), dict) else 'N/A'}"
                )
            
            # 로깅 최적화: DEBUG 레벨로 변경 (성능 향상)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"🔍 [AFTER RERANK] _apply_keyword_weights_and_rerank completed: "
                    f"weighted_docs={len(weighted_docs)}"
                )

            filtered_docs = self._filter_and_validate_documents(
                state, weighted_docs, query, extracted_keywords, merged_docs
            )

            # textToSQL 결과 분리 및 벡터 임베딩 결과 재랭킹
            text2sql_docs, vector_docs = self._separate_text2sql_and_vector_results(
                filtered_docs, weighted_docs, merged_docs
            )
            
            # 벡터 임베딩 결과만 재랭킹
            reranked_vector_docs = self._rerank_vector_results_only(
                state, vector_docs, query, query_type_str, extracted_keywords
            )
            
            # 우선순위 4: 법령 조문 부스팅 적용 (병렬 처리 최적화)
            from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
            
            start_boosting_time = time.time()
            doc_count = len(text2sql_docs) + len(reranked_vector_docs)
            timeout = max(10.0, min(30.0, doc_count * 0.1))
            
            results = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(self._apply_statute_article_boosting, text2sql_docs, query): "text2sql",
                    executor.submit(self._apply_statute_article_boosting, reranked_vector_docs, query): "reranked"
                }
                
                for future in as_completed(futures, timeout=timeout):
                    task_name = futures[future]
                    try:
                        results[task_name] = future.result()
                        self.logger.debug(f"{task_name} boosting completed")
                    except FutureTimeoutError:
                        self.logger.warning(f"{task_name} boosting timeout ({timeout}s), using original docs")
                        results[task_name] = text2sql_docs if task_name == "text2sql" else reranked_vector_docs
                    except Exception as e:
                        self.logger.error(f"{task_name} boosting error: {e}, using original docs", exc_info=self.logger.isEnabledFor(logging.DEBUG))
                        results[task_name] = text2sql_docs if task_name == "text2sql" else reranked_vector_docs
            
            text2sql_docs = results.get("text2sql", text2sql_docs)
            reranked_vector_docs = results.get("reranked", reranked_vector_docs)
            
            elapsed = time.time() - start_boosting_time
            self.logger.debug(f"Statute article boosting completed in {elapsed:.2f}s")
            
            # textToSQL 결과와 재랭킹된 벡터 결과 결합
            final_docs = self._combine_text2sql_and_reranked_vector(
                text2sql_docs, reranked_vector_docs, query_type_str
            )
            
            # 개선: final_docs가 비어있을 때 원본 검색 결과에서 최소한의 결과 보장
            if not final_docs or len(final_docs) == 0:
                self.logger.warning(
                    f"⚠️ [SEARCH RESULTS] final_docs가 비어있음. "
                    f"원본 검색 결과에서 최소한의 결과 보장 시도: "
                    f"semantic={len(semantic_results)}, keyword={len(keyword_results)}, "
                    f"merged={len(merged_docs)}, weighted={len(weighted_docs)}, "
                    f"filtered={len(filtered_docs)}"
                )
                
                # 원본 검색 결과에서 최소한의 결과 선택
                fallback_candidates = []
                
                # 1. text2sql_docs가 있으면 우선 사용
                if text2sql_docs:
                    fallback_candidates.extend(text2sql_docs[:3])
                
                # 2. reranked_vector_docs가 있으면 사용
                if reranked_vector_docs:
                    fallback_candidates.extend(reranked_vector_docs[:3])
                
                # 3. filtered_docs가 있으면 사용
                if filtered_docs:
                    fallback_candidates.extend(filtered_docs[:3])
                
                # 4. weighted_docs가 있으면 사용
                if weighted_docs and not fallback_candidates:
                    fallback_candidates.extend(weighted_docs[:5])
                
                # 5. merged_docs가 있으면 사용
                if merged_docs and not fallback_candidates:
                    fallback_candidates.extend(merged_docs[:5])
                
                # 6. 원본 검색 결과에서 직접 선택
                if not fallback_candidates:
                    all_original_results = semantic_results + keyword_results
                    if all_original_results:
                        # 점수 순으로 정렬하여 상위 문서 선택
                        sorted_original = sorted(
                            all_original_results,
                            key=lambda x: (
                                x.get("relevance_score", 0.0),
                                x.get("similarity", 0.0),
                                x.get("final_weighted_score", 0.0)
                            ),
                            reverse=True
                        )
                        fallback_candidates = sorted_original[:5]
                
                # 중복 제거
                seen_ids = set()
                final_fallback = []
                for doc in fallback_candidates:
                    doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        final_fallback.append(doc)
                        if len(final_fallback) >= 5:
                            break
                
                if final_fallback:
                    final_docs = final_fallback
                    self.logger.debug(
                        f"✅ [SEARCH RESULTS] 원본 검색 결과에서 {len(final_docs)}개 문서 복원 완료"
                    )
                else:
                    self.logger.error(
                        "❌ [SEARCH RESULTS] 원본 검색 결과에서도 문서를 찾을 수 없음. "
                        "검색이 완전히 실패했을 수 있습니다."
                    )
            
            # 개선: final_docs의 모든 문서에 type 정보 보장 (DocumentType Enum 사용)
            from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
            
            # 🔍 로깅: final_docs 처리 전 메타데이터 확인
            if final_docs:
                sample_final_before = final_docs[0] if final_docs else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] Final docs before type setting (sample): "
                    f"type={sample_final_before.get('type')}, "
                    f"has_statute_fields={bool(sample_final_before.get('statute_name') or sample_final_before.get('law_name') or sample_final_before.get('article_no'))}, "
                    f"has_case_fields={bool(sample_final_before.get('case_id') or sample_final_before.get('court') or sample_final_before.get('doc_id'))}"
                )
            
            # final_docs의 각 문서에 타입 정보 보장
            for doc in final_docs:
                if not isinstance(doc, dict):
                    continue
                
                # metadata 필드 보존 및 보강
                if "metadata" not in doc:
                    doc["metadata"] = {}
                if not isinstance(doc["metadata"], dict):
                    doc["metadata"] = {}
                
                # 최상위 필드의 정보를 metadata에 복사 (DocumentType 추론을 위해)
                # statute_article 관련 필드
                if doc.get("statute_name") or doc.get("law_name") or doc.get("article_no"):
                    doc["metadata"]["statute_name"] = doc.get("statute_name") or doc.get("law_name")
                    doc["metadata"]["law_name"] = doc.get("law_name") or doc.get("statute_name")
                    doc["metadata"]["article_no"] = doc.get("article_no") or doc.get("article_number")
                
                # precedent_content 관련 필드
                if doc.get("case_id") or doc.get("court") or doc.get("doc_id") or doc.get("casenames"):
                    doc["metadata"]["case_id"] = doc.get("case_id")
                    doc["metadata"]["court"] = doc.get("court") or doc.get("ccourt")
                    doc["metadata"]["doc_id"] = doc.get("doc_id")
                    doc["metadata"]["casenames"] = doc.get("casenames")
                    doc["metadata"]["precedent_id"] = doc.get("precedent_id")
                
                # 기존 metadata의 정보도 최상위 필드로 복사 (일관성 유지)
                metadata = doc["metadata"]
                if metadata.get("statute_name") and not doc.get("statute_name"):
                    doc["statute_name"] = metadata.get("statute_name")
                if metadata.get("law_name") and not doc.get("law_name"):
                    doc["law_name"] = metadata.get("law_name")
                if metadata.get("article_no") and not doc.get("article_no"):
                    doc["article_no"] = metadata.get("article_no")
                if metadata.get("case_id") and not doc.get("case_id"):
                    doc["case_id"] = metadata.get("case_id")
                if metadata.get("court") and not doc.get("court"):
                    doc["court"] = metadata.get("court")
                if metadata.get("doc_id") and not doc.get("doc_id"):
                    doc["doc_id"] = metadata.get("doc_id")
                if metadata.get("casenames") and not doc.get("casenames"):
                    doc["casenames"] = metadata.get("casenames")
                
                # DocumentType Enum을 사용하여 타입 추출
                doc_type = DocumentType.from_metadata(doc)
                doc_type_str = doc_type.value
                
                # 🔍 로깅: final_docs 타입 추론 과정 추적 (처음 3개만)
                if final_docs.index(doc) < 3:
                    self.logger.info(
                        f"🔍 [METADATA TRACE] Final doc {final_docs.index(doc)} type inference: "
                        f"inferred_type={doc_type_str}, "
                        f"has_statute_fields={bool(doc.get('statute_name') or doc.get('law_name') or doc.get('article_no') or doc.get('metadata', {}).get('statute_name') or doc.get('metadata', {}).get('law_name') or doc.get('metadata', {}).get('article_no'))}, "
                        f"has_case_fields={bool(doc.get('case_id') or doc.get('court') or doc.get('doc_id') or doc.get('metadata', {}).get('case_id') or doc.get('metadata', {}).get('court') or doc.get('metadata', {}).get('doc_id'))}, "
                        f"top_level_keys={[k for k in doc.keys() if k in ['statute_name', 'law_name', 'article_no', 'case_id', 'court', 'doc_id', 'casenames', 'precedent_id']][:5]}, "
                        f"metadata_keys={[k for k in doc.get('metadata', {}).keys() if k in ['statute_name', 'law_name', 'article_no', 'case_id', 'court', 'doc_id', 'casenames', 'precedent_id']][:5]}"
                    )
                
                # 타입 정보 설정
                doc["type"] = doc_type_str
                # metadata에도 타입 정보 저장 (일관성 유지)
                doc["metadata"]["type"] = doc_type_str
                
                # 점수도 보장 (정규화)
                doc = self._ensure_scores(doc)
            
            # 🔍 로깅: final_docs 처리 후 메타데이터 확인
            if final_docs:
                sample_final_after = final_docs[0] if final_docs else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] Final docs after type setting (sample): "
                    f"type={sample_final_after.get('type')}, "
                    f"metadata_type={sample_final_after.get('metadata', {}).get('type') if isinstance(sample_final_after.get('metadata'), dict) else 'N/A'}"
                )

            # 개선 1.1: 최소 문서 수 보장
            MIN_DOCUMENTS_FOR_ANSWER = 5
            MAX_DOCUMENTS_FOR_ANSWER = 20
            
            if len(final_docs) < MIN_DOCUMENTS_FOR_ANSWER and len(merged_docs) > len(final_docs):
                additional_docs = [
                    doc for doc in merged_docs 
                    if doc not in final_docs 
                    and doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) >= 0.5
                ]
                additional_docs.sort(
                    key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), 
                    reverse=True
                )
                needed_count = MIN_DOCUMENTS_FOR_ANSWER - len(final_docs)
                final_docs.extend(additional_docs[:needed_count])
                self.logger.debug(
                    f"🔍 [MIN DOCS] 최소 문서 수 보장: {len(final_docs)}개 "
                    f"(추가: {needed_count}개)"
                )
            
            if len(final_docs) > MAX_DOCUMENTS_FOR_ANSWER:
                final_docs = final_docs[:MAX_DOCUMENTS_FOR_ANSWER]
                self.logger.debug(f"🔍 [MAX DOCS] 최대 문서 수 제한: {MAX_DOCUMENTS_FOR_ANSWER}개")

            # 개선 5.1: 문서 다양성 보장
            MIN_DOCS_FOR_DIVERSITY = 5
            if len(final_docs) < MIN_DOCS_FOR_DIVERSITY:
                type_distribution = {}
                for doc in final_docs:
                    doc_type = doc.get("type", "unknown")
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                
                required_types = ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]
                missing_types = [t for t in required_types if type_distribution.get(t, 0) == 0]
                
                if missing_types and len(merged_docs) > len(final_docs):
                    for doc_type in missing_types:
                        additional = [
                            doc for doc in merged_docs
                            if doc not in final_docs
                            and doc.get("type") == doc_type
                        ]
                        if additional:
                            additional.sort(
                                key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                                reverse=True
                            )
                            final_docs.append(additional[0])
                            self.logger.debug(f"✅ [DIVERSITY] {doc_type} 타입 추가: 1개")

            self._save_final_results_to_state(
                state, final_docs, merged_docs, filtered_docs, overall_quality,
                semantic_count, keyword_count, needs_retry, start_time,
                query=query, query_type_str=query_type_str, extracted_keywords=extracted_keywords
            )
            
            # 🔥 개선: 검색 결과를 여러 위치에 명시적으로 저장 (reduction 방지)
            # metadata에도 명시적으로 저장 (reduction으로부터 보호)
            metadata = self._get_metadata_safely(state)
            if "search" not in metadata:
                metadata["search"] = {}
            metadata["search"]["retrieved_docs"] = final_docs.copy() if final_docs else []
            metadata["retrieved_docs"] = final_docs.copy() if final_docs else []
            self._set_state_value(state, "metadata", metadata)
            
            # top-level에 명시적으로 저장 (가장 안전한 위치)
            if final_docs:
                # 🔍 로깅: 최종 저장 전 메타데이터 확인
                sample_save = final_docs[0] if final_docs else {}
                self.logger.info(
                    f"🔍 [METADATA TRACE] 최종 저장 전 (sample): "
                    f"type={sample_save.get('type')}, "
                    f"metadata_type={sample_save.get('metadata', {}).get('type') if isinstance(sample_save.get('metadata'), dict) else 'N/A'}"
                )
                
                state["retrieved_docs"] = final_docs.copy()
                state["structured_documents"] = final_docs.copy()

        except Exception as e:
            # 개선 1: 예외 발생 시에도 로깅 보장
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(
                f"❌ [SEARCH RESULTS ERROR] process_search_results_combined 실행 중 예외 발생: {str(e)}\n"
                f"   Query: '{query[:50] if 'query' in locals() else 'N/A'}...', "
                f"Query type: {query_type_str if 'query_type_str' in locals() else 'N/A'}\n"
                f"   스택 트레이스:\n{error_traceback}"
            )
            self._handle_error(state, str(e), "검색 결과 처리 중 오류 발생")

            # 폴백: 기존 검색 결과가 있으면 사용 시도
            existing_semantic = self._get_state_value(state, "semantic_results", [])
            existing_keyword = self._get_state_value(state, "keyword_results", [])

            fallback_docs = []
            if existing_semantic:
                for doc in existing_semantic[:5]:  # 최대 5개
                    if isinstance(doc, dict) and (doc.get("content") or doc.get("text")):
                        fallback_docs.append(doc)
            if not fallback_docs and existing_keyword:
                for doc in existing_keyword[:5]:  # 최대 5개
                    if isinstance(doc, dict) and (doc.get("content") or doc.get("text")):
                        fallback_docs.append(doc)

            if fallback_docs:
                self.logger.debug(
                    f"🔄 [FALLBACK] Using {len(fallback_docs)} documents from original search results "
                    f"as fallback after processing error"
                )
                self._set_state_value(state, "retrieved_docs", fallback_docs)
                self._set_state_value(state, "merged_documents", fallback_docs)
            else:
                # 최종 폴백: 빈 리스트
                self.logger.warning(
                    "⚠️ [FALLBACK] No fallback documents available. Setting empty retrieved_docs."
                )
                self._set_state_value(state, "retrieved_docs", [])
                self._set_state_value(state, "merged_documents", [])

        return state

    def _get_search_params_batch(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """SearchExecutionProcessor.get_search_params 래퍼"""
        if self.search_execution_processor:
            return self.search_execution_processor.get_search_params(state)
        return {
            "optimized_queries": {},
            "search_params": {},
            "query_type_str": "",
            "legal_field": "",
            "extracted_keywords": [],
            "original_query": ""
        }

    @with_state_optimization("execute_searches_parallel", enable_reduction=True)
    def execute_searches_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """의미적 검색과 키워드 검색을 병렬로 실행"""
        try:
            start_time = time.time()

            if self.search_execution_processor:
                state = self.search_execution_processor.execute_searches_parallel(state)
                self._update_processing_time(state, start_time)
            else:
                self.logger.warning("SearchExecutionProcessor not available, falling back to sequential search")
                # 폴백: 순차 검색 실행
                state = self._fallback_sequential_search(state)
                self._update_processing_time(state, start_time)

        except Exception as e:
            self._handle_error(state, str(e), "병렬 검색 중 오류 발생")
            if self.search_execution_processor:
                try:
                    return self.search_execution_processor.fallback_sequential_search(state)
                except Exception as fallback_err:
                    self.logger.error(f"SearchExecutionProcessor fallback also failed: {fallback_err}")
                    # 최종 폴백: 직접 순차 검색
                    return self._fallback_sequential_search(state)
            else:
                # SearchExecutionProcessor가 없으면 직접 순차 검색
                return self._fallback_sequential_search(state)

        return state

    def _execute_semantic_search_internal(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        original_query: str = "",
        extracted_keywords: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """SearchExecutionProcessor.execute_semantic_search 래퍼"""
        if self.search_execution_processor:
            return self.search_execution_processor.execute_semantic_search(
                optimized_queries, search_params, original_query, extracted_keywords
            )
        # SearchExecutionProcessor가 없으면 직접 semantic_search 실행
        if self.semantic_search and original_query:
            try:
                semantic_query = optimized_queries.get("semantic_query", original_query)
                k = search_params.get("semantic_k", 10)
                results, count = self.semantic_search.search(
                    query=semantic_query,
                    k=k,
                    extracted_keywords=extracted_keywords
                )
                return results or [], count or 0
            except Exception as e:
                self.logger.warning(f"Direct semantic search failed: {e}")
        return [], 0

    def _execute_keyword_search_internal(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        query_type_str: str,
        legal_field: str,
        extracted_keywords: List[str],
        original_query: str = ""
    ) -> Tuple[List[Dict[str, Any]], int]:
        """SearchExecutionProcessor.execute_keyword_search 래퍼"""
        if self.search_execution_processor:
            return self.search_execution_processor.execute_keyword_search(
                optimized_queries, search_params, query_type_str, legal_field, extracted_keywords, original_query
            )
        # SearchExecutionProcessor가 없으면 직접 keyword_search 실행
        if original_query:
            try:
                keyword_limit = search_params.get("keyword_limit", 10)
                results, count = self._keyword_search(
                    query=original_query,
                    query_type_str=query_type_str,
                    limit=keyword_limit,
                    legal_field=legal_field,
                    extracted_keywords=extracted_keywords or []
                )
                return results or [], count or 0
            except Exception as e:
                self.logger.warning(f"Direct keyword search failed: {e}")
        return [], 0

    def _fallback_sequential_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """SearchExecutionProcessor.fallback_sequential_search 래퍼"""
        if self.search_execution_processor:
            return self.search_execution_processor.fallback_sequential_search(state)
        self.logger.warning("SearchExecutionProcessor not available, cannot perform fallback sequential search")
        return state

    @with_state_optimization("evaluate_search_quality", enable_reduction=True)
    def evaluate_search_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """각 검색 결과의 품질 평가"""
        try:
            start_time = time.time()

            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            query = self._get_state_value(state, "query", "")
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            search_params = self._get_state_value(state, "search_params", {})

            # 의미적 검색 품질 평가
            semantic_quality = self._evaluate_semantic_search_quality(
                semantic_results=semantic_results,
                query=query,
                query_type=query_type_str,
                min_results=search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K) // 2
            )

            # 키워드 검색 품질 평가
            keyword_quality = self._evaluate_keyword_search_quality(
                keyword_results=keyword_results,
                query=query,
                query_type=query_type_str,
                min_results=search_params.get("keyword_limit", WorkflowConstants.CATEGORY_SEARCH_LIMIT) // 2
            )

            # 품질 평가 결과 저장
            overall_quality = (semantic_quality["score"] + keyword_quality["score"]) / 2.0
            quality_evaluation = {
                "semantic_quality": semantic_quality,
                "keyword_quality": keyword_quality,
                "overall_quality": overall_quality,
                "needs_retry": semantic_quality["needs_retry"] or keyword_quality["needs_retry"]
            }

            self._set_state_value(state, "search_quality_evaluation", quality_evaluation)
            # search_quality도 별도로 저장 (호환성)
            self._set_state_value(state, "search_quality", {
                "overall_quality": overall_quality,
                "relevance": semantic_quality.get("avg_relevance", 0.0),
                "coverage": keyword_quality.get("category_match", 0.0),
                "sufficiency": overall_quality
            })
            # metadata에도 저장
            self._save_metadata_safely(state, "search_quality", {
                "overall_quality": overall_quality,
                "relevance": semantic_quality.get("avg_relevance", 0.0),
                "coverage": keyword_quality.get("category_match", 0.0),
                "sufficiency": overall_quality
            })
            self._save_metadata_safely(state, "_last_executed_node", "evaluate_search_quality")
            self._update_processing_time(state, start_time)

            self.logger.debug(
                f"📊 [SEARCH QUALITY] Semantic: {semantic_quality['score']:.2f} "
                f"(needs_retry: {semantic_quality['needs_retry']}), "
                f"Keyword: {keyword_quality['score']:.2f} (needs_retry: {keyword_quality['needs_retry']})"
            )

        except Exception as e:
            self._handle_error(state, str(e), "검색 결과 품질 평가 중 오류 발생")
            # 폴백: 기본 품질 평가
            quality_evaluation = {
                "semantic_quality": {"score": 0.5, "needs_retry": False},
                "keyword_quality": {"score": 0.5, "needs_retry": False},
                "overall_quality": 0.5,
                "needs_retry": False
            }
            self._set_state_value(state, "search_quality_evaluation", quality_evaluation)

        return state

    def _evaluate_semantic_search_quality(
        self,
        semantic_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        min_results: int = 5
    ) -> Dict[str, Any]:
        """의미적 검색 품질 평가 (WorkflowValidator 사용)"""
        return self.workflow_validator.evaluate_semantic_search_quality(
            semantic_results=semantic_results,
            query=query,
            query_type=query_type,
            min_results=min_results
        )

    def _evaluate_keyword_search_quality(
        self,
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        min_results: int = 3
    ) -> Dict[str, Any]:
        """키워드 검색 품질 평가 (WorkflowValidator 사용)"""
        return self.workflow_validator.evaluate_keyword_search_quality(
            keyword_results=keyword_results,
            query=query,
            query_type=query_type,
            min_results=min_results
        )

    @with_state_optimization("conditional_retry_search", enable_reduction=True)
    def conditional_retry_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """품질이 낮은 검색 결과에 대한 조건부 재검색"""
        try:
            start_time = time.time()

            quality_evaluation = self._get_state_value(state, "search_quality_evaluation", {})
            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})

            semantic_quality = quality_evaluation.get("semantic_quality", {})
            keyword_quality = quality_evaluation.get("keyword_quality", {})

            semantic_needs_retry = semantic_quality.get("needs_retry", False)
            keyword_needs_retry = keyword_quality.get("needs_retry", False)

            # 재검색 카운터 확인 (무한 루프 방지)
            retry_metadata = self._get_state_value(state, "search_retry_metadata", {})
            semantic_retry_count = retry_metadata.get("semantic_retry_count", 0)
            keyword_retry_count = retry_metadata.get("keyword_retry_count", 0)

            max_retry_per_type = 1  # 각 검색 타입당 최대 1회 재검색

            # 의미적 검색 재검색
            if semantic_needs_retry and semantic_retry_count < max_retry_per_type:
                self.logger.debug(f"🔄 [RETRY SEMANTIC] Retrying semantic search (count: {semantic_retry_count})")

                # 쿼리 개선
                improved_semantic_query = self._improve_search_query_for_retry(
                    optimized_queries.get("semantic_query", ""),
                    {"failed_checks": semantic_quality.get("issues", [])},
                    state
                )

                # 의미적 검색 재실행
                retry_semantic, retry_count = self.search_handler.semantic_search(
                    improved_semantic_query,
                    k=search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K) + 5  # 더 많은 결과
                )

                semantic_results = retry_semantic
                semantic_retry_count += 1
                retry_metadata["semantic_retry_count"] = semantic_retry_count

            # 키워드 검색 재검색
            if keyword_needs_retry and keyword_retry_count < max_retry_per_type:
                self.logger.debug(f"🔄 [RETRY KEYWORD] Retrying keyword search (count: {keyword_retry_count})")

                query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
                legal_field = self._get_state_value(state, "legal_field", "")
                extracted_keywords = optimized_queries.get("expanded_keywords", [])

                # 키워드 검색 재실행
                retry_keyword, retry_count = self._keyword_search(
                    query=optimized_queries.get("keyword_queries", [""])[0],
                    query_type_str=query_type_str,
                    limit=search_params.get("keyword_limit", WorkflowConstants.CATEGORY_SEARCH_LIMIT) + 3,
                    legal_field=legal_field,
                    extracted_keywords=extracted_keywords
                )

                keyword_results = retry_keyword
                keyword_retry_count += 1
                retry_metadata["keyword_retry_count"] = keyword_retry_count

            # 재검색 결과 저장
            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "search_retry_metadata", retry_metadata)

            self._save_metadata_safely(state, "_last_executed_node", "conditional_retry_search")
            self._update_processing_time(state, start_time)

            self.logger.debug(
                f"✅ [CONDITIONAL RETRY] Semantic retry: {semantic_retry_count}, "
                f"Keyword retry: {keyword_retry_count}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "조건부 재검색 중 오류 발생")

        return state

    @with_state_optimization("merge_and_rerank_with_keyword_weights", enable_reduction=True)
    def merge_and_rerank_with_keyword_weights(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드별 가중치를 적용한 결과 병합 및 Reranking"""
        try:
            start_time = time.time()

            self.logger.debug("[DEBUG] MERGE: START - merge_and_rerank_with_keyword_weights")

            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            search_params = self._get_state_value(state, "search_params", {})
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            self.logger.debug(f"[DEBUG] MERGE: Input - semantic_results={len(semantic_results)}, keyword_results={len(keyword_results)}")

            # state에서 직접 확인 (디버깅)
            # _get_state_value가 제대로 작동하지 않는 경우를 대비하여 직접 state["search"]에서 읽기
            if len(semantic_results) == 0 and len(keyword_results) == 0:
                self.logger.debug("[DEBUG] MERGE: _get_state_value returned empty, checking state['search'] directly...")
                # state["search"]에서 직접 읽기 시도
                if "search" in state and isinstance(state.get("search"), dict):
                    direct_semantic = state["search"].get("semantic_results", [])
                    direct_keyword = state["search"].get("keyword_results", [])
                    self.logger.debug(f"[DEBUG] MERGE: Direct state['search'] check - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
                    if direct_semantic or direct_keyword:
                        self.logger.debug(f"[DEBUG] MERGE: Found results in state['search'] - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
                        semantic_results = direct_semantic
                        keyword_results = direct_keyword
                else:
                    self.logger.debug(f"[DEBUG] MERGE: state['search'] not found or not dict, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

            # 여전히 비어있으면 state 전체에서 찾기
            if len(semantic_results) == 0 and len(keyword_results) == 0:
                self.logger.debug("[DEBUG] MERGE: Still empty, checking all state keys...")
                if isinstance(state, dict):
                    # flat 구조일 수도 있으므로 직접 확인
                    if "semantic_results" in state:
                        flat_semantic = state.get("semantic_results", [])
                        if isinstance(flat_semantic, list) and len(flat_semantic) > 0:
                            self.logger.debug(f"[DEBUG] MERGE: Found semantic_results in flat state: {len(flat_semantic)}")
                            semantic_results = flat_semantic
                    if "keyword_results" in state:
                        flat_keyword = state.get("keyword_results", [])
                        if isinstance(flat_keyword, list) and len(flat_keyword) > 0:
                            self.logger.debug(f"[DEBUG] MERGE: Found keyword_results in flat state: {len(flat_keyword)}")
                            keyword_results = flat_keyword

            # 키워드 중요도 가중치 계산
            keyword_weights = self._calculate_keyword_weights(
                extracted_keywords=extracted_keywords,
                query=query,
                query_type=query_type_str,
                legal_field=legal_field
            )

            # 모든 검색 결과 수집 및 검색 타입 정보 보존
            all_results = []

            # 의미적 검색 결과에 검색 타입 정보 추가
            for doc in semantic_results:
                if not doc.get("search_type"):
                    doc["search_type"] = "semantic"
                    doc["search_method"] = "vector_search"
                all_results.append(doc)

            # 키워드 검색 결과에 검색 타입 정보 추가
            for doc in keyword_results:
                if not doc.get("search_type"):
                    doc["search_type"] = "keyword"
                    doc["search_method"] = "keyword_search"
                all_results.append(doc)

            # 중복 제거 (검색 타입 정보 보존)
            unique_results = self._remove_duplicate_results_for_merge(all_results)

            # 각 문서에 키워드 매칭 점수 계산 및 가중치 적용
            weighted_results = []
            for doc in unique_results:
                # 키워드 매칭 점수 계산
                keyword_scores = self._calculate_keyword_match_score(
                    document=doc,
                    keyword_weights=keyword_weights,
                    query=query
                )

                # 가중치 적용 최종 점수 계산 (강화: query_type 추가)
                final_score = self._calculate_weighted_final_score(
                    document=doc,
                    keyword_scores=keyword_scores,
                    search_params=search_params,
                    query_type=query_type_str
                )

                # 문서에 점수 정보 추가
                doc["keyword_match_score"] = keyword_scores["keyword_match_score"]
                doc["keyword_coverage"] = keyword_scores["keyword_coverage"]
                doc["matched_keywords"] = keyword_scores["matched_keywords"]
                doc["weighted_keyword_score"] = keyword_scores["weighted_keyword_score"]
                doc["final_weighted_score"] = final_score

                # 검색 타입 정보 유지 (없으면 기본값 설정)
                if not doc.get("search_type"):
                    doc["search_type"] = "hybrid"  # 중복 제거로 인한 병합 결과
                    doc["search_method"] = "hybrid_search"

                weighted_results.append(doc)

            # Reranking 수행
            reranked_results = self._rerank_with_keyword_weights(
                results=weighted_results,
                keyword_weights=keyword_weights,
                rerank_params=search_params.get("rerank", {})
            )

            # 검색 품질 검증 추가 (4단계)
            quality_valid, quality_message = self._validate_search_quality(
                results=reranked_results,
                query=query,
                query_type=query_type_str
            )

            if not quality_valid:
                self.logger.warning(f"⚠️ [SEARCH QUALITY] Validation failed: {quality_message}")
                self.logger.debug(f"[DEBUG] MERGE: Search quality validation failed - {quality_message}")
                # 품질 검증 실패 시 상위 점수 문서만 유지 (최소 5개)
                if reranked_results:
                    min_score = 0.4  # 최소 점수 기준 완화 (0.5 → 0.4)
                    filtered_reranked = [
                        doc for doc in reranked_results
                        if doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) >= min_score
                    ]
                    if len(filtered_reranked) >= 5:  # 최소 5개 보장
                        reranked_results = filtered_reranked[:10]  # 상위 10개만
                        self.logger.debug(f"🔧 [SEARCH QUALITY] Filtered to {len(reranked_results)} high-quality documents")
                        self.logger.debug(f"[DEBUG] MERGE: Filtered to {len(reranked_results)} high-quality documents")
                    elif len(filtered_reranked) >= 3:
                        reranked_results = filtered_reranked  # 최소 3개 이상이면 모두 유지
                        self.logger.warning(f"⚠️ [SEARCH QUALITY] Low quality results, keeping {len(reranked_results)} documents")
                        self.logger.debug(f"[DEBUG] MERGE: Low quality, keeping {len(reranked_results)} documents")
                    else:
                        # 최소 3개 미만이면 상위 5개만 유지 (점수 상관없이, 3개 → 5개로 증가)
                        reranked_results = reranked_results[:5]
                        self.logger.debug("[DEBUG] MERGE: Very low quality, keeping top 5 only")
            else:
                self.logger.debug(f"✅ [SEARCH QUALITY] Validation passed: {quality_message}")
                self.logger.debug(f"[DEBUG] MERGE: Search quality validation passed - {quality_message}")

            # 결과 저장
            self._set_state_value(state, "merged_documents", reranked_results)
            self._set_state_value(state, "keyword_weights", keyword_weights)

            # 중요: 병합된 결과를 retrieved_docs에도 저장 (다음 노드에서 사용하기 위해)
            # 모든 벡터 스토어 검색 결과(semantic_query, original_query, keyword_queries)가 포함됨
            self._set_state_value(state, "retrieved_docs", reranked_results)
            self.logger.debug(f"[DEBUG] MERGE: Saved {len(reranked_results)} documents to retrieved_docs")

            # 저장 확인
            stored_merged = self._get_state_value(state, "merged_documents", [])
            stored_retrieved = self._get_state_value(state, "retrieved_docs", [])
            self.logger.debug(f"[DEBUG] MERGE: After save - merged_documents={len(stored_merged)}, retrieved_docs={len(stored_retrieved)}")

            self._save_metadata_safely(state, "_last_executed_node", "merge_and_rerank_with_keyword_weights")
            self._update_processing_time(state, start_time)

            self.logger.debug(
                f"✅ [KEYWORD-WEIGHTED RERANKING] Merged {len(unique_results)} results, "
                f"reranked to {len(reranked_results)} with {len(keyword_weights)} weighted keywords"
            )
            self.logger.debug(f"[DEBUG] MERGE: Semantic input={len(semantic_results)}, Keyword input={len(keyword_results)}, Unique={len(unique_results)}, Reranked={len(reranked_results)}")

            # 병합 결과 상세 디버깅 로그
            if reranked_results:
                top_scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in reranked_results[:5]]
                search_type_counts = {
                    "semantic": sum(1 for doc in reranked_results if doc.get("search_type") == "semantic"),
                    "keyword": sum(1 for doc in reranked_results if doc.get("search_type") == "keyword"),
                    "hybrid": sum(1 for doc in reranked_results if doc.get("search_type") not in ["semantic", "keyword"])
                }
                self.logger.debug(
                    f"🔍 [DEBUG] Merge & Rerank details: "
                    f"Total merged: {len(unique_results)}, "
                    f"After rerank: {len(reranked_results)}, "
                    f"Top scores: {top_scores}, "
                    f"Search types: {search_type_counts}"
                )
            else:
                self.logger.warning(
                    f"⚠️ [DEBUG] Merge & Rerank resulted in 0 documents. "
                    f"Input: semantic={len(semantic_results)}, keyword={len(keyword_results)}, "
                    f"unique={len(unique_results)}, weighted={len(weighted_results)}"
                )

        except Exception as e:
            self._handle_error(state, str(e), "키워드 가중치 기반 병합 및 Reranking 중 오류 발생")
            # 폴백: 간단한 병합
            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            all_results = semantic_results + keyword_results
            fallback_docs = all_results[:20]
            self._set_state_value(state, "merged_documents", fallback_docs)
            # 폴백 결과도 retrieved_docs에 저장
            self._set_state_value(state, "retrieved_docs", fallback_docs)
            self.logger.debug(f"[DEBUG] MERGE: Fallback - Saved {len(fallback_docs)} documents to retrieved_docs")

        return state

    def _remove_duplicate_results_for_merge(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """병합을 위한 중복 제거"""
        seen_texts = set()
        unique_results = []

        for doc in results:
            doc_content = doc.get("content", "")
            if not doc_content:
                continue

            content_preview = doc_content[:100]
            content_hash = hash(content_preview)

            if content_hash not in seen_texts:
                seen_texts.add(content_hash)
                unique_results.append(doc)

        return unique_results

    def _calculate_keyword_weights(
        self,
        extracted_keywords: List[str],
        query: str,
        query_type: str,
        legal_field: str
    ) -> Dict[str, float]:
        """키워드별 중요도 가중치 계산 (SearchResultProcessor 사용)"""
        return self.search_result_processor.calculate_keyword_weights(
            extracted_keywords=extracted_keywords,
            query=query,
            query_type=query_type,
            legal_field=legal_field
        )

    def _calculate_keyword_match_score(
        self,
        document: Dict[str, Any],
        keyword_weights: Dict[str, float],
        query: str
    ) -> Dict[str, float]:
        """문서에 대한 키워드 매칭 점수 계산 (SearchResultProcessor 사용)"""
        return self.search_result_processor.calculate_keyword_match_score(
            document=document,
            keyword_weights=keyword_weights,
            query=query
        )

    def _calculate_weighted_final_score(
        self,
        document: Dict[str, Any],
        keyword_scores: Dict[str, float],
        search_params: Dict[str, Any],
        query_type: Optional[str] = None
    ) -> float:
        """가중치를 적용한 최종 점수 계산 (SearchResultProcessor 사용)"""
        return self.search_result_processor.calculate_weighted_final_score(
            document=document,
            keyword_scores=keyword_scores,
            search_params=search_params,
            query_type=query_type
        )
    
    def _calculate_dynamic_weights(
        self,
        query_type: str = "",
        search_quality: float = 0.7,
        document_count: int = 10
    ) -> Dict[str, float]:
        """동적 가중치 계산 (SearchResultProcessor 사용)"""
        return self.search_result_processor.calculate_dynamic_weights(
            query_type=query_type,
            search_quality=search_quality,
            document_count=document_count
        )

    def _validate_search_quality(
        self,
        results: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> Tuple[bool, str]:
        """
        검색 결과 품질 검증 (quality_validators 모듈 사용)

        Args:
            results: 검색 결과 리스트
            query: 원본 쿼리
            query_type: 질문 유형

        Returns:
            (is_valid, message): 검증 통과 여부와 메시지
        """
        validation_result = SearchValidator.validate_search_quality(
            search_results=results,
            query=query,
            query_type=query_type
        )

        is_valid = validation_result.get("is_valid", False)
        quality_score = validation_result.get("quality_score", 0.0)
        avg_relevance = validation_result.get("avg_relevance", 0.0)
        issues = validation_result.get("issues", [])

        if is_valid:
            message = f"검색 품질 양호 (평균 점수: {avg_relevance:.2f}, 품질 점수: {quality_score:.2f})"
        else:
            message = "; ".join(issues) if issues else f"검색 품질 부족 (품질 점수: {quality_score:.2f})"

        return is_valid, message

    def _apply_statute_article_boosting(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """법령 조문 우선순위 부스팅 (우선순위 4)"""
        import re
        
        # 질문에서 법령명과 조문번호 추출
        law_pattern = re.compile(r'([가-힣]+법)\s*제\s*(\d+)\s*조')
        match = law_pattern.search(query)
        
        if match:
            query_law_name = match.group(1)
            query_article_no = match.group(2)
            
            for doc in documents:
                if doc.get("type") == "statute_article":
                    # 법령 조문 타입: 기본 부스팅
                    score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc["final_weighted_score"] = min(1.0, score * 1.3)  # 30% 부스팅
                    
                    # 질문의 법령명/조문번호와 일치: 추가 부스팅
                    doc_law_name = doc.get("statute_name") or doc.get("law_name", "")
                    doc_article_no = doc.get("article_no", "")
                    
                    if query_law_name in doc_law_name and query_article_no == doc_article_no:
                        doc["final_weighted_score"] = 1.0  # 최고 점수
                        doc["direct_match"] = True
                        doc["relevance_score"] = 1.0
                        self.logger.debug(
                            f"✅ [STATUTE BOOSTING] Direct match: {query_law_name} 제{query_article_no}조 "
                            f"(score: {doc['final_weighted_score']:.3f})"
                        )
        
        return documents

    def _rerank_with_keyword_weights(
        self,
        results: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        rerank_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """키워드 가중치를 적용한 Reranking (SearchResultProcessor 사용)"""
        return self.search_result_processor.rerank_with_keyword_weights(
            results=results,
            keyword_weights=keyword_weights,
            rerank_params=rerank_params,
            result_ranker=self.result_ranker
        )

    @with_state_optimization("filter_and_validate_results", enable_reduction=True)
    def filter_and_validate_results(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 필터링 및 품질 검증"""
        try:
            start_time = time.time()

            documents = self._get_state_value(state, "merged_documents", [])

            # merged_documents가 비어있으면 retrieved_docs에서 가져오기 (fallback)
            if not documents or len(documents) == 0:
                documents = self._get_state_value(state, "retrieved_docs", [])
                if documents:
                    self.logger.debug(f"[DEBUG] FILTER: merged_documents is empty, using retrieved_docs ({len(documents)} documents)")

            search_params = self._get_state_value(state, "search_params", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            # 메타데이터 필터 적용
            documents = self.search_handler.apply_metadata_filters(
                documents,
                query_type_str,
                legal_field
            )

            # 결과 품질 검증 및 필터링
            filtered_docs = self.search_handler.filter_low_quality_results(
                documents,
                min_relevance=search_params.get("min_relevance", self.config.similarity_threshold),
                max_diversity=search_params.get("max_results", WorkflowConstants.MAX_DOCUMENTS)
            )

            # Pruning 및 최종 정리
            pruned_docs = prune_retrieved_docs(
                filtered_docs[:WorkflowConstants.MAX_DOCUMENTS],
                max_items=MAX_RETRIEVED_DOCS,
                max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
            )

            self._set_state_value(state, "retrieved_docs", pruned_docs)
            self._save_metadata_safely(state, "_last_executed_node", "filter_and_validate_results")
            self._update_processing_time(state, start_time)

            # 상세 로깅: 검색 결과 요약
            self.logger.debug(
                f"✅ [FILTER & VALIDATE] Results filtered and validated: "
                f"{len(pruned_docs)} final documents "
                f"(from {len(documents)} input documents)"
            )

            # 필터링 상세 디버깅 로그
            min_relevance = search_params.get("min_relevance", self.config.similarity_threshold)
            self.logger.debug(
                f"🔍 [DEBUG] Filter & Validate details: "
                f"Input documents: {len(documents)}, "
                f"After metadata filter: {len(documents)}, "
                f"After quality filter: {len(filtered_docs)}, "
                f"After pruning: {len(pruned_docs)}, "
                f"Min relevance threshold: {min_relevance:.3f}"
            )
            self.logger.debug(f"[DEBUG] FILTER: Input={len(documents)}, After quality={len(filtered_docs)}, After prune={len(pruned_docs)}, Min relevance={min_relevance:.3f}")

            if documents:
                # 점수 범위 분석
                all_scores = []
                for doc in documents:
                    score = doc.get("combined_score") or doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)
                    all_scores.append(score)

                if all_scores:
                    min_score = min(all_scores)
                    max_score = max(all_scores)
                    avg_score = sum(all_scores) / len(all_scores)
                    below_threshold = sum(1 for s in all_scores if s < min_relevance)

                    self.logger.debug(
                        f"🔍 [DEBUG] Score statistics: "
                        f"min={min_score:.3f}, max={max_score:.3f}, avg={avg_score:.3f}, "
                        f"below threshold ({min_relevance:.3f}): {below_threshold}/{len(all_scores)}"
                    )

                    # 필터링으로 제거된 문서 수
                    removed_by_score = len(documents) - len(filtered_docs)
                    if removed_by_score > 0:
                        self.logger.warning(
                            f"⚠️ [DEBUG] {removed_by_score} documents removed by relevance score filter "
                            f"(score < {min_relevance:.3f})"
                        )

            # 문서 샘플 로깅 (상위 3개)
            if pruned_docs:
                self.logger.debug("📄 [FILTER & VALIDATE] Document samples:")
                for i, doc in enumerate(pruned_docs[:3], 1):
                    source = doc.get("source", "Unknown")
                    content = doc.get("content") or doc.get("text", "")
                    content_length = len(content) if content else 0
                    score = doc.get("relevance_score", doc.get("combined_score", doc.get("final_weighted_score", doc.get("score", 0.0))))
                    search_type = doc.get("search_type", "unknown")
                    self.logger.debug(
                        f"  [{i}] {source} | score={score:.3f} | "
                        f"type={search_type} | "
                        f"content={content_length}chars | "
                        f"preview={content[:50] if content else 'N/A'}..."
                    )
            else:
                self.logger.warning(
                    f"⚠️ [FILTER & VALIDATE] No documents after filtering and validation! "
                    f"Input: {len(documents)}, After metadata filter: {len(documents)}, "
                    f"After quality filter: {len(filtered_docs)}, After pruning: {len(pruned_docs)}"
                )

        except Exception as e:
            self._handle_error(state, str(e), "결과 필터링 및 검증 중 오류 발생")
            self._fallback_search(state)

        return state

    @with_state_optimization("update_search_metadata", enable_reduction=True)
    def update_search_metadata(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 메타데이터 업데이트"""
        try:
            start_time = time.time()

            semantic_count = self._get_state_value(state, "semantic_count", 0)
            keyword_count = self._get_state_value(state, "keyword_count", 0)
            filtered_docs = self._get_state_value(state, "retrieved_docs", [])
            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            search_start_time = self._get_state_value(state, "search_start_time", time.time())

            # 검색 메타데이터 업데이트
            self.search_handler.update_search_metadata(
                state,
                semantic_count,
                keyword_count,
                filtered_docs,
                query_type_str,
                search_start_time,
                optimized_queries
            )

            # 캐시 저장
            if optimized_queries:
                self.performance_optimizer.cache.cache_documents(
                    optimized_queries.get("semantic_query", ""),
                    query_type_str,
                    filtered_docs
                )

            self._save_metadata_safely(state, "_last_executed_node", "update_search_metadata")
            self._update_processing_time(state, start_time)

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            self.logger.debug(f"Search metadata updated: {len(retrieved_docs)} documents retrieved")

        except Exception as e:
            self._handle_error(state, str(e), "검색 메타데이터 업데이트 중 오류 발생")

        return state

    @with_state_optimization("prepare_document_context_for_prompt", enable_reduction=True)
    def prepare_document_context_for_prompt(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """프롬프트에 최대한 반영되도록 문서 컨텍스트 준비"""
        try:
            start_time = time.time()

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            # retrieved_docs 검증
            if not retrieved_docs:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] No retrieved_docs to prepare for prompt. "
                    f"Query: '{query[:50]}...', Query type: {query_type_str}"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            # retrieved_docs 타입 검증
            if not isinstance(retrieved_docs, list):
                self.logger.error(
                    f"⚠️ [PREPARE CONTEXT] retrieved_docs is not a list: {type(retrieved_docs).__name__}"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            # 문서 내용 검증
            valid_docs_count = 0
            docs_without_content = 0
            total_content_length = 0

            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    docs_without_content += 1
                    continue

                # content 필드 확인 (여러 가능한 필드명 지원)
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if content and len(content.strip()) >= 10:  # 최소 10자 이상
                    valid_docs_count += 1
                    total_content_length += len(content)
                else:
                    docs_without_content += 1
                    source = doc.get("source", "Unknown")
                    self.logger.debug(
                        f"[PREPARE CONTEXT] Document filtered: content missing or too short "
                        f"(source: {source}, content_length: {len(content) if content else 0})"
                    )

            # 검증 결과 로깅
            if docs_without_content > 0:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] Found {docs_without_content} documents without valid content "
                    f"out of {len(retrieved_docs)} total documents. "
                    f"Valid docs: {valid_docs_count}, Total content: {total_content_length} chars"
                )

            # 유효한 문서가 없으면 에러
            if valid_docs_count == 0:
                self.logger.error(
                    f"❌ [PREPARE CONTEXT] No valid documents with content found! "
                    f"Total docs: {len(retrieved_docs)}, "
                    f"Query: '{query[:50]}...'"
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
                return state

            self.logger.debug(
                f"✅ [PREPARE CONTEXT] Preparing context from {valid_docs_count} valid documents "
                f"(total: {len(retrieved_docs)}, content: {total_content_length} chars)"
            )

            # 프롬프트 최적화 컨텍스트 구축
            prompt_optimized_context = self._build_prompt_optimized_context(
                retrieved_docs=retrieved_docs,
                query=query,
                extracted_keywords=extracted_keywords,
                query_type=query_type_str,
                legal_field=legal_field
            )

            # State에 저장
            self._set_state_value(state, "prompt_optimized_context", prompt_optimized_context)

            self._save_metadata_safely(state, "_last_executed_node", "prepare_document_context_for_prompt")
            self._update_processing_time(state, start_time)

            # 상세 로깅: 프롬프트 컨텍스트 준비 결과
            doc_count = prompt_optimized_context.get("document_count", 0)
            context_length = prompt_optimized_context.get("total_context_length", 0)
            content_validation = prompt_optimized_context.get("content_validation", {})

            self.logger.debug(
                f"✅ [DOCUMENT PREPARATION] Prepared prompt context: "
                f"{doc_count} documents, "
                f"{context_length} chars, "
                f"input docs: {len(retrieved_docs)}"
            )

            if content_validation:
                has_content = content_validation.get("has_document_content", False)
                docs_with_content = content_validation.get("documents_with_content", 0)
                self.logger.debug(
                    f"📊 [DOCUMENT PREPARATION] Content validation: "
                    f"has_content={has_content}, "
                    f"docs_with_content={docs_with_content}"
                )

            # 프롬프트 텍스트 샘플 로깅 (처음 200자)
            prompt_text = prompt_optimized_context.get("prompt_optimized_text", "")
            if prompt_text:
                self.logger.debug(
                    f"📝 [DOCUMENT PREPARATION] Prompt text preview (first 200 chars):\n"
                    f"{prompt_text[:200]}..."
                )
            else:
                self.logger.warning(
                    "⚠️ [DOCUMENT PREPARATION] prompt_optimized_text is empty!"
                )

        except Exception as e:
            self._handle_error(state, str(e), "문서 컨텍스트 준비 중 오류 발생")
            # 폴백: 빈 컨텍스트
            self._set_state_value(state, "prompt_optimized_context", {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            })

        return state

    @with_state_optimization("prepare_documents_and_terms", enable_reduction=True)
    def prepare_documents_and_terms(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 문서 준비 및 용어 처리 (prepare_document_context_for_prompt + process_legal_terms)"""
        try:
            overall_start_time = time.time()

            # ========== Part 1: prepare_document_context_for_prompt 로직 ==========
            context_start_time = time.time()

            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")
            
            # 질의와 검색된 문서의 relevance_score 로깅 (prepare_documents_and_terms 진입점)
            self.logger.debug(f"📊 [PREPARE DOCS ENTRY] 질의: '{query}'")
            self.logger.debug(f"📊 [PREPARE DOCS ENTRY] retrieved_docs 수: {len(retrieved_docs)}개")
            
            # retrieved_docs 검증 (전역 캐시에서 복원 시도)
            if not retrieved_docs:
                # 전역 캐시에서 복원 시도
                try:
                    from core.shared.wrappers.node_wrappers import _global_search_results_cache
                    if _global_search_results_cache and "retrieved_docs" in _global_search_results_cache:
                        retrieved_docs = _global_search_results_cache.get("retrieved_docs", [])
                        if retrieved_docs:
                            self.logger.debug(f"✅ [PREPARE CONTEXT] Restored {len(retrieved_docs)} retrieved_docs from global cache")
                except (ImportError, AttributeError):
                    pass
            
            has_valid_docs = False
            if not retrieved_docs:
                self.logger.debug(
                    f"ℹ️ [PREPARE CONTEXT] No retrieved_docs to prepare for prompt. "
                    f"Query: '{query[:50]}...', Query type: {query_type_str}. "
                    f"Skipping document context preparation and term extraction."
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
            elif not isinstance(retrieved_docs, list):
                self.logger.error(
                    f"⚠️ [PREPARE CONTEXT] retrieved_docs is not a list: {type(retrieved_docs).__name__}. "
                    f"Skipping document context preparation and term extraction."
                )
                self._set_state_value(state, "prompt_optimized_context", {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                })
            else:
                # 문서 내용 검증
                valid_docs_count = 0
                docs_without_content = 0
                total_content_length = 0

                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        docs_without_content += 1
                        continue

                    content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                    if content and len(content.strip()) >= 10:
                        valid_docs_count += 1
                        total_content_length += len(content)
                    else:
                        docs_without_content += 1

                if docs_without_content > 0:
                    self.logger.warning(
                        f"⚠️ [PREPARE CONTEXT] Found {docs_without_content} documents without valid content "
                        f"out of {len(retrieved_docs)} total documents. "
                        f"Valid docs: {valid_docs_count}, Total content: {total_content_length} chars"
                    )

                if valid_docs_count == 0:
                    self.logger.error(
                        f"❌ [PREPARE CONTEXT] No valid documents with content found! "
                        f"Total docs: {len(retrieved_docs)}, "
                        f"Query: '{query[:50]}...'"
                    )
                    self._set_state_value(state, "prompt_optimized_context", {
                        "prompt_optimized_text": "",
                        "structured_documents": {},
                        "document_count": 0,
                        "total_context_length": 0
                    })
                else:
                    self.logger.debug(
                        f"✅ [PREPARE CONTEXT] Preparing context from {valid_docs_count} valid documents "
                        f"(total: {len(retrieved_docs)}, content: {total_content_length} chars)"
                    )

                    # 프롬프트 최적화 컨텍스트 구축
                    prompt_optimized_context = self._build_prompt_optimized_context(
                        retrieved_docs=retrieved_docs,
                        query=query,
                        extracted_keywords=extracted_keywords,
                        query_type=query_type_str,
                        legal_field=legal_field
                    )

                    # State에 저장
                    self._set_state_value(state, "prompt_optimized_context", prompt_optimized_context)

                    # 상세 로깅
                    doc_count = prompt_optimized_context.get("document_count", 0)
                    context_length = prompt_optimized_context.get("total_context_length", 0)
                    self.logger.debug(
                        f"✅ [DOCUMENT PREPARATION] Prepared prompt context: "
                        f"{doc_count} documents, "
                        f"{context_length} chars, "
                        f"input docs: {len(retrieved_docs)}"
                    )
                    has_valid_docs = True

            self._save_metadata_safely(state, "_last_executed_node", "prepare_documents_and_terms")
            self._update_processing_time(state, context_start_time)
            self._add_step(state, "문서 컨텍스트 준비", "프롬프트 최적화 컨텍스트 준비 완료")

            # ========== Part 2: process_legal_terms 로직 ==========
            # 개선: retrieved_docs가 없거나 유효하지 않으면 용어 처리 생략
            if has_valid_docs:
                terms_start_time = time.time()

                # 메타데이터 설정 (process_legal_terms 로직)
                if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                    state["metadata"] = {}
                state["metadata"] = dict(state["metadata"])
                state["metadata"]["_last_executed_node"] = "prepare_documents_and_terms"

                if "common" not in state or not isinstance(state.get("common"), dict):
                    state["common"] = {}
                if "metadata" not in state["common"]:
                    state["common"]["metadata"] = {}
                state["common"]["metadata"]["_last_executed_node"] = "prepare_documents_and_terms"

                # 기존 재시도 카운터 보존
                existing_metadata_direct = state.get("metadata", {})
                existing_metadata_common = state.get("common", {}).get("metadata", {}) if isinstance(state.get("common"), dict) else {}
                existing_metadata = existing_metadata_direct if isinstance(existing_metadata_direct, dict) else {}
                if isinstance(existing_metadata_common, dict):
                    existing_metadata = {**existing_metadata, **existing_metadata_common}

                existing_top_level = state.get("retry_count", 0)
                saved_gen_retry = max(
                    existing_metadata.get("generation_retry_count", 0),
                    existing_top_level
                )
                saved_val_retry = existing_metadata.get("validation_retry_count", 0)

                # 법률 용어 추출 및 통합
                retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
                all_terms = self._extract_terms_from_documents(retrieved_docs)
                self.logger.debug(f"추출된 용어 수: {len(all_terms)}")

                if all_terms:
                    representative_terms = self._integrate_and_process_terms(all_terms)
                    metadata = dict(existing_metadata)
                    metadata["extracted_terms"] = representative_terms
                    metadata["total_terms_extracted"] = len(all_terms)
                    metadata["unique_terms"] = len(representative_terms)
                    # 재시도 카운터 보존
                    metadata["generation_retry_count"] = saved_gen_retry
                    metadata["validation_retry_count"] = saved_val_retry
                    metadata["_last_executed_node"] = "prepare_documents_and_terms"
                    state["metadata"] = metadata

                    # common 그룹에도 동기화
                    if "common" not in state:
                        state["common"] = {}
                    if "metadata" not in state["common"]:
                        state["common"]["metadata"] = {}
                    state["common"]["metadata"].update(metadata)
                    state["retry_count"] = saved_gen_retry

                    self._set_state_value(state, "metadata", metadata)
                    self._add_step(state, "용어 통합 완료", f"용어 통합 완료: {len(representative_terms)}개")
                    self.logger.debug(f"통합된 용어 수: {len(representative_terms)}")
                else:
                    metadata = dict(existing_metadata)
                    metadata["extracted_terms"] = []
                    # 재시도 카운터 보존
                    metadata["generation_retry_count"] = saved_gen_retry
                    metadata["validation_retry_count"] = saved_val_retry
                    metadata["_last_executed_node"] = "prepare_documents_and_terms"
                    state["metadata"] = metadata

                    # common 그룹에도 동기화
                    if "common" not in state:
                        state["common"] = {}
                    if "metadata" not in state["common"]:
                        state["common"]["metadata"] = {}
                    state["common"]["metadata"].update(metadata)
                    state["retry_count"] = saved_gen_retry

                    self._set_state_value(state, "metadata", metadata)
                    self._add_step(state, "용어 추출 없음", "용어 추출 없음 (문서 내용 부족)")

                self._update_processing_time(state, terms_start_time)
            else:  # if has_valid_docs의 else
                # retrieved_docs가 없거나 유효하지 않을 때: 용어 처리는 생략하되 메타데이터만 설정
                self.logger.debug(
                    "⏭️ [TERM PROCESSING] Skipping term extraction and processing "
                    "(no valid retrieved_docs available)"
                )

                # 메타데이터 기본 설정 (재시도 카운터 보존)
                if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                    state["metadata"] = {}
                existing_metadata = dict(state["metadata"])

                existing_top_level = state.get("retry_count", 0)
                saved_gen_retry = max(
                    existing_metadata.get("generation_retry_count", 0),
                    existing_top_level
                )
                saved_val_retry = existing_metadata.get("validation_retry_count", 0)

                metadata = dict(existing_metadata)
                metadata["extracted_terms"] = []
                metadata["total_terms_extracted"] = 0
                metadata["unique_terms"] = 0
                metadata["generation_retry_count"] = saved_gen_retry
                metadata["validation_retry_count"] = saved_val_retry
                metadata["_last_executed_node"] = "prepare_documents_and_terms"
                state["metadata"] = metadata

                # common 그룹에도 동기화
                if "common" not in state:
                    state["common"] = {}
                if "metadata" not in state["common"]:
                    state["common"]["metadata"] = {}
                state["common"]["metadata"].update(metadata)
                state["retry_count"] = saved_gen_retry

                self._set_state_value(state, "metadata", metadata)
                self._add_step(state, "용어 처리 생략", "문서 없음으로 인한 용어 처리 생략")

        except Exception as e:
            self._handle_error(state, str(e), "문서 준비 및 용어 처리 중 오류 발생")
            # 폴백: 빈 컨텍스트 및 빈 용어
            self._set_state_value(state, "prompt_optimized_context", {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            })
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["extracted_terms"] = []
            self._set_state_value(state, "metadata", metadata)

        self._update_processing_time(state, overall_start_time)
        return state

    def _build_prompt_optimized_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str,
        legal_field: str
    ) -> Dict[str, Any]:
        """프롬프트에 최대한 반영되도록 최적화된 컨텍스트 구축 (WorkflowDocumentProcessor 사용)"""
        return self.workflow_document_processor.build_prompt_optimized_context(
            retrieved_docs=retrieved_docs,
                query=query,
            extracted_keywords=extracted_keywords,
            query_type=query_type,
            legal_field=legal_field,
            select_balanced_documents_func=self._select_balanced_documents,
            extract_query_relevant_sentences_func=self._extract_query_relevant_sentences,
            generate_document_based_instructions_func=self._generate_document_based_instructions
        )

    def _extract_query_relevant_sentences(
        self,
        doc_content: str,
        query: str,
        extracted_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """QueryEnhancer.extract_query_relevant_sentences 래퍼"""
        return self.query_enhancer.extract_query_relevant_sentences(doc_content, query, extracted_keywords)

    def _generate_document_based_instructions(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> str:
        """문서를 기반으로 답변 생성하라는 명시적 지시사항 생성 (WorkflowDocumentProcessor 사용)"""
        return self.workflow_document_processor.generate_document_based_instructions(
            documents=documents,
            query=query,
            query_type=query_type
        )

    def _select_balanced_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        max_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """의미적 검색과 키워드 검색 결과의 균형을 맞춰서 문서 선택 (WorkflowDocumentProcessor 사용)"""
        return self.workflow_document_processor.select_balanced_documents(sorted_docs, max_docs)

    def _validate_context_dict_content(self, context_dict: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]):
        """context_dict에 실제 문서 내용이 포함되었는지 검증"""
        if not isinstance(context_dict, dict):
            self.logger.warning(f"⚠️ [CONTEXT VALIDATION] context_dict is not a dict (type: {type(context_dict)})")
            return
        context_text = context_dict.get("context", "")
        if retrieved_docs and len(retrieved_docs) > 0:
            if not context_text or len(context_text.strip()) < 100:
                self.logger.error(
                    f"⚠️ [CONTEXT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                    f"but context_dict['context'] is empty or too short ({len(context_text)} chars). "
                    f"This may cause LLM to generate answer without document references!"
                )
            else:
                has_doc_reference = False
                for doc in retrieved_docs[:3]:
                    source = doc.get("source", "")
                    content_preview = (doc.get("content") or doc.get("text", ""))[:50]
                    if source and source in context_text:
                        has_doc_reference = True
                        break
                    if content_preview and content_preview in context_text:
                        has_doc_reference = True
                        break

                if not has_doc_reference:
                    self.logger.warning(
                        f"⚠️ [CONTEXT VALIDATION] context_text does not seem to contain references to retrieved_docs. "
                        f"Length: {len(context_text)} chars"
                    )

    def _extract_legal_references_from_docs(self, documents: List[Dict[str, Any]]) -> List[str]:
        """문서에서 법률 참조 정보 추출"""
        return DocumentExtractor.extract_legal_references_from_docs(documents)
    
    def _validate_and_generate_prompt_context(
        self,
        state: LegalWorkflowState,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str
    ) -> Dict[str, Any]:
        """prompt_optimized_context 검증 및 생성"""
        prompt_optimized_context = self._get_state_value(state, "prompt_optimized_context", {})
        
        if not isinstance(prompt_optimized_context, dict):
            if isinstance(prompt_optimized_context, str):
                self.logger.warning("⚠️ [PROMPT CONTEXT] prompt_optimized_context is str, converting to dict")
                prompt_optimized_context = {"prompt_optimized_text": prompt_optimized_context}
            else:
                self.logger.warning(f"⚠️ [PROMPT CONTEXT] prompt_optimized_context is not dict (type: {type(prompt_optimized_context)}), using empty dict")
                prompt_optimized_context = {}
        
        if not prompt_optimized_context or not prompt_optimized_context.get("prompt_optimized_text"):
            if retrieved_docs and len(retrieved_docs) > 0:
                self.logger.debug("⚠️ [FALLBACK] prompt_optimized_context is missing or invalid, generating automatically")
                try:
                    legal_field = self._get_state_value(state, "legal_field", "")
                    prompt_optimized_context = self._build_prompt_optimized_context(
                        retrieved_docs=retrieved_docs,
                        query=query,
                        extracted_keywords=extracted_keywords,
                        query_type=query_type,
                        legal_field=legal_field
                    )
                    self._set_state_value(state, "prompt_optimized_context", prompt_optimized_context)
                    self.logger.debug(f"✅ [AUTO GENERATE] Generated prompt_optimized_context: {prompt_optimized_context.get('document_count', 0)} docs, {prompt_optimized_context.get('total_context_length', 0)} chars")
                except Exception as e:
                    self.logger.warning(f"⚠️ [FALLBACK] Failed to generate prompt_optimized_context: {e}, using _build_intelligent_context as fallback")
                    prompt_optimized_context = {}
        
        return prompt_optimized_context
    
    def _build_and_validate_context_dict(
        self,
        state: LegalWorkflowState,
        query_type: str,
        retrieved_docs: List[Dict[str, Any]],
        prompt_optimized_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """context_dict 구축 및 검증"""
        context_dict = self._build_context_dict(state, query_type, retrieved_docs, prompt_optimized_context)
        
        if not isinstance(context_dict, dict):
            self.logger.warning(f"⚠️ [CONTEXT DICT] context_dict is not a dict (type: {type(context_dict)}), converting to dict")
            if isinstance(context_dict, str):
                context_dict = {"context": context_dict}
            elif context_dict is None:
                self.logger.warning("⚠️ [CONTEXT DICT] context_dict is None, using empty dict")
                context_dict = {}
            else:
                self.logger.warning(f"⚠️ [CONTEXT DICT] context_dict is unexpected type {type(context_dict)}, using empty dict")
                context_dict = {}

        self._validate_context_dict_content(context_dict, retrieved_docs)
        
        return context_dict
    
    def _validate_context_quality_and_expand(
        self,
        state: LegalWorkflowState,
        context_dict: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        retrieved_docs: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """컨텍스트 품질 검증 및 확장"""
        try:
            meta_forced = self._get_metadata_safely(state)
            if meta_forced.get("force_rag_fallback"):
                self.logger.debug("[ROUTER FALLBACK] SQL 0건 → 키워드+벡터 검색으로 컨텍스트 보강 재시도")
                state = self._adaptive_context_expansion(state, {"reason": "router_zero_rows", "needs_expansion": True})
                expanded_context = self.context_builder.build_intelligent_context(state)
                if isinstance(expanded_context, dict):
                    context_dict = expanded_context
                elif isinstance(expanded_context, str):
                    context_dict = {"context": expanded_context}
                else:
                    self.logger.warning(f"⚠️ [CONTEXT DICT] Expanded context is not dict or str (type: {type(expanded_context)})")
        except Exception as e:
            self.logger.warning(f"Router fallback expansion skipped due to error: {e}")

        if not isinstance(context_dict, dict):
            if isinstance(context_dict, str):
                self.logger.warning("⚠️ [VALIDATE CONTEXT] context_dict is str before validation, converting to dict")
                context_dict = {"context": context_dict}
            else:
                self.logger.warning(f"⚠️ [VALIDATE CONTEXT] context_dict is not dict (type: {type(context_dict)}), using empty dict")
                context_dict = {}
        
        # 성능 최적화: retrieved_docs가 있고 overall_score가 이미 충분히 높으면 검증 스킵
        metadata_before = self._get_metadata_safely(state)
        cached_overall_score = metadata_before.get("context_validation", {}).get("overall_score") if isinstance(metadata_before.get("context_validation"), dict) else None
        
        # 성능 최적화: 캐시된 overall_score가 0.5 이상이면 검증 스킵
        if cached_overall_score is not None and cached_overall_score >= 0.5:
            self.logger.debug(f"[PERFORMANCE] Skipping context validation: cached overall_score={cached_overall_score} >= 0.5")
            validation_results = metadata_before.get("context_validation", {"overall_score": cached_overall_score})
            overall_score = cached_overall_score
            retrieved_docs = self._monitor_search_quality(validation_results, overall_score, state, retrieved_docs)
        else:
            validation_results = self._validate_context_quality(
                context_dict, query, query_type, extracted_keywords, state
            )
            
            if not isinstance(validation_results, dict):
                self.logger.error(f"❌ [VALIDATION] validation_results is not dict (type: {type(validation_results)}), using empty dict")
                if isinstance(validation_results, str):
                    validation_results = {"error": validation_results, "overall_score": 0.0}
                else:
                    validation_results = {"overall_score": 0.0}

            overall_score = validation_results.get("overall_score", 0.0) if isinstance(validation_results, dict) else 0.0
            retrieved_docs = self._monitor_search_quality(validation_results, overall_score, state, retrieved_docs)

        # 성능 최적화: 컨텍스트 확장은 overall_score가 매우 낮을 때만 수행 (0.2 미만)
        needs_expansion = validation_results.get("needs_expansion", False) if isinstance(validation_results, dict) else False
        overall_score = validation_results.get("overall_score", 0.0) if isinstance(validation_results, dict) else 0.0
        
        # 성능 최적화: overall_score가 0.2 이상이면 확장 스킵 (이미 충분한 품질)
        if needs_expansion and overall_score >= 0.2:
            self.logger.debug(f"⏭️ [PERFORMANCE] Skipping context expansion: overall_score={overall_score:.2f} >= 0.2")
            needs_expansion = False
        # 성능 최적화: overall_score가 0.2 미만일 때만 확장 (기존: needs_expansion만 확인)
        if needs_expansion and overall_score < 0.2:
            context_dict, validation_results = self._expand_context_and_revalidate(
                state, context_dict, query, query_type, extracted_keywords, validation_results
            )
        elif needs_expansion and overall_score >= 0.2:
            # overall_score가 0.2 이상이면 확장 스킵 (성능 최적화)
            self.logger.debug(f"⏭️ [PERFORMANCE] Skipping context expansion: overall_score={overall_score} >= 0.2")

        return context_dict, validation_results, retrieved_docs
    
    def _monitor_search_quality(
        self,
        validation_results: Dict[str, Any],
        overall_score: float,
        state: LegalWorkflowState,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """검색 품질 모니터링"""
        if 0.4 <= overall_score < 0.5:
            relevance = validation_results.get('relevance_score', 0.0)
            coverage = validation_results.get('coverage_score', 0.0)
            sufficiency = validation_results.get('sufficiency_score', 0.0)
            self.logger.warning(
                f"⚠️ [SEARCH QUALITY] Low quality detected: overall_score={overall_score:.2f} "
                f"(relevance={relevance:.2f}, coverage={coverage:.2f}, sufficiency={sufficiency:.2f})"
            )
        elif overall_score < 0.4:
            relevance = validation_results.get('relevance_score', 0.0)
            coverage = validation_results.get('coverage_score', 0.0)
            sufficiency = validation_results.get('sufficiency_score', 0.0)
            self.logger.warning(
                f"⚠️ [SEARCH QUALITY] Very low quality detected: overall_score={overall_score:.2f} "
                f"(relevance={relevance:.2f}, coverage={coverage:.2f}, sufficiency={sufficiency:.2f})"
            )
            
            if overall_score == 0.0:
                self.logger.warning(
                    f"⚠️ [SEARCH QUALITY] CRITICAL: Search quality is 0.00. "
                    f"This may cause answer generation failure. "
                    f"retrieved_docs_count={len(state.get('retrieved_docs', []))}, "
                    f"semantic_results_count={len(state.get('search', {}).get('semantic_results', []) if isinstance(state.get('search'), dict) else [])}"
                )
                retrieved_docs = self._recover_retrieved_docs_for_low_quality(state, retrieved_docs)
        
        return retrieved_docs
    
    def _recover_retrieved_docs_for_low_quality(
        self,
        state: LegalWorkflowState,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """검색 품질이 낮을 때 retrieved_docs 복구"""
        if not retrieved_docs or len(retrieved_docs) == 0:
            self.logger.warning("⚠️ [SEARCH QUALITY] No retrieved_docs available. Attempting to recover from search results...")
            search_group = state.get("search", {})
            if isinstance(search_group, dict):
                semantic_results = search_group.get("semantic_results", [])
                keyword_results = search_group.get("keyword_results", [])
                if semantic_results and len(semantic_results) > 0:
                    retrieved_docs = semantic_results[:10]
                    state["retrieved_docs"] = retrieved_docs
                    self.logger.debug(f"⚠️ [SEARCH QUALITY] Recovered {len(retrieved_docs)} docs from semantic_results")
                elif keyword_results and len(keyword_results) > 0:
                    retrieved_docs = keyword_results[:10]
                    state["retrieved_docs"] = retrieved_docs
                    self.logger.debug(f"⚠️ [SEARCH QUALITY] Recovered {len(retrieved_docs)} docs from keyword_results")
            
            if not retrieved_docs or len(retrieved_docs) == 0:
                self.logger.warning("⚠️ [SEARCH QUALITY] Failed to recover retrieved_docs. Answer generation may fail.")
        else:
            self.logger.debug(f"⚠️ [SEARCH QUALITY] Retrieved {len(retrieved_docs)} docs despite low quality. Proceeding with answer generation.")
        
        return retrieved_docs
    
    def _expand_context_and_revalidate(
        self,
        state: LegalWorkflowState,
        context_dict: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        validation_results: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """컨텍스트 확장 및 재검증"""
        state = self._adaptive_context_expansion(state, validation_results)
        expanded_context = self.context_builder.build_intelligent_context(state)
        
        if isinstance(expanded_context, dict):
            context_dict = expanded_context
        elif isinstance(expanded_context, str):
            context_dict = {"context": expanded_context}
        else:
            self.logger.warning(f"⚠️ [CONTEXT DICT] Expanded context is not dict or str (type: {type(expanded_context)})")
            context_dict = {}
        
        if not isinstance(context_dict, dict):
            if isinstance(context_dict, str):
                context_dict = {"context": context_dict}
            else:
                context_dict = {}
        
        validation_results = self._validate_context_quality(
            context_dict, query, query_type, extracted_keywords, state
        )
        
        if not isinstance(validation_results, dict):
            self.logger.error(f"❌ [VALIDATION] validation_results after expansion is not dict (type: {type(validation_results)}), using empty dict")
            if isinstance(validation_results, str):
                validation_results = {"error": validation_results, "overall_score": 0.0}
            else:
                validation_results = {"overall_score": 0.0}
        
        metadata = self._get_metadata_safely(state)
        expansion_stats = metadata.get("context_expansion_stats", {}) if isinstance(metadata, dict) else {}
        if expansion_stats:
            final_overall_score = validation_results.get("overall_score", 0.0) if isinstance(validation_results, dict) else 0.0
            score_improvement = final_overall_score - expansion_stats.get("initial_overall_score", 0.0)
            expansion_stats["final_overall_score"] = final_overall_score
            expansion_stats["score_improvement"] = score_improvement
            
            metadata["context_expansion_stats"] = expansion_stats
            self._set_state_value(state, "metadata", metadata)
            
            self.logger.debug(
                f"📊 [CONTEXT EXPANSION] Effect analysis: "
                f"score improvement={score_improvement:+.2f} "
                f"({expansion_stats.get('initial_overall_score', 0.0):.2f} → {final_overall_score:.2f}), "
                f"docs added={expansion_stats.get('added_doc_count', 0)}, "
                f"duration={expansion_stats.get('expansion_duration', 0.0):.2f}s"
            )
        
        return context_dict, validation_results
    
    def _prepare_quality_feedback_and_context(
        self,
        state: LegalWorkflowState,
        is_retry: bool,
        context_dict: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        """품질 피드백 준비 및 context_dict에 추가"""
        quality_feedback = None
        base_prompt_type = "korean_legal_expert"
        
        regeneration_reason = (
            self._get_state_value(state, "regeneration_reason") or
            state.get("regeneration_reason") or
            state.get("metadata", {}).get("regeneration_reason")
        )

        if is_retry:
            quality_feedback = self.answer_generator.get_quality_feedback_for_retry(state)
            base_prompt_type = self.answer_generator.determine_retry_prompt_type(quality_feedback)

            self.logger.debug(
                f"🔄 [RETRY WITH FEEDBACK] Previous score: {quality_feedback.get('previous_score', 0):.2f}, "
                f"Failed checks: {len(quality_feedback.get('failed_checks', []))}, "
                f"Prompt type: {base_prompt_type}, "
                f"Regeneration reason: {regeneration_reason}"
            )

            metadata = self._get_metadata_safely(state)
            metadata["retry_feedback"] = quality_feedback
            if regeneration_reason:
                metadata["regeneration_reason"] = regeneration_reason
            self._set_state_value(state, "metadata", metadata)

        if quality_feedback and isinstance(context_dict, dict):
            context_dict["quality_feedback"] = quality_feedback
        
        if regeneration_reason and isinstance(context_dict, dict):
            context_dict["regeneration_reason"] = regeneration_reason
            retry_counts = self.retry_manager.get_retry_counts(state)
            context_dict["retry_count"] = retry_counts.get("generation", 0)
            self.logger.debug(
                f"🔄 [REGENERATION PROMPT] Adding regeneration reason to context: {regeneration_reason}, "
                f"retry_count: {retry_counts.get('generation', 0)}"
            )
        
        return quality_feedback, base_prompt_type, context_dict
    
    def _inject_search_results_into_context(
        self,
        state: LegalWorkflowState,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """검색 결과를 context_dict에 주입"""
        if not retrieved_docs or len(retrieved_docs) == 0:
            self.logger.warning(
                f"⚠️ [NO SEARCH RESULTS] retrieved_docs is empty. "
                f"LLM will generate answer without document references. "
                f"Query: '{query[:50]}...'"
            )
            if isinstance(context_dict, dict):
                context_dict["has_search_results"] = False
                context_dict["search_results_note"] = (
                    "현재 데이터베이스에서 관련 법률 문서를 찾지 못했습니다. "
                    "일반적인 법률 정보를 바탕으로 답변을 제공하되, "
                    "구체적인 조문이나 판례를 인용할 수 없음을 명시하세요."
                )
        else:
            if isinstance(context_dict, dict):
                context_dict["has_search_results"] = True

        if retrieved_docs and len(retrieved_docs) > 0 and isinstance(context_dict, dict):
            # 🔥 개선: retrieved_docs를 context_dict에 포함 (폴백용)
            context_dict["retrieved_docs"] = retrieved_docs
            context_dict["retrieved_docs_count"] = len(retrieved_docs)
            
            structured_docs = context_dict.get("structured_documents", {})
            documents_in_structured = []

            if isinstance(structured_docs, dict):
                documents_in_structured = structured_docs.get("documents", [])

            min_required_docs = max(1, min(3, int(len(retrieved_docs) * 0.3))) if len(retrieved_docs) > 5 else 1

            has_valid_documents = (
                isinstance(structured_docs, dict)
                and documents_in_structured
                and len(documents_in_structured) > 0
                and len(documents_in_structured) >= min_required_docs
            )

            self.logger.debug(
                f"🔍 [STRUCTURED DOCS CHECK] retrieved_docs={len(retrieved_docs)}, "
                f"structured_docs_count={len(documents_in_structured)}, "
                f"min_required={min_required_docs}, "
                f"has_valid={has_valid_documents}"
            )

            if not has_valid_documents:
                normalized_documents = self._normalize_retrieved_docs_to_structured(retrieved_docs)
                
                if normalized_documents:
                    structured_docs = self._create_structured_documents_from_normalized(
                        normalized_documents, retrieved_docs, context_dict, state
                    )
                    # 🔥 개선: context_dict에 명시적으로 할당 (이중 보장)
                    if isinstance(context_dict, dict) and structured_docs:
                        context_dict["structured_documents"] = structured_docs
                        context_dict["document_count"] = len(normalized_documents)
                        context_dict["docs_included"] = len(normalized_documents)
                    self.logger.info(
                        f"✅ [SEARCH RESULTS INJECTION] Added {len(normalized_documents)} documents "
                        f"from retrieved_docs to context_dict.structured_documents"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ [SEARCH RESULTS INJECTION] retrieved_docs has {len(retrieved_docs)} docs "
                        f"but none have valid content (>10 chars)"
                    )
            else:
                doc_count = len(documents_in_structured)
                # 🔥 개선: has_valid_documents가 True여도 retrieved_docs를 context_dict에 명시적으로 포함
                if isinstance(context_dict, dict) and retrieved_docs:
                    context_dict["retrieved_docs"] = retrieved_docs
                    context_dict["retrieved_docs_count"] = len(retrieved_docs)
                    self.logger.debug(
                        f"✅ [SEARCH RESULTS INJECTION] retrieved_docs already in structured_documents, "
                        f"also added to context_dict.retrieved_docs ({len(retrieved_docs)} docs)"
                    )
                # 🔥 개선: 기존 structured_docs도 context_dict에 명시적으로 할당 (이중 보장)
                if isinstance(context_dict, dict) and structured_docs:
                    context_dict["structured_documents"] = structured_docs
                    context_dict["document_count"] = doc_count
                    context_dict["docs_included"] = doc_count
                self.logger.debug(
                    f"✅ [SEARCH RESULTS] structured_documents already has {doc_count} valid documents "
                    f"(retrieved_docs: {len(retrieved_docs)}, required: {min_required_docs})"
                )
                self._save_structured_documents_to_state(state, structured_docs)
                
                if doc_count < len(retrieved_docs) * 0.5:
                    self.logger.warning(
                        f"⚠️ [SEARCH RESULTS] structured_documents has only {doc_count} documents "
                        f"while retrieved_docs has {len(retrieved_docs)}. "
                        f"This may indicate some documents were lost during preparation."
                    )
        
        return context_dict
    
    def _normalize_retrieved_docs_to_structured(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """retrieved_docs를 structured_documents 형태로 정규화"""
        normalized_documents = []
        self.logger.debug(f"🔍 [NORMALIZE] Processing {len(retrieved_docs)} retrieved_docs")
        
        for idx, doc in enumerate(retrieved_docs[:10], 1):
            if not isinstance(doc, dict):
                self.logger.warning(f"⚠️ [NORMALIZE] Doc {idx} is not a dict, skipping")
                continue
            
            # 멀티 질의 검색 결과의 다양한 필드명 지원
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("content_text")
                or doc.get("chunk_text")
                or doc.get("document_text")
                or doc.get("full_text")
                or doc.get("body")
                or doc.get("summary")
                or (doc.get("metadata", {}) or {}).get("content", "")
                or (doc.get("metadata", {}) or {}).get("text", "")
                or ""
            )
            
            self.logger.debug(f"🔍 [NORMALIZE] Doc {idx}: content length={len(content)}, "
                            f"has sub_query={bool(doc.get('sub_query'))}, "
                            f"has metadata={bool(doc.get('metadata'))}")

            source = (
                doc.get("source")
                or doc.get("title")
                or doc.get("document_id")
                or f"Document_{idx}"
            )

            relevance_score = (
                doc.get("relevance_score")
                or doc.get("score")
                or doc.get("final_weighted_score")
                or 0.5
            )

            # 멀티 질의 메타데이터 확인
            has_multi_query = bool(doc.get("sub_query") or 
                                 (isinstance(doc.get("metadata"), dict) and doc.get("metadata", {}).get("sub_query")))
            
            # 멀티 질의 결과는 content 길이 체크 완화 (10자 → 3자)
            min_content_length = 3 if has_multi_query else 10
            
            if content and len(content.strip()) >= min_content_length:
                # 멀티 질의 검색 결과의 메타데이터 보존
                doc_metadata = doc.get("metadata", {})
                if not isinstance(doc_metadata, dict):
                    doc_metadata = {}
                
                # 멀티 질의 관련 메타데이터 추가
                if doc.get("sub_query"):
                    doc_metadata["sub_query"] = doc.get("sub_query")
                if doc.get("search_type"):
                    doc_metadata["search_type"] = doc.get("search_type")
                if doc.get("original_query"):
                    doc_metadata["original_query"] = doc.get("original_query")
                
                # 문서 타입 정보 보존 (DocumentType 추론을 위해)
                from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
                
                # 최상위 필드의 정보를 metadata에 복사
                if doc.get("statute_name") or doc.get("law_name") or doc.get("article_no"):
                    doc_metadata["statute_name"] = doc.get("statute_name") or doc.get("law_name")
                    doc_metadata["law_name"] = doc.get("law_name") or doc.get("statute_name")
                    doc_metadata["article_no"] = doc.get("article_no") or doc.get("article_number")
                
                if doc.get("case_id") or doc.get("court") or doc.get("doc_id") or doc.get("casenames"):
                    doc_metadata["case_id"] = doc.get("case_id")
                    doc_metadata["court"] = doc.get("court") or doc.get("ccourt")
                    doc_metadata["doc_id"] = doc.get("doc_id")
                    doc_metadata["casenames"] = doc.get("casenames")
                    doc_metadata["precedent_id"] = doc.get("precedent_id")
                
                # 기존 metadata의 정보도 최상위 필드로 복사
                if doc_metadata.get("statute_name") and not doc.get("statute_name"):
                    doc["statute_name"] = doc_metadata.get("statute_name")
                if doc_metadata.get("law_name") and not doc.get("law_name"):
                    doc["law_name"] = doc_metadata.get("law_name")
                if doc_metadata.get("article_no") and not doc.get("article_no"):
                    doc["article_no"] = doc_metadata.get("article_no")
                if doc_metadata.get("case_id") and not doc.get("case_id"):
                    doc["case_id"] = doc_metadata.get("case_id")
                if doc_metadata.get("court") and not doc.get("court"):
                    doc["court"] = doc_metadata.get("court")
                if doc_metadata.get("doc_id") and not doc.get("doc_id"):
                    doc["doc_id"] = doc_metadata.get("doc_id")
                if doc_metadata.get("casenames") and not doc.get("casenames"):
                    doc["casenames"] = doc_metadata.get("casenames")
                
                # DocumentType Enum을 사용하여 타입 추출
                doc_type = DocumentType.from_metadata(doc)
                doc_type_str = doc_type.value
                
                # 🔍 로깅: 정규화 중 타입 추론 과정 추적 (처음 3개만)
                if idx < 3:
                    self.logger.info(
                        f"🔍 [METADATA TRACE] Normalize doc {idx} type inference: "
                        f"inferred_type={doc_type_str}, "
                        f"has_statute_fields={bool(doc.get('statute_name') or doc.get('law_name') or doc.get('article_no') or doc_metadata.get('statute_name') or doc_metadata.get('law_name') or doc_metadata.get('article_no'))}, "
                        f"has_case_fields={bool(doc.get('case_id') or doc.get('court') or doc.get('doc_id') or doc_metadata.get('case_id') or doc_metadata.get('court') or doc_metadata.get('doc_id'))}"
                    )
                
                # 타입 정보를 metadata와 최상위 필드에 저장 (명시적으로 설정)
                doc["type"] = doc_type_str
                doc_metadata["type"] = doc_type_str
                
                # content가 너무 짧으면 source로 보완
                if len(content.strip()) < 10 and has_multi_query:
                    content = f"{source}: {content}".strip() if source else content
                
                normalized_documents.append({
                    "document_id": idx,
                    "source": source,
                    "content": content[:2000],
                    "relevance_score": float(relevance_score),
                    "type": doc_type_str,  # type 정보 추가
                    "law_name": doc.get("law_name") or doc_metadata.get("law_name"),  # 법령명 추가
                    "article_no": doc.get("article_no") or doc_metadata.get("article_no"),  # 조문번호 추가
                    "court": doc.get("court") or doc_metadata.get("court"),  # 법원명 추가
                    "doc_id": doc.get("doc_id") or doc_metadata.get("doc_id"),  # 판례 ID 추가
                    "metadata": doc_metadata
                })
            else:
                self.logger.debug(f"⚠️ [NORMALIZE] Doc {idx} skipped: content length={len(content) if content else 0}, "
                                f"min_length={min_content_length}, has_multi_query={has_multi_query}")
        
        # 🔍 로깅: 정규화 후 출력 문서 메타데이터 확인
        if normalized_documents:
            sample_output = normalized_documents[0] if normalized_documents else {}
            self.logger.info(
                f"🔍 [METADATA TRACE] _normalize_retrieved_docs_to_structured 출력 (sample): "
                f"type={sample_output.get('type')}, "
                f"metadata_type={sample_output.get('metadata', {}).get('type') if isinstance(sample_output.get('metadata'), dict) else 'N/A'}"
            )
        
        return normalized_documents
    
    def _create_structured_documents_from_normalized(
        self,
        normalized_documents: List[Dict[str, Any]],
        retrieved_docs: List[Dict[str, Any]],
        context_dict: Dict[str, Any],
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """정규화된 문서들로 structured_documents 생성"""
        structured_docs = {
            "documents": normalized_documents,
            "total_count": len(normalized_documents),
            "source_mapping": [
                {
                    "original_index": idx,
                    "document_id": doc.get("document_id"),
                    "source": doc.get("source"),
                    "transformed": True
                }
                for idx, doc in enumerate(normalized_documents)
            ]
        }
        
        if isinstance(context_dict, dict):
            context_dict["structured_documents"] = structured_docs
            context_dict["document_count"] = len(normalized_documents)
            context_dict["docs_included"] = len(normalized_documents)
            context_dict["retrieved_to_structured_mapping"] = {
                "total_retrieved": len(retrieved_docs),
                "total_transformed": len(normalized_documents),
                "transformation_rate": len(normalized_documents) / len(retrieved_docs) if retrieved_docs else 0
            }
        
        self._save_structured_documents_to_state(state, structured_docs)
        
        return structured_docs
    
    def _save_structured_documents_to_state(self, state: LegalWorkflowState, structured_docs: Dict[str, Any]) -> None:
        """structured_documents를 state에 저장"""
        if structured_docs and isinstance(structured_docs, dict):
            self._set_state_value(state, "structured_documents", structured_docs)
            if "search" not in state:
                state["search"] = {}
            state["search"]["structured_documents"] = structured_docs
            
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["structured_documents"] = structured_docs
    
    def _generate_and_validate_prompt(
        self,
        state: LegalWorkflowState,
        context_dict: Dict[str, Any],
        query: str,
        question_type: str,
        domain: str,
        model_type: Any,
        base_prompt_type: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[Path], int, int]:
        """프롬프트 생성 및 검증"""
        optimized_prompt = self.unified_prompt_manager.get_optimized_prompt(
            query=query,
            question_type=question_type,
            domain=domain,
            context=context_dict,
            model_type=model_type,
            base_prompt_type=base_prompt_type
        )

        prompt_length = len(optimized_prompt) if isinstance(optimized_prompt, str) else 0
        context_length_in_dict = context_dict.get("context_length", 0) if isinstance(context_dict, dict) else 0
        docs_included = (
            context_dict.get("docs_included", context_dict.get("document_count", 0))
            if isinstance(context_dict, dict) else 0
        )

        # 🔥 개선: 문서 섹션 검증 로직 개선 (다양한 섹션 제목 지원)
        has_documents_section = isinstance(optimized_prompt, str) and (
            "검색된 법률 문서" in optimized_prompt or 
            "검색된 참고 문서" in optimized_prompt or
            "## 검색된" in optimized_prompt or
            "## 🔍" in optimized_prompt or
            "[문서 1]" in optimized_prompt or
            "[문서 2]" in optimized_prompt
        )
        documents_in_prompt = optimized_prompt.count("문서") if (isinstance(optimized_prompt, str) and has_documents_section) else 0
        structured_docs_count = 0
        structured_docs_in_context = context_dict.get("structured_documents", {}) if isinstance(context_dict, dict) else {}
        if isinstance(structured_docs_in_context, dict):
            structured_docs_count = len(structured_docs_in_context.get("documents", []))

        self.logger.debug(
            f"✅ [PROMPT GENERATED] Final prompt created: "
            f"{prompt_length} chars, "
            f"context: {context_length_in_dict} chars, "
            f"docs_in_context_dict: {docs_included}, "
            f"structured_docs: {structured_docs_count}, "
            f"has_documents_section: {has_documents_section}, "
            f"'문서' mentions in prompt: {documents_in_prompt}"
        )

        metadata = self._get_state_value(state, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if len(optimized_prompt) > 10000:
            metadata["optimized_prompt"] = optimized_prompt[:10000] + "... (truncated)"
            metadata["optimized_prompt_length"] = len(optimized_prompt)
        else:
            metadata["optimized_prompt"] = optimized_prompt
            metadata["optimized_prompt_length"] = len(optimized_prompt)

        prompt_generation_info = {
            "prompt_length": prompt_length,
            "context_length": context_length_in_dict,
            "docs_in_context_dict": docs_included,
            "structured_docs_count": structured_docs_count,
            "has_documents_section": has_documents_section,
            "documents_in_prompt": documents_in_prompt,
            "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0
        }
        metadata["prompt_generation_info"] = prompt_generation_info
        self._set_state_value(state, "metadata", metadata)

        # 성능 최적화: 개발 환경이 아닐 때는 프롬프트 검증 간소화
        current_env = Environment.get_current()
        is_development = current_env.is_debug_enabled() or os.getenv("DEBUG", "false").lower() == "true"
        if is_development:
            prompt_validation_result = self._validate_prompt_content(
                optimized_prompt, prompt_length, context_dict, retrieved_docs, structured_docs_count
            )
            metadata["prompt_validation"] = prompt_validation_result
            self._set_state_value(state, "metadata", metadata)

            self.logger.debug(
                f"📝 [PROMPT PREVIEW] Final prompt preview (last 300 chars):\n"
                f"{optimized_prompt[-300:] if len(optimized_prompt) > 300 else optimized_prompt}"
            )

            prompt_file = None
            try:
                debug_dir = lawfirm_langgraph_path.parent.parent / "logs" / "test" / "prompts"
                debug_dir.mkdir(parents=True, exist_ok=True)
                prompt_file = debug_dir / f"prompt_{int(time.time())}.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(optimized_prompt)
                self.logger.debug(f"💾 [PROMPT SAVED] Full prompt saved to {prompt_file} ({prompt_length} chars)")
            except Exception as e:
                self.logger.debug(f"Could not save prompt to file: {e}")
        else:
            # 성능 최적화: 프로덕션 환경에서는 간소화된 검증만 수행
            # 🔥 개선: 문서 섹션 검증 로직 개선 (다양한 섹션 제목 지원)
            has_documents_section = isinstance(optimized_prompt, str) and (
                "검색된 법률 문서" in optimized_prompt or 
                "검색된 참고 문서" in optimized_prompt or
                "## 검색된" in optimized_prompt or
                "## 🔍" in optimized_prompt or
                "[문서 1]" in optimized_prompt or
                "[문서 2]" in optimized_prompt
            )
            prompt_validation_result = {
                "has_documents_section": has_documents_section,
                "prompt_length": prompt_length,
                "validation_errors": [] if has_documents_section else ["Documents section not found"]
            }
            metadata["prompt_validation"] = prompt_validation_result
            self._set_state_value(state, "metadata", metadata)
            prompt_file = None

        return optimized_prompt, prompt_file, prompt_length, structured_docs_count
    
    def _validate_prompt_content(
        self,
        optimized_prompt: str,
        prompt_length: int,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        structured_docs_count: int
    ) -> Dict[str, Any]:
        """프롬프트 내용 검증 (성능 최적화: 간소화된 검증)"""
        # 🔥 개선: 문서 섹션 검증 로직 개선 (다양한 섹션 제목 지원)
        has_documents_section = isinstance(optimized_prompt, str) and (
            "검색된 법률 문서" in optimized_prompt or 
            "검색된 참고 문서" in optimized_prompt or
            "## 검색된" in optimized_prompt or
            "## 🔍" in optimized_prompt or
            "[문서 1]" in optimized_prompt or
            "[문서 2]" in optimized_prompt
        )
        documents_in_prompt = optimized_prompt.count("문서") if (isinstance(optimized_prompt, str) and has_documents_section) else 0
        
        # 성능 최적화: 상세 검증은 최대 3개 문서만 확인
        context_text = context_dict.get("context", "") if isinstance(context_dict, dict) else ""
        structured_docs_in_context = context_dict.get("structured_documents", {}) if isinstance(context_dict, dict) else {}
        documents_in_context = []
        if isinstance(structured_docs_in_context, dict):
            documents_in_context = structured_docs_in_context.get("documents", [])[:3]  # 성능 최적화: 5 -> 3으로 감소

        doc_found_count = 0
        has_search_section = False
        
        if retrieved_docs and len(retrieved_docs) > 0:
            if not has_documents_section:
                # 🔥 개선: retrieved_docs 폴백 사용 시에도 검증 통과
                context_retrieved_docs = context_dict.get("retrieved_docs", [])
                if context_retrieved_docs and len(context_retrieved_docs) > 0:
                    # 폴백 처리 시도 여부 확인 (로그에서 확인 가능)
                    self.logger.warning(
                        f"⚠️ [PROMPT VALIDATION] retrieved_docs has {len(retrieved_docs)} documents "
                        f"but prompt does not contain documents_section. "
                        f"Fallback may have been used. Check logs for 'FINAL PROMPT' messages."
                    )
                else:
                    self.logger.error(
                        f"❌ [PROMPT VALIDATION ERROR] retrieved_docs has {len(retrieved_docs)} documents "
                        f"but prompt does not contain documents_section! "
                        f"This may cause LLM to generate answer without sources!"
                    )
            else:
                self.logger.debug(
                    f"✅ [PROMPT VALIDATION] Documents section found in prompt "
                    f"(retrieved_docs: {len(retrieved_docs)}, structured_docs: {structured_docs_count})"
                )
                doc_section_start = optimized_prompt.find("## 🔍")
                if doc_section_start >= 0:
                    doc_section_preview = optimized_prompt[doc_section_start:doc_section_start+500]
                    self.logger.debug(f"📄 [PROMPT PREVIEW] Documents section preview:\n{doc_section_preview}...")

            if context_text and len(context_text) > 100:
                context_preview = context_text[:100]
                if context_preview in optimized_prompt:
                    self.logger.debug(
                        f"✅ [PROMPT VALIDATION] Context text confirmed in final prompt "
                        f"({len(context_text)} chars included)"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ [PROMPT VALIDATION] Context text may not be fully included in final prompt. "
                        f"Context length: {len(context_text)} chars, Prompt length: {prompt_length} chars"
                    )

            if documents_in_context:
                for doc in documents_in_context[:5]:
                    if isinstance(doc, dict):
                        doc_content = doc.get("content", "")
                        doc_source = doc.get("source", "")
                        if doc_content and len(doc_content) > 50:
                            content_preview = doc_content[:150].strip()
                            if content_preview and content_preview in optimized_prompt:
                                doc_found_count += 1
                            elif doc_source and doc_source in optimized_prompt:
                                doc_found_count += 1

                if doc_found_count > 0:
                    self.logger.debug(
                        f"✅ [PROMPT VALIDATION] Found {doc_found_count}/{min(5, len(documents_in_context))} "
                        f"documents in final prompt"
                    )
                else:
                    # 🔥 개선: 프롬프트가 너무 짧으면 프롬프트 생성 실패 가능성
                    if prompt_length < 500:
                        self.logger.warning(
                            f"⚠️ [PROMPT VALIDATION] Prompt is too short ({prompt_length} chars), "
                            f"may indicate prompt generation failure. Documents in context: {len(documents_in_context)}. "
                            f"Check prompt generation logs for errors."
                        )
                    else:
                        self.logger.error(
                            f"❌ [PROMPT VALIDATION FAILED] No documents from structured_documents "
                            f"found in final prompt! Documents in context: {len(documents_in_context)}, "
                            f"Prompt length: {prompt_length} chars. "
                            f"This may cause LLM to generate answer without document references!"
                        )
            else:
                self.logger.warning(
                    f"⚠️ [PROMPT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                    f"but structured_documents is empty. Prompt may not include search results."
                )

            search_section_keywords = [
                "검색된 법률 문서",
                "검색 결과",
                "반드시 참고",
                "문서들",
                "structured_documents"
            ]
            has_search_section = any(keyword in optimized_prompt for keyword in search_section_keywords)

            if not has_search_section and documents_in_context:
                self.logger.warning(
                    f"⚠️ [PROMPT VALIDATION] Search results section keywords not found in prompt "
                    f"despite having {len(documents_in_context)} documents in context."
                )
        else:
            self.logger.debug("ℹ️ [PROMPT VALIDATION] No retrieved_docs to validate against")

        prompt_validation_result = {
            "has_documents_section": has_documents_section,
            "documents_in_prompt": documents_in_prompt,
            "structured_docs_in_prompt": doc_found_count,
            "has_search_section": has_search_section,
            "validation_warnings": [],
            "validation_errors": []
        }
        
        if retrieved_docs and len(retrieved_docs) > 0:
            if not has_documents_section:
                prompt_validation_result["validation_errors"].append(
                    f"retrieved_docs has {len(retrieved_docs)} documents but prompt does not contain documents_section"
                )
            if not has_search_section and documents_in_context:
                prompt_validation_result["validation_warnings"].append(
                    f"Search results section keywords not found in prompt despite having {len(documents_in_context)} documents"
                )
            if doc_found_count == 0 and documents_in_context:
                prompt_validation_result["validation_errors"].append(
                    "No documents from structured_documents found in final prompt"
                )
        
        return prompt_validation_result
    
    def _generate_answer_with_cache(
        self,
        state: LegalWorkflowState,
        optimized_prompt: str,
        query: str,
        query_type: str,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        quality_feedback: Optional[Dict[str, Any]],
        is_retry: bool,
        callbacks: Optional[List[Any]] = None
    ) -> str:
        """답변 생성 및 캐시 처리
        
        Args:
            state: 워크플로우 상태
            optimized_prompt: 최적화된 프롬프트
            query: 사용자 질문
            query_type: 질문 타입
            context_dict: 컨텍스트 딕셔너리
            retrieved_docs: 검색된 문서 리스트
            quality_feedback: 품질 피드백
            is_retry: 재시도 여부
            callbacks: 콜백 핸들러 리스트 (스트리밍용, 선택적)
        """
        normalized_response = None
        
        current_env = Environment.get_current()
        is_development = current_env.is_debug_enabled() or os.getenv("DEBUG", "false").lower() == "true"
        
        if not is_retry and not is_development:
            cached_answer = self._check_cache_for_answer(query, query_type, context_dict, retrieved_docs)
            if cached_answer:
                normalized_response = cached_answer
        
        if not normalized_response:
            # 긴 글/코드 생성이 필요한지 판단 (프롬프트 길이 또는 컨텍스트 길이 기준)
            prompt_length = len(optimized_prompt)
            context_length = len(str(context_dict.get("context", "")))
            total_length = prompt_length + context_length
            
            # 긴 답변이 필요한 경우 판단 기준:
            # 1. 프롬프트가 5000자 이상
            # 2. 컨텍스트가 10000자 이상
            # 3. 총 길이가 12000자 이상
            # 4. 질문에 "코드", "생성", "작성", "긴" 등의 키워드 포함
            needs_long_text = (
                prompt_length >= 5000 or
                context_length >= 10000 or
                total_length >= 12000 or
                any(keyword in query for keyword in ["코드", "생성", "작성", "긴", "상세", "자세히"])
            )
            
            # 적절한 LLM 선택
            original_llm = self.answer_generator.llm
            if needs_long_text and self.llm_long_text:
                self.answer_generator.llm = self.llm_long_text
                self.logger.debug(
                    f"📝 [LONG TEXT MODE] 긴 글/코드 생성 모드 활성화 "
                    f"(프롬프트: {prompt_length}자, 컨텍스트: {context_length}자, "
                    f"timeout: {WorkflowConstants.TIMEOUT_LONG_TEXT}초)"
                )
            
            try:
                # 🔥 개선: 스트리밍 모드 확인 및 콜백 전달
                use_streaming = os.getenv("USE_STREAMING_MODE", "true").lower() == "true"
                final_callbacks = callbacks if (use_streaming and callbacks) else None
                
                if final_callbacks:
                    self.logger.debug(f"📡 [STREAMING] 콜백 {len(final_callbacks)}개를 LLM 호출에 전달")
                
                normalized_response = self.answer_generator.generate_answer_with_chain(
                    optimized_prompt=optimized_prompt,
                    query=query,
                    context_dict=context_dict,
                    quality_feedback=quality_feedback,
                    is_retry=is_retry,
                    callbacks=final_callbacks
                )
                
                # generate_answer_with_chain이 빈 응답을 반환한 경우 직접 LLM 호출
                if not normalized_response or not isinstance(normalized_response, str) or len(normalized_response.strip()) < 10:
                    self.logger.warning(
                        f"⚠️ [ANSWER GENERATION] generate_answer_with_chain returned empty response "
                        f"(length: {len(normalized_response) if normalized_response else 0}), "
                        f"falling back to direct LLM call"
                    )
                    normalized_response = self.answer_generator.call_llm_with_retry(optimized_prompt)
                    normalized_response = WorkflowUtils.normalize_answer(normalized_response)
                    self.logger.info(
                        f"✅ [ANSWER GENERATION] Direct LLM call successful: length={len(normalized_response) if normalized_response else 0}"
                    )
            finally:
                # 원래 LLM으로 복원
                self.answer_generator.llm = original_llm
            
            if normalized_response:
                self.logger.debug(
                    f"📡 [STREAMING] 답변 생성 완료 - "
                    f"길이: {len(normalized_response)} chars, "
                    f"on_llm_stream 이벤트가 발생하여 클라이언트로 실시간 전달됨"
                )
            
            if not is_retry and not is_development and normalized_response:
                self._cache_answer_if_quality_good(
                    query, query_type, context_dict, retrieved_docs, normalized_response
                )
        
        if normalized_response:
            self.logger.debug(
                f"📝 [ANSWER GENERATED] Response received:\n"
                f"   Normalized response length: {len(normalized_response)} characters\n"
                f"   Normalized response content: '{normalized_response[:300]}'\n"
                f"   Normalized response repr: {repr(normalized_response[:100])}"
            )
        else:
            self.logger.warning(
                f"⚠️ [ANSWER GENERATED] normalized_response is None or empty. "
                f"Query: '{query[:50]}...'"
            )
        
        # 빈 응답 검증 및 처리
        if not normalized_response or not isinstance(normalized_response, str) or len(normalized_response.strip()) < 10:
            self.logger.error(
                f"❌ [ANSWER GENERATED] Empty or invalid response detected. "
                f"Length: {len(normalized_response) if normalized_response else 0}, "
                f"Type: {type(normalized_response).__name__}"
            )
            # 빈 응답인 경우 빈 문자열 반환 (상위에서 폴백 처리)
            return ""
        
        return normalized_response
    
    def _check_cache_for_answer(
        self,
        query: str,
        query_type: str,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]]
    ) -> Optional[str]:
        """캐시에서 답변 확인"""
        context_text = context_dict.get("context", "")[:500] if isinstance(context_dict, dict) else ""
        docs_count = len(retrieved_docs) if retrieved_docs else 0
        cache_key_parts = [
            query,
            query_type,
            context_text,
            str(docs_count)
        ]
        cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
        
        try:
            cached_result = self.performance_optimizer.cache.get_cached_answer(
                f"answer_gen:{cache_key}", query_type
            )
            if cached_result and isinstance(cached_result, dict) and "answer" in cached_result:
                cached_answer = cached_result.get("answer")
                
                if isinstance(cached_answer, dict):
                    cached_answer = cached_answer.get("answer", "") if "answer" in cached_answer else str(cached_answer)
                elif not isinstance(cached_answer, str):
                    cached_answer = str(cached_answer)
                
                if cached_answer and isinstance(cached_answer, str):
                    if self._validate_cached_answer_quality(cached_answer, query):
                        answer_length = len(cached_answer.strip())
                        quality_score = self._calculate_answer_quality_score(cached_answer, query)
                        self.logger.debug(
                            f"✅ [CACHE HIT] 답변 생성 결과 캐시 히트: {cache_key[:16]}... "
                            f"(length: {answer_length} chars, quality: {quality_score:.2f})"
                        )
                        return cached_answer
                    else:
                        answer_length = len(cached_answer.strip())
                        quality_score = self._calculate_answer_quality_score(cached_answer, query)
                        self.logger.warning(
                            f"⚠️ [CACHE REJECT] 캐시된 답변 품질이 낮아 재생성: "
                            f"length={answer_length}, quality_score={quality_score:.2f} < 0.6"
                        )
        except Exception as e:
            self.logger.debug(f"답변 생성 캐시 확인 중 오류 (무시): {e}")
        
        return None
    
    def _validate_cached_answer_quality(self, cached_answer: str, query: str) -> bool:
        """캐시된 답변 품질 검증"""
        answer_length = len(cached_answer.strip())
        
        is_too_short = answer_length < WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION
        is_evasive = any(phrase in cached_answer for phrase in [
            "관련 정보를 찾을 수 없습니다",
            "더 구체적인 질문을 해주시면",
            "정보를 찾을 수 없습니다"
        ])
        
        quality_score = self._calculate_answer_quality_score(cached_answer, query)
        quality_threshold = 0.6
        
        return quality_score >= quality_threshold and not is_too_short and not is_evasive
    
    def _calculate_answer_quality_score(self, answer: str, query: str) -> float:
        """답변 품질 점수 계산"""
        answer_length = len(answer.strip())
        quality_score = 0.0
        
        if answer_length >= WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION:
            quality_score += 0.3
        
        if not any(phrase in answer for phrase in [
            "관련 정보를 찾을 수 없습니다",
            "더 구체적인 질문을 해주시면",
            "정보를 찾을 수 없습니다"
        ]):
            quality_score += 0.2
        
        if not self._detect_format_errors(answer):
            quality_score += 0.2
        
        if bool(re.search(r'[가-힣]+법\s*제?\s*\d+\s*조', answer)) or bool(re.search(r'대법원|법원.*\d{4}[다나마]\d+', answer)):
            quality_score += 0.2
        
        query_words = set(query.lower().split()) if query else set()
        answer_words = set(answer.lower().split())
        keyword_overlap = len(query_words.intersection(answer_words)) if query_words else 0
        if keyword_overlap >= 2 if query_words else True:
            quality_score += 0.1
        
        return quality_score
    
    def _cache_answer_if_quality_good(
        self,
        query: str,
        query_type: str,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        normalized_response: str
    ) -> None:
        """답변 품질이 좋으면 캐시에 저장"""
        answer_str = normalized_response if isinstance(normalized_response, str) else str(normalized_response)
        answer_length = len(answer_str.strip())
        is_evasive = any(phrase in answer_str for phrase in [
            "관련 정보를 찾을 수 없습니다",
            "더 구체적인 질문을 해주시면",
            "정보를 찾을 수 없습니다"
        ])
        
        if answer_length >= 500 and not is_evasive:
            try:
                context_text = context_dict.get("context", "")[:500] if isinstance(context_dict, dict) else ""
                docs_count = len(retrieved_docs) if retrieved_docs else 0
                cache_key_parts = [
                    query,
                    query_type,
                    context_text,
                    str(docs_count)
                ]
                cache_key = hashlib.md5(":".join(cache_key_parts).encode('utf-8')).hexdigest()
                self.performance_optimizer.cache.cache_answer(
                    f"answer_gen:{cache_key}",
                    query_type,
                    {"answer": normalized_response},
                    confidence=1.0,
                    sources=[]
                )
                self.logger.debug(f"✅ [CACHE STORE] 답변 생성 결과 캐시 저장: {cache_key[:16]}... (length: {answer_length} chars)")
            except Exception as e:
                self.logger.debug(f"답변 생성 캐시 저장 중 오류 (무시): {e}")
        else:
            self.logger.debug(
                f"⚠️ [CACHE SKIP] 답변 품질이 낮아 캐시 저장 건너뜀: "
                f"length={answer_length}, is_evasive={is_evasive}"
            )
    
    def _validate_and_enhance_answer(
        self,
        state: LegalWorkflowState,
        normalized_response: str,
        query: str,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        prompt_length: int,
        prompt_file: Optional[Path],
        structured_docs_count: int
    ) -> str:
        """답변 검증 및 보강"""
        normalized_response = self._validate_answer_start_and_retry(
            state, normalized_response
        )
        
        self._set_answer_safely(state, normalized_response)

        metadata = self._get_metadata_safely(state)
        metadata["answer_generation"] = {
            "prompt_length": prompt_length,
            "answer_length": len(normalized_response),
            "prompt_file": str(prompt_file) if prompt_file else None,
            "generation_timestamp": time.time(),
            "used_search_results": bool(retrieved_docs and len(retrieved_docs) > 0),
            "structured_docs_used": structured_docs_count
        }
        self._set_state_value(state, "metadata", metadata)

        validation_result = self._validate_and_enhance_citations(
            state, normalized_response, query, context_dict, retrieved_docs
        )
        
        metadata["answer_validation"] = validation_result
        search_usage_tracking = self._build_search_usage_tracking(
            validation_result, retrieved_docs, structured_docs_count, normalized_response
        )
        metadata["search_usage_tracking"] = search_usage_tracking
        
        self._monitor_answer_quality(validation_result, retrieved_docs)
        
        if validation_result.get("needs_regeneration", False):
            self.logger.warning(
                f"⚠️ [VALIDATION] Context usage low (coverage: {validation_result.get('coverage_score', 0.0):.2f}), "
                f"but regeneration is disabled. Answer generated with current validation result."
            )
        
        return normalized_response
    
    def _validate_answer_start_and_retry(
        self,
        state: LegalWorkflowState,
        normalized_response: str
    ) -> str:
        """답변 시작 부분 검증 및 재생성"""
        answer_str = normalized_response if isinstance(normalized_response, str) else str(normalized_response)
        first_500 = answer_str[:500] if len(answer_str) > 500 else answer_str
        
        # 개선: 더 유연한 검증 로직 (너무 엄격하지 않도록)
        has_specific_case_in_start = (
            re.search(r'\[문서[:\s]+[^\]]*[가-힣]*지방법원[가-힣]*[^\]]*-\s*\d{4}[가나다라마바사아자차카타파하]\d+[^\]]*\]', first_500) or
            re.search(r'나아가[^.]*이\s*사건[^.]*\.', first_500) or
            re.search(r'[가-힣]*지방법원[가-힣]*\s*-\s*\d{4}[가나다라마바사아자차카타파하]\d+', first_500) or
            (re.search(r'피고\s+[가-힣]+', first_500) and re.search(r'원고\s+[가-힣]+', first_500))  # 피고와 원고가 모두 있어야 특정 사건으로 판단
        )
        
        # 개선: 더 다양한 일반 원칙 패턴 인식
        has_general_principle_in_start = (
            re.search(r'일반적인?\s*법적?\s*원칙', first_500) or
            re.search(r'일반적으로?\s*적용되는?\s*법적?\s*원칙', first_500) or
            re.search(r'주의해야\s*할\s*일반적인?\s*법적?\s*원칙', first_500) or
            re.search(r'[가-힣]+법\s*제\d+조', first_500) or  # 모든 법령 조문
            re.search(r'법률에\s*따르면', first_500) or
            re.search(r'법률상', first_500) or
            re.search(r'규정에\s*따르면', first_500) or
            re.search(r'원칙적으로', first_500) or
            re.search(r'일반적으로', first_500) or
            len(first_500) >= 100  # 충분히 긴 답변은 일반 원칙이 있다고 간주
        )
        
        # 개선: 검증 기준 완화 - 특정 사건이 명확히 있고 일반 원칙이 없을 때만 재시도
        if has_specific_case_in_start and not has_general_principle_in_start:
            retry_counts = self.retry_manager.get_retry_counts(state)
            can_retry = self.retry_manager.should_allow_retry(state, "generation")
            
            if can_retry and retry_counts['generation'] < RetryConfig.MAX_GENERATION_RETRIES:
                self.logger.warning(
                    f"⚠️ [IMMEDIATE VALIDATION] Answer start validation failed:\n"
                    f"   has_specific_case_in_start: {has_specific_case_in_start}\n"
                    f"   has_general_principle_in_start: {has_general_principle_in_start}\n"
                    f"   Retrying immediately (retry count: {retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES})"
                )
                self._set_state_value(state, "needs_regeneration", True)
                state["needs_regeneration"] = True
                if "metadata" not in state:
                    state["metadata"] = {}
                state["metadata"]["needs_regeneration"] = True
                
                if has_specific_case_in_start:
                    self._set_state_value(state, "regeneration_reason", "specific_case_in_start")
                    state["regeneration_reason"] = "specific_case_in_start"
                    state["metadata"]["regeneration_reason"] = "specific_case_in_start"
                elif not has_general_principle_in_start:
                    self._set_state_value(state, "regeneration_reason", "general_principle_not_in_start")
                    state["regeneration_reason"] = "general_principle_not_in_start"
                    state["metadata"]["regeneration_reason"] = "general_principle_not_in_start"
                
                self.retry_manager.increment_retry_count(state, "generation")
                state = self.generate_answer_enhanced(state)
                normalized_response = self._get_state_value(state, "answer", "")
                if not normalized_response:
                    normalized_response = state.get("answer", "")
            else:
                self.logger.warning(
                    f"⚠️ [IMMEDIATE VALIDATION] Answer start validation failed but cannot retry "
                    f"(retry count: {retry_counts['generation']}/{RetryConfig.MAX_GENERATION_RETRIES})"
                )
        
        return normalized_response
    
    def _validate_and_enhance_citations(
        self,
        state: LegalWorkflowState,
        normalized_response: str,
        query: str,
        context_dict: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Citation 검증 및 보강"""
        # 먼저 validation_result를 가져와서 실제 citation_coverage 확인
        validation_result = self.answer_generator.validate_answer_uses_context(
            answer=normalized_response,
            context=context_dict,
            query=query,
            retrieved_docs=retrieved_docs
        )
        
        citation_coverage = validation_result.get("citation_coverage", 0.0)
        citation_count = validation_result.get("citation_count", 0)
        
        # 로그 출력 강화
        self.logger.debug(
            f"🔍 [CITATION VALIDATION] citation_coverage={citation_coverage:.2f}, "
            f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}"
        )
        self.logger.debug(
            f"🔍 [CITATION VALIDATION] citation_coverage={citation_coverage:.2f}, "
            f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}"
        )
        
        # 개선: citation_coverage 임계값을 0.5로 낮춰서 더 적극적으로 보강
        if citation_coverage < 0.5 and retrieved_docs:
            self.logger.debug(
                f"🔧 [CITATION ENHANCEMENT] Triggering enhancement: "
                f"citation_coverage={citation_coverage:.2f} < 0.5 (target: >= 0.5), "
                f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs)}"
            )
            self.logger.debug(
                f"🔧 [CITATION ENHANCEMENT] Triggering enhancement: "
                f"citation_coverage={citation_coverage:.2f} < 0.5 (target: >= 0.5), "
                f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs)}"
            )
            legal_references = context_dict.get("legal_references", []) if isinstance(context_dict, dict) else []
            citations = context_dict.get("citations", []) if isinstance(context_dict, dict) else []
            
            enhanced_answer = self._enhance_answer_with_citations(
                normalized_response,
                retrieved_docs,
                legal_references,
                citations
            )
            
            if enhanced_answer != normalized_response:
                normalized_response = enhanced_answer
                self._set_answer_safely(state, normalized_response)
                
                # 보강 후 다시 검증
                validation_result = self.answer_generator.validate_answer_uses_context(
                    answer=normalized_response,
                    context=context_dict,
                    query=query,
                    retrieved_docs=retrieved_docs
                )
                citation_coverage = validation_result.get("citation_coverage", citation_coverage)
                citation_count = validation_result.get("citation_count", citation_count)
        
        self.logger.debug(
            f"🔍 [CITATION CHECK] citation_coverage={citation_coverage:.2f}, "
            f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}"
        )
        
        return validation_result
    
    def _build_search_usage_tracking(
        self,
        validation_result: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]],
        structured_docs_count: int,
        normalized_response: str
    ) -> Dict[str, Any]:
        """검색 결과 사용 추적 정보 구축"""
        search_usage_tracking = {
            "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0,
            "structured_docs_count": structured_docs_count,
            "citation_count": validation_result.get("citation_count", 0),
            "coverage_score": validation_result.get("coverage_score", 0.0),
            "has_document_references": validation_result.get("has_document_references", False),
            "sources_in_answer": []
        }
        
        if retrieved_docs and isinstance(normalized_response, str):
            sources_found = []
            for doc in retrieved_docs:
                source = doc.get("source") or doc.get("title") or ""
                if source and source in normalized_response:
                    sources_found.append(source)
            search_usage_tracking["sources_in_answer"] = sources_found[:10]
        
        return search_usage_tracking
    
    def _monitor_answer_quality(
        self,
        validation_result: Dict[str, Any],
        retrieved_docs: List[Dict[str, Any]]
    ) -> None:
        """답변 품질 모니터링"""
        citation_count = validation_result.get("citation_count", 0)
        coverage_score = validation_result.get("coverage_score", 0.0)
        keyword_coverage = validation_result.get("keyword_coverage", 0.0)
        citation_coverage = validation_result.get("citation_coverage", 0.0)
        concept_coverage = validation_result.get("concept_coverage", 0.0)
        has_document_references = validation_result.get("has_document_references", False)
        
        if citation_coverage < 0.3:
            self.logger.warning(
                f"⚠️ [ANSWER QUALITY] Low citation coverage: {citation_coverage:.2f} "
                f"(expected >= 0.3). Citation count: {citation_count}, "
                f"Expected: {validation_result.get('citations_expected', 0)}, "
                f"Found: {validation_result.get('citations_found', 0)}"
            )
        elif citation_coverage < 0.5:
            self.logger.warning(
                f"⚠️ [ANSWER QUALITY] Moderate citation coverage: {citation_coverage:.2f} "
                f"(expected >= 0.5). Citation count: {citation_count}"
            )
        
        if coverage_score < 0.4:
            self.logger.warning(
                f"⚠️ [ANSWER QUALITY] Low coverage: {coverage_score:.2f} "
                f"(expected >= 0.4). Keyword: {keyword_coverage:.2f}, "
                f"Citation: {citation_coverage:.2f}, Concept: {concept_coverage:.2f}"
            )
        elif coverage_score < 0.6:
            self.logger.warning(
                f"⚠️ [ANSWER QUALITY] Moderate coverage: {coverage_score:.2f} "
                f"(expected >= 0.6). Keyword: {keyword_coverage:.2f}, "
                f"Citation: {citation_coverage:.2f}"
            )
        
        if coverage_score < 0.5 and retrieved_docs:
            self.logger.debug(
                f"🔧 [ANSWER QUALITY] Attempting automatic improvement: "
                f"coverage={coverage_score:.2f} < 0.5"
            )
            
            if keyword_coverage < 0.5:
                self.logger.debug(
                    f"🔧 [ANSWER QUALITY] Low keyword coverage: {keyword_coverage:.2f}, "
                    f"considering keyword enhancement..."
                )
        
        if retrieved_docs and len(retrieved_docs) > 0:
            if citation_count < 2:
                self.logger.warning(
                    f"⚠️ [VALIDATION] Low citation count: {citation_count} (expected >= 2) "
                    f"for {len(retrieved_docs)} documents. "
                    f"Coverage: {coverage_score:.2f}, Has refs: {has_document_references}"
                )
            elif not has_document_references:
                self.logger.warning(
                    f"⚠️ [VALIDATION] Citations found ({citation_count}) but no document source references detected. "
                    f"Answer may not be using retrieved documents effectively."
                )
            else:
                self.logger.debug(
                    f"✅ [VALIDATION] Good context usage: {citation_count} citations, "
                    f"coverage: {coverage_score:.2f}, document references: {has_document_references}"
                )
    
    def _recover_retrieved_docs_comprehensive(self, state: LegalWorkflowState) -> List[Dict[str, Any]]:
        """검색 결과를 모든 가능한 위치에서 복구 (강화된 버전)
        
        Args:
            state: 워크플로우 상태
            
        Returns:
            복구된 검색 결과 리스트
        """
        # 1. 최상위 레벨 확인
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        if retrieved_docs and len(retrieved_docs) > 0:
            self.logger.debug(f"✅ [RECOVER] 최상위 레벨에서 복구: {len(retrieved_docs)}개")
            return retrieved_docs
        
        # 2. search.retrieved_docs 확인
        if "search" in state and isinstance(state.get("search"), dict):
            search_docs = state["search"].get("retrieved_docs", [])
            if search_docs and len(search_docs) > 0:
                self.logger.debug(f"✅ [RECOVER] search.retrieved_docs에서 복구: {len(search_docs)}개")
                self._set_state_value(state, "retrieved_docs", search_docs)
                return search_docs
        
        # 3. search.results 확인 (multi-query 결과일 수 있음)
        if "search" in state and isinstance(state.get("search"), dict):
            search_results = state["search"].get("results", [])
            if search_results and len(search_results) > 0:
                self.logger.debug(f"✅ [RECOVER] search.results에서 복구: {len(search_results)}개")
                self._set_state_value(state, "retrieved_docs", search_results)
                return search_results
        
        # 4. common.search.retrieved_docs 확인
        if "common" in state and isinstance(state.get("common"), dict):
            common_search = state["common"].get("search", {})
            if isinstance(common_search, dict):
                common_docs = common_search.get("retrieved_docs", [])
                if common_docs and len(common_docs) > 0:
                    self.logger.debug(f"✅ [RECOVER] common.search.retrieved_docs에서 복구: {len(common_docs)}개")
                    self._set_state_value(state, "retrieved_docs", common_docs)
                    return common_docs
        
        # 5. metadata.search.retrieved_docs 확인
        metadata = self._get_metadata_safely(state)
        if "search" in metadata and isinstance(metadata.get("search"), dict):
            metadata_docs = metadata["search"].get("retrieved_docs", [])
            if metadata_docs and len(metadata_docs) > 0:
                self.logger.debug(f"✅ [RECOVER] metadata.search.retrieved_docs에서 복구: {len(metadata_docs)}개")
                self._set_state_value(state, "retrieved_docs", metadata_docs)
                return metadata_docs
        
        # 6. metadata.retrieved_docs 확인
        if "retrieved_docs" in metadata:
            metadata_docs = metadata["retrieved_docs"]
            if metadata_docs and len(metadata_docs) > 0:
                self.logger.debug(f"✅ [RECOVER] metadata.retrieved_docs에서 복구: {len(metadata_docs)}개")
                self._set_state_value(state, "retrieved_docs", metadata_docs)
                return metadata_docs
        
        # 7. 전역 캐시 확인
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache:
                cached_docs = (
                    _global_search_results_cache.get("retrieved_docs", []) or
                    _global_search_results_cache.get("search", {}).get("retrieved_docs", []) or
                    _global_search_results_cache.get("common", {}).get("search", {}).get("retrieved_docs", [])
                )
                if cached_docs and len(cached_docs) > 0:
                    self.logger.debug(f"✅ [RECOVER] 전역 캐시에서 복구: {len(cached_docs)}개")
                    self._set_state_value(state, "retrieved_docs", cached_docs)
                    return cached_docs
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"전역 캐시 복구 실패: {e}")
        
        # 8. semantic_results + keyword_results에서 재구성
        semantic_results = self._get_state_value(state, "semantic_results", [])
        keyword_results = self._get_state_value(state, "keyword_results", [])
        
        if semantic_results or keyword_results:
            combined_docs = (semantic_results or []) + (keyword_results or [])
            if combined_docs:
                self.logger.debug(f"✅ [RECOVER] semantic_results + keyword_results에서 재구성: {len(combined_docs)}개")
                self._set_state_value(state, "retrieved_docs", combined_docs)
                return combined_docs
        
        self.logger.warning("⚠️ [RECOVER] 모든 위치에서 retrieved_docs를 찾을 수 없음")
        return []
    
    def _recover_retrieved_docs_at_start(self, state: LegalWorkflowState) -> None:
        """답변 생성 시작 시 retrieved_docs 복구 (기존 메서드, _recover_retrieved_docs_comprehensive 사용)"""
        retrieved_docs = self._recover_retrieved_docs_comprehensive(state)
        if retrieved_docs:
            self.logger.debug(f"✅ [RESTORE] retrieved_docs 복구 완료: {len(retrieved_docs)}개")
        else:
            self.logger.warning("⚠️ [RESTORE] retrieved_docs 복원 실패")

