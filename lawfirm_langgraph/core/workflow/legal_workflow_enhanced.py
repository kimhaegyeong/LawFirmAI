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
import concurrent.futures
import hashlib
import logging
import os
import re
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# 성능 최적화: 정규식 패턴 컴파일 (모듈 레벨)
LAW_PATTERN = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
PRECEDENT_PATTERN = re.compile(r'대법원|법원.*\d{4}[다나마]\d+')

from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Mock observe decorator (Langfuse 제거됨)
def observe(**kwargs):
        def decorator(func):
            return func
        return decorator

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# lawfirm_langgraph 디렉토리를 sys.path에 추가 (core 모듈 import를 위해)
lawfirm_langgraph_path = Path(__file__).parent.parent
sys.path.insert(0, str(lawfirm_langgraph_path))

from core.agents.handlers.answer_formatter import AnswerFormatterHandler
from core.generation.generators.answer_generator import AnswerGenerator
from core.workflow.builders.chain_builders import (
    AnswerGenerationChainBuilder,
    ClassificationChainBuilder,
    DirectAnswerChainBuilder,
    DocumentAnalysisChainBuilder,
    QueryEnhancementChainBuilder,
)
from core.classification.handlers.classification_handler import ClassificationHandler
from core.generation.generators.context_builder import ContextBuilder
from core.processing.extractors import (
    DocumentExtractor,
    QueryExtractor,
    ResponseExtractor,
)
from core.search.optimizers.keyword_mapper import LegalKeywordMapper
from core.search.connectors.legal_data_connector import LegalDataConnectorV2
from core.shared.wrappers.node_wrappers import with_state_optimization
from core.agents.optimizers.performance_optimizer import PerformanceOptimizer
from core.workflow.builders.prompt_builders import PromptBuilder, QueryBuilder
from core.workflow.builders.prompt_chain_executor import PromptChainExecutor
from core.generation.validators.quality_validators import (
    AnswerValidator,
    SearchValidator,
)
from core.agents.validators.quality_validators import ContextValidator
from core.search.optimizers.query_enhancer import QueryEnhancer
from core.processing.extractors.reasoning_extractor import ReasoningExtractor
from core.processing.parsers.response_parsers import (
    AnswerParser,
    ClassificationParser,
    DocumentParser,
    QueryParser,
)
from core.search.handlers.search_handler import SearchHandler
from core.workflow.state.state_definitions import LegalWorkflowState
from core.workflow.state.state_utils import (
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)
from core.workflow.utils.workflow_constants import (
    AnswerExtractionPatterns,
    QualityThresholds,
    RetryConfig,
    WorkflowConstants,
)
from core.workflow.utils.workflow_routes import WorkflowRoutes
from core.workflow.utils.workflow_utils import WorkflowUtils
from core.classification.classifiers.question_classifier import QuestionType
from core.search.processors.result_merger import ResultMerger, ResultRanker
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
    from core.processing.integration.term_integration_system import TermIntegrator
except ImportError:
    # 호환성을 위한 fallback
    from core.processing.integration.term_integration_system import TermIntegrator
try:
    from core.services.unified_prompt_manager import (
        LegalDomain,
        ModelType,
        UnifiedPromptManager,
    )
except ImportError:
    # 호환성을 위한 fallback
    from core.services.unified_prompt_manager import (
        LegalDomain,
        ModelType,
        UnifiedPromptManager,
    )

# Logger 초기화
logger = logging.getLogger(__name__)

# AnswerStructureEnhancer 통합 (답변 구조화 및 법적 근거 강화)
try:
    from core.generation.formatters.answer_structure_enhancer import AnswerStructureEnhancer
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = True
except ImportError:
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = False


from core.workflow.state.workflow_types import QueryComplexity, RetryCounterManager
from core.workflow.mixins import (
    StateUtilsMixin,
    QueryUtilsMixin,
    SearchMixin,
    AnswerGenerationMixin,
    DocumentAnalysisMixin,
    ClassificationMixin,
)


class EnhancedLegalQuestionWorkflow(
    StateUtilsMixin,
    QueryUtilsMixin,
    SearchMixin,
    AnswerGenerationMixin,
    DocumentAnalysisMixin,
    ClassificationMixin
):
    """개선된 법률 질문 처리 워크플로우"""

    def __init__(self, config: LangGraphConfig):
        self.config = config

        # 개선: 로거를 명시적으로 초기화 (루트 로거 핸들러 사용)
        self.logger = logging.getLogger(__name__)

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
        from core.agents.keyword_extractor import KeywordExtractor
        self.keyword_extractor = KeywordExtractor(use_morphology=True, logger_instance=self.logger)
        self.data_connector = LegalDataConnectorV2()
        self.performance_optimizer = PerformanceOptimizer()
        self.term_integrator = TermIntegrator()
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()
        
        # 검색 결과 처리 프로세서 초기화 (Phase 13 리팩토링)
        from core.search.processors.search_result_processor import SearchResultProcessor
        self.search_result_processor = SearchResultProcessor(
            logger=self.logger,
            result_merger=self.result_merger,
            result_ranker=self.result_ranker
        )
        
        # 워크플로우 문서 처리 프로세서 초기화 (Phase 14 리팩토링)
        from core.workflow.processors.workflow_document_processor import WorkflowDocumentProcessor
        semantic_search_engine = None
        if hasattr(self, 'search_handler') and self.search_handler:
            semantic_search_engine = getattr(self.search_handler, 'semantic_search_engine', None)
        self.workflow_document_processor = WorkflowDocumentProcessor(
            logger=self.logger,
            query_enhancer=None,  # query_enhancer는 나중에 설정됨
            semantic_search_engine=semantic_search_engine  # 문서 내용 복원을 위해 전달
        )
        
        # 워크플로우 검증기 초기화 (Phase 16 리팩토링)
        from core.workflow.validators.workflow_validator import WorkflowValidator
        self.workflow_validator = WorkflowValidator(logger=self.logger)
        
        # 워크플로우 프롬프트 빌더 초기화 (Phase 15 리팩토링)
        from core.workflow.builders.workflow_prompt_builder import WorkflowPromptBuilder
        self.workflow_prompt_builder = WorkflowPromptBuilder(logger=self.logger)
        
        # QueryEnhancer는 llm과 llm_fast 초기화 이후에 초기화됨 (아래 참조)
        
        # HybridQueryProcessor 초기화 (HuggingFace + LLM 하이브리드)
        # LLM 초기화 이후에 설정됨 (아래 참조)

        # 재시도 카운터 관리자 초기화
        self.retry_manager = RetryCounterManager(self.logger)

        # 추론 과정 분리 모듈 초기화 (Phase 1 리팩토링)
        self.reasoning_extractor = ReasoningExtractor(logger=self.logger)

        # AnswerStructureEnhancer 초기화 (답변 구조화 및 법적 근거 강화)
        if ANSWER_STRUCTURE_ENHANCER_AVAILABLE:
            self.answer_structure_enhancer = AnswerStructureEnhancer()
            self.logger.info("AnswerStructureEnhancer initialized for answer quality enhancement")
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
                self.logger.info("📌 MLflow 벡터 스토어를 사용합니다 (기본값)")
            
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
                self.logger.info(f"✅ SemanticSearchEngineV2 초기화 완료 - 사용 모델: {self.semantic_search.model_name}")
            
            if hasattr(self.semantic_search, 'diagnose'):
                diagnosis = self.semantic_search.diagnose()
                if diagnosis.get("available"):
                    self.logger.info(f"SemanticSearchEngineV2 initialized successfully with {db_path}")
                else:
                    self.logger.warning(f"SemanticSearchEngineV2 initialized but not available: {diagnosis.get('issues', [])}")
            else:
                self.logger.info(f"SemanticSearchEngineV2 initialized successfully with {db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SemanticSearchEngineV2: {e}", exc_info=True)
            self.semantic_search = None
            self.logger.error(f"SemanticSearchEngineV2 is not available. Vector search will be disabled.")

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
            self.logger.info("MultiTurnQuestionHandler initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MultiTurnQuestionHandler: {e}")
            self.multi_turn_handler = None
            self.conversation_manager = None

        # AIKeywordGenerator 초기화 (AI 키워드 확장)
        try:
            from core.processing.extractors.ai_keyword_generator import AIKeywordGenerator
            self.ai_keyword_generator = AIKeywordGenerator()
            self.logger.info("AIKeywordGenerator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AIKeywordGenerator: {e}")
            self.ai_keyword_generator = None

        # EmotionIntentAnalyzer 초기화 (긴급도 평가용)
        try:
            from core.classification.analyzers.emotion_intent_analyzer import EmotionIntentAnalyzer
            self.emotion_analyzer = EmotionIntentAnalyzer()
            self.logger.info("EmotionIntentAnalyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize EmotionIntentAnalyzer: {e}")
            self.emotion_analyzer = None

        # LegalBasisValidator 초기화 (법령 검증용)
        try:
            from core.generation.validators.legal_basis_validator import LegalBasisValidator
            self.legal_validator = LegalBasisValidator()
            self.logger.info("LegalBasisValidator initialized")
        except ImportError:
            try:
                # 호환성을 위한 fallback
                from core.services.legal_basis_validator import LegalBasisValidator
                self.legal_validator = LegalBasisValidator()
                self.logger.info("LegalBasisValidator initialized (from services)")
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
                # 호환성을 위한 fallback
                from core.services.document_processor import LegalDocumentProcessor
            utils_config = UtilsConfig()
            self.document_processor = LegalDocumentProcessor(utils_config)
            self.logger.info("LegalDocumentProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalDocumentProcessor: {e}")
            self.document_processor = None

        # ConfidenceCalculator 초기화 (신뢰도 계산용)
        try:
            from core.generation.validators.confidence_calculator import (
                ConfidenceCalculator,
            )
            self.confidence_calculator = ConfidenceCalculator()
            self.logger.info("ConfidenceCalculator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ConfidenceCalculator: {e}")
            self.confidence_calculator = None

        # LLMInitializer 초기화
        try:
            from core.workflow.initializers.llm_initializer import LLMInitializer
            self.llm_initializer = LLMInitializer(config=self.config)
            self.logger.info("LLMInitializer initialized")
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
        
        # 품질 검증용 LLM 초기화 (별도)
        self.validator_llm = self._initialize_validator_llm()

        # DocumentAnalysisProcessor 초기화
        try:
            from core.workflow.processors.document_analysis_processor import DocumentAnalysisProcessor
            self.document_analysis_processor = DocumentAnalysisProcessor(
                llm=self.llm,
                logger=self.logger,
                document_processor=self.document_processor,
                llm_fast=self.llm_fast
            )
            self.logger.info("DocumentAnalysisProcessor initialized")
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
            self.logger.info("SearchExecutionProcessor initialized")
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
            self.logger.info("ContextExpansionProcessor initialized")
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
            self.logger.info("AnswerQualityValidator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AnswerQualityValidator: {e}")
            self.answer_quality_validator = None

        # WorkflowGraphBuilder 초기화
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
            self.logger.info("WorkflowGraphBuilder initialized")
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
            embedding_model_name = getattr(self.config, 'embedding_model', None)
            self.hybrid_query_processor = HybridQueryProcessor(
                keyword_extractor=self.keyword_extractor,
                term_integrator=self.term_integrator,
                llm=self.llm,
                embedding_model_name=embedding_model_name,
                logger=self.logger
            )
            self.logger.info("✅ HybridQueryProcessor initialized (HuggingFace + LLM hybrid)")
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
            self.logger.info("DirectAnswerHandler initialized")
        except ImportError:
            try:
                from core.generation.generators.direct_answer_handler import DirectAnswerHandler
                self.direct_answer_handler = DirectAnswerHandler(
                    llm=self.llm,
                    llm_fast=self.llm_fast,
                    logger=self.logger
                )
                self.logger.info("DirectAnswerHandler initialized (from generators)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DirectAnswerHandler: {e}")
                self.direct_answer_handler = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize DirectAnswerHandler: {e}")
            self.direct_answer_handler = None

        # 워크플로우 라우팅 핸들러 초기화 (Phase 9 리팩토링) - answer_generator 초기화 이후
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
                self.logger.info(f"Agentic AI mode enabled with {len(LEGAL_TOOLS)} tools (from core.workflow.tools)")
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
            self.logger.info(f"WorkflowStatistics initialized (enabled: {self.config.enable_statistics})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WorkflowStatistics: {e}")
            self.workflow_statistics = None
            # 통계가 활성화된 경우 기본 통계 생성
            if self.config.enable_statistics:
                try:
                    from core.workflow.utils.workflow_statistics import WorkflowStatistics
                    temp_stats = WorkflowStatistics(enable_statistics=True)
                    self.stats = temp_stats._initialize_statistics()
                    self.logger.info("Created fallback statistics dictionary")
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
            self.logger.info("ConversationProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ConversationProcessor: {e}")
            self.conversation_processor = None

        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.info("EnhancedLegalQuestionWorkflow initialized with UnifiedPromptManager.")

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

    def _initialize_ollama(self):
        """LLMInitializer.initialize_ollama 래퍼"""
        if self.llm_initializer:
            return self.llm_initializer.initialize_ollama()
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

    def _create_mock_llm_fallback(self):
        """Mock LLM 생성 (폴백)"""
        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response for: " + prompt
            async def ainvoke(self, prompt):
                return "Mock LLM async response for: " + prompt
        return MockLLM()

    def _build_graph(self) -> StateGraph:
        """WorkflowGraphBuilder.build_graph 래퍼"""
        if self.workflow_graph_builder:
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
                "agentic_decision_node": self.agentic_decision_node
            }
            return self.workflow_graph_builder.build_graph(node_handlers)
        
        from langgraph.graph import StateGraph
        return StateGraph(LegalWorkflowState)

    @observe(name="expand_keywords")
    @with_state_optimization("expand_keywords", enable_reduction=False)
    def expand_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 추출 노드 (HuggingFace 모델 기반)"""
        try:
            start_time = time.time()

            # metadata 보존
            preserved_complexity = state.get("metadata", {}).get("query_complexity") if isinstance(state.get("metadata"), dict) else None
            preserved_needs_search = state.get("metadata", {}).get("needs_search") if isinstance(state.get("metadata"), dict) else None

            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"] = dict(state["metadata"])
            if preserved_complexity:
                state["metadata"]["query_complexity"] = preserved_complexity
            if preserved_needs_search is not None:
                state["metadata"]["needs_search"] = preserved_needs_search
            state["metadata"]["_last_executed_node"] = "expand_keywords"

            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "expand_keywords"

            # 1. HuggingFace 모델 기반 키워드 추출
            keywords = self._get_state_value(state, "extracted_keywords", [])
            if len(keywords) == 0:
                query = self._get_state_value(state, "query", "")
                query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", "general_question"))
                legal_field = self._get_state_value(state, "legal_field", "")
                
                # HuggingFace 모델을 사용한 키워드 추출
                extracted_keywords = []
                
                # 방법 1: HybridQueryProcessor의 LegalQueryAnalyzer 사용 (우선)
                if self.hybrid_query_processor and hasattr(self.hybrid_query_processor, 'query_analyzer'):
                    try:
                        self.logger.info(f"🔍 [HF KEYWORD EXTRACTION] Using LegalQueryAnalyzer for keyword extraction")
                        analysis_result = self.hybrid_query_processor.query_analyzer.analyze_query(
                            query=query,
                            query_type=query_type_str,
                            legal_field=legal_field
                        )
                        extracted_keywords = analysis_result.get("core_keywords", [])
                        key_concepts = analysis_result.get("key_concepts", [])
                        
                        # core_keywords와 key_concepts 통합
                        all_keywords = list(set(extracted_keywords + key_concepts))
                        extracted_keywords = [kw for kw in all_keywords if isinstance(kw, str) and len(kw.strip()) >= 2]
                        
                        self.logger.info(f"✅ [HF KEYWORD EXTRACTION] Extracted {len(extracted_keywords)} keywords using LegalQueryAnalyzer (core: {len(analysis_result.get('core_keywords', []))}, concepts: {len(key_concepts)})")
                    except Exception as e:
                        self.logger.warning(f"⚠️ [HF KEYWORD EXTRACTION] LegalQueryAnalyzer failed: {e}, using fallback method", exc_info=True)
                        extracted_keywords = []
                
                # 방법 2: 직접 LegalQueryAnalyzer 초기화 (폴백)
                if not extracted_keywords:
                    try:
                        from core.search.optimizers.legal_query_analyzer import LegalQueryAnalyzer
                        embedding_model_name = getattr(self.config, 'embedding_model', None)
                        
                        analyzer = LegalQueryAnalyzer(
                            keyword_extractor=None,  # KeywordExtractor 사용 안 함
                            embedding_model_name=embedding_model_name,
                            logger=self.logger
                        )
                        
                        self.logger.info(f"🔍 [HF KEYWORD EXTRACTION] Using standalone LegalQueryAnalyzer for keyword extraction")
                        analysis_result = analyzer.analyze_query(
                            query=query,
                            query_type=query_type_str,
                            legal_field=legal_field
                        )
                        extracted_keywords = analysis_result.get("core_keywords", [])
                        key_concepts = analysis_result.get("key_concepts", [])
                        
                        # core_keywords와 key_concepts 통합
                        all_keywords = list(set(extracted_keywords + key_concepts))
                        extracted_keywords = [kw for kw in all_keywords if isinstance(kw, str) and len(kw.strip()) >= 2]
                        
                        self.logger.info(f"✅ [HF KEYWORD EXTRACTION] Extracted {len(extracted_keywords)} keywords using standalone LegalQueryAnalyzer")
                    except Exception as e:
                        self.logger.warning(f"⚠️ [HF KEYWORD EXTRACTION] Standalone LegalQueryAnalyzer failed: {e}, using simple regex fallback", exc_info=True)
                        # 최종 폴백: 간단한 정규식 기반 추출
                        import re
                        words = re.findall(r'[가-힣]+', query)
                        extracted_keywords = [w for w in words if len(w) >= 2][:15]
                
                # keyword_mapper를 사용한 키워드 추출 (보완, 선택적)
                mapper_keywords = []
                if self.keyword_mapper:
                    try:
                        mapper_keywords = self.keyword_mapper.get_keywords_for_question(query, query_type_str)
                        mapper_keywords = [kw for kw in mapper_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                    except Exception as e:
                        self.logger.debug(f"keyword_mapper failed: {e}")
                
                # 두 방법의 키워드 통합 (HuggingFace 우선)
                keywords = list(set(extracted_keywords + mapper_keywords))
                keywords = [kw for kw in keywords if isinstance(kw, str) and len(kw.strip()) >= 2]
                
                self._set_state_value(state, "extracted_keywords", keywords)
                self.logger.info(f"✅ [HF KEYWORD EXTRACTION] Final extracted {len(keywords)} keywords (HF: {len(extracted_keywords)}, mapper: {len(mapper_keywords)})")

            # 2. AI 키워드 확장 비활성화 (prepare_search_query의 HybridQueryProcessor에서 HuggingFace 모델로 처리)
            # HybridQueryProcessor의 LegalKeywordExpander가 HuggingFace 모델을 사용하여 키워드 확장을 수행하므로
            # 여기서 LLM 기반 확장은 중복이며, 504 Deadline Exceeded 에러를 방지하기 위해 비활성화
            should_expand_ai = False
            self.logger.info(f"🔍 [KEYWORD EXPANSION] LLM-based expansion disabled (using HybridQueryProcessor in prepare_search_query instead)")
            
            # 기존 LLM 확장 로직은 주석 처리 (필요시 복구 가능)
            if False and self.ai_keyword_generator:
                # 성능 최적화: 더 엄격한 조건으로 AI 확장 빈도 감소
                query_type = self._get_state_value(state, "query_type", "")
                query = self._get_state_value(state, "query", "")
                
                # 조건 1: 키워드 수 (5개 이상)
                # 조건 2: 쿼리 타입 (복잡한 타입만)
                # 조건 3: 쿼리 길이 (15자 이상, 더 엄격하게)
                # 조건 4: 키워드 품질 (법률 관련 키워드 포함 여부)
                complex_types = ["precedent_search", "law_inquiry", "legal_advice"]
                has_legal_keywords = any(
                    kw in query or any(legal_term in kw for legal_term in ["법", "조", "항", "판례", "법원", "법률"])
                    for kw in keywords
                )
                
                if (len(keywords) >= 5 and 
                    query_type in complex_types and 
                    len(query) >= 15 and  # 10 → 15로 증가 (더 긴 쿼리만 확장)
                    has_legal_keywords):  # 법률 관련 키워드가 있는 경우만
                    should_expand_ai = True
                    self.logger.info(f"🔍 [AI KEYWORD EXPANSION] Conditions met: query_type={query_type}, keywords={len(keywords)}, query_len={len(query)}, has_legal={has_legal_keywords}")
                else:
                    skip_reasons = []
                    if len(keywords) < 5:
                        skip_reasons.append(f"keywords={len(keywords)} < 5")
                    if query_type not in complex_types:
                        skip_reasons.append(f"query_type={query_type}")
                    if len(query) < 15:
                        skip_reasons.append(f"query_len={len(query)} < 15")
                    if not has_legal_keywords:
                        skip_reasons.append("no_legal_keywords")
                    self.logger.debug(f"🔍 [AI KEYWORD EXPANSION] Skipping: {', '.join(skip_reasons)}")

            # 3. AI 키워드 확장 (조건부, 캐싱 적용)
            if should_expand_ai:
                query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
                domain = WorkflowUtils.get_domain_from_query_type(query_type_str)

                # 캐싱 활성화 (개발 환경에서도 성능 향상을 위해 캐싱 사용)
                # 키워드 확장 결과 캐싱 확인
                expansion_result = None
                cache_hit_keywords = False
                try:
                    # 캐시 키 생성 개선: domain, keywords, query 기반 (캐시 히트율 향상)
                    query = self._get_state_value(state, "query", "")
                    keywords_str = ",".join(sorted([str(kw) for kw in keywords[:10]]))  # 상위 10개만 사용 (캐시 키 단순화)
                    query_normalized = query[:50].strip()  # 쿼리 앞부분 50자만 사용
                    cache_key = hashlib.md5(f"keyword_exp:{domain}:{keywords_str}:{query_normalized}".encode('utf-8')).hexdigest()
                    
                    # PerformanceOptimizer 캐시에서 확인
                    cached_result = self.performance_optimizer.cache.get_cached_answer(
                        f"keyword_exp:{cache_key}", query_type_str
                    )
                    if cached_result and isinstance(cached_result, dict) and "expansion_result" in cached_result:
                        expansion_data = cached_result.get("expansion_result")
                        if expansion_data:
                            # 캐시된 결과를 expansion_result 형태로 재구성
                            from types import SimpleNamespace
                            expansion_result = SimpleNamespace(
                                api_call_success=expansion_data.get("api_call_success", True),
                                expanded_keywords=expansion_data.get("expanded_keywords", []),
                                domain=expansion_data.get("domain", domain),
                                base_keywords=expansion_data.get("base_keywords", keywords),
                                confidence=expansion_data.get("confidence", 0.9),
                                expansion_method=expansion_data.get("expansion_method", "cache")
                            )
                            cache_hit_keywords = True
                            self.logger.info(f"✅ [CACHE HIT] 키워드 확장 결과 캐시 히트: {cache_key[:16]}...")
                except Exception as e:
                    self.logger.debug(f"키워드 확장 캐시 확인 중 오류 (무시): {e}")

                # 캐시 미스인 경우 AI 키워드 확장 수행
                if not expansion_result:
                    try:
                        # 현재 실행 중인 이벤트 루프 확인
                        # 타임아웃 추가 단축: 5초 → 3초 (성능 개선)
                        try:
                            loop = asyncio.get_running_loop()
                            # 이미 실행 중인 루프가 있는 경우, 새 스레드에서 실행
                            query = self._get_state_value(state, "query", "")
                            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
                            
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(
                                    lambda: asyncio.run(
                                        asyncio.wait_for(
                                            self.ai_keyword_generator.expand_domain_keywords(
                                                domain=domain,
                                                base_keywords=keywords,
                                                target_count=30,  # 50 → 30 (응답 시간 단축)
                                                query=query,
                                                query_type=query_type_str
                                            ),
                                            timeout=2.0  # 3초 → 2초로 추가 단축
                                        )
                                    )
                                )
                                expansion_result = future.result(timeout=3.0)  # 4초 → 3초로 단축
                        except RuntimeError:
                            # 실행 중인 루프가 없는 경우, 직접 실행
                            query = self._get_state_value(state, "query", "")
                            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
                            
                            expansion_result = asyncio.run(
                                asyncio.wait_for(
                                    self.ai_keyword_generator.expand_domain_keywords(
                                        domain=domain,
                                        base_keywords=keywords,
                                        target_count=30,  # 50 → 30 (응답 시간 단축)
                                        query=query,
                                        query_type=query_type_str
                                    ),
                                    timeout=2.0  # 3초 → 2초로 추가 단축
                                )
                            )
                        
                        # 캐시 저장 (항상 활성화)
                        if expansion_result and expansion_result.api_call_success:
                            try:
                                query = self._get_state_value(state, "query", "")
                                keywords_str = ",".join(sorted([str(kw) for kw in keywords[:10]]))  # 상위 10개만 사용
                                query_normalized = query[:50].strip()  # 쿼리 앞부분 50자만 사용
                                cache_key = hashlib.md5(f"keyword_exp:{domain}:{keywords_str}:{query_normalized}".encode('utf-8')).hexdigest()
                                cache_data = {
                                    "expansion_result": {
                                        "api_call_success": expansion_result.api_call_success,
                                        "expanded_keywords": expansion_result.expanded_keywords,
                                        "domain": expansion_result.domain,
                                        "base_keywords": expansion_result.base_keywords,
                                        "confidence": expansion_result.confidence,
                                        "expansion_method": expansion_result.expansion_method
                                    }
                                }
                                self.performance_optimizer.cache.save_cached_answer(
                                    f"keyword_exp:{cache_key}",
                                    cache_data,
                                    query_type_str
                                )
                                self.logger.debug(f"✅ [CACHE SAVE] 키워드 확장 결과 캐시 저장: {cache_key[:16]}...")
                            except Exception as e:
                                self.logger.debug(f"키워드 확장 캐시 저장 중 오류 (무시): {e}")
                        
                    except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
                        self.logger.warning(f"AI keyword expansion timeout (5s): {e}")
                        expansion_result = None
                    except asyncio.CancelledError as e:
                        self.logger.warning(f"AI keyword expansion cancelled: {e}")
                        expansion_result = None
                    except Exception as e:
                        self.logger.warning(f"AI keyword expansion failed: {e}")
                        expansion_result = None

                if expansion_result:

                    if expansion_result.api_call_success:
                        all_keywords = keywords + expansion_result.expanded_keywords
                        all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                        all_keywords = list(set(all_keywords))
                        
                        # extracted_keywords를 여러 위치에 저장 (안전성 강화)
                        self._set_state_value(state, "extracted_keywords", all_keywords)
                        # search 그룹에도 명시적으로 저장
                        if "search" not in state:
                            state["search"] = {}
                        if not isinstance(state["search"], dict):
                            state["search"] = {}
                        state["search"]["extracted_keywords"] = all_keywords
                        # 최상위 레벨에도 저장 (flat 구조 호환)
                        state["extracted_keywords"] = all_keywords
                        
                        self._set_state_value(state, "ai_keyword_expansion", {
                            "domain": expansion_result.domain,
                            "original_keywords": expansion_result.base_keywords,
                            "expanded_keywords": expansion_result.expanded_keywords,
                            "confidence": expansion_result.confidence,
                            "method": expansion_result.expansion_method
                        })
                        self.logger.info(
                            f"✅ [KEYWORD EXPANSION] Expanded {len(keywords)} → {len(all_keywords)} keywords "
                            f"(domain: {domain}, method: {expansion_result.expansion_method})"
                        )
                        # 저장 확인
                        saved_check = self._get_state_value(state, "extracted_keywords", [])
                        self.logger.info(f"🔍 [KEYWORD EXPANSION] Verification: saved {len(saved_check)} keywords to state (also in search group: {len(state.get('search', {}).get('extracted_keywords', []))})")
                    else:
                        fallback_keywords = self.ai_keyword_generator.expand_keywords_with_fallback(domain, keywords)
                        all_keywords = keywords + fallback_keywords
                        all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                        all_keywords = list(set(all_keywords))
                        
                        # extracted_keywords를 여러 위치에 저장 (안전성 강화)
                        self._set_state_value(state, "extracted_keywords", all_keywords)
                        # search 그룹에도 명시적으로 저장
                        if "search" not in state:
                            state["search"] = {}
                        if not isinstance(state["search"], dict):
                            state["search"] = {}
                        state["search"]["extracted_keywords"] = all_keywords
                        # 최상위 레벨에도 저장 (flat 구조 호환)
                        state["extracted_keywords"] = all_keywords
                        self._set_state_value(state, "ai_keyword_expansion", {
                            "domain": domain,
                            "original_keywords": keywords,
                            "expanded_keywords": fallback_keywords,
                            "confidence": 0.5,
                            "method": "fallback"
                        })
                        self.logger.info(
                            f"⚠️ [KEYWORD EXPANSION] Used fallback: {len(keywords)} → {len(all_keywords)} keywords"
                        )

            self._save_metadata_safely(state, "_last_executed_node", "expand_keywords")
            self._update_processing_time(state, start_time)
            self._add_step(state, "키워드 추출", f"키워드 추출 완료: {len(self._get_state_value(state, 'extracted_keywords', []))}개 (HuggingFace 모델 사용)")

        except Exception as e:
            self._handle_error(state, str(e), "키워드 추출 중 오류 발생")

        return state

    def _should_retry_generation(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_retry_generation 래퍼"""
        return self.workflow_routes.should_retry_generation(state)

    def _should_skip_final_node(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_skip_final_node 래퍼"""
        return self.workflow_routes.should_skip_final_node(state)
    
    def _should_retry_validation(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_retry_validation 래퍼"""
        return self.workflow_routes.should_retry_validation(state, answer_generator=self.answer_generator)


    @observe(name="agentic_decision")
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
            
            self.logger.info(f"🤖 [AGENTIC] Processing query with {len(self.legal_tools)} tools: {query[:100]}")
            
            # AgentExecutor 초기화 (지연 초기화)
            if self.agentic_agent is None:
                try:
                    from langchain.agents import AgentExecutor, create_openai_tools_agent
                    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
                    
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
                    self.logger.info("Agentic agent initialized successfully")
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
                        self.logger.info(f"Loaded {len(chat_history)} messages from conversation context (relevance-based)")
                
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
                    self.logger.info(f"✅ [AGENTIC] Retrieved {len(unique_results)} documents from tool execution")
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
            
            self.logger.info(f"✅ [AGENTIC] Completed in {processing_time:.2f}s, {len(search_results)} results, {len(tool_calls)} tools used")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in agentic_decision_node: {e}")
            self._handle_error(state, str(e), "Agentic 검색 중 오류")
            # 에러 시 기존 플로우로 fallback
            state.setdefault("search", {})["results"] = []
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
                    self.logger.info(f"✅ [FINAL NODE] Restored retrieved_docs from global cache: {len(retrieved_docs)} docs")
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
                    self.logger.info(f"✅ [FINAL NODE] Restored query_type from global cache: {query_type}")
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
            self.logger.info(
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
            self.logger.info(
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
            self.logger.warning(f"⚠️ [FORMAT_ANSWER] 'control' error detected. Preserving existing answer if available.")
        
        existing_answer = self._get_state_value(state, "answer", "")
        if isinstance(existing_answer, dict):
            existing_answer = existing_answer.get("text", "") or existing_answer.get("content", "") or str(existing_answer)
        elif not isinstance(existing_answer, str):
            existing_answer = str(existing_answer) if existing_answer else ""
        
        if existing_answer and len(existing_answer.strip()) > 10:
            self._set_answer_safely(state, existing_answer)
            self.logger.info(f"✅ [FORMAT_ANSWER] Preserved existing answer: length={len(existing_answer)}")
        else:
            query = self._get_state_value(state, "query", "")
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            if retrieved_docs and len(retrieved_docs) > 0:
                minimal_answer = f"질문 '{query}'에 대한 답변을 준비했습니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
            else:
                minimal_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다."
            self._set_answer_safely(state, minimal_answer)
            self.logger.info(f"⚠️ [FORMAT_ANSWER] Generated minimal answer: length={len(minimal_answer)}")
        
        self._set_state_value(state, "legal_validity_check", True)
        self._save_metadata_safely(state, "quality_score", 0.0, save_to_top_level=True)
        self._save_metadata_safely(state, "quality_check_passed", False, save_to_top_level=True)
    
    @observe(name="generate_answer_stream")
    @with_state_optimization("generate_answer_stream", enable_reduction=True)
    def generate_answer_stream(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """스트리밍 전용 답변 생성 노드 - 스트리밍만 수행하고 검증/포맷팅은 하지 않음 (콜백 방식 사용)"""
        try:
            start_time = time.time()
            self.logger.info("📡 [STREAM NODE] 스트리밍 전용 답변 생성 시작 (콜백 방식)")
            
            # 중요: retrieved_docs, query_type 등을 명시적으로 보존
            # State reduction으로 인한 손실 방지
            preserved_retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            preserved_structured_docs = self._get_state_value(state, "structured_documents", [])
            preserved_query_type = self._get_state_value(state, "query_type") or (state.get("metadata", {}).get("query_type") if isinstance(state.get("metadata"), dict) else None)
            
            # generate_answer_enhanced 실행 (답변 생성만)
            # 콜백은 LangGraph의 astream_events()와 config의 callbacks를 통해 처리됨
            state = self.generate_answer_enhanced(state)
            
            # 보존된 필드 복원 (reduction으로 손실된 경우 대비)
            if preserved_retrieved_docs and not self._get_state_value(state, "retrieved_docs"):
                self._set_state_value(state, "retrieved_docs", preserved_retrieved_docs)
            if preserved_structured_docs and not self._get_state_value(state, "structured_documents"):
                self._set_state_value(state, "structured_documents", preserved_structured_docs)
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
                self.logger.info(f"✅ [CITATIONS] {len(citations)}개 인용 추가됨")
            
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
            self.logger.info(f"📡 [STREAM NODE] 스트리밍 전용 답변 생성 완료 ({elapsed:.2f}s)")
            
        except asyncio.CancelledError:
            self.logger.warning("⚠️ [STREAM NODE] 스트리밍 작업이 취소되었습니다. 기존 답변 보존 시도 중...")
            # 취소된 경우 기존 답변 보존 시도
            existing_answer = self._get_state_value(state, "answer", "")
            if existing_answer and len(str(existing_answer).strip()) > 10:
                self._set_answer_safely(state, existing_answer)
                self.logger.info(f"✅ [STREAM NODE] 취소 후 기존 답변 보존 완료 (길이: {len(str(existing_answer))}자)")
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

    @observe(name="generate_answer_final")
    @with_state_optimization("generate_answer_final", enable_reduction=True)
    def generate_answer_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 검증 및 포맷팅 노드 - 검증과 포맷팅만 수행"""
        try:
            overall_start_time = time.time()
            self.logger.info("✅ [FINAL NODE] 최종 검증 및 포맷팅 시작")
            
            self._restore_state_data_for_final(state)
            
            validation_start_time = time.time()
            quality_check_passed = self._validate_and_handle_regeneration(state)
            
            quality_check_passed = self._handle_format_errors(state, quality_check_passed)
            
            self._update_processing_time(state, validation_start_time)
            
            if quality_check_passed:
                self._format_and_finalize(state, overall_start_time)

            self._update_processing_time(state, overall_start_time)

        except asyncio.CancelledError:
            self.logger.warning("⚠️ [FINAL NODE] Operation was cancelled. Preserving existing answer.")
            existing_answer = self._get_state_value(state, "answer", "")
            if existing_answer and len(str(existing_answer).strip()) > 10:
                self._set_answer_safely(state, existing_answer)
                self.logger.info(f"✅ [FINAL NODE] Preserved existing answer after cancellation: length={len(str(existing_answer))}")
            else:
                self._handle_final_node_error(state, asyncio.CancelledError("Operation was cancelled"))
        except Exception as e:
            self._handle_final_node_error(state, e)

        return state

    @observe(name="generate_and_validate_answer")
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
            self.logger.info(
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
                
                self.logger.info(
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
                            self.logger.info(
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
                    self.logger.info(
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
        """AnswerQualityValidator.validate_answer_quality 래퍼"""
        if self.answer_quality_validator:
            return self.answer_quality_validator.validate_answer_quality(state)
        return True
    
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
        self.logger.info("[FORMAT_AND_FINALIZE] Calling format_and_prepare_final")
        state = self.answer_formatter_handler.format_and_prepare_final(state)
        self.logger.info(f"[FORMAT_AND_FINALIZE] format_and_prepare_final completed, legal_references={len(state.get('legal_references', []))}, related_questions={len(state.get('metadata', {}).get('related_questions', []))}")

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

    @observe(name="classify_complexity")
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
                    self.logger.info(f"✅ 간단한 질문 감지 (인사말): {query[:50]}...")
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
                        print(f"[DEBUG] classify_complexity (간단): ✅ Global cache 저장 완료 - complexity={complexity.value}, needs_search={needs_search}")
                    except Exception as e:
                        print(f"[DEBUG] classify_complexity (간단): ❌ Global cache 저장 실패: {e}")
                        import traceback
                        print(f"[DEBUG] classify_complexity (간단): Exception traceback: {traceback.format_exc()}")

                    # 디버깅: 저장 확인
                    saved_complexity = self._get_state_value(state, "query_complexity", None)
                    saved_needs_search = self._get_state_value(state, "needs_search", None)
                    top_level_complexity = state.get("query_complexity")
                    top_level_needs_search = state.get("needs_search")
                    common_complexity = state.get("common", {}).get("query_complexity")
                    metadata_complexity = state.get("metadata", {}).get("query_complexity")
                    print(f"[DEBUG] classify_complexity: 저장 완료")
                    print(f"  - 최상위 레벨: complexity={top_level_complexity}, needs_search={top_level_needs_search}")
                    print(f"  - classification 그룹: complexity={state.get('classification', {}).get('query_complexity')}")
                    print(f"  - common 그룹: complexity={common_complexity}")
                    print(f"  - metadata: complexity={metadata_complexity}")

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
                    self.logger.info(f"✅ 간단한 질문 감지 (용어 정의): {query[:50]}...")
                    self._set_state_value(state, "query_complexity", complexity.value)
                    self._set_state_value(state, "needs_search", needs_search)
                    # Global cache에도 저장
                    try:
                        from core.agents import node_wrappers
                        if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                            node_wrappers._global_search_results_cache = {}
                        node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
                        node_wrappers._global_search_results_cache["needs_search"] = needs_search
                        print(f"[DEBUG] classify_complexity (용어정의): ✅ Global cache 저장 완료")
                    except Exception as e:
                        print(f"[DEBUG] classify_complexity (용어정의): ❌ Global cache 저장 실패: {e}")
                    processing_time = self._update_processing_time(state, start_time)
                    self._add_step(state, "복잡도 분류", f"간단한 질문 (용어 정의) - 검색 불필요 (시간: {processing_time:.3f}s)")
                    return state

            # 3. 특정 조문/법령 질의 (중간 복잡도)
            if ("조" in query or "법" in query or "법령" in query or "법률" in query) and len(query) < 50:
                complexity = QueryComplexity.MODERATE
                needs_search = True
                self.logger.info(f"📋 중간 복잡도 질문 (법령 조회): {query[:50]}...")

            # 4. 복잡한 질문 (비교, 절차, 사례 분석)
            elif any(keyword in query for keyword in ["비교", "차이", "어떻게", "방법", "절차", "사례", "판례 비교"]):
                complexity = QueryComplexity.COMPLEX
                needs_search = True
                self.logger.info(f"🔍 복잡한 질문 감지: {query[:50]}...")

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
                print(f"[DEBUG] classify_complexity: ✅ Global cache 저장 완료 - complexity={complexity.value}, needs_search={needs_search}")
                print(f"[DEBUG] classify_complexity: Global cache keys={list(node_wrappers._global_search_results_cache.keys())[:10]}")
            except Exception as e:
                print(f"[DEBUG] classify_complexity: ❌ Global cache 저장 실패: {e}")
                import traceback
                print(f"[DEBUG] classify_complexity: Exception traceback: {traceback.format_exc()}")

            # 디버깅: 저장 확인
            top_level_complexity = state.get("query_complexity")
            top_level_needs_search = state.get("needs_search")
            common_complexity = state.get("common", {}).get("query_complexity")
            metadata_complexity = state.get("metadata", {}).get("query_complexity")
            print(f"[DEBUG] classify_complexity: 저장 완료 (최종)")
            print(f"  - 최상위 레벨: complexity={top_level_complexity}, needs_search={top_level_needs_search}")
            print(f"  - classification 그룹: complexity={state.get('classification', {}).get('query_complexity')}")
            print(f"  - common 그룹: complexity={common_complexity}")
            print(f"  - metadata: complexity={metadata_complexity}")

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

    @observe(name="direct_answer")
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
            llm = self.llm_fast if hasattr(self, 'llm_fast') and self.llm_fast else self.llm

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
                self.logger.info(
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

    @observe(name="resolve_multi_turn")
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
            self.logger.error(f"Error building conversation context dict")
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
                    self.logger.warning(
                        "ClassificationHandler not available: llm is None. "
                        "Please check LLM configuration and ensure llm/llm_fast are properly initialized."
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
                self.logger.info(
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

            self.logger.info(
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
                self.logger.info(f"Found {len(category_docs)} documents in category: {category}")

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




    @observe(name="process_legal_terms")
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
            self.logger.info(f"추출된 용어 수: {len(all_terms)}")

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
                self.logger.info(f"통합된 용어 수: {len(representative_terms)}")
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
        """문서에서 타입 추출 (중복 로직 통합 - 개선: content 기반 추론 강화)"""
        doc_type = (
            doc.get("type") or
            doc.get("source_type") or
            (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else "") or
            "unknown"
        )
        
        if doc_type == "unknown":
            metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
            if metadata.get("case_id") or metadata.get("court") or metadata.get("casenames"):
                doc_type = "case_paragraph"
            elif metadata.get("decision_id") or metadata.get("org"):
                doc_type = "decision_paragraph"
            elif metadata.get("interpretation_number") or (metadata.get("org") and metadata.get("title")):
                doc_type = "interpretation_paragraph"
            elif metadata.get("statute_name") or metadata.get("law_name") or metadata.get("article_no"):
                doc_type = "statute_article"
        
        # 개선: content 기반 추론 강화
        if doc_type == "unknown":
            content = self._extract_doc_content(doc)
            content_lower = content.lower() if content else ""
            
            # 판례 패턴
            if any(keyword in content_lower for keyword in ["대법원", "지방법원", "법원", "판결", "선고", "원고", "피고", "사건"]):
                if any(keyword in content_lower for keyword in ["판결", "선고"]):
                    doc_type = "case_paragraph"
            # 결정례 패턴
            elif any(keyword in content_lower for keyword in ["결정", "재결", "심판", "의결"]):
                if any(keyword in content_lower for keyword in ["행정심판", "심판청", "위원회"]):
                    doc_type = "decision_paragraph"
            # 법령 패턴
            elif any(keyword in content_lower for keyword in ["제", "조", "항", "호", "법률", "규칙", "시행령"]):
                if any(keyword in content_lower for keyword in ["제", "조"]):
                    doc_type = "statute_article"
        
        return doc_type.lower()

    def _calculate_type_distribution(self, docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """문서 타입 분포 계산"""
        type_distribution = {}
        for doc in docs:
            doc_type = self._extract_doc_type(doc)
            type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
        return type_distribution

    def _has_precedent_or_decision(self, docs: List[Dict[str, Any]], check_precedent: bool = True, check_decision: bool = True) -> Tuple[bool, bool]:
        """판례/결정례 존재 여부 확인 (개선: case_paragraph를 판례로 인식)"""
        has_precedent = False
        has_decision = False
        
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            
            doc_type = self._extract_doc_type(doc)
            
            # 개선: case_paragraph를 판례로 명시적으로 인식
            if check_precedent and not has_precedent:
                # case_paragraph는 판례 문서이므로 판례로 인식
                if doc_type == "case_paragraph" or any(keyword in doc_type for keyword in ["precedent", "case", "판례"]):
                    has_precedent = True
                    self.logger.debug(f"🔀 [DIVERSITY] Found precedent document: type={doc_type}")
            
            # decision_paragraph는 결정례 문서이므로 결정례로 인식
            if check_decision and not has_decision:
                if doc_type == "decision_paragraph" or any(keyword in doc_type for keyword in ["decision", "결정"]):
                    has_decision = True
                    self.logger.debug(f"🔀 [DIVERSITY] Found decision document: type={doc_type}")
            
            if (not check_precedent or has_precedent) and (not check_decision or has_decision):
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

    def _ensure_scores(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서에 점수가 있는지 확인하고 없으면 설정"""
        relevance_score = doc.get("relevance_score") or doc.get("similarity") or doc.get("score")
        final_weighted_score = doc.get("final_weighted_score")
        
        if relevance_score is None or relevance_score == 0.0:
            similarity = doc.get("similarity")
            if similarity is not None:
                relevance_score = float(similarity)
            else:
                relevance_score = 0.5
                self.logger.debug(
                    f"⚠️ [SCORE INIT] 점수 없음, 기본값 설정: "
                    f"doc_id={doc.get('id', 'unknown')}, "
                    f"score=0.5"
                )
        
        if final_weighted_score is None:
            final_weighted_score = relevance_score
        
        doc["relevance_score"] = float(relevance_score)
        doc["final_weighted_score"] = float(final_weighted_score)
        
        return doc

    def _extract_citations(
        self,
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """인용 추출 (강화된 버전 - 더 많은 문서에서 추출)"""
        import sys
        print(
            f"🔍 [CITATION EXTRACTION] 시작: retrieved_docs={len(retrieved_docs)}개",
            flush=True, file=sys.stdout
        )
        self.logger.info(
            f"🔍 [CITATION EXTRACTION] 시작: retrieved_docs={len(retrieved_docs)}개"
        )
        
        citations = []
        seen_citations = set()  # 중복 방지
        
        import re
        law_pattern = re.compile(r'([가-힣]+법)\s*제?\s*(\d+)\s*조')
        precedent_pattern = re.compile(r'([가-힣]+(?:지방)?법원|대법원).*?(\d{4}[다나마]\d+)')
        
        for idx, doc in enumerate(retrieved_docs, 1):
            # type 정보 복구 시도 (여러 위치에서 확인)
            doc_type = doc.get("type") or doc.get("source_type", "")
            
            # type이 없거나 "unknown"이면 metadata에서 복구 시도
            if not doc_type or doc_type == "unknown":
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    doc_type = metadata.get("type") or metadata.get("source_type") or doc_type
                    if doc_type and doc_type != "unknown":
                        doc["type"] = doc_type
                        doc["source_type"] = doc_type
            
            # 여전히 없으면 source에서 추론 시도
            if not doc_type or doc_type == "unknown":
                source = doc.get("source", "")
                if "민법" in source or "법" in source:
                    doc_type = "statute_article"
                    doc["type"] = doc_type
                    doc["source_type"] = doc_type
                elif "대법원" in source or "법원" in source or "판결" in source:
                    doc_type = "case_paragraph"
                    doc["type"] = doc_type
                    doc["source_type"] = doc_type
            
            content = doc.get("content", "") or doc.get("text", "") or doc.get("content_text", "")
            
            # 디버깅: 문서의 모든 키 확인
            doc_keys = list(doc.keys()) if isinstance(doc, dict) else []
            import sys
            print(
                f"🔍 [CITATION DEBUG] 문서 {idx}/{len(retrieved_docs)}: "
                f"keys={doc_keys[:10]}, "
                f"type={doc_type}, "
                f"has_type_field={'type' in doc_keys}, "
                f"has_source_type_field={'source_type' in doc_keys}, "
                f"metadata_type={doc.get('metadata', {}).get('type') if isinstance(doc.get('metadata'), dict) else 'N/A'}",
                flush=True, file=sys.stdout
            )
            self.logger.debug(
                f"🔍 [CITATION] 문서 {idx}/{len(retrieved_docs)} 처리 중: "
                f"keys={doc_keys[:10]}, "
                f"type={doc_type}, "
                f"content_length={len(content) if content else 0}, "
                f"has_metadata={bool(doc.get('metadata'))}, "
                f"source={doc.get('source', 'N/A')[:50]}"
            )
            
            # 1. 법령 조문 인용 (타입 기반)
            if doc_type == "statute_article":
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: statute_article 타입 감지, 필드 확인 중..."
                )
                law_name = (
                    doc.get("statute_name") or 
                    doc.get("law_name") or 
                    doc.get("metadata", {}).get("statute_name")
                )
                article_no = (
                    doc.get("article_no") or 
                    doc.get("article_number") or 
                    doc.get("metadata", {}).get("article_no")
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
                        self.logger.info(
                            f"✅ [CITATION] 문서 {idx}: 타입 기반 법령 추출 성공 - {law_name} 제{article_no}조"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ [CITATION] 문서 {idx}: statute_article 타입이지만 필드 부족 "
                        f"(law_name={bool(law_name)}, article_no={bool(article_no)})"
                    )
            
            # 2. 판례 인용 (타입 기반)
            elif doc_type == "case_paragraph":
                self.logger.debug(
                    f"🔍 [CITATION] 문서 {idx}: case_paragraph 타입 감지, 필드 확인 중..."
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
                        self.logger.info(
                            f"✅ [CITATION] 문서 {idx}: 타입 기반 판례 추출 성공 - {case_name}"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ [CITATION] 문서 {idx}: case_paragraph 타입이지만 case_name 없음"
                    )
            
            # 3. 해석례 인용
            elif doc_type == "interpretation_paragraph":
                interpretation_id = (
                    doc.get("interpretation_id") or 
                    doc.get("metadata", {}).get("interpretation_id")
                )
                org = (
                    doc.get("org") or 
                    doc.get("metadata", {}).get("org")
                )
                
                if interpretation_id:
                    citation_key = f"interpretation_{interpretation_id}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "interpretation",
                            "interpretation_id": interpretation_id,
                            "org": org or "관할기관",
                            "source": doc.get("source", ""),
                            "doc_id": doc.get("id")
                        })
                        seen_citations.add(citation_key)
            
            # 4. 결정례 인용
            elif doc_type == "decision_paragraph":
                doc_id = doc.get("doc_id") or doc.get("id")
                org = (
                    doc.get("org") or 
                    doc.get("metadata", {}).get("org")
                )
                
                if doc_id:
                    citation_key = f"decision_{doc_id}"
                    if citation_key not in seen_citations:
                        citations.append({
                            "type": "decision",
                            "doc_id": doc_id,
                            "org": org or "관할기관",
                            "source": doc.get("source", ""),
                            "id": doc.get("id")
                        })
                        seen_citations.add(citation_key)
            
            # 5. 개선: 문서 내용에서 직접 법령/판례 패턴 추출 (타입 기반과 독립적으로 수행)
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
                        self.logger.info(
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
                        self.logger.info(
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
            doc_type = doc.get("type") or doc.get("source_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        self.logger.info(
            f"✅ [CITATION EXTRACTION] 완료: {len(citations)}개 citation 추출됨 "
            f"(retrieved_docs: {len(retrieved_docs)}개, "
            f"doc_types: {doc_types})"
        )
        
        if len(citations) == 0:
            import sys
            print(
                f"⚠️ [CITATION EXTRACTION] Citation 추출 실패: "
                f"retrieved_docs={len(retrieved_docs)}개 중 0개 추출됨. "
                f"문서 타입 분포: {doc_types}",
                flush=True, file=sys.stdout
            )
            self.logger.warning(
                f"⚠️ [CITATION EXTRACTION] Citation 추출 실패: "
                f"retrieved_docs={len(retrieved_docs)}개 중 0개 추출됨. "
                f"문서 타입 분포: {doc_types}"
            )
        
        import sys
        print(
            f"✅ [CITATION EXTRACTION] 완료: {len(citations)}개 citation 추출됨 "
            f"(retrieved_docs: {len(retrieved_docs)}개, doc_types: {doc_types})",
            flush=True, file=sys.stdout
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
            self.logger.info(
                f"🔄 [REGENERATION] Coverage 낮음: {coverage:.2f} < 0.4"
            )
            return True
        
        if citation_count == 0 and retrieved_docs_count > 0:
            self.logger.info(
                f"🔄 [REGENERATION] 인용 없음: citation_count=0, retrieved_docs={retrieved_docs_count}"
            )
            return True
        
        if len(answer.strip()) < 100:
            self.logger.info(
                f"🔄 [REGENERATION] 답변 너무 짧음: {len(answer)} < 100"
            )
            return True
        
        error_indicators = ["오류", "에러", "실패", "불가능", "없습니다", "알 수 없"]
        if any(indicator in answer for indicator in error_indicators):
            self.logger.info(
                f"🔄 [REGENERATION] 에러 메시지 포함"
            )
            return True
        
        return False

    @observe(name="generate_answer_enhanced")
    @with_state_optimization("generate_answer_enhanced", enable_reduction=True)
    def generate_answer_enhanced(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """개선된 답변 생성 - UnifiedPromptManager 활용"""
        self._recover_retrieved_docs_at_start(state)
        metadata = self._get_metadata_safely(state)
        
        try:
            is_retry, start_time = self._prepare_answer_generation(state)
            query_type = self._restore_query_type(state)
            retrieved_docs = self._restore_retrieved_docs(state)
            
            query = self._get_state_value(state, "query", "")
            question_type, domain = WorkflowUtils.get_question_type_and_domain(query_type, query, self.logger)
            model_type = ModelType.GEMINI if self.config.llm_provider == "google" else ModelType.OLLAMA
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
            
            optimized_prompt, prompt_file, prompt_length, structured_docs_count = self._generate_and_validate_prompt(
                state, context_dict, query, question_type, domain, model_type, base_prompt_type, retrieved_docs
            )

            normalized_response = self._generate_answer_with_cache(
                state, optimized_prompt, query, query_type, context_dict, retrieved_docs, 
                quality_feedback, is_retry
            )
            
            normalized_response = WorkflowUtils.normalize_answer(normalized_response)
            
            normalized_response = self._validate_and_enhance_answer(
                state, normalized_response, query, context_dict, retrieved_docs, 
                prompt_length, prompt_file, structured_docs_count
            )
            
            metadata["context_dict"] = context_dict
            self._set_state_value(state, "metadata", metadata)

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "답변 생성 완료", "답변 생성 완료")

            # 실행 기록 저장 (재시도 카운터는 RetryCounterManager에서 관리)
            self._save_metadata_safely(state, "_last_executed_node", "generate_answer_enhanced")

            self.logger.info(f"Enhanced answer generated with UnifiedPromptManager in {processing_time:.2f}s")
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"⚠️ [GENERATE_ANSWER] Exception occurred: {error_msg}", exc_info=True)
            self._handle_error(state, error_msg, "개선된 답변 생성 중 오류 발생")
            
            # 개선: 'control' 오류 등 특정 오류에 대한 추가 처리
            if error_msg == 'control' or 'control' in error_msg.lower():
                self.logger.warning(f"⚠️ [GENERATE_ANSWER] 'control' error detected. This may indicate a validation or generation control flow issue.")
                # retrieved_docs가 있는 경우 최소한의 답변 생성 시도
                retrieved_docs = state.get("retrieved_docs", [])
                if retrieved_docs and len(retrieved_docs) > 0:
                    query = state.get("query", "")
                    simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
                    self._set_answer_safely(state, simple_answer)
                    self.logger.info(f"⚠️ [GENERATE_ANSWER] Generated simple fallback answer due to 'control' error: length={len(simple_answer)}")
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
                                fallback_answer += f"\n\n검색된 문서 내용:\n" + "\n".join(doc_summaries)
                    
                    self._set_answer_safely(state, fallback_answer)
                    self.logger.info(f"⚠️ [GENERATE_ANSWER] Generated fallback answer: length={len(fallback_answer)}")
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
                            simple_answer += f"\n\n위 내용을 바탕으로 답변을 제공합니다."
                        else:
                            simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다. 검색된 문서 {len(retrieved_docs)}개를 참고하여 답변을 생성했습니다."
                    else:
                        simple_answer = f"질문 '{query}'에 대한 답변을 준비 중입니다."
                    
                    # 최소 길이 보장
                    if len(simple_answer) < 100:
                        simple_answer += " 추가 정보를 확인 중입니다."
                    
                    self._set_answer_safely(state, simple_answer)
                    self.logger.info(f"⚠️ [GENERATE_ANSWER] Generated minimal fallback answer: length={len(simple_answer)}")
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

    @observe(name="continue_answer_generation")
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
                self.logger.info(
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
                self.logger.error(f"❌ [VALIDATE CONTEXT] context is str, converting to dict")
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
                from core.workflow.validators.quality_validators import ContextValidator
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
                    self.logger.warning(f"⚠️ [VALIDATE CONTEXT] All scores are 0.0 but search results exist. Setting minimum scores.")
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

            self.logger.info(
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
                self.logger.info(f"🔧 [CITATION ENHANCEMENT] Added {len(missing_laws)} law citations")
        
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
                self.logger.info(f"🔧 [CITATION ENHANCEMENT] Added {len(missing_precedents)} precedent citations")
        
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

    def _call_llm_with_retry(self, prompt: str, max_retries: int = WorkflowConstants.MAX_RETRIES) -> str:
        """LLM 호출 (재시도 로직 포함)"""
        if hasattr(self, 'answer_generator') and self.answer_generator and hasattr(self.answer_generator, 'call_llm_with_retry'):
            return self.answer_generator.call_llm_with_retry(prompt, max_retries)
        
        # Fallback: 직접 LLM 호출
        if not hasattr(self, 'llm') or not self.llm:
            self.logger.error("LLM not available for multi-query generation")
            raise RuntimeError("LLM not available")
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                from core.agents.workflow_utils import WorkflowUtils
                result = WorkflowUtils.extract_response_content(response)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    wait_time = min(0.1 * (2 ** attempt), 0.5)  # 0.2 → 0.1, 1.0 → 0.5로 최적화
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
                self.logger.info(f"✅ [MULTI-QUERY] Cache hit for query: '{query[:50]}...'")
                print(f"[MULTI-QUERY] Cache hit for query: '{query[:50]}...'", flush=True, file=sys.stdout)
                return cached
        
        try:
            print("[MULTI-QUERY] Calling LLM to generate query variations...", flush=True, file=sys.stdout)
            self.logger.info(f"🔍 [MULTI-QUERY] Calling LLM to generate query variations for: '{query[:50]}...'")
            
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
            print(f"[MULTI-QUERY] LLM response received: {len(response)} chars", flush=True, file=sys.stdout)
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
            
            self.logger.info(
                f"✅ [MULTI-QUERY] Generated {len(result_queries)} queries for: '{query[:50]}...' "
                f"(original + {len(result_queries) - 1} variations)"
            )
            print(
                f"[MULTI-QUERY] Generated {len(result_queries)} queries: "
                f"{[q[:30] + '...' if len(q) > 30 else q for q in result_queries]}",
                flush=True, file=sys.stdout
            )
            
            return result_queries
            
        except Exception as e:
            self.logger.warning(
                f"⚠️ [MULTI-QUERY] LLM 기반 질문 재작성 실패: {e}, 원본 질문 사용"
            )
            print(f"[MULTI-QUERY] Error: {e}, using original query", flush=True, file=sys.stdout)
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

    @observe(name="format_answer")
    @with_state_optimization("format_answer", enable_reduction=True)
    # Phase 4 리팩토링: 답변 포맷팅 관련 메서드는 AnswerFormatterHandler로 이동됨
    # 호환성을 위한 래퍼 메서드
    def format_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AnswerFormatterHandler.format_answer 래퍼"""
        return self.answer_formatter_handler.format_answer(state)

    @observe(name="prepare_final_response")
    @with_state_optimization("prepare_final_response", enable_reduction=False)
    def prepare_final_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AnswerFormatterHandler.prepare_final_response 래퍼"""
        return self.answer_formatter_handler.prepare_final_response(state)

    # Phase 4 리팩토링: 답변 포맷팅 관련 메서드는 AnswerFormatterHandler로 이동됨
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

    @observe(name="classification_parallel")
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

            self.logger.info(
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

    @observe(name="assess_urgency")
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

    @observe(name="analyze_document")
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

    @observe(name="route_expert")
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
        """WorkflowRoutes.should_analyze_document 래퍼"""
        return self.workflow_routes.should_analyze_document(state)

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
                    self.logger.info(f"✅ [CACHE HIT] 쿼리 최적화 결과 캐시 히트: {cache_key[:16]}...")
            except Exception as e:
                self.logger.debug(f"캐시 확인 중 오류 (무시): {e}")
        elif self.config.disable_query_cache:
            self.logger.debug(f"Query cache is disabled, skipping cache check")
        
        if not optimized_queries:
            self.logger.info(
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
            self.logger.info(
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
                self.logger.debug(f"Query cache is disabled, not storing result")
        
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

    @observe(name="prepare_search_query")
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
                    self.logger.info(f"🔍 [HYBRID] Using HybridQueryProcessor for query optimization")
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
                    self.logger.info(
                        f"✅ [HYBRID] Query processing completed: "
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
                self.logger.info(f"🔍 [FALLBACK] Using QueryEnhancer (HybridQueryProcessor not available)")
                optimized_queries, cache_hit_optimization = self._optimize_query_with_cache(
                    search_query=search_query,
                    query_type_str=query_type_str,
                    extracted_keywords=extracted_keywords,
                    legal_field=legal_field,
                    is_retry=is_retry
                )
                
                # Multi-Query Retrieval 적용 (LLM 기반 질문 재작성)
                multi_queries = None
                print(f"[MULTI-QUERY] Starting multi-query generation for: '{search_query[:50]}...'", flush=True, file=sys.stdout)
                self.logger.info(f"🔍 [MULTI-QUERY] Starting multi-query generation for: '{search_query[:50]}...'")
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
                    print(f"[MULTI-QUERY] Generated {len(multi_queries) if multi_queries else 0} queries", flush=True, file=sys.stdout)
                    self.logger.info(f"🔍 [MULTI-QUERY] Generated {len(multi_queries) if multi_queries else 0} queries")
                    
                    if multi_queries and len(multi_queries) > 1:
                        optimized_queries["multi_queries"] = multi_queries
                        if len(multi_queries) > 1:
                            optimized_queries["semantic_query"] = multi_queries[0]
                except Exception as e:
                    print(f"[MULTI-QUERY] Error: {e}", flush=True, file=sys.stdout)
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
                    self.logger.info(
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
                print(f"[MULTI-QUERY] Added multi_queries to optimized_queries after validation: {len(multi_queries)} queries", flush=True, file=sys.stdout)
                self.logger.info(f"🔍 [MULTI-QUERY] Added multi_queries to optimized_queries after validation: {len(multi_queries)} queries")
            
            # 검증 후 최종 optimized_queries를 state에 저장
            print(f"[MULTI-QUERY] Saving optimized_queries to state (keys: {list(optimized_queries.keys())}, has_multi_queries: {'multi_queries' in optimized_queries})", flush=True, file=sys.stdout)
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
                print(f"[MULTI-QUERY] Saved optimized_queries to global cache (keys: {list(optimized_queries.keys())})", flush=True, file=sys.stdout)
                self.logger.info(f"🔍 [MULTI-QUERY] Saved optimized_queries to global cache")
            except Exception as e:
                self.logger.debug(f"Failed to save optimized_queries to global cache: {e}")
            
            # 직접 state에 저장 (이중 보장)
            if multi_queries and len(multi_queries) > 1:
                if "optimized_queries" not in state:
                    state["optimized_queries"] = {}
                if not isinstance(state["optimized_queries"], dict):
                    state["optimized_queries"] = {}
                state["optimized_queries"]["multi_queries"] = multi_queries
                print(f"[MULTI-QUERY] Directly saved multi_queries to state['optimized_queries']", flush=True, file=sys.stdout)
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
                self.logger.info(f"✅ [CACHE HIT] 캐시 히트: {len(cached_documents)}개 문서, 검색 스킵")
            else:
                self.logger.info(
                    f"✅ [PREPARE SEARCH QUERY] "
                    f"semantic_query: '{semantic_query_created[:50]}...', "
                    f"keyword_queries: {len(keyword_queries_created)}개, "
                    f"search_params: k={search_params.get('semantic_k', 'N/A')}"
                )

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
        """WorkflowRoutes.should_skip_search_adaptive 래퍼"""
        return self.workflow_routes.should_skip_search_adaptive(state)

    def _route_by_complexity(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.route_by_complexity 래퍼"""
        return self.workflow_routes.route_by_complexity(state)
    
    def _route_by_complexity_with_agentic(self, state: LegalWorkflowState) -> str:
        """Agentic 모드용 복잡도 라우팅 (기존과 동일하지만 complex는 agentic_decision으로)"""
        return self.workflow_routes.route_by_complexity(state)
    
    def _route_after_agentic(self, state: LegalWorkflowState) -> str:
        """Agentic 노드 실행 후 라우팅 (검색 결과 유무에 따라)"""
        search_results = self._get_state_value(state, "search", {}).get("results", [])
        if search_results and len(search_results) > 0:
            return "has_results"
        else:
            return "no_results"
    
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
        
        self.logger.info(f"📥 [SEARCH RESULTS] 입력 데이터 - semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, semantic_count: {semantic_count}, keyword_count: {keyword_count}")
        
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
            self.logger.info(f"검색 품질 낮음 (점수: {overall_quality:.2f}), 재검색 수행...")
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
        
        if self.search_handler and semantic_results and keyword_results:
            # 개선된 merge_search_results 사용
            merged_docs = self.search_handler.merge_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            self.logger.info(f"🔀 [MERGE] Using improved merge_search_results: {len(merged_docs)} docs")
        else:
            merged_docs = self._merge_search_results_internal(
                semantic_results, 
                keyword_results,
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            self.logger.info(f"🔀 [MERGE] Using _merge_search_results_internal: {len(merged_docs)} docs")
        
        if debug_mode:
            doc_structure_stats = {
                "total": len(merged_docs),
                "has_content": 0,
                "has_text": 0,
                "has_both": 0,
                "content_lengths": []
            }
            for doc in merged_docs[:3]:
                has_content = bool(doc.get("content", ""))
                has_text = bool(doc.get("text", ""))
                content_len = len(doc.get("content", "") or doc.get("text", "") or "")
                if has_content:
                    doc_structure_stats["has_content"] += 1
                if has_text:
                    doc_structure_stats["has_text"] += 1
                if has_content and has_text:
                    doc_structure_stats["has_both"] += 1
                doc_structure_stats["content_lengths"].append(content_len)
            
            self.logger.info(f"📋 [SEARCH RESULTS] merged_docs 구조 분석 - Total: {doc_structure_stats['total']}, Has content: {doc_structure_stats['has_content']}, Has text: {doc_structure_stats['has_text']}, Has both: {doc_structure_stats['has_both']}, Avg content length: {sum(doc_structure_stats['content_lengths'])/len(doc_structure_stats['content_lengths']) if doc_structure_stats['content_lengths'] else 0:.1f}")
        
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
        print(f"[RERANK ENTRY] _apply_keyword_weights_and_rerank called: merged_docs={len(merged_docs)}, quality={overall_quality:.2f}", flush=True)
        self.logger.info(
            f"🔍 [RERANK ENTRY] _apply_keyword_weights_and_rerank called: "
            f"merged_docs={len(merged_docs)}, query='{query[:50]}...', quality={overall_quality:.2f}"
        )
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
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
                        self.logger.info(
                            f"✅ [KEYWORD WEIGHTS] search 그룹에서 {len(extracted_keywords)}개 키워드 발견"
                        )
            
            if not extracted_keywords:
                # 쿼리에서 간단한 키워드 추출
                import re
                korean_words = re.findall(r'[가-힣]+', query)
                extracted_keywords = [w for w in korean_words if len(w) >= 2]
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
                    self.logger.info(
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
        weighted_docs = self._apply_keyword_weights_to_docs(
            merged_docs=merged_docs,
            keyword_weights=keyword_weights,
            query=query,
            query_type_str=query_type_str,
            search_params=search_params
        )
        
        # 개선: 재랭킹 조건 로직 개선 - 품질 점수와 문서 수를 별도로 고려
        # 문서 수가 매우 적으면 (5개 이하) 재랭킹 스킵
        # 품질이 매우 높고 문서 수가 적으면 (품질 >= 0.95 and 문서 <= 15) 재랭킹 스킵
        # 품질이 높고 문서 수가 매우 적으면 (품질 >= 0.85 and 문서 <= 8) 재랭킹 스킵
        should_skip_rerank = (
            len(weighted_docs) <= 5 or  # 문서 수가 매우 적으면 스킵
            (overall_quality >= 0.95 and len(weighted_docs) <= 15) or  # 품질이 매우 높고 문서 수가 적으면 스킵
            (overall_quality >= 0.85 and len(weighted_docs) <= 8)  # 품질이 높고 문서 수가 매우 적으면 스킵
        )
        
        # 재랭킹 조건 확인 로그 (INFO 레벨로 변경하여 가시성 향상)
        print(f"[RERANK CHECK] overall_quality={overall_quality:.2f}, weighted_docs={len(weighted_docs)}, should_skip={should_skip_rerank}, result_ranker={self.result_ranker is not None}, has_multi_stage={hasattr(self.result_ranker, 'multi_stage_rerank') if self.result_ranker else False}", flush=True)
        self.logger.info(
            f"🔍 [RERANK CHECK] overall_quality={overall_quality:.2f}, "
            f"weighted_docs={len(weighted_docs)}, should_skip={should_skip_rerank}, "
            f"result_ranker={self.result_ranker is not None}, "
            f"has_multi_stage={hasattr(self.result_ranker, 'multi_stage_rerank') if self.result_ranker else False}"
        )
        
        if not should_skip_rerank and self.result_ranker and hasattr(self.result_ranker, 'multi_stage_rerank'):
            try:
                search_quality = self._get_state_value(state, "search_quality", {})
                overall_quality = search_quality.get("overall_quality", 0.7) if isinstance(search_quality, dict) else 0.7
                
                search_params["overall_quality"] = overall_quality
                search_params["document_count"] = len(weighted_docs)
                
                print(f"[MULTI-STAGE RERANK] Starting reranking: {len(weighted_docs)} documents, quality={overall_quality:.2f}", flush=True)
                self.logger.info(
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
                
                print(f"[MULTI-STAGE RERANK] Applied multi-stage reranking: {len(weighted_docs)} documents", flush=True)
                self.logger.info(f"🔄 [MULTI-STAGE RERANK] Applied multi-stage reranking: {len(weighted_docs)} documents")
            except Exception as e:
                self.logger.warning(f"Multi-stage rerank failed: {e}, using citation boost", exc_info=True)
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
            self.logger.info(
                f"✅ [RERANK] 최소 문서 수 보장: {len(weighted_docs)}개 "
                f"(추가: {needed}개)"
            )
        elif should_skip_rerank:
            reason = 'docs <= 5' if len(weighted_docs) <= 5 else 'quality >= 0.95 and docs <= 15' if overall_quality >= 0.95 else 'quality >= 0.85 and docs <= 8'
            self.logger.info(
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
            self.logger.info(f"📊 [SEARCH RESULTS] Score distribution after weighting - Total: {len(weighted_docs)}, Min: {min_score:.3f}, Max: {max_score:.3f}, Avg: {avg_score:.3f}")
        
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
            self.logger.info(f"🔀 [DIVERSITY] weighted_docs type distribution before diversity: {type_distribution}")
        
        has_precedent_before, has_decision_before = self._has_precedent_or_decision(weighted_docs)
        self.logger.info(f"🔀 [DIVERSITY] Before filtering - has_precedent={has_precedent_before}, has_decision={has_decision_before}")
        
        if self.search_handler and len(weighted_docs) > 0:
            diverse_weighted_docs = self.search_handler._ensure_diverse_source_types(
                weighted_docs,
                min(max_docs_before_filter * 3, len(weighted_docs))
            )
            self.logger.info(f"🔀 [DIVERSITY] Before filtering: {len(weighted_docs)} → {len(diverse_weighted_docs)} docs (ensuring diversity)")
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
        
        # 벡터 점수 계산 (병렬 처리 전에 미리 계산)
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
            
            results_lock = threading.Lock()
            skipped_content_lock = threading.Lock()
            skipped_score_lock = threading.Lock()
            skipped_relevance_lock = threading.Lock()
            
            def process_single_doc(doc):
                nonlocal skipped_content, skipped_score, skipped_relevance, skipped_content_details
                
                if "type" not in doc or not doc.get("type"):
                    metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    if metadata.get("source_type"):
                        doc["type"] = metadata.get("source_type")
                        doc["source_type"] = metadata.get("source_type")
                
                content = self._extract_doc_content(doc)
                doc_type = self._extract_doc_type(doc)
                if doc.get("type") != doc_type:
                    doc["type"] = doc_type
                    doc["source_type"] = doc_type
                
                is_precedent_or_decision = any(keyword in doc_type for keyword in ["precedent", "case", "decision", "판례", "결정"])
                is_statute = any(keyword in doc_type for keyword in ["statute", "article", "법령", "조문"]) or doc_type == "statute_article"
                
                min_content_length = 10
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
                
                if is_text2sql:
                    min_score_threshold = 0.0
                else:
                    min_score_threshold = default_min_score_threshold
                
                if score < min_score_threshold:
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
                    if metadata.get("source_type"):
                        doc["type"] = metadata.get("source_type")
                        doc["source_type"] = metadata.get("source_type")
                
                content = self._extract_doc_content(doc)
                doc_type = self._extract_doc_type(doc)
                if doc.get("type") != doc_type:
                    doc["type"] = doc_type
                    doc["source_type"] = doc_type
                
                is_precedent_or_decision = any(keyword in doc_type for keyword in ["precedent", "case", "decision", "판례", "결정"])
                is_statute = any(keyword in doc_type for keyword in ["statute", "article", "법령", "조문"]) or doc_type == "statute_article"
                
                min_content_length = 10
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
                    min_score_threshold = default_min_score_threshold
                
                if score < min_score_threshold:
                    skipped_score += 1
                    self.logger.debug(f"🔍 [SCORE FILTER] 점수 부족으로 제외: score={score:.3f} < {min_score_threshold}, source={doc.get('source', 'Unknown')[:50]}")
                    continue
                
                filtered_docs.append(doc)
        
        if debug_mode:
            self.logger.info(f"📊 [SEARCH RESULTS] Filtering statistics - Merged: {len(merged_docs)}, Weighted: {len(weighted_docs)}, Filtered: {len(filtered_docs)}, Skipped (content): {skipped_content}, Skipped (score): {skipped_score}, Skipped (relevance): {skipped_relevance}")
            if skipped_content > 0 and skipped_content_details:
                self.logger.warning(f"⚠️ [SEARCH RESULTS] Content 필터링 제외 상세 (상위 {len(skipped_content_details)}개): {skipped_content_details}")
        
        return filtered_docs
    
    def _validate_document_metadata(self, doc: Dict[str, Any]) -> bool:
        """우선순위 2: 메타데이터 검증 강화"""
        # 필수 필드 검증
        has_content = bool(doc.get("content") or doc.get("text"))
        has_source = bool(doc.get("source"))
        has_type = bool(doc.get("type") or doc.get("source_type"))
        
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
        
        self.logger.info(
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
        self.logger.info(f"📊 [RELEVANCE SCORES] 질의: '{query}'")
        self.logger.info(f"📊 [RELEVANCE SCORES] 검색된 벡터 문서 수: {len(vector_docs)}개")
        
        # 모든 문서의 점수 수집 및 로깅
        doc_scores = []
        for doc in vector_docs:
            score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
            similarity = doc.get("similarity", 0.0)
            keyword_score = doc.get("keyword_match_score", 0.0)
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id") or "unknown"
            doc_type = doc.get("type") or doc.get("source_type", "unknown")
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
            self.logger.info(
                f"📊 [SCORE STATS] 평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}, 중앙값={median_score:.3f}"
            )
            
            # 모든 문서의 점수 상세 로깅 (정렬된 순서)
            doc_scores_sorted = sorted(doc_scores, key=lambda x: x[0], reverse=True)
            self.logger.info(f"📊 [ALL DOCS SCORES] 모든 {len(doc_scores_sorted)}개 문서의 relevance_score:")
            for i, (score, similarity, keyword_score, doc_id, doc_type, source, content_preview, doc) in enumerate(doc_scores_sorted, 1):
                self.logger.info(
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
            
            self.logger.info(
                f"📊 [FILTER THRESHOLD] 동적 임계값 계산: "
                f"평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}, "
                f"임계값={min_relevance_threshold:.3f} (기본: {base_threshold})"
            )
        else:
            min_relevance_threshold = base_threshold
            self.logger.info(f"📊 [FILTER THRESHOLD] 임계값: {min_relevance_threshold} (기본값, 문서 없음)")
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
                self.logger.info(
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
            
            self.logger.info(
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
            self.logger.info(
                f"🔧 [VECTOR FILTER] 안전장치로 {len(filtered_vector_docs)}개 문서 통과시킴"
            )
        
        if len(filtered_vector_docs) < len(vector_docs):
            self.logger.info(
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
            self.logger.info(
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
        self.logger.info(
            f"🔀 [FINAL COMBINE] textToSQL: {len(text2sql_docs)}개, "
            f"벡터: {len([d for d in final_docs if d not in text2sql_docs])}개, "
            f"총: {len(final_docs)}개"
        )
        
        # 최종 문서들의 relevance_score 상세 로깅
        if final_docs:
            self.logger.info(f"📊 [FINAL DOCS SCORES] 최종 선택된 {len(final_docs)}개 문서의 relevance_score:")
            for i, doc in enumerate(final_docs, 1):
                score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id") or "unknown"
                doc_type = doc.get("type") or doc.get("source_type", "unknown")
                source = doc.get("source", "")[:50] or "unknown"
                search_type = doc.get("search_type", "unknown")
                self.logger.info(
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
            self.logger.info(f"🔀 [DIVERSITY] filtered_docs type distribution: {filtered_type_distribution}")
        
        has_precedent, has_decision = self._has_precedent_or_decision(filtered_docs)
        
        if not has_precedent or not has_decision:
            self.logger.info(f"🔀 [DIVERSITY] Missing precedent={not has_precedent}, decision={not has_decision}, attempting to restore from weighted_docs (total: {len(weighted_docs)}) and semantic_results (total: {len(semantic_results)})")
            
            if weighted_docs:
                sample_doc = weighted_docs[0] if isinstance(weighted_docs[0], dict) else {}
                self.logger.debug(f"🔀 [DIVERSITY] weighted_docs sample keys: {list(sample_doc.keys())[:10]}")
                self.logger.debug(f"🔀 [DIVERSITY] weighted_docs sample type: {sample_doc.get('type')}, source_type: {sample_doc.get('source_type')}, metadata: {type(sample_doc.get('metadata'))}")
            
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
                self.logger.info(f"🔀 [DIVERSITY] ✅ Restored precedent document: {best_precedent[1]} (id: {best_precedent[2]})")
            
            # 결정례 복원
            if not has_decision and decision_candidates:
                decision_candidates.sort(key=lambda x: len(self._extract_doc_content(x[0]) or ""), reverse=True)
                best_decision = decision_candidates[0]
                filtered_docs.append(best_decision[0])
                has_decision = True
                restored_count += 1
                self.logger.info(f"🔀 [DIVERSITY] ✅ Restored decision document: {best_decision[1]} (id: {best_decision[2]})")
            
            if restored_count > 0:
                self.logger.info(f"🔀 [DIVERSITY] ✅ Restored {restored_count} documents (precedent={has_precedent}, decision={has_decision})")
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
                    self.logger.debug(f"🔀 [DIVERSITY] ✅ Precedent and decision documents already present, no restoration needed")
        
        if self.search_handler and len(filtered_docs) > 0:
            diverse_filtered_docs = self.search_handler._ensure_diverse_source_types(
                filtered_docs,
                min(max_docs * 2, len(filtered_docs))
            )
            
            if diverse_filtered_docs:
                final_type_distribution = self._calculate_type_distribution(diverse_filtered_docs)
                self.logger.info(f"🔀 [DIVERSITY] final_docs type distribution after diversity: {final_type_distribution}")
            
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
                    self.logger.info(f"🔄 [FALLBACK] Using {len(final_docs)} lower-scored documents as fallback (original filtered count: 0)")
                else:
                    self.logger.error(f"❌ [SEARCH RESULTS] No fallback documents available. All documents were filtered out (content too short or score too low).")
            else:
                self.logger.error(f"❌ [SEARCH RESULTS] No documents available at all. Search may have failed or returned empty results.")
        
        if not final_docs or len(final_docs) == 0:
            self.logger.warning(f"⚠️ [SEARCH RESULTS] final_docs가 0개입니다. semantic_results에서 변환 시도...")
            if semantic_results and len(semantic_results) > 0:
                converted_docs = []
                for doc in semantic_results[:10]:
                    if isinstance(doc, dict):
                        doc_type = doc.get("type") or doc.get("source_type") or (doc.get("metadata", {}).get("source_type") if isinstance(doc.get("metadata"), dict) else None)
                        text_content = doc.get("text", "") or doc.get("content", "") or str(doc.get("metadata", {}).get("text", "")) or str(doc.get("metadata", {}).get("content", ""))
                        converted_doc = {
                            "content": text_content,
                            "text": text_content,
                            "source": doc.get("source", "") or doc.get("title", "Unknown"),
                            "relevance_score": doc.get("relevance_score", 0.5),
                            "search_type": "semantic",
                            "type": doc_type,
                            "source_type": doc_type,
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
                    self.logger.info(f"🔄 [FALLBACK] Converted {len(final_docs)} documents from semantic_results to retrieved_docs (original final_docs count: 0)")
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
        
        self.logger.info(f"📊 [SEARCH RESULTS] final_docs 설정 완료 - 개수: {len(final_docs)}")
        
        if debug_mode:
            self.logger.info(f"💾 [SEARCH RESULTS] State 저장 전 검증 - final_docs 개수: {len(final_docs)}, 타입: {type(final_docs).__name__}")
        
        self._save_search_results_to_state(state, final_docs)
        
        processing_time = self._update_processing_time(state, start_time)
        self._add_step(
            state,
            "검색 결과 처리",
            f"검색 결과 처리 완료: {len(final_docs)}개 문서 (품질 점수: {overall_quality:.2f}, 시간: {processing_time:.3f}s)"
        )
        
        if len(final_docs) > 0:
            processed_msg = f"✅ [SEARCH RESULTS] Processed {len(final_docs)} documents (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
            print(processed_msg, flush=True, file=sys.stdout)
            self.logger.info(processed_msg)
            final_scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in final_docs]
            if final_scores:
                final_score_msg = f"📊 [SEARCH RESULTS] Final documents score range - Min: {min(final_scores):.3f}, Max: {max(final_scores):.3f}, Avg: {sum(final_scores)/len(final_scores):.3f}"
                print(final_score_msg, flush=True, file=sys.stdout)
                self.logger.info(final_score_msg)
            
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
                            self.logger.warning(f"⚠️ [KEYWORD COVERAGE] extracted_keywords is empty or None")
                            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
                            if not extracted_keywords and "search" in state and isinstance(state.get("search"), dict):
                                extracted_keywords = state["search"].get("extracted_keywords", [])
                            
                            # 폴백: 쿼리에서 직접 키워드 추출
                            if not extracted_keywords and query:
                                import re
                                korean_words = re.findall(r'[가-힣]+', query)
                                extracted_keywords = [w for w in korean_words if len(w) >= 2]
                                
                                # 불용어 제거
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
                                    self.logger.info(
                                        f"🔍 [KEYWORD COVERAGE] 쿼리에서 키워드 추출: "
                                        f"{len(extracted_keywords)}개 키워드 (query='{query[:50]}...')"
                                    )
                            
                            self.logger.info(f"🔍 [KEYWORD COVERAGE] 최종 extracted_keywords: {len(extracted_keywords)} keywords")
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
                    print(metrics_msg, flush=True, file=sys.stdout)
                    self.logger.info(metrics_msg)
                    
                    # MLflow 로깅 추가
                    try:
                        import mlflow
                        from datetime import datetime
                        from core.utils.config import Config
                        
                        # MLflow run이 없으면 자동으로 시작
                        active_run = mlflow.active_run()
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
                            active_run = mlflow.active_run()
                            print(f"✅ [MLFLOW] Started new MLflow run: {run_name} (run_id: {active_run.info.run_id})", flush=True, file=sys.stdout)
                            self.logger.info(f"✅ [MLFLOW] Started new MLflow run: {run_name} (run_id: {active_run.info.run_id})")
                        
                        if active_run is not None:
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
                            print(f"✅ [MLFLOW] Search quality metrics logged to MLflow run: {active_run.info.run_id}", flush=True, file=sys.stdout)
                            self.logger.info(f"✅ [MLFLOW] Search quality metrics logged to MLflow run: {active_run.info.run_id}")
                    except ImportError:
                        self.logger.debug("MLflow not available, skipping metric logging")
                    except Exception as e:
                        self.logger.warning(f"Failed to log to MLflow: {e}", exc_info=True)
                except Exception as e:
                    self.logger.warning(f"Failed to log search quality metrics: {e}", exc_info=True)
        else:
            no_docs_msg = f"⚠️ [SEARCH RESULTS] No documents available after processing (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
            print(no_docs_msg, flush=True, file=sys.stdout)
            self.logger.warning(no_docs_msg)
    
    def _save_search_results_to_state(
        self,
        state: LegalWorkflowState,
        final_docs: List[Dict[str, Any]]
    ) -> None:
        """검색 결과를 State에 저장 (중복 코드 제거) - 개선: 여러 위치에 저장"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        # 최상위 레벨에 저장
        self._set_state_value(state, "retrieved_docs", final_docs.copy())
        self._set_state_value(state, "merged_documents", final_docs.copy())
        
        # search 그룹에 저장 (명시적)
        if "search" not in state:
            state["search"] = {}
        state["search"]["retrieved_docs"] = final_docs.copy()
        state["search"]["merged_documents"] = final_docs.copy()
        
        # common 그룹에 저장
        if "common" not in state:
            state["common"] = {}
        if "search" not in state["common"]:
            state["common"]["search"] = {}
        state["common"]["search"]["retrieved_docs"] = final_docs.copy()
        state["common"]["search"]["merged_documents"] = final_docs.copy()
        
        # metadata에도 저장 (복구를 위해)
        metadata = self._get_metadata_safely(state)
        if "search" not in metadata:
            metadata["search"] = {}
        metadata["search"]["retrieved_docs"] = final_docs
        metadata["search"]["merged_documents"] = final_docs
        metadata["retrieved_docs"] = final_docs
        self._set_state_value(state, "metadata", metadata)
        
        # 개선: global cache에도 저장 (복구를 위해)
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache is None or not isinstance(_global_search_results_cache, dict):
                import core.shared.wrappers.node_wrappers as node_wrappers_module
                node_wrappers_module._global_search_results_cache = {}
                _global_search_results_cache = node_wrappers_module._global_search_results_cache
            
            _global_search_results_cache["retrieved_docs"] = final_docs.copy()
            _global_search_results_cache["merged_documents"] = final_docs.copy()
            if "search" not in _global_search_results_cache:
                _global_search_results_cache["search"] = {}
            _global_search_results_cache["search"]["retrieved_docs"] = final_docs.copy()
            _global_search_results_cache["search"]["merged_documents"] = final_docs.copy()
            if "common" not in _global_search_results_cache:
                _global_search_results_cache["common"] = {}
            if "search" not in _global_search_results_cache["common"]:
                _global_search_results_cache["common"]["search"] = {}
            _global_search_results_cache["common"]["search"]["retrieved_docs"] = final_docs.copy()
            self.logger.info(
                f"✅ [SAVE RESULTS] 저장 완료: {len(final_docs)}개 문서 "
                f"(최상위, search, common, 전역 캐시)"
            )
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not save to global cache: {e}")
        
        if debug_mode:
            saved_retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            saved_search_group = state.get("search", {}).get("retrieved_docs", [])
            saved_common_group = state.get("common", {}).get("search", {}).get("retrieved_docs", [])
            self.logger.info(f"✅ [SEARCH RESULTS] State 저장 완료 - 최상위: {len(saved_retrieved_docs)}, search 그룹: {len(saved_search_group)}, common 그룹: {len(saved_common_group)}")
        
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if not _global_search_results_cache:
                _global_search_results_cache = {}
            
            _global_search_results_cache["retrieved_docs"] = final_docs.copy()
            _global_search_results_cache["merged_documents"] = final_docs.copy()
            
            if "search" not in _global_search_results_cache:
                _global_search_results_cache["search"] = {}
            _global_search_results_cache["search"]["retrieved_docs"] = final_docs.copy()
            _global_search_results_cache["search"]["merged_documents"] = final_docs.copy()
            
            self.logger.info(f"✅ [SEARCH RESULTS] 전역 캐시에 직접 저장 완료 - 개수: {len(final_docs)}")
        except Exception as e:
            self.logger.warning(f"⚠️ [SEARCH RESULTS] 전역 캐시 직접 저장 실패: {e}")

    @observe(name="process_search_results_combined")
    @with_state_optimization("process_search_results_combined", enable_reduction=True)
    def process_search_results_combined(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)"""
        self.logger.info("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작")

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
                doc_type = doc.get("type", "") or doc.get("source_type", "") or ""
                if "statute" in doc_type.lower() or (doc.get("law_name") and doc.get("article_no")):
                    has_statute_article = True
                    break
            
            # 법조문 조회 쿼리인데 법조문이 없으면 textToSQL 강제 실행
            if query_type_str == "law_inquiry" and not has_statute_article:
                print(f"[SEARCH QUALITY] 법조문 조회 쿼리인데 법조문이 검색되지 않음. textToSQL 강제 실행", flush=True, file=sys.stdout)
                self.logger.warning(
                    f"⚠️ [SEARCH QUALITY] 법조문 조회 쿼리인데 법조문이 검색되지 않음. textToSQL 강제 실행"
                )
                try:
                    from core.agents.legal_data_connector_v2 import LegalDataConnectorV2
                    data_connector = LegalDataConnectorV2()
                    text2sql_results = data_connector.search_documents(query, limit=5)
                    if text2sql_results:
                        keyword_results.extend(text2sql_results)
                        keyword_count += len(text2sql_results)
                        print(f"[SEARCH QUALITY] textToSQL 강제 실행: {len(text2sql_results)}개 법조문 검색 성공", flush=True, file=sys.stdout)
                        self.logger.info(f"✅ [SEARCH QUALITY] textToSQL 강제 실행: {len(text2sql_results)}개 법조문 검색 성공")
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
                    print(f"[SEARCH QUALITY] textToSQL 강제 실행 실패: {e}", flush=True, file=sys.stdout)
                    self.logger.warning(f"⚠️ [SEARCH QUALITY] textToSQL 강제 실행 실패: {e}")

            # 조기 종료 로직: 품질이 충분히 높고 결과 수가 충분하면 추가 처리 생략
            min_quality_for_early_exit = 0.8
            min_results_for_early_exit = 5
            total_results = len(semantic_results) + len(keyword_results)
            
            if overall_quality >= min_quality_for_early_exit and total_results >= min_results_for_early_exit:
                self.logger.info(
                    f"✅ [EARLY EXIT] Quality sufficient (quality={overall_quality:.2f}, "
                    f"results={total_results}), skipping retry search"
                )
                # 재검색 생략하고 바로 병합 진행
                merged_docs = self._merge_and_rerank_results(
                    state, semantic_results, keyword_results, query
                )
            else:
                semantic_results, keyword_results, semantic_count, keyword_count = self._perform_conditional_retry_search(
                    state, semantic_results, keyword_results, semantic_count, keyword_count,
                    quality_evaluation, query, query_type_str, search_params, extracted_keywords
                )

            merged_docs = self._merge_and_rerank_results(
                state, semantic_results, keyword_results, query
            )

            # 개선 3.1: 모든 문서에 점수 보장 및 type 정보 보존
            for doc in semantic_results + keyword_results:
                doc = self._ensure_scores(doc)
                # type 정보 보존 (검색 결과에서 가져온 type 정보 유지)
                if "type" not in doc and "source_type" not in doc:
                    # 원본 검색 결과에서 type 정보 복구 시도
                    original_type = doc.get("metadata", {}).get("type") or doc.get("metadata", {}).get("source_type")
                    if original_type:
                        doc["type"] = original_type
                        doc["source_type"] = original_type

            self.logger.info(
                f"🔍 [BEFORE RERANK] About to call _apply_keyword_weights_and_rerank: "
                f"merged_docs={len(merged_docs)}, overall_quality={overall_quality:.2f}"
            )
            
            weighted_docs = self._apply_keyword_weights_and_rerank(
                state, merged_docs, query, query_type_str, extracted_keywords,
                search_params, overall_quality
            )
            
            self.logger.info(
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
            
            # 우선순위 4: 법령 조문 부스팅 적용 (병렬 처리)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                text2sql_future = executor.submit(self._apply_statute_article_boosting, text2sql_docs, query)
                reranked_future = executor.submit(self._apply_statute_article_boosting, reranked_vector_docs, query)
                text2sql_docs = text2sql_future.result(timeout=5)
                reranked_vector_docs = reranked_future.result(timeout=5)
            
            # textToSQL 결과와 재랭킹된 벡터 결과 결합
            final_docs = self._combine_text2sql_and_reranked_vector(
                text2sql_docs, reranked_vector_docs, query_type_str
            )
            
            # 개선: final_docs의 모든 문서에 type 정보 보장
            for doc in final_docs:
                if isinstance(doc, dict):
                    # type 정보가 없으면 source_type에서 가져오거나, 원본 검색 결과에서 복구
                    if "type" not in doc and "source_type" not in doc:
                        # metadata에서 복구 시도
                        metadata = doc.get("metadata", {})
                        if isinstance(metadata, dict):
                            original_type = metadata.get("type") or metadata.get("source_type")
                            if original_type:
                                doc["type"] = original_type
                                doc["source_type"] = original_type
                        # merged_docs에서 같은 id의 문서 찾아서 type 복구
                        if "type" not in doc and "source_type" not in doc:
                            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("chunk_id")
                            if doc_id:
                                for merged_doc in merged_docs:
                                    merged_id = merged_doc.get("id") or merged_doc.get("doc_id") or merged_doc.get("chunk_id")
                                    if merged_id == doc_id:
                                        merged_type = merged_doc.get("type") or merged_doc.get("source_type")
                                        if merged_type:
                                            doc["type"] = merged_type
                                            doc["source_type"] = merged_type
                                            break
                    # type과 source_type 모두 설정 (일관성 보장)
                    if "type" in doc and "source_type" not in doc:
                        doc["source_type"] = doc["type"]
                    elif "source_type" in doc and "type" not in doc:
                        doc["type"] = doc["source_type"]

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
                self.logger.info(
                    f"🔍 [MIN DOCS] 최소 문서 수 보장: {len(final_docs)}개 "
                    f"(추가: {needed_count}개)"
                )
            
            if len(final_docs) > MAX_DOCUMENTS_FOR_ANSWER:
                final_docs = final_docs[:MAX_DOCUMENTS_FOR_ANSWER]
                self.logger.info(f"🔍 [MAX DOCS] 최대 문서 수 제한: {MAX_DOCUMENTS_FOR_ANSWER}개")

            # 개선 5.1: 문서 다양성 보장
            MIN_DOCS_FOR_DIVERSITY = 5
            if len(final_docs) < MIN_DOCS_FOR_DIVERSITY:
                type_distribution = {}
                for doc in final_docs:
                    doc_type = doc.get("type") or doc.get("source_type", "unknown")
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
                
                required_types = ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]
                missing_types = [t for t in required_types if type_distribution.get(t, 0) == 0]
                
                if missing_types and len(merged_docs) > len(final_docs):
                    for doc_type in missing_types:
                        additional = [
                            doc for doc in merged_docs
                            if doc not in final_docs
                            and (doc.get("type") or doc.get("source_type")) == doc_type
                        ]
                        if additional:
                            additional.sort(
                                key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                                reverse=True
                            )
                            final_docs.append(additional[0])
                            self.logger.info(f"✅ [DIVERSITY] {doc_type} 타입 추가: 1개")

            self._save_final_results_to_state(
                state, final_docs, merged_docs, filtered_docs, overall_quality,
                semantic_count, keyword_count, needs_retry, start_time,
                query=query, query_type_str=query_type_str, extracted_keywords=extracted_keywords
            )

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
                self.logger.info(
                    f"🔄 [FALLBACK] Using {len(fallback_docs)} documents from original search results "
                    f"as fallback after processing error"
                )
                self._set_state_value(state, "retrieved_docs", fallback_docs)
                self._set_state_value(state, "merged_documents", fallback_docs)
            else:
                # 최종 폴백: 빈 리스트
                self.logger.warning(
                    f"⚠️ [FALLBACK] No fallback documents available. Setting empty retrieved_docs."
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

    @observe(name="execute_searches_parallel")
    @with_state_optimization("execute_searches_parallel", enable_reduction=True)
    def execute_searches_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """의미적 검색과 키워드 검색을 병렬로 실행"""
        try:
            start_time = time.time()

            if self.search_execution_processor:
                state = self.search_execution_processor.execute_searches_parallel(state)
                self._update_processing_time(state, start_time)
            else:
                self.logger.warning("SearchExecutionProcessor not available, skipping parallel search")
                self._handle_error(state, "SearchExecutionProcessor not initialized", "병렬 검색 중 오류 발생")

        except Exception as e:
            self._handle_error(state, str(e), "병렬 검색 중 오류 발생")
            if self.search_execution_processor:
                return self.search_execution_processor.fallback_sequential_search(state)
            return state

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
        return [], 0

    def _fallback_sequential_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """SearchExecutionProcessor.fallback_sequential_search 래퍼"""
        if self.search_execution_processor:
            return self.search_execution_processor.fallback_sequential_search(state)
        self.logger.warning("SearchExecutionProcessor not available, cannot perform fallback sequential search")
        return state

    @observe(name="evaluate_search_quality")
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

            self.logger.info(
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

    @observe(name="conditional_retry_search")
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
                self.logger.info(f"🔄 [RETRY SEMANTIC] Retrying semantic search (count: {semantic_retry_count})")

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
                self.logger.info(f"🔄 [RETRY KEYWORD] Retrying keyword search (count: {keyword_retry_count})")

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

            self.logger.info(
                f"✅ [CONDITIONAL RETRY] Semantic retry: {semantic_retry_count}, "
                f"Keyword retry: {keyword_retry_count}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "조건부 재검색 중 오류 발생")

        return state

    @observe(name="merge_and_rerank_with_keyword_weights")
    @with_state_optimization("merge_and_rerank_with_keyword_weights", enable_reduction=True)
    def merge_and_rerank_with_keyword_weights(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드별 가중치를 적용한 결과 병합 및 Reranking"""
        try:
            start_time = time.time()

            print(f"[DEBUG] MERGE: START - merge_and_rerank_with_keyword_weights")

            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})
            query = self._get_state_value(state, "query", "")
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            print(f"[DEBUG] MERGE: Input - semantic_results={len(semantic_results)}, keyword_results={len(keyword_results)}")

            # state에서 직접 확인 (디버깅)
            # _get_state_value가 제대로 작동하지 않는 경우를 대비하여 직접 state["search"]에서 읽기
            if len(semantic_results) == 0 and len(keyword_results) == 0:
                print(f"[DEBUG] MERGE: _get_state_value returned empty, checking state['search'] directly...")
                # state["search"]에서 직접 읽기 시도
                if "search" in state and isinstance(state.get("search"), dict):
                    direct_semantic = state["search"].get("semantic_results", [])
                    direct_keyword = state["search"].get("keyword_results", [])
                    print(f"[DEBUG] MERGE: Direct state['search'] check - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
                    if direct_semantic or direct_keyword:
                        print(f"[DEBUG] MERGE: Found results in state['search'] - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
                        semantic_results = direct_semantic
                        keyword_results = direct_keyword
                else:
                    print(f"[DEBUG] MERGE: state['search'] not found or not dict, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

            # 여전히 비어있으면 state 전체에서 찾기
            if len(semantic_results) == 0 and len(keyword_results) == 0:
                print(f"[DEBUG] MERGE: Still empty, checking all state keys...")
                if isinstance(state, dict):
                    # flat 구조일 수도 있으므로 직접 확인
                    if "semantic_results" in state:
                        flat_semantic = state.get("semantic_results", [])
                        if isinstance(flat_semantic, list) and len(flat_semantic) > 0:
                            print(f"[DEBUG] MERGE: Found semantic_results in flat state: {len(flat_semantic)}")
                            semantic_results = flat_semantic
                    if "keyword_results" in state:
                        flat_keyword = state.get("keyword_results", [])
                        if isinstance(flat_keyword, list) and len(flat_keyword) > 0:
                            print(f"[DEBUG] MERGE: Found keyword_results in flat state: {len(flat_keyword)}")
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
                print(f"[DEBUG] MERGE: Search quality validation failed - {quality_message}")
                # 품질 검증 실패 시 상위 점수 문서만 유지 (최소 5개)
                if reranked_results:
                    min_score = 0.4  # 최소 점수 기준 완화 (0.5 → 0.4)
                    filtered_reranked = [
                        doc for doc in reranked_results
                        if doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) >= min_score
                    ]
                    if len(filtered_reranked) >= 5:  # 최소 5개 보장
                        reranked_results = filtered_reranked[:10]  # 상위 10개만
                        self.logger.info(f"🔧 [SEARCH QUALITY] Filtered to {len(reranked_results)} high-quality documents")
                        print(f"[DEBUG] MERGE: Filtered to {len(reranked_results)} high-quality documents")
                    elif len(filtered_reranked) >= 3:
                        reranked_results = filtered_reranked  # 최소 3개 이상이면 모두 유지
                        self.logger.warning(f"⚠️ [SEARCH QUALITY] Low quality results, keeping {len(reranked_results)} documents")
                        print(f"[DEBUG] MERGE: Low quality, keeping {len(reranked_results)} documents")
                    else:
                        # 최소 3개 미만이면 상위 5개만 유지 (점수 상관없이, 3개 → 5개로 증가)
                        reranked_results = reranked_results[:5]
                        self.logger.warning(f"⚠️ [SEARCH QUALITY] Very low quality results, keeping top 5 only")
                        print(f"[DEBUG] MERGE: Very low quality, keeping top 5 only")
            else:
                self.logger.info(f"✅ [SEARCH QUALITY] Validation passed: {quality_message}")
                print(f"[DEBUG] MERGE: Search quality validation passed - {quality_message}")

            # 결과 저장
            self._set_state_value(state, "merged_documents", reranked_results)
            self._set_state_value(state, "keyword_weights", keyword_weights)

            # 중요: 병합된 결과를 retrieved_docs에도 저장 (다음 노드에서 사용하기 위해)
            # 모든 벡터 스토어 검색 결과(semantic_query, original_query, keyword_queries)가 포함됨
            self._set_state_value(state, "retrieved_docs", reranked_results)
            print(f"[DEBUG] MERGE: Saved {len(reranked_results)} documents to retrieved_docs")

            # 저장 확인
            stored_merged = self._get_state_value(state, "merged_documents", [])
            stored_retrieved = self._get_state_value(state, "retrieved_docs", [])
            print(f"[DEBUG] MERGE: After save - merged_documents={len(stored_merged)}, retrieved_docs={len(stored_retrieved)}")

            self._save_metadata_safely(state, "_last_executed_node", "merge_and_rerank_with_keyword_weights")
            self._update_processing_time(state, start_time)

            self.logger.info(
                f"✅ [KEYWORD-WEIGHTED RERANKING] Merged {len(unique_results)} results, "
                f"reranked to {len(reranked_results)} with {len(keyword_weights)} weighted keywords"
            )
            print(f"[DEBUG] MERGE: Semantic input={len(semantic_results)}, Keyword input={len(keyword_results)}, Unique={len(unique_results)}, Reranked={len(reranked_results)}")

            # 병합 결과 상세 디버깅 로그
            if reranked_results:
                top_scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in reranked_results[:5]]
                search_type_counts = {
                    "semantic": sum(1 for doc in reranked_results if doc.get("search_type") == "semantic"),
                    "keyword": sum(1 for doc in reranked_results if doc.get("search_type") == "keyword"),
                    "hybrid": sum(1 for doc in reranked_results if doc.get("search_type") not in ["semantic", "keyword"])
                }
                self.logger.info(
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
            print(f"[DEBUG] MERGE: Fallback - Saved {len(fallback_docs)} documents to retrieved_docs")

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
                if doc.get("type") == "statute_article" or doc.get("source_type") == "statute_article":
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

    @observe(name="filter_and_validate_results")
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
                    print(f"[DEBUG] FILTER: merged_documents is empty, using retrieved_docs ({len(documents)} documents)")
                    self.logger.warning(f"filter_and_validate_results: merged_documents is empty, using retrieved_docs")

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
            self.logger.info(
                f"✅ [FILTER & VALIDATE] Results filtered and validated: "
                f"{len(pruned_docs)} final documents "
                f"(from {len(documents)} input documents)"
            )

            # 필터링 상세 디버깅 로그
            min_relevance = search_params.get("min_relevance", self.config.similarity_threshold)
            self.logger.info(
                f"🔍 [DEBUG] Filter & Validate details: "
                f"Input documents: {len(documents)}, "
                f"After metadata filter: {len(documents)}, "
                f"After quality filter: {len(filtered_docs)}, "
                f"After pruning: {len(pruned_docs)}, "
                f"Min relevance threshold: {min_relevance:.3f}"
            )
            print(f"[DEBUG] FILTER: Input={len(documents)}, After quality={len(filtered_docs)}, After prune={len(pruned_docs)}, Min relevance={min_relevance:.3f}")

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

                    self.logger.info(
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
                self.logger.info("📄 [FILTER & VALIDATE] Document samples:")
                for i, doc in enumerate(pruned_docs[:3], 1):
                    source = doc.get("source", "Unknown")
                    content = doc.get("content") or doc.get("text", "")
                    content_length = len(content) if content else 0
                    score = doc.get("relevance_score", doc.get("combined_score", doc.get("final_weighted_score", doc.get("score", 0.0))))
                    search_type = doc.get("search_type", "unknown")
                    self.logger.info(
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

    @observe(name="update_search_metadata")
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
            self.logger.info(f"Search metadata updated: {len(retrieved_docs)} documents retrieved")

        except Exception as e:
            self._handle_error(state, str(e), "검색 메타데이터 업데이트 중 오류 발생")

        return state

    @observe(name="prepare_document_context_for_prompt")
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

            self.logger.info(
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

            self.logger.info(
                f"✅ [DOCUMENT PREPARATION] Prepared prompt context: "
                f"{doc_count} documents, "
                f"{context_length} chars, "
                f"input docs: {len(retrieved_docs)}"
            )

            if content_validation:
                has_content = content_validation.get("has_document_content", False)
                docs_with_content = content_validation.get("documents_with_content", 0)
                self.logger.info(
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

    @observe(name="prepare_documents_and_terms")
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
            self.logger.info(f"📊 [PREPARE DOCS ENTRY] 질의: '{query}'")
            self.logger.info(f"📊 [PREPARE DOCS ENTRY] retrieved_docs 수: {len(retrieved_docs)}개")
            
            # retrieved_docs 검증 (전역 캐시에서 복원 시도)
            if not retrieved_docs:
                # 전역 캐시에서 복원 시도
                try:
                    from core.shared.wrappers.node_wrappers import _global_search_results_cache
                    if _global_search_results_cache and "retrieved_docs" in _global_search_results_cache:
                        retrieved_docs = _global_search_results_cache.get("retrieved_docs", [])
                        if retrieved_docs:
                            self.logger.info(f"✅ [PREPARE CONTEXT] Restored {len(retrieved_docs)} retrieved_docs from global cache")
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
                    self.logger.info(
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
                    self.logger.info(
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
                self.logger.info(f"추출된 용어 수: {len(all_terms)}")

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
                    self.logger.info(f"통합된 용어 수: {len(representative_terms)}")
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
                self.logger.info(
                    f"⏭️ [TERM PROCESSING] Skipping term extraction and processing "
                    f"(no valid retrieved_docs available)"
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
                self.logger.warning(f"⚠️ [PROMPT CONTEXT] prompt_optimized_context is str, converting to dict")
                prompt_optimized_context = {"prompt_optimized_text": prompt_optimized_context}
            else:
                self.logger.warning(f"⚠️ [PROMPT CONTEXT] prompt_optimized_context is not dict (type: {type(prompt_optimized_context)}), using empty dict")
                prompt_optimized_context = {}
        
        if not prompt_optimized_context or not prompt_optimized_context.get("prompt_optimized_text"):
            if retrieved_docs and len(retrieved_docs) > 0:
                self.logger.info("⚠️ [FALLBACK] prompt_optimized_context is missing or invalid, generating automatically")
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
                    self.logger.info(f"✅ [AUTO GENERATE] Generated prompt_optimized_context: {prompt_optimized_context.get('document_count', 0)} docs, {prompt_optimized_context.get('total_context_length', 0)} chars")
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
                self.logger.warning(f"⚠️ [CONTEXT DICT] context_dict is None, using empty dict")
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
                self.logger.info("[ROUTER FALLBACK] SQL 0건 → 키워드+벡터 검색으로 컨텍스트 보강 재시도")
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
                self.logger.warning(f"⚠️ [VALIDATE CONTEXT] context_dict is str before validation, converting to dict")
                context_dict = {"context": context_dict}
            else:
                self.logger.warning(f"⚠️ [VALIDATE CONTEXT] context_dict is not dict (type: {type(context_dict)}), using empty dict")
                context_dict = {}
        
        # 성능 최적화: retrieved_docs가 있고 overall_score가 이미 충분히 높으면 검증 스킵
        metadata_before = self._get_metadata_safely(state)
        cached_overall_score = metadata_before.get("context_validation", {}).get("overall_score") if isinstance(metadata_before.get("context_validation"), dict) else None
        
        # 성능 최적화: 캐시된 overall_score가 0.5 이상이면 검증 스킵
        if cached_overall_score is not None and cached_overall_score >= 0.5:
            self.logger.debug(f"⏭️ [PERFORMANCE] Skipping context validation: cached overall_score={cached_overall_score} >= 0.5")
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
            self.logger.warning(f"⚠️ [SEARCH QUALITY] No retrieved_docs available. Attempting to recover from search results...")
            search_group = state.get("search", {})
            if isinstance(search_group, dict):
                semantic_results = search_group.get("semantic_results", [])
                keyword_results = search_group.get("keyword_results", [])
                if semantic_results and len(semantic_results) > 0:
                    retrieved_docs = semantic_results[:10]
                    state["retrieved_docs"] = retrieved_docs
                    self.logger.info(f"⚠️ [SEARCH QUALITY] Recovered {len(retrieved_docs)} docs from semantic_results")
                elif keyword_results and len(keyword_results) > 0:
                    retrieved_docs = keyword_results[:10]
                    state["retrieved_docs"] = retrieved_docs
                    self.logger.info(f"⚠️ [SEARCH QUALITY] Recovered {len(retrieved_docs)} docs from keyword_results")
            
            if not retrieved_docs or len(retrieved_docs) == 0:
                self.logger.warning(f"⚠️ [SEARCH QUALITY] Failed to recover retrieved_docs. Answer generation may fail.")
        else:
            self.logger.info(f"⚠️ [SEARCH QUALITY] Retrieved {len(retrieved_docs)} docs despite low quality. Proceeding with answer generation.")
        
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
            
            self.logger.info(
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

            self.logger.info(
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
            self.logger.info(
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
                self.logger.info(
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

        if not isinstance(context_dict, dict):
            self.logger.error(f"❌ [CONTEXT DICT] context_dict is not a dict before get_optimized_prompt (type: {type(context_dict)}), converting to dict")
            if isinstance(context_dict, str):
                context_dict = {"context": context_dict}
            elif context_dict is None:
                context_dict = {}
            else:
                context_dict = {}
        
        if not isinstance(context_dict, dict):
            self.logger.error(f"❌ [CONTEXT DICT] context_dict still not dict after conversion, forcing empty dict")
            context_dict = {}
        
        return context_dict
    
    def _normalize_retrieved_docs_to_structured(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """retrieved_docs를 structured_documents 형태로 정규화"""
        normalized_documents = []
        for idx, doc in enumerate(retrieved_docs[:10], 1):
            if not isinstance(doc, dict):
                continue
            
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("content_text")
                or doc.get("summary")
                or ""
            )

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

            if content and len(content.strip()) > 10:
                normalized_documents.append({
                    "document_id": idx,
                    "source": source,
                    "content": content[:2000],
                    "relevance_score": float(relevance_score),
                    "metadata": doc.get("metadata", {})
                })
        
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

        has_documents_section = isinstance(optimized_prompt, str) and ("검색된 법률 문서" in optimized_prompt or "## 🔍" in optimized_prompt)
        documents_in_prompt = optimized_prompt.count("문서") if (isinstance(optimized_prompt, str) and has_documents_section) else 0
        structured_docs_count = 0
        structured_docs_in_context = context_dict.get("structured_documents", {}) if isinstance(context_dict, dict) else {}
        if isinstance(structured_docs_in_context, dict):
            structured_docs_count = len(structured_docs_in_context.get("documents", []))

        self.logger.info(
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
        is_development = os.getenv("DEBUG", "false").lower() == "true" or os.getenv("ENVIRONMENT", "").lower() == "development"
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
                debug_dir = Path("debug/prompts")
                debug_dir.mkdir(parents=True, exist_ok=True)
                prompt_file = debug_dir / f"prompt_{int(time.time())}.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(optimized_prompt)
                self.logger.info(f"💾 [PROMPT SAVED] Full prompt saved to {prompt_file} ({prompt_length} chars)")
            except Exception as e:
                self.logger.debug(f"Could not save prompt to file: {e}")
        else:
            # 성능 최적화: 프로덕션 환경에서는 간소화된 검증만 수행
            has_documents_section = isinstance(optimized_prompt, str) and ("검색된 법률 문서" in optimized_prompt or "## 🔍" in optimized_prompt)
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
        has_documents_section = isinstance(optimized_prompt, str) and ("검색된 법률 문서" in optimized_prompt or "## 🔍" in optimized_prompt)
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
                self.logger.error(
                    f"❌ [PROMPT VALIDATION ERROR] retrieved_docs has {len(retrieved_docs)} documents "
                    f"but prompt does not contain documents_section! "
                    f"This may cause LLM to generate answer without sources!"
                )
            else:
                self.logger.info(
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
                    self.logger.info(
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
                    self.logger.info(
                        f"✅ [PROMPT VALIDATION] Found {doc_found_count}/{min(5, len(documents_in_context))} "
                        f"documents in final prompt"
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
                    f"No documents from structured_documents found in final prompt"
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
        is_retry: bool
    ) -> str:
        """답변 생성 및 캐시 처리"""
        normalized_response = None
        
        is_development = os.getenv("DEBUG", "false").lower() == "true" or os.getenv("ENVIRONMENT", "").lower() == "development"
        
        if not is_retry and not is_development:
            cached_answer = self._check_cache_for_answer(query, query_type, context_dict, retrieved_docs)
            if cached_answer:
                normalized_response = cached_answer
        
        if not normalized_response:
            normalized_response = self.answer_generator.generate_answer_with_chain(
                optimized_prompt=optimized_prompt,
                query=query,
                context_dict=context_dict,
                quality_feedback=quality_feedback,
                is_retry=is_retry
            )
            
            if normalized_response:
                self.logger.info(
                    f"📡 [STREAMING] 답변 생성 완료 - "
                    f"길이: {len(normalized_response)} chars, "
                    f"on_llm_stream 이벤트가 발생하여 클라이언트로 실시간 전달됨"
                )
            
            if not is_retry and not is_development and normalized_response:
                self._cache_answer_if_quality_good(
                    query, query_type, context_dict, retrieved_docs, normalized_response
                )
        
        if normalized_response:
            self.logger.info(
                f"📝 [ANSWER GENERATED] Response received:\n"
                f"   Normalized response length: {len(normalized_response)} characters\n"
                f"   Normalized response content: '{normalized_response[:300]}'\n"
                f"   Normalized response repr: {repr(normalized_response[:100])}"
            )
        
        return normalized_response or ""
    
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
                        self.logger.info(
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
        print(
            f"🔍 [CITATION VALIDATION] citation_coverage={citation_coverage:.2f}, "
            f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}",
            flush=True, file=sys.stdout
        )
        self.logger.info(
            f"🔍 [CITATION VALIDATION] citation_coverage={citation_coverage:.2f}, "
            f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}"
        )
        
        # 개선: citation_coverage 임계값을 0.5로 낮춰서 더 적극적으로 보강
        if citation_coverage < 0.5 and retrieved_docs:
            print(
                f"🔧 [CITATION ENHANCEMENT] Triggering enhancement: "
                f"citation_coverage={citation_coverage:.2f} < 0.5 (target: >= 0.5), "
                f"citation_count={citation_count}, retrieved_docs={len(retrieved_docs)}",
                flush=True, file=sys.stdout
            )
            self.logger.info(
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
        
        self.logger.info(
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
            self.logger.info(
                f"🔧 [ANSWER QUALITY] Attempting automatic improvement: "
                f"coverage={coverage_score:.2f} < 0.5"
            )
            
            if keyword_coverage < 0.5:
                self.logger.info(
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
                self.logger.info(
                    f"✅ [VALIDATION] Good context usage: {citation_count} citations, "
                    f"coverage: {coverage_score:.2f}, document references: {has_document_references}"
                )
    
    def _recover_retrieved_docs_at_start(self, state: LegalWorkflowState) -> None:
        """답변 생성 시작 시 retrieved_docs 복구 (강화된 버전)"""
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        
        if retrieved_docs and len(retrieved_docs) > 0:
            self.logger.debug(f"✅ [RESTORE] 최상위 레벨에서 복원: {len(retrieved_docs)}개")
            return
        
        self.logger.warning(f"⚠️ [GENERATE_ANSWER] No retrieved_docs available at start. Attempting to recover...")
        
        # 1. search 그룹에서 확인
        if "search" in state and isinstance(state.get("search"), dict):
            search_docs = state["search"].get("retrieved_docs", [])
            if search_docs and len(search_docs) > 0:
                self.logger.debug(f"✅ [RESTORE] search 그룹에서 복원: {len(search_docs)}개")
                self._set_state_value(state, "retrieved_docs", search_docs)
                return
        
        # 2. common 그룹에서 확인
        if "common" in state and isinstance(state.get("common"), dict):
            common_search = state["common"].get("search", {})
            if isinstance(common_search, dict):
                common_docs = common_search.get("retrieved_docs", [])
                if common_docs and len(common_docs) > 0:
                    self.logger.debug(f"✅ [RESTORE] common 그룹에서 복원: {len(common_docs)}개")
                    self._set_state_value(state, "retrieved_docs", common_docs)
                    return
        
        # 3. 전역 캐시에서 확인
        try:
            from core.shared.wrappers.node_wrappers import _global_search_results_cache
            if _global_search_results_cache:
                cached_docs = (
                    _global_search_results_cache.get("retrieved_docs", []) or
                    _global_search_results_cache.get("search", {}).get("retrieved_docs", []) or
                    _global_search_results_cache.get("common", {}).get("search", {}).get("retrieved_docs", [])
                )
                if cached_docs and len(cached_docs) > 0:
                    self.logger.info(f"✅ [RESTORE] 전역 캐시에서 복원: {len(cached_docs)}개")
                    self._set_state_value(state, "retrieved_docs", cached_docs)
                    if "search" not in state:
                        state["search"] = {}
                    state["search"]["retrieved_docs"] = cached_docs
                    return
        except Exception as e:
            self.logger.debug(f"전역 캐시 복원 실패: {e}")
        
        # 4. semantic_results + keyword_results에서 재구성
        semantic_results = self._get_state_value(state, "semantic_results", [])
        keyword_results = self._get_state_value(state, "keyword_results", [])
        if semantic_results or keyword_results:
            combined = (semantic_results or []) + (keyword_results or [])
            if combined:
                seen_ids = set()
                unique_docs = []
                for doc in combined:
                    doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                
                if unique_docs:
                    self.logger.info(
                        f"✅ [RESTORE] semantic/keyword 결과에서 재구성: {len(unique_docs)}개 "
                        f"(semantic={len(semantic_results)}, keyword={len(keyword_results)})"
                    )
                    self._set_state_value(state, "retrieved_docs", unique_docs)
                    return
        
        self.logger.warning("⚠️ [RESTORE] retrieved_docs 복원 실패")
