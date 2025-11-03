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
import logging
import re
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Langfuse observe 데코레이터 추가
try:
    from langfuse import observe
    LANGFUSE_OBSERVE_AVAILABLE = True
except ImportError:
    LANGFUSE_OBSERVE_AVAILABLE = False
    # Mock observe decorator
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
from core.agents.handlers.answer_generator import AnswerGenerator
from core.agents.chain_builders import (
    AnswerGenerationChainBuilder,
    ClassificationChainBuilder,
    DirectAnswerChainBuilder,
    DocumentAnalysisChainBuilder,
    QueryEnhancementChainBuilder,
)
from core.agents.handlers.classification_handler import ClassificationHandler
from core.agents.handlers.context_builder import ContextBuilder
from core.agents.extractors import (
    DocumentExtractor,
    QueryExtractor,
    ResponseExtractor,
)
from core.agents.keyword_mapper import LegalKeywordMapper
from core.agents.legal_data_connector_v2 import LegalDataConnectorV2
from core.agents.node_wrappers import with_state_optimization
from core.agents.optimizers.performance_optimizer import PerformanceOptimizer
from core.agents.prompt_builders import PromptBuilder, QueryBuilder
from core.agents.prompt_chain_executor import PromptChainExecutor
from core.agents.validators.quality_validators import (
    AnswerValidator,
    ContextValidator,
    SearchValidator,
)
from core.agents.query_enhancer import QueryEnhancer
from core.agents.extractors.reasoning_extractor import ReasoningExtractor
from core.agents.parsers.response_parsers import (
    AnswerParser,
    ClassificationParser,
    DocumentParser,
    QueryParser,
)
from core.agents.handlers.search_handler import SearchHandler
from core.agents.state_definitions import LegalWorkflowState
from core.agents.state_utils import (
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)
from core.agents.workflow_constants import (
    AnswerExtractionPatterns,
    QualityThresholds,
    RetryConfig,
    WorkflowConstants,
)
from core.agents.workflow_routes import WorkflowRoutes
from core.agents.workflow_utils import WorkflowUtils
from core.services.question_classifier import QuestionType
from core.services.result_merger import ResultMerger, ResultRanker
# 설정 파일 import (lawfirm_langgraph 구조 우선 시도)
try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
except ImportError:
    # Fallback: 기존 경로 (호환성 유지)
    try:
        from core.utils.langgraph_config import LangGraphConfig
    except ImportError:
        from core.utils.langgraph_config import LangGraphConfig
from core.services.term_integration_system import TermIntegrator
from core.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)

# Logger 초기화
logger = logging.getLogger(__name__)

# AnswerStructureEnhancer 통합 (답변 구조화 및 법적 근거 강화)
try:
    from core.services.answer_structure_enhancer import AnswerStructureEnhancer
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = True
except ImportError:
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("AnswerStructureEnhancer not available")


class QueryComplexity(str, Enum):
    """질문 복잡도"""
    SIMPLE = "simple"           # 검색 불필요 (일반 상식, 정의)
    MODERATE = "moderate"       # 단일 검색 필요
    COMPLEX = "complex"         # 다중 검색 필요
    MULTI_HOP = "multi_hop"     # 추론 체인 필요


class RetryCounterManager:
    """재시도 카운터를 안전하게 관리하는 클래스"""

    def __init__(self, logger):
        self.logger = logger

    def get_retry_counts(self, state: LegalWorkflowState) -> Dict[str, int]:
        """
        모든 경로에서 재시도 카운터 안전하게 읽기

        Args:
            state: LegalWorkflowState

        Returns:
            재시도 카운터 딕셔너리 (generation, validation, total)
        """
        generation_retry = 0
        validation_retry = 0

        # 1순위: common.metadata (상태 최적화에서 항상 포함됨)
        if "common" in state and isinstance(state.get("common"), dict):
            common_meta = state["common"].get("metadata", {})
            if isinstance(common_meta, dict):
                generation_retry = max(generation_retry, common_meta.get("generation_retry_count", 0))
                validation_retry = max(validation_retry, common_meta.get("validation_retry_count", 0))

        # 2순위: 최상위 레벨 retry_count
        top_level_retry = state.get("retry_count", 0)
        generation_retry = max(generation_retry, top_level_retry)

        # 3순위: 최상위 레벨 _generation_retry_count, _validation_retry_count
        if isinstance(state, dict):
            generation_retry = max(generation_retry, state.get("_generation_retry_count", 0))
            validation_retry = max(validation_retry, state.get("_validation_retry_count", 0))

        # 4순위: metadata 직접 확인
        metadata = state.get("metadata", {})
        if isinstance(metadata, dict):
            generation_retry = max(generation_retry, metadata.get("generation_retry_count", 0))
            validation_retry = max(validation_retry, metadata.get("validation_retry_count", 0))

        total = generation_retry + validation_retry

        return {
            "generation": generation_retry,
            "validation": validation_retry,
            "total": total
        }

    def increment_retry_count(self, state: LegalWorkflowState, retry_type: str) -> int:
        """
        재시도 카운터 안전하게 증가 (모든 경로에 저장)

        Args:
            state: LegalWorkflowState
            retry_type: "generation" 또는 "validation"

        Returns:
            증가后的 재시도 횟수
        """
        counts = self.get_retry_counts(state)

        if retry_type == "generation":
            new_count = counts["generation"] + 1
            if new_count > RetryConfig.MAX_GENERATION_RETRIES:
                self.logger.warning(
                    f"Generation retry count would exceed limit: {new_count} > {RetryConfig.MAX_GENERATION_RETRIES}"
                )
                new_count = RetryConfig.MAX_GENERATION_RETRIES
        elif retry_type == "validation":
            new_count = counts["validation"] + 1
            if new_count > RetryConfig.MAX_VALIDATION_RETRIES:
                self.logger.warning(
                    f"Validation retry count would exceed limit: {new_count} > {RetryConfig.MAX_VALIDATION_RETRIES}"
                )
                new_count = RetryConfig.MAX_VALIDATION_RETRIES
        else:
            self.logger.error(f"Unknown retry_type: {retry_type}")
            return counts.get(retry_type, 0)

        # 모든 경로에 저장
        self._save_retry_count(state, retry_type, new_count)

        self.logger.info(
            f"✅ [RETRY] {retry_type.capitalize()} retry count: {counts[retry_type]} → {new_count}"
        )

        return new_count

    def _save_retry_count(self, state: LegalWorkflowState, retry_type: str, count: int) -> None:
        """재시도 카운터를 모든 경로에 저장"""
        key = f"{retry_type}_retry_count"

        # metadata에 저장
        if "metadata" not in state or not isinstance(state.get("metadata"), dict):
            state["metadata"] = {}
        state["metadata"][key] = count

        # common.metadata에 저장
        if "common" not in state or not isinstance(state.get("common"), dict):
            state["common"] = {}
        if "metadata" not in state["common"]:
            state["common"]["metadata"] = {}
        state["common"]["metadata"][key] = count

        # 최상위 레벨에 저장 (조건부 엣지 접근용)
        if retry_type == "generation":
            state["retry_count"] = count
            state["_generation_retry_count"] = count
        elif retry_type == "validation":
            state["_validation_retry_count"] = count

    def should_allow_retry(self, state: LegalWorkflowState, retry_type: str) -> bool:
        """
        재시도 허용 여부 확인

        Args:
            state: LegalWorkflowState
            retry_type: "generation" 또는 "validation"

        Returns:
            재시도 허용 여부
        """
        counts = self.get_retry_counts(state)

        # 전역 재시도 횟수 체크
        if counts["total"] >= RetryConfig.MAX_TOTAL_RETRIES:
            self.logger.warning(
                f"Maximum total retry count ({RetryConfig.MAX_TOTAL_RETRIES}) reached"
            )
            return False

        # 개별 재시도 횟수 체크
        if retry_type == "generation":
            if counts["generation"] >= RetryConfig.MAX_GENERATION_RETRIES:
                return False
        elif retry_type == "validation":
            if counts["validation"] >= RetryConfig.MAX_VALIDATION_RETRIES:
                return False

        return True


class EnhancedLegalQuestionWorkflow:
    """개선된 법률 질문 처리 워크플로우"""

    def __init__(self, config: LangGraphConfig):
        self.config = config

        # 개선: 로거를 명시적으로 초기화하고 핸들러 보장
        self.logger = logging.getLogger(__name__)

        # 로거 레벨 설정 (명시적으로 설정)
        self.logger.setLevel(logging.DEBUG)

        # 로거의 propagate 설정 (루트 로거로 전파 보장)
        self.logger.propagate = True

        # 핸들러가 없으면 추가 (표준 출력 스트림 핸들러)
        if not self.logger.handlers:
            import sys
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # 로거 설정 확인 (디버깅용 - 한 번만 출력)
        self.logger.debug(f"Logger initialized: name={__name__}, level={self.logger.level}, handlers={len(self.logger.handlers)}")

        # 통합 프롬프트 관리자 초기화 (우선)
        self.unified_prompt_manager = UnifiedPromptManager()

        # 컴포넌트 초기화
        self.keyword_mapper = LegalKeywordMapper()
        self.data_connector = LegalDataConnectorV2()
        self.performance_optimizer = PerformanceOptimizer()
        self.term_integrator = TermIntegrator()
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()

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
            self.logger.warning("AnswerStructureEnhancer not available")

        # AnswerFormatter 초기화 (시각적 포맷팅)
        try:
            from core.services.answer_formatter import AnswerFormatter
            self.answer_formatter = AnswerFormatter()
            self.logger.info("AnswerFormatter initialized for visual formatting")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AnswerFormatter: {e}")
            self.answer_formatter = None

        # Semantic Search Engine 초기화 (벡터 검색을 위한 - lawfirm_v2_faiss.index 사용)
        try:
            from core.services.semantic_search_engine_v2 import SemanticSearchEngineV2
            from core.utils.config import Config
            # lawfirm_v2.db 기반으로 자동으로 ./data/lawfirm_v2_faiss.index 사용
            config = Config()
            db_path = config.database_path
            self.semantic_search = SemanticSearchEngineV2(db_path=db_path)
            self.logger.info(f"SemanticSearchEngineV2 initialized successfully with {db_path}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SemanticSearchEngineV2: {e}")
            self.semantic_search = None

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
            from core.services.conversation_manager import ConversationManager
            from core.services.multi_turn_handler import MultiTurnQuestionHandler
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.conversation_manager = ConversationManager()
            self.logger.info("MultiTurnQuestionHandler initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MultiTurnQuestionHandler: {e}")
            self.multi_turn_handler = None
            self.conversation_manager = None

        # AIKeywordGenerator 초기화 (AI 키워드 확장)
        try:
            from core.services.ai_keyword_generator import AIKeywordGenerator
            self.ai_keyword_generator = AIKeywordGenerator()
            self.logger.info("AIKeywordGenerator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AIKeywordGenerator: {e}")
            self.ai_keyword_generator = None

        # EmotionIntentAnalyzer 초기화 (긴급도 평가용)
        try:
            from core.services.emotion_intent_analyzer import EmotionIntentAnalyzer
            self.emotion_analyzer = EmotionIntentAnalyzer()
            self.logger.info("EmotionIntentAnalyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize EmotionIntentAnalyzer: {e}")
            self.emotion_analyzer = None

        # LegalBasisValidator 초기화 (법령 검증용)
        try:
            from core.services.legal_basis_validator import LegalBasisValidator
            self.legal_validator = LegalBasisValidator()
            self.logger.info("LegalBasisValidator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalBasisValidator: {e}")
            self.legal_validator = None

        # DocumentProcessor 초기화 (문서 분석용)
        try:
            from core.utils.config import Config as UtilsConfig
            from core.services.document_processor import LegalDocumentProcessor
            utils_config = UtilsConfig()
            self.document_processor = LegalDocumentProcessor(utils_config)
            self.logger.info("LegalDocumentProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalDocumentProcessor: {e}")
            self.document_processor = None

        # ConfidenceCalculator 초기화 (신뢰도 계산용)
        try:
            from core.services.confidence_calculator import (
                ConfidenceCalculator,
            )
            self.confidence_calculator = ConfidenceCalculator()
            self.logger.info("ConfidenceCalculator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ConfidenceCalculator: {e}")
            self.confidence_calculator = None

        # LLM 초기화
        self.llm = self._initialize_llm()

        # 빠른 LLM 초기화 (간단한 질문용)
        self.llm_fast = self._initialize_llm_fast()

        # 답변 생성 핸들러 초기화 (Phase 5 리팩토링) - LLM 초기화 이후
        self.answer_generator = AnswerGenerator(
            llm=self.llm,
            logger=self.logger
        )

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

        # 쿼리 강화 캐시 초기화
        # Phase 7 리팩토링: 쿼리 강화 캐시는 QueryEnhancer로 이동됨

        # 복잡도 분류 캐시 초기화
        self._complexity_cache: Dict[str, Tuple[QueryComplexity, bool]] = {}

        # 통합 분류 캐시 초기화 (질문 유형 + 복잡도 동시 분류)
        self._classification_cache: Dict[str, Tuple[QuestionType, float, QueryComplexity, bool]] = {}


        # Agentic AI Tool 시스템 초기화 (Tool Use/Function Calling)
        # lawfirm_langgraph 구조 사용 (langgraph_core.tools)
        if self.config.use_agentic_mode:
            try:
                from langgraph_core.tools import LEGAL_TOOLS
                self.legal_tools = LEGAL_TOOLS
                self.agentic_agent = None  # 지연 초기화
                self.logger.info(f"Agentic AI mode enabled with {len(LEGAL_TOOLS)} tools (from langgraph_core.tools)")
            except ImportError as e:
                self.logger.warning(f"Failed to import legal tools from langgraph_core.tools: {e}. Agentic mode disabled.")
                self.legal_tools = []
                self.agentic_agent = None
            except Exception as e:
                self.logger.warning(f"Failed to initialize Agentic tools: {e}. Agentic mode disabled.")
                self.legal_tools = []
                self.agentic_agent = None
        else:
            self.legal_tools = []
            self.agentic_agent = None

        # 통계 관리 (config에서 활성화 여부 확인)
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
            'avg_complexity_classification_time': 0.0
        } if self.config.enable_statistics else None

        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.info("EnhancedLegalQuestionWorkflow initialized with UnifiedPromptManager.")

    def _initialize_llm(self):
        """LLM 초기화 (Google Gemini 우선, Ollama 백업)"""
        if self.config.llm_provider == "google":
            try:
                return self._initialize_gemini()
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Falling back to Ollama.")

        if self.config.llm_provider == "ollama":
            try:
                return self._initialize_ollama()
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}. Using Mock LLM.")

        return self._create_mock_llm()

    def _initialize_gemini(self):
        """Google Gemini LLM 초기화"""
        gemini_llm = ChatGoogleGenerativeAI(
            model=self.config.google_model,
            temperature=WorkflowConstants.TEMPERATURE,
            max_output_tokens=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=WorkflowConstants.TIMEOUT,
            api_key=self.config.google_api_key
        )
        logger.info(f"Initialized Google Gemini LLM: {self.config.google_model}")
        return gemini_llm

    def _initialize_ollama(self):
        """Ollama LLM 초기화"""
        ollama_llm = Ollama(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
            temperature=WorkflowConstants.TEMPERATURE,
            num_predict=WorkflowConstants.MAX_OUTPUT_TOKENS,
            timeout=20
        )
        logger.info(f"Initialized Ollama LLM: {self.config.ollama_model}")
        return ollama_llm

    def _create_mock_llm(self):
        """Mock LLM 생성"""
        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response for: " + prompt
            async def ainvoke(self, prompt):
                return "Mock LLM async response for: " + prompt

        logger.warning("No valid LLM provider configured or failed to initialize. Using Mock LLM.")
        return MockLLM()

    def _initialize_llm_fast(self):
        """빠른 LLM 초기화 (간단한 질문용 - Gemini Flash 또는 작은 모델)"""
        if self.config.llm_provider == "google":
            try:
                # Gemini Flash 모델 사용 (더 빠름)
                flash_model = "gemini-1.5-flash"
                if self.config.google_model and "flash" in self.config.google_model.lower():
                    flash_model = self.config.google_model

                gemini_llm_fast = ChatGoogleGenerativeAI(
                    model=flash_model,
                    temperature=0.3,
                    max_output_tokens=500,  # 간단한 답변만 필요
                    timeout=10,
                    api_key=self.config.google_api_key
                )
                logger.info(f"Initialized fast LLM: {flash_model}")
                return gemini_llm_fast
            except Exception as e:
                logger.warning(f"Failed to initialize fast LLM: {e}. Using main LLM.")
                return self.llm

        # 기본 LLM 사용
        return self.llm

    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구축 - Adaptive RAG 및 최적화 적용"""
        workflow = StateGraph(LegalWorkflowState)

        # 통합된 질문 분류 및 복잡도 판단 노드 (Phase 4)
        workflow.add_node("classify_query_and_complexity", self.classify_query_and_complexity)

        # 직접 답변 (간단한 질문용)
        workflow.add_node("direct_answer", self.direct_answer_node)

        # 병렬 분류 (긴급도 + 멀티턴)
        workflow.add_node("classification_parallel", self.classification_parallel)
        workflow.add_node("assess_urgency", self.assess_urgency)
        workflow.add_node("resolve_multi_turn", self.resolve_multi_turn)
        workflow.add_node("route_expert", self.route_expert)  # Phase 9: route_expert는 래퍼 메서드로 유지
        workflow.add_node("analyze_document", self.analyze_document)

        # 키워드 확장 및 검색 쿼리 준비 노드 (Phase 6 - 서브노드로 분리)
        workflow.add_node("expand_keywords", self.expand_keywords)
        workflow.add_node("prepare_search_query", self.prepare_search_query)

        # 개선된 검색 노드들
        workflow.add_node("execute_searches_parallel", self.execute_searches_parallel)

        # 검색 결과 처리 통합 (6개 노드 병합)
        workflow.add_node("process_search_results_combined", self.process_search_results_combined)

        # 통합된 문서 준비 및 용어 처리 노드 (Phase 3)
        workflow.add_node("prepare_documents_and_terms", self.prepare_documents_and_terms)

        # 통합된 답변 생성, 검증, 포맷팅 및 최종 준비 노드 (Phase 5 + Phase 2 통합)
        workflow.add_node("generate_and_validate_answer", self.generate_and_validate_answer)

        # Agentic AI 노드 (Tool Use/Function Calling)
        if self.config.use_agentic_mode:
            workflow.add_node("agentic_decision", self.agentic_decision_node)
            self.logger.info("Agentic decision node added to workflow")

        # Entry point
        workflow.set_entry_point("classify_query_and_complexity")

        # 복잡도 분류 후 라우팅 (Phase 4)
        # Phase 9 리팩토링: 조건부 엣지 함수를 WorkflowRoutes로 이동
        # Agentic 모드가 활성화되어 있고 복잡한 질문인 경우 Agentic 노드로 라우팅
        if self.config.use_agentic_mode:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self._route_by_complexity_with_agentic,  # Agentic 모드용 라우팅
                {
                    "simple": "direct_answer",
                    "moderate": "classification_parallel",
                    "complex": "agentic_decision",  # 복잡한 질문은 Agentic 노드로
                }
            )
        else:
            workflow.add_conditional_edges(
                "classify_query_and_complexity",
                self._route_by_complexity,  # 래퍼 메서드 사용
                {
                    "simple": "direct_answer",      # 간단한 질문 → 직접 답변
                    "moderate": "classification_parallel",  # 중간 질문 → 병렬 분류
                    "complex": "classification_parallel",  # 복잡한 질문 → 병렬 분류
                }
            )

        # 간단한 질문은 직접 답변 생성 후 END로 (포맷팅은 direct_answer_node 내부에서 처리)
        workflow.add_edge("direct_answer", END)

        # Agentic 노드에서 나온 결과 처리
        if self.config.use_agentic_mode:
            workflow.add_conditional_edges(
                "agentic_decision",
                self._route_after_agentic,  # Agentic 노드 후 라우팅
                {
                    "has_results": "prepare_documents_and_terms",  # 검색 결과가 있으면 문서 준비
                    "no_results": "generate_and_validate_answer",  # 검색 결과가 없으면 바로 답변 생성
                }
            )

        # 병렬 분류 후 전문가 라우팅
        workflow.add_edge("classification_parallel", "route_expert")

        # 조건부: 전문가 라우팅 후 문서 분석 여부
        # Phase 9 리팩토링: 조건부 엣지 함수를 WorkflowRoutes로 이동
        workflow.add_conditional_edges(
            "route_expert",
            self._should_analyze_document,  # 래퍼 메서드 사용
            {
                "analyze": "analyze_document",
                "skip": "expand_keywords"
            }
        )

        workflow.add_edge("analyze_document", "expand_keywords")

        # 키워드 확장 → 검색 쿼리 준비 (Phase 6)
        workflow.add_edge("expand_keywords", "prepare_search_query")

        # 개선된 검색 플로우 (Adaptive RAG 적용, Phase 6)
        # Phase 9 리팩토링: 조건부 엣지 함수를 WorkflowRoutes로 이동
        # 개선: 검색 스킵 시 prepare_documents_and_terms를 건너뛰고 답변 생성으로 직접 라우팅
        workflow.add_conditional_edges(
            "prepare_search_query",
            self._should_skip_search_adaptive,  # 래퍼 메서드 사용
            {
                "skip": "generate_and_validate_answer",  # 검색 스킵 시 문서 준비 생략하고 바로 답변 생성
                "continue": "execute_searches_parallel"
            }
        )

        # 병렬 검색 → 통합 검색 결과 처리 (6개 노드 병합)
        workflow.add_edge("execute_searches_parallel", "process_search_results_combined")
        workflow.add_edge("process_search_results_combined", "prepare_documents_and_terms")
        workflow.add_edge("prepare_documents_and_terms", "generate_and_validate_answer")

        # 통합된 답변 생성, 검증, 포맷팅 후 제한된 재시도 (Phase 5 + Phase 2 통합)
        # Phase 9 리팩토링: 조건부 엣지 함수를 WorkflowRoutes로 이동
        workflow.add_conditional_edges(
            "generate_and_validate_answer",
            self._should_retry_validation,  # 래퍼 메서드 사용
            {
                "accept": END,  # 포맷팅 완료 후 직접 END로
                "retry_generate": "generate_and_validate_answer",
                "retry_search": "expand_keywords"
            }
        )

        return workflow

    @observe(name="expand_keywords")
    @with_state_optimization("expand_keywords", enable_reduction=False)
    def expand_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 확장 전용 노드 (Part 1)"""
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

            # AI 키워드 확장 (조건부)
            if self.ai_keyword_generator:
                keywords = self._get_state_value(state, "extracted_keywords", [])
                if len(keywords) == 0:
                    query = self._get_state_value(state, "query", "")
                    query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", "general_question"))
                    keywords = self.keyword_mapper.get_keywords_for_question(query, query_type_str)
                    keywords = [kw for kw in keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                    keywords = list(set(keywords))
                    self._set_state_value(state, "extracted_keywords", keywords)

                query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
                domain = self._get_domain_from_query_type(query_type_str)

                try:
                    expansion_result = asyncio.run(
                        self.ai_keyword_generator.expand_domain_keywords(
                            domain=domain,
                            base_keywords=keywords,
                            target_count=30
                        )
                    )

                    if expansion_result.api_call_success:
                        all_keywords = keywords + expansion_result.expanded_keywords
                        all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                        all_keywords = list(set(all_keywords))
                        self._set_state_value(state, "extracted_keywords", all_keywords)
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
                    else:
                        fallback_keywords = self.ai_keyword_generator.expand_keywords_with_fallback(domain, keywords)
                        all_keywords = keywords + fallback_keywords
                        all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                        all_keywords = list(set(all_keywords))
                        self._set_state_value(state, "extracted_keywords", all_keywords)
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
                except Exception as e:
                    self.logger.warning(f"AI keyword expansion failed: {e}")

            self._save_metadata_safely(state, "_last_executed_node", "expand_keywords")
            self._update_processing_time(state, start_time)
            self._add_step(state, "키워드 확장", f"키워드 확장 완료: {len(self._get_state_value(state, 'extracted_keywords', []))}개")

        except Exception as e:
            self._handle_error(state, str(e), "키워드 확장 중 오류 발생")

        return state

    # Phase 9 리팩토링: 라우팅 관련 메서드는 WorkflowRoutes로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _should_expand_keywords_ai(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_expand_keywords_ai 래퍼"""
        return self.workflow_routes.should_expand_keywords_ai(state)

    def _should_retry_generation(self, state: LegalWorkflowState) -> str:
        """WorkflowRoutes.should_retry_generation 래퍼"""
        return self.workflow_routes.should_retry_generation(state)

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
                result = self.agentic_agent.invoke({
                    "input": query,
                    "chat_history": []  # 대화 이력은 향후 확장 가능
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

    @observe(name="validate_answer_quality")
    @with_state_optimization("validate_answer_quality", enable_reduction=False)
    def validate_answer_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 품질 및 법령 검증"""
        try:
            # 실행 기록 저장 (헬퍼 메서드 사용)
            self._save_metadata_safely(state, "_last_executed_node", "validate_answer_quality")

            start_time = time.time()
            answer = self._normalize_answer(self._get_state_value(state, "answer", ""))
            errors = self._get_state_value(state, "errors", [])
            sources = self._get_state_value(state, "sources", [])

            # 답변 내용 로깅 (품질 검증 전)
            # answer가 문자열인지 확인 후 안전하게 처리
            answer_content_preview = ""
            if isinstance(answer, str):
                answer_content_preview = answer[:500] if len(answer) > 500 else answer
            else:
                answer_str = str(answer)
                answer_content_preview = answer_str[:500] if len(answer_str) > 500 else answer_str

            answer_length = len(answer) if isinstance(answer, str) else len(str(answer))
            self.logger.info(
                f"🔍 [QUALITY VALIDATION] Answer received for validation:\n"
                f"   Answer length: {answer_length} characters\n"
                f"   Answer content: '{answer_content_preview}'\n"
                f"   Answer type: {type(answer).__name__}\n"
                f"   Error count: {len(errors)}\n"
                f"   Source count: {len(sources)}"
            )

            # 품질 검증
            # answer가 문자열인지 확인 후 길이 계산
            answer_str_for_check = answer if isinstance(answer, str) else str(answer) if answer else ""
            quality_checks = {
                "has_answer": len(answer_str_for_check) > 0,
                "min_length": len(answer_str_for_check) >= WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION,
                "no_errors": len(errors) == 0,
                "has_sources": len(sources) > 0 or len(self._get_state_value(state, "retrieved_docs", [])) > 0
            }

            # 법령 검증 추가
            if self.legal_validator and len(answer_str_for_check) > 0:
                try:
                    query = self._get_state_value(state, "query", "")
                    # answer가 문자열인지 확인 (법령 검증에는 문자열만 전달)
                    answer_for_validation = answer if isinstance(answer, str) else answer_str_for_check
                    validation_result = self.legal_validator.validate_legal_basis(query, answer_for_validation)
                    self._set_state_value(state, "legal_validity_check", validation_result.is_valid)
                    self._set_state_value(state, "legal_basis_validation", {
                        "confidence": validation_result.confidence,
                        "issues": validation_result.issues,
                        "recommendations": validation_result.recommendations
                    })
                    quality_checks["legal_basis_valid"] = validation_result.is_valid
                    self.logger.info(f"Legal basis validation: {validation_result.is_valid}")
                except Exception as e:
                    self.logger.warning(f"Legal validation failed: {e}")
                    self._set_state_value(state, "legal_validity_check", True)
                    quality_checks["legal_basis_valid"] = True
            else:
                self._set_state_value(state, "legal_validity_check", True)
                quality_checks["legal_basis_valid"] = True

            # 품질 점수 계산
            passed_checks = sum(quality_checks.values())
            total_checks = len(quality_checks)
            quality_score = passed_checks / total_checks

            # 품질 검증 상세 결과 로깅
            self.logger.info(
                f"📊 [QUALITY CHECKS] Detailed validation results:\n"
                f"   Quality checks: {quality_checks}\n"
                f"   Passed checks: {passed_checks}/{total_checks}\n"
                f"   Quality score: {quality_score:.2f} (threshold: {QualityThresholds.QUALITY_PASS_THRESHOLD})"
            )

            # 품질 점수 기반 통과 여부 결정
            quality_check_passed = quality_score >= QualityThresholds.QUALITY_PASS_THRESHOLD

            # 조건부 엣지 및 공용 메타 접근을 위해 저장 (최상위+common.metadata)
            self._save_metadata_safely(state, "quality_score", quality_score, save_to_top_level=True)
            self._save_metadata_safely(state, "quality_check_passed", quality_check_passed, save_to_top_level=True)

            self._update_processing_time(state, start_time)

            quality_status = "통과" if quality_check_passed else "실패"
            legal_validity = self._get_state_value(state, "legal_validity_check", True)
            self._add_step(state, "답변 검증",
                         f"품질: {quality_score:.2f}, 법령: {legal_validity}")

            self.logger.info(
                f"Answer quality validation: {quality_status}, "
                f"score: {quality_score:.2f}, checks: {passed_checks}/{total_checks}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "답변 검증 중 오류")
            self._set_state_value(state, "legal_validity_check", True)

            # 에러 시에도 메타데이터 저장
            self._save_metadata_safely(state, "quality_score", 0.0, save_to_top_level=True)
            self._save_metadata_safely(state, "quality_check_passed", False, save_to_top_level=True)

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
            state = self.generate_answer_enhanced(state)

            self._update_processing_time(state, generation_start_time)
            self._save_metadata_safely(state, "_last_executed_node", "generate_and_validate_answer")

            # Part 2: 품질 검증 (validate_answer_quality 로직)
            validation_start_time = time.time()

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
        """품질 검증 (내부 메서드)"""
        # Phase 4: 검증 전 answer 정규화 보장
        answer_raw = self._get_state_value(state, "answer", "")
        normalized_answer = self._normalize_answer(answer_raw)
        if answer_raw != normalized_answer or not isinstance(answer_raw, str):
            self._set_answer_safely(state, normalized_answer)
        answer = normalized_answer
        errors = self._get_state_value(state, "errors", [])
        sources = self._get_state_value(state, "sources", [])

        # 품질 검증
        answer_str_for_check = answer if isinstance(answer, str) else str(answer) if answer else ""
        quality_checks = {
            "has_answer": len(answer_str_for_check) > 0,
            "min_length": len(answer_str_for_check) >= WorkflowConstants.MIN_ANSWER_LENGTH_VALIDATION,
            "no_errors": len(errors) == 0,
            "has_sources": len(sources) > 0 or len(self._get_state_value(state, "retrieved_docs", [])) > 0
        }

        # 법령 검증
        query = self._get_state_value(state, "query", "")
        if self.legal_validator and len(answer_str_for_check) > 0:
            try:
                answer_for_validation = answer if isinstance(answer, str) else answer_str_for_check
                validation_result = self.legal_validator.validate_legal_basis(query, answer_for_validation)
                self._set_state_value(state, "legal_validity_check", validation_result.is_valid)
                self._set_state_value(state, "legal_basis_validation", {
                    "confidence": validation_result.confidence,
                    "issues": validation_result.issues,
                    "recommendations": validation_result.recommendations
                })
                quality_checks["legal_basis_valid"] = validation_result.is_valid
            except Exception as e:
                self.logger.warning(f"Legal validation failed: {e}")
                self._set_state_value(state, "legal_validity_check", True)
                quality_checks["legal_basis_valid"] = True
        else:
            self._set_state_value(state, "legal_validity_check", True)
            quality_checks["legal_basis_valid"] = True

        # 품질 점수 계산
        passed_checks = sum(quality_checks.values())
        total_checks = len(quality_checks)
        quality_score = passed_checks / total_checks if total_checks > 0 else 0.0
        quality_check_passed = quality_score >= QualityThresholds.QUALITY_PASS_THRESHOLD

        # 메타데이터 저장
        self._save_metadata_safely(state, "quality_score", quality_score, save_to_top_level=True)
        self._save_metadata_safely(state, "quality_check_passed", quality_check_passed, save_to_top_level=True)

        legal_validity = self._get_state_value(state, "legal_validity_check", True)
        self._add_step(state, "답변 검증",
                     f"품질: {quality_score:.2f}, 법령: {legal_validity}")

        return quality_check_passed

    def _format_and_finalize_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """포맷팅 및 최종 준비 (내부 메서드)"""
        # AnswerFormatterHandler 사용 - format_and_prepare_final 사용하여 메타 정보 분리 포함
        state = self.answer_formatter_handler.format_and_prepare_final(state)

        # 통계 업데이트
        self.update_statistics(state)

        return state

    def _prepare_final_response_minimal(self, state: LegalWorkflowState) -> None:
        """최소한의 최종 준비 (포맷팅 실패 시)"""
        query_complexity = state.get("metadata", {}).get("query_complexity")
        needs_search = state.get("metadata", {}).get("needs_search", False)
        self.answer_formatter_handler.prepare_final_response_part(state, query_complexity, needs_search)

    # Phase 8 리팩토링: Helper methods는 WorkflowUtils로 이동됨
    # 호환성을 위한 래퍼 메서드 (기존 코드에서 self._method() 호출 시 사용)
    def _get_state_value(self, state: LegalWorkflowState, key: str, default: Any = None) -> Any:
        """WorkflowUtils.get_state_value 래퍼"""
        return WorkflowUtils.get_state_value(state, key, default)

    def _set_state_value(self, state: LegalWorkflowState, key: str, value: Any) -> None:
        """WorkflowUtils.set_state_value 래퍼"""
        WorkflowUtils.set_state_value(state, key, value, self.logger)

    def _update_processing_time(self, state: LegalWorkflowState, start_time: float):
        """WorkflowUtils.update_processing_time 래퍼"""
        return WorkflowUtils.update_processing_time(state, start_time)

    def _add_step(self, state: LegalWorkflowState, step_prefix: str, step_message: str):
        """WorkflowUtils.add_step 래퍼"""
        WorkflowUtils.add_step(state, step_prefix, step_message)

    def _handle_error(self, state: LegalWorkflowState, error_msg: str, context: str = ""):
        """WorkflowUtils.handle_error 래퍼"""
        WorkflowUtils.handle_error(state, error_msg, context, self.logger)

    def _normalize_answer(self, answer_raw: Any) -> str:
        """WorkflowUtils.normalize_answer 래퍼"""
        return WorkflowUtils.normalize_answer(answer_raw)

    def _set_answer_safely(self, state: LegalWorkflowState, answer: Any) -> None:
        """
        Answer를 안전하게 저장하는 헬퍼 메서드

        - normalize_answer를 자동으로 호출하여 문자열 보장
        - 타입 검증 및 로깅 추가
        - 모든 노드에서 일관된 answer 저장 보장

        Args:
            state: LegalWorkflowState
            answer: 저장할 answer 값 (str, dict, 또는 다른 타입)
        """
        # 타입 확인 및 정규화
        original_type = type(answer).__name__
        normalized_answer = self._normalize_answer(answer)
        final_type = type(normalized_answer).__name__

        # 타입 변환이 발생한 경우 로깅
        if original_type != final_type or original_type != 'str':
            self.logger.debug(
                f"[ANSWER TYPE] Type conversion: {original_type} → {final_type} "
                f"(length: {len(normalized_answer)})"
            )

        # answer 저장
        self._set_state_value(state, "answer", normalized_answer)

        # answer 길이 로깅 (너무 짧은 경우 경고)
        if len(normalized_answer) < 10:
            self.logger.warning(
                f"[ANSWER WARNING] Answer is very short (length: {len(normalized_answer)})"
            )

    # Phase 1 리팩토링: 추론 과정 분리 메서드들은 ReasoningExtractor로 이동됨
    # Phase 8 리팩토링: 유틸리티 메서드들은 WorkflowUtils로 이동됨

    def _save_metadata_safely(self, state: LegalWorkflowState, key: str, value: Any,
                             save_to_top_level: bool = False) -> None:
        """WorkflowUtils.save_metadata_safely 래퍼"""
        WorkflowUtils.save_metadata_safely(state, key, value, save_to_top_level)

    def _get_quality_metadata(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """WorkflowUtils.get_quality_metadata 래퍼"""
        return WorkflowUtils.get_quality_metadata(state)

    def _get_category_mapping(self) -> Dict[str, List[str]]:
        """WorkflowUtils.get_category_mapping 래퍼"""
        return WorkflowUtils.get_category_mapping()

    @observe(name="classify_query")
    @with_state_optimization("classify_query", enable_reduction=True)
    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        # 중요: 노드 시작 시 input 그룹 보장
        # LangGraph가 초기 state를 제대로 전달하지 않는 경우 대비
        self.logger.debug(f"classify_query: START - State keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")

        # input 그룹 확인 및 생성
        if "input" not in state or not isinstance(state.get("input"), dict):
            state["input"] = {}
            self.logger.debug(f"classify_query: Created empty input group")

        # query가 없으면 여러 위치에서 찾기
        current_query = state["input"].get("query", "")
        if not current_query:
            # 1. 최상위 레벨에서 찾기
            query_from_top = state.get("query", "")
            session_id_from_top = state.get("session_id", "")
            if query_from_top:
                state["input"]["query"] = query_from_top
                if session_id_from_top:
                    state["input"]["session_id"] = session_id_from_top
                self.logger.debug(f"classify_query: Restored query from top-level: length={len(query_from_top)}")

        # 디버깅: 초기 state의 query 확인
        query_value = self._get_state_value(state, "query", "")
        if not query_value or not str(query_value).strip():
            # state에서 직접 확인
            if "input" in state and isinstance(state.get("input"), dict):
                query_value = state["input"].get("query", "")
                print(f"[DEBUG] classify_query: query from state['input']: '{query_value[:50] if query_value else 'EMPTY'}...'")
            elif isinstance(state, dict) and "query" in state:
                query_value = state["query"]
                print(f"[DEBUG] classify_query: query from state directly: '{query_value[:50] if query_value else 'EMPTY'}...'")
            else:
                print(f"[DEBUG] classify_query: query NOT FOUND, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                # query를 찾을 수 없으면 input에 빈 문자열 설정 (나중에 복원 시도)
                if "input" not in state:
                    state["input"] = {}
                state["input"]["query"] = ""
        else:
            self.logger.debug(f"classify_query: query from _get_state_value: '{query_value[:50]}...'")
            # query를 찾았으면 input에 저장
            if "input" not in state:
                state["input"] = {}
            state["input"]["query"] = query_value

        """
        질문 분류 (LLM 기반)

        사용하는 State 그룹:
        - input: query, session_id
        - classification: query_type, confidence, legal_field, legal_domain (출력)
        - common: processing_steps, errors
        """
        try:
            start_time = time.time()

            query = self._get_state_value(state, "query", "")
            classified_type, confidence = self._classify_with_llm(query)

            # QuestionType enum을 문자열로 변환하여 저장
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            self._set_state_value(state, "query_type", query_type_str)
            self._set_state_value(state, "confidence", confidence)

            # 로깅: LLM 분류 결과
            self.logger.info(
                f"✅ [QUESTION CLASSIFICATION] "
                f"QuestionType={classified_type.name if hasattr(classified_type, 'name') else classified_type} "
                f"(confidence: {confidence:.2f})"
            )

            # 법률 분야 추출
            legal_field = self._extract_legal_field(query_type_str, query)
            self._set_state_value(state, "legal_field", legal_field)
            self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {query_type_str}, 법률분야: {legal_field} (시간: {processing_time:.3f}s)")

            self.logger.info(f"LLM classified query as {query_type_str} with confidence {confidence}, field: {legal_field}")

            # 중요: state에 input 그룹이 없으면 생성 (LangGraph state 병합 시 input이 사라지는 것을 방지)
            # LangGraph는 TypedDict의 각 필드를 병합하는데, input 필드가 결과에 없으면 이전 값이 사라질 수 있음
            query_value = self._get_state_value(state, "query", "")
            session_id_value = self._get_state_value(state, "session_id", "")

            # 항상 input 그룹을 result에 포함 (LangGraph state 병합 시 보존)
            if "input" not in state:
                state["input"] = {}
            state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
            state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

            if not state["input"]["query"]:
                self.logger.warning(f"classify_query: query is empty after ensuring input group!")
            else:
                self.logger.debug(f"Ensured input group in state after classify_query: query length={len(state['input']['query'])}")

        except Exception as e:
            self._handle_error(state, str(e), "LLM 질문 분류 중 오류 발생")
            query = self._get_state_value(state, "query", "")
            classified_type, confidence = self._fallback_classification(query)
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            self._set_state_value(state, "query_type", query_type_str)
            self._set_state_value(state, "confidence", confidence)

            # 법률 분야 추출 (폴백)
            legal_field = self._extract_legal_field(query_type_str, query)
            self._set_state_value(state, "legal_field", legal_field)
            self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))

            self._add_step(state, "폴백 키워드 기반 분류 사용", "폴백 키워드 기반 분류 사용")

            # 중요: state에 input 그룹이 없으면 생성 (폴백 경로에서도 보장)
            query_value = self._get_state_value(state, "query", "")
            session_id_value = self._get_state_value(state, "session_id", "")

            if "input" not in state:
                state["input"] = {}
            state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
            state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

            if not state["input"]["query"]:
                self.logger.warning(f"classify_query (fallback): query is empty after ensuring input group!")
            else:
                self.logger.debug(f"Ensured input group in state after classify_query (fallback): query length={len(state['input']['query'])}")

        # 중요: 반환 전에 항상 input 그룹 보장 (모든 경로에서)
        # LangGraph는 노드가 반환한 state만 다음 노드에 전달하므로, input을 반드시 포함해야 함

        # 1. 현재 state에서 query 찾기
        query_value = None
        session_id_value = None

        # 우선순위 1: state["input"]["query"]
        if "input" in state and isinstance(state.get("input"), dict):
            query_value = state["input"].get("query", "")
            session_id_value = state["input"].get("session_id", "")

        # 우선순위 2: 최상위 레벨 query
        if not query_value:
            query_value = state.get("query", "")
            session_id_value = state.get("session_id", "")

        # 우선순위 3: search.search_query
        if not query_value and "search" in state and isinstance(state.get("search"), dict):
            query_value = state["search"].get("search_query", "")

        # input 그룹 생성 및 설정
        if "input" not in state:
            state["input"] = {}

        # query가 있으면 반드시 저장
        if query_value and str(query_value).strip():
            state["input"]["query"] = query_value
            if session_id_value:
                state["input"]["session_id"] = session_id_value
            self.logger.debug(f"classify_query returning with input.query length={len(query_value)}")
            self.logger.debug(f"classify_query: Returning state with input.query length={len(query_value)}")
        else:
            # query를 찾을 수 없으면 에러 로그만 남기고 계속 진행
            # (다음 노드나 workflow_service에서 복원할 것으로 기대)
            self.logger.error(f"classify_query returning with EMPTY input.query! State keys: {list(state.keys())}")
            self.logger.error(f"classify_query: ERROR - Returning state with EMPTY input.query! State keys: {list(state.keys())}")
            self.logger.debug(f"classify_query: State structure - input={state.get('input')}, has query key={bool(state.get('query'))}")

        return state

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

    @observe(name="classify_query_and_complexity")
    @with_state_optimization("classify_query_and_complexity", enable_reduction=False)
    def classify_query_and_complexity(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 질문 분류 및 복잡도 판단 (classify_query + classify_complexity)"""
        try:
            overall_start_time = time.time()

            # ========== Part 1: classify_query 로직 ==========
            query_start_time = time.time()

            # 중요: 노드 시작 시 input 그룹 보장
            if "input" not in state or not isinstance(state.get("input"), dict):
                state["input"] = {}

            # query가 없으면 여러 위치에서 찾기
            current_query = state["input"].get("query", "")
            if not current_query:
                query_from_top = state.get("query", "")
                session_id_from_top = state.get("session_id", "")
                if query_from_top:
                    state["input"]["query"] = query_from_top
                    if session_id_from_top:
                        state["input"]["session_id"] = session_id_from_top

            # query 확인
            query_value = self._get_state_value(state, "query", "")
            if not query_value or not str(query_value).strip():
                if "input" in state and isinstance(state.get("input"), dict):
                    query_value = state["input"].get("query", "")
                elif isinstance(state, dict) and "query" in state:
                    query_value = state["query"]
                else:
                    if "input" not in state:
                        state["input"] = {}
                    state["input"]["query"] = ""
            else:
                if "input" not in state:
                    state["input"] = {}
                state["input"]["query"] = query_value

            # ========== 통합 분류: 질문 유형 + 복잡도 동시 분류 ==========
            query = self._get_state_value(state, "query", "")

            if not query:
                # 기본값 설정
                classified_type, confidence = self._fallback_classification("")
                complexity = QueryComplexity.MODERATE
                needs_search = True
            else:
                # LLM 기반 체인 분류 (질문 유형 → 법률 분야 → 복잡도 → 검색 필요성)
                if self.config.use_llm_for_complexity:
                    try:
                        classified_type, confidence, complexity, needs_search = self._classify_query_with_chain(query)
                        self.logger.info(
                            f"✅ [CHAIN CLASSIFICATION] "
                            f"QuestionType={classified_type.value}, complexity={complexity.value}, "
                            f"needs_search={needs_search}, confidence={confidence:.2f}"
                        )
                    except Exception as e:
                        self.logger.warning(f"체인 LLM 분류 실패, 폴백 사용: {e}")
                        classified_type, confidence = self._fallback_classification(query)
                        complexity, needs_search = self._fallback_complexity_classification(query)
                else:
                    # 키워드 기반 폴백 직접 사용
                    classified_type, confidence = self._fallback_classification(query)
                    complexity, needs_search = self._fallback_complexity_classification(query)

            # 질문 유형 처리
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            legal_field = self._extract_legal_field(query_type_str, query)

            # State에 저장 (질문 유형)
            self._set_state_value(state, "query_type", query_type_str)
            self._set_state_value(state, "confidence", confidence)
            self._set_state_value(state, "legal_field", legal_field)
            self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))

            self._update_processing_time(state, query_start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {query_type_str}, 법률분야: {legal_field}")

            # input 그룹 보장
            query_value = self._get_state_value(state, "query", "")
            session_id_value = self._get_state_value(state, "session_id", "")
            if "input" not in state:
                state["input"] = {}
            state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
            state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

            # ========== 복잡도 분류 결과 저장 ==========
            # State에 저장 (모든 위치에)
            self._set_state_value(state, "query_complexity", complexity.value)
            self._set_state_value(state, "needs_search", needs_search)

            # 최상위 레벨에도 직접 저장
            if "classification" not in state:
                state["classification"] = {}
            state["classification"]["query_complexity"] = complexity.value
            state["classification"]["needs_search"] = needs_search
            state["query_complexity"] = complexity.value
            state["needs_search"] = needs_search

            # common 그룹과 metadata에도 저장
            if "common" not in state:
                state["common"] = {}
            state["common"]["query_complexity"] = complexity.value
            state["common"]["needs_search"] = needs_search
            if "metadata" not in state:
                state["metadata"] = {}
            elif not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            state["metadata"]["query_complexity"] = complexity.value
            state["metadata"]["needs_search"] = needs_search

            # Global cache에도 저장
            try:
                from core.agents import node_wrappers
                if not hasattr(node_wrappers, '_global_search_results_cache') or node_wrappers._global_search_results_cache is None:
                    node_wrappers._global_search_results_cache = {}
                node_wrappers._global_search_results_cache["query_complexity"] = complexity.value
                node_wrappers._global_search_results_cache["needs_search"] = needs_search
            except Exception as e:
                self.logger.warning(f"Global cache 저장 실패: {e}")

            self._add_step(
                state,
                "복잡도 분류",
                f"질문 복잡도: {complexity.value}, 검색 필요: {needs_search}"
            )

            self._update_processing_time(state, overall_start_time)

        except Exception as e:
            self._handle_error(state, str(e), "질문 분류 및 복잡도 판단 중 오류 발생")
            # 기본값 설정
            try:
                query = self._get_state_value(state, "query", "")
                classified_type, confidence = self._fallback_classification(query)
                query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
                legal_field = self._extract_legal_field(query_type_str, query)
                self._set_state_value(state, "query_type", query_type_str)
                self._set_state_value(state, "confidence", confidence)
                self._set_state_value(state, "legal_field", legal_field)
                self._set_state_value(state, "legal_domain", self._map_to_legal_domain(legal_field))
            except:
                self._set_state_value(state, "query_type", "general_question")
                self._set_state_value(state, "confidence", 0.5)
                self._set_state_value(state, "legal_field", "general")
                self._set_state_value(state, "legal_domain", "general")

            # 기본 복잡도
            self._set_state_value(state, "query_complexity", QueryComplexity.MODERATE.value)
            self._set_state_value(state, "needs_search", True)

        # input 그룹 보장
        query_value = self._get_state_value(state, "query", "")
        session_id_value = self._get_state_value(state, "session_id", "")
        if "input" not in state:
            state["input"] = {}
        state["input"]["query"] = query_value or state.get("input", {}).get("query", "")
        state["input"]["session_id"] = session_id_value or state.get("input", {}).get("session_id", "")

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
        """멀티턴 질문 해결 노드"""
        try:
            start_time = time.time()

            # 멀티턴 핸들러와 세션 관리자가 없으면 스킵
            if not self.multi_turn_handler or not self.conversation_manager:
                self._set_state_value(state, "is_multi_turn", False)
                query = self._get_state_value(state, "query", "")
                self._set_state_value(state, "search_query", query)
                self.logger.debug("Multi-turn handler not available, skipping multi-turn resolution")
                return state

            query = self._get_state_value(state, "query", "")
            session_id = self._get_state_value(state, "session_id", "")

            # 세션에서 대화 맥락 가져오기
            conversation_context = self._get_or_create_conversation_context(session_id)

            if conversation_context and conversation_context.turns:
                # 멀티턴 질문 감지
                is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, conversation_context)
                self._set_state_value(state, "is_multi_turn", is_multi_turn)

                if is_multi_turn:
                    # 완전한 질문 구성
                    multi_turn_result = self.multi_turn_handler.build_complete_query(query, conversation_context)

                    resolved_query = multi_turn_result.get("resolved_query", query)
                    self._set_state_value(state, "multi_turn_confidence", multi_turn_result.get("confidence", 1.0))

                    # 대화 맥락 정보 저장
                    self._set_state_value(state, "conversation_context", self._build_conversation_context_dict(conversation_context))

                    # 검색 쿼리 업데이트 (해결된 쿼리 사용)
                    self._set_state_value(state, "search_query", resolved_query)

                    self.logger.info(f"Multi-turn question resolved: '{query}' -> '{resolved_query}'")
                    self._add_step(state, "멀티턴 처리",
                                 f"멀티턴 질문 해결: {multi_turn_result.get('reasoning', '')}")
                else:
                    # 멀티턴 질문이 아님
                    self._set_state_value(state, "multi_turn_confidence", 1.0)

                    # 단일 턴이므로 search_query는 그대로
                    self._set_state_value(state, "search_query", query)
            else:
                # 대화 맥락이 없음
                self._set_state_value(state, "is_multi_turn", False)
                self._set_state_value(state, "multi_turn_confidence", 1.0)
                self._set_state_value(state, "search_query", query)

            self._update_processing_time(state, start_time)

        except Exception as e:
            self.logger.error(f"Error in resolve_multi_turn: {e}")
            # 에러 발생 시 원본 쿼리 유지
            self._set_state_value(state, "is_multi_turn", False)
            search_query = self._get_state_value(state, "search_query")
            if not search_query:
                search_query = self._get_state_value(state, "query", "")
            self._set_state_value(state, "search_query", search_query)
            self._handle_error(state, str(e), "멀티턴 처리 중 오류 발생")

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

    # Phase 10 리팩토링: 직접 답변 생성 메서드는 DirectAnswerHandler로 이동됨
    # 호환성을 위한 래퍼 메서드 (필요시)
    def _generate_direct_answer_with_chain(self, query: str, llm) -> Optional[str]:
        """DirectAnswerHandler.generate_direct_answer_with_chain 래퍼 (LLM 파라미터 무시)"""
        return self.direct_answer_handler.generate_direct_answer_with_chain(query)

    def _parse_query_type_analysis_response(self, response: str) -> Dict[str, Any]:
        """WorkflowUtils.parse_query_type_analysis_response 래퍼"""
        return WorkflowUtils.parse_query_type_analysis_response(response, self.logger)

    def _parse_quality_validation_response(self, response: str) -> Dict[str, Any]:
        """WorkflowUtils.parse_quality_validation_response 래퍼"""
        return WorkflowUtils.parse_quality_validation_response(response, self.logger)

    # Phase 3 리팩토링: 분류 관련 메서드는 ClassificationHandler로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _classify_with_llm(self, query: str) -> Tuple[QuestionType, float]:
        """ClassificationHandler.classify_with_llm 래퍼"""
        return self.classification_handler.classify_with_llm(query)

    def _fallback_classification(self, query: str) -> Tuple[QuestionType, float]:
        """ClassificationHandler.fallback_classification 래퍼"""
        return self.classification_handler.fallback_classification(query)

    def _fallback_complexity_classification(self, query: str) -> Tuple[QueryComplexity, bool]:
        """ClassificationHandler.fallback_complexity_classification 래퍼"""
        return self.classification_handler.fallback_complexity_classification(query)

    def _parse_complexity_response(self, response: str) -> Optional[Dict[str, Any]]:
        """ClassificationHandler.parse_complexity_response 래퍼"""
        return self.classification_handler.parse_complexity_response(response)

    def _parse_unified_classification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """ClassificationHandler.parse_unified_classification_response 래퍼"""
        return self.classification_handler.parse_unified_classification_response(response)

    def _classify_query_with_chain(self, query: str) -> Tuple[QuestionType, float, QueryComplexity, bool]:
        """
        Prompt Chaining을 사용한 질문 분류 (다단계 체인)

        Step 1: 질문 유형 분류
        Step 2: 법률 분야 추출 (질문 유형 기반)
        Step 3: 복잡도 평가 (질문 + 유형 + 분야 기반)
        Step 4: 검색 필요성 판단 (복잡도 기반)

        Returns:
            Tuple[QuestionType, float, QueryComplexity, bool]: (질문 유형, 신뢰도, 복잡도, 검색 필요 여부)
        """
        try:
            # 캐시 키 생성
            cache_key = f"query_chain:{query}"

            # 캐시 확인
            if cache_key in self._classification_cache:
                self.logger.debug(f"Using cached chain classification for: {query[:50]}...")
                if hasattr(self, 'stats'):
                    self.stats['complexity_cache_hits'] = self.stats.get('complexity_cache_hits', 0) + 1
                return self._classification_cache[cache_key]

            if hasattr(self, 'stats'):
                self.stats['complexity_cache_misses'] = self.stats.get('complexity_cache_misses', 0) + 1

            start_time = time.time()

            # PromptChainExecutor 인스턴스 생성
            llm = self.llm_fast if hasattr(self, 'llm_fast') and self.llm_fast else self.llm
            chain_executor = PromptChainExecutor(llm, self.logger)

            # 체인 스텝 정의
            chain_steps = []

            # Step 1: 질문 유형 분류
            def build_question_type_prompt(prev_output, initial_input):
                query = prev_output.get("query") if isinstance(prev_output, dict) else (initial_input.get("query") if isinstance(initial_input, dict) else "")
                if not query:
                    query = str(prev_output) if not isinstance(prev_output, dict) else ""

                return f"""다음 법률 질문의 유형을 분류해주세요.

질문: {query}

다음 유형 중 하나를 선택하세요:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

다음 형식으로 응답해주세요:
{{
    "question_type": "precedent_search" | "law_inquiry" | "legal_advice" | "procedure_guide" | "term_explanation" | "general_question",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "question_type_classification",
                "prompt_builder": build_question_type_prompt,
                "input_extractor": lambda prev: {"query": query} if isinstance(prev, dict) or not prev else prev,
                "output_parser": lambda response, prev: ClassificationParser.parse_question_type_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "question_type" in output,
                "required": True
            })

            # Step 2: 법률 분야 추출 (질문 유형 기반)
            def build_legal_field_prompt(prev_output, initial_input):
                # prev_output은 Step 1의 결과 (question_type 포함)
                if not isinstance(prev_output, dict):
                    return None

                question_type = prev_output.get("question_type", "")
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                if not question_type:
                    return None

                return f"""다음 질문과 질문 유형을 바탕으로 법률 분야를 추출해주세요.

질문: {query_value}
질문 유형: {question_type}

법률 분야 예시:
- family_law (가족법): 이혼, 양육권, 상속, 부양 등
- civil_law (민법): 계약, 손해배상, 물권, 채권 등
- corporate_law (기업법): 회사법, 상법, 금융법 등
- intellectual_property (지적재산권): 특허, 상표, 저작권 등
- criminal_law (형법): 형사소송, 범죄 등
- labor_law (노동법): 근로법, 근로기준법 등
- administrative_law (행정법): 행정처분, 행정소송 등
- general (일반): 분류되지 않는 경우

다음 형식으로 응답해주세요:
{{
    "legal_field": "family_law" | "civil_law" | "corporate_law" | "intellectual_property" | "criminal_law" | "labor_law" | "administrative_law" | "general",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "legal_field_extraction",
                "prompt_builder": build_legal_field_prompt,
                "input_extractor": lambda prev: prev,  # Step 1의 출력을 그대로 사용
                "output_parser": lambda response, prev: ClassificationParser.parse_legal_field_response(response),
                "validator": lambda output: output is None or (isinstance(output, dict) and "legal_field" in output),
                "required": False,  # 선택 단계 (없어도 진행 가능)
                "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("question_type")
            })

            # Step 3: 복잡도 평가 (질문 + 유형 + 분야 기반)
            def build_complexity_prompt(prev_output, initial_input):
                # prev_output은 Step 2의 결과 또는 Step 1의 결과
                if not isinstance(prev_output, dict):
                    prev_output = {}

                # Step 1과 Step 2의 결과 통합
                question_type = prev_output.get("question_type", "")
                legal_field = prev_output.get("legal_field", "")
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                # Step 1 결과에서 질문 유형 찾기
                if not question_type:
                    # prev_output에서 질문 유형 찾기 시도 (이전 단계 출력 통합)
                    if isinstance(prev_output, dict):
                        question_type = prev_output.get("question_type", "")

                return f"""다음 질문의 복잡도를 평가해주세요.

질문: {query_value}
질문 유형: {question_type}
법률 분야: {legal_field if legal_field else "미지정"}

다음 복잡도 중 하나를 선택하세요:
1. simple (간단):
   - 단순 인사말: "안녕하세요", "고마워요" 등
   - 매우 간단한 법률 용어 정의 (10자 이내, 일반 상식 수준)
   - 검색이 불필요한 경우

2. moderate (중간):
   - 특정 법령 조문 조회: "민법 제123조", "형법 제250조" 등
   - 단일 법률 개념 질문: "계약이란?", "손해배상의 요건은?"
   - 단일 판례 검색: "XX 사건 판례"
   - 검색이 필요하지만 단순한 경우

3. complex (복잡):
   - 비교 분석 질문: "계약 해지와 해제의 차이", "이혼과 재혼의 차이"
   - 절차/방법 질문: "이혼 절차는?", "소송 방법은?"
   - 다중 법령/판례 필요: "손해배상 관련 최근 판례와 법령"
   - 복합적 법률 분석: "계약 해지 시 위약금과 손해배상"
   - 검색과 분석이 모두 필요한 경우

다음 형식으로 응답해주세요:
{{
    "complexity": "simple" | "moderate" | "complex",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "complexity_assessment",
                "prompt_builder": build_complexity_prompt,
                "input_extractor": lambda prev: prev,  # 이전 단계의 통합 결과 사용
                "output_parser": lambda response, prev: ClassificationParser.parse_complexity_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "complexity" in output,
                "required": True
            })

            # Step 4: 검색 필요성 판단 (복잡도 기반)
            def build_search_necessity_prompt(prev_output, initial_input):
                # prev_output은 Step 3의 결과 (complexity 포함)
                if not isinstance(prev_output, dict):
                    return None

                complexity = prev_output.get("complexity", "")
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                if not complexity:
                    return None

                return f"""다음 질문의 검색 필요성을 판단해주세요.

질문: {query_value}
복잡도: {complexity}

검색이 필요한 경우:
- simple이 아닌 경우 (moderate 또는 complex)
- 법률 조문, 판례, 규정을 찾아야 하는 경우
- 최신 정보가 필요한 경우

검색이 불필요한 경우:
- simple 복잡도인 경우
- 일반적인 법률 상식으로 답변 가능한 경우
- 단순 인사말이나 정의 질문

다음 형식으로 응답해주세요:
{{
    "needs_search": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "search_necessity_assessment",
                "prompt_builder": build_search_necessity_prompt,
                "input_extractor": lambda prev: prev,  # Step 3의 출력 사용
                "output_parser": lambda response, prev: ClassificationParser.parse_search_necessity_response(response),
                "validator": lambda output: output is None or (isinstance(output, dict) and "needs_search" in output),
                "required": False,  # 선택 단계
                "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("complexity")
            })

            # 체인 실행
            initial_input_dict = {"query": query}
            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=2,
                stop_on_failure=False
            )

            # 결과 추출 및 변환
            chain_history = chain_result.get("chain_history", [])

            # Step 1 결과: 질문 유형
            question_type_result = None
            for step in chain_history:
                if step.get("step_name") == "question_type_classification" and step.get("success"):
                    question_type_result = step.get("output", {})
                    break

            # Step 2 결과: 법률 분야 (선택적)
            legal_field_result = None
            for step in chain_history:
                if step.get("step_name") == "legal_field_extraction" and step.get("success"):
                    legal_field_result = step.get("output", {})
                    break

            # Step 3 결과: 복잡도
            complexity_result = None
            for step in chain_history:
                if step.get("step_name") == "complexity_assessment" and step.get("success"):
                    complexity_result = step.get("output", {})
                    break

            # Step 4 결과: 검색 필요성
            search_necessity_result = None
            for step in chain_history:
                if step.get("step_name") == "search_necessity_assessment" and step.get("success"):
                    search_necessity_result = step.get("output", {})
                    break

            # 결과 변환
            if not question_type_result or not isinstance(question_type_result, dict):
                raise ValueError("Question type classification failed")

            # QuestionType 변환
            question_type_mapping = {
                "precedent_search": QuestionType.PRECEDENT_SEARCH,
                "law_inquiry": QuestionType.LAW_INQUIRY,
                "legal_advice": QuestionType.LEGAL_ADVICE,
                "procedure_guide": QuestionType.PROCEDURE_GUIDE,
                "term_explanation": QuestionType.TERM_EXPLANATION,
                "general_question": QuestionType.GENERAL_QUESTION,
            }
            question_type_str = question_type_result.get("question_type", "general_question")
            classified_type = question_type_mapping.get(question_type_str, QuestionType.GENERAL_QUESTION)
            confidence = float(question_type_result.get("confidence", 0.85))

            # QueryComplexity 변환
            if complexity_result and isinstance(complexity_result, dict):
                complexity_str = complexity_result.get("complexity", "moderate")
            else:
                complexity_str = "moderate"

            complexity_mapping = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MODERATE,
                "complex": QueryComplexity.COMPLEX,
            }
            complexity = complexity_mapping.get(complexity_str, QueryComplexity.MODERATE)

            # 검색 필요성
            if search_necessity_result and isinstance(search_necessity_result, dict):
                needs_search = search_necessity_result.get("needs_search", True)
            else:
                # 복잡도 기반 기본값
                needs_search = complexity != QueryComplexity.SIMPLE

            elapsed_time = time.time() - start_time

            self.logger.info(
                f"✅ [CHAIN CLASSIFICATION] "
                f"question_type={classified_type.value}, complexity={complexity.value}, "
                f"needs_search={needs_search}, confidence={confidence:.2f}, "
                f"(시간: {elapsed_time:.3f}s)"
            )

            result_tuple = (classified_type, confidence, complexity, needs_search)

            # 캐시에 저장
            if len(self._classification_cache) >= 100:
                oldest_key = next(iter(self._classification_cache))
                del self._classification_cache[oldest_key]

            self._classification_cache[cache_key] = result_tuple

            return result_tuple

        except Exception as e:
            self.logger.warning(f"Chain classification failed: {e}, using fallback")
            if hasattr(self, 'stats'):
                self.stats['complexity_fallback_count'] = self.stats.get('complexity_fallback_count', 0) + 1
            question_type, confidence = self._fallback_classification(query)
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

    # Helper methods for retrieve_documents
    def _get_query_type_str(self, query_type) -> str:
        """WorkflowUtils.get_query_type_str 래퍼"""
        return WorkflowUtils.get_query_type_str(query_type)

    def _normalize_query_type_for_prompt(self, query_type) -> str:
        """WorkflowUtils.normalize_query_type_for_prompt 래퍼"""
        return WorkflowUtils.normalize_query_type_for_prompt(query_type, self.logger)

    def _optimize_search_query(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Dict[str, Any]:
        """
        검색 쿼리 최적화 (LLM 강화 포함, 폴백 지원)

        Returns:
            {
                "semantic_query": "의미적 검색용 쿼리",
                "keyword_queries": ["키워드 query 1", ...],
                "expanded_keywords": ["확장된 키워드", ...],
                "llm_enhanced": bool  # LLM 강화 사용 여부
            }
        """
        """QueryEnhancer.optimize_search_query 래퍼"""
        return self.query_enhancer.optimize_search_query(query, query_type, extracted_keywords, legal_field)

    def _normalize_legal_terms(self, query: str, keywords: List[str]) -> List[str]:
        """QueryEnhancer.normalize_legal_terms 래퍼"""
        return self.query_enhancer.normalize_legal_terms(query, keywords)

    def _expand_legal_terms(
        self,
        terms: List[str],
        legal_field: str
    ) -> List[str]:
        """QueryEnhancer.expand_legal_terms 래퍼"""
        return self.query_enhancer.expand_legal_terms(terms, legal_field)

    def _clean_query_for_fallback(self, query: str) -> str:
        """QueryEnhancer.clean_query_for_fallback 래퍼"""
        return self.query_enhancer.clean_query_for_fallback(query)

    def _build_semantic_query(self, query: str, expanded_terms: List[str]) -> str:
        """QueryEnhancer.build_semantic_query 래퍼"""
        return self.query_enhancer.build_semantic_query(query, expanded_terms)

    def _build_keyword_queries(
        self,
        query: str,
        expanded_terms: List[str],
        query_type: str
    ) -> List[str]:
        """QueryEnhancer.build_keyword_queries 래퍼"""
        return self.query_enhancer.build_keyword_queries(query, expanded_terms, query_type)

    # Phase 7 리팩토링: 쿼리 강화 관련 메서드는 QueryEnhancer로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _enhance_query_with_llm(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.enhance_query_with_llm 래퍼"""
        return self.query_enhancer.enhance_query_with_llm(query, query_type, extracted_keywords, legal_field)

    # Phase 7 리팩토링: 쿼리 강화 관련 메서드는 QueryEnhancer로 이동됨
    def _build_query_enhancement_prompt(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> str:
        """QueryEnhancer.build_query_enhancement_prompt 래퍼"""
        return self.query_enhancer.build_query_enhancement_prompt(query, query_type, extracted_keywords, legal_field)

    def _format_field_info(self, legal_field: str, field_info: Dict[str, Any]) -> str:
        """QueryEnhancer.format_field_info 래퍼"""
        return self.query_enhancer.format_field_info(legal_field, field_info)


    def _enhance_query_with_chain(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.enhance_query_with_chain 래퍼"""
        return self.query_enhancer.enhance_query_with_chain(query, query_type, extracted_keywords, legal_field)

    # Phase 7 리팩토링: 파싱 메서드들은 QueryParser로 이동됨 (response_parsers 모듈 사용)
    def _parse_llm_query_enhancement(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """QueryEnhancer.parse_llm_query_enhancement 래퍼"""
        return self.query_enhancer.parse_llm_query_enhancement(llm_output)

    def _determine_search_parameters(
        self,
        query_type: str,
        query_complexity: int,
        keyword_count: int,
        is_retry: bool
    ) -> Dict[str, Any]:
        """QueryEnhancer.determine_search_parameters 래퍼"""
        return self.query_enhancer.determine_search_parameters(query_type, query_complexity, keyword_count, is_retry)

    # Phase 2 리팩토링: 검색 관련 메서드는 SearchHandler로 이동됨
    # 호환성을 위한 래퍼 메서드
    def _check_cache(self, state: LegalWorkflowState, query: str, query_type_str: str, start_time: float) -> bool:
        """SearchHandler.check_cache 래퍼"""
        return self.search_handler.check_cache(state, query, query_type_str, start_time)

    def _semantic_search(self, query: str, k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """SearchHandler.semantic_search 래퍼"""
        return self.search_handler.semantic_search(query, k)

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

    def _filter_low_quality_results(
        self,
        documents: List[Dict],
        min_relevance: float,
        max_diversity: int
    ) -> List[Dict]:
        """SearchHandler.filter_low_quality_results 래퍼"""
        return self.search_handler.filter_low_quality_results(
            documents, min_relevance, max_diversity
        )

    def _apply_metadata_filters(
        self,
        documents: List[Dict],
        query_type: str,
        legal_field: str
    ) -> List[Dict]:
        """SearchHandler.apply_metadata_filters 래퍼"""
        return self.search_handler.apply_metadata_filters(
            documents, query_type, legal_field
        )

    def _calculate_field_match(self, legal_field: str, doc_category: str) -> float:
        """SearchHandler.calculate_field_match 래퍼"""
        return self.search_handler.calculate_field_match(legal_field, doc_category)

    def _calculate_recency_score(self, doc_date: Any) -> float:
        """SearchHandler.calculate_recency_score 래퍼"""
        return self.search_handler.calculate_recency_score(doc_date)

    def _calculate_source_credibility(self, source: str) -> float:
        """SearchHandler.calculate_source_credibility 래퍼"""
        return self.search_handler.calculate_source_credibility(source)

    def _merge_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """SearchHandler.merge_search_results 래퍼"""
        return self.search_handler.merge_search_results(semantic_results, keyword_results)

    def _update_search_metadata(
        self,
        state: LegalWorkflowState,
        semantic_count: int,
        keyword_count: int,
        documents: List[Dict],
        query_type_str: str,
        start_time: float,
        optimized_queries: Optional[Dict[str, Any]] = None
    ) -> None:
        """SearchHandler.update_search_metadata 래퍼"""
        self.search_handler.update_search_metadata(
            state, semantic_count, keyword_count, documents,
            query_type_str, start_time, optimized_queries
        )

    def _fallback_search(self, state: LegalWorkflowState) -> None:
        """SearchHandler.fallback_search 래퍼"""
        self.search_handler.fallback_search(state)


    @observe(name="expand_keywords_ai")
    @with_state_optimization("expand_keywords_ai", enable_reduction=True)
    def expand_keywords_ai(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AI 키워드 확장 (AIKeywordGenerator 사용)"""
        try:
            # 방법 1: 노드 호출 추적 - 실행 기록 남기기
            # 직접 설정하여 상태 최적화로 인한 손실 방지
            if "metadata" not in state or not isinstance(state.get("metadata"), dict):
                state["metadata"] = {}
            # 중요: query_complexity와 needs_search 보존
            preserved_complexity = state.get("metadata", {}).get("query_complexity")
            preserved_needs_search = state.get("metadata", {}).get("needs_search")
            state["metadata"] = dict(state["metadata"])  # 복사본 생성
            # 보존된 값 복원
            if preserved_complexity:
                state["metadata"]["query_complexity"] = preserved_complexity
            if preserved_needs_search is not None:
                state["metadata"]["needs_search"] = preserved_needs_search
            state["metadata"]["_last_executed_node"] = "expand_keywords_ai"
            # common 그룹에도 설정 (nested 구조 지원)
            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "expand_keywords_ai"

            start_time = time.time()

            if not self.ai_keyword_generator:
                self.logger.debug("AIKeywordGenerator not available, skipping")
                return state

            # 기존 키워드가 없으면 기본 키워드 생성
            keywords = self._get_state_value(state, "extracted_keywords", [])
            if len(keywords) == 0:
                query = self._get_state_value(state, "query", "")
                query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", "general_question"))
                # LegalKeywordMapper를 사용하여 기본 키워드 추출
                keywords = self.keyword_mapper.get_keywords_for_question(query, query_type_str)
                # 중복 제거 (hashable 타입만 필터링)
                keywords = [kw for kw in keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                keywords = list(set(keywords))  # 중복 제거
                self._set_state_value(state, "extracted_keywords", keywords)
                self.logger.info(f"Generated base keywords: {len(keywords)} keywords from query")

            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            domain = self._get_domain_from_query_type(query_type_str)

            # AI 키워드 확장 (동기 실행 - 비동기는 지원 안됨)
            expansion_result = asyncio.run(
                self.ai_keyword_generator.expand_domain_keywords(
                    domain=domain,
                    base_keywords=keywords,
                    target_count=30
                )
            )

            if expansion_result.api_call_success:
                # 확장된 키워드 추가
                all_keywords = keywords + expansion_result.expanded_keywords
                # hashable 타입만 필터링 (슬라이스 객체 등 unhashable 타입 방지)
                all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                all_keywords = list(set(all_keywords))
                self._set_state_value(state, "extracted_keywords", all_keywords)

                # 메타데이터 저장 (새 딕셔너리 생성)
                self._set_state_value(state, "ai_keyword_expansion", {
                    "domain": expansion_result.domain,
                    "original_keywords": expansion_result.base_keywords,
                    "expanded_keywords": expansion_result.expanded_keywords,
                    "confidence": expansion_result.confidence,
                    "method": expansion_result.expansion_method
                })

                processing_time = self._update_processing_time(state, start_time)
                self._add_step(state, "AI 키워드 확장",
                             f"AI 키워드 확장 완료: +{len(expansion_result.expanded_keywords)}개 (총 {len(all_keywords)}개, {processing_time:.3f}s)")

                self.logger.info(f"AI keyword expansion: {len(expansion_result.expanded_keywords)} keywords added in {processing_time:.3f}s")
            else:
                # 폴백 사용
                self.logger.warning("AI keyword expansion failed, using fallback")
                fallback_keywords = self.ai_keyword_generator.expand_keywords_with_fallback(
                    domain, keywords
                )
                all_keywords = keywords + fallback_keywords
                # hashable 타입만 필터링 (슬라이스 객체 등 unhashable 타입 방지)
                all_keywords = [kw for kw in all_keywords if isinstance(kw, (str, int, float, tuple)) and kw is not None]
                all_keywords = list(set(all_keywords))
                self._set_state_value(state, "extracted_keywords", all_keywords)

                # 메타데이터 저장 (새 딕셔너리 생성)
                self._set_state_value(state, "ai_keyword_expansion", {
                    "domain": domain,
                    "original_keywords": keywords,
                    "expanded_keywords": fallback_keywords,
                    "confidence": 0.5,
                    "method": "fallback"
                })

                processing_time = self._update_processing_time(state, start_time)
                self._add_step(state, "AI 키워드 확장 (폴백)",
                             f"AI 키워드 확장 (폴백): +{len(fallback_keywords)}개 ({processing_time:.3f}s)")
                self.logger.info(f"Fallback keyword expansion: {len(fallback_keywords)} keywords added in {processing_time:.3f}s")

        except Exception as e:
            self._handle_error(state, str(e), "AI 키워드 확장 중 오류 발생")

        return state

    def _get_domain_from_query_type(self, query_type: str) -> str:
        """WorkflowUtils.get_domain_from_query_type 래퍼"""
        return WorkflowUtils.get_domain_from_query_type(query_type)

    def _get_supported_domains(self) -> List[LegalDomain]:
        """WorkflowUtils.get_supported_domains 래퍼"""
        return WorkflowUtils.get_supported_domains()

    def _is_supported_domain(self, domain: Optional[LegalDomain]) -> bool:
        """WorkflowUtils.is_supported_domain 래퍼"""
        return WorkflowUtils.is_supported_domain(domain)

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

    @observe(name="generate_answer_enhanced")
    @with_state_optimization("generate_answer_enhanced", enable_reduction=True)
    def generate_answer_enhanced(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """개선된 답변 생성 - UnifiedPromptManager 활용"""
        try:
            # 이전에 실행된 노드 확인
            metadata = state.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            # common 그룹에서도 확인 (nested 구조 지원)
            if "common" in state and isinstance(state.get("common"), dict):
                common_metadata = state["common"].get("metadata", {})
                if isinstance(common_metadata, dict):
                    metadata = {**metadata, **common_metadata}

            last_executed_node = metadata.get("_last_executed_node", "")

            # validation 재시도로 호출된 경우 (validate_answer_quality 이후)
            is_retry = (last_executed_node == "validate_answer_quality")
            if is_retry:
                if self.retry_manager.should_allow_retry(state, "validation"):
                    self.retry_manager.increment_retry_count(state, "validation")

            start_time = time.time()

            query_type = self._get_state_value(state, "query_type", "")
            query = self._get_state_value(state, "query", "")
            question_type, domain = self._get_question_type_and_domain(query_type, query)
            model_type = ModelType.GEMINI if self.config.llm_provider == "google" else ModelType.OLLAMA
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])

            # 프롬프트 최적화된 컨텍스트 사용 (있는 경우)
            prompt_optimized_context = self._get_state_value(state, "prompt_optimized_context", {})
            retrieved_docs = self._get_state_value(state, "retrieved_docs", [])

            # 개선 사항 9: retrieved_docs를 최종 state에 명확하게 보존
            if retrieved_docs:
                # 최상위 레벨과 search 그룹 모두에 저장
                self._set_state_value(state, "retrieved_docs", retrieved_docs)
                # search 그룹 직접 저장
                if "search" not in state:
                    state["search"] = {}
                state["search"]["retrieved_docs"] = retrieved_docs
                # common 그룹에도 저장하여 reduction 후에도 유지
                if "common" not in state:
                    state["common"] = {}
                if "search" not in state["common"]:
                    state["common"]["search"] = {}
                state["common"]["search"]["retrieved_docs"] = retrieved_docs

            # 검색 결과가 없으면 global cache에서 직접 가져오기 시도
            if not retrieved_docs or len(retrieved_docs) == 0:
                try:
                    from core.agents.node_wrappers import _global_search_results_cache
                    if _global_search_results_cache:
                        cached_docs = _global_search_results_cache.get("retrieved_docs", [])
                        if cached_docs:
                            retrieved_docs = cached_docs
                            self.logger.info(
                                f"🔄 [ANSWER GENERATION] Restored {len(retrieved_docs)} retrieved_docs from global cache"
                            )
                            # state에도 저장하여 다음 노드에서 사용 가능하도록
                            self._set_state_value(state, "retrieved_docs", retrieved_docs)
                except (ImportError, AttributeError, TypeError) as e:
                    self.logger.debug(f"Could not restore from global cache: {e}")

            # 검색 결과 통계 계산 (답변 생성에 활용)
            semantic_results_count = sum(1 for doc in retrieved_docs if doc.get("search_type") == "semantic") if retrieved_docs else 0
            keyword_results_count = sum(1 for doc in retrieved_docs if doc.get("search_type") == "keyword") if retrieved_docs else 0

            # 개선 사항 9 계속: 검색 결과 통계도 metadata에 저장
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if retrieved_docs:
                metadata["search_results"] = {
                    "count": len(retrieved_docs),
                    "semantic_count": semantic_results_count,
                    "keyword_count": keyword_results_count,
                    "sources": [doc.get("source", "Unknown") for doc in retrieved_docs[:10]]
                }
                self._set_state_value(state, "metadata", metadata)

            if retrieved_docs:
                self.logger.info(
                    f"📊 [ANSWER GENERATION] Using {len(retrieved_docs)} documents for answer generation: "
                    f"Semantic: {semantic_results_count}, Keyword: {keyword_results_count}"
                )

            # prompt_optimized_context 검증
            has_valid_optimized_context = (
                prompt_optimized_context
                and isinstance(prompt_optimized_context, dict)
                and prompt_optimized_context.get("prompt_optimized_text")
                and len(prompt_optimized_context.get("prompt_optimized_text", "").strip()) > 0
            )

            if has_valid_optimized_context:
                prompt_text = prompt_optimized_context["prompt_optimized_text"]
                doc_count = prompt_optimized_context.get("document_count", 0)
                context_length = prompt_optimized_context.get("total_context_length", 0)
                structured_docs_from_context = prompt_optimized_context.get("structured_documents", {})

                # 검증: retrieved_docs가 있는데 prompt_optimized_context의 문서 수가 0이면 경고
                if retrieved_docs and len(retrieved_docs) > 0 and doc_count == 0:
                    self.logger.warning(
                        f"⚠️ [PROMPT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                        f"but prompt_optimized_context has 0 documents. "
                        f"This will trigger forced conversion from retrieved_docs."
                    )

                # 추가 검증: structured_documents가 비어있거나 retrieved_docs보다 현저히 적으면 경고
                structured_docs_count = 0
                if isinstance(structured_docs_from_context, dict):
                    structured_docs_count = len(structured_docs_from_context.get("documents", []))

                if retrieved_docs and len(retrieved_docs) > 0:
                    if structured_docs_count == 0:
                        self.logger.warning(
                            f"⚠️ [PROMPT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                            f"but prompt_optimized_context.structured_documents is empty. "
                            f"This will trigger forced conversion from retrieved_docs."
                        )
                    elif structured_docs_count < len(retrieved_docs) * 0.3:
                        self.logger.warning(
                            f"⚠️ [PROMPT VALIDATION] retrieved_docs has {len(retrieved_docs)} docs "
                            f"but prompt_optimized_context has only {structured_docs_count} documents. "
                            f"This may trigger forced conversion to include more documents."
                        )

                # 검증: prompt_text가 너무 짧으면 경고 (지시사항만 있고 문서 내용이 없을 수 있음)
                if context_length < 500:  # 최소 500자 이상이어야 실제 문서 내용 포함
                    self.logger.warning(
                        f"⚠️ [PROMPT VALIDATION] prompt_optimized_text is too short ({context_length} chars). "
                        f"May contain only instructions without document content."
                    )
                    # retrieved_docs가 있으면 폴백으로 전환 고려
                    if retrieved_docs and len(retrieved_docs) > 0:
                        self.logger.info(
                            f"🔄 [FALLBACK] Switching to _build_intelligent_context due to short prompt_optimized_text"
                        )
                        context_dict = self._build_intelligent_context(state)
                    else:
                        # optimized context 사용 (빈 문서일 수 있음)
                        context_dict = {
                            "context": prompt_text,
                            "structured_documents": prompt_optimized_context.get("structured_documents", {}),
                            "document_count": doc_count,
                            "legal_references": self._extract_legal_references_from_docs(retrieved_docs),
                            "query_type": query_type,
                            "context_length": context_length,
                            "docs_included": doc_count
                        }
                else:
                    # 프롬프트 최적화된 컨텍스트 사용 (정상)
                    # structured_documents가 비어있거나 retrieved_docs와 불일치하면 보강
                    structured_docs = prompt_optimized_context.get("structured_documents", {})

                    # retrieved_docs가 있는데 structured_documents가 비어있거나 적으면 보강
                    if retrieved_docs and len(retrieved_docs) > 0:
                        docs_in_structured = structured_docs.get("documents", []) if isinstance(structured_docs, dict) else []
                        # 최소 요구사항: retrieved_docs의 50% 이상 또는 최소 1개
                        min_required = max(1, min(3, int(len(retrieved_docs) * 0.5))) if len(retrieved_docs) > 5 else 1

                        if not docs_in_structured or len(docs_in_structured) < min_required:
                            # retrieved_docs를 structured_documents 형태로 강제 변환하여 보강
                            normalized_documents = []
                            for idx, doc in enumerate(retrieved_docs[:10], 1):
                                if isinstance(doc, dict):
                                    content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                                    source = doc.get("source") or doc.get("title") or f"Document_{idx}"
                                    relevance_score = doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)

                                    if content and len(content.strip()) > 10:
                                        normalized_documents.append({
                                            "document_id": idx,
                                            "source": source,
                                            "content": content[:2000],
                                            "relevance_score": float(relevance_score),
                                            "metadata": doc.get("metadata", {})
                                        })

                            if normalized_documents:
                                if not isinstance(structured_docs, dict):
                                    structured_docs = {}
                                structured_docs["documents"] = normalized_documents
                                structured_docs["total_count"] = len(normalized_documents)
                                self.logger.info(
                                    f"✅ [SEARCH RESULTS ENFORCED] Added {len(normalized_documents)} documents "
                                    f"from retrieved_docs to structured_documents "
                                    f"(original structured_docs had {len(docs_in_structured)} docs)"
                                )
                                # 문서 수 업데이트
                                doc_count = len(normalized_documents)

                    # prompt_optimized_text를 context와 prompt_optimized_text 두 키에 모두 포함하여
                    # UnifiedPromptManager가 확실히 인식하도록 함
                    context_dict = {
                        "context": prompt_text,  # 메인 컨텍스트
                        "prompt_optimized_text": prompt_text,  # 프롬프트 최적화 텍스트도 직접 포함
                        "structured_documents": structured_docs,  # 강제 보강된 structured_documents
                        "document_count": doc_count,
                        "legal_references": self._extract_legal_references_from_docs(retrieved_docs),
                        "query_type": query_type,
                        "context_length": context_length,
                        "docs_included": len(structured_docs.get("documents", [])) if isinstance(structured_docs, dict) else 0
                    }

                    # content_validation 메타데이터가 있으면 포함
                    content_validation = prompt_optimized_context.get("content_validation")
                    if content_validation:
                        context_dict["content_validation"] = content_validation
                        if not content_validation.get("has_document_content", False):
                            self.logger.warning(
                                f"⚠️ [PROMPT VALIDATION] content_validation indicates no document content in prompt"
                            )

                    self.logger.info(
                        f"✅ [PROMPT OPTIMIZED] Using optimized document context "
                        f"({doc_count} docs, {context_length} chars)"
                    )
            else:
                # 폴백: 기존 방식 사용
                if retrieved_docs and len(retrieved_docs) > 0:
                    self.logger.warning(
                        f"⚠️ [FALLBACK] prompt_optimized_context is missing or invalid, "
                        f"but retrieved_docs exists ({len(retrieved_docs)} docs). "
                        f"Using _build_intelligent_context as fallback."
                    )
                else:
                    self.logger.info(
                        f"ℹ️ [FALLBACK] No prompt_optimized_context and no retrieved_docs. "
                        f"Using _build_intelligent_context."
                    )
                context_dict = self._build_intelligent_context(state)

            # 최종 검증: context_dict에 실제 문서 내용이 포함되었는지 확인
            context_text = context_dict.get("context", "")
            if retrieved_docs and len(retrieved_docs) > 0:
                # retrieved_docs가 있는데 context_text가 너무 짧거나 비어있으면 경고
                if not context_text or len(context_text.strip()) < 100:
                    self.logger.error(
                        f"⚠️ [CONTEXT VALIDATION] retrieved_docs exists ({len(retrieved_docs)} docs) "
                        f"but context_dict['context'] is empty or too short ({len(context_text)} chars). "
                        f"This may cause LLM to generate answer without document references!"
                    )
                else:
                    # context_text에 실제 문서 내용이 포함되어 있는지 간단히 확인
                    # (최소한 하나의 문서 source나 content 일부가 포함되어야 함)
                    has_doc_reference = False
                    for doc in retrieved_docs[:3]:  # 상위 3개만 확인
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

            # SQL 0건 폴백 신호가 있으면 즉시 컨텍스트 보강 재시도
            try:
                meta_forced = self._get_state_value(state, "metadata", {})
                if isinstance(meta_forced, dict) and meta_forced.get("force_rag_fallback"):
                    self.logger.info("[ROUTER FALLBACK] SQL 0건 → 키워드+벡터 검색으로 컨텍스트 보강 재시도")
                    state = self._adaptive_context_expansion(state, {"reason": "router_zero_rows", "needs_expansion": True})
                    context_dict = self._build_intelligent_context(state)
            except Exception as e:
                self.logger.warning(f"Router fallback expansion skipped due to error: {e}")

            # 컨텍스트 품질 검증
            validation_results = self._validate_context_quality(
                context_dict,
                query,
                query_type,
                extracted_keywords
            )

            # 품질 부족 시 컨텍스트 확장
            if validation_results.get("needs_expansion", False):
                state = self._adaptive_context_expansion(state, validation_results)
                # 확장 후 컨텍스트 재구축
                context_dict = self._build_intelligent_context(state)
                # 재검증 (선택적)
                validation_results = self._validate_context_quality(
                    context_dict, query, query_type, extracted_keywords
                )

            # 검증 결과를 메타데이터에 저장
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["context_validation"] = validation_results

            # 재시도 시 품질 피드백 가져오기
            quality_feedback = None
            base_prompt_type = "korean_legal_expert"

            if is_retry:
                quality_feedback = self.answer_generator.get_quality_feedback_for_retry(state)
                base_prompt_type = self.answer_generator.determine_retry_prompt_type(quality_feedback)

                self.logger.info(
                    f"🔄 [RETRY WITH FEEDBACK] Previous score: {quality_feedback.get('previous_score', 0):.2f}, "
                    f"Failed checks: {len(quality_feedback.get('failed_checks', []))}, "
                    f"Prompt type: {base_prompt_type}"
                )

                # 피드백을 메타데이터에 저장 (프롬프트에 포함할 수 있도록)
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["retry_feedback"] = quality_feedback
                self._set_state_value(state, "metadata", metadata)

            # 피드백이 있으면 context_dict에 추가
            if quality_feedback:
                context_dict["quality_feedback"] = quality_feedback

            # 🔍 검색 결과 강제 포함 보강 로직 (중요!)
            # retrieved_docs가 없는 경우 경고 및 처리
            if not retrieved_docs or len(retrieved_docs) == 0:
                self.logger.warning(
                    f"⚠️ [NO SEARCH RESULTS] retrieved_docs is empty. "
                    f"LLM will generate answer without document references. "
                    f"Query: '{query[:50]}...'"
                )
                # 검색 결과가 없을 때 context_dict에 명시적으로 표시
                context_dict["has_search_results"] = False
                context_dict["search_results_note"] = (
                    "현재 데이터베이스에서 관련 법률 문서를 찾지 못했습니다. "
                    "일반적인 법률 정보를 바탕으로 답변을 제공하되, "
                    "구체적인 조문이나 판례를 인용할 수 없음을 명시하세요."
                )
            else:
                context_dict["has_search_results"] = True

            # retrieved_docs가 있는데 structured_documents가 비어있거나 없으면 강제 변환
            # 개선: prompt_optimized_context 사용 시에도 structured_documents가 비어있을 수 있으므로
            # retrieved_docs와 비교하여 실제 문서가 충분히 포함되었는지 확인
            if retrieved_docs and len(retrieved_docs) > 0:
                structured_docs = context_dict.get("structured_documents", {})
                documents_in_structured = []

                if isinstance(structured_docs, dict):
                    documents_in_structured = structured_docs.get("documents", [])

                # has_valid_documents 체크 개선:
                # 1. documents가 존재해야 함
                # 2. documents 수가 retrieved_docs의 최소 30% 이상이어야 함 (너무 엄격하지 않게)
                # 3. 또는 retrieved_docs가 적은 경우(5개 이하) 최소 1개 이상이어야 함
                min_required_docs = max(1, min(3, int(len(retrieved_docs) * 0.3))) if len(retrieved_docs) > 5 else 1

                has_valid_documents = (
                    isinstance(structured_docs, dict)
                    and documents_in_structured
                    and len(documents_in_structured) > 0
                    and len(documents_in_structured) >= min_required_docs
                )

                # 로깅 추가 (디버깅용)
                self.logger.debug(
                    f"🔍 [STRUCTURED DOCS CHECK] retrieved_docs={len(retrieved_docs)}, "
                    f"structured_docs_count={len(documents_in_structured)}, "
                    f"min_required={min_required_docs}, "
                    f"has_valid={has_valid_documents}"
                )

                if not has_valid_documents:
                    # retrieved_docs를 structured_documents 형태로 강제 변환
                    normalized_documents = []
                    for idx, doc in enumerate(retrieved_docs[:10], 1):  # 상위 10개만
                        if isinstance(doc, dict):
                            # 다양한 필드명에서 content 추출
                            content = (
                                doc.get("content")
                                or doc.get("text")
                                or doc.get("content_text")
                                or doc.get("summary")
                                or ""
                            )

                            # source 추출
                            source = (
                                doc.get("source")
                                or doc.get("title")
                                or doc.get("document_id")
                                or f"Document_{idx}"
                            )

                            # relevance_score 추출
                            relevance_score = (
                                doc.get("relevance_score")
                                or doc.get("score")
                                or doc.get("final_weighted_score")
                                or 0.5
                            )

                            # content가 유효한 경우에만 추가
                            if content and len(content.strip()) > 10:
                                normalized_documents.append({
                                    "document_id": idx,
                                    "source": source,
                                    "content": content[:2000],  # 최대 2000자로 제한
                                    "relevance_score": float(relevance_score),
                                    "metadata": doc.get("metadata", {})
                                })

                    if normalized_documents:
                        # structured_documents 생성 또는 업데이트
                        if not isinstance(structured_docs, dict):
                            structured_docs = {}

                        structured_docs["documents"] = normalized_documents
                        structured_docs["total_count"] = len(normalized_documents)
                        context_dict["structured_documents"] = structured_docs
                        context_dict["document_count"] = len(normalized_documents)
                        context_dict["docs_included"] = len(normalized_documents)

                        # 개선 사항 4: 검색 결과와 structured_documents 간의 매핑 정보 추가
                        structured_docs["source_mapping"] = [
                            {
                                "original_index": idx,
                                "document_id": doc.get("document_id"),
                                "source": doc.get("source"),
                                "transformed": True
                            }
                            for idx, doc in enumerate(normalized_documents)
                        ]
                        context_dict["retrieved_to_structured_mapping"] = {
                            "total_retrieved": len(retrieved_docs),
                            "total_transformed": len(normalized_documents),
                            "transformation_rate": len(normalized_documents) / len(retrieved_docs) if retrieved_docs else 0
                        }

                        # 개선 사항 2: structured_documents를 state에 명시적으로 저장
                        if structured_docs and isinstance(structured_docs, dict):
                            # search 그룹에 명시적으로 저장
                            self._set_state_value(state, "structured_documents", structured_docs)
                            # search 그룹 직접 저장
                            if "search" not in state:
                                state["search"] = {}
                            state["search"]["structured_documents"] = structured_docs

                        # 개선 사항 7: state reduction으로 인한 데이터 손실 방지
                        # common 그룹에도 저장하여 reduction 후에도 유지
                        if "common" not in state:
                            state["common"] = {}
                        if "search" not in state["common"]:
                            state["common"]["search"] = {}
                        state["common"]["search"]["structured_documents"] = structured_docs

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
                    # 이미 valid한 structured_documents가 있음
                    doc_count = len(documents_in_structured)
                    self.logger.info(
                        f"✅ [SEARCH RESULTS] structured_documents already has {doc_count} valid documents "
                        f"(retrieved_docs: {len(retrieved_docs)}, required: {min_required_docs})"
                    )

                    # 개선 사항 2 계속: 이미 존재하는 structured_documents도 state에 저장
                    if structured_docs and isinstance(structured_docs, dict):
                        # search 그룹에 명시적으로 저장
                        self._set_state_value(state, "structured_documents", structured_docs)
                        # search 그룹 직접 저장
                        if "search" not in state:
                            state["search"] = {}
                        state["search"]["structured_documents"] = structured_docs

                    # 개선 사항 7 계속: common 그룹에도 저장하여 reduction 후에도 유지
                    if structured_docs:
                        if "common" not in state:
                            state["common"] = {}
                        if "search" not in state["common"]:
                            state["common"]["search"] = {}
                        state["common"]["search"]["structured_documents"] = structured_docs

                    # 추가 검증: structured_documents의 문서들이 retrieved_docs와 일치하는지 확인
                    # (완벽한 일치는 아니지만, 최소한 retrieved_docs에 있는 문서들이 포함되어야 함)
                    if doc_count < len(retrieved_docs) * 0.5:
                        self.logger.warning(
                            f"⚠️ [SEARCH RESULTS] structured_documents has only {doc_count} documents "
                            f"while retrieved_docs has {len(retrieved_docs)}. "
                            f"This may indicate some documents were lost during preparation."
                        )

            optimized_prompt = self.unified_prompt_manager.get_optimized_prompt(
                query=query,
                question_type=question_type,
                domain=domain,
                context=context_dict,
                model_type=model_type,
                base_prompt_type=base_prompt_type
            )

            # 프롬프트 생성 후 상세 로깅
            prompt_length = len(optimized_prompt)
            context_length_in_dict = context_dict.get("context_length", 0)
            docs_included = context_dict.get("docs_included", context_dict.get("document_count", 0))

            # 프롬프트에 문서 섹션이 포함되어 있는지 확인
            has_documents_section = "검색된 법률 문서" in optimized_prompt or "## 🔍" in optimized_prompt
            documents_in_prompt = optimized_prompt.count("문서") if has_documents_section else 0
            structured_docs_count = 0
            structured_docs_in_context = context_dict.get("structured_documents", {})
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

            # 개선 사항 3: optimized_prompt를 state에 저장
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if len(optimized_prompt) > 10000:
                metadata["optimized_prompt"] = optimized_prompt[:10000] + "... (truncated)"
                metadata["optimized_prompt_length"] = len(optimized_prompt)
            else:
                metadata["optimized_prompt"] = optimized_prompt
                metadata["optimized_prompt_length"] = len(optimized_prompt)

            # 개선 사항 5: 프롬프트 생성 결과 정보를 metadata에 저장
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

            # 검증: retrieved_docs가 있는데 프롬프트에 문서 섹션이 없으면 경고
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
                    # 프롬프트에서 문서 섹션 일부 출력 (확인용)
                    doc_section_start = optimized_prompt.find("## 🔍")
                    if doc_section_start >= 0:
                        doc_section_preview = optimized_prompt[doc_section_start:doc_section_start+500]
                        self.logger.debug(
                            f"📄 [PROMPT PREVIEW] Documents section preview:\n{doc_section_preview}..."
                        )

            # 프롬프트에 문서 내용 포함 여부 확인 (강화된 검증)
            context_text = context_dict.get("context", "")
            structured_docs_in_context = context_dict.get("structured_documents", {})
            documents_in_context = []
            if isinstance(structured_docs_in_context, dict):
                documents_in_context = structured_docs_in_context.get("documents", [])

            # 검색 결과 문서가 있는 경우 프롬프트에 포함 여부 확인
            if retrieved_docs and len(retrieved_docs) > 0:
                # 1. context_text 확인
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

                # 2. structured_documents 확인
                if documents_in_context:
                    doc_found_count = 0
                    for doc in documents_in_context[:5]:  # 상위 5개만 확인
                        if isinstance(doc, dict):
                            doc_content = doc.get("content", "")
                            doc_source = doc.get("source", "")

                            # 문서 내용이 프롬프트에 포함되어 있는지 확인
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

                # 3. 프롬프트에 "검색된 법률 문서" 섹션이 있는지 확인
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
                # retrieved_docs가 없는 경우는 정상 (검색 결과가 없는 경우)
                self.logger.debug(
                    "ℹ️ [PROMPT VALIDATION] No retrieved_docs to validate against"
                )

            # 개선 사항 8: 검증 로직 결과를 state에 체계적으로 저장
            prompt_validation_result = {
                "has_documents_section": has_documents_section,
                "documents_in_prompt": documents_in_prompt,
                "structured_docs_in_prompt": doc_found_count if 'doc_found_count' in locals() else 0,
                "has_search_section": has_search_section if 'has_search_section' in locals() else False,
                "validation_warnings": [],  # 검증 경고 목록
                "validation_errors": []     # 검증 오류 목록
            }
            # 검증 경고 및 오류 수집
            if retrieved_docs and len(retrieved_docs) > 0:
                if not has_documents_section:
                    prompt_validation_result["validation_errors"].append(
                        f"retrieved_docs has {len(retrieved_docs)} documents but prompt does not contain documents_section"
                    )
                if not has_search_section and documents_in_context:
                    prompt_validation_result["validation_warnings"].append(
                        f"Search results section keywords not found in prompt despite having {len(documents_in_context)} documents"
                    )
                if 'doc_found_count' in locals() and doc_found_count == 0 and documents_in_context:
                    prompt_validation_result["validation_errors"].append(
                        f"No documents from structured_documents found in final prompt"
                    )
            metadata["prompt_validation"] = prompt_validation_result
            self._set_state_value(state, "metadata", metadata)

            # 프롬프트 샘플 로깅 (디버깅용)
            self.logger.debug(
                f"📝 [PROMPT PREVIEW] Final prompt preview (last 300 chars):\n"
                f"{optimized_prompt[-300:] if len(optimized_prompt) > 300 else optimized_prompt}"
            )

            # 🔴 프롬프트 전체 저장 (평가용)
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

            # 🔄 Prompt Chaining을 사용한 답변 생성 및 개선 (Phase 5 리팩토링)
            normalized_response = self.answer_generator.generate_answer_with_chain(
                optimized_prompt=optimized_prompt,
                query=query,
                context_dict=context_dict,
                quality_feedback=quality_feedback,
                is_retry=is_retry
            )

            # 응답 생성 직후 상세 로깅 (디버깅용)
            self.logger.info(
                f"📝 [ANSWER GENERATED] Response received:\n"
                f"   Normalized response length: {len(normalized_response)} characters\n"
                f"   Normalized response content: '{normalized_response[:300]}'\n"
                f"   Normalized response repr: {repr(normalized_response[:100])}"
            )

            # Phase 1: _set_answer_safely 사용하여 answer 정규화 보장
            self._set_answer_safely(state, normalized_response)

            # 개선 사항 10: 프롬프트-답변 간 연결 정보 추가
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["answer_generation"] = {
                "prompt_length": prompt_length,
                "answer_length": len(normalized_response),
                "prompt_file": str(prompt_file) if 'prompt_file' in locals() else None,
                "generation_timestamp": time.time(),
                "used_search_results": bool(retrieved_docs and len(retrieved_docs) > 0),
                "structured_docs_used": structured_docs_count
            }
            self._set_state_value(state, "metadata", metadata)

            # 답변-컨텍스트 일치도 검증 (재생성 없이 검증 결과만 기록) (Phase 5 리팩토링)
            validation_result = self.answer_generator.validate_answer_uses_context(
                answer=normalized_response,
                context=context_dict,
                query=query,
                retrieved_docs=retrieved_docs  # 검색 결과 정보 추가
            )

            # 검증 결과를 메타데이터에 저장 (재생성 로직 제거)
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # 검증 결과 저장
            metadata["answer_validation"] = validation_result

            # 개선 사항 6: 검색 결과 사용 추적 정보 추가
            search_usage_tracking = {
                "retrieved_docs_count": len(retrieved_docs) if retrieved_docs else 0,
                "structured_docs_count": structured_docs_count,
                "citation_count": validation_result.get("citation_count", 0),
                "coverage_score": validation_result.get("coverage_score", 0.0),
                "has_document_references": validation_result.get("has_document_references", False),
                "sources_in_answer": []  # 실제 사용된 소스 목록 (답변에서 추출)
            }
            # 답변에서 소스 추출
            if retrieved_docs and isinstance(normalized_response, str):
                sources_found = []
                for doc in retrieved_docs:
                    source = doc.get("source") or doc.get("title") or ""
                    if source and source in normalized_response:
                        sources_found.append(source)
                search_usage_tracking["sources_in_answer"] = sources_found[:10]  # 상위 10개만
            metadata["search_usage_tracking"] = search_usage_tracking

            # 검색 결과 활용도 상세 로깅
            citation_count = validation_result.get("citation_count", 0)
            coverage_score = validation_result.get("coverage_score", 0.0)
            has_document_references = validation_result.get("has_document_references", False)

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

            # 재생성이 필요했는지 로그만 기록 (실제 재생성은 하지 않음)
            if validation_result.get("needs_regeneration", False):
                self.logger.warning(
                    f"⚠️ [VALIDATION] Context usage low (coverage: {validation_result.get('coverage_score', 0.0):.2f}), "
                    f"but regeneration is disabled. Answer generated with current validation result."
                )

            # 개선 사항 1: context_dict를 state에 저장
            metadata["context_dict"] = context_dict  # 검증 및 디버깅을 위해 저장

            self._set_state_value(state, "metadata", metadata)

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "답변 생성 완료", "답변 생성 완료")

            # 실행 기록 저장 (재시도 카운터는 RetryCounterManager에서 관리)
            self._save_metadata_safely(state, "_last_executed_node", "generate_answer_enhanced")

            self.logger.info(f"Enhanced answer generated with UnifiedPromptManager in {processing_time:.2f}s")
        except Exception as e:
            self._handle_error(state, str(e), "개선된 답변 생성 중 오류 발생")
            # Phase 1/Phase 7: 폴백 answer 생성 - _set_answer_safely 사용
            fallback_answer = self.answer_generator.generate_fallback_answer(state)
            self._set_answer_safely(state, fallback_answer)
        return state

    def _get_question_type_and_domain(self, query_type, query: str = "") -> Tuple[QuestionType, Optional[LegalDomain]]:
        """WorkflowUtils.get_question_type_and_domain 래퍼"""
        return WorkflowUtils.get_question_type_and_domain(query_type, query, self.logger)

    def _normalize_question_type(self, query_type) -> QuestionType:
        """WorkflowUtils.normalize_question_type 래퍼"""
        return WorkflowUtils.normalize_question_type(query_type, self.logger)

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

    def _rerank_documents_by_relevance(
        self,
        documents: List[Dict],
        query: str,
        extracted_keywords: List[str]
    ) -> List[Dict]:
        """ContextBuilder.rerank_documents_by_relevance 래퍼"""
        return self.context_builder.rerank_documents_by_relevance(documents, query, extracted_keywords)

    def _select_high_value_documents(
        self,
        documents: List[Dict],
        query: str,
        min_relevance: float = 0.7,
        max_docs: int = 5
    ) -> List[Dict]:
        """정보 밀도 기반 문서 선택"""
        if not documents:
            return documents

        try:
            high_value_docs = []

            for doc in documents:
                doc_content = doc.get("content", "")
                if not doc_content or len(doc_content) < 20:
                    continue

                # 1. 법률 조항 인용 수 계산
                citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
                citations = re.findall(citation_pattern, doc_content)
                citation_count = len(citations)
                citation_score = min(1.0, citation_count / 5.0)

                # 2. 핵심 개념 설명 완성도 평가
                query_words = set(query.lower().split())
                content_words = set(doc_content.lower().split())
                explanation_completeness = 0.0
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    explanation_completeness = min(1.0, overlap / max(1, len(query_words)))

                sentences = doc_content.split('。') or doc_content.split('.')
                avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(1, len(sentences))

                descriptive_score_bonus = 0.0
                if 20 <= avg_sentence_length <= 100:
                    descriptive_score_bonus = 0.2
                elif avg_sentence_length > 100:
                    descriptive_score_bonus = 0.1

                explanation_completeness = min(1.0, explanation_completeness + descriptive_score_bonus)

                # 3. 질문 키워드 포함도
                keyword_coverage = 0.0
                if query_words and content_words:
                    keyword_coverage = len(query_words.intersection(content_words)) / max(1, len(query_words))

                # 4. 정보 밀도 종합 점수
                relevance_score = doc.get("final_relevance_score") or doc.get("combined_score", 0.0) or doc.get("relevance_score", 0.0)

                information_density = (
                    0.3 * citation_score +
                    0.3 * explanation_completeness +
                    0.2 * keyword_coverage +
                    0.2 * min(1.0, relevance_score)
                )

                doc["information_density_score"] = information_density
                doc["citation_count"] = citation_count
                doc["explanation_completeness"] = explanation_completeness

                # 관련성 점수와 정보 밀도 점수 가중 평균
                combined_value_score = 0.6 * relevance_score + 0.4 * information_density
                doc["combined_value_score"] = combined_value_score

                # 임계값 체크
                if combined_value_score >= min_relevance:
                    high_value_docs.append(doc)

            # combined_value_score로 정렬
            high_value_docs.sort(key=lambda x: x.get("combined_value_score", 0.0), reverse=True)

            # 최대 문서 수 제한
            selected_docs = high_value_docs[:max_docs]

            self.logger.info(
                f"📚 [HIGH VALUE SELECTION] Selected {len(selected_docs)}/{len(documents)} documents. "
                f"Avg density: {sum(d.get('information_density_score', 0.0) for d in selected_docs) / max(1, len(selected_docs)):.3f}"
            )

            return selected_docs

        except Exception as e:
            self.logger.warning(f"High value document selection failed: {e}, using first {max_docs} documents")
            return documents[:max_docs]

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

    def _optimize_context_composition(
        self,
        high_value_docs: List[Dict],
        query: str,
        max_length: int
    ) -> Dict[str, Any]:
        """ContextBuilder.optimize_context_composition 래퍼"""
        return self.context_builder.optimize_context_composition(high_value_docs, query, max_length)

    def _build_intelligent_context(
        self,
        state: LegalWorkflowState
    ) -> Dict[str, Any]:
        """ContextBuilder.build_intelligent_context 래퍼"""
        return self.context_builder.build_intelligent_context(state)

    def _calculate_context_relevance(
        self,
        context: Dict[str, Any],
        query: str
    ) -> float:
        """ContextBuilder.calculate_context_relevance 래퍼"""
        return self.context_builder.calculate_context_relevance(context, query)

    def _calculate_information_coverage(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> float:
        """ContextBuilder.calculate_information_coverage 래퍼"""
        return self.context_builder.calculate_information_coverage(context, query, query_type, extracted_keywords)

    def _calculate_context_sufficiency(
        self,
        context: Dict[str, Any],
        query_type: str
    ) -> float:
        """ContextBuilder.calculate_context_sufficiency 래퍼"""
        return self.context_builder.calculate_context_sufficiency(context, query_type)

    def _identify_missing_information(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> List[str]:
        """ContextBuilder.identify_missing_information 래퍼"""
        return self.context_builder.identify_missing_information(context, query, query_type, extracted_keywords)

    def _validate_context_quality(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        컨텍스트 품질 검증 (quality_validators 모듈 사용)
        """
        # 의미적 유사도 계산 함수 (semantic_search 사용)
        def calculate_relevance(context_text: str, query: str) -> float:
            try:
                if self.semantic_search and hasattr(self.semantic_search, '_calculate_semantic_score'):
                    return self.semantic_search._calculate_semantic_score(query, context_text)
            except Exception:
                pass
            return ContextValidator.calculate_relevance(context_text, query)

        return ContextValidator.validate_context_quality(
            context=context,
            query=query,
            query_type=query_type,
            extracted_keywords=extracted_keywords,
            calculate_relevance_func=calculate_relevance,
            calculate_coverage_func=None
        )

    def _validate_context_quality_original(
        self,
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str]
    ) -> Dict[str, Any]:
        """컨텍스트 적합성 검증"""
        try:
            # 1. 관련성 점수
            relevance_score = self._calculate_context_relevance(context, query)

            # 2. 정보 커버리지
            coverage_score = self._calculate_information_coverage(
                context, query, query_type, extracted_keywords
            )

            # 3. 충분성 평가
            sufficiency_score = self._calculate_context_sufficiency(context, query_type)

            # 4. 부족한 정보 식별
            missing_info = self._identify_missing_information(
                context, query, query_type, extracted_keywords
            )

            # 종합 점수 계산
            overall_score = (relevance_score * 0.4 + coverage_score * 0.3 + sufficiency_score * 0.3)

            # 검증 결과
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

    def _adaptive_context_expansion(
        self,
        state: LegalWorkflowState,
        validation_results: Dict[str, Any]
    ) -> LegalWorkflowState:
        """적응형 컨텍스트 확장"""
        try:
            if not validation_results.get("needs_expansion", False):
                return state

            # 확장 횟수 확인 (무한 루프 방지)
            metadata = self._get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            expansion_count = metadata.get("context_expansion_count", 0)
            if expansion_count >= 1:  # 최대 1회 확장
                self.logger.info("Context expansion skipped: maximum expansion count reached")
                return state

            missing_info = validation_results.get("missing_information", [])
            query = self._get_state_value(state, "query", "")
            query_type = self._get_state_value(state, "query_type", "")

            if not missing_info:
                return state

            self.logger.info(f"🔧 [CONTEXT EXPANSION] Expanding context for missing: {missing_info[:3]}")

            # 부족한 정보로 추가 검색 쿼리 생성
            expanded_query = query
            if missing_info:
                # 상위 3개 누락 정보를 쿼리에 추가
                safe_missing = [m for m in missing_info[:3] if isinstance(m, str)]
                if safe_missing:
                    expanded_query = f"{query} {' '.join(safe_missing)}"

            # 추가 검색 수행
            try:
                semantic_results, semantic_count = self._semantic_search(expanded_query, k=5)
                keyword_results, keyword_count = self._keyword_search(
                    expanded_query,
                    query_type,
                    limit=3
                )

                # 기존 문서와 통합
                existing_docs = self._get_state_value(state, "retrieved_docs", [])
                all_docs = existing_docs + semantic_results + keyword_results

                # 중복 제거
                seen_ids = set()
                unique_docs = []
                for doc in all_docs:
                    doc_id = doc.get("id") or hash(doc.get("content", "")[:100])
                    if isinstance(doc_id, (str, int)) and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)

                self._set_state_value(state, "retrieved_docs", unique_docs[:10])  # 최대 10개
                metadata["context_expansion_count"] = expansion_count + 1
                self._set_state_value(state, "metadata", metadata)

                self.logger.info(
                    f"✅ [CONTEXT EXPANSION] Added {len(unique_docs) - len(existing_docs)} documents, "
                    f"total: {len(unique_docs)}"
                )

            except Exception as e:
                self.logger.warning(f"Context expansion search failed: {e}")

            return state

        except Exception as e:
            self.logger.error(f"Adaptive context expansion failed: {e}")
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

    def _call_llm_with_retry(self, prompt: str, max_retries: int = WorkflowConstants.MAX_RETRIES) -> str:
        """AnswerGenerator.call_llm_with_retry 래퍼"""
        return self.answer_generator.call_llm_with_retry(prompt, max_retries)

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

    def _extract_response_content(self, response) -> str:
        """WorkflowUtils.extract_response_content 래퍼"""
        return WorkflowUtils.extract_response_content(response)


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

    # Phase 11 리팩토링: 중복된 원본 메서드 코드 제거 완료
    # 답변 포맷팅 관련 메서드는 AnswerFormatterHandler로 이동되어 래퍼만 남음
    #
    # DEPRECATED: 이 메서드는 더 이상 워크플로우 그래프에서 사용되지 않습니다.
    # 포맷팅 로직은 generate_and_validate_answer와 direct_answer_node에 통합되었습니다.
    # 호환성을 위해 남겨두었지만, 향후 제거될 수 있습니다.
    @observe(name="format_and_prepare_final")
    @with_state_optimization("format_and_prepare_final", enable_reduction=False)
    def format_and_prepare_final(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합된 답변 포맷팅 및 최종 준비 (DEPRECATED: generate_and_validate_answer에 통합됨)"""
        try:
            # Phase 4 리팩토링: AnswerFormatterHandler 사용
            state = self.answer_formatter_handler.format_and_prepare_final(state)

            # 통계 업데이트
            self.update_statistics(state)

            confidence = state.get("confidence", 0.0)
            self.logger.info(
                f"format_and_prepare_final completed, confidence: {confidence:.3f}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "답변 포맷팅 및 최종 준비 중 오류 발생")
            answer = self._get_state_value(state, "answer", "")
            if not state.get("answer"):
                state["answer"] = self._normalize_answer(answer)

        return state

    def update_statistics(self, state: LegalWorkflowState):
        """통계 업데이트 (이동 평균 사용)"""
        if not self.stats:
            return

        try:
            self.stats['total_queries'] += 1
            processing_time = state.get("processing_time", 0.0)
            confidence = state.get("confidence", 0.0)
            docs_count = len(state.get("retrieved_docs", []))
            errors_count = len(state.get("errors", []))

            # 이동 평균 계산
            alpha = self.config.stats_update_alpha

            if self.stats['total_queries'] == 1:
                self.stats['avg_response_time'] = processing_time
                self.stats['avg_confidence'] = confidence
            else:
                # 이동 평균 업데이트
                self.stats['avg_response_time'] = (
                    (1 - alpha) * self.stats['avg_response_time'] +
                    alpha * processing_time
                )
                self.stats['avg_confidence'] = (
                    (1 - alpha) * self.stats['avg_confidence'] +
                    alpha * confidence
                )

            # 누적 통계
            self.stats['total_documents_retrieved'] += docs_count
            self.stats['total_errors'] += errors_count

            self.logger.debug(
                f"Statistics updated: queries={self.stats['total_queries']}, "
                f"avg_time={self.stats['avg_response_time']:.2f}s, "
                f"avg_conf={self.stats['avg_confidence']:.2f}"
            )
        except Exception as e:
            self.logger.warning(f"Statistics update failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        if not self.stats:
            return {"enabled": False}

        return {
            "enabled": True,
            "total_queries": self.stats['total_queries'],
            "total_documents_retrieved": self.stats['total_documents_retrieved'],
            "avg_response_time": round(self.stats['avg_response_time'], 3),
            "avg_confidence": round(self.stats['avg_confidence'], 3),
            "total_errors": self.stats['total_errors'],
            "avg_docs_per_query": (
                round(self.stats['total_documents_retrieved'] / self.stats['total_queries'], 2)
                if self.stats['total_queries'] > 0 else 0
            )
        }

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
        """긴급도 평가 노드"""
        try:
            start_time = time.time()
            query = self._get_state_value(state, "query", "")

            if self.emotion_analyzer:
                # 감정 및 의도 분석 (의도 분석만 사용)
                intent_result = self.emotion_analyzer.analyze_intent(query, None)

                # 긴급도 설정
                urgency_level = intent_result.urgency_level.value
                self._set_state_value(state, "urgency_level", urgency_level)
                self._set_state_value(state, "urgency_reasoning", intent_result.reasoning)

                # 긴급 유형 판별
                if "기한" in query or "마감" in query or "데드라인" in query:
                    self._set_state_value(state, "emergency_type", "legal_deadline")
                elif "소송" in query or "재판" in query or "법원" in query:
                    self._set_state_value(state, "emergency_type", "case_progress")
                else:
                    self._set_state_value(state, "emergency_type", None)

                self.logger.info(f"Urgency assessed: {urgency_level}")
            else:
                # 폴백: 키워드 기반 긴급도 평가
                urgency_level = self._assess_urgency_fallback(query)
                self._set_state_value(state, "urgency_level", urgency_level)
                self._set_state_value(state, "urgency_reasoning", "키워드 기반 평가")
                self._set_state_value(state, "emergency_type", None)

            self._update_processing_time(state, start_time)
            self._add_step(state, "긴급도 평가", f"긴급도: {urgency_level}")

        except Exception as e:
            self._handle_error(state, str(e), "긴급도 평가 중 오류")
            self._set_state_value(state, "urgency_level", "medium")
            self._set_state_value(state, "urgency_reasoning", "기본값")
            self._set_state_value(state, "emergency_type", None)

        return state

    def _assess_urgency_fallback(self, query: str) -> str:
        """폴백 긴급도 평가"""
        urgent_keywords = ["긴급", "급해", "빨리", "즉시", "당장"]
        high_keywords = ["오늘", "내일", "이번주", "마감"]

        query_lower = query.lower()
        if any(k in query_lower for k in urgent_keywords):
            return "critical"
        elif any(k in query_lower for k in high_keywords):
            return "high"
        else:
            return "medium"

    @observe(name="analyze_document")
    @with_state_optimization("analyze_document", enable_reduction=True)
    def analyze_document(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """업로드된 문서 분석"""
        try:
            start_time = time.time()

            # Check if document analysis is available in metadata or session context
            # Document would be passed via external parameter or session
            doc_text = state.get("document_analysis", {}).get("raw_text") if isinstance(state.get("document_analysis"), dict) else None

            if not doc_text:
                # Check if there's a document in document_analysis field
                if state.get("document_analysis") and isinstance(state["document_analysis"], str):
                    doc_text = state["document_analysis"]
                else:
                    self.logger.info("No document provided for analysis, skipping")
                    return state

            # 문서 유형 판별 (키워드 기반 또는 LLM)
            doc_type = self._detect_document_type(doc_text)
            state["document_type"] = doc_type

            # 문서 분석 (Prompt Chaining 사용)
            analysis_result = self._analyze_legal_document_with_chain(doc_text, doc_type)

            # Store analysis in state (summarized)
            if isinstance(state["document_analysis"], dict):
                state["document_analysis"].update({
                    "document_type": doc_type,
                    "summary": analysis_result.get("summary", ""),
                    "analysis_time": time.time() - start_time
                })
            else:
                state["document_analysis"] = {
                    "document_type": doc_type,
                    "summary": analysis_result.get("summary", ""),
                    "analysis_time": time.time() - start_time
                }

            state["key_clauses"] = analysis_result.get("key_clauses", [])
            state["potential_issues"] = analysis_result.get("issues", [])

            # 분석 결과를 컨텍스트에 추가 (pruned)
            doc_summary = self._create_document_summary(analysis_result)
            summary_doc = {
                "content": doc_summary[:MAX_DOCUMENT_CONTENT_LENGTH],  # Summarize
                "source": "Uploaded Document Analysis",
                "type": "document_analysis",
                "relevance_score": 1.0,
                "is_summarized": True
            }

            # Prune retrieved_docs before inserting
            if len(state["retrieved_docs"]) >= MAX_RETRIEVED_DOCS:
                state["retrieved_docs"] = prune_retrieved_docs(
                    state["retrieved_docs"],
                    max_items=MAX_RETRIEVED_DOCS - 1,
                    max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
                )

            state["retrieved_docs"].insert(0, summary_doc)

            self._update_processing_time(state, start_time)
            self._add_step(state, "문서 분석", f"{doc_type} 분석 완료")

        except Exception as e:
            self._handle_error(state, str(e), "문서 분석 중 오류")

        return state

    def _detect_document_type(self, text: str) -> str:
        """문서 유형 감지"""
        type_keywords = {
            "contract": ["계약서", "계약", "갑", "을", "본 계약"],
            "complaint": ["고소장", "피고소인", "고소인", "고소취지"],
            "agreement": ["합의서", "합의", "쌍방"],
            "power_of_attorney": ["위임장", "위임인", "수임인"]
        }

        text_lower = text.lower()
        for doc_type, keywords in type_keywords.items():
            if any(k in text_lower for k in keywords):
                return doc_type

        return "general_legal_document"

    def _analyze_legal_document(self, text: str, doc_type: str) -> Dict[str, Any]:
        """법률 문서 분석"""
        analysis = {
            "document_type": doc_type,
            "key_clauses": [],
            "issues": [],
            "summary": "",
            "recommendations": []
        }

        # 주요 조항 추출
        if doc_type == "contract":
            analysis["key_clauses"] = self._extract_contract_clauses(text)
            analysis["issues"] = self._identify_contract_issues(text, analysis["key_clauses"])
        elif doc_type == "complaint":
            analysis["key_clauses"] = self._extract_complaint_elements(text)
            analysis["issues"] = self._identify_complaint_issues(text)

        # 요약 생성
        analysis["summary"] = self._generate_document_summary(text, doc_type, analysis)

        return analysis

    def _extract_contract_clauses(self, text: str) -> List[Dict[str, Any]]:
        """계약서 주요 조항 추출"""
        return DocumentExtractor.extract_contract_clauses(text)

    def _identify_contract_issues(self, text: str, clauses: List[Dict]) -> List[Dict[str, Any]]:
        """계약서 잠재 문제점 식별"""
        issues = []

        # 필수 조항 확인
        required_clauses = ["payment", "period", "termination"]
        # hashable 타입만 필터링 (슬라이스 객체 등 unhashable 타입 방지)
        found_types = set()
        for c in clauses:
            clause_type = c.get("type")
            if clause_type is not None and isinstance(clause_type, (str, int, float, tuple)):
                found_types.add(clause_type)
            elif clause_type is not None:
                # unhashable 타입인 경우 문자열로 변환
                try:
                    found_types.add(str(clause_type))
                except Exception:
                    pass

        for req_type in required_clauses:
            if req_type not in found_types:
                issues.append({
                    "severity": "high",
                    "type": "missing_clause",
                    "description": f"필수 조항 누락: {req_type}",
                    "recommendation": f"{req_type} 조항을 추가하십시오"
                })

        # 불명확한 표현 확인
        vague_terms = ["기타", "등등", "적절한", "합당한"]
        for term in vague_terms:
            if term in text:
                issues.append({
                    "severity": "medium",
                    "type": "vague_term",
                    "description": f"불명확한 용어 사용: {term}",
                    "recommendation": "구체적인 용어로 대체하십시오"
                })

        return issues[:5]  # 상위 5개만

    def _extract_complaint_elements(self, text: str) -> List[Dict[str, Any]]:
        """고소장 요건 추출"""
        return DocumentExtractor.extract_complaint_elements(text)

    def _identify_complaint_issues(self, text: str) -> List[Dict[str, Any]]:
        """고소장 문제점 식별"""
        issues = []

        # 필수 요소 확인
        required_elements = ["피고소인", "사실관계", "청구"]
        for elem in required_elements:
            if elem not in text:
                issues.append({
                    "severity": "high",
                    "type": "missing_element",
                    "description": f"필수 요소 누락: {elem}",
                    "recommendation": f"{elem} 정보를 추가하십시오"
                })

        return issues

    def _analyze_legal_document_with_chain(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Prompt Chaining을 사용한 법률 문서 분석 (다단계 체인)

        Step 1: 문서 유형 확인 (키워드 기반 결과 검증)
        Step 2: 주요 조항 추출 (문서 유형 기반)
        Step 3: 문제점 식별 (조항 기반)
        Step 4: 요약 생성 (조항 + 문제점 기반)
        Step 5: 개선 권고 생성 (문제점 기반)

        Returns:
            Dict[str, Any]: 분석 결과
        """
        try:
            # PromptChainExecutor 인스턴스 생성
            chain_executor = PromptChainExecutor(self.llm, self.logger)

            # 체인 스텝 정의
            chain_steps = []

            # Step 1: 문서 유형 확인 (LLM 검증)
            def build_document_type_verification_prompt(prev_output, initial_input):
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:2000]  # 처음 2000자만
                detected_type = initial_input.get("doc_type") if isinstance(initial_input, dict) else doc_type

                return f"""다음 문서의 유형을 확인하고 검증해주세요.

문서 내용 (일부):
{doc_text}

키워드 기반 감지 결과: {detected_type}

다음 문서 유형 중 하나로 확인해주세요:
- contract (계약서): 계약서, 갑/을, 계약 조건 등
- complaint (고소장): 고소장, 피고소인, 고소인 등
- agreement (합의서): 합의서, 합의, 쌍방 합의 등
- power_of_attorney (위임장): 위임장, 위임인, 수임인 등
- general_legal_document (일반 법률 문서): 위에 해당하지 않는 경우

다음 형식으로 응답해주세요:
{{
    "document_type": "contract" | "complaint" | "agreement" | "power_of_attorney" | "general_legal_document",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "document_type_verification",
                "prompt_builder": build_document_type_verification_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: DocumentParser.parse_document_type_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "document_type" in output,
                "required": True
            })

            # Step 2: 주요 조항 추출 (문서 유형 기반)
            def build_clause_extraction_prompt(prev_output, initial_input):
                # prev_output은 Step 1의 결과 (document_type 포함)
                if not isinstance(prev_output, dict):
                    prev_output = {}

                verified_doc_type = prev_output.get("document_type", doc_type)
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:3000]  # 처음 3000자만

                if verified_doc_type == "contract":
                    return f"""다음 계약서 문서에서 주요 조항을 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 유형의 조항을 찾아주세요:
- payment (대금/지급): 대금, 금액, 지급, 결제 관련 조항
- period (기간/기한): 기간, 기한, 만료 관련 조항
- termination (해지/해제): 해지, 해제, 종료 관련 조항
- liability (책임): 책임, 손해배상, 위약금 관련 조항
- confidentiality (비밀/기밀): 비밀, 기밀, 보안 관련 조항

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "payment",
            "text": "제1조 대금은...",
            "article_number": "제1조"
        }},
        ...
    ],
    "clause_count": 5
}}
"""
                elif verified_doc_type == "complaint":
                    return f"""다음 고소장 문서에서 주요 요소를 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 요소를 찾아주세요:
- parties (당사자): 피고소인, 고소인, 피해자, 가해자 등
- facts (사실관계): 사실관계, 경위, 내용 등
- claims (청구사항): 청구, 요구, 주장 등

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "parties",
            "text": "피고소인: ...",
            "found": true
        }},
        ...
    ],
    "clause_count": 3
}}
"""
                else:
                    # 일반 문서
                    return f"""다음 법률 문서에서 주요 내용을 추출해주세요.

문서 내용:
{doc_text[:3000]}

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "type": "general",
            "text": "주요 내용 1...",
            "summary": "요약"
        }},
        ...
    ],
    "clause_count": 3
}}
"""

            chain_steps.append({
                "name": "clause_extraction",
                "prompt_builder": build_clause_extraction_prompt,
                "input_extractor": lambda prev: prev,  # Step 1의 출력 사용
                "output_parser": lambda response, prev: DocumentParser.parse_clause_extraction_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "key_clauses" in output,
                "required": True
            })

            # Step 3: 문제점 식별 (조항 기반)
            def build_issue_identification_prompt(prev_output, initial_input):
                # prev_output은 Step 2의 결과 (key_clauses 포함)
                if not isinstance(prev_output, dict):
                    prev_output = {}

                key_clauses = prev_output.get("key_clauses", [])
                verified_doc_type = initial_input.get("verified_doc_type") or doc_type
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:2000]

                # Step 1에서 문서 유형 가져오기
                if hasattr(chain_executor, 'chain_history'):
                    for step in chain_executor.chain_history:
                        if step.get("step_name") == "document_type_verification" and step.get("success"):
                            step_output = step.get("output", {})
                            if isinstance(step_output, dict):
                                verified_doc_type = step_output.get("document_type", doc_type)
                                break

                if verified_doc_type == "contract":
                    return f"""다음 계약서의 조항을 분석하여 잠재적 문제점을 식별해주세요.

문서 내용 (일부):
{doc_text[:2000]}

추출된 주요 조항:
{chr(10).join([f"- {clause.get('type', 'unknown')}: {clause.get('text', '')[:100]}..." for clause in key_clauses[:5]])}

다음 관점에서 문제점을 찾아주세요:
1. 필수 조항 누락: 대금, 기간, 해지 조항 등
2. 불명확한 표현: "기타", "등등", "적절한" 등
3. 불공정한 조항: 일방적 불리한 조건
4. 법적 문제: 법령 위반 가능성

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_clause" | "vague_term" | "unfair_clause" | "legal_issue",
            "description": "문제점 설명",
            "recommendation": "개선 권고사항"
        }},
        ...
    ],
    "issue_count": 3
}}
"""
                elif verified_doc_type == "complaint":
                    return f"""다음 고소장의 요소를 분석하여 문제점을 식별해주세요.

문서 내용 (일부):
{doc_text[:2000]}

추출된 주요 요소:
{chr(10).join([f"- {clause.get('type', 'unknown')}" for clause in key_clauses[:5]])}

다음 관점에서 문제점을 찾아주세요:
1. 필수 요소 누락: 피고소인, 사실관계, 청구사항 등
2. 불명확한 사실: 모호한 서술, 불충분한 증거 제시
3. 법적 요건 미비: 고소 요건 불충족 가능성

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_element" | "vague_facts" | "insufficient_evidence",
            "description": "문제점 설명",
            "recommendation": "개선 권고사항"
        }},
        ...
    ],
    "issue_count": 2
}}
"""
                else:
                    # 일반 문서는 문제점 식별 생략 가능
                    return None

            chain_steps.append({
                "name": "issue_identification",
                "prompt_builder": build_issue_identification_prompt,
                "input_extractor": lambda prev: prev,  # Step 2의 출력 사용
                "output_parser": lambda response, prev: DocumentParser.parse_issue_identification_response_with_context(response, prev),
                "validator": lambda output: output is None or (isinstance(output, dict) and "issues" in output),
                "required": False,  # 선택 단계 (일반 문서는 생략 가능)
                "skip_if": lambda prev: not isinstance(prev, dict) or not prev.get("key_clauses")
            })

            # Step 4: 요약 생성 (조항 + 문제점 기반)
            def build_summary_generation_prompt(prev_output, initial_input):
                # prev_output은 Step 3의 결과 (issues 포함) 또는 Step 2의 결과 (key_clauses 포함)
                if not isinstance(prev_output, dict):
                    prev_output = {}

                # Step 3 결과에서 key_clauses와 issues 모두 가져오기 (통합된 결과)
                key_clauses = prev_output.get("key_clauses", [])
                issues = prev_output.get("issues", [])
                verified_doc_type = prev_output.get("document_type") or initial_input.get("verified_doc_type") or doc_type

                return f"""다음 문서 분석 결과를 바탕으로 요약을 생성해주세요.

문서 유형: {verified_doc_type}
주요 조항 수: {len(key_clauses)}
발견된 문제점 수: {len(issues)}

다음 형식으로 요약을 작성해주세요:

문서 유형: {verified_doc_type}
분석된 조항 수: {len(key_clauses)}
발견된 문제점: {len(issues)}

주요 문제점:
{chr(10).join([f"- {issue.get('description', '')}" for issue in issues[:3]]) if issues else "없음"}

위 형식에 맞춰 간결하게 요약해주세요.
"""

            # Step 4의 input_extractor: 이전 단계 결과 통합
            def extract_summary_input(prev_output):
                # prev_output은 Step 3의 결과 또는 Step 2의 결과
                if not isinstance(prev_output, dict):
                    return prev_output

                # Step 3 결과에서 key_clauses와 issues 통합 (Step 3가 key_clauses도 포함하도록 함)
                result = {
                    "key_clauses": prev_output.get("key_clauses", []),
                    "issues": prev_output.get("issues", []),
                    "document_type": prev_output.get("document_type", doc_type)
                }
                return result

            chain_steps.append({
                "name": "summary_generation",
                "prompt_builder": build_summary_generation_prompt,
                "input_extractor": extract_summary_input,  # 이전 단계의 통합 결과 사용
                "output_parser": lambda response, prev: self._normalize_answer(response),
                "validator": lambda output: output and len(output.strip()) > 10,
                "required": True
            })

            # Step 5의 input_extractor: Step 3 결과 찾기
            def extract_improvement_input(prev_output):
                # prev_output은 Step 4의 결과 (summary 문자열)
                # Step 3 결과를 찾기 위해 이전 단계 출력들을 확인해야 하지만,
                # prompt_builder 시점에는 체인 히스토리가 없으므로
                # None을 반환하면 건너뛰기 (체인 실행 후에 issues 확인)
                return prev_output

            # Step 5: 개선 권고 생성 (문제점 기반)
            def build_improvement_recommendations_prompt(prev_output, initial_input):
                # prev_output은 Step 4의 결과 (summary 문자열)
                # Step 3의 issues를 찾기 위해서는 체인 히스토리가 필요한데,
                # prompt_builder 시점에는 아직 체인 히스토리가 없음
                # 따라서 Step 5는 체인 실행 후에 issues를 확인하여 조건부로 실행
                # 여기서는 항상 None 반환하여 prompt_builder에서 건너뛰고,
                # 체인 실행 후에 issues가 있으면 별도로 실행
                return None  # 항상 건너뛰기 (체인 실행 후 처리)

                verified_doc_type = initial_input.get("verified_doc_type") or doc_type
                doc_text = initial_input.get("text") if isinstance(initial_input, dict) else text[:1500]

                return f"""다음 문서 분석 결과를 바탕으로 개선 권고사항을 작성해주세요.

문서 유형: {verified_doc_type}
문서 내용 (일부):
{doc_text[:1500]}

발견된 문제점:
{chr(10).join([f"{idx+1}. [{issue.get('severity', 'medium')}] {issue.get('description', '')}" for idx, issue in enumerate(issues[:5])])}

다음 형식으로 개선 권고사항을 작성해주세요:
{{
    "recommendations": [
        {{
            "priority": "high" | "medium" | "low",
            "issue_type": "문제점 유형",
            "recommendation": "구체적인 개선 권고사항",
            "rationale": "권고 근거"
        }},
        ...
    ],
    "recommendation_count": 3
}}
"""

            # Step 5는 prompt_builder에서 None을 반환하여 건너뛰고,
            # 체인 실행 후에 issues가 있으면 별도로 실행
            # 여기서는 체인에 추가하지 않고, 체인 실행 후에 조건부로 처리

            # 체인 실행
            initial_input_dict = {
                "text": text,
                "doc_type": doc_type
            }

            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=2,
                stop_on_failure=False
            )

            # 결과 추출 및 통합
            chain_history = chain_result.get("chain_history", [])

            # Step 1: 문서 유형 확인
            verified_doc_type = doc_type
            for step in chain_history:
                if step.get("step_name") == "document_type_verification" and step.get("success"):
                    step_output = step.get("output", {})
                    if isinstance(step_output, dict):
                        verified_doc_type = step_output.get("document_type", doc_type)
                        break

            # Step 2: 주요 조항 추출
            key_clauses = []
            for step in chain_history:
                if step.get("step_name") == "clause_extraction" and step.get("success"):
                    step_output = step.get("output", {})
                    if isinstance(step_output, dict):
                        key_clauses = step_output.get("key_clauses", [])
                        break

            # Step 3: 문제점 식별
            issues = []
            for step in chain_history:
                if step.get("step_name") == "issue_identification" and step.get("success"):
                    step_output = step.get("output", {})
                    if isinstance(step_output, dict):
                        issues = step_output.get("issues", [])
                        break

            # Step 4: 요약 생성
            summary = ""
            for step in chain_history:
                if step.get("step_name") == "summary_generation" and step.get("success"):
                    summary = step.get("output", "")
                    if isinstance(summary, str):
                        break
                    elif isinstance(summary, dict):
                        summary = summary.get("summary", "")
                        break

            # Step 5: 개선 권고 (조건부 실행 - issues가 있는 경우)
            recommendations = []
            if issues and len(issues) > 0:
                try:
                    # issues가 있으면 개선 권고 생성
                    improvement_prompt = f"""다음 문서 분석 결과를 바탕으로 개선 권고사항을 작성해주세요.

문서 유형: {verified_doc_type}
문서 내용 (일부):
{text[:1500]}

발견된 문제점:
{chr(10).join([f"{idx+1}. [{issue.get('severity', 'medium')}] {issue.get('description', '')}" for idx, issue in enumerate(issues[:5])])}

다음 형식으로 개선 권고사항을 작성해주세요:
{{
    "recommendations": [
        {{
            "priority": "high" | "medium" | "low",
            "issue_type": "문제점 유형",
            "recommendation": "구체적인 개선 권고사항",
            "rationale": "권고 근거"
        }},
        ...
    ],
    "recommendation_count": 3
}}
"""
                    llm = self.llm_fast if hasattr(self, 'llm_fast') and self.llm_fast else self.llm
                    improvement_response = llm.invoke(improvement_prompt)
                    improvement_content = self._extract_response_content(improvement_response)
                    improvement_result = DocumentParser.parse_improvement_recommendations_response(improvement_content)
                    if improvement_result and isinstance(improvement_result, dict):
                        recommendations = improvement_result.get("recommendations", [])
                except Exception as e:
                    self.logger.warning(f"Failed to generate improvement recommendations: {e}")

            # 결과 통합
            analysis_result = {
                "document_type": verified_doc_type,
                "key_clauses": key_clauses,
                "issues": issues,
                "summary": summary if summary else self._generate_document_summary_fallback(text, verified_doc_type, key_clauses, issues),
                "recommendations": recommendations
            }

            # 체인 실행 결과 로깅
            chain_summary = chain_executor.get_chain_summary()
            self.logger.info(
                f"✅ [DOCUMENT CHAIN] Executed {chain_summary['total_steps']} steps, "
                f"{chain_summary['successful_steps']} successful, "
                f"{chain_summary['failed_steps']} failed, "
                f"Total time: {chain_summary['total_time']:.2f}s"
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ [DOCUMENT CHAIN ERROR] Prompt chain failed: {e}")
            # 폴백: 기존 방식 사용
            return self._analyze_legal_document(text, doc_type)

    def _generate_document_summary(self, text: str, doc_type: str, analysis: Dict[str, Any]) -> str:
        """문서 요약 생성"""
        summary_parts = [f"문서 유형: {doc_type}"]
        summary_parts.append(f"분석된 조항 수: {len(analysis.get('key_clauses', []))}")
        summary_parts.append(f"발견된 문제점: {len(analysis.get('issues', []))}")

        if analysis.get("issues"):
            summary_parts.append("\n주요 문제점:")
            for issue in analysis["issues"][:3]:
                summary_parts.append(f"- {issue['description']}")

        return "\n".join(summary_parts)

    def _generate_document_summary_fallback(self, text: str, doc_type: str, key_clauses: List[Dict], issues: List[Dict]) -> str:
        """문서 요약 생성 (폴백)"""
        summary_parts = [f"문서 유형: {doc_type}"]
        summary_parts.append(f"분석된 조항 수: {len(key_clauses)}")
        summary_parts.append(f"발견된 문제점: {len(issues)}")

        if issues:
            summary_parts.append("\n주요 문제점:")
            for issue in issues[:3]:
                if isinstance(issue, dict):
                    summary_parts.append(f"- {issue.get('description', '')}")
                else:
                    summary_parts.append(f"- {str(issue)}")

        return "\n".join(summary_parts)

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
        """문서 분석 요약 생성"""
        summary_parts = [f"## 업로드 문서 분석 ({analysis['document_type']})"]

        if analysis.get("key_clauses"):
            summary_parts.append("\n### 주요 조항")
            for clause in analysis["key_clauses"][:3]:
                summary_parts.append(f"- {clause['type']}: {clause['text'][:100]}...")

        if analysis.get("issues"):
            summary_parts.append("\n### 잠재 문제점")
            for issue in analysis["issues"]:
                summary_parts.append(f"- [{issue['severity']}] {issue['description']}")

        return "\n".join(summary_parts)

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

    @observe(name="prepare_search_query")
    @with_state_optimization("prepare_search_query", enable_reduction=False)
    def prepare_search_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 쿼리 준비 및 최적화 전용 노드 (Part 2)"""
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
            state["metadata"]["_last_executed_node"] = "prepare_search_query"

            if "common" not in state or not isinstance(state.get("common"), dict):
                state["common"] = {}
            if "metadata" not in state["common"]:
                state["common"]["metadata"] = {}
            state["common"]["metadata"]["_last_executed_node"] = "prepare_search_query"

            # 중요: 노드 시작 시 input 그룹 보장
            # LangGraph가 이전 노드의 결과만 전달하는 경우, input이 사라질 수 있음
            # input 그룹 확인 및 생성
            if "input" not in state or not isinstance(state.get("input"), dict):
                state["input"] = {}

            # query가 없으면 여러 위치에서 찾기
            current_query = state["input"].get("query", "")
            if not current_query:
                # 1. 최상위 레벨에서 찾기
                query_from_top = state.get("query", "")
                session_id_from_top = state.get("session_id", "")
                if query_from_top:
                    state["input"]["query"] = query_from_top
                    if session_id_from_top:
                        state["input"]["session_id"] = session_id_from_top
                # 2. search 그룹에서 찾기
                elif "search" in state and isinstance(state["search"], dict):
                    search_query = state["search"].get("search_query", "")
                    if search_query:
                        state["input"]["query"] = search_query

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

            # 쿼리 정보 가져오기 및 검증 (여러 방법 시도)
            # 중요: state의 input 그룹에서 query 가져오기 (이미 시작 부분에서 복원했어야 함)
            query = None

            # 우선순위 1: state["input"]["query"]
            if "input" in state and isinstance(state["input"], dict):
                query = state["input"].get("query", "")

            # 우선순위 2: _get_state_value 사용
            if not query or not str(query).strip():
                query = self._get_state_value(state, "query", "")

            # 우선순위 3: state에서 직접 읽기
            if not query or not str(query).strip():
                if isinstance(state, dict) and "query" in state:
                    query = state["query"]

            search_query = self._get_state_value(state, "search_query") or query

            # query가 빈 문자열이면 에러
            if not query or not str(query).strip():
                self.logger.error(f"prepare_search_query: query is empty! State keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
                if "input" in state:
                    self.logger.error(f"prepare_search_query: state['input'] = {state['input']}")
                # Phase 1/Phase 7: 에러 메시지 설정 - _set_answer_safely 사용
                self._set_answer_safely(state, "죄송합니다. 질문을 이해하지 못했습니다. 다시 질문해주세요.")
                return state

            # query_type 정규화
            query_type_raw = self._get_state_value(state, "query_type", "")
            query_type_str = self._get_query_type_str(query_type_raw)
            # 프롬프트용으로 한번 더 정규화
            query_type_str = self._normalize_query_type_for_prompt(query_type_str)

            # extracted_keywords 검증
            extracted_keywords_raw = self._get_state_value(state, "extracted_keywords", [])
            if not isinstance(extracted_keywords_raw, list):
                self.logger.warning(f"extracted_keywords is not a list: {type(extracted_keywords_raw)}, converting to empty list")
                extracted_keywords = []
            else:
                # 유효한 키워드만 필터링
                extracted_keywords = [kw for kw in extracted_keywords_raw if kw and isinstance(kw, str) and len(str(kw).strip()) > 0]

            # legal_field 검증
            legal_field_raw = self._get_state_value(state, "legal_field", "")
            legal_field = str(legal_field_raw).strip() if legal_field_raw else ""

            # 로깅: 전달되는 데이터 확인
            self.logger.debug(
                f"📋 [PREPARE SEARCH QUERY] Data for query optimization:\n"
                f"   query: '{query[:50]}{'...' if len(query) > 50 else ''}'\n"
                f"   search_query: '{search_query[:50]}{'...' if len(search_query) > 50 else ''}'\n"
                f"   query_type (raw): '{query_type_raw}' → (normalized): '{query_type_str}'\n"
                f"   extracted_keywords: {len(extracted_keywords)} items {extracted_keywords[:5] if extracted_keywords else '[]'}\n"
                f"   legal_field: '{legal_field}'"
            )

            is_retry = (last_executed_node == "validate_answer_quality")

            # 검색 쿼리 최적화
            optimized_queries = self._optimize_search_query(
                query=search_query,
                query_type=query_type_str,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field
            )

            # 재시도 시 추가 개선
            if is_retry:
                quality_feedback = self.answer_generator.get_quality_feedback_for_retry(state)
                improved_query = self._improve_search_query_for_retry(
                    optimized_queries["semantic_query"],
                    quality_feedback,
                    state
                )
                if improved_query != optimized_queries["semantic_query"]:
                    self.logger.info(
                        f"🔍 [SEARCH RETRY] Improved query: '{optimized_queries['semantic_query']}' → '{improved_query}'"
                    )
                    optimized_queries["semantic_query"] = improved_query
                    optimized_queries["keyword_queries"][0] = improved_query

            # 검색 파라미터 결정
            search_params = self._determine_search_parameters(
                query_type=query_type_str,
                query_complexity=len(query),
                keyword_count=len(extracted_keywords),
                is_retry=is_retry
            )

            # 최적화된 쿼리와 파라미터를 state에 저장
            self._set_state_value(state, "optimized_queries", optimized_queries)
            self._set_state_value(state, "search_params", search_params)
            self._set_state_value(state, "is_retry_search", is_retry)
            self._set_state_value(state, "search_start_time", start_time)

            # semantic_query 검증 및 수정 (빈 문자열이면 기본 쿼리 사용)
            semantic_query_created = optimized_queries.get("semantic_query", "")
            if not semantic_query_created or not str(semantic_query_created).strip():
                self.logger.warning(f"semantic_query is empty, using base query: '{query[:50]}...'")
                optimized_queries["semantic_query"] = query
                semantic_query_created = query
                # 수정된 값을 다시 저장
                self._set_state_value(state, "optimized_queries", optimized_queries)

            # keyword_queries도 확인 및 수정
            keyword_queries_created = optimized_queries.get("keyword_queries", [])
            if not keyword_queries_created or len(keyword_queries_created) == 0:
                self.logger.warning("keyword_queries is empty, using base query")
                optimized_queries["keyword_queries"] = [query]
                keyword_queries_created = [query]
                # 수정된 값을 다시 저장
                self._set_state_value(state, "optimized_queries", optimized_queries)

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

        # 중요: 반환 전에 input 그룹 보장 (LangGraph state 병합 시 보존)
        if "input" not in state or not isinstance(state.get("input"), dict):
            state["input"] = {}
        if not state["input"].get("query"):
            query_value = self._get_state_value(state, "query", "")
            session_id_value = self._get_state_value(state, "session_id", "")
            if query_value:
                state["input"]["query"] = query_value
                if session_id_value:
                    state["input"]["session_id"] = session_id_value
                self.logger.debug(f"Ensured input group in state after prepare_search_query: query length={len(query_value)}")

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

    @observe(name="process_search_results_combined")
    @with_state_optimization("process_search_results_combined", enable_reduction=True)
    def process_search_results_combined(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검색 결과 처리 통합 노드 (6개 노드를 1개로 병합)"""
        # 개선: print 문을 추가하여 즉시 출력 보장 (로거 설정과 무관)
        import sys
        print("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작", flush=True, file=sys.stdout)

        # 개선 1: 함수 시작 부분에 즉시 로깅 추가 (이중 보장)
        self.logger.info("🔄 [SEARCH RESULTS] process_search_results_combined 실행 시작")

        try:
            start_time = time.time()

            # 검색 결과 가져오기
            semantic_results = self._get_state_value(state, "semantic_results", [])
            keyword_results = self._get_state_value(state, "keyword_results", [])
            semantic_count = self._get_state_value(state, "semantic_count", 0)
            keyword_count = self._get_state_value(state, "keyword_count", 0)
            query = self._get_state_value(state, "query", "")
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            search_params = self._get_state_value(state, "search_params", {})
            extracted_keywords = self._get_state_value(state, "extracted_keywords", [])

            # 개선 1: 입력 검증 로깅 (print + logger)
            input_msg = f"📥 [SEARCH RESULTS] 입력 데이터 - semantic: {len(semantic_results)}, keyword: {len(keyword_results)}, semantic_count: {semantic_count}, keyword_count: {keyword_count}"
            print(input_msg, flush=True, file=sys.stdout)
            self.logger.info(input_msg)

            # 1. 품질 평가 (기존 evaluate_search_quality 로직)
            semantic_quality = self._evaluate_semantic_search_quality(
                semantic_results=semantic_results,
                query=query,
                query_type=query_type_str,
                min_results=search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K) // 2
            )

            keyword_quality = self._evaluate_keyword_search_quality(
                keyword_results=keyword_results,
                query=query,
                query_type=query_type_str,
                min_results=search_params.get("keyword_limit", WorkflowConstants.CATEGORY_SEARCH_LIMIT) // 2
            )

            overall_quality = (semantic_quality["score"] + keyword_quality["score"]) / 2.0
            needs_retry = semantic_quality["needs_retry"] or keyword_quality["needs_retry"]

            quality_evaluation = {
                "semantic_quality": semantic_quality,
                "keyword_quality": keyword_quality,
                "overall_quality": overall_quality,
                "needs_retry": needs_retry
            }

            self._set_state_value(state, "search_quality_evaluation", quality_evaluation)

            # 2. 조건부 재검색 (기존 conditional_retry_search 로직)
            if needs_retry and overall_quality < 0.6 and semantic_count + keyword_count < 10:
                self.logger.info(f"검색 품질 낮음 (점수: {overall_quality:.2f}), 재검색 수행...")
                try:
                    # 재검색 로직 (간단한 버전)
                    retry_semantic = []
                    retry_keyword = []

                    if semantic_quality["needs_retry"]:
                        # 의미 검색 재시도
                        optimized_queries = self._get_state_value(state, "optimized_queries", {})
                        retry_semantic = self._execute_semantic_search_internal(
                            optimized_queries, search_params, query
                        )[0][:5]  # 최대 5개

                    if keyword_quality["needs_retry"]:
                        # 키워드 검색 재시도
                        optimized_queries = self._get_state_value(state, "optimized_queries", {})
                        retry_keyword = self._execute_keyword_search_internal(
                            optimized_queries, search_params, query_type_str,
                            self._get_state_value(state, "legal_field", ""),
                            extracted_keywords, query
                        )[0][:5]  # 최대 5개

                    # 기존 결과에 추가
                    semantic_results.extend(retry_semantic)
                    keyword_results.extend(retry_keyword)
                    semantic_count += len(retry_semantic)
                    keyword_count += len(retry_keyword)
                except Exception as e:
                    self.logger.warning(f"재검색 실패: {e}")

            # 3. 병합 및 재순위 (기존 merge_and_rerank 로직)
            # result_merger.merge_results는 Dict[str, List[Dict]] 형태의 exact_results를 기대하므로
            # keyword_results를 dict 형태로 변환
            exact_results_dict = {
                "keyword": keyword_results if isinstance(keyword_results, list) else []
            } if keyword_results else {}

            merged_results = self.result_merger.merge_results(
                exact_results=exact_results_dict,
                semantic_results=semantic_results if isinstance(semantic_results, list) else [],
                weights={"exact": 0.7, "semantic": 0.3}
            )

            # MergedResult 객체를 dict로 변환
            merged_docs = []
            for merged_result in merged_results:
                if hasattr(merged_result, 'text'):
                    text_value = merged_result.text

                    # 개선 1.2: text가 비어있으면 다른 속성 시도
                    if not text_value or len(str(text_value).strip()) == 0:
                        # content 속성 확인
                        if hasattr(merged_result, 'content'):
                            text_value = merged_result.content
                        # metadata에서 확인
                        elif hasattr(merged_result, 'metadata') and isinstance(merged_result.metadata, dict):
                            text_value = (
                                merged_result.metadata.get('content') or
                                merged_result.metadata.get('text') or
                                merged_result.metadata.get('document') or
                                ''
                            )

                    # 최종적으로 text_value가 비어있으면 경고 및 로깅
                    if not text_value or len(str(text_value).strip()) == 0:
                        source_name = getattr(merged_result, 'source', 'Unknown')
                        score_value = getattr(merged_result, 'score', 0.0)
                        warning_msg = f"⚠️ [DEBUG] MergedResult text가 비어있음 - source: {source_name}, score: {score_value:.3f}"
                        print(warning_msg, flush=True, file=sys.stdout)
                        self.logger.warning(warning_msg)

                    merged_docs.append({
                        "content": str(text_value) if text_value else "",
                        "text": str(text_value) if text_value else "",
                        "relevance_score": getattr(merged_result, 'score', 0.0),
                        "source": getattr(merged_result, 'source', 'Unknown'),
                        "metadata": getattr(merged_result, 'metadata', {}) if hasattr(merged_result, 'metadata') else {}
                    })
                elif isinstance(merged_result, dict):
                    # 개선 1: dict 형태의 문서도 content/text 필드 보장
                    doc = merged_result.copy()
                    # content 또는 text 필드가 없으면 추가
                    if "content" not in doc and "text" in doc:
                        doc["content"] = doc["text"]
                    elif "text" not in doc and "content" in doc:
                        doc["text"] = doc["content"]
                    elif "content" not in doc and "text" not in doc:
                        # content와 text가 모두 없으면 빈 문자열로 설정
                        doc["content"] = ""
                        doc["text"] = ""
                    merged_docs.append(doc)

            # 개선 1: merged_docs 문서 구조 검증 및 로깅
            doc_structure_stats = {
                "total": len(merged_docs),
                "has_content": 0,
                "has_text": 0,
                "has_both": 0,
                "content_lengths": []
            }
            for doc in merged_docs[:5]:  # 상위 5개만 검사
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

            structure_msg = f"📋 [SEARCH RESULTS] merged_docs 구조 분석 - Total: {doc_structure_stats['total']}, Has content: {doc_structure_stats['has_content']}, Has text: {doc_structure_stats['has_text']}, Has both: {doc_structure_stats['has_both']}, Avg content length: {sum(doc_structure_stats['content_lengths'])/len(doc_structure_stats['content_lengths']) if doc_structure_stats['content_lengths'] else 0:.1f}"
            print(structure_msg, flush=True, file=sys.stdout)
            self.logger.info(structure_msg)

            # 키워드 가중치 적용 (merge_and_rerank_with_keyword_weights와 동일한 로직)
            # 키워드 중요도 가중치 계산
            keyword_weights = self._calculate_keyword_weights(
                extracted_keywords=extracted_keywords,
                query=query,
                query_type=query_type_str,
                legal_field=self._get_state_value(state, "legal_field", "")
            )

            # 각 문서에 키워드 매칭 점수 계산 및 가중치 적용
            weighted_docs = []
            for doc in merged_docs:
                # 키워드 매칭 점수 계산
                keyword_scores = self._calculate_keyword_match_score(
                    document=doc,
                    keyword_weights=keyword_weights,
                    query=query
                )

                # 가중치 적용 최종 점수 계산
                final_score = self._calculate_weighted_final_score(
                    document=doc,
                    keyword_scores=keyword_scores,
                    search_params=search_params,
                    query_type=query_type_str
                )

                # 문서에 점수 정보 추가
                doc["keyword_match_score"] = keyword_scores.get("keyword_match_score", 0.0)
                doc["keyword_coverage"] = keyword_scores.get("keyword_coverage", 0.0)
                doc["matched_keywords"] = keyword_scores.get("matched_keywords", [])
                doc["weighted_keyword_score"] = keyword_scores.get("weighted_keyword_score", 0.0)
                doc["final_weighted_score"] = final_score

                weighted_docs.append(doc)

            # Reranking 수행 (점수순 정렬)
            weighted_docs.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)

            # 상세 로깅: 점수 분포 분석 (print + logger)
            if weighted_docs:
                scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in weighted_docs]
                min_score = min(scores)
                max_score = max(scores)
                avg_score = sum(scores) / len(scores) if scores else 0.0
                score_dist_msg = f"📊 [SEARCH RESULTS] Score distribution after weighting - Total: {len(weighted_docs)}, Min: {min_score:.3f}, Max: {max_score:.3f}, Avg: {avg_score:.3f}"
                print(score_dist_msg, flush=True, file=sys.stdout)
                self.logger.info(score_dist_msg)

            # 4. 필터링 및 검증 (기존 filter_and_validate_results 로직)
            # 개선 1: 필터링 전 문서 구조 확인
            if weighted_docs:
                sample_doc = weighted_docs[0]
                sample_structure = f"Sample doc keys: {list(sample_doc.keys())}, has content: {'content' in sample_doc}, has text: {'text' in sample_doc}, content type: {type(sample_doc.get('content', 'N/A')).__name__}"
                print(f"🔍 [SEARCH RESULTS] {sample_structure}", flush=True, file=sys.stdout)
                self.logger.debug(sample_structure)

            filtered_docs = []
            skipped_content = 0
            skipped_score = 0
            skipped_content_details = []  # 디버깅용

            for doc in weighted_docs:
                # 개선 1: content 추출 로직 개선 - 다양한 필드명 시도
                content = (
                    doc.get("content", "") or
                    doc.get("text", "") or
                    doc.get("content_text", "") or
                    doc.get("document", "") or
                    str(doc.get("metadata", {}).get("content", "")) or
                    str(doc.get("metadata", {}).get("text", "")) or
                    ""
                )

                # content가 문자열이 아니면 문자열로 변환
                if not isinstance(content, str):
                    content = str(content) if content else ""

                # 개선 1: 필터링 조건 완화 (10자 → 5자로 낮춤)
                if not content or len(content.strip()) < 5:  # 10자에서 5자로 낮춤
                    skipped_content += 1
                    # 디버깅: 첫 3개만 상세 정보 수집
                    if skipped_content <= 3:
                        skipped_content_details.append({
                            "keys": list(doc.keys()),
                            "content_type": type(doc.get("content", None)).__name__,
                            "text_type": type(doc.get("text", None)).__name__,
                            "content_len": len(str(doc.get("content", ""))),
                            "text_len": len(str(doc.get("text", "")))
                        })
                    continue

                # 관련성 점수 확인 (Phase 1: 임계값 0.1 → 0.05로 조정)
                score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                if score < 0.05:  # 개선: 임계값을 0.05로 낮춤 (더 많은 문서 포함)
                    skipped_score += 1
                    continue

                filtered_docs.append(doc)

            # 상세 로깅: 필터링 단계별 문서 수 (print + logger)
            filter_stats_msg = f"📊 [SEARCH RESULTS] Filtering statistics - Merged: {len(merged_docs)}, Weighted: {len(weighted_docs)}, Filtered: {len(filtered_docs)}, Skipped (content): {skipped_content}, Skipped (score): {skipped_score}"
            print(filter_stats_msg, flush=True, file=sys.stdout)
            self.logger.info(filter_stats_msg)

            # 개선 1: content 필터링에서 제외된 문서 상세 정보 (디버깅용)
            if skipped_content > 0 and skipped_content_details:
                skip_details_msg = f"⚠️ [SEARCH RESULTS] Content 필터링 제외 상세 (상위 {len(skipped_content_details)}개): {skipped_content_details}"
                print(skip_details_msg, flush=True, file=sys.stdout)
                self.logger.warning(skip_details_msg)

            # 최대 문서 수 제한
            max_docs = self.config.max_retrieved_docs or 20
            final_docs = filtered_docs[:max_docs]

            # 개선: 검색 결과가 없을 때 명확한 로깅 및 폴백 전략 적용
            if not final_docs:
                self.logger.warning(
                    f"⚠️ [SEARCH RESULTS] No valid documents found after filtering. "
                    f"Query: '{query[:50]}...', Query type: {query_type_str}, "
                    f"Total merged: {len(merged_docs)}, Filtered: {len(filtered_docs)}"
                )

                # 폴백: 낮은 점수라도 문서가 있으면 사용 (최소 1개라도 제공)
                if weighted_docs:
                    # 점수 순으로 정렬되어 있으므로, 상위 3개를 선택 (점수가 낮아도)
                    fallback_docs = []
                    for doc in weighted_docs[:3]:
                        content = doc.get("content", "") or doc.get("text", "")
                        if content and len(content.strip()) >= 10:
                            # 점수가 낮아도 최소 임계값(0.05) 이상이면 사용
                            score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                            if score >= 0.05:
                                fallback_docs.append(doc)

                    if fallback_docs:
                        final_docs = fallback_docs
                        self.logger.info(
                            f"🔄 [FALLBACK] Using {len(final_docs)} lower-scored documents "
                            f"as fallback (original filtered count: 0)"
                        )
                    else:
                        self.logger.error(
                            f"❌ [SEARCH RESULTS] No fallback documents available. "
                            f"All documents were filtered out (content too short or score too low)."
                        )
                else:
                    self.logger.error(
                        f"❌ [SEARCH RESULTS] No documents available at all. "
                        f"Search may have failed or returned empty results."
                    )

            # 5. 메타데이터 업데이트 (기존 update_search_metadata 로직)
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

            # 개선 2: final_docs 설정 후 즉시 로깅
            self.logger.info(
                f"📊 [SEARCH RESULTS] final_docs 설정 완료 - 개수: {len(final_docs)}"
            )

            # Phase 3: final_docs가 0개일 때 semantic_results를 직접 변환
            if not final_docs or len(final_docs) == 0:
                fallback_warning_msg = f"⚠️ [SEARCH RESULTS] final_docs가 0개입니다. semantic_results에서 변환 시도..."
                print(fallback_warning_msg, flush=True, file=sys.stdout)
                self.logger.warning(fallback_warning_msg)
                if semantic_results and len(semantic_results) > 0:
                    # semantic_results를 retrieved_docs 형식으로 변환
                    converted_docs = []
                    for doc in semantic_results[:10]:  # 최대 10개
                        if isinstance(doc, dict):
                            converted_doc = {
                                "content": doc.get("content", "") or doc.get("text", ""),
                                "text": doc.get("text", "") or doc.get("content", ""),
                                "source": doc.get("source", "") or doc.get("title", "Unknown"),
                                "relevance_score": doc.get("relevance_score", 0.5),
                                "search_type": "semantic",
                                "metadata": doc.get("metadata", {})
                            }
                            if converted_doc["content"] and len(converted_doc["content"].strip()) >= 10:
                                converted_docs.append(converted_doc)

                    if converted_docs:
                        final_docs = converted_docs
                        fallback_success_msg = f"🔄 [FALLBACK] Converted {len(final_docs)} documents from semantic_results to retrieved_docs (original final_docs count: 0)"
                        print(fallback_success_msg, flush=True, file=sys.stdout)
                        self.logger.info(fallback_success_msg)
                    else:
                        fallback_error_msg = f"❌ [SEARCH RESULTS] semantic_results에서도 변환 실패 - semantic_results 개수: {len(semantic_results)}"
                        print(fallback_error_msg, flush=True, file=sys.stdout)
                        self.logger.error(fallback_error_msg)

            # 개선 2: State 저장 전 검증 (print + logger)
            save_before_msg = f"💾 [SEARCH RESULTS] State 저장 전 검증 - final_docs 개수: {len(final_docs)}, 타입: {type(final_docs).__name__}"
            print(save_before_msg, flush=True, file=sys.stdout)
            self.logger.info(save_before_msg)

            # 6. State 저장 (Phase 2: 저장 경로 보장 - 최상위 + search + common 그룹)
            self._set_state_value(state, "retrieved_docs", final_docs)
            self._set_state_value(state, "merged_documents", final_docs)

            # search 그룹에도 저장
            if "search" not in state:
                state["search"] = {}
            state["search"]["retrieved_docs"] = final_docs
            state["search"]["merged_documents"] = final_docs

            # common 그룹에도 저장 (State Reduction 후에도 유지)
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["retrieved_docs"] = final_docs
            state["common"]["search"]["merged_documents"] = final_docs

            # 개선 2: State 저장 후 검증 (print + logger)
            saved_retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
            saved_search_group = state.get("search", {}).get("retrieved_docs", [])
            saved_common_group = state.get("common", {}).get("search", {}).get("retrieved_docs", [])
            save_after_msg = f"✅ [SEARCH RESULTS] State 저장 완료 - 최상위: {len(saved_retrieved_docs)}, search 그룹: {len(saved_search_group)}, common 그룹: {len(saved_common_group)}"
            print(save_after_msg, flush=True, file=sys.stdout)
            self.logger.info(save_after_msg)

            # 개선 3.1: 전역 캐시에도 직접 저장 시도 (State Reduction 전에 저장 보장)
            try:
                from core.agents.node_wrappers import _global_search_results_cache
                if not _global_search_results_cache:
                    _global_search_results_cache = {}

                _global_search_results_cache["retrieved_docs"] = final_docs
                _global_search_results_cache["merged_documents"] = final_docs

                if "search" not in _global_search_results_cache:
                    _global_search_results_cache["search"] = {}
                _global_search_results_cache["search"]["retrieved_docs"] = final_docs
                _global_search_results_cache["search"]["merged_documents"] = final_docs

                cache_save_msg = f"✅ [SEARCH RESULTS] 전역 캐시에 직접 저장 완료 - 개수: {len(final_docs)}"
                print(cache_save_msg, flush=True, file=sys.stdout)
                self.logger.info(cache_save_msg)
            except Exception as e:
                cache_error_msg = f"⚠️ [SEARCH RESULTS] 전역 캐시 직접 저장 실패: {e}"
                print(cache_error_msg, flush=True, file=sys.stdout)
                self.logger.warning(cache_error_msg)

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(
                state,
                "검색 결과 처리",
                f"검색 결과 처리 완료: {len(final_docs)}개 문서 (품질 점수: {overall_quality:.2f}, 시간: {processing_time:.3f}s)"
            )

            # 상세 로깅: 최종 결과 (print + logger)
            if len(final_docs) > 0:
                processed_msg = f"✅ [SEARCH RESULTS] Processed {len(final_docs)} documents (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
                print(processed_msg, flush=True, file=sys.stdout)
                self.logger.info(processed_msg)
                # 최종 문서들의 점수 분포
                final_scores = [doc.get("final_weighted_score", doc.get("relevance_score", 0.0)) for doc in final_docs]
                if final_scores:
                    final_score_msg = f"📊 [SEARCH RESULTS] Final documents score range - Min: {min(final_scores):.3f}, Max: {max(final_scores):.3f}, Avg: {sum(final_scores)/len(final_scores):.3f}"
                    print(final_score_msg, flush=True, file=sys.stdout)
                    self.logger.info(final_score_msg)
            else:
                no_docs_msg = f"⚠️ [SEARCH RESULTS] No documents available after processing (quality: {overall_quality:.2f}, retry: {needs_retry}, time: {processing_time:.3f}s)"
                print(no_docs_msg, flush=True, file=sys.stdout)
                self.logger.warning(no_docs_msg)

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

    @observe(name="execute_searches_parallel")
    @with_state_optimization("execute_searches_parallel", enable_reduction=True)
    def execute_searches_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """의미적 검색과 키워드 검색을 병렬로 실행"""
        try:
            from concurrent.futures import ThreadPoolExecutor

            start_time = time.time()

            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")

            # 디버깅: state에서 직접 확인
            from .state_helpers import get_field
            optimized_queries_raw = get_field(state, "optimized_queries")
            search_params_raw = get_field(state, "search_params")

            self.logger.debug(f"execute_searches_parallel: START")
            self.logger.debug(f"  - optimized_queries (via _get_state_value): {type(optimized_queries).__name__}, exists={bool(optimized_queries)}")
            self.logger.debug(f"  - optimized_queries (via get_field): {type(optimized_queries_raw).__name__}, is None={optimized_queries_raw is None}")
            self.logger.debug(f"  - search_params (via _get_state_value): {type(search_params).__name__}, exists={bool(search_params)}")
            self.logger.debug(f"  - search_params (via get_field): {type(search_params_raw).__name__}, is None={search_params_raw is None}")

            # state 구조 확인
            if "search" in state:
                self.logger.debug(f"  - state has 'search' key: {type(state['search']).__name__}")
                if isinstance(state.get("search"), dict):
                    self.logger.debug(f"  - search keys: {list(state['search'].keys())}")

            # state["search"]에서 직접 읽기 (get_field가 빈 딕셔너리를 반환하는 경우 대비)
            if "search" in state and isinstance(state["search"], dict):
                direct_optimized = state["search"].get("optimized_queries")
                if direct_optimized and isinstance(direct_optimized, dict) and len(direct_optimized) > 0:
                    optimized_queries = direct_optimized
                    extracted_keywords = optimized_queries.get("expanded_keywords", [])
                    self.logger.debug(f"  - Using direct state['search']['optimized_queries'], keys: {list(optimized_queries.keys())}")
                    self.logger.debug(f"  - semantic_query: '{optimized_queries.get('semantic_query', 'N/A')[:50]}...'")
                    self.logger.debug(f"  - keyword_queries: {len(optimized_queries.get('keyword_queries', []))} queries")
                elif optimized_queries_raw is not None and len(optimized_queries_raw) > 0:
                    optimized_queries = optimized_queries_raw
                    extracted_keywords = optimized_queries.get("expanded_keywords", [])
                    self.logger.debug(f"  - Using optimized_queries_raw (direct was empty), keys: {list(optimized_queries.keys())}")
                else:
                    optimized_queries = {}
                    extracted_keywords = []
                    self.logger.debug(f"  - Both direct and raw are empty/None")
            elif optimized_queries_raw is not None and len(optimized_queries_raw) > 0:
                optimized_queries = optimized_queries_raw
                extracted_keywords = optimized_queries.get("expanded_keywords", [])
                self.logger.debug(f"  - Using optimized_queries_raw (no direct access), keys: {list(optimized_queries.keys())}")
            else:
                optimized_queries = {}
                extracted_keywords = []
                self.logger.debug(f"  - optimized_queries not found anywhere")

            # search_params_raw도 확인
            if search_params_raw is not None and len(search_params_raw) > 0:
                search_params = search_params_raw
                self.logger.debug(f"  - Using search_params_raw, keys: {list(search_params.keys())}")
            elif search_params_raw is not None:
                # 빈 딕셔너리인 경우 - 직접 state["search"]에서 확인
                self.logger.debug(f"  - search_params_raw is empty dict, checking state['search'] directly")
                if "search" in state and isinstance(state["search"], dict):
                    direct_params = state["search"].get("search_params")
                    if direct_params and len(direct_params) > 0:
                        search_params = direct_params
                        self.logger.debug(f"  - Found in state['search'], keys: {list(search_params.keys())}")
                    else:
                        search_params = {}
                else:
                    search_params = {}
            else:
                search_params = {}

            # 검증: optimized_queries와 search_params가 None이 아니고, 필수 키가 있는지 확인
            semantic_query_value = optimized_queries.get("semantic_query", "") if optimized_queries else ""

            # semantic_query가 빈 문자열이면 기본 쿼리 사용
            if not semantic_query_value or not str(semantic_query_value).strip():
                query = self._get_state_value(state, "query", "")
                if query:
                    self.logger.warning(f"semantic_query is empty in execute_searches_parallel, using base query: '{query[:50]}...'")
                    optimized_queries["semantic_query"] = query
                    semantic_query_value = query

            has_semantic_query = optimized_queries and semantic_query_value and len(str(semantic_query_value).strip()) > 0
            keyword_queries_value = optimized_queries.get("keyword_queries", []) if optimized_queries else []

            # keyword_queries가 비어있으면 기본 쿼리 사용
            if not keyword_queries_value or len(keyword_queries_value) == 0:
                query = self._get_state_value(state, "query", "")
                if query:
                    self.logger.warning(f"keyword_queries is empty in execute_searches_parallel, using base query")
                    optimized_queries["keyword_queries"] = [query]
                    keyword_queries_value = [query]

            has_keyword_queries = optimized_queries and keyword_queries_value and len(keyword_queries_value) > 0

            self.logger.debug(f"  - Validation: semantic_query='{semantic_query_value[:50] if semantic_query_value else 'EMPTY'}...', has_semantic_query={has_semantic_query}")
            self.logger.debug(f"  - Validation: keyword_queries={len(keyword_queries_value) if keyword_queries_value else 0}, has_keyword_queries={has_keyword_queries}")
            self.logger.debug(f"  - Validation: search_params is None={search_params is None}, is empty={search_params == {}}, keys={list(search_params.keys()) if search_params else []}")

            if optimized_queries_raw is None or search_params_raw is None or not has_semantic_query:
                self.logger.warning("Optimized queries or search params not found")
                self.logger.debug(f"PARALLEL SEARCH SKIP: optimized_queries={optimized_queries is not None}, search_params={search_params is not None}")
                self._set_state_value(state, "semantic_results", [])
                self._set_state_value(state, "keyword_results", [])
                self._set_state_value(state, "semantic_count", 0)
                self._set_state_value(state, "keyword_count", 0)
                return state

            semantic_results = []
            semantic_count = 0
            keyword_results = []
            keyword_count = 0

            # 원본 query 가져오기 (추가 검색용)
            original_query = self._get_state_value(state, "query", "")
            if not original_query:
                # input 그룹에서도 확인
                if "input" in state and isinstance(state.get("input"), dict):
                    original_query = state["input"].get("query", "")

            self.logger.debug(f"PARALLEL SEARCH START: semantic_query={optimized_queries.get('semantic_query', 'N/A')[:50]}, keyword_queries={len(optimized_queries.get('keyword_queries', []))}, original_query={original_query[:50] if original_query else 'N/A'}...")

            with ThreadPoolExecutor(max_workers=2) as executor:
                # 의미적 검색 작업 제출 (원본 query 포함)
                semantic_future = executor.submit(
                    self._execute_semantic_search_internal,
                    optimized_queries,
                    search_params,
                    original_query  # 원본 query 추가
                )

                # 키워드 검색 작업 제출 (원본 query 포함)
                keyword_future = executor.submit(
                    self._execute_keyword_search_internal,
                    optimized_queries,
                    search_params,
                    query_type_str,
                    legal_field,
                    extracted_keywords,
                    original_query  # 원본 query 추가
                )

                # 두 작업이 완료될 때까지 대기
                try:
                    semantic_results, semantic_count = semantic_future.result(timeout=30)
                    self.logger.debug(f"Semantic future completed: {semantic_count} results")
                except Exception as e:
                    self.logger.error(f"Semantic search failed: {e}")
                    self.logger.debug(f"Semantic search exception: {e}")
                    semantic_results, semantic_count = [], 0

                try:
                    keyword_results, keyword_count = keyword_future.result(timeout=30)
                    self.logger.debug(f"Keyword future completed: {keyword_count} results")
                except Exception as e:
                    self.logger.error(f"Keyword search failed: {e}")
                    self.logger.debug(f"Keyword search exception: {e}")
                    keyword_results, keyword_count = [], 0

            # 결과 저장
            # 중요: search 그룹이 확실히 존재하도록 ensure_state_group 호출
            from .state_helpers import ensure_state_group
            ensure_state_group(state, "search")

            self.logger.debug(f"PARALLEL SEARCH: Before save - semantic_results={len(semantic_results)}, keyword_results={len(keyword_results)}")

            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)

            # 저장 확인 로그
            stored_semantic = self._get_state_value(state, "semantic_results", [])
            stored_keyword = self._get_state_value(state, "keyword_results", [])
            self.logger.debug(f"PARALLEL SEARCH: After save - semantic_results={len(stored_semantic)}, keyword_results={len(stored_keyword)}")

            # state["search"]에서 직접 확인 (디버깅)
            if "search" in state and isinstance(state.get("search"), dict):
                direct_semantic = state["search"].get("semantic_results", [])
                direct_keyword = state["search"].get("keyword_results", [])
                self.logger.debug(f"PARALLEL SEARCH: Direct state['search'] check - semantic={len(direct_semantic)}, keyword={len(direct_keyword)}")
            else:
                self.logger.debug(f"PARALLEL SEARCH: state['search'] not found or not dict, state keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

            self._save_metadata_safely(state, "_last_executed_node", "execute_searches_parallel")
            self._update_processing_time(state, start_time)

            elapsed_time = time.time() - start_time

            # 상세 디버깅 로그
            self.logger.info(
                f"✅ [PARALLEL SEARCH] Completed in {elapsed_time:.3f}s - "
                f"Semantic: {semantic_count} results, Keyword: {keyword_count} results"
            )
            self.logger.debug(f"PARALLEL SEARCH: Semantic={semantic_count}, Keyword={keyword_count}")

            # 검색 결과 상세 정보 로깅
            if semantic_results:
                semantic_scores = [doc.get("relevance_score", 0.0) for doc in semantic_results[:5]]
                self.logger.info(
                    f"🔍 [DEBUG] Semantic search details: "
                    f"Top scores: {semantic_scores}, "
                    f"Sample sources: {[doc.get('source', 'Unknown')[:30] for doc in semantic_results[:3]]}"
                )
            else:
                self.logger.warning("⚠️ [DEBUG] Semantic search returned 0 results")

            if keyword_results:
                keyword_scores = [doc.get("relevance_score", doc.get("score", 0.0)) for doc in keyword_results[:5]]
                self.logger.info(
                    f"🔍 [DEBUG] Keyword search details: "
                    f"Top scores: {keyword_scores}, "
                    f"Sample sources: {[doc.get('source', 'Unknown')[:30] for doc in keyword_results[:3]]}"
                )
            else:
                self.logger.warning("⚠️ [DEBUG] Keyword search returned 0 results")

        except Exception as e:
            self._handle_error(state, str(e), "병렬 검색 중 오류 발생")
            # 폴백: 순차 실행
            return self._fallback_sequential_search(state)

        # 반환 전에 search 그룹 확인 및 로깅
        if "search" in state and isinstance(state.get("search"), dict):
            final_search = state["search"]
            final_semantic = len(final_search.get("semantic_results", []))
            final_keyword = len(final_search.get("keyword_results", []))
            print(f"[DEBUG] execute_searches_parallel: Returning state with search group - semantic_results={final_semantic}, keyword_results={final_keyword}")
            print(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")
        else:
            print(f"[DEBUG] execute_searches_parallel: WARNING - Returning state WITHOUT search group!")
            print(f"[DEBUG] execute_searches_parallel: Returning state keys={list(state.keys()) if isinstance(state, dict) else 'N/A'}")

        return state

    def _execute_semantic_search_internal(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        original_query: str = ""
    ) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 검색 실행 (내부 헬퍼)"""
        semantic_results = []
        semantic_count = 0

        semantic_query = optimized_queries.get("semantic_query", "")
        semantic_k = search_params.get("semantic_k", WorkflowConstants.SEMANTIC_SEARCH_K)

        self.logger.info(
            f"🔍 [DEBUG] Executing semantic search: query='{semantic_query[:50]}...', k={semantic_k}, original_query='{original_query[:50] if original_query else 'N/A'}...'"
        )

        # 메인 쿼리로 의미적 검색
        main_semantic, main_count = self._semantic_search(
            semantic_query,
            k=semantic_k
        )
        semantic_results.extend(main_semantic)
        semantic_count += main_count

        self.logger.info(
            f"🔍 [DEBUG] Main semantic search: {main_count} results (query: '{semantic_query[:50]}...')"
        )

        # 원본 query로도 의미적 검색 수행 (항상 포함)
        # 중요: 원본 query는 사용자의 직접적인 의도이므로 semantic_query와 같아도 별도로 검색
        if original_query and original_query.strip():
            original_semantic, original_count = self._semantic_search(
                original_query,
                k=semantic_k // 2
            )
            semantic_results.extend(original_semantic)
            semantic_count += original_count
            self.logger.info(
                f"🔍 [DEBUG] Original query semantic search: {original_count} results (query: '{original_query[:50]}...')"
            )
            print(f"[DEBUG] _execute_semantic_search_internal: Added {original_count} results from original query search")

        # 키워드 쿼리로 추가 의미적 검색
        keyword_queries = optimized_queries.get("keyword_queries", [])[:2]
        for i, kw_query in enumerate(keyword_queries, 1):
            # semantic_query와는 다르지만, original_query와는 중복 허용 가능
            # (키워드 쿼리가 원본 query를 포함할 수 있으므로)
            if kw_query and kw_query.strip() and kw_query != semantic_query:
                kw_semantic, kw_count = self._semantic_search(
                    kw_query,
                    k=semantic_k // 2
                )
                semantic_results.extend(kw_semantic)
                semantic_count += kw_count
                self.logger.info(
                    f"🔍 [DEBUG] Keyword-based semantic search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                )
                print(f"[DEBUG] _execute_semantic_search_internal: Added {kw_count} results from keyword query #{i}")

        self.logger.info(
            f"🔍 [DEBUG] Total semantic search results: {semantic_count} (unique: {len(semantic_results)})"
        )
        print(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Total={semantic_count}, Unique={len(semantic_results)}")

        # 검색 쿼리 구성 요약 로그
        search_queries_used = []
        if semantic_query:
            search_queries_used.append(f"semantic_query({len(semantic_query)} chars)")
        if original_query:
            search_queries_used.append(f"original_query({len(original_query)} chars)")
        keyword_queries_used = optimized_queries.get("keyword_queries", [])[:2]
        if keyword_queries_used:
            search_queries_used.append(f"keyword_queries({len(keyword_queries_used)} queries)")
        print(f"[DEBUG] SEMANTIC SEARCH INTERNAL: Queries used: {', '.join(search_queries_used)}")

        return semantic_results, semantic_count

    def _execute_keyword_search_internal(
        self,
        optimized_queries: Dict[str, Any],
        search_params: Dict[str, Any],
        query_type_str: str,
        legal_field: str,
        extracted_keywords: List[str],
        original_query: str = ""
    ) -> Tuple[List[Dict[str, Any]], int]:
        """키워드 검색 실행 (내부 헬퍼)"""
        keyword_results = []
        keyword_count = 0

        keyword_queries = optimized_queries.get("keyword_queries", [])
        keyword_limit = search_params.get("keyword_limit", WorkflowConstants.CATEGORY_SEARCH_LIMIT)

        self.logger.info(
            f"🔍 [DEBUG] Executing keyword search: {len(keyword_queries)} queries, "
            f"limit={keyword_limit}, field={legal_field}, "
            f"keywords={extracted_keywords[:5] if extracted_keywords else []}, "
            f"original_query='{original_query[:50] if original_query else 'N/A'}...'"
        )

        # 원본 query로도 키워드 검색 수행 (비어있지 않은 경우)
        if original_query and original_query.strip():
            original_kw_results, original_kw_count = self._keyword_search(
                query=original_query,
                query_type_str=query_type_str,
                limit=keyword_limit,
                legal_field=legal_field,
                extracted_keywords=extracted_keywords
            )
            keyword_results.extend(original_kw_results)
            keyword_count += original_kw_count
            self.logger.info(
                f"🔍 [DEBUG] Original query keyword search: {original_kw_count} results (query: '{original_query[:50]}...')"
            )

        # 최적화된 키워드 쿼리로 검색
        for i, kw_query in enumerate(keyword_queries, 1):
            # 원본 query와 중복되지 않는 경우에만 검색
            if kw_query and kw_query.strip() and kw_query != original_query:
                kw_results, kw_count = self._keyword_search(
                    query=kw_query,
                    query_type_str=query_type_str,
                    limit=keyword_limit,
                    legal_field=legal_field,
                    extracted_keywords=extracted_keywords
                )
                keyword_results.extend(kw_results)
                keyword_count += kw_count
                self.logger.info(
                    f"🔍 [DEBUG] Keyword search #{i}: {kw_count} results (query: '{kw_query[:50]}...')"
                )

        self.logger.info(
            f"🔍 [DEBUG] Total keyword search results: {keyword_count} (unique: {len(keyword_results)})"
        )
        print(f"[DEBUG] KEYWORD SEARCH INTERNAL: Total={keyword_count}, Unique={len(keyword_results)}")

        return keyword_results, keyword_count

    def _fallback_sequential_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """순차 검색 실행 (폴백)"""
        try:
            self.logger.warning("Falling back to sequential search")

            optimized_queries = self._get_state_value(state, "optimized_queries", {})
            search_params = self._get_state_value(state, "search_params", {})
            query_type_str = self._get_query_type_str(self._get_state_value(state, "query_type", ""))
            legal_field = self._get_state_value(state, "legal_field", "")
            extracted_keywords = optimized_queries.get("expanded_keywords", [])

            # 원본 query 가져오기
            original_query = self._get_state_value(state, "query", "")
            if not original_query and "input" in state and isinstance(state.get("input"), dict):
                original_query = state["input"].get("query", "")

            # 의미적 검색 (순차)
            semantic_results, semantic_count = self._execute_semantic_search_internal(
                optimized_queries, search_params, original_query
            )

            # 키워드 검색 (순차)
            keyword_results, keyword_count = self._execute_keyword_search_internal(
                optimized_queries, search_params, query_type_str, legal_field, extracted_keywords, original_query
            )

            self._set_state_value(state, "semantic_results", semantic_results)
            self._set_state_value(state, "keyword_results", keyword_results)
            self._set_state_value(state, "semantic_count", semantic_count)
            self._set_state_value(state, "keyword_count", keyword_count)

            self.logger.info(f"Sequential search completed: {semantic_count} semantic, {keyword_count} keyword")

        except Exception as e:
            self._handle_error(state, str(e), "순차 검색 중 오류 발생")

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
            quality_evaluation = {
                "semantic_quality": semantic_quality,
                "keyword_quality": keyword_quality,
                "overall_quality": (semantic_quality["score"] + keyword_quality["score"]) / 2.0,
                "needs_retry": semantic_quality["needs_retry"] or keyword_quality["needs_retry"]
            }

            self._set_state_value(state, "search_quality_evaluation", quality_evaluation)
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
        """의미적 검색 품질 평가"""
        quality = {
            "score": 0.0,
            "result_count": len(semantic_results),
            "avg_relevance": 0.0,
            "diversity_ratio": 0.0,
            "query_match": 0.0,
            "needs_retry": False,
            "issues": []
        }

        if not semantic_results:
            quality["needs_retry"] = True
            quality["issues"].append("검색 결과가 없음")
            return quality

        # 결과 수 평가
        result_count_score = min(1.0, len(semantic_results) / max(1, min_results))
        quality["result_count"] = len(semantic_results)

        # 평균 관련성 점수 계산
        relevance_scores = [
            doc.get("relevance_score", doc.get("score", 0.0))
            for doc in semantic_results
            if doc.get("relevance_score") or doc.get("score")
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        quality["avg_relevance"] = avg_relevance

        # 다양성 평가 (중복 제거 비율)
        seen_contents = set()
        unique_count = 0
        for doc in semantic_results:
            content_preview = doc.get("content", "")[:100]
            if content_preview and content_preview not in seen_contents:
                seen_contents.add(content_preview)
                unique_count += 1

        diversity_ratio = unique_count / len(semantic_results) if semantic_results else 0.0
        quality["diversity_ratio"] = diversity_ratio

        # 쿼리 일치도 평가
        query_words = set(query.lower().split())
        match_count = 0
        for doc in semantic_results:
            doc_content = doc.get("content", "").lower()
            doc_words = set(doc_content.split())
            if query_words and doc_words:
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    match_count += 1

        query_match = match_count / len(semantic_results) if semantic_results else 0.0
        quality["query_match"] = query_match

        # 종합 품질 점수 계산
        quality_score = (
            result_count_score * 0.25 +
            avg_relevance * 0.30 +
            diversity_ratio * 0.20 +
            query_match * 0.25
        )
        quality["score"] = quality_score

        # 재검색 필요 여부 판단
        needs_retry = (
            result_count_score < 0.5 or
            avg_relevance < 0.6 or
            query_match < 0.3
        )
        quality["needs_retry"] = needs_retry

        if needs_retry:
            if result_count_score < 0.5:
                quality["issues"].append(f"결과 수 부족: {len(semantic_results)}개")
            if avg_relevance < 0.6:
                quality["issues"].append(f"평균 관련성 점수 낮음: {avg_relevance:.2f}")
            if query_match < 0.3:
                quality["issues"].append(f"쿼리 일치도 낮음: {query_match:.2f}")

        return quality

    def _evaluate_keyword_search_quality(
        self,
        keyword_results: List[Dict[str, Any]],
        query: str,
        query_type: str,
        min_results: int = 3
    ) -> Dict[str, Any]:
        """키워드 검색 품질 평가"""
        quality = {
            "score": 0.0,
            "result_count": len(keyword_results),
            "avg_relevance": 0.0,
            "category_match": 0.0,
            "legal_citation_ratio": 0.0,
            "needs_retry": False,
            "issues": []
        }

        if not keyword_results:
            quality["needs_retry"] = True
            quality["issues"].append("검색 결과가 없음")
            return quality

        # 결과 수 평가
        result_count_score = min(1.0, len(keyword_results) / max(1, min_results))
        quality["result_count"] = len(keyword_results)

        # 평균 관련성 점수 계산
        relevance_scores = [
            doc.get("relevance_score", doc.get("score", 0.0))
            for doc in keyword_results
            if doc.get("relevance_score") or doc.get("score")
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        quality["avg_relevance"] = avg_relevance

        # 카테고리 매칭도 평가
        category_match_count = sum(1 for doc in keyword_results if doc.get("category_boost", 1.0) > 1.0)
        category_match = category_match_count / len(keyword_results) if keyword_results else 0.0
        quality["category_match"] = category_match

        # 법률 조항 포함도 평가
        legal_citation_count = 0
        for doc in keyword_results:
            content = doc.get("content", "")
            # 법률 조항 패턴 확인
            if re.search(r'[가-힣]+법\s*제?\s*\d+\s*조', content):
                legal_citation_count += 1

        legal_citation_ratio = legal_citation_count / len(keyword_results) if keyword_results else 0.0
        quality["legal_citation_ratio"] = legal_citation_ratio

        # 종합 품질 점수 계산
        quality_score = (
            result_count_score * 0.25 +
            avg_relevance * 0.25 +
            category_match * 0.25 +
            legal_citation_ratio * 0.25
        )
        quality["score"] = quality_score

        # 재검색 필요 여부 판단
        needs_retry = (
            result_count_score < 0.5 or
            avg_relevance < 0.6 or
            legal_citation_ratio < 0.2
        )
        quality["needs_retry"] = needs_retry

        if needs_retry:
            if result_count_score < 0.5:
                quality["issues"].append(f"결과 수 부족: {len(keyword_results)}개")
            if avg_relevance < 0.6:
                quality["issues"].append(f"평균 관련성 점수 낮음: {avg_relevance:.2f}")
            if legal_citation_ratio < 0.2:
                quality["issues"].append(f"법률 조항 포함도 낮음: {legal_citation_ratio:.2f}")

        return quality

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
                retry_semantic, retry_count = self._semantic_search(
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
        """키워드별 중요도 가중치 계산"""
        keyword_weights = {}

        if not extracted_keywords:
            return keyword_weights

        query_lower = query.lower()

        # 법률 용어 패턴
        legal_term_patterns = [
            r'[가-힣]+법', r'[가-힣]+규정', r'[가-힣]+조항',
            r'판례', r'대법원', r'법원', r'판결',
            r'계약', r'손해배상', r'소송', r'청구'
        ]

        # 질문 유형별 중요 키워드
        query_type_keywords = {
            "precedent_search": ["판례", "사건", "판결", "대법원"],
            "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
            "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
            "procedure_guide": ["절차", "방법", "대응", "소송"],
            "term_explanation": ["의미", "정의", "개념", "해석"]
        }

        # 법률 분야별 관련 키워드
        field_keywords = {
            "family": ["가족", "이혼", "양육", "상속", "부부"],
            "civil": ["민사", "계약", "손해배상", "채권", "채무"],
            "criminal": ["형사", "범죄", "처벌", "형량"],
            "labor": ["노동", "근로", "해고", "임금", "근로자"],
            "corporate": ["기업", "회사", "주주", "법인"]
        }

        important_keywords_for_type = query_type_keywords.get(query_type, [])
        important_keywords_for_field = field_keywords.get(legal_field, [])

        for keyword in extracted_keywords:
            if not keyword or not isinstance(keyword, str):
                continue

            keyword_lower = keyword.lower()
            weight = 0.0

            # 1. 쿼리 출현 빈도 (30%)
            query_frequency = query_lower.count(keyword_lower)
            query_weight = min(0.3, (query_frequency / max(1, len(query.split()))) * 0.3)
            weight += query_weight

            # 2. 법률 용어 여부 (30%)
            is_legal_term = any(
                re.search(pattern, keyword, re.IGNORECASE)
                for pattern in legal_term_patterns
            )
            if is_legal_term:
                weight += 0.3

            # 3. 질문 유형별 중요도 (20%)
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_type):
                weight += 0.2

            # 4. 법률 분야별 관련성 (20%)
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_field):
                weight += 0.2

            # 기본 가중치 (최소값)
            if weight == 0.0:
                weight = 0.1

            # 정규화 (0.0-1.0)
            keyword_weights[keyword] = min(1.0, weight)

        # 가중치 정규화
        total_weight = sum(keyword_weights.values())
        if total_weight > 0:
            max_weight = max(keyword_weights.values()) if keyword_weights else 1.0
            if max_weight > 0:
                for kw in keyword_weights:
                    keyword_weights[kw] = keyword_weights[kw] / max_weight

        return keyword_weights

    def _calculate_keyword_match_score(
        self,
        document: Dict[str, Any],
        keyword_weights: Dict[str, float],
        query: str
    ) -> Dict[str, float]:
        """문서에 대한 키워드 매칭 점수 계산"""
        doc_content = document.get("content", "")
        if not doc_content:
            return {
                "keyword_match_score": 0.0,
                "keyword_coverage": 0.0,
                "matched_keywords": [],
                "weighted_keyword_score": 0.0
            }

        doc_content_lower = doc_content.lower()

        matched_keywords = []
        total_weight = 0.0
        matched_weight = 0.0

        for keyword, weight in keyword_weights.items():
            if not keyword:
                continue

            total_weight += weight
            keyword_lower = keyword.lower()

            if keyword_lower in doc_content_lower:
                matched_keywords.append(keyword)
                matched_weight += weight

                # 추가 보너스: 키워드가 여러 번 출현
                keyword_count = doc_content_lower.count(keyword_lower)
                if keyword_count > 1:
                    matched_weight += weight * 0.1 * min(2, keyword_count - 1)

        keyword_coverage = len(matched_keywords) / max(1, len(keyword_weights))
        keyword_match_score = matched_weight / max(0.1, total_weight) if total_weight > 0 else 0.0
        weighted_keyword_score = min(1.0, matched_weight / max(1, len(keyword_weights)))

        return {
            "keyword_match_score": keyword_match_score,
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "weighted_keyword_score": weighted_keyword_score
        }

    def _calculate_weighted_final_score(
        self,
        document: Dict[str, Any],
        keyword_scores: Dict[str, float],
        search_params: Dict[str, Any],
        query_type: Optional[str] = None
    ) -> float:
        """가중치를 적용한 최종 점수 계산 (강화된 버전)"""
        base_relevance = (
            document.get("relevance_score", 0.0) or
            document.get("combined_score", 0.0) or
            document.get("score", 0.0)
        )

        keyword_match = keyword_scores.get("weighted_keyword_score", 0.0)

        search_type = document.get("search_type", "")
        type_weight = 1.2 if search_type == "semantic" else 1.0

        # 문서 타입별 가중치 (강화)
        doc_type = document.get("type", "").lower() if document.get("type") else ""
        doc_type_weight = 1.0
        if "법령" in doc_type or "law" in doc_type:
            doc_type_weight = 1.2  # 법령 우선
        elif "판례" in doc_type or "precedent" in doc_type:
            doc_type_weight = 1.1  # 판례 차순
        else:
            doc_type_weight = 0.9

        # 질문 유형별 가중치 (강화)
        query_type_weight = 1.0
        if query_type:
            if query_type == "precedent_search" and ("판례" in doc_type or "precedent" in doc_type):
                query_type_weight = 1.3  # 판례 검색에서 판례 문서
            elif query_type == "law_inquiry" and ("법령" in doc_type or "law" in doc_type):
                query_type_weight = 1.3  # 법령 문의에서 법령 문서

        category_boost = document.get("category_boost", 1.0)
        field_match_score = document.get("field_match_score", 0.5)
        category_bonus = (category_boost * 0.7 + field_match_score * 0.3)

        # 최종 점수 계산 (강화된 가중치 적용)
        final_score = (
            base_relevance * 0.50 +  # 기본 점수 비중 증가 (40% → 50%)
            keyword_match * 0.30 +  # 키워드 점수 비중 감소 (35% → 30%)
            (base_relevance * doc_type_weight * query_type_weight) * 0.10 +  # 문서/질문 타입 가중치 추가
            type_weight * 0.05 +  # 검색 타입 가중치 감소 (15% → 5%)
            category_bonus * 0.05  # 카테고리 보너스 감소 (10% → 5%)
        )

        return min(1.0, max(0.0, final_score))

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

    def _rerank_with_keyword_weights(
        self,
        results: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        rerank_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """키워드 가중치를 적용한 Reranking"""
        try:
            # 최종 가중치 점수로 정렬
            sorted_results = sorted(
                results,
                key=lambda x: (
                    x.get("final_weighted_score", 0.0),
                    x.get("keyword_match_score", 0.0),
                    x.get("keyword_coverage", 0.0)
                ),
                reverse=True
            )

            # 키워드 커버리지 보너스 적용
            for doc in sorted_results:
                coverage = doc.get("keyword_coverage", 0.0)
                if coverage > 0.7:
                    doc["final_weighted_score"] *= 1.1
                elif coverage > 0.5:
                    doc["final_weighted_score"] *= 1.05

            # 다시 정렬
            sorted_results = sorted(
                sorted_results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )

            # Reranker 사용 (있는 경우)
            top_k = rerank_params.get("top_k", 20)
            if self.result_ranker and len(sorted_results) > 0:
                try:
                    reranked_results = self.result_ranker.rank_results(
                        sorted_results[:top_k * 2],
                        top_k=top_k
                    )
                    # ResultRanker 결과를 Dict 형태로 변환
                    if reranked_results and hasattr(reranked_results[0], 'score'):
                        reranked_dicts = []
                        for result in reranked_results:
                            doc = {
                                "content": result.text,
                                "relevance_score": result.score,
                                "source": result.source,
                                "id": f"{result.source}_{hash(result.text)}",
                                "final_weighted_score": result.score
                            }
                            if isinstance(result.metadata, dict):
                                doc.update(result.metadata)
                            reranked_dicts.append(doc)
                        sorted_results = reranked_dicts[:top_k]
                    else:
                        sorted_results = sorted_results[:top_k]
                except Exception as e:
                    self.logger.warning(f"Reranker failed, using keyword-weighted scores: {e}")
                    sorted_results = sorted_results[:top_k]
            else:
                sorted_results = sorted_results[:top_k]

            # 다양성 필터 적용
            try:
                if self.result_ranker and hasattr(self.result_ranker, 'apply_diversity_filter'):
                    diverse_results = self.result_ranker.apply_diversity_filter(
                        sorted_results,
                        max_per_type=5,
                        diversity_weight=rerank_params.get("diversity_weight", 0.3)
                    )
                else:
                    diverse_results = sorted_results
            except Exception as e:
                self.logger.warning(f"Diversity filter failed: {e}")
                diverse_results = sorted_results

            return diverse_results

        except Exception as e:
            self.logger.warning(f"Reranking with keyword weights failed: {e}")
            return sorted(
                results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )[:rerank_params.get("top_k", 20)]

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
            documents = self._apply_metadata_filters(
                documents,
                query_type_str,
                legal_field
            )

            # 결과 품질 검증 및 필터링
            filtered_docs = self._filter_low_quality_results(
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
            self._update_search_metadata(
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

            # retrieved_docs 검증
            has_valid_docs = False
            if not retrieved_docs:
                self.logger.warning(
                    f"⚠️ [PREPARE CONTEXT] No retrieved_docs to prepare for prompt. "
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
        """프롬프트에 최대한 반영되도록 최적화된 컨텍스트 구축"""
        try:
            # 입력 검증
            if not retrieved_docs:
                self.logger.warning("_build_prompt_optimized_context: retrieved_docs is empty")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }

            # 문서 검증: content 필드와 관련도 점수 기준 필터링 (검색 타입 고려)
            valid_docs = []
            invalid_docs_count = 0
            # 검색 타입별 다른 관련도 기준 적용
            # 키워드 검색 결과는 관련도가 낮아도 키워드 매칭이 있으면 포함
            min_relevance_score_semantic = 0.3  # 의미적 검색: 0.3 이상
            min_relevance_score_keyword = 0.15  # 키워드 검색: 0.15 이상 (키워드 매칭 시)

            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    invalid_docs_count += 1
                    continue

                # content 필드 확인 (여러 가능한 필드명 지원)
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if not content or len(content.strip()) < 10:  # 최소 10자 이상
                    invalid_docs_count += 1
                    self.logger.debug(f"Document filtered: content too short or empty (source: {doc.get('source', 'Unknown')})")
                    continue

                # 관련도 점수 확인 (검색 타입별 기준 적용)
                search_type = doc.get("search_type", "semantic")  # 기본값은 semantic
                relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                has_keyword_match = keyword_match_score > 0.0 or len(doc.get("matched_keywords", [])) > 0

                # 검색 타입별 필터링 기준
                min_score = min_relevance_score_keyword if search_type == "keyword" else min_relevance_score_semantic

                # 키워드 검색 결과는 키워드 매칭이 있으면 관련도 기준 완화
                if search_type == "keyword" and has_keyword_match:
                    min_score = min_relevance_score_keyword  # 0.15 이상
                elif search_type == "semantic":
                    min_score = min_relevance_score_semantic  # 0.3 이상

                if relevance_score < min_score:
                    invalid_docs_count += 1
                    self.logger.debug(
                        f"Document filtered: relevance score too low ({relevance_score:.3f} < {min_score}) "
                        f"(source: {doc.get('source', 'Unknown')}, type: {search_type})"
                    )
                    continue

                valid_docs.append(doc)

            if invalid_docs_count > 0:
                self.logger.warning(
                    f"_build_prompt_optimized_context: Filtered {invalid_docs_count} invalid documents "
                    f"(no content, content too short, or relevance < 0.3). Valid docs: {len(valid_docs)}"
                )

            # 유효한 문서가 없으면 에러 반환
            if not valid_docs:
                self.logger.error("_build_prompt_optimized_context: No valid documents with content found")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }

            # 최종 가중치 점수로 정렬 (이미 reranked되어 있을 수 있음)
            sorted_docs = sorted(
                valid_docs,
                key=lambda x: (
                    x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                    x.get("keyword_match_score", 0.0)
                ),
                reverse=True
            )

            # 의미적 검색과 키워드 검색 결과의 균형을 맞춰서 선택
            balanced_docs = self._select_balanced_documents(sorted_docs, max_docs=10)

            # 최소 1개 문서 보장
            if not balanced_docs and sorted_docs:
                balanced_docs = sorted_docs[:min(8, len(sorted_docs))]

            sorted_docs = balanced_docs

            # 최소 1개 문서 보장
            if not sorted_docs:
                self.logger.error("_build_prompt_optimized_context: sorted_docs is empty after filtering")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }

            # 문서 기반 답변 지시사항 생성
            document_instructions = self._generate_document_based_instructions(
                documents=sorted_docs,
                query=query,
                query_type=query_type
            )

            # 검색 결과 통계 계산
            semantic_count = sum(1 for doc in sorted_docs if doc.get("search_type") == "semantic")
            keyword_count = sum(1 for doc in sorted_docs if doc.get("search_type") == "keyword")
            hybrid_count = len(sorted_docs) - semantic_count - keyword_count

            # 프롬프트 섹션 구축 (검색 결과 통계 포함)
            prompt_section = f"""## 답변 생성 지시사항

{document_instructions}

## 참고 문서 목록

다음 {len(sorted_docs)}개의 문서를 반드시 참고하여 답변을 생성하세요.
각 문서는 관련성 점수와 핵심 내용이 표시되어 있습니다.

**검색 결과 통계:**
- 의미적 검색 결과: {semantic_count}개
- 키워드 검색 결과: {keyword_count}개
- 하이브리드 검색 결과: {hybrid_count}개
- 총 문서 수: {len(sorted_docs)}개

**참고:** 의미적 검색 결과는 의미적 유사도를, 키워드 검색 결과는 키워드 매칭 정도를 나타냅니다.
두 검색 방식의 결과를 종합하여 정확하고 포괄적인 답변을 생성하세요.

"""

            # 우선순위 높은 문서부터 구조화하여 제공
            for idx, doc in enumerate(sorted_docs, 1):
                relevance_score = doc.get("final_weighted_score") or doc.get("relevance_score", 0.0)
                source = doc.get("source", "Unknown")
                content = doc.get("content", "")

                # 질문과 직접 관련된 문장 추출
                relevant_sentences = self._extract_query_relevant_sentences(
                    doc_content=content,
                    query=query,
                    extracted_keywords=extracted_keywords
                )

                # 검색 메타데이터 가져오기
                search_type = doc.get("search_type", "hybrid")
                search_method = doc.get("search_method", "hybrid_search")
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                matched_keywords = doc.get("matched_keywords", [])

                # 문서 섹션 구성 (검색 메타데이터 포함)
                doc_section = f"""
### 문서 {idx}: {source} (관련성 점수: {relevance_score:.2f})

**검색 정보:**
- 검색 방식: {search_type} ({search_method})
- 키워드 매칭 점수: {keyword_match_score:.2f}
- 매칭된 키워드: {', '.join(matched_keywords[:5]) if matched_keywords else '없음'}

**핵심 내용:**
"""

                # 관련 문장 강조 표시
                if relevant_sentences:
                    doc_section += "\n".join([
                        f"- [중요] {sent['sentence']}"
                        for sent in relevant_sentences[:3]
                    ])
                    doc_section += "\n\n"

                # 전체 문서 내용 (적절한 길이로 제한)
                max_content_length = 1500
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "..."

                doc_section += f"""**전체 내용:**
{content}

---
"""

                prompt_section += doc_section

            # 문서 인용 형식 지정
            prompt_section += """
## 문서 인용 규칙

답변에서 위 문서를 인용할 때는 다음과 같이 명시하세요:
- "문서 {0}에 따르면..." 또는 "[{0}] 인용 내용"
- 각 문서의 출처를 명확히 표시

## 중요 사항

- 위 문서의 내용을 바탕으로 답변을 생성하세요
- 문서에서 추론하거나 추측하지 말고, 문서에 명시된 내용만 사용하세요
- 문서에 없는 정보는 포함하지 마세요
- 여러 문서의 내용을 종합하여 일관된 답변을 구성하세요
""".format("n")

            # 프롬프트 섹션 검증: 실제 문서 내용이 포함되었는지 확인
            content_validation = {
                "has_document_content": False,
                "total_content_length": 0,
                "documents_with_content": 0
            }

            # 프롬프트에 실제 문서 내용이 포함되었는지 확인
            for doc in sorted_docs:
                content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                if content and len(content.strip()) >= 10:
                    # 프롬프트 섹션에 문서 내용이 포함되어 있는지 확인
                    content_preview = content[:100]  # 처음 100자만 확인
                    if content_preview in prompt_section:
                        content_validation["has_document_content"] = True
                        content_validation["total_content_length"] += len(content)
                        content_validation["documents_with_content"] += 1

            # 검증 결과 로깅
            if not content_validation["has_document_content"]:
                self.logger.error(
                    f"_build_prompt_optimized_context: WARNING - prompt_section does not contain actual document content! "
                    f"Documents processed: {len(sorted_docs)}, "
                    f"Prompt length: {len(prompt_section)}"
                )
            else:
                self.logger.info(
                    f"_build_prompt_optimized_context: Successfully included content from {content_validation['documents_with_content']} documents "
                    f"(total content length: {content_validation['total_content_length']} chars, "
                    f"prompt length: {len(prompt_section)} chars)"
                )

            # 최소 검증: 프롬프트에 문서 내용이 실제로 포함되어야 함
            if not content_validation["has_document_content"] and len(sorted_docs) > 0:
                # 문서가 있는데 내용이 없으면 에러 (하지만 빈 결과는 반환하지 않고 경고만)
                self.logger.warning(
                    f"_build_prompt_optimized_context: Content validation failed, but returning prompt anyway "
                    f"(may contain instructions only without actual document content)"
                )

            return {
                "prompt_optimized_text": prompt_section,
                "structured_documents": {
                    "total_count": len(sorted_docs),
                    "documents": [{
                        "document_id": idx,
                        "source": doc.get("source", "Unknown"),
                        "relevance_score": doc.get("final_weighted_score") or doc.get("relevance_score", 0.0),
                        "content": (doc.get("content") or doc.get("text") or doc.get("content_text", ""))[:2000]
                    } for idx, doc in enumerate(sorted_docs, 1)]
                },
                "document_count": len(sorted_docs),
                "total_context_length": len(prompt_section),
                "content_validation": content_validation  # 검증 메타데이터 추가
            }

        except Exception as e:
            self.logger.error(f"Prompt optimized context building failed: {e}")
            return {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            }

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
        """문서를 기반으로 답변 생성하라는 명시적 지시사항 생성"""
        instructions = f"""당신은 법률 전문가입니다. 아래 제공된 문서들을 반드시 참고하여 다음 질문에 답변하세요.

**질문**: {query}
**질문 유형**: {query_type}

**답변 생성 규칙**:
1. **문서 기반 답변**: 제공된 문서의 내용을 바탕으로 답변을 생성하세요.
2. **문서 인용 필수**: 답변에서 문서를 인용할 때는 "문서 [번호]에 따르면..." 형식으로 명시하세요.
3. **정확성**: 문서에 명시된 내용만 사용하고, 추론하거나 추측하지 마세요.
4. **구조화**: 답변은 다음 구조로 작성하세요:
   - 핵심 답변
   - 관련 법령 및 조항
   - 실무 적용 시 주의사항
   - 참고할 만한 판례 (있는 경우)
5. **출처 명시**: 각 인용문에 대해 문서 번호를 명시하세요.
"""

        return instructions

    def _select_balanced_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        max_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """
        의미적 검색과 키워드 검색 결과의 균형을 맞춰서 문서 선택

        Args:
            sorted_docs: 점수로 정렬된 문서 리스트
            max_docs: 선택할 최대 문서 수

        Returns:
            균형잡힌 문서 리스트
        """
        if not sorted_docs:
            return []

        # 검색 타입별로 분류
        semantic_docs = [doc for doc in sorted_docs if doc.get("search_type") == "semantic"]
        keyword_docs = [doc for doc in sorted_docs if doc.get("search_type") == "keyword"]
        hybrid_docs = [doc for doc in sorted_docs if doc.get("search_type") not in ["semantic", "keyword"]]

        # 균형잡힌 선택 전략
        selected_docs = []

        # 1. 상위 문서는 무조건 포함 (최대 절반)
        top_count = max(1, max_docs // 2)
        selected_docs.extend(sorted_docs[:top_count])

        # 2. 의미적 검색과 키워드 검색 결과를 균형있게 포함
        remaining_slots = max_docs - len(selected_docs)

        if remaining_slots > 0:
            # 의미적 검색 결과에서 선택
            semantic_to_add = []
            for doc in semantic_docs:
                if doc not in selected_docs:
                    semantic_to_add.append(doc)

            # 키워드 검색 결과에서 선택
            keyword_to_add = []
            for doc in keyword_docs:
                if doc not in selected_docs:
                    keyword_to_add.append(doc)

            # 교대로 추가하여 균형 유지
            max_alternate = remaining_slots // 2
            for i in range(min(max_alternate, max(len(semantic_to_add), len(keyword_to_add)))):
                if i < len(semantic_to_add) and len(selected_docs) < max_docs:
                    if semantic_to_add[i] not in selected_docs:
                        selected_docs.append(semantic_to_add[i])
                if i < len(keyword_to_add) and len(selected_docs) < max_docs:
                    if keyword_to_add[i] not in selected_docs:
                        selected_docs.append(keyword_to_add[i])

            # 남은 슬롯이 있으면 hybrid 문서 추가
            if len(selected_docs) < max_docs:
                for doc in hybrid_docs:
                    if doc not in selected_docs and len(selected_docs) < max_docs:
                        selected_docs.append(doc)

            # 아직 슬롯이 남으면 점수 순으로 추가
            if len(selected_docs) < max_docs:
                for doc in sorted_docs:
                    if doc not in selected_docs and len(selected_docs) < max_docs:
                        selected_docs.append(doc)

        # 원래 점수 순서 유지
        selected_docs = sorted(
            selected_docs,
            key=lambda x: (
                x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                x.get("keyword_match_score", 0.0)
            ),
            reverse=True
        )

        return selected_docs[:max_docs]

    def _extract_legal_references_from_docs(self, documents: List[Dict[str, Any]]) -> List[str]:
        """문서에서 법률 참조 정보 추출"""
        return DocumentExtractor.extract_legal_references_from_docs(documents)
