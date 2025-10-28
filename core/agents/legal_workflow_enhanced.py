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
from typing import Any, Dict, List, Tuple

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

from core.agents.keyword_mapper import LegalKeywordMapper
from core.agents.legal_data_connector import LegalDataConnector
from core.agents.performance_optimizer import PerformanceOptimizer
from core.agents.state_definitions import LegalWorkflowState
from core.agents.state_utils import (
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)
from core.services.search.question_classifier import QuestionType
from core.services.search.result_merger import ResultMerger, ResultRanker
from core.services.search.semantic_search_engine import SemanticSearchEngine
from infrastructure.utils.langgraph_config import LangGraphConfig
from source.services.term_integration_system import TermIntegrator
from source.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)

# Logger 초기화
logger = logging.getLogger(__name__)

# AnswerStructureEnhancer 통합 (답변 구조화 및 법적 근거 강화)
try:
    from source.services.answer_structure_enhancer import AnswerStructureEnhancer
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = True
except ImportError:
    ANSWER_STRUCTURE_ENHANCER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("AnswerStructureEnhancer not available")


class WorkflowConstants:
    """워크플로우 상수 정의"""

    # LLM 설정
    MAX_OUTPUT_TOKENS = 200
    TEMPERATURE = 0.3
    TIMEOUT = 15

    # 검색 설정
    SEMANTIC_SEARCH_K = 10
    MAX_DOCUMENTS = 5
    CATEGORY_SEARCH_LIMIT = 3

    # 재시도 설정
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    # 신뢰도 설정
    LLM_CLASSIFICATION_CONFIDENCE = 0.85
    FALLBACK_CONFIDENCE = 0.7
    DEFAULT_CONFIDENCE = 0.6


class EnhancedLegalQuestionWorkflow:
    """개선된 법률 질문 처리 워크플로우"""

    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 통합 프롬프트 관리자 초기화 (우선)
        self.unified_prompt_manager = UnifiedPromptManager()

        # 컴포넌트 초기화
        self.keyword_mapper = LegalKeywordMapper()
        self.data_connector = LegalDataConnector()
        self.performance_optimizer = PerformanceOptimizer()
        self.term_integrator = TermIntegrator()
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()


        # AnswerStructureEnhancer 초기화 (답변 구조화 및 법적 근거 강화)
        if ANSWER_STRUCTURE_ENHANCER_AVAILABLE:
            self.answer_structure_enhancer = AnswerStructureEnhancer()
            self.logger.info("AnswerStructureEnhancer initialized for answer quality enhancement")
        else:
            self.answer_structure_enhancer = None
            self.logger.warning("AnswerStructureEnhancer not available")

        # AnswerFormatter 초기화 (시각적 포맷팅)
        try:
            from core.services.generation.answer_formatter import AnswerFormatter
            self.answer_formatter = AnswerFormatter()
            self.logger.info("AnswerFormatter initialized for visual formatting")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AnswerFormatter: {e}")
            self.answer_formatter = None

        # Semantic Search Engine 초기화 (벡터 검색을 위한)
        try:
            self.semantic_search = SemanticSearchEngine()
            self.logger.info("SemanticSearchEngine initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SemanticSearchEngine: {e}")
            self.semantic_search = None

        # MultiTurnQuestionHandler 초기화 (멀티턴 질문 처리)
        try:
            from source.services.conversation_manager import ConversationManager
            from source.services.multi_turn_handler import MultiTurnQuestionHandler
            self.multi_turn_handler = MultiTurnQuestionHandler()
            self.conversation_manager = ConversationManager()
            self.logger.info("MultiTurnQuestionHandler initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MultiTurnQuestionHandler: {e}")
            self.multi_turn_handler = None
            self.conversation_manager = None

        # AIKeywordGenerator 초기화 (AI 키워드 확장)
        try:
            from source.services.ai_keyword_generator import AIKeywordGenerator
            self.ai_keyword_generator = AIKeywordGenerator()
            self.logger.info("AIKeywordGenerator initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize AIKeywordGenerator: {e}")
            self.ai_keyword_generator = None

        # EmotionIntentAnalyzer 초기화 (긴급도 평가용)
        try:
            from source.services.emotion_intent_analyzer import EmotionIntentAnalyzer
            self.emotion_analyzer = EmotionIntentAnalyzer()
            self.logger.info("EmotionIntentAnalyzer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize EmotionIntentAnalyzer: {e}")
            self.emotion_analyzer = None

        # LegalBasisValidator 초기화 (법령 검증용)
        try:
            from source.services.legal_basis_validator import LegalBasisValidator
            self.legal_validator = LegalBasisValidator()
            self.logger.info("LegalBasisValidator initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalBasisValidator: {e}")
            self.legal_validator = None

        # DocumentProcessor 초기화 (문서 분석용)
        try:
            from infrastructure.utils.config import Config as UtilsConfig
            from source.services.document_processor import LegalDocumentProcessor
            utils_config = UtilsConfig()
            self.document_processor = LegalDocumentProcessor(utils_config)
            self.logger.info("LegalDocumentProcessor initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LegalDocumentProcessor: {e}")
            self.document_processor = None

        # LLM 초기화
        self.llm = self._initialize_llm()

        # 통계 관리 (config에서 활성화 여부 확인)
        self.stats = {
            'total_queries': 0,
            'total_documents_retrieved': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'total_errors': 0
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
        gemini_llm.invoke("test")
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
        ollama_llm.invoke("test")
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

    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구축 - 전체 노드 포함"""
        workflow = StateGraph(LegalWorkflowState)

        # 노드 추가
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("assess_urgency", self.assess_urgency)  # 새로 추가
        workflow.add_node("resolve_multi_turn", self.resolve_multi_turn)
        workflow.add_node("route_expert", self.route_expert)  # 새로 추가
        workflow.add_node("analyze_document", self.analyze_document)  # 새로 추가
        workflow.add_node("expand_keywords_ai", self.expand_keywords_ai)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("process_legal_terms", self.process_legal_terms)
        workflow.add_node("generate_answer_enhanced", self.generate_answer_enhanced)
        workflow.add_node("validate_answer_quality", self.validate_answer_quality)
        workflow.add_node("enhance_answer_structure", self.enhance_answer_structure)
        workflow.add_node("apply_visual_formatting", self.apply_visual_formatting)
        workflow.add_node("prepare_final_response", self.prepare_final_response)


        # Entry point
        workflow.set_entry_point("classify_query")

        # 플로우 정의
        workflow.add_edge("classify_query", "assess_urgency")
        workflow.add_edge("assess_urgency", "resolve_multi_turn")
        workflow.add_edge("resolve_multi_turn", "route_expert")

        # 조건부: 전문가 라우팅 후 문서 분석 여부
        workflow.add_conditional_edges(
            "route_expert",
            self._should_analyze_document,
            {
                "analyze": "analyze_document",
                "skip": "expand_keywords_ai"
            }
        )

        workflow.add_edge("analyze_document", "expand_keywords_ai")

        # 키워드 확장 파이프라인
        workflow.add_conditional_edges(
            "expand_keywords_ai",
            self._should_expand_keywords_ai,
            {
                "expand": "retrieve_documents",
                "skip": "retrieve_documents"
            }
        )

        # Linear flow (검색을 수행하는 경우)
        workflow.add_edge("retrieve_documents", "process_legal_terms")
        workflow.add_edge("process_legal_terms", "generate_answer_enhanced")

        # 조건부 엣지 2: 답변 품질 검증 후 재시도 여부 결정
        workflow.add_conditional_edges(
            "generate_answer_enhanced",
            self._should_retry_or_continue,
            {
                "validate": "validate_answer_quality",
                "format": "enhance_answer_structure",  # 새로운 구조화 노드로
                "retry": "retrieve_documents"
            }
        )

        # 검증 후 구조화 또는 재시도
        workflow.add_conditional_edges(
            "validate_answer_quality",
            self._should_accept_answer,
            {
                "accept": "enhance_answer_structure",  # 새로운 구조화 노드로
                "retry": "retrieve_documents"
            }
        )

        # 새로운 포맷팅 파이프라인
        workflow.add_edge("enhance_answer_structure", "apply_visual_formatting")
        workflow.add_edge("apply_visual_formatting", "prepare_final_response")
        workflow.add_edge("prepare_final_response", END)

        return workflow


    def _should_expand_keywords_ai(self, state: LegalWorkflowState) -> str:
        """AI 키워드 확장 여부 결정"""
        # AI 확장 조건:
        # 1. AI 키워드 생성기가 초기화되어 있는가
        # 2. 추출된 키워드가 충분히 있는가
        # 3. 질문 복잡도가 충분히 높은가

        if not self.ai_keyword_generator:
            return "skip"

        keywords = state.get("extracted_keywords", [])
        if len(keywords) < 3:
            return "skip"

        # 복잡한 질문인 경우 확장
        query_type = state.get("query_type", "")
        complex_types = ["precedent_search", "law_inquiry", "legal_advice"]

        if query_type in complex_types:
            return "expand"

        return "skip"

    def _should_retry_or_continue(self, state: LegalWorkflowState) -> str:
        """재시도 여부 결정 - 답변 품질 체크"""
        answer = state.get("answer", "")
        errors = state.get("errors", [])
        retry_count = state.get("retry_count", 0)

        # 심각한 에러가 있고 재시도 횟수가 남아있는 경우
        if len(errors) > 0 and retry_count < 2:
            state["retry_count"] = retry_count + 1
            state["needs_enhancement"] = True
            self.logger.warning(f"Retrying due to errors (attempt {retry_count + 1}/2)")
            return "retry"

        # 답변이 너무 짧은 경우
        if len(answer) < 50 and retry_count < 1:
            state["retry_count"] = retry_count + 1
            state["needs_enhancement"] = True
            self.logger.info(f"Retrying due to short answer (attempt {retry_count + 1}/1)")
            return "retry"

        # 기본적으로 검증 진행
        if len(answer) > 0:
            return "validate"

        return "format"

    def _should_accept_answer(self, state: LegalWorkflowState) -> str:
        """답변 수락 여부 결정"""
        quality_check_passed = state.get("quality_check_passed", False)
        retry_count = state.get("retry_count", 0)

        if quality_check_passed:
            return "accept"

        # 품질 검증 실패한 경우, 재시도 횟수에 따라 결정
        if retry_count < 2:
            state["retry_count"] = retry_count + 1
            return "retry"

        # 최대 재시도 횟수 초과 시 수락
        return "accept"

    @observe(name="validate_answer_quality")
    def validate_answer_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 품질 및 법령 검증"""
        try:
            start_time = time.time()
            answer = state.get("answer", "")
            errors = state.get("errors", [])
            sources = state.get("sources", [])

            # 기존 품질 검증
            quality_checks = {
                "has_answer": len(answer) > 0,
                "min_length": len(answer) >= 50,
                "no_errors": len(errors) == 0,
                "has_sources": len(sources) > 0 or len(state.get("retrieved_docs", [])) > 0
            }

            # 법령 검증 추가
            if self.legal_validator and len(answer) > 0:
                try:
                    validation_result = self.legal_validator.validate_legal_basis(
                        state["query"], answer
                    )
                    state["legal_validity_check"] = validation_result.is_valid
                    state["legal_basis_validation"] = {
                        "confidence": validation_result.confidence,
                        "issues": validation_result.issues,
                        "recommendations": validation_result.recommendations
                    }

                    # 품질 체크에 법령 검증 추가
                    quality_checks["legal_basis_valid"] = validation_result.is_valid

                    self.logger.info(f"Legal basis validation: {validation_result.is_valid}")
                except Exception as e:
                    self.logger.warning(f"Legal validation failed: {e}")
                    state["legal_validity_check"] = True  # 실패시 통과
                    quality_checks["legal_basis_valid"] = True
            else:
                state["legal_validity_check"] = True
                quality_checks["legal_basis_valid"] = True

            # 품질 점수 계산
            passed_checks = sum(quality_checks.values())
            total_checks = len(quality_checks)
            quality_score = passed_checks / total_checks

            # 품질 점수가 0.75 이상이면 통과
            quality_check_passed = quality_score >= 0.75
            state["quality_check_passed"] = quality_check_passed

            self._update_processing_time(state, start_time)

            quality_status = "통과" if quality_check_passed else "실패"
            self._add_step(state, "답변 검증",
                         f"품질: {quality_score:.2f}, 법령: {state['legal_validity_check']}")

            self.logger.info(
                f"Answer quality validation: {quality_status}, "
                f"score: {quality_score:.2f}, checks: {passed_checks}/{total_checks}"
            )

        except Exception as e:
            self._handle_error(state, str(e), "답변 검증 중 오류")
            state["quality_check_passed"] = False
            state["legal_validity_check"] = True

        return state

    # Helper methods for common operations
    def _update_processing_time(self, state: LegalWorkflowState, start_time: float):
        """처리 시간 업데이트"""
        processing_time = time.time() - start_time
        state["processing_time"] = state.get("processing_time", 0.0) + processing_time
        return processing_time

    def _add_step(self, state: LegalWorkflowState, step_prefix: str, step_message: str):
        """처리 단계 추가 (중복 방지 및 pruning)"""
        existing_steps = state.get("processing_steps", [])
        if not any(step_prefix in step for step in existing_steps):
            state["processing_steps"].append(step_message)

        # Always prune if too many steps (check on every add)
        if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
            state["processing_steps"] = prune_processing_steps(
                state["processing_steps"],
                max_items=MAX_PROCESSING_STEPS
            )

    def _handle_error(self, state: LegalWorkflowState, error_msg: str, context: str = ""):
        """에러 처리 헬퍼"""
        full_error = f"{context}: {error_msg}" if context else error_msg
        state["errors"].append(full_error)
        state["processing_steps"].append(full_error)
        self.logger.error(full_error)

    def _get_category_mapping(self) -> Dict[str, List[str]]:
        """카테고리 매핑 반환"""
        return {
            "precedent_search": ["family_law", "civil_law", "criminal_law"],
            "law_inquiry": ["family_law", "civil_law", "contract_review"],
            "legal_advice": ["family_law", "civil_law", "labor_law"],
            "procedure_guide": ["civil_procedure", "family_law", "labor_law"],
            "term_explanation": ["civil_law", "family_law", "contract_review"],
            "general_question": ["civil_law", "family_law", "contract_review"]
        }

    @observe(name="classify_query")
    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 (LLM 기반)"""
        try:
            start_time = time.time()

            classified_type, confidence = self._classify_with_llm(state["query"])

            # QuestionType enum을 문자열로 변환하여 저장
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            state["query_type"] = query_type_str
            state["confidence"] = confidence

            # 법률 분야 추출
            state["legal_field"] = self._extract_legal_field(query_type_str, state["query"])
            state["legal_domain"] = self._map_to_legal_domain(state["legal_field"])

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {query_type_str}, 법률분야: {state['legal_field']} (시간: {processing_time:.3f}s)")

            self.logger.info(f"LLM classified query as {query_type_str} with confidence {confidence}, field: {state['legal_field']}")

        except Exception as e:
            self._handle_error(state, str(e), "LLM 질문 분류 중 오류 발생")
            classified_type, confidence = self._fallback_classification(state["query"])
            query_type_str = classified_type.value if hasattr(classified_type, 'value') else str(classified_type)
            state["query_type"] = query_type_str
            state["confidence"] = confidence

            # 법률 분야 추출 (폴백)
            state["legal_field"] = self._extract_legal_field(query_type_str, state["query"])
            state["legal_domain"] = self._map_to_legal_domain(state["legal_field"])

            self._add_step(state, "폴백 키워드 기반 분류 사용", "폴백 키워드 기반 분류 사용")

        return state

    @observe(name="resolve_multi_turn")
    def resolve_multi_turn(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """멀티턴 질문 해결 노드"""
        try:
            start_time = time.time()

            # 멀티턴 핸들러와 세션 관리자가 없으면 스킵
            if not self.multi_turn_handler or not self.conversation_manager:
                state["is_multi_turn"] = False
                state["search_query"] = state["query"]
                self.logger.debug("Multi-turn handler not available, skipping multi-turn resolution")
                return state

            query = state["query"]
            session_id = state["session_id"]

            # 세션에서 대화 맥락 가져오기
            conversation_context = self._get_or_create_conversation_context(session_id)

            if conversation_context and conversation_context.turns:
                # 멀티턴 질문 감지
                is_multi_turn = self.multi_turn_handler.detect_multi_turn_question(query, conversation_context)
                state["is_multi_turn"] = is_multi_turn

                if is_multi_turn:
                    # 완전한 질문 구성
                    multi_turn_result = self.multi_turn_handler.build_complete_query(query, conversation_context)

                    resolved_query = multi_turn_result.get("resolved_query", query)
                    state["multi_turn_confidence"] = multi_turn_result.get("confidence", 1.0)

                    # 대화 맥락 정보 저장
                    state["conversation_context"] = self._build_conversation_context_dict(conversation_context)

                    # 검색 쿼리 업데이트 (해결된 쿼리 사용)
                    state["search_query"] = resolved_query

                    self.logger.info(f"Multi-turn question resolved: '{query}' -> '{resolved_query}'")
                    self._add_step(state, "멀티턴 처리",
                                 f"멀티턴 질문 해결: {multi_turn_result.get('reasoning', '')}")
                else:
                    # 멀티턴 질문이 아님
                    state["multi_turn_confidence"] = 1.0

                    # 단일 턴이므로 search_query는 그대로
                    state["search_query"] = query
            else:
                # 대화 맥락이 없음
                state["is_multi_turn"] = False
                state["multi_turn_confidence"] = 1.0
                state["search_query"] = query

            self._update_processing_time(state, start_time)

        except Exception as e:
            self.logger.error(f"Error in resolve_multi_turn: {e}")
            # 에러 발생 시 원본 쿼리 유지
            state["is_multi_turn"] = False
            state["search_query"] = state.get("search_query", state["query"])
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
        try:
            if not context:
                return None

            return {
                "session_id": context.session_id if hasattr(context, 'session_id') else "",
                "turn_count": len(context.turns) if hasattr(context, 'turns') else 0,
                "entities": {
                    entity_type: list(entity_set)
                    for entity_type, entity_set in (context.entities or {}).items()
                } if hasattr(context, 'entities') and context.entities else {},
                "topic_stack": list(context.topic_stack) if hasattr(context, 'topic_stack') and context.topic_stack else [],
                "recent_topics": list(context.topic_stack[-3:]) if hasattr(context, 'topic_stack') and context.topic_stack else []
            }
        except Exception as e:
            self.logger.error(f"Error building conversation context dict: {e}")
            return None

    def _classify_with_llm(self, query: str) -> Tuple[QuestionType, float]:
        """LLM 기반 분류"""
        classification_prompt = f"""다음 법률 질문을 질문 유형으로 분류해주세요.

질문: {query}

분류 가능한 유형:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

중요: 질문의 핵심 의도를 파악하여 가장 적합한 유형 하나만 선택하세요.
응답 형식: 유형명만 답변 (예: legal_advice)"""

        classification_response = self._call_llm_with_retry(classification_prompt, max_retries=2)
        classification_result = classification_response.strip().lower().replace(" ", "")

        question_type_mapping = {
            "precedent_search": QuestionType.PRECEDENT_SEARCH,
            "law_inquiry": QuestionType.LAW_INQUIRY,
            "legal_advice": QuestionType.LEGAL_ADVICE,
            "procedure_guide": QuestionType.PROCEDURE_GUIDE,
            "term_explanation": QuestionType.TERM_EXPLANATION,
            "general_question": QuestionType.GENERAL_QUESTION,
        }

        for key, question_type in question_type_mapping.items():
            if key in classification_result or key.replace("_", "") in classification_result:
                return question_type, WorkflowConstants.LLM_CLASSIFICATION_CONFIDENCE

        self.logger.warning(f"Could not classify query, LLM response: {classification_result}")
        return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE

    def _fallback_classification(self, query: str) -> Tuple[QuestionType, float]:
        """폴백 키워드 기반 분류"""
        self.logger.info("Using fallback keyword-based classification")
        query_lower = query.lower()

        if any(k in query_lower for k in ["판례", "사건", "판결"]):
            return QuestionType.PRECEDENT_SEARCH, WorkflowConstants.FALLBACK_CONFIDENCE
        elif any(k in query_lower for k in ["법률", "조문", "법령", "규정"]):
            return QuestionType.LAW_INQUIRY, WorkflowConstants.FALLBACK_CONFIDENCE
        elif any(k in query_lower for k in ["절차", "방법", "대응"]):
            return QuestionType.PROCEDURE_GUIDE, WorkflowConstants.FALLBACK_CONFIDENCE
        else:
            return QuestionType.GENERAL_QUESTION, WorkflowConstants.DEFAULT_CONFIDENCE

    @observe(name="retrieve_documents")
    def retrieve_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색 (하이브리드: 벡터 + 키워드 검색)"""
        try:
            start_time = time.time()
            # 강화된 검색 쿼리 사용 (키워드 추출 노드에서 생성됨)
            search_query = state.get("search_query", state["query"])
            query_type_str = self._get_query_type_str(state["query_type"])

            # 캐시 확인
            if self._check_cache(state, search_query, query_type_str, start_time):
                return state

            # 하이브리드 검색 (강화된 쿼리 사용)
            semantic_results, semantic_count = self._semantic_search(search_query)
            keyword_results, keyword_count = self._keyword_search(search_query, query_type_str)

            # 결과 통합
            documents = self._merge_search_results(semantic_results, keyword_results)

            # Apply pruning and summarization to reduce state size
            state["retrieved_docs"] = prune_retrieved_docs(
                documents[:WorkflowConstants.MAX_DOCUMENTS],
                max_items=MAX_RETRIEVED_DOCS,
                max_content_per_doc=MAX_DOCUMENT_CONTENT_LENGTH
            )

            # 메타데이터 및 상태 업데이트
            self._update_search_metadata(state, semantic_count, keyword_count, documents, query_type_str, start_time)
            self.performance_optimizer.cache.cache_documents(search_query, query_type_str, state["retrieved_docs"])
            self._update_processing_time(state, start_time)

            self.logger.info(f"Hybrid search completed: {len(state['retrieved_docs'])} documents retrieved")
        except Exception as e:
            self._handle_error(state, str(e), "문서 검색 중 오류 발생")
            self._fallback_search(state)
        return state

    # Helper methods for retrieve_documents
    def _get_query_type_str(self, query_type) -> str:
        """QueryType을 문자열로 변환"""
        return query_type.value if hasattr(query_type, 'value') else str(query_type)

    def _check_cache(self, state: LegalWorkflowState, query: str, query_type_str: str, start_time: float) -> bool:
        """캐시에서 문서 확인"""
        cached_documents = self.performance_optimizer.cache.get_cached_documents(query, query_type_str)
        if cached_documents:
            state["retrieved_docs"] = cached_documents
            self._add_step(state, "문서 검색 완료", f"문서 검색 완료: {len(cached_documents)}개 (캐시)")
            self.logger.info(f"Using cached documents for query: {query[:50]}...")
            self._update_processing_time(state, start_time)
            return True
        return False

    def _semantic_search(self, query: str) -> Tuple[List[Dict[str, Any]], int]:
        """의미적 벡터 검색"""
        if not self.semantic_search:
            self.logger.info("Semantic search not available")
            return [], 0

        try:
            results = self.semantic_search.search(query, k=WorkflowConstants.SEMANTIC_SEARCH_K)
            self.logger.info(f"Semantic search found {len(results)} results")

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': f"semantic_{result.get('metadata', {}).get('id', hash(result.get('text', '')))}",
                    'content': result.get('text', ''),
                    'source': result.get('source', 'Vector Search'),
                    'relevance_score': result.get('score', 0.8),
                    'type': result.get('type', 'unknown'),
                    'metadata': result.get('metadata', {}),
                    'search_type': 'semantic'
                })
            return formatted_results, len(results)
        except Exception as e:
            self.logger.warning(f"Semantic search failed: {e}")
            return [], 0

    def _keyword_search(self, query: str, query_type_str: str) -> Tuple[List[Dict[str, Any]], int]:
        """키워드 기반 검색"""
        try:
            category_mapping = self._get_category_mapping()
            categories_to_search = category_mapping.get(query_type_str, ["civil_law"])

            keyword_results = []
            for category in categories_to_search:
                category_docs = self.data_connector.search_documents(
                    query, category, limit=WorkflowConstants.CATEGORY_SEARCH_LIMIT
                )
                for doc in category_docs:
                    doc['search_type'] = 'keyword'
                    doc['category'] = category
                keyword_results.extend(category_docs)
                self.logger.info(f"Found {len(category_docs)} documents in category: {category}")

            return keyword_results, len(keyword_results)
        except Exception as e:
            self.logger.warning(f"Keyword search failed: {e}")
            return [], 0

    def _merge_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """검색 결과 통합 및 중복 제거 (Rerank 로직 적용 + 유사도 필터링)"""
        try:
            # Step 0: 유사도 임계값 필터링
            similarity_threshold = self.config.similarity_threshold
            filtered_semantic = [
                doc for doc in semantic_results
                if doc.get('relevance_score', doc.get('similarity', 0.0)) >= similarity_threshold
            ]
            filtered_keyword = [
                doc for doc in keyword_results
                if doc.get('relevance_score', doc.get('similarity', 0.0)) >= similarity_threshold
            ]

            if len(filtered_semantic) < len(semantic_results) or len(filtered_keyword) < len(keyword_results):
                self.logger.info(f"Similarity filtering: {len(semantic_results)} → {len(filtered_semantic)}, {len(keyword_results)} → {len(filtered_keyword)}")

            # Step 1: 결과를 ResultMerger가 처리할 수 있는 형태로 변환
            exact_results = {"semantic": filtered_semantic}

            # Step 2: 결과 병합 (가중치 적용)
            merged = self.result_merger.merge_results(
                exact_results=exact_results,
                semantic_results=filtered_keyword,
                weights={"exact": 0.6, "semantic": 0.4}
            )

            # Step 3: 순위 결정
            ranked = self.result_ranker.rank_results(merged, top_k=20)

            # Step 4: 다양성 필터 적용
            filtered = self.result_ranker.apply_diversity_filter(ranked, max_per_type=5)

            # Step 5: MergedResult를 Dict 형태로 변환
            documents = []
            for result in filtered:
                doc = {
                    "content": result.text,
                    "relevance_score": result.score,
                    "source": result.source,
                    "id": f"{result.source}_{hash(result.text)}",
                    "type": "merged"
                }
                # metadata를 기존 Dict 형태로 병합
                if isinstance(result.metadata, dict):
                    doc.update(result.metadata)

                documents.append(doc)

            self.logger.info(f"Rerank applied: {len(semantic_results)} semantic + {len(keyword_results)} keyword → {len(documents)} final")
            return documents

        except Exception as e:
            self.logger.warning(f"Rerank failed, using simple merge: {e}")
            # 폴백: 간단한 병합 및 정렬
            seen_ids = set()
            documents = []

            for doc in semantic_results:
                doc_id = doc.get('id')
                if doc_id and doc_id not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc_id)

            for doc in keyword_results:
                doc_id = doc.get('id')
                if doc_id and doc_id not in seen_ids:
                    documents.append(doc)
                    seen_ids.add(doc_id)

            documents.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

            # 폴백에도 유사도 필터링 적용
            similarity_threshold = self.config.similarity_threshold
            documents = [
                doc for doc in documents
                if doc.get('relevance_score', 0.0) >= similarity_threshold
            ]
            return documents

    def _update_search_metadata(self, state: LegalWorkflowState, semantic_count: int,
                                keyword_count: int, documents: List[Dict], query_type_str: str, start_time: float):
        """검색 메타데이터 업데이트 (consolidated metadata)"""
        # Merge into consolidated metadata dict
        if "metadata" not in state or not isinstance(state["metadata"], dict):
            state["metadata"] = {}

        state["metadata"]["search"] = {
            "semantic_results_count": semantic_count,
            "keyword_results_count": keyword_count,
            "total_candidates": len(documents),
            "final_count": len(state["retrieved_docs"]),
            "search_time": time.time() - start_time,
            "query_type": query_type_str,
            "search_mode": "hybrid"
        }

        self._add_step(state, "하이브리드 검색 완료",
                      f"하이브리드 검색 완료: 의미적 {semantic_count}개, 키워드 {keyword_count}개, 최종 {len(state['retrieved_docs'])}개")

    def _fallback_search(self, state: LegalWorkflowState):
        """폴백 검색"""
        try:
            query_type_str = self._get_query_type_str(state.get("query_type"))
            category_mapping = self._get_category_mapping()
            fallback_categories = category_mapping.get(query_type_str, ["civil_law"])

            fallback_docs = []
            for category in fallback_categories:
                category_docs = self.data_connector.get_document_by_category(category, limit=2)
                fallback_docs.extend(category_docs)
                if len(fallback_docs) >= 3:
                    break

            if fallback_docs:
                state["retrieved_docs"] = fallback_docs
                self._add_step(state, "폴백", f"폴백: {len(fallback_docs)}개 문서 사용")
                self.logger.info(f"Using fallback documents: {len(fallback_docs)} docs")
            else:
                state["retrieved_docs"] = [
                    {"content": f"'{state['query']}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
                ]
                self.logger.warning("No fallback documents available")
        except Exception as fallback_error:
            self.logger.error(f"Fallback also failed: {fallback_error}")
            state["retrieved_docs"] = [
                {"content": f"'{state['query']}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
            ]


    @observe(name="expand_keywords_ai")
    def expand_keywords_ai(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """AI 키워드 확장 (AIKeywordGenerator 사용)"""
        try:
            start_time = time.time()

            if not self.ai_keyword_generator:
                self.logger.debug("AIKeywordGenerator not available, skipping")
                return state

            # 기존 키워드가 없으면 기본 키워드 생성
            keywords = state.get("extracted_keywords", [])
            if len(keywords) == 0:
                query = state.get("query", "")
                query_type_str = self._get_query_type_str(state.get("query_type", "general_question"))
                # LegalKeywordMapper를 사용하여 기본 키워드 추출
                keywords = self.keyword_mapper.get_keywords_for_question(query, query_type_str)
                keywords = list(set(keywords))  # 중복 제거
                state["extracted_keywords"] = keywords
                self.logger.info(f"Generated base keywords: {len(keywords)} keywords from query")

            query_type_str = self._get_query_type_str(state["query_type"])
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
                all_keywords = list(set(all_keywords))
                state["extracted_keywords"] = all_keywords

                # 메타데이터 저장 (새 딕셔너리 생성)
                state["ai_keyword_expansion"] = {
                    "domain": expansion_result.domain,
                    "original_keywords": expansion_result.base_keywords,
                    "expanded_keywords": expansion_result.expanded_keywords,
                    "confidence": expansion_result.confidence,
                    "method": expansion_result.expansion_method
                }

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
                all_keywords = list(set(all_keywords))
                state["extracted_keywords"] = all_keywords

                # 메타데이터 저장 (새 딕셔너리 생성)
                state["ai_keyword_expansion"] = {
                    "domain": domain,
                    "original_keywords": keywords,
                    "expanded_keywords": fallback_keywords,
                    "confidence": 0.5,
                    "method": "fallback"
                }

                processing_time = self._update_processing_time(state, start_time)
                self._add_step(state, "AI 키워드 확장 (폴백)",
                             f"AI 키워드 확장 (폴백): +{len(fallback_keywords)}개 ({processing_time:.3f}s)")
                self.logger.info(f"Fallback keyword expansion: {len(fallback_keywords)} keywords added in {processing_time:.3f}s")

        except Exception as e:
            self._handle_error(state, str(e), "AI 키워드 확장 중 오류 발생")

        return state

    def _get_domain_from_query_type(self, query_type: str) -> str:
        """질문 유형에서 도메인 추출"""
        domain_mapping = {
            "precedent_search": "민사법",
            "law_inquiry": "민사법",
            "legal_advice": "민사법",
            "procedure_guide": "민사소송법",
            "term_explanation": "기타/일반",
            "general_question": "기타/일반"
        }
        return domain_mapping.get(query_type, "기타/일반")

    @observe(name="process_legal_terms")
    def process_legal_terms(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """법률 용어 추출 및 통합 (문서 검색 후, 답변 생성 전)"""
        try:
            start_time = time.time()

            all_terms = self._extract_terms_from_documents(state["retrieved_docs"])
            self.logger.info(f"추출된 용어 수: {len(all_terms)}")

            if all_terms:
                representative_terms = self._integrate_and_process_terms(all_terms)
                state["metadata"]["extracted_terms"] = representative_terms
                state["metadata"]["total_terms_extracted"] = len(all_terms)
                state["metadata"]["unique_terms"] = len(representative_terms)
                self._add_step(state, "용어 통합 완료", f"용어 통합 완료: {len(representative_terms)}개")
                self.logger.info(f"통합된 용어 수: {len(representative_terms)}")
            else:
                state["metadata"]["extracted_terms"] = []
                self._add_step(state, "용어 추출 없음", "용어 추출 없음 (문서 내용 부족)")

            self._update_processing_time(state, start_time)
        except Exception as e:
            self._handle_error(state, str(e), "법률 용어 처리 중 오류 발생")
            state["metadata"]["extracted_terms"] = []
        return state

    def _extract_terms_from_documents(self, docs: List[Dict]) -> List[str]:
        """문서에서 법률 용어 추출"""
        all_terms = []
        for doc in docs:
            content = doc.get("content", "")
            korean_terms = re.findall(r'[가-힣0-9A-Za-z]+', content)
            legal_terms = [term for term in korean_terms
                          if len(term) >= 2 and any('\uac00' <= c <= '\ud7af' for c in term)]
            all_terms.extend(legal_terms)
        return all_terms

    def _integrate_and_process_terms(self, all_terms: List[str]) -> List[str]:
        """용어 통합 및 처리"""
        processed_terms = self.term_integrator.integrate_terms(all_terms)
        return [term["representative_term"] for term in processed_terms]

    @observe(name="generate_answer_enhanced")
    def generate_answer_enhanced(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """개선된 답변 생성 - UnifiedPromptManager 활용"""
        try:
            start_time = time.time()

            question_type, domain = self._get_question_type_and_domain(state["query_type"])
            model_type = ModelType.GEMINI if self.config.llm_provider == "google" else ModelType.OLLAMA

            context_dict = self._build_context(state)
            optimized_prompt = self.unified_prompt_manager.get_optimized_prompt(
                query=state["query"],
                question_type=question_type,
                domain=domain,
                context=context_dict,
                model_type=model_type,
                base_prompt_type="korean_legal_expert"
            )

            response = self._call_llm_with_retry(optimized_prompt)
            state["answer"] = response

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "답변 생성 완료", "답변 생성 완료")

            self.logger.info(f"Enhanced answer generated with UnifiedPromptManager in {processing_time:.2f}s")
        except Exception as e:
            self._handle_error(state, str(e), "개선된 답변 생성 중 오류 발생")
            state["answer"] = self._generate_fallback_answer(state)
        return state

    def _get_question_type_and_domain(self, query_type) -> Tuple[QuestionType, str]:
        """질문 유형과 도메인 매핑"""
        query_type_mapping = {
            "contract_review": (QuestionType.LEGAL_ADVICE, "민사법"),
            "family_law": (QuestionType.LEGAL_ADVICE, "가족법"),
            "criminal_law": (QuestionType.LEGAL_ADVICE, "형사법"),
            "civil_law": (QuestionType.LEGAL_ADVICE, "민사법"),
            "labor_law": (QuestionType.LEGAL_ADVICE, "노동법"),
            "property_law": (QuestionType.LEGAL_ADVICE, "부동산법"),
            "intellectual_property": (QuestionType.LEGAL_ADVICE, "지적재산권법"),
            "tax_law": (QuestionType.LEGAL_ADVICE, "세법"),
            "civil_procedure": (QuestionType.PROCEDURE_GUIDE, "민사소송법"),
            "general_question": (QuestionType.GENERAL_QUESTION, "일반")
        }
        return query_type_mapping.get(query_type, (QuestionType.GENERAL_QUESTION, "일반"))

    def _build_context(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """컨텍스트 구성 (길이 제한 관리)"""
        max_length = self.config.max_context_length
        context_parts = []
        current_length = 0
        docs_truncated = 0

        for doc in state["retrieved_docs"]:
            doc_content = doc.get("content", "")
            doc_length = len(doc_content)

            # 컨텍스트 길이 확인
            if current_length + doc_length > max_length:
                # 가능한 만큼만 추가
                remaining_length = max_length - current_length - 200  # 여유 공간
                if remaining_length > 100:  # 최소 100자
                    truncated_content = doc_content[:remaining_length] + "..."
                    context_parts.append(f"[문서: {doc.get('source', 'unknown')}]\n{truncated_content}")
                    docs_truncated += 1
                    self.logger.warning("Document truncated due to context length limit")
                break

            context_part = f"[문서: {doc.get('source', 'unknown')}]\n{doc_content}"
            context_parts.append(context_part)
            current_length += len(context_part)

        context_text = "\n\n".join(context_parts)

        if docs_truncated > 0:
            self.logger.info(f"Context length management: {current_length}/{max_length} chars, {docs_truncated} docs truncated")

        return {
            "context": context_text,
            "legal_references": state.get("legal_references", []),
            "query_type": state["query_type"],
            "context_length": current_length,
            "docs_truncated": docs_truncated
        }

    def _call_llm_with_retry(self, prompt: str, max_retries: int = WorkflowConstants.MAX_RETRIES) -> str:
        """LLM 호출 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                return self._extract_response_content(response)
            except Exception as e:
                self.logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(WorkflowConstants.RETRY_DELAY)

        return "LLM 호출에 실패했습니다."

    def _extract_response_content(self, response) -> str:
        """응답에서 내용 추출"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)


    def _generate_fallback_answer(self, state: LegalWorkflowState) -> str:
        """폴백 답변 생성"""
        query = state["query"]
        query_type = state["query_type"]
        context = "\n".join([doc["content"] for doc in state["retrieved_docs"]])

        return f"""## 답변

질문: {query}

이 질문은 {query_type} 영역에 해당합니다.

## 관련 법률 정보
{context}

## 주요 포인트
1. 위 정보를 바탕으로 구체적인 조치를 취하시기 바랍니다.
2. 정확한 법률적 조언을 위해서는 전문가와 상담하시는 것을 권장합니다.
3. 관련 법조문과 판례를 추가로 확인하시기 바랍니다.

## 주의사항
- 이 답변은 일반적인 정보 제공 목적이며, 구체적인 법률적 조언이 아닙니다.
- 실제 사안에 대해서는 전문 변호사와 상담하시기 바랍니다."""

    @observe(name="enhance_answer_structure")
    def enhance_answer_structure(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 구조화 및 법적 근거 강화"""
        try:
            start_time = time.time()

            answer = state.get("answer", "")
            query = state.get("query", "")
            query_type = state.get("query_type", "general_question")

            if self.answer_structure_enhancer:
                try:
                    enhanced_result = self.answer_structure_enhancer.enhance_answer_structure(
                        answer=answer,
                        question_type=query_type,
                        question=query,
                        domain="general"
                    )

                    if enhanced_result and "structured_answer" in enhanced_result:
                        # 구조화된 답변 저장
                        state["enhanced_answer"] = enhanced_result["structured_answer"]

                        # 품질 메트릭 저장
                        quality_metrics = enhanced_result.get("quality_metrics", {})
                        if quality_metrics:
                            state["quality_metrics"] = quality_metrics
                            state["structure_confidence"] = quality_metrics.get("overall_score", state.get("confidence", 0.0))

                        # 법적 근거 저장
                        if "legal_citations" in enhanced_result:
                            state["legal_citations"] = enhanced_result["legal_citations"]

                        self.logger.info("Answer structure enhanced successfully")
                except Exception as e:
                    self.logger.warning(f"AnswerStructureEnhancer failed: {e}")
                    state["enhanced_answer"] = answer
            else:
                state["enhanced_answer"] = answer

            self._update_processing_time(state, start_time)
            self._add_step(state, "구조화", "답변 구조화 완료")

        except Exception as e:
            self._handle_error(state, str(e), "답변 구조화 중 오류 발생")
            state["enhanced_answer"] = state.get("answer", "")

        return state

    @observe(name="apply_visual_formatting")
    def apply_visual_formatting(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """시각적 포맷팅 (이모지 + 섹션 구조)"""
        try:
            start_time = time.time()

            # AnswerFormatter 사용
            if self.answer_formatter:
                try:
                    # 답변 가져오기 (구조화된 것이 있으면 그것 사용)
                    raw_answer = state.get("enhanced_answer") or state.get("answer", "")
                    query_type = state.get("query_type", "general_question")
                    confidence = state.get("structure_confidence") or state.get("confidence", 0.0)

                    # QuestionType 매핑
                    from core.services.search.question_classifier import (
                        QuestionType as QType,
                    )
                    question_type_mapping = {
                        "precedent_search": QType.PRECEDENT_SEARCH,
                        "law_inquiry": QType.LAW_INQUIRY,
                        "legal_advice": QType.LEGAL_ADVICE,
                        "procedure_guide": QType.PROCEDURE_GUIDE,
                        "term_explanation": QType.TERM_EXPLANATION,
                        "general_question": QType.GENERAL_QUESTION,
                    }
                    q_type = question_type_mapping.get(query_type, QType.GENERAL_QUESTION)

                    # ConfidenceInfo 생성
                    from core.services.enhancement.confidence_calculator import (
                        ConfidenceInfo,
                    )
                    confidence_info = ConfidenceInfo(
                        confidence=confidence,
                        level=self._map_confidence_level(confidence),
                        factors={"answer_quality": confidence},
                        explanation=f"신뢰도: {confidence:.1%}"
                    )

                    # Sources 준비
                    sources = {
                        "law_results": [d for d in state.get("retrieved_docs", []) if "law" in d.get("type", "").lower() or "law" in d.get("source", "").lower()],
                        "precedent_results": [d for d in state.get("retrieved_docs", []) if "precedent" in d.get("type", "").lower() or "precedent" in d.get("source", "").lower()]
                    }

                    # AnswerFormatter 적용
                    formatted_result = self.answer_formatter.format_answer(
                        raw_answer=raw_answer,
                        question_type=q_type,
                        sources=sources,
                        confidence=confidence_info
                    )

                    if formatted_result:
                        state["answer"] = formatted_result.formatted_content
                        state["format_metadata"] = formatted_result.metadata
                        self.logger.info("Visual formatting applied successfully")
                    else:
                        state["answer"] = raw_answer

                except Exception as e:
                    self.logger.warning(f"AnswerFormatter failed: {e}")
                    state["answer"] = state.get("enhanced_answer") or state.get("answer", "")
            else:
                # AnswerFormatter가 없으면 구조화된 답변 사용
                state["answer"] = state.get("enhanced_answer") or state.get("answer", "")

            self._update_processing_time(state, start_time)
            self._add_step(state, "포맷팅", "시각적 포맷팅 완료")

        except Exception as e:
            self._handle_error(state, str(e), "시각적 포맷팅 중 오류 발생")
            state["answer"] = state.get("enhanced_answer") or state.get("answer", "")

        return state

    @observe(name="prepare_final_response")
    def prepare_final_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 응답 상태 준비"""
        try:
            start_time = time.time()

            # Final pruning before returning response
            # Prune processing steps
            if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
                state["processing_steps"] = prune_processing_steps(
                    state["processing_steps"],
                    max_items=MAX_PROCESSING_STEPS
                )

            # Prune errors if too many
            if len(state.get("errors", [])) > 10:
                state["errors"] = state["errors"][-10:]

            # 신뢰도 조정
            final_confidence = state.get("structure_confidence") or state.get("confidence", 0.0)

            # 키워드 포함도 기반 보정
            keyword_coverage = self._calculate_keyword_coverage(state, state.get("answer", ""))
            adjusted_confidence = min(0.9, max(final_confidence, final_confidence + (keyword_coverage * 0.2)))

            # 상태 업데이트
            state["confidence"] = adjusted_confidence
            state["sources"] = list(set([doc.get("source", "Unknown") for doc in state.get("retrieved_docs", [])]))

            # 법적 참조 정보 추가
            if "legal_references" not in state:
                state["legal_references"] = []

            # 메타데이터 설정
            self._set_metadata(state, state.get("answer", ""), keyword_coverage)

            self._update_processing_time(state, start_time)
            self._add_step(state, "최종 준비", "최종 응답 준비 완료")

            # Final pruning after adding last step
            if len(state.get("processing_steps", [])) > MAX_PROCESSING_STEPS:
                state["processing_steps"] = prune_processing_steps(
                    state["processing_steps"],
                    max_items=MAX_PROCESSING_STEPS
                )

            # 통계 업데이트
            self.update_statistics(state)

            self.logger.info(f"Final response prepared with confidence: {adjusted_confidence:.3f}")

        except Exception as e:
            self._handle_error(state, str(e), "최종 준비 중 오류 발생")
            self.update_statistics(state)

        return state

    def _map_confidence_level(self, confidence: float):
        """신뢰도 점수에 따른 레벨 매핑"""
        from core.services.enhancement.confidence_calculator import ConfidenceLevel

        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_keyword_coverage(self, state: LegalWorkflowState, answer: str) -> float:
        """키워드 포함도 계산"""
        try:
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                state["query"], state["query_type"]
            )
            return self.keyword_mapper.calculate_keyword_coverage(answer, required_keywords)
        except Exception as e:
            self.logger.warning(f"Keyword coverage calculation failed: {e}")
            return 0.0

    def _set_metadata(self, state: LegalWorkflowState, answer: str, keyword_coverage: float):
        """메타데이터 설정"""
        try:
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                state["query"], state["query_type"]
            )
            missing_keywords = self.keyword_mapper.get_missing_keywords(answer, required_keywords)

            state["metadata"] = {
                "keyword_coverage": keyword_coverage,
                "required_keywords_count": len(required_keywords),
                "matched_keywords_count": len(required_keywords) - len(missing_keywords),
                "response_length": len(answer),
                "query_type": state["query_type"]
            }
        except Exception as e:
            self.logger.warning(f"Metadata setting failed: {e}")
            state["metadata"] = {
                "response_length": len(answer),
                "query_type": state["query_type"]
            }

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
        # 키워드 매핑
        field_keywords = {
            "civil": ["민사", "계약", "손해배상", "재산", "계약서"],
            "criminal": ["형사", "범죄", "처벌", "형량", "범죄자"],
            "family": ["가족", "이혼", "양육", "상속", "부부"],
            "labor": ["노동", "근로", "해고", "임금", "근로자"],
            "corporate": ["기업", "회사", "주주", "법인", "회사법"],
            "tax": ["세금", "세무", "과세", "세법"],
            "intellectual_property": ["특허", "상표", "저작권", "지적재산"]
        }

        query_lower = query.lower()
        for field, keywords in field_keywords.items():
            if any(k in query_lower for k in keywords):
                return field

        # 질문 유형 기반 폴백
        type_to_field = {
            "precedent_search": "civil",
            "law_inquiry": "civil",
            "procedure_guide": "civil",
            "term_explanation": "civil",
            "legal_advice": "civil"
        }
        return type_to_field.get(query_type, "general")

    def _map_to_legal_domain(self, legal_field: str) -> str:
        """LegalDomain enum으로 매핑"""
        # LegalDomain enum 값 확인 후 매핑
        mapping = {
            "civil": LegalDomain.CIVIL_LAW.value if hasattr(LegalDomain.CIVIL_LAW, 'value') else "민사법",
            "criminal": LegalDomain.CRIMINAL_LAW.value if hasattr(LegalDomain.CRIMINAL_LAW, 'value') else "형사법",
            "family": LegalDomain.FAMILY_LAW.value if hasattr(LegalDomain.FAMILY_LAW, 'value') else "가족법",
            "labor": LegalDomain.LABOR_LAW.value if hasattr(LegalDomain.LABOR_LAW, 'value') else "노동법",
            "corporate": LegalDomain.COMMERCIAL_LAW.value if hasattr(LegalDomain.COMMERCIAL_LAW, 'value') else "상사법",
            "tax": LegalDomain.TAX_LAW.value if hasattr(LegalDomain.TAX_LAW, 'value') else "세법",
            "intellectual_property": LegalDomain.INTELLECTUAL_PROPERTY.value if hasattr(LegalDomain.INTELLECTUAL_PROPERTY, 'value') else "지적재산권법",
        }
        return mapping.get(legal_field, "일반")

    @observe(name="assess_urgency")
    def assess_urgency(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """긴급도 평가 노드"""
        try:
            start_time = time.time()
            query = state["query"]

            if self.emotion_analyzer:
                # 감정 및 의도 분석 (의도 분석만 사용)
                intent_result = self.emotion_analyzer.analyze_intent(query, None)

                # 긴급도 설정
                state["urgency_level"] = intent_result.urgency_level.value
                state["urgency_reasoning"] = intent_result.reasoning

                # 긴급 유형 판별
                if "기한" in query or "마감" in query or "데드라인" in query:
                    state["emergency_type"] = "legal_deadline"
                elif "소송" in query or "재판" in query or "법원" in query:
                    state["emergency_type"] = "case_progress"
                else:
                    state["emergency_type"] = None

                self.logger.info(f"Urgency assessed: {state['urgency_level']}")
            else:
                # 폴백: 키워드 기반 긴급도 평가
                state["urgency_level"] = self._assess_urgency_fallback(query)
                state["urgency_reasoning"] = "키워드 기반 평가"
                state["emergency_type"] = None

            self._update_processing_time(state, start_time)
            self._add_step(state, "긴급도 평가", f"긴급도: {state['urgency_level']}")

        except Exception as e:
            self._handle_error(state, str(e), "긴급도 평가 중 오류")
            state["urgency_level"] = "medium"
            state["urgency_reasoning"] = "기본값"
            state["emergency_type"] = None

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

            # 문서 유형 판별
            doc_type = self._detect_document_type(doc_text)
            state["document_type"] = doc_type

            # 문서 분석
            analysis_result = self._analyze_legal_document(doc_text, doc_type)

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
        clauses = []

        # 조항 패턴 매칭
        clause_patterns = {
            "payment": r"(대금|금액|지급|결제).*?조",
            "period": r"(기간|기한|만료).*?조",
            "termination": r"(해지|해제|종료).*?조",
            "liability": r"(책임|손해배상|위약).*?조",
            "confidentiality": r"(비밀|기밀|보안).*?조"
        }

        for clause_type, pattern in clause_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 조항 전체 추출 (제N조 형식)
                article_match = re.search(r'제\d+조[^제]*', text[match.start():match.start()+500])
                if article_match:
                    clauses.append({
                        "type": clause_type,
                        "text": article_match.group(0).strip()[:200],
                        "position": match.start()
                    })

        return clauses[:10]  # 상위 10개만

    def _identify_contract_issues(self, text: str, clauses: List[Dict]) -> List[Dict[str, Any]]:
        """계약서 잠재 문제점 식별"""
        issues = []

        # 필수 조항 확인
        required_clauses = ["payment", "period", "termination"]
        found_types = set(c["type"] for c in clauses)

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
        elements = []

        # 기본 요소 패턴
        patterns = {
            "parties": r"(피고소인|피해자|가해자)",
            "facts": r"(사실관계|경위|내용)",
            "claims": r"(청구|요구|주장)",
        }

        for elem_type, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                elements.append({
                    "type": elem_type,
                    "found": True
                })

        return elements

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
    def route_expert(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """전문가 서브그래프로 라우팅"""
        try:
            start_time = time.time()

            # 복잡도 평가
            complexity = self._assess_complexity(state)
            state["complexity_level"] = complexity

            # 전문가 라우팅 필요 여부
            requires_expert = (
                complexity == "complex" and
                state.get("legal_field") in ["family", "corporate", "intellectual_property"]
            )
            state["requires_expert"] = requires_expert

            if requires_expert:
                # 전문가 서브그래프 결정
                expert_map = {
                    "family": "family_law_expert",
                    "corporate": "corporate_law_expert",
                    "intellectual_property": "ip_law_expert"
                }
                state["expert_subgraph"] = expert_map.get(state.get("legal_field"))
                self.logger.info(f"Routing to expert: {state['expert_subgraph']}")
            else:
                state["expert_subgraph"] = None

            self._update_processing_time(state, start_time)
            self._add_step(state, "전문가 라우팅", f"복잡도: {complexity}, 전문가: {requires_expert}")

        except Exception as e:
            self._handle_error(state, str(e), "전문가 라우팅 중 오류")
            state["complexity_level"] = "simple"
            state["requires_expert"] = False
            state["expert_subgraph"] = None

        return state

    def _assess_complexity(self, state: LegalWorkflowState) -> str:
        """질문 복잡도 평가"""
        # 복잡도 지표들
        indicators = {
            "query_length": len(state["query"]),
            "num_keywords": len(state.get("extracted_keywords", [])),
            "has_document": bool(state.get("uploaded_document")),
            "high_urgency": state.get("urgency_level") in ["high", "critical"],
            "multiple_legal_issues": len(state.get("potential_issues", [])) > 2
        }

        complexity_score = 0

        # 점수 계산
        if indicators["query_length"] > 200:
            complexity_score += 2
        if indicators["num_keywords"] > 10:
            complexity_score += 2
        if indicators["has_document"]:
            complexity_score += 3
        if indicators["high_urgency"]:
            complexity_score += 1
        if indicators["multiple_legal_issues"]:
            complexity_score += 2

        # 복잡도 판정
        if complexity_score >= 7:
            return "complex"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "simple"

    def _should_analyze_document(self, state: LegalWorkflowState) -> str:
        """문서 분석 필요 여부 결정"""
        if state.get("uploaded_document"):
            return "analyze"
        return "skip"
