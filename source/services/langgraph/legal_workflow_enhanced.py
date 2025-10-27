# -*- coding: utf-8 -*-
"""
개선된 LangGraph Legal Workflow
답변 품질 향상을 위한 향상된 워크플로우 구현
UnifiedPromptManager 통합 완료
"""

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from ...utils.langgraph_config import LangGraphConfig
from ..semantic_search_engine import SemanticSearchEngine
from ..term_integration_system import TermIntegrator
from .keyword_mapper import LegalKeywordMapper
from .legal_data_connector import LegalDataConnector
from .performance_optimizer import PerformanceOptimizer
from .state_definitions import LegalWorkflowState

# 프로젝트 경로 추가하여 unified_prompt_manager import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 상대 경로로 import (더 안정적)
from ..question_classifier import QuestionType
from ..unified_prompt_manager import LegalDomain, ModelType, UnifiedPromptManager

logger = logging.getLogger(__name__)


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

        # Semantic Search Engine 초기화 (벡터 검색을 위한)
        try:
            self.semantic_search = SemanticSearchEngine()
            self.logger.info("SemanticSearchEngine initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize SemanticSearchEngine: {e}")
            self.semantic_search = None

        # LLM 초기화
        self.llm = self._initialize_llm()

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
        """워크플로우 그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)

        # 노드 추가
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("process_legal_terms", self.process_legal_terms)
        workflow.add_node("generate_answer_enhanced", self.generate_answer_enhanced)
        workflow.add_node("format_response", self.format_response)

        # 엣지 설정
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "process_legal_terms")
        workflow.add_edge("process_legal_terms", "generate_answer_enhanced")
        workflow.add_edge("generate_answer_enhanced", "format_response")
        workflow.add_edge("format_response", END)

        return workflow

    # Helper methods for common operations
    def _update_processing_time(self, state: LegalWorkflowState, start_time: float):
        """처리 시간 업데이트"""
        processing_time = time.time() - start_time
        state["processing_time"] = state.get("processing_time", 0.0) + processing_time
        return processing_time

    def _add_step(self, state: LegalWorkflowState, step_prefix: str, step_message: str):
        """처리 단계 추가 (중복 방지)"""
        existing_steps = state.get("processing_steps", [])
        if not any(step_prefix in step for step in existing_steps):
            state["processing_steps"].append(step_message)

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

    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 (LLM 기반)"""
        try:
            start_time = time.time()

            classified_type, confidence = self._classify_with_llm(state["query"])

            state["query_type"] = classified_type
            state["confidence"] = confidence

            processing_time = self._update_processing_time(state, start_time)
            self._add_step(state, "질문 분류 완료",
                         f"질문 분류 완료: {classified_type.value} (시간: {processing_time:.3f}s)")

            self.logger.info(f"LLM classified query as {classified_type.value} with confidence {confidence}")

        except Exception as e:
            self._handle_error(state, str(e), "LLM 질문 분류 중 오류 발생")
            classified_type, confidence = self._fallback_classification(state["query"])
            state["query_type"] = classified_type
            state["confidence"] = confidence
            self._add_step(state, "폴백 키워드 기반 분류 사용", "폴백 키워드 기반 분류 사용")

        return state

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

    def retrieve_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색 (하이브리드: 벡터 + 키워드 검색)"""
        try:
            start_time = time.time()
            query = state["query"]
            query_type_str = self._get_query_type_str(state["query_type"])

            # 캐시 확인
            if self._check_cache(state, query, query_type_str, start_time):
                return state

            # 하이브리드 검색
            semantic_results, semantic_count = self._semantic_search(query)
            keyword_results, keyword_count = self._keyword_search(query, query_type_str)

            # 결과 통합
            documents = self._merge_search_results(semantic_results, keyword_results)
            state["retrieved_docs"] = documents[:WorkflowConstants.MAX_DOCUMENTS]

            # 메타데이터 및 상태 업데이트
            self._update_search_metadata(state, semantic_count, keyword_count, documents, query_type_str, start_time)
            self.performance_optimizer.cache.cache_documents(query, query_type_str, state["retrieved_docs"])
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
        """검색 결과 통합 및 중복 제거"""
        seen_ids = set()
        documents = []

        # 벡터 검색 결과 우선 추가
        for doc in semantic_results:
            doc_id = doc.get('id')
            if doc_id and doc_id not in seen_ids:
                documents.append(doc)
                seen_ids.add(doc_id)

        # 키워드 검색 결과 추가
        for doc in keyword_results:
            doc_id = doc.get('id')
            if doc_id and doc_id not in seen_ids:
                documents.append(doc)
                seen_ids.add(doc_id)
                if len(documents) >= 10:
                    break

        # 관련성 점수로 정렬
        documents.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return documents

    def _update_search_metadata(self, state: LegalWorkflowState, semantic_count: int,
                                keyword_count: int, documents: List[Dict], query_type_str: str, start_time: float):
        """검색 메타데이터 업데이트"""
        state["search_metadata"] = {
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

    def _get_question_type_and_domain(self, query_type) -> Tuple[QuestionType, LegalDomain]:
        """질문 유형과 도메인 매핑"""
        query_type_mapping = {
            "contract_review": (QuestionType.LEGAL_ADVICE, LegalDomain.CIVIL_LAW),
            "family_law": (QuestionType.LEGAL_ADVICE, LegalDomain.FAMILY_LAW),
            "criminal_law": (QuestionType.LEGAL_ADVICE, LegalDomain.CRIMINAL_LAW),
            "civil_law": (QuestionType.LEGAL_ADVICE, LegalDomain.CIVIL_LAW),
            "labor_law": (QuestionType.LEGAL_ADVICE, LegalDomain.LABOR_LAW),
            "property_law": (QuestionType.LEGAL_ADVICE, LegalDomain.PROPERTY_LAW),
            "intellectual_property": (QuestionType.LEGAL_ADVICE, LegalDomain.INTELLECTUAL_PROPERTY),
            "tax_law": (QuestionType.LEGAL_ADVICE, LegalDomain.TAX_LAW),
            "civil_procedure": (QuestionType.PROCEDURE_GUIDE, LegalDomain.CIVIL_PROCEDURE),
            "general_question": (QuestionType.GENERAL_QUESTION, LegalDomain.GENERAL)
        }
        return query_type_mapping.get(query_type, (QuestionType.GENERAL_QUESTION, LegalDomain.GENERAL))

    def _build_context(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """컨텍스트 구성"""
        context_text = "\n".join([doc["content"] for doc in state["retrieved_docs"]])
        return {
            "context": context_text,
            "legal_references": state.get("legal_references", []),
            "query_type": state["query_type"]
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

    def format_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 포맷팅 (개선된 버전)"""
        try:
            start_time = time.time()

            final_answer = state.get("answer", "답변을 생성하지 못했습니다.")
            final_confidence = state.get("confidence", 0.0)

            # 키워드 포함도 계산
            keyword_coverage = self._calculate_keyword_coverage(state, final_answer)
            adjusted_confidence = min(0.9, final_confidence + (keyword_coverage * 0.2))

            # 상태 업데이트
            state["answer"] = final_answer
            state["confidence"] = adjusted_confidence
            state["sources"] = list(set([doc.get("source", "Unknown") for doc in state.get("retrieved_docs", [])]))
            state["legal_references"] = []

            # 메타데이터 설정
            self._set_metadata(state, final_answer, keyword_coverage)

            self._update_processing_time(state, start_time)
            self._add_step(state, "응답 포맷팅 완료", "응답 포맷팅 완료")
            self.logger.info("Enhanced response formatting completed")
        except Exception as e:
            self._handle_error(state, str(e), "응답 포맷팅 중 오류 발생")
        return state

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
