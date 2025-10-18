# -*- coding: utf-8 -*-
"""
개선된 LangGraph Legal Workflow
답변 품질 향상을 위한 향상된 워크플로우 구현
"""

import logging
import time
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from .state_definitions import LegalWorkflowState
from .performance_optimizer import PerformanceOptimizer
from .legal_data_connector import LegalDataConnector
from .prompt_templates import LegalPromptTemplates
from .keyword_mapper import LegalKeywordMapper
from ...utils.langgraph_config import LangGraphConfig

logger = logging.getLogger(__name__)

# Mock QuestionType for enhanced workflow
class QuestionType:
    GENERAL_QUESTION = "general_question"
    LAW_INQUIRY = "law_inquiry"
    PRECEDENT_SEARCH = "precedent_search"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURE_GUIDE = "procedure_guide"
    TERM_EXPLANATION = "term_explanation"
    CONTRACT_REVIEW = "contract_review"
    FAMILY_LAW = "family_law"
    CRIMINAL_LAW = "criminal_law"
    CIVIL_LAW = "civil_law"
    LABOR_LAW = "labor_law"
    PROPERTY_LAW = "property_law"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    TAX_LAW = "tax_law"
    CIVIL_PROCEDURE = "civil_procedure"


class EnhancedLegalQuestionWorkflow:
    """개선된 법률 질문 처리 워크플로우"""
    
    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.prompt_templates = LegalPromptTemplates()
        self.keyword_mapper = LegalKeywordMapper()
        self.data_connector = LegalDataConnector()
        self.performance_optimizer = PerformanceOptimizer()
        
        # LLM 초기화
        self.llm = self._initialize_llm()
        
        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.info("EnhancedLegalQuestionWorkflow initialized.")
    
    def _initialize_llm(self):
        """LLM 초기화 (Google Gemini 우선, Ollama 백업)"""
        if self.config.llm_provider == "google":
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=self.config.google_model,
                    temperature=0.3,
                    max_output_tokens=200,  # 답변 길이 증가
                    timeout=15,  # 타임아웃 증가
                    api_key=self.config.google_api_key
                )
                gemini_llm.invoke("test")  # 모델 로드 확인
                logger.info(f"Initialized Google Gemini LLM: {self.config.google_model}")
                return gemini_llm
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Falling back to Ollama.")
        
        if self.config.llm_provider == "ollama":
            try:
                ollama_llm = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=0.3,
                    num_predict=200,  # 답변 길이 증가
                    timeout=20  # 타임아웃 증가
                )
                ollama_llm.invoke("test")  # 모델 로드 확인
                logger.info(f"Initialized Ollama LLM: {self.config.ollama_model}")
                return ollama_llm
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}. Using Mock LLM.")
        
        # Fallback to a simple mock LLM if all providers fail
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
        workflow.add_node("generate_answer_enhanced", self.generate_answer_enhanced)
        workflow.add_node("format_response", self.format_response)
        
        # 엣지 설정
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer_enhanced")
        workflow.add_edge("generate_answer_enhanced", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 (개선된 버전)"""
        try:
            start_time = time.time()
            query = state["query"].lower()
            
            # 개선된 키워드 기반 분류
            if any(k in query for k in ["계약", "계약서", "매매", "임대", "도급"]):
                state["query_type"] = QuestionType.CONTRACT_REVIEW
            elif any(k in query for k in ["이혼", "가족", "상속", "양육", "입양"]):
                state["query_type"] = QuestionType.FAMILY_LAW
            elif any(k in query for k in ["절도", "범죄", "형사", "사기", "폭행", "강도", "살인"]):
                state["query_type"] = QuestionType.CRIMINAL_LAW
            elif any(k in query for k in ["손해배상", "민사", "불법행위", "채권", "소유권"]):
                state["query_type"] = QuestionType.CIVIL_LAW
            elif any(k in query for k in ["해고", "노동", "임금", "근로시간", "휴가", "산업재해"]):
                state["query_type"] = QuestionType.LABOR_LAW
            elif any(k in query for k in ["부동산", "매매", "등기", "공시", "토지"]):
                state["query_type"] = QuestionType.PROPERTY_LAW
            elif any(k in query for k in ["특허", "지적재산권", "저작권", "상표", "디자인"]):
                state["query_type"] = QuestionType.INTELLECTUAL_PROPERTY
            elif any(k in query for k in ["세금", "소득세", "부가가치세", "법인세", "상속세", "가산세"]):
                state["query_type"] = QuestionType.TAX_LAW
            elif any(k in query for k in ["소송", "관할", "증거", "판결", "집행", "민사소송"]):
                state["query_type"] = QuestionType.CIVIL_PROCEDURE
            else:
                state["query_type"] = QuestionType.GENERAL_QUESTION
            
            state["confidence"] = 0.8  # 분류 신뢰도 향상
            state["processing_steps"].append(f"질문 분류 완료 (개선): {state['query_type']}")
            
            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time
            
            self.logger.info(f"Query classified as {state['query_type']} with confidence {state['confidence']}")
            
        except Exception as e:
            error_msg = f"질문 분류 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본값 설정
            state["query_type"] = QuestionType.GENERAL_QUESTION
            state["confidence"] = 0.5
        
        return state
    
    def retrieve_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색 (캐싱 적용)"""
        try:
            start_time = time.time()
            
            query = state["query"]
            query_type = state["query_type"]
            
            # 캐시에서 문서 확인
            cached_documents = self.performance_optimizer.cache.get_cached_documents(query, query_type)
            
            if cached_documents:
                state["retrieved_docs"] = cached_documents
                state["processing_steps"].append(f"{len(cached_documents)}개 캐시된 문서 사용")
                self.logger.info(f"Using cached documents for query: {query[:50]}...")
            else:
                # 실제 데이터베이스에서 문서 검색
                documents = self.data_connector.search_documents(query, query_type, limit=5)
                
                # 검색 결과가 부족한 경우 카테고리별 문서 추가
                if len(documents) < 3:
                    category_docs = self.data_connector.get_document_by_category(query_type, limit=3)
                    # 중복 제거
                    existing_ids = {doc["id"] for doc in documents}
                    for doc in category_docs:
                        if doc["id"] not in existing_ids:
                            documents.append(doc)
                
                state["retrieved_docs"] = documents
                state["processing_steps"].append(f"{len(documents)}개 실제 문서 검색 완료")
                
                # 문서 캐싱
                self.performance_optimizer.cache.cache_documents(query, query_type, documents)
            
            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time
            
            self.logger.info(f"Retrieved {len(state['retrieved_docs'])} documents for query type {query_type}")
            
        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 폴백: 기본 문서 설정
            state["retrieved_docs"] = [
                {"content": f"'{state['query']}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
            ]
        
        return state
    
    def generate_answer_enhanced(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """개선된 답변 생성"""
        try:
            start_time = time.time()
            
            # 컨텍스트 구성
            context = "\n".join([doc["content"] for doc in state["retrieved_docs"]])
            
            # 질문 유형별 키워드 추출
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                state["query"], state["query_type"]
            )
            
            # 질문 유형별 프롬프트 템플릿 선택
            template = self.prompt_templates.get_template_for_query_type(state["query_type"])
            
            # 프롬프트 구성
            prompt = template.format(
                question=state["query"],
                context=context,
                required_keywords=", ".join(required_keywords[:10])  # 상위 10개 키워드만 사용
            )
            
            # LLM 호출
            response = self._call_llm_with_retry(prompt)
            
            # 답변 후처리 (구조화 강화)
            enhanced_response = self._enhance_response_structure(response, required_keywords)
            
            state["answer"] = enhanced_response
            state["processing_steps"].append("개선된 답변 생성 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time
            
            self.logger.info(f"Enhanced answer generated in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"개선된 답변 생성 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본 답변 설정
            state["answer"] = self._generate_fallback_answer(state)
        
        return state
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """LLM 호출 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    else:
                        return str(response)
                else:
                    return self.llm.invoke(prompt)
            except Exception as e:
                self.logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # 재시도 전 대기
        
        return "LLM 호출에 실패했습니다."
    
    def _enhance_response_structure(self, response: str, required_keywords: List[str]) -> str:
        """답변 구조화 강화"""
        # 키워드 포함 확인 및 강화
        missing_keywords = self.keyword_mapper.get_missing_keywords(response, required_keywords[:5])
        
        if missing_keywords:
            # 누락된 키워드 추가
            response += f"\n\n## 추가 고려사항\n"
            for keyword in missing_keywords[:3]:  # 최대 3개만 추가
                response += f"- {keyword} 관련 사항도 고려해야 합니다.\n"
        
        # 구조화 강화
        if "##" not in response:
            # 제목이 없으면 추가
            response = f"## 답변\n{response}"
        
        if not any(marker in response for marker in ["1.", "2.", "3.", "•", "-"]):
            # 목록이 없으면 추가
            response += "\n\n## 주요 포인트\n- 위 내용을 참고하여 구체적인 조치를 취하시기 바랍니다."
        
        return response
    
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
            
            # 최종 답변 및 메타데이터 정리
            final_answer = state.get("answer", "답변을 생성하지 못했습니다.")
            final_confidence = state.get("confidence", 0.0)
            final_sources = [doc.get("source", "Unknown") for doc in state.get("retrieved_docs", [])]
            
            # 키워드 포함도 계산
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                state["query"], state["query_type"]
            )
            keyword_coverage = self.keyword_mapper.calculate_keyword_coverage(
                final_answer, required_keywords
            )
            
            # 신뢰도 조정 (키워드 포함도 반영)
            adjusted_confidence = min(0.9, final_confidence + (keyword_coverage * 0.2))
            
            state["answer"] = final_answer
            state["confidence"] = adjusted_confidence
            state["sources"] = list(set(final_sources))  # 중복 제거
            state["legal_references"] = []  # 실제 참조가 있다면 여기에 추가
            state["processing_steps"].append("응답 포맷팅 완료 (개선)")
            
            # 메타데이터 추가
            state["metadata"] = {
                "keyword_coverage": keyword_coverage,
                "required_keywords_count": len(required_keywords),
                "matched_keywords_count": len(required_keywords) - len(
                    self.keyword_mapper.get_missing_keywords(final_answer, required_keywords)
                ),
                "response_length": len(final_answer),
                "query_type": state["query_type"]
            }
            
            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time
            
            self.logger.info("Enhanced response formatting completed")
            
        except Exception as e:
            error_msg = f"응답 포맷팅 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
