# -*- coding: utf-8 -*-
"""
LangGraph Legal Workflow
법률 질문 처리 워크플로우 구현
"""

import logging
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not available. Please install langgraph")

from .state_definitions import LegalWorkflowState

# 기존 서비스들을 안전하게 import
try:
    from ..question_classifier import QuestionClassifier
    QUESTION_CLASSIFIER_AVAILABLE = True
except ImportError:
    QUESTION_CLASSIFIER_AVAILABLE = False
    logging.warning("QuestionClassifier not available")

try:
    from ..hybrid_search_engine import HybridSearchEngine
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    logging.warning("HybridSearchEngine not available")

try:
    from ..ollama_client import OllamaClient
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logging.warning("OllamaClient not available")

logger = logging.getLogger(__name__)


class LegalQuestionWorkflow:
    """법률 질문 처리 워크플로우"""
    
    def __init__(self, config):
        """
        워크플로우 초기화
        
        Args:
            config: LangGraph 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화 (안전한 방식)
        if QUESTION_CLASSIFIER_AVAILABLE:
            self.classifier = QuestionClassifier()
        else:
            self.classifier = None
            self.logger.warning("QuestionClassifier not available, using mock")
        
        if HYBRID_SEARCH_AVAILABLE:
            self.search_engine = HybridSearchEngine()
        else:
            self.search_engine = None
            self.logger.warning("HybridSearchEngine not available, using mock")
        
        if OLLAMA_CLIENT_AVAILABLE:
            self.ollama_client = OllamaClient(
                base_url=config.ollama_base_url,
                model_name=config.ollama_model,
                timeout=config.ollama_timeout
            )
        else:
            self.ollama_client = None
            self.logger.warning("OllamaClient not available, using mock")
        
        # 워크플로우 그래프 구축
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None
            self.logger.warning("LangGraph not available, workflow will not be functional")
    
    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)
        
        # 노드 추가
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("search_documents", self.search_documents)
        workflow.add_node("analyze_context", self.analyze_context)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("format_response", self.format_response)
        
        # 엣지 설정
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "search_documents")
        workflow.add_edge("search_documents", "analyze_context")
        workflow.add_edge("analyze_context", "generate_answer")
        workflow.add_edge("generate_answer", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow
    
    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류"""
        try:
            start_time = time.time()
            
            if self.classifier:
                # 질문 분류 수행
                classification = self.classifier.classify(state["query"])
                
                # 상태 업데이트
                state["query_type"] = classification.question_type.value
                state["confidence"] = classification.confidence
                state["processing_steps"].append(f"질문 분류 완료: {classification.question_type.value}")
                
                self.logger.info(f"Query classified as {classification.question_type.value} with confidence {classification.confidence}")
            else:
                # Mock 분류
                state["query_type"] = "general_question"
                state["confidence"] = 0.7
                state["processing_steps"].append("질문 분류 완료 (Mock): general_question")
                self.logger.info("Using mock question classification")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
        except Exception as e:
            error_msg = f"질문 분류 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본값 설정
            state["query_type"] = "general_question"
            state["confidence"] = 0.5
        
        return state
    
    def search_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색"""
        try:
            start_time = time.time()
            
    def search_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색"""
        try:
            start_time = time.time()
            
            if self.search_engine:
                # 질문 유형에 따른 검색 수행
                search_results = self.search_engine.search(
                    query=state["query"],
                    search_types=[state["query_type"]]
                )
                
                # 검색 결과를 상태에 추가
                state["retrieved_docs"] = search_results.get("results", [])
                self.logger.info(f"Found {len(state['retrieved_docs'])} documents for query")
            else:
                # Mock 검색 결과
                state["retrieved_docs"] = [
                    {
                        "content": f"'{state['query']}'에 대한 법률 정보입니다.",
                        "source": "Mock Document",
                        "relevance_score": 0.8
                    }
                ]
                self.logger.info("Using mock search results")
            
            state["search_metadata"] = {
                "total_results": len(state["retrieved_docs"]),
                "search_time": time.time() - start_time,
                "query_type": state["query_type"]
            }
            
            state["processing_steps"].append(f"{len(state['retrieved_docs'])}개 문서 검색 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            
        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 빈 결과로 설정
            state["retrieved_docs"] = []
            state["search_metadata"] = {"error": str(e)}
        
        return state
    
    def analyze_context(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """컨텍스트 분석"""
        try:
            start_time = time.time()
            
            # 검색된 문서에서 법률 참조 추출
            references = []
            for doc in state["retrieved_docs"]:
                if "law_name" in doc:
                    references.append(doc["law_name"])
                elif "source" in doc:
                    references.append(doc["source"])
            
            # 중복 제거
            state["legal_references"] = list(set(references))
            
            # 컨텍스트 분석 (간단한 요약)
            if state["retrieved_docs"]:
                context_summary = f"검색된 {len(state['retrieved_docs'])}개 문서에서 {len(state['legal_references'])}개의 법률 참조를 발견했습니다."
                state["analysis"] = context_summary
            else:
                state["analysis"] = "관련 문서를 찾을 수 없습니다."
            
            state["processing_steps"].append("컨텍스트 분석 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Context analysis completed: {len(state['legal_references'])} references found")
            
        except Exception as e:
            error_msg = f"컨텍스트 분석 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본값 설정
            state["legal_references"] = []
            state["analysis"] = "분석 중 오류가 발생했습니다."
        
        return state
    
    def generate_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 생성 (Ollama 사용)"""
        try:
            start_time = time.time()
            
            # 컨텍스트 구성
            context_parts = []
            for i, doc in enumerate(state["retrieved_docs"][:5], 1):
                content = doc.get("content", "")
                source = doc.get("source", "Unknown")
                context_parts.append(f"[문서 {i}] {source}\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # 프롬프트 구성
            prompt = f"""질문: {state["query"]}

관련 법률 문서:
{context}

위 법률 문서를 바탕으로 질문에 대해 정확하고 전문적인 답변을 제공해주세요.
답변 시 다음 사항을 고려해주세요:
1. 법률 문서의 내용을 정확히 인용하여 답변하세요
2. 관련 법조문이나 판례가 있다면 구체적으로 언급하세요
3. 불확실한 내용은 추측하지 말고 명확히 밝히세요
4. 답변의 근거가 되는 문서를 참조로 표시하세요"""
            
            # Ollama로 답변 생성
            response = self.ollama_client.generate(
                prompt=prompt,
                system_prompt="당신은 법률 전문가입니다. 정확하고 신뢰할 수 있는 법률 정보를 제공합니다."
            )
            
            state["answer"] = response.response
            state["processing_steps"].append("답변 생성 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Answer generated successfully in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본 답변 설정
            state["answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        
        return state
    
    def format_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 포맷팅"""
        try:
            start_time = time.time()
            
            # 소스 정보 추출
            sources = []
            for doc in state["retrieved_docs"][:3]:
                source = doc.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
            
            state["sources"] = sources
            
            # 메타데이터 업데이트
            state["metadata"] = {
                "query_type": state["query_type"],
                "confidence": state["confidence"],
                "processing_time": state["processing_time"],
                "document_count": len(state["retrieved_docs"]),
                "reference_count": len(state["legal_references"]),
                "has_errors": len(state["errors"]) > 0
            }
            
            state["processing_steps"].append("응답 포맷팅 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Response formatted successfully")
            
        except Exception as e:
            error_msg = f"응답 포맷팅 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본 소스 설정
            state["sources"] = ["Unknown"]
        
        return state
    
    def compile(self):
        """워크플로우 컴파일"""
        if not LANGGRAPH_AVAILABLE:
            self.logger.error("LangGraph not available, cannot compile workflow")
            return None
        
        if not self.graph:
            self.logger.error("Workflow graph not initialized")
            return None
        
        try:
            compiled_workflow = self.graph.compile()
            self.logger.info("Workflow compiled successfully")
            return compiled_workflow
        except Exception as e:
            self.logger.error(f"Failed to compile workflow: {e}")
            return None
