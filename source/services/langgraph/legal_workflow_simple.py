# -*- coding: utf-8 -*-
"""
LangGraph Legal Workflow - Simplified Version
의존성 문제를 해결한 간소화된 법률 질문 처리 워크플로우
"""

import logging
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .state_definitions import LegalWorkflowState

logger = logging.getLogger(__name__)


class LegalQuestionWorkflow:
    """법률 질문 처리 워크플로우 (간소화 버전)"""
    
    def __init__(self, config):
        """
        워크플로우 초기화
        
        Args:
            config: LangGraph 설정 객체
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
    
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
        """질문 분류 (Mock 구현)"""
        try:
            start_time = time.time()
            
            # 간단한 키워드 기반 분류
            query = state["query"].lower()
            
            if any(keyword in query for keyword in ["계약", "계약서", "contract"]):
                query_type = "contract_review"
                confidence = 0.8
            elif any(keyword in query for keyword in ["이혼", "위자료", "divorce"]):
                query_type = "family_law"
                confidence = 0.8
            elif any(keyword in query for keyword in ["형사", "범죄", "criminal"]):
                query_type = "criminal_law"
                confidence = 0.8
            else:
                query_type = "general_question"
                confidence = 0.6
            
            state["query_type"] = query_type
            state["confidence"] = confidence
            state["processing_steps"].append(f"질문 분류 완료: {query_type}")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Query classified as {query_type} with confidence {confidence}")
            
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
        """문서 검색 (Mock 구현)"""
        try:
            start_time = time.time()
            
            # Mock 검색 결과 생성
            mock_docs = [
                {
                    "content": f"'{state['query']}'에 대한 법률 정보입니다. 이는 {state['query_type']} 영역의 질문으로 보입니다.",
                    "source": "법률 데이터베이스",
                    "relevance_score": 0.8,
                    "law_name": "관련 법령"
                },
                {
                    "content": f"관련 판례 및 법령 정보가 포함된 문서입니다.",
                    "source": "판례 데이터베이스", 
                    "relevance_score": 0.7,
                    "law_name": "관련 판례"
                }
            ]
            
            state["retrieved_docs"] = mock_docs
            state["search_metadata"] = {
                "total_results": len(mock_docs),
                "search_time": time.time() - start_time,
                "query_type": state["query_type"]
            }
            
            state["processing_steps"].append(f"{len(mock_docs)}개 문서 검색 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Found {len(mock_docs)} mock documents for query")
            
        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본값 설정
            state["retrieved_docs"] = []
        
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
            
            state["legal_references"] = list(set(references))
            state["analysis"] = f"질문 유형: {state['query_type']}, 관련 법령: {', '.join(references)}"
            state["processing_steps"].append("컨텍스트 분석 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info(f"Context analysis completed with {len(references)} references")
            
        except Exception as e:
            error_msg = f"컨텍스트 분석 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    def generate_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 생성 (Ollama 사용)"""
        try:
            start_time = time.time()
            
            # 컨텍스트 구성
            context = "\n\n".join([
                doc.get("content", "") for doc in state["retrieved_docs"][:3]
            ])
            
            # 프롬프트 구성
            prompt = f"""질문: {state["query"]}

관련 법률 문서:
{context}

위 법률 문서를 바탕으로 질문에 대해 정확하고 전문적인 답변을 제공해주세요.
답변은 한국어로 작성하고, 법률적 근거를 포함해주세요."""

            # Ollama로 답변 생성 (간단한 구현)
            try:
                from langchain_community.llms import Ollama
                
                ollama_llm = Ollama(
                    model="qwen2.5:7b",
                    base_url="http://localhost:11434"
                )
                
                response = ollama_llm.invoke(prompt)
                state["answer"] = response
                
            except Exception as ollama_error:
                # Ollama 실패 시 기본 답변
                state["answer"] = f"""질문: {state["query"]}

이 질문은 {state["query_type"]} 영역에 해당합니다.

관련 법률 정보:
{context}

위 정보를 바탕으로 답변드리면, 해당 질문에 대한 구체적인 법률적 조언을 위해서는 전문가와 상담하시는 것을 권장합니다.

참고: 이 답변은 일반적인 정보 제공 목적이며, 구체적인 법률적 조언이 아닙니다."""
                
                self.logger.warning(f"Ollama generation failed, using fallback: {ollama_error}")
            
            state["processing_steps"].append("답변 생성 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info("Answer generation completed")
            
        except Exception as e:
            error_msg = f"답변 생성 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
            
            # 기본 답변 설정
            state["answer"] = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        
        return state
    
    def format_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 포맷팅"""
        try:
            start_time = time.time()
            
            # 소스 정보 추출
            sources = []
            for doc in state["retrieved_docs"][:3]:
                if "source" in doc:
                    sources.append(doc["source"])
            
            state["sources"] = sources
            state["processing_steps"].append("응답 포맷팅 완료")
            
            processing_time = time.time() - start_time
            state["processing_time"] += processing_time
            
            self.logger.info("Response formatting completed")
            
        except Exception as e:
            error_msg = f"응답 포맷팅 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            self.logger.error(error_msg)
        
        return state
    
    def compile(self):
        """워크플로우 컴파일"""
        return self.graph.compile()
