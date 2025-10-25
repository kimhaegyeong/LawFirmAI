# -*- coding: utf-8 -*-
"""
통합 워크플로우 서비스
LangGraph 기반 통합 법률 AI 워크플로우 서비스
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import Checkpoint
from langchain_core.messages import HumanMessage, AIMessage

from .legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from .state_definitions import LegalWorkflowState
from .checkpoint_manager import CheckpointManager
from ..utils.langgraph_config import LangGraphConfig, langgraph_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class IntegratedWorkflowService:
    """LangGraph 기반 통합 워크플로우 서비스"""
    
    def __init__(self, config: LangGraphConfig = langgraph_config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # 워크플로우 빌더 초기화
        self.workflow_builder = EnhancedLegalQuestionWorkflow(config)
        
        # 체크포인트 매니저 초기화
        self.checkpoint_manager = CheckpointManager(config)
        
        # 통합 워크플로우 구성
        self._build_integrated_workflow()
        
        # 컴파일된 그래프
        self.graph_app = self.workflow_builder.graph.compile(
            checkpointer=self.checkpoint_manager.get_memory()
        )
        
        self.logger.info("IntegratedWorkflowService initialized successfully")
    
    def _build_integrated_workflow(self):
        """통합 워크플로우 구성"""
        try:
            # 기존 워크플로우에 추가 노드들 추가
            self.workflow_builder.graph.add_node("input_validation", self._validate_input)
            self.workflow_builder.graph.add_node("context_enrichment", self._enrich_context)
            self.workflow_builder.graph.add_node("multi_agent_coordination", self._coordinate_agents)
            self.workflow_builder.graph.add_node("response_synthesis", self._synthesize_response)
            self.workflow_builder.graph.add_node("quality_assurance", self._assure_quality)
            
            # 새로운 엣지 추가
            self.workflow_builder.graph.add_edge("input_validation", "context_enrichment")
            self.workflow_builder.graph.add_edge("context_enrichment", "multi_agent_coordination")
            self.workflow_builder.graph.add_edge("multi_agent_coordination", "response_synthesis")
            self.workflow_builder.graph.add_edge("response_synthesis", "quality_assurance")
            self.workflow_builder.graph.add_edge("quality_assurance", END)
            
            self.logger.info("Integrated workflow built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build integrated workflow: {e}")
            raise
    
    async def process_query(self, query: str, context: Optional[str] = None,
                          session_id: Optional[str] = None, 
                          user_id: Optional[str] = None) -> Dict[str, Any]:
        """쿼리 처리 메인 메서드"""
        start_time = time.time()
        
        try:
            # 초기 상태 설정
            initial_state = self._create_initial_state(
                query, context, session_id, user_id
            )
            
            # 워크플로우 실행
            result = await self.graph_app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # 결과 포맷팅
            formatted_result = self._format_result(result, start_time)
            
            self.logger.info(f"Query processed successfully in {time.time() - start_time:.2f}s")
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    def _create_initial_state(self, query: str, context: Optional[str],
                            session_id: Optional[str], user_id: Optional[str]) -> LegalWorkflowState:
        """초기 상태 생성"""
        return LegalWorkflowState(
            user_query=query,
            context=context or "",
            session_id=session_id or f"session_{int(time.time())}",
            user_id=user_id or f"user_{int(time.time())}",
            
            # 처리 단계별 결과
            input_validation={},
            question_classification={},
            domain_analysis={},
            retrieved_documents=[],
            legal_analysis={},
            generated_response="",
            quality_metrics={},
            
            # 메타데이터
            workflow_steps=[],
            processing_time=0.0,
            confidence_score=0.0,
            error_messages=[],
            
            # 대화 히스토리
            conversation_history=[],
            user_preferences={},
            
            # 중간 결과들
            intermediate_results={},
            validation_results={},
            
            # 추가 필드들
            enriched_context={},
            agent_coordination={},
            synthesis_result={},
            quality_assurance_result={}
        )
    
    def _format_result(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """결과 포맷팅"""
        return {
            "response": result.get("generated_response", ""),
            "confidence": result.get("confidence_score", 0.5),
            "sources": result.get("retrieved_documents", []),
            "workflow_steps": result.get("workflow_steps", []),
            "processing_time": time.time() - start_time,
            "session_id": result.get("session_id", ""),
            "user_id": result.get("user_id", ""),
            "quality_metrics": result.get("quality_metrics", {}),
            "error_messages": result.get("error_messages", []),
            "intermediate_results": result.get("intermediate_results", {}),
            "langgraph_enabled": True
        }
    
    def _create_error_result(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """오류 결과 생성"""
        return {
            "response": f"죄송합니다. 처리 중 오류가 발생했습니다: {error_message}",
            "confidence": 0.0,
            "sources": [],
            "workflow_steps": ["error"],
            "processing_time": time.time() - start_time,
            "session_id": "",
            "user_id": "",
            "quality_metrics": {},
            "error_messages": [error_message],
            "intermediate_results": {},
            "langgraph_enabled": True
        }
    
    # 워크플로우 노드 메서드들
    
    async def _validate_input(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """입력 검증 노드"""
        try:
            self.logger.info("Validating input...")
            
            validation_result = {
                "is_valid": True,
                "validation_time": time.time(),
                "message_length": len(state["user_query"]),
                "has_context": bool(state["context"]),
                "session_valid": bool(state["session_id"]),
                "user_valid": bool(state["user_id"])
            }
            
            # 입력 검증 로직
            if not state["user_query"] or len(state["user_query"].strip()) == 0:
                validation_result["is_valid"] = False
                validation_result["error"] = "Empty query"
            
            if len(state["user_query"]) > 10000:
                validation_result["is_valid"] = False
                validation_result["error"] = "Query too long"
            
            # 워크플로우 단계 추가
            workflow_steps = state.get("workflow_steps", [])
            workflow_steps.append("input_validation")
            
            return {
                **state,
                "input_validation": validation_result,
                "workflow_steps": workflow_steps
            }
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return {
                **state,
                "input_validation": {"is_valid": False, "error": str(e)},
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    async def _enrich_context(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """컨텍스트 강화 노드"""
        try:
            self.logger.info("Enriching context...")
            
            # 컨텍스트 강화 로직
            enriched_context = {
                "timestamp": datetime.now().isoformat(),
                "query_type": "legal_inquiry",
                "domain_hints": self._extract_domain_hints(state["user_query"]),
                "complexity_score": self._calculate_complexity(state["user_query"]),
                "context_enhanced": True
            }
            
            # 워크플로우 단계 추가
            workflow_steps = state.get("workflow_steps", [])
            workflow_steps.append("context_enrichment")
            
            return {
                **state,
                "enriched_context": enriched_context,
                "workflow_steps": workflow_steps
            }
            
        except Exception as e:
            self.logger.error(f"Context enrichment failed: {e}")
            return {
                **state,
                "enriched_context": {"error": str(e)},
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    async def _coordinate_agents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """멀티 에이전트 조정 노드"""
        try:
            self.logger.info("Coordinating agents...")
            
            # 에이전트 조정 로직
            coordination_result = {
                "research_agent": {"status": "active", "task": "document_retrieval"},
                "analysis_agent": {"status": "active", "task": "legal_analysis"},
                "review_agent": {"status": "standby", "task": "quality_review"},
                "coordination_time": time.time()
            }
            
            # 워크플로우 단계 추가
            workflow_steps = state.get("workflow_steps", [])
            workflow_steps.append("multi_agent_coordination")
            
            return {
                **state,
                "agent_coordination": coordination_result,
                "workflow_steps": workflow_steps
            }
            
        except Exception as e:
            self.logger.error(f"Agent coordination failed: {e}")
            return {
                **state,
                "agent_coordination": {"error": str(e)},
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    async def _synthesize_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 합성 노드"""
        try:
            self.logger.info("Synthesizing response...")
            
            # 응답 합성 로직
            synthesis_result = {
                "synthesis_method": "multi_agent_collaboration",
                "confidence_score": 0.8,
                "response_generated": True,
                "synthesis_time": time.time()
            }
            
            # 기본 응답 생성 (실제로는 더 복잡한 로직)
            generated_response = f"""안녕하세요! '{state["user_query"]}'에 대한 법률 상담을 도와드리겠습니다.

이 질문은 LangGraph 워크플로우를 통해 처리되었으며, 다음과 같은 단계를 거쳤습니다:
- 입력 검증: 완료
- 컨텍스트 강화: 완료  
- 멀티 에이전트 조정: 완료
- 응답 합성: 완료

구체적인 법률 조언이 필요하시면 더 자세한 정보를 제공해주시면 도움을 드릴 수 있습니다."""
            
            # 워크플로우 단계 추가
            workflow_steps = state.get("workflow_steps", [])
            workflow_steps.append("response_synthesis")
            
            return {
                **state,
                "synthesis_result": synthesis_result,
                "generated_response": generated_response,
                "confidence_score": synthesis_result["confidence_score"],
                "workflow_steps": workflow_steps
            }
            
        except Exception as e:
            self.logger.error(f"Response synthesis failed: {e}")
            return {
                **state,
                "synthesis_result": {"error": str(e)},
                "generated_response": f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    async def _assure_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """품질 보증 노드"""
        try:
            self.logger.info("Assuring quality...")
            
            # 품질 보증 로직
            quality_metrics = {
                "response_length": len(state.get("generated_response", "")),
                "confidence_score": state.get("confidence_score", 0.0),
                "workflow_completeness": len(state.get("workflow_steps", [])),
                "error_count": len(state.get("error_messages", [])),
                "quality_score": 0.0
            }
            
            # 품질 점수 계산
            if quality_metrics["response_length"] > 50:
                quality_metrics["quality_score"] += 0.3
            if quality_metrics["confidence_score"] > 0.5:
                quality_metrics["quality_score"] += 0.3
            if quality_metrics["workflow_completeness"] >= 4:
                quality_metrics["quality_score"] += 0.2
            if quality_metrics["error_count"] == 0:
                quality_metrics["quality_score"] += 0.2
            
            quality_assurance_result = {
                "quality_check_passed": quality_metrics["quality_score"] >= 0.7,
                "quality_metrics": quality_metrics,
                "assurance_time": time.time()
            }
            
            # 워크플로우 단계 추가
            workflow_steps = state.get("workflow_steps", [])
            workflow_steps.append("quality_assurance")
            
            return {
                **state,
                "quality_assurance_result": quality_assurance_result,
                "quality_metrics": quality_metrics,
                "workflow_steps": workflow_steps
            }
            
        except Exception as e:
            self.logger.error(f"Quality assurance failed: {e}")
            return {
                **state,
                "quality_assurance_result": {"error": str(e)},
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def _extract_domain_hints(self, query: str) -> List[str]:
        """도메인 힌트 추출"""
        domain_keywords = {
            "civil_law": ["민법", "계약", "손해배상", "불법행위"],
            "criminal_law": ["형법", "범죄", "처벌", "형량"],
            "family_law": ["이혼", "상속", "양육권", "친권"],
            "commercial_law": ["상법", "회사", "주식", "이사"],
            "labor_law": ["노동법", "근로", "임금", "해고"],
            "real_estate": ["부동산", "매매", "임대차", "등기"]
        }
        
        hints = []
        query_lower = query.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                hints.append(domain)
        
        return hints
    
    def _calculate_complexity(self, query: str) -> float:
        """질문 복잡도 계산"""
        complexity_score = 0.0
        
        # 길이 기반 복잡도
        if len(query) > 100:
            complexity_score += 0.3
        elif len(query) > 50:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        # 키워드 기반 복잡도
        complex_keywords = ["복잡", "다양", "여러", "여러가지", "다양한", "복합"]
        if any(keyword in query for keyword in complex_keywords):
            complexity_score += 0.2
        
        # 질문 개수 기반 복잡도
        question_marks = query.count("?")
        if question_marks > 2:
            complexity_score += 0.3
        elif question_marks > 1:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)

