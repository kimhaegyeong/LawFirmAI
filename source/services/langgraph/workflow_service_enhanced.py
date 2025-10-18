# -*- coding: utf-8 -*-
"""
개선된 LangGraph Workflow Service
향상된 답변 품질을 위한 워크플로우 서비스
"""

import logging
import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import Checkpoint

from .legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from .state_definitions import LegalWorkflowState
from .checkpoint_manager import CheckpointManager
from ...utils.langgraph_config import LangGraphConfig, langgraph_config
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class EnhancedLangGraphWorkflowService:
    """개선된 LangGraph 기반 워크플로우 서비스"""
    
    def __init__(self, config: LangGraphConfig = langgraph_config):
        self.config = config
        self.workflow_builder = EnhancedLegalQuestionWorkflow(config)
        self.checkpoint_manager = CheckpointManager(config)
        
        # 워크플로우 컴파일
        self.graph_app = self.workflow_builder.graph.compile(
            checkpointer=self.checkpoint_manager.get_memory()
        )
        
        logger.info("EnhancedLangGraphWorkflowService initialized.")
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """사용자 쿼리를 개선된 LangGraph 워크플로우로 처리"""
        start_time = time.time()
        
        if not session_id:
            session_id = f"enhanced_session_{int(time.time())}"
            logger.info(f"New enhanced session created: {session_id}")
        
        # 초기 상태 설정
        initial_state: LegalWorkflowState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "query_type": None,
            "context": {},
            "retrieved_docs": [],
            "analysis_result": None,
            "answer": None,
            "confidence": 0.0,
            "sources": [],
            "legal_references": [],
            "next_action": None,
            "errors": [],
            "processing_steps": [],
            "session_id": session_id,
            "metadata": {}
        }
        
        try:
            # LangGraph 실행
            final_state = await self.graph_app.ainvoke(
                input=initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            processing_time = time.time() - start_time
            
            # 결과 포맷팅
            return {
                "response": final_state.get("answer", "답변을 생성하지 못했습니다."),
                "confidence": final_state.get("confidence", 0.0),
                "sources": final_state.get("sources", []),
                "processing_time": processing_time,
                "session_id": session_id,
                "query_type": final_state.get("query_type", "unknown"),
                "legal_references": final_state.get("legal_references", []),
                "processing_steps": final_state.get("processing_steps", []),
                "metadata": final_state.get("metadata", {}),
                "errors": final_state.get("errors", []),
                "enhanced": True  # 개선된 워크플로우 사용 표시
            }
            
        except Exception as e:
            logger.error(f"Error processing query with enhanced LangGraph for session {session_id}: {e}", exc_info=True)
            return {
                "response": "죄송합니다. 개선된 LangGraph 워크플로우 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": time.time() - start_time,
                "session_id": session_id,
                "errors": [str(e)],
                "enhanced": True
            }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """체크포인트 관리자를 통해 세션 정보 조회"""
        return self.checkpoint_manager.get_session_info(session_id)
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        return {
            "service_name": "EnhancedLangGraphWorkflowService",
            "status": "running",
            "checkpoint_storage": self.config.checkpoint_storage.value,
            "llm_provider": self.config.llm_provider,
            "enhanced_features": {
                "prompt_templates": True,
                "keyword_mapping": True,
                "response_enhancement": True,
                "structured_output": True
            },
            "timestamp": time.time()
        }
    
    def get_enhancement_metrics(self) -> Dict[str, Any]:
        """개선 기능 메트릭 조회"""
        return {
            "prompt_templates_count": len([
                "CONTRACT_REVIEW_TEMPLATE", "FAMILY_LAW_TEMPLATE", "CRIMINAL_LAW_TEMPLATE",
                "CIVIL_LAW_TEMPLATE", "LABOR_LAW_TEMPLATE", "PROPERTY_LAW_TEMPLATE",
                "INTELLECTUAL_PROPERTY_TEMPLATE", "TAX_LAW_TEMPLATE", "CIVIL_PROCEDURE_TEMPLATE",
                "GENERAL_TEMPLATE"
            ]),
            "keyword_categories_count": len(self.workflow_builder.keyword_mapper.KEYWORD_MAPPING),
            "legal_terms_count": len(self.workflow_builder.keyword_mapper.LEGAL_TERMS),
            "enhancement_features": [
                "질문 유형별 맞춤형 프롬프트",
                "키워드 매핑 및 포함도 검증",
                "답변 구조화 강화",
                "법률 용어 사용 증가",
                "메타데이터 추적"
            ]
        }
