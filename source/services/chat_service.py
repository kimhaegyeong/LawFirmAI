# -*- coding: utf-8 -*-
"""
Chat Service
채팅 메시지 처리 서비스
"""

import os
import time
from typing import Dict, List, Optional, Any
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChatService:
    """채팅 서비스 클래스"""
    
    def __init__(self, config: Config):
        """채팅 서비스 초기화"""
        self.config = config
        self.logger = get_logger(__name__)
        
        # LangGraph 사용 여부 확인
        self.use_langgraph = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
        
        if self.use_langgraph:
            try:
                from .langgraph.workflow_service import LangGraphWorkflowService
                from ..utils.langgraph_config import LangGraphConfig
                
                langgraph_config = LangGraphConfig.from_env()
                self.langgraph_service = LangGraphWorkflowService(langgraph_config)
                self.logger.info("LangGraph workflow service initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize LangGraph service: {e}")
                self.use_langgraph = False
                self.langgraph_service = None
        else:
            self.langgraph_service = None
        
        # 기존 컴포넌트 초기화 (LangGraph 사용하지 않을 때)
        if not self.use_langgraph:
            self.model_manager = None
            self.rag_service = None
        
        self.logger.info(f"ChatService initialized (LangGraph: {self.use_langgraph})")
    
    async def process_message(self, message: str, context: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """사용자 메시지 처리"""
        try:
            start_time = time.time()
            
            # 입력 검증
            if not self.validate_input(message):
                return {
                    "response": "올바른 질문을 입력해주세요.",
                    "confidence": 0.0,
                    "sources": [],
                    "processing_time": 0.0
                }
            
            # LangGraph 사용 여부에 따른 처리
            if self.use_langgraph and self.langgraph_service:
                return await self._process_with_langgraph(message, session_id)
            else:
                return await self._process_legacy(message, context)
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "response": "죄송합니다. 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0
            }
    
    async def _process_with_langgraph(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """LangGraph를 사용한 메시지 처리"""
        try:
            result = await self.langgraph_service.process_query(message, session_id)
            
            # LangGraph 결과를 기존 형식으로 변환
            return {
                "response": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
                "processing_time": result.get("processing_time", 0.0),
                "session_id": result.get("session_id"),
                "query_type": result.get("query_type", ""),
                "legal_references": result.get("legal_references", []),
                "processing_steps": result.get("processing_steps", []),
                "metadata": result.get("metadata", {}),
                "errors": result.get("errors", [])
            }
            
        except Exception as e:
            self.logger.error(f"LangGraph processing error: {e}")
            return {
                "response": "LangGraph 처리 중 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0,
                "errors": [str(e)]
            }
    
    async def _process_legacy(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        """기존 방식으로 메시지 처리"""
        try:
            start_time = time.time()
            
            # 기존 처리 로직 (placeholder)
            response = self._generate_response(message, context)
            
            processing_time = time.time() - start_time
            
            return {
                "response": response,
                "confidence": 0.8,  # Placeholder
                "sources": [],  # Placeholder
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Legacy processing error: {e}")
            return {
                "response": "기존 처리 방식에서 오류가 발생했습니다.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": 0.0,
                "errors": [str(e)]
            }
    
    def _generate_response(self, message: str, context: Optional[str] = None) -> str:
        """응답 생성 (placeholder)"""
        # This will be implemented with actual AI model in future tasks
        return f"안녕하세요! '{message}'에 대한 질문을 받았습니다. 현재 개발 중인 기능입니다."
    
    def validate_input(self, message: str) -> bool:
        """입력 검증"""
        if not message or not message.strip():
            return False
        
        if len(message) > 10000:  # Max 10,000 characters
            return False
        
        return True
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """대화 기록 조회"""
        if self.use_langgraph and self.langgraph_service:
            try:
                session_info = self.langgraph_service.get_session_info(session_id)
                return [session_info] if session_info else []
            except Exception as e:
                self.logger.error(f"Failed to get conversation history: {e}")
                return []
        else:
            # 기존 방식 (placeholder)
            return []
    
    def clear_conversation_history(self, session_id: str) -> None:
        """대화 기록 삭제"""
        if self.use_langgraph and self.langgraph_service:
            try:
                # LangGraph에서는 체크포인트를 통해 세션 관리
                # 실제 삭제는 체크포인트 관리자에서 처리
                self.logger.info(f"Clearing conversation history for session: {session_id}")
            except Exception as e:
                self.logger.error(f"Failed to clear conversation history: {e}")
        else:
            # 기존 방식 (placeholder)
            pass
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 조회"""
        status = {
            "service_name": "ChatService",
            "langgraph_enabled": self.use_langgraph,
            "langgraph_service_available": self.langgraph_service is not None,
            "timestamp": time.time()
        }
        
        if self.use_langgraph and self.langgraph_service:
            try:
                langgraph_status = self.langgraph_service.get_service_status()
                status["langgraph_status"] = langgraph_status
            except Exception as e:
                status["langgraph_error"] = str(e)
        
        return status
    
    async def test_service(self, test_message: str = "테스트 질문입니다") -> Dict[str, Any]:
        """서비스 테스트"""
        try:
            result = await self.process_message(test_message)
            
            test_passed = (
                "response" in result and 
                result["response"] and 
                "processing_time" in result
            )
            
            return {
                "test_passed": test_passed,
                "test_message": test_message,
                "result": result,
                "langgraph_enabled": self.use_langgraph
            }
            
        except Exception as e:
            return {
                "test_passed": False,
                "test_message": test_message,
                "error": str(e),
                "langgraph_enabled": self.use_langgraph
            }
