# -*- coding: utf-8 -*-
"""
LangGraph Workflow Service
워크플로우 서비스 통합 클래스
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...utils.langgraph_config import LangGraphConfig
from .legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

# Langfuse 클라이언트 통합
try:
    from langfuse import Langfuse, trace

    from ...services.langfuse_client import LangfuseClient
    LANGFUSE_CLIENT_AVAILABLE = True
    LANGFUSE_TRACE_AVAILABLE = True
except ImportError:
    LANGFUSE_CLIENT_AVAILABLE = False
    LANGFUSE_TRACE_AVAILABLE = False

logger = logging.getLogger(__name__)

if not LANGFUSE_CLIENT_AVAILABLE:
    logger.warning("LangfuseClient not available for LangGraph workflow tracking")

# from .checkpoint_manager import CheckpointManager
from .state_definitions import create_initial_legal_state


class LangGraphWorkflowService:
    """LangGraph 워크플로우 서비스"""

    def __init__(self, config: Optional[LangGraphConfig] = None):
        """
        워크플로우 서비스 초기화

        Args:
            config: LangGraph 설정 객체
        """
        self.config = config or LangGraphConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 초기화 (체크포인트 제거)
        # self.checkpoint_manager = CheckpointManager(self.config.checkpoint_db_path)
        self.checkpoint_manager = None  # Checkpoint manager is disabled
        self.legal_workflow = EnhancedLegalQuestionWorkflow(self.config)

        # 워크플로우 컴파일
        self.app = self.legal_workflow.graph.compile()
        self.logger.info("워크플로우가 체크포인트 없이 컴파일되었습니다")

        if self.app is None:
            self.logger.error("Failed to compile workflow")
            raise RuntimeError("워크플로우 컴파일에 실패했습니다")

        # LangfuseClient 초기화 (답변 품질 추적)
        self.langfuse_client_service = None
        if LANGFUSE_CLIENT_AVAILABLE and self.config.langfuse_enabled:
            try:
                from ...services.langfuse_client import LangfuseClient
                self.langfuse_client_service = LangfuseClient(self.config)
                if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                    self.logger.info("LangfuseClient initialized for answer quality tracking")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LangfuseClient: {e}")
                self.langfuse_client_service = None

        self.logger.info("LangGraphWorkflowService initialized successfully")

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        enable_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        질문 처리

        Args:
            query: 사용자 질문
            session_id: 세션 ID (없으면 자동 생성)
            enable_checkpoint: 체크포인트 사용 여부

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            start_time = time.time()

            # 세션 ID 생성
            if not session_id:
                session_id = str(uuid.uuid4())

            self.logger.info(f"Processing query: {query[:100]}... (session: {session_id})")

            # 초기 상태 설정
            initial_state = create_initial_legal_state(query, session_id)

            # 워크플로우 실행 설정 (체크포인트 비활성화)
            config = {}
            # if enable_checkpoint:
            #     config = {"configurable": {"thread_id": session_id}}

            # 워크플로우 실행
            if self.app:
                result = await self.app.ainvoke(initial_state, config)
            else:
                raise RuntimeError("워크플로우가 컴파일되지 않았습니다")

            # 처리 시간 계산
            processing_time = time.time() - start_time

            # 결과 포맷팅
            response = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "legal_references": result.get("legal_references", []),
                "processing_steps": result.get("processing_steps", []),
                "session_id": session_id,
                "processing_time": processing_time,
                "query_type": result.get("query_type", ""),
                "metadata": result.get("metadata", {}),
                "errors": result.get("errors", [])
            }

            # Langfuse에 답변 품질 추적
            if self.langfuse_client_service and self.langfuse_client_service.is_enabled():
                self._track_answer_quality(query, response, processing_time)

            self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response

        except Exception as e:
            error_msg = f"질문 처리 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)

            return {
                "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다.",
                "sources": [],
                "confidence": 0.0,
                "legal_references": [],
                "processing_steps": [error_msg],
                "session_id": session_id or str(uuid.uuid4()),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0.0,
                "query_type": "error",
                "metadata": {"error": str(e)},
                "errors": [error_msg]
            }

    async def resume_session(self, session_id: str, query: str) -> Dict[str, Any]:
        """
        세션 재개

        Args:
            session_id: 세션 ID
            query: 새로운 질문

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            self.logger.info(f"Resuming session: {session_id}")

            # 체크포인트에서 이전 상태 확인 (checkpoint_manager가 있는 경우)
            if self.checkpoint_manager:
                checkpoints = self.checkpoint_manager.list_checkpoints(session_id)

                if checkpoints:
                    self.logger.info(f"Found {len(checkpoints)} checkpoints for session {session_id}")
                    # 이전 상태에서 새로운 질문으로 계속
                    return await self.process_query(query, session_id, enable_checkpoint=True)
                else:
                    self.logger.info(f"No checkpoints found for session {session_id}, starting new session")
                    # 새로운 세션으로 시작
                    return await self.process_query(query, session_id, enable_checkpoint=True)
            else:
                # checkpoint_manager가 없으면 일반 프로세스
                return await self.process_query(query, session_id, enable_checkpoint=False)

        except Exception as e:
            error_msg = f"세션 재개 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)

            # 새로운 세션으로 폴백
            return await self.process_query(query, session_id, enable_checkpoint=False)

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        세션 정보 조회

        Args:
            session_id: 세션 ID

        Returns:
            Dict[str, Any]: 세션 정보
        """
        try:
            if self.checkpoint_manager:
                checkpoints = self.checkpoint_manager.list_checkpoints(session_id)

                return {
                    "session_id": session_id,
                    "checkpoint_count": len(checkpoints),
                    "latest_checkpoint": checkpoints[-1] if checkpoints else None,
                    "has_checkpoints": len(checkpoints) > 0
                }
            else:
                # checkpoint_manager가 없으면 기본 정보만 반환
                return {
                    "session_id": session_id,
                    "checkpoint_count": 0,
                    "has_checkpoints": False,
                    "note": "Checkpoint manager is disabled"
                }

        except Exception as e:
            self.logger.error(f"Failed to get session info: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "checkpoint_count": 0,
                "has_checkpoints": False
            }

    def cleanup_old_sessions(self, ttl_hours: int = 24) -> int:
        """
        오래된 세션 정리

        Args:
            ttl_hours: 유지 시간 (시간)

        Returns:
            int: 정리된 체크포인트 수
        """
        try:
            if self.checkpoint_manager:
                cleaned_count = self.checkpoint_manager.cleanup_old_checkpoints(ttl_hours)
                self.logger.info(f"Cleaned up {cleaned_count} old checkpoints")
                return cleaned_count
            else:
                self.logger.info("Checkpoint manager is disabled")
                return 0

        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0

    def get_service_status(self) -> Dict[str, Any]:
        """
        서비스 상태 조회

        Returns:
            Dict[str, Any]: 서비스 상태 정보
        """
        try:
            if self.checkpoint_manager:
                db_info = self.checkpoint_manager.get_database_info()
            else:
                db_info = {"note": "Checkpoint manager is disabled"}

            return {
                "service_name": "LangGraphWorkflowService",
                "status": "running",
                "config": self.config.to_dict(),
                "database_info": db_info,
                "workflow_compiled": self.app is not None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get service status: {e}")
            return {
                "service_name": "LangGraphWorkflowService",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def validate_config(self) -> List[str]:
        """
        설정 유효성 검사

        Returns:
            List[str]: 오류 메시지 목록
        """
        return self.config.validate()

    async def test_workflow(self, test_query: str = "계약서 작성 시 주의사항은?") -> Dict[str, Any]:
        """
        워크플로우 테스트

        Args:
            test_query: 테스트 질문

        Returns:
            Dict[str, Any]: 테스트 결과
        """
        try:
            self.logger.info(f"Testing workflow with query: {test_query}")

            result = await self.process_query(test_query, enable_checkpoint=False)

            # 테스트 결과 검증
            test_passed = (
                "answer" in result and
                result["answer"] and
                len(result["processing_steps"]) > 0 and
                len(result["errors"]) == 0
            )

            return {
                "test_passed": test_passed,
                "test_query": test_query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Workflow test failed: {e}")
            return {
                "test_passed": False,
                "test_query": test_query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _track_answer_quality(self, query: str, response: Dict[str, Any], processing_time: float):
        """
        Langfuse에 답변 품질 추적

        Args:
            query: 사용자 질문
            response: 응답 결과
            processing_time: 처리 시간
        """
        if not self.langfuse_client_service or not self.langfuse_client_service.is_enabled():
            return

        try:
            # 답변 품질 점수 계산 (종합 점수)
            quality_score = self._calculate_quality_score(response)

            # 답변 품질 메트릭 추적
            trace_id = self.langfuse_client_service.track_answer_quality_metrics(
                query=query,
                answer=response.get("answer", ""),
                confidence=response.get("confidence", 0.0),
                sources_count=len(response.get("sources", [])),
                legal_refs_count=len(response.get("legal_references", [])),
                processing_time=processing_time,
                has_errors=len(response.get("errors", [])) > 0,
                overall_quality=quality_score
            )

            self.logger.info(f"Answer quality tracked in Langfuse: quality_score={quality_score:.2f}, trace_id={trace_id}")

        except Exception as e:
            self.logger.error(f"Failed to track answer quality in Langfuse: {e}")

    def _calculate_quality_score(self, response: Dict[str, Any]) -> float:
        """
        답변 품질 점수 계산

        Args:
            response: 응답 결과

        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        max_score = 0.0

        # 답변 길이 (20점)
        answer = response.get("answer", "")
        if len(answer) >= 50:
            score += 10
        if len(answer) >= 100:
            score += 10
        max_score += 20

        # 신뢰도 (30점)
        confidence = response.get("confidence", 0.0)
        score += confidence * 30
        max_score += 30

        # 소스 제공 (20점)
        sources_count = len(response.get("sources", []))
        if sources_count > 0:
            score += min(20, sources_count * 5)
        max_score += 20

        # 법률 참조 (20점)
        legal_refs_count = len(response.get("legal_references", []))
        if legal_refs_count > 0:
            score += min(20, legal_refs_count * 10)
        max_score += 20

        # 에러 없음 (10점)
        errors_count = len(response.get("errors", []))
        if errors_count == 0:
            score += 10
        max_score += 10

        # 정규화된 점수
        quality_score = score / max_score if max_score > 0 else 0.0
        return round(quality_score, 2)
