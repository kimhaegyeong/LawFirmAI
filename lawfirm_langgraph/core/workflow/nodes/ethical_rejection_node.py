# -*- coding: utf-8 -*-
"""
윤리적 거부 응답 노드
윤리적으로 문제되는 질의에 대한 적절한 거부 메시지를 생성
"""

import logging
from typing import Optional

from core.workflow.state.state_definitions import LegalWorkflowState
from core.agents.workflow_utils import WorkflowUtils


logger = logging.getLogger(__name__)


class EthicalRejectionNode:
    """윤리적 거부 응답 노드"""
    
    DEFAULT_REJECTION_MESSAGE = """죄송하지만, 불법 행위를 조장하거나 법적 책임을 회피하려는 의도가 있는 질문에는 답변드릴 수 없습니다.

법률 AI 어시스턴트로서 저는:
- 불법 행위 방법을 안내하거나 조장하는 질문에 답변할 수 없습니다
- 법적 책임을 회피하려는 의도가 있는 질문에 답변할 수 없습니다
- 증거 인멸, 재판 조작 등 사법 제도를 훼손하는 질문에 답변할 수 없습니다

대신 다음과 같은 질문에는 도움을 드릴 수 있습니다:
- 법률 지식에 대한 일반적인 질문
- 법적 절차에 대한 안내
- 법률 문서 해석 및 설명
- 합법적인 법률 상담

다른 법률 관련 질문이 있으시면 언제든지 물어보세요."""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        EthicalRejectionNode 초기화
        
        Args:
            logger_instance: 로거 인스턴스 (없으면 자동 생성)
        """
        self.logger = logger_instance or logger
    
    def generate_rejection_response(
        self,
        state: LegalWorkflowState,
        rejection_reason: Optional[str] = None
    ) -> LegalWorkflowState:
        """
        윤리적 거부 응답 생성
        
        Args:
            state: 워크플로우 상태
            rejection_reason: 거부 사유
        
        Returns:
            업데이트된 상태
        """
        try:
            # 거부 사유가 있으면 포함한 메시지 생성
            if rejection_reason:
                message = f"""{self.DEFAULT_REJECTION_MESSAGE}

[거부 사유]
{rejection_reason}"""
            else:
                message = self.DEFAULT_REJECTION_MESSAGE
            
            # State에 응답 저장
            WorkflowUtils.set_state_value(state, "answer", message)
            WorkflowUtils.set_state_value(state, "is_ethically_problematic", True)
            WorkflowUtils.set_state_value(state, "ethical_rejection_reason", rejection_reason or "윤리적 문제 감지")
            
            # 메타데이터 업데이트
            metadata = WorkflowUtils.get_state_value(state, "metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            import time
            from datetime import datetime
            metadata["ethical_check"] = {
                "rejected": True,
                "reason": rejection_reason,
                "rejection_timestamp": datetime.now().isoformat()
            }
            WorkflowUtils.set_state_value(state, "metadata", metadata)
            
            # 처리 단계 추가
            from core.agents.state_helpers import add_processing_step
            add_processing_step(state, "윤리적 검사: 윤리적으로 문제되는 질문이 감지되어 처리하지 않았습니다.")
            
            self.logger.warning(
                f"윤리적 거부 응답 생성: {rejection_reason or '사유 없음'}"
            )
            
        except Exception as e:
            self.logger.error(f"윤리적 거부 응답 생성 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 기본 메시지라도 설정
            WorkflowUtils.set_state_value(state, "answer", self.DEFAULT_REJECTION_MESSAGE)
            WorkflowUtils.set_state_value(state, "is_ethically_problematic", True)
        
        return state
    
    @staticmethod
    def ethical_rejection_node(state: LegalWorkflowState) -> LegalWorkflowState:
        """
        윤리적 거부 노드 (정적 메서드)
        
        Args:
            state: 워크플로우 상태
        
        Returns:
            업데이트된 상태
        """
        node = EthicalRejectionNode()
        rejection_reason = WorkflowUtils.get_state_value(state, "ethical_rejection_reason", None)
        return node.generate_rejection_response(state, rejection_reason)

