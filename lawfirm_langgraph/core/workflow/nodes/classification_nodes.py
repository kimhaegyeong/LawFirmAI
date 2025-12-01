# -*- coding: utf-8 -*-
"""
Classification Nodes
분류 관련 워크플로우 노드들
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Optional

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState


logger = get_logger(__name__)


class ClassificationNodes:
    """분류 관련 노드 클래스"""
    
    def __init__(
        self,
        workflow_instance=None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        ClassificationNodes 초기화
        
        Args:
            workflow_instance: EnhancedLegalQuestionWorkflow 인스턴스 (Mixin 메서드 사용)
            logger_instance: 로거 인스턴스
        """
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
    
    def classify_query_and_complexity(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 및 복잡도 판단"""
        if self.workflow:
            return self.workflow.classify_query_and_complexity(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def classification_parallel(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """병렬 분류 (긴급도 + 멀티턴)"""
        if self.workflow:
            return self.workflow.classification_parallel(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def assess_urgency(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """긴급도 평가"""
        if self.workflow:
            return self.workflow.assess_urgency(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def resolve_multi_turn(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """멀티턴 처리"""
        if self.workflow:
            return self.workflow.resolve_multi_turn(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def route_expert(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """전문가 라우팅"""
        if self.workflow:
            return self.workflow.route_expert(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def direct_answer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """직접 답변 (간단한 질문용)"""
        if self.workflow:
            return self.workflow.direct_answer_node(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def classify_query_simple(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질의 타입만 빠르게 분류 (검색 필터링에 필수)"""
        if self.workflow:
            return self.workflow.classify_query_simple(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")
    
    def classify_complexity_after_keywords(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """키워드 확장 결과를 반영하여 복잡도 재평가"""
        if self.workflow:
            return self.workflow.classify_complexity_after_keywords(state)
        raise RuntimeError("workflow_instance가 설정되지 않았습니다")