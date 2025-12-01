# -*- coding: utf-8 -*-
"""
Routing Nodes
라우팅 관련 워크플로우 노드들

참고: 현재 노드들은 legal_workflow_enhanced.py와 WorkflowRoutes에 구현되어 있습니다.
향후 필요시 이 파일로 이동할 수 있습니다.
"""

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState

__all__ = []

