# -*- coding: utf-8 -*-
"""
Workflow Tasks
재사용 가능한 워크플로우 Task 정의
"""

from lawfirm_langgraph.core.workflow.tasks.document_summary_tasks import (
    DocumentSummaryTask,
    SummaryStrategy
)

__all__ = [
    "DocumentSummaryTask",
    "SummaryStrategy"
]

