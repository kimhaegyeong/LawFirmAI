"""
LangGraph State Management
상태 관리 모듈
"""

from .state_definitions import LegalWorkflowState, create_initial_legal_state
from .state_utils import (
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)

__all__ = [
    "LegalWorkflowState",
    "create_initial_legal_state",
    "MAX_DOCUMENT_CONTENT_LENGTH",
    "MAX_PROCESSING_STEPS",
    "MAX_RETRIEVED_DOCS",
    "prune_processing_steps",
    "prune_retrieved_docs",
]
