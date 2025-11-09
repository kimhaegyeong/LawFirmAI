"""
Core Agents Module
법률 AI Agent 관련 모듈
"""

# Handlers
from .handlers import (
    AnswerFormatterHandler,
    AnswerGenerator,
    ClassificationHandler,
    ContextBuilder,
    DirectAnswerHandler,
    SearchHandler,
)

# Extractors
from .extractors import (
    DocumentExtractor,
    QueryExtractor,
    ResponseExtractor,
    ReasoningExtractor,
)

# Validators
from .validators import (
    AnswerValidator,
    ContextValidator,
    SearchValidator,
)

# Parsers
from .parsers import (
    AnswerParser,
    ClassificationParser,
    DocumentParser,
    QueryParser,
    ResponseParser,
)

# Optimizers
from .optimizers import (
    PerformanceOptimizer,
    QueryOptimizer,
)

# Workflow - EnhancedLegalQuestionWorkflow only (workflow_service moved to langgraph_core.workflow)
try:
    from .legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
except ImportError:
    EnhancedLegalQuestionWorkflow = None

__all__ = [
    # Handlers
    "AnswerFormatterHandler",
    "AnswerGenerator",
    "ClassificationHandler",
    "ContextBuilder",
    "DirectAnswerHandler",
    "SearchHandler",
    # Extractors
    "DocumentExtractor",
    "QueryExtractor",
    "ResponseExtractor",
    "ReasoningExtractor",
    # Validators
    "AnswerValidator",
    "ContextValidator",
    "SearchValidator",
    # Parsers
    "AnswerParser",
    "ClassificationParser",
    "DocumentParser",
    "QueryParser",
    "ResponseParser",
    # Optimizers
    "PerformanceOptimizer",
    "QueryOptimizer",
    # Workflow
    "EnhancedLegalQuestionWorkflow",
]
