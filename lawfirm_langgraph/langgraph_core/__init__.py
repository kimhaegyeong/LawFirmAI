"""
LangGraph Core Module
LangGraph 핵심 모듈
"""

__version__ = "1.0.0"

# Workflow
from .workflow import (
    LangGraphWorkflowService,
    EnhancedLegalQuestionWorkflow,
)

# State
from .state import (
    LegalWorkflowState,
    create_initial_legal_state,
    MAX_DOCUMENT_CONTENT_LENGTH,
    MAX_PROCESSING_STEPS,
    MAX_RETRIEVED_DOCS,
    prune_processing_steps,
    prune_retrieved_docs,
)

# Nodes
from .nodes import (
    AnswerGenerationChainBuilder,
    ClassificationChainBuilder,
    DirectAnswerChainBuilder,
    DocumentAnalysisChainBuilder,
    QueryEnhancementChainBuilder,
    with_state_optimization,
    NodeIOSpec,
    NodeCategory,
    PromptBuilder,
    QueryBuilder,
)

# Processing
from .processing import (
    DocumentExtractor,
    QueryExtractor,
    ResponseExtractor,
    AnswerParser,
    ClassificationParser,
    DocumentParser,
    QueryParser,
    ReasoningExtractor,
    AnswerValidator,
    ContextValidator,
    SearchValidator,
)

__all__ = [
    "__version__",
    # Workflow
    "LangGraphWorkflowService",
    "EnhancedLegalQuestionWorkflow",
    # State
    "LegalWorkflowState",
    "create_initial_legal_state",
    "MAX_DOCUMENT_CONTENT_LENGTH",
    "MAX_PROCESSING_STEPS",
    "MAX_RETRIEVED_DOCS",
    "prune_processing_steps",
    "prune_retrieved_docs",
    # Nodes
    "AnswerGenerationChainBuilder",
    "ClassificationChainBuilder",
    "DirectAnswerChainBuilder",
    "DocumentAnalysisChainBuilder",
    "QueryEnhancementChainBuilder",
    "with_state_optimization",
    "NodeIOSpec",
    "NodeCategory",
    "PromptBuilder",
    "QueryBuilder",
    # Processing
    "DocumentExtractor",
    "QueryExtractor",
    "ResponseExtractor",
    "AnswerParser",
    "ClassificationParser",
    "DocumentParser",
    "QueryParser",
    "ReasoningExtractor",
    "AnswerValidator",
    "ContextValidator",
    "SearchValidator",
]