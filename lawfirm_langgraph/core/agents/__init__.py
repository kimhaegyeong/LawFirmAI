# -*- coding: utf-8 -*-
"""
Core Agents Module
법률 AI Agent 관련 모듈

참고: 대부분의 기능이 도메인별 디렉토리로 이동했습니다.
이 모듈은 호환성을 위해 re-export만 제공합니다.
"""

# Handlers - 도메인별 경로로 re-export
try:
    from core.generation.formatters.answer_formatter import AnswerFormatterHandler
    from core.generation.generators.answer_generator import AnswerGenerator
    from core.generation.generators.context_builder import ContextBuilder
    from core.generation.generators.direct_answer_handler import DirectAnswerHandler
    from core.classification.handlers.classification_handler import ClassificationHandler
    from core.search.handlers.search_handler import SearchHandler
except ImportError:
    AnswerFormatterHandler = None
    AnswerGenerator = None
    ContextBuilder = None
    DirectAnswerHandler = None
    ClassificationHandler = None
    SearchHandler = None

# Extractors - 도메인별 경로로 re-export
try:
    from core.processing.extractors.extractors import (
        DocumentExtractor,
        QueryExtractor,
        ResponseExtractor,
    )
    from core.processing.extractors.reasoning_extractor import ReasoningExtractor
except ImportError:
    DocumentExtractor = None
    QueryExtractor = None
    ResponseExtractor = None
    ReasoningExtractor = None

# Validators - 도메인별 경로로 re-export
try:
    from core.generation.validators.quality_validators import (
        AnswerValidator,
        ContextValidator,
        SearchValidator,
    )
except ImportError:
    AnswerValidator = None
    ContextValidator = None
    SearchValidator = None

# Parsers - 도메인별 경로로 re-export
try:
    from core.processing.parsers.response_parsers import (
        AnswerParser,
        ClassificationParser,
        DocumentParser,
        QueryParser,
        ResponseParser,
    )
except ImportError:
    AnswerParser = None
    ClassificationParser = None
    DocumentParser = None
    QueryParser = None
    ResponseParser = None

# Optimizers - 도메인별 경로로 re-export
try:
    from core.agents.optimizers.performance_optimizer import PerformanceOptimizer
    from core.agents.optimizers.query_optimizer import QueryOptimizer
except ImportError:
    PerformanceOptimizer = None
    QueryOptimizer = None

# Workflow - 도메인별 경로로 re-export
try:
    from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
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
