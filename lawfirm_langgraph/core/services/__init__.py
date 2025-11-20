# -*- coding: utf-8 -*-
"""
Services Module
비즈니스 로직 서비스 모듈

⚠️ DEPRECATION WARNING:
이 모듈은 리팩토링 중입니다. 많은 서비스가 기능별 폴더로 이동되었습니다.
새로운 코드에서는 다음 경로를 사용하세요:

- Conversation: core.conversation.*
- Prompt: core.agents.prompt_builders.*
- Processing: core.processing.*
- Generation: core.generation.*
- Classification: core.classification.*
- Search: core.search.*
- Shared: core.shared.*

호환성을 위해 일부 서비스는 여전히 이 모듈에서 re-export됩니다.
"""

import warnings

# Deprecation 경고 표시
warnings.warn(
    "core.services 모듈은 리팩토링 중입니다. "
    "새로운 기능별 경로를 사용하세요. "
    "자세한 내용은 모듈 docstring을 참조하세요.",
    DeprecationWarning,
    stacklevel=2
)

# 선택적 import로 의존성 문제 해결
try:
    from .chat_service import ChatService
except ImportError as e:
    # psutil 관련 경고는 무시 (선택적 의존성)
    if "psutil" not in str(e).lower():
        # langgraph 모듈이 없는 경우는 정상 (선택적 기능)
        if "langgraph" not in str(e).lower():
            print(f"Warning: Could not import ChatService: {e}")
    ChatService = None

# RAGService removed - use HybridSearchEngine or other search services instead

try:
    from core.search.handlers.search_service import SearchService
except ImportError:
    try:
        from .search_service import SearchService
    except ImportError:
        SearchService = None

# AnalysisService removed - not implemented yet
# try:
#     from .analysis_service import AnalysisService
# except ImportError as e:
#     print(f"Warning: Could not import AnalysisService: {e}")
#     AnalysisService = None
AnalysisService = None

# TASK 3.2 하이브리드 검색 시스템 모듈들 - 도메인별 경로로 re-export
try:
    from core.search.engines.semantic_search_engine import SemanticSearchEngine
except ImportError:
    try:
        from .semantic_search_engine import SemanticSearchEngine
    except ImportError:
        SemanticSearchEngine = None

try:
    from core.search.processors.result_merger import ResultMerger, ResultRanker
except ImportError:
    try:
        from .result_merger import ResultMerger, ResultRanker
    except ImportError:
        ResultMerger = None
        ResultRanker = None

# Generation 관련 re-export
try:
    from core.generation.formatters.answer_formatter import AnswerFormatter
except ImportError:
    AnswerFormatter = None

try:
    from core.generation.generators.answer_generator import AnswerGenerator
except ImportError:
    AnswerGenerator = None

try:
    from core.generation.formatters.answer_structure_enhancer import AnswerStructureEnhancer
except ImportError:
    AnswerStructureEnhancer = None

# Classification 관련 re-export
try:
    from core.classification.classifiers.question_classifier import QuestionClassifier, QuestionType
except ImportError:
    QuestionClassifier = None
    QuestionType = None

# 호환성을 위한 re-export
try:
    from core.conversation.conversation_flow_tracker import ConversationFlowTracker
except ImportError:
    ConversationFlowTracker = None

try:
    from core.services.unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType
except ImportError:
    UnifiedPromptManager = None
    LegalDomain = None
    ModelType = None

try:
    from core.processing.integration.term_integration_system import TermIntegrator
except ImportError:
    TermIntegrator = None

try:
    from core.processing.extractors.ai_keyword_generator import AIKeywordGenerator
except ImportError:
    AIKeywordGenerator = None

try:
    from core.classification.analyzers.emotion_intent_analyzer import EmotionIntentAnalyzer
except ImportError:
    EmotionIntentAnalyzer = None

try:
    from core.generation.validators.confidence_calculator import ConfidenceCalculator
except ImportError:
    ConfidenceCalculator = None

__all__ = [
    "ChatService",
    "SearchService",
    # "AnalysisService",  # Not implemented yet
    "SemanticSearchEngine",
    "ResultMerger",
    "ResultRanker",
    # Re-exported for compatibility
    "ConversationFlowTracker",
    "UnifiedPromptManager",
    "LegalDomain",
    "ModelType",
    "TermIntegrator",
    "AIKeywordGenerator",
    "EmotionIntentAnalyzer",
    "ConfidenceCalculator",
    # Generation re-exports
    "AnswerFormatter",
    "AnswerGenerator",
    "AnswerStructureEnhancer",
    # Classification re-exports
    "QuestionClassifier",
    "QuestionType",
]
