# -*- coding: utf-8 -*-
"""
Conversation Management Module
대화 관리 및 멀티턴 처리 모듈
"""

# 선택적 import로 의존성 문제 해결
try:
    from .conversation_flow_tracker import ConversationFlowTracker
except ImportError as e:
    ConversationFlowTracker = None

try:
    from .conversation_manager import ConversationContext, ConversationTurn
except ImportError as e:
    ConversationContext = None
    ConversationTurn = None

try:
    from .conversation_quality_monitor import ConversationQualityMonitor
except ImportError as e:
    ConversationQualityMonitor = None

try:
    from .multi_turn_handler import MultiTurnQuestionHandler
except ImportError as e:
    MultiTurnQuestionHandler = None

try:
    from .contextual_memory_manager import ContextualMemoryManager
except ImportError as e:
    ContextualMemoryManager = None

try:
    from .integrated_session_manager import IntegratedSessionManager
except ImportError as e:
    IntegratedSessionManager = None

__all__ = [
    "ConversationFlowTracker",
    "ConversationContext",
    "ConversationTurn",
    "ConversationQualityMonitor",
    "MultiTurnQuestionHandler",
    "ContextualMemoryManager",
    "IntegratedSessionManager",
]

