"""
LawFirm LangGraph Package
법률 AI 어시스턴트를 위한 LangGraph 기반 워크플로우 시스템
"""

__version__ = "1.0.0"

# LangGraph Core 모듈 export
try:
    from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
    from lawfirm_langgraph.langgraph_core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
except ImportError:
    LangGraphWorkflowService = None
    EnhancedLegalQuestionWorkflow = None

# Config export
try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig, CheckpointStorageType
except ImportError:
    LangGraphConfig = None
    CheckpointStorageType = None

# Core 모듈 export (backward compatibility)
try:
    from core.services.chat_service import ChatService
    try:
        from config.app_config import Config
    except ImportError:
        from core.utils.config import Config
    from core.data.database import DatabaseManager
    from core.data.vector_store import VectorStore
except ImportError:
    ChatService = None
    Config = None
    DatabaseManager = None
    VectorStore = None

__all__ = [
    "__version__",
    "LangGraphWorkflowService",
    "EnhancedLegalQuestionWorkflow",
    "LangGraphConfig",
    "CheckpointStorageType",
    "ChatService",
    "Config",
    "DatabaseManager",
    "VectorStore",
]

