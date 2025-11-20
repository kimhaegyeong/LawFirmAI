"""
LawFirm LangGraph Package
법률 AI 어시스턴트를 위한 LangGraph 기반 워크플로우 시스템

⚠️ 성능 최적화: 모듈은 필요할 때만 로드됩니다 (지연 로딩).
"""

__version__ = "1.0.0"

# 지연 로딩 유틸리티 사용
try:
    from .core.utils.lazy_imports import lazy_import, lazy_getattr
except ImportError:
    # 폴백: 간단한 지연 로딩 구현
    _module_cache = {}
    def lazy_import(module_path: str, fallback_path: str = None, default=None, cache_key: str = None):
        cache_key = cache_key or module_path
        if cache_key in _module_cache:
            return _module_cache[cache_key]
        try:
            module = __import__(module_path, fromlist=[''])
            _module_cache[cache_key] = module
            return module
        except ImportError:
            if fallback_path:
                try:
                    module = __import__(fallback_path, fromlist=[''])
                    _module_cache[cache_key] = module
                    return module
                except ImportError:
                    pass
            _module_cache[cache_key] = default
            return default
    def lazy_getattr(module, attr_name, default=None):
        return getattr(module, attr_name, default) if module else default

# LangGraph Core 모듈 (지연 로딩)
LangGraphWorkflowService = lazy_import(
    "lawfirm_langgraph.core.workflow.workflow_service",
    "core.workflow.workflow_service",
    default=None
)
if LangGraphWorkflowService:
    LangGraphWorkflowService = lazy_getattr(LangGraphWorkflowService, "LangGraphWorkflowService")

EnhancedLegalQuestionWorkflow = lazy_import(
    "lawfirm_langgraph.core.workflow.legal_workflow_enhanced",
    "core.workflow.legal_workflow_enhanced",
    default=None
)
if EnhancedLegalQuestionWorkflow:
    EnhancedLegalQuestionWorkflow = lazy_getattr(EnhancedLegalQuestionWorkflow, "EnhancedLegalQuestionWorkflow")

# Config (지연 로딩)
_config_module = lazy_import(
    "lawfirm_langgraph.config.langgraph_config",
    "config.langgraph_config",
    default=None
)
LangGraphConfig = lazy_getattr(_config_module, "LangGraphConfig") if _config_module else None
CheckpointStorageType = lazy_getattr(_config_module, "CheckpointStorageType") if _config_module else None

# Core 모듈 (지연 로딩)
ChatService = lazy_getattr(
    lazy_import("core.services.chat_service", default=None),
    "ChatService"
)

_config_module_core = lazy_import("config.app_config", "core.utils.config", default=None)
Config = lazy_getattr(_config_module_core, "Config") if _config_module_core else None

DatabaseManager = lazy_getattr(
    lazy_import("core.data.database", default=None),
    "DatabaseManager"
)

VectorStore = lazy_getattr(
    lazy_import("core.data.vector_store", default=None),
    "VectorStore"
)

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

