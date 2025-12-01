"""
LawFirm LangGraph Package
법률 AI 어시스턴트를 위한 LangGraph 기반 워크플로우 시스템

⚠️ 성능 최적화: 모듈은 필요할 때만 로드됩니다 (지연 로딩).
"""

__version__ = "1.0.0"

# 전역 로깅 초기화 (패키지 import 시 자동 설정)
# 환경 변수 LOG_LEVEL을 읽어서 루트 로거 레벨 설정
try:
    from .core.utils.logger import configure_global_logging
    configure_global_logging()  # 환경 변수 LOG_LEVEL 자동 읽기
except ImportError:
    # logger 모듈이 없으면 기본 설정
    import logging
    import os
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    log_level = log_level_map.get(log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
except Exception:
    # 로깅 초기화 실패 시 무시 (안전한 실패)
    pass

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

# DatabaseManager는 deprecated되었습니다. LegalDataConnectorV2를 사용하세요.
# from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2

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
    "VectorStore",
]

