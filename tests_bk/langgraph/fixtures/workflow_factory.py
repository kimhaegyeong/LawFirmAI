# -*- coding: utf-8 -*-
"""
?�크?�로???�토�?
모니?�링 모드�??�크?�로???�스?�스 ?�성 �?관�?
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ?�로?�트 루트 경로 추�?
# __file__ = tests/langgraph/fixtures/workflow_factory.py
# .parent = tests/langgraph/fixtures/
# .parent.parent = tests/langgraph/
# .parent.parent.parent = tests/
# .parent.parent.parent.parent = ?�로?�트 루트 (LawFirmAI/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringMode

logger = logging.getLogger(__name__)

# LangGraphWorkflowService�??�택?�으�?import
try:
    from source.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig
    WORKFLOW_SERVICE_AVAILABLE = True
except ImportError as import_err:
    import_error_msg = str(import_err)
    WORKFLOW_SERVICE_AVAILABLE = False
    logger.warning(f"LangGraphWorkflowService�??�용?????�습?�다: {import_error_msg}")
    logger.warning("langgraph ?�키지가 ?�치?��??�组�?��??import ?�류가 ?�습?�다.")
    logger.warning("?�크?�로???�성 기능???�용?�려�?langgraph�??�치?�세?? pip install langgraph langchain")

    # Mock ?�래??(?�???�트??
    class LangGraphWorkflowService:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"LangGraphWorkflowService�??�용?????�습?�다.\n"
                f"langgraph ?�키지�??�치?�세?? pip install langgraph langchain\n"
                f"?�본 ?�류: {import_error_msg}"
            )

    class LangGraphConfig:
        @classmethod
        def from_env(cls):
            return cls()


class WorkflowFactory:
    """모니?�링�??�크?�로???�스?�스 ?�성 �?캐싱"""

    _instances: Dict[str, Any] = {}
    _configs: Dict[str, Any] = {}

    @classmethod
    def get_workflow(
        cls,
        mode: MonitoringMode,
        config: Optional[Any] = None,
        force_recreate: bool = False
    ) -> Any:
        """
        모니?�링 모드??맞는 ?�크?�로???�스?�스 반환

        Args:
            mode: 모니?�링 모드
            config: LangGraphConfig (None?�면 ?�경변?�에???�성)
            force_recreate: 캐시 무시?�고 ?�생???��?

        Returns:
            LangGraphWorkflowService: ?�크?�로???�비???�스?�스

        Raises:
            ImportError: langgraph ?�키지가 ?�치?��? ?��? 경우
        """
        if not WORKFLOW_SERVICE_AVAILABLE:
            raise ImportError(
                "LangGraphWorkflowService�??�용?????�습?�다.\n"
                "langgraph ?�키지�??�치?�세?? pip install langgraph langchain"
            )

        mode_key = mode.value

        # 강제 ?�생???�는 캐시???�으�??�성
        if force_recreate or mode_key not in cls._instances:
            logger.info(f"?�크?�로???�스?�스 ?�성: {mode_key} (force_recreate={force_recreate})")

            # Config ?�성
            if config is None:
                config = LangGraphConfig.from_env()
                cls._configs[mode_key] = config
            else:
                cls._configs[mode_key] = config

            # ?�크?�로???�비???�성
            try:
                service = LangGraphWorkflowService(config=config)
                cls._instances[mode_key] = service
                logger.info(f"?�크?�로???�스?�스 ?�성 ?�료: {mode_key}")
            except Exception as e:
                logger.error(f"?�크?�로???�스?�스 ?�성 ?�패: {mode_key} - {e}")
                raise
        else:
            logger.debug(f"캐시???�크?�로???�스?�스 ?�용: {mode_key}")

        return cls._instances[mode_key]

    @classmethod
    def clear_cache(cls, mode: Optional[MonitoringMode] = None):
        """
        캐시 초기??

        Args:
            mode: ?�정 모드�?초기??(None?�면 ?�체 초기??
        """
        if mode:
            mode_key = mode.value
            if mode_key in cls._instances:
                del cls._instances[mode_key]
                logger.info(f"?�크?�로??캐시 ??��: {mode_key}")
            if mode_key in cls._configs:
                del cls._configs[mode_key]
        else:
            cls._instances.clear()
            cls._configs.clear()
            logger.info("모든 ?�크?�로??캐시 ??��")

    @classmethod
    def get_cached_modes(cls) -> list:
        """캐시??모드 목록 반환"""
        return list(cls._instances.keys())

    @classmethod
    def has_instance(cls, mode: MonitoringMode) -> bool:
        """?�정 모드???�스?�스가 캐시???�는지 ?�인"""
        return mode.value in cls._instances

    @classmethod
    def is_available(cls) -> bool:
        """?�크?�로???�비?��? ?�용?????�는지 ?�인"""
        return WORKFLOW_SERVICE_AVAILABLE
