# -*- coding: utf-8 -*-
"""
워크플로우 팩토리
모니터링 모드별 워크플로우 인스턴스 생성 및 관리
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 프로젝트 루트 경로 추가
# __file__ = tests/langgraph/fixtures/workflow_factory.py
# .parent = tests/langgraph/fixtures/
# .parent.parent = tests/langgraph/
# .parent.parent.parent = tests/
# .parent.parent.parent.parent = 프로젝트 루트 (LawFirmAI/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.langgraph.monitoring_switch import MonitoringMode

logger = logging.getLogger(__name__)

# LangGraphWorkflowService를 선택적으로 import
try:
    from core.agents.workflow_service import LangGraphWorkflowService
    from infrastructure.utils.langgraph_config import LangGraphConfig
    WORKFLOW_SERVICE_AVAILABLE = True
except ImportError as import_err:
    import_error_msg = str(import_err)
    WORKFLOW_SERVICE_AVAILABLE = False
    logger.warning(f"LangGraphWorkflowService를 사용할 수 없습니다: {import_error_msg}")
    logger.warning("langgraph 패키지가 설치되지数组中거나 import 오류가 있습니다.")
    logger.warning("워크플로우 생성 기능을 사용하려면 langgraph를 설치하세요: pip install langgraph langchain")

    # Mock 클래스 (타입 힌트용)
    class LangGraphWorkflowService:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"LangGraphWorkflowService를 사용할 수 없습니다.\n"
                f"langgraph 패키지를 설치하세요: pip install langgraph langchain\n"
                f"원본 오류: {import_error_msg}"
            )

    class LangGraphConfig:
        @classmethod
        def from_env(cls):
            return cls()


class WorkflowFactory:
    """모니터링별 워크플로우 인스턴스 생성 및 캐싱"""

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
        모니터링 모드에 맞는 워크플로우 인스턴스 반환

        Args:
            mode: 모니터링 모드
            config: LangGraphConfig (None이면 환경변수에서 생성)
            force_recreate: 캐시 무시하고 재생성 여부

        Returns:
            LangGraphWorkflowService: 워크플로우 서비스 인스턴스

        Raises:
            ImportError: langgraph 패키지가 설치되지 않은 경우
        """
        if not WORKFLOW_SERVICE_AVAILABLE:
            raise ImportError(
                "LangGraphWorkflowService를 사용할 수 없습니다.\n"
                "langgraph 패키지를 설치하세요: pip install langgraph langchain"
            )

        mode_key = mode.value

        # 강제 재생성 또는 캐시에 없으면 생성
        if force_recreate or mode_key not in cls._instances:
            logger.info(f"워크플로우 인스턴스 생성: {mode_key} (force_recreate={force_recreate})")

            # Config 생성
            if config is None:
                config = LangGraphConfig.from_env()
                cls._configs[mode_key] = config
            else:
                cls._configs[mode_key] = config

            # 워크플로우 서비스 생성
            try:
                service = LangGraphWorkflowService(config=config)
                cls._instances[mode_key] = service
                logger.info(f"워크플로우 인스턴스 생성 완료: {mode_key}")
            except Exception as e:
                logger.error(f"워크플로우 인스턴스 생성 실패: {mode_key} - {e}")
                raise
        else:
            logger.debug(f"캐시된 워크플로우 인스턴스 사용: {mode_key}")

        return cls._instances[mode_key]

    @classmethod
    def clear_cache(cls, mode: Optional[MonitoringMode] = None):
        """
        캐시 초기화

        Args:
            mode: 특정 모드만 초기화 (None이면 전체 초기화)
        """
        if mode:
            mode_key = mode.value
            if mode_key in cls._instances:
                del cls._instances[mode_key]
                logger.info(f"워크플로우 캐시 삭제: {mode_key}")
            if mode_key in cls._configs:
                del cls._configs[mode_key]
        else:
            cls._instances.clear()
            cls._configs.clear()
            logger.info("모든 워크플로우 캐시 삭제")

    @classmethod
    def get_cached_modes(cls) -> list:
        """캐시된 모드 목록 반환"""
        return list(cls._instances.keys())

    @classmethod
    def has_instance(cls, mode: MonitoringMode) -> bool:
        """특정 모드의 인스턴스가 캐시에 있는지 확인"""
        return mode.value in cls._instances

    @classmethod
    def is_available(cls) -> bool:
        """워크플로우 서비스를 사용할 수 있는지 확인"""
        return WORKFLOW_SERVICE_AVAILABLE
