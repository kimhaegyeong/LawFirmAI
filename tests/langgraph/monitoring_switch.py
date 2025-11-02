# -*- coding: utf-8 -*-
"""
모니터링 전환 유틸리티
LangSmith와 Langfuse를 번갈아가며 사용할 수 있도록 하는 유틸리티
"""

import logging
import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """모니터링 모드"""
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> 'MonitoringMode':
        """문자열에서 MonitoringMode 생성"""
        value_lower = value.lower()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        raise ValueError(f"Unknown monitoring mode: {value}")


class MonitoringSwitch:
    """모니터링 도구 전환 유틸리티"""

    @staticmethod
    @contextmanager
    def set_mode(
        mode: MonitoringMode,
        langsmith_api_key: Optional[str] = None,
        langsmith_project: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_public_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        langfuse_enabled: Optional[bool] = None
    ):
        """
        모니터링 모드 설정 (컨텍스트 매니저)

        Args:
            mode: 모니터링 모드
            langsmith_api_key: LangSmith API 키 (None이면 환경변수에서 읽음)
            langsmith_project: LangSmith 프로젝트명
            langfuse_secret_key: Langfuse Secret Key
            langfuse_public_key: Langfuse Public Key
            langfuse_host: Langfuse Host
            langfuse_enabled: Langfuse 활성화 여부

        Yields:
            Dict: 설정된 환경변수 정보
        """
        # 원본 환경변수 백업
        original_env = {}
        env_keys_to_backup = [
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGFUSE_ENABLED",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_HOST",
            "ENABLE_LANGSMITH"
        ]

        for key in env_keys_to_backup:
            original_env[key] = os.environ.get(key)

        try:
            # 모드별 환경변수 설정
            env_settings = {}

            if mode == MonitoringMode.LANGSMITH:
                env_settings = {
                    "LANGCHAIN_TRACING_V2": "true",
                    "LANGFUSE_ENABLED": "false"
                }
                if langsmith_api_key:
                    env_settings["LANGCHAIN_API_KEY"] = langsmith_api_key
                if langsmith_project:
                    env_settings["LANGCHAIN_PROJECT"] = langsmith_project

            elif mode == MonitoringMode.LANGFUSE:
                env_settings = {
                    "LANGCHAIN_TRACING_V2": "false",
                    "LANGFUSE_ENABLED": "true"
                }
                if langfuse_secret_key:
                    env_settings["LANGFUSE_SECRET_KEY"] = langfuse_secret_key
                if langfuse_public_key:
                    env_settings["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
                if langfuse_host:
                    env_settings["LANGFUSE_HOST"] = langfuse_host

            elif mode == MonitoringMode.BOTH:
                env_settings = {
                    "LANGCHAIN_TRACING_V2": "true",
                    "LANGFUSE_ENABLED": "true"
                }
                if langsmith_api_key:
                    env_settings["LANGCHAIN_API_KEY"] = langsmith_api_key
                if langsmith_project:
                    env_settings["LANGCHAIN_PROJECT"] = langsmith_project
                if langfuse_secret_key:
                    env_settings["LANGFUSE_SECRET_KEY"] = langfuse_secret_key
                if langfuse_public_key:
                    env_settings["LANGFUSE_PUBLIC_KEY"] = langfuse_public_key
                if langfuse_host:
                    env_settings["LANGFUSE_HOST"] = langfuse_host

            else:  # MonitoringMode.NONE
                env_settings = {
                    "LANGCHAIN_TRACING_V2": "false",
                    "LANGFUSE_ENABLED": "false"
                }

            # 환경변수 설정
            for key, value in env_settings.items():
                os.environ[key] = value

            # 기존 값이 없었던 경우 삭제를 위해 별도 처리
            keys_to_remove = []
            for key in env_keys_to_backup:
                if key not in env_settings and os.environ.get(key):
                    # 설정에서 제외된 키는 None으로 설정 (이전 값 유지)
                    if key in ["LANGCHAIN_TRACING_V2", "LANGFUSE_ENABLED"]:
                        # 이 키들은 명시적으로 false로 설정됨
                        pass
                    else:
                        # 다른 키들은 제거
                        keys_to_remove.append(key)

            # 불필요한 키 제거
            for key in keys_to_remove:
                if key in os.environ:
                    del os.environ[key]

            logger.info(f"모니터링 모드 설정: {mode.value}")
            logger.debug(f"환경변수 설정: {env_settings}")

            yield {
                "mode": mode.value,
                "env_settings": env_settings,
                "langsmith_enabled": mode in [MonitoringMode.LANGSMITH, MonitoringMode.BOTH],
                "langfuse_enabled": mode in [MonitoringMode.LANGFUSE, MonitoringMode.BOTH]
            }

        finally:
            # 환경변수 복원
            for key in env_keys_to_backup:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]

            logger.debug("환경변수 복원 완료")

    @staticmethod
    def get_current_mode() -> MonitoringMode:
        """현재 설정된 모니터링 모드 반환"""
        langsmith_enabled = (
            os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() in ["true", "1", "yes"]
            and bool(os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY"))
        )
        langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"

        if langsmith_enabled and langfuse_enabled:
            return MonitoringMode.BOTH
        elif langsmith_enabled:
            return MonitoringMode.LANGSMITH
        elif langfuse_enabled:
            return MonitoringMode.LANGFUSE
        else:
            return MonitoringMode.NONE

    @staticmethod
    def verify_mode(service: Any, expected_mode: MonitoringMode) -> Dict[str, Any]:
        """
        서비스의 모니터링 모드 검증

        Args:
            service: LangGraphWorkflowService 인스턴스
            expected_mode: 예상되는 모니터링 모드

        Returns:
            Dict: 검증 결과
        """
        current_env_mode = MonitoringSwitch.get_current_mode()
        result = {
            "expected": expected_mode.value,
            "actual_env": current_env_mode.value,
            "matches": current_env_mode == expected_mode,
            "service_langfuse_enabled": False,
            "warnings": []
        }

        # Langfuse 클라이언트 확인
        if hasattr(service, 'langfuse_client_service') and service.langfuse_client_service:
            if hasattr(service.langfuse_client_service, 'enabled'):
                result["service_langfuse_enabled"] = service.langfuse_client_service.enabled

        # 모드별 검증
        if expected_mode == MonitoringMode.LANGSMITH:
            if not result["matches"]:
                result["warnings"].append("LangSmith 환경변수가 제대로 설정되지 않았습니다")

        elif expected_mode == MonitoringMode.LANGFUSE:
            if not result["service_langfuse_enabled"]:
                result["warnings"].append("Langfuse 클라이언트가 활성화되지 않았습니다")
            if current_env_mode != MonitoringMode.LANGFUSE:
                result["warnings"].append("환경변수 설정이 Langfuse 모드와 일치하지 않습니다")

        elif expected_mode == MonitoringMode.BOTH:
            if not result["matches"]:
                result["warnings"].append("환경변수가 both 모드와 일치하지 않습니다")
            if not result["service_langfuse_enabled"]:
                result["warnings"].append("Langfuse 클라이언트가 활성화되지 않았습니다")

        return result

    @staticmethod
    def load_profile(profile_name: str) -> Dict[str, str]:
        """
        환경변수 프로필 파일 로드

        Args:
            profile_name: 프로필 이름 (예: 'langsmith', 'langfuse')

        Returns:
            Dict: 환경변수 딕셔너리
        """
        project_root = Path(__file__).parent.parent.parent
        profile_path = project_root / ".env.profiles" / f"{profile_name}.env"

        if not profile_path.exists():
            logger.warning(f"프로필 파일을 찾을 수 없습니다: {profile_path}")
            return {}

        env_vars = {}
        with open(profile_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value

        logger.info(f"프로필 로드 완료: {profile_name} ({len(env_vars)}개 환경변수)")
        return env_vars
