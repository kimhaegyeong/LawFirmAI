# -*- coding: utf-8 -*-
"""
ëª¨ë‹ˆ?°ë§ ?„í™˜ ? í‹¸ë¦¬í‹°
LangSmith?€ Langfuseë¥?ë²ˆê°ˆ?„ê?ë©??¬ìš©?????ˆë„ë¡??˜ëŠ” ? í‹¸ë¦¬í‹°
"""

import logging
import os
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """ëª¨ë‹ˆ?°ë§ ëª¨ë“œ"""
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"
    BOTH = "both"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> 'MonitoringMode':
        """ë¬¸ì?´ì—??MonitoringMode ?ì„±"""
        value_lower = value.lower()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        raise ValueError(f"Unknown monitoring mode: {value}")


class MonitoringSwitch:
    """ëª¨ë‹ˆ?°ë§ ?„êµ¬ ?„í™˜ ? í‹¸ë¦¬í‹°"""

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
        ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ?¤ì • (ì»¨í…?¤íŠ¸ ë§¤ë‹ˆ?€)

        Args:
            mode: ëª¨ë‹ˆ?°ë§ ëª¨ë“œ
            langsmith_api_key: LangSmith API ??(None?´ë©´ ?˜ê²½ë³€?˜ì—???½ìŒ)
            langsmith_project: LangSmith ?„ë¡œ?íŠ¸ëª?
            langfuse_secret_key: Langfuse Secret Key
            langfuse_public_key: Langfuse Public Key
            langfuse_host: Langfuse Host
            langfuse_enabled: Langfuse ?œì„±???¬ë?

        Yields:
            Dict: ?¤ì •???˜ê²½ë³€???•ë³´
        """
        # ?ë³¸ ?˜ê²½ë³€??ë°±ì—…
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
            # ëª¨ë“œë³??˜ê²½ë³€???¤ì •
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

            # ?˜ê²½ë³€???¤ì •
            for key, value in env_settings.items():
                os.environ[key] = value

            # ê¸°ì¡´ ê°’ì´ ?†ì—ˆ??ê²½ìš° ?? œë¥??„í•´ ë³„ë„ ì²˜ë¦¬
            keys_to_remove = []
            for key in env_keys_to_backup:
                if key not in env_settings and os.environ.get(key):
                    # ?¤ì •?ì„œ ?œì™¸???¤ëŠ” None?¼ë¡œ ?¤ì • (?´ì „ ê°?? ì?)
                    if key in ["LANGCHAIN_TRACING_V2", "LANGFUSE_ENABLED"]:
                        # ???¤ë“¤?€ ëª…ì‹œ?ìœ¼ë¡?falseë¡??¤ì •??
                        pass
                    else:
                        # ?¤ë¥¸ ?¤ë“¤?€ ?œê±°
                        keys_to_remove.append(key)

            # ë¶ˆí•„?”í•œ ???œê±°
            for key in keys_to_remove:
                if key in os.environ:
                    del os.environ[key]

            logger.info(f"ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ?¤ì •: {mode.value}")
            logger.debug(f"?˜ê²½ë³€???¤ì •: {env_settings}")

            yield {
                "mode": mode.value,
                "env_settings": env_settings,
                "langsmith_enabled": mode in [MonitoringMode.LANGSMITH, MonitoringMode.BOTH],
                "langfuse_enabled": mode in [MonitoringMode.LANGFUSE, MonitoringMode.BOTH]
            }

        finally:
            # ?˜ê²½ë³€??ë³µì›
            for key in env_keys_to_backup:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                elif key in os.environ:
                    del os.environ[key]

            logger.debug("?˜ê²½ë³€??ë³µì› ?„ë£Œ")

    @staticmethod
    def get_current_mode() -> MonitoringMode:
        """?„ì¬ ?¤ì •??ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ë°˜í™˜"""
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
        ?œë¹„?¤ì˜ ëª¨ë‹ˆ?°ë§ ëª¨ë“œ ê²€ì¦?

        Args:
            service: LangGraphWorkflowService ?¸ìŠ¤?´ìŠ¤
            expected_mode: ?ˆìƒ?˜ëŠ” ëª¨ë‹ˆ?°ë§ ëª¨ë“œ

        Returns:
            Dict: ê²€ì¦?ê²°ê³¼
        """
        current_env_mode = MonitoringSwitch.get_current_mode()
        result = {
            "expected": expected_mode.value,
            "actual_env": current_env_mode.value,
            "matches": current_env_mode == expected_mode,
            "service_langfuse_enabled": False,
            "warnings": []
        }

        # Langfuse ?´ë¼?´ì–¸???•ì¸
        if hasattr(service, 'langfuse_client_service') and service.langfuse_client_service:
            if hasattr(service.langfuse_client_service, 'enabled'):
                result["service_langfuse_enabled"] = service.langfuse_client_service.enabled

        # ëª¨ë“œë³?ê²€ì¦?
        if expected_mode == MonitoringMode.LANGSMITH:
            if not result["matches"]:
                result["warnings"].append("LangSmith ?˜ê²½ë³€?˜ê? ?œë?ë¡??¤ì •?˜ì? ?Šì•˜?µë‹ˆ??)

        elif expected_mode == MonitoringMode.LANGFUSE:
            if not result["service_langfuse_enabled"]:
                result["warnings"].append("Langfuse ?´ë¼?´ì–¸?¸ê? ?œì„±?”ë˜ì§€ ?Šì•˜?µë‹ˆ??)
            if current_env_mode != MonitoringMode.LANGFUSE:
                result["warnings"].append("?˜ê²½ë³€???¤ì •??Langfuse ëª¨ë“œ?€ ?¼ì¹˜?˜ì? ?ŠìŠµ?ˆë‹¤")

        elif expected_mode == MonitoringMode.BOTH:
            if not result["matches"]:
                result["warnings"].append("?˜ê²½ë³€?˜ê? both ëª¨ë“œ?€ ?¼ì¹˜?˜ì? ?ŠìŠµ?ˆë‹¤")
            if not result["service_langfuse_enabled"]:
                result["warnings"].append("Langfuse ?´ë¼?´ì–¸?¸ê? ?œì„±?”ë˜ì§€ ?Šì•˜?µë‹ˆ??)

        return result

    @staticmethod
    def load_profile(profile_name: str) -> Dict[str, str]:
        """
        ?˜ê²½ë³€???„ë¡œ???Œì¼ ë¡œë“œ

        Args:
            profile_name: ?„ë¡œ???´ë¦„ (?? 'langsmith', 'langfuse')

        Returns:
            Dict: ?˜ê²½ë³€???•ì…”?ˆë¦¬
        """
        project_root = Path(__file__).parent.parent.parent
        profile_path = project_root / ".env.profiles" / f"{profile_name}.env"

        if not profile_path.exists():
            logger.warning(f"?„ë¡œ???Œì¼??ì°¾ì„ ???†ìŠµ?ˆë‹¤: {profile_path}")
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

        logger.info(f"?„ë¡œ??ë¡œë“œ ?„ë£Œ: {profile_name} ({len(env_vars)}ê°??˜ê²½ë³€??")
        return env_vars
