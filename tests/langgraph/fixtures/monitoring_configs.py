# -*- coding: utf-8 -*-
"""
모니터링 설정 예시
각 모니터링 모드별 설정 예시
"""

from typing import Any, Dict

# LangSmith 전용 설정 예시
LANGSMITH_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGFUSE_ENABLED": "false",
    "description": "LangSmith만 활성화 (LangChain 네이티브 추적)"
}

# Langfuse 전용 설정 예시
LANGFUSE_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "false",
    "LANGFUSE_ENABLED": "true",
    "description": "Langfuse만 활성화 (답변 품질 추적)"
}

# 둘 다 사용 설정 예시
BOTH_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGFUSE_ENABLED": "true",
    "description": "LangSmith + Langfuse 동시 사용"
}

# 모니터링 없음 설정 예시
NONE_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "false",
    "LANGFUSE_ENABLED": "false",
    "description": "모니터링 비활성화 (최소 리소스)"
}

# 모든 설정 매핑
MONITORING_CONFIGS = {
    "langsmith": LANGSMITH_CONFIG,
    "langfuse": LANGFUSE_CONFIG,
    "both": BOTH_CONFIG,
    "none": NONE_CONFIG
}
