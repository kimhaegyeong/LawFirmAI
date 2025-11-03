# -*- coding: utf-8 -*-
"""
모니?�링 ?�정 ?�시
�?모니?�링 모드�??�정 ?�시
"""

from typing import Any, Dict

# LangSmith ?�용 ?�정 ?�시
LANGSMITH_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGFUSE_ENABLED": "false",
    "description": "LangSmith�??�성??(LangChain ?�이?�브 추적)"
}

# Langfuse ?�용 ?�정 ?�시
LANGFUSE_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "false",
    "LANGFUSE_ENABLED": "true",
    "description": "Langfuse�??�성??(?��? ?�질 추적)"
}

# ?????�용 ?�정 ?�시
BOTH_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "true",
    "LANGFUSE_ENABLED": "true",
    "description": "LangSmith + Langfuse ?�시 ?�용"
}

# 모니?�링 ?�음 ?�정 ?�시
NONE_CONFIG: Dict[str, Any] = {
    "LANGCHAIN_TRACING_V2": "false",
    "LANGFUSE_ENABLED": "false",
    "description": "모니?�링 비활?�화 (최소 리소??"
}

# 모든 ?�정 매핑
MONITORING_CONFIGS = {
    "langsmith": LANGSMITH_CONFIG,
    "langfuse": LANGFUSE_CONFIG,
    "both": BOTH_CONFIG,
    "none": NONE_CONFIG
}
