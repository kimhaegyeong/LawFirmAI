# -*- coding: utf-8 -*-
"""
법률 AI Agent용 Tool System
LangChain의 Tool, StructuredTool 패턴 사용
"""

from typing import List
# langchain.tools 또는 langchain_core.tools 사용
try:
    from langchain.tools import Tool, StructuredTool
except ImportError:
    # langchain 패키지가 없는 경우 langchain_core 사용
    try:
        from langchain_core.tools import Tool, StructuredTool
    except ImportError:
        # 최종 fallback: langchain_community.tools
        try:
            from langchain_community.tools import Tool, StructuredTool
        except ImportError:
            # Tool 클래스가 없는 경우 경고
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("langchain tools를 import할 수 없습니다. Agentic 모드가 작동하지 않을 수 있습니다.")
            Tool = None
            StructuredTool = None

# Tool 포함 (지연로딩으로 변경 가능)
try:
    from .legal_search_tools import (
        search_precedent_tool,
        search_law_tool,
        search_legal_term_tool,
        hybrid_search_tool
    )
    
    # Tool이 None이 아닌 경우만 추가
    _tools = []
    if search_precedent_tool is not None:
        _tools.append(search_precedent_tool)
    if search_law_tool is not None:
        _tools.append(search_law_tool)
    if search_legal_term_tool is not None:
        _tools.append(search_legal_term_tool)
    if hybrid_search_tool is not None:
        _tools.append(hybrid_search_tool)
    
    LEGAL_TOOLS: List[Tool] = _tools if Tool is not None else []
    
    __all__ = [
        "LEGAL_TOOLS",
        "search_precedent_tool",
        "search_law_tool",
        "search_legal_term_tool",
        "hybrid_search_tool",
    ]
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import legal search tools: {e}")
    LEGAL_TOOLS: List[Tool] = [] if Tool is None else []
    __all__ = ["LEGAL_TOOLS"]
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Error importing legal search tools: {e}")
    LEGAL_TOOLS: List[Tool] = [] if Tool is None else []
    __all__ = ["LEGAL_TOOLS"]
