# -*- coding: utf-8 -*-
"""
ë²•ë¥  AI Agentë¥??„í•œ Tool System
LangChain??Tool, StructuredTool ?¨í„´ ?œìš©
"""

from typing import List
from langchain.tools import Tool, StructuredTool

# Tool ?„í¬??(ì§€??ë¡œë”©?¼ë¡œ ë³€ê²?ê°€??
try:
    from .legal_search_tools import (
        search_precedent_tool,
        search_law_tool,
        search_legal_term_tool,
        hybrid_search_tool
    )
    
    # ëª¨ë“  ?„êµ¬ë¥??ë™?¼ë¡œ ê°ì??˜ê³  ?±ë¡?˜ëŠ” ?œìŠ¤??
    LEGAL_TOOLS: List[Tool] = [
        # ê²€???„êµ¬
        search_precedent_tool,
        search_law_tool,
        search_legal_term_tool,
        hybrid_search_tool,
    ]
    
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
    LEGAL_TOOLS: List[Tool] = []
    __all__ = ["LEGAL_TOOLS"]

