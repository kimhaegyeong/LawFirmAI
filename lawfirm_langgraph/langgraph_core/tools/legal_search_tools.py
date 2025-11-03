# -*- coding: utf-8 -*-
"""
법률 검??관??Tools
기존 검???�진??LangChain Tool�?캡슐??
"""

import logging
import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 기존 검???�진 import (core ?�더???�시�??��?, 추후 ??�� ?�정)
try:
    from source.services.hybrid_search_engine import HybridSearchEngine
    from source.services.question_classifier import QuestionType
    HYBRID_SEARCH_AVAILABLE = True
except ImportError as e:
    HYBRID_SEARCH_AVAILABLE = False
    logging.warning(f"HybridSearchEngine not available: {e}")

logger = logging.getLogger(__name__)

# 검???�진 ?�스?�스 (?��????�턴)
_search_engine_instance = None

def get_search_engine():
    """검???�진 ?��????�스?�스 반환"""
    global _search_engine_instance
    if _search_engine_instance is None and HYBRID_SEARCH_AVAILABLE:
        try:
            _search_engine_instance = HybridSearchEngine()
            logger.info("HybridSearchEngine initialized for tools")
        except Exception as e:
            logger.error(f"Failed to initialize HybridSearchEngine: {e}")
            return None
    return _search_engine_instance


# ==================== ?�력 ?�키�??�의 ====================

class SearchPrecedentInput(BaseModel):
    """?��? 검???�력"""
    query: str = Field(description="검?�할 ?��? 관??질문 ?�는 ?�워??)
    category: Optional[str] = Field(None, description="?��? 카테고리 (civil, criminal, family, administrative ??")
    max_results: int = Field(5, description="최�? 검??결과 ??, ge=1, le=20)


class SearchLawInput(BaseModel):
    """법령 검???�력"""
    query: str = Field(description="검?�할 법령 관??질문 ?�는 ?�워??)
    law_name: Optional[str] = Field(None, description="?�정 법령�?(?? 민법, ?�법)")
    article_number: Optional[str] = Field(None, description="조문 번호")
    max_results: int = Field(5, description="최�? 검??결과 ??, ge=1, le=20)


class SearchLegalTermInput(BaseModel):
    """법률 ?�어 검???�력"""
    query: str = Field(description="검?�할 법률 ?�어 ?�는 ?�의")
    max_results: int = Field(5, description="최�? 검??결과 ??, ge=1, le=20)


class HybridSearchInput(BaseModel):
    """?�이브리??검???�력 (?�합 검??"""
    query: str = Field(description="검?�할 질문 ?�는 ?�워??)
    search_types: Optional[List[str]] = Field(
        None, 
        description="검?�할 문서 ?�??목록 (law, precedent, constitutional, assembly_law ??"
    )
    max_results: int = Field(10, description="최�? 검??결과 ??, ge=1, le=50)
    include_exact: bool = Field(True, description="?�확??매칭 검???�함 ?��?")
    include_semantic: bool = Field(True, description="?��???검???�함 ?��?")


# ==================== Tool ?�수 구현 ====================

def _search_precedent(
    query: str,
    category: Optional[str] = None,
    max_results: int = 5
) -> str:
    """
    ?��? ?�이?�베?�스?�서 관???��?�?검?�합?�다.
    
    계약 ?�반, ?�해배상, ?�혼, ?�속 ??구체?�인 ?�건�?관?�된 ?��?�?찾을 ???�용?�니??
    
    Args:
        query: 검?�할 ?��? 관??질문
        category: ?��? 카테고리 (?�택?�항)
        max_results: 최�? 검??결과 ??
        
    Returns:
        검??결과�?JSON 문자?�로 반환 (결과 리스?? �?결과??text, score, source, metadata ?�함)
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # ?��? 검??(search_types??"precedent" ?�함)
        search_types = ["precedent"]
        if category:
            # 카테고리�?검?��? ?�후 ?�장 가??
            pass
        
        result = search_engine.search(
            query=query,
            search_types=search_types,
            max_results=max_results,
            include_exact=True,
            include_semantic=True
        )
        
        # 결과�?JSON 문자?�로 변??
        import json
        return json.dumps({
            "success": True,
            "results": result.get("results", [])[:max_results],
            "total_results": result.get("total_results", 0),
            "query": query
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_precedent: {e}")
        return f'{{"success": false, "error": "{str(e)}", "results": []}}'


def _search_law(
    query: str,
    law_name: Optional[str] = None,
    article_number: Optional[str] = None,
    max_results: int = 5
) -> str:
    """
    법령 ?�이?�베?�스?�서 관??법령??검?�합?�다.
    
    법령 조문, 법적 근거, 법적 ?�건 ?�을 찾을 ???�용?�니??
    
    Args:
        query: 검?�할 법령 관??질문
        law_name: ?�정 법령�?(?�택?�항)
        article_number: 조문 번호 (?�택?�항)
        max_results: 최�? 검??결과 ??
        
    Returns:
        검??결과�?JSON 문자?�로 반환
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # 법령 검??(search_types??"law", "constitutional", "assembly_law" ?�함)
        search_types = ["law", "constitutional", "assembly_law"]
        
        # 법령명이??조문 번호가 ?�으�?쿼리??추�?
        enhanced_query = query
        if law_name:
            enhanced_query = f"{law_name} {query}"
        if article_number:
            enhanced_query = f"{enhanced_query} ??article_number}�?
        
        result = search_engine.search(
            query=enhanced_query,
            search_types=search_types,
            max_results=max_results,
            include_exact=True,
            include_semantic=True
        )
        
        import json
        return json.dumps({
            "success": True,
            "results": result.get("results", [])[:max_results],
            "total_results": result.get("total_results", 0),
            "query": query,
            "law_name": law_name,
            "article_number": article_number
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_law: {e}")
        return f'{{"success": false, "error": "{str(e)}", "results": []}}'


def _search_legal_term(
    query: str,
    max_results: int = 5
) -> str:
    """
    법률 ?�어 ?�전?�서 법률 ?�어???�의�?검?�합?�다.
    
    법률 ?�어???��?, ?�의, ?�석??찾을 ???�용?�니??
    
    Args:
        query: 검?�할 법률 ?�어
        max_results: 최�? 검??결과 ??
        
    Returns:
        검??결과�?JSON 문자?�로 반환
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # 법률 ?�어 검??(?��???검???�주)
        result = search_engine.search(
            query=query,
            search_types=["law", "precedent"],  # 법령�??��??�서 ?�어 ?�의 찾기
            max_results=max_results,
            include_exact=False,  # ?�확 매칭보다???��???검??
            include_semantic=True
        )
        
        import json
        return json.dumps({
            "success": True,
            "results": result.get("results", [])[:max_results],
            "total_results": result.get("total_results", 0),
            "query": query
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error in search_legal_term: {e}")
        return f'{{"success": false, "error": "{str(e)}", "results": []}}'


def _hybrid_search(
    query: str,
    search_types: Optional[List[str]] = None,
    max_results: int = 10,
    include_exact: bool = True,
    include_semantic: bool = True
) -> str:
    """
    ?�이브리??검?�을 ?�행?�니?? ?�확??매칭�??��???검?�을 결합?�여 법령, ?��?, ?�석 ?�을 종합?�으�?검?�합?�다.
    
    복잡??질문?�나 ?�양??문서 ?�?�을 검?�해???????�용?�니??
    
    Args:
        query: 검?�할 질문 ?�는 ?�워??
        search_types: 검?�할 문서 ?�??목록 (law, precedent, constitutional, assembly_law ??
        max_results: 최�? 검??결과 ??
        include_exact: ?�확??매칭 검???�함 ?��?
        include_semantic: ?��???검???�함 ?��?
        
    Returns:
        검??결과�?JSON 문자?�로 반환
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        if search_types is None:
            search_types = ["law", "precedent", "constitutional", "assembly_law"]
        
        result = search_engine.search(
            query=query,
            search_types=search_types,
            max_results=max_results,
            include_exact=include_exact,
            include_semantic=include_semantic
        )
        
        import json
        return json.dumps({
            "success": True,
            "results": result.get("results", [])[:max_results],
            "total_results": result.get("total_results", 0),
            "query": query,
            "search_stats": result.get("search_stats", {})
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error in hybrid_search: {e}")
        return f'{{"success": false, "error": "{str(e)}", "results": []}}'


# ==================== Tool ?�스?�스 ?�성 ====================

if HYBRID_SEARCH_AVAILABLE:
    search_precedent_tool = StructuredTool.from_function(
        func=_search_precedent,
        name="search_precedent",
        description="?��? ?�이?�베?�스?�서 관???��?�?검?�합?�다. 계약 ?�반, ?�해배상, ?�혼, ?�속 ??구체?�인 ?�건�?관?�된 ?��?�?찾을 ???�용?�니??",
        args_schema=SearchPrecedentInput
    )
    
    search_law_tool = StructuredTool.from_function(
        func=_search_law,
        name="search_law",
        description="법령 ?�이?�베?�스?�서 관??법령??검?�합?�다. 법령 조문, 법적 근거, 법적 ?�건 ?�을 찾을 ???�용?�니??",
        args_schema=SearchLawInput
    )
    
    search_legal_term_tool = StructuredTool.from_function(
        func=_search_legal_term,
        name="search_legal_term",
        description="법률 ?�어 ?�전?�서 법률 ?�어???�의�?검?�합?�다. 법률 ?�어???��?, ?�의, ?�석??찾을 ???�용?�니??",
        args_schema=SearchLegalTermInput
    )
    
    hybrid_search_tool = StructuredTool.from_function(
        func=_hybrid_search,
        name="hybrid_search",
        description="?�이브리??검?�을 ?�행?�니?? ?�확??매칭�??��???검?�을 결합?�여 법령, ?��?, ?�석 ?�을 종합?�으�?검?�합?�다. 복잡??질문?�나 ?�양??문서 ?�?�을 검?�해???????�용?�니??",
        args_schema=HybridSearchInput
    )
else:
    # 검???�진???�을 경우 Mock Tool ?�성
    logger.warning("Creating mock tools due to unavailable search engine")
    
    def _mock_tool(*args, **kwargs) -> str:
        return '{"success": false, "error": "Search engine not available", "results": []}'
    
    search_precedent_tool = StructuredTool.from_function(
        func=_mock_tool,
        name="search_precedent",
        description="[MOCK] ?��? 검??Tool (검???�진 미사??",
        args_schema=SearchPrecedentInput
    )
    
    search_law_tool = StructuredTool.from_function(
        func=_mock_tool,
        name="search_law",
        description="[MOCK] 법령 검??Tool (검???�진 미사??",
        args_schema=SearchLawInput
    )
    
    search_legal_term_tool = StructuredTool.from_function(
        func=_mock_tool,
        name="search_legal_term",
        description="[MOCK] 법률 ?�어 검??Tool (검???�진 미사??",
        args_schema=SearchLegalTermInput
    )
    
    hybrid_search_tool = StructuredTool.from_function(
        func=_mock_tool,
        name="hybrid_search",
        description="[MOCK] ?�이브리??검??Tool (검???�진 미사??",
        args_schema=HybridSearchInput
    )

