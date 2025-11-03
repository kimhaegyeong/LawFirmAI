# -*- coding: utf-8 -*-
"""
법률 검색 관련 Tools
기존 검색 엔진을 LangChain Tool로 캡슐화
"""

import logging
import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# langchain.tools import (fallback 포함)
try:
    from langchain.tools import StructuredTool
except ImportError:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("langchain tools를 import할 수 없습니다. StructuredTool = None으로 설정됩니다.")
        StructuredTool = None

from pydantic import BaseModel, Field

# 기존 검색 엔진 import
try:
    from core.services.search.hybrid_search_engine_v2 import HybridSearchEngineV2
    from core.services.search.search_handler import SearchHandler
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from source.services.hybrid_search_engine import HybridSearchEngine
        HYBRID_SEARCH_AVAILABLE = True
    except ImportError:
        HYBRID_SEARCH_AVAILABLE = False
        logging.warning("HybridSearchEngine not available")

logger = logging.getLogger(__name__)

# 검색 엔진 싱글톤 인스턴스
_search_engine_instance = None

def get_search_engine():
    """검색 엔진 싱글톤 인스턴스 반환"""
    global _search_engine_instance
    if _search_engine_instance is None and HYBRID_SEARCH_AVAILABLE:
        try:
            # SearchHandler 사용 시도
            try:
                _search_engine_instance = SearchHandler()
                logger.info("SearchHandler initialized for tools")
            except:
                # Fallback: HybridSearchEngine
                try:
                    from source.services.hybrid_search_engine import HybridSearchEngine
                    _search_engine_instance = HybridSearchEngine()
                    logger.info("HybridSearchEngine initialized for tools")
                except:
                    logger.error("Failed to initialize search engine")
                    return None
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            return None
    return _search_engine_instance


# ==================== 입력 스키마 정의 ====================

class SearchPrecedentInput(BaseModel):
    """판례 검색 입력"""
    query: str = Field(description="검색할 판례 관련 질문이나 키워드")
    category: Optional[str] = Field(None, description="판례 카테고리 (civil, criminal, family, administrative 등)")
    max_results: int = Field(5, description="최대 검색결과 수", ge=1, le=20)


class SearchLawInput(BaseModel):
    """법령 검색 입력"""
    query: str = Field(description="검색할 법령 관련 질문이나 키워드")
    law_name: Optional[str] = Field(None, description="특정 법령명 (예: 민법, 형법)")
    article_number: Optional[str] = Field(None, description="조문 번호")
    max_results: int = Field(5, description="최대 검색결과 수", ge=1, le=20)


class SearchLegalTermInput(BaseModel):
    """법률 용어 검색 입력"""
    query: str = Field(description="검색할 법률 용어나 정의")
    max_results: int = Field(5, description="최대 검색결과 수", ge=1, le=20)


class HybridSearchInput(BaseModel):
    """하이브리드 검색 입력 (통합 검색)"""
    query: str = Field(description="검색할 질문이나 키워드")
    search_types: Optional[List[str]] = Field(
        None, 
        description="검색할 문서 유형 목록 (law, precedent, constitutional, assembly_law 등)"
    )
    max_results: int = Field(10, description="최대 검색결과 수", ge=1, le=50)
    include_exact: bool = Field(True, description="정확 매칭 검색 포함 여부")
    include_semantic: bool = Field(True, description="의미 검색 포함 여부")


# ==================== Tool 함수 구현 ====================

def _search_precedent(
    query: str,
    category: Optional[str] = None,
    max_results: int = 5
) -> str:
    """
    판례 데이터베이스에서 관련 판례를 검색합니다.
    
    계약 위반, 손해배상, 이혼, 상속 등 구체적인 사건과 관련된 판례를 찾을 때 사용합니다.
    
    Args:
        query: 검색할 판례 관련 질문
        category: 판례 카테고리 (선택사항)
        max_results: 최대 검색결과 수
        
    Returns:
        검색결과를 JSON 문자열로 반환 (결과 리스트와 각 결과에 text, score, source, metadata 포함)
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # 판례 검색 (search_types에 "precedent" 포함)
        search_types = ["precedent"]
        if category:
            # 카테고리를 검색 쿼리에 추가 가능
            pass
        
        result = search_engine.search(
            query=query,
            search_types=search_types,
            max_results=max_results,
            include_exact=True,
            include_semantic=True
        )
        
        # 결과를 JSON 문자열로 변환
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
    법령 데이터베이스에서 관련 법령을 검색합니다.
    
    법령 조문, 법적 근거, 법적 조건 등을 찾을 때 사용합니다.
    
    Args:
        query: 검색할 법령 관련 질문
        law_name: 특정 법령명 (선택사항)
        article_number: 조문 번호 (선택사항)
        max_results: 최대 검색결과 수
        
    Returns:
        검색결과를 JSON 문자열로 반환
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # 법령 검색 (search_types에 "law", "constitutional", "assembly_law" 포함)
        search_types = ["law", "constitutional", "assembly_law"]
        
        # 법령명이나 조문 번호가 있으면 쿼리에 추가
        enhanced_query = query
        if law_name:
            enhanced_query = f"{law_name} {query}"
        if article_number:
            enhanced_query = f"{enhanced_query} {article_number}조"
        
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
    법률 용어 사전에서 법률 용어나 정의를 검색합니다.
    
    법률 용어의 의미, 정의, 해석을 찾을 때 사용합니다.
    
    Args:
        query: 검색할 법률 용어
        max_results: 최대 검색결과 수
        
    Returns:
        검색결과를 JSON 문자열로 반환
    """
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return '{"success": false, "error": "Search engine not available", "results": []}'
        
        # 법률 용어 검색 (의미 검색 위주)
        result = search_engine.search(
            query=query,
            search_types=["law", "precedent"],  # 법령과 판례에서 용어 정의 찾기
            max_results=max_results,
            include_exact=False,  # 정확 매칭보다는 의미 검색
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
    하이브리드 검색을 수행합니다. 정확 매칭과 의미 검색을 결합하여 법령, 판례, 해석 등을 종합적으로 검색합니다.
    
    복잡한 질문이나 다양한 문서 타입을 검색해야 할 때 사용합니다.
    
    Args:
        query: 검색할 질문이나 키워드
        search_types: 검색할 문서 유형 목록 (law, precedent, constitutional, assembly_law 등)
        max_results: 최대 검색결과 수
        include_exact: 정확 매칭 검색 포함 여부
        include_semantic: 의미 검색 포함 여부
        
    Returns:
        검색결과를 JSON 문자열로 반환
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


# ==================== Tool 인스턴스 생성 ====================

if HYBRID_SEARCH_AVAILABLE and StructuredTool is not None:
    search_precedent_tool = StructuredTool.from_function(
        func=_search_precedent,
        name="search_precedent",
        description="판례 데이터베이스에서 관련 판례를 검색합니다. 계약 위반, 손해배상, 이혼, 상속 등 구체적인 사건과 관련된 판례를 찾을 때 사용합니다.",
        args_schema=SearchPrecedentInput
    )
    
    search_law_tool = StructuredTool.from_function(
        func=_search_law,
        name="search_law",
        description="법령 데이터베이스에서 관련 법령을 검색합니다. 법령 조문, 법적 근거, 법적 조건 등을 찾을 때 사용합니다.",
        args_schema=SearchLawInput
    )
    
    search_legal_term_tool = StructuredTool.from_function(
        func=_search_legal_term,
        name="search_legal_term",
        description="법률 용어 사전에서 법률 용어나 정의를 검색합니다. 법률 용어의 의미, 정의, 해석을 찾을 때 사용합니다.",
        args_schema=SearchLegalTermInput
    )
    
    hybrid_search_tool = StructuredTool.from_function(
        func=_hybrid_search,
        name="hybrid_search",
        description="하이브리드 검색을 수행합니다. 정확 매칭과 의미 검색을 결합하여 법령, 판례, 해석 등을 종합적으로 검색합니다. 복잡한 질문이나 다양한 문서 타입을 검색해야 할 때 사용합니다.",
        args_schema=HybridSearchInput
    )
else:
    # 검색 엔진이나 StructuredTool을 사용할 수 없는 경우 Mock Tool 생성
    logger.warning("Creating mock tools due to unavailable search engine or StructuredTool")
    
    def _mock_tool(*args, **kwargs) -> str:
        return '{"success": false, "error": "Search engine or StructuredTool not available", "results": []}'
    
    if StructuredTool is not None:
        search_precedent_tool = StructuredTool.from_function(
            func=_mock_tool,
            name="search_precedent",
            description="[MOCK] 판례 검색 Tool (검색 엔진 미사용)",
            args_schema=SearchPrecedentInput
        )
        
        search_law_tool = StructuredTool.from_function(
            func=_mock_tool,
            name="search_law",
            description="[MOCK] 법령 검색 Tool (검색 엔진 미사용)",
            args_schema=SearchLawInput
        )
        
        search_legal_term_tool = StructuredTool.from_function(
            func=_mock_tool,
            name="search_legal_term",
            description="[MOCK] 법률 용어 검색 Tool (검색 엔진 미사용)",
            args_schema=SearchLegalTermInput
        )
        
        hybrid_search_tool = StructuredTool.from_function(
            func=_mock_tool,
            name="hybrid_search",
            description="[MOCK] 하이브리드 검색 Tool (검색 엔진 미사용)",
            args_schema=HybridSearchInput
        )
    else:
        # StructuredTool도 없으면 None으로 설정
        search_precedent_tool = None
        search_law_tool = None
        search_legal_term_tool = None
        hybrid_search_tool = None
