# -*- coding: utf-8 -*-
"""
법률 검색 관련 Tools
기존 검색 엔진을 LangChain Tool로 캡슐화
"""

import json
import logging
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

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
        logger = logging.getLogger(__name__)
        logger.error("langchain tools를 import할 수 없습니다. StructuredTool = None으로 설정됩니다.")
        StructuredTool = None

# 기존 검색 엔진 import
try:
    from core.search.engines.hybrid_search_engine_v2 import HybridSearchEngineV2
    from core.search.handlers.search_handler import SearchHandler
    HYBRID_SEARCH_AVAILABLE = True
except ImportError:
    HYBRID_SEARCH_AVAILABLE = False
    logging.warning("HybridSearchEngineV2 not available")

logger = logging.getLogger(__name__)

# 상수 정의
DEFAULT_MAX_RESULTS = 5
DEFAULT_HYBRID_MAX_RESULTS = 10
ERROR_RESPONSE_JSON = '{"success": false, "error": "Search engine not available", "results": []}'
DEFAULT_SEARCH_TYPES = ["law", "precedent", "constitutional", "assembly_law"]

# 검색 엔진 싱글톤 인스턴스
_search_engine_instance = None

def get_search_engine():
    """검색 엔진 싱글톤 인스턴스 반환"""
    global _search_engine_instance
    
    if _search_engine_instance is not None:
        return _search_engine_instance
    
    if not HYBRID_SEARCH_AVAILABLE:
        return None
    
    # SearchHandler 우선 시도, 실패 시 HybridSearchEngineV2 사용
    for engine_class in [SearchHandler, HybridSearchEngineV2]:
        try:
            _search_engine_instance = engine_class()
            logger.info(f"{engine_class.__name__} initialized for tools")
            return _search_engine_instance
        except Exception as e:
            logger.warning(f"Failed to initialize {engine_class.__name__}: {e}")
            continue
    
    logger.error("Failed to initialize any search engine")
    return None


# ==================== 입력 스키마 정의 ====================

class SearchPrecedentInput(BaseModel):
    """판례 검색 입력"""
    query: str = Field(description="검색할 판례 관련 질문이나 키워드")
    category: Optional[str] = Field(None, description="판례 카테고리 (civil, criminal, family, administrative 등)")
    max_results: int = Field(DEFAULT_MAX_RESULTS, description="최대 검색결과 수", ge=1, le=20)


class SearchLawInput(BaseModel):
    """법령 검색 입력"""
    query: str = Field(description="검색할 법령 관련 질문이나 키워드")
    law_name: Optional[str] = Field(None, description="특정 법령명 (예: 민법, 형법)")
    article_number: Optional[str] = Field(None, description="조문 번호")
    max_results: int = Field(DEFAULT_MAX_RESULTS, description="최대 검색결과 수", ge=1, le=20)


class SearchLegalTermInput(BaseModel):
    """법률 용어 검색 입력"""
    query: str = Field(description="검색할 법률 용어나 정의")
    max_results: int = Field(DEFAULT_MAX_RESULTS, description="최대 검색결과 수", ge=1, le=20)


class HybridSearchInput(BaseModel):
    """하이브리드 검색 입력 (통합 검색)"""
    query: str = Field(description="검색할 질문이나 키워드")
    search_types: Optional[List[str]] = Field(
        None, 
        description="검색할 문서 유형 목록 (law, precedent, constitutional, assembly_law 등)"
    )
    max_results: int = Field(DEFAULT_HYBRID_MAX_RESULTS, description="최대 검색결과 수", ge=1, le=50)
    include_exact: bool = Field(True, description="정확 매칭 검색 포함 여부")
    include_semantic: bool = Field(True, description="의미 검색 포함 여부")


# ==================== 공통 헬퍼 함수 ====================

def _execute_search(
    query: str,
    search_types: List[str],
    max_results: int,
    include_exact: bool = True,
    include_semantic: bool = True,
    include_raw_result: bool = False
) -> Dict[str, Any]:
    """공통 검색 실행 로직"""
    search_engine = get_search_engine()
    if not search_engine:
        return {"success": False, "error": "Search engine not available", "results": []}
    
    raw_result = search_engine.search(
        query=query,
        search_types=search_types,
        max_results=max_results,
        include_exact=include_exact,
        include_semantic=include_semantic
    )
    
    result = {
        "success": True,
        "results": raw_result.get("results", [])[:max_results],
        "total_results": raw_result.get("total_results", 0),
        "query": query
    }
    
    if include_raw_result:
        result["_raw_result"] = raw_result
    
    return result


def _format_search_response(result: Dict[str, Any]) -> str:
    """JSON 응답 포맷팅"""
    return json.dumps(result, ensure_ascii=False, indent=2)


def _handle_search_error(func_name: str, error: Exception) -> str:
    """검색 에러 처리"""
    logger.error(f"Error in {func_name}: {error}")
    return json.dumps({"success": False, "error": str(error), "results": []}, ensure_ascii=False)


# ==================== Tool 함수 구현 ====================

def _search_precedent(
    query: str,
    category: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS
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
        # 카테고리가 있으면 쿼리에 추가
        enhanced_query = f"{category} {query}" if category else query
        
        result = _execute_search(
            query=enhanced_query,
            search_types=["precedent"],
            max_results=max_results,
            include_exact=True,
            include_semantic=True
        )
        
        return _format_search_response(result)
    except Exception as e:
        return _handle_search_error("search_precedent", e)


def _search_law(
    query: str,
    law_name: Optional[str] = None,
    article_number: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS
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
        # 법령명이나 조문 번호가 있으면 쿼리에 추가
        enhanced_query = query
        if law_name:
            enhanced_query = f"{law_name} {enhanced_query}"
        if article_number:
            enhanced_query = f"{enhanced_query} {article_number}조"
        
        result = _execute_search(
            query=enhanced_query,
            search_types=["law", "constitutional", "assembly_law"],
            max_results=max_results,
            include_exact=True,
            include_semantic=True
        )
        
        # 추가 파라미터 포함
        result["law_name"] = law_name
        result["article_number"] = article_number
        
        return _format_search_response(result)
    except Exception as e:
        return _handle_search_error("search_law", e)


def _search_legal_term(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS
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
        result = _execute_search(
            query=query,
            search_types=["law", "precedent"],
            max_results=max_results,
            include_exact=False,
            include_semantic=True
        )
        
        return _format_search_response(result)
    except Exception as e:
        return _handle_search_error("search_legal_term", e)


def _hybrid_search(
    query: str,
    search_types: Optional[List[str]] = None,
    max_results: int = DEFAULT_HYBRID_MAX_RESULTS,
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
        if search_types is None:
            search_types = DEFAULT_SEARCH_TYPES
        
        result = _execute_search(
            query=query,
            search_types=search_types,
            max_results=max_results,
            include_exact=include_exact,
            include_semantic=include_semantic,
            include_raw_result=True
        )
        
        # search_stats 추가
        if "_raw_result" in result:
            result["search_stats"] = result["_raw_result"].get("search_stats", {})
            del result["_raw_result"]
        
        return _format_search_response(result)
    except Exception as e:
        return _handle_search_error("hybrid_search", e)


# ==================== Tool 인스턴스 생성 ====================

def _create_tool(func, name: str, description: str, args_schema: BaseModel):
    """Tool 생성 헬퍼 함수"""
    if StructuredTool is None:
        return None
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema
    )


def _mock_tool(*args, **kwargs) -> str:
    """Mock Tool 함수"""
    return json.dumps({
        "success": False,
        "error": "Search engine or StructuredTool not available",
        "results": []
    }, ensure_ascii=False)


# Tool 정의
TOOL_DEFINITIONS = [
    (
        _search_precedent,
        "search_precedent",
        "판례 데이터베이스에서 관련 판례를 검색합니다. 계약 위반, 손해배상, 이혼, 상속 등 구체적인 사건과 관련된 판례를 찾을 때 사용합니다.",
        SearchPrecedentInput
    ),
    (
        _search_law,
        "search_law",
        "법령 데이터베이스에서 관련 법령을 검색합니다. 법령 조문, 법적 근거, 법적 조건 등을 찾을 때 사용합니다.",
        SearchLawInput
    ),
    (
        _search_legal_term,
        "search_legal_term",
        "법률 용어 사전에서 법률 용어나 정의를 검색합니다. 법률 용어의 의미, 정의, 해석을 찾을 때 사용합니다.",
        SearchLegalTermInput
    ),
    (
        _hybrid_search,
        "hybrid_search",
        "하이브리드 검색을 수행합니다. 정확 매칭과 의미 검색을 결합하여 법령, 판례, 해석 등을 종합적으로 검색합니다. 복잡한 질문이나 다양한 문서 타입을 검색해야 할 때 사용합니다.",
        HybridSearchInput
    ),
]

# Tool 인스턴스 생성
if HYBRID_SEARCH_AVAILABLE and StructuredTool is not None:
    search_precedent_tool = _create_tool(*TOOL_DEFINITIONS[0])
    search_law_tool = _create_tool(*TOOL_DEFINITIONS[1])
    search_legal_term_tool = _create_tool(*TOOL_DEFINITIONS[2])
    hybrid_search_tool = _create_tool(*TOOL_DEFINITIONS[3])
else:
    logger.warning("Creating mock tools due to unavailable search engine or StructuredTool")
    
    if StructuredTool is not None:
        search_precedent_tool = _create_tool(_mock_tool, "search_precedent", "[MOCK] 판례 검색 Tool (검색 엔진 미사용)", SearchPrecedentInput)
        search_law_tool = _create_tool(_mock_tool, "search_law", "[MOCK] 법령 검색 Tool (검색 엔진 미사용)", SearchLawInput)
        search_legal_term_tool = _create_tool(_mock_tool, "search_legal_term", "[MOCK] 법률 용어 검색 Tool (검색 엔진 미사용)", SearchLegalTermInput)
        hybrid_search_tool = _create_tool(_mock_tool, "hybrid_search", "[MOCK] 하이브리드 검색 Tool (검색 엔진 미사용)", HybridSearchInput)
    else:
        search_precedent_tool = None
        search_law_tool = None
        search_legal_term_tool = None
        hybrid_search_tool = None
