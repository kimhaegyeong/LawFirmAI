"""
검색 API 엔드포인트
하이브리드 검색 시스템을 위한 REST API 엔드포인트
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from source.services.hybrid_search_engine_v2 import HybridSearchEngineV2
from source.utils.config import Config

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api/search", tags=["search"])

# 하이브리드 검색 엔진 인스턴스
search_engine = None

def get_search_engine() -> HybridSearchEngineV2:
    """검색 엔진 인스턴스 반환 (lawfirm_v2_faiss.index 사용)"""
    global search_engine
    if search_engine is None:
        config = Config()
        db_path = config.database_path
        search_engine = HybridSearchEngineV2(db_path=db_path)
    return search_engine

# Pydantic 모델 정의
class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리", min_length=1, max_length=500)
    search_types: Optional[List[str]] = Field(
        default=["law", "precedent", "constitutional"],
        description="검색할 문서 타입"
    )
    max_results: Optional[int] = Field(
        default=20,
        description="최대 결과 수",
        ge=1,
        le=100
    )
    include_exact: Optional[bool] = Field(default=True, description="정확한 매칭 검색 포함")
    include_semantic: Optional[bool] = Field(default=True, description="의미적 검색 포함")

class SearchResponse(BaseModel):
    """검색 응답 모델"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_stats: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class SimilarDocumentsRequest(BaseModel):
    """유사 문서 검색 요청 모델"""
    doc_id: str = Field(..., description="문서 ID")
    doc_type: Optional[str] = Field(default=None, description="문서 타입")
    max_results: Optional[int] = Field(default=5, description="최대 결과 수", ge=1, le=20)

class IndexBuildRequest(BaseModel):
    """인덱스 구축 요청 모델"""
    documents: List[Dict[str, Any]] = Field(..., description="구축할 문서 목록")
    force_rebuild: Optional[bool] = Field(default=False, description="강제 재구축")

# API 엔드포인트
@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """하이브리드 검색 실행"""
    try:
        logger.info(f"Search request received: {request.query}")

        search_engine = get_search_engine()

        result = search_engine.search(
            query=request.query,
            search_types=request.search_types,
            max_results=request.max_results,
            include_exact=request.include_exact,
            include_semantic=request.include_semantic
        )

        return SearchResponse(
            query=request.query,
            results=result.get("results", []),
            total_results=result.get("total", 0),
            search_stats={
                "exact_count": result.get("exact_count", 0),
                "semantic_count": result.get("semantic_count", 0)
            },
            success=True
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/laws")
async def search_laws(
    query: str = Query(..., description="검색 쿼리"),
    max_results: int = Query(default=20, description="최대 결과 수", ge=1, le=50)
):
    """법령만 검색"""
    try:
        search_engine = get_search_engine()
        result = search_engine.search(
            query=query,
            search_types=["law"],
            max_results=max_results
        )

        return {
            "query": query,
            "results": result.get("results", []),
            "total_results": result.get("total", 0),
            "search_type": "laws_only"
        }

    except Exception as e:
        logger.error(f"Laws search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/precedents")
async def search_precedents(
    query: str = Query(..., description="검색 쿼리"),
    max_results: int = Query(default=20, description="최대 결과 수", ge=1, le=50)
):
    """판례만 검색"""
    try:
        search_engine = get_search_engine()
        result = search_engine.search(
            query=query,
            search_types=["precedent"],
            max_results=max_results
        )

        return {
            "query": query,
            "results": result.get("results", []),
            "total_results": result.get("total", 0),
            "search_type": "precedents_only"
        }

    except Exception as e:
        logger.error(f"Precedents search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/constitutional")
async def search_constitutional(
    query: str = Query(..., description="검색 쿼리"),
    max_results: int = Query(default=20, description="최대 결과 수", ge=1, le=50)
):
    """헌재결정례만 검색"""
    try:
        search_engine = get_search_engine()
        result = search_engine.search(
            query=query,
            search_types=["decision"],
            max_results=max_results
        )

        return {
            "query": query,
            "results": result.get("results", []),
            "total_results": result.get("total", 0),
            "search_type": "constitutional_only"
        }

    except Exception as e:
        logger.error(f"Constitutional search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar")
async def get_similar_documents(request: SimilarDocumentsRequest):
    """유사 문서 검색 (현재는 일반 검색으로 대체)"""
    try:
        logger.info(f"Similar documents request for: {request.doc_id}")

        search_engine = get_search_engine()
        # doc_id를 쿼리로 사용하여 검색
        result = search_engine.search(
            query=request.doc_id,
            search_types=[request.doc_type] if request.doc_type else None,
            max_results=request.max_results
        )

        return {
            "doc_id": request.doc_id,
            "doc_type": request.doc_type,
            "results": result.get("results", []),
            "total_results": result.get("total", 0)
        }

    except Exception as e:
        logger.error(f"Similar documents search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/build-index")
async def build_index(request: IndexBuildRequest):
    """벡터 인덱스 구축 (lawfirm_v2.db 기반으로 자동 생성됨)"""
    try:
        logger.info(f"Index build request for {len(request.documents)} documents")
        logger.warning("Index building is handled automatically by lawfirm_v2.db and SemanticSearchEngineV2")

        # HybridSearchEngineV2는 lawfirm_v2.db의 embeddings 테이블에서 자동으로 인덱스를 생성
        # 별도의 build_index 메서드는 없음

        return {
            "message": "Index is automatically managed by lawfirm_v2.db. Use data insertion to update index.",
            "document_count": len(request.documents),
            "success": True,
            "note": "lawfirm_v2_faiss.index is automatically generated from lawfirm_v2.db"
        }

    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_search_stats():
    """검색 엔진 통계 정보"""
    try:
        search_engine = get_search_engine()
        # HybridSearchEngineV2는 직접적인 stats 메서드가 없으므로 구성 정보 반환
        stats = {
            "db_path": search_engine.db_path,
            "model_name": search_engine.model_name,
            "search_config": search_engine.search_config,
            "index_file": f"{search_engine.db_path.replace('.db', '_faiss.index')}"
        }

        return {
            "search_engine_stats": stats,
            "success": True
        }

    except Exception as e:
        logger.error(f"Failed to get search stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_search_system(
    test_queries: List[str] = Body(..., description="테스트 쿼리 목록")
):
    """검색 시스템 테스트"""
    try:
        logger.info(f"Search system test with {len(test_queries)} queries")

        search_engine = get_search_engine()
        test_results = {}

        for query in test_queries:
            result = search_engine.search(query=query, max_results=10)
            test_results[query] = {
                "total_results": result.get("total", 0),
                "exact_count": result.get("exact_count", 0),
                "semantic_count": result.get("semantic_count", 0),
                "success": "error" not in result
            }

        return {
            "test_results": test_results,
            "total_queries": len(test_queries),
            "success": True
        }

    except Exception as e:
        logger.error(f"Search system test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config")
async def update_search_config(
    config: Dict[str, Any] = Body(..., description="검색 설정")
):
    """검색 설정 업데이트"""
    try:
        logger.info("Updating search configuration")

        search_engine = get_search_engine()
        # search_config 업데이트
        if isinstance(config, dict):
            search_engine.search_config.update(config)

        return {
            "message": "Search configuration updated",
            "new_config": search_engine.search_config,
            "success": True
        }

    except Exception as e:
        logger.error(f"Failed to update search config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 헬스체크 엔드포인트
@router.get("/health")
async def health_check():
    """검색 시스템 헬스체크"""
    try:
        search_engine = get_search_engine()
        # 인덱스 파일 존재 확인
        from pathlib import Path
        index_path = Path(search_engine.db_path.replace('.db', '_faiss.index'))

        return {
            "status": "healthy",
            "search_engine_available": True,
            "index_file": str(index_path),
            "index_exists": index_path.exists(),
            "db_path": search_engine.db_path,
            "success": True
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "search_engine_available": False,
            "error": str(e),
            "success": False
        }
