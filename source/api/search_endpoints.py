"""
검색 API 엔드포인트
하이브리드 검색 시스템을 위한 REST API 엔드포인트
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from services.hybrid_search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)

# API 라우터 생성
router = APIRouter(prefix="/api/search", tags=["search"])

# 하이브리드 검색 엔진 인스턴스
search_engine = None

def get_search_engine() -> HybridSearchEngine:
    """검색 엔진 인스턴스 반환"""
    global search_engine
    if search_engine is None:
        search_engine = HybridSearchEngine()
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
            results=result["results"],
            total_results=result["total_results"],
            search_stats=result["search_stats"],
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
        results = search_engine.search_laws_only(query, max_results)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
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
        results = search_engine.search_precedents_only(query, max_results)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
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
        results = search_engine.search_constitutional_only(query, max_results)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_type": "constitutional_only"
        }
        
    except Exception as e:
        logger.error(f"Constitutional search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similar")
async def get_similar_documents(request: SimilarDocumentsRequest):
    """유사 문서 검색"""
    try:
        logger.info(f"Similar documents request for: {request.doc_id}")
        
        search_engine = get_search_engine()
        results = search_engine.get_similar_documents(
            doc_id=request.doc_id,
            doc_type=request.doc_type,
            k=request.max_results
        )
        
        return {
            "doc_id": request.doc_id,
            "doc_type": request.doc_type,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Similar documents search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/build-index")
async def build_index(request: IndexBuildRequest):
    """벡터 인덱스 구축"""
    try:
        logger.info(f"Index build request for {len(request.documents)} documents")
        
        search_engine = get_search_engine()
        
        if request.force_rebuild:
            # 강제 재구축의 경우 기존 인덱스 삭제 후 재구축
            # 실제 구현에서는 인덱스 파일 삭제 로직 필요
            pass
        
        success = search_engine.build_index(request.documents)
        
        if success:
            return {
                "message": "Index built successfully",
                "document_count": len(request.documents),
                "success": True
            }
        else:
            raise HTTPException(status_code=500, detail="Index building failed")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_search_stats():
    """검색 엔진 통계 정보"""
    try:
        search_engine = get_search_engine()
        stats = search_engine.get_search_stats()
        
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
        test_results = search_engine.test_search(test_queries)
        
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
        search_engine.update_search_config(config)
        
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
        stats = search_engine.get_search_stats()
        
        return {
            "status": "healthy",
            "search_engine_available": True,
            "index_stats": stats.get("semantic_search", {}),
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
