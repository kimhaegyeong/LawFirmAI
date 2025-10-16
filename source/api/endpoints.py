# -*- coding: utf-8 -*-
"""
API Endpoints (ML Enhanced)
RESTful API 엔드포인트 정의 - ML 강화 버전
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from utils.config import Config
from utils.logger import get_logger
from services.chat_service import ChatService
from services.rag_service import MLEnhancedRAGService
from services.search_service import MLEnhancedSearchService
from data.database import DatabaseManager
from data.vector_store import LegalVectorStore
from models.model_manager import LegalModelManager

logger = get_logger(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None

class MLEnhancedChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None
    use_ml_enhanced: bool = True
    quality_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    search_type: str = Field(default="hybrid", regex="^(semantic|keyword|hybrid|supplementary|high_quality)$")

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: list
    processing_time: float

class MLEnhancedChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    ml_enhanced: bool
    ml_stats: Dict[str, Any]
    retrieved_docs_count: int

class SearchRequest(BaseModel):
    query: str
    search_type: str = Field(default="hybrid", regex="^(semantic|keyword|hybrid|supplementary|high_quality)$")
    limit: int = Field(default=10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_type: str
    ml_enhanced: bool
    processing_time: float

class LegalEntityRequest(BaseModel):
    query: str

class LegalEntityResponse(BaseModel):
    laws: List[str]
    articles: List[str]
    cases: List[str]
    supplementary: List[str]

class QualityStatsResponse(BaseModel):
    total_documents: int
    ml_enhanced_documents: int
    avg_quality_score: float
    main_articles_count: int
    supplementary_articles_count: int
    parsing_methods: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ml_enhanced: bool
    vector_store_status: Dict[str, Any]
    database_status: Dict[str, Any]

def setup_routes(app, config: Config):
    """ML 강화 라우트 설정"""
    
    # Initialize services
    chat_service = ChatService(config)
    
    # Initialize ML-enhanced services
    database = DatabaseManager(config.database_url)
    vector_store = LegalVectorStore(
        model_name="BAAI/bge-m3",
        dimension=1024,
        index_type="flat"
    )
    model_manager = LegalModelManager(config)
    
    ml_rag_service = MLEnhancedRAGService(config, model_manager, vector_store, database)
    ml_search_service = MLEnhancedSearchService(config, database, vector_store, model_manager)
    
    # Create API router
    api_router = APIRouter(prefix="/api/v1")
    
    @api_router.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """기본 채팅 엔드포인트 (레거시 호환성)"""
        try:
            logger.info(f"Chat request received: {request.message[:100]}...")
            
            result = chat_service.process_message(
                message=request.message,
                context=request.context
            )
            
            return ChatResponse(**result)
            
        except Exception as e:
            logger.error(f"Chat endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/chat/ml-enhanced", response_model=MLEnhancedChatResponse)
    async def ml_enhanced_chat_endpoint(request: MLEnhancedChatRequest):
        """ML 강화 채팅 엔드포인트"""
        try:
            logger.info(f"ML-enhanced chat request received: {request.message[:100]}...")
            
            # ML 강화 RAG 서비스 사용
            result = ml_rag_service.process_query(
                query=request.message,
                top_k=5,
                filters={"quality_threshold": request.quality_threshold} if request.use_ml_enhanced else None
            )
            
            return MLEnhancedChatResponse(
                response=result.get("response", ""),
                confidence=result.get("confidence", 0.0),
                sources=result.get("sources", []),
                processing_time=result.get("processing_time", 0.0),
                ml_enhanced=result.get("ml_enhanced", True),
                ml_stats=result.get("ml_stats", {}),
                retrieved_docs_count=result.get("retrieved_docs_count", 0)
            )
            
        except Exception as e:
            logger.error(f"ML-enhanced chat endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/search", response_model=SearchResponse)
    async def search_endpoint(request: SearchRequest):
        """ML 강화 검색 엔드포인트"""
        try:
            logger.info(f"Search request received: {request.query[:100]}...")
            
            import time
            start_time = time.time()
            
            results = ml_search_service.search_documents(
                query=request.query,
                search_type=request.search_type,
                limit=request.limit,
                filters=request.filters
            )
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                total_count=len(results),
                search_type=request.search_type,
                ml_enhanced=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Search endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/legal-entities", response_model=LegalEntityResponse)
    async def legal_entities_endpoint(request: LegalEntityRequest):
        """법률 엔티티 추출 엔드포인트"""
        try:
            logger.info(f"Legal entities request received: {request.query[:100]}...")
            
            entities = ml_search_service.search_legal_entities(request.query)
            
            return LegalEntityResponse(
                laws=entities.get("laws", []),
                articles=entities.get("articles", []),
                cases=entities.get("cases", []),
                supplementary=entities.get("supplementary", [])
            )
            
        except Exception as e:
            logger.error(f"Legal entities endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/search/suggestions")
    async def search_suggestions_endpoint(
        query: str = Query(..., description="검색 쿼리"),
        limit: int = Query(default=5, ge=1, le=10, description="제안 개수")
    ):
        """검색 제안 엔드포인트"""
        try:
            logger.info(f"Search suggestions request received: {query[:100]}...")
            
            suggestions = ml_search_service.get_search_suggestions(query, limit)
            
            return {"suggestions": suggestions}
            
        except Exception as e:
            logger.error(f"Search suggestions endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/quality-stats", response_model=QualityStatsResponse)
    async def quality_stats_endpoint():
        """품질 통계 엔드포인트"""
        try:
            logger.info("Quality stats request received")
            
            # 데이터베이스에서 품질 통계 조회
            stats_query = """
                SELECT 
                    COUNT(*) as total_documents,
                    SUM(CASE WHEN ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_documents,
                    AVG(parsing_quality_score) as avg_quality_score,
                    SUM(CASE WHEN article_type = 'main' THEN 1 ELSE 0 END) as main_articles_count,
                    SUM(CASE WHEN article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_articles_count
                FROM assembly_articles
            """
            
            parsing_methods_query = """
                SELECT parsing_method, COUNT(*) as count
                FROM assembly_articles
                GROUP BY parsing_method
            """
            
            stats_result = database.execute_query(stats_query)
            parsing_methods_result = database.execute_query(parsing_methods_query)
            
            stats = stats_result[0] if stats_result else {}
            parsing_methods = {row["parsing_method"]: row["count"] for row in parsing_methods_result}
            
            return QualityStatsResponse(
                total_documents=stats.get("total_documents", 0),
                ml_enhanced_documents=stats.get("ml_enhanced_documents", 0),
                avg_quality_score=round(stats.get("avg_quality_score", 0.0), 3),
                main_articles_count=stats.get("main_articles_count", 0),
                supplementary_articles_count=stats.get("supplementary_articles_count", 0),
                parsing_methods=parsing_methods
            )
            
        except Exception as e:
            logger.error(f"Quality stats endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/health", response_model=HealthResponse)
    async def health_check():
        """ML 강화 헬스체크 엔드포인트"""
        try:
            # 벡터 스토어 상태 확인
            vector_stats = vector_store.get_stats()
            
            # 데이터베이스 상태 확인
            db_stats_query = "SELECT COUNT(*) as count FROM assembly_articles"
            db_result = database.execute_query(db_stats_query)
            db_count = db_result[0]["count"] if db_result else 0
            
            return HealthResponse(
                status="healthy",
                service="LawFirmAI API (ML Enhanced)",
                version="2.0.0",
                ml_enhanced=True,
                vector_store_status=vector_stats,
                database_status={"total_articles": db_count}
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Include router in app
    app.include_router(api_router)
    
    logger.info("ML-enhanced API routes configured successfully")
