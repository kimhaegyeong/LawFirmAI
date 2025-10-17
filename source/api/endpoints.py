# -*- coding: utf-8 -*-
"""
API Endpoints (ML Enhanced)
RESTful API 엔드포인트 정의 - ML 강화 버전
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import time
from datetime import datetime
from source.utils.config import Config
from source.utils.logger import get_logger
from source.services.chat_service import ChatService
from source.services.rag_service import MLEnhancedRAGService
from source.services.search_service import MLEnhancedSearchService
from source.services.question_classifier import QuestionClassifier, QuestionType
from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.prompt_templates import PromptTemplateManager
from source.services.confidence_calculator import ConfidenceCalculator
from source.services.improved_answer_generator import ImprovedAnswerGenerator
from source.services.context_builder import ContextBuilder
from source.services.performance_monitoring import get_performance_monitor, start_monitoring, stop_monitoring
from source.services.feedback_system import get_feedback_collector, get_feedback_analyzer, FeedbackType, FeedbackRating
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.models.model_manager import LegalModelManager

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

class IntelligentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=20)
    include_law_sources: bool = True
    include_precedent_sources: bool = True

class IntelligentChatResponse(BaseModel):
    answer: str
    formatted_answer: Optional[Dict[str, Any]] = None
    question_type: str
    confidence: Dict[str, Any]
    law_sources: List[Dict[str, Any]]
    precedent_sources: List[Dict[str, Any]]
    search_stats: Dict[str, Any]
    processing_time: float
    warnings: List[str]
    recommendations: List[str]

class IntelligentChatV2Request(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=20)
    include_law_sources: bool = True
    include_precedent_sources: bool = True
    include_conversation_history: bool = True
    context_optimization: bool = True
    answer_formatting: bool = True

class FeedbackRequest(BaseModel):
    feedback_type: str = Field(..., regex="^(rating|text|bug_report|feature_request|general)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    text_content: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str
    timestamp: str

class PerformanceStatsResponse(BaseModel):
    current_metrics: Dict[str, Any]
    alert_stats: Dict[str, Any]
    health_status: Dict[str, Any]
    uptime: float
    request_count: int
    error_count: int

class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    feedback_by_type: Dict[str, int]
    rating_stats: Dict[str, int]
    average_rating: float
    daily_feedback: Dict[str, int]
    period_days: int

class IntelligentChatV2Response(BaseModel):
    answer: str
    formatted_answer: Optional[Dict[str, Any]] = None
    question_type: str
    confidence: Dict[str, Any]
    law_sources: List[Dict[str, Any]]
    precedent_sources: List[Dict[str, Any]]
    search_stats: Dict[str, Any]
    context_stats: Optional[Dict[str, Any]] = None
    processing_time: float
    warnings: List[str]
    recommendations: List[str]

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
    
    # Initialize intelligent chat services
    question_classifier = QuestionClassifier()
    hybrid_search_engine = HybridSearchEngine()
    prompt_template_manager = PromptTemplateManager()
    confidence_calculator = ConfidenceCalculator()
    context_builder = ContextBuilder()
    improved_answer_generator = ImprovedAnswerGenerator()
    
    # Initialize Phase 3 services
    performance_monitor = get_performance_monitor()
    feedback_collector = get_feedback_collector()
    feedback_analyzer = get_feedback_analyzer()
    
    # Start performance monitoring
    start_monitoring()
    
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
    
    @api_router.post("/chat/intelligent", response_model=IntelligentChatResponse)
    async def intelligent_chat_endpoint(request: IntelligentChatRequest):
        """지능형 채팅 엔드포인트 - 질문 유형별 최적화된 답변 제공"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Intelligent chat request received: {request.message[:100]}...")
            
            # 1. 질문 분류
            question_classification = question_classifier.classify_question(request.message)
            
            # 2. 지능형 검색 실행
            search_results = hybrid_search_engine.search_with_question_type(
                query=request.message,
                question_type=question_classification,
                max_results=request.max_results
            )
            
            # 3. 프롬프트 생성
            context_data = {
                "precedent_list": search_results.get("precedent_results", []),
                "law_articles": search_results.get("law_results", []),
                "context": search_results.get("results", [])
            }
            
            prompt = prompt_template_manager.format_prompt(
                question_type=question_classification.question_type,
                context_data=context_data,
                user_query=request.message
            )
            
            # 4. 답변 생성 (ImprovedAnswerGenerator 사용)
            answer_result = improved_answer_generator.generate_answer(
                query=request.message,
                question_type=question_classification,
                context=prompt,
                sources=search_results
            )
            
            # 5. 소스 분리
            law_sources = []
            precedent_sources = []
            
            if request.include_law_sources:
                law_sources = search_results.get("law_results", [])
            
            if request.include_precedent_sources:
                precedent_sources = search_results.get("precedent_results", [])
            
            processing_time = time.time() - start_time
            
            # 구조화된 답변 정보 준비
            formatted_answer_info = None
            if answer_result.formatted_answer:
                formatted_answer_info = {
                    "formatted_content": answer_result.formatted_answer.formatted_content,
                    "sections": answer_result.formatted_answer.sections,
                    "metadata": answer_result.formatted_answer.metadata
                }
            
            return IntelligentChatResponse(
                answer=answer_result.answer,
                formatted_answer=formatted_answer_info,
                question_type=question_classification.question_type.value,
                confidence={
                    "confidence": answer_result.confidence.confidence,
                    "reliability_level": answer_result.confidence.reliability_level.value,
                    "similarity_score": answer_result.confidence.similarity_score,
                    "matching_score": answer_result.confidence.matching_score,
                    "answer_quality": answer_result.confidence.answer_quality
                },
                law_sources=law_sources,
                precedent_sources=precedent_sources,
                search_stats=search_results.get("search_stats", {}),
                processing_time=processing_time,
                warnings=answer_result.confidence.warnings,
                recommendations=answer_result.confidence.recommendations
            )
            
        except Exception as e:
            logger.error(f"Intelligent chat endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/chat/intelligent-v2", response_model=IntelligentChatV2Response)
    async def intelligent_chat_v2_endpoint(request: IntelligentChatV2Request):
        """지능형 채팅 엔드포인트 v2 - 모든 개선사항 통합"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Intelligent chat v2 request received: {request.message[:100]}...")
            
            # 1. 질문 분류
            question_classification = question_classifier.classify_question(request.message)
            
            # 2. 지능형 검색 실행
            search_results = hybrid_search_engine.search_with_question_type(
                query=request.message,
                question_type=question_classification,
                max_results=request.max_results
            )
            
            # 3. 대화 이력 준비 (세션 기반)
            conversation_history = None
            if request.include_conversation_history and request.session_id:
                # TODO: 실제 세션 관리 시스템과 연동
                conversation_history = []
            
            # 4. 컨텍스트 최적화
            context_stats = None
            if request.context_optimization:
                context_window = context_builder.build_optimized_context(
                    query=request.message,
                    question_classification=question_classification,
                    search_results=search_results,
                    conversation_history=conversation_history
                )
                context_stats = {
                    "total_items": len(context_window.items),
                    "total_tokens": context_window.total_tokens,
                    "utilization_rate": context_window.utilization_rate,
                    "priority_distribution": context_window.priority_distribution
                }
            
            # 5. 답변 생성 (ImprovedAnswerGenerator 사용)
            answer_result = improved_answer_generator.generate_answer(
                query=request.message,
                question_type=question_classification,
                context="",  # 컨텍스트는 이미 최적화됨
                sources=search_results,
                conversation_history=conversation_history
            )
            
            # 6. 소스 분리
            law_sources = []
            precedent_sources = []
            
            if request.include_law_sources:
                law_sources = search_results.get("law_results", [])
            
            if request.include_precedent_sources:
                precedent_sources = search_results.get("precedent_results", [])
            
            processing_time = time.time() - start_time
            
            # 7. 구조화된 답변 정보 준비
            formatted_answer_info = None
            if request.answer_formatting and answer_result.formatted_answer:
                formatted_answer_info = {
                    "formatted_content": answer_result.formatted_answer.formatted_content,
                    "sections": answer_result.formatted_answer.sections,
                    "metadata": answer_result.formatted_answer.metadata
                }
            
            return IntelligentChatV2Response(
                answer=answer_result.answer,
                formatted_answer=formatted_answer_info,
                question_type=question_classification.question_type.value,
                confidence={
                    "confidence": answer_result.confidence.confidence,
                    "reliability_level": answer_result.confidence.reliability_level.value,
                    "similarity_score": answer_result.confidence.similarity_score,
                    "matching_score": answer_result.confidence.matching_score,
                    "answer_quality": answer_result.confidence.answer_quality
                },
                law_sources=law_sources,
                precedent_sources=precedent_sources,
                search_stats=search_results.get("search_stats", {}),
                context_stats=context_stats,
                processing_time=processing_time,
                warnings=answer_result.confidence.warnings,
                recommendations=answer_result.confidence.recommendations
            )
            
        except Exception as e:
            logger.error(f"Intelligent chat v2 endpoint error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/system/status", response_model=Dict[str, Any])
    async def system_status_endpoint():
        """시스템 상태 확인 엔드포인트 - 모든 컴포넌트 상태 점검"""
        try:
            logger.info("System status check requested")
            
            status = {
                "timestamp": time.time(),
                "overall_status": "healthy",
                "components": {},
                "version": "2.0.0"
            }
            
            # 데이터베이스 상태 확인
            try:
                db_stats_query = "SELECT COUNT(*) as count FROM assembly_articles"
                db_result = database.execute_query(db_stats_query)
                db_count = db_result[0]["count"] if db_result else 0
                
                status["components"]["database"] = {
                    "status": "healthy",
                    "total_articles": db_count,
                    "connection": "active"
                }
            except Exception as e:
                status["components"]["database"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
            
            # 벡터 스토어 상태 확인
            try:
                vector_stats = vector_store.get_stats()
                status["components"]["vector_store"] = {
                    "status": "healthy",
                    "stats": vector_stats
                }
            except Exception as e:
                status["components"]["vector_store"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
            
            # AI 모델 상태 확인
            try:
                # 간단한 모델 테스트
                test_result = question_classifier.classify_question("테스트 질문")
                status["components"]["ai_models"] = {
                    "status": "healthy",
                    "question_classifier": "active",
                    "test_classification": test_result.question_type.value
                }
            except Exception as e:
                status["components"]["ai_models"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
            
            # 검색 엔진 상태 확인
            try:
                # 간단한 검색 테스트
                test_search = hybrid_search_engine.search_with_question_type("테스트 검색", max_results=1)
                status["components"]["search_engines"] = {
                    "status": "healthy",
                    "hybrid_search": "active",
                    "test_results_count": len(test_search.get("results", []))
                }
            except Exception as e:
                status["components"]["search_engines"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
            
            # 답변 생성기 상태 확인
            try:
                # 간단한 답변 생성 테스트
                test_classification = question_classifier.classify_question("테스트")
                test_sources = {"results": [], "law_results": [], "precedent_results": []}
                test_answer = improved_answer_generator.generate_answer(
                    query="테스트",
                    question_type=test_classification,
                    context="테스트 컨텍스트",
                    sources=test_sources
                )
                status["components"]["answer_generator"] = {
                    "status": "healthy",
                    "ollama_client": "active",
                    "answer_formatter": "active",
                    "context_builder": "active",
                    "test_answer_length": len(test_answer.answer)
                }
            except Exception as e:
                status["components"]["answer_generator"] = {
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
            
            return status
            
        except Exception as e:
            logger.error(f"System status check error: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": "error",
                "error": str(e),
                "version": "2.0.0"
            }
    
    @api_router.post("/feedback", response_model=FeedbackResponse)
    async def submit_feedback_endpoint(request: FeedbackRequest):
        """피드백 제출 엔드포인트"""
        try:
            logger.info(f"Feedback submission received: {request.feedback_type}")
            
            # 피드백 타입 변환
            feedback_type = FeedbackType(request.feedback_type)
            rating = FeedbackRating(request.rating) if request.rating else None
            
            # 피드백 제출
            feedback_id = feedback_collector.submit_feedback(
                feedback_type=feedback_type,
                rating=rating,
                text_content=request.text_content,
                question=request.question,
                answer=request.answer,
                session_id=request.session_id,
                user_id=request.user_id,
                context=request.context,
                metadata=request.metadata
            )
            
            return FeedbackResponse(
                feedback_id=feedback_id,
                message="피드백이 성공적으로 제출되었습니다.",
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Feedback submission error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/feedback/stats", response_model=FeedbackStatsResponse)
    async def get_feedback_stats_endpoint(days: int = Query(30, ge=1, le=365)):
        """피드백 통계 조회 엔드포인트"""
        try:
            logger.info(f"Feedback stats requested for {days} days")
            
            stats = feedback_collector.get_feedback_stats(days)
            
            return FeedbackStatsResponse(**stats)
            
        except Exception as e:
            logger.error(f"Feedback stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/performance/stats", response_model=PerformanceStatsResponse)
    async def get_performance_stats_endpoint():
        """성능 통계 조회 엔드포인트"""
        try:
            logger.info("Performance stats requested")
            
            current_metrics = performance_monitor.get_current_metrics()
            alert_stats = performance_monitor.get_alert_stats()
            health_status = performance_monitor.get_health_status()
            
            return PerformanceStatsResponse(
                current_metrics=current_metrics,
                alert_stats=alert_stats,
                health_status=health_status,
                uptime=current_metrics.get("uptime", 0),
                request_count=current_metrics.get("request_count", 0),
                error_count=current_metrics.get("error_count", 0)
            )
            
        except Exception as e:
            logger.error(f"Performance stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/performance/alerts")
    async def get_performance_alerts_endpoint():
        """성능 알림 조회 엔드포인트"""
        try:
            logger.info("Performance alerts requested")
            
            active_alerts = performance_monitor.alert_manager.get_active_alerts()
            
            return {
                "active_alerts": [
                    {
                        "id": alert.id,
                        "timestamp": alert.timestamp.isoformat(),
                        "level": alert.level.value,
                        "metric_type": alert.metric_type.value,
                        "message": alert.message,
                        "value": alert.value,
                        "threshold": alert.threshold
                    }
                    for alert in active_alerts
                ],
                "total_active": len(active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Performance alerts error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/performance/alerts/{alert_id}/resolve")
    async def resolve_alert_endpoint(alert_id: str):
        """알림 해결 엔드포인트"""
        try:
            logger.info(f"Alert resolution requested: {alert_id}")
            
            performance_monitor.alert_manager.resolve_alert(alert_id)
            
            return {
                "message": "알림이 해결되었습니다.",
                "alert_id": alert_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Alert resolution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Include router in app
    app.include_router(api_router)
    
    logger.info("ML-enhanced API routes configured successfully")
