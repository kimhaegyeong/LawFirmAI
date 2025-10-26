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
from source.utils.input_validator import get_input_validator, ValidationResult
from source.utils.security_logger import get_security_logger, SecurityEventType, SecurityLevel
from source.utils.privacy_compliance import get_privacy_compliance_manager, ProcessingPurpose
from source.utils.logger import get_logger
from source.utils.config import Config
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
from source.services.legal_basis_integration_service import LegalBasisIntegrationService
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from source.models.model_manager import LegalModelManager

logger = get_logger(__name__)

# Create API router at module level
api_router = APIRouter(prefix="/api/v1")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    session_id: Optional[str] = None

# Search API Models
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
    success: bool

class SimilarDocumentsRequest(BaseModel):
    """유사 문서 검색 요청 모델"""
    doc_id: str = Field(..., description="문서 ID")
    doc_type: str = Field(..., description="문서 타입")
    max_results: Optional[int] = Field(default=10, description="최대 결과 수", ge=1, le=50)

class IndexBuildRequest(BaseModel):
    """인덱스 구축 요청 모델"""
    documents: List[Dict[str, Any]] = Field(..., description="구축할 문서 목록")
    force_rebuild: Optional[bool] = Field(default=False, description="강제 재구축")

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
    search_type: str = Field(default="hybrid", pattern="^(semantic|keyword|hybrid|supplementary|high_quality)$")
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

class LegalBasisRequest(BaseModel):
    query: str
    answer: str
    question_type: Optional[str] = None
    include_validation: bool = True
    include_citations: bool = True

class LegalBasisResponse(BaseModel):
    success: bool
    original_answer: str
    enhanced_answer: str
    structured_answer: str
    legal_basis: Dict[str, Any]
    confidence: float
    is_legally_sound: bool
    analysis: Dict[str, Any]
    processing_timestamp: str
    error: Optional[str] = None

class LegalCitationRequest(BaseModel):
    text: str
    include_validation: bool = True

class LegalCitationResponse(BaseModel):
    success: bool
    citations_found: int
    valid_citations: int
    invalid_citations: int
    validation_details: List[Dict[str, Any]]
    enhanced_text: str
    legal_basis_summary: Dict[str, Any]
    error: Optional[str] = None

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
    feedback_type: str = Field(..., pattern="^(rating|text|bug_report|feature_request|general)$")
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

    # API router is already created at module level

    @api_router.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """기본 채팅 엔드포인트 (레거시 호환성)"""
        try:
            # 입력 검증
            input_validator = get_input_validator()
            security_logger = get_security_logger()
            privacy_manager = get_privacy_compliance_manager()

            validation_report = input_validator.validate_input(
                input_data=request.message,
                user_id=getattr(request, 'user_id', None),
                session_id=request.session_id
            )

            # 검증 실패 시 에러 반환
            if validation_report.result == ValidationResult.BLOCKED:
                security_logger.log_security_violation(
                    violation_type="input_blocked",
                    description=f"Blocked input: {validation_report.violations}",
                    session_id=request.session_id,
                    details={
                        'input_preview': request.message[:100],
                        'violations': validation_report.violations,
                        'risk_score': validation_report.risk_score
                    }
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"입력이 차단되었습니다: {validation_report.message}"
                )

            # 의심스러운 입력에 대한 경고 로그
            if validation_report.result == ValidationResult.SUSPICIOUS:
                security_logger.log_security_violation(
                    violation_type="suspicious_input",
                    description=f"Suspicious input detected: {validation_report.violations}",
                    session_id=request.session_id,
                    details={
                        'input_preview': request.message[:100],
                        'violations': validation_report.violations,
                        'risk_score': validation_report.risk_score
                    }
                )

            # 개인정보보호법 준수 처리
            privacy_result = privacy_manager.process_text(
                text=validation_report.sanitized_input,
                processing_purpose=ProcessingPurpose.LEGAL_CONSULTATION,
                user_id=getattr(request, 'user_id', None),
                session_id=request.session_id
            )

            # 개인정보가 발견된 경우 처리된 텍스트 사용
            if privacy_result['has_personal_data']:
                security_logger.log_event(
                    SecurityEventType.DATA_ACCESS,
                    SecurityLevel.MEDIUM,
                    f"Personal data processed: {privacy_result['detected_count']} items",
                    {
                        'data_types': privacy_result['data_types'],
                        'retention_period': privacy_result['retention_period']
                    },
                    user_id=getattr(request, 'user_id', None),
                    session_id=request.session_id
                )

            logger.info(f"Chat request received: {request.message[:100]}...")

            # 개인정보가 마스킹된 텍스트로 처리
            processed_message = privacy_result['masked_text']

            result = chat_service.process_message(
                message=processed_message,
                context=request.context
            )

            return ChatResponse(**result)

        except HTTPException:
            raise
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

    # 법적 근거 관련 엔드포인트
    @api_router.post("/legal-basis/enhance", response_model=LegalBasisResponse)
    async def enhance_answer_with_legal_basis(request: LegalBasisRequest):
        """법적 근거를 포함한 답변 강화 엔드포인트"""
        try:
            logger.info(f"Legal basis enhancement requested for query: {request.query[:100]}...")

            # 법적 근거 통합 서비스 초기화
            legal_basis_service = LegalBasisIntegrationService()

            # 질문 유형 변환
            question_type = None
            if request.question_type:
                try:
                    from source.services.answer_structure_enhancer import QuestionType
                    question_type = QuestionType(request.question_type)
                except ValueError:
                    logger.warning(f"Invalid question type: {request.question_type}")

            # 법적 근거 강화 처리
            result = legal_basis_service.process_query_with_legal_basis(
                request.query,
                request.answer,
                question_type
            )

            return LegalBasisResponse(
                success=True,
                original_answer=result["original_answer"],
                enhanced_answer=result["enhanced_answer"],
                structured_answer=result["structured_answer"],
                legal_basis=result["legal_basis"],
                confidence=result["confidence"],
                is_legally_sound=result["is_legally_sound"],
                analysis=result["analysis"],
                processing_timestamp=result["processing_timestamp"]
            )

        except Exception as e:
            logger.error(f"Legal basis enhancement error: {e}")
            return LegalBasisResponse(
                success=False,
                original_answer=request.answer,
                enhanced_answer=request.answer,
                structured_answer=request.answer,
                legal_basis={"citations": {}, "validation": {}, "summary": {}},
                confidence=0.0,
                is_legally_sound=False,
                analysis={"error": str(e)},
                processing_timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    @api_router.post("/legal-citations/validate", response_model=LegalCitationResponse)
    async def validate_legal_citations(request: LegalCitationRequest):
        """법적 인용 검증 엔드포인트"""
        try:
            logger.info(f"Legal citation validation requested for text: {request.text[:100]}...")

            # 법적 근거 통합 서비스 초기화
            legal_basis_service = LegalBasisIntegrationService()

            # 법적 인용 검증
            result = legal_basis_service.validate_legal_citations(request.text)

            if result["success"]:
                return LegalCitationResponse(
                    success=True,
                    citations_found=result["citations_found"],
                    valid_citations=result["valid_citations"],
                    invalid_citations=result["invalid_citations"],
                    validation_details=result["validation_details"],
                    enhanced_text=result["enhanced_text"],
                    legal_basis_summary={}  # 필요시 추가 구현
                )
            else:
                return LegalCitationResponse(
                    success=False,
                    citations_found=0,
                    valid_citations=0,
                    invalid_citations=0,
                    validation_details=[],
                    enhanced_text=request.text,
                    legal_basis_summary={},
                    error=result.get("error", "Unknown error")
                )

        except Exception as e:
            logger.error(f"Legal citation validation error: {e}")
            return LegalCitationResponse(
                success=False,
                citations_found=0,
                valid_citations=0,
                invalid_citations=0,
                validation_details=[],
                enhanced_text=request.text,
                legal_basis_summary={},
                error=str(e)
            )

    @api_router.get("/legal-basis/statistics")
    async def get_legal_basis_statistics(days: int = Query(default=30, ge=1, le=365)):
        """법적 근거 통계 조회 엔드포인트"""
        try:
            logger.info(f"Legal basis statistics requested for {days} days")

            # 법적 근거 통합 서비스 초기화
            legal_basis_service = LegalBasisIntegrationService()

            # 통계 조회
            stats = legal_basis_service.get_legal_basis_statistics(days)

            return {
                "success": True,
                "statistics": stats,
                "period_days": days,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Legal basis statistics error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.post("/legal-basis/enhance-existing")
    async def enhance_existing_answer(request: LegalBasisRequest):
        """기존 답변을 법적 근거로 강화하는 엔드포인트"""
        try:
            logger.info(f"Existing answer enhancement requested for query: {request.query[:100]}...")

            # 법적 근거 통합 서비스 초기화
            legal_basis_service = LegalBasisIntegrationService()

            # 기존 답변 강화
            result = legal_basis_service.enhance_existing_answer(
                request.answer,
                request.query
            )

            return {
                "success": result["success"],
                "original_answer": result["original_answer"],
                "enhanced_answer": result["enhanced_answer"],
                "legal_citations": result.get("legal_citations", {}),
                "validation": result.get("validation", {}),
                "confidence": result.get("confidence", 0.0),
                "is_legally_sound": result.get("is_legally_sound", False),
                "error": result.get("error"),
                "processing_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Existing answer enhancement error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.get("/privacy/compliance-report")
    async def get_privacy_compliance_report():
        """개인정보보호법 준수 보고서 조회"""
        try:
            privacy_manager = get_privacy_compliance_manager()
            report = privacy_manager.generate_compliance_report()

            return {
                "success": True,
                "report": {
                    "total_processed": report.total_processed,
                    "personal_data_found": report.personal_data_found,
                    "consent_required": report.consent_required,
                    "consent_given": report.consent_given,
                    "data_retention_compliant": report.data_retention_compliant,
                    "violations": report.violations,
                    "recommendations": report.recommendations,
                    "compliance_score": report.compliance_score
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Privacy compliance report error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.get("/privacy/notice")
    async def get_privacy_notice():
        """개인정보 처리방침 안내"""
        try:
            privacy_manager = get_privacy_compliance_manager()
            notice = privacy_manager.get_privacy_notice()

            return {
                "success": True,
                "notice": notice,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Privacy notice error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.post("/privacy/cleanup")
    async def cleanup_expired_data():
        """만료된 개인정보 삭제"""
        try:
            privacy_manager = get_privacy_compliance_manager()
            cleaned_count = privacy_manager.cleanup_expired_data()

            return {
                "success": True,
                "cleaned_count": cleaned_count,
                "message": f"만료된 개인정보 {cleaned_count}건이 삭제되었습니다.",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Privacy cleanup error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Search Endpoints
    @api_router.post("/search", response_model=SearchResponse)
    async def search_documents(request: SearchRequest):
        """하이브리드 검색 실행"""
        try:
            logger.info(f"Search request received: {request.query}")

            result = hybrid_search_engine.search(
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
                success=result.get("success", True)
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.get("/search/laws")
    async def search_laws(
        query: str = Query(..., description="검색 쿼리"),
        max_results: int = Query(default=20, description="최대 결과 수", ge=1, le=50)
    ):
        """법률 검색"""
        try:
            logger.info(f"Law search request: {query}")

            result = hybrid_search_engine.search(
                query=query,
                search_types=["law"],
                max_results=max_results
            )

            return {
                "query": query,
                "results": result["results"],
                "total_results": result["total_results"],
                "search_stats": result["search_stats"],
                "success": True
            }

        except Exception as e:
            logger.error(f"Law search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.get("/search/precedents")
    async def search_precedents(
        query: str = Query(..., description="검색 쿼리"),
        max_results: int = Query(default=20, description="최대 결과 수", ge=1, le=50)
    ):
        """판례 검색"""
        try:
            logger.info(f"Precedent search request: {query}")

            result = hybrid_search_engine.search(
                query=query,
                search_types=["precedent"],
                max_results=max_results
            )

            return {
                "query": query,
                "results": result["results"],
                "total_results": result["total_results"],
                "search_stats": result["search_stats"],
                "success": True
            }

        except Exception as e:
            logger.error(f"Precedent search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @api_router.get("/search/stats")
    async def get_search_stats():
        """검색 엔진 통계 정보"""
        try:
            stats = hybrid_search_engine.stats

            return {
                "search_engine_stats": stats,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Include router in app
    app.include_router(api_router)

    logger.info("ML-enhanced API routes configured successfully")

# Initialize routes at module level
def initialize_routes():
    """모듈 레벨에서 라우트 초기화"""
    try:
        # 기본 설정으로 Config 생성
        config = Config()
        setup_routes(None, config)  # app은 None으로 전달 (이미 api_router에 등록됨)
    except Exception as e:
        logger.error(f"Failed to initialize routes: {e}")

# 모듈 로드 시 자동으로 라우트 초기화
initialize_routes()
