#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google AI 모델 RAG Test Script
Google AI 모델을 사용한 RAG 시스템 테스트 스크립트

이 스크립트는 .env 파일에서 환경 변수를 자동으로 로드합니다.
.env 파일에 다음 설정이 필요합니다:

필수 설정:
- LLM_PROVIDER=google
- GOOGLE_API_KEY=your-google-api-key-here

선택적 설정:
- LLM_MODEL=gemini-pro (또는 다른 Google AI 모델)
- LLM_TEMPERATURE=0.7
- LLM_MAX_TOKENS=1000
- LANGFUSE_ENABLED=false
"""

import os
import sys
import logging
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'source'))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

from services.langchain_rag_service import LangChainRAGService
from utils.langchain_config import LangChainConfig

# 로깅 설정
from utils.safe_logging import setup_script_logging
logger = setup_script_logging("test_gemini_pro_rag")


def setup_gemini_environment():
    """Google AI 모델 환경 설정"""
    # .env 파일에서 환경 변수가 이미 로드되었으므로 추가 설정 불필요
    # Google API 키 확인만 수행
    if not os.getenv('GOOGLE_API_KEY'):
        logger.warning("⚠️ GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        logger.info("다음 방법으로 설정하세요:")
        logger.info("1. .env 파일에 GOOGLE_API_KEY=your-google-api-key-here 추가")
        logger.info("2. 또는 환경 변수로 직접 설정: export GOOGLE_API_KEY=your-google-api-key-here")
        return False
    
    model_name = os.getenv('LLM_MODEL', 'gemini-pro')
    logger.info(f"✅ {model_name} 환경 설정 완료")
    logger.info(f"   - LLM Provider: {os.getenv('LLM_PROVIDER', 'Not set')}")
    logger.info(f"   - LLM Model: {model_name}")
    logger.info(f"   - Google API Key: {'설정됨' if os.getenv('GOOGLE_API_KEY') else '미설정'}")
    logger.info(f"   - Langfuse Enabled: {os.getenv('LANGFUSE_ENABLED', 'Not set')}")
    return True


def test_gemini_configuration():
    """Google AI 모델 설정 테스트"""
    logger.info("=== Google AI 모델 설정 테스트 ===")
    
    try:
        config = LangChainConfig.from_env()
        
        # .env 파일에서 로드된 설정 확인
        logger.info("📋 .env 파일에서 로드된 설정:")
        logger.info(f"   - LLM Provider: {config.llm_provider.value}")
        logger.info(f"   - LLM Model: {config.llm_model}")
        logger.info(f"   - LLM Temperature: {config.llm_temperature}")
        logger.info(f"   - LLM Max Tokens: {config.llm_max_tokens}")
        logger.info(f"   - Vector Store Type: {config.vector_store_type.value}")
        logger.info(f"   - Embedding Model: {config.embedding_model}")
        logger.info(f"   - Google API Key: {'설정됨' if config.google_api_key else '미설정'}")
        logger.info(f"   - Langfuse Enabled: {config.langfuse_enabled}")
        
        # Google AI 모델 사용을 위한 검증
        if config.llm_provider.value != "google":
            logger.warning(f"⚠️ LLM 제공자가 'google'이 아닙니다: {config.llm_provider.value}")
            logger.info(f"{config.llm_model}를 사용하려면 .env 파일에서 LLM_PROVIDER=google으로 설정하세요.")
            return False
        
        if not config.google_api_key:
            logger.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
            logger.info(f"{config.llm_model}를 사용하려면 .env 파일에서 GOOGLE_API_KEY를 설정하세요.")
            return False
        
        if config.google_api_key == "your-google-api-key-here" or config.google_api_key == "test-google-api-key":
            logger.warning("⚠️ 테스트용 API 키가 설정되어 있습니다.")
            logger.info("실제 Google API 키로 교체하세요.")
        
        logger.info(f"✅ {config.llm_model} 설정 검증 완료")
        
        # 설정 유효성 검사
        errors = config.validate()
        if errors:
            logger.warning(f"⚠️ 설정 오류: {errors}")
            return False
        else:
            logger.info("✅ 설정 유효성 검사 통과")
            
        return config
        
    except Exception as e:
        logger.error(f"❌ 설정 테스트 실패: {e}")
        return None


def test_gemini_rag_service(config: LangChainConfig):
    """Google AI 모델 RAG 서비스 테스트"""
    logger.info(f"=== {config.llm_model} RAG 서비스 테스트 ===")
    
    try:
        service = LangChainRAGService(config)
        logger.info("✅ RAG 서비스 초기화 성공")
        
        # 서비스 통계 조회
        stats = service.get_service_statistics()
        logger.info("📊 서비스 통계:")
        logger.info(f"   - LLM 모델: {stats['llm_model']}")
        logger.info(f"   - LangChain 사용 가능: {stats['langchain_available']}")
        logger.info(f"   - Langfuse 활성화: {stats['langfuse_enabled']}")
        
        return service
        
    except Exception as e:
        logger.error(f"❌ RAG 서비스 초기화 실패: {e}")
        return None


def test_gemini_query_processing(service: LangChainRAGService):
    """Google AI 모델 쿼리 처리 테스트"""
    # 서비스에서 모델명 가져오기
    stats = service.get_service_statistics()
    model_name = stats['llm_model']
    logger.info(f"=== {model_name} 쿼리 처리 테스트 ===")
    
    test_queries = [
        {
            "query": "계약서에서 중요한 조항은 무엇인가요?",
            "template_type": "contract_review",
            "description": f"계약서 검토 ({model_name})"
        },
        {
            "query": "민법 제1조의 내용을 설명해주세요.",
            "template_type": "legal_qa",
            "description": f"법률 Q&A ({model_name})"
        },
        {
            "query": "최근 판례의 법적 시사점을 분석해주세요.",
            "template_type": "legal_analysis",
            "description": f"법률 분석 ({model_name})"
        }
    ]
    
    session_id = f"{model_name.replace('-', '_')}_demo_session"
    
    for i, test_case in enumerate(test_queries, 1):
        logger.info(f"📝 테스트 {i}: {test_case['description']}")
        
        try:
            result = service.process_query(
                query=test_case["query"],
                session_id=session_id,
                template_type=test_case["template_type"]
            )
            
            logger.info(f"✅ 쿼리 처리 성공")
            logger.info(f"   - 응답 시간: {result.response_time:.2f}초")
            logger.info(f"   - 신뢰도: {result.confidence:.2f}")
            logger.info(f"   - 토큰 사용량: {result.tokens_used}")
            logger.info(f"   - 검색 문서 수: {len(result.retrieved_docs)}")
            logger.info(f"   - 추적 ID: {result.trace_id}")
            logger.info(f"   - 답변 미리보기: {result.answer[:150]}...")
            
        except Exception as e:
            logger.error(f"❌ 쿼리 처리 실패: {e}")
        
        logger.info("")  # 빈 줄


def test_gemini_performance(service: LangChainRAGService):
    """Google AI 모델 성능 테스트"""
    # 서비스에서 모델명 가져오기
    stats = service.get_service_statistics()
    model_name = stats['llm_model']
    logger.info(f"=== {model_name} 성능 테스트 ===")
    
    try:
        # 서비스 통계
        logger.info(f"📊 {model_name} 성능 통계:")
        logger.info(f"   - 총 쿼리 수: {stats['rag_stats']['total_queries']}")
        logger.info(f"   - 평균 응답 시간: {stats['rag_stats']['avg_response_time']:.2f}초")
        logger.info(f"   - 평균 신뢰도: {stats['rag_stats']['avg_confidence']:.2f}")
        
        # 생성기 통계
        generator_stats = stats['generator_stats']
        logger.info(f"🤖 {model_name} 생성기 통계:")
        logger.info(f"   - 총 토큰 사용량: {generator_stats['total_tokens']}")
        logger.info(f"   - 평균 응답 시간: {generator_stats['avg_response_time']:.2f}초")
        logger.info(f"   - 사용 가능한 템플릿: {generator_stats['templates_available']}")
        
    except Exception as e:
        logger.error(f"❌ 성능 테스트 실패: {e}")


def main():
    """메인 함수"""
    # 환경 변수에서 모델명 가져오기
    model_name = os.getenv('LLM_MODEL', 'gemini-pro')
    logger.info(f"🚀 {model_name} RAG 시스템 테스트 시작")
    logger.info("=" * 60)
    
    # 1. 환경 설정
    if not setup_gemini_environment():
        logger.error("❌ 환경 설정 실패로 인해 테스트를 종료합니다.")
        return
    
    # 2. 설정 테스트
    config = test_gemini_configuration()
    if not config:
        logger.error("❌ 설정 테스트 실패로 인해 테스트를 종료합니다.")
        return
    
    # 3. RAG 서비스 테스트
    service = test_gemini_rag_service(config)
    if not service:
        logger.error("❌ RAG 서비스 테스트 실패로 인해 테스트를 종료합니다.")
        return
    
    # 4. 쿼리 처리 테스트
    test_gemini_query_processing(service)
    
    # 5. 성능 테스트
    test_gemini_performance(service)
    
    logger.info("=" * 60)
    logger.info(f"✅ {model_name} RAG 시스템 테스트 완료")
    logger.info("")
    logger.info("📚 추가 정보:")
    logger.info("   - Google AI Studio: https://makersuite.google.com/app/apikey")
    logger.info("   - Gemini Pro 문서: https://ai.google.dev/docs")
    logger.info("   - LangChain Google 통합: https://python.langchain.com/docs/integrations/llms/google_vertex_ai")
    logger.info("")
    logger.info("🔧 다음 단계:")
    logger.info("   1. Google AI Studio에서 API 키 발급 (https://makersuite.google.com/app/apikey)")
    logger.info("   2. .env 파일에서 GOOGLE_API_KEY=your-actual-api-key로 설정")
    logger.info("   3. 필요시 .env 파일에서 다른 설정 조정 (LLM_MODEL, LLM_TEMPERATURE 등)")
    logger.info("   4. 벡터 데이터베이스 구축")
    logger.info(f"   5. 프로덕션 환경에서 {model_name} 활용")
    logger.info("")
    logger.info("📝 .env 파일 설정 예시:")
    logger.info("   LLM_PROVIDER=google")
    logger.info(f"   LLM_MODEL={model_name}")
    logger.info("   GOOGLE_API_KEY=your-actual-google-api-key")
    logger.info("   LANGFUSE_ENABLED=false")


if __name__ == "__main__":
    main()
