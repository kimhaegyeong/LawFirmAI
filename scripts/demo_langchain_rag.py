#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain RAG Demo Script
LangChain 기반 RAG 시스템 데모 스크립트
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
from services.langfuse_client import LangfuseClient

# 로깅 설정
from utils.safe_logging import setup_script_logging
logger = setup_script_logging("demo_langchain_rag")


def setup_demo_environment():
    """데모 환경 설정"""
    # 환경 변수 설정 (데모용)
    os.environ.setdefault('LANGFUSE_ENABLED', 'false')
    os.environ.setdefault('LLM_PROVIDER', 'openai')
    os.environ.setdefault('LLM_MODEL', 'gpt-3.5-turbo')
    os.environ.setdefault('VECTOR_STORE_TYPE', 'faiss')
    os.environ.setdefault('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    logger.info("데모 환경 설정 완료")
    logger.info("참고: sqlite3는 Python 내장 모듈이므로 별도 설치 불필요")
    logger.info("지원 LLM: OpenAI, Anthropic, Google Gemini Pro, 로컬 모델")


def test_configuration():
    """설정 테스트"""
    logger.info("=== 설정 테스트 ===")
    
    try:
        config = LangChainConfig.from_env()
        logger.info(f"✅ 설정 로드 성공")
        logger.info(f"   - 벡터 저장소: {config.vector_store_type.value}")
        logger.info(f"   - 임베딩 모델: {config.embedding_model}")
        logger.info(f"   - LLM 제공자: {config.llm_provider.value}")
        logger.info(f"   - LLM 모델: {config.llm_model}")
        logger.info(f"   - Langfuse 활성화: {config.langfuse_enabled}")
        
        # 설정 유효성 검사
        errors = config.validate()
        if errors:
            logger.warning(f"⚠️ 설정 오류: {errors}")
        else:
            logger.info("✅ 설정 유효성 검사 통과")
            
        return config
        
    except Exception as e:
        logger.error(f"❌ 설정 로드 실패: {e}")
        return None


def test_langfuse_client(config: LangChainConfig):
    """Langfuse 클라이언트 테스트"""
    logger.info("=== Langfuse 클라이언트 테스트 ===")
    
    try:
        client = LangfuseClient(config)
        
        if client.is_enabled():
            logger.info("✅ Langfuse 클라이언트 활성화")
            trace_id = client.get_current_trace_id()
            logger.info(f"   - 현재 추적 ID: {trace_id}")
        else:
            logger.info("ℹ️ Langfuse 클라이언트 비활성화 (정상)")
            
        return client
        
    except Exception as e:
        logger.error(f"❌ Langfuse 클라이언트 초기화 실패: {e}")
        return None


def test_rag_service(config: LangChainConfig):
    """RAG 서비스 테스트"""
    logger.info("=== RAG 서비스 테스트 ===")
    
    try:
        service = LangChainRAGService(config)
        logger.info("✅ RAG 서비스 초기화 성공")
        
        # 서비스 통계 조회
        stats = service.get_service_statistics()
        logger.info("📊 서비스 통계:")
        logger.info(f"   - LangChain 사용 가능: {stats['langchain_available']}")
        logger.info(f"   - 임베딩 모델: {stats['embeddings_model']}")
        logger.info(f"   - LLM 모델: {stats['llm_model']}")
        logger.info(f"   - Langfuse 활성화: {stats['langfuse_enabled']}")
        
        # 설정 유효성 검사
        errors = service.validate_configuration()
        if errors:
            logger.warning(f"⚠️ 서비스 설정 오류: {errors}")
        else:
            logger.info("✅ 서비스 설정 유효성 검사 통과")
            
        return service
        
    except Exception as e:
        logger.error(f"❌ RAG 서비스 초기화 실패: {e}")
        return None


def test_query_processing(service: LangChainRAGService):
    """쿼리 처리 테스트"""
    logger.info("=== 쿼리 처리 테스트 ===")
    
    test_queries = [
        {
            "query": "계약서 검토 요청",
            "template_type": "contract_review",
            "description": "계약서 검토 템플릿 테스트"
        },
        {
            "query": "민법 제1조는 무엇을 규정하고 있나요?",
            "template_type": "legal_qa",
            "description": "법률 Q&A 템플릿 테스트"
        },
        {
            "query": "최근 판례 분석 요청",
            "template_type": "legal_analysis",
            "description": "법률 분석 템플릿 테스트"
        }
    ]
    
    session_id = "demo-session-1"
    
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
            logger.info(f"   - 답변 미리보기: {result.answer[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ 쿼리 처리 실패: {e}")
        
        logger.info("")  # 빈 줄


def test_session_management(service: LangChainRAGService):
    """세션 관리 테스트"""
    logger.info("=== 세션 관리 테스트 ===")
    
    try:
        session_id = "demo-session-2"
        
        # 첫 번째 쿼리
        result1 = service.process_query(
            query="첫 번째 질문",
            session_id=session_id
        )
        logger.info(f"✅ 첫 번째 쿼리 처리: {result1.response_time:.2f}초")
        
        # 두 번째 쿼리 (같은 세션)
        result2 = service.process_query(
            query="두 번째 질문",
            session_id=session_id
        )
        logger.info(f"✅ 두 번째 쿼리 처리: {result2.response_time:.2f}초")
        
        # 세션 컨텍스트 확인
        session_context = service.context_manager.get_session_context(session_id)
        logger.info(f"📋 세션 컨텍스트 길이: {len(session_context)}자")
        
        # 세션 삭제
        success = service.clear_session(session_id)
        if success:
            logger.info("✅ 세션 삭제 성공")
        else:
            logger.warning("⚠️ 세션 삭제 실패")
            
    except Exception as e:
        logger.error(f"❌ 세션 관리 테스트 실패: {e}")


def test_performance_monitoring(service: LangChainRAGService):
    """성능 모니터링 테스트"""
    logger.info("=== 성능 모니터링 테스트 ===")
    
    try:
        # 서비스 통계
        stats = service.get_service_statistics()
        logger.info("📊 서비스 통계:")
        logger.info(f"   - 총 쿼리 수: {stats['rag_stats']['total_queries']}")
        logger.info(f"   - 평균 응답 시간: {stats['rag_stats']['avg_response_time']:.2f}초")
        logger.info(f"   - 평균 신뢰도: {stats['rag_stats']['avg_confidence']:.2f}")
        
        # 컨텍스트 통계
        context_stats = stats['context_stats']
        logger.info("📋 컨텍스트 통계:")
        logger.info(f"   - 활성 세션 수: {context_stats['active_sessions']}")
        logger.info(f"   - 총 컨텍스트 수: {context_stats['total_contexts']}")
        logger.info(f"   - 캐시 적중률: {context_stats['cache_hit_ratio']:.2f}")
        
        # 생성기 통계
        generator_stats = stats['generator_stats']
        logger.info("🤖 생성기 통계:")
        logger.info(f"   - 총 토큰 사용량: {generator_stats['total_tokens']}")
        logger.info(f"   - 평균 응답 시간: {generator_stats['avg_response_time']:.2f}초")
        logger.info(f"   - 사용 가능한 템플릿: {generator_stats['templates_available']}")
        
    except Exception as e:
        logger.error(f"❌ 성능 모니터링 테스트 실패: {e}")


def main():
    """메인 함수"""
    logger.info("🚀 LangChain RAG 시스템 데모 시작")
    logger.info("=" * 50)
    
    # 1. 환경 설정
    setup_demo_environment()
    
    # 2. 설정 테스트
    config = test_configuration()
    if not config:
        logger.error("❌ 설정 테스트 실패로 인해 데모를 종료합니다.")
        return
    
    # 3. Langfuse 클라이언트 테스트
    client = test_langfuse_client(config)
    
    # 4. RAG 서비스 테스트
    service = test_rag_service(config)
    if not service:
        logger.error("❌ RAG 서비스 테스트 실패로 인해 데모를 종료합니다.")
        return
    
    # 5. 쿼리 처리 테스트
    test_query_processing(service)
    
    # 6. 세션 관리 테스트
    test_session_management(service)
    
    # 7. 성능 모니터링 테스트
    test_performance_monitoring(service)
    
    logger.info("=" * 50)
    logger.info("✅ LangChain RAG 시스템 데모 완료")
    logger.info("")
    logger.info("📚 추가 정보:")
    logger.info("   - 아키텍처 문서: docs/langchain_rag_architecture.md")
    logger.info("   - Langfuse 설정 가이드: docs/langfuse_setup_guide.md")
    logger.info("   - 환경 변수 예시: docs/langchain_env_example.md")
    logger.info("")
    logger.info("🔧 다음 단계:")
    logger.info("   1. .env 파일에 실제 API 키 설정")
    logger.info("   2. Langfuse 계정 생성 및 설정")
    logger.info("   3. 벡터 데이터베이스 구축")
    logger.info("   4. 프로덕션 환경 배포")


if __name__ == "__main__":
    main()
