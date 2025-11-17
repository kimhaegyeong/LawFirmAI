#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로덕션 인덱스 및 최적 파라미터 실제 검색 테스트

실제 검색 쿼리를 실행하여 최적 파라미터와 프로덕션 인덱스가 제대로 사용되는지 검증합니다.
"""

import sys
import os
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_semantic_search_with_optimized_params():
    """최적 파라미터를 사용한 실제 검색 테스트"""
    logger.info("=" * 80)
    logger.info("실제 검색 테스트 - 최적 파라미터 적용 확인")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        from lawfirm_langgraph.core.utils.config import Config
        from lawfirm_langgraph.core.search.optimizers.query_enhancer import QueryEnhancer
        
        config = Config()
        db_path = config.database_path
        
        logger.info(f"데이터베이스 경로: {db_path}")
        logger.info(f"외부 인덱스 사용: {config.use_external_vector_store}")
        logger.info(f"인덱스 경로: {config.external_vector_store_base_path}")
        
        search_engine = SemanticSearchEngineV2(
            db_path=db_path,
            use_external_index=config.use_external_vector_store,
            vector_store_version=config.vector_store_version,
            external_index_path=config.external_vector_store_base_path
        )
        
        if not search_engine.index:
            logger.error("❌ FAISS 인덱스가 로드되지 않았습니다")
            return False
        
        logger.info(f"✅ FAISS 인덱스 로드 성공: {search_engine.index.ntotal}개 벡터")
        
        class MockLLM:
            pass
        
        class MockTermIntegrator:
            pass
        
        class MockConfig:
            similarity_threshold = 0.7
        
        query_enhancer = QueryEnhancer(
            llm=MockLLM(),
            llm_fast=None,
            term_integrator=MockTermIntegrator(),
            config=MockConfig()
        )
        
        if not query_enhancer.optimized_params:
            logger.warning("⚠️ 최적 파라미터가 로드되지 않았습니다")
            return False
        
        logger.info("✅ 최적 파라미터 로드 확인:")
        for key, value in query_enhancer.optimized_params.items():
            logger.info(f"   - {key}: {value}")
        
        search_params = query_enhancer.determine_search_parameters(
            query_type="precedent_search",
            query_complexity=50,
            keyword_count=5,
            is_retry=False
        )
        
        logger.info("\n검색 파라미터 결정 결과:")
        logger.info(f"   - semantic_k: {search_params.get('semantic_k')}")
        logger.info(f"   - min_relevance: {search_params.get('min_relevance')}")
        logger.info(f"   - use_reranking: {search_params.get('use_reranking', False)}")
        logger.info(f"   - query_enhancement: {search_params.get('query_enhancement', False)}")
        
        expected_top_k = query_enhancer.optimized_params.get("top_k", 15)
        if search_params.get("semantic_k") >= expected_top_k:
            logger.info(f"✅ 최적 파라미터 반영 확인 (기대: {expected_top_k}, 실제: {search_params.get('semantic_k')})")
        else:
            logger.warning(f"⚠️ 최적 파라미터가 완전히 반영되지 않았습니다 (기대: {expected_top_k}, 실제: {search_params.get('semantic_k')})")
        
        test_query = "계약 해지 사유"
        logger.info(f"\n실제 검색 테스트: '{test_query}'")
        
        results = search_engine.search(
            query=test_query,
            k=search_params.get("semantic_k", 15),
            similarity_threshold=search_params.get("min_relevance", 0.9),
            min_results=5
        )
        
        logger.info(f"✅ 검색 결과: {len(results)}개 문서 반환")
        
        if len(results) > 0:
            logger.info("상위 3개 결과:")
            for i, result in enumerate(results[:3], 1):
                logger.info(f"   {i}. {result.get('content', '')[:100]}...")
            return True
        else:
            logger.warning("⚠️ 검색 결과가 없습니다")
            return False
        
    except Exception as e:
        logger.error(f"❌ 검색 테스트 실패: {e}", exc_info=True)
        return False


def test_langgraph_workflow_search():
    """LangGraph 워크플로우를 통한 실제 검색 테스트"""
    logger.info("=" * 80)
    logger.info("LangGraph 워크플로우 검색 테스트")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        logger.info("LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        logger.info("✅ 서비스 초기화 완료")
        
        if service.workflow and service.workflow.semantic_search:
            if service.workflow.semantic_search.index:
                logger.info(f"✅ 프로덕션 인덱스 로드 확인: {service.workflow.semantic_search.index.ntotal}개 벡터")
            else:
                logger.warning("⚠️ 인덱스가 로드되지 않았습니다")
        
        if service.workflow and service.workflow.query_enhancer:
            if service.workflow.query_enhancer.optimized_params:
                logger.info("✅ 최적 파라미터 로드 확인")
                logger.info(f"   - top_k: {service.workflow.query_enhancer.optimized_params.get('top_k')}")
                logger.info(f"   - similarity_threshold: {service.workflow.query_enhancer.optimized_params.get('similarity_threshold')}")
            else:
                logger.warning("⚠️ 최적 파라미터가 로드되지 않았습니다")
        
        test_query = "임대차 계약 해지 조건"
        logger.info(f"\n실제 질의 테스트: '{test_query}'")
        logger.info("(이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        import asyncio
        result = asyncio.run(service.process_query(
            query=test_query,
            session_id="production_test",
            enable_checkpoint=False
        ))
        
        if result and result.get("answer"):
            logger.info("✅ 질의 처리 성공")
            logger.info(f"답변 길이: {len(result.get('answer', ''))}자")
            if result.get("sources"):
                logger.info(f"참조 문서 수: {len(result.get('sources', []))}")
            return True
        else:
            logger.warning("⚠️ 질의 처리 결과가 없습니다")
            return False
        
    except Exception as e:
        logger.error(f"❌ LangGraph 워크플로우 테스트 실패: {e}", exc_info=True)
        return False


def main():
    """메인 테스트 함수"""
    logger.info("=" * 80)
    logger.info("프로덕션 인덱스 및 최적 파라미터 실제 검색 테스트 시작")
    logger.info("=" * 80)
    
    results = {
        "SemanticSearchEngine 검색": test_semantic_search_with_optimized_params(),
        "LangGraph 워크플로우 검색": test_langgraph_workflow_search()
    }
    
    logger.info("=" * 80)
    logger.info("테스트 결과 요약")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("=" * 80)
        logger.info("✅ 모든 검색 테스트 통과!")
        logger.info("=" * 80)
        logger.info("프로덕션 인덱스와 최적 파라미터가 정상적으로 작동하고 있습니다.")
    else:
        logger.error("=" * 80)
        logger.error("❌ 일부 검색 테스트 실패")
        logger.error("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

