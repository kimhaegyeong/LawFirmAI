#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로덕션 인덱스 및 최적 파라미터 통합 테스트

LangGraph에서 프로덕션 FAISS 인덱스와 최적 파라미터가 제대로 사용되는지 검증합니다.
"""

import sys
import os
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment_variables():
    """환경 변수 설정 확인"""
    logger.info("=" * 80)
    logger.info("1. 환경 변수 설정 확인")
    logger.info("=" * 80)
    
    try:
        from utils.env_loader import ensure_env_loaded
        ensure_env_loaded(project_root)
    except Exception as e:
        logger.debug(f"환경 변수 로더 사용 실패 (무시됨): {e}")
    
    required_vars = {
        "USE_EXTERNAL_VECTOR_STORE": os.getenv("USE_EXTERNAL_VECTOR_STORE"),
        "EXTERNAL_VECTOR_STORE_BASE_PATH": os.getenv("EXTERNAL_VECTOR_STORE_BASE_PATH"),
        "OPTIMIZED_SEARCH_PARAMS_PATH": os.getenv("OPTIMIZED_SEARCH_PARAMS_PATH")
    }
    
    all_set = True
    for var_name, var_value in required_vars.items():
        if var_value:
            logger.info(f"✅ {var_name}={var_value}")
        else:
            logger.warning(f"⚠️ {var_name}이 설정되지 않았습니다")
            if var_name == "OPTIMIZED_SEARCH_PARAMS_PATH":
                logger.info(f"   (기본값 사용: data/ml_config/optimized_search_params.json)")
            elif var_name == "USE_EXTERNAL_VECTOR_STORE":
                logger.warning(f"   (필수 환경 변수입니다)")
                all_set = False
            elif var_name == "EXTERNAL_VECTOR_STORE_BASE_PATH":
                logger.warning(f"   (필수 환경 변수입니다)")
                all_set = False
    
    return all_set


def test_production_index_exists():
    """프로덕션 인덱스 파일 존재 확인"""
    logger.info("=" * 80)
    logger.info("2. 프로덕션 인덱스 파일 확인")
    logger.info("=" * 80)
    
    index_path = os.getenv(
        "EXTERNAL_VECTOR_STORE_BASE_PATH",
        "data/vector_store/production-20251117-091619"
    )
    
    index_dir = Path(index_path)
    if not index_dir.is_absolute():
        index_dir = project_root / index_path
    
    logger.info(f"인덱스 경로: {index_dir}")
    
    required_files = [
        "index.faiss",
        "id_mapping.json",
        "version_info.json"
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = index_dir / file_name
        if file_path.exists():
            logger.info(f"✅ {file_name} 존재")
        else:
            logger.error(f"❌ {file_name} 없음: {file_path}")
            all_exist = False
    
    return all_exist


def test_optimized_params_exists():
    """최적 파라미터 파일 존재 확인"""
    logger.info("=" * 80)
    logger.info("3. 최적 파라미터 파일 확인")
    logger.info("=" * 80)
    
    params_path = os.getenv(
        "OPTIMIZED_SEARCH_PARAMS_PATH",
        "data/ml_config/optimized_search_params.json"
    )
    
    params_file = Path(params_path)
    if not params_file.is_absolute():
        params_file = project_root / params_path
    
    logger.info(f"파라미터 경로: {params_file}")
    
    if params_file.exists():
        logger.info("✅ 최적 파라미터 파일 존재")
        
        import json
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            search_params = config.get("search_parameters", {})
            if search_params:
                logger.info("✅ 최적 파라미터 내용:")
                for key, value in search_params.items():
                    logger.info(f"   - {key}: {value}")
                return True
            else:
                logger.error("❌ search_parameters가 없습니다")
                return False
        except Exception as e:
            logger.error(f"❌ 파일 읽기 실패: {e}")
            return False
    else:
        logger.error(f"❌ 최적 파라미터 파일 없음: {params_file}")
        return False


def test_semantic_search_engine():
    """SemanticSearchEngineV2 초기화 테스트"""
    logger.info("=" * 80)
    logger.info("4. SemanticSearchEngineV2 초기화 테스트")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        from lawfirm_langgraph.core.utils.config import Config
        
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
        
        if search_engine.index is not None:
            logger.info(f"✅ FAISS 인덱스 로드 성공: {search_engine.index.ntotal}개 벡터")
            return True
        else:
            logger.warning("⚠️ FAISS 인덱스가 로드되지 않았습니다 (폴백 모드)")
            return False
            
    except Exception as e:
        logger.error(f"❌ SemanticSearchEngineV2 초기화 실패: {e}", exc_info=True)
        return False


def test_query_enhancer():
    """QueryEnhancer 최적 파라미터 로드 테스트"""
    logger.info("=" * 80)
    logger.info("5. QueryEnhancer 최적 파라미터 로드 테스트")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.core.search.optimizers.query_enhancer import QueryEnhancer
        
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
        
        if query_enhancer.optimized_params:
            logger.info("✅ 최적 파라미터 로드 성공:")
            for key, value in query_enhancer.optimized_params.items():
                logger.info(f"   - {key}: {value}")
            
            search_params = query_enhancer.determine_search_parameters(
                query_type="precedent_search",
                query_complexity=50,
                keyword_count=5,
                is_retry=False
            )
            
            logger.info("검색 파라미터 결정 결과:")
            logger.info(f"   - semantic_k: {search_params.get('semantic_k')}")
            logger.info(f"   - min_relevance: {search_params.get('min_relevance')}")
            logger.info(f"   - use_reranking: {search_params.get('use_reranking', False)}")
            
            return True
        else:
            logger.warning("⚠️ 최적 파라미터가 로드되지 않았습니다")
            return False
            
    except Exception as e:
        logger.error(f"❌ QueryEnhancer 테스트 실패: {e}", exc_info=True)
        return False


def test_langgraph_workflow():
    """LangGraph 워크플로우 통합 테스트"""
    logger.info("=" * 80)
    logger.info("6. LangGraph 워크플로우 통합 테스트")
    logger.info("=" * 80)
    
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        logger.info("EnhancedLegalQuestionWorkflow 초기화 중...")
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        if workflow.semantic_search:
            logger.info("✅ SemanticSearchEngineV2 초기화 성공")
            if workflow.semantic_search.index:
                logger.info(f"   - 인덱스 벡터 수: {workflow.semantic_search.index.ntotal}")
            else:
                logger.warning("   - 인덱스가 로드되지 않았습니다")
        else:
            logger.error("❌ SemanticSearchEngineV2 초기화 실패")
            return False
        
        if workflow.query_enhancer:
            if workflow.query_enhancer.optimized_params:
                logger.info("✅ QueryEnhancer 최적 파라미터 로드 성공")
            else:
                logger.warning("⚠️ QueryEnhancer 최적 파라미터 미로드")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LangGraph 워크플로우 테스트 실패: {e}", exc_info=True)
        return False


def main():
    """메인 테스트 함수"""
    logger.info("=" * 80)
    logger.info("프로덕션 인덱스 및 최적 파라미터 통합 테스트 시작")
    logger.info("=" * 80)
    
    results = {
        "환경 변수": test_environment_variables(),
        "프로덕션 인덱스": test_production_index_exists(),
        "최적 파라미터": test_optimized_params_exists(),
        "SemanticSearchEngine": test_semantic_search_engine(),
        "QueryEnhancer": test_query_enhancer(),
        "LangGraph 워크플로우": test_langgraph_workflow()
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
        logger.info("✅ 모든 테스트 통과!")
        logger.info("=" * 80)
        logger.info("프로덕션 인덱스와 최적 파라미터가 정상적으로 통합되었습니다.")
    else:
        logger.error("=" * 80)
        logger.error("❌ 일부 테스트 실패")
        logger.error("=" * 80)
        logger.error("환경 변수 설정 및 파일 경로를 확인하세요.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

