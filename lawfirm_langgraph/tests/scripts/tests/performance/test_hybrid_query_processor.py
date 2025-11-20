# -*- coding: utf-8 -*-
"""
HybridQueryProcessor 테스트 스크립트
HuggingFace + LLM 하이브리드 쿼리 프로세서 테스트

Usage:
    python lawfirm_langgraph/tests/scripts/test_hybrid_query_processor.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
    scripts_dir = script_dir.parent
    tests_dir = scripts_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 테스트 쿼리
TEST_QUERIES = [
    {
        "query": "계약 해지 사유에 대해 알려주세요",
        "query_type": "legal_advice",
        "legal_field": "",
        "complexity": "moderate"
    },
    {
        "query": "손해배상 청구 요건은 무엇인가요?",
        "query_type": "legal_advice",
        "legal_field": "민사법",
        "complexity": "moderate"
    },
    {
        "query": "민법 제750조",
        "query_type": "statute",
        "legal_field": "민사법",
        "complexity": "simple"
    }
]


def test_hybrid_query_processor():
    """HybridQueryProcessor 테스트"""
    print("=" * 80)
    print("HybridQueryProcessor 테스트 시작")
    print("=" * 80)
    
    try:
        # 컴포넌트 초기화
        from core.agents.keyword_extractor import KeywordExtractor
        from core.search.optimizers.keyword_mapper import LegalKeywordMapper
        try:
            from core.processing.integration.term_integration_system import TermIntegrator
        except ImportError:
            try:
                from core.services.term_integration_system import TermIntegrator
            except ImportError:
                TermIntegrator = None
        from core.workflow.initializers.llm_initializer import LLMInitializer
        from core.utils.config import Config
        from core.utils.langgraph_config import LangGraphConfig
        from core.search.optimizers.hybrid_query_processor import HybridQueryProcessor
        
        # 설정 로드
        try:
            config = LangGraphConfig.from_env()
        except:
            config = Config()
        
        # KeywordExtractor 초기화
        keyword_extractor = KeywordExtractor(use_morphology=True, logger_instance=logger)
        logger.info("✅ KeywordExtractor initialized")
        
        # TermIntegrator 초기화
        if TermIntegrator:
            term_integrator = TermIntegrator()
            logger.info("✅ TermIntegrator initialized")
        else:
            term_integrator = None
            logger.warning("⚠️ TermIntegrator not available, using None")
        
        # LLM 초기화
        try:
            llm_initializer = LLMInitializer(config=config)
            llm = llm_initializer.initialize_llm()
            logger.info("✅ LLM initialized")
        except Exception as e:
            logger.warning(f"⚠️ LLM initialization failed: {e}, using None")
            llm = None
        
        # HybridQueryProcessor 초기화
        embedding_model_name = getattr(config, 'embedding_model', None)
        hybrid_processor = HybridQueryProcessor(
            keyword_extractor=keyword_extractor,
            term_integrator=term_integrator,
            llm=llm,
            embedding_model_name=embedding_model_name,
            logger=logger
        )
        logger.info("✅ HybridQueryProcessor initialized")
        
        # 각 테스트 쿼리 실행
        results = []
        for i, test_query in enumerate(TEST_QUERIES, 1):
            print(f"\n{'=' * 80}")
            print(f"테스트 {i}/{len(TEST_QUERIES)}: {test_query['query']}")
            print(f"{'=' * 80}")
            
            try:
                # 키워드 추출
                extracted_keywords = keyword_extractor.extract_keywords(
                    test_query["query"],
                    max_keywords=10,
                    prefer_morphology=True
                )
                print(f"📝 추출된 키워드: {extracted_keywords}")
                
                # HybridQueryProcessor 실행
                optimized_queries, cache_hit = hybrid_processor.process_query_hybrid(
                    query=test_query["query"],
                    search_query=test_query["query"],
                    query_type=test_query["query_type"],
                    extracted_keywords=extracted_keywords,
                    legal_field=test_query["legal_field"],
                    complexity=test_query["complexity"],
                    is_retry=False
                )
                
                # 결과 출력
                print(f"\n✅ 쿼리 최적화 완료:")
                print(f"  - Semantic Query: {optimized_queries.get('semantic_query', 'N/A')}")
                print(f"  - Keyword Queries: {len(optimized_queries.get('keyword_queries', []))}개")
                print(f"  - Expanded Keywords: {len(optimized_queries.get('expanded_keywords', []))}개")
                print(f"  - Multi Queries: {len(optimized_queries.get('multi_queries', []))}개")
                print(f"  - HF Models Used: {optimized_queries.get('hf_models_used', False)}")
                print(f"  - LLM Enhanced: {optimized_queries.get('llm_enhanced', False)}")
                
                if optimized_queries.get('multi_queries'):
                    print(f"  - Multi Queries 내용:")
                    for j, mq in enumerate(optimized_queries['multi_queries'][:3], 1):
                        print(f"    {j}. {mq}")
                
                results.append({
                    "test_query": test_query,
                    "optimized_queries": optimized_queries,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"❌ 테스트 쿼리 실행 실패: {e}", exc_info=True)
                results.append({
                    "test_query": test_query,
                    "error": str(e),
                    "success": False
                })
        
        # 결과 요약
        print(f"\n{'=' * 80}")
        print("테스트 결과 요약")
        print(f"{'=' * 80}")
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"✅ 성공: {success_count}/{len(TEST_QUERIES)}")
        print(f"❌ 실패: {len(TEST_QUERIES) - success_count}/{len(TEST_QUERIES)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 테스트 초기화 실패: {e}", exc_info=True)
        return None


def test_individual_components():
    """개별 컴포넌트 테스트"""
    print("\n" + "=" * 80)
    print("개별 컴포넌트 테스트")
    print("=" * 80)
    
    try:
        from core.search.optimizers.legal_query_analyzer import LegalQueryAnalyzer
        from core.search.optimizers.legal_keyword_expander import LegalKeywordExpander
        from core.search.optimizers.legal_query_optimizer import LegalQueryOptimizer
        from core.search.optimizers.legal_query_validator import LegalQueryValidator
        from core.agents.keyword_extractor import KeywordExtractor
        
        keyword_extractor = KeywordExtractor(use_morphology=True, logger_instance=logger)
        
        # LegalQueryAnalyzer 테스트
        print("\n1. LegalQueryAnalyzer 테스트")
        analyzer = LegalQueryAnalyzer(
            keyword_extractor=keyword_extractor,
            logger=logger
        )
        analysis_result = analyzer.analyze_query(
            query="계약 해지 사유에 대해 알려주세요",
            query_type="legal_advice",
            legal_field=""
        )
        print(f"  ✅ Core Keywords: {analysis_result.get('core_keywords', [])}")
        print(f"  ✅ Query Intent: {analysis_result.get('query_intent', 'N/A')}")
        print(f"  ✅ Key Concepts: {analysis_result.get('key_concepts', [])}")
        
        # LegalKeywordExpander 테스트
        print("\n2. LegalKeywordExpander 테스트")
        expander = LegalKeywordExpander(logger=logger)
        expansion_result = expander.expand_keywords(
            query="계약 해지 사유에 대해 알려주세요",
            core_keywords=analysis_result.get('core_keywords', []),
            extracted_keywords=analysis_result.get('core_keywords', []),
            legal_field=""
        )
        print(f"  ✅ Expanded Keywords: {len(expansion_result.get('expanded_keywords', []))}개")
        print(f"  ✅ Synonyms: {len(expansion_result.get('synonyms', []))}개")
        
        # LegalQueryOptimizer 테스트
        print("\n3. LegalQueryOptimizer 테스트")
        optimizer = LegalQueryOptimizer(logger=logger)
        optimization_result = optimizer.optimize_query(
            query="계약 해지 사유에 대해 알려주세요",
            core_keywords=analysis_result.get('core_keywords', []),
            expanded_keywords=expansion_result.get('expanded_keywords', []),
            query_type="legal_advice"
        )
        print(f"  ✅ Semantic Query: {optimization_result.get('semantic_query', 'N/A')}")
        print(f"  ✅ Keyword Query: {optimization_result.get('keyword_query', 'N/A')}")
        print(f"  ✅ Quality Score: {optimization_result.get('quality_score', 0.0):.2f}")
        
        # LegalQueryValidator 테스트
        print("\n4. LegalQueryValidator 테스트")
        validator = LegalQueryValidator(logger=logger)
        validation_result = validator.validate_query(
            optimized_queries=optimization_result,
            original_query="계약 해지 사유에 대해 알려주세요"
        )
        print(f"  ✅ Is Valid: {validation_result.get('is_valid', False)}")
        print(f"  ✅ Quality Score: {validation_result.get('quality_score', 0.0):.2f}")
        print(f"  ✅ Improvements: {validation_result.get('improvements', [])}")
        
        print("\n✅ 모든 개별 컴포넌트 테스트 완료")
        
    except Exception as e:
        logger.error(f"❌ 개별 컴포넌트 테스트 실패: {e}", exc_info=True)


if __name__ == "__main__":
    print("🚀 HybridQueryProcessor 테스트 시작\n")
    
    # 개별 컴포넌트 테스트
    test_individual_components()
    
    # 통합 테스트
    results = test_hybrid_query_processor()
    
    if results:
        print("\n✅ 모든 테스트 완료")
    else:
        print("\n❌ 테스트 실패")
        sys.exit(1)

