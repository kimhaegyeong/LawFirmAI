#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 테스트 스크립트

검색 엔진이 제대로 작동하는지 확인하고, 검색 결과를 비교합니다.
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# 로깅 설정을 먼저 초기화
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def test_search_engine(db_path: str, use_external_index: bool = False, 
                      vector_store_version: str = None,
                      external_index_path: str = None):
    """검색 엔진 테스트"""
    try:
        from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
        
        logger.info("=" * 60)
        logger.info(f"Testing SemanticSearchEngineV2")
        logger.info(f"  DB Path: {db_path}")
        logger.info(f"  Use External Index: {use_external_index}")
        logger.info(f"  Vector Store Version: {vector_store_version}")
        logger.info(f"  External Index Path: {external_index_path}")
        logger.info("=" * 60)
        
        engine = SemanticSearchEngineV2(
            db_path=db_path,
            use_external_index=use_external_index,
            vector_store_version=vector_store_version,
            external_index_path=external_index_path
        )
        
        # 진단 정보 확인
        if hasattr(engine, 'diagnose'):
            diagnosis = engine.diagnose()
            logger.info(f"Diagnosis: {diagnosis}")
        
        # 테스트 쿼리
        test_queries = [
            "계약 해제",
            "손해배상",
            "임대차보호법",
            "전세금 반환"
        ]
        
        results_summary = []
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*60}")
            
            try:
                results = engine.search(query, k=5, similarity_threshold=0.5)
                
                logger.info(f"Found {len(results)} results")
                
                if results:
                    logger.info("\nTop 3 results:")
                    for i, result in enumerate(results[:3], 1):
                        logger.info(f"\n  {i}. Score: {result.get('score', 0):.4f}")
                        logger.info(f"     Type: {result.get('type', 'N/A')}")
                        logger.info(f"     Source: {result.get('source', 'N/A')}")
                        text_preview = result.get('text', '')[:100]
                        logger.info(f"     Text: {text_preview}...")
                    
                    results_summary.append({
                        'query': query,
                        'count': len(results),
                        'top_score': results[0].get('score', 0) if results else 0
                    })
                else:
                    logger.warning("No results found")
                    results_summary.append({
                        'query': query,
                        'count': 0,
                        'top_score': 0
                    })
                    
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}", exc_info=True)
                results_summary.append({
                    'query': query,
                    'count': 0,
                    'top_score': 0,
                    'error': str(e)
                })
        
        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        logger.info("=" * 60)
        for summary in results_summary:
            if 'error' in summary:
                logger.error(f"  {summary['query']}: ERROR - {summary['error']}")
            else:
                logger.info(f"  {summary['query']}: {summary['count']} results (top score: {summary['top_score']:.4f})")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Failed to test search engine: {e}", exc_info=True)
        return None


def compare_search_results(db_path: str, external_index_path: str = None):
    """DB 기반과 외부 인덱스 기반 검색 결과 비교"""
    logger.info("\n" + "=" * 60)
    logger.info("Comparing DB-based vs External Index-based Search")
    logger.info("=" * 60)
    
    test_queries = ["계약 해제", "손해배상"]
    
    # DB 기반 검색
    logger.info("\n--- DB-based Search ---")
    db_results = test_search_engine(db_path, use_external_index=False)
    
    # 외부 인덱스 기반 검색
    if external_index_path and Path(external_index_path).exists():
        logger.info("\n--- External Index-based Search ---")
        external_results = test_search_engine(
            db_path, 
            use_external_index=True,
            external_index_path=external_index_path
        )
        
        # 결과 비교
        if db_results and external_results:
            logger.info("\n" + "=" * 60)
            logger.info("Comparison:")
            logger.info("=" * 60)
            for db_summary, ext_summary in zip(db_results, external_results):
                query = db_summary['query']
                db_count = db_summary['count']
                ext_count = ext_summary['count']
                logger.info(f"\nQuery: {query}")
                logger.info(f"  DB-based: {db_count} results")
                logger.info(f"  External: {ext_count} results")
                logger.info(f"  Difference: {abs(db_count - ext_count)}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 테스트 스크립트")
    parser.add_argument('--db-path', 
                        default='data/lawfirm_v2.db',
                        help='데이터베이스 경로')
    parser.add_argument('--use-external-index', 
                        action='store_true',
                        help='외부 인덱스 사용')
    parser.add_argument('--vector-store-version', 
                        default=None,
                        help='벡터스토어 버전')
    parser.add_argument('--external-index-path', 
                        default=None,
                        help='외부 인덱스 경로')
    parser.add_argument('--compare', 
                        action='store_true',
                        help='DB 기반과 외부 인덱스 기반 결과 비교')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_search_results(
            args.db_path,
            args.external_index_path or 'data/embeddings/ml_enhanced_ko_sroberta_precedents'
        )
    else:
        test_search_engine(
            args.db_path,
            use_external_index=args.use_external_index,
            vector_store_version=args.vector_store_version,
            external_index_path=args.external_index_path
        )


if __name__ == "__main__":
    main()

