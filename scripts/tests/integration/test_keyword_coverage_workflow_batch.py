#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keyword Coverage 개선 효과 통합 테스트 (실제 워크플로우 배치 테스트)
run_query_test.py를 활용하여 여러 쿼리를 테스트하고 Keyword Coverage를 측정합니다.
"""

import sys
import os
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging

# 안전한 로깅 설정 (버퍼 분리 오류 방지)
def safe_log(level, msg, *args, **kwargs):
    """안전한 로깅 (예외 처리)"""
    try:
        logger = logging.getLogger(__name__)
        getattr(logger, level)(msg, *args, **kwargs)
    except (ValueError, AttributeError, OSError) as e:
        if "detached" not in str(e).lower():
            print(f"[{level.upper()}] {msg}")
    except Exception:
        print(f"[{level.upper()}] {msg}")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 안전한 로거 래퍼
class SafeLogger:
    def info(self, msg, *args, **kwargs):
        safe_log('info', msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        safe_log('error', msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        safe_log('warning', msg, *args, **kwargs)

logger = SafeLogger()


async def test_query_with_coverage(query: str) -> Dict[str, Any]:
    """단일 쿼리 테스트 및 Keyword Coverage 측정"""
    try:
        from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
        
        logger.info(f"\n{'='*80}")
        logger.info(f"테스트 쿼리: {query}")
        logger.info(f"{'='*80}")
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        
        service = LangGraphWorkflowService(config)
        
        result = await service.process_query(
            query=query,
            session_id=f"test_{int(datetime.now().timestamp())}",
            enable_checkpoint=False
        )
        
        # 검색 결과 추출
        retrieved_docs = result.get("retrieved_docs", [])
        search_results = []
        
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                search_results.append({
                    "content": doc.get("content", doc.get("text", "")),
                    "text": doc.get("text", doc.get("content", "")),
                    "score": doc.get("score", doc.get("similarity", 0.0)),
                    "metadata": doc.get("metadata", {})
                })
        
        # 키워드 추출
        extracted_keywords = []
        if "metadata" in result:
            metadata = result.get("metadata", {})
            extracted_keywords = metadata.get("extracted_keywords", [])
        
        if not extracted_keywords:
            from lawfirm_langgraph.core.agents.keyword_extractor import KeywordExtractor
            extractor = KeywordExtractor(use_morphology=True)
            extracted_keywords = extractor.extract_keywords(query, max_keywords=15, prefer_morphology=True)
        
        # 검색 품질 평가
        from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
        ranker = ResultRanker(use_cross_encoder=False)
        
        metrics = ranker.evaluate_search_quality(
            query=query,
            results=search_results,
            query_type=result.get("query_type", ""),
            extracted_keywords=extracted_keywords
        )
        
        test_result = {
            "query": query,
            "extracted_keywords": extracted_keywords,
            "search_results_count": len(search_results),
            "retrieved_docs_count": len(retrieved_docs),
            "keyword_coverage": metrics.get("keyword_coverage", 0.0),
            "avg_relevance": metrics.get("avg_relevance", 0.0),
            "diversity_score": metrics.get("diversity_score", 0.0),
            "answer_length": len(str(result.get("answer", ""))),
            "processing_time": result.get("processing_time", 0.0)
        }
        
        logger.info(f"   ✅ Keyword Coverage: {test_result['keyword_coverage']:.3f}")
        logger.info(f"   ✅ Avg Relevance: {test_result['avg_relevance']:.3f}")
        logger.info(f"   ✅ Diversity Score: {test_result['diversity_score']:.3f}")
        logger.info(f"   ✅ 검색 결과 수: {test_result['search_results_count']}")
        
        return test_result
        
    except Exception as e:
        error_msg = f"쿼리 테스트 실패: {e}"
        try:
            logger.error(error_msg, exc_info=True)
        except Exception:
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        return {
            "query": query,
            "error": str(e),
            "keyword_coverage": 0.0
        }


async def run_batch_tests():
    """배치 테스트 실행"""
    test_queries = [
        "계약 해지 사유에 대해 알려주세요",
        "손해배상 청구권의 소멸시효는?",
        "불법행위로 인한 손해배상 판례를 찾아주세요"
    ]
    
    logger.info("\n" + "="*80)
    logger.info("Keyword Coverage 개선 효과 통합 테스트 (실제 워크플로우)")
    logger.info("="*80)
    
    all_results = {
        "test_date": datetime.now().isoformat(),
        "test_queries": test_queries,
        "results": [],
        "summary": {}
    }
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n[{i}/{len(test_queries)}] 테스트 진행 중...")
        result = await test_query_with_coverage(query)
        all_results["results"].append(result)
        
        if i < len(test_queries):
            await asyncio.sleep(2)
    
    # 요약 통계 계산
    successful_results = [r for r in all_results["results"] if "error" not in r]
    
    if successful_results:
        keyword_coverages = [r.get("keyword_coverage", 0.0) for r in successful_results]
        avg_relevances = [r.get("avg_relevance", 0.0) for r in successful_results]
        diversity_scores = [r.get("diversity_score", 0.0) for r in successful_results]
        
        all_results["summary"] = {
            "total_queries": len(test_queries),
            "successful_queries": len(successful_results),
            "avg_keyword_coverage": sum(keyword_coverages) / len(keyword_coverages),
            "avg_relevance": sum(avg_relevances) / len(avg_relevances),
            "avg_diversity_score": sum(diversity_scores) / len(diversity_scores),
            "min_keyword_coverage": min(keyword_coverages),
            "max_keyword_coverage": max(keyword_coverages)
        }
        
        logger.info("\n" + "="*80)
        logger.info("통합 테스트 결과 요약")
        logger.info("="*80)
        logger.info(f"총 쿼리 수: {all_results['summary']['total_queries']}")
        logger.info(f"성공한 쿼리 수: {all_results['summary']['successful_queries']}")
        logger.info(f"평균 Keyword Coverage: {all_results['summary']['avg_keyword_coverage']:.3f}")
        logger.info(f"평균 Relevance: {all_results['summary']['avg_relevance']:.3f}")
        logger.info(f"평균 Diversity Score: {all_results['summary']['avg_diversity_score']:.3f}")
        logger.info(f"Keyword Coverage 범위: {all_results['summary']['min_keyword_coverage']:.3f} ~ {all_results['summary']['max_keyword_coverage']:.3f}")
    
    # 결과 저장
    output_file = project_root / "docs" / "improvements" / "keyword_coverage_workflow_batch_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n테스트 결과가 저장되었습니다: {output_file}")
    
    return all_results


if __name__ == "__main__":
    try:
        results = asyncio.run(run_batch_tests())
        print("\n" + "="*80)
        print("통합 테스트 결과 (요약)")
        print("="*80)
        if results.get("summary"):
            summary = results["summary"]
            print(f"평균 Keyword Coverage: {summary.get('avg_keyword_coverage', 0.0):.3f}")
            print(f"평균 Relevance: {summary.get('avg_relevance', 0.0):.3f}")
            print(f"평균 Diversity Score: {summary.get('avg_diversity_score', 0.0):.3f}")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        sys.exit(1)

