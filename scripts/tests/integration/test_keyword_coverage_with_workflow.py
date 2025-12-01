#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keyword Coverage 개선 효과 통합 테스트 (실제 워크플로우 사용)
run_query_test.py를 활용하여 실제 워크플로우를 실행하고 Keyword Coverage를 측정합니다.
"""

import sys
import os
import logging
import time
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# lawfirm_langgraph 경로 추가
lawfirm_langgraph_dir = os.path.join(project_root, 'lawfirm_langgraph')
if os.path.exists(lawfirm_langgraph_dir):
    sys.path.insert(0, lawfirm_langgraph_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeywordCoverageWorkflowTester:
    """Keyword Coverage 개선 효과 통합 테스트 클래스 (실제 워크플로우 사용)"""
    
    def __init__(self):
        self.test_results = {}
        self.test_queries = [
            {
                "query": "계약 해지 사유에 대해 알려주세요",
                "query_type": "law_inquiry",
                "expected_keywords": ["계약", "해지", "사유"]
            },
            {
                "query": "손해배상 청구권의 소멸시효는?",
                "query_type": "legal_advice",
                "expected_keywords": ["손해배상", "청구권", "소멸시효"]
            },
            {
                "query": "불법행위로 인한 손해배상 판례를 찾아주세요",
                "query_type": "precedent_search",
                "expected_keywords": ["불법행위", "손해배상", "판례"]
            }
        ]
    
    async def test_workflow_with_keyword_coverage(self, query: str, query_type: str = "") -> Dict[str, Any]:
        """실제 워크플로우를 통한 검색 테스트 및 Keyword Coverage 측정"""
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
            from lawfirm_langgraph.langgraph_core.workflow.workflow_service import LangGraphWorkflowService
            
            logger.info(f"\n{'='*80}")
            logger.info(f"검색 쿼리: {query}")
            logger.info(f"질문 유형: {query_type}")
            logger.info(f"{'='*80}")
            
            # 설정 로드
            config = LangGraphConfig.from_env()
            config.enable_checkpoint = False
            
            # 서비스 초기화
            service = LangGraphWorkflowService(config)
            
            start_time = time.time()
            
            # 워크플로우 실행
            result = await service.process_query(
                query=query,
                session_id=f"test_{int(time.time())}",
                enable_checkpoint=False,
                use_astream_events=False
            )
            
            elapsed_time = time.time() - start_time
            
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
            
            # 키워드 추출 (워크플로우에서 추출된 키워드 확인)
            extracted_keywords = []
            if "metadata" in result:
                metadata = result.get("metadata", {})
                extracted_keywords = metadata.get("extracted_keywords", [])
            
            # 키워드가 없으면 직접 추출
            if not extracted_keywords:
                from lawfirm_langgraph.core.agents.keyword_extractor import KeywordExtractor
                extractor = KeywordExtractor(use_morphology=True)
                extracted_keywords = extractor.extract_keywords(query, max_keywords=15, prefer_morphology=True)
            
            logger.info(f"추출된 키워드: {extracted_keywords[:10]}... (총 {len(extracted_keywords)}개)")
            logger.info(f"검색 결과 수: {len(search_results)}")
            
            # 검색 품질 평가
            from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
            ranker = ResultRanker(use_cross_encoder=False)
            
            metrics = ranker.evaluate_search_quality(
                query=query,
                results=search_results,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            
            test_result = {
                "query": query,
                "query_type": query_type,
                "extracted_keywords": extracted_keywords,
                "search_results_count": len(search_results),
                "metrics": metrics,
                "keyword_coverage": metrics.get("keyword_coverage", 0.0),
                "avg_relevance": metrics.get("avg_relevance", 0.0),
                "diversity_score": metrics.get("diversity_score", 0.0),
                "elapsed_time": elapsed_time,
                "answer_length": len(result.get("answer", "")),
                "retrieved_docs_count": len(retrieved_docs)
            }
            
            logger.info(f"   ✅ Keyword Coverage: {metrics.get('keyword_coverage', 0.0):.3f}")
            logger.info(f"   ✅ Avg Relevance: {metrics.get('avg_relevance', 0.0):.3f}")
            logger.info(f"   ✅ Diversity Score: {metrics.get('diversity_score', 0.0):.3f}")
            logger.info(f"   ✅ 처리 시간: {elapsed_time:.2f}초")
            
            return test_result
            
        except Exception as e:
            logger.error(f"워크플로우 검색 테스트 실패: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "keyword_coverage": 0.0
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("\n" + "="*80)
        logger.info("Keyword Coverage 개선 효과 통합 테스트 시작 (실제 워크플로우)")
        logger.info("="*80)
        
        all_results = {
            "test_date": datetime.now().isoformat(),
            "workflow_tests": {},
            "summary": {}
        }
        
        try:
            # 각 쿼리별 테스트 실행
            for i, test_case in enumerate(self.test_queries, 1):
                query = test_case["query"]
                query_type = test_case["query_type"]
                
                logger.info(f"\n[{i}/{len(self.test_queries)}] 테스트 진행 중...")
                
                result = await self.test_workflow_with_keyword_coverage(query, query_type)
                all_results["workflow_tests"][query] = result
                
                # API 호출 간격
                if i < len(self.test_queries):
                    await asyncio.sleep(2)
            
            # 요약 통계 계산
            if all_results["workflow_tests"]:
                successful_results = [
                    r for r in all_results["workflow_tests"].values() 
                    if "error" not in r
                ]
                
                if successful_results:
                    keyword_coverages = [r.get("keyword_coverage", 0.0) for r in successful_results]
                    avg_relevances = [r.get("avg_relevance", 0.0) for r in successful_results]
                    diversity_scores = [r.get("diversity_score", 0.0) for r in successful_results]
                    elapsed_times = [r.get("elapsed_time", 0.0) for r in successful_results]
                    
                    all_results["summary"] = {
                        "total_queries": len(self.test_queries),
                        "successful_queries": len(successful_results),
                        "avg_keyword_coverage": sum(keyword_coverages) / len(keyword_coverages),
                        "avg_relevance": sum(avg_relevances) / len(avg_relevances),
                        "avg_diversity_score": sum(diversity_scores) / len(diversity_scores),
                        "avg_elapsed_time": sum(elapsed_times) / len(elapsed_times),
                        "min_keyword_coverage": min(keyword_coverages),
                        "max_keyword_coverage": max(keyword_coverages),
                        "total_elapsed_time": sum(elapsed_times)
                    }
                    
                    logger.info("\n" + "="*80)
                    logger.info("통합 테스트 결과 요약")
                    logger.info("="*80)
                    logger.info(f"총 쿼리 수: {all_results['summary']['total_queries']}")
                    logger.info(f"성공한 쿼리 수: {all_results['summary']['successful_queries']}")
                    logger.info(f"평균 Keyword Coverage: {all_results['summary']['avg_keyword_coverage']:.3f}")
                    logger.info(f"평균 Relevance: {all_results['summary']['avg_relevance']:.3f}")
                    logger.info(f"평균 Diversity Score: {all_results['summary']['avg_diversity_score']:.3f}")
                    logger.info(f"평균 처리 시간: {all_results['summary']['avg_elapsed_time']:.2f}초")
                    logger.info(f"Keyword Coverage 범위: {all_results['summary']['min_keyword_coverage']:.3f} ~ {all_results['summary']['max_keyword_coverage']:.3f}")
            
            logger.info("\n" + "="*80)
            logger.info("✅ 통합 테스트 완료")
            logger.info("="*80)
            
            return all_results
            
        except Exception as e:
            logger.error(f"통합 테스트 실행 중 오류 발생: {e}", exc_info=True)
            return all_results


async def main():
    """메인 실행 함수"""
    tester = KeywordCoverageWorkflowTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    output_file = os.path.join(project_root, "docs", "improvements", "keyword_coverage_workflow_test_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n테스트 결과가 저장되었습니다: {output_file}")
    
    # 결과 출력
    print("\n" + "="*80)
    print("통합 테스트 결과 (요약)")
    print("="*80)
    if "summary" in results and results["summary"]:
        summary = results["summary"]
        print(f"평균 Keyword Coverage: {summary.get('avg_keyword_coverage', 0.0):.3f}")
        print(f"평균 Relevance: {summary.get('avg_relevance', 0.0):.3f}")
        print(f"평균 Diversity Score: {summary.get('avg_diversity_score', 0.0):.3f}")
        print(f"평균 처리 시간: {summary.get('avg_elapsed_time', 0.0):.2f}초")
        print(f"Keyword Coverage 범위: {summary.get('min_keyword_coverage', 0.0):.3f} ~ {summary.get('max_keyword_coverage', 0.0):.3f}")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        sys.exit(1)

