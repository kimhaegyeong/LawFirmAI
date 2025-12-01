#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keyword Coverage 개선 효과 통합 테스트
실제 워크플로우를 통해 검색을 실행하고 Keyword Coverage를 측정합니다.
"""

import sys
import os
import logging
import time
import json
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeywordCoverageIntegrationTester:
    """Keyword Coverage 개선 효과 통합 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {}
        self.test_queries = [
            {
                "query": "계약 해지 사유에 대해 알려주세요",
                "query_type": "law_inquiry",
                "expected_keywords": ["계약", "해지", "사유"],
                "domain": "민사법"
            },
            {
                "query": "손해배상 청구권의 소멸시효는?",
                "query_type": "legal_advice",
                "expected_keywords": ["손해배상", "청구권", "소멸시효"],
                "domain": "민사법"
            },
            {
                "query": "불법행위로 인한 손해배상 판례를 찾아주세요",
                "query_type": "precedent_search",
                "expected_keywords": ["불법행위", "손해배상", "판례"],
                "domain": "민사법"
            },
            {
                "query": "민법 제750조 불법행위에 대해 설명해주세요",
                "query_type": "law_inquiry",
                "expected_keywords": ["민법", "제750조", "불법행위"],
                "domain": "민사법"
            },
            {
                "query": "계약 해제와 계약 해지의 차이점은?",
                "query_type": "legal_advice",
                "expected_keywords": ["계약", "해제", "해지"],
                "domain": "민사법"
            }
        ]
    
    def test_workflow_search(self, query: str, query_type: str = "") -> Dict[str, Any]:
        """실제 워크플로우를 통한 검색 테스트"""
        try:
            from lawfirm_langgraph.core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
            from lawfirm_langgraph.core.workflow.state.state_utils import create_initial_legal_state
            from lawfirm_langgraph.langgraph_core.config.langgraph_config import LangGraphConfig
            
            config = LangGraphConfig()
            workflow = EnhancedLegalQuestionWorkflow(config)
            
            # 초기 상태 생성
            session_id = f"test_{int(time.time())}"
            initial_state = create_initial_legal_state(query, session_id)
            
            logger.info(f"\n검색 쿼리: {query}")
            logger.info(f"질문 유형: {query_type}")
            
            start_time = time.time()
            
            # 키워드 확장 단계 실행
            state_after_expand = workflow.expand_keywords(initial_state)
            extracted_keywords = state_after_expand.get("extracted_keywords", [])
            
            logger.info(f"추출된 키워드: {extracted_keywords[:10]}... (총 {len(extracted_keywords)}개)")
            
            # 검색 실행
            state_after_search = workflow.execute_searches(state_after_expand)
            search_results = state_after_search.get("search_results", [])
            
            elapsed_time = time.time() - start_time
            
            # 검색 품질 평가
            from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
            ranker = ResultRanker(use_cross_encoder=False)
            
            metrics = ranker.evaluate_search_quality(
                query=query,
                results=search_results,
                query_type=query_type,
                extracted_keywords=extracted_keywords
            )
            
            result = {
                "query": query,
                "query_type": query_type,
                "extracted_keywords": extracted_keywords,
                "search_results_count": len(search_results),
                "metrics": metrics,
                "keyword_coverage": metrics.get("keyword_coverage", 0.0),
                "avg_relevance": metrics.get("avg_relevance", 0.0),
                "diversity_score": metrics.get("diversity_score", 0.0),
                "elapsed_time": elapsed_time
            }
            
            logger.info(f"   ✅ Keyword Coverage: {metrics.get('keyword_coverage', 0.0):.3f}")
            logger.info(f"   ✅ Avg Relevance: {metrics.get('avg_relevance', 0.0):.3f}")
            logger.info(f"   ✅ Diversity Score: {metrics.get('diversity_score', 0.0):.3f}")
            logger.info(f"   ✅ 검색 결과 수: {len(search_results)}")
            logger.info(f"   ✅ 처리 시간: {elapsed_time:.2f}초")
            
            return result
            
        except Exception as e:
            logger.error(f"워크플로우 검색 테스트 실패: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "keyword_coverage": 0.0
            }
    
    def test_keyword_extraction_improvement(self) -> Dict[str, Any]:
        """키워드 추출 개선 효과 테스트"""
        logger.info("\n" + "="*80)
        logger.info("키워드 추출 개선 효과 테스트")
        logger.info("="*80)
        
        results = {}
        
        try:
            from lawfirm_langgraph.core.agents.keyword_extractor import KeywordExtractor
            
            extractor = KeywordExtractor(use_morphology=True)
            
            for test_case in self.test_queries:
                query = test_case["query"]
                expected_keywords = test_case["expected_keywords"]
                
                logger.info(f"\n테스트 쿼리: {query}")
                
                extracted_keywords = extractor.extract_keywords(
                    query,
                    max_keywords=15,
                    prefer_morphology=True
                )
                
                # 예상 키워드와의 매칭 확인
                matched_keywords = [kw for kw in expected_keywords if kw in extracted_keywords]
                match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
                
                # 복합어 인식 확인
                compound_keywords = [kw for kw in extracted_keywords if any(
                    pattern.search(kw) for pattern in extractor.LEGAL_COMPOUND_PATTERNS
                )]
                
                # 불용어 제거 확인
                stopwords_found = [kw for kw in extracted_keywords if kw in extractor.BASIC_STOPWORDS]
                
                results[query] = {
                    "extracted_keywords": extracted_keywords,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": matched_keywords,
                    "match_rate": match_rate,
                    "compound_keywords": compound_keywords,
                    "stopwords_found": stopwords_found,
                    "improvement": {
                        "compound_recognition": len(compound_keywords) > 0,
                        "stopword_filtering": len(stopwords_found) == 0
                    }
                }
                
                logger.info(f"   ✅ 추출된 키워드: {extracted_keywords}")
                logger.info(f"   ✅ 예상 키워드 매칭률: {match_rate:.2%}")
                logger.info(f"   ✅ 복합어 인식: {len(compound_keywords)}개 ({compound_keywords})")
                logger.info(f"   ✅ 불용어 제거: {'성공' if len(stopwords_found) == 0 else f'실패 ({stopwords_found})'}")
            
            self.test_results["keyword_extraction"] = results
            return results
            
        except Exception as e:
            logger.error(f"키워드 추출 테스트 실패: {e}", exc_info=True)
            return {}
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """통합 테스트 실행"""
        logger.info("\n" + "="*80)
        logger.info("Keyword Coverage 개선 효과 통합 테스트 시작")
        logger.info("="*80)
        
        all_results = {
            "test_date": datetime.now().isoformat(),
            "workflow_tests": {},
            "keyword_extraction_tests": {},
            "summary": {}
        }
        
        try:
            # 1. 키워드 추출 개선 효과 테스트
            extraction_results = self.test_keyword_extraction_improvement()
            all_results["keyword_extraction_tests"] = extraction_results
            
            # 2. 실제 워크플로우 검색 테스트 (처음 3개 쿼리만)
            logger.info("\n" + "="*80)
            logger.info("실제 워크플로우 검색 테스트")
            logger.info("="*80)
            
            workflow_results = {}
            for test_case in self.test_queries[:3]:  # 처음 3개만 테스트
                query = test_case["query"]
                query_type = test_case["query_type"]
                
                result = self.test_workflow_search(query, query_type)
                workflow_results[query] = result
                
                time.sleep(1)  # API 호출 간격
            
            all_results["workflow_tests"] = workflow_results
            
            # 3. 요약 통계 계산
            if workflow_results:
                keyword_coverages = [
                    r.get("keyword_coverage", 0.0) 
                    for r in workflow_results.values() 
                    if "error" not in r
                ]
                avg_relevances = [
                    r.get("avg_relevance", 0.0) 
                    for r in workflow_results.values() 
                    if "error" not in r
                ]
                diversity_scores = [
                    r.get("diversity_score", 0.0) 
                    for r in workflow_results.values() 
                    if "error" not in r
                ]
                
                all_results["summary"] = {
                    "total_queries": len(workflow_results),
                    "successful_queries": len([r for r in workflow_results.values() if "error" not in r]),
                    "avg_keyword_coverage": sum(keyword_coverages) / len(keyword_coverages) if keyword_coverages else 0.0,
                    "avg_relevance": sum(avg_relevances) / len(avg_relevances) if avg_relevances else 0.0,
                    "avg_diversity_score": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
                    "min_keyword_coverage": min(keyword_coverages) if keyword_coverages else 0.0,
                    "max_keyword_coverage": max(keyword_coverages) if keyword_coverages else 0.0
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
            
            # 키워드 추출 개선 요약
            if extraction_results:
                match_rates = [
                    r.get("match_rate", 0.0) 
                    for r in extraction_results.values()
                ]
                compound_recognition_count = sum(
                    1 for r in extraction_results.values() 
                    if r.get("improvement", {}).get("compound_recognition", False)
                )
                stopword_filtering_count = sum(
                    1 for r in extraction_results.values() 
                    if r.get("improvement", {}).get("stopword_filtering", False)
                )
                
                all_results["summary"]["keyword_extraction"] = {
                    "avg_match_rate": sum(match_rates) / len(match_rates) if match_rates else 0.0,
                    "compound_recognition_success": compound_recognition_count,
                    "stopword_filtering_success": stopword_filtering_count,
                    "total_tests": len(extraction_results)
                }
                
                logger.info(f"\n키워드 추출 개선 요약:")
                logger.info(f"   평균 매칭률: {all_results['summary']['keyword_extraction']['avg_match_rate']:.2%}")
                logger.info(f"   복합어 인식 성공: {compound_recognition_count}/{len(extraction_results)}")
                logger.info(f"   불용어 필터링 성공: {stopword_filtering_count}/{len(extraction_results)}")
            
            logger.info("\n" + "="*80)
            logger.info("✅ 통합 테스트 완료")
            logger.info("="*80)
            
            return all_results
            
        except Exception as e:
            logger.error(f"통합 테스트 실행 중 오류 발생: {e}", exc_info=True)
            return all_results


if __name__ == "__main__":
    tester = KeywordCoverageIntegrationTester()
    results = tester.run_integration_tests()
    
    # 결과를 JSON 파일로 저장
    output_file = os.path.join(project_root, "docs", "improvements", "keyword_coverage_integration_test_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n테스트 결과가 저장되었습니다: {output_file}")
    
    # 결과 출력
    print("\n" + "="*80)
    print("통합 테스트 결과 (요약)")
    print("="*80)
    if "summary" in results:
        summary = results["summary"]
        print(f"평균 Keyword Coverage: {summary.get('avg_keyword_coverage', 0.0):.3f}")
        print(f"평균 Relevance: {summary.get('avg_relevance', 0.0):.3f}")
        print(f"평균 Diversity Score: {summary.get('avg_diversity_score', 0.0):.3f}")
        if "keyword_extraction" in summary:
            ke_summary = summary["keyword_extraction"]
            print(f"\n키워드 추출 개선:")
            print(f"   평균 매칭률: {ke_summary.get('avg_match_rate', 0.0):.2%}")
            print(f"   복합어 인식 성공: {ke_summary.get('compound_recognition_success', 0)}/{ke_summary.get('total_tests', 0)}")
            print(f"   불용어 필터링 성공: {ke_summary.get('stopword_filtering_success', 0)}/{ke_summary.get('total_tests', 0)}")

