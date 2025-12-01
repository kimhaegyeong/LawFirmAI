#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Keyword Coverage 개선 기능 테스트 스크립트
Phase 1, 2, 3에서 구현된 기능들을 테스트합니다.
"""

import sys
import os
import logging
import time
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


class KeywordCoverageImprovementTester:
    """Keyword Coverage 개선 기능 테스트 클래스"""
    
    def __init__(self):
        self.test_results = {
            "phase1": {},
            "phase2": {},
            "phase3": {},
            "overall": {}
        }
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
            }
        ]
    
    def test_phase1_keyword_expansion(self) -> Dict[str, Any]:
        """Phase 1: LLM 기반 키워드 확장 테스트"""
        logger.info("\n" + "="*80)
        logger.info("Phase 1 테스트: LLM 기반 키워드 확장")
        logger.info("="*80)
        
        results = {}
        
        try:
            from lawfirm_langgraph.core.processing.extractors.ai_keyword_generator import AIKeywordGenerator
            
            generator = AIKeywordGenerator()
            
            for test_case in self.test_queries[:1]:  # 첫 번째 쿼리만 테스트
                query = test_case["query"]
                query_type = test_case["query_type"]
                base_keywords = test_case["expected_keywords"]
                domain = test_case["domain"]
                
                logger.info(f"\n테스트 쿼리: {query}")
                logger.info(f"기본 키워드: {base_keywords}")
                
                start_time = time.time()
                
                # 비동기 호출을 동기로 변환
                import asyncio
                try:
                    expansion_result = asyncio.run(
                        generator.expand_domain_keywords(
                            domain=domain,
                            base_keywords=base_keywords,
                            target_count=50,
                            query=query,
                            query_type=query_type
                        )
                    )
                except RuntimeError:
                    # 이미 실행 중인 이벤트 루프가 있는 경우
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    expansion_result = loop.run_until_complete(
                        generator.expand_domain_keywords(
                            domain=domain,
                            base_keywords=base_keywords,
                            target_count=50,
                            query=query,
                            query_type=query_type
                        )
                    )
                    loop.close()
                
                elapsed_time = time.time() - start_time
                
                expanded_keywords = expansion_result.expanded_keywords if expansion_result else []
                
                results[query] = {
                    "base_keywords": base_keywords,
                    "expanded_keywords": expanded_keywords,
                    "expansion_count": len(expanded_keywords),
                    "expansion_ratio": len(expanded_keywords) / len(base_keywords) if base_keywords else 0,
                    "api_call_success": expansion_result.api_call_success if expansion_result else False,
                    "confidence": expansion_result.confidence if expansion_result else 0.0,
                    "elapsed_time": elapsed_time
                }
                
                logger.info(f"   ✅ 확장된 키워드 수: {len(expanded_keywords)}")
                logger.info(f"   ✅ 확장 비율: {results[query]['expansion_ratio']:.2f}")
                logger.info(f"   ✅ 처리 시간: {elapsed_time:.2f}초")
                logger.info(f"   ✅ API 호출 성공: {expansion_result.api_call_success if expansion_result else False}")
                
                # 검증
                assert len(expanded_keywords) > 0, "확장된 키워드가 없습니다"
                assert results[query]['expansion_ratio'] >= 1.0, "키워드 확장이 충분하지 않습니다"
            
            self.test_results["phase1"] = results
            logger.info("\n✅ Phase 1 테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"Phase 1 테스트 실패: {e}", exc_info=True)
            return {}
    
    def test_phase2_keyword_extraction(self) -> Dict[str, Any]:
        """Phase 2: 형태소 분석 기반 키워드 추출 테스트"""
        logger.info("\n" + "="*80)
        logger.info("Phase 2 테스트: 형태소 분석 기반 키워드 추출")
        logger.info("="*80)
        
        results = {}
        
        try:
            from lawfirm_langgraph.core.agents.keyword_extractor import KeywordExtractor
            
            extractor = KeywordExtractor(use_morphology=True)
            
            for test_case in self.test_queries:
                query = test_case["query"]
                expected_keywords = test_case["expected_keywords"]
                
                logger.info(f"\n테스트 쿼리: {query}")
                
                start_time = time.time()
                extracted_keywords = extractor.extract_keywords(
                    query,
                    max_keywords=15,
                    prefer_morphology=True
                )
                elapsed_time = time.time() - start_time
                
                # 예상 키워드와의 매칭 확인
                matched_keywords = [kw for kw in expected_keywords if kw in extracted_keywords]
                match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
                
                results[query] = {
                    "extracted_keywords": extracted_keywords,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": matched_keywords,
                    "match_rate": match_rate,
                    "extraction_count": len(extracted_keywords),
                    "elapsed_time": elapsed_time
                }
                
                logger.info(f"   ✅ 추출된 키워드: {extracted_keywords}")
                logger.info(f"   ✅ 예상 키워드 매칭률: {match_rate:.2%}")
                logger.info(f"   ✅ 처리 시간: {elapsed_time:.3f}초")
                
                # 검증
                assert len(extracted_keywords) > 0, "추출된 키워드가 없습니다"
                assert match_rate >= 0.5, f"예상 키워드 매칭률이 낮습니다: {match_rate:.2%}"
            
            self.test_results["phase2"] = results
            logger.info("\n✅ Phase 2 테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"Phase 2 테스트 실패: {e}", exc_info=True)
            return {}
    
    def test_phase3_semantic_matching(self) -> Dict[str, Any]:
        """Phase 3: 의미 기반 키워드 매칭 테스트"""
        logger.info("\n" + "="*80)
        logger.info("Phase 3 테스트: 의미 기반 키워드 매칭")
        logger.info("="*80)
        
        results = {}
        
        try:
            from lawfirm_langgraph.core.search.processors.result_merger import ResultRanker
            
            ranker = ResultRanker(use_cross_encoder=False)
            
            # 테스트용 검색 결과 생성
            test_results = [
                {
                    "content": "민법 제543조에 따르면 계약 해지 사유는 다음과 같습니다.",
                    "relevance_score": 0.85,
                    "final_weighted_score": 0.85
                },
                {
                    "content": "계약 해지와 관련된 손해배상 청구권에 대한 판례가 있습니다.",
                    "relevance_score": 0.75,
                    "final_weighted_score": 0.75
                },
                {
                    "content": "계약 해지 절차 및 효과에 대해 설명합니다.",
                    "relevance_score": 0.65,
                    "final_weighted_score": 0.65
                }
            ]
            
            for test_case in self.test_queries[:1]:  # 첫 번째 쿼리만 테스트
                query = test_case["query"]
                extracted_keywords = test_case["expected_keywords"]
                
                logger.info(f"\n테스트 쿼리: {query}")
                logger.info(f"추출된 키워드: {extracted_keywords}")
                
                start_time = time.time()
                metrics = ranker.evaluate_search_quality(
                    query=query,
                    results=test_results,
                    query_type=test_case["query_type"],
                    extracted_keywords=extracted_keywords
                )
                elapsed_time = time.time() - start_time
                
                results[query] = {
                    "metrics": metrics,
                    "keyword_coverage": metrics.get("keyword_coverage", 0.0),
                    "avg_relevance": metrics.get("avg_relevance", 0.0),
                    "diversity_score": metrics.get("diversity_score", 0.0),
                    "elapsed_time": elapsed_time
                }
                
                logger.info(f"   ✅ Keyword Coverage: {metrics.get('keyword_coverage', 0.0):.3f}")
                logger.info(f"   ✅ Avg Relevance: {metrics.get('avg_relevance', 0.0):.3f}")
                logger.info(f"   ✅ Diversity Score: {metrics.get('diversity_score', 0.0):.3f}")
                logger.info(f"   ✅ 처리 시간: {elapsed_time:.3f}초")
                
                # 검증
                assert metrics.get("keyword_coverage", 0.0) > 0.0, "Keyword Coverage가 0입니다"
                assert metrics.get("avg_relevance", 0.0) > 0.0, "Avg Relevance가 0입니다"
            
            self.test_results["phase3"] = results
            logger.info("\n✅ Phase 3 테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"Phase 3 테스트 실패: {e}", exc_info=True)
            return {}
    
    def test_phase3_morphological_matching(self) -> Dict[str, Any]:
        """Phase 3: 형태소 분석 기반 부분 매칭 테스트"""
        logger.info("\n" + "="*80)
        logger.info("Phase 3 테스트: 형태소 분석 기반 부분 매칭")
        logger.info("="*80)
        
        results = {}
        
        try:
            from lawfirm_langgraph.core.search.processors.search_result_processor import SearchResultProcessor
            
            processor = SearchResultProcessor()
            
            # 테스트 케이스
            test_cases = [
                {
                    "document": {
                        "content": "손해배상 청구권의 소멸시효에 대한 법령 조문입니다.",
                        "relevance_score": 0.8
                    },
                    "keyword_weights": {
                        "손해배상": 1.0,
                        "청구권": 0.9,
                        "소멸시효": 0.8
                    },
                    "query": "손해배상 청구권 소멸시효"
                },
                {
                    "document": {
                        "content": "불법행위로 인한 손해배상에 대한 판례입니다.",
                        "relevance_score": 0.75
                    },
                    "keyword_weights": {
                        "불법행위": 1.0,
                        "손해배상": 0.9
                    },
                    "query": "불법행위 손해배상"
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"\n테스트 케이스 {i+1}: {test_case['query']}")
                
                start_time = time.time()
                keyword_scores = processor.calculate_keyword_match_score(
                    document=test_case["document"],
                    keyword_weights=test_case["keyword_weights"],
                    query=test_case["query"]
                )
                elapsed_time = time.time() - start_time
                
                results[f"case_{i+1}"] = {
                    "keyword_scores": keyword_scores,
                    "keyword_match_score": keyword_scores.get("keyword_match_score", 0.0),
                    "keyword_coverage": keyword_scores.get("keyword_coverage", 0.0),
                    "matched_keywords": keyword_scores.get("matched_keywords", []),
                    "elapsed_time": elapsed_time
                }
                
                logger.info(f"   ✅ Keyword Match Score: {keyword_scores.get('keyword_match_score', 0.0):.3f}")
                logger.info(f"   ✅ Keyword Coverage: {keyword_scores.get('keyword_coverage', 0.0):.3f}")
                logger.info(f"   ✅ Matched Keywords: {keyword_scores.get('matched_keywords', [])}")
                logger.info(f"   ✅ 처리 시간: {elapsed_time:.3f}초")
                
                # 검증
                assert keyword_scores.get("keyword_match_score", 0.0) > 0.0, "Keyword Match Score가 0입니다"
                assert keyword_scores.get("keyword_coverage", 0.0) > 0.0, "Keyword Coverage가 0입니다"
            
            self.test_results["phase3"]["morphological_matching"] = results
            logger.info("\n✅ Phase 3 형태소 분석 테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"Phase 3 형태소 분석 테스트 실패: {e}", exc_info=True)
            return {}
    
    def generate_test_report(self) -> Dict[str, Any]:
        """테스트 결과 보고서 생성"""
        logger.info("\n" + "="*80)
        logger.info("테스트 결과 요약")
        logger.info("="*80)
        
        report = {
            "test_date": datetime.now().isoformat(),
            "phase1_results": self.test_results.get("phase1", {}),
            "phase2_results": self.test_results.get("phase2", {}),
            "phase3_results": self.test_results.get("phase3", {}),
            "summary": {}
        }
        
        # Phase 1 요약
        if self.test_results.get("phase1"):
            phase1_data = list(self.test_results["phase1"].values())
            if phase1_data:
                report["summary"]["phase1"] = {
                    "avg_expansion_ratio": sum(d.get("expansion_ratio", 0) for d in phase1_data) / len(phase1_data),
                    "avg_expansion_count": sum(d.get("expansion_count", 0) for d in phase1_data) / len(phase1_data),
                    "api_success_rate": sum(1 for d in phase1_data if d.get("api_call_success", False)) / len(phase1_data)
                }
                logger.info(f"\nPhase 1 요약:")
                logger.info(f"   평균 확장 비율: {report['summary']['phase1']['avg_expansion_ratio']:.2f}")
                logger.info(f"   평균 확장 키워드 수: {report['summary']['phase1']['avg_expansion_count']:.1f}")
                logger.info(f"   API 성공률: {report['summary']['phase1']['api_success_rate']:.2%}")
        
        # Phase 2 요약
        if self.test_results.get("phase2"):
            phase2_data = list(self.test_results["phase2"].values())
            if phase2_data:
                report["summary"]["phase2"] = {
                    "avg_match_rate": sum(d.get("match_rate", 0) for d in phase2_data) / len(phase2_data),
                    "avg_extraction_count": sum(d.get("extraction_count", 0) for d in phase2_data) / len(phase2_data)
                }
                logger.info(f"\nPhase 2 요약:")
                logger.info(f"   평균 키워드 매칭률: {report['summary']['phase2']['avg_match_rate']:.2%}")
                logger.info(f"   평균 추출 키워드 수: {report['summary']['phase2']['avg_extraction_count']:.1f}")
        
        # Phase 3 요약
        if self.test_results.get("phase3"):
            phase3_data = [v for v in self.test_results["phase3"].values() if isinstance(v, dict) and "keyword_coverage" in v]
            if phase3_data:
                report["summary"]["phase3"] = {
                    "avg_keyword_coverage": sum(d.get("keyword_coverage", 0) for d in phase3_data) / len(phase3_data),
                    "avg_relevance": sum(d.get("avg_relevance", 0) for d in phase3_data) / len(phase3_data)
                }
                logger.info(f"\nPhase 3 요약:")
                logger.info(f"   평균 Keyword Coverage: {report['summary']['phase3']['avg_keyword_coverage']:.3f}")
                logger.info(f"   평균 Relevance: {report['summary']['phase3']['avg_relevance']:.3f}")
        
        return report
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("\n" + "="*80)
        logger.info("Keyword Coverage 개선 기능 테스트 시작")
        logger.info("="*80)
        
        try:
            # Phase 1 테스트
            self.test_phase1_keyword_expansion()
            
            # Phase 2 테스트
            self.test_phase2_keyword_extraction()
            
            # Phase 3 테스트
            self.test_phase3_semantic_matching()
            self.test_phase3_morphological_matching()
            
            # 보고서 생성
            report = self.generate_test_report()
            
            logger.info("\n" + "="*80)
            logger.info("✅ 모든 테스트 완료")
            logger.info("="*80)
            
            return report
            
        except Exception as e:
            logger.error(f"테스트 실행 중 오류 발생: {e}", exc_info=True)
            return {}


if __name__ == "__main__":
    tester = KeywordCoverageImprovementTester()
    report = tester.run_all_tests()
    
    # 결과 출력
    import json
    print("\n" + "="*80)
    print("테스트 결과 (JSON)")
    print("="*80)
    print(json.dumps(report, ensure_ascii=False, indent=2))

