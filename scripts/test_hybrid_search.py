"""
하이브리드 검색 시스템 테스트 스크립트
TASK 3.2 구현된 하이브리드 검색 시스템의 기능을 테스트
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.exact_search_engine import ExactSearchEngine
from source.services.semantic_search_engine import SemanticSearchEngine

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridSearchTester:
    """하이브리드 검색 시스템 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.search_engine = None
        
    def initialize_search_engine(self):
        """검색 엔진 초기화"""
        try:
            logger.info("Initializing hybrid search engine...")
            self.search_engine = HybridSearchEngine()
            logger.info("Search engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            return False
    
    def test_exact_search(self) -> Dict[str, Any]:
        """정확한 매칭 검색 테스트"""
        logger.info("Testing exact search engine...")
        
        test_cases = [
            {
                "query": "민법",
                "expected_types": ["law"],
                "description": "법령명 검색"
            },
            {
                "query": "제1조",
                "expected_types": ["law"],
                "description": "조문번호 검색"
            },
            {
                "query": "2024다12345",
                "expected_types": ["precedent"],
                "description": "사건번호 검색"
            },
            {
                "query": "대법원",
                "expected_types": ["precedent"],
                "description": "법원명 검색"
            }
        ]
        
        results = {
            "test_name": "exact_search",
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_cases": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                # 정확한 매칭 검색 실행
                exact_results = self.search_engine.exact_search.search_all(test_case["query"])
                
                # 결과 검증
                total_results = sum(len(r) for r in exact_results.values())
                test_passed = total_results > 0
                
                test_result = {
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": test_passed,
                    "total_results": total_results,
                    "results_by_type": {k: len(v) for k, v in exact_results.items()}
                }
                
                results["test_cases"].append(test_result)
                
                if test_passed:
                    results["passed"] += 1
                    logger.info(f"✓ Exact search test {i+1} passed: {test_case['query']}")
                else:
                    results["failed"] += 1
                    logger.warning(f"✗ Exact search test {i+1} failed: {test_case['query']}")
                    
            except Exception as e:
                results["failed"] += 1
                logger.error(f"✗ Exact search test {i+1} error: {e}")
                results["test_cases"].append({
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": False,
                    "error": str(e)
                })
        
        return results
    
    def test_semantic_search(self) -> Dict[str, Any]:
        """의미적 검색 테스트"""
        logger.info("Testing semantic search engine...")
        
        test_cases = [
            {
                "query": "계약서 작성 방법",
                "description": "법률 실무 질문"
            },
            {
                "query": "부동산 매매 계약",
                "description": "특정 법률 영역 질문"
            },
            {
                "query": "손해배상 청구",
                "description": "민사법 질문"
            },
            {
                "query": "형사처벌 기준",
                "description": "형사법 질문"
            }
        ]
        
        results = {
            "test_name": "semantic_search",
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_cases": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                # 의미적 검색 실행
                semantic_results = self.search_engine.semantic_search.search(
                    test_case["query"], k=10, threshold=0.3
                )
                
                # 결과 검증
                test_passed = len(semantic_results) > 0
                
                test_result = {
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": test_passed,
                    "total_results": len(semantic_results),
                    "avg_similarity": sum(r.get("similarity_score", 0) for r in semantic_results) / len(semantic_results) if semantic_results else 0
                }
                
                results["test_cases"].append(test_result)
                
                if test_passed:
                    results["passed"] += 1
                    logger.info(f"✓ Semantic search test {i+1} passed: {test_case['query']}")
                else:
                    results["failed"] += 1
                    logger.warning(f"✗ Semantic search test {i+1} failed: {test_case['query']}")
                    
            except Exception as e:
                results["failed"] += 1
                logger.error(f"✗ Semantic search test {i+1} error: {e}")
                results["test_cases"].append({
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": False,
                    "error": str(e)
                })
        
        return results
    
    def test_hybrid_search(self) -> Dict[str, Any]:
        """하이브리드 검색 테스트"""
        logger.info("Testing hybrid search engine...")
        
        test_cases = [
            {
                "query": "민법 제1조",
                "description": "법령명 + 조문번호 조합"
            },
            {
                "query": "계약서 작성 방법",
                "description": "의미적 검색 중심 질문"
            },
            {
                "query": "2024다12345 판례",
                "description": "사건번호 + 의미적 검색"
            },
            {
                "query": "부동산 매매 계약서 검토",
                "description": "복합 법률 질문"
            }
        ]
        
        results = {
            "test_name": "hybrid_search",
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_cases": []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                # 하이브리드 검색 실행
                hybrid_result = self.search_engine.search(
                    query=test_case["query"],
                    max_results=20
                )
                
                # 결과 검증
                test_passed = hybrid_result["total_results"] > 0
                
                test_result = {
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": test_passed,
                    "total_results": hybrid_result["total_results"],
                    "search_stats": hybrid_result["search_stats"],
                    "success": hybrid_result.get("success", True)
                }
                
                results["test_cases"].append(test_result)
                
                if test_passed:
                    results["passed"] += 1
                    logger.info(f"✓ Hybrid search test {i+1} passed: {test_case['query']}")
                else:
                    results["failed"] += 1
                    logger.warning(f"✗ Hybrid search test {i+1} failed: {test_case['query']}")
                    
            except Exception as e:
                results["failed"] += 1
                logger.error(f"✗ Hybrid search test {i+1} error: {e}")
                results["test_cases"].append({
                    "test_id": i + 1,
                    "query": test_case["query"],
                    "description": test_case["description"],
                    "passed": False,
                    "error": str(e)
                })
        
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        logger.info("Testing search performance...")
        
        test_queries = [
            "민법",
            "계약서 작성",
            "부동산 매매",
            "손해배상",
            "형사처벌"
        ]
        
        results = {
            "test_name": "performance",
            "total_queries": len(test_queries),
            "avg_response_time": 0,
            "min_response_time": float('inf'),
            "max_response_time": 0,
            "query_times": []
        }
        
        total_time = 0
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                # 하이브리드 검색 실행
                hybrid_result = self.search_engine.search(query, max_results=10)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                total_time += response_time
                results["min_response_time"] = min(results["min_response_time"], response_time)
                results["max_response_time"] = max(results["max_response_time"], response_time)
                
                results["query_times"].append({
                    "query": query,
                    "response_time": response_time,
                    "results_count": hybrid_result["total_results"]
                })
                
                logger.info(f"Query '{query}' completed in {response_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Performance test failed for query '{query}': {e}")
                results["query_times"].append({
                    "query": query,
                    "response_time": -1,
                    "error": str(e)
                })
        
        if results["query_times"]:
            results["avg_response_time"] = total_time / len(test_queries)
            results["min_response_time"] = results["min_response_time"] if results["min_response_time"] != float('inf') else 0
        
        return results
    
    def test_search_stats(self) -> Dict[str, Any]:
        """검색 통계 테스트"""
        logger.info("Testing search statistics...")
        
        try:
            stats = self.search_engine.get_search_stats()
            
            results = {
                "test_name": "search_stats",
                "passed": True,
                "stats": stats
            }
            
            logger.info("✓ Search statistics test passed")
            return results
            
        except Exception as e:
            logger.error(f"✗ Search statistics test failed: {e}")
            return {
                "test_name": "search_stats",
                "passed": False,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("Starting comprehensive hybrid search system tests...")
        
        if not self.initialize_search_engine():
            return {
                "success": False,
                "error": "Failed to initialize search engine"
            }
        
        test_results = {
            "test_suite": "hybrid_search_system",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
        # 각 테스트 실행
        tests = [
            self.test_exact_search,
            self.test_semantic_search,
            self.test_hybrid_search,
            self.test_performance,
            self.test_search_stats
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                test_results["tests"].append(result)
                
                if "passed" in result and "failed" in result:
                    test_results["summary"]["total_tests"] += result["total_tests"]
                    test_results["summary"]["passed"] += result["passed"]
                    test_results["summary"]["failed"] += result["failed"]
                elif result.get("passed"):
                    test_results["summary"]["total_tests"] += 1
                    test_results["summary"]["passed"] += 1
                else:
                    test_results["summary"]["total_tests"] += 1
                    test_results["summary"]["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed: {e}")
                test_results["tests"].append({
                    "test_name": test_func.__name__,
                    "passed": False,
                    "error": str(e)
                })
                test_results["summary"]["total_tests"] += 1
                test_results["summary"]["failed"] += 1
        
        # 전체 성공률 계산
        success_rate = (test_results["summary"]["passed"] / test_results["summary"]["total_tests"]) * 100 if test_results["summary"]["total_tests"] > 0 else 0
        test_results["summary"]["success_rate"] = success_rate
        
        logger.info(f"All tests completed. Success rate: {success_rate:.1f}%")
        
        return test_results
    
    def save_test_results(self, results: Dict[str, Any], output_path: str = "results/hybrid_search_test_results.json"):
        """테스트 결과 저장"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Test results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("🚀 Starting TASK 3.2 Hybrid Search System Tests")
    logger.info("=" * 60)
    
    tester = HybridSearchTester()
    
    # 모든 테스트 실행
    results = tester.run_all_tests()
    
    # 결과 저장
    tester.save_test_results(results)
    
    # 결과 요약 출력
    logger.info("=" * 60)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {results['summary']['total_tests']}")
    logger.info(f"Passed: {results['summary']['passed']}")
    logger.info(f"Failed: {results['summary']['failed']}")
    logger.info(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    
    if results['summary']['success_rate'] >= 80:
        logger.info("🎉 Hybrid search system tests PASSED!")
    else:
        logger.warning("⚠️ Hybrid search system tests need improvement")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
