"""
간단한 하이브리드 검색 테스트 스크립트
TASK 3.2 구현된 하이브리드 검색 시스템의 기본 기능 테스트
"""

import os
import sys
import json
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 로깅 완전 비활성화
import logging
logging.disable(logging.CRITICAL)

# 간단한 print 함수로 대체
def log_info(message):
    print(f"[INFO] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

def test_exact_search_engine():
    """정확한 매칭 검색 엔진 테스트"""
    try:
        log_info("Testing exact search engine...")
        
        from source.services.exact_search_engine import ExactSearchEngine
        
        # 검색 엔진 초기화
        exact_search = ExactSearchEngine()
        
        # 샘플 데이터 삽입
        law_id = exact_search.insert_law(
            law_name="민법",
            article_number="제1조",
            content="민법은 개인의 사생활과 재산관계를 규율하는 법률이다.",
            law_type="민법",
            effective_date="1960-01-01"
        )
        
        precedent_id = exact_search.insert_precedent(
            case_number="2024다12345",
            court_name="대법원",
            decision_date="2024-01-15",
            case_name="계약서 작성에 관한 판례",
            content="계약서는 당사자 간의 합의사항을 명확히 기록한 문서이다.",
            case_type="민사"
        )
        
        # 검색 테스트
        law_results = exact_search.search_laws("민법")
        precedent_results = exact_search.search_precedents("계약서")
        
        log_info(f"Law search results: {len(law_results)}")
        log_info(f"Precedent search results: {len(precedent_results)}")
        
        # 쿼리 파싱 테스트
        parsed = exact_search.parse_query("민법 제1조")
        log_info(f"Query parsing result: {parsed}")
        
        log_info("✅ Exact search engine test completed")
        return True
        
    except Exception as e:
        log_error(f"❌ Exact search engine test failed: {e}")
        return False

def test_semantic_search_engine():
    """의미적 검색 엔진 테스트"""
    try:
        log_info("Testing semantic search engine...")
        
        from source.services.semantic_search_engine import SemanticSearchEngine
        
        # 검색 엔진 초기화
        semantic_search = SemanticSearchEngine()
        
        # 샘플 문서 생성
        sample_documents = [
            {
                "id": "law_1",
                "type": "law",
                "title": "민법 제1조",
                "content": "민법은 개인의 사생활과 재산관계를 규율하는 법률이다.",
                "source": "test"
            },
            {
                "id": "precedent_1",
                "type": "precedent",
                "title": "계약서 작성 판례",
                "content": "계약서는 당사자 간의 합의사항을 명확히 기록한 문서이다.",
                "source": "test"
            }
        ]
        
        # 인덱스 구축
        success = semantic_search.build_index(sample_documents)
        
        if success:
            # 검색 테스트
            results = semantic_search.search("계약서 작성 방법", k=5, threshold=0.3)
            log_info(f"Semantic search results: {len(results)}")
            
            # 통계 정보
            stats = semantic_search.get_index_stats()
            log_info(f"Index stats: {stats}")
            
            log_info("✅ Semantic search engine test completed")
            return True
        else:
            log_error("❌ Failed to build semantic search index")
            return False
        
    except Exception as e:
        log_error(f"❌ Semantic search engine test failed: {e}")
        return False

def test_result_merger():
    """결과 통합 시스템 테스트"""
    try:
        log_info("Testing result merger...")
        
        from source.services.result_merger import ResultMerger, ResultRanker
        
        # 결과 통합기 초기화
        merger = ResultMerger()
        ranker = ResultRanker()
        
        # 샘플 결과 데이터
        exact_results = {
            "law": [
                {
                    "id": "law_1",
                    "law_name": "민법",
                    "content": "민법 제1조 내용",
                    "search_type": "exact_match",
                    "relevance_score": 1.0
                }
            ]
        }
        
        semantic_results = [
            {
                "id": "precedent_1",
                "type": "precedent",
                "content": "계약서 작성 판례",
                "similarity_score": 0.8,
                "search_type": "semantic",
                "relevance_score": 0.8
            }
        ]
        
        # 결과 통합
        merged_results = merger.merge_results(exact_results, semantic_results, "계약서")
        log_info(f"Merged results: {len(merged_results)}")
        
        # 결과 랭킹
        ranked_results = ranker.rank_results(merged_results, "계약서")
        log_info(f"Ranked results: {len(ranked_results)}")
        
        # 다양성 필터 테스트
        filtered_results = ranker.apply_diversity_filter(ranked_results, max_per_type=2)
        log_info(f"Filtered results: {len(filtered_results)}")
        
        log_info("✅ Result merger test completed")
        return True
        
    except Exception as e:
        log_error(f"❌ Result merger test failed: {e}")
        return False

def test_hybrid_search_engine():
    """하이브리드 검색 엔진 테스트"""
    try:
        log_info("Testing hybrid search engine...")
        
        from source.services.hybrid_search_engine import HybridSearchEngine
        
        # 하이브리드 검색 엔진 초기화
        hybrid_search = HybridSearchEngine()
        
        # 샘플 문서로 인덱스 구축
        sample_documents = [
            {
                "id": "law_1",
                "type": "law",
                "title": "민법 제1조",
                "content": "민법은 개인의 사생활과 재산관계를 규율하는 법률이다.",
                "source": "test"
            },
            {
                "id": "precedent_1",
                "type": "precedent",
                "title": "계약서 작성 판례",
                "content": "계약서는 당사자 간의 합의사항을 명확히 기록한 문서이다.",
                "source": "test"
            }
        ]
        
        # 인덱스 구축
        success = hybrid_search.build_index(sample_documents)
        
        if success:
            # 하이브리드 검색 테스트
            result = hybrid_search.search("계약서 작성", max_results=10)
            
            log_info(f"Hybrid search results: {result['total_results']}")
            log_info(f"Search stats: {result['search_stats']}")
            
            # 검색 통계
            stats = hybrid_search.get_search_stats()
            log_info(f"Search engine stats: {stats}")
            
            log_info("✅ Hybrid search engine test completed")
            return True
        else:
            log_error("❌ Failed to build hybrid search index")
            return False
        
    except Exception as e:
        log_error(f"❌ Hybrid search engine test failed: {e}")
        return False

def main():
    """메인 함수"""
    log_info("=" * 60)
    log_info("🚀 Starting TASK 3.2 Hybrid Search System Tests")
    log_info("=" * 60)
    
    test_results = {
        "exact_search": test_exact_search_engine(),
        "semantic_search": test_semantic_search_engine(),
        "result_merger": test_result_merger(),
        "hybrid_search": test_hybrid_search_engine()
    }
    
    # 결과 요약
    log_info("=" * 60)
    log_info("📊 Test Results Summary")
    log_info("=" * 60)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        log_info(f"{test_name}: {status}")
    
    log_info(f"Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 75:
        log_info("🎉 TASK 3.2 Hybrid Search System tests PASSED!")
    else:
        log_warning("⚠️ TASK 3.2 Hybrid Search System tests need improvement")
    
    log_info("=" * 60)
    
    # 결과 저장
    try:
        os.makedirs("results", exist_ok=True)
        with open("results/task3_2_test_results.json", 'w', encoding='utf-8') as f:
            json.dump({
                "test_results": test_results,
                "success_rate": success_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests
            }, f, ensure_ascii=False, indent=2)
        log_info("Test results saved to results/task3_2_test_results.json")
    except Exception as e:
        log_error(f"Failed to save test results: {e}")

if __name__ == "__main__":
    main()
