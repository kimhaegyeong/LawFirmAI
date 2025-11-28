# -*- coding: utf-8 -*-
"""
확장된 쿼리 결과 병합 및 중복 제거 로직 단위 테스트

이 테스트는 _consolidate_expanded_query_results 메서드의 로직을 검증합니다.
"""

import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent
lawfirm_langgraph_dir = project_root / "lawfirm_langgraph"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))


def consolidate_expanded_query_results_logic(
    semantic_results: List[Dict[str, Any]],
    original_query: str
) -> List[Dict[str, Any]]:
    """
    확장된 쿼리 결과 통합 및 중복 제거 로직 (테스트용)
    
    실제 구현과 동일한 로직을 사용하여 테스트합니다.
    """
    if not semantic_results:
        return semantic_results
    
    try:
        # 1. 쿼리별 그룹화 및 통계 수집
        results_by_query = {}
        for doc in semantic_results:
            query_id = (
                doc.get("expanded_query_id") or 
                doc.get("sub_query") or 
                doc.get("query_variation") or
                doc.get("source_query") or
                doc.get("multi_query_source") or
                "original"
            )
            if query_id not in results_by_query:
                results_by_query[query_id] = []
            results_by_query[query_id].append(doc)
        
        # 2. 다층 중복 제거
        seen_ids = set()
        seen_content_hashes = {}  # content_hash -> (doc, score)
        consolidated = []
        
        # 원본 쿼리 결과를 먼저 처리 (높은 우선순위)
        for query_id in sorted(results_by_query.keys(), key=lambda x: 0 if x == "original" else 1):
            results = results_by_query[query_id]
            query_weight = 1.0 if query_id == "original" else 0.9
            
            for doc in results:
                # Layer 1: ID 기반 중복 제거
                doc_id = (
                    doc.get("id") or 
                    doc.get("doc_id") or 
                    doc.get("document_id") or
                    doc.get("metadata", {}).get("chunk_id") or
                    doc.get("chunk_id")
                )
                
                if doc_id and doc_id in seen_ids:
                    continue
                
                # Layer 2: Content Hash 기반 중복 제거
                content = doc.get("content") or doc.get("text", "")
                content_hash = None
                if content:
                    content_hash = hashlib.md5(content[:500].encode('utf-8')).hexdigest()
                
                if content_hash:
                    if content_hash in seen_content_hashes:
                        # 중복 발견: 더 높은 점수를 가진 결과로 교체
                        existing_doc, existing_score = seen_content_hashes[content_hash]
                        new_score = doc.get("relevance_score", doc.get("similarity", 0.0)) * query_weight
                        
                        if new_score > existing_score:
                            # 기존 결과 제거하고 새 결과 추가
                            existing_idx = None
                            for idx, consolidated_doc in enumerate(consolidated):
                                if consolidated_doc is existing_doc:
                                    existing_idx = idx
                                    break
                            
                            if existing_idx is not None:
                                consolidated[existing_idx] = doc
                            else:
                                consolidated.append(doc)
                            seen_content_hashes[content_hash] = (doc, new_score)
                            if doc_id:
                                seen_ids.add(doc_id)
                        # 점수가 같거나 낮으면 무시
                        continue
                    else:
                        # 새로운 content hash
                        score = doc.get("relevance_score", doc.get("similarity", 0.0)) * query_weight
                        seen_content_hashes[content_hash] = (doc, score)
                
                # 중복이 아니므로 추가
                if doc_id:
                    seen_ids.add(doc_id)
                
                # 쿼리 정보 및 가중치 저장
                doc["source_query"] = query_id
                doc["query_weight"] = query_weight
                original_score = doc.get("relevance_score", doc.get("similarity", 0.0))
                doc["weighted_score"] = original_score * query_weight
                
                consolidated.append(doc)
        
        # 3. 가중치가 적용된 점수 기준 정렬
        consolidated.sort(key=lambda x: x.get("weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        
        return consolidated
        
    except Exception as e:
        print(f"Error: {e}")
        return semantic_results


def test_id_based_deduplication():
    """ID 기반 중복 제거 테스트"""
    print("\n=== 테스트 1: ID 기반 중복 제거 ===")
    
    semantic_results = [
        {"id": "doc1", "content": "내용 1", "relevance_score": 0.9, "source_query": "original"},
        {"id": "doc1", "content": "내용 1", "relevance_score": 0.8, "sub_query": "sub1"},  # 중복 ID
        {"id": "doc2", "content": "내용 2", "relevance_score": 0.7, "sub_query": "sub1"},
    ]
    
    result = consolidate_expanded_query_results_logic(semantic_results, "테스트 쿼리")
    
    # 검증
    assert len(result) == 2, f"중복 제거 후 2개여야 함, 실제: {len(result)}"
    assert result[0]["id"] == "doc1", "첫 번째는 doc1이어야 함"
    assert result[1]["id"] == "doc2", "두 번째는 doc2이어야 함"
    
    # 가중치 확인 (원본 쿼리가 더 높은 점수)
    assert result[0]["query_weight"] == 1.0, "원본 쿼리는 가중치 1.0"
    assert result[0]["weighted_score"] == 0.9, "가중치 적용 점수 확인"
    
    print("✅ ID 기반 중복 제거 테스트 통과")


def test_content_hash_deduplication():
    """Content Hash 기반 중복 제거 테스트"""
    print("\n=== 테스트 2: Content Hash 기반 중복 제거 ===")
    
    semantic_results = [
        {"id": "doc1", "content": "계약 해지 사유", "relevance_score": 0.9, "source_query": "original"},
        {"id": "doc2", "content": "계약 해지 사유", "relevance_score": 0.8, "sub_query": "sub1"},  # 동일 내용
        {"id": "doc3", "content": "계약 해지 절차", "relevance_score": 0.7, "sub_query": "sub1"},
    ]
    
    result = consolidate_expanded_query_results_logic(semantic_results, "테스트 쿼리")
    
    # 검증: 동일 내용은 하나만 남아야 함 (더 높은 점수)
    assert len(result) == 2, f"중복 제거 후 2개여야 함, 실제: {len(result)}"
    
    # 첫 번째는 원본 쿼리 결과 (더 높은 점수)
    assert result[0]["id"] == "doc1", "원본 쿼리 결과가 우선"
    assert result[0]["weighted_score"] > result[1]["weighted_score"], "가중치 점수 확인"
    
    print("✅ Content Hash 기반 중복 제거 테스트 통과")


def test_query_weight_application():
    """쿼리별 가중치 적용 테스트"""
    print("\n=== 테스트 3: 쿼리별 가중치 적용 ===")
    
    semantic_results = [
        {"id": "doc1", "content": "내용 1", "relevance_score": 0.8, "source_query": "original"},
        {"id": "doc2", "content": "내용 2", "relevance_score": 0.8, "sub_query": "sub1"},  # 동일 점수
    ]
    
    result = consolidate_expanded_query_results_logic(semantic_results, "테스트 쿼리")
    
    # 검증
    assert len(result) == 2, "모든 결과가 포함되어야 함"
    
    # 가중치 확인
    original_doc = next((d for d in result if d.get("source_query") == "original"), None)
    expanded_doc = next((d for d in result if d.get("source_query") == "sub1"), None)
    
    assert original_doc is not None, "원본 쿼리 결과가 있어야 함"
    assert expanded_doc is not None, "확장된 쿼리 결과가 있어야 함"
    
    assert original_doc["query_weight"] == 1.0, "원본 쿼리는 가중치 1.0"
    assert expanded_doc["query_weight"] == 0.9, "확장된 쿼리는 가중치 0.9"
    
    assert original_doc["weighted_score"] == 0.8, f"원본: 0.8 * 1.0 = 0.8, 실제: {original_doc['weighted_score']}"
    expected_expanded = 0.8 * 0.9
    assert abs(expanded_doc["weighted_score"] - expected_expanded) < 0.001, f"확장: 0.8 * 0.9 = {expected_expanded}, 실제: {expanded_doc['weighted_score']}"
    
    # 정렬 확인 (가중치 점수 기준)
    assert result[0]["weighted_score"] >= result[1]["weighted_score"], "가중치 점수 기준 정렬"
    
    print("✅ 쿼리별 가중치 적용 테스트 통과")


def test_sorting_by_weighted_score():
    """가중치 점수 기준 정렬 테스트"""
    print("\n=== 테스트 4: 가중치 점수 기준 정렬 ===")
    
    semantic_results = [
        {"id": "doc1", "content": "내용 1", "relevance_score": 0.7, "source_query": "original"},  # 0.7 * 1.0 = 0.7
        {"id": "doc2", "content": "내용 2", "relevance_score": 0.8, "sub_query": "sub1"},  # 0.8 * 0.9 = 0.72
        {"id": "doc3", "content": "내용 3", "relevance_score": 0.6, "source_query": "original"},  # 0.6 * 1.0 = 0.6
    ]
    
    result = consolidate_expanded_query_results_logic(semantic_results, "테스트 쿼리")
    
    # 검증: 가중치 점수 기준 정렬
    assert len(result) == 3, "모든 결과가 포함되어야 함"
    
    # 정렬 확인 (내림차순)
    assert result[0]["weighted_score"] >= result[1]["weighted_score"], "첫 번째 >= 두 번째"
    assert result[1]["weighted_score"] >= result[2]["weighted_score"], "두 번째 >= 세 번째"
    
    # 예상 순서: doc2 (0.72) > doc1 (0.7) > doc3 (0.6)
    assert result[0]["id"] == "doc2", "가장 높은 가중치 점수"
    assert result[1]["id"] == "doc1", "두 번째"
    assert result[2]["id"] == "doc3", "세 번째"
    
    print("✅ 가중치 점수 기준 정렬 테스트 통과")


def test_empty_results():
    """빈 결과 처리 테스트"""
    print("\n=== 테스트 5: 빈 결과 처리 ===")
    
    result = consolidate_expanded_query_results_logic([], "테스트 쿼리")
    
    assert result == [], "빈 결과는 빈 리스트 반환"
    
    print("✅ 빈 결과 처리 테스트 통과")


def test_multiple_query_sources():
    """여러 쿼리 소스 처리 테스트"""
    print("\n=== 테스트 6: 여러 쿼리 소스 처리 ===")
    
    semantic_results = [
        {"id": "doc1", "content": "내용 1", "relevance_score": 0.9, "source_query": "original"},
        {"id": "doc2", "content": "내용 2", "relevance_score": 0.8, "sub_query": "sub1"},
        {"id": "doc3", "content": "내용 3", "relevance_score": 0.7, "query_variation": "var1"},
        {"id": "doc4", "content": "내용 4", "relevance_score": 0.6, "multi_query_source": "mq1"},
    ]
    
    result = consolidate_expanded_query_results_logic(semantic_results, "테스트 쿼리")
    
    # 검증
    assert len(result) == 4, "모든 결과가 포함되어야 함"
    
    # 쿼리 소스 확인
    sources = {doc.get("source_query") for doc in result}
    assert "original" in sources, "원본 쿼리 포함"
    assert "sub1" in sources, "sub_query 포함"
    assert "var1" in sources, "query_variation 포함"
    assert "mq1" in sources, "multi_query_source 포함"
    
    print("✅ 여러 쿼리 소스 처리 테스트 통과")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 80)
    print("확장된 쿼리 결과 병합 및 중복 제거 로직 테스트")
    print("=" * 80)
    
    tests = [
        test_id_based_deduplication,
        test_content_hash_deduplication,
        test_query_weight_application,
        test_sorting_by_weighted_score,
        test_empty_results,
        test_multiple_query_sources,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ 테스트 실패: {test_func.__name__}")
            print(f"   오류: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 테스트 오류: {test_func.__name__}")
            print(f"   오류: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"테스트 결과: {passed}개 통과, {failed}개 실패 (총 {len(tests)}개)")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

