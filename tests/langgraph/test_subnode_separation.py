# -*- coding: utf-8 -*-
"""
서브노드 분리 테스트 (expand_keywords + prepare_search_query)
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.agents.workflow_service import LangGraphWorkflowService
from infrastructure.utils.langgraph_config import LangGraphConfig


async def test_expand_keywords_node():
    """expand_keywords 노드 테스트"""
    print("\n" + "=" * 80)
    print("테스트 1: expand_keywords 노드 동작 확인")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "계약 해지 요건"
    print(f"\n질문: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_expand_keywords")
    elapsed = time.time() - start
    
    # processing_steps 확인
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # expand_keywords 노드 실행 확인 (더 넓은 범위로 검색)
    has_keyword_expansion = any(
        "키워드 확장" in step or 
        "키워드" in step and "확장" in step or
        "expand_keywords" in step.lower()
        for step in step_texts
    )
    
    # extracted_keywords 확인
    extracted_keywords = result.get("extracted_keywords", [])
    
    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  키워드 확장 단계 포함: {has_keyword_expansion}")
    print(f"  추출된 키워드 수: {len(extracted_keywords)}개")
    print(f"  처리 단계 수: {len(processing_steps)}개")
    
    if extracted_keywords:
        print(f"  키워드 예시: {extracted_keywords[:5]}")
    
    success = has_keyword_expansion and len(extracted_keywords) > 0
    
    if success:
        print("  ✅ [PASS] expand_keywords 노드 정상 작동")
    else:
        print("  ❌ [FAIL] expand_keywords 노드 확인 실패")
        print(f"        키워드 확장 포함: {has_keyword_expansion}, 키워드 수: {len(extracted_keywords)}")
    
    return success


async def test_prepare_search_query_node():
    """prepare_search_query 노드 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: prepare_search_query 노드 동작 확인")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "민법 제111조의 내용을 알려주세요"
    print(f"\n질문: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_prepare_query")
    elapsed = time.time() - start
    
    # processing_steps 확인
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # prepare_search_query 노드 실행 확인 (더 넓은 범위로 검색)
    has_search_query_prep = any(
        "검색 쿼리 준비" in step or 
        "쿼리 준비" in step or
        "search_query" in step.lower() or
        "최적화된 쿼리" in step
        for step in step_texts
    )
    
    # optimized_queries 확인
    optimized_queries = result.get("optimized_queries", {})
    search_query = result.get("search_query", "")
    search_params = result.get("search_params", {})
    
    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  검색 쿼리 준비 단계 포함: {has_search_query_prep}")
    print(f"  최적화된 쿼리 생성: {bool(optimized_queries)}")
    print(f"  검색 쿼리: {search_query[:50] if search_query else 'N/A'}...")
    print(f"  검색 파라미터: {bool(search_params)}")
    print(f"  처리 단계 수: {len(processing_steps)}개")
    
    if optimized_queries:
        semantic_query = optimized_queries.get("semantic_query", "")
        print(f"  의미적 쿼리: {semantic_query[:50] if semantic_query else 'N/A'}...")
    
    success = has_search_query_prep and bool(optimized_queries) and bool(search_params)
    
    if success:
        print("  ✅ [PASS] prepare_search_query 노드 정상 작동")
    else:
        print("  ❌ [FAIL] prepare_search_query 노드 확인 실패")
        print(f"        쿼리 준비 포함: {has_search_query_prep}, "
              f"최적화된 쿼리: {bool(optimized_queries)}, "
              f"검색 파라미터: {bool(search_params)}")
    
    return success


async def test_unified_classification():
    """통합 LLM 분류 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 통합 LLM 분류 (질문 유형 + 복잡도) 동작 확인")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    test_cases = [
        ("안녕하세요", "simple"),
        ("민법 제111조", "moderate"),
        ("계약 해지와 해제의 차이", "complex"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_complexity in test_cases:
        print(f"\n📝 질문: {query}")
        print(f"  예상 복잡도: {expected_complexity}")
        
        try:
            start = time.time()
            result = await workflow_service.process_query(query, f"test_session_{passed}")
            elapsed = time.time() - start
            
            actual_complexity = result.get("query_complexity", "unknown")
            needs_search = result.get("needs_search", True)
            query_type = result.get("query_type", "unknown")
            
            print(f"  실제 복잡도: {actual_complexity}")
            print(f"  질문 유형: {query_type}")
            print(f"  검색 필요: {needs_search}")
            print(f"  응답 시간: {elapsed:.2f}초")
            
            # 복잡도 일치 확인
            if actual_complexity == expected_complexity:
                print("  ✅ [PASS] 복잡도 일치")
                passed += 1
            else:
                print(f"  ⚠️  복잡도 불일치 (예상: {expected_complexity}, 실제: {actual_complexity})")
                failed += 1
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            logger.exception(f"테스트 실패: {query}")
            failed += 1
    
    print(f"\n📊 결과: {passed}개 통과, {failed}개 실패")
    return failed == 0


async def test_subnode_sequence():
    """서브노드 순차 실행 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: 서브노드 순차 실행 확인 (expand_keywords → prepare_search_query)")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    query = "계약 해지 요건과 절차"
    print(f"\n질문: {query}")
    
    start = time.time()
    result = await workflow_service.process_query(query, "test_session_sequence")
    elapsed = time.time() - start
    
    # processing_steps 확인
    processing_steps = result.get("processing_steps", [])
    step_texts = []
    for step in processing_steps:
        if isinstance(step, dict):
            step_texts.append(step.get("step", "") or step.get("message", "") or str(step))
        elif isinstance(step, str):
            step_texts.append(step)
        else:
            step_texts.append(str(step))
    
    # 단계 순서 확인 (더 넓은 범위로 검색)
    keyword_expansion_idx = -1
    search_query_prep_idx = -1
    
    for i, step in enumerate(step_texts):
        step_lower = step.lower()
        if "키워드" in step and ("확장" in step or "expansion" in step_lower):
            keyword_expansion_idx = i
        if ("검색 쿼리" in step or "쿼리 준비" in step or 
            "search_query" in step_lower or "최적화된 쿼리" in step):
            search_query_prep_idx = i
    
    print(f"\n[결과]")
    print(f"  시간: {elapsed:.2f}초")
    print(f"  키워드 확장 단계 인덱스: {keyword_expansion_idx}")
    print(f"  검색 쿼리 준비 단계 인덱스: {search_query_prep_idx}")
    print(f"  총 처리 단계 수: {len(processing_steps)}개")
    
    # 순서 확인: 키워드 확장이 검색 쿼리 준비보다 먼저 실행되어야 함
    correct_sequence = (
        keyword_expansion_idx >= 0 and 
        search_query_prep_idx >= 0 and 
        keyword_expansion_idx < search_query_prep_idx
    )
    
    # 결과 확인
    extracted_keywords = result.get("extracted_keywords", [])
    optimized_queries = result.get("optimized_queries", {})
    search_query = result.get("search_query", "")
    
    has_keywords = len(extracted_keywords) > 0
    has_optimized = bool(optimized_queries) and bool(search_query)
    
    print(f"  키워드 확장 결과: {has_keywords} (키워드 {len(extracted_keywords)}개)")
    print(f"  쿼리 최적화 결과: {has_optimized}")
    
    success = correct_sequence and has_keywords and has_optimized
    
    if success:
        print("  ✅ [PASS] 서브노드 순차 실행 정상")
    else:
        print("  ❌ [FAIL] 서브노드 순차 실행 확인 실패")
        print(f"        올바른 순서: {correct_sequence}, "
              f"키워드 있음: {has_keywords}, "
              f"최적화됨: {has_optimized}")
    
    return success


async def test_end_to_end_workflow():
    """전체 워크플로우 엔드-투-엔드 테스트"""
    print("\n" + "=" * 80)
    print("테스트 5: 전체 워크플로우 엔드-투-엔드 테스트")
    print("=" * 80)
    
    config = LangGraphConfig.from_env()
    workflow_service = LangGraphWorkflowService(config)
    
    test_queries = [
        "안녕하세요",
        "민법 제111조의 내용을 알려주세요",
        "계약 해지와 해제의 차이는 무엇인가요?",
    ]
    
    passed = 0
    failed = 0
    
    for query in test_queries:
        print(f"\n📝 질문: {query}")
        
        try:
            start = time.time()
            result = await workflow_service.process_query(query, f"test_session_e2e_{passed}")
            elapsed = time.time() - start
            
            answer = result.get("answer", "")
            query_complexity = result.get("query_complexity", "unknown")
            needs_search = result.get("needs_search", True)
            extracted_keywords = result.get("extracted_keywords", [])
            optimized_queries = result.get("optimized_queries", {})
            
            print(f"  ⏱️  응답 시간: {elapsed:.2f}초")
            print(f"  📊 복잡도: {query_complexity}")
            print(f"  🔍 검색 필요: {needs_search}")
            print(f"  📝 답변 길이: {len(answer)}자")
            print(f"  🔑 키워드 수: {len(extracted_keywords)}개")
            print(f"  🔍 최적화된 쿼리: {bool(optimized_queries)}")
            
            # 기본 검증
            has_answer = len(answer) > 0
            has_complexity = query_complexity != "unknown"
            
            if has_answer and has_complexity:
                print("  ✅ [PASS] 전체 워크플로우 정상 작동")
                passed += 1
            else:
                print("  ❌ [FAIL] 전체 워크플로우 검증 실패")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            logger.exception(f"테스트 실패: {query}")
            failed += 1
    
    print(f"\n📊 결과: {passed}개 통과, {failed}개 실패")
    return failed == 0


async def main():
    """모든 테스트 실행"""
    print("=" * 80)
    print("서브노드 분리 및 통합 LLM 분류 테스트")
    print("=" * 80)
    
    results = {}
    
    # 테스트 1: expand_keywords 노드
    try:
        results["expand_keywords"] = await test_expand_keywords_node()
    except Exception as e:
        print(f"❌ expand_keywords 테스트 오류: {e}")
        logger.exception("expand_keywords 테스트 실패")
        results["expand_keywords"] = False
    
    # 테스트 2: prepare_search_query 노드
    try:
        results["prepare_search_query"] = await test_prepare_search_query_node()
    except Exception as e:
        print(f"❌ prepare_search_query 테스트 오류: {e}")
        logger.exception("prepare_search_query 테스트 실패")
        results["prepare_search_query"] = False
    
    # 테스트 3: 통합 LLM 분류
    try:
        results["unified_classification"] = await test_unified_classification()
    except Exception as e:
        print(f"❌ 통합 LLM 분류 테스트 오류: {e}")
        logger.exception("통합 LLM 분류 테스트 실패")
        results["unified_classification"] = False
    
    # 테스트 4: 서브노드 순차 실행
    try:
        results["subnode_sequence"] = await test_subnode_sequence()
    except Exception as e:
        print(f"❌ 서브노드 순차 실행 테스트 오류: {e}")
        logger.exception("서브노드 순차 실행 테스트 실패")
        results["subnode_sequence"] = False
    
    # 테스트 5: 엔드-투-엔드
    try:
        results["end_to_end"] = await test_end_to_end_workflow()
    except Exception as e:
        print(f"❌ 엔드-투-엔드 테스트 오류: {e}")
        logger.exception("엔드-투-엔드 테스트 실패")
        results["end_to_end"] = False
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("테스트 최종 결과")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\n📊 총계: {total_passed}/{total_tests} 테스트 통과")
    
    if total_passed == total_tests:
        print("✅ 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

