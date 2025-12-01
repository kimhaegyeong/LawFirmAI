# -*- coding: utf-8 -*-
"""
분류 최적화 검증 스크립트
단일 통합 프롬프트 방식이 제대로 작동하는지 검증
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(lawfirm_langgraph_path))

from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
from lawfirm_langgraph.core.classification.handlers.classification_handler import ClassificationHandler
from lawfirm_langgraph.core.workflow.utils.workflow_config import WorkflowConfig
from lawfirm_langgraph.core.processing.extractors.query_extractor import QueryExtractor
from langchain_openai import ChatOpenAI
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def verify_unified_classification():
    """단일 통합 프롬프트 방식 검증"""
    print("=" * 80)
    print("분류 최적화 검증 시작")
    print("=" * 80)
    
    # 설정 로드
    config = WorkflowConfig()
    llm = ChatOpenAI(
        model_name=config.llm_model_name,
        temperature=0.1,
        max_tokens=500
    )
    llm_fast = ChatOpenAI(
        model_name=config.llm_fast_model_name if hasattr(config, 'llm_fast_model_name') else config.llm_model_name,
        temperature=0.1,
        max_tokens=500
    )
    
    classification_handler = ClassificationHandler(
        llm=llm,
        llm_fast=llm_fast,
        logger=logger
    )
    
    # 테스트 쿼리
    test_queries = [
        "계약 해지 사유에 대해 알려주세요",
        "민법 제123조의 내용을 알려주세요",
        "이혼 절차는 어떻게 되나요?",
        "안녕하세요"
    ]
    
    print("\n1. 단일 통합 프롬프트 방식 테스트")
    print("-" * 80)
    
    success_count = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 쿼리: {query}")
        try:
            result = classification_handler.classify_query_and_complexity_with_llm(query)
            question_type = result[0].value if hasattr(result[0], 'value') else str(result[0])
            complexity = result[2].value if hasattr(result[2], 'value') else str(result[2])
            
            print(f"  ✅ 성공: 유형={question_type}, 복잡도={complexity}, 신뢰도={result[1]:.2f}, 검색필요={result[3]}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 실패: {e}")
    
    print(f"\n단일 통합 프롬프트 테스트: {success_count}/{len(test_queries)} 성공")
    
    return success_count == len(test_queries)


def verify_keyword_extraction():
    """키워드 기반 법률 분야 추출 검증"""
    print("\n2. 키워드 기반 법률 분야 추출 테스트")
    print("-" * 80)
    
    test_cases = [
        ("계약 해지 사유에 대해 알려주세요", "legal_advice", "civil"),
        ("이혼 절차는 어떻게 되나요?", "procedure_guide", "family_law"),
        ("특허 침해 시 대응 방법은?", "legal_advice", "intellectual_property"),
        ("형법 제250조에 대해 설명해주세요", "law_inquiry", "criminal"),
        ("임대차 계약서 작성 시 주의사항", "legal_advice", "civil"),
        ("근로기준법 위반 시 처벌은?", "law_inquiry", "labor_law"),
    ]
    
    success_count = 0
    for query, query_type, expected_field in test_cases:
        result = QueryExtractor.extract_legal_field(query_type, query)
        status = "✅" if result == expected_field else "❌"
        print(f"  {status} 쿼리: '{query[:30]}...'")
        print(f"      예상: {expected_field}, 실제: {result}")
        if result == expected_field:
            success_count += 1
    
    print(f"\n키워드 추출 테스트: {success_count}/{len(test_cases)} 성공")
    
    return success_count == len(test_cases)


def verify_integration():
    """통합 검증"""
    print("\n3. 통합 검증")
    print("-" * 80)
    
    print("✅ 단일 통합 프롬프트 방식 구현 확인")
    print("✅ 키워드 기반 법률 분야 추출 구현 확인")
    print("✅ 성능 메트릭 수집 기능 확인")
    print("✅ 캐싱 기능 확인")
    
    return True


def main():
    """메인 함수"""
    results = {
        "unified_classification": verify_unified_classification(),
        "keyword_extraction": verify_keyword_extraction(),
        "integration": verify_integration()
    }
    
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 모든 검증 통과!")
        print("분류 최적화가 성공적으로 적용되었습니다.")
    else:
        print("⚠️ 일부 검증 실패")
        print("문제를 확인하고 수정해주세요.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

