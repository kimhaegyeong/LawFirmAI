# -*- coding: utf-8 -*-
"""
Multi-Query 프롬프트 테스트 스크립트
새로운 프롬프트 형식과 max_queries 제어 테스트

Usage:
    python lawfirm_langgraph/tests/scripts/test_multi_query_prompt.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
scripts_dir = script_dir.parent
tests_dir = scripts_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

import logging
from typing import List

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 테스트 쿼리
TEST_QUERIES = [
    {
        "query": "계약 해지 사유에 대해 알려주세요",
        "query_type": "legal_advice",
        "max_queries": 3
    },
    {
        "query": "손해배상 청구 요건은 무엇인가요?",
        "query_type": "legal_advice",
        "max_queries": 4
    },
    {
        "query": "민법 제750조",
        "query_type": "statute",
        "max_queries": 5
    }
]


def test_multi_query_generation():
    """Multi-Query 생성 테스트"""
    print("=" * 80)
    print("Multi-Query 프롬프트 테스트 시작")
    print("=" * 80)
    
    try:
        from core.workflow.initializers.llm_initializer import LLMInitializer
        from core.utils.langgraph_config import LangGraphConfig
        from core.workflow.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
        
        # 설정 로드
        try:
            config = LangGraphConfig.from_env()
        except:
            from core.utils.config import Config
            config = Config()
        
        # 워크플로우 초기화
        workflow = EnhancedLegalQuestionWorkflow(config)
        
        # 각 테스트 쿼리 실행
        results = []
        for i, test_case in enumerate(TEST_QUERIES, 1):
            print(f"\n{'=' * 80}")
            print(f"테스트 {i}/{len(TEST_QUERIES)}")
            print(f"쿼리: {test_case['query']}")
            print(f"질문 유형: {test_case['query_type']}")
            print(f"최대 질문 수: {test_case['max_queries']}")
            print(f"{'=' * 80}")
            
            try:
                # Multi-Query 생성
                multi_queries = workflow._generate_multi_queries_with_llm(
                    query=test_case["query"],
                    query_type=test_case["query_type"],
                    max_queries=test_case["max_queries"],
                    use_cache=False  # 캐시 비활성화하여 실제 LLM 호출
                )
                
                print(f"\n✅ 생성된 질문 수: {len(multi_queries)}/{test_case['max_queries']}")
                print(f"\n생성된 질문 목록:")
                for j, q in enumerate(multi_queries, 1):
                    print(f"  {j}. {q}")
                
                # 검증
                is_valid = (
                    len(multi_queries) <= test_case["max_queries"] and
                    len(multi_queries) >= 1 and
                    test_case["query"] in multi_queries  # 원본 포함 확인
                )
                
                results.append({
                    "test_case": test_case,
                    "multi_queries": multi_queries,
                    "count": len(multi_queries),
                    "expected_count": test_case["max_queries"],
                    "is_valid": is_valid,
                    "success": is_valid
                })
                
                if is_valid:
                    print(f"\n✅ 검증 통과")
                else:
                    print(f"\n❌ 검증 실패")
                    
            except Exception as e:
                logger.error(f"❌ 테스트 쿼리 실행 실패: {e}", exc_info=True)
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "success": False
                })
        
        # 결과 요약
        print(f"\n{'=' * 80}")
        print("테스트 결과 요약")
        print(f"{'=' * 80}")
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"✅ 성공: {success_count}/{len(TEST_QUERIES)}")
        print(f"❌ 실패: {len(TEST_QUERIES) - success_count}/{len(TEST_QUERIES)}")
        
        # 상세 결과
        for i, result in enumerate(results, 1):
            if result.get("success"):
                print(f"\n테스트 {i}: ✅")
                print(f"  생성된 질문 수: {result.get('count', 0)}/{result.get('expected_count', 0)}")
            else:
                print(f"\n테스트 {i}: ❌")
                if "error" in result:
                    print(f"  오류: {result['error']}")
                else:
                    print(f"  생성된 질문 수: {result.get('count', 0)}/{result.get('expected_count', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 테스트 초기화 실패: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("🚀 Multi-Query 프롬프트 테스트 시작\n")
    
    results = test_multi_query_generation()
    
    if results and all(r.get("success", False) for r in results):
        print("\n✅ 모든 테스트 완료")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트 실패")
        sys.exit(1)

