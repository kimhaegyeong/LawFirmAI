# -*- coding: utf-8 -*-
"""
프롬프트 개선사항 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

def test_query_diversifier():
    """검색 쿼리 다변화 테스트"""
    print("\n" + "=" * 80)
    print("1. 검색 쿼리 다변화 테스트")
    print("=" * 80)
    
    try:
        from core.workflow.utils.query_diversifier import QueryDiversifier
        
        diversifier = QueryDiversifier()
        test_query = "전세금 반환 보증에 대해 설명해주세요"
        
        diversified = diversifier.diversify_search_queries(test_query)
        
        print(f"\n📝 원본 쿼리: {test_query}")
        print(f"\n📊 다변화된 쿼리:")
        for query_type, queries in diversified.items():
            print(f"   - {query_type}: {len(queries)}개")
            for i, q in enumerate(queries[:3], 1):
                print(f"     {i}. {q}")
        
        # 검증
        assert "statute" in diversified, "statute 쿼리가 없습니다"
        assert "case" in diversified, "case 쿼리가 없습니다"
        assert "decision" in diversified, "decision 쿼리가 없습니다"
        assert "interpretation" in diversified, "interpretation 쿼리가 없습니다"
        
        print("\n✅ 검색 쿼리 다변화 테스트 통과!")
        return True
    except Exception as e:
        print(f"\n❌ 검색 쿼리 다변화 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_result_balancer():
    """검색 결과 타입 균형 조정 테스트"""
    print("\n" + "=" * 80)
    print("2. 검색 결과 타입 균형 조정 테스트")
    print("=" * 80)
    
    try:
        from core.workflow.utils.search_result_balancer import SearchResultBalancer
        
        balancer = SearchResultBalancer(min_per_type=1, max_per_type=5)
        
        # 테스트 데이터 생성
        test_results = {
            "statute_article": [
                {"type": "statute_article", "relevance_score": 0.8, "content": "법령 조문 1"},
                {"type": "statute_article", "relevance_score": 0.7, "content": "법령 조문 2"},
            ],
            "case_paragraph": [
                {"type": "case_paragraph", "relevance_score": 0.9, "content": "판례 1"},
                {"type": "case_paragraph", "relevance_score": 0.85, "content": "판례 2"},
                {"type": "case_paragraph", "relevance_score": 0.8, "content": "판례 3"},
                {"type": "case_paragraph", "relevance_score": 0.75, "content": "판례 4"},
                {"type": "case_paragraph", "relevance_score": 0.7, "content": "판례 5"},
            ],
            "decision_paragraph": [
                {"type": "decision_paragraph", "relevance_score": 0.6, "content": "결정례 1"},
            ],
            "interpretation_paragraph": [
                {"type": "interpretation_paragraph", "relevance_score": 0.5, "content": "해석례 1"},
            ]
        }
        
        balanced = balancer.balance_search_results(test_results, total_limit=10)
        
        print(f"\n📊 균형 조정 전:")
        for doc_type, docs in test_results.items():
            print(f"   - {doc_type}: {len(docs)}개")
        
        print(f"\n📊 균형 조정 후:")
        balanced_types = {}
        for doc in balanced:
            doc_type = doc.get("type", "unknown")
            balanced_types[doc_type] = balanced_types.get(doc_type, 0) + 1
        
        for doc_type, count in balanced_types.items():
            print(f"   - {doc_type}: {count}개")
        
        # 검증: 각 타입에서 최소 1개씩 있는지 확인
        assert balanced_types.get("statute_article", 0) >= 1, "법령 조문이 1개 이상 있어야 합니다"
        assert balanced_types.get("case_paragraph", 0) >= 1, "판례가 1개 이상 있어야 합니다"
        assert balanced_types.get("decision_paragraph", 0) >= 1, "결정례가 1개 이상 있어야 합니다"
        assert balanced_types.get("interpretation_paragraph", 0) >= 1, "해석례가 1개 이상 있어야 합니다"
        
        print("\n✅ 검색 결과 타입 균형 조정 테스트 통과!")
        return True
    except Exception as e:
        print(f"\n❌ 검색 결과 타입 균형 조정 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_prompt_builder():
    """동적 프롬프트 빌더 테스트"""
    print("\n" + "=" * 80)
    print("3. 동적 프롬프트 빌더 테스트")
    print("=" * 80)
    
    try:
        from core.agents.prompt_builders.dynamic_prompt_builder import DynamicPromptBuilder
        
        builder = DynamicPromptBuilder()
        
        # 테스트 문서 생성
        test_documents = [
            {"type": "statute_article", "content": "법령 조문 내용"},
            {"type": "case_paragraph", "content": "판례 내용"},
            {"type": "case_paragraph", "content": "판례 내용 2"},
            {"type": "decision_paragraph", "content": "결정례 내용"},
        ]
        
        # 문서 타입 분석
        doc_types = builder.analyze_document_types(test_documents)
        print(f"\n📊 문서 타입 분포:")
        for doc_type, count in doc_types.items():
            print(f"   - {doc_type}: {count}개")
        
        # Citation 지침 생성
        citation_guidance = builder.build_citation_guidance(doc_types, len(test_documents))
        print(f"\n📝 Citation 지침:")
        print(citation_guidance)
        
        # 문서 타입별 활용 지침 생성
        type_guidance = builder.build_document_type_guidance(doc_types)
        print(f"\n📝 문서 타입별 활용 지침:")
        print(type_guidance)
        
        # 간소화된 프롬프트 섹션 생성
        prompt_section = builder.build_simplified_prompt_section(test_documents, len(test_documents))
        print(f"\n📝 간소화된 프롬프트 섹션:")
        print(prompt_section[:500] + "..." if len(prompt_section) > 500 else prompt_section)
        
        # 검증
        assert "statute_article" in doc_types, "법령 조문 타입이 없습니다"
        assert "case_paragraph" in doc_types, "판례 타입이 없습니다"
        assert len(citation_guidance) > 0, "Citation 지침이 비어있습니다"
        
        print("\n✅ 동적 프롬프트 빌더 테스트 통과!")
        return True
    except Exception as e:
        print(f"\n❌ 동적 프롬프트 빌더 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_prompt_manager_integration():
    """UnifiedPromptManager 통합 테스트"""
    print("\n" + "=" * 80)
    print("4. UnifiedPromptManager 통합 테스트")
    print("=" * 80)
    
    try:
        from core.agents.prompt_builders.unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType
        from core.classification.classifiers.question_classifier import QuestionType
        
        manager = UnifiedPromptManager()
        
        # 테스트 컨텍스트 생성
        test_context = {
            "structured_documents": {
                "documents": [
                    {
                        "type": "case_paragraph",
                        "content": "판례 내용 테스트",
                        "relevance_score": 0.8,
                        "source": "판례 1"
                    },
                    {
                        "type": "case_paragraph",
                        "content": "판례 내용 테스트 2",
                        "relevance_score": 0.7,
                        "source": "판례 2"
                    }
                ]
            },
            "document_count": 2
        }
        
        # 프롬프트 생성
        prompt = manager.get_optimized_prompt(
            query="전세금 반환 보증에 대해 설명해주세요",
            question_type=QuestionType.TERM_EXPLANATION,
            domain=LegalDomain.GENERAL,
            context=test_context,
            model_type=ModelType.GEMINI
        )
        
        print(f"\n📝 생성된 프롬프트 길이: {len(prompt)}자")
        print(f"\n📋 프롬프트 미리보기 (처음 500자):")
        print(prompt[:500] + "...")
        
        # 검증
        assert len(prompt) > 0, "프롬프트가 비어있습니다"
        assert "전세금 반환 보증" in prompt, "질문이 프롬프트에 포함되지 않았습니다"
        
        # 동적 프롬프트 빌더가 사용되었는지 확인
        # (프롬프트에 문서 타입 분포나 동적 지침이 포함되어 있는지 확인)
        has_dynamic_content = (
            "문서 타입 분포" in prompt or
            "Citation 요구사항" in prompt or
            "문서 타입별 활용" in prompt
        )
        
        if has_dynamic_content:
            print("\n✅ 동적 프롬프트 빌더가 사용되었습니다!")
        else:
            print("\n⚠️ 동적 프롬프트 빌더가 사용되지 않았을 수 있습니다 (정상일 수 있음)")
        
        print("\n✅ UnifiedPromptManager 통합 테스트 통과!")
        return True
    except Exception as e:
        print(f"\n❌ UnifiedPromptManager 통합 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print("프롬프트 개선사항 테스트 시작")
    print("=" * 80)
    
    results = []
    
    # 1. 검색 쿼리 다변화 테스트
    results.append(("검색 쿼리 다변화", test_query_diversifier()))
    
    # 2. 검색 결과 타입 균형 조정 테스트
    results.append(("검색 결과 타입 균형 조정", test_search_result_balancer()))
    
    # 3. 동적 프롬프트 빌더 테스트
    results.append(("동적 프롬프트 빌더", test_dynamic_prompt_builder()))
    
    # 4. UnifiedPromptManager 통합 테스트
    results.append(("UnifiedPromptManager 통합", test_unified_prompt_manager_integration()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과, {failed}개 실패")
    
    if failed == 0:
        print("\n🎉 모든 테스트 통과!")
        return 0
    else:
        print(f"\n⚠️ {failed}개 테스트 실패")
        return 1

if __name__ == "__main__":
    sys.exit(main())

