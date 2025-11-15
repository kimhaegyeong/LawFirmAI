# -*- coding: utf-8 -*-
"""
문서 포함 개선사항 테스트
- 관련도 임계값 필터링 (0.2 미만 제외)
- 타입별 균형 조정
- 문서 수 증가 (8개 → 20개)
"""

import sys
import os
import json
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from lawfirm_langgraph.core.agents.prompt_builders.unified_prompt_manager import UnifiedPromptManager
from lawfirm_langgraph.core.agents.prompt_builders.unified_prompt_manager import QuestionType

def create_test_documents() -> List[Dict[str, Any]]:
    """테스트용 문서 생성"""
    documents = []
    
    # 법령 조문 (3개)
    for i in range(3):
        documents.append({
            "type": "statute_article",
            "source_type": "statute_article",
            "relevance_score": 0.8 - i * 0.1,
            "content": f"법령 조문 {i+1} 내용",
            "law_name": f"테스트법령{i+1}",
            "article_no": f"{i+1}",
            "document_id": f"statute_{i+1}"
        })
    
    # 판례 (10개, 관련도 다양)
    for i in range(10):
        relevance = 0.85 - i * 0.05
        documents.append({
            "type": "case_paragraph",
            "source_type": "case_paragraph",
            "relevance_score": relevance,
            "content": f"판례 {i+1} 내용",
            "source": f"대법원 202{i}다12345",
            "document_id": f"case_{i+1}"
        })
    
    # 결정례 (3개)
    for i in range(3):
        documents.append({
            "type": "decision_paragraph",
            "source_type": "decision_paragraph",
            "relevance_score": 0.7 - i * 0.1,
            "content": f"결정례 {i+1} 내용",
            "source": f"결정례 {i+1}",
            "document_id": f"decision_{i+1}"
        })
    
    # 해석례 (2개)
    for i in range(2):
        documents.append({
            "type": "interpretation_paragraph",
            "source_type": "interpretation_paragraph",
            "relevance_score": 0.65 - i * 0.1,
            "content": f"해석례 {i+1} 내용",
            "source": f"해석례 {i+1}",
            "document_id": f"interpretation_{i+1}"
        })
    
    # 관련도가 낮은 문서 (0.2 미만, 제외되어야 함)
    for i in range(3):
        documents.append({
            "type": "case_paragraph",
            "source_type": "case_paragraph",
            "relevance_score": 0.15 - i * 0.02,  # 0.15, 0.13, 0.11
            "content": f"낮은 관련도 판례 {i+1}",
            "source": f"낮은 관련도 판례 {i+1}",
            "document_id": f"low_relevance_case_{i+1}"
        })
    
    return documents

def test_document_filtering_and_balancing():
    """문서 필터링 및 균형 조정 테스트"""
    print("\n" + "=" * 80)
    print("Phase 1 테스트: 문서 필터링 및 타입별 균형 조정")
    print("=" * 80)
    
    # 테스트 문서 생성
    test_documents = create_test_documents()
    print(f"\n📚 테스트 문서 생성: 총 {len(test_documents)}개")
    
    # 타입별 분포 확인
    type_distribution = {}
    for doc in test_documents:
        doc_type = doc.get("type", "unknown")
        type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
    
    print(f"\n📊 원본 문서 타입별 분포:")
    for doc_type, count in type_distribution.items():
        print(f"   - {doc_type}: {count}개")
    
    # 관련도 분포 확인
    relevance_scores = [doc.get("relevance_score", 0.0) for doc in test_documents]
    print(f"\n📈 관련도 분포:")
    print(f"   - 최고: {max(relevance_scores):.3f}")
    print(f"   - 최저: {min(relevance_scores):.3f}")
    print(f"   - 평균: {sum(relevance_scores) / len(relevance_scores):.3f}")
    print(f"   - 0.2 미만: {sum(1 for s in relevance_scores if s < 0.2)}개")
    
    # UnifiedPromptManager 초기화
    prompt_manager = UnifiedPromptManager()
    
    # _build_final_prompt 메서드 테스트를 위한 context 구성
    context = {
        "structured_documents": {
            "documents": test_documents,
            "total_count": len(test_documents)
        },
        "document_count": len(test_documents)
    }
    
    # 프롬프트 생성 (문서 섹션만 추출)
    try:
        base_prompt = "테스트 프롬프트"
        query = "전세금 반환 보증에 대해 설명해주세요"
        
        # _build_final_prompt 호출
        final_prompt = prompt_manager._build_final_prompt(
            base_prompt=base_prompt,
            query=query,
            context=context,
            question_type=QuestionType.TERM_EXPLANATION
        )
        
        # 문서 섹션 추출
        if "## 🔍 검색된 법률 문서" in final_prompt:
            print("\n✅ 문서 섹션이 프롬프트에 포함되었습니다.")
            
            # 타입별 문서 섹션 확인
            type_sections = {
                "법령 조문": "📜 법령 조문" in final_prompt,
                "판례": "⚖️ 판례" in final_prompt,
                "결정례": "📋 결정례" in final_prompt,
                "해석례": "📖 해석례" in final_prompt
            }
            
            print(f"\n📋 타입별 문서 섹션 포함 여부:")
            for section_name, included in type_sections.items():
                status = "✅" if included else "❌"
                print(f"   {status} {section_name}")
            
            # 문서 수 확인
            # 각 타입별 문서 개수 추출 (간단한 방법)
            statute_count = final_prompt.count("**1. 법령 조문") or final_prompt.count("📜 법령 조문")
            case_count = final_prompt.count("**1. 판례") or final_prompt.count("⚖️ 판례")
            decision_count = final_prompt.count("**1. 결정례") or final_prompt.count("📋 결정례")
            interpretation_count = final_prompt.count("**1. 해석례") or final_prompt.count("📖 해석례")
            
            # 더 정확한 방법: 문서 번호 추출
            import re
            doc_numbers = re.findall(r'\*\*(\d+)\.', final_prompt)
            total_included = len(set(doc_numbers)) if doc_numbers else 0
            
            print(f"\n📊 프롬프트에 포함된 문서 수:")
            print(f"   - 총 포함된 문서: {total_included}개 이상")
            print(f"   - 목표: 최대 20개 (법령 5개, 판례 7개, 결정례 4개, 해석례 4개)")
            
            # 관련도 0.2 미만 문서가 제외되었는지 확인
            low_relevance_docs = [doc for doc in test_documents if doc.get("relevance_score", 0.0) < 0.2]
            print(f"\n🔍 관련도 0.2 미만 문서:")
            print(f"   - 원본: {len(low_relevance_docs)}개")
            for doc in low_relevance_docs:
                doc_id = doc.get("document_id", "unknown")
                relevance = doc.get("relevance_score", 0.0)
                if doc_id in final_prompt:
                    print(f"   ⚠️ 경고: {doc_id} (관련도: {relevance:.3f})가 포함되어 있습니다!")
                else:
                    print(f"   ✅ {doc_id} (관련도: {relevance:.3f})가 제외되었습니다.")
            
            # 프롬프트 일부 출력
            print(f"\n📝 프롬프트 문서 섹션 미리보기:")
            doc_section_start = final_prompt.find("## 🔍 검색된 법률 문서")
            if doc_section_start >= 0:
                doc_section = final_prompt[doc_section_start:doc_section_start+500]
                print(f"   {doc_section}...")
            
            return True
        else:
            print("\n❌ 문서 섹션이 프롬프트에 포함되지 않았습니다.")
            return False
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_document_filtering_and_balancing()
    sys.exit(0 if success else 1)

