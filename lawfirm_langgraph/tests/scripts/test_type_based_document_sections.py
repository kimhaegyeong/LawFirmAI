# -*- coding: utf-8 -*-
"""
타입별 문서 섹션 생성 테스트
- 법령/판례/결정례/해석례 섹션 분리
- 각 타입별 최대 문서 수 확인
"""

import sys
import os
import re
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from lawfirm_langgraph.core.agents.prompt_builders.unified_prompt_manager import UnifiedPromptManager
from lawfirm_langgraph.core.agents.prompt_builders.unified_prompt_manager import QuestionType

def create_balanced_test_documents() -> List[Dict[str, Any]]:
    """균형잡힌 테스트 문서 생성"""
    documents = []
    
    # 법령 조문 (6개, 최대 5개 포함 예상)
    for i in range(6):
        documents.append({
            "type": "statute_article",
            "source_type": "statute_article",
            "relevance_score": 0.85 - i * 0.1,
            "content": f"법령 조문 {i+1} 내용",
            "law_name": f"테스트법령{i+1}",
            "article_no": f"{i+1}",
            "document_id": f"statute_{i+1}"
        })
    
    # 판례 (10개, 최대 7개 포함 예상)
    for i in range(10):
        documents.append({
            "type": "case_paragraph",
            "source_type": "case_paragraph",
            "relevance_score": 0.9 - i * 0.05,
            "content": f"판례 {i+1} 내용",
            "source": f"대법원 202{i}다12345",
            "document_id": f"case_{i+1}"
        })
    
    # 결정례 (5개, 최대 4개 포함 예상)
    for i in range(5):
        documents.append({
            "type": "decision_paragraph",
            "source_type": "decision_paragraph",
            "relevance_score": 0.8 - i * 0.1,
            "content": f"결정례 {i+1} 내용",
            "source": f"결정례 {i+1}",
            "document_id": f"decision_{i+1}"
        })
    
    # 해석례 (5개, 최대 4개 포함 예상)
    for i in range(5):
        documents.append({
            "type": "interpretation_paragraph",
            "source_type": "interpretation_paragraph",
            "relevance_score": 0.75 - i * 0.1,
            "content": f"해석례 {i+1} 내용",
            "source": f"해석례 {i+1}",
            "document_id": f"interpretation_{i+1}"
        })
    
    return documents

def count_documents_by_type(prompt: str) -> Dict[str, int]:
    """프롬프트에서 타입별 문서 수 계산"""
    counts = {
        "statute_article": 0,
        "case_paragraph": 0,
        "decision_paragraph": 0,
        "interpretation_paragraph": 0
    }
    
    # 각 섹션 찾기
    statute_section = re.search(r'### 📜 법령 조문\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    case_section = re.search(r'### ⚖️ 판례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    decision_section = re.search(r'### 📋 결정례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    interpretation_section = re.search(r'### 📖 해석례\n\n(.*?)(?=###|$)', prompt, re.DOTALL)
    
    # 각 섹션에서 문서 번호 추출
    if statute_section:
        counts["statute_article"] = len(re.findall(r'\*\*문서 \d+', statute_section.group(1)))
    
    if case_section:
        counts["case_paragraph"] = len(re.findall(r'\*\*문서 \d+', case_section.group(1)))
    
    if decision_section:
        counts["decision_paragraph"] = len(re.findall(r'\*\*문서 \d+', decision_section.group(1)))
    
    if interpretation_section:
        counts["interpretation_paragraph"] = len(re.findall(r'\*\*문서 \d+', interpretation_section.group(1)))
    
    return counts

def test_type_based_document_sections():
    """타입별 문서 섹션 생성 테스트"""
    print("\n" + "=" * 80)
    print("Phase 2 테스트: 타입별 문서 섹션 생성")
    print("=" * 80)
    
    # 테스트 문서 생성
    test_documents = create_balanced_test_documents()
    print(f"\n📚 테스트 문서 생성: 총 {len(test_documents)}개")
    
    # 타입별 분포 확인
    type_distribution = {}
    for doc in test_documents:
        doc_type = doc.get("type", "unknown")
        type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
    
    print(f"\n📊 원본 문서 타입별 분포:")
    for doc_type, count in type_distribution.items():
        print(f"   - {doc_type}: {count}개")
    
    # UnifiedPromptManager 초기화
    prompt_manager = UnifiedPromptManager()
    
    # context 구성
    context = {
        "structured_documents": {
            "documents": test_documents,
            "total_count": len(test_documents)
        },
        "document_count": len(test_documents)
    }
    
    # 프롬프트 생성
    try:
        base_prompt = "테스트 프롬프트"
        query = "전세금 반환 보증에 대해 설명해주세요"
        
        final_prompt = prompt_manager._build_final_prompt(
            base_prompt=base_prompt,
            query=query,
            context=context,
            question_type=QuestionType.TERM_EXPLANATION
        )
        
        # 타입별 문서 수 계산
        type_counts = count_documents_by_type(final_prompt)
        
        print(f"\n📊 프롬프트에 포함된 타입별 문서 수:")
        expected_counts = {
            "statute_article": 5,
            "case_paragraph": 7,
            "decision_paragraph": 4,
            "interpretation_paragraph": 4
        }
        
        total_included = 0
        for doc_type, count in type_counts.items():
            expected = expected_counts.get(doc_type, 0)
            status = "✅" if count <= expected else "⚠️"
            print(f"   {status} {doc_type}: {count}개 (예상: 최대 {expected}개)")
            total_included += count
        
        print(f"\n   총 포함된 문서: {total_included}개 (목표: 최대 20개)")
        
        # 검증
        checks = []
        
        # 1. 각 타입별 최대 문서 수 확인
        if type_counts["statute_article"] <= 5:
            checks.append(("법령 조문 최대 5개", True))
        else:
            checks.append(("법령 조문 최대 5개", False, f"실제: {type_counts['statute_article']}개"))
        
        if type_counts["case_paragraph"] <= 7:
            checks.append(("판례 최대 7개", True))
        else:
            checks.append(("판례 최대 7개", False, f"실제: {type_counts['case_paragraph']}개"))
        
        if type_counts["decision_paragraph"] <= 4:
            checks.append(("결정례 최대 4개", True))
        else:
            checks.append(("결정례 최대 4개", False, f"실제: {type_counts['decision_paragraph']}개"))
        
        if type_counts["interpretation_paragraph"] <= 4:
            checks.append(("해석례 최대 4개", True))
        else:
            checks.append(("해석례 최대 4개", False, f"실제: {type_counts['interpretation_paragraph']}개"))
        
        # 2. 총 문서 수 확인
        if total_included <= 20:
            checks.append(("총 문서 수 최대 20개", True))
        else:
            checks.append(("총 문서 수 최대 20개", False, f"실제: {total_included}개"))
        
        # 3. 모든 타입이 포함되었는지 확인
        all_types_included = all(count > 0 for count in type_counts.values())
        checks.append(("모든 타입 포함", all_types_included))
        
        # 결과 출력
        print(f"\n✅ 검증 결과:")
        passed = 0
        failed = 0
        for check in checks:
            if len(check) == 2:
                check_name, result = check
                detail = ""
            else:
                check_name, result, detail = check
            
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}{f': {detail}' if detail else ''}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n   총 {len(checks)}개 검증 중 {passed}개 통과, {failed}개 실패")
        
        # 프롬프트 섹션 미리보기
        print(f"\n📝 프롬프트 타입별 섹션 미리보기:")
        for doc_type, section_name in [
            ("statute_article", "📜 법령 조문"),
            ("case_paragraph", "⚖️ 판례"),
            ("decision_paragraph", "📋 결정례"),
            ("interpretation_paragraph", "📖 해석례")
        ]:
            section_match = re.search(f'### {section_name}\\n\\n(.*?)(?=###|$)', final_prompt, re.DOTALL)
            if section_match:
                section_content = section_match.group(1)[:200]
                print(f"\n   {section_name}:")
                print(f"   {section_content}...")
        
        return failed == 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_type_based_document_sections()
    sys.exit(0 if success else 1)

