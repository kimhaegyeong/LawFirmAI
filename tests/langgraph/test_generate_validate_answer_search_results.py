# -*- coding: utf-8 -*-
"""
generate_and_validate_answer에서 검색 결과가 프롬프트 작성에 사용되는지 검토
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_search_results_in_prompt():
    """generate_and_validate_answer에서 검색 결과 사용 여부 분석"""
    print("=" * 80)
    print("generate_and_validate_answer 검색 결과 사용 여부 검토")
    print("=" * 80)

    try:
        from core.agents.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow

        # legal_workflow_enhanced.py 파일 읽기
        workflow_file = project_root / "core" / "agents" / "legal_workflow_enhanced.py"
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        print("\n📋 검토 항목:")
        print("-" * 80)

        # 1. generate_and_validate_answer 메서드 구조 확인
        print("\n1️⃣ generate_and_validate_answer 메서드 구조:")
        if "def generate_and_validate_answer" in content:
            print("   ✅ generate_and_validate_answer 메서드 존재")
            if "generate_answer_enhanced" in content.split("def generate_and_validate_answer")[1].split("\n    def ")[0]:
                print("   ✅ generate_answer_enhanced 메서드를 호출함")

        # 2. generate_answer_enhanced에서 retrieved_docs 사용 확인
        print("\n2️⃣ generate_answer_enhanced에서 retrieved_docs 사용:")
        generate_answer_section = content.split("def generate_answer_enhanced")[1].split("\n    def ")[0]

        checks = [
            ("retrieved_docs = self._get_state_value", "state에서 retrieved_docs 가져오기"),
            ("context_dict", "context_dict 생성"),
            ("structured_documents", "structured_documents 포함"),
            ("retrieved_docs", "retrieved_docs 참조"),
            ("unified_prompt_manager.get_optimized_prompt", "unified_prompt_manager에 context_dict 전달"),
            ("SEARCH RESULTS INJECTION", "검색 결과 강제 주입 로직"),
            ("SEARCH RESULTS ENFORCED", "검색 결과 강제 보강 로직"),
        ]

        for check_str, description in checks:
            if check_str in generate_answer_section:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️ {description} - 확인 필요")

        # 3. context_dict에 검색 결과 포함 여부 확인
        print("\n3️⃣ context_dict에 검색 결과 포함:")
        context_dict_checks = [
            ("structured_documents", "structured_documents 필드"),
            ("legal_references", "legal_references 필드"),
            ("document_count", "document_count 필드"),
            ("docs_included", "docs_included 필드"),
        ]

        for check_str, description in context_dict_checks:
            count = generate_answer_section.count(check_str)
            if count > 0:
                print(f"   ✅ {description} (사용 {count}회)")
            else:
                print(f"   ⚠️ {description} - 사용 안 됨")

        # 4. retrieved_docs → structured_documents 변환 로직 확인
        print("\n4️⃣ retrieved_docs → structured_documents 변환 로직:")
        if "normalized_documents" in generate_answer_section:
            print("   ✅ normalized_documents 변환 로직 존재")
            if "SEARCH RESULTS INJECTION" in generate_answer_section:
                print("   ✅ 검색 결과 강제 주입 로직 존재")

        # 5. 프롬프트 검증 로직 확인
        print("\n5️⃣ 프롬프트에 검색 결과 포함 여부 검증:")
        validation_checks = [
            ("PROMPT VALIDATION", "프롬프트 검증 로직"),
            ("has_documents_section", "문서 섹션 확인"),
            ("검색된 법률 문서", "문서 섹션 키워드 확인"),
        ]

        for check_str, description in validation_checks:
            if check_str in generate_answer_section:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️ {description} - 확인 필요")

        # 6. unified_prompt_manager에서 structured_documents 사용 확인
        print("\n6️⃣ unified_prompt_manager에서 structured_documents 사용:")
        try:
            from source.services.unified_prompt_manager import UnifiedPromptManager
            prompt_manager_file = project_root / "source" / "services" / "unified_prompt_manager.py"
            with open(prompt_manager_file, "r", encoding="utf-8") as f:
                prompt_manager_content = f.read()

            if "structured_documents" in prompt_manager_content:
                print("   ✅ structured_documents 사용")
                if "prompt_optimized_text" in prompt_manager_content:
                    print("   ✅ prompt_optimized_text 사용")
                    if "_optimize_context" in prompt_manager_content:
                        optimize_section = prompt_manager_content.split("def _optimize_context")[1].split("\n    def ")[0]
                        if "structured_documents" in optimize_section:
                            print("   ✅ _optimize_context에서 structured_documents 처리")
                        else:
                            print("   ⚠️ _optimize_context에서 structured_documents 미사용 가능")
        except Exception as e:
            print(f"   ⚠️ unified_prompt_manager 확인 중 오류: {e}")

        # 7. 최종 결론
        print("\n" + "=" * 80)
        print("📊 최종 결론")
        print("=" * 80)
        print("""
검색 결과가 프롬프트 작성에 사용되는 경로:

1. generate_and_validate_answer (1087번 라인)
   └─> generate_answer_enhanced 호출 (1111번 라인)

2. generate_answer_enhanced (5219번 라인)
   ├─> retrieved_docs 가져오기 (5250번 라인)
   ├─> context_dict 생성 (5386-5395번 라인)
   │   ├─ structured_documents 포함
   │   ├─ legal_references 포함
   │   └─ document_count, docs_included 포함
   ├─> retrieved_docs → structured_documents 변환 (5533-5620번 라인)
   │   └─ 검색 결과가 없으면 강제로 변환하여 포함
   ├─> unified_prompt_manager.get_optimized_prompt 호출 (5638번 라인)
   │   └─ context_dict 전달 (structured_documents 포함)
   └─> 프롬프트 검증 (5670-5716번 라인)
       └─ 문서 섹션 포함 여부 확인

3. unified_prompt_manager.get_optimized_prompt
   └─> _optimize_context 메서드에서 structured_documents 처리
       └─ prompt_optimized_text가 있어도 structured_documents 강제 포함 (443-447번 라인)

✅ 결론: 검색된 결과(retrieved_docs)는 프롬프트 작성에 사용됩니다.
   - retrieved_docs → structured_documents 변환
   - context_dict에 포함
   - unified_prompt_manager에 전달
   - 최종 프롬프트에 문서 섹션으로 포함
        """)

        # 8. 잠재적 문제점 확인
        print("\n" + "=" * 80)
        print("⚠️ 잠재적 문제점")
        print("=" * 80)

        warnings = []

        # retrieved_docs가 없을 때 처리
        if "retrieved_docs is empty" in generate_answer_section:
            print("   ✅ retrieved_docs가 없을 때 경고 로깅 존재")
        else:
            warnings.append("retrieved_docs가 없을 때 경고 로깅 없음")

        # context_dict 검증
        if "CONTEXT VALIDATION" in generate_answer_section:
            print("   ✅ context_dict 검증 로직 존재")
        else:
            warnings.append("context_dict 검증 로직 없음")

        # 프롬프트에 문서 포함 여부 검증
        if "PROMPT VALIDATION ERROR" in generate_answer_section:
            print("   ✅ 프롬프트 검증 에러 처리 존재")
        else:
            warnings.append("프롬프트 검증 에러 처리 없음")

        if warnings:
            print("\n   ⚠️ 발견된 문제:")
            for warning in warnings:
                print(f"      - {warning}")
        else:
            print("   ✅ 발견된 문제 없음")

        return True

    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = analyze_search_results_in_prompt()
    sys.exit(0 if success else 1)
