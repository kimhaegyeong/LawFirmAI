# -*- coding: utf-8 -*-
"""
빠른 통합 테스트
UnifiedPromptManager 통합 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"

from source.services.question_classifier import QuestionType
from source.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)


def test_integration():
    """빠른 통합 테스트"""
    print("\n" + "="*80)
    print("UnifiedPromptManager 통합 검증")
    print("="*80 + "\n")

    results = []

    # 1. 인스턴스 생성 테스트
    print("📋 테스트 1: UnifiedPromptManager 인스턴스 생성")
    try:
        manager = UnifiedPromptManager()
        assert manager is not None
        print("   ✅ UnifiedPromptManager 생성 성공")
        results.append(True)
    except Exception as e:
        print(f"   ❌ UnifiedPromptManager 생성 실패: {e}")
        results.append(False)

    # 2. 기본 프롬프트 생성 테스트
    print("\n📋 테스트 2: 기본 프롬프트 생성")
    try:
        prompt = manager.get_optimized_prompt(
            query="이혼 절차에 대해 알려주세요",
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.FAMILY_LAW,
            context={"context": "가족법 관련 질문"},
            model_type=ModelType.GEMINI
        )

        assert prompt and len(prompt) > 0
        print(f"   ✅ 프롬프트 생성 성공 ({len(prompt)}자)")

        # 프롬프트 품질 검증
        if "이혼" in prompt or "가족" in prompt or "가족법" in manager.domain_templates.get(LegalDomain.FAMILY_LAW, {}).get('focus', ''):
            print("   ✅ 도메인 특화 반영 확인")

        results.append(True)
    except Exception as e:
        print(f"   ❌ 프롬프트 생성 실패: {e}")
        results.append(False)

    # 3. 다양한 질문 유형 테스트
    print("\n📋 테스트 3: 다양한 질문 유형")
    question_types = [
        (QuestionType.LEGAL_ADVICE, "법적 조언"),
        (QuestionType.PROCEDURE_GUIDE, "절차 안내"),
        (QuestionType.LAW_INQUIRY, "법률 문의"),
        (QuestionType.GENERAL_QUESTION, "일반 질문"),
    ]

    for qtype, name in question_types:
        try:
            prompt = manager.get_optimized_prompt(
                query="테스트 질문",
                question_type=qtype,
                domain=LegalDomain.CIVIL_LAW,
                context={},
                model_type=ModelType.GEMINI
            )
            assert len(prompt) > 0
            print(f"   ✅ {name}: 생성 성공")
            results.append(True)
        except Exception as e:
            print(f"   ❌ {name}: 생성 실패 - {e}")
            results.append(False)

    # 4. 다양한 도메인 테스트
    print("\n📋 테스트 4: 다양한 도메인")
    domains = [
        (LegalDomain.CIVIL_LAW, "민사법"),
        (LegalDomain.CRIMINAL_LAW, "형사법"),
        (LegalDomain.FAMILY_LAW, "가족법"),
        (LegalDomain.LABOR_LAW, "노동법"),
        (LegalDomain.PROPERTY_LAW, "부동산법"),
    ]

    for domain, name in domains:
        try:
            prompt = manager.get_optimized_prompt(
                query="관련 법률 질문",
                question_type=QuestionType.LEGAL_ADVICE,
                domain=domain,
                context={},
                model_type=ModelType.GEMINI
            )
            assert len(prompt) > 0
            print(f"   ✅ {name}: 프롬프트 생성 성공")
            results.append(True)
        except Exception as e:
            print(f"   ❌ {name}: 프롬프트 생성 실패 - {e}")
            results.append(False)

    # 5. 프롬프트 품질 검증
    print("\n📋 테스트 5: 프롬프트 품질 검증")
    try:
        query = "민법 제750조 불법행위에 대해 알려주세요"
        prompt = manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LAW_INQUIRY,
            domain=LegalDomain.CIVIL_LAW,
            context={"context": "민법 제750조"},
            model_type=ModelType.GEMINI
        )

        # 품질 검증
        checks = {
            "프롬프트 길이": len(prompt) > 100,
            "컨텍스트 포함": "민법" in prompt or "불법행위" in prompt.lower(),
            "질문 포함": query[:10] in prompt or "민법" in prompt,
        }

        for check_name, check_result in checks.items():
            if check_result:
                print(f"   ✅ {check_name}: 통과")
                results.append(True)
            else:
                print(f"   ⚠️ {check_name}: 미달")
                results.append(False)

    except Exception as e:
        print(f"   ❌ 품질 검증 실패: {e}")
        results.append(False)

    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)

    passed = sum(results)
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    print(f"📊 성공률: {success_rate:.1f}%")
    print("="*80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = test_integration()

    if success:
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")

    sys.exit(0 if success else 1)
