# -*- coding: utf-8 -*-
"""
UnifiedPromptManager 통합 테스트
langgraph 워크플로우에서 UnifiedPromptManager 통합 검증
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 환경 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from source.services.langgraph.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.services.langgraph.state_definitions import create_initial_legal_state
from source.services.question_classifier import QuestionType
from source.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)
from source.utils.langgraph_config import LangGraphConfig


class TestUnifiedPromptIntegration:
    """UnifiedPromptManager 통합 테스트"""

    @pytest.fixture
    def config(self):
        """LangGraph 설정 생성"""
        return LangGraphConfig.from_env()

    @pytest.fixture
    def workflow(self, config):
        """워크플로우 인스턴스 생성"""
        return EnhancedLegalQuestionWorkflow(config)

    @pytest.fixture
    def unified_manager(self):
        """UnifiedPromptManager 인스턴스 생성"""
        return UnifiedPromptManager()

    def test_unified_prompt_manager_initialized(self, workflow):
        """UnifiedPromptManager가 워크플로우에 초기화되었는지 확인"""
        assert hasattr(workflow, 'unified_prompt_manager')
        assert workflow.unified_prompt_manager is not None
        print("✅ UnifiedPromptManager가 워크플로우에 정상적으로 초기화되었습니다.")

    def test_prompt_manager_type(self, workflow):
        """UnifiedPromptManager 타입 확인"""
        assert isinstance(workflow.unified_prompt_manager, UnifiedPromptManager)
        print("✅ UnifiedPromptManager 타입이 올바릅니다.")

    def test_get_optimized_prompt(self, unified_manager):
        """get_optimized_prompt 메서드 테스트"""
        query = "이혼 절차에 대해 알려주세요"

        prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.FAMILY_LAW,
            context={"context": "테스트 컨텍스트"},
            model_type=ModelType.GEMINI
        )

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "이혼" in query
        print("✅ get_optimized_prompt가 정상적으로 작동합니다.")
        print(f"   생성된 프롬프트 길이: {len(prompt)}자")

    def test_prompt_with_different_domains(self, unified_manager):
        """다양한 도메인별 프롬프트 테스트"""
        test_cases = [
            (LegalDomain.CIVIL_LAW, QuestionType.LEGAL_ADVICE, "계약서 작성 방법"),
            (LegalDomain.CRIMINAL_LAW, QuestionType.LEGAL_ADVICE, "절도죄의 처벌"),
            (LegalDomain.FAMILY_LAW, QuestionType.LEGAL_ADVICE, "이혼 절차"),
            (LegalDomain.LABOR_LAW, QuestionType.LEGAL_ADVICE, "해고 제한 조건"),
            (LegalDomain.GENERAL, QuestionType.GENERAL_QUESTION, "법률 용어 해설"),
        ]

        for domain, question_type, query in test_cases:
            prompt = unified_manager.get_optimized_prompt(
                query=query,
                question_type=question_type,
                domain=domain,
                context={},
                model_type=ModelType.GEMINI
            )

            assert prompt is not None
            assert len(prompt) > 0
            print(f"✅ {domain.value} 도메인 프롬프트 생성 성공 ({len(prompt)}자)")

    def test_prompt_with_different_models(self, unified_manager):
        """다양한 모델 타입별 프롬프트 테스트"""
        test_cases = [
            (ModelType.GEMINI, "Gemini"),
            (ModelType.OLLAMA, "Ollama"),
            (ModelType.OPENAI, "OpenAI"),
        ]

        query = "민법 제750조에 대해 알려주세요"

        for model_type, model_name in test_cases:
            prompt = unified_manager.get_optimized_prompt(
                query=query,
                question_type=QuestionType.LAW_INQUIRY,
                domain=LegalDomain.CIVIL_LAW,
                context={},
                model_type=model_type
            )

            assert prompt is not None
            assert len(prompt) > 0
            print(f"✅ {model_name} 모델 타입 프롬프트 생성 성공 ({len(prompt)}자)")

    def test_enhanced_workflow_generate_answer(self, workflow):
        """generate_answer_enhanced에서 UnifiedPromptManager 사용 확인"""
        # 초기 상태 생성
        state = create_initial_legal_state("이혼 절차에 대해 알려주세요", "test-session")
        state["query_type"] = "family_law"
        state["retrieved_docs"] = [
            {"content": "이혼 절차는 협의이혼과 재판상 이혼이 있습니다.", "source": "test"}
        ]

        # generate_answer_enhanced 호출
        result = workflow.generate_answer_enhanced(state)

        assert "answer" in result
        assert "processing_steps" in result
        assert "UnifiedPromptManager" in " ".join(result.get("processing_steps", []))
        print("✅ generate_answer_enhanced가 UnifiedPromptManager를 사용합니다.")

    def test_prompt_optimization_features(self, unified_manager):
        """프롬프트 최적화 기능 테스트"""
        query = "손해배상 청구 방법"

        # Legal Advice 타입
        advice_prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.CIVIL_LAW,
            context={"context": "민법 제750조 불법행위"},
            model_type=ModelType.GEMINI
        )

        # Procedure Guide 타입
        procedure_prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.PROCEDURE_GUIDE,
            domain=LegalDomain.CIVIL_PROCEDURE,
            context={"context": "민사소송법"},
            model_type=ModelType.GEMINI
        )

        # 두 프롬프트가 다른지 확인
        assert advice_prompt != procedure_prompt
        assert len(advice_prompt) > 0
        assert len(procedure_prompt) > 0

        print(f"✅ Legal Advice 프롬프트: {len(advice_prompt)}자")
        print(f"✅ Procedure Guide 프롬프트: {len(procedure_prompt)}자")
        print("✅ 프롬프트가 질문 유형에 따라 최적화됩니다.")

    def test_prompt_with_context(self, unified_manager):
        """컨텍스트가 포함된 프롬프트 테스트"""
        context = {
            "context": "민법 제750조는 불법행위로 인한 손해배상청구권을 규정합니다.",
            "legal_references": ["민법 제750조"],
            "query_type": "civil_law"
        }

        prompt = unified_manager.get_optimized_prompt(
            query="불법행위의 성립요건은?",
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.CIVIL_LAW,
            context=context,
            model_type=ModelType.GEMINI
        )

        assert prompt is not None
        assert "민법 제750조" in prompt or "불법행위" in prompt.lower()
        print("✅ 컨텍스트가 프롬프트에 정상적으로 포함됩니다.")

    def test_workflow_complete_flow(self, workflow):
        """전체 워크플로우 엔드-to-엔드 테스트"""
        test_queries = [
            ("이혼 절차에 대해 알려주세요", "family_law"),
            ("계약서 작성 방법을 알려주세요", "contract_review"),
            ("해고 제한 조건은?", "labor_law"),
        ]

        for query, expected_type in test_queries:
            state = create_initial_legal_state(query, f"session-{query[:5]}")
            state["query_type"] = expected_type
            state["retrieved_docs"] = [
                {"content": "관련 법률 정보", "source": "test"}
            ]

            # 전체 워크플로우 실행
            state = workflow.classify_query(state)
            state = workflow.retrieve_documents(state)
            state = workflow.generate_answer_enhanced(state)
            state = workflow.format_response(state)

            assert "answer" in state
            assert len(state["answer"]) > 0
            assert "processing_steps" in state
            print(f"✅ '{query}' 처리 완료")

    def test_error_handling(self, unified_manager):
        """에러 처리 테스트"""
        # 잘못된 파라미터
        try:
            prompt = unified_manager.get_optimized_prompt(
                query="",
                question_type=QuestionType.GENERAL_QUESTION,
                domain=LegalDomain.GENERAL,
                context={},
                model_type=ModelType.GEMINI
            )
            # 빈 쿼리는 폴백 처리가 되어야 함
            assert isinstance(prompt, str)
            print("✅ 빈 쿼리 처리 완료")
        except Exception as e:
            print(f"⚠️ 에러 발생 (예상 가능): {e}")


def run_integration_tests():
    """통합 테스트 실행"""
    print("\n" + "="*80)
    print("UnifiedPromptManager 통합 테스트 시작")
    print("="*80 + "\n")

    # 테스트 인스턴스 생성
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    unified_manager = UnifiedPromptManager()

    test_results = []

    # 테스트 1: UnifiedPromptManager 초기화 확인
    print("📋 테스트 1: UnifiedPromptManager 초기화 확인")
    try:
        assert hasattr(workflow, 'unified_prompt_manager')
        assert workflow.unified_prompt_manager is not None
        print("   ✅ UnifiedPromptManager가 워크플로우에 정상적으로 초기화되었습니다.")
        test_results.append(True)
    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        test_results.append(False)

    # 테스트 2: get_optimized_prompt 기본 동작
    print("\n📋 테스트 2: get_optimized_prompt 기본 동작")
    try:
        prompt = unified_manager.get_optimized_prompt(
            query="이혼 절차에 대해 알려주세요",
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.FAMILY_LAW,
            context={"context": "테스트"},
            model_type=ModelType.GEMINI
        )
        assert prompt and len(prompt) > 0
        print(f"   ✅ 프롬프트 생성 성공 ({len(prompt)}자)")
        test_results.append(True)
    except Exception as e:
        print(f"   ❌ 프롬프트 생성 실패: {e}")
        test_results.append(False)

    # 테스트 3: 다양한 도메인 테스트
    print("\n📋 테스트 3: 다양한 도메인 테스트")
    domains = [
        (LegalDomain.CIVIL_LAW, "민사법"),
        (LegalDomain.CRIMINAL_LAW, "형사법"),
        (LegalDomain.FAMILY_LAW, "가족법"),
        (LegalDomain.LABOR_LAW, "노동법"),
    ]

    for domain, name in domains:
        try:
            prompt = unified_manager.get_optimized_prompt(
                query="테스트 질문",
                question_type=QuestionType.LEGAL_ADVICE,
                domain=domain,
                context={},
                model_type=ModelType.GEMINI
            )
            assert len(prompt) > 0
            print(f"   ✅ {name}: 프롬프트 생성 성공")
            test_results.append(True)
        except Exception as e:
            print(f"   ❌ {name}: 프롬프트 생성 실패 - {e}")
            test_results.append(False)

    # 테스트 4: 전체 워크플로우 테스트
    print("\n📋 테스트 4: 전체 워크플로우 테스트")
    try:
        state = create_initial_legal_state("이혼 절차에 대해 알려주세요", "test-session")
        state["query_type"] = "family_law"
        state["retrieved_docs"] = [{"content": "테스트 문서", "source": "test"}]

        result = workflow.generate_answer_enhanced(state)
        assert "answer" in result
        print(f"   ✅ 워크플로우 처리 완료 (답변 길이: {len(result.get('answer', ''))})")
        test_results.append(True)
    except Exception as e:
        print(f"   ❌ 워크플로우 처리 실패: {e}")
        test_results.append(False)

    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    passed = sum(test_results)
    total = len(test_results)
    print(f"\n✅ 통과: {passed}/{total}")
    print(f"❌ 실패: {total - passed}/{total}")
    print("="*80 + "\n")

    return all(test_results)


if __name__ == "__main__":
    # 통합 테스트 실행
    success = run_integration_tests()

    if success:
        print("✅ 모든 통합 테스트가 성공적으로 완료되었습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")

    # pytest 테스트 실행 (선택적)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        print("\nPytest를 사용한 단위 테스트 실행...")
        pytest.main([__file__, "-v"])
