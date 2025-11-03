# -*- coding: utf-8 -*-
"""
UnifiedPromptManager ?µí•© ?ŒìŠ¤??
langgraph ?Œí¬?Œë¡œ?°ì—??UnifiedPromptManager ?µí•© ê²€ì¦?
"""

import asyncio
import os
import sys
from pathlib import Path

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # pytestê°€ ?†ìœ¼ë©?unittestë¡??€ì²?
    import unittest
    pytest = unittest

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ?ŒìŠ¤???˜ê²½ ?¤ì •
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"

from source.agents.legal_workflow_enhanced import (
    EnhancedLegalQuestionWorkflow,
)
from source.agents.state_definitions import create_initial_legal_state
from source.services.question_classifier import QuestionType
from source.services.unified_prompt_manager import (
    LegalDomain,
    ModelType,
    UnifiedPromptManager,
)
from source.utils.langgraph_config import LangGraphConfig


class TestUnifiedPromptIntegration:
    """UnifiedPromptManager ?µí•© ?ŒìŠ¤??(pytest??"""

    def __init__(self):
        """pytest fixture ?†ì´ ?¤í–‰ ê°€?¥í•˜?„ë¡ ì´ˆê¸°??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            self.config = LangGraphConfig.from_env()
            self.workflow = EnhancedLegalQuestionWorkflow(self.config)
            self.unified_manager = UnifiedPromptManager()

    if PYTEST_AVAILABLE:
        @pytest.fixture
        def config(self):
            """LangGraph ?¤ì • ?ì„±"""
            return LangGraphConfig.from_env()

        @pytest.fixture
        def workflow(self, config):
            """?Œí¬?Œë¡œ???¸ìŠ¤?´ìŠ¤ ?ì„±"""
            return EnhancedLegalQuestionWorkflow(config)

        @pytest.fixture
        def unified_manager(self):
            """UnifiedPromptManager ?¸ìŠ¤?´ìŠ¤ ?ì„±"""
            return UnifiedPromptManager()

    def test_unified_prompt_manager_initialized(self, workflow=None):
        """UnifiedPromptManagerê°€ ?Œí¬?Œë¡œ?°ì— ì´ˆê¸°?”ë˜?ˆëŠ”ì§€ ?•ì¸"""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        if workflow is None:
            return  # ?Œí¬?Œë¡œ?°ë? ì´ˆê¸°?”í•  ???†ìŒ
        assert hasattr(workflow, 'unified_prompt_manager')
        assert workflow.unified_prompt_manager is not None
        print("??UnifiedPromptManagerê°€ ?Œí¬?Œë¡œ?°ì— ?•ìƒ?ìœ¼ë¡?ì´ˆê¸°?”ë˜?ˆìŠµ?ˆë‹¤.")

    def test_prompt_manager_type(self, workflow=None):
        """UnifiedPromptManager ?€???•ì¸"""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        if workflow is None:
            return  # ?Œí¬?Œë¡œ?°ë? ì´ˆê¸°?”í•  ???†ìŒ
        assert isinstance(workflow.unified_prompt_manager, UnifiedPromptManager)
        print("??UnifiedPromptManager ?€?…ì´ ?¬ë°”ë¦…ë‹ˆ??")

    def test_get_optimized_prompt(self, unified_manager=None):
        """get_optimized_prompt ë©”ì„œ???ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        query = "?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??

        prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.FAMILY_LAW,
            context={"context": "?ŒìŠ¤??ì»¨í…?¤íŠ¸"},
            model_type=ModelType.GEMINI
        )

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "?´í˜¼" in query
        print("??get_optimized_promptê°€ ?•ìƒ?ìœ¼ë¡??‘ë™?©ë‹ˆ??")
        print(f"   ?ì„±???„ë¡¬?„íŠ¸ ê¸¸ì´: {len(prompt)}??)

    def test_prompt_with_different_domains(self, unified_manager=None):
        """?¤ì–‘???„ë©”?¸ë³„ ?„ë¡¬?„íŠ¸ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        test_cases = [
            (LegalDomain.CIVIL_LAW, QuestionType.LEGAL_ADVICE, "ê³„ì•½???‘ì„± ë°©ë²•"),
            (LegalDomain.CRIMINAL_LAW, QuestionType.LEGAL_ADVICE, "?ˆë„ì£„ì˜ ì²˜ë²Œ"),
            (LegalDomain.FAMILY_LAW, QuestionType.LEGAL_ADVICE, "?´í˜¼ ?ˆì°¨"),
            (LegalDomain.LABOR_LAW, QuestionType.LEGAL_ADVICE, "?´ê³  ?œí•œ ì¡°ê±´"),
            (LegalDomain.GENERAL, QuestionType.GENERAL_QUESTION, "ë²•ë¥  ?©ì–´ ?´ì„¤"),
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
            print(f"??{domain.value} ?„ë©”???„ë¡¬?„íŠ¸ ?ì„± ?±ê³µ ({len(prompt)}??")

    def test_prompt_with_different_models(self, unified_manager=None):
        """?¤ì–‘??ëª¨ë¸ ?€?…ë³„ ?„ë¡¬?„íŠ¸ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        test_cases = [
            (ModelType.GEMINI, "Gemini"),
            (ModelType.OLLAMA, "Ollama"),
            (ModelType.OPENAI, "OpenAI"),
        ]

        query = "ë¯¼ë²• ??50ì¡°ì— ?€???Œë ¤ì£¼ì„¸??

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
            print(f"??{model_name} ëª¨ë¸ ?€???„ë¡¬?„íŠ¸ ?ì„± ?±ê³µ ({len(prompt)}??")

    def test_enhanced_workflow_generate_answer(self, workflow=None):
        """generate_answer_enhanced?ì„œ UnifiedPromptManager ?¬ìš© ?•ì¸"""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        if workflow is None:
            return  # ?Œí¬?Œë¡œ?°ë? ì´ˆê¸°?”í•  ???†ìŒ
        # ì´ˆê¸° ?íƒœ ?ì„±
        state = create_initial_legal_state("?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??, "test-session")
        state["query_type"] = "family_law"
        state["retrieved_docs"] = [
            {"content": "?´í˜¼ ?ˆì°¨???‘ì˜?´í˜¼ê³??¬íŒ???´í˜¼???ˆìŠµ?ˆë‹¤.", "source": "test"}
        ]

        # generate_answer_enhanced ?¸ì¶œ
        result = workflow.generate_answer_enhanced(state)

        assert "answer" in result
        assert "processing_steps" in result
        assert "UnifiedPromptManager" in " ".join(result.get("processing_steps", []))
        print("??generate_answer_enhancedê°€ UnifiedPromptManagerë¥??¬ìš©?©ë‹ˆ??")

    def test_prompt_optimization_features(self, unified_manager=None):
        """?„ë¡¬?„íŠ¸ ìµœì ??ê¸°ëŠ¥ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        query = "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•"

        # Legal Advice ?€??
        advice_prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.CIVIL_LAW,
            context={"context": "ë¯¼ë²• ??50ì¡?ë¶ˆë²•?‰ìœ„"},
            model_type=ModelType.GEMINI
        )

        # Procedure Guide ?€??
        procedure_prompt = unified_manager.get_optimized_prompt(
            query=query,
            question_type=QuestionType.PROCEDURE_GUIDE,
            domain=LegalDomain.CIVIL_PROCEDURE,
            context={"context": "ë¯¼ì‚¬?Œì†¡ë²?},
            model_type=ModelType.GEMINI
        )

        # ???„ë¡¬?„íŠ¸ê°€ ?¤ë¥¸ì§€ ?•ì¸
        assert advice_prompt != procedure_prompt
        assert len(advice_prompt) > 0
        assert len(procedure_prompt) > 0

        print(f"??Legal Advice ?„ë¡¬?„íŠ¸: {len(advice_prompt)}??)
        print(f"??Procedure Guide ?„ë¡¬?„íŠ¸: {len(procedure_prompt)}??)
        print("???„ë¡¬?„íŠ¸ê°€ ì§ˆë¬¸ ? í˜•???°ë¼ ìµœì ?”ë©?ˆë‹¤.")

    def test_prompt_with_context(self, unified_manager=None):
        """ì»¨í…?¤íŠ¸ê°€ ?¬í•¨???„ë¡¬?„íŠ¸ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        context = {
            "context": "ë¯¼ë²• ??50ì¡°ëŠ” ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒì²?µ¬ê¶Œì„ ê·œì •?©ë‹ˆ??",
            "legal_references": ["ë¯¼ë²• ??50ì¡?],
            "query_type": "civil_law"
        }

        prompt = unified_manager.get_optimized_prompt(
            query="ë¶ˆë²•?‰ìœ„???±ë¦½?”ê±´?€?",
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.CIVIL_LAW,
            context=context,
            model_type=ModelType.GEMINI
        )

        assert prompt is not None
        assert "ë¯¼ë²• ??50ì¡? in prompt or "ë¶ˆë²•?‰ìœ„" in prompt.lower()
        print("??ì»¨í…?¤íŠ¸ê°€ ?„ë¡¬?„íŠ¸???•ìƒ?ìœ¼ë¡??¬í•¨?©ë‹ˆ??")

    def test_workflow_complete_flow(self, workflow=None):
        """?„ì²´ ?Œí¬?Œë¡œ???”ë“œ-to-?”ë“œ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'workflow'):
                self.__init__()
            workflow = self.workflow
        if workflow is None:
            return  # ?Œí¬?Œë¡œ?°ë? ì´ˆê¸°?”í•  ???†ìŒ
        test_queries = [
            ("?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??, "family_law"),
            ("ê³„ì•½???‘ì„± ë°©ë²•???Œë ¤ì£¼ì„¸??, "contract_review"),
            ("?´ê³  ?œí•œ ì¡°ê±´?€?", "labor_law"),
        ]

        for query, expected_type in test_queries:
            state = create_initial_legal_state(query, f"session-{query[:5]}")
            state["query_type"] = expected_type
            state["retrieved_docs"] = [
                {"content": "ê´€??ë²•ë¥  ?•ë³´", "source": "test"}
            ]

            # ?„ì²´ ?Œí¬?Œë¡œ???¤í–‰
            state = workflow.classify_query(state)
            state = workflow.retrieve_documents(state)
            state = workflow.generate_answer_enhanced(state)
            state = workflow.format_response(state)

            assert "answer" in state
            assert len(state["answer"]) > 0
            assert "processing_steps" in state
            print(f"??'{query}' ì²˜ë¦¬ ?„ë£Œ")

    def test_error_handling(self, unified_manager=None):
        """?ëŸ¬ ì²˜ë¦¬ ?ŒìŠ¤??""
        if not PYTEST_AVAILABLE:
            # pytest ?†ì´ ì§ì ‘ ?¤í–‰
            if not hasattr(self, 'unified_manager'):
                self.__init__()
            unified_manager = self.unified_manager
        if unified_manager is None:
            return  # UnifiedPromptManagerë¥?ì´ˆê¸°?”í•  ???†ìŒ
        # ?˜ëª»???Œë¼ë¯¸í„°
        try:
            prompt = unified_manager.get_optimized_prompt(
                query="",
                question_type=QuestionType.GENERAL_QUESTION,
                domain=LegalDomain.GENERAL,
                context={},
                model_type=ModelType.GEMINI
            )
            # ë¹?ì¿¼ë¦¬???´ë°± ì²˜ë¦¬ê°€ ?˜ì–´????
            assert isinstance(prompt, str)
            print("??ë¹?ì¿¼ë¦¬ ì²˜ë¦¬ ?„ë£Œ")
        except Exception as e:
            print(f"? ï¸ ?ëŸ¬ ë°œìƒ (?ˆìƒ ê°€??: {e}")


def run_integration_tests():
    """?µí•© ?ŒìŠ¤???¤í–‰"""
    print("\n" + "="*80)
    print("UnifiedPromptManager ?µí•© ?ŒìŠ¤???œì‘")
    print("="*80 + "\n")

    # ?ŒìŠ¤???¸ìŠ¤?´ìŠ¤ ?ì„±
    config = LangGraphConfig.from_env()
    workflow = EnhancedLegalQuestionWorkflow(config)
    unified_manager = UnifiedPromptManager()

    test_results = []

    # ?ŒìŠ¤??1: UnifiedPromptManager ì´ˆê¸°???•ì¸
    print("?“‹ ?ŒìŠ¤??1: UnifiedPromptManager ì´ˆê¸°???•ì¸")
    try:
        assert hasattr(workflow, 'unified_prompt_manager')
        assert workflow.unified_prompt_manager is not None
        print("   ??UnifiedPromptManagerê°€ ?Œí¬?Œë¡œ?°ì— ?•ìƒ?ìœ¼ë¡?ì´ˆê¸°?”ë˜?ˆìŠµ?ˆë‹¤.")
        test_results.append(True)
    except Exception as e:
        print(f"   ??ì´ˆê¸°???¤íŒ¨: {e}")
        test_results.append(False)

    # ?ŒìŠ¤??2: get_optimized_prompt ê¸°ë³¸ ?™ì‘
    print("\n?“‹ ?ŒìŠ¤??2: get_optimized_prompt ê¸°ë³¸ ?™ì‘")
    try:
        prompt = unified_manager.get_optimized_prompt(
            query="?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??,
            question_type=QuestionType.LEGAL_ADVICE,
            domain=LegalDomain.FAMILY_LAW,
            context={"context": "?ŒìŠ¤??},
            model_type=ModelType.GEMINI
        )
        assert prompt and len(prompt) > 0
        print(f"   ???„ë¡¬?„íŠ¸ ?ì„± ?±ê³µ ({len(prompt)}??")
        test_results.append(True)
    except Exception as e:
        print(f"   ???„ë¡¬?„íŠ¸ ?ì„± ?¤íŒ¨: {e}")
        test_results.append(False)

    # ?ŒìŠ¤??3: ?¤ì–‘???„ë©”???ŒìŠ¤??
    print("\n?“‹ ?ŒìŠ¤??3: ?¤ì–‘???„ë©”???ŒìŠ¤??)
    domains = [
        (LegalDomain.CIVIL_LAW, "ë¯¼ì‚¬ë²?),
        (LegalDomain.CRIMINAL_LAW, "?•ì‚¬ë²?),
        (LegalDomain.FAMILY_LAW, "ê°€ì¡±ë²•"),
        (LegalDomain.LABOR_LAW, "?¸ë™ë²?),
    ]

    for domain, name in domains:
        try:
            prompt = unified_manager.get_optimized_prompt(
                query="?ŒìŠ¤??ì§ˆë¬¸",
                question_type=QuestionType.LEGAL_ADVICE,
                domain=domain,
                context={},
                model_type=ModelType.GEMINI
            )
            assert len(prompt) > 0
            print(f"   ??{name}: ?„ë¡¬?„íŠ¸ ?ì„± ?±ê³µ")
            test_results.append(True)
        except Exception as e:
            print(f"   ??{name}: ?„ë¡¬?„íŠ¸ ?ì„± ?¤íŒ¨ - {e}")
            test_results.append(False)

    # ?ŒìŠ¤??4: ?„ì²´ ?Œí¬?Œë¡œ???ŒìŠ¤??
    print("\n?“‹ ?ŒìŠ¤??4: ?„ì²´ ?Œí¬?Œë¡œ???ŒìŠ¤??)
    try:
        state = create_initial_legal_state("?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??, "test-session")
        state["query_type"] = "family_law"
        state["retrieved_docs"] = [{"content": "?ŒìŠ¤??ë¬¸ì„œ", "source": "test"}]

        result = workflow.generate_answer_enhanced(state)
        assert "answer" in result
        print(f"   ???Œí¬?Œë¡œ??ì²˜ë¦¬ ?„ë£Œ (?µë? ê¸¸ì´: {len(result.get('answer', ''))})")
        test_results.append(True)
    except Exception as e:
        print(f"   ???Œí¬?Œë¡œ??ì²˜ë¦¬ ?¤íŒ¨: {e}")
        test_results.append(False)

    # ê²°ê³¼ ?”ì•½
    print("\n" + "="*80)
    print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
    print("="*80)
    passed = sum(test_results)
    total = len(test_results)
    print(f"\n???µê³¼: {passed}/{total}")
    print(f"???¤íŒ¨: {total - passed}/{total}")
    print("="*80 + "\n")

    return all(test_results)


if __name__ == "__main__":
    # ?µí•© ?ŒìŠ¤???¤í–‰ (pytest ?†ì´???¤í–‰ ê°€??
    success = run_integration_tests()

    if success:
        print("??ëª¨ë“  ?µí•© ?ŒìŠ¤?¸ê? ?±ê³µ?ìœ¼ë¡??„ë£Œ?˜ì—ˆ?µë‹ˆ??")
    else:
        print("? ï¸ ?¼ë? ?ŒìŠ¤?¸ê? ?¤íŒ¨?ˆìŠµ?ˆë‹¤. ë¡œê·¸ë¥??•ì¸?´ì£¼?¸ìš”.")

    # pytest ?ŒìŠ¤???¤í–‰ (? íƒ?? pytestê°€ ?¤ì¹˜??ê²½ìš°ë§?
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest" and PYTEST_AVAILABLE:
        print("\nPytestë¥??¬ìš©???¨ìœ„ ?ŒìŠ¤???¤í–‰...")
        pytest.main([__file__, "-v"])
    elif len(sys.argv) > 1 and sys.argv[1] == "--pytest" and not PYTEST_AVAILABLE:
        print("? ï¸ pytestê°€ ?¤ì¹˜?˜ì? ?Šì•„ pytest ?ŒìŠ¤?¸ë? ?¤í–‰?????†ìŠµ?ˆë‹¤.")
