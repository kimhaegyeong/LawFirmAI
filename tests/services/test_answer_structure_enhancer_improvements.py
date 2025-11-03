# -*- coding: utf-8 -*-
"""
AnswerStructureEnhancer ê°œì„  ?¬í•­ ?ŒìŠ¤??
"""

import os
import sys
import unittest
from typing import Any, Dict, List

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ ê²½ë¡œ ì¶”ê?
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..', '..', 'source')
sys.path.insert(0, source_dir)

from services.answer_structure_enhancer import AnswerStructureEnhancer, QuestionType


class TestAnswerStructureEnhancerImprovements(unittest.TestCase):
    """AnswerStructureEnhancer ê°œì„  ?¬í•­ ?ŒìŠ¤??""

    def setUp(self):
        """?ŒìŠ¤???¤ì •"""
        self.enhancer = AnswerStructureEnhancer(
            llm=None,  # LLM ?†ì´ ?ŒìŠ¤??
            max_few_shot_examples=2,
            enable_few_shot=True,
            enable_cot=True
        )

    def test_1_few_shot_examples_loading(self):
        """1. Few-Shot ?ˆì‹œ ë¡œë“œ ?ŒìŠ¤??""
        self.assertIsNotNone(self.enhancer.few_shot_examples)
        self.assertIsInstance(self.enhancer.few_shot_examples, dict)

        # ìºì‹± ?•ì¸
        examples1 = self.enhancer._load_few_shot_examples()
        examples2 = self.enhancer._load_few_shot_examples()
        self.assertIs(examples1, examples2, "ìºì‹±???‘ë™?´ì•¼ ?©ë‹ˆ??)

    def test_2_few_shot_examples_validation(self):
        """2. Few-Shot ?ˆì‹œ ê²€ì¦??ŒìŠ¤??""
        # ? íš¨???ˆì‹œ
        valid_example = {
            "question": "?ŒìŠ¤??ì§ˆë¬¸",
            "original_answer": "?ë³¸ ?µë?",
            "enhanced_answer": "ê°œì„ ???µë?",
            "improvements": ["ê°œì„ 1", "ê°œì„ 2"]
        }
        self.assertTrue(self.enhancer._validate_few_shot_example(valid_example))

        # ? íš¨?˜ì? ?Šì? ?ˆì‹œ (?„ìˆ˜ ???„ë½)
        invalid_example = {
            "question": "?ŒìŠ¤??ì§ˆë¬¸"
            # ?¤ë¥¸ ?„ìˆ˜ ???„ë½
        }
        self.assertFalse(self.enhancer._validate_few_shot_example(invalid_example))

    def test_3_few_shot_examples_selection(self):
        """3. Few-Shot ?ˆì‹œ ? íƒ ë¡œì§ ?ŒìŠ¤??""
        examples = self.enhancer._get_few_shot_examples(QuestionType.LAW_INQUIRY, "ë¯¼ë²• ??11ì¡?)
        self.assertIsInstance(examples, list)
        self.assertLessEqual(len(examples), 2)  # max_few_shot_examples ?œí•œ

    def test_4_example_similarity_sorting(self):
        """4. ?ˆì‹œ ? ì‚¬???•ë ¬ ?ŒìŠ¤??""
        examples = [
            {"question": "ë¯¼ë²• ??11ì¡°ì— ?€???¤ëª…?´ì£¼?¸ìš”"},
            {"question": "ê³„ì•½ ?´ì? ?ë?ë¥?ì°¾ì•„ì£¼ì„¸??},
            {"question": "?„ë‹¬ì£¼ì˜ê°€ ë¬´ì—‡?¸ê???"}
        ]

        question = "ë¯¼ë²• ??11ì¡°ì— ?€???Œë ¤ì£¼ì„¸??
        sorted_examples = self.enhancer._sort_examples_by_similarity(examples, question)

        # ì²?ë²ˆì§¸ ?ˆì‹œê°€ ê°€??? ì‚¬?´ì•¼ ??
        self.assertIn("111ì¡?, sorted_examples[0]["question"])

    def test_5_cot_response_parsing(self):
        """5. Chain-of-Thought ?‘ë‹µ ?Œì‹± ?ŒìŠ¤??""
        response = """### Step 1: ?ë³¸ ?µë? ë¶„ì„
ë²•ì¡°ë¬?ë²ˆí˜¸: ë¯¼ë²• ??11ì¡?
...
### Step 2: ê°œì„  ?„ëµ ?˜ë¦½
ë³´ì¡´???´ìš©: ...
### Step 3: ê°œì„  ?¤í–‰
## ìµœì¢… ?µë?
ê°œì„ ???µë? ?´ìš©"""

        result = self.enhancer._parse_cot_response(response)

        self.assertTrue(result["has_step1"])
        self.assertTrue(result["has_step2"])
        self.assertTrue(result["has_step3"])
        self.assertGreater(len(result["final_answer"]), 0)

    def test_6_cot_format_validation(self):
        """6. Chain-of-Thought ?•ì‹ ê²€ì¦??ŒìŠ¤??""
        # ? íš¨???‘ë‹µ
        valid_response = """### Step 1: ?ë³¸ ?µë? ë¶„ì„
?ì„¸??ë¶„ì„ ?´ìš©...
### Step 2: ê°œì„  ?„ëµ ?˜ë¦½
?„ëµ ?´ìš©...
### Step 3: ê°œì„  ?¤í–‰
ìµœì¢… ?µë? ?´ìš©"""

        validation = self.enhancer._validate_cot_format(valid_response)
        self.assertTrue(validation["is_valid"])
        self.assertGreaterEqual(validation["validation_score"], 0.7)

        # ? íš¨?˜ì? ?Šì? ?‘ë‹µ (Step ?„ë½)
        invalid_response = "ìµœì¢… ?µë?ë§??ˆìŒ"
        validation = self.enhancer._validate_cot_format(invalid_response)
        self.assertFalse(validation["is_valid"])
        self.assertGreater(len(validation["missing_steps"]), 0)

    def test_7_example_quality_metrics(self):
        """7. ?ˆì‹œ ?ˆì§ˆ ë©”íŠ¸ë¦??ŒìŠ¤??""
        example = {
            "question": "?ŒìŠ¤??ì§ˆë¬¸",
            "original_answer": "?ë³¸ ?µë? " * 10,  # ê¸¸ì´ ?•ë³´
            "enhanced_answer": "ê°œì„ ???µë? " * 15,
            "improvements": ["ê°œì„ 1", "ê°œì„ 2", "ê°œì„ 3", "ê°œì„ 4"]
        }

        metrics = self.enhancer._calculate_example_quality(example)

        self.assertIn("completeness_score", metrics)
        self.assertIn("improvement_count", metrics)
        self.assertIn("length_ratio", metrics)
        self.assertIn("effectiveness_score", metrics)
        self.assertEqual(metrics["improvement_count"], 4)

    def test_8_configuration_options(self):
        """8. ?¤ì • ?µì…˜ ?ŒìŠ¤??""
        # Few-Shot ë¹„í™œ?±í™”
        enhancer_no_fewshot = AnswerStructureEnhancer(enable_few_shot=False)
        self.assertEqual(len(enhancer_no_fewshot.few_shot_examples), 0)

        # Chain-of-Thought ë¹„í™œ?±í™”
        enhancer_no_cot = AnswerStructureEnhancer(enable_cot=False)
        self.assertFalse(enhancer_no_cot.enable_cot)

        # ìµœë? ?ˆì‹œ ê°œìˆ˜ ë³€ê²?
        enhancer_custom = AnswerStructureEnhancer(max_few_shot_examples=5)
        self.assertEqual(enhancer_custom.max_few_shot_examples, 5)

    def test_9_error_recovery(self):
        """9. ?ëŸ¬ ë³µêµ¬ ë¡œì§ ?ŒìŠ¤??""
        # ê¸°ë³¸ ?ˆì‹œ ?œê³µ ?•ì¸
        default_examples = self.enhancer._get_default_examples()
        self.assertIsInstance(default_examples, dict)
        self.assertIn("general_question", default_examples)

    def test_10_prompt_length_optimization(self):
        """10. ?„ë¡¬?„íŠ¸ ê¸¸ì´ ìµœì ???ŒìŠ¤??""
        prompt = self.enhancer._build_llm_enhancement_prompt(
            answer="?ˆë…•?˜ì„¸?? ë¯¼ë²• ??11ì¡°ëŠ”..." * 50,  # ê¸??µë?
            question="ë¯¼ë²• ??11ì¡°ì— ?€???¤ëª…?´ì£¼?¸ìš”",
            question_type=QuestionType.LAW_INQUIRY,
            retrieved_docs=[],
            legal_references=[],
            legal_citations=[]
        )

        # ?„ë¡¬?„íŠ¸ê°€ ?ì„±?˜ì—ˆ?”ì? ?•ì¸
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

        # Few-Shot ?ˆì‹œê°€ ?¬í•¨?˜ì—ˆ?”ì? ?•ì¸
        if self.enhancer.enable_few_shot:
            self.assertIn("Few-Shot Learning", prompt)


if __name__ == '__main__':
    unittest.main()
