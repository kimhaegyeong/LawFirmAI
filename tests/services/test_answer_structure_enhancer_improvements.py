# -*- coding: utf-8 -*-
"""
AnswerStructureEnhancer 개선 사항 테스트
"""

import os
import sys
import unittest
from typing import Any, Dict, List

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..', '..', 'source')
sys.path.insert(0, source_dir)

from services.answer_structure_enhancer import AnswerStructureEnhancer, QuestionType


class TestAnswerStructureEnhancerImprovements(unittest.TestCase):
    """AnswerStructureEnhancer 개선 사항 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.enhancer = AnswerStructureEnhancer(
            llm=None,  # LLM 없이 테스트
            max_few_shot_examples=2,
            enable_few_shot=True,
            enable_cot=True
        )

    def test_1_few_shot_examples_loading(self):
        """1. Few-Shot 예시 로드 테스트"""
        self.assertIsNotNone(self.enhancer.few_shot_examples)
        self.assertIsInstance(self.enhancer.few_shot_examples, dict)

        # 캐싱 확인
        examples1 = self.enhancer._load_few_shot_examples()
        examples2 = self.enhancer._load_few_shot_examples()
        self.assertIs(examples1, examples2, "캐싱이 작동해야 합니다")

    def test_2_few_shot_examples_validation(self):
        """2. Few-Shot 예시 검증 테스트"""
        # 유효한 예시
        valid_example = {
            "question": "테스트 질문",
            "original_answer": "원본 답변",
            "enhanced_answer": "개선된 답변",
            "improvements": ["개선1", "개선2"]
        }
        self.assertTrue(self.enhancer._validate_few_shot_example(valid_example))

        # 유효하지 않은 예시 (필수 키 누락)
        invalid_example = {
            "question": "테스트 질문"
            # 다른 필수 키 누락
        }
        self.assertFalse(self.enhancer._validate_few_shot_example(invalid_example))

    def test_3_few_shot_examples_selection(self):
        """3. Few-Shot 예시 선택 로직 테스트"""
        examples = self.enhancer._get_few_shot_examples(QuestionType.LAW_INQUIRY, "민법 제111조")
        self.assertIsInstance(examples, list)
        self.assertLessEqual(len(examples), 2)  # max_few_shot_examples 제한

    def test_4_example_similarity_sorting(self):
        """4. 예시 유사도 정렬 테스트"""
        examples = [
            {"question": "민법 제111조에 대해 설명해주세요"},
            {"question": "계약 해지 판례를 찾아주세요"},
            {"question": "도달주의가 무엇인가요?"}
        ]

        question = "민법 제111조에 대해 알려주세요"
        sorted_examples = self.enhancer._sort_examples_by_similarity(examples, question)

        # 첫 번째 예시가 가장 유사해야 함
        self.assertIn("111조", sorted_examples[0]["question"])

    def test_5_cot_response_parsing(self):
        """5. Chain-of-Thought 응답 파싱 테스트"""
        response = """### Step 1: 원본 답변 분석
법조문 번호: 민법 제111조
...
### Step 2: 개선 전략 수립
보존할 내용: ...
### Step 3: 개선 실행
## 최종 답변
개선된 답변 내용"""

        result = self.enhancer._parse_cot_response(response)

        self.assertTrue(result["has_step1"])
        self.assertTrue(result["has_step2"])
        self.assertTrue(result["has_step3"])
        self.assertGreater(len(result["final_answer"]), 0)

    def test_6_cot_format_validation(self):
        """6. Chain-of-Thought 형식 검증 테스트"""
        # 유효한 응답
        valid_response = """### Step 1: 원본 답변 분석
상세한 분석 내용...
### Step 2: 개선 전략 수립
전략 내용...
### Step 3: 개선 실행
최종 답변 내용"""

        validation = self.enhancer._validate_cot_format(valid_response)
        self.assertTrue(validation["is_valid"])
        self.assertGreaterEqual(validation["validation_score"], 0.7)

        # 유효하지 않은 응답 (Step 누락)
        invalid_response = "최종 답변만 있음"
        validation = self.enhancer._validate_cot_format(invalid_response)
        self.assertFalse(validation["is_valid"])
        self.assertGreater(len(validation["missing_steps"]), 0)

    def test_7_example_quality_metrics(self):
        """7. 예시 품질 메트릭 테스트"""
        example = {
            "question": "테스트 질문",
            "original_answer": "원본 답변 " * 10,  # 길이 확보
            "enhanced_answer": "개선된 답변 " * 15,
            "improvements": ["개선1", "개선2", "개선3", "개선4"]
        }

        metrics = self.enhancer._calculate_example_quality(example)

        self.assertIn("completeness_score", metrics)
        self.assertIn("improvement_count", metrics)
        self.assertIn("length_ratio", metrics)
        self.assertIn("effectiveness_score", metrics)
        self.assertEqual(metrics["improvement_count"], 4)

    def test_8_configuration_options(self):
        """8. 설정 옵션 테스트"""
        # Few-Shot 비활성화
        enhancer_no_fewshot = AnswerStructureEnhancer(enable_few_shot=False)
        self.assertEqual(len(enhancer_no_fewshot.few_shot_examples), 0)

        # Chain-of-Thought 비활성화
        enhancer_no_cot = AnswerStructureEnhancer(enable_cot=False)
        self.assertFalse(enhancer_no_cot.enable_cot)

        # 최대 예시 개수 변경
        enhancer_custom = AnswerStructureEnhancer(max_few_shot_examples=5)
        self.assertEqual(enhancer_custom.max_few_shot_examples, 5)

    def test_9_error_recovery(self):
        """9. 에러 복구 로직 테스트"""
        # 기본 예시 제공 확인
        default_examples = self.enhancer._get_default_examples()
        self.assertIsInstance(default_examples, dict)
        self.assertIn("general_question", default_examples)

    def test_10_prompt_length_optimization(self):
        """10. 프롬프트 길이 최적화 테스트"""
        prompt = self.enhancer._build_llm_enhancement_prompt(
            answer="안녕하세요! 민법 제111조는..." * 50,  # 긴 답변
            question="민법 제111조에 대해 설명해주세요",
            question_type=QuestionType.LAW_INQUIRY,
            retrieved_docs=[],
            legal_references=[],
            legal_citations=[]
        )

        # 프롬프트가 생성되었는지 확인
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

        # Few-Shot 예시가 포함되었는지 확인
        if self.enhancer.enable_few_shot:
            self.assertIn("Few-Shot Learning", prompt)


if __name__ == '__main__':
    unittest.main()
