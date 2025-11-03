#!/usr/bin/env python3
"""
RAG ?œìŠ¤???µí•© ?ŒìŠ¤??

??ëª¨ë“ˆ?€ LawFirmAI??RAG ?œìŠ¤???„ì²´ ?µí•© ?ŒìŠ¤?¸ë? ?˜í–‰?©ë‹ˆ??
- ChatService ?µí•© ?ŒìŠ¤??(LangGraph ê¸°ë°˜)
- ì§ˆë¬¸ ë¶„ë¥˜ ?œìŠ¤???ŒìŠ¤??(6ê°€ì§€ ì§ˆë¬¸ ? í˜•)
- ?µë? ?ì„± ?ˆì§ˆ ?ŒìŠ¤??(? ë¢°??ê³„ì‚° ë°??µë? ?•ì‹ ê²€ì¦?

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import RAG components
try:
    from source.services.chat_service import ChatService
    from source.services.hybrid_search_engine import HybridSearchEngine
    from source.services.improved_answer_generator import (
        AnswerResult,
        ImprovedAnswerGenerator,
    )
    from source.services.question_classifier import (
        QuestionClassification,
        QuestionClassifier,
        QuestionType,
    )
    # RAGService removed - use HybridSearchEngine instead
    RAGService = None
    from source.utils.config import Config
    from source.utils.logger import get_logger
    RAG_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG modules not available: {e}")
    RAG_MODULES_AVAILABLE = False

# Test configuration
TEST_CONFIG = {
    "test_questions": {
        "precedent_search": [
            "?í•´ë°°ìƒ ê´€???ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
            "?´í˜¼ ?„ìë£??ë?ë¥?ê²€?‰í•´ì£¼ì„¸??,
            "ê³„ì•½ ?´ì œ ê´€???€ë²•ì› ?ë?ê°€ ?ˆë‚˜??"
        ],
        "law_inquiry": [
            "ë¯¼ë²• ??50ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???",
            "?•ë²• ??50ì¡??´ì¸ì£„ì— ?€???¤ëª…?´ì£¼?¸ìš”",
            "?ë²• ??34ì¡??´ì‚¬??ì±…ì„???€???Œë ¤ì£¼ì„¸??
        ],
        "legal_advice": [
            "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­??ì¡°ì–¸?´ì£¼?¸ìš”",
            "?´í˜¼ ?ˆì°¨?€ ?„ìš”???œë¥˜ë¥??Œë ¤ì£¼ì„¸??,
            "?í•´ë°°ìƒ ì²?µ¬ ë°©ë²•???ˆë‚´?´ì£¼?¸ìš”"
        ],
        "procedure_guide": [
            "?Œì†¡ ?œê¸° ?ˆì°¨???´ë–»ê²??˜ë‚˜??",
            "ë¶€?™ì‚° ?±ê¸° ? ì²­ ë°©ë²•???Œë ¤ì£¼ì„¸??,
            "?¹í—ˆ ì¶œì› ?ˆì°¨ë¥??¤ëª…?´ì£¼?¸ìš”"
        ],
        "term_explanation": [
            "ë¶ˆë²•?‰ìœ„???•ì˜ë¥??Œë ¤ì£¼ì„¸??,
            "ì±„ê¶Œê³?ì±„ë¬´??ì°¨ì´?ì? ë¬´ì—‡?¸ê???",
            "?Œë©¸?œíš¨??ê°œë…???¤ëª…?´ì£¼?¸ìš”"
        ],
        "general_question": [
            "ë²•ë¥ ???€??ê¶ê¸ˆ??ê²ƒì´ ?ˆìŠµ?ˆë‹¤",
            "ë²•ì  ë¬¸ì œë¡?ê³ ë????ˆìŠµ?ˆë‹¤",
            "ë²•ë¥  ?ë‹´???„ìš”?©ë‹ˆ??
        ]
    },
    "performance_thresholds": {
        "response_time": 10.0,  # seconds
        "min_confidence": 0.3,
        "min_answer_length": 50,
        "max_answer_length": 5000
    }
}

logger = get_logger(__name__)


@dataclass
class TestResult:
    """?ŒìŠ¤??ê²°ê³¼ ?°ì´???´ë˜??""
    test_name: str
    passed: bool
    response_time: float
    confidence: float
    answer_length: int
    question_type: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class RAGIntegrationTestSuite:
    """RAG ?œìŠ¤???µí•© ?ŒìŠ¤???¤ìœ„??""

    def __init__(self):
        """?ŒìŠ¤???¤ìœ„??ì´ˆê¸°??""
        self.logger = get_logger(__name__)
        self.config = Config()
        self.test_results: List[TestResult] = []

        # RAG ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
        self.chat_service = None
        self.question_classifier = None
        self.answer_generator = None
        self.rag_service = None
        self.hybrid_search_engine = None

        self._initialize_components()

    def _initialize_components(self):
        """RAG ì»´í¬?ŒíŠ¸ ì´ˆê¸°??""
        try:
            if not RAG_MODULES_AVAILABLE:
                raise ImportError("RAG modules not available")

            # ChatService ì´ˆê¸°??
            self.chat_service = ChatService(self.config)
            self.logger.info("ChatService initialized")

            # ê°œë³„ ì»´í¬?ŒíŠ¸ ì´ˆê¸°??
            self.question_classifier = QuestionClassifier()
            self.answer_generator = ImprovedAnswerGenerator()

            # RAG ?œë¹„??ì´ˆê¸°??(Mock ?¬ìš©)
            # RAGService removed - use HybridSearchEngine instead
            self.rag_service = None
            self.hybrid_search_engine = Mock(spec=HybridSearchEngine)

            self.logger.info("All RAG components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG components: {e}")
            raise

    async def test_chat_service_integration(self) -> List[TestResult]:
        """ChatService ?µí•© ?ŒìŠ¤??""
        self.logger.info("Starting ChatService integration tests...")
        results = []

        # ?ŒìŠ¤??ì§ˆë¬¸??
        test_questions = [
            "?ˆë…•?˜ì„¸?? ë²•ë¥  ?ë‹´???„ìš”?©ë‹ˆ??,
            "ê³„ì•½??ê²€? ë? ?„ì?ì£¼ì„¸??,
            "?´í˜¼ ?ˆì°¨???€???Œë ¤ì£¼ì„¸??
        ]

        for question in test_questions:
            try:
                start_time = time.time()

                # ChatServiceë¥??µí•œ ë©”ì‹œì§€ ì²˜ë¦¬
                response = await self.chat_service.process_message(question)

                response_time = time.time() - start_time

                # ê²°ê³¼ ê²€ì¦?
                passed = self._validate_chat_response(response, response_time)

                result = TestResult(
                    test_name=f"chat_service_{question[:20]}",
                    passed=passed,
                    response_time=response_time,
                    confidence=response.get("confidence", 0.0),
                    answer_length=len(response.get("response", "")),
                    question_type="chat_service",
                    metadata={
                        "response_keys": list(response.keys()),
                        "langgraph_enabled": response.get("langgraph_enabled", False)
                    }
                )

                results.append(result)
                self.logger.info(f"ChatService test completed: {question[:30]}... - {'PASS' if passed else 'FAIL'}")

            except Exception as e:
                self.logger.error(f"ChatService test failed for '{question}': {e}")
                results.append(TestResult(
                    test_name=f"chat_service_{question[:20]}",
                    passed=False,
                    response_time=0.0,
                    confidence=0.0,
                    answer_length=0,
                    question_type="chat_service",
                    error_message=str(e)
                ))

        return results

    def test_question_classification_system(self) -> List[TestResult]:
        """ì§ˆë¬¸ ë¶„ë¥˜ ?œìŠ¤???ŒìŠ¤??(6ê°€ì§€ ì§ˆë¬¸ ? í˜•)"""
        self.logger.info("Starting question classification system tests...")
        results = []

        for question_type, questions in TEST_CONFIG["test_questions"].items():
            for question in questions:
                try:
                    start_time = time.time()

                    # ì§ˆë¬¸ ë¶„ë¥˜ ?˜í–‰
                    classification = self.question_classifier.classify_question(question)

                    response_time = time.time() - start_time

                    # ê²°ê³¼ ê²€ì¦?
                    passed = self._validate_classification(classification, question_type)

                    result = TestResult(
                        test_name=f"classification_{question_type}_{question[:20]}",
                        passed=passed,
                        response_time=response_time,
                        confidence=classification.confidence,
                        answer_length=0,
                        question_type=question_type,
                        metadata={
                            "classified_type": classification.question_type.value,
                            "law_weight": classification.law_weight,
                            "precedent_weight": classification.precedent_weight,
                            "keywords": classification.keywords,
                            "patterns": classification.patterns
                        }
                    )

                    results.append(result)
                    self.logger.info(f"Classification test completed: {question_type} - {'PASS' if passed else 'FAIL'}")

                except Exception as e:
                    self.logger.error(f"Classification test failed for '{question}': {e}")
                    results.append(TestResult(
                        test_name=f"classification_{question_type}_{question[:20]}",
                        passed=False,
                        response_time=0.0,
                        confidence=0.0,
                        answer_length=0,
                        question_type=question_type,
                        error_message=str(e)
                    ))

        return results

    def test_answer_generation_quality(self) -> List[TestResult]:
        """?µë? ?ì„± ?ˆì§ˆ ?ŒìŠ¤??""
        self.logger.info("Starting answer generation quality tests...")
        results = []

        # ?ŒìŠ¤?¸ìš© ì§ˆë¬¸ ë¶„ë¥˜ ê²°ê³¼ ?ì„±
        test_classifications = {
            QuestionType.PRECEDENT_SEARCH: QuestionClassification(
                question_type=QuestionType.PRECEDENT_SEARCH,
                law_weight=0.2,
                precedent_weight=0.8,
                confidence=0.8,
                keywords=["?ë?", "ê²€??],
                patterns=[]
            ),
            QuestionType.LAW_INQUIRY: QuestionClassification(
                question_type=QuestionType.LAW_INQUIRY,
                law_weight=0.8,
                precedent_weight=0.2,
                confidence=0.9,
                keywords=["ë²•ë¥ ", "ì¡°ë¬¸"],
                patterns=[]
            ),
            QuestionType.LEGAL_ADVICE: QuestionClassification(
                question_type=QuestionType.LEGAL_ADVICE,
                law_weight=0.5,
                precedent_weight=0.5,
                confidence=0.7,
                keywords=["ì¡°ì–¸", "ë°©ë²•"],
                patterns=[]
            )
        }

        test_questions = [
            "?í•´ë°°ìƒ ê´€???ë?ë¥?ì°¾ì•„ì£¼ì„¸??,
            "ë¯¼ë²• ??50ì¡°ì˜ ?´ìš©??ë¬´ì—‡?¸ê???",
            "ê³„ì•½???‘ì„± ??ì£¼ì˜?¬í•­??ì¡°ì–¸?´ì£¼?¸ìš”"
        ]

        for i, question in enumerate(test_questions):
            try:
                start_time = time.time()

                # Mock ?ŒìŠ¤ ?°ì´??
                mock_sources = {
                    "results": [
                        {"type": "law", "law_name": "ë¯¼ë²•", "article_number": "??50ì¡?, "similarity": 0.9},
                        {"type": "precedent", "case_name": "?í•´ë°°ìƒ ?¬ê±´", "case_number": "2023??2345", "similarity": 0.8}
                    ],
                    "law_results": [
                        {"law_name": "ë¯¼ë²•", "article_number": "??50ì¡?, "content": "ë¶ˆë²•?‰ìœ„ë¡??¸í•œ ?í•´ë°°ìƒ"}
                    ],
                    "precedent_results": [
                        {"case_name": "?í•´ë°°ìƒ ?¬ê±´", "case_number": "2023??2345", "summary": "ë¶ˆë²•?‰ìœ„ ?í•´ë°°ìƒ"}
                    ]
                }

                # ì§ˆë¬¸ ? í˜•???°ë¥¸ ë¶„ë¥˜ ê²°ê³¼ ? íƒ
                question_types = list(test_classifications.keys())
                classification = test_classifications[question_types[i % len(question_types)]]

                # ?µë? ?ì„±
                answer_result = self.answer_generator.generate_answer(
                    query=question,
                    question_type=classification,
                    context="?ŒìŠ¤??ì»¨í…?¤íŠ¸",
                    sources=mock_sources
                )

                response_time = time.time() - start_time

                # ê²°ê³¼ ê²€ì¦?
                passed = self._validate_answer_quality(answer_result, response_time)

                result = TestResult(
                    test_name=f"answer_generation_{question[:20]}",
                    passed=passed,
                    response_time=response_time,
                    confidence=answer_result.confidence.confidence,
                    answer_length=len(answer_result.answer),
                    question_type=answer_result.question_type.value if hasattr(answer_result.question_type, 'value') else str(answer_result.question_type),
                    metadata={
                        "formatted_answer_available": answer_result.formatted_answer is not None,
                        "tokens_used": answer_result.tokens_used,
                        "model_info": answer_result.model_info,
                        "confidence_level": answer_result.confidence.reliability_level.value if hasattr(answer_result.confidence.reliability_level, 'value') else str(answer_result.confidence.reliability_level)
                    }
                )

                results.append(result)
                self.logger.info(f"Answer generation test completed: {question[:30]}... - {'PASS' if passed else 'FAIL'}")

            except Exception as e:
                import traceback
                error_details = f"{str(e)}\n{traceback.format_exc()}"
                self.logger.error(f"Answer generation test failed for '{question}': {error_details}")
                results.append(TestResult(
                    test_name=f"answer_generation_{question[:20]}",
                    passed=False,
                    response_time=0.0,
                    confidence=0.0,
                    answer_length=0,
                    question_type="unknown",
                    error_message=error_details
                ))

        return results

    def _validate_chat_response(self, response: Dict[str, Any], response_time: float) -> bool:
        """ChatService ?‘ë‹µ ê²€ì¦?""
        try:
            # ?„ìˆ˜ ??ì¡´ì¬ ?•ì¸
            required_keys = ["response", "confidence", "sources", "processing_time"]
            if not all(key in response for key in required_keys):
                self.logger.warning(f"Missing required keys in response. Has: {list(response.keys())}")
                return False

            # ?‘ë‹µ ?œê°„ ê²€ì¦?(ê²½ê³ ë§?ì¶œë ¥, ?¤íŒ¨ë¡?ì²˜ë¦¬?˜ì? ?ŠìŒ)
            if response_time > TEST_CONFIG["performance_thresholds"]["response_time"]:
                self.logger.warning(f"Response time {response_time:.2f}s exceeds threshold {TEST_CONFIG['performance_thresholds']['response_time']}s")

            # ?‘ë‹µ ?´ìš© ê²€ì¦?
            if not response["response"] or len(response["response"]) < TEST_CONFIG["performance_thresholds"]["min_answer_length"]:
                self.logger.warning(f"Response too short: {len(response.get('response', ''))} chars")
                return False

            # ? ë¢°??ê²€ì¦?
            if response["confidence"] < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                self.logger.warning(f"Confidence {response['confidence']} below threshold {TEST_CONFIG['performance_thresholds']['min_confidence']}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating chat response: {e}")
            return False

    def _validate_classification(self, classification: QuestionClassification, expected_type: str) -> bool:
        """ì§ˆë¬¸ ë¶„ë¥˜ ê²°ê³¼ ê²€ì¦?""
        try:
            # ë¶„ë¥˜ ê²°ê³¼ ì¡´ì¬ ?•ì¸
            if not classification:
                return False

            # ? ë¢°??ê²€ì¦?
            if classification.confidence < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                return False

            # ê°€ì¤‘ì¹˜ ?©ê³„ ê²€ì¦?(?€?µì ?¼ë¡œ 1.0??ê°€ê¹Œì›Œ????
            total_weight = classification.law_weight + classification.precedent_weight
            if not (0.8 <= total_weight <= 1.2):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating classification: {e}")
            return False

    def _validate_answer_quality(self, answer_result: AnswerResult, response_time: float) -> bool:
        """?µë? ?ˆì§ˆ ê²€ì¦?""
        try:
            # ?µë? ê²°ê³¼ ì¡´ì¬ ?•ì¸
            if not answer_result or not answer_result.answer:
                return False

            # ?‘ë‹µ ?œê°„ ê²€ì¦?
            if response_time > TEST_CONFIG["performance_thresholds"]["response_time"]:
                return False

            # ?µë? ê¸¸ì´ ê²€ì¦?
            answer_length = len(answer_result.answer)
            if not (TEST_CONFIG["performance_thresholds"]["min_answer_length"] <=
                   answer_length <= TEST_CONFIG["performance_thresholds"]["max_answer_length"]):
                return False

            # ? ë¢°??ê²€ì¦?
            if answer_result.confidence.confidence < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating answer quality: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """ëª¨ë“  ?µí•© ?ŒìŠ¤???¤í–‰"""
        self.logger.info("Starting RAG system integration tests...")
        start_time = time.time()

        all_results = []

        try:
            # 1. ChatService ?µí•© ?ŒìŠ¤??
            chat_results = await self.test_chat_service_integration()
            all_results.extend(chat_results)

            # 2. ì§ˆë¬¸ ë¶„ë¥˜ ?œìŠ¤???ŒìŠ¤??
            classification_results = self.test_question_classification_system()
            all_results.extend(classification_results)

            # 3. ?µë? ?ì„± ?ˆì§ˆ ?ŒìŠ¤??
            answer_results = self.test_answer_generation_quality()
            all_results.extend(answer_results)

        except Exception as e:
            self.logger.error(f"Error during test execution: {e}")

        total_time = time.time() - start_time

        # ?ŒìŠ¤??ê²°ê³¼ ë¶„ì„
        test_summary = self._analyze_test_results(all_results, total_time)

        return test_summary

    def _analyze_test_results(self, results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """?ŒìŠ¤??ê²°ê³¼ ë¶„ì„"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests

        # ?±ê³µë¥?ê³„ì‚°
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # ?‰ê·  ?±ëŠ¥ ì§€??ê³„ì‚°
        avg_response_time = sum(r.response_time for r in results) / total_tests if total_tests > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total_tests if total_tests > 0 else 0
        avg_answer_length = sum(r.answer_length for r in results) / total_tests if total_tests > 0 else 0

        # ì§ˆë¬¸ ? í˜•ë³??±ê³µë¥?
        type_stats = {}
        for result in results:
            if result.question_type not in type_stats:
                type_stats[result.question_type] = {"total": 0, "passed": 0}
            type_stats[result.question_type]["total"] += 1
            if result.passed:
                type_stats[result.question_type]["passed"] += 1

        for question_type in type_stats:
            stats = type_stats[question_type]
            stats["success_rate"] = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0

        # ?¤íŒ¨???ŒìŠ¤??ëª©ë¡
        failed_tests_list = [r for r in results if not r.passed]

        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_execution_time": total_time
            },
            "performance_metrics": {
                "avg_response_time": avg_response_time,
                "avg_confidence": avg_confidence,
                "avg_answer_length": avg_answer_length
            },
            "question_type_stats": type_stats,
            "failed_tests": [
                {
                    "test_name": r.test_name,
                    "question_type": r.question_type,
                    "error_message": r.error_message,
                    "response_time": r.response_time,
                    "confidence": r.confidence
                } for r in failed_tests_list
            ],
            "test_timestamp": datetime.now().isoformat()
        }

        return summary

    def generate_test_report(self, test_summary: Dict[str, Any]) -> str:
        """?ŒìŠ¤??ë³´ê³ ???ì„±"""
        report = f"""
# RAG ?œìŠ¤???µí•© ?ŒìŠ¤??ë³´ê³ ??

## ?ŒìŠ¤??ê°œìš”
- **?¤í–‰ ?œê°„**: {test_summary['test_summary']['total_execution_time']:.2f}ì´?
- **ì´??ŒìŠ¤????*: {test_summary['test_summary']['total_tests']}ê°?
- **?±ê³µ???ŒìŠ¤??*: {test_summary['test_summary']['passed_tests']}ê°?
- **?¤íŒ¨???ŒìŠ¤??*: {test_summary['test_summary']['failed_tests']}ê°?
- **?±ê³µë¥?*: {test_summary['test_summary']['success_rate']:.1f}%

## ?±ëŠ¥ ì§€??
- **?‰ê·  ?‘ë‹µ ?œê°„**: {test_summary['performance_metrics']['avg_response_time']:.2f}ì´?
- **?‰ê·  ? ë¢°??*: {test_summary['performance_metrics']['avg_confidence']:.3f}
- **?‰ê·  ?µë? ê¸¸ì´**: {test_summary['performance_metrics']['avg_answer_length']:.0f}??

## ì§ˆë¬¸ ? í˜•ë³??±ê³µë¥?
"""

        for question_type, stats in test_summary['question_type_stats'].items():
            report += f"- **{question_type}**: {stats['success_rate']:.1f}% ({stats['passed']}/{stats['total']})\n"

        if test_summary['failed_tests']:
            report += "\n## ?¤íŒ¨???ŒìŠ¤??n"
            for failed_test in test_summary['failed_tests']:
                report += f"- **{failed_test['test_name']}**: {failed_test['error_message']}\n"

        report += f"\n## ?ŒìŠ¤???¤í–‰ ?œê°„\n{test_summary['test_timestamp']}\n"

        return report


async def main():
    """ë©”ì¸ ?ŒìŠ¤???¤í–‰ ?¨ìˆ˜"""
    print("=" * 60)
    print("RAG ?œìŠ¤???µí•© ?ŒìŠ¤???œì‘")
    print("=" * 60)

    try:
        # ?ŒìŠ¤???¤ìœ„??ì´ˆê¸°??
        test_suite = RAGIntegrationTestSuite()

        # ëª¨ë“  ?ŒìŠ¤???¤í–‰
        test_summary = await test_suite.run_all_tests()

        # ê²°ê³¼ ì¶œë ¥
        print("\n" + "=" * 60)
        print("?ŒìŠ¤??ê²°ê³¼ ?”ì•½")
        print("=" * 60)

        summary = test_summary['test_summary']
        print(f"ì´??ŒìŠ¤???? {summary['total_tests']}")
        print(f"?±ê³µ???ŒìŠ¤?? {summary['passed_tests']}")
        print(f"?¤íŒ¨???ŒìŠ¤?? {summary['failed_tests']}")
        print(f"?±ê³µë¥? {summary['success_rate']:.1f}%")
        print(f"ì´??¤í–‰ ?œê°„: {summary['total_execution_time']:.2f}ì´?)

        # ?±ëŠ¥ ì§€??ì¶œë ¥
        metrics = test_summary['performance_metrics']
        print(f"\n?±ëŠ¥ ì§€??")
        print(f"- ?‰ê·  ?‘ë‹µ ?œê°„: {metrics['avg_response_time']:.2f}ì´?)
        print(f"- ?‰ê·  ? ë¢°?? {metrics['avg_confidence']:.3f}")
        print(f"- ?‰ê·  ?µë? ê¸¸ì´: {metrics['avg_answer_length']:.0f}??)

        # ì§ˆë¬¸ ? í˜•ë³?ê²°ê³¼ ì¶œë ¥
        print(f"\nì§ˆë¬¸ ? í˜•ë³??±ê³µë¥?")
        for question_type, stats in test_summary['question_type_stats'].items():
            print(f"- {question_type}: {stats['success_rate']:.1f}% ({stats['passed']}/{stats['total']})")

        # ?¤íŒ¨???ŒìŠ¤??ì¶œë ¥
        if test_summary['failed_tests']:
            print(f"\n?¤íŒ¨???ŒìŠ¤??")
            for failed_test in test_summary['failed_tests']:
                print(f"- {failed_test['test_name']}: {failed_test['error_message']}")

        # ?ì„¸ ë³´ê³ ???ì„± ë°??€??
        report = test_suite.generate_test_report(test_summary)

        # ë³´ê³ ???Œì¼ ?€??
        report_path = Path("reports/rag_integration_test_report.md")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n?ì„¸ ë³´ê³ ?œê? ?€?¥ë˜?ˆìŠµ?ˆë‹¤: {report_path}")

        # JSON ê²°ê³¼ ?€??
        json_path = Path("reports/rag_integration_test_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)

        print(f"JSON ê²°ê³¼ê°€ ?€?¥ë˜?ˆìŠµ?ˆë‹¤: {json_path}")

        return test_summary

    except Exception as e:
        print(f"?ŒìŠ¤???¤í–‰ ì¤??¤ë¥˜ ë°œìƒ: {e}")
        return None


if __name__ == "__main__":
    # ë¹„ë™ê¸??ŒìŠ¤???¤í–‰
    asyncio.run(main())
