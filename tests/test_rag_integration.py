#!/usr/bin/env python3
"""
RAG 시스템 통합 테스트

이 모듈은 LawFirmAI의 RAG 시스템 전체 통합 테스트를 수행합니다.
- ChatService 통합 테스트 (LangGraph 기반)
- 질문 분류 시스템 테스트 (6가지 질문 유형)
- 답변 생성 품질 테스트 (신뢰도 계산 및 답변 형식 검증)

Author: LawFirmAI Development Team
Date: 2024-01-XX
Version: 1.0.0
"""

import os
import sys
import json
import logging
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import RAG components
try:
    from source.services.chat_service import ChatService
    from source.services.question_classifier import QuestionClassifier, QuestionType, QuestionClassification
    from source.services.improved_answer_generator import ImprovedAnswerGenerator, AnswerResult
    from source.services.rag_service import MLEnhancedRAGService
    from source.services.hybrid_search_engine import HybridSearchEngine
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
            "손해배상 관련 판례를 찾아주세요",
            "이혼 위자료 판례를 검색해주세요",
            "계약 해제 관련 대법원 판례가 있나요?"
        ],
        "law_inquiry": [
            "민법 제750조의 내용이 무엇인가요?",
            "형법 제250조 살인죄에 대해 설명해주세요",
            "상법 제434조 이사의 책임에 대해 알려주세요"
        ],
        "legal_advice": [
            "계약서 작성 시 주의사항을 조언해주세요",
            "이혼 절차와 필요한 서류를 알려주세요",
            "손해배상 청구 방법을 안내해주세요"
        ],
        "procedure_guide": [
            "소송 제기 절차는 어떻게 되나요?",
            "부동산 등기 신청 방법을 알려주세요",
            "특허 출원 절차를 설명해주세요"
        ],
        "term_explanation": [
            "불법행위의 정의를 알려주세요",
            "채권과 채무의 차이점은 무엇인가요?",
            "소멸시효의 개념을 설명해주세요"
        ],
        "general_question": [
            "법률에 대해 궁금한 것이 있습니다",
            "법적 문제로 고민이 있습니다",
            "법률 상담이 필요합니다"
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
    """테스트 결과 데이터 클래스"""
    test_name: str
    passed: bool
    response_time: float
    confidence: float
    answer_length: int
    question_type: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class RAGIntegrationTestSuite:
    """RAG 시스템 통합 테스트 스위트"""
    
    def __init__(self):
        """테스트 스위트 초기화"""
        self.logger = get_logger(__name__)
        self.config = Config()
        self.test_results: List[TestResult] = []
        
        # RAG 컴포넌트 초기화
        self.chat_service = None
        self.question_classifier = None
        self.answer_generator = None
        self.rag_service = None
        self.hybrid_search_engine = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """RAG 컴포넌트 초기화"""
        try:
            if not RAG_MODULES_AVAILABLE:
                raise ImportError("RAG modules not available")
            
            # ChatService 초기화
            self.chat_service = ChatService(self.config)
            self.logger.info("ChatService initialized")
            
            # 개별 컴포넌트 초기화
            self.question_classifier = QuestionClassifier()
            self.answer_generator = ImprovedAnswerGenerator()
            
            # RAG 서비스 초기화 (Mock 사용)
            self.rag_service = Mock(spec=MLEnhancedRAGService)
            self.hybrid_search_engine = Mock(spec=HybridSearchEngine)
            
            self.logger.info("All RAG components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    async def test_chat_service_integration(self) -> List[TestResult]:
        """ChatService 통합 테스트"""
        self.logger.info("Starting ChatService integration tests...")
        results = []
        
        # 테스트 질문들
        test_questions = [
            "안녕하세요, 법률 상담이 필요합니다",
            "계약서 검토를 도와주세요",
            "이혼 절차에 대해 알려주세요"
        ]
        
        for question in test_questions:
            try:
                start_time = time.time()
                
                # ChatService를 통한 메시지 처리
                response = await self.chat_service.process_message(question)
                
                response_time = time.time() - start_time
                
                # 결과 검증
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
        """질문 분류 시스템 테스트 (6가지 질문 유형)"""
        self.logger.info("Starting question classification system tests...")
        results = []
        
        for question_type, questions in TEST_CONFIG["test_questions"].items():
            for question in questions:
                try:
                    start_time = time.time()
                    
                    # 질문 분류 수행
                    classification = self.question_classifier.classify_question(question)
                    
                    response_time = time.time() - start_time
                    
                    # 결과 검증
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
        """답변 생성 품질 테스트"""
        self.logger.info("Starting answer generation quality tests...")
        results = []
        
        # 테스트용 질문 분류 결과 생성
        test_classifications = {
            QuestionType.PRECEDENT_SEARCH: QuestionClassification(
                question_type=QuestionType.PRECEDENT_SEARCH,
                law_weight=0.2,
                precedent_weight=0.8,
                confidence=0.8,
                keywords=["판례", "검색"],
                patterns=[]
            ),
            QuestionType.LAW_INQUIRY: QuestionClassification(
                question_type=QuestionType.LAW_INQUIRY,
                law_weight=0.8,
                precedent_weight=0.2,
                confidence=0.9,
                keywords=["법률", "조문"],
                patterns=[]
            ),
            QuestionType.LEGAL_ADVICE: QuestionClassification(
                question_type=QuestionType.LEGAL_ADVICE,
                law_weight=0.5,
                precedent_weight=0.5,
                confidence=0.7,
                keywords=["조언", "방법"],
                patterns=[]
            )
        }
        
        test_questions = [
            "손해배상 관련 판례를 찾아주세요",
            "민법 제750조의 내용이 무엇인가요?",
            "계약서 작성 시 주의사항을 조언해주세요"
        ]
        
        for i, question in enumerate(test_questions):
            try:
                start_time = time.time()
                
                # Mock 소스 데이터
                mock_sources = {
                    "results": [
                        {"type": "law", "law_name": "민법", "article_number": "제750조", "similarity": 0.9},
                        {"type": "precedent", "case_name": "손해배상 사건", "case_number": "2023다12345", "similarity": 0.8}
                    ],
                    "law_results": [
                        {"law_name": "민법", "article_number": "제750조", "content": "불법행위로 인한 손해배상"}
                    ],
                    "precedent_results": [
                        {"case_name": "손해배상 사건", "case_number": "2023다12345", "summary": "불법행위 손해배상"}
                    ]
                }
                
                # 질문 유형에 따른 분류 결과 선택
                question_types = list(test_classifications.keys())
                classification = test_classifications[question_types[i % len(question_types)]]
                
                # 답변 생성
                answer_result = self.answer_generator.generate_answer(
                    query=question,
                    question_type=classification,
                    context="테스트 컨텍스트",
                    sources=mock_sources
                )
                
                response_time = time.time() - start_time
                
                # 결과 검증
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
        """ChatService 응답 검증"""
        try:
            # 필수 키 존재 확인
            required_keys = ["response", "confidence", "sources", "processing_time"]
            if not all(key in response for key in required_keys):
                self.logger.warning(f"Missing required keys in response. Has: {list(response.keys())}")
                return False
            
            # 응답 시간 검증 (경고만 출력, 실패로 처리하지 않음)
            if response_time > TEST_CONFIG["performance_thresholds"]["response_time"]:
                self.logger.warning(f"Response time {response_time:.2f}s exceeds threshold {TEST_CONFIG['performance_thresholds']['response_time']}s")
            
            # 응답 내용 검증
            if not response["response"] or len(response["response"]) < TEST_CONFIG["performance_thresholds"]["min_answer_length"]:
                self.logger.warning(f"Response too short: {len(response.get('response', ''))} chars")
                return False
            
            # 신뢰도 검증
            if response["confidence"] < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                self.logger.warning(f"Confidence {response['confidence']} below threshold {TEST_CONFIG['performance_thresholds']['min_confidence']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating chat response: {e}")
            return False
    
    def _validate_classification(self, classification: QuestionClassification, expected_type: str) -> bool:
        """질문 분류 결과 검증"""
        try:
            # 분류 결과 존재 확인
            if not classification:
                return False
            
            # 신뢰도 검증
            if classification.confidence < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                return False
            
            # 가중치 합계 검증 (대략적으로 1.0에 가까워야 함)
            total_weight = classification.law_weight + classification.precedent_weight
            if not (0.8 <= total_weight <= 1.2):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating classification: {e}")
            return False
    
    def _validate_answer_quality(self, answer_result: AnswerResult, response_time: float) -> bool:
        """답변 품질 검증"""
        try:
            # 답변 결과 존재 확인
            if not answer_result or not answer_result.answer:
                return False
            
            # 응답 시간 검증
            if response_time > TEST_CONFIG["performance_thresholds"]["response_time"]:
                return False
            
            # 답변 길이 검증
            answer_length = len(answer_result.answer)
            if not (TEST_CONFIG["performance_thresholds"]["min_answer_length"] <= 
                   answer_length <= TEST_CONFIG["performance_thresholds"]["max_answer_length"]):
                return False
            
            # 신뢰도 검증
            if answer_result.confidence.confidence < TEST_CONFIG["performance_thresholds"]["min_confidence"]:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating answer quality: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 통합 테스트 실행"""
        self.logger.info("Starting RAG system integration tests...")
        start_time = time.time()
        
        all_results = []
        
        try:
            # 1. ChatService 통합 테스트
            chat_results = await self.test_chat_service_integration()
            all_results.extend(chat_results)
            
            # 2. 질문 분류 시스템 테스트
            classification_results = self.test_question_classification_system()
            all_results.extend(classification_results)
            
            # 3. 답변 생성 품질 테스트
            answer_results = self.test_answer_generation_quality()
            all_results.extend(answer_results)
            
        except Exception as e:
            self.logger.error(f"Error during test execution: {e}")
        
        total_time = time.time() - start_time
        
        # 테스트 결과 분석
        test_summary = self._analyze_test_results(all_results, total_time)
        
        return test_summary
    
    def _analyze_test_results(self, results: List[TestResult], total_time: float) -> Dict[str, Any]:
        """테스트 결과 분석"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # 성공률 계산
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 평균 성능 지표 계산
        avg_response_time = sum(r.response_time for r in results) / total_tests if total_tests > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total_tests if total_tests > 0 else 0
        avg_answer_length = sum(r.answer_length for r in results) / total_tests if total_tests > 0 else 0
        
        # 질문 유형별 성공률
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
        
        # 실패한 테스트 목록
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
        """테스트 보고서 생성"""
        report = f"""
# RAG 시스템 통합 테스트 보고서

## 테스트 개요
- **실행 시간**: {test_summary['test_summary']['total_execution_time']:.2f}초
- **총 테스트 수**: {test_summary['test_summary']['total_tests']}개
- **성공한 테스트**: {test_summary['test_summary']['passed_tests']}개
- **실패한 테스트**: {test_summary['test_summary']['failed_tests']}개
- **성공률**: {test_summary['test_summary']['success_rate']:.1f}%

## 성능 지표
- **평균 응답 시간**: {test_summary['performance_metrics']['avg_response_time']:.2f}초
- **평균 신뢰도**: {test_summary['performance_metrics']['avg_confidence']:.3f}
- **평균 답변 길이**: {test_summary['performance_metrics']['avg_answer_length']:.0f}자

## 질문 유형별 성공률
"""
        
        for question_type, stats in test_summary['question_type_stats'].items():
            report += f"- **{question_type}**: {stats['success_rate']:.1f}% ({stats['passed']}/{stats['total']})\n"
        
        if test_summary['failed_tests']:
            report += "\n## 실패한 테스트\n"
            for failed_test in test_summary['failed_tests']:
                report += f"- **{failed_test['test_name']}**: {failed_test['error_message']}\n"
        
        report += f"\n## 테스트 실행 시간\n{test_summary['test_timestamp']}\n"
        
        return report


async def main():
    """메인 테스트 실행 함수"""
    print("=" * 60)
    print("RAG 시스템 통합 테스트 시작")
    print("=" * 60)
    
    try:
        # 테스트 스위트 초기화
        test_suite = RAGIntegrationTestSuite()
        
        # 모든 테스트 실행
        test_summary = await test_suite.run_all_tests()
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        
        summary = test_summary['test_summary']
        print(f"총 테스트 수: {summary['total_tests']}")
        print(f"성공한 테스트: {summary['passed_tests']}")
        print(f"실패한 테스트: {summary['failed_tests']}")
        print(f"성공률: {summary['success_rate']:.1f}%")
        print(f"총 실행 시간: {summary['total_execution_time']:.2f}초")
        
        # 성능 지표 출력
        metrics = test_summary['performance_metrics']
        print(f"\n성능 지표:")
        print(f"- 평균 응답 시간: {metrics['avg_response_time']:.2f}초")
        print(f"- 평균 신뢰도: {metrics['avg_confidence']:.3f}")
        print(f"- 평균 답변 길이: {metrics['avg_answer_length']:.0f}자")
        
        # 질문 유형별 결과 출력
        print(f"\n질문 유형별 성공률:")
        for question_type, stats in test_summary['question_type_stats'].items():
            print(f"- {question_type}: {stats['success_rate']:.1f}% ({stats['passed']}/{stats['total']})")
        
        # 실패한 테스트 출력
        if test_summary['failed_tests']:
            print(f"\n실패한 테스트:")
            for failed_test in test_summary['failed_tests']:
                print(f"- {failed_test['test_name']}: {failed_test['error_message']}")
        
        # 상세 보고서 생성 및 저장
        report = test_suite.generate_test_report(test_summary)
        
        # 보고서 파일 저장
        report_path = Path("reports/rag_integration_test_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n상세 보고서가 저장되었습니다: {report_path}")
        
        # JSON 결과 저장
        json_path = Path("reports/rag_integration_test_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)
        
        print(f"JSON 결과가 저장되었습니다: {json_path}")
        
        return test_summary
        
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")
        return None


if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(main())
