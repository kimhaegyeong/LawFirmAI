# -*- coding: utf-8 -*-
"""
LawFirmAI 질의 답변 시스템 테스트 스크립트
Gradio 실행 전에 시스템이 정상적으로 동작하는지 확인
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로깅 설정
import logging
import warnings

# 경고 메시지 필터링
warnings.filterwarnings("ignore")

# 로거 설정
logger = logging.getLogger(__name__)

class QuerySystemTester:
    """질의 답변 시스템 테스트 클래스"""

    def __init__(self):
        self.test_results = {}
        self.chat_service = None
        self.config = None

    def setup_test_environment(self) -> bool:
        """테스트 환경 설정"""
        try:
            logger.info("[SETUP] Setting up test environment...")

            # 환경 변수 설정
            os.environ.setdefault('USE_LANGGRAPH', 'false')
            os.environ.setdefault('GEMINI_ENABLED', 'false')

            # Config 초기화
            from core.utils.config import Config
            self.config = Config()

            # ChatService 초기화
            from source.services.chat_service import ChatService
            self.chat_service = ChatService(self.config)

            logger.info("[SETUP] Test environment setup completed")
            return True

        except Exception as e:
            logger.error(f"[FAIL] Failed to setup test environment: {e}")
            return False

    async def test_chat_service_initialization(self) -> Dict[str, Any]:
        """ChatService 초기화 테스트"""
        test_name = "chat_service_initialization"
        logger.info(f"[TEST] Testing {test_name}...")

        try:
            # 서비스 상태 확인
            status = self.chat_service.get_service_status()

            result = {
                "test_name": test_name,
                "passed": True,
                "status": status,
                "message": "ChatService initialized successfully"
            }

            logger.info(f"{test_name} passed")
            return result

        except Exception as e:
            result = {
                "test_name": test_name,
                "passed": False,
                "error": str(e),
                "message": f"ChatService initialization failed: {e}"
            }
            logger.error(f"{test_name} failed: {e}")
            return result

    async def test_basic_query_processing(self) -> Dict[str, Any]:
        """기본 질의 처리 테스트"""
        test_name = "basic_query_processing"
        logger.info(f"[TEST] Testing {test_name}...")

        test_queries = [
            "안녕하세요",
            "계약서 검토 요청",
            "민법 제750조의 내용이 무엇인가요?",
            "손해배상 관련 판례를 찾아주세요",
            "이혼 절차는 어떻게 진행하나요?",
            "부동산 매매계약서 작성 시 주의사항은?",
            "근로기준법상 휴가수당은 어떻게 계산하나요?",
            "형법 제250조 살인의 정의를 알려주세요",
            "상속 포기 절차와 기간은?",
            "임대차보증금 반환 청구 방법은?"
        ]

        results = []

        for i, query in enumerate(test_queries):
            try:
                logger.info(f"[{i+1}/{len(test_queries)}] Processing: {query[:50]}...")
                start_time = time.time()
                result = await self.chat_service.process_message(query)
                processing_time = time.time() - start_time

                test_result = {
                    "query": query,
                    "response": result.get("response", ""),
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": processing_time,
                    "success": bool(result.get("response"))
                }

                results.append(test_result)
                logger.info(f"Query processed in {processing_time:.2f}s")

            except Exception as e:
                test_result = {
                    "query": query,
                    "error": str(e),
                    "success": False
                }
                results.append(test_result)
                logger.error(f"Query failed: {str(e)[:100]}...")

        # 전체 테스트 결과
        passed_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)

        result = {
            "test_name": test_name,
            "passed": passed_count == total_count,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": results,
            "message": f"Processed {passed_count}/{total_count} queries successfully"
        }

        if result["passed"]:
            logger.info(f"{test_name} passed ({passed_count}/{total_count})")
        else:
            logger.warning(f"{test_name} partially passed ({passed_count}/{total_count})")

        return result

    async def test_service_components(self) -> Dict[str, Any]:
        """서비스 컴포넌트 테스트"""
        test_name = "service_components"
        logger.info(f"[TEST] Testing {test_name}...")

        components_status = {}

        # 각 컴포넌트 상태 확인
        components_to_check = [
            ("rag_service", self.chat_service.rag_service),
            ("hybrid_search_engine", self.chat_service.hybrid_search_engine),
            ("question_classifier", self.chat_service.question_classifier),
            ("improved_answer_generator", self.chat_service.improved_answer_generator)
        ]

        for component_name, component in components_to_check:
            try:
                if component is not None:
                    components_status[component_name] = {
                        "available": True,
                        "type": type(component).__name__
                    }
                else:
                    components_status[component_name] = {
                        "available": False,
                        "error": "Component not initialized"
                    }
            except Exception as e:
                components_status[component_name] = {
                    "available": False,
                    "error": str(e)
                }

        # 전체 상태 평가
        available_count = sum(1 for status in components_status.values() if status.get("available", False))
        total_count = len(components_status)

        result = {
            "test_name": test_name,
            "passed": available_count > 0,  # 최소 하나의 컴포넌트라도 사용 가능하면 통과
            "available_count": available_count,
            "total_count": total_count,
            "components_status": components_status,
            "message": f"{available_count}/{total_count} components available"
        }

        if result["passed"]:
            logger.info(f"{test_name} passed ({available_count}/{total_count} components available)")
        else:
            logger.error(f"{test_name} failed (no components available)")

        return result

    async def test_input_validation(self) -> Dict[str, Any]:
        """입력 검증 테스트"""
        test_name = "input_validation"
        logger.info(f"[TEST] Testing {test_name}...")

        test_cases = [
            ("", False, "Empty input"),
            ("   ", False, "Whitespace only"),
            ("Valid question", True, "Valid input"),
            ("A" * 10001, False, "Too long input"),
            ("Normal question with reasonable length", True, "Normal length")
        ]

        results = []

        for test_input, expected, description in test_cases:
            try:
                is_valid = self.chat_service.validate_input(test_input)
                test_result = {
                    "input": test_input[:50] + "..." if len(test_input) > 50 else test_input,
                    "expected": expected,
                    "actual": is_valid,
                    "passed": is_valid == expected,
                    "description": description
                }
                results.append(test_result)

                if test_result["passed"]:
                    logger.info(f"{description}: {is_valid}")
                else:
                    logger.warning(f"{description}: expected {expected}, got {is_valid}")

            except Exception as e:
                test_result = {
                    "input": test_input[:50] + "..." if len(test_input) > 50 else test_input,
                    "error": str(e),
                    "passed": False,
                    "description": description
                }
                results.append(test_result)
                logger.error(f"{description}: {e}")

        passed_count = sum(1 for r in results if r.get("passed", False))
        total_count = len(results)

        result = {
            "test_name": test_name,
            "passed": passed_count == total_count,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": results,
            "message": f"Passed {passed_count}/{total_count} validation tests"
        }

        if result["passed"]:
            logger.info(f"{test_name} passed ({passed_count}/{total_count})")
        else:
            logger.warning(f"{test_name} partially passed ({passed_count}/{total_count})")

        return result

    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """성능 벤치마크 테스트"""
        test_name = "performance_benchmark"
        logger.info(f"[TEST] Testing {test_name}...")

        test_query = "계약서 검토 요청"
        iterations = 5
        response_times = []

        for i in range(iterations):
            try:
                start_time = time.time()
                result = await self.chat_service.process_message(test_query)
                response_time = time.time() - start_time
                response_times.append(response_time)

                logger.info(f"Iteration {i+1}: {response_time:.2f}s")

            except Exception as e:
                logger.error(f"Iteration {i+1} failed: {e}")
                response_times.append(None)

        # 성능 통계 계산
        valid_times = [t for t in response_times if t is not None]

        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)

            result = {
                "test_name": test_name,
                "passed": len(valid_times) >= iterations * 0.8,  # 80% 이상 성공하면 통과
                "iterations": iterations,
                "successful_iterations": len(valid_times),
                "avg_response_time": avg_time,
                "min_response_time": min_time,
                "max_response_time": max_time,
                "response_times": response_times,
                "message": f"Average response time: {avg_time:.2f}s"
            }

            logger.info(f"{test_name} passed - Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")
        else:
            result = {
                "test_name": test_name,
                "passed": False,
                "iterations": iterations,
                "successful_iterations": 0,
                "message": "All iterations failed"
            }
            logger.error(f"{test_name} failed - All iterations failed")

        return result

    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("[TEST] Starting comprehensive query system tests...")

        # 테스트 환경 설정
        if not self.setup_test_environment():
            return {
                "overall_passed": False,
                "error": "Failed to setup test environment",
                "tests": {}
            }

        # 테스트 실행
        tests = [
            self.test_chat_service_initialization(),
            self.test_service_components(),
            self.test_input_validation(),
            self.test_basic_query_processing(),
            self.test_performance_benchmark()
        ]

        test_results = await asyncio.gather(*tests, return_exceptions=True)

        # 결과 정리
        results = {}
        passed_count = 0
        total_count = len(test_results)

        for i, test_result in enumerate(test_results):
            if isinstance(test_result, Exception):
                test_name = f"test_{i}"
                results[test_name] = {
                    "test_name": test_name,
                    "passed": False,
                    "error": str(test_result),
                    "message": f"Test failed with exception: {test_result}"
                }
                logger.error(f"{test_name} failed with exception: {test_result}")
            else:
                test_name = test_result.get("test_name", f"test_{i}")
                results[test_name] = test_result
                if test_result.get("passed", False):
                    passed_count += 1
                    logger.info(f"{test_name} passed")
                else:
                    logger.warning(f"{test_name} failed or partially passed")

        # 전체 결과
        overall_result = {
            "overall_passed": passed_count >= total_count * 0.8,  # 80% 이상 통과하면 전체 통과
            "passed_count": passed_count,
            "total_count": total_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "tests": results,
            "summary": {
                "total_tests": total_count,
                "passed_tests": passed_count,
                "failed_tests": total_count - passed_count,
                "pass_rate": f"{(passed_count / total_count * 100):.1f}%" if total_count > 0 else "0%"
            }
        }

        # 결과 출력
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_count}")
        logger.info(f"Passed: {passed_count}")
        logger.info(f"Failed: {total_count - passed_count}")
        logger.info(f"Pass Rate: {overall_result['pass_rate']:.1%}")
        logger.info(f"Overall Status: {'PASSED' if overall_result['overall_passed'] else 'FAILED'}")
        logger.info("=" * 60)

        return overall_result

async def main():
    """메인 함수"""
    tester = QuerySystemTester()
    result = await tester.run_all_tests()

    # 결과를 JSON 파일로 저장
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"test_results_{timestamp}.json"

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"[INFO] Test results saved to: {result_file}")

    # 종료 코드 설정
    exit_code = 0 if result["overall_passed"] else 1
    logger.info(f"[INFO] Exiting with code: {exit_code}")

    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
