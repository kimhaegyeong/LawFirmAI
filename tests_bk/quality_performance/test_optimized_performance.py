#!/usr/bin/env python3
"""
?�능 최적???�스???�크립트
최적???�후 ?�능 비교 �?벤치마크
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ?�로?�트 루트 경로 추�?
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 ?�정
import logging
import warnings

warnings.filterwarnings("ignore")

class PerformanceTester:
    """?�능 ?�스???�래??""

    def __init__(self):
        self.test_results = {}
        self.optimized_service = None
        self.config = None

    def setup_test_environment(self) -> bool:
        """?�스???�경 ?�정"""
        try:
            print("[SETUP] Setting up optimized test environment...")

            # ?�경 변???�정
            os.environ.setdefault('USE_LANGGRAPH', 'false')
            os.environ.setdefault('GEMINI_ENABLED', 'false')

            # Config 초기??
            from source.utils.config import Config
            self.config = Config()

            # 최적?�된 ChatService 초기??
            from source.services.optimized_chat_service import OptimizedChatService
            self.optimized_service = OptimizedChatService(self.config)

            print("[SETUP] Optimized test environment setup completed")
            return True

        except Exception as e:
            print(f"[FAIL] Failed to setup test environment: {e}")
            return False

    async def test_optimized_performance(self) -> Dict[str, Any]:
        """최적?�된 ?�능 ?�스??""
        test_name = "optimized_performance"
        print(f"[TEST] Testing {test_name}...")

        test_queries = [
            "?�녕?�세??,
            "계약??검???�청",
            "민법 ??50조의 ?�용??무엇?��???",
            "?�해배상 관???��?�?찾아주세??,
            "?�혼 ?�차???�떻�?진행?�나??"
        ]

        results = []

        for i, query in enumerate(test_queries):
            try:
                print(f"[{i+1}/{len(test_queries)}] Processing: {query[:50]}...")
                start_time = time.time()
                result = await self.optimized_service.process_message(query)
                processing_time = time.time() - start_time

                test_result = {
                    "query": query,
                    "response": result.get("response", ""),
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": processing_time,
                    "success": bool(result.get("response")),
                    "cached": result.get("cached", False)
                }

                results.append(test_result)
                print(f"[OK] Query processed in {processing_time:.2f}s (cached: {test_result['cached']})")

            except Exception as e:
                test_result = {
                    "query": query,
                    "error": str(e),
                    "success": False
                }
                results.append(test_result)
                print(f"[FAIL] Query failed: {str(e)[:100]}...")

        # ?�체 ?�스??결과
        passed_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)
        avg_time = sum(r.get("processing_time", 0) for r in results if r.get("success", False)) / max(passed_count, 1)
        cached_count = sum(1 for r in results if r.get("cached", False))

        result = {
            "test_name": test_name,
            "passed": passed_count == total_count,
            "passed_count": passed_count,
            "total_count": total_count,
            "avg_processing_time": avg_time,
            "cached_responses": cached_count,
            "results": results,
            "message": f"Processed {passed_count}/{total_count} queries successfully (avg: {avg_time:.2f}s)"
        }

        if result["passed"]:
            print(f"[OK] {test_name} passed ({passed_count}/{total_count}) - Avg: {avg_time:.2f}s")
        else:
            print(f"[WARN] {test_name} partially passed ({passed_count}/{total_count}) - Avg: {avg_time:.2f}s")

        return result

    async def test_cache_performance(self) -> Dict[str, Any]:
        """캐시 ?�능 ?�스??""
        test_name = "cache_performance"
        print(f"[TEST] Testing {test_name}...")

        # ?�일??질문???�러 �?반복?�여 캐시 ?�과 측정
        test_query = "계약??검???�청"
        iterations = 5

        first_run_time = None
        cached_times = []

        for i in range(iterations):
            try:
                start_time = time.time()
                result = await self.optimized_service.process_message(test_query)
                processing_time = time.time() - start_time

                if i == 0:
                    first_run_time = processing_time
                else:
                    cached_times.append(processing_time)

                print(f"[OK] Iteration {i+1}: {processing_time:.2f}s (cached: {result.get('cached', False)})")

            except Exception as e:
                print(f"[FAIL] Iteration {i+1} failed: {e}")

        # 캐시 ?�과 계산
        avg_cached_time = sum(cached_times) / len(cached_times) if cached_times else 0
        speedup_factor = first_run_time / avg_cached_time if avg_cached_time > 0 else 0

        result = {
            "test_name": test_name,
            "passed": len(cached_times) >= iterations - 1,
            "first_run_time": first_run_time,
            "avg_cached_time": avg_cached_time,
            "speedup_factor": speedup_factor,
            "iterations": iterations,
            "message": f"Cache speedup: {speedup_factor:.1f}x (first: {first_run_time:.2f}s, cached: {avg_cached_time:.2f}s)"
        }

        print(f"[OK] {test_name} - Speedup: {speedup_factor:.1f}x")
        return result

    async def test_concurrent_performance(self) -> Dict[str, Any]:
        """?�시 처리 ?�능 ?�스??""
        test_name = "concurrent_performance"
        print(f"[TEST] Testing {test_name}...")

        test_queries = [
            "?�녕?�세??,
            "계약??검???�청",
            "민법 ??50조의 ?�용??무엇?��???",
            "?�해배상 관???��?�?찾아주세??,
            "?�혼 ?�차???�떻�?진행?�나??"
        ]

        # ?�시 처리
        start_time = time.time()
        tasks = [self.optimized_service.process_message(query) for query in test_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # 결과 분석
        successful_results = [r for r in results if isinstance(r, dict) and r.get("response")]
        failed_count = len(results) - len(successful_results)

        result = {
            "test_name": test_name,
            "passed": failed_count == 0,
            "total_time": total_time,
            "successful_requests": len(successful_results),
            "failed_requests": failed_count,
            "avg_time_per_request": total_time / len(test_queries),
            "throughput": len(test_queries) / total_time,
            "message": f"Processed {len(successful_results)}/{len(test_queries)} requests in {total_time:.2f}s"
        }

        print(f"[OK] {test_name} - {len(successful_results)}/{len(test_queries)} requests in {total_time:.2f}s")
        return result

    async def test_performance_stats(self) -> Dict[str, Any]:
        """?�능 ?�계 ?�스??""
        test_name = "performance_stats"
        print(f"[TEST] Testing {test_name}...")

        try:
            # ?�비???�태 ?�인
            service_status = self.optimized_service.get_service_status()
            performance_stats = self.optimized_service.get_performance_stats()

            result = {
                "test_name": test_name,
                "passed": True,
                "service_status": service_status,
                "performance_stats": performance_stats,
                "message": "Performance stats retrieved successfully"
            }

            print(f"[OK] {test_name} - Stats retrieved")
            return result

        except Exception as e:
            result = {
                "test_name": test_name,
                "passed": False,
                "error": str(e),
                "message": f"Failed to get performance stats: {e}"
            }
            print(f"[FAIL] {test_name} - {e}")
            return result

    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 ?�스???�행"""
        print("[TEST] Starting optimized performance tests...")

        # ?�스???�경 ?�정
        if not self.setup_test_environment():
            return {
                "overall_passed": False,
                "error": "Failed to setup test environment",
                "tests": {}
            }

        # ?�스???�행
        tests = [
            self.test_optimized_performance(),
            self.test_cache_performance(),
            self.test_concurrent_performance(),
            self.test_performance_stats()
        ]

        test_results = await asyncio.gather(*tests, return_exceptions=True)

        # 결과 ?�리
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
                print(f"[FAIL] {test_name} failed with exception: {test_result}")
            else:
                test_name = test_result.get("test_name", f"test_{i}")
                results[test_name] = test_result
                if test_result.get("passed", False):
                    passed_count += 1
                    print(f"[OK] {test_name} passed")
                else:
                    print(f"[WARN] {test_name} failed or partially passed")

        # ?�체 결과
        overall_result = {
            "overall_passed": passed_count >= total_count * 0.8,
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
        print("=" * 60)
        print("OPTIMIZED PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        print(f"Pass Rate: {overall_result['pass_rate']:.1%}")
        print(f"Overall Status: {'PASSED' if overall_result['overall_passed'] else 'FAILED'}")
        print("=" * 60)

        return overall_result

async def main():
    """메인 ?�수"""
    tester = PerformanceTester()
    result = await tester.run_all_tests()

    # 결과�?JSON ?�일�??�??
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"optimized_performance_test_results_{timestamp}.json"

    # JSON 직렬?��? ?�한 커스?� encoder
    def json_default(obj):
        """커스?� 객체�?JSON?�로 직렬??""
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return repr(obj)

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=json_default)

    print(f"[INFO] Test results saved to: {result_file}")

    # 종료 코드 ?�정
    exit_code = 0 if result["overall_passed"] else 1
    print(f"[INFO] Exiting with code: {exit_code}")

    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
