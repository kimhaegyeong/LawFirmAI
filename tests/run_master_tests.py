# -*- coding: utf-8 -*-
"""
LawFirmAI 마스터 테스트 실행기
모든 테스트를 순차적으로 실행하고 종합 결과를 제공
"""

import unittest
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 콘솔 인코딩 설정
def setup_console_encoding():
    """콘솔 인코딩 설정"""
    try:
        if sys.platform == "win32":
            os.system("chcp 65001 > nul")
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
            os.environ['PYTHONIOENCODING'] = 'utf-8'
        return True
    except:
        return False

# 인코딩 설정 실행
setup_console_encoding()

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class MasterTestRunner:
    """마스터 테스트 실행기"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("=" * 80)
        print("LawFirmAI 마스터 테스트 실행")
        print("=" * 80)
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # 테스트 순서 정의
        test_suites = [
            {
                "name": "Phase 1 대화 맥락 강화",
                "module": "test_phase1_context_enhancement",
                "description": "세션 관리, 다중 턴 처리, 컨텍스트 압축"
            },
            {
                "name": "Phase 2 개인화 및 분석",
                "module": "test_phase2_personalization_analysis",
                "description": "사용자 프로필, 감정 분석, 대화 흐름 추적"
            },
            {
                "name": "Phase 3 장기 기억 및 품질 모니터링",
                "module": "test_phase3_memory_quality",
                "description": "맥락적 메모리, 대화 품질 모니터링"
            },
            {
                "name": "Gradio 기본 테스트",
                "module": "test_gradio_basic",
                "description": "Gradio 인터페이스 및 기본 기능"
            },
            {
                "name": "성능 최적화 테스트",
                "module": "test_optimized_performance",
                "description": "성능 모니터링, 메모리 최적화, 캐시 관리"
            },
            {
                "name": "종합 시스템 테스트",
                "module": "test_comprehensive_system",
                "description": "전체 시스템 통합 테스트"
            },
            {
                "name": "스트레스 테스트",
                "module": "test_stress_system",
                "description": "대량 데이터 처리 및 동시성 테스트"
            },
            {
                "name": "성능 벤치마크",
                "module": "test_performance_benchmark",
                "description": "응답 시간, 처리량, 리소스 사용량 측정"
            },
            {
                "name": "AKLS 통합 테스트",
                "module": "akls.test_akls_integration",
                "description": "AKLS 데이터 처리, 검색, RAG 통합 기능"
            },
            {
                "name": "AKLS Gradio 테스트",
                "module": "akls.test_akls_gradio",
                "description": "AKLS Gradio 인터페이스 통합 테스트"
            },
            {
                "name": "AKLS 성능 테스트",
                "module": "akls.test_akls_performance",
                "description": "AKLS 시스템 성능 벤치마크"
            }
        ]
        
        # 각 테스트 스위트 실행
        for suite in test_suites:
            print(f"\n{'='*60}")
            print(f"실행 중: {suite['name']}")
            print(f"설명: {suite['description']}")
            print(f"{'='*60}")
            
            try:
                result = self._run_test_suite(suite)
                self.test_results[suite['name']] = result
                
                # 결과 출력
                self._print_suite_result(suite['name'], result)
                
            except Exception as e:
                print(f"❌ 테스트 스위트 실행 실패: {e}")
                self.test_results[suite['name']] = {
                    "success": False,
                    "error": str(e),
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 0
                }
        
        self.end_time = time.time()
        
        # 종합 결과 출력
        self._print_final_results()
        
        return self.test_results
    
    def _run_test_suite(self, suite: Dict[str, str]) -> Dict[str, Any]:
        """개별 테스트 스위트 실행"""
        try:
            # 모듈 동적 임포트
            module_name = f"tests.{suite['module']}"
            module = __import__(module_name, fromlist=[''])
            
            # 테스트 스위트 생성
            test_suite = unittest.TestSuite()
            
            # 모듈의 모든 테스트 클래스 찾기
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(obj))
            
            # 테스트 실행
            runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
            result = runner.run(test_suite)
            
            return {
                "success": len(result.failures) == 0 and len(result.errors) == 0,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "failure_details": result.failures,
                "error_details": result.errors
            }
            
        except ImportError as e:
            print(f"⚠️ 모듈 임포트 실패: {suite['module']} - {e}")
            return {
                "success": False,
                "error": f"모듈 임포트 실패: {e}",
                "tests_run": 0,
                "failures": 0,
                "errors": 0
            }
        except Exception as e:
            print(f"❌ 테스트 실행 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "tests_run": 0,
                "failures": 0,
                "errors": 0
            }
    
    def _print_suite_result(self, suite_name: str, result: Dict[str, Any]):
        """개별 테스트 스위트 결과 출력"""
        if result["success"]:
            print(f"✅ {suite_name}: 성공")
            print(f"   실행된 테스트: {result['tests_run']}")
            print(f"   통과율: 100%")
        else:
            print(f"❌ {suite_name}: 실패")
            print(f"   실행된 테스트: {result['tests_run']}")
            print(f"   실패: {result['failures']}")
            print(f"   오류: {result['errors']}")
            
            if result.get("error"):
                print(f"   오류 메시지: {result['error']}")
    
    def _print_final_results(self):
        """최종 결과 출력"""
        print("\n" + "="*80)
        print("LawFirmAI 마스터 테스트 최종 결과")
        print("="*80)
        
        total_time = self.end_time - self.start_time
        print(f"총 실행 시간: {total_time:.2f}초")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 전체 통계 계산
        total_tests = sum(result.get("tests_run", 0) for result in self.test_results.values())
        total_failures = sum(result.get("failures", 0) for result in self.test_results.values())
        total_errors = sum(result.get("errors", 0) for result in self.test_results.values())
        successful_suites = sum(1 for result in self.test_results.values() if result.get("success", False))
        
        print(f"\n전체 통계:")
        print(f"  테스트 스위트: {len(self.test_results)}")
        print(f"  성공한 스위트: {successful_suites}")
        print(f"  실패한 스위트: {len(self.test_results) - successful_suites}")
        print(f"  총 테스트: {total_tests}")
        print(f"  총 실패: {total_failures}")
        print(f"  총 오류: {total_errors}")
        
        if total_tests > 0:
            success_rate = (total_tests - total_failures - total_errors) / total_tests * 100
            print(f"  전체 통과율: {success_rate:.1f}%")
        else:
            print(f"  전체 통과율: 0%")
        
        # 스위트별 상세 결과
        print(f"\n스위트별 상세 결과:")
        for suite_name, result in self.test_results.items():
            status = "✅" if result.get("success", False) else "❌"
            tests_run = result.get("tests_run", 0)
            failures = result.get("failures", 0)
            errors = result.get("errors", 0)
            
            if tests_run > 0:
                suite_success_rate = (tests_run - failures - errors) / tests_run * 100
                print(f"  {status} {suite_name}: {suite_success_rate:.1f}% ({tests_run - failures - errors}/{tests_run})")
            else:
                print(f"  {status} {suite_name}: 실행되지 않음")
        
        # 실패한 테스트 상세 정보
        failed_suites = [name for name, result in self.test_results.items() if not result.get("success", False)]
        if failed_suites:
            print(f"\n실패한 스위트 상세 정보:")
            for suite_name in failed_suites:
                result = self.test_results[suite_name]
                print(f"\n  ❌ {suite_name}:")
                
                if result.get("error"):
                    print(f"    오류: {result['error']}")
                
                if result.get("failure_details"):
                    print(f"    실패한 테스트:")
                    for test, traceback in result["failure_details"]:
                        print(f"      - {test}: {traceback.split('AssertionError:')[-1].strip()}")
                
                if result.get("error_details"):
                    print(f"    오류가 발생한 테스트:")
                    for test, traceback in result["error_details"]:
                        print(f"      - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        # 최종 평가
        print(f"\n{'='*80}")
        print("최종 평가")
        print(f"{'='*80}")
        
        if successful_suites == len(self.test_results):
            print("🎉 모든 테스트 스위트가 성공했습니다!")
            print("   LawFirmAI 시스템이 완벽하게 작동합니다.")
            grade = "A+"
        elif successful_suites >= len(self.test_results) * 0.8:
            print("✅ 대부분의 테스트가 성공했습니다!")
            print("   시스템이 안정적으로 작동하지만 일부 개선이 필요합니다.")
            grade = "A"
        elif successful_suites >= len(self.test_results) * 0.6:
            print("⚠️ 일부 테스트가 실패했습니다.")
            print("   시스템의 안정성을 위해 개선이 필요합니다.")
            grade = "B"
        else:
            print("❌ 많은 테스트가 실패했습니다.")
            print("   시스템에 심각한 문제가 있습니다. 즉시 점검해주세요.")
            grade = "C"
        
        print(f"\n최종 등급: {grade}")
        
        # 권장사항
        print(f"\n권장사항:")
        if grade in ["A+", "A"]:
            print("  - 시스템이 안정적으로 작동합니다.")
            print("  - 프로덕션 배포를 진행할 수 있습니다.")
            print("  - 정기적인 모니터링을 통해 성능을 유지하세요.")
        elif grade == "B":
            print("  - 실패한 테스트를 분석하여 문제를 해결하세요.")
            print("  - 시스템 안정성을 확인한 후 배포를 고려하세요.")
            print("  - 추가 테스트를 통해 개선사항을 검증하세요.")
        else:
            print("  - 시스템의 핵심 기능을 점검하세요.")
            print("  - 오류 로그를 분석하여 근본 원인을 파악하세요.")
            print("  - 문제 해결 후 다시 테스트를 실행하세요.")
        
        print(f"\n{'='*80}")
    
    def save_results(self, filename: str = None):
        """결과를 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        import json
        
        # JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for suite_name, result in self.test_results.items():
            serializable_results[suite_name] = {
                "success": result.get("success", False),
                "tests_run": result.get("tests_run", 0),
                "failures": result.get("failures", 0),
                "errors": result.get("errors", 0),
                "error": result.get("error", None)
            }
        
        # 메타데이터 추가
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": self.end_time - self.start_time if self.end_time and self.start_time else 0,
            "total_suites": len(self.test_results),
            "successful_suites": sum(1 for result in self.test_results.values() if result.get("success", False)),
            "total_tests": sum(result.get("tests_run", 0) for result in self.test_results.values()),
            "total_failures": sum(result.get("failures", 0) for result in self.test_results.values()),
            "total_errors": sum(result.get("errors", 0) for result in self.test_results.values())
        }
        
        output_data = {
            "metadata": metadata,
            "results": serializable_results
        }
        
        # 파일 저장
        output_path = Path("reports") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_path}")


def main():
    """메인 실행 함수"""
    runner = MasterTestRunner()
    
    try:
        # 모든 테스트 실행
        results = runner.run_all_tests()
        
        # 결과 저장
        runner.save_results()
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
        return None
    except Exception as e:
        print(f"\n\n테스트 실행 중 오류가 발생했습니다: {e}")
        return None


if __name__ == "__main__":
    main()
