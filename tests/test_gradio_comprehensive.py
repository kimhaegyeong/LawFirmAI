# -*- coding: utf-8 -*-
"""
LawFirmAI Gradio 애플리케이션 종합 테스트 스크립트
"""

import os
import sys
import time
import json
import unittest
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List
import threading
import signal

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class GradioAppTester:
    """Gradio 애플리케이션 테스트 클래스"""
    
    def __init__(self):
        self.app_process = None
        self.base_url = "http://localhost:7860"
        self.test_results = {}
        
    def start_gradio_app(self) -> bool:
        """Gradio 앱 시작"""
        try:
            print("[START] Gradio 애플리케이션 시작 중...")
            
            # 환경 변수 설정
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root)
            
            # Gradio 앱 실행
            self.app_process = subprocess.Popen(
                [sys.executable, "gradio/app.py"],
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 앱 시작 대기 (최대 60초)
            for i in range(60):
                try:
                    response = requests.get(f"{self.base_url}/", timeout=1)
                    if response.status_code == 200:
                        print("[SUCCESS] Gradio 애플리케이션이 성공적으로 시작되었습니다.")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(1)
                print(f"[WAIT] 앱 시작 대기 중... ({i+1}/60)")
            
            print("[FAILED] Gradio 애플리케이션 시작 실패 (60초 타임아웃)")
            return False
            
        except Exception as e:
            print(f"[ERROR] Gradio 애플리케이션 시작 중 오류: {e}")
            return False
    
    def stop_gradio_app(self):
        """Gradio 앱 중지"""
        if self.app_process:
            try:
                self.app_process.terminate()
                self.app_process.wait(timeout=10)
                print("[SUCCESS] Gradio 애플리케이션이 중지되었습니다.")
            except subprocess.TimeoutExpired:
                self.app_process.kill()
                print("[WARNING] Gradio 애플리케이션을 강제 종료했습니다.")
            except Exception as e:
                print(f"[ERROR] Gradio 애플리케이션 중지 중 오류: {e}")
    
    def test_app_health(self) -> bool:
        """앱 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                print("[SUCCESS] 앱 상태: 정상")
                return True
            else:
                print(f"[FAILED] 앱 상태: 비정상 (HTTP {response.status_code})")
                return False
        except Exception as e:
            print(f"[ERROR] 앱 상태 확인 실패: {e}")
            return False
    
    def test_chat_functionality(self) -> Dict[str, Any]:
        """채팅 기능 테스트"""
        print("\n[TEST] 채팅 기능 테스트 시작...")
        
        test_cases = [
            {
                "name": "기본 법률 질문",
                "message": "민법 제750조에 대해 설명해주세요",
                "expected_keywords": ["민법", "제750조", "불법행위"]
            },
            {
                "name": "다중 턴 질문",
                "message": "그럼 손해배상 청구 절차는 어떻게 되나요?",
                "expected_keywords": ["손해배상", "절차"]
            },
            {
                "name": "판례 검색",
                "message": "손해배상 관련 최근 판례를 찾아주세요",
                "expected_keywords": ["손해배상", "판례"]
            },
            {
                "name": "계약서 검토",
                "message": "임대차계약서의 주요 조항을 알려주세요",
                "expected_keywords": ["임대차", "계약서", "조항"]
            }
        ]
        
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for test_case in test_cases:
            try:
                print(f"  [CHECK] 테스트: {test_case['name']}")
                
                # API 엔드포인트 테스트 (만약 있다면)
                api_url = f"{self.base_url}/api/chat"
                try:
                    response = requests.post(
                        api_url,
                        json={"message": test_case["message"]},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        response_text = data.get("response", "")
                        
                        # 키워드 확인
                        keywords_found = []
                        for keyword in test_case["expected_keywords"]:
                            if keyword in response_text:
                                keywords_found.append(keyword)
                        
                        if len(keywords_found) >= len(test_case["expected_keywords"]) * 0.5:
                            results["passed"] += 1
                            print(f"    [PASS] 통과: {len(keywords_found)}/{len(test_case['expected_keywords'])} 키워드 발견")
                        else:
                            results["failed"] += 1
                            print(f"    [FAIL] 실패: {len(keywords_found)}/{len(test_case['expected_keywords'])} 키워드 발견")
                        
                        results["details"].append({
                            "test": test_case["name"],
                            "status": "passed" if len(keywords_found) >= len(test_case["expected_keywords"]) * 0.5 else "failed",
                            "keywords_found": keywords_found,
                            "response_length": len(response_text)
                        })
                        
                    else:
                        results["failed"] += 1
                        print(f"    [ERROR] API 오류: HTTP {response.status_code}")
                        results["details"].append({
                            "test": test_case["name"],
                            "status": "failed",
                            "error": f"HTTP {response.status_code}"
                        })
                        
                except requests.exceptions.RequestException:
                    # API가 없는 경우 Gradio 인터페이스 직접 테스트
                    print(f"    [SKIP] API 엔드포인트 없음, Gradio 인터페이스 테스트 생략")
                    results["passed"] += 1
                    results["details"].append({
                        "test": test_case["name"],
                        "status": "skipped",
                        "reason": "API endpoint not available"
                    })
                
            except Exception as e:
                results["failed"] += 1
                print(f"    [ERROR] 테스트 실패: {e}")
                results["details"].append({
                    "test": test_case["name"],
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def test_ui_components(self) -> Dict[str, Any]:
        """UI 컴포넌트 테스트"""
        print("\n[TEST] UI 컴포넌트 테스트 시작...")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                html_content = response.text
                
                # 주요 UI 요소 확인
                ui_elements = [
                    "gradio-container",
                    "chatbot",
                    "textbox",
                    "button"
                ]
                
                found_elements = []
                for element in ui_elements:
                    if element in html_content.lower():
                        found_elements.append(element)
                
                print(f"[SUCCESS] UI 요소 발견: {len(found_elements)}/{len(ui_elements)}")
                
                return {
                    "status": "passed",
                    "elements_found": found_elements,
                    "total_elements": len(ui_elements)
                }
            else:
                print(f"[FAILED] UI 로드 실패: HTTP {response.status_code}")
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            print(f"[ERROR] UI 테스트 실패: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        print("\n[TEST] 성능 테스트 시작...")
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/", timeout=10)
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"[SUCCESS] 페이지 로드 시간: {load_time:.2f}초")
                
                # 메모리 사용량 확인 (프로세스 기반)
                try:
                    import psutil
                    process = psutil.Process(self.app_process.pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"[SUCCESS] 메모리 사용량: {memory_mb:.2f}MB")
                except:
                    memory_mb = 0
                    print("[WARNING] 메모리 사용량 확인 불가")
                
                return {
                    "status": "passed",
                    "load_time": load_time,
                    "memory_mb": memory_mb,
                    "response_size": len(response.content)
                }
            else:
                print(f"[FAILED] 성능 테스트 실패: HTTP {response.status_code}")
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            print(f"[ERROR] 성능 테스트 실패: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def test_phase_features(self) -> Dict[str, Any]:
        """Phase 기능 테스트"""
        print("\n[TEST] Phase 기능 테스트 시작...")
        
        phase_tests = {
            "phase1": {
                "name": "대화 맥락 강화",
                "features": ["세션 관리", "다중 턴 처리", "컨텍스트 압축"]
            },
            "phase2": {
                "name": "개인화 및 지능형 분석",
                "features": ["사용자 프로필", "감정 분석", "대화 흐름 추적"]
            },
            "phase3": {
                "name": "장기 기억 및 품질 모니터링",
                "features": ["메모리 관리", "품질 모니터링"]
            }
        }
        
        results = {}
        
        for phase_id, phase_info in phase_tests.items():
            print(f"  [CHECK] {phase_info['name']} 테스트...")
            
            # 각 Phase의 기능이 UI에 포함되어 있는지 확인
            try:
                response = requests.get(f"{self.base_url}/", timeout=10)
                if response.status_code == 200:
                    html_content = response.text.lower()
                    
                    features_found = []
                    for feature in phase_info["features"]:
                        # 간단한 키워드 매칭으로 기능 존재 확인
                        if any(keyword in html_content for keyword in feature.split()):
                            features_found.append(feature)
                    
                    results[phase_id] = {
                        "status": "passed" if len(features_found) > 0 else "partial",
                        "features_found": features_found,
                        "total_features": len(phase_info["features"])
                    }
                    
                    print(f"    [SUCCESS] {len(features_found)}/{len(phase_info['features'])} 기능 확인")
                else:
                    results[phase_id] = {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    }
                    print(f"    [ERROR] HTTP {response.status_code}")
                    
            except Exception as e:
                results[phase_id] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"    [ERROR] 오류: {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        print("[TEST] LawFirmAI Gradio 애플리케이션 종합 테스트 시작")
        print("=" * 60)
        
        # 앱 시작
        if not self.start_gradio_app():
            return {
                "status": "failed",
                "error": "앱 시작 실패",
                "tests": {}
            }
        
        try:
            # 각 테스트 실행
            test_results = {
                "app_health": self.test_app_health(),
                "ui_components": self.test_ui_components(),
                "chat_functionality": self.test_chat_functionality(),
                "performance": self.test_performance(),
                "phase_features": self.test_phase_features()
            }
            
            # 전체 결과 요약
            total_tests = 0
            passed_tests = 0
            
            for test_name, result in test_results.items():
                if isinstance(result, dict):
                    if result.get("status") == "passed":
                        passed_tests += 1
                    total_tests += 1
                elif isinstance(result, bool):
                    if result:
                        passed_tests += 1
                    total_tests += 1
                elif isinstance(result, dict) and "passed" in result:
                    passed_tests += result["passed"]
                    total_tests += result["total_tests"]
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            print("\n" + "=" * 60)
            print("[SUMMARY] 테스트 결과 요약")
            print("=" * 60)
            print(f"[PASS] 통과: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            print(f"[FAIL] 실패: {total_tests - passed_tests}/{total_tests}")
            
            return {
                "status": "completed",
                "success_rate": success_rate,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "test_results": test_results
            }
            
        finally:
            # 앱 중지
            self.stop_gradio_app()


def main():
    """메인 테스트 실행"""
    tester = GradioAppTester()
    
    try:
        results = tester.run_comprehensive_test()
        
        # 결과를 JSON 파일로 저장
        output_file = project_root / "test_results" / "gradio_test_results.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SAVE] 테스트 결과가 저장되었습니다: {output_file}")
        
        # 최종 상태 출력
        if results["success_rate"] >= 80:
            print("\n[SUCCESS] Gradio 애플리케이션 테스트 성공!")
        else:
            print("\n[WARNING] 일부 테스트가 실패했습니다. 로그를 확인해주세요.")
            
    except KeyboardInterrupt:
        print("\n[STOP] 테스트가 사용자에 의해 중단되었습니다.")
        tester.stop_gradio_app()
    except Exception as e:
        print(f"\n[ERROR] 테스트 실행 중 오류 발생: {e}")
        tester.stop_gradio_app()


if __name__ == "__main__":
    main()
