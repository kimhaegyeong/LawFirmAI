# -*- coding: utf-8 -*-
"""
LawFirmAI Gradio 채팅 테스트
실제 Gradio 인터페이스를 통한 채팅 기능 테스트
"""

import unittest
import time
import requests
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestGradioChat(unittest.TestCase):
    """Gradio 채팅 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.base_url = "http://localhost:7860"
        self.api_url = f"{self.base_url}/api"
        self.chat_url = f"{self.api_url}/chat"
        
        # 테스트 데이터
        self.test_queries = [
            "안녕하세요",
            "손해배상에 대해 알려주세요",
            "계약 해지 절차는 어떻게 되나요?",
            "민법 제750조의 요건을 알려주세요",
            "불법행위의 성립요건은 무엇인가요?"
        ]
        
        # 대화 시나리오
        self.conversation_scenarios = [
            {
                "name": "기본 인사",
                "queries": ["안녕하세요", "도움이 필요합니다"]
            },
            {
                "name": "법률 상담",
                "queries": ["손해배상에 대해 알려주세요", "그것의 요건은 무엇인가요?", "위의 사건에서 과실비율은 어떻게 정해지나요?"]
            },
            {
                "name": "계약 관련",
                "queries": ["계약 해지 절차는 어떻게 되나요?", "계약서 검토를 도와주실 수 있나요?", "그 계약서에서 문제가 될 수 있는 부분이 있나요?"]
            }
        ]
        
    def test_gradio_server_status(self):
        """Gradio 서버 상태 확인"""
        print("\n=== Gradio 서버 상태 확인 ===")
        
        try:
            # 서버 상태 확인
            response = requests.get(self.base_url, timeout=10)
            self.assertEqual(response.status_code, 200)
            print(f"✅ Gradio 서버 정상 작동: {response.status_code}")
            
            # API 엔드포인트 확인
            try:
                api_response = requests.get(f"{self.api_url}/health", timeout=5)
                print(f"✅ API 엔드포인트 접근 가능: {api_response.status_code}")
            except requests.exceptions.RequestException:
                print("⚠️ API 엔드포인트 접근 불가 (정상일 수 있음)")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Gradio 서버 접근 실패: {e}")
            self.fail("Gradio 서버가 실행되지 않았습니다. 먼저 'python gradio/app.py'를 실행해주세요.")
    
    def test_single_chat_queries(self):
        """단일 채팅 질문 테스트"""
        print("\n=== 단일 채팅 질문 테스트 ===")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n질문 {i}: {query}")
            
            try:
                # 채팅 요청
                payload = {
                    "message": query,
                    "context": None,
                    "user_id": f"test_user_{i}",
                    "session_id": f"test_session_{i}"
                }
                
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=30
                )
                
                self.assertEqual(response.status_code, 200)
                
                result = response.json()
                self.assertIn("response", result)
                self.assertIn("confidence", result)
                
                print(f"  응답: {result['response'][:100]}...")
                print(f"  신뢰도: {result['confidence']:.2f}")
                
                # 응답 품질 검증
                self.assertGreater(len(result['response']), 10)
                self.assertGreaterEqual(result['confidence'], 0.0)
                self.assertLessEqual(result['confidence'], 1.0)
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 요청 실패: {e}")
                self.fail(f"채팅 요청 실패: {e}")
            except Exception as e:
                print(f"  ❌ 예상치 못한 오류: {e}")
                self.fail(f"예상치 못한 오류: {e}")
    
    def test_conversation_scenarios(self):
        """대화 시나리오 테스트"""
        print("\n=== 대화 시나리오 테스트 ===")
        
        for scenario in self.conversation_scenarios:
            print(f"\n시나리오: {scenario['name']}")
            
            session_id = f"scenario_{scenario['name']}"
            user_id = f"scenario_user_{scenario['name']}"
            
            for i, query in enumerate(scenario['queries'], 1):
                print(f"  턴 {i}: {query}")
                
                try:
                    payload = {
                        "message": query,
                        "context": None,
                        "user_id": user_id,
                        "session_id": session_id
                    }
                    
                    response = requests.post(
                        self.chat_url,
                        json=payload,
                        timeout=30
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    
                    result = response.json()
                    print(f"    응답: {result['response'][:80]}...")
                    print(f"    신뢰도: {result['confidence']:.2f}")
                    
                    # 대화 맥락 검증
                    if i > 1:  # 두 번째 턴부터는 맥락이 있어야 함
                        self.assertIn("context", result)
                        if result.get("context"):
                            print(f"    맥락: {result['context'][:50]}...")
                    
                except requests.exceptions.RequestException as e:
                    print(f"    ❌ 요청 실패: {e}")
                    self.fail(f"시나리오 '{scenario['name']}' 실패: {e}")
                except Exception as e:
                    print(f"    ❌ 예상치 못한 오류: {e}")
                    self.fail(f"시나리오 '{scenario['name']}' 오류: {e}")
    
    def test_chat_with_context(self):
        """맥락이 있는 채팅 테스트"""
        print("\n=== 맥락이 있는 채팅 테스트 ===")
        
        # 첫 번째 질문
        first_query = "손해배상에 대해 알려주세요"
        print(f"첫 번째 질문: {first_query}")
        
        try:
            payload = {
                "message": first_query,
                "context": None,
                "user_id": "context_test_user",
                "session_id": "context_test_session"
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=30)
            self.assertEqual(response.status_code, 200)
            
            first_result = response.json()
            print(f"첫 번째 응답: {first_result['response'][:100]}...")
            
            # 두 번째 질문 (맥락 활용)
            second_query = "그것의 요건은 무엇인가요?"
            print(f"\n두 번째 질문: {second_query}")
            
            payload = {
                "message": second_query,
                "context": first_result.get("context", ""),
                "user_id": "context_test_user",
                "session_id": "context_test_session"
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=30)
            self.assertEqual(response.status_code, 200)
            
            second_result = response.json()
            print(f"두 번째 응답: {second_result['response'][:100]}...")
            
            # 맥락 활용 검증
            self.assertIn("손해배상", second_result['response'])
            print("✅ 맥락이 올바르게 활용되었습니다")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 실패: {e}")
            self.fail(f"맥락 채팅 테스트 실패: {e}")
        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
            self.fail(f"맥락 채팅 테스트 오류: {e}")
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        print("\n=== 오류 처리 테스트 ===")
        
        error_scenarios = [
            {
                "name": "빈 메시지",
                "payload": {"message": "", "user_id": "error_test_user", "session_id": "error_test_session"}
            },
            {
                "name": "매우 긴 메시지",
                "payload": {"message": "x" * 10000, "user_id": "error_test_user", "session_id": "error_test_session"}
            },
            {
                "name": "특수 문자 포함",
                "payload": {"message": "!@#$%^&*()_+{}|:<>?[]\\;'\",./", "user_id": "error_test_user", "session_id": "error_test_session"}
            }
        ]
        
        for scenario in error_scenarios:
            print(f"\n오류 시나리오: {scenario['name']}")
            
            try:
                response = requests.post(self.chat_url, json=scenario['payload'], timeout=30)
                
                # 오류 상황에서도 적절한 응답이 있어야 함
                self.assertEqual(response.status_code, 200)
                
                result = response.json()
                self.assertIn("response", result)
                
                print(f"  응답: {result['response'][:50]}...")
                print(f"  신뢰도: {result['confidence']:.2f}")
                
                # 오류 상황에서도 응답이 있어야 함
                self.assertGreater(len(result['response']), 0)
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 요청 실패: {e}")
                # 오류 상황에서는 요청 실패도 허용
                pass
            except Exception as e:
                print(f"  ❌ 예상치 못한 오류: {e}")
                # 오류 상황에서는 예외도 허용
                pass
    
    def test_performance_metrics(self):
        """성능 메트릭 테스트"""
        print("\n=== 성능 메트릭 테스트 ===")
        
        response_times = []
        
        for i, query in enumerate(self.test_queries[:3], 1):  # 처음 3개만 테스트
            print(f"성능 테스트 {i}: {query}")
            
            start_time = time.time()
            
            try:
                payload = {
                    "message": query,
                    "context": None,
                    "user_id": f"perf_test_user_{i}",
                    "session_id": f"perf_test_session_{i}"
                }
                
                response = requests.post(self.chat_url, json=payload, timeout=30)
                self.assertEqual(response.status_code, 200)
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                print(f"  응답 시간: {response_time:.3f}초")
                
                # 성능 기준 검증
                self.assertLess(response_time, 10.0)  # 10초 이내
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 요청 실패: {e}")
                self.fail(f"성능 테스트 실패: {e}")
            except Exception as e:
                print(f"  ❌ 예상치 못한 오류: {e}")
                self.fail(f"성능 테스트 오류: {e}")
        
        # 평균 응답 시간 계산
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"\n평균 응답 시간: {avg_response_time:.3f}초")
            
            # 성능 기준 검증
            self.assertLess(avg_response_time, 5.0)  # 평균 5초 이내
            print("✅ 성능 기준 통과")
    
    def test_concurrent_requests(self):
        """동시 요청 테스트"""
        print("\n=== 동시 요청 테스트 ===")
        
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def send_request(query, user_id, session_id):
            try:
                payload = {
                    "message": query,
                    "context": None,
                    "user_id": user_id,
                    "session_id": session_id
                }
                
                start_time = time.time()
                response = requests.post(self.chat_url, json=payload, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    results.put({
                        "query": query,
                        "response_time": end_time - start_time,
                        "confidence": result.get("confidence", 0)
                    })
                else:
                    errors.put(f"HTTP {response.status_code}: {query}")
                    
            except Exception as e:
                errors.put(f"Error: {query} - {e}")
        
        # 동시 요청 생성
        threads = []
        for i in range(5):
            query = f"동시 테스트 질문 {i+1}"
            user_id = f"concurrent_user_{i+1}"
            session_id = f"concurrent_session_{i+1}"
            
            thread = threading.Thread(target=send_request, args=(query, user_id, session_id))
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 결과 분석
        successful_requests = []
        while not results.empty():
            successful_requests.append(results.get())
        
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())
        
        print(f"성공한 요청: {len(successful_requests)}/5")
        print(f"실패한 요청: {len(error_list)}/5")
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            print(f"평균 응답 시간: {avg_response_time:.3f}초")
            
            # 동시성 테스트 기준
            self.assertGreaterEqual(len(successful_requests), 3)  # 최소 3개 성공
            self.assertLess(avg_response_time, 15.0)  # 평균 15초 이내
        
        if error_list:
            print("오류 목록:")
            for error in error_list:
                print(f"  - {error}")


def run_gradio_chat_tests():
    """Gradio 채팅 테스트 실행"""
    print("=" * 60)
    print("LawFirmAI Gradio 채팅 테스트 시작")
    print("=" * 60)
    
    # Gradio 서버 상태 확인
    try:
        response = requests.get("http://localhost:7860", timeout=5)
        print("✅ Gradio 서버가 실행 중입니다")
    except requests.exceptions.RequestException:
        print("❌ Gradio 서버가 실행되지 않았습니다.")
        print("먼저 다음 명령어로 Gradio 서버를 시작해주세요:")
        print("python gradio/app.py")
        return False
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGradioChat))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("Gradio 채팅 테스트 결과 요약")
    print("=" * 60)
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")
    
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n전체 통과율: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 우수한 채팅 성능! Gradio 인터페이스가 안정적으로 작동합니다.")
    elif success_rate >= 75:
        print("✅ 양호한 채팅 성능! 일부 개선이 필요합니다.")
    else:
        print("⚠️ 채팅 기능 개선이 필요합니다. 시스템을 점검해주세요.")
    
    return result


if __name__ == "__main__":
    run_gradio_chat_tests()
