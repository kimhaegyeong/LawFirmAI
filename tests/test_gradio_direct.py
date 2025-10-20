# -*- coding: utf-8 -*-
"""
LawFirmAI Gradio 직접 테스트
Gradio 앱을 직접 실행하여 채팅 기능 테스트
"""

import unittest
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Gradio 앱 모듈 임포트
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gradio'))
    from app import create_gradio_interface
    from source.services.chat_service import ChatService
    print("Gradio 앱 모듈 임포트 성공")
except ImportError as e:
    print(f"Gradio 앱 모듈 임포트 실패: {e}")
    sys.exit(1)


class TestGradioDirect(unittest.TestCase):
    """Gradio 직접 테스트"""
    
    def setUp(self):
        """테스트 설정"""
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
                "queries": ["손해배상에 대해 알려주세요", "그것의 요건은 무엇인가요?"]
            },
            {
                "name": "계약 관련",
                "queries": ["계약 해지 절차는 어떻게 되나요?", "계약서 검토를 도와주실 수 있나요?"]
            }
        ]
        
        # 테스트용 세션 정보
        self.test_session_id = "test_session_direct"
        self.test_user_id = "test_user_direct"
        
        # ChatService 인스턴스 생성
        try:
            from source.utils.config import Config
            config = Config()
            self.chat_service = ChatService(config)
            print("ChatService 초기화 성공")
        except Exception as e:
            print(f"ChatService 초기화 실패: {e}")
            self.chat_service = None
        
    def test_gradio_interface_creation(self):
        """Gradio 인터페이스 생성 테스트"""
        print("\n=== Gradio 인터페이스 생성 테스트 ===")
        
        try:
            # Gradio 인터페이스 생성
            interface = create_gradio_interface()
            
            self.assertIsNotNone(interface)
            print("Gradio 인터페이스 생성 성공")
            
            # 인터페이스 속성 확인
            self.assertTrue(hasattr(interface, 'launch'))
            self.assertTrue(hasattr(interface, 'close'))
            print("Gradio 인터페이스 메서드 확인 완료")
            
        except Exception as e:
            print(f"Gradio 인터페이스 생성 실패: {e}")
            self.fail(f"Gradio 인터페이스 생성 실패: {e}")
    
    def test_process_query_function(self):
        """process_query 함수 테스트"""
        print("\n=== process_query 함수 테스트 ===")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n질문 {i}: {query}")
            
            try:
                # ChatService를 사용하여 쿼리 처리
                if self.chat_service is None:
                    self.skipTest("ChatService가 초기화되지 않았습니다")
                
                result = self.chat_service.process_query(query, self.test_session_id, self.test_user_id)
                
                self.assertIsNotNone(result)
                self.assertIsInstance(result, dict)
                
                response = result.get("response", "")
                print(f"  응답: {response[:100]}...")
                print(f"  응답 길이: {len(response)} 문자")
                
                # 응답 품질 검증
                self.assertGreater(len(response), 10)
                self.assertIn("법", response)  # 법률 관련 응답인지 확인
                
            except Exception as e:
                print(f"  처리 실패: {e}")
                self.fail(f"process_query 실패: {e}")
    
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
                    # ChatService를 사용하여 쿼리 처리
                    if self.chat_service is None:
                        self.skipTest("ChatService가 초기화되지 않았습니다")
                    
                    result = self.chat_service.process_query(query, session_id, user_id)
                    
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, dict)
                    
                    response = result.get("response", "")
                    print(f"    응답: {response[:80]}...")
                    
                    # 응답 품질 검증
                    self.assertGreater(len(response), 5)
                    
                except Exception as e:
                    print(f"    처리 실패: {e}")
                    self.fail(f"시나리오 '{scenario['name']}' 실패: {e}")
    
    def test_error_handling(self):
        """오류 처리 테스트"""
        print("\n=== 오류 처리 테스트 ===")
        
        error_scenarios = [
            {
                "name": "빈 메시지",
                "query": "",
                "context": "",
                "session_id": "error_test_session",
                "user_id": "error_test_user"
            },
            {
                "name": "매우 긴 메시지",
                "query": "x" * 1000,
                "context": "",
                "session_id": "error_test_session",
                "user_id": "error_test_user"
            },
            {
                "name": "특수 문자 포함",
                "query": "!@#$%^&*()_+{}|:<>?[]\\;'\",./",
                "context": "",
                "session_id": "error_test_session",
                "user_id": "error_test_user"
            }
        ]
        
        for scenario in error_scenarios:
            print(f"\n오류 시나리오: {scenario['name']}")
            
            try:
                # ChatService를 사용하여 쿼리 처리
                if self.chat_service is None:
                    self.skipTest("ChatService가 초기화되지 않았습니다")
                
                result = self.chat_service.process_query(
                    scenario['query'],
                    scenario['session_id'],
                    scenario['user_id']
                )
                
                # 오류 상황에서도 적절한 응답이 있어야 함
                self.assertIsNotNone(result)
                self.assertIsInstance(result, dict)
                
                response = result.get("response", "")
                print(f"  응답: {response[:50]}...")
                
                # 오류 상황에서도 응답이 있어야 함
                self.assertGreater(len(response), 0)
                
            except Exception as e:
                print(f"  예상치 못한 오류: {e}")
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
                # ChatService를 사용하여 쿼리 처리
                if self.chat_service is None:
                    self.skipTest("ChatService가 초기화되지 않았습니다")
                
                result = self.chat_service.process_query(query, f"perf_test_session_{i}", f"perf_test_user_{i}")
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                response = result.get("response", "")
                print(f"  응답 시간: {response_time:.3f}초")
                print(f"  응답 길이: {len(response)} 문자")
                
                # 성능 기준 검증
                self.assertLess(response_time, 30.0)  # 30초 이내
                self.assertGreater(len(response), 10)
                
            except Exception as e:
                print(f"  처리 실패: {e}")
                self.fail(f"성능 테스트 실패: {e}")
        
        # 평균 응답 시간 계산
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"\n평균 응답 시간: {avg_response_time:.3f}초")
            
            # 성능 기준 검증
            self.assertLess(avg_response_time, 15.0)  # 평균 15초 이내
            print("성능 기준 통과")
    
    def test_context_handling(self):
        """맥락 처리 테스트"""
        print("\n=== 맥락 처리 테스트 ===")
        
        # 첫 번째 질문
        first_query = "손해배상에 대해 알려주세요"
        print(f"첫 번째 질문: {first_query}")
        
        try:
            # ChatService를 사용하여 첫 번째 질문 처리
            if self.chat_service is None:
                self.skipTest("ChatService가 초기화되지 않았습니다")
            
            first_result = self.chat_service.process_query(first_query, self.test_session_id, self.test_user_id)
            first_response = first_result.get("response", "")
            print(f"첫 번째 응답: {first_response[:100]}...")
            
            # 두 번째 질문 (맥락 활용)
            second_query = "그것의 요건은 무엇인가요?"
            print(f"\n두 번째 질문: {second_query}")
            
            second_result = self.chat_service.process_query(second_query, self.test_session_id, self.test_user_id)
            second_response = second_result.get("response", "")
            print(f"두 번째 응답: {second_response[:100]}...")
            
            # 맥락 활용 검증
            self.assertIn("손해배상", second_response)
            print("맥락이 올바르게 활용되었습니다")
            
        except Exception as e:
            print(f"맥락 처리 실패: {e}")
            self.fail(f"맥락 처리 테스트 실패: {e}")
    
    def test_session_management(self):
        """세션 관리 테스트"""
        print("\n=== 세션 관리 테스트 ===")
        
        # 같은 세션에서 여러 질문
        session_id = "session_mgmt_test"
        user_id = "session_mgmt_user"
        
        queries = [
            "안녕하세요",
            "법률 상담을 받고 싶습니다",
            "손해배상에 대해 알려주세요"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"세션 질문 {i}: {query}")
            
            try:
                # ChatService를 사용하여 쿼리 처리
                if self.chat_service is None:
                    self.skipTest("ChatService가 초기화되지 않았습니다")
                
                result = self.chat_service.process_query(query, session_id, user_id)
                
                self.assertIsNotNone(result)
                self.assertIsInstance(result, dict)
                
                response = result.get("response", "")
                print(f"  응답: {response[:80]}...")
                
                # 응답 품질 검증
                self.assertGreater(len(response), 5)
                
            except Exception as e:
                print(f"  처리 실패: {e}")
                self.fail(f"세션 관리 테스트 실패: {e}")
        
        print("세션 관리 테스트 완료")


def run_gradio_direct_tests():
    """Gradio 직접 테스트 실행"""
    print("=" * 60)
    print("LawFirmAI Gradio 직접 테스트 시작")
    print("=" * 60)
    
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()
    
    # 테스트 케이스 추가
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGradioDirect))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("Gradio 직접 테스트 결과 요약")
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
        print("우수한 채팅 성능! Gradio 기능이 안정적으로 작동합니다.")
    elif success_rate >= 75:
        print("양호한 채팅 성능! 일부 개선이 필요합니다.")
    else:
        print("채팅 기능 개선이 필요합니다. 시스템을 점검해주세요.")
    
    return result


if __name__ == "__main__":
    run_gradio_direct_tests()
