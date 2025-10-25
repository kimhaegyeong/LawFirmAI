# -*- coding: utf-8 -*-
"""
Langfuse 디버그 테스트
Langfuse 데이터 수집 문제 진단 및 해결
"""

import os
import sys
import time
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, '.')

# .env 파일 로드
load_dotenv()

print("🔍 Langfuse 디버그 테스트 시작")
print("=" * 60)

# 환경 변수 확인
def check_environment_variables():
    """환경 변수 설정 확인"""
    print("\n📋 환경 변수 확인")
    print("-" * 30)
    
    required_vars = [
        "LANGFUSE_ENABLED",
        "LANGFUSE_SECRET_KEY", 
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_HOST"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                # API 키는 일부만 표시
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 설정되지 않음")
    
    return all(os.getenv(var) for var in required_vars)

# Langfuse 패키지 설치 확인
def check_langfuse_installation():
    """Langfuse 패키지 설치 확인"""
    print("\n📦 Langfuse 패키지 확인")
    print("-" * 30)
    
    try:
        import langfuse
        # 버전 정보가 없을 수 있으므로 다른 방법으로 확인
        try:
            version = langfuse.__version__
            print(f"✅ langfuse 버전: {version}")
        except AttributeError:
            print("✅ langfuse 패키지 설치됨 (버전 정보 없음)")
        return True
    except ImportError as e:
        print(f"❌ langfuse 설치되지 않음: {e}")
        print("💡 설치 방법: pip install langfuse")
        return False
    except Exception as e:
        print(f"⚠️ langfuse 확인 중 오류: {e}")
        return False

# 기본 Langfuse 연결 테스트
def test_basic_langfuse_connection():
    """기본 Langfuse 연결 테스트"""
    print("\n🔗 기본 Langfuse 연결 테스트")
    print("-" * 30)
    
    try:
        from langfuse import Langfuse
        
        # Langfuse 클라이언트 초기화
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
        print("✅ Langfuse 클라이언트 초기화 성공")
        
        # 간단한 트레이스 생성 테스트 (올바른 API 사용)
        trace_id = langfuse.create_trace_id()
        print("✅ 트레이스 ID 생성 성공")
        print(f"   트레이스 ID: {trace_id}")
        
        # 간단한 이벤트 생성
        event = langfuse.create_event(
            name="debug_test_event",
            input="테스트 입력",
            output="테스트 출력",
            metadata={"test": True, "timestamp": time.time(), "trace_id": trace_id}
        )
        
        print("✅ 생성 이벤트 추가 성공")
        
        # 데이터 플러시
        langfuse.flush()
        print("✅ 데이터 플러시 완료")
        
        return True, trace_id
        
    except Exception as e:
        print(f"❌ Langfuse 연결 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False, None

# 프로젝트의 Langfuse 모니터 테스트
def test_project_langfuse_monitor():
    """프로젝트의 Langfuse 모니터 테스트"""
    print("\n🔧 프로젝트 Langfuse 모니터 테스트")
    print("-" * 30)
    
    try:
        from source.utils.langfuse_monitor import get_langfuse_monitor
        
        monitor = get_langfuse_monitor()
        print(f"✅ Langfuse 모니터 인스턴스 생성: {type(monitor)}")
        
        # 모니터링 활성화 상태 확인
        is_enabled = monitor.is_enabled()
        print(f"📊 모니터링 활성화 상태: {is_enabled}")
        
        if is_enabled:
            # 트레이스 생성 테스트
            trace = monitor.create_trace(
                name="project_monitor_test",
                user_id="project_test_user",
                session_id="project_test_session"
            )
            
            if trace:
                print("✅ 프로젝트 모니터 트레이스 생성 성공")
                print(f"   트레이스 객체: {trace}")
                
                # 로깅 테스트
                success = monitor.log_generation(
                    trace_id=trace.id if hasattr(trace, 'id') else str(trace),
                    name="project_test_generation",
                    input_data={"test": "project_monitor"},
                    output_data={"result": "success"},
                    metadata={"source": "project_monitor_test"}
                )
                
                if success:
                    print("✅ 프로젝트 모니터 로깅 성공")
                else:
                    print("❌ 프로젝트 모니터 로깅 실패")
                
                # 데이터 플러시
                monitor.flush()
                print("✅ 프로젝트 모니터 데이터 플러시 완료")
                
                return True, trace
            else:
                print("❌ 프로젝트 모니터 트레이스 생성 실패")
                return False, None
        else:
            print("⚠️ 프로젝트 모니터가 비활성화됨")
            return False, None
            
    except Exception as e:
        print(f"❌ 프로젝트 모니터 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False, None

# Enhanced Chat Service와 함께 테스트
async def test_enhanced_chat_service():
    """Enhanced Chat Service와 함께 테스트"""
    print("\n💬 Enhanced Chat Service 테스트")
    print("-" * 30)
    
    try:
        from source.utils.config import Config
        from source.services.enhanced_chat_service import EnhancedChatService
        
        # 설정 로드
        config = Config()
        print("✅ Config 로드 성공")
        
        # Enhanced Chat Service 초기화
        chat_service = EnhancedChatService(config)
        print("✅ Enhanced Chat Service 초기화 성공")
        
        # 간단한 메시지 처리 테스트
        test_message = "안녕하세요. 간단한 법률 질문이 있습니다."
        
        print(f"📝 테스트 메시지: {test_message}")
        
        result = await chat_service.process_message(
            message=test_message,
            user_id="langfuse_debug_user",
            session_id="langfuse_debug_session"
        )
        
        print("✅ 메시지 처리 완료")
        print(f"   응답 길이: {len(result.get('response', ''))}")
        print(f"   신뢰도: {result.get('confidence', 0.0):.2f}")
        print(f"   처리 시간: {result.get('processing_time', 0.0):.3f}초")
        print(f"   생성 방법: {result.get('generation_method', 'unknown')}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ Enhanced Chat Service 테스트 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return False, None

# 네트워크 연결 테스트
def test_network_connection():
    """네트워크 연결 테스트"""
    print("\n🌐 네트워크 연결 테스트")
    print("-" * 30)
    
    try:
        import requests
        
        # Langfuse 호스트 연결 테스트
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        try:
            response = requests.get(host, timeout=10)
            print(f"✅ Langfuse 호스트 연결 성공: {host}")
            print(f"   상태 코드: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Langfuse 호스트 연결 실패: {e}")
            return False
        
        # 인터넷 연결 테스트
        try:
            response = requests.get("https://www.google.com", timeout=5)
            print(f"✅ 인터넷 연결 성공")
        except requests.exceptions.RequestException as e:
            print(f"❌ 인터넷 연결 실패: {e}")
            return False
        
        return True
        
    except ImportError:
        print("⚠️ requests 패키지가 설치되지 않음")
        return False
    except Exception as e:
        print(f"❌ 네트워크 테스트 실패: {e}")
        return False

# 메인 테스트 함수
async def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 Langfuse 디버그 테스트 시작")
    print("=" * 60)
    
    results = {}
    
    # 1. 환경 변수 확인
    results['env_vars'] = check_environment_variables()
    
    # 2. Langfuse 패키지 확인
    results['langfuse_package'] = check_langfuse_installation()
    
    # 3. 네트워크 연결 테스트
    results['network'] = test_network_connection()
    
    # 4. 기본 Langfuse 연결 테스트
    if results['env_vars'] and results['langfuse_package'] and results['network']:
        success, trace_id = test_basic_langfuse_connection()
        results['basic_connection'] = success
        results['trace_id'] = trace_id
    else:
        print("\n⚠️ 기본 연결 테스트를 건너뜀 (필수 조건 미충족)")
        results['basic_connection'] = False
    
    # 5. 프로젝트 모니터 테스트
    if results['basic_connection']:
        success, trace = test_project_langfuse_monitor()
        results['project_monitor'] = success
        results['project_trace'] = trace
    else:
        print("\n⚠️ 프로젝트 모니터 테스트를 건너뜀 (기본 연결 실패)")
        results['project_monitor'] = False
    
    # 6. Enhanced Chat Service 테스트
    if results['project_monitor']:
        success, result = await test_enhanced_chat_service()
        results['chat_service'] = success
        results['chat_result'] = result
    else:
        print("\n⚠️ Enhanced Chat Service 테스트를 건너뜀 (프로젝트 모니터 실패)")
        results['chat_service'] = False
    
    # 결과 요약
    print("\n📊 테스트 결과 요약")
    print("=" * 60)
    
    total_tests = len([k for k in results.keys() if k in ['env_vars', 'langfuse_package', 'network', 'basic_connection', 'project_monitor', 'chat_service']])
    passed_tests = sum(1 for k in ['env_vars', 'langfuse_package', 'network', 'basic_connection', 'project_monitor', 'chat_service'] if results.get(k, False))
    
    print(f"총 테스트: {total_tests}")
    print(f"통과한 테스트: {passed_tests}")
    print(f"실패한 테스트: {total_tests - passed_tests}")
    
    print("\n상세 결과:")
    for test_name, result in results.items():
        if test_name in ['env_vars', 'langfuse_package', 'network', 'basic_connection', 'project_monitor', 'chat_service']:
            status = "✅ 통과" if result else "❌ 실패"
            print(f"  {test_name}: {status}")
    
    # 권장사항
    print("\n💡 권장사항:")
    if not results['env_vars']:
        print("  - .env 파일의 Langfuse 환경 변수를 확인하세요")
    if not results['langfuse_package']:
        print("  - pip install langfuse 명령으로 패키지를 설치하세요")
    if not results['network']:
        print("  - 인터넷 연결을 확인하세요")
    if not results['basic_connection']:
        print("  - Langfuse API 키가 올바른지 확인하세요")
    if not results['project_monitor']:
        print("  - 프로젝트의 Langfuse 모니터 설정을 확인하세요")
    if not results['chat_service']:
        print("  - Enhanced Chat Service의 Langfuse 통합을 확인하세요")
    
    if passed_tests == total_tests:
        print("\n🎉 모든 테스트가 통과했습니다! Langfuse가 정상적으로 작동합니다.")
        print("📊 Langfuse 대시보드에서 데이터를 확인하세요.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests}개의 테스트가 실패했습니다. 위의 권장사항을 확인하세요.")
    
    return results

if __name__ == "__main__":
    print("🔍 Langfuse 디버그 테스트")
    print("=" * 80)
    
    # 모든 테스트 실행
    results = asyncio.run(run_all_tests())
    
    print("\n🏁 Langfuse 디버그 테스트 완료!")
