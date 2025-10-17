#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 API 테스트 스크립트 (유니코드 문제 해결)
"""

import requests
import json
import time
from typing import Dict, Any

# API 기본 설정
BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}

def test_system_status():
    """시스템 상태 확인 테스트"""
    print("=" * 60)
    print("시스템 상태 확인 테스트")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/system/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"시스템 상태: {data.get('overall_status', 'unknown')}")
            print(f"버전: {data.get('version', 'unknown')}")
            
            components = data.get('components', {})
            for comp_name, comp_data in components.items():
                status = comp_data.get('status', 'unknown')
                print(f"  - {comp_name}: {status}")
            
            return True
        else:
            print(f"시스템 상태 확인 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"시스템 상태 확인 오류: {e}")
        return False

def test_basic_chat():
    """기본 채팅 API 테스트"""
    print("\n" + "=" * 60)
    print("기본 채팅 API 테스트")
    print("=" * 60)
    
    test_case = {
        "message": "손해배상 청구 방법",
        "context": "민법 관련 질문"
    }
    
    print(f"테스트 질문: {test_case['message']}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            headers=HEADERS,
            json=test_case,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"응답 성공")
            print(f"답변 길이: {len(data.get('response', ''))} 문자")
            print(f"신뢰도: {data.get('confidence', 0):.2f}")
            print(f"소스 수: {len(data.get('sources', []))}")
            print(f"답변 미리보기: {data.get('response', '')[:100]}...")
            return True
        else:
            print(f"응답 실패: {response.status_code}")
            print(f"오류 내용: {response.text}")
            return False
            
    except Exception as e:
        print(f"요청 오류: {e}")
        return False

def test_intelligent_chat_v2():
    """지능형 채팅 API v2 테스트"""
    print("\n" + "=" * 60)
    print("지능형 채팅 API v2 테스트")
    print("=" * 60)
    
    test_case = {
        "message": "손해배상 관련 판례를 찾아주세요",
        "session_id": "test_session_1",
        "max_results": 5,
        "include_law_sources": True,
        "include_precedent_sources": True,
        "include_conversation_history": True,
        "context_optimization": True,
        "answer_formatting": True
    }
    
    print(f"테스트 질문: {test_case['message']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat/intelligent-v2",
            headers=HEADERS,
            json=test_case,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"응답 성공 (처리 시간: {end_time - start_time:.2f}초)")
            print(f"질문 유형: {data.get('question_type', 'unknown')}")
            print(f"답변 길이: {len(data.get('answer', ''))} 문자")
            
            confidence = data.get('confidence', {})
            print(f"신뢰도: {confidence.get('confidence', 0):.2f}")
            print(f"신뢰도 수준: {confidence.get('reliability_level', 'unknown')}")
            
            search_stats = data.get('search_stats', {})
            print(f"검색 결과: {search_stats.get('total_results', 0)}개")
            print(f"법률 결과: {search_stats.get('law_results_count', 0)}개")
            print(f"판례 결과: {search_stats.get('precedent_results_count', 0)}개")
            
            print(f"답변 미리보기: {data.get('answer', '')[:200]}...")
            return True
        else:
            print(f"응답 실패: {response.status_code}")
            print(f"오류 내용: {response.text}")
            return False
            
    except Exception as e:
        print(f"요청 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LawFirmAI API 테스트 시작")
    print("=" * 60)
    
    # API 서버가 실행 중인지 확인
    try:
        response = requests.get(f"{BASE_URL}/system/status", timeout=5)
        if response.status_code != 200:
            print("API 서버가 실행되지 않았습니다.")
            print("다음 명령어로 서버를 시작하세요:")
            print("cd api && python main.py")
            return False
    except:
        print("API 서버에 연결할 수 없습니다.")
        print("다음 명령어로 서버를 시작하세요:")
        print("cd api && python main.py")
        return False
    
    # 테스트 실행
    tests = [
        ("시스템 상태 확인", test_system_status),
        ("기본 채팅 API", test_basic_chat),
        ("지능형 채팅 API v2", test_intelligent_chat_v2)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 테스트 중 오류 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "통과" if result else "실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체 결과: {passed}/{len(results)} 테스트 통과")
    
    if passed == len(results):
        print("모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인하세요.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()