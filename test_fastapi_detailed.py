#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 상세 테스트 스크립트
"""

import requests
import json
import time
from typing import Dict, Any

# API 기본 설정
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_root_endpoint():
    """루트 엔드포인트 테스트"""
    print("루트 엔드포인트 테스트")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"서버 상태: {data.get('status', 'unknown')}")
            print(f"버전: {data.get('version', 'unknown')}")
            print(f"메시지: {data.get('message', 'unknown')}")
            return True
        else:
            print(f"루트 엔드포인트 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"루트 엔드포인트 오류: {e}")
        return False

def test_health_endpoint():
    """헬스 체크 엔드포인트 테스트"""
    print("\n헬스 체크 엔드포인트 테스트")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"상태: {data.get('status', 'unknown')}")
            print(f"타임스탬프: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"헬스 체크 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"헬스 체크 오류: {e}")
        return False

def test_system_status():
    """시스템 상태 확인 테스트"""
    print("\n시스템 상태 확인 테스트")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/system/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"시스템 상태: {data.get('status', 'unknown')}")
            print(f"데이터베이스 상태: {data.get('database_status', 'unknown')}")
            print(f"총 법률 조문 수: {data.get('total_articles', 0):,}")
            print(f"버전: {data.get('version', 'unknown')}")
            return True
        else:
            print(f"시스템 상태 확인 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"시스템 상태 확인 오류: {e}")
        return False

def test_chat_endpoint():
    """채팅 엔드포인트 테스트"""
    print("\n채팅 엔드포인트 테스트")
    print("=" * 40)
    
    test_cases = [
        {
            "message": "손해배상 청구 방법",
            "context": "민법 관련 질문"
        },
        {
            "message": "계약 해제 조건",
            "context": "상법 관련"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {test_case['message']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/chat",
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
                print(f"처리 시간: {data.get('processing_time', 0):.2f}초")
                print(f"답변 미리보기: {data.get('response', '')[:100]}...")
                success_count += 1
            else:
                print(f"응답 실패: {response.status_code}")
                print(f"오류 내용: {response.text}")
                
        except Exception as e:
            print(f"요청 오류: {e}")
    
    print(f"\n채팅 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def test_search_endpoint():
    """검색 엔드포인트 테스트"""
    print("\n검색 엔드포인트 테스트")
    print("=" * 40)
    
    test_queries = [
        "손해배상",
        "계약",
        "민법"
    ]
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n검색 쿼리 {i}: {query}")
        
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/search",
                params={"q": query, "limit": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"검색 성공")
                print(f"검색 쿼리: {data.get('query', 'unknown')}")
                print(f"결과 수: {data.get('total_count', 0)}")
                print(f"처리 시간: {data.get('processing_time', 0):.2f}초")
                
                results = data.get('results', [])
                for j, result in enumerate(results, 1):
                    print(f"  {j}. {result.get('law_name', 'unknown')} {result.get('article_number', 'unknown')}")
                    print(f"     내용: {result.get('content', '')[:100]}...")
                
                success_count += 1
            else:
                print(f"검색 실패: {response.status_code}")
                print(f"오류 내용: {response.text}")
                
        except Exception as e:
            print(f"검색 요청 오류: {e}")
    
    print(f"\n검색 테스트 결과: {success_count}/{len(test_queries)} 성공")
    return success_count == len(test_queries)

def test_api_docs():
    """API 문서 접근 테스트"""
    print("\nAPI 문서 접근 테스트")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=10)
        if response.status_code == 200:
            print("API 문서 접근 성공")
            print("Swagger UI가 정상적으로 로드되었습니다")
            return True
        else:
            print(f"API 문서 접근 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"API 문서 접근 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LawFirmAI FastAPI 상세 테스트 시작")
    print("=" * 60)
    
    # API 서버가 실행 중인지 확인
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("API 서버가 실행되지 않았습니다.")
            print("다음 명령어로 서버를 시작하세요:")
            print("python simple_fastapi_server.py")
            return False
    except:
        print("API 서버에 연결할 수 없습니다.")
        print("다음 명령어로 서버를 시작하세요:")
        print("python simple_fastapi_server.py")
        return False
    
    # 테스트 실행
    tests = [
        ("루트 엔드포인트", test_root_endpoint),
        ("헬스 체크", test_health_endpoint),
        ("시스템 상태 확인", test_system_status),
        ("채팅 엔드포인트", test_chat_endpoint),
        ("검색 엔드포인트", test_search_endpoint),
        ("API 문서 접근", test_api_docs)
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
        print("모든 FastAPI 테스트가 성공적으로 완료되었습니다!")
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인하세요.")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
