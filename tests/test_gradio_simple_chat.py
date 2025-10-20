#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 앱 간단한 채팅 테스트
"""

import requests
import json
import time
import sys
import os

def test_gradio_chat():
    """Gradio 앱 채팅 기능 테스트"""
    print("=== Gradio 앱 채팅 테스트 ===")
    
    base_url = "http://localhost:7861"
    
    # 테스트 쿼리들
    test_queries = [
        "안녕하세요",
        "계약서 검토에 대해 알려주세요",
        "손해배상 청구 방법은?",
        "민법 제750조에 대해 설명해주세요"
    ]
    
    session_id = f"test_session_{int(time.time())}"
    user_id = "test_user_001"
    
    print(f"세션 ID: {session_id}")
    print(f"사용자 ID: {user_id}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"테스트 {i}: {query}")
        
        try:
            # Gradio API를 통한 채팅 테스트
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "message": query,
                    "session_id": session_id,
                    "user_id": user_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  성공: {result.get('response', '')[:100]}...")
                
                # 추가 정보 출력
                if 'confidence' in result:
                    print(f"  신뢰도: {result['confidence']:.2f}")
                if 'sources' in result:
                    print(f"  소스 수: {len(result['sources'])}")
                    
            else:
                print(f"  실패: HTTP {response.status_code}")
                print(f"  오류: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  연결 오류: {e}")
        except Exception as e:
            print(f"  예상치 못한 오류: {e}")
        
        print()
        time.sleep(1)  # 요청 간 간격
    
    print("=== 테스트 완료 ===")

if __name__ == "__main__":
    test_gradio_chat()