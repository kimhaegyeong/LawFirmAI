#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 앱 직접 테스트 (내장 API 사용)
"""

import requests
import json
import time
import sys
import os

def test_gradio_direct():
    """Gradio 앱 직접 테스트"""
    print("=== Gradio 앱 직접 테스트 ===")
    
    base_url = "http://localhost:7861"
    
    # 테스트 쿼리들
    test_queries = [
        "안녕하세요",
        "계약서 검토에 대해 알려주세요",
        "손해배상 청구 방법은?",
        "민법 제750조에 대해 설명해주세요"
    ]
    
    print(f"테스트 URL: {base_url}")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"테스트 {i}: {query}")
        
        try:
            # Gradio 내장 API를 통한 채팅 테스트
            # /respond 엔드포인트 사용
            response = requests.post(
                f"{base_url}/gradio_api/respond",
                json={
                    "data": [query, []],  # [message, history]
                    "fn_index": 0
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  성공: HTTP {response.status_code}")
                
                if "data" in result and len(result["data"]) > 0:
                    response_data = result["data"][0]
                    if isinstance(response_data, list) and len(response_data) > 0:
                        # 채팅 응답에서 마지막 메시지 추출
                        last_message = response_data[-1]
                        if isinstance(last_message, dict) and "content" in last_message:
                            content = last_message["content"]
                            print(f"  응답: {content[:100]}...")
                        else:
                            print(f"  응답 데이터: {str(response_data)[:100]}...")
                    else:
                        print(f"  응답: {str(response_data)[:100]}...")
                else:
                    print(f"  응답 구조: {result}")
                    
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
    test_gradio_direct()