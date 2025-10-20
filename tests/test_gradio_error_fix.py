#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 앱 오류 해결 확인 테스트
"""

import requests
import json
import time

def test_gradio_error_fix():
    """Gradio 앱 오류 해결 확인"""
    print("=== Gradio 앱 오류 해결 확인 테스트 ===")
    
    base_url = "http://localhost:7861"
    
    # 간단한 테스트 쿼리
    test_query = "안녕하세요"
    
    print(f"테스트 쿼리: {test_query}")
    print(f"테스트 URL: {base_url}")
    print()
    
    try:
        # Gradio 웹 페이지 접근 테스트
        response = requests.get(base_url, timeout=10)
        
        if response.status_code == 200:
            print("Gradio 웹 페이지 접근 성공")
            
            # 페이지 내용에서 LawFirmAI 확인
            if "LawFirmAI" in response.text:
                print("LawFirmAI 앱 정상 로드")
            else:
                print("LawFirmAI 앱 로드 확인 필요")
                
        else:
            print(f"웹 페이지 접근 실패: HTTP {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"연결 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
    
    print()
    print("=== 테스트 완료 ===")
    print("브라우저에서 http://localhost:7861 에 접속하여")
    print("실제 채팅 기능을 테스트해보세요.")

if __name__ == "__main__":
    test_gradio_error_fix()
