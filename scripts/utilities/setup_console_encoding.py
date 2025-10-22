#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
콘솔 인코딩 설정 유틸리티
Windows 콘솔에서 한국어 출력을 위한 인코딩 설정
"""

import sys
import os
import locale
import codecs

def setup_console_encoding():
    """콘솔 인코딩 설정"""
    try:
        # Windows에서 콘솔 인코딩을 UTF-8로 설정
        if sys.platform == "win32":
            # 콘솔 코드페이지를 UTF-8로 설정
            os.system("chcp 65001 > nul")
            
            # stdout과 stderr의 인코딩을 UTF-8로 설정
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8')
            
            # 환경 변수 설정
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
        # 로케일 설정
        try:
            locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'Korean_Korea.utf8')
            except:
                pass  # 로케일 설정 실패 시 무시
        
        print("✅ 콘솔 인코딩 설정 완료")
        return True
        
    except Exception as e:
        print(f"❌ 콘솔 인코딩 설정 실패: {e}")
        return False

def test_korean_output():
    """한국어 출력 테스트"""
    test_messages = [
        "안녕하세요! 한국어 출력 테스트입니다.",
        "법률 AI 어시스턴트 테스트",
        "대화 맥락 강화 기능",
        "개인화 및 분석 기능", 
        "장기 기억 및 품질 모니터링",
        "성능 최적화 및 메모리 관리"
    ]
    
    print("\n=== 한국어 출력 테스트 ===")
    for i, message in enumerate(test_messages, 1):
        print(f"{i}. {message}")
    
    print("\n✅ 한국어 출력 테스트 완료")

def safe_print(message, *args, **kwargs):
    """안전한 한국어 출력 함수"""
    try:
        # 메시지를 UTF-8로 인코딩하여 출력
        if isinstance(message, str):
            print(message, *args, **kwargs)
        else:
            print(str(message), *args, **kwargs)
    except UnicodeEncodeError:
        # 인코딩 오류 시 ASCII로 변환하여 출력
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message, *args, **kwargs)

if __name__ == "__main__":
    print("콘솔 인코딩 설정 시작...")
    
    # 인코딩 설정
    success = setup_console_encoding()
    
    if success:
        # 한국어 출력 테스트
        test_korean_output()
    else:
        print("인코딩 설정에 실패했습니다.")
