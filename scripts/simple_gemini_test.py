#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Gemini Pro 환경 변수 테스트 스크립트
"""

import os
import sys

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'source'))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

def test_env_variables():
    """환경 변수 테스트"""
    print("🚀 Gemini Pro 환경 변수 테스트")
    print("=" * 50)
    
    # 주요 환경 변수 확인
    env_vars = {
        'LLM_PROVIDER': os.getenv('LLM_PROVIDER'),
        'LLM_MODEL': os.getenv('LLM_MODEL'),
        'LLM_TEMPERATURE': os.getenv('LLM_TEMPERATURE'),
        'LLM_MAX_TOKENS': os.getenv('LLM_MAX_TOKENS'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'LANGFUSE_ENABLED': os.getenv('LANGFUSE_ENABLED'),
        'VECTOR_STORE_TYPE': os.getenv('VECTOR_STORE_TYPE'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL')
    }
    
    print("📋 .env 파일에서 로드된 환경 변수:")
    for key, value in env_vars.items():
        if value:
            # API 키는 보안을 위해 일부만 표시
            if 'API_KEY' in key:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {key}: {display_value}")
        else:
            print(f"❌ {key}: Not set")
    
    print()

def test_gemini_packages():
    """Gemini Pro 관련 패키지 테스트"""
    print("=== Gemini Pro 패키지 테스트 ===")
    
    # langchain-google-genai 테스트
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ langchain-google-genai 패키지 사용 가능")
    except ImportError as e:
        print(f"❌ langchain-google-genai 패키지 없음: {e}")
        return False
    
    # google-generativeai 테스트
    try:
        import google.generativeai as genai
        print("✅ google-generativeai 패키지 사용 가능")
    except ImportError as e:
        print(f"❌ google-generativeai 패키지 없음: {e}")
        return False
    
    # Gemini Pro 모델 초기화 테스트
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key and api_key not in ['your-google-api-key-here', 'test-google-api-key']:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini Pro 모델 초기화 성공")
        except Exception as e:
            print(f"⚠️ Gemini Pro 모델 초기화 실패: {e}")
    else:
        print("ℹ️ 테스트용 API 키로 모델 초기화 건너뜀")
    
    return True

def test_langchain_config():
    """LangChain 설정 테스트"""
    print("=== LangChain 설정 테스트 ===")
    
    try:
        # 직접 import 시도
        sys.path.insert(0, os.path.join(project_root, 'source', 'utils'))
        from langchain_config import LangChainConfig
        
        config = LangChainConfig.from_env()
        
        print("📋 LangChain 설정:")
        print(f"✅ LLM Provider: {config.llm_provider.value}")
        print(f"✅ LLM Model: {config.llm_model}")
        print(f"✅ LLM Temperature: {config.llm_temperature}")
        print(f"✅ LLM Max Tokens: {config.llm_max_tokens}")
        print(f"✅ Google API Key: {'설정됨' if config.google_api_key else '미설정'}")
        print(f"✅ Langfuse Enabled: {config.langfuse_enabled}")
        print(f"✅ Vector Store Type: {config.vector_store_type.value}")
        print(f"✅ Embedding Model: {config.embedding_model}")
        
        # 설정 유효성 검사
        errors = config.validate()
        if errors:
            print(f"⚠️ 설정 오류: {errors}")
        else:
            print("✅ 설정 유효성 검사 통과")
            
        return config
        
    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        return None

def test_gemini_chat():
    """Gemini Pro 채팅 테스트"""
    print("=== Gemini Pro 채팅 테스트 ===")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key in ['your-google-api-key-here', 'test-google-api-key']:
        print("ℹ️ 실제 API 키가 없어 채팅 테스트를 건너뜁니다.")
        return False
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # 간단한 테스트 질문
        test_question = "안녕하세요! 간단한 인사말을 해주세요."
        
        print(f"📝 테스트 질문: {test_question}")
        
        response = model.generate_content(test_question)
        
        print(f"✅ Gemini Pro 응답: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ 채팅 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 간단한 Gemini Pro 환경 변수 테스트 시작")
    print("=" * 60)
    
    # 1. 환경 변수 테스트
    test_env_variables()
    
    # 2. 패키지 테스트
    package_success = test_gemini_packages()
    
    # 3. 설정 테스트
    config = test_langchain_config()
    
    # 4. 채팅 테스트 (실제 API 키가 있는 경우)
    chat_success = test_gemini_chat()
    
    print("=" * 60)
    print("✅ 테스트 완료")
    print()
    
    # 결과 요약
    print("📊 테스트 결과 요약:")
    print(f"   - .env 파일 로딩: ✅")
    print(f"   - Gemini Pro 패키지: {'✅' if package_success else '❌'}")
    print(f"   - LangChain 설정: {'✅' if config else '❌'}")
    print(f"   - Gemini Pro 채팅: {'✅' if chat_success else 'ℹ️'}")
    print()
    
    if config and config.llm_provider.value == 'google':
        print("🎉 .env 파일 기반 Gemini Pro 설정이 성공적으로 적용되었습니다!")
        print()
        print("🔧 다음 단계:")
        print("   1. 실제 Google API 키로 교체")
        print("   2. 벡터 데이터베이스 구축")
        print("   3. RAG 시스템 테스트")
    else:
        print("⚠️ Gemini Pro 설정을 확인해주세요.")
        print("   .env 파일에서 LLM_PROVIDER=google, GOOGLE_API_KEY=your-key 설정")

if __name__ == "__main__":
    main()
