#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Variables Test Script
환경 변수 적용 테스트 스크립트
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

def test_environment_variables():
    """환경 변수 테스트"""
    print("=== 환경 변수 테스트 ===")
    
    # 기본 환경 변수 확인
    env_vars = [
        'LLM_PROVIDER',
        'LLM_MODEL', 
        'LLM_TEMPERATURE',
        'LLM_MAX_TOKENS',
        'GOOGLE_API_KEY',
        'LANGFUSE_ENABLED',
        'VECTOR_STORE_TYPE',
        'EMBEDDING_MODEL'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    print()

def test_langchain_config():
    """LangChain 설정 테스트"""
    print("=== LangChain 설정 테스트 ===")
    
    try:
        from utils.langchain_config import LangChainConfig
        
        config = LangChainConfig.from_env()
        
        print(f"✅ LLM Provider: {config.llm_provider.value}")
        print(f"✅ LLM Model: {config.llm_model}")
        print(f"✅ LLM Temperature: {config.llm_temperature}")
        print(f"✅ LLM Max Tokens: {config.llm_max_tokens}")
        print(f"✅ Google API Key: {'Set' if config.google_api_key else 'Not set'}")
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

def test_gemini_support():
    """Gemini Pro 지원 테스트"""
    print("=== Gemini Pro 지원 테스트 ===")
    
    try:
        # LangChain Google 통합 확인
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("✅ langchain-google-genai 패키지 사용 가능")
        except ImportError:
            print("❌ langchain-google-genai 패키지 없음")
            return False
        
        # Google AI SDK 확인
        try:
            import google.generativeai as genai
            print("✅ google-generativeai 패키지 사용 가능")
        except ImportError:
            print("❌ google-generativeai 패키지 없음")
            return False
        
        # Gemini Pro 모델 초기화 테스트 (API 키가 유효한 경우에만)
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key and api_key != 'test-key':
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                print("✅ Gemini Pro 모델 초기화 성공")
            except Exception as e:
                print(f"⚠️ Gemini Pro 모델 초기화 실패: {e}")
        else:
            print("ℹ️ 테스트용 API 키로 모델 초기화 건너뜀")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini Pro 지원 테스트 실패: {e}")
        return False

def test_answer_generator():
    """답변 생성기 테스트"""
    print("=== 답변 생성기 테스트 ===")
    
    try:
        from services.answer_generator import AnswerGenerator
        from utils.langchain_config import LangChainConfig
        
        config = LangChainConfig.from_env()
        generator = AnswerGenerator(config)
        
        print(f"✅ 답변 생성기 초기화 성공")
        print(f"✅ LLM 사용 가능: {generator.llm is not None}")
        print(f"✅ 템플릿 사용 가능: {list(generator.prompt_templates.keys())}")
        print(f"✅ 체인 사용 가능: {list(generator.llm_chains.keys())}")
        
        # 기본 답변 생성 테스트
        try:
            result = generator.generate_answer(
                query="테스트 질문",
                context="테스트 컨텍스트입니다.",
                template_type="legal_qa"
            )
            print(f"✅ 답변 생성 성공: {result.response_time:.2f}초")
            print(f"✅ 신뢰도: {result.confidence:.2f}")
            print(f"✅ 토큰 사용량: {result.tokens_used}")
            
        except Exception as e:
            print(f"⚠️ 답변 생성 테스트 실패: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 답변 생성기 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 환경 변수 적용 테스트 시작")
    print("=" * 50)
    
    # 1. 환경 변수 테스트
    test_environment_variables()
    
    # 2. LangChain 설정 테스트
    config = test_langchain_config()
    if not config:
        print("❌ 설정 테스트 실패로 인해 테스트를 종료합니다.")
        return
    
    # 3. Gemini Pro 지원 테스트
    gemini_support = test_gemini_support()
    
    # 4. 답변 생성기 테스트
    generator_test = test_answer_generator()
    
    print("=" * 50)
    print("✅ 환경 변수 적용 테스트 완료")
    print()
    
    # 결과 요약
    print("📊 테스트 결과 요약:")
    print(f"   - 환경 변수 설정: {'✅' if any(os.getenv(var) for var in ['LLM_PROVIDER', 'GOOGLE_API_KEY']) else '❌'}")
    print(f"   - LangChain 설정: {'✅' if config else '❌'}")
    print(f"   - Gemini Pro 지원: {'✅' if gemini_support else '❌'}")
    print(f"   - 답변 생성기: {'✅' if generator_test else '❌'}")
    print()
    
    if config and config.llm_provider.value == 'google':
        print("🎉 Gemini Pro 설정이 성공적으로 적용되었습니다!")
        print()
        print("🔧 다음 단계:")
        print("   1. 실제 Google API 키로 교체")
        print("   2. 벡터 데이터베이스 구축")
        print("   3. RAG 시스템 테스트")
    else:
        print("⚠️ Gemini Pro 설정을 확인해주세요.")
        print("   환경 변수: LLM_PROVIDER=google, GOOGLE_API_KEY=your-key")

if __name__ == "__main__":
    main()
