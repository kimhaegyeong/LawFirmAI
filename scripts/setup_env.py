#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
환경변수 설정 도우미 스크립트
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """환경변수 설정"""
    print("🔧 LawFirmAI 환경변수 설정")
    print("=" * 50)
    
    # LAW_OPEN_API_OC 설정
    current_oc = os.getenv("LAW_OPEN_API_OC")
    if current_oc:
        print(f"✅ LAW_OPEN_API_OC가 이미 설정되어 있습니다: {current_oc}")
    else:
        print("❌ LAW_OPEN_API_OC가 설정되지 않았습니다.")
        print("\n📝 설정 방법:")
        print("1. PowerShell에서:")
        print("   $env:LAW_OPEN_API_OC='your_email@example.com'")
        print("\n2. CMD에서:")
        print("   set LAW_OPEN_API_OC=your_email@example.com")
        print("\n3. .env 파일 생성:")
        print("   LAW_OPEN_API_OC=your_email@example.com")
        
        # 사용자 입력 받기
        email = input("\n이메일 주소를 입력하세요 (또는 Enter로 건너뛰기): ").strip()
        if email:
            os.environ["LAW_OPEN_API_OC"] = email
            print(f"✅ 환경변수 설정 완료: {email}")
        else:
            print("⚠️ 환경변수를 수동으로 설정해주세요.")
    
    # .env 파일 생성
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📄 .env 파일을 생성합니다...")
        env_content = f"""# LawFirmAI 환경변수 설정
LAW_OPEN_API_OC={os.getenv("LAW_OPEN_API_OC", "your_email@example.com")}

# 데이터베이스 설정
DATABASE_URL=sqlite:///./data/lawfirm.db

# 모델 설정
MODEL_PATH=./models
MODEL_CACHE_DIR=./cache

# 로깅 설정
LOG_LEVEL=INFO
LOG_DIR=./logs

# API 설정
API_HOST=0.0.0.0
API_PORT=8000
GRADIO_PORT=7860

# 성능 설정
MAX_WORKERS=4
BATCH_SIZE=50
TIMEOUT=60

# 보안 설정
SECRET_KEY=your_secret_key_here
API_KEY=your_api_key_here
"""
        try:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print("✅ .env 파일 생성 완료")
        except Exception as e:
            print(f"❌ .env 파일 생성 실패: {e}")
    else:
        print("✅ .env 파일이 이미 존재합니다.")
    
    print("\n🎉 환경변수 설정 완료!")
    print("이제 판례 수집 스크립트를 실행할 수 있습니다.")

if __name__ == "__main__":
    setup_environment()
