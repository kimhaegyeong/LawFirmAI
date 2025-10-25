# -*- coding: utf-8 -*-
"""
Langfuse API 확인 테스트
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Langfuse API 확인")
print("=" * 40)

try:
    from langfuse import Langfuse
    
    # Langfuse 클라이언트 초기화
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    print("✅ Langfuse 클라이언트 초기화 성공")
    
    # 사용 가능한 메서드 확인
    print("\n📋 사용 가능한 메서드:")
    methods = [method for method in dir(langfuse) if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
    # 간단한 테스트
    print("\n🧪 간단한 테스트:")
    
    # trace_id 생성
    trace_id = langfuse.create_trace_id()
    print(f"✅ trace_id 생성: {trace_id}")
    
    # 이벤트 생성
    try:
        event = langfuse.create_event(
            trace_id=trace_id,
            name="test_event",
            input="test input",
            output="test output"
        )
        print("✅ 이벤트 생성 성공")
    except Exception as e:
        print(f"❌ 이벤트 생성 실패: {e}")
    
    # 플러시
    langfuse.flush()
    print("✅ 데이터 플러시 완료")
    
except Exception as e:
    print(f"❌ 오류: {e}")
    import traceback
    print(traceback.format_exc())
