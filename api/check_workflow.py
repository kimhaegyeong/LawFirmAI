"""
워크플로우 초기화 확인 스크립트
"""
import sys
from pathlib import Path
import os

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=" * 60)
print("LangGraph 워크플로우 초기화 확인")
print("=" * 60)

# 환경 변수 확인
print("\n[환경 변수 확인]")
google_api_key = os.getenv("GOOGLE_API_KEY", "")
print(f"GOOGLE_API_KEY: {'✅ 설정됨' if google_api_key else '❌ 설정되지 않음'}")
print(f"GOOGLE_MODEL: {os.getenv('GOOGLE_MODEL', '기본값 사용')}")
print(f"LANGGRAPH_ENABLED: {os.getenv('LANGGRAPH_ENABLED', 'true')}")

# ChatService 초기화 테스트
print("\n[ChatService 초기화 테스트]")
try:
    from api.services.chat_service import get_chat_service
    
    chat_service = get_chat_service()
    if chat_service.is_available():
        print("✅ ChatService: 워크플로우 사용 가능")
    else:
        print("❌ ChatService: 워크플로우 사용 불가능")
        print("   → API 서버 시작 시 로그를 확인하세요.")
except Exception as e:
    print(f"❌ ChatService 초기화 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

