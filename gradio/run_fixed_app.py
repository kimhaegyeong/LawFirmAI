#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LawFirmAI - 수정된 Gradio 앱 실행 스크립트
JavaScript 오류와 manifest.json 404 오류가 수정된 버전
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    """수정된 앱 실행"""
    print("🚀 LawFirmAI - 수정된 Gradio 앱을 시작합니다...")
    print("📝 수정 사항:")
    print("   - share-modal.js 오류 해결")
    print("   - manifest.json 404 오류 해결")
    print("   - 정적 파일 서빙 개선")
    print("   - 안정적인 launch 설정")
    print()
    
    # 환경 변수 설정
    os.environ["USE_LANGGRAPH"] = os.getenv("USE_LANGGRAPH", "true")
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = "7860"
    
    # 수정된 앱 실행
    from gradio.app import main as app_main
    app_main()

if __name__ == "__main__":
    main()
