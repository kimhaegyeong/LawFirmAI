# -*- coding: utf-8 -*-
"""
Pytest Configuration
Pytest 공통 설정 및 픽스처
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ["USE_LANGGRAPH"] = "true"
os.environ["LANGGRAPH_ENABLED"] = "true"
os.environ["LANGFUSE_ENABLED"] = "false"  # 테스트 시에는 기본적으로 비활성화
