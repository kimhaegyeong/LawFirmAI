#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Law Open API 실행 스크립트 모듈

법령용어 수집을 위한 실행 스크립트들을 제공합니다.
"""

# 스크립트 모듈들을 import할 수 있도록 설정
import sys
from pathlib import Path

# 스크립트 디렉토리를 Python 경로에 추가
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

__all__ = [
    'start_legal_term_scheduler',
    'manual_collect_legal_terms', 
    'monitor_collection_status'
]




