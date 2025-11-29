#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가상환경 확인 유틸리티

scripts/.venv 가상환경이 활성화되어 있는지 확인합니다.
"""

import sys
import os
from pathlib import Path


def check_venv() -> bool:
    """
    scripts/.venv 가상환경이 활성화되어 있는지 확인
    
    Returns:
        bool: 가상환경이 활성화되어 있으면 True, 아니면 False
    """
    # 현재 파일의 위치에서 scripts 디렉토리 찾기
    current_file = Path(__file__).resolve()
    scripts_dir = current_file.parent.parent
    venv_path = scripts_dir / ".venv"
    
    # 가상환경이 활성화되어 있는지 확인
    if hasattr(sys, 'real_prefix'):
        # virtualenv를 사용하는 경우
        venv_active = True
        venv_base = Path(sys.real_prefix)
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        # venv 모듈을 사용하는 경우
        venv_active = True
        venv_base = Path(sys.base_prefix)
    else:
        # 가상환경이 활성화되지 않음
        venv_active = False
        venv_base = None
    
    if not venv_active:
        print("⚠️  Warning: scripts/.venv 가상환경이 활성화되지 않았습니다.")
        print(f"   활성화 방법: cd scripts && .\\.venv\\Scripts\\Activate.ps1 (Windows)")
        print(f"   활성화 방법: cd scripts && source .venv/bin/activate (Linux/Mac)")
        return False
    
    # scripts/.venv가 활성화된 가상환경인지 확인
    current_prefix = Path(sys.prefix).resolve()
    expected_venv = venv_path.resolve()
    
    if str(expected_venv) in str(current_prefix) or current_prefix == expected_venv:
        return True
    
    # 경로가 정확히 일치하지 않지만 가상환경이 활성화되어 있는 경우 경고
    print(f"⚠️  Warning: 다른 가상환경이 활성화되어 있습니다.")
    print(f"   현재 가상환경: {current_prefix}")
    print(f"   예상 가상환경: {expected_venv}")
    print(f"   scripts/.venv를 활성화해주세요.")
    return False


if __name__ == "__main__":
    """테스트용"""
    if check_venv():
        print("✅ scripts/.venv 가상환경이 활성화되어 있습니다.")
    else:
        print("❌ scripts/.venv 가상환경이 활성화되지 않았습니다.")
        sys.exit(1)

