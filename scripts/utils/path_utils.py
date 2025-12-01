#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로젝트 경로 설정 유틸리티

프로젝트 루트 경로를 sys.path에 추가하는 공통 함수
"""

import sys
from pathlib import Path
from typing import Optional


def setup_project_path(script_path: Optional[Path] = None) -> Path:
    """
    프로젝트 루트를 sys.path에 추가
    
    Args:
        script_path: 현재 스크립트의 경로 (None이면 자동 감지)
    
    Returns:
        Path: 프로젝트 루트 경로
    """
    if script_path is None:
        # 호출한 스크립트의 경로를 자동으로 감지
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            script_path = Path(frame.f_back.f_globals.get('__file__', __file__)).resolve()
        else:
            script_path = Path(__file__).resolve()
    
    # scripts/analysis/xxx.py -> scripts/ -> 프로젝트 루트
    # scripts/utils/xxx.py -> scripts/ -> 프로젝트 루트
    current = Path(script_path).resolve()
    
    # scripts 디렉토리 찾기
    project_root = None
    for parent in [current] + list(current.parents):
        if parent.name == 'scripts' and parent.parent.name == 'LawFirmAI':
            project_root = parent.parent
            break
    
    # scripts 디렉토리를 찾지 못한 경우, scripts의 부모를 프로젝트 루트로 간주
    if project_root is None:
        if 'scripts' in current.parts:
            scripts_idx = current.parts.index('scripts')
            project_root = Path(*current.parts[:scripts_idx])
        else:
            # scripts 디렉토리가 경로에 없는 경우, 현재 파일 기준으로 2단계 위
            project_root = current.parent.parent
    
    # sys.path에 추가
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_project_root() -> Path:
    """
    프로젝트 루트 경로 반환 (sys.path 설정 없이)
    
    Returns:
        Path: 프로젝트 루트 경로
    """
    current = Path(__file__).resolve()
    
    # scripts 디렉토리 찾기
    for parent in [current] + list(current.parents):
        if parent.name == 'scripts' and parent.parent.name == 'LawFirmAI':
            return parent.parent
    
    # 찾지 못한 경우 기본값
    return current.parent.parent

