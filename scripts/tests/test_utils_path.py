#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
path_utils 모듈 단위 테스트
"""

import sys
from pathlib import Path
import pytest

from scripts.utils.path_utils import setup_project_path, get_project_root


class TestPathUtils:
    """path_utils 모듈 테스트"""
    
    def test_setup_project_path(self):
        """프로젝트 경로 설정 테스트"""
        project_root = setup_project_path()
        
        assert project_root is not None
        assert isinstance(project_root, Path)
        assert project_root.exists()
        assert str(project_root) in sys.path
    
    def test_get_project_root(self):
        """프로젝트 루트 경로 반환 테스트"""
        project_root = get_project_root()
        
        assert project_root is not None
        assert isinstance(project_root, Path)
        assert project_root.exists()
    
    def test_project_root_consistency(self):
        """두 함수가 동일한 경로를 반환하는지 테스트"""
        root1 = setup_project_path()
        root2 = get_project_root()
        
        assert root1 == root2

