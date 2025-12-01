#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file_utils 모듈 단위 테스트
"""

import json
import tempfile
from pathlib import Path
import pytest

from scripts.utils.file_utils import (
    load_json_file,
    save_json_file,
    load_json_files
)


class TestFileUtils:
    """file_utils 모듈 테스트"""
    
    def test_save_and_load_json_file(self):
        """JSON 파일 저장 및 로드 테스트"""
        test_data = {
            "name": "테스트",
            "value": 123,
            "items": [1, 2, 3]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            
            # 저장
            save_json_file(test_data, file_path)
            assert file_path.exists()
            
            # 로드
            loaded_data = load_json_file(file_path)
            assert loaded_data == test_data
    
    def test_load_json_file_not_found(self):
        """존재하지 않는 파일 로드 시 오류 테스트"""
        non_existent_file = Path("/nonexistent/path/file.json")
        
        with pytest.raises(FileNotFoundError):
            load_json_file(non_existent_file)
    
    def test_load_json_files_directory(self):
        """디렉토리 내 JSON 파일들 로드 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # 테스트 파일 생성
            file1 = tmp_path / "file1.json"
            file2 = tmp_path / "file2.json"
            
            save_json_file({"id": 1}, file1)
            save_json_file({"id": 2}, file2)
            
            # 로드
            results = load_json_files(tmp_path)
            assert len(results) == 2
            assert {"id": 1} in results
            assert {"id": 2} in results
    
    def test_save_json_file_creates_directory(self):
        """JSON 저장 시 디렉토리 자동 생성 테스트"""
        test_data = {"test": "data"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.json"
            
            # 디렉토리가 없는 상태에서 저장
            save_json_file(test_data, file_path)
            assert file_path.exists()
            assert file_path.parent.exists()

