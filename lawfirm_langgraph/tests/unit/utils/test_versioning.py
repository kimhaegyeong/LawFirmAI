# -*- coding: utf-8 -*-
"""
Versioning 테스트
유틸리티 versioning 모듈 단위 테스트
"""

import pytest
import tempfile
import os
from pathlib import Path

from lawfirm_langgraph.core.utils.versioning import (
    _parse_env,
    _dump_env,
    switch_active_versions
)


class TestVersioning:
    """Versioning 테스트"""
    
    def test_parse_env_nonexistent_file(self):
        """존재하지 않는 파일 파싱 테스트"""
        path = Path("/nonexistent/path/.env")
        result = _parse_env(path)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_parse_env_empty_file(self, tmp_path):
        """빈 파일 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text("", encoding="utf-8")
        
        result = _parse_env(env_file)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_parse_env_basic(self, tmp_path):
        """기본 환경 변수 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n", encoding="utf-8")
        
        result = _parse_env(env_file)
        
        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value2"
    
    def test_parse_env_with_comments(self, tmp_path):
        """주석 포함 파일 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\n"
            "KEY1=value1\n"
            "# Another comment\n"
            "KEY2=value2\n",
            encoding="utf-8"
        )
        
        result = _parse_env(env_file)
        
        assert "KEY1" in result
        assert "KEY2" in result
        assert "#" not in result
    
    def test_parse_env_with_empty_lines(self, tmp_path):
        """빈 줄 포함 파일 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text(
            "KEY1=value1\n"
            "\n"
            "KEY2=value2\n",
            encoding="utf-8"
        )
        
        result = _parse_env(env_file)
        
        assert result["KEY1"] == "value1"
        assert result["KEY2"] == "value2"
    
    def test_parse_env_with_equals_in_value(self, tmp_path):
        """값에 등호 포함 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value=with=equals\n", encoding="utf-8")
        
        result = _parse_env(env_file)
        
        assert result["KEY"] == "value=with=equals"
    
    def test_parse_env_with_whitespace(self, tmp_path):
        """공백 포함 파싱 테스트"""
        env_file = tmp_path / ".env"
        env_file.write_text(" KEY1 = value1 \n", encoding="utf-8")
        
        result = _parse_env(env_file)
        
        assert result["KEY1"] == "value1"
    
    def test_dump_env_basic(self, tmp_path):
        """기본 환경 변수 저장 테스트"""
        env_file = tmp_path / ".env"
        data = {"KEY1": "value1", "KEY2": "value2"}
        
        _dump_env(env_file, data)
        
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "KEY1=value1" in content
        assert "KEY2=value2" in content
    
    def test_dump_env_empty(self, tmp_path):
        """빈 데이터 저장 테스트"""
        env_file = tmp_path / ".env"
        data = {}
        
        _dump_env(env_file, data)
        
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert content == ""
    
    def test_switch_active_versions_new_file(self, tmp_path):
        """새 파일에 버전 전환 테스트"""
        env_file = tmp_path / ".env"
        
        switch_active_versions(
            str(env_file),
            "v2",
            "model@2.0"
        )
        
        assert env_file.exists()
        result = _parse_env(env_file)
        assert result["ACTIVE_CORPUS_VERSION"] == "v2"
        assert result["ACTIVE_MODEL_VERSION"] == "model@2.0"
    
    def test_switch_active_versions_existing_file(self, tmp_path):
        """기존 파일에 버전 전환 테스트"""
        env_file = tmp_path / ".env"
        backup_file = tmp_path / ".env.bak"
        
        env_file.write_text("EXISTING_KEY=existing_value\n", encoding="utf-8")
        
        switch_active_versions(
            str(env_file),
            "v2",
            "model@2.0"
        )
        
        assert backup_file.exists()
        result = _parse_env(env_file)
        assert result["ACTIVE_CORPUS_VERSION"] == "v2"
        assert result["ACTIVE_MODEL_VERSION"] == "model@2.0"
        assert result["EXISTING_KEY"] == "existing_value"
    
    def test_switch_active_versions_backup_content(self, tmp_path):
        """백업 파일 내용 확인 테스트"""
        env_file = tmp_path / ".env"
        backup_file = tmp_path / ".env.bak"
        
        original_content = "ORIGINAL_KEY=original_value\n"
        env_file.write_text(original_content, encoding="utf-8")
        
        switch_active_versions(
            str(env_file),
            "v2",
            "model@2.0"
        )
        
        backup_content = backup_file.read_text(encoding="utf-8")
        assert backup_content == original_content
    
    def test_switch_active_versions_defaults(self, tmp_path):
        """기본값 설정 테스트"""
        env_file = tmp_path / ".env"
        
        switch_active_versions(
            str(env_file),
            "v2",
            "model@2.0"
        )
        
        result = _parse_env(env_file)
        assert "ACTIVE_CORPUS_VERSION" in result
        assert "ACTIVE_MODEL_VERSION" in result

