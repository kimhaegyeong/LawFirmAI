# -*- coding: utf-8 -*-
"""
Date Utils 테스트
core/utils/date_utils.py 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.utils.date_utils import yyyymmdd_to_iso


class TestDateUtils:
    """Date Utils 테스트"""
    
    def test_yyyymmdd_to_iso_valid(self):
        """유효한 YYYYMMDD 형식 변환 테스트"""
        result = yyyymmdd_to_iso("20231215")
        assert result == "2023-12-15"
    
    def test_yyyymmdd_to_iso_different_dates(self):
        """다양한 날짜 형식 변환 테스트"""
        assert yyyymmdd_to_iso("20240101") == "2024-01-01"
        assert yyyymmdd_to_iso("20240229") == "2024-02-29"
        assert yyyymmdd_to_iso("20231231") == "2023-12-31"
    
    def test_yyyymmdd_to_iso_none(self):
        """None 입력 테스트"""
        result = yyyymmdd_to_iso(None)
        assert result is None
    
    def test_yyyymmdd_to_iso_empty_string(self):
        """빈 문자열 입력 테스트"""
        result = yyyymmdd_to_iso("")
        assert result is None
    
    def test_yyyymmdd_to_iso_invalid_length(self):
        """잘못된 길이 입력 테스트"""
        result = yyyymmdd_to_iso("2023121")
        assert result == "2023121"
        
        result = yyyymmdd_to_iso("202312150")
        assert result == "202312150"
    
    def test_yyyymmdd_to_iso_non_digit(self):
        """숫자가 아닌 문자 포함 테스트"""
        result = yyyymmdd_to_iso("2023-12-15")
        assert result == "2023-12-15"
        
        result = yyyymmdd_to_iso("2023abcd")
        assert result == "2023abcd"
    
    def test_yyyymmdd_to_iso_whitespace(self):
        """공백 포함 입력 테스트"""
        result = yyyymmdd_to_iso(" 20231215 ")
        assert result == "2023-12-15"
    
    def test_yyyymmdd_to_iso_integer_input(self):
        """정수 입력 테스트"""
        result = yyyymmdd_to_iso(20231215)
        assert result == "2023-12-15"
    
    def test_yyyymmdd_to_iso_already_iso_format(self):
        """이미 ISO 형식인 경우 테스트"""
        result = yyyymmdd_to_iso("2023-12-15")
        assert result == "2023-12-15"
