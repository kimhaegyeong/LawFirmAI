# -*- coding: utf-8 -*-
"""
Date Utils 테스트
유틸리티 date_utils 모듈 단위 테스트
"""

import pytest

from lawfirm_langgraph.core.utils.date_utils import yyyymmdd_to_iso


class TestDateUtils:
    """Date Utils 테스트"""
    
    def test_yyyymmdd_to_iso_valid(self):
        """유효한 YYYYMMDD 형식 변환 테스트"""
        result = yyyymmdd_to_iso("20231225")
        assert result == "2023-12-25"
    
    def test_yyyymmdd_to_iso_different_dates(self):
        """다양한 날짜 형식 변환 테스트"""
        test_cases = [
            ("20230101", "2023-01-01"),
            ("20231231", "2023-12-31"),
            ("20240229", "2024-02-29"),
            ("19991231", "1999-12-31"),
        ]
        
        for input_date, expected in test_cases:
            result = yyyymmdd_to_iso(input_date)
            assert result == expected
    
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
        result = yyyymmdd_to_iso("2023122")
        assert result == "2023122"
        
        result = yyyymmdd_to_iso("202312251")
        assert result == "202312251"
    
    def test_yyyymmdd_to_iso_non_digit(self):
        """숫자가 아닌 문자 포함 테스트"""
        result = yyyymmdd_to_iso("2023-12-25")
        assert result == "2023-12-25"
        
        result = yyyymmdd_to_iso("2023abcd")
        assert result == "2023abcd"
    
    def test_yyyymmdd_to_iso_whitespace(self):
        """공백 포함 입력 테스트"""
        result = yyyymmdd_to_iso(" 20231225 ")
        assert result == "2023-12-25"
        
        result = yyyymmdd_to_iso("   ")
        assert result is None
    
    def test_yyyymmdd_to_iso_numeric_string(self):
        """숫자 문자열 입력 테스트"""
        result = yyyymmdd_to_iso(20231225)
        assert result == "2023-12-25"

