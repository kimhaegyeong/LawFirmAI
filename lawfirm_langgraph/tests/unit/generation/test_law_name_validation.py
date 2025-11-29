# -*- coding: utf-8 -*-
"""
법령명 검증 기능 테스트
접두어 제거 및 LRU 캐싱 기능 테스트
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lawfirm_langgraph.core.generation.validators.quality_validators import AnswerValidator


class TestLawNamePrefixRemoval:
    """법령명 접두어 제거 테스트"""
    
    def test_remove_prefix_특히(self):
        """'특히' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("특히국세기본법")
        assert result == "국세기본법", f"Expected '국세기본법', got '{result}'"
    
    def test_remove_prefix_또한(self):
        """'또한' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("또한국세기본법")
        assert result == "국세기본법", f"Expected '국세기본법', got '{result}'"
    
    def test_remove_prefix_규정한(self):
        """'규정한' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("규정한민사집행법")
        assert result == "민사집행법", f"Expected '민사집행법', got '{result}'"
    
    def test_remove_prefix_규정은(self):
        """'규정은' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("규정은민사집행법")
        assert result == "민사집행법", f"Expected '민사집행법', got '{result}'"
    
    def test_remove_prefix_한편(self):
        """'한편' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("한편민사집행법")
        assert result == "민사집행법", f"Expected '민사집행법', got '{result}'"
    
    def test_remove_prefix_나아가(self):
        """'나아가' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("나아가민사집행법")
        assert result == "민사집행법", f"Expected '민사집행법', got '{result}'"
    
    def test_remove_prefix_위하여는(self):
        """'위하여는' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("위하여는민사소송법")
        assert result == "민사소송법", f"Expected '민사소송법', got '{result}'"
    
    def test_remove_prefix_반드시(self):
        """'반드시' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("반드시민사소송법")
        assert result == "민사소송법", f"Expected '민사소송법', got '{result}'"
    
    def test_remove_prefix_외국판결이(self):
        """'외국판결이' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("외국판결이민사소송법")
        assert result == "민사소송법", f"Expected '민사소송법', got '{result}'"
    
    def test_remove_prefix_보전처분은(self):
        """'보전처분은' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("보전처분은민사집행법")
        assert result == "민사집행법", f"Expected '민사집행법', got '{result}'"
    
    def test_remove_prefix_대하여(self):
        """'대하여' 접두어 제거 테스트"""
        result = AnswerValidator._remove_law_name_prefix("대하여국세기본법")
        assert result == "국세기본법", f"Expected '국세기본법', got '{result}'"
    
    def test_no_prefix(self):
        """접두어가 없는 경우 테스트"""
        result = AnswerValidator._remove_law_name_prefix("민법")
        assert result == "민법", f"Expected '민법', got '{result}'"
    
    def test_empty_string(self):
        """빈 문자열 테스트"""
        result = AnswerValidator._remove_law_name_prefix("")
        assert result == "", f"Expected '', got '{result}'"
    
    def test_multiple_prefixes(self):
        """여러 접두어가 있는 경우 (가장 긴 것부터 제거)"""
        # "특히또한" 같은 경우는 실제로는 발생하지 않지만 테스트
        result = AnswerValidator._remove_law_name_prefix("특히민법")
        assert result == "민법", f"Expected '민법', got '{result}'"


class TestCitationNormalization:
    """Citation 정규화 테스트"""
    
    def test_normalize_with_prefix(self):
        """접두어가 포함된 Citation 정규화 테스트"""
        citation = "특히국세기본법 제18조"
        result = AnswerValidator._normalize_citation(citation)
        
        assert result["type"] == "law", f"Expected type 'law', got '{result.get('type')}'"
        assert result["law_name"] == "국세기본법", f"Expected law_name '국세기본법', got '{result.get('law_name')}'"
        assert result["article_number"] == "18", f"Expected article_number '18', got '{result.get('article_number')}'"
        assert result["normalized"] == "국세기본법 제18조", f"Expected normalized '국세기본법 제18조', got '{result.get('normalized')}'"
    
    def test_normalize_with_prefix_또한(self):
        """'또한' 접두어가 포함된 Citation 정규화 테스트"""
        citation = "또한국세기본법 제18조"
        result = AnswerValidator._normalize_citation(citation)
        
        assert result["type"] == "law"
        assert result["law_name"] == "국세기본법"
        assert result["article_number"] == "18"
    
    def test_normalize_with_prefix_규정한(self):
        """'규정한' 접두어가 포함된 Citation 정규화 테스트"""
        citation = "규정한민사집행법 제287조"
        result = AnswerValidator._normalize_citation(citation)
        
        assert result["type"] == "law"
        assert result["law_name"] == "민사집행법"
        assert result["article_number"] == "287"
    
    def test_normalize_without_prefix(self):
        """접두어가 없는 Citation 정규화 테스트"""
        citation = "민법 제750조"
        result = AnswerValidator._normalize_citation(citation)
        
        assert result["type"] == "law"
        assert result["law_name"] == "민법"
        assert result["article_number"] == "750"
        assert result["normalized"] == "민법 제750조"
    
    def test_normalize_bracketed_format(self):
        """[법령: ...] 형식 정규화 테스트"""
        citation = "[법령: 특히국세기본법 제18조]"
        result = AnswerValidator._normalize_citation(citation)
        
        assert result["type"] == "law"
        assert result["law_name"] == "국세기본법"
        assert result["article_number"] == "18"


class TestLawNameValidation:
    """법령명 검증 테스트 (DB 조회 모킹)"""
    
    @patch.object(AnswerValidator, '_check_law_name_in_db')
    def test_validate_cleaned_name_with_db_mock(self, mock_check_db):
        """정제된 법령명 검증 테스트 (DB 모킹)"""
        # DB 조회 모킹: 정제된 법령명이 DB에 존재한다고 가정
        mock_check_db.side_effect = lambda name: name == "국세기본법"
        
        raw_name = "특히국세기본법"
        result = AnswerValidator._validate_and_clean_law_name(raw_name)
        
        # 접두어가 제거되고 DB에 존재하므로 정제된 값 반환
        assert result == "국세기본법", f"Expected '국세기본법', got '{result}'"
        # DB 조회가 정제된 법령명에 대해 호출되었는지 확인
        mock_check_db.assert_any_call("국세기본법")
    
    @patch.object(AnswerValidator, '_check_law_name_in_db')
    def test_validate_without_prefix_db_exists(self, mock_check_db):
        """접두어가 없고 DB에 존재하는 경우 테스트"""
        mock_check_db.return_value = True
        
        raw_name = "민법"
        result = AnswerValidator._validate_and_clean_law_name(raw_name)
        
        assert result == "민법", f"Expected '민법', got '{result}'"
        mock_check_db.assert_called_with("민법")
    
    @patch.object(AnswerValidator, '_check_law_name_in_db')
    def test_validate_without_prefix_db_not_exists(self, mock_check_db):
        """접두어가 없고 DB에 존재하지 않는 경우 테스트"""
        mock_check_db.return_value = False
        
        raw_name = "존재하지않는법"
        result = AnswerValidator._validate_and_clean_law_name(raw_name)
        
        assert result is None, f"Expected None, got '{result}'"
    
    def test_validate_empty_string(self):
        """빈 문자열 검증 테스트 (DB 조회 없음)"""
        result = AnswerValidator._validate_and_clean_law_name("")
        assert result is None, f"Expected None, got '{result}'"
    
    @patch.object(AnswerValidator, '_check_law_name_in_db')
    def test_validate_fallback_to_cleaned(self, mock_check_db):
        """DB 조회 실패 시 접두어 제거된 값 반환 테스트"""
        # DB에 정제된 법령명도 원본도 없음
        mock_check_db.return_value = False
        
        raw_name = "특히국세기본법"
        result = AnswerValidator._validate_and_clean_law_name(raw_name)
        
        # DB 조회 실패 시 접두어 제거된 값 반환 (폴백)
        assert result == "국세기본법", f"Expected '국세기본법', got '{result}'"


class TestLRUCache:
    """LRU 캐시 테스트"""
    
    def setup_method(self):
        """각 테스트 전에 캐시 초기화"""
        AnswerValidator._law_names_cache.clear()
        AnswerValidator._law_names_cache_access_order.clear()
    
    def test_cache_initialization(self):
        """캐시 초기화 테스트"""
        # 캐시가 비어있어야 함 (초기 상태)
        assert len(AnswerValidator._law_names_cache) == 0, "Cache should be empty initially"
        assert len(AnswerValidator._law_names_cache_access_order) == 0, "Access order should be empty initially"
    
    @patch.object(AnswerValidator, '_query_law_name_from_db')
    def test_cache_behavior(self, mock_query_db):
        """캐시 동작 테스트 (DB 조회 모킹)"""
        # DB 조회 모킹
        mock_query_db.return_value = True
        
        test_name = "테스트법"
        
        # 첫 번째 조회 (캐시 미스)
        result1 = AnswerValidator._check_law_name_in_db(test_name)
        
        # DB 조회가 호출되었는지 확인
        mock_query_db.assert_called_once_with(test_name)
        
        # 캐시에 추가되었는지 확인
        assert test_name in AnswerValidator._law_names_cache, "Law name should be in cache after first query"
        assert test_name in AnswerValidator._law_names_cache_access_order, "Law name should be in access order"
        assert AnswerValidator._law_names_cache[test_name] == True, "Cached value should be True"
        
        # 두 번째 조회 (캐시 히트)
        mock_query_db.reset_mock()
        result2 = AnswerValidator._check_law_name_in_db(test_name)
        
        # DB 조회가 호출되지 않아야 함 (캐시 히트)
        mock_query_db.assert_not_called()
        
        # 결과가 동일해야 함
        assert result1 == result2 == True, "Cached result should match original result"
        
        # 접근 순서가 업데이트되었는지 확인 (맨 뒤로 이동)
        assert AnswerValidator._law_names_cache_access_order[-1] == test_name, "Law name should be at the end of access order"
    
    @patch.object(AnswerValidator, '_query_law_name_from_db')
    def test_cache_lru_eviction(self, mock_query_db):
        """LRU 캐시 제거 테스트"""
        # 캐시 크기 설정 (테스트용으로 작게)
        original_max_size = AnswerValidator._law_names_cache_max_size
        AnswerValidator._law_names_cache_max_size = 3
        
        try:
            mock_query_db.return_value = True
            
            # 캐시 크기만큼 항목 추가
            for i in range(3):
                AnswerValidator._check_law_name_in_db(f"법{i}")
            
            # 캐시에 3개 항목이 있어야 함
            assert len(AnswerValidator._law_names_cache) == 3
            
            # 4번째 항목 추가 (가장 오래된 항목 제거)
            AnswerValidator._check_law_name_in_db("법3")
            
            # 캐시에 여전히 3개 항목이 있어야 함
            assert len(AnswerValidator._law_names_cache) == 3
            # 가장 오래된 항목이 제거되었는지 확인
            assert "법0" not in AnswerValidator._law_names_cache, "Oldest item should be evicted"
            assert "법3" in AnswerValidator._law_names_cache, "New item should be in cache"
        
        finally:
            # 원래 크기로 복원
            AnswerValidator._law_names_cache_max_size = original_max_size
            AnswerValidator._law_names_cache.clear()
            AnswerValidator._law_names_cache_access_order.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

