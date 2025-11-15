# -*- coding: utf-8 -*-
"""
Legal Search Tools 테스트
langgraph_core/tools/legal_search_tools.py 단위 테스트
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import List, Optional

from lawfirm_langgraph.langgraph_core.tools.legal_search_tools import (
    SearchPrecedentInput,
    SearchLawInput,
    SearchLegalTermInput,
    HybridSearchInput,
    get_search_engine,
    _search_precedent,
    _search_law,
    _search_legal_term,
    _hybrid_search
)


class TestSearchInputSchemas:
    """검색 입력 스키마 테스트"""
    
    def test_search_precedent_input(self):
        """판례 검색 입력 스키마 테스트"""
        input_data = SearchPrecedentInput(
            query="계약 해지",
            category="civil",
            max_results=5
        )
        
        assert input_data.query == "계약 해지"
        assert input_data.category == "civil"
        assert input_data.max_results == 5
    
    def test_search_law_input(self):
        """법령 검색 입력 스키마 테스트"""
        input_data = SearchLawInput(
            query="계약 해지",
            law_name="민법",
            article_number="543",
            max_results=5
        )
        
        assert input_data.query == "계약 해지"
        assert input_data.law_name == "민법"
        assert input_data.article_number == "543"
    
    def test_search_legal_term_input(self):
        """법률 용어 검색 입력 스키마 테스트"""
        input_data = SearchLegalTermInput(
            query="계약",
            max_results=5
        )
        
        assert input_data.query == "계약"
        assert input_data.max_results == 5
    
    def test_hybrid_search_input(self):
        """하이브리드 검색 입력 스키마 테스트"""
        input_data = HybridSearchInput(
            query="계약 해지",
            search_types=["law", "precedent"],
            max_results=10
        )
        
        assert input_data.query == "계약 해지"
        assert input_data.search_types == ["law", "precedent"]
        assert input_data.max_results == 10


class TestGetSearchEngine:
    """get_search_engine 테스트"""
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools._search_engine_instance', None)
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.HYBRID_SEARCH_AVAILABLE', True)
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.SearchHandler')
    def test_get_search_engine_success(self, mock_search_handler):
        """검색 엔진 가져오기 성공 테스트"""
        mock_instance = Mock()
        mock_search_handler.return_value = mock_instance
        
        result = get_search_engine()
        
        assert result == mock_instance
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools._search_engine_instance', None)
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.HYBRID_SEARCH_AVAILABLE', False)
    def test_get_search_engine_not_available(self):
        """검색 엔진 사용 불가 테스트"""
        result = get_search_engine()
        
        assert result is None


class TestSearchPrecedent:
    """_search_precedent 테스트"""
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_precedent_success(self, mock_get_engine):
        """판례 검색 성공 테스트"""
        mock_engine = Mock()
        mock_engine.search.return_value = {
            "results": [
                {"content": "판례 내용 1", "score": 0.9},
                {"content": "판례 내용 2", "score": 0.8}
            ],
            "total_results": 2
        }
        mock_get_engine.return_value = mock_engine
        
        result = _search_precedent("계약 해지", category="civil", max_results=5)
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 2
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_precedent_no_engine(self, mock_get_engine):
        """검색 엔진 없음 테스트"""
        mock_get_engine.return_value = None
        
        result = _search_precedent("계약 해지")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is False
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_precedent_exception(self, mock_get_engine):
        """예외 발생 테스트"""
        mock_engine = Mock()
        mock_engine.search.side_effect = Exception("Test error")
        mock_get_engine.return_value = mock_engine
        
        result = _search_precedent("계약 해지")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data


class TestSearchLaw:
    """_search_law 테스트"""
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_law_success(self, mock_get_engine):
        """법령 검색 성공 테스트"""
        mock_engine = Mock()
        mock_engine.search.return_value = {
            "results": [
                {"content": "법령 내용 1", "score": 0.9}
            ],
            "total_results": 1
        }
        mock_get_engine.return_value = mock_engine
        
        result = _search_law("계약 해지", law_name="민법", article_number="543")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is True
        assert "law_name" in data
        assert data["law_name"] == "민법"
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_law_no_engine(self, mock_get_engine):
        """검색 엔진 없음 테스트"""
        mock_get_engine.return_value = None
        
        result = _search_law("계약 해지")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is False


class TestSearchLegalTerm:
    """_search_legal_term 테스트"""
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_legal_term_success(self, mock_get_engine):
        """법률 용어 검색 성공 테스트"""
        mock_engine = Mock()
        mock_engine.search.return_value = {
            "results": [
                {"content": "용어 정의", "score": 0.9}
            ],
            "total_results": 1
        }
        mock_get_engine.return_value = mock_engine
        
        result = _search_legal_term("계약")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is True
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_search_legal_term_no_engine(self, mock_get_engine):
        """검색 엔진 없음 테스트"""
        mock_get_engine.return_value = None
        
        result = _search_legal_term("계약")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is False


class TestHybridSearch:
    """_hybrid_search 테스트"""
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_hybrid_search_success(self, mock_get_engine):
        """하이브리드 검색 성공 테스트"""
        mock_engine = Mock()
        mock_engine.search.return_value = {
            "results": [
                {"content": "결과 1", "score": 0.9},
                {"content": "결과 2", "score": 0.8}
            ],
            "total_results": 2
        }
        mock_get_engine.return_value = mock_engine
        
        result = _hybrid_search(
            "계약 해지",
            search_types=["law", "precedent"],
            max_results=10
        )
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is True
        assert len(data["results"]) == 2
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_hybrid_search_default_types(self, mock_get_engine):
        """기본 검색 타입 하이브리드 검색 테스트"""
        mock_engine = Mock()
        mock_engine.search.return_value = {
            "results": [],
            "total_results": 0
        }
        mock_get_engine.return_value = mock_engine
        
        result = _hybrid_search("계약 해지")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is True
    
    @patch('lawfirm_langgraph.langgraph_core.tools.legal_search_tools.get_search_engine')
    def test_hybrid_search_no_engine(self, mock_get_engine):
        """검색 엔진 없음 테스트"""
        mock_get_engine.return_value = None
        
        result = _hybrid_search("계약 해지")
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert data["success"] is False

