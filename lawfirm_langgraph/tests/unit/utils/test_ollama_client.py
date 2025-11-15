# -*- coding: utf-8 -*-
"""
Ollama Client 테스트
core/utils/ollama_client.py 단위 테스트
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import ConnectionError, Timeout, RequestException

from lawfirm_langgraph.core.utils.ollama_client import OllamaClient


class TestOllamaClient:
    """OllamaClient 테스트"""
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_init_success(self, mock_session_class):
        """초기화 성공 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model", base_url="http://localhost:11434")
        
        assert client.model == "test-model"
        assert client.base_url == "http://localhost:11434"
        assert client.session is not None
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_init_connection_error(self, mock_session_class):
        """초기화 연결 오류 테스트"""
        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("Connection failed")
        mock_session_class.return_value = mock_session
        
        with pytest.raises(ConnectionError):
            OllamaClient(model="test-model", base_url="http://localhost:11434")
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_init_server_error(self, mock_session_class):
        """초기화 서버 오류 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        with pytest.raises(ConnectionError):
            OllamaClient(model="test-model", base_url="http://localhost:11434")
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_single(self, mock_session_class):
        """단일 응답 생성 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        result = client.generate("Test prompt", stream=False)
        
        assert result == "Test response"
        mock_session.post.assert_called_once()
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_with_system_prompt(self, mock_session_class):
        """시스템 프롬프트 포함 생성 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        result = client.generate(
            "Test prompt",
            system_prompt="You are a helpful assistant",
            stream=False
        )
        
        assert result == "Test response"
        call_args = mock_session.post.call_args
        assert call_args[1]['json']['system'] == "You are a helpful assistant"
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_stream(self, mock_session_class):
        """스트리밍 응답 생성 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "Test", "done": false}',
            b'{"response": " response", "done": false}',
            b'{"response": "", "done": true}'
        ]
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        results = list(client.generate("Test prompt", stream=True))
        
        assert len(results) == 2
        assert "Test" in results
        assert " response" in results
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_timeout(self, mock_session_class):
        """타임아웃 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.side_effect = Timeout("Request timeout")
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        
        with pytest.raises(Timeout):
            client.generate("Test prompt", stream=False)
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_request_exception(self, mock_session_class):
        """요청 예외 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.side_effect = RequestException("Request failed")
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        
        with pytest.raises(RequestException):
            client.generate("Test prompt", stream=False)
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_qa_pairs(self, mock_session_class):
        """Q&A 쌍 생성 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '[{"question": "Test question", "answer": "Test answer", "type": "concept"}]'
        }
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        qa_pairs = client.generate_qa_pairs("Test context", qa_count=1)
        
        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Test question"
        assert qa_pairs[0]["answer"] == "Test answer"
        assert qa_pairs[0]["type"] == "concept"
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_qa_pairs_invalid_json(self, mock_session_class):
        """Q&A 쌍 생성 - 잘못된 JSON 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Invalid JSON response"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        qa_pairs = client.generate_qa_pairs("Test context", qa_count=1)
        
        assert len(qa_pairs) == 0
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_generate_qa_pairs_exception(self, mock_session_class):
        """Q&A 쌍 생성 예외 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.side_effect = Exception("Test error")
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        qa_pairs = client.generate_qa_pairs("Test context", qa_count=1)
        
        assert len(qa_pairs) == 0
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_test_connection_success(self, mock_session_class):
        """연결 테스트 성공 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello"}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        result = client.test_connection()
        
        assert result is True
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_test_connection_failure(self, mock_session_class):
        """연결 테스트 실패 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": ""}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        result = client.test_connection()
        
        assert result is False
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_test_connection_exception(self, mock_session_class):
        """연결 테스트 예외 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.side_effect = Exception("Test error")
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        result = client.test_connection()
        
        assert result is False
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_get_model_info(self, mock_session_class):
        """모델 정보 조회 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test-model", "size": 1000}
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        info = client.get_model_info()
        
        assert info["name"] == "test-model"
        assert info["size"] == 1000
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_get_model_info_exception(self, mock_session_class):
        """모델 정보 조회 예외 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.side_effect = Exception("Test error")
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        info = client.get_model_info()
        
        assert info == {}
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_parse_qa_response_valid(self, mock_session_class):
        """Q&A 응답 파싱 - 유효한 JSON 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        response = '[{"question": "Q1", "answer": "A1", "type": "type1"}]'
        qa_pairs = client._parse_qa_response(response)
        
        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Q1"
        assert qa_pairs[0]["answer"] == "A1"
        assert qa_pairs[0]["type"] == "type1"
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_parse_qa_response_invalid(self, mock_session_class):
        """Q&A 응답 파싱 - 잘못된 JSON 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        response = "Invalid response"
        qa_pairs = client._parse_qa_response(response)
        
        assert len(qa_pairs) == 0
    
    @patch('lawfirm_langgraph.core.utils.ollama_client.requests.Session')
    def test_parse_qa_response_missing_keys(self, mock_session_class):
        """Q&A 응답 파싱 - 필수 키 누락 테스트"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.get.return_value = mock_response
        mock_session.post.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        client = OllamaClient(model="test-model")
        response = '[{"question": "Q1"}]'
        qa_pairs = client._parse_qa_response(response)
        
        assert len(qa_pairs) == 0

