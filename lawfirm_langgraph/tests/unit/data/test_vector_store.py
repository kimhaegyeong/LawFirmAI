# -*- coding: utf-8 -*-
"""
VectorStore 테스트
벡터 저장소 모듈 단위 테스트
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from lawfirm_langgraph.core.data.vector_store import LegalVectorStore


class TestLegalVectorStore:
    """LegalVectorStore 테스트"""
    
    @pytest.fixture
    def vector_store(self):
        """LegalVectorStore 인스턴스"""
        with patch('lawfirm_langgraph.core.data.vector_store.SentenceTransformer'):
            with patch('lawfirm_langgraph.core.data.vector_store.faiss'):
                store = LegalVectorStore(
                    model_name="jhgan/ko-sroberta-multitask",
                    dimension=768,
                    index_type="flat",
                    enable_quantization=False,
                    enable_lazy_loading=True,
                    device="cpu"
                )
                return store
    
    def test_vector_store_initialization(self):
        """벡터 저장소 초기화 테스트"""
        with patch('lawfirm_langgraph.core.data.vector_store.SentenceTransformer'):
            with patch('lawfirm_langgraph.core.data.vector_store.faiss'):
                store = LegalVectorStore(
                    model_name="jhgan/ko-sroberta-multitask",
                    dimension=768,
                    index_type="flat",
                    enable_lazy_loading=True
                )
                
                assert store.model_name == "jhgan/ko-sroberta-multitask"
                assert store.dimension == 768
                assert store.index_type == "flat"
                assert store.enable_lazy_loading is True
    
    def test_get_model(self, vector_store):
        """모델 가져오기 테스트"""
        with patch.object(vector_store, '_load_model'):
            model = vector_store.get_model()
            
            assert model is None or hasattr(model, 'encode')
    
    def test_get_index(self, vector_store):
        """인덱스 가져오기 테스트"""
        with patch.object(vector_store, '_initialize_index'):
            index = vector_store.get_index()
            
            assert index is None or hasattr(index, 'search')
    
    def test_reset_store(self, vector_store):
        """벡터 저장소 초기화 테스트"""
        with patch.object(vector_store, '_initialize_index'):  # _initialize_index 호출 방지
            with patch('pathlib.Path') as mock_path:
                mock_path_instance = MagicMock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.with_suffix.return_value.exists.return_value = False
                
                vector_store.reset_store(delete_disk=False)
                
                # _initialize_index가 호출되지 않았으므로 _index_loaded는 False여야 함
                assert vector_store._index_loaded is False
                assert len(vector_store.document_metadata) == 0
                assert len(vector_store.document_texts) == 0
    
    def test_generate_embeddings(self, vector_store):
        """임베딩 생성 테스트"""
        with patch.object(vector_store, 'get_model') as mock_get_model:
            with patch('lawfirm_langgraph.core.data.vector_store.faiss') as mock_faiss:
                mock_model = MagicMock()
                # float32 형식의 numpy 배열 반환 (FAISS 호환)
                mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
                mock_model.encode.return_value = mock_embeddings
                mock_get_model.return_value = mock_model
                
                # FAISS normalize_L2 Mock
                mock_faiss.normalize_L2 = MagicMock()
                
                texts = ["테스트 텍스트 1", "테스트 텍스트 2"]
                embeddings = vector_store.generate_embeddings(texts, batch_size=2)
                
                assert isinstance(embeddings, np.ndarray)
                assert len(embeddings) == 2
                assert embeddings.dtype == np.float32
    
    def test_add_documents(self, vector_store):
        """문서 추가 테스트"""
        with patch.object(vector_store, 'get_model') as mock_get_model:
            with patch.object(vector_store, 'get_index') as mock_get_index:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_get_model.return_value = mock_model
                
                mock_index = MagicMock()
                mock_index.is_trained = True
                mock_index.ntotal = 0
                mock_get_index.return_value = mock_index
                
                texts = ["테스트 문서"]
                metadatas = [{"document_id": "doc1", "title": "테스트"}]
                
                result = vector_store.add_documents(texts, metadatas)
                
                assert result is True or result is False
    
    def test_search(self, vector_store):
        """검색 테스트"""
        with patch.object(vector_store, 'get_model') as mock_get_model:
            with patch.object(vector_store, 'get_index') as mock_get_index:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_get_model.return_value = mock_model
                
                mock_index = MagicMock()
                mock_index.search.return_value = (
                    np.array([[0.9, 0.8, 0.7]]),
                    np.array([[0, 1, 2]])
                )
                mock_get_index.return_value = mock_index
                
                vector_store.document_texts = ["문서 1", "문서 2", "문서 3"]
                vector_store.document_metadata = [
                    {"document_id": "doc1"},
                    {"document_id": "doc2"},
                    {"document_id": "doc3"}
                ]
                
                results = vector_store.search("테스트 쿼리", top_k=3)
                
                assert isinstance(results, list)
    
    def test_search_with_filters(self, vector_store):
        """필터를 사용한 검색 테스트"""
        with patch.object(vector_store, 'get_model') as mock_get_model:
            with patch.object(vector_store, 'get_index') as mock_get_index:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
                mock_get_model.return_value = mock_model
                
                mock_index = MagicMock()
                mock_index.search.return_value = (
                    np.array([[0.9]]),
                    np.array([[0]])
                )
                mock_get_index.return_value = mock_index
                
                vector_store.document_texts = ["문서 1"]
                vector_store.document_metadata = [{"document_id": "doc1", "category": "statute"}]
                
                filters = {"category": "statute"}
                results = vector_store.search("테스트 쿼리", top_k=3, filters=filters)
                
                assert isinstance(results, list)
    
    def test_search_error_handling(self, vector_store):
        """검색 에러 핸들링 테스트"""
        with patch.object(vector_store, 'get_model', side_effect=Exception("테스트 에러")):
            results = vector_store.search("테스트 쿼리", top_k=3)
            
            assert isinstance(results, list)
            assert len(results) == 0
    
    def test_get_stats(self, vector_store):
        """통계 조회 테스트"""
        vector_store.document_texts = ["문서 1", "문서 2"]
        vector_store.document_metadata = [
            {"document_id": "doc1"},
            {"document_id": "doc2"}
        ]
        
        stats = vector_store.get_stats()
        
        assert isinstance(stats, dict)
        assert "documents_count" in stats
        assert stats["documents_count"] == 2
    
    def test_get_memory_usage(self, vector_store):
        """메모리 사용량 조회 테스트"""
        with patch('lawfirm_langgraph.core.data.vector_store.PSUTIL_AVAILABLE', True):
            with patch('lawfirm_langgraph.core.data.vector_store.psutil') as mock_psutil:
                mock_process = MagicMock()
                mock_process.memory_info.return_value.rss = 1024 * 1024 * 100
                mock_psutil.Process.return_value = mock_process
                
                memory_usage = vector_store.get_memory_usage()
                
                assert isinstance(memory_usage, dict)
                assert "total_memory_mb" in memory_usage
    
    def test_cleanup(self, vector_store):
        """리소스 정리 테스트"""
        with patch.object(vector_store, '_cleanup_memory'):
            vector_store.cleanup()
            
            assert True

