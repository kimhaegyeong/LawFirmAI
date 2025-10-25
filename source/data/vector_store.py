# -*- coding: utf-8 -*-
"""
Vector Store
벡터 임베딩 생성 및 관리 모듈

법률 문서의 벡터 임베딩을 생성하고 FAISS를 사용하여 벡터 검색을 수행합니다.
"""

import logging
import json
import numpy as np
import gc
import psutil
import threading
import time
import weakref
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime
from dataclasses import dataclass

# FAISS 관련 import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Please install faiss-cpu or faiss-gpu")

# Sentence-BERT 관련 import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Please install sentence-transformers")

# FlagEmbedding (BGE-M3) 관련 import
try:
    from FlagEmbedding import FlagReranker, FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    logging.warning("FlagEmbedding not available. Please install FlagEmbedding")

# PyTorch 관련 import (양자화용)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for quantization")

logger = logging.getLogger(__name__)


class LegalVectorStore:
    """법률 문서 벡터 스토어 클래스"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat",
                 enable_quantization: bool = True,
                 enable_lazy_loading: bool = True,
                 memory_threshold_mb: int = 512,  # 메모리 임계값을 512MB로 긴급 낮춤
                 use_gpu: bool = True, device: str = "auto"):
        """
        벡터 스토어 초기화
        
        Args:
            model_name: 사용할 임베딩 모델명 (Sentence-BERT 또는 BGE-M3)
            dimension: 벡터 차원
            index_type: FAISS 인덱스 타입 ("flat", "ivf", "hnsw")
            enable_quantization: Float16 양자화 활성화
            enable_lazy_loading: 지연 로딩 활성화
            memory_threshold_mb: 메모리 임계값 (MB)
            use_gpu: GPU 사용 여부
            device: 사용할 디바이스 ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        self.enable_quantization = enable_quantization
        self.enable_lazy_loading = enable_lazy_loading
        self.memory_threshold_mb = memory_threshold_mb
        self.use_gpu = use_gpu
        
        # 디바이스 설정
        if device == "auto":
            if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
                self.logger = logging.getLogger(__name__)
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                self.device = "cpu"
                if use_gpu:
                    self.logger = logging.getLogger(__name__)
                    self.logger.warning("GPU requested but not available, using CPU")
        else:
            self.device = device
        
        # 모델 관련
        self.model = None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        
        # 인덱스 관련
        self.index = None
        self.document_metadata = []
        self.document_texts = []
        self._index_loaded = False
        self._index_lock = threading.Lock()
        
        # 메모리 관리 (긴급 최적화 - 더 적극적인 설정)
        self._memory_cache = weakref.WeakValueDictionary()
        self._last_memory_check = time.time()
        self._memory_check_interval = 15  # 15초마다 메모리 체크 (더 자주)
        self._memory_threshold_mb = 512   # 512MB 임계값 (더 낮게 설정)
        self._aggressive_cleanup_threshold = 768  # 768MB에서 적극적 정리
        
        # 모델 타입 감지
        self.is_bge_model = "bge-m3" in model_name.lower()
        
        # 로거 초기화 (디바이스 설정 후)
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"LegalVectorStore initialized with model: {model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model type: {'BGE-M3' if self.is_bge_model else 'Sentence-BERT'}")
        self.logger.info(f"Quantization: {'Enabled' if enable_quantization else 'Disabled'}")
        self.logger.info(f"Lazy Loading: {'Enabled' if enable_lazy_loading else 'Disabled'}")
        
        # 지연 로딩이 비활성화된 경우에만 즉시 로딩
        if not self.enable_lazy_loading:
            self._load_model()
            self._initialize_index()
    
    def _check_memory_usage(self):
        """메모리 사용량 체크 및 정리 (개선된 버전)"""
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return
        
        self._last_memory_check = current_time
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            
            # 적극적 정리 임계값 초과 시
            if memory_mb > self._aggressive_cleanup_threshold:
                self.logger.warning(f"Memory usage critical: {memory_mb:.2f} MB > {self._aggressive_cleanup_threshold} MB")
                self._aggressive_memory_cleanup()
            # 일반 임계값 초과 시
            elif memory_mb > self._memory_threshold_mb:
                self.logger.warning(f"Memory usage exceeded threshold: {memory_mb:.2f} MB > {self._memory_threshold_mb} MB")
                self._cleanup_memory()
                
        except Exception as e:
            self.logger.error(f"Memory check failed: {e}")
    
    def _aggressive_memory_cleanup(self):
        """적극적 메모리 정리"""
        self.logger.warning("Starting aggressive memory cleanup...")
        
        # 1. 가비지 컬렉션 강제 실행 (여러 번)
        for i in range(3):
            collected = gc.collect()
            self.logger.info(f"Garbage collection cycle {i+1}: collected {collected} objects")
        
        # 2. 캐시 완전 정리
        if hasattr(self, '_memory_cache'):
            cache_size = len(self._memory_cache)
            self._memory_cache.clear()
            self.logger.info(f"Cleared {cache_size} cached items")
        
        # 3. 모델 언로드 (필요시)
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
            self._model_loaded = False
            self.logger.info("Model unloaded to free memory")
        
        # 4. PyTorch 캐시 정리
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("PyTorch CUDA cache cleared")
        
        # 5. 메모리 사용량 재확인
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            self.logger.info(f"Memory after aggressive cleanup: {memory_mb:.2f} MB")
        except Exception as e:
            self.logger.error(f"Failed to check memory after aggressive cleanup: {e}")
    
    def _cleanup_memory(self):
        """메모리 정리"""
        self.logger.info("Starting memory cleanup...")
        
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        self.logger.info(f"Garbage collection collected {collected} objects")
        
        # 캐시 정리
        if hasattr(self, '_memory_cache'):
            cache_size = len(self._memory_cache)
            self._memory_cache.clear()
            self.logger.info(f"Cleared {cache_size} cached items")
        
        # 메모리 사용량 재확인
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            self.logger.info(f"Memory after cleanup: {memory_mb:.2f} MB")
        except Exception as e:
            self.logger.error(f"Failed to check memory after cleanup: {e}")
    
    def _load_model(self):
        """임베딩 모델 로딩 (양자화 적용)"""
        if self._model_loaded:
            return
        
        with self._model_lock:
            if self._model_loaded:
                return
            
            try:
                if self.is_bge_model:
                    # BGE-M3 모델 로딩
                    if not FLAG_EMBEDDING_AVAILABLE:
                        raise ImportError("FlagEmbedding is required for BGE-M3 models but not installed")
                    
                    self.model = FlagModel(self.model_name, query_instruction_for_retrieval="")
                    self.logger.info(f"BGE-M3 model loaded: {self.model_name}")
                else:
                    # Sentence-BERT 모델 로딩
                    if not SENTENCE_TRANSFORMERS_AVAILABLE:
                        raise ImportError("SentenceTransformers is required but not installed")
                    
                    # 디바이스 설정에 따라 모델 로딩
                    if self.device == "cuda":
                        self.model = SentenceTransformer(self.model_name, device=self.device)
                        self.logger.info(f"Sentence-BERT model loaded on GPU: {self.model_name}")
                    else:
                        self.model = SentenceTransformer(self.model_name)
                        self.logger.info(f"Sentence-BERT model loaded on CPU: {self.model_name}")
                    
                    # Float16 양자화 적용 (GPU에서 더 효과적)
                    if self.enable_quantization and TORCH_AVAILABLE:
                        try:
                            if self.device == "cuda":
                                # GPU에서 양자화 적용
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                                    self.model.model = self.model.model.half()
                                    self.logger.info("Model quantized to Float16 on GPU")
                                elif hasattr(self.model, 'half'):
                                    self.model = self.model.half()
                                    self.logger.info("Model quantized to Float16 on GPU")
                            else:
                                # CPU에서 양자화 적용
                                if hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                                    self.model.model = self.model.model.half()
                                    self.logger.info("Model quantized to Float16 on CPU")
                                elif hasattr(self.model, 'half'):
                                    self.model = self.model.half()
                                    self.logger.info("Model quantized to Float16 on CPU")
                        except Exception as e:
                            self.logger.warning(f"Quantization failed: {e}")
                
                self._model_loaded = True
                
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def _initialize_index(self):
        """FAISS 인덱스 초기화"""
        if self._index_loaded:
            return
        
        with self._index_lock:
            if self._index_loaded:
                return
            
            try:
                if not FAISS_AVAILABLE:
                    raise ImportError("FAISS is required but not installed")
                
                if self.index_type == "flat":
                    self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
                elif self.index_type == "ivf":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                elif self.index_type == "hnsw":
                    self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
                
                self.logger.info(f"FAISS index initialized: {self.index_type}")
                self._index_loaded = True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize FAISS index: {e}")
                raise
    
    def get_model(self):
        """모델 가져오기 (지연 로딩)"""
        if self.enable_lazy_loading and not self._model_loaded:
            self._load_model()
        
        self._check_memory_usage()
        return self.model
    
    def get_index(self):
        """인덱스 가져오기 (지연 로딩)"""
        if self.enable_lazy_loading and not self._index_loaded:
            self._initialize_index()
        
        self._check_memory_usage()
        return self.index
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """텍스트 리스트에 대한 임베딩 생성 (배치 처리)"""
        model = self.get_model()
        
        if not model:
            raise RuntimeError("Model not loaded")
        
        try:
            # 배치 처리로 메모리 효율성 향상
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                if self.is_bge_model:
                    batch_embeddings = model.encode(batch_texts)
                    if hasattr(batch_embeddings, 'numpy'):
                        batch_embeddings = batch_embeddings.numpy()
                    elif not isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = np.array(batch_embeddings)
                else:
                    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
                
                # Float16 양자화 적용
                if self.enable_quantization:
                    batch_embeddings = batch_embeddings.astype(np.float16)
                
                embeddings.append(batch_embeddings)
                
                # 메모리 체크
                self._check_memory_usage()
            
            # 모든 배치 결과 결합
            result = np.vstack(embeddings)
            
            # L2 정규화 (Float32로 변환 후 정규화)
            if self.enable_quantization:
                result = result.astype(np.float32)
            faiss.normalize_L2(result)
            
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """문서들을 벡터 스토어에 추가 (하이브리드 검색용)"""
        try:
            if not texts or not metadatas:
                self.logger.warning("No texts or metadatas to process")
                return False
            
            if len(texts) != len(metadatas):
                self.logger.error("Texts and metadatas must have the same length")
                return False
            
            # 임베딩 생성
            embeddings = self.generate_embeddings(texts)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            
            # 인덱스 가져오기 (지연 로딩 지원)
            index = self.get_index()
            
            # 인덱스에 추가
            if index.is_trained:
                index.add(embeddings)
            else:
                # IVF 인덱스의 경우 훈련 필요
                if self.index_type == "ivf":
                    index.train(embeddings)
                    index.add(embeddings)
                else:
                    index.add(embeddings)
            
            # 메타데이터 저장
            self.document_metadata.extend(metadatas)
            self.document_texts.extend(texts)
            
            self.logger.info(f"Added {len(texts)} documents to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    def add_documents_legacy(self, documents: List[Dict[str, Any]]) -> bool:
        """문서들을 벡터 스토어에 추가"""
        try:
            # 텍스트 추출
            texts = []
            metadata = []
            
            for doc in documents:
                # 청크별로 처리
                chunks = doc.get('chunks', [])
                if chunks:
                    for chunk in chunks:
                        texts.append(chunk.get('text', ''))
                        metadata.append({
                            'document_id': doc.get('id', ''),
                            'document_type': doc.get('type', 'unknown'),
                            'chunk_id': chunk.get('id', ''),
                            'chunk_start': chunk.get('start_pos', 0),
                            'chunk_end': chunk.get('end_pos', 0),
                            'law_name': doc.get('law_name', doc.get('case_name', '')),
                            'category': doc.get('category', ''),
                            'entities': chunk.get('entities', {})
                        })
                else:
                    # 청크가 없는 경우 전체 텍스트 사용
                    text = doc.get('cleaned_content', '')
                    if text:
                        texts.append(text)
                        metadata.append({
                            'document_id': doc.get('id', ''),
                            'document_type': doc.get('type', 'unknown'),
                            'chunk_id': 'full',
                            'chunk_start': 0,
                            'chunk_end': len(text),
                            'law_name': doc.get('law_name', doc.get('case_name', '')),
                            'category': doc.get('category', ''),
                            'entities': doc.get('entities', {})
                        })
            
            if not texts:
                self.logger.warning("No texts to process")
                return False
            
            # 임베딩 생성
            embeddings = self.generate_embeddings(texts)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            
            # 인덱스에 추가
            if self.index.is_trained:
                self.index.add(embeddings)
            else:
                # IVF 인덱스의 경우 훈련 필요
                if self.index_type == "ivf":
                    self.index.train(embeddings)
                    self.index.add(embeddings)
                else:
                    self.index.add(embeddings)
            
            # 메타데이터 저장
            self.document_metadata.extend(metadata)
            self.document_texts.extend(texts)
            
            self.logger.info(f"Added {len(texts)} document chunks to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None, enhanced: bool = True) -> List[Dict[str, Any]]:
        """벡터 검색 수행 (하이브리드 검색용)"""
        # 메모리 사용량 체크
        self._check_memory_usage()
        
        try:
            model = self.get_model()
            index = self.get_index()
            
            if not model or not index:
                raise RuntimeError("Model or index not initialized")
            
            if index.ntotal == 0:
                self.logger.warning("Index is empty")
                return []
            
            # 쿼리 임베딩 생성
            if self.is_bge_model:
                query_embedding = model.encode([query])
                if hasattr(query_embedding, 'numpy'):
                    query_embedding = query_embedding.numpy()
                elif not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)
            else:
                query_embedding = model.encode([query], convert_to_numpy=True)
            
            # Float16 양자화 적용 후 Float32로 변환하여 정규화
            if self.enable_quantization:
                query_embedding = query_embedding.astype(np.float32)
            
            faiss.normalize_L2(query_embedding)
            
            # 검색 수행 (필터링을 위해 더 많은 결과 가져옴)
            search_k = min(top_k * 5, index.ntotal) if index.ntotal > 0 else top_k
            scores, indices = index.search(query_embedding, search_k)
            
            # 결과 처리
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS에서 -1은 유효하지 않은 인덱스
                    continue
                
                # 필터링 적용
                if filters:
                    match = True
                    for key, value in filters.items():
                        if idx < len(self.document_metadata) and key in self.document_metadata[idx] and self.document_metadata[idx][key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                result = {
                    'score': float(score),
                    'text': self.document_texts[idx] if idx < len(self.document_texts) else '',
                    'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            # 향상된 검색 적용
            if enhanced and results:
                results = self._apply_enhanced_scoring(query, results)
            
            self.logger.info(f"Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _apply_enhanced_scoring(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """향상된 점수 계산 적용"""
        try:
            # 법률 용어 확장 사전
            legal_expansions = {
                "손해배상": ["손해배상", "배상", "피해보상", "손실보상", "금전적 손해", "물질적 손해", "정신적 손해"],
                "이혼": ["이혼", "혼인해소", "혼인무효", "별거", "가정파탄", "부부갈등", "혼인관계"],
                "계약": ["계약", "계약서", "약정", "합의", "계약관계", "계약체결", "계약이행"],
                "변호인": ["변호인", "변호사", "법정변호인", "국선변호인", "선임변호인", "변호"],
                "형사처벌": ["형사처벌", "형사처분", "형사제재", "형사처리", "형사처분", "형사처벌"],
                "재산분할": ["재산분할", "재산분배", "재산정리", "재산처분", "재산분할", "재산분할"],
                "친권": ["친권", "친권자", "친권행사", "친권포기", "친권상실", "친권"],
                "양육비": ["양육비", "양육비용", "양육비지급", "양육비부담", "양육비지원", "양육비"],
                "소송": ["소송", "소송절차", "소송진행", "소송제기", "소송제출", "소송"],
                "법원": ["법원", "법정", "재판부", "법원판결", "법원결정", "법원"],
                "청구": ["청구", "청구권", "청구서", "청구사유", "청구이유"],
                "요건": ["요건", "요소", "조건", "기준", "요구사항"]
            }
            
            # 카테고리별 가중치
            category_weights = {
                "civil": 1.1,
                "criminal": 1.1, 
                "family": 1.1,
                "constitutional": 1.3,
                "assembly_law": 1.2
            }
            
            # 키워드 매칭 가중치
            keyword_weights = {
                "exact_match": 2.0,      # 정확한 매칭
                "partial_match": 1.5,    # 부분 매칭
                "synonym_match": 1.3     # 동의어 매칭
            }
            
            enhanced_results = []
            for result in results:
                # 기본 점수
                base_score = result.get('score', 0.0)
                
                # 키워드 매칭 점수 계산
                text = result.get('text', '')
                keyword_score = self._calculate_keyword_score(query, text, legal_expansions, keyword_weights)
                
                # 카테고리 부스트
                metadata = result.get('metadata', {})
                category_boost = category_weights.get(metadata.get('category', 'unknown'), 1.0)
                
                # 품질 부스트
                quality_score = metadata.get('parsing_quality_score', 0.0)
                if isinstance(quality_score, (int, float)):
                    quality_boost = 0.9 + (quality_score * 0.1)
                else:
                    quality_boost = 0.95
                
                # 길이 부스트
                text_length = len(text)
                if 100 <= text_length <= 1000:
                    length_boost = 1.1
                elif 50 <= text_length < 100 or 1000 < text_length <= 2000:
                    length_boost = 1.0
                else:
                    length_boost = 0.9
                
                # 최종 점수 계산 (매우 보수적인 가중치)
                final_score = (
                    base_score * 0.95 +          # 기본 벡터 점수 95% (매우 높게)
                    keyword_score * 0.03 +       # 키워드 매칭 3% (매우 낮게)
                    (category_boost - 1.0) * 0.01 +  # 카테고리 부스트 1% (매우 낮게)
                    (quality_boost - 0.95) * 0.005 + # 품질 부스트 0.5% (매우 낮게)
                    (length_boost - 1.0) * 0.005     # 길이 부스트 0.5% (매우 낮게)
                )
                
                # 결과에 개선된 점수 추가
                enhanced_result = result.copy()
                enhanced_result['enhanced_score'] = final_score
                enhanced_result['base_score'] = base_score
                enhanced_result['keyword_score'] = keyword_score
                enhanced_result['category_boost'] = category_boost
                enhanced_result['quality_boost'] = quality_boost
                enhanced_result['length_boost'] = length_boost
                
                enhanced_results.append(enhanced_result)
            
            # 개선된 점수로 정렬
            enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Enhanced scoring failed: {e}")
            return results
    
    def _calculate_keyword_score(self, query: str, text: str, legal_expansions: Dict, keyword_weights: Dict) -> float:
        """키워드 매칭 점수 계산"""
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        exact_matches = 0
        partial_matches = 0
        synonym_matches = 0
        
        for term in query_terms:
            # 정확한 매칭
            if term in text_lower:
                exact_matches += 1
            else:
                # 부분 매칭 (2글자 이상)
                if len(term) >= 2:
                    for i in range(len(text_lower) - len(term) + 1):
                        if text_lower[i:i+len(term)] == term:
                            partial_matches += 1
                            break
                
                # 동의어 매칭
                for key, expansions in legal_expansions.items():
                    if term in expansions:
                        for expansion in expansions:
                            if expansion in text_lower:
                                synonym_matches += 1
                                break
                        break
        
        # 가중치 적용한 점수 계산
        keyword_score = (
            exact_matches * keyword_weights["exact_match"] +
            partial_matches * keyword_weights["partial_match"] +
            synonym_matches * keyword_weights["synonym_match"]
        ) / len(query_terms) if query_terms else 0
        
        return min(keyword_score, 2.0)  # 최대 2.0으로 제한
    
    def search_legacy(self, query: str, top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """벡터 검색 수행 (레거시)"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 쿼리 임베딩 생성
            if self.is_bge_model:
                query_embedding = self.model.encode([query])
                if hasattr(query_embedding, 'numpy'):
                    query_embedding = query_embedding.numpy()
                elif not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)
                faiss.normalize_L2(query_embedding)
            else:
                query_embedding = self.model.encode([query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
            
            # 검색 수행
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # 결과 처리
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS에서 -1은 유효하지 않은 인덱스
                    continue
                
                if score >= score_threshold:
                    result = {
                        'score': float(score),
                        'text': self.document_texts[idx],
                        'metadata': self.document_metadata[idx]
                    }
                    results.append(result)
            
            self.logger.info(f"Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def search_by_law(self, law_name: str, query: str = "", top_k: int = 10) -> List[Dict[str, Any]]:
        """특정 법령 내에서 검색"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 해당 법령의 문서들만 필터링
            law_indices = []
            for i, metadata in enumerate(self.document_metadata):
                if law_name in metadata.get('law_name', ''):
                    law_indices.append(i)
            
            if not law_indices:
                self.logger.warning(f"No documents found for law: {law_name}")
                return []
            
            # 쿼리 임베딩 생성
            search_query = f"{law_name} {query}" if query else law_name
            if self.is_bge_model:
                query_embedding = self.model.encode([search_query], normalize_embeddings=True)
                if hasattr(query_embedding, 'numpy'):
                    query_embedding = query_embedding.numpy()
                elif not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)
            else:
                query_embedding = self.model.encode([search_query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
            
            # 해당 법령의 문서들만 검색
            law_texts = [self.document_texts[i] for i in law_indices]
            if self.is_bge_model:
                law_embeddings = self.model.encode(law_texts)
                if hasattr(law_embeddings, 'numpy'):
                    law_embeddings = law_embeddings.numpy()
                elif not isinstance(law_embeddings, np.ndarray):
                    law_embeddings = np.array(law_embeddings)
                faiss.normalize_L2(law_embeddings)
            else:
                law_embeddings = self.model.encode(law_texts, convert_to_numpy=True)
                faiss.normalize_L2(law_embeddings)
            
            # 유사도 계산
            similarities = np.dot(query_embedding, law_embeddings.T)[0]
            
            # 결과 정렬
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in sorted_indices:
                original_idx = law_indices[idx]
                result = {
                    'score': float(similarities[idx]),
                    'text': self.document_texts[original_idx],
                    'metadata': self.document_metadata[original_idx]
                }
                results.append(result)
            
            self.logger.info(f"Law-specific search completed: {len(results)} results found for {law_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Law-specific search failed: {e}")
            return []
    
    def search_by_category(self, category: str, query: str = "", top_k: int = 10) -> List[Dict[str, Any]]:
        """특정 카테고리 내에서 검색"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 해당 카테고리의 문서들만 필터링
            category_indices = []
            for i, metadata in enumerate(self.document_metadata):
                if category in metadata.get('category', ''):
                    category_indices.append(i)
            
            if not category_indices:
                self.logger.warning(f"No documents found for category: {category}")
                return []
            
            # 쿼리 임베딩 생성
            search_query = f"{category} {query}" if query else category
            if self.is_bge_model:
                query_embedding = self.model.encode([search_query], normalize_embeddings=True)
                if hasattr(query_embedding, 'numpy'):
                    query_embedding = query_embedding.numpy()
                elif not isinstance(query_embedding, np.ndarray):
                    query_embedding = np.array(query_embedding)
            else:
                query_embedding = self.model.encode([search_query], convert_to_numpy=True)
                faiss.normalize_L2(query_embedding)
            
            # 해당 카테고리의 문서들만 검색
            category_texts = [self.document_texts[i] for i in category_indices]
            if self.is_bge_model:
                category_embeddings = self.model.encode(category_texts)
                if hasattr(category_embeddings, 'numpy'):
                    category_embeddings = category_embeddings.numpy()
                elif not isinstance(category_embeddings, np.ndarray):
                    category_embeddings = np.array(category_embeddings)
                faiss.normalize_L2(category_embeddings)
            else:
                category_embeddings = self.model.encode(category_texts, convert_to_numpy=True)
                faiss.normalize_L2(category_embeddings)
            
            # 유사도 계산
            similarities = np.dot(query_embedding, category_embeddings.T)[0]
            
            # 결과 정렬
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in sorted_indices:
                original_idx = category_indices[idx]
                result = {
                    'score': float(similarities[idx]),
                    'text': self.document_texts[original_idx],
                    'metadata': self.document_metadata[original_idx]
                }
                results.append(result)
            
            self.logger.info(f"Category-specific search completed: {len(results)} results found for {category}")
            return results
            
        except Exception as e:
            self.logger.error(f"Category-specific search failed: {e}")
            return []
    
    def get_similar_documents(self, document_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """특정 문서와 유사한 문서들 검색"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 문서 ID로 인덱스 찾기
            doc_index = None
            for i, metadata in enumerate(self.document_metadata):
                if metadata.get('document_id') == document_id:
                    doc_index = i
                    break
            
            if doc_index is None:
                self.logger.warning(f"Document not found: {document_id}")
                return []
            
            # 해당 문서의 임베딩 추출
            if self.is_bge_model:
                doc_embedding = self.model.encode([self.document_texts[doc_index]])
                if hasattr(doc_embedding, 'numpy'):
                    doc_embedding = doc_embedding.numpy()
                elif not isinstance(doc_embedding, np.ndarray):
                    doc_embedding = np.array(doc_embedding)
                faiss.normalize_L2(doc_embedding)
            else:
                doc_embedding = self.model.encode([self.document_texts[doc_index]], convert_to_numpy=True)
                faiss.normalize_L2(doc_embedding)
            
            # 검색 수행 (자기 자신 제외)
            scores, indices = self.index.search(doc_embedding, top_k + 1)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx == doc_index:  # 유효하지 않은 인덱스나 자기 자신 제외
                    continue
                
                result = {
                    'score': float(score),
                    'text': self.document_texts[idx],
                    'metadata': self.document_metadata[idx]
                }
                results.append(result)
            
            self.logger.info(f"Similar documents search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            self.logger.error(f"Similar documents search failed: {e}")
            return []
    
    def save_index(self, filepath: str = "data/embeddings/legal_vector_index") -> bool:
        """인덱스와 메타데이터 저장"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # FAISS 인덱스 저장
            faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
            
            # 메타데이터 저장
            metadata = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'document_count': len(self.document_metadata),
                'created_at': datetime.now().isoformat(),
                'document_metadata': self.document_metadata,
                'document_texts': self.document_texts
            }
            
            with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Index saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            return False
    
    def load_index(self, filepath: str = None) -> bool:
        """인덱스와 메타데이터 로딩"""
        try:
            if filepath is None:
                filepath = "data/embeddings/legal_vector_index"
            filepath = Path(filepath)
            
            # FAISS 인덱스 로딩 - 확장자가 없으면 직접 로드, 있으면 기존 방식 사용
            if filepath.suffix == '':
                # 확장자가 없는 경우 직접 로드
                faiss_file = str(filepath) + '.faiss'
                if not Path(faiss_file).exists():
                    faiss_file = str(filepath)
                self.index = faiss.read_index(faiss_file)
                # 메타데이터 파일명 처리 (ml_enhanced_faiss_index -> ml_enhanced_faiss_index.json)
                metadata_file = str(filepath) + '.json'
            else:
                # 확장자가 있는 경우 기존 방식 사용
                faiss_file = str(filepath.with_suffix('.faiss'))
                if not Path(faiss_file).exists():
                    faiss_file = str(filepath)
                self.index = faiss.read_index(faiss_file)
                metadata_file = filepath.with_suffix('.json')
            
            # 메타데이터 로딩
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.model_name = metadata.get('model_name', self.model_name)
            self.dimension = metadata.get('dimension', self.dimension)
            self.index_type = metadata.get('index_type', self.index_type)
            
            # 메타데이터 구조 처리
            if 'document_metadata' in metadata:
                self.document_metadata = metadata.get('document_metadata', [])
            else:
                # simple_vector_metadata.json 형식 처리
                self.document_metadata = []
            
            if 'document_texts' in metadata:
                self.document_texts = metadata.get('document_texts', [])
            elif 'texts' in metadata:
                # simple_vector_metadata.json 형식 처리
                self.document_texts = metadata.get('texts', [])
            else:
                self.document_texts = []
            
            # 디버깅을 위한 로그 추가
            self.logger.info(f"Loaded {len(self.document_texts)} texts and {len(self.document_metadata)} metadata entries")
            
            self._index_loaded = True
            self.logger.info(f"Index loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 정보 반환"""
        return {
            'documents_count': len(self.document_metadata),
            'index_is_trained': self.index.is_trained if self.index else False,
            'index_type': type(self.index).__name__ if self.index else 'None',
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'quantization_enabled': self.enable_quantization,
            'lazy_loading_enabled': self.enable_lazy_loading,
            'model_loaded': self._model_loaded,
            'index_loaded': self._index_loaded
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 정보 반환"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            
            return {
                'total_memory_mb': memory_mb,
                'model_loaded': self._model_loaded,
                'index_loaded': self._index_loaded,
                'document_count': len(self.document_texts),
                'quantization_enabled': self.enable_quantization,
                'lazy_loading_enabled': self.enable_lazy_loading,
                'memory_threshold_mb': self.memory_threshold_mb
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def cleanup(self):
        """리소스 정리"""
        self.logger.info("Cleaning up resources...")
        
        # 모델 정리
        if self.model:
            del self.model
            self.model = None
        
        # 인덱스 정리
        if self.index:
            del self.index
            self.index = None
        
        # 메타데이터 정리
        self.document_metadata.clear()
        self.document_texts.clear()
        
        # 캐시 정리
        if hasattr(self, '_memory_cache'):
            self._memory_cache.clear()
        
        # 가비지 컬렉션
        gc.collect()
        
        self.logger.info("Resource cleanup completed")
    
    # 헌재결정례 관련 메서드들
    
    def add_constitutional_decisions(self, decisions: List[Dict[str, Any]]) -> bool:
        """
        헌재결정례 벡터 임베딩 추가
        
        Args:
            decisions: 헌재결정례 데이터 리스트
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            texts = []
            metadata = []
            
            for decision in decisions:
                # 텍스트 조합 (사건명 + 판시사항 + 결정요지 + 전문)
                text_parts = [
                    decision.get('사건명', ''),
                    decision.get('판시사항', ''),
                    decision.get('결정요지', ''),
                    decision.get('전문', '')
                ]
                
                combined_text = ' '.join(filter(None, text_parts))
                if combined_text.strip():
                    texts.append(combined_text)
                    
                    metadata.append({
                        'document_type': 'constitutional_decision',
                        'decision_id': decision.get('헌재결정례일련번호'),
                        'case_name': decision.get('사건명'),
                        'case_number': decision.get('사건번호'),
                        'decision_date': decision.get('종국일자'),
                        'case_type': decision.get('사건종류명'),
                        'court_division_code': decision.get('재판부구분코드'),
                        'summary': decision.get('판시사항'),
                        'decision_gist': decision.get('결정요지'),
                        'reference_articles': decision.get('참조조문'),
                        'reference_precedents': decision.get('참조판례'),
                        'target_articles': decision.get('심판대상조문'),
                        'collected_at': decision.get('collected_at', datetime.now().isoformat())
                    })
            
            if not texts:
                self.logger.warning("No valid texts found in constitutional decisions")
                return False
            
            # 임베딩 생성
            embeddings = self.generate_embeddings(texts)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            
            # 인덱스에 추가
            if self.index.is_trained:
                self.index.add(embeddings)
            else:
                # IVF 인덱스의 경우 훈련 필요
                if self.index_type == "ivf":
                    self.index.train(embeddings)
                    self.index.add(embeddings)
                else:
                    self.index.add(embeddings)
            
            # 메타데이터 저장
            self.document_metadata.extend(metadata)
            self.document_texts.extend(texts)
            
            self.logger.info(f"Added {len(texts)} constitutional decisions to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add constitutional decisions: {e}")
            return False
    
    def search_constitutional_decisions(self, 
                                      query: str, 
                                      top_k: int = 10,
                                      filter_by_date: Optional[str] = None,
                                      filter_by_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        헌재결정례 벡터 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_by_date: 날짜 필터 (YYYY-MM-DD 형식)
            filter_by_type: 사건종류 필터
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.generate_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # 검색 실행
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)  # 필터링을 위해 더 많이 검색
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx].copy()
                    
                    # 문서 타입 필터링
                    if metadata.get('document_type') != 'constitutional_decision':
                        continue
                    
                    # 날짜 필터링
                    if filter_by_date and metadata.get('decision_date'):
                        if not metadata['decision_date'].startswith(filter_by_date):
                            continue
                    
                    # 사건종류 필터링
                    if filter_by_type and metadata.get('case_type'):
                        if filter_by_type not in metadata['case_type']:
                            continue
                    
                    metadata['similarity_score'] = float(score)
                    results.append(metadata)
                    
                    if len(results) >= top_k:
                        break
            
            self.logger.info(f"Found {len(results)} constitutional decision results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search constitutional decisions: {e}")
            return []
    
    def get_constitutional_decisions_by_similarity(self, 
                                                 decision_id: int, 
                                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 헌재결정례와 유사한 결정례 검색
        
        Args:
            decision_id: 기준 헌재결정례 ID
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 유사한 결정례 목록
        """
        try:
            # 기준 결정례 찾기
            target_idx = None
            for i, metadata in enumerate(self.document_metadata):
                if (metadata.get('document_type') == 'constitutional_decision' and 
                    metadata.get('decision_id') == decision_id):
                    target_idx = i
                    break
            
            if target_idx is None:
                self.logger.warning(f"Constitutional decision not found: {decision_id}")
                return []
            
            # 해당 결정례의 임베딩을 쿼리로 사용
            target_embedding = self.index.reconstruct(target_idx).reshape(1, -1)
            
            # 검색 실행 (자기 자신 제외)
            scores, indices = self.index.search(target_embedding.astype('float32'), top_k + 1)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx].copy()
                    
                    # 문서 타입 필터링 및 자기 자신 제외
                    if (metadata.get('document_type') == 'constitutional_decision' and 
                        metadata.get('decision_id') != decision_id):
                        metadata['similarity_score'] = float(score)
                        results.append(metadata)
                        
                        if len(results) >= top_k:
                            break
            
            self.logger.info(f"Found {len(results)} similar constitutional decisions for ID: {decision_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get similar constitutional decisions: {e}")
            return []
    
    def get_constitutional_decisions_stats(self) -> Dict[str, Any]:
        """
        헌재결정례 벡터 스토어 통계
        
        Returns:
            Dict: 통계 정보
        """
        try:
            constitutional_count = 0
            by_year = {}
            by_type = {}
            
            for metadata in self.document_metadata:
                if metadata.get('document_type') == 'constitutional_decision':
                    constitutional_count += 1
                    
                    # 연도별 통계
                    decision_date = metadata.get('decision_date', '')
                    if decision_date and len(decision_date) >= 4:
                        year = decision_date[:4]
                        by_year[year] = by_year.get(year, 0) + 1
                    
                    # 사건종류별 통계
                    case_type = metadata.get('case_type', '')
                    if case_type:
                        by_type[case_type] = by_type.get(case_type, 0) + 1
            
            return {
                'total_constitutional_decisions': constitutional_count,
                'by_year': by_year,
                'by_type': by_type,
                'total_documents': len(self.document_metadata),
                'constitutional_ratio': constitutional_count / len(self.document_metadata) if self.document_metadata else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get constitutional decisions stats: {e}")
            return {}
    
    # 현행법령 관련 메서드들
    
    def add_current_laws(self, laws: List[Dict[str, Any]]) -> bool:
        """
        현행법령을 벡터 저장소에 추가
        
        Args:
            laws: 현행법령 목록
            
        Returns:
            bool: 성공 여부
        """
        try:
            documents = []
            metadatas = []
            
            for law in laws:
                # 문서 내용 구성
                law_name = law.get('법령명한글', '')
                law_id = law.get('법령ID', '')
                
                # 상세 정보에서 조문 내용 추출
                detailed_info = law.get('detailed_info', {})
                content_parts = []
                
                # 조문 내용 추가
                if isinstance(detailed_info, dict):
                    # 조문 내용 추출
                    if '조문내용' in detailed_info:
                        content_parts.append(f"조문: {detailed_info['조문내용']}")
                    
                    # 별표 내용 추가
                    if '별표내용' in detailed_info:
                        content_parts.append(f"별표: {detailed_info['별표내용']}")
                    
                    # 부칙 내용 추가
                    if '부칙내용' in detailed_info:
                        content_parts.append(f"부칙: {detailed_info['부칙내용']}")
                
                # 문서 내용 구성
                content = f"법령명: {law_name}\n"
                if content_parts:
                    content += "\n".join(content_parts)
                else:
                    content += f"법령ID: {law_id}"
                
                documents.append(content)
                
                # 메타데이터 구성
                metadata = {
                    'document_type': 'current_law',
                    'law_id': law_id,
                    'law_name': law_name,
                    'ministry_name': law.get('소관부처명', ''),
                    'effective_date': law.get('시행일자', 0),
                    'promulgation_date': law.get('공포일자', 0),
                    'law_type': law.get('법령구분명', ''),
                    'amendment_type': law.get('제개정구분명', ''),
                    'collected_at': law.get('collected_at', datetime.now().isoformat())
                }
                metadatas.append(metadata)
            
                # 벡터 저장소에 추가
                if documents:
                    self.add_documents(
                        texts=documents,
                        metadatas=metadatas
                    )
                
                self.logger.info(f"현행법령 {len(documents)}개를 벡터 저장소에 추가")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"현행법령 벡터 저장소 추가 실패: {e}")
            return False
    
    def search_current_laws(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        현행법령 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 검색 결과
        """
        try:
            # 임베딩 생성
            query_embeddings = self.generate_embeddings([query])
            if query_embeddings is None or len(query_embeddings) == 0:
                return []
            
            query_embedding = query_embeddings[0].reshape(1, -1)
            
            # 인덱스 가져오기
            index = self.get_index()
            
            # 검색 실행
            scores, indices = index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx].copy()
                    
                    # 문서 타입 필터링
                    if metadata.get('document_type') == 'current_law':
                        metadata['similarity_score'] = float(score)
                        results.append(metadata)
            
            self.logger.info(f"Found {len(results)} current law results for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search current laws: {e}")
            return []
    
    def get_current_laws_by_similarity(self, 
                                     law_id: str, 
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        특정 현행법령과 유사한 법령 검색
        
        Args:
            law_id: 기준 법령 ID
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 유사한 법령 목록
        """
        try:
            # 기준 법령 찾기
            target_idx = None
            for i, metadata in enumerate(self.document_metadata):
                if (metadata.get('document_type') == 'current_law' and 
                    metadata.get('law_id') == law_id):
                    target_idx = i
                    break
            
            if target_idx is None:
                self.logger.warning(f"Current law not found: {law_id}")
                return []
            
            # 해당 법령의 임베딩을 쿼리로 사용
            target_embedding = self.index.reconstruct(target_idx).reshape(1, -1)
            
            # 검색 실행 (자기 자신 제외)
            scores, indices = self.index.search(target_embedding.astype('float32'), top_k + 1)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx].copy()
                    
                    # 문서 타입 필터링 및 자기 자신 제외
                    if (metadata.get('document_type') == 'current_law' and 
                        metadata.get('law_id') != law_id):
                        metadata['similarity_score'] = float(score)
                        results.append(metadata)
                        
                        if len(results) >= top_k:
                            break
            
            self.logger.info(f"Found {len(results)} similar current laws for ID: {law_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get similar current laws: {e}")
            return []
    
    def get_current_laws_by_ministry(self, ministry_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        소관부처별 현행법령 검색
        
        Args:
            ministry_name: 소관부처명
            top_k: 반환할 결과 수
            
        Returns:
            List[Dict]: 현행법령 목록
        """
        try:
            results = []
            for metadata in self.document_metadata:
                if (metadata.get('document_type') == 'current_law' and 
                    metadata.get('ministry_name') == ministry_name):
                    results.append(metadata.copy())
                    
                    if len(results) >= top_k:
                        break
            
            self.logger.info(f"Found {len(results)} current laws for ministry: {ministry_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get current laws by ministry: {e}")
            return []
    
    def get_current_laws_stats(self) -> Dict[str, Any]:
        """
        현행법령 벡터 스토어 통계
        
        Returns:
            Dict: 통계 정보
        """
        try:
            current_law_count = 0
            by_ministry = {}
            by_type = {}
            by_year = {}
            
            for metadata in self.document_metadata:
                if metadata.get('document_type') == 'current_law':
                    current_law_count += 1
                    
                    # 소관부처별 통계
                    ministry = metadata.get('ministry_name', '')
                    if ministry:
                        by_ministry[ministry] = by_ministry.get(ministry, 0) + 1
                    
                    # 법령종류별 통계
                    law_type = metadata.get('law_type', '')
                    if law_type:
                        by_type[law_type] = by_type.get(law_type, 0) + 1
                    
                    # 연도별 통계
                    effective_date = metadata.get('effective_date', 0)
                    if effective_date and effective_date > 0:
                        year = str(effective_date)[:4]
                        by_year[year] = by_year.get(year, 0) + 1
            
            return {
                'total_current_laws': current_law_count,
                'by_ministry': by_ministry,
                'by_type': by_type,
                'by_year': by_year,
                'total_documents': len(self.document_metadata),
                'current_law_ratio': current_law_count / len(self.document_metadata) if self.document_metadata else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current laws stats: {e}")
            return {}
    
    def remove_current_law(self, law_id: str) -> bool:
        """
        현행법령 제거
        
        Args:
            law_id: 법령 ID
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            # 해당 법령의 인덱스 찾기
            target_idx = None
            for i, metadata in enumerate(self.document_metadata):
                if (metadata.get('document_type') == 'current_law' and 
                    metadata.get('law_id') == law_id):
                    target_idx = i
                    break
            
            if target_idx is None:
                self.logger.warning(f"Current law not found for removal: {law_id}")
                return False
            
            # 인덱스에서 제거
            self._remove_from_index(target_idx)
            
            self.logger.info(f"현행법령 제거 성공: {law_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove current law: {e}")
            return False