# -*- coding: utf-8 -*-
"""
Vector Store
벡터 임베딩 생성 및 관리 모듈

법률 문서의 벡터 임베딩을 생성하고 FAISS를 사용하여 벡터 검색을 수행합니다.
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime

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

logger = logging.getLogger(__name__)


class LegalVectorStore:
    """법률 문서 벡터 스토어 클래스"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat"):
        """
        벡터 스토어 초기화
        
        Args:
            model_name: 사용할 Sentence-BERT 모델명
            dimension: 벡터 차원
            index_type: FAISS 인덱스 타입 ("flat", "ivf", "hnsw")
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        
        self.model = None
        self.index = None
        self.document_metadata = []
        self.document_texts = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LegalVectorStore initialized with model: {model_name}")
        
        # 모델 로딩
        self._load_model()
        
        # 인덱스 초기화
        self._initialize_index()
    
    def _load_model(self):
        """Sentence-BERT 모델 로딩"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers is required but not installed")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _initialize_index(self):
        """FAISS 인덱스 초기화"""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not installed")
        
        try:
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
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트에 대한 임베딩 생성"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
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
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict[str, Any]]:
        """벡터 검색 수행 (하이브리드 검색용)"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # 검색 수행 (필터링을 위해 더 많은 결과 가져옴)
            search_k = min(top_k * 5, self.index.ntotal) if self.index.ntotal > 0 else top_k
            scores, indices = self.index.search(query_embedding, search_k)
            
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
            
            self.logger.info(f"Search completed: {len(results)} results found")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def search_legacy(self, query: str, top_k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """벡터 검색 수행 (레거시)"""
        if not self.model or not self.index:
            raise RuntimeError("Model or index not initialized")
        
        try:
            # 쿼리 임베딩 생성
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
            query_embedding = self.model.encode([search_query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # 해당 법령의 문서들만 검색
            law_embeddings = np.array([self.document_texts[i] for i in law_indices])
            law_embeddings = self.model.encode(law_embeddings, convert_to_numpy=True)
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
            query_embedding = self.model.encode([search_query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # 해당 카테고리의 문서들만 검색
            category_embeddings = np.array([self.document_texts[i] for i in category_indices])
            category_embeddings = self.model.encode(category_embeddings, convert_to_numpy=True)
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
    
    def load_index(self, filepath: str) -> bool:
        """인덱스와 메타데이터 로딩"""
        try:
            filepath = Path(filepath)
            
            # FAISS 인덱스 로딩 - 확장자가 없으면 직접 로드, 있으면 기존 방식 사용
            if filepath.suffix == '':
                # 확장자가 없는 경우 직접 로드
                self.index = faiss.read_index(str(filepath))
                # 메타데이터 파일명 처리 (simple_vector_index -> simple_vector_metadata.json)
                metadata_file = filepath.parent / f"{filepath.name.replace('_index', '_metadata')}.json"
            else:
                # 확장자가 있는 경우 기존 방식 사용
                self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
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
            'model_name': self.model_name
        }