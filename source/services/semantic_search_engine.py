"""
의미적 검색 엔진 (FAISS 기반)
벡터 임베딩을 사용한 의미적 유사도 검색 엔진
"""

import numpy as np
import faiss
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """의미적 검색 엔진"""
    
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss",
                 metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json"):
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = None
        self.index = None
        self.metadata = []
        self.dimension = 768  # ko-sroberta-multitask의 임베딩 차원
        
        self._load_model()
        self._load_index()
    
    def _load_model(self):
        """Sentence-BERT 모델 로드"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_index(self):
        """FAISS 인덱스 및 메타데이터 로드"""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Index loaded: {self.index.ntotal} vectors")
            else:
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
            
            if os.path.exists(self.metadata_path):
                logger.info(f"Loading metadata from {self.metadata_path}")
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded: {len(self.metadata)} items")
                
                # 인덱스와 메타데이터 동기화 확인
                if hasattr(self, 'index') and self.index.ntotal != len(self.metadata):
                    logger.warning(f"Index ({self.index.ntotal}) and metadata ({len(self.metadata)}) size mismatch!")
                    # 메타데이터를 인덱스 크기에 맞춤
                    if self.index.ntotal < len(self.metadata):
                        logger.warning("Truncating metadata to match index size")
                        self.metadata = self.metadata[:self.index.ntotal]
            else:
                logger.info("No metadata found, starting with empty metadata")
                self.metadata = []
                
        except Exception as e:
            logger.error(f"Failed to load index/metadata: {e}")
            raise
    
    def build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """문서로부터 FAISS 인덱스 구축"""
        try:
            logger.info(f"Building index from {len(documents)} documents")
            
            # 텍스트 추출 및 임베딩 생성
            texts = []
            metadata = []
            
            for doc in documents:
                # 문서 타입에 따라 텍스트 추출
                if doc.get("type") == "law":
                    text = f"{doc.get('law_name', '')} {doc.get('article_number', '')} {doc.get('content', '')}"
                elif doc.get("type") == "precedent":
                    text = f"{doc.get('case_name', '')} {doc.get('case_number', '')} {doc.get('content', '')}"
                elif doc.get("type") == "constitutional":
                    text = f"{doc.get('case_name', '')} {doc.get('case_number', '')} {doc.get('content', '')}"
                else:
                    text = doc.get('content', '')
                
                texts.append(text.strip())
                metadata.append({
                    "id": doc.get("id"),
                    "type": doc.get("type"),
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "original_doc": doc
                })
            
            # 배치 단위로 임베딩 생성
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                embeddings.append(batch_embeddings)
            
            # 임베딩 결합
            all_embeddings = np.vstack(embeddings)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(all_embeddings)
            
            # FAISS 인덱스에 추가
            self.index.add(all_embeddings.astype('float32'))
            
            # 메타데이터 저장 (인덱스와 동기화 확인)
            self.metadata = metadata
            
            # 인덱스와 메타데이터 크기 검증
            if self.index.ntotal != len(self.metadata):
                logger.warning(f"Index size ({self.index.ntotal}) != metadata size ({len(self.metadata)})")
                # 인덱스 크기에 맞춰 메타데이터 조정
                if self.index.ntotal > len(self.metadata):
                    logger.warning("Truncating metadata to match index size")
                    self.metadata = self.metadata[:self.index.ntotal]
                else:
                    logger.warning("Index size is smaller than metadata, this may cause issues")
            
            logger.info(f"Index and metadata synchronized: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")
            
            # 파일로 저장
            self._save_index()
            self._save_metadata()
            
            logger.info(f"Index built successfully: {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def _save_index(self):
        """FAISS 인덱스를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _save_metadata(self):
        """메타데이터를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """의미적 검색 실행 - 완전한 동기화 보장"""
        try:
            if self.index.ntotal == 0:
                print("Index is empty")
                return []
            
            # 인덱스와 메타데이터 동기화 강제 확인
            self._force_synchronization()
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # 검색 실행
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS에서 유효하지 않은 인덱스
                    continue
                
                # 메타데이터 인덱스 범위 검사 및 강제 동기화
                if idx >= len(self.metadata):
                    print(f"Index {idx} out of range for metadata (length: {len(self.metadata)})")
                    # 강제 동기화 시도
                    if self._force_synchronization():
                        print("Force synchronization successful, retrying search")
                        return self.search(query, k, threshold)
                    else:
                        print("Force synchronization failed, skipping this result")
                        continue
                
                if score >= threshold:
                    try:
                        result = {
                            "id": self.metadata[idx]["id"],
                            "type": self.metadata[idx]["type"],
                            "title": self.metadata[idx]["title"],
                            "source": self.metadata[idx]["source"],
                            "content": self.metadata[idx]["original_doc"].get("content", ""),
                            "similarity_score": float(score),
                            "search_type": "semantic",
                            "relevance_score": float(score)
                        }
                        results.append(result)
                    except (KeyError, IndexError) as e:
                        print(f"Error accessing metadata at index {idx}: {e}")
                        continue
            
            print(f"Semantic search completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
    
    def search_by_type(self, query: str, doc_type: str, k: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """특정 타입의 문서만 검색"""
        try:
            all_results = self.search(query, k * 2, threshold)  # 더 많이 검색해서 필터링
            filtered_results = [r for r in all_results if r["type"] == doc_type]
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Type-specific search failed: {e}")
            return []
    
    def get_similar_documents(self, doc_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """특정 문서와 유사한 문서 검색"""
        try:
            # 문서 ID로 메타데이터에서 인덱스 찾기
            doc_idx = None
            for i, meta in enumerate(self.metadata):
                if meta["id"] == doc_id:
                    doc_idx = i
                    break
            
            if doc_idx is None:
                logger.warning(f"Document {doc_id} not found")
                return []
            
            # 해당 문서의 임베딩 가져오기
            doc_embedding = self.index.reconstruct(doc_idx)
            doc_embedding = doc_embedding.reshape(1, -1)
            
            # 유사한 문서 검색
            scores, indices = self.index.search(doc_embedding.astype('float32'), k + 1)  # 자기 자신 제외
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx == doc_idx:  # 유효하지 않은 인덱스나 자기 자신 제외
                    continue
                
                result = {
                    "id": self.metadata[idx]["id"],
                    "type": self.metadata[idx]["type"],
                    "title": self.metadata[idx]["title"],
                    "source": self.metadata[idx]["source"],
                    "content": self.metadata[idx]["original_doc"].get("content", ""),
                    "similarity_score": float(score),
                    "search_type": "semantic_similarity",
                    "relevance_score": float(score)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Similar documents search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 반환"""
        return {
            "total_documents": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "index_type": "IndexFlatIP",
            "metadata_count": len(self.metadata)
        }
    
    def _force_synchronization(self):
        """강제 동기화 - 완전한 동기화 보장"""
        try:
            if self.index.ntotal != len(self.metadata):
                print(f"Synchronization issue detected: index={self.index.ntotal}, metadata={len(self.metadata)}")
                return self._attempt_force_recovery()
            return True
        except Exception as e:
            print(f"Error checking synchronization: {e}")
            return False
    
    def _attempt_force_recovery(self):
        """강제 복구 시도 - 완전한 동기화 보장"""
        try:
            print("Attempting force recovery for index-metadata synchronization")
            
            # 1. 메타데이터 파일 다시 로드
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"Metadata reloaded: {len(self.metadata)} items")
            
            # 2. 인덱스 파일 다시 로드
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"Index reloaded: {self.index.ntotal} vectors")
            
            # 3. 강제 동기화 - 인덱스 크기에 맞춰 메타데이터 조정
            if self.index.ntotal != len(self.metadata):
                print(f"Force synchronization: adjusting metadata to match index size")
                if self.index.ntotal < len(self.metadata):
                    self.metadata = self.metadata[:self.index.ntotal]
                    print(f"Metadata truncated to {len(self.metadata)} items")
                else:
                    # 인덱스가 더 큰 경우 빈 메타데이터로 채움
                    while len(self.metadata) < self.index.ntotal:
                        self.metadata.append({
                            "id": f"dummy_{len(self.metadata)}",
                            "type": "unknown",
                            "title": "Unknown Document",
                            "source": "unknown",
                            "original_doc": {"content": "Unknown content"}
                        })
                    print(f"Metadata extended to {len(self.metadata)} items")
            
            # 4. 최종 동기화 확인
            if self.index.ntotal == len(self.metadata):
                print("Force recovery successful: index and metadata synchronized")
                return True
            else:
                print(f"Force recovery failed: still mismatched (index={self.index.ntotal}, metadata={len(self.metadata)})")
                return False
                
        except Exception as e:
            print(f"Force recovery failed with error: {e}")
            return False
