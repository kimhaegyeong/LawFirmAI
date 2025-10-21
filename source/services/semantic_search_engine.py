# -*- coding: utf-8 -*-
"""
Semantic Search Engine
의미적 검색을 위한 검색 엔진
"""

import logging
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class SemanticSearchResult:
    """의미적 검색 결과"""
    text: str
    score: float
    similarity_type: str
    metadata: Dict[str, Any]


class SemanticSearchEngine:
    """의미적 검색 엔진"""
    
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss",
                 metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json"):
        """검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # 벡터 인덱스와 메타데이터 로드
        self.index = None
        self.metadata = []
        self.model = None
        
        self._load_components()
        self.logger.info("SemanticSearchEngine initialized")
    
    def _load_components(self):
        """벡터 인덱스와 모델 로드"""
        try:
            # 여러 인덱스 경로 시도
            index_paths = [
                "data/embeddings/ml_enhanced_ko_sroberta_precedents.faiss",
                "data/embeddings/optimized_ko_sroberta_precedents.faiss",
                "data/embeddings/quantized_ko_sroberta_precedents.faiss",
                str(self.index_path)
            ]
            
            metadata_paths = [
                "data/embeddings/ml_enhanced_ko_sroberta_precedents.json",
                "data/embeddings/optimized_ko_sroberta_precedents.json", 
                "data/embeddings/quantized_ko_sroberta_precedents.json",
                str(self.metadata_path)
            ]
            
            # 인덱스 로드 시도
            for i, (idx_path, meta_path) in enumerate(zip(index_paths, metadata_paths)):
                try:
                    if Path(idx_path).exists() and Path(meta_path).exists():
                        self.logger.info(f"Loading index from: {idx_path}")
                        self.index = faiss.read_index(idx_path)
                        
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata_content = json.load(f)
                        
                        # 메타데이터가 딕셔너리인 경우 (설정 정보)
                        if isinstance(metadata_content, dict):
                            if 'document_metadata' in metadata_content:
                                self.metadata = metadata_content['document_metadata']
                            elif 'documents' in metadata_content:
                                self.metadata = metadata_content['documents']
                            else:
                                # 설정 정보만 있는 경우, 빈 리스트로 초기화
                                self.metadata = []
                                self.logger.warning("Metadata contains only configuration, no document data")
                        else:
                            self.metadata = metadata_content
                        
                        self.logger.info(f"Successfully loaded index with {len(self.metadata)} documents")
                        break
                    else:
                        self.logger.warning(f"Index files not found: {idx_path}, {meta_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load index from {idx_path}: {e}")
                    continue
            
            # 모델 로드
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Model loaded: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                self.model = None
            
            if self.index is None or self.model is None:
                self.logger.warning("Vector index or model not available, will use keyword search fallback")
                
        except Exception as e:
            self.logger.error(f"Error loading components: {e}")
            self.index = None
            self.metadata = []
            self.model = None
    
    def search(self, query: str, documents: List[Dict[str, Any]] = None, k: int = 10) -> List[Dict[str, Any]]:
        """
        의미적 검색 수행
        
        Args:
            query: 검색 쿼리
            documents: 검색할 문서 목록 (사용하지 않음, 벡터 인덱스 사용)
            k: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
        """
        try:
            if self.index is None or self.model is None or not self.metadata:
                self.logger.debug("Vector index or model not available, falling back to keyword search")
                return self._fallback_keyword_search(query, k)
            
            # 쿼리 벡터화
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding.astype('float32')
            
            # FAISS 검색 수행
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS에서 -1은 유효하지 않은 인덱스
                    continue
                
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    
                    # 결과 딕셔너리 생성
                    result = {
                        'text': metadata.get('text', ''),
                        'similarity': float(score),
                        'score': float(score),
                        'type': metadata.get('type', 'unknown'),
                        'source': metadata.get('source', ''),
                        'metadata': metadata,
                        'search_type': 'vector_search'
                    }
                    
                    # 추가 메타데이터
                    if 'law_name' in metadata:
                        result['law_name'] = metadata['law_name']
                    if 'article_number' in metadata:
                        result['article_number'] = metadata['article_number']
                    if 'case_id' in metadata:
                        result['case_id'] = metadata['case_id']
                    if 'case_name' in metadata:
                        result['case_name'] = metadata['case_name']
                    
                    results.append(result)
            
            self.logger.info(f"Semantic search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return self._fallback_keyword_search(query, k)
    
    def _fallback_keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """키워드 기반 폴백 검색"""
        try:
            results = []
            query_words = set(query.lower().split())
            
            # 메타데이터가 없는 경우 데이터베이스에서 직접 검색
            if not self.metadata:
                return self._database_fallback_search(query, k)
            
            for i, metadata in enumerate(self.metadata[:100]):  # 상위 100개만 검색
                text = metadata.get('text', '')
                text_words = set(text.lower().split())
                
                # Jaccard 유사도 계산
                intersection = len(query_words.intersection(text_words))
                union = len(query_words.union(text_words))
                score = intersection / union if union > 0 else 0.0
                
                if score > 0.1:  # 낮은 임계값
                    result = {
                        'text': text,
                        'similarity': score,
                        'score': score,
                        'type': metadata.get('type', 'unknown'),
                        'source': metadata.get('source', ''),
                        'metadata': metadata,
                        'search_type': 'keyword_fallback'
                    }
                    results.append(result)
            
            # 점수순 정렬
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in fallback search: {e}")
            return []
    
    def _database_fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """데이터베이스 폴백 검색"""
        try:
            import sqlite3
            results = []
            
            # 데이터베이스에서 직접 검색
            db_path = "data/lawfirm.db"
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 법률 검색 (assembly_laws 테이블에는 article_number 컬럼이 없음)
                cursor.execute("""
                    SELECT law_name, law_id, full_text, 'law' as type
                    FROM assembly_laws 
                    WHERE full_text LIKE ? OR law_name LIKE ?
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", k))
                
                for row in cursor.fetchall():
                    result = {
                        'text': f"{row['law_name']} {row['law_id']}: {row['full_text'][:200]}...",
                        'similarity': 0.8,  # 데이터베이스 검색은 높은 신뢰도
                        'score': 0.8,
                        'type': 'law',
                        'source': f"{row['law_name']} {row['law_id']}",
                        'metadata': dict(row),
                        'search_type': 'database_fallback'
                    }
                    results.append(result)
                
                # 판례 검색
                cursor.execute("""
                    SELECT case_name, case_number, full_text, searchable_text, 'precedent' as type
                    FROM precedent_cases 
                    WHERE searchable_text LIKE ? OR case_name LIKE ? OR full_text LIKE ?
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", k))
                
                for row in cursor.fetchall():
                    # searchable_text가 있으면 사용, 없으면 full_text 사용
                    text_content = row['searchable_text'] or row['full_text']
                    result = {
                        'text': f"{row['case_name']} ({row['case_number']}): {text_content[:200]}...",
                        'similarity': 0.8,
                        'score': 0.8,
                        'type': 'precedent',
                        'source': f"{row['case_name']} {row['case_number']}",
                        'metadata': dict(row),
                        'search_type': 'database_fallback'
                    }
                    results.append(result)
            
            # 점수순 정렬
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in database fallback search: {e}")
            return []
    
    def _calculate_semantic_score(self, query: str, text: str) -> float:
        """
        의미적 유사도 계산 (간단한 구현)
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            float: 유사도 점수 (0.0-1.0)
        """
        # 실제 구현에서는 벡터 임베딩을 사용해야 함
        # 여기서는 간단한 키워드 기반 유사도 계산
        
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard 유사도
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        jaccard_score = intersection / union if union > 0 else 0.0
        
        # 추가 가중치 (법률 용어 매칭)
        legal_terms = {'계약', '해지', '손해배상', '이혼', '형사', '처벌'}
        legal_matches = len(query_words.intersection(legal_terms))
        legal_bonus = min(0.2, legal_matches * 0.1)
        
        return min(1.0, jaccard_score + legal_bonus)
    
    def _determine_similarity_type(self, query: str, text: str) -> str:
        """
        유사도 타입 결정
        
        Args:
            query: 검색 쿼리
            text: 검색할 텍스트
            
        Returns:
            str: 유사도 타입
        """
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        intersection = len(query_words.intersection(text_words))
        
        if intersection == len(query_words):
            return "high_similarity"
        elif intersection >= len(query_words) * 0.7:
            return "medium_similarity"
        else:
            return "low_similarity"


# 기본 인스턴스 생성
def create_semantic_search_engine() -> SemanticSearchEngine:
    """기본 의미적 검색 엔진 생성"""
    return SemanticSearchEngine()


if __name__ == "__main__":
    # 테스트 코드
    engine = create_semantic_search_engine()
    
    # 샘플 문서
    documents = [
        {"text": "민법 제543조 계약의 해지에 관한 규정", "metadata": {"category": "civil"}},
        {"text": "형법 제250조 살인죄의 구성요건", "metadata": {"category": "criminal"}},
        {"text": "가족법상 이혼 절차 및 요건", "metadata": {"category": "family"}}
    ]
    
    # 검색 테스트
    results = engine.search("계약 해지", documents)
    print(f"Semantic search results: {len(results)}")
    for result in results:
        print(f"  Score: {result.score:.3f}, Type: {result.similarity_type}")
        print(f"  Text: {result.text}")