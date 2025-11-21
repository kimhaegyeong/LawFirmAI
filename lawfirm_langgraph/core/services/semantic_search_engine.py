# -*- coding: utf-8 -*-
"""
Semantic Search Engine
의미적 검색을 위한 검색 엔진
"""

import json
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@dataclass
class SemanticSearchResult:
    """의미적 검색 결과"""
    text: str
    score: float
    similarity_type: str
    metadata: Dict[str, Any]


class SemanticSearchEngine:
    """
    의미적 검색 엔진

    DEPRECATED: 이 클래스는 외부 FAISS 인덱스를 사용합니다.
    새로운 프로젝트는 SemanticSearchEngineV2 (lawfirm_v2.db의 embeddings 테이블 사용)를 사용하세요.
    """

    def __init__(self,
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss",
                 metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json"):
        """검색 엔진 초기화"""
        self.logger = get_logger(__name__)
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
            # FAISS 인덱스 로드
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
            else:
                self.logger.warning(f"FAISS index not found: {self.index_path}")
                return

            # 메타데이터 로드
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata_content = json.load(f)

                # 메타데이터가 딕셔너리인 경우 (설정 정보)
                if isinstance(metadata_content, dict):
                    if 'documents' in metadata_content:
                        self.metadata = metadata_content['documents']
                    else:
                        # 설정 정보만 있는 경우, 빈 리스트로 초기화
                        self.metadata = []
                        self.logger.debug("Metadata contains only configuration, no document data")
                else:
                    self.metadata = metadata_content

                self.logger.info(f"Metadata loaded: {len(self.metadata)} items")
            else:
                self.logger.warning(f"Metadata not found: {self.metadata_path}")
                return

            # 모델 로드
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Model loaded: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load model {self.model_name}: {e}")
                self.model = None

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
                    if 'article_number' in metadata and metadata['article_number']:
                        result['article_number'] = metadata['article_number']
                    if 'case_id' in metadata:
                        result['case_id'] = metadata['case_id']
                    if 'case_name' in metadata:
                        result['case_name'] = metadata['case_name']
                    if 'article_id' in metadata:
                        result['article_id'] = metadata['article_id']

                    results.append(result)

            # 벡터 검색 결과가 0개인 경우 키워드 검색으로 폴백
            if len(results) == 0:
                self.logger.warning(f"Vector search returned 0 results for query '{query}', falling back to keyword search")
                return self._fallback_keyword_search(query, k)

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

            # 키워드 검색 결과가 0개인 경우 데이터베이스 검색 시도
            if len(results) == 0:
                self.logger.warning(f"Keyword search returned 0 results for query '{query}', trying database search")
                db_results = self._database_fallback_search(query, k)
                if db_results:
                    return db_results[:k]
                # 데이터베이스 검색도 실패한 경우 빈 리스트 반환
                self.logger.error(f"All search methods failed for query '{query}'")
                return []

            return results[:k]

        except Exception as e:
            self.logger.error(f"Error in fallback search: {e}")
            # 키워드 검색 실패 시 데이터베이스 검색 시도
            try:
                return self._database_fallback_search(query, k)
            except Exception as db_error:
                self.logger.error(f"Database fallback also failed: {db_error}")
                return []

    def _database_fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """데이터베이스 폴백 검색"""
        try:
            import sqlite3
            results = []

            # 데이터베이스 경로 확인
            import os
            db_path = "data/lawfirm_v2.db"
            if not os.path.exists(db_path):
                self.logger.error(f"Database not found at {db_path}")
                raise FileNotFoundError(f"Database file not found at {db_path}")

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 테이블 존재 확인 후 법률 검색
                try:
                    cursor.execute("""
                        SELECT law_name, COALESCE(searchable_text, full_text) as content, 'law' as type
                        FROM assembly_laws
                        WHERE (COALESCE(searchable_text, '') || ' ' || COALESCE(full_text, '') || ' ' || COALESCE(law_name, '')) LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", k))

                    for row in cursor.fetchall():
                        content = row.get('content', '') or ''
                        result = {
                            'text': f"{row.get('law_name', 'Unknown')}: {content[:200]}...",
                            'similarity': 0.6,
                            'score': 0.6,
                            'type': 'law',
                            'source': row.get('law_name', 'Unknown'),
                            'metadata': dict(row),
                            'search_type': 'database_fallback'
                        }
                        results.append(result)
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not search assembly_laws table: {e}")

                # 판례 검색
                try:
                    cursor.execute("""
                        SELECT case_name, case_number, COALESCE(searchable_text, full_text, '') as content, 'precedent' as type
                        FROM precedent_cases
                        WHERE (COALESCE(searchable_text, '') || ' ' || COALESCE(full_text, '') || ' ' || COALESCE(case_name, '')) LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", k))

                    for row in cursor.fetchall():
                        text_content = row.get('content', '') or ''
                        result = {
                            'text': f"{row.get('case_name', 'Unknown')} ({row.get('case_number', 'N/A')}): {text_content[:200]}...",
                            'similarity': 0.6,
                            'score': 0.6,
                            'type': 'precedent',
                            'source': f"{row.get('case_name', 'Unknown')} {row.get('case_number', 'N/A')}",
                            'metadata': dict(row),
                            'search_type': 'database_fallback'
                        }
                        results.append(result)
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not search precedent_cases table: {e}")

            # 점수순 정렬
            results.sort(key=lambda x: x['score'], reverse=True)

            # 결과가 없으면 로그만 남기고 빈 리스트 반환 (Mock 결과 대신)
            if len(results) == 0:
                self.logger.debug(f"No results found in database for query '{query}'")
                return []

            return results[:k]

        except FileNotFoundError:
            # 데이터베이스 파일이 없는 경우 경고만 남기고 빈 리스트 반환
            self.logger.warning(f"Database file not found at {db_path}")
            return []
        except Exception as e:
            self.logger.error(f"Database fallback search failed: {e}")
            # 에러 발생시키지 않고 빈 리스트 반환
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
