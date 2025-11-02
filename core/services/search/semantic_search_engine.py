# -*- coding: utf-8 -*-
"""
Semantic Search Engine
의미적 검색을 위한 검색 엔진
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
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
            return self._search_hybrid(query, k)
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return self._fallback_keyword_search(query, k)

    def _search_hybrid(self, query: str, k: int) -> List[Dict[str, Any]]:
        """하이브리드(BM25 유사 + 벡터) 검색 파이프라인"""
        # 쿼리 리라이트
        rewrites = self._rewrite_query(query)
        unique_rewrites = list(dict.fromkeys([query] + rewrites))[:5]

        # 키워드 후보 수집
        keyword_candidates: List[Dict[str, Any]] = []
        for q in unique_rewrites:
            keyword_candidates.extend(self._keyword_rank(q, k=max(20, k)))
        # 키워드 중복 제거
        seen = set()
        dedup_keyword = []
        for r in keyword_candidates:
            t = r.get('metadata', {}).get('id') or r.get('text')
            if t in seen:
                continue
            seen.add(t)
            dedup_keyword.append(r)

        # 벡터 후보 수집
        vector_candidates: List[Dict[str, Any]] = []
        if self.index is not None and self.model is not None and self.metadata:
            try:
                query_embedding = self.model.encode([query]).astype('float32')
                scores, indices = self.index.search(query_embedding, max(20, k))
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or idx >= len(self.metadata):
                        continue
                    md = self.metadata[idx]
                    vector_candidates.append({
                        'text': md.get('text', ''),
                        'similarity': float(score),
                        'score': float(score),
                        'type': md.get('type', 'unknown'),
                        'source': md.get('source', ''),
                        'metadata': md,
                        'search_type': 'vector_search'
                    })
            except Exception as e:
                self.logger.warning(f"Vector search failed, continuing with keyword only: {e}")

        # 병합 및 재랭킹
        merged = self._merge_and_rerank(dedup_keyword, vector_candidates, k)

        # 최소 리콜 폴백
        MIN_RESULTS = max(5, k // 2)
        if len(merged) < MIN_RESULTS:
            db = self._database_fallback_search(query, k=MIN_RESULTS)
            merged = self._merge_and_rerank(merged, db, k)

        if len(merged) == 0:
            return self._fallback_keyword_search(query, k)

        self.logger.info(f"Hybrid search completed: {len(merged)} results for query '{query}'")
        return merged[:k]

    def _merge_and_rerank(self, a: List[Dict[str, Any]], b: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """결과 병합 및 재랭킹"""
        combined = []
        for r in a:
            rr = dict(r)
            rr['hybrid_score'] = 0.4 * float(rr.get('score', 0.0))
            combined.append(rr)
        for r in b:
            rr = dict(r)
            rr['hybrid_score'] = rr.get('hybrid_score', 0.0) + 0.6 * float(rr.get('score', 0.0))
            combined.append(rr)
        # 중복 제거
        seen = set()
        dedup = []
        for r in combined:
            md = r.get('metadata', {}) or {}
            key = md.get('article_id') or md.get('case_id') or md.get('id') or r.get('text')
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        dedup.sort(key=lambda x: x.get('hybrid_score', 0.0), reverse=True)
        return dedup[:max(50, k)]

    def _rewrite_query(self, query: str) -> List[str]:
        """간단한 쿼리 리라이트: 동의어/숫자 변형/법률 용어 정규화"""
        q = query.strip()
        rewrites: List[str] = []
        synonyms = {
            '손해배상': ['배상', '보상'],
            '이혼': ['혼인해소', '결혼해소'],
            '계약 해지': ['계약 종료', '해지'],
            '조문': ['조항', '법조문'],
        }
        for key, vals in synonyms.items():
            if key in q:
                for v in vals:
                    rewrites.append(q.replace(key, v))
        if '제' in q and '조' in q:
            rewrites.append(q.replace('제', '제 ').replace('조', ' 조'))
            rewrites.append(q.replace('제 ', '제').replace(' 조', '조'))
        if '민법' in q:
            rewrites.append(q.replace('민법', '대한민국 민법'))
        return [r for r in rewrites if r and r != q]

    def _keyword_rank(self, query: str, k: int) -> List[Dict[str, Any]]:
        """간단한 BM25 유사 키워드 랭킹"""
        try:
            if not self.metadata:
                return []
            q_terms = query.lower().split()
            N = min(len(self.metadata), 5000)
            df: Dict[str, int] = {}
            docs = []
            for md in self.metadata[:N]:
                text = (md.get('text') or '').lower()
                terms = set(text.split())
                for t in terms:
                    df[t] = df.get(t, 0) + 1
                docs.append((text, md))
            import math
            scored = []
            for text, md in docs:
                if not text:
                    continue
                score = 0.0
                for t in q_terms:
                    tf = text.count(t)
                    if tf == 0:
                        continue
                    idf = math.log((N + 1) / (df.get(t, 0) + 1))
                    score += math.sqrt(tf) * idf
                if score > 0:
                    scored.append({
                        'text': md.get('text', ''),
                        'similarity': float(score),
                        'score': float(score),
                        'type': md.get('type', 'unknown'),
                        'source': md.get('source', ''),
                        'metadata': md,
                        'search_type': 'keyword_rank'
                    })
            scored.sort(key=lambda x: x['score'], reverse=True)
            return scored[:k]
        except Exception as e:
            self.logger.warning(f"Keyword rank failed: {e}")
            return []

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
            import sys
            # source 모듈 경로 추가
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            from source.utils.config import Config
            config = Config()
            db_path = config.database_path
            if not os.path.exists(db_path):
                self.logger.error(f"Database not found at {db_path}")
                raise FileNotFoundError(f"Database file not found at {db_path}")

            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 테이블 존재 확인 후 법률 검색
                try:
                    # lawfirm_v2.db의 statutes 테이블 사용
                    cursor.execute("""
                        SELECT s.name as law_name, sa.text as content, 'law' as type
                        FROM statute_articles sa
                        JOIN statutes s ON sa.statute_id = s.id
                        WHERE sa.text LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", k))

                    for row in cursor.fetchall():
                        # sqlite3.Row는 .get() 메서드가 없으므로 딕셔너리로 변환하거나 직접 접근
                        row_dict = dict(row)
                        content = row_dict.get('content', '') or ''
                        result = {
                            'text': f"{row_dict.get('law_name', 'Unknown')}: {content[:200]}...",
                            'similarity': 0.6,
                            'score': 0.6,
                            'type': 'law',
                            'source': row_dict.get('law_name', 'Unknown'),
                            'metadata': row_dict,
                            'search_type': 'database_fallback'
                        }
                        results.append(result)
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not search statutes table: {e}")

                # 판례 검색 (lawfirm_v2.db의 cases 테이블 사용)
                try:
                    cursor.execute("""
                        SELECT c.casenames as case_name, c.doc_id as case_number, cp.text as content, 'precedent' as type
                        FROM case_paragraphs cp
                        JOIN cases c ON cp.case_id = c.id
                        WHERE cp.text LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", k))

                    for row in cursor.fetchall():
                        # sqlite3.Row는 .get() 메서드가 없으므로 딕셔너리로 변환하거나 직접 접근
                        row_dict = dict(row)
                        text_content = row_dict.get('content', '') or ''
                        result = {
                            'text': f"{row_dict.get('case_name', 'Unknown')} ({row_dict.get('case_number', 'N/A')}): {text_content[:200]}...",
                            'similarity': 0.6,
                            'score': 0.6,
                            'type': 'precedent',
                            'source': f"{row_dict.get('case_name', 'Unknown')} {row_dict.get('case_number', 'N/A')}",
                            'metadata': row_dict,
                            'search_type': 'database_fallback'
                        }
                        results.append(result)
                except sqlite3.OperationalError as e:
                    self.logger.warning(f"Could not search cases table: {e}")

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
