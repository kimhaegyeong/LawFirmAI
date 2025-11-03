
import sqlite3
import time
import hashlib
from typing import List, Dict, Any, Optional

from ..utils.config import Config

class OptimizedPrecedentSearchEngine:
    """최적화된 판례 검색 엔진"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            config = Config()
            db_path = config.database_path
        self.db_path = db_path
        self.cache = {}
        self.cache_size = 1000
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_query_hash(self, query: str, top_k: int) -> str:
        """쿼리 해시 생성"""
        query_string = f"{query}_{top_k}"
        return hashlib.md5(query_string.encode()).hexdigest()

    def _get_from_cache(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """캐시에서 결과 조회"""
        if query_hash in self.cache:
            self.cache_hits += 1
            return self.cache[query_hash]
        self.cache_misses += 1
        return None

    def _save_to_cache(self, query_hash: str, results: List[Dict[str, Any]]):
        """결과를 캐시에 저장"""
        if len(self.cache) < self.cache_size:
            self.cache[query_hash] = results

    def search_precedents_optimized(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """최적화된 판례 검색"""
        # 캐시 확인
        query_hash = self._generate_query_hash(query, top_k)
        cached_result = self._get_from_cache(query_hash)

        if cached_result is not None:
            print(f"캐시 히트: {query}")
            return cached_result

        # 데이터베이스에서 검색
        start_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 최적화된 쿼리 (FTS5 안전 바인딩)
            optimized_query = """
            SELECT
                case_id,
                case_name,
                case_number,
                rank
            FROM fts_precedent_cases
            WHERE fts_precedent_cases MATCH ?
            ORDER BY rank
            LIMIT ?
            """

            # FTS5 안전 쿼리 변환
            safe_query = self._make_fts5_safe_query(query)
            cursor.execute(optimized_query, (safe_query, top_k))
            rows = cursor.fetchall()

            # 결과 처리
            results = []
            for row in rows:
                result = {
                    'case_id': row['case_id'],
                    'case_name': row['case_name'],
                    'case_number': row['case_number'],
                    'similarity': self._normalize_fts_score(row['rank']),
                    'search_type': 'fts_optimized'
                }
                results.append(result)

        search_time = time.time() - start_time
        print(f"DB 검색: {query} - {len(results)}개 결과, {search_time:.4f}초")

        # 캐시에 저장
        self._save_to_cache(query_hash, results)

        return results

    def _make_fts5_safe_query(self, query: str) -> str:
        """FTS5 안전 쿼리 변환"""
        if not query or not query.strip():
            return '""'

        # 특수 문자 제거 및 이스케이핑
        safe_query = query.strip()

        # FTS5 특수 문자들 이스케이핑
        fts5_special_chars = ['"', '*', '^', '(', ')', 'AND', 'OR', 'NOT']

        # 단순 키워드 검색으로 변환 (따옴표로 감싸기)
        if any(char in safe_query for char in fts5_special_chars):
            # 특수 문자가 있으면 단순 키워드로 변환
            safe_query = safe_query.replace('"', '""')  # 따옴표 이스케이핑
            safe_query = f'"{safe_query}"'
        else:
            # 특수 문자가 없으면 그대로 사용하되 따옴표로 감싸기
            safe_query = f'"{safe_query}"'

        return safe_query

    def _normalize_fts_score(self, rank: float) -> float:
        """FTS 점수를 0-1 범위로 정규화"""
        if rank == 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 / abs(rank)))

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

# 사용 예제
if __name__ == "__main__":
    engine = OptimizedPrecedentSearchEngine()

    # 검색 테스트
    results = engine.search_precedents_optimized("계약", 5)
    print(f"검색 결과: {len(results)}개")

    # 캐시 통계
    stats = engine.get_cache_stats()
    print(f"캐시 통계: {stats}")
