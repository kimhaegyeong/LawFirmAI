# -*- coding: utf-8 -*-
"""
판례 전용 검색 엔진
판례 데이터베이스와 벡터 인덱스를 활용한 전문 검색 시스템
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...data.database import DatabaseManager
from ...data.vector_store import LegalVectorStore

logger = logging.getLogger(__name__)


@dataclass
class PrecedentSearchResult:
    """판례 검색 결과"""
    case_id: str
    case_name: str
    case_number: str
    category: str
    court: str
    decision_date: str
    field: str
    summary: str
    judgment_summary: Optional[str] = None
    judgment_gist: Optional[str] = None
    similarity: float = 0.0
    search_type: str = "unknown"  # fts, vector, hybrid


class PrecedentSearchEngine:
    """판례 전용 검색 엔진"""

    def __init__(self,
                 db_path: str = "data/lawfirm.db",
                 vector_index_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index",
                 vector_metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.json"):
        """판례 검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)

        # 데이터베이스 연결
        self.db_manager = DatabaseManager(db_path)

        # 벡터 저장소 초기화
        self.vector_store = None
        self.vector_metadata = None

        try:
            self.vector_store = LegalVectorStore(
                model_name="jhgan/ko-sroberta-multitask",
                dimension=768,
                index_type="flat"
            )

            # 기존 벡터 인덱스 로드
            faiss_file = Path(vector_index_path)
            if not faiss_file.exists():
                faiss_file = Path(vector_index_path + ".faiss")

            if faiss_file.exists() and Path(vector_metadata_path).exists():
                # 벡터 스토어에 인덱스 로드
                self.vector_store.load_index(str(faiss_file))
                self.logger.info(f"Loaded precedent vector index from {faiss_file}")
            else:
                self.logger.warning(f"Precedent vector index not found at {vector_index_path}")
                self.logger.warning("Falling back to keyword search only")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None

        # 검색 설정
        self.search_config = {
            "max_results": 20,
            "similarity_threshold": 0.3,
            "fts_weight": 0.4,
            "vector_weight": 0.6,
            "category_weights": {
                "civil": 1.0,
                "criminal": 1.0,
                "family": 1.0
            }
        }

    def search_precedents(self,
                         query: str,
                         category: str = 'civil',
                         top_k: int = 10,
                         search_type: str = 'hybrid') -> List[PrecedentSearchResult]:
        """
        판례 검색 실행

        Args:
            query: 검색 쿼리
            category: 판례 카테고리 (civil, criminal, family)
            top_k: 반환할 결과 수
            search_type: 검색 유형 (fts, vector, hybrid)

        Returns:
            List[PrecedentSearchResult]: 검색 결과 리스트
        """
        try:
            self.logger.info(f"Searching precedents: query='{query}', category='{category}', type='{search_type}'")

            results = []

            if search_type in ['fts', 'hybrid']:
                fts_results = self._search_fts(query, category, top_k)
                results.extend(fts_results)

            if search_type in ['vector', 'hybrid'] and self.vector_store:
                vector_results = self._search_vector(query, category, top_k)
                results.extend(vector_results)

            # 결과 통합 및 중복 제거
            merged_results = self._merge_and_deduplicate_results(results)

            # 점수 기반 정렬
            sorted_results = self._rank_results(merged_results, query, category)

            # 상위 결과 반환
            final_results = sorted_results[:top_k]

            self.logger.info(f"Found {len(final_results)} precedent results")
            return final_results

        except Exception as e:
            self.logger.error(f"Error searching precedents: {e}")
            return []

    def _search_fts(self, query: str, category: str, top_k: int) -> List[PrecedentSearchResult]:
        """FTS5 전체 텍스트 검색 (최적화된 버전)"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # FTS5 검색 쿼리 (안전한 파라미터 바인딩)
                fts_query = """
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

                # FTS5 쿼리 실행 (사용자 입력을 안전하게 처리)
                try:
                    # 사용자 입력을 FTS5 안전 쿼리로 변환
                    safe_query = self._make_fts5_safe_query(query)
                    cursor.execute(fts_query, (safe_query, top_k * 2))
                except Exception as e:
                    # FTS5 파라미터 바인딩 실패 시 대안 방법 사용
                    self.logger.warning(f"FTS5 parameter binding failed: {e}, trying alternative method")
                    safe_query = self._make_fts5_safe_query(query)
                    alternative_query = f"""
                    SELECT
                        case_id,
                        case_name,
                        case_number,
                        rank
                    FROM fts_precedent_cases
                    WHERE fts_precedent_cases MATCH '{safe_query}'
                    ORDER BY rank
                    LIMIT {top_k * 2}
                    """
                    cursor.execute(alternative_query)
                rows = cursor.fetchall()

                # 추가 정보를 위한 별도 쿼리 (필요한 경우에만)
                results = []
                for row in rows:
                    # 카테고리 필터링을 위한 추가 쿼리
                    category_query = """
                    SELECT category, court, decision_date, field, full_text
                    FROM precedent_cases
                    WHERE case_id = ?
                    """
                    cursor.execute(category_query, (row['case_id'],))
                    case_info = cursor.fetchone()

                    if case_info and case_info['category'] == category:
                        result = PrecedentSearchResult(
                            case_id=row['case_id'],
                            case_name=row['case_name'],
                            case_number=row['case_number'],
                            category=case_info['category'],
                            court=case_info['court'],
                            decision_date=case_info['decision_date'],
                            field=case_info['field'],
                            summary=self._extract_summary(case_info['full_text']),
                            similarity=self._normalize_fts_score(row['rank']),
                            search_type="fts_optimized"
                        )

                        # 판시사항과 판결요지 추출
                        self._extract_judgment_info(result, row['case_id'])

                        results.append(result)

                        if len(results) >= top_k:
                            break

                self.logger.info(f"FTS search found {len(results)} results")
                return results

        except Exception as e:
            self.logger.error(f"FTS search error: {e}")
            return []

    def _search_vector(self, query: str, category: str, top_k: int) -> List[PrecedentSearchResult]:
        """벡터 유사도 검색"""
        try:
            if not self.vector_store:
                return []

            # 벡터 검색 실행
            vector_results = self.vector_store.search(query, top_k=top_k * 2)

            results = []
            for result in vector_results:
                # 메타데이터에서 판례 정보 추출
                metadata = result.get('metadata', {})
                case_id = metadata.get('case_id')

                if not case_id:
                    continue

                # 데이터베이스에서 상세 정보 조회
                case_info = self._get_case_info(case_id)
                if not case_info or case_info['category'] != category:
                    continue

                precedent_result = PrecedentSearchResult(
                    case_id=case_info['case_id'],
                    case_name=case_info['case_name'],
                    case_number=case_info['case_number'],
                    category=case_info['category'],
                    court=case_info['court'],
                    decision_date=case_info['decision_date'],
                    field=case_info['field'],
                    summary=self._extract_summary(case_info['full_text']),
                    similarity=result.get('score', 0.0),
                    search_type="vector"
                )

                # 판시사항과 판결요지 추출
                self._extract_judgment_info(precedent_result, case_id)

                results.append(precedent_result)

            self.logger.info(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []

    def _get_case_info(self, case_id: str) -> Optional[Dict[str, Any]]:
        """판례 사건 정보 조회"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT * FROM precedent_cases WHERE case_id = ?
                """

                cursor.execute(query, (case_id,))
                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except Exception as e:
            self.logger.error(f"Error getting case info: {e}")
            return None

    def _extract_judgment_info(self, result: PrecedentSearchResult, case_id: str):
        """판시사항과 판결요지 추출"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 판시사항과 판결요지 조회
                query = """
                SELECT section_type, section_content
                FROM precedent_sections
                WHERE case_id = ?
                AND section_type IN ('판시사항', '판결요지')
                ORDER BY
                    CASE section_type
                        WHEN '판시사항' THEN 1
                        WHEN '판결요지' THEN 2
                        ELSE 3
                    END
                """

                cursor.execute(query, (case_id,))
                rows = cursor.fetchall()

                for row in rows:
                    if row['section_type'] == '판시사항':
                        result.judgment_summary = row['section_content']
                    elif row['section_type'] == '판결요지':
                        result.judgment_gist = row['section_content']

        except Exception as e:
            self.logger.error(f"Error extracting judgment info: {e}")

    def _extract_summary(self, full_text: str, max_length: int = 200) -> str:
        """전체 텍스트에서 요약 추출"""
        if not full_text:
            return ""

        # 첫 번째 문장이나 지정된 길이만큼 추출
        sentences = full_text.split('.')
        if sentences:
            summary = sentences[0].strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary

        return full_text[:max_length] + "..." if len(full_text) > max_length else full_text

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
        # FTS rank는 낮을수록 좋은 점수이므로 역정규화
        if rank <= 0:
            return 1.0
        elif rank <= 1:
            return 0.9
        elif rank <= 5:
            return 0.7
        elif rank <= 10:
            return 0.5
        else:
            return 0.3

    def _merge_and_deduplicate_results(self, results: List[PrecedentSearchResult]) -> List[PrecedentSearchResult]:
        """결과 통합 및 중복 제거"""
        case_dict = {}

        for result in results:
            case_id = result.case_id

            if case_id not in case_dict:
                case_dict[case_id] = result
            else:
                # 기존 결과와 통합 (더 높은 점수 유지)
                existing = case_dict[case_id]
                if result.similarity > existing.similarity:
                    case_dict[case_id] = result
                elif result.similarity == existing.similarity:
                    # 검색 유형 우선순위: hybrid > vector > fts
                    type_priority = {'hybrid': 3, 'vector': 2, 'fts': 1}
                    if type_priority.get(result.search_type, 0) > type_priority.get(existing.search_type, 0):
                        case_dict[case_id] = result

        return list(case_dict.values())

    def _rank_results(self, results: List[PrecedentSearchResult], query: str, category: str) -> List[PrecedentSearchResult]:
        """결과 랭킹"""
        def calculate_final_score(result: PrecedentSearchResult) -> float:
            # 기본 유사도 점수
            base_score = result.similarity

            # 카테고리 가중치 적용
            category_weight = self.search_config['category_weights'].get(category, 1.0)

            # 검색 유형별 가중치
            search_type_weight = {
                'hybrid': 1.0,
                'vector': 0.9,
                'fts': 0.8
            }.get(result.search_type, 0.7)

            # 최종 점수 계산
            final_score = base_score * category_weight * search_type_weight

            return final_score

        # 점수 계산 및 정렬
        for result in results:
            result.similarity = calculate_final_score(result)

        return sorted(results, key=lambda x: x.similarity, reverse=True)

    def get_precedent_by_id(self, case_id: str) -> Optional[PrecedentSearchResult]:
        """ID로 판례 조회"""
        try:
            case_info = self._get_case_info(case_id)
            if not case_info:
                return None

            result = PrecedentSearchResult(
                case_id=case_info['case_id'],
                case_name=case_info['case_name'],
                case_number=case_info['case_number'],
                category=case_info['category'],
                court=case_info['court'],
                decision_date=case_info['decision_date'],
                field=case_info['field'],
                summary=self._extract_summary(case_info['full_text']),
                similarity=1.0,
                search_type="direct"
            )

            # 판시사항과 판결요지 추출
            self._extract_judgment_info(result, case_id)

            return result

        except Exception as e:
            self.logger.error(f"Error getting precedent by ID: {e}")
            return None

    def get_precedent_sections(self, case_id: str) -> List[Dict[str, Any]]:
        """판례 섹션 정보 조회"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT section_type, section_type_korean, section_content, section_length
                FROM precedent_sections
                WHERE case_id = ?
                ORDER BY
                    CASE section_type
                        WHEN '판시사항' THEN 1
                        WHEN '판결요지' THEN 2
                        WHEN '주문' THEN 3
                        ELSE 4
                    END
                """

                cursor.execute(query, (case_id,))
                rows = cursor.fetchall()

                sections = []
                for row in rows:
                    sections.append({
                        'section_type': row['section_type'],
                        'section_type_korean': row['section_type_korean'],
                        'section_content': row['section_content'],
                        'section_length': row['section_length']
                    })

                return sections

        except Exception as e:
            self.logger.error(f"Error getting precedent sections: {e}")
            return []

    def get_precedent_parties(self, case_id: str) -> List[Dict[str, Any]]:
        """판례 당사자 정보 조회"""
        try:
            with self.db_manager.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                SELECT party_type, party_type_korean, party_content, party_length
                FROM precedent_parties
                WHERE case_id = ?
                ORDER BY party_type
                """

                cursor.execute(query, (case_id,))
                rows = cursor.fetchall()

                parties = []
                for row in rows:
                    parties.append({
                        'party_type': row['party_type'],
                        'party_type_korean': row['party_type_korean'],
                        'party_content': row['party_content'],
                        'party_length': row['party_length']
                    })

                return parties

        except Exception as e:
            self.logger.error(f"Error getting precedent parties: {e}")
            return []


# 테스트 함수
def test_precedent_search_engine():
    """판례 검색 엔진 테스트"""
    engine = PrecedentSearchEngine()

    test_queries = [
        "손해배상",
        "계약 해지",
        "불법행위",
        "민사소송"
    ]

    print("=== 판례 검색 엔진 테스트 ===")
    for query in test_queries:
        print(f"\n검색 쿼리: {query}")

        # FTS 검색
        fts_results = engine.search_precedents(query, category='civil', top_k=3, search_type='fts')
        print(f"FTS 검색 결과: {len(fts_results)}개")
        for i, result in enumerate(fts_results[:2], 1):
            print(f"  {i}. {result.case_name} ({result.case_number}) - 유사도: {result.similarity:.3f}")

        # 벡터 검색
        vector_results = engine.search_precedents(query, category='civil', top_k=3, search_type='vector')
        print(f"벡터 검색 결과: {len(vector_results)}개")
        for i, result in enumerate(vector_results[:2], 1):
            print(f"  {i}. {result.case_name} ({result.case_number}) - 유사도: {result.similarity:.3f}")

        # 하이브리드 검색
        hybrid_results = engine.search_precedents(query, category='civil', top_k=3, search_type='hybrid')
        print(f"하이브리드 검색 결과: {len(hybrid_results)}개")
        for i, result in enumerate(hybrid_results[:2], 1):
            print(f"  {i}. {result.case_name} ({result.case_number}) - 유사도: {result.similarity:.3f}")


if __name__ == "__main__":
    test_precedent_search_engine()
