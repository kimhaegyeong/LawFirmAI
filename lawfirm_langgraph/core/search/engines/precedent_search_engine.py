# -*- coding: utf-8 -*-
"""
판례 전용 검색 엔진
판례 데이터베이스와 벡터 인덱스를 활용한 전문 검색 시스템
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
import sqlite3
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Database adapter import
try:
    from core.data.db_adapter import DatabaseAdapter
    from core.data.sql_adapter import SQLAdapter
except ImportError:
    try:
        from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
        from lawfirm_langgraph.core.data.sql_adapter import SQLAdapter
    except ImportError:
        DatabaseAdapter = None
        SQLAdapter = None

try:
    from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
except ImportError:
    from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
try:
    from lawfirm_langgraph.core.data.vector_store import LegalVectorStore
except ImportError:
    from core.data.vector_store import LegalVectorStore
try:
    from lawfirm_langgraph.core.utils.config import Config
except ImportError:
    from core.utils.config import Config
try:
    from lawfirm_langgraph.core.processing.extractors.ai_keyword_generator import AIKeywordGenerator
except ImportError:
    from core.processing.extractors.ai_keyword_generator import AIKeywordGenerator

try:
    from lawfirm_langgraph.core.utils.korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    try:
        from core.utils.korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        KoreanStopwordProcessor = None

logger = get_logger(__name__)


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
                 db_path: Optional[str] = None,
                 vector_index_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents",
                 vector_metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents.json"):
        """판례 검색 엔진 초기화"""
        self.logger = get_logger(__name__)

        if db_path is None:
            config = Config()
            db_path = config.database_path

        # KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor.get_instance()
            except Exception as e:
                self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")

        # 데이터베이스 연결
        self.db_manager = LegalDataConnectorV2(db_path)

        # 벡터 저장소 초기화
        self.vector_store = None
        self.vector_metadata = None

        try:
            import os
            model_name = os.getenv("EMBEDDING_MODEL")
            if model_name is None:
                from core.utils.config import Config
                config = Config()
                model_name = config.embedding_model
            
            self.vector_store = LegalVectorStore(
                model_name=model_name,
                dimension=768,
                index_type="flat"
            )

            # 기존 벡터 인덱스 로드
            if Path(vector_index_path).exists() and Path(vector_metadata_path).exists():
                # 벡터 스토어에 인덱스 로드
                self.vector_store.load_index(vector_index_path)
                self.logger.info(f"Loaded precedent vector index from {vector_index_path}")
            else:
                self.logger.warning(f"Precedent vector index not found at {vector_index_path}")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            self.vector_store = None

        # LLM 기반 키워드 확장기 초기화
        self.ai_keyword_generator = AIKeywordGenerator()
        self._keyword_cache = {}  # 키워드 확장 캐시
        self._max_cache_size = 100  # 최대 캐시 크기
        
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

                # FTS5 검색 쿼리 (안전한 파라미터 바인딩, bm25 사용)
                fts_query = """
                SELECT
                    case_id,
                    case_name,
                    case_number,
                    bm25(fts_precedent_cases) as rank_score
                FROM fts_precedent_cases
                WHERE fts_precedent_cases MATCH ?
                ORDER BY rank_score
                LIMIT ?
                """

                # FTS5 쿼리 실행 (사용자 입력을 안전하게 처리)
                try:
                    # 사용자 입력을 FTS5 안전 쿼리로 변환 (LLM 키워드 확장 포함)
                    safe_query = self._make_fts5_safe_query(query, category)
                    cursor.execute(fts_query, (safe_query, top_k * 2))
                except Exception as e:
                    # FTS5 파라미터 바인딩 실패 시 대안 방법 사용
                    self.logger.warning(f"FTS5 parameter binding failed: {e}, trying alternative method")
                    safe_query = self._make_fts5_safe_query(query, category)
                    alternative_query = f"""
                    SELECT
                        case_id,
                        case_name,
                        case_number,
                        bm25(fts_precedent_cases) as rank_score
                    FROM fts_precedent_cases
                    WHERE fts_precedent_cases MATCH '{safe_query}'
                    ORDER BY rank_score
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
                            similarity=self._normalize_fts_score(row.get('rank_score', -100.0)),
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

    def _extract_base_keywords(self, query: str) -> List[str]:
        """기본 키워드 추출 (조사 제거, 불용어 제거)"""
        import re
        
        if not query or not query.strip():
            return []
        
        query = query.strip()
        
        # FTS5 특수 문자 제거
        fts5_special_chars = ['"', '*', '^', '(', ')']
        for char in fts5_special_chars:
            query = query.replace(char, ' ')
        
        # 키워드 추출 (한글, 영문, 숫자만)
        keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        
        if not keywords:
            return []
        
        # 조사 제거 패턴 (한글 조사)
        josa_pattern = r'(에|에서|로|으로|와|과|의|을|를|이|가|은|는|도|만|부터|까지|대해|관련)$'
        
        # 키워드 정리 (1글자 제외, 불용어 제거, 조사 제거 - KoreanStopwordProcessor 사용)
        filtered_keywords = []
        for kw in keywords:
            # 조사 제거
            cleaned_kw = re.sub(josa_pattern, '', kw)
            if cleaned_kw and len(cleaned_kw) > 1:
                if not self.stopword_processor or not self.stopword_processor.is_stopword(cleaned_kw):
                    filtered_keywords.append(cleaned_kw)
        
        return filtered_keywords if filtered_keywords else keywords[:3]
    
    async def _expand_keywords_with_llm(self, query: str, base_keywords: List[str], 
                                       category: str = 'civil') -> List[str]:
        """LLM을 사용한 키워드 확장 (판례 검색용)"""
        try:
            if not base_keywords:
                return []
            
            # 캐시 키 생성
            cache_key = f"{query}:{category}:{':'.join(sorted(base_keywords))}"
            if cache_key in self._keyword_cache:
                cached_value = self._keyword_cache.pop(cache_key)
                self._keyword_cache[cache_key] = cached_value
                return cached_value
            
            # 도메인 매핑 (category -> domain)
            domain_map = {
                'civil': '민사법',
                'criminal': '형사법',
                'family': '가족법'
            }
            domain = domain_map.get(category, 'general')
            
            # 키워드가 너무 적으면 LLM 호출 스킵
            if len(base_keywords) < 2:
                return base_keywords
            
            # LLM 키워드 확장 (타임아웃 5초)
            try:
                expansion_result = await asyncio.wait_for(
                    self.ai_keyword_generator.expand_domain_keywords(
                        domain=domain,
                        base_keywords=base_keywords,
                        target_count=20
                    ),
                    timeout=5.0
                )
                
                if expansion_result.api_call_success and expansion_result.expanded_keywords:
                    expanded = base_keywords + expansion_result.expanded_keywords
                    expanded = list(dict.fromkeys(expanded))[:25]
                    self._add_to_cache(cache_key, expanded)
                    return expanded
                else:
                    # 폴백 확장
                    fallback_keywords = self.ai_keyword_generator.expand_keywords_with_fallback(
                        domain, base_keywords
                    )
                    expanded = list(dict.fromkeys(base_keywords + fallback_keywords))[:25]
                    self._add_to_cache(cache_key, expanded)
                    return expanded
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"판례 검색 LLM 키워드 확장 타임아웃")
                return base_keywords
            except Exception as e:
                self.logger.error(f"판례 검색 LLM 키워드 확장 오류: {e}")
                return base_keywords
                
        except Exception as e:
            self.logger.error(f"판례 검색 키워드 확장 중 오류: {e}")
            return base_keywords
    
    def _add_to_cache(self, cache_key: str, value: List[str]):
        """캐시에 추가 (LRU 방식)"""
        try:
            if len(self._keyword_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._keyword_cache))
                del self._keyword_cache[oldest_key]
            self._keyword_cache[cache_key] = value
        except Exception as e:
            self.logger.error(f"캐시 추가 중 오류: {e}")
    
    def _make_fts5_safe_query(self, query: str, category: str = 'civil') -> str:
        """FTS5 안전 쿼리 변환 (LLM 키워드 확장 포함)"""
        if not query or not query.strip():
            return '""'
        
        # 기본 키워드 추출
        base_keywords = self._extract_base_keywords(query)
        if not base_keywords:
            return '""'
        
        # LLM 키워드 확장 (동기 래퍼)
        expanded_keywords = base_keywords
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    expanded_keywords = asyncio.run(
                        self._expand_keywords_with_llm(query, base_keywords, category)
                    )
                else:
                    expanded_keywords = loop.run_until_complete(
                        self._expand_keywords_with_llm(query, base_keywords, category)
                    )
            except RuntimeError:
                expanded_keywords = asyncio.run(
                    self._expand_keywords_with_llm(query, base_keywords, category)
                )
        except Exception as e:
            self.logger.warning(f"판례 검색 LLM 키워드 확장 실패, 기본 키워드 사용: {e}")
            expanded_keywords = base_keywords
        
        if not expanded_keywords:
            expanded_keywords = base_keywords
        
        # FTS5 쿼리 생성 (최대 5개 키워드)
        selected_keywords = expanded_keywords[:5]
        
        if len(selected_keywords) == 1:
            safe_query = f'"{selected_keywords[0]}"'
        else:
            quoted_keywords = [f'"{kw}"' for kw in selected_keywords]
            safe_query = ' OR '.join(quoted_keywords)
        
        return safe_query

    def _normalize_fts_score(self, rank_score: float) -> float:
        """FTS BM25 점수를 0-1 범위로 정규화"""
        # BM25 rank_score는 음수이므로 절댓값 사용
        if rank_score is None:
            return 0.5
        
        abs_score = abs(rank_score)
        
        # BM25 점수를 0-1 범위로 정규화
        if abs_score <= 1:
            return 1.0
        elif abs_score <= 5:
            return 0.9
        elif abs_score <= 10:
            return 0.8
        elif abs_score <= 20:
            return 0.7
        elif abs_score <= 50:
            return 0.6
        else:
            return max(0.3, 1.0 / (1.0 + abs_score / 100.0))

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
