# -*- coding: utf-8 -*-
"""
lawfirm_v2.db 전용 법률 데이터 연동 서비스
PostgreSQL tsvector 키워드 검색 + 벡터 의미 검색 지원
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import os
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
    from lawfirm_langgraph.core.utils.korean_stopword_processor import KoreanStopwordProcessor
except ImportError:
    try:
        from core.utils.korean_stopword_processor import KoreanStopwordProcessor
    except ImportError:
        KoreanStopwordProcessor = None

logger = get_logger(__name__)


# Query routing patterns
ARTICLE_PATTERN = re.compile(r"제\s*\d+\s*조")
DATE_PATTERN = re.compile(r"\d{4}[.\-]\s*\d{1,2}[.\-]\s*\d{1,2}")
COURT_KEYWORDS = ["대법원", "고등법원", "지방법원", "가정법원", "행정법원"]


def route_query(query: str) -> str:
    """Query routing: 'text2sql' or 'vector'"""
    q = (query or "").strip()
    if not q:
        return "vector"

    # Strong textual cues → Text2SQL
    if ARTICLE_PATTERN.search(q):
        return "text2sql"
    if DATE_PATTERN.search(q):
        return "text2sql"
    if any(k in q for k in COURT_KEYWORDS):
        return "text2sql"
    if re.search(r"(사건|사건번호|doc_id|문서번호)", q):
        return "text2sql"

    return "vector"


class LegalDataConnectorV2:
    """lawfirm_v2.db 전용 법률 데이터베이스 연결 및 검색 서비스"""
    
    _okt_logged: bool = False  # KoNLPy Okt 초기화 로그 출력 여부 (최초 1회만)

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # PostgreSQL URL 우선 확인 (build_database_url 사용)
            database_url = None
            
            # 1. scripts.ingest.open_law.utils의 build_database_url 사용 시도
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).parents[4]  # legal_data_connector_v2.py -> connectors -> search -> core -> lawfirm_langgraph -> 프로젝트 루트
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from scripts.ingest.open_law.utils import build_database_url
                database_url = build_database_url()
            except ImportError:
                pass
            
            # 2. PostgreSQL 환경 변수 직접 확인
            if not database_url or not database_url.startswith('postgresql'):
                from urllib.parse import quote_plus
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                db = os.getenv('POSTGRES_DB')
                user = os.getenv('POSTGRES_USER')
                password = os.getenv('POSTGRES_PASSWORD')
                if db and user and password:
                    encoded_password = quote_plus(password)
                    database_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
            
            # 3. Config에서 확인 (PostgreSQL URL만)
            if not database_url or not database_url.startswith('postgresql'):
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
                from core.utils.config import Config
                config = Config()
                config_url = getattr(config, 'database_url', None)
                if config_url and config_url.startswith('postgresql'):
                    database_url = config_url
            
            if database_url and database_url.startswith('postgresql'):
                self.database_url = database_url
                self.db_path = None
            else:
                raise ValueError(
                    "PostgreSQL database URL is required. "
                    "Please set POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD "
                    "or DATABASE_URL (must start with 'postgresql://')"
                )
        else:
            # db_path가 제공된 경우 PostgreSQL URL로 변환 시도
            if db_path and (db_path.startswith('postgresql://') or db_path.startswith('postgres://')):
                self.database_url = db_path
                self.db_path = None
            else:
                raise ValueError(f"Invalid database path: {db_path}. PostgreSQL URL is required (e.g., postgresql://user:password@host:port/database)")
        
        self.logger = get_logger(__name__)
        
        # DatabaseAdapter 초기화 (필수)
        if not DatabaseAdapter:
            raise ImportError("DatabaseAdapter is required. PostgreSQL support is mandatory.")
        
        try:
            self._db_adapter = DatabaseAdapter(self.database_url)
            # DatabaseAdapter 초기화 로그는 DatabaseAdapter 내부에서 출력되므로 중복 방지
            # (캐시에서 재사용 시에는 DEBUG 레벨로만 출력됨)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DatabaseAdapter: {e}") from e
        
        self._connection_pool = None
        
        # 데이터베이스 URL 로깅
        self.logger.info(f"LegalDataConnectorV2 initialized with database URL: {self._mask_url(self.database_url)}")
        
        # FTS 테이블 존재 여부 확인
        self._check_fts_tables()
        
        # 🔥 수정: KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용, 싱글톤)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor.get_instance()
            except Exception as e:
                self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")
        
        # 🔥 수정: KoNLPy 형태소 분석기 초기화 (싱글톤 사용)
        self._okt = None
        try:
            from lawfirm_langgraph.core.utils.konlpy_singleton import get_okt_instance
            self._okt = get_okt_instance()
        except ImportError:
            try:
                from core.utils.konlpy_singleton import get_okt_instance
                self._okt = get_okt_instance()
            except ImportError:
                # 폴백: 직접 초기화 (싱글톤 유틸리티가 없는 경우)
                try:
                    from konlpy.tag import Okt  # pyright: ignore[reportMissingImports]
                    self._okt = Okt()
                except (ImportError, Exception):
                    pass
        
        # PostgreSQL 정규화 계수 설정 (환경 변수 지원, 기본값 15.0)
        self.postgresql_normalization_coefficient = float(
            os.getenv("POSTGRESQL_NORMALIZATION_COEFFICIENT", "15.0")
        )
        self.logger.info(
            f"PostgreSQL normalization coefficient: {self.postgresql_normalization_coefficient}"
        )
        
        # 테이블별 설정 딕셔너리 (테이블명: (text_vector_column, text_content_column, table_alias))
        self.table_configs = {
            'statutes_articles': ('text_search_vector', 'article_content', 'sa'),
            'precedent_contents': ('text_search_vector', 'section_content', 'pc'),
            'statute_articles': ('text_search_vector', 'text', 'sa'),
            'case_paragraphs': ('text_search_vector', 'text', 'cp'),
            'decision_paragraphs': ('text_search_vector', 'text', 'dp'),
            'interpretation_paragraphs': ('text_search_vector', 'text', 'ip'),
        }
        
        # PGroonga 사용 가능 여부 확인 (초기화 시 한 번만 확인)
        # Docker 환경에서는 PGroonga가 필수이므로 항상 사용 가능하다고 가정
        self._pgroonga_available = None
        if self._db_adapter and self._db_adapter.db_type == "postgresql":
            pgroonga_available = self._check_pgroonga_available()
            if not pgroonga_available:
                self.logger.warning(
                    "⚠️ PGroonga is not available. Korean text search ('korean' config) requires PGroonga. "
                    "Please ensure PGroonga is installed in your PostgreSQL instance."
                )
    
    def _mask_url(self, url: str) -> str:
        """URL에서 비밀번호 마스킹"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.password:
                return url.replace(parsed.password, "***")
        except Exception:
            pass
        return url
    
    def _check_column_exists(self, table_name: str, column_name: str) -> bool:
        """
        PostgreSQL 테이블에 컬럼이 존재하는지 확인
        
        Args:
            table_name: 테이블명
            column_name: 컬럼명
            
        Returns:
            컬럼 존재 여부
        """
        try:
            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = %s 
                        AND column_name = %s
                    )
                """, (table_name, column_name))
                row = cursor.fetchone()
                result = row[0] if row else False
                return bool(result)
        except Exception as e:
            self.logger.warning(f"Error checking column existence for {table_name}.{column_name}: {e}")
            return False
    
    def _check_pgroonga_available(self) -> bool:
        """
        PGroonga 확장이 설치되어 있고 사용 가능한지 확인
        
        Returns:
            PGroonga 사용 가능 여부
        """
        # 이미 확인한 경우 캐시된 값 반환
        if hasattr(self, '_pgroonga_available') and self._pgroonga_available is not None:
            return self._pgroonga_available
        
        self._pgroonga_available = False
        try:
            # Connection Pool 안전 사용 (컨텍스트 매니저)
            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                try:
                    # 1. PGroonga 확장 확인
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_extension 
                            WHERE extname = 'pgroonga'
                        ) as exists
                    """)
                    row = cursor.fetchone()
                    # RealDictCursor 사용 시 딕셔너리 형태로 반환되므로 안전하게 접근
                    has_extension = row.get('exists', False) if row else False
                    # 딕셔너리가 아닌 경우를 대비한 폴백
                    if not isinstance(has_extension, bool):
                        has_extension = list(row.values())[0] if row else False
                    
                    if not has_extension:
                        self.logger.debug("PGroonga extension is not installed. Using 'simple' text search configuration.")
                        return False
                    
                    # 2. PGroonga 함수 확인 (여러 함수 확인하여 더 안정적인 감지)
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_proc 
                            WHERE proname IN ('pgroonga_query_extract_keywords', 'pgroonga_match_positions_byte', 'pgroonga_normalize')
                        ) as exists
                    """)
                    row = cursor.fetchone()
                    # RealDictCursor 사용 시 딕셔너리 형태로 반환되므로 안전하게 접근
                    has_function = row.get('exists', False) if row else False
                    # 딕셔너리가 아닌 경우를 대비한 폴백
                    if not isinstance(has_function, bool):
                        has_function = list(row.values())[0] if row else False
                    
                    if has_function:
                        self._pgroonga_available = True
                        self.logger.info("✅ PGroonga extension is available. Korean text search will be enhanced.")
                    else:
                        # 확장은 있지만 함수가 없는 경우 - 더 자세한 디버깅 정보 제공
                        cursor.execute("""
                            SELECT proname, pronamespace::regnamespace as schema
                            FROM pg_proc 
                            WHERE proname LIKE '%pgroonga%' 
                            ORDER BY proname
                            LIMIT 10
                        """)
                        functions = cursor.fetchall()
                        # RealDictCursor 사용 시 딕셔너리 형태로 반환되므로 안전하게 접근
                        found_functions = []
                        if functions:
                            for row in functions:
                                if isinstance(row, dict):
                                    proname = row.get('proname', '')
                                    schema = row.get('schema', '')
                                else:
                                    # 튜플 형태인 경우 (폴백)
                                    proname = row[0] if len(row) > 0 else ''
                                    schema = row[1] if len(row) > 1 else ''
                                if proname:
                                    found_functions.append(f"{proname} ({schema})")
                        
                        if found_functions:
                            self.logger.warning(
                                f"PGroonga extension exists but required functions are not available. "
                                f"Found PGroonga functions: {', '.join(found_functions)}. "
                                f"Please ensure PGroonga is properly installed and all functions are available."
                            )
                        else:
                            self.logger.warning(
                                "PGroonga extension exists but no PGroonga functions found. "
                                "Please ensure PGroonga is properly installed: CREATE EXTENSION IF NOT EXISTS pgroonga;"
                            )
                        self._pgroonga_available = False
                        
                finally:
                    cursor.close()
                    
        except Exception as e:
            # 예외 발생 시 더 자세한 로그 출력 (WARNING 레벨)
            error_type = type(e).__name__
            error_msg = str(e)
            self.logger.warning(
                f"Error checking PGroonga availability: {error_type}: {error_msg}. "
                f"Using 'simple' text search configuration."
            )
            # DEBUG 레벨에서 상세한 traceback 제공
            import traceback
            self.logger.debug(f"PGroonga check traceback:\n{traceback.format_exc()}")
            self._pgroonga_available = False
        
        return self._pgroonga_available
    
    def _normalize_relevance_score(self, rank_score: float, log_context: Optional[str] = None) -> float:
        """
        PostgreSQL ts_rank_cd 점수를 relevance_score로 정규화
        
        Args:
            rank_score: PostgreSQL ts_rank_cd 점수
            log_context: 로깅 컨텍스트 (선택적)
            
        Returns:
            정규화된 relevance_score (0.0 ~ 1.0)
        """
        # 🔥 개선: rank_score >= 1.0인 경우에도 차별화를 위해 로그 스케일 사용
        if rank_score >= 1.0:
            # 로그 스케일을 사용하여 높은 rank_score도 차별화
            # rank_score가 클수록 relevance_score가 높아지지만, 1.0에 가까워지도록 제한
            # 예: rank_score=319 → log(319+1) / log(1000) ≈ 0.8
            import math
            # 최대 rank_score를 1000으로 가정 (실제 최대값에 따라 조정 가능)
            max_rank_score = 1000.0
            # 로그 스케일 정규화: log(rank_score + 1) / log(max_rank_score + 1)
            log_normalized = math.log(rank_score + 1) / math.log(max_rank_score + 1)
            # 0.7 ~ 1.0 범위로 매핑 (높은 rank_score도 차별화)
            relevance_score = 0.7 + (log_normalized * 0.3)
            relevance_score = min(1.0, max(0.7, relevance_score))
        else:
            relevance_score = max(0.0, min(1.0, rank_score * self.postgresql_normalization_coefficient))
        
        # 디버그 로깅
        if self.logger.isEnabledFor(logging.DEBUG) and log_context:
            self.logger.debug(
                f"[RANK_SCORE] {log_context} "
                f"rank_score={rank_score:.6f}, "
                f"relevance_score={relevance_score:.4f}, "
                f"coefficient={self.postgresql_normalization_coefficient}, "
                f"normalized={'log_scale' if rank_score >= 1.0 else 'linear'}"
            )
        
        return relevance_score
    
    @contextmanager
    def _db_connection_context(self):
        """
        데이터베이스 연결 컨텍스트 매니저 (간소화 버전)
        
        Usage:
            with self._db_connection_context() as (conn, cursor):
                cursor.execute(...)
        """
        conn_wrapper = None
        cursor = None
        try:
            conn_wrapper = self._get_connection()
            conn = conn_wrapper.conn if hasattr(conn_wrapper, 'conn') else conn_wrapper
            cursor = conn_wrapper.cursor()
            yield (conn_wrapper, cursor)
        finally:
            # 리소스 정리
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            # PostgreSQL 연결 풀에 반환
            if conn_wrapper and self._db_adapter and self._db_adapter.connection_pool:
                try:
                    if hasattr(conn_wrapper, 'conn'):
                        if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                            self._db_adapter.connection_pool.putconn(conn_wrapper.conn)
                    else:
                        if hasattr(conn_wrapper, 'closed') and conn_wrapper.closed == 0:
                            self._db_adapter.connection_pool.putconn(conn_wrapper)
                except Exception as e:
                    self.logger.warning(f"Error returning connection to pool: {e}")
    
    def _build_statute_article_result(self, row: Dict[str, Any], text_content: str, 
                                     relevance_score: float, search_type: str = "keyword",
                                     fallback_strategy: Optional[str] = None,
                                     direct_match: bool = False) -> Dict[str, Any]:
        """
        법령 조문 검색 결과 딕셔너리 생성
        
        Args:
            row: 데이터베이스 행 (딕셔너리 또는 튜플)
            text_content: 텍스트 내용
            relevance_score: 관련도 점수
            search_type: 검색 타입
            fallback_strategy: 폴백 전략 (선택적)
            direct_match: 직접 매칭 여부
            
        Returns:
            검색 결과 딕셔너리
        """
        # row가 딕셔너리가 아닌 경우 처리
        if not isinstance(row, dict):
            # 튜플인 경우 기본 필드 매핑 (실제로는 RealDictCursor 사용하므로 dict일 가능성 높음)
            return {}
        
        return {
            "id": f"statute_article_{row.get('id', 'unknown')}",
            "type": "statute_article",
            "content": text_content,
            "text": text_content,
            "source": row.get('statute_name', ''),
            "statute_name": row.get('statute_name', ''),
            "article_no": row.get('article_no'),
            "clause_no": row.get('clause_no'),
            "item_no": row.get('item_no'),
            "metadata": {
                "statute_id": row.get('statute_id'),
                "article_no": row.get('article_no'),
                "clause_no": row.get('clause_no'),
                "item_no": row.get('item_no'),
                "heading": row.get('heading'),
                "statute_abbrv": row.get('statute_abbrv'),
                "statute_type": row.get('statute_type'),
                "category": row.get('category'),
                "type": "statute_article"
            },
            "relevance_score": relevance_score,
            "search_type": search_type,
            "fallback_strategy": fallback_strategy,
            "direct_match": direct_match
        }
    
    def _build_precedent_result(self, row: Dict[str, Any], text_content: str,
                               relevance_score: float, search_type: str = "keyword",
                               fallback_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        판례 검색 결과 딕셔너리 생성
        
        Args:
            row: 데이터베이스 행
            text_content: 텍스트 내용
            relevance_score: 관련도 점수
            search_type: 검색 타입
            fallback_strategy: 폴백 전략 (선택적)
            
        Returns:
            검색 결과 딕셔너리
        """
        if not isinstance(row, dict):
            return {}
        
        return {
            "id": f"case_para_{row.get('id', 'unknown')}",
            "type": "precedent_content",
            "content": text_content,
            "text": text_content,
            "source": f"{row.get('court', '')} {row.get('doc_id', '')}".strip(),
            "metadata": {
                "precedent_id": row.get('precedent_id'),
                "doc_id": row.get('doc_id'),
                "court": row.get('court'),
                "case_type": row.get('case_type'),
                "casenames": row.get('casenames'),
                "announce_date": row.get('announce_date'),
                "type": "precedent_content",
            },
            "relevance_score": relevance_score,
            "search_type": search_type,
            "fallback_strategy": fallback_strategy
        }
    
    def _build_decision_result(self, row: Dict[str, Any], text_content: str,
                              relevance_score: float, search_type: str = "keyword",
                              fallback_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        심결례 검색 결과 딕셔너리 생성
        
        Args:
            row: 데이터베이스 행
            text_content: 텍스트 내용
            relevance_score: 관련도 점수
            search_type: 검색 타입
            fallback_strategy: 폴백 전략 (선택적)
            
        Returns:
            검색 결과 딕셔너리
        """
        if not isinstance(row, dict):
            return {}
        
        return {
            "id": f"decision_para_{row.get('id', 'unknown')}",
            "type": "decision",
            "content": text_content,
            "text": text_content,
            "source": f"{row.get('org', '')} {row.get('doc_id', '')}".strip(),
            "metadata": {
                "decision_id": row.get('decision_id'),
                "org": row.get('org'),
                "doc_id": row.get('doc_id'),
                "decision_date": row.get('decision_date'),
                "result": row.get('result'),
                "para_index": row.get('para_index'),
            },
            "relevance_score": relevance_score,
            "search_type": search_type,
            "fallback_strategy": fallback_strategy
        }
    
    def _build_interpretation_result(self, row: Dict[str, Any], text_content: str,
                                    relevance_score: float, search_type: str = "keyword",
                                    fallback_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        유권해석 검색 결과 딕셔너리 생성
        
        Args:
            row: 데이터베이스 행
            text_content: 텍스트 내용
            relevance_score: 관련도 점수
            search_type: 검색 타입
            fallback_strategy: 폴백 전략 (선택적)
            
        Returns:
            검색 결과 딕셔너리
        """
        if not isinstance(row, dict):
            return {}
        
        return {
            "id": f"interpretation_para_{row.get('id', 'unknown')}",
            "type": "interpretation",
            "content": text_content,
            "text": text_content,
            "source": f"{row.get('org', '')} {row.get('title', '')}".strip(),
            "metadata": {
                "interpretation_id": row.get('interpretation_id'),
                "org": row.get('org'),
                "doc_id": row.get('doc_id'),
                "title": row.get('title'),
                "response_date": row.get('response_date'),
                "para_index": row.get('para_index'),
            },
            "relevance_score": relevance_score,
            "search_type": search_type,
            "fallback_strategy": fallback_strategy
        }

    def _check_fts_tables(self):
        """PostgreSQL embeddings 테이블 확인"""
        if not self._db_adapter or self._db_adapter.db_type != "postgresql":
            self.logger.warning("PostgreSQL only: FTS tables check skipped")
            return
        
        # embeddings 테이블 확인
        has_embeddings = False
        try:
            has_embeddings = self._db_adapter.table_exists('embeddings')
        except Exception as e:
            self.logger.warning(f"Error checking embeddings table: {e}")
        
        if not has_embeddings:
            self.logger.warning(
                "⚠️ embeddings table not found. "
                "Semantic search will not work until embeddings are generated."
            )
        else:
            self.logger.info("✅ embeddings table exists.")

    def _convert_fts5_to_postgresql_fts(self, query: str, table_alias: str = 'sa', text_vector_column: str = 'text_search_vector', text_content_column: str = None, table_name: str = None, use_pgroonga: Optional[bool] = None) -> tuple[str, str, str, str]:
        """
        쿼리를 PostgreSQL tsvector 쿼리로 변환 (PGroonga 지원)
        
        Args:
            query: 검색 쿼리 (공백으로 구분된 단어들 또는 OR 조건 포함)
            table_alias: 테이블 별칭 (예: 'sa', 'pc', 'dp', 'ip')
            text_vector_column: tsvector 컬럼명 (기본값: 'text_search_vector', 없으면 None)
            text_content_column: 실제 텍스트 컬럼명 (text_vector_column이 None일 때 사용, 예: 'article_content', 'section_content')
            table_name: 테이블명 (text_search_vector 컬럼 존재 여부 확인용, 선택적)
            use_pgroonga: PGroonga 사용 여부 (None이면 자동 감지)
        
        Returns:
            (WHERE 절, ORDER BY 절, rank_score 표현식, tsquery 문자열) 튜플
        """
        # PGroonga 사용 여부 결정
        if use_pgroonga is None:
            use_pgroonga = self._check_pgroonga_available()
        
        # PostgreSQL FTS 변환
        # 한국어 검색을 위해 plainto_tsquery 사용 (더 유연한 검색)
        if not query or not query.strip():
            return "1=0", "1", "0", ""
        
        query_clean = query.strip()
        
        # OR 조건이 있는 경우 처리
        # PGroonga 사용 시: &@~ 연산자로 'keyword1 OR keyword2' 형식 직접 사용
        # PostgreSQL 기본 사용 시: to_tsquery 형식으로 변환 (A | B)
        has_or_condition = " OR " in query_clean.upper()
        use_to_tsquery = False
        
        if has_or_condition:
            import re
            parts = re.split(r'\s+OR\s+', query_clean, flags=re.IGNORECASE)
            if len(parts) > 1:
                if use_pgroonga:
                    # PGroonga 사용 시: OR 조건을 그대로 유지 (PGroonga 쿼리 구문)
                    # 예: 'keyword1 OR keyword2' -> 'keyword1 OR keyword2' (그대로 사용)
                    query_clean = " OR ".join(part.strip() for part in parts if part.strip())
                    self.logger.debug(f"PGroonga OR condition detected, using as-is: '{query_clean}'")
                else:
                    # PostgreSQL 기본 사용 시: to_tsquery 형식으로 변환 (A | B)
                    or_parts = []
                    for part in parts:
                        part_clean = part.strip()
                        if part_clean:
                            # 공백으로 구분된 단어들을 &로 연결
                            words = part_clean.split()
                            if len(words) > 1:
                                or_parts.append(" & ".join(words))
                            else:
                                or_parts.append(part_clean)
                    
                    if or_parts:
                        # OR 조건을 |로 연결하여 to_tsquery 형식으로 변환
                        query_clean = " | ".join(or_parts)
                        self.logger.debug(f"OR condition detected, converted to to_tsquery: '{query_clean}'")
                        use_to_tsquery = True
                    else:
                        query_clean = parts[0].strip() if parts else query_clean
                        use_to_tsquery = False
            else:
                query_clean = parts[0] if parts else query_clean
                use_to_tsquery = False
        
        # text_search_vector 컬럼 존재 여부 확인 (table_name이 제공된 경우)
        if table_name and not text_vector_column and not text_content_column:
            if self._check_column_exists(table_name, 'text_search_vector'):
                text_vector_column = 'text_search_vector'
                self.logger.debug(f"text_search_vector column found in {table_name}")
        
        # PGroonga 사용 시
        if use_pgroonga:
            # PGroonga 전용 연산자 사용 (&@: 단일 키워드, &@~: 쿼리 구문)
            # PGroonga는 인덱스를 직접 사용하므로 to_tsvector/to_tsquery 불필요
            # 참고: https://pgroonga.github.io/tutorial/
            
            # 검색 대상 컬럼 결정 (PGroonga 인덱스가 있는 컬럼 사용)
            if text_content_column:
                # text_content_column 사용 (PGroonga 인덱스가 있으면 자동으로 사용됨)
                search_column = f"{table_alias}.{text_content_column}"
            elif text_vector_column:
                # text_search_vector 컬럼이 있으면 사용 (PGroonga 인덱스가 있을 수 있음)
                search_column = f"{table_alias}.{text_vector_column}"
            else:
                # 기본값: text_search_vector 컬럼 사용 시도
                search_column = f"{table_alias}.text_search_vector"
            
            # WHERE 절, ORDER BY 절, rank_score 표현식 생성 (PGroonga 전용 연산자 사용)
            # 참고: https://pgroonga.github.io/tutorial/
            if use_to_tsquery or has_or_condition:
                # &@~ 연산자 사용 (쿼리 구문 지원: keyword1 OR keyword2)
                # PGroonga 쿼리 구문: keyword1 OR keyword2 (공백은 AND, OR은 OR)
                where_clause = f"{search_column} &@~ %s"
                # pgroonga_score 함수 사용 (정확도 점수)
                # tableoid와 ctid는 시스템 컬럼이므로 테이블 별칭 없이도 접근 가능
                # 하지만 명확성을 위해 테이블 별칭 사용
                rank_score_expr = f"pgroonga_score({table_alias}.tableoid, {table_alias}.ctid)"
                order_clause = f"{rank_score_expr} DESC"
            else:
                # &@ 연산자 사용 (단일 키워드 또는 공백으로 구분된 단어들)
                # PGroonga는 공백으로 구분된 단어들을 자동으로 AND 조건으로 처리
                where_clause = f"{search_column} &@ %s"
                # pgroonga_score 함수 사용 (정확도 점수)
                rank_score_expr = f"pgroonga_score({table_alias}.tableoid, {table_alias}.ctid)"
                order_clause = f"{rank_score_expr} DESC"
        else:
            # PGroonga가 없으면 'simple' 설정 사용 (PGroonga 없이는 'korean' 설정이 제대로 작동하지 않음)
            self.logger.debug("PGroonga is not available. Using 'simple' text search configuration.")
            lang_config = 'simple'
            
            # tsvector 생성: text_vector_column이 있으면 사용, 없으면 text_content_column으로 생성
            if text_vector_column:
                # text_search_vector 컬럼 사용 (인덱스 활용, 성능 최적화)
                tsvector_expr = f"{table_alias}.{text_vector_column}"
            elif text_content_column:
                # text_content_column으로 tsvector 생성 ('simple' 설정 사용)
                tsvector_expr = f"to_tsvector('{lang_config}', {table_alias}.{text_content_column})"
            else:
                # 기본값: text_search_vector 컬럼 사용 시도
                tsvector_expr = f"{table_alias}.text_search_vector"
            
            # WHERE 절, ORDER BY 절, rank_score 표현식 생성
            # OR 조건이 있으면 to_tsquery 사용, 없으면 plainto_tsquery 사용
            if use_to_tsquery:
                # to_tsquery 사용 (OR 조건 지원)
                where_clause = f"{tsvector_expr} @@ to_tsquery('{lang_config}', %s)"
                rank_score_expr = f"ts_rank_cd({tsvector_expr}, to_tsquery('{lang_config}', %s))"
                order_clause = f"{rank_score_expr} DESC"
            else:
                # plainto_tsquery 사용 (일반 검색)
                where_clause = f"{tsvector_expr} @@ plainto_tsquery('{lang_config}', %s)"
                rank_score_expr = f"ts_rank_cd({tsvector_expr}, plainto_tsquery('{lang_config}', %s))"
                order_clause = f"{rank_score_expr} DESC"
        
        return where_clause, order_clause, rank_score_expr, query_clean
    
    def _get_query_params(self, tsquery: str, limit: int, use_pgroonga: Optional[bool] = None, rank_score_expr: Optional[str] = None) -> tuple:
        """
        SQL 쿼리 파라미터 튜플 생성 (PGroonga 사용 여부에 따라 플레이스홀더 개수 결정)
        
        Args:
            tsquery: 검색 쿼리 문자열
            limit: 결과 제한 개수
            use_pgroonga: PGroonga 사용 여부 (None이면 자동 감지)
            rank_score_expr: rank_score 표현식 (플레이스홀더 개수 확인용)
            
        Returns:
            SQL 쿼리 파라미터 튜플
        """
        if use_pgroonga is None:
            use_pgroonga = self._check_pgroonga_available()
        
        if use_pgroonga:
            # PGroonga: where_clause에 1개, rank_score_expr에 0개 (pgroonga_score는 플레이스홀더 없음)
            return (tsquery, limit)
        else:
            # PostgreSQL 기본: where_clause에 1개, rank_score_expr에 1개 (ts_rank_cd에 플레이스홀더 있음)
            # rank_score_expr에 플레이스홀더가 있는지 확인
            if rank_score_expr and '%s' in rank_score_expr:
                return (tsquery, tsquery, limit)
            else:
                # rank_score_expr에 플레이스홀더가 없으면 where_clause만 (예: 하드코딩된 경우)
                return (tsquery, limit)
    
    def _get_connection(self):
        """
        Get database connection (PostgreSQL only)
        
        Note: Each thread gets its own connection for thread safety.
        Connection pool reuses connections per thread to improve performance.
        """
        if not self._db_adapter:
            raise RuntimeError("DatabaseAdapter is required")
        
        return self._db_adapter.get_connection()

    def _analyze_query_plan(self, query: str, table_name: str) -> Optional[Dict[str, Any]]:
        """
        PostgreSQL tsvector 쿼리 실행 계획 분석

        Args:
            query: 검색 쿼리
            table_name: 실제 테이블명 (statutes_articles, case_paragraphs 등)

        Returns:
            실행 계획 정보 딕셔너리 또는 None
        """
        try:
            safe_query = self._sanitize_tsquery(query)
            if not safe_query:
                return None

            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()

                # 테이블별 텍스트 컬럼 매핑
                # ⚠️ 사용 중단 테이블 제거: case_paragraphs, decision_paragraphs, interpretation_paragraphs
                # Open Law 스키마 테이블만 사용: statutes_articles, precedent_contents
                text_column_map = {
                    'statutes_articles': 'article_content',  # Open Law 스키마
                    'precedent_contents': 'section_content',  # Open Law 스키마
                    # 사용 중단: 'case_paragraphs', 'decision_paragraphs', 'interpretation_paragraphs'
                }
                
                text_column = text_column_map.get(table_name, 'text')
                
                # PostgreSQL EXPLAIN
                explain_query = f"""
                    EXPLAIN
                    SELECT id, 
                           ts_rank_cd(to_tsvector('simple', {text_column}), 
                                      plainto_tsquery('simple', %s)) as rank_score
                    FROM {table_name}
                    WHERE to_tsvector('simple', {text_column}) @@ plainto_tsquery('simple', %s)
                    ORDER BY rank_score DESC
                    LIMIT 10
                """
                cursor.execute(explain_query, (safe_query, safe_query))
                plan_rows = cursor.fetchall()
                
                # 실행 계획 분석
                plan_info = {
                    "uses_index": any("Index" in str(row) or "GIN" in str(row) for row in plan_rows),
                    "scan_type": "GIN" if any("GIN" in str(row) for row in plan_rows) else "Seq Scan",
                    "plan_detail": [str(row) for row in plan_rows]
                }
                
                self.logger.debug(f"Query plan for '{query[:30]}': {plan_info}")
                return plan_info
        except Exception as e:
            self.logger.debug(f"Error analyzing query plan: {e}")
            return None

    def _optimize_tsquery(self, query: str) -> str:
        """
        tsquery 최적화 (형태소 분석 기반 조사/어미 제거)

        Args:
            query: 원본 쿼리

        Returns:
            최적화된 쿼리
        """
        # 🔥 수정: 안전한 체크 - hasattr 사용
        if hasattr(self, '_okt') and self._okt is not None:
            try:
                return self._optimize_tsquery_morphological(query)
            except Exception as e:
                self.logger.warning(f"Error in morphological analysis: {e}, using fallback")
                return self._optimize_tsquery_fallback(query)
        else:
            return self._optimize_tsquery_fallback(query)
    
    def _optimize_tsquery_morphological(self, query: str) -> str:
        """
        형태소 분석 기반 tsquery 최적화
        
        Args:
            query: 원본 쿼리
            
        Returns:
            최적화된 쿼리
        """
        import re
        
        # 형태소 분석으로 품사 태깅
        pos_tags = self._okt.pos(query)
        
        # 허용된 품사 태그
        # 조사: JKS, JKO, JKB, JKG, JKI, JKQ, JX, JC
        # 어미: EP, EF, EC, ETM, ETN
        # 보조용언: VX
        allowed_pos = [
            'NNG', 'NNP', 'NNB', 'NR', 'NP',  # 명사류 (일반명사, 고유명사, 의존명사, 수사, 대명사)
            'VV', 'VA', 'VX', 'VCP', 'VCN',   # 용언 (동사, 형용사, 보조용언, 긍정지정사, 부정지정사)
            'MM', 'MDT', 'MDN',                # 관형사, 수 관형사, 명사 관형사
            'SL', 'SH', 'SN', 'XR'             # 외국어, 한자, 숫자, 어근
        ]
        
        core_keywords = []
        
        # 법령명 추출 (우선순위 최고)
        law_match = re.search(r'([가-힣]+법|민법|형법|상법|행정법|헌법|노동법|가족법|특허법|상표법|저작권법)', query)
        if law_match:
            law_name = law_match.group(1)
            if law_name not in core_keywords:
                core_keywords.append(law_name)
        
        # 조문 번호 추출 (우선순위 높음)
        article_patterns = [
            r'제\s*\d+\s*조',  # 제750조
            r'제\s*\d+\s*조\s*제',  # 제750조 제
            r'\d+\s*조',  # 750조
        ]
        for pattern in article_patterns:
            article_match = re.search(pattern, query)
            if article_match:
                article_text = article_match.group().replace(' ', '').strip()
                if not article_text.startswith('제'):
                    article_text = '제' + article_text
                if article_text not in core_keywords:
                    core_keywords.append(article_text)
                break
        
        # 형태소 분석으로 핵심 단어 추출
        for word, pos in pos_tags:
            # 허용된 품사이거나 법률 관련 용어인 경우
            if (pos in allowed_pos or 
                word.endswith('법') or 
                re.match(r'^제\d+조$', word)):
                
                # 길이 체크 및 중복 방지
                if len(word) >= 2 and word not in core_keywords:
                    # 한글/영문 포함 단어만 (조문 번호는 이미 처리됨)
                    if re.match(r'^[가-힣a-zA-Z]+$', word) or re.match(r'^제\d+조$', word):
                        core_keywords.append(word)
                        if len(core_keywords) >= 5:  # 최대 5개
                            break
        
        # 핵심 키워드가 없으면 폴백 방식 사용
        if not core_keywords:
            return self._optimize_tsquery_fallback(query)
        
        optimized = " ".join(core_keywords)
        self.logger.debug(f"Query optimized (morphological): '{query}' -> '{optimized}'")
        
        return optimized
    
    def _optimize_tsquery_fallback(self, query: str) -> str:
        """
        tsquery 최적화 (폴백 방식: 기존 방식)
        
        Args:
            query: 원본 쿼리
            
        Returns:
            최적화된 쿼리
        """
        import re
        
        words = query.split()
        
        # 조사 패턴 (정규식으로 한 번에 제거)
        josa_pattern = re.compile(r'(에|에서|에게|한테|께|으로|로|의|을|를|이|가|는|은|와|과|도|만|부터|까지)$')
        
        core_keywords = []
        
        # 법령명 추출
        law_match = re.search(r'([가-힣]+법|민법|형법|상법|행정법|헌법|노동법|가족법|특허법|상표법|저작권법)', query)
        if law_match:
            law_name = law_match.group(1)
            if law_name not in core_keywords:
                core_keywords.append(law_name)
        
        # 조문 번호 추출
        article_patterns = [
            r'제\s*\d+\s*조',
            r'제\s*\d+\s*조\s*제',
            r'\d+\s*조',
        ]
        for pattern in article_patterns:
            article_match = re.search(pattern, query)
            if article_match:
                article_text = article_match.group().replace(' ', '').strip()
                if not article_text.startswith('제'):
                    article_text = '제' + article_text
                if article_text not in core_keywords:
                    core_keywords.append(article_text)
                break
        
        # 나머지 키워드 처리 (KoreanStopwordProcessor 사용)
        for w in words:
            w_clean = josa_pattern.sub('', w.strip())  # 조사 제거
            
            if not w_clean or len(w_clean) < 2:
                continue
            
            # 불용어 필터링 (KoreanStopwordProcessor 사용)
            if (self.stopword_processor and self.stopword_processor.is_stopword(w_clean)) or w_clean in core_keywords:
                continue
            
            # 한글/영문 포함 단어만
            if re.match(r'^[가-힣a-zA-Z]+$', w_clean) or re.match(r'^제\d+조$', w_clean):
                core_keywords.append(w_clean)
                if len(core_keywords) >= 5:
                    break
        
        # 핵심 키워드가 없으면 원본 단어 사용 (불용어만 제거)
        if not core_keywords:
            for w in words:
                w_clean = josa_pattern.sub('', w.strip())
                if w_clean and len(w_clean) >= 2:
                    if self.stopword_processor and not self.stopword_processor.is_stopword(w_clean):
                        core_keywords.append(w_clean)
                    elif not self.stopword_processor:
                        core_keywords.append(w_clean)
                    if len(core_keywords) >= 5:
                        break
        
        optimized = " ".join(core_keywords) if core_keywords else query
        self.logger.debug(f"Query optimized (fallback): '{query}' -> '{optimized}'")
        
        return optimized

    def _sanitize_tsquery(self, query: str) -> str:
        """
        tsquery를 안전하게 변환
        - 특수 문자가 있으면 이스케이프 처리
        - 빈 쿼리는 빈 문자열 반환
        - 단순 키워드 검색에 최적화
        - PostgreSQL tsquery는 기본적으로 공백으로 구분된 단어를 AND 조건으로 처리
        - 따옴표와 AND/OR 구문을 포함한 복잡한 쿼리 처리
        - 개선: 법령명과 조문번호 추출, "제XX조" 패턴 인식, 불필요한 단어 필터링
        """
        if not query or not query.strip():
            return ""
        
        import re
        query = query.strip()
        
        # 1단계: 법령명과 조문번호 추출 (개선)
        law_pattern = re.compile(r'([가-힣]+법)\s*제\s*(\d+)\s*조')
        law_matches = law_pattern.findall(query)
        
        # 2단계: "제XX조" 패턴 추출 (법령명 없이)
        article_pattern = re.compile(r'제\s*(\d+)\s*조')
        article_matches = article_pattern.findall(query)
        
        # 3단계: 따옴표로 묶인 구문 제거
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        query_no_quotes = re.sub(r'"[^"]*"', '', query)
        
        # 4단계: AND, OR, NOT 키워드 제거
        query_cleaned = re.sub(r'\b(AND|OR|NOT)\b', ' ', query_no_quotes, flags=re.IGNORECASE)
        
        # 5단계: 법령명과 조문번호를 구문으로 추가 (우선순위 높음)
        clean_words = []
        seen = set()
        article_nos_in_law_matches = set()
        
        # 법령명과 조문번호 조합 추가 (우선)
        for law_name, article_no in law_matches:
            phrase = f"{law_name} 제{article_no}조"
            if phrase not in seen:
                clean_words.append(phrase)
                seen.add(phrase)
                article_nos_in_law_matches.add(article_no)
        
        # "제XX조" 패턴 추가 (법령명과 조합되지 않은 경우만)
        for article_no in article_matches:
            if article_no not in article_nos_in_law_matches:
                phrase = f"제{article_no}조"
                if phrase not in seen:
                    clean_words.append(phrase)
                    seen.add(phrase)
        
        # 6단계: 따옴표로 묶인 구문에서 단어 추출
        for phrase in quoted_phrases:
            phrase_words = re.findall(r'[가-힣a-zA-Z0-9]+', phrase)
            for word in phrase_words:
                if len(word) >= 2 and len(word) <= 20 and word not in seen:
                    if not self.stopword_processor or not self.stopword_processor.is_stopword(word):
                        clean_words.append(word)
                        seen.add(word)
        
        # 7단계: 나머지 쿼리에서 단어 추출 (불필요한 단어 제외)
        remaining_words = re.findall(r'[가-힣a-zA-Z0-9]+', query_cleaned)
        for word in remaining_words:
            # 불필요한 단어 필터링 강화 (KoreanStopwordProcessor 사용)
            if len(word) >= 2 and len(word) <= 20 and word not in seen:
                if not self.stopword_processor or not self.stopword_processor.is_stopword(word):
                    clean_words.append(word)
                    seen.add(word)
        
        # 6단계: 단어가 없으면 빈 문자열 반환 (개선: 더 공격적인 폴백)
        if not clean_words:
            self.logger.warning(f"No valid words found in query: '{query[:100]}'")
            # 개선: 빈 문자열 대신 원본 쿼리의 첫 단어 사용 (폴백)
            if query:
                # 전략 1: 원본 쿼리에서 첫 번째 단어 추출 (더 공격적으로)
                first_word = re.findall(r'[가-힣a-zA-Z0-9]+', query)
                if first_word:
                    # 최소 1자 이상이면 사용
                    if len(first_word[0]) >= 1:
                        clean_words = [first_word[0]]
                        self.logger.info(f"Using fallback word from original query: '{clean_words[0]}'")
                    else:
                        # 전략 2: 공백으로 분리된 첫 단어 사용
                        words_by_space = query.strip().split()
                        if words_by_space:
                            first_word_space = re.findall(r'[가-힣a-zA-Z0-9]+', words_by_space[0])
                            if first_word_space and len(first_word_space[0]) >= 1:
                                clean_words = [first_word_space[0]]
                                self.logger.info(f"Using fallback word from space-separated query: '{clean_words[0]}'")
                            else:
                                # 전략 3: 모든 문자에서 첫 2자 이상 추출
                                all_chars = re.findall(r'[가-힣a-zA-Z0-9]{2,}', query)
                                if all_chars:
                                    clean_words = [all_chars[0]]
                                    self.logger.info(f"Using fallback word from all chars: '{clean_words[0]}'")
                                else:
                                    return ""
                        else:
                            return ""
                else:
                    # 전략 2: 공백으로 분리된 첫 단어 사용
                    words_by_space = query.strip().split()
                    if words_by_space:
                        first_word_space = re.findall(r'[가-힣a-zA-Z0-9]+', words_by_space[0])
                        if first_word_space and len(first_word_space[0]) >= 1:
                            clean_words = [first_word_space[0]]
                            self.logger.info(f"Using fallback word from space-separated query: '{clean_words[0]}'")
                        else:
                            # 전략 3: 모든 문자에서 첫 2자 이상 추출
                            all_chars = re.findall(r'[가-힣a-zA-Z0-9]{2,}', query)
                            if all_chars:
                                clean_words = [all_chars[0]]
                                self.logger.info(f"Using fallback word from all chars: '{clean_words[0]}'")
                            else:
                                return ""
                    else:
                        return ""
            else:
                return ""
        
        # 8단계: 최대 5개 단어/구문만 사용 (tsquery 성능 최적화)
        clean_words = clean_words[:5]
        
        # 9단계: tsquery 생성 (개선: 법령명+조문번호 구문은 하나의 구문으로 처리)
        if not clean_words:
            sanitized = ""
        elif len(clean_words) == 1:
            sanitized = clean_words[0]
        else:
            # 법령명+조문번호 구문이 있으면 우선 사용, 나머지는 OR
            law_article_phrases = [w for w in clean_words if '제' in w and '조' in w and '법' in w]
            article_only_phrases = [w for w in clean_words if '제' in w and '조' in w and '법' not in w]
            other_words = [w for w in clean_words if w not in law_article_phrases and w not in article_only_phrases]
            
            if law_article_phrases:
                # 법령명+조문번호 구문이 있으면 이것만 사용 (가장 정확함)
                sanitized = law_article_phrases[0]
                if other_words:
                    # 추가 키워드가 있으면 OR로 연결
                    sanitized = f"{sanitized} OR {' OR '.join(other_words[:2])}"
            elif article_only_phrases:
                # "제XX조"만 있으면 이것을 우선 사용
                sanitized = article_only_phrases[0]
                if other_words:
                    sanitized = f"{sanitized} OR {' OR '.join(other_words[:2])}"
            else:
                # 일반 단어만 있으면 OR 조건
                sanitized = " OR ".join(clean_words[:3])
        
        # 10단계: tsquery 정제 (개선: "OR OR" 오류 방지)
        # 빈 문자열 제거 및 중복 OR 제거
        sanitized = sanitized.strip()
        if sanitized:
            # "OR OR" 패턴 제거
            while " OR OR " in sanitized or sanitized.startswith("OR ") or sanitized.endswith(" OR"):
                sanitized = sanitized.replace(" OR OR ", " OR ")
                sanitized = sanitized.replace(" OR  OR ", " OR ")
                if sanitized.startswith("OR "):
                    sanitized = sanitized[3:].strip()
                if sanitized.endswith(" OR"):
                    sanitized = sanitized[:-3].strip()
            
            # 빈 문자열이면 원본 쿼리에서 첫 단어 사용
            if not sanitized or sanitized == "OR":
                first_word = re.findall(r'[가-힣a-zA-Z0-9]+', query)
                if first_word:
                    sanitized = first_word[0]
                else:
                    sanitized = ""
        
        # 11단계: SQL injection 방지
        if sanitized:
            sanitized = sanitized.replace("'", "''")
        
        self.logger.debug(f"tsquery sanitized: '{query[:100]}' -> '{sanitized}'")
        return sanitized

    def search_statutes_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """PostgreSQL tsvector 키워드 검색: 법령 조문"""
        try:
            # 🔥 개선: 조문 번호가 있으면 직접 검색 먼저 시도
            import re
            article_pattern = re.compile(r'제\s*(\d+)\s*조')
            article_match = article_pattern.search(query)
            law_pattern = re.compile(r'([가-힣]+법|민법|형법|상법|공법|사법)')
            law_match = law_pattern.search(query)
            
            if article_match and law_match:
                # 조문 번호 직접 검색 먼저 시도
                direct_results = self.search_statute_article_direct(query, limit=limit)
                if direct_results:
                    self.logger.info(f"✅ [TSVECTOR] 조문 직접 검색 성공: {len(direct_results)}개")
                    return direct_results
            
            # tsquery 최적화 및 안전화
            optimized_query = self._optimize_tsquery(query)
            safe_query = self._sanitize_tsquery(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid tsquery: '{query}'")
                return []
            
            # 검색 쿼리 로깅
            self.logger.info(f"TSVECTOR search: original='{query}', optimized='{optimized_query}', safe='{safe_query}'")
            
            # 쿼리 최적화: 불완전한 단어 제거 (예: "손해배상에" -> "손해배상")
            words = safe_query.split()
            cleaned_words = []
            original_cleaned = safe_query
            
            for w in words:
                # 조사나 불완전한 단어 끝 제거 (에, 에서, 에게 등)
                w_clean = re.sub(r'에$|에서$|에게$|한테$|께$', '', w)
                if w_clean and len(w_clean) >= 2:  # 최소 2자 이상
                    cleaned_words.append(w_clean)
                elif w_clean:  # 1자 단어도 추가 (예: "에")
                    cleaned_words.append(w_clean)
            
            if cleaned_words:
                new_safe_query = " ".join(cleaned_words)
                if new_safe_query != safe_query:
                    safe_query = new_safe_query
                    self.logger.info(f"Query cleaned: '{original_cleaned}' -> '{safe_query}'")
                    words = safe_query.split()  # words 업데이트

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "statutes_articles")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # text_search_vector 컬럼 존재 여부 확인
                has_text_search_vector = self._check_column_exists('statutes_articles', 'text_search_vector')
                
                # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                if has_text_search_vector:
                    where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                        safe_query, table_alias='sa', text_vector_column='text_search_vector', 
                        text_content_column=None, table_name='statutes_articles'
                    )
                else:
                    where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                        safe_query, table_alias='sa', text_vector_column=None, 
                        text_content_column='article_content', table_name='statutes_articles'
                    )
                
                if not tsquery:
                    self.logger.warning(f"Empty tsquery generated from query: '{safe_query}'")
                    return []
                
                # PostgreSQL tsvector 쿼리 (PGroonga 사용 시 'korean', 미사용 시 'simple' 설정 자동 적용)
                sql_query = f"""
                    SELECT
                        sa.id,
                        sa.statute_id,
                        sa.article_no,
                        sa.clause_no,
                        sa.item_no,
                        sa.article_title as heading,
                        sa.article_content as text,
                        s.law_name_kr as statute_name,
                        s.law_abbrv as statute_abbrv,
                        s.law_type as statute_type,
                        s.domain as category,
                        {rank_score_expr} as rank_score
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT %s
                """
                # 플레이스홀더 개수 자동 결정
                query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                cursor.execute(sql_query, query_params)

                results = []
                for row in cursor.fetchall():
                    # text 필드가 비어있으면 경고
                    text_content = row['text'] if row['text'] else ""
                    if not text_content:
                        self.logger.warning(f"Empty text content for statute article id={row['id']}, article_no={row['article_no']}")
                    
                    # PostgreSQL ts_rank_cd 점수 정규화
                    rank_score = row.get('rank_score', 0.0)
                    relevance_score = self._normalize_relevance_score(
                        rank_score, 
                        log_context=f"query='{query[:50]}', type=statute_article_tsvector"
                    )
                    
                    result = self._build_statute_article_result(
                        row, text_content, relevance_score, 
                        search_type="keyword", fallback_strategy=None
                    )
                    if result:
                        results.append(result)
            
            self.logger.info(f"TSVECTOR search found {len(results)} statute articles for query: '{query}'")
            
            # 결과가 없으면 폴백 검색 시도
            if len(results) == 0:
                self.logger.warning(f"No TSVECTOR results for query: '{query}'. Trying fallback strategies...")
                results = self._fallback_statute_search(query, safe_query, words, limit)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in tsvector search for query '{query}': {e}", exc_info=True)
            return []
    
    def search_statute_article_direct(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        법령 조문 직접 검색 (개선 #10)
        질문에서 법령명과 조문번호를 추출하여 해당 조문을 직접 검색
        
        Args:
            query: 검색 쿼리 (예: "민법 제750조 손해배상에 대해 설명해주세요")
            limit: 최대 결과 수
            
        Returns:
            검색 결과 리스트 (relevance_score: 1.0으로 설정)
        """
        import re
        results = []
        seen_ids = set()
        
        try:
            # 개선 1: 법령명 추출 패턴 확장 (민사법 위주)
            # 민사법 관련 법령명: 민법, 민사소송법, 계약법, 채권법, 물권법, 가족법, 상법, 상사법 등
            law_pattern = re.compile(r'([가-힣]+법|민법|형법|상법|행정법|헌법|노동법|가족법|민사소송법|형사소송법|상사법|공법|사법|계약법|채권법|물권법|가사소송법|가사법|호적법|부동산등기법|임대차보호법|집행법|강제집행법)')
            law_match = law_pattern.search(query)
            
            # 조문번호 추출 강화 (제750조, 제 750 조, 750조, 제750조 등 다양한 패턴 지원)
            article_patterns = [
                re.compile(r'제\s*(\d+)\s*조'),  # 제750조, 제 750 조
                re.compile(r'(\d+)\s*조'),      # 750조, 750 조
                re.compile(r'제\s*(\d+)'),      # 제750
                re.compile(r'(\d+)조'),          # 750조 (공백 없음)
            ]
            
            article_match = None
            article_no = None
            for pattern in article_patterns:
                article_match = pattern.search(query)
                if article_match:
                    article_no = article_match.group(1)
                    break
            
            if not law_match or not article_match:
                self.logger.debug(f"법령 조문 직접 검색: 법령명 또는 조문번호를 찾을 수 없음 (query: '{query}')")
                return []
            
            law_name = law_match.group(1)
            # article_no는 이미 위에서 추출됨
            
            self.logger.info(f"⚖️ [DIRECT STATUTE SEARCH] 법령명: {law_name}, 조문번호: {article_no}")
            
            # 🔥 개선: 조문 번호를 여러 형식으로 변환하여 시도
            # 로그 패턴 분석: 046800 = 제468조, 052600 = 제526조, 053700 = 제537조
            # 따라서 제750조는 "075000" 형식일 가능성이 높음
            article_no_clean = article_no.lstrip('0')  # 앞의 0 제거 ("0750" -> "750")
            article_no_variants = []
            
            if article_no_clean.isdigit():
                # 방법 1: 앞에 0 채우기 (기존 방식) - "750" -> "000750"
                variant1 = article_no_clean.zfill(6)
                # 방법 2: 뒤에 00 붙이기 (로그 패턴 기반) - "750" -> "075000"
                variant2 = article_no_clean.zfill(4) + "00"
                # 방법 3: 중간에 0 채우기 - "750" -> "007500"
                variant3 = article_no_clean.zfill(5) + "0"
                # 방법 4: 원본 그대로
                variant4 = article_no_clean
                
                article_no_variants = [variant1, variant2, variant3, variant4]
                # 중복 제거
                article_no_variants = list(dict.fromkeys(article_no_variants))
            else:
                article_no_variants = [article_no]
            
            self.logger.debug(f"조문 번호 형식 변환: '{article_no}' -> variants={article_no_variants}")
            
            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                # 개선 2: 법령명 매칭 로직 강화 (유사도 매칭)
                # 🔥 수정: PostgreSQL은 %s 플레이스홀더 사용
                # 🔥 수정: Open Law API 스키마는 law_name_kr, law_abbrv 사용
                # 전략 1: 정확한 이름 매칭 (동적 컬럼명 확인)
                # 먼저 컬럼 존재 여부 확인
                # 🔥 수정: RealDictCursor를 사용하므로 딕셔너리로 접근 가능
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'statutes' 
                    AND column_name IN ('name', 'law_name_kr', 'law_name')
                    LIMIT 1
                """)
                col_result = cursor.fetchone()
                # RealDictCursor는 딕셔너리처럼 접근 가능하지만, information_schema는 튜플일 수 있음
                if col_result:
                    if isinstance(col_result, dict):
                        name_col = col_result.get('column_name', 'law_name_kr')
                    else:
                        name_col = col_result[0] if len(col_result) > 0 else 'law_name_kr'
                else:
                    name_col = 'law_name_kr'  # 기본값
                
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'statutes' 
                    AND column_name IN ('abbrv', 'law_abbrv')
                    LIMIT 1
                """)
                abbrv_result = cursor.fetchone()
                # RealDictCursor는 딕셔너리처럼 접근 가능하지만, information_schema는 튜플일 수 있음
                if abbrv_result:
                    if isinstance(abbrv_result, dict):
                        abbrv_col = abbrv_result.get('column_name', 'law_abbrv')
                    else:
                        abbrv_col = abbrv_result[0] if len(abbrv_result) > 0 else 'law_abbrv'
                else:
                    abbrv_col = 'law_abbrv'  # 기본값
                
                # 동적 쿼리 생성
                cursor.execute(f"""
                    SELECT id, {name_col} as name, {abbrv_col} as abbrv 
                    FROM statutes 
                    WHERE {name_col} = %s OR {abbrv_col} = %s
                    LIMIT 1
                """, (law_name, law_name))
                
                statute_row = cursor.fetchone()
                
                # 전략 2: LIKE 검색 (정확한 매칭 실패 시)
                if not statute_row:
                    cursor.execute(f"""
                        SELECT id, {name_col} as name, {abbrv_col} as abbrv 
                        FROM statutes 
                        WHERE {name_col} LIKE %s OR {abbrv_col} LIKE %s OR {name_col} LIKE %s OR {abbrv_col} LIKE %s
                        LIMIT 5
                    """, (
                        f"%{law_name}%", 
                        f"%{law_name}%", 
                        f"{law_name}%", 
                        f"{law_name}%"
                    ))
                    
                    candidates = cursor.fetchall()
                    if candidates:
                        # 🔥 수정: RealDictCursor를 사용하므로 딕셔너리로 접근 가능
                        # 가장 유사한 법령명 선택 (길이가 가장 가까운 것)
                        def get_name_length(row):
                            if isinstance(row, dict):
                                return len(row.get('name', ''))
                            elif isinstance(row, tuple) and len(row) > 1:
                                return len(row[1])
                            return 999
                        
                        def get_name(row):
                            if isinstance(row, dict):
                                return row.get('name', '')
                            elif isinstance(row, tuple) and len(row) > 1:
                                return row[1]
                            return ''
                        
                        best_match = min(candidates, key=lambda x: abs(get_name_length(x) - len(law_name)) if get_name(x) else 999)
                        statute_row = best_match
                        statute_name = get_name(best_match)
                        self.logger.info(f"법령명 유사도 매칭 성공: '{law_name}' -> '{statute_name}'")
                
                if not statute_row:
                    self.logger.warning(f"법령을 찾을 수 없음: {law_name}")
                    return []
                
                # 🔥 수정: RealDictCursor를 사용하므로 딕셔너리로 접근 가능
                if isinstance(statute_row, dict):
                    statute_id = statute_row.get('id')
                elif isinstance(statute_row, tuple) and len(statute_row) > 0:
                    statute_id = statute_row[0]
                else:
                    statute_id = None
                
                if not statute_id:
                    self.logger.warning(f"법령 ID를 찾을 수 없음: {law_name}")
                    return []
                
                # 해당 조문 직접 조회 (여러 형식으로 시도)
                # 🔥 개선: 여러 article_no 형식을 OR 조건으로 시도
                where_conditions = " OR ".join(["sa.article_no = %s" for _ in article_no_variants])
                params = [statute_id] + article_no_variants + [limit * 2]
                
                cursor.execute(f"""
                    SELECT 
                        sa.id,
                        sa.statute_id,
                        sa.article_no,
                        sa.clause_no,
                        sa.item_no,
                        sa.article_title as heading,
                        sa.article_content as text,
                        s.law_name_kr as statute_name,
                        s.law_abbrv as statute_abbrv,
                        s.law_type as statute_type,
                        s.domain as category
                    FROM statutes_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.statute_id = %s AND ({where_conditions})
                    ORDER BY 
                        CASE WHEN sa.clause_no IS NULL THEN 1 ELSE 0 END,
                        sa.clause_no,
                        CASE WHEN sa.item_no IS NULL THEN 1 ELSE 0 END,
                        sa.item_no
                    LIMIT %s
                """, tuple(params))
                
                for row in cursor.fetchall():
                    if row['id'] not in seen_ids:
                        seen_ids.add(row['id'])
                        text_content = row['text'] if row['text'] else ""
                        
                        if not text_content:
                            continue
                        
                        # 직접 검색된 조문은 최고 relevance score 부여
                        results.append({
                            "id": f"statute_article_{row['id']}",
                            "type": "statute_article",
                            "content": text_content,
                            "text": text_content,
                            "source": row['statute_name'],
                            "metadata": {
                                "statute_id": row['statute_id'],
                                "statute_name": row['statute_name'],
                                "law_name": row['statute_name'],
                                "article_no": row['article_no'],
                                "clause_no": row['clause_no'],
                                "item_no": row['item_no'],
                                "heading": row['heading'],
                                "statute_abbrv": row['statute_abbrv'],
                                "statute_type": row['statute_type'],
                                "category": row['category'],
                                "type": "statute_article"
                            },
                            "relevance_score": 1.0,
                            "final_weighted_score": 1.0,
                            "search_type": "direct_statute",
                            "direct_match": True
                        })
            
            if results:
                self.logger.info(f"✅ [DIRECT STATUTE SEARCH] {len(results)}개 조문 직접 검색 성공: {law_name} 제{article_no}조")
            else:
                self.logger.warning(f"⚠️ [DIRECT STATUTE SEARCH] 조문을 찾을 수 없음: {law_name} 제{article_no}조")
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"법령 조문 직접 검색 오류: {e}", exc_info=True)
            return []
    
    def _fallback_statute_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        폴백 검색 전략: 결과가 없을 때 여러 방법으로 재시도
        
        Args:
            original_query: 원본 쿼리
            safe_query: 안전화된 쿼리
            words: 쿼리 단어 리스트
            limit: 최대 결과 수
            
        Returns:
            검색 결과 리스트
        """
        import re
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 법령명 + 조문 번호만으로 검색 (예: "민법 제750조") - 패턴 강화
            article_patterns = [
                re.compile(r'제\s*\d+\s*조'),  # 제750조, 제 750 조
                re.compile(r'\d+\s*조'),        # 750조, 750 조
                re.compile(r'제\s*\d+'),        # 제750
                re.compile(r'\d+조'),            # 750조 (공백 없음)
            ]
            article_match = None
            for pattern in article_patterns:
                article_match = pattern.search(original_query)
                if article_match:
                    break
            
            law_pattern = re.compile(r'([가-힣]+법|민법|형법|상법|행정법|헌법|노동법|가족법|민사소송법|형사소송법|상사법|공법|사법|계약법|채권법|물권법|가사소송법|가사법|호적법|부동산등기법|임대차보호법|집행법|강제집행법)')
            law_match = law_pattern.search(original_query)
            
            if law_match and article_match:
                law_name = law_match.group()
                article_no = article_match.group().replace(' ', '')
                fallback_query = f"{law_name} OR {article_no}"
                self.logger.info(f"Fallback 1: Searching with law name + article: '{fallback_query}'")
                
                with self._db_adapter.get_connection_context() as conn:
                    cursor = conn.cursor()
                    # text_search_vector 컬럼 존재 여부 확인
                    has_text_search_vector = self._check_column_exists('statutes_articles', 'text_search_vector')
                    
                    # PostgreSQL tsvector 쿼리
                    if has_text_search_vector:
                        where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                            fallback_query, table_alias='sa', text_vector_column='text_search_vector',
                            text_content_column=None, table_name='statutes_articles'
                        )
                    else:
                        where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                            fallback_query, table_alias='sa', text_vector_column=None,
                            text_content_column='article_content', table_name='statutes_articles'
                        )
                    
                    if not tsquery:
                        self.logger.warning(f"Empty tsquery generated from fallback query: '{fallback_query}'")
                        return []
                    
                    # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                    sql_query = f"""
                        SELECT
                            sa.id,
                            sa.statute_id,
                            sa.article_no,
                            sa.clause_no,
                            sa.item_no,
                            sa.article_title as heading,
                            sa.article_content as text,
                            s.law_name_kr as statute_name,
                            s.law_abbrv as statute_abbrv,
                            s.law_type as statute_type,
                            s.domain as category,
                            {rank_score_expr} as rank_score
                        FROM statutes_articles sa
                        JOIN statutes s ON sa.statute_id = s.id
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        LIMIT %s
                    """
                    query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                    cursor.execute(sql_query, query_params)
                    
                    for row in cursor.fetchall():
                        if row['id'] not in seen_ids:
                            seen_ids.add(row['id'])
                            text_content = row['text'] if row['text'] else ""
                            # PostgreSQL ts_rank_cd 점수 정규화
                            rank_score = row.get('rank_score', 0.0)
                            relevance_score = self._normalize_relevance_score(
                                rank_score,
                                log_context="fallback_strategy=law_article_only"
                            )
                            
                            result = self._build_statute_article_result(
                                row, text_content, relevance_score,
                                search_type="keyword", fallback_strategy="law_article_only"
                            )
                            if result:
                                fallback_results.append(result)
                
                if fallback_results:
                    self.logger.info(f"Fallback 1 found {len(fallback_results)} results")
                    return fallback_results[:limit]
            
            # 전략 2: 핵심 키워드만으로 검색 (법령명 또는 주요 키워드)
            if not fallback_results and words:
                # 법령명이 있으면 법령명으로, 없으면 첫 번째 핵심 키워드로
                if law_match:
                    keyword_query = law_match.group()
                else:
                    # 핵심 키워드 추출 (2자 이상, 불용어 제외 - KoreanStopwordProcessor 사용)
                    keywords = []
                    for w in words[:3]:
                        if len(w) >= 2:
                            if not self.stopword_processor or not self.stopword_processor.is_stopword(w):
                                keywords.append(w)
                    if keywords:
                        keyword_query = keywords[0]  # 첫 번째 핵심 키워드만
                    else:
                        return []
                
                self.logger.info(f"Fallback 2: Searching with keyword only: '{keyword_query}'")
                
                with self._db_adapter.get_connection_context() as conn:
                    cursor = conn.cursor()
                    # text_search_vector 컬럼 존재 여부 확인
                    has_text_search_vector = self._check_column_exists('statutes_articles', 'text_search_vector')
                    
                    # PostgreSQL tsvector 쿼리
                    if has_text_search_vector:
                        where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                            keyword_query, table_alias='sa', text_vector_column='text_search_vector',
                            text_content_column=None, table_name='statutes_articles'
                        )
                    else:
                        where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                            keyword_query, table_alias='sa', text_vector_column=None,
                            text_content_column='article_content', table_name='statutes_articles'
                        )
                    
                    if not tsquery:
                        self.logger.warning(f"Empty tsquery generated from keyword query: '{keyword_query}'")
                        return []
                    
                    # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                    sql_query = f"""
                        SELECT
                            sa.id,
                            sa.statute_id,
                            sa.article_no,
                            sa.clause_no,
                            sa.item_no,
                            sa.article_title as heading,
                            sa.article_content as text,
                            s.law_name_kr as statute_name,
                            s.law_abbrv as statute_abbrv,
                            s.law_type as statute_type,
                            s.domain as category,
                            {rank_score_expr} as rank_score
                        FROM statutes_articles sa
                        JOIN statutes s ON sa.statute_id = s.id
                        WHERE {where_clause}
                        ORDER BY {order_clause}
                        LIMIT %s
                    """
                    query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                    cursor.execute(sql_query, query_params)
                    
                    for row in cursor.fetchall():
                        if row['id'] not in seen_ids:
                            seen_ids.add(row['id'])
                            text_content = row['text'] if row['text'] else ""
                            # PostgreSQL ts_rank_cd 점수 정규화
                            rank_score = row.get('rank_score', 0.0)
                            relevance_score = self._normalize_relevance_score(
                                rank_score,
                                log_context="fallback_strategy=keyword_only"
                            )
                            
                            result = self._build_statute_article_result(
                                row, text_content, relevance_score,
                                search_type="keyword", fallback_strategy="keyword_only"
                            )
                            if result:
                                fallback_results.append(result)
                
                if fallback_results:
                    self.logger.info(f"Fallback 2 found {len(fallback_results)} results")
                    return fallback_results[:limit]
            
            # 전략 3: 조문 번호만으로 검색 (예: "750조" 또는 "제750조") - 강화
            if not fallback_results and article_match:
                article_no_clean = article_match.group().replace(' ', '').replace('제', '').replace('조', '')
                # 숫자만 추출
                article_num_match = re.search(r'\d+', article_no_clean)
                if article_num_match:
                    article_num = article_num_match.group()
                    # 다양한 패턴으로 검색 시도 (FTS 쿼리에서는 article_no: 패턴 사용 불가)
                    fallback_query = f"{article_num}조 OR 제{article_num}조 OR {article_num}"
                    
                    self.logger.info(f"Fallback 3: Searching with article number only: '{fallback_query}'")
                    
                    with self._db_adapter.get_connection_context() as conn:
                        cursor = conn.cursor()
                        # text_search_vector 컬럼 존재 여부 확인
                        has_text_search_vector = self._check_column_exists('statutes_articles', 'text_search_vector')
                        
                        # PostgreSQL tsvector 쿼리
                        if has_text_search_vector:
                            where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                fallback_query, table_alias='sa', text_vector_column='text_search_vector',
                                text_content_column=None, table_name='statutes_articles'
                            )
                        else:
                            where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                fallback_query, table_alias='sa', text_vector_column=None,
                                text_content_column='article_content', table_name='statutes_articles'
                            )
                        
                        if not tsquery:
                            self.logger.warning(f"Empty tsquery generated from fallback query: '{fallback_query}'")
                            return []
                        
                        # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                        sql_query = f"""
                            SELECT
                                sa.id,
                                sa.statute_id,
                                sa.article_no,
                                sa.clause_no,
                                sa.item_no,
                                sa.article_title as heading,
                                sa.article_content as text,
                                s.law_name_kr as statute_name,
                                s.law_abbrv as statute_abbrv,
                                s.law_type as statute_type,
                                s.domain as category,
                                {rank_score_expr} as rank_score
                            FROM statutes_articles sa
                            JOIN statutes s ON sa.statute_id = s.id
                            WHERE {where_clause}
                            ORDER BY {order_clause}
                            LIMIT %s
                        """
                        query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                        cursor.execute(sql_query, query_params)
                        
                        for row in cursor.fetchall():
                            if row['id'] not in seen_ids:
                                seen_ids.add(row['id'])
                                text_content = row['text'] if row['text'] else ""
                                # PostgreSQL ts_rank_cd 점수 정규화
                                rank_score = row.get('rank_score', 0.0)
                                relevance_score = self._normalize_relevance_score(
                                    rank_score,
                                    log_context="fallback_strategy=article_number_only"
                                )
                                
                                result = self._build_statute_article_result(
                                    row, text_content, relevance_score,
                                    search_type="keyword", fallback_strategy="article_number_only"
                                )
                                if result:
                                    fallback_results.append(result)
                    
                    if fallback_results:
                        self.logger.info(f"Fallback 3 found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            if not fallback_results:
                self.logger.warning(f"All fallback strategies failed for query: '{original_query}'")
                # 최종 폴백: 단일 키워드로 검색 시도
                if words:
                    try:
                        first_word = words[0]
                        if len(first_word) >= 2:  # 최소 2자 이상
                            self.logger.info(f"Trying final fallback: single keyword '{first_word}'")
                            with self._db_adapter.get_connection_context() as conn:
                                cursor = conn.cursor()
                                
                                # 단일 키워드로 간단한 검색 (PostgreSQL tsvector)
                            simple_query = f"{first_word}"
                            where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                simple_query,
                                table_alias='sa',
                                text_content_column='article_content'
                            )
                            
                            # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                            query_sql = f"""
                                SELECT sa.id, 
                                       {rank_score_expr} as rank_score
                                FROM statutes_articles sa
                                WHERE {where_clause}
                                ORDER BY {order_clause}
                                LIMIT %s
                            """
                            cursor.execute(query_sql, (tsquery, tsquery, tsquery, limit))
                            
                            rows = cursor.fetchall()
                            if rows:
                                self.logger.info(f"✅ Final fallback found {len(rows)} results")
                                # 결과를 직접 처리
                                for row in rows:
                                    doc_id = row[0] if isinstance(row, (tuple, list)) else row['id']
                                    if doc_id not in seen_ids:
                                        # statutes_articles에서 직접 조회
                                        cursor2 = conn.cursor()
                                        cursor2.execute("""
                                            SELECT sa.id, sa.statute_id, sa.article_no, sa.article_content as text,
                                                   s.law_name_kr as statute_name
                                            FROM statutes_articles sa
                                            JOIN statutes s ON sa.statute_id = s.id
                                            WHERE sa.id = %s
                                        """, (doc_id,))
                                        doc_row = cursor2.fetchone()
                                        cursor2.close()
                                        
                                        if doc_row:
                                            doc = {
                                                "id": f"statute_article_{doc_row['id']}",
                                                "type": "statute_article",
                                                "content": doc_row['text'],
                                                "text": doc_row['text'],
                                                "source": doc_row['statute_name'],
                                                "statute_name": doc_row['statute_name'],
                                                "article_no": doc_row['article_no'],
                                                "relevance_score": 0.2,
                                                "search_type": "keyword",
                                                "fallback_strategy": "final_fallback"
                                            }
                                            fallback_results.append(doc)
                                            seen_ids.add(doc_id)
                            
                            cursor.close()
                            conn.close()
                    except Exception as e:
                        self.logger.debug(f"Final fallback failed: {e}")
            
            return fallback_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in fallback statute search: {e}", exc_info=True)
            return []
    
    def _fallback_case_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """폴백 검색 전략: 판례 검색"""
        import re
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 핵심 키워드만으로 검색 (개선: 더 공격적인 키워드 추출 - KoreanStopwordProcessor 사용)
            if words:
                # 더 많은 키워드 추출 (3개 -> 5개)
                keywords = []
                for w in words[:5]:
                    if len(w) >= 1 and (not self.stopword_processor or not self.stopword_processor.is_stopword(w)):
                        keywords.append(w)
                if not keywords:
                    # 키워드가 없으면 모든 단어 사용 (불용어만 제거)
                    keywords = [w for w in words if len(w) >= 1 and (not self.stopword_processor or not self.stopword_processor.is_stopword(w))]
                
                if keywords:
                    # OR 조건으로 연결 (검색 범위 확장)
                    keyword_query = " OR ".join(keywords[:5])  # 최대 5개
                    self.logger.info(f"Fallback case search: Using keywords: '{keyword_query}'")
                    
                    with self._db_adapter.get_connection_context() as conn:
                        cursor = conn.cursor()
                        # text_search_vector 컬럼 존재 여부 확인
                        has_text_search_vector = self._check_column_exists('precedent_contents', 'text_search_vector')
                        
                        # PostgreSQL tsvector 쿼리
                        if has_text_search_vector:
                            where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                keyword_query, table_alias='pc', text_vector_column='text_search_vector',
                                text_content_column=None, table_name='precedent_contents'
                            )
                        else:
                            where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                keyword_query, table_alias='pc', text_vector_column=None,
                                text_content_column='section_content', table_name='precedent_contents'
                            )
                        
                        if not tsquery:
                            self.logger.warning(f"Empty tsquery generated from keyword query: '{keyword_query}'")
                            return []
                        
                        # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                        sql_query = f"""
                            SELECT
                                pc.id,
                                pc.precedent_id,
                                pc.section_content as text,
                                p.case_number as doc_id,
                                p.court_name as court,
                                p.case_type_name as case_type,
                                p.case_name as casenames,
                                p.decision_date as announce_date,
                                {rank_score_expr} as rank_score
                            FROM precedent_contents pc
                            JOIN precedents p ON pc.precedent_id = p.id
                            WHERE {where_clause}
                            ORDER BY {order_clause}
                            LIMIT %s
                        """
                        query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                        cursor.execute(sql_query, query_params)
                        
                        for row in cursor.fetchall():
                                if row['id'] not in seen_ids:
                                    seen_ids.add(row['id'])
                                text_content = row['text'] if row['text'] else ""
                                # PostgreSQL ts_rank_cd 점수 정규화 (통일된 로직 사용)
                                rank_score = row.get('rank_score', 0.0)
                                relevance_score = self._normalize_relevance_score(
                                    rank_score,
                                    log_context="fallback_strategy=keyword_only"
                                )
                                
                                fallback_results.append({
                                    "id": f"case_para_{row['id']}",
                                    "type": "precedent_content",
                                    "content": text_content,
                                    "text": text_content,
                                    "source": f"{row['court']} {row['doc_id']}",
                                    "metadata": {
                                        "case_id": row.get('case_id'),
                                        "doc_id": row.get('doc_id'),
                                        "court": row.get('court'),
                                        "case_type": row.get('case_type'),
                                        "casenames": row.get('casenames'),
                                        "announce_date": row.get('announce_date'),
                                        "para_index": row.get('para_index'),
                                        "type": "precedent_content",
                                    },
                                    "relevance_score": relevance_score,
                                    "search_type": "keyword",
                                    "fallback_strategy": "keyword_only"
                                })
                    
                    if fallback_results:
                        self.logger.info(f"Fallback case search found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            # 전략 2: 원본 쿼리에서 직접 키워드 추출 (전략 1 실패 시)
            if not fallback_results and original_query:
                import re
                # 원본 쿼리에서 모든 한글/영문 단어 추출
                all_words = re.findall(r'[가-힣a-zA-Z]+', original_query)
                if all_words:
                    # 불용어 제거 (KoreanStopwordProcessor 사용)
                    keywords = [w for w in all_words if len(w) >= 1 and (not self.stopword_processor or not self.stopword_processor.is_stopword(w))]
                    if keywords:
                        keyword_query = " OR ".join(keywords[:3])  # 최대 3개
                        self.logger.info(f"Fallback case search (strategy 2): Using keywords from original query: '{keyword_query}'")
                        
                        try:
                            with self._db_adapter.get_connection_context() as conn:
                                cursor = conn.cursor()
                                # text_search_vector 컬럼 존재 여부 확인
                                has_text_search_vector = self._check_column_exists('precedent_contents', 'text_search_vector')
                                
                                # ⚠️ 사용 중단: case_paragraphs 테이블은 더 이상 사용하지 않음
                                # Open Law 스키마의 precedent_contents 테이블 사용
                                # FTS5 대신 PostgreSQL FTS 사용
                                if has_text_search_vector:
                                    where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                        keyword_query, table_alias='pc', text_vector_column='text_search_vector',
                                        text_content_column=None, table_name='precedent_contents'
                                    )
                                else:
                                    where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                                        keyword_query, table_alias='pc', text_vector_column=None,
                                        text_content_column='section_content', table_name='precedent_contents'
                                    )
                                
                                if not tsquery:
                                    self.logger.warning(f"Empty tsquery generated from keyword query: '{keyword_query}'")
                                    return []
                                
                                # PostgreSQL tsvector 쿼리 (PGroonga 지원)
                                sql_query = f"""
                                    SELECT
                                        pc.id,
                                        pc.precedent_id,
                                        pc.section_type,
                                        pc.section_content as text,
                                        p.precedent_id as doc_id,
                                        p.court_name as court,
                                        p.case_type_name as case_type,
                                        p.case_name as casenames,
                                        p.decision_date as announce_date,
                                        {rank_score_expr} as rank_score
                                    FROM precedent_contents pc
                                    JOIN precedents p ON pc.precedent_id = p.id
                                    WHERE {where_clause}
                                    ORDER BY {order_clause}
                                    LIMIT %s
                                """
                                query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                                cursor.execute(sql_query, query_params)
                                
                                for row in cursor.fetchall():
                                    if row['id'] not in seen_ids:
                                        seen_ids.add(row['id'])
                                        text_content = row['text'] if row['text'] else ""
                                        # PostgreSQL ts_rank_cd 점수 정규화 (통일된 로직 사용)
                                        rank_score = row.get('rank_score', 0.0)
                                        relevance_score = self._normalize_relevance_score(
                                            rank_score,
                                            log_context="fallback_strategy=keyword_from_original"
                                        )
                                        
                                        fallback_results.append({
                                            "id": f"case_para_{row['id']}",
                                            "type": "precedent_content",
                                            "content": text_content,
                                            "text": text_content,
                                            "source": f"{row['court']} {row['doc_id']}",
                                            "metadata": {
                                                "case_id": row.get('case_id'),
                                                "doc_id": row.get('doc_id'),
                                                "court": row.get('court'),
                                                "case_type": row.get('case_type'),
                                                "casenames": row.get('casenames'),
                                                "announce_date": row.get('announce_date'),
                                                "para_index": row.get('para_index'),
                                                "type": "precedent_content",
                                            },
                                            "relevance_score": relevance_score,
                                            "search_type": "keyword",
                                            "fallback_strategy": "original_query_keywords"
                                        })
                            
                            if fallback_results:
                                self.logger.info(f"Fallback case search (strategy 2) found {len(fallback_results)} results")
                                return fallback_results[:limit]
                        except Exception as e2:
                            self.logger.warning(f"Fallback case search (strategy 2) failed: {e2}")
            
            return fallback_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in fallback case search: {e}", exc_info=True)
            return []
    
    def _fallback_decision_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        폴백 검색 전략: 심결례 검색
        
        ⚠️ 사용 중단: decision_paragraphs 테이블은 더 이상 사용하지 않음
        이 메서드는 현재 비활성화되어 있으며, 향후 제거 예정입니다.
        """
        # ⚠️ 사용 중단 테이블(decision_paragraphs) 참조로 인해 비활성화
        self.logger.warning("_fallback_decision_search is disabled: decision_paragraphs table is deprecated")
        return []
    
    def _fallback_interpretation_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        폴백 검색 전략: 유권해석 검색
        
        ⚠️ 사용 중단: interpretation_paragraphs 테이블은 더 이상 사용하지 않음
        이 메서드는 현재 비활성화되어 있으며, 향후 제거 예정입니다.
        """
        # ⚠️ 사용 중단 테이블(interpretation_paragraphs) 참조로 인해 비활성화
        self.logger.warning("_fallback_interpretation_search is disabled: interpretation_paragraphs table is deprecated")
        return []

    def search_cases_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """PostgreSQL tsvector 키워드 검색: 판례"""
        try:
            # tsquery 최적화 및 안전화
            optimized_query = self._optimize_tsquery(query)
            safe_query = self._sanitize_tsquery(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid tsquery: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            # ⚠️ 사용 중단: case_paragraphs → precedent_contents로 변경
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "precedent_contents")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            with self._db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # PostgreSQL tsvector 쿼리
                # 주의: precedent_contents (open_law 스키마)는 text_search_vector 컬럼이 없어서 text_content_column 사용
                where_clause, order_clause, rank_score_expr, tsquery = self._convert_fts5_to_postgresql_fts(
                    safe_query, table_alias='pc', text_vector_column=None, text_content_column='section_content'
                )
                
                if not tsquery:
                    self.logger.warning(f"Empty tsquery generated from query: '{safe_query}'")
                    return []
                
                # PostgreSQL tsvector 쿼리
                # precedent_contents는 text_search_vector가 없으므로 to_tsvector 사용
                # rank_score_expr 사용 (PGroonga 또는 PostgreSQL 기본)
                sql_query = f"""
                    SELECT
                        pc.id,
                        pc.precedent_id,
                        pc.section_content as text,
                        p.case_number as doc_id,
                        p.court_name as court,
                        p.case_type_name as case_type,
                        p.case_name as casenames,
                        p.decision_date as announce_date,
                        {rank_score_expr} as rank_score
                    FROM precedent_contents pc
                    JOIN precedents p ON pc.precedent_id = p.id
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT %s
                """
                # 플레이스홀더 개수 자동 결정
                query_params = self._get_query_params(tsquery, limit, rank_score_expr=rank_score_expr)
                cursor.execute(sql_query, query_params)

                # 법령명과 조문번호 추출 (점수 가중치 계산용)
                import re
                law_pattern = re.compile(r'([가-힣]+법)')
                article_pattern = re.compile(r'제\s*(\d+)\s*조')
                law_match = law_pattern.search(query)
                article_match = article_pattern.search(query)
                query_law_name = law_match.group(1) if law_match else None
                query_article_no = article_match.group(1) if article_match else None
                
                results = []
                for row in cursor.fetchall():
                    text_content = row['text'] if row['text'] else ""
                    # PostgreSQL ts_rank_cd 점수 정규화
                    rank_score = row.get('rank_score', 0.0)
                    relevance_score = self._normalize_relevance_score(
                        rank_score,
                        log_context=f"query='{query[:50]}', type=precedent_content_tsvector"
                    )
                    
                    # 개선: 법령명과 조문번호가 일치하는 경우 점수 가중치 부여
                    if query_law_name and query_article_no:
                        # 문서 내용에서 법령명과 조문번호가 모두 일치하는지 확인
                        law_in_text = query_law_name in text_content
                        article_in_text = f"제{query_article_no}조" in text_content or f"제 {query_article_no} 조" in text_content
                        
                        if law_in_text and article_in_text:
                            # 법령명과 조문번호가 모두 일치하면 강력한 점수 가중치 부여
                            # 가중치를 3.0배로 증가하고, 최소 점수 0.7 보장
                            boosted_score = relevance_score * 3.0
                            relevance_score = max(0.7, min(1.0, boosted_score))
                            self.logger.debug(f"Score boosted for law+article match: {query_law_name} 제{query_article_no}조 (original: {relevance_score / 3.0:.4f} -> boosted: {relevance_score:.4f})")
                    
                    result = self._build_precedent_result(
                        row, text_content, relevance_score,
                        search_type="keyword"
                    )
                    if result:
                        results.append(result)
            
            self.logger.info(f"FTS search found {len(results)} case paragraphs for query: {query}")
            
            # 결과가 없으면 폴백 검색 시도
            if len(results) == 0:
                self.logger.warning(f"No FTS results for query: '{query}' -> safe_query: '{safe_query}'. Trying fallback strategies...")
                import re
                words = safe_query.split() if safe_query else []
                results = self._fallback_case_search(query, safe_query, words, limit)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in FTS case search: {e}", exc_info=True)
            # 에러 발생 시에도 폴백 시도
            try:
                import re
                words = safe_query.split() if safe_query else []
                return self._fallback_case_search(query, safe_query, words, limit)
            except Exception:
                return []

    def search_decisions_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        PostgreSQL tsvector 키워드 검색: 심결례
        
        ⚠️ 사용 중단: decision_paragraphs 테이블은 더 이상 사용하지 않음
        이 메서드는 현재 비활성화되어 있으며, 향후 제거 예정입니다.
        """
        # ⚠️ 사용 중단 테이블(decision_paragraphs) 참조로 인해 비활성화
        self.logger.warning("search_decisions_fts is disabled: decision_paragraphs table is deprecated")
        return []

    def search_interpretations_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        PostgreSQL tsvector 키워드 검색: 유권해석
        
        ⚠️ 사용 중단: interpretation_paragraphs 테이블은 더 이상 사용하지 않음
        이 메서드는 현재 비활성화되어 있으며, 향후 제거 예정입니다.
        """
        # ⚠️ 사용 중단 테이블(interpretation_paragraphs) 참조로 인해 비활성화
        self.logger.warning("search_interpretations_fts is disabled: interpretation_paragraphs table is deprecated")
        return []

    def search_documents(self, query: str, category: Optional[str] = None, limit: int = 10, force_fts: bool = False) -> List[Dict[str, Any]]:
        """
        통합 검색: 라우팅에 따라 tsvector 또는 벡터 검색 (병렬 처리)

        Args:
            query: 검색 쿼리
            category: 카테고리 (하위 호환성을 위해 유지, 현재 사용하지 않음)
            limit: 최대 결과 수
            force_fts: True이면 라우팅과 관계없이 강제로 tsvector 검색 수행

        Returns:
            검색 결과 리스트
        """
        # force_fts가 True이면 라우팅 무시하고 강제로 tsvector 검색
        if force_fts:
            return self._search_documents_parallel(query, limit=limit)
        
        # 기존 라우팅 로직
        route = route_query(query)
        
        if route == "text2sql":
            # tsvector 키워드 검색 (병렬 처리)
            return self._search_documents_parallel(query, limit=limit)
        else:
            # 벡터 검색은 별도 SemanticSearchEngineV2에서 처리
            # 여기서는 빈 리스트 반환하거나 기본 FTS 결과 반환
            self.logger.info(f"Vector search requested for: {query}, delegating to SemanticSearchEngineV2")
            return []
    
    def _search_documents_parallel(self, query: str, limit: int = 10, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """
        병렬 tsvector 검색 실행
        
        Args:
            query: 검색 쿼리
            limit: 각 테이블당 최대 결과 수
            timeout: 각 검색 작업의 타임아웃 (초)
            
        Returns:
            검색 결과 리스트 (relevance_score 기준 정렬)
        """
        start_time = time.time()
        results = []
        
        # 법령명과 조문번호가 있는 경우 직접 검색 먼저 시도
        direct_statute_results = self.search_statute_article_direct(query, limit=limit)
        if direct_statute_results:
            results.extend(direct_statute_results)
            self.logger.info(f"✅ [DIRECT STATUTE] {len(direct_statute_results)}개 조문 직접 검색 성공")
        
        # 병렬 검색 작업 정의
        # 주의: 해석례(interpretation)와 결정례(decision)는 현재 데이터베이스에 없음
        # 추후 헌법결정례, 법령해석례가 추가될 수 있으나 현재는 제외
        search_tasks = {
            'statute': (self.search_statutes_fts, query, limit),
            'case': (self.search_cases_fts, query, limit),
            # 'decision': (self.search_decisions_fts, query, limit),  # 제거: 현재 데이터베이스에 없음
            # 'interpretation': (self.search_interpretations_fts, query, limit)  # 제거: 현재 데이터베이스에 없음
        }
        
        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="FTS_Search") as executor:
            # 모든 검색 작업 제출
            future_to_table = {}
            for table_type, (search_func, search_query, search_limit) in search_tasks.items():
                future = executor.submit(search_func, search_query, search_limit)
                future_to_table[future] = table_type
            
            # 완료된 작업 처리 (타임아웃 포함)
            for future in as_completed(future_to_table, timeout=timeout):
                table_type = future_to_table[future]
                try:
                    table_results = future.result(timeout=timeout)
                    if table_results:
                        results.extend(table_results)
                        self.logger.debug(
                            f"Parallel search completed for {table_type}: "
                            f"{len(table_results)} results"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Error in parallel {table_type} search for query '{query[:50]}...': {e}",
                        exc_info=self.logger.isEnabledFor(logging.DEBUG)
                    )
                    # 에러가 발생해도 다른 검색 결과는 계속 수집
        
        # 중복 제거 (direct_match가 True인 결과 우선)
        seen_ids = set()
        unique_results = []
        direct_results = []
        other_results = []
        
        for doc in results:
            doc_id = doc.get("id") or doc.get("doc_id")
            if doc_id and doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            # direct_match가 True인 결과는 별도로 분리
            if doc.get("direct_match", False):
                direct_results.append(doc)
            else:
                other_results.append(doc)
        
        # direct_match 결과를 먼저 추가
        unique_results.extend(direct_results)
        unique_results.extend(other_results)
        results = unique_results
        
        # relevance_score 기준 정렬 (statute_article 타입 우선)
        def sort_key(x):
            doc_type = x.get("type") or x.get("source_type", "")
            score = x.get("relevance_score", 0.0)
            # direct_match는 최우선
            if x.get("direct_match", False):
                return score + 2.0
            # statute_article 타입은 우선순위 부여 (점수에 1.0 추가)
            if doc_type == "statute_article":
                return score + 1.0
            return score
        
        results.sort(key=sort_key, reverse=True)
        
        # 개선 3: 타입별 다양성 확보 (점수 필터링 전에 적용)
        # 타입별 최소 보장을 위해 먼저 타입별로 분류
        type_counts = {}
        type_docs = {}
        for doc in results:
            doc_type = doc.get("type") or doc.get("source_type", "unknown")
            if doc_type not in type_docs:
                type_docs[doc_type] = []
            type_docs[doc_type].append(doc)
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # 타입별 최소 보장 (statute_article: 1개, precedent_content: 2개, decision: 1개, interpretation: 1개)
        min_counts = {
            "statute_article": 1,
            "precedent_content": 2,
            "case": 2,  # 레거시 호환
            "decision": 1,
            "interpretation": 1
        }
        
        # 타입별 최소 보장을 위한 결과 (점수와 무관하게 포함)
        diverse_results = []
        seen_diverse_ids = set()
        
        # 1단계: 각 타입별 최소 개수 보장 (점수와 무관하게)
        for doc_type, min_count in min_counts.items():
            if doc_type in type_docs:
                for doc in type_docs[doc_type][:min_count]:
                    doc_id = doc.get("id") or doc.get("doc_id")
                    if doc_id and doc_id not in seen_diverse_ids:
                        diverse_results.append(doc)
                        seen_diverse_ids.add(doc_id)
        
        # 점수 필터링: 0.7 이하는 제외 (단, direct_match와 법령명+조문번호 일치, 타입별 최소 보장은 예외)
        min_score_threshold = 0.7
        original_count = len(results)
        filtered_results = []
        
        # 법령명과 조문번호 추출 (예외 처리용)
        import re
        law_pattern = re.compile(r'([가-힣]+법)')
        article_pattern = re.compile(r'제\s*(\d+)\s*조')
        law_match = law_pattern.search(query)
        article_match = article_pattern.search(query)
        query_law_name = law_match.group(1) if law_match else None
        query_article_no = article_match.group(1) if article_match else None
        
        for doc in results:
            doc_id = doc.get("id") or doc.get("doc_id")
            # 이미 타입별 최소 보장으로 포함된 것은 제외
            if doc_id and doc_id in seen_diverse_ids:
                continue
                
            score = doc.get("relevance_score", 0.0)
            # direct_match는 점수와 무관하게 포함
            if doc.get("direct_match", False):
                filtered_results.append(doc)
            # 법령명과 조문번호가 일치하는 경우 예외 처리
            elif query_law_name and query_article_no:
                text_content = doc.get("content") or doc.get("text") or ""
                law_in_text = query_law_name in text_content
                article_in_text = f"제{query_article_no}조" in text_content or f"제 {query_article_no} 조" in text_content
                if law_in_text and article_in_text:
                    # 법령명과 조문번호가 모두 일치하면 점수와 무관하게 포함
                    filtered_results.append(doc)
                elif score > min_score_threshold:
                    filtered_results.append(doc)
            # 점수가 0.7 초과인 경우만 포함
            elif score > min_score_threshold:
                filtered_results.append(doc)
        
        # 타입별 최소 보장 결과 + 점수 필터링 결과 병합
        diverse_results.extend(filtered_results)
        results = diverse_results
        
        # 개선 4: FTS 검색 결과 포함 로직 개선
        # 직접 검색 실패 시 FTS 검색 결과 우선 포함
        if not direct_statute_results:
            # FTS 검색 결과 중 statute_article 타입이 있으면 우선 포함
            fts_statute_results = [doc for doc in results if doc.get("type") == "statute_article" and not doc.get("direct_match", False)]
            if fts_statute_results:
                # FTS 결과를 앞쪽에 배치 (점수는 낮지만 관련성 있음)
                for doc in fts_statute_results[:3]:  # 최대 3개
                    doc_id = doc.get("id") or doc.get("doc_id")
                    if doc_id and doc_id not in seen_diverse_ids:
                        diverse_results.insert(min(5, len(diverse_results)), doc)
                        seen_diverse_ids.add(doc_id)
                results = diverse_results
        
        final_count = len(results)
        
        # 성능 로깅
        elapsed_time = time.time() - start_time
        type_distribution = {k: v for k, v in type_counts.items()}
        self.logger.info(
            f"Parallel FTS search completed: {final_count} total results "
            f"(filtered from {original_count} results, min_score={min_score_threshold}) "
            f"from 2 tables (statute, case) in {elapsed_time:.3f}s for query: '{query[:50]}...' "
            f"(type distribution: {type_distribution})"
        )
        
        return results[:limit]

    @contextmanager
    def get_connection(self):
        """
        데이터베이스 연결 컨텍스트 매니저 (DatabaseManager 호환성)
        """
        conn = None
        conn_wrapper = None
        try:
            conn_wrapper = self._get_connection()
            conn = conn_wrapper.conn if hasattr(conn_wrapper, 'conn') else conn_wrapper
            yield conn_wrapper
        except Exception as e:
            if conn_wrapper:
                try:
                    conn_wrapper.rollback()
                except Exception:
                    pass
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            # 연결 풀에 반환 (PostgreSQL 연결 풀 사용 시)
            if self._db_adapter and self._db_adapter.connection_pool and conn_wrapper:
                try:
                    if hasattr(conn_wrapper, 'conn'):
                        # 연결이 닫혀있지 않은 경우에만 풀에 반환
                        if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                            self._db_adapter.connection_pool.putconn(conn_wrapper.conn)
                        else:
                            self.logger.debug("Connection already closed, not returning to pool")
                    else:
                        # conn 속성이 없으면 직접 반환 시도
                        if conn and hasattr(conn, 'closed') and conn.closed == 0:
                            self._db_adapter.connection_pool.putconn(conn)
                except Exception as put_error:
                    self.logger.warning(f"Error returning connection to pool: {put_error}")
                    # 연결이 손상된 경우 닫기 시도
                    try:
                        if conn and hasattr(conn, 'close'):
                            conn.close()
                    except Exception:
                        pass

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        쿼리 실행 (DatabaseManager 호환성)
        
        최적화: DatabaseAdapter의 get_connection_context를 사용하여 연결 재사용
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            쿼리 결과 리스트
        """
        if self._db_adapter:
            # DatabaseAdapter의 컨텍스트 매니저 사용 (연결 자동 반환)
            with self._db_adapter.get_connection_context() as conn_wrapper:
                cursor = conn_wrapper.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        else:
            # 하위 호환성: 기존 방식 사용
            conn_wrapper = None
            conn = None
            try:
                conn_wrapper = self._get_connection()
                conn = conn_wrapper.conn if hasattr(conn_wrapper, 'conn') else conn_wrapper
                cursor = conn_wrapper.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                self.logger.error(f"Error executing query: {e}")
                raise
            finally:
                # 연결 풀에 반환 (PostgreSQL 연결 풀 사용 시)
                if self._db_adapter and self._db_adapter.connection_pool and conn_wrapper:
                    try:
                        if hasattr(conn_wrapper, 'conn'):
                            # 연결이 닫혀있지 않은 경우에만 풀에 반환
                            if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                                self._db_adapter.connection_pool.putconn(conn_wrapper.conn)
                            else:
                                self.logger.debug("Connection already closed, not returning to pool")
                        else:
                            # conn 속성이 없으면 직접 반환 시도
                            if conn and hasattr(conn, 'closed') and conn.closed == 0:
                                self._db_adapter.connection_pool.putconn(conn)
                    except Exception as put_error:
                        self.logger.warning(f"Error returning connection to pool: {put_error}")
                        # 연결이 손상된 경우 닫기 시도
                        try:
                            if conn and hasattr(conn, 'close'):
                                conn.close()
                        except Exception:
                            pass

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        업데이트 쿼리 실행 (DatabaseManager 호환성)
        
        최적화: DatabaseAdapter의 get_connection_context를 사용하여 연결 재사용
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            영향받은 행 수
        """
        if self._db_adapter:
            # DatabaseAdapter의 컨텍스트 매니저 사용 (연결 자동 반환, 자동 커밋)
            with self._db_adapter.get_connection_context() as conn_wrapper:
                cursor = conn_wrapper.cursor()
                cursor.execute(query, params)
                # 컨텍스트 매니저가 자동으로 커밋하지만 명시적으로 커밋
                conn_wrapper.commit()
                return cursor.rowcount
        else:
            # 하위 호환성: 기존 방식 사용
            conn_wrapper = None
            conn = None
            try:
                conn_wrapper = self._get_connection()
                conn = conn_wrapper.conn if hasattr(conn_wrapper, 'conn') else conn_wrapper
                cursor = conn_wrapper.cursor()
                cursor.execute(query, params)
                conn_wrapper.commit()
                return cursor.rowcount
            except Exception as e:
                self.logger.error(f"Error executing update: {e}")
                if conn_wrapper:
                    try:
                        conn_wrapper.rollback()
                    except Exception:
                        pass
                raise
            finally:
                # 연결 풀에 반환 (PostgreSQL 연결 풀 사용 시)
                if self._db_adapter and self._db_adapter.connection_pool and conn_wrapper:
                    try:
                        if hasattr(conn_wrapper, 'conn'):
                            # 연결이 닫혀있지 않은 경우에만 풀에 반환
                            if hasattr(conn_wrapper, '_is_closed') and not conn_wrapper._is_closed():
                                self._db_adapter.connection_pool.putconn(conn_wrapper.conn)
                            else:
                                self.logger.debug("Connection already closed, not returning to pool")
                        else:
                            # conn 속성이 없으면 직접 반환 시도
                            if conn and hasattr(conn, 'closed') and conn.closed == 0:
                                self._db_adapter.connection_pool.putconn(conn)
                    except Exception as put_error:
                        self.logger.warning(f"Error returning connection to pool: {put_error}")
                        # 연결이 손상된 경우 닫기 시도
                        try:
                            if conn and hasattr(conn, 'close'):
                                conn.close()
                        except Exception:
                            pass

    def get_all_categories(self) -> List[str]:
        """도메인 목록 반환 (하위 호환성)"""
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT name FROM domains ORDER BY name")
                categories = [row[0] for row in cursor.fetchall()]
                return categories
            except Exception as e:
                self.logger.error(f"Error getting categories: {e}")
                return []
