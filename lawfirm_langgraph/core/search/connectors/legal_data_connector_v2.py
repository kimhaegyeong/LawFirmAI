# -*- coding: utf-8 -*-
"""
lawfirm_v2.db 전용 법률 데이터 연동 서비스
FTS5 키워드 검색 + 벡터 의미 검색 지원
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from core.data.connection_pool import get_connection_pool
except ImportError:
    try:
        from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
    except ImportError:
        get_connection_pool = None

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

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            import sys
            # source 모듈 경로 추가
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from core.utils.config import Config
            config = Config()
            db_path = config.database_path
        # 상대 경로를 절대 경로로 변환
        if db_path and not os.path.isabs(db_path):
            db_path = os.path.abspath(db_path)
        self.db_path = db_path
        self.logger = get_logger(__name__)
        # 연결 풀 초기화
        if get_connection_pool:
            self._connection_pool = get_connection_pool(self.db_path)
            self.logger.debug("Using connection pool for database connections")
        else:
            self._connection_pool = None
            self.logger.warning("Connection pool not available, using direct connections")
        # 데이터베이스 경로 로깅
        self.logger.info(f"LegalDataConnectorV2 initialized with database path: {self.db_path}")
        if not Path(self.db_path).exists():
            self.logger.warning(f"Database {self.db_path} not found. Please initialize it first.")
        else:
            self.logger.info(f"Database {self.db_path} exists and is ready.")
            # FTS 테이블 존재 여부 확인
            self._check_fts_tables()
        
        # KoreanStopwordProcessor 초기화 (KoNLPy 우선 사용)
        self.stopword_processor = None
        if KoreanStopwordProcessor:
            try:
                self.stopword_processor = KoreanStopwordProcessor()
                self.logger.debug("KoreanStopwordProcessor initialized successfully")
            except Exception as e:
                self.logger.warning(f"Error initializing KoreanStopwordProcessor: {e}")
        
        # KoNLPy 형태소 분석기 초기화 (선택적, 기존 호환성 유지)
        self._okt = None
        try:
            from konlpy.tag import Okt  # pyright: ignore[reportMissingImports]
            self._okt = Okt()
            self.logger.debug("KoNLPy Okt initialized successfully")
        except ImportError:
            self.logger.debug("KoNLPy not available, will use fallback method")
        except Exception as e:
            self.logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")

    def _check_fts_tables(self):
        """FTS 테이블 존재 여부 확인 및 초기화 필요 여부 안내"""
        conn = None
        missing_tables = []
        has_embeddings = False
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 필수 FTS 테이블 목록
            required_fts_tables = [
                'statute_articles_fts',
                'case_paragraphs_fts',
                'decision_paragraphs_fts',
                'interpretation_paragraphs_fts'
            ]
            
            for table_name in required_fts_tables:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                if not cursor.fetchone():
                    missing_tables.append(table_name)
            
            # embeddings 테이블 확인
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
            )
            has_embeddings = cursor.fetchone() is not None
        finally:
            # 연결 풀링 사용 시 close() 호출하지 않음
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass
        
        if missing_tables:
            self.logger.error(
                f"❌ Missing FTS tables: {', '.join(missing_tables)}. "
                f"Please run: python scripts/init_lawfirm_v2_db.py"
            )
        if not has_embeddings:
            self.logger.warning(
                "⚠️ embeddings table not found. "
                "Semantic search will not work until embeddings are generated."
            )
        
        if missing_tables or not has_embeddings:
            self.logger.error(
                f"Database initialization incomplete. "
                f"Required tables missing: FTS={len(missing_tables)}, embeddings={'missing' if not has_embeddings else 'exists'}"
            )
        else:
            self.logger.info("✅ All required FTS tables and embeddings table exist.")

    def _get_connection(self):
        """
        Get database connection (using connection pool if available)
        
        Note: Each thread gets its own connection for thread safety.
        SQLite connections are not thread-safe, so each parallel search
        operation uses a separate connection.
        Connection pool reuses connections per thread to improve performance.
        """
        if self._connection_pool:
            return self._connection_pool.get_connection()
        else:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            return conn

    def _analyze_query_plan(self, query: str, table_name: str) -> Optional[Dict[str, Any]]:
        """
        FTS5 쿼리 실행 계획 분석

        Args:
            query: 검색 쿼리
            table_name: FTS5 테이블명

        Returns:
            실행 계획 정보 딕셔너리 또는 None
        """
        conn = None
        try:
            safe_query = self._sanitize_fts5_query(query)
            if not safe_query:
                return None

            conn = self._get_connection()
            cursor = conn.cursor()

            # EXPLAIN QUERY PLAN 실행
            explain_query = f"""
                EXPLAIN QUERY PLAN
                SELECT rowid, bm25({table_name}) as rank_score
                FROM {table_name}
                WHERE {table_name} MATCH ?
                ORDER BY rank_score
                LIMIT 10
            """
            cursor.execute(explain_query, (safe_query,))
            plan_rows = cursor.fetchall()
            
            # 실행 계획 분석
            plan_info = {
                "uses_index": any("FTS" in str(row) or "MATCH" in str(row) for row in plan_rows),
                "scan_type": "FTS" if any("FTS" in str(row) for row in plan_rows) else "UNKNOWN",
                "plan_detail": [str(row) for row in plan_rows]
            }
            
            self.logger.debug(f"Query plan for '{query[:30]}': {plan_info}")
            return plan_info
        except Exception as e:
            self.logger.debug(f"Error analyzing query plan: {e}")
            return None
        finally:
            # 연결 풀링 사용 시 close() 호출하지 않음
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass

    def _optimize_fts5_query(self, query: str) -> str:
        """
        FTS5 쿼리 최적화 (형태소 분석 기반 조사/어미 제거)

        Args:
            query: 원본 쿼리

        Returns:
            최적화된 쿼리
        """
        # KoNLPy가 사용 가능하면 형태소 분석 사용, 아니면 폴백
        if self._okt is not None:
            try:
                return self._optimize_fts5_query_morphological(query)
            except Exception as e:
                self.logger.warning(f"Error in morphological analysis: {e}, using fallback")
                return self._optimize_fts5_query_fallback(query)
        else:
            return self._optimize_fts5_query_fallback(query)
    
    def _optimize_fts5_query_morphological(self, query: str) -> str:
        """
        형태소 분석 기반 FTS5 쿼리 최적화
        
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
            return self._optimize_fts5_query_fallback(query)
        
        optimized = " ".join(core_keywords)
        self.logger.debug(f"Query optimized (morphological): '{query}' -> '{optimized}'")
        
        return optimized
    
    def _optimize_fts5_query_fallback(self, query: str) -> str:
        """
        FTS5 쿼리 최적화 (폴백 방식: 기존 방식)
        
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

    def _sanitize_fts5_query(self, query: str) -> str:
        """
        FTS5 쿼리를 안전하게 변환
        - 특수 문자가 있으면 이스케이프 처리
        - 빈 쿼리는 빈 문자열 반환
        - 단순 키워드 검색에 최적화
        - FTS5는 기본적으로 공백으로 구분된 단어를 AND 조건으로 처리
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
        
        # 8단계: 최대 5개 단어/구문만 사용 (FTS5 성능 최적화)
        clean_words = clean_words[:5]
        
        # 9단계: FTS5 쿼리 생성 (개선: 법령명+조문번호 구문은 하나의 구문으로 처리)
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
        
        # 10단계: FTS5 쿼리 정제 (개선: "OR OR" 오류 방지)
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
        
        self.logger.debug(f"FTS5 query sanitized: '{query[:100]}' -> '{sanitized}'")
        return sanitized

    def search_statutes_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 법령 조문 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}' (optimized: '{optimized_query}')")
                return []
            
            # 검색 쿼리 로깅
            self.logger.info(f"FTS statute search: original='{query}', optimized='{optimized_query}', safe='{safe_query}'")
            
            # 쿼리 최적화: 불완전한 단어 제거 (예: "손해배상에" -> "손해배상")
            import re
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
                plan_info = self._analyze_query_plan(query, "statute_articles_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # FTS5 검색 (BM25 랭킹)
                cursor.execute("""
                    SELECT
                        sa.id,
                        sa.statute_id,
                        sa.article_no,
                        sa.clause_no,
                        sa.item_no,
                        sa.heading,
                        sa.text,
                        s.name as statute_name,
                        s.abbrv as statute_abbrv,
                        s.statute_type,
                        s.category,
                        bm25(statute_articles_fts) as rank_score
                    FROM statute_articles_fts
                    JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE statute_articles_fts MATCH ?
                    ORDER BY rank_score
                    LIMIT ?
                """, (safe_query, limit))

                results = []
                for row in cursor.fetchall():
                    # text 필드가 비어있으면 경고
                    text_content = row['text'] if row['text'] else ""
                    if not text_content:
                        self.logger.warning(f"Empty text content for statute article id={row['id']}, article_no={row['article_no']}")
                    
                    # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                    if row['rank_score']:
                        # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                        relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                    else:
                        # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                        relevance_score = 0.3  # 기본값을 낮춤 (0.5 -> 0.3)
                    
                    results.append({
                        "id": f"statute_article_{row['id']}",
                        "type": "statute_article",
                        "source_type": "statute_article",
                        "content": text_content,
                        "text": text_content,  # text 필드도 추가 (호환성)
                        "source": row['statute_name'],
                        "statute_name": row['statute_name'],
                        "article_no": row['article_no'],
                        "clause_no": row['clause_no'],
                        "item_no": row['item_no'],
                        "metadata": {
                            "statute_id": row['statute_id'],
                            "article_no": row['article_no'],
                            "clause_no": row['clause_no'],
                            "item_no": row['item_no'],
                            "heading": row['heading'],
                            "statute_abbrv": row['statute_abbrv'],
                            "statute_type": row['statute_type'],
                            "category": row['category'],
                            "source_type": "statute_article"
                        },
                        "relevance_score": relevance_score,
                        "search_type": "keyword"
                    })
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if not self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            
            self.logger.info(f"FTS search found {len(results)} statute articles for query: '{query}' (safe_query: '{safe_query}')")
            
            # 결과가 없으면 폴백 검색 시도
            if len(results) == 0:
                self.logger.warning(f"No FTS results for query: '{query}' -> safe_query: '{safe_query}'. Trying fallback strategies...")
                results = self._fallback_statute_search(query, safe_query, words, limit)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in FTS statute search for query '{query}': {e}", exc_info=True)
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
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # 개선 2: 법령명 매칭 로직 강화 (유사도 매칭)
                # 전략 1: 정확한 이름 매칭
                cursor.execute("""
                    SELECT id, name, abbrv 
                    FROM statutes 
                    WHERE name = ? OR abbrv = ?
                    LIMIT 1
                """, (law_name, law_name))
                
                statute_row = cursor.fetchone()
                
                # 전략 2: LIKE 검색 (정확한 매칭 실패 시)
                if not statute_row:
                    cursor.execute("""
                        SELECT id, name, abbrv 
                        FROM statutes 
                        WHERE name LIKE ? OR abbrv LIKE ? OR name LIKE ? OR abbrv LIKE ?
                        LIMIT 5
                    """, (
                        f"%{law_name}%", 
                        f"%{law_name}%", 
                        f"{law_name}%", 
                        f"{law_name}%"
                    ))
                    
                    candidates = cursor.fetchall()
                    if candidates:
                        # 가장 유사한 법령명 선택 (길이가 가장 가까운 것)
                        best_match = min(candidates, key=lambda x: abs(len(x['name']) - len(law_name)))
                        statute_row = best_match
                        self.logger.info(f"법령명 유사도 매칭 성공: '{law_name}' -> '{statute_row['name']}'")
                
                if not statute_row:
                    self.logger.warning(f"법령을 찾을 수 없음: {law_name}")
                    return []
                
                statute_id = statute_row['id']
                
                # 해당 조문 직접 조회 (SQLite는 NULLS LAST 미지원, CASE 문 사용)
                cursor.execute("""
                    SELECT 
                        sa.id,
                        sa.statute_id,
                        sa.article_no,
                        sa.clause_no,
                        sa.item_no,
                        sa.heading,
                        sa.text,
                        s.name as statute_name,
                        s.abbrv as statute_abbrv,
                        s.statute_type,
                        s.category
                    FROM statute_articles sa
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE sa.statute_id = ? AND sa.article_no = ?
                    ORDER BY 
                        CASE WHEN sa.clause_no IS NULL THEN 1 ELSE 0 END,
                        sa.clause_no,
                        CASE WHEN sa.item_no IS NULL THEN 1 ELSE 0 END,
                        sa.item_no
                    LIMIT ?
                """, (statute_id, article_no, limit * 2))
                
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
                            "source_type": "statute_article",
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
                                "source_type": "statute_article"
                            },
                            "relevance_score": 1.0,
                            "final_weighted_score": 1.0,
                            "search_type": "direct_statute",
                            "direct_match": True
                        })
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if not self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            
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
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        SELECT
                            sa.id,
                            sa.statute_id,
                            sa.article_no,
                            sa.clause_no,
                            sa.item_no,
                            sa.heading,
                            sa.text,
                            s.name as statute_name,
                            s.abbrv as statute_abbrv,
                            s.statute_type,
                            s.category,
                            bm25(statute_articles_fts) as rank_score
                        FROM statute_articles_fts
                        JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                        JOIN statutes s ON sa.statute_id = s.id
                        WHERE statute_articles_fts MATCH ?
                        ORDER BY rank_score
                        LIMIT ?
                    """, (fallback_query, limit))
                    
                    for row in cursor.fetchall():
                        if row['id'] not in seen_ids:
                            seen_ids.add(row['id'])
                            text_content = row['text'] if row['text'] else ""
                            # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                            if row['rank_score']:
                                # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                                relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                            else:
                                # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                                relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                            
                            fallback_results.append({
                                "id": f"statute_article_{row['id']}",
                                "type": "statute_article",
                                "source_type": "statute_article",
                                "content": text_content,
                                "text": text_content,
                                "source": row['statute_name'],
                                "statute_name": row['statute_name'],
                                "article_no": row['article_no'],
                                "clause_no": row['clause_no'],
                                "item_no": row['item_no'],
                                "metadata": {
                                    "statute_id": row['statute_id'],
                                    "article_no": row['article_no'],
                                    "clause_no": row['clause_no'],
                                    "item_no": row['item_no'],
                                    "heading": row['heading'],
                                    "statute_abbrv": row['statute_abbrv'],
                                    "statute_type": row['statute_type'],
                                    "category": row['category'],
                                    "source_type": "statute_article"
                                },
                                "relevance_score": relevance_score,
                                "search_type": "keyword",
                                "fallback_strategy": "law_article_only"
                            })
                finally:
                    # 연결 풀링 사용 시 close() 호출하지 않음
                    if not self._connection_pool:
                        try:
                            conn.close()
                        except Exception:
                            pass
                
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
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        SELECT
                            sa.id,
                            sa.statute_id,
                            sa.article_no,
                            sa.clause_no,
                            sa.item_no,
                            sa.heading,
                            sa.text,
                            s.name as statute_name,
                            s.abbrv as statute_abbrv,
                            s.statute_type,
                            s.category,
                            bm25(statute_articles_fts) as rank_score
                        FROM statute_articles_fts
                        JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                        JOIN statutes s ON sa.statute_id = s.id
                        WHERE statute_articles_fts MATCH ?
                        ORDER BY rank_score
                        LIMIT ?
                    """, (keyword_query, limit))
                    
                    for row in cursor.fetchall():
                        if row['id'] not in seen_ids:
                            seen_ids.add(row['id'])
                            text_content = row['text'] if row['text'] else ""
                            # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                            if row['rank_score']:
                                # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                                relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                            else:
                                # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                                relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                            
                            fallback_results.append({
                                "id": f"statute_article_{row['id']}",
                                "type": "statute_article",
                                "source_type": "statute_article",
                                "content": text_content,
                                "text": text_content,
                                "source": row['statute_name'],
                                "statute_name": row['statute_name'],
                                "article_no": row['article_no'],
                                "clause_no": row['clause_no'],
                                "item_no": row['item_no'],
                                "metadata": {
                                    "statute_id": row['statute_id'],
                                    "article_no": row['article_no'],
                                    "clause_no": row['clause_no'],
                                    "item_no": row['item_no'],
                                    "heading": row['heading'],
                                    "statute_abbrv": row['statute_abbrv'],
                                    "statute_type": row['statute_type'],
                                    "category": row['category'],
                                    "source_type": "statute_article"
                                },
                                "relevance_score": relevance_score,
                                "search_type": "keyword",
                                "fallback_strategy": "keyword_only"
                            })
                finally:
                    # 연결 풀링 사용 시 close() 호출하지 않음
                    if not self._connection_pool:
                        try:
                            conn.close()
                        except Exception:
                            pass
                
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
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT
                                sa.id,
                                sa.statute_id,
                                sa.article_no,
                                sa.clause_no,
                                sa.item_no,
                                sa.heading,
                                sa.text,
                                s.name as statute_name,
                                s.abbrv as statute_abbrv,
                                s.statute_type,
                                s.category,
                                bm25(statute_articles_fts) as rank_score
                            FROM statute_articles_fts
                            JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                            JOIN statutes s ON sa.statute_id = s.id
                            WHERE statute_articles_fts MATCH ?
                            ORDER BY rank_score
                            LIMIT ?
                        """, (fallback_query, limit))
                        
                        for row in cursor.fetchall():
                            if row['id'] not in seen_ids:
                                seen_ids.add(row['id'])
                                text_content = row['text'] if row['text'] else ""
                                # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                                if row['rank_score']:
                                    # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                                    relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                                else:
                                    # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                                    relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                                
                                fallback_results.append({
                                    "id": f"statute_article_{row['id']}",
                                    "type": "statute_article",
                                    "source_type": "statute_article",
                                    "content": text_content,
                                    "text": text_content,
                                    "source": row['statute_name'],
                                    "statute_name": row['statute_name'],
                                    "article_no": row['article_no'],
                                    "clause_no": row['clause_no'],
                                    "item_no": row['item_no'],
                                    "metadata": {
                                        "statute_id": row['statute_id'],
                                        "article_no": row['article_no'],
                                        "clause_no": row['clause_no'],
                                        "item_no": row['item_no'],
                                        "heading": row['heading'],
                                        "statute_abbrv": row['statute_abbrv'],
                                        "statute_type": row['statute_type'],
                                        "category": row['category'],
                                        "source_type": "statute_article"
                                    },
                                    "relevance_score": relevance_score,
                                    "search_type": "keyword",
                                    "fallback_strategy": "article_number_only"
                                })
                    finally:
                        # 연결 풀링 사용 시 close() 호출하지 않음
                        if not self._connection_pool:
                            try:
                                conn.close()
                            except Exception:
                                pass
                    
                    if fallback_results:
                        self.logger.info(f"Fallback 3 found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            if not fallback_results:
                self.logger.warning(f"All fallback strategies failed for query: '{original_query}'")
            
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
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT
                                cp.id,
                                cp.case_id,
                                cp.para_index,
                                cp.text,
                                c.doc_id,
                                c.court,
                                c.case_type,
                                c.casenames,
                                c.announce_date,
                                bm25(case_paragraphs_fts) as rank_score
                            FROM case_paragraphs_fts
                            JOIN case_paragraphs cp ON case_paragraphs_fts.rowid = cp.id
                            JOIN cases c ON cp.case_id = c.id
                            WHERE case_paragraphs_fts MATCH ?
                            ORDER BY rank_score
                            LIMIT ?
                        """, (keyword_query, limit))
                        
                        for row in cursor.fetchall():
                            if row['id'] not in seen_ids:
                                seen_ids.add(row['id'])
                                text_content = row['text'] if row['text'] else ""
                                # relevance_score 계산 개선
                                if row['rank_score']:
                                    relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                                else:
                                    relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                                
                                fallback_results.append({
                                    "id": f"case_para_{row['id']}",
                                    "type": "case",
                                    "content": text_content,
                                    "text": text_content,
                                    "source": f"{row['court']} {row['doc_id']}",
                                    "metadata": {
                                        "case_id": row['case_id'],
                                        "doc_id": row['doc_id'],
                                        "court": row['court'],
                                        "case_type": row['case_type'],
                                        "casenames": row['casenames'],
                                        "announce_date": row['announce_date'],
                                        "para_index": row['para_index'],
                                    },
                                    "relevance_score": relevance_score,
                                    "search_type": "keyword",
                                    "fallback_strategy": "keyword_only"
                                })
                    finally:
                        # 연결 풀링 사용 시 close() 호출하지 않음
                        if not self._connection_pool:
                            try:
                                conn.close()
                            except Exception:
                                pass
                    
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
                            conn = self._get_connection()
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    SELECT
                                        cp.id,
                                        cp.case_id,
                                        cp.para_index,
                                        cp.text,
                                        c.doc_id,
                                        c.court,
                                        c.case_type,
                                        c.casenames,
                                        c.announce_date,
                                        bm25(case_paragraphs_fts) as rank_score
                                    FROM case_paragraphs_fts
                                    JOIN case_paragraphs cp ON case_paragraphs_fts.rowid = cp.id
                                    JOIN cases c ON cp.case_id = c.id
                                    WHERE case_paragraphs_fts MATCH ?
                                    ORDER BY rank_score
                                    LIMIT ?
                                """, (keyword_query, limit))
                                
                                for row in cursor.fetchall():
                                    if row['id'] not in seen_ids:
                                        seen_ids.add(row['id'])
                                        text_content = row['text'] if row['text'] else ""
                                        if row['rank_score']:
                                            relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                                        else:
                                            relevance_score = 0.15  # 전략 2는 더 낮은 점수
                                        
                                        fallback_results.append({
                                            "id": f"case_para_{row['id']}",
                                            "type": "case",
                                            "content": text_content,
                                            "text": text_content,
                                            "source": f"{row['court']} {row['doc_id']}",
                                            "metadata": {
                                                "case_id": row['case_id'],
                                                "doc_id": row['doc_id'],
                                                "court": row['court'],
                                                "case_type": row['case_type'],
                                                "casenames": row['casenames'],
                                                "announce_date": row['announce_date'],
                                                "para_index": row['para_index'],
                                            },
                                            "relevance_score": relevance_score,
                                            "search_type": "keyword",
                                            "fallback_strategy": "original_query_keywords"
                                        })
                            finally:
                                # 연결 풀링 사용 시 close() 호출하지 않음
                                if not self._connection_pool:
                                    try:
                                        conn.close()
                                    except Exception:
                                        pass
                            
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
        """폴백 검색 전략: 심결례 검색"""
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 핵심 키워드만으로 검색 (KoreanStopwordProcessor 사용)
            if words:
                keywords = [w for w in words[:3] if len(w) >= 2 and (not self.stopword_processor or not self.stopword_processor.is_stopword(w))]
                if keywords:
                    keyword_query = " OR ".join(keywords)
                    self.logger.info(f"Fallback decision search: Using keywords: '{keyword_query}'")
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            SELECT
                                dp.id,
                                dp.decision_id,
                                dp.para_index,
                                dp.text,
                                d.org,
                                d.doc_id,
                                d.decision_date,
                                d.result,
                                bm25(decision_paragraphs_fts) as rank_score
                            FROM decision_paragraphs_fts
                            JOIN decision_paragraphs dp ON decision_paragraphs_fts.rowid = dp.id
                            JOIN decisions d ON dp.decision_id = d.id
                            WHERE decision_paragraphs_fts MATCH ?
                            ORDER BY rank_score
                            LIMIT ?
                        """, (keyword_query, limit))
                        
                        for row in cursor.fetchall():
                            if row['id'] not in seen_ids:
                                seen_ids.add(row['id'])
                                text_content = row['text'] if row['text'] else ""
                                # relevance_score 계산 개선
                                if row['rank_score']:
                                    relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                                else:
                                    relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                                
                                fallback_results.append({
                                    "id": f"decision_para_{row['id']}",
                                    "type": "decision",
                                    "content": text_content,
                                    "text": text_content,
                                    "source": f"{row['org']} {row['doc_id']}",
                                    "metadata": {
                                        "decision_id": row['decision_id'],
                                        "org": row['org'],
                                        "doc_id": row['doc_id'],
                                        "decision_date": row['decision_date'],
                                        "result": row['result'],
                                        "para_index": row['para_index'],
                                    },
                                    "relevance_score": relevance_score,
                                    "search_type": "keyword",
                                    "fallback_strategy": "keyword_only"
                                })
                    finally:
                        # 연결 풀링 사용 시 close() 호출하지 않음
                        if not self._connection_pool:
                            try:
                                conn.close()
                            except Exception:
                                pass
                    
                    if fallback_results:
                        self.logger.info(f"Fallback decision search found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            return fallback_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in fallback decision search: {e}", exc_info=True)
            return []
    
    def _fallback_interpretation_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """폴백 검색 전략: 유권해석 검색"""
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 핵심 키워드만으로 검색 (KoreanStopwordProcessor 사용)
            if words:
                keywords = [w.strip() for w in words[:3] if w.strip() and len(w.strip()) >= 2 and (not self.stopword_processor or not self.stopword_processor.is_stopword(w.strip()))]
                if keywords:
                    # 빈 문자열 제거 및 중복 제거
                    keywords = list(dict.fromkeys(keywords))  # 순서 유지하면서 중복 제거
                    keyword_query = " OR ".join(keywords)
                    # "OR OR" 오류 방지
                    keyword_query = keyword_query.replace(" OR OR ", " OR ").strip()
                    if keyword_query and not keyword_query.startswith("OR") and not keyword_query.endswith("OR") and keyword_query != "OR":
                        self.logger.info(f"Fallback interpretation search: Using keywords: '{keyword_query}'")
                        
                        conn = self._get_connection()
                        cursor = conn.cursor()
                        
                        try:
                            cursor.execute("""
                                SELECT
                                    ip.id,
                                    ip.interpretation_id,
                                    ip.para_index,
                                    ip.text,
                                    i.org,
                                    i.doc_id,
                                    i.title,
                                    i.response_date,
                                    bm25(interpretation_paragraphs_fts) as rank_score
                                FROM interpretation_paragraphs_fts
                                JOIN interpretation_paragraphs ip ON interpretation_paragraphs_fts.rowid = ip.id
                                JOIN interpretations i ON ip.interpretation_id = i.id
                                WHERE interpretation_paragraphs_fts MATCH ?
                                ORDER BY rank_score
                                LIMIT ?
                            """, (keyword_query, limit))
                        except Exception as e:
                            self.logger.warning(f"FTS query error in fallback interpretation search: {e}, query: '{keyword_query}'")
                            if not self._connection_pool:
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                            keywords = []
                    else:
                        self.logger.warning(f"Fallback interpretation search: Invalid keyword query: '{keyword_query}', skipping")
                        keywords = []
                    
                    if keywords:  # keywords가 있을 때만 fetchall 실행
                        try:
                            for row in cursor.fetchall():
                                if row['id'] not in seen_ids:
                                    seen_ids.add(row['id'])
                                    text_content = row['text'] if row['text'] else ""
                                    # relevance_score 계산 개선
                                    if row['rank_score']:
                                        relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                                    else:
                                        relevance_score = 0.2  # 폴백 전략은 더 낮은 점수
                                    
                                    fallback_results.append({
                                        "id": f"interpretation_para_{row['id']}",
                                        "type": "interpretation",
                                        "content": text_content,
                                        "text": text_content,
                                        "source": f"{row['org']} {row['title']}",
                                        "metadata": {
                                            "interpretation_id": row['interpretation_id'],
                                            "org": row['org'],
                                            "doc_id": row['doc_id'],
                                            "title": row['title'],
                                            "response_date": row['response_date'],
                                            "para_index": row['para_index'],
                                        },
                                        "relevance_score": relevance_score,
                                        "search_type": "keyword",
                                        "fallback_strategy": "keyword_only"
                                    })
                        finally:
                            # 연결 풀링 사용 시 close() 호출하지 않음
                            if not self._connection_pool:
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                        
                        if fallback_results:
                            self.logger.info(f"Fallback interpretation search found {len(fallback_results)} results")
                            return fallback_results[:limit]
            
            return fallback_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in fallback interpretation search: {e}", exc_info=True)
            return []

    def search_cases_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 판례 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "case_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # SQL injection 방지: 파라미터 바인딩 사용
                cursor.execute("""
                    SELECT
                        cp.id,
                        cp.case_id,
                        cp.para_index,
                        cp.text,
                        c.doc_id,
                        c.court,
                        c.case_type,
                        c.casenames,
                        c.announce_date,
                        bm25(case_paragraphs_fts) as rank_score
                    FROM case_paragraphs_fts
                    JOIN case_paragraphs cp ON case_paragraphs_fts.rowid = cp.id
                    JOIN cases c ON cp.case_id = c.id
                    WHERE case_paragraphs_fts MATCH ?
                    ORDER BY rank_score
                    LIMIT ?
                """, (safe_query, limit))

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
                    # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                    if row['rank_score']:
                        # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                        relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                    else:
                        # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                        relevance_score = 0.3  # 기본값을 낮춤 (0.5 -> 0.3)
                    
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
                    
                    results.append({
                        "id": f"case_para_{row['id']}",
                        "type": "case",
                        "content": text_content,
                        "text": text_content,
                        "source": f"{row['court']} {row['doc_id']}",
                        "metadata": {
                            "case_id": row['case_id'],
                            "doc_id": row['doc_id'],
                            "court": row['court'],
                            "case_type": row['case_type'],
                            "casenames": row['casenames'],
                            "announce_date": row['announce_date'],
                            "para_index": row['para_index'],
                        },
                        "relevance_score": relevance_score,
                        "search_type": "keyword"
                    })
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if not self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            
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
        """FTS5 키워드 검색: 심결례 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "decision_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # SQL injection 방지: 파라미터 바인딩 사용
                cursor.execute("""
                    SELECT
                        dp.id,
                        dp.decision_id,
                        dp.para_index,
                        dp.text,
                        d.org,
                        d.doc_id,
                        d.decision_date,
                        d.result,
                        bm25(decision_paragraphs_fts) as rank_score
                    FROM decision_paragraphs_fts
                    JOIN decision_paragraphs dp ON decision_paragraphs_fts.rowid = dp.id
                    JOIN decisions d ON dp.decision_id = d.id
                    WHERE decision_paragraphs_fts MATCH ?
                    ORDER BY rank_score
                    LIMIT ?
                """, (safe_query, limit))

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
                    # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                    if row['rank_score']:
                        # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                        relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                    else:
                        # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                        relevance_score = 0.3  # 기본값을 낮춤 (0.5 -> 0.3)
                    
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
                    
                    results.append({
                        "id": f"decision_para_{row['id']}",
                        "type": "decision",
                        "content": text_content,
                        "text": text_content,
                        "source": f"{row['org']} {row['doc_id']}",
                        "metadata": {
                            "decision_id": row['decision_id'],
                            "org": row['org'],
                            "doc_id": row['doc_id'],
                            "decision_date": row['decision_date'],
                            "result": row['result'],
                            "para_index": row['para_index'],
                        },
                        "relevance_score": relevance_score,
                        "search_type": "keyword"
                    })
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if not self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            
            self.logger.info(f"FTS search found {len(results)} decision paragraphs for query: {query}")
            
            # 결과가 없으면 폴백 검색 시도
            if len(results) == 0:
                self.logger.warning(f"No FTS results for query: '{query}' -> safe_query: '{safe_query}'. Trying fallback strategies...")
                import re
                words = safe_query.split() if safe_query else []
                results = self._fallback_decision_search(query, safe_query, words, limit)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in FTS decision search: {e}", exc_info=True)
            # 에러 발생 시에도 폴백 시도
            try:
                import re
                words = safe_query.split() if safe_query else []
                return self._fallback_decision_search(query, safe_query, words, limit)
            except Exception:
                return []

    def search_interpretations_fts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 키워드 검색: 유권해석 (최적화됨)"""
        try:
            # FTS5 쿼리 최적화 및 안전화
            optimized_query = self._optimize_fts5_query(query)
            safe_query = self._sanitize_fts5_query(optimized_query)
            if not safe_query:
                self.logger.warning(f"Empty or invalid FTS5 query: '{query}'")
                return []

            # 실행 계획 분석 (디버그 모드)
            if self.logger.isEnabledFor(logging.DEBUG):
                plan_info = self._analyze_query_plan(query, "interpretation_paragraphs_fts")
                if plan_info:
                    self.logger.debug(f"Query plan analysis: {plan_info}")

            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # SQL injection 방지: 파라미터 바인딩 사용
                cursor.execute("""
                    SELECT
                        ip.id,
                        ip.interpretation_id,
                        ip.para_index,
                        ip.text,
                        i.org,
                        i.doc_id,
                        i.title,
                        i.response_date,
                        bm25(interpretation_paragraphs_fts) as rank_score
                    FROM interpretation_paragraphs_fts
                    JOIN interpretation_paragraphs ip ON interpretation_paragraphs_fts.rowid = ip.id
                    JOIN interpretations i ON ip.interpretation_id = i.id
                    WHERE interpretation_paragraphs_fts MATCH ?
                    ORDER BY rank_score
                    LIMIT ?
                """, (safe_query, limit))

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
                    # relevance_score 계산 개선: rank_score가 없을 때 기본값 대신 계산된 점수 사용
                    if row['rank_score']:
                        # BM25 rank_score는 음수이므로 절댓값을 사용하여 정규화
                        relevance_score = max(0.0, min(1.0, abs(row['rank_score']) / 100.0))
                    else:
                        # rank_score가 없으면 문서 타입과 키워드 매칭에 기반한 점수 계산
                        relevance_score = 0.3  # 기본값을 낮춤 (0.5 -> 0.3)
                    
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
                    
                    results.append({
                        "id": f"interpretation_para_{row['id']}",
                        "type": "interpretation",
                        "content": text_content,
                        "text": text_content,
                        "source": f"{row['org']} {row['title']}",
                        "metadata": {
                            "interpretation_id": row['interpretation_id'],
                            "org": row['org'],
                            "doc_id": row['doc_id'],
                            "title": row['title'],
                            "response_date": row['response_date'],
                            "para_index": row['para_index'],
                        },
                        "relevance_score": relevance_score,
                        "search_type": "keyword"
                    })
            finally:
                # 연결 풀링 사용 시 close() 호출하지 않음
                if not self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            self.logger.info(f"FTS search found {len(results)} interpretation paragraphs for query: {query}")
            
            # 결과가 없으면 폴백 검색 시도
            if len(results) == 0:
                self.logger.warning(f"No FTS results for query: '{query}' -> safe_query: '{safe_query}'. Trying fallback strategies...")
                import re
                words = safe_query.split() if safe_query else []
                results = self._fallback_interpretation_search(query, safe_query, words, limit)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in FTS interpretation search: {e}", exc_info=True)
            # 에러 발생 시에도 폴백 시도
            try:
                import re
                words = safe_query.split() if safe_query else []
                return self._fallback_interpretation_search(query, safe_query, words, limit)
            except Exception:
                return []

    def search_documents(self, query: str, category: Optional[str] = None, limit: int = 10, force_fts: bool = False) -> List[Dict[str, Any]]:
        """
        통합 검색: 라우팅에 따라 FTS5 또는 벡터 검색 (병렬 처리)

        Args:
            query: 검색 쿼리
            category: 카테고리 (하위 호환성을 위해 유지, 현재 사용하지 않음)
            limit: 최대 결과 수
            force_fts: True이면 라우팅과 관계없이 강제로 FTS5 검색 수행

        Returns:
            검색 결과 리스트
        """
        # force_fts가 True이면 라우팅 무시하고 강제로 FTS5 검색
        if force_fts:
            return self._search_documents_parallel(query, limit=limit)
        
        # 기존 라우팅 로직
        route = route_query(query)
        
        if route == "text2sql":
            # FTS5 키워드 검색 (병렬 처리)
            return self._search_documents_parallel(query, limit=limit)
        else:
            # 벡터 검색은 별도 SemanticSearchEngineV2에서 처리
            # 여기서는 빈 리스트 반환하거나 기본 FTS 결과 반환
            self.logger.info(f"Vector search requested for: {query}, delegating to SemanticSearchEngineV2")
            return []
    
    def _search_documents_parallel(self, query: str, limit: int = 10, timeout: float = 10.0) -> List[Dict[str, Any]]:
        """
        병렬 FTS5 검색 실행
        
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
        search_tasks = {
            'statute': (self.search_statutes_fts, query, limit),
            'case': (self.search_cases_fts, query, limit),
            'decision': (self.search_decisions_fts, query, limit),
            'interpretation': (self.search_interpretations_fts, query, limit)
        }
        
        # ThreadPoolExecutor로 병렬 실행
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="FTS_Search") as executor:
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
        
        # 타입별 최소 보장 (statute_article: 1개, case: 2개, decision: 1개, interpretation: 1개)
        min_counts = {
            "statute_article": 1,
            "case": 2,
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
            f"from 4 tables in {elapsed_time:.3f}s for query: '{query[:50]}...' "
            f"(type distribution: {type_distribution})"
        )
        
        return results[:limit]

    @contextmanager
    def get_connection(self):
        """
        데이터베이스 연결 컨텍스트 매니저 (DatabaseManager 호환성)
        """
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            # 연결 풀링 사용 시 close() 호출하지 않음
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        쿼리 실행 (DatabaseManager 호환성)
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            쿼리 결과 리스트
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
        finally:
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """
        업데이트 쿼리 실행 (DatabaseManager 호환성)
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터
            
        Returns:
            영향받은 행 수
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Error executing update: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_all_categories(self) -> List[str]:
        """도메인 목록 반환 (하위 호환성)"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM domains ORDER BY name")
            categories = [row[0] for row in cursor.fetchall()]
            return categories
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return []
        finally:
            # 연결 풀링 사용 시 close() 호출하지 않음
            if conn and not self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass
