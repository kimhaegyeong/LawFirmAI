# -*- coding: utf-8 -*-
"""
lawfirm_v2.db 전용 법률 데이터 연동 서비스
FTS5 키워드 검색 + 벡터 의미 검색 지원
"""

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


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
        self.logger = logging.getLogger(__name__)
        # 데이터베이스 경로 로깅
        self.logger.info(f"LegalDataConnectorV2 initialized with database path: {self.db_path}")
        if not Path(self.db_path).exists():
            self.logger.warning(f"Database {self.db_path} not found. Please initialize it first.")
            self.logger.warning(f"To initialize, run: python scripts/init_lawfirm_v2_db.py")
        else:
            self.logger.info(f"Database {self.db_path} exists and is ready.")
            # FTS 테이블 존재 여부 확인
            self._check_fts_tables()
        
        # KoNLPy 형태소 분석기 초기화 (선택적)
        self._okt = None
        try:
            from konlpy.tag import Okt
            self._okt = Okt()
            self.logger.debug("KoNLPy Okt initialized successfully")
        except ImportError:
            self.logger.debug("KoNLPy not available, will use fallback method")
        except Exception as e:
            self.logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")

    def _check_fts_tables(self):
        """FTS 테이블 존재 여부 확인 및 초기화 필요 여부 안내"""
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
            
            missing_tables = []
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
            
            conn.close()
            
            if missing_tables:
                self.logger.error(
                    f"❌ Missing FTS tables: {', '.join(missing_tables)}. "
                    f"Please run: python scripts/init_lawfirm_v2_db.py"
                )
            if not has_embeddings:
                self.logger.warning(
                    f"⚠️ embeddings table not found. "
                    f"Semantic search will not work until embeddings are generated."
                )
            
            if missing_tables or not has_embeddings:
                self.logger.error(
                    f"Database initialization incomplete. "
                    f"Required tables missing: FTS={len(missing_tables)}, embeddings={'missing' if not has_embeddings else 'exists'}"
                )
            else:
                self.logger.info(f"✅ All required FTS tables and embeddings table exist.")
        except Exception as e:
            self.logger.warning(f"Error checking FTS tables: {e}")

    def _get_connection(self):
        """
        Get database connection
        
        Note: Each thread gets its own connection for thread safety.
        SQLite connections are not thread-safe, so each parallel search
        operation uses a separate connection.
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False  # Allow use in different threads (each thread has its own connection)
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

            conn.close()

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
        
        # 불용어 목록 (set으로 변환하여 빠른 조회)
        stopwords = {
            '에', '대해', '설명해주세요', '설명', '의', '을', '를', '이', '가', '는', '은', 
            '으로', '로', '에서', '에게', '한테', '께', '와', '과', '하고', '그리고', 
            '또는', '또한', '때문에', '위해', '통해', '관련', '및', '등', '등등', 
            '어떻게', '무엇', '언제', '어디', '어떤', '무엇인가', '요청', '질문', 
            '답변', '알려주세요', '알려주시기', '바랍니다'
        }
        
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
        
        # 나머지 키워드 처리
        for w in words:
            w_clean = josa_pattern.sub('', w.strip())  # 조사 제거
            
            if not w_clean or len(w_clean) < 2:
                continue
            
            if w_clean in stopwords or w_clean in core_keywords:
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
                if w_clean and len(w_clean) >= 2 and w_clean not in stopwords:
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
        """
        if not query or not query.strip():
            return ""

        import re
        query = query.strip()

        # FTS5 특수 문자 목록 (FTS5 문법에서 사용되는 문자)
        special_chars = ['"', ':', '(', ')', '?', '*', '-']

        # 1단계: 따옴표로 묶인 구문 제거 (예: "월세 50만원" -> 월세 50만원)
        # 따옴표 안의 내용은 단어로 추출
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        query_no_quotes = re.sub(r'"[^"]*"', '', query)
        
        # 2단계: AND, OR, NOT 키워드 제거 (대소문자 구분 없이)
        query_cleaned = re.sub(r'\b(AND|OR|NOT)\b', ' ', query_no_quotes, flags=re.IGNORECASE)
        
        # 3단계: 따옴표로 묶인 구문에서 단어 추출
        all_words = []
        for phrase in quoted_phrases:
            # 따옴표 안의 구문을 단어로 분리
            phrase_words = re.findall(r'[가-힣a-zA-Z0-9]+', phrase)
            all_words.extend(phrase_words)
        
        # 4단계: 나머지 쿼리에서 단어 추출
        remaining_words = re.findall(r'[가-힣a-zA-Z0-9]+', query_cleaned)
        all_words.extend(remaining_words)
        
        # 5단계: 중복 제거 및 길이 필터링
        clean_words = []
        seen = set()
        for word in all_words:
            # 최소 2자 이상, 최대 20자 이하
            if len(word) >= 2 and len(word) <= 20 and word not in seen:
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
        
        # 7단계: 최대 5개 단어만 사용 (FTS5 성능 최적화)
        clean_words = clean_words[:5]
        
        # 8단계: FTS5 쿼리 생성 (공백으로 구분하면 기본 AND 조건)
        # 개선: 단어가 1개일 때도 검색 가능하도록 처리, 더 공격적인 OR 조건 사용
        if len(clean_words) == 1:
            # 단어가 1개일 때는 그대로 사용
            sanitized = clean_words[0]
        elif len(clean_words) == 2:
            # 2개일 때는 OR 조건으로 연결 (검색 범위 확장)
            sanitized = " OR ".join(clean_words)
        elif len(clean_words) > 2:
            # 3개 이상이면 OR 조건으로 연결 (검색 범위 확장)
            sanitized = " OR ".join(clean_words)
        else:
            # 빈 경우는 이미 처리됨
            sanitized = " ".join(clean_words) if clean_words else ""
        
        # 9단계: SQL injection 방지
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
                    "type": "statute",
                    "content": text_content,
                    "text": text_content,  # text 필드도 추가 (호환성)
                    "source": row['statute_name'],
                    "metadata": {
                        "statute_id": row['statute_id'],
                        "article_no": row['article_no'],
                        "clause_no": row['clause_no'],
                        "item_no": row['item_no'],
                        "heading": row['heading'],
                        "statute_abbrv": row['statute_abbrv'],
                        "statute_type": row['statute_type'],
                        "category": row['category'],
                    },
                    "relevance_score": relevance_score,
                    "search_type": "keyword"
                })

            conn.close()
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
            # 법령명 추출 (민법, 형법, 상법 등)
            law_pattern = re.compile(r'(민법|형법|상법|행정법|헌법|노동법|가족법|민사소송법|형사소송법|상사법|공법|사법)')
            law_match = law_pattern.search(query)
            
            # 조문번호 추출 (제750조, 제 750 조 등)
            article_pattern = re.compile(r'제\s*(\d+)\s*조')
            article_match = article_pattern.search(query)
            
            if not law_match or not article_match:
                self.logger.debug(f"법령 조문 직접 검색: 법령명 또는 조문번호를 찾을 수 없음 (query: '{query}')")
                return []
            
            law_name = law_match.group(1)
            article_no = article_match.group(1)
            
            self.logger.info(f"⚖️ [DIRECT STATUTE SEARCH] 법령명: {law_name}, 조문번호: {article_no}")
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 법령명으로 statute_id 찾기
            cursor.execute("""
                SELECT id, name, abbrv 
                FROM statutes 
                WHERE name LIKE ? OR abbrv LIKE ? OR name = ?
                LIMIT 1
            """, (f"%{law_name}%", f"%{law_name}%", law_name))
            
            statute_row = cursor.fetchone()
            if not statute_row:
                self.logger.warning(f"법령을 찾을 수 없음: {law_name}")
                conn.close()
                return []
            
            statute_id = statute_row['id']
            statute_name = statute_row['name']
            
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
            
            conn.close()
            
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
            # 전략 1: 법령명 + 조문 번호만으로 검색 (예: "민법 제750조")
            article_match = re.search(r'제\s*\d+\s*조', original_query)
            law_match = re.search(r'(민법|형법|상법|행정법|헌법|노동법|가족법)', original_query)
            
            if law_match and article_match:
                law_name = law_match.group()
                article_no = article_match.group().replace(' ', '')
                fallback_query = f"{law_name} OR {article_no}"
                self.logger.info(f"Fallback 1: Searching with law name + article: '{fallback_query}'")
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
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
                            "type": "statute",
                            "content": text_content,
                            "text": text_content,
                            "source": row['statute_name'],
                            "metadata": {
                                "statute_id": row['statute_id'],
                                "article_no": row['article_no'],
                                "clause_no": row['clause_no'],
                                "item_no": row['item_no'],
                                "heading": row['heading'],
                                "statute_abbrv": row['statute_abbrv'],
                                "statute_type": row['statute_type'],
                                "category": row['category'],
                            },
                            "relevance_score": relevance_score,
                            "search_type": "keyword",
                            "fallback_strategy": "law_article_only"
                        })
                
                conn.close()
                
                if fallback_results:
                    self.logger.info(f"Fallback 1 found {len(fallback_results)} results")
                    return fallback_results[:limit]
            
            # 전략 2: 핵심 키워드만으로 검색 (법령명 또는 주요 키워드)
            if not fallback_results and words:
                # 법령명이 있으면 법령명으로, 없으면 첫 번째 핵심 키워드로
                if law_match:
                    keyword_query = law_match.group()
                else:
                    # 핵심 키워드 추출 (2자 이상, 불용어 제외)
                    stopwords = ['에', '대해', '설명해주세요', '의', '을', '를', '이', '가', '는', '은']
                    keywords = [w for w in words[:3] if w not in stopwords and len(w) >= 2]
                    if keywords:
                        keyword_query = keywords[0]  # 첫 번째 핵심 키워드만
                    else:
                        return []
                
                self.logger.info(f"Fallback 2: Searching with keyword only: '{keyword_query}'")
                
                conn = self._get_connection()
                cursor = conn.cursor()
                
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
                            "type": "statute",
                            "content": text_content,
                            "text": text_content,
                            "source": row['statute_name'],
                            "metadata": {
                                "statute_id": row['statute_id'],
                                "article_no": row['article_no'],
                                "clause_no": row['clause_no'],
                                "item_no": row['item_no'],
                                "heading": row['heading'],
                                "statute_abbrv": row['statute_abbrv'],
                                "statute_type": row['statute_type'],
                                "category": row['category'],
                            },
                            "relevance_score": relevance_score,
                            "search_type": "keyword",
                            "fallback_strategy": "keyword_only"
                        })
                
                conn.close()
                
                if fallback_results:
                    self.logger.info(f"Fallback 2 found {len(fallback_results)} results")
                    return fallback_results[:limit]
            
            # 전략 3: 조문 번호만으로 검색 (예: "750조" 또는 "제750조")
            if not fallback_results and article_match:
                article_no_clean = article_match.group().replace(' ', '')
                # 숫자만 추출
                article_num_match = re.search(r'\d+', article_no_clean)
                if article_num_match:
                    article_num = article_num_match.group()
                    # "750조" 또는 "제750조" 패턴으로 검색
                    fallback_query = f"{article_num}조 OR 제{article_num}조"
                    
                    self.logger.info(f"Fallback 3: Searching with article number only: '{fallback_query}'")
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    
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
                                "type": "statute",
                                "content": text_content,
                                "text": text_content,
                                "source": row['statute_name'],
                                "metadata": {
                                    "statute_id": row['statute_id'],
                                    "article_no": row['article_no'],
                                    "clause_no": row['clause_no'],
                                    "item_no": row['item_no'],
                                    "heading": row['heading'],
                                    "statute_abbrv": row['statute_abbrv'],
                                    "statute_type": row['statute_type'],
                                    "category": row['category'],
                                },
                                "relevance_score": relevance_score,
                                "search_type": "keyword",
                                "fallback_strategy": "article_number_only"
                            })
                    
                    conn.close()
                    
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
            # 전략 1: 핵심 키워드만으로 검색 (개선: 더 공격적인 키워드 추출)
            if words:
                stopwords = {'에', '대해', '설명해주세요', '설명', '의', '을', '를', '이', '가', '는', '은',
                            '으로', '로', '에서', '에게', '한테', '께', '와', '과', '하고', '그리고',
                            '또는', '또한', '때문에', '위해', '통해', '관련', '및', '등', '등등',
                            '어떻게', '무엇', '언제', '어디', '어떤', '무엇인가', '요청', '질문',
                            '답변', '알려주세요', '알려주시기', '바랍니다', '해주세요', '해주시기'}
                # 더 많은 키워드 추출 (3개 -> 5개)
                keywords = [w for w in words[:5] if w not in stopwords and len(w) >= 1]  # 최소 길이 2 -> 1로 완화
                if not keywords:
                    # 키워드가 없으면 모든 단어 사용 (불용어만 제거)
                    keywords = [w for w in words if w not in stopwords and len(w) >= 1]
                
                if keywords:
                    # OR 조건으로 연결 (검색 범위 확장)
                    keyword_query = " OR ".join(keywords[:5])  # 최대 5개
                    self.logger.info(f"Fallback case search: Using keywords: '{keyword_query}'")
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    
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
                    
                    conn.close()
                    
                    if fallback_results:
                        self.logger.info(f"Fallback case search found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            # 전략 2: 원본 쿼리에서 직접 키워드 추출 (전략 1 실패 시)
            if not fallback_results and original_query:
                import re
                # 원본 쿼리에서 모든 한글/영문 단어 추출
                all_words = re.findall(r'[가-힣a-zA-Z]+', original_query)
                if all_words:
                    # 불용어 제거
                    stopwords = {'에', '대해', '설명해주세요', '설명', '의', '을', '를', '이', '가', '는', '은'}
                    keywords = [w for w in all_words if w not in stopwords and len(w) >= 1]
                    if keywords:
                        keyword_query = " OR ".join(keywords[:3])  # 최대 3개
                        self.logger.info(f"Fallback case search (strategy 2): Using keywords from original query: '{keyword_query}'")
                        
                        try:
                            conn = self._get_connection()
                            cursor = conn.cursor()
                            
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
                            
                            conn.close()
                            
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
        import re
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 핵심 키워드만으로 검색
            if words:
                stopwords = {'에', '대해', '설명해주세요', '의', '을', '를', '이', '가', '는', '은'}
                keywords = [w for w in words[:3] if w not in stopwords and len(w) >= 2]
                if keywords:
                    keyword_query = " OR ".join(keywords)
                    self.logger.info(f"Fallback decision search: Using keywords: '{keyword_query}'")
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    
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
                    
                    conn.close()
                    
                    if fallback_results:
                        self.logger.info(f"Fallback decision search found {len(fallback_results)} results")
                        return fallback_results[:limit]
            
            return fallback_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in fallback decision search: {e}", exc_info=True)
            return []
    
    def _fallback_interpretation_search(self, original_query: str, safe_query: str, words: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """폴백 검색 전략: 유권해석 검색"""
        import re
        fallback_results = []
        seen_ids = set()
        
        try:
            # 전략 1: 핵심 키워드만으로 검색
            if words:
                stopwords = {'에', '대해', '설명해주세요', '의', '을', '를', '이', '가', '는', '은'}
                keywords = [w for w in words[:3] if w not in stopwords and len(w) >= 2]
                if keywords:
                    keyword_query = " OR ".join(keywords)
                    self.logger.info(f"Fallback interpretation search: Using keywords: '{keyword_query}'")
                    
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    
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
                    
                    conn.close()
                    
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

            conn.close()
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

            conn.close()
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

            conn.close()
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
        
        # relevance_score 기준 정렬
        results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        # 성능 로깅
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Parallel FTS search completed: {len(results)} total results "
            f"from 4 tables in {elapsed_time:.3f}s for query: '{query[:50]}...'"
        )
        
        return results[:limit]

    def get_all_categories(self) -> List[str]:
        """도메인 목록 반환 (하위 호환성)"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM domains ORDER BY name")
            categories = [row[0] for row in cursor.fetchall()]
            conn.close()
            return categories
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return []
