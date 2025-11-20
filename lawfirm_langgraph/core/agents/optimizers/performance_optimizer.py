# -*- coding: utf-8 -*-
"""
성능 최적화를 위한 캐싱 시스템
답변 캐싱, 키워드 캐싱, 문서 캐싱 구현
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from functools import lru_cache
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceCache:
    """성능 최적화를 위한 캐싱 시스템"""
    
    def __init__(self, cache_db_path: str = "./data/cache.db"):
        self.cache_db_path = cache_db_path
        self.logger = logging.getLogger(__name__)
        # 연결 풀 초기화
        try:
            from core.data.connection_pool import get_connection_pool
            self._connection_pool = get_connection_pool(self.cache_db_path)
            self.logger.debug("Using connection pool for cache database")
        except ImportError:
            try:
                from lawfirm_langgraph.core.data.connection_pool import get_connection_pool
                self._connection_pool = get_connection_pool(self.cache_db_path)
                self.logger.debug("Using connection pool for cache database")
            except ImportError:
                self._connection_pool = None
                self.logger.debug("Connection pool not available, using direct connections")
        self._initialize_cache_db()
    
    def _initialize_cache_db(self):
        """캐시 데이터베이스 초기화"""
        Path(self.cache_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        if self._connection_pool:
            conn = self._connection_pool.get_connection()
        else:
            conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # 답변 캐시 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answer_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                sources TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 키워드 캐시 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                keywords TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 문서 캐시 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                query_type TEXT NOT NULL,
                documents TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_answer_query_hash ON answer_cache(query_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_keyword_query_hash ON keyword_cache(query_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_query_hash ON document_cache(query_hash)')
        
        conn.commit()
        if not self._connection_pool:
            conn.close()
        self.logger.info("Cache database initialized")
    
    def _generate_query_hash(self, query: str, query_type: str = "") -> str:
        """쿼리 해시 생성"""
        combined = f"{query.lower().strip()}:{query_type}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get_cached_answer(self, query: str, query_type: str) -> Optional[Dict[str, Any]]:
        """캐시된 답변 조회"""
        try:
            query_hash = self._generate_query_hash(query, query_type)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT answer, confidence, sources, access_count
                FROM answer_cache 
                WHERE query_hash = ?
            ''', (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # 접근 횟수 증가 및 마지막 접근 시간 업데이트
                cursor.execute('''
                    UPDATE answer_cache 
                    SET access_count = access_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                ''', (query_hash,))
                conn.commit()
                
                result = {
                    "answer": row["answer"],
                    "confidence": row["confidence"],
                    "sources": json.loads(row["sources"]),
                    "cached": True,
                    "access_count": row["access_count"]
                }
                
                if not self._connection_pool:
                    conn.close()
                self.logger.info(f"Cache hit for query: {query[:50]}...")
                return result
            
            if not self._connection_pool:
                conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached answer: {e}")
            return None
    
    def cache_answer(self, query: str, query_type: str, answer: str, confidence: float, sources: List[str]) -> bool:
        """답변 캐시 저장"""
        try:
            query_hash = self._generate_query_hash(query, query_type)
            
            # answer가 dict인 경우 JSON 문자열로 변환
            if isinstance(answer, dict):
                answer_str = json.dumps(answer, ensure_ascii=False)
            elif isinstance(answer, str):
                answer_str = answer
            else:
                answer_str = str(answer)
            
            # sources가 dict 리스트인 경우 처리
            if sources and isinstance(sources, list):
                # 각 source가 dict인지 확인하고 문자열로 변환
                sources_list = []
                for source in sources:
                    if isinstance(source, dict):
                        sources_list.append(json.dumps(source, ensure_ascii=False))
                    elif isinstance(source, str):
                        sources_list.append(source)
                    else:
                        sources_list.append(str(source))
                sources_json = json.dumps(sources_list, ensure_ascii=False)
            elif isinstance(sources, (dict, list)):
                sources_json = json.dumps(sources, ensure_ascii=False)
            else:
                sources_json = json.dumps([], ensure_ascii=False)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO answer_cache 
                (query_hash, query_text, query_type, answer, confidence, sources)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (query_hash, query, query_type, answer_str, confidence, sources_json))
            
            conn.commit()
            if not self._connection_pool:
                conn.close()
            
            self.logger.info(f"Cached answer for query: {query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching answer: {e}")
            return False
    
    def get_cached_keywords(self, query: str) -> Optional[List[str]]:
        """캐시된 키워드 조회"""
        try:
            query_hash = self._generate_query_hash(query)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT keywords, access_count
                FROM keyword_cache 
                WHERE query_hash = ?
            ''', (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # 접근 횟수 증가
                cursor.execute('''
                    UPDATE keyword_cache 
                    SET access_count = access_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                ''', (query_hash,))
                conn.commit()
                
                keywords = json.loads(row["keywords"])
                if not self._connection_pool:
                    conn.close()
                
                self.logger.info(f"Cache hit for keywords: {query[:50]}...")
                return keywords
            
            if not self._connection_pool:
                conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached keywords: {e}")
            return None
    
    def cache_keywords(self, query: str, keywords: List[str]) -> bool:
        """키워드 캐시 저장"""
        try:
            query_hash = self._generate_query_hash(query)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO keyword_cache 
                (query_hash, query_text, keywords)
                VALUES (?, ?, ?)
            ''', (query_hash, query, json.dumps(keywords)))
            
            conn.commit()
            if not self._connection_pool:
                conn.close()
            
            self.logger.info(f"Cached keywords for query: {query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching keywords: {e}")
            return False
    
    def get_cached_documents(self, query: str, query_type: str) -> Optional[List[Dict[str, Any]]]:
        """캐시된 문서 조회"""
        try:
            query_hash = self._generate_query_hash(query, query_type)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT documents, access_count
                FROM document_cache 
                WHERE query_hash = ?
            ''', (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # 접근 횟수 증가
                cursor.execute('''
                    UPDATE document_cache 
                    SET access_count = access_count + 1, 
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE query_hash = ?
                ''', (query_hash,))
                conn.commit()
                
                documents = json.loads(row["documents"])
                if not self._connection_pool:
                    conn.close()
                
                self.logger.info(f"Cache hit for documents: {query[:50]}...")
                return documents
            
            if not self._connection_pool:
                conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached documents: {e}")
            return None
    
    def cache_documents(self, query: str, query_type: str, documents: List[Dict[str, Any]]) -> bool:
        """문서 캐시 저장"""
        try:
            query_hash = self._generate_query_hash(query, query_type)
            
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO document_cache 
                (query_hash, query_text, query_type, documents)
                VALUES (?, ?, ?, ?)
            ''', (query_hash, query, query_type, json.dumps(documents)))
            
            conn.commit()
            if not self._connection_pool:
                conn.close()
            
            self.logger.info(f"Cached documents for query: {query[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching documents: {e}")
            return False
    
    def clear_cache(self, cache_type: str = "all") -> bool:
        """캐시 정리"""
        try:
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            if cache_type == "all":
                cursor.execute('DELETE FROM answer_cache')
                cursor.execute('DELETE FROM keyword_cache')
                cursor.execute('DELETE FROM document_cache')
            elif cache_type == "answer":
                cursor.execute('DELETE FROM answer_cache')
            elif cache_type == "keyword":
                cursor.execute('DELETE FROM keyword_cache')
            elif cache_type == "document":
                cursor.execute('DELETE FROM document_cache')
            
            conn.commit()
            if not self._connection_pool:
                conn.close()
            
            self.logger.info(f"Cleared {cache_type} cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        try:
            if self._connection_pool:
                conn = self._connection_pool.get_connection()
            else:
                conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # 각 캐시 테이블의 통계
            cursor.execute('SELECT COUNT(*) FROM answer_cache')
            answer_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM keyword_cache')
            keyword_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM document_cache')
            document_count = cursor.fetchone()[0]
            
            # 총 접근 횟수
            cursor.execute('SELECT SUM(access_count) FROM answer_cache')
            total_answer_access = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(access_count) FROM keyword_cache')
            total_keyword_access = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(access_count) FROM document_cache')
            total_document_access = cursor.fetchone()[0] or 0
            
            if not self._connection_pool:
                conn.close()
            
            return {
                "answer_cache": {
                    "count": answer_count,
                    "total_access": total_answer_access
                },
                "keyword_cache": {
                    "count": keyword_count,
                    "total_access": total_keyword_access
                },
                "document_cache": {
                    "count": document_count,
                    "total_access": total_document_access
                },
                "total_cache_size": answer_count + keyword_count + document_count,
                "total_access": total_answer_access + total_keyword_access + total_document_access
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}


class PerformanceOptimizer:
    """성능 최적화 관리자"""
    
    def __init__(self):
        self.cache = PerformanceCache()
        self.logger = logging.getLogger(__name__)
    
    def optimize_query_processing(self, query: str, query_type: str) -> Dict[str, Any]:
        """쿼리 처리 최적화"""
        start_time = time.time()
        
        # 캐시에서 답변 확인
        cached_answer = self.cache.get_cached_answer(query, query_type)
        if cached_answer:
            processing_time = time.time() - start_time
            cached_answer["processing_time"] = processing_time
            cached_answer["optimization_applied"] = "answer_cache"
            return cached_answer
        
        # 캐시에서 키워드 확인
        cached_keywords = self.cache.get_cached_keywords(query)
        
        # 캐시에서 문서 확인
        cached_documents = self.cache.get_cached_documents(query, query_type)
        
        optimization_info = {
            "answer_cached": cached_answer is not None,
            "keywords_cached": cached_keywords is not None,
            "documents_cached": cached_documents is not None,
            "processing_time": time.time() - start_time
        }
        
        return optimization_info
    
    def cache_query_result(self, query: str, query_type: str, result: Dict[str, Any]) -> bool:
        """쿼리 결과 캐싱"""
        try:
            # 답변 캐싱
            if "response" in result:
                self.cache.cache_answer(
                    query, query_type, 
                    result["response"], 
                    result.get("confidence", 0.0),
                    result.get("sources", [])
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching query result: {e}")
            return False
