"""
정확한 매칭 검색 엔진 (SQLite 기반)
법령명, 조문번호, 사건번호 등 정확한 매칭을 위한 검색 엔진
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import re

logger = logging.getLogger(__name__)

class ExactSearchEngine:
    """정확한 매칭 검색 엔진"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 법령 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS laws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_name TEXT NOT NULL,
                    article_number TEXT,
                    content TEXT NOT NULL,
                    law_type TEXT,
                    effective_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 판례 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_number TEXT NOT NULL,
                    court_name TEXT,
                    decision_date TEXT,
                    case_name TEXT,
                    content TEXT NOT NULL,
                    case_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 헌재결정례 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS constitutional_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_number TEXT NOT NULL,
                    decision_date TEXT,
                    case_name TEXT,
                    content TEXT NOT NULL,
                    decision_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_name ON laws(law_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_laws_article ON laws(article_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedents_case_number ON precedents(case_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedents_court ON precedents(court_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_constitutional_case_number ON constitutional_decisions(case_number)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def search_laws(self, query: str, law_name: str = None, article_number: str = None) -> List[Dict[str, Any]]:
        """법령 검색"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if law_name:
                conditions.append("law_name LIKE ?")
                params.append(f"%{law_name}%")
            
            if article_number:
                conditions.append("article_number LIKE ?")
                params.append(f"%{article_number}%")
            
            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            sql = f"""
                SELECT id, law_name, article_number, content, law_type, effective_date
                FROM laws
                WHERE {where_clause}
                ORDER BY law_name, article_number
                LIMIT 50
            """
            
            cursor.execute(sql, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "law_name": row["law_name"],
                    "article_number": row["article_number"],
                    "content": row["content"],
                    "law_type": row["law_type"],
                    "effective_date": row["effective_date"],
                    "search_type": "exact_match",
                    "relevance_score": 1.0
                })
            
            return results
    
    def search_precedents(self, query: str, case_number: str = None, court_name: str = None) -> List[Dict[str, Any]]:
        """판례 검색"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if case_number:
                conditions.append("case_number LIKE ?")
                params.append(f"%{case_number}%")
            
            if court_name:
                conditions.append("court_name LIKE ?")
                params.append(f"%{court_name}%")
            
            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            sql = f"""
                SELECT id, case_number, court_name, decision_date, case_name, content, case_type
                FROM precedents
                WHERE {where_clause}
                ORDER BY decision_date DESC
                LIMIT 50
            """
            
            cursor.execute(sql, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "case_number": row["case_number"],
                    "court_name": row["court_name"],
                    "decision_date": row["decision_date"],
                    "case_name": row["case_name"],
                    "content": row["content"],
                    "case_type": row["case_type"],
                    "search_type": "exact_match",
                    "relevance_score": 1.0
                })
            
            return results
    
    def search_constitutional_decisions(self, query: str, case_number: str = None) -> List[Dict[str, Any]]:
        """헌재결정례 검색"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if case_number:
                conditions.append("case_number LIKE ?")
                params.append(f"%{case_number}%")
            
            if query:
                conditions.append("content LIKE ?")
                params.append(f"%{query}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            sql = f"""
                SELECT id, case_number, decision_date, case_name, content, decision_type
                FROM constitutional_decisions
                WHERE {where_clause}
                ORDER BY decision_date DESC
                LIMIT 50
            """
            
            cursor.execute(sql, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "case_number": row["case_number"],
                    "decision_date": row["decision_date"],
                    "case_name": row["case_name"],
                    "content": row["content"],
                    "decision_type": row["decision_type"],
                    "search_type": "exact_match",
                    "relevance_score": 1.0
                })
            
            return results
    
    def search_all(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """전체 검색"""
        results = {
            "laws": self.search_laws(query),
            "precedents": self.search_precedents(query),
            "constitutional_decisions": self.search_constitutional_decisions(query)
        }
        
        return results
    
    def parse_query(self, query: str) -> Dict[str, str]:
        """쿼리 파싱 (법령명, 조문번호, 사건번호 등 추출)"""
        parsed = {
            "law_name": None,
            "article_number": None,
            "case_number": None,
            "court_name": None,
            "raw_query": query
        }
        
        # 법령명 패턴 (예: 민법, 상법, 형법 등)
        law_patterns = [
            r'([가-힣]+법)',
            r'([가-힣]+법률)',
            r'([가-힣]+령)',
            r'([가-힣]+규칙)'
        ]
        
        for pattern in law_patterns:
            match = re.search(pattern, query)
            if match:
                parsed["law_name"] = match.group(1)
                break
        
        # 조문번호 패턴 (예: 제1조, 제2항, 제3호 등)
        article_patterns = [
            r'제(\d+)조',
            r'제(\d+)항',
            r'제(\d+)호'
        ]
        
        for pattern in article_patterns:
            match = re.search(pattern, query)
            if match:
                parsed["article_number"] = f"제{match.group(1)}조"
                break
        
        # 사건번호 패턴 (예: 2024다12345, 2024구합123 등)
        case_patterns = [
            r'(\d{4}[가-힣]+\d+)',
            r'(\d{4}다\d+)',
            r'(\d{4}구합\d+)',
            r'(\d{4}헌마\d+)'
        ]
        
        for pattern in case_patterns:
            match = re.search(pattern, query)
            if match:
                parsed["case_number"] = match.group(1)
                break
        
        # 법원명 패턴
        court_patterns = [
            r'(대법원)',
            r'(고등법원)',
            r'(지방법원)',
            r'(가정법원)',
            r'(행정법원)'
        ]
        
        for pattern in court_patterns:
            match = re.search(pattern, query)
            if match:
                parsed["court_name"] = match.group(1)
                break
        
        return parsed
    
    def insert_law(self, law_name: str, article_number: str, content: str, 
                   law_type: str = None, effective_date: str = None) -> int:
        """법령 데이터 삽입"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO laws (law_name, article_number, content, law_type, effective_date)
                VALUES (?, ?, ?, ?, ?)
            """, (law_name, article_number, content, law_type, effective_date))
            conn.commit()
            return cursor.lastrowid
    
    def insert_precedent(self, case_number: str, court_name: str, decision_date: str,
                        case_name: str, content: str, case_type: str = None) -> int:
        """판례 데이터 삽입"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO precedents (case_number, court_name, decision_date, case_name, content, case_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (case_number, court_name, decision_date, case_name, content, case_type))
            conn.commit()
            return cursor.lastrowid
    
    def insert_constitutional_decision(self, case_number: str, decision_date: str,
                                     case_name: str, content: str, decision_type: str = None) -> int:
        """헌재결정례 데이터 삽입"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO constitutional_decisions (case_number, decision_date, case_name, content, decision_type)
                VALUES (?, ?, ?, ?, ?)
            """, (case_number, decision_date, case_name, content, decision_type))
            conn.commit()
            return cursor.lastrowid
