# -*- coding: utf-8 -*-
"""
Database Management
데이터베이스 관리 모듈
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/lawfirm.db"):
        """데이터베이스 관리자 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 채팅 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    processing_time REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 법률 문서 테이블 (하이브리드 검색용)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    document_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 법령 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS law_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    law_name TEXT,
                    article_number INTEGER,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 판례 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedent_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    case_number TEXT,
                    court_name TEXT,
                    decision_date TEXT,
                    case_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 헌재결정례 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS constitutional_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    case_number TEXT,
                    decision_date TEXT,
                    case_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 법령해석례 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interpretation_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    interpretation_date TEXT,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 행정규칙 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS administrative_rule_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 자치법규 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS local_ordinance_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    local_government TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 처리된 파일 추적 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'completed',
                    record_count INTEGER DEFAULT 0,
                    processing_version TEXT DEFAULT '1.0',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성 (검색 성능 향상)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_law_metadata_name ON law_metadata(law_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_metadata_court ON precedent_metadata(court_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_metadata_date ON precedent_metadata(decision_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_path ON processed_files(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_type ON processed_files(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_status ON processed_files(processing_status)")
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """쿼리 실행"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """업데이트 쿼리 실행"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def insert_chat_message(self, session_id: str, user_message: str, 
                           bot_response: str, confidence: float = 0.0, 
                           processing_time: float = 0.0) -> int:
        """채팅 메시지 저장"""
        query = """
            INSERT INTO chat_history 
            (session_id, user_message, bot_response, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?)
        """
        params = (session_id, user_message, bot_response, confidence, processing_time)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """채팅 기록 조회"""
        query = """
            SELECT * FROM chat_history 
            WHERE session_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        """
        return self.execute_query(query, (session_id, limit))
    
    def clear_chat_history(self, session_id: str) -> int:
        """채팅 기록 삭제"""
        query = "DELETE FROM chat_history WHERE session_id = ?"
        return self.execute_update(query, (session_id,))
    
    def add_document(self, doc_data: dict, law_meta: dict = None, prec_meta: dict = None, 
                    const_meta: dict = None, interp_meta: dict = None, 
                    admin_meta: dict = None, local_meta: dict = None) -> bool:
        """문서 추가 (하이브리드 검색용)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 문서 기본 정보 저장
                cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, document_type, title, content, source_url)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    doc_data['id'],
                    doc_data['document_type'],
                    doc_data['title'],
                    doc_data['content'],
                    doc_data.get('source_url')
                ))
                
                # 법령 메타데이터 저장
                if law_meta and doc_data['document_type'] == 'law':
                    cursor.execute("""
                        INSERT OR REPLACE INTO law_metadata 
                        (document_id, law_name, article_number, promulgation_date, enforcement_date, department)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        doc_data['id'],
                        law_meta.get('law_name'),
                        law_meta.get('article_number'),
                        law_meta.get('promulgation_date'),
                        law_meta.get('enforcement_date'),
                        law_meta.get('department')
                    ))
                
                # 판례 메타데이터 저장
                if prec_meta and doc_data['document_type'] == 'precedent':
                    cursor.execute("""
                        INSERT OR REPLACE INTO precedent_metadata 
                        (document_id, case_number, court_name, decision_date, case_type)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        doc_data['id'],
                        prec_meta.get('case_number'),
                        prec_meta.get('court_name'),
                        prec_meta.get('decision_date'),
                        prec_meta.get('case_type')
                    ))
                
                # 헌재결정례 메타데이터 저장
                if const_meta and doc_data['document_type'] == 'constitutional_decision':
                    cursor.execute("""
                        INSERT OR REPLACE INTO constitutional_metadata 
                        (document_id, case_number, decision_date, case_type)
                        VALUES (?, ?, ?, ?)
                    """, (
                        doc_data['id'],
                        const_meta.get('case_number'),
                        const_meta.get('decision_date'),
                        const_meta.get('case_type')
                    ))
                
                # 법령해석례 메타데이터 저장
                if interp_meta and doc_data['document_type'] == 'legal_interpretation':
                    cursor.execute("""
                        INSERT OR REPLACE INTO interpretation_metadata 
                        (document_id, interpretation_date, department)
                        VALUES (?, ?, ?)
                    """, (
                        doc_data['id'],
                        interp_meta.get('interpretation_date'),
                        interp_meta.get('department')
                    ))
                
                # 행정규칙 메타데이터 저장
                if admin_meta and doc_data['document_type'] == 'administrative_rule':
                    cursor.execute("""
                        INSERT OR REPLACE INTO administrative_rule_metadata 
                        (document_id, promulgation_date, enforcement_date, department)
                        VALUES (?, ?, ?, ?)
                    """, (
                        doc_data['id'],
                        admin_meta.get('promulgation_date'),
                        admin_meta.get('enforcement_date'),
                        admin_meta.get('department')
                    ))
                
                # 자치법규 메타데이터 저장
                if local_meta and doc_data['document_type'] == 'local_ordinance':
                    cursor.execute("""
                        INSERT OR REPLACE INTO local_ordinance_metadata 
                        (document_id, promulgation_date, enforcement_date, local_government)
                        VALUES (?, ?, ?, ?)
                    """, (
                        doc_data['id'],
                        local_meta.get('promulgation_date'),
                        local_meta.get('enforcement_date'),
                        local_meta.get('local_government')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def search_exact(self, query: str, filters: dict = None, limit: int = 20, offset: int = 0) -> tuple:
        """정확한 매칭 검색 (하이브리드 검색용)"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # 기본 쿼리
                base_query = """
                    SELECT d.id, d.title, d.content, d.document_type, d.source_url
                    FROM documents d
                    WHERE d.content LIKE ?
                """
                params = [f"%{query}%"]
                
                # 필터 적용
                if filters:
                    if 'document_type' in filters:
                        base_query += " AND d.document_type = ?"
                        params.append(filters['document_type'])
                    
                    if 'law_name' in filters:
                        base_query += " AND EXISTS (SELECT 1 FROM law_metadata lm WHERE lm.document_id = d.id AND lm.law_name LIKE ?)"
                        params.append(f"%{filters['law_name']}%")
                    
                    if 'court_name' in filters:
                        base_query += " AND EXISTS (SELECT 1 FROM precedent_metadata pm WHERE pm.document_id = d.id AND pm.court_name LIKE ?)"
                        params.append(f"%{filters['court_name']}%")
                    
                    if 'department' in filters:
                        base_query += " AND (EXISTS (SELECT 1 FROM law_metadata lm WHERE lm.document_id = d.id AND lm.department LIKE ?) OR EXISTS (SELECT 1 FROM interpretation_metadata im WHERE im.document_id = d.id AND im.department LIKE ?) OR EXISTS (SELECT 1 FROM administrative_rule_metadata arm WHERE arm.document_id = d.id AND arm.department LIKE ?))"
                        params.extend([f"%{filters['department']}%", f"%{filters['department']}%", f"%{filters['department']}%"])
                
                # 정렬 및 제한
                base_query += " ORDER BY d.updated_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                # 결과 조회
                cursor.execute(base_query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # 전체 개수 조회
                count_query = base_query.replace("SELECT d.id, d.title, d.content, d.document_type, d.source_url", "SELECT COUNT(*)")
                count_query = count_query.replace("ORDER BY d.updated_at DESC LIMIT ? OFFSET ?", "")
                count_params = params[:-2]  # LIMIT, OFFSET 제거
                
                cursor.execute(count_query, count_params)
                total_count = cursor.fetchone()[0]
                
                return results, total_count
                
        except Exception as e:
            logger.error(f"Error in exact search: {e}")
            return [], 0
    
    def get_document_by_id(self, doc_id: str) -> dict:
        """ID로 문서 조회"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT d.*, 
                           lm.law_name, lm.article_number, lm.promulgation_date, lm.enforcement_date, lm.department,
                           pm.case_number, pm.court_name, pm.decision_date, pm.case_type,
                           cm.case_number as const_case_number, cm.decision_date as const_decision_date, cm.case_type as const_case_type,
                           im.interpretation_date, im.department as interp_department,
                           arm.promulgation_date as admin_promulgation_date, arm.enforcement_date as admin_enforcement_date, arm.department as admin_department,
                           lom.promulgation_date as local_promulgation_date, lom.enforcement_date as local_enforcement_date, lom.local_government
                    FROM documents d
                    LEFT JOIN law_metadata lm ON d.id = lm.document_id
                    LEFT JOIN precedent_metadata pm ON d.id = pm.document_id
                    LEFT JOIN constitutional_metadata cm ON d.id = cm.document_id
                    LEFT JOIN interpretation_metadata im ON d.id = im.document_id
                    LEFT JOIN administrative_rule_metadata arm ON d.id = arm.document_id
                    LEFT JOIN local_ordinance_metadata lom ON d.id = lom.document_id
                    WHERE d.id = ?
                """, (doc_id,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def search_assembly_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Assembly 데이터베이스에서 문서 검색"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Assembly 테이블에서 검색
                sql = """
                    SELECT al.law_id, al.law_name, aa.article_number, aa.article_title, 
                           aa.article_content, aa.article_type, aa.is_supplementary,
                           aa.ml_confidence_score, aa.parsing_method, aa.parsing_quality_score,
                           aa.word_count, aa.char_count
                    FROM assembly_laws al
                    JOIN assembly_articles aa ON al.law_id = aa.law_id
                    WHERE (al.law_name LIKE ? OR aa.article_content LIKE ? OR aa.article_title LIKE ?)
                    ORDER BY aa.parsing_quality_score DESC, aa.word_count DESC
                    LIMIT ?
                """
                
                search_term = f"%{query}%"
                cursor.execute(sql, (search_term, search_term, search_term, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        "law_id": row["law_id"],
                        "law_name": row["law_name"],
                        "article_number": row["article_number"],
                        "article_title": row["article_title"],
                        "content": row["article_content"],
                        "article_type": row["article_type"],
                        "is_supplementary": bool(row["is_supplementary"]),
                        "ml_confidence_score": row["ml_confidence_score"],
                        "parsing_method": row["parsing_method"],
                        "quality_score": row["parsing_quality_score"],
                        "word_count": row["word_count"],
                        "char_count": row["char_count"],
                        "relevance_score": 1.0  # 정확한 매칭이므로 높은 점수
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching assembly documents: {e}")
            return []
    
    # 파일 처리 이력 추적 메서드들
    def mark_file_as_processed(self, file_path: str, file_hash: str, 
                              data_type: str, record_count: int = 0,
                              processing_version: str = "1.0",
                              error_message: str = None) -> int:
        """
        파일을 처리 완료로 표시
        
        Args:
            file_path: 처리된 파일 경로
            file_hash: 파일 해시값
            data_type: 데이터 유형
            record_count: 처리된 레코드 수
            processing_version: 처리 버전
            error_message: 에러 메시지 (실패한 경우)
        
        Returns:
            int: 삽입된 레코드의 ID
        """
        query = """
            INSERT OR REPLACE INTO processed_files 
            (file_path, file_hash, data_type, record_count, 
             processing_version, error_message, processing_status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        status = "failed" if error_message else "completed"
        params = (file_path, file_hash, data_type, record_count, 
                 processing_version, error_message, status)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def is_file_processed(self, file_path: str) -> bool:
        """
        파일이 이미 처리되었는지 확인
        
        Args:
            file_path: 확인할 파일 경로
        
        Returns:
            bool: 처리 여부
        """
        query = "SELECT COUNT(*) as count FROM processed_files WHERE file_path = ?"
        result = self.execute_query(query, (file_path,))
        return result[0]['count'] > 0 if result else False
    
    def get_file_processing_status(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        파일의 처리 상태 조회
        
        Args:
            file_path: 파일 경로
        
        Returns:
            Optional[Dict]: 처리 상태 정보 또는 None
        """
        query = """
            SELECT file_path, file_hash, data_type, processing_status, 
                   record_count, processing_version, error_message, 
                   processed_at, created_at, updated_at
            FROM processed_files 
            WHERE file_path = ?
        """
        result = self.execute_query(query, (file_path,))
        return result[0] if result else None
    
    def get_unprocessed_files(self, data_type: str, base_path: str) -> List[str]:
        """
        특정 데이터 유형의 미처리 파일 목록 조회
        
        Args:
            data_type: 데이터 유형
            base_path: 검색할 기본 경로
        
        Returns:
            List[str]: 미처리 파일 경로 목록
        """
        # 실제 파일 시스템에서 파일 목록을 가져와야 하므로
        # 이 메서드는 외부에서 파일 목록을 받아 처리하는 방식으로 구현
        pass
    
    def get_processed_files_by_type(self, data_type: str, 
                                   status: str = "completed") -> List[Dict[str, Any]]:
        """
        특정 데이터 유형의 처리된 파일 목록 조회
        
        Args:
            data_type: 데이터 유형
            status: 처리 상태 (기본값: completed)
        
        Returns:
            List[Dict]: 처리된 파일 목록
        """
        query = """
            SELECT file_path, file_hash, data_type, processing_status, 
                   record_count, processing_version, error_message, 
                   processed_at, created_at, updated_at
            FROM processed_files 
            WHERE data_type = ? AND processing_status = ?
            ORDER BY processed_at DESC
        """
        return self.execute_query(query, (data_type, status))
    
    def get_processing_statistics(self, data_type: str = None) -> Dict[str, Any]:
        """
        처리 통계 조회
        
        Args:
            data_type: 데이터 유형 (None이면 전체)
        
        Returns:
            Dict: 처리 통계
        """
        if data_type:
            query = """
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed_files,
                    SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed_files,
                    SUM(record_count) as total_records,
                    AVG(record_count) as avg_records_per_file
                FROM processed_files 
                WHERE data_type = ?
            """
            result = self.execute_query(query, (data_type,))
        else:
            query = """
                SELECT 
                    COUNT(*) as total_files,
                    SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed_files,
                    SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed_files,
                    SUM(record_count) as total_records,
                    AVG(record_count) as avg_records_per_file
                FROM processed_files
            """
            result = self.execute_query(query)
        
        return result[0] if result else {}
    
    def update_file_processing_status(self, file_path: str, 
                                    status: str, error_message: str = None) -> bool:
        """
        파일 처리 상태 업데이트
        
        Args:
            file_path: 파일 경로
            status: 새로운 상태
            error_message: 에러 메시지 (실패한 경우)
        
        Returns:
            bool: 업데이트 성공 여부
        """
        query = """
            UPDATE processed_files 
            SET processing_status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
            WHERE file_path = ?
        """
        params = (status, error_message, file_path)
        
        try:
            rows_affected = self.execute_update(query, params)
            return rows_affected > 0
        except Exception as e:
            logger.error(f"Error updating file processing status: {e}")
            return False