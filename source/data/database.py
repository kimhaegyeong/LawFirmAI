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
from dataclasses import dataclass

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
        
        # FTS5 확장 활성화 (Windows 호환)
        try:
            # Windows에서는 FTS5가 기본적으로 포함되어 있음
            conn.execute("SELECT fts5(1)")
        except Exception as e:
            logger.warning(f"FTS5 not available: {e}")
            # FTS5가 없어도 계속 진행
        
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
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # 판례 메타데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedent_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    precedent_number TEXT,
                    case_name TEXT,
                    court_name TEXT,
                    decision_date TEXT,
                    case_type TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            
            # 헌재결정례 상세 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS constitutional_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER UNIQUE NOT NULL,
                    decision_name TEXT NOT NULL,
                    case_number TEXT,
                    case_type TEXT,
                    case_type_code INTEGER,
                    court_division_code INTEGER,
                    decision_date TEXT,
                    final_date TEXT,
                    summary TEXT,
                    decision_gist TEXT,
                    full_text TEXT,
                    reference_articles TEXT,
                    reference_precedents TEXT,
                    target_articles TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 헌재결정례 FTS 테이블 (전문 검색용)
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS constitutional_decisions_fts USING fts5(
                    decision_name,
                    summary,
                    decision_gist,
                    full_text,
                    content='constitutional_decisions',
                    content_rowid='id'
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
            
            # 판례 케이스 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedent_cases (
                    case_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    case_name TEXT NOT NULL,
                    case_number TEXT NOT NULL,
                    decision_date TEXT,
                    field TEXT,
                    court TEXT,
                    detail_url TEXT,
                    full_text TEXT,
                    searchable_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 법적 근거 검증 로그 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_basis_validation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    validation_result TEXT NOT NULL,
                    confidence_score REAL,
                    validation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    citations_found INTEGER DEFAULT 0,
                    valid_citations INTEGER DEFAULT 0
                )
            """)
            
            # 법적 근거 처리 로그 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_basis_processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    question_type TEXT,
                    confidence_score REAL,
                    is_legally_sound BOOLEAN,
                    citations_count INTEGER DEFAULT 0,
                    processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 판례 섹션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedent_sections (
                    section_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    section_type TEXT NOT NULL,
                    section_type_korean TEXT,
                    section_content TEXT NOT NULL,
                    section_length INTEGER DEFAULT 0,
                    has_content BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES precedent_cases(case_id)
                )
            """)
            
            # 판례 당사자 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS precedent_parties (
                    party_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    party_type TEXT NOT NULL,
                    party_type_korean TEXT,
                    party_content TEXT,
                    party_length INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES precedent_cases(case_id)
                )
            """)
            
            # Assembly 법률 테이블 (기존 + 새로운 품질 관리 컬럼)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assembly_laws (
                    law_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    law_name TEXT NOT NULL,
                    law_type TEXT,
                    category TEXT,
                    row_number TEXT,
                    promulgation_number TEXT,
                    promulgation_date TEXT,
                    enforcement_date TEXT,
                    amendment_type TEXT,
                    ministry TEXT,
                    parent_law TEXT,
                    related_laws TEXT,
                    full_text TEXT,
                    searchable_text TEXT,
                    keywords TEXT,
                    summary TEXT,
                    html_clean_text TEXT,
                    main_article_count INTEGER DEFAULT 0,
                    supplementary_article_count INTEGER DEFAULT 0,
                    ml_enhanced BOOLEAN DEFAULT FALSE,
                    parsing_quality_score REAL DEFAULT 0.0,
                    processing_version TEXT DEFAULT '1.0',
                    
                    -- 새로운 품질 관리 컬럼들
                    law_name_hash TEXT UNIQUE,
                    content_hash TEXT UNIQUE,
                    quality_score REAL DEFAULT 0.0,
                    duplicate_group_id TEXT,
                    is_primary_version BOOLEAN DEFAULT TRUE,
                    version_number INTEGER DEFAULT 1,
                    parsing_method TEXT DEFAULT 'legacy',
                    auto_corrected BOOLEAN DEFAULT FALSE,
                    manual_review_required BOOLEAN DEFAULT FALSE,
                    migration_timestamp TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Assembly 조문 테이블 (기존)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS assembly_articles (
                    article_id TEXT PRIMARY KEY,
                    law_id TEXT NOT NULL,
                    article_number INTEGER NOT NULL,
                    article_title TEXT,
                    article_content TEXT NOT NULL,
                    is_supplementary BOOLEAN DEFAULT FALSE,
                    ml_confidence_score REAL DEFAULT 0.0,
                    parsing_method TEXT DEFAULT 'rule_based',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (law_id) REFERENCES assembly_laws(law_id)
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
            
            # 중복 그룹 테이블 (새로운)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicate_groups (
                    group_id TEXT PRIMARY KEY,
                    group_type TEXT NOT NULL,
                    primary_law_id TEXT NOT NULL,
                    duplicate_law_ids TEXT NOT NULL,
                    resolution_strategy TEXT NOT NULL,
                    confidence_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (primary_law_id) REFERENCES assembly_laws (law_id)
                )
            """)
            
            # 품질 보고서 테이블 (새로운)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id TEXT PRIMARY KEY,
                    law_id TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    article_count_score REAL NOT NULL,
                    title_extraction_score REAL NOT NULL,
                    article_sequence_score REAL NOT NULL,
                    structure_completeness_score REAL NOT NULL,
                    issues TEXT,
                    suggestions TEXT,
                    validation_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (law_id) REFERENCES assembly_laws (law_id)
                )
            """)
            
            # 마이그레이션 히스토리 테이블 (새로운)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    migration_id TEXT PRIMARY KEY,
                    migration_version TEXT NOT NULL,
                    migration_timestamp TIMESTAMP NOT NULL,
                    description TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    records_affected INTEGER DEFAULT 0
                )
            """)
            
            # 스키마 버전 테이블 (새로운)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            
            # FTS5 전체 텍스트 검색 테이블
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_precedent_cases USING fts5(
                    case_id,
                    case_name,
                    case_number,
                    full_text,
                    searchable_text,
                    content='precedent_cases',
                    content_rowid='rowid'
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_precedent_sections USING fts5(
                    section_id,
                    case_id,
                    section_content,
                    content='precedent_sections',
                    content_rowid='rowid'
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_assembly_laws USING fts5(
                    law_id,
                    law_name,
                    full_text,
                    searchable_text,
                    content='assembly_laws',
                    content_rowid='rowid'
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_assembly_articles USING fts5(
                    article_id,
                    law_id,
                    article_title,
                    article_content,
                    content='assembly_articles',
                    content_rowid='rowid'
                )
            """)
            
            # 인덱스 생성 (검색 성능 향상)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_law_metadata_name ON law_metadata(law_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_metadata_court ON precedent_metadata(court_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_metadata_date ON precedent_metadata(decision_date)")
            
            # 판례 테이블 인덱스
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_cases_category ON precedent_cases(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_cases_date ON precedent_cases(decision_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_cases_court ON precedent_cases(court)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_sections_case_id ON precedent_sections(case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_sections_type ON precedent_sections(section_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_parties_case_id ON precedent_parties(case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_precedent_parties_type ON precedent_parties(party_type)")
            
            # Assembly 테이블 인덱스
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_source ON assembly_laws(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_category ON assembly_laws(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_ministry ON assembly_laws(ministry)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_created_at ON assembly_laws(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_articles_law_id ON assembly_articles(law_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_articles_number ON assembly_articles(article_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_articles_supplementary ON assembly_articles(is_supplementary)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_path ON processed_files(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_type ON processed_files(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_files_status ON processed_files(processing_status)")
            
            # 새로운 품질 관리 인덱스들
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_law_name_hash ON assembly_laws(law_name_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_content_hash ON assembly_laws(content_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_quality_score ON assembly_laws(quality_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_duplicate_group_id ON assembly_laws(duplicate_group_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_is_primary_version ON assembly_laws(is_primary_version)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_parsing_method ON assembly_laws(parsing_method)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_assembly_laws_manual_review_required ON assembly_laws(manual_review_required)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_groups_group_type ON duplicate_groups(group_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_groups_primary_law_id ON duplicate_groups(primary_law_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_groups_confidence_score ON duplicate_groups(confidence_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_reports_law_id ON quality_reports(law_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_reports_overall_score ON quality_reports(overall_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_reports_validation_timestamp ON quality_reports(validation_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_migration_history_version ON migration_history(migration_version)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_migration_history_success ON migration_history(success)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_schema_version_updated_at ON schema_version(updated_at)")
            
            conn.commit()
            logger.info("Database tables and indices created successfully")
    
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
    
    # 새로운 품질 관리 메서드들
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        품질 통계 조회
        
        Returns:
            Dict: 품질 통계
        """
        query = """
            SELECT 
                COUNT(*) as total_laws,
                AVG(quality_score) as avg_quality_score,
                MIN(quality_score) as min_quality_score,
                MAX(quality_score) as max_quality_score,
                SUM(CASE WHEN quality_score >= 0.8 THEN 1 ELSE 0 END) as high_quality_count,
                SUM(CASE WHEN quality_score < 0.6 THEN 1 ELSE 0 END) as low_quality_count,
                SUM(CASE WHEN manual_review_required = TRUE THEN 1 ELSE 0 END) as manual_review_count,
                SUM(CASE WHEN auto_corrected = TRUE THEN 1 ELSE 0 END) as auto_corrected_count
            FROM assembly_laws
        """
        result = self.execute_query(query)
        return result[0] if result else {}
    
    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """
        중복 통계 조회
        
        Returns:
            Dict: 중복 통계
        """
        query = """
            SELECT 
                COUNT(*) as total_duplicate_groups,
                AVG(confidence_score) as avg_confidence_score,
                SUM(CASE WHEN group_type = 'file' THEN 1 ELSE 0 END) as file_duplicates,
                SUM(CASE WHEN group_type = 'content' THEN 1 ELSE 0 END) as content_duplicates,
                SUM(CASE WHEN group_type = 'semantic' THEN 1 ELSE 0 END) as semantic_duplicates
            FROM duplicate_groups
        """
        result = self.execute_query(query)
        return result[0] if result else {}
    
    def get_laws_by_quality_score(self, min_score: float = 0.0, max_score: float = 1.0, 
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        품질 점수 범위로 법률 조회
        
        Args:
            min_score: 최소 품질 점수
            max_score: 최대 품질 점수
            limit: 조회 제한 수
        
        Returns:
            List[Dict]: 법률 목록
        """
        query = """
            SELECT law_id, law_name, quality_score, parsing_method, 
                   auto_corrected, manual_review_required, created_at
            FROM assembly_laws 
            WHERE quality_score BETWEEN ? AND ?
            ORDER BY quality_score DESC
            LIMIT ?
        """
        return self.execute_query(query, (min_score, max_score, limit))
    
    def get_laws_requiring_review(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        수동 검토가 필요한 법률 조회
        
        Args:
            limit: 조회 제한 수
        
        Returns:
            List[Dict]: 법률 목록
        """
        query = """
            SELECT law_id, law_name, quality_score, parsing_method, 
                   auto_corrected, created_at
            FROM assembly_laws 
            WHERE manual_review_required = TRUE
            ORDER BY quality_score ASC, created_at DESC
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
    def get_duplicate_groups(self, group_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        중복 그룹 조회
        
        Args:
            group_type: 그룹 유형 (None이면 전체)
            limit: 조회 제한 수
        
        Returns:
            List[Dict]: 중복 그룹 목록
        """
        if group_type:
            query = """
                SELECT group_id, group_type, primary_law_id, duplicate_law_ids,
                       resolution_strategy, confidence_score, created_at
                FROM duplicate_groups 
                WHERE group_type = ?
                ORDER BY confidence_score DESC, created_at DESC
                LIMIT ?
            """
            return self.execute_query(query, (group_type, limit))
        else:
            query = """
                SELECT group_id, group_type, primary_law_id, duplicate_law_ids,
                       resolution_strategy, confidence_score, created_at
                FROM duplicate_groups 
                ORDER BY confidence_score DESC, created_at DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,))
    
    def get_quality_reports(self, law_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        품질 보고서 조회
        
        Args:
            law_id: 법률 ID (None이면 전체)
            limit: 조회 제한 수
        
        Returns:
            List[Dict]: 품질 보고서 목록
        """
        if law_id:
            query = """
                SELECT report_id, law_id, overall_score, article_count_score,
                       title_extraction_score, article_sequence_score,
                       structure_completeness_score, issues, suggestions, validation_timestamp
                FROM quality_reports 
                WHERE law_id = ?
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (law_id, limit))
        else:
            query = """
                SELECT report_id, law_id, overall_score, article_count_score,
                       title_extraction_score, article_sequence_score,
                       structure_completeness_score, issues, suggestions, validation_timestamp
                FROM quality_reports 
                ORDER BY validation_timestamp DESC
                LIMIT ?
            """
            return self.execute_query(query, (limit,))
    
    def update_law_quality_score(self, law_id: str, quality_score: float, 
                                parsing_method: str = None, auto_corrected: bool = False) -> int:
        """
        법률 품질 점수 업데이트
        
        Args:
            law_id: 법률 ID
            quality_score: 품질 점수
            parsing_method: 파싱 방법
            auto_corrected: 자동 수정 여부
        
        Returns:
            int: 업데이트된 행 수
        """
        query = """
            UPDATE assembly_laws 
            SET quality_score = ?, parsing_method = ?, auto_corrected = ?, updated_at = CURRENT_TIMESTAMP
            WHERE law_id = ?
        """
        params = [quality_score]
        if parsing_method:
            params.append(parsing_method)
        else:
            params.append(None)
        params.append(auto_corrected)
        params.append(law_id)
        
        return self.execute_update(query, tuple(params))
    
    def mark_law_for_review(self, law_id: str, requires_review: bool = True) -> int:
        """
        법률을 검토 대상으로 표시
        
        Args:
            law_id: 법률 ID
            requires_review: 검토 필요 여부
        
        Returns:
            int: 업데이트된 행 수
        """
        query = """
            UPDATE assembly_laws 
            SET manual_review_required = ?, updated_at = CURRENT_TIMESTAMP
            WHERE law_id = ?
        """
        return self.execute_update(query, (requires_review, law_id))
    
    def get_schema_version(self) -> str:
        """
        현재 스키마 버전 조회
        
        Returns:
            str: 스키마 버전
        """
        query = """
            SELECT version FROM schema_version 
            ORDER BY updated_at DESC 
            LIMIT 1
        """
        result = self.execute_query(query)
        return result[0]['version'] if result else "1.0"
    
    def get_migration_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        마이그레이션 히스토리 조회
        
        Args:
            limit: 조회 제한 수
        
        Returns:
            List[Dict]: 마이그레이션 히스토리
        """
        query = """
            SELECT migration_id, migration_version, migration_timestamp,
                   description, success, error_message, records_affected
            FROM migration_history 
            ORDER BY migration_timestamp DESC
            LIMIT ?
        """
        return self.execute_query(query, (limit,))
    
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
    
    # 헌재결정례 관련 메서드들
    
    def insert_constitutional_decision(self, decision_data: Dict[str, Any]) -> bool:
        """
        헌재결정례 데이터 삽입
        
        Args:
            decision_data: 헌재결정례 데이터
            
        Returns:
            bool: 삽입 성공 여부
        """
        query = """
            INSERT OR REPLACE INTO constitutional_decisions (
                decision_id, decision_name, case_number, case_type, case_type_code,
                court_division_code, decision_date, final_date, summary, decision_gist,
                full_text, reference_articles, reference_precedents, target_articles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            params = (
                decision_data.get('헌재결정례일련번호'),
                decision_data.get('사건명'),
                decision_data.get('사건번호'),
                decision_data.get('사건종류명'),
                decision_data.get('사건종류코드'),
                decision_data.get('재판부구분코드'),
                decision_data.get('종국일자'),
                decision_data.get('종국일자'),
                decision_data.get('판시사항'),
                decision_data.get('결정요지'),
                decision_data.get('전문'),
                decision_data.get('참조조문'),
                decision_data.get('참조판례'),
                decision_data.get('심판대상조문')
            )
            
            self.execute_update(query, params)
            logger.info(f"헌재결정례 삽입 성공: {decision_data.get('사건명', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"헌재결정례 삽입 실패: {e}")
            return False
    
    def insert_constitutional_decisions_batch(self, decisions: List[Dict[str, Any]]) -> int:
        """
        헌재결정례 배치 삽입
        
        Args:
            decisions: 헌재결정례 데이터 리스트
            
        Returns:
            int: 삽입된 행 수
        """
        query = """
            INSERT OR REPLACE INTO constitutional_decisions (
                decision_id, decision_name, case_number, case_type, case_type_code,
                court_division_code, decision_date, final_date, summary, decision_gist,
                full_text, reference_articles, reference_precedents, target_articles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                params_list = []
                for decision_data in decisions:
                    params = (
                        decision_data.get('헌재결정례일련번호'),
                        decision_data.get('사건명'),
                        decision_data.get('사건번호'),
                        decision_data.get('사건종류명'),
                        decision_data.get('사건종류코드'),
                        decision_data.get('재판부구분코드'),
                        decision_data.get('종국일자'),
                        decision_data.get('종국일자'),
                        decision_data.get('판시사항'),
                        decision_data.get('결정요지'),
                        decision_data.get('전문'),
                        decision_data.get('참조조문'),
                        decision_data.get('참조판례'),
                        decision_data.get('심판대상조문')
                    )
                    params_list.append(params)
                
                cursor.executemany(query, params_list)
                conn.commit()
                
                logger.info(f"헌재결정례 배치 삽입 성공: {len(decisions)}개")
                return len(decisions)
                
        except Exception as e:
            logger.error(f"헌재결정례 배치 삽입 실패: {e}")
            return 0
    
    def get_constitutional_decision_by_id(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """
        헌재결정례 ID로 조회
        
        Args:
            decision_id: 헌재결정례 ID
            
        Returns:
            Dict: 헌재결정례 데이터 또는 None
        """
        query = """
            SELECT * FROM constitutional_decisions 
            WHERE decision_id = ?
        """
        result = self.execute_query(query, (decision_id,))
        return result[0] if result else None
    
    def search_constitutional_decisions_fts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        헌재결정례 FTS 검색
        
        Args:
            query: 검색 쿼리
            limit: 결과 제한 수
            
        Returns:
            List[Dict]: 검색 결과
        """
        fts_query = """
            SELECT cd.*, 
                   rank as fts_rank
            FROM constitutional_decisions_fts fts
            JOIN constitutional_decisions cd ON fts.rowid = cd.id
            WHERE constitutional_decisions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        return self.execute_query(fts_query, (query, limit))
    
    def get_constitutional_decisions_by_date_range(self, 
                                                  start_date: str, 
                                                  end_date: str,
                                                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        날짜 범위로 헌재결정례 조회
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            limit: 결과 제한 수
            
        Returns:
            List[Dict]: 헌재결정례 목록
        """
        query = """
            SELECT * FROM constitutional_decisions 
            WHERE decision_date BETWEEN ? AND ?
            ORDER BY decision_date ASC
            LIMIT ?
        """
        return self.execute_query(query, (start_date, end_date, limit))
    
    def get_constitutional_decisions_by_keyword(self, 
                                              keyword: str, 
                                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        키워드로 헌재결정례 조회
        
        Args:
            keyword: 검색 키워드
            limit: 결과 제한 수
            
        Returns:
            List[Dict]: 헌재결정례 목록
        """
        query = """
            SELECT * FROM constitutional_decisions 
            WHERE decision_name LIKE ? OR summary LIKE ? OR decision_gist LIKE ?
            ORDER BY decision_date DESC
            LIMIT ?
        """
        keyword_pattern = f"%{keyword}%"
        return self.execute_query(query, (keyword_pattern, keyword_pattern, keyword_pattern, limit))
    
    def get_constitutional_decisions_count(self) -> int:
        """
        헌재결정례 총 개수 조회
        
        Returns:
            int: 총 개수
        """
        query = "SELECT COUNT(*) as count FROM constitutional_decisions"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def get_constitutional_decisions_stats(self) -> Dict[str, Any]:
        """
        헌재결정례 통계 조회
        
        Returns:
            Dict: 통계 정보
        """
        stats = {}
        
        # 총 개수
        stats['total_count'] = self.get_constitutional_decisions_count()
        
        # 연도별 통계
        year_query = """
            SELECT SUBSTR(decision_date, 1, 4) as year, COUNT(*) as count
            FROM constitutional_decisions 
            WHERE decision_date IS NOT NULL
            GROUP BY SUBSTR(decision_date, 1, 4)
            ORDER BY year DESC
        """
        stats['by_year'] = self.execute_query(year_query)
        
        # 사건종류별 통계
        type_query = """
            SELECT case_type, COUNT(*) as count
            FROM constitutional_decisions 
            WHERE case_type IS NOT NULL
            GROUP BY case_type
            ORDER BY count DESC
        """
        stats['by_type'] = self.execute_query(type_query)
        
        return stats