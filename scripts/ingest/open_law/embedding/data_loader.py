# -*- coding: utf-8 -*-
"""
PostgreSQL 데이터 로더
법령 조문 및 판례 청크 데이터를 PostgreSQL에서 로드
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# logger는 직접 사용 (lawfirm_langgraph 패키지 import 방지)
# lawfirm_langgraph 패키지 import 시 LegalDataConnectorV2가 초기화되면서 SQLite URL 오류 발생
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostgreSQLDataLoader:
    """PostgreSQL 데이터 로더"""
    
    def __init__(self, db_url: str):
        """
        데이터 로더 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
        """
        self.db_url = db_url
        self.logger = logging.getLogger(__name__)
        
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
    
    def load_statute_articles(
        self,
        domain: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        skip_embedded: bool = False
    ) -> List[Dict[str, Any]]:
        """
        법령 조문 로드
        
        Args:
            domain: 도메인 필터 (civil_law, criminal_law, administrative_law)
            limit: 최대 개수
            offset: 시작 위치
            skip_embedded: 이미 임베딩된 데이터 건너뛰기 (statute_embeddings 테이블 확인)
        
        Returns:
            법령 조문 리스트
        """
        query = """
            SELECT 
                sa.id,
                sa.statute_id,
                sa.article_no,
                sa.article_title,
                sa.article_content,
                sa.clause_no,
                sa.clause_content,
                sa.item_no,
                sa.item_content,
                sa.sub_item_no,
                sa.sub_item_content,
                s.law_name_kr,
                s.law_abbrv,
                s.domain,
                s.effective_date
            FROM statutes_articles sa
            JOIN statutes s ON sa.statute_id = s.id
            WHERE 1=1
        """
        
        params = {"limit": limit, "offset": offset}
        
        if domain:
            query += " AND s.domain = :domain"
            params["domain"] = domain
        
        if skip_embedded:
            query += """
                AND NOT EXISTS (
                    SELECT 1 FROM statute_embeddings se
                    WHERE se.article_id = sa.id
                )
            """
        
        query += " ORDER BY sa.id LIMIT :limit OFFSET :offset"
        
        results = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            for row in result:
                results.append({
                    "id": row.id,
                    "statute_id": row.statute_id,
                    "article_no": row.article_no,
                    "article_title": row.article_title,
                    "article_content": row.article_content,
                    "clause_no": row.clause_no,
                    "clause_content": row.clause_content,
                    "item_no": row.item_no,
                    "item_content": row.item_content,
                    "sub_item_no": row.sub_item_no,
                    "sub_item_content": row.sub_item_content,
                    "law_name_kr": row.law_name_kr,
                    "law_abbrv": row.law_abbrv,
                    "domain": row.domain,
                    "effective_date": str(row.effective_date) if row.effective_date else None
                })
        
        return results
    
    def load_precedent_chunks(
        self,
        domain: Optional[str] = None,
        section_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        skip_embedded: bool = False
    ) -> List[Dict[str, Any]]:
        """
        판례 청크 로드
        
        Args:
            domain: 도메인 필터 (civil_law, criminal_law, administrative_law)
            section_type: 섹션 타입 필터 (판시사항, 판결요지, 판례내용)
            limit: 최대 개수
            offset: 시작 위치
            skip_embedded: 이미 임베딩된 데이터 건너뛰기 (embedding_vector가 NULL이 아닌 경우)
        
        Returns:
            판례 청크 리스트
        """
        query = """
            SELECT 
                pc.id,
                pc.precedent_content_id,
                pc.chunk_index,
                pc.chunk_content,
                pc.chunk_length,
                pc.metadata,
                pcon.section_type,
                pcon.referenced_articles,
                pcon.referenced_precedents,
                p.precedent_id as precedent_original_id,
                p.case_name,
                p.case_number,
                p.decision_date,
                p.court_name,
                p.domain
            FROM precedent_chunks pc
            JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id
            JOIN precedents p ON pcon.precedent_id = p.id
            WHERE 1=1
        """
        
        params = {"limit": limit, "offset": offset}
        
        if domain:
            query += " AND p.domain = :domain"
            params["domain"] = domain
        
        if section_type:
            query += " AND pcon.section_type = :section_type"
            params["section_type"] = section_type
        
        if skip_embedded:
            query += " AND pc.embedding_vector IS NULL"
        
        query += " ORDER BY pc.id LIMIT :limit OFFSET :offset"
        
        results = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            for row in result:
                import json
                # PostgreSQL JSONB는 SQLAlchemy가 dict로 변환할 수 있음
                if row.metadata:
                    if isinstance(row.metadata, dict):
                        metadata = row.metadata
                    else:
                        metadata = json.loads(row.metadata)
                else:
                    metadata = {}
                
                results.append({
                    "id": row.id,
                    "precedent_content_id": row.precedent_content_id,
                    "chunk_index": row.chunk_index,
                    "chunk_content": row.chunk_content,
                    "chunk_length": row.chunk_length,
                    "metadata": metadata,
                    "section_type": row.section_type,
                    "referenced_articles": row.referenced_articles,
                    "referenced_precedents": row.referenced_precedents,
                    "precedent_id": row.precedent_original_id,
                    "case_name": row.case_name,
                    "case_number": row.case_number,
                    "decision_date": str(row.decision_date) if row.decision_date else None,
                    "court_name": row.court_name,
                    "domain": row.domain
                })
        
        return results
    
    def get_statute_articles_count(
        self,
        domain: Optional[str] = None,
        skip_embedded: bool = False
    ) -> int:
        """법령 조문 개수 조회"""
        query = """
            SELECT COUNT(*)
            FROM statutes_articles sa
            JOIN statutes s ON sa.statute_id = s.id
            WHERE 1=1
        """
        
        params = {}
        
        if domain:
            query += " AND s.domain = :domain"
            params["domain"] = domain
        
        if skip_embedded:
            query += """
                AND NOT EXISTS (
                    SELECT 1 FROM statute_embeddings se
                    WHERE se.article_id = sa.id
                )
            """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return result.scalar()
    
    def get_precedent_chunks_count(
        self,
        domain: Optional[str] = None,
        section_type: Optional[str] = None,
        skip_embedded: bool = False
    ) -> int:
        """판례 청크 개수 조회"""
        query = """
            SELECT COUNT(*)
            FROM precedent_chunks pc
            JOIN precedent_contents pcon ON pc.precedent_content_id = pcon.id
            JOIN precedents p ON pcon.precedent_id = p.id
            WHERE 1=1
        """
        
        params = {}
        
        if domain:
            query += " AND p.domain = :domain"
            params["domain"] = domain
        
        if section_type:
            query += " AND pcon.section_type = :section_type"
            params["section_type"] = section_type
        
        if skip_embedded:
            query += " AND pc.embedding_vector IS NULL"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            return result.scalar()

