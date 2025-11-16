"""
원본 문서 조회 서비스

청크에서 원본 문서를 조회하는 기능 제공
"""
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
import os


class OriginalDocumentService:
    """원본 문서 조회 서비스"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        초기화
        
        Args:
            db_path: 데이터베이스 경로 (None이면 기본 경로 사용)
        """
        if db_path is None:
            # 기본 데이터베이스 경로
            base_dir = Path(__file__).parent.parent.parent
            db_path = base_dir / "data" / "lawfirm_v2.db"
        
        self.db_path = str(db_path)
    
    def _get_connection(self) -> sqlite3.Connection:
        """데이터베이스 연결 반환"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_original_document(
        self,
        source_type: str,
        source_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        원본 문서 조회
        
        Args:
            source_type: 문서 타입 (statute_article, case_paragraph, etc.)
            source_id: 문서 ID
        
        Returns:
            원본 문서 정보 딕셔너리 또는 None
        """
        conn = self._get_connection()
        try:
            if source_type == "statute_article":
                cursor = conn.execute(
                    """
                    SELECT id, statute_name, article_no, clause_no, item_no, text
                    FROM statute_articles
                    WHERE id = ?
                    """,
                    (source_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": f"{row['statute_name']} 제{row['article_no']}조",
                        "text": row["text"],
                        "article_no": row["article_no"],
                        "clause_no": row["clause_no"],
                        "item_no": row["item_no"],
                        "source_type": source_type
                    }
            
            elif source_type == "case_paragraph":
                cursor = conn.execute(
                    """
                    SELECT cp.id, cp.text, c.court, c.casenames, c.doc_id
                    FROM case_paragraphs cp
                    JOIN cases c ON cp.case_id = c.id
                    WHERE cp.id = ?
                    """,
                    (source_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": f"{row['court']} {row['casenames']}",
                        "text": row["text"],
                        "doc_id": row["doc_id"],
                        "source_type": source_type
                    }
            
            elif source_type == "decision_paragraph":
                cursor = conn.execute(
                    """
                    SELECT dp.id, dp.text, d.court, d.casenames, d.doc_id
                    FROM decision_paragraphs dp
                    JOIN decisions d ON dp.decision_id = d.id
                    WHERE dp.id = ?
                    """,
                    (source_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": f"{row['court']} {row['casenames']}",
                        "text": row["text"],
                        "doc_id": row["doc_id"],
                        "source_type": source_type
                    }
            
            elif source_type == "interpretation_paragraph":
                cursor = conn.execute(
                    """
                    SELECT ip.id, ip.text, i.title, i.doc_id
                    FROM interpretation_paragraphs ip
                    JOIN interpretations i ON ip.interpretation_id = i.id
                    WHERE ip.id = ?
                    """,
                    (source_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": row["title"],
                        "text": row["text"],
                        "doc_id": row["doc_id"],
                        "source_type": source_type
                    }
            
            return None
        
        finally:
            conn.close()
    
    def get_chunks_by_group(
        self,
        chunk_group_id: str
    ) -> List[Dict[str, Any]]:
        """
        청크 그룹 ID로 관련 청크 조회
        
        Args:
            chunk_group_id: 청크 그룹 ID
        
        Returns:
            청크 리스트
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT 
                    id, source_type, source_id, chunk_index,
                    text, chunk_size_category, chunking_strategy
                FROM text_chunks
                WHERE chunk_group_id = ?
                ORDER BY chunk_size_category, chunk_index
                """,
                (chunk_group_id,)
            )
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "chunk_size_category": row["chunk_size_category"],
                    "chunking_strategy": row["chunking_strategy"]
                })
            
            return chunks
        
        finally:
            conn.close()
    
    def get_chunk_info(
        self,
        chunk_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        청크 정보 조회
        
        Args:
            chunk_id: 청크 ID
        
        Returns:
            청크 정보 딕셔너리 또는 None
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT 
                    id, source_type, source_id, chunk_index,
                    text, chunk_size_category, chunk_group_id,
                    chunking_strategy, query_type, original_document_id
                FROM text_chunks
                WHERE id = ?
                """,
                (chunk_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "chunk_size_category": row["chunk_size_category"],
                    "chunk_group_id": row["chunk_group_id"],
                    "chunking_strategy": row["chunking_strategy"],
                    "query_type": row["query_type"],
                    "original_document_id": row["original_document_id"]
                }
            
            return None
        
        finally:
            conn.close()

