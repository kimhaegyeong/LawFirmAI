"""
FAISS 버전 간 점진적 마이그레이션 관리

대량의 문서를 재처리할 때 사용하는 안전한 마이그레이션 방법을 제공합니다.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FAISSMigrationManager:
    """버전 간 점진적 마이그레이션 관리 클래스"""
    
    def __init__(self, faiss_version_manager, embedding_version_manager, db_path: str):
        """
        초기화
        
        Args:
            faiss_version_manager: FAISSVersionManager 인스턴스
            embedding_version_manager: EmbeddingVersionManager 인스턴스
            db_path: 데이터베이스 경로
        """
        self.faiss_version_manager = faiss_version_manager
        self.embedding_version_manager = embedding_version_manager
        self.db_path = db_path
        self.migration_log = {}
        self.migration_log_path = Path(db_path).parent / "migration_log.json"
        self._load_migration_log()
    
    def _load_migration_log(self):
        """마이그레이션 로그 로드"""
        if self.migration_log_path.exists():
            try:
                with open(self.migration_log_path, 'r', encoding='utf-8') as f:
                    self.migration_log = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load migration log: {e}")
                self.migration_log = {}
        else:
            self.migration_log = {}
    
    def _save_migration_log(self):
        """마이그레이션 로그 저장"""
        try:
            with open(self.migration_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.migration_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save migration log: {e}")
    
    def get_original_document(self, source_type: str, source_id: int) -> Optional[Dict[str, Any]]:
        """
        원본 문서 가져오기
        
        Args:
            source_type: 소스 타입
            source_id: 소스 ID
        
        Returns:
            Optional[Dict]: 원본 문서 데이터
        """
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            
            if source_type == "statute_article":
                cursor.execute("""
                    SELECT id, statute_id, article_number, content
                    FROM statute_articles
                    WHERE id = ?
                """, (source_id,))
            elif source_type == "case_paragraph":
                cursor.execute("""
                    SELECT cp.id, cp.case_id, cp.paragraph_index, cp.content,
                           c.case_number, c.case_name
                    FROM case_paragraphs cp
                    JOIN cases c ON cp.case_id = c.id
                    WHERE cp.id = ?
                """, (source_id,))
            elif source_type == "decision_paragraph":
                cursor.execute("""
                    SELECT dp.id, dp.case_id, dp.paragraph_index, dp.content,
                           c.case_number, c.case_name
                    FROM decision_paragraphs dp
                    JOIN cases c ON dp.case_id = c.id
                    WHERE dp.id = ?
                """, (source_id,))
            elif source_type == "interpretation_paragraph":
                cursor.execute("""
                    SELECT ip.id, ip.case_id, ip.paragraph_index, ip.content,
                           c.case_number, c.case_name
                    FROM interpretation_paragraphs ip
                    JOIN cases c ON ip.case_id = c.id
                    WHERE ip.id = ?
                """, (source_id,))
            else:
                logger.warning(f"Unknown source_type: {source_type}")
                return None
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get original document: {e}")
            return None
        finally:
            conn.close()
    
    async def migrate_documents(
        self,
        source_version: str,
        target_version: str,
        document_ids: List[tuple],  # [(source_type, source_id), ...]
        rechunk_fn: Callable,
        reembed_fn: Callable,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        문서들을 새 버전으로 마이그레이션
        
        Args:
            source_version: 원본 FAISS 버전 이름
            target_version: 대상 FAISS 버전 이름
            document_ids: 마이그레이션할 문서 ID 리스트 [(source_type, source_id), ...]
            rechunk_fn: 재청킹 함수
            reembed_fn: 재임베딩 함수
            batch_size: 배치 크기
        
        Returns:
            Dict: 마이그레이션 결과 통계
        """
        success_count = 0
        failed_count = 0
        failed_docs = []
        
        total = len(document_ids)
        logger.info(f"Starting migration: {source_version} -> {target_version} ({total} documents)")
        
        for i in range(0, total, batch_size):
            batch = document_ids[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
            
            for source_type, source_id in batch:
                doc_key = f"{source_type}_{source_id}"
                
                try:
                    if doc_key in self.migration_log and self.migration_log[doc_key].get('status') == 'success':
                        logger.debug(f"Skipping already migrated document: {doc_key}")
                        success_count += 1
                        continue
                    
                    original_doc = self.get_original_document(source_type, source_id)
                    if not original_doc:
                        logger.warning(f"Original document not found: {doc_key}")
                        self.migration_log[doc_key] = {"status": "failed", "error": "Document not found"}
                        failed_count += 1
                        failed_docs.append((source_type, source_id))
                        continue
                    
                    new_chunks = rechunk_fn(original_doc)
                    if not new_chunks:
                        logger.warning(f"No chunks generated for: {doc_key}")
                        self.migration_log[doc_key] = {"status": "failed", "error": "No chunks generated"}
                        failed_count += 1
                        failed_docs.append((source_type, source_id))
                        continue
                    
                    new_embeddings = reembed_fn(new_chunks)
                    if not new_embeddings or len(new_embeddings) != len(new_chunks):
                        logger.warning(f"Embedding mismatch for: {doc_key}")
                        self.migration_log[doc_key] = {"status": "failed", "error": "Embedding mismatch"}
                        failed_count += 1
                        failed_docs.append((source_type, source_id))
                        continue
                    
                    self.migration_log[doc_key] = {
                        "status": "success",
                        "source_version": source_version,
                        "target_version": target_version,
                        "chunk_count": len(new_chunks)
                    }
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Migration failed for {doc_key}: {e}")
                    self.migration_log[doc_key] = {"status": "failed", "error": str(e)}
                    failed_count += 1
                    failed_docs.append((source_type, source_id))
            
            self._save_migration_log()
        
        result = {
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "failed_docs": failed_docs,
            "source_version": source_version,
            "target_version": target_version
        }
        
        logger.info(f"Migration completed: {success_count} success, {failed_count} failed")
        return result
    
    def rollback_migration(self, target_version: str) -> List[tuple]:
        """
        마이그레이션 롤백
        
        Args:
            target_version: 롤백할 대상 버전
        
        Returns:
            List[tuple]: 롤백된 문서 리스트 [(source_type, source_id), ...]
        """
        failed_docs = [
            tuple(doc_key.split('_', 1))
            for doc_key, status in self.migration_log.items()
            if status.get('status') != 'success' and status.get('target_version') == target_version
        ]
        
        if failed_docs:
            logger.info(f"Found {len(failed_docs)} failed documents for rollback")
        
        return failed_docs
    
    def get_migration_status(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        마이그레이션 진행 상태 조회
        
        Args:
            target_version: 대상 버전 필터 (None이면 전체)
        
        Returns:
            Dict: 마이그레이션 상태 통계
        """
        if target_version:
            filtered_log = {
                k: v for k, v in self.migration_log.items()
                if v.get('target_version') == target_version
            }
        else:
            filtered_log = self.migration_log
        
        total = len(filtered_log)
        success = sum(1 for v in filtered_log.values() if v.get('status') == 'success')
        failed = total - success
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0.0,
            "target_version": target_version
        }
    
    def clear_migration_log(self, target_version: Optional[str] = None):
        """
        마이그레이션 로그 정리
        
        Args:
            target_version: 정리할 대상 버전 (None이면 전체)
        """
        if target_version:
            self.migration_log = {
                k: v for k, v in self.migration_log.items()
                if v.get('target_version') != target_version
            }
        else:
            self.migration_log = {}
        
        self._save_migration_log()
        logger.info(f"Cleared migration log for version: {target_version or 'all'}")

