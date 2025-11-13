#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 임베딩 데이터베이스 마이그레이션 스크립트

버전별 FAISS 인덱스에서 벡터를 읽어 lawfirm_v2.db의 embeddings 테이블로 마이그레이션합니다.
"""

import logging
import json
import sys
import sqlite3
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Please install faiss-cpu or faiss-gpu")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

logger = logging.getLogger(__name__)


class VectorStoreMigrator:
    """벡터스토어 마이그레이션 클래스"""
    
    def __init__(self, 
                 vector_store_base_path: Path,
                 db_path: Path,
                 version: Optional[str] = None,
                 model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        마이그레이터 초기화
        
        Args:
            vector_store_base_path: 벡터스토어 기본 경로
            db_path: 데이터베이스 경로
            version: 버전 번호 (None이면 최신 버전)
            model_name: 모델명
        """
        self.vector_store_base_path = Path(vector_store_base_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.version_manager = VectorStoreVersionManager(self.vector_store_base_path)
        self.version = version or self.version_manager.get_current_version()
        
        if self.version:
            self.version_path = self.version_manager.get_version_path(self.version)
        else:
            self.version_path = self.vector_store_base_path
        
        self.index_path = self.version_path / "ml_enhanced_faiss_index.faiss"
        self.metadata_path = self.version_path / "ml_enhanced_faiss_index.json"
        
        self.stats = {
            'total_vectors': 0,
            'migrated_chunks': 0,
            'migrated_embeddings': 0,
            'skipped_duplicates': 0,
            'failed_migrations': 0,
            'errors': []
        }
    
    def backup_database(self) -> Optional[Path]:
        """데이터베이스 백업"""
        if not self.db_path.exists():
            logger.warning(f"Database not found: {self.db_path}. Skipping backup.")
            return None
        
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_{backup_suffix}.db"
        
        logger.info(f"Backing up database from {self.db_path} to {backup_path}")
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Backup completed: {backup_path}")
        return backup_path
    
    def validate_migration(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """마이그레이션 검증"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            cursor = conn.execute(
                """SELECT COUNT(*) as count FROM embeddings WHERE model = ?""",
                (self.model_name,)
            )
            db_count = cursor.fetchone()['count']
            
            if db_count != self.stats['migrated_embeddings']:
                validation_results['warnings'].append(
                    f"Embedding count mismatch: DB has {db_count}, migrated {self.stats['migrated_embeddings']}"
                )
            
            cursor = conn.execute(
                """SELECT COUNT(*) as count FROM text_chunks""",
            )
            chunk_count = cursor.fetchone()['count']
            
            if chunk_count < self.stats['migrated_chunks']:
                validation_results['warnings'].append(
                    f"Chunk count mismatch: DB has {chunk_count}, migrated {self.stats['migrated_chunks']}"
                )
            
            cursor = conn.execute(
                """SELECT COUNT(*) as count 
                   FROM embeddings e
                   JOIN text_chunks tc ON e.chunk_id = tc.id
                   WHERE e.model = ? AND tc.source_type IS NOT NULL""",
                (self.model_name,)
            )
            valid_count = cursor.fetchone()['count']
            
            if valid_count < db_count:
                validation_results['errors'].append(
                    f"Some embeddings have invalid source_type: {db_count - valid_count} embeddings"
                )
                validation_results['valid'] = False
            
            logger.info(f"Validation completed: {valid_count} valid embeddings, {db_count} total")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['errors'].append(str(e))
            validation_results['valid'] = False
        
        return validation_results
    
    def rollback(self, backup_path: Path) -> bool:
        """롤백 실행"""
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            logger.info(f"Rolling back database from {backup_path} to {self.db_path}")
            shutil.copy2(backup_path, self.db_path)
            logger.info("Rollback completed successfully")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def load_faiss_index(self) -> Optional[Any]:
        """FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return None
        
        if not self.index_path.exists():
            logger.error(f"FAISS index not found: {self.index_path}")
            return None
        
        try:
            index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded: {index.ntotal} vectors")
            self.stats['total_vectors'] = index.ntotal
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """메타데이터 로드"""
        if not self.metadata_path.exists():
            logger.error(f"Metadata file not found: {self.metadata_path}")
            return []
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = json.load(f)
            
            if isinstance(metadata_content, dict):
                if 'documents' in metadata_content:
                    metadata = metadata_content['documents']
                elif 'document_metadata' in metadata_content and 'document_texts' in metadata_content:
                    # document_metadata와 document_texts를 결합
                    metadata_list = metadata_content['document_metadata']
                    texts_list = metadata_content['document_texts']
                    metadata = []
                    for meta, text in zip(metadata_list, texts_list):
                        combined = meta.copy()
                        combined['content'] = text
                        combined['text'] = text
                        metadata.append(combined)
                    logger.info(f"Combined {len(metadata)} metadata items with texts")
                else:
                    logger.warning("Metadata contains only configuration, no document data")
                    metadata = []
            else:
                metadata = metadata_content
            
            logger.info(f"Metadata loaded: {len(metadata)} items")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []
    
    def get_or_create_domain(self, conn: sqlite3.Connection, domain_name: str) -> int:
        """도메인 조회 또는 생성"""
        cursor = conn.execute("SELECT id FROM domains WHERE name = ?", (domain_name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute("INSERT INTO domains (name) VALUES (?)", (domain_name,))
        return cursor.lastrowid
    
    def find_or_create_statute(self, conn: sqlite3.Connection, metadata: Dict[str, Any], domain_id: int) -> Optional[int]:
        """법령 조회 또는 생성"""
        law_name = metadata.get('law_name') or metadata.get('statute_name', '')
        if not law_name:
            return None
        
        cursor = conn.execute(
            "SELECT id FROM statutes WHERE domain_id = ? AND name = ?",
            (domain_id, law_name)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            """INSERT INTO statutes (domain_id, name, abbrv, statute_type, category)
               VALUES (?, ?, ?, ?, ?)""",
            (
                domain_id,
                law_name,
                metadata.get('abbrv'),
                metadata.get('statute_type'),
                metadata.get('category')
            )
        )
        return cursor.lastrowid
    
    def find_or_create_statute_article(self, conn: sqlite3.Connection, 
                                       statute_id: int, 
                                       metadata: Dict[str, Any]) -> Optional[int]:
        """법령 조문 조회 또는 생성"""
        article_no = metadata.get('article_number') or metadata.get('article_no', '')
        if not article_no:
            return None
        
        cursor = conn.execute(
            """SELECT id FROM statute_articles 
               WHERE statute_id = ? AND article_no = ? AND clause_no = ? AND item_no = ?""",
            (
                statute_id,
                article_no,
                metadata.get('clause_number') or metadata.get('clause_no', ''),
                metadata.get('item_no', '')
            )
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            """INSERT INTO statute_articles (statute_id, article_no, clause_no, item_no, heading, text)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                statute_id,
                article_no,
                metadata.get('clause_number') or metadata.get('clause_no', ''),
                metadata.get('item_no', ''),
                metadata.get('article_title', ''),
                ''  # text는 나중에 업데이트
            )
        )
        return cursor.lastrowid
    
    def find_or_create_case(self, conn: sqlite3.Connection, metadata: Dict[str, Any], domain_id: int) -> Optional[int]:
        """판례 조회 또는 생성"""
        doc_id = metadata.get('doc_id') or metadata.get('case_number', '') or metadata.get('case_id', '')
        if not doc_id:
            return None
        
        cursor = conn.execute("SELECT id FROM cases WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            """INSERT INTO cases (domain_id, doc_id, court, case_type, casenames, announce_date)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                domain_id,
                doc_id,
                metadata.get('court', ''),
                metadata.get('type', ''),
                metadata.get('casenames') or metadata.get('case_name', ''),
                metadata.get('announce_date') or metadata.get('decision_date', '')
            )
        )
        return cursor.lastrowid
    
    def find_or_create_case_paragraph(self, conn: sqlite3.Connection, 
                                     case_id: int, 
                                     chunk_index: int,
                                     text: str) -> Optional[int]:
        """판례 문단 조회 또는 생성"""
        cursor = conn.execute(
            "SELECT id FROM case_paragraphs WHERE case_id = ? AND para_index = ?",
            (case_id, chunk_index)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            "INSERT INTO case_paragraphs (case_id, para_index, text) VALUES (?, ?, ?)",
            (case_id, chunk_index, text)
        )
        return cursor.lastrowid
    
    def find_or_create_decision(self, conn: sqlite3.Connection, metadata: Dict[str, Any], domain_id: int) -> Optional[int]:
        """결정례 조회 또는 생성"""
        doc_id = metadata.get('doc_id', '')
        if not doc_id:
            return None
        
        cursor = conn.execute("SELECT id FROM decisions WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            """INSERT INTO decisions (domain_id, doc_id, org, decision_date, result)
               VALUES (?, ?, ?, ?, ?)""",
            (
                domain_id,
                doc_id,
                metadata.get('org', ''),
                metadata.get('decision_date', ''),
                metadata.get('result')
            )
        )
        return cursor.lastrowid
    
    def find_or_create_decision_paragraph(self, conn: sqlite3.Connection,
                                         decision_id: int,
                                         chunk_index: int,
                                         text: str) -> Optional[int]:
        """결정례 문단 조회 또는 생성"""
        cursor = conn.execute(
            "SELECT id FROM decision_paragraphs WHERE decision_id = ? AND para_index = ?",
            (decision_id, chunk_index)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            "INSERT INTO decision_paragraphs (decision_id, para_index, text) VALUES (?, ?, ?)",
            (decision_id, chunk_index, text)
        )
        return cursor.lastrowid
    
    def find_or_create_interpretation(self, conn: sqlite3.Connection, metadata: Dict[str, Any], domain_id: int) -> Optional[int]:
        """해석례 조회 또는 생성"""
        doc_id = metadata.get('doc_id', '')
        if not doc_id:
            return None
        
        cursor = conn.execute("SELECT id FROM interpretations WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            """INSERT INTO interpretations (domain_id, doc_id, org, title, response_date)
               VALUES (?, ?, ?, ?, ?)""",
            (
                domain_id,
                doc_id,
                metadata.get('org', ''),
                metadata.get('title', ''),
                metadata.get('response_date', '')
            )
        )
        return cursor.lastrowid
    
    def find_or_create_interpretation_paragraph(self, conn: sqlite3.Connection,
                                                interpretation_id: int,
                                                chunk_index: int,
                                                text: str) -> Optional[int]:
        """해석례 문단 조회 또는 생성"""
        cursor = conn.execute(
            "SELECT id FROM interpretation_paragraphs WHERE interpretation_id = ? AND para_index = ?",
            (interpretation_id, chunk_index)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor = conn.execute(
            "INSERT INTO interpretation_paragraphs (interpretation_id, para_index, text) VALUES (?, ?, ?)",
            (interpretation_id, chunk_index, text)
        )
        return cursor.lastrowid
    
    def migrate(self, backup: bool = True, batch_size: int = 1000) -> Dict[str, Any]:
        """
        마이그레이션 실행
        
        Args:
            backup: 백업 여부
            batch_size: 배치 크기
        
        Returns:
            마이그레이션 통계
        """
        logger.info(f"Starting migration from {self.version_path} to {self.db_path}")
        
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Cannot proceed with migration.")
            return self.stats
        
        index = self.load_faiss_index()
        if index is None:
            return self.stats
        
        metadata_list = self.load_metadata()
        if not metadata_list:
            logger.error("No metadata found. Cannot proceed with migration.")
            return self.stats
        
        if len(metadata_list) != index.ntotal:
            logger.warning(
                f"Metadata count ({len(metadata_list)}) != index size ({index.ntotal}). "
                "Some vectors may not have metadata."
            )
        
        backup_path = None
        if backup:
            backup_path = self.backup_database()
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # 벡터 추출
            vectors = index.reconstruct_n(0, index.ntotal)
            dim = vectors.shape[1] if len(vectors.shape) > 1 else vectors.shape[0]
            
            logger.info(f"Extracted {index.ntotal} vectors with dimension {dim}")
            
            # 배치 처리
            for i in tqdm(range(0, index.ntotal, batch_size), desc="Migrating vectors"):
                batch_end = min(i + batch_size, index.ntotal)
                batch_vectors = vectors[i:batch_end]
                batch_metadata = metadata_list[i:batch_end] if i < len(metadata_list) else []
                
                self._migrate_batch(conn, batch_vectors, batch_metadata, i, dim)
            
            conn.commit()
            logger.info("Migration completed successfully")
            
            # 검증
            validation = self.validate_migration(conn)
            if not validation['valid']:
                logger.error("Migration validation failed")
                if validation['errors']:
                    for error in validation['errors']:
                        logger.error(f"  - {error}")
                        self.stats['errors'].append(error)
                
                if backup_path and backup_path.exists():
                    logger.info("Rolling back due to validation failure...")
                    if self.rollback(backup_path):
                        logger.info("Rollback completed")
                    else:
                        logger.error("Rollback failed")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"  - {warning}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}", exc_info=True)
            self.stats['errors'].append(str(e))
            
            if backup_path and backup_path.exists():
                logger.info("Rolling back due to migration failure...")
                if self.rollback(backup_path):
                    logger.info("Rollback completed")
                else:
                    logger.error("Rollback failed")
        finally:
            conn.close()
        
        return self.stats
    
    def _migrate_batch(self, conn: sqlite3.Connection, 
                      vectors: np.ndarray,
                      metadata_list: List[Dict[str, Any]],
                      start_idx: int,
                      dim: int):
        """배치 마이그레이션"""
        for idx, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
            try:
                global_idx = start_idx + idx
                source_type = metadata.get('type') or metadata.get('source_type', '')
                
                # source_type이 없으면 메타데이터에서 추론
                if not source_type:
                    if metadata.get('case_id') or metadata.get('case_number') or metadata.get('doc_id'):
                        source_type = 'case_paragraph'
                    elif metadata.get('law_id') or metadata.get('law_name') or metadata.get('article_number'):
                        source_type = 'statute_article'
                    elif metadata.get('decision_id') or metadata.get('org'):
                        source_type = 'decision_paragraph'
                    elif metadata.get('interpretation_id'):
                        source_type = 'interpretation_paragraph'
                    else:
                        logger.warning(f"Missing source_type in metadata at index {global_idx}, metadata keys: {list(metadata.keys())[:10]}")
                        self.stats['failed_migrations'] += 1
                        continue
                
                chunk_id = self._create_text_chunk(conn, metadata, source_type, global_idx)
                if chunk_id is None:
                    self.stats['failed_migrations'] += 1
                    continue
                
                self._create_embedding(conn, chunk_id, vector, dim)
                self.stats['migrated_embeddings'] += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate vector at index {start_idx + idx}: {e}")
                self.stats['failed_migrations'] += 1
                self.stats['errors'].append(f"Index {start_idx + idx}: {str(e)}")
    
    def _create_text_chunk(self, conn: sqlite3.Connection, 
                          metadata: Dict[str, Any],
                          source_type: str,
                          chunk_index: int) -> Optional[int]:
        """text_chunks 테이블에 청크 생성"""
        try:
            domain_id = self.get_or_create_domain(conn, metadata.get('category', 'general'))
            source_id = None
            
            if source_type == 'statute_article':
                statute_id = self.find_or_create_statute(conn, metadata, domain_id)
                if statute_id:
                    source_id = self.find_or_create_statute_article(conn, statute_id, metadata)
            
            elif source_type == 'case_paragraph':
                case_id = self.find_or_create_case(conn, metadata, domain_id)
                if case_id:
                    text = metadata.get('content', '') or metadata.get('text', '')
                    source_id = self.find_or_create_case_paragraph(conn, case_id, chunk_index, text)
            
            elif source_type == 'decision_paragraph':
                decision_id = self.find_or_create_decision(conn, metadata, domain_id)
                if decision_id:
                    text = metadata.get('content', '') or metadata.get('text', '')
                    source_id = self.find_or_create_decision_paragraph(conn, decision_id, chunk_index, text)
            
            elif source_type == 'interpretation_paragraph':
                interpretation_id = self.find_or_create_interpretation(conn, metadata, domain_id)
                if interpretation_id:
                    text = metadata.get('content', '') or metadata.get('text', '')
                    source_id = self.find_or_create_interpretation_paragraph(conn, interpretation_id, chunk_index, text)
            
            if source_id is None:
                logger.warning(f"Failed to create source for {source_type} at index {chunk_index}")
                return None
            
            text = metadata.get('content', '') or metadata.get('text', '')
            
            cursor = conn.execute(
                """INSERT INTO text_chunks 
                   (source_type, source_id, level, chunk_index, text, meta)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    source_type,
                    source_id,
                    metadata.get('chunk_type', ''),
                    chunk_index,
                    text,
                    json.dumps(metadata, ensure_ascii=False)
                )
            )
            
            self.stats['migrated_chunks'] += 1
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Failed to create text chunk: {e}")
            return None
    
    def _create_embedding(self, conn: sqlite3.Connection, chunk_id: int, vector: np.ndarray, dim: int):
        """embeddings 테이블에 벡터 삽입"""
        try:
            vector_bytes = vector.astype(np.float32).tobytes()
            
            cursor = conn.execute(
                """SELECT id FROM embeddings WHERE chunk_id = ? AND model = ?""",
                (chunk_id, self.model_name)
            )
            if cursor.fetchone():
                self.stats['skipped_duplicates'] += 1
                return
            
            conn.execute(
                """INSERT INTO embeddings (chunk_id, model, dim, vector)
                   VALUES (?, ?, ?, ?)""",
                (chunk_id, self.model_name, dim, vector_bytes)
            )
            
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="벡터 임베딩 데이터베이스 마이그레이션")
    parser.add_argument('--vector-store-path', 
                        default='data/embeddings/ml_enhanced_ko_sroberta_precedents',
                        help='벡터스토어 기본 경로')
    parser.add_argument('--db-path', 
                        default='data/lawfirm_v2.db',
                        help='데이터베이스 경로')
    parser.add_argument('--version', 
                        default=None,
                        help='버전 번호 (예: v2.0.0). None이면 최신 버전 사용')
    parser.add_argument('--model-name', 
                        default='jhgan/ko-sroberta-multitask',
                        help='모델명')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=1000,
                        help='배치 크기')
    parser.add_argument('--no-backup', 
                        action='store_true',
                        help='백업하지 않음')
    parser.add_argument('--verbose', '-v', 
                        action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    migrator = VectorStoreMigrator(
        vector_store_base_path=Path(args.vector_store_path),
        db_path=Path(args.db_path),
        version=args.version,
        model_name=args.model_name
    )
    
    stats = migrator.migrate(backup=not args.no_backup, batch_size=args.batch_size)
    
    logger.info("=" * 60)
    logger.info("Migration Statistics:")
    logger.info(f"  Total vectors: {stats['total_vectors']}")
    logger.info(f"  Migrated chunks: {stats['migrated_chunks']}")
    logger.info(f"  Migrated embeddings: {stats['migrated_embeddings']}")
    logger.info(f"  Skipped duplicates: {stats['skipped_duplicates']}")
    logger.info(f"  Failed migrations: {stats['failed_migrations']}")
    
    if stats['errors']:
        logger.error(f"Errors: {len(stats['errors'])}")
        for error in stats['errors'][:10]:
            logger.error(f"  - {error}")
    
    return stats['failed_migrations'] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

