"""
데이터베이스 기반 임베딩 버전 관리 시스템

청킹 전략별로 임베딩 버전을 관리하고 완전 교체 방식을 지원합니다.
"""
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingVersionManager:
    """데이터베이스 기반 임베딩 버전 관리 클래스"""
    
    def __init__(self, db_path: str):
        """
        버전 관리자 초기화
        
        Args:
            db_path: 데이터베이스 경로
        """
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """데이터베이스 연결 생성"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        return conn
    
    def register_version(
        self,
        version_name: str,
        chunking_strategy: str,
        model_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        set_active: bool = False,
        create_faiss_version: bool = True,
        faiss_version_manager = None
    ) -> int:
        """
        새 임베딩 버전 등록
        
        Args:
            version_name: 버전 이름 (예: "v1.0.0-dynamic")
            chunking_strategy: 청킹 전략 (standard, dynamic, hybrid)
            model_name: 임베딩 모델명
            description: 버전 설명
            metadata: 추가 메타데이터 (JSON 문자열로 저장)
            set_active: 활성 버전으로 설정할지 여부
            create_faiss_version: FAISS 버전도 함께 생성할지 여부
            faiss_version_manager: FAISSVersionManager 인스턴스 (None이면 자동 생성 시도)
        
        Returns:
            int: 생성된 버전 ID
        """
        conn = self._get_connection()
        try:
            # 기존 활성 버전 비활성화 (같은 전략의 경우)
            if set_active:
                conn.execute("""
                    UPDATE embedding_versions 
                    SET is_active = 0 
                    WHERE chunking_strategy = ?
                """, (chunking_strategy,))
            
            # 새 버전 등록
            import json
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = conn.execute("""
                INSERT INTO embedding_versions 
                (version_name, chunking_strategy, model_name, description, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                version_name,
                chunking_strategy,
                model_name,
                description,
                1 if set_active else 0,
                metadata_json
            ))
            
            version_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Registered embedding version: {version_name} (ID: {version_id})")
            
            # FAISS 버전도 함께 생성
            if create_faiss_version and faiss_version_manager:
                try:
                    chunking_config = metadata.get('chunking_config', {}) if metadata else {}
                    embedding_config = {
                        'model': model_name,
                        'dimension': metadata.get('dimension', 768) if metadata else 768
                    }
                    
                    faiss_version_name = f"{version_name}-{chunking_strategy}"
                    faiss_version_manager.create_version(
                        version_name=faiss_version_name,
                        embedding_version_id=version_id,
                        chunking_strategy=chunking_strategy,
                        chunking_config=chunking_config,
                        embedding_config=embedding_config,
                        document_count=0,
                        total_chunks=0,
                        status='active' if set_active else 'inactive'
                    )
                    logger.info(f"Created FAISS version: {faiss_version_name}")
                except Exception as e:
                    logger.warning(f"Failed to create FAISS version: {e}")
            
            return version_id
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            logger.error(f"Version {version_name} already exists: {e}")
            raise
        finally:
            conn.close()
    
    def get_active_version(self, chunking_strategy: str) -> Optional[Dict]:
        """
        특정 청킹 전략의 활성 버전 조회
        
        Args:
            chunking_strategy: 청킹 전략
        
        Returns:
            Optional[Dict]: 활성 버전 정보, 없으면 None
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM embedding_versions
                WHERE chunking_strategy = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """, (chunking_strategy,))
            
            row = cursor.fetchone()
            if row:
                import json
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                return result
            return None
            
        finally:
            conn.close()
    
    def get_version_by_name(self, version_name: str) -> Optional[Dict]:
        """
        버전명으로 버전 조회
        
        Args:
            version_name: 버전 이름
        
        Returns:
            Optional[Dict]: 버전 정보, 없으면 None
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM embedding_versions
                WHERE version_name = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (version_name,))
            
            row = cursor.fetchone()
            if row:
                import json
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                return result
            return None
            
        finally:
            conn.close()
    
    def list_versions(self, chunking_strategy: Optional[str] = None) -> List[Dict]:
        """
        버전 목록 조회
        
        Args:
            chunking_strategy: 청킹 전략 필터 (None이면 전체)
        
        Returns:
            List[Dict]: 버전 정보 리스트
        """
        conn = self._get_connection()
        try:
            if chunking_strategy:
                cursor = conn.execute("""
                    SELECT * FROM embedding_versions
                    WHERE chunking_strategy = ?
                    ORDER BY created_at DESC
                """, (chunking_strategy,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM embedding_versions
                    ORDER BY created_at DESC
                """)
            
            results = []
            import json
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                results.append(result)
            
            return results
            
        finally:
            conn.close()
    
    def set_active_version(self, version_id: int) -> bool:
        """
        활성 버전 설정
        
        Args:
            version_id: 버전 ID
        
        Returns:
            bool: 성공 여부
        """
        conn = self._get_connection()
        try:
            # 해당 버전의 청킹 전략 조회
            cursor = conn.execute("""
                SELECT chunking_strategy FROM embedding_versions WHERE id = ?
            """, (version_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"Version ID {version_id} not found")
                return False
            
            chunking_strategy = row['chunking_strategy']
            
            # 같은 전략의 다른 버전들 비활성화
            conn.execute("""
                UPDATE embedding_versions 
                SET is_active = 0 
                WHERE chunking_strategy = ?
            """, (chunking_strategy,))
            
            # 지정된 버전 활성화
            conn.execute("""
                UPDATE embedding_versions 
                SET is_active = 1 
                WHERE id = ?
            """, (version_id,))
            
            conn.commit()
            logger.info(f"Set active version: {version_id}")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to set active version: {e}")
            return False
        finally:
            conn.close()
    
    def delete_chunks_by_version(
        self,
        source_type: str,
        source_id: int,
        old_version_id: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        특정 문서의 기존 청크 및 임베딩 삭제 (완전 교체 방식)
        
        Args:
            source_type: 소스 타입 (statute_article, case_paragraph, etc.)
            source_id: 소스 ID
            old_version_id: 삭제할 버전 ID (None이면 해당 문서의 모든 청크 삭제)
        
        Returns:
            Tuple[int, int]: (삭제된 청크 수, 삭제된 임베딩 수)
        """
        conn = self._get_connection()
        try:
            # 삭제할 청크 ID 조회
            if old_version_id:
                cursor = conn.execute("""
                    SELECT id FROM text_chunks
                    WHERE source_type = ? AND source_id = ? AND embedding_version_id = ?
                """, (source_type, source_id, old_version_id))
            else:
                cursor = conn.execute("""
                    SELECT id FROM text_chunks
                    WHERE source_type = ? AND source_id = ?
                """, (source_type, source_id))
            
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            if not chunk_ids:
                logger.debug(f"No chunks found for {source_type}/{source_id}")
                return (0, 0)
            
            # 임베딩 삭제 (CASCADE로 자동 삭제되지만 명시적으로 삭제)
            placeholders = ",".join(["?"] * len(chunk_ids))
            embedding_cursor = conn.execute(
                f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )
            deleted_embeddings = embedding_cursor.rowcount
            
            # 청크 삭제
            chunk_cursor = conn.execute(
                f"DELETE FROM text_chunks WHERE id IN ({placeholders})",
                chunk_ids
            )
            deleted_chunks = chunk_cursor.rowcount
            
            conn.commit()
            logger.info(
                f"Deleted {deleted_chunks} chunks and {deleted_embeddings} embeddings "
                f"for {source_type}/{source_id}"
            )
            
            return (deleted_chunks, deleted_embeddings)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete chunks: {e}")
            raise
        finally:
            conn.close()
    
    def assign_version_to_chunks(
        self,
        chunk_ids: List[int],
        version_id: int
    ) -> int:
        """
        청크에 버전 ID 할당
        
        Args:
            chunk_ids: 청크 ID 리스트
            version_id: 버전 ID
        
        Returns:
            int: 업데이트된 청크 수
        """
        if not chunk_ids:
            return 0
        
        conn = self._get_connection()
        try:
            placeholders = ",".join(["?"] * len(chunk_ids))
            cursor = conn.execute(
                f"""
                UPDATE text_chunks 
                SET embedding_version_id = ? 
                WHERE id IN ({placeholders})
                """,
                (version_id, *chunk_ids)
            )
            
            updated_count = cursor.rowcount
            conn.commit()
            
            logger.debug(f"Assigned version {version_id} to {updated_count} chunks")
            return updated_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to assign version to chunks: {e}")
            raise
        finally:
            conn.close()
    
    def get_version_statistics(self, version_id: int) -> Dict[str, Any]:
        """
        버전별 통계 정보 조회
        
        Args:
            version_id: 버전 ID
        
        Returns:
            Dict: 통계 정보 (청크 수, 임베딩 수, 문서 수 등)
        """
        conn = self._get_connection()
        try:
            # 버전 정보 조회
            cursor = conn.execute(
                "SELECT * FROM embedding_versions WHERE id = ?",
                (version_id,)
            )
            version_row = cursor.fetchone()
            if not version_row:
                return {}
            
            version_info = dict(version_row)
            
            # 청크 수
            cursor = conn.execute(
                "SELECT COUNT(*) FROM text_chunks WHERE embedding_version_id = ?",
                (version_id,)
            )
            chunk_count = cursor.fetchone()[0]
            
            # 임베딩 수
            cursor = conn.execute(
                "SELECT COUNT(*) FROM embeddings e JOIN text_chunks tc ON e.chunk_id = tc.id WHERE tc.embedding_version_id = ?",
                (version_id,)
            )
            embedding_count = cursor.fetchone()[0]
            
            # 문서 수 (고유한 source_type, source_id 조합)
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT source_type || '_' || source_id) FROM text_chunks WHERE embedding_version_id = ?",
                (version_id,)
            )
            document_count = cursor.fetchone()[0]
            
            # 소스 타입별 분포
            cursor = conn.execute(
                "SELECT source_type, COUNT(*) as count FROM text_chunks WHERE embedding_version_id = ? GROUP BY source_type",
                (version_id,)
            )
            source_type_distribution = {row['source_type']: row['count'] for row in cursor.fetchall()}
            
            # metadata 파싱
            import json
            metadata = None
            if version_info.get('metadata'):
                try:
                    metadata = json.loads(version_info['metadata']) if isinstance(version_info['metadata'], str) else version_info['metadata']
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            # chunking_config 추출
            chunking_config = metadata.get('chunking_config', {}) if metadata else {}
            
            # embedding_config 추출 및 생성
            embedding_config = metadata.get('embedding_config', {}) if metadata else {}
            if not embedding_config or not embedding_config.get('model'):
                model_name = version_info.get('model_name')
                dimension = None
                
                if embedding_count > 0:
                    cursor = conn.execute(
                        "SELECT DISTINCT dim FROM embeddings e JOIN text_chunks tc ON e.chunk_id = tc.id WHERE tc.embedding_version_id = ? LIMIT 1",
                        (version_id,)
                    )
                    dim_row = cursor.fetchone()
                    if dim_row:
                        dimension = dim_row[0]
                
                if not dimension and metadata:
                    dimension = metadata.get('dimension')
                
                embedding_config = {
                    'model': model_name,
                    'dimension': dimension
                }
            
            return {
                'version_id': version_id,
                'version_name': version_info.get('version_name'),
                'chunking_strategy': version_info.get('chunking_strategy'),
                'model_name': version_info.get('model_name'),
                'is_active': bool(version_info.get('is_active')),
                'created_at': version_info.get('created_at'),
                'chunk_count': chunk_count,
                'embedding_count': embedding_count,
                'document_count': document_count,
                'source_type_distribution': source_type_distribution,
                'chunking_config': chunking_config,
                'embedding_config': embedding_config
            }
            
        except Exception as e:
            logger.error(f"Failed to get version statistics: {e}")
            return {}
        finally:
            conn.close()
    
    def compare_versions(self, version_id1: int, version_id2: int) -> Dict[str, Any]:
        """
        두 버전 비교
        
        Args:
            version_id1: 첫 번째 버전 ID
            version_id2: 두 번째 버전 ID
        
        Returns:
            Dict: 비교 결과
        """
        stats1 = self.get_version_statistics(version_id1)
        stats2 = self.get_version_statistics(version_id2)
        
        return {
            'version1': stats1,
            'version2': stats2,
            'differences': {
                'chunk_count_diff': stats1.get('chunk_count', 0) - stats2.get('chunk_count', 0),
                'embedding_count_diff': stats1.get('embedding_count', 0) - stats2.get('embedding_count', 0),
                'document_count_diff': stats1.get('document_count', 0) - stats2.get('document_count', 0)
            }
        }
    
    def delete_version(self, version_id: int, delete_chunks: bool = False) -> Tuple[int, int]:
        """
        버전 삭제
        
        Args:
            version_id: 버전 ID
            delete_chunks: 청크 및 임베딩도 함께 삭제할지 여부
        
        Returns:
            Tuple[int, int]: (삭제된 청크 수, 삭제된 임베딩 수)
        """
        conn = self._get_connection()
        try:
            deleted_chunks = 0
            deleted_embeddings = 0
            
            if delete_chunks:
                # 청크 ID 조회
                cursor = conn.execute(
                    "SELECT id FROM text_chunks WHERE embedding_version_id = ?",
                    (version_id,)
                )
                chunk_ids = [row[0] for row in cursor.fetchall()]
                
                if chunk_ids:
                    # 임베딩 삭제
                    placeholders = ",".join(["?"] * len(chunk_ids))
                    embedding_cursor = conn.execute(
                        f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})",
                        chunk_ids
                    )
                    deleted_embeddings = embedding_cursor.rowcount
                    
                    # 청크 삭제
                    chunk_cursor = conn.execute(
                        f"DELETE FROM text_chunks WHERE id IN ({placeholders})",
                        chunk_ids
                    )
                    deleted_chunks = chunk_cursor.rowcount
            
            # 버전 삭제
            conn.execute(
                "DELETE FROM embedding_versions WHERE id = ?",
                (version_id,)
            )
            
            conn.commit()
            logger.info(f"Deleted version {version_id}: {deleted_chunks} chunks, {deleted_embeddings} embeddings")
            return (deleted_chunks, deleted_embeddings)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete version: {e}")
            raise
        finally:
            conn.close()
    
    def get_all_active_versions(self) -> List[Dict]:
        """
        모든 활성 버전 조회 (청킹 전략별)
        
        Returns:
            List[Dict]: 활성 버전 리스트
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT * FROM embedding_versions
                WHERE is_active = 1
                ORDER BY chunking_strategy, created_at DESC
            """)
            
            results = []
            import json
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                results.append(result)
            
            return results
            
        finally:
            conn.close()

