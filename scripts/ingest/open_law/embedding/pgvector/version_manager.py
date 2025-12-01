# -*- coding: utf-8 -*-
"""
PostgreSQL 임베딩 버전 관리 클래스
법령/판례 임베딩 버전 관리
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)


class PgEmbeddingVersionManager:
    """PostgreSQL 임베딩 버전 관리자"""
    
    def __init__(self, db_url: str):
        """
        버전 관리자 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
        """
        self.db_url = db_url
        self.logger = get_logger(__name__)
        
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
    
    def get_or_create_version(
        self,
        version: int,
        model_name: str,
        dim: int,
        data_type: str,  # 'statutes' or 'precedents'
        chunking_strategy: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> int:
        """
        버전 조회 또는 생성
        
        Args:
            version: 버전 번호
            model_name: 모델 이름
            dim: 벡터 차원
            data_type: 데이터 타입 ('statutes' or 'precedents')
            chunking_strategy: 청킹 전략
            description: 설명
            metadata: 추가 메타데이터
            set_active: 활성 버전으로 설정할지 여부
        
        Returns:
            버전 ID
        """
        import json
        
        # 기존 버전 조회
        check_sql = """
            SELECT id FROM embedding_versions
            WHERE version = :version AND data_type = :data_type
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(
                text(check_sql),
                {"version": version, "data_type": data_type}
            )
            row = result.fetchone()
            
            if row:
                version_id = row[0]
                self.logger.info(f"기존 버전 사용: version={version}, data_type={data_type}, id={version_id}")
                
                # 활성 버전 설정 요청 시
                if set_active:
                    self.set_active_version(version, data_type)
                
                return version_id
            
            # 새 버전 생성
            insert_sql = """
                INSERT INTO embedding_versions (
                    version, model_name, dim, data_type,
                    chunking_strategy, description, is_active, metadata
                ) VALUES (
                    :version, :model_name, :dim, :data_type,
                    :chunking_strategy, :description, :is_active, CAST(:metadata AS jsonb)
                )
                RETURNING id
            """
            
            # 활성 버전 설정 시 기존 활성 버전 비활성화
            if set_active:
                deactivate_sql = """
                    UPDATE embedding_versions
                    SET is_active = FALSE
                    WHERE data_type = :data_type AND is_active = TRUE
                """
                conn.execute(text(deactivate_sql), {"data_type": data_type})
                conn.commit()
            
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            
            result = conn.execute(
                text(insert_sql),
                {
                    "version": version,
                    "model_name": model_name,
                    "dim": dim,
                    "data_type": data_type,
                    "chunking_strategy": chunking_strategy,
                    "description": description,
                    "is_active": set_active,
                    "metadata": metadata_json
                }
            )
            conn.commit()
            
            version_id = result.scalar()
            self.logger.info(
                f"새 버전 생성: version={version}, data_type={data_type}, "
                f"model={model_name}, id={version_id}"
            )
            
            return version_id
    
    def get_active_version(self, data_type: str) -> Optional[Dict[str, Any]]:
        """
        활성 버전 조회
        
        Args:
            data_type: 데이터 타입 ('statutes' or 'precedents')
        
        Returns:
            버전 정보 딕셔너리 또는 None
        """
        sql = """
            SELECT id, version, model_name, dim, chunking_strategy,
                   description, metadata, created_at
            FROM embedding_versions
            WHERE data_type = :data_type AND is_active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"data_type": data_type})
            row = result.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "version": row[1],
                    "model_name": row[2],
                    "dim": row[3],
                    "chunking_strategy": row[4],
                    "description": row[5],
                    "metadata": row[6],
                    "created_at": row[7]
                }
            
            return None
    
    def set_active_version(self, version: int, data_type: str) -> bool:
        """
        활성 버전 설정
        
        Args:
            version: 버전 번호
            data_type: 데이터 타입
        
        Returns:
            성공 여부
        """
        with self.engine.connect() as conn:
            trans = conn.begin()
            try:
                # 기존 활성 버전 비활성화
                deactivate_sql = """
                    UPDATE embedding_versions
                    SET is_active = FALSE
                    WHERE data_type = :data_type AND is_active = TRUE
                """
                conn.execute(text(deactivate_sql), {"data_type": data_type})
                
                # 새 활성 버전 설정
                activate_sql = """
                    UPDATE embedding_versions
                    SET is_active = TRUE
                    WHERE version = :version AND data_type = :data_type
                """
                result = conn.execute(
                    text(activate_sql),
                    {"version": version, "data_type": data_type}
                )
                
                trans.commit()
                
                if result.rowcount > 0:
                    self.logger.info(f"활성 버전 설정: version={version}, data_type={data_type}")
                    return True
                else:
                    self.logger.warning(f"버전을 찾을 수 없음: version={version}, data_type={data_type}")
                    return False
            
            except Exception as e:
                trans.rollback()
                self.logger.error(f"활성 버전 설정 실패: {e}")
                return False
    
    def list_versions(self, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        버전 목록 조회
        
        Args:
            data_type: 데이터 타입 필터 (None이면 전체)
        
        Returns:
            버전 목록
        """
        if data_type:
            sql = """
                SELECT id, version, model_name, dim, data_type,
                       chunking_strategy, description, is_active,
                       metadata, created_at
                FROM embedding_versions
                WHERE data_type = :data_type
                ORDER BY version DESC, created_at DESC
            """
            params = {"data_type": data_type}
        else:
            sql = """
                SELECT id, version, model_name, dim, data_type,
                       chunking_strategy, description, is_active,
                       metadata, created_at
                FROM embedding_versions
                ORDER BY data_type, version DESC, created_at DESC
            """
            params = {}
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            versions = []
            
            for row in result:
                versions.append({
                    "id": row[0],
                    "version": row[1],
                    "model_name": row[2],
                    "dim": row[3],
                    "data_type": row[4],
                    "chunking_strategy": row[5],
                    "description": row[6],
                    "is_active": row[7],
                    "metadata": row[8],
                    "created_at": row[9]
                })
            
            return versions
    
    def get_version_info(self, version: int, data_type: str) -> Optional[Dict[str, Any]]:
        """
        특정 버전 정보 조회
        
        Args:
            version: 버전 번호
            data_type: 데이터 타입
        
        Returns:
            버전 정보 또는 None
        """
        sql = """
            SELECT id, version, model_name, dim, chunking_strategy,
                   description, is_active, metadata, created_at
            FROM embedding_versions
            WHERE version = :version AND data_type = :data_type
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(
                text(sql),
                {"version": version, "data_type": data_type}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "version": row[1],
                    "model_name": row[2],
                    "dim": row[3],
                    "chunking_strategy": row[4],
                    "description": row[5],
                    "is_active": row[6],
                    "metadata": row[7],
                    "created_at": row[8]
                }
            
            return None
    
    def get_next_version(self, data_type: str) -> int:
        """
        다음 버전 번호 조회
        
        Args:
            data_type: 데이터 타입
        
        Returns:
            다음 버전 번호
        """
        sql = """
            SELECT COALESCE(MAX(version), 0) + 1
            FROM embedding_versions
            WHERE data_type = :data_type
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"data_type": data_type})
            return result.scalar() or 1

