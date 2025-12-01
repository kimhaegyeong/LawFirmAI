# -*- coding: utf-8 -*-
"""
pgvector 인덱스 생성기
PostgreSQL pgvector 인덱스 생성 및 관리
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger(__name__)


class PgVectorIndexer:
    """pgvector 인덱스 생성기"""
    
    def __init__(self, db_url: str):
        """
        인덱스 생성기 초기화
        
        Args:
            db_url: PostgreSQL 데이터베이스 URL
        """
        self.db_url = db_url
        try:
            from lawfirm_langgraph.core.utils.logger import get_logger
            self.logger = get_logger(__name__)
        except ImportError:
            self.logger = logging.getLogger(__name__)
        
        self.engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False
        )
    
    def create_ivfflat_index(
        self,
        table_name: str,
        column_name: str = "embedding_vector",
        lists: int = 100,
        replace: bool = False,
        version: Optional[int] = None
    ) -> bool:
        """
        ivfflat 인덱스 생성 (버전별 인덱스 지원)
        
        Args:
            table_name: 테이블 이름
            column_name: 벡터 컬럼 이름
            lists: 클러스터 수 (일반적으로 sqrt(총_벡터_수))
            replace: 기존 인덱스 교체 여부
            version: 임베딩 버전 (None이면 전체, 버전별 인덱스 생성)
        
        Returns:
            성공 여부
        """
        if version is not None:
            index_name = f"idx_{table_name}_{column_name}_v{version}_ivfflat"
        else:
            index_name = f"idx_{table_name}_{column_name}_ivfflat"
        
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                
                try:
                    # 기존 인덱스 삭제 (replace=True인 경우)
                    if replace:
                        drop_sql = f"DROP INDEX IF EXISTS {index_name}"
                        conn.execute(text(drop_sql))
                        self.logger.info(f"기존 인덱스 삭제: {index_name}")
                    
                    # ivfflat 인덱스 생성 (버전별 필터 포함)
                    if version is not None:
                        create_sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING ivfflat ({column_name} vector_cosine_ops)
                            WHERE embedding_version = :version
                            WITH (lists = :lists)
                        """
                        conn.execute(text(create_sql), {"lists": lists, "version": version})
                    else:
                        create_sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING ivfflat ({column_name} vector_cosine_ops)
                            WITH (lists = :lists)
                        """
                        conn.execute(text(create_sql), {"lists": lists})
                    trans.commit()
                    
                    self.logger.info(
                        f"ivfflat 인덱스 생성 완료: {index_name} "
                        f"(lists={lists})"
                    )
                    
                    return True
                
                except Exception as e:
                    trans.rollback()
                    # 인덱스가 이미 존재하는 경우 무시
                    if "already exists" not in str(e).lower():
                        self.logger.error(f"인덱스 생성 실패: {e}")
                        raise
                    return True
        
        except Exception as e:
            self.logger.error(f"인덱스 생성 오류: {e}")
            return False
    
    def create_hnsw_index(
        self,
        table_name: str,
        column_name: str = "embedding_vector",
        m: int = 16,
        ef_construction: int = 64,
        replace: bool = False,
        version: Optional[int] = None
    ) -> bool:
        """
        HNSW 인덱스 생성 (선택, 성능 테스트용, 버전별 인덱스 지원)
        
        Args:
            table_name: 테이블 이름
            column_name: 벡터 컬럼 이름
            m: 각 레벨에서 연결할 최대 노드 수
            ef_construction: 인덱스 빌드 시 검색 범위
            replace: 기존 인덱스 교체 여부
            version: 임베딩 버전 (None이면 전체, 버전별 인덱스 생성)
        
        Returns:
            성공 여부
        """
        if version is not None:
            index_name = f"idx_{table_name}_{column_name}_v{version}_hnsw"
        else:
            index_name = f"idx_{table_name}_{column_name}_hnsw"
        
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                
                try:
                    # 기존 인덱스 삭제 (replace=True인 경우)
                    if replace:
                        drop_sql = f"DROP INDEX IF EXISTS {index_name}"
                        conn.execute(text(drop_sql))
                        self.logger.info(f"기존 인덱스 삭제: {index_name}")
                    
                    # HNSW 인덱스 생성 (버전별 필터 포함)
                    if version is not None:
                        create_sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING hnsw ({column_name} vector_cosine_ops)
                            WHERE embedding_version = :version
                            WITH (m = :m, ef_construction = :ef_construction)
                        """
                        conn.execute(
                            text(create_sql),
                            {"m": m, "ef_construction": ef_construction, "version": version}
                        )
                    else:
                        create_sql = f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {table_name}
                            USING hnsw ({column_name} vector_cosine_ops)
                            WITH (m = :m, ef_construction = :ef_construction)
                        """
                        conn.execute(
                            text(create_sql),
                            {"m": m, "ef_construction": ef_construction}
                        )
                    trans.commit()
                    
                    self.logger.info(
                        f"HNSW 인덱스 생성 완료: {index_name} "
                        f"(m={m}, ef_construction={ef_construction})"
                    )
                    
                    return True
                
                except Exception as e:
                    trans.rollback()
                    # 인덱스가 이미 존재하는 경우 무시
                    if "already exists" not in str(e).lower():
                        self.logger.error(f"인덱스 생성 실패: {e}")
                        raise
                    return True
        
        except Exception as e:
            self.logger.error(f"인덱스 생성 오류: {e}")
            return False
    
    def get_index_stats(
        self,
        table_name: str,
        column_name: str = "embedding_vector"
    ) -> Dict[str, Any]:
        """
        인덱스 통계 조회
        
        Args:
            table_name: 테이블 이름
            column_name: 벡터 컬럼 이름
        
        Returns:
            인덱스 통계 정보
        """
        stats = {}
        
        try:
            with self.engine.connect() as conn:
                # 테이블 크기
                size_query = f"""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('{table_name}')) as table_size,
                        COUNT(*) as total_rows
                    FROM {table_name}
                    WHERE {column_name} IS NOT NULL
                """
                result = conn.execute(text(size_query))
                row = result.fetchone()
                if row:
                    stats["table_size"] = row.table_size
                    stats["total_rows"] = row.total_rows
                
                # 인덱스 정보
                index_query = """
                    SELECT 
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE tablename = :table_name
                      AND indexdef LIKE '%' || :column_name || '%'
                """
                result = conn.execute(
                    text(index_query),
                    {"table_name": table_name, "column_name": column_name}
                )
                indexes = []
                for row in result:
                    indexes.append({
                        "name": row.indexname,
                        "definition": row.indexdef
                    })
                stats["indexes"] = indexes
        
        except Exception as e:
            self.logger.error(f"인덱스 통계 조회 실패: {e}")
        
        return stats
    
    def create_version_indexes(
        self,
        data_type: str,  # 'statutes' or 'precedents'
        version: int,
        index_type: str = "ivfflat",  # "ivfflat" or "hnsw"
        lists: int = 100,
        m: int = 16,
        ef_construction: int = 64
    ) -> Dict[str, bool]:
        """
        버전별 인덱스 생성 (법령/판례)
        
        Args:
            data_type: 데이터 타입 ('statutes' or 'precedents')
            version: 임베딩 버전
            index_type: 인덱스 타입 ('ivfflat' or 'hnsw')
            lists: ivfflat 클러스터 수
            m: HNSW m 파라미터
            ef_construction: HNSW ef_construction 파라미터
        
        Returns:
            인덱스 생성 결과 딕셔너리
        """
        results = {}
        
        if data_type == "statutes":
            table_name = "statute_embeddings"
        elif data_type == "precedents":
            table_name = "precedent_chunks"
        else:
            self.logger.error(f"Unknown data_type: {data_type}")
            return results
        
        if index_type == "ivfflat":
            success = self.create_ivfflat_index(
                table_name=table_name,
                column_name="embedding_vector",
                lists=lists,
                version=version,
                replace=False
            )
            results[f"{table_name}_v{version}_ivfflat"] = success
        elif index_type == "hnsw":
            success = self.create_hnsw_index(
                table_name=table_name,
                column_name="embedding_vector",
                m=m,
                ef_construction=ef_construction,
                version=version,
                replace=False
            )
            results[f"{table_name}_v{version}_hnsw"] = success
        else:
            self.logger.error(f"Unknown index_type: {index_type}")
        
        return results

