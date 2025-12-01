"""
스키마 검증 유틸리티
"""

import logging
from typing import List, Tuple, Optional
from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


def check_table_exists(adapter: DatabaseAdapter, table_name: str) -> bool:
    """테이블 존재 여부 확인"""
    return adapter.table_exists(table_name)


def check_column_exists(adapter: DatabaseAdapter, table_name: str, column_name: str) -> bool:
    """컬럼 존재 여부 확인"""
    try:
        with adapter.get_connection_context() as conn:
            cursor = conn.cursor()
            if adapter.db_type == 'postgresql':
                cursor.execute(
                    """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = %s 
                    AND column_name = %s
                    """,
                    (table_name, column_name)
                )
            else:
                cursor.execute(
                    "PRAGMA table_info(?)",
                    (table_name,)
                )
                columns = [row[1] for row in cursor.fetchall()]
                return column_name in columns
            
            return cursor.fetchone() is not None
    except Exception as e:
        logger.warning(f"컬럼 확인 중 오류: {e}")
        return False


def check_index_exists(adapter: DatabaseAdapter, index_name: str) -> bool:
    """인덱스 존재 여부 확인"""
    try:
        with adapter.get_connection_context() as conn:
            cursor = conn.cursor()
            if adapter.db_type == 'postgresql':
                cursor.execute(
                    """
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE schemaname = 'public' 
                    AND indexname = %s
                    """,
                    (index_name,)
                )
            else:
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                    (index_name,)
                )
            
            return cursor.fetchone() is not None
    except Exception as e:
        logger.warning(f"인덱스 확인 중 오류: {e}")
        return False


def check_extension_exists(adapter: DatabaseAdapter, extension_name: str) -> bool:
    """PostgreSQL 확장 존재 여부 확인"""
    if adapter.db_type != 'postgresql':
        return False
    
    try:
        with adapter.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT extname FROM pg_extension WHERE extname = %s",
                (extension_name,)
            )
            return cursor.fetchone() is not None
    except Exception as e:
        logger.warning(f"확장 확인 중 오류: {e}")
        return False


def validate_schema(adapter: DatabaseAdapter) -> Tuple[bool, List[str]]:
    """
    스키마 검증
    
    Args:
        adapter: DatabaseAdapter 인스턴스
    
    Returns:
        (성공 여부, 오류 목록)
    """
    errors = []
    
    logger.info("=" * 80)
    logger.info("PostgreSQL 스키마 검증")
    logger.info("=" * 80)
    
    # 1. pgvector 확장 확인
    logger.info("\n1. PostgreSQL 확장 확인...")
    if adapter.db_type == 'postgresql':
        if check_extension_exists(adapter, 'vector'):
            logger.info("   ✅ pgvector 확장 설치됨")
        else:
            logger.error("   ❌ pgvector 확장이 설치되지 않음")
            errors.append("pgvector 확장이 설치되지 않음")
    else:
        logger.warning("   ⚠️  SQLite 사용 중 (확장 확인 건너뜀)")
    
    # 2. 필수 테이블 확인
    logger.info("\n2. 필수 테이블 확인...")
    required_tables = [
        'domains',
        'sources',
        'statutes',
        'statute_articles',
        'cases',
        'case_paragraphs',
        'decisions',
        'decision_paragraphs',
        'interpretations',
        'interpretation_paragraphs',
        'text_chunks',
        'embeddings',
        'embedding_versions',
        'retrieval_cache'
    ]
    
    for table in required_tables:
        if check_table_exists(adapter, table):
            logger.info(f"   ✅ {table} 테이블 존재")
        else:
            logger.error(f"   ❌ {table} 테이블 없음")
            errors.append(f"{table} 테이블 없음")
    
    # 3. FTS 컬럼 확인 (PostgreSQL만)
    if adapter.db_type == 'postgresql':
        logger.info("\n3. Full-Text Search 컬럼 확인...")
        fts_tables = [
            ('statute_articles', 'text_search_vector'),
            ('case_paragraphs', 'text_search_vector'),
            ('decision_paragraphs', 'text_search_vector'),
            ('interpretation_paragraphs', 'text_search_vector')
        ]
        
        for table, column in fts_tables:
            if check_column_exists(adapter, table, column):
                logger.info(f"   ✅ {table}.{column} 컬럼 존재")
            else:
                logger.error(f"   ❌ {table}.{column} 컬럼 없음")
                errors.append(f"{table}.{column} 컬럼 없음")
    
    # 4. 벡터 컬럼 확인
    logger.info("\n4. 벡터 컬럼 확인...")
    if adapter.db_type == 'postgresql':
        if check_column_exists(adapter, 'embeddings', 'vector'):
            logger.info("   ✅ embeddings.vector 컬럼 존재")
            # VECTOR 타입 확인
            try:
                with adapter.get_connection_context() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT data_type 
                        FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND table_name = 'embeddings' 
                        AND column_name = 'vector'
                        """
                    )
                    result = cursor.fetchone()
                    if result and 'vector' in result[0].lower():
                        logger.info("   ✅ embeddings.vector가 VECTOR 타입임")
                    else:
                        logger.warning("   ⚠️  embeddings.vector가 VECTOR 타입이 아님")
                        errors.append("embeddings.vector가 VECTOR 타입이 아님")
            except Exception as e:
                logger.warning(f"   ⚠️  벡터 타입 확인 중 오류: {e}")
        else:
            logger.error("   ❌ embeddings.vector 컬럼 없음")
            errors.append("embeddings.vector 컬럼 없음")
    else:
        if check_column_exists(adapter, 'embeddings', 'vector'):
            logger.info("   ✅ embeddings.vector 컬럼 존재 (SQLite BLOB)")
        else:
            logger.error("   ❌ embeddings.vector 컬럼 없음")
            errors.append("embeddings.vector 컬럼 없음")
    
    # 5. JSONB 컬럼 확인 (PostgreSQL만)
    if adapter.db_type == 'postgresql':
        logger.info("\n5. JSONB 컬럼 확인...")
        jsonb_columns = [
            ('text_chunks', 'meta'),
            ('embedding_versions', 'metadata')
        ]
        
        for table, column in jsonb_columns:
            if check_column_exists(adapter, table, column):
                logger.info(f"   ✅ {table}.{column} 컬럼 존재")
                # JSONB 타입 확인
                try:
                    with adapter.get_connection_context() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            SELECT data_type 
                            FROM information_schema.columns 
                            WHERE table_schema = 'public' 
                            AND table_name = %s 
                            AND column_name = %s
                            """,
                            (table, column)
                        )
                        result = cursor.fetchone()
                        if result and result[0] == 'jsonb':
                            logger.info(f"   ✅ {table}.{column}가 JSONB 타입임")
                        else:
                            logger.warning(f"   ⚠️  {table}.{column}가 JSONB 타입이 아님 (타입: {result[0] if result else 'N/A'})")
                            errors.append(f"{table}.{column}가 JSONB 타입이 아님")
                except Exception as e:
                    logger.warning(f"   ⚠️  JSONB 타입 확인 중 오류: {e}")
            else:
                logger.error(f"   ❌ {table}.{column} 컬럼 없음")
                errors.append(f"{table}.{column} 컬럼 없음")
    
    # 6. 인덱스 확인
    logger.info("\n6. 인덱스 확인...")
    required_indexes = [
        'idx_statute_articles_fts',
        'idx_case_paragraphs_fts',
        'idx_decision_paragraphs_fts',
        'idx_interpretation_paragraphs_fts',
        'idx_embeddings_vector',
        'idx_text_chunks_meta_gin',
        'idx_embedding_versions_active'
    ]
    
    for index in required_indexes:
        if check_index_exists(adapter, index):
            logger.info(f"   ✅ {index} 인덱스 존재")
        else:
            logger.warning(f"   ⚠️  {index} 인덱스 없음 (선택적)")
    
    # 7. 트리거 함수 확인 (PostgreSQL만)
    if adapter.db_type == 'postgresql':
        logger.info("\n7. 트리거 함수 확인...")
        trigger_functions = [
            'update_statute_articles_fts',
            'update_case_paragraphs_fts',
            'update_decision_paragraphs_fts',
            'update_interpretation_paragraphs_fts'
        ]
        
        for func in trigger_functions:
            try:
                with adapter.get_connection_context() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT routine_name 
                        FROM information_schema.routines 
                        WHERE routine_schema = 'public' 
                        AND routine_name = %s
                        """,
                        (func,)
                    )
                    if cursor.fetchone():
                        logger.info(f"   ✅ {func} 함수 존재")
                    else:
                        logger.error(f"   ❌ {func} 함수 없음")
                        errors.append(f"{func} 함수 없음")
            except Exception as e:
                logger.warning(f"   ⚠️  함수 확인 중 오류: {e}")
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    if errors:
        logger.error(f"❌ 검증 실패: {len(errors)}개 오류 발견")
        logger.error("\n오류 목록:")
        for i, error in enumerate(errors, 1):
            logger.error(f"   {i}. {error}")
        return False, errors
    else:
        logger.info("✅ 스키마 검증 성공!")
        return True, []

