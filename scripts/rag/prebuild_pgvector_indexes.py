"""
PostgreSQL pgvector 인덱스 사전 구축 스크립트
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
except ImportError:
    from core.data.db_adapter import DatabaseAdapter

def get_database_url():
    """환경 변수에서 데이터베이스 URL 가져오기 (POSTGRES_* 환경 변수 조합)"""
    import os
    from urllib.parse import quote_plus
    
    # DATABASE_URL이 명시적으로 설정되어 있으면 사용
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # PostgreSQL 환경변수 조합 (프로젝트 루트 .env 파일의 설정 우선 사용)
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "lawfirmai_local")
    postgres_user = os.getenv("POSTGRES_USER", "lawfirmai")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "local_password")
    
    # URL 인코딩 (특수문자 처리)
    encoded_password = quote_plus(postgres_password)
    
    # PostgreSQL URL 생성
    database_url = f"postgresql://{postgres_user}:{encoded_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    return database_url

def create_pgvector_indexes(
    index_type: str = "hnsw",  # "hnsw" or "ivfflat"
    m: int = 16,  # HNSW 파라미터
    ef_construction: int = 64,  # HNSW 파라미터
    lists: int = 100  # IVFFlat 파라미터
):
    """
    PostgreSQL pgvector 인덱스 생성
    
    Args:
        index_type: 인덱스 타입 ("hnsw" or "ivfflat")
        m: HNSW m 파라미터 (연결 수)
        ef_construction: HNSW ef_construction 파라미터
        lists: IVFFlat lists 파라미터 (클러스터 수)
    """
    database_url = get_database_url()
    db_adapter = DatabaseAdapter(database_url)
    
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        # pgvector 확장 확인
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        row = cursor.fetchone()
        has_extension = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
        
        if not has_extension:
            print("❌ pgvector extension not found. Creating extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            print("✅ pgvector extension created")
        
        # 테이블 목록 (실제 테이블 이름 사용)
        tables = [
            ("statute_embeddings", "embedding_vector"),
            ("precedent_chunks", "embedding_vector"),
            ("embeddings", "vector"),
            ("interpretation_paragraphs", "embedding_vector"),
            ("decision_paragraphs", "embedding_vector")
        ]
        
        for table_name, vector_column in tables:
            # 테이블 존재 확인
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            row = cursor.fetchone()
            table_exists = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
            if not table_exists:
                print(f"⚠️  Table {table_name} not found, skipping...")
                continue
            
            # 벡터 컬럼 존재 확인
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = %s
                )
            """, (table_name, vector_column))
            
            row = cursor.fetchone()
            column_exists = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
            if not column_exists:
                print(f"⚠️  Column {table_name}.{vector_column} not found, skipping...")
                continue
            
            # 인덱스 이름
            if index_type == "hnsw":
                index_name = f"idx_{table_name}_{vector_column}_hnsw"
                create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table_name}
                    USING hnsw ({vector_column} vector_cosine_ops)
                    WITH (m = {m}, ef_construction = {ef_construction})
                """
            else:  # ivfflat
                index_name = f"idx_{table_name}_{vector_column}_ivfflat"
                create_index_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table_name}
                    USING ivfflat ({vector_column} vector_cosine_ops)
                    WITH (lists = {lists})
                """
            
            # 기존 인덱스 확인
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = %s
                )
            """, (index_name,))
            
            row = cursor.fetchone()
            index_exists = row[0] if isinstance(row, tuple) else (row.get('exists', False) if isinstance(row, dict) else False)
            if index_exists:
                print(f"✅ Index {index_name} already exists")
            else:
                print(f"🔨 Creating index {index_name}...")
                try:
                    cursor.execute(create_index_sql)
                    conn.commit()
                    print(f"✅ Index {index_name} created successfully")
                except Exception as e:
                    print(f"❌ Failed to create index {index_name}: {e}")
                    conn.rollback()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prebuild pgvector indexes")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw",
                       help="Index type: hnsw (faster) or ivfflat (less memory)")
    parser.add_argument("--m", type=int, default=16, help="HNSW m parameter")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW ef_construction parameter")
    parser.add_argument("--lists", type=int, default=100, help="IVFFlat lists parameter")
    
    args = parser.parse_args()
    create_pgvector_indexes(
        index_type=args.index_type,
        m=args.m,
        ef_construction=args.ef_construction,
        lists=args.lists
    )

