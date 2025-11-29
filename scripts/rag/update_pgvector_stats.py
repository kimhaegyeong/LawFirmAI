"""
PostgreSQL 통계 정보 업데이트
"""

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

def update_vector_table_stats():
    """벡터 테이블 통계 정보 업데이트"""
    database_url = get_database_url()
    db_adapter = DatabaseAdapter(database_url)
    
    tables = [
        "statute_embeddings",
        "precedent_chunks",
        "embeddings",
        "interpretation_paragraphs",
        "decision_paragraphs",
        "precedent_contents",
        "statute_articles"
    ]
    
    with db_adapter.get_connection_context() as conn:
        cursor = conn.cursor()
        
        for table_name in tables:
            try:
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
                
                print(f"Updating statistics for {table_name}...")
                cursor.execute(f"ANALYZE {table_name}")
                conn.commit()
                print(f"✅ Statistics updated for {table_name}")
            except Exception as e:
                print(f"❌ Failed to update statistics for {table_name}: {e}")

if __name__ == "__main__":
    update_vector_table_stats()

