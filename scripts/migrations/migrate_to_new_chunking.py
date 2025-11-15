"""
기존 데이터 마이그레이션 스크립트

기존 청크에 기본 메타데이터 추가 및 선택적 하이브리드 청킹 재생성
"""
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def migrate_existing_chunks(db_path: str, add_default_metadata: bool = True):
    """
    기존 청크에 기본 메타데이터 추가
    
    Args:
        db_path: 데이터베이스 경로
        add_default_metadata: 기본 메타데이터 추가 여부
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # 컬럼 존재 여부 확인
        cursor = conn.execute("PRAGMA table_info(text_chunks)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # 필요한 컬럼이 없으면 마이그레이션 스크립트 실행 필요
        required_columns = ['chunking_strategy', 'chunk_size_category', 'query_type', 'chunk_group_id', 'original_document_id']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"⚠️  필요한 컬럼이 없습니다: {missing_columns}")
            print("   먼저 scripts/migrations/002_add_chunking_metadata.sql을 실행하세요.")
            return False
        
        if add_default_metadata:
            print("기존 청크에 기본 메타데이터 추가 중...")
            
            # 기존 청크에 기본값 설정
            cursor = conn.execute("""
                UPDATE text_chunks 
                SET chunking_strategy = COALESCE(chunking_strategy, 'standard'),
                    chunk_size_category = COALESCE(chunk_size_category, 
                        CASE 
                            WHEN LENGTH(text) < 800 THEN 'small'
                            WHEN LENGTH(text) < 1500 THEN 'medium'
                            ELSE 'large'
                        END),
                    original_document_id = COALESCE(original_document_id, source_id)
                WHERE chunking_strategy IS NULL 
                   OR chunk_size_category IS NULL
                   OR original_document_id IS NULL
            """)
            
            updated_count = cursor.rowcount
            conn.commit()
            print(f"✅ {updated_count}개의 청크에 메타데이터가 추가되었습니다.")
        
        return True
    
    except Exception as e:
        print(f"❌ 마이그레이션 중 오류 발생: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='기존 데이터 마이그레이션')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--skip-metadata', action='store_true', help='메타데이터 추가 건너뛰기')
    
    args = parser.parse_args()
    
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        sys.exit(1)
    
    print(f"데이터베이스: {db_path}")
    print(f"메타데이터 추가: {not args.skip_metadata}")
    print()
    
    success = migrate_existing_chunks(str(db_path), add_default_metadata=not args.skip_metadata)
    
    if success:
        print("\n✅ 마이그레이션이 완료되었습니다.")
    else:
        print("\n❌ 마이그레이션에 실패했습니다.")
        sys.exit(1)


if __name__ == '__main__':
    main()

