# -*- coding: utf-8 -*-
"""
assembly_articles 테이블 마이그레이션 스크립트
article_type 등 컬럼 추가
"""

import sys
import sqlite3
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.utils.config import Config
except ImportError:
    sys.path.insert(0, str(project_root / "lawfirm_langgraph"))
    from core.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """컬럼 존재 여부 확인"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return column_name in columns
    except Exception as e:
        logger.error(f"Error checking column {column_name} in {table_name}: {e}")
        return False


def add_column_if_not_exists(cursor, table_name: str, column_name: str, column_definition: str) -> bool:
    """컬럼 추가 (존재하지 않는 경우)"""
    try:
        if not check_column_exists(cursor, table_name, column_name):
            sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
            cursor.execute(sql)
            logger.info(f"✅ {table_name}.{column_name} 컬럼 추가 완료")
            return True
        else:
            logger.info(f"⏭️  {table_name}.{column_name} 컬럼이 이미 존재합니다")
            return False
    except Exception as e:
        logger.error(f"❌ {table_name}.{column_name} 컬럼 추가 실패: {e}")
        return False


def migrate_assembly_articles(db_path: str):
    """assembly_articles 테이블 마이그레이션"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info(f"데이터베이스 연결: {db_path}")
        
        # 추가할 컬럼들
        columns_to_add = [
            ("article_type", "TEXT DEFAULT 'main'"),
            ("parsing_quality_score", "REAL DEFAULT 0.0"),
            ("word_count", "INTEGER DEFAULT 0"),
            ("char_count", "INTEGER DEFAULT 0"),
            ("ml_enhanced", "BOOLEAN DEFAULT 0"),
        ]
        
        logger.info("assembly_articles 테이블 마이그레이션 시작...")
        
        success_count = 0
        for column_name, column_definition in columns_to_add:
            if add_column_if_not_exists(cursor, "assembly_articles", column_name, column_definition):
                success_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info("=" * 80)
        logger.info(f"마이그레이션 완료: {success_count}개 컬럼 추가")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    config = Config()
    db_path = config.database_path
    
    logger.info("=" * 80)
    logger.info("assembly_articles 테이블 마이그레이션 스크립트")
    logger.info("=" * 80)
    logger.info(f"데이터베이스 경로: {db_path}")
    
    success = migrate_assembly_articles(db_path)
    
    if success:
        logger.info("✅ 마이그레이션 성공")
    else:
        logger.error("❌ 마이그레이션 실패")

