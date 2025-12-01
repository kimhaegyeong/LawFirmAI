# -*- coding: utf-8 -*-
"""
FTS5 테이블 생성 및 데이터 동기화 스크립트
"""

import sys
import os
import sqlite3
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from lawfirm_langgraph.core.utils.config import Config
except ImportError:
    # 직접 경로로 시도
    sys.path.insert(0, str(project_root / "lawfirm_langgraph"))
    from core.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fts5_tables(db_path: str):
    """FTS5 테이블 생성 및 데이터 동기화"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info(f"데이터베이스 연결: {db_path}")
        
        # 1. assembly_articles_fts 테이블 생성
        logger.info("assembly_articles_fts 테이블 생성 중...")
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS assembly_articles_fts USING fts5(
                article_number,
                article_title,
                article_content,
                content='assembly_articles',
                content_rowid='id'
            )
        ''')
        logger.info("✅ assembly_articles_fts 테이블 생성 완료")
        
        # 2. assembly_laws_fts 테이블 생성
        logger.info("assembly_laws_fts 테이블 생성 중...")
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS assembly_laws_fts USING fts5(
                law_name,
                full_text,
                summary,
                content='assembly_laws',
                content_rowid='id'
            )
        ''')
        logger.info("✅ assembly_laws_fts 테이블 생성 완료")
        
        # 3. 테이블 구조 확인
        logger.info("테이블 구조 확인 중...")
        
        # assembly_articles 테이블 구조 확인
        cursor.execute("PRAGMA table_info(assembly_articles)")
        articles_columns = {row[1]: row[2] for row in cursor.fetchall()}
        logger.info(f"assembly_articles 컬럼: {list(articles_columns.keys())}")
        
        # assembly_laws 테이블 구조 확인
        cursor.execute("PRAGMA table_info(assembly_laws)")
        laws_columns = {row[1]: row[2] for row in cursor.fetchall()}
        logger.info(f"assembly_laws 컬럼: {list(laws_columns.keys())}")
        
        # 4. 기존 데이터 동기화
        logger.info("기존 데이터 동기화 중...")
        
        # assembly_articles 데이터 동기화
        # rowid 컬럼 확인
        if 'id' in articles_columns:
            id_column = 'id'
        elif 'rowid' in articles_columns:
            id_column = 'rowid'
        else:
            # rowid는 SQLite의 내장 컬럼
            id_column = 'rowid'
        
        try:
            cursor.execute(f'''
                INSERT INTO assembly_articles_fts(rowid, article_number, article_title, article_content)
                SELECT {id_column}, article_number, article_title, article_content
                FROM assembly_articles
                WHERE {id_column} NOT IN (SELECT rowid FROM assembly_articles_fts)
            ''')
            articles_synced = cursor.rowcount
            logger.info(f"✅ assembly_articles 동기화 완료: {articles_synced}개 레코드")
        except Exception as e:
            logger.warning(f"assembly_articles 동기화 실패 (테이블이 비어있을 수 있음): {e}")
            articles_synced = 0
        
        # assembly_laws 데이터 동기화
        if 'id' in laws_columns:
            id_column = 'id'
        elif 'rowid' in laws_columns:
            id_column = 'rowid'
        else:
            id_column = 'rowid'
        
        try:
            cursor.execute(f'''
                INSERT INTO assembly_laws_fts(rowid, law_name, full_text, summary)
                SELECT {id_column}, law_name, full_text, summary
                FROM assembly_laws
                WHERE {id_column} NOT IN (SELECT rowid FROM assembly_laws_fts)
            ''')
            laws_synced = cursor.rowcount
            logger.info(f"✅ assembly_laws 동기화 완료: {laws_synced}개 레코드")
        except Exception as e:
            logger.warning(f"assembly_laws 동기화 실패 (테이블이 비어있을 수 있음): {e}")
            laws_synced = 0
        
        # 5. 트리거 생성 (자동 동기화)
        logger.info("트리거 생성 중...")
        
        # assembly_articles 트리거
        # rowid는 SQLite의 내장 컬럼이므로 직접 사용
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_articles_fts_ai 
            AFTER INSERT ON assembly_articles BEGIN
                INSERT INTO assembly_articles_fts(rowid, article_number, article_title, article_content)
                VALUES (new.rowid, new.article_number, new.article_title, new.article_content);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_articles_fts_ad 
            AFTER DELETE ON assembly_articles BEGIN
                INSERT INTO assembly_articles_fts(assembly_articles_fts, rowid, article_number, article_title, article_content)
                VALUES ('delete', old.rowid, old.article_number, old.article_title, old.article_content);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_articles_fts_au 
            AFTER UPDATE ON assembly_articles BEGIN
                INSERT INTO assembly_articles_fts(assembly_articles_fts, rowid, article_number, article_title, article_content)
                VALUES ('delete', old.rowid, old.article_number, old.article_title, old.article_content);
                INSERT INTO assembly_articles_fts(rowid, article_number, article_title, article_content)
                VALUES (new.rowid, new.article_number, new.article_title, new.article_content);
            END
        ''')
        
        # assembly_laws 트리거
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_laws_fts_ai 
            AFTER INSERT ON assembly_laws BEGIN
                INSERT INTO assembly_laws_fts(rowid, law_name, full_text, summary)
                VALUES (new.rowid, new.law_name, new.full_text, new.summary);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_laws_fts_ad 
            AFTER DELETE ON assembly_laws BEGIN
                INSERT INTO assembly_laws_fts(assembly_laws_fts, rowid, law_name, full_text, summary)
                VALUES ('delete', old.rowid, old.law_name, old.full_text, old.summary);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS assembly_laws_fts_au 
            AFTER UPDATE ON assembly_laws BEGIN
                INSERT INTO assembly_laws_fts(assembly_laws_fts, rowid, law_name, full_text, summary)
                VALUES ('delete', old.rowid, old.law_name, old.full_text, old.summary);
                INSERT INTO assembly_laws_fts(rowid, law_name, full_text, summary)
                VALUES (new.rowid, new.law_name, new.full_text, new.summary);
            END
        ''')
        
        logger.info("✅ 트리거 생성 완료")
        
        conn.commit()
        conn.close()
        
        logger.info("=" * 80)
        logger.info("FTS5 테이블 생성 및 동기화 완료!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"FTS5 테이블 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_fts5_tables(db_path: str):
    """FTS5 테이블 존재 여부 확인"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        tables = ['assembly_articles_fts', 'assembly_laws_fts']
        results = {}
        
        for table in tables:
            try:
                cursor.execute('''
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                ''', (table,))
                exists = cursor.fetchone() is not None
                results[table] = exists
                
                if exists:
                    # 레코드 수 확인
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    logger.info(f"✅ {table}: 존재 (레코드 수: {count})")
                else:
                    logger.warning(f"❌ {table}: 존재하지 않음")
            except Exception as e:
                logger.warning(f"❌ {table}: 확인 실패 - {e}")
                results[table] = False
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"FTS5 테이블 확인 실패: {e}")
        return {}


if __name__ == "__main__":
    config = Config()
    db_path = config.database_path
    
    logger.info("=" * 80)
    logger.info("FTS5 테이블 생성 및 동기화 스크립트")
    logger.info("=" * 80)
    logger.info(f"데이터베이스 경로: {db_path}")
    
    # 테이블 존재 여부 확인
    logger.info("\n[1단계] 기존 FTS5 테이블 확인")
    verify_fts5_tables(db_path)
    
    # FTS5 테이블 생성 및 동기화
    logger.info("\n[2단계] FTS5 테이블 생성 및 동기화")
    success = create_fts5_tables(db_path)
    
    if success:
        # 최종 확인
        logger.info("\n[3단계] 최종 확인")
        verify_fts5_tables(db_path)
    else:
        logger.error("FTS5 테이블 생성 실패")

