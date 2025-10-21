#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 데이터 백업 및 정리 스크립트
Raw 데이터 재적재 전 기존 데이터를 백업하고 정리합니다.
"""

import os
import sys
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
import logging

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_backup_clear.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def backup_existing_database(db_path: str = "data/lawfirm.db") -> str:
    """
    기존 데이터베이스 백업
    
    Args:
        db_path (str): 데이터베이스 파일 경로
        
    Returns:
        str: 백업 파일 경로
    """
    if not os.path.exists(db_path):
        logger.warning(f"Database file not found: {db_path}")
        return None
    
    # 백업 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        # 데이터베이스 백업
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        
        # 백업 크기 확인
        original_size = os.path.getsize(db_path)
        backup_size = os.path.getsize(backup_path)
        
        logger.info(f"Original size: {original_size:,} bytes")
        logger.info(f"Backup size: {backup_size:,} bytes")
        
        if original_size == backup_size:
            logger.info("✅ Backup completed successfully")
            return backup_path
        else:
            logger.error("❌ Backup size mismatch")
            return None
            
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None


def clear_existing_data(db_path: str = "data/lawfirm.db") -> bool:
    """
    기존 데이터 완전 삭제
    
    Args:
        db_path (str): 데이터베이스 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 삭제할 테이블 목록
            tables_to_clear = [
                'assembly_articles',
                'assembly_laws', 
                'precedent_cases',
                'precedent_sections',
                'precedent_parties',
                'documents',
                'laws',
                'precedents',
                'chat_history',
                'processed_files'
            ]
            
            # FTS 테이블 목록
            fts_tables_to_clear = [
                'assembly_laws_fts',
                'assembly_articles_fts',
                'fts_precedent_cases',
                'fts_precedent_sections',
                'fts_assembly_laws',
                'fts_assembly_articles'
            ]
            
            logger.info("🗑️ Clearing existing data...")
            
            # 일반 테이블 데이터 삭제
            for table in tables_to_clear:
                try:
                    cursor.execute(f'DELETE FROM {table}')
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleared {table}: {deleted_count} records")
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e):
                        logger.info(f"Table {table} does not exist, skipping")
                    else:
                        logger.error(f"Error clearing {table}: {e}")
            
            # FTS 테이블 정리
            for table in fts_tables_to_clear:
                try:
                    cursor.execute(f'DELETE FROM {table}')
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleared FTS table {table}: {deleted_count} records")
                except sqlite3.OperationalError as e:
                    if "no such table" in str(e):
                        logger.info(f"FTS table {table} does not exist, skipping")
                    else:
                        logger.error(f"Error clearing FTS table {table}: {e}")
            
            # 시퀀스 테이블 정리
            try:
                cursor.execute('DELETE FROM sqlite_sequence')
                logger.info("Cleared sqlite_sequence")
            except sqlite3.OperationalError:
                logger.info("sqlite_sequence table does not exist, skipping")
            
            # 통계 테이블 정리
            try:
                cursor.execute('DELETE FROM sqlite_stat1')
                logger.info("Cleared sqlite_stat1")
            except sqlite3.OperationalError:
                logger.info("sqlite_stat1 table does not exist, skipping")
            
            conn.commit()
            logger.info("✅ All existing data cleared successfully")
            
            # 테이블 상태 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            remaining_tables = cursor.fetchall()
            logger.info(f"Remaining tables: {len(remaining_tables)}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error clearing existing data: {e}")
        return False


def verify_database_structure(db_path: str = "data/lawfirm.db") -> bool:
    """
    데이터베이스 구조 확인
    
    Args:
        db_path (str): 데이터베이스 파일 경로
        
    Returns:
        bool: 구조 확인 성공 여부
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 테이블 목록 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            logger.info("📋 Database structure verification:")
            logger.info(f"Total tables: {len(tables)}")
            
            # 주요 테이블 확인
            required_tables = [
                'assembly_laws',
                'assembly_articles',
                'precedent_cases',
                'precedent_sections'
            ]
            
            for table in required_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  {table}: {count} records")
            
            # FTS 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%fts%'")
            fts_tables = cursor.fetchall()
            logger.info(f"FTS tables: {len(fts_tables)}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error verifying database structure: {e}")
        return False


def main():
    """메인 함수"""
    logger.info("🚀 Starting database backup and clear process...")
    
    db_path = "data/lawfirm.db"
    
    # Phase 1: 데이터베이스 백업
    logger.info("\n📋 Phase 1: Backing up existing database...")
    backup_path = backup_existing_database(db_path)
    
    if not backup_path:
        logger.error("❌ Backup failed, aborting process")
        return False
    
    # Phase 2: 기존 데이터 정리
    logger.info("\n🗑️ Phase 2: Clearing existing data...")
    clear_success = clear_existing_data(db_path)
    
    if not clear_success:
        logger.error("❌ Data clearing failed")
        return False
    
    # Phase 3: 구조 확인
    logger.info("\n🔍 Phase 3: Verifying database structure...")
    verify_success = verify_database_structure(db_path)
    
    if not verify_success:
        logger.error("❌ Database structure verification failed")
        return False
    
    logger.info("\n✅ Database backup and clear process completed successfully!")
    logger.info(f"📁 Backup file: {backup_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Process completed successfully!")
        print("You can now proceed with raw data reprocessing.")
    else:
        print("\n❌ Process failed. Please check the logs.")
        sys.exit(1)
