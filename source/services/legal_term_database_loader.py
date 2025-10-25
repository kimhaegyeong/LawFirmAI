# -*- coding: utf-8 -*-
"""
Legal Term Database Loader with File Management
파일 관리가 통합된 법률용어 데이터베이스 적재기
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

try:
    from .legal_term_file_manager import LegalTermFileManager
except ImportError:
    # 절대 경로로 import 시도
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from legal_term_file_manager import LegalTermFileManager

logger = logging.getLogger(__name__)


class LegalTermDuplicatePrevention:
    """법률용어 중복 방지 시스템"""
    
    def __init__(self, db_path: str):
        """
        중복 방지 시스템 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._create_tables()
        
    def _create_tables(self):
        """필요한 테이블들 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 파일 처리 이력 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_term_file_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    파일명 TEXT UNIQUE NOT NULL,
                    파일크기 INTEGER,
                    파일수정일시 TIMESTAMP,
                    처리상태 TEXT DEFAULT 'pending',
                    처리시작일시 TIMESTAMP,
                    처리완료일시 TIMESTAMP,
                    처리된용어수 INTEGER DEFAULT 0,
                    실패한용어수 INTEGER DEFAULT 0,
                    오류메시지 TEXT,
                    체크섬 TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 용어별 처리 이력 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_term_processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    법령용어일련번호 INTEGER NOT NULL,
                    법령용어ID TEXT,
                    파일명 TEXT NOT NULL,
                    처리상태 TEXT DEFAULT 'pending',
                    처리일시 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    오류메시지 TEXT,
                    UNIQUE(법령용어일련번호, 파일명)
                )
            """)
            
            # 법률용어 상세 테이블 (기존 테이블과 호환)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS legal_term_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    법령용어일련번호 INTEGER UNIQUE NOT NULL,
                    법령용어ID TEXT,
                    법령용어명_한글 TEXT NOT NULL,
                    법령용어명_한자 TEXT,
                    법령용어코드 INTEGER,
                    법령용어코드명 TEXT,
                    출처 TEXT,
                    법령용어정의 TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    quality_score REAL DEFAULT 0.0,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # 기존 테이블에 컬럼이 없는 경우 추가
            try:
                cursor.execute("ALTER TABLE legal_term_details ADD COLUMN processed_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # 컬럼이 이미 존재하는 경우
                
            try:
                cursor.execute("ALTER TABLE legal_term_details ADD COLUMN vectorized_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # 컬럼이 이미 존재하는 경우
            
            conn.commit()
            
    def is_file_processed(self, file_name: str) -> bool:
        """
        파일이 이미 처리되었는지 확인
        
        Args:
            file_name: 확인할 파일명
            
        Returns:
            처리 여부
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 처리상태 FROM legal_term_file_history 
                WHERE 파일명 = ? AND 처리상태 = 'completed'
            """, (file_name,))
            
            return cursor.fetchone() is not None
    
    def is_term_processed(self, 법령용어일련번호: int) -> bool:
        """
        용어가 이미 처리되었는지 확인
        
        Args:
            법령용어일련번호: 확인할 용어 일련번호
            
        Returns:
            처리 여부
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM legal_term_details 
                WHERE 법령용어일련번호 = ?
            """, (법령용어일련번호,))
            
            return cursor.fetchone()[0] > 0
    
    def mark_file_processing(self, file_path: Path):
        """
        파일 처리 시작 표시
        
        Args:
            file_path: 처리할 파일 경로
        """
        file_name = file_path.name
        file_size = file_path.stat().st_size
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO legal_term_file_history 
                (파일명, 파일크기, 파일수정일시, 처리상태, 처리시작일시)
                VALUES (?, ?, ?, 'processing', CURRENT_TIMESTAMP)
            """, (file_name, file_size, file_mtime))
            conn.commit()
    
    def mark_file_completed(self, file_name: str, processed_count: int, failed_count: int):
        """
        파일 처리 완료 표시
        
        Args:
            file_name: 파일명
            processed_count: 처리된 용어 수
            failed_count: 실패한 용어 수
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE legal_term_file_history 
                SET 처리상태 = 'completed', 처리완료일시 = CURRENT_TIMESTAMP,
                    처리된용어수 = ?, 실패한용어수 = ?
                WHERE 파일명 = ?
            """, (processed_count, failed_count, file_name))
            conn.commit()
    
    def mark_file_failed(self, file_name: str, error_message: str):
        """
        파일 처리 실패 표시
        
        Args:
            file_name: 파일명
            error_message: 오류 메시지
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE legal_term_file_history 
                SET 처리상태 = 'failed', 처리완료일시 = CURRENT_TIMESTAMP,
                    오류메시지 = ?
                WHERE 파일명 = ?
            """, (error_message, file_name))
            conn.commit()


class LegalTermDatabaseLoaderWithFileManagement:
    """파일 관리가 통합된 법률용어 데이터베이스 적재기"""
    
    def __init__(self, db_path: str, base_dir: str):
        """
        데이터베이스 적재기 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            base_dir: 법률용어 파일들이 저장된 기본 디렉토리
        """
        self.db_path = db_path
        self.file_manager = LegalTermFileManager(base_dir)
        self.duplicate_prevention = LegalTermDuplicatePrevention(db_path)
        
        logger.info(f"LegalTermDatabaseLoaderWithFileManagement 초기화 완료")
        
    def load_and_move_files(self):
        """파일 적재 후 완료 폴더로 이동"""
        # 1. 현재 디렉토리에서 새 파일들 스캔
        new_files = self._scan_new_files()
        
        if not new_files:
            logger.info("새로 추가된 파일이 없습니다.")
            return
            
        logger.info(f"새로 발견된 파일 {len(new_files)}개 처리 시작")
        
        # 2. 각 파일 처리
        for file_path in new_files:
            try:
                # 파일을 processing으로 이동
                processing_path = self.file_manager.move_to_processing(file_path)
                
                # 데이터베이스에 적재
                success = self._load_file_to_database(processing_path)
                
                if success:
                    # 성공 시 complete로 이동
                    self.file_manager.move_to_complete(processing_path)
                    logger.info(f"파일 처리 완료: {file_path.name}")
                else:
                    # 실패 시 failed로 이동
                    self.file_manager.move_to_failed(processing_path, "데이터베이스 적재 실패")
                    
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생 {file_path}: {e}")
                # 오류 발생 시 failed로 이동
                try:
                    self.file_manager.move_to_failed(file_path, str(e))
                except:
                    pass  # 이동도 실패하면 로그만 남김
                    
    def _scan_new_files(self) -> List[Path]:
        """새로 추가된 파일들 스캔 (processing, complete, failed 제외)"""
        all_files = list(self.file_manager.base_dir.glob("legal_term_detail_batch_*.json"))
        
        # 이미 처리된 파일들 제외
        new_files = []
        for file_path in all_files:
            if not self._is_file_already_processed(file_path):
                new_files.append(file_path)
                
        return new_files
        
    def _is_file_already_processed(self, file_path: Path) -> bool:
        """파일이 이미 처리되었는지 확인"""
        file_name = file_path.name
        
        # 데이터베이스에서 확인
        if self.duplicate_prevention.is_file_processed(file_name):
            return True
            
        # 파일 시스템에서 확인
        if self.file_manager.is_file_processed(file_name):
            return True
            
        return False
        
    def _load_file_to_database(self, file_path: Path) -> bool:
        """파일을 데이터베이스에 적재"""
        file_name = file_path.name
        
        try:
            # 파일 처리 시작 표시
            self.duplicate_prevention.mark_file_processing(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"빈 파일 또는 잘못된 형식: {file_path}")
                self.duplicate_prevention.mark_file_failed(file_name, "빈 파일 또는 잘못된 형식")
                return False
            
            processed_count = 0
            failed_count = 0
            
            # 각 용어 처리
            for term in data:
                try:
                    if self._process_term(term, file_name):
                        processed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"용어 처리 실패: {e}")
                    failed_count += 1
            
            # 파일 처리 완료 표시
            self.duplicate_prevention.mark_file_completed(file_name, processed_count, failed_count)
            
            logger.info(f"데이터베이스 적재 완료: {processed_count}개 용어, {failed_count}개 실패")
            return processed_count > 0
            
        except Exception as e:
            logger.error(f"데이터베이스 적재 실패: {e}")
            self.duplicate_prevention.mark_file_failed(file_name, str(e))
            return False
    
    def _process_term(self, term: Dict, file_name: str) -> bool:
        """개별 용어 처리"""
        법령용어일련번호 = term.get('법령용어일련번호')
        
        if not 법령용어일련번호:
            logger.warning(f"법령용어일련번호가 없는 용어: {term}")
            return False
        
        # 중복 체크
        if self.duplicate_prevention.is_term_processed(법령용어일련번호):
            logger.debug(f"용어 이미 처리됨: {법령용어일련번호}")
            return True
        
        # 데이터베이스에 삽입
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO legal_term_details 
                    (법령용어일련번호, 법령용어명_한글, 법령용어명_한자, 
                     법령용어코드, 법령용어코드명, 출처, 법령용어정의, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    term.get('법령용어일련번호'),
                    term.get('법령용어명_한글'),
                    term.get('법령용어명_한자'),
                    term.get('법령용어코드'),
                    term.get('법령용어코드명'),
                    term.get('출처'),
                    term.get('법령용어정의')
                ))
                conn.commit()
                
            logger.debug(f"용어 처리 완료: {법령용어일련번호}")
            return True
            
        except Exception as e:
            logger.error(f"용어 데이터베이스 삽입 실패: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 조회"""
        file_stats = self.file_manager.get_processing_stats()
        
        # 데이터베이스 통계 추가
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 총 용어 수
            cursor.execute("SELECT COUNT(*) FROM legal_term_details")
            total_terms = cursor.fetchone()[0]
            
            # 오늘 처리된 용어 수
            cursor.execute("""
                SELECT COUNT(*) FROM legal_term_details 
                WHERE DATE(processed_at) = DATE('now')
            """)
            today_terms = cursor.fetchone()[0]
            
            # 처리 상태별 파일 수
            cursor.execute("""
                SELECT 처리상태, COUNT(*) FROM legal_term_file_history 
                GROUP BY 처리상태
            """)
            status_counts = dict(cursor.fetchall())
            
        file_stats.update({
            "total_terms": total_terms,
            "today_terms": today_terms,
            "db_status_counts": status_counts
        })
        
        return file_stats
    
    def reprocess_failed_files(self):
        """실패한 파일들을 다시 처리"""
        failed_files = list(self.file_manager.failed_dir.glob("*.json"))
        
        if not failed_files:
            logger.info("재처리할 실패 파일이 없습니다.")
            return
            
        logger.info(f"실패한 파일 {len(failed_files)}개 재처리 시작")
        
        reprocessed_count = 0
        still_failed_count = 0
        
        for file_path in failed_files:
            try:
                # 파일을 processing으로 이동
                processing_path = self.file_manager.move_to_processing(file_path)
                
                # 데이터베이스에 적재
                success = self._load_file_to_database(processing_path)
                
                if success:
                    # 성공 시 complete로 이동
                    self.file_manager.move_to_complete(processing_path)
                    reprocessed_count += 1
                    logger.info(f"파일 재처리 성공: {file_path.name}")
                else:
                    # 실패 시 다시 failed로 이동
                    self.file_manager.move_to_failed(processing_path, "재처리 실패")
                    still_failed_count += 1
                    logger.warning(f"파일 재처리 실패: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"파일 재처리 중 오류 발생 {file_path}: {e}")
                still_failed_count += 1
                # 오류 발생 시 원본 파일 유지
                try:
                    if processing_path.exists():
                        self.file_manager.move_to_failed(processing_path, str(e))
                except:
                    pass
                    
        logger.info(f"재처리 완료: 성공 {reprocessed_count}개, 여전히 실패 {still_failed_count}개")
        
    def clear_failed_files(self):
        """실패한 파일들을 삭제 (주의: 데이터 손실 가능)"""
        failed_files = list(self.file_manager.failed_dir.glob("*.json"))
        
        if not failed_files:
            logger.info("삭제할 실패 파일이 없습니다.")
            return
            
        logger.warning(f"실패한 파일 {len(failed_files)}개를 삭제합니다.")
        
        for file_path in failed_files:
            try:
                file_path.unlink()
                logger.info(f"파일 삭제: {file_path.name}")
            except Exception as e:
                logger.error(f"파일 삭제 실패 {file_path}: {e}")
                
        logger.info("실패 파일 삭제 완료")
