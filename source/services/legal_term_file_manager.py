# -*- coding: utf-8 -*-
"""
Legal Term File Manager
법률용어 파일 관리 시스템
"""

import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class LegalTermFileManager:
    """법률용어 파일 관리 시스템"""
    
    def __init__(self, base_dir: str):
        """
        파일 관리자 초기화
        
        Args:
            base_dir: 법률용어 파일들이 저장된 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.processing_dir = self.base_dir / "processing"
        self.complete_dir = self.base_dir / "complete"
        self.failed_dir = self.base_dir / "failed"
        self.archive_dir = self.base_dir / "archive"
        
        # 디렉토리 생성
        self._create_directories()
        
        logger.info(f"LegalTermFileManager 초기화 완료: {self.base_dir}")
        
    def _create_directories(self):
        """필요한 디렉토리들 생성"""
        directories = [
            self.processing_dir,
            self.complete_dir, 
            self.failed_dir,
            self.archive_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"디렉토리 생성/확인: {directory}")
            
    def move_to_processing(self, file_path: Path) -> Path:
        """
        파일을 processing 폴더로 이동
        
        Args:
            file_path: 이동할 파일 경로
            
        Returns:
            이동된 파일의 새 경로
        """
        target_path = self.processing_dir / file_path.name
        
        try:
            shutil.move(str(file_path), str(target_path))
            logger.info(f"파일을 processing으로 이동: {file_path.name}")
            return target_path
        except Exception as e:
            logger.error(f"파일 이동 실패: {e}")
            raise
            
    def move_to_complete(self, file_path: Path) -> Path:
        """
        파일을 complete 폴더로 이동 (날짜별 정리)
        
        Args:
            file_path: 이동할 파일 경로
            
        Returns:
            이동된 파일의 새 경로
        """
        today = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.complete_dir / today
        date_dir.mkdir(exist_ok=True)
        
        target_path = date_dir / file_path.name
        
        try:
            shutil.move(str(file_path), str(target_path))
            logger.info(f"파일을 complete로 이동: {file_path.name}")
            return target_path
        except Exception as e:
            logger.error(f"파일 이동 실패: {e}")
            raise
            
    def move_to_failed(self, file_path: Path, error_message: str = "") -> Path:
        """
        파일을 failed 폴더로 이동
        
        Args:
            file_path: 이동할 파일 경로
            error_message: 오류 메시지
            
        Returns:
            이동된 파일의 새 경로
        """
        # 오류 메시지를 파일명에 포함
        if error_message:
            error_suffix = f"_error_{datetime.now().strftime('%H%M%S')}"
            new_name = file_path.stem + error_suffix + file_path.suffix
        else:
            new_name = file_path.name
            
        target_path = self.failed_dir / new_name
        
        try:
            shutil.move(str(file_path), str(target_path))
            logger.warning(f"파일을 failed로 이동: {file_path.name} - {error_message}")
            return target_path
        except Exception as e:
            logger.error(f"파일 이동 실패: {e}")
            raise
            
    def archive_old_files(self, days_old: int = 30):
        """
        오래된 완료 파일들을 archive로 이동
        
        Args:
            days_old: 아카이브할 기준 일수 (기본 30일)
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_count = 0
        
        for date_dir in self.complete_dir.iterdir():
            if date_dir.is_dir():
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        archive_path = self.archive_dir / date_dir.name
                        shutil.move(str(date_dir), str(archive_path))
                        archived_count += 1
                        logger.info(f"오래된 파일들을 archive로 이동: {date_dir.name}")
                except ValueError:
                    # 날짜 형식이 아닌 디렉토리는 무시
                    continue
                    
        logger.info(f"총 {archived_count}개 디렉토리를 archive로 이동")
        
    def is_file_processed(self, file_name: str) -> bool:
        """
        파일이 이미 처리되었는지 확인
        
        Args:
            file_name: 확인할 파일명
            
        Returns:
            처리 여부
        """
        # complete 폴더에서 확인
        for date_dir in self.complete_dir.iterdir():
            if date_dir.is_dir() and (date_dir / file_name).exists():
                return True
                
        # failed 폴더에서 확인 (실패한 파일도 처리된 것으로 간주)
        if (self.failed_dir / file_name).exists():
            return True
            
        return False
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        처리 통계 조회
        
        Returns:
            처리 통계 딕셔너리
        """
        stats = {
            "processing": len(list(self.processing_dir.glob("*.json"))),
            "complete_today": 0,
            "complete_total": 0,
            "failed": len(list(self.failed_dir.glob("*.json"))),
            "archive": 0
        }
        
        # 오늘 완료된 파일 수
        today_dir = self.complete_dir / datetime.now().strftime("%Y-%m-%d")
        if today_dir.exists():
            stats["complete_today"] = len(list(today_dir.glob("*.json")))
            
        # 전체 완료된 파일 수
        for date_dir in self.complete_dir.iterdir():
            if date_dir.is_dir():
                stats["complete_total"] += len(list(date_dir.glob("*.json")))
                
        # 아카이브된 파일 수
        for date_dir in self.archive_dir.iterdir():
            if date_dir.is_dir():
                stats["archive"] += len(list(date_dir.glob("*.json")))
                
        stats["total_processed"] = stats["complete_total"] + stats["failed"] + stats["archive"]
        stats["success_rate"] = (stats["complete_total"] / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0
        
        return stats
        
    def print_daily_report(self):
        """일일 처리 리포트 출력"""
        stats = self.get_processing_stats()
        
        print("=== 법률용어 파일 처리 현황 ===")
        print(f"처리 중: {stats['processing']}개")
        print(f"오늘 완료: {stats['complete_today']}개")
        print(f"총 완료: {stats['complete_total']}개")
        print(f"실패: {stats['failed']}개")
        print(f"아카이브: {stats['archive']}개")
        print(f"성공률: {stats['success_rate']:.1f}%")
        
    def scan_new_files(self) -> List[Path]:
        """
        새로 추가된 파일들 스캔 (processing, complete, failed 제외)
        
        Returns:
            새로 발견된 파일 목록
        """
        all_files = list(self.base_dir.glob("legal_term_detail_batch_*.json"))
        new_files = []
        
        for file_path in all_files:
            if not self.is_file_processed(file_path.name):
                new_files.append(file_path)
                
        logger.info(f"새로 발견된 파일: {len(new_files)}개")
        return new_files
