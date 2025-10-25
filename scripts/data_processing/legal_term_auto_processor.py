# -*- coding: utf-8 -*-
"""
Legal Term Auto Processor
법률용어 자동 처리 시스템
"""

import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트 경로 추가
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from source.services.legal_term_database_loader import LegalTermDatabaseLoaderWithFileManagement
from source.services.legal_term_file_manager import LegalTermFileManager

logger = logging.getLogger(__name__)


class LegalTermAutoProcessor:
    """법률용어 자동 처리 시스템"""
    
    def __init__(self, db_path: str, base_dir: str):
        """
        자동 처리기 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            base_dir: 법률용어 파일들이 저장된 기본 디렉토리
        """
        self.db_path = db_path
        self.base_dir = base_dir
        self.loader = LegalTermDatabaseLoaderWithFileManagement(db_path, base_dir)
        self.file_manager = LegalTermFileManager(base_dir)
        
        # 처리 통계
        self.stats = {
            "total_processed": 0,
            "total_failed": 0,
            "last_check_time": None,
            "start_time": datetime.now()
        }
        
        logger.info(f"LegalTermAutoProcessor 초기화 완료")
        
    def run_continuous_processing(self, check_interval: int = 300, archive_days: int = 30):
        """
        지속적인 파일 처리 실행
        
        Args:
            check_interval: 파일 체크 간격 (초, 기본 5분)
            archive_days: 아카이브 기준 일수 (기본 30일)
        """
        logger.info(f"지속적인 파일 처리 시작 (체크 간격: {check_interval}초)")
        
        try:
            while True:
                try:
                    # 새 파일들 처리
                    self._process_new_files()
                    
                    # 오래된 파일들 아카이브
                    if archive_days > 0:
                        self.loader.archive_old_files(archive_days)
                    
                    # 통계 업데이트
                    self._update_stats()
                    
                    # 처리 현황 출력
                    self._print_status()
                    
                    # 대기
                    time.sleep(check_interval)
                    
                except KeyboardInterrupt:
                    logger.info("자동 처리 중단 요청됨")
                    break
                except Exception as e:
                    logger.error(f"자동 처리 중 오류: {e}")
                    time.sleep(60)  # 오류 시 1분 대기 후 재시도
                    
        except Exception as e:
            logger.error(f"자동 처리 시스템 오류: {e}")
        finally:
            self._print_final_stats()
            
    def run_single_processing(self):
        """한 번만 파일 처리 실행"""
        logger.info("단일 파일 처리 실행")
        
        try:
            # 새 파일들 처리
            self._process_new_files()
            
            # 통계 업데이트
            self._update_stats()
            
            # 처리 현황 출력
            self._print_status()
            
        except Exception as e:
            logger.error(f"단일 처리 중 오류: {e}")
            
    def reprocess_failed_files(self):
        """실패한 파일들 재처리"""
        logger.info("실패한 파일들 재처리 시작")
        
        try:
            # 실패한 파일들 재처리
            self.loader.reprocess_failed_files()
            
            # 통계 업데이트
            self._update_stats()
            
            # 처리 현황 출력
            self._print_status()
            
        except Exception as e:
            logger.error(f"재처리 중 오류: {e}")
            
    def clear_failed_files(self):
        """실패한 파일들 삭제"""
        logger.info("실패한 파일들 삭제 시작")
        
        try:
            # 실패한 파일들 삭제
            self.loader.clear_failed_files()
            
            # 통계 업데이트
            self._update_stats()
            
            # 처리 현황 출력
            self._print_status()
            
        except Exception as e:
            logger.error(f"삭제 중 오류: {e}")
            
    def _process_new_files(self):
        """새로운 파일들 처리"""
        self.loader.load_and_move_files()
        
    def _update_stats(self):
        """통계 업데이트"""
        current_stats = self.loader.get_processing_stats()
        
        self.stats.update({
            "total_processed": current_stats.get("complete_total", 0),
            "total_failed": current_stats.get("failed", 0),
            "last_check_time": datetime.now()
        })
        
    def _print_status(self):
        """처리 현황 출력"""
        stats = self.loader.get_processing_stats()
        
        print(f"\n=== 법률용어 자동 처리 현황 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        print(f"처리 중: {stats['processing']}개")
        print(f"오늘 완료: {stats['complete_today']}개")
        print(f"총 완료: {stats['complete_total']}개")
        print(f"실패: {stats['failed']}개")
        print(f"아카이브: {stats['archive']}개")
        print(f"성공률: {stats['success_rate']:.1f}%")
        print(f"총 용어 수: {stats['total_terms']}개")
        print(f"오늘 처리된 용어: {stats['today_terms']}개")
        
        # 실행 시간 계산
        runtime = datetime.now() - self.stats['start_time']
        print(f"실행 시간: {runtime}")
        
    def _print_final_stats(self):
        """최종 통계 출력"""
        runtime = datetime.now() - self.stats['start_time']
        
        print(f"\n=== 최종 처리 통계 ===")
        print(f"총 실행 시간: {runtime}")
        print(f"마지막 체크 시간: {self.stats['last_check_time']}")
        
        final_stats = self.loader.get_processing_stats()
        print(f"최종 완료 파일: {final_stats['complete_total']}개")
        print(f"최종 실패 파일: {final_stats['failed']}개")
        print(f"최종 성공률: {final_stats['success_rate']:.1f}%")


class LegalTermMonitor:
    """법률용어 처리 모니터링 시스템"""
    
    def __init__(self, db_path: str, base_dir: str):
        """
        모니터링 시스템 초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
            base_dir: 법률용어 파일들이 저장된 기본 디렉토리
        """
        self.db_path = db_path
        self.base_dir = base_dir
        self.file_manager = LegalTermFileManager(base_dir)
        
    def generate_daily_report(self) -> Dict[str, Any]:
        """일일 처리 리포트 생성"""
        stats = self.file_manager.get_processing_stats()
        
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "processing": stats['processing'],
            "complete_today": stats['complete_today'],
            "complete_total": stats['complete_total'],
            "failed": stats['failed'],
            "archive": stats['archive'],
            "success_rate": stats['success_rate'],
            "total_processed": stats['total_processed']
        }
        
        return report
        
    def print_detailed_report(self):
        """상세 리포트 출력"""
        stats = self.file_manager.get_processing_stats()
        
        print("=== 법률용어 처리 상세 현황 ===")
        print(f"처리 중인 파일: {stats['processing']}개")
        print(f"오늘 완료된 파일: {stats['complete_today']}개")
        print(f"총 완료된 파일: {stats['complete_total']}개")
        print(f"실패한 파일: {stats['failed']}개")
        print(f"아카이브된 파일: {stats['archive']}개")
        print(f"전체 처리된 파일: {stats['total_processed']}개")
        print(f"성공률: {stats['success_rate']:.1f}%")
        
        # 폴더별 파일 수 상세
        print("\n=== 폴더별 파일 현황 ===")
        print(f"Processing 폴더: {len(list(self.file_manager.processing_dir.glob('*.json')))}개")
        print(f"Complete 폴더 (오늘): {len(list((self.file_manager.complete_dir / datetime.now().strftime('%Y-%m-%d')).glob('*.json')))}개")
        print(f"Failed 폴더: {len(list(self.file_manager.failed_dir.glob('*.json')))}개")
        print(f"Archive 폴더: {len(list(self.file_manager.archive_dir.glob('*.json')))}개")
        
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 상태 체크"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "issues": []
        }
        
        # 폴더 존재 여부 체크
        required_dirs = [
            self.file_manager.processing_dir,
            self.file_manager.complete_dir,
            self.file_manager.failed_dir,
            self.file_manager.archive_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                health_status["issues"].append(f"필수 디렉토리 없음: {directory}")
                health_status["status"] = "warning"
                
        # 데이터베이스 파일 체크
        if not Path(self.db_path).exists():
            health_status["issues"].append(f"데이터베이스 파일 없음: {self.db_path}")
            health_status["status"] = "error"
            
        # 처리 중인 파일이 너무 많은지 체크
        processing_count = len(list(self.file_manager.processing_dir.glob('*.json')))
        if processing_count > 100:
            health_status["issues"].append(f"처리 중인 파일이 너무 많음: {processing_count}개")
            health_status["status"] = "warning"
            
        return health_status


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="법률용어 자동 처리 시스템")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="실행 모드: single(한 번만), continuous(지속적)")
    parser.add_argument("--check-interval", type=int, default=300,
                       help="파일 체크 간격 (초, 기본 300초)")
    parser.add_argument("--archive-days", type=int, default=30,
                       help="아카이브 기준 일수 (기본 30일)")
    parser.add_argument("--db-path", default="data/legal_terms.db",
                       help="데이터베이스 파일 경로")
    parser.add_argument("--base-dir", default="data/raw/law_open_api/legal_terms",
                       help="법률용어 파일 기본 디렉토리")
    parser.add_argument("--monitor", action="store_true",
                       help="모니터링 모드 실행")
    parser.add_argument("--reprocess-failed", action="store_true",
                       help="실패한 파일들 재처리")
    parser.add_argument("--clear-failed", action="store_true",
                       help="실패한 파일들 삭제 (주의: 데이터 손실 가능)")
    parser.add_argument("--verbose", action="store_true",
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.monitor:
        # 모니터링 모드
        monitor = LegalTermMonitor(args.db_path, args.base_dir)
        monitor.print_detailed_report()
        
        health = monitor.check_system_health()
        print(f"\n=== 시스템 상태 ===")
        print(f"상태: {health['status']}")
        if health['issues']:
            print("문제점:")
            for issue in health['issues']:
                print(f"  - {issue}")
        else:
            print("문제 없음")
            
    elif args.reprocess_failed:
        # 실패한 파일들 재처리 모드
        processor = LegalTermAutoProcessor(args.db_path, args.base_dir)
        processor.reprocess_failed_files()
        
    elif args.clear_failed:
        # 실패한 파일들 삭제 모드
        processor = LegalTermAutoProcessor(args.db_path, args.base_dir)
        processor.clear_failed_files()
            
    else:
        # 자동 처리 모드
        processor = LegalTermAutoProcessor(args.db_path, args.base_dir)
        
        if args.mode == "continuous":
            processor.run_continuous_processing(args.check_interval, args.archive_days)
        else:
            processor.run_single_processing()


if __name__ == "__main__":
    main()
