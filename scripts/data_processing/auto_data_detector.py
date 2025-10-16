#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동 데이터 감지 시스템

새로운 데이터 소스를 자동으로 감지하고 분류하는 시스템입니다.
날짜별 폴더와 파일 패턴을 분석하여 처리할 데이터를 식별합니다.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

logger = logging.getLogger(__name__)


class AutoDataDetector:
    """자동 데이터 감지 클래스"""
    
    def __init__(self, raw_data_base_path: str = "data/raw/assembly", db_manager: DatabaseManager = None):
        """
        자동 데이터 감지기 초기화
        
        Args:
            raw_data_base_path: 원본 데이터 기본 경로
            db_manager: 데이터베이스 관리자 (선택사항)
        """
        self.raw_data_base_path = Path(raw_data_base_path)
        self.db_manager = db_manager or DatabaseManager()
        
        # 데이터 패턴 정의
        self.data_patterns = {
            'law_only': {
                'file_pattern': r'law_only_page_\d+_\d+_\d+\.json',
                'directory_pattern': r'\d{8}',  # YYYYMMDD 형식
                'metadata_key': 'data_type',
                'expected_value': 'law_only'
            },
            'precedents': {
                'file_pattern': r'precedent_page_\d+_\d+\.json',
                'directory_pattern': r'\d{8}',
                'metadata_key': 'data_type',
                'expected_value': 'precedents'
            },
            'constitutional': {
                'file_pattern': r'constitutional_page_\d+\.json',
                'directory_pattern': r'\d{8}',
                'metadata_key': 'data_type',
                'expected_value': 'constitutional'
            }
        }
        
        # 기본 경로 설정
        self.base_paths = {
            'law_only': 'data/raw/assembly/law_only',
            'precedents': 'data/raw/assembly/precedents',
            'constitutional': 'data/raw/constitutional'
        }
        
        logger.info("AutoDataDetector initialized")
    
    def detect_new_data_sources(self, base_path: str, data_type: str = None) -> Dict[str, List[Path]]:
        """
        새로운 데이터 소스 감지
        
        Args:
            base_path: 검색할 기본 경로
            data_type: 특정 데이터 유형 (None이면 모든 유형)
        
        Returns:
            Dict[str, List[Path]]: 데이터 유형별 파일 목록
        """
        logger.info(f"Detecting new data sources in: {base_path}")
        
        detected_files = defaultdict(list)
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return dict(detected_files)
        
        # 날짜별 폴더 스캔
        for date_folder in base_path_obj.iterdir():
            if not date_folder.is_dir():
                continue
            
            # 날짜 폴더 패턴 확인 (YYYYMMDD)
            if not self._is_date_folder(date_folder.name):
                continue
            
            logger.info(f"Scanning date folder: {date_folder.name}")
            
            # 폴더 내 파일 스캔
            for file_path in date_folder.glob("*.json"):
                if not file_path.is_file():
                    continue
                
                # 파일 유형 분류
                file_data_type = self.classify_data_type(file_path)
                
                if file_data_type and (data_type is None or file_data_type == data_type):
                    # 이미 처리된 파일인지 확인
                    if not self.db_manager.is_file_processed(str(file_path)):
                        detected_files[file_data_type].append(file_path)
                        logger.debug(f"New file detected: {file_path} (type: {file_data_type})")
                    else:
                        logger.debug(f"File already processed: {file_path}")
        
        # 결과 요약
        total_files = sum(len(files) for files in detected_files.values())
        logger.info(f"Detection completed: {total_files} new files found")
        for data_type, files in detected_files.items():
            logger.info(f"  {data_type}: {len(files)} files")
        
        return dict(detected_files)
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일 내용의 SHA256 해시를 계산하여 반환"""
        import hashlib
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    
    def classify_data_type(self, file_path: Path) -> Optional[str]:
        """
        파일 내용 기반 데이터 유형 분류
        
        Args:
            file_path: 분류할 파일 경로
        
        Returns:
            Optional[str]: 데이터 유형 또는 None
        """
        try:
            # 파일 크기 확인 (너무 큰 파일은 스킵)
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"File too large to analyze: {file_path}")
                return None
            
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 메타데이터에서 데이터 유형 확인
            if isinstance(data, dict) and 'metadata' in data:
                metadata = data['metadata']
                data_type = metadata.get('data_type')
                
                if data_type in self.data_patterns:
                    return data_type
            
            # 파일명 패턴으로 분류
            filename = file_path.name
            for data_type, pattern_info in self.data_patterns.items():
                import re
                if re.match(pattern_info['file_pattern'], filename):
                    return data_type
            
            # items 구조로 분류 (law_only 특화)
            if isinstance(data, dict) and 'items' in data:
                items = data['items']
                if items and isinstance(items, list):
                    first_item = items[0]
                    if isinstance(first_item, dict):
                        # law_name이 있으면 법률 데이터로 분류
                        if 'law_name' in first_item and 'law_content' in first_item:
                            return 'law_only'
                        # case_number가 있으면 판례 데이터로 분류
                        elif 'case_number' in first_item:
                            return 'precedents'
            
            logger.warning(f"Could not classify file: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error classifying file {file_path}: {e}")
            return None
    
    def get_data_statistics(self, files: List[Path]) -> Dict[str, Any]:
        """
        파일 목록의 통계 정보 생성
        
        Args:
            files: 분석할 파일 목록
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        if not files:
            return {
                'total_files': 0,
                'total_size': 0,
                'date_range': None,
                'file_types': {},
                'estimated_records': 0
            }
        
        total_size = 0
        file_types = defaultdict(int)
        dates = []
        estimated_records = 0
        
        for file_path in files:
            # 파일 크기
            total_size += file_path.stat().st_size
            
            # 파일 유형
            file_type = self.classify_data_type(file_path)
            if file_type:
                file_types[file_type] += 1
            
            # 날짜 추출
            date_folder = file_path.parent.name
            if self._is_date_folder(date_folder):
                dates.append(date_folder)
            
            # 예상 레코드 수 추정
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'items' in data:
                        estimated_records += len(data['items'])
            except Exception:
                pass
        
        # 날짜 범위 계산
        date_range = None
        if dates:
            dates.sort()
            date_range = {
                'start': dates[0],
                'end': dates[-1],
                'total_days': len(set(dates))
            }
        
        return {
            'total_files': len(files),
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'date_range': date_range,
            'file_types': dict(file_types),
            'estimated_records': estimated_records,
            'avg_file_size_mb': round(total_size / len(files) / (1024 * 1024), 2) if files else 0
        }
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        파일 해시 계산
        
        Args:
            file_path: 해시를 계산할 파일 경로
        
        Returns:
            str: 파일의 SHA-256 해시값
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _is_date_folder(self, folder_name: str) -> bool:
        """
        폴더명이 날짜 형식인지 확인
        
        Args:
            folder_name: 확인할 폴더명
        
        Returns:
            bool: 날짜 형식 여부
        """
        import re
        return bool(re.match(r'^\d{8}$', folder_name))
    
    def get_processing_priority(self, data_type: str) -> int:
        """
        데이터 유형별 처리 우선순위 반환
        
        Args:
            data_type: 데이터 유형
        
        Returns:
            int: 우선순위 (낮을수록 높은 우선순위)
        """
        priority_map = {
            'law_only': 1,
            'precedents': 2,
            'constitutional': 3,
            'legal_interpretations': 4,
            'administrative_rules': 5
        }
        return priority_map.get(data_type, 99)
    
    def generate_detection_report(self, detected_files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """
        감지 결과 리포트 생성
        
        Args:
            detected_files: 감지된 파일 목록
        
        Returns:
            Dict[str, Any]: 감지 리포트
        """
        report = {
            'detection_time': datetime.now().isoformat(),
            'total_data_types': len(detected_files),
            'total_files': sum(len(files) for files in detected_files.values()),
            'data_types': {}
        }
        
        for data_type, files in detected_files.items():
            stats = self.get_data_statistics(files)
            stats['priority'] = self.get_processing_priority(data_type)
            report['data_types'][data_type] = stats
        
        return report


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='자동 데이터 감지 시스템')
    parser.add_argument('--base-path', default='data/raw/assembly/law_only',
                       help='검색할 기본 경로')
    parser.add_argument('--data-type', choices=['law_only', 'precedents', 'constitutional'],
                       help='특정 데이터 유형만 검색')
    parser.add_argument('--output-report', help='감지 리포트를 저장할 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 데이터 감지기 초기화
        detector = AutoDataDetector()
        
        # 데이터 감지 실행
        logger.info("Starting data detection...")
        detected_files = detector.detect_new_data_sources(args.base_path, args.data_type)
        
        # 결과 출력
        if detected_files:
            logger.info("Detection Results:")
            for data_type, files in detected_files.items():
                logger.info(f"  {data_type}: {len(files)} files")
                
                # 처음 몇 개 파일 경로 출력
                for i, file_path in enumerate(files[:3]):
                    logger.info(f"    - {file_path}")
                if len(files) > 3:
                    logger.info(f"    ... and {len(files) - 3} more files")
        else:
            logger.info("No new files detected")
        
        # 리포트 생성 및 저장
        if args.output_report:
            report = detector.generate_detection_report(detected_files)
            report_path = Path(args.output_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Detection report saved to: {report_path}")
        
        return len(detected_files) > 0
        
    except Exception as e:
        logger.error(f"Error in data detection: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
