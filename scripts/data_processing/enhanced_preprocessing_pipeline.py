#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Preprocessing Pipeline

통합된 전처리 파이프라인으로 법령과 판례 데이터를 효율적으로 처리합니다.
- 메모리 최적화
- 병렬 처리
- 품질 검증
- 오류 복구
- 진행 상황 추적
"""

import os
import sys
import json
import logging
import hashlib
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import gc
import psutil

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.data.database import DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager


@dataclass
class ProcessingConfig:
    """처리 설정 데이터 클래스"""
    # 기본 설정
    max_workers: int = min(multiprocessing.cpu_count(), 8)
    batch_size: int = 100
    max_memory_gb: float = 8.0
    
    # 전처리 설정
    enable_legal_analysis: bool = True
    enable_quality_validation: bool = True
    enable_duplicate_detection: bool = True
    
    # 출력 설정
    output_format: str = "json"  # json, parquet, csv
    compress_output: bool = False
    
    # 오류 처리
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True


@dataclass
class ProcessingResult:
    """처리 결과 데이터 클래스"""
    success: bool
    processed_files: List[Path] = field(default_factory=list)
    failed_files: List[Path] = field(default_factory=list)
    total_records: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_found: int = 0


class MemoryMonitor:
    """메모리 사용량 모니터링 클래스"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.peak_memory_mb = 0.0
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량을 MB 단위로 반환"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        return memory_mb
    
    def check_memory_limit(self) -> bool:
        """메모리 한계 확인"""
        current_mb = self.get_memory_usage()
        return current_mb < (self.max_memory_gb * 1024)
    
    def force_gc(self):
        """강제 가비지 컬렉션 실행"""
        gc.collect()


class QualityValidator:
    """데이터 품질 검증 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_law_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """법령 데이터 품질 검증"""
        issues = []
        
        # 필수 필드 검증
        required_fields = ['law_id', 'law_name', 'articles']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append({
                    'type': 'missing_field',
                    'field': field,
                    'severity': 'high',
                    'message': f'Missing required field: {field}'
                })
        
        # 조문 데이터 검증
        if 'articles' in data and isinstance(data['articles'], list):
            for i, article in enumerate(data['articles']):
                if not isinstance(article, dict):
                    issues.append({
                        'type': 'invalid_article',
                        'index': i,
                        'severity': 'medium',
                        'message': f'Article {i} is not a valid dictionary'
                    })
                elif 'article_number' not in article or 'content' not in article:
                    issues.append({
                        'type': 'incomplete_article',
                        'index': i,
                        'severity': 'high',
                        'message': f'Article {i} missing required fields'
                    })
        
        return issues
    
    def validate_precedent_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """판례 데이터 품질 검증"""
        issues = []
        
        # 필수 필드 검증
        required_fields = ['case_id', 'case_name', 'case_number', 'decision_date']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append({
                    'type': 'missing_field',
                    'field': field,
                    'severity': 'high',
                    'message': f'Missing required field: {field}'
                })
        
        # 날짜 형식 검증
        if 'decision_date' in data:
            try:
                datetime.strptime(data['decision_date'], '%Y-%m-%d')
            except ValueError:
                issues.append({
                    'type': 'invalid_date',
                    'field': 'decision_date',
                    'severity': 'medium',
                    'message': f'Invalid date format: {data["decision_date"]}'
                })
        
        return issues


class DuplicateDetector:
    """중복 데이터 탐지 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.seen_hashes = set()
    
    def calculate_content_hash(self, data: Dict[str, Any]) -> str:
        """데이터 내용의 해시값 계산"""
        # 핵심 필드만으로 해시 계산
        if 'law_id' in data:
            # 법령 데이터
            key_fields = ['law_id', 'law_name']
        else:
            # 판례 데이터
            key_fields = ['case_id', 'case_name', 'case_number']
        
        content = ''.join(str(data.get(field, '')) for field in key_fields)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, data: Dict[str, Any]) -> bool:
        """중복 데이터 여부 확인"""
        content_hash = self.calculate_content_hash(data)
        
        if content_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(content_hash)
        return False


class EnhancedPreprocessingPipeline:
    """향상된 전처리 파이프라인"""
    
    def __init__(self, 
                 config: ProcessingConfig = None,
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None):
        """
        전처리 파이프라인 초기화
        
        Args:
            config: 처리 설정
            checkpoint_manager: 체크포인트 관리자
            db_manager: 데이터베이스 관리자
        """
        self.config = config or ProcessingConfig()
        self.checkpoint_manager = checkpoint_manager
        self.db_manager = db_manager
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 모니터링 및 검증 컴포넌트
        self.memory_monitor = MemoryMonitor(self.config.max_memory_gb)
        self.quality_validator = QualityValidator()
        self.duplicate_detector = DuplicateDetector()
        
        # 통계
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_records': 0,
            'duplicates_found': 0,
            'quality_issues': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_law_files(self, 
                         input_paths: List[Path], 
                         output_dir: Path) -> ProcessingResult:
        """법령 파일들 처리"""
        self.logger.info(f"Processing {len(input_paths)} law files...")
        
        result = ProcessingResult(success=True)
        result.processing_time = datetime.now()
        
        # 배치별 처리
        for i in range(0, len(input_paths), self.config.batch_size):
            batch = input_paths[i:i + self.config.batch_size]
            
            # 메모리 확인
            if not self.memory_monitor.check_memory_limit():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                self.memory_monitor.force_gc()
            
            # 배치 처리
            batch_result = self._process_law_batch(batch, output_dir)
            
            # 결과 병합
            result.processed_files.extend(batch_result.processed_files)
            result.failed_files.extend(batch_result.failed_files)
            result.total_records += batch_result.total_records
            result.quality_issues.extend(batch_result.quality_issues)
            result.duplicates_found += batch_result.duplicates_found
            
            if not batch_result.success:
                result.success = False
                result.errors.extend(batch_result.errors)
        
        result.processing_time = (datetime.now() - result.processing_time).total_seconds()
        result.memory_peak_mb = self.memory_monitor.peak_memory_mb
        
        return result
    
    def process_precedent_files(self, 
                               input_paths: List[Path], 
                               output_dir: Path) -> ProcessingResult:
        """판례 파일들 처리"""
        self.logger.info(f"Processing {len(input_paths)} precedent files...")
        
        result = ProcessingResult(success=True)
        result.processing_time = datetime.now()
        
        # 배치별 처리
        for i in range(0, len(input_paths), self.config.batch_size):
            batch = input_paths[i:i + self.config.batch_size]
            
            # 메모리 확인
            if not self.memory_monitor.check_memory_limit():
                self.logger.warning("Memory limit reached, forcing garbage collection")
                self.memory_monitor.force_gc()
            
            # 배치 처리
            batch_result = self._process_precedent_batch(batch, output_dir)
            
            # 결과 병합
            result.processed_files.extend(batch_result.processed_files)
            result.failed_files.extend(batch_result.failed_files)
            result.total_records += batch_result.total_records
            result.quality_issues.extend(batch_result.quality_issues)
            result.duplicates_found += batch_result.duplicates_found
            
            if not batch_result.success:
                result.success = False
                result.errors.extend(batch_result.errors)
        
        result.processing_time = (datetime.now() - result.processing_time).total_seconds()
        result.memory_peak_mb = self.memory_monitor.peak_memory_mb
        
        return result
    
    def _process_law_batch(self, 
                          batch: List[Path], 
                          output_dir: Path) -> ProcessingResult:
        """법령 배치 처리"""
        result = ProcessingResult(success=True)
        
        for file_path in tqdm(batch, desc="Processing law batch"):
            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 중복 확인
                if self.duplicate_detector.is_duplicate(data):
                    result.duplicates_found += 1
                    continue
                
                # 품질 검증
                if self.config.enable_quality_validation:
                    quality_issues = self.quality_validator.validate_law_data(data)
                    if quality_issues:
                        result.quality_issues.extend(quality_issues)
                        result.quality_issues += len(quality_issues)
                
                # 전처리 (기존 파서 사용)
                processed_data = self._preprocess_law_data(data)
                
                # 출력 파일 저장
                output_file = output_dir / f"{file_path.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                result.processed_files.append(output_file)
                result.total_records += len(processed_data.get('articles', []))
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.logger.error(error_msg)
                result.failed_files.append(file_path)
                result.errors.append(error_msg)
                
                if not self.config.continue_on_error:
                    result.success = False
                    break
        
        return result
    
    def _process_precedent_batch(self, 
                                batch: List[Path], 
                                output_dir: Path) -> ProcessingResult:
        """판례 배치 처리"""
        result = ProcessingResult(success=True)
        
        for file_path in tqdm(batch, desc="Processing precedent batch"):
            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 데이터 구조 확인 및 처리
                if isinstance(data, dict) and 'items' in data:
                    # items 배열 처리
                    items = data.get('items', [])
                    if not isinstance(items, list):
                        self.logger.warning(f"Items is not a list in {file_path}")
                        continue
                    
                    processed_items = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        
                        # 중복 확인
                        if self.duplicate_detector.is_duplicate(item):
                            result.duplicates_found += 1
                            continue
                        
                        # 품질 검증
                        if self.config.enable_quality_validation:
                            quality_issues = self.quality_validator.validate_precedent_data(item)
                            if quality_issues:
                                result.quality_issues.extend(quality_issues)
                        
                        # 전처리
                        processed_item = self._preprocess_precedent_data(item)
                        processed_items.append(processed_item)
                    
                    # 출력 파일 저장
                    output_file = output_dir / f"{file_path.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'metadata': data.get('metadata', {}),
                            'items': processed_items
                        }, f, ensure_ascii=False, indent=2)
                    
                    result.processed_files.append(output_file)
                    result.total_records += len(processed_items)
                
                elif isinstance(data, list):
                    # 직접 배열인 경우
                    processed_items = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        
                        # 중복 확인
                        if self.duplicate_detector.is_duplicate(item):
                            result.duplicates_found += 1
                            continue
                        
                        # 품질 검증
                        if self.config.enable_quality_validation:
                            quality_issues = self.quality_validator.validate_precedent_data(item)
                            if quality_issues:
                                result.quality_issues.extend(quality_issues)
                        
                        # 전처리
                        processed_item = self._preprocess_precedent_data(item)
                        processed_items.append(processed_item)
                    
                    # 출력 파일 저장
                    output_file = output_dir / f"{file_path.stem}_processed.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_items, f, ensure_ascii=False, indent=2)
                    
                    result.processed_files.append(output_file)
                    result.total_records += len(processed_items)
                
                else:
                    self.logger.warning(f"Unexpected data structure in {file_path}")
                    continue
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                self.logger.error(error_msg)
                result.failed_files.append(file_path)
                result.errors.append(error_msg)
                
                if not self.config.continue_on_error:
                    result.success = False
                    break
        
        return result
    
    def _preprocess_law_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """법령 데이터 전처리"""
        # 기본 전처리 로직 (기존 파서 활용)
        processed = {
            'law_id': data.get('law_id', ''),
            'law_name': data.get('law_name', ''),
            'law_type': data.get('law_type', ''),
            'enactment_date': data.get('enactment_date', ''),
            'articles': []
        }
        
        # 조문 처리
        if 'articles' in data and isinstance(data['articles'], list):
            for article in data['articles']:
                if isinstance(article, dict):
                    processed_article = {
                        'article_number': article.get('article_number', ''),
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'searchable_text': self._generate_searchable_text(article.get('content', ''))
                    }
                    processed['articles'].append(processed_article)
        
        return processed
    
    def _preprocess_precedent_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """판례 데이터 전처리"""
        # 기본 전처리 로직
        processed = {
            'case_id': data.get('case_id', ''),
            'case_name': data.get('case_name', ''),
            'case_number': data.get('case_number', ''),
            'decision_date': data.get('decision_date', ''),
            'court': data.get('court', ''),
            'category': data.get('category', ''),
            'field': data.get('field', ''),
            'detail_url': data.get('detail_url', ''),
            'full_text': data.get('precedent_content', ''),
            'searchable_text': self._generate_searchable_text(data.get('precedent_content', '')),
            'created_at': datetime.now().isoformat()
        }
        
        return processed
    
    def _generate_searchable_text(self, content: str) -> str:
        """검색 가능한 텍스트 생성"""
        if not content:
            return ""
        
        # 기본 정규화
        import re
        # HTML 태그 제거
        content = re.sub(r'<[^>]+>', '', content)
        # 연속 공백 제거
        content = re.sub(r'\s+', ' ', content)
        # 앞뒤 공백 제거
        content = content.strip()
        
        return content
    
    def generate_report(self, result: ProcessingResult) -> Dict[str, Any]:
        """처리 결과 보고서 생성"""
        return {
            'summary': {
                'success': result.success,
                'processed_files': len(result.processed_files),
                'failed_files': len(result.failed_files),
                'total_records': result.total_records,
                'processing_time_seconds': result.processing_time,
                'memory_peak_mb': result.memory_peak_mb,
                'duplicates_found': result.duplicates_found,
                'quality_issues': len(result.quality_issues)
            },
            'errors': result.errors,
            'quality_issues': result.quality_issues[:10],  # 상위 10개만
            'failed_files': [str(f) for f in result.failed_files[:10]]  # 상위 10개만
        }


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Enhanced Preprocessing Pipeline')
    parser.add_argument('--input', required=True, help='Input directory or file pattern')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--data-type', choices=['law', 'precedent'], required=True, help='Data type')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of workers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--max-memory-gb', type=float, default=8.0, help='Maximum memory usage in GB')
    parser.add_argument('--enable-quality-validation', action='store_true', help='Enable quality validation')
    parser.add_argument('--enable-duplicate-detection', action='store_true', help='Enable duplicate detection')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정 생성
    config = ProcessingConfig(
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_memory_gb=args.max_memory_gb,
        enable_quality_validation=args.enable_quality_validation,
        enable_duplicate_detection=args.enable_duplicate_detection
    )
    
    # 파이프라인 초기화
    pipeline = EnhancedPreprocessingPipeline(config=config)
    
    # 입력 파일 수집
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    else:
        input_files = list(input_path.rglob('*.json'))
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 처리 실행
    if args.data_type == 'law':
        result = pipeline.process_law_files(input_files, output_dir)
    else:
        result = pipeline.process_precedent_files(input_files, output_dir)
    
    # 보고서 생성 및 저장
    report = pipeline.generate_report(result)
    report_file = output_dir / 'processing_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 결과 출력
    print(f"\n=== Processing Complete ===")
    print(f"Success: {result.success}")
    print(f"Processed files: {len(result.processed_files)}")
    print(f"Failed files: {len(result.failed_files)}")
    print(f"Total records: {result.total_records}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Memory peak: {result.memory_peak_mb:.2f} MB")
    print(f"Duplicates found: {result.duplicates_found}")
    print(f"Quality issues: {len(result.quality_issues)}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
