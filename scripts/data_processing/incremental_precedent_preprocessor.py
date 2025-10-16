#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
증분 판례 전처리 프로세서

미처리 판례 파일만 선별하여 전처리하는 증분 처리 시스템입니다.
기존 PrecedentPreprocessor를 재사용하고 체크포인트 시스템을 통합합니다.
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.precedent_preprocessor import PrecedentPreprocessor
from source.data.database import DatabaseManager
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_processing.auto_data_detector import AutoDataDetector


@dataclass
class ProcessingResult:
    """처리 결과 데이터 클래스"""
    success: bool
    processed_files: List[Path]
    failed_files: List[Path]
    total_records: int
    processing_time: float
    error_messages: List[str]


class IncrementalPrecedentPreprocessor:
    """증분 판례 전처리 프로세서 클래스"""
    
    def __init__(self, 
                 raw_data_base_path: str = "data/raw/assembly",
                 processed_data_base_path: str = "data/processed/assembly",
                 processing_version: str = "1.0",
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None,
                 enable_term_normalization: bool = True,
                 batch_size: int = 100):
        """
        증분 판례 전처리 프로세서 초기화
        
        Args:
            raw_data_base_path: 원본 데이터 기본 경로
            processed_data_base_path: 전처리된 데이터 기본 경로
            processing_version: 처리 버전
            checkpoint_manager: 체크포인트 관리자
            db_manager: 데이터베이스 관리자
            enable_term_normalization: 법률 용어 정규화 활성화
            batch_size: 배치 처리 크기
        """
        self.raw_data_base_path = Path(raw_data_base_path)
        self.processed_data_base_path = Path(processed_data_base_path)
        self.processed_data_base_path.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = checkpoint_manager
        self.db_manager = db_manager or DatabaseManager()
        self.batch_size = batch_size
        self.processing_version = processing_version
        
        # 판례 전처리기 초기화
        self.preprocessor = PrecedentPreprocessor(enable_term_normalization)
        
        # 자동 데이터 감지기 초기화
        self.auto_detector = AutoDataDetector(raw_data_base_path)
        
        # 출력 디렉토리 설정
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 처리 통계
        self.stats = {
            'total_scanned_files': 0,
            'new_files_to_process': 0,
            'successfully_processed': 0,
            'failed_to_process': 0,
            'skipped_already_processed': 0,
            'errors': []
        }
        
        self.logger.info("IncrementalPrecedentPreprocessor initialized")
    
    def process_new_data_only(self, category: str = "civil") -> Dict[str, Any]:
        """
        새로 추가된 판례 데이터만 감지하여 전처리
        
        Args:
            category: 처리할 특정 카테고리 (civil, criminal, family)
            
        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        self.logger.info(f"Starting incremental precedent preprocessing for category: {category}")
        start_time = datetime.now()
        
        # 데이터 타입 결정
        data_type = f"precedent_{category}"
        
        # 새로운 파일 감지
        new_files_by_type = self.auto_detector.detect_new_data_sources(
            str(self.raw_data_base_path / "precedent"), 
            data_type
        )
        
        files_to_process = new_files_by_type.get(data_type, [])
        
        self.stats['total_scanned_files'] = sum(len(f) for f in new_files_by_type.values())
        self.stats['new_files_to_process'] = len(files_to_process)
        
        if not files_to_process:
            self.logger.info("No new precedent files to preprocess.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.logger.info(f"Found {len(files_to_process)} new precedent files for preprocessing.")

        for file_path in files_to_process:
            raw_file_path_str = str(file_path)
            file_hash = self.auto_detector.get_file_hash(file_path)
            
            # 이미 처리된 파일인지 다시 확인 (경쟁 조건 방지)
            if self.db_manager.is_file_processed(raw_file_path_str):
                self.stats['skipped_already_processed'] += 1
                self.logger.info(f"Skipping already processed file: {file_path}")
                continue

            try:
                # 데이터 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 전처리 수행 - 판례 데이터 구조에 맞게 변환
                processed_data = self._process_assembly_precedent_data(raw_data, category)
                
                # 출력 경로 설정 (예: data/processed/assembly/precedent/civil/20251016/ml_enhanced_...)
                relative_path = file_path.relative_to(self.raw_data_base_path / "precedent")
                output_subdir = self.processed_data_base_path / "precedent" / category / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                output_file_name = f"ml_enhanced_{file_path.stem}.json"
                output_file_path = output_subdir / output_file_name
                
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                # 처리 이력 기록
                record_count = len(processed_data.get('cases', [])) if isinstance(processed_data, dict) else 1
                self.db_manager.mark_file_as_processed(
                    raw_file_path_str, file_hash, data_type, 
                    record_count=record_count, processing_version=self.processing_version
                )
                self.stats['successfully_processed'] += 1
                self.logger.info(f"Successfully preprocessed and marked as processed: {file_path}")

            except Exception as e:
                error_msg = f"Failed to preprocess file {file_path}: {e}"
                self.logger.error(error_msg)
                self.stats['failed_to_process'] += 1
                self.stats['errors'].append(error_msg)
                self.db_manager.mark_file_as_processed(
                    raw_file_path_str, file_hash, data_type, 
                    error_message=error_msg,
                    processing_version=self.processing_version
                )
                
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration'] = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Incremental precedent preprocessing completed. Stats: {self.stats}")
        return self.stats
    
    def _process_assembly_precedent_data(self, raw_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        국회 판례 데이터 구조에 맞게 전처리
        
        Args:
            raw_data: 원본 데이터 (metadata, items 구조)
            category: 판례 카테고리
            
        Returns:
            Dict[str, Any]: 전처리된 데이터
        """
        try:
            # PrecedentPreprocessor를 사용하여 처리
            processed_data = self.preprocessor.process_precedent_data(raw_data)
            
            # 카테고리 정보 추가
            if isinstance(processed_data, dict) and 'metadata' in processed_data:
                processed_data['metadata']['category'] = category
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing assembly precedent data: {e}")
            return {'cases': []}
    
    def process_new_files_only(self, files: List[Path], 
                              category: str = "civil") -> ProcessingResult:
        """
        새로운 파일만 처리
        
        Args:
            files: 처리할 파일 목록
            category: 판례 카테고리
        
        Returns:
            ProcessingResult: 처리 결과
        """
        self.logger.info(f"Starting processing for {len(files)} new precedent files of category: {category}")
        start_time = datetime.now()
        
        processed_files = []
        failed_files = []
        total_records = 0
        error_messages = []
        
        # 체크포인트에서 재개할 수 있는지 확인
        checkpoint_data = self._load_checkpoint()
        if checkpoint_data:
            self.logger.info("Resuming from checkpoint...")
            start_index = checkpoint_data.get('current_file_index', 0)
            processed_files.extend(checkpoint_data.get('processed_files', []))
            failed_files.extend(checkpoint_data.get('failed_files', []))
        else:
            start_index = 0
        
        # 파일별 처리
        for i, file_path in enumerate(tqdm(files[start_index:], 
                                          desc="Processing precedent files",
                                          initial=start_index,
                                          total=len(files))):
            try:
                # 파일 해시 계산
                file_hash = self._calculate_file_hash(file_path)
                
                # 이미 처리된 파일인지 확인
                if self.db_manager.is_file_processed(str(file_path)):
                    self.logger.debug(f"File already processed: {file_path}")
                    continue
                
                # 단일 파일 처리
                file_result = self._process_single_file(file_path, category)
                
                if file_result['success']:
                    processed_files.append(file_path)
                    total_records += file_result['record_count']
                    self.db_manager.mark_file_as_processed(
                        str(file_path), file_hash, f"precedent_{category}", 
                        record_count=file_result['record_count'], 
                        processing_version=self.processing_version
                    )
                else:
                    failed_files.append(file_path)
                    error_messages.append(file_result['error'])
                    self.db_manager.mark_file_as_processed(
                        str(file_path), file_hash, f"precedent_{category}", 
                        error_message=file_result['error'],
                        processing_version=self.processing_version
                    )
                
                # 체크포인트 저장
                self._save_checkpoint(i + start_index + 1, processed_files, failed_files)
                
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                self.logger.error(error_msg)
                failed_files.append(file_path)
                error_messages.append(error_msg)
                self.db_manager.mark_file_as_processed(
                    str(file_path), self._calculate_file_hash(file_path), f"precedent_{category}", 
                    error_message=error_msg,
                    processing_version=self.processing_version
                )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = ProcessingResult(
            success=len(failed_files) == 0,
            processed_files=processed_files,
            failed_files=failed_files,
            total_records=total_records,
            processing_time=duration,
            error_messages=error_messages
        )
        
        self.logger.info(f"Processing completed: {len(processed_files)} success, "
                        f"{len(failed_files)} failed, {total_records} total records")
        
        return result
    
    def _process_single_file(self, file_path: Path, category: str) -> Dict[str, Any]:
        """
        단일 파일 처리
        
        Args:
            file_path: 처리할 파일 경로
            category: 판례 카테고리
        
        Returns:
            Dict[str, Any]: 처리 결과 (성공 여부, 레코드 수, 에러 메시지)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            processed_data = self._process_assembly_precedent_data(raw_data, category)
            
            # 처리된 데이터를 임시 파일로 저장하거나, 메모리에서 다음 단계로 전달
            record_count = len(processed_data.get('cases', [])) if isinstance(processed_data, dict) else 1
            
            return {'success': True, 'record_count': record_count, 'error': None}
        except Exception as e:
            return {'success': False, 'record_count': 0, 'error': str(e)}

    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 내용의 SHA256 해시를 계산하여 반환"""
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _save_checkpoint(self, current_file_index: int, 
                         processed_files: List[Path], 
                         failed_files: List[Path]):
        """
        현재 처리 상태를 체크포인트로 저장
        """
        if self.checkpoint_manager:
            checkpoint_data = {
                'current_file_index': current_file_index,
                'processed_files': [str(p) for p in processed_files],
                'failed_files': [str(p) for p in failed_files],
                'timestamp': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_checkpoint('incremental_precedent_preprocessing', checkpoint_data)
            self.logger.debug(f"Checkpoint saved at index: {current_file_index}")

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        """
        if self.checkpoint_manager:
            return self.checkpoint_manager.load_checkpoint('incremental_precedent_preprocessing')
        return None


def main():
    parser = argparse.ArgumentParser(description="증분 판례 데이터 전처리 프로세서")
    parser.add_argument('--input-files', nargs='*', type=Path,
                        help='처리할 파일 목록')
    parser.add_argument('--category', default='civil',
                        choices=['civil', 'criminal', 'family'],
                        help='판례 카테고리')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='배치 처리 크기')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints',
                        help='체크포인트 디렉토리')
    parser.add_argument('--resume', action='store_true',
                        help='체크포인트에서 재개')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 체크포인트 관리자 초기화
        checkpoint_manager = CheckpointManager(args.checkpoint_dir)
        
        # 증분 판례 전처리 프로세서 초기화
        preprocessor = IncrementalPrecedentPreprocessor(
            checkpoint_manager=checkpoint_manager,
            batch_size=args.batch_size
        )
        
        # 파일 처리 실행
        if args.input_files:
            result = preprocessor.process_new_files_only(args.input_files, args.category)
            # 결과 출력
            logging.info("Processing Results:")
            logging.info(f"  Success: {len(result.processed_files)} files")
            logging.info(f"  Failed: {len(result.failed_files)} files")
            logging.info(f"  Total records: {result.total_records}")
            logging.info(f"  Processing time: {result.processing_time:.2f} seconds")
            
            if result.error_messages:
                logging.error("Error messages:")
                for error in result.error_messages:
                    logging.error(f"  - {error}")
            
            return result.success
        else:
            # 자동으로 새 파일 감지하여 처리
            stats = preprocessor.process_new_data_only(args.category)
            
            # 결과 출력
            logging.info("Processing Results:")
            logging.info(f"  Total scanned files: {stats['total_scanned_files']}")
            logging.info(f"  New files to process: {stats['new_files_to_process']}")
            logging.info(f"  Successfully processed: {stats['successfully_processed']}")
            logging.info(f"  Failed to process: {stats['failed_to_process']}")
            logging.info(f"  Skipped already processed: {stats['skipped_already_processed']}")
            logging.info(f"  Duration: {stats['duration']:.2f} seconds")
            
            if stats['errors']:
                logging.error("Error messages:")
                for error in stats['errors']:
                    logging.error(f"  - {error}")
            
            return stats['successfully_processed'] > 0
            
    except Exception as e:
        logging.error(f"Error in incremental precedent preprocessing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
