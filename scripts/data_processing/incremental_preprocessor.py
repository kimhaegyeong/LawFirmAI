#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
증분 전처리 프로세서

미처리 파일만 선별하여 전처리하는 증분 처리 시스템입니다.
기존 LegalDataProcessor를 재사용하고 체크포인트 시스템을 통합합니다.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.data.data_processor import LegalDataProcessor
from core.data.database import DatabaseManager
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


class IncrementalPreprocessor:
    """증분 전처리 프로세서 클래스"""

    def __init__(self,
                 raw_data_base_path: str = "data/raw/assembly",
                 processed_data_base_path: str = "data/processed/assembly",
                 processing_version: str = "1.0",
                 checkpoint_manager: CheckpointManager = None,
                 db_manager: DatabaseManager = None,
                 enable_term_normalization: bool = True,
                 batch_size: int = 100):
        """
        증분 전처리 프로세서 초기화

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

        # 기존 LegalDataProcessor 초기화
        self.processor = LegalDataProcessor(enable_term_normalization)

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

        self.logger.info("IncrementalPreprocessor initialized")

    def process_new_files_only(self, files: List[Path],
                              data_type: str = "law_only") -> ProcessingResult:
        """
        새로운 파일만 처리

        Args:
            files: 처리할 파일 목록
            data_type: 데이터 유형

        Returns:
            ProcessingResult: 처리 결과
        """
        self.logger.info(f"Starting incremental processing of {len(files)} files")
        self.stats['start_time'] = datetime.now()

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
                                          desc="Processing files",
                                          initial=start_index,
                                          total=len(files))):
            try:
                # 파일 해시 계산
                file_hash = self._calculate_file_hash(file_path)

                # 이미 처리된 파일인지 확인
                if self.db_manager.is_file_processed(str(file_path)):
                    self.logger.debug(f"File already processed: {file_path}")
                    continue

                # 파일 처리
                self.logger.info(f"Processing file {i+1}/{len(files)}: {file_path.name}")

                result = self._process_single_file(file_path, data_type)

                if result['success']:
                    processed_files.append(file_path)
                    total_records += result['record_count']

                    # DB에 처리 완료 기록
                    self.db_manager.mark_file_as_processed(
                        file_path=str(file_path),
                        file_hash=file_hash,
                        data_type=data_type,
                        record_count=result['record_count'],
                        processing_version="1.0"
                    )

                    self.logger.info(f"Successfully processed: {file_path.name} "
                                   f"({result['record_count']} records)")
                else:
                    failed_files.append(file_path)
                    error_messages.append(f"{file_path}: {result['error']}")

                    # DB에 실패 기록
                    self.db_manager.mark_file_as_processed(
                        file_path=str(file_path),
                        file_hash=file_hash,
                        data_type=data_type,
                        record_count=0,
                        processing_version="1.0",
                        error_message=result['error']
                    )

                    self.logger.error(f"Failed to process: {file_path.name} - {result['error']}")

                # 배치 단위로 체크포인트 저장
                if (i + 1) % self.batch_size == 0:
                    self._save_checkpoint({
                        'current_file_index': i + 1,
                        'processed_files': [str(f) for f in processed_files],
                        'failed_files': [str(f) for f in failed_files],
                        'total_records': total_records,
                        'data_type': data_type
                    })
                    self.logger.info(f"Checkpoint saved at file {i + 1}")

            except Exception as e:
                error_msg = f"Unexpected error processing {file_path}: {e}"
                error_messages.append(error_msg)
                failed_files.append(file_path)
                self.logger.error(error_msg)

        # 최종 통계 계산
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()
        self.stats['total_files'] = len(files)
        self.stats['processed_files'] = len(processed_files)
        self.stats['failed_files'] = len(failed_files)
        self.stats['total_records'] = total_records

        # 체크포인트 정리
        self._cleanup_checkpoint()

        # 결과 생성
        result = ProcessingResult(
            success=len(failed_files) == 0,
            processed_files=processed_files,
            failed_files=failed_files,
            total_records=total_records,
            processing_time=self.stats['processing_time'],
            error_messages=error_messages
        )

        self.logger.info(f"Processing completed: {len(processed_files)} success, "
                        f"{len(failed_files)} failed, {total_records} total records")

        return result

    def process_new_data_only(self, data_type: str = "law_only") -> Dict[str, Any]:
        """
        새로 추가된 데이터만 감지하여 전처리

        Args:
            data_type: 처리할 특정 데이터 유형 (예: 'law_only'). 'all'이면 모든 유형 처리.

        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        self.logger.info(f"Starting incremental preprocessing for data type: {data_type}")
        start_time = datetime.now()

        new_files_by_type = self.auto_detector.detect_new_data_sources(str(self.raw_data_base_path / "law_only"), data_type)

        files_to_process = []
        if data_type == "all":
            for files in new_files_by_type.values():
                files_to_process.extend(files)
        elif data_type in new_files_by_type:
            files_to_process = new_files_by_type[data_type]
        else:
            self.logger.warning(f"Data type '{data_type}' not recognized or no new files found.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.stats['total_scanned_files'] = sum(len(f) for f in new_files_by_type.values())
        self.stats['new_files_to_process'] = len(files_to_process)

        if not files_to_process:
            self.logger.info("No new files to preprocess.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        self.logger.info(f"Found {len(files_to_process)} new files for preprocessing.")

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

                # 전처리 수행 - 원본 데이터 구조에 맞게 변환
                processed_data = self._process_assembly_law_data(raw_data)

                # 출력 경로 설정 (예: data/processed/assembly/law_only/20251016/ml_enhanced_...)
                relative_path = file_path.relative_to(self.raw_data_base_path)
                output_subdir = self.processed_data_base_path / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                output_file_name = f"ml_enhanced_{file_path.stem}.json"
                output_file_path = output_subdir / output_file_name

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)

                # 처리 이력 기록
                record_count = len(processed_data.get('laws', [])) if isinstance(processed_data, dict) else 1
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
        self.logger.info(f"Incremental preprocessing completed. Stats: {self.stats}")
        return self.stats

    def _process_assembly_law_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        국회 법률 데이터 구조에 맞게 전처리

        Args:
            raw_data: 원본 데이터 (metadata, items 구조)

        Returns:
            Dict[str, Any]: 전처리된 데이터
        """
        try:
            metadata = raw_data.get('metadata', {})
            items = raw_data.get('items', [])

            processed_laws = []

            for item in items:
                # 각 법률 항목을 처리
                processed_law = {
                    'id': item.get('cont_id', ''),
                    'law_name': item.get('law_name', ''),
                    'law_id': item.get('cont_id', ''),
                    'mst': None,
                    'effective_date': None,
                    'promulgation_date': None,
                    'ministry': None,
                    'category': metadata.get('data_type', 'law_only'),
                    'status': 'success',
                    'processed_at': datetime.now().isoformat(),
                    'articles': self._extract_articles(item.get('law_content', '')),
                    'full_content': item.get('law_content', ''),
                    'cleaned_content': self._clean_content(item.get('law_content', '')),
                    'chunks': [],
                    'article_chunks': [],
                    'entities': {},
                    'data_quality': {
                        'parsing_quality_score': 0.8,  # 기본값
                        'content_length': len(item.get('law_content', '')),
                        'article_count': len(self._extract_articles(item.get('law_content', '')))
                    }
                }

                processed_laws.append(processed_law)

            return {'laws': processed_laws}

        except Exception as e:
            self.logger.error(f"Error processing assembly law data: {e}")
            return {'laws': []}

    def _extract_articles(self, content: str) -> List[Dict[str, Any]]:
        """
        법률 내용에서 조문 추출

        Args:
            content: 법률 내용

        Returns:
            List[Dict[str, Any]]: 추출된 조문 목록
        """
        articles = []

        if not content:
            return articles

        # 간단한 조문 추출 로직 (실제로는 더 정교한 파싱이 필요)
        import re

        # 조문 패턴 찾기 (예: "제1조", "제2조" 등)
        article_pattern = r'제(\d+)조\s*([^제]*?)(?=제\d+조|$)'
        matches = re.findall(article_pattern, content, re.DOTALL)

        for i, (article_num, article_content) in enumerate(matches):
            article = {
                'article_number': int(article_num),
                'article_title': f"제{article_num}조",
                'article_content': article_content.strip(),
                'is_supplementary': False,
                'ml_confidence_score': 0.8,
                'parsing_method': 'regex'
            }
            articles.append(article)

        return articles

    def _clean_content(self, content: str) -> str:
        """
        내용 정리

        Args:
            content: 원본 내용

        Returns:
            str: 정리된 내용
        """
        if not content:
            return ""

        # 기본적인 정리 (HTML 태그 제거, 공백 정리 등)
        import re

        # HTML 태그 제거
        cleaned = re.sub(r'<[^>]+>', '', content)

        # 연속된 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _process_single_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """
        단일 파일 처리

        Args:
            file_path: 처리할 파일 경로
            data_type: 데이터 유형

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 데이터 유형별 처리
            if data_type == 'law_only':
                return self._process_law_only_file(file_path, raw_data)
            elif data_type == 'precedents':
                return self._process_precedents_file(file_path, raw_data)
            else:
                return self._process_generic_file(file_path, raw_data)

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }

    def _process_law_only_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        law_only 파일 처리

        Args:
            file_path: 파일 경로
            raw_data: 원본 데이터

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 기존 LegalDataProcessor 사용
            processed_data = self.data_processor.process_law_data(raw_data)

            if not processed_data or 'laws' not in processed_data:
                return {
                    'success': False,
                    'error': 'No laws found in processed data',
                    'record_count': 0
                }

            # 출력 파일 경로 생성
            output_path = self._get_output_path(file_path, 'law_only')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 처리된 데이터 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            record_count = len(processed_data['laws'])

            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }

    def _process_precedents_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        precedents 파일 처리

        Args:
            file_path: 파일 경로
            raw_data: 원본 데이터

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 판례 데이터 처리 (기본 구조 유지)
            processed_data = {
                'metadata': raw_data.get('metadata', {}),
                'precedents': raw_data.get('items', [])
            }

            # 출력 파일 경로 생성
            output_path = self._get_output_path(file_path, 'precedents')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 처리된 데이터 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            record_count = len(processed_data['precedents'])

            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }

    def _process_generic_file(self, file_path: Path, raw_data: Dict) -> Dict[str, Any]:
        """
        일반 파일 처리

        Args:
            file_path: 파일 경로
            raw_data: 원본 데이터

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 기본 구조 유지
            processed_data = raw_data

            # 출력 파일 경로 생성
            output_path = self._get_output_path(file_path, 'generic')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 처리된 데이터 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            # 레코드 수 계산
            record_count = 0
            if isinstance(raw_data, dict) and 'items' in raw_data:
                record_count = len(raw_data['items'])

            return {
                'success': True,
                'error': None,
                'record_count': record_count,
                'output_path': output_path
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'record_count': 0
            }

    def _get_output_path(self, input_path: Path, data_type: str) -> Path:
        """
        출력 파일 경로 생성

        Args:
            input_path: 입력 파일 경로
            data_type: 데이터 유형

        Returns:
            Path: 출력 파일 경로
        """
        # 날짜 폴더 추출
        date_folder = input_path.parent.name

        # 출력 경로 구성
        output_path = self.output_dir / "assembly" / data_type / date_folder

        # 파일명 변경 (원본 파일명 유지하되 접두사 추가)
        filename = f"processed_{input_path.name}"

        return output_path / filename

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        파일 해시 계산

        Args:
            file_path: 파일 경로

        Returns:
            str: 파일 해시값
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        체크포인트 저장

        Args:
            checkpoint_data: 체크포인트 데이터

        Returns:
            bool: 저장 성공 여부
        """
        if not self.checkpoint_manager:
            return False

        try:
            checkpoint_data['timestamp'] = datetime.now().isoformat()
            checkpoint_data['stage'] = 'preprocessing'
            return self.checkpoint_manager.save_checkpoint(checkpoint_data)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            return False

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드

        Returns:
            Optional[Dict]: 체크포인트 데이터 또는 None
        """
        if not self.checkpoint_manager:
            return None

        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data and checkpoint_data.get('stage') == 'preprocessing':
                return checkpoint_data
            return None
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return None

    def _cleanup_checkpoint(self) -> bool:
        """
        체크포인트 정리

        Returns:
            bool: 정리 성공 여부
        """
        if not self.checkpoint_manager:
            return False

        try:
            # 체크포인트 파일 삭제
            checkpoint_file = self.checkpoint_manager.checkpoint_file
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                self.logger.info("Checkpoint cleaned up")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoint: {e}")
            return False

    def resume_from_checkpoint(self) -> bool:
        """
        체크포인트에서 재개

        Returns:
            bool: 재개 가능 여부
        """
        checkpoint_data = self._load_checkpoint()
        return checkpoint_data is not None

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        처리 통계 조회

        Returns:
            Dict[str, Any]: 처리 통계
        """
        return self.stats.copy()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='증분 전처리 프로세서')
    parser.add_argument('--input-files', nargs='+', type=Path,
                       help='처리할 파일 목록')
    parser.add_argument('--data-type', default='law_only',
                       choices=['law_only', 'precedents', 'constitutional'],
                       help='데이터 유형')
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

        # 증분 전처리 프로세서 초기화
        preprocessor = IncrementalPreprocessor(
            checkpoint_manager=checkpoint_manager,
            batch_size=args.batch_size
        )

        # 체크포인트에서 재개 확인
        if args.resume and not preprocessor.resume_from_checkpoint():
            logging.warning("No checkpoint found, starting from beginning")

        # 파일 처리 실행
        if args.input_files:
            result = preprocessor.process_new_files_only(args.input_files, args.data_type)
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
            stats = preprocessor.process_new_data_only(args.data_type)

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
        logging.error(f"Error in incremental preprocessing: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
