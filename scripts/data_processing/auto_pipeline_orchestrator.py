#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 자동화 파이프라인 오케스트레이터

데이터 감지부터 벡터 임베딩까지 전체 파이프라인을 자동화하는 시스템입니다.
각 단계별로 진행 상황을 추적하고 체크포인트를 관리합니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.data_processing.incremental_preprocessor import IncrementalPreprocessor
from scripts.ml_training.vector_embedding.incremental_vector_builder import IncrementalVectorBuilder
from scripts.data_processing.utilities.import_laws_to_db import AssemblyLawImporter
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from source.data.database import DatabaseManager


@dataclass
class PipelineResult:
    """파이프라인 실행 결과 데이터 클래스"""
    success: bool
    total_files_detected: int
    files_processed: int
    vectors_added: int
    laws_imported: int
    processing_time: float
    stage_results: Dict[str, Any]
    error_messages: List[str]


class AutoPipelineOrchestrator:
    """자동화 파이프라인 오케스트레이터"""
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 checkpoint_dir: str = "data/checkpoints",
                 db_path: str = "data/lawfirm.db"):
        """
        파이프라인 오케스트레이터 초기화
        
        Args:
            config: 파이프라인 설정
            checkpoint_dir: 체크포인트 디렉토리
            db_path: 데이터베이스 경로
        """
        self.config = config or self._get_default_config()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.db_manager = DatabaseManager(db_path)
        self.checkpoint_manager = CheckpointManager(str(self.checkpoint_dir))
        
        # 각 단계별 컴포넌트
        self.data_detector = AutoDataDetector(self.db_manager)
        self.preprocessor = IncrementalPreprocessor(
            checkpoint_manager=self.checkpoint_manager,
            db_manager=self.db_manager,
            batch_size=self.config['preprocessing']['batch_size']
        )
        self.vector_builder = IncrementalVectorBuilder(
            model_name=self.config['vectorization']['model_name'],
            batch_size=self.config['vectorization']['batch_size'],
            chunk_size=self.config['vectorization']['chunk_size']
        )
        self.db_importer = AssemblyLawImporter(db_path)
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 파이프라인 상태
        self.pipeline_state = {
            'current_stage': None,
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'stage_results': {}
        }
        
        self.logger.info("AutoPipelineOrchestrator initialized")
    
    def run_auto_pipeline(self, 
                         data_source: str = "law_only",
                         auto_detect: bool = True,
                         specific_path: str = None) -> PipelineResult:
        """
        전체 자동화 파이프라인 실행
        
        Args:
            data_source: 데이터 소스 유형
            auto_detect: 자동 감지 여부
            specific_path: 특정 경로 지정
        
        Returns:
            PipelineResult: 파이프라인 실행 결과
        """
        self.pipeline_state['start_time'] = datetime.now()
        self.logger.info(f"Starting auto pipeline for data source: {data_source}")
        
        stage_results = {}
        error_messages = []
        
        try:
            # Step 1: 데이터 감지
            self.pipeline_state['current_stage'] = 'detection'
            self.logger.info("Step 1: Detecting new data sources...")
            
            if specific_path:
                detected_files = self._detect_specific_path(specific_path, data_source)
            elif auto_detect:
                detected_files = self._detect_new_data_sources(data_source)
            else:
                self.logger.error("No detection method specified")
                return PipelineResult(
                    success=False,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results={},
                    error_messages=["No detection method specified"]
                )
            
            stage_results['detection'] = {
                'success': len(detected_files) > 0,
                'files_detected': sum(len(files) for files in detected_files.values()),
                'data_types': list(detected_files.keys())
            }
            
            if not detected_files:
                self.logger.info("No new files detected")
                return PipelineResult(
                    success=True,
                    total_files_detected=0,
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=[]
                )
            
            # Step 2: 증분 전처리
            self.pipeline_state['current_stage'] = 'preprocessing'
            self.logger.info("Step 2: Incremental preprocessing...")
            
            preprocessing_results = self._run_preprocessing_stage(detected_files)
            stage_results['preprocessing'] = preprocessing_results
            
            if not preprocessing_results['success']:
                error_messages.extend(preprocessing_results['errors'])
                return PipelineResult(
                    success=False,
                    total_files_detected=sum(len(files) for files in detected_files.values()),
                    files_processed=0,
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=error_messages
                )
            
            # Step 3: 증분 벡터 임베딩
            self.pipeline_state['current_stage'] = 'vectorization'
            self.logger.info("Step 3: Incremental vector embedding...")
            
            vectorization_results = self._run_vectorization_stage(preprocessing_results['processed_files'])
            stage_results['vectorization'] = vectorization_results
            
            if not vectorization_results['success']:
                error_messages.extend(vectorization_results['errors'])
                return PipelineResult(
                    success=False,
                    total_files_detected=sum(len(files) for files in detected_files.values()),
                    files_processed=preprocessing_results['processed_files_count'],
                    vectors_added=0,
                    laws_imported=0,
                    processing_time=0,
                    stage_results=stage_results,
                    error_messages=error_messages
                )
            
            # Step 4: DB 증분 임포트
            self.pipeline_state['current_stage'] = 'import'
            self.logger.info("Step 4: Incremental database import...")
            
            import_results = self._run_import_stage(preprocessing_results['processed_files'])
            stage_results['import'] = import_results
            
            if not import_results['success']:
                error_messages.extend(import_results['errors'])
            
            # Step 5: 최종 통계 생성
            self.pipeline_state['current_stage'] = 'finalization'
            self.logger.info("Step 5: Generating final statistics...")
            
            final_stats = self._generate_final_statistics()
            stage_results['finalization'] = final_stats
            
            # 파이프라인 완료
            self.pipeline_state['end_time'] = datetime.now()
            processing_time = (
                self.pipeline_state['end_time'] - self.pipeline_state['start_time']
            ).total_seconds()
            
            success = len(error_messages) == 0
            
            result = PipelineResult(
                success=success,
                total_files_detected=sum(len(files) for files in detected_files.values()),
                files_processed=preprocessing_results['processed_files_count'],
                vectors_added=vectorization_results['vectors_added'],
                laws_imported=import_results['laws_imported'],
                processing_time=processing_time,
                stage_results=stage_results,
                error_messages=error_messages
            )
            
            self.logger.info(f"Pipeline completed: {success}")
            self.logger.info(f"Total files detected: {result.total_files_detected}")
            self.logger.info(f"Files processed: {result.files_processed}")
            self.logger.info(f"Vectors added: {result.vectors_added}")
            self.logger.info(f"Laws imported: {result.laws_imported}")
            self.logger.info(f"Processing time: {result.processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline execution error: {e}"
            self.logger.error(error_msg)
            error_messages.append(error_msg)
            
            return PipelineResult(
                success=False,
                total_files_detected=0,
                files_processed=0,
                vectors_added=0,
                laws_imported=0,
                processing_time=0,
                stage_results=stage_results,
                error_messages=error_messages
            )
    
    def _detect_new_data_sources(self, data_source: str) -> Dict[str, List[Path]]:
        """새로운 데이터 소스 감지"""
        try:
            base_path = self.config['data_sources'][data_source]['raw_path']
            detected_files = self.data_detector.detect_new_data_sources(base_path, data_source)
            
            self.logger.info(f"Detected {sum(len(files) for files in detected_files.values())} new files")
            return detected_files
            
        except Exception as e:
            self.logger.error(f"Error in data detection: {e}")
            return {}
    
    def _detect_specific_path(self, specific_path: str, data_source: str) -> Dict[str, List[Path]]:
        """특정 경로에서 데이터 감지"""
        try:
            path_obj = Path(specific_path)
            if not path_obj.exists():
                self.logger.error(f"Specified path does not exist: {specific_path}")
                return {}
            
            # 특정 경로의 파일들을 감지된 파일로 처리
            files = list(path_obj.glob("*.json"))
            detected_files = {data_source: files}
            
            self.logger.info(f"Detected {len(files)} files in specific path")
            return detected_files
            
        except Exception as e:
            self.logger.error(f"Error detecting specific path: {e}")
            return {}
    
    def _run_preprocessing_stage(self, detected_files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """전처리 단계 실행"""
        try:
            all_processed_files = []
            total_records = 0
            errors = []
            
            # 데이터 유형별로 처리
            for data_type, files in detected_files.items():
                if not files:
                    continue
                
                self.logger.info(f"Preprocessing {len(files)} {data_type} files...")
                
                result = self.preprocessor.process_new_files_only(files, data_type)
                
                if result.success:
                    all_processed_files.extend(result.processed_files)
                    total_records += result.total_records
                    self.logger.info(f"Successfully processed {len(result.processed_files)} {data_type} files")
                else:
                    errors.extend(result.error_messages)
                    self.logger.error(f"Failed to process {data_type} files")
            
            return {
                'success': len(errors) == 0,
                'processed_files': all_processed_files,
                'processed_files_count': len(all_processed_files),
                'total_records': total_records,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Preprocessing stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'processed_files': [],
                'processed_files_count': 0,
                'total_records': 0,
                'errors': [error_msg]
            }
    
    def _run_vectorization_stage(self, processed_files: List[Path]) -> Dict[str, Any]:
        """벡터화 단계 실행"""
        try:
            if not processed_files:
                self.logger.info("No files to vectorize")
                return {
                    'success': True,
                    'vectors_added': 0,
                    'errors': []
                }
            
            # 기존 인덱스 로드
            existing_index_path = self.config['vectorization']['existing_index_path']
            if not self.vector_builder.load_existing_index(existing_index_path):
                self.logger.error(f"Failed to load existing index from: {existing_index_path}")
                return {
                    'success': False,
                    'vectors_added': 0,
                    'errors': [f"Failed to load existing index from: {existing_index_path}"]
                }
            
            # 새로운 문서 추가
            self.logger.info(f"Adding {len(processed_files)} processed files to vector index...")
            result = self.vector_builder.add_new_documents(processed_files)
            
            if result.success:
                # 업데이트된 인덱스 저장
                output_path = self.config['vectorization']['output_path']
                if self.vector_builder.save_updated_index(output_path):
                    self.logger.info(f"Successfully added {result.new_vectors} vectors")
                    return {
                        'success': True,
                        'vectors_added': result.new_vectors,
                        'errors': []
                    }
                else:
                    return {
                        'success': False,
                        'vectors_added': 0,
                        'errors': ["Failed to save updated index"]
                    }
            else:
                return {
                    'success': False,
                    'vectors_added': 0,
                    'errors': result.error_messages
                }
            
        except Exception as e:
            error_msg = f"Vectorization stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'vectors_added': 0,
                'errors': [error_msg]
            }
    
    def _run_import_stage(self, processed_files: List[Path]) -> Dict[str, Any]:
        """DB 임포트 단계 실행"""
        try:
            if not processed_files:
                self.logger.info("No files to import")
                return {
                    'success': True,
                    'laws_imported': 0,
                    'errors': []
                }
            
            total_imported = 0
            total_updated = 0
            total_skipped = 0
            errors = []
            
            # 파일별로 증분 임포트
            for file_path in tqdm(processed_files, desc="Importing to database"):
                try:
                    result = self.db_importer.import_file(file_path, incremental=True)
                    
                    if 'error' not in result:
                        total_imported += result.get('imported_laws', 0)
                        total_updated += result.get('updated_laws', 0)
                        total_skipped += result.get('skipped_laws', 0)
                    else:
                        errors.append(result['error'])
                        
                except Exception as e:
                    error_msg = f"Error importing {file_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            success = len(errors) == 0
            
            self.logger.info(f"Import completed: {total_imported} imported, "
                           f"{total_updated} updated, {total_skipped} skipped")
            
            return {
                'success': success,
                'laws_imported': total_imported + total_updated,
                'laws_updated': total_updated,
                'laws_skipped': total_skipped,
                'errors': errors
            }
            
        except Exception as e:
            error_msg = f"Import stage error: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'laws_imported': 0,
                'errors': [error_msg]
            }
    
    def _generate_final_statistics(self) -> Dict[str, Any]:
        """최종 통계 생성"""
        try:
            # 데이터베이스 통계
            db_stats = self.db_manager.get_processing_statistics()
            
            # 처리된 파일 통계
            processed_files_stats = self.db_manager.get_processed_files_by_type('law_only')
            
            return {
                'success': True,
                'database_statistics': db_stats,
                'processed_files_count': len(processed_files_stats),
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating final statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'data_sources': {
                'law_only': {
                    'enabled': True,
                    'priority': 1,
                    'raw_path': 'data/raw/assembly/law_only',
                    'processed_path': 'data/processed/assembly/law_only'
                }
            },
            'preprocessing': {
                'batch_size': 100,
                'enable_term_normalization': True,
                'enable_ml_enhancement': True
            },
            'vectorization': {
                'model_name': 'jhgan/ko-sroberta-multitask',
                'dimension': 768,
                'batch_size': 20,
                'chunk_size': 200,
                'index_type': 'flat',
                'existing_index_path': 'data/embeddings/ml_enhanced_ko_sroberta',
                'output_path': 'data/embeddings/ml_enhanced_ko_sroberta'
            },
            'incremental': {
                'enabled': True,
                'check_file_hash': True,
                'skip_duplicates': True
            }
        }
    
    def save_pipeline_report(self, result: PipelineResult, output_path: str = None):
        """파이프라인 실행 리포트 저장"""
        try:
            if not output_path:
                output_path = f"reports/pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_path = Path(output_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                'pipeline_result': {
                    'success': result.success,
                    'total_files_detected': result.total_files_detected,
                    'files_processed': result.files_processed,
                    'vectors_added': result.vectors_added,
                    'laws_imported': result.laws_imported,
                    'processing_time': result.processing_time,
                    'error_messages': result.error_messages
                },
                'stage_results': result.stage_results,
                'pipeline_state': self.pipeline_state,
                'config': self.config,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Pipeline report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline report: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='자동화 파이프라인 오케스트레이터')
    parser.add_argument('--data-source', default='law_only',
                       choices=['law_only', 'precedents', 'constitutional'],
                       help='데이터 소스 유형')
    parser.add_argument('--auto-detect', action='store_true',
                       help='자동 데이터 감지 활성화')
    parser.add_argument('--data-path', help='특정 데이터 경로 지정')
    parser.add_argument('--config', help='설정 파일 경로')
    parser.add_argument('--checkpoint-dir', default='data/checkpoints',
                       help='체크포인트 디렉토리')
    parser.add_argument('--db-path', default='data/lawfirm.db',
                       help='데이터베이스 경로')
    parser.add_argument('--output-report', help='리포트 출력 경로')
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
        # 설정 로드
        config = None
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 파이프라인 오케스트레이터 초기화
        orchestrator = AutoPipelineOrchestrator(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            db_path=args.db_path
        )
        
        # 파이프라인 실행
        result = orchestrator.run_auto_pipeline(
            data_source=args.data_source,
            auto_detect=args.auto_detect,
            specific_path=args.data_path
        )
        
        # 리포트 저장
        orchestrator.save_pipeline_report(result, args.output_report)
        
        # 결과 출력
        print("\n" + "="*60)
        print("AUTO PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Success: {result.success}")
        print(f"Total files detected: {result.total_files_detected}")
        print(f"Files processed: {result.files_processed}")
        print(f"Vectors added: {result.vectors_added}")
        print(f"Laws imported: {result.laws_imported}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        if result.error_messages:
            print("\nErrors:")
            for error in result.error_messages:
                print(f"  - {error}")
        
        print("="*60)
        
        return result.success
        
    except Exception as e:
        logging.error(f"Error in auto pipeline: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
