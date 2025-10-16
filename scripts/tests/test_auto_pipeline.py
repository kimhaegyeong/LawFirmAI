#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동화 파이프라인 통합 테스트 스크립트

데이터 감지부터 벡터 임베딩까지 전체 파이프라인의 각 단계를 검증합니다.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import unittest
from unittest.mock import Mock, patch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.data_processing.incremental_preprocessor import IncrementalPreprocessor
from scripts.ml_training.vector_embedding.incremental_vector_builder import IncrementalVectorBuilder
from scripts.data_processing.utilities.import_laws_to_db import AssemblyLawImporter
from scripts.data_processing.auto_pipeline_orchestrator import AutoPipelineOrchestrator
from source.data.database import DatabaseManager


class TestAutoPipeline(unittest.TestCase):
    """자동화 파이프라인 통합 테스트 클래스"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 테스트용 임시 디렉토리 생성
        cls.test_dir = Path(tempfile.mkdtemp(prefix="lawfirm_test_"))
        cls.test_data_dir = cls.test_dir / "data"
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트용 데이터베이스 생성
        cls.test_db_path = cls.test_dir / "test_lawfirm.db"
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
        cls.logger.info(f"Test directory created: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # 임시 디렉토리 삭제
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.logger.info("Test directory cleaned up")
    
    def setUp(self):
        """각 테스트 메서드 실행 전 초기화"""
        # 테스트용 데이터 생성
        self._create_test_data()
        
        # 테스트용 데이터베이스 초기화
        self.db_manager = DatabaseManager(str(self.test_db_path))
    
    def tearDown(self):
        """각 테스트 메서드 실행 후 정리"""
        # 테스트 데이터 정리
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_test_data(self):
        """테스트용 데이터 생성"""
        # 원본 데이터 디렉토리 생성
        raw_dir = self.test_data_dir / "raw" / "assembly" / "law_only" / "20251016"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 테스트용 법률 데이터 생성
        test_law_data = {
            "metadata": {
                "data_type": "law_only",
                "category": None,
                "page_number": 1,
                "page_start_item": 0,
                "batch_number": 1,
                "count": 1,
                "collected_at": "2025-10-16T16:14:58.612939",
                "file_version": "1.0",
                "total_collected": 1
            },
            "items": [
                {
                    "cont_id": "1970010100000001",
                    "cont_sid": "0001",
                    "law_name": "테스트법",
                    "law_content": "제1조(목적) 이 법은 테스트를 위한 법률이다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.\n1. \"테스트\"란 검증을 의미한다."
                }
            ]
        }
        
        # 테스트 파일 저장
        test_file = raw_dir / "law_only_page_001_20251016_161458_1.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_law_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Test data created: {test_file}")
    
    def test_auto_data_detector(self):
        """자동 데이터 감지 시스템 테스트"""
        self.logger.info("Testing AutoDataDetector...")
        
        # 데이터 감지기 초기화
        detector = AutoDataDetector(self.db_manager)
        
        # 데이터 감지 실행
        base_path = str(self.test_data_dir / "raw" / "assembly" / "law_only")
        detected_files = detector.detect_new_data_sources(base_path, "law_only")
        
        # 결과 검증
        self.assertIn("law_only", detected_files)
        self.assertEqual(len(detected_files["law_only"]), 1)
        
        # 파일 분류 테스트
        test_file = detected_files["law_only"][0]
        data_type = detector.classify_data_type(test_file)
        self.assertEqual(data_type, "law_only")
        
        # 통계 생성 테스트
        stats = detector.get_data_statistics(detected_files["law_only"])
        self.assertGreater(stats["total_files"], 0)
        self.assertGreater(stats["estimated_records"], 0)
        
        self.logger.info("AutoDataDetector test passed")
    
    def test_incremental_preprocessor(self):
        """증분 전처리 프로세서 테스트"""
        self.logger.info("Testing IncrementalPreprocessor...")
        
        # 전처리 프로세서 초기화
        preprocessor = IncrementalPreprocessor(
            db_manager=self.db_manager,
            batch_size=10
        )
        
        # 테스트 파일 목록 생성
        test_files = list((self.test_data_dir / "raw" / "assembly" / "law_only" / "20251016").glob("*.json"))
        
        # 전처리 실행
        result = preprocessor.process_new_files_only(test_files, "law_only")
        
        # 결과 검증
        self.assertTrue(result.success)
        self.assertEqual(len(result.processed_files), 1)
        self.assertGreater(result.total_records, 0)
        
        # 처리된 파일 확인
        processed_file = result.processed_files[0]
        self.assertTrue(processed_file.exists())
        
        # 처리된 데이터 검증
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        
        self.assertIn("laws", processed_data)
        self.assertGreater(len(processed_data["laws"]), 0)
        
        self.logger.info("IncrementalPreprocessor test passed")
    
    @patch('scripts.ml_training.vector_embedding.incremental_vector_builder.FAISS_AVAILABLE', True)
    def test_incremental_vector_builder(self):
        """증분 벡터 임베딩 생성기 테스트"""
        self.logger.info("Testing IncrementalVectorBuilder...")
        
        # 벡터 빌더 초기화
        vector_builder = IncrementalVectorBuilder(
            model_name="jhgan/ko-sroberta-multitask",
            batch_size=5,
            chunk_size=100
        )
        
        # 테스트용 기존 인덱스 생성 (모킹)
        test_index_dir = self.test_dir / "test_index"
        test_index_dir.mkdir(exist_ok=True)
        
        # FAISS 인덱스 모킹
        with patch('faiss.read_index') as mock_read_index, \
             patch('faiss.write_index') as mock_write_index:
            
            # 모킹된 인덱스 설정
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_read_index.return_value = mock_index
            
            # 기존 인덱스 로드 테스트
            success = vector_builder.load_existing_index(str(test_index_dir))
            self.assertTrue(success)
            
            # 처리된 파일 목록 생성
            processed_files = list((self.test_data_dir / "processed").glob("**/*.json"))
            
            if processed_files:
                # 새로운 문서 추가 테스트
                result = vector_builder.add_new_documents(processed_files)
                
                # 결과 검증
                self.assertTrue(result.success)
                self.assertGreaterEqual(result.new_vectors, 0)
                
                # 업데이트된 인덱스 저장 테스트
                save_success = vector_builder.save_updated_index(str(test_index_dir))
                self.assertTrue(save_success)
        
        self.logger.info("IncrementalVectorBuilder test passed")
    
    def test_db_importer_incremental(self):
        """DB 임포터 증분 모드 테스트"""
        self.logger.info("Testing AssemblyLawImporter incremental mode...")
        
        # DB 임포터 초기화
        importer = AssemblyLawImporter(str(self.test_db_path))
        
        # 처리된 파일 생성
        processed_dir = self.test_data_dir / "processed" / "assembly" / "law_only" / "20251016"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = {
            "laws": [
                {
                    "law_id": "test_law_001",
                    "law_name": "테스트법",
                    "law_content": "제1조(목적) 이 법은 테스트를 위한 법률이다.",
                    "articles": [
                        {
                            "article_number": "1",
                            "article_title": "목적",
                            "article_content": "이 법은 테스트를 위한 법률이다.",
                            "is_supplementary": False
                        }
                    ],
                    "processing_version": "1.0"
                }
            ]
        }
        
        processed_file = processed_dir / "processed_law_only_page_001_20251016_161458_1.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # 증분 임포트 실행
        result = importer.import_file(processed_file, incremental=True)
        
        # 결과 검증
        self.assertIn("imported_laws", result)
        self.assertGreaterEqual(result["imported_laws"], 0)
        
        # 데이터베이스에서 데이터 확인
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assembly_laws WHERE law_id = ?", ("test_law_001",))
            count = cursor.fetchone()[0]
            self.assertGreater(count, 0)
        
        self.logger.info("AssemblyLawImporter incremental mode test passed")
    
    def test_auto_pipeline_orchestrator(self):
        """자동화 파이프라인 오케스트레이터 테스트"""
        self.logger.info("Testing AutoPipelineOrchestrator...")
        
        # 테스트용 설정
        test_config = {
            'data_sources': {
                'law_only': {
                    'enabled': True,
                    'priority': 1,
                    'raw_path': str(self.test_data_dir / "raw" / "assembly" / "law_only"),
                    'processed_path': str(self.test_data_dir / "processed" / "assembly" / "law_only")
                }
            },
            'preprocessing': {
                'batch_size': 10,
                'enable_term_normalization': True,
                'enable_ml_enhancement': True
            },
            'vectorization': {
                'model_name': 'jhgan/ko-sroberta-multitask',
                'dimension': 768,
                'batch_size': 5,
                'chunk_size': 100,
                'index_type': 'flat',
                'existing_index_path': str(self.test_dir / "test_index"),
                'output_path': str(self.test_dir / "test_index")
            },
            'incremental': {
                'enabled': True,
                'check_file_hash': True,
                'skip_duplicates': True
            }
        }
        
        # 오케스트레이터 초기화
        orchestrator = AutoPipelineOrchestrator(
            config=test_config,
            checkpoint_dir=str(self.test_dir / "checkpoints"),
            db_path=str(self.test_db_path)
        )
        
        # 특정 경로로 파이프라인 실행
        test_path = str(self.test_data_dir / "raw" / "assembly" / "law_only" / "20251016")
        result = orchestrator.run_auto_pipeline(
            data_source="law_only",
            auto_detect=False,
            specific_path=test_path
        )
        
        # 결과 검증
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.total_files_detected, 0)
        self.assertGreaterEqual(result.files_processed, 0)
        
        # 단계별 결과 확인
        self.assertIn("detection", result.stage_results)
        self.assertIn("preprocessing", result.stage_results)
        
        self.logger.info("AutoPipelineOrchestrator test passed")
    
    def test_database_tracking(self):
        """데이터베이스 처리 이력 추적 테스트"""
        self.logger.info("Testing database file tracking...")
        
        # 테스트 파일 경로
        test_file_path = str(self.test_data_dir / "test_file.json")
        test_file_hash = "test_hash_123"
        
        # 파일 처리 완료 표시
        record_id = self.db_manager.mark_file_as_processed(
            file_path=test_file_path,
            file_hash=test_file_hash,
            data_type="law_only",
            record_count=5
        )
        
        self.assertIsNotNone(record_id)
        
        # 파일 처리 여부 확인
        is_processed = self.db_manager.is_file_processed(test_file_path)
        self.assertTrue(is_processed)
        
        # 처리 상태 조회
        status = self.db_manager.get_file_processing_status(test_file_path)
        self.assertIsNotNone(status)
        self.assertEqual(status["data_type"], "law_only")
        self.assertEqual(status["record_count"], 5)
        
        # 처리 통계 조회
        stats = self.db_manager.get_processing_statistics("law_only")
        self.assertGreater(stats["total_files"], 0)
        
        self.logger.info("Database file tracking test passed")
    
    def test_error_handling(self):
        """에러 처리 테스트"""
        self.logger.info("Testing error handling...")
        
        # 존재하지 않는 파일로 테스트
        non_existent_file = self.test_data_dir / "non_existent.json"
        
        # 데이터 감지기 테스트
        detector = AutoDataDetector(self.db_manager)
        detected_files = detector.detect_new_data_sources(str(non_existent_file))
        self.assertEqual(len(detected_files), 0)
        
        # 전처리 프로세서 테스트
        preprocessor = IncrementalPreprocessor(db_manager=self.db_manager)
        result = preprocessor.process_new_files_only([non_existent_file], "law_only")
        self.assertFalse(result.success)
        self.assertGreater(len(result.error_messages), 0)
        
        self.logger.info("Error handling test passed")


def run_integration_tests():
    """통합 테스트 실행"""
    print("="*60)
    print("LAWFIRM AI AUTO PIPELINE INTEGRATION TESTS")
    print("="*60)
    
    # 테스트 스위트 생성
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAutoPipeline)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("="*60)
    
    return len(result.failures) == 0 and len(result.errors) == 0


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='자동화 파이프라인 통합 테스트')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세 로그 출력')
    parser.add_argument('--test-specific', choices=[
        'detector', 'preprocessor', 'vector_builder', 
        'db_importer', 'orchestrator', 'tracking', 'error'
    ], help='특정 테스트만 실행')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.test_specific:
            # 특정 테스트만 실행
            test_class = TestAutoPipeline()
            test_method = f"test_{args.test_specific}"
            
            if hasattr(test_class, test_method):
                print(f"Running specific test: {test_method}")
                test_class.setUpClass()
                test_class.setUp()
                
                try:
                    getattr(test_class, test_method)()
                    print(f"✅ {test_method} passed")
                    success = True
                except Exception as e:
                    print(f"❌ {test_method} failed: {e}")
                    success = False
                finally:
                    test_class.tearDown()
                    test_class.tearDownClass()
            else:
                print(f"Test method not found: {test_method}")
                success = False
        else:
            # 전체 통합 테스트 실행
            success = run_integration_tests()
        
        return success
        
    except Exception as e:
        print(f"Test execution error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
