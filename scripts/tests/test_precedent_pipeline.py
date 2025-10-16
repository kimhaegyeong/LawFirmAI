#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 파이프라인 테스트 스크립트

판례 데이터 처리 파이프라인의 각 단계를 테스트하는 스크립트입니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.data_processing.precedent_preprocessor import PrecedentPreprocessor
from scripts.data_processing.incremental_precedent_preprocessor import IncrementalPrecedentPreprocessor
from scripts.ml_training.vector_embedding.incremental_precedent_vector_builder import IncrementalPrecedentVectorBuilder
from scripts.data_processing.utilities.import_precedents_to_db import PrecedentDataImporter
from scripts.data_processing.auto_pipeline_orchestrator import AutoPipelineOrchestrator
from source.data.database import DatabaseManager

logger = logging.getLogger(__name__)


def test_precedent_data_detection():
    """판례 데이터 감지 테스트"""
    logger.info("Testing precedent data detection...")
    
    try:
        detector = AutoDataDetector()
        
        # civil 카테고리 데이터 감지 테스트
        detected_files = detector.detect_new_data_sources(
            "data/raw/assembly/precedent", 
            "precedent_civil"
        )
        
        logger.info(f"Detected files: {detected_files}")
        
        if detected_files.get('precedent_civil'):
            logger.info(f"✓ Found {len(detected_files['precedent_civil'])} civil precedent files")
            return True
        else:
            logger.warning("✗ No civil precedent files detected")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data detection test failed: {e}")
        return False


def test_precedent_preprocessing():
    """판례 전처리 테스트"""
    logger.info("Testing precedent preprocessing...")
    
    try:
        # 샘플 판례 데이터 생성
        sample_data = {
            "metadata": {
                "data_type": "precedent",
                "category": "civil",
                "page_number": 1,
                "collected_at": "2025-10-16T10:00:00"
            },
            "items": [
                {
                    "case_name": "특허침해금지및손해배상청구의소",
                    "case_number": "2017나2684",
                    "decision_date": "2018.8.16",
                    "field": "민사",
                    "court": "특허법원",
                    "detail_url": "https://example.com",
                    "structured_content": {
                        "case_info": {
                            "case_title": "특허법원 2018. 8. 16. 선고 2017나2684 판결",
                            "decision_date": "2018-08-16",
                            "case_number": "2017나2684"
                        },
                        "legal_sections": {
                            "판시사항": "특허침해에 관한 판시사항",
                            "판결요지": "특허침해가 인정되지 않는다는 판결요지",
                            "참조조문": "특허법 제126조",
                            "참조판례": "대법원 2015다12345 판결",
                            "주문": "원고의 청구를 기각한다",
                            "이유": "특허침해가 인정되지 않는 이유"
                        },
                        "parties": {
                            "plaintiff": "원고",
                            "defendant": "피고"
                        }
                    }
                }
            ]
        }
        
        preprocessor = PrecedentPreprocessor()
        processed_data = preprocessor.process_precedent_data(sample_data)
        
        if processed_data.get('cases') and len(processed_data['cases']) > 0:
            case = processed_data['cases'][0]
            logger.info(f"✓ Preprocessing successful: {case['case_name']}")
            logger.info(f"  - Sections: {len(case.get('sections', []))}")
            logger.info(f"  - Parties: {len(case.get('parties', []))}")
            return True
        else:
            logger.error("✗ Preprocessing failed: No cases processed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        return False


def test_database_schema():
    """데이터베이스 스키마 테스트"""
    logger.info("Testing database schema...")
    
    try:
        db_manager = DatabaseManager()
        
        # 판례 테이블 존재 확인
        tables_to_check = [
            'precedent_cases',
            'precedent_sections', 
            'precedent_parties',
            'fts_precedent_cases',
            'fts_precedent_sections'
        ]
        
        for table in tables_to_check:
            result = db_manager.execute_query(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if result:
                logger.info(f"✓ Table {table} exists")
            else:
                logger.error(f"✗ Table {table} does not exist")
                return False
        
        logger.info("✓ All precedent tables exist")
        return True
        
    except Exception as e:
        logger.error(f"✗ Database schema test failed: {e}")
        return False


def test_precedent_import():
    """판례 DB 임포트 테스트"""
    logger.info("Testing precedent database import...")
    
    try:
        # 샘플 전처리된 데이터 생성
        sample_processed_data = {
            "metadata": {
                "data_type": "precedent",
                "category": "civil",
                "processed_at": datetime.now().isoformat(),
                "total_cases": 1,
                "processing_version": "1.0"
            },
            "cases": [
                {
                    "case_id": "test_case_001",
                    "category": "civil",
                    "case_name": "테스트 사건",
                    "case_number": "2025테001",
                    "decision_date": "2025-10-16",
                    "field": "민사",
                    "court": "서울중앙지방법원",
                    "detail_url": "https://test.com",
                    "full_text": "테스트 사건의 전체 텍스트",
                    "searchable_text": "테스트 사건",
                    "sections": [
                        {
                            "section_type": "points_at_issue",
                            "section_type_korean": "판시사항",
                            "section_content": "테스트 판시사항",
                            "section_length": 10,
                            "has_content": True
                        }
                    ],
                    "parties": [
                        {
                            "party_type": "plaintiff",
                            "party_type_korean": "원고",
                            "party_content": "테스트 원고",
                            "party_length": 5
                        }
                    ],
                    "data_quality": {
                        "parsing_quality_score": 0.9,
                        "content_length": 100,
                        "section_count": 1
                    },
                    "processed_at": datetime.now().isoformat(),
                    "status": "success"
                }
            ]
        }
        
        # 임시 파일 생성
        temp_file = Path("temp_test_precedent.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(sample_processed_data, f, ensure_ascii=False, indent=2)
        
        # 임포트 테스트
        importer = PrecedentDataImporter()
        result = importer.import_file(temp_file, incremental=True)
        
        # 임시 파일 삭제
        temp_file.unlink()
        
        if result.get('imported_cases', 0) > 0 or result.get('updated_cases', 0) > 0:
            logger.info(f"✓ Import successful: {result}")
            return True
        else:
            logger.error(f"✗ Import failed: {result}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False


def test_precedent_pipeline():
    """전체 판례 파이프라인 테스트"""
    logger.info("Testing full precedent pipeline...")
    
    try:
        orchestrator = AutoPipelineOrchestrator()
        
        # civil 카테고리 파이프라인 실행
        result = orchestrator.run_precedent_pipeline(
            category="civil",
            auto_detect=True
        )
        
        if result.success:
            logger.info(f"✓ Pipeline successful: {result}")
            return True
        else:
            logger.error(f"✗ Pipeline failed: {result.error_messages}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """메인 테스트 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting precedent pipeline tests...")
    
    tests = [
        ("Data Detection", test_precedent_data_detection),
        ("Preprocessing", test_precedent_preprocessing),
        ("Database Schema", test_database_schema),
        ("Database Import", test_precedent_import),
        ("Full Pipeline", test_precedent_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # 결과 요약
    logger.info(f"\n{'='*50}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
