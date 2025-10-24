#!/usr/bin/env python3
"""
법률 용어 수집 시스템 테스트 스크립트

이 스크립트는 법률 용어 수집 시스템의 기본 기능을 테스트합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from scripts.data_collection.law_open_api.legal_terms.legal_term_collector import LegalTermCollector
from scripts.data_collection.law_open_api.legal_terms.legal_term_vector_store import LegalTermVectorStore
from scripts.data_collection.law_open_api.legal_terms.legal_term_collection_manager import LegalTermCollectionManager, CollectionConfig
from scripts.data_collection.law_open_api.legal_terms.legal_term_collection_config import get_config
from source.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_api_connection():
    """API 연결 테스트"""
    logger.info("API 연결 테스트 시작")
    
    config = get_config()
    
    try:
        async with LegalTermCollector(config) as collector:
            # 간단한 API 요청 테스트
            response = await collector.get_term_list(page=1)
            
            if response:
                logger.info("API 연결 성공")
                logger.info(f"응답 데이터: {response}")
                return True
            else:
                logger.error("API 응답 없음")
                return False
                
    except Exception as e:
        logger.error(f"API 연결 테스트 실패: {e}")
        return False

async def test_database_operations():
    """데이터베이스 작업 테스트"""
    logger.info("데이터베이스 작업 테스트 시작")
    
    config = get_config()
    
    try:
        collector = LegalTermCollector(config)
        
        # 데이터베이스 초기화 테스트
        collector._init_database()
        logger.info("데이터베이스 초기화 성공")
        
        # 통계 조회 테스트
        stats = collector.get_collection_stats()
        logger.info(f"데이터베이스 통계: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터베이스 작업 테스트 실패: {e}")
        return False

async def test_vector_store():
    """벡터스토어 테스트"""
    logger.info("벡터스토어 테스트 시작")
    
    config = get_config()
    
    try:
        vector_store = LegalTermVectorStore(config)
        
        # 통계 조회 테스트
        stats = vector_store.get_vector_store_stats()
        logger.info(f"벡터스토어 통계: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"벡터스토어 테스트 실패: {e}")
        return False

async def test_collection_manager():
    """수집 관리자 테스트"""
    logger.info("수집 관리자 테스트 시작")
    
    config = get_config()
    
    try:
        manager = LegalTermCollectionManager(config)
        
        # 상태 조회 테스트
        status = manager.get_collection_status()
        logger.info(f"수집 상태: {status}")
        
        # 보고서 생성 테스트
        report = manager.get_collection_report()
        logger.info(f"수집 보고서: {report}")
        
        return True
        
    except Exception as e:
        logger.error(f"수집 관리자 테스트 실패: {e}")
        return False

async def test_small_collection():
    """소규모 수집 테스트"""
    logger.info("소규모 수집 테스트 시작")
    
    config = get_config()
    manager = LegalTermCollectionManager(config)
    
    try:
        # 테스트용 설정 (1페이지만 수집)
        collection_config = CollectionConfig(
            start_page=1,
            end_page=1,
            query="",
            gana="",
            list_batch_size=1,
            detail_batch_size=5,
            vector_batch_size=10
        )
        
        logger.info("테스트 수집 시작 (1페이지만)")
        success = await manager.collect_legal_terms(collection_config)
        
        if success:
            logger.info("소규모 수집 테스트 성공")
            return True
        else:
            logger.error("소규모 수집 테스트 실패")
            return False
            
    except Exception as e:
        logger.error(f"소규모 수집 테스트 실패: {e}")
        return False

async def run_all_tests():
    """모든 테스트 실행"""
    logger.info("=== 법률 용어 수집 시스템 테스트 시작 ===")
    
    tests = [
        ("API 연결", test_api_connection),
        ("데이터베이스 작업", test_database_operations),
        ("벡터스토어", test_vector_store),
        ("수집 관리자", test_collection_manager),
        # ("소규모 수집", test_small_collection),  # 실제 API 호출하므로 주석 처리
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} 테스트 ---")
        try:
            result = await test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} 테스트 통과")
            else:
                logger.error(f"❌ {test_name} 테스트 실패")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 테스트 오류: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    logger.info("\n=== 테스트 결과 요약 ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 모든 테스트가 통과했습니다!")
        return True
    else:
        logger.error("⚠️ 일부 테스트가 실패했습니다.")
        return False

def main():
    """메인 함수"""
    try:
        success = asyncio.run(run_all_tests())
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("테스트가 사용자에 의해 중단되었습니다.")
        return 0
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
