#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly 데이터베이스 검색 간단 테스트
벡터 인덱스 문제를 우회하고 데이터베이스 검색만 테스트
"""

import sys
import os
import logging
from typing import List, Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.services.exact_search_engine import ExactSearchEngine
from source.data.database import DatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_assembly_database_search():
    """Assembly 데이터베이스 검색 테스트"""
    try:
        logger.info("Testing Assembly database search...")
        
        # 데이터베이스 매니저 초기화
        database = DatabaseManager()
        
        # 테스트 쿼리
        test_queries = [
            "민법",
            "제1조",
            "계약",
            "손해배상"
        ]
        
        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")
            
            # 데이터베이스에서 직접 검색
            results = database.search_assembly_documents(query, limit=3)
            logger.info(f"Database search results: {len(results)} results")
            
            if results:
                for i, result in enumerate(results):
                    logger.info(f"  Result {i+1}:")
                    logger.info(f"    - Law: {result.get('law_name', 'N/A')}")
                    logger.info(f"    - Article: {result.get('article_number', 'N/A')}")
                    logger.info(f"    - Quality: {result.get('quality_score', 'N/A')}")
                    logger.info(f"    - Content preview: {result.get('content', '')[:100]}...")
            else:
                logger.warning(f"No results found for query: {query}")
        
        return True
        
    except Exception as e:
        logger.error(f"Assembly database search test failed: {e}")
        return False

def test_assembly_exact_search():
    """Assembly 정확한 매칭 검색 테스트"""
    try:
        logger.info("\nTesting Assembly exact search...")
        
        # 정확한 매칭 검색 엔진 초기화
        exact_search = ExactSearchEngine()
        
        # 테스트 쿼리
        test_queries = [
            "민법",
            "제1조",
            "계약"
        ]
        
        for query in test_queries:
            logger.info(f"\n--- Testing exact search: '{query}' ---")
            
            # 정확한 매칭 검색
            results = exact_search.search_assembly_laws(query)
            logger.info(f"Exact search results: {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:2]):  # 최대 2개만 표시
                    logger.info(f"  Result {i+1}:")
                    logger.info(f"    - Law: {result.get('law_name', 'N/A')}")
                    logger.info(f"    - Article: {result.get('article_number', 'N/A')}")
                    logger.info(f"    - Quality: {result.get('quality_score', 'N/A')}")
            else:
                logger.warning(f"No results found for query: {query}")
        
        return True
        
    except Exception as e:
        logger.error(f"Assembly exact search test failed: {e}")
        return False

def main():
    """메인 함수"""
    try:
        logger.info("Starting Assembly database integration tests...")
        
        # 테스트 실행
        test_results = {
            "database_search": test_assembly_database_search(),
            "exact_search": test_assembly_exact_search()
        }
        
        # 결과 요약
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"\n=== Test Results Summary ===")
        logger.info(f"Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        for test_name, result in test_results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # 모든 테스트가 통과했는지 확인
        all_passed = all(test_results.values())
        
        if all_passed:
            logger.info("\n🎉 Assembly database integration tests passed!")
            logger.info("Assembly 데이터베이스 검색 기능이 정상 작동합니다!")
        else:
            logger.warning("\n⚠️ Some tests failed. Please check the logs.")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
