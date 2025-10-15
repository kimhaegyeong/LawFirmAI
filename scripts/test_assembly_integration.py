#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly 데이터 검색 통합 테스트 스크립트
TASK 3.5 완료 검증을 위한 테스트
"""

import sys
import os
import logging
from typing import List, Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.services.hybrid_search_engine import HybridSearchEngine
from source.services.exact_search_engine import ExactSearchEngine
from source.services.semantic_search_engine import SemanticSearchEngine
from source.services.rag_service import MLEnhancedRAGService
from source.data.database import DatabaseManager
from source.models.model_manager import LegalModelManager
from source.data.vector_store import LegalVectorStore
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssemblyIntegrationTester:
    """Assembly 데이터 통합 테스트 클래스"""
    
    def __init__(self):
        """테스트 초기화"""
        self.config = Config()
        self.database = DatabaseManager()
        self.hybrid_search = HybridSearchEngine()
        self.exact_search = ExactSearchEngine()
        self.semantic_search = SemanticSearchEngine()
        
        # RAG 서비스 초기화
        self.model_manager = LegalModelManager()
        self.vector_store = LegalVectorStore()
        self.rag_service = MLEnhancedRAGService(
            self.config, self.model_manager, self.vector_store, self.database
        )
        
        logger.info("Assembly Integration Tester initialized")
    
    def test_assembly_exact_search(self) -> bool:
        """Assembly 정확한 매칭 검색 테스트"""
        try:
            logger.info("Testing Assembly exact search...")
            
            # 테스트 쿼리
            test_queries = [
                "민법",
                "제1조",
                "계약",
                "손해배상"
            ]
            
            for query in test_queries:
                results = self.exact_search.search_assembly_laws(query)
                logger.info(f"Query '{query}': {len(results)} results")
                
                if results:
                    result = results[0]
                    logger.info(f"  - Law: {result.get('law_name', 'N/A')}")
                    logger.info(f"  - Article: {result.get('article_number', 'N/A')}")
                    logger.info(f"  - Quality Score: {result.get('quality_score', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Assembly exact search test failed: {e}")
            return False
    
    def test_assembly_hybrid_search(self) -> bool:
        """Assembly 하이브리드 검색 테스트"""
        try:
            logger.info("Testing Assembly hybrid search...")
            
            # Assembly 검색 타입 포함하여 검색
            results = self.hybrid_search.search(
                query="민법 계약",
                search_types=["assembly_law"],
                max_results=5
            )
            
            logger.info(f"Hybrid search results: {results.get('total_results', 0)} results")
            
            if results.get('results'):
                for i, result in enumerate(results['results'][:3]):
                    logger.info(f"  Result {i+1}:")
                    logger.info(f"    - Title: {result.get('title', 'N/A')}")
                    logger.info(f"    - Source: {result.get('source', 'N/A')}")
                    logger.info(f"    - Score: {result.get('relevance_score', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Assembly hybrid search test failed: {e}")
            return False
    
    def test_assembly_semantic_search(self) -> bool:
        """Assembly 의미적 검색 테스트"""
        try:
            logger.info("Testing Assembly semantic search...")
            
            # 의미적 검색 테스트
            results = self.semantic_search.search(
                query="계약의 성립과 효력",
                k=5,
                threshold=0.3
            )
            
            logger.info(f"Semantic search results: {len(results)} results")
            
            assembly_results = [r for r in results if r.get('type') == 'assembly_law']
            logger.info(f"Assembly results: {len(assembly_results)} results")
            
            if assembly_results:
                for i, result in enumerate(assembly_results[:3]):
                    logger.info(f"  Assembly Result {i+1}:")
                    logger.info(f"    - Title: {result.get('title', 'N/A')}")
                    logger.info(f"    - Score: {result.get('score', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Assembly semantic search test failed: {e}")
            return False
    
    def test_assembly_rag_integration(self) -> bool:
        """Assembly RAG 통합 테스트"""
        try:
            logger.info("Testing Assembly RAG integration...")
            
            # RAG 서비스에서 Assembly 문서 검색
            documents = self.rag_service.retrieve_relevant_documents(
                query="민법상 계약의 성립 요건",
                top_k=3
            )
            
            logger.info(f"RAG retrieved documents: {len(documents)} documents")
            
            assembly_docs = [d for d in documents if d.get('source') == 'assembly']
            logger.info(f"Assembly documents: {len(assembly_docs)} documents")
            
            if assembly_docs:
                for i, doc in enumerate(assembly_docs):
                    logger.info(f"  Assembly Document {i+1}:")
                    logger.info(f"    - Title: {doc.get('title', 'N/A')}")
                    logger.info(f"    - Article: {doc.get('article_number', 'N/A')}")
                    logger.info(f"    - Quality: {doc.get('quality_score', 'N/A')}")
                    logger.info(f"    - Similarity: {doc.get('similarity', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Assembly RAG integration test failed: {e}")
            return False
    
    def test_database_assembly_search(self) -> bool:
        """데이터베이스 Assembly 검색 테스트"""
        try:
            logger.info("Testing database Assembly search...")
            
            # 데이터베이스에서 직접 검색
            results = self.database.search_assembly_documents("민법", limit=5)
            
            logger.info(f"Database search results: {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:3]):
                    logger.info(f"  Database Result {i+1}:")
                    logger.info(f"    - Law: {result.get('law_name', 'N/A')}")
                    logger.info(f"    - Article: {result.get('article_number', 'N/A')}")
                    logger.info(f"    - Quality: {result.get('quality_score', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database Assembly search test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트 실행"""
        logger.info("Starting Assembly integration tests...")
        
        test_results = {
            "exact_search": self.test_assembly_exact_search(),
            "hybrid_search": self.test_assembly_hybrid_search(),
            "semantic_search": self.test_assembly_semantic_search(),
            "rag_integration": self.test_assembly_rag_integration(),
            "database_search": self.test_database_assembly_search()
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
        
        return test_results

def main():
    """메인 함수"""
    try:
        tester = AssemblyIntegrationTester()
        results = tester.run_all_tests()
        
        # 모든 테스트가 통과했는지 확인
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n🎉 All Assembly integration tests passed!")
            logger.info("TASK 3.5 Assembly 데이터 검색 통합 및 최적화 완료!")
        else:
            logger.warning("\n⚠️ Some tests failed. Please check the logs.")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
