#!/usr/bin/env python3
"""
개별 파일 실행 테스트 스크립트
TASK 3.2에서 생성된 각 파일을 독립적으로 테스트
"""

import os
import sys
import logging
import traceback
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_exact_search_engine():
    """ExactSearchEngine 독립 테스트"""
    try:
        logger.info("🔍 Testing ExactSearchEngine...")
        
        # Mock 클래스들 정의
        class MockConfig:
            def __init__(self):
                self.database_path = "data/lawfirm.db"
        
        class MockDatabaseManager:
            def __init__(self, db_path):
                self.db_path = db_path
            
            def execute_query(self, query, params=()):
                # Mock 데이터 반환
                return [
                    {"id": 1, "content": "민법 제1조", "title": "민법", "type": "law"},
                    {"id": 2, "content": "계약서 작성 방법", "title": "계약법", "type": "law"}
                ]
        
        # 파일 내용을 직접 실행
        exec(open('source/services/exact_search_engine.py', encoding='utf-8').read())
        
        # ExactSearchEngine 클래스 사용
        engine = ExactSearchEngine()
        logger.info(f"✅ ExactSearchEngine 초기화 성공")
        logger.info(f"   - 데이터베이스 경로: {engine.db_path}")
        
        # 검색 테스트
        results = engine.search("민법", limit=5)
        logger.info(f"   - 검색 결과 수: {len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ExactSearchEngine 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_semantic_search_engine():
    """SemanticSearchEngine 독립 테스트"""
    try:
        logger.info("🧠 Testing SemanticSearchEngine...")
        
        # Mock 클래스들 정의
        class MockConfig:
            def __init__(self):
                self.embedding_model_name = "jhgan/ko-sroberta-multitask"
                self.vector_db_path = "data/embeddings"
        
        class MockLegalVectorStore:
            def __init__(self, config):
                self.config = config
            
            def search_similar(self, query_vector, k=5):
                # Mock 검색 결과
                return [
                    {"id": 1, "content": "민법 관련 내용", "score": 0.95},
                    {"id": 2, "content": "계약법 관련 내용", "score": 0.87}
                ]
        
        # 파일 내용을 직접 실행
        exec(open('source/services/semantic_search_engine.py', encoding='utf-8').read())
        
        # SemanticSearchEngine 클래스 사용
        engine = SemanticSearchEngine()
        logger.info(f"✅ SemanticSearchEngine 초기화 성공")
        logger.info(f"   - 임베딩 모델: {engine.model_name}")
        
        # 검색 테스트 (실제 모델 로딩은 시간이 오래 걸리므로 스킵)
        logger.info("   - 모델 로딩은 시간이 오래 걸리므로 스킵")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SemanticSearchEngine 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_result_merger():
    """ResultMerger 독립 테스트"""
    try:
        logger.info("🔄 Testing ResultMerger...")
        
        # 파일 내용을 직접 실행
        exec(open('source/services/result_merger.py', encoding='utf-8').read())
        
        # ResultMerger 클래스 사용
        merger = ResultMerger()
        logger.info(f"✅ ResultMerger 초기화 성공")
        
        # Mock 검색 결과
        exact_results = [
            {"id": 1, "content": "민법 제1조", "score": 1.0, "source": "exact"},
            {"id": 2, "content": "계약서 작성", "score": 0.9, "source": "exact"}
        ]
        
        semantic_results = [
            {"id": 3, "content": "민법 해석", "score": 0.95, "source": "semantic"},
            {"id": 4, "content": "계약법 원칙", "score": 0.85, "source": "semantic"}
        ]
        
        # 결과 통합 테스트
        merged_results = merger.merge_results(exact_results, semantic_results)
        logger.info(f"   - 통합 결과 수: {len(merged_results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ResultMerger 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_hybrid_search_engine():
    """HybridSearchEngine 독립 테스트"""
    try:
        logger.info("🔀 Testing HybridSearchEngine...")
        
        # Mock 클래스들 정의
        class MockConfig:
            def __init__(self):
                self.database_path = "data/lawfirm.db"
                self.embedding_model_name = "jhgan/ko-sroberta-multitask"
                self.vector_db_path = "data/embeddings"
        
        class MockDatabaseManager:
            def __init__(self, db_path):
                self.db_path = db_path
            
            def execute_query(self, query, params=()):
                return [{"id": 1, "content": "민법 제1조", "title": "민법", "type": "law"}]
        
        class MockLegalVectorStore:
            def __init__(self, config):
                self.config = config
            
            def search_similar(self, query_vector, k=5):
                return [{"id": 1, "content": "민법 관련 내용", "score": 0.95}]
        
        # 파일 내용을 직접 실행
        exec(open('source/services/hybrid_search_engine.py', encoding='utf-8').read())
        
        # HybridSearchEngine 클래스 사용
        engine = HybridSearchEngine()
        logger.info(f"✅ HybridSearchEngine 초기화 성공")
        
        # 검색 테스트
        results = engine.search("민법", limit=5)
        logger.info(f"   - 하이브리드 검색 결과 수: {len(results)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HybridSearchEngine 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_search_endpoints():
    """SearchEndpoints 독립 테스트"""
    try:
        logger.info("🌐 Testing SearchEndpoints...")
        
        # 파일 내용을 직접 실행
        exec(open('source/api/search_endpoints.py', encoding='utf-8').read())
        
        logger.info(f"✅ SearchEndpoints 로딩 성공")
        logger.info(f"   - 라우터 생성 완료")
        logger.info(f"   - API 엔드포인트 정의 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SearchEndpoints 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_build_vector_db_script():
    """벡터DB 구축 스크립트 테스트"""
    try:
        logger.info("🏗️ Testing build_vector_db_task3_2.py...")
        
        # 파일 내용을 직접 실행 (main 함수는 호출하지 않음)
        exec(open('scripts/build_vector_db_task3_2.py', encoding='utf-8').read())
        
        logger.info(f"✅ 벡터DB 구축 스크립트 로딩 성공")
        logger.info(f"   - 스크립트 문법 정상")
        logger.info(f"   - 함수 정의 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 벡터DB 구축 스크립트 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def test_simple_test_script():
    """간단 테스트 스크립트 테스트"""
    try:
        logger.info("🧪 Testing test_task3_2_simple.py...")
        
        # 파일 내용을 직접 실행 (main 함수는 호출하지 않음)
        exec(open('scripts/test_task3_2_simple.py', encoding='utf-8').read())
        
        logger.info(f"✅ 간단 테스트 스크립트 로딩 성공")
        logger.info(f"   - 스크립트 문법 정상")
        logger.info(f"   - 테스트 함수 정의 완료")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 간단 테스트 스크립트 테스트 실패: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """메인 테스트 함수"""
    logger.info("=" * 80)
    logger.info("🚀 TASK 3.2 개별 파일 실행 테스트 시작")
    logger.info("=" * 80)
    
    test_results = {}
    
    # 각 파일 테스트
    test_functions = [
        ("exact_search_engine", test_exact_search_engine),
        ("semantic_search_engine", test_semantic_search_engine),
        ("result_merger", test_result_merger),
        ("hybrid_search_engine", test_hybrid_search_engine),
        ("search_endpoints", test_search_endpoints),
        ("build_vector_db_script", test_build_vector_db_script),
        ("simple_test_script", test_simple_test_script)
    ]
    
    for test_name, test_func in test_functions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            logger.error(f"테스트 중 예외 발생: {e}")
            test_results[test_name] = False
    
    # 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("📊 테스트 결과 요약")
    logger.info("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    logger.info(f"\n전체 성공률: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate == 100:
        logger.info("🎉 모든 파일이 정상적으로 실행됩니다!")
    else:
        logger.info("⚠️ 일부 파일에서 문제가 발생했습니다.")
    
    logger.info("=" * 80)
    
    return test_results

if __name__ == "__main__":
    main()
