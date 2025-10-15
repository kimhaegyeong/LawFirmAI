#!/usr/bin/env python3
"""
벡터 임베딩 최종 성능 테스트
"""

import sys
import os
from pathlib import Path
import time
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore
from source.services.rag_service import MLEnhancedRAGService
from source.services.search_service import MLEnhancedSearchService
from source.data.database import DatabaseManager
from source.models.model_manager import LegalModelManager
from source.utils.config import Config

def test_vector_store():
    """벡터 스토어 테스트"""
    print("🔍 벡터 스토어 테스트 시작...")
    
    try:
        # 벡터 스토어 로드
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        
        # 인덱스 로드 (올바른 경로 사용)
        index_path = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index"
        vector_store.load_index(index_path)
        
        # 통계 확인
        stats = vector_store.get_stats()
        print(f"✅ 벡터 스토어 로드 성공")
        print(f"   - 총 문서 수: {stats.get('documents_count', 0):,}")
        print(f"   - 인덱스 크기: {stats.get('index_size', 0):,}")
        print(f"   - 모델명: {stats.get('model_name', 'Unknown')}")
        
        # 검색 테스트
        test_queries = [
            "계약서 해지 조건",
            "손해배상 책임",
            "부칙 시행일",
            "민법 제1조",
            "상법 회사"
        ]
        
        print(f"\n🔍 검색 테스트 ({len(test_queries)}개 쿼리)...")
        total_search_time = 0
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            results = vector_store.search(query, top_k=5)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            print(f"   {i}. '{query}' → {len(results)}개 결과 ({search_time:.3f}초)")
            
            if results:
                best_result = results[0]
                print(f"      최고 점수: {best_result.get('score', 0):.3f}")
                print(f"      문서 ID: {best_result.get('metadata', {}).get('document_id', 'Unknown')}")
        
        avg_search_time = total_search_time / len(test_queries)
        print(f"\n📊 검색 성능:")
        print(f"   - 평균 검색 시간: {avg_search_time:.3f}초")
        print(f"   - 총 검색 시간: {total_search_time:.3f}초")
        
        return True
        
    except Exception as e:
        print(f"❌ 벡터 스토어 테스트 실패: {e}")
        return False

def test_database_integration():
    """데이터베이스 통합 테스트"""
    print("\n🗄️ 데이터베이스 통합 테스트 시작...")
    
    try:
        # 데이터베이스 경로 수정
        db_path = os.path.abspath("data/lawfirm.db")
        database = DatabaseManager(db_path)
        
        # ML 강화 데이터 확인
        ml_stats_query = """
            SELECT 
                COUNT(*) as total_articles,
                SUM(CASE WHEN ml_enhanced = 1 THEN 1 ELSE 0 END) as ml_enhanced_articles,
                AVG(parsing_quality_score) as avg_quality_score,
                SUM(CASE WHEN article_type = 'main' THEN 1 ELSE 0 END) as main_articles,
                SUM(CASE WHEN article_type = 'supplementary' THEN 1 ELSE 0 END) as supplementary_articles
            FROM assembly_articles
        """
        
        result = database.execute_query(ml_stats_query)
        stats = result[0] if result else {}
        
        print(f"✅ 데이터베이스 통합 성공")
        print(f"   - 총 조문 수: {stats.get('total_articles', 0):,}")
        print(f"   - ML 강화 조문: {stats.get('ml_enhanced_articles', 0):,}")
        print(f"   - 평균 품질 점수: {stats.get('avg_quality_score', 0):.3f}")
        print(f"   - 본칙 조문: {stats.get('main_articles', 0):,}")
        print(f"   - 부칙 조문: {stats.get('supplementary_articles', 0):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 통합 테스트 실패: {e}")
        return False

def test_rag_service():
    """RAG 서비스 테스트"""
    print("\n🤖 RAG 서비스 테스트 시작...")
    
    try:
        # 설정 및 서비스 초기화
        config = Config()
        db_path = os.path.abspath("data/lawfirm.db")
        database = DatabaseManager(db_path)
        
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index")
        
        model_manager = LegalModelManager(config)
        rag_service = MLEnhancedRAGService(config, model_manager, vector_store, database)
        
        # RAG 테스트 쿼리
        test_queries = [
            "계약서 해지 조건에 대해 설명해주세요",
            "손해배상 책임의 범위는 어떻게 되나요?",
            "부칙의 시행일은 언제인가요?"
        ]
        
        print(f"🔍 RAG 테스트 ({len(test_queries)}개 쿼리)...")
        total_rag_time = 0
        
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            result = rag_service.process_query(query, top_k=3)
            rag_time = time.time() - start_time
            total_rag_time += rag_time
            
            print(f"   {i}. '{query}'")
            print(f"      응답 길이: {len(result.get('response', ''))}자")
            print(f"      신뢰도: {result.get('confidence', 0):.3f}")
            print(f"      소스 수: {len(result.get('sources', []))}")
            print(f"      처리 시간: {rag_time:.3f}초")
            print(f"      ML 강화: {result.get('ml_enhanced', False)}")
            
            if result.get('ml_stats'):
                ml_stats = result['ml_stats']
                print(f"      ML 통계: 품질 {ml_stats.get('avg_quality_score', 0):.3f}, 신뢰도 {ml_stats.get('avg_ml_confidence', 0):.3f}")
        
        avg_rag_time = total_rag_time / len(test_queries)
        print(f"\n📊 RAG 성능:")
        print(f"   - 평균 처리 시간: {avg_rag_time:.3f}초")
        print(f"   - 총 처리 시간: {total_rag_time:.3f}초")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG 서비스 테스트 실패: {e}")
        return False

def test_search_service():
    """검색 서비스 테스트"""
    print("\n🔍 검색 서비스 테스트 시작...")
    
    try:
        # 설정 및 서비스 초기화
        config = Config()
        db_path = os.path.abspath("data/lawfirm.db")
        database = DatabaseManager(db_path)
        
        vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index")
        
        model_manager = LegalModelManager(config)
        search_service = MLEnhancedSearchService(config, database, vector_store, model_manager)
        
        # 검색 테스트
        search_types = ["semantic", "keyword", "hybrid", "supplementary", "high_quality"]
        
        print(f"🔍 검색 타입별 테스트 ({len(search_types)}개 타입)...")
        total_search_time = 0
        
        for search_type in search_types:
            start_time = time.time()
            results = search_service.search_documents(
                "계약서 해지 조건", 
                search_type=search_type, 
                limit=5
            )
            search_time = time.time() - start_time
            total_search_time += search_time
            
            print(f"   {search_type}: {len(results)}개 결과 ({search_time:.3f}초)")
            
            if results:
                best_result = results[0]
                print(f"      최고 점수: {best_result.get('similarity', best_result.get('hybrid_score', 0)):.3f}")
                print(f"      품질 점수: {best_result.get('quality_score', 0):.3f}")
        
        avg_search_time = total_search_time / len(search_types)
        print(f"\n📊 검색 서비스 성능:")
        print(f"   - 평균 검색 시간: {avg_search_time:.3f}초")
        print(f"   - 총 검색 시간: {total_search_time:.3f}초")
        
        return True
        
    except Exception as e:
        print(f"❌ 검색 서비스 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 벡터 임베딩 최종 성능 테스트 시작")
    print("=" * 50)
    
    test_results = []
    
    # 1. 벡터 스토어 테스트
    test_results.append(("벡터 스토어", test_vector_store()))
    
    # 2. 데이터베이스 통합 테스트
    test_results.append(("데이터베이스 통합", test_database_integration()))
    
    # 3. RAG 서비스 테스트
    test_results.append(("RAG 서비스", test_rag_service()))
    
    # 4. 검색 서비스 테스트
    test_results.append(("검색 서비스", test_search_service()))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 최종 테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 전체 결과: {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 벡터 임베딩 시스템이 정상적으로 작동합니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 시스템 점검이 필요합니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)