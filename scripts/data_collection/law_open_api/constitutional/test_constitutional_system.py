#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 시스템 테스트 스크립트

구현된 헌재결정례 수집, 데이터베이스 저장, 벡터 검색 기능을 테스트합니다.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 현재 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(project_root)

from source.data.law_open_api_client import LawOpenAPIClient
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from scripts.data_collection.constitutional.constitutional_decision_collector import ConstitutionalDecisionCollector
from scripts.data_collection.constitutional.constitutional_checkpoint_manager import ConstitutionalCheckpointManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_client():
    """API 클라이언트 테스트"""
    print("🔍 API 클라이언트 테스트")
    print("-" * 30)
    
    try:
        client = LawOpenAPIClient()
        
        # 연결 테스트
        if client.test_connection():
            print("✅ API 연결 테스트 성공")
        else:
            print("❌ API 연결 테스트 실패")
            return False
        
        # 헌재결정례 목록 조회 테스트
        print("\n📋 헌재결정례 목록 조회 테스트")
        response = client.search_constitutional_decisions(
            query="헌법",
            display=5,
            sort="dasc"
        )
        
        if response and 'DetcSearch' in response:
            decisions = response['DetcSearch'].get('detc', [])
            if isinstance(decisions, dict):
                decisions = [decisions]
            
            print(f"✅ 목록 조회 성공: {len(decisions)}개")
            
            # 첫 번째 결정례의 상세 정보 조회 테스트
            if decisions:
                first_decision = decisions[0]
                decision_id = first_decision.get('헌재결정례일련번호')
                
                if decision_id:
                    print(f"\n📄 상세 정보 조회 테스트 (ID: {decision_id})")
                    detail = client.get_constitutional_decision_detail(decision_id)
                    
                    if detail and 'error' not in detail:
                        print("✅ 상세 정보 조회 성공")
                    else:
                        print("❌ 상세 정보 조회 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ API 클라이언트 테스트 실패: {e}")
        return False


def test_database():
    """데이터베이스 테스트"""
    print("\n🗄️ 데이터베이스 테스트")
    print("-" * 30)
    
    try:
        db_manager = DatabaseManager()
        
        # 테이블 생성 확인
        print("📊 테이블 생성 확인")
        
        # 헌재결정례 테이블 존재 확인
        tables_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='constitutional_decisions'
        """
        result = db_manager.execute_query(tables_query)
        
        if result:
            print("✅ 헌재결정례 테이블 존재")
        else:
            print("❌ 헌재결정례 테이블 없음")
            return False
        
        # FTS 테이블 존재 확인
        fts_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='constitutional_decisions_fts'
        """
        result = db_manager.execute_query(fts_query)
        
        if result:
            print("✅ 헌재결정례 FTS 테이블 존재")
        else:
            print("❌ 헌재결정례 FTS 테이블 없음")
        
        # 샘플 데이터 삽입 테스트
        print("\n📝 샘플 데이터 삽입 테스트")
        sample_decision = {
            '헌재결정례일련번호': 999999,
            '사건명': '테스트 헌재결정례',
            '사건번호': '2024헌마999',
            '사건종류명': '헌법소원',
            '사건종류코드': 1,
            '재판부구분코드': 430201,
            '종국일자': '20241201',
            '판시사항': '테스트 판시사항',
            '결정요지': '테스트 결정요지',
            '전문': '테스트 전문 내용',
            '참조조문': '헌법 제10조',
            '참조판례': '테스트 판례',
            '심판대상조문': '테스트 대상 조문'
        }
        
        if db_manager.insert_constitutional_decision(sample_decision):
            print("✅ 샘플 데이터 삽입 성공")
            
            # 데이터 조회 테스트
            retrieved = db_manager.get_constitutional_decision_by_id(999999)
            if retrieved:
                print("✅ 데이터 조회 성공")
                
                # 테스트 데이터 삭제
                delete_query = "DELETE FROM constitutional_decisions WHERE decision_id = 999999"
                db_manager.execute_update(delete_query)
                print("✅ 테스트 데이터 삭제 완료")
            else:
                print("❌ 데이터 조회 실패")
        else:
            print("❌ 샘플 데이터 삽입 실패")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 테스트 실패: {e}")
        return False


def test_vector_store():
    """벡터 저장소 테스트"""
    print("\n🔍 벡터 저장소 테스트")
    print("-" * 30)
    
    try:
        vector_store = LegalVectorStore()
        
        # 샘플 헌재결정례 데이터
        sample_decisions = [
            {
                '헌재결정례일련번호': 999998,
                '사건명': '테스트 헌재결정례 1',
                '사건번호': '2024헌마998',
                '사건종류명': '헌법소원',
                '종국일자': '20241201',
                '판시사항': '표현의 자유에 관한 테스트 판시사항',
                '결정요지': '표현의 자유는 헌법상 기본권으로 보장된다는 테스트 결정요지',
                '전문': '표현의 자유에 관한 테스트 전문 내용입니다.',
                '참조조문': '헌법 제21조',
                '참조판례': '테스트 판례',
                '심판대상조문': '테스트 대상 조문'
            },
            {
                '헌재결정례일련번호': 999997,
                '사건명': '테스트 헌재결정례 2',
                '사건번호': '2024헌마997',
                '사건종류명': '위헌법률심판',
                '종국일자': '20241202',
                '판시사항': '평등권에 관한 테스트 판시사항',
                '결정요지': '평등권은 헌법상 기본권으로 보장된다는 테스트 결정요지',
                '전문': '평등권에 관한 테스트 전문 내용입니다.',
                '참조조문': '헌법 제11조',
                '참조판례': '테스트 판례',
                '심판대상조문': '테스트 대상 조문'
            }
        ]
        
        # 벡터 임베딩 추가 테스트
        print("📝 벡터 임베딩 추가 테스트")
        if vector_store.add_constitutional_decisions(sample_decisions):
            print("✅ 벡터 임베딩 추가 성공")
        else:
            print("❌ 벡터 임베딩 추가 실패")
            return False
        
        # 벡터 검색 테스트
        print("\n🔍 벡터 검색 테스트")
        search_results = vector_store.search_constitutional_decisions(
            query="표현의 자유",
            top_k=5
        )
        
        if search_results:
            print(f"✅ 벡터 검색 성공: {len(search_results)}개 결과")
            for i, result in enumerate(search_results[:2], 1):
                print(f"  {i}. {result.get('case_name', 'N/A')} (유사도: {result.get('similarity_score', 0):.3f})")
        else:
            print("❌ 벡터 검색 실패")
        
        # 유사 결정례 검색 테스트
        print("\n🔗 유사 결정례 검색 테스트")
        similar_results = vector_store.get_constitutional_decisions_by_similarity(
            decision_id=999998,
            top_k=3
        )
        
        if similar_results:
            print(f"✅ 유사 결정례 검색 성공: {len(similar_results)}개 결과")
            for i, result in enumerate(similar_results[:2], 1):
                print(f"  {i}. {result.get('case_name', 'N/A')} (유사도: {result.get('similarity_score', 0):.3f})")
        else:
            print("❌ 유사 결정례 검색 실패")
        
        # 통계 조회 테스트
        print("\n📊 벡터 저장소 통계 테스트")
        stats = vector_store.get_constitutional_decisions_stats()
        if stats:
            print(f"✅ 통계 조회 성공: 헌재결정례 {stats.get('total_constitutional_decisions', 0)}개")
        else:
            print("❌ 통계 조회 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 벡터 저장소 테스트 실패: {e}")
        return False


def test_collector():
    """수집기 테스트"""
    print("\n📥 수집기 테스트")
    print("-" * 30)
    
    try:
        collector = ConstitutionalDecisionCollector()
        
        # 키워드 기반 수집 테스트 (소량)
        print("🔍 키워드 기반 수집 테스트 (5개)")
        decisions = collector.collect_decisions_by_keyword(
            keyword="헌법",
            max_count=5,
            include_details=False
        )
        
        if decisions:
            print(f"✅ 키워드 기반 수집 성공: {len(decisions)}개")
            for i, decision in enumerate(decisions[:3], 1):
                print(f"  {i}. {decision.get('사건명', 'N/A')}")
        else:
            print("❌ 키워드 기반 수집 실패")
            return False
        
        # 통계 조회 테스트
        print("\n📊 수집 통계 테스트")
        stats = collector.get_collection_stats()
        if stats:
            print(f"✅ 통계 조회 성공: 수집 {stats.get('total_collected', 0)}개")
        else:
            print("❌ 통계 조회 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 수집기 테스트 실패: {e}")
        return False


def test_checkpoint_manager():
    """체크포인트 관리자 테스트"""
    print("\n💾 체크포인트 관리자 테스트")
    print("-" * 30)
    
    try:
        manager = ConstitutionalCheckpointManager()
        
        # 체크포인트 생성 테스트
        print("📝 체크포인트 생성 테스트")
        checkpoint_id = manager.create_checkpoint(
            collection_type="keyword",
            keyword="테스트",
            sort_order="dasc"
        )
        
        if checkpoint_id:
            print(f"✅ 체크포인트 생성 성공: {checkpoint_id}")
        else:
            print("❌ 체크포인트 생성 실패")
            return False
        
        # 체크포인트 로드 테스트
        print("\n📖 체크포인트 로드 테스트")
        checkpoint = manager.load_checkpoint(checkpoint_id)
        
        if checkpoint:
            print(f"✅ 체크포인트 로드 성공: {checkpoint.checkpoint_id}")
        else:
            print("❌ 체크포인트 로드 실패")
            return False
        
        # 체크포인트 업데이트 테스트
        print("\n🔄 체크포인트 업데이트 테스트")
        if manager.update_checkpoint(
            checkpoint_id,
            current_page=5,
            collected_count=100,
            status="in_progress"
        ):
            print("✅ 체크포인트 업데이트 성공")
        else:
            print("❌ 체크포인트 업데이트 실패")
            return False
        
        # 체크포인트 완료 테스트
        print("\n✅ 체크포인트 완료 테스트")
        if manager.complete_checkpoint(checkpoint_id):
            print("✅ 체크포인트 완료 처리 성공")
        else:
            print("❌ 체크포인트 완료 처리 실패")
        
        # 체크포인트 목록 조회 테스트
        print("\n📋 체크포인트 목록 조회 테스트")
        checkpoints = manager.list_checkpoints()
        
        if checkpoints:
            print(f"✅ 체크포인트 목록 조회 성공: {len(checkpoints)}개")
        else:
            print("❌ 체크포인트 목록 조회 실패")
        
        # 테스트 체크포인트 삭제
        print("\n🗑️ 테스트 체크포인트 삭제")
        if manager.delete_checkpoint(checkpoint_id):
            print("✅ 테스트 체크포인트 삭제 성공")
        else:
            print("❌ 테스트 체크포인트 삭제 실패")
        
        return True
        
    except Exception as e:
        print(f"❌ 체크포인트 관리자 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 헌재결정례 시스템 통합 테스트")
    print("=" * 50)
    
    # 환경 변수 확인
    if not os.getenv("LAW_OPEN_API_OC"):
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        return 1
    
    print(f"✅ 환경 변수 설정 확인: {os.getenv('LAW_OPEN_API_OC')}")
    
    # 테스트 실행
    tests = [
        ("API 클라이언트", test_api_client),
        ("데이터베이스", test_database),
        ("벡터 저장소", test_vector_store),
        ("수집기", test_collector),
        ("체크포인트 관리자", test_checkpoint_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print(f"통과: {passed}/{total}")
    print(f"성공률: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        return 0
    else:
        print("⚠️ 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
