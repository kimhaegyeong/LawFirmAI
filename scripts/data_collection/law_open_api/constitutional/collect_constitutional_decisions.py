#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 수집 실행 스크립트

선고일자 오름차순으로 100개 단위 배치로 헌재결정례를 수집하고
데이터베이스와 벡터 저장소에 저장하는 스크립트입니다.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIClient
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from scripts.data_collection.constitutional.constitutional_decision_collector import (
    ConstitutionalDecisionCollector, CollectionConfig
)

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성
    log_filename = f'logs/constitutional_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, encoding='utf-8')
        ]
    )
    
    # 로그 파일 경로 출력
    print(f"📝 로그 파일: {log_filename}")
    return log_filename

# 로깅 초기화
log_file = setup_logging()

logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """환경 변수 검증"""
    oc_param = os.getenv("LAW_OPEN_API_OC")
    if not oc_param or oc_param == "{OC}":
        print("❌ LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        print("다음과 같이 설정해주세요:")
        print("export LAW_OPEN_API_OC='your_email@example.com'")
        return False
    
    print(f"✅ OC 파라미터: {oc_param}")
    return True


def test_api_connection(client: LawOpenAPIClient) -> bool:
    """API 연결 테스트"""
    try:
        print("API 연결 테스트 중...")
        if client.test_connection():
            print("✅ API 연결 테스트 성공")
            return True
        else:
            print("❌ API 연결 테스트 실패")
            return False
    except Exception as e:
        print(f"❌ API 연결 테스트 실패: {e}")
        return False


def collect_constitutional_decisions(
    keyword: str = "",
    max_count: int = 1000,
    batch_size: int = 100,
    include_details: bool = True,
    update_database: bool = True,
    update_vectors: bool = True,
    sort_order: str = "dasc"
) -> Dict[str, Any]:
    """
    헌재결정례 수집 실행
    
    Args:
        keyword: 검색 키워드
        max_count: 최대 수집 개수
        batch_size: 배치 크기
        include_details: 상세 정보 포함 여부
        update_database: 데이터베이스 업데이트 여부
        update_vectors: 벡터 저장소 업데이트 여부
        sort_order: 정렬 순서
        
    Returns:
        Dict: 수집 결과
    """
    logger.info("=" * 60)
    logger.info("헌재결정례 수집 시작")
    logger.info(f"키워드: '{keyword}'")
    logger.info(f"최대 개수: {max_count:,}개")
    logger.info(f"배치 크기: {batch_size}개")
    logger.info(f"상세 정보: {'포함' if include_details else '제외'}")
    logger.info(f"정렬 순서: {sort_order}")
    logger.info(f"데이터베이스 업데이트: {'예' if update_database else '아니오'}")
    logger.info(f"벡터 저장소 업데이트: {'예' if update_vectors else '아니오'}")
    logger.info("=" * 60)
    
    result = {
        "status": "success",
        "total_collected": 0,
        "database_updated": False,
        "vectors_updated": False,
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    try:
        # 수집기 설정
        logger.info("수집기 설정 중...")
        config = CollectionConfig(
            batch_size=batch_size,
            include_details=include_details,
            sort_order=sort_order,
            save_batches=True
        )
        
        collector = ConstitutionalDecisionCollector(config)
        logger.info("수집기 초기화 완료")
        
        print(f"\n헌재결정례 수집 시작")
        print(f"키워드: '{keyword}'")
        print(f"최대 개수: {max_count:,}개")
        print(f"배치 크기: {batch_size}개")
        print(f"상세 정보: {'포함' if include_details else '제외'}")
        print(f"정렬 순서: {sort_order}")
        print("=" * 50)
        
        # 데이터 수집
        logger.info("데이터 수집 시작...")
        collection_start_time = datetime.now()
        
        if keyword:
            logger.info(f"키워드 기반 수집: '{keyword}'")
            decisions = collector.collect_decisions_by_keyword(
                keyword=keyword,
                max_count=max_count,
                include_details=include_details
            )
        else:
            logger.info("전체 헌재결정례 수집")
            decisions = collector.collect_all_decisions(
                query=keyword,
                include_details=include_details
            )
        
        collection_end_time = datetime.now()
        collection_duration = (collection_end_time - collection_start_time).total_seconds()
        
        result["total_collected"] = len(decisions)
        
        logger.info(f"데이터 수집 완료: {len(decisions):,}개 ({collection_duration:.2f}초)")
        
        if not decisions:
            logger.error("수집된 헌재결정례가 없습니다.")
            print("❌ 수집된 헌재결정례가 없습니다.")
            result["status"] = "failed"
            result["errors"].append("No decisions collected")
            return result
        
        print(f"\n✅ 수집 완료: {len(decisions):,}개 헌재결정례")
        
        # 데이터베이스 업데이트
        if update_database:
            logger.info("데이터베이스 업데이트 시작...")
            print("\n데이터베이스 업데이트 중...")
            try:
                db_start_time = datetime.now()
                db_manager = DatabaseManager()
                
                # 배치별로 데이터베이스에 삽입
                batch_count = 0
                total_inserted = 0
                for i in range(0, len(decisions), batch_size):
                    batch = decisions[i:i + batch_size]
                    batch_start_time = datetime.now()
                    inserted_count = db_manager.insert_constitutional_decisions_batch(batch)
                    batch_end_time = datetime.now()
                    batch_duration = (batch_end_time - batch_start_time).total_seconds()
                    
                    batch_count += 1
                    total_inserted += inserted_count
                    
                    logger.info(f"데이터베이스 배치 {batch_count} 삽입: {inserted_count}개 ({batch_duration:.2f}초)")
                    print(f"  배치 {batch_count} 삽입: {inserted_count}개 ({batch_duration:.2f}초)")
                
                db_end_time = datetime.now()
                db_duration = (db_end_time - db_start_time).total_seconds()
                
                logger.info(f"데이터베이스 업데이트 완료: 총 {total_inserted:,}개 삽입 ({db_duration:.2f}초)")
                print(f"✅ 데이터베이스 업데이트 완료: 총 {total_inserted:,}개 삽입 ({db_duration:.2f}초)")
                result["database_updated"] = True
                
            except Exception as e:
                error_msg = f"데이터베이스 업데이트 실패: {e}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                result["errors"].append(error_msg)
        else:
            logger.info("데이터베이스 업데이트 건너뜀")
            print("데이터베이스 업데이트 건너뜀")
        
        # 벡터 저장소 업데이트
        if update_vectors:
            logger.info("벡터 저장소 업데이트 시작...")
            print("\n벡터 저장소 업데이트 중...")
            try:
                vector_start_time = datetime.now()
                vector_store = LegalVectorStore()
                
                # 배치별로 벡터 저장소에 추가
                batch_count = 0
                successful_batches = 0
                for i in range(0, len(decisions), batch_size):
                    batch = decisions[i:i + batch_size]
                    batch_start_time = datetime.now()
                    success = vector_store.add_constitutional_decisions(batch)
                    batch_end_time = datetime.now()
                    batch_duration = (batch_end_time - batch_start_time).total_seconds()
                    
                    batch_count += 1
                    if success:
                        successful_batches += 1
                    
                    logger.info(f"벡터 저장소 배치 {batch_count} 처리: {'성공' if success else '실패'} ({batch_duration:.2f}초)")
                    print(f"  배치 {batch_count} 벡터화: {'성공' if success else '실패'} ({batch_duration:.2f}초)")
                
                vector_end_time = datetime.now()
                vector_duration = (vector_end_time - vector_start_time).total_seconds()
                
                logger.info(f"벡터 저장소 업데이트 완료: {successful_batches}/{batch_count} 배치 성공 ({vector_duration:.2f}초)")
                print(f"✅ 벡터 저장소 업데이트 완료: {successful_batches}/{batch_count} 배치 성공 ({vector_duration:.2f}초)")
                result["vectors_updated"] = True
                
            except Exception as e:
                error_msg = f"벡터 저장소 업데이트 실패: {e}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                result["errors"].append(error_msg)
        else:
            logger.info("벡터 저장소 업데이트 건너뜀")
            print("벡터 저장소 업데이트 건너뜀")
        
        # 통계 출력
        logger.info("수집 통계 생성 중...")
        print("\n📊 수집 통계:")
        stats = collector.get_collection_stats()
        print(f"  총 수집: {stats['total_collected']:,}개")
        print(f"  API 요청: {stats['api_requests_made']:,}회")
        print(f"  배치 수: {stats['batch_count']:,}개")
        if stats['errors']:
            print(f"  오류: {len(stats['errors'])}개")
        
        logger.info(f"수집 통계: 총 {stats['total_collected']:,}개, API 요청 {stats['api_requests_made']:,}회, 배치 {stats['batch_count']:,}개")
        
        # 데이터베이스 통계
        if update_database:
            try:
                logger.info("데이터베이스 통계 조회 중...")
                db_stats = db_manager.get_constitutional_decisions_stats()
                print(f"\n📊 데이터베이스 통계:")
                print(f"  총 헌재결정례: {db_stats['total_count']:,}개")
                print(f"  연도별 분포: {len(db_stats['by_year'])}개 연도")
                print(f"  사건종류별 분포: {len(db_stats['by_type'])}개 종류")
                logger.info(f"데이터베이스 통계: 총 {db_stats['total_count']:,}개, 연도 {len(db_stats['by_year'])}개, 종류 {len(db_stats['by_type'])}개")
            except Exception as e:
                logger.warning(f"데이터베이스 통계 조회 실패: {e}")
        
        # 벡터 저장소 통계
        if update_vectors:
            try:
                logger.info("벡터 저장소 통계 조회 중...")
                vector_stats = vector_store.get_constitutional_decisions_stats()
                print(f"\n📊 벡터 저장소 통계:")
                print(f"  헌재결정례 벡터: {vector_stats['total_constitutional_decisions']:,}개")
                print(f"  전체 문서 비율: {vector_stats['constitutional_ratio']:.2%}")
                logger.info(f"벡터 저장소 통계: 헌재결정례 {vector_stats['total_constitutional_decisions']:,}개, 비율 {vector_stats['constitutional_ratio']:.2%}")
            except Exception as e:
                logger.warning(f"벡터 저장소 통계 조회 실패: {e}")
        
        # 최종 결과 로그
        result["end_time"] = datetime.now().isoformat()
        total_duration = (datetime.now() - datetime.fromisoformat(result["start_time"])).total_seconds()
        result["total_duration"] = total_duration
        
        logger.info("=" * 60)
        logger.info("헌재결정례 수집 완료")
        logger.info(f"총 수집: {result['total_collected']:,}개")
        logger.info(f"데이터베이스 업데이트: {'성공' if result['database_updated'] else '실패'}")
        logger.info(f"벡터 저장소 업데이트: {'성공' if result['vectors_updated'] else '실패'}")
        logger.info(f"총 소요 시간: {total_duration:.2f}초")
        if result['errors']:
            logger.warning(f"오류 발생: {len(result['errors'])}개")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"수집 과정에서 오류 발생: {e}"
        print(f"❌ {error_msg}")
        result["status"] = "failed"
        result["errors"].append(error_msg)
        logger.error(error_msg)
    
    finally:
        result["end_time"] = datetime.now().isoformat()
    
    return result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="헌재결정례 수집 스크립트")
    
    # 기본 옵션
    parser.add_argument("--keyword", type=str, default="", 
                       help="검색 키워드 (기본값: 빈 문자열 - 모든 결정례)")
    parser.add_argument("--max-count", type=int, default=1000, 
                       help="최대 수집 개수 (기본값: 1000)")
    parser.add_argument("--batch-size", type=int, default=100, 
                       help="배치 크기 (기본값: 100)")
    parser.add_argument("--sort-order", type=str, default="dasc", 
                       choices=["dasc", "ddes", "lasc", "ldes", "nasc", "ndes", "efasc", "efdes"],
                       help="정렬 순서 (기본값: dasc - 선고일자 오름차순)")
    
    # 기능 옵션
    parser.add_argument("--no-details", action="store_true", 
                       help="상세 정보 제외")
    parser.add_argument("--no-database", action="store_true", 
                       help="데이터베이스 업데이트 제외")
    parser.add_argument("--no-vectors", action="store_true", 
                       help="벡터 저장소 업데이트 제외")
    
    # 테스트 옵션
    parser.add_argument("--test", action="store_true", 
                       help="API 연결 테스트만 실행")
    parser.add_argument("--sample", type=int, default=0, 
                       help="샘플 수집 (지정된 개수만 수집)")
    
    args = parser.parse_args()
    
    print("헌재결정례 수집 스크립트")
    print("=" * 50)
    
    # 환경 검증
    if not validate_environment():
        return 1
    
    # API 클라이언트 생성
    try:
        client = LawOpenAPIClient()
    except Exception as e:
        print(f"❌ API 클라이언트 생성 실패: {e}")
        return 1
    
    # API 연결 테스트
    if not test_api_connection(client):
        return 1
    
    # 테스트 모드
    if args.test:
        print("\n✅ API 연결 테스트 완료")
        return 0
    
    # 샘플 수집 모드
    if args.sample > 0:
        print(f"\n샘플 수집 모드: {args.sample}개")
        args.max_count = args.sample
    
    # 수집 실행
    try:
        result = collect_constitutional_decisions(
            keyword=args.keyword,
            max_count=args.max_count,
            batch_size=args.batch_size,
            include_details=not args.no_details,
            update_database=not args.no_database,
            update_vectors=not args.no_vectors,
            sort_order=args.sort_order
        )
        
        # 결과 저장
        result_file = f"results/constitutional_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("results").mkdir(exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
        # 최종 결과
        if result["status"] == "success":
            print(f"\n✅ 헌재결정례 수집 완료!")
            print(f"   수집: {result['total_collected']:,}개")
            print(f"   데이터베이스: {'업데이트됨' if result['database_updated'] else '제외됨'}")
            print(f"   벡터 저장소: {'업데이트됨' if result['vectors_updated'] else '제외됨'}")
            return 0
        else:
            print(f"\n❌ 헌재결정례 수집 실패")
            if result["errors"]:
                print("오류 목록:")
                for error in result["errors"]:
                    print(f"  - {error}")
            return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의한 중단")
        return 0
    except Exception as e:
        print(f"\n❌ 스크립트 실행 실패: {e}")
        logger.error(f"스크립트 실행 실패: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)