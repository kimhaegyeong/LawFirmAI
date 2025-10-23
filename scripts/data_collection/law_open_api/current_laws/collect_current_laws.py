#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 데이터 수집 스크립트

현행법령 목록 조회 후 각 법령의 본문을 수집하여 배치 파일로 저장합니다.
데이터베이스나 벡터 저장소 업데이트는 별도 스크립트에서 처리합니다.
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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_collection.law_open_api.current_laws.current_law_collector import (
    CurrentLawCollector, CollectionConfig
)

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성
    log_filename = f'logs/current_laws_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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


def collect_current_laws_data(
    query: str = "",
    max_pages: int = None,
    start_page: int = 1,
    batch_size: int = 10,
    include_details: bool = True,
    sort_order: str = "ldes",
    resume_from_checkpoint: bool = False
) -> Dict[str, Any]:
    """
    현행법령 데이터 수집
    
    Args:
        query: 검색 질의
        max_pages: 최대 페이지 수
        batch_size: 배치 크기
        include_details: 상세 정보 포함 여부
        sort_order: 정렬 순서
        resume_from_checkpoint: 체크포인트에서 재시작 여부
        
    Returns:
        Dict: 수집 결과
    """
    logger.info("=" * 60)
    logger.info("현행법령 데이터 수집 시작")
    logger.info(f"검색어: '{query}'")
    logger.info(f"최대 페이지: {max_pages or '무제한'}")
    logger.info(f"배치 크기: {batch_size}개")
    logger.info(f"상세 정보: {'포함' if include_details else '제외'}")
    logger.info(f"정렬 순서: {sort_order}")
    logger.info(f"체크포인트 재시작: {'예' if resume_from_checkpoint else '아니오'}")
    logger.info("=" * 60)
    
    result = {
        "status": "success",
        "total_collected": 0,
        "collection_time": None,
        "batch_files": [],
        "summary_file": None,
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    try:
        # 수집기 설정
        config = CollectionConfig(
            batch_size=batch_size,
            include_details=include_details,
            sort_order=sort_order,
            save_batches=True,
            max_pages=max_pages,
            query=query,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        collector = CurrentLawCollector(config)
        logger.info("수집기 초기화 완료")
        
        print(f"\n현행법령 데이터 수집 시작")
        print(f"검색어: '{query}'")
        print(f"최대 페이지: {max_pages or '무제한'}")
        print(f"배치 크기: {batch_size}개")
        print(f"상세 정보: {'포함' if include_details else '제외'}")
        print(f"정렬 순서: {sort_order}")
        print("=" * 50)
        
        # 데이터 수집
        logger.info("데이터 수집 시작...")
        collection_start_time = datetime.now()
        
        if query:
            laws = collector.collect_laws_by_query(query, max_pages)
        else:
            laws = collector.collect_all_laws(max_pages, start_page)
        
        collection_end_time = datetime.now()
        collection_duration = (collection_end_time - collection_start_time).total_seconds()
        
        result["total_collected"] = len(laws)
        result["collection_time"] = collection_duration
        
        logger.info(f"데이터 수집 완료: {len(laws):,}개 ({collection_duration:.2f}초)")
        
        if not laws:
            logger.error("수집된 현행법령이 없습니다.")
            print("❌ 수집된 현행법령이 없습니다.")
            result["status"] = "failed"
            result["errors"].append("No laws collected")
            return result
        
        print(f"\n✅ 수집 완료: {len(laws):,}개 현행법령")
        
        # 배치 파일 목록 수집
        if config.save_batches:
            batch_dir = Path("data/raw/law_open_api/current_laws/batches")
            if batch_dir.exists():
                batch_files = list(batch_dir.glob(f"current_law_batch_{collector.timestamp}_*.json"))
                result["batch_files"] = [str(f) for f in batch_files]
                print(f"📁 배치 파일: {len(batch_files)}개")
        
        # 수집 보고서 저장
        report_file = collector.save_collection_report(laws)
        result["summary_file"] = report_file
        
        # 통계 출력
        logger.info("수집 통계 생성 중...")
        print("\n📊 수집 통계:")
        stats = collector.get_collection_stats()
        print(f"  총 수집: {stats['total_collected']:,}개")
        print(f"  수집 시간: {collection_duration:.2f}초")
        if stats['errors']:
            print(f"  오류: {len(stats['errors'])}개")
        
        logger.info(f"수집 통계: 총 {stats['total_collected']:,}개, 시간 {collection_duration:.2f}초")
        
        # 최종 결과 로그
        result["end_time"] = datetime.now().isoformat()
        total_duration = (datetime.now() - datetime.fromisoformat(result["start_time"])).total_seconds()
        result["total_duration"] = total_duration
        
        # datetime 객체들을 문자열로 변환
        result["start_time"] = result["start_time"]  # 이미 ISO 형식
        result["end_time"] = result["end_time"]  # 이미 ISO 형식
        
        logger.info("=" * 60)
        logger.info("현행법령 데이터 수집 완료")
        logger.info(f"총 수집: {result['total_collected']:,}개")
        logger.info(f"수집 시간: {collection_duration:.2f}초")
        logger.info(f"총 소요 시간: {total_duration:.2f}초")
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
    parser = argparse.ArgumentParser(description="현행법령 데이터 수집 스크립트")
    
    # 기본 옵션
    parser.add_argument("--query", type=str, default="", 
                       help="검색 질의 (기본값: 빈 문자열 - 모든 법령)")
    parser.add_argument("--max-pages", type=int, default=None, 
                       help="최대 페이지 수 (기본값: 무제한)")
    parser.add_argument("--start-page", type=int, default=1, 
                       help="시작 페이지 (기본값: 1)")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="배치 크기 (기본값: 10)")
    parser.add_argument("--sort-order", type=str, default="ldes", 
                       choices=["ldes", "lasc", "dasc", "ddes", "nasc", "ndes", "efasc", "efdes"],
                       help="정렬 순서 (기본값: ldes - 법령내림차순)")
    
    # 기능 옵션
    parser.add_argument("--no-details", action="store_true", 
                       help="상세 정보 제외")
    parser.add_argument("--resume-checkpoint", action="store_true", 
                       help="체크포인트에서 재시작")
    
    # 테스트 옵션
    parser.add_argument("--test", action="store_true", 
                       help="API 연결 테스트만 실행")
    parser.add_argument("--sample", type=int, default=0, 
                       help="샘플 수집 (지정된 개수만 수집)")
    
    args = parser.parse_args()
    
    print("현행법령 데이터 수집 스크립트")
    print("=" * 50)
    
    # 환경 검증
    if not validate_environment():
        return 1
    
    # API 클라이언트 테스트
    try:
        from source.data.law_open_api_client import LawOpenAPIClient
        client = LawOpenAPIClient()
        if not client.test_connection():
            print("❌ API 연결 테스트 실패")
            return 1
        print("✅ API 연결 테스트 성공")
    except Exception as e:
        print(f"❌ API 클라이언트 생성 실패: {e}")
        return 1
    
    # 테스트 모드
    if args.test:
        print("\n✅ API 연결 테스트 완료")
        return 0
    
    # 샘플 수집 모드
    if args.sample > 0:
        print(f"\n샘플 수집 모드: {args.sample}개")
        args.max_pages = max(1, args.sample // 100)  # 페이지당 100개씩
    
    # 수집 실행
    try:
        result = collect_current_laws_data(
            query=args.query,
            max_pages=args.max_pages,
            start_page=args.start_page,
            batch_size=args.batch_size,
            include_details=not args.no_details,
            sort_order=args.sort_order,
            resume_from_checkpoint=args.resume_checkpoint
        )
        
        # 결과 저장
        result_file = f"results/current_laws_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("results").mkdir(exist_ok=True)
        
        # datetime 객체를 문자열로 변환
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # 결과 딕셔너리의 모든 datetime 객체를 문자열로 변환
        serializable_result = json.loads(json.dumps(result, default=convert_datetime, ensure_ascii=False))
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
        # 최종 결과
        if result["status"] == "success":
            print(f"\n✅ 현행법령 데이터 수집 완료!")
            print(f"   수집: {result['total_collected']:,}개")
            print(f"   배치 파일: {len(result['batch_files'])}개")
            print(f"   수집 시간: {result['collection_time']:.2f}초")
            print(f"\n📋 다음 단계:")
            print(f"   1. 데이터베이스 업데이트: python scripts/data_collection/law_open_api/current_laws/update_database.py")
            print(f"   2. 벡터 저장소 업데이트: python scripts/data_collection/law_open_api/current_laws/update_vectors.py")
            return 0
        else:
            print(f"\n❌ 현행법령 데이터 수집 실패")
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
