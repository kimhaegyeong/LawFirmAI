#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 데이터베이스 업데이트 스크립트

수집된 현행법령 배치 파일을 읽어서 데이터베이스에 저장합니다.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from source.data.database import DatabaseManager

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성
    log_filename = f'logs/current_laws_database_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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


def load_batch_files(batch_dir: str, pattern: str = "current_law_batch_*.json") -> List[Dict[str, Any]]:
    """
    배치 파일들을 로드하여 현행법령 데이터 반환
    
    Args:
        batch_dir: 배치 파일 디렉토리
        pattern: 파일 패턴
        
    Returns:
        List[Dict]: 현행법령 목록
    """
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        logger.error(f"배치 디렉토리가 존재하지 않습니다: {batch_dir}")
        return []
    
    batch_files = list(batch_path.glob(pattern))
    if not batch_files:
        logger.warning(f"배치 파일이 없습니다: {batch_dir}/{pattern}")
        return []
    
    all_laws = []
    loaded_files = []
    
    logger.info(f"배치 파일 {len(batch_files)}개 발견")
    print(f"📁 배치 파일 {len(batch_files)}개 발견")
    
    for batch_file in sorted(batch_files):
        try:
            logger.info(f"배치 파일 로드 중: {batch_file.name}")
            print(f"  📄 로드 중: {batch_file.name}")
            
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            if 'laws' in batch_data:
                laws = batch_data['laws']
                all_laws.extend(laws)
                loaded_files.append(str(batch_file))
                logger.info(f"배치 파일 로드 완료: {batch_file.name} ({len(laws)}개)")
                print(f"    ✅ {len(laws)}개 법령 로드")
            else:
                logger.warning(f"배치 파일에 'laws' 키가 없습니다: {batch_file.name}")
                print(f"    ⚠️ 'laws' 키 없음")
                
        except Exception as e:
            logger.error(f"배치 파일 로드 실패: {batch_file.name} - {e}")
            print(f"    ❌ 로드 실패: {e}")
    
    logger.info(f"총 {len(all_laws)}개 현행법령 로드 완료")
    print(f"✅ 총 {len(all_laws)}개 현행법령 로드 완료")
    
    return all_laws, loaded_files


def load_summary_file(summary_file: str) -> Optional[Dict[str, Any]]:
    """
    요약 파일 로드
    
    Args:
        summary_file: 요약 파일 경로
        
    Returns:
        Dict: 요약 데이터 또는 None
    """
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        logger.info(f"요약 파일 로드 완료: {summary_file}")
        return summary_data
        
    except Exception as e:
        logger.error(f"요약 파일 로드 실패: {summary_file} - {e}")
        return None


def update_database_with_laws(
    laws: List[Dict[str, Any]], 
    batch_size: int = 100,
    clear_existing: bool = False
) -> Dict[str, Any]:
    """
    현행법령 데이터를 데이터베이스에 저장
    
    Args:
        laws: 현행법령 목록
        batch_size: 배치 크기
        clear_existing: 기존 데이터 삭제 여부
        
    Returns:
        Dict: 업데이트 결과
    """
    logger.info("=" * 60)
    logger.info("데이터베이스 업데이트 시작")
    logger.info(f"총 법령 수: {len(laws):,}개")
    logger.info(f"배치 크기: {batch_size}개")
    logger.info(f"기존 데이터 삭제: {'예' if clear_existing else '아니오'}")
    logger.info("=" * 60)
    
    result = {
        "status": "success",
        "total_processed": 0,
        "total_inserted": 0,
        "batch_count": 0,
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    try:
        # 데이터베이스 관리자 초기화
        db_manager = DatabaseManager()
        logger.info("데이터베이스 관리자 초기화 완료")
        
        print(f"\n데이터베이스 업데이트 시작")
        print(f"총 법령 수: {len(laws):,}개")
        print(f"배치 크기: {batch_size}개")
        print(f"기존 데이터 삭제: {'예' if clear_existing else '아니오'}")
        print("=" * 50)
        
        # 기존 데이터 삭제 (선택사항)
        if clear_existing:
            logger.info("기존 현행법령 데이터 삭제 중...")
            print("🗑️ 기존 데이터 삭제 중...")
            
            # 기존 데이터 개수 확인
            existing_count = db_manager.get_current_laws_count()
            logger.info(f"기존 현행법령 수: {existing_count:,}개")
            print(f"  기존 현행법령 수: {existing_count:,}개")
            
            if existing_count > 0:
                # 모든 현행법령 삭제
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM current_laws")
                    conn.commit()
                
                logger.info(f"기존 현행법령 {existing_count:,}개 삭제 완료")
                print(f"  ✅ {existing_count:,}개 삭제 완료")
        
        # 배치별로 데이터베이스에 삽입
        db_start_time = datetime.now()
        batch_count = 0
        total_inserted = 0
        
        for i in range(0, len(laws), batch_size):
            batch = laws[i:i + batch_size]
            batch_start_time = datetime.now()
            
            try:
                inserted_count = db_manager.insert_current_laws_batch(batch)
                batch_end_time = datetime.now()
                batch_duration = (batch_end_time - batch_start_time).total_seconds()
                
                batch_count += 1
                total_inserted += inserted_count
                
                logger.info(f"데이터베이스 배치 {batch_count} 삽입: {inserted_count}개 ({batch_duration:.2f}초)")
                print(f"  배치 {batch_count} 삽입: {inserted_count}개 ({batch_duration:.2f}초)")
                
            except Exception as e:
                error_msg = f"배치 {batch_count + 1} 삽입 실패: {e}"
                logger.error(error_msg)
                print(f"  ❌ {error_msg}")
                result["errors"].append(error_msg)
        
        db_end_time = datetime.now()
        db_duration = (db_end_time - db_start_time).total_seconds()
        
        result["total_processed"] = len(laws)
        result["total_inserted"] = total_inserted
        result["batch_count"] = batch_count
        
        logger.info(f"데이터베이스 업데이트 완료: 총 {total_inserted:,}개 삽입 ({db_duration:.2f}초)")
        print(f"✅ 데이터베이스 업데이트 완료: 총 {total_inserted:,}개 삽입 ({db_duration:.2f}초)")
        
        # 데이터베이스 통계 출력
        try:
            logger.info("데이터베이스 통계 조회 중...")
            db_stats = db_manager.get_current_laws_stats()
            print(f"\n📊 데이터베이스 통계:")
            print(f"  총 현행법령: {db_stats['total_count']:,}개")
            print(f"  소관부처별 분포: {len(db_stats['by_ministry'])}개 부처")
            print(f"  법령종류별 분포: {len(db_stats['by_type'])}개 종류")
            print(f"  연도별 분포: {len(db_stats['by_year'])}개 연도")
            
            # 상위 5개 소관부처 출력
            if db_stats['by_ministry']:
                print(f"\n  상위 소관부처:")
                for i, ministry in enumerate(db_stats['by_ministry'][:5], 1):
                    print(f"    {i}. {ministry['ministry_name']}: {ministry['count']:,}개")
            
            logger.info(f"데이터베이스 통계: 총 {db_stats['total_count']:,}개, 부처 {len(db_stats['by_ministry'])}개, 종류 {len(db_stats['by_type'])}개")
            
        except Exception as e:
            logger.warning(f"데이터베이스 통계 조회 실패: {e}")
        
        # 최종 결과 로그
        result["end_time"] = datetime.now().isoformat()
        total_duration = (datetime.now() - datetime.fromisoformat(result["start_time"])).total_seconds()
        result["total_duration"] = total_duration
        
        logger.info("=" * 60)
        logger.info("데이터베이스 업데이트 완료")
        logger.info(f"총 처리: {result['total_processed']:,}개")
        logger.info(f"총 삽입: {result['total_inserted']:,}개")
        logger.info(f"배치 수: {result['batch_count']:,}개")
        logger.info(f"총 소요 시간: {total_duration:.2f}초")
        if result['errors']:
            logger.warning(f"오류 발생: {len(result['errors'])}개")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"데이터베이스 업데이트 과정에서 오류 발생: {e}"
        print(f"❌ {error_msg}")
        result["status"] = "failed"
        result["errors"].append(error_msg)
        logger.error(error_msg)
    
    finally:
        result["end_time"] = datetime.now().isoformat()
    
    return result


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="현행법령 데이터베이스 업데이트 스크립트")
    
    # 입력 옵션
    parser.add_argument("--batch-dir", type=str, 
                       default="data/raw/law_open_api/current_laws/batches",
                       help="배치 파일 디렉토리 (기본값: data/raw/law_open_api/current_laws/batches)")
    parser.add_argument("--pattern", type=str, default="current_law_batch_*.json",
                       help="배치 파일 패턴 (기본값: current_law_batch_*.json)")
    parser.add_argument("--summary-file", type=str, default=None,
                       help="요약 파일 경로 (선택사항)")
    
    # 처리 옵션
    parser.add_argument("--batch-size", type=int, default=100,
                       help="데이터베이스 배치 크기 (기본값: 100)")
    parser.add_argument("--clear-existing", action="store_true",
                       help="기존 데이터 삭제 후 삽입")
    
    # 테스트 옵션
    parser.add_argument("--test", action="store_true",
                       help="데이터베이스 연결 테스트만 실행")
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 삽입 없이 테스트만 실행")
    
    args = parser.parse_args()
    
    print("현행법령 데이터베이스 업데이트 스크립트")
    print("=" * 50)
    
    # 데이터베이스 연결 테스트
    try:
        db_manager = DatabaseManager()
        logger.info("데이터베이스 연결 테스트 성공")
        print("✅ 데이터베이스 연결 테스트 성공")
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return 1
    
    # 테스트 모드
    if args.test:
        print("\n✅ 데이터베이스 연결 테스트 완료")
        return 0
    
    # 배치 파일 로드
    print(f"\n📁 배치 파일 로드 중: {args.batch_dir}")
    laws, loaded_files = load_batch_files(args.batch_dir, args.pattern)
    
    if not laws:
        print("❌ 로드할 현행법령 데이터가 없습니다.")
        return 1
    
    # 요약 파일 로드 (선택사항)
    summary_data = None
    if args.summary_file and Path(args.summary_file).exists():
        print(f"\n📄 요약 파일 로드 중: {args.summary_file}")
        summary_data = load_summary_file(args.summary_file)
        if summary_data:
            print("✅ 요약 파일 로드 완료")
    
    # Dry run 모드
    if args.dry_run:
        print(f"\n🔍 Dry run 모드 - 실제 삽입 없이 테스트")
        print(f"  처리할 법령 수: {len(laws):,}개")
        print(f"  배치 크기: {args.batch_size}개")
        print(f"  예상 배치 수: {(len(laws) + args.batch_size - 1) // args.batch_size}개")
        print(f"  기존 데이터 삭제: {'예' if args.clear_existing else '아니오'}")
        return 0
    
    # 데이터베이스 업데이트 실행
    try:
        result = update_database_with_laws(
            laws=laws,
            batch_size=args.batch_size,
            clear_existing=args.clear_existing
        )
        
        # 결과 저장
        result_file = f"results/current_laws_database_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("results").mkdir(exist_ok=True)
        
        # 추가 정보 포함
        result["loaded_files"] = loaded_files
        result["summary_data"] = summary_data
        result["args"] = vars(args)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
        # 최종 결과
        if result["status"] == "success":
            print(f"\n✅ 현행법령 데이터베이스 업데이트 완료!")
            print(f"   처리: {result['total_processed']:,}개")
            print(f"   삽입: {result['total_inserted']:,}개")
            print(f"   배치: {result['batch_count']:,}개")
            print(f"   소요 시간: {result['total_duration']:.2f}초")
            return 0
        else:
            print(f"\n❌ 현행법령 데이터베이스 업데이트 실패")
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
