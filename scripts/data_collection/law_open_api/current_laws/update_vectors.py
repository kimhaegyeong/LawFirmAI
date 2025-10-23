#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 벡터 저장소 업데이트 스크립트

수집된 현행법령 배치 파일을 읽어서 벡터 저장소에 추가합니다.
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

from source.data.vector_store import LegalVectorStore

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성
    log_filename = f'logs/current_laws_vector_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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


def update_vector_store_with_laws(
    laws: List[Dict[str, Any]], 
    batch_size: int = 50,
    model_name: str = "jhgan/ko-sroberta-multitask",
    clear_existing: bool = False
) -> Dict[str, Any]:
    """
    현행법령 데이터를 벡터 저장소에 추가
    
    Args:
        laws: 현행법령 목록
        batch_size: 배치 크기
        model_name: 임베딩 모델명
        clear_existing: 기존 현행법령 벡터 삭제 여부
        
    Returns:
        Dict: 업데이트 결과
    """
    logger.info("=" * 60)
    logger.info("벡터 저장소 업데이트 시작")
    logger.info(f"총 법령 수: {len(laws):,}개")
    logger.info(f"배치 크기: {batch_size}개")
    logger.info(f"모델명: {model_name}")
    logger.info(f"기존 데이터 삭제: {'예' if clear_existing else '아니오'}")
    logger.info("=" * 60)
    
    result = {
        "status": "success",
        "total_processed": 0,
        "successful_batches": 0,
        "failed_batches": 0,
        "batch_count": 0,
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    try:
        # 벡터 저장소 초기화
        vector_store = LegalVectorStore(model_name=model_name)
        logger.info("벡터 저장소 초기화 완료")
        
        print(f"\n벡터 저장소 업데이트 시작")
        print(f"총 법령 수: {len(laws):,}개")
        print(f"배치 크기: {batch_size}개")
        print(f"모델명: {model_name}")
        print(f"기존 데이터 삭제: {'예' if clear_existing else '아니오'}")
        print("=" * 50)
        
        # 기존 현행법령 벡터 삭제 (선택사항)
        if clear_existing:
            logger.info("기존 현행법령 벡터 삭제 중...")
            print("🗑️ 기존 현행법령 벡터 삭제 중...")
            
            # 기존 현행법령 통계 확인
            try:
                existing_stats = vector_store.get_current_laws_stats()
                existing_count = existing_stats.get('total_current_laws', 0)
                logger.info(f"기존 현행법령 벡터 수: {existing_count:,}개")
                print(f"  기존 현행법령 벡터 수: {existing_count:,}개")
                
                if existing_count > 0:
                    # 기존 현행법령 벡터들을 찾아서 삭제
                    removed_count = 0
                    for metadata in vector_store.document_metadata[:]:
                        if metadata.get('document_type') == 'current_law':
                            law_id = metadata.get('law_id')
                            if law_id:
                                if vector_store.remove_current_law(law_id):
                                    removed_count += 1
                    
                    logger.info(f"기존 현행법령 벡터 {removed_count:,}개 삭제 완료")
                    print(f"  ✅ {removed_count:,}개 삭제 완료")
                    
            except Exception as e:
                logger.warning(f"기존 벡터 삭제 중 오류 발생: {e}")
                print(f"  ⚠️ 삭제 중 오류: {e}")
        
        # 배치별로 벡터 저장소에 추가
        vector_start_time = datetime.now()
        batch_count = 0
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(laws), batch_size):
            batch = laws[i:i + batch_size]
            batch_start_time = datetime.now()
            
            try:
                success = vector_store.add_current_laws(batch)
                batch_end_time = datetime.now()
                batch_duration = (batch_end_time - batch_start_time).total_seconds()
                
                batch_count += 1
                if success:
                    successful_batches += 1
                    logger.info(f"벡터 저장소 배치 {batch_count} 처리 성공: {len(batch)}개 ({batch_duration:.2f}초)")
                    print(f"  배치 {batch_count} 벡터화: ✅ 성공 ({batch_duration:.2f}초)")
                else:
                    failed_batches += 1
                    logger.warning(f"벡터 저장소 배치 {batch_count} 처리 실패: {len(batch)}개 ({batch_duration:.2f}초)")
                    print(f"  배치 {batch_count} 벡터화: ❌ 실패 ({batch_duration:.2f}초)")
                
            except Exception as e:
                failed_batches += 1
                error_msg = f"배치 {batch_count + 1} 벡터화 실패: {e}"
                logger.error(error_msg)
                print(f"  ❌ {error_msg}")
                result["errors"].append(error_msg)
        
        vector_end_time = datetime.now()
        vector_duration = (vector_end_time - vector_start_time).total_seconds()
        
        result["total_processed"] = len(laws)
        result["successful_batches"] = successful_batches
        result["failed_batches"] = failed_batches
        result["batch_count"] = batch_count
        
        logger.info(f"벡터 저장소 업데이트 완료: {successful_batches}/{batch_count} 배치 성공 ({vector_duration:.2f}초)")
        print(f"✅ 벡터 저장소 업데이트 완료: {successful_batches}/{batch_count} 배치 성공 ({vector_duration:.2f}초)")
        
        # 벡터 저장소 통계 출력
        try:
            logger.info("벡터 저장소 통계 조회 중...")
            vector_stats = vector_store.get_current_laws_stats()
            print(f"\n📊 벡터 저장소 통계:")
            print(f"  현행법령 벡터: {vector_stats['total_current_laws']:,}개")
            print(f"  전체 문서 비율: {vector_stats['current_law_ratio']:.2%}")
            print(f"  소관부처별 분포: {len(vector_stats['by_ministry'])}개 부처")
            print(f"  법령종류별 분포: {len(vector_stats['by_type'])}개 종류")
            print(f"  연도별 분포: {len(vector_stats['by_year'])}개 연도")
            
            # 상위 5개 소관부처 출력
            if vector_stats['by_ministry']:
                print(f"\n  상위 소관부처:")
                sorted_ministries = sorted(vector_stats['by_ministry'].items(), key=lambda x: x[1], reverse=True)
                for i, (ministry, count) in enumerate(sorted_ministries[:5], 1):
                    print(f"    {i}. {ministry}: {count:,}개")
            
            logger.info(f"벡터 저장소 통계: 현행법령 {vector_stats['total_current_laws']:,}개, 비율 {vector_stats['current_law_ratio']:.2%}")
            
        except Exception as e:
            logger.warning(f"벡터 저장소 통계 조회 실패: {e}")
        
        # 최종 결과 로그
        result["end_time"] = datetime.now().isoformat()
        total_duration = (datetime.now() - datetime.fromisoformat(result["start_time"])).total_seconds()
        result["total_duration"] = total_duration
        
        logger.info("=" * 60)
        logger.info("벡터 저장소 업데이트 완료")
        logger.info(f"총 처리: {result['total_processed']:,}개")
        logger.info(f"성공 배치: {result['successful_batches']:,}개")
        logger.info(f"실패 배치: {result['failed_batches']:,}개")
        logger.info(f"총 소요 시간: {total_duration:.2f}초")
        if result['errors']:
            logger.warning(f"오류 발생: {len(result['errors'])}개")
        logger.info("=" * 60)
        
    except Exception as e:
        error_msg = f"벡터 저장소 업데이트 과정에서 오류 발생: {e}"
        print(f"❌ {error_msg}")
        result["status"] = "failed"
        result["errors"].append(error_msg)
        logger.error(error_msg)
    
    finally:
        result["end_time"] = datetime.now().isoformat()
    
    return result


def test_vector_search(vector_store: LegalVectorStore, test_queries: List[str] = None) -> None:
    """
    벡터 검색 테스트
    
    Args:
        vector_store: 벡터 저장소 인스턴스
        test_queries: 테스트 쿼리 목록
    """
    if test_queries is None:
        test_queries = [
            "자동차 관리",
            "교육 관련 법령",
            "환경 보호",
            "건강보험",
            "노동 관련"
        ]
    
    print(f"\n🔍 벡터 검색 테스트:")
    for query in test_queries:
        try:
            results = vector_store.search_current_laws(query, top_k=3)
            print(f"  '{query}': {len(results)}개 결과")
            for i, result in enumerate(results[:2], 1):
                law_name = result.get('law_name', 'Unknown')
                ministry = result.get('ministry_name', 'Unknown')
                score = result.get('similarity_score', 0)
                print(f"    {i}. {law_name} ({ministry}) - 점수: {score:.3f}")
        except Exception as e:
            print(f"  '{query}': 검색 실패 - {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="현행법령 벡터 저장소 업데이트 스크립트")
    
    # 입력 옵션
    parser.add_argument("--batch-dir", type=str, 
                       default="data/raw/law_open_api/current_laws/batches",
                       help="배치 파일 디렉토리 (기본값: data/raw/law_open_api/current_laws/batches)")
    parser.add_argument("--pattern", type=str, default="current_law_batch_*.json",
                       help="배치 파일 패턴 (기본값: current_law_batch_*.json)")
    
    # 처리 옵션
    parser.add_argument("--batch-size", type=int, default=50,
                       help="벡터화 배치 크기 (기본값: 50)")
    parser.add_argument("--model-name", type=str, default="jhgan/ko-sroberta-multitask",
                       help="임베딩 모델명 (기본값: jhgan/ko-sroberta-multitask)")
    parser.add_argument("--clear-existing", action="store_true",
                       help="기존 현행법령 벡터 삭제 후 추가")
    
    # 테스트 옵션
    parser.add_argument("--test", action="store_true",
                       help="벡터 저장소 연결 테스트만 실행")
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 벡터화 없이 테스트만 실행")
    parser.add_argument("--search-test", action="store_true",
                       help="벡터 검색 테스트 실행")
    
    args = parser.parse_args()
    
    print("현행법령 벡터 저장소 업데이트 스크립트")
    print("=" * 50)
    
    # 벡터 저장소 연결 테스트
    try:
        vector_store = LegalVectorStore(model_name=args.model_name)
        logger.info("벡터 저장소 연결 테스트 성공")
        print("✅ 벡터 저장소 연결 테스트 성공")
    except Exception as e:
        print(f"❌ 벡터 저장소 연결 실패: {e}")
        return 1
    
    # 테스트 모드
    if args.test:
        print("\n✅ 벡터 저장소 연결 테스트 완료")
        return 0
    
    # 검색 테스트 모드
    if args.search_test:
        print(f"\n🔍 벡터 검색 테스트 실행")
        test_vector_search(vector_store)
        return 0
    
    # 배치 파일 로드
    print(f"\n📁 배치 파일 로드 중: {args.batch_dir}")
    laws, loaded_files = load_batch_files(args.batch_dir, args.pattern)
    
    if not laws:
        print("❌ 로드할 현행법령 데이터가 없습니다.")
        return 1
    
    # Dry run 모드
    if args.dry_run:
        print(f"\n🔍 Dry run 모드 - 실제 벡터화 없이 테스트")
        print(f"  처리할 법령 수: {len(laws):,}개")
        print(f"  배치 크기: {args.batch_size}개")
        print(f"  예상 배치 수: {(len(laws) + args.batch_size - 1) // args.batch_size}개")
        print(f"  모델명: {args.model_name}")
        print(f"  기존 데이터 삭제: {'예' if args.clear_existing else '아니오'}")
        return 0
    
    # 벡터 저장소 업데이트 실행
    try:
        result = update_vector_store_with_laws(
            laws=laws,
            batch_size=args.batch_size,
            model_name=args.model_name,
            clear_existing=args.clear_existing
        )
        
        # 결과 저장
        result_file = f"results/current_laws_vector_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("results").mkdir(exist_ok=True)
        
        # 추가 정보 포함
        result["loaded_files"] = loaded_files
        result["args"] = vars(args)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
        # 벡터 검색 테스트 (성공한 경우)
        if result["status"] == "success" and result["successful_batches"] > 0:
            print(f"\n🔍 벡터 검색 테스트:")
            test_vector_search(vector_store)
        
        # 최종 결과
        if result["status"] == "success":
            print(f"\n✅ 현행법령 벡터 저장소 업데이트 완료!")
            print(f"   처리: {result['total_processed']:,}개")
            print(f"   성공 배치: {result['successful_batches']:,}개")
            print(f"   실패 배치: {result['failed_batches']:,}개")
            print(f"   소요 시간: {result['total_duration']:.2f}초")
            return 0
        else:
            print(f"\n❌ 현행법령 벡터 저장소 업데이트 실패")
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
