#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현행법령 통합 실행 스크립트

현행법령 데이터 수집, 데이터베이스 업데이트, 벡터 저장소 업데이트를
순차적으로 또는 선택적으로 실행할 수 있는 통합 스크립트입니다.
"""

import os
import sys
import argparse
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 생성
    log_filename = f'logs/current_laws_integration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
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


def run_command(command: List[str], description: str) -> Dict[str, Any]:
    """
    명령어 실행
    
    Args:
        command: 실행할 명령어 리스트
        description: 명령어 설명
        
    Returns:
        Dict: 실행 결과
    """
    logger.info(f"명령어 실행 시작: {description}")
    print(f"\n🔄 {description}")
    print(f"   명령어: {' '.join(command)}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=str(project_root)
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            logger.info(f"명령어 실행 성공: {description} ({duration:.2f}초)")
            print(f"   ✅ 성공 ({duration:.2f}초)")
            return {
                "status": "success",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }
        else:
            logger.error(f"명령어 실행 실패: {description} (코드: {result.returncode})")
            print(f"   ❌ 실패 (코드: {result.returncode})")
            if result.stderr:
                print(f"   오류: {result.stderr}")
            return {
                "status": "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration
            }
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_msg = f"명령어 실행 중 예외 발생: {e}"
        logger.error(error_msg)
        print(f"   ❌ 예외 발생: {e}")
        
        return {
            "status": "error",
            "returncode": -1,
            "stdout": "",
            "stderr": error_msg,
            "duration": duration
        }


def run_collection_step(args) -> Dict[str, Any]:
    """데이터 수집 단계 실행"""
    command = [
        "python", "scripts/data_collection/law_open_api/current_laws/collect_current_laws.py"
    ]
    
    # 기본 옵션 추가
    if args.query:
        command.extend(["--query", args.query])
    if args.max_pages:
        command.extend(["--max-pages", str(args.max_pages)])
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.sort_order:
        command.extend(["--sort-order", args.sort_order])
    if args.no_details:
        command.append("--no-details")
    if args.resume_checkpoint:
        command.append("--resume-checkpoint")
    if args.sample:
        command.extend(["--sample", str(args.sample)])
    
    return run_command(command, "현행법령 데이터 수집")


def run_database_step(args) -> Dict[str, Any]:
    """데이터베이스 업데이트 단계 실행"""
    command = [
        "python", "scripts/data_collection/law_open_api/current_laws/update_database.py"
    ]
    
    # 기본 옵션 추가
    if args.batch_dir:
        command.extend(["--batch-dir", args.batch_dir])
    if args.pattern:
        command.extend(["--pattern", args.pattern])
    if args.db_batch_size:
        command.extend(["--batch-size", str(args.db_batch_size)])
    if args.clear_existing:
        command.append("--clear-existing")
    if args.summary_file:
        command.extend(["--summary-file", args.summary_file])
    
    return run_command(command, "데이터베이스 업데이트")


def run_vector_step(args) -> Dict[str, Any]:
    """벡터 저장소 업데이트 단계 실행"""
    command = [
        "python", "scripts/data_collection/law_open_api/current_laws/update_vectors.py"
    ]
    
    # 기본 옵션 추가
    if args.batch_dir:
        command.extend(["--batch-dir", args.batch_dir])
    if args.pattern:
        command.extend(["--pattern", args.pattern])
    if args.vector_batch_size:
        command.extend(["--batch-size", str(args.vector_batch_size)])
    if args.model_name:
        command.extend(["--model-name", args.model_name])
    if args.clear_existing:
        command.append("--clear-existing")
    
    return run_command(command, "벡터 저장소 업데이트")


def run_integration_pipeline(args) -> Dict[str, Any]:
    """
    통합 파이프라인 실행
    
    Args:
        args: 명령행 인수
        
    Returns:
        Dict: 전체 실행 결과
    """
    logger.info("=" * 60)
    logger.info("현행법령 통합 파이프라인 시작")
    logger.info(f"수집: {'예' if args.collect else '아니오'}")
    logger.info(f"데이터베이스: {'예' if args.database else '아니오'}")
    logger.info(f"벡터 저장소: {'예' if args.vectors else '아니오'}")
    logger.info("=" * 60)
    
    result = {
        "status": "success",
        "steps": {},
        "total_duration": 0,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "errors": []
    }
    
    pipeline_start_time = datetime.now()
    
    try:
        # 1단계: 데이터 수집
        if args.collect:
            print(f"\n{'='*60}")
            print(f"1단계: 현행법령 데이터 수집")
            print(f"{'='*60}")
            
            collection_result = run_collection_step(args)
            result["steps"]["collection"] = collection_result
            
            if collection_result["status"] != "success":
                error_msg = f"데이터 수집 실패: {collection_result.get('stderr', 'Unknown error')}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                
                if args.stop_on_error:
                    result["status"] = "failed"
                    return result
        
        # 2단계: 데이터베이스 업데이트
        if args.database:
            print(f"\n{'='*60}")
            print(f"2단계: 데이터베이스 업데이트")
            print(f"{'='*60}")
            
            database_result = run_database_step(args)
            result["steps"]["database"] = database_result
            
            if database_result["status"] != "success":
                error_msg = f"데이터베이스 업데이트 실패: {database_result.get('stderr', 'Unknown error')}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                
                if args.stop_on_error:
                    result["status"] = "failed"
                    return result
        
        # 3단계: 벡터 저장소 업데이트
        if args.vectors:
            print(f"\n{'='*60}")
            print(f"3단계: 벡터 저장소 업데이트")
            print(f"{'='*60}")
            
            vector_result = run_vector_step(args)
            result["steps"]["vectors"] = vector_result
            
            if vector_result["status"] != "success":
                error_msg = f"벡터 저장소 업데이트 실패: {vector_result.get('stderr', 'Unknown error')}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                
                if args.stop_on_error:
                    result["status"] = "failed"
                    return result
        
        # 전체 실행 시간 계산
        pipeline_end_time = datetime.now()
        total_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
        result["total_duration"] = total_duration
        result["end_time"] = pipeline_end_time.isoformat()
        
        # 성공 단계 요약
        successful_steps = [step for step, result_data in result["steps"].items() 
                           if result_data["status"] == "success"]
        
        logger.info("=" * 60)
        logger.info("현행법령 통합 파이프라인 완료")
        logger.info(f"성공 단계: {', '.join(successful_steps) if successful_steps else '없음'}")
        logger.info(f"총 소요 시간: {total_duration:.2f}초")
        if result["errors"]:
            logger.warning(f"오류 발생: {len(result['errors'])}개")
        logger.info("=" * 60)
        
        print(f"\n{'='*60}")
        print(f"통합 파이프라인 완료")
        print(f"{'='*60}")
        print(f"성공 단계: {', '.join(successful_steps) if successful_steps else '없음'}")
        print(f"총 소요 시간: {total_duration:.2f}초")
        
        if result["errors"]:
            print(f"오류 발생: {len(result['errors'])}개")
            for error in result["errors"]:
                print(f"  - {error}")
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의한 중단")
        result["status"] = "interrupted"
        result["end_time"] = datetime.now().isoformat()
        return result
    except Exception as e:
        error_msg = f"파이프라인 실행 중 예외 발생: {e}"
        logger.error(error_msg)
        result["status"] = "error"
        result["errors"].append(error_msg)
        result["end_time"] = datetime.now().isoformat()
        return result


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


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="현행법령 통합 실행 스크립트")
    
    # 실행 단계 선택
    parser.add_argument("--collect", action="store_true",
                       help="데이터 수집 단계 실행")
    parser.add_argument("--database", action="store_true",
                       help="데이터베이스 업데이트 단계 실행")
    parser.add_argument("--vectors", action="store_true",
                       help="벡터 저장소 업데이트 단계 실행")
    parser.add_argument("--all", action="store_true",
                       help="모든 단계 실행 (수집 → 데이터베이스 → 벡터)")
    
    # 수집 옵션
    parser.add_argument("--query", type=str, default="",
                       help="검색 질의 (기본값: 빈 문자열 - 모든 법령)")
    parser.add_argument("--max-pages", type=int, default=None,
                       help="최대 페이지 수 (기본값: 무제한)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="수집 배치 크기 (기본값: 10)")
    parser.add_argument("--sort-order", type=str, default="ldes",
                       choices=["ldes", "lasc", "dasc", "ddes", "nasc", "ndes", "efasc", "efdes"],
                       help="정렬 순서 (기본값: ldes)")
    parser.add_argument("--no-details", action="store_true",
                       help="상세 정보 제외")
    parser.add_argument("--resume-checkpoint", action="store_true",
                       help="체크포인트에서 재시작")
    parser.add_argument("--sample", type=int, default=0,
                       help="샘플 수집 (지정된 개수만 수집)")
    
    # 데이터베이스 옵션
    parser.add_argument("--batch-dir", type=str,
                       default="data/raw/law_open_api/current_laws/batches",
                       help="배치 파일 디렉토리")
    parser.add_argument("--pattern", type=str, default="current_law_batch_*.json",
                       help="배치 파일 패턴")
    parser.add_argument("--db-batch-size", type=int, default=100,
                       help="데이터베이스 배치 크기 (기본값: 100)")
    parser.add_argument("--summary-file", type=str, default=None,
                       help="요약 파일 경로 (선택사항)")
    
    # 벡터 저장소 옵션
    parser.add_argument("--vector-batch-size", type=int, default=50,
                       help="벡터화 배치 크기 (기본값: 50)")
    parser.add_argument("--model-name", type=str, default="jhgan/ko-sroberta-multitask",
                       help="임베딩 모델명 (기본값: jhgan/ko-sroberta-multitask)")
    
    # 공통 옵션
    parser.add_argument("--clear-existing", action="store_true",
                       help="기존 데이터 삭제 후 처리")
    parser.add_argument("--stop-on-error", action="store_true",
                       help="오류 발생 시 중단")
    
    # 테스트 옵션
    parser.add_argument("--test", action="store_true",
                       help="연결 테스트만 실행")
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 실행 없이 계획만 출력")
    
    args = parser.parse_args()
    
    print("현행법령 통합 실행 스크립트")
    print("=" * 50)
    
    # 환경 검증
    if not validate_environment():
        return 1
    
    # 실행 단계 결정
    if args.all:
        args.collect = True
        args.database = True
        args.vectors = True
    
    if not any([args.collect, args.database, args.vectors]):
        print("❌ 실행할 단계를 선택해주세요.")
        print("   --collect: 데이터 수집")
        print("   --database: 데이터베이스 업데이트")
        print("   --vectors: 벡터 저장소 업데이트")
        print("   --all: 모든 단계 실행")
        return 1
    
    # 테스트 모드
    if args.test:
        print("\n🔍 연결 테스트 실행")
        
        # API 연결 테스트
        try:
            from source.data.law_open_api_client import LawOpenAPIClient
            client = LawOpenAPIClient()
            if client.test_connection():
                print("✅ API 연결 테스트 성공")
            else:
                print("❌ API 연결 테스트 실패")
                return 1
        except Exception as e:
            print(f"❌ API 연결 테스트 실패: {e}")
            return 1
        
        # 데이터베이스 연결 테스트
        try:
            from source.data.database import DatabaseManager
            db_manager = DatabaseManager()
            print("✅ 데이터베이스 연결 테스트 성공")
        except Exception as e:
            print(f"❌ 데이터베이스 연결 테스트 실패: {e}")
            return 1
        
        # 벡터 저장소 연결 테스트
        try:
            from source.data.vector_store import LegalVectorStore
            vector_store = LegalVectorStore()
            print("✅ 벡터 저장소 연결 테스트 성공")
        except Exception as e:
            print(f"❌ 벡터 저장소 연결 테스트 실패: {e}")
            return 1
        
        print("\n✅ 모든 연결 테스트 완료")
        return 0
    
    # Dry run 모드
    if args.dry_run:
        print(f"\n🔍 Dry run 모드 - 실행 계획")
        print(f"실행할 단계:")
        if args.collect:
            print(f"  ✅ 데이터 수집")
            print(f"    - 검색어: '{args.query}'")
            print(f"    - 최대 페이지: {args.max_pages or '무제한'}")
            print(f"    - 배치 크기: {args.batch_size}개")
            print(f"    - 상세 정보: {'제외' if args.no_details else '포함'}")
        if args.database:
            print(f"  ✅ 데이터베이스 업데이트")
            print(f"    - 배치 디렉토리: {args.batch_dir}")
            print(f"    - 패턴: {args.pattern}")
            print(f"    - 배치 크기: {args.db_batch_size}개")
        if args.vectors:
            print(f"  ✅ 벡터 저장소 업데이트")
            print(f"    - 배치 디렉토리: {args.batch_dir}")
            print(f"    - 패턴: {args.pattern}")
            print(f"    - 배치 크기: {args.vector_batch_size}개")
            print(f"    - 모델명: {args.model_name}")
        
        print(f"\n공통 옵션:")
        print(f"  - 기존 데이터 삭제: {'예' if args.clear_existing else '아니오'}")
        print(f"  - 오류 시 중단: {'예' if args.stop_on_error else '아니오'}")
        return 0
    
    # 통합 파이프라인 실행
    try:
        result = run_integration_pipeline(args)
        
        # 결과 저장
        result_file = f"results/current_laws_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path("results").mkdir(exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 결과 저장: {result_file}")
        
        # 최종 결과
        if result["status"] == "success":
            print(f"\n✅ 현행법령 통합 파이프라인 완료!")
            successful_steps = [step for step, result_data in result["steps"].items() 
                              if result_data["status"] == "success"]
            print(f"   성공 단계: {', '.join(successful_steps)}")
            print(f"   총 소요 시간: {result['total_duration']:.2f}초")
            return 0
        else:
            print(f"\n❌ 현행법령 통합 파이프라인 실패")
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
