#!/usr/bin/env python3
"""
법률 용어 수집 스크립트

국가법령정보센터 OpenAPI를 활용하여 법률 용어를 수집합니다.

사용법:
1. 환경변수 설정:
   export LAW_OPEN_API_OC="your_email@example.com"

2. 스크립트 실행:
   python scripts/legal_term/collect_legal_terms.py

기능:
- 법률 용어 수집 및 저장
- 체크포인트를 통한 중단/재개 지원
"""

import os
import sys
import logging
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 현재 디렉토리를 Python 경로에 추가 (상대 import를 위해)
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# python-dotenv를 사용하여 .env 파일 로드
try:
    from dotenv import load_dotenv
    # 프로젝트 루트에서 .env 파일 로드
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] 환경변수 로드 완료: {env_path}")
    else:
        print(f"[WARN] .env 파일을 찾을 수 없습니다: {env_path}")
        # 현재 디렉토리에서도 시도
        current_env = Path('.env')
        if current_env.exists():
            load_dotenv(current_env)
            print(f"[OK] 현재 디렉토리에서 환경변수 로드 완료: {current_env.absolute()}")
except ImportError:
    print("[ERROR] python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치하세요.")

from term_collector import LegalTermCollector

# 로깅 설정 (간략한 버전, UTF-8 인코딩)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collect_legal_terms(max_terms: int = None, collection_type: str = "all", use_mock_data: bool = False, 
                       resume: bool = True, memory_limit_mb: int = None, target_year: int = None) -> bool:
    """법률 용어 수집 및 사전 구축 (메모리 최적화 및 체크포인트 지원)"""
    if max_terms is None:
        logger.info(f"법률 용어 수집 시작 - 타입: {collection_type}, 모드: 무제한 (메모리 최적화)")
    else:
        logger.info(f"법률 용어 수집 시작 - 타입: {collection_type}, 최대: {max_terms}개 (메모리 최적화)")
    
    try:
        # 1. 메모리 설정 생성
        from term_collector import MemoryConfig
        memory_config = MemoryConfig()
        
        if memory_limit_mb:
            memory_config.max_memory_mb = memory_limit_mb
        
        # 2. 수집기 초기화 (메모리 최적화)
        collector = LegalTermCollector(memory_config=memory_config)
        
        # 3. 메모리 설정 최적화
        collector.optimize_memory_settings()
        
        # 4. 진행 상태 확인
        progress_status = collector.get_progress_status()
        if progress_status['status'] != 'not_started':
            logger.info(f"기존 수집 세션 발견 - 진행률: {progress_status['progress_percent']:.1f}%")
        
        # 5. 수집 타입에 따른 용어 수집
        if collection_type == "all":
            success = collector.collect_all_terms(max_terms, use_mock_data, resume)
        elif collection_type == "categories":
            categories = ["민사법", "형사법", "상사법", "노동법", "행정법", "환경법", "소비자법", "지적재산권법", "금융법"]
            success = collector.collect_terms_by_categories(categories, max_terms // len(categories), resume)
        elif collection_type == "keywords":
            keywords = ["계약", "손해배상", "불법행위", "채권", "채무", "소멸시효", "취소권", "해제권", "대리권", "대표권"]
            success = collector.collect_terms_by_keywords(keywords, max_terms // len(keywords), resume)
        elif collection_type == "year":
            if not target_year:
                logger.error("연도별 수집을 위해서는 --target-year 옵션이 필요합니다.")
                return False
            success = collector.collect_terms_by_year(target_year, max_terms, resume)
        else:
            logger.error(f"지원하지 않는 수집 타입: {collection_type}")
            return False
        
        if not success:
            logger.error("용어 수집 실패")
            return False
        
        
        # 7. 사전 저장
        collector.save_dictionary()
        
        
        
        
        logger.info("법률 용어 수집 완료")
        return True
        
    except Exception as e:
        logger.error(f"법률 용어 수집 실패: {e}")
        return False








def main():
    """메인 함수 (메모리 최적화 및 체크포인트 지원)"""
    # Graceful shutdown 설정
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        signal_name = signal.Signals(signum).name
        logger.info(f"메인 프로세스에서 시그널 {signal_name}({signum}) 수신 - graceful shutdown 시작")
        shutdown_requested = True
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='법률 용어 수집 스크립트 (메모리 최적화 및 체크포인트 지원)')
    parser.add_argument('--max-terms', type=int, default=None, help='최대 수집 용어 수 (기본값: 무제한, 0 또는 음수 입력 시 무제한 모드)')
    parser.add_argument('--collection-type', choices=['all', 'categories', 'keywords', 'year'], default='all', 
                       help='수집 타입 (all: 전체, categories: 카테고리별, keywords: 키워드별, year: 지정연도)')
    parser.add_argument('--use-mock-data', action='store_true', help='API 대신 모의 데이터 사용')
    parser.add_argument('--no-resume', action='store_true', help='체크포인트 무시하고 새로 시작')
    parser.add_argument('--memory-limit', type=int, help='메모리 사용량 제한 (MB)')
    parser.add_argument('--clear-checkpoint', action='store_true', help='기존 체크포인트 삭제')
    parser.add_argument('--clear-dictionary', action='store_true', help='기존 사전 데이터 완전 삭제 후 새로 시작')
    parser.add_argument('--status', action='store_true', help='현재 수집 상태 확인')
    parser.add_argument('--target-year', type=int, help='수집할 대상 연도 (예: 2024)')
    parser.add_argument('--search-detail', type=str, help='특정 법령용어의 상세조회 (예: --search-detail "계약")')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("법률 용어 수집 스크립트 시작 (메모리 최적화)")
    logger.info("=" * 60)
    
    # 환경 변수 확인
    if not os.getenv("LAW_OPEN_API_OC"):
        logger.error("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
        logger.error("다음과 같이 환경변수를 설정해주세요:")
        logger.error("export LAW_OPEN_API_OC='your_email@example.com'")
        return False
    
    logger.info(f"API 키 확인: {os.getenv('LAW_OPEN_API_OC')}")
    logger.info(f"수집 타입: {args.collection_type}")
    if args.max_terms is None or args.max_terms <= 0:
        logger.info("최대 용어 수: 무제한 모드")
    else:
        logger.info(f"최대 용어 수: {args.max_terms}")
    logger.info(f"체크포인트 재개: {'아니오' if args.no_resume else '예'}")
    if args.memory_limit:
        logger.info(f"메모리 제한: {args.memory_limit}MB")
    if args.target_year:
        logger.info(f"대상 연도: {args.target_year}년")
    
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    Path("data/raw/legal_terms").mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = datetime.now()
        
        # 체크포인트 삭제 요청
        if args.clear_checkpoint:
            logger.info("기존 체크포인트 삭제 중...")
            from term_collector import LegalTermCollector
            collector = LegalTermCollector()
            collector.clear_checkpoint()
            logger.info("체크포인트 삭제 완료")
        
        # 사전 데이터 완전 삭제 요청
        if args.clear_dictionary:
            logger.info("기존 사전 데이터 완전 삭제 중...")
            
            # 기본 파일들 삭제
            dictionary_path = Path("data/raw/legal_terms/legal_term_dictionary.json")
            checkpoint_path = Path("data/raw/legal_terms/checkpoint.json")
            
            if dictionary_path.exists():
                dictionary_path.unlink()
                logger.info("사전 파일 삭제 완료")
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("체크포인트 파일 삭제 완료")
            
            # 세션 폴더들 삭제
            import shutil
            session_folders = list(Path("data/raw/legal_terms").glob("session_*"))
            for session_folder in session_folders:
                if session_folder.is_dir():
                    shutil.rmtree(session_folder)
                    logger.info(f"세션 폴더 삭제 완료: {session_folder}")
            
            # 기타 파일들 삭제
            other_files = list(Path("data/raw/legal_terms").glob("legal_terms_*.json"))
            other_files.extend(list(Path("data/raw/legal_terms").glob("checkpoint_*.json")))
            
            for file_path in other_files:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"파일 삭제 완료: {file_path}")
            
            logger.info("모든 데이터 삭제 완료 - 새로 시작합니다")
            return True
        
        # 상태 확인 요청
        if args.status:
            logger.info("현재 수집 상태 확인 중...")
            from term_collector import LegalTermCollector, MemoryConfig
            memory_config = MemoryConfig()
            collector = LegalTermCollector(memory_config=memory_config)
            progress_status = collector.get_progress_status()
            memory_status = collector.get_memory_status()
            
            logger.info("=" * 40)
            logger.info("수집 상태 정보")
            logger.info("=" * 40)
            logger.info(f"상태: {progress_status['status']}")
            if progress_status['status'] != 'not_started':
                logger.info(f"진행률: {progress_status['progress_percent']:.1f}%")
                logger.info(f"수집된 용어: {progress_status['collected_count']}/{progress_status['total_target']}개")
                logger.info(f"세션 ID: {progress_status['session_id']}")
                logger.info(f"에러 발생: {progress_status['errors_count']}회")
                logger.info(f"재개 가능: {'예' if progress_status['can_resume'] else '아니오'}")
            
            logger.info("=" * 40)
            logger.info("메모리 상태 정보")
            logger.info("=" * 40)
            logger.info(f"현재 메모리: {memory_status['current_memory_mb']:.1f}MB")
            logger.info(f"최대 메모리: {memory_status['peak_memory_mb']:.1f}MB")
            logger.info(f"메모리 한계: {memory_status['memory_limit_mb']:.1f}MB")
            logger.info(f"사용률: {memory_status['memory_usage_percent']:.1f}%")
            logger.info(f"가비지 컬렉션: {memory_status['gc_runs']}회")
            logger.info(f"체크포인트 저장: {memory_status['checkpoints_saved']}회")
            logger.info("=" * 40)
            return True
        
        # 상세조회 요청
        if args.search_detail:
            logger.info(f"법령용어 상세조회 시작: {args.search_detail}")
            from source.data.legal_term_collection_api import LegalTermCollectionAPI
            from source.utils.config import Config
            
            config = Config()
            api_client = LegalTermCollectionAPI(config)
            
            detail_info = api_client.get_term_detail(args.search_detail)
            
            if detail_info:
                logger.info("=" * 60)
                logger.info(f"법령용어 상세조회 결과: {args.search_detail}")
                logger.info("=" * 60)
                
                for key, value in detail_info.items():
                    if isinstance(value, list):
                        logger.info(f"{key}: {', '.join(map(str, value))}")
                    else:
                        logger.info(f"{key}: {value}")
                
                logger.info("=" * 60)
                logger.info("법령용어 상세조회 완료")
                return True
            else:
                logger.warning(f"법령용어 상세조회 결과 없음: {args.search_detail}")
                return False
        
        # 법률 용어 수집
        logger.info("법률 용어 수집 시작")
        resume = not args.no_resume
        
        # shutdown 요청 확인
        if shutdown_requested:
            logger.info("종료 요청으로 인한 수집 중단")
            return False
        
        # 무제한 모드 처리
        max_terms = args.max_terms
        if max_terms is None or max_terms <= 0:
            max_terms = None  # 무제한 모드
            logger.info("무제한 모드로 수집을 시작합니다")
        
        if not collect_legal_terms(max_terms, args.collection_type, args.use_mock_data, 
                                 resume, args.memory_limit, args.target_year):
            logger.error("법률 용어 수집 실패")
            return False
        
        # 수집 완료 후 shutdown 요청 확인
        if shutdown_requested:
            logger.info("수집 완료 후 종료 요청 확인")
            return True
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("법률 용어 수집 완료")
        logger.info(f"총 소요 시간: {duration.total_seconds():.2f}초")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)