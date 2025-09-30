#!/usr/bin/env python3
"""
법령용어 상세조회 스크립트
가이드 API를 사용하여 특정 법령용어의 상세 정보를 조회합니다.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.legal_term_collection_api import LegalTermCollectionAPI
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/term_detail_search.log')
    ]
)
logger = logging.getLogger(__name__)

def search_term_detail(term_name: str) -> bool:
    """법령용어 상세조회"""
    try:
        logger.info(f"법령용어 상세조회 시작: {term_name}")
        
        # API 클라이언트 초기화
        config = Config()
        api_client = LegalTermCollectionAPI(config)
        
        # 상세조회 실행
        detail_info = api_client.get_term_detail(term_name)
        
        if detail_info:
            logger.info("=" * 60)
            logger.info(f"법령용어 상세조회 결과: {term_name}")
            logger.info("=" * 60)
            
            # 상세 정보 출력
            for key, value in detail_info.items():
                if isinstance(value, list):
                    logger.info(f"{key}: {', '.join(map(str, value))}")
                else:
                    logger.info(f"{key}: {value}")
            
            logger.info("=" * 60)
            return True
        else:
            logger.warning(f"법령용어 상세조회 결과 없음: {term_name}")
            return False
            
    except Exception as e:
        logger.error(f"법령용어 상세조회 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='법령용어 상세조회 스크립트 (가이드 API 사용)')
    parser.add_argument('term_name', help='상세조회하고자 하는 법령용어명')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("법령용어 상세조회 스크립트 시작 (가이드 API 사용)")
    logger.info("=" * 60)
    logger.info(f"조회 대상: {args.term_name}")
    
    try:
        start_time = datetime.now()
        
        # 상세조회 실행
        success = search_term_detail(args.term_name)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        if success:
            logger.info("법령용어 상세조회 완료")
        else:
            logger.error("법령용어 상세조회 실패")
        logger.info(f"총 소요 시간: {duration.total_seconds():.2f}초")
        logger.info("=" * 60)
        
        return success
        
    except Exception as e:
        logger.error(f"스크립트 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
