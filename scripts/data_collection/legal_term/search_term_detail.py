#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법령?�어 ?�세조회 ?�크립트
가?�드 API�??�용?�여 ?�정 법령?�어???�세 ?�보�?조회?�니??
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# ?�로?�트 루트 ?�렉?�리�?Python 경로??추�?
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.legal_term_collection_api import LegalTermCollectionAPI
from source.utils.config import Config

# 로깅 ?�정
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
    """법령?�어 ?�세조회"""
    try:
        logger.info(f"법령?�어 ?�세조회 ?�작: {term_name}")
        
        # API ?�라?�언??초기??
        config = Config()
        api_client = LegalTermCollectionAPI(config)
        
        # ?�세조회 ?�행
        detail_info = api_client.get_term_detail(term_name)
        
        if detail_info:
            logger.info("=" * 60)
            logger.info(f"법령?�어 ?�세조회 결과: {term_name}")
            logger.info("=" * 60)
            
            # ?�세 ?�보 출력
            for key, value in detail_info.items():
                if isinstance(value, list):
                    logger.info(f"{key}: {', '.join(map(str, value))}")
                else:
                    logger.info(f"{key}: {value}")
            
            logger.info("=" * 60)
            return True
        else:
            logger.warning(f"법령?�어 ?�세조회 결과 ?�음: {term_name}")
            return False
            
    except Exception as e:
        logger.error(f"법령?�어 ?�세조회 �??�류 발생: {e}")
        return False

def main():
    """메인 ?�수"""
    parser = argparse.ArgumentParser(description='법령?�어 ?�세조회 ?�크립트 (가?�드 API ?�용)')
    parser.add_argument('term_name', help='?�세조회?�고???�는 법령?�어�?)
    parser.add_argument('--verbose', '-v', action='store_true', help='?�세 로그 출력')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 로그 ?�렉?�리 ?�성
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("법령?�어 ?�세조회 ?�크립트 ?�작 (가?�드 API ?�용)")
    logger.info("=" * 60)
    logger.info(f"조회 ?�?? {args.term_name}")
    
    try:
        start_time = datetime.now()
        
        # ?�세조회 ?�행
        success = search_term_detail(args.term_name)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        if success:
            logger.info("법령?�어 ?�세조회 ?�료")
        else:
            logger.error("법령?�어 ?�세조회 ?�패")
        logger.info(f"�??�요 ?�간: {duration.total_seconds():.2f}�?)
        logger.info("=" * 60)
        
        return success
        
    except Exception as e:
        logger.error(f"?�크립트 ?�행 ?�패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
