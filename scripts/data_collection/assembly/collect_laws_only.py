#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 법률정보시스템 법률만 수집 (특정 URL용)

제공된 URL의 법률만 수집하는 스크립트:
https://likms.assembly.go.kr/law/lawsLawtInqyList2020.do?genActiontypeCd=2ACT1010&genMenuId=menu_serv_nlaw_lawt_1020&uid=R310CV1620579091049F522&genDoctreattypeCd=DOCT2041&topicCd=PJJG00_000&pageSize=100&srchNm=&srchType=contNm&orderType=&orderObj=&srchDtType=promDt&srchStaDt=&srchEndDt=

총 1,895건의 법률 수집

사용법:
  python collect_laws_only.py --sample 10     # 샘플 10개
  python collect_laws_only.py --sample 100    # 샘플 100개
  python collect_laws_only.py --full          # 전체 1895개
  python collect_laws_only.py --resume        # 중단 지점에서 재개
"""

import argparse
import sys
import signal
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.assembly.assembly_collector import AssemblyCollector
from scripts.assembly.checkpoint_manager import CheckpointManager

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/law_collection_only.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown 처리
interrupted = False

def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    global interrupted
    logger.warning("\nInterrupt signal received. Saving progress...")
    interrupted = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class LawOnlyCollector:
    """법률만 수집하는 클래스"""
    
    def __init__(self, base_dir: str = "data/raw/assembly"):
        self.base_dir = Path(base_dir)
        self.collected_items = []
        self.failed_items = []
        self.start_time = None
        
        # 출력 디렉토리 생성
        self.output_dir = self.base_dir / "law_only" / datetime.now().strftime("%Y%m%d")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        # 특정 URL 설정 (법률만 필터링된 URL)
        self.base_url = "https://likms.assembly.go.kr/law/lawsLawtInqyList2020.do"
        self.url_params = {
            'genActiontypeCd': '2ACT1010',
            'genMenuId': 'menu_serv_nlaw_lawt_1020',
            'uid': 'R310CV1620579091049F522',
            'genDoctreattypeCd': 'DOCT2041',  # 법률만 필터링
            'topicCd': 'PJJG00_000',
            'pageSize': '100',
            'srchNm': '',
            'srchType': 'contNm',
            'orderType': '',
            'orderObj': '',
            'srchDtType': 'promDt',
            'srchStaDt': '',
            'srchEndDt': ''
        }
        
        logger.info(f"Target URL: {self.base_url}")
        logger.info(f"Expected total: 1,895 laws")
    
    def build_url(self, page_num: int = 1) -> str:
        """페이지 번호에 따른 URL 구성"""
        params = self.url_params.copy()
        params['pageNum'] = str(page_num)
        
        param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.base_url}?{param_str}"
    
    def collect_laws(self, target_count: int = None, page_size: int = 100, resume: bool = True, start_page: int = 1):
        """법률 수집 메인 함수"""
        
        print(f"\n{'='*60}")
        print(f"LAW ONLY COLLECTION STARTED")
        print(f"Target: Laws only (1,895 total)")
        print(f"{'='*60}")
        
        # 체크포인트 매니저
        checkpoint_mgr = CheckpointManager("data/checkpoints/laws_only")
        print(f"Checkpoint directory: data/checkpoints/laws_only")
        
        # 체크포인트 로드
        actual_start_page = start_page
        checkpoint = None
        
        if resume:
            print(f"Checking for existing checkpoint...")
            checkpoint = checkpoint_mgr.load_checkpoint()
            if checkpoint:
                print(f"Resuming from checkpoint")
                print(f"   Page: {checkpoint.get('current_page', 0)}")
                print(f"   Collected: {checkpoint.get('collected_count', 0)} items")
                actual_start_page = checkpoint['current_page'] + 1
            else:
                print(f"No checkpoint found, starting from page {start_page}")
        else:
            print(f"Resume disabled, starting from page {start_page}")
        
        # 수집기 초기화
        print(f"\nInitializing collector...")
        collector = AssemblyCollector(
            base_dir=str(self.base_dir),
            data_type="law_only",
            batch_size=20,
            memory_limit_mb=600
        )
        
        self.start_time = datetime.now()
        
        try:
            with AssemblyPlaywrightClient(rate_limit=2.0, timeout=30000, headless=True) as client:
                print(f"Browser started successfully")
                
                page_num = actual_start_page
                collected_count = checkpoint.get('collected_count', 0) if checkpoint else 0
                
                while True:
                    if interrupted:
                        print(f"\nCollection interrupted by user")
                        break
                    
                    if target_count and collected_count >= target_count:
                        print(f"\nTarget count reached: {collected_count}/{target_count}")
                        break
                    
                    print(f"\nProcessing page {page_num}...")
                    
                    try:
                        # 페이지 정보 설정 (page_number를 위해 필요)
                        collector.set_page_info(page_num)
                        
                        # 특정 URL로 페이지 이동
                        url = self.build_url(page_num)
                        print(f"URL: {url}")
                        
                        # 페이지 로드
                        client.page.goto(url, wait_until='domcontentloaded')
                        client.page.wait_for_timeout(3000)  # 3초 대기
                        
                        # 법률 목록 파싱
                        laws = client._parse_law_table()
                        
                        if not laws:
                            print(f"No laws found on page {page_num}, stopping...")
                            break
                        
                        print(f"Found {len(laws)} laws on page {page_num}")
                        
                        # 법률 상세 정보 수집
                        for i, law in enumerate(laws):
                            if interrupted:
                                break
                            
                            if target_count and collected_count >= target_count:
                                break
                            
                            try:
                                print(f"   Collecting law {collected_count + 1}: {law['law_name'][:50]}...")
                                
                                # 법률 상세 정보 수집
                                detail = client.get_law_detail(law['cont_id'], law['cont_sid'])
                                
                                if detail:
                                    # 수집기에 추가
                                    collector.save_item(detail)
                                    collected_count += 1
                                    
                                    print(f"   Collected: {detail['law_name'][:50]}...")
                                else:
                                    print(f"   Failed to collect: {law['law_name'][:50]}...")
                                    self.failed_items.append(law)
                                
                                # 배치 저장
                                if collected_count % collector.batch_size == 0:
                                    collector._save_batch()
                                    print(f"   Batch saved: {collected_count} items")
                                
                            except Exception as e:
                                print(f"   Error collecting law: {e}")
                                self.failed_items.append(law)
                                continue
                        
                        # 페이지 완료 후 체크포인트 저장
                        checkpoint_data = {
                            'data_type': 'law_only',
                            'current_page': page_num,
                            'collected_count': collected_count,
                            'failed_count': len(self.failed_items),
                            'timestamp': datetime.now().isoformat(),
                            'target_count': target_count
                        }
                        checkpoint_mgr.save_checkpoint(checkpoint_data)
                        
                        print(f"Page {page_num} completed: {collected_count} total items")
                        
                        page_num += 1
                        
                    except Exception as e:
                        print(f"Error processing page {page_num}: {e}")
                        logger.error(f"Page {page_num} error: {e}")
                        break
                
                # 최종 배치 저장
                collector._save_batch()
                
        except Exception as e:
            print(f"Collection failed: {e}")
            logger.error(f"Collection failed: {e}")
            return False
        
        # 최종 통계
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n{'='*60}")
        print(f"LAW COLLECTION COMPLETED!")
        print(f"{'='*60}")
        print(f"Final Statistics:")
        print(f"   Total collected: {collected_count} laws")
        print(f"   Failed items: {len(self.failed_items)}")
        print(f"   Duration: {duration}")
        print(f"   Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        return True

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='국회 법률정보시스템 법률만 수집 (특정 URL용)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws_only.py --sample 10     # 샘플 10개
  python collect_laws_only.py --sample 100    # 샘플 100개
  python collect_laws_only.py --full          # 전체 1895개
  python collect_laws_only.py --resume        # 중단 지점에서 재개
        """
    )
    
    parser.add_argument('--sample', type=int, metavar='N',
                       help='샘플 수집 개수 (10, 100, 1000 등)')
    parser.add_argument('--full', action='store_true',
                       help='전체 수집 (1895개)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='체크포인트에서 재개 (기본값)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='처음부터 시작')
    parser.add_argument('--start-page', type=int, default=1,
                       help='시작 페이지 번호 (기본: 1)')
    parser.add_argument('--page-size', type=int, default=100,
                       help='페이지당 항목 수 (기본: 100)')
    
    args = parser.parse_args()
    
    # 목표 수집 개수 결정
    if args.full:
        target_count = 1895  # 전체 법률 수
        print(f"Full collection mode: {target_count} laws")
    elif args.sample:
        target_count = args.sample
        print(f"Sample mode: {target_count} items")
    else:
        target_count = 100  # 기본값
        print(f"Default mode: {target_count} items")
    
    # 수집기 생성 및 실행
    collector = LawOnlyCollector()
    
    try:
        success = collector.collect_laws(
            target_count=target_count,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
        
        if success:
            print(f"\nCollection completed successfully!")
            return 0
        else:
            print(f"\nCollection failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\nCollection interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
