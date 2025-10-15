#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 법률정보시스템 법률 수집 (최적화 버전)

성능 개선 사항:
1. 배치 처리로 메모리 효율성 향상
2. 불필요한 메모리 체크 최소화
3. 파일 I/O 최적화
4. 페이지 네비게이션 최적화
5. 에러 처리 개선

사용법:
  python collect_laws_optimized.py --sample 10     # 샘플 10개
  python collect_laws_optimized.py --sample 100    # 샘플 100개
  python collect_laws_optimized.py --sample 1000   # 샘플 1000개
  python collect_laws_optimized.py --full          # 전체 7602개
  python collect_laws_optimized.py --resume        # 중단 지점에서 재개
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
from scripts.assembly.assembly_logger import setup_logging, log_progress, log_memory_usage, log_collection_stats, log_checkpoint_info
from scripts.monitoring.metrics_collector import LawCollectionMetrics

# 로거 설정
logger = setup_logging("law_collection_optimized")

# Graceful shutdown 처리
interrupted = False

def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    global interrupted
    logger.warning("\n⚠️ Interrupt signal received. Saving progress...")
    interrupted = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class OptimizedLawCollector:
    """최적화된 법률 수집기 (페이지별 저장)"""
    
    def __init__(self, base_dir: str = "data/raw/assembly", page_size: int = 10, enable_metrics: bool = True):
        self.base_dir = Path(base_dir)
        self.page_size = page_size
        self.collected_items = []
        self.failed_items = []
        self.start_time = None
        self.enable_metrics = enable_metrics
        
        # 출력 디렉토리 생성
        self.output_dir = self.base_dir / "law" / datetime.now().strftime("%Y%m%d")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # 메트릭 수집기 초기화
        if self.enable_metrics:
            try:
                # 기존 메트릭 서버가 실행 중인지 확인
                import requests
                try:
                    response = requests.get("http://localhost:8000/metrics", timeout=1)
                    if response.status_code == 200:
                        print(f"📊 Using existing metrics server")
                        # 기존 서버를 사용하되, 메트릭 인스턴스는 새로 생성
                        self.metrics = LawCollectionMetrics()
                        self.metrics.start_collection()
                        print(f"📊 Connected to existing metrics server")
                    else:
                        raise Exception("Metrics server not responding")
                except:
                    # 메트릭 서버가 없으면 새로 시작
                    print(f"📊 Starting new metrics server")
                    self.metrics = LawCollectionMetrics()
                    self.metrics.start_server()
                    self.metrics.start_collection()
                    print(f"📊 Metrics server started")
                
                print(f"📊 Metrics collection enabled")
            except Exception as e:
                print(f"⚠️ Failed to start metrics: {e}")
                self.enable_metrics = False
                self.metrics = None
        else:
            self.metrics = None
            print(f"📊 Metrics collection disabled")
    
    def add_item(self, item: Dict):
        """아이템 추가 (페이지별 처리용)"""
        self.collected_items.append(item)
    
    def save_page(self, page_number: int):
        """페이지별 저장 (10개씩)"""
        if not self.collected_items:
            return
        
        # 페이지 처리 시간 기록
        page_start_time = time.time()
        
        timestamp = datetime.now().strftime("%H%M%S")
        page_filename = f"law_page_{page_number:03d}_{timestamp}.json"
        page_filepath = self.output_dir / page_filename
        
        page_data = {
            "page_info": {
                "page_number": page_number,
                "laws_count": len(self.collected_items),
                "saved_at": datetime.now().isoformat(),
                "page_size": self.page_size
            },
            "laws": self.collected_items
        }
        
        try:
            with open(page_filepath, 'w', encoding='utf-8') as f:
                json.dump(page_data, f, ensure_ascii=False, indent=2)
            
            print(f"📄 Page {page_number} saved: {page_filename} ({len(self.collected_items)} laws)")
            
            # 메트릭 기록
            if self.metrics:
                page_duration = time.time() - page_start_time
                self.metrics.record_page_processed(page_number)
                self.metrics.record_laws_collected(len(self.collected_items))
                self.metrics.record_page_processing_time(page_duration)
                
                # 페이지 정보에 처리 시간 추가
                page_data["page_info"]["processing_time"] = page_duration
            
            self.collected_items.clear()
            
        except Exception as e:
            print(f"❌ Failed to save page {page_number}: {e}")
            if self.metrics:
                self.metrics.record_error("file_save_error")
    
    def add_failed_item(self, item: Dict, error: str):
        """실패한 아이템 추가"""
        self.failed_items.append({
            'item': item,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        
        # 에러 메트릭 기록
        if self.metrics:
            error_type = "network_error" if "network" in error.lower() else "parsing_error"
            self.metrics.record_error(error_type)
    
    def finalize(self):
        """최종 저장"""
        if self.collected_items:
            # 남은 아이템들을 마지막 페이지로 저장
            timestamp = datetime.now().strftime("%H%M%S")
            page_filename = f"law_page_final_{timestamp}.json"
            page_filepath = self.output_dir / page_filename
            
            page_data = {
                "page_info": {
                    "page_number": "final",
                    "laws_count": len(self.collected_items),
                    "saved_at": datetime.now().isoformat(),
                    "page_size": self.page_size
                },
                "laws": self.collected_items
            }
            
            try:
                with open(page_filepath, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                
                print(f"📄 Final page saved: {page_filename} ({len(self.collected_items)} laws)")
                
                # 메트릭 기록
                if self.metrics:
                    self.metrics.record_laws_collected(len(self.collected_items))
                
                self.collected_items.clear()
                
            except Exception as e:
                print(f"❌ Failed to save final page: {e}")
                if self.metrics:
                    self.metrics.record_error("file_save_error")
        
        # 실패한 아이템들 저장
        if self.failed_items:
            failed_filename = f"failed_items_{datetime.now().strftime('%H%M%S')}.json"
            failed_filepath = self.output_dir / failed_filename
            
            with open(failed_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.failed_items, f, ensure_ascii=False, indent=2)
            
            print(f"❌ Failed items saved: {failed_filename} ({len(self.failed_items)} items)")
        
        # 메트릭 수집 중지
        if self.metrics:
            self.metrics.stop_collection()
    
    @property
    def collected_count(self) -> int:
        """수집된 총 아이템 수"""
        # 페이지 파일들에서 총 개수 계산
        total_count = len(self.collected_items)  # 현재 메모리에 있는 아이템들
        
        # 저장된 페이지 파일들에서 개수 합산
        for page_file in self.output_dir.glob("law_page_*.json"):
            try:
                with open(page_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                    if 'page_info' in page_data and 'laws_count' in page_data['page_info']:
                        total_count += page_data['page_info']['laws_count']
            except Exception:
                continue
        
        return total_count

def collect_laws_optimized(
    target_count: int = None,
    page_size: int = 100,
    resume: bool = True,
    start_page: int = 1,
    laws_per_page: int = 10,
    enable_metrics: bool = True
):
    """
    최적화된 점진적 법률 수집 (페이지별 저장)
    
    Args:
        target_count: 목표 수집 건수 (None=전체)
        page_size: 페이지당 항목 수 (100 권장)
        resume: 체크포인트에서 재개
        start_page: 시작 페이지 번호
        laws_per_page: 페이지당 법률 수 (10개 고정)
        enable_metrics: 메트릭 수집 활성화 여부
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZED LAW COLLECTION STARTED")
    print(f"{'='*60}")
    
    # 체크포인트 매니저
    checkpoint_mgr = CheckpointManager("data/checkpoints/laws")
    print(f"📁 Checkpoint directory: data/checkpoints/laws")
    
    # 체크포인트 로드
    actual_start_page = start_page
    checkpoint = None
    
    if resume:
        print(f"🔍 Checking for existing checkpoint...")
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            print(f"📂 Resuming from checkpoint")
            print(f"   Data type: {checkpoint.get('data_type', 'unknown')}")
            print(f"   Category: {checkpoint.get('category', 'None')}")
            print(f"   Page: {checkpoint.get('current_page', 0)}/{checkpoint.get('total_pages', 0)}")
            print(f"   Collected: {checkpoint.get('collected_count', 0)} items")
            print(f"   Memory: {checkpoint.get('memory_usage_mb', 0):.1f}MB")
            actual_start_page = checkpoint['current_page'] + 1
        else:
            print(f"📂 No checkpoint found, starting from page {start_page}")
    else:
        print(f"📂 Resume disabled, starting from page {start_page}")
    
    # 최적화된 수집기 초기화
    print(f"\n📦 Initializing optimized collector...")
    collector = OptimizedLawCollector(
        base_dir="data/raw/assembly",
        page_size=laws_per_page,
        enable_metrics=enable_metrics
    )
    print(f"✅ Optimized collector initialized (page size: {laws_per_page})")
    
    # 시작 시간 설정
    start_time = datetime.now().isoformat()
    collector.start_time = start_time
    
    # 전체 페이지 계산
    if target_count:
        total_pages = actual_start_page + (target_count + 9) // 10 - 1  # 페이지당 10개
    else:
        total_pages = 100  # 대략적인 페이지 수
    
    print(f"\n📊 Collection Parameters:")
    print(f"   Target: {target_count or 'ALL (7602)'} items")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Laws per page: {laws_per_page} (fixed)")
    print(f"   Save mode: Page-by-page")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    last_memory_check = 0
    
    try:
        print(f"\n🌐 Starting Playwright browser...")
        # Playwright 시작 (최적화된 설정)
        with AssemblyPlaywrightClient(
            rate_limit=3.0,  # Rate limiting 유지
            headless=True,
            memory_limit_mb=1000  # 메모리 제한 증가
        ) as client:
            print(f"✅ Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if interrupted:
                    print(f"\n⚠️ INTERRUPTED by user signal")
                    break
                
                print(f"\n{'─'*50}")
                print(f"📄 Processing Page {page}/{total_pages}")
                print(f"{'─'*50}")
                
                # 메모리 체크 최적화 (10페이지마다만 체크)
                if page - last_memory_check >= 10:
                    memory_mb = client.check_memory_usage()
                    print(f"📊 Memory usage: {memory_mb:.1f}MB")
                    last_memory_check = page
                
                # 목록 조회
                print(f"🔍 Fetching law list from page {page}...")
                laws = client.get_law_list_page(page_num=page, page_size=10)
                print(f"✅ Found {len(laws)} laws on page")
                
                if not laws:
                    print(f"⚠️ No laws found on page {page}, skipping...")
                    continue
                
                # 각 법률 상세 수집 (최적화된 처리)
                print(f"📋 Processing {len(laws)} laws...")
                page_start_time = time.time()
                
                for idx, law_item in enumerate(laws, 1):
                    if interrupted:
                        print(f"\n⚠️ INTERRUPTED during law processing")
                        break
                    
                    try:
                        print(f"   [{idx:2d}/{len(laws)}] Processing: {law_item['law_name'][:50]}...")
                        
                        detail = client.get_law_detail(
                            law_item['cont_id'],
                            law_item['cont_sid']
                        )
                        
                        # 목록 정보 병합
                        detail.update({
                            'row_number': law_item['row_number'],
                            'category': law_item['category'],
                            'law_type': law_item['law_type'],
                            'promulgation_number': law_item['promulgation_number'],
                            'promulgation_date': law_item['promulgation_date'],
                            'enforcement_date': law_item['enforcement_date'],
                            'amendment_type': law_item['amendment_type']
                        })
                        
                        collector.add_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ✅ Collected (Page: {len(collector.collected_items)}/{laws_per_page})")
                        
                        # 목표 달성 체크
                        if target_count and collected_this_run >= target_count:
                            print(f"\n🎯 TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ❌ Failed: {str(e)[:100]}...")
                        collector.add_failed_item(law_item, str(e))
                        continue
                
                # 페이지별 저장
                collector.save_page(page)
                
                # 페이지 처리 시간 계산
                page_time = time.time() - page_start_time
                print(f"⏱️ Page {page} processed in {page_time:.1f}s")
                
                # 진행률 로그 (최적화된 출력)
                print(f"\n📈 Progress Summary:")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collected_this_run/(collected_this_run + len(collector.failed_items))*100:.1f}%" if (collected_this_run + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                # 체크포인트 저장 (최적화된 빈도)
                if page % 5 == 0:  # 5페이지마다만 체크포인트 저장
                    checkpoint_data = {
                        'data_type': 'law',
                        'category': None,
                        'current_page': page,
                        'total_pages': total_pages,
                        'collected_count': collector.collected_count,
                        'collected_this_run': collected_this_run,
                        'start_time': checkpoint['start_time'] if checkpoint else start_time,
                        'memory_usage_mb': client.check_memory_usage(),
                        'target_count': target_count,
                        'page_size': page_size,
                        'laws_per_page': laws_per_page
                    }
                    
                    checkpoint_mgr.save_checkpoint(checkpoint_data)
                    print(f"💾 Checkpoint saved at page {page}")
                
                # 목표 달성 시 종료
                if target_count and collected_this_run >= target_count:
                    print(f"\n🎯 Target achieved, stopping collection")
                    break
            
            # 수집 완료
            print(f"\n🏁 Finalizing collection...")
            collector.finalize()
            
            if not interrupted:
                checkpoint_mgr.clear_checkpoint()
                print(f"\n✅ COLLECTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n⚠️ COLLECTION INTERRUPTED (progress saved)")
            
            # 최종 통계
            print(f"\n📊 Final Statistics:")
            print(f"   Total collected: {collector.collected_count} items")
            print(f"   Failed: {len(collector.failed_items)} items")
            print(f"   Requests made: {client.request_count}")
            print(f"   Rate limit: {client.get_stats()['rate_limit']}s")
            print(f"   Timeout: {client.get_stats()['timeout']}ms")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"🔧 Finalizing collector...")
        collector.finalize()
        raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='국회 법률정보시스템 법률 수집 (최적화 버전)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws_optimized.py --sample 10                    # 샘플 10개 수집
  python collect_laws_optimized.py --sample 100                   # 샘플 100개 수집
  python collect_laws_optimized.py --sample 100 --start-page 5     # 5페이지부터 100개 수집
  python collect_laws_optimized.py --sample 50 --start-page 10     # 10페이지부터 50개 수집
  python collect_laws_optimized.py --full                          # 전체 7602개 수집
  python collect_laws_optimized.py --resume                        # 중단 지점에서 재개
  python collect_laws_optimized.py --sample 100 --laws-per-page 10   # 페이지당 10개로 수집
        """
    )
    
    parser.add_argument('--sample', type=int, metavar='N',
                        help='샘플 수집 개수 (10, 100, 1000 등)')
    parser.add_argument('--full', action='store_true',
                        help='전체 수집 (7602개)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='체크포인트에서 재개 (기본값)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='처음부터 시작')
    parser.add_argument('--page-size', type=int, default=100,
                        help='페이지당 항목 수 (기본: 100)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='시작 페이지 번호 (기본: 1)')
    parser.add_argument('--laws-per-page', type=int, default=10,
                        help='페이지당 법률 수 (기본: 10)')
    parser.add_argument('--enable-metrics', action='store_true', default=True,
                        help='메트릭 수집 활성화 (기본값)')
    parser.add_argument('--disable-metrics', dest='enable_metrics', action='store_false',
                        help='메트릭 수집 비활성화')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='로그 레벨 (기본: INFO)')
    
    args = parser.parse_args()
    
    # 로그 레벨 재설정
    if args.log_level != 'INFO':
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.sample:
        print(f"📦 Sample mode: {args.sample} items")
        collect_laws_optimized(
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            laws_per_page=args.laws_per_page,
            enable_metrics=args.enable_metrics
        )
    elif args.full:
        logger.info(f"📦 Full mode: 7602 items")
        collect_laws_optimized(
            target_count=None,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            laws_per_page=args.laws_per_page,
            enable_metrics=args.enable_metrics
        )
    else:
        parser.print_help()
        logger.error("\n❌ Please specify --sample N or --full")
        sys.exit(1)

if __name__ == "__main__":
    main()
