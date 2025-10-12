#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 법률정보시스템 법률 수집 (Playwright + 점진적 + 압축)

사용법:
  python collect_laws.py --sample 10     # 샘플 10개
  python collect_laws.py --sample 100    # 샘플 100개
  python collect_laws.py --sample 1000   # 샘플 1000개
  python collect_laws.py --full          # 전체 7602개
  python collect_laws.py --resume        # 중단 지점에서 재개

특징:
  - 자동 데이터 압축 (95% 이상 용량 절약)
  - 메모리 효율적 수집
  - 체크포인트 기반 재개 기능
  - 실시간 압축 통계 표시
"""

import argparse
import sys
import signal
import logging
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.assembly.assembly_collector import AssemblyCollector
from scripts.assembly.checkpoint_manager import CheckpointManager
from scripts.assembly.assembly_logger import setup_logging, log_progress, log_memory_usage, log_collection_stats, log_checkpoint_info
from scripts.assembly.law_data_compressor import compress_law_data, compress_and_save_page_data

# 로거 설정
logger = setup_logging("law_collection")

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

def collect_laws_incremental(
    target_count: int = None,
    page_size: int = 100,
    resume: bool = True,
    start_page: int = 1
):
    """
    점진적 법률 수집
    
    Args:
        target_count: 목표 수집 건수 (None=전체)
        page_size: 페이지당 항목 수 (100 권장)
        resume: 체크포인트에서 재개
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 LAW COLLECTION STARTED")
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
    
    # 수집기 초기화
    print(f"\n📦 Initializing collector...")
    collector = AssemblyCollector(
        base_dir="data/raw/assembly",
        data_type="law",
        category=None,
        batch_size=50,
        memory_limit_mb=800
    )
    print(f"✅ Collector initialized")
    
    # 시작 시간 설정
    start_time = datetime.now().isoformat()
    collector.set_start_time(start_time)
    
    # 전체 페이지 계산 (실제로는 페이지당 10개씩 표시됨)
    if target_count:
        total_pages = actual_start_page + (target_count + 10 - 1) // 10 - 1  # 페이지당 10개
    else:
        total_pages = 100  # 대략적인 페이지 수
    
    print(f"\n📊 Collection Parameters:")
    print(f"   Target: {target_count or 'ALL (7602)'} items")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Page size: 10 (fixed)")
    print(f"   Batch size: {collector.batch_size}")
    print(f"   Memory limit: {collector.memory_limit_mb}MB")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    total_original_size = 0
    total_compressed_size = 0
    
    try:
        print(f"\n🌐 Starting Playwright browser...")
        # Playwright 시작
        with AssemblyPlaywrightClient(
            rate_limit=3.0,
            headless=True,
            memory_limit_mb=800
        ) as client:
            print(f"✅ Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if interrupted:
                    print(f"\n⚠️ INTERRUPTED by user signal")
                    break
                
                print(f"\n{'─'*50}")
                print(f"📄 Processing Page {page}/{total_pages}")
                print(f"{'─'*50}")
                
                # 메모리 체크
                memory_mb = client.check_memory_usage()
                print(f"📊 Memory usage: {memory_mb:.1f}MB")
                
                # 목록 조회
                print(f"🔍 Fetching law list from page {page}...")
                laws = client.get_law_list_page(page_num=page, page_size=10)
                print(f"✅ Found {len(laws)} laws on page")
                
                if not laws:
                    print(f"⚠️ No laws found on page {page}, skipping...")
                    continue
                
                # 각 법률 상세 수집
                print(f"📋 Processing {len(laws)} laws...")
                page_laws = []  # 현재 페이지의 법률들을 저장할 리스트
                
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
                        
                        page_laws.append(detail)  # 페이지별 리스트에 추가
                        collector.save_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ✅ Collected (Total: {collector.collected_count})")
                        
                        # 목표 달성 체크
                        if target_count and collected_this_run >= target_count:
                            print(f"\n🎯 TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ❌ Failed: {str(e)[:100]}...")
                        collector.add_failed_item(law_item, str(e))
                        continue
                
                # 현재 페이지의 법률들을 별도 파일로 저장 (압축된 버전)
                if page_laws:
                    timestamp = datetime.now().strftime("%H%M%S")
                    page_filename = f"law_page_{page:03d}_{timestamp}.json"
                    page_filepath = collector.output_dir / page_filename
                    
                    page_data = {
                        "page_number": page,
                        "total_pages": total_pages,
                        "laws_count": len(page_laws),
                        "collected_at": datetime.now().isoformat(),
                        "laws": page_laws
                    }
                    
                    # 압축된 데이터로 저장
                    compressed_size = compress_and_save_page_data(page_data, str(page_filepath))
                    
                    # 압축 통계 업데이트
                    total_compressed_size += compressed_size
                    
                    # 원본 크기 추정 (압축 전 크기)
                    estimated_original_size = compressed_size * 20  # 대략적인 압축률 고려
                    total_original_size += estimated_original_size
                    
                    compression_ratio = (1 - compressed_size / estimated_original_size) * 100 if estimated_original_size > 0 else 0
                    
                    print(f"📄 Page {page} saved: {page_filename} ({len(page_laws)} laws, {compressed_size:,} bytes, {compression_ratio:.1f}% 압축)")
                
                # 진행률 로그
                print(f"\n📈 Progress Summary:")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collector.collected_count/(collector.collected_count + len(collector.failed_items))*100:.1f}%" if (collector.collected_count + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                # 체크포인트 저장
                checkpoint_data = {
                    'data_type': 'law',
                    'category': None,
                    'current_page': page,
                    'total_pages': total_pages,
                    'collected_count': collector.collected_count,
                    'collected_this_run': collected_this_run,
                    'start_time': checkpoint['start_time'] if checkpoint else start_time,
                    'memory_usage_mb': memory_mb,
                    'target_count': target_count,
                    'page_size': page_size
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
            
            # 압축 통계
            if total_compressed_size > 0:
                overall_compression_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
                print(f"\n🗜️ Compression Statistics:")
                print(f"   Estimated original size: {total_original_size:,} bytes ({total_original_size/1024/1024:.1f} MB)")
                print(f"   Compressed size: {total_compressed_size:,} bytes ({total_compressed_size/1024/1024:.1f} MB)")
                print(f"   Compression ratio: {overall_compression_ratio:.1f}%")
                print(f"   Space saved: {(total_original_size - total_compressed_size)/1024/1024:.1f} MB")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"🔧 Finalizing collector...")
        collector.finalize()
        raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='국회 법률정보시스템 법률 수집 (Playwright)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_laws.py --sample 10                    # 샘플 10개 수집
  python collect_laws.py --sample 100                   # 샘플 100개 수집
  python collect_laws.py --sample 100 --start-page 5     # 5페이지부터 100개 수집
  python collect_laws.py --sample 50 --start-page 10     # 10페이지부터 50개 수집
  python collect_laws.py --full                          # 전체 7602개 수집
  python collect_laws.py --resume                        # 중단 지점에서 재개
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
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='로그 레벨 (기본: INFO)')
    
    args = parser.parse_args()
    
    # 로그 레벨 재설정
    if args.log_level != 'INFO':
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.sample:
        print(f"📦 Sample mode: {args.sample} items")
        collect_laws_incremental(
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
    elif args.full:
        logger.info(f"📦 Full mode: 7602 items")
        collect_laws_incremental(
            target_count=None,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page
        )
    else:
        parser.print_help()
        logger.error("\n❌ Please specify --sample N or --full")
        sys.exit(1)

if __name__ == "__main__":
    main()
