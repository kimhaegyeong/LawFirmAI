#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국회 법률정보시스템 분야별 판례 수집 (Playwright + 점진적)

사용법:
  python collect_precedents_by_category.py --category civil --sample 50
  python collect_precedents_by_category.py --category criminal --sample 100
  python collect_precedents_by_category.py --category family --sample 30
  python collect_precedents_by_category.py --all-categories --sample 20
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.assembly_playwright_client import AssemblyPlaywrightClient
from scripts.data_collection.common.assembly_collector import AssemblyCollector
from scripts.data_collection.common.checkpoint_manager import CheckpointManager
from scripts.data_collection.common.common_utils import (
    CollectionConfig, CollectionLogger, SignalHandler, 
    MemoryManager, RetryManager, DataOptimizer, 
    check_system_requirements, memory_monitor, retry_on_failure
)

# 로깅 설정
logger = CollectionLogger.setup_logging("precedent_category_collection")

# 시그널 핸들러 등록
signal_handler = SignalHandler()

# 분야별 코드 매핑 (실제 국회 시스템 기준)
CATEGORY_CODES = {
    'civil': 'PREC00_001',      # 민사
    'criminal': 'PREC00_002',   # 형사
    'tax': 'PREC00_003',        # 조세
    'administrative': 'PREC00_004',  # 행정
    'family': 'PREC00_005',     # 가사
    'patent': 'PREC00_006',     # 특허
    'maritime': 'PREC00_009',   # 해사
    'military': 'PREC00_010'    # 군사
}

CATEGORY_NAMES = {
    'civil': '민사',
    'criminal': '형사', 
    'tax': '조세',
    'administrative': '행정',
    'family': '가사',
    'patent': '특허',
    'maritime': '해사',
    'military': '군사'
}

@memory_monitor(threshold_mb=500.0)  # 300MB → 500MB로 조정
def collect_precedents_by_category(
    category: str,
    target_count: int = None,
    page_size: int = 10,
    resume: bool = True,
    start_page: int = 1,
    config: CollectionConfig = None
):
    """
    분야별 판례 수집 (개선된 버전)
    
    Args:
        category: 분야 코드 (civil, criminal, family 등)
        target_count: 목표 수집 건수 (None=전체)
        page_size: 페이지당 항목 수 (실제로는 10개 고정)
        resume: 체크포인트에서 재개 여부
        start_page: 시작 페이지 번호
        config: 수집 설정 객체
    """
    if config is None:
        config = CollectionConfig()
    
    # 시스템 요구사항 확인
    if not check_system_requirements(min_memory_gb=2.0):
        logger.warning("Proceeding with caution due to system constraints")
    category_code = CATEGORY_CODES.get(category)
    category_name = CATEGORY_NAMES.get(category, category)
    
    if not category_code:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(CATEGORY_CODES.keys())}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 PRECEDENT COLLECTION BY CATEGORY STARTED")
    print(f"📋 Category: {category_name} ({category_code})")
    print(f"{'='*60}")
    
    # 체크포인트 매니저
    checkpoint_mgr = CheckpointManager(f"data/checkpoints/precedents_{category}")
    print(f"📁 Checkpoint directory: data/checkpoints/precedents_{category}")
    
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
    
    # 메모리 매니저 및 재시도 매니저 초기화
    memory_manager = MemoryManager(
        memory_limit_mb=config.get('memory_limit_mb'),
        cleanup_threshold=config.get('cleanup_threshold')
    )
    retry_manager = RetryManager(
        max_retries=config.get('max_retries'),
        base_delay=1.0
    )
    
    # 수집기 초기화 (설정 기반)
    logger.info("Initializing collector...")
    collector = AssemblyCollector(
        base_dir="data/raw/assembly",
        data_type="precedent",
        category=category,
        batch_size=config.get('batch_size'),
        memory_limit_mb=config.get('memory_limit_mb')
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
    print(f"   Category: {category_name} ({category_code})")
    print(f"   Target: {target_count or 'ALL'}")
    print(f"   Pages: {actual_start_page} to {total_pages}")
    print(f"   Page size: 10 (fixed)")
    print(f"   Batch size: {collector.batch_size}")
    print(f"   Memory limit: {collector.memory_limit_mb}MB")
    print(f"   Start time: {start_time}")
    
    collected_this_run = 0
    
    try:
        print(f"\n🌐 Starting Playwright browser...")
        with AssemblyPlaywrightClient(
            rate_limit=config.get('rate_limit'),
            headless=True,
            memory_limit_mb=config.get('memory_limit_mb')
        ) as client:
            print(f"✅ Playwright browser started")
            
            for page in range(actual_start_page, total_pages + 1):
                if signal_handler.is_interrupted():
                    logger.warning("Collection interrupted by user signal")
                    break
                
                print(f"\n{'─'*50}")
                print(f"📄 Processing Page {page}/{total_pages}")
                print(f"📋 Category: {category_name}")
                print(f"{'─'*50}")
                
                # 페이지 정보 설정
                collector.set_page_info(page)
                
                # 메모리 상태 확인
                memory_mb = memory_manager.get_memory_usage()
                CollectionLogger.log_memory_usage(logger, memory_mb, config.get('memory_limit_mb'))
                
                # 재시도 로직으로 판례 목록 가져오기
                logger.info(f"Fetching {category_name} precedent list from page {page}...")
                precedents = retry_manager.retry(
                    client.get_precedent_list_page_by_category,
                    category_code=category_code,
                    page_num=page, 
                    page_size=10
                )
                print(f"✅ Found {len(precedents)} precedents on page")
                
                if not precedents:
                    print(f"⚠️ No precedents found on page {page}, skipping...")
                    continue
                
                # 각 판례 상세 수집 (스트리밍 처리)
                print(f"📋 Processing {len(precedents)} precedents...")
                
                for idx, precedent_item in enumerate(precedents, 1):
                    if signal_handler.is_interrupted():
                        print(f"\n⚠️ INTERRUPTED during precedent processing")
                        break
                    
                    try:
                        print(f"   [{idx:2d}/{len(precedents)}] Processing: {precedent_item['case_name'][:50]}...")
                        
                        detail = client.get_precedent_detail(precedent_item)
                        
                        # 분야 정보 추가
                        detail.update({
                            'category': category_name,
                            'category_code': category_code
                        })
                        
                        # 데이터 최적화 적용
                        from scripts.data_collection.common.common_utils import DataOptimizer
                        detail = DataOptimizer.optimize_item(detail, 'precedent')
                        print(f"      🔧 Data optimized for memory efficiency")
                        
                        # 스트리밍 처리: 즉시 저장하고 메모리에서 해제
                        collector.save_item(detail)
                        collected_this_run += 1
                        
                        print(f"      ✅ Collected (Total: {collector.collected_count})")
                        
                        # 즉시 메모리에서 해제
                        del detail
                        
                        # 메모리 정리 (매 3개마다 - 더 자주)
                        if idx % 3 == 0:
                            import gc
                            gc.collect()
                            print(f"      🧹 Memory cleanup at item {idx}")
                        
                        # 목표 달성 체크
                        if target_count and collected_this_run >= target_count:
                            print(f"\n🎯 TARGET REACHED: {collected_this_run}/{target_count}")
                            break
                        
                    except Exception as e:
                        print(f"      ❌ Failed: {str(e)[:100]}...")
                        collector.add_failed_item(precedent_item, str(e))
                        continue
                
                # 스트리밍 처리로 인해 페이지별 저장은 생략 (이미 개별 저장됨)
                # 메모리 정리만 수행
                import gc
                gc.collect()
                print(f"🧹 Page {page} processing completed and memory cleaned up")
                
                # 진행률 로그
                print(f"\n📈 Progress Summary:")
                print(f"   Category: {category_name}")
                print(f"   Page: {page}/{total_pages} ({page/total_pages*100:.1f}%)")
                print(f"   Collected this run: {collected_this_run}")
                print(f"   Total collected: {collector.collected_count}")
                print(f"   Failed: {len(collector.failed_items)}")
                print(f"   Success rate: {collector.collected_count/(collector.collected_count + len(collector.failed_items))*100:.1f}%" if (collector.collected_count + len(collector.failed_items)) > 0 else "   Success rate: N/A")
                
                checkpoint_data = {
                    'data_type': 'precedent',
                    'category': category,
                    'category_code': category_code,
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
                
                if target_count and collected_this_run >= target_count:
                    print(f"\n🎯 Target achieved, stopping collection")
                    break
            
            print(f"\n🏁 Finalizing collection...")
            collector.finalize()
            
            if not signal_handler.is_interrupted():
                checkpoint_mgr.clear_checkpoint()
                print(f"\n✅ COLLECTION COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n⚠️ COLLECTION INTERRUPTED (progress saved)")
            
            # 최종 통계
            print(f"\n📊 Final Statistics:")
            print(f"   Category: {category_name}")
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

def collect_all_categories(target_count_per_category: int = 50):
    """모든 분야별로 판례 수집"""
    categories = ['civil', 'criminal', 'tax', 'administrative', 'family', 'patent']
    
    print(f"\n{'='*60}")
    print(f"🚀 COLLECTING PRECEDENTS FOR ALL CATEGORIES")
    print(f"📋 Categories: {', '.join([CATEGORY_NAMES[cat] for cat in categories])}")
    print(f"📊 Target per category: {target_count_per_category}")
    print(f"{'='*60}")
    
    total_collected = 0
    
    for category in categories:
        try:
            print(f"\n🔄 Starting collection for {CATEGORY_NAMES[category]}...")
            collect_precedents_by_category(
                category=category,
                target_count=target_count_per_category,
                resume=False,  # 각 분야별로 새로 시작
                start_page=1
            )
            print(f"✅ Completed {CATEGORY_NAMES[category]} collection")
            
        except Exception as e:
            print(f"❌ Failed to collect {CATEGORY_NAMES[category]}: {e}")
            continue
    
    print(f"\n🎉 ALL CATEGORIES COLLECTION COMPLETED!")
    print(f"📊 Total collected: {total_collected} precedents")

def main():
    """메인 함수 (개선된 버전)"""
    parser = argparse.ArgumentParser(
        description='국회 법률정보시스템 분야별 판례 수집 (Playwright + 개선된 메모리 관리)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available categories:
  civil          - 민사 (PREC00_001)
  criminal        - 형사 (PREC00_002)  
  tax             - 조세 (PREC00_003)
  administrative  - 행정 (PREC00_004)
  family          - 가사 (PREC00_005)
  patent          - 특허 (PREC00_006)
  maritime        - 해사 (PREC00_009)
  military        - 군사 (PREC00_010)

Examples:
  python collect_precedents_by_category.py --category civil --sample 50
  python collect_precedents_by_category.py --category criminal --sample 100 --memory-limit 500
  python collect_precedents_by_category.py --all-categories --sample 20 --batch-size 15
        """
    )
    
    parser.add_argument('--category', type=str, 
                        choices=list(CATEGORY_CODES.keys()),
                        help='수집할 분야 선택')
    parser.add_argument('--all-categories', action='store_true',
                        help='모든 분야 수집 (민사, 형사, 가사)')
    parser.add_argument('--sample', type=int, metavar='N',
                        help='샘플 수집 개수 (10, 50, 100 등)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='체크포인트에서 재개 (기본값)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='처음부터 시작')
    parser.add_argument('--page-size', type=int, default=100,
                        help='페이지당 항목 수 (기본: 100)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='시작 페이지 번호 (기본: 1)')
    parser.add_argument('--memory-limit', type=int, default=600,
                        help='메모리 제한 (MB, 기본: 600)')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='배치 크기 (기본: 20)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='최대 재시도 횟수 (기본: 3)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='로그 레벨 (기본: INFO)')
    
    args = parser.parse_args()
    
    # 설정 객체 생성
    config = CollectionConfig(
        memory_limit_mb=args.memory_limit,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        log_level=args.log_level
    )
    
    # 로그 레벨 재설정
    if args.log_level != 'INFO':
        logger.setLevel(getattr(logging, args.log_level))
    
    if args.all_categories:
        if not args.sample:
            args.sample = 50  # 기본값
        print(f"📦 All categories mode: {args.sample} items per category")
        collect_all_categories(target_count_per_category=args.sample)
    elif args.category:
        if not args.sample:
            print("❌ Please specify --sample N")
            sys.exit(1)
        print(f"📦 Category mode: {CATEGORY_NAMES[args.category]} - {args.sample} items")
        collect_precedents_by_category(
            category=args.category,
            target_count=args.sample,
            page_size=args.page_size,
            resume=args.resume,
            start_page=args.start_page,
            config=config
        )
    else:
        parser.print_help()
        logger.error("\n❌ Please specify --category CATEGORY or --all-categories")
        sys.exit(1)

if __name__ == "__main__":
    main()
