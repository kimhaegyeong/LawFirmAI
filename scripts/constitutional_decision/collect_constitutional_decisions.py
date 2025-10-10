#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 수집 스크립트 (리팩토링된 버전)

국가법령정보센터 LAW OPEN API를 사용하여 헌재결정례를 수집합니다.
- 최근 5년간 헌재결정례 1,000건 수집
- 헌법재판소 결정례의 상세 내용 수집
- 결정유형별 분류 (위헌, 합헌, 각하, 기각 등)
- 향상된 에러 처리, 성능 최적화, 모니터링 기능
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.law_open_api_client import LawOpenAPIConfig, load_env_file
from scripts.constitutional_decision.constitutional_collector import ConstitutionalDecisionCollector
from scripts.constitutional_decision.constitutional_logger import setup_logging

logger = setup_logging()


def check_progress():
    """중단된 지점 확인 유틸리티 함수"""
    try:
        output_dir = Path("data/raw/constitutional_decisions")
        
        if not output_dir.exists():
            print("❌ 수집된 데이터가 없습니다.")
            return
        
        print("=" * 60)
        print("📊 헌재결정례 수집 진행 상황 확인")
        print("=" * 60)
        
        # 체크포인트 파일 확인
        checkpoint_files = list(output_dir.glob("collection_checkpoint_*.json"))
        
        if not checkpoint_files:
            print("⚠️ 체크포인트 파일이 없습니다.")
            # 기존 수집된 파일들 확인
            batch_files = list(output_dir.glob("batch_*.json"))
            if batch_files:
                print(f"📁 수집된 배치 파일: {len(batch_files)}개")
                total_count = 0
                for file_path in batch_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            count = data.get('metadata', {}).get('count', 0)
                            total_count += count
                    except:
                        pass
                print(f"추정 수집 건수: {total_count}건")
            return
        
        # 가장 최근 체크포인트 파일 로드
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['stats']
        resume_info = data['resume_info']
        shutdown_info = data.get('shutdown_info', {})
        
        print("=" * 60)
        print("헌재결정례 수집 진행 상황 확인")
        print("=" * 60)
        print(f"체크포인트 파일: {latest_checkpoint.name}")
        print(f"수집 진행률: {resume_info['progress_percentage']:.1f}%")
        print(f"수집된 건수: {stats['collected_count']:,}건")
        print(f"목표 건수: {stats['target_count']:,}건")
        print(f"중복 제외 건수: {stats['duplicate_count']:,}건")
        print(f"실패 건수: {stats['failed_count']:,}건")
        print(f"처리된 키워드: {stats['keywords_processed']:,}개")
        print(f"총 키워드: {stats['total_keywords']:,}개")
        print(f"마지막 처리된 키워드: {resume_info.get('last_keyword_processed', '없음')}")
        print(f"API 요청 수: {stats['api_requests_made']:,}회")
        print(f"API 오류 수: {stats['api_errors']:,}회")
        print(f"상태: {stats['status']}")
        
        # Graceful shutdown 정보
        if shutdown_info.get('graceful_shutdown_supported'):
            print(f"Graceful shutdown 지원: 예")
            if shutdown_info.get('shutdown_requested'):
                print(f"종료 요청됨: {shutdown_info.get('shutdown_reason', '알 수 없음')}")
        else:
            print(f"Graceful shutdown 지원: 아니오")
        
        if stats.get('start_time'):
            start_time = datetime.fromisoformat(stats['start_time'])
            print(f"시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if stats.get('end_time'):
            end_time = datetime.fromisoformat(stats['end_time'])
            print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            duration = end_time - start_time
            print(f"소요 시간: {duration}")
        
        print("=" * 60)
        
        # 수집된 배치 파일들 확인
        batch_files = list(output_dir.glob("batch_*.json"))
        if batch_files:
            print(f"수집된 배치 파일: {len(batch_files)}개")
            
            # 결정유형별 통계
            decision_type_stats = {}
            for file_path in batch_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    category = data.get('metadata', {}).get('category', 'unknown')
                    count = data.get('metadata', {}).get('count', 0)
                    decision_type_stats[category] = decision_type_stats.get(category, 0) + count
                except:
                    pass
            
            if decision_type_stats:
                print("\n결정유형별 수집 현황:")
                for decision_type, count in sorted(decision_type_stats.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {decision_type}: {count:,}건")
        
        print("\n재시작하려면 다음 명령을 실행하세요:")
        print("LAW_OPEN_API_OC=your_email_id python collect_constitutional_decisions.py")
        
    except Exception as e:
        print(f"진행 상황 확인 중 오류 발생: {e}")
        print(traceback.format_exc())


def main():
    """메인 함수 (리팩토링된 버전)"""
    try:
        # 환경변수 파일 로딩
        load_env_file()
        
        # Windows에서 UTF-8 환경 설정
        if sys.platform.startswith('win'):
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            # 콘솔 코드페이지를 UTF-8로 설정
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
            except:
                pass
        
        # 환경변수 확인
        oc = os.getenv("LAW_OPEN_API_OC")
        if not oc:
            logger.error("LAW_OPEN_API_OC 환경변수가 설정되지 않았습니다.")
            logger.info("사용법: LAW_OPEN_API_OC=your_email_id python collect_constitutional_decisions.py")
            return 1
        
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # API 설정
        config = LawOpenAPIConfig(oc=oc)
        
        # 헌재결정례 수집기 생성
        collector = ConstitutionalDecisionCollector(config)
        
        # 명령행 인수 처리
        if len(sys.argv) > 1:
            if sys.argv[1] == "--check" or sys.argv[1] == "-c":
                # 진행 상황 확인 모드
                check_progress()
                return 0
            elif sys.argv[1] == "--monitor" or sys.argv[1] == "-m":
                # 실시간 모니터링 모드
                from scripts.constitutional_decision.monitor_progress import monitor_progress
                monitor_progress()
                return 0
            elif sys.argv[1] == "--no-keyword" or sys.argv[1] == "-n":
                # 키워드 없이 전체 데이터 수집 모드
                logger.info("🔍 키워드 없이 전체 헌재결정례 수집 모드")
                logger.info("📊 실시간 모니터링을 위해 별도 터미널에서 다음 명령을 실행하세요:")
                logger.info("   python scripts/constitutional_decision/collect_constitutional_decisions.py --monitor")
                collector.collect_all_decisions(target_count=1000, keyword_mode=False)
                logger.info("헌재결정례 수집이 성공적으로 완료되었습니다.")
                return 0
            elif sys.argv[1] == "--date-based" or sys.argv[1] == "-d":
                # 날짜 기반 수집 모드
                logger.info("🗓️ 날짜 기반 헌재결정례 수집 모드")
                logger.info("📋 사용법:")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy yearly --year 2024 --unlimited")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200")
                logger.info("   python scripts/constitutional_decision/collect_by_date.py --check  # 기존 데이터 확인")
                return 0
            elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
                # 도움말 출력
                print("헌재결정례 수집 스크립트 사용법 (우선순위 기반):")
                print("  python collect_constitutional_decisions.py [옵션] [목표수량]")
                print("")
                print("옵션:")
                print("  --check, -c     진행 상황 확인")
                print("  --monitor, -m   실시간 모니터링")
                print("  --no-keyword, -n  키워드 없이 전체 데이터 수집")
                print("  --date-based, -d  날짜 기반 수집 모드 안내")
                print("  --help, -h      도움말 출력")
                print("")
                print("수집 방식:")
                print("  - 우선순위 키워드 우선 수집 (헌법소원, 위헌법률심판 등)")
                print("  - 키워드별 차등 목표 건수 (최고우선: 100건, 고우선: 50건, 중우선: 30건)")
                print("  - 총 50개 이상 키워드로 체계적 수집")
                print("  - 이미 처리된 키워드는 자동으로 건너뛰기")
                print("")
                print("예시:")
                print("  python collect_constitutional_decisions.py              # 기본 1,000건 수집 (키워드 기반)")
                print("  python collect_constitutional_decisions.py 2000        # 2,000건 수집 (키워드 기반)")
                print("  python collect_constitutional_decisions.py --no-keyword # 키워드 없이 전체 데이터 수집")
                print("  python collect_constitutional_decisions.py --date-based # 날짜 기반 수집 모드 안내")
                print("  python collect_constitutional_decisions.py --check     # 진행 상황 확인")
                print("  python collect_constitutional_decisions.py --monitor   # 실시간 모니터링")
                print("")
                print("날짜 기반 수집 예시:")
                print("  python collect_by_date.py --strategy yearly --year 2024 --unlimited")
                print("  python collect_by_date.py --strategy quarterly --year 2024 --quarter 4 --target 500")
                print("  python collect_by_date.py --strategy monthly --year 2024 --month 12 --target 200")
                return 0
            else:
                try:
                    target_count = int(sys.argv[1])
                    logger.info(f"명령행 인수로 목표 수량 설정: {target_count}건")
                except ValueError:
                    logger.warning(f"잘못된 목표 수량: {sys.argv[1]}, 기본값 사용: 1000건")
                    target_count = 1000
        else:
            target_count = 1000
        
        # 수집 실행 (키워드 모드 기본값)
        collector.collect_all_decisions(target_count=target_count, keyword_mode=True)
        
        logger.info("헌재결정례 수집이 성공적으로 완료되었습니다.")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 프로그램이 중단되었습니다.")
        return 130
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
