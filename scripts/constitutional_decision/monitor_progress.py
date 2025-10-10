#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 수집 진행 상황 실시간 모니터링 스크립트
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def monitor_progress():
    """실시간 진행 상황 모니터링"""
    output_dir = Path("data/raw/constitutional_decisions")
    
    if not output_dir.exists():
        print("❌ 수집 디렉토리가 존재하지 않습니다.")
        return
    
    print("🔍 헌재결정례 수집 진행 상황 실시간 모니터링")
    print("=" * 60)
    print("Ctrl+C를 눌러 종료하세요.")
    print("=" * 60)
    
    try:
        while True:
            # 화면 클리어
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"⏰ 현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 수집 디렉토리: {output_dir}")
            print("=" * 60)
            
            # 체크포인트 파일 확인
            checkpoint_files = list(output_dir.glob("collection_checkpoint_*.json"))
            
            if checkpoint_files:
                # 가장 최근 체크포인트 파일 읽기
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                try:
                    with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    stats = checkpoint_data.get('stats', {})
                    collected_count = stats.get('collected_count', 0)
                    target_count = stats.get('target_count', 1000)
                    keywords_processed = stats.get('keywords_processed', 0)
                    last_keyword = stats.get('last_keyword_processed', 'N/A')
                    api_requests = stats.get('api_requests_made', 0)
                    api_errors = stats.get('api_errors', 0)
                    status = stats.get('status', 'running')
                    
                    progress = (collected_count / target_count * 100) if target_count > 0 else 0
                    
                    print(f"📊 수집 진행률: {progress:.1f}% ({collected_count:,}/{target_count:,}건)")
                    print(f"🔍 처리된 키워드: {keywords_processed}개")
                    print(f"📝 마지막 처리 키워드: {last_keyword}")
                    print(f"🌐 API 요청 수: {api_requests:,}회")
                    print(f"❌ API 오류 수: {api_errors:,}회")
                    print(f"📈 상태: {status}")
                    
                    # 예상 완료 시간 계산
                    if collected_count > 0 and status == 'running':
                        remaining = target_count - collected_count
                        if remaining > 0:
                            # 간단한 예상 시간 계산 (API 요청 수 기반)
                            estimated_remaining_requests = remaining // 100 + 1
                            print(f"⏱️  예상 남은 API 요청: {estimated_remaining_requests:,}회")
                    
                    if status == 'completed':
                        print("🎉 수집이 완료되었습니다!")
                        break
                    elif status == 'interrupted':
                        print("⚠️ 수집이 중단되었습니다.")
                        break
                        
                except Exception as e:
                    print(f"❌ 체크포인트 파일 읽기 오류: {e}")
            else:
                # 배치 파일로 진행 상황 추정
                batch_files = list(output_dir.glob("batch_*.json"))
                if batch_files:
                    total_count = 0
                    for file_path in batch_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                count = data.get('metadata', {}).get('count', 0)
                                total_count += count
                        except:
                            pass
                    
                    print(f"📁 수집된 배치 파일: {len(batch_files)}개")
                    print(f"📊 추정 수집 건수: {total_count:,}건")
                else:
                    print("❌ 수집 데이터가 없습니다.")
            
            print("=" * 60)
            print("5초 후 새로고침... (Ctrl+C로 종료)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 모니터링을 종료합니다.")
    except Exception as e:
        print(f"❌ 모니터링 중 오류 발생: {e}")

if __name__ == "__main__":
    monitor_progress()
