#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
체크포인트 관리 스크립트

법령용어 수집의 체크포인트를 관리하는 스크립트입니다.
- 체크포인트 목록 조회
- 체크포인트 삭제
- 체크포인트 정리
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from scripts.data_collection.law_open_api.utils.checkpoint_manager import CheckpointManager


def list_checkpoints():
    """체크포인트 목록 조회"""
    print("=" * 60)
    print("체크포인트 목록")
    print("=" * 60)
    
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()
    
    if not checkpoints:
        print("체크포인트가 없습니다.")
        return
    
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint['file']}")
        print(f"   데이터 타입: {checkpoint['data_type']}")
        print(f"   상태: {checkpoint['status']}")
        print(f"   시간: {checkpoint['timestamp']}")
        print()


def delete_checkpoint(data_type: str):
    """체크포인트 삭제"""
    print(f"체크포인트 삭제: {data_type}")
    
    checkpoint_manager = CheckpointManager()
    
    # 페이지 체크포인트 삭제
    checkpoint_manager.clear_page_checkpoint(data_type)
    
    # 수집 체크포인트 삭제
    checkpoint_manager.clear_collection_checkpoint(data_type)
    
    print(f"✅ {data_type} 체크포인트 삭제 완료")


def cleanup_checkpoints(days: int = 7):
    """오래된 체크포인트 정리"""
    print(f"오래된 체크포인트 정리 (보관일수: {days}일)")
    
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.cleanup_old_checkpoints(days)
    
    print("✅ 체크포인트 정리 완료")


def show_checkpoint_details(data_type: str):
    """체크포인트 상세 정보 조회"""
    print(f"체크포인트 상세 정보: {data_type}")
    print("=" * 60)
    
    checkpoint_manager = CheckpointManager()
    
    # 페이지 체크포인트
    page_checkpoint = checkpoint_manager.load_page_checkpoint(data_type)
    if page_checkpoint:
        print("📄 페이지 체크포인트:")
        print(f"  현재 페이지: {page_checkpoint.get('current_page', 'N/A')}")
        print(f"  전체 페이지: {page_checkpoint.get('total_pages', 'N/A')}")
        print(f"  수집된 항목: {page_checkpoint.get('collected_count', 'N/A'):,}개")
        print(f"  마지막 용어 ID: {page_checkpoint.get('last_term_id', 'N/A')}")
        print(f"  저장 시간: {page_checkpoint.get('timestamp', 'N/A')}")
    else:
        print("📄 페이지 체크포인트: 없음")
    
    print()
    
    # 수집 체크포인트
    collection_checkpoint = checkpoint_manager.load_collection_checkpoint(data_type)
    if collection_checkpoint:
        print("📚 수집 체크포인트:")
        collection_info = collection_checkpoint.get('collection_info', {})
        for key, value in collection_info.items():
            print(f"  {key}: {value}")
        print(f"  저장 시간: {collection_checkpoint.get('timestamp', 'N/A')}")
    else:
        print("📚 수집 체크포인트: 없음")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='체크포인트 관리')
    parser.add_argument('action', choices=['list', 'delete', 'cleanup', 'show'],
                       help='실행할 작업')
    parser.add_argument('--data-type', type=str, default='legal_terms',
                       help='데이터 타입 (기본값: legal_terms)')
    parser.add_argument('--days', type=int, default=7,
                       help='정리할 때 보관할 일수 (기본값: 7)')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'list':
            list_checkpoints()
        elif args.action == 'delete':
            delete_checkpoint(args.data_type)
        elif args.action == 'cleanup':
            cleanup_checkpoints(args.days)
        elif args.action == 'show':
            show_checkpoint_details(args.data_type)
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
