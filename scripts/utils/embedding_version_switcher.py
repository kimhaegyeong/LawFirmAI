#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
임베딩 버전 전환 및 관리 스크립트

버전 간 전환, 통계 조회, 비교 등의 기능 제공
"""
import argparse
import sqlite3
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 sys.path에 추가
_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _CURRENT_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.embedding_version_manager import EmbeddingVersionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_versions(db_path: str, chunking_strategy: Optional[str] = None):
    """버전 목록 조회 및 표시"""
    manager = EmbeddingVersionManager(db_path)
    versions = manager.list_versions(chunking_strategy)
    
    if not versions:
        print("등록된 버전이 없습니다.")
        return
    
    # 테이블 형식으로 표시
    print("\n임베딩 버전 목록:")
    print("=" * 120)
    print(f"{'ID':<5} {'버전명':<30} {'청킹 전략':<15} {'모델명':<40} {'활성':<5} {'생성일':<20} {'설명':<30}")
    print("-" * 120)
    
    for v in versions:
        active = "✓" if v['is_active'] else ""
        desc = (v.get('description', '')[:30] if v.get('description') else '')
        print(f"{v['id']:<5} {v['version_name']:<30} {v['chunking_strategy']:<15} {v['model_name']:<40} {active:<5} {v['created_at']:<20} {desc:<30}")
    
    print()


def show_version_stats(db_path: str, version_id: int):
    """버전 통계 정보 표시"""
    manager = EmbeddingVersionManager(db_path)
    stats = manager.get_version_statistics(version_id)
    
    if not stats:
        print(f"버전 ID {version_id}를 찾을 수 없습니다.")
        return
    
    print(f"\n버전 통계: {stats['version_name']} (ID: {version_id})")
    print("=" * 80)
    print(f"청킹 전략: {stats['chunking_strategy']}")
    print(f"모델명: {stats['model_name']}")
    print(f"활성 상태: {'활성' if stats['is_active'] else '비활성'}")
    print(f"생성일: {stats['created_at']}")
    print(f"\n통계:")
    print(f"  - 청크 수: {stats['chunk_count']:,}개")
    print(f"  - 임베딩 수: {stats['embedding_count']:,}개")
    print(f"  - 문서 수: {stats['document_count']:,}개")
    print(f"\n소스 타입별 분포:")
    for source_type, count in stats['source_type_distribution'].items():
        print(f"  - {source_type}: {count:,}개")
    print()


def compare_versions(db_path: str, version_id1: int, version_id2: int):
    """두 버전 비교"""
    manager = EmbeddingVersionManager(db_path)
    comparison = manager.compare_versions(version_id1, version_id2)
    
    if not comparison['version1'] or not comparison['version2']:
        print("비교할 버전을 찾을 수 없습니다.")
        return
    
    v1 = comparison['version1']
    v2 = comparison['version2']
    diff = comparison['differences']
    
    print(f"\n버전 비교:")
    print("=" * 80)
    print(f"버전 1: {v1['version_name']} (ID: {version_id1})")
    print(f"버전 2: {v2['version_name']} (ID: {version_id2})")
    print()
    print(f"{'항목':<20} {'버전 1':<15} {'버전 2':<15} {'차이':<15}")
    print("-" * 80)
    print(f"{'청크 수':<20} {v1['chunk_count']:<15,} {v2['chunk_count']:<15,} {diff['chunk_count_diff']:+,}")
    print(f"{'임베딩 수':<20} {v1['embedding_count']:<15,} {v2['embedding_count']:<15,} {diff['embedding_count_diff']:+,}")
    print(f"{'문서 수':<20} {v1['document_count']:<15,} {v2['document_count']:<15,} {diff['document_count_diff']:+,}")
    print()


def switch_version(db_path: str, version_id: int, confirm: bool = True):
    """활성 버전 전환"""
    manager = EmbeddingVersionManager(db_path)
    
    # 버전 정보 조회
    versions = manager.list_versions()
    target_version = next((v for v in versions if v['id'] == version_id), None)
    
    if not target_version:
        print(f"버전 ID {version_id}를 찾을 수 없습니다.")
        return False
    
    # 현재 활성 버전 확인
    current_active = manager.get_active_version(target_version['chunking_strategy'])
    
    if current_active and current_active['id'] == version_id:
        print(f"버전 {version_id} ({target_version['version_name']})는 이미 활성 상태입니다.")
        return True
    
    # 확인
    if confirm:
        if current_active:
            print(f"현재 활성 버전: {current_active['id']} ({current_active['version_name']})")
        print(f"전환할 버전: {version_id} ({target_version['version_name']})")
        response = input("\n정말 전환하시겠습니까? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("전환 취소됨.")
            return False
    
    # 버전 전환
    success = manager.set_active_version(version_id)
    
    if success:
        print(f"✓ 버전 {version_id} ({target_version['version_name']})로 전환되었습니다.")
        return True
    else:
        print(f"✗ 버전 전환 실패.")
        return False


def delete_version(db_path: str, version_id: int, delete_chunks: bool = False, confirm: bool = True):
    """버전 삭제"""
    manager = EmbeddingVersionManager(db_path)
    
    # 버전 정보 조회
    versions = manager.list_versions()
    target_version = next((v for v in versions if v['id'] == version_id), None)
    
    if not target_version:
        print(f"버전 ID {version_id}를 찾을 수 없습니다.")
        return False
    
    # 통계 조회
    stats = manager.get_version_statistics(version_id)
    
    # 확인
    if confirm:
        print(f"\n삭제할 버전: {target_version['version_name']} (ID: {version_id})")
        print(f"청크 수: {stats.get('chunk_count', 0):,}개")
        print(f"임베딩 수: {stats.get('embedding_count', 0):,}개")
        if delete_chunks:
            print("⚠️  청크 및 임베딩도 함께 삭제됩니다!")
        response = input("\n정말 삭제하시겠습니까? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("삭제 취소됨.")
            return False
    
    # 버전 삭제
    try:
        deleted_chunks, deleted_embeddings = manager.delete_version(version_id, delete_chunks=delete_chunks)
        print(f"✓ 버전 {version_id} 삭제 완료: {deleted_chunks}개 청크, {deleted_embeddings}개 임베딩")
        return True
    except Exception as e:
        print(f"✗ 버전 삭제 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='임베딩 버전 관리 및 전환')
    parser.add_argument('--db', default='data/lawfirm_v2.db', help='데이터베이스 경로')
    parser.add_argument('--action', choices=['list', 'stats', 'compare', 'switch', 'delete'],
                       required=True, help='수행할 작업')
    parser.add_argument('--version-id', type=int, help='버전 ID')
    parser.add_argument('--version-id2', type=int, help='비교할 두 번째 버전 ID (compare 시)')
    parser.add_argument('--chunking-strategy', help='청킹 전략 필터 (list 시)')
    parser.add_argument('--delete-chunks', action='store_true', help='버전 삭제 시 청크 및 임베딩도 함께 삭제')
    parser.add_argument('--no-confirm', action='store_true', help='확인 없이 실행')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_versions(args.db, args.chunking_strategy)
    
    elif args.action == 'stats':
        if not args.version_id:
            print("--version-id가 필요합니다.")
            return
        show_version_stats(args.db, args.version_id)
    
    elif args.action == 'compare':
        if not args.version_id or not args.version_id2:
            print("--version-id와 --version-id2가 필요합니다.")
            return
        compare_versions(args.db, args.version_id, args.version_id2)
    
    elif args.action == 'switch':
        if not args.version_id:
            print("--version-id가 필요합니다.")
            return
        switch_version(args.db, args.version_id, confirm=not args.no_confirm)
    
    elif args.action == 'delete':
        if not args.version_id:
            print("--version-id가 필요합니다.")
            return
        delete_version(args.db, args.version_id, delete_chunks=args.delete_chunks, confirm=not args.no_confirm)


if __name__ == '__main__':
    main()

