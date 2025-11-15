"""
FAISS 버전 관리 CLI 도구

FAISS 버전 목록 조회, 전환, 비교, 삭제, 백업/복원 기능을 제공합니다.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from faiss_version_manager import FAISSVersionManager
from faiss_backup_manager import FAISSBackupManager
from version_performance_monitor import VersionPerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_versions(vector_store_base: str):
    """버전 목록 조회"""
    manager = FAISSVersionManager(vector_store_base)
    versions = manager.list_versions()
    active_version = manager.get_active_version()
    
    print("\nFAISS Versions:")
    print("=" * 80)
    for version in versions:
        version_name = version.get('version', 'unknown')
        is_active = version.get('is_active', False)
        status = "ACTIVE" if is_active else "inactive"
        created_at = version.get('created_at', 'unknown')
        total_chunks = version.get('total_chunks', 0)
        
        print(f"{version_name:<40} {status:<10} {created_at[:19]:<20} {total_chunks:>10} chunks")
    
    if not versions:
        print("No versions found")
    
    print("=" * 80)


def show_version_stats(vector_store_base: str, version_name: str):
    """버전 통계 조회"""
    manager = FAISSVersionManager(vector_store_base)
    info = manager.get_version_info(version_name)
    
    if not info:
        print(f"Version {version_name} not found")
        return
    
    print(f"\nVersion: {version_name}")
    print("=" * 80)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("=" * 80)


def switch_version(vector_store_base: str, version_name: str):
    """버전 전환"""
    manager = FAISSVersionManager(vector_store_base)
    
    if manager.set_active_version(version_name):
        print(f"Switched to version: {version_name}")
    else:
        print(f"Failed to switch to version: {version_name}")
        sys.exit(1)


def compare_versions(vector_store_base: str, version1: str, version2: str):
    """버전 비교"""
    manager = FAISSVersionManager(vector_store_base)
    
    info1 = manager.get_version_info(version1)
    info2 = manager.get_version_info(version2)
    
    if not info1 or not info2:
        print("One or both versions not found")
        return
    
    print(f"\nComparing {version1} vs {version2}")
    print("=" * 80)
    print(f"Version 1: {version1}")
    print(f"  Chunks: {info1.get('total_chunks', 0)}")
    print(f"  Documents: {info1.get('document_count', 0)}")
    print(f"  Created: {info1.get('created_at', 'unknown')}")
    print(f"\nVersion 2: {version2}")
    print(f"  Chunks: {info2.get('total_chunks', 0)}")
    print(f"  Documents: {info2.get('document_count', 0)}")
    print(f"  Created: {info2.get('created_at', 'unknown')}")
    print("=" * 80)


def delete_version(vector_store_base: str, version_name: str, force: bool = False):
    """버전 삭제"""
    manager = FAISSVersionManager(vector_store_base)
    
    if manager.delete_version(version_name, force=force):
        print(f"Deleted version: {version_name}")
    else:
        print(f"Failed to delete version: {version_name}")
        sys.exit(1)


def backup_version(vector_store_base: str, backup_path: str, version_name: str):
    """버전 백업"""
    manager = FAISSVersionManager(vector_store_base)
    backup_manager = FAISSBackupManager(backup_path, manager)
    
    backup_file = backup_manager.backup_version(version_name)
    if backup_file:
        print(f"Backup created: {backup_file}")
    else:
        print(f"Failed to backup version: {version_name}")
        sys.exit(1)


def show_performance(vector_store_base: str, version_name: Optional[str] = None):
    """성능 통계 조회"""
    monitor = VersionPerformanceMonitor()
    
    if version_name:
        metrics = monitor.get_version_metrics(version_name)
        if metrics:
            print(f"\nPerformance metrics for {version_name}:")
            print("=" * 80)
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            print("=" * 80)
        else:
            print(f"No metrics found for version: {version_name}")
    else:
        versions = monitor.list_versions()
        print("\nMonitored versions:")
        print("=" * 80)
        for version in versions:
            metrics = monitor.get_version_metrics(version)
            if metrics:
                print(f"{version}:")
                print(f"  Total queries: {metrics.get('total_queries', 0)}")
                print(f"  Avg latency: {metrics.get('avg_latency', 0):.2f} ms")
                print(f"  Avg relevance: {metrics.get('avg_relevance', 0):.2f}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="FAISS Version Manager CLI")
    parser.add_argument("--vector-store-base", default="data/vector_store", help="Vector store base path")
    parser.add_argument("--backup-path", default="data/backups/faiss_versions", help="Backup path")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    list_parser = subparsers.add_parser("list", help="List all versions")
    
    stats_parser = subparsers.add_parser("stats", help="Show version statistics")
    stats_parser.add_argument("version", help="Version name")
    
    switch_parser = subparsers.add_parser("switch", help="Switch active version")
    switch_parser.add_argument("version", help="Version name")
    
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("version1", help="First version name")
    compare_parser.add_argument("version2", help="Second version name")
    
    delete_parser = subparsers.add_parser("delete", help="Delete version")
    delete_parser.add_argument("version", help="Version name")
    delete_parser.add_argument("--force", action="store_true", help="Force delete active version")
    
    backup_parser = subparsers.add_parser("backup", help="Backup version")
    backup_parser.add_argument("version", help="Version name")
    
    perf_parser = subparsers.add_parser("performance", help="Show performance metrics")
    perf_parser.add_argument("--version", help="Specific version (None for all)")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_versions(args.vector_store_base)
    elif args.command == "stats":
        show_version_stats(args.vector_store_base, args.version)
    elif args.command == "switch":
        switch_version(args.vector_store_base, args.version)
    elif args.command == "compare":
        compare_versions(args.vector_store_base, args.version1, args.version2)
    elif args.command == "delete":
        delete_version(args.vector_store_base, args.version, args.force)
    elif args.command == "backup":
        backup_version(args.vector_store_base, args.backup_path, args.version)
    elif args.command == "performance":
        show_performance(args.vector_store_base, args.version)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

