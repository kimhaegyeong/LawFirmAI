#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터스토어 버전 스위칭 스크립트

특정 버전으로 롤백하거나 활성 버전을 변경합니다.
"""

import logging
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

logger = logging.getLogger(__name__)


def list_versions(base_path: Path):
    """버전 목록 출력"""
    version_manager = VectorStoreVersionManager(base_path)
    versions = version_manager.list_versions()
    current_version = version_manager.get_current_version()
    
    if not versions:
        logger.info("No versions found.")
        return
    
    logger.info("Available versions:")
    logger.info("-" * 80)
    for v in versions:
        version_str = v["version"]
        is_current = " (CURRENT)" if version_str == current_version else ""
        logger.info(f"  {version_str}{is_current}")
        logger.info(f"    Created: {v.get('created_at', 'N/A')}")
        logger.info(f"    Model: {v.get('model_name', 'N/A')}")
        logger.info(f"    Documents: {v.get('document_count', 'N/A')}")
        if v.get('changes'):
            logger.info(f"    Changes: {', '.join(v['changes'][:2])}")
        logger.info("")


def switch_version(base_path: Path, version: str) -> bool:
    """
    활성 버전 변경
    
    Args:
        base_path: 벡터스토어 기본 경로
        version: 변경할 버전 번호
    
    Returns:
        bool: 성공 여부
    """
    version_manager = VectorStoreVersionManager(base_path)
    
    version_info = version_manager.get_version_info(version)
    if not version_info:
        logger.error(f"Version {version} not found")
        return False
    
    version_path = version_manager.get_version_path(version)
    if not version_path.exists():
        logger.error(f"Version path does not exist: {version_path}")
        return False
    
    if version_manager.set_current_version(version):
        logger.info(f"Successfully switched to version {version}")
        logger.info(f"Version path: {version_path}")
        logger.info(f"Model: {version_info.get('model_name', 'N/A')}")
        logger.info(f"Documents: {version_info.get('document_count', 'N/A')}")
        return True
    else:
        logger.error(f"Failed to switch to version {version}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="벡터스토어 버전 스위칭")
    parser.add_argument('base_path', type=str,
                        help='벡터스토어 기본 경로 (예: data/embeddings/ml_enhanced_ko_sroberta_precedents)')
    parser.add_argument('--version', type=str,
                        help='변경할 버전 번호 (예: v2.0.0). 지정하지 않으면 버전 목록만 출력')
    parser.add_argument('--list', '-l', action='store_true',
                        help='버전 목록만 출력')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    base_path = Path(args.base_path)
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        return False
    
    if args.list or not args.version:
        list_versions(base_path)
        return True
    
    return switch_version(base_path, args.version)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

