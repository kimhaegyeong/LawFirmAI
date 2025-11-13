#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터스토어 버전 관리 시스템

벡터스토어의 버전을 생성, 조회, 관리하는 기능을 제공합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class VectorStoreVersionManager:
    """벡터스토어 버전 관리 클래스"""
    
    def __init__(self, base_path: Path):
        """
        버전 관리자 초기화
        
        Args:
            base_path: 벡터스토어 기본 경로 (예: data/embeddings/ml_enhanced_ko_sroberta_precedents)
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.version_file = self.base_path / "versions.json"
        self._ensure_version_file()
    
    def _ensure_version_file(self):
        """versions.json 파일이 없으면 생성"""
        if not self.version_file.exists():
            self._write_versions({
                "current_version": None,
                "versions": []
            })
    
    def _read_versions(self) -> Dict:
        """versions.json 파일 읽기"""
        try:
            with open(self.version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read versions.json: {e}. Creating new file.")
            return {
                "current_version": None,
                "versions": []
            }
    
    def _write_versions(self, data: Dict):
        """versions.json 파일 쓰기"""
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write versions.json: {e}")
            raise
    
    def create_version(self, version: str, metadata: Dict) -> bool:
        """
        새 버전 생성
        
        Args:
            version: 버전 번호 (예: "v2.0.0")
            metadata: 버전 메타데이터
                - model_name: 모델명
                - dimension: 벡터 차원
                - index_type: 인덱스 타입
                - document_count: 문서 수
                - metadata_schema_version: 메타데이터 스키마 버전
                - changes: 변경 사항 리스트
        
        Returns:
            bool: 성공 여부
        """
        if not self._validate_version_format(version):
            logger.error(f"Invalid version format: {version}. Expected format: vX.Y.Z")
            return False
        
        versions_data = self._read_versions()
        
        # 중복 버전 확인
        for v in versions_data["versions"]:
            if v["version"] == version:
                logger.warning(f"Version {version} already exists. Use update_version to modify.")
                return False
        
        version_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            **metadata
        }
        
        versions_data["versions"].append(version_info)
        
        # 첫 버전이면 자동으로 현재 버전으로 설정
        if versions_data["current_version"] is None:
            versions_data["current_version"] = version
        
        self._write_versions(versions_data)
        logger.info(f"Created version {version}")
        return True
    
    def update_version(self, version: str, metadata: Dict) -> bool:
        """
        기존 버전 정보 업데이트
        
        Args:
            version: 버전 번호
            metadata: 업데이트할 메타데이터
        
        Returns:
            bool: 성공 여부
        """
        versions_data = self._read_versions()
        
        for v in versions_data["versions"]:
            if v["version"] == version:
                v.update(metadata)
                v["updated_at"] = datetime.now().isoformat()
                self._write_versions(versions_data)
                logger.info(f"Updated version {version}")
                return True
        
        logger.warning(f"Version {version} not found")
        return False
    
    def get_current_version(self) -> Optional[str]:
        """
        현재 활성 버전 조회
        
        Returns:
            Optional[str]: 현재 버전 번호, 없으면 None
        """
        versions_data = self._read_versions()
        return versions_data.get("current_version")
    
    def set_current_version(self, version: str) -> bool:
        """
        활성 버전 설정
        
        Args:
            version: 버전 번호
        
        Returns:
            bool: 성공 여부
        """
        if not self._validate_version_format(version):
            logger.error(f"Invalid version format: {version}")
            return False
        
        versions_data = self._read_versions()
        
        # 버전 존재 확인
        version_exists = any(v["version"] == version for v in versions_data["versions"])
        if not version_exists:
            logger.error(f"Version {version} does not exist")
            return False
        
        versions_data["current_version"] = version
        self._write_versions(versions_data)
        logger.info(f"Set current version to {version}")
        return True
    
    def list_versions(self) -> List[Dict]:
        """
        모든 버전 목록 조회
        
        Returns:
            List[Dict]: 버전 정보 리스트
        """
        versions_data = self._read_versions()
        return versions_data.get("versions", [])
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """
        특정 버전 정보 조회
        
        Args:
            version: 버전 번호
        
        Returns:
            Optional[Dict]: 버전 정보, 없으면 None
        """
        versions_data = self._read_versions()
        for v in versions_data["versions"]:
            if v["version"] == version:
                return v
        return None
    
    def get_latest_version(self) -> Optional[str]:
        """
        최신 버전 번호 조회 (버전 번호 기준)
        
        Returns:
            Optional[str]: 최신 버전 번호, 없으면 None
        """
        versions = self.list_versions()
        if not versions:
            return None
        
        # 버전 번호를 파싱하여 정렬
        def version_key(v: Dict) -> tuple:
            version_str = v["version"].lstrip("v")
            parts = version_str.split(".")
            return tuple(int(p) for p in parts)
        
        sorted_versions = sorted(versions, key=version_key, reverse=True)
        return sorted_versions[0]["version"]
    
    def get_version_path(self, version: Optional[str] = None) -> Path:
        """
        버전별 인덱스 경로 조회
        
        Args:
            version: 버전 번호. None이면 현재 버전 사용
        
        Returns:
            Path: 버전별 인덱스 경로
        """
        if version is None:
            version = self.get_current_version()
            if version is None:
                # 버전이 없으면 기본 경로 반환
                return self.base_path
        
        return self.base_path / version
    
    def _validate_version_format(self, version: str) -> bool:
        """
        버전 형식 검증 (semantic versioning: vX.Y.Z)
        
        Args:
            version: 버전 번호
        
        Returns:
            bool: 유효 여부
        """
        pattern = r'^v\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
    
    def delete_version(self, version: str) -> bool:
        """
        버전 삭제 (주의: 인덱스 파일은 삭제하지 않음)
        
        Args:
            version: 버전 번호
        
        Returns:
            bool: 성공 여부
        """
        versions_data = self._read_versions()
        
        # 현재 버전이면 삭제 불가
        if versions_data.get("current_version") == version:
            logger.error(f"Cannot delete current version {version}. Switch to another version first.")
            return False
        
        # 버전 목록에서 제거
        versions_data["versions"] = [
            v for v in versions_data["versions"] if v["version"] != version
        ]
        
        self._write_versions(versions_data)
        logger.info(f"Deleted version {version} from registry")
        return True

