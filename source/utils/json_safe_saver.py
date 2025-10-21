# -*- coding: utf-8 -*-
"""
JSON Safe Saver
안전한 JSON 저장을 위한 유틸리티 모듈
"""

import json
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONSafeSaver:
    """안전한 JSON 저장을 위한 클래스"""
    
    def __init__(self, backup_dir: str = "data/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, file_path: str, data: Any, max_retries: int = 3) -> bool:
        """
        안전한 JSON 저장
        
        Args:
            file_path: 저장할 파일 경로
            data: 저장할 데이터
            max_retries: 최대 재시도 횟수
            
        Returns:
            bool: 저장 성공 여부
        """
        file_path = Path(file_path)
        
        # 디렉토리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(max_retries):
            try:
                # 1차 시도: 직접 저장
                if self._save_direct(file_path, data):
                    logger.info(f"Successfully saved JSON to {file_path}")
                    return True
                
            except Exception as e:
                logger.warning(f"Direct save attempt {attempt + 1} failed: {e}")
                
                # 2차 시도: 임시 파일 사용
                try:
                    if self._save_via_temp(file_path, data):
                        logger.info(f"Successfully saved JSON via temp file to {file_path}")
                        return True
                except Exception as temp_error:
                    logger.warning(f"Temp save attempt {attempt + 1} failed: {temp_error}")
                    
                    # 3차 시도: 백업 후 새로 생성
                    try:
                        if self._save_via_backup(file_path, data):
                            logger.info(f"Successfully saved JSON via backup to {file_path}")
                            return True
                    except Exception as backup_error:
                        logger.warning(f"Backup save attempt {attempt + 1} failed: {backup_error}")
        
        logger.error(f"Failed to save JSON to {file_path} after {max_retries} attempts")
        return False
    
    def _save_direct(self, file_path: Path, data: Any) -> bool:
        """직접 저장 시도"""
        try:
            # 기존 파일이 있으면 백업
            if file_path.exists():
                self._backup_file(file_path)
            
            # 새 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 파일 검증
            self._validate_json_file(file_path)
            return True
            
        except Exception as e:
            logger.warning(f"Direct save failed: {e}")
            return False
    
    def _save_via_temp(self, file_path: Path, data: Any) -> bool:
        """임시 파일을 통한 저장 시도"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.json') as temp_file:
                json.dump(data, temp_file, ensure_ascii=False, indent=2)
                temp_path = Path(temp_file.name)
            
            # 파일 검증
            self._validate_json_file(temp_path)
            
            # 기존 파일이 있으면 백업
            if file_path.exists():
                self._backup_file(file_path)
            
            # 임시 파일을 최종 위치로 이동
            shutil.move(str(temp_path), str(file_path))
            return True
            
        except Exception as e:
            logger.warning(f"Temp save failed: {e}")
            # 임시 파일 정리
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            return False
    
    def _save_via_backup(self, file_path: Path, data: Any) -> bool:
        """백업을 통한 저장 시도"""
        try:
            # 기존 파일 백업
            if file_path.exists():
                self._backup_file(file_path)
                file_path.unlink()  # 기존 파일 삭제
            
            # 새 파일 생성
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 파일 검증
            self._validate_json_file(file_path)
            return True
            
        except Exception as e:
            logger.warning(f"Backup save failed: {e}")
            return False
    
    def _backup_file(self, file_path: Path) -> None:
        """파일 백업"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up {file_path} to {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to backup {file_path}: {e}")
    
    def _validate_json_file(self, file_path: Path) -> None:
        """JSON 파일 검증"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid JSON file {file_path}: {e}")
    
    def load_json(self, file_path: str, default: Any = None) -> Any:
        """
        안전한 JSON 로드
        
        Args:
            file_path: 로드할 파일 경로
            default: 로드 실패 시 기본값
            
        Returns:
            Any: 로드된 데이터 또는 기본값
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"JSON file {file_path} does not exist")
            return default
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load JSON from {file_path}: {e}")
            
            # 백업 파일에서 복구 시도
            try:
                return self._recover_from_backup(file_path)
            except Exception as recovery_error:
                logger.error(f"Failed to recover from backup: {recovery_error}")
                return default
        
        except Exception as e:
            logger.error(f"Unexpected error loading JSON from {file_path}: {e}")
            return default
    
    def _recover_from_backup(self, file_path: Path) -> Any:
        """백업 파일에서 복구 시도"""
        try:
            # 가장 최근 백업 파일 찾기
            backup_files = list(self.backup_dir.glob(f"{file_path.stem}_*{file_path.suffix}"))
            if not backup_files:
                raise FileNotFoundError("No backup files found")
            
            # 파일명에서 타임스탬프 추출하여 정렬
            backup_files.sort(key=lambda x: x.stem.split('_')[-2:], reverse=True)
            latest_backup = backup_files[0]
            
            logger.info(f"Attempting to recover from backup: {latest_backup}")
            
            with open(latest_backup, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 복구된 데이터로 원본 파일 재생성
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully recovered from backup: {latest_backup}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to recover from backup: {e}")
            raise


# 전역 인스턴스
_json_safe_saver = None

def get_json_safe_saver() -> JSONSafeSaver:
    """JSONSafeSaver 싱글톤 인스턴스 반환"""
    global _json_safe_saver
    if _json_safe_saver is None:
        _json_safe_saver = JSONSafeSaver()
    return _json_safe_saver

def safe_save_json(file_path: str, data: Any) -> bool:
    """안전한 JSON 저장 (편의 함수)"""
    return get_json_safe_saver().save_json(file_path, data)

def safe_load_json(file_path: str, default: Any = None) -> Any:
    """안전한 JSON 로드 (편의 함수)"""
    return get_json_safe_saver().load_json(file_path, default)

