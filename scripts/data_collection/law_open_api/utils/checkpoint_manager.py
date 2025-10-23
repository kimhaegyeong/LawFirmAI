#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
체크포인트 관리 유틸리티

수집 작업의 중단점을 관리하고 재시작을 지원하는 모듈입니다.
- 페이지별 체크포인트 저장
- 중단 후 재시작 지원
- 진행 상황 추적
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """체크포인트 관리자"""
    
    def __init__(self, checkpoint_dir: str = "data/raw/law_open_api/checkpoints"):
        """
        체크포인트 관리자 초기화
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager 초기화 완료 - 디렉토리: {self.checkpoint_dir}")
    
    def save_page_checkpoint(self, data_type: str, page: int, total_pages: int, 
                            collected_count: int, last_term_id: str = None) -> None:
        """
        페이지 체크포인트 저장
        
        Args:
            data_type: 데이터 타입
            page: 현재 페이지
            total_pages: 전체 페이지 수
            collected_count: 수집된 항목 수
            last_term_id: 마지막 수집된 용어 ID
        """
        checkpoint_data = {
            "data_type": data_type,
            "current_page": page,
            "total_pages": total_pages,
            "collected_count": collected_count,
            "last_term_id": last_term_id,
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress"
        }
        
        checkpoint_file = self.checkpoint_dir / f"{data_type}_page_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"페이지 체크포인트 저장: 페이지 {page}/{total_pages}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
    
    def load_page_checkpoint(self, data_type: str) -> Optional[Dict[str, Any]]:
        """
        페이지 체크포인트 로드
        
        Args:
            data_type: 데이터 타입
            
        Returns:
            체크포인트 데이터 또는 None
        """
        checkpoint_file = self.checkpoint_dir / f"{data_type}_page_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"체크포인트 로드: 페이지 {checkpoint_data.get('current_page', 0)}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            return None
    
    def clear_page_checkpoint(self, data_type: str) -> None:
        """
        페이지 체크포인트 삭제
        
        Args:
            data_type: 데이터 타입
        """
        checkpoint_file = self.checkpoint_dir / f"{data_type}_page_checkpoint.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"체크포인트 삭제: {data_type}")
        except Exception as e:
            logger.error(f"체크포인트 삭제 실패: {e}")
    
    def save_collection_checkpoint(self, data_type: str, collection_info: Dict[str, Any]) -> None:
        """
        수집 작업 체크포인트 저장
        
        Args:
            data_type: 데이터 타입
            collection_info: 수집 정보
        """
        checkpoint_data = {
            "data_type": data_type,
            "collection_info": collection_info,
            "timestamp": datetime.now().isoformat(),
            "status": "collection_in_progress"
        }
        
        checkpoint_file = self.checkpoint_dir / f"{data_type}_collection_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"수집 체크포인트 저장: {data_type}")
            
        except Exception as e:
            logger.error(f"수집 체크포인트 저장 실패: {e}")
    
    def load_collection_checkpoint(self, data_type: str) -> Optional[Dict[str, Any]]:
        """
        수집 작업 체크포인트 로드
        
        Args:
            data_type: 데이터 타입
            
        Returns:
            체크포인트 데이터 또는 None
        """
        checkpoint_file = self.checkpoint_dir / f"{data_type}_collection_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"수집 체크포인트 로드: {data_type}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"수집 체크포인트 로드 실패: {e}")
            return None
    
    def clear_collection_checkpoint(self, data_type: str) -> None:
        """
        수집 작업 체크포인트 삭제
        
        Args:
            data_type: 데이터 타입
        """
        checkpoint_file = self.checkpoint_dir / f"{data_type}_collection_checkpoint.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"수집 체크포인트 삭제: {data_type}")
        except Exception as e:
            logger.error(f"수집 체크포인트 삭제 실패: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        모든 체크포인트 목록 조회
        
        Returns:
            체크포인트 목록
        """
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    checkpoints.append({
                        "file": checkpoint_file.name,
                        "data_type": data.get("data_type", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "status": data.get("status", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"체크포인트 파일 읽기 실패: {checkpoint_file} - {e}")
            
            return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"체크포인트 목록 조회 실패: {e}")
            return []
    
    def cleanup_old_checkpoints(self, days: int = 7) -> None:
        """
        오래된 체크포인트 정리
        
        Args:
            days: 보관할 일수
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    logger.info(f"오래된 체크포인트 삭제: {checkpoint_file.name}")
                    
        except Exception as e:
            logger.error(f"체크포인트 정리 실패: {e}")


def create_checkpoint_manager(checkpoint_dir: str = None) -> CheckpointManager:
    """
    체크포인트 관리자 생성
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리
        
    Returns:
        체크포인트 관리자 인스턴스
    """
    return CheckpointManager(checkpoint_dir)
