#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수집 타임스탬프 관리 유틸리티

법령용어 수집의 타임스탬프를 관리하고 추적하는 모듈입니다.
- 마지막 수집 시간 저장/조회
- 수집 횟수 추적
- 수집 이력 관리
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TimestampManager:
    """수집 타임스탬프 관리"""
    
    def __init__(self, metadata_dir: str = "data/raw/law_open_api/metadata"):
        """
        타임스탬프 매니저 초기화
        
        Args:
            metadata_dir: 메타데이터 저장 디렉토리
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp_file = self.metadata_dir / "collection_timestamps.json"
        self.timestamps = self._load_timestamps()
        
        logger.info(f"TimestampManager 초기화 완료 - 메타데이터 디렉토리: {self.metadata_dir}")
    
    def _load_timestamps(self) -> Dict[str, Any]:
        """타임스탬프 데이터 로드"""
        if self.timestamp_file.exists():
            try:
                with open(self.timestamp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"타임스탬프 데이터 로드 완료: {len(data)}개 항목")
                    return data
            except Exception as e:
                logger.error(f"타임스탬프 데이터 로드 실패: {e}")
                return {}
        else:
            logger.info("타임스탬프 파일이 존재하지 않음 - 새로 생성")
            return {}
    
    def _save_timestamps(self):
        """타임스탬프 데이터 저장"""
        try:
            with open(self.timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(self.timestamps, f, ensure_ascii=False, indent=2)
            logger.debug("타임스탬프 데이터 저장 완료")
        except Exception as e:
            logger.error(f"타임스탬프 데이터 저장 실패: {e}")
    
    def get_last_collection_time(self, data_type: str) -> Optional[datetime]:
        """
        마지막 수집 시간 조회
        
        Args:
            data_type: 데이터 타입 (예: "legal_terms")
            
        Returns:
            마지막 수집 시간 (datetime 객체) 또는 None
        """
        timestamp_str = self.timestamps.get(data_type, {}).get("last_collection")
        
        if timestamp_str:
            try:
                return datetime.fromisoformat(timestamp_str)
            except ValueError as e:
                logger.error(f"타임스탬프 파싱 실패 ({data_type}): {e}")
                return None
        
        logger.debug(f"마지막 수집 시간 없음: {data_type}")
        return None
    
    def update_collection_time(self, data_type: str, success: bool = True):
        """
        수집 시간 업데이트
        
        Args:
            data_type: 데이터 타입
            success: 수집 성공 여부
        """
        if data_type not in self.timestamps:
            self.timestamps[data_type] = {
                "collection_count": 0,
                "success_count": 0,
                "error_count": 0,
                "first_collection": None,
                "last_collection": None,
                "last_successful_collection": None
            }
        
        now = datetime.now()
        data_info = self.timestamps[data_type]
        
        # 기본 정보 업데이트
        data_info["collection_count"] = data_info.get("collection_count", 0) + 1
        data_info["last_collection"] = now.isoformat()
        
        if not data_info.get("first_collection"):
            data_info["first_collection"] = now.isoformat()
        
        # 성공/실패 카운트 업데이트
        if success:
            data_info["success_count"] = data_info.get("success_count", 0) + 1
            data_info["last_successful_collection"] = now.isoformat()
        else:
            data_info["error_count"] = data_info.get("error_count", 0) + 1
        
        # 저장
        self._save_timestamps()
        
        logger.info(f"수집 시간 업데이트 완료: {data_type} (성공: {success})")
    
    def get_collection_stats(self, data_type: str) -> Dict[str, Any]:
        """
        수집 통계 조회
        
        Args:
            data_type: 데이터 타입
            
        Returns:
            수집 통계 정보
        """
        if data_type not in self.timestamps:
            return {
                "collection_count": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "first_collection": None,
                "last_collection": None,
                "last_successful_collection": None
            }
        
        data_info = self.timestamps[data_type]
        collection_count = data_info.get("collection_count", 0)
        success_count = data_info.get("success_count", 0)
        
        success_rate = (success_count / collection_count * 100) if collection_count > 0 else 0.0
        
        return {
            "collection_count": collection_count,
            "success_count": success_count,
            "error_count": data_info.get("error_count", 0),
            "success_rate": round(success_rate, 2),
            "first_collection": data_info.get("first_collection"),
            "last_collection": data_info.get("last_collection"),
            "last_successful_collection": data_info.get("last_successful_collection")
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        모든 데이터 타입의 수집 통계 조회
        
        Returns:
            모든 데이터 타입의 통계 정보
        """
        all_stats = {}
        
        for data_type in self.timestamps.keys():
            all_stats[data_type] = self.get_collection_stats(data_type)
        
        return all_stats
    
    def clear_timestamps(self, data_type: str = None):
        """
        타임스탬프 데이터 삭제
        
        Args:
            data_type: 삭제할 데이터 타입 (None이면 전체 삭제)
        """
        if data_type:
            if data_type in self.timestamps:
                del self.timestamps[data_type]
                logger.info(f"타임스탬프 데이터 삭제 완료: {data_type}")
            else:
                logger.warning(f"삭제할 데이터 타입이 존재하지 않음: {data_type}")
        else:
            self.timestamps = {}
            logger.info("모든 타임스탬프 데이터 삭제 완료")
        
        self._save_timestamps()
    
    def export_timestamps(self, output_file: str = None) -> str:
        """
        타임스탬프 데이터 내보내기
        
        Args:
            output_file: 출력 파일 경로 (None이면 자동 생성)
            
        Returns:
            내보낸 파일 경로
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.metadata_dir / f"timestamps_export_{timestamp}.json"
        
        output_path = Path(output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.timestamps, f, ensure_ascii=False, indent=2)
            
            logger.info(f"타임스탬프 데이터 내보내기 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"타임스탬프 데이터 내보내기 실패: {e}")
            raise


# 편의 함수들
def get_last_collection_time(data_type: str, metadata_dir: str = "data/raw/law_open_api/metadata") -> Optional[datetime]:
    """
    마지막 수집 시간 조회 (편의 함수)
    
    Args:
        data_type: 데이터 타입
        metadata_dir: 메타데이터 디렉토리
        
    Returns:
        마지막 수집 시간
    """
    manager = TimestampManager(metadata_dir)
    return manager.get_last_collection_time(data_type)


def update_collection_time(data_type: str, success: bool = True, 
                          metadata_dir: str = "data/raw/law_open_api/metadata"):
    """
    수집 시간 업데이트 (편의 함수)
    
    Args:
        data_type: 데이터 타입
        success: 수집 성공 여부
        metadata_dir: 메타데이터 디렉토리
    """
    manager = TimestampManager(metadata_dir)
    manager.update_collection_time(data_type, success)


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("TimestampManager 테스트")
    print("=" * 40)
    
    # 매니저 생성
    manager = TimestampManager("test_metadata")
    
    # 테스트 데이터 타입
    test_data_type = "legal_terms"
    
    # 마지막 수집 시간 조회 (처음에는 None)
    last_time = manager.get_last_collection_time(test_data_type)
    print(f"마지막 수집 시간: {last_time}")
    
    # 수집 시간 업데이트 (성공)
    manager.update_collection_time(test_data_type, success=True)
    
    # 업데이트 후 조회
    last_time = manager.get_last_collection_time(test_data_type)
    print(f"업데이트 후 마지막 수집 시간: {last_time}")
    
    # 통계 조회
    stats = manager.get_collection_stats(test_data_type)
    print(f"수집 통계: {stats}")
    
    # 실패 케이스 테스트
    manager.update_collection_time(test_data_type, success=False)
    
    # 최종 통계
    final_stats = manager.get_collection_stats(test_data_type)
    print(f"최종 통계: {final_stats}")
    
    print("✅ TimestampManager 테스트 완료")




