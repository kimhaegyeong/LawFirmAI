#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 변경사항 감지 유틸리티

법령용어 데이터의 변경사항을 감지하고 분석하는 모듈입니다.
- 새로운 레코드 감지
- 업데이트된 레코드 감지
- 삭제된 레코드 감지
- 변경사항 분석 및 저장
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)


class ChangeDetector:
    """데이터 변경사항 감지"""
    
    def __init__(self, data_dir: str = "data/raw/law_open_api/legal_terms"):
        """
        변경사항 감지기 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 변경사항 로그 파일
        self.change_log_file = self.data_dir.parent / "metadata" / "change_log.json"
        self.change_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChangeDetector 초기화 완료 - 데이터 디렉토리: {self.data_dir}")
    
    def analyze_changes(self, data_type: str, new_data: List[Dict], 
                       last_collection: Optional[datetime] = None) -> Dict[str, Any]:
        """
        변경사항 분석
        
        Args:
            data_type: 데이터 타입
            new_data: 새로운 데이터 목록
            last_collection: 마지막 수집 시간
            
        Returns:
            변경사항 분석 결과
        """
        logger.info(f"변경사항 분석 시작: {data_type}, 새 데이터: {len(new_data)}개")
        
        # 기존 데이터 로드
        existing_data = self._load_existing_data(data_type)
        
        changes = {
            "data_type": data_type,
            "analysis_time": datetime.now().isoformat(),
            "last_collection": last_collection.isoformat() if last_collection else None,
            "new_records": [],
            "updated_records": [],
            "deleted_records": [],
            "unchanged_records": [],
            "summary": {
                "total_existing": len(existing_data),
                "total_new": len(new_data),
                "new_count": 0,
                "updated_count": 0,
                "deleted_count": 0,
                "unchanged_count": 0
            }
        }
        
        # 기존 데이터의 ID 집합 생성
        existing_ids = self._extract_ids(existing_data)
        new_ids = self._extract_ids(new_data)
        
        logger.debug(f"기존 데이터 ID 수: {len(existing_ids)}, 새 데이터 ID 수: {len(new_ids)}")
        
        # 새로운 레코드 감지
        for record in new_data:
            record_id = self._get_record_id(record)
            if record_id not in existing_ids:
                changes["new_records"].append(record)
                changes["summary"]["new_count"] += 1
            else:
                # 기존 레코드와 비교
                existing_record = self._find_record_by_id(existing_data, record_id)
                if existing_record and self._has_changes(existing_record, record):
                    changes["updated_records"].append({
                        "old": existing_record,
                        "new": record,
                        "changes": self._get_field_changes(existing_record, record)
                    })
                    changes["summary"]["updated_count"] += 1
                else:
                    changes["unchanged_records"].append(record)
                    changes["summary"]["unchanged_count"] += 1
        
        # 삭제된 레코드 감지
        for record in existing_data:
            record_id = self._get_record_id(record)
            if record_id not in new_ids:
                changes["deleted_records"].append(record)
                changes["summary"]["deleted_count"] += 1
        
        # 변경사항 로그 저장
        self._save_change_log(changes)
        
        logger.info(f"변경사항 분석 완료: 새 {changes['summary']['new_count']}개, "
                   f"업데이트 {changes['summary']['updated_count']}개, "
                   f"삭제 {changes['summary']['deleted_count']}개")
        
        return changes
    
    def _load_existing_data(self, data_type: str) -> List[Dict]:
        """기존 데이터 로드"""
        # 전체 데이터 파일에서 로드
        full_data_file = self.data_dir / "full" / f"{data_type}_latest.json"
        
        if full_data_file.exists():
            try:
                with open(full_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"기존 데이터 로드 완료: {len(data)}개 레코드")
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.error(f"기존 데이터 로드 실패: {e}")
                return []
        else:
            logger.info(f"기존 데이터 파일이 존재하지 않음: {full_data_file}")
            return []
    
    def _extract_ids(self, records: List[Dict]) -> Set[str]:
        """레코드에서 ID 추출"""
        ids = set()
        for record in records:
            record_id = self._get_record_id(record)
            if record_id:
                ids.add(record_id)
        return ids
    
    def _get_record_id(self, record: Dict) -> Optional[str]:
        """레코드에서 ID 추출"""
        # 법령용어의 경우 termId 사용
        return record.get("termId") or record.get("id") or record.get("term_id")
    
    def _find_record_by_id(self, records: List[Dict], record_id: str) -> Optional[Dict]:
        """ID로 레코드 찾기"""
        for record in records:
            if self._get_record_id(record) == record_id:
                return record
        return None
    
    def _has_changes(self, old_record: Dict, new_record: Dict) -> bool:
        """레코드 변경 여부 확인"""
        # 중요 필드들 비교
        important_fields = [
            "termName", "termContent", "termDefinition", 
            "relatedTerms", "category", "status", "updateDate"
        ]
        
        for field in important_fields:
            if old_record.get(field) != new_record.get(field):
                return True
        
        return False
    
    def _get_field_changes(self, old_record: Dict, new_record: Dict) -> Dict[str, Any]:
        """필드별 변경사항 추출"""
        changes = {}
        
        all_fields = set(old_record.keys()) | set(new_record.keys())
        
        for field in all_fields:
            old_value = old_record.get(field)
            new_value = new_record.get(field)
            
            if old_value != new_value:
                changes[field] = {
                    "old": old_value,
                    "new": new_value
                }
        
        return changes
    
    def _save_change_log(self, changes: Dict[str, Any]):
        """변경사항 로그 저장"""
        try:
            # 기존 로그 로드
            change_logs = []
            if self.change_log_file.exists():
                with open(self.change_log_file, 'r', encoding='utf-8') as f:
                    change_logs = json.load(f)
            
            # 새 로그 추가
            change_logs.append(changes)
            
            # 최근 100개만 유지
            if len(change_logs) > 100:
                change_logs = change_logs[-100:]
            
            # 저장
            with open(self.change_log_file, 'w', encoding='utf-8') as f:
                json.dump(change_logs, f, ensure_ascii=False, indent=2)
            
            logger.debug("변경사항 로그 저장 완료")
            
        except Exception as e:
            logger.error(f"변경사항 로그 저장 실패: {e}")
    
    def save_full_data(self, data_type: str, data: List[Dict]):
        """
        전체 데이터 저장 (최신 버전)
        
        Args:
            data_type: 데이터 타입
            data: 저장할 데이터
        """
        full_dir = self.data_dir / "full"
        full_dir.mkdir(parents=True, exist_ok=True)
        
        # 최신 데이터 파일
        latest_file = full_dir / f"{data_type}_latest.json"
        
        # 타임스탬프가 포함된 백업 파일
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = full_dir / f"{data_type}_{timestamp}.json"
        
        try:
            # 최신 데이터 저장
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 백업 파일 저장
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"전체 데이터 저장 완료: {len(data)}개 레코드")
            
        except Exception as e:
            logger.error(f"전체 데이터 저장 실패: {e}")
            raise
    
    def save_incremental_data(self, data_type: str, changes: Dict[str, Any]):
        """
        증분 데이터 저장
        
        Args:
            data_type: 데이터 타입
            changes: 변경사항 데이터
        """
        incremental_dir = self.data_dir / "incremental"
        date_str = datetime.now().strftime("%Y%m%d")
        daily_dir = incremental_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 새로운 레코드 저장
            if changes["new_records"]:
                new_file = daily_dir / "new_records.json"
                with open(new_file, 'w', encoding='utf-8') as f:
                    json.dump(changes["new_records"], f, ensure_ascii=False, indent=2)
            
            # 업데이트된 레코드 저장
            if changes["updated_records"]:
                updated_file = daily_dir / "updated_records.json"
                with open(updated_file, 'w', encoding='utf-8') as f:
                    json.dump(changes["updated_records"], f, ensure_ascii=False, indent=2)
            
            # 삭제된 레코드 저장
            if changes["deleted_records"]:
                deleted_file = daily_dir / "deleted_records.json"
                with open(deleted_file, 'w', encoding='utf-8') as f:
                    json.dump(changes["deleted_records"], f, ensure_ascii=False, indent=2)
            
            # 요약 정보 저장
            summary_file = daily_dir / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(changes["summary"], f, ensure_ascii=False, indent=2)
            
            logger.info(f"증분 데이터 저장 완료: {daily_dir}")
            
        except Exception as e:
            logger.error(f"증분 데이터 저장 실패: {e}")
            raise
    
    def get_change_history(self, data_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        변경 이력 조회
        
        Args:
            data_type: 데이터 타입 (None이면 모든 타입)
            limit: 조회할 이력 수
            
        Returns:
            변경 이력 목록
        """
        if not self.change_log_file.exists():
            return []
        
        try:
            with open(self.change_log_file, 'r', encoding='utf-8') as f:
                change_logs = json.load(f)
            
            # 데이터 타입 필터링
            if data_type:
                filtered_logs = [log for log in change_logs if log.get("data_type") == data_type]
            else:
                filtered_logs = change_logs
            
            # 최근 이력만 반환
            return filtered_logs[-limit:] if limit else filtered_logs
            
        except Exception as e:
            logger.error(f"변경 이력 조회 실패: {e}")
            return []


# 편의 함수들
def analyze_changes(data_type: str, new_data: List[Dict], 
                   last_collection: datetime = None,
                   data_dir: str = "data/raw/law_open_api/legal_terms") -> Dict[str, Any]:
    """
    변경사항 분석 (편의 함수)
    
    Args:
        data_type: 데이터 타입
        new_data: 새로운 데이터
        last_collection: 마지막 수집 시간
        data_dir: 데이터 디렉토리
        
    Returns:
        변경사항 분석 결과
    """
    detector = ChangeDetector(data_dir)
    return detector.analyze_changes(data_type, new_data, last_collection)


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("ChangeDetector 테스트")
    print("=" * 40)
    
    # 감지기 생성
    detector = ChangeDetector("test_data")
    
    # 테스트 데이터
    existing_data = [
        {"termId": "1", "termName": "계약", "termContent": "계약 내용"},
        {"termId": "2", "termName": "손해배상", "termContent": "손해배상 내용"},
        {"termId": "3", "termName": "불법행위", "termContent": "불법행위 내용"}
    ]
    
    new_data = [
        {"termId": "1", "termName": "계약", "termContent": "계약 내용 수정"},  # 업데이트
        {"termId": "2", "termName": "손해배상", "termContent": "손해배상 내용"},  # 변경 없음
        {"termId": "4", "termName": "채권", "termContent": "채권 내용"},  # 새로운 레코드
        # termId "3"은 삭제됨
    ]
    
    # 기존 데이터 저장
    detector.save_full_data("legal_terms", existing_data)
    
    # 변경사항 분석
    changes = detector.analyze_changes("legal_terms", new_data)
    
    print(f"변경사항 분석 결과:")
    print(f"  - 새로운 레코드: {changes['summary']['new_count']}개")
    print(f"  - 업데이트된 레코드: {changes['summary']['updated_count']}개")
    print(f"  - 삭제된 레코드: {changes['summary']['deleted_count']}개")
    print(f"  - 변경 없는 레코드: {changes['summary']['unchanged_count']}개")
    
    # 증분 데이터 저장
    detector.save_incremental_data("legal_terms", changes)
    
    # 변경 이력 조회
    history = detector.get_change_history("legal_terms", 5)
    print(f"변경 이력: {len(history)}개")
    
    print("✅ ChangeDetector 테스트 완료")




