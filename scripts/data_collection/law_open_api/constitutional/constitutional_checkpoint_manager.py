#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헌재결정례 체크포인트 관리 시스템

수집 중단 시 재개할 수 있도록 체크포인트를 관리하는 시스템입니다.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """체크포인트 데이터"""
    checkpoint_id: str
    collection_type: str  # "keyword", "all", "date_range"
    keyword: str = ""
    start_date: str = ""
    end_date: str = ""
    current_page: int = 1
    total_pages: int = 0
    collected_count: int = 0
    batch_count: int = 0
    last_decision_id: str = ""
    sort_order: str = "dasc"
    include_details: bool = True
    batch_size: int = 100
    created_at: str = ""
    updated_at: str = ""
    status: str = "in_progress"  # "in_progress", "completed", "failed"


class ConstitutionalCheckpointManager:
    """헌재결정례 체크포인트 관리자"""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints/constitutional"):
        """
        체크포인트 관리자 초기화
        
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConstitutionalCheckpointManager 초기화 - 디렉토리: {self.checkpoint_dir}")
    
    def create_checkpoint(self, 
                         collection_type: str,
                         keyword: str = "",
                         start_date: str = "",
                         end_date: str = "",
                         sort_order: str = "dasc",
                         include_details: bool = True,
                         batch_size: int = 100) -> str:
        """
        새로운 체크포인트 생성
        
        Args:
            collection_type: 수집 유형
            keyword: 검색 키워드
            start_date: 시작 날짜
            end_date: 종료 날짜
            sort_order: 정렬 순서
            include_details: 상세 정보 포함 여부
            batch_size: 배치 크기
            
        Returns:
            str: 체크포인트 ID
        """
        checkpoint_id = f"constitutional_{collection_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_data = CheckpointData(
            checkpoint_id=checkpoint_id,
            collection_type=collection_type,
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            sort_order=sort_order,
            include_details=include_details,
            batch_size=batch_size,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint_data), f, ensure_ascii=False, indent=2)
        
        logger.info(f"체크포인트 생성: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointData]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_id: 체크포인트 ID
            
        Returns:
            CheckpointData: 체크포인트 데이터 또는 None
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"체크포인트 파일이 존재하지 않음: {checkpoint_file}")
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoint_data = CheckpointData(**data)
            logger.info(f"체크포인트 로드: {checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {checkpoint_id} - {e}")
            return None
    
    def update_checkpoint(self, 
                         checkpoint_id: str,
                         current_page: int = None,
                         total_pages: int = None,
                         collected_count: int = None,
                         batch_count: int = None,
                         last_decision_id: str = None,
                         status: str = None) -> bool:
        """
        체크포인트 업데이트
        
        Args:
            checkpoint_id: 체크포인트 ID
            current_page: 현재 페이지
            total_pages: 총 페이지 수
            collected_count: 수집된 개수
            batch_count: 배치 수
            last_decision_id: 마지막 결정례 ID
            status: 상태
            
        Returns:
            bool: 업데이트 성공 여부
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"체크포인트 파일이 존재하지 않음: {checkpoint_file}")
            return False
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 업데이트할 필드들
            if current_page is not None:
                data['current_page'] = current_page
            if total_pages is not None:
                data['total_pages'] = total_pages
            if collected_count is not None:
                data['collected_count'] = collected_count
            if batch_count is not None:
                data['batch_count'] = batch_count
            if last_decision_id is not None:
                data['last_decision_id'] = last_decision_id
            if status is not None:
                data['status'] = status
            
            data['updated_at'] = datetime.now().isoformat()
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"체크포인트 업데이트: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 업데이트 실패: {checkpoint_id} - {e}")
            return False
    
    def complete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        체크포인트 완료 처리
        
        Args:
            checkpoint_id: 체크포인트 ID
            
        Returns:
            bool: 완료 처리 성공 여부
        """
        return self.update_checkpoint(checkpoint_id, status="completed")
    
    def fail_checkpoint(self, checkpoint_id: str) -> bool:
        """
        체크포인트 실패 처리
        
        Args:
            checkpoint_id: 체크포인트 ID
            
        Returns:
            bool: 실패 처리 성공 여부
        """
        return self.update_checkpoint(checkpoint_id, status="failed")
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        체크포인트 삭제
        
        Args:
            checkpoint_id: 체크포인트 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"체크포인트 삭제: {checkpoint_id}")
                return True
            else:
                logger.warning(f"체크포인트 파일이 존재하지 않음: {checkpoint_file}")
                return False
                
        except Exception as e:
            logger.error(f"체크포인트 삭제 실패: {checkpoint_id} - {e}")
            return False
    
    def list_checkpoints(self, status: str = None) -> List[Dict[str, Any]]:
        """
        체크포인트 목록 조회
        
        Args:
            status: 상태 필터 (None이면 모든 상태)
            
        Returns:
            List[Dict]: 체크포인트 목록
        """
        checkpoints = []
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("constitutional_*.json"):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if status is None or data.get('status') == status:
                        checkpoints.append(data)
                        
                except Exception as e:
                    logger.warning(f"체크포인트 파일 읽기 실패: {checkpoint_file} - {e}")
                    continue
            
            # 생성일시 기준 내림차순 정렬
            checkpoints.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"체크포인트 목록 조회 실패: {e}")
        
        return checkpoints
    
    def get_latest_checkpoint(self, collection_type: str = None) -> Optional[CheckpointData]:
        """
        최신 체크포인트 조회
        
        Args:
            collection_type: 수집 유형 필터 (None이면 모든 유형)
            
        Returns:
            CheckpointData: 최신 체크포인트 또는 None
        """
        checkpoints = self.list_checkpoints()
        
        for checkpoint_data in checkpoints:
            if collection_type is None or checkpoint_data.get('collection_type') == collection_type:
                return CheckpointData(**checkpoint_data)
        
        return None
    
    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        오래된 체크포인트 정리
        
        Args:
            days: 보관 기간 (일)
            
        Returns:
            int: 삭제된 체크포인트 수
        """
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        try:
            for checkpoint_file in self.checkpoint_dir.glob("constitutional_*.json"):
                try:
                    file_time = checkpoint_file.stat().st_mtime
                    
                    if file_time < cutoff_time:
                        checkpoint_file.unlink()
                        deleted_count += 1
                        logger.info(f"오래된 체크포인트 삭제: {checkpoint_file.name}")
                        
                except Exception as e:
                    logger.warning(f"체크포인트 파일 삭제 실패: {checkpoint_file} - {e}")
                    continue
            
            logger.info(f"오래된 체크포인트 정리 완료: {deleted_count}개 삭제")
            
        except Exception as e:
            logger.error(f"체크포인트 정리 실패: {e}")
        
        return deleted_count


def main():
    """메인 함수 (체크포인트 관리 도구)"""
    import argparse
    
    parser = argparse.ArgumentParser(description="헌재결정례 체크포인트 관리 도구")
    parser.add_argument("--list", action="store_true", help="체크포인트 목록 조회")
    parser.add_argument("--status", type=str, help="상태별 필터 (in_progress, completed, failed)")
    parser.add_argument("--latest", type=str, help="최신 체크포인트 조회 (수집 유형)")
    parser.add_argument("--delete", type=str, help="체크포인트 삭제 (체크포인트 ID)")
    parser.add_argument("--cleanup", type=int, default=7, help="오래된 체크포인트 정리 (보관 기간)")
    
    args = parser.parse_args()
    
    manager = ConstitutionalCheckpointManager()
    
    if args.list:
        print("헌재결정례 체크포인트 목록")
        print("=" * 50)
        
        checkpoints = manager.list_checkpoints(args.status)
        
        if not checkpoints:
            print("체크포인트가 없습니다.")
        else:
            for i, checkpoint in enumerate(checkpoints, 1):
                print(f"{i}. {checkpoint['checkpoint_id']}")
                print(f"   유형: {checkpoint['collection_type']}")
                print(f"   키워드: {checkpoint.get('keyword', 'N/A')}")
                print(f"   페이지: {checkpoint['current_page']}/{checkpoint['total_pages']}")
                print(f"   수집: {checkpoint['collected_count']:,}개")
                print(f"   상태: {checkpoint['status']}")
                print(f"   생성: {checkpoint['created_at']}")
                print()
    
    elif args.latest:
        print(f"최신 {args.latest} 체크포인트")
        print("=" * 50)
        
        checkpoint = manager.get_latest_checkpoint(args.latest)
        
        if checkpoint:
            print(f"체크포인트 ID: {checkpoint.checkpoint_id}")
            print(f"유형: {checkpoint.collection_type}")
            print(f"키워드: {checkpoint.keyword}")
            print(f"페이지: {checkpoint.current_page}/{checkpoint.total_pages}")
            print(f"수집: {checkpoint.collected_count:,}개")
            print(f"상태: {checkpoint.status}")
            print(f"생성: {checkpoint.created_at}")
        else:
            print("해당 유형의 체크포인트가 없습니다.")
    
    elif args.delete:
        print(f"체크포인트 삭제: {args.delete}")
        
        if manager.delete_checkpoint(args.delete):
            print("✅ 삭제 완료")
        else:
            print("❌ 삭제 실패")
    
    elif args.cleanup:
        print(f"오래된 체크포인트 정리 (보관 기간: {args.cleanup}일)")
        
        deleted_count = manager.cleanup_old_checkpoints(args.cleanup)
        print(f"✅ {deleted_count}개 체크포인트 삭제 완료")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
