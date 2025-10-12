# -*- coding: utf-8 -*-
"""
Checkpoint Manager
체크포인트 관리 모듈

수집 진행 상황을 저장하고 복구하는 기능을 제공합니다.
중단된 지점에서 정확히 재개할 수 있도록 지원합니다.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_dir: str):
        """
        체크포인트 매니저 초기화
        
        Args:
            checkpoint_dir: 체크포인트 파일 저장 디렉토리
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.backup_file = self.checkpoint_dir / "checkpoint_backup.json"
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        체크포인트 저장 (백업 없이 간단하게)
        
        Args:
            checkpoint_data: 저장할 체크포인트 데이터
        
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 타임스탬프 추가
            checkpoint_data['saved_at'] = datetime.now().isoformat()
            checkpoint_data['checkpoint_version'] = '1.0'
            
            # 새 체크포인트 저장 (덮어쓰기)
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Checkpoint saved: {self.checkpoint_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        체크포인트 로드
        
        Returns:
            Optional[Dict]: 체크포인트 데이터 또는 None
        """
        try:
            if not self.checkpoint_file.exists():
                self.logger.info("📂 No checkpoint found")
                return None
            
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"📂 Checkpoint loaded: {self.checkpoint_file}")
            self.logger.info(f"   Data type: {checkpoint_data.get('data_type', 'unknown')}")
            self.logger.info(f"   Current page: {checkpoint_data.get('current_page', 0)}")
            self.logger.info(f"   Collected: {checkpoint_data.get('collected_count', 0)} items")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            
            # 백업에서 로드 시도
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    self.logger.info("✅ Loaded from backup checkpoint")
                    return checkpoint_data
                    
                except Exception as backup_e:
                    self.logger.error(f"❌ Failed to load backup: {backup_e}")
            
            return None
    
    def clear_checkpoint(self) -> bool:
        """
        체크포인트 삭제 (수집 완료 시)
        
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            if self.backup_file.exists():
                self.backup_file.unlink()
            
            print("✅ Checkpoint cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear checkpoint: {e}")
            return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        체크포인트 정보 조회
        
        Returns:
            Dict: 체크포인트 상태 정보
        """
        info = {
            'checkpoint_exists': self.checkpoint_file.exists(),
            'backup_exists': self.backup_file.exists(),
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoint_file': str(self.checkpoint_file),
            'backup_file': str(self.backup_file)
        }
        
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                info.update({
                    'data_type': checkpoint_data.get('data_type'),
                    'category': checkpoint_data.get('category'),
                    'current_page': checkpoint_data.get('current_page'),
                    'total_pages': checkpoint_data.get('total_pages'),
                    'collected_count': checkpoint_data.get('collected_count'),
                    'saved_at': checkpoint_data.get('saved_at')
                })
                
            except Exception as e:
                info['load_error'] = str(e)
        
        return info
    
    def validate_checkpoint(self) -> bool:
        """
        체크포인트 유효성 검증
        
        Returns:
            bool: 유효성 여부
        """
        try:
            checkpoint_data = self.load_checkpoint()
            if not checkpoint_data:
                return True  # 체크포인트가 없으면 유효
            
            # 필수 필드 검증
            required_fields = ['data_type', 'current_page', 'total_pages', 'collected_count']
            for field in required_fields:
                if field not in checkpoint_data:
                    self.logger.error(f"❌ Missing required field: {field}")
                    return False
            
            # 데이터 타입 검증
            if checkpoint_data['data_type'] not in ['law', 'precedent']:
                self.logger.error(f"❌ Invalid data_type: {checkpoint_data['data_type']}")
                return False
            
            # 페이지 번호 검증
            if not isinstance(checkpoint_data['current_page'], int) or checkpoint_data['current_page'] < 0:
                self.logger.error(f"❌ Invalid current_page: {checkpoint_data['current_page']}")
                return False
            
            if not isinstance(checkpoint_data['total_pages'], int) or checkpoint_data['total_pages'] <= 0:
                self.logger.error(f"❌ Invalid total_pages: {checkpoint_data['total_pages']}")
                return False
            
            if checkpoint_data['current_page'] > checkpoint_data['total_pages']:
                self.logger.error(f"❌ current_page > total_pages")
                return False
            
            self.logger.info("✅ Checkpoint validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint validation failed: {e}")
            return False
