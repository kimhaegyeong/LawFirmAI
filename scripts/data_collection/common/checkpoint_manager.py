# -*- coding: utf-8 -*-
"""
Checkpoint Manager
ì²´í¬?¬ì¸??ê´€ë¦?ëª¨ë“ˆ

?˜ì§‘ ì§„í–‰ ?í™©???€?¥í•˜ê³?ë³µêµ¬?˜ëŠ” ê¸°ëŠ¥???œê³µ?©ë‹ˆ??
ì¤‘ë‹¨??ì§€?ì—???•í™•???¬ê°œ?????ˆë„ë¡?ì§€?í•©?ˆë‹¤.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """ì²´í¬?¬ì¸??ê´€ë¦??´ë˜??""
    
    def __init__(self, checkpoint_dir: str):
        """
        ì²´í¬?¬ì¸??ë§¤ë‹ˆ?€ ì´ˆê¸°??
        
        Args:
            checkpoint_dir: ì²´í¬?¬ì¸???Œì¼ ?€???”ë ‰? ë¦¬
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.backup_file = self.checkpoint_dir / "checkpoint_backup.json"
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        ì²´í¬?¬ì¸???€??(ë°±ì—… ?†ì´ ê°„ë‹¨?˜ê²Œ)
        
        Args:
            checkpoint_data: ?€?¥í•  ì²´í¬?¬ì¸???°ì´??
        
        Returns:
            bool: ?€???±ê³µ ?¬ë?
        """
        try:
            # ?€?„ìŠ¤?¬í”„ ì¶”ê?
            checkpoint_data['saved_at'] = datetime.now().isoformat()
            checkpoint_data['checkpoint_version'] = '1.0'
            
            # ??ì²´í¬?¬ì¸???€??(??–´?°ê¸°)
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            print(f"??Checkpoint saved: {self.checkpoint_file}")
            return True
            
        except Exception as e:
            print(f"??Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        ì²´í¬?¬ì¸??ë¡œë“œ
        
        Returns:
            Optional[Dict]: ì²´í¬?¬ì¸???°ì´???ëŠ” None
        """
        try:
            if not self.checkpoint_file.exists():
                self.logger.info("?“‚ No checkpoint found")
                return None
            
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"?“‚ Checkpoint loaded: {self.checkpoint_file}")
            self.logger.info(f"   Data type: {checkpoint_data.get('data_type', 'unknown')}")
            self.logger.info(f"   Current page: {checkpoint_data.get('current_page', 0)}")
            self.logger.info(f"   Collected: {checkpoint_data.get('collected_count', 0)} items")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"??Failed to load checkpoint: {e}")
            
            # ë°±ì—…?ì„œ ë¡œë“œ ?œë„
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    self.logger.info("??Loaded from backup checkpoint")
                    return checkpoint_data
                    
                except Exception as backup_e:
                    self.logger.error(f"??Failed to load backup: {backup_e}")
            
            return None
    
    def clear_checkpoint(self) -> bool:
        """
        ì²´í¬?¬ì¸???? œ (?˜ì§‘ ?„ë£Œ ??
        
        Returns:
            bool: ?? œ ?±ê³µ ?¬ë?
        """
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            if self.backup_file.exists():
                self.backup_file.unlink()
            
            print("??Checkpoint cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"??Failed to clear checkpoint: {e}")
            return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        ì²´í¬?¬ì¸???•ë³´ ì¡°íšŒ
        
        Returns:
            Dict: ì²´í¬?¬ì¸???íƒœ ?•ë³´
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
        ì²´í¬?¬ì¸??? íš¨??ê²€ì¦?
        
        Returns:
            bool: ? íš¨???¬ë?
        """
        try:
            checkpoint_data = self.load_checkpoint()
            if not checkpoint_data:
                return True  # ì²´í¬?¬ì¸?¸ê? ?†ìœ¼ë©?? íš¨
            
            # ?„ìˆ˜ ?„ë“œ ê²€ì¦?
            required_fields = ['data_type', 'current_page', 'total_pages', 'collected_count']
            for field in required_fields:
                if field not in checkpoint_data:
                    self.logger.error(f"??Missing required field: {field}")
                    return False
            
            # ?°ì´???€??ê²€ì¦?
            if checkpoint_data['data_type'] not in ['law', 'precedent']:
                self.logger.error(f"??Invalid data_type: {checkpoint_data['data_type']}")
                return False
            
            # ?˜ì´ì§€ ë²ˆí˜¸ ê²€ì¦?
            if not isinstance(checkpoint_data['current_page'], int) or checkpoint_data['current_page'] < 0:
                self.logger.error(f"??Invalid current_page: {checkpoint_data['current_page']}")
                return False
            
            if not isinstance(checkpoint_data['total_pages'], int) or checkpoint_data['total_pages'] <= 0:
                self.logger.error(f"??Invalid total_pages: {checkpoint_data['total_pages']}")
                return False
            
            if checkpoint_data['current_page'] > checkpoint_data['total_pages']:
                self.logger.error(f"??current_page > total_pages")
                return False
            
            self.logger.info("??Checkpoint validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"??Checkpoint validation failed: {e}")
            return False
