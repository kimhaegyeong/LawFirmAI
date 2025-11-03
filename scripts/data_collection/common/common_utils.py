#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Collection Common Utilities
êµ?šŒ ?˜ì§‘ ?¤í¬ë¦½íŠ¸?¤ì˜ ê³µí†µ ? í‹¸ë¦¬í‹° ëª¨ë“ˆ

??ëª¨ë“ˆ?€ ëª¨ë“  ?˜ì§‘ ?¤í¬ë¦½íŠ¸?ì„œ ê³µí†µ?¼ë¡œ ?¬ìš©?˜ëŠ” ê¸°ëŠ¥?¤ì„ ?œê³µ?©ë‹ˆ??
- ë©”ëª¨ë¦?ê´€ë¦?
- ?ëŸ¬ ì²˜ë¦¬
- ë¡œê¹… ?¤ì •
- ?œê·¸???¸ë“¤ë§?
- ?¬ì‹œ??ë¡œì§
"""

import gc
import psutil
import logging
import signal
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
import json


class MemoryManager:
    """ë©”ëª¨ë¦?ê´€ë¦?? í‹¸ë¦¬í‹° ?´ë˜??(ìµœì ?”ëœ ë²„ì „)"""
    
    def __init__(self, memory_limit_mb: int = 1000, cleanup_threshold: float = 0.3):  # 1000MBë¡?ì¦ê?, 30%?ì„œ ?•ë¦¬ ?œì‘
        """
        ë©”ëª¨ë¦?ë§¤ë‹ˆ?€ ì´ˆê¸°??
        
        Args:
            memory_limit_mb: ë©”ëª¨ë¦??œí•œ (MB) - ê¸°ë³¸ê°’ì„ 1000MBë¡??¤ì •
            cleanup_threshold: ?•ë¦¬ ?œì‘ ?„ê³„ê°?(0.0-1.0) - 30%?ì„œ ?œì‘
        """
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.warning_threshold = 0.2  # 20%?ì„œ ê²½ê³ 
        self.critical_threshold = 0.6  # 60%?ì„œ ì¤‘ë‹¨
        self.logger = logging.getLogger(__name__)
        self.cleanup_count = 0
        
    def get_memory_usage(self) -> float:
        """?„ì¬ ë©”ëª¨ë¦??¬ìš©??ë°˜í™˜ (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_and_cleanup(self) -> bool:
        """
        ë©”ëª¨ë¦??¬ìš©??ì²´í¬ ë°??•ë¦¬ (ê°•í™”??ë²„ì „)
        
        Returns:
            bool: ë©”ëª¨ë¦??œí•œ ?´ì— ?ˆìœ¼ë©?True, ì´ˆê³¼?˜ë©´ False
        """
        memory_mb = self.get_memory_usage()
        
        # 20%?ì„œ ê²½ê³ 
        if memory_mb > self.memory_limit_mb * self.warning_threshold:
            self.logger.warning(f"Memory usage warning: {memory_mb:.1f}MB ({memory_mb/self.memory_limit_mb*100:.1f}%)")
        
        # 30%?ì„œ ?•ë¦¬ ?œì‘
        if memory_mb > self.memory_limit_mb * self.cleanup_threshold:
            self.logger.warning(f"High memory usage: {memory_mb:.1f}MB, cleaning up...")
            self.aggressive_cleanup()
            
            memory_after = self.get_memory_usage()
            self.logger.info(f"After cleanup: {memory_after:.1f}MB")
            
            if memory_after > self.memory_limit_mb * 0.7:  # 70% ?´ìƒ?´ë©´ ì¤‘ë‹¨
                self.logger.error(f"Memory limit exceeded: {memory_after:.1f}MB")
                return False
        
        # 60%?ì„œ ì¤‘ë‹¨
        if memory_mb > self.memory_limit_mb * self.critical_threshold:
            self.logger.error(f"Critical memory usage: {memory_mb:.1f}MB - stopping")
            return False
        
        return True
    
    def aggressive_cleanup(self):
        """ê°•í™”??ë©”ëª¨ë¦??•ë¦¬"""
        self.cleanup_count += 1
        
        # ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰??
        gc.collect()
        
        # ë©”ëª¨ë¦??¬ìš©???¬í™•??
        current_mb = self.get_memory_usage()
        if current_mb > self.memory_limit_mb * 0.8:
            if self.cleanup_count > 3:
                raise MemoryError(f"Memory cleanup failed after {self.cleanup_count} attempts: {current_mb:.1f}MB")
        
        self.logger.info(f"Aggressive cleanup completed (attempt {self.cleanup_count})")
    
    def force_cleanup(self):
        """ê°•ì œ ë©”ëª¨ë¦??•ë¦¬"""
        gc.collect()
        self.logger.info("Forced memory cleanup completed")
    
    def adaptive_batch_size(self, current_memory_mb: float, base_batch_size: int = 10) -> int:
        """ë©”ëª¨ë¦??¬ìš©?‰ì— ?°ë¥¸ ë°°ì¹˜ ?¬ê¸° ?™ì  ì¡°ì •"""
        if current_memory_mb > self.memory_limit_mb * 0.7:  # 70% ì´ˆê³¼??
            return max(5, base_batch_size - 3)
        elif current_memory_mb > self.memory_limit_mb * 0.5:  # 50% ì´ˆê³¼??
            return max(8, base_batch_size - 2)
        else:
            return min(15, base_batch_size + 1)


class RetryManager:
    """?¬ì‹œ??ê´€ë¦?? í‹¸ë¦¬í‹° ?´ë˜??""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        """
        ?¬ì‹œ??ë§¤ë‹ˆ?€ ì´ˆê¸°??
        
        Args:
            max_retries: ìµœë? ?¬ì‹œ???Ÿìˆ˜
            base_delay: ê¸°ë³¸ ?€ê¸??œê°„ (ì´?
            backoff_factor: ì§€??ë°±ì˜¤??ê³„ìˆ˜
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        ?¨ìˆ˜ ?¬ì‹œ???¤í–‰
        
        Args:
            func: ?¤í–‰???¨ìˆ˜
            *args: ?¨ìˆ˜ ?¸ì
            **kwargs: ?¨ìˆ˜ ?¤ì›Œ???¸ì
            
        Returns:
            ?¨ìˆ˜ ?¤í–‰ ê²°ê³¼
            
        Raises:
            Exception: ìµœë? ?¬ì‹œ???„ì—???¤íŒ¨??ê²½ìš°
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.max_retries + 1} attempts failed")
        
        raise last_exception


class SignalHandler:
    """?œê·¸???¸ë“¤ë§?? í‹¸ë¦¬í‹° ?´ë˜??""
    
    def __init__(self):
        """?œê·¸???¸ë“¤??ì´ˆê¸°??""
        self.interrupted = False
        self.logger = logging.getLogger(__name__)
        self._setup_handlers()
    
    def _setup_handlers(self):
        """?œê·¸???¸ë“¤???¤ì •"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """?œê·¸???¸ë“¤??""
        self.logger.warning(f"Signal {signum} received. Initiating graceful shutdown...")
        self.interrupted = True
    
    def is_interrupted(self) -> bool:
        """ì¤‘ë‹¨ ? í˜¸ ?•ì¸"""
        return self.interrupted


class CollectionLogger:
    """?˜ì§‘ ?‘ì—… ë¡œê¹… ? í‹¸ë¦¬í‹° ?´ë˜??""
    
    @staticmethod
    def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
        """
        ë¡œê¹… ?¤ì •
        
        Args:
            name: ë¡œê±° ?´ë¦„
            level: ë¡œê·¸ ?ˆë²¨
            
        Returns:
            ?¤ì •??ë¡œê±°
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            # ì½˜ì†” ?¸ë“¤??
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            
            # ?Œì¼ ?¸ë“¤??
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # ?¬ë§·??
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_progress(logger: logging.Logger, current: int, total: int, item_type: str = "items"):
        """ì§„í–‰ë¥?ë¡œê¹…"""
        if total > 0:
            percentage = (current / total) * 100
            logger.info(f"Progress: {current}/{total} {item_type} ({percentage:.1f}%)")
    
    @staticmethod
    def log_memory_usage(logger: logging.Logger, memory_mb: float, limit_mb: float):
        """ë©”ëª¨ë¦??¬ìš©??ë¡œê¹…"""
        usage_percent = (memory_mb / limit_mb) * 100
        logger.info(f"Memory: {memory_mb:.1f}MB / {limit_mb}MB ({usage_percent:.1f}%)")


class DataOptimizer:
    """?°ì´??ìµœì ??? í‹¸ë¦¬í‹° ?´ë˜??(ìµœì ?”ëœ ë²„ì „)"""
    
    # ?€?©ëŸ‰ ?„ë“œ ?¬ê¸° ?œí•œ ?¤ì • (???„ê²©?˜ê²Œ)
    FIELD_LIMITS = {
        'content_html': 500_000,        # 1MB ??500KBë¡?ê°ì†Œ
        'precedent_content': 500_000,   # 1MB ??500KBë¡?ê°ì†Œ
        'law_content': 500_000,          # 1MB ??500KBë¡?ê°ì†Œ
        'full_text': 500_000,           # 1MB ??500KBë¡?ê°ì†Œ
        'structured_content': 300_000,  # 500KB ??300KBë¡?ê°ì†Œ
        'case_summary': 50_000,          # ?ˆë¡œ ì¶”ê?: ì¼€?´ìŠ¤ ?”ì•½ 50KB ?œí•œ
        'legal_sections': 100_000,       # ?ˆë¡œ ì¶”ê?: ë²•ì¡°ë¬?100KB ?œí•œ
    }
    
    # ?„ìˆ˜ ?„ë“œë§?? ì??˜ëŠ” ?¤ì • (?„ì „???ë? ?°ì´??? ì?)
    ESSENTIAL_FIELDS = {
        'precedent': [
            'row_number', 'case_name', 'case_number', 'decision_date', 'field', 'court',
            'detail_url', 'structured_content', 'precedent_content', 'content_html',
            'full_text_length', 'extracted_content_length', 'params', 'collected_at',
            'source_url', 'category', 'category_code'
        ],
        'law': [
            'law_name', 'law_number', 'enactment_date', 
            'summary', 'main_content', 'category'
        ]
    }
    
    @classmethod
    def optimize_item(cls, item: Dict[str, Any], data_type: str = 'precedent') -> Dict[str, Any]:
        """
        ?„ì´??ë©”ëª¨ë¦?ìµœì ??(?„ì „???°ì´??? ì? ë²„ì „)
        
        Args:
            item: ìµœì ?”í•  ?„ì´??
            data_type: ?°ì´???€??('precedent' ?ëŠ” 'law')
            
        Returns:
            ìµœì ?”ëœ ?„ì´??
        """
        # ?ë³¸ ?°ì´?°ê? ë¹„ì–´?ˆê±°??? íš¨?˜ì? ?Šìœ¼ë©?ê·¸ë?ë¡?ë°˜í™˜
        if not item or not isinstance(item, dict):
            return item
        
        # ?ë? ?°ì´?°ëŠ” ?„ì „??êµ¬ì¡°ë¥?? ì??˜ë˜ ë©”ëª¨ë¦¬ë§Œ ìµœì ??
        if data_type == 'precedent':
            optimized = item.copy()
        else:
            # ë²•ë¥  ?°ì´?°ë§Œ ?„ìˆ˜ ?„ë“œ ì¶”ì¶œ
            if data_type in cls.ESSENTIAL_FIELDS and item:
                optimized = {}
                for field in cls.ESSENTIAL_FIELDS[data_type]:
                    if field in item and item[field] is not None:
                        optimized[field] = item[field]
                
                # ?„ìˆ˜ ?„ë“œê°€ ?˜ë‚˜???†ìœ¼ë©??ë³¸ ?°ì´??? ì?
                if not optimized:
                    optimized = item.copy()
            else:
                optimized = item.copy()
        
        # ?€?©ëŸ‰ ?„ë“œ ?¬ê¸° ?œí•œ
        for field, limit in cls.FIELD_LIMITS.items():
            if field in optimized and isinstance(optimized[field], str):
                if len(optimized[field]) > limit:
                    optimized[field] = optimized[field][:limit] + "... [TRUNCATED]"
        
        # structured_content ?´ë? ?„ë“œ??ìµœì ??
        if 'structured_content' in optimized and isinstance(optimized['structured_content'], dict):
            structured = optimized['structured_content']
            for key, value in structured.items():
                if isinstance(value, str) and len(value) > 100_000:  # 200KB ??100KBë¡?ê°ì†Œ
                    structured[key] = value[:100_000] + "... [TRUNCATED]"
        
        # ë¶ˆí•„?”í•œ ë¹??„ë“œ ?œê±° (?ë? ?°ì´?°ëŠ” ?œì™¸)
        if data_type != 'precedent':
            optimized = {k: v for k, v in optimized.items() if v is not None and v != ""}
        
        return optimized
    
    @classmethod
    def save_compressed_json(cls, data: Any, filepath: Path) -> bool:
        """
        ?•ì¶•??JSON?¼ë¡œ ?€??
        
        Args:
            data: ?€?¥í•  ?°ì´??
            filepath: ?€?¥í•  ?Œì¼ ê²½ë¡œ
            
        Returns:
            ?€???±ê³µ ?¬ë?
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            return True
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save compressed JSON: {e}")
            return False


class CollectionConfig:
    """?˜ì§‘ ?¤ì • ê´€ë¦??´ë˜??""
    
    # ê¸°ë³¸ ?¤ì •ê°’ë“¤ (ìµœì ?”ëœ ë²„ì „)
    DEFAULT_CONFIG = {
        'memory_limit_mb': 600,  # 400MB ??600MBë¡?ì¡°ì •
        'batch_size': 10,        # 20 ??10?¼ë¡œ ê°ì†Œ
        'page_size': 10,
        'rate_limit': 3.0,
        'timeout': 30000,
        'max_retries': 3,
        'cleanup_threshold': 0.6,  # 0.7 ??0.6?¼ë¡œ ??ë³´ìˆ˜?ìœ¼ë¡?
        'log_level': 'INFO'
    }
    
    def __init__(self, **kwargs):
        """
        ?¤ì • ì´ˆê¸°??
        
        Args:
            **kwargs: ?¤ì • ?¤ë²„?¼ì´??
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """?¤ì •ê°?ì¡°íšŒ"""
        return self.config.get(key, default)
    
    def update(self, **kwargs):
        """?¤ì • ?…ë°?´íŠ¸"""
        self.config.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """?¤ì •???•ì…”?ˆë¦¬ë¡?ë°˜í™˜"""
        return self.config.copy()


def memory_monitor(threshold_mb: float = 600.0):  # 600MBë¡?ì¡°ì •
    """
    ë©”ëª¨ë¦?ëª¨ë‹ˆ?°ë§ ?°ì½”?ˆì´??(ìµœì ?”ëœ ë²„ì „)
    
    Args:
        threshold_mb: ë©”ëª¨ë¦??„ê³„ê°?(MB)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_manager = MemoryManager()
            
            # ?¨ìˆ˜ ?¤í–‰ ??ë©”ëª¨ë¦?ì²´í¬
            if not memory_manager.check_and_cleanup():
                raise MemoryError(f"Memory limit exceeded before {func.__name__}")
            
            result = func(*args, **kwargs)
            
            # ?¨ìˆ˜ ?¤í–‰ ??ë©”ëª¨ë¦?ì²´í¬
            memory_mb = memory_manager.get_memory_usage()
            if memory_mb > threshold_mb:
                logging.getLogger(__name__).warning(
                    f"High memory usage after {func.__name__}: {memory_mb:.1f}MB"
                )
                memory_manager.aggressive_cleanup()
            
            return result
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    ?¤íŒ¨ ???¬ì‹œ???°ì½”?ˆì´??
    
    Args:
        max_retries: ìµœë? ?¬ì‹œ???Ÿìˆ˜
        delay: ?¬ì‹œ??ê°„ê²© (ì´?
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = RetryManager(max_retries=max_retries, base_delay=delay)
            return retry_manager.retry(func, *args, **kwargs)
        return wrapper
    return decorator


# ?¸ì˜ ?¨ìˆ˜??
def get_system_memory_info() -> Dict[str, float]:
    """?œìŠ¤??ë©”ëª¨ë¦??•ë³´ ë°˜í™˜"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    except Exception:
        return {'total_gb': 0, 'available_gb': 0, 'used_percent': 0, 'free_gb': 0}


def check_system_requirements(min_memory_gb: float = 2.0) -> bool:
    """
    ?œìŠ¤???”êµ¬?¬í•­ ?•ì¸
    
    Args:
        min_memory_gb: ìµœì†Œ ?„ìš” ë©”ëª¨ë¦?(GB)
        
    Returns:
        ?”êµ¬?¬í•­ ì¶©ì¡± ?¬ë?
    """
    memory_info = get_system_memory_info()
    
    if memory_info['available_gb'] < min_memory_gb:
        logging.getLogger(__name__).warning(
            f"Low available memory: {memory_info['available_gb']:.1f}GB "
            f"(minimum required: {min_memory_gb}GB)"
        )
        return False
    
    if memory_info['used_percent'] > 80:
        logging.getLogger(__name__).warning(
            f"High memory usage: {memory_info['used_percent']:.1f}%"
        )
        return False
    
    return True
