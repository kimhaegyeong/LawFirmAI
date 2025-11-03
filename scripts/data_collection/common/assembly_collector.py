# -*- coding: utf-8 -*-
"""
Assembly Collector
ë©”ëª¨ë¦??ˆì „ ?˜ì§‘ê¸?(? ì§œ/ì¹´í…Œê³ ë¦¬ë³??€??

?˜ì§‘???°ì´?°ë? ë©”ëª¨ë¦??¨ìœ¨?ìœ¼ë¡??€?¥í•˜ê³?ê´€ë¦¬í•©?ˆë‹¤.
- ë°°ì¹˜ ?¨ìœ„ ?€??(50ê°œì”©)
- ë©”ëª¨ë¦??¬ìš©??ëª¨ë‹ˆ?°ë§
- ? ì§œ/ì¹´í…Œê³ ë¦¬ë³??”ë ‰? ë¦¬ êµ¬ì¡°
- ?¤íŒ¨ ??ª© ì¶”ì 
"""

from pathlib import Path
from datetime import datetime
import json
import logging
import psutil
import gc
from typing import Dict, Any, List, Optional

# ê³µí†µ ? í‹¸ë¦¬í‹° import
try:
    from scripts.assembly.common_utils import DataOptimizer, CollectionLogger
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False

# ?•ì¶• ëª¨ë“ˆ import
try:
    from scripts.assembly.law_data_compressor import compress_law_data
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

logger = logging.getLogger(__name__)


class AssemblyCollector:
    """ë©”ëª¨ë¦??ˆì „ ?˜ì§‘ê¸?(? ì§œ/ì¹´í…Œê³ ë¦¬ë³??€??"""
    
    def __init__(self, 
                 base_dir: str,
                 data_type: str,  # 'law' or 'precedent'
                 category: Optional[str] = None,  # 'ë¯¼ì‚¬', '?•ì‚¬' etc
                 batch_size: int = 20,  # ê¸°ë³¸ê°?ê°ì†Œ (50 ??20)
                 memory_limit_mb: int = 600):  # ê¸°ë³¸ê°?ê°ì†Œ (800 ??600)
        """
        ?˜ì§‘ê¸?ì´ˆê¸°??
        
        Args:
            base_dir: ê¸°ë³¸ ?€???”ë ‰? ë¦¬
            data_type: ?°ì´???€??('law' ?ëŠ” 'precedent')
            category: ì¹´í…Œê³ ë¦¬ (?ë???ê²½ìš° 'ë¯¼ì‚¬', '?•ì‚¬' ??
            batch_size: ë°°ì¹˜ ?¬ê¸°
            memory_limit_mb: ë©”ëª¨ë¦??¬ìš©???œí•œ (MB)
        """
        self.data_type = data_type
        self.category = category
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        
        # ?”ë ‰? ë¦¬ êµ¬ì¡°: base_dir/data_type/YYYYMMDD/category/
        self.base_dir = Path(base_dir)
        today = datetime.now().strftime("%Y%m%d")
        
        # ? ì§œë³??”ë ‰? ë¦¬
        self.date_dir = self.base_dir / data_type / today
        
        # ì¹´í…Œê³ ë¦¬ë³??”ë ‰? ë¦¬ (?ë?ë§?
        if category:
            self.output_dir = self.date_dir / category
        else:
            self.output_dir = self.date_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?˜ì§‘ ?íƒœ
        self.batch = []
        self.collected_count = 0
        self.failed_items = []
        self.batch_count = 0
        
        # ?˜ì´ì§€ ?•ë³´ ì¶”ì 
        self.current_page = None
        self.page_start_item = 0
        
        self.logger = logging.getLogger(__name__)
        
        print(f"?“ Collector initialized:")
        print(f"   Data type: {data_type}")
        print(f"   Category: {category or 'None'}")
        print(f"   Output dir: {self.output_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Memory limit: {memory_limit_mb}MB")
    
    def set_page_info(self, page_number: int):
        """?˜ì´ì§€ ?•ë³´ ?¤ì •"""
        self.current_page = page_number
        self.page_start_item = self.collected_count
        self.logger.info(f"?“„ Page {page_number} started, item count: {self.page_start_item}")
    
    def save_item(self, item: Dict[str, Any]):
        """
        ??ª© ?€??(ê°œì„ ??ë²„ì „)
        
        Args:
            item: ?€?¥í•  ?°ì´????ª©
        """
        # ?°ì´???€?…ì— ?°ë¥¸ ìµœì ??
        if COMMON_UTILS_AVAILABLE:
            optimized_item = DataOptimizer.optimize_item(item, self.data_type)
        else:
            optimized_item = self._optimize_item_memory(item)
        
        # ?•ì¶•?€ ?ë? ?°ì´?°ì—???ìš©?˜ì? ?ŠìŒ (ë²•ë¥  ?°ì´?°ë§Œ)
        if COMPRESSION_AVAILABLE and self.data_type == 'law':
            compressed_item = compress_law_data(optimized_item)
            self.batch.append(compressed_item)
        else:
            self.batch.append(optimized_item)
        
        self.collected_count += 1
        
        if len(self.batch) >= self.batch_size:
            self._save_batch()
            self._check_memory()
    
    def _optimize_item_memory(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        ?„ì´??ë©”ëª¨ë¦?ìµœì ??
        
        Args:
            item: ìµœì ?”í•  ?„ì´??
            
        Returns:
            Dict: ìµœì ?”ëœ ?„ì´??
        """
        optimized = item.copy()
        
        # ?€?©ëŸ‰ ?„ë“œ ?¬ê¸° ?œí•œ
        large_fields = ['content_html', 'precedent_content', 'law_content', 'full_text']
        
        for field in large_fields:
            if field in optimized and isinstance(optimized[field], str):
                if len(optimized[field]) > 500000:  # 500KB ?œí•œ
                    optimized[field] = optimized[field][:500000] + "... [TRUNCATED]"
                    self.logger.info(f"? ï¸ {field} truncated to 500KB")
        
        # structured_content ?´ë? ?„ë“œ??ìµœì ??
        if 'structured_content' in optimized and isinstance(optimized['structured_content'], dict):
            structured = optimized['structured_content']
            for key, value in structured.items():
                if isinstance(value, str) and len(value) > 200000:  # 200KB ?œí•œ
                    structured[key] = value[:200000] + "... [TRUNCATED]"
                    self.logger.info(f"? ï¸ structured_content.{key} truncated to 200KB")
        
        return optimized
    
    def _save_batch(self):
        """ë°°ì¹˜ ?Œì¼ ?€??(?˜ì´ì§€ ?•ë³´ ?¬í•¨)"""
        if not self.batch:
            return
        
        self.batch_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ?Œì¼ëª?êµ¬ì„± (?˜ì´ì§€ ?•ë³´ ?¬í•¨)
        if self.category and self.current_page is not None:
            filename = f"{self.data_type}_{self.category}_page_{self.current_page:03d}_{timestamp}_{len(self.batch)}.json"
        elif self.category:
            filename = f"{self.data_type}_{self.category}_{timestamp}_{len(self.batch)}.json"
        elif self.current_page is not None:
            filename = f"{self.data_type}_page_{self.current_page:03d}_{timestamp}_{len(self.batch)}.json"
        else:
            filename = f"{self.data_type}_{timestamp}_{len(self.batch)}.json"
        
        filepath = self.output_dir / filename
        
        try:
            batch_data = {
                'metadata': {
                    'data_type': self.data_type,
                    'category': self.category,
                    'page_number': self.current_page,
                    'page_start_item': self.page_start_item,
                    'batch_number': self.batch_count,
                    'count': len(self.batch),
                    'collected_at': datetime.now().isoformat(),
                    'file_version': '1.0',
                    'total_collected': self.collected_count
                },
                'items': self.batch
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # ?•ì¶•??JSON ?•ì‹?¼ë¡œ ?€??(ê³µë°± ?œê±°)
                json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))
            
            print(f"??Batch saved: {filename} ({len(self.batch)} items)")
            self.batch = []
            
        except Exception as e:
            self.logger.error(f"??Failed to save batch: {e}")
            raise
    
    def _check_memory(self):
        """ë©”ëª¨ë¦?ì²´í¬ ë°??ë™ ?•ë¦¬ (ìµœì ?”ëœ ë²„ì „)"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.logger.info(f"?“Š Memory: {memory_mb:.1f}MB / {self.memory_limit_mb}MB ({memory_mb/self.memory_limit_mb*100:.1f}%)")
            
            # ?™ì  ë°°ì¹˜ ?¬ê¸° ì¡°ì •
            new_batch_size = self._calculate_adaptive_batch_size(memory_mb)
            if new_batch_size != self.batch_size:
                self.logger.info(f"?”„ Batch size adjusted: {self.batch_size} ??{new_batch_size}")
                self.batch_size = new_batch_size
            
            # ë©”ëª¨ë¦??¬ìš©?‰ì´ ?’ìœ¼ë©?ê°•ì œ ?•ë¦¬
            if memory_mb > self.memory_limit_mb * 0.6:  # 60%?ì„œ ?•ë¦¬ ?œìž‘
                self.logger.warning(f"? ï¸ High memory ({memory_mb:.1f}MB), forcing cleanup")
                
                # ê°•ì œ ê°€ë¹„ì? ì»¬ë ‰??
                gc.collect()
                
                memory_after = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"??After GC: {memory_after:.1f}MB")
                
                if memory_after > self.memory_limit_mb:
                    raise MemoryError(f"Memory limit exceeded: {memory_after:.1f}MB")
                    
        except Exception as e:
            self.logger.error(f"??Memory check failed: {e}")
            raise
    
    def _calculate_adaptive_batch_size(self, current_memory_mb: float) -> int:
        """ë©”ëª¨ë¦??¬ìš©?‰ì— ?°ë¥¸ ë°°ì¹˜ ?¬ê¸° ?™ì  ê³„ì‚°"""
        memory_ratio = current_memory_mb / self.memory_limit_mb
        
        if memory_ratio > 0.7:  # 70% ì´ˆê³¼??
            return max(5, self.batch_size - 3)
        elif memory_ratio > 0.5:  # 50% ì´ˆê³¼??
            return max(8, self.batch_size - 2)
        elif memory_ratio < 0.3:  # 30% ë¯¸ë§Œ??
            return min(20, self.batch_size + 2)
        else:
            return self.batch_size
    
    def add_failed_item(self, item_data: Dict[str, Any], error: str):
        """
        ?¤íŒ¨ ??ª© ì¶”ê?
        
        Args:
            item_data: ?¤íŒ¨????ª© ?°ì´??
            error: ?¤ë¥˜ ë©”ì‹œì§€
        """
        failed_item = {
            'item_data': item_data,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.failed_items.append(failed_item)
        self.logger.warning(f"? ï¸ Failed item added: {error}")
    
    def finalize(self):
        """?˜ì§‘ ì¢…ë£Œ ì²˜ë¦¬"""
        try:
            # ?¨ì? ë°°ì¹˜ ?€??
            if self.batch:
                self._save_batch()
            
            # ?¤íŒ¨ ??ª© ?€??
            if self.failed_items:
                fail_file = self.output_dir / f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                fail_data = {
                    'metadata': {
                        'data_type': self.data_type,
                        'category': self.category,
                        'failed_count': len(self.failed_items),
                        'created_at': datetime.now().isoformat()
                    },
                    'failed_items': self.failed_items
                }
                
                with open(fail_file, 'w', encoding='utf-8') as f:
                    json.dump(fail_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"?“ Failed items saved: {fail_file}")
            
            # ìµœì¢… ë©”ëª¨ë¦??•ë¦¬
            gc.collect()
            
            # ?˜ì§‘ ?”ì•½ ?ì„±
            summary = self._create_summary()
            summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"??Collection finalized:")
            print(f"   Total collected: {self.collected_count} items")
            print(f"   Failed: {len(self.failed_items)} items")
            print(f"   Batches: {self.batch_count}")
            print(f"   Summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"??Finalization failed: {e}")
            raise
    
    def _create_summary(self) -> Dict[str, Any]:
        """?˜ì§‘ ?”ì•½ ?ì„±"""
        return {
            'collection_info': {
                'data_type': self.data_type,
                'category': self.category,
                'start_time': getattr(self, 'start_time', None),
                'end_time': datetime.now().isoformat(),
                'total_collected': self.collected_count,
                'total_failed': len(self.failed_items),
                'batch_count': self.batch_count,
                'batch_size': self.batch_size,
                'memory_limit_mb': self.memory_limit_mb
            },
            'output_directory': str(self.output_dir),
            'files_created': {
                'batch_files': self.batch_count,
                'failed_file': 1 if self.failed_items else 0,
                'summary_file': 1
            },
            'statistics': {
                'success_rate': self.collected_count / (self.collected_count + len(self.failed_items)) if (self.collected_count + len(self.failed_items)) > 0 else 0,
                'average_batch_size': self.collected_count / self.batch_count if self.batch_count > 0 else 0
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """?„ìž¬ ?˜ì§‘ ?µê³„ ë°˜í™˜"""
        return {
            'collected_count': self.collected_count,
            'failed_count': len(self.failed_items),
            'batch_count': self.batch_count,
            'current_batch_size': len(self.batch),
            'data_type': self.data_type,
            'category': self.category,
            'output_dir': str(self.output_dir)
        }
    
    def set_start_time(self, start_time: str):
        """?˜ì§‘ ?œìž‘ ?œê°„ ?¤ì •"""
        self.start_time = start_time
