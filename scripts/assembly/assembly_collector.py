# -*- coding: utf-8 -*-
"""
Assembly Collector
메모리 안전 수집기 (날짜/카테고리별 저장)

수집된 데이터를 메모리 효율적으로 저장하고 관리합니다.
- 배치 단위 저장 (50개씩)
- 메모리 사용량 모니터링
- 날짜/카테고리별 디렉토리 구조
- 실패 항목 추적
"""

from pathlib import Path
from datetime import datetime
import json
import logging
import psutil
import gc
from typing import Dict, Any, List, Optional

# 공통 유틸리티 import
try:
    from scripts.assembly.common_utils import DataOptimizer, CollectionLogger
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False

# 압축 모듈 import
try:
    from scripts.assembly.law_data_compressor import compress_law_data
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

logger = logging.getLogger(__name__)


class AssemblyCollector:
    """메모리 안전 수집기 (날짜/카테고리별 저장)"""
    
    def __init__(self, 
                 base_dir: str,
                 data_type: str,  # 'law' or 'precedent'
                 category: Optional[str] = None,  # '민사', '형사' etc
                 batch_size: int = 20,  # 기본값 감소 (50 → 20)
                 memory_limit_mb: int = 600):  # 기본값 감소 (800 → 600)
        """
        수집기 초기화
        
        Args:
            base_dir: 기본 저장 디렉토리
            data_type: 데이터 타입 ('law' 또는 'precedent')
            category: 카테고리 (판례의 경우 '민사', '형사' 등)
            batch_size: 배치 크기
            memory_limit_mb: 메모리 사용량 제한 (MB)
        """
        self.data_type = data_type
        self.category = category
        self.batch_size = batch_size
        self.memory_limit_mb = memory_limit_mb
        
        # 디렉토리 구조: base_dir/data_type/YYYYMMDD/category/
        self.base_dir = Path(base_dir)
        today = datetime.now().strftime("%Y%m%d")
        
        # 날짜별 디렉토리
        self.date_dir = self.base_dir / data_type / today
        
        # 카테고리별 디렉토리 (판례만)
        if category:
            self.output_dir = self.date_dir / category
        else:
            self.output_dir = self.date_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 수집 상태
        self.batch = []
        self.collected_count = 0
        self.failed_items = []
        self.batch_count = 0
        
        # 페이지 정보 추적
        self.current_page = None
        self.page_start_item = 0
        
        self.logger = logging.getLogger(__name__)
        
        print(f"📁 Collector initialized:")
        print(f"   Data type: {data_type}")
        print(f"   Category: {category or 'None'}")
        print(f"   Output dir: {self.output_dir}")
        print(f"   Batch size: {batch_size}")
        print(f"   Memory limit: {memory_limit_mb}MB")
    
    def set_page_info(self, page_number: int):
        """페이지 정보 설정"""
        self.current_page = page_number
        self.page_start_item = self.collected_count
        self.logger.info(f"📄 Page {page_number} started, item count: {self.page_start_item}")
    
    def save_item(self, item: Dict[str, Any]):
        """
        항목 저장 (개선된 버전)
        
        Args:
            item: 저장할 데이터 항목
        """
        # 데이터 타입에 따른 최적화
        if COMMON_UTILS_AVAILABLE:
            optimized_item = DataOptimizer.optimize_item(item, self.data_type)
        else:
            optimized_item = self._optimize_item_memory(item)
        
        # 압축은 판례 데이터에는 적용하지 않음 (법률 데이터만)
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
        아이템 메모리 최적화
        
        Args:
            item: 최적화할 아이템
            
        Returns:
            Dict: 최적화된 아이템
        """
        optimized = item.copy()
        
        # 대용량 필드 크기 제한
        large_fields = ['content_html', 'precedent_content', 'law_content', 'full_text']
        
        for field in large_fields:
            if field in optimized and isinstance(optimized[field], str):
                if len(optimized[field]) > 500000:  # 500KB 제한
                    optimized[field] = optimized[field][:500000] + "... [TRUNCATED]"
                    self.logger.info(f"⚠️ {field} truncated to 500KB")
        
        # structured_content 내부 필드도 최적화
        if 'structured_content' in optimized and isinstance(optimized['structured_content'], dict):
            structured = optimized['structured_content']
            for key, value in structured.items():
                if isinstance(value, str) and len(value) > 200000:  # 200KB 제한
                    structured[key] = value[:200000] + "... [TRUNCATED]"
                    self.logger.info(f"⚠️ structured_content.{key} truncated to 200KB")
        
        return optimized
    
    def _save_batch(self):
        """배치 파일 저장 (페이지 정보 포함)"""
        if not self.batch:
            return
        
        self.batch_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 파일명 구성 (페이지 정보 포함)
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
                # 압축된 JSON 형식으로 저장 (공백 제거)
                json.dump(batch_data, f, ensure_ascii=False, separators=(',', ':'))
            
            print(f"✅ Batch saved: {filename} ({len(self.batch)} items)")
            self.batch = []
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save batch: {e}")
            raise
    
    def _check_memory(self):
        """메모리 체크 및 자동 정리 (최적화된 버전)"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.logger.info(f"📊 Memory: {memory_mb:.1f}MB / {self.memory_limit_mb}MB ({memory_mb/self.memory_limit_mb*100:.1f}%)")
            
            # 동적 배치 크기 조정
            new_batch_size = self._calculate_adaptive_batch_size(memory_mb)
            if new_batch_size != self.batch_size:
                self.logger.info(f"🔄 Batch size adjusted: {self.batch_size} → {new_batch_size}")
                self.batch_size = new_batch_size
            
            # 메모리 사용량이 높으면 강제 정리
            if memory_mb > self.memory_limit_mb * 0.6:  # 60%에서 정리 시작
                self.logger.warning(f"⚠️ High memory ({memory_mb:.1f}MB), forcing cleanup")
                
                # 강제 가비지 컬렉션
                gc.collect()
                
                memory_after = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"✅ After GC: {memory_after:.1f}MB")
                
                if memory_after > self.memory_limit_mb:
                    raise MemoryError(f"Memory limit exceeded: {memory_after:.1f}MB")
                    
        except Exception as e:
            self.logger.error(f"❌ Memory check failed: {e}")
            raise
    
    def _calculate_adaptive_batch_size(self, current_memory_mb: float) -> int:
        """메모리 사용량에 따른 배치 크기 동적 계산"""
        memory_ratio = current_memory_mb / self.memory_limit_mb
        
        if memory_ratio > 0.7:  # 70% 초과시
            return max(5, self.batch_size - 3)
        elif memory_ratio > 0.5:  # 50% 초과시
            return max(8, self.batch_size - 2)
        elif memory_ratio < 0.3:  # 30% 미만시
            return min(20, self.batch_size + 2)
        else:
            return self.batch_size
    
    def add_failed_item(self, item_data: Dict[str, Any], error: str):
        """
        실패 항목 추가
        
        Args:
            item_data: 실패한 항목 데이터
            error: 오류 메시지
        """
        failed_item = {
            'item_data': item_data,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.failed_items.append(failed_item)
        self.logger.warning(f"⚠️ Failed item added: {error}")
    
    def finalize(self):
        """수집 종료 처리"""
        try:
            # 남은 배치 저장
            if self.batch:
                self._save_batch()
            
            # 실패 항목 저장
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
                
                self.logger.info(f"📝 Failed items saved: {fail_file}")
            
            # 최종 메모리 정리
            gc.collect()
            
            # 수집 요약 생성
            summary = self._create_summary()
            summary_file = self.output_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Collection finalized:")
            print(f"   Total collected: {self.collected_count} items")
            print(f"   Failed: {len(self.failed_items)} items")
            print(f"   Batches: {self.batch_count}")
            print(f"   Summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Finalization failed: {e}")
            raise
    
    def _create_summary(self) -> Dict[str, Any]:
        """수집 요약 생성"""
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
        """현재 수집 통계 반환"""
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
        """수집 시작 시간 설정"""
        self.start_time = start_time
