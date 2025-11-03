#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë²•ë¥  ?˜ì§‘ ?±ëŠ¥ ë©”íŠ¸ë¦??˜ì§‘ê¸?

Prometheus ë©”íŠ¸ë¦?„ ?˜ì§‘?˜ê³  ?¸ì¶œ?˜ëŠ” ëª¨ë“ˆ
"""

from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary
import time
import psutil
import threading
from datetime import datetime
from typing import Optional
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LawCollectionMetrics:
    """ë²•ë¥  ?˜ì§‘ ?±ëŠ¥ ë©”íŠ¸ë¦??˜ì§‘ê¸?""
    
    _instance = None
    _server_started = False
    
    def __new__(cls, port: int = 8000):
        if cls._instance is None:
            cls._instance = super(LawCollectionMetrics, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, port: int = 8000):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.port = port
        self.start_time = time.time()
        
        # ë©”íŠ¸ë¦??Œì¼ ê²½ë¡œ
        self.metrics_file = Path("data/metrics_state.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # ë©”íŠ¸ë¦??•ì˜
        self.pages_processed = Counter(
            'law_collection_pages_processed_total',
            'Total number of pages processed'
        )
        
        self.laws_collected = Counter(
            'law_collection_laws_collected_total',
            'Total number of laws collected'
        )
        
        self.collection_errors = Counter(
            'law_collection_errors_total',
            'Total number of collection errors',
            ['error_type']
        )
        
        self.page_processing_time = Histogram(
            'law_collection_page_processing_seconds',
            'Time spent processing each page',
            buckets=[1, 5, 10, 30, 60, 120, 300, 600]
        )
        
        self.memory_usage = Gauge(
            'law_collection_memory_usage_bytes',
            'Current memory usage of the collection process'
        )
        
        self.cpu_usage = Gauge(
            'law_collection_cpu_usage_percent',
            'Current CPU usage of the collection process'
        )
        
        self.collection_duration = Summary(
            'law_collection_duration_seconds',
            'Total time spent on collection'
        )
        
        self.current_page = Gauge(
            'law_collection_current_page',
            'Current page being processed'
        )
        
        self.throughput = Gauge(
            'law_collection_throughput_laws_per_minute',
            'Current throughput in laws per minute'
        )
        
        self.collection_status = Gauge(
            'law_collection_status',
            'Collection status (0=stopped, 1=running, 2=paused)'
        )
        
        # ?µê³„ ë³€??
        self.total_laws_collected = 0
        self.collection_start_time = None
        self.is_running = False
        
        # ?€?¥ëœ ë©”íŠ¸ë¦??íƒœ ë³µì›
        self._load_metrics_state()
        
        # ?œìŠ¤??ë©”íŠ¸ë¦??˜ì§‘ ?¤ë ˆ??
        self.metrics_thread = threading.Thread(target=self._collect_system_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        
        # ë©”íŠ¸ë¦??íƒœ ?€???¤ë ˆ??
        self.save_thread = threading.Thread(target=self._save_metrics_state)
        self.save_thread.daemon = True
        self.save_thread.start()
    
    def _collect_system_metrics(self):
        """?œìŠ¤??ë©”íŠ¸ë¦??˜ì§‘ (ë°±ê·¸?¼ìš´???¤ë ˆ??"""
        process = psutil.Process()
        while True:
            try:
                if self.is_running:
                    # ë©”ëª¨ë¦??¬ìš©??
                    memory_info = process.memory_info()
                    self.memory_usage.set(memory_info.rss)
                    
                    # CPU ?¬ìš©ë¥?
                    cpu_percent = process.cpu_percent()
                    self.cpu_usage.set(cpu_percent)
                
                time.sleep(5)  # 5ì´ˆë§ˆ???…ë°?´íŠ¸
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(10)
    
    def start_server(self):
        """ë©”íŠ¸ë¦??œë²„ ?œìž‘"""
        if LawCollectionMetrics._server_started:
            logger.info(f"Metrics server already running on port {self.port}")
            return
            
        try:
            start_http_server(self.port)
            LawCollectionMetrics._server_started = True
            logger.info(f"Metrics server started on port {self.port}")
            print(f"Metrics server started on http://localhost:{self.port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def start_collection(self):
        """?˜ì§‘ ?œìž‘"""
        self.collection_start_time = time.time()
        self.is_running = True
        self.collection_status.set(1)  # running
        logger.info("Collection started - metrics tracking enabled")
    
    def stop_collection(self):
        """?˜ì§‘ ì¤‘ì?"""
        self.is_running = False
        self.collection_status.set(0)  # stopped
        
        if self.collection_start_time:
            duration = time.time() - self.collection_start_time
            self.collection_duration.observe(duration)
            logger.info(f"Collection stopped - total duration: {duration:.2f}s")
    
    def pause_collection(self):
        """?˜ì§‘ ?¼ì‹œ ì¤‘ì?"""
        self.is_running = False
        self.collection_status.set(2)  # paused
        logger.info("Collection paused")
    
    def resume_collection(self):
        """?˜ì§‘ ?¬ê°œ"""
        self.is_running = True
        self.collection_status.set(1)  # running
        logger.info("Collection resumed")
    
    def record_page_processed(self, page_number: int):
        """?˜ì´ì§€ ì²˜ë¦¬ ?„ë£Œ ê¸°ë¡"""
        self.pages_processed.inc()
        self.current_page.set(page_number)
        logger.info(f"Page {page_number} processed - Total: {self.pages_processed._value._value}")
    
    def record_laws_collected(self, count: int):
        """?˜ì§‘??ë²•ë¥  ??ê¸°ë¡"""
        self.laws_collected.inc(count)
        self.total_laws_collected += count
        
        # ì²˜ë¦¬??ê³„ì‚° (ë¶„ë‹¹ ë²•ë¥  ??
        if self.collection_start_time:
            elapsed_time = time.time() - self.collection_start_time
            if elapsed_time > 0:
                throughput = (self.total_laws_collected / elapsed_time) * 60
                self.throughput.set(throughput)
        
        logger.info(f"Laws collected: {count} (total: {self.total_laws_collected})")
    
    def record_error(self, error_type: str):
        """?ëŸ¬ ê¸°ë¡"""
        self.collection_errors.labels(error_type=error_type).inc()
        logger.warning(f"Error recorded: {error_type}")
    
    def record_page_processing_time(self, duration: float):
        """?˜ì´ì§€ ì²˜ë¦¬ ?œê°„ ê¸°ë¡"""
        self.page_processing_time.observe(duration)
        logger.debug(f"Page processing time: {duration:.2f}s")
    
    def _load_metrics_state(self):
        """?€?¥ëœ ë©”íŠ¸ë¦??íƒœ ë³µì›"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # ë©”íŠ¸ë¦?ê°?ë³µì›
                self.pages_processed._value._value = state.get('pages_processed', 0)
                self.laws_collected._value._value = state.get('laws_collected', 0)
                self.total_laws_collected = state.get('total_laws_collected', 0)
                
                logger.info(f"Metrics state restored: {state}")
        except Exception as e:
            logger.warning(f"Failed to load metrics state: {e}")
    
    def _save_metrics_state(self):
        """ë©”íŠ¸ë¦??íƒœë¥??Œì¼???€??(ë°±ê·¸?¼ìš´???¤ë ˆ??"""
        while True:
            try:
                state = {
                    'pages_processed': self.pages_processed._value._value,
                    'laws_collected': self.laws_collected._value._value,
                    'total_laws_collected': self.total_laws_collected,
                    'last_updated': time.time()
                }
                
                with open(self.metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                
                time.sleep(30)  # 30ì´ˆë§ˆ???€??
            except Exception as e:
                logger.error(f"Failed to save metrics state: {e}")
                time.sleep(60)
    
    def get_stats(self) -> dict:
        """?„ìž¬ ?µê³„ ë°˜í™˜"""
        return {
            'pages_processed': self.pages_processed._value._value,
            'laws_collected': self.laws_collected._value._value,
            'total_laws_collected': self.total_laws_collected,
            'current_page': self.current_page._value._value,
            'throughput': self.throughput._value._value,
            'memory_usage_mb': self.memory_usage._value._value / 1024 / 1024,
            'cpu_usage_percent': self.cpu_usage._value._value,
            'is_running': self.is_running,
            'uptime_seconds': time.time() - self.start_time
        }


def main():
    """ë©”íŠ¸ë¦??œë²„ ?…ë¦½ ?¤í–‰"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Law Collection Metrics Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to expose metrics')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ë©”íŠ¸ë¦??œë²„ ?œìž‘
    metrics = LawCollectionMetrics(port=args.port)
    metrics.start_server()
    
    print(f"Metrics server running on http://localhost:{args.port}/metrics")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMetrics server stopped")


if __name__ == "__main__":
    main()
