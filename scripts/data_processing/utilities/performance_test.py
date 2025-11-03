#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Collection Performance Test
�?�� ?�집 ?�크립트 ?�능 ?�스??�?벤치마크

???�크립트???�집 ?�크립트?�의 ?�능???�스?�하�?벤치마크�??�공?�니??
"""

import time
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# ?�로?�트 루트�?Python 경로??추�?
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(project_root))

from scripts.assembly.common_utils import (
    MemoryManager, CollectionConfig, CollectionLogger,
    get_system_memory_info, check_system_requirements
)


class PerformanceMonitor:
    """?�능 모니?�링 ?�래??""
    
    def __init__(self):
        """?�능 모니??초기??""
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.logger = CollectionLogger.setup_logging("performance_test")
    
    def start_monitoring(self):
        """모니?�링 ?�작"""
        self.start_time = time.time()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니?�링 중�?"""
        self.end_time = time.time()
        self.logger.info("Performance monitoring stopped")
    
    def sample_system_metrics(self):
        """?�스??메트�??�플�?""
        try:
            # 메모�??�용??
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            
            # CPU ?�용�?
            cpu_percent = psutil.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            self.logger.error(f"Failed to sample metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """?�능 리포???�성"""
        if not self.start_time or not self.end_time:
            return {"error": "Monitoring not completed"}
        
        duration = self.end_time - self.start_time
        
        # 메모�??�계
        memory_stats = {}
        if self.memory_samples:
            memory_stats = {
                "min_mb": min(self.memory_samples),
                "max_mb": max(self.memory_samples),
                "avg_mb": sum(self.memory_samples) / len(self.memory_samples),
                "samples": len(self.memory_samples)
            }
        
        # CPU ?�계
        cpu_stats = {}
        if self.cpu_samples:
            cpu_stats = {
                "min_percent": min(self.cpu_samples),
                "max_percent": max(self.cpu_samples),
                "avg_percent": sum(self.cpu_samples) / len(self.cpu_samples),
                "samples": len(self.cpu_samples)
            }
        
        return {
            "duration_seconds": duration,
            "memory_stats": memory_stats,
            "cpu_stats": cpu_stats,
            "system_info": get_system_memory_info(),
            "timestamp": datetime.now().isoformat()
        }


class CollectionBenchmark:
    """?�집 ?�능 벤치마크 ?�래??""
    
    def __init__(self):
        """벤치마크 초기??""
        self.logger = CollectionLogger.setup_logging("collection_benchmark")
        self.results = []
    
    def benchmark_memory_manager(self, iterations: int = 100) -> Dict[str, Any]:
        """메모�?매니?� 벤치마크"""
        self.logger.info(f"Benchmarking MemoryManager with {iterations} iterations")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # 메모�?매니?� ?�스??
        memory_manager = MemoryManager(memory_limit_mb=600)
        
        for i in range(iterations):
            # 메모�??�용??체크
            memory_manager.get_memory_usage()
            
            # 주기?�으�??�플�?
            if i % 10 == 0:
                monitor.sample_system_metrics()
        
        monitor.stop_monitoring()
        
        result = {
            "test_name": "MemoryManager",
            "iterations": iterations,
            "performance": monitor.get_performance_report()
        }
        
        self.results.append(result)
        return result
    
    def benchmark_data_optimizer(self, test_data_size: int = 1000) -> Dict[str, Any]:
        """?�이??최적??벤치마크"""
        self.logger.info(f"Benchmarking DataOptimizer with {test_data_size} items")
        
        from scripts.assembly.common_utils import DataOptimizer
        
        # ?�스???�이???�성
        test_items = []
        for i in range(test_data_size):
            test_items.append({
                'content_html': 'x' * 2000000,  # 2MB HTML
                'precedent_content': 'y' * 1500000,  # 1.5MB content
                'structured_content': {
                    'full_text': 'z' * 1000000,  # 1MB text
                    'case_info': 'a' * 100000
                }
            })
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # ?�이??최적???�스??
        optimized_items = []
        for i, item in enumerate(test_items):
            optimized_item = DataOptimizer.optimize_item(item)
            optimized_items.append(optimized_item)
            
            # 주기?�으�??�플�?
            if i % 100 == 0:
                monitor.sample_system_metrics()
        
        monitor.stop_monitoring()
        
        # ?�기 비교
        original_size = sum(len(str(item)) for item in test_items)
        optimized_size = sum(len(str(item)) for item in optimized_items)
        compression_ratio = optimized_size / original_size if original_size > 0 else 0
        
        result = {
            "test_name": "DataOptimizer",
            "items_processed": test_data_size,
            "original_size_bytes": original_size,
            "optimized_size_bytes": optimized_size,
            "compression_ratio": compression_ratio,
            "performance": monitor.get_performance_report()
        }
        
        self.results.append(result)
        return result
    
    def benchmark_collection_config(self) -> Dict[str, Any]:
        """?�집 ?�정 벤치마크"""
        self.logger.info("Benchmarking CollectionConfig")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # ?�정 ?�성 �?조회 ?�스??
        configs = []
        for i in range(1000):
            config = CollectionConfig(
                memory_limit_mb=600 + i,
                batch_size=20 + (i % 10),
                max_retries=3 + (i % 3)
            )
            configs.append(config)
        
        # ?�정 조회 ?�스??
        for config in configs:
            _ = config.get('memory_limit_mb')
            _ = config.get('batch_size')
            _ = config.get('max_retries')
        
        monitor.stop_monitoring()
        
        result = {
            "test_name": "CollectionConfig",
            "configs_created": len(configs),
            "performance": monitor.get_performance_report()
        }
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """모든 벤치마크 ?�행"""
        self.logger.info("Starting comprehensive benchmark suite")
        
        # ?�스???�구?�항 ?�인
        if not check_system_requirements(min_memory_gb=2.0):
            self.logger.warning("System requirements not met, proceeding with caution")
        
        # 벤치마크 ?�행
        self.benchmark_memory_manager(iterations=100)
        self.benchmark_data_optimizer(test_data_size=500)
        self.benchmark_collection_config()
        
        self.logger.info(f"Completed {len(self.results)} benchmarks")
        return self.results
    
    def save_results(self, output_file: Path):
        """결과 ?�??""
        results_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "system_info": get_system_memory_info()
            },
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self):
        """결과 ?�약 출력"""
        print("\n" + "="*60)
        print("?�� BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for result in self.results:
            test_name = result["test_name"]
            performance = result.get("performance", {})
            duration = performance.get("duration_seconds", 0)
            
            print(f"\n?�� {test_name}:")
            print(f"   Duration: {duration:.3f} seconds")
            
            if "memory_stats" in performance:
                mem_stats = performance["memory_stats"]
                print(f"   Memory: {mem_stats.get('avg_mb', 0):.1f}MB avg "
                      f"({mem_stats.get('min_mb', 0):.1f}-{mem_stats.get('max_mb', 0):.1f}MB)")
            
            if "compression_ratio" in result:
                ratio = result["compression_ratio"]
                print(f"   Compression: {ratio:.2%} of original size")
        
        print("\n" + "="*60)


def main():
    """메인 ?�수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Assembly Collection Performance Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_test.py --all                    # 모든 벤치마크 ?�행
  python performance_test.py --memory-manager         # 메모�?매니?��??�스??
  python performance_test.py --data-optimizer         # ?�이??최적?�만 ?�스??
  python performance_test.py --config                # ?�정 관리만 ?�스??
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='모든 벤치마크 ?�행')
    parser.add_argument('--memory-manager', action='store_true',
                        help='메모�?매니?� 벤치마크')
    parser.add_argument('--data-optimizer', action='store_true',
                        help='?�이??최적??벤치마크')
    parser.add_argument('--config', action='store_true',
                        help='?�정 관�?벤치마크')
    parser.add_argument('--iterations', type=int, default=100,
                        help='반복 ?�수 (기본: 100)')
    parser.add_argument('--data-size', type=int, default=500,
                        help='?�스???�이???�기 (기본: 500)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='결과 ?�???�일 (기본: benchmark_results.json)')
    
    args = parser.parse_args()
    
    # 벤치마크 ?�행
    benchmark = CollectionBenchmark()
    
    if args.all or not any([args.memory_manager, args.data_optimizer, args.config]):
        # 모든 벤치마크 ?�행
        benchmark.run_all_benchmarks()
    else:
        # ?�택??벤치마크 ?�행
        if args.memory_manager:
            benchmark.benchmark_memory_manager(args.iterations)
        if args.data_optimizer:
            benchmark.benchmark_data_optimizer(args.data_size)
        if args.config:
            benchmark.benchmark_collection_config()
    
    # 결과 ?�??�?출력
    output_file = Path(args.output)
    benchmark.save_results(output_file)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
