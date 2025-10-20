#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assembly Collection Performance Test
국회 수집 스크립트 성능 테스트 및 벤치마크

이 스크립트는 수집 스크립트들의 성능을 테스트하고 벤치마크를 제공합니다.
"""

import time
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(project_root))

from scripts.assembly.common_utils import (
    MemoryManager, CollectionConfig, CollectionLogger,
    get_system_memory_info, check_system_requirements
)


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        """성능 모니터 초기화"""
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.logger = CollectionLogger.setup_logging("performance_test")
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.end_time = time.time()
        self.logger.info("Performance monitoring stopped")
    
    def sample_system_metrics(self):
        """시스템 메트릭 샘플링"""
        try:
            # 메모리 사용량
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            self.logger.error(f"Failed to sample metrics: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        if not self.start_time or not self.end_time:
            return {"error": "Monitoring not completed"}
        
        duration = self.end_time - self.start_time
        
        # 메모리 통계
        memory_stats = {}
        if self.memory_samples:
            memory_stats = {
                "min_mb": min(self.memory_samples),
                "max_mb": max(self.memory_samples),
                "avg_mb": sum(self.memory_samples) / len(self.memory_samples),
                "samples": len(self.memory_samples)
            }
        
        # CPU 통계
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
    """수집 성능 벤치마크 클래스"""
    
    def __init__(self):
        """벤치마크 초기화"""
        self.logger = CollectionLogger.setup_logging("collection_benchmark")
        self.results = []
    
    def benchmark_memory_manager(self, iterations: int = 100) -> Dict[str, Any]:
        """메모리 매니저 벤치마크"""
        self.logger.info(f"Benchmarking MemoryManager with {iterations} iterations")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # 메모리 매니저 테스트
        memory_manager = MemoryManager(memory_limit_mb=600)
        
        for i in range(iterations):
            # 메모리 사용량 체크
            memory_manager.get_memory_usage()
            
            # 주기적으로 샘플링
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
        """데이터 최적화 벤치마크"""
        self.logger.info(f"Benchmarking DataOptimizer with {test_data_size} items")
        
        from scripts.assembly.common_utils import DataOptimizer
        
        # 테스트 데이터 생성
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
        
        # 데이터 최적화 테스트
        optimized_items = []
        for i, item in enumerate(test_items):
            optimized_item = DataOptimizer.optimize_item(item)
            optimized_items.append(optimized_item)
            
            # 주기적으로 샘플링
            if i % 100 == 0:
                monitor.sample_system_metrics()
        
        monitor.stop_monitoring()
        
        # 크기 비교
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
        """수집 설정 벤치마크"""
        self.logger.info("Benchmarking CollectionConfig")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # 설정 생성 및 조회 테스트
        configs = []
        for i in range(1000):
            config = CollectionConfig(
                memory_limit_mb=600 + i,
                batch_size=20 + (i % 10),
                max_retries=3 + (i % 3)
            )
            configs.append(config)
        
        # 설정 조회 테스트
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
        """모든 벤치마크 실행"""
        self.logger.info("Starting comprehensive benchmark suite")
        
        # 시스템 요구사항 확인
        if not check_system_requirements(min_memory_gb=2.0):
            self.logger.warning("System requirements not met, proceeding with caution")
        
        # 벤치마크 실행
        self.benchmark_memory_manager(iterations=100)
        self.benchmark_data_optimizer(test_data_size=500)
        self.benchmark_collection_config()
        
        self.logger.info(f"Completed {len(self.results)} benchmarks")
        return self.results
    
    def save_results(self, output_file: Path):
        """결과 저장"""
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
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for result in self.results:
            test_name = result["test_name"]
            performance = result.get("performance", {})
            duration = performance.get("duration_seconds", 0)
            
            print(f"\n🔬 {test_name}:")
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
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Assembly Collection Performance Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_test.py --all                    # 모든 벤치마크 실행
  python performance_test.py --memory-manager         # 메모리 매니저만 테스트
  python performance_test.py --data-optimizer         # 데이터 최적화만 테스트
  python performance_test.py --config                # 설정 관리만 테스트
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='모든 벤치마크 실행')
    parser.add_argument('--memory-manager', action='store_true',
                        help='메모리 매니저 벤치마크')
    parser.add_argument('--data-optimizer', action='store_true',
                        help='데이터 최적화 벤치마크')
    parser.add_argument('--config', action='store_true',
                        help='설정 관리 벤치마크')
    parser.add_argument('--iterations', type=int, default=100,
                        help='반복 횟수 (기본: 100)')
    parser.add_argument('--data-size', type=int, default=500,
                        help='테스트 데이터 크기 (기본: 500)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='결과 저장 파일 (기본: benchmark_results.json)')
    
    args = parser.parse_args()
    
    # 벤치마크 실행
    benchmark = CollectionBenchmark()
    
    if args.all or not any([args.memory_manager, args.data_optimizer, args.config]):
        # 모든 벤치마크 실행
        benchmark.run_all_benchmarks()
    else:
        # 선택적 벤치마크 실행
        if args.memory_manager:
            benchmark.benchmark_memory_manager(args.iterations)
        if args.data_optimizer:
            benchmark.benchmark_data_optimizer(args.data_size)
        if args.config:
            benchmark.benchmark_collection_config()
    
    # 결과 저장 및 출력
    output_file = Path(args.output)
    benchmark.save_results(output_file)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
