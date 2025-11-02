#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 메모리 모니터링 도구
LawFirmAI 프로젝트 - TASK 3.1 훈련 환경 구성
"""

import psutil
import torch
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

class GPUMemoryMonitor:
    """GPU 메모리 모니터링 클래스"""
    
    def __init__(self, log_file: str = "logs/gpu_memory.log"):
        self.log_file = log_file
        self.setup_logging()
        self.monitoring = False
        self.memory_history = []
        
    def setup_logging(self):
        """로깅 설정"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_system_memory_info(self) -> Dict:
        """시스템 메모리 정보 수집"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percentage": memory.percent
        }
    
    def get_gpu_memory_info(self) -> Optional[Dict]:
        """GPU 메모리 정보 수집 (CUDA 사용 가능한 경우)"""
        if not torch.cuda.is_available():
            return None
            
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)
        
        return {
            "total_gb": round(gpu_memory / (1024**3), 2),
            "allocated_gb": round(gpu_allocated / (1024**3), 2),
            "cached_gb": round(gpu_cached / (1024**3), 2),
            "free_gb": round((gpu_memory - gpu_allocated) / (1024**3), 2),
            "utilization_percent": round((gpu_allocated / gpu_memory) * 100, 2)
        }
    
    def get_memory_status(self) -> Dict:
        """전체 메모리 상태 수집"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_memory": self.get_system_memory_info(),
            "gpu_memory": self.get_gpu_memory_info()
        }
        
        # 메모리 히스토리에 추가
        self.memory_history.append(status)
        
        # 최근 100개만 유지
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        return status
    
    def log_memory_status(self):
        """메모리 상태 로깅"""
        status = self.get_memory_status()
        
        # 시스템 메모리 로깅
        sys_mem = status["system_memory"]
        self.logger.info(f"System Memory: {sys_mem['used_gb']:.2f}GB/{sys_mem['total_gb']:.2f}GB ({sys_mem['percentage']:.1f}%)")
        
        # GPU 메모리 로깅 (CUDA 사용 가능한 경우)
        if status["gpu_memory"]:
            gpu_mem = status["gpu_memory"]
            self.logger.info(f"GPU Memory: {gpu_mem['allocated_gb']:.2f}GB/{gpu_mem['total_gb']:.2f}GB ({gpu_mem['utilization_percent']:.1f}%)")
        else:
            self.logger.info("GPU Memory: CUDA not available")
    
    def start_monitoring(self, interval: int = 30):
        """메모리 모니터링 시작"""
        self.monitoring = True
        self.logger.info(f"Starting GPU memory monitoring (interval: {interval}s)")
        
        try:
            while self.monitoring:
                self.log_memory_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring = False
        self.logger.info("Monitoring stopped")
    
    def get_memory_summary(self) -> Dict:
        """메모리 사용량 요약"""
        if not self.memory_history:
            return {"error": "No monitoring data available"}
        
        # 최근 상태
        latest = self.memory_history[-1]
        
        # 평균 사용량 계산
        if len(self.memory_history) > 1:
            avg_sys_usage = sum([h["system_memory"]["percentage"] for h in self.memory_history]) / len(self.memory_history)
            
            gpu_data = [h["gpu_memory"] for h in self.memory_history if h["gpu_memory"]]
            avg_gpu_usage = 0
            if gpu_data:
                avg_gpu_usage = sum([g["utilization_percent"] for g in gpu_data]) / len(gpu_data)
        else:
            avg_sys_usage = latest["system_memory"]["percentage"]
            avg_gpu_usage = latest["gpu_memory"]["utilization_percent"] if latest["gpu_memory"] else 0
        
        return {
            "monitoring_duration": len(self.memory_history),
            "latest_status": latest,
            "average_system_usage": round(avg_sys_usage, 2),
            "average_gpu_usage": round(avg_gpu_usage, 2),
            "peak_system_usage": max([h["system_memory"]["percentage"] for h in self.memory_history]),
            "peak_gpu_usage": max([h["gpu_memory"]["utilization_percent"] for h in self.memory_history if h["gpu_memory"]]) if any(h["gpu_memory"] for h in self.memory_history) else 0
        }
    
    def save_memory_report(self, output_file: str = "logs/memory_report.json"):
        """메모리 사용량 보고서 저장"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_memory_summary(),
            "history": self.memory_history
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Memory report saved to {output_file}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Memory Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=0, help="Monitoring duration in seconds (0 = infinite)")
    parser.add_argument("--report", action="store_true", help="Generate memory report")
    
    args = parser.parse_args()
    
    monitor = GPUMemoryMonitor()
    
    if args.report:
        # 기존 데이터로 보고서 생성
        monitor.save_memory_report()
        return
    
    try:
        if args.duration > 0:
            # 제한된 시간 동안 모니터링
            start_time = time.time()
            while time.time() - start_time < args.duration:
                monitor.log_memory_status()
                time.sleep(args.interval)
        else:
            # 무한 모니터링
            monitor.start_monitoring(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        # 최종 보고서 생성
        monitor.save_memory_report()

if __name__ == "__main__":
    main()
