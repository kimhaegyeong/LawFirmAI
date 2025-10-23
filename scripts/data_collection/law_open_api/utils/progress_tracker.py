#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
진행상황 추적 유틸리티

수집 작업의 실시간 진행상황을 추적하고 표시하는 모듈입니다.
- 실시간 진행률 표시
- ETA (예상 완료 시간) 계산
- 속도 모니터링
- 진행상황 로깅
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading

logger = logging.getLogger(__name__)


class ProgressTracker:
    """진행상황 추적기"""
    
    def __init__(self, total_items: int, item_name: str = "항목", 
                 update_interval: float = 1.0):
        """
        진행상황 추적기 초기화
        
        Args:
            total_items: 전체 항목 수
            item_name: 항목 이름
            update_interval: 업데이트 간격 (초)
        """
        self.total_items = total_items
        self.item_name = item_name
        self.update_interval = update_interval
        
        # 진행상황 상태
        self.current_count = 0
        self.start_time = None
        self.last_update_time = None
        self.last_count = 0
        
        # 속도 계산
        self.speed_history = []
        self.max_speed_history = 10
        
        # 오류 통계
        self.error_count = 0
        self.retry_count = 0
        
        # 스레드 안전성을 위한 락
        self.lock = threading.Lock()
        
        logger.info(f"ProgressTracker 초기화: {total_items:,}개 {item_name}")
    
    def start(self) -> None:
        """진행상황 추적 시작"""
        with self.lock:
            self.start_time = datetime.now()
            self.last_update_time = self.start_time
            self.last_count = 0
            
        logger.info(f"진행상황 추적 시작: {self.total_items:,}개 {self.item_name}")
        self._print_initial_status()
    
    def update(self, count: int, error_count: int = 0, retry_count: int = 0) -> None:
        """
        진행상황 업데이트
        
        Args:
            count: 현재 완료된 항목 수
            error_count: 오류 발생 횟수
            retry_count: 재시도 횟수
        """
        with self.lock:
            current_time = datetime.now()
            
            # 카운트 업데이트
            self.current_count = min(count, self.total_items)
            self.error_count = error_count
            self.retry_count = retry_count
            
            # 속도 계산
            if self.last_update_time and self.start_time:
                time_diff = (current_time - self.last_update_time).total_seconds()
                count_diff = self.current_count - self.last_count
                
                if time_diff > 0 and count_diff > 0:
                    speed = count_diff / time_diff
                    self.speed_history.append(speed)
                    
                    # 최대 히스토리 수 유지
                    if len(self.speed_history) > self.max_speed_history:
                        self.speed_history = self.speed_history[-self.max_speed_history:]
            
            self.last_update_time = current_time
            self.last_count = self.current_count
            
            # 주기적으로 상태 출력
            if self._should_update_display():
                self._print_progress_status()
    
    def increment(self, increment_by: int = 1, error_count: int = 0, retry_count: int = 0) -> None:
        """
        진행상황 증가
        
        Args:
            increment_by: 증가할 항목 수
            error_count: 오류 발생 횟수
            retry_count: 재시도 횟수
        """
        self.update(self.current_count + increment_by, error_count, retry_count)
    
    def complete(self) -> None:
        """진행상황 완료"""
        with self.lock:
            self.current_count = self.total_items
            
        self._print_completion_status()
        logger.info(f"진행상황 추적 완료: {self.total_items:,}개 {self.item_name}")
    
    def _should_update_display(self) -> bool:
        """디스플레이 업데이트 여부 확인"""
        if not self.last_update_time:
            return True
            
        time_since_last_update = (datetime.now() - self.last_update_time).total_seconds()
        return time_since_last_update >= self.update_interval
    
    def _print_initial_status(self) -> None:
        """초기 상태 출력"""
        print(f"\n🚀 {self.item_name} 수집 시작")
        print(f"📊 전체 대상: {self.total_items:,}개")
        print(f"⏰ 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
    
    def _print_progress_status(self) -> None:
        """진행상황 상태 출력"""
        if not self.start_time:
            return
            
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        
        # 진행률 계산
        progress_percent = (self.current_count / self.total_items) * 100 if self.total_items > 0 else 0
        
        # 평균 속도 계산
        avg_speed = self._calculate_average_speed()
        
        # ETA 계산
        eta = self._calculate_eta(avg_speed)
        
        # 진행률 바 생성
        progress_bar = self._create_progress_bar(progress_percent)
        
        # 상태 출력
        print(f"\r{progress_bar} {progress_percent:5.1f}% | "
              f"{self.current_count:,}/{self.total_items:,} | "
              f"속도: {avg_speed:.1f}/초 | "
              f"경과: {self._format_duration(elapsed_time)} | "
              f"ETA: {eta}", end="", flush=True)
        
        # 오류 정보가 있으면 별도 출력
        if self.error_count > 0 or self.retry_count > 0:
            print(f"\n⚠️ 오류: {self.error_count}회, 재시도: {self.retry_count}회")
    
    def _print_completion_status(self) -> None:
        """완료 상태 출력"""
        if not self.start_time:
            return
            
        completion_time = datetime.now()
        total_time = completion_time - self.start_time
        avg_speed = self.current_count / total_time.total_seconds() if total_time.total_seconds() > 0 else 0
        
        print(f"\n\n✅ {self.item_name} 수집 완료!")
        print(f"📊 수집 결과: {self.current_count:,}개")
        print(f"⏰ 총 소요 시간: {self._format_duration(total_time)}")
        print(f"🚀 평균 속도: {avg_speed:.1f}개/초")
        
        if self.error_count > 0:
            print(f"⚠️ 오류 발생: {self.error_count}회")
        if self.retry_count > 0:
            print(f"🔄 재시도: {self.retry_count}회")
    
    def _calculate_average_speed(self) -> float:
        """평균 속도 계산"""
        if not self.speed_history:
            return 0.0
        
        return sum(self.speed_history) / len(self.speed_history)
    
    def _calculate_eta(self, avg_speed: float) -> str:
        """예상 완료 시간 계산"""
        if avg_speed <= 0 or self.current_count >= self.total_items:
            return "완료"
        
        remaining_items = self.total_items - self.current_count
        remaining_seconds = remaining_items / avg_speed
        
        eta_time = datetime.now() + timedelta(seconds=remaining_seconds)
        return eta_time.strftime('%H:%M:%S')
    
    def _create_progress_bar(self, progress_percent: float, width: int = 30) -> str:
        """진행률 바 생성"""
        filled_width = int((progress_percent / 100) * width)
        bar = "█" * filled_width + "░" * (width - filled_width)
        return f"[{bar}]"
    
    def _format_duration(self, duration: timedelta) -> str:
        """시간 형식 포맷팅"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 조회
        
        Returns:
            현재 상태 정보
        """
        with self.lock:
            current_time = datetime.now()
            elapsed_time = current_time - self.start_time if self.start_time else timedelta(0)
            
            avg_speed = self._calculate_average_speed()
            eta = self._calculate_eta(avg_speed)
            
            return {
                "total_items": self.total_items,
                "current_count": self.current_count,
                "progress_percent": (self.current_count / self.total_items) * 100 if self.total_items > 0 else 0,
                "elapsed_time": elapsed_time.total_seconds(),
                "average_speed": avg_speed,
                "eta": eta,
                "error_count": self.error_count,
                "retry_count": self.retry_count,
                "is_completed": self.current_count >= self.total_items
            }


class BatchProgressTracker:
    """배치별 진행상황 추적기"""
    
    def __init__(self, total_batches: int, batch_size: int, 
                 item_name: str = "배치"):
        """
        배치별 진행상황 추적기 초기화
        
        Args:
            total_batches: 전체 배치 수
            batch_size: 배치 크기
            item_name: 항목 이름
        """
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.item_name = item_name
        
        self.current_batch = 0
        self.completed_items = 0
        self.start_time = None
        
        logger.info(f"BatchProgressTracker 초기화: {total_batches}개 배치, 배치 크기 {batch_size}")
    
    def start(self) -> None:
        """배치 진행상황 추적 시작"""
        self.start_time = datetime.now()
        print(f"\n🚀 {self.item_name} 수집 시작")
        print(f"📊 전체 배치: {self.total_batches}개 (배치 크기: {self.batch_size:,}개)")
        print(f"📊 전체 항목: {self.total_batches * self.batch_size:,}개")
        print(f"⏰ 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
    
    def update_batch(self, batch_number: int, completed_items: int) -> None:
        """
        배치 진행상황 업데이트
        
        Args:
            batch_number: 현재 배치 번호
            completed_items: 완료된 항목 수
        """
        self.current_batch = batch_number
        self.completed_items = completed_items
        
        progress_percent = (batch_number / self.total_batches) * 100 if self.total_batches > 0 else 0
        
        print(f"📦 배치 {batch_number}/{self.total_batches} 완료 ({progress_percent:.1f}%) - "
              f"{completed_items:,}개 수집")
    
    def complete(self) -> None:
        """배치 진행상황 완료"""
        completion_time = datetime.now()
        total_time = completion_time - self.start_time if self.start_time else timedelta(0)
        
        print(f"\n✅ 모든 {self.item_name} 수집 완료!")
        print(f"📊 총 배치: {self.total_batches}개")
        print(f"📊 총 항목: {self.completed_items:,}개")
        print(f"⏰ 총 소요 시간: {self._format_duration(total_time)}")
    
    def _format_duration(self, duration: timedelta) -> str:
        """시간 형식 포맷팅"""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"


def create_progress_tracker(total_items: int, item_name: str = "항목", 
                           update_interval: float = 1.0) -> ProgressTracker:
    """
    진행상황 추적기 생성
    
    Args:
        total_items: 전체 항목 수
        item_name: 항목 이름
        update_interval: 업데이트 간격 (초)
        
    Returns:
        진행상황 추적기 인스턴스
    """
    return ProgressTracker(total_items, item_name, update_interval)


def create_batch_progress_tracker(total_batches: int, batch_size: int, 
                                 item_name: str = "배치") -> BatchProgressTracker:
    """
    배치별 진행상황 추적기 생성
    
    Args:
        total_batches: 전체 배치 수
        batch_size: 배치 크기
        item_name: 항목 이름
        
    Returns:
        배치별 진행상황 추적기 인스턴스
    """
    return BatchProgressTracker(total_batches, batch_size, item_name)
