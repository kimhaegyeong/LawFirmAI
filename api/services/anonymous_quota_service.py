"""
익명 사용자 질의 제한 서비스
IP 주소 기반으로 익명 사용자의 질의 횟수를 추적하고 제한합니다.
"""
import os
import logging
from typing import Dict, Optional
from datetime import datetime, date, time
from collections import defaultdict

from api.config import api_config

logger = logging.getLogger(__name__)


class AnonymousQuotaService:
    """익명 사용자 질의 제한 서비스"""
    
    def __init__(self):
        """초기화"""
        self.enabled = api_config.anonymous_quota_enabled
        self.quota_limit = api_config.anonymous_quota_limit
        self.reset_hour = api_config.anonymous_quota_reset_hour
        
        # IP 주소별 질의 횟수 및 마지막 리셋 날짜 저장
        # 구조: {ip_address: {"count": int, "last_reset_date": date}}
        self._quota_store: Dict[str, Dict[str, any]] = defaultdict(
            lambda: {"count": 0, "last_reset_date": date.today()}
        )
        
        if self.enabled:
            logger.info(f"익명 사용자 질의 제한 활성화: {self.quota_limit}회/일")
    
    def is_enabled(self) -> bool:
        """익명 사용자 제한 활성화 여부"""
        return self.enabled
    
    def _get_quota_key(self, ip_address: str) -> str:
        """IP 주소를 키로 사용 (정규화)"""
        return ip_address.strip()
    
    def _should_reset(self, ip_address: str) -> bool:
        """일일 리셋이 필요한지 확인"""
        key = self._get_quota_key(ip_address)
        if key not in self._quota_store:
            return True
        
        last_reset_date = self._quota_store[key]["last_reset_date"]
        today = date.today()
        
        # 날짜가 변경되었으면 리셋
        if last_reset_date < today:
            return True
        
        return False
    
    def _reset_quota(self, ip_address: str):
        """특정 IP 주소의 질의 횟수 리셋"""
        key = self._get_quota_key(ip_address)
        self._quota_store[key] = {
            "count": 0,
            "last_reset_date": date.today()
        }
        logger.debug(f"익명 사용자 질의 횟수 리셋: {ip_address}")
    
    def check_quota(self, ip_address: str) -> bool:
        """남은 질의 횟수 확인 (질의 가능 여부)"""
        if not self.enabled:
            return True
        
        key = self._get_quota_key(ip_address)
        
        # 일일 리셋 확인
        if self._should_reset(ip_address):
            self._reset_quota(ip_address)
        
        # 현재 질의 횟수 확인
        current_count = self._quota_store[key]["count"]
        return current_count < self.quota_limit
    
    def get_remaining_quota(self, ip_address: str) -> int:
        """남은 질의 횟수 조회"""
        if not self.enabled:
            return self.quota_limit
        
        key = self._get_quota_key(ip_address)
        
        # 일일 리셋 확인
        if self._should_reset(ip_address):
            self._reset_quota(ip_address)
        
        current_count = self._quota_store[key]["count"]
        remaining = max(0, self.quota_limit - current_count)
        return remaining
    
    def increment_quota(self, ip_address: str) -> int:
        """질의 횟수 증가 및 남은 횟수 반환"""
        if not self.enabled:
            return self.quota_limit
        
        key = self._get_quota_key(ip_address)
        
        # 일일 리셋 확인
        if self._should_reset(ip_address):
            self._reset_quota(ip_address)
        
        # 질의 횟수 증가
        self._quota_store[key]["count"] += 1
        current_count = self._quota_store[key]["count"]
        
        remaining = max(0, self.quota_limit - current_count)
        logger.debug(f"익명 사용자 질의 횟수 증가: {ip_address}, 현재: {current_count}/{self.quota_limit}, 남은 횟수: {remaining}")
        
        return remaining
    
    def reset_daily_quota(self):
        """모든 IP 주소의 일일 질의 횟수 리셋"""
        if not self.enabled:
            return
        
        today = date.today()
        reset_count = 0
        
        for key in list(self._quota_store.keys()):
            if self._quota_store[key]["last_reset_date"] < today:
                self._quota_store[key] = {
                    "count": 0,
                    "last_reset_date": today
                }
                reset_count += 1
        
        if reset_count > 0:
            logger.info(f"익명 사용자 질의 횟수 일일 리셋 완료: {reset_count}개 IP 주소")
    
    def get_quota_info(self, ip_address: str) -> Dict[str, any]:
        """IP 주소의 질의 제한 정보 조회"""
        if not self.enabled:
            return {
                "enabled": False,
                "limit": self.quota_limit,
                "remaining": self.quota_limit
            }
        
        key = self._get_quota_key(ip_address)
        
        # 일일 리셋 확인
        if self._should_reset(ip_address):
            self._reset_quota(ip_address)
        
        current_count = self._quota_store[key]["count"]
        remaining = max(0, self.quota_limit - current_count)
        
        return {
            "enabled": True,
            "limit": self.quota_limit,
            "used": current_count,
            "remaining": remaining,
            "last_reset_date": self._quota_store[key]["last_reset_date"].isoformat()
        }


# 전역 인스턴스
anonymous_quota_service = AnonymousQuotaService()

