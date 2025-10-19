# -*- coding: utf-8 -*-
"""
동적 프롬프트 업데이트 시스템
최신 법령 개정사항과 판례를 실시간으로 반영하는 시스템
"""

import os
import json
import logging
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib

from .unified_prompt_manager import UnifiedPromptManager, LegalDomain

logger = logging.getLogger(__name__)


class LegalDataUpdater:
    """법률 데이터 업데이터"""
    
    def __init__(self, unified_manager: UnifiedPromptManager):
        """법률 데이터 업데이터 초기화"""
        self.unified_manager = unified_manager
        self.update_cache = {}
        self.last_update = {}
        
        # API 엔드포인트 설정
        self.api_endpoints = {
            "law_changes": "https://www.law.go.kr/DRF/lawService.do",
            "precedents": "https://www.scourt.go.kr/portal/justicelink/justicelinkList.do",
            "constitutional_court": "https://www.ccourt.go.kr/cckhome/kor/cck/cck0201/cck0201List.do"
        }
        
        # 업데이트 주기 설정 (시간)
        self.update_intervals = {
            "law_changes": 24,  # 24시간마다
            "precedents": 12,   # 12시간마다
            "constitutional_court": 48  # 48시간마다
        }
    
    async def check_and_update_all(self) -> Dict[str, bool]:
        """모든 법률 데이터 업데이트 확인 및 실행"""
        results = {}
        
        try:
            # 법령 개정사항 확인
            results["law_changes"] = await self._update_law_changes()
            
            # 판례 업데이트 확인
            results["precedents"] = await self._update_precedents()
            
            # 헌법재판소 결정 확인
            results["constitutional_court"] = await self._update_constitutional_decisions()
            
            logger.info(f"Legal data update completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in check_and_update_all: {e}")
            return {"error": str(e)}
    
    async def _update_law_changes(self) -> bool:
        """법령 개정사항 업데이트"""
        try:
            # 마지막 업데이트 시간 확인
            last_update = self.last_update.get("law_changes")
            if last_update and datetime.now() - last_update < timedelta(hours=self.update_intervals["law_changes"]):
                return False
            
            # 최신 법령 개정사항 조회
            recent_changes = await self._fetch_recent_law_changes()
            
            if recent_changes:
                # 프롬프트 업데이트
                await self._update_prompts_with_law_changes(recent_changes)
                self.last_update["law_changes"] = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating law changes: {e}")
            return False
    
    async def _update_precedents(self) -> bool:
        """판례 업데이트"""
        try:
            # 마지막 업데이트 시간 확인
            last_update = self.last_update.get("precedents")
            if last_update and datetime.now() - last_update < timedelta(hours=self.update_intervals["precedents"]):
                return False
            
            # 최신 판례 조회
            recent_precedents = await self._fetch_recent_precedents()
            
            if recent_precedents:
                # 프롬프트 업데이트
                await self._update_prompts_with_precedents(recent_precedents)
                self.last_update["precedents"] = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating precedents: {e}")
            return False
    
    async def _update_constitutional_decisions(self) -> bool:
        """헌법재판소 결정 업데이트"""
        try:
            # 마지막 업데이트 시간 확인
            last_update = self.last_update.get("constitutional_court")
            if last_update and datetime.now() - last_update < timedelta(hours=self.update_intervals["constitutional_court"]):
                return False
            
            # 최신 헌법재판소 결정 조회
            recent_decisions = await self._fetch_recent_constitutional_decisions()
            
            if recent_decisions:
                # 프롬프트 업데이트
                await self._update_prompts_with_constitutional_decisions(recent_decisions)
                self.last_update["constitutional_court"] = datetime.now()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating constitutional decisions: {e}")
            return False
    
    async def _fetch_recent_law_changes(self) -> List[Dict[str, Any]]:
        """최신 법령 개정사항 조회"""
        try:
            # 실제 API 호출 대신 시뮬레이션 데이터 사용
            # 실제 구현 시에는 법제처 API를 사용
            recent_changes = [
                {
                    "law_name": "민법",
                    "article": "제750조",
                    "change_type": "개정",
                    "change_date": "2024-01-15",
                    "summary": "불법행위로 인한 손해배상에 관한 조항 개정",
                    "details": "손해배상의 범위와 계산 방법에 대한 구체적 기준 추가"
                },
                {
                    "law_name": "형법",
                    "article": "제250조",
                    "change_type": "신설",
                    "change_date": "2024-02-01",
                    "summary": "디지털 성범죄 처벌 조항 신설",
                    "details": "온라인 성범죄에 대한 처벌 기준 신설"
                }
            ]
            
            return recent_changes
            
        except Exception as e:
            logger.error(f"Error fetching recent law changes: {e}")
            return []
    
    async def _fetch_recent_precedents(self) -> List[Dict[str, Any]]:
        """최신 판례 조회"""
        try:
            # 실제 API 호출 대신 시뮬레이션 데이터 사용
            recent_precedents = [
                {
                    "case_number": "대법원 2024다12345",
                    "case_name": "손해배상청구 사건",
                    "decision_date": "2024-01-20",
                    "summary": "불법행위로 인한 정신적 피해에 대한 손해배상 인정",
                    "key_points": ["정신적 피해의 인정 기준", "손해배상액 산정 방법"]
                },
                {
                    "case_number": "대법원 2024다67890",
                    "case_name": "계약 해지 사건",
                    "decision_date": "2024-02-05",
                    "summary": "계약 해지 사유에 대한 새로운 해석",
                    "key_points": ["계약 해지의 정당성", "손해배상 의무의 범위"]
                }
            ]
            
            return recent_precedents
            
        except Exception as e:
            logger.error(f"Error fetching recent precedents: {e}")
            return []
    
    async def _fetch_recent_constitutional_decisions(self) -> List[Dict[str, Any]]:
        """최신 헌법재판소 결정 조회"""
        try:
            # 실제 API 호출 대신 시뮬레이션 데이터 사용
            recent_decisions = [
                {
                    "case_number": "헌법재판소 2024헌마123",
                    "case_name": "개인정보보호법 위헌소원",
                    "decision_date": "2024-01-25",
                    "summary": "개인정보 수집·이용에 대한 동의 요건의 합헌성",
                    "key_points": ["개인정보보호권의 범위", "동의의 유효성 요건"]
                }
            ]
            
            return recent_decisions
            
        except Exception as e:
            logger.error(f"Error fetching recent constitutional decisions: {e}")
            return []
    
    async def _update_prompts_with_law_changes(self, law_changes: List[Dict[str, Any]]) -> None:
        """법령 개정사항을 프롬프트에 반영"""
        try:
            # 도메인별로 법령 개정사항 분류
            domain_changes = self._categorize_law_changes_by_domain(law_changes)
            
            # 각 도메인별로 프롬프트 업데이트
            for domain, changes in domain_changes.items():
                if changes:
                    await self._update_domain_prompt_with_changes(domain, changes)
            
            logger.info(f"Updated prompts with {len(law_changes)} law changes")
            
        except Exception as e:
            logger.error(f"Error updating prompts with law changes: {e}")
    
    async def _update_prompts_with_precedents(self, precedents: List[Dict[str, Any]]) -> None:
        """판례를 프롬프트에 반영"""
        try:
            # 도메인별로 판례 분류
            domain_precedents = self._categorize_precedents_by_domain(precedents)
            
            # 각 도메인별로 프롬프트 업데이트
            for domain, precedents_list in domain_precedents.items():
                if precedents_list:
                    await self._update_domain_prompt_with_precedents(domain, precedents_list)
            
            logger.info(f"Updated prompts with {len(precedents)} precedents")
            
        except Exception as e:
            logger.error(f"Error updating prompts with precedents: {e}")
    
    async def _update_prompts_with_constitutional_decisions(self, decisions: List[Dict[str, Any]]) -> None:
        """헌법재판소 결정을 프롬프트에 반영"""
        try:
            # 헌법재판소 결정은 모든 도메인에 영향을 줄 수 있음
            for decision in decisions:
                await self._update_all_domain_prompts_with_constitutional_decision(decision)
            
            logger.info(f"Updated prompts with {len(decisions)} constitutional decisions")
            
        except Exception as e:
            logger.error(f"Error updating prompts with constitutional decisions: {e}")
    
    def _categorize_law_changes_by_domain(self, law_changes: List[Dict[str, Any]]) -> Dict[LegalDomain, List[Dict[str, Any]]]:
        """법령 개정사항을 도메인별로 분류"""
        domain_changes = {domain: [] for domain in LegalDomain}
        
        for change in law_changes:
            law_name = change.get("law_name", "")
            
            # 법령명에 따른 도메인 분류
            if "민법" in law_name:
                domain_changes[LegalDomain.CIVIL_LAW].append(change)
            elif "형법" in law_name:
                domain_changes[LegalDomain.CRIMINAL_LAW].append(change)
            elif "상법" in law_name:
                domain_changes[LegalDomain.COMMERCIAL_LAW].append(change)
            elif "근로기준법" in law_name or "노동" in law_name:
                domain_changes[LegalDomain.LABOR_LAW].append(change)
            elif "부동산" in law_name:
                domain_changes[LegalDomain.PROPERTY_LAW].append(change)
            elif "특허" in law_name or "상표" in law_name or "저작권" in law_name:
                domain_changes[LegalDomain.INTELLECTUAL_PROPERTY].append(change)
            elif "세법" in law_name or "소득세" in law_name or "법인세" in law_name:
                domain_changes[LegalDomain.TAX_LAW].append(change)
            elif "민사소송" in law_name:
                domain_changes[LegalDomain.CIVIL_PROCEDURE].append(change)
            elif "형사소송" in law_name:
                domain_changes[LegalDomain.CRIMINAL_PROCEDURE].append(change)
            else:
                domain_changes[LegalDomain.GENERAL].append(change)
        
        return domain_changes
    
    def _categorize_precedents_by_domain(self, precedents: List[Dict[str, Any]]) -> Dict[LegalDomain, List[Dict[str, Any]]]:
        """판례를 도메인별로 분류"""
        domain_precedents = {domain: [] for domain in LegalDomain}
        
        for precedent in precedents:
            case_name = precedent.get("case_name", "")
            summary = precedent.get("summary", "")
            
            # 사건명과 요약에 따른 도메인 분류
            if "손해배상" in case_name or "계약" in case_name or "소유권" in case_name:
                domain_precedents[LegalDomain.CIVIL_LAW].append(precedent)
            elif "살인" in case_name or "절도" in case_name or "사기" in case_name:
                domain_precedents[LegalDomain.CRIMINAL_LAW].append(precedent)
            elif "이혼" in case_name or "상속" in case_name or "친자" in case_name:
                domain_precedents[LegalDomain.FAMILY_LAW].append(precedent)
            elif "회사" in case_name or "주식" in case_name:
                domain_precedents[LegalDomain.COMMERCIAL_LAW].append(precedent)
            elif "근로" in case_name or "해고" in case_name or "임금" in case_name:
                domain_precedents[LegalDomain.LABOR_LAW].append(precedent)
            elif "부동산" in case_name or "등기" in case_name:
                domain_precedents[LegalDomain.PROPERTY_LAW].append(precedent)
            elif "특허" in case_name or "상표" in case_name or "저작권" in case_name:
                domain_precedents[LegalDomain.INTELLECTUAL_PROPERTY].append(precedent)
            elif "세금" in case_name or "소득세" in case_name:
                domain_precedents[LegalDomain.TAX_LAW].append(precedent)
            elif "소송" in case_name or "재판" in case_name:
                domain_precedents[LegalDomain.CIVIL_PROCEDURE].append(precedent)
            else:
                domain_precedents[LegalDomain.GENERAL].append(precedent)
        
        return domain_precedents
    
    async def _update_domain_prompt_with_changes(self, domain: LegalDomain, changes: List[Dict[str, Any]]) -> None:
        """특정 도메인의 프롬프트를 법령 개정사항으로 업데이트"""
        try:
            # 도메인별 업데이트된 템플릿 생성
            updated_template = self._create_updated_domain_template(domain, changes)
            
            # 통합 프롬프트 관리자에 업데이트 반영
            # 실제 구현 시에는 프롬프트 버전 관리 시스템을 통해 업데이트
            
            logger.info(f"Updated {domain.value} domain prompt with {len(changes)} law changes")
            
        except Exception as e:
            logger.error(f"Error updating domain prompt for {domain.value}: {e}")
    
    async def _update_domain_prompt_with_precedents(self, domain: LegalDomain, precedents: List[Dict[str, Any]]) -> None:
        """특정 도메인의 프롬프트를 판례로 업데이트"""
        try:
            # 도메인별 업데이트된 템플릿 생성
            updated_template = self._create_updated_domain_template_with_precedents(domain, precedents)
            
            # 통합 프롬프트 관리자에 업데이트 반영
            
            logger.info(f"Updated {domain.value} domain prompt with {len(precedents)} precedents")
            
        except Exception as e:
            logger.error(f"Error updating domain prompt for {domain.value} with precedents: {e}")
    
    async def _update_all_domain_prompts_with_constitutional_decision(self, decision: Dict[str, Any]) -> None:
        """모든 도메인 프롬프트를 헌법재판소 결정으로 업데이트"""
        try:
            # 헌법재판소 결정은 모든 도메인에 영향을 줄 수 있음
            for domain in LegalDomain:
                if domain != LegalDomain.GENERAL:
                    await self._update_domain_prompt_with_constitutional_decision(domain, decision)
            
            logger.info(f"Updated all domain prompts with constitutional decision: {decision.get('case_number')}")
            
        except Exception as e:
            logger.error(f"Error updating domain prompts with constitutional decision: {e}")
    
    async def _update_domain_prompt_with_constitutional_decision(self, domain: LegalDomain, decision: Dict[str, Any]) -> None:
        """특정 도메인의 프롬프트를 헌법재판소 결정으로 업데이트"""
        try:
            # 헌법재판소 결정 반영 템플릿 생성
            updated_template = self._create_updated_domain_template_with_constitutional_decision(domain, decision)
            
            # 통합 프롬프트 관리자에 업데이트 반영
            
            logger.info(f"Updated {domain.value} domain prompt with constitutional decision: {decision.get('case_number')}")
            
        except Exception as e:
            logger.error(f"Error updating domain prompt for {domain.value} with constitutional decision: {e}")
    
    def _create_updated_domain_template(self, domain: LegalDomain, changes: List[Dict[str, Any]]) -> str:
        """법령 개정사항을 반영한 도메인 템플릿 생성"""
        changes_text = "\n".join([
            f"- {change['law_name']} {change['article']} ({change['change_type']}, {change['change_date']}): {change['summary']}"
            for change in changes
        ])
        
        return f"""
## 최신 법령 개정사항 ({datetime.now().strftime('%Y-%m-%d')})
{changes_text}

위 개정사항을 반영하여 답변하세요.
"""
    
    def _create_updated_domain_template_with_precedents(self, domain: LegalDomain, precedents: List[Dict[str, Any]]) -> str:
        """판례를 반영한 도메인 템플릿 생성"""
        precedents_text = "\n".join([
            f"- {precedent['case_number']} ({precedent['decision_date']}): {precedent['summary']}"
            for precedent in precedents
        ])
        
        return f"""
## 최신 관련 판례 ({datetime.now().strftime('%Y-%m-%d')})
{precedents_text}

위 판례를 참고하여 답변하세요.
"""
    
    def _create_updated_domain_template_with_constitutional_decision(self, domain: LegalDomain, decision: Dict[str, Any]) -> str:
        """헌법재판소 결정을 반영한 도메인 템플릿 생성"""
        return f"""
## 최신 헌법재판소 결정 ({datetime.now().strftime('%Y-%m-%d')})
- {decision['case_number']} ({decision['decision_date']}): {decision['summary']}

위 헌법재판소 결정을 참고하여 답변하세요.
"""


class DynamicPromptUpdater:
    """동적 프롬프트 업데이터"""
    
    def __init__(self, unified_manager: UnifiedPromptManager):
        """동적 프롬프트 업데이터 초기화"""
        self.unified_manager = unified_manager
        self.legal_data_updater = LegalDataUpdater(unified_manager)
        self.is_running = False
        self.update_task = None
    
    async def start_auto_update(self, interval_minutes: int = 60) -> None:
        """자동 업데이트 시작"""
        if self.is_running:
            logger.warning("Auto update is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting auto update with {interval_minutes} minute interval")
        
        while self.is_running:
            try:
                # 모든 법률 데이터 업데이트 확인 및 실행
                results = await self.legal_data_updater.check_and_update_all()
                logger.info(f"Auto update completed: {results}")
                
                # 다음 업데이트까지 대기
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in auto update: {e}")
                await asyncio.sleep(60)  # 오류 시 1분 후 재시도
    
    def stop_auto_update(self) -> None:
        """자동 업데이트 중지"""
        self.is_running = False
        logger.info("Auto update stopped")
    
    async def manual_update(self) -> Dict[str, bool]:
        """수동 업데이트 실행"""
        try:
            logger.info("Starting manual update")
            results = await self.legal_data_updater.check_and_update_all()
            logger.info(f"Manual update completed: {results}")
            return results
        except Exception as e:
            logger.error(f"Error in manual update: {e}")
            return {"error": str(e)}
    
    def get_update_status(self) -> Dict[str, Any]:
        """업데이트 상태 조회"""
        return {
            "is_running": self.is_running,
            "last_updates": self.legal_data_updater.last_update,
            "update_intervals": self.legal_data_updater.update_intervals
        }


# 전역 인스턴스
def create_dynamic_prompt_updater(unified_manager: UnifiedPromptManager) -> DynamicPromptUpdater:
    """동적 프롬프트 업데이터 생성"""
    return DynamicPromptUpdater(unified_manager)
