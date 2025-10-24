# -*- coding: utf-8 -*-
"""
Personal Data Protection Compliance System
개인정보보호법 준수 시스템
"""

import re
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from enum import Enum
from dataclasses import dataclass, asdict
from .security_logger import get_security_logger, SecurityEventType, SecurityLevel


class PersonalDataType(Enum):
    """개인정보 유형"""
    NAME = "name"  # 이름
    PHONE = "phone"  # 전화번호
    EMAIL = "email"  # 이메일
    ID_NUMBER = "id_number"  # 주민번호
    ADDRESS = "address"  # 주소
    BANK_ACCOUNT = "bank_account"  # 계좌번호
    CREDIT_CARD = "credit_card"  # 신용카드번호
    IP_ADDRESS = "ip_address"  # IP주소
    LOCATION = "location"  # 위치정보
    BIOMETRIC = "biometric"  # 생체정보
    HEALTH = "health"  # 건강정보
    FINANCIAL = "financial"  # 금융정보
    OTHER = "other"  # 기타


class ProcessingPurpose(Enum):
    """처리 목적"""
    LEGAL_CONSULTATION = "legal_consultation"  # 법률 상담
    SERVICE_PROVISION = "service_provision"  # 서비스 제공
    SYSTEM_OPERATION = "system_operation"  # 시스템 운영
    SECURITY = "security"  # 보안
    STATISTICS = "statistics"  # 통계
    RESEARCH = "research"  # 연구


class ConsentStatus(Enum):
    """동의 상태"""
    GIVEN = "given"  # 동의함
    NOT_GIVEN = "not_given"  # 동의하지 않음
    WITHDRAWN = "withdrawn"  # 동의 철회
    REQUIRED = "required"  # 동의 필요


@dataclass
class PersonalDataRecord:
    """개인정보 기록"""
    data_type: PersonalDataType
    original_data: str
    hashed_data: str
    masked_data: str
    processing_purpose: ProcessingPurpose
    consent_status: ConsentStatus
    collected_at: datetime
    retention_period_days: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class PrivacyComplianceReport:
    """개인정보보호법 준수 보고서"""
    total_processed: int
    personal_data_found: int
    consent_required: int
    consent_given: int
    data_retention_compliant: int
    violations: List[str]
    recommendations: List[str]
    compliance_score: float  # 0.0 ~ 1.0


class PersonalDataDetector:
    """개인정보 탐지기"""
    
    def __init__(self):
        self.patterns = {
            PersonalDataType.NAME: [
                r'[가-힣]{2,4}(?=씨|님|군|양|선생|교수|박사|원장|사장|대표|팀장|부장|과장|대리|주임|사원)',
            ],
            PersonalDataType.PHONE: [
                r'\d{2,3}-\d{3,4}-\d{4}',
                r'\d{2,3}\s\d{3,4}\s\d{4}',
                r'\d{10,11}',
                r'\+82\s?\d{1,2}\s?\d{3,4}\s?\d{4}',
            ],
            PersonalDataType.EMAIL: [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            ],
            PersonalDataType.ID_NUMBER: [
                r'\d{6}-\d{7}',
                r'\d{6}\s\d{7}',
                r'\d{13}',
            ],
            PersonalDataType.ADDRESS: [
                r'[가-힣]{2,}(?:시|도|구|군|동|로|길|번지)(?:\s|$)',  # 주소로 끝나는 경우만 (최소 2글자)
                r'\d{3}-\d{3}(?:\s|$)',  # 우편번호로 끝나는 경우만
                r'[가-힣]{2,}(?:아파트|빌라|오피스텔|상가|건물)(?:\s|$)',  # 건물명으로 끝나는 경우만 (최소 2글자)
            ],
            PersonalDataType.BANK_ACCOUNT: [
                r'\d{3}-\d{2}-\d{6}',
                r'\d{3}-\d{3}-\d{6}',
                r'\d{4}-\d{2}-\d{6}',
                r'\d{4}-\d{3}-\d{6}',
            ],
            PersonalDataType.CREDIT_CARD: [
                r'\d{4}-\d{4}-\d{4}-\d{4}',
                r'\d{4}\s\d{4}\s\d{4}\s\d{4}',
                r'\d{16}',
            ],
            PersonalDataType.IP_ADDRESS: [
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            ],
        }
    
    def detect_personal_data(self, text: str) -> List[PersonalDataRecord]:
        """텍스트에서 개인정보 탐지"""
        detected_data = []
        
        for data_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    original_data = match.group()
                    hashed_data = self._hash_data(original_data)
                    masked_data = self._mask_data(original_data, data_type)
                    
                    record = PersonalDataRecord(
                        data_type=data_type,
                        original_data=original_data,
                        hashed_data=hashed_data,
                        masked_data=masked_data,
                        processing_purpose=ProcessingPurpose.LEGAL_CONSULTATION,
                        consent_status=ConsentStatus.REQUIRED,
                        collected_at=datetime.now(),
                        retention_period_days=30  # 기본 30일
                    )
                    
                    detected_data.append(record)
        
        return detected_data
    
    def _hash_data(self, data: str) -> str:
        """개인정보 해시화 (복호화 불가)"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _mask_data(self, data: str, data_type: PersonalDataType) -> str:
        """개인정보 마스킹"""
        if data_type == PersonalDataType.NAME:
            if len(data) <= 2:
                return data[0] + "*"
            else:
                return data[0] + "*" * (len(data) - 2) + data[-1]
        
        elif data_type == PersonalDataType.PHONE:
            if "-" in data:
                parts = data.split("-")
                return f"{parts[0]}-****-{parts[2]}"
            else:
                return data[:3] + "****" + data[-4:]
        
        elif data_type == PersonalDataType.EMAIL:
            username, domain = data.split("@")
            if len(username) <= 2:
                return f"*@{domain}"
            else:
                return f"{username[0]}***@{domain}"
        
        elif data_type == PersonalDataType.ID_NUMBER:
            if "-" in data:
                return data[:6] + "-*******"
            else:
                return data[:6] + "*******"
        
        elif data_type == PersonalDataType.BANK_ACCOUNT:
            if "-" in data:
                parts = data.split("-")
                return f"{parts[0]}-{parts[1]}-******"
            else:
                return data[:6] + "******"
        
        elif data_type == PersonalDataType.CREDIT_CARD:
            if "-" in data:
                parts = data.split("-")
                return f"{parts[0]}-****-****-{parts[3]}"
            else:
                return data[:4] + "****" + "****" + data[-4:]
        
        elif data_type == PersonalDataType.IP_ADDRESS:
            parts = data.split(".")
            return f"{parts[0]}.{parts[1]}.***.***"
        
        else:
            # 기본 마스킹
            if len(data) <= 4:
                return "*" * len(data)
            else:
                return data[:2] + "*" * (len(data) - 4) + data[-2:]


class PrivacyComplianceManager:
    """개인정보보호법 준수 관리자"""
    
    def __init__(self):
        self.detector = PersonalDataDetector()
        self.security_logger = get_security_logger()
        self.data_records: List[PersonalDataRecord] = []
        self.retention_policies = {
            PersonalDataType.NAME: 30,
            PersonalDataType.PHONE: 30,
            PersonalDataType.EMAIL: 30,
            PersonalDataType.ID_NUMBER: 7,  # 주민번호는 최소 보관
            PersonalDataType.ADDRESS: 30,
            PersonalDataType.BANK_ACCOUNT: 7,
            PersonalDataType.CREDIT_CARD: 7,
            PersonalDataType.IP_ADDRESS: 90,
            PersonalDataType.LOCATION: 30,
            PersonalDataType.BIOMETRIC: 7,
            PersonalDataType.HEALTH: 7,
            PersonalDataType.FINANCIAL: 7,
            PersonalDataType.OTHER: 30,
        }
    
    def process_text(self, 
                    text: str,
                    processing_purpose: ProcessingPurpose = ProcessingPurpose.LEGAL_CONSULTATION,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    ip_address: Optional[str] = None) -> Dict[str, Any]:
        """텍스트 처리 및 개인정보 보호"""
        
        # 개인정보 탐지
        detected_data = self.detector.detect_personal_data(text)
        
        # 개인정보가 발견된 경우
        if detected_data:
            # 보안 로그 기록
            self.security_logger.log_event(
                SecurityEventType.DATA_ACCESS,
                SecurityLevel.MEDIUM,
                f"Personal data detected: {len(detected_data)} items",
                {
                    'data_types': [record.data_type.value for record in detected_data],
                    'processing_purpose': processing_purpose.value,
                    'text_length': len(text)
                },
                user_id,
                ip_address,
                session_id
            )
            
            # 개인정보 마스킹
            masked_text = self._mask_text(text, detected_data)
            
            # 개인정보 기록 저장
            for record in detected_data:
                record.user_id = user_id
                record.session_id = session_id
                record.ip_address = ip_address
                record.processing_purpose = processing_purpose
                self.data_records.append(record)
            
            return {
                'has_personal_data': True,
                'detected_count': len(detected_data),
                'data_types': [record.data_type.value for record in detected_data],
                'masked_text': masked_text,
                'original_text': text,
                'consent_required': True,
                'retention_period': max([record.retention_period_days for record in detected_data])
            }
        
        return {
            'has_personal_data': False,
            'detected_count': 0,
            'data_types': [],
            'masked_text': text,
            'original_text': text,
            'consent_required': False,
            'retention_period': 0
        }
    
    def _mask_text(self, text: str, detected_data: List[PersonalDataRecord]) -> str:
        """텍스트에서 개인정보 마스킹 (중복 제거 개선 버전)"""
        masked_text = text
        
        # 중복 제거: 같은 위치의 개인정보는 가장 긴 것만 유지
        unique_data = []
        for record in detected_data:
            is_duplicate = False
            for existing in unique_data:
                # 같은 위치에 겹치는 개인정보가 있는지 확인
                if (record.original_data in existing.original_data or 
                    existing.original_data in record.original_data):
                    # 더 긴 것을 유지
                    if len(record.original_data) > len(existing.original_data):
                        unique_data.remove(existing)
                        unique_data.append(record)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_data.append(record)
        
        # 탐지된 개인정보를 길이 순으로 정렬 (긴 것부터 처리하여 중복 교체 방지)
        sorted_data = sorted(unique_data, key=lambda x: len(x.original_data), reverse=True)
        
        # 원본 데이터를 마스킹된 데이터로 교체
        for record in sorted_data:
            # 정확한 매칭을 위해 원본 데이터가 여전히 존재하는지 확인
            if record.original_data in masked_text:
                masked_text = masked_text.replace(record.original_data, record.masked_data, 1)  # 첫 번째 매칭만 교체
        
        return masked_text
    
    def get_privacy_notice(self) -> str:
        """개인정보 처리방침 안내"""
        return """
        🔒 개인정보 처리방침 안내
        
        LawFirmAI는 개인정보보호법에 따라 다음과 같이 개인정보를 처리합니다:
        
        📋 수집하는 개인정보
        - 이름, 전화번호, 이메일 주소
        - 법률 상담 관련 정보
        - 시스템 이용 기록 (IP주소, 접속시간 등)
        
        🎯 처리 목적
        - 법률 상담 서비스 제공
        - 서비스 품질 향상
        - 시스템 보안 및 안정성 확보
        
        ⏰ 보관 기간
        - 일반 개인정보: 30일
        - 민감정보(주민번호, 계좌번호 등): 7일
        - 시스템 로그: 90일
        
        🔐 보호 조치
        - 개인정보 자동 마스킹 처리
        - 암호화 저장
        - 접근 권한 제한
        - 정기적 삭제
        
        📞 문의사항
        개인정보 처리에 대한 문의사항이 있으시면 언제든지 연락주세요.
        """
    
    def check_retention_compliance(self) -> Dict[str, Any]:
        """보관 기간 준수 검사"""
        now = datetime.now()
        expired_records = []
        
        for record in self.data_records:
            expiry_date = record.collected_at + timedelta(days=record.retention_period_days)
            if now > expiry_date:
                expired_records.append(record)
        
        return {
            'total_records': len(self.data_records),
            'expired_records': len(expired_records),
            'expired_data_types': list(set([record.data_type.value for record in expired_records])),
            'compliance_status': 'compliant' if len(expired_records) == 0 else 'non_compliant'
        }
    
    def cleanup_expired_data(self) -> int:
        """만료된 개인정보 삭제"""
        now = datetime.now()
        expired_count = 0
        
        # 만료된 기록 식별
        expired_records = []
        remaining_records = []
        
        for record in self.data_records:
            expiry_date = record.collected_at + timedelta(days=record.retention_period_days)
            if now > expiry_date:
                expired_records.append(record)
                expired_count += 1
            else:
                remaining_records.append(record)
        
        # 만료된 기록 삭제
        self.data_records = remaining_records
        
        # 보안 로그 기록
        if expired_count > 0:
            self.security_logger.log_event(
                SecurityEventType.DATA_MODIFICATION,
                SecurityLevel.LOW,
                f"Expired personal data cleaned up: {expired_count} records",
                {
                    'expired_count': expired_count,
                    'remaining_count': len(remaining_records)
                }
            )
        
        return expired_count
    
    def generate_compliance_report(self) -> PrivacyComplianceReport:
        """개인정보보호법 준수 보고서 생성"""
        total_processed = len(self.data_records)
        personal_data_found = total_processed
        
        # 동의 상태 분석
        consent_given = len([r for r in self.data_records if r.consent_status == ConsentStatus.GIVEN])
        consent_required = len([r for r in self.data_records if r.consent_status == ConsentStatus.REQUIRED])
        
        # 보관 기간 준수 검사
        retention_check = self.check_retention_compliance()
        data_retention_compliant = retention_check['total_records'] - retention_check['expired_records']
        
        # 위반 사항 식별
        violations = []
        if retention_check['expired_records'] > 0:
            violations.append(f"만료된 개인정보 {retention_check['expired_records']}건 보관 중")
        
        if consent_required > 0:
            violations.append(f"동의가 필요한 개인정보 {consent_required}건 존재")
        
        # 권고사항 생성
        recommendations = []
        if violations:
            recommendations.append("만료된 개인정보 즉시 삭제 필요")
            recommendations.append("개인정보 처리 동의 절차 강화 필요")
        
        if total_processed > 1000:
            recommendations.append("개인정보 처리 현황 정기 검토 필요")
        
        # 준수 점수 계산
        compliance_score = 1.0
        if total_processed > 0:
            compliance_score = data_retention_compliant / total_processed
            if consent_required > 0:
                compliance_score *= 0.8  # 동의 미완료 시 20% 감점
        
        return PrivacyComplianceReport(
            total_processed=total_processed,
            personal_data_found=personal_data_found,
            consent_required=consent_required,
            consent_given=consent_given,
            data_retention_compliant=data_retention_compliant,
            violations=violations,
            recommendations=recommendations,
            compliance_score=compliance_score
        )
    
    def export_compliance_data(self) -> Dict[str, Any]:
        """준수 데이터 내보내기 (감사용)"""
        return {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(self.data_records),
            'data_summary': {
                data_type.value: len([r for r in self.data_records if r.data_type == data_type])
                for data_type in PersonalDataType
            },
            'retention_status': self.check_retention_compliance(),
            'compliance_report': asdict(self.generate_compliance_report())
        }


# 전역 개인정보보호법 준수 관리자 인스턴스
privacy_compliance_manager = PrivacyComplianceManager()


def get_privacy_compliance_manager() -> PrivacyComplianceManager:
    """개인정보보호법 준수 관리자 인스턴스 반환"""
    return privacy_compliance_manager
