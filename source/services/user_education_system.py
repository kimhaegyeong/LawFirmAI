# -*- coding: utf-8 -*-
"""
User Education and Warning System
사용자 교육 및 경고 시스템
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .legal_restriction_system import RestrictionResult, RestrictionLevel, LegalArea
from .content_filter_engine import FilterResult, IntentType, ContextType
from .response_validation_system import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)


class EducationType(Enum):
    """교육 유형"""
    ONBOARDING = "onboarding"           # 온보딩 교육
    WARNING = "warning"                 # 경고
    GUIDANCE = "guidance"               # 안내
    DISCLAIMER = "disclaimer"           # 면책 조항
    BEST_PRACTICES = "best_practices"   # 모범 사례


class UserLevel(Enum):
    """사용자 수준"""
    BEGINNER = "beginner"      # 초보자
    INTERMEDIATE = "intermediate"  # 중급자
    ADVANCED = "advanced"      # 고급자


class WarningType(Enum):
    """경고 유형"""
    LEGAL_ADVICE_REQUEST = "legal_advice_request"      # 법률 자문 요청
    SPECIFIC_CASE_QUESTION = "specific_case_question"  # 구체적 사건 질문
    SUSPICIOUS_REQUEST = "suspicious_request"         # 의심스러운 요청
    REPEATED_VIOLATIONS = "repeated_violations"        # 반복 위반
    SYSTEM_LIMITATION = "system_limitation"           # 시스템 한계


@dataclass
class EducationContent:
    """교육 콘텐츠"""
    id: str
    type: EducationType
    title: str
    content: str
    target_user_level: UserLevel
    priority: int
    conditions: List[str]
    interactive_elements: List[Dict[str, Any]]


@dataclass
class WarningMessage:
    """경고 메시지"""
    id: str
    type: WarningType
    severity: str
    title: str
    message: str
    action_required: str
    help_resources: List[str]
    dismissible: bool


@dataclass
class UserEducationRecord:
    """사용자 교육 기록"""
    user_id: str
    education_type: EducationType
    content_id: str
    timestamp: datetime
    acknowledged: bool
    quiz_score: Optional[float]


class UserEducationSystem:
    """사용자 교육 및 경고 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.education_content = self._initialize_education_content()
        self.warning_messages = self._initialize_warning_messages()
        self.user_records = {}  # 사용자별 교육 기록
        self.user_violations = {}  # 사용자별 위반 기록
        
    def _initialize_education_content(self) -> List[EducationContent]:
        """교육 콘텐츠 초기화"""
        return [
            # 온보딩 교육
            EducationContent(
                id="onboarding_001",
                type=EducationType.ONBOARDING,
                title="법률 AI 어시스턴트 사용 안내",
                content="""
                안녕하세요! 법률 AI 어시스턴트에 오신 것을 환영합니다.
                
                이 시스템은 다음과 같은 기능을 제공합니다:
                • 일반적인 법률 정보 제공
                • 법령 및 판례 참조
                • 법적 절차 안내
                • 관련 기관 정보 제공
                
                ⚠️ 중요한 제한사항:
                • 구체적인 법률 자문은 제공하지 않습니다
                • 개인 사건에 대한 판단은 하지 않습니다
                • 변호사를 대신할 수 없습니다
                
                올바른 사용법을 위해 다음 가이드를 확인해주세요.
                """,
                target_user_level=UserLevel.BEGINNER,
                priority=1,
                conditions=["first_time_user"],
                interactive_elements=[
                    {"type": "quiz", "question": "법률 AI가 제공할 수 있는 서비스는?", "options": ["법률 자문", "일반 정보", "소송 대리", "판결 예측"], "correct": 1}
                ]
            ),
            
            # 모범 사례 교육
            EducationContent(
                id="best_practices_001",
                type=EducationType.BEST_PRACTICES,
                title="올바른 질문 방법",
                content="""
                효과적인 질문을 위한 모범 사례:
                
                ✅ 좋은 질문 예시:
                • "계약서 작성 시 주의사항은 무엇인가요?"
                • "소송 제기 절차는 어떻게 되나요?"
                • "관련 법령을 알려주세요"
                
                ❌ 피해야 할 질문:
                • "제 경우 어떻게 해야 하나요?"
                • "소송하시겠습니까?"
                • "승소할 가능성이 있나요?"
                
                💡 팁:
                • 구체적인 개인 사안보다는 일반적인 정보를 요청하세요
                • "일반적으로", "보통" 같은 표현을 사용하세요
                • 전문가 상담이 필요한 경우를 인식하세요
                """,
                target_user_level=UserLevel.BEGINNER,
                priority=2,
                conditions=["repeated_violations"],
                interactive_elements=[
                    {"type": "example", "text": "좋은 질문과 나쁜 질문의 차이점을 학습합니다."}
                ]
            ),
            
            # 시스템 한계 교육
            EducationContent(
                id="limitations_001",
                type=EducationType.GUIDANCE,
                title="시스템의 한계와 전문가 상담의 필요성",
                content="""
                법률 AI 어시스턴트의 한계:
                
                🚫 제공하지 않는 서비스:
                • 구체적인 법률 자문
                • 개인 사건에 대한 판단
                • 소송 전략 수립
                • 법률 문서 작성 대리
                
                ✅ 언제 전문가 상담이 필요한가요?
                • 구체적인 사건이 있을 때
                • 법적 조치를 고려할 때
                • 복잡한 법적 문제가 있을 때
                • 중요한 결정을 내려야 할 때
                
                📞 전문가 상담 방법:
                • 변호사 상담
                • 법률구조공단 (1588-8282)
                • 국선변호인 신청
                • 관련 기관 문의
                """,
                target_user_level=UserLevel.INTERMEDIATE,
                priority=3,
                conditions=["system_limitation_warning"],
                interactive_elements=[
                    {"type": "resource_links", "links": ["법률구조공단", "국선변호인", "변호사 찾기"]}
                ]
            ),
            
            # 면책 조항 교육
            EducationContent(
                id="disclaimer_001",
                type=EducationType.DISCLAIMER,
                title="면책 조항 및 이용 약관",
                content="""
                법률 AI 어시스턴트 이용 약관:
                
                📋 서비스 범위:
                • 일반적인 법률 정보 제공
                • 법령 및 판례 참조
                • 법적 절차 안내
                
                ⚖️ 면책 사항:
                • 제공되는 정보는 참고용입니다
                • 구체적인 사안은 전문가와 상담하세요
                • 시스템의 답변에 대한 법적 책임을 지지 않습니다
                • 변호사-의뢰인 관계가 아닙니다
                
                🔒 개인정보 보호:
                • 질문 내용은 개인정보 보호법에 따라 처리됩니다
                • 개인 식별 정보는 저장하지 않습니다
                • 통계 목적으로만 익명화된 데이터를 사용합니다
                """,
                target_user_level=UserLevel.BEGINNER,
                priority=1,
                conditions=["disclaimer_required"],
                interactive_elements=[
                    {"type": "agreement_checkbox", "text": "이용 약관을 읽고 동의합니다."}
                ]
            )
        ]
    
    def _initialize_warning_messages(self) -> List[WarningMessage]:
        """경고 메시지 초기화"""
        return [
            # 법률 자문 요청 경고
            WarningMessage(
                id="warning_legal_advice_001",
                type=WarningType.LEGAL_ADVICE_REQUEST,
                severity="high",
                title="법률 자문 요청 감지",
                message="구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다. 일반적인 법률 정보나 절차는 안내드릴 수 있습니다.",
                action_required="변호사 상담을 권합니다",
                help_resources=["변호사 찾기", "법률구조공단", "국선변호인 신청"],
                dismissible=True
            ),
            
            # 구체적 사건 질문 경고
            WarningMessage(
                id="warning_specific_case_001",
                type=WarningType.SPECIFIC_CASE_QUESTION,
                severity="medium",
                title="구체적 사건 질문 감지",
                message="개인 사건에 대한 구체적인 조언은 제공할 수 없습니다. 일반적인 절차나 방법은 안내드릴 수 있습니다.",
                action_required="일반적인 정보 요청으로 질문을 수정하세요",
                help_resources=["올바른 질문 방법", "전문가 상담 안내"],
                dismissible=True
            ),
            
            # 의심스러운 요청 경고
            WarningMessage(
                id="warning_suspicious_001",
                type=WarningType.SUSPICIOUS_REQUEST,
                severity="critical",
                title="의심스러운 요청 감지",
                message="법적으로 부적절한 요청이 감지되었습니다. 합법적인 방법으로 도움을 받으시기 바랍니다.",
                action_required="합법적인 방법으로 질문을 수정하세요",
                help_resources=["법률 상담", "윤리 가이드"],
                dismissible=False
            ),
            
            # 반복 위반 경고
            WarningMessage(
                id="warning_repeated_001",
                type=WarningType.REPEATED_VIOLATIONS,
                severity="medium",
                title="반복적인 부적절한 질문",
                message="부적절한 질문이 반복되고 있습니다. 올바른 사용법을 확인해주세요.",
                action_required="사용 가이드를 다시 확인하세요",
                help_resources=["사용 가이드", "모범 사례", "FAQ"],
                dismissible=True
            ),
            
            # 시스템 한계 경고
            WarningMessage(
                id="warning_limitation_001",
                type=WarningType.SYSTEM_LIMITATION,
                severity="low",
                title="시스템 한계 안내",
                message="이 질문은 시스템의 한계로 인해 완전한 답변을 제공할 수 없습니다. 전문가 상담을 권합니다.",
                action_required="전문가와 상담하세요",
                help_resources=["전문가 찾기", "상담 방법"],
                dismissible=True
            )
        ]
    
    def get_onboarding_content(self, user_id: str) -> List[EducationContent]:
        """온보딩 콘텐츠 제공"""
        try:
            # 사용자 레벨 결정
            user_level = self._determine_user_level(user_id)
            
            # 온보딩 콘텐츠 필터링
            onboarding_content = [
                content for content in self.education_content
                if content.type == EducationType.ONBOARDING
                and content.target_user_level == user_level
            ]
            
            # 우선순위별 정렬
            onboarding_content.sort(key=lambda x: x.priority)
            
            return onboarding_content
            
        except Exception as e:
            self.logger.error(f"Error getting onboarding content: {e}")
            return []
    
    def generate_warning(self, restriction_result: RestrictionResult,
                        filter_result: FilterResult,
                        validation_result: ValidationResult,
                        user_id: str, query: str = "") -> Optional[WarningMessage]:
        """경고 메시지 생성"""
        try:
            # 위반 유형 결정
            warning_type = self._determine_warning_type(
                restriction_result, filter_result, validation_result, query
            )
            
            if not warning_type:
                return None
            
            # 적절한 경고 메시지 선택
            warning_message = self._select_warning_message(warning_type, user_id)
            
            if warning_message:
                # 사용자 위반 기록 업데이트
                self._update_user_violations(user_id, warning_type)
                
                # 경고 로깅
                self.logger.warning(f"Warning generated for user {user_id}: {warning_type.value}")
            
            return warning_message
            
        except Exception as e:
            self.logger.error(f"Error generating warning: {e}")
            return None
    
    def _determine_user_level(self, user_id: str) -> UserLevel:
        """사용자 레벨 결정"""
        # 사용자 기록이 없으면 초보자
        if user_id not in self.user_records:
            return UserLevel.BEGINNER
        
        # 교육 완료 횟수에 따른 레벨 결정
        completed_educations = len([
            record for record in self.user_records[user_id]
            if record.acknowledged
        ])
        
        if completed_educations < 2:
            return UserLevel.BEGINNER
        elif completed_educations < 5:
            return UserLevel.INTERMEDIATE
        else:
            return UserLevel.ADVANCED
    
    def _determine_warning_type(self, restriction_result: RestrictionResult,
                               filter_result: FilterResult,
                               validation_result: ValidationResult,
                               query: str = "") -> Optional[WarningType]:
        """경고 유형 결정 (더 엄격한 조건)"""
        # 절차 관련 질문에 대한 특별 처리 (더욱 관대한 처리)
        if any(keyword in query.lower() for keyword in ["절차", "방법", "과정", "규정", "제도"]):
            # 절차 관련 질문은 더욱 관대하게 처리
            if restriction_result.restriction_level == RestrictionLevel.CRITICAL:
                return WarningType.SUSPICIOUS_REQUEST
            else:
                # critical이 아니면 경고하지 않음
                return None
        
        # 명확한 개인 법률 자문 요청만 경고 (일반 정보 요청 제외)
        if (filter_result.intent_analysis.intent_type == IntentType.LEGAL_ADVICE_REQUEST and
            filter_result.intent_analysis.confidence > 0.8):  # 높은 신뢰도만
            return WarningType.LEGAL_ADVICE_REQUEST
        
        # 구체적 사건 질문 (높은 신뢰도만)
        if (filter_result.intent_analysis.intent_type == IntentType.CASE_SPECIFIC_QUESTION and
            filter_result.intent_analysis.confidence > 0.8):
            return WarningType.SPECIFIC_CASE_QUESTION
        
        # 의심스러운 요청 (높은 신뢰도만)
        if (filter_result.intent_analysis.intent_type == IntentType.SUSPICIOUS_REQUEST and
            filter_result.intent_analysis.confidence > 0.8):
            return WarningType.SUSPICIOUS_REQUEST
        
        # 검증 실패 (심각한 경우만)
        if (validation_result.status == ValidationStatus.REJECTED and
            validation_result.confidence > 0.9):
            return WarningType.SYSTEM_LIMITATION
        
        return None
    
    def _select_warning_message(self, warning_type: WarningType, user_id: str) -> Optional[WarningMessage]:
        """경고 메시지 선택"""
        # 해당 유형의 경고 메시지 찾기
        warning_messages = [
            msg for msg in self.warning_messages
            if msg.type == warning_type
        ]
        
        if not warning_messages:
            return None
        
        # 사용자별 맞춤 경고 메시지 생성
        base_message = warning_messages[0]
        
        # 반복 위반 확인
        if user_id in self.user_violations:
            violation_count = len(self.user_violations[user_id])
            if violation_count > 3:
                # 반복 위반 경고로 변경
                repeated_warning = next(
                    (msg for msg in self.warning_messages 
                     if msg.type == WarningType.REPEATED_VIOLATIONS), None
                )
                if repeated_warning:
                    return repeated_warning
        
        return base_message
    
    def _update_user_violations(self, user_id: str, warning_type: WarningType):
        """사용자 위반 기록 업데이트"""
        if user_id not in self.user_violations:
            self.user_violations[user_id] = []
        
        violation_record = {
            "timestamp": datetime.now(),
            "warning_type": warning_type,
            "severity": self._get_warning_severity(warning_type)
        }
        
        self.user_violations[user_id].append(violation_record)
        
        # 최근 10개만 유지
        if len(self.user_violations[user_id]) > 10:
            self.user_violations[user_id] = self.user_violations[user_id][-10:]
    
    def _get_warning_severity(self, warning_type: WarningType) -> str:
        """경고 심각도 반환"""
        severity_map = {
            WarningType.LEGAL_ADVICE_REQUEST: "high",
            WarningType.SPECIFIC_CASE_QUESTION: "medium",
            WarningType.SUSPICIOUS_REQUEST: "critical",
            WarningType.REPEATED_VIOLATIONS: "medium",
            WarningType.SYSTEM_LIMITATION: "low"
        }
        return severity_map.get(warning_type, "medium")
    
    def get_educational_content(self, user_id: str, content_type: EducationType) -> List[EducationContent]:
        """교육 콘텐츠 제공"""
        try:
            user_level = self._determine_user_level(user_id)
            
            # 해당 유형의 콘텐츠 필터링
            content_list = [
                content for content in self.education_content
                if content.type == content_type
                and content.target_user_level == user_level
            ]
            
            # 우선순위별 정렬
            content_list.sort(key=lambda x: x.priority)
            
            return content_list
            
        except Exception as e:
            self.logger.error(f"Error getting educational content: {e}")
            return []
    
    def record_education_completion(self, user_id: str, content_id: str, 
                                  quiz_score: Optional[float] = None) -> bool:
        """교육 완료 기록"""
        try:
            if user_id not in self.user_records:
                self.user_records[user_id] = []
            
            # 교육 콘텐츠 찾기
            content = next(
                (c for c in self.education_content if c.id == content_id), None
            )
            
            if not content:
                return False
            
            # 교육 완료 기록 생성
            record = UserEducationRecord(
                user_id=user_id,
                education_type=content.type,
                content_id=content_id,
                timestamp=datetime.now(),
                acknowledged=True,
                quiz_score=quiz_score
            )
            
            self.user_records[user_id].append(record)
            
            self.logger.info(f"Education completed: {user_id} - {content_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording education completion: {e}")
            return False
    
    def get_user_education_status(self, user_id: str) -> Dict[str, Any]:
        """사용자 교육 상태 조회"""
        try:
            if user_id not in self.user_records:
                return {
                    "user_level": UserLevel.BEGINNER.value,
                    "completed_educations": 0,
                    "pending_educations": [],
                    "violation_count": 0,
                    "last_education": None,
                    "needs_onboarding": True
                }
            
            records = self.user_records[user_id]
            violations = self.user_violations.get(user_id, [])
            
            # 완료된 교육 목록
            completed_educations = [
                {
                    "content_id": record.content_id,
                    "type": record.education_type.value,
                    "timestamp": record.timestamp.isoformat(),
                    "quiz_score": record.quiz_score
                }
                for record in records if record.acknowledged
            ]
            
            # 미완료 교육 목록
            completed_content_ids = {record.content_id for record in records if record.acknowledged}
            pending_educations = [
                {
                    "content_id": content.id,
                    "title": content.title,
                    "type": content.type.value,
                    "priority": content.priority
                }
                for content in self.education_content
                if content.id not in completed_content_ids
            ]
            
            # 마지막 교육 시간
            last_education = None
            if records:
                last_record = max(records, key=lambda x: x.timestamp)
                last_education = last_record.timestamp.isoformat()
            
            return {
                "user_level": self._determine_user_level(user_id).value,
                "completed_educations": len(completed_educations),
                "completed_education_list": completed_educations,
                "pending_educations": pending_educations,
                "violation_count": len(violations),
                "last_education": last_education,
                "needs_onboarding": len(completed_educations) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user education status: {e}")
            return {"error": str(e)}
    
    def get_help_resources(self, user_id: str) -> Dict[str, List[str]]:
        """도움말 리소스 제공"""
        try:
            user_level = self._determine_user_level(user_id)
            
            resources = {
                "beginner": [
                    "사용 가이드",
                    "FAQ",
                    "모범 사례",
                    "시스템 한계 안내"
                ],
                "intermediate": [
                    "고급 사용법",
                    "법령 검색 방법",
                    "판례 찾기",
                    "전문가 상담 안내"
                ],
                "advanced": [
                    "API 문서",
                    "고급 검색 기능",
                    "법률 데이터베이스",
                    "전문가 네트워크"
                ]
            }
            
            return {
                "resources": resources.get(user_level.value, resources["beginner"]),
                "user_level": user_level.value,
                "additional_help": [
                    "변호사 찾기",
                    "법률구조공단",
                    "국선변호인 신청",
                    "법원 정보"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting help resources: {e}")
            return {"error": str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        try:
            total_users = len(self.user_records)
            total_violations = sum(len(violations) for violations in self.user_violations.values())
            
            # 교육 완료 통계
            education_stats = {}
            for user_records in self.user_records.values():
                for record in user_records:
                    if record.acknowledged:
                        education_type = record.education_type.value
                        education_stats[education_type] = education_stats.get(education_type, 0) + 1
            
            # 위반 유형 통계
            violation_stats = {}
            for violations in self.user_violations.values():
                for violation in violations:
                    violation_type = violation["warning_type"].value
                    violation_stats[violation_type] = violation_stats.get(violation_type, 0) + 1
            
            return {
                "total_users": total_users,
                "total_violations": total_violations,
                "education_completions": education_stats,
                "violation_types": violation_stats,
                "total_education_content": len(self.education_content),
                "total_warning_messages": len(self.warning_messages)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system statistics: {e}")
            return {"error": str(e)}
