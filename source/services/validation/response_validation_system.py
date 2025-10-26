# -*- coding: utf-8 -*-
"""
Response Validation System
답변 검증 및 승인 시스템
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# from .improved_legal_restriction_system import ImprovedLegalRestrictionSystem, ImprovedRestrictionResult, RestrictionLevel
# from .content_filter_engine import ContentFilterEngine, FilterResult, IntentType

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """검증 상태"""
    APPROVED = "approved"          # 승인
    REJECTED = "rejected"          # 거부
    MODIFIED = "modified"          # 수정됨
    PENDING_REVIEW = "pending_review"  # 검토 대기


class ValidationLevel(Enum):
    """검증 수준"""
    AUTOMATIC = "automatic"        # 자동 검증
    MANUAL_REVIEW = "manual_review"  # 수동 검토
    EXPERT_REVIEW = "expert_review"  # 전문가 검토


@dataclass
class ValidationResult:
    """검증 결과"""
    status: ValidationStatus
    validation_level: ValidationLevel
    confidence: float
    issues: List[str]
    recommendations: List[str]
    modified_response: Optional[str]
    validation_details: Dict[str, Any]
    timestamp: datetime


@dataclass
class ValidationRule:
    """검증 규칙"""
    id: str
    name: str
    description: str
    priority: int
    enabled: bool
    validation_function: str


class ResponseValidationSystem:
    """답변 검증 및 승인 시스템"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # self.legal_restriction_system = ImprovedLegalRestrictionSystem()
        # self.content_filter_engine = ContentFilterEngine()
        self.legal_restriction_system = None
        self.content_filter_engine = None
        self.validation_rules = self._initialize_validation_rules()
        self.validation_history = []

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """검증 규칙 초기화"""
        return [
            ValidationRule(
                id="legal_restriction_check",
                name="법률 제한 검사",
                description="법률적 제한 사항 검사",
                priority=1,
                enabled=True,
                validation_function="check_legal_restrictions"
            ),
            ValidationRule(
                id="content_filter_check",
                name="콘텐츠 필터 검사",
                description="의도 분석 및 콘텐츠 필터링",
                priority=2,
                enabled=True,
                validation_function="check_content_filter"
            ),
            ValidationRule(
                id="response_quality_check",
                name="답변 품질 검사",
                description="답변의 품질 및 완성도 검사",
                priority=3,
                enabled=True,
                validation_function="check_response_quality"
            ),
            ValidationRule(
                id="safety_check",
                name="안전성 검사",
                description="답변의 안전성 및 적절성 검사",
                priority=4,
                enabled=True,
                validation_function="check_safety"
            ),
            ValidationRule(
                id="compliance_check",
                name="법적 준수 검사",
                description="법적 준수 사항 검사",
                priority=5,
                enabled=True,
                validation_function="check_compliance"
            )
        ]

    def validate_response(self, query: str, response: str,
                         context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """답변 검증"""
        try:
            self.logger.info(f"Starting response validation for query: {query[:100]}...")

            validation_details = {}
            all_issues = []
            all_recommendations = []
            overall_confidence = 1.0
            validation_level = ValidationLevel.AUTOMATIC

            # 각 검증 규칙 실행
            for rule in self.validation_rules:
                if not rule.enabled:
                    continue

                rule_result = self._execute_validation_rule(rule, query, response, context)
                validation_details[rule.id] = rule_result

                # 이슈 및 권장사항 수집
                if rule_result.get("issues"):
                    all_issues.extend(rule_result["issues"])

                if rule_result.get("recommendations"):
                    all_recommendations.extend(rule_result["recommendations"])

                # 신뢰도 조정
                if rule_result.get("confidence"):
                    overall_confidence *= rule_result["confidence"]

                # 검증 수준 결정
                if rule_result.get("requires_manual_review"):
                    validation_level = ValidationLevel.MANUAL_REVIEW

                if rule_result.get("requires_expert_review"):
                    validation_level = ValidationLevel.EXPERT_REVIEW

            # 최종 상태 결정
            status = self._determine_validation_status(
                validation_details, all_issues, overall_confidence
            )

            # 수정된 답변 생성
            modified_response = None
            if status == ValidationStatus.MODIFIED:
                modified_response = self._generate_modified_response(
                    response, all_issues, all_recommendations
                )

            # 검증 결과 생성
            result = ValidationResult(
                status=status,
                validation_level=validation_level,
                confidence=overall_confidence,
                issues=all_issues,
                recommendations=all_recommendations,
                modified_response=modified_response,
                validation_details=validation_details,
                timestamp=datetime.now()
            )

            # 검증 히스토리 저장
            self._save_validation_history(query, response, result)

            self.logger.info(f"Response validation completed. Status: {status.value}")
            return result

        except Exception as e:
            self.logger.error(f"Error validating response: {e}")
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                validation_level=ValidationLevel.MANUAL_REVIEW,
                confidence=0.0,
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Manual review required"],
                modified_response=None,
                validation_details={"error": str(e)},
                timestamp=datetime.now()
            )

    def _execute_validation_rule(self, rule: ValidationRule, query: str,
                                response: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """검증 규칙 실행"""
        try:
            if rule.validation_function == "check_legal_restrictions":
                return self._check_legal_restrictions(query, response)
            elif rule.validation_function == "check_content_filter":
                return self._check_content_filter(query, response)
            elif rule.validation_function == "check_response_quality":
                return self._check_response_quality(query, response)
            elif rule.validation_function == "check_safety":
                return self._check_safety(query, response)
            elif rule.validation_function == "check_compliance":
                return self._check_compliance(query, response)
            else:
                return {"error": f"Unknown validation function: {rule.validation_function}"}
        except Exception as e:
            self.logger.error(f"Error executing validation rule {rule.id}: {e}")
            return {"error": str(e)}

    def _check_legal_restrictions(self, query: str, response: str) -> Dict[str, Any]:
        """법률 제한 검사"""
        try:
            restriction_result = self.legal_restriction_system.check_restrictions(query, response)

            issues = []
            recommendations = []
            confidence = 1.0
            requires_manual_review = False
            requires_expert_review = False

            if restriction_result.is_restricted:
                if restriction_result.restriction_level == RestrictionLevel.CRITICAL:
                    issues.append("법적으로 매우 민감한 내용이 포함되어 있습니다.")
                    recommendations.append("전문가 검토가 필요합니다.")
                    requires_expert_review = True
                    confidence = 0.0
                elif restriction_result.restriction_level == RestrictionLevel.HIGH:
                    issues.append("법률 자문에 해당할 수 있는 내용이 포함되어 있습니다.")
                    recommendations.append("수동 검토가 필요합니다.")
                    requires_manual_review = True
                    confidence = 0.3
                else:
                    issues.append("주의가 필요한 법률적 내용이 포함되어 있습니다.")
                    recommendations.append("안전한 대안 답변을 사용하세요.")
                    confidence = 0.6

            return {
                "restriction_result": restriction_result,
                "issues": issues,
                "recommendations": recommendations,
                "confidence": confidence,
                "requires_manual_review": requires_manual_review,
                "requires_expert_review": requires_expert_review
            }

        except Exception as e:
            self.logger.error(f"Error checking legal restrictions: {e}")
            return {"error": str(e)}

    def _check_content_filter(self, query: str, response: str) -> Dict[str, Any]:
        """콘텐츠 필터 검사"""
        try:
            filter_result = self.content_filter_engine.filter_content(query, response)

            issues = []
            recommendations = []
            confidence = 1.0
            requires_manual_review = False

            if filter_result.is_blocked:
                issues.append(filter_result.block_reason)
                recommendations.extend(filter_result.safe_alternatives)
                confidence = 0.2
                requires_manual_review = True

            # 의도 분석 결과 기반 추가 검사
            intent_analysis = filter_result.intent_analysis
            if intent_analysis.intent_type == IntentType.SUSPICIOUS_REQUEST:
                issues.append("의심스러운 요청이 감지되었습니다.")
                recommendations.append("보안 검토가 필요합니다.")
                confidence = 0.0
                requires_manual_review = True

            return {
                "filter_result": filter_result,
                "intent_analysis": intent_analysis,
                "issues": issues,
                "recommendations": recommendations,
                "confidence": confidence,
                "requires_manual_review": requires_manual_review
            }

        except Exception as e:
            self.logger.error(f"Error checking content filter: {e}")
            return {"error": str(e)}

    def _check_response_quality(self, query: str, response: str) -> Dict[str, Any]:
        """답변 품질 검사"""
        try:
            issues = []
            recommendations = []
            confidence = 1.0

            # 답변 길이 검사
            if len(response) < 10:
                issues.append("답변이 너무 짧습니다.")
                recommendations.append("더 자세한 답변을 제공하세요.")
                confidence *= 0.7

            if len(response) > 5000:
                issues.append("답변이 너무 깁니다.")
                recommendations.append("답변을 간결하게 정리하세요.")
                confidence *= 0.8

            # 답변 완성도 검사
            if not response.strip():
                issues.append("답변이 비어있습니다.")
                recommendations.append("적절한 답변을 제공하세요.")
                confidence = 0.0

            # 문법적 완성도 검사
            if not response.endswith(('.', '!', '?', '다', '요', '니다')):
                issues.append("답변이 문법적으로 완성되지 않았습니다.")
                recommendations.append("답변을 문법적으로 완성하세요.")
                confidence *= 0.9

            # 질문과 답변의 관련성 검사
            if not self._check_relevance(query, response):
                issues.append("답변이 질문과 관련성이 낮습니다.")
                recommendations.append("질문에 더 직접적으로 답변하세요.")
                confidence *= 0.8

            return {
                "issues": issues,
                "recommendations": recommendations,
                "confidence": confidence
            }

        except Exception as e:
            self.logger.error(f"Error checking response quality: {e}")
            return {"error": str(e)}

    def _check_safety(self, query: str, response: str) -> Dict[str, Any]:
        """안전성 검사"""
        try:
            issues = []
            recommendations = []
            confidence = 1.0

            # 위험한 키워드 검사
            dangerous_keywords = [
                "자살", "자해", "살인", "폭력", "테러", "폭탄",
                "마약", "도박", "사기", "절도", "강도"
            ]

            response_lower = response.lower()
            for keyword in dangerous_keywords:
                if keyword in response_lower:
                    issues.append(f"위험한 키워드 '{keyword}'가 포함되어 있습니다.")
                    recommendations.append("안전한 대안 답변을 사용하세요.")
                    confidence *= 0.5

            # 개인정보 노출 검사
            personal_info_patterns = [
                r'\d{3}-\d{4}-\d{4}',  # 전화번호
                r'\d{6}-\d{7}',        # 주민등록번호
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # 이메일
            ]

            import re
            for pattern in personal_info_patterns:
                if re.search(pattern, response):
                    issues.append("개인정보가 포함되어 있습니다.")
                    recommendations.append("개인정보를 제거하세요.")
                    confidence *= 0.3

            return {
                "issues": issues,
                "recommendations": recommendations,
                "confidence": confidence
            }

        except Exception as e:
            self.logger.error(f"Error checking safety: {e}")
            return {"error": str(e)}

    def _check_compliance(self, query: str, response: str) -> Dict[str, Any]:
        """법적 준수 검사"""
        try:
            issues = []
            recommendations = []
            confidence = 1.0

            # 변호사법 준수 검사
            if self._contains_legal_advice(response):
                issues.append("변호사법 위반 가능성이 있습니다.")
                recommendations.append("법률 자문을 제공하지 마세요.")
                confidence *= 0.2

            # 의료법 준수 검사
            if self._contains_medical_advice(response):
                issues.append("의료법 위반 가능성이 있습니다.")
                recommendations.append("의료 조언을 제공하지 마세요.")
                confidence *= 0.3

            # 형사법 준수 검사
            if self._contains_criminal_advice(response):
                issues.append("형사법 위반 가능성이 있습니다.")
                recommendations.append("형사 조언을 제공하지 마세요.")
                confidence *= 0.2

            return {
                "issues": issues,
                "recommendations": recommendations,
                "confidence": confidence
            }

        except Exception as e:
            self.logger.error(f"Error checking compliance: {e}")
            return {"error": str(e)}

    def _check_relevance(self, query: str, response: str) -> bool:
        """질문과 답변의 관련성 검사"""
        try:
            # 간단한 키워드 매칭으로 관련성 검사
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())

            # 공통 단어 비율 계산
            common_words = query_words.intersection(response_words)
            relevance_ratio = len(common_words) / len(query_words) if query_words else 0

            return relevance_ratio > 0.3

        except Exception as e:
            self.logger.error(f"Error checking relevance: {e}")
            return True

    def _contains_legal_advice(self, response: str) -> bool:
        """법률 자문 포함 여부 검사"""
        legal_advice_indicators = [
            "소송하세요", "소송하지 마세요", "승소할 것입니다", "패소할 것입니다",
            "당신의 경우", "귀하의 경우", "반드시 해야", "꼭 해야"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in legal_advice_indicators)

    def _contains_medical_advice(self, response: str) -> bool:
        """의료 조언 포함 여부 검사"""
        medical_advice_indicators = [
            "의료사고의 과실", "의료진이 잘못", "장애등급은", "의학적 인과관계",
            "치료하세요", "약을 드세요", "수술하세요"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in medical_advice_indicators)

    def _contains_criminal_advice(self, response: str) -> bool:
        """형사 조언 포함 여부 검사"""
        criminal_advice_indicators = [
            "자백하세요", "부인하세요", "증거를 인멸", "증거를 숨기",
            "형량은", "몇 년 형", "범죄 수법"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in criminal_advice_indicators)

    def _determine_validation_status(self, validation_details: Dict[str, Any],
                                   issues: List[str], confidence: float) -> ValidationStatus:
        """검증 상태 결정"""
        # 심각한 이슈가 있으면 거부
        critical_issues = [issue for issue in issues if "매우 민감" in issue or "위반" in issue]
        if critical_issues:
            return ValidationStatus.REJECTED

        # 신뢰도가 낮으면 수정 필요
        if confidence < 0.5:
            return ValidationStatus.MODIFIED

        # 이슈가 있으면 수정 필요
        if issues:
            return ValidationStatus.MODIFIED

        # 모든 검사 통과
        return ValidationStatus.APPROVED

    def _generate_modified_response(self, original_response: str,
                                  issues: List[str], recommendations: List[str]) -> str:
        """수정된 답변 생성"""
        try:
            # 기본 안전한 응답
            safe_response = "죄송합니다. 구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다. "

            # 일반적인 정보 제공 가능
            if "일반적인" in str(recommendations):
                safe_response += "일반적인 법률 정보나 절차는 안내드릴 수 있습니다."
            else:
                safe_response += "관련 법령이나 절차에 대한 일반적인 정보를 안내드릴 수 있습니다."

            return safe_response

        except Exception as e:
            self.logger.error(f"Error generating modified response: {e}")
            return "죄송합니다. 구체적인 법률 자문은 변호사와 상담하시는 것이 좋습니다."

    def _save_validation_history(self, query: str, response: str, result: ValidationResult):
        """검증 히스토리 저장"""
        try:
            history_entry = {
                "timestamp": result.timestamp.isoformat(),
                "query": query[:200],  # 길이 제한
                "response": response[:200],  # 길이 제한
                "status": result.status.value,
                "confidence": result.confidence,
                "issues_count": len(result.issues),
                "validation_level": result.validation_level.value
            }

            self.validation_history.append(history_entry)

            # 히스토리 크기 제한 (최근 1000개만 유지)
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error saving validation history: {e}")

    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 정보"""
        try:
            if not self.validation_history:
                return {"total_validations": 0}

            total_validations = len(self.validation_history)
            status_counts = {}
            confidence_scores = []

            for entry in self.validation_history:
                status = entry["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
                confidence_scores.append(entry["confidence"])

            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            return {
                "total_validations": total_validations,
                "status_distribution": status_counts,
                "average_confidence": avg_confidence,
                "enabled_rules": len([rule for rule in self.validation_rules if rule.enabled]),
                "total_rules": len(self.validation_rules)
            }

        except Exception as e:
            self.logger.error(f"Error getting validation statistics: {e}")
            return {"error": str(e)}
