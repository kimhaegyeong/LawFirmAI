# -*- coding: utf-8 -*-
"""
실시간 대화 품질 피드백 시스템
사용자 피드백 기반 실시간 개선 시스템
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """피드백 유형"""
    EXPLICIT = "explicit"  # 명시적 피드백 (사용자가 직접 평가)
    IMPLICIT = "implicit"  # 암시적 피드백 (행동 패턴 분석)
    BEHAVIORAL = "behavioral"  # 행동 기반 피드백 (클릭, 체류시간 등)


class ImprovementPriority(Enum):
    """개선 우선순위"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FeedbackRecord:
    """피드백 기록 데이터 클래스"""
    session_id: str
    response_id: str
    feedback_type: FeedbackType
    satisfaction_score: float
    accuracy_score: float
    speed_score: float
    usability_score: float
    specific_feedback: str
    issue_types: List[str]
    timestamp: str
    user_id: Optional[str] = None
    question_type: Optional[str] = None
    response_length: Optional[int] = None


@dataclass
class ImprovementSuggestion:
    """개선 제안 데이터 클래스"""
    suggestion_id: str
    priority: ImprovementPriority
    category: str
    description: str
    implementation_hint: str
    expected_impact: float
    created_at: str
    applied: bool = False
    applied_at: Optional[str] = None


class RealtimeFeedbackSystem:
    """사용자 피드백 기반 실시간 개선"""
    
    def __init__(self, quality_monitor=None):
        """실시간 피드백 시스템 초기화"""
        self.logger = logging.getLogger(__name__)
        self.quality_monitor = quality_monitor
        
        # 피드백 버퍼 (메모리 기반)
        self.feedback_buffer: List[FeedbackRecord] = []
        self.improvement_suggestions: List[ImprovementSuggestion] = []
        
        # 실시간 개선 설정
        self.improvement_config = {
            "min_feedback_for_improvement": 3,
            "feedback_analysis_window": 10,  # 최근 10개 피드백 분석
            "improvement_application_threshold": 0.7,
            "feedback_decay_factor": 0.95,
            "max_suggestions_per_session": 5
        }
        
        # 개선 제안 템플릿
        self.improvement_templates = {
            "accuracy": {
                "high": [
                    "답변의 정확성을 높이기 위해 더 신뢰할 수 있는 소스를 참조하세요",
                    "법령 조문을 더 정확히 인용하고 최신 판례를 확인하세요",
                    "불확실한 내용은 명시하고 전문가 상담을 권유하세요"
                ],
                "medium": [
                    "답변에 더 구체적인 예시를 포함하세요",
                    "관련 법령이나 판례를 추가로 언급하세요",
                    "실무 적용 방법을 자세히 설명하세요"
                ],
                "low": [
                    "답변의 신뢰도를 높이기 위해 소스를 명시하세요",
                    "법적 근거를 더 명확히 제시하세요"
                ]
            },
            "speed": {
                "high": [
                    "답변 생성 시간을 최적화하세요",
                    "캐시를 활용하여 반복 질문에 빠르게 응답하세요",
                    "불필요한 처리 단계를 제거하세요"
                ],
                "medium": [
                    "답변 길이를 적절히 조절하세요",
                    "핵심 내용을 우선적으로 제공하세요"
                ],
                "low": [
                    "응답 시간을 모니터링하세요"
                ]
            },
            "usability": {
                "high": [
                    "답변을 더 이해하기 쉽게 구성하세요",
                    "단계별 설명을 추가하세요",
                    "사용자 친화적인 언어를 사용하세요"
                ],
                "medium": [
                    "답변 구조를 개선하세요",
                    "핵심 내용을 요약하여 제시하세요"
                ],
                "low": [
                    "답변 형식을 일관성 있게 유지하세요"
                ]
            },
            "satisfaction": {
                "high": [
                    "사용자의 감정 상태를 고려한 응답 톤을 사용하세요",
                    "공감적이고 이해하기 쉬운 언어를 사용하세요",
                    "사용자의 전문성 수준에 맞는 설명을 제공하세요"
                ],
                "medium": [
                    "더 친근하고 자연스러운 톤으로 답변하세요",
                    "사용자의 상황을 고려한 맞춤형 조언을 제공하세요"
                ],
                "low": [
                    "답변의 친근함을 높이세요"
                ]
            }
        }
        
        # 피드백 패턴 분석 결과
        self.pattern_analysis = {}
        
        self.logger.info("RealtimeFeedbackSystem initialized")
    
    def collect_feedback(self, session_id: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 피드백 수집 및 분석
        
        Args:
            session_id: 세션 ID
            feedback_data: 피드백 데이터
            
        Returns:
            Dict[str, Any]: 피드백 처리 결과
        """
        try:
            # 1. 피드백 기록 생성
            feedback_record = self._create_feedback_record(session_id, feedback_data)
            
            # 2. 피드백 버퍼에 추가
            self.feedback_buffer.append(feedback_record)
            
            # 3. 즉시 분석 및 개선 제안
            improvement_suggestions = self._analyze_feedback_immediately(feedback_record)
            
            # 4. 실시간 개선 적용
            if improvement_suggestions:
                self._apply_immediate_improvements(improvement_suggestions)
            
            # 5. 패턴 분석 업데이트
            self._update_pattern_analysis(feedback_record)
            
            return {
                "feedback_received": True,
                "feedback_id": feedback_record.response_id,
                "improvement_suggestions": improvement_suggestions,
                "next_response_will_be_improved": len(improvement_suggestions) > 0,
                "pattern_insights": self._get_pattern_insights()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {e}")
            return {"feedback_received": False, "error": str(e)}
    
    def _create_feedback_record(self, session_id: str, feedback_data: Dict[str, Any]) -> FeedbackRecord:
        """피드백 기록 생성"""
        try:
            return FeedbackRecord(
                session_id=session_id,
                response_id=feedback_data.get("response_id", f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                feedback_type=FeedbackType(feedback_data.get("type", "implicit")),
                satisfaction_score=feedback_data.get("satisfaction", 0.5),
                accuracy_score=feedback_data.get("accuracy", 0.5),
                speed_score=feedback_data.get("speed", 0.5),
                usability_score=feedback_data.get("usability", 0.5),
                specific_feedback=feedback_data.get("comments", ""),
                issue_types=feedback_data.get("issue_types", []),
                timestamp=datetime.now().isoformat(),
                user_id=feedback_data.get("user_id"),
                question_type=feedback_data.get("question_type"),
                response_length=feedback_data.get("response_length")
            )
        except Exception as e:
            self.logger.error(f"Error creating feedback record: {e}")
            # 기본값으로 생성
            return FeedbackRecord(
                session_id=session_id,
                response_id=f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                feedback_type=FeedbackType.IMPLICIT,
                satisfaction_score=0.5,
                accuracy_score=0.5,
                speed_score=0.5,
                usability_score=0.5,
                specific_feedback="",
                issue_types=[],
                timestamp=datetime.now().isoformat()
            )
    
    def _analyze_feedback_immediately(self, feedback_record: FeedbackRecord) -> List[ImprovementSuggestion]:
        """피드백 즉시 분석 및 개선 제안 생성"""
        try:
            suggestions = []
            
            # 만족도 기반 제안
            if feedback_record.satisfaction_score < 0.3:
                suggestions.extend(self._generate_high_priority_suggestions(feedback_record))
            elif feedback_record.satisfaction_score < 0.6:
                suggestions.extend(self._generate_medium_priority_suggestions(feedback_record))
            else:
                suggestions.extend(self._generate_low_priority_suggestions(feedback_record))
            
            # 세부 점수 기반 제안
            if feedback_record.accuracy_score < 0.4:
                suggestions.extend(self._get_accuracy_suggestions("high"))
            elif feedback_record.accuracy_score < 0.7:
                suggestions.extend(self._get_accuracy_suggestions("medium"))
            
            if feedback_record.speed_score < 0.4:
                suggestions.extend(self._get_speed_suggestions("high"))
            elif feedback_record.speed_score < 0.7:
                suggestions.extend(self._get_speed_suggestions("medium"))
            
            if feedback_record.usability_score < 0.4:
                suggestions.extend(self._get_usability_suggestions("high"))
            elif feedback_record.usability_score < 0.7:
                suggestions.extend(self._get_usability_suggestions("medium"))
            
            # 텍스트 피드백 분석
            text_suggestions = self._analyze_text_feedback(feedback_record.specific_feedback)
            suggestions.extend(text_suggestions)
            
            # 중복 제거 및 우선순위 정렬
            unique_suggestions = self._deduplicate_and_prioritize(suggestions)
            
            return unique_suggestions[:self.improvement_config["max_suggestions_per_session"]]
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback immediately: {e}")
            return []
    
    def _generate_high_priority_suggestions(self, feedback_record: FeedbackRecord) -> List[ImprovementSuggestion]:
        """높은 우선순위 개선 제안 생성"""
        suggestions = []
        
        # 만족도가 매우 낮은 경우
        if feedback_record.satisfaction_score < 0.3:
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"high_satisfaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority.HIGH,
                category="satisfaction",
                description="전체적인 답변 품질을 대폭 개선하세요",
                implementation_hint="사용자 피드백을 반영하여 답변 스타일과 내용을 전면 개선",
                expected_impact=0.8,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _generate_medium_priority_suggestions(self, feedback_record: FeedbackRecord) -> List[ImprovementSuggestion]:
        """중간 우선순위 개선 제안 생성"""
        suggestions = []
        
        if feedback_record.satisfaction_score < 0.6:
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"medium_satisfaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority.MEDIUM,
                category="satisfaction",
                description="답변의 친근함과 이해도를 높이세요",
                implementation_hint="더 자연스럽고 친근한 톤으로 답변하고 예시를 추가",
                expected_impact=0.6,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _generate_low_priority_suggestions(self, feedback_record: FeedbackRecord) -> List[ImprovementSuggestion]:
        """낮은 우선순위 개선 제안 생성"""
        suggestions = []
        
        # 만족도가 높은 경우에도 미세한 개선 제안
        suggestions.append(ImprovementSuggestion(
            suggestion_id=f"low_satisfaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            priority=ImprovementPriority.LOW,
            category="satisfaction",
            description="현재 스타일을 유지하면서 미세한 개선을 적용하세요",
            implementation_hint="사용자가 만족하는 현재 스타일을 유지하면서 세부 사항 개선",
            expected_impact=0.3,
            created_at=datetime.now().isoformat()
        ))
        
        return suggestions
    
    def _get_accuracy_suggestions(self, priority: str) -> List[ImprovementSuggestion]:
        """정확성 관련 개선 제안"""
        suggestions = []
        
        for template in self.improvement_templates["accuracy"].get(priority, []):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"accuracy_{priority}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority(priority),
                category="accuracy",
                description=template,
                implementation_hint=f"정확성 개선을 위한 {priority} 우선순위 조치",
                expected_impact=0.7 if priority == "high" else 0.5,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _get_speed_suggestions(self, priority: str) -> List[ImprovementSuggestion]:
        """속도 관련 개선 제안"""
        suggestions = []
        
        for template in self.improvement_templates["speed"].get(priority, []):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"speed_{priority}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority(priority),
                category="speed",
                description=template,
                implementation_hint=f"응답 속도 개선을 위한 {priority} 우선순위 조치",
                expected_impact=0.6 if priority == "high" else 0.4,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _get_usability_suggestions(self, priority: str) -> List[ImprovementSuggestion]:
        """사용성 관련 개선 제안"""
        suggestions = []
        
        for template in self.improvement_templates["usability"].get(priority, []):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"usability_{priority}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority(priority),
                category="usability",
                description=template,
                implementation_hint=f"사용성 개선을 위한 {priority} 우선순위 조치",
                expected_impact=0.5 if priority == "high" else 0.3,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _analyze_text_feedback(self, text_feedback: str) -> List[ImprovementSuggestion]:
        """텍스트 피드백 분석"""
        suggestions = []
        
        if not text_feedback:
            return suggestions
        
        text_lower = text_feedback.lower()
        
        # 키워드 기반 분석
        if any(keyword in text_lower for keyword in ["너무 길어", "복잡해", "어려워"]):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"text_simplify_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority.MEDIUM,
                category="usability",
                description="답변을 더 간단하고 명확하게 구성하세요",
                implementation_hint="복잡한 내용을 단계별로 나누어 설명하고 핵심만 요약",
                expected_impact=0.6,
                created_at=datetime.now().isoformat()
            ))
        
        if any(keyword in text_lower for keyword in ["이해가 안 돼", "모르겠어", "혼란스러워"]):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"text_clarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority.HIGH,
                category="usability",
                description="단계별로 더 자세히 설명하세요",
                implementation_hint="예시를 추가하고 단계별 설명으로 이해도 향상",
                expected_impact=0.7,
                created_at=datetime.now().isoformat()
            ))
        
        if any(keyword in text_lower for keyword in ["친근하지 않아", "딱딱해", "차갑다"]):
            suggestions.append(ImprovementSuggestion(
                suggestion_id=f"text_tone_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                priority=ImprovementPriority.MEDIUM,
                category="satisfaction",
                description="더 친근하고 자연스러운 톤을 사용하세요",
                implementation_hint="격식적인 표현을 자연스러운 표현으로 변경",
                expected_impact=0.5,
                created_at=datetime.now().isoformat()
            ))
        
        return suggestions
    
    def _deduplicate_and_prioritize(self, suggestions: List[ImprovementSuggestion]) -> List[ImprovementSuggestion]:
        """중복 제거 및 우선순위 정렬"""
        try:
            # 중복 제거 (description 기준)
            unique_suggestions = []
            seen_descriptions = set()
            
            for suggestion in suggestions:
                if suggestion.description not in seen_descriptions:
                    unique_suggestions.append(suggestion)
                    seen_descriptions.add(suggestion.description)
            
            # 우선순위별 정렬 (HIGH > MEDIUM > LOW)
            priority_order = {
                ImprovementPriority.HIGH: 0,
                ImprovementPriority.MEDIUM: 1,
                ImprovementPriority.LOW: 2
            }
            
            unique_suggestions.sort(key=lambda x: priority_order[x.priority])
            
            return unique_suggestions
            
        except Exception as e:
            self.logger.error(f"Error deduplicating and prioritizing: {e}")
            return suggestions
    
    def _apply_immediate_improvements(self, suggestions: List[ImprovementSuggestion]) -> None:
        """즉시 개선사항 적용"""
        try:
            # 다음 응답에 적용할 개선사항 저장
            self.improvement_suggestions.extend(suggestions)
            
            # 글로벌 설정 업데이트 (필요시)
            self._update_global_settings(suggestions)
            
            self.logger.info(f"Applied {len(suggestions)} immediate improvements")
            
        except Exception as e:
            self.logger.error(f"Error applying immediate improvements: {e}")
    
    def _update_global_settings(self, suggestions: List[ImprovementSuggestion]) -> None:
        """글로벌 설정 업데이트"""
        try:
            # 우선순위가 높은 제안들을 글로벌 설정에 반영
            high_priority_suggestions = [s for s in suggestions if s.priority == ImprovementPriority.HIGH]
            
            if high_priority_suggestions:
                # 글로벌 설정 업데이트 로직
                self.logger.info(f"Updated global settings with {len(high_priority_suggestions)} high priority suggestions")
                
        except Exception as e:
            self.logger.error(f"Error updating global settings: {e}")
    
    def _update_pattern_analysis(self, feedback_record: FeedbackRecord) -> None:
        """패턴 분석 업데이트"""
        try:
            # 세션별 패턴 분석
            session_id = feedback_record.session_id
            
            if session_id not in self.pattern_analysis:
                self.pattern_analysis[session_id] = {
                    "feedback_count": 0,
                    "avg_satisfaction": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_speed": 0.0,
                    "avg_usability": 0.0,
                    "common_issues": [],
                    "improvement_trend": "stable"
                }
            
            # 통계 업데이트
            pattern = self.pattern_analysis[session_id]
            pattern["feedback_count"] += 1
            
            # 평균 점수 업데이트
            n = pattern["feedback_count"]
            pattern["avg_satisfaction"] = (pattern["avg_satisfaction"] * (n-1) + feedback_record.satisfaction_score) / n
            pattern["avg_accuracy"] = (pattern["avg_accuracy"] * (n-1) + feedback_record.accuracy_score) / n
            pattern["avg_speed"] = (pattern["avg_speed"] * (n-1) + feedback_record.speed_score) / n
            pattern["avg_usability"] = (pattern["avg_usability"] * (n-1) + feedback_record.usability_score) / n
            
            # 일반적인 문제점 추적
            if feedback_record.issue_types:
                pattern["common_issues"].extend(feedback_record.issue_types)
            
        except Exception as e:
            self.logger.error(f"Error updating pattern analysis: {e}")
    
    def _get_pattern_insights(self) -> Dict[str, Any]:
        """패턴 인사이트 반환"""
        try:
            if not self.pattern_analysis:
                return {"insights": "아직 충분한 피드백 데이터가 없습니다"}
            
            # 전체 세션 통계
            total_sessions = len(self.pattern_analysis)
            total_feedback = sum(pattern["feedback_count"] for pattern in self.pattern_analysis.values())
            
            # 평균 점수 계산
            avg_satisfaction = sum(pattern["avg_satisfaction"] for pattern in self.pattern_analysis.values()) / total_sessions
            avg_accuracy = sum(pattern["avg_accuracy"] for pattern in self.pattern_analysis.values()) / total_sessions
            avg_speed = sum(pattern["avg_speed"] for pattern in self.pattern_analysis.values()) / total_sessions
            avg_usability = sum(pattern["avg_usability"] for pattern in self.pattern_analysis.values()) / total_sessions
            
            return {
                "total_sessions": total_sessions,
                "total_feedback": total_feedback,
                "avg_satisfaction": round(avg_satisfaction, 2),
                "avg_accuracy": round(avg_accuracy, 2),
                "avg_speed": round(avg_speed, 2),
                "avg_usability": round(avg_usability, 2),
                "insights": self._generate_insights(avg_satisfaction, avg_accuracy, avg_speed, avg_usability)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pattern insights: {e}")
            return {"insights": "패턴 분석 중 오류가 발생했습니다"}
    
    def _generate_insights(self, satisfaction: float, accuracy: float, speed: float, usability: float) -> str:
        """인사이트 생성"""
        try:
            insights = []
            
            if satisfaction < 0.5:
                insights.append("전체적인 만족도가 낮습니다. 답변 품질을 전면적으로 개선해야 합니다.")
            elif satisfaction > 0.8:
                insights.append("사용자 만족도가 높습니다. 현재 스타일을 유지하세요.")
            
            if accuracy < 0.6:
                insights.append("답변 정확성이 부족합니다. 더 신뢰할 수 있는 소스를 활용하세요.")
            
            if speed < 0.6:
                insights.append("응답 속도가 느립니다. 성능 최적화가 필요합니다.")
            
            if usability < 0.6:
                insights.append("사용성이 부족합니다. 더 이해하기 쉬운 답변을 제공하세요.")
            
            return " ".join(insights) if insights else "전반적으로 양호한 수준입니다."
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return "인사이트 생성 중 오류가 발생했습니다"
    
    def get_next_response_improvements(self) -> List[str]:
        """다음 응답에 적용할 개선사항 반환"""
        try:
            improvements = []
            
            for suggestion in self.improvement_suggestions:
                if not suggestion.applied:
                    improvements.append(suggestion.description)
            
            # 사용 후 초기화
            self.improvement_suggestions = []
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Error getting next response improvements: {e}")
            return []
    
    def get_feedback_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """피드백 요약 반환"""
        try:
            if session_id:
                # 특정 세션의 피드백
                session_feedback = [f for f in self.feedback_buffer if f.session_id == session_id]
            else:
                # 전체 피드백
                session_feedback = self.feedback_buffer
            
            if not session_feedback:
                return {"message": "피드백 데이터가 없습니다"}
            
            # 통계 계산
            total_feedback = len(session_feedback)
            avg_satisfaction = sum(f.satisfaction_score for f in session_feedback) / total_feedback
            avg_accuracy = sum(f.accuracy_score for f in session_feedback) / total_feedback
            avg_speed = sum(f.speed_score for f in session_feedback) / total_feedback
            avg_usability = sum(f.usability_score for f in session_feedback) / total_feedback
            
            # 일반적인 문제점
            all_issues = []
            for feedback in session_feedback:
                all_issues.extend(feedback.issue_types)
            
            common_issues = list(set(all_issues))
            
            return {
                "total_feedback": total_feedback,
                "avg_satisfaction": round(avg_satisfaction, 2),
                "avg_accuracy": round(avg_accuracy, 2),
                "avg_speed": round(avg_speed, 2),
                "avg_usability": round(avg_usability, 2),
                "common_issues": common_issues,
                "pattern_insights": self._get_pattern_insights()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feedback summary: {e}")
            return {"error": str(e)}
    
    def clear_old_feedback(self, days: int = 7) -> int:
        """오래된 피드백 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            initial_count = len(self.feedback_buffer)
            
            self.feedback_buffer = [
                f for f in self.feedback_buffer 
                if datetime.fromisoformat(f.timestamp) > cutoff_date
            ]
            
            removed_count = initial_count - len(self.feedback_buffer)
            self.logger.info(f"Cleared {removed_count} old feedback records")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error clearing old feedback: {e}")
            return 0
