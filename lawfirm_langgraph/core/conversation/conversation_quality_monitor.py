# -*- coding: utf-8 -*-
"""
대화 품질 모니터
대화 품질을 모니터링하고 개선점을 제안합니다.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from ..data.conversation_store import ConversationStore
from .conversation_manager import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """품질 지표"""
    completeness_score: float
    satisfaction_score: float
    accuracy_score: float
    response_time: float
    issues_detected: List[str]
    timestamp: datetime


@dataclass
class QualityTrend:
    """품질 트렌드"""
    period: str
    avg_completeness: float
    avg_satisfaction: float
    avg_accuracy: float
    avg_response_time: float
    total_sessions: int
    improvement_suggestions: List[str]


class ConversationQualityMonitor:
    """대화 품질 모니터"""
    
    def __init__(self, conversation_store: Optional[ConversationStore] = None):
        """
        대화 품질 모니터 초기화
        
        Args:
            conversation_store: 대화 저장소 (None이면 새로 생성)
        """
        self.logger = logging.getLogger(__name__)
        self.conversation_store = conversation_store or ConversationStore()
        
        # 품질 평가 기준
        self.quality_criteria = {
            "completeness": {
                "keywords": ["완전", "완벽", "충분", "자세", "구체", "상세"],
                "negative_keywords": ["부족", "모자람", "부족함", "불완전", "미흡"],
                "min_length": 50,  # 최소 답변 길이
                "max_length": 2000  # 최대 답변 길이
            },
            "satisfaction": {
                "positive_keywords": ["만족", "좋아", "훌륭", "완벽", "감사", "도움"],
                "negative_keywords": ["불만", "화나", "답답", "이상", "문제", "틀림"],
                "neutral_keywords": ["보통", "그냥", "괜찮", "평범"]
            },
            "accuracy": {
                "legal_keywords": ["법", "조문", "조항", "판례", "법원", "법정", "법률"],
                "citation_patterns": [
                    r"(\w+)법\s+제(\d+)조",
                    r"(\w+)판례",
                    r"(\w+)법령",
                    r"(\w+)규정"
                ],
                "uncertainty_keywords": ["아마", "추정", "가능성", "아마도", "일반적으로"]
            }
        }
        
        # 문제점 감지 패턴
        self.issue_patterns = {
            "incomplete_response": [
                r"더\s+자세히",
                r"추가로",
                r"완전하지",
                r"부족한",
                r"모자란"
            ],
            "unclear_response": [
                r"이해\s*안\s*돼",
                r"모르겠",
                r"복잡해",
                r"헷갈려",
                r"명확하지"
            ],
            "inaccurate_response": [
                r"틀렸",
                r"잘못",
                r"아니다",
                r"그게\s*아니",
                r"정정"
            ],
            "slow_response": [
                r"느려",
                r"오래\s*걸려",
                r"시간\s*걸려",
                r"지연"
            ],
            "irrelevant_response": [
                r"관련\s*없",
                r"다른\s*질문",
                r"잘못\s*이해",
                r"오해"
            ]
        }
        
        # 개선 제안 템플릿
        self.improvement_suggestions = {
            "completeness": [
                "답변에 더 구체적인 예시를 포함하세요",
                "관련 법령이나 판례를 추가로 언급하세요",
                "실무 적용 방법을 자세히 설명하세요"
            ],
            "satisfaction": [
                "사용자의 감정 상태를 고려한 응답 톤을 사용하세요",
                "공감적이고 이해하기 쉬운 언어를 사용하세요",
                "사용자의 전문성 수준에 맞는 설명을 제공하세요"
            ],
            "accuracy": [
                "법령 조문을 정확히 인용하세요",
                "최신 판례 정보를 확인하세요",
                "불확실한 내용은 명시하세요"
            ],
            "response_time": [
                "답변 생성 시간을 최적화하세요",
                "캐시를 활용하여 반복 질문에 빠르게 응답하세요",
                "불필요한 처리 단계를 제거하세요"
            ]
        }
        
        self.logger.info("ConversationQualityMonitor initialized")
    
    def assess_conversation_quality(self, context: ConversationContext) -> Dict[str, Any]:
        """
        대화 품질 평가
        
        Args:
            context: 대화 맥락
            
        Returns:
            Dict[str, Any]: 품질 평가 결과
        """
        try:
            if not context.turns:
                return {
                    "overall_score": 0.0,
                    "completeness_score": 0.0,
                    "satisfaction_score": 0.0,
                    "accuracy_score": 0.0,
                    "issues": [],
                    "suggestions": []
                }
            
            # 각 턴의 품질 평가
            turn_scores = []
            all_issues = []
            
            for turn in context.turns:
                turn_quality = self.calculate_turn_quality(turn, context)
                turn_scores.append(turn_quality)
                
                # 문제점 감지
                issues = self.detect_turn_issues(turn)
                all_issues.extend(issues)
            
            # 전체 품질 점수 계산
            avg_completeness = statistics.mean([score["completeness"] for score in turn_scores])
            avg_satisfaction = statistics.mean([score["satisfaction"] for score in turn_scores])
            avg_accuracy = statistics.mean([score["accuracy"] for score in turn_scores])
            
            # 전체 점수 (가중 평균)
            overall_score = (avg_completeness * 0.4 + avg_satisfaction * 0.3 + avg_accuracy * 0.3)
            
            # 개선 제안 생성
            suggestions = self.generate_improvement_suggestions(
                avg_completeness, avg_satisfaction, avg_accuracy, all_issues
            )
            
            # 품질 메트릭 저장
            quality_metrics = QualityMetrics(
                completeness_score=avg_completeness,
                satisfaction_score=avg_satisfaction,
                accuracy_score=avg_accuracy,
                response_time=0.0,  # 실제 응답 시간은 별도 측정
                issues_detected=all_issues,
                timestamp=datetime.now()
            )
            
            self._store_quality_metrics(context.session_id, quality_metrics)
            
            return {
                "overall_score": overall_score,
                "completeness_score": avg_completeness,
                "satisfaction_score": avg_satisfaction,
                "accuracy_score": avg_accuracy,
                "turn_scores": turn_scores,
                "issues": list(set(all_issues)),  # 중복 제거
                "suggestions": suggestions,
                "assessment_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing conversation quality: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e)
            }
    
    def detect_conversation_issues(self, context: ConversationContext) -> List[str]:
        """
        대화 문제점 감지
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 감지된 문제점들
        """
        try:
            issues = []
            
            # 각 턴의 문제점 감지
            for turn in context.turns:
                turn_issues = self.detect_turn_issues(turn)
                issues.extend(turn_issues)
            
            # 대화 전체의 문제점 감지
            conversation_issues = self._detect_conversation_level_issues(context)
            issues.extend(conversation_issues)
            
            return list(set(issues))  # 중복 제거
            
        except Exception as e:
            self.logger.error(f"Error detecting conversation issues: {e}")
            return [f"문제점 감지 오류: {str(e)}"]
    
    def suggest_improvements(self, context: ConversationContext) -> List[str]:
        """
        개선 제안 생성
        
        Args:
            context: 대화 맥락
            
        Returns:
            List[str]: 개선 제안들
        """
        try:
            # 품질 평가 수행
            quality_assessment = self.assess_conversation_quality(context)
            
            # 문제점 기반 제안
            issue_based_suggestions = []
            for issue in quality_assessment.get("issues", []):
                if "incomplete" in issue.lower():
                    issue_based_suggestions.extend(self.improvement_suggestions["completeness"])
                elif "unclear" in issue.lower():
                    issue_based_suggestions.extend(self.improvement_suggestions["satisfaction"])
                elif "inaccurate" in issue.lower():
                    issue_based_suggestions.extend(self.improvement_suggestions["accuracy"])
                elif "slow" in issue.lower():
                    issue_based_suggestions.extend(self.improvement_suggestions["response_time"])
            
            # 점수 기반 제안
            score_based_suggestions = quality_assessment.get("suggestions", [])
            
            # 모든 제안 통합 및 중복 제거
            all_suggestions = issue_based_suggestions + score_based_suggestions
            unique_suggestions = list(dict.fromkeys(all_suggestions))  # 순서 유지하며 중복 제거
            
            return unique_suggestions[:10]  # 최대 10개 제안
            
        except Exception as e:
            self.logger.error(f"Error suggesting improvements: {e}")
            return [f"개선 제안 생성 오류: {str(e)}"]
    
    def analyze_quality_trends(self, session_ids: List[str]) -> Dict[str, Any]:
        """
        품질 트렌드 분석
        
        Args:
            session_ids: 분석할 세션 ID 목록
            
        Returns:
            Dict[str, Any]: 품질 트렌드 분석 결과
        """
        try:
            # 각 세션의 품질 메트릭 조회
            quality_data = []
            for session_id in session_ids:
                metrics = self._get_quality_metrics(session_id)
                if metrics:
                    quality_data.extend(metrics)
            
            if not quality_data:
                return {"error": "분석할 품질 데이터가 없습니다"}
            
            # 시간대별 그룹화
            trends = self._group_metrics_by_period(quality_data)
            
            # 트렌드 분석
            trend_analysis = {
                "periods": [],
                "overall_trend": "stable",
                "key_insights": [],
                "recommendations": []
            }
            
            for period, metrics_list in trends.items():
                if not metrics_list:
                    continue
                
                avg_completeness = statistics.mean([m.completeness_score for m in metrics_list])
                avg_satisfaction = statistics.mean([m.satisfaction_score for m in metrics_list])
                avg_accuracy = statistics.mean([m.accuracy_score for m in metrics_list])
                avg_response_time = statistics.mean([m.response_time for m in metrics_list])
                
                # 개선 제안 생성
                suggestions = []
                if avg_completeness < 0.7:
                    suggestions.extend(self.improvement_suggestions["completeness"])
                if avg_satisfaction < 0.7:
                    suggestions.extend(self.improvement_suggestions["satisfaction"])
                if avg_accuracy < 0.7:
                    suggestions.extend(self.improvement_suggestions["accuracy"])
                
                trend = QualityTrend(
                    period=period,
                    avg_completeness=avg_completeness,
                    avg_satisfaction=avg_satisfaction,
                    avg_accuracy=avg_accuracy,
                    avg_response_time=avg_response_time,
                    total_sessions=len(metrics_list),
                    improvement_suggestions=list(set(suggestions))
                )
                
                trend_analysis["periods"].append(trend)
            
            # 전체 트렌드 분석
            trend_analysis.update(self._analyze_overall_trend(trend_analysis["periods"]))
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing quality trends: {e}")
            return {"error": str(e)}
    
    def calculate_turn_quality(self, turn: ConversationTurn, context: ConversationContext) -> Dict[str, float]:
        """
        개별 턴의 품질 계산
        
        Args:
            turn: 대화 턴
            context: 대화 맥락
            
        Returns:
            Dict[str, float]: 품질 점수들
        """
        try:
            # 완결성 점수
            completeness_score = self._calculate_completeness_score(turn)
            
            # 만족도 점수
            satisfaction_score = self._calculate_satisfaction_score(turn)
            
            # 정확성 점수
            accuracy_score = self._calculate_accuracy_score(turn)
            
            return {
                "completeness": completeness_score,
                "satisfaction": satisfaction_score,
                "accuracy": accuracy_score,
                "turn_id": turn.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating turn quality: {e}")
            return {
                "completeness": 0.0,
                "satisfaction": 0.0,
                "accuracy": 0.0,
                "error": str(e)
            }
    
    def get_quality_dashboard_data(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        품질 대시보드 데이터 생성
        
        Args:
            user_id: 사용자 ID (선택사항)
            
        Returns:
            Dict[str, Any]: 대시보드 데이터
        """
        try:
            # 최근 품질 메트릭 조회
            recent_metrics = self._get_recent_quality_metrics(user_id, days=7)
            
            if not recent_metrics:
                return {"message": "최근 품질 데이터가 없습니다"}
            
            # 기본 통계
            avg_completeness = statistics.mean([m.completeness_score for m in recent_metrics])
            avg_satisfaction = statistics.mean([m.satisfaction_score for m in recent_metrics])
            avg_accuracy = statistics.mean([m.accuracy_score for m in recent_metrics])
            avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
            
            # 문제점 통계
            all_issues = []
            for metric in recent_metrics:
                all_issues.extend(metric.issues_detected)
            
            issue_counts = defaultdict(int)
            for issue in all_issues:
                issue_counts[issue] += 1
            
            # 품질 등급 계산
            overall_score = (avg_completeness * 0.4 + avg_satisfaction * 0.3 + avg_accuracy * 0.3)
            quality_grade = self._calculate_quality_grade(overall_score)
            
            return {
                "overall_score": overall_score,
                "quality_grade": quality_grade,
                "completeness_score": avg_completeness,
                "satisfaction_score": avg_satisfaction,
                "accuracy_score": avg_accuracy,
                "avg_response_time": avg_response_time,
                "total_sessions": len(recent_metrics),
                "issue_statistics": dict(issue_counts),
                "top_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality dashboard data: {e}")
            return {"error": str(e)}
    
    def _calculate_completeness_score(self, turn: ConversationTurn) -> float:
        """완결성 점수 계산"""
        response_text = turn.bot_response.lower()
        score = 0.5  # 기본 점수
        
        # 긍정적 키워드
        for keyword in self.quality_criteria["completeness"]["keywords"]:
            if keyword in response_text:
                score += 0.1
        
        # 부정적 키워드
        for keyword in self.quality_criteria["completeness"]["negative_keywords"]:
            if keyword in response_text:
                score -= 0.2
        
        # 길이 기반 조정
        response_length = len(turn.bot_response)
        min_length = self.quality_criteria["completeness"]["min_length"]
        max_length = self.quality_criteria["completeness"]["max_length"]
        
        if response_length < min_length:
            score -= 0.3
        elif response_length > max_length:
            score -= 0.1
        else:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_satisfaction_score(self, turn: ConversationTurn) -> float:
        """만족도 점수 계산"""
        # 사용자 질문에서 만족도 신호 추출
        user_text = turn.user_query.lower()
        score = 0.5  # 기본 점수
        
        # 긍정적 키워드
        for keyword in self.quality_criteria["satisfaction"]["positive_keywords"]:
            if keyword in user_text:
                score += 0.2
        
        # 부정적 키워드
        for keyword in self.quality_criteria["satisfaction"]["negative_keywords"]:
            if keyword in user_text:
                score -= 0.3
        
        # 중립적 키워드
        for keyword in self.quality_criteria["satisfaction"]["neutral_keywords"]:
            if keyword in user_text:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_accuracy_score(self, turn: ConversationTurn) -> float:
        """정확성 점수 계산"""
        response_text = turn.bot_response.lower()
        score = 0.5  # 기본 점수
        
        # 법률 키워드 포함 여부
        legal_keywords_found = 0
        for keyword in self.quality_criteria["accuracy"]["legal_keywords"]:
            if keyword in response_text:
                legal_keywords_found += 1
        
        if legal_keywords_found > 0:
            score += min(0.3, legal_keywords_found * 0.1)
        
        # 법령 인용 패턴
        citations_found = 0
        for pattern in self.quality_criteria["accuracy"]["citation_patterns"]:
            if re.search(pattern, turn.bot_response):
                citations_found += 1
        
        if citations_found > 0:
            score += min(0.2, citations_found * 0.1)
        
        # 불확실성 키워드 감지
        uncertainty_count = 0
        for keyword in self.quality_criteria["accuracy"]["uncertainty_keywords"]:
            if keyword in response_text:
                uncertainty_count += 1
        
        if uncertainty_count > 2:  # 너무 많은 불확실성 표현
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def detect_turn_issues(self, turn: ConversationTurn) -> List[str]:
        """턴의 문제점 감지"""
        issues = []
        user_text = turn.user_query.lower()
        
        for issue_type, patterns in self.issue_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_text):
                    issues.append(issue_type)
                    break
        
        return issues
    
    def _detect_conversation_level_issues(self, context: ConversationContext) -> List[str]:
        """대화 수준의 문제점 감지"""
        issues = []
        
        # 대화 길이 문제
        if len(context.turns) > 20:
            issues.append("conversation_too_long")
        elif len(context.turns) < 2:
            issues.append("conversation_too_short")
        
        # 주제 일관성 문제
        if len(context.topic_stack) > 5:
            issues.append("topic_drift")
        
        # 엔티티 일관성 문제
        total_entities = sum(len(entities) for entities in context.entities.values())
        if total_entities > 20:
            issues.append("too_many_entities")
        
        return issues
    
    def generate_improvement_suggestions(self, completeness: float, satisfaction: float, 
                                       accuracy: float, issues: List[str]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        # 점수 기반 제안
        if completeness < 0.7:
            suggestions.extend(self.improvement_suggestions["completeness"])
        
        if satisfaction < 0.7:
            suggestions.extend(self.improvement_suggestions["satisfaction"])
        
        if accuracy < 0.7:
            suggestions.extend(self.improvement_suggestions["accuracy"])
        
        # 문제점 기반 제안
        for issue in issues:
            if issue in self.improvement_suggestions:
                suggestions.extend(self.improvement_suggestions[issue])
        
        return list(set(suggestions))  # 중복 제거
    
    def _store_quality_metrics(self, session_id: str, metrics: QualityMetrics):
        """품질 메트릭 저장"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                INSERT INTO quality_metrics 
                (session_id, turn_id, completeness_score, satisfaction_score, 
                 accuracy_score, response_time, issues_detected, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    0,  # 전체 세션 메트릭
                    metrics.completeness_score,
                    metrics.satisfaction_score,
                    metrics.accuracy_score,
                    metrics.response_time,
                    json.dumps(metrics.issues_detected),
                    metrics.timestamp.isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing quality metrics: {e}")
    
    def _get_quality_metrics(self, session_id: str) -> List[QualityMetrics]:
        """세션의 품질 메트릭 조회"""
        try:
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                SELECT * FROM quality_metrics 
                WHERE session_id = ?
                ORDER BY timestamp DESC
                """, (session_id,))
                
                metrics = []
                for row in cursor.fetchall():
                    metric = QualityMetrics(
                        completeness_score=row["completeness_score"],
                        satisfaction_score=row["satisfaction_score"],
                        accuracy_score=row["accuracy_score"],
                        response_time=row["response_time"],
                        issues_detected=json.loads(row["issues_detected"]) if row["issues_detected"] else [],
                        timestamp=datetime.fromisoformat(row["timestamp"])
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error getting quality metrics: {e}")
            return []
    
    def _get_recent_quality_metrics(self, user_id: Optional[str], days: int) -> List[QualityMetrics]:
        """최근 품질 메트릭 조회"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.conversation_store.get_connection() as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute("""
                    SELECT qm.* FROM quality_metrics qm
                    JOIN conversation_sessions cs ON qm.session_id = cs.session_id
                    WHERE cs.user_id = ? AND qm.timestamp >= ?
                    ORDER BY qm.timestamp DESC
                    """, (user_id, cutoff_date.isoformat()))
                else:
                    cursor.execute("""
                    SELECT * FROM quality_metrics 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    """, (cutoff_date.isoformat(),))
                
                metrics = []
                for row in cursor.fetchall():
                    metric = QualityMetrics(
                        completeness_score=row["completeness_score"],
                        satisfaction_score=row["satisfaction_score"],
                        accuracy_score=row["accuracy_score"],
                        response_time=row["response_time"],
                        issues_detected=json.loads(row["issues_detected"]) if row["issues_detected"] else [],
                        timestamp=datetime.fromisoformat(row["timestamp"])
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error getting recent quality metrics: {e}")
            return []
    
    def _group_metrics_by_period(self, metrics: List[QualityMetrics]) -> Dict[str, List[QualityMetrics]]:
        """메트릭을 기간별로 그룹화"""
        groups = defaultdict(list)
        
        for metric in metrics:
            # 일별 그룹화
            date_key = metric.timestamp.strftime("%Y-%m-%d")
            groups[date_key].append(metric)
        
        return dict(groups)
    
    def _analyze_overall_trend(self, periods: List[QualityTrend]) -> Dict[str, Any]:
        """전체 트렌드 분석"""
        if len(periods) < 2:
            return {
                "overall_trend": "insufficient_data",
                "key_insights": ["데이터가 부족하여 트렌드 분석이 어렵습니다"],
                "recommendations": ["더 많은 데이터를 수집한 후 다시 분석하세요"]
            }
        
        # 최근 3일과 이전 3일 비교
        recent_periods = periods[:3]
        older_periods = periods[3:6] if len(periods) > 3 else []
        
        if not older_periods:
            return {
                "overall_trend": "improving",
                "key_insights": ["최근 품질이 향상되고 있습니다"],
                "recommendations": ["현재 방향을 유지하세요"]
            }
        
        # 평균 점수 계산
        recent_avg = statistics.mean([p.avg_completeness + p.avg_satisfaction + p.avg_accuracy for p in recent_periods])
        older_avg = statistics.mean([p.avg_completeness + p.avg_satisfaction + p.avg_accuracy for p in older_periods])
        
        # 트렌드 결정
        if recent_avg > older_avg + 0.1:
            trend = "improving"
            insights = ["품질이 지속적으로 향상되고 있습니다"]
            recommendations = ["현재 접근 방식을 유지하세요"]
        elif recent_avg < older_avg - 0.1:
            trend = "declining"
            insights = ["품질이 하락하고 있습니다"]
            recommendations = ["품질 개선이 필요합니다"]
        else:
            trend = "stable"
            insights = ["품질이 안정적으로 유지되고 있습니다"]
            recommendations = ["지속적인 모니터링이 필요합니다"]
        
        return {
            "overall_trend": trend,
            "key_insights": insights,
            "recommendations": recommendations
        }
    
    def _calculate_quality_grade(self, score: float) -> str:
        """품질 등급 계산"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"


# 테스트 함수
def test_conversation_quality_monitor():
    """대화 품질 모니터 테스트"""
    monitor = ConversationQualityMonitor()
    
    print("=== 대화 품질 모니터 테스트 ===")
    
    # 테스트 대화 컨텍스트 생성
    from core.conversation.conversation_manager import ConversationContext, ConversationTurn
    
    context = ConversationContext(
        session_id="test_session_quality",
        turns=[
            ConversationTurn(
                user_query="민법 제750조에 대해 자세히 설명해주세요",
                bot_response="민법 제750조는 불법행위로 인한 손해배상 책임을 규정하는 중요한 조문입니다. 이 조문에 따르면...",
                timestamp=datetime.now(),
                question_type="law_inquiry"
            ),
            ConversationTurn(
                user_query="감사합니다. 정말 도움이 되었어요!",
                bot_response="천만에요. 추가로 궁금한 것이 있으시면 언제든지 문의해주세요.",
                timestamp=datetime.now(),
                question_type="thanks"
            )
        ],
        entities={"laws": {"민법"}, "articles": {"제750조"}, "precedents": set(), "legal_terms": {"손해배상"}},
        topic_stack=["민법", "손해배상"],
        created_at=datetime.now(),
        last_updated=datetime.now()
    )
    
    # 1. 대화 품질 평가
    print("\n1. 대화 품질 평가 테스트")
    quality_assessment = monitor.assess_conversation_quality(context)
    print(f"전체 점수: {quality_assessment['overall_score']:.2f}")
    print(f"완결성: {quality_assessment['completeness_score']:.2f}")
    print(f"만족도: {quality_assessment['satisfaction_score']:.2f}")
    print(f"정확성: {quality_assessment['accuracy_score']:.2f}")
    print(f"감지된 문제점: {quality_assessment['issues']}")
    print(f"개선 제안: {quality_assessment['suggestions'][:3]}")
    
    # 2. 문제점 감지
    print("\n2. 문제점 감지 테스트")
    issues = monitor.detect_conversation_issues(context)
    print(f"감지된 문제점: {issues}")
    
    # 3. 개선 제안
    print("\n3. 개선 제안 테스트")
    suggestions = monitor.suggest_improvements(context)
    print(f"개선 제안: {suggestions[:5]}")
    
    # 4. 개별 턴 품질 계산
    print("\n4. 개별 턴 품질 계산 테스트")
    for i, turn in enumerate(context.turns):
        turn_quality = monitor.calculate_turn_quality(turn, context)
        print(f"턴 {i+1}: 완결성={turn_quality['completeness']:.2f}, "
              f"만족도={turn_quality['satisfaction']:.2f}, "
              f"정확성={turn_quality['accuracy']:.2f}")
    
    # 5. 품질 대시보드 데이터
    print("\n5. 품질 대시보드 데이터 테스트")
    dashboard_data = monitor.get_quality_dashboard_data()
    print(f"대시보드 데이터: {dashboard_data}")
    
    # 6. 품질 트렌드 분석
    print("\n6. 품질 트렌드 분석 테스트")
    trend_analysis = monitor.analyze_quality_trends([context.session_id])
    print(f"트렌드 분석: {trend_analysis}")
    
    print("\n테스트 완료")


if __name__ == "__main__":
    test_conversation_quality_monitor()
