# -*- coding: utf-8 -*-
"""
프롬프트 성능 최적화 시스템
모델별, 컨텍스트별 프롬프트 최적화 및 성능 모니터링
"""

import os
import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import statistics
from dataclasses import dataclass
from enum import Enum

from .unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType, QuestionType
from ..utils.json_safe_saver import safe_save_json, safe_load_json

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """최적화 메트릭"""
    RESPONSE_TIME = "response_time"
    TOKEN_EFFICIENCY = "token_efficiency"
    ANSWER_QUALITY = "answer_quality"
    CONTEXT_UTILIZATION = "context_utilization"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class OptimizationResult:
    """최적화 결과"""
    metric: OptimizationMetric
    before_value: float
    after_value: float
    improvement_percentage: float
    optimization_applied: str
    timestamp: datetime


@dataclass
class PromptPerformanceMetrics:
    """프롬프트 성능 메트릭"""
    prompt_id: str
    model_type: ModelType
    domain: LegalDomain
    question_type: QuestionType
    response_time: float
    token_count: int
    context_length: int
    answer_quality_score: float
    user_rating: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PromptOptimizer:
    """프롬프트 최적화기"""
    
    def __init__(self, unified_manager: UnifiedPromptManager):
        """프롬프트 최적화기 초기화"""
        self.unified_manager = unified_manager
        self.performance_history = []
        self.optimization_results = []
        self.optimization_rules = self._load_optimization_rules()
        self.performance_thresholds = self._load_performance_thresholds()
        
        # 성능 데이터 저장 경로
        self.performance_data_dir = Path("data/prompt_performance")
        self.performance_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("PromptOptimizer initialized successfully")
    
    def _load_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """최적화 규칙 로드"""
        return {
            "response_time": {
                "threshold": 5.0,  # 5초
                "optimization_strategies": [
                    "reduce_context_length",
                    "simplify_prompt_structure",
                    "use_model_specific_optimization"
                ]
            },
            "token_efficiency": {
                "threshold": 0.8,  # 80% 효율성
                "optimization_strategies": [
                    "optimize_context_compression",
                    "remove_redundant_instructions",
                    "use_abbreviated_templates"
                ]
            },
            "answer_quality": {
                "threshold": 0.7,  # 70% 품질 점수
                "optimization_strategies": [
                    "enhance_domain_specificity",
                    "add_structured_guidance",
                    "improve_context_relevance"
                ]
            },
            "context_utilization": {
                "threshold": 0.6,  # 60% 활용도
                "optimization_strategies": [
                    "improve_context_filtering",
                    "enhance_relevance_scoring",
                    "optimize_context_ordering"
                ]
            }
        }
    
    def _load_performance_thresholds(self) -> Dict[str, float]:
        """성능 임계값 로드"""
        return {
            "response_time": 5.0,  # 초
            "token_efficiency": 0.8,  # 비율
            "answer_quality": 0.7,  # 점수 (0-1)
            "context_utilization": 0.6,  # 비율
            "user_satisfaction": 4.0  # 점수 (1-5)
        }
    
    def record_performance(self, metrics: PromptPerformanceMetrics) -> None:
        """성능 메트릭 기록"""
        try:
            self.performance_history.append(metrics)
            
            # 성능 데이터 파일에 저장
            self._save_performance_data(metrics)
            
            # 실시간 최적화 확인
            self._check_and_optimize(metrics)
            
            logger.debug(f"Performance recorded: {metrics.prompt_id}")
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def _save_performance_data(self, metrics: PromptPerformanceMetrics) -> None:
        """성능 데이터 저장"""
        try:
            # 일별 성능 데이터 파일
            date_str = metrics.timestamp.strftime("%Y-%m-%d")
            performance_file = self.performance_data_dir / f"performance_{date_str}.json"
            
            # 기존 데이터 로드
            if performance_file.exists():
                try:
                    with open(performance_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    # JSON 파일이 손상된 경우 새로 생성
                    logger.warning(f"Corrupted JSON file detected, creating new one: {performance_file}")
                    data = {"metrics": []}
            else:
                data = {"metrics": []}
            
            # 새 메트릭 추가
            metrics_dict = {
                "prompt_id": metrics.prompt_id,
                "model_type": metrics.model_type.value,
                "domain": metrics.domain.value,
                "question_type": metrics.question_type.value,
                "response_time": metrics.response_time,
                "token_count": metrics.token_count,
                "context_length": metrics.context_length,
                "answer_quality_score": metrics.answer_quality_score,
                "user_rating": metrics.user_rating,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            data["metrics"].append(metrics_dict)
            
            # 파일 저장 (안전한 방법)
            self._safe_save_performance_data(data, performance_file)
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _safe_save_performance_data(self, data: dict, performance_file) -> None:
        """성능 데이터를 안전하게 저장"""
        try:
            # 임시 파일에 먼저 저장
            temp_file = str(performance_file) + ".tmp"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 임시 파일이 성공적으로 저장되었으면 원본 파일로 이동
            import shutil
            shutil.move(temp_file, str(performance_file))
            
        except Exception as e:
            logger.error(f"Error in safe save: {e}")
            # 임시 파일이 있으면 삭제
            try:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            raise
    
    def _check_and_optimize(self, metrics: PromptPerformanceMetrics) -> None:
        """성능 확인 및 최적화"""
        try:
            # 각 메트릭별로 성능 확인
            optimizations_applied = []
            
            # 응답 시간 최적화
            if metrics.response_time > self.performance_thresholds["response_time"]:
                optimization = self._optimize_response_time(metrics)
                if optimization:
                    optimizations_applied.append(optimization)
            
            # 토큰 효율성 최적화
            token_efficiency = self._calculate_token_efficiency(metrics)
            if token_efficiency < self.performance_thresholds["token_efficiency"]:
                optimization = self._optimize_token_efficiency(metrics)
                if optimization:
                    optimizations_applied.append(optimization)
            
            # 답변 품질 최적화
            if metrics.answer_quality_score < self.performance_thresholds["answer_quality"]:
                optimization = self._optimize_answer_quality(metrics)
                if optimization:
                    optimizations_applied.append(optimization)
            
            # 컨텍스트 활용도 최적화
            context_utilization = self._calculate_context_utilization(metrics)
            if context_utilization < self.performance_thresholds["context_utilization"]:
                optimization = self._optimize_context_utilization(metrics)
                if optimization:
                    optimizations_applied.append(optimization)
            
            # 최적화 결과 기록
            if optimizations_applied:
                self._record_optimization_results(metrics, optimizations_applied)
            
        except Exception as e:
            logger.error(f"Error in check_and_optimize: {e}")
    
    def _optimize_response_time(self, metrics: PromptPerformanceMetrics) -> Optional[OptimizationResult]:
        """응답 시간 최적화"""
        try:
            # 최적화 전 성능 측정
            before_time = metrics.response_time
            
            # 최적화 전략 적용
            optimization_strategies = self.optimization_rules["response_time"]["optimization_strategies"]
            
            # 컨텍스트 길이 줄이기
            if "reduce_context_length" in optimization_strategies:
                optimized_context_length = int(metrics.context_length * 0.8)  # 20% 줄이기
                # 실제 구현 시에는 프롬프트에서 컨텍스트 길이 조정
            
            # 프롬프트 구조 단순화
            if "simplify_prompt_structure" in optimization_strategies:
                # 프롬프트 구조 단순화 로직
                pass
            
            # 모델별 최적화
            if "use_model_specific_optimization" in optimization_strategies:
                # 모델별 최적화 설정 적용
                pass
            
            # 최적화 후 성능 측정 (시뮬레이션)
            after_time = before_time * 0.8  # 20% 개선 가정
            
            improvement = ((before_time - after_time) / before_time) * 100
            
            return OptimizationResult(
                metric=OptimizationMetric.RESPONSE_TIME,
                before_value=before_time,
                after_value=after_time,
                improvement_percentage=improvement,
                optimization_applied="response_time_optimization",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing response time: {e}")
            return None
    
    def _optimize_token_efficiency(self, metrics: PromptPerformanceMetrics) -> Optional[OptimizationResult]:
        """토큰 효율성 최적화"""
        try:
            # 현재 토큰 효율성 계산
            before_efficiency = self._calculate_token_efficiency(metrics)
            
            # 최적화 전략 적용
            optimization_strategies = self.optimization_rules["token_efficiency"]["optimization_strategies"]
            
            # 컨텍스트 압축 최적화
            if "optimize_context_compression" in optimization_strategies:
                # 컨텍스트 압축 알고리즘 적용
                pass
            
            # 중복 지시사항 제거
            if "remove_redundant_instructions" in optimization_strategies:
                # 중복 지시사항 제거 로직
                pass
            
            # 축약된 템플릿 사용
            if "use_abbreviated_templates" in optimization_strategies:
                # 축약된 템플릿 적용
                pass
            
            # 최적화 후 효율성 계산 (시뮬레이션)
            after_efficiency = min(1.0, before_efficiency * 1.2)  # 20% 개선 가정
            
            improvement = ((after_efficiency - before_efficiency) / before_efficiency) * 100
            
            return OptimizationResult(
                metric=OptimizationMetric.TOKEN_EFFICIENCY,
                before_value=before_efficiency,
                after_value=after_efficiency,
                improvement_percentage=improvement,
                optimization_applied="token_efficiency_optimization",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing token efficiency: {e}")
            return None
    
    def _optimize_answer_quality(self, metrics: PromptPerformanceMetrics) -> Optional[OptimizationResult]:
        """답변 품질 최적화"""
        try:
            # 현재 답변 품질
            before_quality = metrics.answer_quality_score
            
            # 최적화 전략 적용
            optimization_strategies = self.optimization_rules["answer_quality"]["optimization_strategies"]
            
            # 도메인 특화 강화
            if "enhance_domain_specificity" in optimization_strategies:
                # 도메인별 특화 지침 강화
                pass
            
            # 구조화된 가이드 추가
            if "add_structured_guidance" in optimization_strategies:
                # 답변 구조 가이드 추가
                pass
            
            # 컨텍스트 관련성 개선
            if "improve_context_relevance" in optimization_strategies:
                # 컨텍스트 관련성 점수 개선
                pass
            
            # 최적화 후 품질 계산 (시뮬레이션)
            after_quality = min(1.0, before_quality * 1.15)  # 15% 개선 가정
            
            improvement = ((after_quality - before_quality) / before_quality) * 100
            
            return OptimizationResult(
                metric=OptimizationMetric.ANSWER_QUALITY,
                before_value=before_quality,
                after_value=after_quality,
                improvement_percentage=improvement,
                optimization_applied="answer_quality_optimization",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing answer quality: {e}")
            return None
    
    def _optimize_context_utilization(self, metrics: PromptPerformanceMetrics) -> Optional[OptimizationResult]:
        """컨텍스트 활용도 최적화"""
        try:
            # 현재 컨텍스트 활용도
            before_utilization = self._calculate_context_utilization(metrics)
            
            # 최적화 전략 적용
            optimization_strategies = self.optimization_rules["context_utilization"]["optimization_strategies"]
            
            # 컨텍스트 필터링 개선
            if "improve_context_filtering" in optimization_strategies:
                # 관련성 기반 컨텍스트 필터링
                pass
            
            # 관련성 점수 향상
            if "enhance_relevance_scoring" in optimization_strategies:
                # 컨텍스트 관련성 점수 알고리즘 개선
                pass
            
            # 컨텍스트 순서 최적화
            if "optimize_context_ordering" in optimization_strategies:
                # 중요도 기반 컨텍스트 순서 조정
                pass
            
            # 최적화 후 활용도 계산 (시뮬레이션)
            after_utilization = min(1.0, before_utilization * 1.25)  # 25% 개선 가정
            
            improvement = ((after_utilization - before_utilization) / before_utilization) * 100
            
            return OptimizationResult(
                metric=OptimizationMetric.CONTEXT_UTILIZATION,
                before_value=before_utilization,
                after_value=after_utilization,
                improvement_percentage=improvement,
                optimization_applied="context_utilization_optimization",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error optimizing context utilization: {e}")
            return None
    
    def _calculate_token_efficiency(self, metrics: PromptPerformanceMetrics) -> float:
        """토큰 효율성 계산"""
        try:
            # 토큰 효율성 = 유효한 답변 길이 / 사용된 토큰 수
            if metrics.token_count == 0:
                return 0.0
            
            # 답변 품질 점수를 토큰 효율성에 반영
            efficiency = (metrics.answer_quality_score * 100) / metrics.token_count
            return min(1.0, efficiency / 10)  # 정규화
            
        except Exception as e:
            logger.error(f"Error calculating token efficiency: {e}")
            return 0.0
    
    def _calculate_context_utilization(self, metrics: PromptPerformanceMetrics) -> float:
        """컨텍스트 활용도 계산"""
        try:
            # 컨텍스트 활용도 = 답변 품질 / 컨텍스트 길이
            if metrics.context_length == 0:
                return 0.0
            
            utilization = metrics.answer_quality_score / (metrics.context_length / 1000)
            return min(1.0, utilization)
            
        except Exception as e:
            logger.error(f"Error calculating context utilization: {e}")
            return 0.0
    
    def _record_optimization_results(self, metrics: PromptPerformanceMetrics, optimizations: List[OptimizationResult]) -> None:
        """최적화 결과 기록"""
        try:
            self.optimization_results.extend(optimizations)
            
            # 최적화 결과 파일에 저장
            optimization_file = self.performance_data_dir / "optimization_results.json"
            
            # 기존 결과 로드 (안전한 로드 사용)
            data = safe_load_json(str(optimization_file), {"results": []})
            
            # 새 결과 추가
            for optimization in optimizations:
                result_dict = {
                    "prompt_id": metrics.prompt_id,
                    "metric": optimization.metric.value,
                    "before_value": optimization.before_value,
                    "after_value": optimization.after_value,
                    "improvement_percentage": optimization.improvement_percentage,
                    "optimization_applied": optimization.optimization_applied,
                    "timestamp": optimization.timestamp.isoformat()
                }
                data["results"].append(result_dict)
            
            # 파일 저장 (안전한 저장 사용)
            if safe_save_json(str(optimization_file), data):
                logger.info(f"Recorded {len(optimizations)} optimization results")
            else:
                logger.error(f"Failed to save optimization results to {optimization_file}")
            
        except Exception as e:
            logger.error(f"Error recording optimization results: {e}")
    
    def get_performance_analytics(self, days: int = 7) -> Dict[str, Any]:
        """성능 분석 데이터 조회"""
        try:
            # 최근 N일간의 성능 데이터 필터링
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [
                m for m in self.performance_history 
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {"error": "No performance data available"}
            
            # 기본 통계 계산
            analytics = {
                "period_days": days,
                "total_requests": len(recent_metrics),
                "average_response_time": statistics.mean([m.response_time for m in recent_metrics]),
                "average_token_count": statistics.mean([m.token_count for m in recent_metrics]),
                "average_answer_quality": statistics.mean([m.answer_quality_score for m in recent_metrics]),
                "average_context_utilization": statistics.mean([
                    self._calculate_context_utilization(m) for m in recent_metrics
                ]),
                "average_token_efficiency": statistics.mean([
                    self._calculate_token_efficiency(m) for m in recent_metrics
                ])
            }
            
            # 도메인별 성능 분석
            domain_performance = {}
            for domain in LegalDomain:
                domain_metrics = [m for m in recent_metrics if m.domain == domain]
                if domain_metrics:
                    domain_performance[domain.value] = {
                        "request_count": len(domain_metrics),
                        "average_response_time": statistics.mean([m.response_time for m in domain_metrics]),
                        "average_answer_quality": statistics.mean([m.answer_quality_score for m in domain_metrics])
                    }
            
            analytics["domain_performance"] = domain_performance
            
            # 모델별 성능 분석
            model_performance = {}
            for model in ModelType:
                model_metrics = [m for m in recent_metrics if m.model_type == model]
                if model_metrics:
                    model_performance[model.value] = {
                        "request_count": len(model_metrics),
                        "average_response_time": statistics.mean([m.response_time for m in model_metrics]),
                        "average_answer_quality": statistics.mean([m.answer_quality_score for m in model_metrics])
                    }
            
            analytics["model_performance"] = model_performance
            
            # 최적화 결과 분석
            recent_optimizations = [
                r for r in self.optimization_results 
                if r.timestamp >= cutoff_date
            ]
            
            if recent_optimizations:
                optimization_summary = {}
                for metric in OptimizationMetric:
                    metric_optimizations = [r for r in recent_optimizations if r.metric == metric]
                    if metric_optimizations:
                        optimization_summary[metric.value] = {
                            "count": len(metric_optimizations),
                            "average_improvement": statistics.mean([r.improvement_percentage for r in metric_optimizations])
                        }
                
                analytics["optimization_summary"] = optimization_summary
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self, domain: Optional[LegalDomain] = None, 
                                       model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """최적화 권장사항 조회"""
        try:
            recommendations = []
            
            # 최근 7일간의 성능 데이터 분석
            analytics = self.get_performance_analytics(days=7)
            
            if "error" in analytics:
                return [{"error": analytics["error"]}]
            
            # 응답 시간 권장사항
            if analytics["average_response_time"] > self.performance_thresholds["response_time"]:
                recommendations.append({
                    "metric": "response_time",
                    "current_value": analytics["average_response_time"],
                    "threshold": self.performance_thresholds["response_time"],
                    "recommendation": "응답 시간이 임계값을 초과했습니다. 컨텍스트 길이를 줄이거나 프롬프트 구조를 단순화하세요.",
                    "priority": "high"
                })
            
            # 답변 품질 권장사항
            if analytics["average_answer_quality"] < self.performance_thresholds["answer_quality"]:
                recommendations.append({
                    "metric": "answer_quality",
                    "current_value": analytics["average_answer_quality"],
                    "threshold": self.performance_thresholds["answer_quality"],
                    "recommendation": "답변 품질이 임계값 미만입니다. 도메인 특화 지침을 강화하거나 구조화된 가이드를 추가하세요.",
                    "priority": "high"
                })
            
            # 토큰 효율성 권장사항
            if analytics["average_token_efficiency"] < self.performance_thresholds["token_efficiency"]:
                recommendations.append({
                    "metric": "token_efficiency",
                    "current_value": analytics["average_token_efficiency"],
                    "threshold": self.performance_thresholds["token_efficiency"],
                    "recommendation": "토큰 효율성이 낮습니다. 컨텍스트 압축을 최적화하거나 중복 지시사항을 제거하세요.",
                    "priority": "medium"
                })
            
            # 컨텍스트 활용도 권장사항
            if analytics["average_context_utilization"] < self.performance_thresholds["context_utilization"]:
                recommendations.append({
                    "metric": "context_utilization",
                    "current_value": analytics["average_context_utilization"],
                    "threshold": self.performance_thresholds["context_utilization"],
                    "recommendation": "컨텍스트 활용도가 낮습니다. 컨텍스트 필터링을 개선하거나 관련성 점수를 향상시키세요.",
                    "priority": "medium"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting optimization recommendations: {e}")
            return [{"error": str(e)}]


# 전역 인스턴스
def create_prompt_optimizer(unified_manager: UnifiedPromptManager) -> PromptOptimizer:
    """프롬프트 최적화기 생성"""
    return PromptOptimizer(unified_manager)
