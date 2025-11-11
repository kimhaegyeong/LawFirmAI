#!/usr/bin/env python3
"""
A/B 테스트 관리자
워크플로우 최적화 효과를 검증하기 위한 A/B 테스트 프레임워크
"""

import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ExperimentResult:
    """실험 결과 데이터 클래스"""
    session_id: str
    experiment: str
    variant: str
    metric: str
    value: float
    timestamp: float


class ABTestManager:
    """A/B 테스트 관리자"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 실험 설정
        self.experiments = {
            "cache_enabled": {
                "control": {"cache": False},
                "variant_a": {"cache": True, "cache_ttl": 1800},
                "variant_b": {"cache": True, "cache_ttl": 3600}
            },
            "parallel_processing": {
                "control": {"max_workers": 2},
                "variant_a": {"max_workers": 4},
                "variant_b": {"max_workers": 8}
            }
        }
        
        # 실험 결과 저장
        self.results: List[ExperimentResult] = []
        
        # 통계 계산용 캐시
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
    
    def assign_variant(self, session_id: str, experiment: str) -> str:
        """세션에 실험 변형 할당"""
        if experiment not in self.experiments:
            return "control"
        
        # 세션 ID 기반 해시로 일관된 변형 할당
        hash_value = int(hashlib.md5(f"{session_id}_{experiment}".encode()).hexdigest(), 16)
        variants = list(self.experiments[experiment].keys())
        variant = variants[hash_value % len(variants)]
        
        return variant
    
    def track_metric(self, session_id: str, experiment: str, variant: str, 
                     metric: str, value: float):
        """메트릭 추적"""
        result = ExperimentResult(
            session_id=session_id,
            experiment=experiment,
            variant=variant,
            metric=metric,
            value=value,
            timestamp=time.time()
        )
        self.results.append(result)
        
        # 통계 캐시 무효화
        cache_key = f"{experiment}_{variant}_{metric}"
        if cache_key in self._stats_cache:
            del self._stats_cache[cache_key]
    
    def get_results(self, experiment: str) -> Dict[str, Any]:
        """실험 결과 분석"""
        if experiment not in self.experiments:
            return {}
        
        experiment_results = [r for r in self.results if r.experiment == experiment]
        
        if not experiment_results:
            return {}
        
        # 변형별 메트릭별로 그룹화
        metrics_by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        for result in experiment_results:
            metrics_by_variant[result.variant][result.metric].append(result.value)
        
        # 통계 계산
        stats = {}
        for variant, metrics in metrics_by_variant.items():
            stats[variant] = {}
            for metric, values in metrics.items():
                if not values:
                    continue
                
                sorted_values = sorted(values)
                n = len(values)
                
                stats[variant][metric] = {
                    "mean": sum(values) / n,
                    "median": sorted_values[n // 2] if n > 0 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": n,
                    "std": self._calculate_std(values) if n > 1 else 0
                }
        
        return stats
    
    def _calculate_std(self, values: List[float]) -> float:
        """표준편차 계산"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def compare_variants(self, experiment: str, metric: str, 
                        variant1: str = "control", variant2: str = "variant_a") -> Dict[str, Any]:
        """두 변형 비교"""
        results = self.get_results(experiment)
        
        if variant1 not in results or variant2 not in results:
            return {}
        
        if metric not in results[variant1] or metric not in results[variant2]:
            return {}
        
        stats1 = results[variant1][metric]
        stats2 = results[variant2][metric]
        
        mean1 = stats1["mean"]
        mean2 = stats2["mean"]
        
        improvement = ((mean1 - mean2) / mean1 * 100) if mean1 > 0 else 0
        
        return {
            "variant1": {
                "name": variant1,
                "mean": mean1,
                "count": stats1["count"]
            },
            "variant2": {
                "name": variant2,
                "mean": mean2,
                "count": stats2["count"]
            },
            "improvement": improvement,
            "improvement_abs": mean1 - mean2
        }
    
    def get_experiment_config(self, experiment: str, variant: str) -> Dict[str, Any]:
        """실험 설정 가져오기"""
        if experiment not in self.experiments:
            return {}
        
        if variant not in self.experiments[experiment]:
            return {}
        
        return self.experiments[experiment][variant]
    
    def add_experiment(self, experiment: str, variants: Dict[str, Dict[str, Any]]):
        """새 실험 추가"""
        self.experiments[experiment] = variants
    
    def clear_results(self, experiment: Optional[str] = None):
        """실험 결과 초기화"""
        if experiment:
            self.results = [r for r in self.results if r.experiment != experiment]
        else:
            self.results = []
        
        self._stats_cache.clear()
    
    def export_results(self, experiment: str) -> List[Dict[str, Any]]:
        """실험 결과 내보내기"""
        experiment_results = [r for r in self.results if r.experiment == experiment]
        return [asdict(result) for result in experiment_results]
    
    def get_summary(self) -> Dict[str, Any]:
        """전체 실험 요약"""
        summary = {}
        
        for experiment in self.experiments.keys():
            results = self.get_results(experiment)
            if results:
                summary[experiment] = {
                    "variants": list(results.keys()),
                    "total_results": len([r for r in self.results if r.experiment == experiment]),
                    "metrics": set()
                }
                
                for variant, metrics in results.items():
                    summary[experiment]["metrics"].update(metrics.keys())
                
                summary[experiment]["metrics"] = list(summary[experiment]["metrics"])
        
        return summary

