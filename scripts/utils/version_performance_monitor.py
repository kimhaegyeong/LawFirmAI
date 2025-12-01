"""
버전별 성능 모니터링 시스템

FAISS 버전별 검색 성능을 추적하고 비교합니다.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class VersionPerformanceMonitor:
    """버전별 성능 추적 클래스"""
    
    def __init__(self, log_path: str = "data/performance_logs"):
        """
        초기화
        
        Args:
            log_path: 성능 로그 저장 경로
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """메트릭 로드"""
        for version_file in self.log_path.glob("*_metrics.json"):
            version_name = version_file.stem.replace("_metrics", "")
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    self.metrics[version_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics for {version_name}: {e}")
    
    def log_search(
        self,
        version: str,
        query_id: str,
        latency_ms: float,
        relevance_score: Optional[float] = None,
        user_feedback: Optional[str] = None
    ):
        """
        검색 성능 로깅
        
        Args:
            version: FAISS 버전 이름
            query_id: 쿼리 ID
            latency_ms: 검색 지연 시간 (밀리초)
            relevance_score: 관련성 점수 (0-1)
            user_feedback: 사용자 피드백 ("positive", "negative", None)
        """
        if version not in self.metrics:
            self.metrics[version] = {
                "total_queries": 0,
                "avg_latency": 0.0,
                "avg_relevance": 0.0,
                "feedback_positive": 0,
                "feedback_negative": 0,
                "queries": []
            }
        
        m = self.metrics[version]
        m["total_queries"] += 1
        
        n = m["total_queries"]
        m["avg_latency"] = (m["avg_latency"] * (n - 1) + latency_ms) / n
        
        if relevance_score is not None:
            if m["avg_relevance"] == 0.0:
                m["avg_relevance"] = relevance_score
            else:
                m["avg_relevance"] = (m["avg_relevance"] * (n - 1) + relevance_score) / n
        
        if user_feedback == "positive":
            m["feedback_positive"] += 1
        elif user_feedback == "negative":
            m["feedback_negative"] += 1
        
        query_log = {
            "query_id": query_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "relevance_score": relevance_score,
            "user_feedback": user_feedback
        }
        m["queries"].append(query_log)
        
        if len(m["queries"]) > 1000:
            m["queries"] = m["queries"][-1000:]
        
        self._save_log(version)
    
    def _save_log(self, version: str):
        """메트릭을 파일에 저장"""
        log_file = self.log_path / f"{version}_metrics.json"
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics[version], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metrics for {version}: {e}")
    
    def compare_performance(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        두 버전의 성능 비교
        
        Args:
            version1: 첫 번째 버전 이름
            version2: 두 번째 버전 이름
        
        Returns:
            Dict: 성능 비교 결과
        """
        if version1 not in self.metrics or version2 not in self.metrics:
            return {"error": "버전 메트릭이 없습니다"}
        
        m1 = self.metrics[version1]
        m2 = self.metrics[version2]
        
        latency_improvement = 0.0
        if m1["avg_latency"] > 0:
            latency_improvement = ((m1["avg_latency"] - m2["avg_latency"]) / m1["avg_latency"]) * 100
        
        relevance_improvement = 0.0
        if m1["avg_relevance"] > 0:
            relevance_improvement = ((m2["avg_relevance"] - m1["avg_relevance"]) / m1["avg_relevance"]) * 100
        
        feedback_total_1 = m1["feedback_positive"] + m1["feedback_negative"]
        feedback_total_2 = m2["feedback_positive"] + m2["feedback_negative"]
        
        feedback_score_v1 = 0.0
        if feedback_total_1 > 0:
            feedback_score_v1 = m1["feedback_positive"] / feedback_total_1
        
        feedback_score_v2 = 0.0
        if feedback_total_2 > 0:
            feedback_score_v2 = m2["feedback_positive"] / feedback_total_2
        
        return {
            "version1": version1,
            "version2": version2,
            "latency_improvement_percent": latency_improvement,
            "relevance_improvement_percent": relevance_improvement,
            "feedback_score_v1": feedback_score_v1,
            "feedback_score_v2": feedback_score_v2,
            "metrics_v1": {
                "avg_latency_ms": m1["avg_latency"],
                "avg_relevance": m1["avg_relevance"],
                "total_queries": m1["total_queries"],
                "feedback_positive": m1["feedback_positive"],
                "feedback_negative": m1["feedback_negative"]
            },
            "metrics_v2": {
                "avg_latency_ms": m2["avg_latency"],
                "avg_relevance": m2["avg_relevance"],
                "total_queries": m2["total_queries"],
                "feedback_positive": m2["feedback_positive"],
                "feedback_negative": m2["feedback_negative"]
            }
        }
    
    def get_version_metrics(self, version: str) -> Optional[Dict[str, Any]]:
        """
        특정 버전의 메트릭 조회
        
        Args:
            version: 버전 이름
        
        Returns:
            Optional[Dict]: 버전 메트릭
        """
        return self.metrics.get(version)
    
    def list_versions(self) -> List[str]:
        """
        모니터링 중인 버전 목록 조회
        
        Returns:
            List[str]: 버전 이름 리스트
        """
        return list(self.metrics.keys())
    
    def clear_metrics(self, version: Optional[str] = None):
        """
        메트릭 정리
        
        Args:
            version: 특정 버전만 정리 (None이면 전체)
        """
        if version:
            if version in self.metrics:
                del self.metrics[version]
                log_file = self.log_path / f"{version}_metrics.json"
                if log_file.exists():
                    log_file.unlink()
                logger.info(f"Cleared metrics for version {version}")
        else:
            self.metrics.clear()
            for log_file in self.log_path.glob("*_metrics.json"):
                log_file.unlink()
            logger.info("Cleared all metrics")

