import logging
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Global logger 사용
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)

class ExtractionEvaluator:
    """추출 성능 평가 시스템"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def calculate_precision(self, extracted_terms: List[str], ground_truth: List[str]) -> float:
        """정밀도 계산"""
        if not extracted_terms:
            return 0.0
        
        true_positives = len(set(extracted_terms) & set(ground_truth))
        return true_positives / len(extracted_terms)
    
    def calculate_recall(self, extracted_terms: List[str], ground_truth: List[str]) -> float:
        """재현율 계산"""
        if not ground_truth:
            return 0.0
        
        true_positives = len(set(extracted_terms) & set(ground_truth))
        return true_positives / len(ground_truth)
    
    def calculate_f1_score(self, extracted_terms: List[str], ground_truth: List[str]) -> float:
        """F1 점수 계산"""
        precision = self.calculate_precision(extracted_terms, ground_truth)
        recall = self.calculate_recall(extracted_terms, ground_truth)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_extraction_quality(self, extracted_terms: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """추출 품질 평가"""
        metrics = {
            "precision": self.calculate_precision(extracted_terms, ground_truth),
            "recall": self.calculate_recall(extracted_terms, ground_truth),
            "f1_score": self.calculate_f1_score(extracted_terms, ground_truth)
        }
        
        # 추가 메트릭
        metrics["extracted_count"] = len(extracted_terms)
        metrics["ground_truth_count"] = len(ground_truth)
        metrics["true_positives"] = len(set(extracted_terms) & set(ground_truth))
        metrics["false_positives"] = len(set(extracted_terms) - set(ground_truth))
        metrics["false_negatives"] = len(set(ground_truth) - set(extracted_terms))
        
        return metrics
    
    def evaluate_method_performance(self, method_results: Dict[str, List[str]], ground_truth: List[str]) -> Dict[str, Dict[str, float]]:
        """방법별 성능 평가"""
        method_metrics = {}
        
        for method, extracted_terms in method_results.items():
            metrics = self.evaluate_extraction_quality(extracted_terms, ground_truth)
            method_metrics[method] = metrics
        
        return method_metrics

class ClassificationMonitor:
    """분류 성능 모니터링"""
    
    def __init__(self, log_file: str = "logs/classification_performance.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.performance_history = self.load_performance_history()
        self.logger = get_logger(__name__)
    
    def load_performance_history(self) -> List[Dict[str, Any]]:
        """성능 이력 로드"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"성능 이력 로드 중 오류: {e}")
            return []
    
    def save_performance_history(self):
        """성능 이력 저장"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"성능 이력 저장 중 오류: {e}")
    
    def record_classification(self, query: str, predicted_domain: str, confidence: float, 
                            actual_domain: Optional[str] = None, processing_time: float = 0.0):
        """분류 결과 기록"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "predicted_domain": predicted_domain,
            "confidence": confidence,
            "actual_domain": actual_domain,
            "processing_time": processing_time,
            "is_correct": predicted_domain == actual_domain if actual_domain else None
        }
        
        self.performance_history.append(record)
        self.save_performance_history()
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        # 최근 N일 데이터 필터링
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_records = [
            record for record in self.performance_history
            if datetime.fromisoformat(record["timestamp"]).timestamp() > cutoff_date
        ]
        
        if not recent_records:
            return {"error": "최근 데이터가 없습니다"}
        
        # 기본 통계
        total_classifications = len(recent_records)
        avg_confidence = np.mean([r["confidence"] for r in recent_records])
        avg_processing_time = np.mean([r["processing_time"] for r in recent_records])
        
        # 정확도 계산 (실제 도메인이 있는 경우만)
        correct_records = [r for r in recent_records if r["is_correct"] is not None]
        accuracy = np.mean([r["is_correct"] for r in correct_records]) if correct_records else None
        
        # 도메인별 통계
        domain_stats = Counter([r["predicted_domain"] for r in recent_records])
        
        # 신뢰도 분포
        confidence_ranges = {
            "high": len([r for r in recent_records if r["confidence"] >= 0.8]),
            "medium": len([r for r in recent_records if 0.6 <= r["confidence"] < 0.8]),
            "low": len([r for r in recent_records if r["confidence"] < 0.6])
        }
        
        return {
            "total_classifications": total_classifications,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "domain_distribution": dict(domain_stats),
            "confidence_distribution": confidence_ranges,
            "period_days": days
        }
    
    def analyze_failure_cases(self, days: int = 7) -> List[Dict[str, Any]]:
        """실패 사례 분석"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_records = [
            record for record in self.performance_history
            if datetime.fromisoformat(record["timestamp"]).timestamp() > cutoff_date
        ]
        
        failure_cases = [
            record for record in recent_records
            if record["is_correct"] is False
        ]
        
        return failure_cases
    
    def get_domain_performance(self, domain: str, days: int = 7) -> Dict[str, Any]:
        """특정 도메인 성능 분석"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        domain_records = [
            record for record in self.performance_history
            if (datetime.fromisoformat(record["timestamp"]).timestamp() > cutoff_date and
                record["predicted_domain"] == domain)
        ]
        
        if not domain_records:
            return {"error": f"{domain} 도메인 데이터가 없습니다"}
        
        # 도메인별 통계
        total_predictions = len(domain_records)
        avg_confidence = np.mean([r["confidence"] for r in domain_records])
        
        # 정확도 (실제 도메인이 있는 경우만)
        correct_records = [r for r in domain_records if r["is_correct"] is not None]
        accuracy = np.mean([r["is_correct"] for r in correct_records]) if correct_records else None
        
        return {
            "domain": domain,
            "total_predictions": total_predictions,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "period_days": days
        }

class TermQualityAnalyzer:
    """용어 품질 분석기"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_term_quality(self, terms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """용어 품질 분석"""
        if not terms_data:
            return {"error": "분석할 용어 데이터가 없습니다"}
        
        # 기본 통계
        total_terms = len(terms_data)
        valid_terms = len([t for t in terms_data if t.get("is_valid", False)])
        high_quality_terms = len([t for t in terms_data if t.get("is_high_quality", False)])
        
        # 품질 점수 분포
        quality_scores = [t.get("quality_score", 0) for t in terms_data]
        avg_quality_score = np.mean(quality_scores)
        quality_distribution = {
            "excellent": len([s for s in quality_scores if s >= 90]),
            "good": len([s for s in quality_scores if 80 <= s < 90]),
            "fair": len([s for s in quality_scores if 70 <= s < 80]),
            "poor": len([s for s in quality_scores if s < 70])
        }
        
        # 신뢰도 분포
        confidences = [t.get("final_confidence", 0.0) for t in terms_data]
        avg_confidence = np.mean(confidences)
        confidence_distribution = {
            "high": len([c for c in confidences if c >= 0.8]),
            "medium": len([c for c in confidences if 0.6 <= c < 0.8]),
            "low": len([c for c in confidences if c < 0.6])
        }
        
        # 도메인별 분포
        domains = [t.get("domain", "기타/일반") for t in terms_data]
        domain_distribution = dict(Counter(domains))
        
        # 가중치 분포
        weights = [t.get("weight", 0.5) for t in terms_data]
        avg_weight = np.mean(weights)
        
        return {
            "total_terms": total_terms,
            "valid_terms": valid_terms,
            "high_quality_terms": high_quality_terms,
            "validity_rate": valid_terms / total_terms if total_terms > 0 else 0,
            "high_quality_rate": high_quality_terms / total_terms if total_terms > 0 else 0,
            "avg_quality_score": avg_quality_score,
            "quality_distribution": quality_distribution,
            "avg_confidence": avg_confidence,
            "confidence_distribution": confidence_distribution,
            "domain_distribution": domain_distribution,
            "avg_weight": avg_weight
        }
    
    def identify_problematic_terms(self, terms_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문제가 있는 용어 식별"""
        problematic_terms = []
        
        for term_data in terms_data:
            issues = []
            
            # 품질 점수 낮음
            if term_data.get("quality_score", 0) < 60:
                issues.append("품질 점수 낮음")
            
            # 신뢰도 낮음
            if term_data.get("final_confidence", 0.0) < 0.5:
                issues.append("신뢰도 낮음")
            
            # 유효하지 않음
            if not term_data.get("is_valid", False):
                issues.append("유효하지 않음")
            
            # 정의 없음
            if not term_data.get("definition", "").strip():
                issues.append("정의 없음")
            
            # 동의어 없음
            if not term_data.get("synonyms", []):
                issues.append("동의어 없음")
            
            if issues:
                problematic_terms.append({
                    "term": term_data.get("term", ""),
                    "issues": issues,
                    "quality_score": term_data.get("quality_score", 0),
                    "confidence": term_data.get("final_confidence", 0.0)
                })
        
        return problematic_terms

class PerformanceEvaluator:
    """성능 평가 메인 클래스"""
    
    def __init__(self, log_file: str = "logs/classification_performance.json"):
        self.extraction_evaluator = ExtractionEvaluator()
        self.classification_monitor = ClassificationMonitor(log_file)
        self.term_quality_analyzer = TermQualityAnalyzer()
        self.logger = get_logger(__name__)
    
    def evaluate_full_pipeline(self, 
                             extracted_terms: List[str],
                             validated_terms: List[Dict[str, Any]],
                             ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
        """전체 파이프라인 성능 평가"""
        results = {}
        
        # 1. 추출 성능 평가
        if ground_truth:
            extraction_metrics = self.extraction_evaluator.evaluate_extraction_quality(
                extracted_terms, ground_truth
            )
            results["extraction_performance"] = extraction_metrics
        
        # 2. 용어 품질 분석
        quality_analysis = self.term_quality_analyzer.analyze_term_quality(validated_terms)
        results["term_quality_analysis"] = quality_analysis
        
        # 3. 문제 용어 식별
        problematic_terms = self.term_quality_analyzer.identify_problematic_terms(validated_terms)
        results["problematic_terms"] = problematic_terms
        
        # 4. 분류 성능 모니터링
        classification_metrics = self.classification_monitor.get_performance_metrics()
        results["classification_performance"] = classification_metrics
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """성능 보고서 생성"""
        report = []
        report.append("=" * 50)
        report.append("법률 용어 추출 시스템 성능 보고서")
        report.append("=" * 50)
        report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 추출 성능
        if "extraction_performance" in results:
            ep = results["extraction_performance"]
            report.append("1. 추출 성능")
            report.append(f"   - 정밀도: {ep['precision']:.3f}")
            report.append(f"   - 재현율: {ep['recall']:.3f}")
            report.append(f"   - F1 점수: {ep['f1_score']:.3f}")
            report.append(f"   - 추출된 용어 수: {ep['extracted_count']}")
            report.append(f"   - 정답 용어 수: {ep['ground_truth_count']}")
            report.append("")
        
        # 용어 품질 분석
        if "term_quality_analysis" in results:
            tqa = results["term_quality_analysis"]
            report.append("2. 용어 품질 분석")
            report.append(f"   - 총 용어 수: {tqa['total_terms']}")
            report.append(f"   - 유효 용어 수: {tqa['valid_terms']}")
            report.append(f"   - 고품질 용어 수: {tqa['high_quality_terms']}")
            report.append(f"   - 유효성 비율: {tqa['validity_rate']:.3f}")
            report.append(f"   - 고품질 비율: {tqa['high_quality_rate']:.3f}")
            report.append(f"   - 평균 품질 점수: {tqa['avg_quality_score']:.1f}")
            report.append(f"   - 평균 신뢰도: {tqa['avg_confidence']:.3f}")
            report.append("")
        
        # 문제 용어
        if "problematic_terms" in results:
            pt = results["problematic_terms"]
            report.append("3. 문제 용어")
            report.append(f"   - 문제 용어 수: {len(pt)}")
            if pt:
                report.append("   - 주요 문제:")
                for term_info in pt[:5]:  # 상위 5개만 표시
                    report.append(f"     * {term_info['term']}: {', '.join(term_info['issues'])}")
            report.append("")
        
        # 분류 성능
        if "classification_performance" in results:
            cp = results["classification_performance"]
            report.append("4. 분류 성능")
            report.append(f"   - 총 분류 수: {cp.get('total_classifications', 0)}")
            report.append(f"   - 정확도: {cp.get('accuracy', 0):.3f}")
            report.append(f"   - 평균 신뢰도: {cp.get('avg_confidence', 0):.3f}")
            report.append(f"   - 평균 처리 시간: {cp.get('avg_processing_time', 0):.3f}초")
            report.append("")
        
        return "\n".join(report)
