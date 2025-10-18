#!/usr/bin/env python3
"""
법률 용어 품질 검증 시스템
추출된 용어들의 품질을 검증하고 개선 제안을 제공합니다.
"""

import json
import os
import re
import logging
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """품질 메트릭 데이터 클래스"""
    term: str
    frequency_score: float
    diversity_score: float
    legal_relevance_score: float
    domain_coherence_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]

class LegalTermQualityValidator:
    """법률 용어 품질 검증기"""
    
    def __init__(self):
        self.extracted_terms_file = "data/extracted_terms/extracted_legal_terms.json"
        self.semantic_relations_file = "data/extracted_terms/semantic_relations.json"
        self.output_dir = "data/extracted_terms/quality_validation"
        
        # 품질 검증 기준
        self.quality_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.2
        }
        
        # 법률 용어 패턴
        self.legal_patterns = self._initialize_legal_patterns()
        
        # 불용어 및 저품질 용어
        self.stop_words = self._initialize_stop_words()
        self.low_quality_patterns = self._initialize_low_quality_patterns()
    
    def _initialize_legal_patterns(self) -> Dict[str, List[str]]:
        """법률 용어 패턴 초기화"""
        return {
            "high_quality": [
                r"[가-힣]+법$",  # 법률명
                r"[가-힣]+권$",  # 권리
                r"[가-힣]+의무$",  # 의무
                r"[가-힣]+절차$",  # 절차
                r"[가-힣]+신청$",  # 신청
                r"[가-힣]+신고$",  # 신고
                r"[가-힣]+허가$",  # 허가
                r"[가-힣]+인가$",  # 인가
                r"[가-힣]+원$",  # 기관
                r"[가-힣]+청$",  # 기관
                r"[가-힣]+부$",  # 기관
                r"[가-힣]+위원회$",  # 기관
                r"[가-힣]+법원$",  # 법원
                r"[가-힣]+소송$",  # 소송
                r"[가-힣]+재판$",  # 재판
                r"[가-힣]+판결$",  # 판결
                r"[가-힣]+처분$",  # 처분
                r"[가-힣]+결정$",  # 결정
                r"[가-힣]+명령$",  # 명령
                r"[가-힣]+지시$"  # 지시
            ],
            "medium_quality": [
                r"[가-힣]{2,4}$",  # 일반적인 2-4글자 용어
                r"제\d+조$",  # 조문
                r"제\d+항$",  # 항
                r"제\d+호$",  # 호
                r"제\d+장$",  # 장
                r"제\d+절$",  # 절
                r"제\d+편$"  # 편
            ],
            "low_quality": [
                r"^\d+$",  # 숫자만
                r"^[가-힣]{1}$",  # 한 글자
                r"^[a-zA-Z]+$",  # 영문만
                r"^[가-힣]*[0-9]+[가-힣]*$",  # 숫자 포함
                r"^[가-힣]*[a-zA-Z]+[가-힣]*$"  # 영문 포함
            ]
        }
    
    def _initialize_stop_words(self) -> Set[str]:
        """불용어 초기화"""
        return {
            "것", "수", "등", "및", "또는", "그", "이", "저", "의", "가", "을", "를",
            "에", "에서", "로", "으로", "와", "과", "는", "은", "도", "만", "부터",
            "까지", "까지의", "에의", "에대한", "에관한", "에따른", "에의한",
            "있", "하", "되", "되", "되", "되", "되", "되", "되", "되", "되",
            "있", "하", "되", "되", "되", "되", "되", "되", "되", "되", "되"
        }
    
    def _initialize_low_quality_patterns(self) -> List[str]:
        """저품질 패턴 초기화"""
        return [
            r"^[0-9]+$",  # 숫자만
            r"^[a-zA-Z]+$",  # 영문만
            r"^[가-힣]{1}$",  # 한 글자
            r"^[가-힣]*[0-9]+[가-힣]*$",  # 숫자 포함
            r"^[가-힣]*[a-zA-Z]+[가-힣]*$",  # 영문 포함
            r"^[가-힣]*[!@#$%^&*()]+[가-힣]*$"  # 특수문자 포함
        ]
    
    def load_extracted_terms(self) -> Dict[str, Any]:
        """추출된 용어 로드"""
        logger.info("추출된 용어 로드 중...")
        
        with open(self.extracted_terms_file, 'r', encoding='utf-8') as f:
            extracted_terms = json.load(f)
        
        logger.info(f"로드된 용어 수: {len(extracted_terms)}")
        return extracted_terms
    
    def validate_term_quality(self, term: str, term_data: Dict[str, Any]) -> QualityMetrics:
        """개별 용어 품질 검증"""
        issues = []
        recommendations = []
        
        # 1. 빈도수 점수 (0-0.3)
        frequency_score = self._calculate_frequency_score(term_data.get('frequency', 0))
        
        # 2. 다양성 점수 (0-0.2)
        diversity_score = self._calculate_diversity_score(term_data)
        
        # 3. 법률 관련성 점수 (0-0.3)
        legal_relevance_score = self._calculate_legal_relevance_score(term)
        
        # 4. 도메인 일관성 점수 (0-0.2)
        domain_coherence_score = self._calculate_domain_coherence_score(term, term_data)
        
        # 전체 점수 계산
        overall_score = frequency_score + diversity_score + legal_relevance_score + domain_coherence_score
        
        # 문제점 식별
        if frequency_score < 0.1:
            issues.append("빈도수가 너무 낮음")
            recommendations.append("더 많은 데이터에서 사용되는 용어인지 확인")
        
        if diversity_score < 0.1:
            issues.append("소스 다양성이 부족함")
            recommendations.append("다양한 법령/판례에서 사용되는 용어인지 확인")
        
        if legal_relevance_score < 0.1:
            issues.append("법률 관련성이 낮음")
            recommendations.append("법률 용어로서의 적절성 검토")
        
        if domain_coherence_score < 0.1:
            issues.append("도메인 일관성이 부족함")
            recommendations.append("도메인 분류의 정확성 검토")
        
        # 불용어 검사
        if term in self.stop_words:
            issues.append("불용어로 분류됨")
            recommendations.append("불용어 목록에서 제거 고려")
        
        # 저품질 패턴 검사
        for pattern in self.low_quality_patterns:
            if re.match(pattern, term):
                issues.append("저품질 패턴에 해당")
                recommendations.append("용어 형식 개선 필요")
                break
        
        return QualityMetrics(
            term=term,
            frequency_score=frequency_score,
            diversity_score=diversity_score,
            legal_relevance_score=legal_relevance_score,
            domain_coherence_score=domain_coherence_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_frequency_score(self, frequency: int) -> float:
        """빈도수 점수 계산"""
        if frequency >= 50:
            return 0.3
        elif frequency >= 20:
            return 0.25
        elif frequency >= 10:
            return 0.2
        elif frequency >= 5:
            return 0.15
        elif frequency >= 2:
            return 0.1
        else:
            return 0.05
    
    def _calculate_diversity_score(self, term_data: Dict[str, Any]) -> float:
        """다양성 점수 계산"""
        sources = term_data.get('sources', [])
        contexts = term_data.get('context', [])
        
        # 소스 다양성 (0-0.1)
        unique_sources = len(set(sources))
        source_score = min(unique_sources / 10.0, 0.1)
        
        # 컨텍스트 다양성 (0-0.1)
        context_score = min(len(contexts) / 5.0, 0.1)
        
        return source_score + context_score
    
    def _calculate_legal_relevance_score(self, term: str) -> float:
        """법률 관련성 점수 계산"""
        # 고품질 패턴 검사
        for pattern in self.legal_patterns["high_quality"]:
            if re.match(pattern, term):
                return 0.3
        
        # 중품질 패턴 검사
        for pattern in self.legal_patterns["medium_quality"]:
            if re.match(pattern, term):
                return 0.2
        
        # 저품질 패턴 검사
        for pattern in self.legal_patterns["low_quality"]:
            if re.match(pattern, term):
                return 0.05
        
        # 일반적인 법률 용어 지표
        legal_indicators = [
            '법', '규칙', '령', '권', '의무', '책임', '절차', '신청', '신고',
            '허가', '인가', '승인', '원', '청', '부', '위원회', '법원',
            '행위', '처분', '결정', '명령', '지시', '소송', '재판', '판결'
        ]
        
        if any(indicator in term for indicator in legal_indicators):
            return 0.15
        
        return 0.1
    
    def _calculate_domain_coherence_score(self, term: str, term_data: Dict[str, Any]) -> float:
        """도메인 일관성 점수 계산"""
        domain = term_data.get('domain', '기타')
        category = term_data.get('category', '일반')
        
        # 도메인별 일관성 검사
        domain_keywords = {
            "형사법": ["범죄", "처벌", "형벌", "구속", "기소", "공소", "피고", "검사"],
            "민사법": ["계약", "손해배상", "소유권", "채권", "채무", "이행", "위반"],
            "가족법": ["혼인", "이혼", "상속", "양육", "위자료", "재산분할", "양육권"],
            "상사법": ["회사", "주식", "어음", "수표", "상행위", "회사법", "상법"],
            "노동법": ["근로", "근로자", "근로계약", "임금", "근로시간", "해고"],
            "부동산법": ["부동산", "토지", "건물", "등기", "소유권이전", "매매"],
            "특허법": ["특허", "특허권", "특허출원", "특허등록", "특허침해"],
            "행정법": ["행정처분", "행정소송", "행정법", "허가", "인가", "승인"]
        }
        
        if domain in domain_keywords:
            for keyword in domain_keywords[domain]:
                if keyword in term:
                    return 0.2
        
        # 카테고리별 일관성 검사
        category_keywords = {
            "법률명": ["법", "규칙", "령"],
            "권리": ["권"],
            "의무": ["의무", "책임"],
            "절차": ["절차", "신청", "신고"],
            "기관": ["원", "청", "부", "위원회", "법원"],
            "소송": ["소송", "재판", "판결"]
        }
        
        if category in category_keywords:
            for keyword in category_keywords[category]:
                if keyword in term:
                    return 0.15
        
        return 0.1
    
    def validate_all_terms(self, extracted_terms: Dict[str, Any]) -> Dict[str, QualityMetrics]:
        """모든 용어 품질 검증"""
        logger.info("모든 용어 품질 검증 시작")
        
        quality_metrics = {}
        total_terms = len(extracted_terms)
        
        for i, (term, term_data) in enumerate(extracted_terms.items()):
            if i % 1000 == 0:
                logger.info(f"진행률: {i}/{total_terms} ({i/total_terms*100:.1f}%)")
            
            quality_metrics[term] = self.validate_term_quality(term, term_data)
        
        logger.info("모든 용어 품질 검증 완료")
        return quality_metrics
    
    def generate_quality_report(self, quality_metrics: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """품질 보고서 생성"""
        logger.info("품질 보고서 생성 중")
        
        total_terms = len(quality_metrics)
        
        # 품질 등급별 분포
        quality_distribution = {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0
        }
        
        # 문제점별 통계
        issue_stats = defaultdict(int)
        
        # 도메인별 품질 통계
        domain_quality = defaultdict(list)
        
        for term, metrics in quality_metrics.items():
            # 품질 등급 분류
            if metrics.overall_score >= self.quality_thresholds["excellent"]:
                quality_distribution["excellent"] += 1
            elif metrics.overall_score >= self.quality_thresholds["good"]:
                quality_distribution["good"] += 1
            elif metrics.overall_score >= self.quality_thresholds["fair"]:
                quality_distribution["fair"] += 1
            else:
                quality_distribution["poor"] += 1
            
            # 문제점 통계
            for issue in metrics.issues:
                issue_stats[issue] += 1
        
        # 상위/하위 품질 용어
        sorted_terms = sorted(quality_metrics.items(), key=lambda x: x[1].overall_score, reverse=True)
        top_quality_terms = [term for term, metrics in sorted_terms[:20]]
        bottom_quality_terms = [term for term, metrics in sorted_terms[-20:]]
        
        # 품질 개선 권장사항
        improvement_recommendations = self._generate_improvement_recommendations(quality_metrics)
        
        report = {
            "summary": {
                "total_terms": total_terms,
                "validation_date": datetime.now().isoformat(),
                "average_quality_score": sum(m.overall_score for m in quality_metrics.values()) / total_terms
            },
            "quality_distribution": quality_distribution,
            "issue_statistics": dict(issue_stats),
            "top_quality_terms": top_quality_terms,
            "bottom_quality_terms": bottom_quality_terms,
            "improvement_recommendations": improvement_recommendations
        }
        
        logger.info("품질 보고서 생성 완료")
        return report
    
    def _generate_improvement_recommendations(self, quality_metrics: Dict[str, QualityMetrics]) -> List[str]:
        """품질 개선 권장사항 생성"""
        recommendations = []
        
        # 빈도수 문제
        low_frequency_count = sum(1 for m in quality_metrics.values() if m.frequency_score < 0.1)
        if low_frequency_count > 0:
            recommendations.append(f"빈도수가 낮은 용어 {low_frequency_count}개 제거 고려")
        
        # 다양성 문제
        low_diversity_count = sum(1 for m in quality_metrics.values() if m.diversity_score < 0.1)
        if low_diversity_count > 0:
            recommendations.append(f"소스 다양성이 부족한 용어 {low_diversity_count}개 검토 필요")
        
        # 법률 관련성 문제
        low_relevance_count = sum(1 for m in quality_metrics.values() if m.legal_relevance_score < 0.1)
        if low_relevance_count > 0:
            recommendations.append(f"법률 관련성이 낮은 용어 {low_relevance_count}개 재검토 필요")
        
        # 도메인 일관성 문제
        low_coherence_count = sum(1 for m in quality_metrics.values() if m.domain_coherence_score < 0.1)
        if low_coherence_count > 0:
            recommendations.append(f"도메인 일관성이 부족한 용어 {low_coherence_count}개 분류 재검토 필요")
        
        return recommendations
    
    def filter_high_quality_terms(self, quality_metrics: Dict[str, QualityMetrics], threshold: float = 0.6) -> Dict[str, QualityMetrics]:
        """고품질 용어 필터링"""
        logger.info(f"고품질 용어 필터링 (임계값: {threshold})")
        
        filtered_metrics = {
            term: metrics for term, metrics in quality_metrics.items()
            if metrics.overall_score >= threshold
        }
        
        logger.info(f"필터링 결과: {len(filtered_metrics)}/{len(quality_metrics)} 용어 유지")
        return filtered_metrics
    
    def save_quality_validation_results(self, quality_metrics: Dict[str, QualityMetrics], quality_report: Dict[str, Any]):
        """품질 검증 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 품질 메트릭 저장
        metrics_data = {}
        for term, metrics in quality_metrics.items():
            metrics_data[term] = {
                "term": metrics.term,
                "frequency_score": metrics.frequency_score,
                "diversity_score": metrics.diversity_score,
                "legal_relevance_score": metrics.legal_relevance_score,
                "domain_coherence_score": metrics.domain_coherence_score,
                "overall_score": metrics.overall_score,
                "issues": metrics.issues,
                "recommendations": metrics.recommendations
            }
        
        metrics_file = os.path.join(self.output_dir, "quality_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        # 품질 보고서 저장
        report_file = os.path.join(self.output_dir, "quality_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"품질 검증 결과 저장 완료: {self.output_dir}")
    
    def run_quality_validation(self):
        """품질 검증 실행"""
        logger.info("법률 용어 품질 검증 시작")
        
        try:
            # 추출된 용어 로드
            extracted_terms = self.load_extracted_terms()
            
            # 품질 검증
            quality_metrics = self.validate_all_terms(extracted_terms)
            
            # 품질 보고서 생성
            quality_report = self.generate_quality_report(quality_metrics)
            
            # 고품질 용어 필터링
            high_quality_metrics = self.filter_high_quality_terms(quality_metrics, 0.6)
            
            # 결과 저장
            self.save_quality_validation_results(quality_metrics, quality_report)
            
            # 고품질 용어만 별도 저장
            high_quality_file = os.path.join(self.output_dir, "high_quality_terms.json")
            with open(high_quality_file, 'w', encoding='utf-8') as f:
                json.dump(list(high_quality_metrics.keys()), f, ensure_ascii=False, indent=2)
            
            logger.info("법률 용어 품질 검증 완료")
            
            # 결과 요약 출력
            print(f"\n=== 품질 검증 결과 요약 ===")
            print(f"전체 용어 수: {len(quality_metrics)}")
            print(f"고품질 용어 수: {len(high_quality_metrics)}")
            print(f"평균 품질 점수: {quality_report['summary']['average_quality_score']:.3f}")
            print(f"품질 등급 분포: {quality_report['quality_distribution']}")
            
        except Exception as e:
            logger.error(f"품질 검증 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    validator = LegalTermQualityValidator()
    validator.run_quality_validation()

if __name__ == "__main__":
    main()
