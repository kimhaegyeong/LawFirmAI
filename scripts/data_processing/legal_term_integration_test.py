#!/usr/bin/env python3
"""
확장된 법률 용어 사전 통합 테스트
확장된 키워드 매핑 시스템의 성능을 테스트하고 개선 효과를 측정합니다.
"""

import json
import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.agents.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_term_integration_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalTermIntegrationTester:
    """법률 용어 통합 테스트기"""

    def __init__(self):
        self.test_questions = self._initialize_test_questions()
        self.expected_keywords = self._initialize_expected_keywords()
        self.output_dir = "data/extracted_terms/integration_test"

        # 테스트 결과 저장
        self.test_results = {
            "basic_mapper": {},
            "enhanced_mapper": {},
            "performance_comparison": {},
            "quality_metrics": {}
        }

    def _initialize_test_questions(self) -> List[Dict[str, str]]:
        """테스트 질문 초기화"""
        return [
            {
                "question": "계약서 검토 시 주의해야 할 사항은 무엇인가요?",
                "query_type": "contract_review",
                "domain": "민사법"
            },
            {
                "question": "손해배상 청구 절차는 어떻게 진행되나요?",
                "query_type": "damage_compensation",
                "domain": "민사법"
            },
            {
                "question": "이혼 소송에서 위자료는 어떻게 결정되나요?",
                "query_type": "divorce_proceedings",
                "domain": "가족법"
            },
            {
                "question": "부동산 매매 계약 시 등기 절차는 어떻게 되나요?",
                "query_type": "real_estate_transaction",
                "domain": "부동산법"
            },
            {
                "question": "특허 출원 시 필요한 서류는 무엇인가요?",
                "query_type": "patent_application",
                "domain": "특허법"
            },
            {
                "question": "근로자 해고 시 법적 절차는 어떻게 되나요?",
                "query_type": "employment_termination",
                "domain": "노동법"
            },
            {
                "question": "회사 설립 시 필요한 허가 절차는 무엇인가요?",
                "query_type": "company_establishment",
                "domain": "상사법"
            },
            {
                "question": "형사 사건에서 변호사 선임 절차는 어떻게 되나요?",
                "query_type": "criminal_defense",
                "domain": "형사법"
            },
            {
                "question": "행정처분에 대한 이의신청 방법은 무엇인가요?",
                "query_type": "administrative_appeal",
                "domain": "행정법"
            },
            {
                "question": "상속 포기 절차는 어떻게 진행되나요?",
                "query_type": "inheritance_renunciation",
                "domain": "가족법"
            }
        ]

    def _initialize_expected_keywords(self) -> Dict[str, List[str]]:
        """예상 키워드 초기화"""
        return {
            "contract_review": ["계약서", "당사자", "조건", "기간", "해지", "손해배상", "계약금", "위약금"],
            "damage_compensation": ["손해배상", "청구", "절차", "불법행위", "과실", "인과관계", "손해액"],
            "divorce_proceedings": ["이혼", "소송", "위자료", "재산분할", "양육권", "면접교섭권", "가정법원"],
            "real_estate_transaction": ["부동산", "매매", "계약", "등기", "소유권이전", "등기부등본", "부동산등기법"],
            "patent_application": ["특허", "출원", "서류", "특허청", "특허법", "발명", "특허권"],
            "employment_termination": ["근로자", "해고", "절차", "부당해고", "노동위원회", "근로기준법"],
            "company_establishment": ["회사", "설립", "허가", "절차", "주식회사", "자본금", "상법"],
            "criminal_defense": ["형사", "사건", "변호사", "선임", "절차", "피고", "검사", "형사소송법"],
            "administrative_appeal": ["행정처분", "이의신청", "방법", "행정소송", "행정법", "허가", "승인"],
            "inheritance_renunciation": ["상속", "포기", "절차", "상속인", "상속분", "유류분", "상속법"]
        }

    def test_basic_keyword_mapper(self) -> Dict[str, Any]:
        """기본 키워드 매퍼 테스트"""
        logger.info("기본 키워드 매퍼 테스트 시작")

        basic_mapper = LegalKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # 키워드 추출
            keywords = basic_mapper.get_keywords_for_question(question, query_type)

            # 가중치별 키워드 추출
            weighted_keywords = basic_mapper.get_weighted_keywords_for_question(question, query_type)

            # 키워드 포함도 계산
            sample_answer = f"{question}에 대한 답변입니다. {', '.join(keywords[:5])} 등의 내용을 포함합니다."
            coverage = basic_mapper.calculate_weighted_keyword_coverage(sample_answer, query_type, question)

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "keywords": keywords,
                "weighted_keywords": weighted_keywords,
                "coverage": coverage,
                "processing_time": end_time - start_time,
                "keyword_count": len(keywords)
            }

        logger.info("기본 키워드 매퍼 테스트 완료")
        return results

    def test_enhanced_keyword_mapper(self) -> Dict[str, Any]:
        """향상된 키워드 매퍼 테스트"""
        logger.info("향상된 키워드 매퍼 테스트 시작")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # 종합적인 키워드 매핑
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "comprehensive_result": comprehensive_result,
                "processing_time": end_time - start_time,
                "total_keywords": len(comprehensive_result.get("all_keywords", [])),
                "base_keywords": comprehensive_result.get("base_keywords", []),
                "contextual_keywords": comprehensive_result.get("contextual_data", {}).get("all_keywords", []),
                "semantic_keywords": comprehensive_result.get("semantic_data", {}).get("recommended_keywords", [])
            }

        logger.info("향상된 키워드 매퍼 테스트 완료")
        return results

    def test_semantic_keyword_mapper(self) -> Dict[str, Any]:
        """의미적 키워드 매퍼 테스트"""
        logger.info("의미적 키워드 매퍼 테스트 시작")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for test_case in self.test_questions:
            question = test_case["question"]
            query_type = test_case["query_type"]

            start_time = time.time()

            # 기본 키워드 추출
            basic_keywords = ["계약", "손해배상", "소송", "이혼", "부동산", "특허", "근로", "회사", "형사", "행정"]

            # 종합적인 키워드 매핑에서 의미적 데이터 추출
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)
            semantic_data = comprehensive_result.get("semantic_data", {})

            # 키워드 확장 (의미적 키워드 추천 사용)
            expanded_keywords = semantic_data.get("recommended_keywords", [])

            # 키워드 클러스터링 (의미적 클러스터 사용)
            clusters = semantic_data.get("semantic_clusters", {})

            end_time = time.time()

            results[query_type] = {
                "question": question,
                "semantic_recommendations": semantic_data,
                "expanded_keywords": expanded_keywords,
                "clusters": clusters,
                "processing_time": end_time - start_time,
                "expansion_ratio": len(expanded_keywords) / len(basic_keywords) if basic_keywords else 0
            }

        logger.info("의미적 키워드 매퍼 테스트 완료")
        return results

    def calculate_performance_metrics(self, basic_results: Dict, enhanced_results: Dict, semantic_results: Dict) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        logger.info("성능 메트릭 계산 중")

        metrics = {
            "keyword_coverage_improvement": {},
            "processing_time_comparison": {},
            "keyword_expansion_metrics": {},
            "overall_improvement": {}
        }

        # 키워드 커버리지 개선 측정
        for query_type in basic_results.keys():
            if query_type in enhanced_results:
                basic_keywords = set(basic_results[query_type]["keywords"])
                enhanced_keywords = set(enhanced_results[query_type]["base_keywords"])

                # 키워드 확장률
                expansion_ratio = len(enhanced_keywords) / len(basic_keywords) if basic_keywords else 0

                # 예상 키워드와의 매칭률
                expected_keywords = set(self.expected_keywords.get(query_type, []))
                basic_match_rate = len(basic_keywords & expected_keywords) / len(expected_keywords) if expected_keywords else 0
                enhanced_match_rate = len(enhanced_keywords & expected_keywords) / len(expected_keywords) if expected_keywords else 0

                metrics["keyword_coverage_improvement"][query_type] = {
                    "basic_keyword_count": len(basic_keywords),
                    "enhanced_keyword_count": len(enhanced_keywords),
                    "expansion_ratio": expansion_ratio,
                    "basic_match_rate": basic_match_rate,
                    "enhanced_match_rate": enhanced_match_rate,
                    "improvement_rate": enhanced_match_rate - basic_match_rate
                }

        # 처리 시간 비교
        for query_type in basic_results.keys():
            if query_type in enhanced_results:
                basic_time = basic_results[query_type]["processing_time"]
                enhanced_time = enhanced_results[query_type]["processing_time"]

                metrics["processing_time_comparison"][query_type] = {
                    "basic_time": basic_time,
                    "enhanced_time": enhanced_time,
                    "time_ratio": enhanced_time / basic_time if basic_time > 0 else 0
                }

        # 키워드 확장 메트릭
        for query_type in semantic_results.keys():
            expansion_ratio = semantic_results[query_type]["expansion_ratio"]
            metrics["keyword_expansion_metrics"][query_type] = {
                "expansion_ratio": expansion_ratio,
                "cluster_count": len(semantic_results[query_type]["clusters"])
            }

        # 전체 개선도 계산
        total_expansion_ratio = sum(m["expansion_ratio"] for m in metrics["keyword_coverage_improvement"].values()) / len(metrics["keyword_coverage_improvement"])
        total_match_improvement = sum(m["improvement_rate"] for m in metrics["keyword_coverage_improvement"].values()) / len(metrics["keyword_coverage_improvement"])
        total_time_ratio = sum(m["time_ratio"] for m in metrics["processing_time_comparison"].values()) / len(metrics["processing_time_comparison"])

        metrics["overall_improvement"] = {
            "average_expansion_ratio": total_expansion_ratio,
            "average_match_improvement": total_match_improvement,
            "average_time_ratio": total_time_ratio,
            "performance_score": (total_expansion_ratio + total_match_improvement) / 2
        }

        logger.info("성능 메트릭 계산 완료")
        return metrics

    def generate_test_report(self, basic_results: Dict, enhanced_results: Dict, semantic_results: Dict, performance_metrics: Dict) -> Dict[str, Any]:
        """테스트 보고서 생성"""
        logger.info("테스트 보고서 생성 중")

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_test_cases": len(self.test_questions),
                "test_duration": "약 5분"
            },
            "basic_mapper_results": {
                "total_keywords": sum(len(r["keywords"]) for r in basic_results.values()),
                "average_keywords_per_query": sum(len(r["keywords"]) for r in basic_results.values()) / len(basic_results),
                "average_processing_time": sum(r["processing_time"] for r in basic_results.values()) / len(basic_results)
            },
            "enhanced_mapper_results": {
                "total_keywords": sum(r["total_keywords"] for r in enhanced_results.values()),
                "average_keywords_per_query": sum(r["total_keywords"] for r in enhanced_results.values()) / len(enhanced_results),
                "average_processing_time": sum(r["processing_time"] for r in enhanced_results.values()) / len(enhanced_results)
            },
            "semantic_mapper_results": {
                "average_expansion_ratio": sum(r["expansion_ratio"] for r in semantic_results.values()) / len(semantic_results),
                "average_processing_time": sum(r["processing_time"] for r in semantic_results.values()) / len(semantic_results)
            },
            "performance_metrics": performance_metrics,
            "recommendations": self._generate_recommendations(performance_metrics)
        }

        logger.info("테스트 보고서 생성 완료")
        return report

    def _generate_recommendations(self, performance_metrics: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []

        overall_improvement = performance_metrics.get("overall_improvement", {})

        # 확장률 기반 권장사항
        expansion_ratio = overall_improvement.get("average_expansion_ratio", 0)
        if expansion_ratio > 2.0:
            recommendations.append("키워드 확장률이 우수합니다. 현재 설정을 유지하세요.")
        elif expansion_ratio > 1.5:
            recommendations.append("키워드 확장률이 양호합니다. 추가 개선 여지가 있습니다.")
        else:
            recommendations.append("키워드 확장률을 개선하기 위해 의미적 관계를 더 확장하세요.")

        # 매칭률 기반 권장사항
        match_improvement = overall_improvement.get("average_match_improvement", 0)
        if match_improvement > 0.2:
            recommendations.append("예상 키워드 매칭률이 크게 개선되었습니다.")
        elif match_improvement > 0.1:
            recommendations.append("예상 키워드 매칭률이 개선되었습니다.")
        else:
            recommendations.append("예상 키워드 매칭률 개선이 필요합니다.")

        # 처리 시간 기반 권장사항
        time_ratio = overall_improvement.get("average_time_ratio", 0)
        if time_ratio > 2.0:
            recommendations.append("처리 시간이 증가했습니다. 성능 최적화를 고려하세요.")
        elif time_ratio > 1.5:
            recommendations.append("처리 시간이 약간 증가했습니다. 모니터링이 필요합니다.")
        else:
            recommendations.append("처리 시간이 적절합니다.")

        # 성능 점수 기반 권장사항
        performance_score = overall_improvement.get("performance_score", 0)
        if performance_score > 0.8:
            recommendations.append("전체 성능이 우수합니다. 현재 설정을 유지하세요.")
        elif performance_score > 0.6:
            recommendations.append("전체 성능이 양호합니다. 추가 개선을 통해 더 향상시킬 수 있습니다.")
        else:
            recommendations.append("전체 성능 개선이 필요합니다. 키워드 매핑 전략을 재검토하세요.")

        return recommendations

    def save_test_results(self, test_report: Dict[str, Any]):
        """테스트 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 테스트 보고서 저장
        report_file = os.path.join(self.output_dir, "integration_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"테스트 결과 저장 완료: {self.output_dir}")

    def run_integration_test(self):
        """통합 테스트 실행"""
        logger.info("법률 용어 사전 통합 테스트 시작")

        try:
            # 기본 키워드 매퍼 테스트
            basic_results = self.test_basic_keyword_mapper()

            # 향상된 키워드 매퍼 테스트
            enhanced_results = self.test_enhanced_keyword_mapper()

            # 의미적 키워드 매퍼 테스트
            semantic_results = self.test_semantic_keyword_mapper()

            # 성능 메트릭 계산
            performance_metrics = self.calculate_performance_metrics(basic_results, enhanced_results, semantic_results)

            # 테스트 보고서 생성
            test_report = self.generate_test_report(basic_results, enhanced_results, semantic_results, performance_metrics)

            # 결과 저장
            self.save_test_results(test_report)

            logger.info("법률 용어 사전 통합 테스트 완료")

            # 결과 요약 출력
            print(f"\n=== 통합 테스트 결과 요약 ===")
            print(f"테스트 케이스 수: {test_report['test_summary']['total_test_cases']}")
            print(f"기본 매퍼 평균 키워드 수: {test_report['basic_mapper_results']['average_keywords_per_query']:.1f}")
            print(f"향상된 매퍼 평균 키워드 수: {test_report['enhanced_mapper_results']['average_keywords_per_query']:.1f}")
            print(f"평균 확장률: {test_report['semantic_mapper_results']['average_expansion_ratio']:.2f}")
            print(f"전체 성능 점수: {performance_metrics['overall_improvement']['performance_score']:.3f}")

            print(f"\n=== 개선 권장사항 ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

        except Exception as e:
            logger.error(f"통합 테스트 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    tester = LegalTermIntegrationTester()
    tester.run_integration_test()

if __name__ == "__main__":
    main()
