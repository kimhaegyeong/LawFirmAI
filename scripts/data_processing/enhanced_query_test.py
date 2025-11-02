#!/usr/bin/env python3
"""
향상된 법률 질의 테스트 시스템
실제 사용자 질문에 대한 키워드 매핑 성능을 더 정확하게 측정합니다.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.agents.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedQueryTester:
    """향상된 질의 테스트기"""

    def __init__(self):
        self.test_queries = self._initialize_realistic_queries()
        self.output_dir = "data/extracted_terms/enhanced_query_test"

    def _initialize_realistic_queries(self) -> List[Dict[str, Any]]:
        """현실적인 테스트 질의 초기화"""
        return [
            {
                "question": "계약서에서 위약금 조항이 너무 높게 설정되어 있는데, 법적으로 문제가 될까요?",
                "query_type": "contract_review",
                "domain": "민사법",
                "legal_keywords": ["계약서", "위약금", "조항", "민법", "계약법", "손해배상", "위약금제한"],
                "context_keywords": ["법적", "문제", "설정", "높게", "유효성"],
                "expected_terms": ["민법", "398조", "위약금", "손해배상", "계약서", "조항"]
            },
            {
                "question": "교통사고로 인한 손해배상 청구 시 필요한 증거자료는 무엇인가요?",
                "query_type": "damage_compensation",
                "domain": "민사법",
                "legal_keywords": ["교통사고", "손해배상", "청구", "증거자료", "불법행위", "과실", "인과관계"],
                "context_keywords": ["필요한", "시", "인한", "로", "무엇인가요"],
                "expected_terms": ["민법", "750조", "불법행위", "손해배상", "교통사고", "증거"]
            },
            {
                "question": "이혼 소송에서 자녀 양육권을 결정하는 기준은 무엇인가요?",
                "query_type": "divorce_proceedings",
                "domain": "가족법",
                "legal_keywords": ["이혼", "소송", "자녀", "양육권", "결정", "기준", "가정법원", "가족법"],
                "context_keywords": ["에서", "을", "하는", "는", "무엇인가요"],
                "expected_terms": ["가족법", "양육권", "이혼", "자녀", "가정법원", "기준"]
            },
            {
                "question": "부동산 매매 계약 시 등기 이전 절차와 필요한 서류는 무엇인가요?",
                "query_type": "real_estate_transaction",
                "domain": "부동산법",
                "legal_keywords": ["부동산", "매매", "계약", "등기", "이전", "절차", "서류", "등기부등본"],
                "context_keywords": ["시", "와", "필요한", "무엇인가요"],
                "expected_terms": ["부동산등기법", "등기", "소유권이전", "매매계약", "등기부등본"]
            },
            {
                "question": "특허 출원 시 발명의 신규성과 진보성을 어떻게 입증해야 하나요?",
                "query_type": "patent_application",
                "domain": "특허법",
                "legal_keywords": ["특허", "출원", "발명", "신규성", "진보성", "입증", "특허청", "특허법"],
                "context_keywords": ["시", "의", "과", "를", "어떻게", "해야", "하나요"],
                "expected_terms": ["특허법", "신규성", "진보성", "특허출원", "발명", "특허청"]
            }
        ]

    def test_keyword_extraction_accuracy(self) -> Dict[str, Any]:
        """키워드 추출 정확도 테스트"""
        logger.info("키워드 추출 정확도 테스트 시작")

        enhanced_mapper = EnhancedKeywordMapper()
        results = {}

        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            legal_keywords = query["legal_keywords"]
            context_keywords = query["context_keywords"]
            expected_terms = query["expected_terms"]

            logger.info(f"테스트 {i+1}/{len(self.test_queries)}: {query_type}")

            start_time = time.time()

            # 종합적인 키워드 매핑
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)

            end_time = time.time()

            # 추출된 키워드들
            all_keywords = comprehensive_result.get("all_keywords", [])
            base_keywords = comprehensive_result.get("base_keywords", [])
            contextual_keywords = comprehensive_result.get("contextual_data", {}).get("all_keywords", [])
            semantic_keywords = comprehensive_result.get("semantic_data", {}).get("recommended_keywords", [])

            # 정확도 계산
            legal_match = [kw for kw in legal_keywords if kw in all_keywords]
            context_match = [kw for kw in context_keywords if kw in all_keywords]
            expected_match = [kw for kw in expected_terms if kw in all_keywords]

            legal_accuracy = len(legal_match) / len(legal_keywords) if legal_keywords else 0
            context_accuracy = len(context_match) / len(context_keywords) if context_keywords else 0
            expected_accuracy = len(expected_match) / len(expected_terms) if expected_terms else 0

            results[query_type] = {
                "question": question,
                "legal_keywords": legal_keywords,
                "context_keywords": context_keywords,
                "expected_terms": expected_terms,
                "extracted_keywords": {
                    "all_keywords": all_keywords,
                    "base_keywords": base_keywords,
                    "contextual_keywords": contextual_keywords,
                    "semantic_keywords": semantic_keywords
                },
                "accuracy_metrics": {
                    "legal_accuracy": legal_accuracy,
                    "context_accuracy": context_accuracy,
                    "expected_accuracy": expected_accuracy,
                    "overall_accuracy": (legal_accuracy + context_accuracy + expected_accuracy) / 3
                },
                "matched_keywords": {
                    "legal_matched": legal_match,
                    "context_matched": context_match,
                    "expected_matched": expected_match
                },
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }

        logger.info("키워드 추출 정확도 테스트 완료")
        return results

    def test_keyword_coverage_analysis(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 커버리지 분석"""
        logger.info("키워드 커버리지 분석 시작")

        coverage_analysis = {
            "domain_coverage": {},
            "keyword_type_coverage": {},
            "missing_keywords": {},
            "recommendations": []
        }

        # 도메인별 커버리지
        domain_stats = {}
        for query_type, result in extraction_results.items():
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "기타")

            if domain not in domain_stats:
                domain_stats[domain] = {
                    "total_legal_keywords": 0,
                    "matched_legal_keywords": 0,
                    "total_expected_terms": 0,
                    "matched_expected_terms": 0,
                    "queries": 0
                }

            domain_stats[domain]["total_legal_keywords"] += len(result["legal_keywords"])
            domain_stats[domain]["matched_legal_keywords"] += len(result["matched_keywords"]["legal_matched"])
            domain_stats[domain]["total_expected_terms"] += len(result["expected_terms"])
            domain_stats[domain]["matched_expected_terms"] += len(result["matched_keywords"]["expected_matched"])
            domain_stats[domain]["queries"] += 1

        # 도메인별 커버리지 계산
        for domain, stats in domain_stats.items():
            legal_coverage = stats["matched_legal_keywords"] / stats["total_legal_keywords"] if stats["total_legal_keywords"] > 0 else 0
            expected_coverage = stats["matched_expected_terms"] / stats["total_expected_terms"] if stats["total_expected_terms"] > 0 else 0

            coverage_analysis["domain_coverage"][domain] = {
                "legal_coverage": legal_coverage,
                "expected_coverage": expected_coverage,
                "overall_coverage": (legal_coverage + expected_coverage) / 2,
                "query_count": stats["queries"]
            }

        # 키워드 타입별 커버리지
        total_legal = sum(len(r["legal_keywords"]) for r in extraction_results.values())
        total_context = sum(len(r["context_keywords"]) for r in extraction_results.values())
        total_expected = sum(len(r["expected_terms"]) for r in extraction_results.values())

        matched_legal = sum(len(r["matched_keywords"]["legal_matched"]) for r in extraction_results.values())
        matched_context = sum(len(r["matched_keywords"]["context_matched"]) for r in extraction_results.values())
        matched_expected = sum(len(r["matched_keywords"]["expected_matched"]) for r in extraction_results.values())

        coverage_analysis["keyword_type_coverage"] = {
            "legal_keywords": {
                "total": total_legal,
                "matched": matched_legal,
                "coverage_rate": matched_legal / total_legal if total_legal > 0 else 0
            },
            "context_keywords": {
                "total": total_context,
                "matched": matched_context,
                "coverage_rate": matched_context / total_context if total_context > 0 else 0
            },
            "expected_terms": {
                "total": total_expected,
                "matched": matched_expected,
                "coverage_rate": matched_expected / total_expected if total_expected > 0 else 0
            }
        }

        # 누락된 키워드 분석
        for query_type, result in extraction_results.items():
            missing_legal = [kw for kw in result["legal_keywords"] if kw not in result["extracted_keywords"]["all_keywords"]]
            missing_expected = [kw for kw in result["expected_terms"] if kw not in result["extracted_keywords"]["all_keywords"]]

            coverage_analysis["missing_keywords"][query_type] = {
                "missing_legal_keywords": missing_legal,
                "missing_expected_terms": missing_expected,
                "missing_count": len(missing_legal) + len(missing_expected)
            }

        # 개선 권장사항 생성
        overall_coverage = sum(c["overall_coverage"] for c in coverage_analysis["domain_coverage"].values()) / len(coverage_analysis["domain_coverage"])

        if overall_coverage < 0.3:
            coverage_analysis["recommendations"].append("전체 키워드 커버리지가 매우 낮습니다. 키워드 매핑 시스템을 전면 재검토하세요.")
        elif overall_coverage < 0.5:
            coverage_analysis["recommendations"].append("키워드 커버리지가 낮습니다. 의미적 관계를 확장하고 도메인별 용어를 보강하세요.")
        elif overall_coverage < 0.7:
            coverage_analysis["recommendations"].append("키워드 커버리지가 보통입니다. 추가 개선을 통해 더 향상시킬 수 있습니다.")
        else:
            coverage_analysis["recommendations"].append("키워드 커버리지가 양호합니다. 현재 설정을 유지하세요.")

        # 도메인별 권장사항
        for domain, coverage in coverage_analysis["domain_coverage"].items():
            if coverage["overall_coverage"] < 0.3:
                coverage_analysis["recommendations"].append(f"{domain} 도메인의 커버리지가 매우 낮습니다. 해당 도메인 용어를 대폭 확장하세요.")
            elif coverage["overall_coverage"] < 0.5:
                coverage_analysis["recommendations"].append(f"{domain} 도메인의 커버리지를 개선하세요.")

        logger.info("키워드 커버리지 분석 완료")
        return coverage_analysis

    def test_keyword_expansion_effectiveness(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 확장 효과성 테스트"""
        logger.info("키워드 확장 효과성 테스트 시작")

        expansion_analysis = {
            "expansion_metrics": {},
            "semantic_effectiveness": {},
            "contextual_effectiveness": {},
            "overall_effectiveness": {}
        }

        for query_type, result in extraction_results.items():
            base_keywords = result["extracted_keywords"]["base_keywords"]
            contextual_keywords = result["extracted_keywords"]["contextual_keywords"]
            semantic_keywords = result["extracted_keywords"]["semantic_keywords"]
            all_keywords = result["extracted_keywords"]["all_keywords"]

            # 확장 메트릭
            base_count = len(base_keywords)
            contextual_count = len(contextual_keywords)
            semantic_count = len(semantic_keywords)
            total_count = len(all_keywords)

            expansion_ratio = total_count / base_count if base_count > 0 else 0
            contextual_expansion = contextual_count / base_count if base_count > 0 else 0
            semantic_expansion = semantic_count / base_count if base_count > 0 else 0

            expansion_analysis["expansion_metrics"][query_type] = {
                "base_keywords": base_count,
                "contextual_keywords": contextual_count,
                "semantic_keywords": semantic_count,
                "total_keywords": total_count,
                "expansion_ratio": expansion_ratio,
                "contextual_expansion": contextual_expansion,
                "semantic_expansion": semantic_expansion
            }

            # 의미적 효과성 (예상 용어와의 매칭)
            semantic_matches = [kw for kw in semantic_keywords if kw in result["expected_terms"]]
            semantic_effectiveness = len(semantic_matches) / len(semantic_keywords) if semantic_keywords else 0

            expansion_analysis["semantic_effectiveness"][query_type] = {
                "semantic_matches": semantic_matches,
                "effectiveness_rate": semantic_effectiveness
            }

            # 컨텍스트 효과성 (법률 키워드와의 매칭)
            contextual_matches = [kw for kw in contextual_keywords if kw in result["legal_keywords"]]
            contextual_effectiveness = len(contextual_matches) / len(contextual_keywords) if contextual_keywords else 0

            expansion_analysis["contextual_effectiveness"][query_type] = {
                "contextual_matches": contextual_matches,
                "effectiveness_rate": contextual_effectiveness
            }

        # 전체 효과성 계산
        avg_expansion_ratio = sum(m["expansion_ratio"] for m in expansion_analysis["expansion_metrics"].values()) / len(expansion_analysis["expansion_metrics"])
        avg_semantic_effectiveness = sum(e["effectiveness_rate"] for e in expansion_analysis["semantic_effectiveness"].values()) / len(expansion_analysis["semantic_effectiveness"])
        avg_contextual_effectiveness = sum(e["effectiveness_rate"] for e in expansion_analysis["contextual_effectiveness"].values()) / len(expansion_analysis["contextual_effectiveness"])

        expansion_analysis["overall_effectiveness"] = {
            "average_expansion_ratio": avg_expansion_ratio,
            "average_semantic_effectiveness": avg_semantic_effectiveness,
            "average_contextual_effectiveness": avg_contextual_effectiveness,
            "overall_score": (avg_expansion_ratio + avg_semantic_effectiveness + avg_contextual_effectiveness) / 3
        }

        logger.info("키워드 확장 효과성 테스트 완료")
        return expansion_analysis

    def generate_comprehensive_report(self, extraction_results: Dict[str, Any], coverage_analysis: Dict[str, Any], expansion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """종합 보고서 생성"""
        logger.info("종합 보고서 생성 중")

        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "test_type": "enhanced_keyword_extraction_accuracy"
            },
            "extraction_results": extraction_results,
            "coverage_analysis": coverage_analysis,
            "expansion_analysis": expansion_analysis,
            "performance_summary": {
                "average_legal_accuracy": sum(r["accuracy_metrics"]["legal_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_context_accuracy": sum(r["accuracy_metrics"]["context_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_expected_accuracy": sum(r["accuracy_metrics"]["expected_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_overall_accuracy": sum(r["accuracy_metrics"]["overall_accuracy"] for r in extraction_results.values()) / len(extraction_results),
                "average_processing_time": sum(r["processing_time"] for r in extraction_results.values()) / len(extraction_results)
            },
            "recommendations": self._generate_final_recommendations(extraction_results, coverage_analysis, expansion_analysis)
        }

        logger.info("종합 보고서 생성 완료")
        return report

    def _generate_final_recommendations(self, extraction_results: Dict[str, Any], coverage_analysis: Dict[str, Any], expansion_analysis: Dict[str, Any]) -> List[str]:
        """최종 권장사항 생성"""
        recommendations = []

        # 정확도 기반 권장사항
        avg_accuracy = sum(r["accuracy_metrics"]["overall_accuracy"] for r in extraction_results.values()) / len(extraction_results)

        if avg_accuracy < 0.3:
            recommendations.append("키워드 추출 정확도가 매우 낮습니다. 키워드 매핑 알고리즘을 전면 재검토하세요.")
        elif avg_accuracy < 0.5:
            recommendations.append("키워드 추출 정확도가 낮습니다. 의미적 관계와 컨텍스트 매핑을 강화하세요.")
        elif avg_accuracy < 0.7:
            recommendations.append("키워드 추출 정확도가 보통입니다. 추가 개선을 통해 더 향상시킬 수 있습니다.")
        else:
            recommendations.append("키워드 추출 정확도가 양호합니다. 현재 설정을 유지하세요.")

        # 커버리지 기반 권장사항
        overall_coverage = sum(c["overall_coverage"] for c in coverage_analysis["domain_coverage"].values()) / len(coverage_analysis["domain_coverage"])

        if overall_coverage < 0.3:
            recommendations.append("키워드 커버리지가 매우 낮습니다. 도메인별 용어 사전을 대폭 확장하세요.")
        elif overall_coverage < 0.5:
            recommendations.append("키워드 커버리지를 개선하기 위해 의미적 관계를 확장하세요.")

        # 확장 효과성 기반 권장사항
        expansion_score = expansion_analysis["overall_effectiveness"]["overall_score"]

        if expansion_score < 0.3:
            recommendations.append("키워드 확장 효과성이 낮습니다. 의미적 매핑과 컨텍스트 인식을 개선하세요.")
        elif expansion_score < 0.5:
            recommendations.append("키워드 확장 효과성을 개선하기 위해 학습 데이터를 확장하세요.")

        # 처리 시간 기반 권장사항
        avg_processing_time = sum(r["processing_time"] for r in extraction_results.values()) / len(extraction_results)

        if avg_processing_time > 0.1:
            recommendations.append("처리 시간이 길어 성능 최적화가 필요합니다.")
        elif avg_processing_time > 0.05:
            recommendations.append("처리 시간이 적절하지만 추가 최적화 여지가 있습니다.")
        else:
            recommendations.append("처리 시간이 우수합니다. 현재 성능을 유지하세요.")

        return recommendations

    def save_test_results(self, test_report: Dict[str, Any]):
        """테스트 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 테스트 보고서 저장
        report_file = os.path.join(self.output_dir, "enhanced_query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)

        logger.info(f"테스트 결과 저장 완료: {self.output_dir}")

    def run_enhanced_test(self):
        """향상된 테스트 실행"""
        logger.info("향상된 법률 질의 테스트 시작")

        try:
            # 키워드 추출 정확도 테스트
            extraction_results = self.test_keyword_extraction_accuracy()

            # 키워드 커버리지 분석
            coverage_analysis = self.test_keyword_coverage_analysis(extraction_results)

            # 키워드 확장 효과성 테스트
            expansion_analysis = self.test_keyword_expansion_effectiveness(extraction_results)

            # 종합 보고서 생성
            test_report = self.generate_comprehensive_report(extraction_results, coverage_analysis, expansion_analysis)

            # 결과 저장
            self.save_test_results(test_report)

            logger.info("향상된 법률 질의 테스트 완료")

            # 결과 요약 출력
            print(f"\n=== 향상된 법률 질의 테스트 결과 요약 ===")
            print(f"총 테스트 질의 수: {test_report['test_summary']['total_queries']}")
            print(f"평균 법률 키워드 정확도: {test_report['performance_summary']['average_legal_accuracy']:.3f}")
            print(f"평균 컨텍스트 키워드 정확도: {test_report['performance_summary']['average_context_accuracy']:.3f}")
            print(f"평균 예상 용어 정확도: {test_report['performance_summary']['average_expected_accuracy']:.3f}")
            print(f"평균 전체 정확도: {test_report['performance_summary']['average_overall_accuracy']:.3f}")
            print(f"평균 처리 시간: {test_report['performance_summary']['average_processing_time']:.4f}초")

            print(f"\n=== 도메인별 커버리지 ===")
            for domain, coverage in test_report['coverage_analysis']['domain_coverage'].items():
                print(f"{domain}: {coverage['overall_coverage']:.3f}")

            print(f"\n=== 확장 효과성 ===")
            expansion_metrics = test_report['expansion_analysis']['overall_effectiveness']
            print(f"평균 확장률: {expansion_metrics['average_expansion_ratio']:.2f}")
            print(f"평균 의미적 효과성: {expansion_metrics['average_semantic_effectiveness']:.3f}")
            print(f"평균 컨텍스트 효과성: {expansion_metrics['average_contextual_effectiveness']:.3f}")
            print(f"전체 효과성 점수: {expansion_metrics['overall_score']:.3f}")

            print(f"\n=== 개선 권장사항 ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")

            # 상세 결과 예시 출력
            print(f"\n=== 상세 결과 예시 (첫 번째 질의) ===")
            first_query = list(extraction_results.values())[0]
            print(f"질문: {first_query['question']}")
            print(f"법률 키워드 정확도: {first_query['accuracy_metrics']['legal_accuracy']:.3f}")
            print(f"예상 용어 정확도: {first_query['accuracy_metrics']['expected_accuracy']:.3f}")
            print(f"전체 정확도: {first_query['accuracy_metrics']['overall_accuracy']:.3f}")
            print(f"매칭된 법률 키워드: {first_query['matched_keywords']['legal_matched']}")
            print(f"매칭된 예상 용어: {first_query['matched_keywords']['expected_matched']}")

        except Exception as e:
            logger.error(f"향상된 테스트 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    tester = EnhancedQueryTester()
    tester.run_enhanced_test()

if __name__ == "__main__":
    main()
