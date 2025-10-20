#!/usr/bin/env python3
"""
실제 사용자 질문 기반 키워드 매핑 테스트
실제 법률 상담에서 나올 수 있는 질문들로 키워드 매핑 성능을 평가합니다.
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

from source.services.langgraph.keyword_mapper import EnhancedKeywordMapper, LegalKeywordMapper

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/realistic_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealisticQueryTester:
    """현실적인 질의 테스트기"""
    
    def __init__(self):
        self.test_queries = self._initialize_realistic_queries()
        self.output_dir = "data/extracted_terms/realistic_query_test"
        
    def _initialize_realistic_queries(self) -> List[Dict[str, Any]]:
        """현실적인 테스트 질의 초기화"""
        return [
            {
                "question": "아파트 계약서에 위약금이 계약금의 10배로 되어 있는데, 이게 합법적인가요?",
                "query_type": "contract_review",
                "domain": "민사법",
                "realistic_keywords": ["아파트", "계약서", "위약금", "계약금", "10배", "합법", "민법", "398조"],
                "legal_concepts": ["위약금제한", "손해배상", "계약서", "민법"]
            },
            {
                "question": "교통사고 났는데 상대방이 보험처리 안 해주려고 해요. 어떻게 해야 하나요?",
                "query_type": "damage_compensation",
                "domain": "민사법",
                "realistic_keywords": ["교통사고", "상대방", "보험처리", "손해배상", "보험회사", "민사소송"],
                "legal_concepts": ["교통사고", "손해배상", "보험", "민사소송", "불법행위"]
            },
            {
                "question": "이혼하려는데 아이 양육권을 어떻게 결정하나요?",
                "query_type": "divorce_proceedings",
                "domain": "가족법",
                "realistic_keywords": ["이혼", "아이", "양육권", "결정", "가정법원", "양육비"],
                "legal_concepts": ["이혼", "양육권", "가정법원", "가족법", "양육비"]
            },
            {
                "question": "집 사려는데 중개업자가 등기부등본을 안 보여주려고 해요. 어떻게 해야 하나요?",
                "query_type": "real_estate_transaction",
                "domain": "부동산법",
                "realistic_keywords": ["집", "중개업자", "등기부등본", "부동산", "매매", "등기"],
                "legal_concepts": ["부동산매매", "등기부등본", "중개업", "부동산등기법"]
            },
            {
                "question": "회사에서 갑자기 해고 통보를 받았는데, 이게 부당해고인가요?",
                "query_type": "employment_termination",
                "domain": "노동법",
                "realistic_keywords": ["회사", "해고", "통보", "부당해고", "근로기준법", "노동위원회"],
                "legal_concepts": ["해고", "부당해고", "근로기준법", "노동위원회", "구제신청"]
            },
            {
                "question": "특허 출원하려는데 비슷한 발명이 이미 있는지 어떻게 확인하나요?",
                "query_type": "patent_application",
                "domain": "특허법",
                "realistic_keywords": ["특허", "출원", "발명", "신규성", "특허청", "선행기술"],
                "legal_concepts": ["특허출원", "신규성", "특허청", "선행기술조사", "특허법"]
            },
            {
                "question": "회사 설립하려는데 자본금은 얼마나 필요한가요?",
                "query_type": "company_establishment",
                "domain": "상사법",
                "realistic_keywords": ["회사", "설립", "자본금", "주식회사", "상법", "등기"],
                "legal_concepts": ["회사설립", "자본금", "주식회사", "상법", "등기"]
            },
            {
                "question": "형사사건으로 기소되었는데 변호사 선임이 필수인가요?",
                "query_type": "criminal_defense",
                "domain": "형사법",
                "realistic_keywords": ["형사사건", "기소", "변호사", "선임", "국선변호", "형사소송법"],
                "legal_concepts": ["형사사건", "변호사선임", "국선변호", "형사소송법", "피고"]
            }
        ]
    
    def test_keyword_relevance(self) -> Dict[str, Any]:
        """키워드 관련성 테스트"""
        logger.info("키워드 관련성 테스트 시작")
        
        enhanced_mapper = EnhancedKeywordMapper()
        results = {}
        
        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            realistic_keywords = query["realistic_keywords"]
            legal_concepts = query["legal_concepts"]
            
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
            
            # 관련성 분석
            realistic_matches = [kw for kw in realistic_keywords if kw in all_keywords]
            legal_concept_matches = [kw for kw in legal_concepts if kw in all_keywords]
            
            realistic_relevance = len(realistic_matches) / len(realistic_keywords) if realistic_keywords else 0
            legal_concept_relevance = len(legal_concept_matches) / len(legal_concepts) if legal_concepts else 0
            
            # 키워드 품질 평가
            quality_score = self._evaluate_keyword_quality(all_keywords, realistic_keywords, legal_concepts)
            
            results[query_type] = {
                "question": question,
                "realistic_keywords": realistic_keywords,
                "legal_concepts": legal_concepts,
                "extracted_keywords": {
                    "all_keywords": all_keywords,
                    "base_keywords": base_keywords,
                    "contextual_keywords": contextual_keywords,
                    "semantic_keywords": semantic_keywords
                },
                "relevance_metrics": {
                    "realistic_relevance": realistic_relevance,
                    "legal_concept_relevance": legal_concept_relevance,
                    "overall_relevance": (realistic_relevance + legal_concept_relevance) / 2,
                    "quality_score": quality_score
                },
                "matched_keywords": {
                    "realistic_matched": realistic_matches,
                    "legal_concept_matched": legal_concept_matches
                },
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }
        
        logger.info("키워드 관련성 테스트 완료")
        return results
    
    def _evaluate_keyword_quality(self, extracted_keywords: List[str], realistic_keywords: List[str], legal_concepts: List[str]) -> float:
        """키워드 품질 평가"""
        quality_score = 0.0
        
        # 법률 관련성 점수 (0-0.4)
        legal_indicators = [
            '법', '규칙', '령', '권', '의무', '책임', '절차', '신청', '신고',
            '허가', '인가', '승인', '원', '청', '부', '위원회', '법원',
            '행위', '처분', '결정', '명령', '지시', '소송', '재판', '판결',
            '계약', '손해배상', '이혼', '상속', '부동산', '특허', '근로', '회사'
        ]
        
        legal_count = sum(1 for kw in extracted_keywords if any(indicator in kw for indicator in legal_indicators))
        legal_score = min(legal_count / len(extracted_keywords), 0.4) if extracted_keywords else 0
        quality_score += legal_score
        
        # 현실성 점수 (0-0.3)
        realistic_count = sum(1 for kw in extracted_keywords if kw in realistic_keywords)
        realistic_score = min(realistic_count / len(realistic_keywords), 0.3) if realistic_keywords else 0
        quality_score += realistic_score
        
        # 법률 개념 일치 점수 (0-0.3)
        concept_count = sum(1 for kw in extracted_keywords if kw in legal_concepts)
        concept_score = min(concept_count / len(legal_concepts), 0.3) if legal_concepts else 0
        quality_score += concept_score
        
        return min(quality_score, 1.0)
    
    def test_keyword_diversity_and_coverage(self, relevance_results: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 다양성 및 커버리지 테스트"""
        logger.info("키워드 다양성 및 커버리지 테스트 시작")
        
        diversity_analysis = {
            "keyword_diversity": {},
            "domain_coverage": {},
            "concept_coverage": {},
            "overall_metrics": {}
        }
        
        # 키워드 다양성 분석
        for query_type, result in relevance_results.items():
            all_keywords = result["extracted_keywords"]["all_keywords"]
            base_keywords = result["extracted_keywords"]["base_keywords"]
            contextual_keywords = result["extracted_keywords"]["contextual_keywords"]
            semantic_keywords = result["extracted_keywords"]["semantic_keywords"]
            
            # 다양성 메트릭
            total_keywords = len(all_keywords)
            unique_keywords = len(set(all_keywords))
            diversity_ratio = unique_keywords / total_keywords if total_keywords > 0 else 0
            
            # 확장 메트릭
            expansion_ratio = total_keywords / len(base_keywords) if base_keywords else 0
            contextual_ratio = len(contextual_keywords) / len(base_keywords) if base_keywords else 0
            semantic_ratio = len(semantic_keywords) / len(base_keywords) if base_keywords else 0
            
            diversity_analysis["keyword_diversity"][query_type] = {
                "total_keywords": total_keywords,
                "unique_keywords": unique_keywords,
                "diversity_ratio": diversity_ratio,
                "expansion_ratio": expansion_ratio,
                "contextual_ratio": contextual_ratio,
                "semantic_ratio": semantic_ratio
            }
        
        # 도메인별 커버리지
        domain_stats = {}
        for query_type, result in relevance_results.items():
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "기타")
            
            if domain not in domain_stats:
                domain_stats[domain] = {
                    "total_realistic_keywords": 0,
                    "matched_realistic_keywords": 0,
                    "total_legal_concepts": 0,
                    "matched_legal_concepts": 0,
                    "queries": 0
                }
            
            domain_stats[domain]["total_realistic_keywords"] += len(result["realistic_keywords"])
            domain_stats[domain]["matched_realistic_keywords"] += len(result["matched_keywords"]["realistic_matched"])
            domain_stats[domain]["total_legal_concepts"] += len(result["legal_concepts"])
            domain_stats[domain]["matched_legal_concepts"] += len(result["matched_keywords"]["legal_concept_matched"])
            domain_stats[domain]["queries"] += 1
        
        # 도메인별 커버리지 계산
        for domain, stats in domain_stats.items():
            realistic_coverage = stats["matched_realistic_keywords"] / stats["total_realistic_keywords"] if stats["total_realistic_keywords"] > 0 else 0
            concept_coverage = stats["matched_legal_concepts"] / stats["total_legal_concepts"] if stats["total_legal_concepts"] > 0 else 0
            
            diversity_analysis["domain_coverage"][domain] = {
                "realistic_coverage": realistic_coverage,
                "concept_coverage": concept_coverage,
                "overall_coverage": (realistic_coverage + concept_coverage) / 2,
                "query_count": stats["queries"]
            }
        
        # 전체 메트릭 계산
        avg_diversity = sum(d["diversity_ratio"] for d in diversity_analysis["keyword_diversity"].values()) / len(diversity_analysis["keyword_diversity"])
        avg_expansion = sum(d["expansion_ratio"] for d in diversity_analysis["keyword_diversity"].values()) / len(diversity_analysis["keyword_diversity"])
        avg_coverage = sum(c["overall_coverage"] for c in diversity_analysis["domain_coverage"].values()) / len(diversity_analysis["domain_coverage"])
        
        diversity_analysis["overall_metrics"] = {
            "average_diversity": avg_diversity,
            "average_expansion": avg_expansion,
            "average_coverage": avg_coverage,
            "overall_score": (avg_diversity + avg_expansion + avg_coverage) / 3
        }
        
        logger.info("키워드 다양성 및 커버리지 테스트 완료")
        return diversity_analysis
    
    def generate_realistic_test_report(self, relevance_results: Dict[str, Any], diversity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """현실적인 테스트 보고서 생성"""
        logger.info("현실적인 테스트 보고서 생성 중")
        
        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "test_type": "realistic_keyword_relevance_test"
            },
            "relevance_results": relevance_results,
            "diversity_analysis": diversity_analysis,
            "performance_summary": {
                "average_realistic_relevance": sum(r["relevance_metrics"]["realistic_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_legal_concept_relevance": sum(r["relevance_metrics"]["legal_concept_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_overall_relevance": sum(r["relevance_metrics"]["overall_relevance"] for r in relevance_results.values()) / len(relevance_results),
                "average_quality_score": sum(r["relevance_metrics"]["quality_score"] for r in relevance_results.values()) / len(relevance_results),
                "average_processing_time": sum(r["processing_time"] for r in relevance_results.values()) / len(relevance_results)
            },
            "recommendations": self._generate_realistic_recommendations(relevance_results, diversity_analysis)
        }
        
        logger.info("현실적인 테스트 보고서 생성 완료")
        return report
    
    def _generate_realistic_recommendations(self, relevance_results: Dict[str, Any], diversity_analysis: Dict[str, Any]) -> List[str]:
        """현실적인 권장사항 생성"""
        recommendations = []
        
        # 관련성 기반 권장사항
        avg_relevance = sum(r["relevance_metrics"]["overall_relevance"] for r in relevance_results.values()) / len(relevance_results)
        
        if avg_relevance < 0.3:
            recommendations.append("키워드 관련성이 매우 낮습니다. 실제 사용자 질문에 맞는 키워드 매핑을 개선하세요.")
        elif avg_relevance < 0.5:
            recommendations.append("키워드 관련성이 낮습니다. 현실적인 용어와 법률 개념의 매핑을 강화하세요.")
        elif avg_relevance < 0.7:
            recommendations.append("키워드 관련성이 보통입니다. 추가 개선을 통해 더 향상시킬 수 있습니다.")
        else:
            recommendations.append("키워드 관련성이 양호합니다. 현재 설정을 유지하세요.")
        
        # 품질 점수 기반 권장사항
        avg_quality = sum(r["relevance_metrics"]["quality_score"] for r in relevance_results.values()) / len(relevance_results)
        
        if avg_quality < 0.3:
            recommendations.append("키워드 품질이 매우 낮습니다. 법률 관련성과 현실성을 모두 고려한 키워드 추출을 개선하세요.")
        elif avg_quality < 0.5:
            recommendations.append("키워드 품질을 개선하기 위해 법률 용어와 현실적 용어의 균형을 맞추세요.")
        
        # 다양성 기반 권장사항
        overall_score = diversity_analysis["overall_metrics"]["overall_score"]
        
        if overall_score < 0.3:
            recommendations.append("키워드 다양성과 커버리지가 매우 낮습니다. 도메인별 용어 확장이 필요합니다.")
        elif overall_score < 0.5:
            recommendations.append("키워드 다양성을 개선하기 위해 의미적 관계와 컨텍스트 매핑을 확장하세요.")
        
        # 도메인별 권장사항
        for domain, coverage in diversity_analysis["domain_coverage"].items():
            if coverage["overall_coverage"] < 0.3:
                recommendations.append(f"{domain} 도메인의 키워드 커버리지가 낮습니다. 해당 도메인 용어를 확장하세요.")
        
        return recommendations
    
    def save_test_results(self, test_report: Dict[str, Any]):
        """테스트 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 테스트 보고서 저장
        report_file = os.path.join(self.output_dir, "realistic_query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"테스트 결과 저장 완료: {self.output_dir}")
    
    def run_realistic_test(self):
        """현실적인 테스트 실행"""
        logger.info("현실적인 법률 질의 테스트 시작")
        
        try:
            # 키워드 관련성 테스트
            relevance_results = self.test_keyword_relevance()
            
            # 키워드 다양성 및 커버리지 테스트
            diversity_analysis = self.test_keyword_diversity_and_coverage(relevance_results)
            
            # 현실적인 테스트 보고서 생성
            test_report = self.generate_realistic_test_report(relevance_results, diversity_analysis)
            
            # 결과 저장
            self.save_test_results(test_report)
            
            logger.info("현실적인 법률 질의 테스트 완료")
            
            # 결과 요약 출력
            print(f"\n=== 현실적인 법률 질의 테스트 결과 요약 ===")
            print(f"총 테스트 질의 수: {test_report['test_summary']['total_queries']}")
            print(f"평균 현실적 관련성: {test_report['performance_summary']['average_realistic_relevance']:.3f}")
            print(f"평균 법률 개념 관련성: {test_report['performance_summary']['average_legal_concept_relevance']:.3f}")
            print(f"평균 전체 관련성: {test_report['performance_summary']['average_overall_relevance']:.3f}")
            print(f"평균 품질 점수: {test_report['performance_summary']['average_quality_score']:.3f}")
            print(f"평균 처리 시간: {test_report['performance_summary']['average_processing_time']:.4f}초")
            
            print(f"\n=== 도메인별 커버리지 ===")
            for domain, coverage in test_report['diversity_analysis']['domain_coverage'].items():
                print(f"{domain}: {coverage['overall_coverage']:.3f}")
            
            print(f"\n=== 키워드 다양성 ===")
            diversity_metrics = test_report['diversity_analysis']['overall_metrics']
            print(f"평균 다양성: {diversity_metrics['average_diversity']:.3f}")
            print(f"평균 확장률: {diversity_metrics['average_expansion']:.2f}")
            print(f"평균 커버리지: {diversity_metrics['average_coverage']:.3f}")
            print(f"전체 점수: {diversity_metrics['overall_score']:.3f}")
            
            print(f"\n=== 개선 권장사항 ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")
            
            # 상세 결과 예시 출력
            print(f"\n=== 상세 결과 예시 (첫 번째 질의) ===")
            first_query = list(relevance_results.values())[0]
            print(f"질문: {first_query['question']}")
            print(f"현실적 관련성: {first_query['relevance_metrics']['realistic_relevance']:.3f}")
            print(f"법률 개념 관련성: {first_query['relevance_metrics']['legal_concept_relevance']:.3f}")
            print(f"전체 관련성: {first_query['relevance_metrics']['overall_relevance']:.3f}")
            print(f"품질 점수: {first_query['relevance_metrics']['quality_score']:.3f}")
            print(f"매칭된 현실적 키워드: {first_query['matched_keywords']['realistic_matched']}")
            print(f"매칭된 법률 개념: {first_query['matched_keywords']['legal_concept_matched']}")
            
        except Exception as e:
            logger.error(f"현실적인 테스트 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    tester = RealisticQueryTester()
    tester.run_realistic_test()

if __name__ == "__main__":
    main()
