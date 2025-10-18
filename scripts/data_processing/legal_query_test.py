#!/usr/bin/env python3
"""
법률 질의 테스트 시스템
확장된 키워드 매핑 시스템을 실제 법률 질문으로 테스트합니다.
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
from source.services.langgraph.legal_workflow_enhanced import EnhancedLegalQuestionWorkflow
from source.services.langgraph.workflow_service import LangGraphWorkflowService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/legal_query_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalQueryTester:
    """법률 질의 테스트기"""
    
    def __init__(self):
        self.test_queries = self._initialize_test_queries()
        self.output_dir = "data/extracted_terms/query_test"
        
        # 테스트 결과 저장
        self.test_results = {
            "keyword_mapping_tests": {},
            "workflow_tests": {},
            "performance_metrics": {},
            "quality_assessment": {}
        }
    
    def _initialize_test_queries(self) -> List[Dict[str, str]]:
        """테스트 질의 초기화"""
        return [
            {
                "question": "계약서에서 위약금 조항이 너무 높게 설정되어 있는데, 법적으로 문제가 될까요?",
                "query_type": "contract_review",
                "domain": "민사법",
                "expected_keywords": ["계약서", "위약금", "조항", "법적", "문제", "민법", "계약법", "손해배상"]
            },
            {
                "question": "교통사고로 인한 손해배상 청구 시 필요한 증거자료는 무엇인가요?",
                "query_type": "damage_compensation",
                "domain": "민사법",
                "expected_keywords": ["교통사고", "손해배상", "청구", "증거자료", "불법행위", "과실", "인과관계"]
            },
            {
                "question": "이혼 소송에서 자녀 양육권을 결정하는 기준은 무엇인가요?",
                "query_type": "divorce_proceedings",
                "domain": "가족법",
                "expected_keywords": ["이혼", "소송", "자녀", "양육권", "결정", "기준", "가정법원", "가족법"]
            },
            {
                "question": "부동산 매매 계약 시 등기 이전 절차와 필요한 서류는 무엇인가요?",
                "query_type": "real_estate_transaction",
                "domain": "부동산법",
                "expected_keywords": ["부동산", "매매", "계약", "등기", "이전", "절차", "서류", "등기부등본"]
            },
            {
                "question": "특허 출원 시 발명의 신규성과 진보성을 어떻게 입증해야 하나요?",
                "query_type": "patent_application",
                "domain": "특허법",
                "expected_keywords": ["특허", "출원", "발명", "신규성", "진보성", "입증", "특허청", "특허법"]
            },
            {
                "question": "근로자가 부당해고를 당했을 때 구제 절차는 어떻게 되나요?",
                "query_type": "employment_termination",
                "domain": "노동법",
                "expected_keywords": ["근로자", "부당해고", "구제", "절차", "노동위원회", "근로기준법", "해고"]
            },
            {
                "question": "주식회사 설립 시 필요한 자본금과 등기 절차는 무엇인가요?",
                "query_type": "company_establishment",
                "domain": "상사법",
                "expected_keywords": ["주식회사", "설립", "자본금", "등기", "절차", "상법", "회사법", "주주"]
            },
            {
                "question": "형사 사건에서 변호사 선임권과 변호사 비용은 어떻게 되나요?",
                "query_type": "criminal_defense",
                "domain": "형사법",
                "expected_keywords": ["형사", "사건", "변호사", "선임권", "비용", "피고", "형사소송법", "국선변호"]
            },
            {
                "question": "행정처분에 대한 이의신청과 행정소송의 차이점은 무엇인가요?",
                "query_type": "administrative_appeal",
                "domain": "행정법",
                "expected_keywords": ["행정처분", "이의신청", "행정소송", "차이점", "행정법", "허가", "승인"]
            },
            {
                "question": "상속 포기와 한정승인 중 어떤 것을 선택해야 할까요?",
                "query_type": "inheritance_renunciation",
                "domain": "가족법",
                "expected_keywords": ["상속", "포기", "한정승인", "선택", "상속인", "상속분", "상속법"]
            }
        ]
    
    def test_keyword_mapping(self) -> Dict[str, Any]:
        """키워드 매핑 테스트"""
        logger.info("키워드 매핑 테스트 시작")
        
        enhanced_mapper = EnhancedKeywordMapper()
        results = {}
        
        for i, query in enumerate(self.test_queries):
            question = query["question"]
            query_type = query["query_type"]
            expected_keywords = query["expected_keywords"]
            
            logger.info(f"테스트 {i+1}/{len(self.test_queries)}: {query_type}")
            
            start_time = time.time()
            
            # 종합적인 키워드 매핑
            comprehensive_result = enhanced_mapper.get_comprehensive_keyword_mapping(question, query_type)
            
            end_time = time.time()
            
            # 예상 키워드와의 매칭률 계산
            all_keywords = comprehensive_result.get("all_keywords", [])
            matched_keywords = [kw for kw in expected_keywords if kw in all_keywords]
            match_rate = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0
            
            results[query_type] = {
                "question": question,
                "expected_keywords": expected_keywords,
                "extracted_keywords": all_keywords,
                "matched_keywords": matched_keywords,
                "match_rate": match_rate,
                "processing_time": end_time - start_time,
                "comprehensive_result": comprehensive_result
            }
        
        logger.info("키워드 매핑 테스트 완료")
        return results
    
    def test_workflow_integration(self) -> Dict[str, Any]:
        """워크플로우 통합 테스트"""
        logger.info("워크플로우 통합 테스트 시작")
        
        try:
            # 워크플로우 서비스 초기화
            workflow_service = LangGraphWorkflowService()
            
            results = {}
            
            for i, query in enumerate(self.test_queries[:3]):  # 처음 3개만 테스트 (시간 절약)
                question = query["question"]
                query_type = query["query_type"]
                
                logger.info(f"워크플로우 테스트 {i+1}/3: {query_type}")
                
                start_time = time.time()
                
                try:
                    # 워크플로우 실행
                    response = workflow_service.process_question(question, query_type)
                    
                    end_time = time.time()
                    
                    results[query_type] = {
                        "question": question,
                        "response": response,
                        "processing_time": end_time - start_time,
                        "success": True
                    }
                    
                except Exception as e:
                    logger.error(f"워크플로우 실행 오류 ({query_type}): {e}")
                    results[query_type] = {
                        "question": question,
                        "error": str(e),
                        "success": False
                    }
            
            logger.info("워크플로우 통합 테스트 완료")
            return results
            
        except Exception as e:
            logger.error(f"워크플로우 서비스 초기화 오류: {e}")
            return {"error": str(e)}
    
    def analyze_keyword_quality(self, mapping_results: Dict[str, Any]) -> Dict[str, Any]:
        """키워드 품질 분석"""
        logger.info("키워드 품질 분석 시작")
        
        quality_metrics = {
            "overall_match_rate": 0,
            "domain_coverage": {},
            "keyword_diversity": {},
            "processing_efficiency": {},
            "recommendations": []
        }
        
        total_match_rate = 0
        total_queries = len(mapping_results)
        
        for query_type, result in mapping_results.items():
            match_rate = result["match_rate"]
            total_match_rate += match_rate
            
            # 도메인별 커버리지
            domain = next((q["domain"] for q in self.test_queries if q["query_type"] == query_type), "기타")
            if domain not in quality_metrics["domain_coverage"]:
                quality_metrics["domain_coverage"][domain] = []
            quality_metrics["domain_coverage"][domain].append(match_rate)
            
            # 키워드 다양성
            extracted_count = len(result["extracted_keywords"])
            expected_count = len(result["expected_keywords"])
            diversity_ratio = extracted_count / expected_count if expected_count > 0 else 0
            
            quality_metrics["keyword_diversity"][query_type] = {
                "extracted_count": extracted_count,
                "expected_count": expected_count,
                "diversity_ratio": diversity_ratio
            }
            
            # 처리 효율성
            quality_metrics["processing_efficiency"][query_type] = {
                "processing_time": result["processing_time"],
                "keywords_per_second": extracted_count / result["processing_time"] if result["processing_time"] > 0 else 0
            }
        
        # 전체 매칭률
        quality_metrics["overall_match_rate"] = total_match_rate / total_queries
        
        # 도메인별 평균 매칭률
        for domain, rates in quality_metrics["domain_coverage"].items():
            quality_metrics["domain_coverage"][domain] = sum(rates) / len(rates)
        
        # 개선 권장사항 생성
        if quality_metrics["overall_match_rate"] < 0.5:
            quality_metrics["recommendations"].append("전체 키워드 매칭률이 낮습니다. 키워드 매핑 전략을 재검토하세요.")
        
        if quality_metrics["overall_match_rate"] > 0.8:
            quality_metrics["recommendations"].append("키워드 매칭률이 우수합니다. 현재 설정을 유지하세요.")
        
        # 도메인별 권장사항
        for domain, rate in quality_metrics["domain_coverage"].items():
            if rate < 0.4:
                quality_metrics["recommendations"].append(f"{domain} 도메인의 키워드 매칭률이 낮습니다. 해당 도메인 용어를 확장하세요.")
        
        logger.info("키워드 품질 분석 완료")
        return quality_metrics
    
    def generate_test_report(self, mapping_results: Dict[str, Any], workflow_results: Dict[str, Any], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """테스트 보고서 생성"""
        logger.info("테스트 보고서 생성 중")
        
        report = {
            "test_summary": {
                "test_date": datetime.now().isoformat(),
                "total_queries": len(self.test_queries),
                "successful_mappings": len([r for r in mapping_results.values() if r.get("match_rate", 0) > 0]),
                "successful_workflows": len([r for r in workflow_results.values() if r.get("success", False)])
            },
            "keyword_mapping_results": {
                "average_match_rate": quality_metrics["overall_match_rate"],
                "domain_performance": quality_metrics["domain_coverage"],
                "processing_efficiency": quality_metrics["processing_efficiency"]
            },
            "workflow_integration_results": workflow_results,
            "quality_assessment": quality_metrics,
            "detailed_results": mapping_results,
            "recommendations": quality_metrics["recommendations"]
        }
        
        logger.info("테스트 보고서 생성 완료")
        return report
    
    def save_test_results(self, test_report: Dict[str, Any]):
        """테스트 결과 저장"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 테스트 보고서 저장
        report_file = os.path.join(self.output_dir, "query_test_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"테스트 결과 저장 완료: {self.output_dir}")
    
    def run_query_test(self):
        """질의 테스트 실행"""
        logger.info("법률 질의 테스트 시작")
        
        try:
            # 키워드 매핑 테스트
            mapping_results = self.test_keyword_mapping()
            
            # 워크플로우 통합 테스트
            workflow_results = self.test_workflow_integration()
            
            # 키워드 품질 분석
            quality_metrics = self.analyze_keyword_quality(mapping_results)
            
            # 테스트 보고서 생성
            test_report = self.generate_test_report(mapping_results, workflow_results, quality_metrics)
            
            # 결과 저장
            self.save_test_results(test_report)
            
            logger.info("법률 질의 테스트 완료")
            
            # 결과 요약 출력
            print(f"\n=== 법률 질의 테스트 결과 요약 ===")
            print(f"총 테스트 질의 수: {test_report['test_summary']['total_queries']}")
            print(f"성공적인 키워드 매핑: {test_report['test_summary']['successful_mappings']}")
            print(f"성공적인 워크플로우: {test_report['test_summary']['successful_workflows']}")
            print(f"평균 키워드 매칭률: {test_report['keyword_mapping_results']['average_match_rate']:.3f}")
            
            print(f"\n=== 도메인별 성능 ===")
            for domain, rate in test_report['keyword_mapping_results']['domain_performance'].items():
                print(f"{domain}: {rate:.3f}")
            
            print(f"\n=== 개선 권장사항 ===")
            for i, recommendation in enumerate(test_report['recommendations'], 1):
                print(f"{i}. {recommendation}")
            
            # 상세 결과 예시 출력
            print(f"\n=== 상세 결과 예시 (첫 번째 질의) ===")
            first_query = list(mapping_results.values())[0]
            print(f"질문: {first_query['question']}")
            print(f"예상 키워드: {first_query['expected_keywords']}")
            print(f"추출된 키워드: {first_query['extracted_keywords'][:10]}...")  # 처음 10개만
            print(f"매칭된 키워드: {first_query['matched_keywords']}")
            print(f"매칭률: {first_query['match_rate']:.3f}")
            
        except Exception as e:
            logger.error(f"질의 테스트 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    tester = LegalQueryTester()
    tester.run_query_test()

if __name__ == "__main__":
    main()
