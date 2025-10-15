#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전처리된 데이터 검증 스크립트

전처리된 데이터의 품질을 검증하고 보고서를 생성합니다.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from source.data.data_processor import LegalDataProcessor

class ProcessedDataValidator:
    """전처리된 데이터 검증 클래스"""
    
    def __init__(self):
        self.processor = LegalDataProcessor()
        self.logger = logging.getLogger(__name__)
        
        # 품질 지표 설정
        self.quality_metrics = {
            "completeness": 0.95,      # 완성도 95% 이상
            "accuracy": 0.98,          # 정확도 98% 이상
            "consistency": 0.90,       # 일관성 90% 이상
            "term_normalization": 0.90 # 용어 정규화 90% 이상
        }
    
    def validate_all_data(self, processed_dir: str = "data/processed") -> Dict[str, Any]:
        """모든 전처리된 데이터 검증"""
        self.logger.info("전처리된 데이터 검증 시작")
        
        validation_results = {
            "overall": {
                "total_documents": 0,
                "valid_documents": 0,
                "invalid_documents": 0,
                "validation_passed": False,
                "quality_score": 0.0
            },
            "by_type": {},
            "issues": [],
            "recommendations": []
        }
        
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            self.logger.error(f"전처리된 데이터 디렉토리가 존재하지 않음: {processed_dir}")
            return validation_results
        
        # 데이터 유형별 검증
        data_types = ["laws", "precedents", "constitutional_decisions", "legal_interpretations", "legal_terms"]
        
        for data_type in data_types:
            type_result = self.validate_data_type(data_type, processed_path)
            validation_results["by_type"][data_type] = type_result
            validation_results["overall"]["total_documents"] += type_result["total_documents"]
            validation_results["overall"]["valid_documents"] += type_result["valid_documents"]
            validation_results["overall"]["invalid_documents"] += type_result["invalid_documents"]
        
        # 전체 품질 점수 계산
        if validation_results["overall"]["total_documents"] > 0:
            # 유효성 기반 품질 점수
            validity_score = validation_results["overall"]["valid_documents"] / validation_results["overall"]["total_documents"]
            
            # 데이터 유형별 품질 지표 평균
            quality_scores = []
            for data_type, type_result in validation_results["by_type"].items():
                if type_result.get("quality_score"):
                    quality_scores.append(type_result["quality_score"])
            
            # 전체 품질 점수 (유효성 70% + 품질지표 30%)
            if quality_scores:
                avg_quality_score = sum(quality_scores) / len(quality_scores)
                validation_results["overall"]["quality_score"] = validity_score * 0.7 + avg_quality_score * 0.3
            else:
                validation_results["overall"]["quality_score"] = validity_score
            
            validation_results["overall"]["validation_passed"] = (
                validation_results["overall"]["quality_score"] >= 0.8
            )
        
        # 권장사항 생성
        validation_results["recommendations"] = self.generate_recommendations(validation_results)
        
        # 검증 결과 저장
        self.save_validation_report(validation_results)
        
        return validation_results
    
    def validate_data_type(self, data_type: str, processed_path: Path) -> Dict[str, Any]:
        """특정 데이터 유형 검증"""
        self.logger.info(f"{data_type} 데이터 검증 시작")
        
        data_dir = processed_path / data_type
        if not data_dir.exists():
            return {
                "total_documents": 0,
                "valid_documents": 0,
                "invalid_documents": 0,
                "validation_passed": False,
                "issues": [f"{data_type} 디렉토리가 존재하지 않음"],
                "quality_metrics": {}
            }
        
        json_files = list(data_dir.glob("*.json"))
        total_documents = 0
        valid_documents = 0
        invalid_documents = 0
        issues = []
        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "term_normalization": 0.0
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
                
                for doc in documents:
                    total_documents += 1
                    
                    # 문서 유효성 검사
                    is_valid, doc_issues = self.processor.validate_document(doc)
                    
                    if is_valid:
                        valid_documents += 1
                    else:
                        invalid_documents += 1
                        issues.extend(doc_issues)
                    
                    # 품질 지표 계산
                    doc_metrics = self.calculate_quality_metrics(doc)
                    for metric, value in doc_metrics.items():
                        if metric in quality_metrics:
                            quality_metrics[metric] += value
                
            except Exception as e:
                self.logger.error(f"검증 중 오류 {json_file}: {e}")
                issues.append(f"파일 읽기 오류: {json_file}")
        
        # 평균 품질 지표 계산
        if total_documents > 0:
            for metric in quality_metrics:
                quality_metrics[metric] = quality_metrics[metric] / total_documents
        
        # 전체 품질 점수 계산
        quality_score = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.0
        
        validation_passed = (
            total_documents > 0 and 
            (valid_documents / total_documents) >= 0.9
        )
        
        return {
            "total_documents": total_documents,
            "valid_documents": valid_documents,
            "invalid_documents": invalid_documents,
            "validation_passed": validation_passed,
            "issues": issues,
            "quality_metrics": quality_metrics,
            "quality_score": quality_score
        }
    
    def calculate_quality_metrics(self, document: Dict[str, Any]) -> Dict[str, float]:
        """문서의 품질 지표 계산"""
        metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "term_normalization": 0.0
        }
        
        # 완성도 계산 - 법령 데이터에 맞는 필드들
        required_fields = ["id", "law_name", "articles"]
        present_fields = sum(1 for field in required_fields if field in document and document[field])
        
        # 추가 필드들 (있으면 보너스)
        bonus_fields = ["chunks", "article_chunks", "cleaned_content", "entities"]
        bonus_count = sum(1 for field in bonus_fields if field in document and document[field])
        
        # 완성도 = 필수 필드 (70%) + 보너스 필드 (30%)
        metrics["completeness"] = (present_fields / len(required_fields)) * 0.7 + (bonus_count / len(bonus_fields)) * 0.3
        
        # 일관성 계산 - 조문과 청크의 일관성
        if "articles" in document and document["articles"]:
            articles_count = len(document["articles"])
            chunks_count = len(document.get("chunks", []))
            article_chunks_count = len(document.get("article_chunks", []))
            
            # 조문이 있고 청크가 생성되었으면 일관성 높음
            if articles_count > 0 and (chunks_count > 0 or article_chunks_count > 0):
                metrics["consistency"] = 1.0
            else:
                metrics["consistency"] = 0.5
        else:
            # 조문이 없으면 기본 일관성
            metrics["consistency"] = 0.3
        
        # 용어 정규화 계산 - 법령 데이터 특성 고려
        if "entities" in document and document["entities"]:
            # 엔티티가 있으면 정규화 진행된 것으로 간주
            metrics["term_normalization"] = 0.8
        elif "cleaned_content" in document and document["cleaned_content"]:
            # 정리된 내용이 있으면 기본 정규화
            metrics["term_normalization"] = 0.6
        else:
            # 기본값
            metrics["term_normalization"] = 0.4
        
        return metrics
    
    def generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """검증 결과 기반 권장사항 생성"""
        recommendations = []
        
        overall = validation_results["overall"]
        
        if overall["quality_score"] < 0.9:
            recommendations.append("전체 데이터 품질이 목표치(90%) 미만입니다. 데이터 전처리 과정을 재검토하세요.")
        
        if overall["invalid_documents"] > 0:
            recommendations.append(f"{overall['invalid_documents']}개의 유효하지 않은 문서가 있습니다. 해당 문서들을 수정하세요.")
        
        for data_type, type_result in validation_results["by_type"].items():
            if not type_result["validation_passed"]:
                recommendations.append(f"{data_type} 데이터의 품질이 기준에 미달합니다. 추가 전처리가 필요합니다.")
            
            # quality_metrics가 존재하는 경우에만 체크
            if "quality_metrics" in type_result:
                if type_result["quality_metrics"].get("completeness", 1.0) < 0.95:
                    recommendations.append(f"{data_type} 데이터의 완성도가 낮습니다. 필수 필드가 누락되었을 수 있습니다.")
                
                if type_result["quality_metrics"].get("term_normalization", 1.0) < 0.9:
                    recommendations.append(f"{data_type} 데이터의 용어 정규화가 부족합니다. 법률 용어 사전을 업데이트하세요.")
        
        return recommendations
    
    def save_validation_report(self, validation_results: Dict[str, Any]):
        """검증 보고서 저장"""
        report_file = Path("data/processed") / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"검증 보고서 저장 완료: {report_file}")
    
    def print_validation_summary(self, validation_results: Dict[str, Any]):
        """검증 결과 요약 출력"""
        overall = validation_results["overall"]
        
        print("\n=== 데이터 검증 결과 ===")
        print(f"총 문서 수: {overall['total_documents']}")
        print(f"유효한 문서: {overall['valid_documents']}")
        print(f"유효하지 않은 문서: {overall['invalid_documents']}")
        print(f"품질 점수: {overall.get('quality_score', 0.0):.2%}")
        print(f"검증 통과: {'예' if overall['validation_passed'] else '아니오'}")
        
        print("\n=== 데이터 유형별 결과 ===")
        for data_type, type_result in validation_results["by_type"].items():
            print(f"{data_type}:")
            print(f"  - 총 문서: {type_result['total_documents']}")
            print(f"  - 유효한 문서: {type_result['valid_documents']}")
            print(f"  - 검증 통과: {'예' if type_result['validation_passed'] else '아니오'}")
            if type_result['issues']:
                print(f"  - 이슈 수: {len(type_result['issues'])}")
        
        if validation_results["recommendations"]:
            print("\n=== 권장사항 ===")
            for i, rec in enumerate(validation_results["recommendations"], 1):
                print(f"{i}. {rec}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="전처리된 데이터 검증")
    parser.add_argument("--processed-dir", default="data/processed",
                       help="전처리된 데이터 디렉토리")
    parser.add_argument("--data-type", 
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       default="all",
                       help="검증할 데이터 유형")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = ProcessedDataValidator()
    
    if args.data_type == "all":
        validation_results = validator.validate_all_data(args.processed_dir)
    else:
        # 특정 데이터 유형만 검증
        processed_path = Path(args.processed_dir)
        type_result = validator.validate_data_type(args.data_type, processed_path)
        validation_results = {
            "overall": type_result,
            "by_type": {args.data_type: type_result},
            "issues": type_result.get("issues", []),
            "recommendations": []
        }
    
    validator.print_validation_summary(validation_results)

if __name__ == "__main__":
    main()
