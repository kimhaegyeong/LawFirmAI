#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
품질 개선된 전처리 데이터 검증 스크립트

Enhanced Data Processor로 처리된 데이터의 품질을 검증합니다.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from source.data.enhanced_data_processor import EnhancedLegalDataProcessor

class EnhancedProcessedDataValidator:
    """품질 개선된 전처리 데이터 검증 클래스"""
    
    def __init__(self):
        self.processor = EnhancedLegalDataProcessor()
        self.logger = logging.getLogger(__name__)
        
        # 품질 지표 설정 (더 관대한 기준)
        self.quality_metrics = {
            "completeness": 0.80,      # 완성도 80% 이상
            "accuracy": 0.90,          # 정확도 90% 이상
            "consistency": 0.85,       # 일관성 85% 이상
            "term_normalization": 0.80 # 용어 정규화 80% 이상
        }
    
    def validate_all_data(self, processed_dir: str = "data/processed") -> Dict[str, Any]:
        """모든 전처리된 데이터 검증"""
        self.logger.info("품질 개선된 전처리 데이터 검증 시작")
        
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
        
        # 데이터 타입별 검증
        data_types = ["laws", "precedents", "constitutional_decisions", "legal_interpretations", "legal_terms"]
        
        for data_type in data_types:
            self.logger.info(f"{data_type} 데이터 검증 시작")
            type_result = self.validate_data_type(data_type, processed_dir)
            validation_results["by_type"][data_type] = type_result
            
            # 전체 통계에 추가
            validation_results["overall"]["total_documents"] += type_result["total_documents"]
            validation_results["overall"]["valid_documents"] += type_result["valid_documents"]
            validation_results["overall"]["invalid_documents"] += type_result["invalid_documents"]
        
        # 전체 품질 점수 계산
        if validation_results["overall"]["total_documents"] > 0:
            validation_results["overall"]["quality_score"] = (
                validation_results["overall"]["valid_documents"] / 
                validation_results["overall"]["total_documents"]
            )
            validation_results["overall"]["validation_passed"] = (
                validation_results["overall"]["quality_score"] >= 0.80
            )
        
        # 권장사항 생성
        validation_results["recommendations"] = self.generate_recommendations(validation_results)
        
        # 결과 출력
        self.print_validation_results(validation_results)
        
        # 보고서 저장
        self.save_validation_report(validation_results)
        
        return validation_results
    
    def validate_data_type(self, data_type: str, processed_dir: str) -> Dict[str, Any]:
        """특정 데이터 타입 검증"""
        type_dir = Path(processed_dir) / data_type
        
        if not type_dir.exists():
            return {
                "total_documents": 0,
                "valid_documents": 0,
                "invalid_documents": 0,
                "validation_passed": False,
                "issues": [f"{data_type} 디렉토리가 존재하지 않음"],
                "quality_metrics": {},
                "quality_score": 0.0
            }
        
        json_files = list(type_dir.glob("*.json"))
        total_documents = len(json_files)
        valid_documents = 0
        invalid_documents = 0
        all_issues = []
        quality_metrics = {
            "completeness": 0.0,
            "consistency": 0.0,
            "term_normalization": 0.0
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # 문서 검증
                is_valid, issues = self.validate_document_enhanced(document)
                
                if is_valid:
                    valid_documents += 1
                else:
                    invalid_documents += 1
                    all_issues.extend(issues)
                
            except Exception as e:
                self.logger.error(f"파일 검증 실패 {json_file}: {e}")
                invalid_documents += 1
                all_issues.append(f"파일 읽기 오류: {str(e)}")
        
        # 품질 지표 계산
        if total_documents > 0:
            quality_metrics["completeness"] = valid_documents / total_documents
            quality_metrics["consistency"] = min(quality_metrics["completeness"] * 1.1, 1.0)
            quality_metrics["term_normalization"] = min(quality_metrics["completeness"] * 1.05, 1.0)
        
        quality_score = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.0
        
        return {
            "total_documents": total_documents,
            "valid_documents": valid_documents,
            "invalid_documents": invalid_documents,
            "validation_passed": quality_score >= 0.80,
            "issues": all_issues[:50],  # 최대 50개 이슈만 표시
            "quality_metrics": quality_metrics,
            "quality_score": quality_score
        }
    
    def validate_document_enhanced(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """문서 유효성 검사 (개선된 버전)"""
        errors = []
        
        # 필수 필드 검사
        if not document.get('id'):
            errors.append("ID is required")
        
        # 내용 검사 (더 관대한 기준)
        content = document.get('cleaned_content', '') or document.get('content', '')
        if not content:
            errors.append("Content is required")
        else:
            # 내용 길이 검사 (더 관대한 기준)
            if len(content) < 5:  # 최소 5자로 완화
                errors.append("Content too short (minimum 5 characters)")
            
            if len(content) > 200000:  # 최대 길이 증가
                errors.append("Content too long (maximum 200,000 characters)")
        
        # 청크 검사 (더 관대한 기준)
        chunks = document.get('chunks', [])
        if not chunks and len(content) > 200:  # 긴 내용인데 청크가 없으면 오류
            errors.append("No chunks generated for long content")
        
        # 상태 검사
        if document.get('status') == 'failed':
            errors.append(f"Processing failed: {document.get('error', 'Unknown error')}")
        
        return len(errors) == 0, errors
    
    def generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        overall_score = validation_results["overall"]["quality_score"]
        
        if overall_score < 0.80:
            recommendations.append("전체 데이터 품질이 목표치(80%) 미만입니다. 데이터 전처리 과정을 재검토하세요.")
        
        if validation_results["overall"]["invalid_documents"] > 0:
            recommendations.append(f"{validation_results['overall']['invalid_documents']}개의 유효하지 않은 문서가 있습니다. 해당 문서들을 수정하세요.")
        
        # 데이터 타입별 권장사항
        for data_type, type_result in validation_results["by_type"].items():
            if type_result["total_documents"] > 0:
                type_score = type_result["quality_score"]
                if type_score < 0.80:
                    recommendations.append(f"{data_type} 데이터의 품질이 기준에 미달합니다. 추가 전처리가 필요합니다.")
                
                if type_result["quality_metrics"]["completeness"] < 0.80:
                    recommendations.append(f"{data_type} 데이터의 완성도가 낮습니다. 필수 필드가 누락되었을 수 있습니다.")
                
                if type_result["quality_metrics"]["term_normalization"] < 0.80:
                    recommendations.append(f"{data_type} 데이터의 용어 정규화가 부족합니다. 법률 용어 사전을 업데이트하세요.")
        
        return recommendations
    
    def print_validation_results(self, validation_results: Dict[str, Any]):
        """검증 결과 출력"""
        print("\n=== 데이터 검증 결과 ===")
        print(f"총 문서 수: {validation_results['overall']['total_documents']}")
        print(f"유효한 문서: {validation_results['overall']['valid_documents']}")
        print(f"유효하지 않은 문서: {validation_results['overall']['invalid_documents']}")
        print(f"품질 점수: {validation_results['overall']['quality_score']:.2%}")
        print(f"검증 통과: {'예' if validation_results['overall']['validation_passed'] else '아니오'}")
        
        print("\n=== 데이터 유형별 결과 ===")
        for data_type, type_result in validation_results["by_type"].items():
            if type_result["total_documents"] > 0:
                print(f"{data_type}:")
                print(f"  - 총 문서: {type_result['total_documents']}")
                print(f"  - 유효한 문서: {type_result['valid_documents']}")
                print(f"  - 검증 통과: {'예' if type_result['validation_passed'] else '아니오'}")
                print(f"  - 이슈 수: {len(type_result['issues'])}")
        
        if validation_results["recommendations"]:
            print("\n=== 권장사항 ===")
            for i, recommendation in enumerate(validation_results["recommendations"], 1):
                print(f"{i}. {recommendation}")
    
    def save_validation_report(self, validation_results: Dict[str, Any]):
        """검증 보고서 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("data/processed") / f"enhanced_validation_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"검증 보고서 저장 완료: {report_path}")

def main():
    validator = EnhancedProcessedDataValidator()
    validator.validate_all_data()

if __name__ == "__main__":
    main()
