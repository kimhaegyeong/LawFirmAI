#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 품질 검증 스크립트

이 스크립트는 수집된 법률 데이터의 품질을 검증하고 보고서를 생성합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.data_processor import DataProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_quality_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """데이터 품질 검증 클래스"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.validation_results = {
            "laws": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "precedents": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "legal_terms": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "administrative_rules": {"total": 0, "valid": 0, "invalid": 0, "errors": []},
            "local_ordinances": {"total": 0, "valid": 0, "invalid": 0, "errors": []}
        }
    
    def validate_law_data(self, law_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """법령 데이터 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ["law_id", "law_name", "effective_date", "content"]
        for field in required_fields:
            if not law_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 법령명 형식 검증
        law_name = law_data.get("law_name", "")
        if law_name and not re.match(r'^[가-힣]+법$', law_name):
            if not re.match(r'^[가-힣]+법률$', law_name):
                errors.append(f"Invalid law name format: {law_name}")
        
        # 시행일자 형식 검증
        effective_date = law_data.get("effective_date", "")
        if effective_date and not re.match(r'^\d{4}\.\d{2}\.\d{2}$', effective_date):
            errors.append(f"Invalid effective date format: {effective_date}")
        
        # 내용 길이 검증
        content = law_data.get("content", "")
        if content and len(content) < 100:
            errors.append("Content too short (minimum 100 characters)")
        
        return len(errors) == 0, errors
    
    def validate_precedent_data(self, precedent_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """판례 데이터 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ["precedent_id", "case_name", "court", "decision_date"]
        for field in required_fields:
            if not precedent_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 사건번호 형식 검증
        precedent_id = precedent_data.get("precedent_id", "")
        if precedent_id and not re.match(r'^\d{4}[가-힣]\d{4}$', precedent_id):
            errors.append(f"Invalid precedent ID format: {precedent_id}")
        
        # 법원명 검증
        court = precedent_data.get("court", "")
        valid_courts = ["대법원", "고등법원", "지방법원", "가정법원", "행정법원", "특허법원", "수원지방법원", "서울고등법원"]
        if court and not any(valid_court in court for valid_court in valid_courts):
            errors.append(f"Invalid court name: {court}")
        
        # 선고일자 형식 검증
        decision_date = precedent_data.get("decision_date", "")
        if decision_date and not re.match(r'^\d{4}\.\d{2}\.\d{2}$', decision_date):
            errors.append(f"Invalid decision date format: {decision_date}")
        
        return len(errors) == 0, errors
    
    def validate_legal_term_data(self, term_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """법령용어 데이터 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ["term_id", "term_name", "definition"]
        for field in required_fields:
            if not term_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 용어명 길이 검증
        term_name = term_data.get("term_name", "")
        if term_name and len(term_name) < 2:
            errors.append("Term name too short (minimum 2 characters)")
        
        # 정의 길이 검증
        definition = term_data.get("definition", "")
        if definition and len(definition) < 10:
            errors.append("Definition too short (minimum 10 characters)")
        
        return len(errors) == 0, errors
    
    def validate_administrative_rule_data(self, rule_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """행정규칙 데이터 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ["rule_id", "rule_name", "issuing_agency", "content"]
        for field in required_fields:
            if not rule_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 발령기관 검증
        issuing_agency = rule_data.get("issuing_agency", "")
        if issuing_agency and len(issuing_agency) < 3:
            errors.append("Issuing agency name too short")
        
        return len(errors) == 0, errors
    
    def validate_local_ordinance_data(self, ordinance_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """자치법규 데이터 검증"""
        errors = []
        
        # 필수 필드 검증
        required_fields = ["ordinance_id", "ordinance_name", "issuing_authority", "content"]
        for field in required_fields:
            if not ordinance_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 발령기관 검증 (지자체명 포함)
        issuing_authority = ordinance_data.get("issuing_authority", "")
        if issuing_authority and not any(keyword in issuing_authority for keyword in ["시", "도", "구", "군", "특별시", "광역시"]):
            errors.append(f"Invalid issuing authority format: {issuing_authority}")
        
        return len(errors) == 0, errors
    
    def validate_data_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """데이터 파일 검증"""
        logger.info(f"검증 중: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            validation_result = {
                "file_path": str(file_path),
                "data_type": data_type,
                "total_items": len(data),
                "valid_items": 0,
                "invalid_items": 0,
                "errors": []
            }
            
            for i, item in enumerate(data):
                is_valid, errors = self._validate_item(item, data_type)
                
                if is_valid:
                    validation_result["valid_items"] += 1
                else:
                    validation_result["invalid_items"] += 1
                    validation_result["errors"].extend([f"Item {i}: {error}" for error in errors])
            
            # 전체 통계 업데이트
            self.validation_results[data_type]["total"] += validation_result["total_items"]
            self.validation_results[data_type]["valid"] += validation_result["valid_items"]
            self.validation_results[data_type]["invalid"] += validation_result["invalid_items"]
            self.validation_results[data_type]["errors"].extend(validation_result["errors"])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"파일 검증 중 오류 발생 {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "data_type": data_type,
                "error": str(e),
                "total_items": 0,
                "valid_items": 0,
                "invalid_items": 0
            }
    
    def _validate_item(self, item: Dict[str, Any], data_type: str) -> Tuple[bool, List[str]]:
        """개별 데이터 항목 검증"""
        if data_type == "laws":
            return self.validate_law_data(item)
        elif data_type == "precedents":
            return self.validate_precedent_data(item)
        elif data_type == "legal_terms":
            return self.validate_legal_term_data(item)
        elif data_type == "administrative_rules":
            return self.validate_administrative_rule_data(item)
        elif data_type == "local_ordinances":
            return self.validate_local_ordinance_data(item)
        else:
            return False, [f"Unknown data type: {data_type}"]
    
    def validate_all_data(self) -> Dict[str, Any]:
        """모든 데이터 검증"""
        logger.info("전체 데이터 품질 검증 시작")
        
        data_directories = {
            "laws": Path("data/processed/laws"),
            "precedents": Path("data/processed/precedents"),
            "legal_terms": Path("data/processed/legal_terms"),
            "administrative_rules": Path("data/processed/administrative_rules"),
            "local_ordinances": Path("data/processed/local_ordinances")
        }
        
        validation_results = {}
        
        for data_type, data_dir in data_directories.items():
            if not data_dir.exists():
                logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
                continue
            
            validation_results[data_type] = []
            
            # JSON 파일 검색
            for json_file in data_dir.glob("*.json"):
                result = self.validate_data_file(json_file, data_type)
                validation_results[data_type].append(result)
        
        return validation_results
    
    def generate_quality_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """품질 검증 보고서 생성"""
        report = {
            "validation_date": datetime.now().isoformat(),
            "overall_summary": {},
            "detailed_results": validation_results,
            "quality_metrics": {}
        }
        
        # 전체 요약 계산
        total_items = sum(self.validation_results[data_type]["total"] for data_type in self.validation_results)
        total_valid = sum(self.validation_results[data_type]["valid"] for data_type in self.validation_results)
        total_invalid = sum(self.validation_results[data_type]["invalid"] for data_type in self.validation_results)
        
        report["overall_summary"] = {
            "total_items": total_items,
            "valid_items": total_valid,
            "invalid_items": total_invalid,
            "quality_score": (total_valid / total_items * 100) if total_items > 0 else 0
        }
        
        # 데이터 유형별 품질 지표
        for data_type, stats in self.validation_results.items():
            if stats["total"] > 0:
                report["quality_metrics"][data_type] = {
                    "completeness": (stats["valid"] / stats["total"]) * 100,
                    "error_rate": (stats["invalid"] / stats["total"]) * 100,
                    "total_errors": len(stats["errors"])
                }
        
        # 보고서 저장
        report_path = Path("docs/data_quality_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 데이터 품질 검증 보고서\n\n")
            f.write(f"**검증 일시**: {report['validation_date']}\n\n")
            
            f.write(f"## 전체 요약\n")
            f.write(f"- **총 데이터 항목**: {report['overall_summary']['total_items']:,}개\n")
            f.write(f"- **유효한 항목**: {report['overall_summary']['valid_items']:,}개\n")
            f.write(f"- **무효한 항목**: {report['overall_summary']['invalid_items']:,}개\n")
            f.write(f"- **품질 점수**: {report['overall_summary']['quality_score']:.1f}%\n\n")
            
            f.write(f"## 데이터 유형별 품질 지표\n")
            for data_type, metrics in report["quality_metrics"].items():
                f.write(f"### {data_type}\n")
                f.write(f"- **완성도**: {metrics['completeness']:.1f}%\n")
                f.write(f"- **오류율**: {metrics['error_rate']:.1f}%\n")
                f.write(f"- **총 오류 수**: {metrics['total_errors']}개\n\n")
            
            f.write(f"## 상세 오류 목록\n")
            for data_type, stats in self.validation_results.items():
                if stats["errors"]:
                    f.write(f"### {data_type}\n")
                    for error in stats["errors"][:10]:  # 상위 10개 오류만 표시
                        f.write(f"- {error}\n")
                    if len(stats["errors"]) > 10:
                        f.write(f"- ... 및 {len(stats['errors']) - 10}개 추가 오류\n")
                    f.write("\n")
        
        logger.info(f"품질 검증 보고서 생성: {report_path}")
        return report


def main():
    """메인 실행 함수"""
    logger.info("데이터 품질 검증 스크립트 시작")
    
    try:
        # 검증기 초기화
        validator = DataQualityValidator()
        
        # 전체 데이터 검증
        validation_results = validator.validate_all_data()
        
        # 품질 보고서 생성
        quality_report = validator.generate_quality_report(validation_results)
        
        logger.info("데이터 품질 검증 완료")
        logger.info(f"전체 품질 점수: {quality_report['overall_summary']['quality_score']:.1f}%")
        logger.info(f"유효한 항목: {quality_report['overall_summary']['valid_items']:,}개")
        logger.info(f"무효한 항목: {quality_report['overall_summary']['invalid_items']:,}개")
        
    except Exception as e:
        logger.error(f"데이터 품질 검증 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
