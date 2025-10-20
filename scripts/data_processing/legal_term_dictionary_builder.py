# -*- coding: utf-8 -*-
"""
법률 용어 사전 구축 통합 빌더
전체 파이프라인을 실행하여 향상된 법률 용어 사전을 구축
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.legal_term_extraction.term_extractor import LegalTermExtractor
from scripts.data_processing.legal_term_extraction.domain_expander import DomainTermExpander
from scripts.data_processing.legal_term_extraction.quality_validator import QualityValidator
from scripts.data_processing.legal_term_extraction.dictionary_integrator import DictionaryIntegrator

logger = logging.getLogger(__name__)


class LegalTermDictionaryBuilder:
    """법률 용어 사전 구축 통합 빌더"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 파이프라인 구성 요소
        self.extractor = LegalTermExtractor()
        self.expander = DomainTermExpander()
        self.validator = QualityValidator()
        self.integrator = DictionaryIntegrator()
        
        # 결과 저장
        self.pipeline_results = {}
    
    def run_full_pipeline(self, 
                         data_directory: str,
                         existing_dictionary_path: str = "data/legal_term_dictionary.json",
                         output_directory: str = "data/extracted_terms",
                         min_frequency: int = 5,
                         quality_threshold: float = 0.6) -> Dict:
        """전체 파이프라인 실행"""
        
        self.logger.info("Starting legal term dictionary building pipeline")
        
        # 1단계: 용어 추출
        self.logger.info("Step 1: Extracting terms from corpus")
        extraction_results = self._extract_terms(data_directory, min_frequency)
        
        # 2단계: 도메인별 확장
        self.logger.info("Step 2: Expanding domain-specific terms")
        domain_results = self._expand_domain_terms(extraction_results)
        
        # 3단계: 품질 검증
        self.logger.info("Step 3: Validating term quality")
        validation_results = self._validate_quality(domain_results)
        
        # 4단계: 사전 통합
        self.logger.info("Step 4: Integrating dictionaries")
        integration_results = self._integrate_dictionaries(
            existing_dictionary_path, validation_results, quality_threshold
        )
        
        # 최종 결과 정리
        final_results = {
            "pipeline_summary": {
                "extraction": extraction_results,
                "domain_expansion": domain_results,
                "validation": validation_results,
                "integration": integration_results
            },
            "final_dictionary_path": integration_results["output_path"],
            "total_terms": integration_results["total_terms"],
            "quality_score": integration_results["quality_score"]
        }
        
        self.logger.info("Pipeline completed successfully")
        return final_results
    
    def _extract_terms(self, data_directory: str, min_frequency: int) -> Dict:
        """1단계: 용어 추출"""
        try:
            # 용어 추출 실행
            extraction_results = self.extractor.process_directory(data_directory, min_frequency)
            
            # 결과 저장
            output_path = "data/extracted_terms/raw_extracted_terms.json"
            self.extractor.save_results(output_path, extraction_results)
            
            self.pipeline_results["extraction"] = {
                "output_path": output_path,
                "processed_files": extraction_results["processed_files"],
                "total_terms": extraction_results["total_terms_extracted"],
                "filtered_terms": sum(len(terms) for terms in extraction_results["filtered_terms"].values())
            }
            
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error in term extraction: {e}")
            raise
    
    def _expand_domain_terms(self, extraction_results: Dict) -> Dict:
        """2단계: 도메인별 용어 확장"""
        try:
            # 추출된 용어로 도메인 확장
            domain_terms = self.expander.expand_domain_terms(extraction_results["filtered_terms"])
            
            # 향상된 사전 생성
            enhanced_dict = self.expander.generate_enhanced_dictionary(
                extraction_results["filtered_terms"], domain_terms
            )
            
            # 결과 저장
            output_path = "data/extracted_terms/domain_expanded_terms.json"
            self.expander.save_enhanced_dictionary(enhanced_dict, output_path)
            
            self.pipeline_results["domain_expansion"] = {
                "output_path": output_path,
                "total_terms": len(enhanced_dict),
                "domains": list(domain_terms.keys())
            }
            
            return enhanced_dict
            
        except Exception as e:
            self.logger.error(f"Error in domain expansion: {e}")
            raise
    
    def _validate_quality(self, enhanced_dict: Dict) -> Dict:
        """3단계: 품질 검증"""
        try:
            # 품질 검증 실행
            validation_summary = self.validator.validate_dictionary_quality(enhanced_dict)
            
            # 개선 제안 생성
            suggestions = self.validator.generate_improvement_suggestions(enhanced_dict)
            
            # 고품질 용어 필터링
            high_quality_dict = self.validator.filter_high_quality_terms(enhanced_dict)
            
            # 결과 저장
            output_path = "data/extracted_terms/quality_validated_terms.json"
            self.validator.save_validation_results(output_path, validation_summary, suggestions)
            
            # 고품질 사전 저장
            high_quality_path = "data/extracted_terms/high_quality_terms.json"
            with open(high_quality_path, 'w', encoding='utf-8') as f:
                json.dump(high_quality_dict, f, ensure_ascii=False, indent=2)
            
            self.pipeline_results["validation"] = {
                "output_path": output_path,
                "high_quality_path": high_quality_path,
                "total_terms": len(enhanced_dict),
                "high_quality_terms": len(high_quality_dict),
                "quality_ratio": len(high_quality_dict) / len(enhanced_dict) if enhanced_dict else 0
            }
            
            return high_quality_dict
            
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            raise
    
    def _integrate_dictionaries(self, 
                              existing_dict_path: str, 
                              validated_dict: Dict, 
                              quality_threshold: float) -> Dict:
        """4단계: 사전 통합"""
        try:
            # 기존 사전 로드
            existing_dict = self.integrator.load_existing_dictionary(existing_dict_path)
            
            # 사전 통합
            merged_dict, integration_stats = self.integrator.merge_dictionaries(
                existing_dict, validated_dict
            )
            
            # 통합된 사전 검증
            validation_results = self.integrator.validate_integrated_dictionary(merged_dict)
            
            # 최종 사전 저장
            output_path = "data/enhanced_legal_term_dictionary.json"
            self.integrator.save_integrated_dictionary(merged_dict, output_path)
            
            # 통합 보고서 생성
            report = self.integrator.generate_integration_report(integration_stats, validation_results)
            
            # 보고서 저장
            report_path = "data/extracted_terms/integration_report.txt"
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.pipeline_results["integration"] = {
                "output_path": output_path,
                "report_path": report_path,
                "total_terms": len(merged_dict),
                "new_terms": integration_stats["new_terms"],
                "updated_terms": integration_stats["updated_terms"],
                "quality_score": validation_results["high_confidence_terms"] / len(merged_dict) if merged_dict else 0
            }
            
            return {
                "output_path": output_path,
                "total_terms": len(merged_dict),
                "quality_score": validation_results["high_confidence_terms"] / len(merged_dict) if merged_dict else 0,
                "integration_stats": integration_stats,
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in dictionary integration: {e}")
            raise
    
    def generate_pipeline_report(self, results: Dict) -> str:
        """파이프라인 실행 보고서 생성"""
        report = []
        report.append("=== 법률 용어 사전 구축 파이프라인 실행 보고서 ===\n")
        
        # 전체 요약
        report.append("1. 전체 요약")
        report.append(f"   최종 사전 경로: {results['final_dictionary_path']}")
        report.append(f"   총 용어 수: {results['total_terms']}")
        report.append(f"   품질 점수: {results['quality_score']:.2f}")
        report.append("")
        
        # 단계별 결과
        pipeline_summary = results["pipeline_summary"]
        
        report.append("2. 1단계: 용어 추출")
        extraction = pipeline_summary["extraction"]
        report.append(f"   처리된 파일: {extraction['processed_files']}")
        report.append(f"   추출된 용어: {extraction['total_terms']}")
        report.append(f"   필터링 후 용어: {extraction['filtered_terms']}")
        report.append("")
        
        report.append("3. 2단계: 도메인별 확장")
        domain_expansion = pipeline_summary["domain_expansion"]
        report.append(f"   확장된 용어: {domain_expansion['total_terms']}")
        report.append(f"   도메인 수: {len(domain_expansion['domains'])}")
        report.append("")
        
        report.append("4. 3단계: 품질 검증")
        validation = pipeline_summary["validation"]
        report.append(f"   검증된 용어: {validation['total_terms']}")
        report.append(f"   고품질 용어: {validation['high_quality_terms']}")
        report.append(f"   품질 비율: {validation['quality_ratio']:.1%}")
        report.append("")
        
        report.append("5. 4단계: 사전 통합")
        integration = pipeline_summary["integration"]
        report.append(f"   통합된 용어: {integration['total_terms']}")
        report.append(f"   새로 추가된 용어: {integration['new_terms']}")
        report.append(f"   업데이트된 용어: {integration['updated_terms']}")
        report.append(f"   최종 품질 점수: {integration['quality_score']:.2f}")
        
        return "\n".join(report)
    
    def save_pipeline_results(self, results: Dict, output_path: str):
        """파이프라인 결과 저장"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Pipeline results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline results: {e}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='법률 용어 사전 구축 통합 빌더')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='법률 데이터 디렉토리 경로')
    parser.add_argument('--existing_dict', type=str,
                       default='data/legal_term_dictionary.json',
                       help='기존 사전 파일 경로')
    parser.add_argument('--output_dir', type=str,
                       default='data/extracted_terms',
                       help='출력 디렉토리 경로')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='최소 빈도 (기본값: 5)')
    parser.add_argument('--quality_threshold', type=float, default=0.6,
                       help='품질 기준 (기본값: 0.6)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 파이프라인 실행
    builder = LegalTermDictionaryBuilder()
    
    try:
        results = builder.run_full_pipeline(
            data_directory=args.data_dir,
            existing_dictionary_path=args.existing_dict,
            output_directory=args.output_dir,
            min_frequency=args.min_frequency,
            quality_threshold=args.quality_threshold
        )
        
        # 파이프라인 보고서 생성 및 출력
        report = builder.generate_pipeline_report(results)
        print(report)
        
        # 결과 저장
        builder.save_pipeline_results(results, f"{args.output_dir}/pipeline_results.json")
        
        print(f"\n파이프라인이 성공적으로 완료되었습니다!")
        print(f"최종 사전: {results['final_dictionary_path']}")
        print(f"총 용어 수: {results['total_terms']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
