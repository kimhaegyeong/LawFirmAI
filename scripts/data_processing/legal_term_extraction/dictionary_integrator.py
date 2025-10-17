# -*- coding: utf-8 -*-
"""
법률 용어 사전 통합기
추출된 용어들을 기존 사전과 통합하여 최종 사전을 생성
"""

import json
import logging
from typing import Dict, List, Set
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class DictionaryIntegrator:
    """법률 용어 사전 통합기"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 통합 설정
        self.integration_settings = {
            "merge_strategy": "enhance",  # enhance, replace, append
            "duplicate_handling": "merge",  # merge, keep_original, keep_new
            "quality_threshold": 0.6,
            "min_frequency": 3
        }
    
    def load_existing_dictionary(self, file_path: str) -> Dict:
        """기존 사전 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            self.logger.info(f"Loaded existing dictionary with {len(dictionary)} terms")
            return dictionary
        except FileNotFoundError:
            self.logger.warning(f"Existing dictionary not found: {file_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading existing dictionary: {e}")
            return {}
    
    def load_enhanced_dictionary(self, file_path: str) -> Dict:
        """향상된 사전 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
            self.logger.info(f"Loaded enhanced dictionary with {len(dictionary)} terms")
            return dictionary
        except Exception as e:
            self.logger.error(f"Error loading enhanced dictionary: {e}")
            return {}
    
    def merge_dictionaries(self, existing_dict: Dict, enhanced_dict: Dict) -> Dict:
        """사전 통합"""
        self.logger.info("Starting dictionary integration")
        
        merged_dict = existing_dict.copy()
        integration_stats = {
            "existing_terms": len(existing_dict),
            "enhanced_terms": len(enhanced_dict),
            "merged_terms": 0,
            "new_terms": 0,
            "updated_terms": 0,
            "rejected_terms": 0
        }
        
        for term, term_info in enhanced_dict.items():
            if term in merged_dict:
                # 기존 용어 업데이트
                merged_dict[term] = self._merge_term_info(
                    merged_dict[term], term_info
                )
                integration_stats["updated_terms"] += 1
            else:
                # 새 용어 추가 (모든 용어 추가)
                merged_dict[term] = term_info
                integration_stats["new_terms"] += 1
            
            integration_stats["merged_terms"] += 1
        
        self.logger.info(f"Integration completed: {integration_stats}")
        return merged_dict, integration_stats
    
    def _merge_term_info(self, existing_info: Dict, new_info: Dict) -> Dict:
        """용어 정보 통합"""
        merged_info = existing_info.copy()
        
        # 동의어 통합
        existing_synonyms = set(existing_info.get("synonyms", []))
        new_synonyms = set(new_info.get("synonyms", []))
        merged_info["synonyms"] = list(existing_synonyms.union(new_synonyms))
        
        # 관련 용어 통합
        existing_related = set(existing_info.get("related_terms", []))
        new_related = set(new_info.get("related_terms", []))
        merged_info["related_terms"] = list(existing_related.union(new_related))
        
        # 관련 법률 통합
        existing_laws = set(existing_info.get("related_laws", []))
        new_laws = set(new_info.get("related_laws", []))
        merged_info["related_laws"] = list(existing_laws.union(new_laws))
        
        # 판례 키워드 통합
        existing_precedents = set(existing_info.get("precedent_keywords", []))
        new_precedents = set(new_info.get("precedent_keywords", []))
        merged_info["precedent_keywords"] = list(existing_precedents.union(new_precedents))
        
        # 신뢰도 업데이트 (더 높은 값 선택)
        existing_confidence = existing_info.get("confidence", 0.0)
        new_confidence = new_info.get("confidence", 0.0)
        merged_info["confidence"] = max(existing_confidence, new_confidence)
        
        # 빈도 업데이트 (합산)
        existing_frequency = existing_info.get("frequency", 0)
        new_frequency = new_info.get("frequency", 0)
        merged_info["frequency"] = existing_frequency + new_frequency
        
        # 도메인 및 카테고리 업데이트
        if "domain" in new_info:
            merged_info["domain"] = new_info["domain"]
        if "category" in new_info:
            merged_info["category"] = new_info["category"]
        
        return merged_info
    
    def _should_add_term(self, term_info: Dict) -> bool:
        """용어 추가 여부 판단"""
        # 품질 기준 확인
        confidence = term_info.get("confidence", 0.0)
        if confidence < self.integration_settings["quality_threshold"]:
            return False
        
        # 빈도 기준 확인
        frequency = term_info.get("frequency", 0)
        if frequency < self.integration_settings["min_frequency"]:
            return False
        
        # 기본적인 정보가 있으면 추가 (더 관대한 기준)
        return True
    
    def validate_integrated_dictionary(self, dictionary: Dict) -> Dict:
        """통합된 사전 검증"""
        validation_results = {
            "total_terms": len(dictionary),
            "terms_with_synonyms": 0,
            "terms_with_related_terms": 0,
            "terms_with_related_laws": 0,
            "terms_with_precedent_keywords": 0,
            "high_confidence_terms": 0,
            "domain_distribution": defaultdict(int),
            "category_distribution": defaultdict(int)
        }
        
        for term, term_info in dictionary.items():
            # 필수 필드 확인
            if term_info.get("synonyms"):
                validation_results["terms_with_synonyms"] += 1
            if term_info.get("related_terms"):
                validation_results["terms_with_related_terms"] += 1
            if term_info.get("related_laws"):
                validation_results["terms_with_related_laws"] += 1
            if term_info.get("precedent_keywords"):
                validation_results["terms_with_precedent_keywords"] += 1
            
            # 신뢰도 확인
            if term_info.get("confidence", 0.0) >= 0.8:
                validation_results["high_confidence_terms"] += 1
            
            # 도메인 및 카테고리 분포
            domain = term_info.get("domain", "기타")
            category = term_info.get("category", "기타")
            validation_results["domain_distribution"][domain] += 1
            validation_results["category_distribution"][category] += 1
        
        return validation_results
    
    def generate_integration_report(self, integration_stats: Dict, validation_results: Dict) -> str:
        """통합 보고서 생성"""
        report = []
        report.append("=== 법률 용어 사전 통합 보고서 ===\n")
        
        # 통합 통계
        report.append("1. 통합 통계")
        report.append(f"   기존 용어 수: {integration_stats['existing_terms']}")
        report.append(f"   향상된 용어 수: {integration_stats['enhanced_terms']}")
        report.append(f"   통합된 용어 수: {integration_stats['merged_terms']}")
        report.append(f"   새로 추가된 용어: {integration_stats['new_terms']}")
        report.append(f"   업데이트된 용어: {integration_stats['updated_terms']}")
        report.append(f"   제외된 용어: {integration_stats['rejected_terms']}")
        report.append("")
        
        # 검증 결과
        report.append("2. 검증 결과")
        report.append(f"   총 용어 수: {validation_results['total_terms']}")
        report.append(f"   동의어가 있는 용어: {validation_results['terms_with_synonyms']} ({validation_results['terms_with_synonyms']/validation_results['total_terms']:.1%})")
        report.append(f"   관련 용어가 있는 용어: {validation_results['terms_with_related_terms']} ({validation_results['terms_with_related_terms']/validation_results['total_terms']:.1%})")
        report.append(f"   관련 법률이 있는 용어: {validation_results['terms_with_related_laws']} ({validation_results['terms_with_related_laws']/validation_results['total_terms']:.1%})")
        report.append(f"   판례 키워드가 있는 용어: {validation_results['terms_with_precedent_keywords']} ({validation_results['terms_with_precedent_keywords']/validation_results['total_terms']:.1%})")
        report.append(f"   고신뢰도 용어: {validation_results['high_confidence_terms']} ({validation_results['high_confidence_terms']/validation_results['total_terms']:.1%})")
        report.append("")
        
        # 도메인 분포
        report.append("3. 도메인별 분포")
        for domain, count in validation_results["domain_distribution"].items():
            percentage = count / validation_results["total_terms"] * 100
            report.append(f"   {domain}: {count}개 ({percentage:.1f}%)")
        report.append("")
        
        # 카테고리 분포
        report.append("4. 카테고리별 분포 (상위 10개)")
        sorted_categories = sorted(validation_results["category_distribution"].items(), 
                                 key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            percentage = count / validation_results["total_terms"] * 100
            report.append(f"   {category}: {count}개 ({percentage:.1f}%)")
        
        return "\n".join(report)
    
    def save_integrated_dictionary(self, dictionary: Dict, output_path: str):
        """통합된 사전 저장"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Integrated dictionary saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving integrated dictionary: {e}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='법률 용어 사전 통합기')
    parser.add_argument('--existing_dict', type=str, 
                       default='data/legal_term_dictionary.json',
                       help='기존 사전 파일 경로')
    parser.add_argument('--enhanced_dict', type=str, required=True,
                       help='향상된 사전 파일 경로')
    parser.add_argument('--output_file', type=str,
                       default='data/enhanced_legal_term_dictionary.json',
                       help='통합된 사전 출력 파일 경로')
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
    
    # 사전 통합 실행
    integrator = DictionaryIntegrator()
    
    # 기존 사전 로드
    existing_dict = integrator.load_existing_dictionary(args.existing_dict)
    
    # 향상된 사전 로드
    enhanced_dict = integrator.load_enhanced_dictionary(args.enhanced_dict)
    
    # 사전 통합
    merged_dict, integration_stats = integrator.merge_dictionaries(
        existing_dict, enhanced_dict
    )
    
    # 통합된 사전 검증
    validation_results = integrator.validate_integrated_dictionary(merged_dict)
    
    # 통합 보고서 생성 및 출력
    report = integrator.generate_integration_report(integration_stats, validation_results)
    print(report)
    
    # 통합된 사전 저장
    integrator.save_integrated_dictionary(merged_dict, args.output_file)
    
    print(f"\n통합된 사전이 {args.output_file}에 저장되었습니다.")
    print(f"총 {len(merged_dict)}개의 용어가 포함되어 있습니다.")


if __name__ == "__main__":
    main()
