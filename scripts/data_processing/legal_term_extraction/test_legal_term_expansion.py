# -*- coding: utf-8 -*-
"""
법률 용어 사전 확장 테스트 스크립트
"""

import json
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.legal_term_extraction.term_extractor import LegalTermExtractor
from scripts.data_processing.legal_term_extraction.domain_expander import DomainTermExpander
from scripts.data_processing.legal_term_extraction.quality_validator import QualityValidator
from scripts.data_processing.legal_term_extraction.dictionary_integrator import DictionaryIntegrator

logger = logging.getLogger(__name__)


def test_term_extraction():
    """용어 추출 테스트"""
    print("=== 용어 추출 테스트 ===")
    
    extractor = LegalTermExtractor()
    
    # 테스트 데이터 디렉토리
    test_data_dir = "data/processed/assembly/law/20251013_ml/20251010"
    
    if not Path(test_data_dir).exists():
        print(f"테스트 데이터 디렉토리가 존재하지 않습니다: {test_data_dir}")
        return False
    
    try:
        # 용어 추출 실행
        results = extractor.process_directory(test_data_dir, min_frequency=2)
        
        print(f"처리된 파일: {results['processed_files']}")
        print(f"추출된 총 용어: {results['total_terms_extracted']}")
        print(f"필터링 후 용어: {sum(len(terms) for terms in results['filtered_terms'].values())}")
        
        # 패턴별 용어 수 출력
        print("\n패턴별 용어 수:")
        for pattern_name, terms in results['filtered_terms'].items():
            print(f"  {pattern_name}: {len(terms)}개")
        
        # 상위 빈도 용어 출력
        print("\n상위 빈도 용어 (상위 10개):")
        for term, freq in list(results['term_frequencies'].items())[:10]:
            print(f"  {term}: {freq}회")
        
        return True
        
    except Exception as e:
        print(f"용어 추출 테스트 실패: {e}")
        return False


def test_domain_expansion():
    """도메인 확장 테스트"""
    print("\n=== 도메인 확장 테스트 ===")
    
    expander = DomainTermExpander()
    
    # 테스트용 추출된 용어 데이터
    test_extracted_terms = {
        "legal_concepts": ["손해배상", "계약", "소송", "형벌"],
        "legal_actions": ["배상", "보상", "청구", "제기"],
        "legal_procedures": ["절차", "신청", "심리", "판결"],
        "legal_entities": ["법원", "검사", "변호사", "피고인"],
        "legal_documents": ["소장", "답변서", "증거", "판결서"]
    }
    
    try:
        # 도메인별 용어 확장
        domain_terms = expander.expand_domain_terms(test_extracted_terms)
        
        print("도메인별 용어 수:")
        for domain, categories in domain_terms.items():
            total_terms = sum(len(terms) for terms in categories.values())
            print(f"  {domain}: {total_terms}개")
        
        # 향상된 사전 생성
        enhanced_dict = expander.generate_enhanced_dictionary(test_extracted_terms, domain_terms)
        
        print(f"\n향상된 사전 총 용어 수: {len(enhanced_dict)}")
        
        # 샘플 용어 정보 출력
        print("\n샘플 용어 정보:")
        sample_terms = list(enhanced_dict.keys())[:3]
        for term in sample_terms:
            info = enhanced_dict[term]
            print(f"  {term}:")
            print(f"    동의어: {info.get('synonyms', [])}")
            print(f"    관련 용어: {info.get('related_terms', [])}")
            print(f"    관련 법률: {info.get('related_laws', [])}")
            print(f"    신뢰도: {info.get('confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"도메인 확장 테스트 실패: {e}")
        return False


def test_quality_validation():
    """품질 검증 테스트"""
    print("\n=== 품질 검증 테스트 ===")
    
    validator = QualityValidator()
    
    # 테스트용 사전 데이터
    test_dictionary = {
        "손해배상": {
            "synonyms": ["배상", "보상", "피해보상"],
            "related_terms": ["불법행위", "채무불이행", "과실", "고의"],
            "related_laws": ["민법 제750조", "민법 제751조"],
            "precedent_keywords": ["손해배상청구권", "배상책임"],
            "confidence": 0.9,
            "frequency": 10
        },
        "계약": {
            "synonyms": ["계약서", "약정"],
            "related_terms": ["계약해지", "계약위반"],
            "related_laws": ["민법 제105조"],
            "precedent_keywords": ["계약해지권"],
            "confidence": 0.8,
            "frequency": 8
        },
        "저품질용어": {
            "synonyms": [],
            "related_terms": [],
            "related_laws": [],
            "precedent_keywords": [],
            "confidence": 0.3,
            "frequency": 1
        }
    }
    
    try:
        # 품질 검증 실행
        validation_summary = validator.validate_dictionary_quality(test_dictionary)
        
        print("품질 검증 결과:")
        print(f"  총 용어 수: {validation_summary['total_terms']}")
        print(f"  고품질 용어: {validation_summary['high_quality_terms']}")
        print(f"  중품질 용어: {validation_summary['medium_quality_terms']}")
        print(f"  저품질 용어: {validation_summary['low_quality_terms']}")
        print(f"  제외된 용어: {validation_summary['rejected_terms']}")
        
        # 개선 제안 생성
        suggestions = validator.generate_improvement_suggestions(test_dictionary)
        
        print("\n개선 제안:")
        for suggestion in suggestions["overall_suggestions"]:
            print(f"  • {suggestion}")
        
        # 고품질 용어 필터링
        high_quality_dict = validator.filter_high_quality_terms(test_dictionary)
        
        print(f"\n고품질 용어 수: {len(high_quality_dict)}")
        
        return True
        
    except Exception as e:
        print(f"품질 검증 테스트 실패: {e}")
        return False


def test_dictionary_integration():
    """사전 통합 테스트"""
    print("\n=== 사전 통합 테스트 ===")
    
    integrator = DictionaryIntegrator()
    
    # 기존 사전 (테스트용)
    existing_dict = {
        "손해배상": {
            "synonyms": ["배상", "보상"],
            "related_terms": ["불법행위", "채무불이행"],
            "related_laws": ["민법 제750조"],
            "precedent_keywords": ["손해배상청구권"],
            "confidence": 0.8,
            "frequency": 5
        }
    }
    
    # 향상된 사전 (테스트용)
    enhanced_dict = {
        "손해배상": {
            "synonyms": ["배상", "보상", "피해보상"],
            "related_terms": ["불법행위", "채무불이행", "과실", "고의"],
            "related_laws": ["민법 제750조", "민법 제751조"],
            "precedent_keywords": ["손해배상청구권", "배상책임"],
            "confidence": 0.9,
            "frequency": 10
        },
        "계약": {
            "synonyms": ["계약서", "약정", "합의"],
            "related_terms": ["계약해지", "계약위반", "계약이행"],
            "related_laws": ["민법 제105조", "민법 제543조"],
            "precedent_keywords": ["계약해지권", "계약위반"],
            "confidence": 0.8,
            "frequency": 8
        }
    }
    
    try:
        # 사전 통합 실행
        merged_dict, integration_stats = integrator.merge_dictionaries(
            existing_dict, enhanced_dict
        )
        
        print("통합 결과:")
        print(f"  기존 용어 수: {integration_stats['existing_terms']}")
        print(f"  향상된 용어 수: {integration_stats['enhanced_terms']}")
        print(f"  통합된 용어 수: {integration_stats['merged_terms']}")
        print(f"  새로 추가된 용어: {integration_stats['new_terms']}")
        print(f"  업데이트된 용어: {integration_stats['updated_terms']}")
        print(f"  제외된 용어: {integration_stats['rejected_terms']}")
        
        # 통합된 사전 검증
        validation_results = integrator.validate_integrated_dictionary(merged_dict)
        
        print(f"\n통합된 사전 검증:")
        print(f"  총 용어 수: {validation_results['total_terms']}")
        print(f"  동의어가 있는 용어: {validation_results['terms_with_synonyms']}")
        print(f"  관련 용어가 있는 용어: {validation_results['terms_with_related_terms']}")
        print(f"  관련 법률이 있는 용어: {validation_results['terms_with_related_laws']}")
        print(f"  고신뢰도 용어: {validation_results['high_confidence_terms']}")
        
        return True
        
    except Exception as e:
        print(f"사전 통합 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("법률 용어 사전 확장 시스템 테스트 시작\n")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_results = []
    
    # 각 단계별 테스트 실행
    test_results.append(("용어 추출", test_term_extraction()))
    test_results.append(("도메인 확장", test_domain_expansion()))
    test_results.append(("품질 검증", test_quality_validation()))
    test_results.append(("사전 통합", test_dictionary_integration()))
    
    # 테스트 결과 요약
    print("\n=== 테스트 결과 요약 ===")
    passed_tests = 0
    for test_name, result in test_results:
        status = "통과" if result else "실패"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n총 {len(test_results)}개 테스트 중 {passed_tests}개 통과")
    
    if passed_tests == len(test_results):
        print("모든 테스트가 성공적으로 완료되었습니다!")
        return True
    else:
        print("일부 테스트가 실패했습니다. 로그를 확인해주세요.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
