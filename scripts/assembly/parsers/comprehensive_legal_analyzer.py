"""
Comprehensive Legal Analyzer for Korean Law Data

This module provides comprehensive analysis of Korean law data by combining
hierarchy classification, field classification, structure analysis, and reference analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .legal_hierarchy_classifier import LegalHierarchyClassifier
from .legal_field_classifier import LegalFieldClassifier
from .legal_structure_parser import LegalStructureParser
from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class ComprehensiveLegalAnalyzer:
    """통합 법률 분석기"""
    
    def __init__(self):
        """Initialize comprehensive analyzer with all sub-analyzers"""
        self.hierarchy_classifier = LegalHierarchyClassifier()
        self.field_classifier = LegalFieldClassifier()
        self.structure_parser = LegalStructureParser()
        self.metadata_extractor = MetadataExtractor()
    
    def analyze_law_comprehensively(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        종합 법률 분석
        
        Args:
            law_data (Dict[str, Any]): Law data dictionary
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        try:
            analysis_result = {
                'law_id': law_data.get('law_id', ''),
                'law_name': law_data.get('law_name', ''),
                'analysis_timestamp': datetime.now().isoformat(),
                
                # 위계 구조 분석
                'hierarchy_analysis': self.hierarchy_classifier.classify_law_hierarchy(law_data),
                
                # 분야 분류
                'field_classification': self.field_classifier.classify_legal_field(law_data),
                
                # 구조 분석
                'structure_analysis': self.structure_parser.parse_legal_structure(
                    law_data.get('law_content', '')
                ),
                
                # 메타데이터 추출 (기존)
                'metadata_extraction': self.metadata_extractor.extract(law_data),
                
                # 종합 평가
                'comprehensive_score': 0.0,
                'analysis_quality': 'unknown',
                'analysis_recommendations': []
            }
            
            # 종합 점수 계산
            analysis_result['comprehensive_score'] = self._calculate_comprehensive_score(analysis_result)
            
            # 분석 품질 평가
            analysis_result['analysis_quality'] = self._evaluate_analysis_quality(analysis_result)
            
            # 분석 권장사항 생성
            analysis_result['analysis_recommendations'] = self._generate_recommendations(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive legal analysis: {e}")
            return {
                'law_id': law_data.get('law_id', ''),
                'law_name': law_data.get('law_name', ''),
                'comprehensive_score': 0.0,
                'analysis_quality': 'poor',
                'error': str(e)
            }
    
    def _calculate_comprehensive_score(self, analysis_result: Dict[str, Any]) -> float:
        """종합 점수 계산"""
        scores = []
        
        # 위계 분류 점수
        hierarchy_score = analysis_result['hierarchy_analysis'].get('hierarchy_confidence', 0.0)
        scores.append(hierarchy_score)
        
        # 분야 분류 점수
        field_score = analysis_result['field_classification'].get('field_confidence', 0.0)
        scores.append(field_score)
        
        # 구조 분석 점수
        structure_score = analysis_result['structure_analysis'].get('structure_complexity', 0.0)
        scores.append(structure_score)
        
        # 메타데이터 추출 점수
        metadata_score = self._calculate_metadata_score(analysis_result['metadata_extraction'])
        scores.append(metadata_score)
        
        # 가중 평균 계산
        weights = [0.3, 0.3, 0.2, 0.2]  # 위계, 분야, 구조, 메타데이터 순
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return min(1.0, max(0.0, weighted_score))
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any]) -> float:
        """메타데이터 점수 계산"""
        score = 0.0
        total_factors = 0
        
        # 시행 정보 점수
        if metadata.get('enforcement_info'):
            score += 0.3
        total_factors += 0.3
        
        # 개정 정보 점수
        if metadata.get('amendment_info'):
            score += 0.3
        total_factors += 0.3
        
        # 부서 정보 점수
        if metadata.get('ministry'):
            score += 0.2
        total_factors += 0.2
        
        # 참조 정보 점수
        if metadata.get('references'):
            score += 0.2
        total_factors += 0.2
        
        return score / total_factors if total_factors > 0 else 0.0
    
    def _evaluate_analysis_quality(self, analysis_result: Dict[str, Any]) -> str:
        """분석 품질 평가"""
        score = analysis_result['comprehensive_score']
        
        # 각 분석 요소의 품질도 고려
        hierarchy_quality = analysis_result['hierarchy_analysis'].get('hierarchy_confidence', 0.0)
        field_quality = analysis_result['field_classification'].get('field_confidence', 0.0)
        structure_quality = analysis_result['structure_analysis'].get('structure_complexity', 0.0)
        
        # 최소 품질 임계값 확인
        min_quality_threshold = 0.3
        
        if (score >= 0.8 and 
            hierarchy_quality >= min_quality_threshold and 
            field_quality >= min_quality_threshold):
            return 'excellent'
        elif (score >= 0.6 and 
              hierarchy_quality >= min_quality_threshold):
            return 'good'
        elif (score >= 0.4 or 
              hierarchy_quality >= min_quality_threshold or 
              field_quality >= min_quality_threshold):
            return 'fair'
        else:
            return 'poor'
    
    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """분석 권장사항 생성"""
        recommendations = []
        
        # 위계 분류 권장사항
        hierarchy_confidence = analysis_result['hierarchy_analysis'].get('hierarchy_confidence', 0.0)
        if hierarchy_confidence < 0.5:
            recommendations.append("법률 위계 분류의 신뢰도가 낮습니다. 법률명이나 공포번호를 확인해주세요.")
        
        # 분야 분류 권장사항
        field_confidence = analysis_result['field_classification'].get('field_confidence', 0.0)
        if field_confidence < 0.5:
            recommendations.append("법률 분야 분류의 신뢰도가 낮습니다. 법률 내용을 더 자세히 분석해주세요.")
        
        # 구조 분석 권장사항
        structure_complexity = analysis_result['structure_analysis'].get('structure_complexity', 0.0)
        if structure_complexity < 0.3:
            recommendations.append("법률 구조가 단순합니다. 조문, 항, 호 등의 구조를 확인해주세요.")
        
        # 메타데이터 권장사항
        metadata = analysis_result['metadata_extraction']
        if not metadata.get('enforcement_info'):
            recommendations.append("시행 정보가 누락되었습니다. 시행 조항을 확인해주세요.")
        
        if not metadata.get('amendment_info'):
            recommendations.append("개정 정보가 누락되었습니다. 개정 이력을 확인해주세요.")
        
        # 종합 점수 기반 권장사항
        comprehensive_score = analysis_result['comprehensive_score']
        if comprehensive_score < 0.5:
            recommendations.append("전체적인 분석 품질이 낮습니다. 원본 데이터의 품질을 확인해주세요.")
        
        return recommendations
    
    def batch_analyze_laws(self, law_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        법률 데이터 배치 분석
        
        Args:
            law_data_list (List[Dict[str, Any]]): List of law data dictionaries
            
        Returns:
            Dict[str, Any]: Batch analysis results
        """
        try:
            batch_results = {
                'total_laws': len(law_data_list),
                'analysis_results': [],
                'batch_statistics': {},
                'batch_timestamp': datetime.now().isoformat()
            }
            
            # 각 법률 분석
            for law_data in law_data_list:
                analysis_result = self.analyze_law_comprehensively(law_data)
                batch_results['analysis_results'].append(analysis_result)
            
            # 배치 통계 생성
            batch_results['batch_statistics'] = self._generate_batch_statistics(
                batch_results['analysis_results']
            )
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return {
                'total_laws': len(law_data_list),
                'error': str(e),
                'batch_timestamp': datetime.now().isoformat()
            }
    
    def _generate_batch_statistics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """배치 통계 생성"""
        if not analysis_results:
            return {}
        
        # 기본 통계
        total_laws = len(analysis_results)
        scores = [result.get('comprehensive_score', 0.0) for result in analysis_results]
        
        # 위계별 통계
        hierarchy_stats = {}
        for result in analysis_results:
            hierarchy_type = result.get('hierarchy_analysis', {}).get('hierarchy_type', 'unknown')
            if hierarchy_type not in hierarchy_stats:
                hierarchy_stats[hierarchy_type] = 0
            hierarchy_stats[hierarchy_type] += 1
        
        # 분야별 통계
        field_stats = {}
        for result in analysis_results:
            field = result.get('field_classification', {}).get('primary_field', 'unknown')
            if field not in field_stats:
                field_stats[field] = 0
            field_stats[field] += 1
        
        # 품질별 통계
        quality_stats = {}
        for result in analysis_results:
            quality = result.get('analysis_quality', 'unknown')
            if quality not in quality_stats:
                quality_stats[quality] = 0
            quality_stats[quality] += 1
        
        return {
            'total_laws': total_laws,
            'average_score': sum(scores) / len(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'hierarchy_distribution': hierarchy_stats,
            'field_distribution': field_stats,
            'quality_distribution': quality_stats,
            'high_quality_count': sum(1 for score in scores if score >= 0.7),
            'low_quality_count': sum(1 for score in scores if score < 0.4)
        }
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 요약"""
        return {
            'law_id': analysis_result.get('law_id', ''),
            'law_name': analysis_result.get('law_name', ''),
            'hierarchy_type': analysis_result.get('hierarchy_analysis', {}).get('hierarchy_type', 'unknown'),
            'hierarchy_level': analysis_result.get('hierarchy_analysis', {}).get('hierarchy_level', 0),
            'primary_field': analysis_result.get('field_classification', {}).get('primary_field', 'unknown'),
            'structure_type': analysis_result.get('structure_analysis', {}).get('structure_type', 'unknown'),
            'total_articles': analysis_result.get('structure_analysis', {}).get('total_articles', 0),
            'comprehensive_score': analysis_result.get('comprehensive_score', 0.0),
            'analysis_quality': analysis_result.get('analysis_quality', 'unknown'),
            'recommendation_count': len(analysis_result.get('analysis_recommendations', []))
        }
    
    def validate_analysis_result(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과 검증"""
        validation_result = {
            'is_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # 필수 필드 검증
        required_fields = ['law_id', 'law_name', 'comprehensive_score', 'analysis_quality']
        for field in required_fields:
            if field not in analysis_result or not analysis_result[field]:
                validation_result['validation_errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
        
        # 점수 범위 검증
        score = analysis_result.get('comprehensive_score', 0.0)
        if not (0.0 <= score <= 1.0):
            validation_result['validation_errors'].append(f"Invalid score range: {score}")
            validation_result['is_valid'] = False
        
        # 품질 레벨 검증
        quality = analysis_result.get('analysis_quality', 'unknown')
        valid_qualities = ['excellent', 'good', 'fair', 'poor']
        if quality not in valid_qualities:
            validation_result['validation_warnings'].append(f"Unknown quality level: {quality}")
        
        # 위계 레벨 검증
        hierarchy_level = analysis_result.get('hierarchy_analysis', {}).get('hierarchy_level', 0)
        if not (1 <= hierarchy_level <= 6):
            validation_result['validation_warnings'].append(f"Invalid hierarchy level: {hierarchy_level}")
        
        return validation_result
    
    def export_analysis_report(self, analysis_result: Dict[str, Any], 
                              format: str = 'json') -> str:
        """분석 결과 보고서 내보내기"""
        if format == 'json':
            import json
            return json.dumps(analysis_result, ensure_ascii=False, indent=2)
        elif format == 'summary':
            summary = self.get_analysis_summary(analysis_result)
            return f"""
법률 분석 보고서
================
법률 ID: {summary['law_id']}
법률명: {summary['law_name']}
위계 유형: {summary['hierarchy_type']} (레벨 {summary['hierarchy_level']})
주요 분야: {summary['primary_field']}
구조 유형: {summary['structure_type']}
총 조문 수: {summary['total_articles']}
종합 점수: {summary['comprehensive_score']:.2f}
분석 품질: {summary['analysis_quality']}
권장사항 수: {summary['recommendation_count']}
"""
        else:
            raise ValueError(f"Unsupported format: {format}")
