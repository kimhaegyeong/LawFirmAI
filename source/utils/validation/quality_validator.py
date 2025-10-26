# -*- coding: utf-8 -*-
"""
단계별 품질 검증 시스템
Step-by-Step Quality Validation System
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    """검증 수준"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    level: ValidationLevel

@dataclass
class QualityMetrics:
    """품질 지표"""
    readability_score: float
    completeness_score: float
    accuracy_score: float
    relevance_score: float
    overall_score: float

class QualityValidator:
    """품질 검증 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 검증 규칙들
        self.validation_rules = {
            'critical': [
                self._validate_minimum_length,
                self._validate_has_content,
                self._validate_no_empty_sections
            ],
            'high': [
                self._validate_no_placeholders,
                self._validate_no_disclaimers,
                self._validate_no_duplicate_sections
            ],
            'medium': [
                self._validate_readability,
                self._validate_structure,
                self._validate_completeness
            ],
            'low': [
                self._validate_formatting,
                self._validate_tone,
                self._validate_relevance
            ]
        }
        
        # 품질 임계값
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'fair': 0.5,
            'poor': 0.0
        }
    
    def validate_response(self, text: str, question: str = "") -> Dict[str, Any]:
        """응답 검증 메인 메서드"""
        try:
            validation_results = {}
            
            # 각 수준별 검증 수행
            for level, rules in self.validation_rules.items():
                level_results = []
                for rule in rules:
                    result = rule(text, question)
                    level_results.append(result)
                
                validation_results[level] = level_results
            
            # 전체 품질 지표 계산
            quality_metrics = self._calculate_quality_metrics(text, question)
            
            # 최종 검증 결과 생성
            final_result = self._generate_final_result(validation_results, quality_metrics)
            
            return {
                'validation_results': validation_results,
                'quality_metrics': quality_metrics,
                'final_result': final_result,
                'is_valid': final_result['is_valid'],
                'overall_score': quality_metrics.overall_score
            }
            
        except Exception as e:
            self.logger.error(f"응답 검증 중 오류: {e}")
            return {
                'validation_results': {},
                'quality_metrics': QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0),
                'final_result': {'is_valid': False, 'issues': [str(e)]},
                'is_valid': False,
                'overall_score': 0.0
            }
    
    def _validate_minimum_length(self, text: str, question: str = "") -> ValidationResult:
        """최소 길이 검증"""
        min_length = 50
        is_valid = len(text.strip()) >= min_length
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append(f"응답이 너무 짧습니다 ({len(text.strip())} 문자)")
            recommendations.append("더 자세한 설명을 추가하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=min(len(text.strip()) / min_length, 1.0),
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.CRITICAL
        )
    
    def _validate_has_content(self, text: str, question: str = "") -> ValidationResult:
        """실제 내용 존재 검증"""
        # 의미있는 내용이 있는지 확인
        meaningful_content = re.sub(r'[#\*\s\n]', '', text)
        is_valid = len(meaningful_content) > 20
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("실제 내용이 부족합니다")
            recommendations.append("구체적인 정보를 포함하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=min(len(meaningful_content) / 100, 1.0),
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.CRITICAL
        )
    
    def _validate_no_empty_sections(self, text: str, question: str = "") -> ValidationResult:
        """빈 섹션 검증"""
        empty_sections = re.findall(r'###\s*[^\n]+\s*\n+\s*\n+', text)
        is_valid = len(empty_sections) == 0
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append(f"빈 섹션이 {len(empty_sections)}개 있습니다")
            recommendations.append("빈 섹션을 제거하거나 내용을 추가하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=1.0 - (len(empty_sections) * 0.2),
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.CRITICAL
        )
    
    def _validate_no_placeholders(self, text: str, question: str = "") -> ValidationResult:
        """플레이스홀더 검증"""
        placeholder_patterns = [
            r'\*[^*]+\*',
            r'정확한\s*조문\s*번호와\s*내용',
            r'쉬운\s*말로\s*풀어서\s*설명',
            r'구체적\s*예시와\s*설명',
            r'법적\s*리스크와\s*제한사항'
        ]
        
        placeholder_count = 0
        for pattern in placeholder_patterns:
            placeholder_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        is_valid = placeholder_count == 0
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append(f"플레이스홀더가 {placeholder_count}개 있습니다")
            recommendations.append("플레이스홀더를 실제 내용으로 교체하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=max(1.0 - (placeholder_count * 0.3), 0.0),
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.HIGH
        )
    
    def _validate_no_disclaimers(self, text: str, question: str = "") -> ValidationResult:
        """면책 조항 검증"""
        disclaimer_patterns = [
            r'본\s*답변은.*?바랍니다',
            r'구체적인\s*법률\s*문제는.*?바랍니다',
            r'변호사와\s*상담.*?바랍니다',
            r'법률\s*전문가와\s*상담.*?바랍니다'
        ]
        
        disclaimer_count = 0
        for pattern in disclaimer_patterns:
            disclaimer_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        is_valid = disclaimer_count == 0
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append(f"면책 조항이 {disclaimer_count}개 있습니다")
            recommendations.append("불필요한 면책 조항을 제거하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=max(1.0 - (disclaimer_count * 0.2), 0.0),
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.HIGH
        )
    
    def _validate_no_duplicate_sections(self, text: str, question: str = "") -> ValidationResult:
        """중복 섹션 검증"""
        section_titles = re.findall(r'###\s*([^\n]+)', text)
        unique_titles = set(section_titles)
        
        is_valid = len(section_titles) == len(unique_titles)
        
        issues = []
        recommendations = []
        
        if not is_valid:
            duplicate_count = len(section_titles) - len(unique_titles)
            issues.append(f"중복된 섹션 제목이 {duplicate_count}개 있습니다")
            recommendations.append("중복된 섹션을 제거하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=len(unique_titles) / len(section_titles) if section_titles else 1.0,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.HIGH
        )
    
    def _validate_readability(self, text: str, question: str = "") -> ValidationResult:
        """가독성 검증"""
        # 문장 길이 분석
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len(sentences) if sentences else 0
        
        # 적절한 문장 길이 (10-20 단어)
        readability_score = 1.0 - abs(avg_sentence_length - 15) / 15
        readability_score = max(min(readability_score, 1.0), 0.0)
        
        is_valid = readability_score >= 0.5
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append(f"평균 문장 길이가 부적절합니다 ({avg_sentence_length:.1f} 단어)")
            recommendations.append("문장을 더 간결하게 작성하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=readability_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.MEDIUM
        )
    
    def _validate_structure(self, text: str, question: str = "") -> ValidationResult:
        """구조 검증"""
        # 제목과 내용의 균형
        titles = len(re.findall(r'^#+\s+', text, re.MULTILINE))
        paragraphs = len(re.split(r'\n\s*\n', text))
        
        # 적절한 구조 비율 (제목:내용 = 1:3)
        structure_score = 1.0 - abs(titles / paragraphs - 0.25) if paragraphs > 0 else 0.0
        
        is_valid = structure_score >= 0.6
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("구조가 불균형합니다")
            recommendations.append("제목과 내용의 비율을 조정하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=structure_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.MEDIUM
        )
    
    def _validate_completeness(self, text: str, question: str = "") -> ValidationResult:
        """완성도 검증"""
        # 질문 키워드가 답변에 포함되어 있는지 확인
        if question:
            question_words = set(re.findall(r'\w+', question.lower()))
            answer_words = set(re.findall(r'\w+', text.lower()))
            
            overlap = len(question_words.intersection(answer_words))
            completeness_score = overlap / len(question_words) if question_words else 0.0
        else:
            completeness_score = 0.8  # 질문이 없으면 기본 점수
        
        is_valid = completeness_score >= 0.3
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("질문에 대한 완전한 답변이 아닙니다")
            recommendations.append("질문의 모든 측면을 다루세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=completeness_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.MEDIUM
        )
    
    def _validate_formatting(self, text: str, question: str = "") -> ValidationResult:
        """포맷팅 검증"""
        # 연속된 빈 줄 체크
        consecutive_newlines = len(re.findall(r'\n{3,}', text))
        formatting_score = max(1.0 - (consecutive_newlines * 0.2), 0.0)
        
        is_valid = formatting_score >= 0.8
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("포맷팅이 불규칙합니다")
            recommendations.append("연속된 빈 줄을 정리하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=formatting_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.LOW
        )
    
    def _validate_tone(self, text: str, question: str = "") -> ValidationResult:
        """톤 검증"""
        # 친근한 톤인지 확인
        friendly_words = ['입니다', '예요', '네요', '습니다', '합니다']
        formal_words = ['이다', '하다', '되다']
        
        friendly_count = sum(text.count(word) for word in friendly_words)
        formal_count = sum(text.count(word) for word in formal_words)
        
        tone_score = friendly_count / (friendly_count + formal_count) if (friendly_count + formal_count) > 0 else 0.5
        
        is_valid = tone_score >= 0.3
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("톤이 너무 딱딱합니다")
            recommendations.append("더 친근한 톤으로 작성하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=tone_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.LOW
        )
    
    def _validate_relevance(self, text: str, question: str = "") -> ValidationResult:
        """관련성 검증"""
        # 법률 관련 키워드 포함 여부
        legal_keywords = ['법', '조문', '규정', '계약', '소송', '재판', '판결']
        legal_count = sum(text.count(keyword) for keyword in legal_keywords)
        
        relevance_score = min(legal_count / 5, 1.0)
        
        is_valid = relevance_score >= 0.2
        
        issues = []
        recommendations = []
        
        if not is_valid:
            issues.append("법률 관련 내용이 부족합니다")
            recommendations.append("법률적 근거를 더 포함하세요")
        
        return ValidationResult(
            is_valid=is_valid,
            score=relevance_score,
            issues=issues,
            recommendations=recommendations,
            level=ValidationLevel.LOW
        )
    
    def _calculate_quality_metrics(self, text: str, question: str = "") -> QualityMetrics:
        """품질 지표 계산"""
        try:
            # 가독성 점수
            readability_score = self._calculate_readability_score(text)
            
            # 완성도 점수
            completeness_score = self._calculate_completeness_score(text, question)
            
            # 정확성 점수 (기본값)
            accuracy_score = 0.8
            
            # 관련성 점수
            relevance_score = self._calculate_relevance_score(text)
            
            # 전체 점수
            overall_score = (readability_score + completeness_score + accuracy_score + relevance_score) / 4
            
            return QualityMetrics(
                readability_score=readability_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                relevance_score=relevance_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            self.logger.error(f"품질 지표 계산 중 오류: {e}")
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_readability_score(self, text: str) -> float:
        """가독성 점수 계산"""
        sentences = re.split(r'[.!?]', text)
        if not sentences:
            return 0.0
        
        avg_length = sum(len(s.split()) for s in sentences if s.strip()) / len(sentences)
        return max(1.0 - abs(avg_length - 15) / 15, 0.0)
    
    def _calculate_completeness_score(self, text: str, question: str) -> float:
        """완성도 점수 계산"""
        if not question:
            return 0.8
        
        question_words = set(re.findall(r'\w+', question.lower()))
        answer_words = set(re.findall(r'\w+', text.lower()))
        
        overlap = len(question_words.intersection(answer_words))
        return overlap / len(question_words) if question_words else 0.0
    
    def _calculate_relevance_score(self, text: str) -> float:
        """관련성 점수 계산"""
        legal_keywords = ['법', '조문', '규정', '계약', '소송', '재판', '판결']
        legal_count = sum(text.count(keyword) for keyword in legal_keywords)
        return min(legal_count / 5, 1.0)
    
    def _generate_final_result(self, validation_results: Dict[str, List[ValidationResult]], 
                              quality_metrics: QualityMetrics) -> Dict[str, Any]:
        """최종 결과 생성"""
        all_issues = []
        all_recommendations = []
        
        # 모든 검증 결과에서 이슈와 권장사항 수집
        for level_results in validation_results.values():
            for result in level_results:
                all_issues.extend(result.issues)
                all_recommendations.extend(result.recommendations)
        
        # 전체 유효성 판단 (critical과 high 수준의 검증이 모두 통과해야 함)
        critical_valid = all(result.is_valid for result in validation_results.get('critical', []))
        high_valid = all(result.is_valid for result in validation_results.get('high', []))
        
        is_valid = critical_valid and high_valid
        
        return {
            'is_valid': is_valid,
            'issues': all_issues,
            'recommendations': all_recommendations,
            'quality_grade': self._determine_quality_grade(quality_metrics.overall_score)
        }
    
    def _determine_quality_grade(self, score: float) -> str:
        """품질 등급 결정"""
        if score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif score >= self.quality_thresholds['good']:
            return 'good'
        elif score >= self.quality_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'


# 전역 인스턴스
quality_validator = QualityValidator()
