# -*- coding: utf-8 -*-
"""
고급 응답 후처리 시스템
Advanced Response Post-Processing System
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .semantic_deduplicator import semantic_deduplicator

class ResponseQuality(Enum):
    """응답 품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ProcessingResult:
    """처리 결과"""
    processed_text: str
    quality_score: float
    quality_grade: ResponseQuality
    improvements_made: List[str]
    original_length: int
    processed_length: int
    reduction_rate: float

class AdvancedResponseProcessor:
    """고급 응답 후처리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 품질 임계값
        self.quality_thresholds = {
            ResponseQuality.EXCELLENT: 0.9,
            ResponseQuality.GOOD: 0.7,
            ResponseQuality.FAIR: 0.5,
            ResponseQuality.POOR: 0.0
        }
        
        # 처리 단계별 설정
        self.processing_steps = [
            'remove_section_titles',
            'remove_placeholders',
            'remove_disclaimers',
            'remove_intro_phrases',
            'deduplicate_content',
            'clean_formatting',
            'validate_content'
        ]
    
    def process_response(self, text: str) -> ProcessingResult:
        """응답 후처리 메인 메서드"""
        try:
            original_length = len(text)
            improvements_made = []
            
            # 단계별 처리
            processed_text = text
            
            for step in self.processing_steps:
                step_result = self._execute_processing_step(step, processed_text)
                if step_result['changed']:
                    processed_text = step_result['text']
                    improvements_made.append(step_result['description'])
            
            # 품질 분석
            quality_analysis = semantic_deduplicator.analyze_content_quality(processed_text)
            quality_score = quality_analysis.get('quality_score', 0.0)
            quality_grade = self._determine_quality_grade(quality_score)
            
            # 길이 변화 계산
            processed_length = len(processed_text)
            reduction_rate = ((original_length - processed_length) / original_length) * 100 if original_length > 0 else 0
            
            return ProcessingResult(
                processed_text=processed_text,
                quality_score=quality_score,
                quality_grade=quality_grade,
                improvements_made=improvements_made,
                original_length=original_length,
                processed_length=processed_length,
                reduction_rate=reduction_rate
            )
            
        except Exception as e:
            self.logger.error(f"응답 후처리 중 오류: {e}")
            return ProcessingResult(
                processed_text=text,
                quality_score=0.0,
                quality_grade=ResponseQuality.POOR,
                improvements_made=[],
                original_length=len(text),
                processed_length=len(text),
                reduction_rate=0.0
            )
    
    def _execute_processing_step(self, step: str, text: str) -> Dict[str, Any]:
        """처리 단계 실행"""
        try:
            original_text = text
            
            if step == 'remove_section_titles':
                text = self._remove_section_titles(text)
                description = "섹션 제목 제거"
                
            elif step == 'remove_placeholders':
                text = self._remove_placeholders(text)
                description = "플레이스홀더 제거"
                
            elif step == 'remove_disclaimers':
                text = self._remove_disclaimers(text)
                description = "면책 조항 제거"
                
            elif step == 'remove_intro_phrases':
                text = self._remove_intro_phrases(text)
                description = "불필요한 서론 제거"
                
            elif step == 'deduplicate_content':
                text = semantic_deduplicator.deduplicate_content(text)
                description = "의미 기반 중복 제거"
                
            elif step == 'clean_formatting':
                text = self._clean_formatting(text)
                description = "포맷팅 정리"
                
            elif step == 'validate_content':
                text = self._validate_content(text)
                description = "내용 검증"
                
            else:
                description = "알 수 없는 단계"
            
            return {
                'text': text,
                'changed': text != original_text,
                'description': description
            }
            
        except Exception as e:
            self.logger.error(f"처리 단계 {step} 실행 중 오류: {e}")
            return {
                'text': text,
                'changed': False,
                'description': f"오류 발생: {step}"
            }
    
    def _remove_section_titles(self, text: str) -> str:
        """섹션 제목 제거"""
        patterns = [
            r'###\s*관련\s*법령\s*\n*',
            r'###\s*법령\s*해설\s*\n*',
            r'###\s*적용\s*사례\s*\n*',
            r'###\s*주의사항\s*\n*',
            r'###\s*권장사항\s*\n*',
            r'###\s*법률\s*문의\s*답변\s*\n*',
            r'##\s*법률\s*문의\s*답변\s*\n*',
            
            # 섹션 제목 + 내용 패턴
            r'###\s*관련\s*법령\s*\n+\s*관련\s*법령\s*:\s*\n*',
            r'###\s*법령\s*해설\s*\n+\s*법령\s*해설\s*:\s*\n*',
            r'###\s*적용\s*사례\s*\n+\s*실제\s*적용\s*사례\s*:\s*\n*',
            r'###\s*주의사항\s*\n+\s*주의사항\s*:\s*\n*',
            r'###\s*권장사항\s*\n+\s*권장사항\s*:\s*\n*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_placeholders(self, text: str) -> str:
        """플레이스홀더 제거"""
        patterns = [
            r'###\s*법령\s*해설\s*\n+\s*\*쉬운\s*말로\s*풀어서\s*설명\*\s*\n*',
            r'###\s*적용\s*사례\s*\n+\s*\*구체적\s*예시와\s*설명\*\s*\n*',
            r'###\s*주의사항\s*\n+\s*\*법적\s*리스크와\s*제한사항\*\s*\n*',
            r'###\s*권장사항\s*\n+\s*\*추가\s*권장사항\*\s*\n*',
            r'###\s*관련\s*법령\s*\n+\s*\*정확한\s*조문\s*번호와\s*내용\*\s*\n*',
            
            # 일반적인 플레이스홀더 패턴들
            r'###\s*[^\n]+\s*\n+\s*\*[^*]+\*\s*\n*',
            r'###\s*[^\n]+\s*\n+\s*정확한\s*조문\s*번호와\s*내용\s*\n*',
            r'###\s*[^\n]+\s*\n+\s*쉬운\s*말로\s*풀어서\s*설명\s*\n*',
            r'###\s*[^\n]+\s*\n+\s*구체적\s*예시와\s*설명\s*\n*',
            r'###\s*[^\n]+\s*\n+\s*법적\s*리스크와\s*제한사항\s*\n*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_disclaimers(self, text: str) -> str:
        """면책 조항 제거"""
        patterns = [
            r'---\s*\n\s*💼\s*\*\*면책\s*조항\*\*\s*\n\s*#\s*면책\s*조항\s*제거\s*\n\s*#\s*본\s*답변은.*?바랍니다\.\s*\n*',
            r'💼\s*\*\*면책\s*조항\*\*\s*\n\s*#\s*면책\s*조항\s*제거\s*\n\s*#\s*본\s*답변은.*?바랍니다\.\s*\n*',
            r'###\s*면책\s*조항\s*\n\s*#\s*면책\s*조항\s*제거\s*\n\s*#\s*본\s*답변은.*?바랍니다\.\s*\n*',
            
            # 추가 면책 조항 패턴들
            r'본\s*답변은.*?바랍니다\.\s*\n*',
            r'구체적인\s*법률\s*문제는.*?바랍니다\.\s*\n*',
            r'변호사와\s*상담.*?바랍니다\.\s*\n*',
            r'법률\s*전문가와\s*상담.*?바랍니다\.\s*\n*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        return text
    
    def _remove_intro_phrases(self, text: str) -> str:
        """불필요한 서론 제거"""
        patterns = [
            r'(문의하신|질문하신)\s*내용에\s*대해\s*',
            r'관련해서\s*말씀드리면\s*',
            r'질문하신\s*[^에]*에\s*대해\s*',
            r'문의하신\s*[^에]*에\s*대해\s*',
            r'궁금하시군요\.\s*',
            r'궁금하시네요\.\s*',
            r'에\s*대해\s*궁금하시군요\.\s*',
            r'에\s*대해\s*궁금하시네요\.\s*',
            r'질문해\s*주신\s*내용에\s*대해\s*',
            r'문의해\s*주신\s*내용에\s*대해\s*',
            r'말씀드리면\s*',
            r'설명드리면\s*',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text
    
    def _clean_formatting(self, text: str) -> str:
        """포맷팅 정리"""
        # 연속된 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 시작과 끝의 불필요한 공백 제거
        text = text.strip()
        
        # 중복된 제목 제거
        text = re.sub(r'(###+\s*[^\n]+)\s*\n+\s*\1\s*:', r'\1\n\n', text, flags=re.IGNORECASE)
        
        return text
    
    def _validate_content(self, text: str) -> str:
        """내용 검증"""
        # 너무 짧은 응답은 원본 유지
        if len(text.strip()) < 30:
            self.logger.warning("응답이 너무 짧아서 원본을 유지합니다.")
            return text
        
        # 빈 섹션 완전 제거
        text = re.sub(r'###\s*[^\n]+\s*\n+\s*\n+', '', text)
        
        return text
    
    def _determine_quality_grade(self, score: float) -> ResponseQuality:
        """품질 등급 결정"""
        if score >= self.quality_thresholds[ResponseQuality.EXCELLENT]:
            return ResponseQuality.EXCELLENT
        elif score >= self.quality_thresholds[ResponseQuality.GOOD]:
            return ResponseQuality.GOOD
        elif score >= self.quality_thresholds[ResponseQuality.FAIR]:
            return ResponseQuality.FAIR
        else:
            return ResponseQuality.POOR
    
    def get_processing_report(self, result: ProcessingResult) -> str:
        """처리 결과 리포트 생성"""
        report = f"""
=== 응답 후처리 결과 ===
품질 점수: {result.quality_score:.2f}
품질 등급: {result.quality_grade.value}
원본 길이: {result.original_length} 문자
처리 후 길이: {result.processed_length} 문자
단축률: {result.reduction_rate:.1f}%

개선 사항:
"""
        
        if result.improvements_made:
            for i, improvement in enumerate(result.improvements_made, 1):
                report += f"{i}. {improvement}\n"
        else:
            report += "개선 사항 없음\n"
        
        return report


# 전역 인스턴스
advanced_response_processor = AdvancedResponseProcessor()
