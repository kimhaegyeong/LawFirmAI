# -*- coding: utf-8 -*-
"""
Progressive Response System
단계별 답변 제공 시스템
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseLevel(Enum):
    """답변 수준"""
    SUMMARY = "summary"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class LevelConfig:
    """수준별 설정"""
    max_length: int
    description: str
    include_sections: List[str]


@dataclass
class AdditionalOption:
    """추가 정보 옵션"""
    option_type: str
    title: str
    description: str
    estimated_length: int


@dataclass
class ProgressiveResponse:
    """단계별 답변"""
    response: str
    level: str
    additional_options: List[AdditionalOption]
    has_more: bool
    quality_score: float


class ProgressiveResponseSystem:
    """단계별 답변 제공 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 답변 수준별 설정
        self.response_levels = {
            ResponseLevel.SUMMARY: LevelConfig(
                max_length=300,
                description="핵심 요약",
                include_sections=["핵심내용", "결론"]
            ),
            ResponseLevel.STANDARD: LevelConfig(
                max_length=800,
                description="표준 답변",
                include_sections=["핵심내용", "법적근거", "요약"]
            ),
            ResponseLevel.DETAILED: LevelConfig(
                max_length=1500,
                description="상세 답변",
                include_sections=["핵심내용", "법적근거", "실무팁", "참고사항"]
            ),
            ResponseLevel.COMPREHENSIVE: LevelConfig(
                max_length=3000,
                description="종합 답변",
                include_sections=["전체내용"]
            )
        }
        
        # 섹션 추출 패턴
        self.section_patterns = {
            "핵심내용": [
                r'##\s*([^\n]+)',
                r'\*\*핵심[^:]*:\*\*\s*([^\n]+)',
                r'요약[^:]*:\s*([^\n]+)'
            ],
            "법적근거": [
                r'제\d+조[^.]*\.',
                r'민법[^.]*\.',
                r'형법[^.]*\.',
                r'법령[^.]*\.'
            ],
            "실무팁": [
                r'실무[팁|가이드][^.]*\.',
                r'주의[사항|할점][^.]*\.',
                r'권장[사항|방법][^.]*\.'
            ],
            "참고사항": [
                r'참고[^.]*\.',
                r'참조[^.]*\.',
                r'관련[^.]*\.'
            ]
        }
        
        self.logger.info("Progressive Response System 초기화 완료")
    
    def generate_progressive_response(self, full_response: str, user_preference: str = 'standard') -> ProgressiveResponse:
        """단계별 답변 생성"""
        try:
            # 수준 설정 가져오기
            level_config = self.response_levels.get(
                ResponseLevel(user_preference), 
                self.response_levels[ResponseLevel.STANDARD]
            )
            
            # 기본 답변 생성
            base_response = self._extract_base_response(full_response, level_config)
            
            # 품질 점수 계산
            quality_score = self._calculate_quality_score(base_response, full_response)
            
            # 추가 정보 제공 옵션 생성
            additional_options = self._generate_additional_options(full_response, level_config)
            
            # 더 자세한 정보가 있는지 확인
            has_more = len(full_response) > level_config.max_length
            
            return ProgressiveResponse(
                response=base_response,
                level=user_preference,
                additional_options=additional_options,
                has_more=has_more,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Progressive response generation failed: {e}")
            return ProgressiveResponse(
                response=full_response[:300] + "..." if len(full_response) > 300 else full_response,
                level=user_preference,
                additional_options=[],
                has_more=len(full_response) > 300,
                quality_score=0.5
            )
    
    def _extract_base_response(self, full_response: str, level_config: LevelConfig) -> str:
        """기본 답변 추출"""
        try:
            # 섹션별로 내용 추출
            sections = self._extract_sections(full_response)
            
            # 설정된 섹션만 포함
            included_sections = []
            for section_name in level_config.include_sections:
                if section_name in sections and sections[section_name]:
                    included_sections.append(sections[section_name])
            
            # 섹션 결합
            combined_response = "\n\n".join(included_sections)
            
            # 길이 제한 적용
            if len(combined_response) > level_config.max_length:
                combined_response = self._truncate_to_length(combined_response, level_config.max_length)
            
            return combined_response
            
        except Exception as e:
            self.logger.error(f"Base response extraction failed: {e}")
            return full_response[:level_config.max_length] + "..." if len(full_response) > level_config.max_length else full_response
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """텍스트에서 섹션별 내용 추출"""
        try:
            sections = {}
            
            for section_name, patterns in self.section_patterns.items():
                section_content = []
                
                for pattern in patterns:
                    matches = self._find_pattern_matches(text, pattern)
                    section_content.extend(matches)
                
                if section_content:
                    sections[section_name] = "\n".join(section_content[:3])  # 최대 3개
            
            # 전체 내용이 섹션으로 분할되지 않은 경우
            if not sections:
                sections["전체내용"] = text
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Section extraction failed: {e}")
            return {"전체내용": text}
    
    def _find_pattern_matches(self, text: str, pattern: str) -> List[str]:
        """패턴 매칭으로 내용 찾기"""
        try:
            import re
            matches = re.findall(pattern, text)
            return [match.strip() for match in matches if match.strip()]
        except Exception as e:
            self.logger.error(f"Pattern matching failed: {e}")
            return []
    
    def _truncate_to_length(self, text: str, max_length: int) -> str:
        """지정된 길이로 텍스트 자르기"""
        try:
            if len(text) <= max_length:
                return text
            
            # 문장 단위로 자르기
            sentences = text.split('.')
            truncated = ""
            
            for sentence in sentences:
                if len(truncated + sentence + '.') <= max_length - 3:  # "..." 공간 확보
                    truncated += sentence + '.'
                else:
                    break
            
            if truncated:
                return truncated + "..."
            else:
                return text[:max_length-3] + "..."
                
        except Exception as e:
            self.logger.error(f"Text truncation failed: {e}")
            return text[:max_length-3] + "..."
    
    def _calculate_quality_score(self, base_response: str, full_response: str) -> float:
        """품질 점수 계산"""
        try:
            score = 0.0
            
            # 길이 적절성 (30%)
            length_ratio = len(base_response) / len(full_response) if len(full_response) > 0 else 1.0
            if 0.3 <= length_ratio <= 0.8:  # 적절한 길이 비율
                score += 0.3
            elif length_ratio > 0.8:  # 너무 길면 감점
                score += 0.2
            else:  # 너무 짧으면 감점
                score += 0.1
            
            # 구조 완성도 (40%)
            structure_indicators = ['**', '##', '제', '조', '법']
            structure_score = sum(1 for indicator in structure_indicators if indicator in base_response)
            score += min(0.4, structure_score * 0.1)
            
            # 내용 풍부도 (30%)
            content_indicators = ['법령', '조문', '판례', '절차', '방법', '주의', '권장']
            content_score = sum(1 for indicator in content_indicators if indicator in base_response)
            score += min(0.3, content_score * 0.05)
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 0.5
    
    def _generate_additional_options(self, full_response: str, level_config: LevelConfig) -> List[AdditionalOption]:
        """추가 정보 제공 옵션 생성"""
        try:
            options = []
            
            # 더 자세한 설명 옵션
            if level_config.max_length < 1500:
                options.append(AdditionalOption(
                    option_type="more_detail",
                    title="더 자세한 설명 보기",
                    description="법적 근거와 실무 팁을 포함한 상세 답변",
                    estimated_length=1500
                ))
            
            # 관련 판례 옵션
            if any(keyword in full_response for keyword in ['판례', '사건', '재판', '판결']):
                options.append(AdditionalOption(
                    option_type="precedents",
                    title="관련 판례 보기",
                    description="이와 관련된 대법원 판례 및 하급심 판례",
                    estimated_length=1000
                ))
            
            # 실무 가이드 옵션
            if any(keyword in full_response for keyword in ['절차', '방법', '신청', '제출', '등기', '소송']):
                options.append(AdditionalOption(
                    option_type="practical_guide",
                    title="실무 가이드 보기",
                    description="단계별 실무 진행 방법과 주의사항",
                    estimated_length=1200
                ))
            
            # 관련 법령 옵션
            if any(keyword in full_response for keyword in ['민법', '형법', '상법', '행정법']):
                options.append(AdditionalOption(
                    option_type="related_laws",
                    title="관련 법령 보기",
                    description="관련된 다른 법령과 조문 정보",
                    estimated_length=800
                ))
            
            # 예시 사례 옵션
            if any(keyword in full_response for keyword in ['예시', '사례', '상황', '경우']):
                options.append(AdditionalOption(
                    option_type="examples",
                    title="예시 사례 보기",
                    description="구체적인 사례와 해결 방법",
                    estimated_length=1000
                ))
            
            return options
            
        except Exception as e:
            self.logger.error(f"Additional options generation failed: {e}")
            return []
    
    def generate_expanded_response(self, base_response: str, option_type: str, full_response: str) -> str:
        """확장된 답변 생성"""
        try:
            if option_type == "more_detail":
                return self._generate_detailed_response(full_response)
            elif option_type == "precedents":
                return self._generate_precedents_response(full_response)
            elif option_type == "practical_guide":
                return self._generate_practical_guide_response(full_response)
            elif option_type == "related_laws":
                return self._generate_related_laws_response(full_response)
            elif option_type == "examples":
                return self._generate_examples_response(full_response)
            else:
                return base_response
                
        except Exception as e:
            self.logger.error(f"Expanded response generation failed: {e}")
            return base_response
    
    def _generate_detailed_response(self, full_response: str) -> str:
        """상세 답변 생성"""
        try:
            # 전체 내용을 섹션별로 구성
            sections = self._extract_sections(full_response)
            
            detailed_parts = []
            
            if "핵심내용" in sections:
                detailed_parts.append(f"## 핵심 내용\n{sections['핵심내용']}")
            
            if "법적근거" in sections:
                detailed_parts.append(f"## 법적 근거\n{sections['법적근거']}")
            
            if "실무팁" in sections:
                detailed_parts.append(f"## 실무 팁\n{sections['실무팁']}")
            
            if "참고사항" in sections:
                detailed_parts.append(f"## 참고사항\n{sections['참고사항']}")
            
            return "\n\n".join(detailed_parts)
            
        except Exception as e:
            self.logger.error(f"Detailed response generation failed: {e}")
            return full_response
    
    def _generate_precedents_response(self, full_response: str) -> str:
        """판례 관련 답변 생성"""
        try:
            # 판례 관련 내용 추출
            precedent_keywords = ['판례', '사건', '재판', '판결', '대법원', '법원']
            precedent_content = []
            
            sentences = full_response.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in precedent_keywords):
                    precedent_content.append(sentence.strip())
            
            if precedent_content:
                return "## 관련 판례\n\n" + "\n".join(precedent_content[:5])
            else:
                return "## 관련 판례\n\n관련 판례 정보를 찾을 수 없습니다."
                
        except Exception as e:
            self.logger.error(f"Precedents response generation failed: {e}")
            return "관련 판례 정보를 찾을 수 없습니다."
    
    def _generate_practical_guide_response(self, full_response: str) -> str:
        """실무 가이드 답변 생성"""
        try:
            # 실무 관련 내용 추출
            practical_keywords = ['절차', '방법', '신청', '제출', '등기', '소송', '주의', '권장']
            practical_content = []
            
            sentences = full_response.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in practical_keywords):
                    practical_content.append(sentence.strip())
            
            if practical_content:
                return "## 실무 가이드\n\n" + "\n".join(practical_content[:5])
            else:
                return "## 실무 가이드\n\n실무 가이드 정보를 찾을 수 없습니다."
                
        except Exception as e:
            self.logger.error(f"Practical guide response generation failed: {e}")
            return "실무 가이드 정보를 찾을 수 없습니다."
    
    def _generate_related_laws_response(self, full_response: str) -> str:
        """관련 법령 답변 생성"""
        try:
            # 법령 관련 내용 추출
            law_keywords = ['민법', '형법', '상법', '행정법', '제', '조', '법령']
            law_content = []
            
            sentences = full_response.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in law_keywords):
                    law_content.append(sentence.strip())
            
            if law_content:
                return "## 관련 법령\n\n" + "\n".join(law_content[:5])
            else:
                return "## 관련 법령\n\n관련 법령 정보를 찾을 수 없습니다."
                
        except Exception as e:
            self.logger.error(f"Related laws response generation failed: {e}")
            return "관련 법령 정보를 찾을 수 없습니다."
    
    def _generate_examples_response(self, full_response: str) -> str:
        """예시 사례 답변 생성"""
        try:
            # 예시 관련 내용 추출
            example_keywords = ['예시', '사례', '상황', '경우', '예를 들어']
            example_content = []
            
            sentences = full_response.split('.')
            for sentence in sentences:
                if any(keyword in sentence for keyword in example_keywords):
                    example_content.append(sentence.strip())
            
            if example_content:
                return "## 예시 사례\n\n" + "\n".join(example_content[:5])
            else:
                return "## 예시 사례\n\n예시 사례 정보를 찾을 수 없습니다."
                
        except Exception as e:
            self.logger.error(f"Examples response generation failed: {e}")
            return "예시 사례 정보를 찾을 수 없습니다."
