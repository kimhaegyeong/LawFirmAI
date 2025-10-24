# -*- coding: utf-8 -*-
"""
Adaptive Response Manager
사용자 환경에 맞는 적응형 답변 길이 관리
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UserType(Enum):
    """사용자 유형"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    EXPERT = "expert"


class ResponseLevel(Enum):
    """답변 수준"""
    SUMMARY = "summary"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ResponseTemplate:
    """답변 템플릿"""
    max_length: int
    structure: str
    sections: List[str]
    priority_order: List[str]


@dataclass
class CompressedResponse:
    """압축된 응답"""
    content: str
    sections: Dict[str, str]
    compression_ratio: float
    quality_score: float


class AdaptiveResponseManager:
    """사용자 환경에 맞는 적응형 답변 길이 관리"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 길이 템플릿 설정
        self.length_templates = {
            UserType.MOBILE: ResponseTemplate(
                max_length=600,
                structure="compact",
                sections=["핵심내용", "요약", "참고사항"],
                priority_order=["핵심내용", "요약", "참고사항"]
            ),
            UserType.DESKTOP: ResponseTemplate(
                max_length=1200,
                structure="detailed",
                sections=["상세설명", "법적근거", "실무팁", "참고사항"],
                priority_order=["상세설명", "법적근거", "실무팁", "참고사항"]
            ),
            UserType.EXPERT: ResponseTemplate(
                max_length=2500,
                structure="comprehensive",
                sections=["법적분석", "판례참조", "실무가이드", "주의사항"],
                priority_order=["법적분석", "판례참조", "실무가이드", "주의사항"]
            )
        }
        
        # 답변 수준별 템플릿
        self.level_templates = {
            ResponseLevel.SUMMARY: ResponseTemplate(
                max_length=300,
                structure="summary",
                sections=["핵심내용", "결론"],
                priority_order=["핵심내용", "결론"]
            ),
            ResponseLevel.STANDARD: ResponseTemplate(
                max_length=800,
                structure="standard",
                sections=["핵심내용", "법적근거", "요약"],
                priority_order=["핵심내용", "법적근거", "요약"]
            ),
            ResponseLevel.DETAILED: ResponseTemplate(
                max_length=1500,
                structure="detailed",
                sections=["핵심내용", "법적근거", "실무팁", "참고사항"],
                priority_order=["핵심내용", "법적근거", "실무팁", "참고사항"]
            ),
            ResponseLevel.COMPREHENSIVE: ResponseTemplate(
                max_length=3000,
                structure="comprehensive",
                sections=["전체내용"],
                priority_order=["전체내용"]
            )
        }
        
        # 압축 전략 설정
        self.compression_strategies = {
            'legal_citations': self._compress_legal_citations,
            'example_cases': self._compress_examples,
            'procedural_steps': self._compress_procedures,
            'redundant_text': self._remove_redundancy,
            'long_paragraphs': self._compress_paragraphs
        }
        
        self.logger.info("Adaptive Response Manager 초기화 완료")
    
    def adapt_response_length(self, response: str, user_context: Dict[str, Any]) -> str:
        """사용자 컨텍스트에 맞게 답변 길이 조정"""
        try:
            # 사용자 유형 결정
            user_type = self._determine_user_type(user_context)
            template = self.length_templates.get(user_type, self.length_templates[UserType.DESKTOP])
            
            # 현재 길이 확인
            if len(response) <= template.max_length:
                return response
            
            # 답변 구조화 및 압축
            structured_response = self._structure_response(response, template)
            compressed_response = self._compress_response(structured_response, template.max_length)
            
            return compressed_response
            
        except Exception as e:
            self.logger.error(f"Response length adaptation failed: {e}")
            return response
    
    def _determine_user_type(self, user_context: Dict[str, Any]) -> UserType:
        """사용자 유형 결정"""
        try:
            # 디바이스 정보 확인
            device_info = user_context.get('device_info', {})
            if device_info.get('type') == 'mobile':
                return UserType.MOBILE
            
            # 전문성 수준 확인
            expertise_level = user_context.get('expertise_level', 'beginner')
            if expertise_level in ['expert', 'professional']:
                return UserType.EXPERT
            
            # 기본값
            return UserType.DESKTOP
            
        except Exception as e:
            self.logger.error(f"User type determination failed: {e}")
            return UserType.DESKTOP
    
    def _structure_response(self, response: str, template: ResponseTemplate) -> Dict[str, str]:
        """답변을 구조화된 섹션으로 분할"""
        try:
            sections = {}
            
            # 핵심 내용 추출
            sections['핵심내용'] = self._extract_key_points(response)
            
            # 법적 근거 추출
            sections['법적근거'] = self._extract_legal_basis(response)
            
            # 실무 팁 추출
            sections['실무팁'] = self._extract_practical_tips(response)
            
            # 참고사항 추출
            sections['참고사항'] = self._extract_references(response)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Response structuring failed: {e}")
            return {'전체내용': response}
    
    def _extract_key_points(self, text: str) -> str:
        """핵심 내용 추출"""
        try:
            # 제목이나 핵심 문장 추출
            patterns = [
                r'##\s*([^\n]+)',  # 마크다운 제목
                r'\*\*([^*]+)\*\*',  # 굵은 글씨
                r'핵심[은는이]?\s*:?\s*([^\n]+)',
                r'요약[은는이]?\s*:?\s*([^\n]+)',
                r'결론[은는이]?\s*:?\s*([^\n]+)'
            ]
            
            key_points = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                key_points.extend(matches)
            
            if key_points:
                return '\n'.join(key_points[:3])  # 최대 3개
            
            # 첫 번째 문단 추출
            paragraphs = text.split('\n\n')
            if paragraphs:
                return paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0]
            
            return text[:200] + "..." if len(text) > 200 else text
            
        except Exception as e:
            self.logger.error(f"Key points extraction failed: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def _extract_legal_basis(self, text: str) -> str:
        """법적 근거 추출"""
        try:
            # 법률 조문 패턴
            patterns = [
                r'제\d+조[^.]*\.',
                r'민법[^.]*\.',
                r'형법[^.]*\.',
                r'상법[^.]*\.',
                r'법령[^.]*\.',
                r'규정[^.]*\.'
            ]
            
            legal_basis = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                legal_basis.extend(matches)
            
            if legal_basis:
                return '\n'.join(legal_basis[:2])  # 최대 2개
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Legal basis extraction failed: {e}")
            return ""
    
    def _extract_practical_tips(self, text: str) -> str:
        """실무 팁 추출"""
        try:
            # 실무 팁 패턴
            patterns = [
                r'실무[팁|가이드][^.]*\.',
                r'주의[사항|할점][^.]*\.',
                r'권장[사항|방법][^.]*\.',
                r'팁[^.]*\.',
                r'조언[^.]*\.'
            ]
            
            tips = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                tips.extend(matches)
            
            if tips:
                return '\n'.join(tips[:2])  # 최대 2개
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Practical tips extraction failed: {e}")
            return ""
    
    def _extract_references(self, text: str) -> str:
        """참고사항 추출"""
        try:
            # 참고사항 패턴
            patterns = [
                r'참고[^.]*\.',
                r'참조[^.]*\.',
                r'관련[^.]*\.',
                r'추가[^.]*\.',
                r'※[^.]*\.'
            ]
            
            references = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                references.extend(matches)
            
            if references:
                return '\n'.join(references[:2])  # 최대 2개
            
            return ""
            
        except Exception as e:
            self.logger.error(f"References extraction failed: {e}")
            return ""
    
    def _compress_response(self, sections: Dict[str, str], max_length: int) -> str:
        """섹션별로 압축하여 최대 길이 내로 조정"""
        try:
            compressed = {}
            remaining_length = max_length
            
            # 우선순위별로 압축
            priority_order = ["핵심내용", "법적근거", "실무팁", "참고사항"]
            
            for section in priority_order:
                if section in sections and sections[section]:
                    content = sections[section]
                    if len(content) <= remaining_length:
                        compressed[section] = content
                        remaining_length -= len(content)
                    else:
                        # 압축 필요
                        compressed[section] = self._smart_truncate(content, remaining_length)
                        break
            
            return self._format_compressed_response(compressed)
            
        except Exception as e:
            self.logger.error(f"Response compression failed: {e}")
            return sections.get('핵심내용', '')
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """스마트 텍스트 자르기"""
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
            self.logger.error(f"Smart truncation failed: {e}")
            return text[:max_length-3] + "..."
    
    def _format_compressed_response(self, compressed: Dict[str, str]) -> str:
        """압축된 응답 포맷팅"""
        try:
            formatted_parts = []
            
            for section, content in compressed.items():
                if content:
                    if section == "핵심내용":
                        formatted_parts.append(f"**{section}:**\n{content}")
                    else:
                        formatted_parts.append(f"**{section}:**\n{content}")
            
            return "\n\n".join(formatted_parts)
            
        except Exception as e:
            self.logger.error(f"Response formatting failed: {e}")
            return "\n".join(compressed.values())
    
    def _compress_legal_citations(self, text: str, target_length: int) -> str:
        """법적 인용문 압축"""
        try:
            # 긴 법적 인용문을 요약형으로 변환
            pattern = r'제\d+조[^.]*\.'
            matches = re.findall(pattern, text)
            
            for match in matches:
                if len(match) > 100:
                    # 핵심 내용만 추출
                    compressed = self._extract_key_legal_content(match)
                    text = text.replace(match, compressed)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Legal citations compression failed: {e}")
            return text
    
    def _compress_examples(self, text: str, target_length: int) -> str:
        """예시 압축"""
        try:
            # 예시 섹션을 요약형으로 변환
            example_pattern = r'### \d+\. 상황:.*?(?=###|\Z)'
            examples = re.findall(example_pattern, text, re.DOTALL)
            
            if len(examples) > 2:  # 예시가 2개 이상이면 압축
                # 첫 번째 예시만 상세히, 나머지는 요약
                compressed_examples = [examples[0]]
                for example in examples[1:]:
                    compressed_examples.append(self._summarize_example(example))
                
                # 원본 텍스트에서 예시 부분 교체
                for i, example in enumerate(examples):
                    text = text.replace(example, compressed_examples[i])
            
            return text
            
        except Exception as e:
            self.logger.error(f"Examples compression failed: {e}")
            return text
    
    def _compress_procedures(self, text: str, target_length: int) -> str:
        """절차 압축"""
        try:
            # 절차 섹션 압축
            procedure_pattern = r'\d+\.\s*[^.]*\.'
            procedures = re.findall(procedure_pattern, text)
            
            if len(procedures) > 5:  # 절차가 5개 이상이면 압축
                # 중요한 절차만 유지
                important_procedures = procedures[:3]  # 처음 3개만 유지
                compressed_text = text
                
                for procedure in procedures[3:]:
                    compressed_text = compressed_text.replace(procedure, "")
                
                return compressed_text
            
            return text
            
        except Exception as e:
            self.logger.error(f"Procedures compression failed: {e}")
            return text
    
    def _remove_redundancy(self, text: str, target_length: int) -> str:
        """중복 텍스트 제거"""
        try:
            # 중복 문장 제거
            sentences = text.split('.')
            unique_sentences = []
            seen_sentences = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen_sentences:
                    unique_sentences.append(sentence)
                    seen_sentences.add(sentence)
            
            return '. '.join(unique_sentences) + '.'
            
        except Exception as e:
            self.logger.error(f"Redundancy removal failed: {e}")
            return text
    
    def _compress_paragraphs(self, text: str, target_length: int) -> str:
        """긴 문단 압축"""
        try:
            paragraphs = text.split('\n\n')
            compressed_paragraphs = []
            
            for paragraph in paragraphs:
                if len(paragraph) > 200:
                    # 긴 문단을 압축
                    compressed_paragraph = self._smart_truncate(paragraph, 200)
                    compressed_paragraphs.append(compressed_paragraph)
                else:
                    compressed_paragraphs.append(paragraph)
            
            return '\n\n'.join(compressed_paragraphs)
            
        except Exception as e:
            self.logger.error(f"Paragraph compression failed: {e}")
            return text
    
    def _extract_key_legal_content(self, legal_text: str) -> str:
        """법적 내용에서 핵심만 추출"""
        try:
            # 핵심 키워드 추출
            keywords = ['고의', '과실', '손해', '배상', '책임', '위법', '행위']
            
            sentences = legal_text.split('.')
            key_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in keywords):
                    key_sentences.append(sentence.strip())
            
            if key_sentences:
                return '. '.join(key_sentences[:2]) + '.'
            
            return legal_text[:100] + "..."
            
        except Exception as e:
            self.logger.error(f"Key legal content extraction failed: {e}")
            return legal_text[:100] + "..."
    
    def _summarize_example(self, example: str) -> str:
        """예시 요약"""
        try:
            # 상황 부분만 추출
            situation_match = re.search(r'상황:\s*([^분석]+)', example)
            if situation_match:
                situation = situation_match.group(1).strip()
                return f"상황: {situation[:100]}..."
            
            return example[:150] + "..."
            
        except Exception as e:
            self.logger.error(f"Example summarization failed: {e}")
            return example[:150] + "..."

