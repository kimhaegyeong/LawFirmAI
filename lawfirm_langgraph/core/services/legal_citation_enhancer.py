# -*- coding: utf-8 -*-
"""
법령 인용 강화 시스템
법령 조문, 판례 번호 등을 자동으로 추출하고 인용 형식을 표준화
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LegalCitation:
    """법적 인용 정보"""
    type: str  # law_article, precedent, court_decision 등
    text: str  # 원본 텍스트
    formatted_text: str  # 형식화된 텍스트
    position: Tuple[int, int]  # 텍스트 내 위치
    context: str  # 주변 맥락
    confidence: float  # 신뢰도
    metadata: Dict[str, Any]  # 추가 메타데이터


class LegalCitationEnhancer:
    """법령 인용 강화 시스템"""
    
    def __init__(self):
        """초기화"""
        self.citation_patterns = self._load_citation_patterns()
        self.formatting_templates = self._load_formatting_templates()
        self.logger = logging.getLogger(__name__)
        
    def _load_citation_patterns(self) -> Dict[str, re.Pattern]:
        """인용 패턴 로드"""
        return {
            # 법령 조문 패턴
            'law_articles': re.compile(r'제\s*(\d+)\s*조\s*(?:\(([^)]+)\))?', re.IGNORECASE),
            'law_paragraphs': re.compile(r'제\s*(\d+)\s*항', re.IGNORECASE),
            'law_subparagraphs': re.compile(r'제\s*(\d+)\s*호', re.IGNORECASE),
            'law_items': re.compile(r'제\s*([가-힣])\s*목', re.IGNORECASE),
            
            # 판례 패턴
            'precedent_numbers': re.compile(r'(\d{4}[가-힣]\d+)', re.IGNORECASE),
            'court_decisions': re.compile(r'([가-힣]+법원\s*\d{4}[가-힣]\d+)', re.IGNORECASE),
            'supreme_court': re.compile(r'(대법원\s*\d{4}[가-힣]\d+)', re.IGNORECASE),
            'high_court': re.compile(r'([가-힣]+고등법원\s*\d{4}[가-힣]\d+)', re.IGNORECASE),
            
            # 법률명 패턴
            'law_names': re.compile(r'([가-힣]+법(?:\s*시행령|\s*시행규칙)?)', re.IGNORECASE),
            'constitution': re.compile(r'(헌법\s*제\s*\d+\s*조)', re.IGNORECASE),
            
            # 날짜 패턴
            'dates': re.compile(r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)', re.IGNORECASE),
            'amendment_dates': re.compile(r'(개정\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일)', re.IGNORECASE),
            
            # 특별 조항 패턴
            'penalty_clauses': re.compile(r'제\s*(\d+)\s*조\s*\(벌칙\)', re.IGNORECASE),
            'transitional_clauses': re.compile(r'제\s*(\d+)\s*조\s*\(경과조치\)', re.IGNORECASE),
            'enforcement_clauses': re.compile(r'제\s*(\d+)\s*조\s*\(시행\)', re.IGNORECASE),
        }
    
    def _load_formatting_templates(self) -> Dict[str, str]:
        """형식화 템플릿 로드"""
        return {
            'law_articles': "**제{number}조** ({title})",
            'law_paragraphs': "**제{number}항**",
            'law_subparagraphs': "**제{number}호**",
            'law_items': "**제{letter}목**",
            'precedent_numbers': "**{number}** (관련 판례)",
            'court_decisions': "**{court}** (법원 판결)",
            'supreme_court': "**{decision}** (대법원 판결)",
            'law_names': "**{law_name}**",
            'constitution': "**{article}** (헌법 조문)",
            'penalty_clauses': "**제{number}조 (벌칙)**",
            'transitional_clauses': "**제{number}조 (경과조치)**",
            'enforcement_clauses': "**제{number}조 (시행)**",
        }
    
    def extract_citations(self, text: str) -> List[LegalCitation]:
        """텍스트에서 법적 인용 추출"""
        citations = []
        
        for pattern_name, pattern in self.citation_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                try:
                    citation = self._create_citation(
                        pattern_name, match, text
                    )
                    if citation:
                        citations.append(citation)
                except Exception as e:
                    self.logger.warning(f"Failed to create citation for {pattern_name}: {e}")
                    continue
        
        # 중복 제거 및 정렬
        citations = self._deduplicate_citations(citations)
        citations.sort(key=lambda x: x.position[0])
        
        return citations
    
    def _create_citation(self, pattern_name: str, match: re.Match, text: str) -> Optional[LegalCitation]:
        """인용 객체 생성"""
        try:
            # 기본 정보 추출
            original_text = match.group(0)
            position = match.span()
            context = self._get_context(text, position)
            
            # 패턴별 특별 처리
            formatted_text = self._format_citation(pattern_name, match)
            confidence = self._calculate_confidence(pattern_name, match, context)
            metadata = self._extract_metadata(pattern_name, match)
            
            return LegalCitation(
                type=pattern_name,
                text=original_text,
                formatted_text=formatted_text,
                position=position,
                context=context,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating citation: {e}")
            return None
    
    def _format_citation(self, pattern_name: str, match: re.Match) -> str:
        """인용 형식화"""
        template = self.formatting_templates.get(pattern_name, "**{text}**")
        
        try:
            if pattern_name == 'law_articles':
                number = match.group(1)
                title = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                return template.format(number=number, title=title)
            
            elif pattern_name in ['law_paragraphs', 'law_subparagraphs']:
                number = match.group(1)
                return template.format(number=number)
            
            elif pattern_name == 'law_items':
                letter = match.group(1)
                return template.format(letter=letter)
            
            elif pattern_name == 'precedent_numbers':
                number = match.group(1)
                return template.format(number=number)
            
            elif pattern_name == 'court_decisions':
                court = match.group(1)
                return template.format(court=court)
            
            elif pattern_name == 'supreme_court':
                decision = match.group(1)
                return template.format(decision=decision)
            
            elif pattern_name == 'law_names':
                law_name = match.group(1)
                return template.format(law_name=law_name)
            
            elif pattern_name == 'constitution':
                article = match.group(1)
                return template.format(article=article)
            
            elif pattern_name in ['penalty_clauses', 'transitional_clauses', 'enforcement_clauses']:
                number = match.group(1)
                return template.format(number=number)
            
            else:
                return template.format(text=match.group(0))
                
        except Exception as e:
            self.logger.warning(f"Formatting error for {pattern_name}: {e}")
            return f"**{match.group(0)}**"
    
    def _get_context(self, text: str, position: Tuple[int, int], context_length: int = 50) -> str:
        """인용 주변 맥락 추출"""
        start, end = position
        context_start = max(0, start - context_length)
        context_end = min(len(text), end + context_length)
        
        context = text[context_start:context_end]
        
        # 문장 경계에서 자르기
        if context_start > 0:
            sentence_start = context.find('.', 0, start - context_start)
            if sentence_start != -1:
                context = context[sentence_start + 1:]
        
        if context_end < len(text):
            sentence_end = context.find('.', end - context_start)
            if sentence_end != -1:
                context = context[:sentence_end + 1]
        
        return context.strip()
    
    def _calculate_confidence(self, pattern_name: str, match: re.Match, context: str) -> float:
        """신뢰도 계산"""
        base_confidence = 0.8
        
        # 패턴별 가중치
        pattern_weights = {
            'law_articles': 0.95,
            'precedent_numbers': 0.9,
            'supreme_court': 0.95,
            'constitution': 0.98,
            'law_names': 0.85,
            'court_decisions': 0.9,
        }
        
        weight = pattern_weights.get(pattern_name, 0.8)
        
        # 맥락 기반 조정
        context_indicators = ['법률', '판례', '조문', '항', '호', '목', '법원', '판결']
        context_bonus = sum(0.05 for indicator in context_indicators if indicator in context)
        
        # 길이 기반 조정
        length_penalty = max(0, (len(match.group(0)) - 20) * 0.01)
        
        final_confidence = min(1.0, weight + context_bonus - length_penalty)
        return round(final_confidence, 2)
    
    def _extract_metadata(self, pattern_name: str, match: re.Match) -> Dict[str, Any]:
        """메타데이터 추출"""
        metadata = {
            'pattern_name': pattern_name,
            'extracted_at': datetime.now().isoformat(),
            'groups': match.groups(),
        }
        
        if pattern_name == 'law_articles':
            metadata.update({
                'article_number': int(match.group(1)),
                'article_title': match.group(2) if len(match.groups()) > 1 else None,
            })
        
        elif pattern_name == 'precedent_numbers':
            metadata.update({
                'precedent_number': match.group(1),
                'year': match.group(1)[:4] if len(match.group(1)) >= 4 else None,
            })
        
        elif pattern_name == 'law_names':
            metadata.update({
                'law_name': match.group(1),
                'is_enforcement_regulation': '시행령' in match.group(1),
                'is_enforcement_rule': '시행규칙' in match.group(1),
            })
        
        return metadata
    
    def _deduplicate_citations(self, citations: List[LegalCitation]) -> List[LegalCitation]:
        """중복 인용 제거"""
        seen = set()
        unique_citations = []
        
        for citation in citations:
            # 위치와 텍스트로 중복 판단
            key = (citation.position, citation.text)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def enhance_text_with_citations(self, text: str) -> Dict[str, Any]:
        """텍스트를 법적 인용으로 강화"""
        citations = self.extract_citations(text)
        
        # 인용별 통계
        citation_stats = {}
        for citation in citations:
            citation_type = citation.type
            citation_stats[citation_type] = citation_stats.get(citation_type, 0) + 1
        
        # 신뢰도별 분류
        high_confidence = [c for c in citations if c.confidence >= 0.9]
        medium_confidence = [c for c in citations if 0.7 <= c.confidence < 0.9]
        low_confidence = [c for c in citations if c.confidence < 0.7]
        
        return {
            'original_text': text,
            'citations': citations,
            'citation_count': len(citations),
            'citation_stats': citation_stats,
            'confidence_distribution': {
                'high': len(high_confidence),
                'medium': len(medium_confidence),
                'low': len(low_confidence)
            },
            'enhanced_text': self._apply_citations_to_text(text, citations),
            'legal_basis_summary': self._generate_legal_basis_summary(citations)
        }
    
    def _apply_citations_to_text(self, text: str, citations: List[LegalCitation]) -> str:
        """텍스트에 인용 형식 적용"""
        enhanced_text = text
        
        # 뒤에서부터 처리하여 위치 변경 방지
        for citation in sorted(citations, key=lambda x: x.position[0], reverse=True):
            start, end = citation.position
            original = enhanced_text[start:end]
            enhanced_text = enhanced_text[:start] + citation.formatted_text + enhanced_text[end:]
        
        return enhanced_text
    
    def _generate_legal_basis_summary(self, citations: List[LegalCitation]) -> Dict[str, Any]:
        """법적 근거 요약 생성"""
        summary = {
            'laws_referenced': [],
            'precedents_referenced': [],
            'court_decisions': [],
            'constitutional_articles': [],
            'total_citations': len(citations)
        }
        
        for citation in citations:
            if citation.type == 'law_articles':
                summary['laws_referenced'].append({
                    'article': citation.text,
                    'formatted': citation.formatted_text,
                    'confidence': citation.confidence
                })
            elif citation.type in ['precedent_numbers', 'supreme_court', 'high_court']:
                summary['precedents_referenced'].append({
                    'precedent': citation.text,
                    'formatted': citation.formatted_text,
                    'confidence': citation.confidence
                })
            elif citation.type == 'court_decisions':
                summary['court_decisions'].append({
                    'decision': citation.text,
                    'formatted': citation.formatted_text,
                    'confidence': citation.confidence
                })
            elif citation.type == 'constitution':
                summary['constitutional_articles'].append({
                    'article': citation.text,
                    'formatted': citation.formatted_text,
                    'confidence': citation.confidence
                })
        
        return summary
    
    def validate_citations(self, citations: List[LegalCitation]) -> Dict[str, Any]:
        """인용 유효성 검증"""
        validation_result = {
            'total_citations': len(citations),
            'valid_citations': 0,
            'invalid_citations': 0,
            'validation_details': []
        }
        
        for citation in citations:
            is_valid = self._validate_single_citation(citation)
            
            if is_valid:
                validation_result['valid_citations'] += 1
            else:
                validation_result['invalid_citations'] += 1
            
            validation_result['validation_details'].append({
                'citation': citation.text,
                'type': citation.type,
                'is_valid': is_valid,
                'confidence': citation.confidence,
                'issues': self._identify_citation_issues(citation)
            })
        
        return validation_result
    
    def _validate_single_citation(self, citation: LegalCitation) -> bool:
        """개별 인용 유효성 검증"""
        # 기본 검증 규칙
        if citation.confidence < 0.5:
            return False
        
        if len(citation.text.strip()) < 3:
            return False
        
        # 패턴별 특별 검증
        if citation.type == 'law_articles':
            # 조문 번호가 합리적인 범위인지 확인
            article_num = citation.metadata.get('article_number')
            if article_num and (article_num < 1 or article_num > 10000):
                return False
        
        elif citation.type == 'precedent_numbers':
            # 판례 번호 형식 확인
            precedent_num = citation.metadata.get('precedent_number', '')
            if len(precedent_num) < 8:  # 최소 길이 확인
                return False
        
        return True
    
    def _identify_citation_issues(self, citation: LegalCitation) -> List[str]:
        """인용 문제점 식별"""
        issues = []
        
        if citation.confidence < 0.7:
            issues.append("낮은 신뢰도")
        
        if len(citation.text.strip()) < 5:
            issues.append("텍스트가 너무 짧음")
        
        if not citation.context.strip():
            issues.append("맥락 정보 부족")
        
        return issues
