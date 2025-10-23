# -*- coding: utf-8 -*-
"""
답변 구조화 개선 시스템
일관된 형식의 구조화된 답변 제공
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .question_classifier import QuestionType
from .confidence_calculator import ConfidenceInfo

logger = logging.getLogger(__name__)


@dataclass
class FormattedAnswer:
    """구조화된 답변"""
    formatted_content: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]


class AnswerFormatter:
    """답변 구조화기 - 템플릿 완전 제거"""
    
    def __init__(self):
        """답변 구조화기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 모든 템플릿 제거 - 자연스러운 답변만 생성
        self.templates = {}
        
        # 이모지 매핑 (강화된 시각적 요소)
        self.emoji_map = {
            "analysis": "🔍",
            "precedents": "📋",
            "laws": "⚖️",
            "confidence": "💡",
            "explanation": "📖",
            "examples": "💼",
            "advice": "🎯",
            "steps": "📝",
            "overview": "📊",
            "documents": "📄",
            "timeline": "⏰",
            "definition": "📚",
            "related": "🔗",
            "answer": "💬",
            "sources": "📚",
            # 추가된 구조화 이모지
            "warnings": "⚠️",
            "recommendations": "💡",
            "important": "❗",
            "checklist": "✅",
            "caution": "🚨"
        }
    
    def format_answer(self, 
                     raw_answer: str,
                     question_type: QuestionType,
                     sources: Dict[str, List[Dict[str, Any]]],
                     confidence: ConfidenceInfo) -> FormattedAnswer:
        """
        간결하고 자연스러운 답변 포맷팅
        
        Args:
            raw_answer: 원본 답변
            question_type: 질문 유형
            sources: 검색된 소스들
            confidence: 신뢰도 정보
            
        Returns:
            FormattedAnswer: 구조화된 답변
        """
        try:
            self.logger.info(f"Formatting answer for question type: {question_type.value}")
            
            # 원본 답변이 이미 잘 구성되어 있으면 그대로 사용
            if self._is_clean_answer(raw_answer):
                formatted_content = raw_answer
                sections = {"answer": raw_answer}
            else:
                # 간단한 정리만 수행
                formatted_content = self._clean_answer(raw_answer)
                sections = {"answer": formatted_content}
            
            # 메타데이터 생성
            metadata = {
                "question_type": question_type.value,
                "confidence_level": confidence.reliability_level,
                "confidence_score": confidence.confidence,
                "source_count": {
                    "laws": len(self._extract_sources(sources, "law_results")),
                    "precedents": len(self._extract_sources(sources, "precedent_results"))
                },
                "sections_count": len(sections)
            }
            
            result = FormattedAnswer(
                formatted_content=formatted_content,
                sections=sections,
                metadata=metadata
            )
            
            self.logger.info(f"Answer formatted successfully: {len(formatted_content)} chars")
            return result
            
        except Exception as e:
            self.logger.error(f"Error formatting answer: {e}")
            return self._create_fallback_answer(raw_answer, confidence)
    
    def _format_precedent_answer(self, 
                                answer: str, 
                                sources: Dict[str, List[Dict[str, Any]]], 
                                confidence: ConfidenceInfo) -> Dict[str, str]:
        """판례 답변 구조화"""
        try:
            sections = {}
            
            # 분석 섹션
            sections["analysis"] = self._clean_and_structure_text(answer)
            
            # 판례 섹션
            precedents = self._extract_sources(sources, "precedent_results")
            if precedents:
                sections["precedents"] = self._format_precedent_sources(precedents)
            else:
                sections["precedents"] = "관련 판례를 찾을 수 없습니다."
            
            # 법률 섹션
            laws = self._extract_sources(sources, "law_results")
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            else:
                sections["laws"] = "관련 법률을 찾을 수 없습니다."
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting precedent answer: {e}")
            return {"analysis": answer}
    
    def _format_law_explanation(self, 
                               answer: str, 
                               sources: Dict[str, List[Dict[str, Any]]], 
                               confidence: ConfidenceInfo) -> Dict[str, str]:
        """법률 해설 구조화"""
        try:
            sections = {}
            
            # 해설 섹션
            sections["explanation"] = self._clean_and_structure_text(answer)
            
            # 법률 섹션
            laws = self._extract_sources(sources, "law_results")
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            else:
                sections["laws"] = "관련 법률을 찾을 수 없습니다."
            
            # 예시 섹션 (간단한 예시 추가)
            sections["examples"] = self._generate_law_examples(answer, laws)
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting law explanation: {e}")
            return {"explanation": answer}
    
    def _format_legal_advice(self, 
                            answer: str, 
                            sources: Dict[str, List[Dict[str, Any]]], 
                            confidence: ConfidenceInfo) -> Dict[str, str]:
        """법적 조언 구조화 (강화된 불릿 포인트)"""
        try:
            sections = {}
            
            # 조언 섹션 (강화)
            sections["advice"] = self._clean_and_structure_text(answer)
            
            # 법률 섹션
            laws = self._extract_sources(sources, "law_results")
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            else:
                sections["laws"] = "관련 법률을 찾을 수 없습니다."
            
            # 판례 섹션
            precedents = self._extract_sources(sources, "precedent_results")
            if precedents:
                sections["precedents"] = self._format_precedent_sources(precedents)
            else:
                sections["precedents"] = "관련 판례를 찾을 수 없습니다."
            
            # 단계별 가이드 섹션 (강화)
            sections["steps"] = self._extract_steps_from_answer(answer)
            
            # 주의사항 및 권장사항 추가
            warnings_recs = self._extract_warnings_and_recommendations(answer)
            sections["warnings"] = warnings_recs["warnings"]
            sections["recommendations"] = warnings_recs["recommendations"]
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting legal advice: {e}")
            return {"advice": answer}
    
    def _format_procedure_guide(self, 
                               answer: str, 
                               sources: Dict[str, List[Dict[str, Any]]], 
                               confidence: ConfidenceInfo) -> Dict[str, str]:
        """절차 안내 구조화 (강화된 단계별 세분화)"""
        try:
            sections = {}
            
            # 개요 섹션 (강화)
            sections["overview"] = self._extract_enhanced_overview(answer)
            
            # 단계별 절차 (강화)
            sections["steps"] = self._extract_enhanced_steps(answer)
            
            # 필요 서류 (강화)
            sections["documents"] = self._extract_enhanced_documents(answer)
            
            # 처리 기간 (강화)
            sections["timeline"] = self._extract_enhanced_timeline(answer)
            
            # 주의사항 및 권장사항 추가
            warnings_recs = self._extract_warnings_and_recommendations(answer)
            sections["warnings"] = warnings_recs["warnings"]
            sections["recommendations"] = warnings_recs["recommendations"]
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting procedure guide: {e}")
            return {"overview": answer}
    
    def _extract_enhanced_overview(self, answer: str) -> str:
        """강화된 개요 추출"""
        try:
            # 첫 번째 문단을 개요로 사용하되 더 구조화
            paragraphs = answer.split('\n\n')
            if paragraphs:
                overview = paragraphs[0].strip()
                
                # 개요에 핵심 키워드 강조 추가
                enhanced_overview = f"""
### 📊 절차 개요
{overview}

### 🎯 핵심 포인트
{self._extract_key_points(overview)}
"""
                return enhanced_overview
            
            return f"### 📊 절차 개요\n{answer[:300]}{'...' if len(answer) > 300 else ''}"
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced overview: {e}")
            return answer
    
    def _extract_enhanced_steps(self, answer: str) -> str:
        """강화된 단계별 절차 추출"""
        try:
            # 기존 단계 추출 메서드 사용
            basic_steps = self._extract_steps_from_answer(answer)
            
            # 추가적인 단계 정보 추출
            additional_info = self._extract_step_details(answer)
            
            enhanced_steps = f"""
### 📝 단계별 절차

{basic_steps}

{additional_info}
"""
            return enhanced_steps
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced steps: {e}")
            return self._extract_steps_from_answer(answer)
    
    def _extract_enhanced_documents(self, answer: str) -> str:
        """강화된 필요 서류 추출"""
        try:
            # 기본 서류 추출
            basic_docs = self._extract_documents_from_answer(answer)
            
            # 서류별 상세 정보 추출
            doc_details = self._extract_document_details(answer)
            
            enhanced_docs = f"""
### 📄 필요 서류

{basic_docs}

### 📋 서류별 상세 정보
{doc_details}
"""
            return enhanced_docs
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced documents: {e}")
            return self._extract_documents_from_answer(answer)
    
    def _extract_enhanced_timeline(self, answer: str) -> str:
        """강화된 처리 기간 추출"""
        try:
            # 기본 기간 추출
            basic_timeline = self._extract_timeline_from_answer(answer)
            
            # 단계별 소요 시간 추출
            step_times = self._extract_step_timings(answer)
            
            enhanced_timeline = f"""
### ⏰ 전체 처리 기간
{basic_timeline}

### 📅 단계별 소요 시간
{step_times}
"""
            return enhanced_timeline
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced timeline: {e}")
            return self._extract_timeline_from_answer(answer)
    
    def _extract_key_points(self, text: str) -> str:
        """핵심 포인트 추출"""
        try:
            # 핵심 키워드 추출
            key_phrases = [
                '중요한', '핵심', '주의', '필수', '반드시', '꼭', '특히',
                '가장', '주요', '기본', '원칙', '요건', '조건'
            ]
            
            sentences = re.split(r'[.!?]\s*', text)
            key_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                for phrase in key_phrases:
                    if phrase in sentence:
                        key_sentences.append(sentence)
                        break
            
            if key_sentences:
                return self._format_enhanced_bullet_points("\n".join(key_sentences[:3]), "important")
            
            return "핵심 포인트를 추출할 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting key points: {e}")
            return "핵심 포인트 추출 오류"
    
    def _extract_step_details(self, answer: str) -> str:
        """단계별 상세 정보 추출"""
        try:
            # 단계별 상세 설명 패턴 찾기
            detail_patterns = [
                r'(\d+)\.\s*([^\n]+)\s*\n\s*([^\n]+)',
                r'단계\s*(\d+)[:.]\s*([^\n]+)\s*\n\s*([^\n]+)'
            ]
            
            details = []
            for pattern in detail_patterns:
                matches = re.findall(pattern, answer, re.MULTILINE)
                for match in matches:
                    if len(match) >= 3:
                        details.append(f"**{match[0]}단계 상세**: {match[2].strip()}")
            
            if details:
                return "\n\n".join(details[:3])
            
            return "단계별 상세 정보를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting step details: {e}")
            return "단계별 상세 정보 추출 오류"
    
    def _extract_document_details(self, answer: str) -> str:
        """서류별 상세 정보 추출"""
        try:
            # 서류 관련 상세 정보 패턴
            doc_patterns = [
                r'([^.]*서류[^.]*)',
                r'([^.]*신청서[^.]*)',
                r'([^.]*증명서[^.]*)',
                r'([^.]*계약서[^.]*)'
            ]
            
            doc_details = []
            for pattern in doc_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    if len(match.strip()) > 15:
                        doc_details.append(match.strip())
            
            if doc_details:
                return self._format_enhanced_bullet_points("\n".join(doc_details[:5]), "documents")
            
            return "서류별 상세 정보를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting document details: {e}")
            return "서류별 상세 정보 추출 오류"
    
    def _extract_step_timings(self, answer: str) -> str:
        """단계별 소요 시간 추출"""
        try:
            # 시간 관련 패턴 찾기
            time_patterns = [
                r'(\d+)\s*일\s*([^\n]+)',
                r'(\d+)\s*주\s*([^\n]+)',
                r'(\d+)\s*개월\s*([^\n]+)',
                r'(\d+)\s*시간\s*([^\n]+)'
            ]
            
            timings = []
            for pattern in time_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    if len(match) >= 2:
                        timings.append(f"**{match[0]}**: {match[1].strip()}")
            
            if timings:
                return "\n\n".join(timings[:5])
            
            return "단계별 소요 시간 정보를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting step timings: {e}")
            return "단계별 소요 시간 추출 오류"
    
    def _format_term_explanation(self, 
                                answer: str, 
                                sources: Dict[str, List[Dict[str, Any]]], 
                                confidence: ConfidenceInfo) -> Dict[str, str]:
        """용어 해설 구조화"""
        try:
            sections = {}
            
            # 정의 섹션
            sections["definition"] = self._extract_definition_from_answer(answer)
            
            # 관련 법률
            laws = sources.get("law_results", [])
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            else:
                sections["laws"] = "관련 법률을 찾을 수 없습니다."
            
            # 예시 섹션
            sections["examples"] = self._extract_examples_from_answer(answer)
            
            # 관련 용어
            sections["related"] = self._extract_related_terms_from_answer(answer)
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting term explanation: {e}")
            return {"definition": answer}
    
    def _format_general_answer(self, 
                              answer: str, 
                              sources: Dict[str, List[Dict[str, Any]]], 
                              confidence: ConfidenceInfo) -> Dict[str, str]:
        """일반 답변 구조화"""
        try:
            sections = {}
            
            # 답변 섹션
            sections["answer"] = self._clean_and_structure_text(answer)
            
            # 소스 섹션 (UnifiedSearchResult 객체 처리)
            all_sources = []
            if hasattr(sources, '__dict__') and not isinstance(sources, dict):
                # UnifiedSearchResult 객체인 경우 딕셔너리로 변환
                all_sources = [{
                    'content': getattr(sources, 'content', ''),
                    'title': getattr(sources, 'title', ''),
                    'source': getattr(sources, 'source', ''),
                    'score': getattr(sources, 'score', 0.0)
                }]
            elif isinstance(sources, dict):
                all_sources.extend(sources.get("law_results", []))
                all_sources.extend(sources.get("precedent_results", []))
            elif isinstance(sources, list):
                all_sources = sources
            
            if all_sources:
                sections["sources"] = self._format_general_sources(all_sources)
            else:
                sections["sources"] = "관련 정보를 찾을 수 없습니다."
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting general answer: {e}")
            return {"answer": answer}
    
    def _extract_sources(self, sources: Any, source_type: str) -> List[Dict[str, Any]]:
        """소스에서 특정 타입의 소스들을 안전하게 추출"""
        try:
            if hasattr(sources, '__dict__') and not isinstance(sources, dict):
                # UnifiedSearchResult 객체인 경우
                if source_type == "law_results" or source_type == "precedent_results":
                    return [{
                        'content': getattr(sources, 'content', ''),
                        'title': getattr(sources, 'title', ''),
                        'source': getattr(sources, 'source', ''),
                        'score': getattr(sources, 'score', 0.0)
                    }]
                else:
                    return []
            elif isinstance(sources, dict):
                return sources.get(source_type, [])
            elif isinstance(sources, list):
                return sources
            else:
                return []
        except Exception as e:
            self.logger.debug(f"Error extracting sources: {e}")
            return []
    
    def _clean_and_structure_text(self, text: str) -> str:
        """텍스트 정리 및 구조화"""
        try:
            # 기본 정리
            cleaned = text.strip()
            
            # 문단 구분 개선
            cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
            
            # 번호 목록 정리
            cleaned = re.sub(r'(\d+)\.\s*', r'\1. ', cleaned)
            
            # 불릿 포인트 정리
            cleaned = re.sub(r'[-•]\s*', '• ', cleaned)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return text
    
    def _format_precedent_sources(self, precedents: List[Dict[str, Any]]) -> str:
        """판례 소스 포맷팅"""
        try:
            if not precedents:
                return "관련 판례를 찾을 수 없습니다."
            
            formatted = []
            for i, prec in enumerate(precedents[:5], 1):
                case_name = prec.get('case_name', '사건명 없음')
                case_number = prec.get('case_number', '사건번호 없음')
                court = prec.get('court', '법원 정보 없음')
                decision_date = prec.get('decision_date', '판결일 없음')
                summary = prec.get('summary', '요약 없음')
                similarity = prec.get('similarity', 0.0)
                
                formatted.append(f"""
{i}. **{case_name}** ({case_number})
   - 법원: {court}
   - 판결일: {decision_date}
   - 판결요지: {summary[:200]}{'...' if len(summary) > 200 else ''}
   - 유사도: {similarity:.1%}
""")
            
            return "\n".join(formatted)
            
        except Exception as e:
            self.logger.error(f"Error formatting precedent sources: {e}")
            return "판례 정보 포맷팅 오류"
    
    def _format_law_sources(self, laws: List[Dict[str, Any]]) -> str:
        """법률 소스 포맷팅"""
        try:
            if not laws:
                return "관련 법률을 찾을 수 없습니다."
            
            formatted = []
            for i, law in enumerate(laws[:5], 1):
                law_name = law.get('law_name', '법률명 없음')
                article_number = law.get('article_number', '조문번호 없음')
                content = law.get('content', '내용 없음')
                similarity = law.get('similarity', 0.0)
                
                formatted.append(f"""
{i}. **{law_name} {article_number}**
   - 내용: {content[:200]}{'...' if len(content) > 200 else ''}
   - 유사도: {similarity:.1%}
""")
            
            return "\n".join(formatted)
            
        except Exception as e:
            self.logger.error(f"Error formatting law sources: {e}")
            return "법률 정보 포맷팅 오류"
    
    def _format_confidence_info(self, confidence: ConfidenceInfo) -> str:
        """신뢰도 정보 포맷팅"""
        try:
            level_emoji = {
                "very_high": "🟢",
                "high": "🟢", 
                "medium": "🟡",
                "low": "🟠",
                "very_low": "🔴"
            }.get(confidence.reliability_level, "⚪")
            
            formatted = f"""
{level_emoji} **신뢰도: {confidence.confidence:.1%}** ({confidence.reliability_level})

**상세 점수:**"""
            
            # factors에서 점수 정보 추출
            if 'similarity_score' in confidence.factors:
                formatted += f"\n- 검색 결과 유사도: {confidence.factors['similarity_score']:.1%}"
            if 'matching_score' in confidence.factors:
                formatted += f"\n- 법률/판례 매칭 정확도: {confidence.factors['matching_score']:.1%}"
            if 'answer_quality' in confidence.factors:
                formatted += f"\n- 답변 품질: {confidence.factors['answer_quality']:.1%}"
            
            # explanation 추가
            if confidence.explanation:
                formatted += f"\n\n**설명:** {confidence.explanation}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting confidence info: {e}")
            return f"신뢰도: {confidence.confidence:.1%}"
    
    def _build_formatted_content(self, 
                                template: Dict[str, Any], 
                                sections: Dict[str, str], 
                                confidence: ConfidenceInfo) -> str:
        """최종 구조화된 내용 생성"""
        try:
            content_parts = []
            
            # 제목
            content_parts.append(template["title"])
            content_parts.append("")
            
            # 각 섹션 추가
            for section_name in template["sections"]:
                if section_name in sections and sections[section_name]:
                    emoji = self.emoji_map.get(section_name, "📝")
                    content_parts.append(f"### {emoji} {self._get_section_title(section_name)}")
                    content_parts.append("")
                    content_parts.append(sections[section_name])
                    content_parts.append("")
            
            # 면책 조항
            if template.get("disclaimer", False):
                content_parts.append(self._get_disclaimer())
            
            return "\n".join(content_parts)
            
        except Exception as e:
            self.logger.error(f"Error building formatted content: {e}")
            return sections.get("analysis", sections.get("answer", "답변 생성 오류"))
    
    def _get_section_title(self, section_name: str) -> str:
        """섹션 제목 반환"""
        titles = {
            "analysis": "판례 분석",
            "precedents": "참고 판례",
            "laws": "적용 법률",
            "confidence": "신뢰도 정보",
            "explanation": "법률 해설",
            "examples": "적용 예시",
            "advice": "법적 조언",
            "steps": "단계별 가이드",
            "overview": "절차 개요",
            "documents": "필요 서류",
            "timeline": "처리 기간",
            "definition": "용어 정의",
            "related": "관련 용어",
            "answer": "답변",
            "sources": "참고 자료",
            # 새로 추가된 섹션들
            "warnings": "주의사항",
            "recommendations": "권장사항"
        }
        return titles.get(section_name, section_name)
    
    def _get_disclaimer(self) -> str:
        """면책 조항 반환"""
        return """---
💼 **면책 조항**
됨
구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
    
    def _extract_steps_from_answer(self, answer: str) -> str:
        """답변에서 단계별 가이드 추출 (강화된 번호 목록)"""
        try:
            # 강화된 번호 목록 패턴 찾기
            steps = re.findall(r'(\d+)\.\s*([^\n]+(?:\n(?:   |\t)[^\n]+)*)', answer, re.MULTILINE)
            if steps:
                formatted_steps = []
                for num, step in steps:
                    # 단계별 상세 설명 포함
                    step_content = step.strip()
                    # 하위 항목이 있는지 확인
                    sub_items = re.findall(r'   - ([^\n]+)', step_content)
                    if sub_items:
                        formatted_steps.append(f"**{num}단계: {step_content.split('\n')[0].strip()}**")
                        for sub_item in sub_items:
                            formatted_steps.append(f"   • {sub_item.strip()}")
                    else:
                        formatted_steps.append(f"**{num}단계: {step_content}**")
                return "\n\n".join(formatted_steps)
            
            # 불릿 포인트를 번호 목록으로 변환
            bullets = re.findall(r'[-•]\s*([^\n]+)', answer)
            if bullets:
                formatted_bullets = []
                for i, bullet in enumerate(bullets, 1):
                    formatted_bullets.append(f"**{i}단계: {bullet.strip()}**")
                return "\n\n".join(formatted_bullets)
            
            # 문장 단위로 단계 추출 시도
            sentences = re.split(r'[.!?]\s*', answer)
            if len(sentences) >= 3:
                formatted_sentences = []
                for i, sentence in enumerate(sentences[:5], 1):
                    if len(sentence.strip()) > 10:  # 의미있는 문장만
                        formatted_sentences.append(f"**{i}단계: {sentence.strip()}**")
                if formatted_sentences:
                    return "\n\n".join(formatted_sentences)
            
            return "단계별 가이드를 추출할 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting steps: {e}")
            return "단계별 가이드 추출 오류"
    
    def _extract_overview_from_answer(self, answer: str) -> str:
        """답변에서 개요 추출"""
        try:
            # 첫 번째 문단을 개요로 사용
            paragraphs = answer.split('\n\n')
            if paragraphs:
                return paragraphs[0].strip()
            return answer[:300] + "..." if len(answer) > 300 else answer
            
        except Exception as e:
            self.logger.error(f"Error extracting overview: {e}")
            return answer
    
    def _extract_documents_from_answer(self, answer: str) -> str:
        """답변에서 필요 서류 추출"""
        try:
            # 서류 관련 키워드 찾기
            doc_keywords = ['서류', '신청서', '증명서', '계약서', '신고서', '소장', '답변서']
            found_docs = []
            
            for keyword in doc_keywords:
                if keyword in answer:
                    # 해당 키워드 주변 텍스트 추출
                    pattern = f'.{{0,50}}{keyword}.{{0,50}}'
                    matches = re.findall(pattern, answer)
                    found_docs.extend(matches)
            
            if found_docs:
                return "\n".join([f"• {doc.strip()}" for doc in found_docs[:5]])
            
            return "필요한 서류 정보를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting documents: {e}")
            return "서류 정보 추출 오류"
    
    def _extract_timeline_from_answer(self, answer: str) -> str:
        """답변에서 처리 기간 추출"""
        try:
            # 기간 관련 패턴 찾기
            time_patterns = [
                r'(\d+)\s*일',
                r'(\d+)\s*주',
                r'(\d+)\s*개월',
                r'(\d+)\s*년',
                r'(\d+)\s*시간'
            ]
            
            found_times = []
            for pattern in time_patterns:
                matches = re.findall(pattern, answer)
                found_times.extend(matches)
            
            if found_times:
                return f"처리 기간: {', '.join(set(found_times))}"
            
            return "처리 기간 정보를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting timeline: {e}")
            return "처리 기간 추출 오류"
    
    def _extract_definition_from_answer(self, answer: str) -> str:
        """답변에서 정의 추출"""
        try:
            # 정의 관련 패턴 찾기
            definition_patterns = [
                r'([^.]*는[^.]*이다[^.]*)',
                r'([^.]*란[^.]*이다[^.]*)',
                r'([^.]*이란[^.]*이다[^.]*)',
                r'([^.]*는[^.]*를[^.]*말한다[^.]*)'
            ]
            
            for pattern in definition_patterns:
                matches = re.findall(pattern, answer)
                if matches:
                    return matches[0].strip()
            
            # 첫 번째 문장을 정의로 사용
            sentences = answer.split('.')
            if sentences:
                return sentences[0].strip() + '.'
            
            return answer[:200] + "..." if len(answer) > 200 else answer
            
        except Exception as e:
            self.logger.error(f"Error extracting definition: {e}")
            return answer
    
    def _extract_examples_from_answer(self, answer: str) -> str:
        """답변에서 예시 추출"""
        try:
            # 예시 관련 키워드 찾기
            example_keywords = ['예시', '예를 들어', '예컨대', '예시로', '사례']
            
            for keyword in example_keywords:
                if keyword in answer:
                    # 해당 키워드 이후 텍스트 추출
                    start_idx = answer.find(keyword)
                    example_text = answer[start_idx:start_idx + 300]
                    return example_text.strip()
            
            return "구체적인 예시를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting examples: {e}")
            return "예시 추출 오류"
    
    def _extract_related_terms_from_answer(self, answer: str) -> str:
        """답변에서 관련 용어 추출"""
        try:
            # 법률 용어 패턴 찾기
            legal_terms = [
                '손해배상', '계약', '임대차', '불법행위', '소송', '상속', '이혼',
                '교통사고', '근로', '부동산', '금융', '지적재산권', '세금', '환경', '의료'
            ]
            
            found_terms = []
            for term in legal_terms:
                if term in answer and term not in found_terms:
                    found_terms.append(term)
            
            if found_terms:
                return f"관련 용어: {', '.join(found_terms)}"
            
            return "관련 용어를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error extracting related terms: {e}")
            return "관련 용어 추출 오류"
    
    def _generate_law_examples(self, answer: str, laws: List[Dict[str, Any]]) -> str:
        """법률 예시 생성"""
        try:
            if not laws:
                return "관련 법률 예시를 찾을 수 없습니다."
            
            examples = []
            for law in laws[:2]:
                law_name = law.get('law_name', '')
                article_number = law.get('article_number', '')
                if law_name and article_number:
                    examples.append(f"• {law_name} {article_number}의 적용 사례")
            
            if examples:
                return "\n".join(examples)
            
            return "법률 적용 예시를 찾을 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"Error generating law examples: {e}")
            return "법률 예시 생성 오류"
    
    def _format_general_sources(self, sources: List[Dict[str, Any]]) -> str:
        """일반 소스 포맷팅"""
        try:
            if not sources:
                return "관련 정보를 찾을 수 없습니다."
            
            formatted = []
            for i, source in enumerate(sources[:5], 1):
                source_type = source.get('type', 'unknown')
                if source_type == 'law':
                    law_name = source.get('law_name', '')
                    article_number = source.get('article_number', '')
                    formatted.append(f"{i}. 법률: {law_name} {article_number}")
                elif source_type == 'precedent':
                    case_name = source.get('case_name', '')
                    case_number = source.get('case_number', '')
                    formatted.append(f"{i}. 판례: {case_name} ({case_number})")
                else:
                    formatted.append(f"{i}. {source.get('title', '정보')}")
            
            return "\n".join(formatted)
            
        except Exception as e:
            self.logger.error(f"Error formatting general sources: {e}")
            return "소스 정보 포맷팅 오류"
    
    def _format_enhanced_bullet_points(self, text: str, section_type: str = "general") -> str:
        """강화된 불릿 포인트 포맷팅"""
        try:
            # 불릿 포인트 패턴 찾기
            bullet_patterns = [
                r'[-•]\s*([^\n]+)',
                r'(\d+)\s*[.)]\s*([^\n]+)',
                r'[가-힣]\s*[.)]\s*([^\n]+)'
            ]
            
            formatted_items = []
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            item_text = match[1] if len(match) > 1 else match[0]
                        else:
                            item_text = match
                        
                        # 섹션 타입에 따른 이모지 선택
                        emoji = self._get_bullet_emoji(section_type)
                        formatted_items.append(f"{emoji} **{item_text.strip()}**")
            
            if formatted_items:
                return "\n\n".join(formatted_items)
            
            # 불릿 포인트가 없으면 문장을 불릿 포인트로 변환
            sentences = re.split(r'[.!?]\s*', text)
            if len(sentences) >= 2:
                formatted_sentences = []
                for sentence in sentences[:5]:
                    if len(sentence.strip()) > 10:
                        emoji = self._get_bullet_emoji(section_type)
                        formatted_sentences.append(f"{emoji} **{sentence.strip()}**")
                if formatted_sentences:
                    return "\n\n".join(formatted_sentences)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error formatting bullet points: {e}")
            return text
    
    def _get_bullet_emoji(self, section_type: str) -> str:
        """섹션 타입에 따른 불릿 이모지 반환"""
        emoji_map = {
            "warnings": "⚠️",
            "recommendations": "💡",
            "important": "❗",
            "steps": "📝",
            "documents": "📄",
            "caution": "🚨",
            "checklist": "✅",
            "general": "•"
        }
        return emoji_map.get(section_type, "•")
    
    def _extract_warnings_and_recommendations(self, answer: str) -> Dict[str, str]:
        """답변에서 주의사항과 권장사항 추출"""
        try:
            warnings = []
            recommendations = []
            
            # 주의사항 키워드
            warning_keywords = ['주의', '경고', '위험', '주의사항', '주의할 점', '조심', '피해야']
            # 권장사항 키워드
            recommendation_keywords = ['권장', '추천', '제안', '권장사항', '권고', '바람직', '좋은']
            
            sentences = re.split(r'[.!?]\s*', answer)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                # 주의사항 추출
                for keyword in warning_keywords:
                    if keyword in sentence:
                        warnings.append(sentence)
                        break
                
                # 권장사항 추출
                for keyword in recommendation_keywords:
                    if keyword in sentence:
                        recommendations.append(sentence)
                        break
            
            return {
                "warnings": self._format_enhanced_bullet_points("\n".join(warnings), "warnings") if warnings else "특별한 주의사항이 없습니다.",
                "recommendations": self._format_enhanced_bullet_points("\n".join(recommendations), "recommendations") if recommendations else "추가 권장사항이 없습니다."
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting warnings and recommendations: {e}")
            return {
                "warnings": "주의사항 추출 오류",
                "recommendations": "권장사항 추출 오류"
            }
    
    def _create_fallback_answer(self, raw_answer: str, confidence: ConfidenceInfo) -> FormattedAnswer:
        """오류 시 기본 답변 생성"""
        try:
            return FormattedAnswer(
                formatted_content=f"""## 답변

{raw_answer}

### 💡 신뢰도 정보
- 신뢰도: {confidence.confidence:.1%}
- 수준: {confidence.reliability_level}

---
💼 됨
구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다.""",
                sections={"answer": raw_answer},
                metadata={"question_type": "general", "confidence_level": confidence.reliability_level}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating fallback answer: {e}")
            return FormattedAnswer(
                formatted_content="답변 포맷팅 중 오류가 발생했습니다.",
                sections={"answer": raw_answer},
                metadata={"question_type": "error"}
            )
    
    def _is_clean_answer(self, answer: str) -> bool:
        """답변이 깔끔하게 구성되어 있는지 확인"""
        # 불필요한 패턴이 있는지 확인
        unwanted_patterns = [
            r'###\s*관련\s*법령\s*\n+\s*관련\s*법령\s*:',
            r'###\s*법령\s*해설\s*\n+\s*법령\s*해설\s*:',
            r'###\s*적용\s*사례\s*\n+\s*실제\s*적용\s*사례\s*:',
            r'\*쉬운\s*말로\s*풀어서\s*설명\*',
            r'\*구체적\s*예시와\s*설명\*',
            r'\*법적\s*리스크와\s*제한사항\*',
            r'---\s*\n\s*💼\s*\*\*면책\s*조항\*\*'
        ]
        
        for pattern in unwanted_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return False
        
        return True
    
    def _clean_answer(self, answer: str) -> str:
        """답변을 깔끔하게 정리 - 템플릿 패턴 완전 제거"""
        import re
        
        # 모든 템플릿 패턴 제거
        template_patterns = [
            r'##\s*법률\s*문의\s*답변\s*',
            r'###\s*관련\s*법령\s*',
            r'###\s*법령\s*해설\s*',
            r'###\s*적용\s*사례\s*',
            r'###\s*주의사항\s*',
            r'###\s*권장사항\s*',
            r'###\s*답변\s*',
            r'###\s*분석\s*',
            r'###\s*해설\s*',
            r'###\s*안내\s*',
        ]
        
        for pattern in template_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # 모든 플레이스홀더 제거
        placeholder_patterns = [
            r'\*정확한\s*조문\s*번호와\s*내용\*',
            r'\*쉬운\s*말로\s*풀어서\s*설명\*',
            r'\*구체적\s*예시와\s*설명\*',
            r'\*법적\s*리스크와\s*제한사항\*',
            r'\*추가\s*권장사항\*',
            r'\*실무적\s*조언\*',
        ]
        
        for pattern in placeholder_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        
        disclaimer_patterns = [
            r'---\s*\n\s*💼\s*\*\*면책\s*조항\*\*\s*\n\s*#\s*면책\s*조항\s*제거\s*\n\s*#\s*본\s*답변은.*?바랍니다\.\s*\n*',
            r'💼\s*\*\*면책\s*조항\*\*\s*\n\s*#\s*면책\s*조항\s*제거\s*\n\s*#\s*본\s*답변은.*?바랍니다\.\s*\n*',
            r'면책\s*조항.*?바랍니다\.',
            r'변호사.*?상담.*?권장',
            r'법률\s*자문.*?필요',
        ]
        
        for pattern in disclaimer_patterns:
            answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
        
        # 연속된 빈 줄 정리
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        return answer.strip()


# 테스트 함수
def test_answer_formatter():
    """답변 구조화기 테스트"""
    formatter = AnswerFormatter()
    
    # 테스트 데이터
    test_answer = """손해배상 청구 방법은 다음과 같습니다:

1. 불법행위 성립 요건 확인
   - 가해행위, 손해 발생, 인과관계, 고의 또는 과실

2. 적용 법률
   - 민법 제750조 (불법행위로 인한 손해배상)

3. 관련 판례
   - 2023다12345 손해배상청구 사건

4. 청구 절차
   - 소장 작성 및 제출
   - 증거 자료 준비
   - 법원에서 소송 진행"""
    
    test_sources = {
        "law_results": [
            {"law_name": "민법", "article_number": "제750조", "content": "불법행위로 인한 손해배상", "similarity": 0.9}
        ],
        "precedent_results": [
            {"case_name": "손해배상청구 사건", "case_number": "2023다12345", "summary": "불법행위 손해배상", "similarity": 0.8}
        ]
    }
    
    test_confidence = ConfidenceInfo(
        confidence=0.85,
        reliability_level="HIGH",
        similarity_score=0.9,
        matching_score=0.8,
        answer_quality=0.85,
        warnings=[],
        recommendations=["전문가 상담 권장"]
    )
    
    print("=== 답변 구조화기 테스트 ===")
    
    # 판례 검색 답변 포맷팅
    print("\n1. 판례 검색 답변 포맷팅:")
    result = formatter.format_answer(
        raw_answer=test_answer,
        question_type=QuestionType.PRECEDENT_SEARCH,
        sources=test_sources,
        confidence=test_confidence
    )
    
    print(f"포맷팅된 답변 길이: {len(result.formatted_content)}")
    print(f"섹션 수: {len(result.sections)}")
    print(f"메타데이터: {result.metadata}")
    print(f"\n포맷팅된 답변 미리보기:")
    print(result.formatted_content[:500] + "..." if len(result.formatted_content) > 500 else result.formatted_content)


if __name__ == "__main__":
    test_answer_formatter()
