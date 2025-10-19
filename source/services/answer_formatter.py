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
    """답변 구조화기"""
    
    def __init__(self):
        """답변 구조화기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 질문 유형별 템플릿
        self.templates = {
            QuestionType.PRECEDENT_SEARCH: {
                "title": "## 관련 판례 분석",
                "sections": ["analysis", "precedents", "laws", "confidence"],
                "disclaimer": True
            },
            QuestionType.LAW_INQUIRY: {
                "title": "## 법률 해설",
                "sections": ["explanation", "laws", "examples", "confidence"],
                "disclaimer": True
            },
            QuestionType.LEGAL_ADVICE: {
                "title": "## 법적 조언",
                "sections": ["advice", "laws", "precedents", "steps", "confidence"],
                "disclaimer": True
            },
            QuestionType.PROCEDURE_GUIDE: {
                "title": "## 절차 안내",
                "sections": ["overview", "steps", "documents", "timeline", "confidence"],
                "disclaimer": True
            },
            QuestionType.TERM_EXPLANATION: {
                "title": "## 용어 해설",
                "sections": ["definition", "laws", "examples", "related", "confidence"],
                "disclaimer": True
            },
            QuestionType.GENERAL_QUESTION: {
                "title": "## 답변",
                "sections": ["answer", "sources", "confidence"],
                "disclaimer": True
            }
        }
        
        # 이모지 매핑
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
            "sources": "📚"
        }
    
    def format_answer(self, 
                     raw_answer: str,
                     question_type: QuestionType,
                     sources: Dict[str, List[Dict[str, Any]]],
                     confidence: ConfidenceInfo) -> FormattedAnswer:
        """
        답변 구조화
        
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
            
            template = self.templates.get(question_type, self.templates[QuestionType.GENERAL_QUESTION])
            
            # 섹션별 내용 생성
            sections = {}
            
            if question_type == QuestionType.PRECEDENT_SEARCH:
                sections = self._format_precedent_answer(raw_answer, sources, confidence)
            elif question_type == QuestionType.LAW_INQUIRY:
                sections = self._format_law_explanation(raw_answer, sources, confidence)
            elif question_type == QuestionType.LEGAL_ADVICE:
                sections = self._format_legal_advice(raw_answer, sources, confidence)
            elif question_type == QuestionType.PROCEDURE_GUIDE:
                sections = self._format_procedure_guide(raw_answer, sources, confidence)
            elif question_type == QuestionType.TERM_EXPLANATION:
                sections = self._format_term_explanation(raw_answer, sources, confidence)
            else:
                sections = self._format_general_answer(raw_answer, sources, confidence)
            
            # 최종 구조화된 답변 생성
            formatted_content = self._build_formatted_content(template, sections, confidence)
            
            # 메타데이터 생성
            metadata = {
                "question_type": question_type.value,
                "confidence_level": confidence.reliability_level,
                "confidence_score": confidence.confidence,
                "source_count": {
                    "laws": len(sources.get("law_results", [])),
                    "precedents": len(sources.get("precedent_results", []))
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
            precedents = sources.get("precedent_results", [])
            if precedents:
                sections["precedents"] = self._format_precedent_sources(precedents)
            else:
                sections["precedents"] = "관련 판례를 찾을 수 없습니다."
            
            # 법률 섹션
            laws = sources.get("law_results", [])
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
            laws = sources.get("law_results", [])
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
        """법적 조언 구조화"""
        try:
            sections = {}
            
            # 조언 섹션
            sections["advice"] = self._clean_and_structure_text(answer)
            
            # 법률 섹션
            laws = sources.get("law_results", [])
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            else:
                sections["laws"] = "관련 법률을 찾을 수 없습니다."
            
            # 판례 섹션
            precedents = sources.get("precedent_results", [])
            if precedents:
                sections["precedents"] = self._format_precedent_sources(precedents)
            else:
                sections["precedents"] = "관련 판례를 찾을 수 없습니다."
            
            # 단계별 가이드 섹션
            sections["steps"] = self._extract_steps_from_answer(answer)
            
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
        """절차 안내 구조화"""
        try:
            sections = {}
            
            # 개요 섹션
            sections["overview"] = self._extract_overview_from_answer(answer)
            
            # 단계별 절차
            sections["steps"] = self._extract_steps_from_answer(answer)
            
            # 필요 서류
            sections["documents"] = self._extract_documents_from_answer(answer)
            
            # 처리 기간
            sections["timeline"] = self._extract_timeline_from_answer(answer)
            
            # 신뢰도 섹션
            sections["confidence"] = self._format_confidence_info(confidence)
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error formatting procedure guide: {e}")
            return {"overview": answer}
    
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
            
            # 소스 섹션
            all_sources = []
            all_sources.extend(sources.get("law_results", []))
            all_sources.extend(sources.get("precedent_results", []))
            
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
                "HIGH": "🟢",
                "MEDIUM": "🟡", 
                "LOW": "🟠",
                "VERY_LOW": "🔴"
            }.get(confidence.reliability_level, "⚪")
            
            formatted = f"""
{level_emoji} **신뢰도: {confidence.confidence:.1%}** ({confidence.reliability_level})

**상세 점수:**
- 검색 결과 유사도: {confidence.similarity_score:.1%}
- 법률/판례 매칭 정확도: {confidence.matching_score:.1%}
- 답변 품질: {confidence.answer_quality:.1%}
"""
            
            if confidence.warnings:
                formatted += f"\n**⚠️ 주의사항:**\n"
                for warning in confidence.warnings:
                    formatted += f"- {warning}\n"
            
            if confidence.recommendations:
                formatted += f"\n**💡 권장사항:**\n"
                for recommendation in confidence.recommendations:
                    formatted += f"- {recommendation}\n"
            
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
            "sources": "참고 자료"
        }
        return titles.get(section_name, section_name)
    
    def _get_disclaimer(self) -> str:
        """면책 조항 반환"""
        return """---
💼 **면책 조항**
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.
구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
    
    def _extract_steps_from_answer(self, answer: str) -> str:
        """답변에서 단계별 가이드 추출"""
        try:
            # 번호 목록 찾기
            steps = re.findall(r'(\d+)\.\s*([^\n]+)', answer)
            if steps:
                formatted_steps = []
                for num, step in steps:
                    formatted_steps.append(f"{num}. {step.strip()}")
                return "\n".join(formatted_steps)
            
            # 불릿 포인트 찾기
            bullets = re.findall(r'[-•]\s*([^\n]+)', answer)
            if bullets:
                formatted_bullets = []
                for i, bullet in enumerate(bullets, 1):
                    formatted_bullets.append(f"{i}. {bullet.strip()}")
                return "\n".join(formatted_bullets)
            
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
💼 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.
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
