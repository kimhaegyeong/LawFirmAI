# -*- coding: utf-8 -*-
"""
답변 구조화 개선 시스템
일관된 형식의 구조화된 답변 제공
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .confidence_calculator import ConfidenceInfo
from .question_classifier import QuestionType

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

        # 질문 유형별 템플릿 (개선: 우선순위 섹션 추가)
        self.templates = {
            QuestionType.PRECEDENT_SEARCH: {
                "title": "## 관련 판례 분석",
                "sections": ["analysis", "precedents", "laws", "confidence"],
                "disclaimer": True,
                "priority_sections": ["analysis", "precedents"]
            },
            QuestionType.LAW_INQUIRY: {
                "title": "## 법률 해설",
                "sections": ["explanation", "laws", "examples", "confidence"],
                "disclaimer": True,
                "priority_sections": ["explanation", "laws"]
            },
            QuestionType.LEGAL_ADVICE: {
                "title": "## 법적 조언",
                "sections": ["advice", "laws", "steps", "warnings", "confidence"],
                "disclaimer": True,
                "priority_sections": ["advice", "steps"]
            },
            QuestionType.PROCEDURE_GUIDE: {
                "title": "## 절차 안내",
                "sections": ["overview", "steps", "documents", "timeline", "warnings", "confidence"],
                "disclaimer": True,
                "priority_sections": ["overview", "steps"]
            },
            QuestionType.TERM_EXPLANATION: {
                "title": "## 용어 해설",
                "sections": ["definition", "laws", "examples", "confidence"],
                "disclaimer": True,
                "priority_sections": ["definition"]
            },
            QuestionType.GENERAL_QUESTION: {
                "title": "## 답변",
                "sections": ["answer", "sources", "confidence"],
                "disclaimer": True,
                "priority_sections": ["answer"]
            }
        }

        # 이모지 매핑 (최소화: 챗봇 친화적)
        # 이모지는 사용하지 않고 텍스트만 사용 (개선)
        self.emoji_map = {}  # 모든 이모지 제거

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
            # raw_answer 타입 검증 및 문자열 변환
            if isinstance(raw_answer, dict):
                self.logger.warning(f"AnswerFormatter.format_answer: raw_answer is dict, converting to string")
                # 중첩 딕셔너리에서 문자열 추출 시도
                if "answer" in raw_answer:
                    raw_answer = raw_answer["answer"]
                elif "content" in raw_answer:
                    raw_answer = raw_answer["content"]
                elif "text" in raw_answer:
                    raw_answer = raw_answer["text"]
                else:
                    raw_answer = str(raw_answer)

            # 최종 문자열 보장
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)

            self.logger.info(f"Formatting answer for question type: {question_type.value} (answer length: {len(raw_answer)})")

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
        """법률 해설 구조화 (개선된 버전 - 빈 섹션 제거)"""
        try:
            sections = {}

            # 해설 섹션
            sections["explanation"] = self._clean_and_structure_text(answer)

            # 법률 섹션 - 빈 경우 추가하지 않음 (개선)
            laws = sources.get("law_results", [])
            if laws:
                sections["laws"] = self._format_law_sources(laws)
            # 빈 섹션은 추가하지 않음 (개선: "관련 법률을 찾을 수 없습니다" 제거)

            # 예시 섹션 (간단한 예시 추가) - 빈 경우 추가하지 않음
            examples = self._generate_law_examples(answer, laws)
            if examples and examples.strip() and examples != "관련 정보를 찾을 수 없습니다.":
                sections["examples"] = examples

            # 신뢰도 섹션 (간소화된 형식)
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
            laws = sources.get("law_results", [])
            if laws:
                sections["laws"] = self._format_law_sources(laws)

            # 판례 섹션
            precedents = sources.get("precedent_results", [])
            if precedents:
                sections["precedents"] = self._format_precedent_sources(precedents)

            # 단계별 가이드 섹션 - 빈 경우 추가하지 않음 (개선)
            steps = self._extract_steps_from_answer(answer)
            if steps and steps.strip() and steps != "관련 정보를 찾을 수 없습니다.":
                sections["steps"] = steps

            # 주의사항 및 권장사항 추가
            warnings_recs = self._extract_warnings_and_recommendations(answer)
            if warnings_recs.get("warnings") and warnings_recs["warnings"].strip() and warnings_recs["warnings"] != "관련 정보를 찾을 수 없습니다.":
                sections["warnings"] = warnings_recs["warnings"]
            if warnings_recs.get("recommendations") and warnings_recs["recommendations"].strip() and warnings_recs["recommendations"] != "관련 정보를 찾을 수 없습니다.":
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

            # 개요 섹션 - 빈 경우 추가하지 않음 (개선)
            overview = self._extract_enhanced_overview(answer)
            if overview and overview.strip() and overview != "관련 정보를 찾을 수 없습니다.":
                sections["overview"] = overview

            # 단계별 절차 - 빈 경우 추가하지 않음 (개선)
            steps = self._extract_enhanced_steps(answer)
            if steps and steps.strip() and steps != "관련 정보를 찾을 수 없습니다.":
                sections["steps"] = steps

            # 필요 서류 - 빈 경우 추가하지 않음 (개선)
            documents = self._extract_enhanced_documents(answer)
            if documents and documents.strip() and documents != "관련 정보를 찾을 수 없습니다.":
                sections["documents"] = documents

            # 처리 기간 - 빈 경우 추가하지 않음 (개선)
            timeline = self._extract_enhanced_timeline(answer)
            if timeline and timeline.strip() and timeline != "관련 정보를 찾을 수 없습니다.":
                sections["timeline"] = timeline

            # 주의사항 및 권장사항 추가 - 빈 경우 추가하지 않음 (개선)
            warnings_recs = self._extract_warnings_and_recommendations(answer)
            if warnings_recs.get("warnings") and warnings_recs["warnings"].strip() and warnings_recs["warnings"] != "관련 정보를 찾을 수 없습니다.":
                sections["warnings"] = warnings_recs["warnings"]
            if warnings_recs.get("recommendations") and warnings_recs["recommendations"].strip() and warnings_recs["recommendations"] != "관련 정보를 찾을 수 없습니다.":
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
        """용어 해설 구조화 (개선된 버전 - 빈 섹션 제거)"""
        try:
            sections = {}

            # 정의 섹션
            definition = self._extract_definition_from_answer(answer)
            if definition and definition.strip():
                sections["definition"] = definition

            # 관련 법률 - 빈 경우 추가하지 않음 (개선)
            laws = sources.get("law_results", [])
            if laws:
                sections["laws"] = self._format_law_sources(laws)

            # 예시 섹션 - 빈 경우 추가하지 않음 (개선)
            examples = self._extract_examples_from_answer(answer)
            if examples and examples.strip() and examples != "관련 정보를 찾을 수 없습니다.":
                sections["examples"] = examples

            # 관련 용어 - 빈 경우 추가하지 않음 (개선)
            related = self._extract_related_terms_from_answer(answer)
            if related and related.strip() and related != "관련 정보를 찾을 수 없습니다.":
                sections["related"] = related

            # 신뢰도 섹션 (간소화된 형식)
            sections["confidence"] = self._format_confidence_info(confidence)

            return sections

        except Exception as e:
            self.logger.error(f"Error formatting term explanation: {e}")
            return {"definition": answer}

    def _format_general_answer(self,
                              answer: str,
                              sources: Dict[str, List[Dict[str, Any]]],
                              confidence: ConfidenceInfo) -> Dict[str, str]:
        """일반 답변 구조화 (개선된 버전 - 빈 섹션 제거)"""
        try:
            sections = {}

            # 답변 섹션
            sections["answer"] = self._clean_and_structure_text(answer)

            # 소스 섹션 - 빈 경우 추가하지 않음 (개선)
            all_sources = []
            all_sources.extend(sources.get("law_results", []))
            all_sources.extend(sources.get("precedent_results", []))

            if all_sources:
                sections["sources"] = self._format_general_sources(all_sources)
            # 빈 섹션은 추가하지 않음 (개선: "관련 정보를 찾을 수 없습니다" 제거)

            # 신뢰도 섹션 (간소화된 형식으로 추가)
            sections["confidence"] = self._format_confidence_info(confidence)

            return sections

        except Exception as e:
            self.logger.error(f"Error formatting general answer: {e}")
            return {"answer": answer}

    def _fix_spacing(self, text: str) -> str:
        """띄어쓰기 자동 보정"""
        try:
            fixed = text

            # 조사 앞 불필요한 띄어쓰기 제거
            particles = ['은', '는', '이', '가', '을', '를', '에', '에서', '와', '과', '도', '만', '부터', '까지', '께서', '에게', '에게서']
            for particle in particles:
                # 조사 앞 띄어쓰기 제거 (예: "사람 은" -> "사람은")
                fixed = re.sub(rf'\s+({particle})\s+', rf'\1 ', fixed)
                fixed = re.sub(rf'\s+({particle})', rf'\1', fixed)

            # 어미 앞 불필요한 띄어쓰기 제거
            endings = ['다', '요', '습니다', '입니다', '있습니다', '없습니다', '합니다', '됩니다']
            for ending in endings:
                # 어미 앞 띄어쓰기 제거 (예: "한다 다" -> "한다다" -> "한다"로 정규화)
                fixed = re.sub(rf'(\w+)\s+({ending})', rf'\1{ending}', fixed)

            # 마침표 뒤 불필요한 띄어쓰기 제거
            fixed = re.sub(r'\.\s+\.', '.', fixed)

            # 쉼표 뒤 띄어쓰기 보정
            fixed = re.sub(r',\s*', ', ', fixed)

            # 괄호 내부 띄어쓰기 보정
            fixed = re.sub(r'\(\s+', '(', fixed)
            fixed = re.sub(r'\s+\)', ')', fixed)

            # 연속된 공백 제거 (단, 문단 구분은 유지)
            fixed = re.sub(r'[ \t]+', ' ', fixed)
            fixed = re.sub(r'\n\s+', '\n', fixed)

            return fixed

        except Exception as e:
            self.logger.error(f"Error fixing spacing: {e}")
            return text

    def _clean_and_structure_text(self, text: str) -> str:
        """텍스트 정리 및 구조화 (개선: 띄어쓰기 보정 추가)"""
        try:
            # 기본 정리
            cleaned = text.strip()

            # 띄어쓰기 보정 (개선)
            cleaned = self._fix_spacing(cleaned)

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
        """신뢰도 정보 포맷팅 (간소화된 버전)"""
        try:
            # 간단한 형식으로 변경: "신뢰도: 61.8% (medium)"
            return f"신뢰도: {confidence.confidence:.1%} ({confidence.reliability_level})"
        except Exception as e:
            self.logger.error(f"Error formatting confidence info: {e}")
            return f"신뢰도: {confidence.confidence:.1%}"

    def _build_formatted_content(self,
                                template: Dict[str, Any],
                                sections: Dict[str, str],
                                confidence: ConfidenceInfo) -> str:
        """최종 구조화된 내용 생성 (개선된 버전 - 빈 섹션 필터링, 제목 중복 방지)"""
        try:
            content_parts = []

            # 제목 (중복 방지: 템플릿 제목만 사용, 섹션 제목과 중복 방지)
            template_title = template["title"]
            # 이미 "## "로 시작하는지 확인하고, 없으면 추가
            if not template_title.startswith("## "):
                template_title = f"## {template_title.replace('## ', '').replace('### ', '')}"
            # 이모지 제거
            template_title_clean = re.sub(r'[📖⚖️💼💡📚📋⭐📌🔍💬🎯📊📝📄⏰🔗⚠️❗✅🚨]+\s*', '', template_title.replace('## ', '').replace('### ', '')).strip()
            content_parts.append(f"## {template_title_clean}")
            content_parts.append("")

            # 각 섹션 추가 (빈 섹션 필터링 강화, 우선순위 적용 - 개선)
            # confidence 섹션은 마지막에 한 번만 표시하도록 별도 처리
            confidence_section = None
            priority_sections = template.get("priority_sections", [])  # 우선 표시 섹션

            # 우선순위 섹션 먼저 처리
            processed_sections = set()
            for section_name in priority_sections:
                if section_name in template["sections"]:
                    processed_sections.add(section_name)
                    if section_name == "confidence":
                        if section_name in sections and sections[section_name]:
                            confidence_section = sections[section_name]
                        continue

                    if section_name in sections and sections[section_name]:
                        section_content = sections[section_name].strip()
                        if not section_content or section_content == "관련 정보를 찾을 수 없습니다." or section_content == "관련 법률을 찾을 수 없습니다." or section_content == "관련 판례를 찾을 수 없습니다.":
                            continue

                        section_title = self._get_section_title(section_name)
                        content_parts.append(f"### {section_title}")
                        content_parts.append("")
                        content_parts.append(section_content)
                        content_parts.append("")

            # 나머지 섹션 처리
            for section_name in template["sections"]:
                if section_name in processed_sections:
                    continue

                # confidence 섹션은 나중에 처리
                if section_name == "confidence":
                    if section_name in sections and sections[section_name]:
                        confidence_section = sections[section_name]
                    continue

                if section_name in sections and sections[section_name]:
                    section_content = sections[section_name].strip()
                    # 빈 섹션이거나 "관련 정보를 찾을 수 없습니다" 같은 메시지 제거
                    if not section_content or section_content == "관련 정보를 찾을 수 없습니다." or section_content == "관련 법률을 찾을 수 없습니다." or section_content == "관련 판례를 찾을 수 없습니다.":
                        continue

                    # 섹션 제목 (이모지 제거 - 개선)
                    section_title = self._get_section_title(section_name)
                    # 이모지는 완전히 제거 (개선: 챗봇 친화적)
                    content_parts.append(f"### {section_title}")
                    content_parts.append("")
                    content_parts.append(section_content)
                    content_parts.append("")

            # 신뢰도 정보는 마지막에 한 번만 표시 (간소화된 형식)
            if confidence_section:
                content_parts.append(confidence_section)
                content_parts.append("")

            # 면책 조항 (신뢰도가 낮을 때만 표시 - 개선)
            if template.get("disclaimer", False) and confidence.confidence < 0.5:
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
        """면책 조항 반환 (간소화된 버전)"""
        return "※ 본 답변은 일반적인 법률 정보이며, 구체적인 법률 문제는 변호사와 상담하시기 바랍니다."

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
                        first_line = step_content.split('\n')[0].strip()
                        formatted_steps.append(f"**{num}단계: {first_line}**")
                        for sub_item in sub_items:
                            formatted_steps.append(f"   • {sub_item.strip()}")
                    else:
                        # 전체 내용을 포맷팅 (줄바꿈 처리)
                        step_lines = step_content.split('\n')
                        main_step = step_lines[0].strip() if step_lines else step_content
                        formatted_steps.append(f"**{num}단계: {main_step}**")
                        if len(step_lines) > 1:
                            for line in step_lines[1:]:
                                if line.strip():
                                    formatted_steps.append(f"   {line.strip()}")
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
