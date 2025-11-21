# -*- coding: utf-8 -*-
"""
추출 유틸리티 모듈
리팩토링: legal_workflow_enhanced.py에서 추출 메서드 분리
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import re
from typing import Any, Dict, List

logger = get_logger(__name__)


class DocumentExtractor:
    """문서 관련 추출 유틸리티"""

    @staticmethod
    def extract_terms_from_documents(docs: List[Dict]) -> List[str]:
        """문서에서 법률 용어 추출"""
        all_terms = []
        try:
            for doc in docs:
                content = doc.get("content", "")
                if not content:
                    continue

                korean_terms = re.findall(r'[가-힣0-9A-Za-z]+', content)
                legal_terms = [
                    term for term in korean_terms
                    if len(term) >= 2 and any('\uac00' <= c <= '\ud7af' for c in term)
                ]
                all_terms.extend(legal_terms)
        except Exception as e:
            logger.warning(f"Failed to extract terms from documents: {e}")

        return all_terms

    @staticmethod
    def extract_key_insights(
        documents: List[Dict],
        query: str
    ) -> List[str]:
        """핵심 정보 추출 - 질문과 직접 관련된 핵심 문장 추출"""
        insights = []

        try:
            query_words = set(query.lower().split())

            for doc in documents[:10]:
                doc_content = doc.get("content", "")
                if not doc_content:
                    continue

                sentences = re.split(r'[。.！!?？]\s*', doc_content)

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue

                    sentence_words = set(sentence.lower().split())
                    if query_words and sentence_words:
                        overlap = len(query_words.intersection(sentence_words))
                        relevance = overlap / max(1, len(query_words))

                        if relevance >= 0.3:
                            insights.append(sentence)

                            if len(insights) >= 20:
                                break

                if len(insights) >= 20:
                    break

            # 중복 제거
            unique_insights = []
            seen_hashes = set()

            for insight in insights:
                insight_hash = hash(insight[:50])
                if insight_hash not in seen_hashes:
                    seen_hashes.add(insight_hash)
                    unique_insights.append(insight)

            return unique_insights[:15]

        except Exception as e:
            logger.warning(f"Key insights extraction failed: {e}")
            return []

    @staticmethod
    def extract_legal_citations(
        documents: List[Dict]
    ) -> List[Dict[str, str]]:
        """법률 인용 정보 추출"""
        citations = []

        try:
            seen_citations = set()

            citation_pattern = r'([가-힣]+법)\s*제?\s*(\d+)\s*조'
            precedent_pattern = r'(대법원|법원)\s*(\d{4})[.\s]*(\d{1,2})[.\s]*(\d{1,2})?[.\s]*선고\s*(\d{4}[다나마]\d+)'
            simple_precedent_pattern = r'(대법원|법원)\s*(\d{4}[다나마]\d+)'
            law_name_pattern = r'([가-힣]+법)'

            for doc in documents[:10]:
                doc_content = doc.get("content", "")
                doc_source = doc.get("source", "unknown")

                if not doc_content:
                    continue

                # 법률 조항 인용 추출
                law_matches = re.finditer(citation_pattern, doc_content)
                for match in law_matches:
                    law_name = match.group(1)
                    article_num = match.group(2)
                    citation_key = f"{law_name} 제{article_num}조"

                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append({
                            "type": "law_article",
                            "text": citation_key,
                            "law_name": law_name,
                            "article_number": article_num,
                            "source": doc_source
                        })

                # 판례 인용 추출
                precedent_matches = re.finditer(precedent_pattern, doc_content)
                for match in precedent_matches:
                    court = match.group(1)
                    case_number = match.group(5) if len(match.groups()) > 4 else None
                    if not case_number:
                        simple_match = re.search(simple_precedent_pattern, match.group(0))
                        if simple_match:
                            case_number = simple_match.group(2) if len(simple_match.groups()) > 1 else None

                    if case_number:
                        citation_key = f"{court} {case_number}"
                        if citation_key not in seen_citations:
                            seen_citations.add(citation_key)
                            citations.append({
                                "type": "precedent",
                                "text": citation_key,
                                "court": court,
                                "case_number": case_number,
                                "source": doc_source
                            })

                # 법령명 추출
                law_names = re.findall(law_name_pattern, doc_content)
                for law_name in law_names:
                    if f"{law_name} 제" not in doc_content[:500]:
                        citation_key = law_name
                        if citation_key not in seen_citations and len(law_name) >= 2:
                            seen_citations.add(citation_key)
                            citations.append({
                                "type": "law_name",
                                "text": citation_key,
                                "law_name": law_name,
                                "source": doc_source
                            })

            return citations[:20]

        except Exception as e:
            logger.warning(f"Legal citations extraction failed: {e}")
            return []

    @staticmethod
    def extract_legal_references_from_docs(documents: List[Dict[str, Any]]) -> List[str]:
        """문서에서 법률 참조 정보 추출"""
        legal_references = []

        try:
            citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
            precedent_pattern = r'(대법원|법원)\s*(\d{4}[다나마]\d+)'

            for doc in documents[:10]:  # 상위 10개만
                content = doc.get("content", "")
                if not content:
                    continue

                # 법률 조항 인용 추출
                citations = re.findall(citation_pattern, content)
                legal_references.extend(citations)

                # 판례 인용 추출
                precedents = re.findall(precedent_pattern, content)
                for precedent in precedents:
                    legal_references.append(" ".join(precedent))

            # 중복 제거
            legal_references = list(set(legal_references))

        except Exception as e:
            logger.warning(f"Failed to extract legal references: {e}")

        return legal_references[:20]  # 최대 20개만

    @staticmethod
    def extract_contract_clauses(text: str) -> List[Dict[str, Any]]:
        """계약서 주요 조항 추출"""
        clauses = []

        try:
            # 조항 패턴 매칭
            clause_patterns = {
                "payment": r"(대금|금액|지급|결제).*?조",
                "period": r"(기간|기한|만료).*?조",
                "termination": r"(해지|해제|종료).*?조",
                "liability": r"(책임|손해배상|위약).*?조",
                "confidentiality": r"(비밀|기밀|보안).*?조"
            }

            for clause_type, pattern in clause_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # 조항 전체 추출 (제N조 형식)
                    article_match = re.search(r'제\d+조[^제]*', text[match.start():match.start()+500])
                    if article_match:
                        clauses.append({
                            "type": clause_type,
                            "text": article_match.group(0).strip()[:200],
                            "position": match.start()
                        })

            return clauses[:10]  # 상위 10개만

        except Exception as e:
            logger.warning(f"Contract clauses extraction failed: {e}")
            return []

    @staticmethod
    def extract_complaint_elements(text: str) -> List[Dict[str, Any]]:
        """고소장 요건 추출"""
        elements = []

        try:
            # 기본 요소 패턴
            patterns = {
                "parties": r"(피고소인|피해자|가해자)",
                "facts": r"(사실관계|경위|내용)",
                "claims": r"(청구|요구|주장)",
            }

            for elem_type, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    elements.append({
                        "type": elem_type,
                        "found": True
                    })

            return elements

        except Exception as e:
            logger.warning(f"Complaint elements extraction failed: {e}")
            return []

    @staticmethod
    def extract_query_relevant_sentences(
        doc_content: str,
        query: str,
        extracted_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """문서 내용에서 질문과 직접 관련된 문장 추출"""
        relevant_sentences = []

        if not doc_content:
            return relevant_sentences

        try:
            # 문장 분리
            sentences = re.split(r'[。.！!?？]\s*', doc_content)

            query_words = set(query.lower().split())

            for sentence in sentences:
                if not sentence.strip() or len(sentence.strip()) < 10:
                    continue

                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())

                # 질문 키워드 매칭 점수
                query_match = len(query_words.intersection(sentence_words)) / max(1, len(query_words)) if query_words else 0

                # 추출된 키워드 매칭 점수
                keyword_matches = sum(1 for kw in extracted_keywords
                                    if isinstance(kw, str) and kw.lower() in sentence_lower)
                keyword_match = keyword_matches / max(1, len(extracted_keywords)) if extracted_keywords else 0

                # 종합 관련성 점수
                relevance_score = (query_match * 0.6 + keyword_match * 0.4)

                if relevance_score > 0.2:  # 임계값
                    relevant_sentences.append({
                        "sentence": sentence.strip(),
                        "relevance_score": round(relevance_score, 3),
                        "query_match": round(query_match, 3),
                        "keyword_match": round(keyword_match, 3)
                    })

            # 관련성 점수로 정렬
            relevant_sentences.sort(key=lambda x: x["relevance_score"], reverse=True)

            return relevant_sentences[:5]  # 상위 5개만

        except Exception as e:
            logger.warning(f"Query relevant sentences extraction failed: {e}")
            return []


class ResponseExtractor:
    """응답 관련 추출 유틸리티"""

    @staticmethod
    def extract_response_content(response) -> str:
        """응답에서 내용 추출"""
        try:
            if hasattr(response, 'content'):
                content = response.content
                # content가 문자열인지 확인
                if isinstance(content, dict):
                    content = content.get("content", content.get("answer", str(content)))
                return str(content) if not isinstance(content, str) else content

            # response 자체를 처리
            if isinstance(response, dict):
                return response.get("content", response.get("answer", str(response)))

            return str(response)

        except Exception as e:
            logger.warning(f"Failed to extract response content: {e}")
            return str(response) if response else ""


class QueryExtractor:
    """쿼리 관련 추출 유틸리티"""

    @staticmethod
    def extract_legal_field(query_type: str, query: str) -> str:
        """법률 분야 추출"""
        # 키워드 매핑
        field_keywords = {
            "civil": ["민사", "계약", "손해배상", "재산", "계약서"],
            "criminal": ["형사", "범죄", "처벌", "형량", "범죄자"],
            "intellectual_property": ["특허", "상표", "저작권", "지적재산"],
            "administrative": ["행정", "행정처분", "행정소송", "행정심판"]
        }

        query_lower = query.lower()
        for field, keywords in field_keywords.items():
            if any(k in query_lower for k in keywords):
                return field

        # 질문 유형 기반 폴백
        type_to_field = {
            "precedent_search": "civil",
            "law_inquiry": "civil",
            "procedure_guide": "civil",
            "term_explanation": "civil",
            "legal_advice": "civil"
        }
        return type_to_field.get(query_type, "general")
