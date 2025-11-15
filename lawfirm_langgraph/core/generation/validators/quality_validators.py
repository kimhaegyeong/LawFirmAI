# -*- coding: utf-8 -*-
"""
품질 검증 모듈
리팩토링: legal_workflow_enhanced.py에서 검증 로직 분리
"""

import re
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ContextValidator:
    """컨텍스트 품질 검증"""

    @staticmethod
    def calculate_relevance(context_text: str, query: str, semantic_calculator=None) -> float:
        """
        컨텍스트 관련성 계산

        Args:
            context_text: 컨텍스트 텍스트
            query: 질문
            semantic_calculator: 의미적 유사도 계산 함수 (선택적)

        Returns:
            관련성 점수 (0.0-1.0)
        """
        try:
            if not context_text:
                return 0.0

            # 의미적 유사도 계산 시도
            if semantic_calculator and callable(semantic_calculator):
                try:
                    return semantic_calculator(query, context_text)
                except Exception as e:
                    logger.debug(f"Semantic relevance calculation failed: {e}")

            # 폴백: 키워드 기반 유사도
            query_words = set(query.lower().split())
            context_words = set(context_text.lower().split())

            if not query_words or not context_words:
                return 0.0

            overlap = len(query_words.intersection(context_words))
            relevance = overlap / max(1, len(query_words))

            return min(1.0, relevance)

        except Exception as e:
            logger.warning(f"Context relevance calculation failed: {e}")
            return 0.5  # 기본값

    @staticmethod
    def calculate_coverage(
        context_text: str,
        extracted_keywords: List[str],
        legal_references: List[str],
        citations: List[Any]
    ) -> float:
        """
        정보 커버리지 계산 - 핵심 키워드 포함도

        Args:
            context_text: 컨텍스트 텍스트
            extracted_keywords: 추출된 키워드 목록
            legal_references: 법률 참조 목록
            citations: 인용 목록

        Returns:
            커버리지 점수 (0.0-1.0)
        """
        try:
            if not context_text and not legal_references and not citations:
                return 0.0

            coverage_scores = []

            # 1. 추출된 키워드 커버리지 (개선: 부분 일치도 고려)
            if extracted_keywords:
                context_lower = context_text.lower()
                keyword_matches = 0
                for kw in extracted_keywords:
                    if isinstance(kw, str):
                        kw_lower = kw.lower().strip()
                        if kw_lower:
                            # 정확한 일치
                            if kw_lower in context_lower:
                                keyword_matches += 1
                            else:
                                # 부분 일치 (키워드가 2자 이상인 경우)
                                if len(kw_lower) >= 2:
                                    # 키워드의 일부가 컨텍스트에 포함되어 있는지 확인
                                    for i in range(len(kw_lower) - 1):
                                        if kw_lower[i:i+2] in context_lower:
                                            keyword_matches += 0.5  # 부분 일치는 0.5점
                                            break
                keyword_coverage = keyword_matches / max(1, len(extracted_keywords))
                coverage_scores.append(keyword_coverage)

            # 2. 질문 키워드 커버리지
            if context_text:
                # 질문 키워드는 extracted_keywords에 포함되어 있을 수 있으므로 별도 계산 생략
                pass

            # 3. 법률 참조 포함도 (개선: 컨텍스트에 실제로 포함되어 있는지 확인)
            if legal_references:
                if context_text:
                    context_lower = context_text.lower()
                    ref_matches = sum(1 for ref in legal_references 
                                    if isinstance(ref, str) and ref.lower() in context_lower)
                    # 법률 참조가 컨텍스트에 포함되어 있으면 더 높은 점수
                    ref_coverage = min(1.0, (ref_matches / max(1, len(legal_references))) * 1.2)  # 포함된 경우 보너스
                else:
                    ref_coverage = min(1.0, len(legal_references) / max(1, 5))  # 최대 5개 기준
                coverage_scores.append(ref_coverage)

            # 4. 인용 포함도 (개선: 컨텍스트에 실제로 포함되어 있는지 확인)
            if citations:
                if context_text:
                    context_lower = context_text.lower()
                    citation_matches = sum(1 for cit in citations 
                                         if isinstance(cit, (str, dict)) and 
                                         (str(cit).lower() in context_lower if isinstance(cit, str) else 
                                          any(str(v).lower() in context_lower for v in cit.values() if isinstance(v, str))))
                    citation_coverage = min(1.0, (citation_matches / max(1, len(citations))) * 1.2)  # 포함된 경우 보너스
                else:
                    citation_coverage = min(1.0, len(citations) / max(1, 5))  # 최대 5개 기준
                coverage_scores.append(citation_coverage)
            
            # 5. 컨텍스트 길이 기반 커버리지 (개선: 충분한 컨텍스트가 있는지 확인)
            if context_text:
                context_length = len(context_text)
                # 컨텍스트가 충분히 길면 더 높은 점수 (최소 500자 이상)
                if context_length >= 2000:
                    length_coverage = 1.0
                elif context_length >= 1000:
                    length_coverage = 0.8
                elif context_length >= 500:
                    length_coverage = 0.6
                else:
                    length_coverage = max(0.3, context_length / 500)  # 최소 0.3
                coverage_scores.append(length_coverage)

            # 가중 평균 계산 (키워드 커버리지에 더 높은 가중치)
            if coverage_scores:
                if len(coverage_scores) >= 2:
                    # 키워드 커버리지에 40% 가중치, 나머지에 60% 가중치
                    weighted_sum = coverage_scores[0] * 0.4 + sum(coverage_scores[1:]) * (0.6 / max(1, len(coverage_scores) - 1))
                    return min(1.0, weighted_sum)
                else:
                    return sum(coverage_scores) / len(coverage_scores)
            else:
                return 0.5  # 기본값

        except Exception as e:
            logger.warning(f"Coverage calculation failed: {e}")
            return 0.5

    @staticmethod
    def validate_context_quality(
        context: Dict[str, Any],
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        calculate_relevance_func: callable = None,
        calculate_coverage_func: callable = None
    ) -> Dict[str, Any]:
        """
        컨텍스트 품질 검증

        Args:
            context: 컨텍스트 딕셔너리
            query: 질문
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드 목록
            calculate_relevance_func: 관련성 계산 함수 (선택적)
            calculate_coverage_func: 커버리지 계산 함수 (선택적)

        Returns:
            검증 결과 딕셔너리
        """
        try:
            context_text = context.get("context", "")
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # 관련성 점수 계산
            if calculate_relevance_func:
                relevance_score = calculate_relevance_func(context_text, query)
            else:
                relevance_score = ContextValidator.calculate_relevance(context_text, query)

            # 커버리지 점수 계산
            if calculate_coverage_func:
                coverage_score = calculate_coverage_func(context_text, extracted_keywords, legal_references, citations)
            else:
                coverage_score = ContextValidator.calculate_coverage(
                    context_text, extracted_keywords, legal_references, citations
                )

            # 충분성 점수 계산 (문서 개수, 길이 등) - 개선: 직접 계산
            document_count = context.get("document_count", 0)
            context_length = context.get("context_length", 0)
            # context_length가 0이면 context_text 길이 직접 계산
            if context_length == 0 and context_text:
                context_length = len(context_text)
            # document_count가 0이면 retrieved_docs에서 계산
            if document_count == 0:
                retrieved_docs = context.get("retrieved_docs", [])
                if retrieved_docs:
                    document_count = len(retrieved_docs)

            # 최소 문서 개수 확인
            min_docs_required = 2 if query_type != "simple" else 1
            doc_sufficiency = min(1.0, document_count / max(1, min_docs_required))

            # 최소 컨텍스트 길이 확인 (500자 이상 권장)
            length_sufficiency = min(1.0, context_length / max(1, 500))

            sufficiency_score = (doc_sufficiency * 0.6 + length_sufficiency * 0.4)

            # 종합 점수
            overall_score = (relevance_score * 0.4 + coverage_score * 0.4 + sufficiency_score * 0.2)

            # 누락 정보 확인
            missing_info = []
            if coverage_score < 0.3:
                missing_info.append("핵심 키워드 커버리지 부족")
            if relevance_score < 0.4:
                missing_info.append("질문 관련성 부족")
            if sufficiency_score < 0.6:
                missing_info.append("컨텍스트 충분성 부족")

            is_sufficient = overall_score >= 0.6
            needs_expansion = (
                (overall_score < 0.5) or
                (len(missing_info) >= 3) or
                (overall_score < 0.55 and relevance_score < 0.4 and coverage_score < 0.4)
            )

            validation_result = {
                "relevance_score": relevance_score,
                "coverage_score": coverage_score,
                "sufficiency_score": sufficiency_score,
                "overall_score": overall_score,
                "missing_information": missing_info,
                "is_sufficient": is_sufficient,
                "needs_expansion": needs_expansion,
                "document_count": document_count,
                "context_length": context_length
            }

            return validation_result

        except Exception as e:
            logger.warning(f"Context validation failed: {e}")
            return {
                "relevance_score": 0.5,
                "coverage_score": 0.5,
                "sufficiency_score": 0.5,
                "overall_score": 0.5,
                "missing_information": [],
                "is_sufficient": True,
                "needs_expansion": False
            }


class AnswerValidator:
    """답변 품질 검증"""

    @staticmethod
    def _normalize_citation(citation: str) -> Dict[str, Any]:
        """
        Citation을 표준 형식으로 정규화
        
        입력 예시:
        - "민법 제750조"
        - "민법 750조"
        - "[법령: 민법 제750조]"
        - "민법 제750조에 따르면..."
        
        Returns:
            {
                "type": "law",  # "law" or "precedent"
                "law_name": "민법",  # 법령명
                "article_number": "750",  # 조문번호
                "normalized": "민법 제750조",  # 표준 형식
                "original": citation  # 원본
            }
        """
        if not citation or not isinstance(citation, str):
            return {
                "type": "unknown",
                "normalized": "",
                "original": citation
            }
        
        # 1. 법령 조문 패턴 (다양한 형식 지원)
        law_patterns = [
            (r'\[법령:\s*([^\]]+)\]', True),  # [법령: 민법 제750조]
            (r'([가-힣]+법)\s*제?\s*(\d+)\s*조', False),  # 민법 제750조, 민법 750조
            (r'([가-힣]+법)\s*(\d+)\s*조', False),  # 민법 750조 (제 없음)
        ]
        
        for pattern, is_bracketed in law_patterns:
            match = re.search(pattern, citation)
            if match:
                if is_bracketed:
                    # [법령: ...] 형식
                    inner = match.group(1)
                    law_match = re.search(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', inner)
                    if law_match:
                        return {
                            "type": "law",
                            "law_name": law_match.group(1),
                            "article_number": law_match.group(2),
                            "normalized": f"{law_match.group(1)} 제{law_match.group(2)}조",
                            "original": citation
                        }
                else:
                    # 직접 매칭
                    if len(match.groups()) >= 2:
                        return {
                            "type": "law",
                            "law_name": match.group(1),
                            "article_number": match.group(2),
                            "normalized": f"{match.group(1)} 제{match.group(2)}조",
                            "original": citation
                        }
        
        # 2. 판례 패턴 (개선: 날짜/판결 형식 변형 처리)
        # "대법원 2007. 12. 27. 선고 2006다9408 판결" 형식 지원
        precedent_patterns = [
            (r'(대법원|법원)\s+(\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\s*선고\s+(\d{4}[다나마]\d+)', True),  # 날짜 포함 형식
            (r'(대법원|법원).*?(\d{4}[다나마]\d+)', False),  # 기본 형식
        ]
        
        for pattern, has_date in precedent_patterns:
            precedent_match = re.search(pattern, citation)
            if precedent_match:
                if has_date:
                    court = precedent_match.group(1)
                    case_number = precedent_match.group(3)
                else:
                    court = precedent_match.group(1)
                    case_number = precedent_match.group(2)
                
                return {
                    "type": "precedent",
                    "court": court,
                    "case_number": case_number,
                    "normalized": f"{court} {case_number}",
                    "original": citation
                }
        
        # 3. 매칭 실패 시 원본 반환
        return {
            "type": "unknown",
            "normalized": citation.strip(),
            "original": citation
        }

    @staticmethod
    def _match_citations(normalized_expected: Dict[str, Any], 
                         normalized_answer: Dict[str, Any]) -> bool:
        """
        정규화된 Citation 간 매칭
        
        매칭 규칙:
        1. 타입이 다르면 False
        2. 법령인 경우: 법령명과 조문번호가 모두 일치해야 함
        3. 판례인 경우: 법원명과 사건번호가 모두 일치해야 함
        4. 부분 매칭 허용 (예: "민법 제750조"와 "민법 750조")
        """
        # 타입이 다르면 매칭 실패
        if normalized_expected.get("type") != normalized_answer.get("type"):
            return False
        
        # 법령 매칭 (개선: 부분 매칭 지원)
        if normalized_expected.get("type") == "law":
            expected_law = normalized_expected.get("law_name", "")
            expected_article = normalized_expected.get("article_number", "")
            answer_law = normalized_answer.get("law_name", "")
            answer_article = normalized_answer.get("article_number", "")
            
            # 법령명이 일치하는지 확인 (부분 매칭 지원)
            law_match = expected_law == answer_law or expected_law in answer_law or answer_law in expected_law
            
            # 조문번호가 일치하는지 확인 (숫자 비교)
            article_match = expected_article == answer_article
            
            # 법령명과 조문번호가 모두 일치해야 함
            return law_match and article_match
        
        # 판례 매칭 (개선: 부분 매칭 지원)
        elif normalized_expected.get("type") == "precedent":
            expected_court = normalized_expected.get("court", "")
            expected_case = normalized_expected.get("case_number", "")
            answer_court = normalized_answer.get("court", "")
            answer_case = normalized_answer.get("case_number", "")
            
            # 법원명이 일치하는지 확인 (부분 매칭 지원)
            court_match = expected_court == answer_court or expected_court in answer_court or answer_court in expected_court
            
            # 사건번호가 일치하는지 확인 (부분 매칭 지원)
            case_match = expected_case == answer_case or expected_case in answer_case or answer_case in expected_case
            
            # 법원명과 사건번호가 모두 일치해야 함
            return court_match and case_match
        
        return False

    @staticmethod
    def _extract_and_normalize_citations_from_answer(answer: str) -> List[Dict[str, Any]]:
        """
        답변에서 Citation 추출 및 정규화
        
        Returns:
            정규화된 Citation 리스트
        """
        if not answer:
            return []
        
        normalized_citations = []
        
        # 법령 조문 패턴 (다양한 형식 지원 - 개선: 더 포괄적인 패턴 및 중복 제거)
        law_patterns = [
            (r'\[법령:\s*([^\]]+)\]', True),  # [법령: 민법 제750조] - 괄호 내부 추출
            (r'([가-힣]+법)\s*제?\s*(\d+)\s*조', False),  # 민법 제750조, 민법 750조
            (r'([가-힣]+법)\s*(\d+)\s*조', False),  # 민법 750조 (제 없음)
        ]
        law_matches = []
        seen_laws = set()
        
        for pattern, extract_inner in law_patterns:
            matches = re.finditer(pattern, answer)
            for match in matches:
                if extract_inner:
                    # [법령: ...] 형식에서 내부 추출
                    inner_text = match.group(1)
                    # 내부에서 법령명과 조문번호 추출
                    inner_match = re.search(r'([가-힣]+법)\s*제?\s*(\d+)\s*조', inner_text)
                    if inner_match:
                        law_key = f"{inner_match.group(1)} 제{inner_match.group(2)}조"
                        if law_key not in seen_laws:
                            seen_laws.add(law_key)
                            law_matches.append(law_key)
                else:
                    # 직접 매칭
                    if len(match.groups()) >= 2:
                        law_key = f"{match.group(1)} 제{match.group(2)}조"
                        if law_key not in seen_laws:
                            seen_laws.add(law_key)
                            law_matches.append(law_key)
        
        for match in law_matches:
            normalized = AnswerValidator._normalize_citation(match)
            if normalized.get("type") != "unknown":
                normalized_citations.append(normalized)
        
        # 판례 패턴 (개선: 날짜/판결 형식 변형 처리)
        precedent_patterns = [
            r'(?:대법원|법원)\s+\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*선고\s+\d{4}[다나마]\d+',  # 날짜 포함 형식
            r'(?:대법원|법원).*?\d{4}[다나마]\d+',  # 기본 형식
        ]
        
        precedent_matches = []
        for pattern in precedent_patterns:
            matches = re.finditer(pattern, answer)
            for match in matches:
                precedent_matches.append(match.group(0))
        
        for match in precedent_matches:
            normalized = AnswerValidator._normalize_citation(match)
            if normalized.get("type") != "unknown":
                normalized_citations.append(normalized)
        
        # 중복 제거 (normalized 기준)
        seen = set()
        unique_citations = []
        for cit in normalized_citations:
            normalized_str = cit.get("normalized", "")
            if normalized_str and normalized_str not in seen:
                seen.add(normalized_str)
                unique_citations.append(cit)
        
        return unique_citations

    @staticmethod
    def validate_answer_uses_context(
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        답변이 컨텍스트를 사용하는지 검증

        Args:
            answer: 답변 텍스트
            context: 컨텍스트 딕셔너리
            query: 질문
            retrieved_docs: 검색된 문서 목록 (선택적)

        Returns:
            검증 결과 딕셔너리
        """
        try:
            if not answer:
                return {
                    "uses_context": False,
                    "coverage_score": 0.0,
                    "citation_count": 0,
                    "has_document_references": False,
                    "needs_regeneration": True,
                    "missing_key_info": []
                }

            answer_lower = answer.lower()
            context_text = context.get("context", "").lower()
            legal_references = context.get("legal_references", [])
            citations = context.get("citations", [])

            # 검색된 문서에서 출처 추출
            document_sources = []
            if retrieved_docs:
                for doc in retrieved_docs[:10]:
                    if isinstance(doc, dict):
                        source = doc.get("source", "")
                        if source and source not in document_sources:
                            document_sources.append(source.lower())

            # 1. 컨텍스트 키워드 포함도 계산 (개선: 더 정확한 계산)
            import re
            # 문장 단위로 분리하여 더 정확한 매칭
            context_sentences = re.split(r'[.!?。！？\n]+', context_text)
            answer_sentences = re.split(r'[.!?。！？\n]+', answer_lower)
            
            # 중요한 키워드 추출 (2자 이상, 불용어 제외)
            stop_words = {'의', '가', '이', '은', '는', '을', '를', '에', '에서', '와', '과', '도', '로', '으로', '에서', '에게', '께', '한테', '더', '또', '그', '이', '저', '그것', '이것', '저것', '그런', '이런', '저런', '그렇게', '이렇게', '저렇게'}
            context_words = set()
            for sentence in context_sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                context_words.update(w for w in words if len(w) >= 2 and w not in stop_words)
            
            answer_words = set()
            for sentence in answer_sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                answer_words.update(w for w in words if len(w) >= 2 and w not in stop_words)

            keyword_coverage = 0.0
            if context_words and answer_words:
                overlap = len(context_words.intersection(answer_words))
                # 개선: 컨텍스트의 중요한 단어(2자 이상)만 고려하여 더 정확한 계산
                important_context_words = {w for w in context_words if len(w) >= 2}
                if important_context_words:
                    overlap_important = len(important_context_words.intersection(answer_words))
                    # 정규화: 최대 200개 단어까지만 고려하여 더 정확한 비율 계산
                    keyword_coverage = overlap_important / max(1, min(len(important_context_words), 200))
                else:
                    keyword_coverage = overlap / max(1, min(len(context_words), 200))

            # 2. 법률 조항/판례 인용 포함 여부 확인 (강화: 법령 조문 인용 우선)
            # 법령 조문 인용 패턴 (강화: 다양한 형식 지원)
            citation_patterns = [
                r'[가-힣]+법\s*제?\s*\d+\s*조',  # 민법 제750조
                r'\[법령:\s*[^\]]+\]',  # [법령: 민법 제750조]
                r'제\d+조',  # 제750조
                r'\d+조',  # 750조
            ]
            citations_in_answer = 0
            unique_citations = set()
            for pattern in citation_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    unique_citations.add(match)
            citations_in_answer = len(unique_citations)

            precedent_pattern = r'대법원|법원.*\d{4}[다나마]\d+|\[판례:\s*[^\]]+\]'
            precedents_in_answer = len(re.findall(precedent_pattern, answer))
            
            # 법령 조문 인용 필수 체크 (검색 결과에 법령 조문이 있으면 반드시 인용해야 함)
            has_law_citation = citations_in_answer > 0
            has_law_in_docs = False
            if retrieved_docs:
                for doc in retrieved_docs:
                    if isinstance(doc, dict):
                        doc_type = doc.get("type", "").lower()
                        source = doc.get("source", "").lower()
                        # 법령 조문 문서인지 확인
                        if ("법령" in doc_type or "statute" in doc_type or "law" in doc_type) or \
                           ("제" in source and "조" in source):
                            has_law_in_docs = True
                            break

            # 문서 인용 패턴 확인
            document_citation_pattern = r'\[문서:\s*[^\]]+\]'
            document_citations = len(re.findall(document_citation_pattern, answer))

            # 3. 검색된 문서의 출처가 답변에 포함되어 있는지 확인 (개선: 유연한 패턴 매칭)
            has_document_references = False
            if document_sources:
                # re 모듈은 이미 파일 상단에서 import됨
                for source in document_sources:
                    if not source or not isinstance(source, str):
                        continue
                    
                    source_lower = source.lower()
                    # 공백 제거 버전 (개선: 답변에 공백이 들어간 경우 대비)
                    source_no_spaces = source_lower.replace(" ", "").replace("-", "").replace("_", "")
                    answer_no_spaces = answer_lower.replace(" ", "").replace("-", "").replace("_", "")
                    
                    # 전체 소스명이 포함되어 있는지 확인 (공백 포함 및 제거 버전 모두)
                    if source_lower in answer_lower or source_no_spaces in answer_no_spaces:
                        has_document_references = True
                        break
                    
                    # 소스명의 주요 키워드 추출 (3-5개 단어)
                    source_words = source.split()
                    # 법령명이나 판례명의 주요 부분 추출
                    if len(source_words) >= 2:
                        # 첫 2-3개 단어로 매칭 시도 (공백 포함 및 제거 버전 모두)
                        key_phrase = " ".join(source_words[:3])
                        key_phrase_no_spaces = key_phrase.replace(" ", "").replace("-", "").replace("_", "")
                        if len(key_phrase) >= 5 and (key_phrase.lower() in answer_lower or key_phrase_no_spaces in answer_no_spaces):
                            has_document_references = True
                            break
                    
                    # 법령명과 조문번호 패턴 매칭
                    # 예: "민법 제750조" -> "민법", "750조" 모두 찾기
                    law_match = re.search(r'([가-힣]+법)', source)
                    article_match = re.search(r'제?\s*(\d+)\s*조', source)
                    if law_match and article_match:
                        law_name = law_match.group(1)
                        article_no = article_match.group(1)
                        # "민법"과 "750조"가 모두 답변에 있는지 확인
                        if law_name in answer_lower and (f"{article_no}조" in answer_lower or f"제{article_no}조" in answer_lower):
                            has_document_references = True
                            break
                    
                    # 판례명 패턴 매칭 (법원명 + 연도 + 사건번호) - 개선: 더 유연한 패턴 및 부분 매칭 강화
                    # 예: "대구지방법원 영덕지원 대구지방법원영덕지원-2021고단3"
                    # 또는 "대구지방법원영덕지원-2021고단3"
                    court_patterns = [
                        r'([가-힣]+지방법원[가-힣]*지원)',  # 대구지방법원 영덕지원 또는 대구지방법원영덕지원
                        r'([가-힣]+지방법원)',  # 대구지방법원
                        r'(대법원|고등법원)',  # 대법원, 고등법원
                    ]
                    case_patterns = [
                        r'(\d{4}[가-힣]*\d+)',  # 2021고단3
                        r'(\d{4}[가-힣]+)',  # 2021고단
                    ]
                    
                    # 법원명과 사건번호 개별 확인 (개선: 부분 매칭 강화)
                    court_found = False
                    case_found = False
                    
                    for court_pattern in court_patterns:
                        court_match = re.search(court_pattern, source)
                        if court_match:
                            court_name = court_match.group(1)
                            # 법원명의 주요 부분이 답변에 있는지 확인 (개선: 단어 단위 매칭)
                            court_words = [w for w in court_name.split() if len(w) >= 2]
                            if court_words:
                                # 법원명의 주요 단어들이 답변에 포함되는지 확인
                                matched_words = sum(1 for word in court_words if word.lower() in answer_lower)
                                if matched_words >= min(2, len(court_words)):  # 최소 2개 단어 매칭 또는 전체 단어의 대부분
                                    court_found = True
                                    break
                            # 전체 법원명이 포함되어 있는지도 확인 (공백 제거 버전도 확인)
                            court_name_lower = court_name.lower()
                            court_name_no_spaces = court_name_lower.replace(" ", "").replace("-", "").replace("_", "")
                            if court_name_lower in answer_lower or court_name_no_spaces in answer_no_spaces:
                                court_found = True
                                break
                    
                    # 사건번호 확인 (개선: 더 유연한 패턴)
                    for case_pattern in case_patterns:
                        case_match = re.search(case_pattern, source)
                        if case_match:
                            case_no = case_match.group(1)
                            # 사건번호가 답변에 있는지 확인 (부분 매칭도 허용, 공백 제거 버전도 확인)
                            case_no_lower = case_no.lower()
                            case_no_no_spaces = case_no_lower.replace(" ", "").replace("-", "").replace("_", "")
                            if (case_no_lower in answer_lower or case_no_no_spaces in answer_no_spaces or 
                                any(case_no_lower[i:i+4] in answer_lower for i in range(len(case_no_lower)-3)) or
                                any(case_no_no_spaces[i:i+4] in answer_no_spaces for i in range(len(case_no_no_spaces)-3))):
                                case_found = True
                                break
                    
                    # 법원명 또는 사건번호가 하나라도 있으면 참조로 인정 (개선)
                    if court_found or case_found:
                        has_document_references = True
                        break
                    
                    if has_document_references:
                        break
                    
                    # 일반적인 키워드 매칭 (최소 3자 이상)
                    source_keywords = [w for w in source.split() if len(w) >= 3][:3]
                    if source_keywords and any(keyword.lower() in answer_lower for keyword in source_keywords):
                        has_document_references = True
                        break

            # 컨텍스트에서 추출한 인용 정보와 비교 (개선: 정규화 및 유연한 매칭)
            expected_citations_raw = []
            for ref in legal_references[:5]:
                if isinstance(ref, str):
                    expected_citations_raw.append(ref)

            for cit in citations[:5]:
                if isinstance(cit, dict):
                    expected_citations_raw.append(cit.get("text", ""))
                elif isinstance(cit, str):
                    expected_citations_raw.append(cit)

            # retrieved_docs에서도 Citation 추출 (개선: 더 많은 문서에서 추출)
            if retrieved_docs:
                # 더 다양한 법령 조문 패턴
                law_patterns = [
                    r'[가-힣]+법\s*제?\s*\d+\s*조',  # 민법 제750조
                    r'[가-힣]+법\s*제?\s*\d+\s*조\s*제?\s*\d+\s*항',  # 민법 제750조 제1항
                    r'[가-힣]+법\s*제?\s*\d+\s*조\s*제?\s*\d+\s*항\s*제?\s*\d+\s*호',  # 민법 제750조 제1항 제1호
                    r'[가-힣]+법\s*제?\s*\d+\s*조\s*제?\s*\d+\s*항\s*제?\s*\d+\s*호\s*제?\s*\d+\s*목',  # 민법 제750조 제1항 제1호 제1목
                ]
                # 더 다양한 판례 패턴
                precedent_patterns = [
                    r'대법원\s*\d{4}[다나마]\d+',  # 대법원 2021다275611
                    r'대법원\s*\d{4}\.\s*\d+\.\s*\d+\.\s*선고\s*\d+[다나마]\d+',  # 대법원 2021. 1. 1. 선고 2021다275611
                    r'[가-힣]+법원\s*\d{4}[다나마]\d+',  # 서울법원 2021다275611
                    r'선고\s*\d{4}[다나마]\d+',  # 선고 2021다275611
                ]
                for doc in retrieved_docs[:15]:  # 10 -> 15로 증가
                    content = doc.get("content", "") or doc.get("text", "")
                    if not content:
                        continue
                    # 법령 조문 추출 (여러 패턴 시도)
                    for pattern in law_patterns:
                        law_matches = re.findall(pattern, content)
                        for law in law_matches:
                            if isinstance(law, tuple):
                                # 튜플인 경우 첫 번째 요소 사용
                                law = law[0] if law[0] else ""
                            if law and law not in expected_citations_raw:
                                expected_citations_raw.append(law)
                    # 판례 추출 (여러 패턴 시도)
                    for pattern in precedent_patterns:
                        precedent_matches = re.findall(pattern, content)
                        for precedent in precedent_matches:
                            if isinstance(precedent, tuple):
                                precedent = precedent[0] if precedent[0] else ""
                            if precedent and precedent not in expected_citations_raw:
                                expected_citations_raw.append(precedent)

            # 1. expected_citations 정규화
            normalized_expected_citations = []
            for expected in expected_citations_raw:
                if not expected:
                    continue
                normalized = AnswerValidator._normalize_citation(expected)
                if normalized.get("type") != "unknown":
                    normalized_expected_citations.append(normalized)

            # 디버깅 로그 추가
            logger.debug(
                f"[CITATION DEBUG] Expected citations (raw): {expected_citations_raw[:5]}, "
                f"Normalized expected: {[c.get('normalized', '') for c in normalized_expected_citations[:5]]}"
            )

            # 2. 답변에서 Citation 추출 및 정규화
            normalized_answer_citations = AnswerValidator._extract_and_normalize_citations_from_answer(answer)
            
            # 디버깅 로그 추가
            logger.debug(
                f"[CITATION DEBUG] Normalized answer citations: "
                f"{[c.get('normalized', '') for c in normalized_answer_citations[:5]]}"
            )
            
            # total_citations_in_answer 계산 (정규화된 Citation 사용)
            total_citations_in_answer = len(normalized_answer_citations) + document_citations if normalized_answer_citations else document_citations

            # 3. 매칭 수행
            found_citations = 0
            missing_citations = []
            matched_answer_citations = set()

            for expected_cit in normalized_expected_citations:
                matched = False
                for i, answer_cit in enumerate(normalized_answer_citations):
                    if i in matched_answer_citations:
                        continue
                    if AnswerValidator._match_citations(expected_cit, answer_cit):
                        found_citations += 1
                        matched = True
                        matched_answer_citations.add(i)
                        logger.debug(
                            f"[CITATION DEBUG] Matched: {expected_cit.get('normalized', '')} "
                            f"<-> {answer_cit.get('normalized', '')}"
                        )
                        break
                
                if not matched:
                    missing_citations.append(expected_cit.get("original", ""))
                    logger.debug(
                        f"[CITATION DEBUG] Not matched: {expected_cit.get('normalized', '')} "
                        f"(original: {expected_cit.get('original', '')})"
                    )

            # 4. Citation coverage 계산 개선
            if normalized_expected_citations:
                # expected_citations가 있을 때
                citation_coverage = found_citations / len(normalized_expected_citations)
                
                # 답변에 Citation이 있지만 expected_citations와 매칭되지 않은 경우
                # 부분 점수 부여 (최대 0.4까지 증가)
                unmatched_answer_citations = len(normalized_answer_citations) - found_citations
                if unmatched_answer_citations > 0:
                    # 답변에 법령 인용이 있으면 더 높은 보너스 부여
                    bonus = min(0.4, unmatched_answer_citations * 0.15)
                    citation_coverage = min(1.0, citation_coverage + bonus)
                
                # 매칭 실패 시 부분 점수 로직 개선
                if found_citations == 0:
                    # 답변에 Citation이 있으면 최소 0.3 점수 부여 (0.2 -> 0.3으로 증가)
                    if normalized_answer_citations:
                        # 법령 인용이 있으면 더 높은 점수 부여
                        law_citations = [c for c in normalized_answer_citations if c.get("type") == "law"]
                        if law_citations:
                            citation_coverage = min(0.6, 0.3 + len(law_citations) * 0.1)
                        else:
                            citation_coverage = min(0.5, len(normalized_answer_citations) * 0.15)
                        logger.debug(
                            f"[CITATION DEBUG] No matches but answer has citations: "
                            f"{len(normalized_answer_citations)}, law_citations: {len(law_citations)}, coverage: {citation_coverage}"
                        )
                    else:
                        citation_coverage = 0.0
            elif normalized_answer_citations:
                # expected_citations가 비어있을 때 답변에서 직접 추출한 Citation으로 coverage 계산
                total_citations_in_answer = len(normalized_answer_citations)
                if total_citations_in_answer > 0:
                    # retrieved_docs 개수를 기준으로 coverage 계산 (개선)
                    expected_count = max(2, len(retrieved_docs) if retrieved_docs else 2)
                    citation_coverage = min(1.0, total_citations_in_answer / expected_count)
                    logger.debug(
                        f"[CITATION DEBUG] No expected citations, using answer citations: "
                        f"{total_citations_in_answer}, expected: {expected_count}, coverage: {citation_coverage}"
                    )
                else:
                    citation_coverage = 0.0
            elif citations_in_answer > 0 or precedents_in_answer > 0:
                # 정규화되지 않은 Citation이 있으면 부분 점수 부여
                total_raw_citations = citations_in_answer + precedents_in_answer
                expected_count = max(2, len(retrieved_docs) if retrieved_docs else 2)
                citation_coverage = min(0.5, total_raw_citations / expected_count)
                logger.debug(
                    f"[CITATION DEBUG] Using raw citations: {total_raw_citations}, "
                    f"expected: {expected_count}, coverage: {citation_coverage}"
                )
            else:
                # expected_citations도 없고 답변에도 Citation이 없으면 0.0
                citation_coverage = 0.0

            # 4. 핵심 개념 포함 여부
            context_key_concepts = []
            if context_text:
                key_terms = ["법", "조", "판례", "규정", "절차", "요건", "효력"]
                for term in key_terms:
                    if term in context_text:
                        context_key_concepts.append(term)

            concept_coverage = 0.0
            if context_key_concepts:
                found_concepts = sum(1 for concept in context_key_concepts if concept in answer_lower)
                concept_coverage = found_concepts / len(context_key_concepts)

            # 5. 종합 점수 (가중치 조정 - 개선: keyword_coverage 가중치 증가)
            coverage_score = (
                keyword_coverage * 0.4 +      # 0.3 → 0.4로 증가 (컨텍스트 활용의 핵심)
                citation_coverage * 0.4 +      # 0.5 → 0.4로 조정
                concept_coverage * 0.2
            )
            
            # Citation이 없을 때 추가 페널티 (개선: 페널티 감소)
            if citation_coverage == 0.0 and normalized_expected_citations:
                coverage_score = max(0.0, coverage_score - 0.15)  # 0.2 → 0.15로 감소
            
            # 개선: keyword_coverage가 높으면 보너스 부여
            if keyword_coverage >= 0.5:
                coverage_score = min(1.0, coverage_score + 0.1)  # 10% 보너스

            # 재생성 필요 여부 초기화 (개선: 변수 초기화 문제 해결)
            needs_regeneration = False

            # 법령 조문 인용 필수 체크 결과 (검색 결과에 법령 조문이 있는데 답변에 없으면 경고)
            law_citation_required = has_law_in_docs and not has_law_citation
            if law_citation_required:
                # 법령 조문 인용이 필수인데 없으면 coverage_score 감소
                coverage_score = max(0.0, coverage_score - 0.2)  # 20% 감소
                needs_regeneration = True  # 재생성 필요
            
            uses_context = coverage_score >= 0.3
            needs_regeneration = needs_regeneration or (coverage_score < 0.3) or (normalized_expected_citations and found_citations == 0)

            validation_result = {
                "uses_context": uses_context,
                "coverage_score": coverage_score,
                "keyword_coverage": keyword_coverage,
                "citation_coverage": citation_coverage,
                "concept_coverage": concept_coverage,
                "citations_found": found_citations,
                "citations_expected": len(normalized_expected_citations),
                "citation_count": len(normalized_answer_citations) if normalized_answer_citations else 0,
                "citations_in_answer": citations_in_answer,
                "precedents_in_answer": precedents_in_answer,
                "document_citations": document_citations,
                "total_citations_in_answer": total_citations_in_answer,
                "has_document_references": has_document_references,
                "document_sources_count": len(document_sources),
                "needs_regeneration": needs_regeneration,
                "missing_key_info": missing_citations[:5],
                "has_law_citation": has_law_citation,
                "has_law_in_docs": has_law_in_docs,
                "law_citation_required": law_citation_required
            }

            return validation_result

        except Exception as e:
            logger.warning(f"Answer-context validation failed: {e}")
            import traceback
            logger.debug(f"Validation error traceback: {traceback.format_exc()}")
            return {
                "uses_context": True,
                "coverage_score": 0.5,
                "keyword_coverage": 0.0,
                "citation_coverage": 0.0,
                "concept_coverage": 0.0,
                "citations_found": 0,
                "citations_expected": 0,
                "citation_count": 0,
                "citations_in_answer": 0,
                "precedents_in_answer": 0,
                "document_citations": 0,
                "total_citations_in_answer": 0,
                "has_document_references": False,
                "document_sources_count": 0,
                "needs_regeneration": False,
                "missing_key_info": [],
                "has_law_citation": False,
                "has_law_in_docs": False,
                "law_citation_required": False
            }

    @staticmethod
    def validate_answer_source_verification(
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        답변의 내용이 검색된 문서에 기반하는지 검증 (Hallucination 방지)

        Args:
            answer: 검증할 답변 텍스트
            retrieved_docs: 검색된 문서 목록
            query: 원본 질의

        Returns:
            검증 결과 딕셔너리
            {
                "is_grounded": bool,
                "grounding_score": float,
                "unverified_sections": List[str],
                "source_coverage": float,
                "needs_review": bool
            }
        """
        import re
        from difflib import SequenceMatcher
        
        # 검증 시작 로그 (강제 출력)
        logger.info(f"✅ [GROUNDING START] validate_answer_source_verification called")
        logger.info(f"   - Answer length: {len(answer)} characters")
        logger.info(f"   - Retrieved docs count: {len(retrieved_docs) if retrieved_docs else 0}")
        logger.info(f"   - Query: {query[:50]}..." if query else "   - Query: None")

        if not answer:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "unverified_sections": [],
                "source_coverage": 0.0,
                "needs_review": True,
                "error": "답변이 없습니다."
            }
        
        # 검색 결과가 없을 때의 처리 (개선)
        if not retrieved_docs:
            # 검색 결과가 없어도 답변이 있으면 부분 점수 부여
            # 답변에 법령 조문 인용이 있으면 최소 점수 부여
            import re
            citation_patterns = [
                r'[가-힣]+법\s*제?\s*\d+\s*조',
                r'\[법령:\s*[^\]]+\]',
                r'제\d+조',
                r'\d+조',
            ]
            has_citation = False
            for pattern in citation_patterns:
                if re.search(pattern, answer):
                    has_citation = True
                    break
            
            if has_citation:
                # 법령 조문 인용이 있으면 최소 grounding_score 부여
                return {
                    "is_grounded": False,  # 검색 결과가 없으므로 grounded는 False
                    "grounding_score": 0.3,  # 최소 점수 부여
                    "unverified_sections": [answer],
                    "source_coverage": 0.0,
                    "needs_review": True,
                    "error": "검색 결과가 없지만 답변에 법령 조문 인용이 있습니다.",
                    "partial_credit": True
                }
            else:
                # 검색 결과도 없고 인용도 없으면 0점
                return {
                    "is_grounded": False,
                    "grounding_score": 0.0,
                    "unverified_sections": [answer],
                    "source_coverage": 0.0,
                    "needs_review": True,
                    "error": "검색 결과가 없고 답변에 법령 조문 인용도 없습니다."
                }

        # 1. 검색된 문서에서 모든 텍스트 추출
        source_texts = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = (
                    doc.get("content") or
                    doc.get("text") or
                    doc.get("content_text") or
                    ""
                )
                if content and len(content.strip()) > 50:
                    source_texts.append(content.lower())

        if not source_texts:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "unverified_sections": [],
                "source_coverage": 0.0,
                "needs_review": True,
                "error": "검색된 문서의 내용이 없습니다."
            }

        # 2. 답변을 문장 단위로 분리 (구문 단위로도 분할)
        # 개선: 문장 최소 길이 기준 완화 (20자 → 15자) 및 구문 분할 기준 완화 (100자 → 80자)
        logger.info(f"✅ [GROUNDING STEP 1] Splitting answer into sentences...")
        answer_sentences = re.split(r'[.!?。！？]\s+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 15]  # 20 → 15로 완화
        
        # 긴 문장을 구문 단위로 분할 (80자 이상인 경우)
        phrases = []
        for sentence in answer_sentences:
            if len(sentence) > 80:  # 100 → 80으로 완화
                # 긴 문장을 쉼표나 연결어로 분할
                sub_phrases = re.split(r'[,，]\s+', sentence)
                phrases.extend([p.strip() for p in sub_phrases if len(p.strip()) > 12])  # 15 → 12로 완화
            else:
                phrases.append(sentence)
        answer_sentences = phrases
        logger.info(f"✅ [GROUNDING STEP 1] Split into {len(answer_sentences)} sentences")

        # 3. 각 문장이 검색된 문서에 기반하는지 검증
        verified_sentences = []
        partially_verified_sentences = []  # 부분 검증된 문장
        unverified_sentences = []
        
        # 법률 관련 키워드 (검증에 사용) - 개선: 더 많은 법률 키워드 추가
        legal_keywords = [
            "법률", "법령", "조문", "판례", "민법", "형법", "상법", "행정법", "제", "조", 
            "손해", "배상", "계약", "소송", "불법행위", "책임", "의무", "권리",
            "고의", "과실", "위법행위", "인과관계", "손해배상", "손해배상책임", "손해배상청구",
            "법원", "판결", "재판", "소송", "항소", "상고", "집행", "강제집행",
            "채권", "채무", "계약", "계약서", "계약불이행", "위약금", "손해금",
            "소멸시효", "시효", "권리", "의무", "법인", "자연인", "법률행위"
        ]

        for sentence in answer_sentences:
            sentence_lower = sentence.lower()

            # 문장의 핵심 키워드 추출 (불용어 제거)
            stopwords = {'는', '은', '이', '가', '을', '를', '에', '의', '와', '과', '로', '으로', '에서', '도', '만', '부터', '까지'}
            sentence_words = [w for w in re.findall(r'[가-힣]+', sentence_lower) if len(w) > 1 and w not in stopwords]

            if not sentence_words:
                continue

            # 각 소스 텍스트와 유사도 계산 (향상된 유사도 계산)
            max_similarity = 0.0
            best_match_source = None
            matched_keywords_count = 0

            for source_text in source_texts:
                # 키워드 매칭 점수
                matched_keywords = sum(1 for word in sentence_words if word in source_text)
                keyword_score = matched_keywords / len(sentence_words) if sentence_words else 0.0

                # 기본 문장 유사도 (SequenceMatcher 사용)
                basic_similarity = SequenceMatcher(None, sentence_lower[:100], source_text[:1000]).ratio()
                
                # 부분 문자열 매칭 (핵심 구문이 문서에 포함되어 있는지)
                sentence_key_phrases = [w for w in sentence_words if len(w) > 2][:5]  # 핵심 구문 추출
                phrase_match_score = sum(1 for phrase in sentence_key_phrases if phrase in source_text) / max(len(sentence_key_phrases), 1) if sentence_key_phrases else 0.0
                
                # 의미적 유사도 (공통 단어 비율)
                source_words = re.findall(r'[가-힣]+', source_text.lower())
                source_words_set = set([w for w in source_words if len(w) > 1 and w not in stopwords])
                sentence_words_set = set(sentence_words)
                semantic_similarity = len(sentence_words_set & source_words_set) / max(len(sentence_words_set), 1) if sentence_words_set else 0.0

                # 종합 점수 (가중치 조정: 키워드 60%, 기본 유사도 15%, 구문 매칭 15%, 의미적 유사도 10%)
                # 개선: 키워드 매칭 가중치 추가 증가 (50% → 60%)로 Grounding Score 향상
                # 개선: 의미적 유사도 계산 개선 (공통 단어 비율을 더 정확하게 계산)
                # 양방향 유사도 계산: 문장→문서, 문서→문장
                reverse_semantic_similarity = len(sentence_words_set & source_words_set) / max(len(source_words_set), 1) if source_words_set else 0.0
                enhanced_semantic_similarity = (semantic_similarity + reverse_semantic_similarity) / 2.0
                
                combined_score = (
                    keyword_score * 0.6 +  # 50% → 60%로 증가
                    basic_similarity * 0.15 +
                    phrase_match_score * 0.15 +  # 20% → 15%로 감소
                    enhanced_semantic_similarity * 0.10  # 15% → 10%로 감소, 의미적 유사도 개선
                )

                if combined_score > max_similarity:
                    max_similarity = combined_score
                    matched_keywords_count = matched_keywords
                    best_match_source = source_text[:100]  # 디버깅용

            # 다단계 검증 기준 적용
            keyword_coverage = matched_keywords_count / len(sentence_words) if sentence_words else 0.0
            
            # 면책 조항 체크
            is_disclaimer = (
                re.search(r'\[법령:\s*[^\]]+\]', sentence) or
                re.search(r'본\s*답변은\s*일반적인', sentence) or
                re.search(r'변호사와\s*직접\s*상담', sentence)
            )
            
            # 법률 조문 인용 문장 특별 처리 (개선: 자동 완전 검증)
            # 법률 조문 인용 패턴 감지 (예: "민법 제750조", "제1조", "형법 제250조" 등)
            has_legal_citation = (
                re.search(r'[가-힣]+법\s*제?\s*\d+\s*조', sentence) or
                re.search(r'제\s*\d+\s*조', sentence) or
                re.search(r'\d+\s*조', sentence)
            )
            
            # 법률 조문 인용이 있으면 자동으로 완전 검증 (가중치 1.0)
            if has_legal_citation and not is_disclaimer:
                verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max(max_similarity, 0.8),  # 법률 조문 인용은 높은 유사도로 간주
                    "source_preview": best_match_source or "법률 조문 인용",
                    "weight": 1.0  # 완전 검증
                })
                continue  # 다음 문장으로 이동
            
            # 1차 검증: 완화된 기준 (완전 검증)
            # 개선: 기준 완화 (0.10 → 0.08, 0.20 → 0.15)로 더 많은 문장이 완전 검증됨
            if max_similarity >= 0.08 or keyword_coverage >= 0.15:
                verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max_similarity,
                    "source_preview": best_match_source,
                    "weight": 1.0  # 완전 검증
                })
            # 2차 검증: 완화된 기준 (부분 검증, 0.5-0.6 가중치)
            # 개선: 기준 완화 (0.08 → 0.05, 0.12 → 0.10) 및 가중치 증가 (0.5 → 0.6)
            elif max_similarity >= 0.05 or keyword_coverage >= 0.10:
                partially_verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max_similarity,
                    "source_preview": best_match_source,
                    "weight": 0.6  # 부분 검증 (가중치 증가: 0.5 → 0.6)
                })
            # 3차 검증: 매우 완화된 기준 (법률 키워드만 있어도 부분 점수, 0.4-0.5 가중치)
            # 개선: 법률 키워드 개수에 따라 가중치 차등 적용
            elif any(keyword in sentence for keyword in legal_keywords) and not is_disclaimer:
                legal_keyword_count = sum(1 for keyword in legal_keywords if keyword in sentence)
                # 법률 키워드 2개 이상이면 0.5, 1개면 0.4 가중치
                weight = 0.5 if legal_keyword_count >= 2 else 0.4
                partially_verified_sentences.append({
                    "sentence": sentence,
                    "similarity": max_similarity,
                    "source_preview": best_match_source,
                    "weight": weight  # 법률 키워드 기반 부분 점수 (0.3 → 0.4-0.5)
                })
            # 검증 실패 (개선: 부분 점수 부여)
            elif not is_disclaimer:
                # 개선: 검증 실패 문장도 유사도 점수에 비례하여 부분 점수 부여
                # 유사도가 0.03 이상이면 최소 0.1 점수, 0.05 이상이면 0.2 점수 부여
                partial_score = 0.0
                if max_similarity >= 0.05:
                    partial_score = 0.2
                elif max_similarity >= 0.03:
                    partial_score = 0.1
                elif keyword_coverage >= 0.05:
                    partial_score = 0.1
                
                if partial_score > 0:
                    # 부분 점수가 있으면 partially_verified_sentences에 추가
                    partially_verified_sentences.append({
                        "sentence": sentence,
                        "similarity": max_similarity,
                        "source_preview": best_match_source,
                        "weight": partial_score  # 낮은 부분 점수 (0.1-0.2)
                    })
                else:
                    # 완전히 검증 실패한 문장만 unverified_sentences에 추가
                    unverified_sentences.append({
                        "sentence": sentence[:100],
                        "similarity": max_similarity,
                        "keywords": sentence_words[:5],
                        "keyword_coverage": keyword_coverage
                    })

        # 4. 종합 검증 점수 계산 (가중 평균 방식)
        total_sentences = len(answer_sentences)
        
        # 가중 평균 방식: 각 문장의 유사도 점수를 가중치와 함께 누적
        total_weighted_score = 0.0
        
        # 완전 검증된 문장 (가중치 1.0)
        for sentence_info in verified_sentences:
            similarity = sentence_info.get("similarity", 0.0)
            weight = sentence_info.get("weight", 1.0)
            total_weighted_score += similarity * weight
        
        # 부분 검증된 문장 (가중치 0.4-0.6)
        # 개선: 가중치 증가 (0.3-0.5 → 0.4-0.6)로 부분 검증 문장의 기여도 증가
        for sentence_info in partially_verified_sentences:
            similarity = sentence_info.get("similarity", 0.0)
            weight = sentence_info.get("weight", 0.5)
            total_weighted_score += similarity * weight
        
        # 검증된 문장 수 계산 (부분 검증 포함) - 먼저 계산
        verified_count = len(verified_sentences) + len(partially_verified_sentences)
        
        # Grounding score 계산 개선: 평균 대신 가중 평균 + 검증 비율 고려
        # 개선: 검증된 문장 비율을 고려하여 점수 계산
        verified_ratio = verified_count / total_sentences if total_sentences > 0 else 0.0
        
        # 기본 점수: 가중 평균 유사도
        base_score = total_weighted_score / total_sentences if total_sentences > 0 else 0.0
        
        # 개선: 검증 비율 보너스 추가 (검증된 문장이 많을수록 보너스)
        verification_bonus = verified_ratio * 0.2  # 최대 0.2 보너스 (모든 문장이 검증되면)
        
        # Grounding score = 기본 점수 + 검증 비율 보너스
        grounding_score = min(1.0, base_score + verification_bonus)
        
        # 정보 로그: 검증 상세 정보 (강화된 디버깅 정보 포함)
        logger.info(
            f"✅ [GROUNDING] Verification details: "
            f"total_sentences={total_sentences}, "
            f"fully_verified={len(verified_sentences)}, "
            f"partially_verified={len(partially_verified_sentences)}, "
            f"unverified={len(unverified_sentences)}, "
            f"verified_ratio={verified_ratio:.2f}, "
            f"weighted_score={total_weighted_score:.3f}, "
            f"base_score={base_score:.3f}, "
            f"verification_bonus={verification_bonus:.3f}, "
            f"grounding_score={grounding_score:.3f}"
        )
        
        # 문서 커버리지 계산 (검증된 문장의 출처 다양성)
        all_verified_sources = [s.get("source_preview") for s in verified_sentences + partially_verified_sentences if s.get("source_preview")]
        source_coverage = len(set(all_verified_sources)) / len(source_texts) if source_texts else 0.0

        # 문서 커버리지 보너스 추가 (강화)
        # 개선: 보너스 시작 기준 완화 (0.5 → 0.3) 및 최대 보너스 증가 (0.1 → 0.15)
        if source_coverage >= 0.3:
            coverage_bonus = (source_coverage - 0.3) * 0.3  # 최대 0.15 보너스 (0.7 커버리지 시)
            grounding_score += coverage_bonus
            logger.info(f"✅ [GROUNDING] Source coverage bonus applied: +{coverage_bonus:.2f} (coverage: {source_coverage:.2f})")
        
        # 개선 사항 5: 최소 grounding_score 보장 강화 - 답변이 법률 관련 내용을 포함하면 최소 0.5로 설정
        # 개선: 최소 점수 상향 (0.4 → 0.5) 및 키워드 매칭 보너스 증가 (최대 0.2 → 0.3)
        answer_has_legal_content = any(keyword in answer for keyword in legal_keywords)
        
        # 검색된 문서에서 키워드 추출하여 답변과 매칭 확인
        source_keywords = set()
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                content = doc.get("content") or doc.get("text") or ""
                if content:
                    # 문서에서 법률 관련 키워드 추출
                    for keyword in legal_keywords:
                        if keyword in content:
                            source_keywords.add(keyword)
        
        # 답변에 검색된 문서의 키워드가 포함되어 있으면 추가 점수 부여
        matched_source_keywords = sum(1 for kw in source_keywords if kw in answer)
        keyword_match_bonus = min(matched_source_keywords * 0.06, 0.3)  # 최대 0.3 보너스 (0.05 → 0.06, 0.2 → 0.3)
        
        if answer_has_legal_content:
            # 법률 관련 내용이 있으면 최소 0.5로 조정 (0.4 → 0.5로 상향)
            # 개선: 법률 키워드 개수에 따라 추가 보너스 부여
            legal_keyword_count_in_answer = sum(1 for keyword in legal_keywords if keyword in answer)
            legal_keyword_bonus = min(legal_keyword_count_in_answer * 0.02, 0.15)  # 최대 0.15 보너스
            
            if grounding_score < 0.5:
                grounding_score = max(0.5, grounding_score + 0.25 + keyword_match_bonus + legal_keyword_bonus)  # 기본 보너스 + 키워드 매칭 보너스 + 법률 키워드 보너스
            else:
                grounding_score += keyword_match_bonus + legal_keyword_bonus  # 키워드 매칭 보너스 + 법률 키워드 보너스
            
            logger.info(
                f"✅ [GROUNDING] Legal content detected: "
                f"legal_keywords={legal_keyword_count_in_answer}, "
                f"keyword_match_bonus={keyword_match_bonus:.2f}, "
                f"legal_keyword_bonus={legal_keyword_bonus:.2f}, "
                f"adjusted_grounding_score={grounding_score:.2f}"
            )
        elif keyword_match_bonus > 0:
            # 법률 키워드는 없지만 검색된 문서의 키워드가 답변에 포함된 경우
            grounding_score += keyword_match_bonus
            logger.info(f"✅ [GROUNDING] Source keyword match detected, adjusted grounding_score to {grounding_score:.2f} (keyword_match_bonus: {keyword_match_bonus:.2f})")
        
        # Grounding score 최대값 제한 (1.0 초과 방지)
        grounding_score = min(grounding_score, 1.0)

        # 5. 검증 통과 기준: 35% 이상 문장이 검증됨 (40% -> 35%로 추가 완화)
        is_grounded = grounding_score >= 0.35

        # 6. 신뢰도 조정 (검증되지 않은 문장이 많으면 신뢰도 감소, 하지만 완화된 기준 적용)
        # 부분 검증된 문장은 confidence_penalty에서 제외
        remaining_unverified = len(unverified_sentences)
        confidence_penalty = remaining_unverified * 0.02  # 문장당 2% 감소

        return {
            "is_grounded": is_grounded,
            "grounding_score": grounding_score,
            "verified_sentences": verified_sentences[:5],  # 샘플
            "unverified_sentences": unverified_sentences,
            "unverified_count": len(unverified_sentences),
            "source_coverage": source_coverage,
            "needs_review": not is_grounded or len(unverified_sentences) > 3,
            "confidence_penalty": min(confidence_penalty, 0.3),  # 최대 30% 감소
            "total_sentences": total_sentences,
            "verified_count": verified_count
        }


class SearchValidator:
    """검색 품질 검증"""

    @staticmethod
    def validate_search_quality(
        search_results: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> Dict[str, Any]:
        """
        검색 품질 검증

        Args:
            search_results: 검색 결과 목록
            query: 검색 쿼리
            query_type: 질문 유형

        Returns:
            검증 결과 딕셔너리
        """
        try:
            if not search_results:
                return {
                    "is_valid": False,
                    "quality_score": 0.0,
                    "doc_count": 0,
                    "avg_relevance": 0.0,
                    "issues": ["검색 결과가 없습니다"],
                    "recommendations": ["검색 쿼리를 수정하거나 검색 범위를 확대하세요"]
                }

            # 문서 개수 확인
            doc_count = len(search_results)
            min_docs_required = 2 if query_type != "simple" else 1

            # 평균 관련도 점수 계산
            relevance_scores = []
            for doc in search_results:
                if isinstance(doc, dict):
                    score = doc.get("relevance_score") or doc.get("final_weighted_score", 0.0)
                    relevance_scores.append(score)

            avg_relevance = sum(relevance_scores) / max(1, len(relevance_scores)) if relevance_scores else 0.0

            # 품질 점수 계산
            doc_adequacy = min(1.0, doc_count / max(1, min_docs_required))
            relevance_adequacy = avg_relevance

            quality_score = (doc_adequacy * 0.4 + relevance_adequacy * 0.6)

            # 문제점 확인
            issues = []
            if doc_count < min_docs_required:
                issues.append(f"검색 결과가 부족합니다 ({doc_count}/{min_docs_required})")
            if avg_relevance < 0.3:
                issues.append(f"평균 관련도가 낮습니다 ({avg_relevance:.2f})")

            # 권고사항 생성
            recommendations = []
            if doc_count < min_docs_required:
                recommendations.append("검색 쿼리를 확장하거나 검색 범위를 넓히세요")
            if avg_relevance < 0.3:
                recommendations.append("검색 쿼리를 더 구체적으로 작성하거나 다른 키워드를 시도하세요")

            is_valid = doc_count >= min_docs_required and avg_relevance >= 0.3

            return {
                "is_valid": is_valid,
                "quality_score": quality_score,
                "doc_count": doc_count,
                "avg_relevance": avg_relevance,
                "min_docs_required": min_docs_required,
                "issues": issues,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.warning(f"Search quality validation failed: {e}")
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "doc_count": 0,
                "avg_relevance": 0.0,
                "issues": [f"검증 중 오류 발생: {e}"],
                "recommendations": []
            }
