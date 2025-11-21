# -*- coding: utf-8 -*-
"""
문서 요약 생성 에이전트
Summary-First 프롬프트 접근법을 위한 문서 요약 생성
"""

import re
import logging
from typing import Any, Dict, List, Optional

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentSummaryAgent:
    """문서 요약 생성 에이전트"""
    
    def __init__(
        self,
        llm: Optional[Any] = None,  # LLM 인스턴스 (선택적)
        llm_fast: Optional[Any] = None,  # 빠른 LLM (선택적)
        logger: Optional[logging.Logger] = None
    ):
        """
        요약 에이전트 초기화
        
        Args:
            llm: LLM 인스턴스 (선택적)
            llm_fast: 빠른 LLM 인스턴스 (선택적)
            logger: 로거 (없으면 자동 생성)
        """
        self.llm = llm
        self.llm_fast = llm_fast or llm
        self.logger = logger or get_logger(__name__)
        
        # 요약 임계값
        self.SUMMARY_THRESHOLD_LAW = 1000
        self.SUMMARY_THRESHOLD_CASE = 600
        self.SUMMARY_THRESHOLD_COMMENTARY = 400
        self.MAX_SUMMARY_LENGTH = 200
    
    def summarize_document(
        self,
        doc: Dict[str, Any],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False  # LLM 사용 여부
    ) -> Dict[str, Any]:
        """
        문서 요약 생성 (Summary-First 접근법)
        
        Args:
            doc: 문서 딕셔너리
            query: 사용자 질문
            max_summary_length: 최대 요약 길이
            use_llm: LLM 사용 여부 (False면 규칙 기반)
        
        Returns:
            {
                'summary': '요약 텍스트',
                'key_points': ['핵심 포인트 1', '핵심 포인트 2', ...],
                'relevance_notes': '질문과의 연관성',
                'document_type': 'law/case/commentary',
                'original_length': 원본 문서 길이,
                'summary_length': 요약 길이
            }
        """
        try:
            content = doc.get("content", "").strip()
            original_length = len(content)
            
            if not content:
                return {
                    'summary': '',
                    'key_points': [],
                    'relevance_notes': '',
                    'document_type': 'unknown',
                    'original_length': 0,
                    'summary_length': 0
                }
            
            doc_type = self._get_document_type(doc)
            
            if use_llm and self.llm_fast:
                self.logger.info(f"[DocumentSummaryAgent] LLM 기반 요약 사용 (use_llm=True, llm_fast={self.llm_fast is not None})")
                result = self._summarize_with_llm(doc, query, doc_type, max_summary_length)
            else:
                if not use_llm:
                    self.logger.debug(f"[DocumentSummaryAgent] 규칙 기반 요약 사용 (use_llm=False)")
                elif not self.llm_fast:
                    self.logger.warning(f"[DocumentSummaryAgent] LLM이 없어 규칙 기반 요약으로 폴백 (llm_fast=None)")
                result = self._summarize_with_rules(doc, query, doc_type, max_summary_length)
            
            result['original_length'] = original_length
            result['summary_length'] = len(result.get('summary', ''))
            
            return result
            
        except Exception as e:
            self.logger.warning(f"문서 요약 생성 실패: {e}")
            # 폴백: 기본 요약
            return {
                'summary': doc.get("content", "")[:max_summary_length],
                'key_points': [],
                'relevance_notes': '',
                'document_type': self._get_document_type(doc),
                'original_length': len(doc.get("content", "")),
                'summary_length': min(len(doc.get("content", "")), max_summary_length)
            }
    
    def summarize_batch(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        max_summary_length: int = 200,
        use_llm: bool = False
    ) -> List[Dict[str, Any]]:
        """
        배치 요약 생성
        
        Args:
            docs: 문서 리스트
            query: 사용자 질문
            max_summary_length: 최대 요약 길이
            use_llm: LLM 사용 여부
        
        Returns:
            요약 결과 리스트
        """
        return [
            self.summarize_document(doc, query, max_summary_length, use_llm)
            for doc in docs
        ]
    
    def _summarize_with_rules(
        self, doc: Dict[str, Any], query: str, doc_type: str, max_length: int
    ) -> Dict[str, Any]:
        """규칙 기반 요약 생성"""
        if doc_type == 'law':
            return self._summarize_law(doc, query, max_length)
        elif doc_type == 'case':
            return self._summarize_case(doc, query, max_length)
        elif doc_type == 'commentary':
            return self._summarize_commentary(doc, query, max_length)
        else:
            return self._summarize_general(doc, query, max_length)
    
    def _summarize_with_llm(
        self, doc: Dict[str, Any], query: str, doc_type: str, max_length: int
    ) -> Dict[str, Any]:
        """LLM 기반 요약 생성 (선택적)"""
        try:
            self.logger.info(f"[DocumentSummaryAgent] LLM 기반 요약 시작: 문서 유형={doc_type}, 제목={self._get_document_title(doc)}")
            content = doc.get("content", "")
            doc_title = self._get_document_title(doc)
            
            prompt = f"""다음 법률 문서를 요약해주세요.

문서 제목: {doc_title}
사용자 질문: {query}

문서 내용:
{content[:2000]}

요약 요구사항:
- {max_length}자 이내로 요약
- 핵심 쟁점 3개 이상 포함
- 질문과의 연관성 명시

응답 형식:
요약: [요약 텍스트]
핵심 쟁점:
1. [쟁점 1]
2. [쟁점 2]
3. [쟁점 3]
연관성: [질문과의 연관성]"""
            
            self.logger.info(f"[DocumentSummaryAgent] LLM 호출 중 (llm_fast 사용)...")
            response = self.llm_fast.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            self.logger.info(f"[DocumentSummaryAgent] LLM 응답 수신 완료 (응답 길이: {len(response_text)}자)")
            
            # 응답 파싱
            summary_match = re.search(r'요약:\s*(.+?)(?=핵심|$)', response_text, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else response_text[:max_length]
            
            key_points_match = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|연관성|$)', response_text, re.DOTALL)
            key_points = [p.strip() for p in key_points_match[:5]]
            
            relevance_match = re.search(r'연관성:\s*(.+?)$', response_text, re.DOTALL)
            relevance = relevance_match.group(1).strip() if relevance_match else '질문과 관련된 내용'
            
            return {
                'summary': summary[:max_length],
                'key_points': key_points,
                'relevance_notes': relevance,
                'document_type': doc_type
            }
            
        except Exception as e:
            self.logger.warning(f"LLM 요약 실패, 규칙 기반으로 폴백: {e}")
            return self._summarize_with_rules(doc, query, doc_type, max_length)
    
    def _summarize_law(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]:
        """법령 문서 요약"""
        law_name = doc.get("law_name", "")
        article_no = doc.get("article_no", "")
        clause_no = doc.get("clause_no", "")
        content = doc.get("content", "")
        
        summary_parts = []
        key_points = []
        
        # 조문 정보
        if law_name and article_no:
            doc_title = f"{law_name} 제{article_no}조"
            if clause_no:
                doc_title += f" 제{clause_no}항"
            summary_parts.append(doc_title)
        
        # 핵심 조항 추출 (질문 키워드 포함 문장 우선)
        key_sentences = self._extract_key_sentences(content, query, max_sentences=3)
        if key_sentences:
            summary_parts.extend(key_sentences[:2])  # 상위 2개만 요약에 포함
            key_points = key_sentences
        
        # 질문 관련성 분석
        relevance = self._analyze_relevance(content, query)
        
        summary = ' '.join(summary_parts)[:max_length]
        
        return {
            'summary': summary,
            'key_points': key_points,
            'relevance_notes': relevance,
            'document_type': 'law'
        }
    
    def _summarize_case(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]:
        """판례 문서 요약"""
        court = doc.get("court", "")
        case_name = doc.get("case_name", "")
        case_number = doc.get("case_number", "")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        
        summary_parts = []
        key_points = []
        
        # 판례 정보
        if court and case_name:
            case_info = f"{court} {case_name}"
            if case_number:
                case_info += f" ({case_number})"
            summary_parts.append(case_info)
        
        # 판시사항 추출
        reasoning = metadata.get("case_reasoning") or self._extract_reasoning(content)
        if reasoning:
            summary_parts.append(reasoning[:150])
        
        # 판결요지 추출
        judgment_points = self._extract_judgment_points(content, query)
        if judgment_points:
            key_points = judgment_points
            summary_parts.append(judgment_points[0][:100] if judgment_points else "")
        
        # 질문 관련성 분석
        relevance = self._analyze_relevance(content, query)
        
        summary = ' '.join(summary_parts)[:max_length]
        
        return {
            'summary': summary,
            'key_points': key_points,
            'relevance_notes': relevance,
            'document_type': 'case'
        }
    
    def _summarize_commentary(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]:
        """해설 문서 요약"""
        content = doc.get("content", "")
        title = doc.get("title", "") or doc.get("source", "")
        
        summary_parts = []
        key_points = []
        
        # 제목
        if title:
            summary_parts.append(title)
        
        # 핵심 내용 추출 (앞부분 + 키워드 관련 부분)
        intro = content[:200] if len(content) > 200 else content
        summary_parts.append(intro)
        
        # 질문 관련 부분 추출
        relevant_parts = self._extract_relevant_parts(content, query, max_length=300)
        if relevant_parts:
            key_points = relevant_parts
            summary_parts.append(relevant_parts[0][:100] if relevant_parts else "")
        
        # 질문 관련성 분석
        relevance = self._analyze_relevance(content, query)
        
        summary = ' '.join(summary_parts)[:max_length]
        
        return {
            'summary': summary,
            'key_points': key_points,
            'relevance_notes': relevance,
            'document_type': 'commentary'
        }
    
    def _summarize_general(
        self, doc: Dict[str, Any], query: str, max_length: int
    ) -> Dict[str, Any]:
        """일반 문서 요약"""
        content = doc.get("content", "")
        title = doc.get("title", "") or doc.get("source", "문서")
        
        # 앞부분 추출
        summary = content[:max_length] if len(content) > max_length else content
        
        # 키워드 관련 문장 추출
        key_sentences = self._extract_key_sentences(content, query, max_sentences=3)
        
        return {
            'summary': summary,
            'key_points': key_sentences,
            'relevance_notes': self._analyze_relevance(content, query),
            'document_type': 'general'
        }
    
    def _get_document_type(self, doc: Dict[str, Any]) -> str:
        """문서 유형 판단"""
        if doc.get("law_name") and doc.get("article_no"):
            return 'law'
        elif doc.get("court") or doc.get("case_name") or doc.get("case_number"):
            return 'case'
        elif doc.get("type") == "commentary" or "해설" in str(doc.get("title", "")):
            return 'commentary'
        else:
            return 'general'
    
    def _get_document_title(self, doc: Dict[str, Any]) -> str:
        """문서 제목 추출"""
        law_name = doc.get("law_name", "")
        article_no = doc.get("article_no", "")
        case_name = doc.get("case_name", "")
        court = doc.get("court", "")
        title = doc.get("title", "")
        source = doc.get("source", "")
        
        if law_name and article_no:
            return f"{law_name} 제{article_no}조"
        elif court and case_name:
            return f"{court} {case_name}"
        elif case_name:
            return case_name
        elif title:
            return title
        elif source:
            return source
        else:
            return "문서"
    
    def _extract_key_sentences(
        self, content: str, query: str, max_sentences: int = 3
    ) -> List[str]:
        """질문과 관련된 핵심 문장 추출"""
        if not content or not query:
            return []
        
        # 질문 키워드 추출
        query_keywords = set(query.split())
        
        # 문장 분리
        sentences = re.split(r'[。\.\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 키워드 매칭 점수 계산
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            for keyword in query_keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # 법률 용어 가중치
            legal_terms = ['법', '조', '항', '호', '판결', '판례', '손해배상', '계약', '소송']
            for term in legal_terms:
                if term in sentence:
                    score += 0.5
            
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # 점수 순 정렬
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 문장 반환
        return [s[1] for s in scored_sentences[:max_sentences]]
    
    def _extract_reasoning(self, content: str) -> str:
        """판시사항 추출"""
        # 판시사항 패턴 찾기
        patterns = [
            r'판시사항[:\s]*(.+?)(?=\n|판결요지|$)',
            r'판시[:\s]*(.+?)(?=\n|판결요지|$)',
            r'요지[:\s]*(.+?)(?=\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 50:  # 최소 길이 확인
                    return reasoning[:300]
        
        # 패턴이 없으면 앞부분 반환
        return content[:200] if len(content) > 200 else content
    
    def _extract_judgment_points(
        self, content: str, query: str, max_points: int = 3
    ) -> List[str]:
        """판결요지 추출"""
        # 판결요지 패턴 찾기
        patterns = [
            r'판결요지[:\s]*(.+?)(?=\n|판시사항|$)',
            r'요지[:\s]*(.+?)(?=\n|판시사항|$)',
        ]
        
        judgment_text = ""
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                judgment_text = match.group(1).strip()
                break
        
        if not judgment_text:
            # 패턴이 없으면 핵심 문장 추출
            return self._extract_key_sentences(content, query, max_sentences=max_points)
        
        # 문장 분리
        sentences = re.split(r'[。\.\n]+', judgment_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        return sentences[:max_points]
    
    def _extract_relevant_parts(
        self, content: str, query: str, max_length: int = 300
    ) -> List[str]:
        """질문과 관련된 부분 추출"""
        key_sentences = self._extract_key_sentences(content, query, max_sentences=5)
        
        # 길이 제한
        result = []
        current_length = 0
        for sentence in key_sentences:
            if current_length + len(sentence) <= max_length:
                result.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        return result
    
    def _analyze_relevance(self, content: str, query: str) -> str:
        """질문과의 연관성 분석"""
        if not content or not query:
            return '관련성 분석 불가'
        
        query_keywords = set(query.split())
        content_lower = content.lower()
        
        # 키워드 매칭 개수
        matched_keywords = [kw for kw in query_keywords if kw.lower() in content_lower]
        match_count = len(matched_keywords)
        total_keywords = len(query_keywords)
        
        if total_keywords == 0:
            return '관련성 분석 불가'
        
        match_ratio = match_count / total_keywords
        
        if match_ratio >= 0.7:
            return f'질문과 직접 관련 ({match_count}/{total_keywords} 키워드 일치)'
        elif match_ratio >= 0.4:
            return f'질문과 부분 관련 ({match_count}/{total_keywords} 키워드 일치)'
        else:
            return f'질문과 간접 관련 ({match_count}/{total_keywords} 키워드 일치)'

