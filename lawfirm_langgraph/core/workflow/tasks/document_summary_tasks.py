# -*- coding: utf-8 -*-
"""
Document Summary Tasks
문서 요약 관련 Task 정의 (재사용 가능한 컴포넌트)
프롬프트 길이 제한을 고려한 배치 분할 처리 포함
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

logger = get_logger(__name__)


class SummaryStrategy(Enum):
    """요약 전략"""
    BATCH = "batch"  # 배치 요약 (권장)
    INDIVIDUAL = "individual"  # 개별 요약
    HYBRID = "hybrid"  # 조건부 하이브리드
    RULE_BASED = "rule_based"  # 규칙 기반


class DocumentSummaryTask:
    """문서 요약 Task (배치 처리 최적화 + 프롬프트 길이 제한 고려)"""
    
    def __init__(
        self,
        llm_fast: Optional[Any] = None,
        logger_instance: Optional[logging.Logger] = None,
        strategy: SummaryStrategy = SummaryStrategy.BATCH,
        batch_size: int = 5,
        max_summary_length: int = 500,
        max_prompt_length: int = 8000,  # 프롬프트 최대 길이 (문자 수)
        max_prompt_tokens: Optional[int] = None  # 프롬프트 최대 토큰 수 (선택적)
    ):
        """
        DocumentSummaryTask 초기화
        
        Args:
            llm_fast: 빠른 LLM 인스턴스
            logger_instance: 로거 인스턴스
            strategy: 요약 전략
            batch_size: 배치 크기
            max_summary_length: 최대 요약 길이
            max_prompt_length: 프롬프트 최대 길이 (문자 수)
            max_prompt_tokens: 프롬프트 최대 토큰 수 (None이면 문자 수 기준)
        """
        self.llm_fast = llm_fast
        self.logger = logger_instance or logger
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_summary_length = max_summary_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_tokens = max_prompt_tokens
        
        # 임계값 설정
        self.SUMMARY_THRESHOLD_LAW = 1000
        self.SUMMARY_THRESHOLD_CASE = 600
        self.SUMMARY_THRESHOLD_COMMENTARY = 400
        
        # 프롬프트 템플릿 길이 (고정 부분)
        self._estimate_base_prompt_length(query="", doc_count=0)
    
    def _estimate_base_prompt_length(self, query: str, doc_count: int) -> int:
        """기본 프롬프트 길이 추정"""
        base_template = f"""다음 {doc_count}개의 법률 문서를 각각 요약해주세요.

사용자 질문: {query}

요구사항:
- 각 문서를 {self.max_summary_length}자 이내로 요약
- 핵심 쟁점 3개 이상 포함
- 질문과의 연관성 명시

응답 형식 (각 문서마다 반복):
[문서 1]
요약: [요약 텍스트]
핵심 쟁점:
1. [쟁점 1]
2. [쟁점 2]
3. [쟁점 3]
연관성: [질문과의 연관성]

[문서 2]
...
"""
        return len(base_template)
    
    def execute(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        use_llm: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Task 실행
        
        Args:
            docs: 문서 리스트
            query: 사용자 질문
            use_llm: LLM 사용 여부
        
        Returns:
            (summaries, metadata) 튜플
            - summaries: 요약 결과 리스트
            - metadata: 실행 메타데이터 (성공 여부, 전략, 시간 등)
        """
        import time
        start_time = time.time()
        
        try:
            if self.strategy == SummaryStrategy.BATCH:
                summaries, metadata = self._execute_batch(docs, query, use_llm)
            elif self.strategy == SummaryStrategy.INDIVIDUAL:
                summaries, metadata = self._execute_individual(docs, query, use_llm)
            elif self.strategy == SummaryStrategy.HYBRID:
                summaries, metadata = self._execute_hybrid(docs, query, use_llm)
            else:  # RULE_BASED
                summaries, metadata = self._execute_rule_based(docs, query)
            
            elapsed_time = time.time() - start_time
            metadata['execution_time'] = elapsed_time
            metadata['success'] = True
            
            self.logger.info(
                f"[DocumentSummaryTask] 요약 완료: "
                f"전략={self.strategy.value}, 문서={len(docs)}, "
                f"시간={elapsed_time:.2f}초, LLM 호출={metadata.get('llm_calls', 0)}"
            )
            
            return summaries, metadata
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"[DocumentSummaryTask] 요약 실패: {e}")
            
            # 폴백: 규칙 기반 요약
            summaries, metadata = self._execute_rule_based(docs, query)
            metadata['execution_time'] = elapsed_time
            metadata['success'] = False
            metadata['error'] = str(e)
            metadata['fallback_used'] = True
            
            return summaries, metadata
    
    def _execute_batch(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        use_llm: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """배치 요약 실행 (프롬프트 길이 제한 고려)"""
        if not use_llm or not self.llm_fast:
            return self._execute_rule_based(docs, query)
        
        self.logger.info(
            f"[DocumentSummaryTask] 배치 요약 시작: "
            f"문서 수={len(docs)}, 배치 크기={self.batch_size}"
        )
        
        # 프롬프트 길이를 고려하여 배치 분할
        batches = self._split_docs_into_batches(docs, query)
        
        all_summaries = []
        total_llm_calls = 0
        global_doc_start = 1  # 전역 문서 번호 시작
        
        for batch_idx, batch_docs in enumerate(batches, 1):
            self.logger.info(
                f"[DocumentSummaryTask] 배치 {batch_idx}/{len(batches)} 처리 중: "
                f"문서 수={len(batch_docs)}, 전역 문서 번호 시작={global_doc_start}"
            )
            
            try:
                # 배치 프롬프트 생성
                batch_prompt = self._build_batch_prompt(batch_docs, query, batch_idx, len(batches), global_doc_start)
                
                # 프롬프트 길이 확인
                prompt_length = len(batch_prompt)
                self.logger.debug(
                    f"[DocumentSummaryTask] 배치 {batch_idx} 프롬프트 길이: {prompt_length}자"
                )
                
                # 한 번의 LLM 호출
                response = self.llm_fast.invoke(batch_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                total_llm_calls += 1
                
                # 배치 응답 파싱 (전역 문서 번호 시작 전달)
                batch_summaries = self._parse_batch_response(
                    response_text, 
                    batch_docs, 
                    batch_idx,
                    len(batches),
                    global_doc_start
                )
                all_summaries.extend(batch_summaries)
                
                # 다음 배치의 전역 문서 번호 시작 업데이트
                global_doc_start += len(batch_docs)
                
            except Exception as e:
                self.logger.warning(
                    f"[DocumentSummaryTask] 배치 {batch_idx} 처리 실패: {e}, "
                    f"규칙 기반으로 폴백"
                )
                # 폴백: 규칙 기반 요약
                for doc in batch_docs:
                    summary = self._summarize_with_rules(doc, query)
                    all_summaries.append(summary)
        
        metadata = {
            'strategy': 'batch',
            'llm_calls': total_llm_calls,
            'total_docs': len(docs),
            'batches': len(batches),
            'batch_sizes': [len(batch) for batch in batches]
        }
        
        return all_summaries, metadata
    
    def _split_docs_into_batches(
        self,
        docs: List[Dict[str, Any]],
        query: str
    ) -> List[List[Dict[str, Any]]]:
        """
        문서를 프롬프트 길이 제한을 고려하여 배치로 분할
        
        Args:
            docs: 문서 리스트
            query: 사용자 질문
        
        Returns:
            배치 리스트
        """
        if not docs:
            return []
        
        batches = []
        current_batch = []
        current_batch_length = 0
        
        # 기본 프롬프트 길이 추정 (질문 포함)
        base_length = self._estimate_base_prompt_length(query, 1)
        available_length = self.max_prompt_length - base_length
        
        for doc in docs:
            # 문서 내용 길이 추정 (최대 2000자로 제한)
            doc_content = doc.get("content", "")[:2000]
            doc_title = self._get_document_title(doc)
            doc_type = self._get_document_type(doc)
            
            # 문서 섹션 길이 추정
            doc_section_length = len(f"""
[문서 X]
제목: {doc_title}
유형: {doc_type}
내용:
{doc_content}
""")
            
            # 현재 배치에 추가 가능한지 확인
            if current_batch and (current_batch_length + doc_section_length) > available_length:
                # 현재 배치 저장하고 새 배치 시작
                batches.append(current_batch)
                current_batch = [doc]
                current_batch_length = doc_section_length
                self.logger.debug(
                    f"[DocumentSummaryTask] 배치 분할: "
                    f"이전 배치={len(batches[-1])}개 문서, "
                    f"새 배치 시작"
                )
            else:
                # 현재 배치에 추가
                current_batch.append(doc)
                current_batch_length += doc_section_length
        
        # 마지막 배치 추가
        if current_batch:
            batches.append(current_batch)
        
        self.logger.info(
            f"[DocumentSummaryTask] 배치 분할 완료: "
            f"총 {len(docs)}개 문서 → {len(batches)}개 배치, "
            f"배치 크기={[len(batch) for batch in batches]}"
        )
        
        return batches
    
    def _build_batch_prompt(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        batch_idx: int = 1,
        total_batches: int = 1,
        global_doc_start: int = 1
    ) -> str:
        """배치 요약 프롬프트 생성"""
        doc_sections = []
        for i, doc in enumerate(docs, 1):
            doc_type = self._get_document_type(doc)
            doc_title = self._get_document_title(doc)
            content = doc.get("content", "")[:2000]  # 각 문서 최대 2000자
            
            # 전역 문서 번호 계산
            global_doc_num = global_doc_start + i - 1
            
            doc_sections.append(f"""
[문서 {global_doc_num}]
제목: {doc_title}
유형: {doc_type}
내용:
{content}
""")
        
        batch_info = ""
        if total_batches > 1:
            batch_info = f"\n(참고: 이 배치는 전체 {total_batches}개 배치 중 {batch_idx}번째입니다.)\n"
        
        # 응답 형식 예시 생성
        response_examples = []
        for i in range(len(docs)):
            doc_num = global_doc_start + i
            response_examples.append(f"[문서 {doc_num}]\n요약: [요약 텍스트]\n핵심 쟁점:\n1. [쟁점 1]\n2. [쟁점 2]\n3. [쟁점 3]\n연관성: [질문과의 연관성]")
        
        return f"""다음 {len(docs)}개의 법률 문서를 각각 요약해주세요.{batch_info}

사용자 질문: {query}

{''.join(doc_sections)}

요구사항:
- 각 문서를 {self.max_summary_length}자 이내로 요약
- 핵심 쟁점 3개 이상 포함
- 질문과의 연관성 명시

응답 형식 (각 문서마다 반복):
{chr(10).join(response_examples)}
"""
    
    def _parse_batch_response(
        self,
        response_text: str,
        docs: List[Dict[str, Any]],
        batch_idx: int = 1,
        total_batches: int = 1,
        global_doc_start: int = 1
    ) -> List[Dict[str, Any]]:
        """배치 요약 응답 파싱 (개선: 다양한 응답 형식 지원)"""
        summaries = []
        
        for i, doc in enumerate(docs, 1):
            doc_type = self._get_document_type(doc)
            
            # 전역 문서 번호 계산
            global_doc_num = global_doc_start + i - 1
            
            # 다양한 패턴으로 문서 섹션 추출 시도 (더 견고한 패턴)
            patterns = [
                rf'\[문서\s*{global_doc_num}\](.+?)(?=\[문서\s*\d+\]|$)',
                rf'문서\s*{global_doc_num}[:\s]+(.+?)(?=문서\s*\d+|$)',
                rf'\[{global_doc_num}\](.+?)(?=\[\d+\]|$)',
                rf'문서\s*{i}[:\s]+(.+?)(?=문서\s*\d+|$)',  # 배치 내 인덱스
                rf'\[문서\s*{i}\](.+?)(?=\[문서\s*\d+\]|$)',  # 배치 내 인덱스
            ]
            
            doc_response = None
            for pattern in patterns:
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    doc_response = match.group(1).strip()
                    if len(doc_response) > 20:  # 최소 길이 확인
                        break
            
            # 🔥 개선: 패턴 매칭 실패 시 배치 응답을 문서 수로 나누어 할당
            if not doc_response and len(docs) > 1:
                # 배치 응답을 문서 수로 나누기
                response_lines = response_text.split('\n')
                total_lines = len(response_lines)
                lines_per_doc = max(1, total_lines // len(docs))
                start_line = (i - 1) * lines_per_doc
                end_line = i * lines_per_doc if i < len(docs) else total_lines
                doc_response = '\n'.join(response_lines[start_line:end_line]).strip()
                if doc_response:
                    self.logger.debug(
                        f"[DocumentSummaryTask] 문서 {global_doc_num} 패턴 매칭 실패, "
                        f"라인 분할로 폴백 (라인 {start_line}-{end_line})"
                    )
            
            if doc_response:
                # 요약 추출 (다양한 형식 지원)
                summary_patterns = [
                    r'요약[:\s]*(.+?)(?=핵심|쟁점|연관성|$)',
                    r'Summary[:\s]*(.+?)(?=Key|Points|Relevance|$)',
                    r'(.+?)(?=핵심|쟁점|연관성|$)',
                ]
                
                summary = None
                for pattern in summary_patterns:
                    match = re.search(pattern, doc_response, re.DOTALL | re.IGNORECASE)
                    if match:
                        summary = match.group(1).strip()
                        if len(summary) > 50:  # 최소 길이 확인
                            break
                
                if not summary:
                    summary = doc_response[:self.max_summary_length]
                
                # 핵심 쟁점 추출
                key_points_patterns = [
                    r'핵심\s*쟁점[:\s]*(.+?)(?=연관성|$)',
                    r'Key\s*Points[:\s]*(.+?)(?=Relevance|$)',
                    r'\d+\.\s*(.+?)(?=\d+\.|연관성|Relevance|$)',
                ]
                
                key_points = []
                for pattern in key_points_patterns:
                    matches = re.findall(pattern, doc_response, re.DOTALL | re.IGNORECASE)
                    if matches:
                        if isinstance(matches[0], str):
                            # 단일 문자열인 경우 줄바꿈으로 분리
                            lines = matches[0].split('\n')
                            key_points = [line.strip() for line in lines if line.strip() and re.match(r'^\d+\.', line.strip())]
                            if key_points:
                                # 번호 제거
                                key_points = [re.sub(r'^\d+\.\s*', '', p).strip() for p in key_points[:5]]
                        else:
                            key_points = [m.strip() for m in matches[:5]]
                        if key_points:
                            break
                
                # 연관성 추출
                relevance_patterns = [
                    r'연관성[:\s]*(.+?)$',
                    r'Relevance[:\s]*(.+?)$',
                ]
                
                relevance = '질문과 관련된 내용'
                for pattern in relevance_patterns:
                    match = re.search(pattern, doc_response, re.DOTALL | re.IGNORECASE)
                    if match:
                        relevance = match.group(1).strip()
                        break
                
                summaries.append({
                    'summary': summary[:self.max_summary_length],
                    'key_points': key_points[:5],
                    'relevance_notes': relevance,
                    'document_type': doc_type,
                    'original_length': len(doc.get("content", "")),
                    'summary_length': len(summary)
                })
            else:
                # 파싱 실패 시 폴백
                self.logger.warning(
                    f"[DocumentSummaryTask] 문서 {global_doc_num} (배치 {batch_idx}, 인덱스 {i}) "
                    f"파싱 실패, 규칙 기반으로 폴백. 응답 길이: {len(response_text)}자"
                )
                # 디버깅: 응답 텍스트 일부 로깅
                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(
                        f"[DocumentSummaryTask] 응답 텍스트 샘플 (처음 500자): "
                        f"{response_text[:500]}"
                    )
                summaries.append(self._summarize_with_rules(doc, ""))
        
        return summaries
    
    def _execute_individual(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        use_llm: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """개별 요약 실행 (각 문서마다 LLM 호출)"""
        if not use_llm or not self.llm_fast:
            return self._execute_rule_based(docs, query)
        
        self.logger.info(
            f"[DocumentSummaryTask] 개별 요약 시작: 문서 수={len(docs)}"
        )
        
        summaries = []
        for i, doc in enumerate(docs, 1):
            try:
                summary = self._summarize_single_doc(doc, query, i)
                summaries.append(summary)
            except Exception as e:
                self.logger.warning(f"문서 {i} 요약 실패: {e}")
                # 폴백: 규칙 기반
                summary = self._summarize_with_rules(doc, query)
                summaries.append(summary)
        
        metadata = {
            'strategy': 'individual',
            'llm_calls': len(docs),
            'total_docs': len(docs)
        }
        
        return summaries, metadata
    
    def _execute_hybrid(
        self,
        docs: List[Dict[str, Any]],
        query: str,
        use_llm: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """하이브리드 요약 (조건부 배치/개별)"""
        # 긴 문서는 LLM, 짧은 문서는 규칙 기반
        docs_for_llm = []
        docs_for_rules = []
        doc_indices = []
        
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            doc_type = self._get_document_type(doc)
            threshold = self._get_summary_threshold(doc_type)
            
            if len(content) > threshold and use_llm and self.llm_fast:
                docs_for_llm.append((i, doc))
            else:
                docs_for_rules.append((i, doc))
        
        summaries = [None] * len(docs)
        llm_calls = 0
        
        # LLM 요약 (배치)
        if docs_for_llm:
            llm_docs = [doc for _, doc in docs_for_llm]
            llm_summaries, batch_metadata = self._execute_batch(llm_docs, query, use_llm=True)
            llm_calls = batch_metadata.get('llm_calls', 0)
            
            for (idx, _), summary in zip(docs_for_llm, llm_summaries):
                summaries[idx] = summary
        
        # 규칙 기반 요약
        for idx, doc in docs_for_rules:
            summaries[idx] = self._summarize_with_rules(doc, query)
        
        metadata = {
            'strategy': 'hybrid',
            'llm_calls': llm_calls,
            'rule_based_count': len(docs_for_rules),
            'llm_count': len(docs_for_llm)
        }
        
        return summaries, metadata
    
    def _execute_rule_based(
        self,
        docs: List[Dict[str, Any]],
        query: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """규칙 기반 요약 (LLM 없이)"""
        summaries = [
            self._summarize_with_rules(doc, query)
            for doc in docs
        ]
        
        metadata = {
            'strategy': 'rule_based',
            'llm_calls': 0,
            'total_docs': len(docs)
        }
        
        return summaries, metadata
    
    def _summarize_single_doc(
        self,
        doc: Dict[str, Any],
        query: str,
        doc_index: int
    ) -> Dict[str, Any]:
        """단일 문서 요약 (개별 LLM 호출)"""
        doc_type = self._get_document_type(doc)
        doc_title = self._get_document_title(doc)
        content = doc.get("content", "")[:2000]
        
        prompt = f"""다음 법률 문서를 요약해주세요.

문서 제목: {doc_title}
사용자 질문: {query}

문서 내용:
{content}

요약 요구사항:
- {self.max_summary_length}자 이내로 요약
- 핵심 쟁점 3개 이상 포함
- 질문과의 연관성 명시

응답 형식:
요약: [요약 텍스트]
핵심 쟁점:
1. [쟁점 1]
2. [쟁점 2]
3. [쟁점 3]
연관성: [질문과의 연관성]"""
        
        response = self.llm_fast.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # 파싱
        summary_match = re.search(r'요약:\s*(.+?)(?=핵심|$)', response_text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else response_text[:self.max_summary_length]
        
        key_points_match = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|연관성|$)', response_text, re.DOTALL)
        key_points = [p.strip() for p in key_points_match[:5]]
        
        relevance_match = re.search(r'연관성:\s*(.+?)$', response_text, re.DOTALL)
        relevance = relevance_match.group(1).strip() if relevance_match else '질문과 관련된 내용'
        
        return {
            'summary': summary[:self.max_summary_length],
            'key_points': key_points,
            'relevance_notes': relevance,
            'document_type': doc_type,
            'original_length': len(doc.get("content", "")),
            'summary_length': len(summary)
        }
    
    def _summarize_with_rules(
        self,
        doc: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """규칙 기반 요약 (LLM 없이)"""
        content = doc.get("content", "")
        summary = content[:self.max_summary_length] if len(content) > self.max_summary_length else content
        
        return {
            'summary': summary,
            'key_points': [],
            'relevance_notes': '규칙 기반 요약',
            'document_type': self._get_document_type(doc),
            'original_length': len(content),
            'summary_length': len(summary)
        }
    
    def _get_document_type(self, doc: Dict[str, Any]) -> str:
        """문서 유형 판단"""
        if doc.get("law_name") and doc.get("article_no"):
            return 'law'
        elif doc.get("court") or doc.get("case_name"):
            return 'case'
        elif doc.get("type") == "commentary":
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
        
        if law_name and article_no:
            return f"{law_name} 제{article_no}조"
        elif court and case_name:
            return f"{court} {case_name}"
        elif case_name:
            return case_name
        elif title:
            return title
        else:
            return "문서"
    
    def _get_summary_threshold(self, doc_type: str) -> int:
        """문서 유형별 요약 임계값"""
        thresholds = {
            'law': self.SUMMARY_THRESHOLD_LAW,
            'case': self.SUMMARY_THRESHOLD_CASE,
            'commentary': self.SUMMARY_THRESHOLD_COMMENTARY,
            'general': 500
        }
        return thresholds.get(doc_type, 500)

