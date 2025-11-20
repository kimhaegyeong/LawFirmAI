# -*- coding: utf-8 -*-
"""
워크플로우 문서 처리 프로세서
검색 결과 문서 선택, 컨텍스트 빌딩, 프롬프트 최적화 등을 담당
"""

import logging
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WorkflowDocumentProcessor:
    """워크플로우 문서 처리 프로세서"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, query_enhancer=None, semantic_search_engine=None):
        self.logger = logger or logging.getLogger(__name__)
        self.query_enhancer = query_enhancer
        self.semantic_search_engine = semantic_search_engine
    
    def _extract_doc_content(self, doc: Dict[str, Any]) -> str:
        """문서 내용 추출 (강화된 버전)"""
        
        # 1. 기본 필드 확인
        content = doc.get("content") or doc.get("text") or doc.get("content_text")
        
        # 2. metadata에서 확인
        if not content:
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                content = metadata.get("content") or metadata.get("text")
        
        # 3. content가 문자열이 아니면 변환 시도
        if content and not isinstance(content, str):
            try:
                content = str(content)
            except Exception:
                content = ""
        
        # 4. 내용이 비어있으면 DB에서 복원 시도
        if not content or len(content.strip()) < 10:
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id")
            chunk_id = doc.get("chunk_id")
            
            if doc_id or chunk_id:
                try:
                    if self.semantic_search_engine and hasattr(self.semantic_search_engine, '_ensure_text_content'):
                        restored_content = self.semantic_search_engine._ensure_text_content(doc)
                        if restored_content and len(restored_content.strip()) >= 10:
                            content = restored_content
                            doc["content"] = content
                            self.logger.debug(f"✅ [CONTENT RESTORE] 문서 내용 복원 성공: doc_id={doc_id}")
                except Exception as e:
                    self.logger.debug(f"문서 내용 복원 실패: {e}")
        
        # 5. 최종 검증
        if not content or len(content.strip()) < 10:
            self.logger.warning(
                f"⚠️ [CONTENT EXTRACT] 문서 내용 부족: "
                f"doc_id={doc.get('id', 'unknown')}, "
                f"content_len={len(content) if content else 0}, "
                f"keys={list(doc.keys())[:10]}"
            )
        
        return content or ""
    
    def _deduplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        중복 문서 제거 (같은 content 또는 같은 source_id를 가진 문서)
        """
        seen_content = set()
        seen_source_ids = set()
        deduplicated = []
        
        for doc in documents:
            content = self._extract_doc_content(doc)
            source_id = doc.get("source_id") or doc.get("id") or doc.get("chunk_id")
            
            # content 해시로 중복 확인 (처음 500자만 해시)
            content_hash = hash(content[:500]) if content else None
            if content_hash and content_hash in seen_content:
                self.logger.debug(f"중복 문서 제거 (content): source_id={source_id}")
                continue
            
            # source_id로 중복 확인
            if source_id and source_id in seen_source_ids:
                self.logger.debug(f"중복 문서 제거 (source_id): source_id={source_id}")
                continue
            
            if content_hash:
                seen_content.add(content_hash)
            if source_id:
                seen_source_ids.add(source_id)
            
            deduplicated.append(doc)
        
        if len(documents) != len(deduplicated):
            self.logger.info(
                f"중복 문서 제거: {len(documents)}개 → {len(deduplicated)}개 "
                f"({len(documents) - len(deduplicated)}개 제거됨)"
            )
        
        return deduplicated
    
    def build_prompt_optimized_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str,
        legal_field: str,
        select_balanced_documents_func=None,
        extract_query_relevant_sentences_func=None,
        generate_document_based_instructions_func=None
    ) -> Dict[str, Any]:
        """프롬프트에 최대한 반영되도록 최적화된 컨텍스트 구축"""
        try:
            if not retrieved_docs:
                self.logger.warning("build_prompt_optimized_context: retrieved_docs is empty")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            valid_docs = []
            invalid_docs_count = 0
            
            # 질의와 검색된 문서의 relevance_score 로깅 (모든 문서)
            self.logger.info(f"📊 [RELEVANCE SCORES] 질의: '{query}'")
            self.logger.info(f"📊 [RELEVANCE SCORES] 검색된 문서 수: {len(retrieved_docs)}개")
            
            # 개선: 동적 임계값 조정 (검색 결과 점수 분포 분석) - 개선 버전
            scores = [doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0) 
                     for doc in retrieved_docs if isinstance(doc, dict)]
            
            # 모든 문서의 점수 상세 로깅
            doc_scores = []
            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    continue
                score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                similarity = doc.get("similarity", 0.0)
                keyword_score = doc.get("keyword_match_score", 0.0)
                doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id") or "unknown"
                doc_type = doc.get("type") or doc.get("source_type", "unknown")
                source = doc.get("source", "")[:100] or "unknown"
                content_preview = (doc.get("content", "")[:100] or "").replace("\n", " ")
                doc_scores.append((score, similarity, keyword_score, doc_id, doc_type, source, content_preview, doc))
            
            # 점수 분포 통계
            if doc_scores:
                scores_only = [s[0] for s in doc_scores]
                avg_score = sum(scores_only) / len(scores_only)
                max_score = max(scores_only)
                min_score = min(scores_only)
                median_score = sorted(scores_only)[len(scores_only) // 2]
                self.logger.info(
                    f"📊 [SCORE STATS] 평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}, 중앙값={median_score:.3f}"
                )
                
                # 모든 문서의 점수 상세 로깅 (정렬된 순서)
                doc_scores_sorted = sorted(doc_scores, key=lambda x: x[0], reverse=True)
                self.logger.info(f"📊 [ALL DOCS SCORES] 모든 {len(doc_scores_sorted)}개 문서의 relevance_score:")
                for i, (score, similarity, keyword_score, doc_id, doc_type, source, content_preview, doc) in enumerate(doc_scores_sorted, 1):
                    self.logger.info(
                        f"   {i}. final_score={score:.3f}, similarity={similarity:.3f}, keyword={keyword_score:.3f}, "
                        f"type={doc_type}, id={doc_id[:50]}, source={source}, "
                        f"content_preview={content_preview}"
                    )
            
            # avg_score를 외부에서도 사용할 수 있도록 미리 정의
            avg_score = 0.0
            if scores:
                import statistics
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                score_range = max_score - min_score
                
                # 표준편차 계산 (더 정교한 분포 분석)
                try:
                    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
                except Exception:
                    std_dev = 0.0
                
                # 분위수 계산 (25%, 50%, 75%)
                sorted_scores = sorted(scores)
                q25_idx = int(len(sorted_scores) * 0.25)
                q50_idx = int(len(sorted_scores) * 0.50)
                q75_idx = int(len(sorted_scores) * 0.75)
                q25 = sorted_scores[q25_idx] if q25_idx < len(sorted_scores) else min_score
                q50 = sorted_scores[q50_idx] if q50_idx < len(sorted_scores) else avg_score
                q75 = sorted_scores[q75_idx] if q75_idx < len(sorted_scores) else max_score
                
                # 검색 결과 수에 따른 임계값 조정
                num_results = len(retrieved_docs)
                if num_results < 5:
                    # 검색 결과가 매우 적으면 임계값을 크게 완화
                    threshold_adjustment = -0.15
                elif num_results < 10:
                    # 검색 결과가 적으면 임계값을 완화
                    threshold_adjustment = -0.10
                elif num_results < 20:
                    # 검색 결과가 보통이면 약간 완화
                    threshold_adjustment = -0.05
                else:
                    # 검색 결과가 충분하면 조정 없음
                    threshold_adjustment = 0.0
                
                # 점수 분포에 따라 동적 임계값 계산 (개선된 로직)
                # 실제 점수 범위를 고려하여 threshold를 더 낮게 설정
                # avg_score가 낮으면(0.2 미만) 임계값을 더 낮춤
                if avg_score < 0.20:
                    # 평균 점수가 매우 낮으면 최소값 기준으로 매우 낮게 설정
                    # 최소값의 95% 이상을 포함하도록 (거의 모든 문서 포함)
                    dynamic_threshold = max(0.10, min_score * 0.95 + threshold_adjustment)
                    self.logger.info(f"📊 [LOW SCORE] Average score is very low ({avg_score:.3f}), using minimum-based threshold: {dynamic_threshold:.3f}")
                elif score_range < 0.15:
                    # 점수가 매우 비슷하면 최소값 기준으로 낮춤 (최소값의 90% 이상)
                    dynamic_threshold = max(0.12, min_score * 0.90 + threshold_adjustment)
                elif score_range < 0.25:
                    # 점수가 비슷하면 25% 분위수 기준 (더 낮게)
                    dynamic_threshold = max(0.15, q25 - 0.05 + threshold_adjustment)
                elif score_range < 0.4:
                    # 점수 차이가 중간이면 평균 기준 (표준편차 고려, 더 낮게)
                    if std_dev > 0.1:
                        # 분산이 크면 평균 - 표준편차 * 1.5 (더 완화)
                        dynamic_threshold = max(0.15, avg_score - std_dev * 1.5 + threshold_adjustment)
                    else:
                        # 분산이 작으면 평균 - 0.10 (더 완화)
                        dynamic_threshold = max(0.15, avg_score - 0.10 + threshold_adjustment)
                else:
                    # 점수 차이가 크면 중위수 기준 (이상치 영향 최소화, 더 낮게)
                    dynamic_threshold = max(0.20, q50 - 0.05 + threshold_adjustment)
                
                threshold_msg = (
                    f"📊 [DYNAMIC THRESHOLD] avg={avg_score:.3f}, "
                    f"std={std_dev:.3f}, range={score_range:.3f}, "
                    f"q25={q25:.3f}, q50={q50:.3f}, q75={q75:.3f}, "
                    f"num_results={num_results}, threshold={dynamic_threshold:.3f}"
                )
                print(threshold_msg, flush=True, file=sys.stdout)
                self.logger.info(threshold_msg)
            else:
                dynamic_threshold = 0.35
            
            # 개선 1, 4: 문서 타입별 필터링 기준 차등화 (동적 임계값 적용 - 검색 품질 개선)
            # 실제 점수 범위를 고려하여 더 완화된 기준 적용
            # avg_score가 낮으면(0.2 미만) 모든 타입의 기준을 더 낮춤
            if avg_score < 0.20:
                # 평균 점수가 낮으면 모든 타입의 기준을 매우 낮게 설정
                min_relevance_score_semantic = max(0.10, dynamic_threshold - 0.05)
                min_relevance_score_keyword = max(0.10, dynamic_threshold - 0.05)
                min_relevance_score_statute_article = max(0.08, dynamic_threshold - 0.12)
                min_relevance_score_precedent = max(0.10, dynamic_threshold - 0.05)
                min_relevance_score_general = max(0.12, dynamic_threshold - 0.08)
                self.logger.info(f"📊 [LOW SCORE FILTER] Using relaxed thresholds due to low average score ({avg_score:.3f})")
            else:
                # 평균 점수가 정상이면 기존 로직 사용
                min_relevance_score_semantic = max(0.15, dynamic_threshold - 0.05)
                min_relevance_score_keyword = max(0.15, dynamic_threshold - 0.05)
                min_relevance_score_statute_article = max(0.10, dynamic_threshold - 0.10)
                min_relevance_score_precedent = max(0.15, dynamic_threshold - 0.05)
                min_relevance_score_general = max(0.20, dynamic_threshold)
            
            # 개선 7: 질문 핵심 키워드 추출 (간단한 버전)
            query_lower = query.lower()
            query_keywords = []
            for keyword in extracted_keywords:
                if keyword and len(keyword) > 1:
                    query_keywords.append(keyword.lower())
            
            # 개선 2.2: 문서 내용 추출 및 검증 (강화된 버전)
            valid_docs_for_prompt = []
            for doc in retrieved_docs:
                if not isinstance(doc, dict):
                    invalid_docs_count += 1
                    continue
                
                # _extract_doc_content 사용 (강화된 내용 추출)
                content = self._extract_doc_content(doc)
                
                # 최소 길이 검증
                if content and len(content.strip()) >= 10:
                    valid_docs_for_prompt.append({
                        **doc,
                        "content": content  # 확실히 content 필드 설정
                    })
                else:
                    self.logger.warning(
                        f"⚠️ [PROMPT BUILD] 문서 제외 (내용 부족): "
                        f"doc_id={doc.get('id', 'unknown')}, "
                        f"content_len={len(content) if content else 0}"
                    )
                    invalid_docs_count += 1
            
            # 유효한 문서가 없으면 경고 및 폴백
            if not valid_docs_for_prompt:
                self.logger.error(
                    f"❌ [PROMPT BUILD] 유효한 문서 없음: "
                    f"retrieved_docs={len(retrieved_docs)}, "
                    f"valid_docs=0"
                )
                # 폴백: 원본 문서에서 최소한의 내용이라도 추출
                for doc in retrieved_docs[:5]:  # 최대 5개만 시도
                    if not isinstance(doc, dict):
                        continue
                    content = str(doc.get("content", "")) + str(doc.get("text", ""))
                    if len(content.strip()) >= 5:  # 최소 길이 완화
                        valid_docs_for_prompt.append({**doc, "content": content})
            
            if not valid_docs_for_prompt:
                self.logger.error(
                    f"❌ [PROMPT BUILD] 폴백 후에도 유효한 문서 없음"
                )
                return {
                    "prompt_optimized_text": f"질문: {query}\n\n참고할 문서가 없습니다.",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            # valid_docs_for_prompt를 사용하여 필터링 및 점수 검증 진행
            retrieved_docs = valid_docs_for_prompt  # 검증된 문서 사용
            
            for doc in retrieved_docs:
                content = doc.get("content", "")
                if not content or len(content.strip()) < 10:
                    invalid_docs_count += 1
                    continue
                
                search_type = doc.get("search_type", "semantic")
                relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                keyword_match_score = doc.get("keyword_match_score", 0.0)
                matched_keywords = doc.get("matched_keywords", [])
                has_keyword_match = keyword_match_score > 0.0 or len(matched_keywords) > 0
                
                # 문서 타입 및 소스 타입 정의 (doc_type 오류 수정)
                doc_type = doc.get("type") or doc.get("source_type", "unknown")
                source_type = doc.get("source_type") or doc.get("type", "unknown")
                is_legal_doc = (
                    "법" in content[:200] or
                    "조문" in content[:200] or
                    "판례" in content[:200] or
                    "대법원" in content[:200] or
                    doc_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"] or
                    source_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]
                )
                
                # 개선 6: 키워드 매칭 점수 기반 필터링 (더욱 완화)
                if keyword_match_score == 0.0 and not matched_keywords:
                    content_lower = content.lower()
                    has_query_keyword = False
                    for qkw in query_keywords:
                        if qkw in content_lower:
                            has_query_keyword = True
                            break
                    
                    # 관련성 임계값 더욱 완화 (법령/판례 문서는 0.30, 기타는 0.40)
                    relevance_threshold = 0.30 if is_legal_doc else 0.40
                    if not has_query_keyword and relevance_score < relevance_threshold:
                        invalid_docs_count += 1
                        self.logger.debug(
                            f"Document filtered: no keyword match and low relevance "
                            f"(relevance: {relevance_score:.3f}, threshold: {relevance_threshold}, source: {doc.get('source', 'Unknown')})"
                        )
                        continue
                
                # 문서 타입 확인
                is_statute_article = (
                    doc_type == "statute_article" or 
                    source_type == "statute_article" or
                    "statute_article" in doc_type or
                    "statute_article" in source_type or
                    doc.get("direct_match", False) or
                    search_type == "direct_statute"
                )
                is_precedent = (
                    doc_type == "precedent" or
                    source_type == "precedent" or
                    "precedent" in doc_type or
                    "precedent" in source_type or
                    "case_paragraph" in doc_type or
                    "case_paragraph" in source_type or
                    "판례" in content[:200] or
                    "대법원" in content[:200]
                )
                # is_legal_doc는 이미 위에서 정의됨
                
                # 개선: 법률 조문 필터링 예외 (우선순위 2) - 법률 조문은 관련도와 무관하게 포함
                if is_statute_article:
                    # 법률 조문은 항상 포함 (관련도 점수 무시)
                    print(f"[STATUTE EXCEPTION] 법률 조문 포함 (관련도 무시): source={doc.get('source', 'Unknown')}, relevance={relevance_score:.3f}", flush=True, file=sys.stdout)
                    self.logger.debug(
                        f"✅ [STATUTE EXCEPTION] 법률 조문 포함 (관련도 무시): "
                        f"source={doc.get('source', 'Unknown')}, relevance={relevance_score:.3f}"
                    )
                    valid_docs.append(doc)
                    continue
                
                # 개선 4: 문서 타입별 필터링 기준 차등화 (키워드 매칭이 있으면 완화)
                if is_precedent:
                    min_score = min_relevance_score_precedent
                elif search_type == "keyword" and has_keyword_match:
                    min_score = min_relevance_score_keyword
                elif search_type == "semantic":
                    min_score = min_relevance_score_semantic
                else:
                    min_score = min_relevance_score_general
                
                # 키워드 매칭이 있으면 기준을 더 완화 (검색 품질 개선 - avg_score가 낮으면 더 완화)
                if has_keyword_match or has_query_keyword:
                    if avg_score < 0.25:
                        # 평균 점수가 낮으면 매우 완화된 기준 사용 (0.20 → 0.25로 확장)
                        min_score = max(0.08, min_score - 0.15)
                    else:
                        min_score = max(0.15, min_score - 0.10)
                
                # 첫 번째 필터링(키워드 매칭 없을 때)을 통과한 경우, 두 번째 필터링은 더 완화
                if not has_keyword_match and not has_query_keyword:
                    # relevance_score >= 0.30 조건 제거 (avg_score가 낮으면 모든 문서에 적용)
                    if avg_score < 0.25:
                        # 평균 점수가 낮으면 매우 완화된 기준 사용 (0.20 → 0.25로 확장)
                        min_score = max(0.10, min_score - 0.20)
                    elif relevance_score >= 0.30:
                        # 평균 점수가 정상이고 relevance_score가 높으면 기존 로직
                        min_score = max(0.20, min_score - 0.15)
                
                if relevance_score < min_score:
                    invalid_docs_count += 1
                    self.logger.debug(
                        f"Document filtered: relevance score too low ({relevance_score:.3f} < {min_score:.3f}) "
                        f"(source: {doc.get('source', 'Unknown')}, type: {search_type}, doc_type: {doc_type}, "
                        f"has_keyword: {has_keyword_match or has_query_keyword})"
                    )
                    continue
                
                valid_docs.append(doc)
            
            if invalid_docs_count > 0:
                self.logger.warning(
                    f"build_prompt_optimized_context: Filtered {invalid_docs_count} invalid documents "
                    f"(no content, content too short, or relevance < threshold). Valid docs: {len(valid_docs)}"
                )
            
            # 검색 결과가 적을 때 필터링 기준 완화하여 최소 문서 수 보장 (검색 품질 개선)
            if not valid_docs and retrieved_docs:
                self.logger.warning(
                    f"build_prompt_optimized_context: No valid documents after filtering. "
                    f"Relaxing criteria to ensure minimum documents (total retrieved: {len(retrieved_docs)})"
                )
                
                # relevance_score 분포 분석
                relevance_scores = []
                for doc in retrieved_docs:
                    if isinstance(doc, dict):
                        score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                        relevance_scores.append(score)
                
                if relevance_scores:
                    min_rel_score = min(relevance_scores)
                    max_rel_score = max(relevance_scores)
                    avg_rel_score = sum(relevance_scores) / len(relevance_scores)
                    
                    # 분포에 따라 동적으로 relaxed_min_score 설정
                    if avg_rel_score < 0.20:
                        # 평균이 매우 낮으면 최소값 기준으로 설정
                        relaxed_min_score = max(0.05, min_rel_score * 0.90)
                    elif avg_rel_score < 0.30:
                        # 평균이 낮으면 평균의 80% 기준
                        relaxed_min_score = max(0.08, avg_rel_score * 0.80)
                    else:
                        # 평균이 정상이면 기존 로직
                        relaxed_min_score = 0.10
                    
                    self.logger.info(
                        f"📊 [RELAXED FILTER] Score distribution - min={min_rel_score:.3f}, "
                        f"max={max_rel_score:.3f}, avg={avg_rel_score:.3f}, "
                        f"relaxed_threshold={relaxed_min_score:.3f}"
                    )
                else:
                    relaxed_min_score = 0.10
                
                # 필터링 기준을 매우 완화하여 재시도
                for doc in retrieved_docs:
                    if not isinstance(doc, dict):
                        continue
                    
                    content = doc.get("content") or doc.get("text") or doc.get("content_text", "")
                    if not content or len(content.strip()) < 5:
                        continue
                    
                    relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
                    if relevance_score >= relaxed_min_score:
                        valid_docs.append(doc)
                        if len(valid_docs) >= 5:  # 최소 5개까지는 보장 (3개 → 5개로 증가)
                            break
                
                if valid_docs:
                    self.logger.info(
                        f"✅ build_prompt_optimized_context: Relaxed criteria applied. "
                        f"Found {len(valid_docs)} documents with relaxed threshold ({relaxed_min_score:.3f})"
                    )
            
            if not valid_docs:
                self.logger.error("build_prompt_optimized_context: No valid documents with content found even after relaxing criteria")
                return {
                    "prompt_optimized_text": "",
                    "structured_documents": {},
                    "document_count": 0,
                    "total_context_length": 0
                }
            
            # 개선: Keyword Coverage 기반 필터링 (Phase 1) - 문서 손실 방지 (더 완화)
            docs_before_filter = len(valid_docs)
            if extracted_keywords:
                # 동적 임계값 계산 (검색 결과 수에 따라 조정) - 매우 완화된 기준
                num_valid_docs = len(valid_docs)
                if num_valid_docs >= 10:
                    min_coverage = 0.1  # 개선: 0.2 → 0.1로 더 완화
                elif num_valid_docs >= 5:
                    min_coverage = 0.05  # 개선: 0.1 → 0.05로 더 완화
                else:
                    min_coverage = 0.0  # 개선: 0.05 → 0.0으로 완전 완화 (결과가 적으면 필터링 안 함)
                
                # 문서가 10개 이하인 경우 필터링 건너뛰기 (문서 손실 방지)
                if num_valid_docs <= 10:
                    self.logger.debug(
                        f"🔍 [KEYWORD FILTERING] Skipping keyword coverage filter "
                        f"(documents={num_valid_docs} <= 10, preventing document loss)"
                    )
                else:
                    valid_docs = self.filter_by_keyword_coverage(
                        valid_docs,
                        extracted_keywords,
                        min_coverage=min_coverage
                    )
                
                # 문서 손실 로깅
                docs_after_filter = len(valid_docs)
                if docs_after_filter < docs_before_filter:
                    lost_count = docs_before_filter - docs_after_filter
                    self.logger.warning(
                        f"⚠️ [DOCUMENT LOSS] filter_by_keyword_coverage: {lost_count} documents lost "
                        f"({docs_before_filter} → {docs_after_filter}, min_coverage={min_coverage})"
                    )
            
            # textToSQL 결과와 벡터 임베딩 결과 분리
            text2sql_docs = []
            vector_docs = []
            seen_ids = set()
            
            for doc in valid_docs:
                doc_id = doc.get("id") or doc.get("document_id") or doc.get("doc_id") or str(doc.get("source", ""))
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                
                # textToSQL 결과 판별
                search_type = doc.get("search_type", "")
                direct_match = doc.get("direct_match", False)
                is_text2sql = (
                    search_type == "text2sql" or
                    search_type == "direct_statute" or
                    direct_match is True or
                    (doc.get("type") == "statute_article" and doc.get("statute_name") and doc.get("article_no"))
                )
                
                if is_text2sql:
                    text2sql_docs.append(doc)
                else:
                    vector_docs.append(doc)
            
            # 우선순위 5: 벡터 결과 관련성 점수 동적 임계값 적용
            # 우선순위 7: 성능 최적화 - 점수 계산 캐싱
            if vector_docs:
                # 점수 계산을 한 번만 수행하여 재사용
                doc_scores = []
                for doc in vector_docs:
                    score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                    doc_scores.append((doc, score))
                
                scores = [score for _, score in doc_scores]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                max_score = max(scores) if scores else 0.0
                min_score = min(scores) if scores else 0.0
                
                # 동적 임계값 계산: 평균 점수의 80% 또는 최소 0.60
                dynamic_threshold = max(0.60, min(0.75, avg_score * 0.8))
                
                # 점수 분포가 낮으면 임계값 완화
                if avg_score < 0.70:
                    dynamic_threshold = max(0.50, avg_score * 0.7)
                
                # statute_article 타입은 더 낮은 임계값 적용
                statute_docs = [d for d in vector_docs if (d.get("type") == "statute_article" or d.get("source_type") == "statute_article")]
                if statute_docs:
                    statute_threshold = max(0.40, dynamic_threshold * 0.8)
                else:
                    statute_threshold = dynamic_threshold
            else:
                dynamic_threshold = 0.75
                statute_threshold = 0.60
            
            # 우선순위 6: 성능 최적화 - 검증 결과 캐싱 및 배치 처리
            filtered_vector_docs = []
            validation_cache = {}  # doc_id -> validation_result
            
            for doc, score in doc_scores:
                doc_type = doc.get("type") or doc.get("source_type", "")
                
                # 타입별 차등 임계값 적용
                threshold = statute_threshold if doc_type == "statute_article" else dynamic_threshold
                
                if score >= threshold:
                    # 우선순위 6: 검증 결과 캐싱 (동일 문서 재검증 방지)
                    doc_id = doc.get("id") or doc.get("doc_id") or str(doc.get("source", ""))
                    if doc_id in validation_cache:
                        if validation_cache[doc_id]:
                            filtered_vector_docs.append(doc)
                        continue
                    
                    # 우선순위 2: 메타데이터 검증
                    metadata_valid = self._validate_document_metadata(doc)
                    if not metadata_valid:
                        validation_cache[doc_id] = False
                        continue
                    
                    # 우선순위 3: 내용 품질 검증
                    content = doc.get("content") or doc.get("text", "")
                    content_valid = self._validate_document_content_quality(doc, content)
                    if not content_valid:
                        validation_cache[doc_id] = False
                        continue
                    
                    # 우선순위 3: 출처 신뢰도 검증
                    source_valid = self._validate_document_source_reliability(doc)
                    if not source_valid:
                        validation_cache[doc_id] = False
                        continue
                    
                    # 모든 검증 통과
                    validation_cache[doc_id] = True
                    filtered_vector_docs.append(doc)
            
            # 결과가 부족하면 임계값을 점진적으로 낮춤
            min_docs_needed = 3
            if len(filtered_vector_docs) < min_docs_needed and len(vector_docs) >= min_docs_needed:
                # 임계값을 0.1씩 낮춰가며 재시도
                for relaxed_threshold in [dynamic_threshold - 0.1, dynamic_threshold - 0.2, 0.30]:
                    if len(filtered_vector_docs) >= min_docs_needed:
                        break
                    for doc in vector_docs:
                        if doc in filtered_vector_docs:
                            continue
                        score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                        if score >= relaxed_threshold:
                            # 검증은 완화된 기준으로 수행
                            content = doc.get("content") or doc.get("text", "")
                            if content and len(content.strip()) >= 5:  # 최소 길이만 확인
                                filtered_vector_docs.append(doc)
                                if len(filtered_vector_docs) >= min_docs_needed:
                                    break
                if len(filtered_vector_docs) < min_docs_needed:
                    self.logger.warning(
                        f"⚠️ [VECTOR FILTER] 최소 문서 수 미달: {len(filtered_vector_docs)}개 (목표: {min_docs_needed}개)"
                    )
            
            # 우선순위 7: 로깅 및 모니터링 개선 - 상세 필터링 통계
            if len(filtered_vector_docs) < len(vector_docs):
                filtered_count = len(vector_docs) - len(filtered_vector_docs)
                filter_reasons = {
                    "threshold": 0,
                    "metadata": 0,
                    "content": 0,
                    "source": 0
                }
                
                # 필터링 사유별 통계 (간단한 추정)
                for doc, score in doc_scores:
                    if doc not in filtered_vector_docs:
                        doc_type = doc.get("type") or doc.get("source_type", "")
                        threshold = statute_threshold if doc_type == "statute_article" else dynamic_threshold
                        if score < threshold:
                            filter_reasons["threshold"] += 1
                        else:
                            # 검증 실패 원인 추정
                            doc_id = doc.get("id") or doc.get("doc_id") or str(doc.get("source", ""))
                            if doc_id in validation_cache and not validation_cache[doc_id]:
                                # 어떤 검증이 실패했는지 확인 (간단한 추정)
                                if not self._validate_document_metadata(doc):
                                    filter_reasons["metadata"] += 1
                                elif not self._validate_document_content_quality(doc, doc.get("content") or doc.get("text", "")):
                                    filter_reasons["content"] += 1
                                elif not self._validate_document_source_reliability(doc):
                                    filter_reasons["source"] += 1
                
                self.logger.info(
                    f"🔀 [VECTOR FILTER] 관련성 점수 필터링: "
                    f"{len(vector_docs)}개 → {len(filtered_vector_docs)}개 "
                    f"(동적 임계값: {dynamic_threshold:.2f}, statute: {statute_threshold:.2f})"
                )
                self.logger.info(
                    f"📊 [FILTER STATS] 필터링 사유별 통계: "
                    f"임계값={filter_reasons['threshold']}, "
                    f"메타데이터={filter_reasons['metadata']}, "
                    f"내용={filter_reasons['content']}, "
                    f"출처={filter_reasons['source']}"
                )
            
            # 벡터 임베딩 결과만 재랭킹
            sorted_vector_docs = sorted(
                filtered_vector_docs,
                key=lambda x: (
                    x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                    x.get("similarity", 0.0),
                    x.get("keyword_match_score", 0.0)
                ),
                reverse=True
            )
            
            # textToSQL 결과를 최우선으로 포함
            max_docs_for_prompt = 10
            text2sql_count = len(text2sql_docs)
            max_vector_docs = max(0, max_docs_for_prompt - text2sql_count)
            
            # 벡터 결과 선택 (관련성 우선)
            if select_balanced_documents_func and sorted_vector_docs:
                selected_vector_docs = select_balanced_documents_func(
                    sorted_vector_docs, max_docs=max_vector_docs
                )
            else:
                selected_vector_docs = self.select_balanced_documents_relevance_first(
                    sorted_vector_docs, 
                    query=query,
                    extracted_keywords=extracted_keywords,
                    query_type=query_type,
                    max_docs=max_vector_docs
                ) if sorted_vector_docs else []
            
            if not selected_vector_docs and sorted_vector_docs:
                selected_vector_docs = sorted_vector_docs[:max_vector_docs]
            
            # textToSQL 결과 + 재랭킹된 벡터 결과 결합
            sorted_docs = text2sql_docs + selected_vector_docs
            
            self.logger.info(
                f"📋 [FINAL DOCS] textToSQL: {len(text2sql_docs)}개, "
                f"벡터(재랭킹): {len(selected_vector_docs)}개, "
                f"총: {len(sorted_docs)}개"
            )
            
            # 우선순위 7: 로깅 및 모니터링 개선 - 문서 손실 상세 분석
            if len(sorted_docs) < len(valid_docs):
                lost_count = len(valid_docs) - len(sorted_docs)
                loss_ratio = lost_count / len(valid_docs) if valid_docs else 0.0
                
                # 손실된 문서의 점수 분포 분석
                lost_docs = [doc for doc in valid_docs if doc not in sorted_docs]
                if lost_docs:
                    lost_scores = [
                        doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                        for doc in lost_docs
                    ]
                    avg_lost_score = sum(lost_scores) / len(lost_scores) if lost_scores else 0.0
                    min_lost_score = min(lost_scores) if lost_scores else 0.0
                    max_lost_score = max(lost_scores) if lost_scores else 0.0
                    
                    self.logger.warning(
                        f"⚠️ [DOCUMENT LOSS] select_balanced_documents: {lost_count} documents lost "
                        f"({len(valid_docs)} → {len(sorted_docs)}, max_docs={max_docs_for_prompt}, "
                        f"loss_ratio={loss_ratio:.1%})"
                    )
                    self.logger.info(
                        f"📊 [LOST DOCS STATS] 손실된 문서 점수 분포: "
                        f"평균={avg_lost_score:.3f}, 최대={max_lost_score:.3f}, 최소={min_lost_score:.3f}"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ [DOCUMENT LOSS] select_balanced_documents: {lost_count} documents lost "
                        f"({len(valid_docs)} → {len(sorted_docs)}, max_docs={max_docs_for_prompt})"
                    )
            
            # 우선순위 1 개선: 빈 문서 처리 - 원본 retrieved_docs에서 상위 문서 선택
            if not sorted_docs:
                self.logger.warning("⚠️ [EMPTY DOCS] build_prompt_optimized_context: sorted_docs is empty after filtering")
                # Fallback: 원본 valid_docs에서 상위 문서 선택
                if valid_docs:
                    # 점수 기준으로 정렬하여 상위 문서 선택
                    fallback_docs = sorted(
                        valid_docs,
                        key=lambda x: (
                            x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                            x.get("similarity", 0.0),
                            x.get("keyword_match_score", 0.0)
                        ),
                        reverse=True
                    )[:max_docs_for_prompt]
                    sorted_docs = fallback_docs
                    self.logger.info(
                        f"📋 [FALLBACK] 원본 문서에서 {len(sorted_docs)}개 선택 (fallback)"
                    )
                else:
                    self.logger.error("build_prompt_optimized_context: valid_docs도 비어있음")
                    return {
                        "prompt_optimized_text": "",
                        "structured_documents": {},
                        "document_count": 0,
                        "total_context_length": 0
                    }
            
            # 중복 문서 제거
            sorted_docs = self._deduplicate_documents(sorted_docs)
            
            if generate_document_based_instructions_func:
                document_instructions = generate_document_based_instructions_func(
                    documents=sorted_docs,
                    query=query,
                    query_type=query_type
                )
            else:
                document_instructions = self.generate_document_based_instructions(
                    documents=sorted_docs,
                    query=query,
                    query_type=query_type
                )
            
            prompt_section = f"""## 답변 생성 지시사항

{document_instructions}

## 참고 문서 목록

다음 {len(sorted_docs)}개의 문서를 반드시 참고하여 답변을 생성하세요.
각 문서는 관련성 점수와 핵심 내용이 표시되어 있습니다.

"""
            
            for idx, doc in enumerate(sorted_docs, 1):
                relevance_score = doc.get("final_weighted_score") or doc.get("relevance_score", 0.0)
                source = doc.get("source", "Unknown")
                content = doc.get("content", "")
                original_content_length = len(content)  # 원본 문서 길이 저장
                
                if extract_query_relevant_sentences_func:
                    relevant_sentences = extract_query_relevant_sentences_func(
                        doc_content=content,
                        query=query,
                        extracted_keywords=extracted_keywords
                    )
                elif self.query_enhancer:
                    relevant_sentences = self.query_enhancer.extract_query_relevant_sentences(
                        content, query, extracted_keywords
                    )
                else:
                    relevant_sentences = []
                
                doc_section = f"""
### 문서 {idx}: {source} (관련성 점수: {relevance_score:.2f})

**핵심 내용:**
"""
                
                if relevant_sentences:
                    doc_section += "\n".join([
                        f"- [중요] {sent['sentence']}"
                        for sent in relevant_sentences[:3]
                    ])
                    doc_section += "\n\n"
                
                # 컨텍스트 길이 최적화: 토큰 수 기반 동적 조정
                # 한글 기준 대략 1토큰 = 2-3자, 영어 기준 1토큰 = 4자
                # 안전하게 1토큰 = 2.5자로 계산
                max_tokens_per_doc = 600  # 문서당 최대 토큰 수
                max_content_length = int(max_tokens_per_doc * 2.5)  # 약 1500자
                
                # 질문 타입별 동적 조정
                if query_type == "law_inquiry":
                    max_tokens_per_doc = 800  # 법령 조회: 더 긴 컨텍스트 허용
                    max_content_length = int(max_tokens_per_doc * 2.5)  # 약 2000자
                elif query_type == "complex_question":
                    max_tokens_per_doc = 1000  # 복잡한 질문: 더 긴 컨텍스트 허용
                    max_content_length = int(max_tokens_per_doc * 2.5)  # 약 2500자
                
                # 스마트 문서 축약 (긴 문서 처리)
                is_truncated = False
                if len(content) > max_content_length:
                    is_truncated = True
                    content = self._smart_truncate_long_document(
                        content=content,
                        doc_type=doc.get("type", ""),
                        query=query,
                        extracted_keywords=extracted_keywords,
                        max_length=max_content_length,
                        relevant_sentences=relevant_sentences,
                        metadata=doc.get("metadata", {})
                    )
                
                # 프롬프트 구조 개선 (핵심 내용 + 전체 요약)
                if is_truncated:
                    doc_section += f"""**핵심 내용 (질문과 직접 관련된 부분):**
{content}

**문서 정보:**
- 전체 문서 길이: {original_content_length:,}자
- 추출된 핵심 내용: {len(content):,}자
- 축약 비율: {len(content)/original_content_length*100:.1f}%

---
"""
                else:
                    doc_section += f"""**전체 내용:**
{content}

---
"""
                
                prompt_section += doc_section
            
            prompt_section += """
## 문서 인용 규칙

답변에서 위 문서를 인용할 때는 다음과 같이 명시하세요:
- "문서 {0}에 따르면..." 또는 "[{0}] 인용 내용"
- 각 문서의 출처를 명확히 표시

## 중요 사항

- 위 문서의 내용을 바탕으로 답변을 생성하세요
- 문서에서 추론하거나 추측하지 말고, 문서에 명시된 내용만 사용하세요
- 문서에 없는 정보는 포함하지 마세요
- 여러 문서의 내용을 종합하여 일관된 답변을 구성하세요
""".format("n")
            
            content_validation = {
                "has_document_content": False,
                "total_content_length": 0,
                "documents_with_content": 0
            }
            
            # 개선 2.2: 최종 검증 강화 (더 정확한 검증)
            for doc in sorted_docs:
                content = self._extract_doc_content(doc)
                if content and len(content.strip()) >= 10:
                    # 여러 방법으로 문서 내용 포함 여부 확인
                    content_preview = content[:200]  # 더 긴 프리뷰로 확인
                    content_middle = content[len(content)//2:len(content)//2+200] if len(content) > 400 else ""
                    doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id", "")
                    source = doc.get("source", "") or doc.get("title", "")
                    
                    # 프롬프트에 문서 내용이 포함되어 있는지 확인
                    has_content = (
                        content_preview in prompt_section or
                        (content_middle and content_middle in prompt_section) or
                        (doc_id and f"문서 {doc_id}" in prompt_section) or
                        (source and source in prompt_section and len(content.strip()) > 0)
                    )
                    
                    if has_content:
                        content_validation["has_document_content"] = True
                        content_validation["total_content_length"] += len(content)
                        content_validation["documents_with_content"] += 1
            
            # 프롬프트 길이 검증
            if len(prompt_section.strip()) < 100:
                self.logger.error(
                    f"❌ [PROMPT BUILD] 프롬프트가 너무 짧음: "
                    f"length={len(prompt_section)}, "
                    f"valid_docs={len(sorted_docs)}"
                )
            
            # 문서 내용 포함 여부 확인
            has_document_content = any(
                len(self._extract_doc_content(doc).strip()) >= 10 
                for doc in sorted_docs
            )
            
            # 문서 내용이 없을 때 재구성 시도
            if not content_validation["has_document_content"] and len(sorted_docs) > 0:
                self.logger.error(
                    f"❌ [PROMPT BUILD] 프롬프트에 문서 내용 없음: "
                    f"valid_docs={len(sorted_docs)}, "
                    f"prompt_length={len(prompt_section)}"
                )
                
                # 재구성 시도: 문서 내용을 직접 추가
                self.logger.warning(
                    f"⚠️ [PROMPT BUILD] 문서 내용 재구성 시도 중..."
                )
                
                # 문서 내용이 있는 문서만 필터링
                docs_with_content = [
                    doc for doc in sorted_docs 
                    if len(self._extract_doc_content(doc).strip()) >= 10
                ]
                
                if docs_with_content:
                    # 프롬프트 재구성: 문서 내용 직접 추가
                    reconstructed_section = "\n\n## 참고 문서 내용\n\n"
                    for idx, doc in enumerate(docs_with_content[:5], 1):
                        content = self._extract_doc_content(doc)
                        if content and len(content.strip()) >= 10:
                            source = doc.get("source", "") or doc.get("title", "") or f"문서 {idx}"
                            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("document_id", f"doc_{idx}")
                            
                            reconstructed_section += f"### 문서 {idx}: {source} (ID: {doc_id})\n\n"
                            reconstructed_section += f"{content[:2000]}\n\n"  # 최대 2000자
                            reconstructed_section += "---\n\n"
                            
                            content_validation["has_document_content"] = True
                            content_validation["total_content_length"] += len(content)
                            content_validation["documents_with_content"] += 1
                    
                    # 재구성된 섹션을 프롬프트에 추가 (여러 위치 시도)
                    if "## 문서 인용 규칙" in prompt_section:
                        prompt_section = prompt_section.replace(
                            "## 문서 인용 규칙",
                            reconstructed_section + "## 문서 인용 규칙"
                        )
                    elif "## 참고 문서" in prompt_section:
                        prompt_section = prompt_section.replace(
                            "## 참고 문서",
                            reconstructed_section + "## 참고 문서"
                        )
                    elif "## 검색된 문서" in prompt_section:
                        prompt_section = prompt_section.replace(
                            "## 검색된 문서",
                            reconstructed_section + "## 검색된 문서"
                        )
                    else:
                        # 문서 인용 규칙이 없으면 프롬프트 끝에 추가
                        prompt_section = prompt_section + "\n\n" + reconstructed_section
                    
                    if content_validation["has_document_content"]:
                        self.logger.info(
                            f"✅ [PROMPT BUILD] 문서 내용 재구성 성공: "
                            f"{content_validation['documents_with_content']}개 문서 추가됨, "
                            f"프롬프트 길이: {len(prompt_section)}자"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ [PROMPT BUILD] 문서 내용 재구성 실패: "
                            f"문서 내용을 추출할 수 없음"
                        )
                else:
                    self.logger.warning(
                        f"⚠️ [PROMPT BUILD] 재구성할 문서 없음: "
                        f"모든 문서의 내용이 비어있음"
                    )
            else:
                self.logger.info(
                    f"✅ [PROMPT BUILD] 프롬프트에 문서 내용 포함됨: "
                    f"{content_validation['documents_with_content']}개 문서, "
                    f"총 내용 길이: {content_validation.get('total_content_length', 0)}자, "
                    f"프롬프트 길이: {len(prompt_section)}자"
                )
            
            if not content_validation["has_document_content"] and len(sorted_docs) > 0:
                self.logger.warning(
                    f"⚠️ [PROMPT BUILD] Content validation failed, but returning prompt anyway "
                    f"(may contain instructions only without actual document content)"
                )
            
            # 개선 10: 프롬프트 생성 후 최종 검증
            final_validation = self._validate_final_documents(
                sorted_docs=sorted_docs,
                query=query,
                extracted_keywords=extracted_keywords,
                query_type=query_type
            )
            
            if final_validation.get("low_relevance_warning"):
                self.logger.warning(
                    f"build_prompt_optimized_context: {final_validation['low_relevance_warning']} "
                    f"(low_relevance_count: {final_validation.get('low_relevance_count', 0)})"
                )
            
            return {
                "prompt_optimized_text": prompt_section,
                "structured_documents": {
                    "total_count": len(sorted_docs),
                    "documents": [{
                        "document_id": idx,
                        "source": doc.get("source", "Unknown"),
                        "relevance_score": doc.get("final_weighted_score") or doc.get("relevance_score", 0.0),
                        "content": (doc.get("content") or doc.get("text") or doc.get("content_text", ""))[:2000]
                    } for idx, doc in enumerate(sorted_docs, 1)]
                },
                "document_count": len(sorted_docs),
                "total_context_length": len(prompt_section),
                "content_validation": content_validation,
                "final_validation": final_validation
            }
        
        except Exception as e:
            self.logger.error(f"Prompt optimized context building failed: {e}", exc_info=True)
            return {
                "prompt_optimized_text": "",
                "structured_documents": {},
                "document_count": 0,
                "total_context_length": 0
            }
    
    def _smart_truncate_long_document(
        self,
        content: str,
        doc_type: str,
        query: str,
        extracted_keywords: List[str],
        max_length: int,
        relevant_sentences: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """긴 문서를 스마트하게 축약 (우선순위 1)"""
        if not content or len(content) <= max_length:
            return content
        
        doc_type_lower = doc_type.lower() if doc_type else ""
        metadata = metadata or {}
        
        # 문서 타입별 최적화
        if "case" in doc_type_lower or "precedent" in doc_type_lower:
            return self._extract_precedent_key_parts(
                content=content,
                metadata=metadata,
                query=query,
                keywords=extracted_keywords,
                max_length=max_length,
                relevant_sentences=relevant_sentences
            )
        elif "statute" in doc_type_lower:
            return self._extract_statute_key_parts(
                content=content,
                metadata=metadata,
                query=query,
                keywords=extracted_keywords,
                max_length=max_length,
                relevant_sentences=relevant_sentences
            )
        elif "interpretation" in doc_type_lower or "decision" in doc_type_lower:
            return self._extract_interpretation_key_parts(
                content=content,
                metadata=metadata,
                query=query,
                keywords=extracted_keywords,
                max_length=max_length,
                relevant_sentences=relevant_sentences
            )
        else:
            return self._extract_general_key_parts(
                content=content,
                query=query,
                keywords=extracted_keywords,
                max_length=max_length,
                relevant_sentences=relevant_sentences
            )
    
    def _extract_precedent_key_parts(
        self,
        content: str,
        metadata: Dict[str, Any],
        query: str,
        keywords: List[str],
        max_length: int,
        relevant_sentences: List[Dict[str, Any]] = None
    ) -> str:
        """판례 핵심 부분 추출 (우선순위 2)"""
        parts = []
        remaining_length = max_length
        
        # 1. 판시사항 (최우선, 최대 400자)
        holding_text = None
        if metadata.get("case_holding"):
            holding_text = metadata["case_holding"][:400]
        elif "판시사항" in content:
            holding_match = re.search(r'판시사항[:\s]*(.+?)(?=\n|판결요지|$)', content, re.DOTALL)
            if holding_match:
                holding_text = holding_match.group(1).strip()[:400]
        
        if holding_text and len(holding_text) <= remaining_length:
            parts.append(f"**판시사항**: {holding_text}")
            remaining_length -= len(holding_text) + 20
        
        # 2. 판결요지 (최대 400자)
        reasoning_text = None
        if metadata.get("case_reasoning"):
            reasoning_text = metadata["case_reasoning"][:400]
        elif "판결요지" in content:
            reasoning_match = re.search(r'판결요지[:\s]*(.+?)(?=\n|$)', content, re.DOTALL)
            if reasoning_match:
                reasoning_text = reasoning_match.group(1).strip()[:400]
        
        if reasoning_text and len(reasoning_text) <= remaining_length:
            parts.append(f"**판결요지**: {reasoning_text}")
            remaining_length -= len(reasoning_text) + 20
        
        # 3. 관련 문장 (키워드 포함, 최대 3개)
        if relevant_sentences:
            relevant_list = []
            for sent in relevant_sentences[:3]:
                sent_text = sent.get("sentence", "")[:300]
                if sent_text and len(sent_text) <= remaining_length - 50:
                    relevant_list.append(f"- {sent_text}")
                    remaining_length -= len(sent_text) + 10
            
            if relevant_list:
                parts.append(f"**관련 문장**:\n" + "\n".join(relevant_list))
        
        # 4. 키워드 주변 문맥 추출 (남은 공간 활용)
        if remaining_length > 200 and keywords:
            keyword_contexts = self._extract_keyword_contexts(content, keywords, remaining_length)
            if keyword_contexts:
                parts.append(f"**키워드 관련 문맥**:\n{keyword_contexts}")
        
        result = "\n\n".join(parts)
        
        # 최종 길이 확인
        if len(result) > max_length:
            # 비율에 맞춰 축약
            ratio = max_length / len(result)
            result = "\n\n".join([
                part[:int(len(part) * ratio)] + ("..." if len(part) > int(len(part) * ratio) else "")
                for part in parts
            ])
        
        return result[:max_length] if len(result) > max_length else result
    
    def _extract_statute_key_parts(
        self,
        content: str,
        metadata: Dict[str, Any],
        query: str,
        keywords: List[str],
        max_length: int,
        relevant_sentences: List[Dict[str, Any]] = None
    ) -> str:
        """법령 핵심 부분 추출"""
        parts = []
        remaining_length = max_length
        
        # 1. 조문번호 정보
        article_no = metadata.get("article_no") or metadata.get("article_number")
        if article_no:
            parts.append(f"**조문번호**: 제{article_no}조")
            remaining_length -= 50
        
        # 2. 제목/헤딩
        heading = metadata.get("heading") or metadata.get("title")
        if heading and len(heading) <= remaining_length:
            parts.append(f"**제목**: {heading[:200]}")
            remaining_length -= len(heading) + 20
        
        # 3. 관련 문장 (최우선)
        if relevant_sentences:
            relevant_list = []
            for sent in relevant_sentences[:5]:
                sent_text = sent.get("sentence", "")[:400]
                if sent_text and len(sent_text) <= remaining_length - 50:
                    relevant_list.append(f"- {sent_text}")
                    remaining_length -= len(sent_text) + 10
            
            if relevant_list:
                parts.append(f"**관련 조문 내용**:\n" + "\n".join(relevant_list))
        
        # 4. 키워드 주변 문맥
        if remaining_length > 200 and keywords:
            keyword_contexts = self._extract_keyword_contexts(content, keywords, remaining_length)
            if keyword_contexts:
                parts.append(f"**관련 문맥**:\n{keyword_contexts}")
        
        result = "\n\n".join(parts)
        return result[:max_length] if len(result) > max_length else result
    
    def _extract_interpretation_key_parts(
        self,
        content: str,
        metadata: Dict[str, Any],
        query: str,
        keywords: List[str],
        max_length: int,
        relevant_sentences: List[Dict[str, Any]] = None
    ) -> str:
        """해석례/결정례 핵심 부분 추출"""
        parts = []
        remaining_length = max_length
        
        # 1. 제목
        title = metadata.get("title") or metadata.get("heading")
        if title and len(title) <= remaining_length:
            parts.append(f"**제목**: {title[:200]}")
            remaining_length -= len(title) + 20
        
        # 2. 관련 문장 (최우선)
        if relevant_sentences:
            relevant_list = []
            for sent in relevant_sentences[:5]:
                sent_text = sent.get("sentence", "")[:400]
                if sent_text and len(sent_text) <= remaining_length - 50:
                    relevant_list.append(f"- {sent_text}")
                    remaining_length -= len(sent_text) + 10
            
            if relevant_list:
                parts.append(f"**핵심 내용**:\n" + "\n".join(relevant_list))
        
        # 3. 키워드 주변 문맥
        if remaining_length > 200 and keywords:
            keyword_contexts = self._extract_keyword_contexts(content, keywords, remaining_length)
            if keyword_contexts:
                parts.append(f"**관련 문맥**:\n{keyword_contexts}")
        
        result = "\n\n".join(parts)
        return result[:max_length] if len(result) > max_length else result
    
    def _extract_general_key_parts(
        self,
        content: str,
        query: str,
        keywords: List[str],
        max_length: int,
        relevant_sentences: List[Dict[str, Any]] = None
    ) -> str:
        """일반 문서 핵심 부분 추출"""
        parts = []
        remaining_length = max_length
        
        # 1. 관련 문장 (최우선)
        if relevant_sentences:
            relevant_list = []
            for sent in relevant_sentences[:5]:
                sent_text = sent.get("sentence", "")[:400]
                if sent_text and len(sent_text) <= remaining_length - 50:
                    relevant_list.append(f"- {sent_text}")
                    remaining_length -= len(sent_text) + 10
            
            if relevant_list:
                parts.append(f"**핵심 내용**:\n" + "\n".join(relevant_list))
        
        # 2. 키워드 주변 문맥
        if remaining_length > 200 and keywords:
            keyword_contexts = self._extract_keyword_contexts(content, keywords, remaining_length)
            if keyword_contexts:
                parts.append(f"**관련 문맥**:\n{keyword_contexts}")
        
        # 3. 폴백: 앞부분 + 뒷부분
        if not parts and len(content) > max_length:
            front = content[:max_length // 2]
            back = content[-max_length // 2:] if len(content) > max_length else ""
            return f"{front}\n\n[... 중간 생략 ...]\n\n{back}"
        
        result = "\n\n".join(parts) if parts else content[:max_length]
        return result[:max_length] if len(result) > max_length else result
    
    def _extract_keyword_contexts(
        self,
        content: str,
        keywords: List[str],
        max_length: int
    ) -> str:
        """키워드 주변 문맥 추출"""
        if not keywords or not content:
            return ""
        
        contexts = []
        content_lower = content.lower()
        used_positions = set()
        
        for keyword in keywords[:5]:
            if not keyword or len(keyword) < 2:
                continue
            
            keyword_lower = keyword.lower()
            if keyword_lower not in content_lower:
                continue
            
            # 키워드 위치 찾기 (중복 방지)
            start_pos = 0
            while True:
                idx = content_lower.find(keyword_lower, start_pos)
                if idx == -1:
                    break
                
                # 이미 사용된 위치 근처인지 확인
                is_duplicate = any(abs(idx - pos) < 100 for pos in used_positions)
                if not is_duplicate:
                    # 앞뒤 200자씩 추출
                    context_start = max(0, idx - 200)
                    context_end = min(len(content), idx + len(keyword) + 200)
                    context = content[context_start:context_end]
                    
                    if context and context not in contexts:
                        contexts.append(context)
                        used_positions.add(idx)
                    
                    if len("\n\n[...]\n\n".join(contexts)) >= max_length:
                        break
                
                start_pos = idx + 1
            
            if len("\n\n[...]\n\n".join(contexts)) >= max_length:
                break
        
        if contexts:
            result = "\n\n[...]\n\n".join(contexts[:3])
            return result[:max_length]
        
        return ""
    
    def select_balanced_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        max_docs: int = 10
    ) -> List[Dict[str, Any]]:
        """의미적 검색과 키워드 검색 결과의 균형을 맞춰서 문서 선택 (문서 손실 방지 강화)"""
        if not sorted_docs:
            return []
        
        # 개선: 문서 수가 max_docs보다 적으면 모든 문서 반환 (손실 방지)
        if len(sorted_docs) <= max_docs:
            self.logger.debug(
                f"✅ [DOCUMENT SELECTION] 모든 문서 선택 (문서 수={len(sorted_docs)} <= max_docs={max_docs})"
            )
            return sorted_docs
        
        semantic_docs = [doc for doc in sorted_docs if doc.get("search_type") == "semantic"]
        keyword_docs = [doc for doc in sorted_docs if doc.get("search_type") == "keyword"]
        hybrid_docs = [doc for doc in sorted_docs if doc.get("search_type") not in ["semantic", "keyword"]]
        
        selected_docs = []
        seen_ids = set()  # 중복 방지를 위한 ID 추적
        
        # 문서 ID 추출 함수
        def get_doc_id(doc):
            return (
                doc.get("id") or 
                doc.get("doc_id") or 
                doc.get("document_id") or 
                id(doc)  # 최후의 폴백
            )
        
        top_count = max(1, max_docs // 2)
        for doc in sorted_docs[:top_count]:
            doc_id = get_doc_id(doc)
            if doc_id not in seen_ids:
                selected_docs.append(doc)
                seen_ids.add(doc_id)
        
        remaining_slots = max_docs - len(selected_docs)
        
        if remaining_slots > 0:
            semantic_to_add = []
            for doc in semantic_docs:
                doc_id = get_doc_id(doc)
                if doc_id not in seen_ids:
                    semantic_to_add.append(doc)
            
            keyword_to_add = []
            for doc in keyword_docs:
                doc_id = get_doc_id(doc)
                if doc_id not in seen_ids:
                    keyword_to_add.append(doc)
            
            max_alternate = remaining_slots // 2
            for i in range(min(max_alternate, max(len(semantic_to_add), len(keyword_to_add)))):
                if i < len(semantic_to_add) and len(selected_docs) < max_docs:
                    doc = semantic_to_add[i]
                    doc_id = get_doc_id(doc)
                    if doc_id not in seen_ids:
                        selected_docs.append(doc)
                        seen_ids.add(doc_id)
                if i < len(keyword_to_add) and len(selected_docs) < max_docs:
                    doc = keyword_to_add[i]
                    doc_id = get_doc_id(doc)
                    if doc_id not in seen_ids:
                        selected_docs.append(doc)
                        seen_ids.add(doc_id)
            
            if len(selected_docs) < max_docs:
                for doc in hybrid_docs:
                    doc_id = get_doc_id(doc)
                    if doc_id not in seen_ids and len(selected_docs) < max_docs:
                        selected_docs.append(doc)
                        seen_ids.add(doc_id)
            
            if len(selected_docs) < max_docs:
                for doc in sorted_docs:
                    doc_id = get_doc_id(doc)
                    if doc_id not in seen_ids and len(selected_docs) < max_docs:
                        selected_docs.append(doc)
                        seen_ids.add(doc_id)
        
        selected_docs = sorted(
            selected_docs,
            key=lambda x: (
                x.get("final_weighted_score", x.get("relevance_score", 0.0)),
                x.get("keyword_match_score", 0.0)
            ),
            reverse=True
        )
        
        result = selected_docs[:max_docs]
        
        # 문서 손실 확인 및 로깅
        if len(result) < len(sorted_docs):
            lost_count = len(sorted_docs) - len(result)
            loss_ratio = lost_count / len(sorted_docs) if sorted_docs else 0.0
            self.logger.warning(
                f"⚠️ [DOCUMENT LOSS] select_balanced_documents: {lost_count} documents lost "
                f"({len(sorted_docs)} → {len(result)}, max_docs={max_docs}, loss_ratio={loss_ratio:.1%})"
            )
        else:
            self.logger.debug(
                f"✅ [DOCUMENT SELECTION] 문서 선택 완료: {len(result)}개 선택됨"
            )
        
        return result
    
    def select_balanced_documents_relevance_first(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str,
        max_docs: int = 7
    ) -> List[Dict[str, Any]]:
        """
        개선 12: 관련성 우선 문서 선택 (다양성보다 관련성 우선) - 문서 손실 방지 강화
        
        Args:
            sorted_docs: 점수로 정렬된 문서 리스트
            query: 사용자 질문
            extracted_keywords: 추출된 키워드 리스트
            query_type: 질문 유형
            max_docs: 선택할 최대 문서 수
        
        Returns:
            관련성이 높은 문서 리스트
        """
        if not sorted_docs:
            return []
        
        # 개선: 문서 수가 max_docs보다 적으면 모든 문서 반환 (손실 방지)
        if len(sorted_docs) <= max_docs:
            self.logger.debug(
                f"✅ [DOCUMENT SELECTION] 모든 문서 선택 (문서 수={len(sorted_docs)} <= max_docs={max_docs})"
            )
            return sorted_docs
        
        # 개선 9: 질문 유형별 문서 필터링 기준 적용
        query_lower = query.lower()
        query_keywords_lower = [kw.lower() for kw in extracted_keywords if kw and len(kw) > 1]
        
        selected_docs = []
        seen_sources = set()
        
        # 개선: Citation 가능성이 높은 문서 식별 (법령 조문, 판례 등)
        import re
        citation_pattern = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
        precedent_pattern = re.compile(r'[가-힣]+(?:지방)?법원|대법원|판결|사건')
        
        # 문서에 citation 점수 부여
        for doc in sorted_docs:
            content = (doc.get("content") or doc.get("text") or "").lower()
            doc_type = doc.get("type") or doc.get("source_type") or ""
            
            citation_score = 0.0
            # 법령 조문 타입이면 높은 점수
            if doc_type in ["statute_article", "statute"]:
                citation_score += 0.5
            # 판례 타입이면 높은 점수
            elif doc_type in ["case_paragraph", "precedent", "decision_paragraph"]:
                citation_score += 0.4
            # 내용에서 법령 조문 발견
            if citation_pattern.search(content):
                citation_score += 0.3
            # 내용에서 판례 발견
            if precedent_pattern.search(content):
                citation_score += 0.2
            
            doc["citation_potential_score"] = min(1.0, citation_score)
        
        # 개선: 문서 수가 적을 때 필터링 완화 (문서 손실 방지)
        # 문서 수가 max_docs의 1.5배 이하면 모든 문서 반환
        if len(sorted_docs) <= int(max_docs * 1.5):
            self.logger.debug(
                f"✅ [DOCUMENT SELECTION] 문서 수가 적어 모든 문서 선택 "
                f"(문서 수={len(sorted_docs)} <= {int(max_docs * 1.5)})"
            )
            return sorted_docs[:max_docs]
        
        # 우선순위 1 개선: 관련도 임계값 대폭 완화 (문서 손실 방지)
        # 1단계: 관련도가 높은 문서 우선 선택 (임계값 완화: 0.40 → 0.20)
        high_relevance_docs = [
            doc for doc in sorted_docs 
            if (doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)) >= 0.20
        ]
        
        # citation 가능성 순으로 정렬
        high_relevance_docs.sort(
            key=lambda x: (
                x.get("citation_potential_score", 0.0),
                x.get("relevance_score", 0.0) or x.get("final_weighted_score", 0.0)
            ),
            reverse=True
        )
        
        for doc in high_relevance_docs:
            if len(selected_docs) >= max_docs:
                break
            
            source = doc.get("source", "") or doc.get("id", "") or str(doc.get("doc_id", ""))
            if not source or source not in seen_sources:
                selected_docs.append(doc)
                if source:
                    seen_sources.add(source)
        
        # 2단계: 관련도가 낮아도 citation 가능성이 높은 문서 선택 (임계값 완화: 0.30 → 0.10)
        if len(selected_docs) < max_docs:
            low_relevance_docs = [
                doc for doc in sorted_docs 
                if (doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)) >= 0.10
                and doc not in selected_docs
            ]
            
            low_relevance_docs.sort(
                key=lambda x: (
                    x.get("citation_potential_score", 0.0),
                    x.get("relevance_score", 0.0) or x.get("final_weighted_score", 0.0)
                ),
                reverse=True
            )
            
            for doc in low_relevance_docs:
                if len(selected_docs) >= max_docs:
                    break
                
                content = (doc.get("content") or doc.get("text") or "").lower()
                has_relevant_keyword = False
                
                for qkw in query_keywords_lower:
                    if qkw in content or qkw in query_lower:
                        has_relevant_keyword = True
                        break
                
                citation_potential = doc.get("citation_potential_score", 0.0)
                keyword_match = doc.get("keyword_match_score", 0.0)
                
                if citation_potential >= 0.2 or has_relevant_keyword or keyword_match > 0.0:
                    source = doc.get("source", "")
                    if not source or source not in seen_sources:
                        selected_docs.append(doc)
                        if source:
                            seen_sources.add(source)
        
        # 3단계: 부족하면 상위 문서로 채우기 (필터링 없이)
        if len(selected_docs) < max_docs:
            for doc in sorted_docs:
                if len(selected_docs) >= max_docs:
                    break
                if doc not in selected_docs:
                    selected_docs.append(doc)
        
        # 최소 문서 수 보장 (문서 손실 방지)
        min_docs = min(len(sorted_docs), max_docs)
        if len(selected_docs) < min_docs:
            for doc in sorted_docs:
                if len(selected_docs) >= min_docs:
                    break
                if doc not in selected_docs:
                    selected_docs.append(doc)
            self.logger.info(
                f"📊 [MIN DOCS] 최소 문서 수 보장: {len(selected_docs)}개 (목표: {min_docs}개)"
            )
        
        # 문서 손실 확인 및 로깅
        if len(selected_docs) < len(sorted_docs):
            lost_count = len(sorted_docs) - len(selected_docs)
            loss_ratio = lost_count / len(sorted_docs) if sorted_docs else 0.0
            self.logger.warning(
                f"⚠️ [DOCUMENT LOSS] select_balanced_documents_relevance_first: {lost_count} documents lost "
                f"({len(sorted_docs)} → {len(selected_docs)}, max_docs={max_docs}, loss_ratio={loss_ratio:.1%})"
            )
        else:
            self.logger.debug(
                f"✅ [DOCUMENT SELECTION] 문서 선택 완료: {len(selected_docs)}개 선택됨"
            )
        
        self.logger.info(
            f"select_balanced_documents_relevance_first: Selected {len(selected_docs)}/{len(sorted_docs)} documents "
            f"(high_relevance: {len([d for d in selected_docs if (d.get('relevance_score', 0.0) or d.get('final_weighted_score', 0.0)) >= 0.40])}, "
            f"medium_relevance: {len([d for d in selected_docs if 0.30 <= (d.get('relevance_score', 0.0) or d.get('final_weighted_score', 0.0)) < 0.40])})"
        )
        
        return selected_docs[:max_docs]
    
    def select_diverse_documents(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        max_docs: int = 7,
        diversity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) 알고리즘을 사용한 다양성과 관련성의 균형을 맞춘 문서 선택
        
        Args:
            documents: 선택할 문서 리스트 (이미 점수로 정렬된 상태)
            query: 검색 쿼리
            max_docs: 선택할 최대 문서 수
            diversity_weight: 다양성 가중치 (0.0 ~ 1.0, 높을수록 다양성 중시)
        
        Returns:
            다양성과 관련성이 균형잡힌 문서 리스트
        """
        if not documents:
            return []
        
        selected = []
        remaining = documents.copy()
        
        # 첫 번째 문서: 가장 관련성 높은 문서
        if remaining:
            selected.append(remaining.pop(0))
        
        # 나머지 문서: MMR 점수로 선택
        while len(selected) < max_docs and remaining:
            best_doc = None
            best_score = -1
            
            for doc in remaining:
                # 관련성 점수
                relevance = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                
                # 다양성 점수 (이미 선택된 문서와의 유사도 최소화)
                min_similarity = 1.0
                doc_content = (doc.get("content") or doc.get("text") or "").lower()
                doc_words = set(doc_content.split())
                
                for selected_doc in selected:
                    selected_content = (selected_doc.get("content") or selected_doc.get("text") or "").lower()
                    selected_words = set(selected_content.split())
                    
                    # Jaccard 유사도 계산
                    if doc_words or selected_words:
                        intersection = len(doc_words & selected_words)
                        union = len(doc_words | selected_words)
                        similarity = intersection / union if union > 0 else 0.0
                        min_similarity = min(min_similarity, similarity)
                
                # MMR 점수: (1 - diversity_weight) * relevance + diversity_weight * (1 - similarity)
                mmr_score = (
                    (1 - diversity_weight) * relevance +
                    diversity_weight * (1 - min_similarity)
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = doc
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        self.logger.info(
            f"MMR diversity selection: {len(selected)}/{len(documents)} documents selected "
            f"(diversity_weight={diversity_weight:.2f})"
        )
        
        return selected
    
    def _validate_final_documents(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str,
        extracted_keywords: List[str],
        query_type: str
    ) -> Dict[str, Any]:
        """
        개선 10: 프롬프트 생성 후 최종 검증
        
        Args:
            sorted_docs: 최종 선택된 문서 리스트
            query: 사용자 질문
            extracted_keywords: 추출된 키워드 리스트
            query_type: 질문 유형
        
        Returns:
            검증 결과 딕셔너리
        """
        validation_result = {
            "total_docs": len(sorted_docs),
            "high_relevance_count": 0,
            "medium_relevance_count": 0,
            "low_relevance_count": 0,
            "low_relevance_warning": None,
            "avg_relevance_score": 0.0,
            "min_relevance_score": 0.0,
            "max_relevance_score": 0.0
        }
        
        if not sorted_docs:
            return validation_result
        
        relevance_scores = []
        query_lower = query.lower()
        query_keywords_lower = [kw.lower() for kw in extracted_keywords if kw and len(kw) > 1]
        
        for doc in sorted_docs:
            relevance_score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            relevance_scores.append(relevance_score)
            
            if relevance_score >= 0.65:
                validation_result["high_relevance_count"] += 1
            elif relevance_score >= 0.55:
                validation_result["medium_relevance_count"] += 1
            else:
                validation_result["low_relevance_count"] += 1
                
                # 관련도가 낮은 문서의 경우 키워드 매칭 확인
                content = (doc.get("content") or doc.get("text") or "").lower()
                has_keyword = False
                for qkw in query_keywords_lower:
                    if qkw in content:
                        has_keyword = True
                        break
                
                if not has_keyword and doc.get("keyword_match_score", 0.0) == 0.0:
                    source = doc.get("source", "Unknown")
                    if not validation_result["low_relevance_warning"]:
                        validation_result["low_relevance_warning"] = f"Low relevance documents detected: {source}"
                    else:
                        validation_result["low_relevance_warning"] += f", {source}"
        
        if relevance_scores:
            validation_result["avg_relevance_score"] = sum(relevance_scores) / len(relevance_scores)
            validation_result["min_relevance_score"] = min(relevance_scores)
            validation_result["max_relevance_score"] = max(relevance_scores)
        
        # 경고 조건: 관련도가 낮은 문서가 전체의 30% 이상이거나 평균 관련도가 0.60 미만
        if validation_result["low_relevance_count"] > 0:
            low_relevance_ratio = validation_result["low_relevance_count"] / validation_result["total_docs"]
            if low_relevance_ratio >= 0.3 or validation_result["avg_relevance_score"] < 0.60:
                if not validation_result["low_relevance_warning"]:
                    validation_result["low_relevance_warning"] = (
                        f"Low relevance ratio: {low_relevance_ratio:.1%}, "
                        f"avg_score: {validation_result['avg_relevance_score']:.3f}"
                    )
        
        self.logger.info(
            f"_validate_final_documents: Validation complete - "
            f"total: {validation_result['total_docs']}, "
            f"high: {validation_result['high_relevance_count']}, "
            f"medium: {validation_result['medium_relevance_count']}, "
            f"low: {validation_result['low_relevance_count']}, "
            f"avg_score: {validation_result['avg_relevance_score']:.3f}"
        )
        
        return validation_result
    
    def select_high_value_documents(
        self,
        documents: List[Dict],
        query: str,
        min_relevance: float = 0.7,
        max_docs: int = 5
    ) -> List[Dict]:
        """정보 밀도 기반 문서 선택"""
        if not documents:
            return documents
        
        try:
            high_value_docs = []
            
            for doc in documents:
                doc_content = doc.get("content", "")
                if not doc_content or len(doc_content) < 20:
                    continue
                
                citation_pattern = r'[가-힣]+법\s*제?\s*\d+\s*조'
                citations = re.findall(citation_pattern, doc_content)
                citation_count = len(citations)
                citation_score = min(1.0, citation_count / 5.0)
                
                query_words = set(query.lower().split())
                content_words = set(doc_content.lower().split())
                explanation_completeness = 0.0
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    explanation_completeness = min(1.0, overlap / max(1, len(query_words)))
                
                sentences = doc_content.split('。') or doc_content.split('.')
                avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(1, len(sentences))
                
                descriptive_score_bonus = 0.0
                if 20 <= avg_sentence_length <= 100:
                    descriptive_score_bonus = 0.2
                elif avg_sentence_length > 100:
                    descriptive_score_bonus = 0.1
                
                explanation_completeness = min(1.0, explanation_completeness + descriptive_score_bonus)
                
                keyword_coverage = 0.0
                if query_words and content_words:
                    keyword_coverage = len(query_words.intersection(content_words)) / max(1, len(query_words))
                
                relevance_score = doc.get("final_relevance_score") or doc.get("combined_score", 0.0) or doc.get("relevance_score", 0.0)
                
                information_density = (
                    0.3 * citation_score +
                    0.3 * explanation_completeness +
                    0.2 * keyword_coverage +
                    0.2 * min(1.0, relevance_score)
                )
                
                doc["information_density_score"] = information_density
                doc["citation_count"] = citation_count
                doc["explanation_completeness"] = explanation_completeness
                
                combined_value_score = 0.6 * relevance_score + 0.4 * information_density
                doc["combined_value_score"] = combined_value_score
                
                if combined_value_score >= min_relevance:
                    high_value_docs.append(doc)
            
            high_value_docs.sort(key=lambda x: x.get("combined_value_score", 0.0), reverse=True)
            
            selected_docs = high_value_docs[:max_docs]
            
            self.logger.info(
                f"📚 [HIGH VALUE SELECTION] Selected {len(selected_docs)}/{len(documents)} documents. "
                f"Avg density: {sum(d.get('information_density_score', 0.0) for d in selected_docs) / max(1, len(selected_docs)):.3f}"
            )
            
            return selected_docs
        
        except Exception as e:
            self.logger.warning(f"High value document selection failed: {e}, using first {max_docs} documents")
            return documents[:max_docs]
    
    def filter_by_keyword_coverage(
        self,
        documents: List[Dict[str, Any]],
        extracted_keywords: List[str],
        min_coverage: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Keyword Coverage 기반 필터링 (개선: Phase 1)"""
        if not documents or not extracted_keywords:
            return documents
        
        filtered = []
        excluded_count = 0
        
        for doc in documents:
            keyword_coverage = doc.get("keyword_coverage", 0.0)
            
            # Keyword Coverage가 임계값 이상인 문서만 포함
            if keyword_coverage >= min_coverage:
                filtered.append(doc)
            else:
                # 개선: 핵심 키워드 매칭 확인 강화 (문서 내용에서 직접 확인)
                has_core_keyword = False
                content = (doc.get("content", "") or doc.get("text", "")).lower()
                core_keywords = extracted_keywords[:3] if len(extracted_keywords) >= 3 else extracted_keywords
                
                # matched_keywords에서 확인
                matched_keywords = doc.get("matched_keywords", [])
                if matched_keywords:
                    has_core_keyword = any(
                        str(kw).lower() in [str(mk).lower() for mk in matched_keywords] 
                        for kw in core_keywords if isinstance(kw, str)
                    )
                
                # 문서 내용에서 직접 확인 (matched_keywords가 없는 경우)
                if not has_core_keyword and content:
                    has_core_keyword = any(
                        str(kw).lower() in content 
                        for kw in core_keywords if isinstance(kw, str) and len(kw) >= 2
                    )
                
                if has_core_keyword:
                    filtered.append(doc)
                    self.logger.debug(
                        f"Document included due to core keyword match: "
                        f"coverage={keyword_coverage:.3f}, core_keywords={core_keywords[:2]}"
                    )
                else:
                    excluded_count += 1
                    self.logger.debug(
                        f"Document filtered: coverage={keyword_coverage:.3f} < {min_coverage}, "
                        f"no core keyword match"
                    )
        
        if excluded_count > 0:
            self.logger.info(
                f"🔍 [KEYWORD FILTERING] Filtered {excluded_count}/{len(documents)} documents "
                f"by keyword coverage (min_coverage={min_coverage})"
            )
        
        return filtered
    
    def generate_document_based_instructions(
        self,
        documents: List[Dict[str, Any]],
        query: str,
        query_type: str
    ) -> str:
        """문서를 기반으로 답변 생성하라는 명시적 지시사항 생성"""
        instructions = f"""당신은 법률 전문가입니다. 아래 제공된 문서들을 반드시 참고하여 다음 질문에 답변하세요.

**질문**: {query}
**질문 유형**: {query_type}

**답변 생성 규칙**:
1. **문서 기반 답변**: 제공된 문서의 내용을 바탕으로 답변을 생성하세요.
2. **문서 인용 필수**: 답변에서 문서를 인용할 때는 "문서 [번호]에 따르면..." 형식으로 명시하세요.
3. **정확성**: 문서에 명시된 내용만 사용하고, 추론하거나 추측하지 마세요.
4. **구조화**: 답변은 다음 구조로 작성하세요:
   - 핵심 답변
   - 관련 법령 및 조항
   - 실무 적용 시 주의사항
   - 참고할 만한 판례 (있는 경우)
5. **출처 명시**: 각 인용문에 대해 문서 번호를 명시하세요.
"""
        
        return instructions
    
    def _validate_document_metadata(self, doc: Dict[str, Any]) -> bool:
        """우선순위 4 개선: 메타데이터 검증 (완화된 기준)"""
        # 필수 필드만 검증 (content는 필수, source와 type은 선택적)
        has_content = bool(doc.get("content") or doc.get("text"))
        
        if not has_content:
            return False
        
        # source와 type이 없어도 경고만 출력하고 통과
        has_source = bool(doc.get("source"))
        has_type = bool(doc.get("type") or doc.get("source_type"))
        
        if not has_source:
            self.logger.debug(f"⚠️ [METADATA] source 필드 없음: {doc.get('id', 'unknown')}")
        if not has_type:
            self.logger.debug(f"⚠️ [METADATA] type 필드 없음: {doc.get('id', 'unknown')}")
        
        # 메타데이터 완전성 검증
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            # metadata가 있으면 최소한의 구조는 있어야 함
            pass
        
        return True
    
    def _validate_document_content_quality(self, doc: Dict[str, Any], content: str) -> bool:
        """우선순위 4 개선: 문서 내용 품질 검증 (완화된 기준)"""
        # 최소 길이 완화: 10자 → 5자
        if not content or len(content.strip()) < 5:
            return False
        
        content_stripped = content.strip()
        
        # 특수 문자만 있는 문서 제외
        # 의미 있는 문자(한글, 영문, 숫자) 비율 계산
        meaningful_chars = re.findall(r'[가-힣a-zA-Z0-9]', content_stripped)
        total_chars = len(content_stripped)
        if total_chars == 0:
            return False
        
        meaningful_ratio = len(meaningful_chars) / total_chars
        # 의미 있는 문자 비율 완화: 50% → 40%
        if meaningful_ratio < 0.4:
            return False
        
        # 불완전한 문장 제외 (문장 끝이 없는 경우가 너무 많으면 제외)
        # 100자 이상인 경우에만 문장 끝 확인 (더 긴 텍스트에서만 적용)
        sentence_endings = content_stripped.count('.') + content_stripped.count('。') + content_stripped.count('!') + content_stripped.count('?')
        if len(content_stripped) > 200 and sentence_endings == 0:
            # 200자 이상인데 문장 끝이 없으면 제외 (100자 → 200자로 완화)
            return False
        
        return True
    
    def _validate_document_source_reliability(self, doc: Dict[str, Any]) -> bool:
        """우선순위 4 개선: 출처 신뢰도 검증 (완화된 기준)"""
        import re
        source = doc.get("source", "")
        
        # 출처가 없어도 내용이 유용하면 포함 (완화)
        if not source or len(source.strip()) < 1:
            # 출처가 없어도 통과 (경고만 출력)
            self.logger.debug(f"⚠️ [SOURCE] source 필드 없음 또는 너무 짧음: {doc.get('id', 'unknown')}")
            return True  # 출처가 없어도 통과
        
        source_stripped = source.strip()
        
        # 출처 형식 검증 (완화된 기준)
        # 기본적인 출처 형식 검증 - 더 관대한 기준
        has_valid_format = (
            any(keyword in source_stripped for keyword in ["법", "법원", "위원회", "부", "청", "원"]) or
            bool(re.match(r'[가-힣]+법', source_stripped)) or
            bool(re.match(r'.*법원.*', source_stripped)) or
            len(source_stripped) >= 2  # 최소 길이 완화: 3 → 2
        )
        
        # 형식이 맞지 않아도 통과 (경고만 출력)
        if not has_valid_format:
            self.logger.debug(f"⚠️ [SOURCE] 출처 형식이 표준과 다름: {source_stripped}")
            return True  # 형식이 맞지 않아도 통과
        
        # 메타데이터에서 출처 정보 확인
        metadata = doc.get("metadata", {})
        if isinstance(metadata, dict):
            # statute_name, case_name 등이 있으면 더 신뢰할 수 있음
            pass
        
        return True

