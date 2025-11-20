# -*- coding: utf-8 -*-
"""
검색 결과 처리 프로세서
검색 결과 병합, 가중치 적용, 필터링, 재정렬 등을 담당
"""

import logging
import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 성능 최적화: 정규식 패턴 컴파일 (모듈 레벨)
LAW_PATTERN = re.compile(r'[가-힣]+법\s*제?\s*\d+\s*조')
PRECEDENT_PATTERN = re.compile(r'대법원|법원.*\d{4}[다나마]\d+')


class SearchResultProcessor:
    """검색 결과 처리 프로세서"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, result_merger=None, result_ranker=None, weight_config=None):
        self.logger = logger or logging.getLogger(__name__)
        self.result_merger = result_merger
        self.result_ranker = result_ranker
        
        # 가중치 설정 (테스트용)
        self.weight_config = weight_config or {
            "hybrid_law": {"semantic": 0.3, "keyword": 0.7},
            "hybrid_case": {"semantic": 0.7, "keyword": 0.3},
            "hybrid_general": {"semantic": 0.5, "keyword": 0.5},
            "doc_type_boost": {"statute": 1.2, "case": 1.15},
            "quality_weight": 0.2,
            "keyword_adjustment": 1.8
        }
        
        # Phase 3: KeywordExtractor 초기화 (형태소 분석용)
        self.keyword_extractor = None
        try:
            from core.agents.keyword_extractor import KeywordExtractor
            self.keyword_extractor = KeywordExtractor(use_morphology=True, logger_instance=self.logger)
        except Exception as e:
            self.logger.debug(f"KeywordExtractor initialization failed: {e}, will use fallback matching")
    
    def merge_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        result_merger=None
    ) -> List[Dict[str, Any]]:
        """검색 결과 병합"""
        merger = result_merger or self.result_merger
        
        if merger:
            exact_results_dict = {
                "semantic": semantic_results if isinstance(semantic_results, list) else [],
                "keyword": keyword_results if isinstance(keyword_results, list) else []
            } if keyword_results else {}
            
            merged_results = merger.merge_results(
                exact_results=exact_results_dict,
                semantic_results=semantic_results if isinstance(semantic_results, list) else [],
                weights={"exact": 0.7, "semantic": 0.3}
            )
        else:
            merged_results = semantic_results + keyword_results
        
        merged_docs = []
        for merged_result in merged_results:
            if hasattr(merged_result, 'text'):
                text_value = merged_result.text
                if not text_value or len(str(text_value).strip()) == 0:
                    if hasattr(merged_result, 'content'):
                        text_value = merged_result.content
                    elif hasattr(merged_result, 'metadata') and isinstance(merged_result.metadata, dict):
                        text_value = (
                            merged_result.metadata.get('content') or
                            merged_result.metadata.get('text') or
                            merged_result.metadata.get('document') or
                            ''
                        )
                
                if not text_value or len(str(text_value).strip()) == 0:
                    source_name = getattr(merged_result, 'source', 'Unknown')
                    score_value = getattr(merged_result, 'score', 0.0)
                    self.logger.warning(f"⚠️ [DEBUG] MergedResult text가 비어있음 - source: {source_name}, score: {score_value:.3f}")
                
                merged_docs.append({
                    "content": str(text_value) if text_value else "",
                    "text": str(text_value) if text_value else "",
                    "relevance_score": getattr(merged_result, 'score', 0.0),
                    "source": getattr(merged_result, 'source', 'Unknown'),
                    "metadata": getattr(merged_result, 'metadata', {}) if hasattr(merged_result, 'metadata') else {}
                })
            elif isinstance(merged_result, dict):
                doc = merged_result.copy()
                if "content" not in doc and "text" in doc:
                    doc["content"] = doc["text"]
                elif "text" not in doc and "content" in doc:
                    doc["text"] = doc["content"]
                elif "content" not in doc and "text" not in doc:
                    doc["content"] = ""
                    doc["text"] = ""
                merged_docs.append(doc)
        
        return merged_docs
    
    def calculate_keyword_weights(
        self,
        extracted_keywords: List[str],
        query: str,
        query_type: str,
        legal_field: str
    ) -> Dict[str, float]:
        """키워드별 중요도 가중치 계산"""
        keyword_weights = {}
        
        if not extracted_keywords:
            return keyword_weights
        
        query_lower = query.lower()
        
        legal_term_patterns = [
            re.compile(r'[가-힣]+법'),
            re.compile(r'[가-힣]+규정'),
            re.compile(r'[가-힣]+조항'),
            re.compile(r'판례'),
            re.compile(r'대법원'),
            re.compile(r'법원'),
            re.compile(r'판결'),
            re.compile(r'계약'),
            re.compile(r'손해배상'),
            re.compile(r'소송'),
            re.compile(r'청구')
        ]
        
        query_type_keywords = {
            "precedent_search": ["판례", "사건", "판결", "대법원"],
            "law_inquiry": ["법률", "조문", "법령", "규정", "조항"],
            "legal_advice": ["조언", "해석", "권리", "의무", "책임"],
            "procedure_guide": ["절차", "방법", "대응", "소송"],
            "term_explanation": ["의미", "정의", "개념", "해석"]
        }
        
        field_keywords = {
            "family": ["가족", "이혼", "양육", "상속", "부부"],
            "civil": ["민사", "계약", "손해배상", "채권", "채무"],
            "criminal": ["형사", "범죄", "처벌", "형량"],
            "labor": ["노동", "근로", "해고", "임금", "근로자"],
            "corporate": ["기업", "회사", "주주", "법인"]
        }
        
        important_keywords_for_type = query_type_keywords.get(query_type, [])
        important_keywords_for_field = field_keywords.get(legal_field, [])
        
        for keyword in extracted_keywords:
            if not keyword or not isinstance(keyword, str):
                continue
            
            keyword_lower = keyword.lower()
            weight = 0.0
            
            query_frequency = query_lower.count(keyword_lower)
            query_weight = min(0.3, (query_frequency / max(1, len(query.split()))) * 0.3)
            weight += query_weight
            
            is_legal_term = any(pattern.search(keyword) for pattern in legal_term_patterns)
            if is_legal_term:
                weight += 0.3
            
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_type):
                weight += 0.2
            
            if any(imp_kw in keyword_lower for imp_kw in important_keywords_for_field):
                weight += 0.2
            
            if weight == 0.0:
                weight = 0.1
            
            keyword_weights[keyword] = min(1.0, weight)
        
        total_weight = sum(keyword_weights.values())
        if total_weight > 0:
            max_weight = max(keyword_weights.values()) if keyword_weights else 1.0
            if max_weight > 0:
                for kw in keyword_weights:
                    keyword_weights[kw] = keyword_weights[kw] / max_weight
        
        # keyword_weights 계산 결과 로깅
        if keyword_weights:
            self.logger.info(
                f"🔍 [KEYWORD WEIGHTS] 계산 완료: "
                f"총 {len(keyword_weights)}개 키워드, "
                f"총 가중치={total_weight:.3f}, "
                f"최대 가중치={max_weight:.3f}, "
                f"상위 5개={dict(list(sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True))[:5])}"
            )
        else:
            self.logger.warning(
                f"🔍 [KEYWORD WEIGHTS] keyword_weights가 비어있음: "
                f"extracted_keywords={extracted_keywords}, query='{query[:50]}'"
            )
        
        return keyword_weights
    
    def calculate_keyword_match_score(
        self,
        document: Dict[str, Any],
        keyword_weights: Dict[str, float],
        query: str
    ) -> Dict[str, float]:
        """문서에 대한 키워드 매칭 점수 계산"""
        doc_content = document.get("content", "")
        if not doc_content:
            doc_content = document.get("text", "") or document.get("content_text", "")
        
        if not doc_content:
            self.logger.debug(
                f"🔍 [KEYWORD MATCH] 문서에 content 없음: "
                f"doc_id={document.get('id', 'unknown')}, "
                f"keys={list(document.keys())[:10]}"
            )
            return {
                "keyword_match_score": 0.0,
                "keyword_coverage": 0.0,
                "matched_keywords": [],
                "weighted_keyword_score": 0.0
            }
        
        doc_content_lower = doc_content.lower()
        
        # keyword 점수 계산 디버깅 로깅
        if not keyword_weights:
            self.logger.warning(
                f"🔍 [KEYWORD MATCH] keyword_weights가 비어있음: "
                f"doc_id={document.get('id', 'unknown')}, query='{query[:50]}'"
            )
        
        matched_keywords = []
        total_weight = 0.0
        matched_weight = 0.0
        unmatched_keywords = []
        
        # 개선 #2: 법률 용어 보너스 점수를 위한 패턴 정의
        legal_term_patterns = [
            (r'제\s*\d+\s*조', 1.5),  # 조문번호 패턴
            (r'[가-힣]+법', 1.3),  # 법령명 패턴
            (r'손해배상|불법행위|계약|해지|해제', 1.2),  # 주요 법률 용어
        ]
        
        for keyword, weight in keyword_weights.items():
            if not keyword:
                continue
            
            total_weight += weight
            keyword_lower = keyword.lower()
            is_matched = False
            match_type = "none"  # none, exact, partial, compound
            
            # Phase 3: 1. 직접 문자열 매칭
            if keyword_lower in doc_content_lower:
                matched_keywords.append(keyword)
                matched_weight += weight
                is_matched = True
                match_type = "exact"
                
                keyword_count = doc_content_lower.count(keyword_lower)
                if keyword_count > 1:
                    matched_weight += weight * 0.1 * min(2, keyword_count - 1)
            else:
                unmatched_keywords.append((keyword, weight, "exact_fail"))
            
            # 우선순위 3: 부분 매칭 강화 (법령명, 조문번호 등)
            if not is_matched:
                # 법령명 부분 매칭 (예: "민법" → "민법", "민법상" 등)
                if "법" in keyword:
                    law_name = keyword.replace("법", "")
                    if law_name in doc_content_lower or keyword in doc_content_lower:
                        matched_keywords.append(keyword)
                        matched_weight += weight * 0.8  # 부분 매칭은 80% 가중치
                        is_matched = True
                        match_type = "partial"
                # 조문번호 부분 매칭 (예: "제750조" → "750조", "제 750 조" 등)
                elif "제" in keyword and "조" in keyword:
                    article_no = re.search(r'\d+', keyword)
                    if article_no and article_no.group() in doc_content_lower:
                        matched_keywords.append(keyword)
                        matched_weight += weight * 0.9  # 조문번호는 90% 가중치
                        is_matched = True
                        match_type = "partial"
            
            # Phase 3: 2. 형태소 분석 기반 부분 매칭 (직접 매칭이 없는 경우)
            if not is_matched and self.keyword_extractor:
                try:
                    # 키워드의 형태소 분석
                    keyword_morphs = self.keyword_extractor._okt.morphs(keyword) if self.keyword_extractor._okt else []
                    keyword_nouns = self.keyword_extractor._okt.nouns(keyword) if self.keyword_extractor._okt else []
                    
                    # 문서 내용의 형태소 분석 (일부만)
                    doc_sample = doc_content[:1000]  # 성능 최적화를 위해 일부만 분석
                    doc_morphs = self.keyword_extractor._okt.morphs(doc_sample) if self.keyword_extractor._okt else []
                    doc_nouns = self.keyword_extractor._okt.nouns(doc_sample) if self.keyword_extractor._okt else []
                    
                    # 형태소 기반 부분 매칭 확인
                    if keyword_morphs:
                        matched_morphs = sum(1 for morph in keyword_morphs if morph in doc_morphs)
                        morph_ratio = matched_morphs / len(keyword_morphs) if keyword_morphs else 0.0
                        
                        if morph_ratio >= 0.6:  # 60% 이상 매칭
                            matched_keywords.append(keyword)
                            matched_weight += weight * morph_ratio * 0.7  # 부분 매칭은 70% 가중치
                            is_matched = True
                            match_type = "partial"
                    
                    # 명사 기반 매칭 확인
                    if not is_matched and keyword_nouns:
                        matched_nouns = sum(1 for noun in keyword_nouns if noun in doc_nouns)
                        noun_ratio = matched_nouns / len(keyword_nouns) if keyword_nouns else 0.0
                        
                        if noun_ratio >= 0.5:  # 50% 이상 매칭
                            matched_keywords.append(keyword)
                            matched_weight += weight * noun_ratio * 0.6  # 명사 매칭은 60% 가중치
                            is_matched = True
                            match_type = "partial"
                    
                except Exception as e:
                    self.logger.debug(f"Morphological matching error for keyword '{keyword}': {e}")
            
            # Phase 3: 3. 복합어 분리 및 매칭 (직접 매칭이 없는 경우)
            if not is_matched:
                # 복합어 패턴 (예: "손해배상" → "손해", "배상")
                compound_patterns = [
                    (r'손해배상', ['손해', '배상']),
                    (r'불법행위', ['불법', '행위']),
                    (r'계약해지', ['계약', '해지']),
                    (r'계약해제', ['계약', '해제']),
                    (r'손해배상청구권', ['손해배상', '청구권', '손해', '배상']),
                ]
                
                for pattern, parts in compound_patterns:
                    if re.search(pattern, keyword):
                        # 복합어의 구성 요소가 모두 문서에 있는지 확인
                        matched_parts = sum(1 for part in parts if part in doc_content_lower)
                        part_ratio = matched_parts / len(parts) if parts else 0.0
                        
                        if part_ratio >= 0.7:  # 70% 이상 구성 요소 매칭
                            matched_keywords.append(keyword)
                            matched_weight += weight * part_ratio * 0.8  # 복합어 매칭은 80% 가중치
                            is_matched = True
                            match_type = "compound"
                            # unmatched에서 제거
                            unmatched_keywords = [u for u in unmatched_keywords if u[0] != keyword]
                            break
                        else:
                            # 복합어 매칭 실패
                            if not any(u[0] == keyword for u in unmatched_keywords):
                                unmatched_keywords.append((keyword, weight, f"compound_partial_{part_ratio:.2f}"))
            
            # 개선 #2: 법률 용어 보너스 점수 추가 (매칭된 경우에만)
            if is_matched:
                for pattern, bonus_multiplier in legal_term_patterns:
                    if re.search(pattern, keyword):
                        matched_weight += weight * (bonus_multiplier - 1.0) * 0.3
                        break
        
        keyword_coverage = len(matched_keywords) / max(1, len(keyword_weights))
        keyword_match_score = matched_weight / max(0.1, total_weight) if total_weight > 0 else 0.0
        weighted_keyword_score = min(1.0, matched_weight / max(1, len(keyword_weights)))
        
        # keyword 점수가 낮은 경우 상세 로깅
        doc_id = document.get("id") or document.get("chunk_id") or document.get("doc_id") or "unknown"
        if keyword_match_score < 0.3 and len(keyword_weights) > 0:
            self.logger.info(
                f"🔍 [KEYWORD MATCH LOW SCORE] doc_id={doc_id[:50]}, "
                f"keyword_match_score={keyword_match_score:.3f}, "
                f"weighted_keyword_score={weighted_keyword_score:.3f}, "
                f"keyword_coverage={keyword_coverage:.3f}, "
                f"matched={len(matched_keywords)}/{len(keyword_weights)}, "
                f"matched_keywords={matched_keywords[:5]}, "
                f"unmatched_count={len(unmatched_keywords)}, "
                f"total_weight={total_weight:.3f}, matched_weight={matched_weight:.3f}, "
                f"content_preview={doc_content[:100].replace(chr(10), ' ')}"
            )
            # 매칭 실패한 상위 키워드 로깅
            if unmatched_keywords:
                top_unmatched = sorted(unmatched_keywords, key=lambda x: x[1], reverse=True)[:5]
                self.logger.info(
                    f"🔍 [KEYWORD MATCH] 매칭 실패한 상위 키워드: "
                    f"{[(kw, f'weight={w:.3f}', reason) for kw, w, reason in top_unmatched]}"
                )
        
        return {
            "keyword_match_score": keyword_match_score,
            "keyword_coverage": keyword_coverage,
            "matched_keywords": matched_keywords,
            "weighted_keyword_score": weighted_keyword_score
        }
    
    def calculate_weighted_final_score(
        self,
        document: Dict[str, Any],
        keyword_scores: Dict[str, float],
        search_params: Dict[str, Any],
        query_type: Optional[str] = None
    ) -> float:
        """개선된 가중치 계산: 단순화 및 LIKE 검색 반영"""
        
        # 1. 기본 점수 추출
        base_relevance = (
            document.get("relevance_score", 0.0) or
            document.get("similarity", 0.0) or
            document.get("combined_score", 0.0) or
            document.get("score", 0.0) or
            0.0
        )
        
        keyword_match = keyword_scores.get("weighted_keyword_score", 0.0)
        keyword_coverage = keyword_scores.get("keyword_coverage", 0.0)
        
        search_type = document.get("search_type", "")
        doc_type = document.get("type", "").lower() if document.get("type") else ""
        source_type = document.get("source_type", "").lower() if document.get("source_type") else ""
        
        # 2. LIKE 검색 점수 보정
        keyword_match_normalized = self._adjust_keyword_score_for_like_search(
            keyword_score=keyword_match,
            search_type=search_type
        )
        
        # 3. 질문 유형별 가중치 (동적)
        dynamic_weights = self.calculate_dynamic_weights(
            query_type=query_type,
            search_quality=search_params.get("overall_quality", 0.7),
            document_count=search_params.get("document_count", 10)
        )
        
        # 4. 하이브리드 점수 계산 (가중치 설정 사용)
        if query_type == "law_inquiry" or query_type == "statute":
            # 법령 질문: 가중치 설정 사용
            hybrid_score = (
                self.weight_config["hybrid_law"]["semantic"] * base_relevance + 
                self.weight_config["hybrid_law"]["keyword"] * keyword_match_normalized
            )
        elif query_type == "precedent_search" or query_type == "case":
            # 판례 질문: 가중치 설정 사용
            hybrid_score = (
                self.weight_config["hybrid_case"]["semantic"] * base_relevance + 
                self.weight_config["hybrid_case"]["keyword"] * keyword_match_normalized
            )
        else:
            # 일반 질문: 가중치 설정 사용
            hybrid_score = (
                self.weight_config["hybrid_general"]["semantic"] * base_relevance + 
                self.weight_config["hybrid_general"]["keyword"] * keyword_match_normalized
            )
        
        # 5. 문서 타입별 부스팅 (가중치 설정 사용)
        is_statute_article = (
            doc_type == "statute_article" or 
            source_type == "statute_article" or
            "statute_article" in doc_type or
            "statute_article" in source_type or
            document.get("direct_match", False) or
            document.get("search_type") == "direct_statute"
        )
        
        doc_type_boost = 1.0
        if is_statute_article:
            doc_type_boost = self.weight_config["doc_type_boost"]["statute"] if (query_type == "law_inquiry" or query_type == "statute") else 1.1
        elif "case" in doc_type or "precedent" in doc_type or "case_paragraph" in doc_type:
            doc_type_boost = self.weight_config["doc_type_boost"]["case"] if (query_type == "precedent_search" or query_type == "case") else 1.05
        
        # 6. 문서 품질 점수 추가
        content = document.get("content", "") or document.get("text", "")
        quality_score = self._calculate_document_quality_score(
            content=content,
            keyword_coverage=keyword_coverage,
            doc_type=doc_type
        )
        
        # 7. 최종 점수 계산 (가중치 설정 사용)
        quality_weight = self.weight_config["quality_weight"]
        base_weight = 1.0 - quality_weight
        final_score = (
            hybrid_score * doc_type_boost * base_weight +  # 기본 점수
            quality_score * quality_weight  # 품질 점수
        )
        
        # 8. 점수 범위 제한 및 안정화
        final_score = self._normalize_score(final_score, min_val=0.0, max_val=1.0)
        
        # 9. 최소 점수 보정 (법령 조문 등)
        if is_statute_article and final_score < 0.3:
            final_score = max(0.3, final_score)
        
        return final_score
    
    def _adjust_keyword_score_for_like_search(
        self,
        keyword_score: float,
        search_type: str
    ) -> float:
        """LIKE 검색 점수 보정"""
        if search_type == "keyword" or search_type == "text2sql":
            # LIKE 검색은 최대 0.5이므로, 이를 0.0~1.0 범위로 확장
            # 가중치 설정의 조정값 사용
            adjustment = self.weight_config.get("keyword_adjustment", 1.8)
            adjusted = min(1.0, keyword_score * adjustment)  # 0.5 -> 0.9 (기본값)
            return adjusted
        return keyword_score
    
    def _calculate_document_quality_score(
        self,
        content: str,
        keyword_coverage: float,
        doc_type: str
    ) -> float:
        """문서 품질 점수 계산"""
        if not content:
            return 0.0
        
        quality = 0.0
        
        # 1. 적절한 길이 (50~500자: 최적)
        content_length = len(content)
        if 50 <= content_length <= 500:
            quality += 0.4
        elif 500 < content_length <= 1000:
            quality += 0.3
        elif content_length > 1000:
            quality += 0.2
        else:
            quality += 0.1
        
        # 2. 키워드 커버리지
        quality += keyword_coverage * 0.3
        
        # 3. 문서 타입 적합성
        if doc_type in ["statute_article", "case_paragraph", "decision_paragraph", "interpretation_paragraph"]:
            quality += 0.3
        else:
            quality += 0.2
        
        return min(1.0, quality)
    
    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """점수 정규화 (안정화)"""
        if score < min_val:
            return min_val
        elif score > max_val:
            # 초과 점수는 로그 스케일로 완화
            excess = score - max_val
            return max_val + (excess / (1.0 + excess * 10))
        return score
    
    def calculate_dynamic_weights(
        self,
        query_type: str = "",
        search_quality: float = 0.7,
        document_count: int = 10
    ) -> Dict[str, float]:
        """개선된 동적 가중치 계산"""
        base_weights = {
            "relevance": 0.50,  # 증가 (0.40 -> 0.50)
            "keyword": 0.30,    # 감소 (0.35 -> 0.30, LIKE 검색 반영)
            "quality": 0.15,   # 신규 추가
            "type": 0.05       # 감소 (0.15 -> 0.05)
        }
        
        # 질문 유형별 조정
        if query_type == "law_inquiry" or query_type == "statute":
            base_weights["keyword"] = 0.40  # 법령 질문: keyword 중요
            base_weights["relevance"] = 0.35
        elif query_type == "precedent_search" or query_type == "case":
            base_weights["relevance"] = 0.60  # 판례 질문: semantic 중요
            base_weights["keyword"] = 0.20
        
        # 검색 품질에 따른 조정
        if search_quality < 0.5:
            base_weights["keyword"] += 0.1
            base_weights["relevance"] -= 0.1
        elif search_quality > 0.8:
            base_weights["relevance"] += 0.1
            base_weights["keyword"] -= 0.1
        
        # 문서 수에 따른 조정
        if document_count < 5:
            base_weights["relevance"] += 0.05
            base_weights["keyword"] -= 0.05
        
        # 정규화
        total = sum(base_weights.values())
        if total > 0:
            base_weights = {k: v / total for k, v in base_weights.items()}
        
        return base_weights
    
    def apply_keyword_weights_to_docs(
        self,
        merged_docs: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        query: str,
        query_type_str: str,
        search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """키워드 가중치 적용"""
        def process_doc(doc):
            doc_content = doc.get("content", "") or doc.get("text", "")
            if not doc_content or not isinstance(doc_content, str) or len(doc_content.strip()) < 5:
                doc["keyword_match_score"] = 0.0
                doc["keyword_coverage"] = 0.0
                doc["matched_keywords"] = []
                doc["weighted_keyword_score"] = 0.0
                doc["final_weighted_score"] = doc.get("relevance_score", 0.0) * 0.5
                return doc
            
            keyword_scores = self.calculate_keyword_match_score(
                document=doc,
                keyword_weights=keyword_weights,
                query=query
            )
            
            final_score = self.calculate_weighted_final_score(
                document=doc,
                keyword_scores=keyword_scores,
                search_params=search_params,
                query_type=query_type_str
            )
            
            doc["keyword_match_score"] = keyword_scores.get("keyword_match_score", 0.0)
            doc["keyword_coverage"] = keyword_scores.get("keyword_coverage", 0.0)
            doc["matched_keywords"] = keyword_scores.get("matched_keywords", [])
            doc["weighted_keyword_score"] = keyword_scores.get("weighted_keyword_score", 0.0)
            doc["final_weighted_score"] = final_score
            
            return doc
        
        if len(merged_docs) > 10:
            weighted_docs = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_doc, doc): doc for doc in merged_docs}
                for future in as_completed(futures):
                    try:
                        weighted_docs.append(future.result(timeout=2))
                    except Exception as e:
                        self.logger.warning(f"Document processing failed: {e}")
                        weighted_docs.append(futures[future])
        else:
            weighted_docs = [process_doc(doc) for doc in merged_docs]
        
        weighted_docs.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        return weighted_docs
    
    def apply_citation_boost(
        self,
        weighted_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Citation 부스트 적용"""
        citation_boosted = []
        non_citation = []
        
        for doc in weighted_docs:
            content = doc.get("content", "") or doc.get("text", "") or ""
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if len(content) < 50:
                non_citation.append(doc)
                continue
            
            content_sample = content[:500]
            has_law = bool(LAW_PATTERN.search(content_sample))
            has_precedent = bool(PRECEDENT_PATTERN.search(content_sample))
            
            if has_law or has_precedent:
                current_score = doc.get("final_weighted_score", doc.get("relevance_score", 0.0))
                boosted_score = current_score * 1.2
                doc["final_weighted_score"] = boosted_score
                doc["relevance_score"] = boosted_score
                citation_boosted.append(doc)
            else:
                non_citation.append(doc)
        
        citation_boosted.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        non_citation.sort(key=lambda x: x.get("final_weighted_score", x.get("relevance_score", 0.0)), reverse=True)
        
        if citation_boosted:
            self.logger.info(f"🔍 [SEARCH FILTERING] Citation boost applied: {len(citation_boosted)} documents with citations prioritized")
        
        return citation_boosted + non_citation
    
    def filter_documents(
        self,
        weighted_docs: List[Dict[str, Any]],
        max_docs: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """문서 필터링"""
        debug_mode = os.getenv("DEBUG_SEARCH_RESULTS", "false").lower() == "true"
        
        filtered_docs = []
        skipped_content = 0
        skipped_score = 0
        skipped_content_details = []
        
        for doc in weighted_docs:
            content = (
                doc.get("content", "") or
                doc.get("text", "") or
                doc.get("content_text", "") or
                doc.get("document", "") or
                str(doc.get("metadata", {}).get("content", "")) or
                str(doc.get("metadata", {}).get("text", "")) or
                ""
            )
            
            if not isinstance(content, str):
                content = str(content) if content else ""
            
            if not content or len(content.strip()) < 5:
                skipped_content += 1
                if skipped_content <= 3:
                    skipped_content_details.append({
                        "keys": list(doc.keys()),
                        "content_type": type(doc.get("content", None)).__name__,
                        "text_type": type(doc.get("text", None)).__name__,
                        "content_len": len(str(doc.get("content", ""))),
                        "text_len": len(str(doc.get("text", "")))
                    })
                continue
            
            score = doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0)
            if score < 0.05:
                skipped_score += 1
                continue
            
            filtered_docs.append(doc)
        
        if debug_mode:
            self.logger.info(f"📊 [SEARCH RESULTS] Filtering statistics - Weighted: {len(weighted_docs)}, Filtered: {len(filtered_docs)}, Skipped (content): {skipped_content}, Skipped (score): {skipped_score}")
            
            if skipped_content > 0 and skipped_content_details:
                self.logger.warning(f"⚠️ [SEARCH RESULTS] Content 필터링 제외 상세 (상위 {len(skipped_content_details)}개): {skipped_content_details}")
        
        final_docs = filtered_docs[:max_docs]
        
        return final_docs, {
            "skipped_content": skipped_content,
            "skipped_score": skipped_score,
            "filtered_count": len(filtered_docs)
        }
    
    def rerank_with_keyword_weights(
        self,
        results: List[Dict[str, Any]],
        keyword_weights: Dict[str, float],
        rerank_params: Dict[str, Any],
        result_ranker=None
    ) -> List[Dict[str, Any]]:
        """키워드 가중치를 적용한 Reranking"""
        ranker = result_ranker or self.result_ranker
        
        try:
            sorted_results = sorted(
                results,
                key=lambda x: (
                    x.get("final_weighted_score", 0.0),
                    x.get("keyword_match_score", 0.0),
                    x.get("keyword_coverage", 0.0)
                ),
                reverse=True
            )
            
            for doc in sorted_results:
                coverage = doc.get("keyword_coverage", 0.0)
                if coverage > 0.7:
                    doc["final_weighted_score"] *= 1.1
                elif coverage > 0.5:
                    doc["final_weighted_score"] *= 1.05
            
            sorted_results = sorted(
                sorted_results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )
            
            top_k = rerank_params.get("top_k", 20)
            if ranker and len(sorted_results) > 0:
                try:
                    reranked_results = ranker.rank_results(
                        sorted_results[:top_k * 2],
                        top_k=top_k
                    )
                    if reranked_results and hasattr(reranked_results[0], 'score'):
                        reranked_dicts = []
                        for result in reranked_results:
                            doc = {
                                "content": result.text,
                                "relevance_score": result.score,
                                "source": result.source,
                                "id": f"{result.source}_{hash(result.text)}",
                                "final_weighted_score": result.score
                            }
                            if isinstance(result.metadata, dict):
                                doc.update(result.metadata)
                            reranked_dicts.append(doc)
                        sorted_results = reranked_dicts[:top_k]
                    else:
                        sorted_results = sorted_results[:top_k]
                except Exception as e:
                    self.logger.warning(f"Reranker failed, using keyword-weighted scores: {e}")
                    sorted_results = sorted_results[:top_k]
            else:
                sorted_results = sorted_results[:top_k]
            
            try:
                if ranker and hasattr(ranker, 'apply_diversity_filter'):
                    diverse_results = ranker.apply_diversity_filter(
                        sorted_results,
                        max_per_type=5,
                        diversity_weight=rerank_params.get("diversity_weight", 0.3)
                    )
                else:
                    diverse_results = sorted_results
            except Exception as e:
                self.logger.warning(f"Diversity filter failed: {e}")
                diverse_results = sorted_results
            
            return diverse_results
        
        except Exception as e:
            self.logger.warning(f"Reranking with keyword weights failed: {e}")
            return sorted(
                results,
                key=lambda x: x.get("final_weighted_score", 0.0),
                reverse=True
            )[:rerank_params.get("top_k", 20)]

