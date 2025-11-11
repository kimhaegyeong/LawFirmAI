# -*- coding: utf-8 -*-
"""
Search Service (ML Enhanced)
검색 서비스 - ML 강화 버전
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.model_manager import LegalModelManager

from ..data.database import DatabaseManager
from ..data.vector_store import LegalVectorStore as VectorStore
from ..utils.config import Config
from ..services.ai_keyword_generator import AIKeywordGenerator

logger = logging.getLogger(__name__)


class MLEnhancedSearchService:
    """ML 강화 검색 서비스 클래스"""
    
    def __init__(self, config: Config, database: DatabaseManager, 
                 vector_store: VectorStore, model_manager: Optional["LegalModelManager"] = None):
        """ML 강화 검색 서비스 초기화"""
        self.config = config
        self.database = database
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # LLM 기반 키워드 확장기 초기화
        self.ai_keyword_generator = AIKeywordGenerator()
        self._keyword_cache = {}  # 키워드 확장 캐시
        self._max_cache_size = 100  # 최대 캐시 크기 (100개)
        
        # ML 강화 검색 설정
        self.use_ml_enhanced_search = True
        self.quality_threshold = 0.7
        self.confidence_threshold = 0.6  # 신뢰도 60% 미만 필터링
        self.supplementary_weight = 0.8
        self.hybrid_weights = {
            'semantic': 0.6,
            'keyword': 0.3,
            'ml_quality': 0.1
        }
        
        self.logger.info("MLEnhancedSearchService initialized")
    
    def search_documents(self, query: str, search_type: str = "hybrid", 
                        limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 문서 검색"""
        try:
            if search_type == "semantic":
                return self._ml_enhanced_semantic_search(query, limit, filters)
            elif search_type == "keyword":
                return self._ml_enhanced_keyword_search(query, limit, filters)
            elif search_type == "hybrid":
                return self._ml_enhanced_hybrid_search(query, limit, filters)
            elif search_type == "supplementary":
                return self._search_supplementary_provisions(query, limit)
            elif search_type == "high_quality":
                return self._search_high_quality_documents(query, limit)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def _ml_enhanced_semantic_search(self, query: str, limit: int, 
                                   filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 의미적 검색"""
        try:
            # 기본 필터 설정
            search_filters = filters or {}
            if self.use_ml_enhanced_search:
                search_filters['ml_enhanced'] = True
            
            # 벡터 저장소에서 검색
            similar_docs = self.vector_store.search(query, limit * 2, search_filters)
            
            # ML 강화 결과 필터링 및 스코어링
            filtered_docs = self._filter_and_score_documents(similar_docs, query)
            
            # 결과 포맷팅
            results = []
            for doc in filtered_docs[:limit]:
                metadata = doc.get("metadata", {})
                result = {
                    "document_id": metadata.get("document_id"),
                    "title": metadata.get("law_name", ""),
                    "content": doc.get("text", ""),
                    "similarity": doc.get("score", 0.0),
                    "search_type": "semantic",
                    "article_number": metadata.get("article_number", ""),
                    "article_title": metadata.get("article_title", ""),
                    "article_type": metadata.get("article_type", "main"),
                    "is_supplementary": metadata.get("is_supplementary", False),
                    "ml_confidence_score": metadata.get("ml_confidence_score"),
                    "parsing_method": metadata.get("parsing_method", "ml_enhanced"),
                    "quality_score": metadata.get("parsing_quality_score", 0.0),
                    "chunk_index": metadata.get("chunk_id", 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced semantic search: {e}")
            return []
    
    def _ml_enhanced_keyword_search(self, query: str, limit: int, 
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 키워드 검색 (LLM 기반 키워드 확장)"""
        try:
            # 1. 기본 키워드 추출
            base_keywords = self._extract_keywords(query)
            if not base_keywords:
                return []
            
            # 2. LLM 키워드 확장 (동기 래퍼 사용)
            expanded_keywords = base_keywords
            try:
                # 기존 이벤트 루프 확인
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 이미 실행 중인 이벤트 루프가 있으면 새 이벤트 루프 생성
                        expanded_keywords = asyncio.run(
                            self._expand_keywords_with_llm(query, base_keywords)
                        )
                    else:
                        # 실행 중인 이벤트 루프가 없으면 직접 실행
                        expanded_keywords = loop.run_until_complete(
                            self._expand_keywords_with_llm(query, base_keywords)
                        )
                except RuntimeError:
                    # 이벤트 루프가 없으면 새로 생성
                    expanded_keywords = asyncio.run(
                        self._expand_keywords_with_llm(query, base_keywords)
                    )
            except Exception as e:
                self.logger.warning(f"LLM 키워드 확장 실패, 기본 키워드 사용: {e}")
                expanded_keywords = base_keywords
            
            # 확장된 키워드가 없으면 기본 키워드 사용
            if not expanded_keywords:
                expanded_keywords = base_keywords
            
            # 3. FTS5 쿼리 생성 (확장된 키워드 사용)
            fts_query = self._make_fts5_query(expanded_keywords)
            
            # FTS5 테이블 존재 여부 확인 (lawfirm_v2.db의 statute_articles_fts 사용)
            table_exists = self._check_fts5_table_exists("statute_articles_fts")
            
            if table_exists:
                # FTS5를 사용한 검색 쿼리 (lawfirm_v2.db 스키마)
                where_clause = ""
                
                sql = f"""
                    SELECT 
                        sa.id,
                        sa.statute_id,
                        sa.article_no,
                        sa.clause_no,
                        sa.item_no,
                        sa.heading,
                        sa.text,
                        s.name as statute_name,
                        s.abbrv as statute_abbrv,
                        s.statute_type,
                        s.category,
                        bm25(statute_articles_fts) as rank_score
                    FROM statute_articles_fts
                    JOIN statute_articles sa ON statute_articles_fts.rowid = sa.id
                    JOIN statutes s ON sa.statute_id = s.id
                    WHERE statute_articles_fts MATCH ? {where_clause}
                    ORDER BY rank_score
                    LIMIT ?
                """
                
                params = [fts_query, limit]
            else:
                # FTS5 테이블이 없으면 폴백
                self.logger.warning("statute_articles_fts 테이블이 없습니다. 폴백 검색을 사용합니다.")
                return self._fallback_keyword_search(query, limit, filters)
            
            # 데이터베이스에서 검색
            rows = self.database.execute_query(sql, tuple(params))
            
            # 결과 포맷팅 (lawfirm_v2.db 스키마)
            results = []
            for row in rows:
                statute_name = row.get("statute_name", "")
                text_content = row.get("text", "")
                
                matched_keywords = self._find_matched_keywords(
                    f"{statute_name} {text_content}", expanded_keywords
                )
                
                # BM25 rank_score를 관련성 점수로 변환 (개선된 정규화)
                rank_score = row.get('rank_score', -100.0)
                relevance_score = self._normalize_bm25_score(rank_score)
                
                # 품질 점수 계산
                quality_score = self._calculate_quality_score(
                    text_content, statute_name, row.get("heading", ""), expanded_keywords
                )
                
                # article_no, clause_no, item_no 조합
                article_no = row.get("article_no", "")
                clause_no = row.get("clause_no", "")
                item_no = row.get("item_no", "")
                
                article_number = article_no
                if clause_no:
                    article_number += f" 제{clause_no}항"
                if item_no:
                    article_number += f" 제{item_no}호"
                
                # 최종 점수 계산 (관련성 + 품질)
                final_score = relevance_score * 0.7 + quality_score * 0.3
                
                result = {
                    "document_id": f"statute_{row['statute_id']}_article_{row['id']}",
                    "title": statute_name,
                    "content": text_content,
                    "article_number": article_number,
                    "article_title": row.get("heading", ""),
                    "article_type": "main",
                    "is_supplementary": False,
                    "ml_confidence_score": 0.0,
                    "parsing_method": "rule_based",
                    "quality_score": quality_score,
                    "word_count": len(text_content.split()) if text_content else 0,
                    "char_count": len(text_content) if text_content else 0,
                    "search_type": "keyword",
                    "matched_keywords": matched_keywords,
                    "relevance_score": relevance_score,
                    "score": final_score,
                    "metadata": {
                        "statute_id": row.get("statute_id"),
                        "statute_abbrv": row.get("statute_abbrv"),
                        "statute_type": row.get("statute_type"),
                        "category": row.get("category"),
                    }
                }
                results.append(result)
            
            # 최소 결과 수 보장 (결과가 부족하면 폴백 검색 추가)
            if len(results) < 3:
                self.logger.warning(f"검색 결과가 부족합니다 ({len(results)}개). 폴백 검색을 시도합니다.")
                fallback_results = self._fallback_keyword_search(query, limit - len(results), filters)
                # 중복 제거
                seen_ids = {r['document_id'] for r in results}
                for fallback_result in fallback_results:
                    if fallback_result['document_id'] not in seen_ids:
                        results.append(fallback_result)
                        seen_ids.add(fallback_result['document_id'])
            
            # 점수 기준 정렬
            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced keyword search: {e}")
            # FTS5 실패 시 기존 LIKE 방식으로 폴백
            return self._fallback_keyword_search(query, limit, filters)
    
    def _make_fts5_query(self, keywords: List[str]) -> str:
        """FTS5 쿼리 생성 (개선된 버전 - 필수/선택 키워드 구분, 와일드카드 지원)"""
        if not keywords:
            return '""'
        
        # 키워드 정리 (1글자 제외, 불용어 제거)
        stopwords = {'의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', 
                     '와', '과', '도', '만', '부터', '까지', '에', '대해', '관련', '질문'}
        filtered_keywords = [kw for kw in keywords if len(kw) > 1 and kw not in stopwords]
        
        if not filtered_keywords:
            filtered_keywords = keywords[:5]
        
        # 중요 키워드와 일반 키워드 구분 (길이 기준)
        important_keywords = [kw for kw in filtered_keywords if len(kw) >= 3]
        
        # 중요 키워드 우선 사용
        if important_keywords:
            selected_keywords = important_keywords[:5]
        else:
            selected_keywords = filtered_keywords[:5]
        
        # FTS5 특수 문자 이스케이핑
        safe_keywords = []
        for kw in selected_keywords:
            safe_kw = kw.replace('"', '""')
            # 와일드카드 추가 (부분 매칭 지원) - 단, 너무 짧은 키워드는 와일드카드 제거
            if len(kw) >= 2:
                safe_keywords.append(f'"{safe_kw}"*')
            else:
                safe_keywords.append(f'"{safe_kw}"')
        
        if len(safe_keywords) == 1:
            return safe_keywords[0]
        else:
            return ' OR '.join(safe_keywords)
    
    
    def _normalize_bm25_score(self, rank_score: float) -> float:
        """BM25 점수를 0-1 범위의 관련성 점수로 정규화 (개선된 버전)"""
        import math
        
        if rank_score is None:
            return 0.5
        
        # BM25 rank_score는 음수이므로 절댓값 사용
        abs_score = abs(rank_score)
        
        # 로그 스케일 정규화 (더 나은 점수 분포)
        if abs_score <= 1:
            return 1.0
        elif abs_score <= 5:
            return 0.9
        elif abs_score <= 10:
            return 0.85
        elif abs_score <= 20:
            return 0.75
        elif abs_score <= 50:
            return 0.65
        elif abs_score <= 100:
            return 0.5
        else:
            # 로그 스케일 정규화 (큰 값에 대해서도 적절한 점수 부여)
            log_score = math.log1p(abs_score / 10.0) / math.log1p(100.0)
            return max(0.3, min(1.0, 1.0 - log_score * 0.5))
    
    def _calculate_keyword_relevance_score(self, content: str, title: str, heading: str, keywords: List[str]) -> float:
        """키워드 기반 관련성 점수 계산 (폴백 검색용)"""
        try:
            score = 0.0
            
            if not keywords or not content:
                return 0.3
            
            content_lower = content.lower()
            title_text = f"{title} {heading}".lower()
            
            # 키워드 매칭 점수
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            keyword_ratio = keyword_matches / len(keywords) if keywords else 0.0
            
            # 제목 매칭 점수 (가중치 높음)
            title_matches = sum(1 for kw in keywords if kw.lower() in title_text)
            title_ratio = title_matches / len(keywords) if keywords else 0.0
            
            # 관련성 점수 계산
            score = keyword_ratio * 0.6 + title_ratio * 0.4
            
            # 최소 점수 보장
            return max(0.3, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating keyword relevance score: {e}")
            return 0.3
    
    def _calculate_quality_score(self, content: str, title: str, heading: str, keywords: List[str]) -> float:
        """품질 점수 계산 (문서 길이, 키워드 밀도, 제목 매칭 등)"""
        try:
            score = 0.0
            
            # 1. 문서 길이 점수 (적절한 길이에 가중치)
            content_length = len(content) if content else 0
            if content_length > 0:
                if 50 <= content_length <= 500:
                    score += 0.3  # 적절한 길이
                elif content_length > 500:
                    score += 0.2  # 긴 문서
                else:
                    score += 0.1  # 짧은 문서
            
            # 2. 키워드 밀도 점수
            if keywords and content:
                content_lower = content.lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
                keyword_density = keyword_matches / len(keywords) if keywords else 0.0
                score += keyword_density * 0.3
            
            # 3. 제목/헤딩 매칭 점수
            if keywords:
                title_text = f"{title} {heading}".lower()
                title_matches = sum(1 for kw in keywords if kw.lower() in title_text)
                if title_matches > 0:
                    title_score = title_matches / len(keywords)
                    score += title_score * 0.2
            
            # 4. 법률 조항 패턴 매칭 점수
            if content:
                import re
                legal_patterns = [
                    r'[가-힣]+법\s*제?\s*\d+\s*조',
                    r'제\d+조',
                    r'제\d+항',
                    r'제\d+호'
                ]
                pattern_matches = sum(1 for pattern in legal_patterns if re.search(pattern, content))
                if pattern_matches > 0:
                    score += min(0.2, pattern_matches * 0.05)
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _check_fts5_table_exists(self, table_name: str) -> bool:
        """FTS5 테이블 존재 여부 확인 (lawfirm_v2.db)"""
        try:
            check_sql = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """
            result = self.database.execute_query(check_sql, (table_name,))
            return len(result) > 0
        except Exception as e:
            self.logger.warning(f"FTS5 테이블 확인 실패: {e}")
            return False
    
    def _fallback_keyword_search(self, query: str, limit: int, 
                                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """FTS5 실패 시 기존 LIKE 방식으로 폴백 (lawfirm_v2.db)"""
        try:
            # 기본 키워드 추출
            base_keywords = self._extract_keywords(query)
            if not base_keywords:
                return []
            
            # LLM 키워드 확장 시도 (동기 래퍼 사용)
            expanded_keywords = base_keywords
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        expanded_keywords = asyncio.run(
                            self._expand_keywords_with_llm(query, base_keywords)
                        )
                    else:
                        expanded_keywords = loop.run_until_complete(
                            self._expand_keywords_with_llm(query, base_keywords)
                        )
                except RuntimeError:
                    expanded_keywords = asyncio.run(
                        self._expand_keywords_with_llm(query, base_keywords)
                    )
            except Exception as e:
                self.logger.warning(f"폴백 검색에서 LLM 키워드 확장 실패, 기본 키워드 사용: {e}")
                expanded_keywords = base_keywords
            
            if not expanded_keywords:
                expanded_keywords = base_keywords
            
            keyword_conditions = []
            params = []
            
            for keyword in expanded_keywords:
                keyword_conditions.append("(s.name LIKE ? OR sa.text LIKE ? OR sa.heading LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = " OR ".join(keyword_conditions)
            
            sql = f"""
                SELECT 
                    sa.id,
                    sa.statute_id,
                    sa.article_no,
                    sa.clause_no,
                    sa.item_no,
                    sa.heading,
                    sa.text,
                    s.name as statute_name,
                    s.abbrv as statute_abbrv,
                    s.statute_type,
                    s.category
                FROM statute_articles sa
                JOIN statutes s ON sa.statute_id = s.id
                WHERE {where_clause}
                ORDER BY sa.id DESC
                LIMIT ?
            """
            params.append(limit)
            
            rows = self.database.execute_query(sql, tuple(params))
            
            results = []
            for row in rows:
                statute_name = row.get("statute_name", "")
                text_content = row.get("text", "")
                
                matched_keywords = self._find_matched_keywords(
                    f"{statute_name} {text_content}", expanded_keywords
                )
                
                # 관련성 점수 계산 (키워드 매칭 기반)
                relevance_score = self._calculate_keyword_relevance_score(
                    text_content, statute_name, row.get("heading", ""), expanded_keywords
                )
                
                # 품질 점수 계산
                quality_score = self._calculate_quality_score(
                    text_content, statute_name, row.get("heading", ""), expanded_keywords
                )
                
                # article_no, clause_no, item_no 조합
                article_no = row.get("article_no", "")
                clause_no = row.get("clause_no", "")
                item_no = row.get("item_no", "")
                
                article_number = article_no
                if clause_no:
                    article_number += f" 제{clause_no}항"
                if item_no:
                    article_number += f" 제{item_no}호"
                
                # 최종 점수 계산 (관련성 + 품질)
                final_score = relevance_score * 0.7 + quality_score * 0.3
                
                result = {
                    "document_id": f"statute_{row['statute_id']}_article_{row['id']}",
                    "title": statute_name,
                    "content": text_content,
                    "article_number": article_number,
                    "article_title": row.get("heading", ""),
                    "article_type": "main",
                    "is_supplementary": False,
                    "ml_confidence_score": 0.0,
                    "parsing_method": "rule_based",
                    "quality_score": quality_score,
                    "word_count": len(text_content.split()) if text_content else 0,
                    "char_count": len(text_content) if text_content else 0,
                    "search_type": "keyword",
                    "matched_keywords": matched_keywords,
                    "relevance_score": relevance_score,
                    "score": final_score,
                    "metadata": {
                        "statute_id": row.get("statute_id"),
                        "statute_abbrv": row.get("statute_abbrv"),
                        "statute_type": row.get("statute_type"),
                        "category": row.get("category"),
                    }
                }
                results.append(result)
            
            # 점수 기준 정렬
            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in fallback keyword search: {e}")
            return []
    
    def _ml_enhanced_hybrid_search(self, query: str, limit: int, 
                                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ML 강화 하이브리드 검색"""
        try:
            # 의미적 검색
            semantic_results = self._ml_enhanced_semantic_search(query, limit // 2, filters)
            
            # 키워드 검색
            keyword_results = self._ml_enhanced_keyword_search(query, limit // 2, filters)
            
            # 결과 병합 및 하이브리드 스코어링
            all_results = self._merge_and_score_results(semantic_results, keyword_results)
            
            # 중복 제거 (document_id 기준)
            seen_ids = set()
            unique_results = []
            
            for result in all_results:
                doc_id = result.get("document_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
            
            # 하이브리드 스코어 기준 정렬
            unique_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            return unique_results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error in ML-enhanced hybrid search: {e}")
            return []
    
    def _merge_and_score_results(self, semantic_results: List[Dict[str, Any]], 
                               keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """결과 병합 및 하이브리드 스코어링"""
        try:
            all_results = []
            
            # 의미적 검색 결과에 하이브리드 스코어 추가
            for result in semantic_results:
                semantic_score = result.get("similarity", 0.0)
                quality_score = result.get("quality_score", 0.0)
                ml_confidence = result.get("ml_confidence_score", 0.5)
                
                # 하이브리드 스코어 계산
                hybrid_score = (
                    self.hybrid_weights['semantic'] * semantic_score +
                    self.hybrid_weights['ml_quality'] * quality_score +
                    self.hybrid_weights['keyword'] * ml_confidence
                )
                
                result["hybrid_score"] = hybrid_score
                result["search_method"] = "semantic"
                all_results.append(result)
            
            # 키워드 검색 결과에 하이브리드 스코어 추가
            for result in keyword_results:
                keyword_score = len(result.get("matched_keywords", [])) / 10.0  # 키워드 매칭 비율
                quality_score = result.get("quality_score", 0.0)
                ml_confidence = result.get("ml_confidence_score", 0.5)
                
                # 하이브리드 스코어 계산
                hybrid_score = (
                    self.hybrid_weights['keyword'] * keyword_score +
                    self.hybrid_weights['ml_quality'] * quality_score +
                    self.hybrid_weights['semantic'] * ml_confidence
                )
                
                result["hybrid_score"] = hybrid_score
                result["search_method"] = "keyword"
                all_results.append(result)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error merging and scoring results: {e}")
            return semantic_results + keyword_results
    
    def _filter_and_score_documents(self, documents: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """ML 강화 문서 필터링 및 스코어링"""
        try:
            filtered_docs = []
            filtered_count = 0
            
            for doc in documents:
                metadata = doc.get("metadata", {})
                
                # 품질 필터링
                quality_score = metadata.get("parsing_quality_score", 0.0)
                if quality_score < self.quality_threshold:
                    filtered_count += 1
                    continue
                
                # 신뢰도 필터링 (60% 미만 제외)
                ml_confidence = metadata.get("ml_confidence_score")
                if ml_confidence is not None and ml_confidence < self.confidence_threshold:
                    filtered_count += 1
                    self.logger.debug(f"Filtered document with confidence {ml_confidence:.2f} < {self.confidence_threshold}")
                    continue
                
                # 부칙 가중치 적용
                is_supplementary = metadata.get("is_supplementary", False)
                if is_supplementary:
                    doc["score"] *= self.supplementary_weight
                
                # ML 신뢰도 점수 고려
                if ml_confidence is not None:
                    doc["score"] *= (0.5 + 0.5 * ml_confidence)
                
                filtered_docs.append(doc)
            
            if filtered_count > 0:
                self.logger.info(f"Filtered out {filtered_count} documents with low confidence or quality")
            
            # 스코어 기준으로 정렬
            filtered_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Error filtering documents: {e}")
            return documents
    
    def _search_supplementary_provisions(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """부칙 조문 전용 검색"""
        try:
            filters = {"is_supplementary": True}
            return self._ml_enhanced_hybrid_search(query, limit, filters)
            
        except Exception as e:
            self.logger.error(f"Error searching supplementary provisions: {e}")
            return []
    
    def _search_high_quality_documents(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """고품질 문서 전용 검색"""
        try:
            # 임시로 품질 임계값을 높게 설정
            original_threshold = self.quality_threshold
            self.quality_threshold = 0.9
            
            results = self._ml_enhanced_hybrid_search(query, limit)
            
            # 원래 임계값 복원
            self.quality_threshold = original_threshold
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching high quality documents: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """기본 키워드 추출 (조사 제거, 불용어 제거만 수행)"""
        try:
            # 불용어 제거 (확장된 목록)
            stopwords = {
                '의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', 
                '와', '과', '도', '만', '부터', '까지', '에', '대해', '관련', '질문',
                '어떻게', '무엇', '언제', '어디', '왜', '어떤', '누구', '몇', '얼마',
                '법률', '법', '조문', '항', '호', '목', '단', '절', '장', '편'
            }
            
            # 조사 제거 패턴 (한글 조사)
            josa_pattern = r'(에|에서|로|으로|와|과|의|을|를|이|가|은|는|도|만|부터|까지|대해|관련)$'
            
            # 단어 분리 (한글, 숫자, 영문 포함)
            words = re.findall(r'[가-힣0-9a-zA-Z]+', query)
            
            # 조사 제거 및 불용어 제거
            keywords = []
            for word in words:
                # 조사 제거
                cleaned_word = re.sub(josa_pattern, '', word)
                if cleaned_word and len(cleaned_word) > 1 and cleaned_word not in stopwords:
                    keywords.append(cleaned_word)
            
            # 중복 제거
            keywords = list(dict.fromkeys(keywords))  # 순서 유지하면서 중복 제거
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _add_to_cache(self, cache_key: str, value: List[str]):
        """캐시에 추가 (LRU 방식으로 크기 제한)"""
        try:
            # 캐시 크기 제한
            if len(self._keyword_cache) >= self._max_cache_size:
                # 가장 오래된 항목 제거 (FIFO)
                oldest_key = next(iter(self._keyword_cache))
                del self._keyword_cache[oldest_key]
                self.logger.debug(f"캐시 크기 제한으로 가장 오래된 항목 제거: {oldest_key[:50]}")
            
            # 새 항목 추가
            self._keyword_cache[cache_key] = value
            
        except Exception as e:
            self.logger.error(f"캐시 추가 중 오류: {e}")
    
    def _detect_legal_domain(self, query: str, keywords: List[str]) -> str:
        """법률 도메인 자동 감지"""
        try:
            query_lower = query.lower()
            keywords_lower = [kw.lower() for kw in keywords]
            
            # 도메인별 키워드 패턴
            domain_patterns = {
                '민사법': ['계약', '손해배상', '소유권', '채권', '채무', '불법행위', '민사'],
                '형사법': ['살인', '상해', '폭행', '협박', '강도', '절도', '사기', '형사'],
                '가족법': ['혼인', '이혼', '양육', '면접교섭', '친권', '가족'],
                '상사법': ['회사', '주식', '법인', '상법', '상사'],
                '노동법': ['근로', '임금', '해고', '노동', '근로자'],
                '부동산법': ['부동산', '토지', '건물', '임대차', '등기'],
                '민사소송법': ['민사소송', '소송', '재판', '항소', '상고'],
                '형사소송법': ['형사소송', '기소', '공소', '피고인', '피의자']
            }
            
            # 도메인 점수 계산
            domain_scores = {}
            for domain, patterns in domain_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern in query_lower:
                        score += 2
                    for kw in keywords_lower:
                        if pattern in kw:
                            score += 1
                if score > 0:
                    domain_scores[domain] = score
            
            # 가장 높은 점수의 도메인 반환
            if domain_scores:
                return max(domain_scores, key=domain_scores.get)
            
            return "general"
            
        except Exception as e:
            self.logger.error(f"Error detecting legal domain: {e}")
            return "general"
    
    async def _expand_keywords_with_llm(
        self, 
        query: str, 
        base_keywords: List[str],
        domain: Optional[str] = None
    ) -> List[str]:
        """LLM을 사용한 키워드 및 동의어 확장"""
        try:
            if not base_keywords:
                return []
            
            # 캐시 키 생성
            cache_key = f"{query}:{':'.join(sorted(base_keywords))}"
            if cache_key in self._keyword_cache:
                self.logger.debug(f"키워드 확장 캐시 히트: {cache_key[:50]}")
                # LRU: 캐시 히트 시 맨 뒤로 이동
                cached_value = self._keyword_cache.pop(cache_key)
                self._keyword_cache[cache_key] = cached_value
                return cached_value
            
            # 도메인 자동 감지
            if not domain:
                domain = self._detect_legal_domain(query, base_keywords)
            
            # 키워드가 너무 적으면 LLM 호출 스킵 (성능 최적화)
            if len(base_keywords) < 2:
                self.logger.debug(f"키워드가 부족하여 LLM 확장 스킵: {base_keywords}")
                return base_keywords
            
            # LLM 키워드 확장 (타임아웃 5초)
            try:
                expansion_result = await asyncio.wait_for(
                    self.ai_keyword_generator.expand_domain_keywords(
                        domain=domain,
                        base_keywords=base_keywords,
                        target_count=20
                    ),
                    timeout=5.0
                )
                
                if expansion_result.api_call_success and expansion_result.expanded_keywords:
                    # 기본 키워드 + 확장된 키워드 병합
                    expanded = base_keywords + expansion_result.expanded_keywords
                    # 중복 제거
                    expanded = list(dict.fromkeys(expanded))
                    # 최대 25개로 제한
                    expanded = expanded[:25]
                    
                    # 캐시 저장 (LRU: 크기 제한)
                    self._add_to_cache(cache_key, expanded)
                    self.logger.debug(f"LLM 키워드 확장 성공: {len(base_keywords)} → {len(expanded)}")
                    return expanded
                else:
                    # LLM 실패 시 폴백 확장 시도
                    self.logger.warning(f"LLM 키워드 확장 실패, 폴백 사용")
                    fallback_keywords = self.ai_keyword_generator.expand_keywords_with_fallback(
                        domain, base_keywords
                    )
                    expanded = list(dict.fromkeys(base_keywords + fallback_keywords))[:25]
                    self._add_to_cache(cache_key, expanded)
                    return expanded
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"LLM 키워드 확장 타임아웃 (5초 초과)")
                # 타임아웃 시 기본 키워드만 반환
                return base_keywords
            except Exception as e:
                self.logger.error(f"LLM 키워드 확장 중 오류: {e}")
                # 오류 시 기본 키워드만 반환
                return base_keywords
                
        except Exception as e:
            self.logger.error(f"Error in LLM keyword expansion: {e}")
            return base_keywords
    
    def _find_matched_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """콘텐츠에서 매칭된 키워드 찾기 (개선된 버전 - 대소문자 무시, 부분 매칭)"""
        try:
            matched = []
            content_lower = content.lower()
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 정확한 매칭
                if keyword_lower in content_lower:
                    matched.append(keyword)
                # 부분 매칭 (키워드가 3글자 이상인 경우)
                elif len(keyword) >= 3:
                    # 키워드의 일부가 포함되어 있는지 확인
                    for i in range(len(keyword) - 1):
                        subword = keyword[i:i+2].lower()
                        if subword in content_lower and len(subword) >= 2:
                            matched.append(keyword)
                            break
            
            # 중복 제거
            return list(dict.fromkeys(matched))
            
        except Exception as e:
            self.logger.error(f"Error finding matched keywords: {e}")
            return []
    
    def search_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """ML 강화 법률 엔티티 검색"""
        try:
            # 법률명 패턴 (확장된 패턴)
            law_pattern = r'([가-힣]+법|([가-힣]+법률)|([가-힣]+규칙)|([가-힣]+령)|([가-힣]+고시))'
            laws = re.findall(law_pattern, query)
            
            # 조문 패턴 (항, 호, 목 포함)
            article_pattern = r'제(\d+)조(?:제(\d+)항)?(?:제(\d+)호)?(?:제(\d+)목)?'
            articles = re.findall(article_pattern, query)
            
            # 판례 패턴 (확장된 패턴)
            case_pattern = r'([가-힣]+[0-9]+[가-힣]*[0-9]*[가-힣]*[0-9]*[가-힣]*)'
            cases = re.findall(case_pattern, query)
            
            # 부칙 패턴
            supplementary_pattern = r'부칙(?:제(\d+)조)?'
            supplementary = re.findall(supplementary_pattern, query)
            
            return {
                "laws": [law[0] for law in laws if law[0]],
                "articles": [f"제{article[0]}조" + (f"제{article[1]}항" if article[1] else "") + 
                           (f"제{article[2]}호" if article[2] else "") + 
                           (f"제{article[3]}목" if article[3] else "") for article in articles],
                "cases": cases[:5],
                "supplementary": [f"부칙제{sup}조" if sup else "부칙" for sup in supplementary]
            }
            
        except Exception as e:
            self.logger.error(f"Error searching legal entities: {e}")
            return {"laws": [], "articles": [], "cases": [], "supplementary": []}
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """ML 강화 검색 제안"""
        try:
            suggestions = []
            
            # 키워드 기반 제안
            keywords = self._extract_keywords(query)
            for keyword in keywords[:3]:
                suggestions.append(f"{keyword} 관련 법률")
                suggestions.append(f"{keyword} 판례")
                suggestions.append(f"{keyword} 계약서")
                suggestions.append(f"{keyword} 부칙")
            
            # 법률 엔티티 기반 제안
            entities = self.search_legal_entities(query)
            if entities["laws"]:
                for law in entities["laws"][:2]:
                    suggestions.append(f"{law} 조문 검색")
                    suggestions.append(f"{law} 부칙 검색")
            
            if entities["articles"]:
                for article in entities["articles"][:2]:
                    suggestions.append(f"{article} 내용")
                    suggestions.append(f"{article} 해석")
            
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting search suggestions: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """ML 강화 검색 통계"""
        try:
            return {
                "ml_enhanced": True,
                "quality_threshold": self.quality_threshold,
                "confidence_threshold": self.confidence_threshold,
                "supplementary_weight": self.supplementary_weight,
                "hybrid_weights": self.hybrid_weights,
                "use_ml_enhanced_search": self.use_ml_enhanced_search
            }
            
        except Exception as e:
            self.logger.error(f"Error getting search stats: {e}")
            return {"error": str(e)}


# 레거시 호환성을 위한 기존 SearchService 클래스
class SearchService(MLEnhancedSearchService):
    """레거시 호환성을 위한 검색 서비스 클래스"""
    
    def __init__(self, config: Config, database: DatabaseManager, 
                 vector_store: VectorStore, model_manager: Optional["LegalModelManager"] = None):
        """레거시 검색 서비스 초기화"""
        super().__init__(config, database, vector_store, model_manager)
        self.logger.info("Legacy SearchService initialized (using MLEnhancedSearchService)")
