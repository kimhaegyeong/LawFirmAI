# -*- coding: utf-8 -*-
"""
Context Manager
컨텍스트 관리 시스템 구현
"""

import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """컨텍스트 윈도우 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    timestamp: datetime
    source_document: str
    chunk_id: str


@dataclass
class ContextSession:
    """컨텍스트 세션 데이터 클래스"""
    session_id: str
    query_history: List[str]
    context_windows: List[ContextWindow]
    created_at: datetime
    last_accessed: datetime
    total_tokens: int


class ContextManager:
    """컨텍스트 관리 시스템"""
    
    def __init__(self, config):
        """컨텍스트 관리자 초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 세션 관리
        self.sessions: Dict[str, ContextSession] = {}
        self.max_sessions = 100
        
        # 컨텍스트 윈도우 설정
        self.max_context_length = config.max_context_length
        self.context_overlap_threshold = 0.3
        self.relevance_threshold = 0.5
        
        # 캐시 설정
        self.context_cache: Dict[str, List[ContextWindow]] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # 통계
        self.stats = {
            'total_sessions': 0,
            'total_contexts': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def create_context_session(self, session_id: str) -> ContextSession:
        """새로운 컨텍스트 세션 생성"""
        session = ContextSession(
            session_id=session_id,
            query_history=[],
            context_windows=[],
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            total_tokens=0
        )
        
        self.sessions[session_id] = session
        self.stats['total_sessions'] += 1
        
        self.logger.info(f"Created new context session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ContextSession]:
        """세션 조회"""
        session = self.sessions.get(session_id)
        if session:
            session.last_accessed = datetime.now()
        return session
    
    def add_query_to_session(self, session_id: str, query: str) -> bool:
        """세션에 쿼리 추가"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_context_session(session_id)
        
        session.query_history.append(query)
        session.last_accessed = datetime.now()
        
        self.logger.info(f"Added query to session {session_id}: {query[:50]}...")
        return True
    
    def build_context_window(self, retrieved_docs: List[Dict[str, Any]], 
                           query: str, session_id: Optional[str] = None) -> List[ContextWindow]:
        """컨텍스트 윈도우 구축"""
        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(query, retrieved_docs)
            if cache_key in self.context_cache:
                cached_contexts = self.context_cache[cache_key]
                if self._is_cache_valid(cached_contexts):
                    self.stats['cache_hits'] += 1
                    return cached_contexts
            
            self.stats['cache_misses'] += 1
            
            # 컨텍스트 윈도우 생성
            context_windows = []
            current_length = 0
            
            # 관련성 점수 계산 및 정렬
            scored_docs = self._score_documents_relevance(retrieved_docs, query)
            scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            for doc in scored_docs:
                # 컨텍스트 길이 확인
                doc_content = doc.get('content', '')
                doc_length = len(doc_content)
                
                if current_length + doc_length > self.max_context_length:
                    # 부분적으로 포함할 수 있는지 확인
                    remaining_length = self.max_context_length - current_length
                    if remaining_length > 100:  # 최소 100자 이상
                        doc_content = self._truncate_content(doc_content, remaining_length)
                        doc_length = len(doc_content)
                    else:
                        break
                
                # 컨텍스트 윈도우 생성
                context_window = ContextWindow(
                    content=doc_content,
                    metadata=doc.get('metadata', {}),
                    relevance_score=doc['relevance_score'],
                    timestamp=datetime.now(),
                    source_document=doc.get('source', 'unknown'),
                    chunk_id=doc.get('chunk_id', 'unknown')
                )
                
                context_windows.append(context_window)
                current_length += doc_length
            
            # 세션에 컨텍스트 추가
            if session_id:
                self._add_context_to_session(session_id, context_windows)
            
            # 캐시에 저장
            self.context_cache[cache_key] = context_windows
            
            self.stats['total_contexts'] += len(context_windows)
            self.logger.info(f"Built context window with {len(context_windows)} contexts, "
                           f"total length: {current_length}")
            
            return context_windows
            
        except Exception as e:
            self.logger.error(f"Failed to build context window: {e}")
            return []
    
    def _score_documents_relevance(self, documents: List[Dict[str, Any]], 
                                  query: str) -> List[Dict[str, Any]]:
        """문서 관련성 점수 계산"""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in documents:
            content = doc.get('content', '').lower()
            content_words = set(content.split())
            
            # 단어 겹침 기반 관련성 계산
            word_overlap = len(query_words.intersection(content_words))
            word_similarity = word_overlap / len(query_words) if query_words else 0
            
            # 기존 유사도 점수와 결합
            existing_similarity = doc.get('similarity', 0.0)
            
            # 가중 평균으로 최종 점수 계산
            relevance_score = (word_similarity * 0.6 + existing_similarity * 0.4)
            
            doc['relevance_score'] = relevance_score
            scored_docs.append(doc)
        
        return scored_docs
    
    def _truncate_content(self, content: str, max_length: int) -> str:
        """내용을 최대 길이에 맞게 자르기"""
        if len(content) <= max_length:
            return content
        
        # 문장 경계에서 자르기
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.8:  # 80% 이상이면 문장 끝에서 자르기
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def _add_context_to_session(self, session_id: str, context_windows: List[ContextWindow]):
        """세션에 컨텍스트 추가"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_context_session(session_id)
        
        # 중복 제거 및 관련성 기반 필터링
        existing_chunk_ids = {cw.chunk_id for cw in session.context_windows}
        new_contexts = []
        
        for cw in context_windows:
            if cw.chunk_id not in existing_chunk_ids and cw.relevance_score >= self.relevance_threshold:
                new_contexts.append(cw)
        
        # 세션에 추가
        session.context_windows.extend(new_contexts)
        
        # 토큰 수 업데이트
        session.total_tokens += sum(len(cw.content) for cw in new_contexts)
        
        # 최대 컨텍스트 길이 초과 시 오래된 것부터 제거
        self._trim_session_context(session)
        
        self.logger.info(f"Added {len(new_contexts)} contexts to session {session_id}")
    
    def _trim_session_context(self, session: ContextSession):
        """세션 컨텍스트 정리"""
        if session.total_tokens <= self.max_context_length:
            return
        
        # 관련성 점수 순으로 정렬
        session.context_windows.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 최대 길이에 맞게 컨텍스트 선택
        trimmed_contexts = []
        current_tokens = 0
        
        for cw in session.context_windows:
            if current_tokens + len(cw.content) <= self.max_context_length:
                trimmed_contexts.append(cw)
                current_tokens += len(cw.content)
            else:
                break
        
        session.context_windows = trimmed_contexts
        session.total_tokens = current_tokens
        
        self.logger.info(f"Trimmed session {session.session_id} context to {len(trimmed_contexts)} windows")
    
    def get_session_context(self, session_id: str, max_length: Optional[int] = None) -> str:
        """세션 컨텍스트 조회"""
        session = self.get_session(session_id)
        if not session or not session.context_windows:
            return ""
        
        # 관련성 점수 순으로 정렬
        sorted_contexts = sorted(session.context_windows, 
                                key=lambda x: x.relevance_score, reverse=True)
        
        context_parts = []
        current_length = 0
        max_len = max_length or self.max_context_length
        
        for cw in sorted_contexts:
            if current_length + len(cw.content) <= max_len:
                context_parts.append(f"[문서: {cw.source_document}]\n{cw.content}")
                current_length += len(cw.content)
            else:
                break
        
        return "\n\n".join(context_parts)
    
    def _generate_cache_key(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """캐시 키 생성"""
        doc_ids = [doc.get('chunk_id', '') for doc in documents]
        return f"{hash(query)}_{hash(tuple(doc_ids))}"
    
    def _is_cache_valid(self, cached_contexts: List[ContextWindow]) -> bool:
        """캐시 유효성 확인"""
        if not cached_contexts:
            return False
        
        # 가장 오래된 컨텍스트의 시간 확인
        oldest_timestamp = min(cw.timestamp for cw in cached_contexts)
        return datetime.now() - oldest_timestamp < self.cache_ttl
    
    def clear_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """오래된 세션 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.last_accessed < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """컨텍스트 통계 반환"""
        total_contexts = sum(len(session.context_windows) for session in self.sessions.values())
        total_tokens = sum(session.total_tokens for session in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "total_contexts": total_contexts,
            "total_tokens": total_tokens,
            "cache_hits": self.stats['cache_hits'],
            "cache_misses": self.stats['cache_misses'],
            "cache_hit_ratio": self.stats['cache_hits'] / 
                              (self.stats['cache_hits'] + self.stats['cache_misses']) 
                              if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            "avg_contexts_per_session": total_contexts / len(self.sessions) 
                                       if self.sessions else 0
        }
    
    def optimize_context_for_query(self, query: str, context_windows: List[ContextWindow]) -> List[ContextWindow]:
        """쿼리에 최적화된 컨텍스트 선택"""
        if not context_windows:
            return []
        
        # 쿼리 키워드 추출
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        # 각 컨텍스트의 키워드 매칭 점수 계산
        scored_contexts = []
        for cw in context_windows:
            content_keywords = set(re.findall(r'\w+', cw.content.lower()))
            
            # 키워드 겹침 점수
            keyword_overlap = len(query_keywords.intersection(content_keywords))
            keyword_score = keyword_overlap / len(query_keywords) if query_keywords else 0
            
            # 기존 관련성 점수와 결합
            combined_score = (keyword_score * 0.7 + cw.relevance_score * 0.3)
            
            scored_contexts.append((cw, combined_score))
        
        # 점수 순으로 정렬
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 컨텍스트 선택
        optimized_contexts = []
        current_length = 0
        
        for cw, score in scored_contexts:
            if current_length + len(cw.content) <= self.max_context_length:
                optimized_contexts.append(cw)
                current_length += len(cw.content)
            else:
                break
        
        self.logger.info(f"Optimized context: {len(optimized_contexts)}/{len(context_windows)} contexts selected")
        return optimized_contexts
