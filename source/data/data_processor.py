"""
Data Processor
데이터 전처리 및 변환 모듈
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataProcessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        """데이터 프로세서 초기화"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataProcessor initialized")
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?;:()]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, 
                              overlap: int = 100) -> List[str]:
        """텍스트를 청크로 분할"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마지막 문장 끝 찾기
                last_sentence_end = text.rfind('.', start, end)
                if last_sentence_end > start:
                    end = last_sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """법률 엔티티 추출"""
        entities = {
            "laws": [],
            "articles": [],
            "cases": [],
            "keywords": []
        }
        
        # 법률명 패턴
        law_pattern = r'([가-힣]+법|([가-힣]+법률))'
        laws = re.findall(law_pattern, text)
        entities["laws"] = [law[0] for law in laws]
        
        # 조문 패턴
        article_pattern = r'제(\d+)조'
        articles = re.findall(article_pattern, text)
        entities["articles"] = [f"제{article}조" for article in articles]
        
        # 판례 패턴
        case_pattern = r'([가-힣]+[0-9]+[가-힣]*[0-9]*[가-힣]*[0-9]*[가-힣]*)'
        cases = re.findall(case_pattern, text)
        entities["cases"] = cases[:10]  # 상위 10개만
        
        # 키워드 추출 (간단한 형태소 분석)
        keywords = self._extract_keywords(text)
        entities["keywords"] = keywords
        
        return entities
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """키워드 추출"""
        # 불용어 제거
        stopwords = {'의', '을', '를', '이', '가', '은', '는', '에', '에서', '로', '으로', '와', '과', '도', '만', '부터', '까지'}
        
        # 단어 분리 (간단한 공백 기준)
        words = re.findall(r'[가-힣]+', text)
        
        # 불용어 제거 및 길이 필터링
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        # 빈도 계산
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도순 정렬
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def process_legal_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """법률 문서 처리"""
        try:
            content = document.get("content", "")
            title = document.get("title", "")
            
            # 텍스트 정리
            cleaned_content = self.clean_text(content)
            cleaned_title = self.clean_text(title)
            
            # 청크 분할
            chunks = self.split_text_into_chunks(cleaned_content)
            
            # 엔티티 추출
            entities = self.extract_legal_entities(cleaned_content)
            
            # 처리 결과
            processed_doc = {
                "original_title": title,
                "original_content": content,
                "cleaned_title": cleaned_title,
                "cleaned_content": cleaned_content,
                "chunks": chunks,
                "entities": entities,
                "chunk_count": len(chunks),
                "word_count": len(cleaned_content.split()),
                "processed_at": self._get_current_timestamp()
            }
            
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing legal document: {e}")
            return {
                "error": str(e),
                "original_title": document.get("title", ""),
                "original_content": document.get("content", "")
            }
    
    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_document(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """문서 유효성 검사"""
        errors = []
        
        # 필수 필드 검사
        if not document.get("title"):
            errors.append("Title is required")
        
        if not document.get("content"):
            errors.append("Content is required")
        
        # 내용 길이 검사
        content = document.get("content", "")
        if len(content) < 10:
            errors.append("Content too short (minimum 10 characters)")
        
        if len(content) > 100000:
            errors.append("Content too long (maximum 100,000 characters)")
        
        # 제목 길이 검사
        title = document.get("title", "")
        if len(title) > 500:
            errors.append("Title too long (maximum 500 characters)")
        
        return len(errors) == 0, errors
