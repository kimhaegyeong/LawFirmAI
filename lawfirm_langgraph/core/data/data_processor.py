# -*- coding: utf-8 -*-
"""
Legal Data Processor
법률 데이터 처리 클래스
"""

import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = get_logger(__name__)


@dataclass
class ProcessedDocument:
    """처리된 문서 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    chunks: List[str]
    keywords: List[str]
    categories: List[str]
    processing_time: float


class LegalDataProcessor:
    """법률 데이터 처리기"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        데이터 처리기 초기화
        
        Args:
            config: 처리 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # 처리 설정
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        
        # 법률 키워드 패턴
        self.legal_patterns = {
            'articles': r'제\d+조',
            'paragraphs': r'제\d+항',
            'subparagraphs': r'제\d+호',
            'laws': r'[가-힣]+법',
            'courts': r'[가-힣]+법원',
            'cases': r'[가-힣]+사건',
            'dates': r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일'
        }
        
        self.logger.info("LegalDataProcessor initialized")
    
    def process_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ProcessedDocument:
        """
        문서 처리
        
        Args:
            content: 문서 내용
            metadata: 문서 메타데이터
            
        Returns:
            ProcessedDocument: 처리된 문서
        """
        start_time = datetime.now()
        
        try:
            # 기본 전처리
            cleaned_content = self._clean_content(content)
            
            # 청킹
            chunks = self._chunk_text(cleaned_content)
            
            # 키워드 추출
            keywords = self._extract_keywords(cleaned_content)
            
            # 카테고리 분류
            categories = self._classify_categories(cleaned_content)
            
            # 메타데이터 처리
            processed_metadata = metadata or {}
            processed_metadata.update({
                'processed_at': datetime.now().isoformat(),
                'chunk_count': len(chunks),
                'keyword_count': len(keywords),
                'category_count': len(categories)
            })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedDocument(
                content=cleaned_content,
                metadata=processed_metadata,
                chunks=chunks,
                keywords=keywords,
                categories=categories,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            raise
    
    def _clean_content(self, content: str) -> str:
        """
        내용 정리
        
        Args:
            content: 원본 내용
            
        Returns:
            str: 정리된 내용
        """
        # 불필요한 공백 제거
        content = re.sub(r'\s+', ' ', content)
        
        # 특수 문자 정리
        content = re.sub(r'[^\w\s가-힣.,!?()[]{}]', '', content)
        
        # 연속된 구두점 정리
        content = re.sub(r'[.]{2,}', '.', content)
        content = re.sub(r'[,]{2,}', ',', content)
        
        return content.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        텍스트 청킹
        
        Args:
            text: 청킹할 텍스트
            
        Returns:
            List[str]: 청크 목록
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        키워드 추출
        
        Args:
            text: 키워드를 추출할 텍스트
            
        Returns:
            List[str]: 키워드 목록
        """
        keywords = []
        
        # 법률 패턴 매칭
        for pattern_name, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text)
            keywords.extend(matches)
        
        # 중복 제거 및 정렬
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def _classify_categories(self, text: str) -> List[str]:
        """
        카테고리 분류
        
        Args:
            text: 분류할 텍스트
            
        Returns:
            List[str]: 카테고리 목록
        """
        categories = []
        
        # 법률 분야별 키워드 매칭
        category_keywords = {
            'civil': ['민사', '계약', '손해배상', '소유권', '임대차'],
            'criminal': ['형사', '범죄', '처벌', '벌금', '징역'],
            'family': ['가족', '이혼', '양육', '부양', '상속'],
            'tax': ['세금', '소득세', '부가가치세', '신고', '납부'],
            'administrative': ['행정', '허가', '인허가', '행정처분'],
            'patent': ['특허', '상표', '디자인', '지식재산']
        }
        
        for category, keywords_list in category_keywords.items():
            for keyword in keywords_list:
                if keyword in text:
                    categories.append(category)
                    break
        
        return list(set(categories))
    
    def process_file(self, file_path: Path) -> ProcessedDocument:
        """
        파일 처리
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            ProcessedDocument: 처리된 문서
        """
        try:
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 메타데이터 생성
            metadata = {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            return self.process_document(content, metadata)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def batch_process(self, file_paths: List[Path]) -> List[ProcessedDocument]:
        """
        배치 처리
        
        Args:
            file_paths: 처리할 파일 경로 목록
            
        Returns:
            List[ProcessedDocument]: 처리된 문서 목록
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
                self.logger.info(f"Processed file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path}: {e}")
        
        return results
    
    def get_processing_stats(self, documents: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        처리 통계 반환
        
        Args:
            documents: 처리된 문서 목록
            
        Returns:
            Dict[str, Any]: 처리 통계
        """
        if not documents:
            return {}
        
        total_chunks = sum(len(doc.chunks) for doc in documents)
        total_keywords = sum(len(doc.keywords) for doc in documents)
        total_categories = sum(len(doc.categories) for doc in documents)
        total_processing_time = sum(doc.processing_time for doc in documents)
        
        return {
            'document_count': len(documents),
            'total_chunks': total_chunks,
            'total_keywords': total_keywords,
            'total_categories': total_categories,
            'avg_processing_time': total_processing_time / len(documents),
            'total_processing_time': total_processing_time
        }


# 기본 인스턴스 생성
def create_data_processor() -> LegalDataProcessor:
    """기본 데이터 처리기 생성"""
    return LegalDataProcessor()


if __name__ == "__main__":
    # 테스트 코드
    processor = create_data_processor()
    
    # 샘플 문서 처리
    sample_text = """
    민법 제543조에 따르면, 계약의 해지는 당사자 일방의 의사표시로 인하여 계약의 효력이 장래에 향하여 소멸하는 것을 말한다.
    계약 해지 시 손해배상 청구가 가능하며, 이는 민법 제544조에서 규정하고 있다.
    """
    
    result = processor.process_document(sample_text)
    print(f"Processed document:")
    print(f"  Chunks: {len(result.chunks)}")
    print(f"  Keywords: {result.keywords}")
    print(f"  Categories: {result.categories}")
    print(f"  Processing time: {result.processing_time:.3f}s")