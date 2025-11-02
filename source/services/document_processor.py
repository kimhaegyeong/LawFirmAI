# -*- coding: utf-8 -*-
"""
Document Processor
문서 처리 및 청킹 시스템 구현
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

try:
    from langchain.schema import Document
    from langchain.text_splitter import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )
    from langchain_community.document_loaders import (
        DirectoryLoader,
        PyPDFLoader,
        TextLoader,
    )
    LANCHAIN_AVAILABLE = True
except ImportError:
    LANCHAIN_AVAILABLE = False
    # Mock classes for when LangChain is not available
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass
        def split_documents(self, docs):
            return docs

    class DirectoryLoader:
        def __init__(self, *args, **kwargs):
            pass
        def load(self):
            return []

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    chunk_index: int
    source_document: str


class LegalDocumentProcessor:
    """법률 문서 처리기"""

    def __init__(self, config):
        """문서 처리기 초기화"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 텍스트 분할기 설정
        self.text_splitter = self._create_text_splitter()

        # 법률 문서 특화 패턴
        self.legal_patterns = {
            'law_article': r'제\d+조',
            'law_paragraph': r'제\d+조\s*제\d+항',
            'law_subparagraph': r'제\d+조\s*제\d+항\s*제\d+호',
            'case_number': r'[0-9]{4}[가-힣]\d+',
            'court_name': r'(대법원|고등법원|지방법원|가정법원|특허법원)',
            'date_pattern': r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일'
        }

    def _create_text_splitter(self):
        """텍스트 분할기 생성"""
        if not LANCHAIN_AVAILABLE:
            logger.debug("LangChain is not available. Using basic text splitter.")
            return None

        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # 문단 구분
                "\n",    # 줄 구분
                "제",    # 법조문 구분
                "조",    # 조항 구분
                "항",    # 항 구분
                "호",    # 호 구분
                ".",     # 문장 구분
                " ",     # 단어 구분
                ""       # 문자 구분
            ]
        )

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """디렉토리에서 문서 로드"""
        if not LANCHAIN_AVAILABLE:
            logger.debug("LangChain is not available. Cannot load documents.")
            return []

        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()

            self.logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load documents from {directory_path}: {e}")
            return []

    def load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """파일 목록에서 문서 로드"""
        documents = []

        for file_path in file_paths:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        metadata = {
                            'source': file_path,
                            'file_name': os.path.basename(file_path),
                            'file_type': 'text'
                        }
                        documents.append(Document(page_content=content, metadata=metadata))

                elif file_path.endswith('.pdf'):
                    if LANCHAIN_AVAILABLE:
                        loader = PyPDFLoader(file_path)
                        pdf_docs = loader.load()
                        documents.extend(pdf_docs)
                    else:
                        self.logger.warning(f"PDF loading requires LangChain. Skipping {file_path}")

                else:
                    self.logger.warning(f"Unsupported file type: {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to load document {file_path}: {e}")

        self.logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} files")
        return documents

    def preprocess_document(self, document: Document) -> Document:
        """문서 전처리"""
        content = document.page_content

        # 기본 정리
        content = self._clean_text(content)

        # 법률 문서 특화 전처리
        content = self._preprocess_legal_content(content)

        # 메타데이터 업데이트
        metadata = document.metadata.copy()
        metadata.update({
            'processed': True,
            'original_length': len(document.page_content),
            'processed_length': len(content),
            'document_type': self._detect_document_type(content)
        })

        return Document(page_content=content, metadata=metadata)

    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)

        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,;:!?()\[\]{}""''""''「」『』]', '', text)

        # 줄바꿈 정리
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

    def _preprocess_legal_content(self, text: str) -> str:
        """법률 문서 특화 전처리"""
        # 법조문 번호 정리
        text = re.sub(r'제(\d+)조', r'제\1조', text)
        text = re.sub(r'제(\d+)항', r'제\1항', text)
        text = re.sub(r'제(\d+)호', r'제\1호', text)

        # 판례 번호 정리
        text = re.sub(r'([0-9]{4}[가-힣]\d+)', r'\1', text)

        # 날짜 형식 정리
        text = re.sub(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', r'\1년 \2월 \3일', text)

        return text

    def _detect_document_type(self, content: str) -> str:
        """문서 타입 감지"""
        # 법령 패턴 확인
        if re.search(self.legal_patterns['law_article'], content):
            return 'law'

        # 판례 패턴 확인
        if re.search(self.legal_patterns['case_number'], content):
            return 'precedent'

        # 행정규칙 패턴 확인
        if re.search(self.legal_patterns['court_name'], content):
            return 'administrative_rule'

        return 'general'

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """문서를 청크로 분할"""
        try:
            if self.text_splitter and LANCHAIN_AVAILABLE:
                # LangChain 텍스트 분할기 사용
                chunks = self.text_splitter.split_documents([document])
            else:
                # 기본 분할 방식
                chunks = self._basic_split(document)

            # DocumentChunk 객체로 변환
            document_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_obj = DocumentChunk(
                    content=chunk.page_content,
                    metadata=chunk.metadata.copy(),
                    chunk_id=f"{chunk.metadata.get('source', 'unknown')}_{i}",
                    chunk_index=i,
                    source_document=chunk.metadata.get('source', 'unknown')
                )

                # 청크별 메타데이터 추가
                chunk_obj.metadata.update({
                    'chunk_id': chunk_obj.chunk_id,
                    'chunk_index': i,
                    'chunk_length': len(chunk.page_content),
                    'legal_patterns': self._extract_legal_patterns(chunk.page_content)
                })

                document_chunks.append(chunk_obj)

            self.logger.info(f"Split document into {len(document_chunks)} chunks")
            return document_chunks

        except Exception as e:
            self.logger.error(f"Failed to split document: {e}")
            return []

    def _basic_split(self, document: Document) -> List[Document]:
        """기본 문서 분할"""
        content = document.page_content
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # 문장 경계에서 자르기
            if end < len(content):
                # 마지막 문장 끝 찾기
                last_period = content.rfind('.', start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1

            chunk_content = content[start:end]
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_start': start,
                'chunk_end': end
            })

            chunks.append(Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            ))

            start = end - chunk_overlap
            if start >= len(content):
                break

        return chunks

    def _extract_legal_patterns(self, text: str) -> Dict[str, List[str]]:
        """법률 패턴 추출"""
        patterns = {}

        for pattern_name, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                patterns[pattern_name] = matches

        return patterns

    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """문서 배치 처리"""
        all_chunks = []

        for i, document in enumerate(documents):
            try:
                # 문서 전처리
                processed_doc = self.preprocess_document(document)

                # 문서 분할
                chunks = self.split_document(processed_doc)
                all_chunks.extend(chunks)

                self.logger.info(f"Processed document {i+1}/{len(documents)}: {len(chunks)} chunks")

            except Exception as e:
                self.logger.error(f"Failed to process document {i+1}: {e}")
                continue

        self.logger.info(f"Total processed chunks: {len(all_chunks)}")
        return all_chunks

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """청크 통계 반환"""
        if not chunks:
            return {"error": "No chunks available"}

        chunk_lengths = [len(chunk.content) for chunk in chunks]
        document_types = [chunk.metadata.get('document_type', 'unknown') for chunk in chunks]

        # 문서 타입별 통계
        type_counts = {}
        for doc_type in document_types:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "document_types": type_counts,
            "total_characters": sum(chunk_lengths)
        }

    def filter_chunks_by_relevance(self, chunks: List[DocumentChunk],
                                  query: str, min_similarity: float = 0.5) -> List[DocumentChunk]:
        """관련성 기반 청크 필터링"""
        # 간단한 키워드 기반 필터링 (실제 구현에서는 임베딩 기반 유사도 사용)
        query_words = set(query.lower().split())

        relevant_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())

            # 단어 겹침 비율 계산
            overlap = len(query_words.intersection(chunk_words))
            similarity = overlap / len(query_words) if query_words else 0

            if similarity >= min_similarity:
                chunk.metadata['relevance_score'] = similarity
                relevant_chunks.append(chunk)

        # 관련성 점수 순으로 정렬
        relevant_chunks.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)

        return relevant_chunks
