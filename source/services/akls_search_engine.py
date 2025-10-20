# -*- coding: utf-8 -*-
"""
AKLS (법률전문대학원협의회) 전용 검색 엔진
표준판례 자료의 특화된 검색 기능 제공
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import re

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS or SentenceTransformer not available. Vector search will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class AKLSSearchResult:
    """AKLS 검색 결과 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    score: float
    law_area: str
    case_number: Optional[str]
    court: Optional[str]
    date: Optional[str]
    extracted_sections: Dict[str, str]


class AKLSSearchEngine:
    """AKLS 자료 특화 검색 엔진"""
    
    def __init__(self, 
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/akls_precedents"):
        """AKLS 검색 엔진 초기화"""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.index_path = index_path
        
        # 법률 영역 매핑
        self.law_area_mapping = {
            "criminal_law": "형법",
            "commercial_law": "상법",
            "civil_procedure": "민사소송법", 
            "administrative_law": "행정법",
            "constitutional_law": "헌법",
            "criminal_procedure": "형사소송법",
            "civil_law": "민법",
            "standard_precedent": "표준판례"
        }
        
        # 검색 인덱스 및 모델
        self.index = None
        self.model = None
        self.documents = []
        self.metadata = []
        
        # 초기화
        self._initialize_model()
        self._load_index()
    
    def _initialize_model(self):
        """임베딩 모델 초기화"""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS/SentenceTransformer not available. Using text-based search only.")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"임베딩 모델 로드 완료: {self.model_name}")
        except Exception as e:
            self.logger.error(f"임베딩 모델 로드 실패: {e}")
            self.model = None
    
    def _load_index(self):
        """FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE or not self.model:
            return
        
        try:
            index_file = os.path.join(self.index_path, "akls_index.faiss")
            metadata_file = os.path.join(self.index_path, "akls_metadata.json")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                self.logger.info(f"AKLS 인덱스 로드 완료: {self.index.ntotal}개 문서")
            else:
                self.logger.info("AKLS 인덱스가 없습니다. 새로 생성합니다.")
                
        except Exception as e:
            self.logger.error(f"AKLS 인덱스 로드 실패: {e}")
            self.index = None
    
    def create_index_from_documents(self, documents: List[Dict[str, Any]]):
        """처리된 문서들로부터 FAISS 인덱스 생성"""
        if not FAISS_AVAILABLE or not self.model:
            self.logger.warning("벡터 인덱스 생성 불가. 텍스트 기반 검색만 사용.")
            return
        
        try:
            # 문서 텍스트 추출
            texts = []
            metadata_list = []
            
            for doc in documents:
                # 검색용 텍스트 구성 (제목 + 요약 + 내용 일부)
                search_text = self._create_search_text(doc)
                texts.append(search_text)
                metadata_list.append(doc)
            
            # 임베딩 생성
            self.logger.info(f"AKLS 문서 {len(texts)}개에 대한 임베딩 생성 중...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # FAISS 인덱스 생성
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            # 메타데이터 저장
            self.metadata = metadata_list
            
            # 인덱스 저장
            os.makedirs(self.index_path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(self.index_path, "akls_index.faiss"))
            
            with open(os.path.join(self.index_path, "akls_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"AKLS 인덱스 생성 완료: {self.index.ntotal}개 문서")
            
        except Exception as e:
            self.logger.error(f"AKLS 인덱스 생성 실패: {e}")
            raise
    
    def _create_search_text(self, doc: Dict[str, Any]) -> str:
        """검색용 텍스트 생성"""
        parts = []
        
        # 파일명 (제목 역할)
        if "filename" in doc:
            parts.append(doc["filename"])
        
        # 추출된 섹션들
        if "extracted_sections" in doc:
            sections = doc["extracted_sections"]
            for section_name, section_content in sections.items():
                if section_content:
                    parts.append(f"{section_name}: {section_content}")
        
        # 메타데이터 정보
        if "metadata" in doc:
            metadata = doc["metadata"]
            if "law_area" in metadata:
                law_area_korean = self.law_area_mapping.get(metadata["law_area"], metadata["law_area"])
                parts.append(f"법률영역: {law_area_korean}")
        
        return " ".join(parts)
    
    def search_by_law_area(self, query: str, law_area: str, top_k: int = 5) -> List[AKLSSearchResult]:
        """특정 법률 영역에서 검색"""
        results = self.search(query, top_k=top_k * 2)  # 더 많이 가져와서 필터링
        
        # 법률 영역 필터링
        filtered_results = [
            result for result in results 
            if result.law_area == law_area
        ]
        
        return filtered_results[:top_k]
    
    def search_standard_precedents(self, query: str, top_k: int = 5) -> List[AKLSSearchResult]:
        """표준판례만 검색"""
        return self.search(query, top_k=top_k)
    
    def search_by_case_type(self, query: str, case_type: str, top_k: int = 5) -> List[AKLSSearchResult]:
        """사건 유형별 검색 (사건번호 패턴 기반)"""
        results = self.search(query, top_k=top_k * 2)
        
        # 사건 유형 필터링 (사건번호 패턴 기반)
        filtered_results = []
        for result in results:
            if result.case_number:
                # 사건번호에서 유형 추출 (예: 2023다12345 -> 다)
                case_pattern = re.search(r'(\d{4})([가-힣])(\d+)', result.case_number)
                if case_pattern and case_pattern.group(2) == case_type:
                    filtered_results.append(result)
        
        return filtered_results[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> List[AKLSSearchResult]:
        """일반 검색"""
        if not self.index or not self.model:
            return self._text_based_search(query, top_k)
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # 검색 실행
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 결과 구성
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    doc = self.metadata[idx]
                    result = self._create_search_result(doc, float(score))
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"벡터 검색 실패: {e}")
            return self._text_based_search(query, top_k)
    
    def _text_based_search(self, query: str, top_k: int) -> List[AKLSSearchResult]:
        """텍스트 기반 검색 (벡터 검색 실패 시 폴백)"""
        if not self.metadata:
            return []
        
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.metadata:
            score = 0
            
            # 파일명 매칭
            if "filename" in doc and query_lower in doc["filename"].lower():
                score += 2
            
            # 내용 매칭
            if "content" in doc and query_lower in doc["content"].lower():
                score += 1
            
            # 추출된 섹션 매칭
            if "extracted_sections" in doc:
                for section_content in doc["extracted_sections"].values():
                    if section_content and query_lower in section_content.lower():
                        score += 1.5
            
            if score > 0:
                scored_docs.append((doc, score))
        
        # 점수순 정렬
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 결과 생성
        results = []
        for doc, score in scored_docs[:top_k]:
            result = self._create_search_result(doc, score)
            results.append(result)
        
        return results
    
    def _create_search_result(self, doc: Dict[str, Any], score: float) -> AKLSSearchResult:
        """검색 결과 객체 생성"""
        metadata = doc.get("metadata", {})
        extracted_sections = doc.get("extracted_sections", {})
        
        return AKLSSearchResult(
            content=doc.get("content", ""),
            metadata=metadata,
            score=score,
            law_area=metadata.get("law_area", "unknown"),
            case_number=extracted_sections.get("case_number"),
            court=extracted_sections.get("court"),
            date=extracted_sections.get("date"),
            extracted_sections=extracted_sections
        )
    
    def get_law_area_statistics(self) -> Dict[str, int]:
        """법률 영역별 문서 통계"""
        if not self.metadata:
            return {}
        
        stats = {}
        for doc in self.metadata:
            law_area = doc.get("metadata", {}).get("law_area", "unknown")
            stats[law_area] = stats.get(law_area, 0) + 1
        
        return stats
    
    def get_document_by_case_number(self, case_number: str) -> Optional[AKLSSearchResult]:
        """사건번호로 문서 검색"""
        if not self.metadata:
            return None
        
        for doc in self.metadata:
            extracted_sections = doc.get("extracted_sections", {})
            if extracted_sections.get("case_number") == case_number:
                return self._create_search_result(doc, 1.0)
        
        return None


def main():
    """AKLS 검색 엔진 테스트"""
    import sys
    import os
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 처리된 AKLS 문서 로드
    processed_dir = "data/processed/akls"
    if not os.path.exists(processed_dir):
        print(f"처리된 AKLS 문서 디렉토리를 찾을 수 없습니다: {processed_dir}")
        return
    
    # JSON 파일들 로드
    documents = []
    for file in os.listdir(processed_dir):
        if file.endswith('.json'):
            file_path = os.path.join(processed_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                print(f"문서 로드 실패 {file}: {e}")
    
    if not documents:
        print("로드된 AKLS 문서가 없습니다.")
        return
    
    # 검색 엔진 초기화 및 인덱스 생성
    search_engine = AKLSSearchEngine()
    search_engine.create_index_from_documents(documents)
    
    # 테스트 검색
    test_queries = [
        "계약 해지",
        "손해배상",
        "형법",
        "대법원"
    ]
    
    print(f"\n=== AKLS 검색 엔진 테스트 ===")
    print(f"인덱스된 문서 수: {len(documents)}")
    
    # 법률 영역별 통계
    stats = search_engine.get_law_area_statistics()
    print(f"\n법률 영역별 문서 수:")
    for area, count in stats.items():
        korean_name = search_engine.law_area_mapping.get(area, area)
        print(f"  {korean_name}: {count}개")
    
    # 테스트 쿼리 실행
    for query in test_queries:
        print(f"\n--- 검색 쿼리: '{query}' ---")
        results = search_engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. 점수: {result.score:.3f}")
            print(f"   법률영역: {search_engine.law_area_mapping.get(result.law_area, result.law_area)}")
            if result.case_number:
                print(f"   사건번호: {result.case_number}")
            if result.court:
                print(f"   법원: {result.court}")
            print(f"   파일명: {result.metadata.get('filename', 'N/A')}")
            print()


if __name__ == "__main__":
    main()
