# -*- coding: utf-8 -*-
"""
AKLS (법률전문대학원협의회) 자료 전용 처리기
표준판례 PDF 문서의 텍스트 추출, 구조 파싱, 메타데이터 생성 기능 제공
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. PDF processing will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class AKLSDocument:
    """AKLS 문서 데이터 클래스"""
    filename: str
    content: str
    metadata: Dict[str, Any]
    law_area: str
    document_type: str
    extracted_sections: Dict[str, str]
    processing_timestamp: str


class AKLSProcessor:
    """법률전문대학원협의회 자료 전용 처리기"""
    
    def __init__(self):
        """AKLS 프로세서 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 법률 영역 매핑
        self.law_area_mapping = {
            "형법": "criminal_law",
            "상법": "commercial_law", 
            "민사소송법": "civil_procedure",
            "행정법": "administrative_law",
            "헌법": "constitutional_law",
            "형사소송법": "criminal_procedure",
            "민법": "civil_law",
            "표준판례": "standard_precedent"
        }
        
        # 표준판례 구조 패턴
        self.precedent_patterns = {
            "case_number": r"(\d{4}[가-힣]\d+)",
            "court": r"(대법원|고등법원|지방법원|가정법원|특허법원)",
            "date": r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)",
            "summary": r"(요약|판시사항|판결요지)",
            "reasoning": r"(이유|판단)",
            "conclusion": r"(결론|주문)"
        }
        
        # 법률 조항 패턴
        self.legal_patterns = {
            "article": r"제(\d+)조",
            "paragraph": r"제(\d+)조\s*제(\d+)항",
            "subparagraph": r"제(\d+)조\s*제(\d+)항\s*제(\d+)호",
            "law_name": r"([가-힣]+법)"
        }
    
    def extract_law_area_from_filename(self, filename: str) -> str:
        """파일명에서 법률 영역 추출"""
        filename_lower = filename.lower()
        
        for korean_name, english_code in self.law_area_mapping.items():
            if korean_name in filename:
                return english_code
        
        # 기본값
        return "standard_precedent"
    
    def extract_year_from_filename(self, filename: str) -> Optional[str]:
        """파일명에서 연도 추출"""
        year_match = re.search(r"(\d{4})", filename)
        return year_match.group(1) if year_match else None
    
    def extract_pdf_text(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2가 설치되지 않았습니다. pip install PyPDF2로 설치하세요.")
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- 페이지 {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        self.logger.warning(f"페이지 {page_num + 1} 텍스트 추출 실패: {e}")
                        continue
                
                return text.strip()
                
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 오류: {e}")
            raise ValueError(f"PDF 파일을 읽을 수 없습니다: {str(e)}")
    
    def parse_standard_precedent_structure(self, content: str) -> Dict[str, str]:
        """표준판례의 구조적 요소 추출"""
        sections = {}
        
        # 사건번호 추출
        case_number_match = re.search(self.precedent_patterns["case_number"], content)
        if case_number_match:
            sections["case_number"] = case_number_match.group(1)
        
        # 법원명 추출
        court_match = re.search(self.precedent_patterns["court"], content)
        if court_match:
            sections["court"] = court_match.group(1)
        
        # 날짜 추출
        date_match = re.search(self.precedent_patterns["date"], content)
        if date_match:
            sections["date"] = date_match.group(1)
        
        # 섹션별 내용 추출
        section_keywords = ["요약", "판시사항", "판결요지", "이유", "판단", "결론", "주문"]
        
        for keyword in section_keywords:
            pattern = rf"{keyword}[:\s]*([^\n]*(?:\n(?!\n)[^\n]*)*)"
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                sections[keyword] = match.group(1).strip()
        
        return sections
    
    def extract_legal_references(self, content: str) -> List[Dict[str, str]]:
        """법률 조항 참조 추출"""
        references = []
        
        # 법률명 추출
        law_matches = re.finditer(self.legal_patterns["law_name"], content)
        for match in law_matches:
            references.append({
                "type": "law_name",
                "content": match.group(1),
                "position": match.start()
            })
        
        # 조항 추출
        article_matches = re.finditer(self.legal_patterns["article"], content)
        for match in article_matches:
            references.append({
                "type": "article",
                "content": f"제{match.group(1)}조",
                "position": match.start()
            })
        
        return references
    
    def create_metadata(self, filename: str, content: str, extracted_sections: Dict[str, str]) -> Dict[str, Any]:
        """AKLS 문서 메타데이터 생성"""
        law_area = self.extract_law_area_from_filename(filename)
        year = self.extract_year_from_filename(filename)
        legal_references = self.extract_legal_references(content)
        
        metadata = {
            "source": "akls",
            "filename": filename,
            "law_area": law_area,
            "year": year,
            "document_type": "standard_precedent",
            "file_type": "pdf",
            "processing_timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "has_case_number": "case_number" in extracted_sections,
            "has_court_info": "court" in extracted_sections,
            "has_date": "date" in extracted_sections,
            "legal_references_count": len(legal_references),
            "legal_references": legal_references[:10],  # 처음 10개만 저장
            "sections_found": list(extracted_sections.keys())
        }
        
        return metadata
    
    def process_single_document(self, file_path: str) -> Optional[AKLSDocument]:
        """단일 AKLS 문서 처리"""
        try:
            filename = os.path.basename(file_path)
            self.logger.info(f"AKLS 문서 처리 시작: {filename}")
            
            # PDF 텍스트 추출
            content = self.extract_pdf_text(file_path)
            if not content.strip():
                self.logger.warning(f"텍스트가 추출되지 않음: {filename}")
                return None
            
            # 법률 영역 추출
            law_area = self.extract_law_area_from_filename(filename)
            
            # 표준판례 구조 파싱
            extracted_sections = self.parse_standard_precedent_structure(content)
            
            # 메타데이터 생성
            metadata = self.create_metadata(filename, content, extracted_sections)
            
            # AKLSDocument 객체 생성
            akls_doc = AKLSDocument(
                filename=filename,
                content=content,
                metadata=metadata,
                law_area=law_area,
                document_type="standard_precedent",
                extracted_sections=extracted_sections,
                processing_timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(f"AKLS 문서 처리 완료: {filename} (길이: {len(content)})")
            return akls_doc
            
        except Exception as e:
            self.logger.error(f"AKLS 문서 처리 실패 {file_path}: {e}")
            return None
    
    def process_akls_directory(self, directory_path: str) -> List[AKLSDocument]:
        """AKLS 디렉토리의 모든 PDF 파일 처리"""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        
        pdf_files = []
        for file in os.listdir(directory_path):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, file))
        
        self.logger.info(f"AKLS 디렉토리에서 {len(pdf_files)}개 PDF 파일 발견")
        
        processed_docs = []
        for file_path in pdf_files:
            doc = self.process_single_document(file_path)
            if doc:
                processed_docs.append(doc)
        
        self.logger.info(f"AKLS 문서 처리 완료: {len(processed_docs)}개 성공")
        return processed_docs
    
    def save_processed_documents(self, documents: List[AKLSDocument], output_dir: str):
        """처리된 문서들을 JSON 파일로 저장"""
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, doc in enumerate(documents):
            output_file = os.path.join(output_dir, f"akls_{i:03d}_{doc.law_area}.json")
            
            # 문서 데이터를 딕셔너리로 변환
            doc_data = {
                "filename": doc.filename,
                "content": doc.content,
                "metadata": doc.metadata,
                "law_area": doc.law_area,
                "document_type": doc.document_type,
                "extracted_sections": doc.extracted_sections,
                "processing_timestamp": doc.processing_timestamp
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"처리된 AKLS 문서 {len(documents)}개를 {output_dir}에 저장 완료")


def main():
    """AKLS 문서 처리 메인 함수"""
    import sys
    import os
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # AKLS 디렉토리 경로
    akls_dir = "data/raw/akls"
    output_dir = "data/processed/akls"
    
    if not os.path.exists(akls_dir):
        print(f"AKLS 디렉토리를 찾을 수 없습니다: {akls_dir}")
        return
    
    # AKLS 프로세서 초기화 및 실행
    processor = AKLSProcessor()
    
    try:
        # 문서 처리
        processed_docs = processor.process_akls_directory(akls_dir)
        
        if processed_docs:
            # 결과 저장
            processor.save_processed_documents(processed_docs, output_dir)
            
            # 처리 결과 요약
            print(f"\n=== AKLS 문서 처리 완료 ===")
            print(f"처리된 문서 수: {len(processed_docs)}")
            
            # 법률 영역별 통계
            area_stats = {}
            for doc in processed_docs:
                area = doc.law_area
                area_stats[area] = area_stats.get(area, 0) + 1
            
            print(f"\n법률 영역별 문서 수:")
            for area, count in area_stats.items():
                print(f"  {area}: {count}개")
                
        else:
            print("처리된 문서가 없습니다.")
            
    except Exception as e:
        print(f"AKLS 문서 처리 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
