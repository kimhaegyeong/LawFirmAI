#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터DB 구축 전용 스크립트 (JSON → 벡터DB)

data/raw/ 디렉토리의 JSON 파일들을 읽어와서 벡터DB를 구축합니다.
하이브리드 검색을 위해 SQLite와 FAISS를 모두 구축합니다.

사용법:
    python scripts/build_vector_db.py --mode build
    python scripts/build_vector_db.py --mode laws
    python scripts/build_vector_db.py --mode precedents
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager
from source.data.data_processor import LegalDataProcessor
from source.utils.logger import get_logger

logger = get_logger(__name__)


class VectorDBBuilder:
    """벡터DB 구축 전용 클래스"""
    
    def __init__(self, raw_data_dir: str = "./data/raw"):
        """초기화"""
        self.raw_data_dir = Path(raw_data_dir)
        self.vector_store = LegalVectorStore()
        self.db_manager = DatabaseManager()
        self.processor = LegalDataProcessor()
        
        # 구축 통계
        self.build_stats = {
            'start_time': datetime.now().isoformat(),
            'laws_processed': 0,
            'precedents_processed': 0,
            'constitutional_decisions_processed': 0,
            'legal_interpretations_processed': 0,
            'administrative_rules_processed': 0,
            'local_ordinances_processed': 0,
            'total_documents_processed': 0,
            'total_vectors_created': 0,
            'errors': []
        }
    
    def _process_law_document(self, doc: Dict) -> Dict:
        """법령 문서 처리"""
        try:
            law_name = doc.get('법령명한글', 'Unknown Law')
            article_content = []
            
            # 조문 내용 추출
            if '조문' in doc:
                for article in doc['조문']:
                    article_number = article.get('조문번호')
                    article_text = article.get('조문내용')
                    if article_number and article_text:
                        article_content.append(f"제{article_number}조 {article_text}")
            
            # 전체 내용 구성
            full_content = f"{law_name}\n\n" + "\n".join(article_content)
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('법령ID'),
                "document_type": "law",
                "title": law_name,
                "law_name": law_name,
                "promulgation_date": doc.get('공포일자'),
                "enforcement_date": doc.get('시행일자'),
                "department": doc.get('소관부처명'),
                "source_url": doc.get('법령상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing law document: {e}")
            return None
    
    def _process_precedent_document(self, doc: Dict) -> Dict:
        """판례 문서 처리"""
        try:
            case_name = doc.get('사건명', 'Unknown Case')
            judgment_summary = doc.get('판결요지', '')
            judgment_reason = doc.get('판시사항', '')
            
            # 전체 내용 구성
            full_content = f"{case_name}\n\n판결요지: {judgment_summary}\n\n판시사항: {judgment_reason}"
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('판례일련번호'),
                "document_type": "precedent",
                "title": case_name,
                "case_number": doc.get('사건번호'),
                "court_name": doc.get('법원명'),
                "decision_date": doc.get('선고일자'),
                "case_type": doc.get('사건종류명'),
                "source_url": doc.get('판례상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing precedent document: {e}")
            return None
    
    def _process_constitutional_decision_document(self, doc: Dict) -> Dict:
        """헌재결정례 문서 처리"""
        try:
            case_name = doc.get('사건명', 'Unknown Constitutional Case')
            decision_summary = doc.get('결정요지', '')
            decision_reason = doc.get('판시사항', '')
            
            # 전체 내용 구성
            full_content = f"{case_name}\n\n결정요지: {decision_summary}\n\n판시사항: {decision_reason}"
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('헌재결정례일련번호'),
                "document_type": "constitutional_decision",
                "title": case_name,
                "case_number": doc.get('사건번호'),
                "decision_date": doc.get('선고일자'),
                "case_type": doc.get('사건종류명'),
                "source_url": doc.get('헌재결정례상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing constitutional decision document: {e}")
            return None
    
    def _process_legal_interpretation_document(self, doc: Dict) -> Dict:
        """법령해석례 문서 처리"""
        try:
            interpretation_title = doc.get('해석례명', 'Unknown Interpretation')
            interpretation_content = doc.get('해석내용', '')
            
            # 전체 내용 구성
            full_content = f"{interpretation_title}\n\n{interpretation_content}"
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('법령해석례일련번호'),
                "document_type": "legal_interpretation",
                "title": interpretation_title,
                "interpretation_date": doc.get('해석일자'),
                "department": doc.get('소관부처명'),
                "source_url": doc.get('법령해석례상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing legal interpretation document: {e}")
            return None
    
    def _process_administrative_rule_document(self, doc: Dict) -> Dict:
        """행정규칙 문서 처리"""
        try:
            rule_name = doc.get('행정규칙명', 'Unknown Administrative Rule')
            rule_content = doc.get('행정규칙내용', '')
            
            # 전체 내용 구성
            full_content = f"{rule_name}\n\n{rule_content}"
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('행정규칙일련번호'),
                "document_type": "administrative_rule",
                "title": rule_name,
                "promulgation_date": doc.get('공포일자'),
                "enforcement_date": doc.get('시행일자'),
                "department": doc.get('소관부처명'),
                "source_url": doc.get('행정규칙상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing administrative rule document: {e}")
            return None
    
    def _process_local_ordinance_document(self, doc: Dict) -> Dict:
        """자치법규 문서 처리"""
        try:
            ordinance_name = doc.get('자치법규명', 'Unknown Local Ordinance')
            ordinance_content = doc.get('자치법규내용', '')
            
            # 전체 내용 구성
            full_content = f"{ordinance_name}\n\n{ordinance_content}"
            
            # 메타데이터 구성
            metadata = {
                "id": doc.get('자치법규일련번호'),
                "document_type": "local_ordinance",
                "title": ordinance_name,
                "promulgation_date": doc.get('공포일자'),
                "enforcement_date": doc.get('시행일자'),
                "local_government": doc.get('지방자치단체명'),
                "source_url": doc.get('자치법규상세링크'),
                "collected_at": doc.get('collected_at')
            }
            
            return {"text": full_content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing local ordinance document: {e}")
            return None
    
    def _process_documents_by_type(self, doc_type: str) -> bool:
        """특정 타입의 문서들을 처리"""
        try:
            logger.info(f"Processing {doc_type} documents...")
            
            # JSON 파일들 찾기
            json_files = list(self.raw_data_dir.glob(f"{doc_type}_*.json"))
            if not json_files:
                logger.warning(f"No {doc_type} JSON files found in {self.raw_data_dir}")
                return False
            
            all_texts = []
            all_metadatas = []
            processed_count = 0
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, list):
                        data = [data]
                    
                    for doc in data:
                        processed_doc = None
                        
                        # 문서 타입에 따른 처리
                        if doc_type == "laws":
                            processed_doc = self._process_law_document(doc)
                        elif doc_type == "precedents":
                            processed_doc = self._process_precedent_document(doc)
                        elif doc_type == "constitutional_decisions":
                            processed_doc = self._process_constitutional_decision_document(doc)
                        elif doc_type == "legal_interpretations":
                            processed_doc = self._process_legal_interpretation_document(doc)
                        elif doc_type == "administrative_rules":
                            processed_doc = self._process_administrative_rule_document(doc)
                        elif doc_type == "local_ordinances":
                            processed_doc = self._process_local_ordinance_document(doc)
                        
                        if processed_doc:
                            all_texts.append(processed_doc['text'])
                            all_metadatas.append(processed_doc['metadata'])
                            processed_count += 1
                            
                            # SQLite에도 저장
                            self._save_to_sqlite(processed_doc)
                
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {e}")
                    continue
            
            # 벡터DB에 추가
            if all_texts and all_metadatas:
                self.vector_store.add_documents(all_texts, all_metadatas)
                logger.info(f"Added {len(all_texts)} {doc_type} documents to vector store")
                
                # 통계 업데이트
                if doc_type == "laws":
                    self.build_stats['laws_processed'] += processed_count
                elif doc_type == "precedents":
                    self.build_stats['precedents_processed'] += processed_count
                elif doc_type == "constitutional_decisions":
                    self.build_stats['constitutional_decisions_processed'] += processed_count
                elif doc_type == "legal_interpretations":
                    self.build_stats['legal_interpretations_processed'] += processed_count
                elif doc_type == "administrative_rules":
                    self.build_stats['administrative_rules_processed'] += processed_count
                elif doc_type == "local_ordinances":
                    self.build_stats['local_ordinances_processed'] += processed_count
                
                self.build_stats['total_documents_processed'] += processed_count
                self.build_stats['total_vectors_created'] += len(all_texts)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing {doc_type} documents: {e}")
            self.build_stats['errors'].append(f"{doc_type} processing error: {e}")
            return False
    
    def _save_to_sqlite(self, processed_doc: Dict):
        """SQLite에 문서 저장"""
        try:
            doc_id = processed_doc['metadata']['id']
            doc_type = processed_doc['metadata']['document_type']
            title = processed_doc['metadata']['title']
            content = processed_doc['text']
            source_url = processed_doc['metadata'].get('source_url')
            
            doc_data = {
                "id": doc_id,
                "document_type": doc_type,
                "title": title,
                "content": content,
                "source_url": source_url
            }
            
            # 문서 타입별 메타데이터 구성
            law_meta = None
            prec_meta = None
            
            if doc_type == "law":
                law_meta = {
                    "law_name": processed_doc['metadata'].get('law_name'),
                    "promulgation_date": processed_doc['metadata'].get('promulgation_date'),
                    "enforcement_date": processed_doc['metadata'].get('enforcement_date'),
                    "department": processed_doc['metadata'].get('department')
                }
            elif doc_type == "precedent":
                prec_meta = {
                    "case_number": processed_doc['metadata'].get('case_number'),
                    "court_name": processed_doc['metadata'].get('court_name'),
                    "decision_date": processed_doc['metadata'].get('decision_date'),
                    "case_type": processed_doc['metadata'].get('case_type')
                }
            
            # SQLite에 저장
            self.db_manager.add_document(doc_data, law_meta, prec_meta)
            
        except Exception as e:
            logger.error(f"Error saving to SQLite: {e}")
    
    def build_vector_db(self) -> bool:
        """벡터DB 구축"""
        try:
            logger.info("Starting vector DB build process...")
            
            # 처리할 문서 타입들
            doc_types = [
                "laws", "precedents", "constitutional_decisions", 
                "legal_interpretations", "administrative_rules", "local_ordinances"
            ]
            
            success_count = 0
            for doc_type in doc_types:
                if self._process_documents_by_type(doc_type):
                    success_count += 1
            
            # 벡터 인덱스 저장
            self.vector_store.save_index()
            
            # 최종 통계 생성
            self._generate_build_report()
            
            logger.info(f"Vector DB build completed: {success_count}/{len(doc_types)} document types processed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error in vector DB build: {e}")
            return False
    
    def build_specific_type(self, doc_type: str) -> bool:
        """특정 타입의 벡터DB 구축"""
        try:
            logger.info(f"Building vector DB for {doc_type}...")
            
            if self._process_documents_by_type(doc_type):
                # 벡터 인덱스 저장
                self.vector_store.save_index()
                logger.info(f"Vector DB build completed for {doc_type}")
                return True
            else:
                logger.error(f"Failed to build vector DB for {doc_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error building vector DB for {doc_type}: {e}")
            return False
    
    def _generate_build_report(self):
        """구축 보고서 생성"""
        try:
            self.build_stats['end_time'] = datetime.now().isoformat()
            self.build_stats['total_duration'] = (
                datetime.fromisoformat(self.build_stats['end_time']) - 
                datetime.fromisoformat(self.build_stats['start_time'])
            ).total_seconds()
            
            # 벡터 스토어 통계
            vector_stats = self.vector_store.get_stats()
            self.build_stats['vector_stats'] = vector_stats
            
            # 보고서 저장
            report_file = self.raw_data_dir / "vector_db_build_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.build_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Vector DB build report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating build report: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LawFirmAI Vector DB Builder")
    parser.add_argument("--mode", type=str, choices=["build", "laws", "precedents", "constitutional", "interpretations", "administrative", "local", "multiple"], 
                        default="build", help="Build mode")
    parser.add_argument("--raw_dir", type=str, default="./data/raw", help="Raw data directory")
    parser.add_argument("--types", type=str, nargs="+", 
                        choices=["laws", "precedents", "constitutional", "interpretations", "administrative", "local"],
                        help="Specific data types to build (use with --mode multiple)")
    
    args = parser.parse_args()
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 벡터DB 구축 실행
    builder = VectorDBBuilder(args.raw_dir)
    
    success = False
    if args.mode == "build":
        success = builder.build_vector_db()
    elif args.mode == "multiple":
        # 여러 타입 벡터DB 구축
        if not args.types:
            logger.error("--types parameter is required when using --mode multiple")
            logger.info("Example: python scripts/build_vector_db.py --mode multiple --types laws precedents")
            return
        
        # 데이터 타입 매핑
        type_mapping = {
            "laws": "laws",
            "precedents": "precedents",
            "constitutional": "constitutional_decisions",
            "interpretations": "legal_interpretations",
            "administrative": "administrative_rules",
            "local": "local_ordinances"
        }
        
        success_count = 0
        for data_type in args.types:
            logger.info(f"Building vector DB for {data_type}...")
            
            mapped_type = type_mapping.get(data_type)
            if not mapped_type:
                logger.error(f"Unknown data type: {data_type}")
                continue
            
            if builder.build_specific_type(mapped_type):
                success_count += 1
                logger.info(f"{data_type} vector DB build completed successfully!")
            else:
                logger.error(f"{data_type} vector DB build failed!")
        
        success = success_count > 0
        logger.info(f"Multiple types vector DB build completed: {success_count}/{len(args.types)} types successful")
        
    elif args.mode == "laws":
        success = builder.build_specific_type("laws")
    elif args.mode == "precedents":
        success = builder.build_specific_type("precedents")
    elif args.mode == "constitutional":
        success = builder.build_specific_type("constitutional_decisions")
    elif args.mode == "interpretations":
        success = builder.build_specific_type("legal_interpretations")
    elif args.mode == "administrative":
        success = builder.build_specific_type("administrative_rules")
    elif args.mode == "local":
        success = builder.build_specific_type("local_ordinances")
    
    if success:
        logger.info("Vector DB build completed successfully!")
    else:
        logger.error("Vector DB build failed!")


if __name__ == "__main__":
    main()