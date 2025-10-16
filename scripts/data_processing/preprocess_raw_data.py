#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw 데이터 전처리 메인 스크립트

국가법령정보센터 OpenAPI를 통해 수집된 raw 데이터를 전처리하여
벡터 데이터베이스 구축과 RAG 시스템에 적합한 형태로 변환합니다.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from source.data.data_processor import LegalDataProcessor
from source.data.legal_term_normalizer import LegalTermNormalizer

class RawDataPreprocessingPipeline:
    """Raw 데이터 전처리 통합 파이프라인"""
    
    def __init__(self, enable_term_normalization=True):
        """전처리 파이프라인 초기화"""
        self.processor = LegalDataProcessor(enable_term_normalization)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 통계 초기화
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {}
        }
        
        # 데이터 유형별 우선순위
        self.preprocessing_priority = {
            "laws": 1,           # 법령 (최우선)
            "precedents": 2,     # 판례 (High)
            "constitutional_decisions": 3,  # 헌재결정례 (Medium)
            "legal_interpretations": 4,     # 법령해석례 (Medium)
            "legal_terms": 5,    # 법률 용어 (Medium)
            "administrative_rules": 6,      # 행정규칙 (Low)
            "local_ordinances": 7,          # 자치법규 (Low)
            "committee_decisions": 8,       # 위원회결정문 (Low)
            "treaties": 9        # 조약 (Low)
        }
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_preprocessing(self):
        """전체 전처리 파이프라인 실행"""
        self.logger.info("=== Raw 데이터 전처리 시작 ===")
        
        start_time = datetime.now()
        
        try:
            # Phase 1: 핵심 데이터 전처리
            self.logger.info("Phase 1: 핵심 데이터 전처리 시작")
            self.process_laws()
            self.process_precedents()
            
            # Phase 2: 확장 데이터 전처리
            self.logger.info("Phase 2: 확장 데이터 전처리 시작")
            self.process_constitutional_decisions()
            self.process_legal_interpretations()
            self.process_legal_terms()
            
            # Phase 3: 품질 검증 및 통합
            self.logger.info("Phase 3: 품질 검증 및 통합")
            self.validate_processed_data()
            self.consolidate_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"=== 전처리 완료 (소요시간: {duration}) ===")
            self.print_statistics()
            
        except Exception as e:
            self.logger.error(f"전처리 중 오류 발생: {e}")
            raise
    
    def process_laws(self):
        """법령 데이터 전처리"""
        self.logger.info("법령 데이터 전처리 시작")
        
        law_files = list(Path("data/raw/laws").glob("*.json"))
        processed_laws = []
        
        for law_file in law_files:
            try:
                self.logger.info(f"처리 중: {law_file}")
                with open(law_file, 'r', encoding='utf-8') as f:
                    law_data = json.load(f)
                
                # 기본 정보 확인
                basic_info = law_data.get('basic_info', {})
                self.logger.info(f"  - 법령명: {basic_info.get('name', 'N/A')}")
                self.logger.info(f"  - ID: {basic_info.get('id', 'N/A')}")
                
                processed_law = self.processor.process_law_data(law_data)
                
                # 처리 결과 확인
                if processed_law.get('status') == 'success':
                    content_length = len(processed_law.get('full_content', ''))
                    chunks_count = len(processed_law.get('chunks', []))
                    self.logger.info(f"  - 처리 성공: 내용 길이 {content_length}자, 청크 {chunks_count}개")
                    processed_laws.append(processed_law)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                    self.logger.warning(f"법령 전처리 실패: {law_file} - {processed_law.get('error', 'Unknown error')}")
                
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"법령 전처리 실패 {law_file}: {e}")
                self.stats['failed'] += 1
                self.stats['total_processed'] += 1
        
        # 결과 저장
        self.save_processed_data(processed_laws, "laws")
        self.stats['by_type']['laws'] = len(processed_laws)
        
        self.logger.info(f"법령 데이터 전처리 완료: {len(processed_laws)}개")
    
    def process_precedents(self):
        """판례 데이터 전처리"""
        self.logger.info("판례 데이터 전처리 시작")
        
        precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
        all_processed_precedents = []
        
        for precedent_dir in precedent_dirs:
            json_files = list(precedent_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"처리 중: {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # 판례 데이터는 precedents 배열 안에 있음
                    if isinstance(file_data, dict) and 'precedents' in file_data:
                        precedents_list = file_data['precedents']
                        self.logger.info(f"  - 판례 수: {len(precedents_list)}개")
                        
                        processed_precedents = self.processor.process_batch(
                            precedents_list, 'precedent'
                        )
                        
                        # 처리 결과 로깅
                        success_count = len([p for p in processed_precedents if p.get('status') == 'success'])
                        self.logger.info(f"  - 처리 성공: {success_count}/{len(precedents_list)}개")
                        
                    elif isinstance(file_data, list):
                        # 직접 배열 형태인 경우
                        processed_precedents = self.processor.process_batch(
                            file_data, 'precedent'
                        )
                    else:
                        # 단일 판례 데이터 처리
                        processed_precedents = [self.processor.process_precedent_data(file_data)]
                    
                    # 성공한 것만 추가
                    successful_precedents = [p for p in processed_precedents if p.get('status') == 'success']
                    all_processed_precedents.extend(successful_precedents)
                    
                    self.stats['total_processed'] += len(processed_precedents)
                    self.stats['successful'] += len(successful_precedents)
                    self.stats['failed'] += len(processed_precedents) - len(successful_precedents)
                    
                except Exception as e:
                    self.logger.error(f"판례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # 결과 저장
        self.save_processed_data(all_processed_precedents, "precedents")
        self.stats['by_type']['precedents'] = len(all_processed_precedents)
        
        self.logger.info(f"판례 데이터 전처리 완료: {len(all_processed_precedents)}개")
    
    def process_constitutional_decisions(self):
        """헌재결정례 데이터 전처리"""
        self.logger.info("헌재결정례 데이터 전처리 시작")
        
        constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
        all_processed_decisions = []
        
        for constitutional_dir in constitutional_dirs:
            json_files = list(constitutional_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        decision_data = json.load(f)
                    
                    if isinstance(decision_data, list):
                        processed_decisions = self.processor.process_batch(
                            decision_data, 'constitutional_decision'
                        )
                    else:
                        processed_decisions = [self.processor.process_constitutional_decision_data(decision_data)]
                    
                    # 성공한 것만 추가
                    successful_decisions = [p for p in processed_decisions if p.get('status') == 'success']
                    all_processed_decisions.extend(successful_decisions)
                    
                    self.stats['total_processed'] += len(processed_decisions)
                    self.stats['successful'] += len(successful_decisions)
                    self.stats['failed'] += len(processed_decisions) - len(successful_decisions)
                    
                except Exception as e:
                    self.logger.error(f"헌재결정례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # 결과 저장
        self.save_processed_data(all_processed_decisions, "constitutional_decisions")
        self.stats['by_type']['constitutional_decisions'] = len(all_processed_decisions)
        
        self.logger.info(f"헌재결정례 데이터 전처리 완료: {len(all_processed_decisions)}개")
    
    def process_legal_interpretations(self):
        """법령해석례 데이터 전처리"""
        self.logger.info("법령해석례 데이터 전처리 시작")
        
        interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
        all_processed_interpretations = []
        
        for interpretation_dir in interpretation_dirs:
            json_files = list(interpretation_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        interpretation_data = json.load(f)
                    
                    if isinstance(interpretation_data, list):
                        processed_interpretations = self.processor.process_batch(
                            interpretation_data, 'legal_interpretation'
                        )
                    else:
                        processed_interpretations = [self.processor.process_legal_interpretation_data(interpretation_data)]
                    
                    # 성공한 것만 추가
                    successful_interpretations = [p for p in processed_interpretations if p.get('status') == 'success']
                    all_processed_interpretations.extend(successful_interpretations)
                    
                    self.stats['total_processed'] += len(processed_interpretations)
                    self.stats['successful'] += len(successful_interpretations)
                    self.stats['failed'] += len(processed_interpretations) - len(successful_interpretations)
                    
                except Exception as e:
                    self.logger.error(f"법령해석례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # 결과 저장
        self.save_processed_data(all_processed_interpretations, "legal_interpretations")
        self.stats['by_type']['legal_interpretations'] = len(all_processed_interpretations)
        
        self.logger.info(f"법령해석례 데이터 전처리 완료: {len(all_processed_interpretations)}개")
    
    def process_legal_terms(self):
        """법률 용어 데이터 전처리"""
        self.logger.info("법률 용어 데이터 전처리 시작")
        
        term_dirs = list(Path("data/raw/legal_terms").glob("session_*"))
        all_processed_terms = []
        
        for term_dir in term_dirs:
            json_files = list(term_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        term_data = json.load(f)
                    
                    # 용어 데이터는 특별한 처리가 필요할 수 있음
                    processed_terms = self.process_legal_term_data(term_data)
                    all_processed_terms.extend(processed_terms)
                    
                    self.stats['total_processed'] += len(processed_terms)
                    self.stats['successful'] += len(processed_terms)
                    
                except Exception as e:
                    self.logger.error(f"법률 용어 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
                    self.stats['total_processed'] += 1
        
        # 결과 저장
        self.save_processed_data(all_processed_terms, "legal_terms")
        self.stats['by_type']['legal_terms'] = len(all_processed_terms)
        
        self.logger.info(f"법률 용어 데이터 전처리 완료: {len(all_processed_terms)}개")
    
    def process_legal_term_data(self, term_data):
        """법률 용어 데이터 처리"""
        processed_terms = []
        
        if isinstance(term_data, dict) and 'terms' in term_data:
            for term in term_data['terms']:
                processed_term = {
                    'id': term.get('term_sequence_number', ''),
                    'term_name_korean': term.get('term_name_korean', ''),
                    'term_name_chinese': term.get('term_name_chinese', ''),
                    'definition': term.get('definition', ''),
                    'source': term.get('source', ''),
                    'category': 'legal_term',
                    'status': 'success',
                    'processed_at': datetime.now().isoformat()
                }
                processed_terms.append(processed_term)
        elif isinstance(term_data, list):
            for term in term_data:
                processed_term = {
                    'id': term.get('term_sequence_number', ''),
                    'term_name_korean': term.get('term_name_korean', ''),
                    'term_name_chinese': term.get('term_name_chinese', ''),
                    'definition': term.get('definition', ''),
                    'source': term.get('source', ''),
                    'category': 'legal_term',
                    'status': 'success',
                    'processed_at': datetime.now().isoformat()
                }
                processed_terms.append(processed_term)
        
        return processed_terms
    
    def save_processed_data(self, data, data_type):
        """전처리된 데이터 저장"""
        output_dir = self.output_dir / data_type
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{data_type}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"{data_type} 데이터 저장 완료: {output_file}")
    
    def validate_processed_data(self):
        """전처리된 데이터 검증"""
        self.logger.info("전처리된 데이터 검증 시작")
        
        validation_results = {}
        
        for data_type in self.stats['by_type'].keys():
            validation_results[data_type] = self.validate_data_type(data_type)
        
        # 검증 결과 저장
        validation_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"데이터 검증 완료: {validation_file}")
    
    def validate_data_type(self, data_type):
        """특정 데이터 유형 검증"""
        data_dir = self.output_dir / data_type
        if not data_dir.exists():
            return {
                "total_documents": 0,
                "validation_passed": False,
                "issues": ["데이터 디렉토리가 존재하지 않음"]
            }
        
        json_files = list(data_dir.glob("*.json"))
        total_documents = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_documents += len(data)
                else:
                    total_documents += 1
            except Exception as e:
                self.logger.error(f"검증 중 오류 {json_file}: {e}")
        
        return {
            "total_documents": total_documents,
            "validation_passed": total_documents > 0,
            "issues": [] if total_documents > 0 else ["문서가 없음"]
        }
    
    def consolidate_results(self):
        """결과 통합"""
        self.logger.info("전처리 결과 통합 시작")
        
        # 통합 인덱스 생성
        consolidated_index = {
            "metadata": {
                "total_processed": self.stats['total_processed'],
                "successful": self.stats['successful'],
                "failed": self.stats['failed'],
                "by_type": self.stats['by_type'],
                "processed_at": datetime.now().isoformat()
            },
            "data_types": list(self.stats['by_type'].keys()),
            "file_locations": {}
        }
        
        # 파일 위치 정보 추가
        for data_type in self.stats['by_type'].keys():
            data_dir = self.output_dir / data_type
            if data_dir.exists():
                files = list(data_dir.glob("*.json"))
                consolidated_index["file_locations"][data_type] = [str(f) for f in files]
        
        # 통합 인덱스 저장
        index_file = self.output_dir / "consolidated_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_index, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"결과 통합 완료: {index_file}")
    
    def print_statistics(self):
        """통계 출력"""
        self.logger.info("=== 전처리 통계 ===")
        self.logger.info(f"총 처리: {self.stats['total_processed']}개")
        self.logger.info(f"성공: {self.stats['successful']}개")
        self.logger.info(f"실패: {self.stats['failed']}개")
        
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_processed'] * 100
            self.logger.info(f"성공률: {success_rate:.2f}%")
        
        self.logger.info("=== 데이터 유형별 통계 ===")
        for data_type, count in self.stats['by_type'].items():
            self.logger.info(f"{data_type}: {count}개")
    
    def process_specific_type(self, data_type):
        """특정 데이터 유형만 처리"""
        if data_type == "laws":
            self.process_laws()
        elif data_type == "precedents":
            self.process_precedents()
        elif data_type == "constitutional":
            self.process_constitutional_decisions()
        elif data_type == "interpretations":
            self.process_legal_interpretations()
        elif data_type == "terms":
            self.process_legal_terms()
        else:
            self.logger.error(f"알 수 없는 데이터 유형: {data_type}")
    
    def dry_run(self, data_type):
        """드라이런 모드 - 실제 처리 없이 계획만 출력"""
        self.logger.info("=== 드라이런 모드 ===")
        
        if data_type == "all":
            data_types = ["laws", "precedents", "constitutional", "interpretations", "terms"]
        else:
            data_types = [data_type]
        
        for dt in data_types:
            self.logger.info(f"처리 예정: {dt}")
            
            if dt == "laws":
                law_files = list(Path("data/raw/laws").glob("*.json"))
                self.logger.info(f"  - 법령 파일: {len(law_files)}개")
            elif dt == "precedents":
                precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in precedent_dirs)
                self.logger.info(f"  - 판례 폴더: {len(precedent_dirs)}개")
                self.logger.info(f"  - 판례 파일: {total_files}개")
            elif dt == "constitutional":
                constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in constitutional_dirs)
                self.logger.info(f"  - 헌재결정례 폴더: {len(constitutional_dirs)}개")
                self.logger.info(f"  - 헌재결정례 파일: {total_files}개")
            elif dt == "interpretations":
                interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in interpretation_dirs)
                self.logger.info(f"  - 법령해석례 폴더: {len(interpretation_dirs)}개")
                self.logger.info(f"  - 법령해석례 파일: {total_files}개")
            elif dt == "terms":
                term_dirs = list(Path("data/raw/legal_terms").glob("session_*"))
                total_files = sum(len(list(d.glob("*.json"))) for d in term_dirs)
                self.logger.info(f"  - 법률 용어 폴더: {len(term_dirs)}개")
                self.logger.info(f"  - 법률 용어 파일: {total_files}개")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Raw 데이터 전처리")
    parser.add_argument("--data-type", default="all",
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       help="전처리할 데이터 유형")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="법률 용어 정규화 활성화")
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 처리 없이 계획만 출력")
    
    args = parser.parse_args()
    
    pipeline = RawDataPreprocessingPipeline(args.enable_normalization)
    
    if args.dry_run:
        pipeline.dry_run(args.data_type)
    else:
        if args.data_type == "all":
            pipeline.run_full_preprocessing()
        else:
            pipeline.process_specific_type(args.data_type)

if __name__ == "__main__":
    main()
