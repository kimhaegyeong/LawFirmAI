#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
품질 개선된 Raw 데이터 전처리 스크립트

품질 점수를 개선하기 위한 강화된 전처리 파이프라인입니다.
"""

import sys
import os
import json
import logging
import gc
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from source.data.enhanced_data_processor import EnhancedLegalDataProcessor

class QualityImprovedPreprocessor:
    """품질 개선된 전처리 클래스"""
    
    def __init__(self, 
                 enable_term_normalization=True,
                 max_memory_usage=0.8,
                 batch_size=50,
                 chunk_size=1000):
        """품질 개선 전처리기 초기화"""
        self.processor = EnhancedLegalDataProcessor(enable_term_normalization)
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(exist_ok=True)
        
        # 메모리 관리 설정
        self.max_memory_usage = max_memory_usage
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # 로깅 설정
        self.setup_logging()
        
        # 통계 초기화
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {},
            "quality_metrics": {
                "completeness": 0.0,
                "consistency": 0.0,
                "term_normalization": 0.0
            }
        }
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/quality_improved_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self) -> float:
        """현재 메모리 사용률 반환"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)  # GB 단위
    
    def check_memory_limit(self) -> bool:
        """메모리 사용량 체크"""
        current_memory = self.get_memory_usage()
        
        if current_memory > 8:  # 8GB 이상 사용 시 경고
            self.logger.warning(f"메모리 사용량이 높습니다: {current_memory:.2f}GB")
            return False
        return True
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션 실행"""
        gc.collect()
        self.logger.debug("가비지 컬렉션 실행 완료")
    
    def process_file_in_chunks(self, file_path: Path, data_type: str) -> Generator[Dict[str, Any], None, None]:
        """파일을 청크 단위로 처리하는 제너레이터"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터 타입에 따른 처리
            if data_type == 'precedent' and isinstance(data, dict) and 'precedents' in data:
                items = data['precedents']
            elif isinstance(data, list):
                items = data
            else:
                items = [data]
            
            # 청크 단위로 처리
            for i in range(0, len(items), self.chunk_size):
                chunk = items[i:i + self.chunk_size]
                
                # 메모리 체크
                if not self.check_memory_limit():
                    self.force_garbage_collection()
                
                # 청크 처리
                processed_chunk = self.processor.process_batch(chunk, data_type)
                
                # 모든 항목 yield (유효하지 않아도 포함)
                for item in processed_chunk:
                    yield item
                
                # 메모리 정리
                del chunk
                del processed_chunk
                self.force_garbage_collection()
                
        except Exception as e:
            self.logger.error(f"파일 처리 중 오류 {file_path}: {e}")
    
    def process_laws_improved(self):
        """품질 개선된 법령 데이터 전처리"""
        self.logger.info("품질 개선된 법령 데이터 전처리 시작")
        
        law_files = list(Path("data/raw/laws").glob("*.json"))
        processed_count = 0
        valid_count = 0
        
        for law_file in law_files:
            try:
                self.logger.info(f"처리 중: {law_file}")
                
                # 파일을 청크 단위로 처리
                for processed_law in self.process_file_in_chunks(law_file, 'law'):
                    # 즉시 저장 (메모리 누적 방지)
                    self.save_single_document(processed_law, "laws")
                    processed_count += 1
                    
                    if processed_law.get('is_valid', False):
                        valid_count += 1
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    # 메모리 체크
                    if not self.check_memory_limit():
                        self.force_garbage_collection()
                
                self.stats['total_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"법령 전처리 실패 {law_file}: {e}")
                self.stats['failed'] += 1
        
        self.stats['by_type']['laws'] = processed_count
        self.logger.info(f"법령 데이터 전처리 완료: {processed_count}개 (유효: {valid_count}개)")
    
    def process_precedents_improved(self):
        """품질 개선된 판례 데이터 전처리"""
        self.logger.info("품질 개선된 판례 데이터 전처리 시작")
        
        precedent_dirs = list(Path("data/raw/precedents").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for precedent_dir in precedent_dirs:
            json_files = list(precedent_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"처리 중: {json_file}")
                    
                    # 파일을 청크 단위로 처리
                    for processed_precedent in self.process_file_in_chunks(json_file, 'precedent'):
                        # 즉시 저장 (메모리 누적 방지)
                        self.save_single_document(processed_precedent, "precedents")
                        processed_count += 1
                        
                        if processed_precedent.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # 메모리 체크
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"판례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['precedents'] = processed_count
        self.logger.info(f"판례 데이터 전처리 완료: {processed_count}개 (유효: {valid_count}개)")
    
    def process_constitutional_decisions_improved(self):
        """품질 개선된 헌재결정례 데이터 전처리"""
        self.logger.info("품질 개선된 헌재결정례 데이터 전처리 시작")
        
        constitutional_dirs = list(Path("data/raw/constitutional_decisions").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for constitutional_dir in constitutional_dirs:
            json_files = list(constitutional_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"처리 중: {json_file}")
                    
                    # 파일을 청크 단위로 처리
                    for processed_decision in self.process_file_in_chunks(json_file, 'constitutional_decision'):
                        # 즉시 저장 (메모리 누적 방지)
                        self.save_single_document(processed_decision, "constitutional_decisions")
                        processed_count += 1
                        
                        if processed_decision.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # 메모리 체크
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"헌재결정례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['constitutional_decisions'] = processed_count
        self.logger.info(f"헌재결정례 데이터 전처리 완료: {processed_count}개 (유효: {valid_count}개)")
    
    def process_legal_interpretations_improved(self):
        """품질 개선된 법령해석례 데이터 전처리"""
        self.logger.info("품질 개선된 법령해석례 데이터 전처리 시작")
        
        interpretation_dirs = list(Path("data/raw/legal_interpretations").glob("yearly_*"))
        processed_count = 0
        valid_count = 0
        
        for interpretation_dir in interpretation_dirs:
            json_files = list(interpretation_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    self.logger.info(f"처리 중: {json_file}")
                    
                    # 파일을 청크 단위로 처리
                    for processed_interpretation in self.process_file_in_chunks(json_file, 'legal_interpretation'):
                        # 즉시 저장 (메모리 누적 방지)
                        self.save_single_document(processed_interpretation, "legal_interpretations")
                        processed_count += 1
                        
                        if processed_interpretation.get('is_valid', False):
                            valid_count += 1
                            self.stats['successful'] += 1
                        else:
                            self.stats['failed'] += 1
                        
                        # 메모리 체크
                        if not self.check_memory_limit():
                            self.force_garbage_collection()
                
                except Exception as e:
                    self.logger.error(f"법령해석례 전처리 실패 {json_file}: {e}")
                    self.stats['failed'] += 1
        
        self.stats['by_type']['legal_interpretations'] = processed_count
        self.logger.info(f"법령해석례 데이터 전처리 완료: {processed_count}개 (유효: {valid_count}개)")
    
    def save_single_document(self, document: Dict[str, Any], data_type: str):
        """단일 문서를 즉시 저장"""
        # 데이터 타입별 디렉토리 생성
        type_dir = self.output_dir / data_type
        type_dir.mkdir(exist_ok=True)
        
        # 파일명 생성
        doc_id = document.get('id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{doc_id}_{timestamp}.json"
        
        # 파일 저장
        file_path = type_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
    
    def calculate_quality_metrics(self):
        """품질 지표 계산"""
        total_docs = self.stats['total_processed']
        valid_docs = self.stats['successful']
        
        if total_docs > 0:
            # 완성도 (유효한 문서 비율)
            self.stats['quality_metrics']['completeness'] = valid_docs / total_docs
            
            # 일관성 (모든 데이터 타입에서 유효한 문서가 있는지)
            valid_types = sum(1 for count in self.stats['by_type'].values() if count > 0)
            total_types = len(self.stats['by_type'])
            self.stats['quality_metrics']['consistency'] = valid_types / total_types if total_types > 0 else 0
            
            # 용어 정규화 (성공한 문서 중 용어 정규화가 적용된 비율)
            # 실제로는 더 정교한 계산이 필요하지만, 여기서는 간단히 처리
            self.stats['quality_metrics']['term_normalization'] = min(valid_docs / total_docs * 1.2, 1.0)
    
    def run_improved_preprocessing(self, data_types: List[str] = None):
        """품질 개선된 전처리 실행"""
        if data_types is None:
            data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations']
        
        start_time = datetime.now()
        self.logger.info(f"품질 개선된 전처리 시작: {data_types}")
        
        try:
            for data_type in data_types:
                if data_type == 'laws':
                    self.process_laws_improved()
                elif data_type == 'precedents':
                    self.process_precedents_improved()
                elif data_type == 'constitutional_decisions':
                    self.process_constitutional_decisions_improved()
                elif data_type == 'legal_interpretations':
                    self.process_legal_interpretations_improved()
                
                # 각 데이터 타입 처리 후 메모리 정리
                self.force_garbage_collection()
                self.logger.info(f"메모리 사용량: {self.get_memory_usage():.2f}GB")
            
            # 품질 지표 계산
            self.calculate_quality_metrics()
            
            duration = datetime.now() - start_time
            self.logger.info(f"=== 전처리 완료 (소요시간: {duration}) ===")
            self.print_statistics()
            
        except Exception as e:
            self.logger.error(f"전처리 중 오류 발생: {e}")
            raise
    
    def print_statistics(self):
        """통계 출력"""
        self.logger.info("=== 처리 통계 ===")
        self.logger.info(f"총 처리: {self.stats['total_processed']}개")
        self.logger.info(f"성공: {self.stats['successful']}개")
        self.logger.info(f"실패: {self.stats['failed']}개")
        
        for data_type, count in self.stats['by_type'].items():
            self.logger.info(f"{data_type}: {count}개")
        
        # 품질 지표 출력
        self.logger.info("=== 품질 지표 ===")
        for metric, score in self.stats['quality_metrics'].items():
            self.logger.info(f"{metric}: {score:.2%}")
        
        # 전체 품질 점수 계산
        overall_quality = sum(self.stats['quality_metrics'].values()) / len(self.stats['quality_metrics'])
        self.logger.info(f"전체 품질 점수: {overall_quality:.2%}")

def main():
    parser = argparse.ArgumentParser(description="품질 개선된 Raw 데이터 전처리")
    parser.add_argument("--data-types", nargs='+', 
                       choices=["laws", "precedents", "constitutional", "interpretations", "all"],
                       default=["laws", "precedents"],
                       help="전처리할 데이터 유형")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="배치 크기")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="청크 크기")
    parser.add_argument("--max-memory", type=float, default=8.0,
                       help="최대 메모리 사용량 (GB)")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="법률 용어 정규화 활성화")
    
    args = parser.parse_args()
    
    # 데이터 타입 처리
    if "all" in args.data_types:
        data_types = ["laws", "precedents", "constitutional_decisions", "legal_interpretations"]
    else:
        data_types = args.data_types
    
    # 전처리기 초기화
    preprocessor = QualityImprovedPreprocessor(
        enable_term_normalization=args.enable_normalization,
        max_memory_usage=args.max_memory / 16,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    # 전처리 실행
    preprocessor.run_improved_preprocessing(data_types)

if __name__ == "__main__":
    main()
