#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
증분 벡터 임베딩 생성기

기존 FAISS 인덱스를 로드하고 새로운 문서만 임베딩하여 기존 인덱스에 추가하는 시스템입니다.
ko-sroberta-multitask 모델을 사용하여 증분 업데이트를 수행합니다.
"""

import logging
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager # 파일 처리 이력 추적용
from scripts.data_processing.auto_data_detector import AutoDataDetector # 파일 해시 생성용

logger = logging.getLogger(__name__)

class IncrementalVectorBuilder:
    """증분 벡터 임베딩 생성기"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat",
                 processed_data_base_path: str = "data/processed/assembly",
                 embedding_output_path: str = "data/embeddings/ml_enhanced_ko_sroberta"):
        """
        증분 벡터 빌더 초기화
        
        Args:
            model_name: 사용할 Sentence-BERT 모델명
            dimension: 벡터 차원
            index_type: FAISS 인덱스 타입
            processed_data_base_path: 전처리된 데이터가 저장된 기본 디렉토리
            embedding_output_path: 임베딩 및 FAISS 인덱스 저장 경로
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        self.processed_data_base_path = Path(processed_data_base_path)
        self.embedding_output_path = Path(embedding_output_path)
        self.embedding_output_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=dimension,
            index_type=index_type
        )
        self.db_manager = DatabaseManager()
        self.auto_detector = AutoDataDetector() # 파일 해시 계산용
        
        # 기존 FAISS 인덱스 로드 시도
        self.vector_store.load_index(self.embedding_output_path)
        
        self.stats = {
            'total_files_scanned': 0,
            'new_files_for_embedding': 0,
            'successfully_embedded_files': 0,
            'failed_embedding_files': 0,
            'skipped_already_embedded': 0,
            'total_chunks_added': 0,
            'processing_time': 0,
            'errors': []
        }
        
        logger.info(f"IncrementalVectorBuilder initialized with model: {model_name}")
        logger.info(f"Existing FAISS index loaded: {self.vector_store.index is not None}")

    def build_incremental_embeddings(self, data_type: str = "law_only", batch_size: int = 100) -> Dict[str, Any]:
        """
        새로 전처리된 데이터로부터 증분 벡터 임베딩 생성
        
        Args:
            data_type: 임베딩할 특정 데이터 유형 (예: 'law_only'). 'all'이면 모든 유형 처리.
            batch_size: 임베딩 배치 처리 크기
            
        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        logger.info(f"Starting incremental vector embedding for data type: {data_type}")
        start_time = datetime.now()
        
        # 데이터베이스에서 'completed' 상태의 전처리된 파일 목록 가져오기
        processed_files_info = self.db_manager.get_processed_files_by_type(data_type, status="completed")
        
        files_to_embed = []
        for file_info in processed_files_info:
            processed_file_path = Path(file_info['file_path'])
            
            # 원본 raw 파일 경로를 기반으로 ml_enhanced_*.json 파일 경로 추론
            # 예: data/raw/assembly/law_only/20251016/law_only_page_001_...json
            # -> data/processed/assembly/law_only/20251016/ml_enhanced_law_only_page_001_...json
            relative_path = processed_file_path.relative_to(self.auto_detector.raw_data_base_path)
            ml_enhanced_file_name = f"ml_enhanced_{processed_file_path.stem}.json"
            ml_enhanced_file_path = self.processed_data_base_path / relative_path.parent / ml_enhanced_file_name
            
            if not ml_enhanced_file_path.exists():
                logger.warning(f"ML enhanced file not found for {processed_file_path}. Skipping.")
                continue
            
            # 이미 임베딩된 파일인지 확인 (processed_files 테이블에서 'embedded' 상태로 추적)
            # 여기서는 processed_files 테이블에 'embedded' 상태를 추가하지 않고,
            # 단순히 해당 파일이 벡터 스토어에 이미 존재하는지 여부로 판단
            # 또는 별도의 임베딩 이력 테이블을 만들 수 있음.
            # 현재는 processed_files 테이블의 'processing_status'를 'embedded'로 업데이트하는 방식으로 진행
            file_status = self.db_manager.get_file_processing_status(str(processed_file_path))
            if file_status and file_status['processing_status'] == 'embedded':
                self.stats['skipped_already_embedded'] += 1
                continue
            
            files_to_embed.append(ml_enhanced_file_path)

        self.stats['total_files_scanned'] = len(processed_files_info)
        self.stats['new_files_for_embedding'] = len(files_to_embed)
        
        if not files_to_embed:
            logger.info("No new preprocessed files to embed.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        logger.info(f"Found {len(files_to_embed)} new preprocessed files for embedding.")

        all_documents_to_add = []
        for file_path in tqdm(files_to_embed, desc="Collecting documents for embedding"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # LegalVectorStore의 add_documents 메서드에 맞는 형식으로 변환
                # file_data는 { 'laws': [...] } 형태일 수 있음
                laws_data = file_data.get('laws', [file_data]) if isinstance(file_data, dict) else file_data
                
                for law_data in laws_data:
                    law_id = law_data.get('law_id')
                    law_name = law_data.get('law_name')
                    
                    articles = law_data.get('articles', [])
                    for article in articles:
                        chunk_id = f"{law_id}_{article.get('article_number')}"
                        content = article.get('article_content')
                        metadata = {
                            "law_id": law_id,
                            "law_name": law_name,
                            "article_number": article.get('article_number'),
                            "article_title": article.get('article_title'),
                            "is_supplementary": article.get('is_supplementary', False),
                            "ml_confidence_score": article.get('ml_confidence_score'),
                            "parsing_method": article.get('parsing_method'),
                            "quality_score": law_data.get('data_quality', {}).get('parsing_quality_score', 0.0),
                            "source_file": str(file_path) # 원본 파일 경로 추가
                        }
                        all_documents_to_add.append((chunk_id, content, metadata))
                        self.stats['total_chunks_added'] += 1
                
                # 원본 raw 파일 경로를 'embedded' 상태로 업데이트
                # (ml_enhanced_file_path가 아닌 raw_file_path를 추적)
                original_raw_file_path = self._get_original_raw_file_path(file_path)
                if original_raw_file_path:
                    self.db_manager.update_file_processing_status(original_raw_file_path, "embedded")
                    self.stats['successfully_embedded_files'] += 1

            except Exception as e:
                error_msg = f"Failed to prepare documents from {file_path}: {e}"
                logger.error(error_msg)
                self.stats['failed_embedding_files'] += 1
                self.stats['errors'].append(error_msg)
                original_raw_file_path = self._get_original_raw_file_path(file_path)
                if original_raw_file_path:
                    self.db_manager.update_file_processing_status(original_raw_file_path, "failed_embedding", error_msg)

        if all_documents_to_add:
            logger.info(f"Adding {len(all_documents_to_add)} document chunks to vector store...")
            
            # 배치 처리로 문서 추가
            texts = [doc[1] for doc in all_documents_to_add]  # content
            metadatas = [doc[2] for doc in all_documents_to_add]  # metadata
            
            # 배치 크기로 나누어 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                self.vector_store.add_documents(batch_texts, batch_metadatas)
            
            self.vector_store.save_index(self.embedding_output_path)
            logger.info("Vector store updated and saved.")
        else:
            logger.info("No new document chunks to add to vector store.")

        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration'] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Incremental vector embedding completed. Stats: {self.stats}")
        return self.stats

    def _get_original_raw_file_path(self, ml_enhanced_file_path: Path) -> Optional[str]:
        """ML enhanced 파일 경로로부터 원본 raw 파일 경로를 추론"""
        # 예: data/processed/assembly/law_only/20251016/ml_enhanced_law_only_page_001_...json
        # -> data/raw/assembly/law_only/20251016/law_only_page_001_...json
        try:
            relative_path_from_processed = ml_enhanced_file_path.relative_to(self.processed_data_base_path)
            original_file_name = ml_enhanced_file_path.name.replace("ml_enhanced_", "")
            original_raw_path = self.auto_detector.raw_data_base_path / relative_path_from_processed.parent / original_file_name
            return str(original_raw_path)
        except ValueError:
            logger.error(f"Could not determine original raw file path for {ml_enhanced_file_path}")
            return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    builder = IncrementalVectorBuilder()
    builder.build_incremental_embeddings(data_type="law_only")