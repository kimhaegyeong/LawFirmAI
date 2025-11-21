#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¦ë¶„ ë²¡í„° ?„ë² ???ì„±ê¸?

ê¸°ì¡´ FAISS ?¸ë±?¤ë? ë¡œë“œ?˜ê³  ?ˆë¡œ??ë¬¸ì„œë§??„ë² ?©í•˜??ê¸°ì¡´ ?¸ë±?¤ì— ì¶”ê??˜ëŠ” ?œìŠ¤?œìž…?ˆë‹¤.
ko-sroberta-multitask ëª¨ë¸???¬ìš©?˜ì—¬ ì¦ë¶„ ?…ë°?´íŠ¸ë¥??˜í–‰?©ë‹ˆ??
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

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore
from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager # ?Œì¼ ì²˜ë¦¬ ?´ë ¥ ì¶”ì ??
from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager # ?Œì¼ ?´ì‹œ ?ì„±??

logger = logging.getLogger(__name__)

class IncrementalVectorBuilder:
    """ì¦ë¶„ ë²¡í„° ?„ë² ???ì„±ê¸?""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat",
                 processed_data_base_path: str = "data/processed/assembly",
                 embedding_output_path: str = "data/embeddings/ml_enhanced_ko_sroberta",
                 version: Optional[str] = None):
        """
        ì¦ë¶„ ë²¡í„° ë¹Œë” ì´ˆê¸°??
        
        Args:
            model_name: ?¬ìš©??Sentence-BERT ëª¨ë¸ëª?
            dimension: ë²¡í„° ì°¨ì›
            index_type: FAISS ?¸ë±???€??
            processed_data_base_path: ?„ì²˜ë¦¬ëœ ?°ì´?°ê? ?€?¥ëœ ê¸°ë³¸ ?”ë ‰? ë¦¬
            embedding_output_path: ?„ë² ??ë°?FAISS ?¸ë±???€??ê²½ë¡œ
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        self.processed_data_base_path = Path(processed_data_base_path)
        self.embedding_output_path = Path(embedding_output_path)
        self.embedding_output_path.mkdir(parents=True, exist_ok=True)
        
        self.version_manager = VectorStoreVersionManager(self.embedding_output_path)
        self.version = version or self.version_manager.get_current_version()
        
        if self.version:
            self.version_path = self.version_manager.get_version_path(self.version)
            self.version_path.mkdir(parents=True, exist_ok=True)
        else:
            self.version_path = self.embedding_output_path
        
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=dimension,
            index_type=index_type
        )
        self.db_manager = DatabaseManager()
        self.auto_detector = AutoDataDetector() # ?Œì¼ ?´ì‹œ ê³„ì‚°??
        
        # ê¸°ì¡´ FAISS ?¸ë±??ë¡œë“œ ?œë„
        self.vector_store.load_index(self.version_path)
        
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
        ?ˆë¡œ ?„ì²˜ë¦¬ëœ ?°ì´?°ë¡œë¶€??ì¦ë¶„ ë²¡í„° ?„ë² ???ì„±
        
        Args:
            data_type: ?„ë² ?©í•  ?¹ì • ?°ì´??? í˜• (?? 'law_only'). 'all'?´ë©´ ëª¨ë“  ? í˜• ì²˜ë¦¬.
            batch_size: ?„ë² ??ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸°
            
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼ ?µê³„
        """
        logger.info(f"Starting incremental vector embedding for data type: {data_type}")
        start_time = datetime.now()
        
        # ?°ì´?°ë² ?´ìŠ¤?ì„œ 'completed' ?íƒœ???„ì²˜ë¦¬ëœ ?Œì¼ ëª©ë¡ ê°€?¸ì˜¤ê¸?
        processed_files_info = self.db_manager.get_processed_files_by_type(data_type, status="completed")
        
        files_to_embed = []
        for file_info in processed_files_info:
            processed_file_path = Path(file_info['file_path'])
            
            # ?ë³¸ raw ?Œì¼ ê²½ë¡œë¥?ê¸°ë°˜?¼ë¡œ ml_enhanced_*.json ?Œì¼ ê²½ë¡œ ì¶”ë¡ 
            # ?? data/raw/assembly/law_only/20251016/law_only_page_001_...json
            # -> data/processed/assembly/law_only/20251016/ml_enhanced_law_only_page_001_...json
            relative_path = processed_file_path.relative_to(self.auto_detector.raw_data_base_path)
            ml_enhanced_file_name = f"ml_enhanced_{processed_file_path.stem}.json"
            ml_enhanced_file_path = self.processed_data_base_path / relative_path.parent / ml_enhanced_file_name
            
            if not ml_enhanced_file_path.exists():
                logger.warning(f"ML enhanced file not found for {processed_file_path}. Skipping.")
                continue
            
            # ?´ë? ?„ë² ?©ëœ ?Œì¼?¸ì? ?•ì¸ (processed_files ?Œì´ë¸”ì—??'embedded' ?íƒœë¡?ì¶”ì )
            # ?¬ê¸°?œëŠ” processed_files ?Œì´ë¸”ì— 'embedded' ?íƒœë¥?ì¶”ê??˜ì? ?Šê³ ,
            # ?¨ìˆœ???´ë‹¹ ?Œì¼??ë²¡í„° ?¤í† ?´ì— ?´ë? ì¡´ìž¬?˜ëŠ”ì§€ ?¬ë?ë¡??ë‹¨
            # ?ëŠ” ë³„ë„???„ë² ???´ë ¥ ?Œì´ë¸”ì„ ë§Œë“¤ ???ˆìŒ.
            # ?„ìž¬??processed_files ?Œì´ë¸”ì˜ 'processing_status'ë¥?'embedded'ë¡??…ë°?´íŠ¸?˜ëŠ” ë°©ì‹?¼ë¡œ ì§„í–‰
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
                
                # LegalVectorStore??add_documents ë©”ì„œ?œì— ë§žëŠ” ?•ì‹?¼ë¡œ ë³€??
                # file_data??{ 'laws': [...] } ?•íƒœ?????ˆìŒ
                laws_data = file_data.get('laws', [file_data]) if isinstance(file_data, dict) else file_data
                
                original_raw_file_path = self._get_original_raw_file_path(file_path)
                
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
                            "source_file": str(file_path),
                            "raw_source_file": str(original_raw_file_path) if original_raw_file_path else "",
                            "type": "statute_article",
                            "source_type": "statute_article" # ?ë³¸ ?Œì¼ ê²½ë¡œ ì¶”ê?
                        }
                        all_documents_to_add.append((chunk_id, content, metadata))
                        self.stats['total_chunks_added'] += 1
                
                # ?ë³¸ raw ?Œì¼ ê²½ë¡œë¥?'embedded' ?íƒœë¡??…ë°?´íŠ¸
                # (ml_enhanced_file_pathê°€ ?„ë‹Œ raw_file_pathë¥?ì¶”ì )
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
            
            # ë°°ì¹˜ ì²˜ë¦¬ë¡?ë¬¸ì„œ ì¶”ê?
            texts = [doc[1] for doc in all_documents_to_add]  # content
            metadatas = [doc[2] for doc in all_documents_to_add]  # metadata
            
            # ë°°ì¹˜ ?¬ê¸°ë¡??˜ëˆ„??ì²˜ë¦¬
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                self.vector_store.add_documents(batch_texts, batch_metadatas)
            
            self.vector_store.save_index(self.version_path)
            logger.info(f"Vector store updated and saved to {self.version_path}")
        else:
            logger.info("No new document chunks to add to vector store.")

        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration'] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Incremental vector embedding completed. Stats: {self.stats}")
        return self.stats

    def _get_original_raw_file_path(self, ml_enhanced_file_path: Path) -> Optional[str]:
        """ML enhanced ?Œì¼ ê²½ë¡œë¡œë????ë³¸ raw ?Œì¼ ê²½ë¡œë¥?ì¶”ë¡ """
        # ?? data/processed/assembly/law_only/20251016/ml_enhanced_law_only_page_001_...json
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