#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ì¦ë¶„ ?ë? ë²¡í„° ?„ë² ???ì„±ê¸?

?ˆë¡œ ?„ì²˜ë¦¬ëœ ?ë? ?°ì´?°ë¡œë¶€??ì¦ë¶„ ë²¡í„° ?„ë² ?©ì„ ?ì„±?˜ëŠ” ?œìŠ¤?œìž…?ˆë‹¤.
ê¸°ì¡´ FAISS ?¸ë±?¤ë? ?…ë°?´íŠ¸?˜ì—¬ ?ˆë¡œ???ë? ?°ì´?°ë? ì¶”ê??©ë‹ˆ??
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
from lawfirm_langgraph.core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2 as DatabaseManager
from scripts.data_processing.auto_data_detector import AutoDataDetector
from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

logger = logging.getLogger(__name__)


class IncrementalPrecedentVectorBuilder:
    """ì¦ë¶„ ?ë? ë²¡í„° ?„ë² ???ì„±ê¸?""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat",
                 processed_data_base_path: str = "data/processed/assembly",
                 embedding_output_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents",
                 version: Optional[str] = None):
        """
        ì¦ë¶„ ?ë? ë²¡í„° ë¹Œë” ì´ˆê¸°??
        
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
        self.auto_detector = AutoDataDetector()
        
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
        
        logger.info(f"IncrementalPrecedentVectorBuilder initialized with model: {model_name}")
        logger.info(f"Existing FAISS index loaded: {self.vector_store.index is not None}")

    def build_incremental_embeddings(self, category: str = "civil", batch_size: int = 100) -> Dict[str, Any]:
        """
        ?ˆë¡œ ?„ì²˜ë¦¬ëœ ?ë? ?°ì´?°ë¡œë¶€??ì¦ë¶„ ë²¡í„° ?„ë² ???ì„±
        
        Args:
            category: ?„ë² ?©í•  ?¹ì • ì¹´í…Œê³ ë¦¬ (civil, criminal, family)
            batch_size: ?„ë² ??ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸°
            
        Returns:
            Dict[str, Any]: ì²˜ë¦¬ ê²°ê³¼ ?µê³„
        """
        logger.info(f"Starting incremental precedent vector embedding for category: {category}")
        start_time = datetime.now()
        
        # ?°ì´???€??ê²°ì •
        data_type = f"precedent_{category}"
        
        # ?°ì´?°ë² ?´ìŠ¤?ì„œ 'completed' ?íƒœ???„ì²˜ë¦¬ëœ ?Œì¼ ëª©ë¡ ê°€?¸ì˜¤ê¸?
        processed_files_info = self.db_manager.get_processed_files_by_type(data_type, status="completed")
        
        files_to_embed = []
        for file_info in processed_files_info:
            processed_file_path = Path(file_info['file_path'])
            
            # ?ë³¸ raw ?Œì¼ ê²½ë¡œë¥?ê¸°ë°˜?¼ë¡œ ml_enhanced_*.json ?Œì¼ ê²½ë¡œ ì¶”ë¡ 
            # ?? data/raw/assembly/precedent/20251016/civil/precedent_civil_page_001_...json
            # -> data/processed/assembly/precedent/civil/20251016/ml_enhanced_precedent_civil_page_001_...json
            relative_path = processed_file_path.relative_to(self.auto_detector.raw_data_base_path / "precedent")
            ml_enhanced_file_name = f"ml_enhanced_{processed_file_path.stem}.json"
            ml_enhanced_file_path = self.processed_data_base_path / "precedent" / category / relative_path.parent / ml_enhanced_file_name
            
            if not ml_enhanced_file_path.exists():
                logger.warning(f"ML enhanced file not found for {processed_file_path}. Skipping.")
                continue
            
            # ?´ë? ?„ë² ?©ëœ ?Œì¼?¸ì? ?•ì¸
            file_status = self.db_manager.get_file_processing_status(str(processed_file_path))
            if file_status and file_status['processing_status'] == 'embedded':
                self.stats['skipped_already_embedded'] += 1
                continue
            
            files_to_embed.append(ml_enhanced_file_path)

        self.stats['total_files_scanned'] = len(processed_files_info)
        self.stats['new_files_for_embedding'] = len(files_to_embed)
        
        if not files_to_embed:
            logger.info("No new preprocessed precedent files to embed.")
            self.stats['end_time'] = datetime.now().isoformat()
            self.stats['duration'] = (datetime.now() - start_time).total_seconds()
            return self.stats

        logger.info(f"Found {len(files_to_embed)} new preprocessed precedent files for embedding.")

        all_documents_to_add = []
        for file_path in tqdm(files_to_embed, desc="Collecting precedent documents for embedding"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                # LegalVectorStore??add_documents ë©”ì„œ?œì— ë§žëŠ” ?•ì‹?¼ë¡œ ë³€??
                cases_data = file_data.get('cases', [file_data]) if isinstance(file_data, dict) else file_data
                
                for case_data in cases_data:
                    case_id = case_data.get('case_id')
                    case_name = case_data.get('case_name')
                    category = case_data.get('category')
                    
                    # ì¼€?´ìŠ¤ ?„ì²´ ?ìŠ¤?¸ë? ì²?¬ë¡?ì¶”ê?
                    full_text = case_data.get('full_text', '')
                    if full_text:
                        chunk_id = f"{case_id}_full"
                        metadata = {
                            "case_id": case_id,
                            "case_name": case_name,
                            "category": category,
                            "chunk_type": "full_text",
                            "source_file": str(file_path),
                            "case_number": case_data.get('case_number', ''),
                            "court": case_data.get('court', ''),
                            "decision_date": case_data.get('decision_date', ''),
                            "announce_date": case_data.get('announce_date', '') or case_data.get('decision_date', ''),
                            "field": case_data.get('field', ''),
                            "doc_id": case_data.get('case_number', '') or case_id,
                            "casenames": case_name,
                            "type": "case_paragraph",
                            "source_type": "case_paragraph"
                        }
                        all_documents_to_add.append((chunk_id, full_text, metadata))
                        self.stats['total_chunks_added'] += 1
                    
                    # ?¹ì…˜ë³„ë¡œ ì²?¬ ì¶”ê?
                    sections = case_data.get('sections', [])
                    for section in sections:
                        if section.get('has_content') and section.get('section_content'):
                            section_id = f"{case_id}_{section.get('section_type')}"
                            content = section.get('section_content')
                            metadata = {
                                "case_id": case_id,
                                "case_name": case_name,
                                "category": category,
                                "section_type": section.get('section_type'),
                                "section_type_korean": section.get('section_type_korean'),
                                "chunk_type": "section",
                                "source_file": str(file_path),
                                "case_number": case_data.get('case_number', ''),
                                "court": case_data.get('court', ''),
                                "decision_date": case_data.get('decision_date', ''),
                                "announce_date": case_data.get('announce_date', '') or case_data.get('decision_date', ''),
                                "field": case_data.get('field', ''),
                                "doc_id": case_data.get('case_number', '') or case_id,
                                "casenames": case_name,
                                "type": "case_paragraph",
                                "source_type": "case_paragraph"
                            }
                            all_documents_to_add.append((section_id, content, metadata))
                            self.stats['total_chunks_added'] += 1
                
                # ?ë³¸ raw ?Œì¼ ê²½ë¡œë¥?'embedded' ?íƒœë¡??…ë°?´íŠ¸
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
            logger.info(f"Adding {len(all_documents_to_add)} precedent document chunks to vector store...")
            
            # ë°°ì¹˜ ì²˜ë¦¬ë¡?ë¬¸ì„œ ì¶”ê?
            texts = [doc[1] for doc in all_documents_to_add]  # content
            metadatas = [doc[2] for doc in all_documents_to_add]  # metadata
            
            # ë°°ì¹˜ ?¬ê¸°ë¡??˜ëˆ„??ì²˜ë¦¬
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                self.vector_store.add_documents(batch_texts, metadatas=batch_metadatas)
            
            self.vector_store.save_index(self.version_path)
            logger.info(f"Precedent vector store updated and saved to {self.version_path}")
        else:
            logger.info("No new precedent document chunks to add to vector store.")

        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration'] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Incremental precedent vector embedding completed. Stats: {self.stats}")
        return self.stats

    def _get_original_raw_file_path(self, ml_enhanced_file_path: Path) -> Optional[str]:
        """ML enhanced ?Œì¼ ê²½ë¡œë¡œë????ë³¸ raw ?Œì¼ ê²½ë¡œë¥?ì¶”ë¡ """
        # ?? data/processed/assembly/precedent/civil/20251016/ml_enhanced_precedent_civil_page_001_...json
        # -> data/raw/assembly/precedent/20251016/civil/precedent_civil_page_001_...json
        try:
            relative_path_from_processed = ml_enhanced_file_path.relative_to(self.processed_data_base_path)
            original_file_name = ml_enhanced_file_path.name.replace("ml_enhanced_", "")
            
            # ê²½ë¡œ êµ¬ì¡°: precedent/civil/20251016/ml_enhanced_*.json
            # -> precedent/20251016/civil/*.json
            path_parts = list(relative_path_from_processed.parts)
            if len(path_parts) >= 3:  # precedent/category/date/filename
                category = path_parts[1]  # civil, criminal, family
                date = path_parts[2]      # 20251016
                original_raw_path = self.auto_detector.raw_data_base_path / "precedent" / date / category / original_file_name
                return str(original_raw_path)
            else:
                logger.error(f"Unexpected path structure for {ml_enhanced_file_path}")
                return None
        except ValueError:
            logger.error(f"Could not determine original raw file path for {ml_enhanced_file_path}")
            return None


def main():
    """ë©”ì¸ ?¨ìˆ˜"""
    parser = argparse.ArgumentParser(description="ì¦ë¶„ ?ë? ë²¡í„° ?„ë² ???ì„±ê¸?)
    parser.add_argument('--category', default='civil', 
                        choices=['civil', 'criminal', 'family', 'tax', 'administrative', 'patent'],
                        help='ì²˜ë¦¬???ë? ì¹´í…Œê³ ë¦¬')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='ë°°ì¹˜ ì²˜ë¦¬ ?¬ê¸°')
    parser.add_argument('--model-name', default='jhgan/ko-sroberta-multitask',
                        help='?¬ìš©??Sentence-BERT ëª¨ë¸ëª?)
    parser.add_argument('--dimension', type=int, default=768,
                        help='ë²¡í„° ì°¨ì›')
    parser.add_argument('--index-type', default='flat',
                        choices=['flat', 'ivf', 'hnsw'],
                        help='FAISS ?¸ë±???€??)
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='?ì„¸ ë¡œê·¸ ì¶œë ¥')
    
    args = parser.parse_args()
    
    # ë¡œê¹… ?¤ì •
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # ì¦ë¶„ ?ë? ë²¡í„° ë¹Œë” ì´ˆê¸°??
        builder = IncrementalPrecedentVectorBuilder(
            model_name=args.model_name,
            dimension=args.dimension,
            index_type=args.index_type
        )
        
        # ì¦ë¶„ ë²¡í„° ?„ë² ???ì„±
        stats = builder.build_incremental_embeddings(
            category=args.category,
            batch_size=args.batch_size
        )
        
        # ê²°ê³¼ ì¶œë ¥
        logger.info("Precedent Vector Embedding Results:")
        logger.info(f"  Total files scanned: {stats['total_files_scanned']}")
        logger.info(f"  New files for embedding: {stats['new_files_for_embedding']}")
        logger.info(f"  Successfully embedded files: {stats['successfully_embedded_files']}")
        logger.info(f"  Failed embedding files: {stats['failed_embedding_files']}")
        logger.info(f"  Skipped already embedded: {stats['skipped_already_embedded']}")
        logger.info(f"  Total chunks added: {stats['total_chunks_added']}")
        logger.info(f"  Processing time: {stats['duration']:.2f} seconds")
        
        if stats['errors']:
            logger.error("Error messages:")
            for error in stats['errors']:
                logger.error(f"  - {error}")
        
        return stats['successfully_embedded_files'] > 0
        
    except Exception as e:
        logger.error(f"Error in incremental precedent vector embedding: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
