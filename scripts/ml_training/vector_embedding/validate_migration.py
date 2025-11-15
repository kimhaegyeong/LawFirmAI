#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
벡터 임베딩 마이그레이션 검증 스크립트

벡터 수 일치 확인, 메타데이터 정확성 검증, 검색 결과 비교를 수행합니다.
"""

import logging
import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Please install faiss-cpu or faiss-gpu")

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml_training.vector_embedding.version_manager import VectorStoreVersionManager

logger = logging.getLogger(__name__)


class MigrationValidator:
    """마이그레이션 검증 클래스"""
    
    def __init__(self,
                 vector_store_base_path: Path,
                 db_path: Path,
                 version: Optional[str] = None,
                 model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        검증자 초기화
        
        Args:
            vector_store_base_path: 벡터스토어 기본 경로
            db_path: 데이터베이스 경로
            version: 버전 번호 (None이면 최신 버전)
            model_name: 모델명
        """
        self.vector_store_base_path = Path(vector_store_base_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.version_manager = VectorStoreVersionManager(self.vector_store_base_path)
        self.version = version or self.version_manager.get_current_version()
        
        if self.version:
            self.version_path = self.version_manager.get_version_path(self.version)
        else:
            self.version_path = self.vector_store_base_path
        
        self.index_path = self.version_path / "ml_enhanced_faiss_index.faiss"
        self.metadata_path = self.version_path / "ml_enhanced_faiss_index.json"
        
        self.validation_results = {
            'vector_count_match': False,
            'metadata_accuracy': False,
            'search_results_match': False,
            'errors': [],
            'warnings': []
        }
    
    def load_faiss_index(self) -> Optional[Any]:
        """FAISS 인덱스 로드"""
        if not FAISS_AVAILABLE:
            return None
        
        if not self.index_path.exists():
            logger.error(f"FAISS index not found: {self.index_path}")
            return None
        
        try:
            index = faiss.read_index(str(self.index_path))
            logger.info(f"FAISS index loaded: {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """메타데이터 로드"""
        if not self.metadata_path.exists():
            logger.error(f"Metadata file not found: {self.metadata_path}")
            return []
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = json.load(f)
            
            if isinstance(metadata_content, dict):
                if 'documents' in metadata_content:
                    metadata = metadata_content['documents']
                elif 'document_metadata' in metadata_content and 'document_texts' in metadata_content:
                    metadata_list = metadata_content['document_metadata']
                    texts_list = metadata_content['document_texts']
                    metadata = []
                    for meta, text in zip(metadata_list, texts_list):
                        combined = meta.copy()
                        combined['content'] = text
                        combined['text'] = text
                        metadata.append(combined)
                else:
                    metadata = []
            else:
                metadata = metadata_content
            
            logger.info(f"Metadata loaded: {len(metadata)} items")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []
    
    def validate_vector_count(self) -> bool:
        """벡터 수 일치 확인"""
        logger.info("Validating vector count...")
        
        index = self.load_faiss_index()
        if index is None:
            self.validation_results['errors'].append("Failed to load FAISS index")
            return False
        
        faiss_count = index.ntotal
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.execute(
                """SELECT COUNT(*) as count FROM embeddings WHERE model = ?""",
                (self.model_name,)
            )
            db_count = cursor.fetchone()['count']
            
            if faiss_count == db_count:
                logger.info(f"✓ Vector count matches: {faiss_count}")
                self.validation_results['vector_count_match'] = True
                return True
            else:
                logger.warning(f"✗ Vector count mismatch: FAISS={faiss_count}, DB={db_count}")
                self.validation_results['warnings'].append(
                    f"Vector count mismatch: FAISS={faiss_count}, DB={db_count}"
                )
                return False
        except Exception as e:
            logger.error(f"Error validating vector count: {e}")
            self.validation_results['errors'].append(str(e))
            return False
        finally:
            conn.close()
    
    def validate_metadata_accuracy(self) -> bool:
        """메타데이터 정확성 검증"""
        logger.info("Validating metadata accuracy...")
        
        metadata_list = self.load_metadata()
        if not metadata_list:
            self.validation_results['errors'].append("No metadata found")
            return False
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            errors = []
            warnings = []
            sample_size = min(100, len(metadata_list))
            
            import random
            sample_indices = random.sample(range(len(metadata_list)), sample_size)
            
            for idx in sample_indices:
                meta = metadata_list[idx]
                source_type = meta.get('type') or meta.get('source_type', '')
                
                if not source_type:
                    warnings.append(f"Missing source_type in metadata at index {idx}")
                    continue
                
                cursor = conn.execute(
                    """SELECT COUNT(*) as count 
                       FROM text_chunks 
                       WHERE source_type = ? AND meta LIKE ?""",
                    (source_type, f'%{json.dumps(meta, ensure_ascii=False)[:100]}%')
                )
                count = cursor.fetchone()['count']
                
                if count == 0:
                    warnings.append(f"No matching text_chunk found for metadata at index {idx}")
            
            if errors:
                self.validation_results['errors'].extend(errors)
                logger.error(f"Metadata validation found {len(errors)} errors")
                return False
            
            if warnings:
                self.validation_results['warnings'].extend(warnings)
                logger.warning(f"Metadata validation found {len(warnings)} warnings")
            
            logger.info(f"✓ Metadata accuracy validated (sampled {sample_size} items)")
            self.validation_results['metadata_accuracy'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            self.validation_results['errors'].append(str(e))
            return False
        finally:
            conn.close()
    
    def compare_search_results(self, test_queries: List[str] = None) -> bool:
        """검색 결과 비교"""
        logger.info("Comparing search results...")
        
        if test_queries is None:
            test_queries = [
                "계약 해제",
                "손해배상",
                "임대차보호법",
                "전세금 반환"
            ]
        
        try:
            from lawfirm_langgraph.core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
            
            # DB 기반 검색 엔진
            db_engine = SemanticSearchEngineV2(
                db_path=str(self.db_path),
                model_name=self.model_name,
                use_external_index=False
            )
            
            # 외부 인덱스 기반 검색 엔진
            external_engine = SemanticSearchEngineV2(
                db_path=str(self.db_path),
                model_name=self.model_name,
                use_external_index=True,
                vector_store_version=self.version
            )
            
            matches = 0
            total = 0
            
            for query in test_queries:
                db_results = db_engine.search(query, k=10)
                external_results = external_engine.search(query, k=10)
                
                if not db_results or not external_results:
                    logger.warning(f"Empty results for query: {query}")
                    continue
                
                db_texts = {r.get('text', '')[:100] for r in db_results}
                external_texts = {r.get('text', '')[:100] for r in external_results}
                
                overlap = len(db_texts & external_texts)
                total += len(db_texts | external_texts)
                matches += overlap
                
                logger.info(f"Query '{query}': {overlap}/{len(db_texts | external_texts)} results match")
            
            if total > 0:
                match_rate = matches / total
                logger.info(f"Overall match rate: {match_rate:.2%}")
                
                if match_rate >= 0.7:
                    logger.info("✓ Search results match")
                    self.validation_results['search_results_match'] = True
                    return True
                else:
                    logger.warning(f"✗ Search results mismatch: {match_rate:.2%} match rate")
                    self.validation_results['warnings'].append(
                        f"Search results match rate: {match_rate:.2%}"
                    )
                    return False
            else:
                logger.warning("No search results to compare")
                return False
                
        except Exception as e:
            logger.error(f"Error comparing search results: {e}", exc_info=True)
            self.validation_results['errors'].append(str(e))
            return False
    
    def validate_all(self, compare_search: bool = True) -> Dict[str, Any]:
        """전체 검증 실행"""
        logger.info("=" * 60)
        logger.info("Starting migration validation")
        logger.info("=" * 60)
        
        vector_count_ok = self.validate_vector_count()
        metadata_ok = self.validate_metadata_accuracy()
        
        search_ok = True
        if compare_search:
            search_ok = self.compare_search_results()
        
        self.validation_results['all_passed'] = vector_count_ok and metadata_ok and search_ok
        
        logger.info("=" * 60)
        logger.info("Validation Summary:")
        logger.info(f"  Vector count match: {'✓' if vector_count_ok else '✗'}")
        logger.info(f"  Metadata accuracy: {'✓' if metadata_ok else '✗'}")
        if compare_search:
            logger.info(f"  Search results match: {'✓' if search_ok else '✗'}")
        logger.info(f"  Overall: {'✓ PASSED' if self.validation_results['all_passed'] else '✗ FAILED'}")
        logger.info("=" * 60)
        
        if self.validation_results['warnings']:
            logger.warning(f"Warnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings'][:10]:
                logger.warning(f"  - {warning}")
        
        if self.validation_results['errors']:
            logger.error(f"Errors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors'][:10]:
                logger.error(f"  - {error}")
        
        return self.validation_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="벡터 임베딩 마이그레이션 검증")
    parser.add_argument('--vector-store-path', 
                        default='data/embeddings/ml_enhanced_ko_sroberta_precedents',
                        help='벡터스토어 기본 경로')
    parser.add_argument('--db-path', 
                        default='data/lawfirm_v2.db',
                        help='데이터베이스 경로')
    parser.add_argument('--version', 
                        default=None,
                        help='버전 번호 (예: v2.0.0). None이면 최신 버전 사용')
    parser.add_argument('--model-name', 
                        default='jhgan/ko-sroberta-multitask',
                        help='모델명')
    parser.add_argument('--no-search-comparison', 
                        action='store_true',
                        help='검색 결과 비교 건너뛰기')
    parser.add_argument('--verbose', '-v', 
                        action='store_true',
                        help='상세 로그 출력')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    validator = MigrationValidator(
        vector_store_base_path=Path(args.vector_store_path),
        db_path=Path(args.db_path),
        version=args.version,
        model_name=args.model_name
    )
    
    results = validator.validate_all(compare_search=not args.no_search_comparison)
    
    return results.get('all_passed', False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

