#!/usr/bin/env python3
"""
ML 강화 벡터 ?�베???�성�?

ML 강화 ?�싱??법률 ?�이?�로부??벡터 ?�베?�을 ?�성?�고 FAISS ?�덱?��? 구축?�니??
본칙�?부칙을 구분?�여 ?�베?�하�? ML ?�뢰?��? ?�질 ?�수�?메�??�이?�에 ?�함?�니??
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

# ?�로?�트 루트�?Python 경로??추�?
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from source.data.vector_store import LegalVectorStore

# Windows 콘솔?�서 UTF-8 ?�코???�정
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # ?��? UTF-8�??�정??경우 무시
        pass

logger = logging.getLogger(__name__)


class MLEnhancedVectorBuilder:
    """ML 강화 벡터 ?�베???�성�?""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask", 
                 dimension: int = 768, index_type: str = "flat"):
        """
        ML 강화 벡터 빌더 초기??
        
        Args:
            model_name: ?�용??Sentence-BERT 모델�?
            dimension: 벡터 차원
            index_type: FAISS ?�덱???�??
        """
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        
        # 벡터 ?�토??초기??
        self.vector_store = LegalVectorStore(
            model_name=model_name,
            dimension=dimension,
            index_type=index_type
        )
        
        # ?�계 ?�보
        self.stats = {
            'total_laws_processed': 0,
            'total_articles_processed': 0,
            'main_articles_processed': 0,
            'supplementary_articles_processed': 0,
            'total_chunks_created': 0,
            'processing_time': 0,
            'errors': []
        }
        
        logger.info(f"MLEnhancedVectorBuilder initialized with model: {model_name}")
    
    def build_embeddings(self, processed_dir: Path, batch_size: int = 100) -> bool:
        """
        ML 강화 법률 ?�이?�로부??벡터 ?�베???�성
        
        Args:
            processed_dir: ML 강화 처리???�이???�렉?�리
            batch_size: 배치 처리 ?�기
            
        Returns:
            bool: ?�공 ?��?
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting ML-enhanced vector embedding generation from: {processed_dir}")
            
            # JSON ?�일 목록 ?�집
            json_files = list(processed_dir.rglob("ml_enhanced_*.json"))
            logger.info(f"Found {len(json_files)} ML-enhanced files to process")
            
            if not json_files:
                logger.error("No ML-enhanced files found")
                return False
            
            # 배치별로 처리
            all_documents = []
            batch_count = 0
            
            for i in range(0, len(json_files), batch_size):
                batch_files = json_files[i:i + batch_size]
                batch_count += 1
                
                logger.info(f"Processing batch {batch_count}: {len(batch_files)} files")
                
                batch_documents = self._process_batch(batch_files)
                all_documents.extend(batch_documents)
                
                # 중간 ?�??(메모�?관�?
                if len(all_documents) >= batch_size * 2:
                    self._add_documents_to_index(all_documents)
                    all_documents = []
            
            # ?��? 문서??처리
            if all_documents:
                self._add_documents_to_index(all_documents)
            
            # 처리 ?�간 기록
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Vector embedding generation completed in {self.stats['processing_time']:.2f} seconds")
            logger.info(f"Statistics: {self.stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build embeddings: {e}")
            self.stats['errors'].append(str(e))
            return False
    
    def _process_batch(self, batch_files: List[Path]) -> List[Dict[str, Any]]:
        """배치 ?�일?�을 처리?�여 문서 리스???�성"""
        batch_documents = []
        
        for file_path in tqdm(batch_files, desc="Processing files"):
            try:
                documents = self._process_single_file(file_path)
                batch_documents.extend(documents)
                self.stats['total_laws_processed'] += 1
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        return batch_documents
    
    def _process_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """?�일 ?�일 처리"""
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        documents = []
        
        # ?�일 구조 ?�인
        if isinstance(file_data, dict) and 'laws' in file_data:
            laws = file_data['laws']
        elif isinstance(file_data, list):
            laws = file_data
        else:
            laws = [file_data]
        
        for law_data in laws:
            try:
                # 법률 메�??�이??추출
                law_metadata = self._extract_law_metadata(law_data)
                
                # 본칙 조문 처리
                articles = law_data.get('articles', [])
                if not isinstance(articles, list):
                    articles = []
                main_documents = self._create_article_documents(
                    articles,
                    law_metadata,
                    article_type='main'
                )
                documents.extend(main_documents)
                
                # 부�?조문 처리 (별도 ?�드가 ?�는 경우)
                # 부�?조문 처리
                supplementary_articles = law_data.get('supplementary_articles', [])
                if not isinstance(supplementary_articles, list):
                    supplementary_articles = []
                supp_documents = self._create_article_documents(
                    supplementary_articles,
                    law_metadata,
                    article_type='supplementary'
                )
                documents.extend(supp_documents)
                
                # ?�계 ?�데?�트
                all_articles = law_data.get('articles', [])
                if not isinstance(all_articles, list):
                    all_articles = []
                main_articles = [a for a in all_articles if not a.get('is_supplementary', False)]
                supp_articles = [a for a in all_articles if a.get('is_supplementary', False)]
                
                self.stats['total_articles_processed'] += len(all_articles)
                self.stats['main_articles_processed'] += len(main_articles)
                self.stats['supplementary_articles_processed'] += len(supp_articles)
                
            except Exception as e:
                error_msg = f"Error processing law {law_data.get('law_name', 'Unknown')}: {e}"
                logger.error(error_msg)
                logger.error(f"Law data keys: {list(law_data.keys())}")
                if law_data.get('articles'):
                    logger.error(f"First article keys: {list(law_data['articles'][0].keys())}")
                    logger.error(f"First article sub_articles type: {type(law_data['articles'][0].get('sub_articles'))}")
                    logger.error(f"First article references type: {type(law_data['articles'][0].get('references'))}")
                    # �?번째 조문??sub_articles ?�용 ?�인
                    first_article = law_data['articles'][0]
                    sub_articles = first_article.get('sub_articles', [])
                    if sub_articles:
                        logger.error(f"First sub_article type: {type(sub_articles[0])}")
                        logger.error(f"First sub_article value: {sub_articles[0]}")
                        if isinstance(sub_articles[0], dict):
                            logger.error(f"First sub_article keys: {list(sub_articles[0].keys())}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.stats['errors'].append(error_msg)
        
        return documents
    
    def _extract_law_metadata(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """법률 메�??�이??추출"""
        return {
            'law_id': law_data.get('law_id') or f"ml_enhanced_{law_data.get('law_name', 'unknown').replace(' ', '_')}",
            'law_name': law_data.get('law_name', ''),
            'law_type': law_data.get('law_type', ''),
            'category': law_data.get('category', ''),
            'promulgation_number': law_data.get('promulgation_number', ''),
            'promulgation_date': law_data.get('promulgation_date', ''),
            'enforcement_date': law_data.get('enforcement_date', ''),
            'amendment_type': law_data.get('amendment_type', ''),
            'ministry': law_data.get('ministry', ''),
            'ml_enhanced': law_data.get('ml_enhanced', True),
            'parsing_quality_score': law_data.get('data_quality', {}).get('parsing_quality_score', 0.0) if isinstance(law_data.get('data_quality'), dict) else 0.0,
            'processing_version': law_data.get('processing_version', 'ml_enhanced_v1.0'),
            'control_characters_removed': True
        }
    
    def _create_article_documents(self, articles: List[Dict[str, Any]], 
                                law_metadata: Dict[str, Any], 
                                article_type: str) -> List[Dict[str, Any]]:
        """조문?�을 문서�?변??""
        documents = []
        
        for article in articles:
            try:
                # 조문 메�??�이???�성
                article_metadata = {
                    **law_metadata,
                    'article_number': article.get('article_number', ''),
                    'article_title': article.get('article_title', ''),
                    'article_type': article_type,
                    'is_supplementary': article.get('is_supplementary', article_type == 'supplementary'),
                    'ml_confidence_score': article.get('ml_confidence_score'),
                    'parsing_method': article.get('parsing_method', 'ml_enhanced'),
                    'word_count': article.get('word_count', 0),
                    'char_count': article.get('char_count', 0),
                    'sub_articles_count': len(article.get('sub_articles', [])) if isinstance(article.get('sub_articles'), list) else 0,
                    'references_count': len(article.get('references', [])) if isinstance(article.get('references'), list) else 0
                }
                
                # 문서 ID ?�성
                document_id = f"{law_metadata['law_id']}_article_{article_metadata['article_number']}"
                article_metadata['document_id'] = document_id
                
                # ?�스??구성
                text_parts = []
                
                # 조문 번호?� ?�목
                if article_metadata['article_number']:
                    if article_metadata['article_title']:
                        text_parts.append(f"{article_metadata['article_number']}({article_metadata['article_title']})")
                    else:
                        text_parts.append(article_metadata['article_number'])
                
                # 조문 ?�용
                article_content = article.get('article_content', '')
                if article_content:
                    text_parts.append(article_content)
                
                # ?�위 조문??
                sub_articles = article.get('sub_articles', [])
                if not isinstance(sub_articles, list):
                    sub_articles = []
                for sub_idx, sub_article in enumerate(sub_articles):
                    try:
                        if not isinstance(sub_article, dict):
                            logger.error(f"Sub-article {sub_idx} is not a dict: {type(sub_article)} = {sub_article}")
                            continue
                        sub_content = sub_article.get('content', '')
                        if sub_content:
                            text_parts.append(sub_content)
                    except Exception as e:
                        logger.error(f"Error processing sub-article {sub_idx}: {e}")
                        logger.error(f"Sub-article type: {type(sub_article)}")
                        logger.error(f"Sub-article value: {sub_article}")
                        continue
                
                # 최종 ?�스??
                full_text = ' '.join(text_parts)
                
                if full_text.strip():
                    document = {
                        'id': document_id,
                        'text': full_text,
                        'metadata': article_metadata,
                        'chunks': [{
                            'id': f"{document_id}_chunk_0",
                            'text': full_text,
                            'start_pos': 0,
                            'end_pos': len(full_text),
                            'entities': article.get('references', [])
                        }]
                    }
                    documents.append(document)
                    self.stats['total_chunks_created'] += 1
                
            except Exception as e:
                error_msg = f"Error processing article {article.get('article_number', 'Unknown')}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        return documents
    
    def _add_documents_to_index(self, documents: List[Dict[str, Any]]) -> bool:
        """문서?�을 벡터 ?�덱?�에 추�?"""
        try:
            if not documents:
                return True
            
            # ?�스?��? 메�??�이??추출
            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = doc.get('chunks', [])
                for chunk in chunks:
                    texts.append(chunk.get('text', ''))
                    metadatas.append({
                        'document_id': doc.get('id', ''),
                        'document_type': 'law_article',
                        'chunk_id': chunk.get('id', ''),
                        'chunk_start': chunk.get('start_pos', 0),
                        'chunk_end': chunk.get('end_pos', 0),
                        'law_name': doc.get('metadata', {}).get('law_name', ''),
                        'category': doc.get('metadata', {}).get('category', ''),
                        'entities': chunk.get('entities', []) if isinstance(chunk.get('entities'), list) else [],
                        # ML 강화 메�??�이??추�?
                        'article_number': doc.get('metadata', {}).get('article_number', ''),
                        'article_title': doc.get('metadata', {}).get('article_title', ''),
                        'article_type': doc.get('metadata', {}).get('article_type', ''),
                        'is_supplementary': doc.get('metadata', {}).get('is_supplementary', False),
                        'ml_confidence_score': doc.get('metadata', {}).get('ml_confidence_score'),
                        'parsing_method': doc.get('metadata', {}).get('parsing_method', 'ml_enhanced'),
                        'parsing_quality_score': doc.get('metadata', {}).get('parsing_quality_score', 0.0),
                        'ml_enhanced': doc.get('metadata', {}).get('ml_enhanced', True),
                        'word_count': doc.get('metadata', {}).get('word_count', 0),
                        'char_count': doc.get('metadata', {}).get('char_count', 0)
                    })
            
            # 벡터 ?�토?�에 추�?
            success = self.vector_store.add_documents(texts, metadatas)
            
            if success:
                logger.info(f"Added {len(texts)} document chunks to vector index")
            else:
                logger.error("Failed to add documents to vector index")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return False
    
    def save_index(self, output_dir: Path) -> bool:
        """?�성???�덱???�??""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # ?�덱???�일�?
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            index_filename = f"ml_enhanced_faiss_index_{timestamp}"
            index_path = output_dir / index_filename
            
            # ?�덱???�??
            success = self.vector_store.save_index(str(index_path))
            
            if success:
                # ?�계 ?�보 ?�??
                stats_path = output_dir / f"ml_enhanced_vector_stats_{timestamp}.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, ensure_ascii=False, indent=2)
                
                logger.info(f"ML-enhanced vector index saved to: {index_path}")
                logger.info(f"Statistics saved to: {stats_path}")
                
                return True
            else:
                logger.error("Failed to save vector index")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """?�계 ?�보 반환"""
        vector_stats = self.vector_store.get_stats()
        return {
            **self.stats,
            'vector_store_stats': vector_stats
        }


def main():
    """메인 ?�수"""
    parser = argparse.ArgumentParser(description="ML 강화 벡터 ?�베???�성�?)
    parser.add_argument("--input", type=str, required=True,
                       help="ML 강화 처리???�이???�렉?�리 경로")
    parser.add_argument("--output", type=str, default="data/embeddings/ml_enhanced",
                       help="출력 ?�렉?�리 경로")
    parser.add_argument("--model", type=str, default="jhgan/ko-sroberta-multitask",
                       help="?�용??Sentence-BERT 모델�?)
    parser.add_argument("--dimension", type=int, default=768,
                       help="벡터 차원")
    parser.add_argument("--index-type", type=str, default="flat",
                       choices=["flat", "ivf", "hnsw"],
                       help="FAISS ?�덱???�??)
    parser.add_argument("--batch-size", type=int, default=100,
                       help="배치 처리 ?�기")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="로그 ?�벨")
    
    args = parser.parse_args()
    
    # 로깅 ?�정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ml_enhanced_vector_builder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # ?�력 ?�렉?�리 ?�인
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # 출력 ?�렉?�리 ?�성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ML 강화 벡터 빌더 초기??
        builder = MLEnhancedVectorBuilder(
            model_name=args.model,
            dimension=args.dimension,
            index_type=args.index_type
        )
        
        # 벡터 ?�베???�성
        logger.info("Starting ML-enhanced vector embedding generation...")
        success = builder.build_embeddings(input_dir, batch_size=args.batch_size)
        
        if not success:
            logger.error("Vector embedding generation failed")
            return 1
        
        # ?�덱???�??
        logger.info("Saving vector index...")
        save_success = builder.save_index(output_dir)
        
        if not save_success:
            logger.error("Failed to save vector index")
            return 1
        
        # 최종 ?�계 출력
        stats = builder.get_stats()
        logger.info("=== ML Enhanced Vector Building Completed ===")
        logger.info(f"Total laws processed: {stats['total_laws_processed']}")
        logger.info(f"Total articles processed: {stats['total_articles_processed']}")
        logger.info(f"Main articles: {stats['main_articles_processed']}")
        logger.info(f"Supplementary articles: {stats['supplementary_articles_processed']}")
        logger.info(f"Total chunks created: {stats['total_chunks_created']}")
        logger.info(f"Processing time: {stats['processing_time']:.2f} seconds")
        logger.info(f"Vector store documents: {stats['vector_store_stats']['documents_count']}")
        
        if stats['errors']:
            logger.warning(f"Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # 처음 5�??�류�?출력
                logger.warning(f"  - {error}")
        
        logger.info("ML-enhanced vector embedding generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
