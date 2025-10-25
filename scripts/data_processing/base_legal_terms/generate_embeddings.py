"""
법령정보지식베이스 법령용어 벡터 임베딩 생성

이 모듈은 처리된 법령용어 데이터에 대해 벡터 임베딩을 생성하고
FAISS 인덱스를 구축하는 기능을 제공합니다.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
from datetime import datetime
import pickle

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# 설정 파일 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'base_legal_terms', 'config'))
from base_legal_term_collection_config import BaseLegalTermCollectionConfig as Config

# 로거 설정
from source.utils.logger import setup_logging, get_logger

# 로거 초기화
logger = get_logger(__name__)

class BaseLegalTermEmbeddingGenerator:
    """법령정보지식베이스 법령용어 벡터 임베딩 생성기"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store_config = config.get_vector_store_config()
        self.file_storage_config = config.get_file_storage_config()
        
        # 디렉토리 설정
        self.base_dir = Path(self.file_storage_config.get("base_dir", "data/base_legal_terms"))
        self.processed_dir = Path(self.file_storage_config.get("processed_data_dir", "data/base_legal_terms/processed"))
        self.embeddings_dir = Path(self.file_storage_config.get("embeddings_dir", "data/base_legal_terms/embeddings"))
        
        # 벡터스토어 설정
        self.model_name = self.vector_store_config.get("model_name", "jhgan/ko-sroberta-multitask")
        self.vector_dimension = self.vector_store_config.get("vector_dimension", 768)
        self.index_type = self.vector_store_config.get("index_type", "flat")
        self.similarity_metric = self.vector_store_config.get("similarity_metric", "l2")
        
        # 파일 경로
        self.index_file = self.embeddings_dir / self.vector_store_config.get("index_file", "base_legal_terms_index.faiss")
        self.metadata_file = self.embeddings_dir / self.vector_store_config.get("metadata_file", "base_legal_terms_metadata.json")
        self.cache_dir = Path(self.vector_store_config.get("cache_dir", "data/base_legal_terms/embeddings/cache"))
        
        # 디렉토리 생성
        self._create_directories()
        
        # 임베딩 모델 로드
        self.model = None
        self.tokenizer = None
        self._load_embedding_model()
        
        # 통계
        self.stats = {
            "total_terms": 0,
            "processed_terms": 0,
            "embedding_errors": 0,
            "index_built": False
        }
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.embeddings_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"임베딩 모델 로딩: {self.model_name}")
            
            # 토크나이저와 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("GPU 사용으로 임베딩 모델 로드")
            else:
                logger.info("CPU 사용으로 임베딩 모델 로드")
            
            # 평가 모드로 설정
            self.model.eval()
            
            logger.info("임베딩 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트에 대한 임베딩 생성"""
        try:
            import torch
            
            # 텍스트 토크나이징
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # GPU 사용 가능하면 GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 임베딩 생성
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 평균 풀링 사용
                embeddings = outputs.last_hidden_state.mean(dim=1)
                # CPU로 이동 후 numpy 변환
                embeddings = embeddings.cpu().numpy()
            
            return embeddings[0]  # 첫 번째 배치의 임베딩 반환
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None
    
    def create_term_embedding(self, term_data: Dict) -> Optional[Dict]:
        """용어 데이터에 대한 임베딩 생성"""
        try:
            # 임베딩을 생성할 텍스트 구성
            term_name = term_data.get('용어명', '')
            definition = term_data.get('용어정의', '')
            keywords = term_data.get('키워드', [])
            
            # 텍스트 조합
            text_parts = [term_name, definition]
            if keywords:
                text_parts.append(' '.join(keywords))
            
            combined_text = ' '.join(text_parts)
            
            # 임베딩 생성
            embedding = self.generate_embedding(combined_text)
            if embedding is None:
                return None
            
            # 임베딩 데이터 구성
            embedding_data = {
                'term_id': term_data.get('원본ID', ''),
                'term_name': term_name,
                'definition': definition,
                'keywords': keywords,
                'category': term_data.get('카테고리', ''),
                'quality_score': term_data.get('품질점수', 0.0),
                'embedding': embedding.tolist(),
                'created_at': datetime.now().isoformat()
            }
            
            return embedding_data
            
        except Exception as e:
            logger.error(f"용어 임베딩 생성 실패: {e}")
            return None
    
    def load_processed_terms(self) -> List[Dict]:
        """처리된 용어 데이터 로드"""
        try:
            processed_terms = []
            
            # 통합 파일들 로드
            integrated_files = list(self.processed_dir.glob("integrated_terms/*.json"))
            
            if not integrated_files:
                logger.warning("처리된 용어 파일이 없습니다.")
                return []
            
            for file_path in integrated_files:
                logger.info(f"처리된 용어 파일 로드: {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    terms = json.load(f)
                
                if isinstance(terms, list):
                    processed_terms.extend(terms)
                else:
                    logger.warning(f"파일 형식 오류: {file_path.name}")
            
            logger.info(f"총 {len(processed_terms)}개 처리된 용어 로드")
            return processed_terms
            
        except Exception as e:
            logger.error(f"처리된 용어 로드 실패: {e}")
            return []
    
    def generate_all_embeddings(self) -> List[Dict]:
        """모든 용어에 대한 임베딩 생성"""
        try:
            logger.info("벡터 임베딩 생성 시작")
            
            # 처리된 용어 로드
            processed_terms = self.load_processed_terms()
            if not processed_terms:
                logger.warning("처리된 용어가 없습니다.")
                return []
            
            self.stats["total_terms"] = len(processed_terms)
            
            embeddings_data = []
            
            for i, term in enumerate(processed_terms):
                if i % 100 == 0:
                    logger.info(f"임베딩 생성 진행: {i}/{len(processed_terms)}")
                
                embedding_data = self.create_term_embedding(term)
                if embedding_data:
                    embeddings_data.append(embedding_data)
                    self.stats["processed_terms"] += 1
                else:
                    self.stats["embedding_errors"] += 1
            
            logger.info(f"벡터 임베딩 생성 완료: {len(embeddings_data)}개")
            return embeddings_data
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {e}")
            return []
    
    def build_faiss_index(self, embeddings_data: List[Dict]) -> bool:
        """FAISS 인덱스 구축"""
        try:
            if not embeddings_data:
                logger.warning("임베딩 데이터가 없습니다.")
                return False
            
            logger.info("FAISS 인덱스 구축 시작")
            
            # 임베딩 벡터 추출
            embeddings = np.array([data['embedding'] for data in embeddings_data])
            
            # 인덱스 타입에 따른 FAISS 인덱스 생성
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.vector_dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.vector_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.vector_dimension, 32)
            else:
                logger.error(f"지원하지 않는 인덱스 타입: {self.index_type}")
                return False
            
            # 인덱스에 벡터 추가
            if self.index_type == "ivf":
                index.train(embeddings)
            
            index.add(embeddings)
            
            # 인덱스 저장
            faiss.write_index(index, str(self.index_file))
            
            # 메타데이터 저장
            metadata = {
                "index_type": self.index_type,
                "vector_dimension": self.vector_dimension,
                "similarity_metric": self.similarity_metric,
                "total_vectors": len(embeddings_data),
                "model_name": self.model_name,
                "created_at": datetime.now().isoformat(),
                "terms": [
                    {
                        "term_id": data["term_id"],
                        "term_name": data["term_name"],
                        "category": data["category"],
                        "quality_score": data["quality_score"]
                    }
                    for data in embeddings_data
                ]
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.stats["index_built"] = True
            logger.info(f"FAISS 인덱스 구축 완료: {len(embeddings_data)}개 벡터")
            
            return True
            
        except Exception as e:
            logger.error(f"FAISS 인덱스 구축 실패: {e}")
            return False
    
    def save_embeddings_cache(self, embeddings_data: List[Dict]):
        """임베딩 캐시 저장"""
        try:
            cache_file = self.cache_dir / f"embeddings_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            logger.info(f"임베딩 캐시 저장: {cache_file}")
            
        except Exception as e:
            logger.error(f"임베딩 캐시 저장 실패: {e}")
    
    def save_statistics(self):
        """생성 통계 저장"""
        try:
            stats_file = self.embeddings_dir / f"embedding_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            stats_data = {
                "생성통계": self.stats,
                "생성일시": datetime.now().isoformat(),
                "설정": {
                    "모델명": self.model_name,
                    "벡터차원": self.vector_dimension,
                    "인덱스타입": self.index_type,
                    "유사도메트릭": self.similarity_metric
                }
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"생성 통계 저장: {stats_file}")
            
        except Exception as e:
            logger.error(f"통계 저장 실패: {e}")
    
    def generate_embeddings_pipeline(self) -> bool:
        """임베딩 생성 파이프라인 실행"""
        try:
            logger.info("=== 벡터 임베딩 생성 파이프라인 시작 ===")
            
            # 1. 임베딩 생성
            embeddings_data = self.generate_all_embeddings()
            if not embeddings_data:
                logger.error("임베딩 생성 실패")
                return False
            
            # 2. FAISS 인덱스 구축
            index_built = self.build_faiss_index(embeddings_data)
            if not index_built:
                logger.error("FAISS 인덱스 구축 실패")
                return False
            
            # 3. 캐시 저장
            self.save_embeddings_cache(embeddings_data)
            
            # 4. 통계 저장
            self.save_statistics()
            
            logger.info("=== 벡터 임베딩 생성 파이프라인 완료 ===")
            logger.info(f"생성 통계: {self.stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"임베딩 생성 파이프라인 실패: {e}")
            return False


def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        config = Config()
        
        # 임베딩 생성기 생성 및 실행
        generator = BaseLegalTermEmbeddingGenerator(config)
        
        success = generator.generate_embeddings_pipeline()
        
        if success:
            logger.info("=== 벡터 임베딩 생성 완료 ===")
        else:
            logger.error("=== 벡터 임베딩 생성 실패 ===")
            
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
