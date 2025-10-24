"""
법률 용어 벡터스토어 업데이트 서비스

이 모듈은 수집된 법률 용어 데이터를 벡터스토어에 업데이트하는 기능을 제공합니다.
"""

import asyncio
import json
import logging
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from source.models.sentence_bert import SentenceBERTModel
from source.utils.config import Config
from source.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class VectorUpdateProgress:
    """벡터 업데이트 진행 상황"""
    total_terms: int = 0
    processed_terms: int = 0
    failed_terms: int = 0
    last_update_time: Optional[str] = None
    index_size: int = 0

class LegalTermVectorStore:
    """법률 용어 벡터스토어 관리"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 모델 설정
        self.model_name = config.get("SENTENCE_BERT_MODEL", "jhgan/ko-sroberta-multitask")
        self.model = None
        
        # 데이터베이스 경로
        self.data_dir = Path(config.get("DATA_DIR", "data"))
        self.db_path = self.data_dir / "legal_terms.db"
        
        # 벡터스토어 경로
        self.embeddings_dir = self.data_dir / "embeddings" / "legal_terms"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.embeddings_dir / "legal_term_index.faiss"
        self.metadata_file = self.embeddings_dir / "legal_term_metadata.json"
        self.progress_file = self.embeddings_dir / "vector_update_progress.json"
        
        # 벡터 설정
        self.vector_dim = 768  # Ko-SRoBERTa 기본 차원
        self.index = None
        self.metadata = []
        
        # 진행 상황
        self.progress = VectorUpdateProgress()
        
        # 모델 초기화
        self._init_model()
        
    def _init_model(self):
        """문장 임베딩 모델 초기화"""
        try:
            self.model = SentenceBERTModel(self.model_name)
            logger.info(f"문장 임베딩 모델 초기화 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"문장 임베딩 모델 초기화 실패: {e}")
            raise
    
    def _load_progress(self) -> VectorUpdateProgress:
        """진행 상황 로드"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return VectorUpdateProgress(**data)
        except Exception as e:
            logger.warning(f"진행 상황 로드 실패: {e}")
        
        return VectorUpdateProgress()
    
    def _save_progress(self):
        """진행 상황 저장"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.progress), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"진행 상황 저장 실패: {e}")
    
    def _load_existing_index(self) -> bool:
        """기존 인덱스 로드"""
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                # FAISS 인덱스 로드
                self.index = faiss.read_index(str(self.index_file))
                
                # 메타데이터 로드
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"기존 벡터 인덱스 로드 완료: {len(self.metadata)}개 항목")
                return True
                
        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패: {e}")
        
        return False
    
    def _create_new_index(self):
        """새로운 인덱스 생성"""
        try:
            # FAISS 인덱스 생성 (L2 거리 사용)
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata = []
            
            logger.info("새로운 벡터 인덱스 생성 완료")
            
        except Exception as e:
            logger.error(f"새로운 인덱스 생성 실패: {e}")
            raise
    
    def _save_index(self):
        """인덱스 저장"""
        try:
            if self.index is not None:
                # FAISS 인덱스 저장
                faiss.write_index(self.index, str(self.index_file))
                
                # 메타데이터 저장
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
                
                logger.info(f"벡터 인덱스 저장 완료: {len(self.metadata)}개 항목")
                
        except Exception as e:
            logger.error(f"벡터 인덱스 저장 실패: {e}")
            raise
    
    def _get_unprocessed_terms(self) -> List[Dict[str, Any]]:
        """벡터화되지 않은 용어들 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 벡터화되지 않은 용어들 조회
                cursor.execute("""
                    SELECT lt.법령용어ID, lt.법령용어명, lt.법령용어상세검색,
                           ltd.법령용어정의, ltd.출처, ltd.법령용어코드명
                    FROM legal_term_list lt
                    LEFT JOIN legal_term_details ltd ON lt.법령용어명 = ltd.법령용어명_한글
                    WHERE lt.vectorized = FALSE OR lt.vectorized IS NULL
                    ORDER BY lt.id
                """)
                
                terms = []
                for row in cursor.fetchall():
                    term_data = {
                        "법령용어ID": row[0],
                        "법령용어명": row[1],
                        "법령용어상세검색": row[2],
                        "법령용어정의": row[3] or "",
                        "출처": row[4] or "",
                        "법령용어코드명": row[5] or ""
                    }
                    terms.append(term_data)
                
                return terms
                
        except Exception as e:
            logger.error(f"미처리 용어 조회 실패: {e}")
            return []
    
    def _create_text_for_embedding(self, term_data: Dict[str, Any]) -> str:
        """임베딩을 위한 텍스트 생성"""
        # 용어명과 정의를 결합하여 임베딩용 텍스트 생성
        text_parts = []
        
        if term_data["법령용어명"]:
            text_parts.append(f"용어: {term_data['법령용어명']}")
        
        if term_data["법령용어정의"]:
            text_parts.append(f"정의: {term_data['법령용어정의']}")
        
        if term_data["출처"]:
            text_parts.append(f"출처: {term_data['출처']}")
        
        if term_data["법령용어코드명"]:
            text_parts.append(f"분류: {term_data['법령용어코드명']}")
        
        return " | ".join(text_parts)
    
    def _update_term_vectorized_flag(self, term_id: str, success: bool):
        """용어의 벡터화 플래그 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE legal_term_list 
                    SET vectorized = ? 
                    WHERE 법령용어ID = ?
                """, (success, term_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"벡터화 플래그 업데이트 실패: {e}")
    
    async def update_vector_store(self, batch_size: int = 100):
        """벡터스토어 업데이트"""
        logger.info("벡터스토어 업데이트 시작")
        
        # 진행 상황 로드
        self.progress = self._load_progress()
        
        # 기존 인덱스 로드 또는 새로 생성
        if not self._load_existing_index():
            self._create_new_index()
        
        # 미처리 용어들 조회
        unprocessed_terms = self._get_unprocessed_terms()
        total_terms = len(unprocessed_terms)
        
        if total_terms == 0:
            logger.info("업데이트할 용어가 없습니다.")
            return
        
        logger.info(f"벡터화할 용어 수: {total_terms}개")
        
        self.progress.total_terms = total_terms
        self.progress.processed_terms = 0
        self.progress.failed_terms = 0
        
        try:
            # 배치 단위로 처리
            for i in range(0, total_terms, batch_size):
                batch_terms = unprocessed_terms[i:i + batch_size]
                logger.info(f"배치 처리 중 ({i+1}-{min(i+batch_size, total_terms)}/{total_terms})")
                
                # 텍스트 생성
                texts = []
                term_ids = []
                
                for term_data in batch_terms:
                    text = self._create_text_for_embedding(term_data)
                    texts.append(text)
                    term_ids.append(term_data["법령용어ID"])
                
                # 임베딩 생성
                try:
                    embeddings = self.model.encode(texts)
                    
                    # FAISS 인덱스에 추가
                    self.index.add(embeddings.astype('float32'))
                    
                    # 메타데이터 추가
                    for j, term_data in enumerate(batch_terms):
                        metadata_item = {
                            "id": len(self.metadata),
                            "법령용어ID": term_data["법령용어ID"],
                            "법령용어명": term_data["법령용어명"],
                            "법령용어정의": term_data["법령용어정의"],
                            "출처": term_data["출처"],
                            "법령용어코드명": term_data["법령용어코드명"],
                            "text": texts[j]
                        }
                        self.metadata.append(metadata_item)
                        
                        # 벡터화 플래그 업데이트
                        self._update_term_vectorized_flag(term_data["법령용어ID"], True)
                    
                    self.progress.processed_terms += len(batch_terms)
                    logger.info(f"배치 처리 완료: {len(batch_terms)}개")
                    
                except Exception as e:
                    logger.error(f"배치 처리 실패: {e}")
                    self.progress.failed_terms += len(batch_terms)
                    
                    # 실패한 용어들의 플래그 업데이트
                    for term_data in batch_terms:
                        self._update_term_vectorized_flag(term_data["법령용어ID"], False)
                
                # 진행 상황 저장
                self.progress.index_size = len(self.metadata)
                self.progress.last_update_time = str(datetime.now())
                self._save_progress()
                
                # 메모리 사용량이 많아지면 중간 저장
                if (i + batch_size) % (batch_size * 10) == 0:
                    self._save_index()
                    logger.info("중간 저장 완료")
            
            # 최종 저장
            self._save_index()
            
            logger.info(f"벡터스토어 업데이트 완료: {self.progress.processed_terms}개 성공, {self.progress.failed_terms}개 실패")
            
        except Exception as e:
            logger.error(f"벡터스토어 업데이트 중 오류: {e}")
            raise
    
    def search_similar_terms(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """유사한 법률 용어 검색"""
        try:
            if self.index is None or len(self.metadata) == 0:
                logger.warning("벡터 인덱스가 비어있습니다.")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])
            
            # 유사도 검색
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 결과 생성
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result["similarity_score"] = float(1 / (1 + distance))  # 거리를 유사도로 변환
                    result["rank"] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"유사 용어 검색 실패: {e}")
            return []
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """벡터스토어 통계 조회"""
        try:
            stats = {
                "total_vectors": len(self.metadata) if self.metadata else 0,
                "index_size": self.index.ntotal if self.index else 0,
                "vector_dimension": self.vector_dim,
                "progress": asdict(self.progress)
            }
            
            # 데이터베이스에서 벡터화 상태 통계
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM legal_term_list WHERE vectorized = TRUE")
                vectorized_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM legal_term_list WHERE vectorized = FALSE OR vectorized IS NULL")
                unvectorized_count = cursor.fetchone()[0]
                
                stats["vectorized_terms"] = vectorized_count
                stats["unvectorized_terms"] = unvectorized_count
            
            return stats
            
        except Exception as e:
            logger.error(f"벡터스토어 통계 조회 실패: {e}")
            return {}
    
    def rebuild_vector_store(self):
        """벡터스토어 완전 재구성"""
        logger.info("벡터스토어 재구성 시작")
        
        try:
            # 기존 인덱스 삭제
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            # 모든 용어의 벡터화 플래그 초기화
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE legal_term_list SET vectorized = FALSE")
                conn.commit()
            
            # 새로 생성
            self._create_new_index()
            
            # 진행 상황 초기화
            self.progress = VectorUpdateProgress()
            self._save_progress()
            
            logger.info("벡터스토어 재구성 완료")
            
        except Exception as e:
            logger.error(f"벡터스토어 재구성 실패: {e}")
            raise
    
    async def update_legal_terms_vectors(self, terms: List[Dict[str, Any]]) -> bool:
        """법률용어를 기존 벡터스토어에 추가"""
        try:
            logger.info(f"법률용어 벡터 업데이트 시작: {len(terms)}개")
            
            if not terms:
                logger.warning("업데이트할 법률용어가 없습니다")
                return True
            
            # 임베딩 생성할 텍스트 준비
            texts_to_embed = []
            metadata_list = []
            
            for term in terms:
                # 법률용어 정의와 한글명을 결합하여 임베딩 생성
                term_text = f"{term.get('법령용어명_한글', '')} {term.get('법령용어정의', '')}"
                if term.get('출처'):
                    term_text += f" 출처: {term['출처']}"
                
                texts_to_embed.append(term_text)
                
                # 메타데이터 구성
                metadata = {
                    "document_id": f"legal_term_{term.get('법령용어ID', '')}",
                    "document_type": "legal_term",
                    "title": term.get('법령용어명_한글', ''),
                    "content": term_text,
                    "source": term.get('출처', ''),
                    "term_id": term.get('법령용어ID', ''),
                    "term_code": term.get('법령용어코드', ''),
                    "term_code_name": term.get('법령용어코드명', ''),
                    "hanja_name": term.get('법령용어명_한자', ''),
                    "definition": term.get('법령용어정의', ''),
                    "created_at": datetime.now().isoformat()
                }
                metadata_list.append(metadata)
            
            # 임베딩 생성
            logger.info("법률용어 임베딩 생성 중...")
            embeddings = self.model.encode(texts_to_embed, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()
            
            # 기존 벡터스토어 로드
            if not self.load_index():
                logger.error("기존 벡터스토어 로드 실패")
                return False
            
            # 새로운 벡터 추가
            self.index.add(embeddings_np)
            
            # 메타데이터 업데이트
            self.document_metadata.extend(metadata_list)
            
            # 업데이트된 인덱스 저장
            if self.save_index():
                logger.info(f"법률용어 벡터 업데이트 완료: {len(terms)}개 추가")
                
                # 데이터베이스에서 벡터화 상태 업데이트
                await self._update_vectorization_status(terms)
                
                return True
            else:
                logger.error("벡터스토어 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"법률용어 벡터 업데이트 실패: {e}")
            return False
    
    async def _update_vectorization_status(self, terms: List[Dict[str, Any]]):
        """데이터베이스에서 벡터화 상태 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for term in terms:
                    term_id = term.get('법령용어ID')
                    if term_id:
                        cursor.execute("""
                            UPDATE legal_term_details 
                            SET vectorized_at = CURRENT_TIMESTAMP 
                            WHERE 법령용어ID = ?
                        """, (term_id,))
                
                conn.commit()
                logger.info(f"벡터화 상태 업데이트 완료: {len(terms)}개")
                
        except Exception as e:
            logger.error(f"벡터화 상태 업데이트 실패: {e}")
    
    def search_legal_terms(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """법률용어 검색"""
        try:
            if not self.load_index():
                logger.error("벡터스토어 로드 실패")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            query_vector = query_embedding.cpu().numpy()
            
            # 검색 수행
            scores, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_metadata):
                    metadata = self.document_metadata[idx].copy()
                    metadata['similarity_score'] = float(score)
                    metadata['rank'] = i + 1
                    results.append(metadata)
            
            logger.info(f"법률용어 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"법률용어 검색 실패: {e}")
            return []


# 사용 예시
async def main():
    """메인 실행 함수"""
    config = Config()
    
    vector_store = LegalTermVectorStore(config)
    
    try:
        # 벡터스토어 업데이트
        await vector_store.update_vector_store(batch_size=50)
        
        # 통계 출력
        stats = vector_store.get_vector_store_stats()
        logger.info(f"벡터스토어 통계: {stats}")
        
        # 검색 테스트
        results = vector_store.search_similar_terms("계약", top_k=5)
        logger.info(f"검색 결과: {results}")
        
    except Exception as e:
        logger.error(f"벡터스토어 업데이트 중 오류: {e}")


if __name__ == "__main__":
    asyncio.run(main())
