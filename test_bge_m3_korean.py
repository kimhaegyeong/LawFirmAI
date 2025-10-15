#!/usr/bin/env python3
"""
BGE-M3-Korean 모델 테스트 스크립트
"""

import sys
import os
from pathlib import Path
import logging

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Windows 콘솔에서 UTF-8 인코딩 설정
if os.name == 'nt':  # Windows
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except AttributeError:
        # 이미 UTF-8로 설정된 경우 무시
        pass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_bge_m3_korean():
    """BGE-M3-Korean 모델 테스트"""
    try:
        # FlagEmbedding 라이브러리 테스트
        logger.info("Testing FlagEmbedding library...")
        from FlagEmbedding import FlagModel
        
        # BGE-M3 모델 로딩
        logger.info("Loading BGE-M3 model...")
        model = FlagModel("BAAI/bge-m3", query_instruction_for_retrieval="")
        
        # 테스트 텍스트
        test_texts = [
            "계약서 검토 요청",
            "민법 제1조는 민법의 기본 원칙을 규정한다",
            "부동산 매매계약서 작성 방법",
            "손해배상 청구권의 시효는 3년이다"
        ]
        
        logger.info("Generating embeddings...")
        embeddings = model.encode(test_texts, normalize_embeddings=True)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Embeddings dimension: {embeddings.shape[1]}")
        
        # LegalVectorStore 테스트
        logger.info("Testing LegalVectorStore with BGE-M3...")
        from source.data.vector_store import LegalVectorStore
        
        vector_store = LegalVectorStore(
            model_name="BAAI/bge-m3",
            dimension=1024,
            index_type="flat"
        )
        
        # 문서 추가 테스트
        test_documents = [
            {
                'text': '민법 제1조는 민법의 기본 원칙을 규정한다',
                'metadata': {
                    'law_name': '민법',
                    'article_number': '제1조',
                    'category': '민사법'
                }
            },
            {
                'text': '계약서 검토 시 주의사항',
                'metadata': {
                    'law_name': '계약법',
                    'article_number': '제1조',
                    'category': '민사법'
                }
            }
        ]
        
        texts = [doc['text'] for doc in test_documents]
        metadatas = [doc['metadata'] for doc in test_documents]
        
        success = vector_store.add_documents(texts, metadatas)
        logger.info(f"Documents added successfully: {success}")
        
        # 검색 테스트
        logger.info("Testing search functionality...")
        results = vector_store.search("계약서 검토", top_k=2)
        
        logger.info(f"Search results: {len(results)}")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: Score={result['score']:.4f}, Text={result['text'][:50]}...")
        
        # 통계 정보
        stats = vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
        logger.info("BGE-M3-Korean test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install FlagEmbedding: pip install FlagEmbedding")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_sentence_bert_comparison():
    """Sentence-BERT와 BGE-M3 비교 테스트"""
    try:
        logger.info("Testing Sentence-BERT vs BGE-M3 comparison...")
        
        from source.data.vector_store import LegalVectorStore
        
        # Sentence-BERT 모델
        logger.info("Testing Sentence-BERT model...")
        st_model = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat"
        )
        
        # BGE-M3 모델
        logger.info("Testing BGE-M3 model...")
        bge_model = LegalVectorStore(
            model_name="BAAI/bge-m3",
            dimension=1024,
            index_type="flat"
        )
        
        # 테스트 텍스트
        test_text = "계약서 검토 요청"
        
        # 임베딩 생성 시간 비교
        import time
        
        # Sentence-BERT
        start_time = time.time()
        st_embedding = st_model.generate_embeddings([test_text])
        st_time = time.time() - start_time
        
        # BGE-M3
        start_time = time.time()
        bge_embedding = bge_model.generate_embeddings([test_text])
        bge_time = time.time() - start_time
        
        logger.info(f"Sentence-BERT embedding time: {st_time:.4f}s, dimension: {st_embedding.shape[1]}")
        logger.info(f"BGE-M3 embedding time: {bge_time:.4f}s, dimension: {bge_embedding.shape[1]}")
        
        logger.info("Comparison test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Comparison test failed: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("Starting BGE-M3-Korean model tests...")
    
    # 기본 테스트
    success1 = test_bge_m3_korean()
    
    # 비교 테스트
    success2 = test_sentence_bert_comparison()
    
    if success1 and success2:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
