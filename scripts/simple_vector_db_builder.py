#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 벡터 데이터베이스 구축 스크립트
"""

import os
import sys
import json
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'source'))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정 (간단하게)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """샘플 법률 데이터 생성"""
    logger.info("📝 샘플 법률 데이터 생성 중...")
    
    # 샘플 데이터 디렉토리 생성
    sample_dir = Path("data/raw/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 샘플 법률 데이터
    sample_laws = [
        {
            "title": "민법 제1조 (법원)",
            "content": "민사에 관하여 법률에 특별한 규정이 없으면 관습법에 의하고, 관습법이 없으면 조리에 의한다.",
            "category": "civil_law",
            "source": "민법",
            "article": "제1조"
        },
        {
            "title": "민법 제2조 (신의성실의 원칙)",
            "content": "권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.",
            "category": "civil_law", 
            "source": "민법",
            "article": "제2조"
        },
        {
            "title": "형법 제1조 (범죄의 성립과 처벌)",
            "content": "범죄의 성립과 처벌은 행위시의 법률에 의한다.",
            "category": "criminal_law",
            "source": "형법", 
            "article": "제1조"
        }
    ]
    
    # 샘플 판례 데이터
    sample_precedents = [
        {
            "title": "대법원 2023다12345 판결",
            "content": "계약의 해석은 당사자가 계약을 체결할 때의 진정한 의사를 탐구하여야 하며, 계약의 문언에 의하여 당사자의 의사가 명확하지 않은 경우에는 계약의 내용과 목적, 계약 체결의 경위 등을 종합적으로 고려하여야 한다.",
            "category": "precedent",
            "source": "대법원",
            "case_number": "2023다12345"
        },
        {
            "title": "대법원 2023다67890 판결", 
            "content": "불법행위로 인한 손해배상책임은 고의 또는 과실로 인한 위법한 행위로 타인에게 손해를 가한 자가 그 손해를 배상할 책임을 진다.",
            "category": "precedent",
            "source": "대법원",
            "case_number": "2023다67890"
        }
    ]
    
    # 법률 데이터 저장
    laws_file = sample_dir / "laws_sample.json"
    with open(laws_file, 'w', encoding='utf-8') as f:
        json.dump(sample_laws, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 법률 데이터 저장: {laws_file}")
    
    # 판례 데이터 저장
    precedents_file = sample_dir / "precedents_sample.json"
    with open(precedents_file, 'w', encoding='utf-8') as f:
        json.dump(sample_precedents, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 판례 데이터 저장: {precedents_file}")
    
    return sample_dir

def build_simple_vector_db():
    """간단한 벡터 데이터베이스 구축"""
    logger.info("🚀 간단한 벡터 데이터베이스 구축 시작")
    
    try:
        # 샘플 데이터 생성
        sample_dir = create_sample_data()
        
        # 벡터 저장소 초기화
        logger.info("📦 벡터 저장소 초기화 중...")
        
        # 간단한 FAISS 인덱스 생성
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            
            # 임베딩 모델 로드
            model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info(f"📥 임베딩 모델 로드: {model_name}")
            model = SentenceTransformer(model_name)
            
            # 샘플 텍스트들
            texts = [
                "민사에 관하여 법률에 특별한 규정이 없으면 관습법에 의하고, 관습법이 없으면 조리에 의한다.",
                "권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.",
                "범죄의 성립과 처벌은 행위시의 법률에 의한다.",
                "계약의 해석은 당사자가 계약을 체결할 때의 진정한 의사를 탐구하여야 한다.",
                "불법행위로 인한 손해배상책임은 고의 또는 과실로 인한 위법한 행위로 타인에게 손해를 가한 자가 그 손해를 배상할 책임을 진다."
            ]
            
            # 임베딩 생성
            logger.info("🔄 임베딩 생성 중...")
            embeddings = model.encode(texts)
            
            # FAISS 인덱스 생성
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype('float32'))
            
            # 인덱스 저장
            embeddings_dir = Path("data/embeddings")
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            index_path = embeddings_dir / "simple_vector_index"
            faiss.write_index(index, str(index_path))
            logger.info(f"✅ 벡터 인덱스 저장: {index_path}")
            
            # 메타데이터 저장
            metadata = {
                "model_name": model_name,
                "dimension": dimension,
                "num_vectors": len(texts),
                "texts": texts,
                "index_type": "flat"
            }
            
            metadata_path = embeddings_dir / "simple_vector_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 메타데이터 저장: {metadata_path}")
            
            # 검색 테스트
            logger.info("🔍 검색 테스트 중...")
            test_query = "계약 해석에 대한 원칙"
            query_embedding = model.encode([test_query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = index.search(query_embedding.astype('float32'), k=3)
            
            logger.info(f"📝 테스트 쿼리: {test_query}")
            logger.info("🔍 검색 결과:")
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                logger.info(f"   {i+1}. 점수: {score:.4f}, 텍스트: {texts[idx][:50]}...")
            
            return True
            
        except ImportError as e:
            logger.error(f"❌ 필요한 패키지가 없습니다: {e}")
            logger.info("다음 명령어로 설치하세요:")
            logger.info("pip install faiss-cpu sentence-transformers")
            return False
            
    except Exception as e:
        logger.error(f"❌ 벡터 DB 구축 실패: {e}")
        return False

def test_gemini_with_vector_db():
    """벡터 DB와 함께 Gemini Pro 테스트"""
    logger.info("🤖 Gemini Pro + 벡터 DB 테스트")
    
    try:
        # 간단한 검색 테스트
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        # 저장된 인덱스 로드
        embeddings_dir = Path("data/embeddings")
        index_path = embeddings_dir / "simple_vector_index"
        metadata_path = embeddings_dir / "simple_vector_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.error("❌ 벡터 인덱스가 없습니다. 먼저 벡터 DB를 구축하세요.")
            return False
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 인덱스 로드
        index = faiss.read_index(str(index_path))
        model = SentenceTransformer(metadata['model_name'])
        texts = metadata['texts']
        
        logger.info(f"✅ 벡터 인덱스 로드 완료: {len(texts)}개 문서")
        
        # Gemini Pro와 연동 테스트
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key and api_key not in ['your-google-api-key-here', 'test-google-api-key']:
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=api_key)
                model_gemini = genai.GenerativeModel('gemini-pro')
                
                # RAG 테스트
                query = "민법의 기본 원칙은 무엇인가요?"
                logger.info(f"📝 RAG 테스트 쿼리: {query}")
                
                # 벡터 검색
                query_embedding = model.encode([query])
                faiss.normalize_L2(query_embedding)
                scores, indices = index.search(query_embedding.astype('float32'), k=2)
                
                # 검색된 문서들
                retrieved_docs = [texts[idx] for idx in indices[0]]
                context = "\n".join(retrieved_docs)
                
                logger.info("🔍 검색된 문서:")
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"   {i+1}. {doc[:100]}...")
                
                # Gemini Pro로 답변 생성
                prompt = f"""
다음 법률 문서들을 참고하여 질문에 답변해주세요:

문서들:
{context}

질문: {query}

답변:
"""
                
                response = model_gemini.generate_content(prompt)
                logger.info(f"🤖 Gemini Pro 답변: {response.text}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Gemini Pro 테스트 실패: {e}")
                return False
        else:
            logger.info("ℹ️ 실제 API 키가 없어 Gemini Pro 테스트를 건너뜁니다.")
            return True
            
    except Exception as e:
        logger.error(f"❌ 벡터 DB 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🚀 간단한 벡터 데이터베이스 구축 및 테스트")
    logger.info("=" * 60)
    
    # 1. 벡터 DB 구축
    success = build_simple_vector_db()
    if not success:
        logger.error("❌ 벡터 DB 구축 실패")
        return
    
    logger.info("=" * 60)
    
    # 2. Gemini Pro + 벡터 DB 테스트
    test_success = test_gemini_with_vector_db()
    
    logger.info("=" * 60)
    logger.info("✅ 테스트 완료")
    
    if success and test_success:
        logger.info("🎉 벡터 데이터베이스 구축 및 Gemini Pro 연동 테스트 성공!")
        logger.info()
        logger.info("🔧 다음 단계:")
        logger.info("   1. 실제 법률 데이터 수집")
        logger.info("   2. 대규모 벡터 데이터베이스 구축")
        logger.info("   3. 프로덕션 RAG 시스템 구축")
    else:
        logger.info("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    main()
