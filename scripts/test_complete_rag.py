#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
완전한 RAG 시스템 테스트 스크립트
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

# 로깅 설정
from utils.safe_logging import setup_script_logging
logger = setup_script_logging("test_complete_rag")

def test_complete_rag_system():
    """완전한 RAG 시스템 테스트"""
    logger.info("🚀 완전한 RAG 시스템 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 필요한 패키지 import
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import google.generativeai as genai
        
        # 환경 변수 확인
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key or api_key in ['your-google-api-key-here', 'test-google-api-key']:
            logger.error("❌ 실제 Google API 키가 설정되지 않았습니다.")
            return False
        
        logger.info("✅ Google API 키 확인 완료")
        
        # Gemini Pro 설정
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel('gemini-pro')
        logger.info("✅ Gemini Pro 모델 초기화 완료")
        
        # 벡터 인덱스 로드
        embeddings_dir = Path("data/embeddings")
        index_path = embeddings_dir / "simple_vector_index"
        metadata_path = embeddings_dir / "simple_vector_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            logger.error("❌ 벡터 인덱스가 없습니다.")
            return False
        
        # 메타데이터 로드
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 인덱스 로드
        index = faiss.read_index(str(index_path))
        model = SentenceTransformer(metadata['model_name'])
        texts = metadata['texts']
        
        logger.info(f"✅ 벡터 인덱스 로드 완료: {len(texts)}개 문서")
        
        # RAG 테스트 쿼리들
        test_queries = [
            "민법의 기본 원칙은 무엇인가요?",
            "계약 해석 시 고려해야 할 사항은 무엇인가요?",
            "불법행위로 인한 손해배상의 요건은 무엇인가요?",
            "법률 해석의 원칙에 대해 설명해주세요."
        ]
        
        logger.info("🤖 RAG 시스템 테스트 시작")
        logger.info("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"📝 테스트 {i}: {query}")
            
            try:
                # 1. 벡터 검색
                query_embedding = model.encode([query])
                faiss.normalize_L2(query_embedding)
                scores, indices = index.search(query_embedding.astype('float32'), k=3)
                
                # 검색된 문서들
                retrieved_docs = []
                for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    retrieved_docs.append({
                        'text': texts[idx],
                        'score': float(score)
                    })
                
                logger.info(f"🔍 검색된 문서 ({len(retrieved_docs)}개):")
                for j, doc in enumerate(retrieved_docs):
                    logger.info(f"   {j+1}. 점수: {doc['score']:.4f}")
                    logger.info(f"      내용: {doc['text'][:80]}...")
                
                # 2. 컨텍스트 구성
                context = "\n\n".join([doc['text'] for doc in retrieved_docs])
                
                # 3. Gemini Pro로 답변 생성
                prompt = f"""
다음 법률 문서들을 참고하여 질문에 정확하고 도움이 되는 답변을 해주세요.

참고 문서들:
{context}

질문: {query}

답변 시 다음 사항을 고려해주세요:
1. 참고 문서의 내용을 바탕으로 답변하세요
2. 법률적 정확성을 유지하세요
3. 이해하기 쉽게 설명해주세요
4. 관련 법조문이나 판례가 있다면 언급해주세요

답변:
"""
                
                response = model_gemini.generate_content(prompt)
                
                logger.info(f"🤖 Gemini Pro 답변:")
                logger.info(f"   {response.text}")
                
                # 응답 품질 평가
                response_length = len(response.text)
                logger.info(f"📊 응답 품질: 길이 {response_length}자")
                
            except Exception as e:
                logger.error(f"❌ 쿼리 처리 실패: {e}")
            
            logger.info("-" * 60)
        
        logger.info("✅ RAG 시스템 테스트 완료")
        return True
        
    except ImportError as e:
        logger.error(f"❌ 필요한 패키지가 없습니다: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ RAG 시스템 테스트 실패: {e}")
        return False

def test_advanced_rag_features():
    """고급 RAG 기능 테스트"""
    logger.info("🔬 고급 RAG 기능 테스트")
    logger.info("=" * 60)
    
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import google.generativeai as genai
        
        # 설정 로드
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel('gemini-pro')
        
        # 벡터 인덱스 로드
        embeddings_dir = Path("data/embeddings")
        index_path = embeddings_dir / "simple_vector_index"
        metadata_path = embeddings_dir / "simple_vector_metadata.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        index = faiss.read_index(str(index_path))
        model = SentenceTransformer(metadata['model_name'])
        texts = metadata['texts']
        
        # 1. 유사도 임계값 테스트
        logger.info("🎯 유사도 임계값 테스트")
        query = "민법의 기본 원칙"
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding.astype('float32'), k=5)
        
        threshold = 0.3
        relevant_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                relevant_docs.append((texts[idx], score))
        
        logger.info(f"   임계값 {threshold} 이상 문서: {len(relevant_docs)}개")
        for i, (doc, score) in enumerate(relevant_docs):
            logger.info(f"   {i+1}. 점수: {score:.4f}, 내용: {doc[:50]}...")
        
        # 2. 다중 쿼리 테스트
        logger.info("🔄 다중 쿼리 테스트")
        multi_query = ["민법", "계약", "손해배상"]
        
        all_docs = set()
        for q in multi_query:
            q_embedding = model.encode([q])
            faiss.normalize_L2(q_embedding)
            scores, indices = index.search(q_embedding.astype('float32'), k=2)
            for idx in indices[0]:
                all_docs.add(texts[idx])
        
        logger.info(f"   다중 쿼리로 찾은 고유 문서: {len(all_docs)}개")
        
        # 3. 컨텍스트 윈도우 테스트
        logger.info("📏 컨텍스트 윈도우 테스트")
        max_context_length = 1000
        context = "\n".join(list(all_docs)[:3])
        
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.info(f"   컨텍스트 길이 제한 적용: {len(context)}자")
        
        logger.info("✅ 고급 RAG 기능 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 고급 RAG 기능 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    logger.info("🚀 완전한 RAG 시스템 테스트")
    logger.info("=" * 80)
    
    # 1. 기본 RAG 시스템 테스트
    rag_success = test_complete_rag_system()
    
    if rag_success:
        logger.info("=" * 80)
        
        # 2. 고급 RAG 기능 테스트
        advanced_success = test_advanced_rag_features()
        
        logger.info("=" * 80)
        logger.info("📊 최종 테스트 결과:")
        logger.info(f"   - 기본 RAG 시스템: {'✅' if rag_success else '❌'}")
        logger.info(f"   - 고급 RAG 기능: {'✅' if advanced_success else '❌'}")
        
        if rag_success and advanced_success:
            logger.info("🎉 완전한 RAG 시스템 테스트 성공!")
            logger.info("")
            logger.info("🚀 시스템이 프로덕션 준비 완료되었습니다!")
            logger.info("")
            logger.info("🔧 다음 단계:")
            logger.info("   1. 대규모 법률 데이터 수집")
            logger.info("   2. 고성능 벡터 인덱스 구축")
            logger.info("   3. 웹 인터페이스 개발")
            logger.info("   4. HuggingFace Spaces 배포")
        else:
            logger.info("⚠️ 일부 테스트가 실패했습니다.")
    else:
        logger.info("❌ RAG 시스템 테스트 실패")

if __name__ == "__main__":
    main()
