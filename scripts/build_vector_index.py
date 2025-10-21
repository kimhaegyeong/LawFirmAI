# -*- coding: utf-8 -*-
"""
벡터 인덱스 구축 스크립트
법률 문서의 벡터 임베딩을 생성하고 FAISS 인덱스를 구축합니다.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.data.vector_store import LegalVectorStore
from source.data.database import DatabaseManager
from source.utils.config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/vector_index_build.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class VectorIndexBuilder:
    """벡터 인덱스 구축 클래스"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = LegalVectorStore(
            model_name="jhgan/ko-sroberta-multitask",
            dimension=768,
            index_type="flat",
            enable_quantization=True,
            enable_lazy_loading=False  # 즉시 로딩으로 변경
        )
        self.database_manager = DatabaseManager()
        
    def build_index_from_database(self) -> bool:
        """데이터베이스에서 법률 문서를 읽어 벡터 인덱스 구축"""
        try:
            logger.info("벡터 인덱스 구축 시작...")
            
            # 1. 데이터베이스에서 법률 문서 로드
            documents = self._load_documents_from_database()
            if not documents:
                logger.error("데이터베이스에서 문서를 로드할 수 없습니다.")
                return False
            
            logger.info(f"총 {len(documents)}개의 문서를 로드했습니다.")
            
            # 2. 문서를 청크로 분할
            chunked_documents = self._chunk_documents(documents)
            logger.info(f"문서를 {len(chunked_documents)}개의 청크로 분할했습니다.")
            
            # 3. 벡터 스토어에 추가
            success = self.vector_store.add_documents_legacy(chunked_documents)
            if not success:
                logger.error("벡터 스토어에 문서 추가 실패")
                return False
            
            # 4. 인덱스 저장
            index_path = "data/embeddings/legal_vector_index"
            success = self.vector_store.save_index(index_path)
            if not success:
                logger.error("인덱스 저장 실패")
                return False
            
            # 5. 통계 정보 출력
            stats = self.vector_store.get_stats()
            logger.info("벡터 인덱스 구축 완료!")
            logger.info(f"통계: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"벡터 인덱스 구축 중 오류 발생: {e}")
            return False
    
    def _load_documents_from_database(self) -> List[Dict[str, Any]]:
        """데이터베이스에서 법률 문서 로드"""
        documents = []
        
        try:
            # assembly_articles 테이블에서 로드 (실제 법률 조문 데이터)
            articles_query = """
            SELECT 
                aa.id,
                aa.law_id,
                aa.article_number,
                aa.article_title,
                aa.article_content,
                aa.sub_articles,
                aa.law_references,
                aa.word_count,
                aa.char_count,
                aa.is_supplementary,
                aa.article_type,
                aa.parsing_quality_score,
                al.law_name,
                al.law_type,
                al.category,
                al.promulgation_date,
                al.enforcement_date
            FROM assembly_articles aa
            LEFT JOIN assembly_laws al ON aa.law_id = al.law_id
            WHERE aa.article_content IS NOT NULL AND aa.article_content != ''
            ORDER BY aa.law_id, aa.article_number
            LIMIT 10000
            """
            
            articles_rows = self.database_manager.execute_query(articles_query)
            logger.info(f"assembly_articles 테이블에서 {len(articles_rows)}개 조문 로드")
            
            for row in articles_rows:
                # 조문 제목과 내용을 결합하여 문서 생성
                title = row['article_title'] or f"제{row['article_number']}조"
                content = f"{title}\n\n{row['article_content']}"
                
                document = {
                    'id': f"article_{row['id']}",
                    'type': 'article',
                    'law_name': row['law_name'] or 'Unknown Law',
                    'category': row['category'] or 'general',
                    'cleaned_content': content,
                    'summary': f"{row['law_name']} {title}",
                    'metadata': {
                        'law_id': row['law_id'],
                        'article_number': row['article_number'],
                        'article_title': row['article_title'],
                        'law_type': row['law_type'],
                        'is_supplementary': row['is_supplementary'],
                        'article_type': row['article_type'],
                        'word_count': row['word_count'],
                        'char_count': row['char_count'],
                        'parsing_quality_score': row['parsing_quality_score'],
                        'promulgation_date': row['promulgation_date'],
                        'enforcement_date': row['enforcement_date']
                    }
                }
                documents.append(document)
            
            logger.info(f"총 {len(documents)}개의 법률 조문을 데이터베이스에서 로드했습니다.")
            return documents
            
        except Exception as e:
            logger.error(f"데이터베이스에서 문서 로드 실패: {e}")
            return []
    
    def _chunk_documents(self, documents: List[Dict[str, Any]], 
                        chunk_size: int = 1000, 
                        overlap: int = 200) -> List[Dict[str, Any]]:
        """문서를 청크로 분할"""
        chunked_documents = []
        
        for doc in documents:
            content = doc.get('cleaned_content', '')
            if not content:
                continue
            
            # 긴 문서만 청크로 분할
            if len(content) <= chunk_size:
                # 짧은 문서는 그대로 사용
                chunked_doc = doc.copy()
                chunked_doc['chunks'] = [{
                    'id': f"{doc['id']}_chunk_0",
                    'text': content,
                    'start_pos': 0,
                    'end_pos': len(content),
                    'entities': self._extract_entities(content)
                }]
                chunked_documents.append(chunked_doc)
            else:
                # 긴 문서는 청크로 분할
                chunks = []
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = min(start + chunk_size, len(content))
                    
                    # 문장 경계에서 자르기 시도
                    if end < len(content):
                        # 마침표, 느낌표, 물음표에서 자르기
                        for punct in ['.', '!', '?', '。', '！', '？']:
                            punct_pos = content.rfind(punct, start, end)
                            if punct_pos > start + chunk_size // 2:  # 너무 앞에서 자르지 않도록
                                end = punct_pos + 1
                                break
                    
                    chunk_text = content[start:end].strip()
                    if chunk_text:
                        chunk = {
                            'id': f"{doc['id']}_chunk_{chunk_id}",
                            'text': chunk_text,
                            'start_pos': start,
                            'end_pos': end,
                            'entities': self._extract_entities(chunk_text)
                        }
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    start = end - overlap
                    if start >= len(content):
                        break
                
                if chunks:
                    chunked_doc = doc.copy()
                    chunked_doc['chunks'] = chunks
                    chunked_documents.append(chunked_doc)
        
        return chunked_documents
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 법률 관련 엔티티 추출 (간단한 버전)"""
        entities = {
            'laws': [],
            'articles': [],
            'dates': [],
            'keywords': []
        }
        
        # 간단한 법률명 패턴 매칭
        import re
        
        # 법률명 패턴 (예: "민법", "형법", "상법" 등)
        law_patterns = [
            r'([가-힣]+법)',
            r'([가-힣]+법률)',
            r'([가-힣]+규칙)',
            r'([가-힣]+령)'
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, text)
            entities['laws'].extend(matches)
        
        # 조문 번호 패턴 (예: "제1조", "제2항" 등)
        article_patterns = [
            r'제(\d+)조',
            r'제(\d+)항',
            r'제(\d+)호',
            r'(\d+)조',
            r'(\d+)항'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, text)
            entities['articles'].extend(matches)
        
        # 날짜 패턴
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
            r'(\d{4})-(\d{1,2})-(\d{1,2})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities['dates'].extend(['-'.join(match) for match in matches])
        
        # 법률 키워드
        legal_keywords = [
            '계약', '손해배상', '이혼', '재산분할', '양육비', '친권',
            '형사처벌', '변호인', '소송', '법원', '판결', '청구',
            '요건', '효력', '무효', '취소', '해지', '해제'
        ]
        
        for keyword in legal_keywords:
            if keyword in text:
                entities['keywords'].append(keyword)
        
        return entities
    
    def test_index(self) -> bool:
        """구축된 인덱스 테스트"""
        try:
            logger.info("벡터 인덱스 테스트 시작...")
            
            test_queries = [
                "계약서 작성 방법",
                "이혼 절차",
                "손해배상 청구",
                "형사처벌 기준",
                "재산분할 원칙"
            ]
            
            for query in test_queries:
                logger.info(f"테스트 쿼리: {query}")
                results = self.vector_store.search(query, top_k=3)
                
                logger.info(f"결과 수: {len(results)}")
                for i, result in enumerate(results):
                    logger.info(f"  {i+1}. 점수: {result['score']:.3f}")
                    logger.info(f"     텍스트: {result['text'][:100]}...")
                    logger.info(f"     메타데이터: {result['metadata']}")
            
            logger.info("벡터 인덱스 테스트 완료!")
            return True
            
        except Exception as e:
            logger.error(f"벡터 인덱스 테스트 실패: {e}")
            return False

async def main():
    """메인 함수"""
    print("=" * 60)
    print("벡터 인덱스 구축 스크립트")
    print("=" * 60)
    
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    Path("data/embeddings").mkdir(parents=True, exist_ok=True)
    
    # 설정 로드
    config = Config()
    
    # 벡터 인덱스 구축
    builder = VectorIndexBuilder(config)
    
    # 인덱스 구축
    success = builder.build_index_from_database()
    if not success:
        logger.error("벡터 인덱스 구축 실패")
        return
    
    # 인덱스 테스트
    test_success = builder.test_index()
    if not test_success:
        logger.error("벡터 인덱스 테스트 실패")
        return
    
    print("\n" + "=" * 60)
    print("벡터 인덱스 구축 완료!")
    print("=" * 60)
    
    # 통계 정보 출력
    stats = builder.vector_store.get_stats()
    print(f"문서 수: {stats['documents_count']}")
    print(f"인덱스 타입: {stats['index_type']}")
    print(f"임베딩 차원: {stats['embedding_dimension']}")
    print(f"모델명: {stats['model_name']}")
    print(f"양자화: {'활성화' if stats['quantization_enabled'] else '비활성화'}")
    print(f"지연 로딩: {'활성화' if stats['lazy_loading_enabled'] else '비활성화'}")

if __name__ == "__main__":
    asyncio.run(main())
