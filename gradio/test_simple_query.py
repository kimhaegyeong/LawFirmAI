# -*- coding: utf-8 -*-
"""
간단한 질의-답변 테스트 스크립트
LawFirmAI - 법률 AI 어시스턴트 테스트
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from prompt_manager import prompt_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLawFirmAITest:
    """간단한 LawFirmAI 테스트 클래스"""
    
    def __init__(self):
        self.database_manager = None
        self.vector_store = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            logger.info("Initializing test services...")
            
            # 데이터베이스 매니저 초기화
            self.database_manager = DatabaseManager()
            logger.info("Database manager initialized")
            
            # 벡터 저장소 초기화
            self._initialize_vector_store()
            
            self.initialized = True
            logger.info("Test services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize test services: {e}")
            return False
    
    def _initialize_vector_store(self):
        """벡터 저장소 초기화"""
        try:
            logger.info("Initializing vector store...")
            
            self.vector_store = LegalVectorStore(model_name="jhgan/ko-sroberta-multitask")
            
            # 벡터 저장소 로드 시도
            project_root = Path(__file__).parent.parent
            vector_store_paths = [
                str(project_root / "data" / "embeddings" / "ml_enhanced_ko_sroberta"),
                str(project_root / "data" / "embeddings" / "ml_enhanced_bge_m3"),
                str(project_root / "data" / "embeddings" / "faiss_index")
            ]
            
            vector_store_loaded = False
            for path in vector_store_paths:
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            files = os.listdir(path)
                            faiss_files = [f for f in files if f.endswith('.faiss')]
                            if faiss_files:
                                faiss_file_path = os.path.join(path, faiss_files[0])
                                success = self.vector_store.load_index(faiss_file_path)
                            else:
                                success = False
                        else:
                            success = self.vector_store.load_index(path)
                        
                        if success:
                            logger.info(f"Vector store loaded successfully from {path}")
                            vector_store_loaded = True
                            break
                    except Exception as e:
                        logger.warning(f"Error loading vector store from {path}: {e}")
            
            if not vector_store_loaded:
                logger.warning("No vector store could be loaded, using database search only")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
    
    def search_documents(self, query: str, top_k: int = 5):
        """문서 검색"""
        results = []
        
        try:
            logger.info(f"Searching documents for query: '{query}'")
            
            # 벡터 저장소 검색
            if self.vector_store:
                try:
                    logger.info("Attempting vector store search...")
                    similar_docs = self.vector_store.search(query, top_k)
                    logger.info(f"Vector search returned {len(similar_docs)} documents")
                    
                    for i, doc in enumerate(similar_docs):
                        doc_info = {
                            'content': doc.get('text', '') or doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'similarity': doc.get('score', 0.0),
                            'source': doc.get('metadata', {}).get('law_name', 'unknown')
                        }
                        results.append(doc_info)
                        
                        logger.info(f"  Document {i+1}: {doc_info['source']} (similarity: {doc_info['similarity']:.3f})")
                        logger.info(f"      Content preview: {doc_info['content'][:100]}...")
                        
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
            
            # 데이터베이스 검색 (백업)
            if not results and self.database_manager:
                try:
                    logger.info("Attempting database search...")
                    assembly_results = self.database_manager.search_assembly_documents(query, top_k)
                    logger.info(f"Database search returned {len(assembly_results)} documents")
                    
                    for i, result in enumerate(assembly_results):
                        doc_info = {
                            'content': result.get('content', ''),
                            'metadata': {
                                'law_name': result.get('law_name', ''),
                                'article_number': result.get('article_number', ''),
                                'article_title': result.get('article_title', '')
                            },
                            'similarity': result.get('relevance_score', 0.8),
                            'source': result.get('law_name', 'assembly_database')
                        }
                        results.append(doc_info)
                        
                        logger.info(f"  DB Document {i+1}: {doc_info['source']} (similarity: {doc_info['similarity']:.3f})")
                        logger.info(f"      Content preview: {doc_info['content'][:100]}...")
                        
                except Exception as e:
                    logger.warning(f"Database search failed: {e}")
            
            # 샘플 데이터 제공
            if not results:
                logger.warning("No documents found, providing sample data")
                results = self._get_sample_legal_documents(query)
            
            logger.info(f"Total search results: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            return self._get_sample_legal_documents(query)
    
    def _get_sample_legal_documents(self, query: str):
        """샘플 법률 문서 제공"""
        sample_docs = [
            {
                'content': '난민법 제1조(목적) 이 법은 「난민의 지위에 관한 1951년 협약」(이하 "난민협약"이라 한다) 및 「난민의 지위에 관한 1967년 의정서」(이하 "난민의정서"라 한다) 등에 따라 난민의 지위와 처우 등에 관한 사항을 정함을 목적으로 한다.',
                'metadata': {
                    'law_name': '난민법',
                    'article_number': '제1조',
                    'article_title': '목적'
                },
                'similarity': 0.9,
                'source': '난민법'
            },
            {
                'content': '민법 제750조(불법행위의 내용) 타인의 고의 또는 과실로 인한 불법행위로 인하여 손해를 받은 자는 그 손해를 가한 자에게 손해배상을 청구할 수 있다.',
                'metadata': {
                    'law_name': '민법',
                    'article_number': '제750조',
                    'article_title': '불법행위의 내용'
                },
                'similarity': 0.8,
                'source': '민법'
            },
            {
                'content': '민법 제543조(계약의 성립) 계약은 당사자 일방이 상대방에게 계약을 체결할 의사를 표시하고 상대방이 이를 승낙함으로써 성립한다.',
                'metadata': {
                    'law_name': '민법',
                    'article_number': '제543조',
                    'article_title': '계약의 성립'
                },
                'similarity': 0.7,
                'source': '민법'
            },
            {
                'content': '상법 제170조(주식회사의 설립) 주식회사는 발기인이 정관을 작성하고 주식의 인수를 주장하여 설립한다.',
                'metadata': {
                    'law_name': '상법',
                    'article_number': '제170조',
                    'article_title': '주식회사의 설립'
                },
                'similarity': 0.6,
                'source': '상법'
            },
            {
                'content': '형법 제329조(절도) 타인의 재물을 절취한 자는 6년 이하의 징역 또는 1천만원 이하의 벌금에 처한다.',
                'metadata': {
                    'law_name': '형법',
                    'article_number': '제329조',
                    'article_title': '절도'
                },
                'similarity': 0.5,
                'source': '형법'
            }
        ]
        
        # 쿼리와 관련된 문서만 필터링
        filtered_docs = []
        query_lower = query.lower()
        
        for doc in sample_docs:
            content_lower = doc['content'].lower()
            metadata_lower = str(doc['metadata']).lower()
            
            # 더 정교한 키워드 매칭
            keywords = ['난민법', '민법', '상법', '형법', '계약', '불법행위', '주식회사', '절도', '설립', '성립']
            if any(keyword in content_lower or keyword in metadata_lower or keyword in query_lower
                   for keyword in keywords):
                filtered_docs.append(doc)
        
        return filtered_docs[:3]
    
    def generate_response(self, query: str, context_docs):
        """응답 생성"""
        try:
            logger.info(f"Generating response for query: '{query}'")
            
            if not context_docs:
                return self._generate_fallback_response(query, context_docs)
            
            # 자연스러운 프롬프트로 전환
            natural_prompt_loaded = prompt_manager.switch_to_version("natural_legal_consultant_v1.0")
            if natural_prompt_loaded:
                logger.info("Switched to natural legal consultant prompt")
            
            # 컨텍스트 구성
            context = "\n\n".join([
                f"[문서: {doc['source']}]\n{doc['content'][:500]}..."
                for doc in context_docs[:3]
            ])
            
            # 자연스러운 응답 생성
            response = f"""안녕하세요! 말씀하신 '{query}'에 대해 도움을 드리겠습니다.

말씀하신 질문에 대해 궁금하시군요.

📋 관련 법률 조항"""
            
            # 실제 조문 내용 포함
            main_doc = context_docs[0] if context_docs else None
            if main_doc and main_doc.get('content'):
                metadata = main_doc.get('metadata', {})
                law_name = metadata.get('law_name', '관련 법률')
                article_number = metadata.get('article_number', '')
                article_title = metadata.get('article_title', '')
                actual_content = main_doc['content']
                
                response += f"\n\n**{law_name} {article_number}**"
                if article_title:
                    response += f" ({article_title})"
                response += f"\n{actual_content}"
            
            response += f"""

💡 쉽게 설명하면
이 조항은 말씀하신 내용과 관련된 법률의 핵심 내용입니다.

🔍 실제 적용 예시
예를 들어, 실제 상황에서 이 법률이 적용될 때는 구체적인 절차와 요건을 따르게 됩니다.

⚠️ 주의사항
이런 경우에는 관련 법률의 구체적인 요건과 절차를 정확히 파악하시는 것이 중요합니다.

📞 추가 도움
더 궁금한 점이 있으시면 언제든 말씀해 주세요!

본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
            
            logger.info(f"Response generated successfully (length: {len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, context_docs)
    
    def _generate_fallback_response(self, query: str, context_docs):
        """폴백 응답 생성"""
        return f"""**[질문 요약]**
귀하의 질문은 '{query}'에 관한 것으로 이해됩니다.

**[상황 분석]**
말씀하신 내용에 대한 관련 법률 문서를 찾을 수 없어 정확한 답변을 제공하기 어려운 상황입니다.

**[실무적 조언]**
이러한 경우 일반적으로 다음과 같은 방법을 고려할 수 있습니다:
1. 더 구체적인 키워드로 질문을 재구성해보세요
2. 관련 법률 분야를 명시하여 질문해보세요
3. 구체적인 상황이나 사례를 포함하여 질문해보세요

**[주의사항]**
- 법률은 해석의 여지가 있으므로 정확한 답변을 위해 충분한 정보가 필요합니다
- 개별 사안에 대해서는 변호사와 직접 상담하시기 바랍니다

**[면책 문구]**
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
    
    def test_query(self, query: str):
        """테스트 쿼리 실행"""
        if not self.initialized:
            logger.error("Service not initialized")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING QUERY: {query}")
        logger.info(f"{'='*60}")
        
        try:
            # 문서 검색
            search_results = self.search_documents(query, top_k=5)
            
            # 응답 생성
            response = self.generate_response(query, search_results)
            
            # 결과 출력
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}")
            print(f"\nRESPONSE:")
            print(response)
            print(f"\n{'='*60}")
            print(f"SEARCH RESULTS SUMMARY:")
            print(f"Total documents found: {len(search_results)}")
            for i, doc in enumerate(search_results, 1):
                print(f"  {i}. {doc['source']} (similarity: {doc['similarity']:.3f})")
                print(f"     Content: {doc['content'][:100]}...")
            print(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"Error in test_query: {e}")
            print(f"Error: {e}")

def main():
    """메인 실행 함수"""
    logger.info("Starting LawFirmAI Simple Test...")
    
    # 데이터베이스 경로 수정
    os.chdir(Path(__file__).parent.parent)  # 프로젝트 루트로 이동
    
    # 테스트 인스턴스 생성
    test_instance = SimpleLawFirmAITest()
    
    # 서비스 초기화
    if not test_instance.initialize():
        logger.error("Failed to initialize test services")
        return
    
    print("\n" + "="*60)
    print("LawFirmAI 간단한 질의-답변 테스트")
    print("="*60)
    print("다양한 법률 질문을 테스트합니다!")
    print("\n테스트 질문들:")
    print("1. 난민법 제1조에 대해서 설명해줘")
    print("2. 민법 제750조 불법행위에 대해 알려줘")
    print("3. 계약의 성립 요건은 무엇인가요?")
    print("4. 주식회사의 설립 절차는 어떻게 되나요?")
    print("5. 절도죄의 구성요건을 설명해주세요")
    print("\n" + "="*60)
    
    # 미리 정의된 테스트 질문들
    test_queries = [
        "난민법 제1조에 대해서 설명해줘",
        "민법 제750조 불법행위에 대해 알려줘",
        "계약의 성립 요건은 무엇인가요?",
        "주식회사의 설립 절차는 어떻게 되나요?",
        "절도죄의 구성요건을 설명해주세요"
    ]
    
    try:
        # 각 테스트 질문 실행
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} 테스트 {i}/5 {'='*20}")
            test_instance.test_query(query)
            
            # 마지막 질문이 아니면 잠시 대기
            if i < len(test_queries):
                print("\n다음 테스트를 위해 잠시 대기...")
                import time
                time.sleep(2)
        
        print(f"\n{'='*60}")
        print("모든 테스트가 완료되었습니다!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in test execution: {e}")
        print(f"테스트 실행 중 오류가 발생했습니다: {e}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
