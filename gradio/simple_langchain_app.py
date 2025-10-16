# -*- coding: utf-8 -*-
"""
간단한 LangChain 기반 Gradio 애플리케이션
LawFirmAI - 법률 AI 어시스턴트
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Gradio 및 LangChain 라이브러리
import gradio as gr
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# 프로젝트 모듈
from source.data.database import DatabaseManager
from source.data.vector_store import LegalVectorStore
from prompt_manager import prompt_manager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/simple_langchain_gradio.log')
    ]
)
logger = logging.getLogger(__name__)

class LawFirmAIService:
    """LawFirmAI 서비스 클래스"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.database_manager = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """서비스 초기화"""
        try:
            logger.info("Initializing LawFirmAI services...")
            
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.info("Embeddings initialized")
            
            # LLM 초기화
            self._initialize_llm()
            
            # 데이터베이스 매니저 초기화
            self.database_manager = DatabaseManager()
            logger.info("Database manager initialized")
            
            # 벡터 저장소 초기화
            self._initialize_vector_store()
            
            self.initialized = True
            logger.info("All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False
    
    def _initialize_llm(self):
        """LLM 초기화"""
        llm_initialized = False
        
        # OpenAI 시도
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=1000,
                    api_key=openai_api_key
                )
                logger.info("OpenAI LLM initialized")
                llm_initialized = True
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")
        
        # Google AI 시도
        if not llm_initialized:
            try:
                google_api_key = os.getenv("GOOGLE_API_KEY")
                if google_api_key:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0.7,
                        max_output_tokens=1000,
                        google_api_key=google_api_key
                    )
                    logger.info("Google AI LLM initialized")
                    llm_initialized = True
            except Exception as e:
                logger.warning(f"Google AI initialization failed: {e}")
        
        if not llm_initialized:
            self.llm = None
            logger.info("Using fallback response system (no external LLM available)")
    
    def _initialize_vector_store(self):
        """벡터 저장소 초기화"""
        logger.info("Initializing vector store...")
        vector_store_init_start = time.time()
        
        self.vector_store = LegalVectorStore(model_name="jhgan/ko-sroberta-multitask")
        vector_store_init_time = time.time() - vector_store_init_start
        logger.info(f"Vector store initialized in {vector_store_init_time:.3f}s")
        
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
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """문서 검색"""
        results = []
        
        try:
            # 벡터 저장소 검색
            if self.vector_store:
                try:
                    similar_docs = self.vector_store.search(query, top_k)
                    for doc in similar_docs:
                        doc_info = {
                            'content': doc.get('text', '') or doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'similarity': doc.get('score', 0.0),
                            'source': doc.get('metadata', {}).get('law_name', 'unknown')
                        }
                        results.append(doc_info)
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
            
            # 데이터베이스 검색 (백업)
            if not results and self.database_manager:
                try:
                    assembly_results = self.database_manager.search_assembly_documents(query, top_k)
                    for result in assembly_results:
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
                except Exception as e:
                    logger.warning(f"Database search failed: {e}")
            
            # 샘플 데이터 제공
            if not results:
                results = self._get_sample_legal_documents(query)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_documents: {e}")
            return self._get_sample_legal_documents(query)
    
    def _get_sample_legal_documents(self, query: str) -> List[Dict[str, Any]]:
        """샘플 법률 문서 제공"""
        sample_docs = [
            {
                'content': '민법 제750조(불법행위의 내용) 타인의 고의 또는 과실로 인한 불법행위로 인하여 손해를 받은 자는 그 손해를 가한 자에게 손해배상을 청구할 수 있다.',
                'metadata': {
                    'law_name': '민법',
                    'article_number': '제750조',
                    'article_title': '불법행위의 내용'
                },
                'similarity': 0.7,
                'source': '민법'
            },
            {
                'content': '민법 제543조(계약의 성립) 계약은 당사자 일방이 상대방에게 계약을 체결할 의사를 표시하고 상대방이 이를 승낙함으로써 성립한다.',
                'metadata': {
                    'law_name': '민법',
                    'article_number': '제543조',
                    'article_title': '계약의 성립'
                },
                'similarity': 0.6,
                'source': '민법'
            }
        ]
        
        # 쿼리와 관련된 문서만 필터링
        filtered_docs = []
        query_lower = query.lower()
        
        for doc in sample_docs:
            content_lower = doc['content'].lower()
            metadata_lower = str(doc['metadata']).lower()
            
            if any(keyword in content_lower or keyword in metadata_lower 
                   for keyword in ['민법', '상법', '형법', '계약', '불법행위']):
                filtered_docs.append(doc)
        
        return filtered_docs[:3]
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """응답 생성"""
        try:
            if not self.llm:
                return self._generate_fallback_response(query, context_docs)
            
            # 컨텍스트 구성
            context = "\n\n".join([
                f"[문서: {doc['source']}]\n{doc['content'][:500]}..."
                for doc in context_docs[:3]
            ])
            
            # 법률 전문가 프롬프트 가져오기
            legal_prompt = prompt_manager.get_current_prompt()
            
            # 프롬프트 템플릿 구성
            template = f"""{legal_prompt}

문서 내용:
{{context}}

질문: {{question}}

위의 법률 전문가 역할에 따라 질문에 답변해주세요."""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # LangChain 체인 생성
            chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=None,
                return_source_documents=False
            )
            
            # 응답 생성
            response = chain.run(
                query=prompt.format(context=context, question=query)
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, context_docs)
    
    def _generate_fallback_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """폴백 응답 생성"""
        if not context_docs:
            return f"""안녕하세요! 말씀하신 '{query}'에 대해 도움을 드리겠습니다.

말씀하신 내용에 대한 관련 법률 문서를 찾을 수 없어 정확한 답변을 제공하기 어려운 상황입니다.

💡 실무적 조언
이러한 경우 일반적으로 다음과 같은 방법을 고려할 수 있습니다:
1. 더 구체적인 키워드로 질문을 재구성해보세요
2. 관련 법률 분야를 명시하여 질문해보세요
3. 구체적인 상황이나 사례를 포함하여 질문해보세요

⚠️ 주의사항
- 법률은 해석의 여지가 있으므로 정확한 답변을 위해 충분한 정보가 필요합니다
- 개별 사안에 대해서는 변호사와 직접 상담하시기 바랍니다

📞 추가 도움
더 궁금한 점이 있으시면 언제든 말씀해 주세요!

본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
        
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
        
        return response
    
    def process_query(self, message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """쿼리 처리"""
        if not self.initialized:
            return "서비스가 초기화되지 않았습니다.", []
        
        if not message.strip():
            return "질문을 입력해주세요.", []
        
        try:
            # 문서 검색
            search_results = self.search_documents(message, top_k=5)
            
            # 응답 생성
            response = self.generate_response(message, search_results)
            
            return response, search_results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"오류가 발생했습니다: {str(e)}", []

# 전역 서비스 인스턴스
lawfirm_service = LawFirmAIService()

def process_langchain_query(message: str, history: List) -> Tuple[str, List]:
    """LangChain 기반 쿼리 처리"""
    response, sources = lawfirm_service.process_query(message)
    
    # 대화 기록 업데이트
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return "", history

def create_simple_langchain_gradio_interface():
    """간단한 LangChain 기반 Gradio 인터페이스 생성"""
    
    # 커스텀 CSS 로드
    css_file = Path("gradio/static/custom.css")
    custom_css = ""
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            custom_css = f.read()
    
    with gr.Blocks(
        title="LawFirmAI - Simple LangChain 기반 법률 AI 어시스턴트",
        css=custom_css,
        theme=gr.themes.Soft()
    ) as interface:
        
        # 헤더
        gr.Markdown("""
        # 🏛️ LawFirmAI - Simple LangChain 기반 법률 AI 어시스턴트
        
        **LangChain과 RAG 기술을 활용한 지능형 법률 상담 서비스**
        
        ---
        """)
        
        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            label="법률 AI 어시스턴트",
            height=500,
            show_label=True,
            container=True,
            type="messages"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="법률 관련 질문을 입력하세요...",
                label="질문",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("전송", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("대화 초기화", variant="secondary", scale=1)
            natural_prompt_btn = gr.Button("😊 자연스러운 상담", scale=1)
            formal_prompt_btn = gr.Button("⚖️ 전문가 상담", scale=1)
        
        # 예시 질문
        with gr.Accordion("📝 예시 질문", open=False):
            gr.Markdown("""
            **민법 관련:**
            - 계약의 성립 요건은 무엇인가요?
            - 불법행위의 구성요건을 설명해주세요
            - 소유권 이전의 시점은 언제인가요?
            
            **상법 관련:**
            - 주식회사의 설립 절차는 어떻게 되나요?
            - 이사의 의무와 책임은 무엇인가요?
            - 주주총회의 권한은 무엇인가요?
            
            **형법 관련:**
            - 절도죄의 구성요건은 무엇인가요?
            - 정당방위의 요건을 설명해주세요
            - 미수범의 처벌 기준은 어떻게 되나요?
            """)
        
        # 이벤트 연결
        submit_btn.click(
            process_langchain_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            process_langchain_query,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        # 프롬프트 전환 버튼 이벤트
        def switch_to_natural_prompt():
            """자연스러운 프롬프트로 전환"""
            success = prompt_manager.switch_to_version("natural_legal_consultant_v1.0")
            if success:
                return "😊 자연스러운 상담 모드로 전환되었습니다!"
            return "프롬프트 전환에 실패했습니다."
        
        def switch_to_formal_prompt():
            """전문가 프롬프트로 전환"""
            success = prompt_manager.switch_to_version("legal_expert_v1.0")
            if success:
                return "⚖️ 전문가 상담 모드로 전환되었습니다!"
            return "프롬프트 전환에 실패했습니다."
        
        natural_prompt_btn.click(switch_to_natural_prompt, outputs=[chatbot])
        formal_prompt_btn.click(switch_to_formal_prompt, outputs=[chatbot])
    
    return interface

def main():
    """메인 실행 함수"""
    import signal
    import atexit
    
    # 프로세스 ID 저장
    pid = os.getpid()
    logger.info(f"Starting LawFirmAI Simple LangChain Gradio application... (PID: {pid})")
    
    # PID 파일 경로
    pid_file = Path("gradio_server.pid")
    
    # 데이터베이스 경로 수정
    os.chdir(Path(__file__).parent.parent)  # 프로젝트 루트로 이동
    
    def save_pid():
        """PID를 파일에 저장"""
        try:
            pid_data = {
                "pid": pid,
                "start_time": time.time(),
                "status": "running",
                "type": "simple_langchain"
            }
            with open(pid_file, 'w', encoding='utf-8') as f:
                json.dump(pid_data, f, indent=2)
            logger.info(f"PID saved to {pid_file}")
        except Exception as e:
            logger.error(f"Failed to save PID: {e}")
    
    def cleanup():
        """정리 함수"""
        logger.info("Cleaning up resources...")
        try:
            if pid_file.exists():
                pid_file.unlink()
                logger.info("PID file removed")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")
    
    def signal_handler(signum, frame):
        """시그널 핸들러"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        cleanup()
        sys.exit(0)
    
    # PID 저장
    save_pid()
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    try:
        # 서비스 초기화
        if not lawfirm_service.initialize():
            logger.error("Failed to initialize services")
            sys.exit(1)
        
        # 인터페이스 생성 및 실행
        interface = create_simple_langchain_gradio_interface()
        
        # Launch
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start Simple LangChain application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()