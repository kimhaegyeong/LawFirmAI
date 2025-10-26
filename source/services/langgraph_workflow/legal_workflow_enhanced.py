# -*- coding: utf-8 -*-
"""
개선된 LangGraph Legal Workflow
답변 품질 향상을 위한 향상된 워크플로우 구현
"""

import logging
import time
from typing import Any, Dict, List

from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from ...utils.langgraph_config import LangGraphConfig
from .keyword_mapper import LegalKeywordMapper
from .legal_data_connector import LegalDataConnector
from .performance_optimizer import PerformanceOptimizer
from .prompt_templates import LegalPromptTemplates
from .state_definitions import LegalWorkflowState

logger = logging.getLogger(__name__)

# Mock QuestionType for enhanced workflow
class QuestionType:
    GENERAL_QUESTION = "general_question"
    LAW_INQUIRY = "law_inquiry"
    PRECEDENT_SEARCH = "precedent_search"
    LEGAL_ADVICE = "legal_advice"
    PROCEDURE_GUIDE = "procedure_guide"
    TERM_EXPLANATION = "term_explanation"
    CONTRACT_REVIEW = "contract_review"
    FAMILY_LAW = "family_law"
    CRIMINAL_LAW = "criminal_law"
    CIVIL_LAW = "civil_law"
    LABOR_LAW = "labor_law"
    PROPERTY_LAW = "property_law"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    TAX_LAW = "tax_law"
    CIVIL_PROCEDURE = "civil_procedure"


class EnhancedLegalQuestionWorkflow:
    """개선된 법률 질문 처리 워크플로우"""

    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 초기화
        self.prompt_templates = LegalPromptTemplates()
        self.keyword_mapper = LegalKeywordMapper()
        self.data_connector = LegalDataConnector()
        self.performance_optimizer = PerformanceOptimizer()

        # 벡터 스토어 초기화
        self.vector_store = self._initialize_vector_store()

        # LLM 초기화
        self.llm = self._initialize_llm()

        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.info("EnhancedLegalQuestionWorkflow initialized.")

    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        try:
            from ...data.vector_store import LegalVectorStore

            vector_store = LegalVectorStore(
                model_name="jhgan/ko-sroberta-multitask",
                dimension=768,
                index_type="flat",
                enable_quantization=True,
                enable_lazy_loading=True,
                memory_threshold_mb=1500  # 메모리 임계값을 1500MB로 낮춤
            )

            # 벡터 인덱스 로드 시도
            index_paths = [
                "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index",
                "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index",
                "data/embeddings/legal_vector_index"
            ]

            index_loaded = False
            for index_path in index_paths:
                try:
                    if vector_store.load_index(index_path):
                        print(f"Vector store loaded from: {index_path}")
                        index_loaded = True
                        break
                except Exception as e:
                    print(f"Failed to load vector store from {index_path}: {e}")
                    continue

            if not index_loaded:
                print("No vector store loaded, will use database search only")

            return vector_store

        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            return None

    def _initialize_llm(self):
        """LLM 초기화 (Google Gemini 우선, Ollama 백업)"""
        if self.config.llm_provider == "google":
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=self.config.google_model,
                    temperature=0.3,
                    max_output_tokens=500,  # 답변 길이 증가
                    timeout=30,  # 타임아웃 증가
                    api_key=self.config.google_api_key
                )
                # 간단한 테스트 호출로 모델 로드 확인 (제거)
                # test_response = gemini_llm.invoke("안녕하세요")
                logger.info(f"Initialized Google Gemini LLM: {self.config.google_model}")
                # logger.info(f"Test response: {test_response.content[:50]}...")
                return gemini_llm
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini LLM: {e}. Falling back to Ollama.")

        if self.config.llm_provider == "ollama":
            try:
                ollama_llm = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    temperature=0.3,
                    num_predict=500,  # 답변 길이 증가
                    timeout=30  # 타임아웃 증가
                )
                # 간단한 테스트 호출로 모델 로드 확인 (제거)
                # test_response = ollama_llm.invoke("안녕하세요")
                logger.info(f"Initialized Ollama LLM: {self.config.ollama_model}")
                # logger.info(f"Test response: {test_response[:50]}...")
                return ollama_llm
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama LLM: {e}. Using Mock LLM.")

        # 🆕 개선된 Mock LLM - 검색 결과 활용
        class ImprovedMockLLM:
            def invoke(self, prompt):
                """검색 결과를 활용한 법률 답변 생성"""
                # 프롬프트에서 컨텍스트(검색 결과) 추출
                context = ""
                question = ""

                if "context:" in prompt.lower():
                    parts = prompt.split("context:")
                    if len(parts) > 1:
                        context = parts[1].strip()

                if "question:" in prompt.lower():
                    question_part = prompt.split("question:")[-1]
                    if "context:" in question_part:
                        question = question_part.split("context:")[0].strip()
                    else:
                        question = question_part.strip()

                # 컨텍스트가 있으면 검색 결과 기반 답변
                if context and context != "" and len(context) > 10:
                    # 검색 결과를 요약하여 답변 생성
                    return self._generate_response_from_context(question, context)

                # 컨텍스트가 없으면 기본 답변
                return "죄송합니다. 해당 질문에 대한 관련 법률 정보를 찾을 수 없었습니다. 다른 법률 조문이나 구체적인 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다."

            def _generate_response_from_context(self, question, context):
                """컨텍스트를 활용한 답변 생성"""
                # 검색 결과에서 핵심 내용 추출
                lines = context.split('\n')

                # 첫 번째 주요 내용 찾기
                main_content = ""
                for line in lines:
                    if line.strip() and len(line.strip()) > 20:
                        main_content = line.strip()
                        break

                # 질문 유형에 따른 답변 생성
                if "상속" in question or "유언" in question:
                    return f"""참고하신 내용에 따르면:

{main_content if main_content else '관련 법률 조문을 찾았습니다.'}

이에 대해 간략히 설명드리면, 상속과 관련된 법률은 민법에 규정되어 있으며, 각 가족 구성원별로 상속분이 다릅니다. 구체적인 조문을 확인하시면 더 정확한 정보를 얻으실 수 있습니다."""

                elif "야간" in question or "근무" in question or "수당" in question:
                    return f"""근로기준법에 따르면:

{main_content if main_content else '야간근무와 관련된 법률 조문을 확인했습니다.'}

일반적으로 야간근무는 특정 시간대(보통 오후 10시 이후)에 수행되는 근무를 의미하며, 야간수당이 별도로 지급되어야 합니다. 연장근무와는 별개의 개념입니다."""

                else:
                    # 일반적인 법률 답변
                    return f"""다음과 같은 법률 정보를 확인했습니다:

{main_content[:300] if main_content else '관련 법률 조문을 확인했습니다'}

이 내용이 도움이 되셨는지 확인해주시고, 추가로 궁금한 사항이 있으시면 알려주세요.

※ 이 답변은 법률 정보 제공을 목적으로 하며, 구체적인 사안에 대한 법률 자문은 변호사와 상담하시기 바랍니다."""

            async def ainvoke(self, prompt):
                return self.invoke(prompt)

        logger.warning("No valid LLM provider configured or failed to initialize. Using Improved Mock LLM.")
        return ImprovedMockLLM()

    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)

        # 노드 추가
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("generate_answer_enhanced", self.generate_answer_enhanced)
        workflow.add_node("format_response", self.format_response)

        # 엣지 설정
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer_enhanced")
        workflow.add_edge("generate_answer_enhanced", "format_response")
        workflow.add_edge("format_response", END)

        return workflow

    def classify_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """질문 분류 (개선된 버전)"""
        try:
            # 로깅 대신 print 사용 (멀티스레드 환경에서 안전)
            print(f"🔍 classify_query 시작: query='{state.get('query', 'NOT_FOUND')}'")
            start_time = time.time()

            # 상태 디버깅
            print(f"classify_query - Received state keys: {list(state.keys())}")
            print(f"classify_query - state['query']: '{state.get('query', 'NOT_FOUND')}'")
            print(f"classify_query - state['user_query']: '{state.get('user_query', 'NOT_FOUND')}'")

            # 상태 초기화 (필요한 키들이 없으면 추가)
            if "query" not in state:
                state["query"] = ""
            if "errors" not in state:
                state["errors"] = []
            if "query_type" not in state:
                state["query_type"] = QuestionType.GENERAL_QUESTION
            if "confidence" not in state:
                state["confidence"] = 0.0
            if "sources" not in state:
                state["sources"] = []
            if "response" not in state:
                state["response"] = ""
            if "processing_time" not in state:
                state["processing_time"] = 0.0
            if "processing_steps" not in state:
                state["processing_steps"] = []

            # 원본 쿼리 보존 (user_query가 있으면 사용)
            original_query = state.get("user_query") or state.get("query", "")
            query = original_query.lower()

            print(f"classify_query - Using query: '{original_query}'")

            # 원본 쿼리를 상태에 저장 (다른 노드에서 사용할 수 있도록)
            state["query"] = original_query
            state["original_query"] = original_query

            # 개선된 키워드 기반 분류
            if any(k in query for k in ["계약", "계약서", "매매", "임대", "도급"]):
                state["query_type"] = QuestionType.CONTRACT_REVIEW
            elif any(k in query for k in ["이혼", "가족", "상속", "양육", "입양"]):
                state["query_type"] = QuestionType.FAMILY_LAW
            elif any(k in query for k in ["절도", "범죄", "형사", "사기", "폭행", "강도", "살인"]):
                state["query_type"] = QuestionType.CRIMINAL_LAW
            elif any(k in query for k in ["손해배상", "민사", "불법행위", "채권", "소유권"]):
                state["query_type"] = QuestionType.CIVIL_LAW
            elif any(k in query for k in ["해고", "노동", "임금", "근로시간", "휴가", "산업재해"]):
                state["query_type"] = QuestionType.LABOR_LAW
            elif any(k in query for k in ["부동산", "매매", "등기", "공시", "토지"]):
                state["query_type"] = QuestionType.PROPERTY_LAW
            elif any(k in query for k in ["특허", "지적재산권", "저작권", "상표", "디자인"]):
                state["query_type"] = QuestionType.INTELLECTUAL_PROPERTY
            elif any(k in query for k in ["세금", "소득세", "부가가치세", "법인세", "상속세", "가산세"]):
                state["query_type"] = QuestionType.TAX_LAW
            elif any(k in query for k in ["소송", "관할", "증거", "판결", "집행", "민사소송"]):
                state["query_type"] = QuestionType.CIVIL_PROCEDURE
            else:
                state["query_type"] = QuestionType.GENERAL_QUESTION

            state["confidence"] = 0.8  # 분류 신뢰도 향상
            state["processing_steps"].append(f"질문 분류 완료 (개선): {state['query_type']}")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"Query classified as {state['query_type']} with confidence {state['confidence']}")

        except Exception as e:
            error_msg = f"질문 분류 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            print(f"❌ {error_msg}")

            # 기본값 설정
            state["query_type"] = QuestionType.GENERAL_QUESTION
            state["confidence"] = 0.3  # 기본 신뢰도를 낮춤

        return state

    def retrieve_documents(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """문서 검색 (캐싱 적용)"""
        try:
            start_time = time.time()

            # 상태 초기화 (안전한 방식)
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "retrieved_docs" not in state:
                state["retrieved_docs"] = []
            if "errors" not in state:
                state["errors"] = []
            if "query" not in state:
                state["query"] = ""
            if "query_type" not in state:
                state["query_type"] = QuestionType.GENERAL_QUESTION
            if "confidence" not in state:
                state["confidence"] = 0.3  # 기본 신뢰도를 낮춤

            # ✅ 개선된 쿼리 추출 로직 (순서 변경)
            query = state.get("user_query") or state.get("query") or state.get("original_query") or ""

            if not query:
                print("❌ 쿼리가 비어있습니다. 검색을 건너뜁니다.")
                state["retrieved_docs"] = []
                state["processing_steps"].append("쿼리가 비어있어 검색을 건너뛰었습니다")
                return state

            print(f"🔍 retrieve_documents 시작: query='{query}'")

            query_type = state["query_type"]

            # 상태 디버깅
            print(f"retrieve_documents - Received state keys: {list(state.keys())}")
            print(f"retrieve_documents - state['query']: '{state.get('query', 'NOT_FOUND')}'")
            print(f"retrieve_documents - state['user_query']: '{state.get('user_query', 'NOT_FOUND')}'")
            print(f"retrieve_documents - state['original_query']: '{state.get('original_query', 'NOT_FOUND')}'")

            # 쿼리 디버깅
            print(f"Document retrieval - Query: '{query}', Type: {query_type}")

            # 캐시에서 문서 확인 (더 적극적인 캐싱)
            cache_key = f"{query}_{query_type}"
            cached_documents = self.performance_optimizer.cache.get_cached_documents(query, query_type)

            if cached_documents:
                state["retrieved_docs"] = cached_documents
                state["processing_steps"].append(f"{len(cached_documents)}개 캐시된 문서 사용")
                print(f"Using cached documents for query: {query[:50]}...")
            else:
                # 벡터 검색 우선 시도
                documents = []

                # 벡터 스토어에서 검색 시도
                try:
                    if hasattr(self, 'vector_store') and self.vector_store and hasattr(self.vector_store, 'search'):
                        vector_results = self.vector_store.search(query, top_k=5)
                        if vector_results:
                            # 벡터 검색 결과를 문서 형식으로 변환
                            for i, result in enumerate(vector_results):
                                doc = {
                                    "id": f"vector_{i}",
                                    "content": result.get("content", str(result)),
                                    "source": "Vector Store",
                                    "relevance_score": result.get("similarity", result.get("score", 0.8)),
                                    "category": query_type
                                }
                                documents.append(doc)
                            print(f"Vector search found {len(documents)} documents")
                except Exception as e:
                    print(f"⚠️ Vector search failed: {e}")

                # 데이터베이스 검색 항상 수행 (실제 법률 문서 사용)
                try:
                    print(f"🔍 데이터베이스 검색 시작: query='{query}', query_type='{query_type}'")
                    db_documents = self.data_connector.search_documents(query, query_type, limit=5)
                    print(f"✅ 데이터베이스 검색 완료: {len(db_documents)}개 문서 발견")

                    # 중복 제거
                    existing_contents = {doc["content"][:100] for doc in documents}
                    for doc in db_documents:
                        if doc.get("content", "")[:100] not in existing_contents:
                            documents.append(doc)
                    print(f"📊 데이터베이스 검색으로 {len(db_documents)}개 문서 추가")
                except Exception as e:
                    print(f"❌ 데이터베이스 검색 실패: {e}")
                    import traceback
                    print(f"상세 오류: {traceback.format_exc()}")

                # 여전히 결과가 부족한 경우 카테고리별 문서 추가
                if len(documents) < 3:
                    try:
                        category_docs = self.data_connector.get_document_by_category(query_type, limit=3)
                        existing_contents = {doc["content"][:100] for doc in documents}
                        for doc in category_docs:
                            if doc.get("content", "")[:100] not in existing_contents:
                                documents.append(doc)
                        print(f"Category search added {len(category_docs)} documents")
                    except Exception as e:
                        print(f"Category search failed: {e}")

                state["retrieved_docs"] = documents
                state["processing_steps"].append(f"{len(documents)}개 문서 검색 완료 (벡터+DB)")

                # 문서 캐싱
                self.performance_optimizer.cache.cache_documents(query, query_type, documents)

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"Retrieved {len(state['retrieved_docs'])} documents for query type {query_type}")

        except Exception as e:
            error_msg = f"문서 검색 중 오류 발생: {str(e)}"
            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            print(f"❌ {error_msg}")

            # 폴백: 기본 문서 설정
            state["retrieved_docs"] = [
                {"content": f"'{state['query']}'에 대한 기본 법률 정보입니다.", "source": "Default DB"}
            ]

        return state

    def generate_answer_enhanced(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """개선된 답변 생성"""
        # user_query를 먼저 확인하고, 없으면 query 사용
        query_value = state.get('user_query') or state.get('query', 'NOT_FOUND')
        print(f"🔍 generate_answer_enhanced 시작: query='{query_value}'")

        # 🆕 상세 상태 디버깅
        print(f"📋 state 키 목록: {list(state.keys())}")
        print(f"📊 retrieved_docs 유무: {'retrieved_docs' in state}")
        print(f"📊 retrieved_docs 타입: {type(state.get('retrieved_docs', 'NOT_FOUND'))}")
        print(f"📊 검색된 문서 수: {len(state.get('retrieved_docs', []))}")

        # 🆕 retrieved_docs 내용 확인
        if 'retrieved_docs' in state:
            docs = state['retrieved_docs']
            if isinstance(docs, list):
                print(f"📊 retrieved_docs 리스트 길이: {len(docs)}")
                if docs:
                    print(f"📊 첫 번째 문서 샘플: {type(docs[0])} - {list(docs[0].keys()) if isinstance(docs[0], dict) else str(docs[0])[:100]}")
            else:
                print(f"📊 retrieved_docs 타입: {type(docs)}")

        try:
            start_time = time.time()

            # 🆕 전체 state 보존 방식으로 변경
            # TypedDict이므로 전체 딕셔너리로 처리
            updated_state = dict(state)  # 전체 state 복사

            # 필요한 필드만 초기화 (기존 값 유지)
            if "query_type" not in updated_state:
                updated_state["query_type"] = QuestionType.GENERAL_QUESTION
            if "query" not in updated_state:
                updated_state["query"] = ""
            if "response" not in updated_state:
                updated_state["response"] = ""
            if "confidence" not in updated_state:
                updated_state["confidence"] = 0.0
            if "sources" not in updated_state:
                updated_state["sources"] = []
            if "processing_steps" not in updated_state:
                updated_state["processing_steps"] = []
            if "errors" not in updated_state:
                updated_state["errors"] = []

            # 🆕 retrieved_docs는 이미 state에 있으므로 그대로 사용
            retrieved_docs = updated_state.get("retrieved_docs", [])

            print(f"🔍 generate_answer_enhanced에서 retrieved_docs 확인: {len(retrieved_docs)}개")
            if not retrieved_docs:
                query = updated_state.get("user_query") or updated_state.get("query") or "질문"
                print("⚠️ 검색된 문서가 없습니다. 기본 답변을 제공합니다.")
                updated_state["generated_response"] = (
                    f"죄송합니다. '{query}'에 대한 관련 문서를 찾을 수 없었습니다. "
                    f"일반적인 법률 정보를 바탕으로 답변드립니다.\n\n"
                    f"이 질문은 {updated_state['query_type']} 영역에 해당합니다. "
                    f"구체적인 사안에 대한 정확한 법률 조언은 전문가와 상담하시는 것을 권장합니다."
                )
                updated_state["confidence"] = 0.3
                updated_state["processing_steps"] = updated_state.get("processing_steps", [])
                updated_state["processing_steps"].append("검색 결과가 없어 기본 답변 제공")
                return updated_state

            # 🔍 디버깅: 검색된 문서 출처 확인
            print(f"📚 검색된 문서 정보:")
            for i, doc in enumerate(retrieved_docs[:5], 1):  # 상위 5개만 표시
                source = doc.get("source", "Unknown")
                title = doc.get("title", doc.get("content", "")[:50])
                category = doc.get("category", "Unknown")
                relevance_score = doc.get("relevance_score", 0.0)
                print(f"  [{i}] Source: {source}, Category: {category}, Relevance: {relevance_score:.2f}")
                print(f"      Title: {title[:80]}...")

            # 컨텍스트 구성
            context = "\n".join([doc.get("content", str(doc)) for doc in retrieved_docs if doc])

            # 원본 쿼리 사용
            original_query = updated_state.get("user_query") or updated_state.get("original_query") or updated_state.get("query")

            # 🔍 디버깅: 입력값 확인
            print(f"🔍 프롬프트 구성 디버깅:")
            print(f"  - original_query: '{original_query}'")
            print(f"  - context 길이: {len(context)}")
            print(f"  - query_type: {updated_state['query_type']}")

            # 질문 유형별 키워드 추출
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                original_query, updated_state["query_type"]
            )
            print(f"  - required_keywords: {required_keywords[:5]}")

            # 질문 유형별 프롬프트 템플릿 선택
            template = self.prompt_templates.get_template_for_query_type(updated_state["query_type"])

            # 🔍 디버깅: 템플릿 확인
            print(f"  - template 타입: {type(template)}")
            print(f"  - template 길이: {len(template) if isinstance(template, str) else 'N/A'}")
            print(f"  - template 시작: {str(template)[:100]}...")

            # 프롬프트 구성
            try:
                # 템플릿이 문자열인지 확인
                if not isinstance(template, str):
                    print(f"⚠️ template이 문자열이 아닙니다. 변환합니다.")
                    template = str(template)

                # 플레이스홀더 확인
                if "{question}" not in template or "{context}" not in template:
                    print(f"⚠️ template에 필수 플레이스홀더가 없습니다. 기본 템플릿 사용")
                    # 기본 템플릿으로 대체
                    template = f"""당신은 전문적인 법률 상담 변호사입니다. 사용자의 질문에 대해 자연스럽고 직접적으로 답변해주세요.

## 사용자 질문
{{question}}

## 관련 법률 문서
{{context}}

## 답변 원칙
1. 일상적인 법률 상담처럼 자연스럽고 친근하게 대화하세요
2. "~입니다", "귀하" 같은 과도하게 격식적인 표현 대신, "~예요", "질문하신" 등 자연스러운 존댓말을 사용하세요
3. 질문을 다시 반복하지 마세요
4. 질문의 범위에 맞는 적절한 양의 정보만 제공하세요
5. 불필요한 형식(제목, 번호 매기기)은 최소화하세요
6. 핵심 내용을 요약하고 주요 주의사항을 제시하세요
7. 법적 근거를 명시하고 실무 권장사항을 제공하세요

답변을 한국어로 작성하고, 전문 법률 용어는 쉽게 풀어서 설명해주세요."""

                prompt = template.format(
                    question=original_query,
                    context=context,
                    required_keywords=", ".join(required_keywords[:10]) if "required_keywords" in template else ""
                )
                print(f"  ✅ 프롬프트 생성 성공: {len(prompt)} 문자")
                print(f"  - 프롬프트 샘플: {prompt[:300]}...")

            except KeyError as e:
                print(f"❌ 프롬프트 생성 실패 - 플레이스홀더 오류: {e}")
                # 플레이스홀더를 수동으로 교체
                prompt = template.replace("{question}", original_query).replace("{context}", context)
                if "{required_keywords}" in prompt:
                    prompt = prompt.replace("{required_keywords}", ", ".join(required_keywords[:10]))
                print(f"  🔧 수동 교체 성공")
            except Exception as e:
                print(f"❌ 프롬프트 생성 실패: {e}")
                # 최후의 수단: 간단한 프롬프트
                prompt = f"""질문: {original_query}

관련 문서:
{context}

위 질문에 대해 법률 전문가로서 답변해주세요."""

            # LLM 호출
            print(f"LLM 호출 시작 - 프롬프트 길이: {len(prompt)}")
            response = self._call_llm_with_retry(prompt)
            print(f"LLM 응답 받음 - 응답 길이: {len(response) if response else 0}")
            print(f"LLM 응답 내용: {response[:100] if response else 'None'}...")

            # 답변 후처리 (구조화 강화)
            enhanced_response = self._enhance_response_structure(response, required_keywords)
            print(f"구조화된 응답 길이: {len(enhanced_response) if enhanced_response else 0}")

            # 모든 응답 필드에 동일한 값 설정
            updated_state["answer"] = enhanced_response
            updated_state["generated_response"] = enhanced_response
            updated_state["response"] = enhanced_response
            updated_state["processing_steps"] = updated_state.get("processing_steps", [])
            updated_state["processing_steps"].append("개선된 답변 생성 완료")

            # 신뢰도 계산 (개선된 로직)
            confidence = self._calculate_dynamic_confidence(
                enhanced_response,
                retrieved_docs,
                original_query,
                updated_state["query_type"]
            )
            updated_state["confidence"] = confidence
            updated_state["processing_steps"].append(f"신뢰도 계산 완료: {confidence:.2f}")

            # 디버깅 로그 추가
            print(f"응답 필드 설정 완료:")
            print(f"  - answer 길이: {len(updated_state['answer'])}")
            print(f"  - generated_response 길이: {len(updated_state['generated_response'])}")
            print(f"  - response 길이: {len(updated_state['response'])}")
            print(f"  - confidence: {confidence:.2f}")

            processing_time = time.time() - start_time
            updated_state["processing_time"] = updated_state.get("processing_time", 0.0) + processing_time

            print(f"Enhanced answer generated in {processing_time:.2f}s")

        except Exception as e:
            error_msg = f"개선된 답변 생성 중 오류 발생: {str(e)}"
            # 상태 초기화 확인
            if "errors" not in updated_state:
                updated_state["errors"] = []
            if "processing_steps" not in updated_state:
                updated_state["processing_steps"] = []

            updated_state["errors"].append(error_msg)
            updated_state["processing_steps"].append(error_msg)
            print(f"❌ {error_msg}")

            # 기본 답변 설정
            updated_state["answer"] = self._generate_fallback_answer(updated_state)

        return updated_state

    def _calculate_dynamic_confidence(self, response: str, retrieved_docs: List[Dict],
                                    query: str, query_type) -> float:
        """동적 신뢰도 계산"""
        try:
            base_confidence = 0.5  # ✅ 기본 신뢰도를 0.5로 증가 (기존 0.2)

            # 1. 응답 길이 기반 점수
            response_length = len(response)
            if response_length > 500:
                length_score = 0.3
            elif response_length > 200:
                length_score = 0.2
            elif response_length > 100:
                length_score = 0.1
            else:
                length_score = 0.0

            # 2. 검색된 문서 수 기반 점수
            doc_count = len(retrieved_docs)
            if doc_count >= 3:
                doc_score = 0.2
            elif doc_count >= 2:
                doc_score = 0.15
            elif doc_count >= 1:
                doc_score = 0.1
            else:
                doc_score = 0.0

            # 3. 질문 유형별 가중치
            type_weights = {
                "LEGAL_ADVICE": 0.8,
                "PROCEDURE_GUIDE": 0.7,
                "TERM_EXPLANATION": 0.6,
                "CONTRACT_REVIEW": 0.9,
                "GENERAL_QUESTION": 0.5
            }
            type_score = type_weights.get(str(query_type), 0.5) * 0.2

            # 4. 응답 품질 기반 점수 (법률 키워드 포함 여부)
            legal_keywords = ["법률", "조문", "판례", "법령", "규정", "소송", "계약", "권리", "의무",
                           "민법", "형법", "상법", "헌법", "행정", "형사", "민사", "이혼", "상속"]
            keyword_count = sum(1 for keyword in legal_keywords if keyword in response)
            quality_score = min(keyword_count * 0.05, 0.2)

            # 최종 신뢰도 계산
            final_confidence = base_confidence + length_score + doc_score + type_score + quality_score

            # 0.0 ~ 1.0 범위로 제한
            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            print(f"신뢰도 계산 실패: {e}")
            return 0.3  # 기본값

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """LLM 호출 (재시도 로직 포함)"""
        print(f"LLM 호출 시작 - 프롬프트: {prompt[:100]}...")

        for attempt in range(max_retries):
            try:
                print(f"LLM 호출 시도 {attempt + 1}/{max_retries}")

                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt)
                    print(f"LLM 원본 응답 타입: {type(response)}")

                    if hasattr(response, 'content'):
                        result = response.content
                        print(f"LLM 응답 내용 추출 성공: {result[:100]}...")
                        result = self._clean_llm_response(result)  # ✅ 응답 정리
                        return result
                    else:
                        result = str(response)
                        print(f"LLM 응답 문자열 변환: {result[:100]}...")
                        result = self._clean_llm_response(result)  # ✅ 응답 정리
                        return result
                else:
                    result = self.llm.invoke(prompt)
                    print(f"LLM 직접 호출 결과: {result[:100]}...")
                    result = self._clean_llm_response(result)  # ✅ 응답 정리
                    return result

            except Exception as e:
                print(f"LLM 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"LLM 호출 최종 실패: {e}")
                    raise e
                time.sleep(1)  # 재시도 전 대기

        print("LLM 호출에 실패했습니다.")
        return "LLM 호출에 실패했습니다."

    def _clean_llm_response(self, result: str) -> str:
        """✅ LLM 응답에서 프롬프트 지침 제거"""
        if not result:
            return result

        # 프롬프트 지침이 포함된 경우 제거
        if "## 사용자 질문" in result and "## 답변 작성 지침" in result:
            # 지침 부분 제거
            if "## 답변 작성" in result:
                parts = result.split("## 답변 작성")
                if len(parts) > 1:
                    result = parts[-1].strip()

            # 추가로 답변 부분만 남기기
            if "## 답변" in result:
                parts = result.split("## 답변")
                if len(parts) > 1:
                    result = parts[-1].strip()

        # 불필요한 "답변 작성" 관련 텍스트 제거
        if "답변 작성 지침" in result:
            result = result.replace("답변 작성 지침", "").strip()

        return result

    def _enhance_response_structure(self, response: str, required_keywords: List[str]) -> str:
        """답변 구조화 강화"""
        # 키워드 포함 확인 및 강화
        missing_keywords = self.keyword_mapper.get_missing_keywords(response, required_keywords[:5])

        if missing_keywords:
            # 누락된 키워드 추가
            response += f"\n\n## 추가 고려사항\n"
            for keyword in missing_keywords[:3]:  # 최대 3개만 추가
                response += f"- {keyword} 관련 사항도 고려해야 합니다.\n"

        # 구조화 강화
        if "##" not in response:
            # 제목이 없으면 추가
            response = f"## 답변\n{response}"

        if not any(marker in response for marker in ["1.", "2.", "3.", "•", "-"]):
            # 목록이 없으면 추가
            response += "\n\n## 주요 포인트\n- 위 내용을 참고하여 구체적인 조치를 취하시기 바랍니다."

        return response

    def _generate_fallback_answer(self, state: LegalWorkflowState) -> str:
        """폴백 답변 생성"""
        query = state["query"]
        query_type = state["query_type"]
        context = "\n".join([doc["content"] for doc in state["retrieved_docs"]])

        return f"""## 답변

질문: {query}

이 질문은 {query_type} 영역에 해당합니다.

## 관련 법률 정보
{context}

## 주요 포인트
1. 위 정보를 바탕으로 구체적인 조치를 취하시기 바랍니다.
2. 정확한 법률적 조언을 위해서는 전문가와 상담하시는 것을 권장합니다.
3. 관련 법조문과 판례를 추가로 확인하시기 바랍니다.

## 주의사항
- 이 답변은 일반적인 정보 제공 목적이며, 구체적인 법률적 조언이 아닙니다.
- 실제 사안에 대해서는 전문 변호사와 상담하시기 바랍니다."""

    def format_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 포맷팅 (개선된 버전)"""
        try:
            start_time = time.time()

            # 상태 디버깅
            print(f"format_response - Received state keys: {list(state.keys())}")
            print(f"format_response - state['answer']: '{state.get('answer', 'NOT_FOUND')}'")
            print(f"format_response - state['response']: '{state.get('response', 'NOT_FOUND')}'")
            print(f"format_response - state['generated_response']: '{state.get('generated_response', 'NOT_FOUND')}'")

            # 상태 초기화
            if "answer" not in state:
                state["answer"] = state.get("response", "답변을 생성하지 못했습니다.")
            if "confidence" not in state:
                state["confidence"] = 0.0
            if "retrieved_docs" not in state:
                state["retrieved_docs"] = []
            if "query" not in state:
                state["query"] = ""
            if "query_type" not in state:
                state["query_type"] = QuestionType.GENERAL_QUESTION
            if "response" not in state:
                state["response"] = ""
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "errors" not in state:
                state["errors"] = []

            # 최종 답변 및 메타데이터 정리
            # generated_response가 있으면 우선 사용
            if state.get("generated_response"):
                final_answer = state["generated_response"]
                print(f"format_response - Using generated_response: '{final_answer[:100]}...'")
            else:
                final_answer = state.get("answer", "답변을 생성하지 못했습니다.")
                print(f"format_response - Using answer: '{final_answer[:100]}...'")

            final_confidence = state.get("confidence", 0.0)
            final_sources = [doc.get("source", "Unknown") for doc in state.get("retrieved_docs", [])]

            # 🔍 참조 출처 추출 및 답변에 추가
            referenced_sources = []
            for doc in state.get("retrieved_docs", [])[:3]:  # 상위 3개만
                source = doc.get("source", "Unknown")
                category = doc.get("category", "")
                if source and source not in [s["name"] for s in referenced_sources]:
                    referenced_sources.append({
                        "name": source,
                        "category": category,
                        "relevance": doc.get("relevance_score", 0.0)
                    })

            # 📚 답변에 참조 출처 추가
            if referenced_sources:
                final_answer += "\n\n## 참조 출처\n"
                for source_info in referenced_sources:
                    category_str = f" ({source_info['category']})" if source_info.get("category") else ""
                    final_answer += f"- {source_info['name']}{category_str}\n"
                print(f"📚 참조 출처 추가됨: {len(referenced_sources)}개")

            # 키워드 포함도 계산 (원본 쿼리 사용)
            original_query = state.get("original_query", state["query"])
            required_keywords = self.keyword_mapper.get_keywords_for_question(
                original_query, state["query_type"]
            )
            keyword_coverage = self.keyword_mapper.calculate_keyword_coverage(
                final_answer, required_keywords
            )

            # 신뢰도 조정 (키워드 포함도 반영)
            adjusted_confidence = min(0.9, final_confidence + (keyword_coverage * 0.2))

            # 모든 응답 필드에 최종 답변 설정
            state["answer"] = final_answer
            state["generated_response"] = final_answer  # generated_response 필드도 설정
            state["response"] = final_answer  # response 필드도 설정
            state["confidence"] = adjusted_confidence
            state["sources"] = list(set(final_sources))  # 중복 제거
            state["legal_references"] = referenced_sources  # 🆕 참조 출처 저장
            state["processing_steps"].append("응답 포맷팅 완료 (개선)")

            # 디버깅 로그 추가
            print(f"format_response - 최종 응답 필드 설정:")
            print(f"  - answer 길이: {len(state['answer'])}")
            print(f"  - generated_response 길이: {len(state['generated_response'])}")
            print(f"  - response 길이: {len(state['response'])}")

            # 메타데이터 추가
            state["metadata"] = {
                "keyword_coverage": keyword_coverage,
                "required_keywords_count": len(required_keywords),
                "matched_keywords_count": len(required_keywords) - len(
                    self.keyword_mapper.get_missing_keywords(final_answer, required_keywords)
                ),
                "response_length": len(final_answer),
                "query_type": state["query_type"],
                "referenced_sources": [s["name"] for s in referenced_sources],  # 🆕 참조 출처 목록
                "reference_count": len(referenced_sources)  # 🆕 참조 출처 개수
            }

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print("Enhanced response formatting completed")

        except Exception as e:
            error_msg = f"응답 포맷팅 중 오류 발생: {str(e)}"
            # 상태 초기화 확인
            if "errors" not in state:
                state["errors"] = []
            if "processing_steps" not in state:
                state["processing_steps"] = []

            state["errors"].append(error_msg)
            state["processing_steps"].append(error_msg)
            print(f"❌ {error_msg}")

        return state
