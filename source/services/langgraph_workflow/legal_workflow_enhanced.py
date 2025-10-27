# -*- coding: utf-8 -*-
"""
개선된 LangGraph Legal Workflow
답변 품질 향상을 위한 향상된 워크플로우 구현
"""

import logging
import time
from typing import Dict, List

from langchain_community.llms import Ollama
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

        # 실제 서비스 초기화
        self.current_law_search_engine = None
        self.unified_search_engine = None
        self.unified_rag_service = None
        self.conversation_store = None
        self.user_profile_manager = None
        self._initialize_external_services()

        # 워크플로우 그래프 구축
        self.graph = self._build_graph()
        logger.info("EnhancedLegalQuestionWorkflow initialized.")

    def _initialize_external_services(self):
        """외부 서비스 초기화"""
        try:
            # CurrentLawSearchEngine 초기화
            from ..current_law_search_engine import CurrentLawSearchEngine
            self.current_law_search_engine = CurrentLawSearchEngine(
                db_path="data/lawfirm.db",
                vector_store=self.vector_store
            )
            logger.info("✅ CurrentLawSearchEngine 초기화 완료")
        except Exception as e:
            logger.warning(f"CurrentLawSearchEngine 초기화 실패: {e}")
            self.current_law_search_engine = None

        try:
            # UnifiedSearchEngine 초기화
            from ..unified_search_engine import UnifiedSearchEngine
            self.unified_search_engine = UnifiedSearchEngine(
                vector_store=self.vector_store,
                current_law_search_engine=self.current_law_search_engine,
                enable_caching=True
            )
            logger.info("✅ UnifiedSearchEngine 초기화 완료")
        except Exception as e:
            logger.warning(f"UnifiedSearchEngine 초기화 실패: {e}")
            self.unified_search_engine = None

        try:
            # UnifiedRAGService 초기화
            logger.info("UnifiedRAGService 초기화 시도 중...")

            # 모델 매니저 초기화
            # 경로 수정: models는 services의 형제 디렉토리
            import sys
            from pathlib import Path
            models_path = Path(__file__).parent.parent.parent / "models"
            if str(models_path) not in sys.path:
                sys.path.insert(0, str(models_path))

            from model_manager import LegalModelManager
            logger.info("LegalModelManager import 성공")

            model_manager = LegalModelManager()
            logger.info("LegalModelManager 인스턴스 생성 성공")

            # UnifiedRAGService import
            from ..unified_rag_service import UnifiedRAGService
            logger.info("UnifiedRAGService import 성공")

            # UnifiedRAGService 초기화 (search_engine이 None이어도 가능하도록 개선)
            if self.unified_search_engine is None:
                logger.warning("unified_search_engine이 None입니다. UnifiedRAGService는 제한적으로 사용됩니다.")

            self.unified_rag_service = UnifiedRAGService(
                model_manager=model_manager,
                search_engine=self.unified_search_engine,
                enable_caching=True
            )
            logger.info("✅ UnifiedRAGService 초기화 완료")
        except ImportError as e:
            logger.error(f"UnifiedRAGService import 실패 (ImportError): {e}")
            logger.debug(f"ImportError 상세: {type(e).__name__}", exc_info=True)
            self.unified_rag_service = None
        except Exception as e:
            logger.error(f"UnifiedRAGService 초기화 실패: {type(e).__name__}: {e}")
            logger.debug(f"Exception 상세: {e.__class__.__name__}", exc_info=True)
            self.unified_rag_service = None

        try:
            # ConversationStore 초기화
            from ...data.conversation_store import ConversationStore
            self.conversation_store = ConversationStore(db_path="data/conversations.db")
            logger.info("✅ ConversationStore 초기화 완료")

            # UserProfileManager 초기화
            from ..user_profile_manager import UserProfileManager
            self.user_profile_manager = UserProfileManager(
                conversation_store=self.conversation_store
            )
            logger.info("✅ UserProfileManager 초기화 완료")
        except Exception as e:
            logger.warning(f"ConversationStore/UserProfileManager 초기화 실패: {e}")
            self.conversation_store = None
            self.user_profile_manager = None

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
                # Google API 키를 환경변수로 설정 (api_key 파라미터는 ADC만 지원)
                import os
                if self.config.google_api_key:
                    os.environ['GOOGLE_API_KEY'] = self.config.google_api_key
                    logger.info(f"GOOGLE_API_KEY set in environment variables")
                else:
                    logger.warning("GOOGLE_API_KEY is not set in config. Falling back to Ollama.")

                gemini_llm = ChatGoogleGenerativeAI(
                    model=self.config.google_model,
                    temperature=0.3,
                    max_output_tokens=500,  # 답변 길이 증가
                    timeout=30,  # 타임아웃 증가
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

                # 프롬프트 파싱 - 다양한 형식 지원
                if "## 관련 법률 문서" in prompt:
                    parts = prompt.split("## 관련 법률 문서")
                    if len(parts) > 1:
                        context = parts[1].strip()
                elif "관련 문서:" in prompt:
                    parts = prompt.split("관련 문서:")
                    if len(parts) > 1:
                        context = parts[1].strip()
                elif "관련 법률 문서" in prompt:
                    parts = prompt.split("관련 법률 문서")
                    if len(parts) > 1:
                        context = parts[1].strip()
                elif "context:" in prompt.lower():
                    parts = prompt.split("context:")
                    if len(parts) > 1:
                        context = parts[1].strip()

                if "## 사용자 질문" in prompt:
                    parts = prompt.split("## 사용자 질문")
                    if len(parts) > 1:
                        question = parts[1].split("##")[0].strip()
                elif "질문:" in prompt:
                    question_part = prompt.split("질문:")[-1]
                    if "##" in question_part:
                        question = question_part.split("##")[0].strip()
                    else:
                        question = question_part.strip()
                elif "question:" in prompt.lower():
                    question_part = prompt.split("question:")[-1]
                    if "context:" in question_part:
                        question = question_part.split("context:")[0].strip()
                    else:
                        question = question_part.strip()

                # 컨텍스트가 있으면 검색 결과 기반 답변
                if context and context != "" and len(context) > 100:
                    # 검색 결과를 요약하여 답변 생성
                    return self._generate_response_from_context(question, context)

                # 컨텍스트가 없으면 기본 답변
                return "죄송합니다. 해당 질문에 대한 관련 법률 정보를 찾을 수 없었습니다. 다른 법률 조문이나 구체적인 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다."

            def _generate_response_from_context(self, question, context):
                """컨텍스트를 활용한 답변 생성 - 질문에 맞춰 핵심 내용 동적 추출"""
                # 검색 결과에서 핵심 내용 추출
                lines = context.split('\n')

                # 전체 컨텍스트 내용 추출
                contents = []
                for line in lines:
                    line = line.strip()
                    # 딕셔너리 형태나 JSON 형태 파싱
                    if line and len(line) > 20:
                        # {'score': ..., 'text': '...'} 형태 처리
                        if "'text':" in line or '"text":' in line:
                            try:
                                # 텍스트 추출
                                if "'text':" in line:
                                    text_part = line.split("'text':")[-1].strip().replace("'", "")
                                else:
                                    text_part = line.split('"text":')[-1].strip().replace('"', "")
                                if text_part and len(text_part) > 20:
                                    contents.append(text_part)
                            except (ValueError, IndexError):
                                pass
                        elif not line.startswith("{'score':"):
                            contents.append(line)

                # 질문의 핵심 키워드 추출
                question_keywords = self._extract_keywords(question)

                # 컨텍스트에서 질문과 가장 관련성 높은 내용 추출
                relevant_contents = self._extract_relevant_content(contents, question_keywords, question)

                # 관련성 높은 내용 선택 (질문 길이와 빈도 기반)
                main_content = self._select_best_content(relevant_contents, question)

                # 답변 생성
                if main_content:
                    # 질문 유형 파악
                    question_type = self._identify_question_type(question)

                    # 질문 유형별 적절한 서론 작성
                    intro = self._generate_intro(question, question_type)

                    # 핵심 내용을 질문에 맞게 구조화
                    structured_content = self._structure_content(main_content, question, question_type)

                    response_text = f"""{intro}

{structured_content}

구체적인 상황을 알려주시면 더 정확한 정보를 제공할 수 있습니다."""

                    return response_text
                else:
                    # 컨텍스트가 없으면 기본 답변
                    return "죄송합니다. 해당 질문에 대한 관련 법률 정보를 찾을 수 없었습니다. 다른 법률 조문이나 구체적인 상황을 알려주시면 더 정확한 답변을 드릴 수 있습니다."

            def _extract_keywords(self, question):
                """질문에서 핵심 키워드 추출"""
                # 법률 관련 키워드
                legal_keywords = [
                    "법", "조문", "조항", "법률", "법령", "규정", "판례", "판결",
                    "이혼", "협의이혼", "재산분할", "양육권", "양육비",
                    "상속", "유언", "상속분", "상속세", "상속인",
                    "근로", "근무", "임금", "퇴직금", "수당", "야간", "휴가",
                    "계약", "매매", "임대", "보증", "대리",
                    "손해배상", "배상", "불법행위", "채권", "채무",
                    "소송", "소제기", "관할", "증거", "집행",
                    "세금", "세법", "소득세", "부가가치세"
                ]

                # 질문을 소문자로 변환하여 키워드 매칭
                question_lower = question.lower()
                matched_keywords = [kw for kw in legal_keywords if kw in question_lower]

                # 질문 단어 중 의미있는 단어 추출 (2글자 이상)
                import re
                words = re.findall(r'\b\w{2,}\b', question)
                matched_keywords.extend([w for w in words if len(w) >= 2 and w not in matched_keywords])

                return matched_keywords

            def _extract_relevant_content(self, contents, keywords, question):
                """컨텍스트에서 질문과 관련성 높은 내용 추출"""
                if not contents:
                    return []

                # 각 컨텍스트의 관련성 점수 계산
                scored_contents = []
                for content in contents:
                    score = 0
                    content_lower = content.lower()
                    question_lower = question.lower()

                    # 키워드 매칭 점수
                    for keyword in keywords:
                        if keyword in content_lower:
                            score += 2

                    # 질문의 핵심 단어가 포함된 경우 추가 점수
                    for word in question_lower.split():
                        if len(word) >= 2 and word in content_lower:
                            score += 1

                    # 컨텍스트 길이도 고려 (너무 짧거나 길면 감점)
                    if 50 <= len(content) <= 1000:
                        score += 1

                    scored_contents.append((score, content))

                # 점수 순으로 정렬
                scored_contents.sort(reverse=True, key=lambda x: x[0])

                return scored_contents

            def _select_best_content(self, scored_contents, question):
                """가장 적합한 내용 선택"""
                if not scored_contents:
                    return ""

                # 상위 3개 선택하고 품질이 좋은 것만 포함
                selected_contents = []
                for score, content in scored_contents[:5]:
                    if score >= 2:  # 최소 점수 이상인 경우만 선택
                        selected_contents.append(content)

                if not selected_contents:
                    # 점수가 낮아도 최고 점수 내용은 포함
                    if scored_contents:
                        selected_contents.append(scored_contents[0][1])

                # 중복 제거 및 길이 조절
                unique_contents = []
                seen = set()
                total_length = 0
                max_length = 800  # 최대 800자

                for content in selected_contents:
                    content_hash = hash(content[:100])  # 중복 체크를 위한 해시
                    if content_hash not in seen:
                        if total_length + len(content) <= max_length:
                            unique_contents.append(content)
                            seen.add(content_hash)
                            total_length += len(content)
                        else:
                            # 공간이 부족하면 자름
                            remaining = max_length - total_length
                            if remaining > 100:
                                unique_contents.append(content[:remaining])
                            break

                return "\n\n".join(unique_contents)

            def _identify_question_type(self, question):
                """질문 유형 파악"""
                question_lower = question.lower()

                if any(kw in question_lower for kw in ["이혼", "협의이혼", "재산분할", "양육권"]):
                    return "가족법"
                elif any(kw in question_lower for kw in ["상속", "유언", "상속분", "상속세"]):
                    return "상속법"
                elif any(kw in question_lower for kw in ["근로", "근무", "임금", "퇴직금", "수당", "야간", "휴가"]):
                    return "노동법"
                elif any(kw in question_lower for kw in ["계약", "매매", "임대", "보증", "대리"]):
                    return "계약법"
                elif any(kw in question_lower for kw in ["손해배상", "배상", "불법행위", "채권", "채무"]):
                    return "민사법"
                elif any(kw in question_lower for kw in ["소송", "소제기", "관할", "증거", "집행"]):
                    return "민사소송법"
                elif any(kw in question_lower for kw in ["절도", "범죄", "형사", "사기", "폭행", "강도"]):
                    return "형사법"
                elif any(kw in question_lower for kw in ["세금", "세법", "소득세", "부가가치세"]):
                    return "세법"
                else:
                    return "일반"

            def _generate_intro(self, question, question_type):
                """질문 유형에 맞는 서론 생성"""
                if question_type == "일반":
                    return "관련 법률 정보를 확인했습니다."
                elif question_type == "가족법":
                    return "가족법 관련 질문이시군요. 관련 법률 정보입니다."
                elif question_type == "상속법":
                    return "상속 관련 법률 정보입니다."
                elif question_type == "노동법":
                    return "노동법 관련 정보입니다."
                elif question_type == "계약법":
                    return "계약 관련 법률 정보입니다."
                elif question_type == "민사법":
                    return "민사법 관련 정보입니다."
                elif question_type == "민사소송법":
                    return "민사소송 관련 법률 정보입니다."
                elif question_type == "형사법":
                    return "형사법 관련 정보입니다."
                elif question_type == "세법":
                    return "세법 관련 정보입니다."
                else:
                    return "관련 법률 정보를 확인했습니다."

            def _structure_content(self, content, question, question_type):
                """내용을 질문에 맞게 구조화"""
                # 이미 구조화된 내용인지 확인
                if "##" in content or "1." in content or "\n\n" in content:
                    return content

                # 내용을 문장 단위로 분리
                sentences = [s.strip() for s in content.split('.') if s.strip()]

                # 핵심 문장 우선 추출
                relevant_sentences = []
                question_words = set(question.split())

                for sentence in sentences:
                    if len(sentence) > 30:  # 너무 짧은 문장은 제외
                        sentence_words = set(sentence.lower().split())
                        # 질문과의 공통 단어가 있으면 우선 포함
                        if question_words & sentence_words:
                            relevant_sentences.insert(0, sentence)
                        else:
                            relevant_sentences.append(sentence)

                # 상위 5개만 선택
                result = ". ".join(relevant_sentences[:5])
                if result and not result.endswith('.'):
                    result += "."

                return result

            async def ainvoke(self, prompt):
                return self.invoke(prompt)

        logger.warning("No valid LLM provider configured or failed to initialize. Using Improved Mock LLM.")
        return ImprovedMockLLM()

    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구축"""
        workflow = StateGraph(LegalWorkflowState)

        # 기존 노드 추가
        workflow.add_node("classify_query", self.classify_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("generate_answer_enhanced", self.generate_answer_enhanced)
        workflow.add_node("format_response", self.format_response)

        # Phase 1: 입력 검증 및 특수 쿼리 처리 노드 추가
        workflow.add_node("validate_input", self.validate_input)
        workflow.add_node("detect_special_queries", self.detect_special_queries)
        workflow.add_node("handle_law_article", self.handle_law_article_query)
        workflow.add_node("handle_contract", self.handle_contract_query)

        # Phase 2: 하이브리드 질문 분석 및 법률 제한 검증 노드 추가
        workflow.add_node("analyze_query_hybrid", self.analyze_query_hybrid)
        workflow.add_node("validate_legal_restrictions", self.validate_legal_restrictions)
        workflow.add_node("generate_restricted_response", self.generate_restricted_response)

        # Phase 4: 답변 생성 폴백 체인 노드 추가
        workflow.add_node("try_specific_law_search", self.try_specific_law_search)
        workflow.add_node("try_unified_search", self.try_unified_search)
        workflow.add_node("try_rag_service", self.try_rag_service)
        workflow.add_node("generate_template_response", self.generate_template_response)

        # Phase 3: Phase 시스템 통합 노드 추가
        workflow.add_node("enrich_conversation_context", self.enrich_conversation_context)
        workflow.add_node("personalize_response", self.personalize_response)
        workflow.add_node("manage_memory_quality", self.manage_memory_quality)

        # Phase 5: 후처리 노드 추가
        workflow.add_node("enhance_completion", self.enhance_completion)
        workflow.add_node("add_disclaimer", self.add_disclaimer)

        # 엣지 설정 (Phase 1: 새로운 엔트리 포인트)
        workflow.set_entry_point("validate_input")
        workflow.add_edge("validate_input", "detect_special_queries")

        # 특수 쿼리 라우팅 (조건부)
        workflow.add_conditional_edges(
            "detect_special_queries",
            self.should_route_special,
            {
                "law_article": "handle_law_article",
                "contract": "handle_contract",
                "regular": "classify_query"
            }
        )

        # 특수 쿼리 핸들러에서 종료
        workflow.add_edge("handle_law_article", END)
        workflow.add_edge("handle_contract", END)

        # Phase 2: classify_query 다음에 하이브리드 분석 노드 추가
        workflow.add_edge("classify_query", "analyze_query_hybrid")
        workflow.add_edge("analyze_query_hybrid", "validate_legal_restrictions")

        # 법률 제한 검증 후 라우팅
        workflow.add_conditional_edges(
            "validate_legal_restrictions",
            self.should_continue_after_restriction,
            {
                "restricted": "generate_restricted_response",
                "continue": "retrieve_documents"
            }
        )
        workflow.add_edge("generate_restricted_response", END)

        # Phase 3: retrieve_documents 다음에 Phase 노드들 병렬 실행
        workflow.add_edge("retrieve_documents", "enrich_conversation_context")
        workflow.add_edge("retrieve_documents", "personalize_response")
        workflow.add_edge("retrieve_documents", "manage_memory_quality")

        # 모든 Phase가 완료되면 답변 생성으로
        workflow.add_edge("enrich_conversation_context", "generate_answer_enhanced")
        workflow.add_edge("personalize_response", "generate_answer_enhanced")
        workflow.add_edge("manage_memory_quality", "generate_answer_enhanced")

        # Phase 4: 폴백 체인 설정
        workflow.add_conditional_edges(
            "generate_answer_enhanced",
            self.route_generation_fallback,
            {
                "success": "format_response",
                "fallback": "try_specific_law_search"
            }
        )

        workflow.add_conditional_edges(
            "try_specific_law_search",
            self.route_generation_fallback,
            {
                "success": "format_response",
                "fallback": "try_unified_search"
            }
        )

        workflow.add_conditional_edges(
            "try_unified_search",
            self.route_generation_fallback,
            {
                "success": "format_response",
                "fallback": "try_rag_service"
            }
        )

        workflow.add_conditional_edges(
            "try_rag_service",
            self.route_generation_fallback,
            {
                "success": "format_response",
                "fallback": "generate_template_response"
            }
        )

        workflow.add_edge("generate_template_response", "format_response")

        # Phase 5: format_response 다음에 후처리 노드 추가
        workflow.add_edge("format_response", "enhance_completion")
        workflow.add_edge("enhance_completion", "add_disclaimer")
        workflow.add_edge("add_disclaimer", END)

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
            cached_documents = self.performance_optimizer.cache.get_cached_documents(query, query_type)

            if cached_documents:
                state["retrieved_docs"] = cached_documents
                state["processing_steps"].append(f"{len(cached_documents)}개 캐시된 문서 사용")
                print(f"Using cached documents for query: {query[:50]}...")
            else:
                # 벡터 검색 우선 시도 (성능 최적화: top_k를 5에서 3으로 감소)
                documents = []

                # 벡터 스토어에서 검색 시도
                try:
                    if hasattr(self, 'vector_store') and self.vector_store and hasattr(self.vector_store, 'search'):
                        vector_results = self.vector_store.search(query, top_k=3)  # 5→3으로 최적화
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

                            # 벡터 검색 결과 캐싱
                            self.performance_optimizer.cache.cache_documents(query, query_type, documents)
                except Exception as e:
                    print(f"⚠️ Vector search failed: {e}")

                # 벡터 검색 결과가 충분하면 DB 검색 생략 (성능 최적화)
                if len(documents) >= 3:
                    print(f"✅ 벡터 검색 결과 충분 ({len(documents)}개). DB 검색 생략")
                else:
                    # 데이터베이스 검색 수행 (결과가 부족한 경우만)
                    try:
                        print(f"🔍 데이터베이스 검색 시작: query='{query}', query_type='{query_type}'")
                        db_documents = self.data_connector.search_documents(query, query_type, limit=3)  # 5→3으로 최적화
                        print(f"✅ 데이터베이스 검색 완료: {len(db_documents)}개 문서 발견")

                        # 중복 제거
                        existing_contents = {doc["content"][:100] for doc in documents}
                        for doc in db_documents:
                            if doc.get("content", "")[:100] not in existing_contents:
                                documents.append(doc)
                        print(f"📊 데이터베이스 검색으로 {len([doc for doc in db_documents if doc.get('content', '')[:100] not in existing_contents])}개 문서 추가")
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

            # 컨텍스트 구성 - 메타데이터 제외하고 실제 내용만 추출
            import re
            context_parts = []
            for doc in retrieved_docs:
                if not doc:
                    continue

                # content 필드에서 실제 텍스트 추출
                content = doc.get("content", "")

                # content가 딕셔너리인 경우 'text' 필드에서 추출
                if isinstance(content, dict):
                    content_text = content.get("text", "")
                # content가 문자열인 경우
                elif isinstance(content, str):
                    # {'score': ..., 'text': '...'} 형태의 문자열인지 확인
                    if "'text':" in content or '"text":' in content:
                        # 텍스트 추출을 위한 정규식
                        text_pattern = r"(?:'text':|[\"']text[\"']:\s*)[\"']([^\"']+)[\"']"
                        matches = re.findall(text_pattern, content)
                        if matches:
                            content_text = matches[0]
                        else:
                            # 간단한 파싱 시도
                            if "'text':" in content:
                                parts = content.split("'text':")
                                if len(parts) > 1:
                                    text_part = parts[1].strip()
                                    # 따옴표 제거
                                    content_text = text_part.strip("'\"")
                                else:
                                    content_text = content
                            else:
                                content_text = content
                    else:
                        content_text = content
                # 그 외의 경우 문자열 변환
                else:
                    content_text = str(content)

                # 메타데이터 키워드 제거
                if "metadata:" in content_text or "law_id:" in content_text:
                    lines = content_text.split('\n')
                    content_text = '\n'.join([line for line in lines if "metadata:" not in line and "law_id:" not in line])

                # 최종 검증 및 추가
                if content_text and len(content_text) > 20 and not content_text.startswith("{"):
                    context_parts.append(content_text)

            context = "\n\n".join(context_parts)

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

            # 성공 플래그 설정
            updated_state["generation_success"] = True
            updated_state["generation_method"] = "enhanced_llm"

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

            # 실패 플래그 설정 (폴백 체인으로)
            updated_state["generation_success"] = False

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

    # ========== Phase 1: 입력 검증 및 특수 쿼리 처리 노드 ==========

    def validate_input(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """입력 검증 (enhanced_chat_service._validate_and_preprocess_input 로직)"""
        print(f"🔍 validate_input 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "errors" not in state:
                state["errors"] = []
            if "validation_results" not in state:
                state["validation_results"] = {}
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")

            # 검증 로직
            if not message or not message.strip():
                error = "메시지가 비어있습니다"
                state["errors"].append(error)
                state["validation_results"] = {"valid": False, "error": error}
            elif len(message) > 10000:
                error = "메시지가 너무 깁니다 (최대 10,000자)"
                state["errors"].append(error)
                state["validation_results"] = {"valid": False, "error": error}
            else:
                state["validation_results"] = {
                    "valid": True,
                    "message": message.strip(),
                    "length": len(message)
                }

            state["processing_steps"].append("입력 검증 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 입력 검증 완료: {state['validation_results'].get('valid', False)}")

        except Exception as e:
            error_msg = f"입력 검증 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def detect_special_queries(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """특수 쿼리 감지"""
        print(f"🔍 detect_special_queries 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "is_law_article_query" not in state:
                state["is_law_article_query"] = False
            if "is_contract_query" not in state:
                state["is_contract_query"] = False
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")

            # 법률 조문 쿼리 감지
            import re
            law_patterns = [
                r'(\w+법)\s*제\s*(\d+)조',
                r'제\s*(\d+)조',
                r'(\w+법)제(\d+)조'
            ]

            is_law_article = False
            for pattern in law_patterns:
                if re.search(pattern, message):
                    is_law_article = True
                    break

            state["is_law_article_query"] = is_law_article

            # 계약서 쿼리 감지
            contract_keywords = ["계약서", "계약", "작성", "체결", "계약이"]
            is_contract = any(keyword in message for keyword in contract_keywords)
            state["is_contract_query"] = is_contract

            state["processing_steps"].append(f"특수 쿼리 감지 완료 (법률조문: {is_law_article}, 계약서: {is_contract})")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 특수 쿼리 감지 완료: law_article={is_law_article}, contract={is_contract}")

        except Exception as e:
            error_msg = f"특수 쿼리 감지 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def should_route_special(self, state: LegalWorkflowState) -> str:
        """특수 쿼리 라우팅 결정"""
        try:
            if state.get("is_law_article_query"):
                return "law_article"
            elif state.get("is_contract_query"):
                return "contract"
            return "regular"
        except Exception as e:
            print(f"라우팅 결정 중 오류: {e}")
            return "regular"

    def handle_law_article_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """법률 조문 쿼리 처리"""
        print(f"🔍 handle_law_article_query 시작")
        start_time = time.time()

        try:
            # 현재는 법률 조문 검색 로직 연결 (향후 CurrentLawSearchEngine 통합)
            message = state.get("user_query", "")

            # 기본 응답 생성
            response_text = f"법률 조문 검색 기능은 현재 개발 중입니다. 질문: {message}"

            state["answer"] = response_text
            state["generated_response"] = response_text
            state["response"] = response_text
            state["generation_method"] = "law_article_query"
            state["generation_success"] = True

            if "processing_steps" not in state:
                state["processing_steps"] = []
            state["processing_steps"].append("법률 조문 쿼리 처리 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 법률 조문 쿼리 처리 완료")

        except Exception as e:
            error_msg = f"법률 조문 쿼리 처리 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def handle_contract_query(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """계약서 쿼리 처리"""
        print(f"🔍 handle_contract_query 시작")
        start_time = time.time()

        try:
            message = state.get("user_query", "")

            # 기본 응답 생성 (향후 ContractQueryHandler 통합)
            response_text = f"계약서 관련 질문입니다. 질문: {message}"

            state["answer"] = response_text
            state["generated_response"] = response_text
            state["response"] = response_text
            state["generation_method"] = "contract_query"
            state["generation_success"] = True

            if "processing_steps" not in state:
                state["processing_steps"] = []
            state["processing_steps"].append("계약서 쿼리 처리 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 계약서 쿼리 처리 완료")

        except Exception as e:
            error_msg = f"계약서 쿼리 처리 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    # ========== Phase 2: 하이브리드 질문 분석 및 법률 제한 검증 노드 ==========

    def analyze_query_hybrid(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """하이브리드 질문 분석 (enhanced_chat_service._analyze_query 로직)"""
        print(f"🔍 analyze_query_hybrid 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "query_analysis" not in state:
                state["query_analysis"] = {}
            if "hybrid_classification" not in state:
                state["hybrid_classification"] = {}
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")

            # 하이브리드 분류기 사용
            try:
                from ..integrated_hybrid_classifier import (
                    IntegratedHybridQuestionClassifier,
                )

                classifier = IntegratedHybridQuestionClassifier(confidence_threshold=0.7)
                classification_result = classifier.classify(message)

                # 도메인 분석 (향후 구현)
                domain_analysis = {}

                # 결과를 state에 저장
                state["query_analysis"] = {
                    "query_type": classification_result.question_type_value,
                    "confidence": classification_result.confidence,
                    "domain": domain_analysis.get("domain", "general"),
                    "keywords": classification_result.features.get("keywords", []) if classification_result.features else [],
                    "classification_method": classification_result.method,
                    "hybrid_analysis": True
                }
                state["hybrid_classification"] = {
                    "result": classification_result,
                    "domain_analysis": domain_analysis
                }

            except Exception as e:
                # 폴백: 기본 분류
                state["query_analysis"] = {
                    "query_type": "general",
                    "confidence": 0.5,
                    "hybrid_analysis": False,
                    "error": str(e)
                }

            state["processing_steps"].append("하이브리드 쿼리 분석 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 하이브리드 쿼리 분석 완료")

        except Exception as e:
            error_msg = f"하이브리드 쿼리 분석 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def validate_legal_restrictions(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """법률 제한 검증 (enhanced_chat_service._validate_legal_restrictions 로직)"""
        print(f"🔍 validate_legal_restrictions 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "legal_restriction_result" not in state:
                state["legal_restriction_result"] = {}
            if "is_restricted" not in state:
                state["is_restricted"] = False
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")
            query_analysis = state.get("query_analysis", {})

            # 법률 제한 시스템 호출 (현재는 비활성화 상태이므로 기본값)
            restriction_result = {
                "restricted": False,
                "reason": None,
                "safe_response": None,
                "confidence": 1.0
            }

            state["legal_restriction_result"] = restriction_result
            state["is_restricted"] = restriction_result["restricted"]
            state["processing_steps"].append("법률 제한 검증 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 법률 제한 검증 완료: restricted={restriction_result['restricted']}")

        except Exception as e:
            error_msg = f"법률 제한 검증 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def should_continue_after_restriction(self, state: LegalWorkflowState) -> str:
        """제한 검증 후 라우팅"""
        try:
            if state.get("is_restricted"):
                return "restricted"
            return "continue"
        except Exception as e:
            print(f"라우팅 결정 중 오류: {e}")
            return "continue"

    def generate_restricted_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """제한된 응답 생성"""
        print(f"🔍 generate_restricted_response 시작")
        start_time = time.time()

        try:
            restriction_result = state.get("legal_restriction_result", {})

            # 제한된 응답 생성
            response_text = "죄송합니다. 해당 질문은 법률 제한으로 인해 답변 드릴 수 없습니다."

            if restriction_result.get("safe_response"):
                response_text = restriction_result["safe_response"]

            state["answer"] = response_text
            state["generated_response"] = response_text
            state["response"] = response_text
            state["generation_method"] = "restricted_response"
            state["generation_success"] = True

            if "processing_steps" not in state:
                state["processing_steps"] = []
            state["processing_steps"].append("제한된 응답 생성 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 제한된 응답 생성 완료")

        except Exception as e:
            error_msg = f"제한된 응답 생성 중 오류 발생: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    # ========== Phase 4: 답변 생성 폴백 체인 노드 ==========

    def route_generation_fallback(self, state: LegalWorkflowState) -> str:
        """답변 생성 폴백 라우팅"""
        try:
            if state.get("generation_success"):
                return "success"
            return "fallback"
        except Exception as e:
            print(f"폴백 라우팅 중 오류: {e}")
            return "fallback"

    def try_specific_law_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """특정 법률 조문 검색 (enhanced_chat_service._generate_enhanced_response 2순위 로직)"""
        print(f"🔍 try_specific_law_search 시작")
        start_time = time.time()

        try:
            query_analysis = state.get("query_analysis", {})
            message = state.get("user_query", "")

            # CurrentLawSearchEngine 사용
            if self.current_law_search_engine:
                try:
                    results = self.current_law_search_engine.search_current_laws(
                        query=message,
                        search_type='hybrid',
                        top_k=5
                    )

                    if results:
                        # 첫 번째 결과 사용
                        first_result = results[0]
                        response_text = f"관련 법령: {first_result.law_name_korean}\n\n{first_result.detailed_info}"

                        state["answer"] = response_text
                        state["generated_response"] = response_text
                        state["response"] = response_text
                        state["generation_method"] = "current_law_search"
                        state["generation_success"] = True
                        state["processing_steps"] = state.get("processing_steps", [])
                        state["processing_steps"].append(f"특정 법률 검색 성공: {len(results)}개 결과")
                        return state
                except Exception as e:
                    print(f"CurrentLawSearchEngine 검색 실패: {e}")

            # 실패 시
            state["generation_success"] = False
            state["processing_steps"] = state.get("processing_steps", [])
            state["processing_steps"].append("특정 법률 검색 실패 또는 미구현")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

        except Exception as e:
            error_msg = f"특정 법률 검색 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            state["generation_success"] = False
            print(f"❌ {error_msg}")

        return state

    def try_unified_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """통합 검색 엔진 (enhanced_chat_service._generate_enhanced_response 3순위 로직)"""
        print(f"🔍 try_unified_search 시작")
        start_time = time.time()

        try:
            message = state.get("user_query", "")

            # UnifiedSearchEngine 사용 (비동기는 동기로 변환)
            if self.unified_search_engine:
                try:
                    import asyncio
                    # 비동기 함수를 동기적으로 호출
                    if hasattr(self.unified_search_engine, 'search'):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        search_result = loop.run_until_complete(
                            self.unified_search_engine.search(
                                query=message,
                                top_k=5,
                                search_types=['vector', 'exact'],
                                use_cache=True
                            )
                        )
                        loop.close()

                        if search_result.results:
                            # 검색 결과를 답변으로 변환
                            sources_text = "\n\n".join([
                                f"- {r.get('title', r.get('content', ''))[:200]}"
                                for r in search_result.results[:3]
                            ])

                            response_text = f"관련 문서를 찾았습니다:\n\n{sources_text}"

                            state["answer"] = response_text
                            state["generated_response"] = response_text
                            state["response"] = response_text
                            state["generation_method"] = "unified_search"
                            state["generation_success"] = True
                            state["processing_steps"] = state.get("processing_steps", [])
                            state["processing_steps"].append(f"통합 검색 성공: {len(search_result.results)}개 결과")
                            return state
                except Exception as e:
                    print(f"UnifiedSearchEngine 검색 실패: {e}")

            # 실패 시
            state["generation_success"] = False
            state["processing_steps"] = state.get("processing_steps", [])
            state["processing_steps"].append("통합 검색 실패 또는 미구현")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

        except Exception as e:
            error_msg = f"통합 검색 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            state["generation_success"] = False
            print(f"❌ {error_msg}")

        return state

    def try_rag_service(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """RAG 서비스 (enhanced_chat_service._generate_enhanced_response 4순위 로직)"""
        print(f"🔍 try_rag_service 시작")
        start_time = time.time()

        try:
            message = state.get("user_query", "")
            query_analysis = state.get("query_analysis", {})

            # UnifiedRAGService 사용 (비동기는 동기로 변환)
            if self.unified_rag_service:
                try:
                    import asyncio
                    # 비동기 함수를 동기적으로 호출
                    if hasattr(self.unified_rag_service, 'generate_response'):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        rag_response = loop.run_until_complete(
                            self.unified_rag_service.generate_response(
                                query=message,
                                max_length=500,
                                top_k=10,
                                use_cache=True
                            )
                        )
                        loop.close()

                        if rag_response and hasattr(rag_response, 'response'):
                            response_text = rag_response.response

                            state["answer"] = response_text
                            state["generated_response"] = response_text
                            state["response"] = response_text
                            state["generation_method"] = "unified_rag"
                            state["generation_success"] = True
                            state["confidence"] = rag_response.confidence if hasattr(rag_response, 'confidence') else 0.7
                            state["processing_steps"] = state.get("processing_steps", [])
                            state["processing_steps"].append("RAG 서비스 성공")
                            return state
                except Exception as e:
                    print(f"UnifiedRAGService 처리 실패: {e}")

            # 실패 시
            state["generation_success"] = False
            state["processing_steps"] = state.get("processing_steps", [])
            state["processing_steps"].append("RAG 서비스 실패 또는 미구현")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

        except Exception as e:
            error_msg = f"RAG 서비스 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            state["generation_success"] = False
            print(f"❌ {error_msg}")

        return state

    def generate_template_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """템플릿 기반 답변 (enhanced_chat_service._generate_improved_template_response 로직)"""
        print(f"🔍 generate_template_response 시작")
        start_time = time.time()

        try:
            message = state.get("user_query", "")
            query_type = state.get("query_type", "GENERAL_QUESTION")

            # 템플릿 기반 기본 답변 생성
            templates = {
                "FAMILY_LAW": "가족법 관련 질문이시군요. 상세한 사안을 알려주시면 더 정확한 답변을 드릴 수 있습니다.",
                "CRIMINAL_LAW": "형사법 관련 질문이시군요. 구체적인 상황을 설명해주시면 관련 조문을 찾아드리겠습니다.",
                "CIVIL_LAW": "민사법 관련 질문이시군요. 자세한 내용을 알려주시면 법률 조언을 드리겠습니다.",
                "LABOR_LAW": "노동법 관련 질문이시군요. 구체적인 사안을 알려주시면 관련 법령을 찾아드리겠습니다.",
            }

            response_text = templates.get(query_type,
                f"죄송합니다. '{message}'에 대한 답변을 생성할 수 없었습니다. "
                "다른 방식으로 문의해주시면 도움을 드리겠습니다.")

            state["answer"] = response_text
            state["generated_response"] = response_text
            state["response"] = response_text
            state["generation_method"] = "template"
            state["generation_success"] = True
            state["confidence"] = 0.5
            state["processing_steps"] = state.get("processing_steps", [])
            state["processing_steps"].append("템플릿 기반 답변 생성 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 템플릿 기반 답변 생성 완료")

        except Exception as e:
            error_msg = f"템플릿 기반 답변 생성 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            state["generation_success"] = False
            print(f"❌ {error_msg}")

        return state

    # ========== Phase 3: Phase 시스템 통합 노드 ==========

    def enrich_conversation_context(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """Phase 1: 대화 맥락 강화 (enhanced_chat_service._process_phase1_context 로직)"""
        print(f"🔍 enrich_conversation_context 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "phase1_context" not in state:
                state["phase1_context"] = {}
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")
            session_id = state.get("session_id", "")
            user_id = state.get("user_id", "")

            # Phase 1 정보 설정
            phase1_info = {
                "session_context": None,
                "multi_turn_context": None,
                "compressed_context": None,
                "enabled": False  # 현재 비활성화 상태
            }

            # 실제 Phase 1 로직 (향후 활성화 시 구현)
            # integrated_session_manager, multi_turn_handler, context_compressor 호출

            state["phase1_context"] = phase1_info
            state["processing_steps"].append("Phase 1: 대화 맥락 강화 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ Phase 1: 대화 맥락 강화 완료")

        except Exception as e:
            error_msg = f"Phase 1 처리 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def personalize_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """Phase 2: 개인화 (enhanced_chat_service._process_phase2_personalization 로직)"""
        print(f"🔍 personalize_response 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "phase2_personalization" not in state:
                state["phase2_personalization"] = {}
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")
            user_id = state.get("user_id", "")
            # session_id와 phase1_info는 향후 확장 시 사용 예정
            _session_id = state.get("session_id", "")
            _phase1_info = state.get("phase1_context", {})

            # Phase 2 정보 설정
            phase2_info = {
                "user_profile": None,
                "emotion_intent": None,
                "conversation_flow": None,
                "enabled": True  # UserProfileManager 사용 시 활성화
            }

            # UserProfileManager를 사용한 실제 Phase 2 로직 구현
            if self.user_profile_manager and user_id:
                try:
                    # 1. 사용자 프로필 조회 또는 생성
                    profile = self.user_profile_manager.get_profile(user_id)
                    if not profile:
                        # 프로필이 없으면 기본 프로필 생성
                        self.user_profile_manager.create_profile(user_id, {})
                        profile = self.user_profile_manager.get_profile(user_id)

                    if profile:
                        # 2. 개인화된 컨텍스트 생성
                        personalized_context = self.user_profile_manager.get_personalized_context(
                            user_id, message
                        )

                        # 3. 관심 분야 업데이트
                        self.user_profile_manager.update_interest_areas(user_id, message)

                        # 4. 상태에 개인화 정보 설정
                        phase2_info["user_profile"] = personalized_context

                        # 5. 전역 상태에도 개인화 정보 반영
                        if "user_expertise_level" not in state:
                            state["user_expertise_level"] = profile.get("expertise_level", "beginner")
                        else:
                            state["user_expertise_level"] = profile.get("expertise_level", state["user_expertise_level"])

                        if "preferred_response_style" not in state:
                            state["preferred_response_style"] = personalized_context.get("response_style", "medium")
                        else:
                            state["preferred_response_style"] = personalized_context.get("response_style", state["preferred_response_style"])

                        state["expertise_context"] = personalized_context.get("expertise_context", {})
                        state["interest_areas"] = personalized_context.get("interest_areas", [])
                        state["personalization_score"] = personalized_context.get("personalization_score", 0.0)

                        logger.info(f"✅ Phase 2: 개인화 완료 - 전문성: {profile.get('expertise_level')}, 관심분야: {len(personalized_context.get('interest_areas', []))}개")
                    else:
                        logger.warning("프로필 생성 또는 조회 실패")

                except Exception as e:
                    logger.error(f"UserProfileManager 처리 중 오류: {e}")
                    # 에러가 나도 계속 진행
                    phase2_info["enabled"] = False
            else:
                logger.info("UserProfileManager를 사용할 수 없음 - 기본 모드로 진행")

            state["phase2_personalization"] = phase2_info
            state["processing_steps"].append("Phase 2: 개인화 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ Phase 2: 개인화 완료")

        except Exception as e:
            error_msg = f"Phase 2 처리 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            logger.error(error_msg)
            print(f"❌ {error_msg}")

        return state

    def manage_memory_quality(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """Phase 3: 장기 기억 및 품질 모니터링 (enhanced_chat_service._process_phase3_memory_quality 로직)"""
        print(f"🔍 manage_memory_quality 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "phase3_memory_quality" not in state:
                state["phase3_memory_quality"] = {}
            if "processing_steps" not in state:
                state["processing_steps"] = []

            message = state.get("user_query", "")
            user_id = state.get("user_id", "")
            # session_id, phase1_info, phase2_info는 향후 확장 시 사용 예정
            _session_id = state.get("session_id", "")
            _phase1_info = state.get("phase1_context", {})
            _phase2_info = state.get("phase2_personalization", {})

            # Phase 3 정보 설정
            phase3_info = {
                "contextual_memory": None,
                "quality_metrics": None,
                "enabled": False  # 현재 비활성화 상태
            }

            # 실제 Phase 3 로직 (향후 활성화 시 구현)
            # contextual_memory_manager, conversation_quality_monitor 호출

            state["phase3_memory_quality"] = phase3_info
            state["processing_steps"].append("Phase 3: 장기 기억 및 품질 모니터링 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ Phase 3: 장기 기억 및 품질 모니터링 완료")

        except Exception as e:
            error_msg = f"Phase 3 처리 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    # ========== Phase 5: 후처리 노드 ==========

    def enhance_completion(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """답변 완성도 검증 및 보완 (enhanced_chat_service.process_message 로직)"""
        print(f"🔍 enhance_completion 시작")
        start_time = time.time()

        try:
            # 상태 초기화
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "completion_result" not in state:
                state["completion_result"] = {}

            response_text = state.get("response", "")
            # message와 query_analysis는 향후 확장 시 사용 예정
            _message = state.get("user_query", "")
            _query_analysis = state.get("query_analysis", {})

            # 향후 enhanced_completion_system 통합
            # 현재는 기본 검증만 수행
            was_truncated = False
            if response_text and len(response_text) < 50:
                # 너무 짧은 답변은 보완 필요
                was_truncated = True
                state["completion_result"] = {
                    "improved": True,
                    "method": "length_validation",
                    "confidence": 0.7
                }

            if was_truncated:
                # 답변을 조금 더 풍부하게 만들어줌
                enhanced_response = response_text + "\n\n추가 정보가 필요하시면 더 구체적으로 질문해주세요."
                state["response"] = enhanced_response
                state["answer"] = enhanced_response
                state["generated_response"] = enhanced_response

            state["processing_steps"].append("답변 완성도 검증 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

            print(f"✅ 답변 완성도 검증 완료")

        except Exception as e:
            error_msg = f"답변 완성도 검증 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")

        return state

    def add_disclaimer(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """면책 조항 추가 (UserPreferenceManager 통합)"""
        print(f"🔍 add_disclaimer 시작")
        start_time = time.time()

        try:
            # UserPreferenceManager import
            from ...user_preference_manager import (
                DisclaimerPosition,
                DisclaimerStyle,
                preference_manager,
            )

            # 상태 초기화
            if "processing_steps" not in state:
                state["processing_steps"] = []
            if "disclaimer_added" not in state:
                state["disclaimer_added"] = False

            response_text = state.get("response", "")

            # 사용자 설정 가져오기 (state에서 또는 기본값)
            user_preferences = state.get("user_preferences", {})
            show_disclaimer = user_preferences.get("show_disclaimer", True)

            # 사용자 설정에 따라 면책 조항 추가
            if response_text and show_disclaimer:
                # 스타일 가져오기
                disclaimer_style_str = user_preferences.get("disclaimer_style", "natural")
                try:
                    disclaimer_style = DisclaimerStyle(disclaimer_style_str)
                except ValueError:
                    disclaimer_style = DisclaimerStyle.NATURAL

                # 위치 가져오기
                disclaimer_position_str = user_preferences.get("disclaimer_position", "end")
                try:
                    disclaimer_position = DisclaimerPosition(disclaimer_position_str)
                except ValueError:
                    disclaimer_position = DisclaimerPosition.END

                # preference_manager에 현재 설정 반영
                if hasattr(preference_manager, 'preferences'):
                    preference_manager.preferences.disclaimer_style = disclaimer_style
                    preference_manager.preferences.disclaimer_position = disclaimer_position
                    preference_manager.preferences.show_disclaimer = show_disclaimer

                # UserPreferenceManager를 사용하여 면책 조항 추가
                question_text = state.get("user_query", "")
                enhanced_response = preference_manager.add_disclaimer_to_response(
                    response_text,
                    question_text
                )

                # 면책 조항이 추가된 경우에만 상태 업데이트
                if enhanced_response != response_text:
                    state["response"] = enhanced_response
                    state["answer"] = enhanced_response
                    state["generated_response"] = enhanced_response
                    state["disclaimer_added"] = True
                    print(f"✅ 면책 조항 추가 완료 (스타일: {disclaimer_style.value}, 위치: {disclaimer_position.value})")
                else:
                    print(f"ℹ️ 면책 조항 추가 안함 (설정에 따라 건너뜀)")

            state["processing_steps"].append("면책 조항 추가 완료")

            processing_time = time.time() - start_time
            state["processing_time"] = state.get("processing_time", 0.0) + processing_time

        except ImportError as e:
            # UserPreferenceManager를 import할 수 없는 경우 기본 로직 사용
            print(f"⚠️ UserPreferenceManager를 import할 수 없습니다. 기본 로직 사용: {e}")
            response_text = state.get("response", "")

            if response_text and not response_text.endswith(".") and not response_text.endswith("!"):
                disclaimer = "\n\n※ 이 답변은 일반적인 법률 정보 제공을 목적으로 하며, 구체적인 법률 자문은 변호사와 상담하시기 바랍니다."
                state["response"] = response_text + disclaimer
                state["answer"] = state["response"]
                state["generated_response"] = state["response"]
                state["disclaimer_added"] = True
                print(f"✅ 기본 면책 조항 추가 완료")

        except Exception as e:
            error_msg = f"면책 조항 추가 중 오류: {str(e)}"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")

        return state
