# -*- coding: utf-8 -*-
"""
고급 법률 워크플로우
멀티 에이전트 시스템과 고급 기능을 포함한 법률 워크플로우
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from ...utils.langgraph_config import LangGraphConfig
from .keyword_mapper import LegalKeywordMapper
from .legal_data_connector import LegalDataConnector
from .performance_optimizer import PerformanceOptimizer
from .prompt_templates import LegalPromptTemplates
from .state_definitions import (
    LegalWorkflowState,
    create_initial_state,
    update_workflow_step,
)

logger = logging.getLogger(__name__)


class AdvancedLegalWorkflow:
    """고급 법률 워크플로우"""

    def __init__(self, config: LangGraphConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 컴포넌트 초기화
        self.performance_optimizer = PerformanceOptimizer()
        self.legal_data_connector = LegalDataConnector()
        self.prompt_templates = LegalPromptTemplates()
        self.keyword_mapper = LegalKeywordMapper()

        # LLM 초기화
        self._initialize_llms()

        # 워크플로우 구성
        self.graph = self._build_advanced_workflow()

        self.logger.info("AdvancedLegalWorkflow initialized")

    def _initialize_llms(self):
        """LLM 초기화"""
        try:
            if self.config.llm_provider == "local":
                self.llm = Ollama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url
                )
            elif self.config.llm_provider == "google":
                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.google_model,
                    temperature=0.7
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

            self.logger.info(f"LLM initialized: {self.config.llm_provider}")

        except Exception as e:
            self.logger.error(f"LLM initialization failed: {e}")
            self.llm = None

    def _build_advanced_workflow(self) -> StateGraph:
        """고급 워크플로우 구성"""
        workflow = StateGraph(LegalWorkflowState)

        # === 입력 처리 단계 ===
        workflow.add_node("input_validation", self._validate_input)
        workflow.add_node("context_enrichment", self._enrich_context)
        workflow.add_node("intent_classification", self._classify_intent)

        # === 멀티 에이전트 시스템 ===
        workflow.add_node("research_agent", self._research_agent)
        workflow.add_node("analysis_agent", self._analysis_agent)
        workflow.add_node("review_agent", self._review_agent)
        workflow.add_node("synthesis_agent", self._synthesis_agent)

        # === 병렬 처리 ===
        workflow.add_node("parallel_search", self._parallel_search)
        workflow.add_node("parallel_analysis", self._parallel_analysis)

        # === 품질 보증 ===
        workflow.add_node("quality_feedback", self._quality_feedback)
        workflow.add_node("refinement", self._refinement)
        workflow.add_node("final_validation", self._final_validation)

        # === 출력 생성 ===
        workflow.add_node("response_generation", self._generate_response)
        workflow.add_node("response_formatting", self._format_response)

        # === 워크플로우 연결 ===
        # 입력 처리 체인
        workflow.add_edge("input_validation", "context_enrichment")
        workflow.add_edge("context_enrichment", "intent_classification")

        # 조건부 라우팅
        workflow.add_conditional_edges(
            "intent_classification",
            self._route_by_intent,
            {
                "simple_query": "research_agent",
                "complex_query": "parallel_search",
                "analysis_needed": "analysis_agent",
                "review_needed": "review_agent"
            }
        )

        # 멀티 에이전트 조정
        workflow.add_edge("research_agent", "synthesis_agent")
        workflow.add_edge("analysis_agent", "synthesis_agent")
        workflow.add_edge("review_agent", "synthesis_agent")
        workflow.add_edge("parallel_search", "parallel_analysis")
        workflow.add_edge("parallel_analysis", "synthesis_agent")

        # 품질 보증 체인
        workflow.add_edge("synthesis_agent", "quality_feedback")
        workflow.add_conditional_edges(
            "quality_feedback",
            self._should_refine,
            {
                "refine": "refinement",
                "validate": "final_validation"
            }
        )
        workflow.add_edge("refinement", "quality_feedback")
        workflow.add_edge("final_validation", "response_generation")

        # 출력 체인
        workflow.add_edge("response_generation", "response_formatting")
        workflow.add_edge("response_formatting", END)

        # 진입점 설정
        workflow.set_entry_point("input_validation")

        return workflow

    # === 입력 처리 노드들 ===

    async def _validate_input(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """입력 검증"""
        try:
            self.logger.info("Validating input...")

            validation_result = {
                "is_valid": True,
                "validation_time": time.time(),
                "message_length": len(state["user_query"]),
                "has_context": bool(state["context"]),
                "session_valid": bool(state["session_id"]),
                "user_valid": bool(state["user_id"]),
                "complexity_score": self._calculate_complexity(state["user_query"])
            }

            # 입력 검증 로직
            if not state["user_query"] or len(state["user_query"].strip()) == 0:
                validation_result["is_valid"] = False
                validation_result["error"] = "Empty query"

            if len(state["user_query"]) > 10000:
                validation_result["is_valid"] = False
                validation_result["error"] = "Query too long"

            # 법률 관련 키워드 검증
            legal_keywords = self.keyword_mapper.extract_keywords(state["user_query"])
            validation_result["legal_keywords"] = legal_keywords
            validation_result["is_legal_query"] = len(legal_keywords) > 0

            return update_workflow_step(state, "input_validation", validation_result)

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return update_workflow_step(state, "input_validation", {
                "is_valid": False,
                "error": str(e)
            })

    async def _enrich_context(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """컨텍스트 강화"""
        try:
            self.logger.info("Enriching context...")

            enriched_context = {
                "timestamp": datetime.now().isoformat(),
                "query_type": "legal_inquiry",
                "domain_hints": self._extract_domain_hints(state["user_query"]),
                "complexity_score": state.get("input_validation", {}).get("complexity_score", 0.5),
                "context_enhanced": True,
                "user_expertise": self._infer_user_expertise(state["user_query"]),
                "legal_jurisdiction": "korea",
                "language": "korean"
            }

            # 대화 히스토리 분석
            conversation_history = state.get("conversation_history", [])
            if conversation_history:
                enriched_context["conversation_context"] = self._analyze_conversation_context(conversation_history)

            return update_workflow_step(state, "context_enrichment", enriched_context)

        except Exception as e:
            self.logger.error(f"Context enrichment failed: {e}")
            return update_workflow_step(state, "context_enrichment", {"error": str(e)})

    async def _classify_intent(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """의도 분류"""
        try:
            self.logger.info("Classifying intent...")

            intent_classification = {
                "primary_intent": "legal_inquiry",
                "secondary_intents": [],
                "confidence": 0.8,
                "intent_features": self._extract_intent_features(state["user_query"]),
                "classification_time": time.time()
            }

            # 의도별 세부 분류
            query_lower = state["user_query"].lower()

            if any(keyword in query_lower for keyword in ["계약서", "작성", "만들"]):
                intent_classification["primary_intent"] = "contract_creation"
            elif any(keyword in query_lower for keyword in ["판례", "사례", "예시"]):
                intent_classification["primary_intent"] = "precedent_search"
            elif any(keyword in query_lower for keyword in ["법률", "조문", "규정"]):
                intent_classification["primary_intent"] = "law_inquiry"
            elif any(keyword in query_lower for keyword in ["절차", "방법", "어떻게"]):
                intent_classification["primary_intent"] = "procedure_guide"

            return update_workflow_step(state, "intent_classification", intent_classification)

        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return update_workflow_step(state, "intent_classification", {"error": str(e)})

    # === 멀티 에이전트 노드들 ===

    async def _research_agent(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """연구 에이전트"""
        try:
            self.logger.info("Research agent working...")

            research_result = {
                "agent_name": "research_agent",
                "task": "document_retrieval",
                "start_time": time.time(),
                "status": "running"
            }

            # 문서 검색 로직
            query = state["user_query"]
            domain_hints = state.get("enriched_context", {}).get("domain_hints", [])

            # 법률 문서 검색
            legal_documents = await self.legal_data_connector.search_legal_documents(
                query, domain_hints
            )

            research_result.update({
                "retrieved_documents": legal_documents,
                "document_count": len(legal_documents),
                "search_confidence": 0.8,
                "end_time": time.time(),
                "status": "completed"
            })

            return update_workflow_step(state, "research_agent", research_result)

        except Exception as e:
            self.logger.error(f"Research agent failed: {e}")
            return update_workflow_step(state, "research_agent", {"error": str(e)})

    async def _analysis_agent(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """분석 에이전트"""
        try:
            self.logger.info("Analysis agent working...")

            analysis_result = {
                "agent_name": "analysis_agent",
                "task": "legal_analysis",
                "start_time": time.time(),
                "status": "running"
            }

            # 법률 분석 로직
            query = state["user_query"]
            documents = state.get("research_agent", {}).get("retrieved_documents", [])

            # LLM을 사용한 분석
            if self.llm:
                analysis_prompt = self.prompt_templates.get_analysis_prompt(query, documents)
                analysis_response = await self.llm.ainvoke(analysis_prompt)

                analysis_result.update({
                    "analysis_result": analysis_response,
                    "analysis_confidence": 0.8,
                    "key_points": self._extract_key_points(analysis_response),
                    "legal_recommendations": self._extract_recommendations(analysis_response)
                })

            analysis_result.update({
                "end_time": time.time(),
                "status": "completed"
            })

            return update_workflow_step(state, "analysis_agent", analysis_result)

        except Exception as e:
            self.logger.error(f"Analysis agent failed: {e}")
            return update_workflow_step(state, "analysis_agent", {"error": str(e)})

    async def _review_agent(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """검토 에이전트"""
        try:
            self.logger.info("Review agent working...")

            review_result = {
                "agent_name": "review_agent",
                "task": "quality_review",
                "start_time": time.time(),
                "status": "running"
            }

            # 품질 검토 로직
            analysis_result = state.get("analysis_agent", {}).get("analysis_result", "")
            research_result = state.get("research_agent", {}).get("retrieved_documents", [])

            # 품질 평가
            quality_score = self._evaluate_quality(analysis_result, research_result)

            review_result.update({
                "quality_score": quality_score,
                "review_feedback": self._generate_review_feedback(quality_score),
                "improvement_suggestions": self._generate_improvement_suggestions(quality_score),
                "end_time": time.time(),
                "status": "completed"
            })

            return update_workflow_step(state, "review_agent", review_result)

        except Exception as e:
            self.logger.error(f"Review agent failed: {e}")
            return update_workflow_step(state, "review_agent", {"error": str(e)})

    async def _synthesis_agent(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """합성 에이전트"""
        try:
            self.logger.info("Synthesis agent working...")

            synthesis_result = {
                "agent_name": "synthesis_agent",
                "task": "response_synthesis",
                "start_time": time.time(),
                "status": "running"
            }

            # 결과 합성 로직
            research_data = state.get("research_agent", {})
            analysis_data = state.get("analysis_agent", {})
            review_data = state.get("review_agent", {})

            # 통합 응답 생성
            synthesized_response = await self._synthesize_response(
                research_data, analysis_data, review_data, state["user_query"]
            )

            synthesis_result.update({
                "synthesized_response": synthesized_response,
                "synthesis_confidence": 0.8,
                "sources_used": len(research_data.get("retrieved_documents", [])),
                "end_time": time.time(),
                "status": "completed"
            })

            return update_workflow_step(state, "synthesis_agent", synthesis_result)

        except Exception as e:
            self.logger.error(f"Synthesis agent failed: {e}")
            return update_workflow_step(state, "synthesis_agent", {"error": str(e)})

    # === 병렬 처리 노드들 ===

    async def _parallel_search(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """병렬 검색"""
        try:
            self.logger.info("Parallel search working...")

            # 병렬 검색 작업들
            tasks = [
                self._search_statutes(state["user_query"]),
                self._search_precedents(state["user_query"]),
                self._search_legal_articles(state["user_query"])
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            parallel_result = {
                "search_type": "parallel",
                "statute_results": results[0] if not isinstance(results[0], Exception) else [],
                "precedent_results": results[1] if not isinstance(results[1], Exception) else [],
                "article_results": results[2] if not isinstance(results[2], Exception) else [],
                "total_results": sum(len(r) for r in results if not isinstance(r, Exception)),
                "search_time": time.time()
            }

            return update_workflow_step(state, "parallel_search", parallel_result)

        except Exception as e:
            self.logger.error(f"Parallel search failed: {e}")
            return update_workflow_step(state, "parallel_search", {"error": str(e)})

    async def _parallel_analysis(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """병렬 분석"""
        try:
            self.logger.info("Parallel analysis working...")

            search_results = state.get("parallel_search", {})

            # 병렬 분석 작업들
            tasks = [
                self._analyze_statutes(search_results.get("statute_results", [])),
                self._analyze_precedents(search_results.get("precedent_results", [])),
                self._analyze_articles(search_results.get("article_results", []))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            parallel_analysis_result = {
                "analysis_type": "parallel",
                "statute_analysis": results[0] if not isinstance(results[0], Exception) else {},
                "precedent_analysis": results[1] if not isinstance(results[1], Exception) else {},
                "article_analysis": results[2] if not isinstance(results[2], Exception) else {},
                "analysis_time": time.time()
            }

            return update_workflow_step(state, "parallel_analysis", parallel_analysis_result)

        except Exception as e:
            self.logger.error(f"Parallel analysis failed: {e}")
            return update_workflow_step(state, "parallel_analysis", {"error": str(e)})

    # === 품질 보증 노드들 ===

    async def _quality_feedback(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """품질 피드백"""
        try:
            self.logger.info("Quality feedback working...")

            synthesis_result = state.get("synthesis_agent", {})
            response = synthesis_result.get("synthesized_response", "")

            # 품질 평가
            quality_metrics = {
                "response_length": len(response),
                "confidence_score": synthesis_result.get("synthesis_confidence", 0.0),
                "relevance_score": self._calculate_relevance(response, state["user_query"]),
                "completeness_score": self._calculate_completeness(response),
                "clarity_score": self._calculate_clarity(response),
                "overall_quality": 0.0
            }

            # 전체 품질 점수 계산
            quality_metrics["overall_quality"] = (
                quality_metrics["confidence_score"] * 0.3 +
                quality_metrics["relevance_score"] * 0.3 +
                quality_metrics["completeness_score"] * 0.2 +
                quality_metrics["clarity_score"] * 0.2
            )

            quality_feedback_result = {
                "quality_metrics": quality_metrics,
                "quality_threshold_met": quality_metrics["overall_quality"] >= 0.7,
                "feedback_time": time.time()
            }

            return update_workflow_step(state, "quality_feedback", quality_feedback_result)

        except Exception as e:
            self.logger.error(f"Quality feedback failed: {e}")
            return update_workflow_step(state, "quality_feedback", {"error": str(e)})

    async def _refinement(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 개선"""
        try:
            self.logger.info("Refinement working...")

            synthesis_result = state.get("synthesis_agent", {})
            quality_metrics = state.get("quality_feedback", {}).get("quality_metrics", {})

            # 개선이 필요한 부분 식별
            improvement_areas = []
            if quality_metrics.get("relevance_score", 0) < 0.7:
                improvement_areas.append("relevance")
            if quality_metrics.get("completeness_score", 0) < 0.7:
                improvement_areas.append("completeness")
            if quality_metrics.get("clarity_score", 0) < 0.7:
                improvement_areas.append("clarity")

            # 응답 개선
            original_response = synthesis_result.get("synthesized_response", "")
            refined_response = await self._improve_response(original_response, improvement_areas)

            refinement_result = {
                "improvement_areas": improvement_areas,
                "original_response": original_response,
                "refined_response": refined_response,
                "improvement_applied": len(improvement_areas) > 0,
                "refinement_time": time.time()
            }

            return update_workflow_step(state, "refinement", refinement_result)

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return update_workflow_step(state, "refinement", {"error": str(e)})

    async def _final_validation(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """최종 검증"""
        try:
            self.logger.info("Final validation working...")

            # 최종 검증 로직
            final_response = state.get("refinement", {}).get("refined_response") or \
                           state.get("synthesis_agent", {}).get("synthesized_response", "")

            validation_result = {
                "validation_passed": True,
                "response_length": len(final_response),
                "contains_legal_disclaimer": "면책조항" in final_response or "참고" in final_response,
                "contains_sources": "참조" in final_response or "출처" in final_response,
                "validation_time": time.time()
            }

            # 최종 검증 조건
            if len(final_response) < 50:
                validation_result["validation_passed"] = False
                validation_result["error"] = "Response too short"

            return update_workflow_step(state, "final_validation", validation_result)

        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            return update_workflow_step(state, "final_validation", {"error": str(e)})

    # === 출력 생성 노드들 ===

    async def _generate_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 생성"""
        try:
            self.logger.info("Generating final response...")

            # 최종 응답 생성
            final_response = state.get("refinement", {}).get("refined_response") or \
                           state.get("synthesis_agent", {}).get("synthesized_response", "")

            # 면책 조항 추가 (UserPreferenceManager 통합)
            try:
                from ...user_preference_manager import (
                    DisclaimerPosition,
                    DisclaimerStyle,
                    preference_manager,
                )

                # 사용자 설정 가져오기
                user_preferences = state.get("user_preferences", {})
                show_disclaimer = user_preferences.get("show_disclaimer", True)

                if show_disclaimer:
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
                    final_response = preference_manager.add_disclaimer_to_response(
                        final_response,
                        question_text
                    )
                    self.logger.info(f"UserPreferenceManager를 사용하여 면책 조항 추가 (스타일: {disclaimer_style.value}, 위치: {disclaimer_position.value})")
                else:
                    self.logger.info("사용자 설정에 따라 면책 조항을 추가하지 않습니다.")

            except ImportError as e:
                # UserPreferenceManager를 import할 수 없는 경우 기본 로직 사용
                self.logger.warning(f"UserPreferenceManager를 import할 수 없습니다. 기본 로직 사용: {e}")
                disclaimer = "\n\n※ 이 답변은 일반적인 법률 정보 제공을 목적으로 하며, 구체적인 법률 문제에 대해서는 변호사와 상담하시기 바랍니다."
                final_response += disclaimer
            except Exception as e:
                # 기타 오류 발생 시 기본 로직 사용
                self.logger.warning(f"면책 조항 추가 중 오류: {e}. 기본 로직 사용.")
                disclaimer = "\n\n※ 이 답변은 일반적인 법률 정보 제공을 목적으로 하며, 구체적인 법률 문제에 대해서는 변호사와 상담하시기 바랍니다."
                final_response += disclaimer

            response_generation_result = {
                "final_response": final_response,
                "response_length": len(final_response),
                "generation_time": time.time(),
                "sources_count": len(state.get("research_agent", {}).get("retrieved_documents", []))
            }

            return update_workflow_step(state, "response_generation", response_generation_result)

        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return update_workflow_step(state, "response_generation", {"error": str(e)})

    async def _format_response(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """응답 포맷팅"""
        try:
            self.logger.info("Formatting response...")

            final_response = state.get("response_generation", {}).get("final_response", "")

            # 응답 포맷팅
            formatted_response = self._format_legal_response(final_response)

            formatting_result = {
                "formatted_response": formatted_response,
                "formatting_applied": True,
                "formatting_time": time.time()
            }

            # 최종 상태 업데이트
            state.update({
                "generated_response": formatted_response,
                "workflow_completed": True,
                "current_step": "completed"
            })

            return update_workflow_step(state, "response_formatting", formatting_result)

        except Exception as e:
            self.logger.error(f"Response formatting failed: {e}")
            return update_workflow_step(state, "response_formatting", {"error": str(e)})

    # === 라우팅 함수들 ===

    def _route_by_intent(self, state: LegalWorkflowState) -> str:
        """의도에 따른 라우팅"""
        intent_classification = state.get("intent_classification", {})
        primary_intent = intent_classification.get("primary_intent", "simple_query")

        routing_map = {
            "contract_creation": "analysis_needed",
            "precedent_search": "research_agent",
            "law_inquiry": "research_agent",
            "procedure_guide": "analysis_needed",
            "legal_inquiry": "simple_query"
        }

        return routing_map.get(primary_intent, "simple_query")

    def _should_refine(self, state: LegalWorkflowState) -> str:
        """개선 필요 여부 판단"""
        quality_feedback = state.get("quality_feedback", {})
        quality_threshold_met = quality_feedback.get("quality_threshold_met", True)

        return "validate" if quality_threshold_met else "refine"

    # === 유틸리티 함수들 ===

    def _calculate_complexity(self, query: str) -> float:
        """질문 복잡도 계산"""
        complexity_score = 0.0

        # 길이 기반 복잡도
        if len(query) > 100:
            complexity_score += 0.3
        elif len(query) > 50:
            complexity_score += 0.2
        else:
            complexity_score += 0.1

        # 키워드 기반 복잡도
        complex_keywords = ["복잡", "다양", "여러", "여러가지", "다양한", "복합"]
        if any(keyword in query for keyword in complex_keywords):
            complexity_score += 0.2

        # 질문 개수 기반 복잡도
        question_marks = query.count("?")
        if question_marks > 2:
            complexity_score += 0.3
        elif question_marks > 1:
            complexity_score += 0.2
        else:
            complexity_score += 0.1

        return min(complexity_score, 1.0)

    def _extract_domain_hints(self, query: str) -> List[str]:
        """도메인 힌트 추출"""
        domain_keywords = {
            "civil_law": ["민법", "계약", "손해배상", "불법행위"],
            "criminal_law": ["형법", "범죄", "처벌", "형량"],
            "family_law": ["이혼", "상속", "양육권", "친권"],
            "commercial_law": ["상법", "회사", "주식", "이사"],
            "labor_law": ["노동법", "근로", "임금", "해고"],
            "real_estate": ["부동산", "매매", "임대차", "등기"]
        }

        hints = []
        query_lower = query.lower()

        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                hints.append(domain)

        return hints

    def _infer_user_expertise(self, query: str) -> str:
        """사용자 전문성 수준 추론"""
        expert_keywords = ["법조문", "판례", "법원", "법령", "규정", "조항"]
        intermediate_keywords = ["법률", "법적", "법률적", "소송", "재판"]

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in expert_keywords):
            return "expert"
        elif any(keyword in query_lower for keyword in intermediate_keywords):
            return "intermediate"
        else:
            return "beginner"

    def _analyze_conversation_context(self, conversation_history: List) -> Dict[str, Any]:
        """대화 컨텍스트 분석"""
        return {
            "conversation_length": len(conversation_history),
            "last_topic": "legal_inquiry",
            "context_available": len(conversation_history) > 0
        }

    def _extract_intent_features(self, query: str) -> Dict[str, Any]:
        """의도 특징 추출"""
        return {
            "question_words": ["무엇", "어떻게", "왜", "언제", "어디서"].count(query),
            "action_words": ["작성", "검토", "분석", "해석", "적용"].count(query),
            "legal_words": ["법률", "법", "규정", "조항", "판례"].count(query)
        }

    def _extract_key_points(self, analysis: str) -> List[str]:
        """핵심 포인트 추출"""
        # 간단한 키 포인트 추출 로직
        sentences = analysis.split(".")
        return [s.strip() for s in sentences if len(s.strip()) > 20][:5]

    def _extract_recommendations(self, analysis: str) -> List[str]:
        """권장사항 추출"""
        # 간단한 권장사항 추출 로직
        recommendations = []
        if "권장" in analysis:
            recommendations.append("전문가 상담 권장")
        if "주의" in analysis:
            recommendations.append("주의사항 확인 필요")
        return recommendations

    def _evaluate_quality(self, analysis: str, documents: List[Dict]) -> float:
        """품질 평가"""
        quality_score = 0.5  # 기본 점수

        # 길이 기반 점수
        if len(analysis) > 200:
            quality_score += 0.2
        elif len(analysis) > 100:
            quality_score += 0.1

        # 문서 기반 점수
        if len(documents) > 0:
            quality_score += 0.2

        # 법률 키워드 기반 점수
        legal_keywords = ["법률", "법", "규정", "조항", "판례"]
        if any(keyword in analysis for keyword in legal_keywords):
            quality_score += 0.1

        return min(quality_score, 1.0)

    def _generate_review_feedback(self, quality_score: float) -> str:
        """검토 피드백 생성"""
        if quality_score >= 0.8:
            return "높은 품질의 분석입니다."
        elif quality_score >= 0.6:
            return "양호한 품질의 분석입니다."
        else:
            return "분석 품질을 개선할 필요가 있습니다."

    def _generate_improvement_suggestions(self, quality_score: float) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        if quality_score < 0.6:
            suggestions.append("더 구체적인 법률 근거를 추가하세요")
            suggestions.append("관련 판례를 더 많이 참조하세요")

        if quality_score < 0.8:
            suggestions.append("답변의 구조를 개선하세요")

        return suggestions

    async def _synthesize_response(self, research_data: Dict, analysis_data: Dict,
                                  review_data: Dict, query: str) -> str:
        """응답 합성 - 실제 검색된 문서 활용"""
        try:
            # 검색된 문서들 수집
            retrieved_docs = research_data.get("retrieved_documents", [])
            parallel_search = analysis_data.get("parallel_analysis", {})

            # 병렬 검색 결과도 수집
            statute_results = parallel_search.get("statute_results", [])
            precedent_results = parallel_search.get("precedent_results", [])
            article_results = parallel_search.get("article_results", [])

            # 병렬 분석 결과도 수집
            parallel_analysis = analysis_data.get("parallel_analysis", {})
            statute_analysis = parallel_analysis.get("statute_analysis", {})
            precedent_analysis = parallel_analysis.get("precedent_analysis", {})
            article_analysis = parallel_analysis.get("article_analysis", {})

            # 모든 문서 통합
            all_documents = retrieved_docs + statute_results + precedent_results + article_results

            if not all_documents:
                return f"죄송합니다. '{query}'와 관련된 법률 문서를 찾을 수 없습니다. 더 구체적인 질문을 해주시거나 다른 키워드로 시도해보세요."

            # 문서 기반 응답 생성
            response_parts = []

            # 관련 문서 요약
            response_parts.append(f"'{query}'에 대한 법률 정보를 찾았습니다:\n")

            # 카테고리별로 문서 정리
            categories = {}
            for doc in all_documents:
                category = doc.get("category", "기타")
                if category not in categories:
                    categories[category] = []
                categories[category].append(doc)

            # 각 카테고리별로 정보 제공
            for category, docs in categories.items():
                if docs:
                    category_name = self._get_category_name(category)
                    response_parts.append(f"\n## {category_name}")

                    # 분석 결과가 있으면 먼저 표시
                    if category == "labor_law" and statute_analysis.get("statute_details"):
                        response_parts.append("\n**관련 법령:**")
                        for detail in statute_analysis["statute_details"][:2]:
                            response_parts.append(f"{detail}")

                    if category in ["civil_law", "criminal_law", "family_law"] and precedent_analysis.get("precedent_details"):
                        response_parts.append("\n**관련 판례:**")
                        for detail in precedent_analysis["precedent_details"][:2]:
                            response_parts.append(f"{detail}")

                    # 문서 내용 표시
                    for doc in docs[:2]:  # 최대 2개 문서만 표시
                        title = doc.get("title", "제목 없음")
                        content = doc.get("content", "")
                        # 내용의 핵심 부분만 추출
                        summary = content[:200] + "..." if len(content) > 200 else content
                        response_parts.append(f"\n**{title}**\n{summary}")

            # 법률 조언 면책 조항 추가
            response_parts.append("\n\n※ 이 답변은 일반적인 법률 정보 제공을 목적으로 하며, 구체적인 법률 자문은 변호사와 상담하시기 바랍니다.")

            return "\n".join(response_parts)

        except Exception as e:
            self.logger.error(f"Response synthesis failed: {e}")
            return f"질문 '{query}'에 대한 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요."

    def _get_category_name(self, category: str) -> str:
        """카테고리 코드를 한국어 이름으로 변환"""
        category_names = {
            "labor_law": "노동법",
            "family_law": "가족법",
            "criminal_law": "형사법",
            "civil_law": "민사법",
            "property_law": "부동산법",
            "intellectual_property": "지적재산권법",
            "tax_law": "세법",
            "contract_review": "계약법",
            "civil_procedure": "민사소송법"
        }
        return category_names.get(category, category)

    async def _search_statutes(self, query: str) -> List[Dict]:
        """법령 검색 - 실제 법률 문서 활용"""
        try:
            # 법령 관련 키워드로 검색
            statute_keywords = ["법", "규정", "조항", "법령", "법률"]
            if any(keyword in query for keyword in statute_keywords):
                results = await self.legal_data_connector.search_legal_documents(query, ["civil", "criminal", "labor"])
                return [r for r in results if r.get("category") in ["civil_law", "criminal_law", "labor_law"]]
            return []
        except Exception as e:
            self.logger.error(f"Statute search failed: {e}")
            return []

    async def _search_precedents(self, query: str) -> List[Dict]:
        """판례 검색 - 실제 법률 문서 활용"""
        try:
            # 판례 관련 키워드로 검색
            precedent_keywords = ["판례", "재판", "법원", "소송", "사건"]
            if any(keyword in query for keyword in precedent_keywords):
                results = await self.legal_data_connector.search_legal_documents(query, ["civil", "criminal", "family"])
                return [r for r in results if r.get("category") in ["civil_law", "criminal_law", "family_law"]]
            return []
        except Exception as e:
            self.logger.error(f"Precedent search failed: {e}")
            return []

    async def _search_legal_articles(self, query: str) -> List[Dict]:
        """법률 논문 검색 - 실제 법률 문서 활용"""
        try:
            # 모든 법률 문서에서 검색
            results = await self.legal_data_connector.search_legal_documents(query)
            return results
        except Exception as e:
            self.logger.error(f"Legal article search failed: {e}")
            return []

    async def _analyze_statutes(self, statutes: List[Dict]) -> Dict[str, Any]:
        """법령 분석 - 실제 문서 내용 활용"""
        if not statutes:
            return {"statute_analysis": "관련 법령을 찾을 수 없습니다."}

        analysis_results = []
        for statute in statutes:
            content = statute.get("content", "")
            title = statute.get("title", "")
            analysis_results.append(f"• {title}: {content[:200]}...")

        return {
            "statute_analysis": f"{len(statutes)}개 법령 분석 완료",
            "statute_details": analysis_results,
            "statute_count": len(statutes)
        }

    async def _analyze_precedents(self, precedents: List[Dict]) -> Dict[str, Any]:
        """판례 분석 - 실제 문서 내용 활용"""
        if not precedents:
            return {"precedent_analysis": "관련 판례를 찾을 수 없습니다."}

        analysis_results = []
        for precedent in precedents:
            content = precedent.get("content", "")
            title = precedent.get("title", "")
            analysis_results.append(f"• {title}: {content[:200]}...")

        return {
            "precedent_analysis": f"{len(precedents)}개 판례 분석 완료",
            "precedent_details": analysis_results,
            "precedent_count": len(precedents)
        }

    async def _analyze_articles(self, articles: List[Dict]) -> Dict[str, Any]:
        """논문 분석 - 실제 문서 내용 활용"""
        if not articles:
            return {"article_analysis": "관련 법률 문서를 찾을 수 없습니다."}

        analysis_results = []
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            category = article.get("category", "")
            analysis_results.append(f"• [{category}] {title}: {content[:200]}...")

        return {
            "article_analysis": f"{len(articles)}개 법률 문서 분석 완료",
            "article_details": analysis_results,
            "article_count": len(articles)
        }

    def _calculate_relevance(self, response: str, query: str) -> float:
        """관련성 계산"""
        # 간단한 관련성 계산
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        common_words = query_words.intersection(response_words)

        if len(query_words) == 0:
            return 0.0

        return len(common_words) / len(query_words)

    def _calculate_completeness(self, response: str) -> float:
        """완성도 계산"""
        # 간단한 완성도 계산
        if len(response) < 50:
            return 0.3
        elif len(response) < 100:
            return 0.6
        else:
            return 0.9

    def _calculate_clarity(self, response: str) -> float:
        """명확성 계산"""
        # 간단한 명확성 계산
        clarity_score = 0.5

        # 문장 구조 확인
        sentences = response.split(".")
        if len(sentences) > 1:
            clarity_score += 0.2

        # 법률 키워드 포함 확인
        legal_keywords = ["법률", "법", "규정", "조항"]
        if any(keyword in response for keyword in legal_keywords):
            clarity_score += 0.3

        return min(clarity_score, 1.0)

    async def _improve_response(self, response: str, improvement_areas: List[str]) -> str:
        """응답 개선"""
        improved_response = response

        if "relevance" in improvement_areas:
            improved_response += "\n\n관련 법률 정보를 더 자세히 확인하시기 바랍니다."

        if "completeness" in improvement_areas:
            improved_response += "\n\n추가적인 정보가 필요하시면 더 구체적으로 질문해주세요."

        if "clarity" in improvement_areas:
            improved_response += "\n\n더 명확한 답변을 위해 전문가와 상담하시는 것을 권장합니다."

        return improved_response

    def _format_legal_response(self, response: str) -> str:
        """법률 응답 포맷팅"""
        # 기본 포맷팅
        formatted = response.strip()

        # 문단 구분 개선
        if "\n\n" not in formatted:
            sentences = formatted.split(".")
            if len(sentences) > 3:
                formatted = ".".join(sentences[:2]) + ".\n\n" + ".".join(sentences[2:])

        return formatted
