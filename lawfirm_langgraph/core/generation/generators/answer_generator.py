# -*- coding: utf-8 -*-
"""
Answer Generator
답변 생성 엔진 구현
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

try:
    from langchain_community.llms import OpenAI, Anthropic
    from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    LANCHAIN_AVAILABLE = True
except ImportError:
    LANCHAIN_AVAILABLE = False
    # Mock classes for when LangChain is not available
    class BaseMessage:
        def __init__(self, content):
            self.content = content
    
    class HumanMessage(BaseMessage):
        pass
    
    class SystemMessage(BaseMessage):
        pass
    
    class PromptTemplate:
        def __init__(self, *args, **kwargs):
            pass
        def format(self, **kwargs):
            return "Mock prompt"
    
    class LLMChain:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, **kwargs):
            return "Mock response"

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """답변 결과 데이터 클래스"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    response_time: float
    tokens_used: int
    model_used: str
    timestamp: datetime
    metadata: Dict[str, Any]


class AnswerGenerator:
    """답변 생성 엔진"""
    
    def __init__(self, config, langfuse_client=None, llm=None):
        """답변 생성기 초기화"""
        self.config = config
        self.langfuse_client = langfuse_client
        self.logger = logging.getLogger(__name__)
        
        # LLM 초기화: 전달된 LLM이 있으면 사용, 없으면 자체 초기화
        if llm is not None:
            self.llm = llm
            self.logger.info("Using provided LLM instance")
        else:
            self.llm = self._initialize_llm()
            if self.llm is None:
                self.logger.warning("LLM initialization failed. LLM will be required for answer generation.")
        
        # 프롬프트 템플릿 초기화
        self.prompt_templates = self._initialize_prompt_templates()
        
        # LLM 체인 초기화
        self.llm_chains = self._initialize_llm_chains()
        
        # 통계
        self.stats = {
            'total_queries': 0,
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0
        }
    
    def _initialize_llm(self):
        """LLM 초기화"""
        try:
            if self.config.llm_provider.value == "openai":
                if self.config.llm_model.startswith("gpt-3.5") or self.config.llm_model.startswith("gpt-4"):
                    return ChatOpenAI(
                        model_name=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens
                    )
                else:
                    return OpenAI(
                        model_name=self.config.llm_model,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_tokens
                    )
            
            elif self.config.llm_provider.value == "anthropic":
                return ChatAnthropic(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens
                )
            
            elif self.config.llm_provider.value == "google":
                return ChatGoogleGenerativeAI(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_tokens,
                    google_api_key=self.config.google_api_key
                )
            
            else:
                logger.error(f"Unsupported LLM provider: {self.config.llm_provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None
    
    def _initialize_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """프롬프트 템플릿 초기화"""
        templates = {}
        
        try:
            from core.shared.utils.langchain_config import PromptTemplates
            
            # 법률 Q&A 템플릿
            templates['legal_qa'] = PromptTemplate(
                input_variables=["context", "question"],
                template=PromptTemplates.LEGAL_QA_TEMPLATE
            )
            
            # 법률 분석 템플릿
            templates['legal_analysis'] = PromptTemplate(
                input_variables=["context", "question"],
                template=PromptTemplates.LEGAL_ANALYSIS_TEMPLATE
            )
            
            # 계약서 검토 템플릿
            templates['contract_review'] = PromptTemplate(
                input_variables=["context", "question"],
                template=PromptTemplates.CONTRACT_REVIEW_TEMPLATE
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize prompt templates: {e}")
        
        return templates
    
    def _initialize_llm_chains(self) -> Dict[str, LLMChain]:
        """LLM 체인 초기화"""
        chains = {}
        
        if not self.llm:
            return chains
        
        try:
            for template_name, template in self.prompt_templates.items():
                chains[template_name] = LLMChain(
                    llm=self.llm,
                    prompt=template,
                    verbose=True
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM chains: {e}")
        
        return chains
    
    def generate_answer(self, query: str, context: str, 
                       template_type: str = "legal_qa",
                       session_id: Optional[str] = None) -> AnswerResult:
        """답변 생성"""
        start_time = time.time()
        
        try:
            # 템플릿 타입 확인
            if template_type not in self.llm_chains:
                template_type = "legal_qa"
                logger.warning(f"Unknown template type: {template_type}, using legal_qa")
            
            # 프롬프트 생성
            if self.llm_chains and template_type in self.llm_chains:
                # LangChain 체인 사용
                chain = self.llm_chains[template_type]
                answer = chain.run(context=context, question=query)
            elif self.llm:
                # LLM 직접 사용
                if hasattr(self.llm, 'invoke'):
                    prompt = self.prompt_templates[template_type].format(context=context, question=query)
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        answer = response.content
                    elif isinstance(response, str):
                        answer = response
                    else:
                        answer = str(response)
                elif hasattr(self.llm, 'predict'):
                    prompt = self.prompt_templates[template_type].format(context=context, question=query)
                    answer = self.llm.predict(prompt)
                else:
                    # 기본 프롬프트 사용
                    answer = self._generate_basic_answer(query, context)
            else:
                # 기본 프롬프트 사용
                answer = self._generate_basic_answer(query, context)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 토큰 수 추정
            tokens_used = self._estimate_tokens(query, context, answer)
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(query, context, answer)
            
            # 소스 정보 추출
            sources = self._extract_sources(context)
            
            # 결과 생성
            result = AnswerResult(
                answer=answer,
                confidence=confidence,
                sources=sources,
                response_time=response_time,
                tokens_used=tokens_used,
                model_used=self.config.llm_model,
                timestamp=datetime.now(),
                metadata={
                    'template_type': template_type,
                    'session_id': session_id,
                    'context_length': len(context),
                    'query_length': len(query)
                }
            )
            
            # 통계 업데이트
            self._update_stats(result)
            
            # Langfuse 추적
            if self.langfuse_client and self.langfuse_client.is_enabled():
                self.langfuse_client.track_llm_call(
                    model=self.config.llm_model,
                    prompt=f"Query: {query}\nContext: {context[:200]}...",
                    response=answer,
                    tokens_used=tokens_used,
                    response_time=response_time
                )
            
            self.logger.info(f"Generated answer in {response_time:.2f}s with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            # 오류 추적
            if self.langfuse_client and self.langfuse_client.is_enabled():
                self.langfuse_client.track_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    context={"query": query, "template_type": template_type}
                )
            
            logger.error(f"Failed to generate answer: {e}")
            
            # 오류 응답 반환
            return AnswerResult(
                answer="죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.",
                confidence=0.0,
                sources=[],
                response_time=response_time,
                tokens_used=0,
                model_used=self.config.llm_model,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _generate_basic_answer(self, query: str, context: str) -> str:
        """기본 답변 생성 (LangChain 없이)"""
        # 간단한 키워드 기반 답변 생성
        query_words = set(query.lower().split())
        context_sentences = context.split('.')
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        # 관련성 순으로 정렬
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 문장들로 답변 구성
        if relevant_sentences:
            answer_parts = []
            for sentence, _ in relevant_sentences[:3]:  # 상위 3개 문장
                if sentence.strip():
                    answer_parts.append(sentence.strip())
            
            answer = '. '.join(answer_parts)
            if not answer.endswith('.'):
                answer += '.'
            
            return f"주어진 문서를 바탕으로 답변드리면:\n\n{answer}"
        else:
            # 문서가 없거나 답변이 비어있는 경우, 일반적인 법률 지식으로 답변
            # "관련 정보를 찾을 수 없습니다" 같은 회피적 답변은 피하고, 일반적인 법률 지식으로 답변
            return "일반적인 법률 지식을 바탕으로 답변드리면, 해당 질문에 대한 법적 원칙과 일반적인 해석을 제공할 수 있습니다. 다만 구체적인 사안에 대한 정확한 답변을 위해서는 관련 법령과 판례를 확인하는 것이 필요합니다."
    
    def _estimate_tokens(self, query: str, context: str, answer: str) -> int:
        """토큰 수 추정"""
        # 간단한 추정: 공백으로 분할된 단어 수 * 1.3
        total_text = f"{query} {context} {answer}"
        word_count = len(total_text.split())
        return int(word_count * 1.3)
    
    def _calculate_confidence(self, query: str, context: str, answer: str) -> float:
        """신뢰도 계산"""
        try:
            # 기본 신뢰도
            confidence = 0.5
            
            # 컨텍스트 길이 기반 보정
            if len(context) > 500:
                confidence += 0.2
            elif len(context) > 200:
                confidence += 0.1
            
            # 답변 길이 기반 보정
            if len(answer) > 100:
                confidence += 0.1
            
            # 키워드 매칭 기반 보정
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            keyword_overlap = len(query_words.intersection(answer_words))
            if keyword_overlap > 0:
                confidence += min(0.2, keyword_overlap * 0.05)
            
            # 최대 1.0으로 제한
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _extract_sources(self, context: str) -> List[Dict[str, Any]]:
        """컨텍스트에서 소스 정보 추출"""
        sources = []
        
        try:
            # 문서 구분자로 분할
            context_parts = context.split('[문서:')
            
            for part in context_parts[1:]:  # 첫 번째는 제외
                if ']' in part:
                    doc_name = part.split(']')[0].strip()
                    content = part.split(']')[1].strip()
                    
                    sources.append({
                        'title': doc_name,
                        'content_preview': content[:200] + '...' if len(content) > 200 else content,
                        'relevance_score': 0.8  # 기본값
                    })
        
        except Exception as e:
            logger.error(f"Failed to extract sources: {e}")
        
        return sources
    
    def _update_stats(self, result: AnswerResult):
        """통계 업데이트"""
        self.stats['total_queries'] += 1
        self.stats['total_tokens'] += result.tokens_used
        
        # 평균 응답 시간 업데이트
        if self.stats['total_queries'] == 1:
            self.stats['avg_response_time'] = result.response_time
            self.stats['avg_confidence'] = result.confidence
        else:
            # 이동 평균 계산
            alpha = 0.1  # 학습률
            self.stats['avg_response_time'] = (
                (1 - alpha) * self.stats['avg_response_time'] + 
                alpha * result.response_time
            )
            self.stats['avg_confidence'] = (
                (1 - alpha) * self.stats['avg_confidence'] + 
                alpha * result.confidence
            )
    
    def generate_batch_answers(self, queries: List[Dict[str, Any]]) -> List[AnswerResult]:
        """배치 답변 생성"""
        results = []
        
        for i, query_data in enumerate(queries):
            try:
                query = query_data.get('query', '')
                context = query_data.get('context', '')
                template_type = query_data.get('template_type', 'legal_qa')
                session_id = query_data.get('session_id')
                
                result = self.generate_answer(
                    query=query,
                    context=context,
                    template_type=template_type,
                    session_id=session_id
                )
                
                results.append(result)
                
                self.logger.info(f"Processed query {i+1}/{len(queries)}")
                
            except Exception as e:
                logger.error(f"Failed to process query {i+1}: {e}")
                
                # 오류 결과 추가
                error_result = AnswerResult(
                    answer="처리 중 오류가 발생했습니다.",
                    confidence=0.0,
                    sources=[],
                    response_time=0.0,
                    tokens_used=0,
                    model_used=self.config.llm_model,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """생성기 통계 반환"""
        return {
            "total_queries": self.stats['total_queries'],
            "total_tokens": self.stats['total_tokens'],
            "avg_response_time": self.stats['avg_response_time'],
            "avg_confidence": self.stats['avg_confidence'],
            "model_used": self.config.llm_model,
            "llm_available": self.llm is not None,
            "templates_available": list(self.prompt_templates.keys()),
            "chains_available": list(self.llm_chains.keys())
        }
    
    def validate_answer_quality(self, result: AnswerResult) -> Dict[str, Any]:
        """답변 품질 검증"""
        quality_metrics = {
            'length_score': 0.0,
            'relevance_score': 0.0,
            'completeness_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            answer = result.answer
            
            # 길이 점수 (100-500자 범위가 이상적)
            answer_length = len(answer)
            if 100 <= answer_length <= 500:
                quality_metrics['length_score'] = 1.0
            elif answer_length < 100:
                quality_metrics['length_score'] = answer_length / 100
            else:
                quality_metrics['length_score'] = max(0.0, 1.0 - (answer_length - 500) / 500)
            
            # 관련성 점수 (신뢰도 기반)
            quality_metrics['relevance_score'] = result.confidence
            
            # 완성도 점수 (문장 구조, 구두점 등)
            sentences = answer.split('.')
            if len(sentences) >= 2:
                quality_metrics['completeness_score'] = 1.0
            else:
                quality_metrics['completeness_score'] = 0.5
            
            # 전체 점수 계산
            quality_metrics['overall_score'] = (
                quality_metrics['length_score'] * 0.3 +
                quality_metrics['relevance_score'] * 0.5 +
                quality_metrics['completeness_score'] * 0.2
            )
            
        except Exception as e:
            logger.error(f"Failed to validate answer quality: {e}")
        
        return quality_metrics
    
    def generate_fallback_answer(self, state: Any) -> str:
        """
        폴백 답변 생성 (오류 발생 시)
        
        Args:
            state: LegalWorkflowState 객체 또는 dict
            
        Returns:
            str: 폴백 답변 메시지
        """
        try:
            # state에서 query 추출 시도
            query = ""
            if isinstance(state, dict):
                query = state.get("query", state.get("question", ""))
            elif hasattr(state, "query"):
                query = state.query
            elif hasattr(state, "question"):
                query = state.question
            
            if query:
                return f"죄송합니다. '{query}'에 대한 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            else:
                return "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        except Exception as e:
            self.logger.error(f"Error generating fallback answer: {e}")
            return "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def assess_improvement_potential(
        self,
        quality_score: float,
        quality_checks: Dict[str, bool],
        state: Any
    ) -> Dict[str, Any]:
        """
        답변 개선 가능성 평가
        
        Args:
            quality_score: 현재 품질 점수
            quality_checks: 품질 체크 결과 딕셔너리
            state: LegalWorkflowState 객체 또는 dict
            
        Returns:
            Dict[str, Any]: 개선 가능성 평가 결과
        """
        try:
            improvement_potential = {
                "has_potential": quality_score < 0.7,
                "improvement_score": max(0.0, 1.0 - quality_score),
                "priority_areas": [],
                "recommendations": []
            }
            
            # 품질 체크 결과 기반으로 개선 영역 식별
            if isinstance(quality_checks, dict):
                if not quality_checks.get("legal_accuracy", True):
                    improvement_potential["priority_areas"].append("legal_accuracy")
                    improvement_potential["recommendations"].append("법률 정확성 향상 필요")
                
                if not quality_checks.get("completeness", True):
                    improvement_potential["priority_areas"].append("completeness")
                    improvement_potential["recommendations"].append("답변 완성도 향상 필요")
                
                if not quality_checks.get("clarity", True):
                    improvement_potential["priority_areas"].append("clarity")
                    improvement_potential["recommendations"].append("답변 명확성 향상 필요")
            
            return improvement_potential
        except Exception as e:
            self.logger.error(f"Error assessing improvement potential: {e}")
            return {
                "has_potential": False,
                "improvement_score": 0.0,
                "priority_areas": [],
                "recommendations": []
            }
    
    def generate_answer_with_chain(
        self,
        optimized_prompt: str,
        query: str,
        context_dict: Dict[str, Any],
        quality_feedback: Optional[Dict[str, Any]] = None,
        is_retry: bool = False
    ) -> str:
        """
        체인을 사용한 답변 생성 (스트리밍 지원)
        
        LangGraph는 노드 내에서 stream() 또는 astream()을 호출하면 자동으로 on_llm_stream 이벤트를 발생시킵니다.
        
        Args:
            optimized_prompt: 최적화된 프롬프트
            query: 사용자 질문
            context_dict: 컨텍스트 딕셔너리
            quality_feedback: 품질 피드백 (선택적)
            is_retry: 재시도 여부
            
        Returns:
            str: 생성된 답변
        """
        try:
            # context_dict에서 context 추출
            context = context_dict.get("context", "") if isinstance(context_dict, dict) else str(context_dict)
            
            # LLM이 없으면 에러 발생
            if not self.llm:
                error_msg = "LLM이 초기화되지 않았습니다. LLM 설정을 확인해주세요."
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 스트리밍으로 답변 생성
            # stream() 우선 사용 - LangGraph가 on_llm_stream 이벤트 발생
            try:
                # stream() 사용 (동기 스트리밍) - LangGraph가 on_llm_stream 이벤트 발생
                if hasattr(self.llm, 'stream'):
                    try:
                        full_answer = ""
                        for chunk in self.llm.stream(optimized_prompt):
                            if hasattr(chunk, 'content'):
                                full_answer += chunk.content
                            elif isinstance(chunk, str):
                                full_answer += chunk
                            else:
                                full_answer += str(chunk)
                        self.logger.debug("stream() 사용 성공 - on_llm_stream 이벤트 발생 예상")
                        return full_answer
                    except Exception as stream_error:
                        # stream() 실패 시 로그만 남기고 invoke()로 폴백
                        self.logger.warning(f"stream() 호출 실패, invoke()로 폴백: {stream_error}")
                        # 아래 invoke() 폴백 로직으로 계속 진행
                
                # invoke() 폴백 (stream()이 없거나 실패한 경우)
                if hasattr(self.llm, 'invoke'):
                    self.logger.debug("invoke() 사용 - on_chain_stream 이벤트만 발생 예상")
                    response = self.llm.invoke(optimized_prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    elif isinstance(response, str):
                        return response
                    else:
                        return str(response)
                elif hasattr(self.llm, 'predict'):
                    return self.llm.predict(optimized_prompt)
                else:
                    error_msg = f"LLM 객체가 스트리밍을 지원하지 않습니다. LLM 타입: {type(self.llm)}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            except Exception as e:
                error_msg = f"LLM 호출 실패: {e}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"generate_answer_with_chain 실행 중 오류 발생: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def validate_answer_uses_context(
        self,
        answer: str,
        context: Dict[str, Any],
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        답변이 컨텍스트를 사용하는지 검증
        
        Args:
            answer: 답변 텍스트
            context: 컨텍스트 딕셔너리
            query: 질문
            retrieved_docs: 검색된 문서 목록 (선택적)
            
        Returns:
            검증 결과 딕셔너리
        """
        try:
            # AnswerValidator 사용
            from core.generation.validators.quality_validators import AnswerValidator
            return AnswerValidator.validate_answer_uses_context(
                answer=answer,
                context=context,
                query=query,
                retrieved_docs=retrieved_docs
            )
        except ImportError:
            # AnswerValidator를 import할 수 없는 경우 기본 검증 수행
            self.logger.warning("AnswerValidator not available, using basic validation")
            return {
                "uses_context": True,
                "coverage_score": 0.5,
                "citation_count": 0,
                "has_document_references": False,
                "needs_regeneration": False,
                "missing_key_info": []
            }
        except Exception as e:
            self.logger.error(f"Error in validate_answer_uses_context: {e}")
            # 기본 검증 결과 반환
            return {
                "uses_context": True,
                "coverage_score": 0.5,
                "citation_count": 0,
                "has_document_references": False,
                "needs_regeneration": False,
                "missing_key_info": []
            }
    
    def track_search_to_answer_pipeline(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        검색에서 답변 생성까지의 파이프라인 추적
        
        Args:
            state: 워크플로우 상태
            
        Returns:
            파이프라인 추적 정보
        """
        try:
            # 검색 결과 추출
            retrieved_docs = state.get("retrieved_docs", [])
            semantic_results = state.get("semantic_results", [])
            keyword_results = state.get("keyword_results", [])
            
            # 답변 정보 추출
            answer = state.get("answer", "")
            if isinstance(answer, dict):
                answer_text = answer.get("content", answer.get("text", str(answer)))
            else:
                answer_text = str(answer)
            
            # 파이프라인 메트릭 계산
            pipeline_metrics = {
                "retrieved_docs_count": len(retrieved_docs),
                "semantic_results_count": len(semantic_results),
                "keyword_results_count": len(keyword_results),
                "answer_length": len(answer_text) if answer_text else 0,
                "has_answer": bool(answer_text),
                "pipeline_complete": bool(answer_text and retrieved_docs)
            }
            
            return pipeline_metrics
        except Exception as e:
            self.logger.error(f"Error in track_search_to_answer_pipeline: {e}")
            return {
                "retrieved_docs_count": 0,
                "semantic_results_count": 0,
                "keyword_results_count": 0,
                "answer_length": 0,
                "has_answer": False,
                "pipeline_complete": False,
                "error": str(e)
            }