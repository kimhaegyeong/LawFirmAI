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
    
    def __init__(self, config, langfuse_client=None):
        """답변 생성기 초기화"""
        self.config = config
        self.langfuse_client = langfuse_client
        self.logger = logging.getLogger(__name__)
        
        # LLM 초기화
        self.llm = self._initialize_llm()
        
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
        if not LANCHAIN_AVAILABLE:
            logger.warning("LangChain is not available. Using mock LLM.")
            return None
        
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
        
        if not LANCHAIN_AVAILABLE:
            return templates
        
        try:
            from utils.langchain_config import PromptTemplates
            
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
        
        if not self.llm or not LANCHAIN_AVAILABLE:
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
            if LANCHAIN_AVAILABLE and self.llm_chains:
                # LangChain 체인 사용
                chain = self.llm_chains[template_type]
                answer = chain.run(context=context, question=query)
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
            return "주어진 문서에서 관련 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다."
    
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
