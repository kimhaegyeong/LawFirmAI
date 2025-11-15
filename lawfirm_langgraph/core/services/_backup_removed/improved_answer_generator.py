# -*- coding: utf-8 -*-
"""
개선된 답변 생성기
Ollama 클라이언트와 프롬프트 템플릿을 활용한 지능형 답변 생성
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .gemini_client import GeminiClient, GeminiResponse
from .prompt_templates import PromptTemplateManager
from .unified_prompt_manager import UnifiedPromptManager, LegalDomain, ModelType
from .prompt_optimizer import PromptOptimizer, PromptPerformanceMetrics
from .semantic_domain_classifier import SemanticDomainClassifier
from .confidence_calculator import ConfidenceCalculator, ConfidenceInfo, ConfidenceLevel
from .question_classifier import QuestionType, QuestionClassification
from .answer_formatter import AnswerFormatter, FormattedAnswer
from .context_builder import ContextBuilder, ContextWindow

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """답변 생성 결과"""
    answer: str
    formatted_answer: Optional[FormattedAnswer]
    raw_answer: str
    confidence: ConfidenceInfo
    question_type: QuestionType
    processing_time: float
    tokens_used: int
    model_info: Dict[str, Any]


class ImprovedAnswerGenerator:
    """개선된 답변 생성기"""
    
    def __init__(self, 
                 gemini_client: Optional[GeminiClient] = None,
                 prompt_template_manager: Optional[PromptTemplateManager] = None,
                 confidence_calculator: Optional[ConfidenceCalculator] = None,
                 answer_formatter: Optional[AnswerFormatter] = None,
                 context_builder: Optional[ContextBuilder] = None,
                 unified_prompt_manager: Optional[UnifiedPromptManager] = None,
                 prompt_optimizer: Optional[PromptOptimizer] = None):
        """답변 생성기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.gemini_client = gemini_client or GeminiClient()
        self.prompt_template_manager = prompt_template_manager or PromptTemplateManager()
        self.confidence_calculator = confidence_calculator or ConfidenceCalculator()
        self.answer_formatter = answer_formatter or AnswerFormatter()
        self.context_builder = context_builder or ContextBuilder()
        self.unified_prompt_manager = unified_prompt_manager or UnifiedPromptManager()
        self.prompt_optimizer = prompt_optimizer or PromptOptimizer(self.unified_prompt_manager)
        self.semantic_domain_classifier = SemanticDomainClassifier()
        
        # 답변 생성 설정
        self.generation_config = {
            "max_tokens": 2048,
            "temperature": 0.7,
            "retry_count": 3,
            "retry_delay": 1.0
        }
        
        # 질문 유형별 특별 설정
        self.question_type_configs = {
            QuestionType.PRECEDENT_SEARCH: {
                "temperature": 0.6,  # 더 정확한 답변
                "max_tokens": 1500
            },
            QuestionType.LAW_INQUIRY: {
                "temperature": 0.5,  # 매우 정확한 답변
                "max_tokens": 1200
            },
            QuestionType.LEGAL_ADVICE: {
                "temperature": 0.7,  # 균형잡힌 답변
                "max_tokens": 2000
            },
            QuestionType.PROCEDURE_GUIDE: {
                "temperature": 0.6,  # 구조화된 답변
                "max_tokens": 1800
            },
            QuestionType.TERM_EXPLANATION: {
                "temperature": 0.5,  # 정확한 설명
                "max_tokens": 1000
            },
            QuestionType.GENERAL_QUESTION: {
                "temperature": 0.7,  # 일반적인 답변
                "max_tokens": 1500
            }
        }
    
    def generate_answer(self, 
                       query: str,
                       question_type: QuestionClassification,
                       context: str,
                       sources: Dict[str, List[Dict[str, Any]]],
                       conversation_history: Optional[List[Dict[str, Any]]] = None) -> AnswerResult:
        """
        답변 생성
        
        Args:
            query: 사용자 질문
            question_type: 질문 분류 결과
            context: 컨텍스트 정보
            sources: 검색된 소스들
            conversation_history: 대화 이력
            
        Returns:
            AnswerResult: 생성된 답변 결과
        """
        try:
            start_time = time.time()
            self.logger.info(f"Generating answer for query: {query[:100]}...")
            
            # 질문 유형별 설정 적용
            config = self.question_type_configs.get(
                question_type.question_type, 
                self.generation_config
            )
            
            # 컨텍스트 윈도우 최적화
            context_window = self.context_builder.build_optimized_context(
                query=query,
                question_classification=question_type,
                search_results=sources,
                conversation_history=conversation_history
            )
            
            # 최적화된 컨텍스트로 프롬프트 생성
            optimized_context = self.context_builder.format_context_for_llm(context_window)
            
            # 의미 기반 도메인 분류 사용
            domain, domain_confidence, domain_reasoning = self.semantic_domain_classifier.classify_domain(
                query, question_type.question_type
            )
            model_type = ModelType.GEMINI  # 기본값
            
            # 도메인 분류 결과 로깅
            self.logger.info(f"Domain classification: {domain.value} (confidence: {domain_confidence:.2f}) - {domain_reasoning}")
            
            prompt = self.unified_prompt_manager.get_optimized_prompt(
                query=query,
                question_type=question_type.question_type,
                domain=domain,
                context=sources,
                model_type=model_type
            )
            
            # Ollama로 답변 생성 (재시도 로직 포함)
            raw_answer = self._generate_with_retry(prompt, config)
            
            # 신뢰도 계산
            confidence = self.confidence_calculator.calculate_confidence(
                answer=raw_answer,
                sources=sources.get("results", []),
                question_type=question_type.question_type.value if hasattr(question_type, 'question_type') else "general"
            )
            
            # 답변 후처리
            processed_answer = self._post_process_answer(raw_answer, question_type, sources)
            
            # 답변 구조화
            formatted_answer = self.answer_formatter.format_answer(
                raw_answer=raw_answer,
                question_type=question_type.question_type,
                sources=sources,
                confidence=confidence
            )
            
            processing_time = time.time() - start_time
            
            result = AnswerResult(
                answer=processed_answer,
                formatted_answer=formatted_answer,
                raw_answer=raw_answer,
                confidence=confidence,
                question_type=question_type.question_type,
                processing_time=processing_time,
                tokens_used=self._estimate_tokens(raw_answer),
                model_info={"model": self.gemini_client.model_name}
            )
            
            # 성능 메트릭 기록
            self._record_performance_metrics(
                query=query,
                question_type=question_type,
                domain=domain,
                model_type=model_type,
                response_time=processing_time,
                answer_quality=confidence.confidence,
                context_length=len(str(sources)),
                token_count=self._estimate_tokens(raw_answer),
                domain_confidence=domain_confidence
            )
            
            self.logger.info(f"Answer generated successfully: {len(processed_answer)} chars, "
                           f"confidence: {confidence.confidence:.3f}, "
                           f"time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            # 오류 시 기본 답변 반환
            return self._create_fallback_answer(query, question_type, e)
    
    
    def _record_performance_metrics(self, query: str, question_type: QuestionClassification, 
                                  domain: LegalDomain, model_type: ModelType,
                                  response_time: float, answer_quality: float,
                                  context_length: int, token_count: int,
                                  domain_confidence: float = 0.0) -> None:
        """성능 메트릭 기록 (도메인 분류 신뢰도 포함)"""
        try:
            # 고유 프롬프트 ID 생성
            prompt_id = f"{domain.value}_{question_type.question_type.value}_{hash(query) % 10000}"
            
            # 성능 메트릭 생성
            metrics = PromptPerformanceMetrics(
                prompt_id=prompt_id,
                model_type=model_type,
                domain=domain,
                question_type=question_type.question_type,
                response_time=response_time,
                token_count=token_count,
                context_length=context_length,
                answer_quality_score=answer_quality
            )
            
            # 도메인 분류 신뢰도 추가 (메타데이터로)
            if hasattr(metrics, 'metadata'):
                metrics.metadata = {
                    "domain_confidence": domain_confidence,
                    "classification_method": "semantic"
                }
            
            # 성능 메트릭 기록
            self.prompt_optimizer.record_performance(metrics)
            
        except Exception as e:
            self.logger.error(f"Error recording performance metrics: {e}")
    
    def _build_enhanced_prompt(self, 
                              query: str,
                              question_type: QuestionClassification,
                              context: str,
                              sources: Dict[str, List[Dict[str, Any]]]) -> str:
        """향상된 프롬프트 구성"""
        try:
            # 컨텍스트 데이터 준비
            context_data = {
                "precedent_list": sources.get("precedent_results", []),
                "law_articles": sources.get("law_results", []),
                "context": sources.get("results", [])
            }
            
            # 질문 유형별 프롬프트 템플릿 사용
            prompt = self.prompt_template_manager.format_prompt(
                question_type=question_type.question_type,
                context_data=context_data,
                user_query=query
            )
            
            # 추가 컨텍스트가 있으면 포함
            if context:
                prompt += f"\n\n추가 컨텍스트:\n{context}"
            
            # 질문 유형별 특별 지시사항 추가
            special_instructions = self._get_special_instructions(question_type.question_type)
            if special_instructions:
                prompt += f"\n\n{special_instructions}"
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building enhanced prompt: {e}")
            return f"질문: {query}\n\n위 질문에 대해 전문적이고 정확한 답변을 작성하세요."
    
    def _get_special_instructions(self, question_type: QuestionType) -> str:
        """질문 유형별 특별 지시사항"""
        instructions = {
            QuestionType.PRECEDENT_SEARCH: """
특별 지시사항:
- 판례의 핵심 판결요지를 명확히 제시하세요
- 사건번호와 법원 정보를 정확히 인용하세요
- 해당 판례의 실무적 시사점을 설명하세요""",
            
            QuestionType.LAW_INQUIRY: """
특별 지시사항:
- 법률 조문을 정확히 인용하세요
- 법률의 목적과 취지를 설명하세요
- 실제 적용 사례를 포함하세요""",
            
            QuestionType.LEGAL_ADVICE: """
특별 지시사항:
- 단계별 해결 방법을 제시하세요
- 필요한 증거 자료를 명시하세요
- 전문가 상담의 필요성을 언급하세요""",
            
            QuestionType.PROCEDURE_GUIDE: """
특별 지시사항:
- 절차를 순서대로 설명하세요
- 필요한 서류와 비용을 명시하세요
- 처리 기간을 포함하세요""",
            
            QuestionType.TERM_EXPLANATION: """
특별 지시사항:
- 용어의 정확한 정의를 제시하세요
- 관련 법률 조문을 인용하세요
- 실제 적용 예시를 포함하세요"""
        }
        
        return instructions.get(question_type, "")
    
    def _generate_with_retry(self, prompt: str, config: Dict[str, Any]) -> str:
        """재시도 로직을 포함한 답변 생성"""
        try:
            for attempt in range(self.generation_config["retry_count"]):
                try:
                    response = self.gemini_client.generate(
                        prompt=prompt,
                        temperature=config.get("temperature", self.generation_config["temperature"]),
                        max_tokens=config.get("max_tokens", self.generation_config["max_tokens"])
                    )
                    
                    if response.response and len(response.response.strip()) > 0:
                        return response.response
                    else:
                        self.logger.warning(f"Empty response on attempt {attempt + 1}")
                        
                except Exception as e:
                    self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    
                if attempt < self.generation_config["retry_count"] - 1:
                    time.sleep(self.generation_config["retry_delay"])
            
            # 모든 재시도 실패 시 기본 답변
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
            
        except Exception as e:
            self.logger.error(f"All generation attempts failed: {e}")
            return "답변 생성에 실패했습니다. 잠시 후 다시 시도해주세요."
    
    def _post_process_answer(self, 
                             raw_answer: str, 
                             question_type: QuestionClassification,
                             sources: Dict[str, List[Dict[str, Any]]]) -> str:
        """답변 후처리"""
        try:
            # 기본 정리
            processed = raw_answer.strip()
            
            # 질문 유형별 후처리
            if question_type.question_type == QuestionType.PRECEDENT_SEARCH:
                processed = self._enhance_precedent_answer(processed, sources)
            elif question_type.question_type == QuestionType.LAW_INQUIRY:
                processed = self._enhance_law_answer(processed, sources)
            elif question_type.question_type == QuestionType.LEGAL_ADVICE:
                processed = self._enhance_advice_answer(processed, sources)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in post-processing: {e}")
            return raw_answer
    
    def _enhance_precedent_answer(self, answer: str, sources: Dict[str, List[Dict[str, Any]]]) -> str:
        """판례 답변 향상"""
        try:
            precedents = sources.get("precedent_results", [])
            if precedents:
                answer += f"\n\n**참고 판례 ({len(precedents)}개):**\n"
                for i, prec in enumerate(precedents[:3], 1):
                    answer += f"{i}. {prec.get('case_name', '')} ({prec.get('case_number', '')})\n"
            return answer
        except Exception as e:
            self.logger.error(f"Error enhancing precedent answer: {e}")
            return answer
    
    def _enhance_law_answer(self, answer: str, sources: Dict[str, List[Dict[str, Any]]]) -> str:
        """법률 답변 향상"""
        try:
            laws = sources.get("law_results", [])
            if laws:
                answer += f"\n\n**관련 법률 ({len(laws)}개):**\n"
                for i, law in enumerate(laws[:3], 1):
                    answer += f"{i}. {law.get('law_name', '')} {law.get('article_number', '')}\n"
            return answer
        except Exception as e:
            self.logger.error(f"Error enhancing law answer: {e}")
            return answer
    
    def _enhance_advice_answer(self, answer: str, sources: Dict[str, List[Dict[str, Any]]]) -> str:
        """조언 답변 향상"""
        try:
            laws = sources.get("law_results", [])
            precedents = sources.get("precedent_results", [])
            
            if laws or precedents:
                answer += f"\n\n**참고 자료:**\n"
                if laws:
                    answer += f"- 관련 법률: {len(laws)}개\n"
                if precedents:
                    answer += f"- 관련 판례: {len(precedents)}개\n"
            
            return answer
        except Exception as e:
            self.logger.error(f"Error enhancing advice answer: {e}")
            return answer
    
    def _add_disclaimer(self, answer: str) -> str:
        """면책 조항 추가"""
        disclaimer = """

---
💼 **면책 조항**
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다.
구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""
        
        return answer + disclaimer
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (대략적)"""
        # 한국어 기준 대략적인 토큰 수 추정
        return len(text) // 2
    
    def _create_fallback_answer(self, 
                               query: str, 
                               question_type: QuestionClassification,
                               error: Exception) -> AnswerResult:
        """오류 시 기본 답변 생성"""
        try:
            fallback_answer = f"""죄송합니다. '{query}'에 대한 답변 생성 중 오류가 발생했습니다.

오류 내용: {str(error)}

다음과 같이 다시 시도해보시기 바랍니다:
1. 질문을 더 구체적으로 작성해주세요
2. 잠시 후 다시 시도해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

전문적인 법률 상담이 필요하시면 변호사와 직접 상담하시기 바랍니다."""
            
            return AnswerResult(
                answer=fallback_answer,
                formatted_answer=None,
                raw_answer=fallback_answer,
                confidence=self.confidence_calculator.calculate_confidence(
                    answer=fallback_answer,
                    sources=[],
                    question_type=question_type.question_type.value if hasattr(question_type, 'question_type') else "general"
                ),
                question_type=question_type.question_type,
                processing_time=0.0,
                tokens_used=self._estimate_tokens(fallback_answer),
                model_info={"model": "fallback"}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating fallback answer: {e}")
            # 최후의 수단
            return AnswerResult(
                answer="답변 생성에 실패했습니다. 다시 시도해주세요.",
                formatted_answer=None,
                raw_answer="답변 생성에 실패했습니다. 다시 시도해주세요.",
                confidence=ConfidenceInfo(
                    confidence=0.0,
                    level=ConfidenceLevel.VERY_LOW,
                    factors={"error": 1.0},
                    explanation="답변 생성 실패"
                ),
                question_type=QuestionType.GENERAL_QUESTION,
                processing_time=0.0,
                tokens_used=0,
                model_info={"model": "error"}
            )


# 테스트 함수
def test_improved_answer_generator():
    """개선된 답변 생성기 테스트"""
    generator = ImprovedAnswerGenerator()
    
    # 테스트 데이터
    test_query = "손해배상 청구 방법"
    test_question_type = QuestionClassification(
        question_type=QuestionType.LEGAL_ADVICE,
        law_weight=0.5,
        precedent_weight=0.5,
        confidence=0.8,
        keywords=["손해배상", "청구"],
        patterns=[]
    )
    test_context = "민법 제750조 불법행위 관련 질문"
    test_sources = {
        "results": [
            {"type": "law", "law_name": "민법", "article_number": "제750조", "similarity": 0.9},
            {"type": "precedent", "case_name": "손해배상 사건", "case_number": "2023다12345", "similarity": 0.8}
        ],
        "law_results": [
            {"law_name": "민법", "article_number": "제750조", "content": "불법행위로 인한 손해배상"}
        ],
        "precedent_results": [
            {"case_name": "손해배상 사건", "case_number": "2023다12345", "summary": "불법행위 손해배상"}
        ]
    }
    
    print("=== 개선된 답변 생성기 테스트 ===")
    print(f"질문: {test_query}")
    print(f"질문 유형: {test_question_type.question_type.value}")
    
    try:
        result = generator.generate_answer(
            query=test_query,
            question_type=test_question_type,
            context=test_context,
            sources=test_sources
        )
        
        print(f"\n답변 결과:")
        print(f"- 답변 길이: {len(result.answer)} 문자")
        print(f"- 구조화된 답변 길이: {len(result.formatted_answer.formatted_content) if result.formatted_answer else 0} 문자")
        print(f"- 신뢰도: {result.confidence.confidence:.3f}")
        print(f"- 신뢰도 수준: {result.confidence.reliability_level.value}")
        print(f"- 처리 시간: {result.processing_time:.2f}초")
        print(f"- 추정 토큰 수: {result.tokens_used}")
        print(f"- 모델: {result.model_info['model']}")
        
        print(f"\n답변 내용:")
        print(result.answer[:500] + "..." if len(result.answer) > 500 else result.answer)
        
        if result.formatted_answer:
            print(f"\n구조화된 답변 미리보기:")
            print(result.formatted_answer.formatted_content[:500] + "..." if len(result.formatted_answer.formatted_content) > 500 else result.formatted_answer.formatted_content)
        
    except Exception as e:
        print(f"테스트 실패: {e}")


if __name__ == "__main__":
    test_improved_answer_generator()
