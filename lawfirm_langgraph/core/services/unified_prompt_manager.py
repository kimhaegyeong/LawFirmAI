# -*- coding: utf-8 -*-
"""
통합 프롬프트 관리 시스템
법률 도메인 특화 프롬프트의 통합 관리 및 최적화
"""

import json
import os
import re
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# QuestionType import
try:
    from core.classification.classifiers.question_classifier import QuestionType
except ImportError:
    try:
        # 호환성을 위한 fallback (더 이상 services에 없음)
        from core.classification.classifiers.question_classifier import QuestionType
    except ImportError:
        try:
            from ..classification.classifiers.question_classifier import QuestionType
        except ImportError:
            # 최종 fallback: enum 직접 정의
            from enum import Enum
            class QuestionType(Enum):
                GENERAL_QUESTION = "general_question"
                LAW_INQUIRY = "law_inquiry"
                PRECEDENT_SEARCH = "precedent_search"
                DOCUMENT_ANALYSIS = "document_analysis"
                LEGAL_ADVICE = "legal_advice"

logger = get_logger(__name__)


class LegalDomain(Enum):
    """법률 도메인 분류"""
    CIVIL_LAW = "민사법"
    CRIMINAL_LAW = "형사법"
    FAMILY_LAW = "가족법"
    COMMERCIAL_LAW = "상사법"
    ADMINISTRATIVE_LAW = "행정법"
    LABOR_LAW = "노동법"
    PROPERTY_LAW = "부동산법"
    INTELLECTUAL_PROPERTY = "지적재산권법"
    TAX_LAW = "세법"
    CIVIL_PROCEDURE = "민사소송법"
    CRIMINAL_PROCEDURE = "형사소송법"
    GENERAL = "기타/일반"


class ModelType(Enum):
    """지원 모델 타입"""
    GEMINI = "gemini"
    OPENAI = "openai"


class UnifiedPromptManager:
    """통합 프롬프트 관리 시스템"""
    
    # 상수 정의
    MAX_INPUT_TOKENS = 1_048_576  # Gemini 2.5 Flash 최대 입력 토큰
    SAFE_MARGIN = 0.1  # 10% 안전 마진
    MAX_DOCUMENTS = 8  # 최대 문서 수
    MAX_LAW_DOCS = 3  # 최대 법률 조문 수
    MIN_CONTENT_LENGTH = 200  # TASK 3: 최소 content 길이 (10 → 200)
    MIN_CONTENT_LENGTH_WITH_LAW_INFO = 200  # TASK 3: 법률 정보가 있을 때 최소 content 길이 (3 → 200)
    MAX_DOC_LENGTH_LAW = 1500  # 법률 조문 최대 길이
    MAX_DOC_LENGTH_CASE = 800  # 판례 최대 길이
    MAX_DOC_LENGTH_COMMENTARY = 500  # 해설 최대 길이
    SUMMARY_THRESHOLD_LAW = 1000  # 법률 조문 요약 임계값
    SUMMARY_THRESHOLD_CASE = 800  # 판례 요약 임계값
    SUMMARY_THRESHOLD_COMMENTARY = 500  # 해설 요약 임계값
    MAX_DETAILED_EXTRACTS = 5  # 최대 상세 추출 수
    MAX_DETAILED_EXTRACT_LENGTH = 1000  # 최대 상세 추출 길이 (문자 수)
    MAX_SUMMARY_LENGTH = 500  # 최대 요약 길이
    
    # 메타데이터 제외 필드 목록
    EXCLUDED_METADATA_FIELDS = {
        'query', 'cross_encoder_score', 'original_score', 'keyword_bonus',
        'keyword_match_score', 'combined_relevance_score', 'source_type_weight',
        'strategy', 'id', 'doc_id', 'announce_date', 'response_date',
        'cross_encoder', 'embedding', 'vector', 'metadata_keys'
    }

    def __init__(self, prompts_dir: str = "streamlit/prompts"):
        """통합 프롬프트 매니저 초기화"""
        self.prompts_dir = Path(prompts_dir)
        # 성능 최적화: 디렉토리 생성은 실제 사용 시점으로 지연
        # self.prompts_dir.mkdir(parents=True, exist_ok=True)  # 제거

        # 성능 최적화: 프롬프트 로드 지연 (실제 사용 시점에 로드)
        self._base_prompts = None
        self._domain_templates = None
        self._question_type_templates = None
        self._model_optimizations = None
        self._prompts_loaded = False
        
        # 요약 에이전트 지연 초기화
        self._summary_agent = None
        self.llm_fast = None  # LLM 인스턴스 (나중에 설정 가능)

        try:
            logger.debug("UnifiedPromptManager initialized (lazy loading enabled)")
        except Exception:
            # 로깅 오류를 무시하고 계속 진행
            pass

    def _ensure_prompts_loaded(self):
        """프롬프트가 로드되지 않았으면 로드 (지연 로딩)"""
        if not self._prompts_loaded:
            # 기본 프롬프트 로드
            self._base_prompts = self._load_base_prompts()
            self._domain_templates = self._load_domain_templates()
            self._question_type_templates = self._load_question_type_templates()
            self._model_optimizations = self._load_model_optimizations()
            self._prompts_loaded = True
            logger.debug("UnifiedPromptManager prompts loaded")

    @property
    def base_prompts(self) -> Dict[str, str]:
        """기본 프롬프트 (지연 로딩)"""
        self._ensure_prompts_loaded()
        return self._base_prompts

    @property
    def domain_templates(self) -> Dict[LegalDomain, Dict[str, Any]]:
        """도메인 템플릿 (지연 로딩)"""
        self._ensure_prompts_loaded()
        return self._domain_templates

    @property
    def question_type_templates(self) -> Dict[QuestionType, Dict[str, Any]]:
        """질문 유형 템플릿 (지연 로딩)"""
        self._ensure_prompts_loaded()
        return self._question_type_templates

    @property
    def model_optimizations(self) -> Dict[ModelType, Dict[str, Any]]:
        """모델 최적화 설정 (지연 로딩)"""
        self._ensure_prompts_loaded()
        return self._model_optimizations

    def _load_base_prompts(self) -> Dict[str, str]:
        """기본 프롬프트 로드 - JSON 파일 우선, 없으면 하드코딩된 프롬프트 사용"""
        base_prompts = {}

        # JSON 파일에서 프롬프트 로드 시도
        # korean_legal_expert -> legal_expert_v1.0.json
        korean_legal_expert = self._load_prompt_from_json("legal_expert_v1.0.json")
        if korean_legal_expert:
            base_prompts["korean_legal_expert"] = korean_legal_expert
            logger.info("✅ Loaded korean_legal_expert prompt from legal_expert_v1.0.json")
        else:
            base_prompts["korean_legal_expert"] = self._get_korean_legal_expert_prompt()
            logger.info("⚠️ Using hardcoded korean_legal_expert prompt (JSON not found)")

        # natural_consultant -> natural_legal_consultant_v1.0.json
        natural_consultant = self._load_prompt_from_json("natural_legal_consultant_v1.0.json")
        if natural_consultant:
            base_prompts["natural_consultant"] = natural_consultant
            logger.info("✅ Loaded natural_consultant prompt from natural_legal_consultant_v1.0.json")
        else:
            base_prompts["natural_consultant"] = self._get_natural_consultant_prompt()
            logger.info("⚠️ Using hardcoded natural_consultant prompt (JSON not found)")

        # professional_advisor는 하드코딩된 프롬프트 사용 (JSON 파일이 없음)
        base_prompts["professional_advisor"] = self._get_professional_advisor_prompt()

        return base_prompts

    def _load_prompt_from_json(self, filename: str) -> Optional[str]:
        """JSON 파일에서 프롬프트 로드"""
        try:
            prompt_file = self.prompts_dir / filename
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = data.get('content', '')
                    if content:
                        logger.debug(f"✅ Loaded prompt from {filename}")
                        return content
                    else:
                        logger.warning(f"⚠️ {filename} exists but 'content' field is empty")
                        return None
            else:
                logger.debug(f"ℹ️ {filename} not found in {self.prompts_dir}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load prompt from {filename}: {e}")
            return None

    def _load_domain_templates(self) -> Dict[LegalDomain, Dict[str, Any]]:
        """도메인별 템플릿 로드"""
        return {
            LegalDomain.CIVIL_LAW: {
                "focus": "계약, 불법행위, 소유권, 상속",
                "key_laws": ["민법", "민사소송법", "부동산등기법"],
                "recent_changes": "2024년 민법 개정사항 반영",
                "template": self._get_civil_law_template()
            },
            LegalDomain.CRIMINAL_LAW: {
                "focus": "범죄 구성요건, 형량, 절차",
                "key_laws": ["형법", "형사소송법", "특별법"],
                "recent_changes": "디지털 성범죄 처벌법 등 신설법",
                "template": self._get_criminal_law_template()
            },
            LegalDomain.FAMILY_LAW: {
                "focus": "혼인, 이혼, 친자관계, 상속",
                "key_laws": ["민법 가족편", "가족관계의 등록 등에 관한 법률"],
                "recent_changes": "2024년 가족법 개정사항",
                "template": self._get_family_law_template()
            },
            LegalDomain.COMMERCIAL_LAW: {
                "focus": "회사법, 상행위, 어음수표",
                "key_laws": ["상법", "주식회사법", "어음법"],
                "recent_changes": "2024년 상법 개정사항",
                "template": self._get_commercial_law_template()
            },
            LegalDomain.ADMINISTRATIVE_LAW: {
                "focus": "행정행위, 행정절차, 행정소송",
                "key_laws": ["행정절차법", "행정소송법", "행정법"],
                "recent_changes": "2024년 행정법 개정사항",
                "template": self._get_administrative_law_template()
            },
            LegalDomain.LABOR_LAW: {
                "focus": "근로계약, 임금, 근로시간, 휴가",
                "key_laws": ["근로기준법", "노동조합법", "고용보험법"],
                "recent_changes": "2024년 노동법 개정사항",
                "template": self._get_labor_law_template()
            },
            LegalDomain.PROPERTY_LAW: {
                "focus": "부동산 계약, 등기, 권리보호",
                "key_laws": ["부동산등기법", "부동산 실권리자명의 등기에 관한 법률"],
                "recent_changes": "2024년 부동산법 개정사항",
                "template": self._get_property_law_template()
            },
            LegalDomain.INTELLECTUAL_PROPERTY: {
                "focus": "특허, 상표, 저작권, 디자인",
                "key_laws": ["특허법", "상표법", "저작권법", "디자인보호법"],
                "recent_changes": "2024년 지적재산권법 개정사항",
                "template": self._get_intellectual_property_template()
            },
            LegalDomain.TAX_LAW: {
                "focus": "소득세, 법인세, 부가가치세",
                "key_laws": ["소득세법", "법인세법", "부가가치세법"],
                "recent_changes": "2024년 세법 개정사항",
                "template": self._get_tax_law_template()
            },
            LegalDomain.CIVIL_PROCEDURE: {
                "focus": "민사소송 절차, 증거, 집행",
                "key_laws": ["민사소송법", "민사집행법", "가사소송법"],
                "recent_changes": "2024년 민사소송법 개정사항",
                "template": self._get_civil_procedure_template()
            },
            LegalDomain.CRIMINAL_PROCEDURE: {
                "focus": "형사소송 절차, 수사, 재판",
                "key_laws": ["형사소송법", "수사절차법"],
                "recent_changes": "2024년 형사소송법 개정사항",
                "template": self._get_criminal_procedure_template()
            }
        }

    def _load_question_type_templates(self) -> Dict[QuestionType, Dict[str, Any]]:
        """질문 유형별 템플릿 로드"""
        return {
            QuestionType.PRECEDENT_SEARCH: {
                "template": self._get_precedent_search_template(),
                "context_keys": ["precedent_list"],
                "max_context_length": 3000,
                "priority": "high"
            },
            QuestionType.LAW_INQUIRY: {
                "template": self._get_law_inquiry_template(),
                "context_keys": ["law_articles"],
                "max_context_length": 2000,
                "priority": "high"
            },
            QuestionType.LEGAL_ADVICE: {
                "template": self._get_legal_advice_template(),
                "context_keys": ["context"],
                "max_context_length": 4000,
                "priority": "high"
            },
            QuestionType.PROCEDURE_GUIDE: {
                "template": self._get_procedure_guide_template(),
                "context_keys": ["procedure_info"],
                "max_context_length": 2500,
                "priority": "medium"
            },
            QuestionType.TERM_EXPLANATION: {
                "template": self._get_term_explanation_template(),
                "context_keys": ["term_info"],
                "max_context_length": 1500,
                "priority": "medium"
            },
            QuestionType.GENERAL_QUESTION: {
                "template": self._get_general_question_template(),
                "context_keys": ["general_context"],
                "max_context_length": 2000,
                "priority": "low"
            }
        }

    def _load_model_optimizations(self) -> Dict[ModelType, Dict[str, Any]]:
        """모델별 최적화 설정 로드"""
        return {
            ModelType.GEMINI: {
                "max_tokens": 8192,
                "temperature": 0.3,
                "system_prompt_style": "structured",
                "context_window": 0.8
            },
            ModelType.OPENAI: {
                "max_tokens": 4096,
                "temperature": 0.1,
                "system_prompt_style": "professional",
                "context_window": 0.9
            }
        }

    def get_optimized_prompt(self,
                           query: str,
                           question_type: QuestionType,
                           domain: Optional[LegalDomain] = None,
                           context: Optional[Dict[str, Any]] = None,
                           model_type: ModelType = ModelType.GEMINI,
                           base_prompt_type: str = "korean_legal_expert") -> str:
        """최적화된 프롬프트 생성"""
        try:
            # Domain 타입 정규화 (문자열을 LegalDomain enum으로 변환)
            normalized_domain = self._normalize_legal_domain(domain)

            # Base prompt type 동적 선택 (지정되지 않았거나 "auto"인 경우)
            if not base_prompt_type or base_prompt_type == "auto":
                base_prompt_type = self._select_base_prompt_type(question_type, normalized_domain)

            # 실제 사용되는 템플릿 로깅
            logger.info(
                f"📋 [PROMPT TEMPLATE] Using templates: "
                f"base={base_prompt_type}, "
                f"domain={normalized_domain.value if normalized_domain else 'None'}, "
                f"question_type={question_type.name if hasattr(question_type, 'name') else question_type}, "
                f"model={model_type.value if hasattr(model_type, 'value') else model_type}"
            )

            # 1. 기본 프롬프트 로드 (property로 지연 로딩)
            base_prompts = self.base_prompts  # property 접근으로 지연 로딩 트리거
            base_prompt = base_prompts.get(base_prompt_type, base_prompts["korean_legal_expert"])

            # 2. 도메인 특화 강화 (정규화된 domain 사용, property로 지연 로딩)
            domain_templates = self.domain_templates  # property 접근으로 지연 로딩 트리거
            if normalized_domain and normalized_domain in domain_templates:
                domain_info = domain_templates[normalized_domain]
                base_prompt = self._add_domain_specificity(base_prompt, domain_info)
                logger.info(f"✅ [DOMAIN TEMPLATE] Applied domain template: {normalized_domain.value}")
            elif normalized_domain:
                logger.warning(f"⚠️ [DOMAIN TEMPLATE] Domain template not found for: {normalized_domain.value}")
            else:
                logger.debug("ℹ️ [DOMAIN TEMPLATE] No domain specified, skipping domain template")

            # 3. 질문 유형별 구조화 (property로 지연 로딩)
            question_type_templates = self.question_type_templates  # property 접근으로 지연 로딩 트리거
            question_template = question_type_templates.get(question_type)
            if question_template:
                base_prompt = self._add_question_structure(base_prompt, question_template)
                logger.info(
                    f"✅ [QUESTION TEMPLATE] Applied question type template: "
                    f"{question_type.name if hasattr(question_type, 'name') else question_type}, "
                    f"priority={question_template.get('priority', 'unknown')}"
                )
            else:
                logger.warning(
                    f"⚠️ [QUESTION TEMPLATE] Question type template not found for: "
                    f"{question_type.name if hasattr(question_type, 'name') else question_type}"
                )

            # 4. 컨텍스트 최적화
            if context:
                base_prompt = self._optimize_context(base_prompt, context, question_template)

            # 5. 모델별 최적화 (property로 지연 로딩)
            model_optimizations = self.model_optimizations  # property 접근으로 지연 로딩 트리거
            model_config = model_optimizations.get(model_type)
            if model_config:
                base_prompt = self._model_specific_optimization(base_prompt, model_config)
                logger.info(
                    f"✅ [MODEL OPTIMIZATION] Applied model optimization: "
                    f"{model_type.value if hasattr(model_type, 'value') else model_type}, "
                    f"temperature={model_config.get('temperature', 'unknown')}, "
                    f"max_tokens={model_config.get('max_tokens', 'unknown')}"
                )
            else:
                logger.warning(
                    f"⚠️ [MODEL OPTIMIZATION] Model optimization not found for: "
                    f"{model_type.value if hasattr(model_type, 'value') else model_type}"
                )

            # 6. 최종 프롬프트 구성
            # 🔥 개선: final_prompt 초기화 보장
            final_prompt = None
            try:
                final_prompt = self._build_final_prompt(base_prompt, query, context, question_type)
            except Exception as build_error:
                logger.error(f"Error building final prompt: {build_error}")
                # fallback 프롬프트 사용
                final_prompt = self._get_fallback_prompt(query)
            
            # final_prompt가 여전히 None이면 fallback 사용
            if final_prompt is None:
                logger.warning("⚠️ [PROMPT BUILD] final_prompt is None, using fallback")
                final_prompt = self._get_fallback_prompt(query)

            # 7. 토큰 수 최종 검증
            final_tokens = self._estimate_tokens(final_prompt)
            
            if final_tokens > self.MAX_INPUT_TOKENS:
                logger.error(
                    f"❌ [TOKEN LIMIT EXCEEDED] Final prompt exceeds maximum: "
                    f"{final_tokens:,} tokens (max: {self.MAX_INPUT_TOKENS:,})"
                )
                # 긴급 축약
                final_prompt = self._emergency_truncate_prompt(
                    final_prompt, 
                    self.MAX_INPUT_TOKENS, 
                    base_prompt, 
                    query
                )
                final_tokens = self._estimate_tokens(final_prompt)
                logger.warning(
                    f"⚠️ [EMERGENCY TRUNCATION] Prompt truncated to {final_tokens:,} tokens"
                )
            
            logger.info(
                f"✅ [PROMPT TOKENS] Final prompt: {final_tokens:,} tokens "
                f"({final_tokens/self.MAX_INPUT_TOKENS*100:.1f}% of max)"
            )

            # 8. 프롬프트 검증: 문서 내용이 포함되었는지 확인 (강화된 검증)
            validation_result = self._validate_prompt_contains_documents(final_prompt, context)
            if not validation_result["has_document_content"]:
                doc_count_in_context = validation_result.get("document_count_in_context", 0)
                doc_count = context.get("document_count", 0) if context else 0
                doc_count = doc_count or doc_count_in_context

                if doc_count > 0:
                    logger.warning(
                        f"⚠️ [PROMPT VALIDATION] Final prompt does not contain document content "
                        f"despite having {doc_count} documents in context. "
                        f"Prompt length: {len(final_prompt)} chars, "
                        f"Context text length: {validation_result.get('context_text_length', 0)} chars"
                    )

                    # 검증 상세 정보 로깅
                    validation_details = validation_result.get("validation_details", [])
                    if validation_details:
                        try:
                            found_count = sum(1 for d in validation_details if d and isinstance(d, dict) and d.get('found_in_prompt'))
                            logger.debug(
                                f"📋 [PROMPT VALIDATION] Details: "
                                f"{found_count}/{len(validation_details)} "
                                f"documents found in prompt"
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ [PROMPT VALIDATION] Error processing validation details: {e}")
                else:
                    logger.info(
                        f"ℹ️ [PROMPT VALIDATION] No documents in context "
                        f"(document_count: {doc_count}, context_count: {doc_count_in_context})"
                    )
            else:
                # 검증 성공 로깅
                doc_count_in_prompt = validation_result.get("document_count_in_prompt", 0)
                doc_count_in_context = validation_result.get("document_count_in_context", 0)
                logger.info(
                    f"✅ [PROMPT VALIDATION] Document content found in prompt: "
                    f"{doc_count_in_prompt}/{doc_count_in_context} documents included"
                )

            return final_prompt

        except Exception as e:
            logger.error(f"Error generating optimized prompt: {e}")
            return self._get_fallback_prompt(query)

    def _add_domain_specificity(self, base_prompt: str, domain_info: Dict[str, Any]) -> str:
        """도메인 특화 강화 - 답변 품질 향상을 위한 개선"""
        domain_specificity = f"""

## 도메인 특화 지침
- **관련 분야**: {domain_info['focus']}
- **주요 법령**: {', '.join(domain_info['key_laws'])}
- **최신 개정사항**: {domain_info['recent_changes']}

### 답변 품질 향상 요구사항
1. **법적 정확성**: 관련 법령의 정확한 조문 인용 필수
2. **판례 활용**: 최신 대법원 판례 및 하급심 판례 적극 활용
3. **실무 관점**: 실제 법원, 검찰, 법무부 실무 기준 반영
4. **구체적 조언**: 실행 가능한 구체적 방안 제시
5. **리스크 관리**: 법적 리스크와 주의사항 명확히 제시

{domain_info['template']}
"""
        return base_prompt + domain_specificity

    def _add_question_structure(self, base_prompt: str, question_template: Dict[str, Any]) -> str:
        """질문 유형별 구조화 - 답변 품질 향상을 위한 개선"""
        structure_guidance = f"""

## 답변 구조 가이드
{question_template['template']}

## 컨텍스트 처리
- 최대 컨텍스트 길이: {question_template['max_context_length']}자
- 우선순위: {question_template['priority']}

### 답변 품질 검증 체크리스트
1. **완성도**: 질문에 대한 완전한 답변 제공 여부
2. **정확성**: 법적 정보의 정확성 및 최신성 확인
3. **구조화**: 논리적이고 체계적인 답변 구조
4. **실용성**: 실행 가능한 구체적 조언 포함
5. **신뢰성**: 근거 있는 법적 분석 및 판례 인용
"""
        return base_prompt + structure_guidance

    def _optimize_context(self, base_prompt: str, context: Dict[str, Any], question_template: Optional[Dict[str, Any]]) -> str:
        """컨텍스트 최적화 - 문서 및 데이터베이스 결과 강제 포함"""
        if not context:
            return base_prompt

        if question_template is None:
            question_template = {}

        max_length = question_template.get('max_context_length', 5000)  # 기본값 증가: 2000 -> 5000
        context_keys = question_template.get('context_keys', [])

        optimized_context = {}

        # 1. context_keys에 지정된 키 포함
        for key in context_keys:
            if key in context:
                content = context[key]
                if isinstance(content, str) and len(content) > max_length:
                    content = content[:max_length] + "..."
                optimized_context[key] = content

        # 2. prompt_optimized_text가 있으면 context로 변환 (최우선)
        if "prompt_optimized_text" in context:
            prompt_text = context["prompt_optimized_text"]
            if prompt_text and len(prompt_text.strip()) > 0:
                # prompt_optimized_text를 context로 사용 (문서 내용 포함)
                if len(prompt_text) > max_length * 2:  # 프롬프트 텍스트는 더 긴 길이 허용
                    prompt_text = prompt_text[:max_length * 2] + "..."
                optimized_context["context"] = prompt_text
                logger.info(f"✅ [CONTEXT OPTIMIZATION] Using prompt_optimized_text as context ({len(prompt_text)} chars)")

                # ⚠️ 중요: prompt_optimized_text를 사용하더라도 structured_documents의 문서 내용을 강제로 추가
                # prompt_optimized_text에 이미 문서 내용이 포함되어 있을 수 있지만,
                # structured_documents의 구조화된 정보를 명시적으로 포함하여 LLM이 명확하게 인식하도록 함
                structured_docs = context.get("structured_documents", {})
                if isinstance(structured_docs, dict):
                    documents = structured_docs.get("documents", [])
                    if documents:
                        # 🔥 개선: "추가 참고 문서" 섹션 제거
                        # "검색된 참고 문서" 섹션만 사용하여 중복 방지
                        logger.debug(
                            f"✅ [CONTEXT OPTIMIZATION] Skipping '추가 참고 문서' section. "
                            f"Documents will be included in '검색된 참고 문서' section only."
                        )

                        # structured_documents를 명시적으로 포함 (항상)
                        # 이는 LLM이 문서 구조를 명확히 이해할 수 있도록 함
                        normalized_docs = []
                        for doc in documents[:8]:
                            normalized = self._normalize_document_fields(doc)
                            if normalized and normalized.get("content"):
                                normalized_docs.append(normalized)

                        if normalized_docs:
                            optimized_context["structured_documents"] = {
                                **structured_docs,
                                "documents": normalized_docs,
                                "total_count": len(normalized_docs)
                            }
                            logger.info(
                                f"✅ [CONTEXT OPTIMIZATION] Included {len(normalized_docs)} structured_documents "
                                f"in optimized_context for explicit reference"
                            )

        # 3. context 키 포함 (prompt_optimized_text가 없는 경우)
        if "context" in context and "context" not in optimized_context:
            content = context["context"]
            if isinstance(content, str) and len(content) > max_length:
                content = content[:max_length] + "..."
            optimized_context["context"] = content

        # 4. structured_documents에서 실제 문서 내용 추출하여 context에 추가
        structured_docs = context.get("structured_documents", {})
        if isinstance(structured_docs, dict):
            documents = structured_docs.get("documents", [])
            if documents:
                doc_contents = []
                valid_doc_count = 0

                for doc in documents[:8]:  # 상위 8개 문서
                    if isinstance(doc, dict):
                        # 문서 필드 정규화
                        normalized_doc = self._normalize_document_fields(doc)
                        if not normalized_doc:
                            logger.debug(f"⚠️ [PROMPT BUILD] Document normalized to None, skipping: {doc.get('id', 'unknown')[:50]}")
                            continue
                        doc_content = normalized_doc.get("content", "")
                        doc_source = normalized_doc.get("source", "Unknown")
                        doc_score = normalized_doc.get("relevance_score", 0.0)

                        if doc_content and len(doc_content.strip()) > 10:
                            doc_contents.append(
                                f"\n[문서 출처: {doc_source}] [관련도: {doc_score:.3f}]\n{doc_content[:1000]}"  # 각 문서 최대 1000자
                            )
                            valid_doc_count += 1

                if doc_contents:
                    docs_text = "\n".join(doc_contents)
                    # 기존 context에 문서 내용 추가 또는 교체
                    if "context" in optimized_context:
                        optimized_context["context"] = optimized_context["context"] + "\n\n" + docs_text
                    else:
                        optimized_context["context"] = docs_text
                    logger.info(
                        f"✅ [CONTEXT OPTIMIZATION] Added {valid_doc_count}/{len(documents)} valid documents "
                        f"from structured_documents ({len(docs_text)} chars)"
                    )
                else:
                    logger.warning(
                        f"⚠️ [CONTEXT OPTIMIZATION] No valid document content found in {len(documents)} documents"
                    )

                # structured_documents를 optimized_context에 명시적으로 포함 (중요!)
                # 정규화된 문서로 업데이트
                normalized_docs = []
                for doc in documents[:8]:
                    normalized = self._normalize_document_fields(doc)
                    if normalized and normalized.get("content"):
                        normalized_docs.append(normalized)

                if normalized_docs:
                    structured_docs_normalized = {
                        **structured_docs,
                        "documents": normalized_docs,
                        "total_count": len(normalized_docs),
                        "original_count": len(documents)
                    }
                    optimized_context["structured_documents"] = structured_docs_normalized
                    logger.info(
                        f"✅ [CONTEXT OPTIMIZATION] Included normalized structured_documents "
                        f"in optimized_context ({len(normalized_docs)}/{len(documents)} valid documents)"
                    )
                else:
                    optimized_context["structured_documents"] = structured_docs
                    logger.warning(
                        "⚠️ [CONTEXT OPTIMIZATION] No valid documents after normalization, "
                        "keeping original structured_documents"
                    )

        # structured_documents가 context에는 있지만 optimized_context에 없는 경우 직접 추가
        elif "structured_documents" in context:
            structured_docs = context.get("structured_documents", {})
            if isinstance(structured_docs, dict) and structured_docs.get("documents"):
                optimized_context["structured_documents"] = structured_docs
                logger.info("✅ [CONTEXT OPTIMIZATION] Added structured_documents from original context to optimized_context")

        # 5. context_keys가 비어있으면 모든 관련 컨텍스트 키 포함
        if not context_keys and context:
            for key, value in context.items():
                if key not in ["query_type", "context_length", "docs_truncated", "prompt_optimized_text"]:  # 메타데이터 및 이미 처리한 키 제외
                    if isinstance(value, str) and len(value) > max_length:
                        value = value[:max_length] + "..."
                    if key not in optimized_context:  # 이미 포함된 키는 중복 추가하지 않음
                        optimized_context[key] = value

        # 질문 유형별 컨텍스트 구조화
        question_type = context.get("query_type")
        if isinstance(question_type, str):
            # 문자열을 QuestionType enum으로 변환
            try:
                question_type_enum = QuestionType[question_type.upper()] if question_type.upper() in [e.name for e in QuestionType] else QuestionType.GENERAL_QUESTION
            except (KeyError, AttributeError, TypeError):
                question_type_enum = QuestionType.GENERAL_QUESTION
        elif hasattr(question_type, 'name'):
            question_type_enum = question_type
        else:
            question_type_enum = QuestionType.GENERAL_QUESTION

        structured_context = self._structure_context_by_question_type(optimized_context, question_type_enum)

        # 🔴 개선: 불필요한 지침 섹션 제거 (base_prompt에 이미 포함되거나 _build_final_prompt에서 추가됨)
        context_guidance = f"{structured_context}"
        try:
            sql_schema = context.get("sql_schema")
            if isinstance(sql_schema, str) and len(sql_schema) > 0:
                context_guidance += f"""

### Text-to-SQL 스키마 요약 (PostgreSQL)
{sql_schema}

### SQL 작성 지침
- SELECT만 사용하세요. DML/DDL은 금지됩니다.
- WHERE 절에 정확한 필터를 명시하고, 반드시 LIMIT를 포함하세요.
- 결과는 조문/사건번호/선고일/법원을 기준으로 재현 가능해야 합니다.
- PostgreSQL 문법을 사용하세요 (예: 문자열 연결은 ||, 날짜 비교는 DATE 타입 사용).

### PostgreSQL 스키마 구조 (PostgreSQL 12+ 기준)

#### 1. 법률 데이터 테이블
- **statutes** (법률 정보 - Open Law API)
  - id: SERIAL PRIMARY KEY
  - law_id: INTEGER NOT NULL, UNIQUE (법령ID)
  - law_name_kr: TEXT NOT NULL (법령명 한글)
  - law_name_hanja: TEXT (법령명 한자)
  - law_name_en: TEXT (법령명 영어)
  - law_abbrv: TEXT (법령약칭)
  - law_type: TEXT (법령종류)
  - law_type_code: TEXT (법종구분코드)
  - proclamation_date: DATE (공포일자)
  - effective_date: DATE (시행일자)
  - ministry_name: TEXT (소관부처명)
  - domain: TEXT (분야: civil_law, criminal_law)

- **statutes_articles** (법률 조문 - Open Law API)
  - id: SERIAL PRIMARY KEY
  - statute_id: INTEGER NOT NULL (FK → statutes(id))
  - article_no: TEXT NOT NULL (조문번호, 예: "000200" = 제2조, "007500" = 제750조)
  - article_title: TEXT (조문제목)
  - article_content: TEXT NOT NULL (조문내용)
  - clause_no: TEXT (항번호)
  - clause_content: TEXT (항내용)
  - item_no: TEXT (호번호)
  - item_content: TEXT (호내용)
  - sub_item_no: TEXT (목번호)
  - sub_item_content: TEXT (목내용)
  - effective_date: DATE (조문시행일자)

**⚠️ 중요 사항:**
- `statutes.law_name_kr`은 법령명 한글입니다. 법령명 검색 시 이 컬럼을 사용하세요.
- `statutes_articles.article_no`는 TEXT 타입이며 6자리 문자열 형식입니다 (예: "007500" = 제750조).
- 조문번호 변환: "제750조" → "007500", "제2조" → "000200" (6자리, 앞에 0으로 패딩)
- 날짜 필드는 DATE 타입이므로 날짜 비교 시 DATE 타입을 사용하세요.
- JOIN 시 외래 키 관계를 정확히 사용하세요: `statutes_articles.statute_id → statutes.id`

### 법령 조문 검색 규칙 (중요)
- **법령 조문은 정확한 법령명과 조문번호만 조회하세요.**
- `statutes` 테이블과 `statutes_articles` 테이블을 JOIN하여 조회하세요.
- 법령명은 `law_name_kr` 컬럼을 사용하며, LIKE '%법령명%'로 매칭하세요.
- 조문번호(`article_no`)는 TEXT 타입이며 6자리 문자열 형식입니다. "제750조"는 "007500"으로 변환해야 합니다.
- 예: "민법 제750조" → JOIN 사용: SELECT s.law_name_kr, sa.article_no, sa.article_content FROM statutes_articles sa JOIN statutes s ON sa.statute_id = s.id WHERE s.law_name_kr LIKE '%민법%' AND sa.article_no = '007500' LIMIT 5;

### 예시(한국어 → PostgreSQL SQL)
- 질의: "민법 제750조 조문 보여줘"
  SQL: SELECT s.law_name_kr, sa.article_no, sa.article_content FROM statutes_articles sa JOIN statutes s ON sa.statute_id = s.id WHERE s.law_name_kr LIKE '%민법%' AND sa.article_no = '007500' LIMIT 5;
- 질의: "형법 제307조 찾아줘"
  SQL: SELECT s.law_name_kr, sa.article_no, sa.article_content FROM statutes_articles sa JOIN statutes s ON sa.statute_id = s.id WHERE s.law_name_kr LIKE '%형법%' AND sa.article_no = '003070' LIMIT 5;
- 질의: "상법 제24조 전문 보여줘"
  SQL: SELECT s.law_name_kr, sa.article_no, sa.article_content FROM statutes_articles sa JOIN statutes s ON sa.statute_id = s.id WHERE s.law_name_kr LIKE '%상법%' AND sa.article_no = '000240' LIMIT 5;
- 질의: "근로기준법 제60조 연차 규정"
  SQL: SELECT s.law_name_kr, sa.article_no, sa.article_content FROM statutes_articles sa JOIN statutes s ON sa.statute_id = s.id WHERE s.law_name_kr LIKE '%근로기준법%' AND sa.article_no = '000600' LIMIT 5;
"""
        except Exception:
            pass

        context_guidance += """

### 답변 작성 체크리스트
- [ ] 제공된 문서의 내용을 바탕으로 답변했는가?
- [ ] 법률 조항이나 판례를 구체적으로 인용했는가?
- [ ] 출처를 명시했는가?
"""
        return base_prompt + context_guidance

    def _remove_duplicate_document_sections(self, prompt: str) -> str:
        """프롬프트에서 중복된 문서 섹션 제거"""
        import re
        
        # 문서 섹션 패턴 찾기
        patterns = [
            r'## 검색된 법률 문서.*?(?=##|\n---|\n## 사용자 질문|$)',
            r'## 제공된 법률 문서.*?(?=##|\n---|\n## 사용자 질문|$)',
            r'## 참고 문서 목록.*?(?=##|\n---|\n## 사용자 질문|$)',
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, prompt, re.DOTALL))
            if len(matches) > 1:
                # 첫 번째만 남기고 나머지 제거
                for match in matches[1:]:
                    prompt = prompt[:match.start()] + prompt[match.end():]
        
        return prompt
    
    def _remove_duplicate_citation_requirements(self, prompt: str) -> str:
        """프롬프트에서 중복된 Citation 요구사항 제거"""
        import re
        
        # Citation 관련 섹션 패턴
        patterns = [
            r'STEP 4: Citation 필수 요구사항.*?(?=STEP|\n---|\n## 사용자 질문|$)',
            r'Citation 필수 요구사항.*?(?=STEP|\n---|\n## 사용자 질문|$)',
            r'최소 2개 이상의 법률 조문.*?(?=STEP|\n---|\n## 사용자 질문|$)',
        ]
        
        for pattern in patterns:
            prompt = re.sub(pattern, '', prompt, flags=re.DOTALL)
        
        return prompt
    
    def _simplify_base_prompt(self, base_prompt: str) -> str:
        """base_prompt 간소화 - 모든 문서 섹션 제거"""
        import re
        
        # 🔥 개선: 예시 부분 보호 (임시로 플레이스홀더로 교체)
        # 예시 패턴: (예: "...") 형식 (여러 줄 포함 가능)
        example_placeholders = {}
        # 예시 패턴: (예: 로 시작하여 ) 로 끝나는 부분 (최대 500자, 여러 줄 포함)
        example_pattern = r'\(예:\s*"[^"]{0,500}"\)'
        example_matches = list(re.finditer(example_pattern, base_prompt, re.DOTALL))
        # 역순으로 교체하여 인덱스 변경 문제 방지
        for i, match in enumerate(reversed(example_matches)):
            placeholder = f"__EXAMPLE_PLACEHOLDER_{len(example_matches) - 1 - i}__"
            example_placeholders[placeholder] = match.group(0)
            base_prompt = base_prompt[:match.start()] + placeholder + base_prompt[match.end():]
        
        # 🔥 개선: 모든 문서 관련 섹션 패턴 추가
        document_section_patterns = [
            r'## 검색된 법률 문서.*?(?=##|$)',
            r'## 검색된 법률 조문 문서.*?(?=##|$)',
            r'## 검색된 판례 문서.*?(?=##|$)',
            r'## 참고 문서 목록.*?(?=##|$)',
            r'## 참고 문서 내용.*?(?=##|$)',
            r'## 조문 해설.*?(?=##|$)',
            r'## Context Summary.*?(?=##|$)',
            r'## Detailed Extracts.*?(?=##|$)',
            r'## \[전체 문서\].*?(?=##|$)',
            r'### 🔴 최우선 문서.*?(?=###|##|$)',
            r'### 🟡 중요 문서.*?(?=###|##|$)',
            r'### 문서 \d+:.*?(?=###|##|$)',  # 개별 문서 항목
            r'\[문서 \d+\].*?(?=\[문서|##|$)',  # 문서 인용 형식
        ]
        
        for pattern in document_section_patterns:
            base_prompt = re.sub(pattern, '', base_prompt, flags=re.DOTALL)
        
        # 불필요한 섹션 제거
        sections_to_remove = [
            r'## 법률 자문 지침.*?(?=##|$)',
            r'## 컨텍스트 처리.*?(?=##|$)',
            r'## 답변 품질 검증 체크리스트.*?(?=##|$)',
            r'## 모델 최적화 설정.*?(?=##|$)',
            r'## 검색 결과 통계.*?(?=##|$)',
            # 🔥 개선: 중복되는 답변 구조 가이드 제거 (instruction_section에 이미 포함)
            r'## 답변 구조 가이드.*?(?=##|$)',
            r'## 법령 질의 지침.*?(?=##|$)',
            # 🔥 개선: 도메인 특화 지침과 민사법 특화 지침 중복 제거 (간소화)
            r'## 도메인 특화 지침.*?(?=##|$)',
            r'## 민사법 특화 지침.*?(?=##|$)',
        ]
        
        for pattern in sections_to_remove:
            base_prompt = re.sub(pattern, '', base_prompt, flags=re.DOTALL)
        
        # 중복된 설명 제거 (더 정확한 패턴 - 섹션 제목으로 시작하는 것만 매칭)
        # 🔥 개선: 예시 부분을 보호하기 위해 더 정확한 패턴 사용
        # "## 한국 법률 특성" 섹션만 매칭하고, 다음 섹션(##) 전까지만 매칭
        base_prompt = re.sub(
            r'(## 한국 법률 특성\s*\n)(?:### .*?\n)*.*?실무적 관점.*?(?=\n## |$)', 
            r'\1- 성문법 중심, 대법원 판례 중시, 실무적 관점\n', 
            base_prompt, 
            flags=re.DOTALL
        )
        
        # 🔥 성능 최적화: 긴 예시 간소화
        # 예시 패턴 찾아서 간소화
        base_prompt = re.sub(
            r'❌ "([^"]+)" \(잘못된 예\)\s*\n\s*✅ "([^"]+)" \(올바른 예\)',
            r'❌ "\1" → ✅ "\2"',
            base_prompt,
            flags=re.MULTILINE
        )
        
        # 반복되는 예시 제거 (3개 이상 연속)
        base_prompt = re.sub(
            r'(❌ "[^"]+" → ✅ "[^"]+"\n){3,}',
            lambda m: m.group(0).split('\n')[0] + '\n',  # 첫 번째 예시만 유지
            base_prompt
        )
        
        # 🔥 성능 최적화: 반복 지침 통합
        # "문서 기반 답변만 허용" 같은 반복 지침 제거
        base_prompt = re.sub(
            r'⚠️\s*\*\*중요\*\*:.*?인용하지 마세요\.\s*',
            '⚠️ 관련 문서만 인용, 최소 2개 필수\n',
            base_prompt,
            flags=re.DOTALL
        )
        
        # "띄어쓰기 필수" 반복 제거 (더 정확한 패턴 - 섹션 제목만 매칭)
        base_prompt = re.sub(
            r'### \d+\. 띄어쓰기 필수 준수\s*\n.*?(?=### |## |$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        
        # "자연스러운 문단 흐름" 반복 제거 (원본 템플릿에는 해당 섹션이 없으므로 비활성화)
        # 🔥 개선: 원본 템플릿에는 "### 3. 자연스러운 문단 흐름" 섹션이 없으므로 이 패턴은 실행하지 않음
        # 대신 "자연스러운 문단 흐름 유지"라는 문구만 있는 경우는 그대로 유지
        # base_prompt = re.sub(
        #     r'### \d+\. 자연스러운 문단 흐름\s*\n.*?(?=### |## |$)',
        #     '',
        #     base_prompt,
        #     flags=re.DOTALL
        # )
        
        # 🔥 개선: 중복된 답변 생성 규칙 제거
        base_prompt = re.sub(
            r'## 답변 생성 지시사항.*?답변 생성 규칙.*?(?=##|$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        
        # 🔥 개선: 중복된 "문서 인용 규칙" 및 "중요 사항" 섹션 제거
        # 문서 인용 규칙 및 중요 사항은 나중에 추가되므로 base_prompt에서 제거
        base_prompt = re.sub(
            r'## 문서 인용 규칙.*?(?=##|$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        base_prompt = re.sub(
            r'## 중요 사항.*?(?=##|$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        
        # 🔥 개선: 원본 문서 데이터 제거 ([문서 출처: ...] 형식)
        base_prompt = re.sub(
            r'\[문서 출처:\s*[^\]]+\]\s*\[관련도:\s*[\d.]+\].*?(?=\n\n|\[문서 출처:|##|$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        
        # 🔥 개선: 의미 없는 단독 헤더 제거 (# 만 있는 줄)
        base_prompt = re.sub(
            r'^\s*#\s*$',
            '',
            base_prompt,
            flags=re.MULTILINE
        )
        
        # 🔥 개선: 이중 헤더 수정 (## ## -> ##)
        base_prompt = re.sub(
            r'##\s+##\s+',
            '## ',
            base_prompt
        )
        
        # 🔥 개선: 연속된 구분선 제거 (--- 다음 ---)
        base_prompt = re.sub(
            r'---\s*\n\s*---',
            '---',
            base_prompt
        )
        
        # 🔥 개선: base_prompt 끝의 구분선 제거 (나중에 추가되므로)
        base_prompt = re.sub(
            r'---\s*$',
            '',
            base_prompt,
            flags=re.MULTILINE
        )
        base_prompt = base_prompt.rstrip()
        
        # 🔥 개선: 연속된 빈 줄 정리 (3개 이상 연속 빈 줄을 2개로, 2개 연속 빈 줄은 1개로)
        base_prompt = re.sub(
            r'\n{3,}',
            '\n\n',
            base_prompt
        )
        
        # 🔥 개선: 섹션 제목 앞의 불필요한 빈 줄 제거 (## 앞에 빈 줄이 2개 이상이면 1개로)
        base_prompt = re.sub(
            r'\n{2,}(## )',
            r'\n\1',
            base_prompt
        )
        
        # 🔥 개선: 중복된 체크리스트 제거
        base_prompt = re.sub(
            r'## 답변 작성 체크리스트.*?(?=##|$)',
            '',
            base_prompt,
            flags=re.DOTALL
        )
        
        # 🔥 개선: 예시 부분 복원
        for placeholder, original_example in example_placeholders.items():
            base_prompt = base_prompt.replace(placeholder, original_example)
        
        return base_prompt.strip()
    
    def _simplify_prompt_for_no_results(self, base_prompt: str) -> str:
        """검색 결과가 없을 때 프롬프트 단순화 - 프롬프트 출력 방지"""
        # base_prompt에서 검색 결과 관련 지시사항 제거
        # 핵심 역할과 기본 답변 스타일만 유지

        # 프롬프트를 줄 단위로 분리하여 필터링
        lines = base_prompt.split('\n')
        simplified_lines = []
        skip_until_next_section = False

        for i, line in enumerate(lines):
            # 검색 결과 관련 섹션 건너뛰기
            if any(keyword in line for keyword in ['검색 결과', '문서', '인용', '판례', '조문', '검색된']):
                if '##' in line or '###' in line:  # 섹션 헤더인 경우
                    skip_until_next_section = True
                    continue

            # 다음 섹션 헤더를 만나면 스킵 중지
            if skip_until_next_section and ('##' in line or '---' in line):
                skip_until_next_section = False
                if '---' in line:
                    simplified_lines.append(line)
                continue

            # 스킵 중이 아닌 경우에만 추가
            if not skip_until_next_section:
                simplified_lines.append(line)

        simplified = '\n'.join(simplified_lines)

        # 간단한 역할 설명 추가
        simplified_prompt = f"""당신은 친절하고 전문적인 법률 상담 변호사입니다.

## 핵심 역할
질문에 대해 가능한 한 정확하고 도움이 되는 답변을 제공하세요.

## 답변 스타일
- 자연스럽고 친근하게 대화하세요
- 법률 용어는 쉽게 풀어서 설명하세요
- 질문의 범위에 맞는 적절한 양의 정보를 제공하세요
- 불확실한 부분은 솔직하게 말하고 전문가 상담을 권하세요

{simplified}
"""

        return simplified_prompt

    def _model_specific_optimization(self, base_prompt: str, model_config: Dict[str, Any]) -> str:
        """모델별 최적화"""
        optimization = f"""

## 모델 최적화 설정
- 최대 토큰: {model_config['max_tokens']}
- 온도: {model_config['temperature']}
- 스타일: {model_config['system_prompt_style']}
- 컨텍스트 윈도우: {model_config['context_window']}
"""
        return base_prompt + optimization

    def _build_final_prompt(self, base_prompt: str, query: str, context: Dict[str, Any], question_type: QuestionType) -> str:
        """최종 프롬프트 구성 - 토큰 제한 적용"""
        
        # 🔥 스트리밍 모드 확인 (context 또는 metadata에서)
        is_streaming = context.get("is_streaming", False) or context.get("metadata", {}).get("streaming_mode", False)
        
        # 🔥 스트리밍 모드 확인 (context 또는 metadata에서)
        is_streaming = context.get("is_streaming", False) or (
            isinstance(context.get("metadata"), dict) and context.get("metadata", {}).get("streaming_mode", False)
        )
        
        # 🔥 개선: final_prompt 초기화 보장
        final_prompt = ""
        
        # 토큰 제한 계산
        MAX_SAFE_TOKENS = int(self.MAX_INPUT_TOKENS * (1 - self.SAFE_MARGIN))  # 약 943,718 토큰
        
        # 1단계: 기본 프롬프트 + 질문 토큰 수 계산
        base_tokens = self._estimate_tokens(base_prompt)
        query_tokens = self._estimate_tokens(query)
        reserved_tokens = base_tokens + query_tokens + 1000  # 여유 공간 1000토큰
        
        # 사용 가능한 문서 토큰 수
        available_doc_tokens = MAX_SAFE_TOKENS - reserved_tokens
        
        logger.info(
            f"📊 [TOKEN BUDGET] Base: {base_tokens:,}, Query: {query_tokens:,}, "
            f"Available for docs: {available_doc_tokens:,} tokens (max: {MAX_SAFE_TOKENS:,})"
        )
        
        # MAX_SAFE_TOKENS를 지역 변수로 사용
        max_safe_tokens = MAX_SAFE_TOKENS

        structured_docs = context.get("structured_documents", {})
        document_count = context.get("document_count", 0)

        # 문서 정규화 및 최적화 (개선: 중복 제거 강화)
        normalized_docs = []
        seen_doc_ids = set()  # 중복 체크용
        
        logger.debug(f"🔍 [DEBUG] structured_docs type: {type(structured_docs)}, value: {structured_docs}")
        
        if isinstance(structured_docs, dict):
            raw_documents = structured_docs.get("documents", [])
            doc_count = len(raw_documents) if raw_documents else 0
            
            logger.info(f"📋 [FINAL PROMPT] Processing {doc_count} raw documents from structured_documents")
            logger.info(f"📋 [FINAL PROMPT] structured_docs keys: {list(structured_docs.keys()) if isinstance(structured_docs, dict) else 'N/A'}")
            if doc_count == 0:
                logger.warning(f"⚠️ [FINAL PROMPT] structured_docs has no documents! structured_docs={structured_docs}")

            # 🔥 개선: retrieved_docs에서 원본 문서의 type 필드를 복원
            retrieved_docs = context.get("retrieved_docs", [])
            if retrieved_docs:
                # retrieved_docs를 content/source 기반으로 매핑
                doc_map = {}
                for orig_doc in retrieved_docs:
                    if isinstance(orig_doc, dict):
                        content = orig_doc.get("content") or orig_doc.get("text", "")
                        source = orig_doc.get("source", "")
                        doc_id = orig_doc.get("id") or orig_doc.get("document_id")
                        # 여러 키로 매핑 가능하도록
                        if content:
                            doc_map[content[:100]] = orig_doc
                            # 🔥 개선: 짧은 content의 경우 더 짧은 키로도 매핑
                            if len(content) > 10:
                                doc_map[content[:50]] = orig_doc
                        if source:
                            doc_map[source] = orig_doc
                        if doc_id:
                            doc_map[str(doc_id)] = orig_doc
                        # 🔥 개선: metadata에서도 id 추출 시도
                        metadata = orig_doc.get("metadata", {})
                        if isinstance(metadata, dict):
                            metadata_id = metadata.get("id") or metadata.get("document_id")
                            if metadata_id:
                                doc_map[str(metadata_id)] = orig_doc
                        # 🔥 개선: metadata에서도 id 추출 시도
                        metadata = orig_doc.get("metadata", {})
                        if isinstance(metadata, dict):
                            metadata_id = metadata.get("id") or metadata.get("document_id")
                            if metadata_id:
                                doc_map[str(metadata_id)] = orig_doc
                
                # raw_documents의 type 필드 복원
                for doc in raw_documents:
                    if not isinstance(doc, dict):
                        continue
                    # content나 source로 원본 문서 찾기
                    content = doc.get("content", "") or doc.get("text", "")
                    source = doc.get("source", "")
                    doc_id = doc.get("document_id") or doc.get("id")
                    
                    orig_doc = None
                    # 🔥 개선: 여러 방법으로 원본 문서 찾기
                    if content:
                        orig_doc = doc_map.get(content[:100])
                    if not orig_doc and source:
                        orig_doc = doc_map.get(source)
                    if not orig_doc and doc_id:
                        orig_doc = doc_map.get(str(doc_id))
                    # 🔥 개선: content의 일부로도 매칭 시도 (부분 일치)
                    if not orig_doc and content and len(content) > 10:
                        for key, orig in doc_map.items():
                            if isinstance(key, str) and (content[:30] in key or key[:30] in content):
                                orig_doc = orig
                                break
                    # 🔥 개선: content의 일부로도 매칭 시도 (짧은 content의 경우)
                    if not orig_doc and content and len(content) > 10:
                        for key, orig in doc_map.items():
                            if isinstance(key, str) and content[:50] in key:
                                orig_doc = orig
                                break
                    
                    # 원본 문서에서 type 필드 및 법률 정보 복원
                    if orig_doc and isinstance(orig_doc, dict):
                        orig_type = orig_doc.get("type") or orig_doc.get("source_type")
                        if orig_type and orig_type not in ["", "unknown", "other"]:
                            doc["type"] = orig_type
                            # source_type도 복원
                            if orig_doc.get("source_type") and orig_doc.get("source_type") not in ["", "unknown", "other"]:
                                doc["source_type"] = orig_doc.get("source_type")
                            
                            # 🔥 개선: 법률 정보 필드도 복원
                            if orig_type in ["statute_article", "statute"]:
                                if orig_doc.get("statute_name") and not doc.get("statute_name"):
                                    doc["statute_name"] = orig_doc.get("statute_name")
                                if orig_doc.get("law_name") and not doc.get("law_name"):
                                    doc["law_name"] = orig_doc.get("law_name")
                                if orig_doc.get("article_no") and not doc.get("article_no"):
                                    doc["article_no"] = orig_doc.get("article_no")
                                if orig_doc.get("article_number") and not doc.get("article_number"):
                                    doc["article_number"] = orig_doc.get("article_number")
                            
                            # metadata에도 복원
                            if "metadata" not in doc:
                                doc["metadata"] = {}
                            if not isinstance(doc["metadata"], dict):
                                doc["metadata"] = {}
                            doc["metadata"]["type"] = orig_type
                            if orig_doc.get("source_type"):
                                doc["metadata"]["source_type"] = orig_doc.get("source_type")
                            
                            # 🔥 개선: metadata에서도 법률 정보 복원
                            orig_metadata = orig_doc.get("metadata", {})
                            if isinstance(orig_metadata, dict):
                                if orig_metadata.get("statute_name") and not doc["metadata"].get("statute_name"):
                                    doc["metadata"]["statute_name"] = orig_metadata.get("statute_name")
                                if orig_metadata.get("law_name") and not doc["metadata"].get("law_name"):
                                    doc["metadata"]["law_name"] = orig_metadata.get("law_name")
                                if orig_metadata.get("article_no") and not doc["metadata"].get("article_no"):
                                    doc["metadata"]["article_no"] = orig_metadata.get("article_no")
                                if orig_metadata.get("article_number") and not doc["metadata"].get("article_number"):
                                    doc["metadata"]["article_number"] = orig_metadata.get("article_number")
                            
                            logger.debug(f"✅ [DOC TYPE RESTORE] retrieved_docs에서 type 및 법률 정보 복원: {orig_type} (source_type={orig_doc.get('source_type', 'N/A')}, statute_name={orig_doc.get('statute_name', 'N/A')}, law_name={orig_doc.get('law_name', 'N/A')}, article_no={orig_doc.get('article_no', 'N/A')})")
            
            # 🔥 개선: statute_id로 법령명 배치 조회 (성능 최적화)
            law_name_cache = self._batch_fetch_law_names(raw_documents)
            
            # 문서 정규화
            normalized_count = 0
            skipped_count = 0
            normalized_docs_temp = []
            
            for idx, doc in enumerate(raw_documents):
                logger.debug(f"🔍 [DEBUG] Processing doc {idx+1}/{doc_count}: type={type(doc)}, keys={list(doc.keys()) if isinstance(doc, dict) else 'N/A'}")
                # 🔥 개선: doc이 None이거나 dict가 아니면 스킵
                if not doc or not isinstance(doc, dict):
                    skipped_count += 1
                    logger.warning(f"⚠️ [DEBUG] Doc {idx+1} is not a valid dict, skipping")
                    continue
                normalized = self._normalize_document_fields(doc, law_name_cache=law_name_cache)
                if normalized and isinstance(normalized, dict):
                    normalized_docs_temp.append(normalized)
                    normalized_count += 1
                    logger.debug(f"✅ [DEBUG] Doc {idx+1} normalized successfully, has metadata: {bool(normalized.get('metadata'))}")
                else:
                    skipped_count += 1
                    logger.warning(f"⚠️ [DEBUG] Doc {idx+1} normalization returned None or invalid (skipped)")

            logger.info(f"📋 [FINAL PROMPT] Normalization: {normalized_count} succeeded, {skipped_count} skipped")
            
            # 🔥 강화된 중복 제거 적용 (프롬프트 실행 전)
            before_dedup = len(normalized_docs_temp)
            normalized_docs = self._enhanced_deduplicate_before_prompt(
                normalized_docs_temp,
                query=query,
                similarity_threshold=0.85,
                mmr_lambda=0.7
            )
            after_dedup = len(normalized_docs)
            
            if before_dedup != after_dedup:
                logger.info(
                    f"✅ [FINAL PROMPT] Enhanced deduplication: {before_dedup} → {after_dedup} documents "
                    f"({before_dedup - after_dedup} duplicates removed)"
                )

            # 문서 최적화 (중복 제거 및 정렬)
            before_optimize = len(normalized_docs)
            normalized_docs = self._optimize_documents_for_prompt(normalized_docs, query)
            after_optimize = len(normalized_docs)

            logger.info(
                f"📋 [FINAL PROMPT] Documents: raw={doc_count}, normalized={before_optimize}, "
                f"after_optimize={after_optimize} (duplicates removed)"
            )
            
            # 멀티 질의 메타데이터 확인 (디버깅)
            if normalized_docs:
                multi_query_count = 0
                for doc in normalized_docs:
                    if isinstance(doc, dict):
                        metadata = doc.get("metadata", {})
                        if isinstance(metadata, dict) and metadata.get("sub_query"):
                            multi_query_count += 1
                            logger.debug(f"🔍 [DEBUG] Found multi-query doc: sub_query='{metadata.get('sub_query')[:50]}...'")
                
                if multi_query_count > 0:
                    logger.info(f"✅ [FINAL PROMPT] Found {multi_query_count}/{len(normalized_docs)} documents with multi-query metadata")
                else:
                    logger.warning(f"⚠️ [FINAL PROMPT] No multi-query metadata found in {len(normalized_docs)} documents")
            else:
                logger.warning(f"⚠️ [FINAL PROMPT] normalized_docs is empty after processing {doc_count} raw documents")
        else:
            logger.warning(f"⚠️ [FINAL PROMPT] structured_documents is not a dict: {type(structured_docs)}")
            logger.warning(f"⚠️ [FINAL PROMPT] structured_docs value: {structured_docs}")
            logger.warning(f"⚠️ [FINAL PROMPT] context keys: {list(context.keys()) if isinstance(context, dict) else 'N/A'}")
            # 폴백: context에서 직접 retrieved_docs 사용
            retrieved_docs = context.get("retrieved_docs", [])
            if retrieved_docs and len(retrieved_docs) > 0:
                logger.info(f"🔄 [FALLBACK] Using retrieved_docs directly: {len(retrieved_docs)} docs")
                # retrieved_docs를 normalized_docs로 변환
                for idx, doc in enumerate(retrieved_docs[:5], 1):
                    try:
                        if isinstance(doc, dict):
                            normalized = self._normalize_document_fields(doc)
                            if normalized:
                                doc_id = self._generate_document_id(normalized)
                                if doc_id not in seen_doc_ids:
                                    seen_doc_ids.add(doc_id)
                                    normalized_docs.append(normalized)
                                    logger.debug(f"✅ [FALLBACK] Doc {idx} normalized from retrieved_docs")
                    except Exception as e:
                        logger.warning(f"⚠️ [FALLBACK] Failed to normalize doc {idx}: {e}")
                        continue

        # base_prompt에 이미 문서가 포함되어 있는지 확인 (개선: 실제 문서 내용 존재 여부 확인)
        # 멀티 질의 검색 결과가 있는 경우 항상 문서 섹션 생성
        has_docs_in_base = False
        has_multi_query_results = False
        
        # 멀티 질의 검색 결과 확인 (metadata에 sub_query가 있는 경우)
        if normalized_docs:
            for doc in normalized_docs:
                if isinstance(doc, dict) and isinstance(doc.get("metadata"), dict):
                    if doc.get("metadata", {}).get("sub_query"):
                        has_multi_query_results = True
                        break
        
        if normalized_docs:
            # 멀티 질의 검색 결과가 있으면 항상 문서 섹션 생성
            if has_multi_query_results:
                has_docs_in_base = False
                logger.info("✅ [FINAL PROMPT] Multi-query results detected, will create documents section")
            else:
                # structured_documents에 실제 문서가 있으면 base_prompt에 문서가 있어도 문서 섹션 생성
                # 단, base_prompt에 실제 문서 내용이 포함되어 있는지 확인
                base_prompt_has_actual_content = False
                if normalized_docs:
                    # 첫 번째 문서의 일부 내용이 base_prompt에 있는지 확인
                    # TASK 3: None 체크 추가
                    first_doc = normalized_docs[0] if normalized_docs else None
                    if first_doc and isinstance(first_doc, dict):
                        first_doc_content = first_doc.get("content", "")[:100]
                        if first_doc_content and len(first_doc_content) > 10:
                            base_prompt_has_actual_content = first_doc_content in base_prompt
                
                if base_prompt_has_actual_content:
                    # base_prompt에 실제 문서 내용이 있으면 중복 제거만 수행
                    base_prompt = self._remove_duplicate_document_sections(base_prompt)
                    has_docs_in_base = True
                else:
                    # base_prompt에 실제 문서 내용이 없으면 문서 섹션 생성
                    has_docs_in_base = False
        else:
            # normalized_docs가 없으면 기존 로직 사용
            has_docs_in_base = any(pattern in base_prompt for pattern in [
                "검색된 법률 문서", "제공된 법률 문서", "검색된 판례 문서",
                "## 검색된 법률 문서", "## 제공된 법률 문서",
            ])
            if has_docs_in_base:
                base_prompt = self._remove_duplicate_document_sections(base_prompt)

        # 문서 섹션 구성 (토큰 제한 적용)
        documents_section = ""
        # 개선: normalized_docs가 있으면 항상 문서 섹션 생성 (base_prompt에 문서가 있어도)
        # 단, base_prompt에 실제 문서 내용이 포함되어 있으면 중복 제거만 수행
        logger.debug(f"🔍 [DOCUMENTS SECTION] normalized_docs={len(normalized_docs) if normalized_docs else 0}, "
                    f"has_docs_in_base={has_docs_in_base}, has_multi_query_results={has_multi_query_results}")
        
        # 개선: normalized_docs가 있으면 항상 문서 섹션 생성
        if normalized_docs:
            logger.info(f"✅ [DOCUMENTS SECTION] Creating documents section: "
                       f"normalized_docs={len(normalized_docs)}, has_docs_in_base={has_docs_in_base}, "
                       f"has_multi_query_results={has_multi_query_results}")
            # 🔥 개선: 관련도 기준 정렬 (법률/판례/기타 각각 할당량 보장)
            # 🔥 개선: 문서 타입 분류 전에 메타데이터 복원 시도
            # 🔥 개선: None 필터링 강화
            normalized_docs = [doc for doc in normalized_docs if doc and isinstance(doc, dict)]
            for doc in normalized_docs:
                if not isinstance(doc, dict):
                    continue
                
                # type이 없거나 unknown/other이면 복원 시도
                current_type = doc.get("type", "").lower() if doc.get("type") else ""
                if not doc.get("type") or current_type in ["unknown", "other"]:
                    # metadata에서 복원
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, dict):
                        metadata_type = metadata.get("type")
                        if metadata_type and metadata_type not in ["unknown", "other"]:
                            doc["type"] = metadata_type
                            logger.debug(f"🔍 [DOC TYPE RESTORE] metadata에서 type 복원: {metadata_type}")
                    
                    # 복원 실패 시 content 기반 추론으로 type 설정
                    if not doc.get("type") or doc.get("type", "").lower() in ["unknown", "other"]:
                        inferred_type = self._classify_document_type(doc)
                        # 🔥 개선: "other"가 아닌 모든 경우 저장 (statute, precedent 모두)
                        if inferred_type:
                            # "statute" → "statute_article", "precedent" → "precedent_content" 변환
                            type_mapping = {
                                "statute": "statute_article",
                                "precedent": "precedent_content",
                                "other": "unknown"
                            }
                            mapped_type = type_mapping.get(inferred_type, inferred_type)
                            doc["type"] = mapped_type
                            # metadata에도 저장
                            if "metadata" not in doc:
                                doc["metadata"] = {}
                            if not isinstance(doc["metadata"], dict):
                                doc["metadata"] = {}
                            doc["metadata"]["type"] = mapped_type
                            logger.debug(f"🔍 [DOC TYPE INFERENCE] content 기반 추론으로 type 설정: {mapped_type} (원본: {inferred_type})")
            
            # 문서를 법률, 판례, 기타로 분리
            statute_docs = []
            precedent_docs = []
            other_docs = []
            
            for doc in normalized_docs:
                # 🔥 개선: 기존 type 필드가 있으면 우선 사용 (statute_article, precedent_content 등)
                current_type = doc.get("type", "")
                if current_type and current_type not in ["", "unknown", "other"]:
                    # 이미 올바른 타입이 있으면 _classify_document_type 호출 생략
                    doc_type = current_type
                    logger.debug(f"✅ [DOC TYPE SAVE] 기존 type 필드 사용: {doc_type}")
                else:
                    doc_type = self._classify_document_type(doc)
                    # 🔥 개선: 분류 결과를 문서에 저장 (statute → statute_article, precedent → precedent_content 변환)
                    if doc_type != "other":
                        # "statute" → "statute_article", "precedent" → "precedent_content" 변환
                        type_mapping = {
                            "statute": "statute_article",
                            "precedent": "precedent_content",
                            "other": "unknown"
                        }
                        mapped_type = type_mapping.get(doc_type, doc_type)
                        # 분류된 타입을 문서에 저장
                        doc["type"] = mapped_type
                        # metadata에도 저장
                        if "metadata" not in doc:
                            doc["metadata"] = {}
                        if not isinstance(doc["metadata"], dict):
                            doc["metadata"] = {}
                        doc["metadata"]["type"] = mapped_type
                        logger.debug(f"🔍 [DOC TYPE SAVE] 분류 결과 저장: {mapped_type} (원본: {doc_type})")
                        doc_type = mapped_type
                
                # 🔥 개선: statute_article, precedent_content도 포함
                if doc_type in ["statute", "statute_article"]:
                    statute_docs.append(doc)
                elif doc_type in ["precedent", "precedent_content"]:
                    precedent_docs.append(doc)
                else:
                    other_docs.append(doc)
            
            # 각 타입별로 관련도 순 정렬
            sorted_statute_docs = sorted(
                statute_docs,
                key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0) or 0.0,
                reverse=True
            )
            
            sorted_precedent_docs = sorted(
                precedent_docs,
                key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0) or 0.0,
                reverse=True
            )
            
            sorted_other_docs = sorted(
                other_docs,
                key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0) or 0.0,
                reverse=True
            )
            
            # 🔥 개선: 각 타입별 할당량 설정 (환경 변수로 조정 가능)
            import os
            # 법률 조문 할당량
            STATUTE_MIN = int(os.getenv("PROMPT_STATUTE_MIN", "2"))  # 최소 2개
            STATUTE_MAX = int(os.getenv("PROMPT_STATUTE_MAX", "5"))  # 최대 5개
            
            # 판례 할당량
            PRECEDENT_MIN = int(os.getenv("PROMPT_PRECEDENT_MIN", "2"))  # 최소 2개
            PRECEDENT_MAX = int(os.getenv("PROMPT_PRECEDENT_MAX", "5"))  # 최대 5개
            
            # 기타 문서 할당량
            OTHER_MIN = int(os.getenv("PROMPT_OTHER_MIN", "1"))  # 최소 1개
            OTHER_MAX = int(os.getenv("PROMPT_OTHER_MAX", "3"))  # 최대 3개
            
            logger.info(
                f"📊 [DOCUMENT TYPE SPLIT] 법률={len(statute_docs)}, 판례={len(precedent_docs)}, 기타={len(other_docs)}, "
                f"할당량: 법률[{STATUTE_MIN}-{STATUTE_MAX}], 판례[{PRECEDENT_MIN}-{PRECEDENT_MAX}], 기타[{OTHER_MIN}-{OTHER_MAX}]"
            )
            
            # 토큰 제한 내에서 문서 선택 및 축약
            selected_docs = []
            current_doc_tokens = 0
            
            # 토큰 할당량 계산 (각 타입별로 비율 할당)
            total_quota = STATUTE_MAX + PRECEDENT_MAX + OTHER_MAX
            statute_token_quota = int(available_doc_tokens * (STATUTE_MAX / total_quota)) if total_quota > 0 else available_doc_tokens // 3
            precedent_token_quota = int(available_doc_tokens * (PRECEDENT_MAX / total_quota)) if total_quota > 0 else available_doc_tokens // 3
            other_token_quota = available_doc_tokens - statute_token_quota - precedent_token_quota
            
            # 1. 법률 조문 선택 (최소/최대 할당량 보장)
            # 🔥 개선: 조문은 원문 그대로 사용 (축약 금지)
            import os
            STATUTE_PRESERVE_FULL = os.getenv("PROMPT_STATUTE_PRESERVE_FULL", "true").lower() == "true"
            STATUTE_MAX_LENGTH = int(os.getenv("PROMPT_STATUTE_MAX_LENGTH", "5000"))  # 조문 최대 길이 (원문 보존 시)
            
            statute_selected = []
            statute_tokens_used = 0
            
            for doc in sorted_statute_docs:
                if len(statute_selected) >= STATUTE_MAX:
                    break
                
                doc_content = doc.get("content", "") or doc.get("text", "")
                doc_tokens = self._estimate_tokens(doc_content)
                
                # 🔥 개선: 조문은 원문 그대로 사용 (축약하지 않음)
                if STATUTE_PRESERVE_FULL:
                    # 조문 원문 보존 - 축약하지 않고 그대로 사용
                    # 단, 매우 긴 조문은 최대 길이 제한 적용
                    if len(doc_content) > STATUTE_MAX_LENGTH:
                        logger.warning(
                            f"⚠️ [STATUTE] 조문이 최대 길이({STATUTE_MAX_LENGTH}자)를 초과합니다. "
                            f"원문 보존 모드이므로 조문 수를 줄이는 방향으로 조정합니다."
                        )
                        # 조문이 너무 길면 이 조문은 건너뛰고 다음 조문으로
                        continue
                    
                    # 원문 그대로 사용 (축약하지 않음)
                    doc["content"] = doc_content
                    doc["preserved_full"] = True
                else:
                    # 기존 로직 (축약 허용 모드 - 하위 호환성)
                    max_tokens_per_doc = min(2000, statute_token_quota // max(1, STATUTE_MAX))
                    
                    if doc_tokens > max_tokens_per_doc:
                        max_chars = int(max_tokens_per_doc * 2.5)
                        doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                        doc_tokens = self._estimate_tokens(doc_content)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                
                # 토큰 제한 체크 (원문 보존 모드에서는 조문 수로 제어)
                if STATUTE_PRESERVE_FULL:
                    # 원문 보존 모드: 토큰 제한보다는 조문 수로 제어
                    if statute_tokens_used + doc_tokens <= statute_token_quota * 1.5:  # 여유 있게 허용
                        statute_selected.append(doc)
                        statute_tokens_used += doc_tokens
                    else:
                        # 토큰이 부족하면 조문 수를 줄이는 방향으로 조정
                        logger.info(
                            f"ℹ️ [STATUTE] 토큰 제한으로 인해 조문 선택 중단. "
                            f"현재 {len(statute_selected)}개 조문 선택됨."
                        )
                        break
                else:
                    # 기존 로직
                    if statute_tokens_used + doc_tokens <= statute_token_quota:
                        statute_selected.append(doc)
                        statute_tokens_used += doc_tokens
                    else:
                        remaining_tokens = statute_token_quota - statute_tokens_used
                        if remaining_tokens > 500 and len(statute_selected) < STATUTE_MIN:
                            max_chars = int(remaining_tokens * 2.5)
                            doc_content = self._smart_truncate_document(doc.get("content", "") or doc.get("text", ""), max_chars, query)
                            doc["content"] = doc_content
                            doc["truncated"] = True
                            statute_selected.append(doc)
                        break
            
            # 최소 할당량 보장 (토큰이 부족해도 최소한은 포함)
            if len(statute_selected) < STATUTE_MIN and len(sorted_statute_docs) >= STATUTE_MIN:
                # 최소 할당량만큼 추가
                for doc in sorted_statute_docs[len(statute_selected):STATUTE_MIN]:
                    if doc not in statute_selected:
                        doc_content = doc.get("content", "") or doc.get("text", "")
                        if STATUTE_PRESERVE_FULL:
                            # 원문 보존 모드: 원문 그대로 사용
                            doc["content"] = doc_content
                            doc["preserved_full"] = True
                        else:
                            # 기존 로직: 축약
                            max_chars = 1000
                            if len(doc_content) > max_chars:
                                doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                                doc["content"] = doc_content
                                doc["truncated"] = True
                        statute_selected.append(doc)
            
            selected_docs.extend(statute_selected)
            current_doc_tokens += statute_tokens_used
            
            # 2. 판례 선택 (최소/최대 할당량 보장)
            # 🔥 개선: 판례는 길이에 따라 요약 또는 축약
            PRECEDENT_SUMMARY_THRESHOLD = int(os.getenv("PRECEDENT_SUMMARY_THRESHOLD", "2000"))  # 판례 요약 임계값 (문자 수)
            PRECEDENT_SUMMARY_MAX_LENGTH = int(os.getenv("PRECEDENT_SUMMARY_MAX_LENGTH", "1000"))  # 판례 요약 최대 길이
            PRECEDENT_SUMMARY_USE_LLM = os.getenv("PRECEDENT_SUMMARY_USE_LLM", "true").lower() == "true"
            
            precedent_selected = []
            precedent_tokens_used = 0
            
            for doc in sorted_precedent_docs:
                if len(precedent_selected) >= PRECEDENT_MAX:
                    break
                
                doc_content = doc.get("content", "") or doc.get("text", "")
                doc_length = len(doc_content)
                doc_tokens = self._estimate_tokens(doc_content)
                
                # 🔥 개선: 판례는 길이에 따라 요약 또는 축약
                if doc_length > PRECEDENT_SUMMARY_THRESHOLD and PRECEDENT_SUMMARY_USE_LLM:
                    # 긴 판례는 LLM 요약 사용
                    try:
                        summary_result = self._summarize_precedent(doc, query)
                        if summary_result and summary_result.get("summary"):
                            doc["content"] = summary_result["summary"]
                            doc["summary_data"] = summary_result
                            doc["summarized"] = True
                            doc_tokens = self._estimate_tokens(summary_result["summary"])
                            logger.info(
                                f"✅ [PRECEDENT SUMMARY] 판례 요약 완료: "
                                f"원본={doc_length}자 → 요약={len(summary_result['summary'])}자 "
                                f"(축약률={(1 - len(summary_result['summary'])/doc_length)*100:.1f}%)"
                            )
                        else:
                            # 요약 실패 시 스마트 축약
                            max_tokens_per_doc = min(1500, precedent_token_quota // max(1, PRECEDENT_MAX))
                            max_chars = int(max_tokens_per_doc * 2.5)
                            doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                            doc_tokens = self._estimate_tokens(doc_content)
                            doc["content"] = doc_content
                            doc["truncated"] = True
                            logger.warning(f"⚠️ [PRECEDENT SUMMARY] 요약 실패, 스마트 축약 사용: {doc_length}자 → {len(doc_content)}자")
                    except Exception as e:
                        # 요약 실패 시 스마트 축약
                        logger.warning(f"⚠️ [PRECEDENT SUMMARY] 요약 중 오류 발생: {e}, 스마트 축약 사용")
                        max_tokens_per_doc = min(1500, precedent_token_quota // max(1, PRECEDENT_MAX))
                        max_chars = int(max_tokens_per_doc * 2.5)
                        doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                        doc_tokens = self._estimate_tokens(doc_content)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                else:
                    # 짧은 판례는 스마트 축약만 수행
                    max_tokens_per_doc = min(1500, precedent_token_quota // max(1, PRECEDENT_MAX))
                    
                    if doc_tokens > max_tokens_per_doc:
                        max_chars = int(max_tokens_per_doc * 2.5)
                        doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                        doc_tokens = self._estimate_tokens(doc_content)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                
                if precedent_tokens_used + doc_tokens <= precedent_token_quota:
                    precedent_selected.append(doc)
                    precedent_tokens_used += doc_tokens
                else:
                    remaining_tokens = precedent_token_quota - precedent_tokens_used
                    if remaining_tokens > 500 and len(precedent_selected) < PRECEDENT_MIN:
                        max_chars = int(remaining_tokens * 2.5)
                        doc_content = self._smart_truncate_document(doc.get("content", "") or doc.get("text", ""), max_chars, query)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                        precedent_selected.append(doc)
                    break
            
            # 최소 할당량 보장
            if len(precedent_selected) < PRECEDENT_MIN and len(sorted_precedent_docs) >= PRECEDENT_MIN:
                for doc in sorted_precedent_docs[len(precedent_selected):PRECEDENT_MIN]:
                    if doc not in precedent_selected:
                        doc_content = doc.get("content", "") or doc.get("text", "")
                        doc_length = len(doc_content)
                        
                        # 최소 할당량 문서도 요약 적용
                        if doc_length > PRECEDENT_SUMMARY_THRESHOLD and PRECEDENT_SUMMARY_USE_LLM:
                            try:
                                summary_result = self._summarize_precedent(doc, query)
                                if summary_result and summary_result.get("summary"):
                                    doc["content"] = summary_result["summary"]
                                    doc["summary_data"] = summary_result
                                    doc["summarized"] = True
                                else:
                                    # 요약 실패 시 축약
                                    max_chars = 1000
                                    if len(doc_content) > max_chars:
                                        doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                                        doc["content"] = doc_content
                                        doc["truncated"] = True
                            except Exception as e:
                                logger.warning(f"⚠️ [PRECEDENT SUMMARY] 최소 할당량 문서 요약 실패: {e}")
                                max_chars = 1000
                                if len(doc_content) > max_chars:
                                    doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                                    doc["content"] = doc_content
                                    doc["truncated"] = True
                        else:
                            max_chars = 1000
                            if len(doc_content) > max_chars:
                                doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                                doc["content"] = doc_content
                                doc["truncated"] = True
                        precedent_selected.append(doc)
            
            selected_docs.extend(precedent_selected)
            current_doc_tokens += precedent_tokens_used
            
            # 3. 기타 문서 선택 (남은 토큰 사용)
            remaining_tokens = available_doc_tokens - current_doc_tokens
            other_selected = []
            other_tokens_used = 0
            
            for doc in sorted_other_docs:
                if len(other_selected) >= OTHER_MAX:
                    break
                
                doc_content = doc.get("content", "") or doc.get("text", "")
                doc_tokens = self._estimate_tokens(doc_content)
                
                max_tokens_per_doc = min(1000, remaining_tokens // max(1, OTHER_MAX - len(other_selected)))
                
                if doc_tokens > max_tokens_per_doc:
                    max_chars = int(max_tokens_per_doc * 2.5)
                    doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                    doc_tokens = self._estimate_tokens(doc_content)
                    doc["content"] = doc_content
                    doc["truncated"] = True
                
                if other_tokens_used + doc_tokens <= remaining_tokens:
                    other_selected.append(doc)
                    other_tokens_used += doc_tokens
                else:
                    remaining_tokens_final = remaining_tokens - other_tokens_used
                    if remaining_tokens_final > 500 and len(other_selected) < OTHER_MIN:
                        max_chars = int(remaining_tokens_final * 2.5)
                        doc_content = self._smart_truncate_document(doc.get("content", "") or doc.get("text", ""), max_chars, query)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                        other_selected.append(doc)
                    break
            
            # 최소 할당량 보장
            if len(other_selected) < OTHER_MIN and len(sorted_other_docs) >= OTHER_MIN:
                for doc in sorted_other_docs[len(other_selected):OTHER_MIN]:
                    if doc not in other_selected:
                        doc_content = doc.get("content", "") or doc.get("text", "")
                        max_chars = 800
                        if len(doc_content) > max_chars:
                            doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                            doc["content"] = doc_content
                            doc["truncated"] = True
                        other_selected.append(doc)
            
            selected_docs.extend(other_selected)
            current_doc_tokens += other_tokens_used
            
            sorted_docs = selected_docs

            if sorted_docs:
                documents_section = self._build_documents_section(sorted_docs, query)

                # 타입별 개수 계산
                statute_count = len([d for d in sorted_docs if self._classify_document_type(d) == "statute"])
                precedent_count = len([d for d in sorted_docs if self._classify_document_type(d) == "precedent"])
                other_count = len([d for d in sorted_docs if self._classify_document_type(d) == "other"])
                
                logger.info(
                    f"✅ [FINAL PROMPT] Added {len(sorted_docs)} documents "
                    f"(statute: {statute_count}, precedent: {precedent_count}, other: {other_count}, "
                    f"tokens: {current_doc_tokens:,}/{available_doc_tokens:,}, "
                    f"multi_query={has_multi_query_results})"
                )
            else:
                logger.warning(f"⚠️ [DOCUMENTS SECTION] No documents selected from {len(normalized_docs)} normalized docs")

        # 폴백 처리: documents_section이 없을 때 (개선: structured_documents 우선 사용)
        # 멀티 질의 검색 결과가 있는 경우 항상 문서 섹션 생성
        if not documents_section:
            # 우선순위 1: structured_documents에서 직접 생성 시도
            if normalized_docs:
                sorted_docs = sorted(
                    normalized_docs,
                    key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )[:5]
                
                if sorted_docs:
                    documents_section = self._build_documents_section(sorted_docs, query)
                    
                    logger.info(f"✅ [FINAL PROMPT] Created documents_section from normalized_docs ({len(sorted_docs)} docs)")
            
            # 멀티 질의 검색 결과 확인 (metadata에 sub_query가 있는 경우)
            if not documents_section and normalized_docs:
                multi_query_docs = [doc for doc in normalized_docs 
                                  if isinstance(doc, dict) and 
                                  isinstance(doc.get("metadata"), dict) and 
                                  doc.get("metadata", {}).get("sub_query")]
                if multi_query_docs:
                    # 멀티 질의 결과가 있으면 관련도 순으로 정렬하여 문서 섹션 생성
                    sorted_multi_docs = sorted(
                        multi_query_docs,
                        key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                        reverse=True
                    )[:5]
                    
                    if sorted_multi_docs:
                        documents_section = self._build_documents_section(sorted_multi_docs, query)
                        logger.info(f"✅ [FINAL PROMPT] Created documents_section from multi-query results ({len(sorted_multi_docs)} docs)")
            
            # 🔥 개선: 우선순위 2: retrieved_docs 직접 사용 (structured_documents가 없을 때)
            if not documents_section:
                retrieved_docs = context.get("retrieved_docs", [])
                structured_docs = context.get("structured_documents", {})
                
                # 🔥 개선: 폴백 처리 로깅 강화
                logger.warning(
                    f"⚠️ [FALLBACK] documents_section is empty. "
                    f"Attempting fallback: retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}, "
                    f"structured_docs={len(structured_docs.get('documents', [])) if isinstance(structured_docs, dict) else 0}"
                )
                
                if retrieved_docs and isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                    # retrieved_docs를 normalized_docs 형식으로 변환
                    fallback_docs = []
                    for idx, doc in enumerate(retrieved_docs[:5], 1):  # 최대 5개만 사용
                        try:
                            if isinstance(doc, dict):
                                normalized = self._normalize_document_fields(doc)
                                if normalized:
                                    # 🔥 개선: relevance_score 보존 (이중 보장)
                                    if "relevance_score" not in normalized or normalized.get("relevance_score", 0.0) == 0.0:
                                        if "relevance_score" in doc:
                                            normalized["relevance_score"] = doc.get("relevance_score", 0.0)
                                        elif "score" in doc:
                                            normalized["relevance_score"] = doc.get("score", 0.0)
                                        elif "final_weighted_score" in doc:
                                            normalized["relevance_score"] = doc.get("final_weighted_score", 0.0)
                                    fallback_docs.append(normalized)
                        except Exception as e:
                            logger.warning(
                                f"⚠️ [FALLBACK] Failed to normalize doc {idx}: {e}. "
                                f"Skipping this document."
                            )
                            continue
                    
                    # 🔥 개선: relevance_score 기준으로 정렬
                    if fallback_docs:
                        fallback_docs = sorted(
                            fallback_docs,
                            key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                            reverse=True
                        )
                        documents_section = self._build_documents_section(fallback_docs, query)
                        logger.info(
                            f"✅ [FINAL PROMPT] Created documents_section from retrieved_docs fallback "
                            f"({len(fallback_docs)}/{len(retrieved_docs)} docs processed)"
                        )
                    else:
                        logger.warning(
                            f"⚠️ [FALLBACK] retrieved_docs fallback failed: "
                            f"{len(retrieved_docs)} docs but none normalized successfully"
                        )
            
            # 우선순위 3: prompt_optimized_text 사용
            if not documents_section:
                prompt_optimized_text = context.get("prompt_optimized_text", "")
                if prompt_optimized_text and len(prompt_optimized_text.strip()) > 100:
                    documents_section = "\n\n## 검색된 참고 문서\n\n"
                    documents_section += prompt_optimized_text[:5000] + ("..." if len(prompt_optimized_text) > 5000 else "")
                    documents_section += "\n\n"
                    logger.info("✅ [FINAL PROMPT] Added prompt_optimized_text as fallback")
            
            # 우선순위 4: context_text 사용
            if not documents_section:
                context_text = context.get("context", "")
                if context_text and len(context_text.strip()) > 100 and document_count > 0:
                    documents_section = "\n\n## 검색된 참고 문서\n\n"
                    documents_section += context_text[:5000] + ("..." if len(context_text) > 5000 else "")
                    documents_section += "\n\n"
                    logger.info("✅ [FINAL PROMPT] Added context_text as fallback")

        # 🔥 개선: 문서 섹션 생성 실패 시 상세 로깅
        if not documents_section or len(documents_section.strip()) == 0:
            structured_docs = context.get("structured_documents", {})
            retrieved_docs = context.get("retrieved_docs", [])
            normalized_docs_count = len(normalized_docs) if normalized_docs else 0
            
            logger.error(
                f"❌ [FINAL PROMPT] documents_section is empty after all fallback attempts! "
                f"Context: normalized_docs={normalized_docs_count}, "
                f"structured_docs={len(structured_docs.get('documents', [])) if isinstance(structured_docs, dict) else 0}, "
                f"retrieved_docs={len(retrieved_docs) if retrieved_docs else 0}, "
                f"prompt_optimized_text={len(context.get('prompt_optimized_text', ''))}, "
                f"context_text={len(context.get('context', ''))}"
            )
        
        # 검색 결과가 없을 때 base_prompt 단순화
        has_no_documents = (
            not documents_section or len(documents_section.strip()) == 0
        ) and (not normalized_docs or len(normalized_docs) == 0)

        if has_no_documents:
            simplified_base = self._simplify_prompt_for_no_results(base_prompt)
            logger.info("📝 [PROMPT SIMPLIFICATION] Simplified prompt for no search results")
        else:
            # base_prompt 간소화 (불필요한 섹션 제거)
            simplified_base = self._simplify_base_prompt(base_prompt)
            # base_prompt에서 중복된 Citation 요구사항 제거
            simplified_base = self._remove_duplicate_citation_requirements(simplified_base)

        # 최종 프롬프트 구성 (개선: documents_section이 없으면 경고 및 폴백)
        if not documents_section or len(documents_section.strip()) == 0:
            logger.warning(
                f"⚠️ [FINAL PROMPT] documents_section is empty! "
                f"normalized_docs={len(normalized_docs) if normalized_docs else 0}, "
                f"structured_docs={len(structured_docs.get('documents', [])) if isinstance(structured_docs, dict) else 0}"
            )
            # 폴백: 최소한의 문서 섹션 생성
            if normalized_docs and len(normalized_docs) > 0:
                logger.info(f"🔄 [FALLBACK] Creating minimal documents section from {len(normalized_docs)} normalized_docs")
                fallback_docs = normalized_docs[:3]  # 최대 3개만 사용
                documents_section = self._build_documents_section(fallback_docs, query)
                if documents_section:
                    logger.info(f"✅ [FALLBACK] Created fallback documents section ({len(fallback_docs)} docs)")
        
        # 🔥 개선: 프롬프트 구조 재구성 (문서 인용 지시 최상단, Few-shot 예시 추가)
        # 1. 핵심 지시사항 (최상단)
        # 2. Few-shot 예시
        # 3. 문서 목록
        # 4. 질문
        
        # 문서 수 확인 및 has_sufficient_docs 설정
        doc_count = len(normalized_docs) if normalized_docs else 0
        if not doc_count and isinstance(structured_docs, dict):
            doc_count = len(structured_docs.get("documents", []))
        has_sufficient_docs = (documents_section and len(documents_section.strip()) > 0) and doc_count > 0
        
        # 핵심 지시사항 구성
        if has_sufficient_docs:
            if doc_count >= 5:
                min_citations = 3
            elif doc_count >= 3:
                min_citations = 2
            else:
                min_citations = 1
            
            # 🔥 개선: 핵심 지시사항을 최상단에 배치 (문서 활용도 개선)
            core_instructions = f"""# ⚠️ 필수 지시사항 (반드시 준수) - 문서 활용도 0% 시 답변 거부

**🔥🔥🔥 CRITICAL: 문서 인용은 절대 필수입니다. 문서를 전혀 인용하지 않으면 답변이 즉시 거부되고 다시 생성됩니다. 🔥🔥🔥**

**⚠️ 경고**: 아래 '검색된 참고 문서' 섹션에 {doc_count}개 문서가 제공되었습니다. 이 문서들을 반드시 활용하여 답변을 작성하세요. 

**문서 인용 규칙 (절대 준수)**:
1. **답변에서 문서를 인용할 때는 반드시 `[문서 N]` 형식을 사용하세요** (N은 1부터 시작)
2. **최소 {min_citations}개 이상의 서로 다른 문서를 반드시 인용해야 합니다** - 검색된 {doc_count}개 문서 중 최소 {min_citations}개는 반드시 사용하세요
3. **답변 본문의 각 문단 끝에 `[문서 N]`을 배치하세요** - 문장 시작 부분이나 문장 중간에 배치하지 마세요
4. **문서 내용을 설명할 때는 반드시 해당 문서를 `[문서 N]` 형식으로 인용하세요** - 인용 없이 설명만 하면 안 됩니다
5. **문서를 전혀 인용하지 않으면 답변이 거부됩니다** - 반드시 최소 {min_citations}개 이상의 문서를 인용하세요

**✅ 올바른 예시**:
"계약 해지는 다음과 같은 사유로 가능합니다. 부당이득반환청구에서 법률상의 원인 없는 사유를 계약의 불성립, 취소, 무효, 해제 등으로 주장할 수 있습니다. [문서 1]

계약의 해제는 유효하게 성립된 계약의 효력을 당사자 일방의 의사표시에 의해 소급적으로 소멸시키는 것을 의미합니다. [문서 2]"

**❌ 잘못된 예시 (절대 금지)**:
- "계약 해지는 [문서 1]에 따르면 가능합니다." (문장 시작 부분에 인용)
- "[문서 1]은 통화옵션계약과 관련된..." (문장 시작 부분에 인용)
- "계약 해지는 다음과 같은 사유로 가능합니다." (인용 없음 - 즉시 거부)
- "민법 제544조에 따르면..." (법령 인용만 있고 문서 인용 없음 - 거부)

**⚠️ 최종 경고**: 아래 '검색된 참고 문서' 섹션에 {doc_count}개 문서가 제공되었습니다. 이 문서들을 반드시 활용하여 답변을 작성하세요. 문서를 전혀 인용하지 않으면 답변이 거부되고 다시 생성됩니다.

---
"""
            
            # 🔥 개선: Few-shot 예시 추가 (스트리밍 모드에 따라 분기)
            # 스트리밍 모드 확인 (context에서)
            is_streaming = context.get("is_streaming", False) or (
                isinstance(context.get("metadata"), dict) and context.get("metadata", {}).get("streaming_mode", False)
            )
            
            if is_streaming:
                # 스트리밍 모드: [END] 키워드 사용, 표 제거
                few_shot_example = f"""## 📋 답변 형식 예시 (반드시 이 형식을 정확히 따르세요)

**질문**: 계약 해지 사유에 대해 알려주세요

**검색된 문서**:
- [문서 1] 대법원 부당이득금 (2020다231928): 계약의 불성립, 취소, 무효, 해제 등이 법률상의 원인 없는 사유로 주장 가능
- [문서 2] 대법원 부당이득금 (2020다231928): 계약 해제는 유효하게 성립된 계약의 효력을 소급적으로 소멸
- [문서 3] 수원지방법원 건물인도 (2021다257255): 임대차계약 종료 사례

**답변** (반드시 이 형식을 따르세요):
계약 해지는 다음과 같은 사유로 가능합니다.

부당이득반환청구에서 법률상의 원인 없는 사유를 계약의 불성립, 취소, 무효, 해제 등으로 주장할 수 있습니다. [문서 1]

계약의 해제는 유효하게 성립된 계약의 효력을 당사자 일방의 의사표시에 의해 소급적으로 소멸시키는 것을 의미합니다. [문서 2]

임대차계약 사례에서도 확인할 수 있듯이, 계약 종료는 다양한 방식으로 이루어질 수 있습니다. [문서 3]

[END]

<metadata>
{{
    "document_usage": [
        {{
            "document_number": 1,
            "source": "대법원 부당이득금 (2020다231928)",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [150],
            "usage_rationale": "계약의 불성립, 취소, 무효, 해제 등이 법률상의 원인 없는 사유로 주장 가능"
        }},
        {{
            "document_number": 2,
            "source": "대법원 부당이득금 (2020다231928)",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [300],
            "usage_rationale": "계약 해제는 유효하게 성립된 계약의 효력을 소급적으로 소멸"
        }},
        {{
            "document_number": 3,
            "source": "수원지방법원 건물인도 (2021다257255)",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [450],
            "usage_rationale": "임대차계약 종료 사례"
        }}
    ],
    "coverage": {{
        "keyword_coverage": 0.85,
        "keyword_total": 20,
        "keyword_matched": 17,
        "citation_coverage": 1.0,
        "citation_count": 3,
        "citation_expected": 3,
        "document_usage_rate": 1.0,
        "documents_used": 3,
        "documents_total": 3,
        "documents_min_required": 2,
        "overall_coverage": 0.90
    }}
}}
</metadata>

**⚠️ 중요**: 위 예시처럼:
1. 답변 본문을 먼저 작성하고
2. 문단 끝에 `[문서 N]`을 배치하세요
3. 답변 본문이 끝나면 반드시 `[END]` 키워드를 작성하세요
4. `[END]` 이후에 `<metadata>` 태그만 작성하세요 (문서 근거 섹션은 작성하지 마세요)
5. `document_usage`의 각 항목에 `source`와 `usage_rationale`을 포함하세요 (이 정보로 문서 근거 목록이 자동 생성됩니다)
6. 최소 {min_citations}개 이상의 문서를 반드시 인용하세요

---
"""
            else:
                # 일반 모드: 기존 예시 유지 (표 포함)
                few_shot_example = f"""## 📋 답변 형식 예시 (반드시 이 형식을 정확히 따르세요)

**질문**: 계약 해지 사유에 대해 알려주세요

**검색된 문서**:
- [문서 1] 대법원 부당이득금 (2020다231928): 계약의 불성립, 취소, 무효, 해제 등이 법률상의 원인 없는 사유로 주장 가능
- [문서 2] 대법원 부당이득금 (2020다231928): 계약 해제는 유효하게 성립된 계약의 효력을 소급적으로 소멸
- [문서 3] 수원지방법원 건물인도 (2021다257255): 임대차계약 종료 사례

**답변** (반드시 이 형식을 따르세요):
계약 해지는 다음과 같은 사유로 가능합니다.

부당이득반환청구에서 법률상의 원인 없는 사유를 계약의 불성립, 취소, 무효, 해제 등으로 주장할 수 있습니다. [문서 1]

계약의 해제는 유효하게 성립된 계약의 효력을 당사자 일방의 의사표시에 의해 소급적으로 소멸시키는 것을 의미합니다. [문서 2]

임대차계약 사례에서도 확인할 수 있듯이, 계약 종료는 다양한 방식으로 이루어질 수 있습니다. [문서 3]

| 문서 번호 | 출처 | 핵심 근거 |
|-----------|------|----------|
| [문서 1] | 대법원 부당이득금 (2020다231928) | 계약의 불성립, 취소, 무효, 해제 등이 법률상의 원인 없는 사유로 주장 가능 |
| [문서 2] | 대법원 부당이득금 (2020다231928) | 계약 해제는 유효하게 성립된 계약의 효력을 소급적으로 소멸 |
| [문서 3] | 수원지방법원 건물인도 (2021다257255) | 임대차계약 종료 사례 |

**⚠️ 중요**: 위 예시처럼:
1. 문서 내용을 먼저 설명하고
2. 문단 끝에 `[문서 N]`을 배치하세요
3. 문장 시작 부분에 `[문서 N]`을 배치하지 마세요
4. 최소 {min_citations}개 이상의 문서를 반드시 인용하세요

---
"""
        else:
            core_instructions = """# ⚠️ 필수 지시사항

답변에서 문서를 인용할 때는 `[문서 N]` 형식을 사용하세요.

---
"""
            few_shot_example = ""
        
        # 🔥 개선: 단순화된 base_prompt 사용 (핵심 역할만)
        simplified_role = """# Role: 대한민국 법률 전문가 AI 어시스턴트

당신은 대한민국 법률 전문 상담 AI입니다. 법학 석사 이상의 전문 지식을 보유하고 있으며, 다양한 법률 분야에 대한 실무 경험을 갖춘 것처럼 행동합니다.

## 답변 원칙
- 정확한 법률 정보 제공
- 관련 판례와 법령 인용
- 실무적 관점 반영
- 자연스러운 대화형 어조

---
"""

        # 답변 생성 지시사항 섹션 생성 (개선: 질문 유형, 답변 생성 규칙)
        # 🔥 개선: answer_generation_instructions 최소화 (citation_requirement와 base_prompt에 이미 포함)
        answer_generation_instructions = ""
        
        # 단계별 답변 지침 구성 (개선: 답변 형식 가이드 상세화)
        # 🔥 CRITICAL: 문서가 3개 이상일 때만 문서별 근거 비교 표 포함
        doc_count = len(normalized_docs) if normalized_docs else 0
        has_sufficient_docs = documents_section and normalized_docs and doc_count >= 3
        
        # document_citation_section 초기화 (조건문에서 정의되므로)
        document_citation_section = ""
        
        # 🔥 개선: 문서 인용 규칙 및 중요 사항 섹션 생성 (표 작성 지침 포함)
        # 문서 활용도 개선: 최소 인용 수를 동적으로 설정
        # 🔥 CRITICAL: 문서가 1개 이상이면 무조건 인용 규칙 추가 (has_sufficient_docs 조건 제거)
        if normalized_docs and doc_count >= 1:
            # 문서 수에 따라 최소 인용 수 결정
            if doc_count >= 5:
                min_citations = 3
            elif doc_count >= 3:
                min_citations = 2
            else:
                min_citations = 1  # 문서가 1개여도 최소 1개는 인용 필수
            
            # 문서 인용 규칙 및 중요 사항 섹션 (스트리밍 모드에 따라 조건부 처리)
            # 🔥 개선: 스트리밍 모드에서는 표 관련 지시사항 제거
            table_instruction = "" if is_streaming else "- **표의 문서 번호 열에 반드시 `[문서 N]` 형식을 포함하세요** - 빈 셀은 절대 허용되지 않습니다\n"
            
            document_citation_section = f"""
## 문서 인용 규칙 및 중요 사항

### ⚠️ 필수 사항 (반드시 준수해야 합니다)
- **답변에서 문서를 인용할 때는 `[문서 N]` 형식을 사용하세요**
- **최소 {min_citations}개 이상의 서로 다른 문서를 인용**해야 합니다
- **문서를 전혀 인용하지 않으면 답변이 거부됩니다**
- **🔥 CRITICAL: 답변 본문에 반드시 `[문서 N]` 형식으로 문서를 인용하세요** - 답변 본문에 인용이 없으면 답변이 거부됩니다
- **🔥 중요: 각 문서는 답변 본문에서 정확히 한 번만 인용하세요** - 같은 문서를 여러 번 인용하면 신뢰도가 떨어집니다
- **🔥 CRITICAL: 인용은 반드시 문단 끝에 배치하세요** - 문장 시작 부분(`[문서 1]은...`, `[문서 1]에 따르면...`)이나 문장 중간에 배치하지 마세요
{table_instruction}
### 문서 인용 방법 (자연스러운 인용)
- 문서의 핵심 내용을 먼저 설명하고, 해당 내용이 나온 문서를 문단 끝에 자연스럽게 인용하세요
- 여러 문서의 내용을 통합하여 설명한 경우, 관련 문서들을 문단 끝에 나란히 인용할 수 있습니다 (예: [문서 1] [문서 2])
- **인용 없이 문서 내용만 설명하면 안 됩니다** - 반드시 `[문서 N]` 형식으로 인용해야 합니다

**올바른 인용 예시 (가독성 높음 - 문단 끝 배치)**:

**예시 1: 각 문단 끝에 단일 문서 인용**
```
임차인은 선량한 관리자의 주의를 다하여 임대차 목적물을 보존하고, 임대차 종료 시에 임대차 목적물을 원상에 회복하여 반환할 의무를 부담합니다. [문서 1]

임대인은 목적물을 임차인에게 인도하고 임대차계약 존속 중에 그 사용, 수익에 필요한 상태를 유지하게 할 의무를 부담합니다. [문서 2]
```

**예시 1-1: 계약 해지 관련 (실제 사용 예시)**
```
계약 해지 사유에 대해 문의하셨군요. 옵션계약의 효력 정지에 관한 가처분 신청 사건을 다루고 있으며, 계약 해지 사유를 직접적으로 명시하고 있지는 않습니다. 다만, 해당 문서에서 신청인은 계약 체결 이후 환율 상승으로 인해 큰 손실을 입었음을 주장하며, 계약의 효력 정지를 신청했습니다. [문서 1]

일반적으로 계약 해지 사유는 계약서에 명시된 경우, 법률에 규정된 경우, 또는 당사자 간의 합의에 의해 결정될 수 있습니다. [문서 2]
```

**예시 2: 여러 문서를 통합 설명 후 함께 인용**
```
임대차 분쟁 시 내용증명 발송과 관련하여 몇 가지 주의사항이 있습니다. 우선, 임차인은 선량한 관리자의 주의 의무를 다하여 임대차 목적물을 보존해야 하며, 임대인은 목적물을 사용·수익하기에 필요한 상태로 유지할 의무가 있습니다. 또한 임대차 목적물이 화재 등으로 소멸된 경우, 임차인은 그 이행불능이 자신의 책임 없는 사유로 인한 것임을 증명하지 못하면 손해를 배상할 책임을 집니다. [문서 1] [문서 2]
```

**예시 3: 각 문서의 내용을 설명한 후 개별 인용**
```
계약 해지는 다음과 같은 사유로 가능합니다. 부당이득반환청구에서 법률상의 원인 없는 사유를 계약의 불성립, 취소, 무효, 해제 등으로 주장할 수 있습니다. [문서 1]

계약의 해제는 유효하게 성립된 계약의 효력을 당사자 일방의 의사표시에 의해 소급적으로 소멸시키는 것을 의미합니다. [문서 2]

임대차계약 사례에서도 확인할 수 있듯이, 계약 종료는 다양한 방식으로 이루어질 수 있습니다. [문서 3]
```

**잘못된 인용 예시 (가독성 낮음 - 절대 금지)**:
- "계약 해지는 [문서 1]에 따르면 다음과 같은 사유로 가능합니다."  ← 문장 중간에 인용이 끼어있어 가독성 낮음
- "임차인은 선량한 관리자의 주의 의무를 다해야 하며[문서 1], 임대인은 목적물을 사용·수익하기에 필요한 상태로 유지할 의무가 있습니다[문서 2]."  ← 문장 중간에 인용이 끼어있어 가독성 낮음
- "임차인은 [문서 1]에 따르면 선량한 관리자의 주의 의무를 [문서 1]에 따라 다해야 하며..."  ← 같은 문서를 여러 번 인용하고 문장 중간에 끼어있음
- "계약 해지 사유에 대해 문의하셨군요. [문서 1]은 옵션계약의 효력 정지에 관한 가처분 신청 사건을 다루고 있습니다."  ← 문장 중간에 인용이 끼어있음 (절대 금지)
- "계약 해지 사유에 대해 문의하셨군요. [문서 1]에 따르면..."  ← 문장 중간에 인용이 끼어있음 (절대 금지)

**올바른 수정 예시**:
- ❌ "계약 해지 사유에 대해 문의하셨군요. [문서 1]은 옵션계약의 효력 정지에 관한 가처분 신청 사건을 다루고 있습니다."
- ✅ "계약 해지 사유에 대해 문의하셨군요. 옵션계약의 효력 정지에 관한 가처분 신청 사건을 다루고 있으며, 계약 해지 사유를 직접적으로 명시하고 있지는 않습니다. [문서 1]"

**인용 위치 가이드 (반드시 준수)**:
- ✅ **문단의 마지막에 인용 배치** (문단 끝, 문장 끝이 아닌 문단 끝)
- ✅ 각 문서의 내용을 설명한 후, 해당 설명이 끝나는 문단 끝에 인용 배치
- ✅ 여러 문서의 내용을 통합 설명한 경우, 관련 문서들을 문단 끝에 나란히 인용 (예: [문서 1] [문서 2])
- ✅ 각 문서는 한 번만 인용
- ❌ **문장 중간에 인용 끼워 넣기 (절대 금지)** - 예: "...해야 하며[문서 1], ..." 또는 "[문서 1]은 ..." 또는 "[문서 1]에 따르면 ..."
- ❌ **문장 시작 부분에 인용 배치 (절대 금지)** - 예: "[문서 1]은 옵션계약의..." 또는 "[문서 1]에 따르면 계약 해지는..."
- ❌ 문장 끝에 인용 배치 (문단 끝이 아닌)
- ❌ 같은 문서를 여러 번 인용하기

**인용 위치 체크 방법**:
1. 문서 내용을 먼저 설명하세요
2. 설명이 끝나는 문단의 마지막에 `[문서 N]`을 배치하세요
3. 문장 중간이나 문장 시작 부분에 `[문서 N]`을 배치하지 마세요

### 문서 활용 원칙 (🔥 중요)
- **🔥 CRITICAL: 답변 본문에 반드시 `[문서 N]` 형식으로 최소 {min_citations}개 이상의 서로 다른 문서를 인용하세요**
- 검색된 {doc_count}개 문서 중 가능하면 {min(doc_count, 5)}개 이상을 활용하는 것이 좋습니다
- 위 문서의 내용을 바탕으로 답변을 생성하세요
- 문서에서 추론하거나 추측하지 말고, 문서에 명시된 내용만 사용하세요
- 문서에 없는 정보는 포함하지 마세요
- 여러 문서의 내용을 종합하여 일관된 답변을 구성하세요
- **각 문서는 한 번만 인용하여 신뢰도 높은 답변을 작성하세요**
- **문서 내용을 설명할 때는 반드시 해당 문서를 `[문서 N]` 형식으로 인용하세요** - 인용 없이 설명만 하면 안 됩니다

### 법령 및 판례 인용 원칙 (🔥 중요)
- 문서에 법령 조문(예: 민법 제750조)이 포함되어 있으면 **해당 조문을 인용**하세요
- 문서에 판례가 포함되어 있으면 **해당 판례를 인용**하세요
- 법령 조문 인용 시 정확한 형식으로 표기하세요 (예: "민법 제750조", "제750조")
- 판례 인용 시 법원명과 판결일을 함께 언급하세요 (예: "대법원 2020다12345")
- **검색된 법령 조문을 적절히 인용**하세요 (최소 {min_citations}개 이상)
- **각 법령/판례 문서는 한 번만 인용하세요**

"""
            
            # 🔥 스트리밍 모드에 따라 문서 표 작성 지시를 조건부로 처리
            if is_streaming:
                # 스트리밍 모드: 문서 표 생성하지 않도록 지시, [END] 키워드 사용
                table_section = f"""### ⚠️ 중요: 스트리밍 모드에서는 문서 표를 생성하지 마세요
- **🔥 CRITICAL: 스트리밍 모드에서는 문서별 근거 비교 표를 생성하지 마세요**
- **문서 표는 생성하지 말고, 답변 본문에만 `[문서 N]` 형식으로 인용하세요**
- **답변 본문이 끝나면 반드시 `[END]` 키워드를 작성하세요**
- **`[END]` 이후에 `<metadata>` 태그만 작성하세요** (문서 근거 섹션은 작성하지 마세요)
- **문서 정보는 metadata에만 포함하세요** - 답변 본문에는 표를 포함하지 마세요

**답변 구조 (스트리밍 모드)**:
1. 답변 본문 작성 (각 문단 끝에 `[문서 N]` 인용)
2. 답변 본문이 끝나면 `[END]` 키워드 작성
3. `[END]` 이후에 `<metadata>` 태그만 작성 (문서 근거 섹션은 작성하지 마세요)

**최종 체크리스트 (답변 작성 후 반드시 확인)**:
1. ✅ **답변 본문에 `[문서 N]` 형식의 인용이 최소 {min_citations}개 이상 있는가?** - 없으면 답변이 거부됩니다
2. ✅ **답변 본문이 끝나면 `[END]` 키워드를 작성했는가?** - 스트리밍 모드에서는 필수입니다
3. ✅ **`[END]` 이후에 `<metadata>` 태그만 작성했는가?** - 문서 근거 섹션은 작성하지 마세요
4. ✅ **문서 표를 생성하지 않았는가?** - 스트리밍 모드에서는 표를 생성하지 마세요
5. ✅ **각 문서가 답변 본문에서 정확히 한 번만 인용되었는가?** - 중복 인용은 신뢰도를 떨어뜨립니다
6. ✅ **인용이 문단 끝에 배치되었는가?** - 문장 중간에 끼워 넣지 않았는가?
7. ✅ **최소 {min_citations}개 이상의 서로 다른 문서를 인용했는가?**

**⚠️ 경고**: 위 체크리스트를 모두 통과하지 못하면 답변이 거부되고 재생성됩니다.
"""
            else:
                # 일반 모드: 문서 표 작성 지시 포함
                table_section = f"""### 문서별 근거 비교 표 작성 (필수 - 절대 금지 사항)
- **🔥 CRITICAL: 표의 첫 번째 열('문서 번호' 열)에 반드시 [문서 1], [문서 2] 형식으로 번호를 포함하세요**
- **빈 셀 절대 금지** - 문서 번호 열이 비어있으면 답변이 즉시 거부됩니다
- 아래 '검색된 참고 문서' 섹션에 표시된 문서 번호를 그대로 사용하세요
- 표의 각 행은 반드시 [문서 N] 형식으로 시작해야 합니다
- **최소 {min_citations}개 이상의 문서를 표에 포함**해야 합니다
- **표 작성 전 체크리스트**:
  - ✅ 각 행의 첫 번째 열에 [문서 N] 형식이 있는가?
  - ✅ 빈 셀이 없는가?
  - ✅ 최소 {min_citations}개 이상의 문서가 포함되어 있는가?

**올바른 표 작성 예시**:
| 문서 번호 | 출처 | 핵심 근거 |
|-----------|------|----------|
| [문서 1] | 민법 제750조 | 고의 또는 과실로 인한 위법행위로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다. |
| [문서 2] | 민법 제537조 | 쌍무계약의 당사자 일방의 채무가 당사자쌍방의 책임없는 사유로 이행할 수 없게 된 때에는 채무자는 상대방의 이행을 청구하지 못한다. |
| [문서 3] | 민법 제526조 | 계약의 해석은 당사자의 진의를 명확히 하여야 한다. |

**잘못된 표 작성 예시 (절대 금지 - 답변 거부됨)**:
| 문서 번호 | 출처 | 핵심 근거 |
|-----------|------|----------|
|  | 민법 제750조 | ... |  ← 문서 번호가 비어있음 (절대 금지, 답변 거부)
| [문서 2] | 민법 제537조 | ... |  ← 첫 번째 행이 비어있으면 안 됨 (답변 거부)
|  | 서울중앙지방법원 옵션계약효력정지가처분 | ... |  ← 문서 번호가 비어있음 (절대 금지, 답변 거부)

**표 작성 체크리스트 (작성 후 반드시 확인)**:
- ✅ 표의 각 행 첫 번째 열에 `[문서 1]`, `[문서 2]` 형식이 있는가?
- ✅ 빈 셀이 없는가? (빈 셀이 있으면 답변이 즉시 거부됩니다)
- ✅ 최소 {min_citations}개 이상의 문서가 포함되어 있는가?

**최종 체크리스트 (답변 작성 후 반드시 확인)**:
1. ✅ **답변 본문에 `[문서 N]` 형식의 인용이 최소 {min_citations}개 이상 있는가?** - 없으면 답변이 거부됩니다
2. ✅ **표의 각 행 첫 번째 열에 [문서 N] 형식이 있는가?** - 빈 셀이 있으면 답변이 거부됩니다
3. ✅ **각 문서가 답변 본문에서 정확히 한 번만 인용되었는가?** - 중복 인용은 신뢰도를 떨어뜨립니다
4. ✅ **인용이 문단 끝에 배치되었는가?** - 문장 중간에 끼워 넣지 않았는가?
5. ✅ **최소 {min_citations}개 이상의 서로 다른 문서를 인용했는가?**

**⚠️ 경고**: 위 체크리스트를 모두 통과하지 못하면 답변이 거부되고 재생성됩니다.
"""
            
            # 🔥 개선: 스트리밍 모드에 따라 메타데이터 위치 명확화
            if is_streaming:
                metadata_section = """
### 출력 형식 (메타데이터)

**⚠️ 중요: 스트리밍 모드에서는 메타데이터만 작성하세요. 문서 근거 섹션은 작성하지 마세요.**

답변을 작성한 후, 다음 형식으로 메타데이터를 추가하세요:

```
<답변 본문>

[END]

<metadata>
{{
    "document_usage": [
        {{
            "document_number": 1,
            "source": "민법 제750조",
            "source_type": "statute_article",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [150],
            "usage_rationale": "손해배상 책임에 대한 법적 근거로 사용"
        }},
        {{
            "document_number": 2,
            "source": "대법원 2020다12345",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [300],
            "usage_rationale": "계약 해지 관련 판례 인용"
        }}
    ],
    "coverage": {{
        "keyword_coverage": 0.85,
        "keyword_total": 20,
        "keyword_matched": 17,
        "citation_coverage": 0.75,
        "citation_count": 2,
        "citation_expected": 3,
        "document_usage_rate": 0.67,
        "documents_used": 2,
        "documents_total": 3,
        "documents_min_required": 2,
        "overall_coverage": 0.80
    }}
}}
</metadata>
```

**중요 사항 (스트리밍 모드)**:
- 답변 본문은 자연스러운 텍스트로 작성하세요
- 답변 본문이 끝나면 `[END]` 키워드를 작성하세요
- `[END]` 이후에 `<metadata>` 태그만 작성하세요 (문서 근거 섹션은 작성하지 마세요)
- 메타데이터는 반드시 `<metadata>` 태그 안에 JSON 형식으로 작성하세요
- `document_usage`의 각 항목에 `source`와 `usage_rationale`을 포함하세요 (이 정보로 문서 근거 목록이 자동 생성됩니다)
- `document_usage`에는 검색된 모든 문서를 포함하세요 (사용 여부와 관계없이)
- `used_in_answer`는 답변 본문에서 실제로 인용되었는지 여부를 나타냅니다
- `citation_positions`는 답변 본문에서 인용이 나타난 문자 위치(인덱스)입니다
- `usage_rationale`는 해당 문서를 사용한 이유를 간단히 설명하세요 (문서 근거 목록 생성에 사용됩니다)
"""
            else:
                metadata_section = """
### 출력 형식 (메타데이터)

답변을 작성한 후, 다음 형식으로 메타데이터를 추가하세요:

```
<답변 본문>

---

<metadata>
{{
    "document_usage": [
        {{
            "document_number": 1,
            "source": "민법 제750조",
            "source_type": "statute_article",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [150],
            "usage_rationale": "손해배상 책임에 대한 법적 근거로 사용"
        }},
        {{
            "document_number": 2,
            "source": "대법원 2020다12345",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [300],
            "usage_rationale": "계약 해지 관련 판례 인용"
        }}
    ],
    "coverage": {{
        "keyword_coverage": 0.85,
        "keyword_total": 20,
        "keyword_matched": 17,
        "citation_coverage": 0.75,
        "citation_count": 2,
        "citation_expected": 3,
        "document_usage_rate": 0.67,
        "documents_used": 2,
        "documents_total": 3,
        "documents_min_required": 2,
        "overall_coverage": 0.80
    }}
}}
</metadata>
```

**중요 사항**:
- 답변 본문은 자연스러운 텍스트로 작성하세요
- 메타데이터는 반드시 `<metadata>` 태그 안에 JSON 형식으로 작성하세요
- `document_usage`에는 검색된 모든 문서를 포함하세요 (사용 여부와 관계없이)
- `used_in_answer`는 답변 본문에서 실제로 인용되었는지 여부를 나타냅니다
- `citation_positions`는 답변 본문에서 인용이 나타난 문자 위치(인덱스)입니다
- `usage_rationale`는 해당 문서를 사용한 이유를 간단히 설명하세요
"""
            
            document_citation_section = document_citation_section.rstrip() + "\n\n" + table_section + metadata_section
            instruction_section = ""
        else:
            # 🔥 개선: 문서가 부족한 경우에도 메타데이터 출력 형식 포함
            if doc_count > 0 and doc_count < 3:
                min_citations = 1  # 최소 1개 인용
                document_citation_section = f"""
## 문서 인용 규칙 및 중요 사항

### ⚠️ 필수 사항 (반드시 준수해야 합니다)
- **답변에서 문서를 인용할 때는 `[문서 N]` 형식을 사용하세요**
- **최소 {min_citations}개 이상의 문서를 인용**해야 합니다
- **🔥 중요: 각 문서는 답변 본문에서 정확히 한 번만 인용하세요** - 같은 문서를 여러 번 인용하면 신뢰도가 떨어집니다
- **🔥 CRITICAL: 인용은 반드시 문단 끝에 배치하세요** - 문장 시작 부분이나 문장 중간에 배치하지 마세요

### 문서 인용 방법 (자연스러운 인용)
- **각 문서는 답변 본문에서 한 번만 인용하세요** - 같은 문서 번호를 반복 사용하지 마세요
- **인용은 반드시 문단 끝에 배치하세요** - 문장 중간이나 문장 끝에 끼워 넣지 마세요
- 문서의 핵심 내용을 먼저 설명하고, 해당 내용이 나온 문서를 문단 끝에 자연스럽게 인용하세요

**올바른 인용 예시**:
```
계약 해지와 관련하여 법원에서는 다양한 사안을 다루고 있습니다. 예를 들어, 통화옵션계약의 경우, 약관규제법 위반을 이유로 계약의 효력 정지를 신청하는 사례가 있었습니다. [문서 1]
```

**잘못된 인용 예시 (절대 금지)**:
- "계약 해지와 관련하여... [문서 1] 이 사건에서는... [문서 1]"  ← 같은 문서를 여러 번 인용 (절대 금지)
- "계약 해지는 [문서 1]에 따르면 가능합니다."  ← 문장 중간에 인용 (절대 금지)

### 문서 활용 원칙
- 위 문서의 내용을 바탕으로 답변을 생성하세요
- 문서에서 추론하거나 추측하지 말고, 문서에 명시된 내용만 사용하세요
- 문서에 없는 정보는 포함하지 마세요
- 여러 문서의 내용을 종합하여 일관된 답변을 구성하세요
- **각 문서는 한 번만 인용하여 신뢰도 높은 답변을 작성하세요**

⚠️ **중요**: 문서가 {doc_count}개로 부족하므로 **문서별 근거 비교 표를 생성하지 마세요**.

### 출력 형식 (메타데이터)

답변을 작성한 후, 다음 형식으로 메타데이터를 추가하세요:

```
<답변 본문>

---

<metadata>
{{
    "document_usage": [
        {{
            "document_number": 1,
            "source": "서울중앙지방법원 옵션계약효력정지가처분 (2009카합393)",
            "source_type": "precedent_content",
            "used_in_answer": true,
            "citation_count": 1,
            "citation_positions": [150],
            "usage_rationale": "계약 해지 관련 판례 인용"
        }}
    ],
    "coverage": {{
        "keyword_coverage": 0.85,
        "keyword_total": 20,
        "keyword_matched": 17,
        "citation_coverage": 0.75,
        "citation_count": 1,
        "citation_expected": 1,
        "document_usage_rate": 1.0,
        "documents_used": 1,
        "documents_total": {doc_count},
        "documents_min_required": 1,
        "overall_coverage": 0.80
    }}
}}
</metadata>
```

**중요 사항**:
- 답변 본문은 자연스러운 텍스트로 작성하세요
- 메타데이터는 반드시 `<metadata>` 태그 안에 JSON 형식으로 작성하세요
- `document_usage`에는 검색된 모든 문서를 포함하세요 (사용 여부와 관계없이)
- `used_in_answer`는 답변 본문에서 실제로 인용되었는지 여부를 나타냅니다
- `citation_positions`는 답변 본문에서 인용이 나타난 문자 위치(인덱스)입니다
- `usage_rationale`는 해당 문서를 사용한 이유를 간단히 설명하세요
"""
            else:
                document_citation_section = """
## 문서 인용 규칙 및 중요 사항

### 문서 활용 원칙
- 일반 법적 원칙 기반 답변, 실무적 조언 중심

⚠️ **중요**: 문서가 없으므로 **문서별 근거 비교 표를 생성하지 마세요**.
"""
            instruction_section = ""

        # 🔥 개선: 질문-문서 불일치 경고 추가
        mismatch_warning = ""
        if normalized_docs and len(normalized_docs) > 0:
            try:
                mismatch_warning = self._check_query_document_mismatch(query, normalized_docs)
            except Exception as e:
                logger.warning(f"⚠️ [MISMATCH CHECK] Error checking query-document mismatch: {e}")
                mismatch_warning = ""
        
        # 🔥 개선: documents_section 포함 확인 로깅
        documents_section_length = len(documents_section) if documents_section else 0
        logger.info(
            f"📋 [FINAL PROMPT BUILD] documents_section length: {documents_section_length}, "
            f"document_citation_section length: {len(document_citation_section) if document_citation_section else 0}, "
            f"mismatch_warning length: {len(mismatch_warning) if mismatch_warning else 0}, "
            f"simplified_role length: {len(simplified_role) if simplified_role else 0}"
        )
        
        # 🔥 개선: 불일치 경고를 핵심 지시사항에 통합
        if mismatch_warning:
            core_instructions = mismatch_warning + "\n" + core_instructions
        
        # 최종 프롬프트 구성: 핵심 지시 → Few-shot → 역할 정의 → 문서 인용 규칙 → 문서 → 질문
        # 🔥 중요: document_citation_section을 생성한 후에 최종 프롬프트 구성
        final_prompt = (
            core_instructions +
            few_shot_example +
            simplified_role +
            (document_citation_section if document_citation_section else "") +
            (documents_section if documents_section else "") +
            f"\n\n## 사용자 질문\n{query}\n\n답변을 시작하세요:\n"
        )
        
        # 최종 토큰 수 검증
        final_tokens = self._estimate_tokens(final_prompt)
        if final_tokens > max_safe_tokens:
            logger.warning(
                f"⚠️ [TOKEN LIMIT] Final prompt exceeds safe limit: "
                f"{final_tokens:,} tokens (max: {max_safe_tokens:,}). "
                f"Applying emergency truncation..."
            )
            # 긴급 축약: 문서 섹션만 축약
            final_prompt = self._emergency_truncate_prompt(final_prompt, max_safe_tokens, simplified_role, query)
            final_tokens = self._estimate_tokens(final_prompt)
        
        logger.info(
            f"✅ [TOKEN COUNT] Final prompt: {final_tokens:,} tokens "
            f"({final_tokens/self.MAX_INPUT_TOKENS*100:.1f}% of max, "
            f"base: {base_tokens:,}, query: {query_tokens:,}, "
            f"docs: {current_doc_tokens if 'current_doc_tokens' in locals() else 0:,})"
        )
        
        # 🔥 개선: final_prompt가 비어있으면 기본 프롬프트 반환
        if not final_prompt or len(final_prompt.strip()) == 0:
            logger.warning("⚠️ [PROMPT BUILD] final_prompt is empty, using base prompt with query")
            final_prompt = f"{base_prompt}\n\n## 사용자 질문\n{query}\n\n답변을 시작하세요:\n"
        
        return final_prompt

    def _validate_prompt_contains_documents(self, final_prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """프롬프트에 실제 문서 내용이 포함되었는지 검증 (강화된 버전)"""
        validation_result = {
            "has_document_content": False,
            "context_text_length": 0,
            "prompt_contains_context": False,
            "document_count_in_context": 0,
            "document_count_in_prompt": 0,
            "validation_details": []
        }

        if not context:
            return validation_result

        # structured_documents 확인 (우선순위 1)
        structured_docs = context.get("structured_documents", {})
        if isinstance(structured_docs, dict):
            documents = structured_docs.get("documents", [])
            validation_result["document_count_in_context"] = len(documents) if documents else 0

            if documents:
                doc_found_count = 0
                for idx, doc in enumerate(documents[:5], 1):  # 상위 5개 확인
                    if isinstance(doc, dict):
                        # 정규화된 문서 사용
                        normalized = self._normalize_document_fields(doc)
                        doc_content = normalized.get("content", "")
                        
                        # 🔥 개선: 원본 문서의 content도 확인 (정규화 전)
                        original_content = doc.get("content", "") or doc.get("text", "")
                        
                        # 🔥 개선: 법률 정보가 있으면 law_name + article_no로도 확인
                        law_name = normalized.get("law_name") or normalized.get("statute_name") or ""
                        article_no = normalized.get("article_no") or normalized.get("article_number") or ""
                        has_law_info = bool(law_name and article_no)

                        found_in_prompt = False
                        
                        # 1. 정규화된 content 확인
                        if doc_content and len(doc_content) > 50:
                            doc_preview = doc_content[:150].strip()
                            doc_mid = doc_content[len(doc_content)//2:len(doc_content)//2+100].strip() if len(doc_content) > 200 else ""

                            if doc_preview and doc_preview in final_prompt:
                                found_in_prompt = True
                                doc_found_count += 1
                            elif doc_mid and doc_mid in final_prompt:
                                found_in_prompt = True
                                doc_found_count += 1
                        
                        # 2. 원본 content 확인 (정규화 전)
                        if not found_in_prompt and original_content and len(original_content) > 50:
                            original_preview = original_content[:150].strip()
                            if original_preview and original_preview in final_prompt:
                                found_in_prompt = True
                                doc_found_count += 1
                        
                        # 3. 법률 정보로 확인 (law_name + article_no가 프롬프트에 있는지)
                        if not found_in_prompt and has_law_info:
                            # 프롬프트에 "[문서 N]" 형식이 있고, 해당 문서의 법률 정보가 포함되어 있는지 확인
                            law_article_pattern = f"{law_name} 제{article_no}"
                            # article_no 정규화 (앞의 0 제거)
                            article_no_normalized = article_no.lstrip('0') if article_no else ""
                            if article_no_normalized:
                                law_article_pattern_normalized = f"{law_name} 제{article_no_normalized}"
                                # 프롬프트에 법률 정보가 포함되어 있고, "[문서" 패턴도 있으면 문서가 포함된 것으로 간주
                                if (law_article_pattern in final_prompt or law_article_pattern_normalized in final_prompt) and "[문서" in final_prompt:
                                    found_in_prompt = True
                                    doc_found_count += 1

                        validation_result["validation_details"].append({
                            "doc_index": idx,
                            "source": normalized.get("source", "Unknown"),
                            "content_length": len(doc_content),
                            "original_content_length": len(original_content),
                            "has_law_info": has_law_info,
                            "found_in_prompt": found_in_prompt
                        })

                validation_result["document_count_in_prompt"] = doc_found_count

                if doc_found_count > 0:
                    validation_result["has_document_content"] = True
                    logger.info(
                        f"✅ [PROMPT VALIDATION] Found {doc_found_count}/{len(documents)} documents in prompt"
                    )
                else:
                    logger.warning(
                        f"⚠️ [PROMPT VALIDATION] No document content found in prompt "
                        f"despite having {len(documents)} documents in context"
                    )

        # context 텍스트 확인 (우선순위 2)
        context_text = context.get("context", "")
        if context_text and len(context_text.strip()) > 100:  # 최소 100자 이상
            validation_result["context_text_length"] = len(context_text)

            # 프롬프트에 context_text가 포함되어 있는지 확인
            # (처음 200자와 중간 200자 확인)
            context_preview = context_text[:200].strip()
            context_mid = context_text[len(context_text)//2:len(context_text)//2+200].strip() if len(context_text) > 400 else ""

            if context_preview and context_preview in final_prompt:
                validation_result["prompt_contains_context"] = True
                validation_result["has_document_content"] = True
            elif context_mid and context_mid in final_prompt:
                validation_result["prompt_contains_context"] = True
                validation_result["has_document_content"] = True

        # prompt_optimized_text 확인 (우선순위 3)
        prompt_optimized_text = context.get("prompt_optimized_text", "")
        if prompt_optimized_text and len(prompt_optimized_text.strip()) > 100:
            preview = prompt_optimized_text[:200].strip()
            if preview and preview in final_prompt:
                validation_result["has_document_content"] = True

        return validation_result

    def _batch_fetch_law_names(self, docs: List[Dict[str, Any]]) -> Dict[int, str]:
        """statute_id로 법령명 배치 조회 (성능 최적화)"""
        statute_ids = set()
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            # law_name이나 statute_name이 없고 statute_id가 있는 경우만 조회
            if doc.get("law_name") or doc.get("statute_name"):
                continue
            statute_id = doc.get("statute_id") or (doc.get("metadata", {}).get("statute_id") if isinstance(doc.get("metadata"), dict) else None)
            if statute_id:
                try:
                    statute_ids.add(int(statute_id))
                except (ValueError, TypeError):
                    continue
        
        if not statute_ids:
            return {}
        
        law_name_map = {}
        try:
            from lawfirm_langgraph.core.data.db_adapter import DatabaseAdapter
            from lawfirm_langgraph.core.utils.config import get_config
            
            config = get_config()
            db_adapter = DatabaseAdapter(config.database_url)
            
            with db_adapter.get_connection_context() as conn:
                cursor = conn.cursor()
                # 배치 조회
                placeholders = ",".join(["%s"] * len(statute_ids))
                cursor.execute(f"""
                    SELECT id, law_name_kr, law_abbrv 
                    FROM statutes 
                    WHERE id IN ({placeholders})
                """, list(statute_ids))
                
                for row in cursor.fetchall():
                    if isinstance(row, dict):
                        statute_id = row.get('id')
                        law_name_kr = row.get('law_name_kr') or row.get('law_abbrv') or ""
                    else:
                        statute_id = row[0]
                        law_name_kr = row[1] if len(row) > 1 else (row[2] if len(row) > 2 else "")
                    
                    if law_name_kr and statute_id:
                        law_name_map[int(statute_id)] = law_name_kr
            
            if law_name_map:
                logger.debug(f"✅ [BATCH FETCH] 법령명 {len(law_name_map)}개 조회 완료: {list(law_name_map.keys())}")
        except Exception as e:
            logger.debug(f"⚠️ [BATCH FETCH] 법령명 배치 조회 실패: {e}")
        
        return law_name_map

    def _normalize_document_fields(self, doc: Dict[str, Any], law_name_cache: Optional[Dict[int, str]] = None) -> Optional[Dict[str, Any]]:
        """문서 필드명 정규화 - 법률명, 조문 번호 등 명시적 추출 (개선: 문서 제목 생성 및 관련성 점수 포함)"""
        if not isinstance(doc, dict):
            return None

        # content 필드: 여러 가능한 필드명에서 추출 (우선순위: content > text > chunk_text)
        # 개선: content 필드를 우선 확인 (workflow_document_processor에서 content로 저장)
        raw_content = (
            doc.get("content", "") or
            doc.get("text", "") or
            doc.get("chunk_text", "") or
            doc.get("content_text", "") or
            doc.get("document_text", "") or
            doc.get("full_text", "") or
            doc.get("body", "") or
            str(doc.get("metadata", {}).get("content", "") if isinstance(doc.get("metadata"), dict) else "") or
            str(doc.get("metadata", {}).get("text", "") if isinstance(doc.get("metadata"), dict) else "") or
            ""
        )
        
        # 🔥 개선: JSON 문자열로 저장된 content 파싱
        if isinstance(raw_content, str) and raw_content.strip().startswith("{") and raw_content.strip().endswith("}"):
            try:
                import ast
                parsed = ast.literal_eval(raw_content)
                if isinstance(parsed, dict):
                    # 파싱된 딕셔너리에서 실제 content 추출
                    raw_content = parsed.get("content") or parsed.get("text") or raw_content
            except (ValueError, SyntaxError):
                # 파싱 실패 시 원본 사용
                pass
        
        # 불필요한 메타데이터 제거
        content = self._clean_content(raw_content).strip()
        
        # 🔥 레거시 호환 필드 정리: normalize_document_type으로 type 보장
        try:
            from lawfirm_langgraph.core.utils.document_type_normalizer import normalize_document_type
            doc = normalize_document_type(doc)
        except ImportError:
            try:
                from core.utils.document_type_normalizer import normalize_document_type
                doc = normalize_document_type(doc)
            except ImportError:
                pass
        
        # 법률 정보가 있으면 content가 짧아도 포함 (개선: 10자 → 3자로 완화)
        # TASK 3 개선: statute_name, type 필드도 확인
        # 단일 소스 원칙: doc.type만 사용
        doc_type = doc.get("type", "")
        # 🔥 개선: type 필드가 statute_article이면 무조건 has_law_info=True
        is_statute_type = doc_type in ["statute_article", "statute"]
        has_law_info = bool(
            doc.get("law_name") or 
            doc.get("statute_name") or  # TASK 3: statute_name 추가
            doc.get("article_no") or 
            doc.get("article_number") or  # TASK 3: article_number 추가
            doc.get("case_name") or 
            doc.get("case_number") or
            is_statute_type or  # 🔥 개선: type 필드 확인 (statute_article이면 무조건 True)
            (isinstance(doc.get("metadata"), dict) and (
                doc.get("metadata", {}).get("law_name") or
                doc.get("metadata", {}).get("statute_name") or  # TASK 3: metadata의 statute_name 추가
                doc.get("metadata", {}).get("article_no") or
                doc.get("metadata", {}).get("article_number") or  # TASK 3: metadata의 article_number 추가
                doc.get("metadata", {}).get("case_name") or
                doc.get("metadata", {}).get("type") in ["statute_article", "statute"]  # TASK 3: metadata의 type 확인
            ))
        )
        
        min_content_length = self.MIN_CONTENT_LENGTH_WITH_LAW_INFO if has_law_info else self.MIN_CONTENT_LENGTH
        
        # 멀티 질의 메타데이터 확인 (디버깅)
        has_multi_query_meta = False
        if isinstance(doc.get("metadata"), dict):
            has_multi_query_meta = bool(doc.get("metadata", {}).get("sub_query"))
        elif doc.get("sub_query"):
            has_multi_query_meta = True
        
        logger.debug(f"🔍 [DOC NORMALIZE] content length={len(content) if content else 0}, "
                    f"min_length={min_content_length}, has_law_info={has_law_info}, "
                    f"has_multi_query_meta={has_multi_query_meta}, doc_type={doc_type}, "
                    f"is_statute_type={is_statute_type}, doc_keys={list(doc.keys())[:10]}")
        
        # 멀티 질의 결과는 content가 짧아도 포함 (최소 길이 완화)
        if has_multi_query_meta and content and len(content) < min_content_length:
            logger.debug(f"✅ [DOC NORMALIZE] Multi-query result with short content ({len(content)} chars), "
                        f"will create minimal doc")
            # 최소 content 생성
            if not content or len(content.strip()) < 3:
                content = doc.get("source", "") or "법률 문서"
        
        if not content or len(content) < min_content_length:
            if has_law_info:
                # 🔥 개선: 법률 정보가 있으면 원본 content가 3자 이상이면 길이 제한 없이 포함
                if content and len(content.strip()) >= 3:
                    # 원본 content가 있으면 그대로 사용 (길이 제한 없음)
                    logger.debug(f"✅ [DOC NORMALIZE] Content length={len(content)} chars, has_law_info=True, keeping original content")
                else:
                    # content가 없거나 3자 미만일 때만 최소 content 생성
                    logger.debug(f"⚠️ [DOC NORMALIZE] Content too short ({len(content) if content else 0} chars) but has law info, creating minimal doc")
                # 법률 정보로 최소 content 생성
                law_name = doc.get("law_name") or doc.get("statute_name") or ""
                article_no = doc.get("article_no") or doc.get("article_number") or ""
                if law_name and article_no:
                    content = f"{law_name} 제{article_no}조"
                elif law_name:
                    content = law_name
                elif not content or len(content.strip()) < 3:
                    content = doc.get("source", "") or "법률 문서"
            elif has_multi_query_meta:
                # 멀티 질의 결과는 최소 content 생성
                logger.debug("✅ [DOC NORMALIZE] Multi-query result, creating minimal doc even with short content")
                content = doc.get("source", "") or "법률 문서"
            else:
                logger.debug(f"⚠️ [DOC NORMALIZE] Content too short ({len(content)} chars) and no law info, returning None")
                return None

        # source 필드: 여러 가능한 필드명에서 추출
        source = (
            doc.get("source", "") or
            doc.get("title", "") or
            doc.get("document_id", "") or
            doc.get("name", "") or
            doc.get("law_name", "") or
            doc.get("case_name", "") or
            ""
        )

        # 메타데이터에서 법률 정보 추출 (원본 메타데이터 보존)
        original_metadata = doc.get("metadata", {})
        if not isinstance(original_metadata, dict):
            original_metadata = {}
        
        # 원본 메타데이터를 복사하여 보존 (멀티 질의 메타데이터 포함)
        metadata = original_metadata.copy() if isinstance(original_metadata, dict) else {}

        # 필드 추출 (헬퍼 메서드 사용)
        law_name = self._extract_field(doc, metadata, ["law_name", "statute_name", "name"])
        article_no = self._extract_field(doc, metadata, ["article_no", "article_number"])
        clause_no = self._extract_field(doc, metadata, ["clause_no", "clause_number"])
        item_no = self._extract_field(doc, metadata, ["item_no", "item_number"])
        heading = self._extract_field(doc, metadata, ["heading", "article_title"])
        
        # 🔥 개선: law_name이 없고 statute_id가 있으면 캐시에서 조회
        if not law_name:
            statute_id = doc.get("statute_id") or (metadata.get("statute_id") if isinstance(metadata, dict) else None)
            if statute_id and law_name_cache:
                try:
                    statute_id_int = int(statute_id)
                    if statute_id_int in law_name_cache:
                        law_name = law_name_cache[statute_id_int]
                        # metadata에도 저장
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata["law_name"] = law_name
                        metadata["statute_name"] = law_name
                        logger.debug(f"✅ [DOC NORMALIZE] 법령명 캐시에서 조회: statute_id={statute_id}, law_name={law_name}")
                except (ValueError, TypeError):
                    pass
        
        # 판례 정보 추출
        court = self._extract_field(doc, metadata, ["court"])
        case_number = self._extract_field(doc, metadata, ["case_number", "doc_id", "case_id"])
        case_name = self._extract_field(doc, metadata, ["case_name", "casenames"])
        announce_date = self._extract_field(doc, metadata, ["announce_date", "decision_date"])
        case_type = self._extract_field(doc, metadata, ["case_type"])
        
        # 판례 본문 정보 추출
        case_summary = self._extract_field(doc, metadata, ["summary", "case_summary"])
        case_holding = self._extract_field(doc, metadata, ["holding", "case_holding", "판시사항"])
        case_reasoning = self._extract_field(doc, metadata, ["reasoning", "case_reasoning", "판결요지"])
        
        # 문서 타입 판단 (DocumentType Enum 사용)
        # 🔥 개선: 원본 문서의 type 필드를 먼저 확인하고 복원
        doc_type_str = self._extract_field(doc, metadata, ["type"])
        
        # 원본 문서의 type 필드가 있으면 우선 사용 (statute_article, precedent_content 등)
        original_type = doc.get("type")
        if original_type and original_type not in ["", "unknown", "other"]:
            doc_type_str = str(original_type).strip()
            logger.debug(f"✅ [DOC NORMALIZE] 원본 type 필드 사용: {doc_type_str}")
        elif isinstance(metadata, dict) and metadata.get("type") and metadata.get("type") not in ["", "unknown", "other"]:
            doc_type_str = str(metadata.get("type")).strip()
            logger.debug(f"✅ [DOC NORMALIZE] metadata에서 type 복원: {doc_type_str}")
        
        # DocumentType Enum을 사용하여 타입 추론 (메타데이터 필드 기준)
        try:
            from lawfirm_langgraph.core.workflow.constants.document_types import DocumentType
            
            # 최상위 필드의 정보를 metadata에 복사 (DocumentType 추론을 위해)
            if doc.get("statute_name") or doc.get("law_name") or doc.get("article_no"):
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["statute_name"] = doc.get("statute_name") or doc.get("law_name")
                metadata["law_name"] = doc.get("law_name") or doc.get("statute_name")
                metadata["article_no"] = doc.get("article_no") or doc.get("article_number")
            
            if doc.get("case_id") or doc.get("court") or doc.get("doc_id") or doc.get("casenames"):
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["case_id"] = doc.get("case_id")
                metadata["court"] = doc.get("court") or doc.get("ccourt")
                metadata["doc_id"] = doc.get("doc_id")
                metadata["casenames"] = doc.get("casenames")
                metadata["precedent_id"] = doc.get("precedent_id")
            
            # type이 없거나 unknown/other인 경우에만 DocumentType Enum으로 추론
            if not doc_type_str or doc_type_str in ["unknown", "other"]:
                # DocumentType Enum을 사용하여 타입 추출
                doc_type = DocumentType.from_metadata(doc)
                inferred_type = doc_type.value
                
                # 추론된 타입이 unknown이 아니면 사용
                if inferred_type and inferred_type != "unknown":
                    doc_type_str = inferred_type
                    logger.debug(f"✅ [DOC NORMALIZE] DocumentType 추론: {doc_type_str}")
            
            # metadata에도 타입 정보 저장
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["type"] = doc_type_str
        except Exception as e:
            logger.debug(f"⚠️ [DOC NORMALIZE] DocumentType 추론 실패: {e}, type={doc_type_str}")

        # 🔥 개선: 관련성 점수 추출 (메타데이터에서도 추출)
        relevance_score = float(
            doc.get("relevance_score", 0.0) or 
            doc.get("final_weighted_score", 0.0) or
            doc.get("score", 0.0) or 
            doc.get("similarity_score", 0.0) or
            doc.get("similarity", 0.0) or
            (isinstance(doc.get("metadata"), dict) and (
                doc.get("metadata", {}).get("relevance_score", 0.0) or
                doc.get("metadata", {}).get("score", 0.0) or
                doc.get("metadata", {}).get("final_weighted_score", 0.0) or
                0.0
            )) or
            0.0
        )
        
        # 문서 제목 생성 (개선: 일관된 형식)
        doc_title = ""
        if law_name and article_no:
            doc_title = f"{law_name} 제{article_no}조"
            if clause_no:
                doc_title += f" 제{clause_no}항"
            if item_no:
                doc_title += f" 제{item_no}호"
        elif case_name:
            doc_title = case_name
            if court:
                doc_title = f"{court} {doc_title}"
        else:
            doc_title = source or doc.get("type", "법률 문서")
        
        # 🔥 개선: has_law_info=True인 경우 길이 제한 없이 원본 content 유지
        if not content or len(content) < min_content_length:
            if has_law_info:
                # 원본 content가 3자 이상이면 길이 제한 없이 포함
                if content and len(content.strip()) >= 3:
                    # 원본 content 유지 (길이 제한 없음)
                    logger.debug(f"✅ [DOC NORMALIZE] Content length={len(content)} chars, has_law_info=True, keeping original content (no length restriction)")
                else:
                    # content가 없거나 3자 미만일 때만 최소 content 생성
                    logger.debug(f"⚠️ [DOC NORMALIZE] Content too short ({len(content) if content else 0} chars) but has law info, creating minimal doc")
                # 법률 정보만으로 최소 content 생성
                if law_name and article_no:
                    content = f"{law_name} 제{article_no}조"
                    if clause_no:
                        content += f" 제{clause_no}항"
                    if heading:
                        content += f" ({heading})"
                elif case_name:
                    content = case_name
                    if court:
                        content = f"{court} {case_name}"
                else:
                    content = doc_title or "법률 문서"
                logger.debug(f"✅ [DOC NORMALIZE] Created minimal content from law info: {content[:50]}")
        
        # 정규화된 문서 생성 (불필요한 메타데이터 제외)
        # EXCLUDED_METADATA_FIELDS에 있는 필드는 제외
        normalized = {
            "content": str(content).strip(),
            "title": doc_title,
            "source": str(source).strip() or "Unknown",
            "relevance_score": relevance_score,
            # 법률 정보
            "law_name": str(law_name).strip(),
            "article_no": str(article_no).strip(),
            "clause_no": str(clause_no).strip(),
            "item_no": str(item_no).strip(),
            "heading": str(heading).strip(),
            # 판례 정보
            "court": str(court).strip(),
            "case_number": str(case_number).strip(),
            "case_name": str(case_name).strip(),
            "casenames": str(case_name).strip(),  # 호환성
            "doc_id": str(case_number).strip() or str(metadata.get("doc_id", "")).strip(),  # 판례 ID 추가
            "announce_date": str(announce_date).strip(),
            "case_type": str(case_type).strip(),
            # 판례 본문 정보
            "case_summary": str(case_summary).strip(),
            "case_holding": str(case_holding).strip(),
            "case_reasoning": str(case_reasoning).strip(),
            "type": str(doc_type_str).strip() if doc_type_str else "unknown",  # type 필드 (DocumentType Enum 호환)
            # 원본 메타데이터 보존 (멀티 질의 메타데이터 포함)
            "metadata": metadata
        }
        
        # None 값 및 빈 문자열 제거 (metadata는 제외)
        normalized = {k: v for k, v in normalized.items() if k == "metadata" or (v and v != "")}
        
        # EXCLUDED_METADATA_FIELDS에 있는 필드 제거 (추가 필터링, metadata는 제외)
        normalized = {k: v for k, v in normalized.items() if k == "metadata" or k not in self.EXCLUDED_METADATA_FIELDS}

        return normalized

    def _format_document_for_prompt(self, doc: Dict[str, Any], idx: int, is_high_priority: bool = False) -> str:
        """문서를 프롬프트 형식으로 포맷팅 (개선: 간소화, 불필요한 섹션 제거)"""
        content = doc.get("content", "")
        score = float(doc.get("relevance_score", 0.0) or doc.get("score", 0.0) or 0.0)

        # 문서 제목 생성 (간소화)
        doc_title, _ = self._get_document_title_and_max_length(doc, idx)

        # 간결한 형식으로 변경 (불필요한 섹션 제거)
        formatted = f"**[문서 {idx}]** {doc_title} (관련도: {score:.2f})\n\n"

        # 내용 길이 최적화
        if is_high_priority:
            content_preview = content[:800] if len(content) > 800 else content
        else:
            content_preview = content[:500] if len(content) > 500 else content

        formatted += f"{content_preview}{'...' if len(content) > len(content_preview) else ''}\n\n"

        return formatted

    def _build_documents_section(self, documents: List[Dict[str, Any]], query: str) -> str:
        """문서 섹션 생성 - 정규화된 문서들을 프롬프트 형식으로 변환"""
        if not documents or len(documents) == 0:
            return ""

        # 문서 수에 따라 최소 인용 수 동적 설정
        doc_count = len(documents)
        if doc_count >= 5:
            min_citations = 3  # 5개 이상이면 최소 3개 인용
            max_docs_to_show = min(10, doc_count)  # 최대 10개까지 표시
        elif doc_count >= 3:
            min_citations = 2  # 3-4개면 최소 2개 인용
            max_docs_to_show = doc_count
        else:
            min_citations = 1  # 1-2개면 최소 1개 인용
            max_docs_to_show = doc_count

        documents_section = "\n\n## 🔍 검색된 참고 문서 (반드시 활용하세요)\n\n"
        documents_section += f"**⚠️ 중요**: 아래 {doc_count}개 문서를 반드시 활용하여 답변을 작성하세요. 각 문서는 `[문서 N]` 형식으로 인용해야 합니다.\n\n"

        # 최대 10개까지 문서 표시 (토큰 제한 내에서)
        for idx, doc in enumerate(documents[:max_docs_to_show], 1):
            # TASK 3: None 체크 추가
            if not doc or not isinstance(doc, dict):
                continue
            
            doc_title, _ = self._get_document_title_and_max_length(doc, idx)
            content = (
                doc.get("content") or
                doc.get("text") or
                doc.get("content_text") or
                ""
            )
            
            # 내용 길이 최적화
            is_high_priority = (idx <= 3)
            if is_high_priority:
                content_preview = content[:800] if len(content) > 800 else content
            else:
                content_preview = content[:500] if len(content) > 500 else content
            
            score = doc.get("relevance_score", doc.get("final_weighted_score", 0.0))
            
            formatted_doc = f"""### [문서 {idx}] {doc_title}
{content_preview}{'...' if len(content) > len(content_preview) else ''}

**인용 방법**: 이 문서를 인용할 때는 반드시 `[문서 {idx}]` 형식을 사용하세요.
**관련도**: {score:.2f}

---
"""
            documents_section += formatted_doc

        documents_section += f"\n**⚠️ 최종 경고**: 위 문서 중 최소 {min_citations}개 이상을 반드시 `[문서 N]` 형식으로 인용하세요. 인용 없이는 답변이 거부됩니다.\n\n"

        return documents_section
    
    def _build_fallback_documents_section(self, documents: List[Dict[str, Any]]) -> str:
        """폴백 문서 섹션 생성 - 문서 섹션이 생성되지 않았을 때 사용"""
        if not documents or len(documents) == 0:
            return ""

        # 문서 수에 따라 최소 인용 수 동적 설정
        doc_count = len(documents)
        if doc_count >= 5:
            min_citations = 3
            max_docs_to_show = min(10, doc_count)
        elif doc_count >= 3:
            min_citations = 2
            max_docs_to_show = doc_count
        else:
            min_citations = 1
            max_docs_to_show = doc_count

        documents_section = "\n\n## 검색된 참고 문서\n\n"
        documents_section += f"다음 {doc_count}개의 문서를 반드시 참고하여 답변하세요. **문서를 인용할 때는 반드시 `[문서 N]` 형식을 사용하세요.**\n\n"
        documents_section += f"**⚠️ 중요**: **최소 {min_citations}개 이상의 문서를 반드시 인용**해야 합니다. 가능하면 {min(doc_count, 5)}개 이상의 문서를 활용하는 것이 좋습니다.\n\n"

        import re

        for idx, doc in enumerate(documents[:max_docs_to_show], 1):  # 최대 10개
            # TASK 3: None 체크 추가
            if not doc or not isinstance(doc, dict):
                continue
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            score = doc.get("relevance_score", 0.0)

            if not content or len(content.strip()) < 10:
                continue

            # 내용 길이 조정
            content_preview = content[:2000] if len(content) > 2000 else content
            core_content = content[:300] if len(content) > 300 else content

            # 조문/판례 번호 추출
            article_match = re.search(r'제\s*\d+\s*조', content[:200])
            case_match = re.search(r'\d{4}[다나]\d+', content[:200])

            # 🔴 개선: 관련도 표기 통일 (.3f → .2f)
            documents_section += f"### 문서 {idx}: {source} (관련도: {score:.2f})\n\n"

            if article_match:
                documents_section += f"**핵심 조문**: {article_match.group()}\n\n"
            if case_match:
                documents_section += f"**판례 번호**: {case_match.group()}\n\n"

            documents_section += f"**핵심 내용**: {core_content}{'...' if len(content) > 300 else ''}\n\n"
            documents_section += f"**전체 내용**:\n{content_preview}\n\n"
            documents_section += "---\n\n"

        return documents_section

    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트 포맷팅"""
        formatted_parts = []
        for key, value in context.items():
            if isinstance(value, list):
                formatted_parts.append(f"**{key}**:\n" + "\n".join([f"- {item}" for item in value]))
            elif isinstance(value, dict):
                formatted_parts.append(f"**{key}**:\n" + "\n".join([f"- {k}: {v}" for k, v in value.items()]))
            else:
                formatted_parts.append(f"**{key}**: {value}")
        return "\n\n".join(formatted_parts)

    def _structure_context_by_question_type(
        self,
        context: Dict[str, Any],
        question_type: QuestionType
    ) -> str:
        """질문 유형별 컨텍스트 구조화 - 문서 및 데이터베이스 결과 강제 포함"""
        context_text = context.get("context", "")
        legal_references = context.get("legal_references", [])
        insights = context.get("insights", [])
        citations = context.get("citations", [])

        # structured_documents에서 문서 내용 추출
        structured_docs = context.get("structured_documents", {})
        document_contents = []

        # 방법 1: structured_documents에서 직접 추출
        if isinstance(structured_docs, dict):
            documents = structured_docs.get("documents", [])
            for doc in documents[:10]:  # 상위 10개 문서
                if isinstance(doc, dict):
                    # content 필드 우선 확인, 없으면 text 필드 사용
                    doc_content = doc.get("content", "") or doc.get("text", "")
                    doc_source = doc.get("source", "Unknown")
                    # relevance_score 우선 확인, 없으면 score 사용
                    doc_score = doc.get("relevance_score", 0.0) or doc.get("score", 0.0)
                    if doc_content and len(doc_content.strip()) > 10:
                        # 법률 정보와 판례 정보도 함께 추출
                        doc_dict = {
                            "source": doc_source,
                            "content": doc_content,
                            "score": doc_score,
                            # 법률 정보
                            "law_name": doc.get("law_name", ""),
                            "article_no": doc.get("article_no", ""),
                            "clause_no": doc.get("clause_no", ""),
                            "item_no": doc.get("item_no", ""),
                            "heading": doc.get("heading", ""),
                            # 판례 정보
                            "court": doc.get("court", ""),
                            "case_number": doc.get("case_number", ""),
                            "case_name": doc.get("case_name", ""),
                            "announce_date": doc.get("announce_date", ""),
                            "case_type": doc.get("case_type", ""),
                            # 판례 본문 정보
                            "case_summary": doc.get("case_summary", ""),
                            "case_holding": doc.get("case_holding", ""),
                            "case_reasoning": doc.get("case_reasoning", ""),
                            "type": doc.get("type", "")
                        }
                        document_contents.append(doc_dict)

        # 방법 2: structured_documents가 없으면 context 텍스트에서 문서 패턴 추출
        if not document_contents:
            context_text = context.get("context", "")
            if context_text and len(context_text) > 100:
                # context 텍스트에서 문서 패턴 추출 ([문서 출처: ...] 패턴)
                import re
                doc_pattern = r'\[문서 출처:\s*([^\]]+)\]\s*\[관련도:\s*([\d.]+)\]\s*\n(.*?)(?=\n\[문서 출처:|$)'
                matches = re.findall(doc_pattern, context_text, re.DOTALL)
                for match in matches[:10]:
                    if len(match) == 3:
                        doc_source = match[0].strip()
                        doc_score = float(match[1]) if match[1] else 0.0
                        doc_content = match[2].strip()
                        if doc_content and len(doc_content) > 10:
                            document_contents.append({
                                "source": doc_source,
                                "content": doc_content,
                                "score": doc_score
                            })

                if document_contents:
                    logger.info(f"✅ [DOCUMENT EXTRACTION] Extracted {len(document_contents)} documents from context text")

        structured_parts = []

        # 🔥 개선: document_contents가 있더라도 문서 섹션은 생성하지 않음
        # 문서 섹션은 _build_documents_section()에서만 생성
        try:
            # 🔥 개선: document_contents가 있더라도 문서 섹션은 생성하지 않음
            # 문서 섹션은 _build_documents_section()에서만 생성
            if document_contents:
                # 문서 섹션 생성 제거
                # 대신 간단한 참고 메시지만 추가
                structured_parts.append(
                    f"\n⚠️ 참고: {len(document_contents)}개의 관련 문서가 검색되었습니다. "
                    f"문서 내용은 아래 '검색된 참고 문서' 섹션을 참고하세요.\n"
                )
            
            # 질문 유형별 구조화
            if question_type == QuestionType.PRECEDENT_SEARCH:
                # 판례 정보 우선 배치
                # 🔥 개선: 문서 섹션 생성 제거 (위에서 참고 메시지 추가됨)

                if citations:
                    precedent_citations = [cit for cit in citations if isinstance(cit, dict) and cit.get("type") == "precedent"]
                    if precedent_citations:
                        structured_parts.append("## 관련 판례\n")
                        for cit in precedent_citations[:5]:
                            structured_parts.append(f"- {cit.get('text', '')}")
                        structured_parts.append("")

                if context_text:
                    structured_parts.append("## 판례 관련 정보\n")
                    structured_parts.append(context_text)
                    structured_parts.append("")

                if legal_references:
                    structured_parts.append("## 관련 법령\n")
                    for ref in legal_references[:5]:
                        if isinstance(ref, str):
                            structured_parts.append(f"- {ref}")
                        else:
                            structured_parts.append(f"- {ref.get('text', '') if isinstance(ref, dict) else str(ref)}")

            elif question_type == QuestionType.LAW_INQUIRY:
                # 법률 조문 중심 구조
                if legal_references:
                    structured_parts.append("## 관련 법률 조문\n")
                    structured_parts.extend(self._format_legal_references(legal_references[:5]))
                    structured_parts.append("")

                if citations:
                    law_citations = [cit for cit in citations if isinstance(cit, dict) and cit.get("type") == "law_article"]
                    if law_citations:
                        structured_parts.append("## 법률 조항 상세\n")
                        for cit in law_citations[:5]:
                            structured_parts.append(f"- {cit.get('text', '')}")
                        structured_parts.append("")

                # 🔥 개선: 문서 섹션 생성 제거 (위에서 참고 메시지 추가됨)

                if context_text:
                    structured_parts.append("## 조문 해설\n")
                    structured_parts.append(context_text)

            elif question_type == QuestionType.LEGAL_ADVICE:
                # 법령 + 판례 + 실무 조언 균형 배치
                if legal_references:
                    structured_parts.append("## 관련 법령\n")
                    structured_parts.extend(self._format_legal_references(legal_references[:3]))
                    structured_parts.append("")

                if citations:
                    precedent_cits = [cit for cit in citations if isinstance(cit, dict) and cit.get("type") == "precedent"]
                    if precedent_cits:
                        structured_parts.append("## 관련 판례\n")
                        for cit in precedent_cits[:3]:
                            structured_parts.append(f"- {cit.get('text', '')}")
                        structured_parts.append("")

                # 🔥 개선: 문서 섹션 생성 제거 (위에서 참고 메시지 추가됨)

                if context_text:
                    structured_parts.append("## 법률 분석 및 실무 조언\n")
                    structured_parts.append(context_text)

                if insights:
                    structured_parts.append("\n## 핵심 요약\n")
                    for insight in insights[:3]:
                        structured_parts.append(f"- {insight}")
            else:
                # 🔥 개선: 문서 섹션 생성 제거 (위에서 참고 메시지 추가됨)

                if legal_references:
                    structured_parts.append("## 관련 법령\n")
                    structured_parts.extend(self._format_legal_references(legal_references[:5]))
                    structured_parts.append("")

                # 🔴 개선: "추가 관련 정보" 섹션 제거 (중복 방지)
                # if context_text:
                #     structured_parts.append("## 추가 관련 정보\n")
                #     structured_parts.append(context_text)
                #     structured_parts.append("")
                # context_text는 이미 다른 곳에서 사용되므로 여기서 중복 제거

            # 최종 검증: structured_parts에 실제 문서 내용이 포함되었는지 확인
            result_text = "\n".join(structured_parts)
            
            # 개선: 문서 내용 포함 검증 강화
            # 1. document_contents가 있는데 result_text에 문서 내용이 거의 없는 경우
            # 2. result_text 길이가 너무 짧은 경우 (500자 미만)
            # 3. result_text에 "문서", "document", "content" 등의 키워드가 거의 없는 경우
            has_doc_content = document_contents and len(document_contents) > 0
            has_doc_keywords = any(keyword in result_text.lower() for keyword in ["문서", "document", "content", "법률", "판례"])
            text_too_short = len(result_text) < 500
            
            # 🔥 개선: 문서 섹션 강제 추가 로직 제거
            # 문서 섹션은 _build_documents_section()에서만 생성하므로 여기서는 제거
            if has_doc_content and (text_too_short or not has_doc_keywords):
                logger.info(
                    f"ℹ️ [CONTEXT STRUCTURE] Document contents will be included in documents_section. "
                    f"(text_len={len(result_text)}, has_keywords={has_doc_keywords})"
                )

            return result_text

        except Exception as e:
            logger.warning(f"Context structuring failed: {e}, using default format")
            # 폴백: 기본 포맷
            return self._format_context(context)

    def _get_relevant_examples(
        self,
        question_type: QuestionType,
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Few-shot 학습을 위한 질문 유형별 답변 예시 (검색 결과 활용 강조)"""
        examples = []

        # 🔴 개선: 실제 검색 결과 기반 예시 생성
        if documents and len(documents) > 0:
            # 실제 문서를 예시로 사용
            sorted_docs = sorted(
                documents,
                key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )
            example_docs = sorted_docs[:2]  # 상위 2개만 예시로 사용
        else:
            example_docs = None

        try:
            if question_type == QuestionType.PRECEDENT_SEARCH:
                # 실제 문서가 있으면 사용, 없으면 일반 예시 사용
                if example_docs and any(doc.get("court") or doc.get("case_number") for doc in example_docs if isinstance(doc, dict)):
                    # 실제 판례 문서로 예시 생성
                    example_text = "## 📚 답변 형식 가이드\n\n"
                    example_text += "**답변 작성 시 다음과 같은 형식으로 검색 결과를 활용하세요**:\n\n"
                    for idx, doc in enumerate(example_docs[:2], 1):
                        if isinstance(doc, dict):
                            case_number = doc.get("case_number", "")
                            court = doc.get("court", "")
                            case_name = doc.get("case_name", "")
                            score = doc.get("relevance_score", 0.0)

                            ref = case_number or (f"{court} {case_name}" if court and case_name else case_name or "판례")
                            example_text += f"**문서 {idx}**: {ref} (관련도: {score:.2f}) - 실제 검색 결과 참고\n"
                    example_text += "\n**좋은 답변 형식**:\n"
                    example_text += "- 검색된 판례를 인용하여 답변: \"[판례: 판례번호]에 따르면...\"\n"
                    example_text += "- 판례의 핵심 내용을 요약하여 설명\n"
                    example_text += "- 실무적 의미를 함께 설명\n\n"
                    examples.append(example_text)
                else:
                    # 일반 예시 (특정 판례 번호 없이)
                    examples.append("""
## 📚 답변 형식 가이드

**좋은 답변 형식**:
- 검색된 판례를 명확히 인용하여 답변
- 판례의 핵심 내용(판시사항, 판결요지) 요약
- 실무적 의미 설명

**나쁜 답변**: 출처 없이 일반 지식만 사용
""")
            elif question_type == QuestionType.LAW_INQUIRY:
                # 실제 법령 문서가 있으면 사용
                if example_docs and any(doc.get("law_name") and doc.get("article_no") for doc in example_docs if isinstance(doc, dict)):
                    example_text = "## 📚 답변 형식 가이드\n\n"
                    example_text += "**답변 작성 시 다음과 같은 형식으로 검색 결과를 활용하세요**:\n\n"
                    for idx, doc in enumerate(example_docs[:2], 1):
                        if isinstance(doc, dict):
                            law_name = doc.get("law_name", "")
                            article_no = doc.get("article_no", "")
                            score = doc.get("relevance_score", 0.0)
                            if law_name and article_no:
                                example_text += f"**문서 {idx}**: {law_name} 제{article_no}조 (관련도: {score:.2f}) - 실제 검색 결과 참고\n"
                    example_text += "\n**좋은 답변 형식**:\n"
                    example_text += "- 조문 내용을 인용하여 설명: \"[법령: 법령명 제XX조]에 따르면...\"\n"
                    example_text += "- 조문의 의미를 이해하기 쉽게 설명\n"
                    example_text += "- 필요시 실무적 조언 포함\n\n"
                    examples.append(example_text)
                else:
                    # 일반 예시
                    examples.append("""
## 📚 답변 형식 가이드

**좋은 답변 형식**:
- 검색된 조문을 명확히 인용: "[법령: 법령명 제XX조]"
- 조문의 의미를 이해하기 쉽게 설명
- 필요시 실무적 의미 포함

**나쁜 답변**: 출처 없이 일반 지식만 사용
""")
            elif question_type == QuestionType.LEGAL_ADVICE:
                # 실제 문서가 있으면 사용
                if example_docs:
                    example_text = "## 📚 답변 형식 가이드\n\n"
                    example_text += "**답변 작성 시 다음과 같은 형식으로 검색 결과를 활용하세요**:\n\n"
                    example_text += "**좋은 답변 구성**:\n"
                    example_text += "1. 상황 파악 및 법적 근거 제시\n"
                    example_text += "2. 검색된 법령/판례 인용하여 설명\n"
                    example_text += "3. 실행 가능한 구체적 조언\n"
                    example_text += "4. 주의사항 (필요시)\n\n"
                    example_text += "**인용 형식**:\n"
                    for idx, doc in enumerate(example_docs[:2], 1):
                        if isinstance(doc, dict):
                            law_name = doc.get("law_name", "")
                            article_no = doc.get("article_no", "")
                            case_number = doc.get("case_number", "")
                            if law_name and article_no:
                                example_text += f"- 법령: \"[법령: {law_name} 제{article_no}조]에 따르면...\"\n"
                            elif case_number:
                                example_text += f"- 판례: \"[판례: {case_number}]에 따르면...\"\n"
                    example_text += "\n**나쁜 답변**: 검색 결과 없이 추측으로 답변\n"
                    examples.append(example_text)
                else:
                    # 일반 예시
                    examples.append("""
## 📚 답변 형식 가이드

**좋은 답변 구성**:
1. 상황 파악 및 법적 근거 제시
2. 검색된 법령/판례를 인용하여 설명
3. 실행 가능한 구체적 조언
4. 주의사항 (필요시)

**나쁜 답변**: 검색 결과 없이 추측으로 답변
""")
            else:
                # 일반 질문용 예시
                if example_docs:
                    example_text = "## 📚 답변 형식 가이드\n\n"
                    example_text += "**답변 작성 시 검색 결과를 활용하여 답변하세요**:\n"
                    example_text += "- 검색된 법령/판례를 명확히 인용\n"
                    example_text += "- 조문/판례의 의미를 이해하기 쉽게 설명\n"
                    example_text += "- 필요시 실무적 조언 포함\n\n"
                    examples.append(example_text)
                else:
                    examples.append("""
## 📚 답변 형식 가이드

**좋은 답변**: 검색 결과를 활용하여 법적 근거를 제시하고 명확히 설명
**나쁜 답변**: 출처 없이 일반 지식만 사용
""")

            return "\n".join(examples) if examples else ""

        except Exception as e:
            logger.warning(f"Example generation failed: {e}")
            return ""

    def _get_fallback_prompt(self, query: str) -> str:
        """폴백 프롬프트"""
        return f"""당신은 법률 전문가입니다. 다음 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {query}

답변 시 다음 사항을 고려해주세요:
1. 관련 법령과 판례를 정확히 인용
2. 실무적 관점에서 실행 가능한 조언 제공
3. 불확실한 부분은 명확히 표시
4. 전문가 상담 권유

답변을 한국어로 작성해주세요."""

    def _normalize_legal_domain(self, domain) -> Optional[LegalDomain]:
        """LegalDomain 정규화 - 문자열을 enum으로 변환"""
        if domain is None:
            return None

        # 이미 LegalDomain enum인 경우
        if isinstance(domain, LegalDomain):
            return domain

        # 문자열인 경우 enum으로 변환
        if isinstance(domain, str):
            # 문자열 값으로 매핑
            domain_mapping = {
                "민사법": LegalDomain.CIVIL_LAW,
                "형사법": LegalDomain.CRIMINAL_LAW,
                "가족법": LegalDomain.FAMILY_LAW,
                "상사법": LegalDomain.COMMERCIAL_LAW,
                "행정법": LegalDomain.ADMINISTRATIVE_LAW,
                "노동법": LegalDomain.LABOR_LAW,
                "부동산법": LegalDomain.PROPERTY_LAW,
                "지적재산권법": LegalDomain.INTELLECTUAL_PROPERTY,
                "세법": LegalDomain.TAX_LAW,
                "민사소송법": LegalDomain.CIVIL_PROCEDURE,
                "형사소송법": LegalDomain.CRIMINAL_PROCEDURE,
                "기타/일반": LegalDomain.GENERAL,
                "일반": LegalDomain.GENERAL,
                "기타": LegalDomain.GENERAL,
            }

            # 직접 매핑 시도
            if domain in domain_mapping:
                return domain_mapping[domain]

            # LegalDomain enum의 value로 찾기
            for legal_domain in LegalDomain:
                if legal_domain.value == domain:
                    return legal_domain

            # LegalDomain enum의 name으로 찾기
            try:
                return LegalDomain[domain.upper()]
            except (KeyError, AttributeError):
                pass

            logger.warning(f"⚠️ [DOMAIN NORMALIZATION] Unknown domain string: '{domain}', defaulting to GENERAL")
            return LegalDomain.GENERAL

        logger.warning(f"⚠️ [DOMAIN NORMALIZATION] Unknown domain type: {type(domain)}, defaulting to None")
        return None

    def _select_base_prompt_type(self, question_type: QuestionType, domain: Optional[LegalDomain] = None) -> str:
        """질문 유형과 도메인에 따라 적절한 base_prompt_type 선택"""
        # 질문 유형별 기본 선택
        if question_type == QuestionType.LEGAL_ADVICE:
            # 법률 자문: 전문가 자문 스타일
            return "professional_advisor"
        elif question_type == QuestionType.GENERAL_QUESTION:
            # 일반 질문: 자연스러운 상담사 스타일
            return "natural_consultant"
        elif question_type in [QuestionType.PRECEDENT_SEARCH, QuestionType.LAW_INQUIRY]:
            # 판례/법령 검색: 전문가 스타일
            return "korean_legal_expert"
        else:
            # 기본: 한국 법률 전문가
            return "korean_legal_expert"

    # 기본 프롬프트 템플릿들
    def _get_korean_legal_expert_prompt(self) -> str:
        """한국 법률 전문가 기본 프롬프트"""
        return """# Role: 대한민국 법률 전문가 AI 어시스턴트

당신은 대한민국 법률 전문 상담 AI입니다. 법학 석사 이상의 전문 지식을 보유하고 있으며, 다양한 법률 분야에 대한 실무 경험을 갖춘 것처럼 행동합니다.

## 한국 법률 특성
- 성문법 중심, 대법원 판례 중시, 실무적 관점

## 핵심 역할

1. **법률 정보 제공**: 민법, 형법, 상법, 행정법 등 각종 법률에 대한 정확한 정보 제공
2. **법적 조언**: 사용자의 상황을 분석하고 적절한 법적 대응 방향 제시
3. **판례 및 법령 해설**: 관련 판례와 법령을 쉽게 설명
4. **절차 안내**: 소송, 고소, 계약 등 법적 절차에 대한 안내

## 답변 원칙

### 1. 정확성과 신중성
- 확실한 법률 정보만 제공하며, 불확실한 경우 명확히 표시
- 법률은 해석의 여지가 있음을 인지하고 단정적 표현 자제
- 최신 법령 개정 사항에 대해서는 확인이 필요함을 안내

### 2. 출처 표기 형식 통일
- 출처는 답변 내용에 자연스럽게 통합하되, 반드시 `[문서 N]` 형식을 사용하세요
- 법조문 인용: "민법 제543조에 따르면..." 형식 사용
- 판례 인용: `[문서 N]` 형식과 함께 법원명과 판결일을 함께 언급하세요
  (예: "[문서 1] 서울고등법원 2018. 5. 15. 선고 2017나2046429 판결에 따르면...")
- 자연스러운 문단 흐름 유지, 과도한 구조화 지양

### 3. 분쟁/소송 관련 질문
```
말씀하신 상황을 정리하면... 이 사안의 핵심 쟁점은... 관련 법리와 판례에 따르면...

[권리 구제 방법을 자연스럽게 설명]
협상/조정, 민사소송, 형사고소 등의 방법이 있으며, 각각의 장단점은...

[증거 자료 및 절차를 자연스럽게 안내]

[전문가 상담 권고를 자연스럽게 포함]
```

**중요**: 명시적 섹션 제목 대신 자연스러운 문단 흐름으로 내용을 연결하세요. 번호 매기기는 3개 이상의 연속 항목일 때만 사용하세요.

## 특별 지침

### 긴급 상황 대응
- 긴급한 법적 위험이 있는 경우 즉시 전문가 상담 권고
- 형사 사건의 경우 변호인 조력권 고지
- 시효 임박 사항은 명확히 경고

### 정보 부족 시
"정확한 답변을 위해 다음 정보가 추가로 필요합니다: [구체적 항목]"

### 관할 및 전문 분야 외
"이 질문은 [특정 분야] 전문 변호사의 자문이 필요한 사안입니다."

## 금지 사항

❌ 개별 사건에 대한 확정적 결론 제시
❌ 승소/패소 가능성에 대한 구체적 확률 제시
❌ 변호사 수임 또는 소송 제기 강요
❌ 불법 행위 조력
❌ 의뢰인-변호사 관계 형성
❌ 개인정보 수집 또는 요구

## 출력 스타일

- **자연스러운 대화형 어조**: 존댓말을 사용하되 딱딱하지 않게, 친근하면서도 전문적으로
- **문단 구분 명확히**: 자연스러운 문단 전환으로 내용을 연결
- **중요 내용은 강조**: 핵심 법적 정보는 강조 표시
- **출처 표기**: 법원명과 판결일을 직접 언급하여 자연스럽게 표기 (예: "[문서 1] 서울고등법원 2018. 5. 15. 선고 2017나2046429 판결에 따르면...")
- **번호 매기기 제한**: 3개 이상의 연속 항목일 때만 사용, 그 외에는 자연스러운 문장으로 연결
- **표 사용 최소화**: 과도한 표는 지양하고, 필요시 답변 말미에만 간략히 포함
- **띄어쓰기 필수**: 모든 문장에 자연스러운 띄어쓰기를 반드시 적용하세요"""

    def _get_natural_consultant_prompt(self) -> str:
        """자연스러운 상담사 프롬프트"""
        return """당신은 친근하고 전문적인 법률 상담사입니다. 사용자의 질문에 대해 다음과 같은 스타일로 답변해주세요:

### 답변 스타일 가이드

1. **친근한 인사**: "안녕하세요! 말씀하신 내용에 대해 도움을 드리겠습니다."

2. **질문 이해 확인**: "말씀하신 [구체적 질문 내용]에 대해 궁금하시군요."

3. **핵심 답변**:
   - 법률 조항을 먼저 제시
   - 쉬운 말로 해석 설명
   - 실제 적용 사례나 예시 포함

4. **실무적 조언**:
   - 구체적인 행동 방안 제시
   - 주의사항을 친절하게 안내
   - 추가 고려사항 언급

5. **마무리**:
   - 요약 정리
   - 추가 질문 유도
   - 전문가 상담 권유

### 언어 스타일
- 존댓말 사용하되 딱딱하지 않게
- 법률 용어는 쉬운 말로 풀어서 설명
- 사용자의 입장에서 이해하기 쉽게
- 감정적 공감과 전문성을 균형있게

### 답변 구조 예시
```
안녕하세요! [질문 내용]에 대해 도움을 드리겠습니다.

📋 관련 법률 조항
[법률명] 제X조에 따르면...

💡 쉽게 설명하면
이 조항은 [쉬운 설명]을 의미합니다.

🔍 실제 적용 예시
예를 들어, [구체적 사례]의 경우...

⚠️ 주의사항
이런 경우에는 [주의사항]을 고려하셔야 합니다.

📞 추가 도움
더 궁금한 점이 있으시면 언제든 말씀해 주세요!
```

### 특별 지침
- 항상 사용자의 상황을 이해하려고 노력하세요
- 복잡한 법률 개념은 일상적인 예시로 설명하세요
- 불확실한 부분은 솔직하게 말하고 전문가 상담을 권하세요
- 사용자가 걱정하고 있다면 공감하고 안심시켜 주세요

### 면책 문구
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""

    def _get_professional_advisor_prompt(self) -> str:
        """전문가 자문 프롬프트"""
        return """당신은 법률 전문가로서 다음과 같은 전문적 자문을 제공합니다:

### 전문가 자문 스타일

1. **전문성 강조**: 법학적 근거와 실무 경험을 바탕으로 한 분석
2. **체계적 접근**: 논리적이고 체계적인 법적 분석
3. **실무 중심**: 실제 법원, 검찰, 법무부 실무 기준 반영
4. **리스크 관리**: 법적 리스크를 최소화하는 방안 제시

### 답변 구조
```
## 법적 분석
[법률적 근거와 분석]

## 실무적 관점
[실무에서의 적용 방법]

## 리스크 평가
[잠재적 법적 리스크]

## 권장사항
[구체적 실행 방안]

## 추가 고려사항
[주의사항 및 참고사항]
```

### 전문성 표시
- 관련 법령의 정확한 조문 인용
- 최신 대법원 판례 참조
- 실무 경험 기반 조언
- 법적 불확실성 명시

### 면책 문구
본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."""

    # 도메인별 템플릿들
    def _get_civil_law_template(self) -> str:
        """민사법 템플릿"""
        return """
## 민사법 특화 지침
- **핵심 분야**: 계약, 불법행위, 소유권, 상속
- **주요 법령**: 민법, 민사소송법, 부동산등기법
- **최신 개정**: 2024년 민법 개정사항 반영

### 답변 구조
1. **법률관계 분석**: 당사자 간 법률관계 명확화
2. **권리와 의무**: 각 당사자의 권리와 의무 분석
3. **구제 방법**: 권리 구제를 위한 구체적 방법
4. **법적 근거**: 관련 민법 조항과 판례 인용
5. **실무 주의사항**: 실제 적용 시 고려사항

### 특별 고려사항
- 시효 제도 (민법 제162조 이하)
- 불법행위의 성립요건 (민법 제750조)
- 계약의 해제와 해지 (민법 제543조 이하)
- 상속의 개시와 상속분 (민법 제997조 이하)
"""

    def _get_criminal_law_template(self) -> str:
        """형사법 템플릿"""
        return """
## 형사법 특화 지침
- **핵심 분야**: 범죄 구성요건, 형량, 절차
- **주요 법령**: 형법, 형사소송법, 특별법
- **최신 개정**: 디지털 성범죄 처벌법 등 신설법

### 답변 구조
1. **구성요건 분석**: 범죄의 성립요건 분석
2. **법정형**: 해당 범죄의 형량 정보
3. **관련 판례**: 대법원 및 하급심 판례
4. **실무 고려사항**: 수사 및 재판 과정에서의 고려사항

### 특별 고려사항
- 구성요건의 해석 (형법 제1조)
- 정당방위와 긴급피난 (형법 제21조, 제22조)
- 미수범과 기수범 (형법 제25조 이하)
- 공범의 성립 (형법 제30조 이하)
"""

    def _get_family_law_template(self) -> str:
        """가족법 템플릿"""
        return """
## 가족법 특화 지침
- **핵심 분야**: 혼인, 이혼, 친자관계, 상속
- **주요 법령**: 민법 가족편, 가족관계의 등록 등에 관한 법률
- **최신 개정**: 2024년 가족법 개정사항

### 답변 구조
1. **절차 개요**: 해당 절차의 전체적인 흐름
2. **단계별 절차**: 구체적인 단계별 절차
3. **필요 서류**: 절차 진행에 필요한 서류
4. **법적 근거**: 관련 법조문과 판례

### 특별 고려사항
- 혼인의 성립과 무효 (민법 제815조 이하)
- 이혼의 사유와 절차 (민법 제840조 이하)
- 친자관계의 인정 (민법 제844조 이하)
- 상속의 순위와 상속분 (민법 제997조 이하)
"""

    def _get_commercial_law_template(self) -> str:
        """상사법 템플릿"""
        return """
## 상사법 특화 지침
- **핵심 분야**: 회사법, 상행위, 어음수표
- **주요 법령**: 상법, 주식회사법, 어음법
- **최신 개정**: 2024년 상법 개정사항

### 답변 구조
1. **회사 설립**: 회사 설립 절차와 요건
2. **주주권과 이사**: 주주의 권리와 이사의 의무
3. **상행위**: 상행위의 성립과 효과
4. **어음수표**: 어음수표의 발행과 양도

### 특별 고려사항
- 주식회사의 설립 (상법 제289조 이하)
- 주주의 권리와 의무 (상법 제335조 이하)
- 이사의 책임 (상법 제399조 이하)
- 상행위의 특칙 (상법 제47조 이하)
"""

    def _get_administrative_law_template(self) -> str:
        """행정법 템플릿"""
        return """
## 행정법 특화 지침
- **핵심 분야**: 행정행위, 행정절차, 행정소송
- **주요 법령**: 행정절차법, 행정소송법, 행정법
- **최신 개정**: 2024년 행정법 개정사항

### 답변 구조
1. **행정행위 분석**: 행정행위의 성립과 효력
2. **절차 요건**: 행정절차의 준수사항
3. **구제 방법**: 행정소송과 행정심판
4. **법적 근거**: 관련 행정법 조항

### 특별 고려사항
- 행정행위의 성립요건 (행정절차법 제1조)
- 행정소송의 제기 (행정소송법 제6조 이하)
- 행정심판의 절차 (행정심판법 제1조 이하)
- 행정지도의 한계 (행정절차법 제4조)
"""

    def _get_labor_law_template(self) -> str:
        """노동법 템플릿"""
        return """
## 노동법 특화 지침
- **핵심 분야**: 근로계약, 임금, 근로시간, 휴가
- **주요 법령**: 근로기준법, 노동조합법, 고용보험법
- **최신 개정**: 2024년 노동법 개정사항

### 답변 구조
1. **법적 근거**: 근로기준법 등 관련 조항
2. **절차 및 방법**: 권리 구제 절차
3. **구제 기관**: 노동위원회, 법원의 역할
4. **실무 권장사항**: 실제 적용 시 고려사항

### 특별 고려사항
- 근로계약의 성립 (근로기준법 제15조)
- 임금의 지급 (근로기준법 제43조 이하)
- 근로시간의 제한 (근로기준법 제50조 이하)
- 해고의 제한 (근로기준법 제23조)
"""

    def _get_property_law_template(self) -> str:
        """부동산법 템플릿"""
        return """
## 부동산법 특화 지침
- **핵심 분야**: 부동산 계약, 등기, 권리보호
- **주요 법령**: 부동산등기법, 부동산 실권리자명의 등기에 관한 법률
- **최신 개정**: 2024년 부동산법 개정사항

### 답변 구조
1. **계약 요건**: 부동산 계약의 성립요건
2. **등기 절차**: 소유권 이전 등기 절차
3. **권리 보호**: 소유권과 담보권의 보호
4. **실무 주의사항**: 실제 거래 시 고려사항

### 특별 고려사항
- 부동산 매매계약의 성립 (민법 제565조)
- 소유권 이전등기 (부동산등기법 제98조)
- 담보권의 설정 (민법 제357조 이하)
- 실권리자명의 등기 (부동산 실권리자명의 등기에 관한 법률)
"""

    def _get_intellectual_property_template(self) -> str:
        """지적재산권법 템플릿"""
        return """
## 지적재산권법 특화 지침
- **핵심 분야**: 특허, 상표, 저작권, 디자인
- **주요 법령**: 특허법, 상표법, 저작권법, 디자인보호법
- **최신 개정**: 2024년 지적재산권법 개정사항

### 답변 구조
1. **권리 내용**: 지적재산권의 내용과 범위
2. **침해 구제**: 권리 침해 시 구제 방법
3. **등록 절차**: 권리 등록 절차
4. **실무 고려사항**: 실제 분쟁·등록 실무에서의 주의사항

### 특별 고려사항
- 특허권의 효력 범위 (특허법)
- 상표의 식별력 및 혼동 가능성 (상표법)
- 저작물의 보호 범위와 공정 이용 (저작권법)
- 디자인의 독창성 판단 (디자인보호법)
"""

    def _get_tax_law_template(self) -> str:
        """세법 템플릿"""
        return """
## 세법 특화 지침
- **핵심 분야**: 소득세, 법인세, 부가가치세, 상속·증여세
- **주요 법령**: 소득세법, 법인세법, 부가가치세법, 상속세및증여세법

### 답변 구조
1. **과세 대상/시기**
2. **세율/공제**
3. **신고/납부 절차**
4. **판례·예규 참고**
"""

    def _get_civil_procedure_template(self) -> str:
        """민사소송법 템플릿"""
        return """
## 민사소송 절차 지침
- 소 제기 → 변론 → 증거조사 → 판결 → 집행 순서
- 관할, 소가산정, 송달, 상소기간 등 실무 체크
"""

    def _get_criminal_procedure_template(self) -> str:
        """형사소송법 템플릿"""
        return """
## 형사절차 지침
- 수사 → 기소 → 공판 → 판결 → 상소
- 구속, 영장, 증거능력, 자백보강법칙 유의
"""

    def _get_precedent_search_template(self) -> str:
        """판례 검색 템플릿"""
        return """
## 판례 검색 지침
- 사건번호/선고일/법원/사건명 등 메타데이터를 명시하세요.
- 요지는 간결하게, 인용은 [판례: 사건번호]로 표기하세요.
"""

    def _get_law_inquiry_template(self) -> str:
        """법령 질의 템플릿 (개선: 상세화)"""
        return """
## 법령 질의 지침
- 해당 법률명과 조문을 정확히 표기: [법령: 법률명 제OO조].
- 조문 요지와 적용 범위를 함께 설명하세요.
- 관련 판례가 있으면 함께 인용하세요.
- 실무 적용 시 주의사항을 명시하세요.
"""

    def _get_legal_advice_template(self) -> str:
        """법률 자문 템플릿"""
        return """
## 법률 자문 지침
- 사실관계 파악 → 법적 평가 → 실무 조언 순서로 작성.
- 불확실 시 필요한 추가 정보와 리스크를 명시하세요.
"""

    def _get_procedure_guide_template(self) -> str:
        """절차 안내 템플릿"""
        return """
## 절차 안내 지침
- 단계별 절차, 필요서류, 기한, 담당기관을 표 형식 또는 목록으로.
"""

    def _get_term_explanation_template(self) -> str:
        """용어 설명 템플릿"""
        return """
## 법률 용어 설명 지침
- 정의 → 법적 근거 → 예시 → 유사개념 비교 순으로 간결하게.
"""

    def _get_general_question_template(self) -> str:
        """일반 질문 템플릿"""
        return """
## 일반 질문 지침
- 핵심 답만 간결히 제시하고, 필요한 경우 관련 법령/판례를 링크 형태로 제시.
"""
    
    def _estimate_tokens(self, text: str) -> int:
        """
        한국어 기준 토큰 수 추정
        
        Args:
            text: 추정할 텍스트
            
        Returns:
            int: 추정된 토큰 수
        """
        if not text:
            return 0
        
        # 한국어 기준: 1토큰 = 2.5자 (보수적 추정)
        # 영어/숫자/특수문자는 더 적은 토큰 사용
        # 한국어 비율 계산
        korean_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        other_chars = len(text) - korean_chars
        
        # 한국어: 1토큰 = 2.5자, 기타: 1토큰 = 4자
        estimated_tokens = (korean_chars / 2.5) + (other_chars / 4.0)
        
        return int(estimated_tokens)
    
    def _clean_content(self, content: str) -> str:
        """content에서 불필요한 메타데이터 제거 (강화)"""
        import re
        
        if not content:
            return ""
        
        # 중첩된 딕셔너리 패턴 제거 (강화)
        # {'key': 'value'} 또는 {"key": "value"} 형태 제거
        content = re.sub(r"\{'[^']+':\s*[^}]+\}", "", content)
        content = re.sub(r'\{"[^"]+":\s*[^}]+\}', "", content)
        # 중첩된 딕셔너리 (여러 레벨)
        while re.search(r"\{'[^']+':\s*[^}]+\}", content) or re.search(r'\{"[^"]+":\s*[^}]+\}', content):
            content = re.sub(r"\{'[^']+':\s*[^}]+\}", "", content)
            content = re.sub(r'\{"[^"]+":\s*[^}]+\}', "", content)
        
        # 검색 쿼리 정보 제거 (강화)
        # 'query': '...' 또는 "query": "..." 형태
        content = re.sub(r"'query':\s*'[^']*'", "", content)
        content = re.sub(r'"query":\s*"[^"]*"', "", content)
        # query=... 형태
        content = re.sub(r"query\s*=\s*'[^']*'", "", content)
        content = re.sub(r'query\s*=\s*"[^"]*"', "", content)
        # , 'query': ... 형태
        content = re.sub(r",\s*'query':\s*'[^']*'", "", content)
        content = re.sub(r',\s*"query":\s*"[^"]*"', "", content)
        
        # 점수 정보 제거 (강화)
        score_patterns = [
            r"'(?:cross_encoder_score|original_score|keyword_bonus|keyword_match_score|combined_relevance_score|source_type_weight|final_weighted_score|similarity_score|similarity)':\s*[\d.]+",
            r'"(?:cross_encoder_score|original_score|keyword_bonus|keyword_match_score|combined_relevance_score|source_type_weight|final_weighted_score|similarity_score|similarity)":\s*[\d.]+',
            r"(?:cross_encoder_score|original_score|keyword_bonus|keyword_match_score|combined_relevance_score|source_type_weight|final_weighted_score|similarity_score|similarity)\s*[:=]\s*[\d.]+",
        ]
        for pattern in score_patterns:
            content = re.sub(pattern, "", content)
        
        # 메타데이터 키 제거 (강화)
        metadata_keys = [
            "'strategy'", "'id'", "'doc_id'", "'announce_date'", "'response_date'",
            "'cross_encoder'", "'embedding'", "'vector'", "'metadata_keys'",
            "'keyword_match_score'", "'source_type'", "'source_type_weight'",
            "'text'", "'content'", "'metadata'", "'score'", "'relevance_score'"
        ]
        for key in metadata_keys:
            # 다양한 패턴 처리
            content = re.sub(rf"{key}:\s*[^,}}\]]+", "", content)
            content = re.sub(rf"{key}\s*=\s*[^,}}\]]+", "", content)
            content = re.sub(rf",\s*{key}:\s*[^,}}\]]+", "", content)
            content = re.sub(rf",\s*{key}\s*=\s*[^,}}\]]+", "", content)
        
        # 딕셔너리 키 패턴 제거 (예: 'text': '...', 'content': '...')
        content = re.sub(r"'text':\s*'[^']*'", "", content)
        content = re.sub(r'"text":\s*"[^"]*"', "", content)
        content = re.sub(r"'content':\s*'[^']*'", "", content)
        content = re.sub(r'"content":\s*"[^"]*"', "", content)
        
        # 불필요한 공백 정리
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 빈 괄호나 중괄호 제거
        content = re.sub(r'\(\s*\)', '', content)
        content = re.sub(r'\{\s*\}', '', content)
        content = re.sub(r'\[\s*\]', '', content)
        
        # 연속된 쉼표나 콜론 제거
        content = re.sub(r',\s*,+', ',', content)
        content = re.sub(r':\s*:+\s*', ':', content)
        
        return content
    
    def _smart_truncate_document(self, content: str, max_chars: int, query: str) -> str:
        """
        질문과 관련된 부분을 우선 유지하며 문서 축약
        
        Args:
            content: 원본 문서 내용
            max_chars: 최대 문자 수
            query: 사용자 질문
            
        Returns:
            str: 축약된 문서 내용
        """
        if len(content) <= max_chars:
            return content
        
        # 질문 키워드 추출
        query_keywords = set(query.split())
        
        # 문장 단위로 분리
        sentences = content.split('。')  # 마침표 기준
        if len(sentences) == 1:
            sentences = content.split('.')
        if len(sentences) == 1:
            sentences = content.split('\n')
        
        # 관련도 점수 계산
        scored_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            score = 0
            sentence_lower = sentence.lower()
            for keyword in query_keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            scored_sentences.append((score, sentence))
        
        # 관련도 순 정렬
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 문장들 선택
        selected_sentences = []
        current_length = 0
        
        # 관련도 높은 문장 우선
        for score, sentence in scored_sentences:
            if current_length + len(sentence) <= max_chars * 0.8:  # 80%는 관련 문장
                selected_sentences.append(sentence)
                current_length += len(sentence)
        
        # 나머지 공간은 원본 순서대로
        remaining_chars = max_chars - current_length
        if remaining_chars > 100:
            for sentence in sentences:
                if sentence not in [s[1] for s in selected_sentences]:
                    if current_length + len(sentence) <= max_chars:
                        selected_sentences.append(sentence)
                        current_length += len(sentence)
                    else:
                        # 마지막 문장 부분 포함
                        remaining = max_chars - current_length
                        if remaining > 50:
                            selected_sentences.append(sentence[:remaining] + "...")
                        break
        
        if selected_sentences:
            return "。".join(selected_sentences) if "。" in content else " ".join(selected_sentences)
        else:
            return content[:max_chars] + "..."
    
    def _emergency_truncate_prompt(self, prompt: str, max_tokens: int, base_prompt: str, query: str) -> str:
        """
        긴급 상황에서 프롬프트 축약 (문서 섹션만 축약)
        
        Args:
            prompt: 원본 프롬프트
            max_tokens: 최대 토큰 수
            base_prompt: 기본 프롬프트
            query: 질문
            
        Returns:
            str: 축약된 프롬프트
        """
        base_tokens = self._estimate_tokens(base_prompt)
        query_tokens = self._estimate_tokens(query)
        reserved_tokens = base_tokens + query_tokens + 500  # 여유 공간
        
        available_doc_tokens = max_tokens - reserved_tokens
        
        # 문서 섹션 찾기
        doc_section_start = prompt.find("## 검색된 법률 문서")
        if doc_section_start == -1:
            # 문서 섹션이 없으면 그대로 반환
            max_chars = int(max_tokens * 2.5)
            return prompt[:max_chars]
        
        # 문서 섹션만 축약
        doc_section = prompt[doc_section_start:]
        doc_section_tokens = self._estimate_tokens(doc_section)
        
        if doc_section_tokens > available_doc_tokens:
            # 문서 섹션 축약
            max_doc_chars = int(available_doc_tokens * 2.5)
            truncated_doc_section = doc_section[:max_doc_chars] + "\n\n(문서 내용이 길어 일부 생략되었습니다.)"
            
            # 프롬프트 재구성
            truncated_prompt = prompt[:doc_section_start] + truncated_doc_section
            
            logger.warning(
                f"⚠️ [EMERGENCY TRUNCATION] Document section truncated: "
                f"{doc_section_tokens:,} → {self._estimate_tokens(truncated_doc_section):,} tokens"
            )
            
            return truncated_prompt
        
        return prompt
    
    def _optimize_documents_for_prompt(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """프롬프트용 문서 최적화 - 중복 제거 및 메타데이터 정리"""
        if not docs:
            return []
        
        seen_contents = set()
        optimized_docs = []
        
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            
            # 멀티 질의 메타데이터 확인
            has_multi_query = (
                isinstance(doc.get("metadata"), dict) and 
                doc.get("metadata", {}).get("sub_query")
            )
            
            # TASK 3: 법률 정보 확인
            doc_type = doc.get("type") or doc.get("source_type") or (doc.get("metadata", {}).get("type") if isinstance(doc.get("metadata"), dict) else None)
            has_law_info = bool(
                doc.get("law_name") or 
                doc.get("statute_name") or
                doc.get("article_no") or 
                doc.get("article_number") or
                doc_type in ["statute_article", "statute"] or
                (isinstance(doc.get("metadata"), dict) and (
                    doc.get("metadata", {}).get("law_name") or
                    doc.get("metadata", {}).get("statute_name") or
                    doc.get("metadata", {}).get("article_no") or
                    doc.get("metadata", {}).get("type") in ["statute_article", "statute"]
                ))
            )
            
            # content 정리
            content = doc.get("content", "").strip()
            
            # 멀티 질의 결과 또는 법률 정보가 있으면 content 길이 체크 완화
            min_content_length = self.MIN_CONTENT_LENGTH_WITH_LAW_INFO if (has_multi_query or has_law_info) else self.MIN_CONTENT_LENGTH
            
            if not content or len(content) < min_content_length:
                if has_multi_query or has_law_info:
                    # 🔥 개선: 법률 정보가 있으면 법률 정보를 포함한 고유한 content 생성
                    if has_law_info:
                        law_name = doc.get("law_name") or doc.get("statute_name") or ""
                        article_no = doc.get("article_no") or doc.get("article_number") or ""
                        if law_name and article_no:
                            # 법률 정보로 고유한 content 생성 (중복 방지)
                            content = f"{law_name} 제{article_no}조"
                            # 원본 content가 있으면 추가
                            original_content = doc.get("content", "").strip()
                            if original_content:
                                content = f"{content}: {original_content}"
                            doc["content"] = content
                            logger.debug(f"✅ [OPTIMIZE] Created unique content from law info: {content[:50]}")
                        else:
                            # 법률 정보가 불완전하면 source + 원본 content 조합
                            source = doc.get("source", "") or doc.get("title", "")
                            original_content = doc.get("content", "").strip()
                            if source and original_content:
                                content = f"{source}: {original_content}"
                            elif source:
                                content = source
                            elif original_content:
                                content = original_content
                            else:
                                continue
                            doc["content"] = content
                            logger.debug(f"✅ [OPTIMIZE] Doc with law-info: using combined content: {content[:50]}")
                    else:
                        # 멀티 질의 결과만 있는 경우
                        source = doc.get("source", "") or doc.get("title", "")
                        original_content = doc.get("content", "").strip()
                        if source and original_content:
                            content = f"{source}: {original_content}"
                        elif source:
                            content = source
                        elif original_content:
                            content = original_content
                        else:
                            continue
                        doc["content"] = content
                        logger.debug(f"✅ [OPTIMIZE] Doc with multi-query: using combined content: {content[:50]}")
                else:
                    continue
            
            # 🔥 개선: 중복 체크 시 법률 정보도 함께 고려 (article_no가 다르면 다른 문서로 간주)
            # 법률 정보가 있으면 law_name + article_no로 중복 체크
            if has_law_info:
                law_name = doc.get("law_name") or doc.get("statute_name") or ""
                article_no = doc.get("article_no") or doc.get("article_number") or ""
                if law_name and article_no:
                    unique_key = f"{law_name}_{article_no}"
                    if unique_key in seen_contents:
                        logger.debug(f"⚠️ [OPTIMIZE] Duplicate document removed (law_info): {unique_key}")
                        continue
                    seen_contents.add(unique_key)
                else:
                    # 법률 정보가 불완전하면 content 해시 사용
                    content_hash = hash(content[:200])
                    if content_hash in seen_contents:
                        logger.debug(f"⚠️ [OPTIMIZE] Duplicate document removed (content_hash): {content_hash}")
                        continue
                    seen_contents.add(content_hash)
            else:
                # 법률 정보가 없으면 content 해시로 중복 체크
                content_hash = hash(content[:200])
                if content_hash in seen_contents:
                    logger.debug(f"⚠️ [OPTIMIZE] Duplicate document removed (content_hash): {content_hash}")
                    continue
                seen_contents.add(content_hash)
            
            optimized_docs.append(doc)
        
        # 관련성 점수 기준 정렬
        optimized_docs.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        # 최대 문서 수만 반환 (멀티 질의 결과는 더 많이 포함)
        # 멀티 질의 메타데이터 확인
        has_multi_query = any(
            isinstance(doc, dict) and 
            isinstance(doc.get("metadata"), dict) and 
            doc.get("metadata", {}).get("sub_query")
            for doc in optimized_docs
        )
        max_docs = self.MAX_DOCUMENTS * 2 if has_multi_query else self.MAX_DOCUMENTS
        logger.debug(f"🔍 [OPTIMIZE] Returning {min(len(optimized_docs), max_docs)}/{len(optimized_docs)} docs (multi_query={has_multi_query})")
        return optimized_docs[:max_docs]
    
    def _extract_field(self, doc: Dict[str, Any], metadata: Dict[str, Any], field_names: List[str]) -> str:
        """문서와 메타데이터에서 필드 추출 (헬퍼 메서드)"""
        for field_name in field_names:
            value = doc.get(field_name, "") or metadata.get(field_name, "")
            if value:
                return str(value).strip()
        return ""
    
    def _generate_document_id(self, doc: Dict[str, Any]) -> str:
        """문서 고유 ID 생성 (중복 체크용)"""
        # 법령 조문: law_name + article_no
        law_name = doc.get("law_name", "")
        article_no = doc.get("article_no", "")
        if law_name and article_no:
            return f"law_{law_name}_{article_no}"
        
        # 판례: court + case_number 또는 case_name
        court = doc.get("court", "")
        case_number = doc.get("case_number", "")
        case_name = doc.get("case_name", "")
        if case_number:
            return f"case_{court}_{case_number}" if court else f"case_{case_number}"
        elif case_name:
            return f"case_{court}_{case_name}" if court else f"case_{case_name}"
        
        # 기타: content의 처음 100자 해시
        content = doc.get("content", "")
        if content:
            import hashlib
            content_hash = hashlib.md5(content[:100].encode('utf-8')).hexdigest()[:8]
            return f"doc_{content_hash}"
        
        # 최후의 수단: source
        source = doc.get("source", "")
        if source:
            return f"source_{source[:50]}"
        
        return "unknown"
    
    def _enhanced_deduplicate_before_prompt(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        similarity_threshold: float = 0.85,
        mmr_lambda: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        프롬프트 실행 전 강화된 중복 제거 (참고: RAG 중복 문서 제거 방법)
        
        다층 중복 제거 전략:
        1. Document-level distinct selection (같은 document_id의 chunk 중 최고 점수만)
        2. Exact duplicate 제거 (MD5 해시)
        3. Near-duplicate 제거 (텍스트 유사도 ≥ threshold)
        4. MMR 적용 (다양성 확보 + 중복 제거)
        
        Args:
            documents: 중복 제거할 문서 리스트
            query: 검색 쿼리 (MMR 계산용)
            similarity_threshold: Near-duplicate 판단 임계값 (기본 0.85)
            mmr_lambda: MMR 람다 파라미터 (0.0=다양성만, 1.0=관련성만, 기본 0.7)
        
        Returns:
            중복 제거된 문서 리스트
        """
        if not documents:
            return []
        
        # 환경 변수로 임계값 조정 가능
        similarity_threshold = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", str(similarity_threshold)))
        mmr_lambda = float(os.getenv("MMR_LAMBDA", str(mmr_lambda)))
        enable_enhanced_dedup = os.getenv("ENABLE_ENHANCED_DEDUP", "true").lower() == "true"
        
        if not enable_enhanced_dedup:
            # 기본 중복 제거만 수행
            return self._basic_deduplicate(documents)
        
        original_count = len(documents)
        logger.info(f"🔍 [ENHANCED DEDUP] Starting enhanced deduplication: {original_count} documents")
        
        # 🔥 개선: 최소 문서 수 계산 (문서 수에 따라 동적 설정)
        if original_count >= 5:
            min_docs_required = 3  # 5개 이상이면 최소 3개 보장
        elif original_count >= 3:
            min_docs_required = 2  # 3-4개면 최소 2개 보장
        else:
            min_docs_required = original_count  # 1-2개면 모두 보장
        
        # 1단계: Document-level distinct selection
        # 같은 document_id/source_id의 chunk 중 가장 점수 높은 것만 선택
        deduplicated = self._document_level_distinct_selection(documents)
        logger.info(f"✅ [ENHANCED DEDUP] Step 1 (Document-level distinct): {original_count} → {len(deduplicated)}")
        
        # 2단계: Exact duplicate 제거 (MD5 해시)
        deduplicated = self._remove_exact_duplicates(deduplicated)
        logger.info(f"✅ [ENHANCED DEDUP] Step 2 (Exact duplicates): {len(deduplicated)} documents")
        
        # 3단계: Near-duplicate 제거 (텍스트 유사도 기반)
        deduplicated = self._remove_near_duplicates(deduplicated, similarity_threshold)
        logger.info(f"✅ [ENHANCED DEDUP] Step 3 (Near-duplicates): {len(deduplicated)} documents")
        
        # 4단계: MMR 적용 (다양성 확보 + 최종 중복 제거)
        deduplicated = self._apply_mmr_deduplication(deduplicated, query, mmr_lambda)
        logger.info(f"✅ [ENHANCED DEDUP] Step 4 (MMR): {len(deduplicated)} documents")
        
        # 🔥 개선: 최소 문서 수 보장
        if len(deduplicated) < min_docs_required:
            logger.warning(
                f"⚠️ [ENHANCED DEDUP] Deduplication removed too many documents: "
                f"{original_count} → {len(deduplicated)} (min required: {min_docs_required}). "
                f"Restoring documents to meet minimum requirement."
            )
            
            # 점수 순으로 정렬하여 상위 문서 추가
            all_docs_sorted = sorted(
                documents,
                key=lambda x: self._get_document_score(x),
                reverse=True
            )
            
            # 이미 포함된 문서의 키 수집
            included_keys = set()
            for doc in deduplicated:
                doc_key = self._get_document_group_key(doc)
                if doc_key:
                    included_keys.add(doc_key)
            
            # 최소 문서 수까지 추가
            for doc in all_docs_sorted:
                if len(deduplicated) >= min_docs_required:
                    break
                
                doc_key = self._get_document_group_key(doc)
                if doc_key and doc_key not in included_keys:
                    deduplicated.append(doc)
                    included_keys.add(doc_key)
                    logger.debug(f"✅ [ENHANCED DEDUP] Restored document to meet minimum: {doc_key}")
        
        removed_count = original_count - len(deduplicated)
        if removed_count > 0:
            logger.info(
                f"🎯 [ENHANCED DEDUP] Final result: {original_count} → {len(deduplicated)} "
                f"({removed_count} duplicates removed, {removed_count/original_count*100:.1f}%, "
                f"min required: {min_docs_required})"
            )
        else:
            logger.info(
                f"✅ [ENHANCED DEDUP] Final result: {original_count} → {len(deduplicated)} "
                f"(no duplicates found, all documents are unique, min required: {min_docs_required})"
            )
        
        return deduplicated
    
    def _document_level_distinct_selection(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Document-level distinct selection
        같은 document_id/source_id의 chunk 중 가장 점수 높은 것만 선택
        (참고: PostgreSQL DISTINCT ON 방식)
        """
        if not documents:
            return []
        
        # 문서 그룹별로 최고 점수 문서만 선택
        doc_groups = {}  # {doc_key: best_doc}
        
        for doc in documents:
            # 문서 식별 키 생성
            doc_key = self._get_document_group_key(doc)
            if not doc_key:
                # 키를 생성할 수 없으면 그대로 추가 (나중에 처리)
                continue
            
            # 관련성 점수 추출 (여러 점수 중 최고값 사용)
            score = self._get_document_score(doc)
            
            # 같은 문서 그룹이 없거나 현재 문서가 더 높은 점수면 교체
            if doc_key not in doc_groups or score > self._get_document_score(doc_groups[doc_key]):
                doc_groups[doc_key] = doc
        
        # 그룹화되지 않은 문서들도 추가
        result = list(doc_groups.values())
        for doc in documents:
            doc_key = self._get_document_group_key(doc)
            if not doc_key:
                # 키를 생성할 수 없는 문서는 그대로 추가
                result.append(doc)
        
        return result
    
    def _get_document_group_key(self, doc: Dict[str, Any]) -> Optional[str]:
        """문서 그룹 키 생성 (document_id 또는 source_id 기반)"""
        import re
        
        source_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("type")
        
        # 🔥 개선: 법령 조문의 경우 조문 번호를 정확히 추출 (우선순위 최상위)
        if source_type == "statute_article":
            # article_no 추출 (여러 위치에서 확인)
            article_no = (
                doc.get("article_no") or 
                doc.get("metadata", {}).get("article_no") or
                doc.get("metadata", {}).get("article_number") or
                ""
            )
            
            # article_no가 문자열인 경우 숫자만 추출
            if article_no:
                # "066800" -> "668", "제668조" -> "668" 형식으로 정규화
                article_no_match = re.search(r'(\d+)', str(article_no))
                if article_no_match:
                    article_no = article_no_match.group(1)
            
            law_name = (
                doc.get("law_name") or 
                doc.get("metadata", {}).get("law_name") or
                doc.get("metadata", {}).get("statute_name") or
                ""
            )
            
            # 법령명이 없으면 content에서 추출 시도
            if not law_name:
                content = doc.get("content", "") or doc.get("text", "")
                if content:
                    # "민법 제668조" 패턴 추출
                    law_match = re.search(r'([가-힣]+법)\s*제\s*(\d+)\s*조', content)
                    if law_match:
                        law_name = law_match.group(1)
                        if not article_no:
                            article_no = law_match.group(2)
            
            # 조문 번호가 있으면 법령명과 조문 번호를 조합하여 고유 키 생성
            if article_no:
                law_name_normalized = law_name.strip() if law_name else "unknown_law"
                return f"statute_article_{law_name_normalized}_{article_no}"
        
        # 우선순위 1: source_id + source_type
        source_id = doc.get("source_id") or doc.get("metadata", {}).get("source_id")
        
        if source_id and source_type:
            return f"{source_type}_{source_id}"
        
        # 우선순위 2: doc_id
        doc_id = doc.get("doc_id") or doc.get("id") or doc.get("metadata", {}).get("doc_id")
        if doc_id and source_type:
            return f"{source_type}_doc_{doc_id}"
        
        # 우선순위 3: 법령 조문 (law_name + article_no) - 기존 로직 (fallback)
        law_name = doc.get("law_name") or doc.get("metadata", {}).get("law_name")
        article_no = doc.get("article_no") or doc.get("metadata", {}).get("article_no")
        if law_name and article_no:
            return f"law_{law_name}_{article_no}"
        
        # 우선순위 4: 판례 (case_number 또는 case_name)
        case_number = doc.get("case_number") or doc.get("metadata", {}).get("case_number")
        case_name = doc.get("case_name") or doc.get("metadata", {}).get("case_name")
        if case_number:
            court = doc.get("court") or doc.get("metadata", {}).get("court", "")
            return f"case_{court}_{case_number}" if court else f"case_{case_number}"
        elif case_name:
            court = doc.get("court") or doc.get("metadata", {}).get("court", "")
            return f"case_{court}_{case_name}" if court else f"case_{case_name}"
        
        return None
    
    def _get_document_score(self, doc: Dict[str, Any]) -> float:
        """문서의 관련성 점수 추출 (여러 점수 중 최고값)"""
        scores = [
            doc.get("combined_relevance_score", 0.0),
            doc.get("cross_encoder_score", 0.0),
            doc.get("relevance_score", 0.0),
            doc.get("final_weighted_score", 0.0),
            doc.get("score", 0.0)
        ]
        return max([s for s in scores if s and isinstance(s, (int, float))], default=0.0)
    
    def _remove_exact_duplicates(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Exact duplicate 제거 (MD5 해시 기반)"""
        if not documents:
            return []
        
        import hashlib
        seen_hashes = set()
        deduplicated = []
        
        for doc in documents:
            content = doc.get("content", "") or doc.get("text", "")
            if not content:
                deduplicated.append(doc)
                continue
            
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated.append(doc)
            else:
                logger.debug(f"⚠️ [EXACT DEDUP] Removed exact duplicate: {content_hash[:16]}...")
        
        return deduplicated
    
    def _remove_near_duplicates(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Near-duplicate 제거 (텍스트 유사도 기반)
        cosine similarity ≥ threshold인 문서 중 하나만 유지
        """
        import re
        
        if not documents or len(documents) <= 1:
            return documents
        
        deduplicated = []
        seen_docs = []  # 이미 선택된 문서들
        
        # 점수 순으로 정렬 (높은 점수 우선)
        sorted_docs = sorted(
            documents,
            key=lambda x: self._get_document_score(x),
            reverse=True
        )
        
        for doc in sorted_docs:
            content = doc.get("content", "") or doc.get("text", "")
            if not content:
                deduplicated.append(doc)
                continue
            
            # 🔥 개선: 법령 조문의 경우 조문 번호가 다르면 중복으로 간주하지 않음
            doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("type")
            if doc_type == "statute_article":
                # 조문 번호 추출
                doc_article_no = (
                    doc.get("article_no") or 
                    doc.get("metadata", {}).get("article_no") or
                    ""
                )
                if doc_article_no:
                    doc_article_match = re.search(r'(\d+)', str(doc_article_no))
                    if doc_article_match:
                        doc_article_no = doc_article_match.group(1)
                
                # 이미 선택된 문서들과 조문 번호 비교
                is_duplicate_by_article = False
                for seen_doc in seen_docs:
                    seen_type = seen_doc.get("type") or seen_doc.get("source_type") or seen_doc.get("metadata", {}).get("type")
                    if seen_type == "statute_article":
                        seen_article_no = (
                            seen_doc.get("article_no") or 
                            seen_doc.get("metadata", {}).get("article_no") or
                            ""
                        )
                        if seen_article_no:
                            seen_article_match = re.search(r'(\d+)', str(seen_article_no))
                            if seen_article_match:
                                seen_article_no = seen_article_match.group(1)
                        
                        # 조문 번호가 같으면 중복으로 간주 (텍스트 유사도 확인 불필요)
                        if doc_article_no and seen_article_no and doc_article_no == seen_article_no:
                            is_duplicate_by_article = True
                            logger.debug(f"⚠️ [NEAR DEDUP] Removed duplicate statute article (article_no={doc_article_no})")
                            break
                
                # 조문 번호가 같으면 중복으로 간주하여 스킵
                if is_duplicate_by_article:
                    continue
            
            # 이미 선택된 문서들과 유사도 계산
            is_near_duplicate = False
            for seen_doc in seen_docs:
                similarity = self._calculate_text_similarity_fast(doc, seen_doc)
                if similarity >= similarity_threshold:
                    # 유사도가 높은 경우, 점수 비교
                    current_score = self._get_document_score(doc)
                    seen_score = self._get_document_score(seen_doc)
                    
                    if current_score > seen_score:
                        # 현재 문서가 더 높은 점수면 기존 문서 제거하고 현재 문서 추가
                        logger.debug(
                            f"🔄 [NEAR DEDUP] Replaced similar document "
                            f"(similarity={similarity:.2f}, score: {seen_score:.2f} → {current_score:.2f})"
                        )
                        deduplicated.remove(seen_doc)
                        seen_docs.remove(seen_doc)
                        is_near_duplicate = False
                        break
                    else:
                        # 기존 문서가 더 높은 점수면 현재 문서 스킵
                        logger.debug(
                            f"⚠️ [NEAR DEDUP] Removed similar document "
                            f"(similarity={similarity:.2f}, score: {current_score:.2f} <= {seen_score:.2f})"
                        )
                        is_near_duplicate = True
                        break
            
            if not is_near_duplicate:
                deduplicated.append(doc)
                seen_docs.append(doc)
        
        return deduplicated
    
    def _calculate_text_similarity_fast(
        self,
        doc1: Dict[str, Any],
        doc2: Dict[str, Any]
    ) -> float:
        """
        빠른 텍스트 유사도 계산 (Jaccard similarity + 키워드 유사도)
        프롬프트 실행 전 단계에서 사용 (성능 최적화)
        """
        content1 = doc1.get("content", "") or doc1.get("text", "") or ""
        content2 = doc2.get("content", "") or doc2.get("text", "") or ""
        
        if not content1 or not content2:
            return 0.0
        
        # 1. Jaccard 유사도 (단어 기반)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_sim = intersection / union if union > 0 else 0.0
        else:
            jaccard_sim = 0.0
        
        # 2. 법률 용어 키워드 유사도 (법률 도메인 특화)
        keywords1 = self._extract_legal_keywords_fast(content1)
        keywords2 = self._extract_legal_keywords_fast(content2)
        
        if keywords1 or keywords2:
            keyword_intersection = len(keywords1.intersection(keywords2))
            keyword_union = len(keywords1.union(keywords2))
            keyword_sim = keyword_intersection / keyword_union if keyword_union > 0 else 0.0
        else:
            keyword_sim = 0.0
        
        # 3. 문서 타입 유사도
        type1 = doc1.get("type", "").lower() if doc1.get("type") else ""
        type2 = doc2.get("type", "").lower() if doc2.get("type") else ""
        type_sim = 1.0 if type1 == type2 and type1 else 0.0
        
        # 가중 평균 (법률 키워드에 더 높은 가중치)
        combined_sim = (
            0.3 * jaccard_sim +      # 단어 유사도
            0.5 * keyword_sim +       # 법률 키워드 유사도 (높은 가중치)
            0.2 * type_sim            # 타입 유사도
        )
        
        return min(1.0, combined_sim)
    
    def _extract_legal_keywords_fast(self, text: str, max_keywords: int = 10) -> set:
        """법률 문서에서 핵심 키워드 빠르게 추출"""
        if not text:
            return set()
        
        import re
        
        keywords = set()
        
        # 법률 용어 패턴 추출
        legal_patterns = [
            r'([가-힣]+법)',  # 법률명
            r'제\s*(\d+)\s*조',  # 조문 번호
            r'법\s*제\s*(\d+)\s*조',  # 법 제N조
            r'대법원\s*(\d{4}\.\d{1,2}\.\d{1,2})',  # 판례 날짜
            r'판결\s*(\d+[가-힣]+\d+)',  # 판결 번호
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text[:1000])  # 처음 1000자만 검색 (성능 최적화)
            for match in matches[:3]:  # 각 패턴당 최대 3개
                if isinstance(match, tuple):
                    keywords.update([m for m in match if m])
                else:
                    keywords.add(match)
        
        # 2글자 이상 한글 단어 추출 (법률 용어 중심)
        words = re.findall(r'[가-힣]{2,}', text[:500])  # 처음 500자만
        word_freq = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 빈도순 정렬하여 상위 키워드 추가
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords.update([word for word, _ in sorted_words[:max_keywords - len(keywords)]])
        
        return keywords
    
    def _apply_mmr_deduplication(
        self,
        documents: List[Dict[str, Any]],
        query: str = "",
        mmr_lambda: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        MMR (Maximal Marginal Relevance) 적용
        중복 제거 + 다양성 확보
        """
        if not documents or len(documents) <= 1:
            return documents
        
        # 점수 순으로 정렬
        sorted_docs = sorted(
            documents,
            key=lambda x: self._get_document_score(x),
            reverse=True
        )
        
        selected = []
        remaining = sorted_docs.copy()
        
        # 첫 번째 문서는 가장 관련성 높은 것으로 선택
        if remaining:
            selected.append(remaining.pop(0))
        
        # MMR 알고리즘 적용
        while remaining and len(selected) < len(documents):  # 모든 문서를 검토하되 중복 제거
            best_doc = None
            best_mmr_score = -float('inf')
            
            for candidate in remaining:
                # 관련성 점수
                relevance = self._get_document_score(candidate)
                
                # 다양성 점수 (이미 선택된 문서와의 최대 유사도)
                max_similarity = 0.0
                for selected_doc in selected:
                    similarity = self._calculate_text_similarity_fast(candidate, selected_doc)
                    max_similarity = max(max_similarity, similarity)
                
                diversity = 1.0 - max_similarity
                
                # MMR 점수
                mmr_score = mmr_lambda * relevance + (1 - mmr_lambda) * diversity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_doc = candidate
            
            if best_doc:
                # 유사도가 너무 높으면 제거 (중복으로 간주)
                max_sim_with_selected = max([
                    self._calculate_text_similarity_fast(best_doc, sel_doc)
                    for sel_doc in selected
                ]) if selected else 0.0
                
                if max_sim_with_selected >= 0.9:  # 매우 유사한 문서는 제거
                    logger.debug(
                        f"⚠️ [MMR DEDUP] Removed highly similar document "
                        f"(similarity={max_sim_with_selected:.2f}, mmr_score={best_mmr_score:.2f})"
                    )
                    remaining.remove(best_doc)
                else:
                    selected.append(best_doc)
                    remaining.remove(best_doc)
            else:
                break
        
        return selected
    
    def _basic_deduplicate(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """기본 중복 제거 (강화된 중복 제거 비활성화 시 사용)"""
        import hashlib
        seen_hashes = set()
        seen_doc_ids = set()
        deduplicated = []
        
        for doc in documents:
            # MD5 해시 기반 중복 제거
            content = doc.get("content", "") or doc.get("text", "")
            if content:
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
            
            # 문서 ID 기반 중복 제거
            doc_id = self._generate_document_id(doc)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                deduplicated.append(doc)
        
        return deduplicated
    
    def _get_summary_agent(self):
        """요약 에이전트 가져오기 (지연 초기화)"""
        if self._summary_agent is None:
            try:
                # 직접 파일 경로로 로드
                import sys
                import importlib.util
                from pathlib import Path
                
                # 현재 파일 경로 기준으로 상대 경로 계산
                current_file = Path(__file__).resolve()
                # lawfirm_langgraph/core/services/unified_prompt_manager.py
                # -> lawfirm_langgraph/core/agents/handlers/document_summary_agent.py
                agent_file = current_file.parent.parent / "agents" / "handlers" / "document_summary_agent.py"
                
                if agent_file.exists():
                    # 파일이 존재하면 직접 로드
                    spec = importlib.util.spec_from_file_location("document_summary_agent", agent_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # sys.modules에 등록하여 중복 로드 방지
                        sys.modules["document_summary_agent"] = module
                        spec.loader.exec_module(module)
                        DocumentSummaryAgent = module.DocumentSummaryAgent
                    else:
                        raise ImportError(f"Cannot load module from {agent_file}")
                else:
                    # 파일이 없으면 일반 import 시도
                    from lawfirm_langgraph.core.agents.handlers.document_summary_agent import DocumentSummaryAgent
                
                self._summary_agent = DocumentSummaryAgent(
                    llm=None,  # 필요시 주입
                    llm_fast=self.llm_fast,  # UnifiedPromptManager의 llm_fast 사용
                    logger=logger
                )
            except Exception as e:
                logger.warning(f"DocumentSummaryAgent를 가져올 수 없습니다. 요약 기능이 비활성화됩니다: {e}")
                return None
        return self._summary_agent
    
    def _extract_article_numbers_from_query(self, query: str) -> List[str]:
        """질문에서 조문 번호 추출 (예: "민법 제750조" -> ["750"])"""
        import re
        # "제XXX조" 패턴 추출
        patterns = [
            r'제\s*(\d+)\s*조',  # 제750조
            r'(\d+)\s*조',  # 750조
        ]
        article_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            article_numbers.extend(matches)
        return list(set(article_numbers))  # 중복 제거
    
    def _check_query_document_mismatch(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """질문-문서 불일치 검사 - 개선된 매칭 로직"""
        # 질문에서 조문 번호 추출
        query_articles = self._extract_article_numbers_from_query(query)
        if not query_articles:
            return ""
        
        # 문서에서 조문 번호 추출 (개선)
        doc_articles = set()
        for doc in documents:
            # article_no 필드 확인 (다양한 형식 지원)
            article_no = doc.get("article_no", "")
            if article_no:
                article_no_str = str(article_no).strip()
                # 🔥 개선: "046800" 형식 처리 (마지막 3자리 또는 전체에서 앞의 0 제거)
                # "046800" = "468" 조문을 의미할 수 있음
                if len(article_no_str) >= 3:
                    # 마지막 3자리 시도
                    last_three = article_no_str[-3:].lstrip('0')
                    if last_three:
                        doc_articles.add(last_three)
                    # 전체에서 앞의 0 제거 시도
                    all_clean = article_no_str.lstrip('0')
                    if all_clean and len(all_clean) <= 4:  # 조문 번호는 보통 4자리 이하
                        doc_articles.add(all_clean)
                else:
                    # 짧은 형식은 그대로 사용
                    article_no_clean = article_no_str.lstrip('0')
                    if article_no_clean:
                        doc_articles.add(article_no_clean)
            
            # content에서도 추출
            content = doc.get("content", "") or doc.get("text", "")
            if content:
                import re
                # "제750조", "제 750 조" 등 다양한 형식 지원
                matches = re.findall(r'제\s*(\d+)\s*조', content[:1000])
                doc_articles.update([m.lstrip('0') for m in matches if m])
        
        # 불일치 확인
        missing_articles = []
        for q_article in query_articles:
            q_article_clean = q_article.lstrip('0')
            found = any(
                q_article_clean == d_article.lstrip('0') 
                for d_article in doc_articles
            )
            if not found:
                missing_articles.append(q_article)
        
        # 경고 메시지 생성
        if missing_articles:
            articles_str = ", ".join([f"제{art}조" for art in missing_articles])
            logger.info(f"⚠️ [QUERY-DOCUMENT MISMATCH] 질문 조문 {articles_str}이 문서에 없음")
            return f"""
⚠️ **중요 경고: 질문-문서 불일치**

질문에서 언급된 '{articles_str}'의 직접적인 내용이 검색된 문서에 포함되어 있지 않습니다.

**권장 사항**:
- 제공된 관련 조문을 참고하여 답변하세요
- 해당 조문의 직접적인 내용은 문서에 없으므로, 일반적인 법적 원칙을 바탕으로 설명하세요
- 불확실한 부분은 명확히 표시하세요

---

"""
        return ""
    
    def _build_documents_section(self, sorted_docs: List[Dict[str, Any]], query: str) -> str:
        """문서 섹션 생성 - 문서 ID 기반 일관된 번호 부여"""
        if not sorted_docs:
            return "\n\n## 검색된 참고 문서\n\n검색된 문서가 없습니다.\n"
        
        # 🔥 개선: 문서 ID 기반 번호 매핑 생성
        doc_id_to_number = {}
        for idx, doc in enumerate(sorted_docs, 1):
            doc_id = (
                doc.get("id") or 
                doc.get("chunk_id") or 
                doc.get("document_id") or
                f"doc_{idx}"  # 폴백
            )
            doc_id_to_number[doc_id] = idx
        
        # 🔥 개선: 질문-문서 불일치 경고 추가
        mismatch_warning = self._check_query_document_mismatch(query, sorted_docs)
        
        # 🔥 개선: 통합된 단일 문서 섹션 생성 (중복 제거)
        documents_section = "\n\n## 검색된 참고 문서\n\n"
        documents_section += f"다음 {len(sorted_docs)}개의 문서를 반드시 참고하여 답변하세요. 문서를 인용할 때는 `[문서 N]` 형식을 사용하세요.\n\n"
        
        # 질문-문서 불일치 경고 추가
        if mismatch_warning:
            documents_section += mismatch_warning
        
        # 문서별로 통합된 정보 표시 (일관된 번호 사용)
        for idx, doc in enumerate(sorted_docs, 1):
            # 🔥 개선: 문서 ID 기반 일관된 번호 사용
            doc_id = (
                doc.get("id") or 
                doc.get("chunk_id") or 
                doc.get("document_id") or
                f"doc_{idx}"
            )
            doc_number = doc_id_to_number.get(doc_id, idx)
            doc_title, _ = self._get_document_title_and_max_length(doc, idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            # 문서 내용 추출 (필요한 필드만 포함)
            content = self._extract_document_content_for_prompt(doc)
            
            # 🔥 개선: 문서 내용 잘림 방지 (기본값을 크게 설정, 환경 변수로 조정 가능)
            # 토큰 제한을 고려하여 기본값을 5000자로 설정 (기존 500자에서 증가)
            MAX_DOC_CONTENT_LENGTH = int(os.getenv("PROMPT_MAX_DOC_CONTENT_LENGTH", "5000"))
            
            original_content_length = len(content) if content else 0
            if content and len(content) > MAX_DOC_CONTENT_LENGTH:
                # 질문과 관련된 부분 우선 추출 (스마트 트렁케이션)
                content = self._smart_truncate_document(content, MAX_DOC_CONTENT_LENGTH, query)
                if len(content) > MAX_DOC_CONTENT_LENGTH:
                    # 문장 단위로 자르기 (문장 중간에서 자르지 않음)
                    sentences = content.split('。') if '。' in content else content.split('.')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated) + len(sentence) > MAX_DOC_CONTENT_LENGTH:
                            break
                        truncated += sentence + ("。" if '。' in content else ".")
                    content = truncated + "..." if truncated else content[:MAX_DOC_CONTENT_LENGTH] + "..."
                logger.debug(f"⚡ [DOC CONTENT] 문서 {doc_number} 내용 축약: {original_content_length}자 → {len(content)}자")
            elif content:
                logger.debug(f"✅ [DOC CONTENT] 문서 {doc_number} 전체 내용 포함: {original_content_length}자")
            
            # 요약 정보가 있으면 사용
            summary_data = doc.get("summary_data")
            has_summary = False
            if summary_data and isinstance(summary_data, dict):
                summary_text = summary_data.get("summary", "")
                key_points = summary_data.get("key_points", [])
                if summary_text or key_points:
                    has_summary = True
            
            # 문서 제목 및 관련도 (일관된 번호 사용)
            documents_section += f"**[문서 {doc_number}]** {doc_title} (관련도: {relevance_score:.2f})\n\n"
            
            # 요약이 있으면 요약 표시, 없으면 전체 내용 표시
            if has_summary:
                if summary_text:
                    # 🔥 개선: 요약도 길이 제한 완화 (전체 요약 포함)
                    # 요약은 일반적으로 짧으므로 전체 포함
                    documents_section += f"**핵심 내용**: {summary_text}\n\n"
                if key_points and isinstance(key_points, list):
                    # 🔥 개선: 핵심 쟁점도 더 많이 포함 (3개 → 5개)
                    key_points_str = "\n".join([f"- {kp}" for kp in key_points[:5]])
                    if key_points_str:
                        documents_section += f"**핵심 쟁점**:\n{key_points_str}\n\n"
            
            # 전체 내용 표시 (필요한 필드만 포함, 길이 제한 완화)
            if content:
                documents_section += f"**내용**:\n{content}\n\n"
            else:
                documents_section += "**내용**: (내용 없음)\n\n"
            
            documents_section += "---\n\n"
        
        # 🔥 개선: documents_section 끝의 중복 지시 제거 (instruction_section에 이미 포함)
        
        return documents_section
    
    def _get_document_title_and_max_length(self, doc: Dict[str, Any], idx: int) -> tuple[str, int]:
        """문서 제목과 최대 길이 반환 (개선: 출처 정보 명확화)"""
        law_name = doc.get("law_name", "")
        article_no = doc.get("article_no", "")
        clause_no = doc.get("clause_no", "")
        case_name = doc.get("case_name", "")
        court = doc.get("court", "")
        case_number = doc.get("case_number", "")
        source = doc.get("source", "")
        
        if law_name and article_no:
            # 법령 조문: "민법 제543조" 형식
            doc_title = f"{law_name} 제{article_no}조"
            if clause_no:
                doc_title += f" 제{clause_no}항"
            max_length = self.MAX_DOC_LENGTH_LAW
        elif case_name or court:
            # 판례: "대법원 판례명" 또는 "서울고등법원 판례명" 형식
            if court and case_name:
                doc_title = f"{court} {case_name}".strip()
            elif court:
                doc_title = court
            elif case_name:
                doc_title = case_name
            else:
                doc_title = source or f"판례 {idx}"
            if case_number:
                doc_title += f" ({case_number})"
            max_length = self.MAX_DOC_LENGTH_CASE
        else:
            # 기타 문서: source 필드 우선 사용
            doc_title = source or doc.get("title", "") or f"문서 {idx}"
            max_length = self.MAX_DOC_LENGTH_COMMENTARY
        
        return doc_title, max_length
    
    def _extract_document_content_for_prompt(self, doc: Dict[str, Any]) -> str:
        """
        프롬프트에 필요한 문서 내용만 추출 (필요한 필드만 포함)
        
        Args:
            doc: 원본 문서 딕셔너리
        
        Returns:
            필터링된 문서 내용 문자열
        """
        if not doc or not isinstance(doc, dict):
            return ""
        
        # 1. content 필드 추출 (우선순위: content > text)
        content = doc.get("content", "") or doc.get("text", "")
        
        if not content:
            return ""
        
        # 2. 불필요한 HTML 태그 및 특수 문자 정리
        content = self._clean_content(content)
        
        # 3. 문서 타입별 추가 정보 포함 (필요한 경우)
        doc_type = doc.get("type") or doc.get("source_type", "")
        
        # 법령 조문인 경우: 법률명과 조문 번호 정보 포함
        if doc_type in ["statute_article", "statute"]:
            law_name = doc.get("law_name") or doc.get("statute_name", "")
            article_no = doc.get("article_no") or doc.get("article_number", "")
            if law_name and article_no and f"{law_name} 제{article_no}조" not in content:
                # 법률 정보가 content에 없으면 앞에 추가
                content = f"{law_name} 제{article_no}조\n\n{content}"
        
        # 판례인 경우: 판례 정보 포함
        elif doc_type in ["precedent_content", "case", "precedent"]:
            case_name = doc.get("case_name", "")
            court = doc.get("court", "")
            case_number = doc.get("case_number", "")
            
            case_info_parts = []
            if court:
                case_info_parts.append(court)
            if case_name:
                case_info_parts.append(case_name)
            if case_number:
                case_info_parts.append(f"({case_number})")
            
            if case_info_parts and " ".join(case_info_parts) not in content[:200]:
                # 판례 정보가 content 앞부분에 없으면 추가
                case_info = " ".join(case_info_parts)
                content = f"{case_info}\n\n{content}"
        
        return content.strip()
    
    def _format_documents_for_context(
        self, 
        documents: List[Dict[str, Any]], 
        is_high_priority: bool = True
    ) -> List[str]:
        """문서 목록을 컨텍스트 형식으로 포맷팅 (중복 코드 제거)"""
        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            # 관련도 점수를 relevance_score로 변환
            doc_for_format = doc.copy()
            doc_for_format["relevance_score"] = doc.get("score", 0.0)
            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=is_high_priority)
            formatted_docs.append(formatted_doc)
        return formatted_docs
    
    def _format_legal_references(self, legal_references: List[Any]) -> List[str]:
        """법령 참조 목록을 포맷팅 (중복 코드 제거)"""
        formatted_refs = []
        for ref in legal_references:
            if isinstance(ref, str):
                formatted_refs.append(f"- {ref}")
            else:
                formatted_refs.append(f"- {ref.get('text', '') if isinstance(ref, dict) else str(ref)}")
        return formatted_refs
    
    def _should_use_summary(self, doc: Dict[str, Any]) -> bool:
        """문서가 요약이 필요한지 판단"""
        content = doc.get("content", "").strip()
        if not content:
            return False
        
        doc_type = self._get_document_type(doc)
        
        thresholds = {
            'law': self.SUMMARY_THRESHOLD_LAW,
            'case': self.SUMMARY_THRESHOLD_CASE,
            'commentary': self.SUMMARY_THRESHOLD_COMMENTARY
        }
        
        threshold = thresholds.get(doc_type, 500)
        return len(content) > threshold
    
    def _get_document_type(self, doc: Dict[str, Any]) -> str:
        """문서 유형 판단"""
        if doc.get("law_name") and doc.get("article_no"):
            return 'law'
        elif doc.get("court") or doc.get("case_name") or doc.get("case_number"):
            return 'case'
        elif doc.get("type") == "commentary" or "해설" in str(doc.get("title", "")):
            return 'commentary'
        else:
            return 'general'
    
    def _classify_document_type(self, doc: Dict[str, Any]) -> str:
        """
        문서 타입 분류 (법률/판례/기타)
        🔥 개선: 메타데이터 손실 후에도 content 기반 추론 지원
        
        Returns:
            "statute", "precedent", "other"
        """
        if not isinstance(doc, dict):
            return "other"
        
        # 1단계: 명시적 타입 필드 확인
        doc_type = doc.get("type", "").lower() if doc.get("type") else ""
        metadata = doc.get("metadata", {})
        
        if isinstance(metadata, dict):
            metadata_type = metadata.get("type", "").lower() if metadata.get("type") else ""
        else:
            metadata_type = ""
        
        # 2단계: 메타데이터 필드 확인
        # 법률 조문 판별 (메타데이터 필드)
        has_statute_fields = (
            doc.get("law_name") or doc.get("statute_name") or 
            doc.get("article_no") or doc.get("article_number") or
            doc.get("statute_id") or doc.get("statute_abbrv")
        )
        
        # 판례 판별 (메타데이터 필드)
        has_precedent_fields = (
            doc.get("case_id") or doc.get("court") or doc.get("ccourt") or
            doc.get("precedent_id") or doc.get("casenames") or
            doc.get("case_name") or doc.get("case_number") or
            doc.get("decision_date")
        )
        
        # 3단계: content 기반 추론 (메타데이터 손실 시 폴백)
        content = doc.get("content", "") or doc.get("text", "")
        source = doc.get("source", "")
        source_lower = source.lower() if source else ""
        
        # 🔥 개선: 판례 패턴을 우선 확인 (판례가 법률 조문 패턴도 포함할 수 있음)
        # 판례 패턴 (content 기반) - 우선 확인
        precedent_patterns = [
            r'【원고',  # 【원고, 피상고인】
            r'【피고',  # 【피고, 상고인】
            r'【청구인',  # 【청구인, 재항고인】
            r'【사건본인',  # 【사건본인】
            r'대법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',  # 대법원 2023. 9. 27.
            r'고등법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',
            r'지방법원.*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}',
            r'선고.*판결',  # 선고 2021다255655 판결
            r'선고.*결정',  # 선고 2017브10 결정
            r'원심판결',  # 【원심판결】
            r'원심결정',  # 【원심결정】
            r'소송대리인',  # 소송대리인 변호사
            r'담당변호사',  # 담당변호사 이종희
            r'사건번호',  # 사건번호
            r'사건.*\d+',  # 사건 2015르3081
            r'판결 참조',  # 판결 참조
            r'판례',  # 판례
        ]
        
        # 법률 조문 패턴 (content 기반) - 판례 패턴이 없을 때만 확인
        statute_patterns = [
            r'제\d+조\s*제\d+항',  # 제750조 제1항 (구체적인 조문 형식)
            r'제\d+조\s*제\d+호',  # 제750조 제1호
            r'법률.*제\d+조.*제\d+항',  # 법률 제750조 제1항
            r'민법.*제\d+조.*제\d+항',  # 민법 제750조 제1항
            r'형법.*제\d+조.*제\d+항',  # 형법 제750조 제1항
            r'상법.*제\d+조.*제\d+항',  # 상법 제750조 제1항
        ]
        
        has_statute_content = False
        has_precedent_content = False
        
        if content:
            # 🔥 개선: 판례 패턴을 먼저 확인 (판례가 법률 조문 패턴도 포함할 수 있음)
            for pattern in precedent_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    has_precedent_content = True
                    break
            
            # 판례 패턴이 없을 때만 법률 조문 패턴 확인
            if not has_precedent_content:
                for pattern in statute_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        has_statute_content = True
                        break
        
        # source 필드 기반 추론
        has_statute_source = (
            "statute" in source_lower or
            "법령" in source or
            "조문" in source
        )
        
        has_precedent_source = (
            "precedent" in source_lower or
            "판례" in source or
            "판결" in source or
            "대법원" in source or
            "법원" in source
        )
        
        # 4단계: 최종 판별 (우선순위: 명시적 타입 > 메타데이터 필드 > content 기반 > source 기반)
        is_statute = (
            "statute" in doc_type or
            "statute" in metadata_type or
            has_statute_fields or
            has_statute_content or
            has_statute_source
        )
        
        is_precedent = (
            "precedent" in doc_type or
            "precedent" in metadata_type or
            has_precedent_fields or
            has_precedent_content or
            has_precedent_source
        )
        
        # 로깅 (디버깅용)
        if doc_type == "unknown" or not doc_type:
            if is_statute or is_precedent:
                logger.debug(
                    f"🔍 [DOC TYPE INFERENCE] Content 기반 추론: "
                    f"type={doc_type}, inferred={'statute' if is_statute else 'precedent'}, "
                    f"has_fields={has_statute_fields or has_precedent_fields}, "
                    f"has_content={has_statute_content or has_precedent_content}, "
                    f"has_source={has_statute_source or has_precedent_source}"
                )
        
        if is_statute:
            return "statute"
        elif is_precedent:
            return "precedent"
        else:
            return "other"
    
    def _build_reference_list_section(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """참고 문서 목록 섹션 생성 (개선: 문서 수 명시, 관련성 점수 표시)"""
        if not sorted_docs:
            return ""
        
        section = "## 참고 문서 목록\n\n"
        section += f"다음 {len(sorted_docs)}개의 문서를 반드시 참고하여 답변을 생성하세요.\n"
        section += "각 문서는 관련성 점수와 핵심 내용이 표시되어 있습니다.\n\n"
        
        for idx, doc in enumerate(sorted_docs, 1):
            doc_title, _ = self._get_document_title_and_max_length(doc, idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            # 핵심 내용 추출 (요약이 있으면 사용, 없으면 content 일부)
            content = doc.get("content", "") or doc.get("text", "")
            summary_data = doc.get("summary_data")
            if summary_data and isinstance(summary_data, dict):
                key_content = summary_data.get("summary", "") or summary_data.get("key_points", [])
                if isinstance(key_content, list):
                    key_content = " ".join(key_content[:2]) if key_content else ""
            else:
                key_content = content[:200] + "..." if len(content) > 200 else content
            
            section += f"### 문서 {idx}: {doc_title} (관련성 점수: {relevance_score:.2f})\n\n"
            section += f"**핵심 내용:**\n{key_content}\n\n"
            section += "---\n\n"
        
        return section
    
    def _build_reference_content_section(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """참고 문서 내용 섹션 생성 (전체 문서 내용 표시)"""
        if not sorted_docs:
            return ""
        
        section = "## 참고 문서 내용\n\n"
        section += "다음은 각 문서의 전체 내용입니다.\n\n"
        
        for idx, doc in enumerate(sorted_docs, 1):
            doc_title, _ = self._get_document_title_and_max_length(doc, idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            content = doc.get("content", "") or doc.get("text", "")
            
            section += f"### 문서 {idx}: {doc_title} (ID: {doc.get('id', doc.get('chunk_id', doc.get('document_id', 'N/A')))}) (관련도: {relevance_score:.2f})\n\n"
            
            if content:
                # 내용이 너무 길면 축약
                max_length = 3000
                if len(content) > max_length:
                    content = content[:max_length] + "\n\n...(내용 생략)..."
                section += f"{content}\n\n"
            else:
                section += "(내용 없음)\n\n"
            
            section += "---\n\n"
        
        return section
    
    def _build_summary_section(
        self, 
        summaries: List[Dict[str, Any]], 
        original_docs: List[Dict[str, Any]],
        all_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Summary 섹션 생성 (개선: 질문 연관성, 핵심 쟁점 형식 통일)"""
        if not summaries or not original_docs:
            return ""
        
        section = "### [Context Summary]\n\n"
        section += "다음은 검색된 문서들의 요약입니다. 각 문서의 핵심 내용을 파악하세요.\n\n"
        
        for summary, doc in zip(summaries, original_docs):
            # 전체 문서 리스트에서의 인덱스 찾기
            try:
                doc_idx = all_docs.index(doc) + 1
            except ValueError:
                doc_idx = len(all_docs) + 1
            
            doc_title, _ = self._get_document_title_and_max_length(doc, doc_idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            section += f"**[문서 {doc_idx}]** {doc_title} (관련도: {relevance_score:.2f})\n\n"
            
            # 요약 내용
            summary_text = summary.get('summary', '')
            if summary_text:
                section += f"**요약:**\n{summary_text}\n\n"
            
            # 핵심 쟁점 (개선: 형식 통일)
            key_points = summary.get('key_points', [])
            if key_points:
                section += "**핵심 쟁점:**\n"
                for point in key_points[:3]:  # 최대 3개
                    if isinstance(point, str) and point.strip():
                        section += f"  • {point[:200]}\n"
                section += "\n"
            
            # 질문 연관성 (개선: 형식 통일)
            relevance_notes = summary.get('relevance_notes', '')
            if relevance_notes:
                section += f"**질문 연관성:**\n{relevance_notes}\n\n"
            
            section += "---\n\n"
        
        return section
    
    def _build_detailed_section(
        self, 
        docs: List[Dict[str, Any]], 
        all_docs: List[Dict[str, Any]],
        query: str,
        max_docs: int = 3,
        summary_map: dict = None
    ) -> str:
        """Detailed Extracts 섹션 생성 (개선: 핵심 내용 서브섹션, 문서 정보 추가)"""
        if not docs:
            return ""
        
        if summary_map is None:
            summary_map = {}
        
        section = "### [Detailed Extracts]\n\n"
        section += "다음은 질문과 직접 관련된 문서의 상세 내용입니다.\n\n"
        
        for doc in docs[:max_docs]:
            # 전체 문서 리스트에서의 인덱스 찾기
            try:
                doc_idx = all_docs.index(doc) + 1
            except ValueError:
                doc_idx = len(all_docs) + 1
            
            doc_title, _ = self._get_document_title_and_max_length(doc, doc_idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            # 전체 문서 길이 계산
            full_content = doc.get("content", "") or doc.get("text", "")
            full_length = len(full_content) if full_content else 0
            
            section += f"**[문서 {doc_idx}]** {doc_title} (관련도: {relevance_score:.2f})\n\n"
            
            # 🔥 개선: 요약 결과가 있으면 우선 사용
            summary = None
            doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
            if doc_id and doc_id in summary_map:
                summary = summary_map[doc_id]
            else:
                # content 기반 매핑 시도
                if full_content:
                    import hashlib
                    content_hash = str(hashlib.md5(full_content[:200].encode('utf-8')).hexdigest())
                    if content_hash in summary_map:
                        summary = summary_map[content_hash]
            
            # 핵심 내용 (질문과 직접 관련된 부분) 서브섹션 추가
            section += "**핵심 내용 (질문과 직접 관련된 부분):**\n\n"
            
            # 변수 초기화
            extracted_length = 0
            
            if summary and isinstance(summary, dict):
                # 요약 결과 사용
                summary_text = summary.get('summary', '')
                key_points = summary.get('key_points', [])
                
                if summary_text:
                    section += f"{summary_text}\n\n"
                    extracted_length = len(summary_text)
                
                if key_points:
                    section += "**핵심 쟁점:**\n"
                    for point in key_points[:3]:  # 최대 3개
                        if isinstance(point, str) and point.strip():
                            section += f"  • {point[:200]}\n"
                    section += "\n"
            else:
                # 요약이 없으면 기존 로직 사용
                detailed_content = self._extract_detailed_relevant_parts(
                    doc, query, self.MAX_DETAILED_EXTRACT_LENGTH
                )
                
                if detailed_content:
                    section += f"{detailed_content}\n\n"
                    extracted_length = len(detailed_content)
                else:
                    # 폴백: 스마트 축약
                    if full_content:
                        max_length = min(self.MAX_DETAILED_EXTRACT_LENGTH, len(full_content))
                        content = self._smart_truncate_document(full_content, max_length, query)
                        section += f"{content}\n\n"
                        extracted_length = len(content)
            
            # 🔥 개선: 문서 정보 간소화 (핵심 정보만 포함)
            # 문서 정보는 디버깅용이므로 프로덕션에서는 제거하거나 최소화
            import os
            SHOW_DOCUMENT_INFO = os.getenv("PROMPT_SHOW_DOCUMENT_INFO", "false").lower() == "true"
            
            if SHOW_DOCUMENT_INFO:
                # 디버깅 모드: 상세 정보 표시
                if not extracted_length and full_content:
                    extracted_length = min(self.MAX_DETAILED_EXTRACT_LENGTH, len(full_content))
                
                section += "**문서 정보:**\n"
                section += f"- 전체 문서 길이: {full_length:,}자\n"
                section += f"- 추출된 핵심 내용: {extracted_length:,}자\n"
                if full_length > 0:
                    reduction_ratio = (1 - extracted_length / full_length) * 100 if extracted_length < full_length else 0
                    section += f"- 축약 비율: {reduction_ratio:.1f}%\n"
                section += "\n---\n\n"
            else:
                # 프로덕션 모드: 간소화 (구분선만)
                section += "\n---\n\n"
        
        return section
    
    def _build_statute_explanation_section(
        self,
        sorted_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """조문 해설 섹션 생성 (법령 조문이 있는 경우)"""
        if not sorted_docs:
            return ""
        
        # 법령 조문 문서만 필터링
        statute_docs = []
        for doc in sorted_docs:
            if self._classify_document_type(doc) == "statute":
                statute_docs.append(doc)
        
        if not statute_docs:
            return ""
        
        section = "## 조문 해설\n\n"
        section += "다음은 검색된 법령 조문에 대한 해설입니다.\n\n"
        
        for idx, doc in enumerate(statute_docs, 1):
            doc_title, _ = self._get_document_title_and_max_length(doc, idx)
            content = doc.get("content", "") or doc.get("text", "")
            
            section += f"### {doc_title}\n\n"
            
            if content:
                # 내용이 너무 길면 축약
                max_length = 2000
                if len(content) > max_length:
                    content = content[:max_length] + "\n\n...(내용 생략)..."
                section += f"{content}\n\n"
            else:
                section += "(내용 없음)\n\n"
            
            section += "---\n\n"
        
        return section
    
    def _build_full_docs_section(
        self, 
        docs: List[Dict[str, Any]], 
        all_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """전체 문서 섹션 생성 (요약 불필요한 짧은 문서)"""
        if not docs:
            return ""
        
        section = "### [전체 문서]\n\n"
        section += "다음은 요약이 필요하지 않은 짧은 문서들입니다.\n\n"
        
        for doc in docs:
            # 전체 문서 리스트에서의 인덱스 찾기
            try:
                doc_idx = all_docs.index(doc) + 1
            except ValueError:
                doc_idx = len(all_docs) + 1
            
            doc_title, max_length = self._get_document_title_and_max_length(doc, doc_idx)
            
            # 관련도 점수 추출
            relevance_score = float(
                doc.get("relevance_score", 0.0) or
                doc.get("score", 0.0) or
                doc.get("final_weighted_score", 0.0) or
                0.0
            )
            
            content = doc.get("content", "").strip()
            if len(content) > max_length:
                content = self._smart_truncate_document(content, max_length, query)
            
            section += f"**[문서 {doc_idx}]** {doc_title} (관련도: {relevance_score:.2f})\n{content}\n\n"
        
        return section
    
    def _summarize_precedent(
        self,
        doc: Dict[str, Any],
        query: str,
        llm_fast: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """판례 문서 요약 (LLM 사용)
        
        Args:
            doc: 판례 문서 딕셔너리
            query: 사용자 질문
            llm_fast: LLM 인스턴스 (없으면 self.llm_fast 사용)
        
        Returns:
            요약 결과 딕셔너리 (summary, key_points, relevance_notes) 또는 None
        """
        try:
            doc_content = doc.get("content", "") or doc.get("text", "")
            if not doc_content or len(doc_content.strip()) < 100:
                logger.warning(f"⚠️ [PRECEDENT SUMMARY] 문서 내용이 너무 짧습니다: {len(doc_content)}자")
                return None
            
            # LLM 인스턴스 확인
            llm = llm_fast or self.llm_fast
            if not llm:
                logger.warning("⚠️ [PRECEDENT SUMMARY] LLM 인스턴스가 없습니다. 요약을 수행할 수 없습니다.")
                return None
            
            # DocumentSummaryTask 사용
            try:
                from lawfirm_langgraph.core.workflow.tasks.document_summary_tasks import (
                    DocumentSummaryTask,
                    SummaryStrategy
                )
                
                import os
                PRECEDENT_SUMMARY_MAX_LENGTH = int(os.getenv("PRECEDENT_SUMMARY_MAX_LENGTH", "1000"))
                
                # 판례 특화 요약 Task 생성
                summary_task = DocumentSummaryTask(
                    llm_fast=llm,
                    logger_instance=logger,
                    strategy=SummaryStrategy.BATCH,  # 배치 전략 (단일 문서도 배치로 처리)
                    max_summary_length=PRECEDENT_SUMMARY_MAX_LENGTH,
                    max_prompt_length=8000
                )
                
                # 판례 특화 프롬프트를 위한 문서 준비
                precedent_doc = {
                    "id": doc.get("id") or doc.get("chunk_id") or doc.get("document_id"),
                    "text": doc_content,
                    "content": doc_content,
                    "type": "precedent",
                    "case_id": doc.get("case_id"),
                    "court": doc.get("court") or doc.get("ccourt"),
                    "case_name": doc.get("case_name") or doc.get("casenames"),
                    "decision_date": doc.get("decision_date")
                }
                
                # Task 실행 (단일 문서 배치)
                summaries, metadata = summary_task.execute(
                    docs=[precedent_doc],
                    query=query,
                    use_llm=True
                )
                
                if summaries and len(summaries) > 0:
                    summary_result = summaries[0]
                    if isinstance(summary_result, dict):
                        logger.info(
                            f"✅ [PRECEDENT SUMMARY] 판례 요약 성공: "
                            f"원본={len(doc_content)}자 → 요약={len(summary_result.get('summary', ''))}자"
                        )
                        return summary_result
                    else:
                        logger.warning(f"⚠️ [PRECEDENT SUMMARY] 요약 결과 형식이 올바르지 않습니다: {type(summary_result)}")
                        return None
                else:
                    logger.warning("⚠️ [PRECEDENT SUMMARY] 요약 결과가 비어있습니다.")
                    return None
                    
            except ImportError:
                logger.warning("⚠️ [PRECEDENT SUMMARY] DocumentSummaryTask를 사용할 수 없습니다.")
                return None
            except Exception as e:
                logger.error(f"❌ [PRECEDENT SUMMARY] 요약 중 오류 발생: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ [PRECEDENT SUMMARY] 판례 요약 실패: {e}")
            return None
    
    def _extract_detailed_relevant_parts(
        self,
        doc: Dict[str, Any],
        query: str,
        max_extract_length: int = 500
    ) -> str:
        """
        질문과 직접 관련된 부분만 상세 추출
        
        전략:
        1. 질문 키워드 포함 문장 우선
        2. 관련 문맥 포함 (전후 2-3문장)
        3. 최대 길이 제한
        """
        content = doc.get("content", "").strip()
        if not content or not query:
            return content[:max_extract_length] if content else ""
        
        # 질문 키워드 추출
        query_keywords = set(query.split())
        
        # 문장 분리
        sentences = re.split(r'[。\.\n]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 키워드 매칭 점수 계산
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            for keyword in query_keywords:
                if keyword.lower() in sentence_lower:
                    score += 1
            
            # 법률 용어 가중치
            legal_terms = ['법', '조', '항', '호', '판결', '판례', '손해배상', '계약', '소송']
            for term in legal_terms:
                if term in sentence:
                    score += 0.5
            
            if score > 0:
                scored_sentences.append((score, i, sentence))
        
        # 점수 순 정렬
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 문장과 주변 문맥 추출
        selected_indices = set()
        result_parts = []
        current_length = 0
        
        for score, idx, sentence in scored_sentences[:5]:  # 상위 5개 문장
            if current_length + len(sentence) > max_extract_length:
                break
            
            # 주변 문맥 포함 (전후 2문장)
            context_start = max(0, idx - 2)
            context_end = min(len(sentences), idx + 3)
            
            for i in range(context_start, context_end):
                if i not in selected_indices:
                    selected_indices.add(i)
                    sent = sentences[i]
                    if current_length + len(sent) <= max_extract_length:
                        result_parts.append((i, sent))
                        current_length += len(sent)
                    else:
                        break
        
        # 인덱스 순 정렬
        result_parts.sort(key=lambda x: x[0])
        
        # 결과 조합
        if result_parts:
            result = ' '.join([sent for _, sent in result_parts])
            return result[:max_extract_length]
        else:
            # 매칭 문장이 없으면 앞부분 반환
            return content[:max_extract_length]