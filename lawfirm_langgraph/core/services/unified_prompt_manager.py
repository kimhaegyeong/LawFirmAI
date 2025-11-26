# -*- coding: utf-8 -*-
"""
통합 프롬프트 관리 시스템
법률 도메인 특화 프롬프트의 통합 관리 및 최적화
"""

import json
import re
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# QuestionType import
try:
    from core.classification.classifiers.question_classifier import QuestionType
except ImportError:
    try:
        # 호환성을 위한 fallback
        from core.services.question_classifier import QuestionType
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from question_classifier import QuestionType
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
    MIN_CONTENT_LENGTH = 10  # 최소 content 길이
    MIN_CONTENT_LENGTH_WITH_LAW_INFO = 3  # 법률 정보가 있을 때 최소 content 길이
    MAX_DOC_LENGTH_LAW = 1500  # 법률 조문 최대 길이
    MAX_DOC_LENGTH_CASE = 800  # 판례 최대 길이
    MAX_DOC_LENGTH_COMMENTARY = 500  # 해설 최대 길이
    
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
            final_prompt = self._build_final_prompt(base_prompt, query, context, question_type)

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
                        logger.debug(
                            f"📋 [PROMPT VALIDATION] Details: "
                            f"{sum(1 for d in validation_details if d.get('found_in_prompt'))}/{len(validation_details)} "
                            f"documents found in prompt"
                        )
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
                        # structured_documents의 문서들을 context에 추가
                        doc_contents = []
                        added_count = 0

                        for doc in documents[:8]:
                            if isinstance(doc, dict):
                                normalized_doc = self._normalize_document_fields(doc)
                                doc_content = normalized_doc.get("content", "")
                                doc_source = normalized_doc.get("source", "Unknown")
                                doc_score = normalized_doc.get("relevance_score", 0.0)

                                if doc_content and len(doc_content.strip()) > 10:
                                    # prompt_optimized_text에 이미 포함되어 있는지 확인
                                    content_preview = doc_content[:100]
                                    # 간단한 중복 체크: content의 일부가 prompt_text에 포함되어 있으면 스킵
                                    if content_preview not in prompt_text:
                                        # 포함되지 않은 문서 내용 추가
                                        doc_contents.append(
                                            f"\n[문서 출처: {doc_source}] [관련도: {doc_score:.3f}]\n{doc_content[:1000]}"
                                        )
                                        added_count += 1
                                    else:
                                        # 이미 포함되어 있지만 structured_documents에 명시적으로 포함시키기 위해 기록
                                        logger.debug(f"Document from {doc_source} already in prompt_optimized_text")

                        if doc_contents:
                            docs_text = "\n\n## 추가 참고 문서\n" + "\n".join(doc_contents)
                            optimized_context["context"] = optimized_context["context"] + docs_text
                            logger.info(
                                f"✅ [CONTEXT OPTIMIZATION] Added {added_count} additional documents "
                                f"from structured_documents to ensure all search results are included "
                                f"({len(doc_contents)} chars added)"
                            )

                        # structured_documents를 명시적으로 포함 (항상)
                        # 이는 LLM이 문서 구조를 명확히 이해할 수 있도록 함
                        normalized_docs = []
                        for doc in documents[:8]:
                            normalized = self._normalize_document_fields(doc)
                            if normalized.get("content"):
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
                    if normalized.get("content"):
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

### Text-to-SQL 스키마 요약
{sql_schema}

### SQL 작성 지침
- SELECT만 사용하세요. DML/DDL은 금지됩니다.
- WHERE 절에 정확한 필터를 명시하고, 반드시 LIMIT를 포함하세요.
- 결과는 조문/사건번호/선고일/법원을 기준으로 재현 가능해야 합니다.

### 법령 조문 검색 규칙 (중요)
- **법령 조문은 정확한 법령명과 조문번호만 조회하세요.**
- 법령명은 정확히 일치하거나 LIKE '%법령명%'로 매칭하되, 조문번호는 반드시 정확히 일치해야 합니다.
- 예: "민법 제750조" → WHERE law_name LIKE '%민법%' AND article_number = 750

### 판례/결정례/해석례 검색 규칙 (중요)
- **판례, 결정례, 해석례에서는 법령명과 조문번호가 모두 일치하는 항목만 조회하세요.**
- 질의에 법령명과 조문번호가 포함된 경우, 반드시 WHERE 절에 법령명과 조문번호 필터를 추가하세요.
- 예: "민법 제750조 관련 판례" → WHERE content LIKE '%민법%' AND content LIKE '%제750조%' 또는 WHERE law_name LIKE '%민법%' AND article_number = 750

### 예시(한국어 → SQL)
- 질의: "민법 제750조 조문 보여줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%민법%' AND article_number = 750 LIMIT 5;
- 질의: "형법 제307조 찾아줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%형법%' AND article_number = 307 LIMIT 5;
- 질의: "상법 제24조 전문 보여줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%상법%' AND article_number = 24 LIMIT 5;
- 질의: "근로기준법 제60조 연차 규정"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%근로기준법%' AND article_number = 60 LIMIT 5;
- 질의: "민법 제750조 관련 판례"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE summary LIKE '%민법%' AND summary LIKE '%제750조%' LIMIT 20;
- 질의: "형법 제307조 관련 결정례"
  SQL: SELECT doc_id, org, decision_date, text FROM decision_paragraphs WHERE text LIKE '%형법%' AND text LIKE '%제307조%' LIMIT 20;
- 질의: "상법 제24조 관련 해석례"
  SQL: SELECT doc_id, org, title, text FROM interpretation_paragraphs WHERE text LIKE '%상법%' AND text LIKE '%제24조%' LIMIT 20;
- 질의: "대법원 2021다12345 사건 요지"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE case_number = '2021다12345' LIMIT 5;
- 질의: "서울고등법원 2022년 이후 판결 요지 10건"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE court LIKE '%고등법원%' AND decision_date >= '2022-01-01' LIMIT 10;
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
        """base_prompt 간소화 (불필요한 섹션 제거)"""
        import re
        
        # 불필요한 섹션 제거
        sections_to_remove = [
            r'## 법률 자문 지침.*?(?=##|$)',
            r'## 컨텍스트 처리.*?(?=##|$)',
            r'## 답변 품질 검증 체크리스트.*?(?=##|$)',
            r'## 모델 최적화 설정.*?(?=##|$)',
            r'## 검색 결과 통계.*?(?=##|$)',
        ]
        
        for pattern in sections_to_remove:
            base_prompt = re.sub(pattern, '', base_prompt, flags=re.DOTALL)
        
        # 중복된 설명 제거
        base_prompt = re.sub(
            r'한국 법률 특성.*?실무적 관점.*?(?=##|$)', 
            '## 한국 법률 특성\n- 성문법 중심, 대법원 판례 중시, 실무적 관점\n', 
            base_prompt, 
            flags=re.DOTALL
        )
        
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

            # 문서 정규화 및 중복 제거
            normalized_count = 0
            skipped_count = 0
            for idx, doc in enumerate(raw_documents):
                logger.debug(f"🔍 [DEBUG] Processing doc {idx+1}/{doc_count}: type={type(doc)}, keys={list(doc.keys()) if isinstance(doc, dict) else 'N/A'}")
                normalized = self._normalize_document_fields(doc)
                if normalized:
                    # 문서 ID 생성 (중복 체크용)
                    doc_id = self._generate_document_id(normalized)
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        normalized_docs.append(normalized)
                        normalized_count += 1
                        logger.debug(f"✅ [DEBUG] Doc {idx+1} normalized successfully, has metadata: {bool(normalized.get('metadata'))}")
                    else:
                        logger.debug(f"⚠️ [FINAL PROMPT] Duplicate document removed: {doc_id}")
                else:
                    skipped_count += 1
                    logger.warning(f"⚠️ [DEBUG] Doc {idx+1} normalization returned None (skipped)")

            logger.info(f"📋 [FINAL PROMPT] Normalization: {normalized_count} succeeded, {skipped_count} skipped")

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
                    first_doc_content = normalized_docs[0].get("content", "")[:100]
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
        # 멀티 질의 검색 결과가 있거나 normalized_docs가 있고 base_prompt에 실제 문서가 없을 때 생성
        logger.debug(f"🔍 [DOCUMENTS SECTION] normalized_docs={len(normalized_docs) if normalized_docs else 0}, "
                    f"has_docs_in_base={has_docs_in_base}, has_multi_query_results={has_multi_query_results}")
        
        if normalized_docs and (not has_docs_in_base or has_multi_query_results):
            logger.info(f"✅ [DOCUMENTS SECTION] Creating documents section: "
                       f"normalized_docs={len(normalized_docs)}, has_docs_in_base={has_docs_in_base}, "
                       f"has_multi_query_results={has_multi_query_results}")
            # 관련도 기준 정렬 (개선: 법률 조문 우선 포함, 최소 5개 이상 포함 보장)
            # 법률 조문과 일반 문서를 분리하여 법률 조문을 우선 포함
            law_docs = [doc for doc in normalized_docs if doc.get("law_name") and doc.get("article_no")]
            other_docs = [doc for doc in normalized_docs if not (doc.get("law_name") and doc.get("article_no"))]
            
            # 법률 조문을 관련도 순으로 정렬
            sorted_law_docs = sorted(
                law_docs,
                key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )
            
            # 일반 문서를 관련도 순으로 정렬
            sorted_other_docs = sorted(
                other_docs,
                key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )
            
            # 토큰 제한 내에서 문서 선택 및 축약
            selected_docs = []
            current_doc_tokens = 0
            
            # 법률 조문 우선 선택
            for doc in sorted_law_docs[:self.MAX_LAW_DOCS]:
                doc_content = doc.get("content", "")
                doc_tokens = self._estimate_tokens(doc_content)
                
                # 문서당 최대 토큰 수 제한
                max_tokens_per_doc = min(2000, available_doc_tokens // 5)  # 문서당 최대 2000토큰 또는 평균값
                
                if doc_tokens > max_tokens_per_doc:
                    # 문서 축약
                    max_chars = int(max_tokens_per_doc * 2.5)
                    doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                    doc_tokens = self._estimate_tokens(doc_content)
                    doc["content"] = doc_content
                    doc["truncated"] = True
                
                if current_doc_tokens + doc_tokens <= available_doc_tokens:
                    selected_docs.append(doc)
                    current_doc_tokens += doc_tokens
                else:
                    # 남은 토큰이 있으면 축약하여 추가
                    remaining_tokens = available_doc_tokens - current_doc_tokens
                    if remaining_tokens > 500:  # 최소 500토큰 이상 남아있을 때만
                        max_chars = int(remaining_tokens * 2.5)
                        doc_content = self._smart_truncate_document(doc.get("content", ""), max_chars, query)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                        selected_docs.append(doc)
                    break
            
            # 일반 문서 선택 (멀티 질의 결과는 더 많이 포함)
            max_other_docs = self.MAX_DOCUMENTS * 2 if has_multi_query_results else self.MAX_DOCUMENTS
            logger.debug(f"🔍 [DOCUMENTS SECTION] max_other_docs={max_other_docs} (multi_query={has_multi_query_results})")
            
            for doc in sorted_other_docs:
                if len(selected_docs) >= max_other_docs:
                    break
                
                doc_content = doc.get("content", "")
                doc_tokens = self._estimate_tokens(doc_content)
                
                max_tokens_per_doc = min(1000, available_doc_tokens // 8)  # 문서당 최대 1000토큰
                
                if doc_tokens > max_tokens_per_doc:
                    max_chars = int(max_tokens_per_doc * 2.5)
                    doc_content = self._smart_truncate_document(doc_content, max_chars, query)
                    doc_tokens = self._estimate_tokens(doc_content)
                    doc["content"] = doc_content
                    doc["truncated"] = True
                
                if current_doc_tokens + doc_tokens <= available_doc_tokens:
                    selected_docs.append(doc)
                    current_doc_tokens += doc_tokens
                else:
                    remaining_tokens = available_doc_tokens - current_doc_tokens
                    if remaining_tokens > 500:
                        max_chars = int(remaining_tokens * 2.5)
                        doc_content = self._smart_truncate_document(doc.get("content", ""), max_chars, query)
                        doc["content"] = doc_content
                        doc["truncated"] = True
                        selected_docs.append(doc)
                    break
            
            sorted_docs = selected_docs

            if sorted_docs:
                documents_section = self._build_documents_section(sorted_docs, query)

                logger.info(
                    f"✅ [FINAL PROMPT] Added {len(sorted_docs)} documents "
                    f"(law_docs: {len([d for d in sorted_docs if d.get('law_name')])}, "
                    f"other_docs: {len([d for d in sorted_docs if not d.get('law_name')])}, "
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
            
            # 우선순위 2: prompt_optimized_text 사용
            if not documents_section:
                prompt_optimized_text = context.get("prompt_optimized_text", "")
                if prompt_optimized_text and len(prompt_optimized_text.strip()) > 100:
                    documents_section = "\n\n## 검색된 법률 문서\n\n"
                    documents_section += prompt_optimized_text[:5000] + ("..." if len(prompt_optimized_text) > 5000 else "")
                    documents_section += "\n\n"
                    logger.info("✅ [FINAL PROMPT] Added prompt_optimized_text as fallback")
            
            # 우선순위 3: context_text 사용
            if not documents_section:
                context_text = context.get("context", "")
                if context_text and len(context_text.strip()) > 100 and document_count > 0:
                    documents_section = "\n\n## 검색된 법률 문서\n\n"
                    documents_section += context_text[:5000] + ("..." if len(context_text) > 5000 else "")
                    documents_section += "\n\n"
                    logger.info("✅ [FINAL PROMPT] Added context_text as fallback")

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

        # 최종 프롬프트 구성
        final_prompt = simplified_base + documents_section + f"\n\n## 질문\n{query}\n\n"
        
        # 최종 토큰 수 검증
        final_tokens = self._estimate_tokens(final_prompt)
        if final_tokens > max_safe_tokens:
            logger.warning(
                f"⚠️ [TOKEN LIMIT] Final prompt exceeds safe limit: "
                f"{final_tokens:,} tokens (max: {max_safe_tokens:,}). "
                f"Applying emergency truncation..."
            )
            # 긴급 축약: 문서 섹션만 축약
            final_prompt = self._emergency_truncate_prompt(final_prompt, max_safe_tokens, simplified_base, query)
            final_tokens = self._estimate_tokens(final_prompt)
        
        logger.info(
            f"✅ [TOKEN COUNT] Final prompt: {final_tokens:,} tokens "
            f"({final_tokens/self.MAX_INPUT_TOKENS*100:.1f}% of max, "
            f"base: {base_tokens:,}, query: {query_tokens:,}, "
            f"docs: {current_doc_tokens if 'current_doc_tokens' in locals() else 0:,})"
        )

        # 단계별 답변 지침 구성
        if documents_section and normalized_docs:
            sorted_docs = sorted(
                normalized_docs,
                key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )[:5]

            # 예시 기반 학습 방식 적용 (간소화)
            instruction_section = """
## 📚 답변 형식 가이드

답변 시 다음 형식을 참고하세요:

**출처 인용**: "민법 제543조에 따르면..." [문서 1]

**문서별 근거 비교 표** (최소 3개 문서 포함):
| 문서 번호 | 출처 | 핵심 근거 |
|-----------|------|----------|
| [문서 1] | 민법 제543조 | 해지권 발생 원인 |
| [문서 2] | 대법원 판결 | 이행지체 해제 |

**답변 구조**: 직접 답변 → 문서별 근거 비교 → 결론 → 실무 조언

**스타일**: 친근한 존댓말, 자연스러운 문단 흐름
"""
        else:
            instruction_section = """
## 답변 작성 단계

**STEP 1**: 질문에 대한 일반적인 법적 원칙 설명
**STEP 2**: 관련 법령명과 조문번호 언급 (구체적 내용은 확인 필요 명시)
**STEP 3**: 실무적 조언

**스타일**: 친근한 존댓말
"""

        # Citation 요구사항을 프롬프트 상단에 배치 (법률 RAG 핵심 원칙 통합 - 간소화)
        citation_requirement = """
⚠️ **필수 요구사항: 법률 RAG 답변 원칙**

답변 생성 시 반드시 다음 핵심 원칙을 준수하세요:

**원칙 1: 문서 외 내용 추론/생성 금지**
- 검색된 문서에 없는 내용은 절대 추론하거나 생성하지 마세요
- 문서에 없으면 "문서에는 해당 내용이 명시되어 있지 않습니다"라고 표현하세요

**원칙 2: 문서 근거 필수 포함**
- 모든 답변은 반드시 문서 근거를 포함해야 합니다
- 주요 문단마다 문서 인용을 포함하세요
- 최소 2개 이상의 문서를 인용하세요
- 인용 형식: "[문서 N]에 따르면..." 또는 "민법 제XXX조에 따르면..." [문서 N]

**원칙 3: 문서 기반 해석만 허용**
- 문서 내용을 바탕으로 한 논리적 추론만 허용
- 문서 외 일반 지식 사용 금지

**원칙 4: 문서 유형 구분**
- 법률 전문: "민법 제XXX조에 따르면..." [문서 N]
- 판례 요약: "대법원 판결에 의하면..." [문서 N]
- 해설: "[문서 N]의 해설에 따르면..." [문서 N]

**원칙 5: 불확실성 명확히 표현**
- 문서에 명시적 내용이 없으면 "문서에는 명시적 내용이 없습니다"라고 표현하세요

**인용 형식 예시**:
- ✅ "민법 제15조에 따르면..." [문서 1]
- ✅ "[문서 1]에 따르면 민법 제15조는..." [문서 1]

**검증 체크리스트**:
- [ ] 모든 답변 내용이 문서에 근거를 두고 있는가?
- [ ] 문서별 근거 비교 표를 포함했는가? (최소 3개 문서)
- [ ] 최소 2개 이상의 문서를 명시적으로 인용했는가?
- [ ] 결론에서 표의 내용을 종합하여 제시했는가?
- [ ] 문서에 없는 내용을 추론하지 않았는가?
- [ ] 불확실한 부분을 명확히 표현했는가?

---
"""
        
        # 최종 프롬프트 구성 (지침 포함)
        final_prompt_with_instructions = f"""{citation_requirement}{simplified_base}{documents_section}

---

## 사용자 질문
{query}

{instruction_section}

답변을 시작하세요:
"""
        
        # 최종 토큰 수 재검증 (지침 추가 후)
        final_tokens_after_instructions = self._estimate_tokens(final_prompt_with_instructions)
        if final_tokens_after_instructions > max_safe_tokens:
            logger.warning(
                f"⚠️ [TOKEN LIMIT] Final prompt with instructions exceeds limit: "
                f"{final_tokens_after_instructions:,} tokens (max: {max_safe_tokens:,}). "
                f"Applying emergency truncation..."
            )
            # 긴급 축약: 문서 섹션만 축약
            final_prompt_with_instructions = self._emergency_truncate_prompt(
                final_prompt_with_instructions, 
                max_safe_tokens, 
                simplified_base, 
                query
            )
            final_tokens_after_instructions = self._estimate_tokens(final_prompt_with_instructions)
            logger.warning(
                f"⚠️ [EMERGENCY TRUNCATION] Prompt truncated to {final_tokens_after_instructions:,} tokens"
            )
        
        logger.info(
            f"✅ [FINAL TOKEN COUNT] Final prompt with instructions: {final_tokens_after_instructions:,} tokens "
            f"({final_tokens_after_instructions/self.MAX_INPUT_TOKENS*100:.1f}% of max)"
        )
        
        return final_prompt_with_instructions

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

                        if doc_content and len(doc_content) > 50:
                            # 문서 내용 일부가 프롬프트에 포함되었는지 확인 (여러 위치 확인)
                            doc_preview = doc_content[:150].strip()
                            doc_mid = doc_content[len(doc_content)//2:len(doc_content)//2+100].strip() if len(doc_content) > 200 else ""

                            found_in_prompt = False
                            if doc_preview and doc_preview in final_prompt:
                                found_in_prompt = True
                                doc_found_count += 1
                            elif doc_mid and doc_mid in final_prompt:
                                found_in_prompt = True
                                doc_found_count += 1

                            validation_result["validation_details"].append({
                                "doc_index": idx,
                                "source": normalized.get("source", "Unknown"),
                                "content_length": len(doc_content),
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

    def _normalize_document_fields(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        
        # 불필요한 메타데이터 제거
        content = self._clean_content(raw_content).strip()
        
        # 법률 정보가 있으면 content가 짧아도 포함 (개선: 10자 → 3자로 완화)
        has_law_info = bool(
            doc.get("law_name") or 
            doc.get("article_no") or 
            doc.get("case_name") or 
            doc.get("case_number") or
            (isinstance(doc.get("metadata"), dict) and (
                doc.get("metadata", {}).get("law_name") or
                doc.get("metadata", {}).get("article_no") or
                doc.get("metadata", {}).get("case_name")
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
                    f"has_multi_query_meta={has_multi_query_meta}")
        
        # 멀티 질의 결과는 content가 짧아도 포함 (최소 길이 완화)
        if has_multi_query_meta and content and len(content) < min_content_length:
            logger.debug(f"✅ [DOC NORMALIZE] Multi-query result with short content ({len(content)} chars), "
                        f"will create minimal doc")
            # 최소 content 생성
            if not content or len(content.strip()) < 3:
                content = doc.get("source", "") or "법률 문서"
        
        if not content or len(content) < min_content_length:
            if has_law_info:
                # 법률 정보가 있으면 content가 없어도 법률 정보만으로 문서 생성
                logger.debug(f"⚠️ [DOC NORMALIZE] Content too short ({len(content)} chars) but has law info, creating minimal doc")
            elif has_multi_query_meta:
                # 멀티 질의 결과는 최소 content 생성
                logger.debug(f"✅ [DOC NORMALIZE] Multi-query result, creating minimal doc even with short content")
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
        
        # 문서 타입 판단
        source_type = self._extract_field(doc, metadata, ["source_type", "type"])

        # 관련성 점수 추출
        relevance_score = float(
            doc.get("relevance_score", 0.0) or 
            doc.get("final_weighted_score", 0.0) or
            doc.get("score", 0.0) or 
            doc.get("similarity_score", 0.0) or
            doc.get("similarity", 0.0) or 
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
            doc_title = source or doc.get("source_type", "법률 문서")
        
        # content가 없거나 짧아도 법률 정보가 있으면 최소 content 생성 (개선)
        if not content or len(content) < min_content_length:
            if has_law_info:
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
            "announce_date": str(announce_date).strip(),
            "case_type": str(case_type).strip(),
            # 판례 본문 정보
            "case_summary": str(case_summary).strip(),
            "case_holding": str(case_holding).strip(),
            "case_reasoning": str(case_reasoning).strip(),
            "source_type": str(source_type).strip(),
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

    def _build_fallback_documents_section(self, documents: List[Dict[str, Any]]) -> str:
        """폴백 문서 섹션 생성 - 문서 섹션이 생성되지 않았을 때 사용"""
        if not documents or len(documents) == 0:
            return ""

        documents_section = "\n\n## 🔍 검색된 법률 문서\n\n"
        documents_section += "다음 문서들은 질문에 대한 답변을 위해 검색된 관련 법률 정보입니다.\n\n"

        import re

        for idx, doc in enumerate(documents[:10], 1):  # 최대 10개
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
                            "source_type": doc.get("source_type", "")
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

        # 개선: 문서 내용을 가장 먼저 추가하여 경고 방지
        # 질문 유형과 관계없이 document_contents가 있으면 항상 먼저 포함
        if document_contents:
            # 문서 내용을 질문 유형에 맞게 구조화하되, 항상 포함되도록 보장
            sorted_docs = sorted(
                document_contents,
                key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )
            
            high_relevance = [d for d in sorted_docs if d.get("score", 0.0) >= 0.65]
            medium_relevance = [d for d in sorted_docs if 0.35 <= d.get("score", 0.0) < 0.65]

        try:
            # 질문 유형별 구조화
            if question_type == QuestionType.PRECEDENT_SEARCH:
                # 판례 정보 우선 배치
                # 문서 내용 강제 포함 (가장 중요) - 이미 위에서 정렬됨
                if document_contents:

                    structured_parts.append("## 검색된 판례 문서\n")
                    structured_parts.append("다음은 질문에 대한 답변을 위해 검색된 관련 판례 문서입니다. **반드시 이 문서들의 내용을 참고하여 답변하세요.**\n")

                    if high_relevance:
                        structured_parts.append("### 🔴 최우선 문서 (관련도 0.65 이상)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(high_relevance[:5], is_high_priority=True)
                        )

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(medium_relevance[:3], is_high_priority=False)
                        )

                    structured_parts.append("")

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

                # 문서 내용 강제 포함 - 이미 위에서 정렬됨
                if document_contents:
                    structured_parts.append("## 검색된 법률 조문 문서\n")

                    if high_relevance:
                        structured_parts.append("### 🔴 최우선 문서 (관련도 0.65 이상)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(high_relevance[:5], is_high_priority=True)
                        )

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(medium_relevance[:3], is_high_priority=False)
                        )

                    structured_parts.append("")

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

                # 문서 내용 강제 포함 - 이미 위에서 정렬됨
                if document_contents:
                    structured_parts.append("## 검색된 법률 문서 및 판례\n")

                    if high_relevance:
                        structured_parts.append("### 🔴 최우선 문서 (관련도 0.65 이상)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(high_relevance[:5], is_high_priority=True)
                        )

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(medium_relevance[:3], is_high_priority=False)
                        )

                    structured_parts.append("")

                if context_text:
                    structured_parts.append("## 법률 분석 및 실무 조언\n")
                    structured_parts.append(context_text)

                if insights:
                    structured_parts.append("\n## 핵심 요약\n")
                    for insight in insights[:3]:
                        structured_parts.append(f"- {insight}")
            else:
                # 기본 구조: 문서 내용 우선 포함 - 이미 위에서 정렬됨
                if document_contents:
                    structured_parts.append("## 검색된 법률 문서\n")
                    structured_parts.append("다음은 질문에 대한 답변을 위해 검색된 관련 법률 문서입니다.\n")

                    if high_relevance:
                        structured_parts.append("### 🔴 최우선 문서 (관련도 0.65 이상)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(high_relevance[:5], is_high_priority=True)
                        )

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        structured_parts.extend(
                            self._format_documents_for_context(medium_relevance[:3], is_high_priority=False)
                        )

                    structured_parts.append("")

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
            
            if has_doc_content and (text_too_short or not has_doc_keywords):
                # 문서 내용이 추가되지 않은 경우 강제 추가
                logger.warning(
                    f"⚠️ [CONTEXT STRUCTURE] Document contents not properly included in structured context. "
                    f"Force adding {len(document_contents)} documents. (text_len={len(result_text)}, has_keywords={has_doc_keywords})"
                )
                doc_section = "\n## 검색된 법률 문서\n"
                doc_section += "다음은 질문에 대한 답변을 위해 검색된 관련 법률 문서입니다.\n\n"
                
                # 상위 5개 문서 추가 (관련도 순)
                sorted_docs = sorted(
                    document_contents,
                    key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )
                
                for idx, doc in enumerate(sorted_docs[:5], 1):
                    content = doc.get("content", "")[:2000] if len(doc.get("content", "")) > 2000 else doc.get("content", "")
                    if content and len(content.strip()) > 10:
                        doc_source = doc.get("source", "Unknown")
                        doc_score = doc.get("score", 0.0)
                        # 문서 형식화
                        doc_section += f"\n### 문서 {idx}: {doc_source} (관련도: {doc_score:.2f})\n{content}\n---\n"
                
                # 문서 섹션을 앞쪽에 추가
                result_text = doc_section + "\n" + result_text

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
        return """---
# Role: 대한민국 법률 전문가 AI 어시스턴트

당신은 대한민국 법률 전문 상담 AI입니다. 법학 석사 이상의 전문 지식을 보유하고 있으며, 다양한 법률 분야에 대한 실무 경험을 갖춘 것처럼 행동합니다.

## 한국 법률 특성

### 1. 성문법 중심
- 민법, 형법, 상법 등 성문법 우선 적용
- 법령의 정확한 조문 인용 필수
- 최신 법령 개정사항 반영

### 2. 대법원 판례 중시
- 대법원 판례의 구속력 인정
- 최신 판례 우선 참조
- 판례 번호와 핵심 판결요지 명시

### 3. 헌법재판소 결정
- 헌법재판소 결정의 중요성
- 위헌법률심판, 헌법소원 등

### 4. 실무적 관점
- 법원, 검찰, 법무부 실무 기준
- 변호사 실무 경험 반영
- 실제 사건 처리 경험 기반

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
- 법조문 인용: "민법 제543조에 따르면..." [문서 1]
- 판례 인용: "대법원 판결에 의하면..." [문서 2] 또는 "서울고등법원 2018. 5. 15. 선고 2017나2046429 판결에 따르면..." [문서 2]
- `[출처: 문서 N]`, `**출처**: 문서명` 등 다른 형식은 사용하지 마세요
- 예시:
  * ❌ "불법행위로 인한 손해배상청구권은 손해 및 가해자를 안 날부터 3년간 행사하지 아니하면 시효로 인해 소멸합니다 [출처: 문서 4]."
  * ✅ "불법행위로 인한 손해배상청구권은 손해 및 가해자를 안 날부터 3년간 행사하지 아니하면 시효로 인해 소멸합니다 [문서 4]."

### 3. 자연스러운 문단 흐름
- **과도한 구조화 지양**: 표나 복잡한 섹션 구분은 최소화하세요
- **자연스러운 문단 흐름**: 법적 내용을 논리적으로 연결된 문단으로 설명하세요
- **표 사용 최소화**: 문서별 근거 비교 표는 답변 말미에만 간략히 포함 (선택사항)
- **번호 매기기 제한**: 3개 이상의 항목이 연속될 때만 번호 사용, 그 외에는 자연스러운 문장으로 연결
- **섹션 제목 최소화**: "[질문 요약]", "[핵심 답변]" 등 명시적 섹션 제목 대신 자연스러운 문단 전환 사용
- 예시:
  * ❌ "1. 소멸시효 관리: ... 2. 과실상계 고려: ... 3. 손해액 산정: ..."
  * ✅ "불법행위로 인한 손해배상과 관련하여 실무에서는 소멸시효 관리가 중요합니다. 손해 및 가해자를 안 날로부터 3년 이내에 행사해야 하므로... 또한 피해자에게도 과실이 있다면 과실상계가 적용될 수 있으므로... 그리고 손해배상액을 청구할 때는..."

### 4. 접근성 있는 언어 및 자연스러운 어조
- 전문 법률 용어는 쉬운 말로 풀어서 설명
- 필요시 예시를 들어 이해를 돕기
- 복잡한 개념은 단계별로 설명
- **격식체 유지하되 친근함 추가**: 존댓말을 사용하되, 딱딱하지 않게 작성
- **직접적 표현**: "귀하의 질문은...으로 이해됩니다" 대신 "말씀하신 [내용]에 대해 설명드리겠습니다" 사용
- **자연스러운 전환**: "또한", "그런데", "다만", "참고로" 등 자연스러운 연결어 사용
- **불확실성 표현 자연스럽게**: "명시되어 있지 않습니다" 대신 "직접 나와 있지는 않지만" 같은 자연스러운 표현
- 예시:
  * ❌ "제공된 문서들에는 민법 제750조의 조문 내용이 직접적으로 명시되어 있지 않습니다. 그러나 문서들은..."
  * ✅ "제공된 문서에는 민법 제750조의 조문이 직접 나와 있지는 않지만, 불법행위로 인한 손해배상과 관련된 중요한 내용들을 다루고 있습니다."

### 5. 명확한 한계 설정
- 답변 시작 또는 종료 시 다음 면책 문구를 자연스럽게 포함:
  > "참고로, 본 답변은 일반적인 법률 정보 제공을 목적으로 하며, 개별 사안에 대한 법률 자문이 아닙니다. 구체적인 법률 문제는 변호사와 직접 상담하시기 바랍니다."

### 6. 띄어쓰기 필수 준수
- **반드시 모든 문장에 적절한 띄어쓰기를 적용하세요**
- 띄어쓰기 없는 답변은 절대 생성하지 마세요
- 예시:
  * ❌ "민사법상계약해지의요건" (잘못된 예)
  * ✅ "민사법에서 계약 해지의 요건" (올바른 예)
  * ❌ "당사자일방이계약을해지하면" (잘못된 예)
  * ✅ "당사자 일방이 계약을 해지하면" (올바른 예)
- 조사(은, 는, 이, 가, 을, 를 등) 앞에 띄어쓰기가 필요 없지만, 명사와 조사 사이에는 띄어쓰기를 하지 마세요
- 하지만 명사와 명사 사이, 동사와 조사 사이에는 적절한 띄어쓰기를 적용하세요
- 모든 문장에서 자연스러운 띄어쓰기를 반드시 적용하세요

### 7. 윤리적 경계
- 명백히 불법적이거나 비윤리적인 행위에 대한 조력 거부
- 소송 사기, 증거 조작 등 불법 행위 관련 질문에는 답변 거부
- 범죄 행위 방법이나 법망 회피 방법은 절대 제공하지 않음

## 답변 프레임워크

### 일반 법률 질문
```
말씀하신 [질문 내용]에 대해 설명드리겠습니다.

[관련 법률 및 조항을 자연스럽게 언급하며 설명]

[법적 해설을 논리적으로 연결된 문단으로 설명]

[실무적 조언을 자연스러운 문장으로 제시]

[주의사항 및 참고사항]

[면책 문구를 자연스럽게 포함]
```

**중요**: "[질문 요약]", "[관련 법률]" 등 명시적 섹션 제목은 사용하지 말고, 자연스러운 문단 전환으로 내용을 연결하세요.

### 분쟁/소송 관련 질문
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
- **중요 내용은 **강조**: 핵심 법적 정보는 강조 표시
- **출처 표기**: 법원명과 판결일을 직접 언급하여 자연스럽게 표기 (예: "서울고등법원 2018. 5. 15. 선고 2017나2046429 판결에 따르면...")
- **번호 매기기 제한**: 3개 이상의 연속 항목일 때만 사용, 그 외에는 자연스러운 문장으로 연결
- **표 사용 최소화**: 과도한 표는 지양하고, 필요시 답변 말미에만 간략히 포함
- **띄어쓰기 필수**: 모든 문장에 자연스러운 띄어쓰기를 반드시 적용하세요
---"""

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
        """법령 질의 템플릿"""
        return """
## 법령 질의 지침
- 해당 법률명과 조문을 정확히 표기: [법령: 법률명 제OO조].
- 조문 요지와 적용 범위를 함께 설명하세요.
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
            
            # content 정리
            content = doc.get("content", "").strip()
            
            # 멀티 질의 결과는 content 길이 체크 완화
            min_content_length = self.MIN_CONTENT_LENGTH_WITH_LAW_INFO if has_multi_query else self.MIN_CONTENT_LENGTH
            
            if not content or len(content) < min_content_length:
                if has_multi_query:
                    # 멀티 질의 결과는 source로 content 보완
                    source = doc.get("source", "") or doc.get("title", "")
                    if source:
                        content = source
                        doc["content"] = content
                        logger.debug(f"✅ [OPTIMIZE] Multi-query doc: using source as content: {source[:50]}")
                    else:
                        continue
                else:
                    continue
            
            # 중복 체크 (내용 기반)
            content_hash = hash(content[:200])  # 처음 200자로 중복 체크
            if content_hash in seen_contents:
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
    
    def _build_documents_section(self, sorted_docs: List[Dict[str, Any]], query: str) -> str:
        """문서 섹션 생성 (Summary-First 방식, 에이전트 사용)"""
        if not sorted_docs:
            return "\n\n## 검색된 법률 문서\n\n검색된 문서가 없습니다.\n"
        
        # 1. 문서 분류 (요약 필요 vs 전체 포함)
        docs_for_summary = []
        docs_for_full = []
        
        for doc in sorted_docs:
            if self._should_use_summary(doc):
                docs_for_summary.append(doc)
            else:
                docs_for_full.append(doc)
        
        # 2. 요약 생성 (에이전트 사용)
        summaries = []
        summary_agent = self._get_summary_agent()
        if summary_agent and docs_for_summary:
            try:
                logger.info(f"[UnifiedPromptManager] 요약 생성 시작: 문서 수={len(docs_for_summary)}, use_llm=True, llm_fast={self.llm_fast is not None}")
                summaries = summary_agent.summarize_batch(
                    docs_for_summary,
                    query,
                    max_summary_length=self.MAX_SUMMARY_LENGTH,
                    use_llm=True  # LLM 기반 요약 사용 (gemini-2.5-flash-lite)
                )
                logger.info(f"[UnifiedPromptManager] 요약 생성 완료: 요약 수={len(summaries)}")
            except Exception as e:
                logger.warning(f"요약 생성 실패: {e}, 전체 문서 사용")
                docs_for_full.extend(docs_for_summary)
                docs_for_summary = []
                summaries = []
        
        # 3. Summary 섹션 생성
        summary_section = self._build_summary_section(summaries, docs_for_summary, sorted_docs, query)
        
        # 4. Detailed Extracts 섹션 생성 (상위 문서만)
        detailed_section = self._build_detailed_section(
            docs_for_summary[:self.MAX_DETAILED_EXTRACTS],
            sorted_docs,
            query
        )
        
        # 5. 전체 문서 섹션 (요약 불필요한 문서)
        full_docs_section = self._build_full_docs_section(docs_for_full, sorted_docs, query)
        
        # 6. 통합
        documents_section = "\n\n## 검색된 법률 문서\n\n"
        documents_section += "위 문서들을 인용할 때는 `[문서 N]` 형식을 사용하세요.\n\n"
        
        if summary_section:
            documents_section += summary_section
        
        if detailed_section:
            documents_section += detailed_section
        
        if full_docs_section:
            documents_section += full_docs_section
        
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
    
    def _build_summary_section(
        self, 
        summaries: List[Dict[str, Any]], 
        original_docs: List[Dict[str, Any]],
        all_docs: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Summary 섹션 생성"""
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
            
            section += f"**[문서 {doc_idx}]** {doc_title} (관련도: {relevance_score:.2f})\n"
            
            # 요약 내용
            summary_text = summary.get('summary', '')
            if summary_text:
                section += f"- 요약: {summary_text}\n"
            
            # 핵심 포인트
            key_points = summary.get('key_points', [])
            if key_points:
                section += "- 핵심 쟁점:\n"
                for point in key_points[:3]:  # 최대 3개
                    if isinstance(point, str) and point.strip():
                        section += f"  • {point[:150]}\n"
            
            # 연관성
            relevance_notes = summary.get('relevance_notes', '')
            if relevance_notes:
                section += f"- 질문 연관성: {relevance_notes}\n"
            
            section += "\n"
        
        return section
    
    def _build_detailed_section(
        self, 
        docs: List[Dict[str, Any]], 
        all_docs: List[Dict[str, Any]],
        query: str,
        max_docs: int = 3
    ) -> str:
        """Detailed Extracts 섹션 생성"""
        if not docs:
            return ""
        
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
            
            section += f"**[문서 {doc_idx}]** {doc_title} (관련도: {relevance_score:.2f}) 상세 내용:\n"
            
            # 질문과 관련된 부분만 추출
            detailed_content = self._extract_detailed_relevant_parts(
                doc, query, self.MAX_DETAILED_EXTRACT_LENGTH
            )
            
            if detailed_content:
                section += f"{detailed_content}\n\n"
            else:
                # 폴백: 스마트 축약
                content = doc.get("content", "").strip()
                if content:
                    max_length = min(self.MAX_DETAILED_EXTRACT_LENGTH, len(content))
                    content = self._smart_truncate_document(content, max_length, query)
                    section += f"{content}\n\n"
        
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