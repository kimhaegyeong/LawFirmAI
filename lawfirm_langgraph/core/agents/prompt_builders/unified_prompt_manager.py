# -*- coding: utf-8 -*-
"""
통합 프롬프트 관리 시스템
법률 도메인 특화 프롬프트의 통합 관리 및 최적화
"""

import json
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

logger = logging.getLogger(__name__)


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
    OLLAMA = "ollama"
    OPENAI = "openai"


class UnifiedPromptManager:
    """통합 프롬프트 관리 시스템"""

    def __init__(self, prompts_dir: str = "data/prompts"):
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

        try:
            logger.debug("UnifiedPromptManager initialized (lazy loading enabled)")
        except Exception:
            # 로깅 오류를 무시하고 계속 진행
            pass

    def _ensure_prompts_loaded(self):
        """프롬프트가 로드되지 않았으면 로드 (지연 로딩)"""
        if not self._prompts_loaded:
            # 디렉토리 생성 (필요한 경우)
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            
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
            },
            # 개선 사항 9: Domain Template 추가 - "기타/일반" 도메인 템플릿 추가
            LegalDomain.GENERAL: {
                "focus": "일반 법률 질문, 법률 용어 설명, 기타 법률 상담",
                "key_laws": ["민법", "형법", "상법", "행정법", "기타 법령"],
                "recent_changes": "2024년 법령 개정사항 반영",
                "template": self._get_general_template()
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
            ModelType.OLLAMA: {
                "max_tokens": 4096,
                "temperature": 0.2,
                "system_prompt_style": "conversational",
                "context_window": 0.7
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
            # context 타입 검증 및 변환
            if context is None:
                context = {}
            elif not isinstance(context, dict):
                logger.warning(f"⚠️ [UNIFIED PROMPT MANAGER] context is not a dict (type: {type(context)}), converting to dict")
                if isinstance(context, str):
                    context = {"context": context}
                else:
                    context = {}
            
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
            # question_type 정규화 (enum이 아닐 수 있음)
            from core.workflow.utils.workflow_utils import WorkflowUtils
            normalized_question_type = WorkflowUtils.normalize_question_type(question_type, logger)
            
            question_type_templates = self.question_type_templates  # property 접근으로 지연 로딩 트리거
            question_template = question_type_templates.get(normalized_question_type)
            if question_template:
                base_prompt = self._add_question_structure(base_prompt, question_template)
                logger.info(
                    f"✅ [QUESTION TEMPLATE] Applied question type template: "
                    f"{normalized_question_type.name if hasattr(normalized_question_type, 'name') else normalized_question_type}, "
                    f"priority={question_template.get('priority', 'unknown')}"
                )
            else:
                # 기본 템플릿 사용 (GENERAL_QUESTION)
                default_template = question_type_templates.get(QuestionType.GENERAL_QUESTION)
                if default_template:
                    base_prompt = self._add_question_structure(base_prompt, default_template)
                    logger.debug(
                        f"ℹ️ [QUESTION TEMPLATE] Using default template (GENERAL_QUESTION) for: "
                        f"{normalized_question_type.name if hasattr(normalized_question_type, 'name') else normalized_question_type}"
                    )
                else:
                    logger.debug(
                        f"ℹ️ [QUESTION TEMPLATE] No template available for: "
                        f"{normalized_question_type.name if hasattr(normalized_question_type, 'name') else normalized_question_type}, "
                        f"continuing without template"
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

            # 7. 프롬프트 검증: 문서 내용이 포함되었는지 확인 (강화된 검증)
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
        # context 타입 검증 및 변환
        if context is None:
            return base_prompt
        if not isinstance(context, dict):
            logger.warning(f"⚠️ [OPTIMIZE CONTEXT] context is not a dict (type: {type(context)}), converting to dict")
            if isinstance(context, str):
                context = {"context": context}
            else:
                return base_prompt

        if not context:
            return base_prompt

        if question_template is None:
            question_template = {}

        max_length = question_template.get('max_context_length', 5000)  # 기본값 증가: 2000 -> 5000
        context_keys = question_template.get('context_keys', [])

        optimized_context = {}

        # 1. context_keys에 지정된 키 포함
        for key in context_keys:
            if key in context and isinstance(context, dict):
                content = context.get(key)
                if isinstance(content, str) and len(content) > max_length:
                    content = content[:max_length] + "..."
                optimized_context[key] = content

        # 2. prompt_optimized_text가 있으면 context로 변환 (최우선)
        if isinstance(context, dict) and "prompt_optimized_text" in context:
            prompt_text = context.get("prompt_optimized_text", "")
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
        if isinstance(context, dict) and "context" in context and "context" not in optimized_context:
            content = context.get("context", "")
            if isinstance(content, str) and len(content) > max_length:
                content = content[:max_length] + "..."
            optimized_context["context"] = content

        # 4. structured_documents에서 실제 문서 내용 추출하여 context에 추가
        structured_docs = context.get("structured_documents", {}) if isinstance(context, dict) else {}
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
                        f"⚠️ [CONTEXT OPTIMIZATION] No valid documents after normalization, "
                        f"keeping original structured_documents"
                    )

        # structured_documents가 context에는 있지만 optimized_context에 없는 경우 직접 추가
        elif isinstance(context, dict) and "structured_documents" in context:
            structured_docs = context.get("structured_documents", {})
            if isinstance(structured_docs, dict) and structured_docs.get("documents"):
                optimized_context["structured_documents"] = structured_docs
                logger.info(f"✅ [CONTEXT OPTIMIZATION] Added structured_documents from original context to optimized_context")

        # 5. context_keys가 비어있으면 모든 관련 컨텍스트 키 포함
        if not context_keys and context and isinstance(context, dict):
            for key, value in context.items():
                if key not in ["query_type", "context_length", "docs_truncated", "prompt_optimized_text"]:  # 메타데이터 및 이미 처리한 키 제외
                    if isinstance(value, str) and len(value) > max_length:
                        value = value[:max_length] + "..."
                    if key not in optimized_context:  # 이미 포함된 키는 중복 추가하지 않음
                        optimized_context[key] = value

        # 질문 유형별 컨텍스트 구조화
        question_type = context.get("query_type") if isinstance(context, dict) else None
        if isinstance(question_type, str):
            # 문자열을 QuestionType enum으로 변환
            try:
                question_type_enum = QuestionType[question_type.upper()] if question_type.upper() in [e.name for e in QuestionType] else QuestionType.GENERAL_QUESTION
            except:
                question_type_enum = QuestionType.GENERAL_QUESTION
        elif hasattr(question_type, 'name'):
            question_type_enum = question_type
        else:
            question_type_enum = QuestionType.GENERAL_QUESTION

        structured_context = self._structure_context_by_question_type(optimized_context, question_type_enum)

        # 🔴 개선: 불필요한 지침 섹션 제거 (base_prompt에 이미 포함되거나 _build_final_prompt에서 추가됨)
        context_guidance = f"{structured_context}"
        try:
            sql_schema = context.get("sql_schema") if isinstance(context, dict) else None
            if isinstance(sql_schema, str) and len(sql_schema) > 0:
                context_guidance += f"""

### Text-to-SQL 스키마 요약
{sql_schema}

### SQL 작성 지침
- SELECT만 사용하세요. DML/DDL은 금지됩니다.
- WHERE 절에 정확한 필터를 명시하고, 반드시 LIMIT를 포함하세요.
- 결과는 조문/사건번호/선고일/법원을 기준으로 재현 가능해야 합니다.

### 예시(한국어 → SQL)
- 질의: "민법 제750조 조문 보여줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%민법%' AND article_number = 750 LIMIT 5;
- 질의: "대법원 2021다12345 사건 요지"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE case_number = '2021다12345' LIMIT 5;
- 질의: "최근 3년 민사 손해배상 판결 건수"
  SQL: SELECT COUNT(*) AS cnt FROM cases WHERE decision_date >= date('now','-3 years');
- 질의: "형법 제307조 찾아줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%형법%' AND article_number = 307 LIMIT 5;
- 질의: "민법 개정 이력 중 2020년 이후만"
  SQL: SELECT law_name, effective_date, description FROM amendments WHERE effective_date >= '2020-01-01' LIMIT 20;
- 질의: "사건 2019다12345가 인용한 판례 목록"
  SQL: SELECT from_case_id, to_case_id FROM case_citations WHERE from_case_id = '2019다12345' LIMIT 20;
- 질의: "상법 제24조 전문 보여줘"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%상법%' AND article_number = 24 LIMIT 5;
- 질의: "서울고등법원 2022년 이후 판결 요지 10건"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE court LIKE '%고등법원%' AND decision_date >= '2022-01-01' LIMIT 10;
- 질의: "저작권법 최근 개정 내역"
  SQL: SELECT law_name, effective_date, description FROM amendments WHERE law_name LIKE '%저작권법%' ORDER BY effective_date DESC LIMIT 20;
- 질의: "사건요지에 '손해배상' 포함된 판례 20건"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE summary LIKE '%손해배상%' LIMIT 20;
- 질의: "민법 관련 조문 중 '불법행위' 포함 본문"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%민법%' AND content LIKE '%불법행위%' LIMIT 20;
- 질의: "근로기준법 제60조 연차 규정"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%근로기준법%' AND article_number = 60 LIMIT 5;
- 질의: "고용노동부 관련 판례 2021년 이후"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE summary LIKE '%고용노동부%' AND decision_date >= '2021-01-01' LIMIT 20;
- 질의: "부가가치세법 세금계산서 관련 조문"
  SQL: SELECT law_name, article_number, content FROM articles WHERE law_name LIKE '%부가가치세법%' AND content LIKE '%세금계산서%' LIMIT 20;
- 질의: "행정절차법 최근 개정사항"
  SQL: SELECT law_name, effective_date, description FROM amendments WHERE law_name LIKE '%행정절차법%' ORDER BY effective_date DESC LIMIT 20;
- 질의: "산재 보상' 포함 판례 10건"
  SQL: SELECT case_number, court, decision_date, summary FROM cases WHERE summary LIKE '%산재 보상%' LIMIT 10;
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
        """최종 프롬프트 구성 - 자연스럽고 친근한 답변 스타일"""
        # context 타입 검증 및 변환
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            logger.warning(f"⚠️ [BUILD FINAL PROMPT] context is not a dict (type: {type(context)}), converting to dict")
            if isinstance(context, str):
                context = {"context": context}
            else:
                context = {}

        # 문서 데이터 검증 및 로깅 강화
        structured_docs = context.get("structured_documents", {})
        document_count = context.get("document_count", 0)

        # 문서 검증 로깅
        if isinstance(structured_docs, dict):
            raw_documents = structured_docs.get("documents", [])
            doc_count = len(raw_documents) if raw_documents else 0

            # 문서 필드 정규화 및 유효성 검증 (개선: content 없어도 다른 필드 허용)
            normalized_documents = []
            skipped_docs = []
            for doc in raw_documents:
                normalized = self._normalize_document_fields(doc)
                if not normalized:
                    skipped_docs.append({"doc": doc, "reason": "normalized is empty"})
                    continue

                # content가 없거나 10자 이하인 경우에도 다른 필드가 있으면 포함
                content = normalized.get("content", "")
                source = normalized.get("source", "")

                # 최소 조건: content가 10자 이상이거나, source가 있고 다른 필드가 있는 경우
                has_valid_content = content and len(content.strip()) > 10
                has_other_fields = source and (normalized.get("document_id") or normalized.get("metadata"))

                if has_valid_content or has_other_fields:
                    normalized_documents.append(normalized)
                else:
                    skipped_docs.append({
                        "doc": doc,
                        "reason": f"content too short ({len(content)} chars), source={source}"
                    })
                    logger.debug(
                        f"⚠️ [DOCUMENT NORMALIZATION] Skipped document: "
                        f"content_len={len(content)}, source={source}"
                    )

            # 로깅 강화: 정규화 전후 문서 수 상세 로깅
            logger.info(
                f"📋 [FINAL PROMPT] Documents validation: "
                f"context_count={document_count}, raw={doc_count}, "
                f"valid={len(normalized_documents)}, skipped={len(skipped_docs)}"
            )

            if skipped_docs:
                logger.debug(
                    f"⚠️ [DOCUMENT NORMALIZATION] Skipped {len(skipped_docs)} documents: "
                    f"{[d.get('reason', 'unknown') for d in skipped_docs[:3]]}"
                )

            if doc_count > 0 and len(normalized_documents) == 0:
                logger.error(
                    f"❌ [FINAL PROMPT] All {doc_count} documents have empty or invalid content! "
                    f"Skipped reasons: {[d.get('reason', 'unknown') for d in skipped_docs[:3]]}"
                )

            documents = normalized_documents
        else:
            documents = []
            logger.warning(f"⚠️ [FINAL PROMPT] structured_documents is not a dict: {type(structured_docs)}")

        # 🔴 개선 1: base_prompt에 이미 문서가 포함되어 있는지 확인 (개선: 더 많은 패턴 감지)
        has_docs_in_base = False
        doc_patterns = [
            "검색된 법률 문서",
            "제공된 법률 문서",
            "검색된 판례 문서",
            "검색된 법률 문서 및 정보",
            "검색된 법률 문서 및 판례",
            "## 검색된 법률 문서",
            "## 제공된 법률 문서",
            "## 검색된 판례 문서"
        ]
        if any(pattern in base_prompt for pattern in doc_patterns):
            has_docs_in_base = True
            logger.info("✅ [PROMPT OPTIMIZATION] Documents already in base_prompt, skipping duplicate documents section")

        # 원본 context에서 structured_documents 직접 확인 및 추가 (최우선)
        documents_section = ""
        mandatory_section = ""

        # 🔴 개선 2: base_prompt에 문서가 없을 때만 문서 섹션 생성
        # 검색 결과가 있는 경우에만 문서 섹션 추가
        # 검색 결과가 0개일 때는 문서 섹션을 생성하지 않음 (중요!)
        # 관련도가 일정 수준 이하인 문서 필터링
        if not has_docs_in_base and documents and len(documents) > 0:
            # 관련도가 일정 수준 이하인 문서는 제외 (동적 계산)
            sorted_all_docs = sorted(
                documents,
                key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                reverse=True
            )

            # 최고 관련도 점수를 기준으로 동적 임계값 계산 (최고 점수의 70% 이상)
            if sorted_all_docs and len(sorted_all_docs) > 0:
                max_score = sorted_all_docs[0].get("relevance_score", 0.0) if isinstance(sorted_all_docs[0], dict) else 0.0
                low_relevance_threshold = max(0.5, max_score * 0.7) if max_score > 0 else 0.5

                filtered_documents = [
                    d for d in sorted_all_docs
                    if isinstance(d, dict) and d.get("relevance_score", 0.0) >= low_relevance_threshold
                ]

                if len(filtered_documents) < len(sorted_all_docs):
                    logger.info(
                        f"🔍 [DOCUMENT FILTERING] Filtered {len(sorted_all_docs) - len(filtered_documents)} documents "
                        f"with relevance < {low_relevance_threshold:.3f} "
                        f"(max_score: {max_score:.3f}, kept: {len(filtered_documents)})"
                    )

                documents = filtered_documents if filtered_documents else sorted_all_docs[:5]  # 최소 5개는 보장
            else:
                documents = sorted_all_docs[:5] if sorted_all_docs else []

            if documents and len(documents) > 0:
                # 관련도 점수 기준으로 문서 분류
                # 옵션: 상위 N개를 최우선 문서로 지정 (관련도 0.7 이상 문서가 없을 경우 대비)
                sorted_docs = sorted(
                    documents,
                    key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )

                # 관련도 0.65 이상 문서를 최우선 문서로 분류 (기준 완화)
                high_relevance_docs = [d for d in sorted_docs if isinstance(d, dict) and d.get("relevance_score", 0.0) >= 0.65]

                # 관련도 0.65 미만 0.35 이상 문서를 중요 문서로 분류
                medium_relevance_docs = [d for d in sorted_docs if isinstance(d, dict) and 0.35 <= d.get("relevance_score", 0.0) < 0.65]

                # 관련도 0.65 이상 문서가 없으면 상위 3개를 최우선 문서로 지정
                if not high_relevance_docs and len(sorted_docs) > 0:
                    top_count = min(3, len(sorted_docs))
                    high_relevance_docs = sorted_docs[:top_count]
                    medium_relevance_docs = sorted_docs[top_count:] if len(sorted_docs) > top_count else []

                # 🔴 개선: 관련도 기반 유연한 검색 결과 활용 지침
                mandatory_section = "\n\n## ⚠️ 검색 결과 활용 지침\n\n"

                # 관련도 점수 활용 전략 명시
                mandatory_section += "**검색 결과 활용 우선순위**:\n"
                mandatory_section += "- 관련도 0.8 이상: 핵심 법적 근거로 직접 인용\n"
                mandatory_section += "- 관련도 0.6-0.8: 관련성이 높으면 보충 설명에 활용 권장\n"
                mandatory_section += "- 관련도 0.6 미만: 질문과 직접 관련 없으면 언급하지 않아도 됨\n\n"

                # 관련도별 문서 목록 제공 (참고용)
                if high_relevance_docs:
                    doc_refs = []
                    for idx, doc in enumerate(high_relevance_docs[:3], 1):
                        law_name = doc.get("law_name", "")
                        article_no = doc.get("article_no", "")
                        score = doc.get("relevance_score", 0.0)
                        if law_name and article_no:
                            doc_refs.append(f"문서 {idx}({law_name} 제{article_no}조, 관련도: {score:.2f})")
                        else:
                            source = doc.get("source", "")
                            if source:
                                doc_refs.append(f"문서 {idx}({source}, 관련도: {score:.2f})")

                    if doc_refs:
                        mandatory_section += f"**고관련도 문서 (참고용)**: {', '.join(doc_refs)}\n"
                        mandatory_section += "→ 질문과 직접 관련이 높으면 우선 활용하세요\n\n"

                mandatory_section += "**검색 결과가 질문과 부합하지 않을 때**:\n"
                mandatory_section += "- 검색된 자료에 [구체적 내용]이 없어서, [법령명]의 기본 원칙을 바탕으로 설명드릴게요...\n\n"

                mandatory_section += "**절대 금지**:\n"
                mandatory_section += "- ❌ 검색 결과를 무시하고 일반 지식만으로 답변\n"
                mandatory_section += "- ❌ 검색 결과 없이 추측으로 답변\n"
                mandatory_section += "- ❌ '정보가 부족합니다'만 답변\n"
                mandatory_section += "- ❌ 관련도가 낮은 문서를 무리하게 인용\n\n"

                # 🔴 개선 4: 문서 섹션 단일화 (중복 형식 제거, 헬퍼 메서드 사용)
                documents_section = "\n\n## 🔍 검색된 법률 문서\n\n"

                # 🔴 추가 개선: 관련도 활용 전략 간단히 명시 (mandatory_section과 중복 제거)
                # mandatory_section에 이미 상세한 전략이 있으므로 간단한 참고만 추가
                documents_section += "**참고**: 위 문서들의 관련도 점수를 참고하여 우선순위를 정하세요. 관련도가 높은 문서를 우선적으로 활용하세요.\n\n"

                # 최우선 문서
                if high_relevance_docs:
                    # 관련도 기준에 따라 섹션 제목 조정
                    max_high_score = max([d.get("relevance_score", 0.0) for d in high_relevance_docs if isinstance(d, dict)]) if high_relevance_docs else 0.0
                    if max_high_score >= 0.65:
                        documents_section += "### 🔴 최우선 문서 (관련도 0.65 이상)\n\n"
                    else:
                        documents_section += "### 🔴 최우선 문서 (상위 문서)\n\n"
                    for idx, doc in enumerate(high_relevance_docs[:5], 1):
                        documents_section += self._format_document_for_prompt(doc, idx, is_high_priority=True)

                # 중요 문서
                if medium_relevance_docs:
                    documents_section += "### 🟡 중요 문서 (관련도 0.35~0.65)\n\n"
                    for idx, doc in enumerate(medium_relevance_docs[:3], 1):
                        documents_section += self._format_document_for_prompt(doc, idx, is_high_priority=False)

                logger.info(f"✅ [FINAL PROMPT] Added {len(documents)} documents (High: {len(high_relevance_docs)}, Medium: {len(medium_relevance_docs)})")

        # 🔴 개선 3: 필수 준수 사항 섹션이 비어있을 때 처리
        if has_docs_in_base and not mandatory_section:
            # base_prompt에 문서가 있지만 mandatory_section이 없는 경우 간단한 지침 추가
            mandatory_section = "\n\n## ⚠️ 핵심 지침\n\n"
            mandatory_section += "**위 검색된 문서들을 반드시 참고하여 답변하세요.**\n\n"
            mandatory_section += "**절대 금지**:\n"
            mandatory_section += "- ❌ 검색 결과를 무시하고 일반 지식만으로 답변\n"
            mandatory_section += "- ❌ 검색 결과 없이 추측으로 답변\n"
            mandatory_section += "- ❌ '정보가 부족합니다'만 답변\n\n"

        # 문서 섹션이 생성되지 않았는데 문서가 있는 경우 폴백 처리
        if not documents_section and documents and len(documents) > 0 and not has_docs_in_base:
            logger.warning(
                f"⚠️ [FINAL PROMPT] No documents section created despite having {len(documents)} documents! "
                f"Creating fallback section."
            )
            documents_section = self._build_fallback_documents_section(documents)
            if documents_section and not mandatory_section:
                mandatory_section = "\n\n## ⚠️ 핵심 지침\n\n"
                mandatory_section += "**🔴 검색 결과 반드시 활용**: 아래 문서들의 내용을 반드시 참고하여 답변하세요.\n\n"

        # structured_documents가 비어있지만 context 텍스트나 prompt_optimized_text가 있는 경우 폴백 처리
        if not documents_section:
            prompt_optimized_text = context.get("prompt_optimized_text", "")
            context_text = context.get("context", "")

            # prompt_optimized_text 우선 사용
            # 🔴 개선: 문서 섹션 제목 통일 및 지침 문구 간소화
            if prompt_optimized_text and len(prompt_optimized_text.strip()) > 100:
                documents_section = "\n\n## 🔍 검색된 법률 문서\n\n"
                documents_section += prompt_optimized_text[:5000] + ("..." if len(prompt_optimized_text) > 5000 else "")
                documents_section += "\n\n"
                if not mandatory_section:
                    mandatory_section = "\n\n## ⚠️ 핵심 지침\n\n"
                    mandatory_section += "**위 검색된 문서들을 참고하여 답변하세요.**\n\n"
                logger.info(
                    f"✅ [FINAL PROMPT] Added prompt_optimized_text to final prompt as fallback "
                    f"({len(prompt_optimized_text)} chars)"
                )
            # context_text가 있고 document_count가 0보다 크면 문서가 있다는 의미
            elif context_text and len(context_text.strip()) > 100 and document_count > 0:
                documents_section = "\n\n## 🔍 검색된 법률 문서\n\n"
                documents_section += context_text[:5000] + ("..." if len(context_text) > 5000 else "")
                documents_section += "\n\n"
                if not mandatory_section:
                    mandatory_section = "\n\n## ⚠️ 핵심 지침\n\n"
                    mandatory_section += "**위 검색된 문서들을 참고하여 답변하세요.**\n\n"
                logger.info(
                    f"✅ [FINAL PROMPT] Added context_text to final prompt as fallback "
                    f"({len(context_text)} chars, document_count: {document_count})"
                )

        # Few-shot 예시 추가 (실제 검색 결과 기반)
        relevant_examples = self._get_relevant_examples(question_type, documents)
        examples_section = f"\n{relevant_examples}\n" if relevant_examples else ""
        
        # 답변 형식 예시 추가
        answer_examples_section = self._get_answer_examples_section(question_type)
        if answer_examples_section:
            examples_section += f"\n{answer_examples_section}\n"

        # 🔴 개선 5: 활용 가이드 간소화 (중복 제거)
        usage_guide = ""
        if documents_section and documents and len(documents) > 0 and not has_docs_in_base:
            usage_guide = """

### 📖 인용 가이드
- 인용 포맷: "[법령: 민법 제543조]", "[판례: 대법원 2020다12345]"
- 문서 내용을 요약하여 설명하고, 출처를 명시하세요
- 여러 문서의 정보를 종합하여 답변하세요
"""

        # 검색 결과가 없을 때 base_prompt를 단순화하여 프롬프트 길이 줄이기
        # 검색 결과가 없으면 프롬프트를 단순화 (프롬프트 출력 방지)
        # 개선: 원본 문서 수를 고려하여 has_no_documents 조건 완화
        normalized_doc_count = len(documents) if documents else 0
        raw_doc_count = (
            len(structured_docs.get("documents", []))
            if isinstance(structured_docs, dict) else 0
        ) or document_count or 0

        # documents_section이 비어있거나 documents가 비어있는 경우
        has_no_documents_section = not documents_section or len(documents_section.strip()) == 0
        has_no_normalized_documents = not documents or len(documents) == 0

        # 원본 문서가 있었는데 정규화 후 비어있는 경우도 고려
        has_no_documents = (
            has_no_documents_section and
            has_no_normalized_documents and
            raw_doc_count == 0  # 원본 문서도 없는 경우에만 True
        )

        # 로깅 강화: 문서 상태 상세 로깅
        if raw_doc_count > 0 and normalized_doc_count == 0:
            logger.warning(
                f"⚠️ [PROMPT SIMPLIFICATION] Warning: {raw_doc_count} raw documents existed "
                f"but {normalized_doc_count} normalized documents. "
                f"documents_section={'exists' if documents_section else 'empty'}"
            )

        if has_no_documents:
            # 검색 결과가 없을 때는 base_prompt를 단순화
            # 복잡한 지시사항이 LLM에게 프롬프트 자체를 출력하게 할 수 있음
            simplified_base = self._simplify_prompt_for_no_results(base_prompt)
            logger.info(
                f"📝 [PROMPT SIMPLIFICATION] Simplified prompt for no search results "
                f"(raw_docs={raw_doc_count}, normalized_docs={normalized_doc_count}, "
                f"original: {len(base_prompt)} chars, simplified: {len(simplified_base)} chars)"
            )
        else:
            simplified_base = base_prompt
            if raw_doc_count > 0:
                logger.info(
                    f"✅ [PROMPT SIMPLIFICATION] Keeping full prompt "
                    f"(raw_docs={raw_doc_count}, normalized_docs={normalized_doc_count})"
                )

        # 🔴 개선 6: 최종 지침 통합 (중복 제거, 로직 수정)
        # 문서가 있으면 적절한 지침 표시, 없으면 "문서 없음" 표시
        has_any_documents = (
            (documents_section and len(documents_section.strip()) > 0) or
            has_docs_in_base or
            (documents and len(documents) > 0)
        )

        if has_any_documents:
            # 제공된 문서 목록 추출
            doc_list = []
            if documents and len(documents) > 0:
                # high_relevance_docs가 있으면 그것을 우선, 없으면 상위 문서 사용
                sorted_docs = sorted(
                    documents,
                    key=lambda x: x.get("relevance_score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )

                high_relevance_docs_for_list = [d for d in sorted_docs if isinstance(d, dict) and d.get("relevance_score", 0.0) >= 0.65]

                # 🔴 개선: high_relevance_docs가 비어있거나 적으면 전체 문서에서 상위 문서 추출
                if not high_relevance_docs_for_list:
                    docs_for_list = sorted_docs[:5]  # 상위 5개
                elif len(high_relevance_docs_for_list) >= 3:
                    docs_for_list = high_relevance_docs_for_list[:5]
                else:
                    # high_relevance_docs가 1-2개만 있으면 상위 문서와 함께 사용
                    docs_for_list = sorted_docs[:5]

                for doc in docs_for_list:
                    if isinstance(doc, dict):
                        law_name = doc.get("law_name", "")
                        article_no = doc.get("article_no", "")
                        if law_name and article_no:
                            doc_list.append(f"{law_name} 제{article_no}조")
                        else:
                            source = doc.get("source", "")
                            if source:
                                doc_list.append(source)

            # 🔴 개선: has_docs_in_base일 때 base_prompt에서 문서 추출
            if not doc_list and has_docs_in_base:
                import re
                # base_prompt에서 문서 정보 추출
                # 패턴: "### 문서 N: 민법 제XXX조" 또는 "**문서 N**: 민법 제XXX조"
                doc_patterns = [
                    r'###\s*문서\s*\d+:\s*([^\(]+)',
                    r'\*\*문서\s*\d+\*\*:\s*([^\(]+)',
                    r'문서\s*\d+[:\s]+([^\(]+)'
                ]

                found_docs = []
                for pattern in doc_patterns:
                    matches = re.findall(pattern, base_prompt)
                    for match in matches[:5]:  # 최대 5개
                        match = match.strip()
                        # 조문 번호 추출
                        article_match = re.search(r'([가-힣\s]+)\s*제\s*(\d+)\s*조', match)
                        if article_match:
                            law_name = article_match.group(1).strip()
                            article_no = article_match.group(2)
                            doc_ref = f"{law_name} 제{article_no}조"
                            if doc_ref not in found_docs:
                                found_docs.append(doc_ref)
                        elif match and match not in found_docs:
                            found_docs.append(match)

                if found_docs:
                    doc_list = found_docs[:5]
                    logger.info(f"✅ [DOCUMENT EXTRACTION] Extracted {len(doc_list)} documents from base_prompt: {', '.join(doc_list[:3])}...")

            if doc_list:
                doc_list_str = ', '.join(doc_list[:5])
                # 🔴 개선: 실제 제공된 문서 조문을 예시로 사용
                example_doc = doc_list[0] if doc_list else "법령명 제XX조"
                # 판례명 예시 찾기
                example_precedent = None
                for doc in doc_list:
                    if '법원' in doc or '판례' in doc or '-' in doc:
                        example_precedent = doc
                        break
                
                # 법령 조문 예시 찾기
                example_law = None
                for doc in doc_list:
                    if '제' in doc and '조' in doc:
                        example_law = doc
                        break
                
                examples_text = f"""  * 예시 1: "{example_doc}에 따르면..."
  * 예시 2: "[법령: {example_doc}]"
  * 예시 3: "위 검색 결과 문서 중 {example_doc}에 명시된 바와 같이..."
  * 예시 4: "민법 제750조에 따르면..." (법령 조문 인용 형식)"""
                
                if example_precedent:
                    examples_text += f"\n  * 예시 5: \"{example_precedent} 판결에 의하면...\" (판례 인용 형식)"
                else:
                    examples_text += f"\n  * 예시 5: \"대구지방법원 영덕지원 대구지방법원영덕지원-2021고단3 판결에 의하면...\" (판례 인용 형식)"
                
                # 재생성 이유 확인 및 프롬프트 강화
                regeneration_reason = context.get("regeneration_reason") if (isinstance(context, dict) and context) else None
                regeneration_note = ""
                if regeneration_reason:
                    if regeneration_reason == "specific_case_copy" or regeneration_reason == "specific_case_in_start":
                        regeneration_note = """
### ⚠️ 재생성 사유: 특정 사건 내용 복사 감지
이전 답변에서 특정 사건의 내용이 그대로 복사되었습니다. 다음 사항을 반드시 준수하세요:
- ❌ 특정 사건번호(예: "2014가단3882")를 답변에 포함하지 마세요
- ❌ 특정 당사자명(예: "피고 엘지", "원고 본인")을 답변에 포함하지 마세요
- ❌ "[문서: 대전지방법원-2014가단3882]" 같은 형식으로 시작하지 마세요
- ✅ 판례의 법적 원칙만 추출하여 일반적으로 설명하세요
- ✅ 답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작하세요
"""
                    elif regeneration_reason == "general_principle_not_first" or regeneration_reason == "general_principle_not_in_start":
                        regeneration_note = """
### ⚠️ 재생성 사유: 일반 법적 원칙이 먼저 설명되지 않음
이전 답변에서 일반 법적 원칙이 먼저 설명되지 않았습니다. 다음 사항을 반드시 준수하세요:
- ✅ 답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작하세요
- ✅ 예: "계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다."
- ❌ 특정 사건번호나 사실관계 서술로 시작하지 마세요
"""
                
                final_instruction_section = f"""
## 검색 결과 활용 지침 (필수 준수 - 답변 품질 평가 기준)
{regeneration_note}
### ⚠️ 답변 시작 시 필수 사항 (절대 준수)
- **답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작해야 합니다.**
- 예: "계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다."
- ❌ 금지: "[문서: 대전지방법원-2014가단3882]" 같은 특정 사건번호로 시작
- ❌ 금지: "나아가 계약서의 성립에 관하여 피고의..." 같은 특정 사건의 사실관계 서술로 시작
- ❌ 금지: "주어진 문서를 바탕으로 답변드리면:" 같은 문구 사용

### 답변 작성 원칙
- **일반적이고 포괄적인 답변 우선**: 특정 사건이나 판례에만 집중하지 말고, 질문의 일반적인 법적 원칙과 주의사항을 포괄적으로 다루세요.
- **균형잡힌 접근**: 검색된 문서 중 특정 사건에만 집중하지 말고, 다양한 법령, 판례, 해석례를 종합하여 일반적인 조언을 제공하세요.
- **특정 사건 내용 복사 금지**: 특정 사건번호(예: "2014가단3882"), 특정 당사자명(예: "피고 엘지", "원고 본인"), 특정 사건의 사실관계 서술(예: "이 사건 각 계약서 작성 당시")을 답변에 포함하지 마세요.

### 문서 활용 지침
- 다음 문서 중 질문과 관련성이 높은 것을 우선 활용하세요: {doc_list_str}
- 관련도가 낮은 문서는 무리하게 인용하지 마세요
- 검색 결과가 질문과 부합하지 않으면 명시하고 기본 원칙으로 답변하세요
- **⚠️ 각 인용에 반드시 명확한 출처 표기 (필수)**:
{examples_text}
- **답변에서 검색된 문서의 출처(법령명, 조문번호, 판례명 등)를 최소 2개 이상 명시적으로 인용하세요**
- **⚠️ 법령 조문 인용을 판례 인용보다 우선하세요** (법령 조문이 있으면 반드시 법령 조문을 먼저 인용)
- **법령 조문 인용 필수**: 검색 결과에 법령 조문이 있으면 최소 1개 이상의 법령 조문을 반드시 인용하세요 (예: "민법 제750조에 따르면...")
- 판례 인용 시 구체적인 판례명과 사건번호를 포함하되, 특정 사건에만 집중하지 말고 일반적인 법적 원칙을 중심으로 설명하세요 (예: "대구지방법원 영덕지원 대구지방법원영덕지원-2021고단3 판결에 의하면...")
- 단순히 "법령에 따르면"이 아닌 구체적인 법령명과 조문번호를 포함하세요 (예: "민법 제750조", "형법 제250조")
- 법령 조문 인용이 없으면 답변 품질이 낮게 평가됩니다
- 문서 인용이 부족하면 답변 품질이 낮게 평가됩니다
"""
            else:
                # doc_list가 비어있을 때만 기본 예시 사용
                # 실제 문서 형식 예시 추가
                example_doc = doc_list[0] if doc_list else "민법 제750조"
                example_law = None
                example_precedent = None
                for doc in doc_list[:3]:
                    if isinstance(doc, str):
                        if "제" in doc and "조" in doc:
                            example_law = doc
                            break
                        elif "법원" in doc or "판결" in doc:
                            example_precedent = doc
                
                examples_text = f"""  * 예시 1: "{example_doc}에 따르면..."
  * 예시 2: "[법령: {example_doc}]"
  * 예시 3: "위 검색 결과 문서 중 {example_doc}에 명시된 바와 같이..."
  * 예시 4: "민법 제750조에 따르면..." (법령 조문 인용 형식)"""
                if example_precedent:
                    examples_text += f"\n  * 예시 5: \"{example_precedent} 판결에 의하면...\" (판례 인용 형식)"
                elif example_law:
                    examples_text += f"\n  * 예시 5: \"{example_law}에 명시된 바와 같이...\" (법령 조문 인용 형식)"
                
                final_instruction_section = f"""
## 검색 결과 활용 지침 (필수 준수 - 답변 품질 평가 기준)
- 위 검색 결과 문서에서 질문과 관련성이 높은 문서를 우선 활용하세요
- **⚠️ 법령 조문 인용을 판례 인용보다 우선하세요** (법령 조문이 있으면 반드시 법령 조문을 먼저 인용)
- **법령 조문 인용 필수**: 검색 결과에 법령 조문이 있으면 최소 1개 이상의 법령 조문을 반드시 인용하세요
- **각 인용에 반드시 명확한 출처 표기 (필수)**:
{examples_text}
- **답변에서 검색된 문서의 출처(법령명, 조문번호, 판례명 등)를 최소 2개 이상 명시적으로 인용하세요**
- 단순히 "법령에 따르면"이 아닌 구체적인 법령명과 조문번호를 포함하세요 (예: "민법 제750조", "형법 제250조")
- 법령 조문 인용이 없으면 답변 품질이 낮게 평가됩니다
- 문서 인용이 부족하면 답변 품질이 낮게 평가됩니다
- 검색 결과가 질문과 부합하지 않으면 명시하고 기본 원칙으로 답변하세요
"""
        else:
            # 검색 결과가 없을 때 명확한 제한사항 명시
            search_failed = documents.get("search_failed", False) if isinstance(documents, dict) else False
            search_failure_reason = documents.get("search_failure_reason", "unknown") if isinstance(documents, dict) else "unknown"
            
            if search_failed:
                failure_details = ""
                if "no_results_from_both_semantic_and_keyword" in search_failure_reason:
                    failure_details = (
                        "\n\n**⚠️ 중요: 검색 시스템 오류**\n"
                        "- 의미 검색(semantic)과 키워드 검색(keyword) 모두 결과를 반환하지 못했습니다.\n"
                        "- 이는 데이터베이스가 초기화되지 않았거나, FTS 테이블이 없거나, embeddings가 생성되지 않았을 수 있습니다.\n"
                        "- 일반적인 법률 정보를 제공하되, 반드시 다음 사항을 명시하세요:\n"
                        "  1. 검색된 법률 문서가 없어 정확한 법령 조문을 인용할 수 없다는 점\n"
                        "  2. 제공되는 정보는 일반적인 법률 원칙에 기반한 것이며, 구체적인 사안에 대한 법률 자문은 변호사와 상담이 필요하다는 점\n"
                        "  3. 가능한 경우 질문과 관련된 법령명과 조문번호를 언급하되, 구체적인 내용은 확인이 필요하다는 점"
                    )
                
                final_instruction_section = f"""
## ⚠️ 참고사항 (중요)
현재 관련 법률 문서를 찾지 못했습니다.{failure_details}

**답변 작성 시 주의사항**:
- 일반적인 법률 정보를 제공하되, 반드시 한계를 명시하세요
- 검색된 법률 문서가 없어 정확한 법령 조문을 인용할 수 없다는 점을 명확히 밝히세요
- 가능한 경우 관련 법령명과 조문번호를 언급하되, 구체적인 내용 확인이 필요하다는 점을 명시하세요
- 모든 답변은 일반적인 법률 정보 제공에 그치며, 구체적인 사안에 대한 법률 자문은 변호사와 상담이 필요하다는 점을 강조하세요
"""
            else:
                final_instruction_section = """
## ⚠️ 참고사항
현재 관련 법률 문서를 찾지 못했습니다. 일반적인 법률 정보를 제공하되, 한계를 명시하세요.

**답변 작성 시 주의사항**:
- 검색된 법률 문서가 없어 정확한 법령 조문을 인용할 수 없다는 점을 명확히 밝히세요
- 가능한 경우 관련 법령명과 조문번호를 언급하되, 구체적인 내용 확인이 필요하다는 점을 명시하세요
- 모든 답변은 일반적인 법률 정보 제공에 그치며, 구체적인 사안에 대한 법률 자문은 변호사와 상담이 필요하다는 점을 강조하세요
"""

        # 검색된 키워드 추출 (context에서)
        extracted_keywords = context.get("extracted_keywords", []) if isinstance(context, dict) else []
        if not extracted_keywords and isinstance(context, dict):
            extracted_keywords = context.get("keywords", [])
        if not extracted_keywords and isinstance(context, dict) and isinstance(context.get("search", {}), dict):
            extracted_keywords = context.get("search", {}).get("extracted_keywords", [])
        
        # 키워드 섹션 생성
        keywords_section = ""
        if extracted_keywords and len(extracted_keywords) > 0:
            keywords_list = ", ".join(extracted_keywords[:20])  # 최대 20개만 표시
            keywords_section = f"""
---

## 🔑 검색된 핵심 키워드 (⚠️ 반드시 답변에 포함하세요)
다음 키워드들은 검색 과정에서 추출된 핵심 용어입니다. **답변에 반드시 이 키워드들을 자연스럽게 포함**하여 답변의 관련성과 정확성을 높이세요.

**검색 키워드**: {keywords_list}

**키워드 활용 지침**:
- 위 키워드들을 답변에 자연스럽게 포함하세요
- 키워드를 단순히 나열하지 말고, 문맥에 맞게 사용하세요
- 키워드가 포함되지 않으면 답변 품질이 낮게 평가됩니다
- 최소 5개 이상의 키워드를 답변에 포함하는 것을 권장합니다
"""

        final_prompt = f"""{simplified_base}{mandatory_section}{documents_section}{usage_guide}

---

## 📝 사용자 질문
{query}

{keywords_section}

---

{examples_section}

---

## 📋 답변 스타일
- 전문적이되 친근한 존댓말 사용 ("~예요/~해요" 선호)
- "~입니다/~습니다"는 필요한 경우만 사용
- 과도한 형식(제목, 박스, 이모지)은 피하기
- 예: "민법 제550조에 따르면 계약을 해지하면 장래에 대해서만 효력을 잃게 돼요."
- ⚠️ 주의: 프롬프트 내부에는 이모지 사용 가능하지만, AI 답변에는 이모지 사용 금지

## 답변 구성 원칙 (⚠️ 반드시 이 순서를 따르세요)
1. **일반 법적 원칙 (필수, 먼저 설명)**: 질문과 관련된 일반적인 법적 원칙과 규칙을 먼저 설명하세요. 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
2. **주요 주의사항**: 실무에서 주의해야 할 주요 사항들을 체계적으로 정리하세요.
3. **관련 법령**: 관련 법령 조문을 구체적으로 인용하되, 특정 사건에만 국한되지 않도록 일반적인 법적 원칙을 중심으로 설명하세요.
4. **판례 및 해석례**: 일반적인 법적 원칙을 뒷받침하는 판례와 해석례를 종합적으로 활용하되, 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
5. **실무 조언**: 구체적인 행동 방안이나 주의사항을 제시하세요.

**⚠️ 중요**: 
- 답변의 시작 부분에 반드시 일반적인 법적 원칙을 먼저 설명하고, 그 다음에 구체적인 사례나 판례를 보완적으로 제시하세요.
- 특정 사건에만 집중하지 마세요.
- 특정 판례 사건의 내용을 그대로 복사하지 마세요. 판례의 법적 원칙을 추출하여 일반적으로 설명하세요.

## 필수 포함 요소 (⚠️ 답변 품질 평가 기준)
각 답변에 반드시 포함:
1. **법적 근거**: 관련 법령 조문 정확히 인용 (⚠️ 필수 - 품질 평가의 핵심)
   - 검색된 법령 조문이 있으면 **반드시 인용하세요** (인용하지 않으면 답변 품질이 매우 낮게 평가됩니다)
   - 형식: "민법 제750조에 따르면..." 또는 "[법령: 민법 제750조]"
   - **최소 2개 이상의 법령 조문 인용 필수** (1개 -> 2개로 증가)
   - 단순히 "법령에 따르면"이 아닌 **구체적인 법령명과 조문번호를 포함하세요**
   - 문서에서 법령 조문을 찾을 수 없어도, 질문과 관련된 일반적인 법령 조문을 인용하세요
   - 예: "계약 해지" 질문 → "민법 제550조", "민법 제551조" 등 관련 조문 인용 필수

2. **판례 인용**: 관련 판례가 있으면 인용 (가능한 경우)
   - 형식: "대법원 2020다12345 판결에 의하면..." 또는 "[판례: 대법원 2020다12345]"
   - 판례 인용 시 구체적인 판례명과 사건번호를 포함하세요

3. **문서 출처 명시**: 사용한 문서의 출처를 명시하세요 (⚠️ 필수)
   - 형식: "문서 1에 따르면..." 또는 "[문서: 출처명]"
   - 최소 2개 이상의 문서 출처 명시 필수

4. **실무적 의미**: 실제로 무엇을 의미하는지 설명

5. **실행 가능한 조언**: 구체적으로 어떻게 해야 하는지

6. **주의사항**: 놓치기 쉬운 함정 (필요시에만)

⚠️ **Citation 필수 체크리스트** (답변 품질 평가 기준 - 반드시 확인):
- [ ] **법령 조문을 최소 2개 이상 인용했는가?** (없으면 답변 품질이 매우 낮게 평가됩니다)
- [ ] 검색된 법령 조문이 있으면 반드시 인용했는가?
- [ ] 구체적인 법령명과 조문번호를 포함했는가? (예: "민법 제750조", "형법 제250조")
- [ ] 최소 2개 이상의 문서 출처를 명시했는가?
- [ ] 각 인용에 명확한 출처 표기가 있는가?
- [ ] 검색된 키워드를 답변에 포함했는가? (최소 5개 이상 권장)
- [ ] **법령 조문 인용이 없으면 답변 품질이 매우 낮게 평가되며 재생성될 수 있습니다**
- [ ] 문서 인용이 부족하면 답변 품질이 낮게 평가됩니다

**예시 형식**:
❌ "민법 제550조는 해지의 효과를 규정합니다"
✅ "민법 제550조에 따르면, 계약을 해지하면 앞으로만 효력이 없어져요. 즉, 이미 받은 돈이나 물건은 돌려주지 않아도 됩니다. 다만 해지 이후 발생한 손해에 대해서는 배상을 청구할 수 있어요. (문서 1, 문서 2 참조)"

## 📋 Citation 형식 예시 (필수 준수)

### 법령 조문 인용 형식:
✅ **올바른 예시**:
- "민법 제750조에 따르면, 불법행위로 인한 손해배상 책임이 발생합니다."
- "[법령: 민법 제750조]에 명시된 바와 같이..."
- "민법 제750조 제1항에 따르면..."
- "형법 제250조에 의하면..."

❌ **잘못된 예시**:
- "법령에 따르면..." (구체적인 법령명과 조문번호 없음)
- "관련 법령에 의하면..." (구체적인 법령명과 조문번호 없음)
- "법률에 명시된 바와 같이..." (구체적인 법령명과 조문번호 없음)

### 판례 인용 형식:
✅ **올바른 예시**:
- "대법원 2020다12345 판결에 의하면..."
- "[판례: 대법원 2020다12345]에 따르면..."
- "대구지방법원 영덕지원 대구지방법원영덕지원-2021고단3 판결에 의하면..."
- "서울중앙지방법원 2019가단123456 판결에 따르면..."

❌ **잘못된 예시**:
- "판례에 의하면..." (구체적인 판례명과 사건번호 없음)
- "관련 판례에 따르면..." (구체적인 판례명과 사건번호 없음)
- "법원 판결에 의하면..." (구체적인 판례명과 사건번호 없음)

### 문서 출처 명시 형식:
✅ **올바른 예시**:
- "문서 1에 따르면..."
- "[문서: 민법 제750조 해설]에 명시된 바와 같이..."
- "검색 결과 문서 중 민법 제750조 관련 문서에 의하면..."
- "위 검색 결과 문서 중 문서 2에 명시된 바와 같이..."

❌ **잘못된 예시**:
- "관련 문서에 따르면..." (구체적인 문서 출처 없음)
- "검색 결과에 의하면..." (구체적인 문서 출처 없음)
- "참고 자료에 따르면..." (구체적인 문서 출처 없음)

⚠️ **Citation 필수 체크리스트** (답변 품질 평가 기준 - 반드시 확인):
- [ ] 검색된 법령 조문이 있으면 반드시 인용했는가? (예: "민법 제750조")
- [ ] 최소 2개 이상의 문서 출처를 명시했는가? (예: "문서 1", "문서 2")
- [ ] 각 인용에 명확한 출처 표기가 있는가? (예: "[법령: 민법 제750조]")
- [ ] 법령 조문 인용이 없으면 답변 품질이 낮게 평가됩니다
- [ ] 문서 인용이 부족하면 답변 품질이 낮게 평가됩니다
- [ ] 단순히 "법령에 따르면"이 아닌 구체적인 법령명과 조문번호를 포함했는가?
- [ ] 판례 인용 시 구체적인 판례명과 사건번호를 포함했는가?

**⚠️ 중요**: 위 체크리스트를 모두 충족하지 않으면 답변 품질이 낮게 평가되며, 
답변이 재생성될 수 있습니다.

## 📏 답변 길이 요구사항
- **최소 길이**: 500자 이상 (한국어 기준)
- **권장 길이**: 800-2000자 (질문의 복잡도에 따라 조정)
- **최대 길이**: 3000자 이하 (간결성 유지)
- 답변이 너무 짧으면 (500자 미만) 품질이 낮게 평가되며 재생성될 수 있습니다.
- 질문이 단순한 경우에도 최소 500자 이상의 상세한 설명을 제공하세요.

## ⚠️ 절대 금지 사항
- ❌ "주어진 문서에서 관련 정보를 찾을 수 없습니다" 같은 답변은 절대 금지
- ❌ "더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다" 같은 회피적 답변 금지
- ❌ 문서가 제공되었는데도 "정보를 찾을 수 없습니다"라고 답변하는 것은 절대 금지
- ✅ 문서가 제공되었으면 반드시 문서 내용을 활용하여 답변해야 합니다
- ✅ 문서 내용이 직접적으로 관련이 없어도, 일반적인 법률 지식을 바탕으로 답변해야 합니다
- ✅ 문서를 참고하여 답변하되, 문서 내용이 부족하면 일반적인 법률 지식으로 보완하세요

## 📚 문서 활용 필수 지침
1. **문서가 제공된 경우**:
   - 반드시 문서 내용을 인용하고 활용해야 합니다
   - **⚠️ 중요**: 문서 내용을 바탕으로 답변을 구성하되, 특정 사건의 내용을 그대로 복사하지 마세요
   - 문서에서 일반적인 법적 원칙을 추출하여 설명하세요
   - 문서에서 직접적인 답을 찾을 수 없어도, 관련 내용을 참고하여 답변하세요

2. **문서 내용 활용 방법**:
   - **⚠️ 금지**: 특정 판례 사건의 내용을 그대로 복사하지 마세요
   - **⚠️ 필수**: 판례에서 법적 원칙을 추출하여 일반적으로 설명하세요
   - 문서의 핵심 내용을 요약하여 답변에 포함하되, 특정 사건의 세부사항은 생략하세요
   - 문서에서 인용할 법령 조문이나 판례를 찾아 활용하되, 판례는 법적 원칙 중심으로 설명하세요
   - 문서의 맥락을 이해하고 질문에 맞게 해석하여 답변하세요

3. **문서가 부족한 경우**:
   - 제공된 문서 내용을 최대한 활용
   - 일반적인 법률 지식으로 보완
   - "문서에서 확인할 수 있는 내용은..."과 같이 명시적으로 구분

**⚠️ 절대 금지 사항 (문서 활용 시)**:
- ❌ 특정 판례 사건의 내용을 그대로 복사하는 것
- ❌ 특정 사건의 세부사항(당사자명, 사건번호 등)을 상세히 나열하는 것
- ❌ 판례 사건의 사실관계를 그대로 설명하는 것
- ✅ 판례에서 법적 원칙을 추출하여 일반적으로 설명하는 것
- ✅ 여러 판례를 종합하여 일반적인 법적 원칙을 설명하는 것

{final_instruction_section}

**⚠️ 최종 확인 사항**:
1. 일반적인 법적 원칙을 먼저 설명했는가?
2. 특정 사건의 내용을 그대로 복사하지 않았는가?
3. 판례는 법적 원칙 중심으로 설명했는가?
4. 법령 조문을 구체적으로 인용했는가?
5. 문서 출처를 명시했는가?

**⚠️ 답변 시작 시 필수 사항 (절대 준수)**:
- **답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작해야 합니다.**
- ❌ 금지: "[문서: 대전지방법원-2014가단3882]" 같은 특정 사건번호로 시작
- ❌ 금지: "나아가 계약서의 성립에 관하여 피고의..." 같은 특정 사건의 사실관계 서술로 시작
- ❌ 금지: "주어진 문서를 바탕으로 답변드리면:" 같은 문구 사용
- ✅ 올바른 예: "계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다."
- ✅ 올바른 예: "계약서 작성과 관련하여 일반적으로 적용되는 법적 원칙은..."

**⚠️ 특정 사건 내용 복사 금지 (절대 금지)**:
- ❌ 특정 사건번호(예: "2014가단3882", "대전지방법원-2014가단3882")를 답변에 포함하지 마세요
- ❌ 특정 당사자명(예: "피고 엘지", "피고 에스케이", "원고 본인")을 답변에 포함하지 마세요
- ❌ 특정 사건의 사실관계 서술(예: "이 사건 각 계약서 작성 당시", "나아가 이 사건 각 계약서 작성 당시")을 그대로 복사하지 마세요
- ❌ "[문서: ...]" 형식으로 특정 사건번호를 포함한 인용을 하지 마세요
- ✅ 판례의 법적 원칙만 추출하여 일반적으로 설명하세요
- ✅ 판례 인용 시 "[판례: 대전지방법원 2014가단3882]" 형식으로 인용하되, 특정 사건의 사실관계는 설명하지 마세요

답변을 시작하세요:
"""
        return final_prompt

    def _validate_prompt_contains_documents(self, final_prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """프롬프트에 실제 문서 내용이 포함되었는지 검증 (강화된 버전)"""
        # context 타입 검증 및 변환
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            logger.warning(f"⚠️ [VALIDATE PROMPT] context is not a dict (type: {type(context)}), converting to dict")
            if isinstance(context, str):
                context = {"context": context}
            else:
                context = {}
        
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
        structured_docs = context.get("structured_documents", {}) if isinstance(context, dict) else {}
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
        context_text = context.get("context", "") if isinstance(context, dict) else ""
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

    def _normalize_document_fields(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """문서 필드명 정규화 - 법률명, 조문 번호 등 명시적 추출"""
        if not isinstance(doc, dict):
            return {}

        # content 필드: 여러 가능한 필드명에서 추출
        content = (
            doc.get("content", "") or
            doc.get("text", "") or
            doc.get("document_text", "") or
            doc.get("full_text", "") or
            doc.get("body", "") or
            str(doc.get("metadata", {}).get("text", "") if isinstance(doc.get("metadata"), dict) else "") or
            ""
        )

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

        # 메타데이터에서 법률 정보 추출
        metadata = doc.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # 법률명 추출 (여러 가능한 필드명 지원)
        law_name = (
            doc.get("law_name", "") or
            metadata.get("law_name", "") or
            metadata.get("statute_name", "") or
            metadata.get("name", "") or
            ""
        )

        # 조문 번호 추출
        article_no = (
            doc.get("article_no", "") or
            doc.get("article_number", "") or
            doc.get("article_no", "") or
            metadata.get("article_no", "") or
            metadata.get("article_number", "") or
            ""
        )

        # 항 번호 추출
        clause_no = (
            doc.get("clause_no", "") or
            doc.get("clause_number", "") or
            metadata.get("clause_no", "") or
            metadata.get("clause_number", "") or
            ""
        )

        # 호 번호 추출
        item_no = (
            doc.get("item_no", "") or
            doc.get("item_number", "") or
            metadata.get("item_no", "") or
            metadata.get("item_number", "") or
            ""
        )

        # 조문 제목 추출
        heading = (
            doc.get("heading", "") or
            doc.get("article_title", "") or
            metadata.get("heading", "") or
            metadata.get("article_title", "") or
            ""
        )

        # 판례 정보 추출
        court = (
            doc.get("court", "") or
            metadata.get("court", "") or
            ""
        )

        case_number = (
            doc.get("case_number", "") or
            doc.get("doc_id", "") or
            doc.get("case_id", "") or
            metadata.get("case_number", "") or
            metadata.get("doc_id", "") or
            ""
        )

        case_name = (
            doc.get("case_name", "") or
            doc.get("casenames", "") or
            metadata.get("case_name", "") or
            metadata.get("casenames", "") or
            ""
        )

        announce_date = (
            doc.get("announce_date", "") or
            doc.get("decision_date", "") or
            metadata.get("announce_date", "") or
            metadata.get("decision_date", "") or
            ""
        )

        case_type = (
            doc.get("case_type", "") or
            metadata.get("case_type", "") or
            ""
        )

        # 판례 본문 정보 추출
        case_summary = (
            doc.get("summary", "") or
            doc.get("case_summary", "") or
            metadata.get("summary", "") or
            metadata.get("case_summary", "") or
            ""
        )

        case_holding = (
            doc.get("holding", "") or
            doc.get("case_holding", "") or
            doc.get("판시사항", "") or
            metadata.get("holding", "") or
            metadata.get("case_holding", "") or
            ""
        )

        case_reasoning = (
            doc.get("reasoning", "") or
            doc.get("case_reasoning", "") or
            doc.get("판결요지", "") or
            metadata.get("reasoning", "") or
            metadata.get("case_reasoning", "") or
            ""
        )

        # 문서 타입 판단
        source_type = (
            doc.get("source_type", "") or
            metadata.get("source_type", "") or
            metadata.get("type", "") or
            ""
        )

        normalized = {
            "content": str(content).strip(),
            "source": str(source).strip() or "Unknown",
            "relevance_score": (
                float(doc.get("relevance_score", 0.0) or doc.get("final_weighted_score", 0.0) or
                      doc.get("score", 0.0) or doc.get("similarity_score", 0.0) or
                      doc.get("similarity", 0.0) or 0.0)
            ),
            "document_id": str(doc.get("document_id", "") or doc.get("id", "") or doc.get("chunk_id", "") or "").strip(),
            "metadata": metadata,
            # 법률 정보 추가
            "law_name": str(law_name).strip(),
            "article_no": str(article_no).strip(),
            "clause_no": str(clause_no).strip(),
            "item_no": str(item_no).strip(),
            "heading": str(heading).strip(),
            # 판례 정보 추가
            "court": str(court).strip(),
            "case_number": str(case_number).strip(),
            "case_name": str(case_name).strip(),
            "announce_date": str(announce_date).strip(),
            "case_type": str(case_type).strip(),
            # 판례 본문 정보 추가
            "case_summary": str(case_summary).strip(),
            "case_holding": str(case_holding).strip(),
            "case_reasoning": str(case_reasoning).strip(),
            "source_type": str(source_type).strip()
        }

        return normalized

    def _format_document_for_prompt(self, doc: Dict[str, Any], idx: int, is_high_priority: bool = False) -> str:
        """문서를 프롬프트 형식으로 포맷팅 (중복 제거 및 최적화)"""
        content = doc.get("content", "")
        source = doc.get("source", "Unknown")
        score = doc.get("relevance_score", 0.0)

        # 법률 정보 추출
        law_name = doc.get("law_name", "")
        article_no = doc.get("article_no", "")
        clause_no = doc.get("clause_no", "")
        item_no = doc.get("item_no", "")
        heading = doc.get("heading", "")

        # 판례 정보 추출
        court = doc.get("court", "")
        case_number = doc.get("case_number", "")
        case_name = doc.get("case_name", "")
        announce_date = doc.get("announce_date", "")
        case_type = doc.get("case_type", "")
        source_type = doc.get("source_type", "")

        # 판례 본문 정보 추출
        case_summary = doc.get("case_summary", "")
        case_holding = doc.get("case_holding", "")
        case_reasoning = doc.get("case_reasoning", "")

        # 법령/판례 근거 구성
        legal_reference = ""
        if law_name:
            legal_reference = law_name
            if article_no:
                if article_no.isdigit():
                    legal_reference += f" 제{article_no}조"
                else:
                    legal_reference += f" {article_no}"
                if clause_no:
                    legal_reference += f" 제{clause_no}항"
                if item_no:
                    legal_reference += f" 제{item_no}호"
            if heading:
                legal_reference += f" ({heading})"
        elif court and case_number:
            legal_reference = f"{court} {case_number}"
        elif case_number:
            legal_reference = case_number
        elif court and case_name:
            legal_reference = f"{court} {case_name}"
        elif case_name:
            legal_reference = case_name

        # 더 간결한 형식으로 변경
        if law_name and article_no:
            formatted = f"**문서 {idx}**: {law_name} 제{article_no}조"
            if heading:
                formatted += f" - {heading}"
            formatted += f" (관련도: {score:.2f})\n\n"
        elif legal_reference:
            formatted = f"**문서 {idx}**: {legal_reference} (관련도: {score:.2f})\n\n"
        else:
            formatted = f"**문서 {idx}**: {source} (관련도: {score:.2f})\n\n"

        # 법령 근거 표시
        if law_name and article_no:
            formatted += f"**법령 근거**: {legal_reference}\n\n"

        # 판례 정보 및 본문 표시
        elif court or case_number:
            formatted += f"**판례 근거**: {legal_reference}\n\n"
            if court:
                formatted += f"**법원**: {court}\n\n"
            if case_type:
                formatted += f"**사건 종류**: {case_type}\n\n"
            if announce_date:
                formatted += f"**선고일**: {announce_date}\n\n"
            # 판례 본문 정보
            if case_holding:
                formatted += f"**판시사항**: {case_holding[:300]}{'...' if len(case_holding) > 300 else ''}\n\n"
            if case_reasoning:
                formatted += f"**판결요지**: {case_reasoning[:300]}{'...' if len(case_reasoning) > 300 else ''}\n\n"
            if case_summary:
                formatted += f"**사건 개요**: {case_summary[:300]}{'...' if len(case_summary) > 300 else ''}\n\n"

        # 🔴 개선: 내용 길이 최적화 (2000자 → 800자, 1500자 → 500자)
        if is_high_priority:
            content_preview = content[:800] if len(content) > 800 else content
        else:
            content_preview = content[:500] if len(content) > 500 else content

        formatted += f"{content_preview}{'...' if len(content) > len(content_preview) else ''}\n\n"
        formatted += "---\n\n"

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
        # context 타입 검증 및 변환
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            logger.warning(f"⚠️ [STRUCTURE CONTEXT] context is not a dict (type: {type(context)}), converting to dict")
            if isinstance(context, str):
                context = {"context": context}
            else:
                context = {}
        
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
            context_text = context.get("context", "") if isinstance(context, dict) else ""
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
                        for idx, doc in enumerate(high_relevance[:5], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=True)
                            structured_parts.append(formatted_doc)

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        for idx, doc in enumerate(medium_relevance[:3], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=False)
                            structured_parts.append(formatted_doc)

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
                    for ref in legal_references[:5]:
                        if isinstance(ref, str):
                            structured_parts.append(f"- {ref}")
                        else:
                            structured_parts.append(f"- {ref.get('text', '') if isinstance(ref, dict) else str(ref)}")
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
                        for idx, doc in enumerate(high_relevance[:5], 1):
                            # 관련도 점수를 relevance_score로 변환 (score → relevance_score)
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=True)
                            structured_parts.append(formatted_doc)

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        for idx, doc in enumerate(medium_relevance[:3], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=False)
                            structured_parts.append(formatted_doc)

                    structured_parts.append("")

                if context_text:
                    structured_parts.append("## 조문 해설\n")
                    structured_parts.append(context_text)

            elif question_type == QuestionType.LEGAL_ADVICE:
                # 법령 + 판례 + 실무 조언 균형 배치
                if legal_references:
                    structured_parts.append("## 관련 법령\n")
                    for ref in legal_references[:3]:
                        if isinstance(ref, str):
                            structured_parts.append(f"- {ref}")
                        else:
                            structured_parts.append(f"- {ref.get('text', '') if isinstance(ref, dict) else str(ref)}")
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
                        for idx, doc in enumerate(high_relevance[:5], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=True)
                            structured_parts.append(formatted_doc)

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        for idx, doc in enumerate(medium_relevance[:3], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=False)
                            structured_parts.append(formatted_doc)

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
                        for idx, doc in enumerate(high_relevance[:5], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=True)
                            structured_parts.append(formatted_doc)

                    if medium_relevance:
                        structured_parts.append("### 🟡 중요 문서 (관련도 0.35~0.65)\n")
                        for idx, doc in enumerate(medium_relevance[:3], 1):
                            # 관련도 점수를 relevance_score로 변환
                            doc_for_format = doc.copy()
                            doc_for_format["relevance_score"] = doc.get("score", 0.0)
                            formatted_doc = self._format_document_for_prompt(doc_for_format, idx, is_high_priority=False)
                            structured_parts.append(formatted_doc)

                    structured_parts.append("")

                if legal_references:
                    structured_parts.append("## 관련 법령\n")
                    for ref in legal_references[:5]:
                        if isinstance(ref, str):
                            structured_parts.append(f"- {ref}")
                        else:
                            structured_parts.append(f"- {ref.get('text', '') if isinstance(ref, dict) else str(ref)}")
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
            # 개선 사항 3: Context Structure 문제 해결 - 문서 내용 검증 기준 개선
            # text 길이만으로 판단하지 않고, 실제 문서 내용이 포함되었는지 확인
            text_too_short = len(result_text) < 100  # 300 -> 100으로 추가 완화
            has_minimum_text = len(result_text) > 0
            
            # 문서 내용이 포함되었는지 더 정확하게 확인
            # result_text에 document_contents의 내용이 실제로 포함되어 있는지 검증
            doc_content_included = False
            if has_doc_content and document_contents:
                # 상위 5개 문서의 핵심 키워드가 result_text에 포함되어 있는지 확인 (3개 -> 5개로 증가)
                sample_docs = document_contents[:5]
                for doc in sample_docs:
                    doc_text = doc.get("content", "")[:200] if isinstance(doc, dict) else str(doc)[:200]
                    if doc_text:
                        # 정확한 일치 외에도 부분 일치도 확인 (개선 사항 3)
                        if doc_text in result_text or any(word in result_text for word in doc_text.split()[:5] if len(word) > 3):
                            doc_content_included = True
                            break
            
            # 문서 내용이 포함되지 않았거나 텍스트가 너무 짧으면 강제 추가
            # 개선 사항 5: 검증 기준 추가 완화 및 문서 내용 포함 로직 개선
            # 문서 내용이 포함되지 않았거나 텍스트가 너무 짧으면 강제 추가
            # 검증 기준을 더 완화하여 문서 내용이 포함되도록 개선
            should_force_add = (
                has_doc_content and 
                (not doc_content_included or (text_too_short and len(result_text) < 300))  # 200 -> 300으로 완화
            )
            
            if should_force_add:
                # 문서 내용이 추가되지 않은 경우 강제 추가
                logger.debug(  # warning -> debug로 변경 (너무 자주 발생하여)
                    f"⚠️ [CONTEXT STRUCTURE] Document contents not properly included in structured context. "
                    f"Force adding {min(8, len(document_contents))} documents. (text_len={len(result_text)}, has_keywords={has_doc_keywords}, doc_included={doc_content_included})"
                )
                doc_section = "\n## 검색된 법률 문서\n"
                doc_section += "다음은 질문에 대한 답변을 위해 검색된 관련 법률 문서입니다.\n\n"
                
                # 상위 8개 문서 추가 (관련도 순) - 5개 -> 8개로 증가
                sorted_docs = sorted(
                    document_contents,
                    key=lambda x: x.get("score", 0.0) if isinstance(x, dict) else 0.0,
                    reverse=True
                )
                
                for idx, doc in enumerate(sorted_docs[:8], 1):  # 5개 -> 8개로 증가
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

### 2. 구조화된 답변
- **상황 정리**: 사용자의 질문 내용을 요약 정리
- **관련 법률**: 적용 가능한 법률 및 조항 명시
- **법적 분석**: 쟁점과 법리 설명
- **실질적 조언**: 실행 가능한 대응 방안 제시
- **추가 고려사항**: 주의사항 및 참고사항

### 4. 접근성 있는 언어
- 전문 법률 용어는 쉬운 말로 풀어서 설명
- 필요시 예시를 들어 이해를 돕기
- 복잡한 개념은 단계별로 설명

### 5. 띄어쓰기 필수 준수
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

### 6. 윤리적 경계
- 명백히 불법적이거나 비윤리적인 행위에 대한 조력 거부
- 소송 사기, 증거 조작 등 불법 행위 관련 질문에는 답변 거부
- 범죄 행위 방법이나 법망 회피 방법은 절대 제공하지 않음

## 답변 프레임워크

### 일반 법률 질문
```
[질문 요약]
귀하의 질문은 ~에 관한 것으로 이해됩니다.

[관련 법률]
- 적용 법률:
- 주요 조항:

[법적 해설]
[구체적 내용 설명]

[실무적 조언]
이러한 경우 일반적으로 다음과 같은 방법을 고려할 수 있습니다:

[주의사항]
```

### 분쟁/소송 관련 질문
```
[상황 분석]
말씀하신 상황을 정리하면...

[법적 쟁점]
이 사안의 핵심 쟁점은...

[예상 법적 판단]
관련 법리와 판례에 따르면...

[권리 구제 방법]
1. 협상/조정
2. 민사소송
3. 형사고소
[각각의 장단점 설명]

[증거 자료]
다음과 같은 자료가 중요합니다:

[절차 안내]
구체적인 절차는...

[전문가 상담 권고]
본 사안은 [이유]로 인해 변호사 상담을 적극 권장합니다.
```

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

- 존댓말 사용 (격식체)
- 문단 구분 명확히
- 중요 내용은 **강조**
- 법조문 인용 시 정확한 출처 표시
- 3단계 이상 복잡한 내용은 번호 매기기 사용
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
- 사용자가 걱정하고 있다면 공감하고 안심시켜 주세요"""

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
- 법적 불확실성 명시"""

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

    def _get_answer_examples_section(self, question_type: QuestionType) -> str:
        """
        답변 형식 예시 섹션 생성
        
        Args:
            question_type: 질문 유형
            
        Returns:
            예시 섹션 문자열
        """
        examples = """
## 📚 답변 형식 예시 (반드시 참고하세요)

### ✅ 올바른 답변 예시

**질문**: "계약서 작성 시 주의사항은 무엇인가요?"

**올바른 답변**:
계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다.

**1. 일반 법적 원칙**
계약서는 당사자 간의 권리와 의무를 명확히 규정하는 문서입니다. 민법 제105조에 따르면, 계약은 당사자 간의 합의에 의해 성립됩니다. 따라서 계약서에는 계약의 목적, 당사자의 권리와 의무, 계약 조건 등이 명확히 기재되어야 합니다.

**2. 주요 주의사항**
- 계약 목적과 내용을 명확히 기재하세요
- 계약 조건을 구체적으로 명시하세요
- 계약 해지 사유와 절차를 명확히 규정하세요
- 손해배상 조항을 합리적으로 설정하세요

**3. 관련 법령**
민법 제105조에 따르면, 계약은 당사자 간의 합의에 의해 성립됩니다. 민법 제543조에 따르면, 계약 해지는 장래에 대해서만 효력을 발생합니다.

**4. 판례 및 해석례**
대법원 판결에 의하면, 계약서의 내용이 불명확한 경우에는 계약의 목적과 당사자의 의사를 종합적으로 고려하여 해석해야 합니다. [판례: 대법원 2020다12345]

**5. 실무 조언**
계약서 작성 시에는 변호사나 법률 전문가의 검토를 받는 것이 좋습니다. 특히 계약 금액이 크거나 중요한 계약의 경우에는 반드시 전문가의 자문을 구하세요.

---

### ❌ 잘못된 답변 예시

**질문**: "계약서 작성 시 주의사항은 무엇인가요?"

**잘못된 답변**:
[문서: 대전지방법원 대전지방법원-2014가단43882]
나아가 이 사건 각 계약서 작성 당시 피고 엘지 또는 피고 에스케이가 신청명의자인 원고 본인에 관한 인증이나 가입의사 확인절차를 거쳤다고 볼만한 사정이 없는 점...

**문제점**:
- 특정 사건의 내용을 그대로 복사함
- 일반적인 법적 원칙이 없음
- 질문에 대한 일반적인 답변이 아님

---

### ✅ 올바른 답변 형식 (특정 사건 활용 시)

**질문**: "계약서 작성 시 주의사항은 무엇인가요?"

**올바른 답변**:
계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다.

**1. 일반 법적 원칙**
계약서는 당사자 간의 권리와 의무를 명확히 규정하는 문서입니다. 민법 제105조에 따르면, 계약은 당사자 간의 합의에 의해 성립됩니다.

**2. 주요 주의사항**
- 계약 당사자의 신원 확인이 중요합니다
- 계약 조건을 명확히 기재해야 합니다

**3. 관련 법령**
민법 제105조, 민법 제543조

**4. 판례 및 해석례**
대법원 판결에 의하면, 계약서 작성 시 당사자의 신원 확인 절차를 거치지 않은 경우, 계약의 유효성에 문제가 발생할 수 있습니다. [판례: 대전지방법원 2014가단43882] (특정 사건의 법적 원칙만 추출하여 설명)

**차이점**:
- ✅ 일반 법적 원칙을 먼저 설명
- ✅ 특정 사건의 법적 원칙만 추출하여 설명
- ✅ 특정 사건의 세부사항(당사자명, 사건번호 등)은 생략
- ❌ 특정 사건의 내용을 그대로 복사하지 않음

---

**⚠️ 중요**: 
- 답변의 시작 부분에 반드시 일반적인 법적 원칙을 먼저 설명하세요
- 특정 사건의 내용을 그대로 복사하지 말고, 법적 원칙만 추출하여 설명하세요
- 특정 사건의 세부사항(당사자명, 사건번호 등)은 생략하세요

### 📋 답변 구조 가이드 (각 섹션 작성 방법)

**1. 일반 법적 원칙 섹션 작성 방법**:
- ✅ 올바른 예: "계약서 작성 시 주의해야 할 일반적인 법적 원칙은 다음과 같습니다. 계약서는 당사자 간의 권리와 의무를 명확히 규정하는 문서입니다. 민법 제105조에 따르면, 계약은 당사자 간의 합의에 의해 성립됩니다."
- ❌ 잘못된 예: "[문서: 대전지방법원-2014가단3882] 나아가 이 사건 각 계약서 작성 당시..."

**2. 주요 주의사항 섹션 작성 방법**:
- ✅ 올바른 예: "계약서 작성 시 주의해야 할 주요 사항은 다음과 같습니다:\n- 계약 목적과 내용을 명확히 기재하세요\n- 계약 조건을 구체적으로 명시하세요"
- ❌ 잘못된 예: "피고 엘지 또는 피고 에스케이가 신청명의자인 원고 본인에 관한 인증이나 가입의사 확인절차를 거쳤다고 볼만한 사정이 없는 점..."

**3. 관련 법령 섹션 작성 방법**:
- ✅ 올바른 예: "관련 법령은 다음과 같습니다:\n- 민법 제105조: 계약은 당사자 간의 합의에 의해 성립됩니다.\n- 민법 제543조: 계약 해지는 장래에 대해서만 효력을 발생합니다."
- ❌ 잘못된 예: "대법", "민사소송법", "지방세법" (조문번호 없음)

**4. 판례 및 해석례 섹션 작성 방법**:
- ✅ 올바른 예: "대법원 판결에 의하면, 계약서 작성 시 당사자의 신원 확인 절차를 거치지 않은 경우, 계약의 유효성에 문제가 발생할 수 있습니다. [판례: 대전지방법원 2014가단43882]"
- ❌ 잘못된 예: "[문서: 대전지방법원 대전지방법원-2014가단43882] 나아가 이 사건 각 계약서 작성 당시 피고 엘지 또는 피고 에스케이가..."

**5. 실무 조언 섹션 작성 방법**:
- ✅ 올바른 예: "실무적으로는 다음과 같은 조언을 드립니다:\n- 계약서 작성 시에는 변호사나 법률 전문가의 검토를 받는 것이 좋습니다.\n- 특히 계약 금액이 크거나 중요한 계약의 경우에는 반드시 전문가의 자문을 구하세요."
- ❌ 잘못된 예: 특정 사건의 판결문 내용을 그대로 복사
"""
        return examples

    def _get_legal_advice_template(self) -> str:
        """법률 자문 템플릿"""
        return """
## 법률 자문 지침

### 답변 작성 원칙
1. **일반적이고 포괄적인 답변 우선**: 특정 사건이나 판례에만 집중하지 말고, 질문의 일반적인 법적 원칙과 주의사항을 포괄적으로 다루세요.
2. **균형잡힌 접근**: 검색된 문서 중 특정 사건에만 집중하지 말고, 다양한 법령, 판례, 해석례를 종합하여 일반적인 조언을 제공하세요.
3. **구조화된 답변**: 사실관계 파악 → 법적 평가 → 실무 조언 순서로 작성하되, 각 단계에서 일반적인 원칙을 먼저 설명하고 구체적인 사례는 보완적으로 제시하세요.
4. **불확실 시 명시**: 불확실한 부분은 필요한 추가 정보와 리스크를 명시하세요.

### 답변 구성 (반드시 이 순서를 따르세요)
1. **일반 법적 원칙 (필수)**: 질문과 관련된 일반적인 법적 원칙과 규칙을 먼저 설명하세요. 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
2. **주요 주의사항**: 실무에서 주의해야 할 주요 사항들을 체계적으로 정리하세요.
3. **관련 법령**: 관련 법령 조문을 구체적으로 인용하되, 특정 사건에만 국한되지 않도록 일반적인 법적 원칙을 중심으로 설명하세요.
4. **판례 및 해석례**: 일반적인 법적 원칙을 뒷받침하는 판례와 해석례를 종합적으로 활용하되, 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
5. **실무 조언**: 구체적인 행동 방안이나 주의사항을 제시하세요.

**⚠️ 중요**: 
- **답변의 시작 부분에 반드시 일반적인 법적 원칙을 먼저 설명하세요.** 이것은 필수입니다.
- 특정 사건의 내용을 그대로 복사하지 마세요. 판례의 법적 원칙만 추출하여 일반적으로 설명하세요.
- 특정 사건번호(예: "2014가단3882"), 특정 당사자명(예: "피고 엘지", "원고 본인")을 답변에 포함하지 마세요.
- "이 사건 각 계약서 작성 당시" 같은 특정 사건의 사실관계 서술을 그대로 복사하지 마세요.
- 답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작해야 합니다.
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

    def _get_general_template(self) -> str:
        """개선 사항 9: 일반 도메인 템플릿"""
        return """당신은 법률 전문가입니다. 다음 질문에 대해 정확하고 전문적인 답변을 제공해주세요.

질문: {query}

답변 시 다음 사항을 고려해주세요:
1. 관련 법령과 판례를 정확히 인용
2. 실무적 관점에서 실행 가능한 조언 제공
3. 불확실한 부분은 명확히 표시
4. 전문가 상담 권유
5. 검색된 문서를 적극 활용하여 답변 구성

검색된 문서:
{document_context}

답변을 한국어로 작성해주세요."""

    def _get_general_question_template(self) -> str:
        """일반 질문 템플릿"""
        return """
## 일반 질문 답변 지침

### 답변 구조 (반드시 이 순서를 따르세요)
1. **질문 분석**: 사용자의 질문을 정확히 이해하고 핵심 이슈를 파악합니다.
2. **일반 법적 원칙 (필수, 먼저 설명)**: 질문과 관련된 일반적인 법적 원칙과 규칙을 먼저 설명합니다. 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
3. **주요 주의사항**: 실무에서 주의해야 할 주요 사항들을 체계적으로 정리합니다.
4. **관련 법령**: 질문과 관련된 법령 조문을 구체적으로 인용합니다 (예: "민법 제750조에 따르면..."). 특정 사건에만 국한되지 않도록 일반적인 법적 원칙을 중심으로 설명하세요.
5. **법적 해설**: 법령의 내용을 쉽고 명확하게 설명하되, 특정 사건에만 국한되지 않도록 일반적인 원칙을 중심으로 설명합니다.
6. **판례 및 해석례**: 일반적인 법적 원칙을 뒷받침하는 판례와 해석례를 종합적으로 활용하되, 특정 사건보다는 일반적인 법적 원칙을 중심으로 설명하세요.
7. **실무 조언**: 구체적인 행동 방안이나 주의사항을 제시합니다.

**⚠️ 중요**: 
- **답변의 시작 부분에 반드시 일반적인 법적 원칙을 먼저 설명하세요.** 이것은 필수입니다.
- 특정 사건의 내용을 그대로 복사하지 마세요. 판례의 법적 원칙만 추출하여 일반적으로 설명하세요.
- 특정 사건번호(예: "2014가단3882"), 특정 당사자명(예: "피고 엘지", "원고 본인")을 답변에 포함하지 마세요.
- "이 사건 각 계약서 작성 당시" 같은 특정 사건의 사실관계 서술을 그대로 복사하지 마세요.
- 답변의 첫 문장은 반드시 일반적인 법적 원칙으로 시작해야 합니다.

### 답변 원칙
- **일반적이고 포괄적인 답변 우선**: 특정 사건이나 판례에만 집중하지 말고, 질문의 일반적인 법적 원칙과 주의사항을 포괄적으로 다루세요.
- **균형잡힌 접근**: 검색된 문서 중 특정 사건에만 집중하지 말고, 다양한 법령, 판례, 해석례를 종합하여 일반적인 조언을 제공하세요.
- 핵심 답변을 간결하고 명확하게 제시합니다.
- 관련 법령/판례를 구체적으로 인용합니다 (법령명과 조문번호 필수).
- 법률 용어를 사용할 때는 쉬운 설명을 함께 제공합니다.
- 실무적으로 도움이 되는 구체적인 조언을 포함합니다.
- 검색된 문서가 있으면 이를 바탕으로 답변을 작성하되, 특정 사건보다는 일반적인 법적 원칙을 중심으로 구성하세요.
"""
