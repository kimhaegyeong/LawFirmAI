# -*- coding: utf-8 -*-
"""
프롬프트 체인 빌더 모듈
리팩토링: legal_workflow_enhanced.py에서 프롬프트 체인 빌더 함수 분리
"""

from typing import Any, Dict, Optional


class DirectAnswerChainBuilder:
    """직접 답변 생성 체인 빌더"""

    @staticmethod
    def build_query_type_analysis_prompt(query: str) -> str:
        """질문 유형 분석 프롬프트 생성"""
        return f"""다음 질문의 유형을 분석해주세요.

질문: {query}

다음 유형 중 하나를 선택하세요:
- greeting (인사말): "안녕하세요", "고마워요", "감사합니다" 등
- term_definition (용어 정의): 법률 용어나 개념의 정의를 묻는 질문
- simple_question (간단한 질문): 일반 법률 상식으로 답변 가능한 간단한 질문

다음 형식으로 응답해주세요:
{{
    "query_type": "greeting" | "term_definition" | "simple_question",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

    @staticmethod
    def build_prompt_generation_prompt(query: str, query_type: str) -> str:
        """적절한 프롬프트 생성"""
        if query_type == "greeting":
            return f"""사용자의 인사에 친절하게 응답하세요:

{query}

간단하고 친절하게 응답해주세요. (1-2문장)"""
        elif query_type == "term_definition":
            return f"""다음 법률 용어에 대해 간단명료하게 정의를 제공하세요:

용어: {query}

다음 형식을 따라주세요:
1. 용어의 정의 (1-2문장)
2. 간단한 설명 (1문장)
총 2-3문장으로 간결하게 작성해주세요."""
        else:
            # simple_question
            return f"""다음 법률 질문에 간단명료하게 답하세요:

질문: {query}

법률 용어나 개념에 대한 정의나 간단한 설명을 제공하세요. 검색 없이 일반적인 법률 지식으로 답변하세요. (2-4문장)"""

    @staticmethod
    def build_initial_answer_prompt(prev_output: Any) -> str:
        """초기 답변 생성 프롬프트"""
        if isinstance(prev_output, str):
            return prev_output
        elif isinstance(prev_output, dict):
            return prev_output.get("prompt", "")
        return ""

    @staticmethod
    def build_quality_validation_prompt(query: str, answer: str) -> str:
        """답변 품질 검증 프롬프트"""
        return f"""다음 답변의 품질을 검증해주세요.

질문: {query}
답변: {answer[:500]}

다음 기준으로 검증하세요:
1. **적절한 길이**: 너무 짧지도 길지도 않음 (10-500자)
2. **질문에 대한 직접적인 답변**: 질문에 맞는 답변인가?
3. **명확성**: 답변이 명확하고 이해하기 쉬운가?
4. **완성도**: 답변이 완전한가?

다음 형식으로 응답해주세요:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "needs_improvement": true | false
}}
"""

    @staticmethod
    def build_answer_improvement_prompt(query: str, original_answer: str, issues: list) -> str:
        """답변 개선 프롬프트"""
        return f"""다음 답변을 개선해주세요.

질문: {query}
원본 답변: {original_answer}
문제점: {', '.join(issues) if issues else "없음"}

다음 문제점을 해결하여 개선된 답변을 작성해주세요:
{chr(10).join([f"- {issue}" for issue in issues[:3]]) if issues else "없음"}
"""


class ClassificationChainBuilder:
    """질문 분류 체인 빌더"""

    @staticmethod
    def build_question_type_prompt(query: str) -> str:
        """질문 유형 분류 프롬프트"""
        return f"""다음 법률 질문을 질문 유형으로 분류해주세요.

질문: {query}

분류 가능한 유형:
1. precedent_search - 판례, 사건, 법원 판결, 판시사항 관련
2. law_inquiry - 법률 조문, 법령, 규정의 내용을 묻는 질문
3. legal_advice - 법률 조언, 해석, 권리 구제 방법을 묻는 질문
4. procedure_guide - 법적 절차, 소송 방법, 대응 방법을 묻는 질문
5. term_explanation - 법률 용어의 정의나 의미를 묻는 질문
6. general_question - 범용적인 법률 질문

다음 형식으로 응답해주세요:
{{
    "query_type": "precedent_search" | "law_inquiry" | "legal_advice" | "procedure_guide" | "term_explanation" | "general_question",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거"
}}
"""

    @staticmethod
    def build_legal_field_prompt(query: str, query_type: str) -> str:
        """법률 분야 추출 프롬프트"""
        return f"""다음 질문에서 관련 법률 분야를 추출해주세요.

질문: {query}
질문 유형: {query_type}

가능한 법률 분야:
- civil (민사법): 계약, 손해배상, 채권채무 등
- criminal (형사법): 형사범죄, 처벌, 형량 등
- administrative (행정법): 행정처분, 행정소송 등
- intellectual_property (지적재산권법): 특허, 상표, 저작권 등

다음 형식으로 응답해주세요:
{{
    "legal_field": "civil" | "criminal" | "administrative" | "intellectual_property",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거"
}}
"""

    @staticmethod
    def build_complexity_prompt(query: str, query_type: str, legal_field: str) -> str:
        """복잡도 평가 프롬프트"""
        return f"""다음 질문의 복잡도를 평가해주세요.

질문: {query}
질문 유형: {query_type}
법률 분야: {legal_field}

복잡도 기준:
- simple (단순): 간단한 용어 정의나 일반 법률 상식 질문
- moderate (보통): 일반적인 법률 질문, 검색이 필요한 경우
- complex (복잡): 여러 법률 조항이나 판례 비교, 복잡한 사례 분석 등

다음 형식으로 응답해주세요:
{{
    "complexity": "simple" | "moderate" | "complex",
    "needs_search": true | false,
    "reasoning": "판단 근거"
}}
"""

    @staticmethod
    def build_search_necessity_prompt(query: str, query_type: str, complexity: str) -> str:
        """검색 필요성 평가 프롬프트"""
        return f"""다음 질문이 검색이 필요한지 평가해주세요.

질문: {query}
질문 유형: {query_type}
복잡도: {complexity}

검색이 필요한 경우:
- 판례나 법령 조문 인용이 필요한 경우
- 최신 법률 정보가 필요한 경우
- 구체적인 법률 사례나 판례가 필요한 경우

검색이 불필요한 경우:
- 간단한 법률 용어 정의
- 일반적인 법률 상식
- 복잡도가 simple인 경우

다음 형식으로 응답해주세요:
{{
    "needs_search": true | false,
    "search_type": "semantic" | "keyword" | "hybrid",
    "reasoning": "판단 근거"
}}
"""


class QueryEnhancementChainBuilder:
    """쿼리 강화 체인 빌더"""

    @staticmethod
    def build_query_analysis_prompt(query: str, query_type: str, legal_field: str) -> str:
        """쿼리 분석 프롬프트"""
        return f"""다음 법률 검색 쿼리를 분석하고 핵심 키워드를 추출해주세요.

원본 쿼리: {query}
질문 유형: {query_type}
법률 분야: {legal_field}

다음 형식으로 응답해주세요:
{{
    "core_keywords": ["키워드1", "키워드2", "키워드3"],
    "query_intent": "검색 의도 설명",
    "key_concepts": ["핵심 개념1", "핵심 개념2"]
}}
"""

    @staticmethod
    def build_keyword_expansion_prompt(query: str, query_analysis: Dict[str, Any]) -> str:
        """키워드 확장 프롬프트"""
        core_keywords = query_analysis.get("core_keywords", [])
        key_concepts = query_analysis.get("key_concepts", [])

        return f"""다음 쿼리의 키워드를 확장하고 변형을 생성해주세요.

원본 쿼리: {query}
핵심 키워드: {', '.join(core_keywords)}
핵심 개념: {', '.join(key_concepts)}

다음 형식으로 응답해주세요:
{{
    "expanded_keywords": ["확장된 키워드1", "확장된 키워드2"],
    "keyword_variants": ["변형 키워드1", "변형 키워드2"],
    "synonyms": ["동의어1", "동의어2"]
}}
"""

    @staticmethod
    def build_query_optimization_prompt(
        query: str,
        query_analysis: Dict[str, Any],
        keyword_expansion: Dict[str, Any]
    ) -> str:
        """쿼리 최적화 프롬프트"""
        expanded_keywords = keyword_expansion.get("expanded_keywords", [])
        keyword_variants = keyword_expansion.get("keyword_variants", [])

        return f"""다음 쿼리를 법률 검색에 최적화해주세요.

원본 쿼리: {query}
확장된 키워드: {', '.join(expanded_keywords)}
변형 키워드: {', '.join(keyword_variants)}

다음 형식으로 응답해주세요:
{{
    "optimized_query": "최적화된 검색 쿼리",
    "semantic_query": "의미적 검색용 쿼리",
    "keyword_queries": ["키워드 검색용 쿼리1", "키워드 검색용 쿼리2"],
    "reasoning": "최적화 사유"
}}
"""

    @staticmethod
    def build_query_validation_prompt(query: str, optimized_query: Dict[str, Any]) -> str:
        """쿼리 검증 프롬프트"""
        return f"""다음 최적화된 쿼리를 검증해주세요.

원본 쿼리: {query}
최적화된 쿼리: {optimized_query.get('optimized_query', '')}

다음 형식으로 응답해주세요:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "recommendations": ["권고사항1", "권고사항2"]
}}
"""


class AnswerGenerationChainBuilder:
    """답변 생성 체인 빌더"""

    @staticmethod
    def build_initial_answer_prompt(optimized_prompt: str) -> str:
        """초기 답변 생성 프롬프트"""
        return optimized_prompt

    @staticmethod
    def build_validation_prompt(answer: str) -> str:
        """답변 검증 프롬프트"""
        return f"""다음 기준으로 답변을 검증하세요:

1. **길이**: 최소 50자 이상
2. **내용 완성도**: 질문에 대한 직접적인 답변 포함
3. **법적 근거**: 관련 법령, 조항, 판례 인용 여부
4. **구조**: 명확한 섹션과 논리적 흐름
5. **일관성**: 답변 전체의 논리적 일관성

답변:
{answer[:2000]}

다음 형식으로 검증 결과를 제공하세요:
{{
    "is_valid": true/false,
    "quality_score": 0.0-1.0,
    "issues": [
        "문제점 1",
        "문제점 2"
    ],
    "strengths": [
        "강점 1",
        "강점 2"
    ],
    "recommendations": [
        "개선 권고 1",
        "개선 권고 2"
    ]
}}
"""

    @staticmethod
    def build_improvement_instructions_prompt(
        original_answer: str,
        validation_result: Dict[str, Any]
    ) -> str:
        """개선 지시 생성 프롬프트"""
        issues = validation_result.get("issues", [])
        recommendations = validation_result.get("recommendations", [])
        quality_score = validation_result.get("quality_score", 1.0)

        return f"""다음 답변의 검증 결과를 바탕으로 개선 지시를 작성하세요.

**원본 답변**:
{original_answer[:1500]}

**검증 결과**:
- 품질 점수: {quality_score:.2f}/1.0
- 문제점: {', '.join(issues[:5]) if issues else '없음'}
- 권고사항: {', '.join(recommendations[:5]) if recommendations else '없음'}

**개선 지시 작성 요청**:
위 검증 결과를 바탕으로 답변을 개선하기 위한 구체적인 지시사항을 작성하세요.

다음 형식으로 제공하세요:
{{
    "needs_improvement": true,
    "improvement_instructions": [
        "개선 지시 1: 구체적으로 어떤 부분을 어떻게 개선할지",
        "개선 지시 2: ..."
    ],
    "preserve_content": [
        "보존할 내용 1",
        "보존할 내용 2"
    ],
    "focus_areas": [
        "중점 개선 영역 1",
        "중점 개선 영역 2"
    ]
}}
"""

    @staticmethod
    def build_improved_answer_prompt(
        original_prompt: str,
        improvement_instructions: Dict[str, Any]
    ) -> str:
        """개선된 답변 생성 프롬프트"""
        improvement_text = "\n".join(improvement_instructions.get("improvement_instructions", []))
        preserve_content = "\n".join(improvement_instructions.get("preserve_content", []))

        return f"""{original_prompt}

---

## 🔧 개선 요청

위 프롬프트로 생성한 답변을 다음 지시사항에 따라 개선하세요:

**개선 지시사항**:
{improvement_text}

**보존할 내용** (반드시 포함):
{preserve_content if preserve_content else "원본 답변의 모든 법적 정보와 근거"}

**중점 개선 영역**:
{', '.join(improvement_instructions.get("focus_areas", []))}

위 지시사항에 따라 답변을 개선하되, 원본 답변의 법적 근거와 정보는 반드시 보존하세요.
"""

    @staticmethod
    def build_final_validation_prompt(answer: str) -> str:
        """최종 검증 프롬프트"""
        return f"""다음 답변을 최종 검증하세요.

답변:
{answer[:2000]}

다음 기준으로 최종 검증하세요:
1. **완성도**: 답변이 완전한가?
2. **정확성**: 법적 정보가 정확한가?
3. **명확성**: 답변이 명확한가?
4. **구조**: 논리적 구조가 있는가?

다음 형식으로 응답하세요:
{{
    "is_valid": true/false,
    "final_score": 0.0-1.0,
    "ready_for_user": true/false
}}
"""


class DocumentAnalysisChainBuilder:
    """문서 분석 체인 빌더"""

    @staticmethod
    def build_document_type_verification_prompt(text: str, detected_type: str) -> str:
        """문서 유형 확인 프롬프트"""
        return f"""다음 문서의 유형을 확인하고 검증해주세요.

문서 내용 (일부):
{text[:2000]}

키워드 기반 감지 결과: {detected_type}

다음 문서 유형 중 하나로 확인해주세요:
- contract (계약서): 계약서, 갑/을, 계약 조건 등
- complaint (고소장): 고소장, 피고소인, 고소인 등
- agreement (합의서): 합의서, 합의, 쌍방 합의 등
- power_of_attorney (위임장): 위임장, 위임인, 수임인 등
- general_legal_document (일반 법률 문서): 위에 해당하지 않는 경우

다음 형식으로 응답해주세요:
{{
    "document_type": "contract" | "complaint" | "agreement" | "power_of_attorney" | "general_legal_document",
    "confidence": 0.0-1.0,
    "reasoning": "판단 근거 (한국어)"
}}
"""

    @staticmethod
    def build_clause_extraction_prompt(text: str, document_type: str) -> str:
        """주요 조항 추출 프롬프트"""
        return f"""다음 문서에서 주요 조항을 추출해주세요.

문서 내용:
{text[:3000]}

문서 유형: {document_type}

다음 형식으로 응답해주세요:
{{
    "key_clauses": [
        {{
            "clause_number": "조항 번호",
            "title": "조항 제목",
            "content": "조항 내용",
            "importance": "high" | "medium" | "low"
        }}
    ],
    "total_clauses": 숫자
}}
"""

    @staticmethod
    def build_issue_identification_prompt(
        text: str,
        document_type: str,
        key_clauses: list
    ) -> str:
        """문제점 식별 프롬프트"""
        clauses_summary = "\n".join([
            f"- {c.get('title', 'N/A')}: {c.get('content', '')[:100]}"
            for c in key_clauses[:5]
        ])

        return f"""다음 문서에서 잠재적 문제점을 식별해주세요.

문서 내용 (일부):
{text[:2000]}

문서 유형: {document_type}

주요 조항:
{clauses_summary}

다음 형식으로 응답해주세요:
{{
    "issues": [
        {{
            "severity": "high" | "medium" | "low",
            "type": "missing_clause" | "vague_term" | "unclear_provision" | "potential_risk",
            "description": "문제점 설명",
            "location": "조항 번호 또는 위치",
            "recommendation": "개선 권고"
        }}
    ],
    "total_issues": 숫자
}}
"""

    @staticmethod
    def build_summary_generation_prompt(
        text: str,
        document_type: str,
        key_clauses: list,
        issues: list
    ) -> str:
        """요약 생성 프롬프트"""
        return f"""다음 문서를 요약해주세요.

문서 유형: {document_type}
주요 조항 수: {len(key_clauses)}
발견된 문제점 수: {len(issues)}

다음 형식으로 응답해주세요:
{{
    "summary": "문서 전체 요약 (3-5문장)",
    "key_points": ["핵심 포인트1", "핵심 포인트2", "핵심 포인트3"],
    "main_clauses": ["주요 조항 요약1", "주요 조항 요약2"],
    "critical_issues": ["중요한 문제점1", "중요한 문제점2"]
}}
"""

    @staticmethod
    def build_improvement_recommendations_prompt(
        document_type: str,
        issues: list
    ) -> str:
        """개선 권고 생성 프롬프트"""
        issues_summary = "\n".join([
            f"- [{i.get('severity', 'unknown')}] {i.get('description', 'N/A')}: {i.get('recommendation', 'N/A')}"
            for i in issues[:5]
        ])

        return f"""다음 문서의 문제점을 바탕으로 개선 권고를 작성해주세요.

문서 유형: {document_type}

발견된 문제점:
{issues_summary}

다음 형식으로 응답해주세요:
{{
    "recommendations": [
        {{
            "priority": "high" | "medium" | "low",
            "description": "개선 권고 설명",
            "action_items": ["구체적 행동 1", "구체적 행동 2"]
        }}
    ],
    "overall_assessment": "전체 평가"
}}
"""
