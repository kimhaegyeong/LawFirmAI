# -*- coding: utf-8 -*-
"""
프롬프트 빌더 모듈
리팩토링: legal_workflow_enhanced.py에서 프롬프트 빌더 메서드 분리
"""

from typing import Any, Dict, List, Optional


class QueryBuilder:
    """쿼리 관련 프롬프트 빌더"""

    @staticmethod
    def build_semantic_query(query: str, expanded_terms: List[str]) -> str:
        """의미적 검색용 쿼리 생성"""
        # 핵심 키워드 3-5개 선택
        key_terms = expanded_terms[:5] if expanded_terms else []
        if key_terms:
            return f"{query} {' '.join(key_terms)}"
        return query

    @staticmethod
    def build_keyword_queries(
        query: str,
        expanded_terms: List[str],
        query_type: str
    ) -> List[str]:
        """키워드 검색용 쿼리 리스트 생성"""
        queries = []

        # 원본 쿼리
        queries.append(query)

        # 질문 유형별 특화 쿼리
        if query_type == "precedent_search":
            # 판례 검색: "판례", "사건", "대법원" 등 추가
            queries.append(f"{query} 판례")
            queries.append(f"{query} 사건")
        elif query_type == "law_inquiry":
            # 법령 조문 검색: "법률", "조항", "조문" 등 추가
            queries.append(f"{query} 법률 조항")
            queries.append(f"{query} 법령")
        elif query_type == "legal_advice":
            # 법률 조언: "조언", "해석", "권리" 등 추가
            queries.append(f"{query} 조언")
            queries.append(f"{query} 해석")

        # 확장된 키워드 조합 (최대 3개)
        if expanded_terms and len(expanded_terms) >= 3:
            queries.append(" ".join(expanded_terms[:3]))

        return queries[:5]  # 최대 5개 쿼리

    @staticmethod
    def build_conversation_context_dict(context) -> Optional[Dict[str, Any]]:
        """ConversationContext를 딕셔너리로 변환"""
        try:
            if not context:
                return None

            return {
                "session_id": context.session_id if hasattr(context, 'session_id') else "",
                "turn_count": len(context.turns) if hasattr(context, 'turns') else 0,
                "entities": {
                    entity_type: list(entity_set)
                    for entity_type, entity_set in (context.entities or {}).items()
                } if hasattr(context, 'entities') and context.entities else {},
                "topic_stack": list(context.topic_stack) if hasattr(context, 'topic_stack') else [],
                "recent_topics": list(context.topic_stack[-3:]) if hasattr(context, 'topic_stack') and context.topic_stack else []
            }
        except Exception as e:
            return None


class PromptBuilder:
    """일반 프롬프트 빌더"""

    @staticmethod
    def build_query_enhancement_prompt_base(
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str,
        format_field_info: callable,
        format_query_guide: callable
    ) -> str:
        """
        쿼리 강화를 위한 LLM 프롬프트 생성 (기본 버전)

        Args:
            query: 원본 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드 목록
            legal_field: 법률 분야
            format_field_info: 법률 분야 정보 포맷 함수
            format_query_guide: 질문 유형별 가이드 포맷 함수

        Returns:
            프롬프트 문자열
        """
        # 입력 데이터 검증 및 정규화
        if not query or not isinstance(query, str):
            raise ValueError("query must be a non-empty string")

        if not query_type or not isinstance(query_type, str):
            query_type = "general_question"

        if not isinstance(extracted_keywords, list):
            extracted_keywords = []

        if not isinstance(legal_field, str):
            legal_field = ""

        legal_field_text = legal_field.strip() if legal_field else "미지정"

        # 키워드 상세 정보 구성
        keywords_info = ""
        if extracted_keywords and len(extracted_keywords) > 0:
            keywords_list = ", ".join(extracted_keywords[:10])
            keywords_info = f"""
### 추출된 키워드
{keywords_list}
**총 {len(extracted_keywords)}개**"""
        else:
            keywords_info = """
### 추출된 키워드
(없음)"""

        # 질문 유형별 가이드 정보 가져오기 (format_query_guide 사용)
        query_guide = format_query_guide(query_type) if format_query_guide else {}
        field_info = format_field_info(legal_field) if format_field_info else {}

        # 데이터베이스 구조 정보
        database_info = """
## 📊 검색 대상 데이터베이스 구조

### 주요 테이블 및 필드

**법령 데이터 (statutes, statute_articles)**
- 법령명 (statutes.name), 약칭 (statutes.abbrv)
- 조문번호 (statute_articles.article_no), 조항 번호 (clause_no, item_no)
- 조문 내용 (statute_articles.text), 제목 (statute_articles.heading)
- 시행일 (statutes.effective_date), 공포일 (statutes.proclamation_date)

**판례 데이터 (cases, case_paragraphs)**
- 사건번호 (cases.case_number, 형식: YYYY다/나XXXXX)
- 법원명 (cases.court: 대법원, 고등법원, 지방법원 등)
- 사건명 (cases.casenames)
- 선고일 (cases.announce_date)
- 판례 본문 (case_paragraphs.text)

**심결례 데이터 (decisions, decision_paragraphs)**
- 기관 (decisions.org)
- 문서 ID (decisions.doc_id)
- 결정일 (decisions.decision_date)
- 심결 내용 (decision_paragraphs.text)

**유권해석 데이터 (interpretations, interpretation_paragraphs)**
- 기관 (interpretations.org)
- 문서 ID (interpretations.doc_id)
- 제목 (interpretations.title)
- 응답일 (interpretations.response_date)
- 해석 내용 (interpretation_paragraphs.text)

### 검색 방식
- **벡터 검색**: 의미 기반 유사도 검색 (법률 조문, 판례 본문 전체 텍스트)
- **키워드 검색**: FTS5 기반 키워드 매칭 (법령명, 조문번호, 사건번호 등)
- **하이브리드 검색**: 벡터 + 키워드 결과 병합 및 재랭킹
"""

        prompt = f"""당신은 법률 검색 쿼리 최적화 전문가입니다. 주어진 검색 쿼리를 법률 데이터베이스 검색에 최적화하도록 개선해주세요.

## 🎯 작업 목표

주어진 질문에 대해 다음을 수행하세요:
1. **검색 정확도 향상**: 법률 데이터베이스에서 관련 문서를 더 정확하게 찾을 수 있도록 키워드 최적화
2. **검색 범위 확장**: 동의어, 관련 용어, 상위 개념을 추가하여 검색 누락 방지
3. **검색 효율성 증대**: 벡터 검색과 키워드 검색 모두에 효과적인 쿼리 생성
4. **법률 전문성 반영**: 법률 분야 특성과 질문 유형에 맞는 전문 용어 활용

{database_info}

## 📋 입력 정보 (상세)

### 기본 정보
**원본 쿼리**: "{query}"
**질문 유형**: {query_type} ({query_guide.get('description', '일반 검색')})
**법률 분야**: {legal_field_text}

{keywords_info}

### 질문 유형별 검색 전략
**현재 질문 유형**: {query_guide.get('description', '일반 검색')}

**검색 초점**: {query_guide.get('search_focus', '관련 법령, 판례, 법률 용어')}

**검색 전략**: {query_guide.get('search_strategy', '핵심 키워드 중심 검색')}

**데이터베이스 필드**: {query_guide.get('database_fields', '전체 데이터베이스')}

**추천 키워드**: {', '.join(query_guide.get('keyword_suggestions', [])[:8])}

### 법률 분야별 정보
{format_field_info(legal_field) if format_field_info else '없음'}

## 🔍 쿼리 최적화 지침

### 1. 의미 보존
- 원본 쿼리의 핵심 의도와 목적을 반드시 유지하세요
- 사용자가 찾고자 하는 법률 정보의 본질을 파악하세요

### 2. 법률 용어 확장
- **동의어 추가**: 법률 용어의 다양한 표현 추가 (예: "계약" → "계약서", "계약관계")
- **상위/하위 개념**: 일반 개념과 구체적 개념 모두 포함 (예: "손해배상" → "불법행위 손해배상", "계약 위반 손해배상")
- **법률 용어 정규화**: 법률에서 사용하는 공식 용어로 변환 (예: "이혼" → "혼인해소")

### 3. 검색 최적화
- **벡터 검색 최적화**: 의미적으로 유사한 문서를 찾기 위한 핵심 개념 키워드 포함
- **키워드 검색 최적화**: 법령명, 조문번호, 사건번호 등 정확한 매칭 가능한 용어 포함
- **하이브리드 검색**: 두 검색 방식 모두에 효과적인 균형 잡힌 쿼리 생성

### 4. 질문 유형별 특화
- **판례 검색**: 사건번호 패턴, 법원명, 판시사항 관련 키워드 추가
- **법령 조회**: 법령명, 조문번호, 조항의 핵심 법리 용어 포함
- **법률 조언**: 문제 상황의 핵심 법률 개념 + 관련 조문 + 유사 판례 패턴 조합

### 5. 간결성 유지
- 핵심 키워드는 반드시 유지
- 검색에 불필요한 수식어나 중복 표현 제거
- 최대 50자 이내로 간결하게 유지

## 📤 출력 형식

다음 JSON 형식으로 응답하세요 (설명 없이 JSON만 출력):

```json
{{
    "optimized_query": "최적화된 검색 쿼리 (50자 이내)",
    "expanded_keywords": ["키워드1", "키워드2", "키워드3", ...],
    "keyword_variants": ["변형 쿼리1", "변형 쿼리2", ...],
    "legal_terms": ["법률 용어1", "법률 용어2", ...],
    "reasoning": "최적화 사유 및 검색 전략 설명 (한국어)"
}}
```

## ⚠️ 주의사항

1. **원본 쿼리 의미 보존**: 최적화 과정에서 사용자의 원래 의도를 왜곡하지 마세요
2. **법률 전문성**: 법률 용어를 정확하게 사용하고, 법률 데이터베이스 구조를 고려하세요
3. **검색 효율성**: 벡터 검색과 키워드 검색 모두에 효과적인 쿼리를 생성하세요
4. **간결성**: 불필요한 단어를 제거하고 핵심 키워드만 포함하세요
"""

        return prompt
