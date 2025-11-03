# -*- coding: utf-8 -*-
"""
쿼리 강화 모듈
검색 쿼리를 최적화하고 강화하는 로직을 독립 모듈로 분리
"""

import logging
import re
from typing import Any, Dict, List, Optional

from langgraph_core.data.extractors import DocumentExtractor
from langgraph_core.models.prompt_builders import QueryBuilder
from langgraph_core.utils.prompt_chain_executor import PromptChainExecutor
from langgraph_core.data.response_parsers import QueryParser
from langgraph_core.utils.state_definitions import LegalWorkflowState
from langgraph_core.utils.workflow_constants import WorkflowConstants
from langgraph_core.utils.workflow_utils import WorkflowUtils


class QueryEnhancer:
    """
    쿼리 강화 클래스

    검색 쿼리를 최적화하고 강화하여 검색 정확도와 효율성을 향상시킵니다.
    """

    def __init__(
        self,
        llm: Any,
        llm_fast: Optional[Any],
        term_integrator: Any,
        config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        QueryEnhancer 초기화

        Args:
            llm: LLM 인스턴스
            llm_fast: 빠른 LLM 인스턴스 (선택)
            term_integrator: 법률 용어 통합기
            config: 설정 객체
            logger: 로거 (없으면 자동 생성)
        """
        self.llm = llm
        self.llm_fast = llm_fast
        self.term_integrator = term_integrator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # 쿼리 강화 캐시
        self._query_enhancement_cache: Dict[str, Dict[str, Any]] = {}

    def optimize_search_query(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Dict[str, Any]:
        """
        검색 쿼리 최적화 (LLM 강화 포함, 폴백 지원)

        Returns:
            {
                "semantic_query": "의미적 검색용 쿼리",
                "keyword_queries": ["키워드 query 1", ...],
                "expanded_keywords": ["확장된 키워드", ...],
                "llm_enhanced": bool  # LLM 강화 사용 여부
            }
        """
        # LLM 쿼리 강화 시도 (실패해도 계속 진행)
        llm_enhanced = None
        try:
            llm_enhanced = self.enhance_query_with_llm(
                query=query,
                query_type=query_type,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field
            )
        except Exception as e:
            self.logger.debug(f"LLM query enhancement skipped: {e}")

        # LLM 강화 결과 사용 또는 원본 사용
        if llm_enhanced and isinstance(llm_enhanced, dict):
            base_query = llm_enhanced.get("optimized_query", query)
            llm_keywords = llm_enhanced.get("expanded_keywords", [])
            llm_variants = llm_enhanced.get("keyword_variants", [])
            llm_used = True
        else:
            # LLM 실패 시 폴백 강화: 기본 쿼리 정제 및 키워드 확장
            base_query = self.clean_query_for_fallback(query)
            llm_keywords = []
            llm_variants = []
            llm_used = False
            self.logger.info(f"LLM enhancement failed, using enhanced fallback query: '{base_query[:50]}...'")

        # 1. 법률 용어 정규화 및 확장 (LLM 강화 쿼리 사용)
        normalized_terms = self.normalize_legal_terms(base_query, extracted_keywords)

        # 2. 동의어 및 관련 용어 확장 (LLM 실패 시에도 강화)
        expanded_terms = self.expand_legal_terms(normalized_terms, legal_field)

        # LLM 실패 시 추가 키워드 확장 시도
        if not llm_used and extracted_keywords:
            # extracted_keywords에서 핵심 키워드 선택
            core_keywords = [kw for kw in extracted_keywords[:5] if isinstance(kw, str) and len(kw) >= 2]
            expanded_terms.extend(core_keywords)
            expanded_terms = list(set(expanded_terms))[:15]  # 최대 15개로 제한

        # LLM 키워드 병합
        if llm_keywords:
            expanded_terms = list(set(expanded_terms + llm_keywords))

        # 3. 의미적 쿼리 생성 (LLM 강화 쿼리 우선 사용)
        semantic_query = self.build_semantic_query(base_query, expanded_terms)

        # semantic_query 검증 및 수정
        if not semantic_query or not str(semantic_query).strip():
            self.logger.warning(f"optimize_search_query: semantic_query is empty, using base_query: '{base_query[:50]}...'")
            semantic_query = base_query

        # 4. 키워드 쿼리 생성 (법률 조항, 판례 검색용)
        keyword_queries = self.build_keyword_queries(base_query, expanded_terms, query_type)

        # keyword_queries 검증 및 수정
        if not keyword_queries or len(keyword_queries) == 0:
            self.logger.warning(f"optimize_search_query: keyword_queries is empty, using base_query")
            keyword_queries = [base_query]

        # LLM 변형 쿼리 추가
        if llm_variants:
            keyword_queries.extend(llm_variants[:3])  # 최대 3개만

        result = {
            "semantic_query": semantic_query,
            "keyword_queries": keyword_queries[:5],  # 최대 5개로 제한
            "expanded_keywords": expanded_terms,
            "llm_enhanced": llm_used
        }

        # 최종 검증 로그
        self.logger.debug(
            f"optimize_search_query result: "
            f"semantic_query length={len(semantic_query)}, "
            f"keyword_queries count={len(keyword_queries)}"
        )

        return result

    def enhance_query_with_llm(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """
        LLM을 사용하여 검색 쿼리를 강화 (캐싱 포함)

        Args:
            query: 원본 검색 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드 목록
            legal_field: 법률 분야

        Returns:
            {
                "optimized_query": "최적화된 쿼리",
                "expanded_keywords": ["키워드1", "키워드2", ...],
                "keyword_variants": ["변형 쿼리1", "변형 쿼리2", ...],
                "legal_terms": ["법률 용어1", "법률 용어2", ...],
                "reasoning": "개선 사유"
            } 또는 None (실패 시)
        """
        if not self.llm:
            self.logger.debug("LLM not available for query enhancement")
            return None

        # 캐시 키 생성
        cache_key = f"query_enhance:{query}:{query_type}:{legal_field}"

        # 캐시 확인
        if cache_key in self._query_enhancement_cache:
            self.logger.debug(f"Using cached query enhancement for: {query[:50]}")
            return self._query_enhancement_cache[cache_key]

        try:
            # 데이터 정규화 및 검증
            normalized_query_type = WorkflowUtils.normalize_query_type_for_prompt(query_type, self.logger)

            # extracted_keywords 검증
            if not extracted_keywords or not isinstance(extracted_keywords, list):
                extracted_keywords = []

            # legal_field 검증
            if not legal_field or not isinstance(legal_field, str):
                legal_field = ""

            # query 검증
            if not query or not isinstance(query, str):
                self.logger.warning("Invalid query provided for enhancement")
                return None

            # 로깅: 전달되는 데이터 확인
            self.logger.debug(
                f"🔍 [QUERY ENHANCEMENT] Building prompt with:\n"
                f"   query: '{query[:50]}...'\n"
                f"   query_type: '{normalized_query_type}'\n"
                f"   extracted_keywords: {len(extracted_keywords)} items\n"
                f"   legal_field: '{legal_field}'"
            )

            # Prompt Chaining을 사용한 쿼리 강화
            enhanced_result = self.enhance_query_with_chain(
                query=query,
                query_type=normalized_query_type,
                extracted_keywords=extracted_keywords,
                legal_field=legal_field
            )

            # 체인 실패 시 기존 방식으로 폴백
            if not enhanced_result:
                self.logger.debug("Chain enhancement failed, using fallback")
                prompt = self.build_query_enhancement_prompt(
                    query=query,
                    query_type=normalized_query_type,
                    extracted_keywords=extracted_keywords,
                    legal_field=legal_field
                )

                # LLM 호출 (짧은 응답만 필요하므로 토큰 수 제한)
                try:
                    response = self.llm.invoke(prompt)
                    if isinstance(response, str):
                        llm_output = response
                    elif hasattr(response, 'content'):
                        llm_output = response.content
                    else:
                        self.logger.warning(f"Unexpected LLM response type: {type(response)}")
                        return None
                except Exception as e:
                    self.logger.warning(f"LLM invocation failed: {e}")
                    return None

                # LLM 응답 파싱
                enhanced_result = self.parse_llm_query_enhancement(llm_output)

            if enhanced_result:
                # 결과 캐싱
                self._query_enhancement_cache[cache_key] = enhanced_result
                # 캐시 크기 제한 (최대 100개)
                if len(self._query_enhancement_cache) > 100:
                    # 오래된 항목 제거 (FIFO)
                    oldest_key = next(iter(self._query_enhancement_cache))
                    del self._query_enhancement_cache[oldest_key]

                self.logger.info(
                    f"✅ [LLM QUERY ENHANCEMENT] Original: '{query}' → "
                    f"Enhanced: '{enhanced_result.get('optimized_query', query)}'"
                )
            else:
                self.logger.debug("Failed to parse LLM enhancement response")

            return enhanced_result

        except Exception as e:
            self.logger.warning(f"Error in LLM query enhancement: {e}")
            return None

    def enhance_query_with_chain(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> Optional[Dict[str, Any]]:
        """
        Prompt Chaining을 사용한 검색 쿼리 강화 (다단계 체인)

        Step 1: 쿼리 분석 및 핵심 키워드 추출
        Step 2: 키워드 확장 및 변형 생성 (동의어, 관련 용어)
        Step 3: 최적화된 쿼리 생성 (벡터 검색용, 키워드 검색용)
        Step 4: 검증 및 개선 제안

        Returns:
            Optional[Dict[str, Any]]: 강화된 쿼리 결과 또는 None
        """
        try:
            # PromptChainExecutor 인스턴스 생성
            llm = self.llm_fast if self.llm_fast else self.llm
            chain_executor = PromptChainExecutor(llm, self.logger)

            # 체인 스텝 정의
            chain_steps = []

            # Step 1: 쿼리 분석 및 핵심 키워드 추출
            def build_query_analysis_prompt(prev_output, initial_input):
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
                query_type_value = initial_input.get("query_type") if isinstance(initial_input, dict) else query_type
                legal_field_value = initial_input.get("legal_field") if isinstance(initial_input, dict) else legal_field

                return f"""다음 법률 검색 쿼리를 분석하고 핵심 키워드를 추출해주세요.

원본 쿼리: {query_value}
질문 유형: {query_type_value}
법률 분야: {legal_field_value if legal_field_value else "미지정"}

다음 형식으로 응답해주세요:
{{
    "core_keywords": ["핵심 키워드1", "핵심 키워드2", "핵심 키워드3"],
    "query_intent": "쿼리의 의도 설명",
    "key_concepts": ["핵심 법률 개념1", "핵심 법률 개념2"],
    "analysis": "쿼리 분석 결과 (한국어)"
}}
"""

            chain_steps.append({
                "name": "query_analysis",
                "prompt_builder": build_query_analysis_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: QueryParser.parse_query_analysis_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "core_keywords" in output,
                "required": True
            })

            # Step 2: 키워드 확장 및 변형 생성
            def build_keyword_expansion_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                core_keywords = prev_output.get("core_keywords", [])
                query_intent = prev_output.get("query_intent", "")
                key_concepts = prev_output.get("key_concepts", [])
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
                query_type_value = initial_input.get("query_type") if isinstance(initial_input, dict) else query_type
                legal_field_value = initial_input.get("legal_field") if isinstance(initial_input, dict) else legal_field

                return f"""다음 핵심 키워드를 바탕으로 검색 범위를 확장하기 위한 키워드 변형과 동의어를 생성해주세요.

원본 쿼리: {query_value}
질문 유형: {query_type_value}
법률 분야: {legal_field_value if legal_field_value else "미지정"}
쿼리 의도: {query_intent}
핵심 키워드: {', '.join(core_keywords) if core_keywords else "없음"}
핵심 개념: {', '.join(key_concepts) if key_concepts else "없음"}

다음 형식으로 응답해주세요:
{{
    "expanded_keywords": ["확장 키워드1", "확장 키워드2", "확장 키워드3", "확장 키워드4", "확장 키워드5"],
    "synonyms": ["동의어1", "동의어2", "동의어3"],
    "related_terms": ["관련 용어1", "관련 용어2"],
    "keyword_variants": ["키워드 변형1", "키워드 변형2"],
    "reasoning": "확장 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "keyword_expansion",
                "prompt_builder": build_keyword_expansion_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: QueryParser.parse_keyword_expansion_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "expanded_keywords" in output,
                "required": True
            })

            # Step 3: 최적화된 쿼리 생성
            def build_query_optimization_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                expanded_keywords = prev_output.get("expanded_keywords", [])
                synonyms = prev_output.get("synonyms", [])
                related_terms = prev_output.get("related_terms", [])
                keyword_variants = prev_output.get("keyword_variants", [])

                # Step 1 결과에서 core_keywords 가져오기
                core_keywords = []
                if hasattr(chain_executor, 'chain_history'):
                    for step in chain_executor.chain_history:
                        if step.get("step_name") == "query_analysis" and step.get("success"):
                            step_output = step.get("output", {})
                            if isinstance(step_output, dict):
                                core_keywords = step_output.get("core_keywords", [])
                                break

                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query
                query_type_value = initial_input.get("query_type") if isinstance(initial_input, dict) else query_type
                legal_field_value = initial_input.get("legal_field") if isinstance(initial_input, dict) else legal_field

                return f"""다음 정보를 바탕으로 법률 데이터베이스 검색에 최적화된 쿼리를 생성해주세요.

원본 쿼리: {query_value}
질문 유형: {query_type_value}
법률 분야: {legal_field_value if legal_field_value else "미지정"}
핵심 키워드: {', '.join(core_keywords) if core_keywords else "없음"}
확장 키워드: {', '.join(expanded_keywords[:10]) if expanded_keywords else "없음"}

다음 형식으로 응답해주세요:
{{
    "optimized_query": "최적화된 메인 쿼리 (최대 50자)",
    "semantic_query": "벡터 검색용 쿼리 (의미 중심)",
    "keyword_query": "키워드 검색용 쿼리 (정확 매칭 중심)",
    "legal_terms": ["법률 전문 용어1", "법률 전문 용어2"],
    "reasoning": "최적화 근거 (한국어)"
}}
"""

            chain_steps.append({
                "name": "query_optimization",
                "prompt_builder": build_query_optimization_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: QueryParser.parse_query_optimization_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "optimized_query" in output,
                "required": True
            })

            # Step 4: 검증 및 개선 제안
            def build_query_validation_prompt(prev_output, initial_input):
                if not isinstance(prev_output, dict):
                    prev_output = {}

                optimized_query = prev_output.get("optimized_query", "")
                query_value = initial_input.get("query") if isinstance(initial_input, dict) else query

                return f"""다음 최적화된 쿼리를 검증하고 개선 제안을 해주세요.

원본 쿼리: {query_value}
최적화된 쿼리: {optimized_query}

다음 관점에서 검증해주세요:
1. 원본 쿼리의 핵심 의도가 유지되었는가?
2. 검색 정확도가 향상되었는가?
3. 검색 범위가 적절히 확장되었는가?
4. 법률 전문성이 반영되었는가?

다음 형식으로 응답해주세요:
{{
    "is_valid": true | false,
    "quality_score": 0.0-1.0,
    "improvements": ["개선 제안1", "개선 제안2"],
    "final_reasoning": "최종 검증 결과 및 개선 사유"
}}
"""

            chain_steps.append({
                "name": "query_validation",
                "prompt_builder": build_query_validation_prompt,
                "input_extractor": lambda prev: prev,
                "output_parser": lambda response, prev: QueryParser.parse_query_validation_response(response),
                "validator": lambda output: output and isinstance(output, dict) and "is_valid" in output,
                "required": False,
            })

            # 체인 실행
            initial_input_dict = {
                "query": query,
                "query_type": query_type,
                "extracted_keywords": extracted_keywords,
                "legal_field": legal_field
            }

            chain_result = chain_executor.execute_chain(
                chain_steps=chain_steps,
                initial_input=initial_input_dict,
                max_iterations=2,
                stop_on_failure=False
            )

            # 결과 추출 및 통합
            chain_history = chain_result.get("chain_history", [])

            # Step 1 결과: 쿼리 분석
            analysis_result = None
            for step in chain_history:
                if step.get("step_name") == "query_analysis" and step.get("success"):
                    analysis_result = step.get("output", {})
                    break

            # Step 2 결과: 키워드 확장
            expansion_result = None
            for step in chain_history:
                if step.get("step_name") == "keyword_expansion" and step.get("success"):
                    expansion_result = step.get("output", {})
                    break

            # Step 3 결과: 쿼리 최적화
            optimization_result = None
            for step in chain_history:
                if step.get("step_name") == "query_optimization" and step.get("success"):
                    optimization_result = step.get("output", {})
                    break

            # Step 4 결과: 검증
            validation_result = None
            for step in chain_history:
                if step.get("step_name") == "query_validation" and step.get("success"):
                    validation_result = step.get("output", {})
                    break

            # 결과 통합
            if not optimization_result or not isinstance(optimization_result, dict):
                return None

            # 최종 결과 생성
            enhanced_result = {
                "optimized_query": optimization_result.get("optimized_query", ""),
                "expanded_keywords": expansion_result.get("expanded_keywords", []) if expansion_result else [],
                "keyword_variants": [
                    optimization_result.get("semantic_query", ""),
                    optimization_result.get("keyword_query", "")
                ],
                "legal_terms": optimization_result.get("legal_terms", []),
                "reasoning": optimization_result.get("reasoning", "") or (validation_result.get("final_reasoning", "") if validation_result else "")
            }

            # 검증 결과 반영
            if validation_result and isinstance(validation_result, dict):
                if not validation_result.get("is_valid", True):
                    self.logger.warning(f"Query validation failed: {validation_result.get('final_reasoning', '')}")
                else:
                    quality_score = validation_result.get("quality_score", 0.8)
                    if quality_score < 0.7:
                        self.logger.warning(f"Low quality score: {quality_score}")

            # 체인 실행 결과 로깅
            chain_summary = chain_executor.get_chain_summary()
            self.logger.info(
                f"✅ [QUERY CHAIN] Executed {chain_summary['total_steps']} steps, "
                f"{chain_summary['successful_steps']} successful, "
                f"{chain_summary['failed_steps']} failed, "
                f"Total time: {chain_summary['total_time']:.2f}s"
            )

            return enhanced_result

        except Exception as e:
            self.logger.error(f"❌ [QUERY CHAIN ERROR] Prompt chain failed: {e}")
            return None

    def build_query_enhancement_prompt(
        self,
        query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str
    ) -> str:
        """쿼리 강화를 위한 LLM 프롬프트 생성 (개선된 버전)"""
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
            valid_keywords = [kw for kw in extracted_keywords if kw and isinstance(kw, str) and len(kw.strip()) > 0]
            if valid_keywords:
                keywords_info = f"""
**추출된 키워드 목록** (총 {len(valid_keywords)}개):
{chr(10).join([f"  - {kw.strip()}" for kw in valid_keywords[:10]])}
"""
                if len(valid_keywords) > 10:
                    keywords_info += f"  ... 외 {len(valid_keywords) - 10}개\n"
            else:
                keywords_info = "**추출된 키워드**: 없음 (쿼리에서 핵심 키워드를 자동으로 추출해야 함)\n\n**주의**: 원본 쿼리를 분석하여 법률 검색에 필요한 핵심 키워드를 식별하세요.\n"
        else:
            keywords_info = "**추출된 키워드**: 없음 (쿼리에서 핵심 키워드를 자동으로 추출해야 함)\n\n**주의**: 원본 쿼리를 분석하여 법률 검색에 필요한 핵심 키워드를 식별하세요.\n"

        # 질문 유형별 상세 가이드
        query_type_guides = {
            "precedent_search": {
                "description": "판례 검색",
                "search_focus": "사건번호, 법원명, 사건명, 판시사항, 판결요지, 관련 법령",
                "keyword_suggestions": ["판례", "대법원", "고등법원", "지방법원", "사건번호", "판결", "선고", "판시사항", "판결요지"],
                "database_fields": "cases.case_number, cases.court, cases.case_type, case_paragraphs.text, cases.announce_date",
                "search_strategy": "사건번호 패턴(YYYY다/나XXXXX), 법원명, 사건명의 핵심 키워드, 관련 법령명 조합"
            },
            "law_inquiry": {
                "description": "법령 조회",
                "search_focus": "법령명, 조문번호, 조항 내용, 시행일, 개정 이력",
                "keyword_suggestions": ["법률", "법령", "조항", "조문", "제XX조", "시행령", "시행규칙"],
                "database_fields": "statutes.name, statute_articles.article_no, statute_articles.text, statutes.effective_date",
                "search_strategy": "법령명 + 조문번호 조합, 조항의 핵심 법리 용어, 관련 판례와의 연계 키워드"
            },
            "legal_advice": {
                "description": "법률 조언",
                "search_focus": "관련 법률 조문, 유사 판례, 법리 해석, 실무 적용",
                "keyword_suggestions": ["법률", "판례", "조문", "법리", "해석", "적용", "요건", "효력"],
                "database_fields": "statute_articles, case_paragraphs, decision_paragraphs, interpretation_paragraphs",
                "search_strategy": "문제 상황의 핵심 법률 개념 + 관련 조문 + 유사 판례 패턴"
            },
            "document_analysis": {
                "description": "문서 분석",
                "search_focus": "문서 유형, 법적 근거, 관련 판례, 계약 조항",
                "keyword_suggestions": ["계약서", "법적 근거", "관련 판례", "조항", "계약 조항", "의무", "권리"],
                "database_fields": "전체 문서 검색 가능 (statutes, cases, decisions, interpretations)",
                "search_strategy": "문서에서 언급된 법령명 + 계약 유형 + 관련 법리"
            },
            "general_question": {
                "description": "일반 법률 질문",
                "search_focus": "관련 법령, 판례, 법률 용어, 실무 해석",
                "keyword_suggestions": ["법률", "법령", "판례", "법률 용어", "해석"],
                "database_fields": "전체 데이터베이스 검색",
                "search_strategy": "질문의 핵심 법률 개념 + 관련 분야 + 주요 용어"
            }
        }

        query_guide = query_type_guides.get(query_type, {
            "description": "일반 검색",
            "search_focus": "관련 법령, 판례, 법률 용어",
            "keyword_suggestions": [],
            "database_fields": "전체 데이터베이스",
            "search_strategy": "핵심 키워드 중심 검색"
        })

        # 법률 분야별 추가 정보
        field_specific_info = {
            "family": {
                "related_laws": ["민법 가족편", "가족관계의 등록 등에 관한 법률"],
                "key_concepts": ["혼인", "이혼", "양육권", "친권", "상속", "위자료", "재산분할"],
                "common_keywords": ["부부", "가족", "이혼", "상속", "친자", "양육", "위자료"]
            },
            "civil": {
                "related_laws": ["민법", "민사소송법"],
                "key_concepts": ["계약", "불법행위", "손해배상", "채권", "채무", "소유권", "점유"],
                "common_keywords": ["계약", "손해배상", "채권", "채무", "소유권"]
            },
            "criminal": {
                "related_laws": ["형법", "형사소송법"],
                "key_concepts": ["범죄", "구성요건", "형량", "처벌", "기소", "공소"],
                "common_keywords": ["범죄", "처벌", "형량", "구성요건", "기소"]
            },
            "labor": {
                "related_laws": ["근로기준법", "노동조합법", "고용보험법"],
                "key_concepts": ["근로계약", "임금", "근로시간", "해고", "퇴직금", "산재"],
                "common_keywords": ["근로", "임금", "해고", "노동", "근로자", "사용자"]
            },
            "corporate": {
                "related_laws": ["상법", "주식회사법", "법인세법"],
                "key_concepts": ["회사", "주주", "이사", "법인", "자본", "이사회"],
                "common_keywords": ["회사", "주주", "이사", "법인", "기업"]
            },
            "tax": {
                "related_laws": ["소득세법", "법인세법", "부가가치세법"],
                "key_concepts": ["소득세", "법인세", "부가가치세", "과세", "공제", "세율"],
                "common_keywords": ["세금", "과세", "소득세", "법인세", "부가가치세"]
            },
            "intellectual_property": {
                "related_laws": ["특허법", "상표법", "저작권법", "디자인보호법"],
                "key_concepts": ["특허", "상표", "저작권", "디자인", "침해", "등록"],
                "common_keywords": ["특허", "상표", "저작권", "지적재산", "침해"]
            }
        }

        field_info = field_specific_info.get(legal_field, {
            "related_laws": [],
            "key_concepts": [],
            "common_keywords": []
        })

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
{self.format_field_info(legal_field, field_info)}

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
    "optimized_query": "개선된 검색 쿼리 (원본 의도 유지, 법률 검색 최적화)",
    "expanded_keywords": ["관련 키워드1", "관련 키워드2", "관련 키워드3", "관련 키워드4", "관련 키워드5"],
    "keyword_variants": ["검색 변형 쿼리1 (법령 검색용)", "검색 변형 쿼리2 (판례 검색용)"],
    "legal_terms": ["법률 전문 용어1", "법률 전문 용어2"],
    "reasoning": "개선 사유: 원본 쿼리를 어떤 방식으로 개선했고, 왜 그렇게 개선했는지 간단히 설명"
}}
```

### 출력 필드 설명
- **optimized_query**: 벡터 검색과 키워드 검색 모두에 사용될 메인 쿼리 (최대 50자 권장)
- **expanded_keywords**: 검색 범위 확장을 위한 관련 키워드 목록 (5-10개 권장)
- **keyword_variants**: 다양한 검색 시도를 위한 쿼리 변형 (2-3개 권장, 법령/판례 검색 구분)
- **legal_terms**: 법률 전문 용어 목록 (법률 용어 사전에 등록될 용어)
- **reasoning**: 개선 사유 (50자 이내, 선택 사항)

## ✅ 검증 체크리스트

출력 전 다음을 확인하세요:
- [ ] optimized_query가 원본 쿼리의 핵심 의도를 유지하는가?
- [ ] expanded_keywords에 법률 분야별 관련 용어가 포함되었는가?
- [ ] keyword_variants에 질문 유형에 맞는 검색 변형이 포함되었는가?
- [ ] 모든 키워드가 한글로 작성되었는가?
- [ ] JSON 형식이 올바른가? (설명 없이 JSON만 출력)

## 📝 좋은 예시

**입력**: "계약 해지 방법"
- optimized_query: "계약 해지 요건 및 절차"
- expanded_keywords: ["계약 해제", "해지 통고", "채무불이행", "이행 최고", "해지 의사표시"]
- keyword_variants: ["계약 해지 요건 법령", "계약 해지 판례"]

**입력**: "대법원 2020다12345 판례"
- optimized_query: "대법원 2020다12345 판결 요지"
- expanded_keywords: ["2020다12345", "대법원 판례", "판결 요지", "사건번호"]
- keyword_variants: ["2020다12345", "대법원 2020다12345"]

이제 다음 쿼리를 개선해주세요:
"""

        return prompt

    def format_field_info(self, legal_field: str, field_info: Dict[str, Any]) -> str:
        """법률 분야별 정보 포맷팅"""
        if legal_field and field_info.get('related_laws'):
            related_laws = ', '.join(field_info.get('related_laws', [])) if field_info.get('related_laws') else '없음'
            key_concepts = ', '.join(field_info.get('key_concepts', [])) if field_info.get('key_concepts') else '없음'
            common_keywords = ', '.join(field_info.get('common_keywords', [])) if field_info.get('common_keywords') else '없음'

            return f"""
**관련 법령**: {related_laws}
**핵심 개념**: {key_concepts}
**일반 키워드**: {common_keywords}
"""
        else:
            return "**법률 분야**: 미지정 (전체 법률 분야 검색)"

    def parse_llm_query_enhancement(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """LLM 응답 파싱"""
        try:
            import json
            import re

            # JSON 추출 (코드 블록 내부 또는 직접 JSON)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 코드 블록 없이 JSON만 있는 경우
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning("No JSON found in LLM response")
                    return None

            # JSON 파싱
            result = json.loads(json_str)

            # 필수 필드 확인
            if not result.get("optimized_query"):
                self.logger.warning("LLM response missing 'optimized_query' field")
                return None

            # 기본값 설정
            enhanced = {
                "optimized_query": result.get("optimized_query", ""),
                "expanded_keywords": result.get("expanded_keywords", []),
                "keyword_variants": result.get("keyword_variants", []),
                "legal_terms": result.get("legal_terms", []),
                "reasoning": result.get("reasoning", "")
            }

            # 유효성 검사
            if not enhanced["optimized_query"] or len(enhanced["optimized_query"]) > 500:
                self.logger.warning("Invalid optimized_query from LLM")
                return None

            return enhanced

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Error parsing LLM enhancement response: {e}")
            return None

    def normalize_legal_terms(self, query: str, keywords: List[str]) -> List[str]:
        """법률 용어 정규화"""
        normalized = []
        all_terms = [query] + (keywords if keywords else [])

        for term in all_terms:
            # 법률 용어 정규화
            if isinstance(term, str) and len(term) >= 2:
                # term_integrator를 사용한 정규화
                try:
                    if hasattr(self.term_integrator, 'normalize_term'):
                        normalized_term = self.term_integrator.normalize_term(term)
                        if normalized_term:
                            normalized.append(normalized_term)
                        else:
                            normalized.append(term)
                    else:
                        normalized.append(term)
                except Exception:
                    normalized.append(term)
            elif isinstance(term, str):
                normalized.append(term)

        return list(set(normalized)) if normalized else [query]

    def expand_legal_terms(
        self,
        terms: List[str],
        legal_field: str
    ) -> List[str]:
        """법률 용어 확장 (동의어, 관련 용어)"""
        expanded = list(terms)

        # 지원되는 법률 분야별 관련 용어 매핑 (민사법, 지식재산권법, 행정법, 형사법만)
        field_expansions = {
            "civil": ["민사", "계약", "손해배상", "채권", "채무"],
            "criminal": ["형사", "범죄", "처벌", "형량"],
            "intellectual_property": ["특허", "상표", "저작권", "지적재산"],
            "administrative": ["행정", "행정처분", "행정소송", "행정심판"]
        }

        # 관련 용어 추가
        if legal_field:
            related_terms = field_expansions.get(legal_field, [])
            expanded.extend(related_terms)

        return list(set(expanded))[:15]  # 최대 15개로 제한

    def clean_query_for_fallback(self, query: str) -> str:
        """LLM 실패 시 기본 쿼리 정제 (폴백 강화)"""
        if not query or not isinstance(query, str):
            return ""

        # 불용어 제거 및 정제
        stopwords = ["은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "와", "과", "도", "만"]
        words = query.split()
        cleaned_words = [w for w in words if w not in stopwords and len(w) >= 2]

        # 정제된 쿼리 반환 (비어있으면 원본 반환)
        cleaned = " ".join(cleaned_words) if cleaned_words else query
        return cleaned.strip()

    def build_semantic_query(self, query: str, expanded_terms: List[str]) -> str:
        """의미적 검색용 쿼리 생성"""
        return QueryBuilder.build_semantic_query(query, expanded_terms)

    def build_keyword_queries(
        self,
        query: str,
        expanded_terms: List[str],
        query_type: str
    ) -> List[str]:
        """키워드 검색용 쿼리 리스트 생성"""
        return QueryBuilder.build_keyword_queries(query, expanded_terms, query_type)

    def determine_search_parameters(
        self,
        query_type: str,
        query_complexity: int,
        keyword_count: int,
        is_retry: bool
    ) -> Dict[str, Any]:
        """검색 파라미터 동적 결정"""
        base_k = WorkflowConstants.SEMANTIC_SEARCH_K
        base_limit = WorkflowConstants.CATEGORY_SEARCH_LIMIT

        # 질문 유형에 따른 조정
        type_multiplier = {
            "precedent_search": 1.5,  # 판례 검색: 더 많은 결과
            "law_inquiry": 1.3,       # 법령 조회: 더 많은 결과
            "legal_advice": 1.2,
            "general_question": 1.0
        }
        multiplier = type_multiplier.get(query_type, 1.0)

        # 복잡도에 따른 조정
        if query_complexity > 100:
            multiplier += 0.3
        if keyword_count > 10:
            multiplier += 0.2

        # 재시도 시 더 많은 결과
        if is_retry:
            multiplier += 0.5

        semantic_k = int(base_k * multiplier)
        keyword_limit = int(base_limit * multiplier)

        # 유사도 임계값 동적 조정
        min_relevance = self.config.similarity_threshold
        if query_type == "precedent_search":
            min_relevance = max(0.6, min_relevance - 0.1)  # 판례 검색: 완화
        elif query_type == "law_inquiry":
            min_relevance = max(0.65, min_relevance - 0.05)  # 법령 조회: 약간 완화

        return {
            "semantic_k": min(25, semantic_k),  # 최대 25개
            "keyword_limit": min(7, keyword_limit),  # 최대 7개
            "min_relevance": min_relevance,
            "max_results": int(base_k * multiplier * 1.2),  # 최종 결과 수
            "rerank": {
                "top_k": min(20, int(base_k * multiplier)),
                "diversity_weight": 0.3,
                "relevance_weight": 0.7
            }
        }

    def extract_query_relevant_sentences(
        self,
        doc_content: str,
        query: str,
        extracted_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """문서 내용에서 질문과 직접 관련된 문장 추출"""
        return DocumentExtractor.extract_query_relevant_sentences(doc_content, query, extracted_keywords)
