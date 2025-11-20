# -*- coding: utf-8 -*-
"""
하이브리드 쿼리 프로세서 (HuggingFace + LLM)
Multi-Query 생성만 LLM 사용, 나머지는 HuggingFace 모델 사용
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.search.optimizers.legal_query_analyzer import LegalQueryAnalyzer
from core.search.optimizers.legal_keyword_expander import LegalKeywordExpander
from core.search.optimizers.legal_query_optimizer import LegalQueryOptimizer
from core.search.optimizers.legal_query_validator import LegalQueryValidator

logger = logging.getLogger(__name__)


class HybridQueryProcessor:
    """하이브리드 쿼리 프로세서 (HuggingFace + LLM)"""
    
    def __init__(
        self,
        keyword_extractor: Optional[Any] = None,
        term_integrator: Optional[Any] = None,
        llm: Optional[Any] = None,
        embedding_model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        HybridQueryProcessor 초기화
        
        Args:
            keyword_extractor: 키워드 추출기
            term_integrator: 법률 용어 통합기
            llm: LLM 인스턴스 (Multi-Query 생성용만)
            embedding_model_name: 임베딩 모델명
            logger: 로거
        """
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm
        
        # HuggingFace 모델 기반 컴포넌트 초기화
        self.query_analyzer = LegalQueryAnalyzer(
            keyword_extractor=keyword_extractor,
            embedding_model_name=embedding_model_name,
            logger=self.logger
        )
        
        self.keyword_expander = LegalKeywordExpander(
            term_integrator=term_integrator,
            embedding_model_name=embedding_model_name,
            logger=self.logger
        )
        
        self.query_optimizer = LegalQueryOptimizer(
            embedding_model_name=embedding_model_name,
            logger=self.logger
        )
        
        self.query_validator = LegalQueryValidator(
            embedding_model_name=embedding_model_name,
            logger=self.logger
        )
        
        self.logger.info("✅ [HYBRID PROCESSOR] HybridQueryProcessor initialized")
    
    def process_query_hybrid(
        self,
        query: str,
        search_query: str,
        query_type: str,
        extracted_keywords: List[str],
        legal_field: str,
        complexity: str,
        is_retry: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        하이브리드 쿼리 처리
        
        Args:
            query: 원본 쿼리
            search_query: 검색용 쿼리
            query_type: 질문 유형
            extracted_keywords: 추출된 키워드
            legal_field: 법률 분야
            complexity: 복잡도 (simple, moderate, complex)
            is_retry: 재시도 여부
            
        Returns:
            (optimized_queries, cache_hit)
        """
        # Step 1: 쿼리 분석 (HuggingFace)
        self.logger.info(f"🔍 [HYBRID] Step 1: Query analysis (HuggingFace)")
        analysis_result = self.query_analyzer.analyze_query(
            query, query_type, legal_field
        )
        core_keywords = analysis_result.get("core_keywords", extracted_keywords)
        
        # Step 2: 키워드 확장 (HuggingFace)
        self.logger.info(f"🔍 [HYBRID] Step 2: Keyword expansion (HuggingFace)")
        expansion_result = self.keyword_expander.expand_keywords(
            query,
            core_keywords,
            extracted_keywords,
            legal_field
        )
        
        # Step 3: 쿼리 최적화 (HuggingFace)
        self.logger.info(f"🔍 [HYBRID] Step 3: Query optimization (HuggingFace)")
        optimization_result = self.query_optimizer.optimize_query(
            query,
            core_keywords,
            expansion_result["expanded_keywords"],
            query_type
        )
        
        # Step 4: 검증 (HuggingFace, 선택적)
        if complexity in ["moderate", "complex"]:
            self.logger.info(f"🔍 [HYBRID] Step 4: Query validation (HuggingFace)")
            validation_result = self.query_validator.validate_query(
                optimization_result, query
            )
            
            if not validation_result["is_valid"] and validation_result["improvements"]:
                self.logger.info(f"⚠️ [HYBRID] Validation failed, applying improvements")
                optimization_result = self._apply_improvements(
                    optimization_result, validation_result["improvements"]
                )
        
        # Step 5: Multi-Query 생성 (LLM만 사용)
        multi_queries = None
        if self.llm:
            self.logger.info(f"🔍 [HYBRID] Step 5: Multi-query generation (LLM)")
            try:
                max_queries = self._get_max_queries_by_complexity(complexity)
                multi_queries = self._generate_multi_queries_with_llm(
                    search_query, query_type, max_queries
                )
            except Exception as e:
                self.logger.warning(f"⚠️ [HYBRID] Multi-query generation failed: {e}")
                multi_queries = [search_query]
        else:
            self.logger.warning("⚠️ [HYBRID] LLM not available, skipping multi-query generation")
            multi_queries = [search_query]
        
        # 결과 통합
        optimized_queries = {
            "semantic_query": optimization_result["semantic_query"],
            "keyword_queries": optimization_result["keyword_queries"],
            "expanded_keywords": expansion_result["expanded_keywords"],
            "synonyms": expansion_result.get("synonyms", []),
            "legal_references": expansion_result.get("legal_references", []),
            "multi_queries": multi_queries,
            "llm_enhanced": False,  # Multi-Query만 LLM 사용
            "hf_models_used": True,
            "quality_score": optimization_result.get("quality_score", 0.7)
        }
        
        # Multi-Query가 있으면 첫 번째를 semantic_query로 사용
        if multi_queries and len(multi_queries) > 1:
            optimized_queries["semantic_query"] = multi_queries[0]
        
        self.logger.info(
            f"✅ [HYBRID] Query processing completed: "
            f"semantic_query='{optimized_queries['semantic_query'][:50]}...', "
            f"keyword_queries={len(optimized_queries['keyword_queries'])}, "
            f"multi_queries={len(multi_queries) if multi_queries else 0}"
        )
        
        return optimized_queries, False  # 캐시 히트는 별도 처리
    
    def _generate_multi_queries_with_llm(
        self,
        query: str,
        query_type: str,
        max_queries: int = 3
    ) -> List[str]:
        """Multi-Query 생성 (LLM만 사용)"""
        if not self.llm or not query:
            return [query] if query else []
        
        try:
            # 간단한 메모리 캐시 사용
            if not hasattr(self.__class__, '_multi_query_cache'):
                self.__class__._multi_query_cache = {}
            
            cache_key = f"multi_query:{query}:{query_type}:{max_queries}"
            
            # 캐시 확인
            if cache_key in self.__class__._multi_query_cache:
                self.logger.info(f"✅ [MULTI-QUERY] Cache hit for query: '{query[:50]}...'")
                return self.__class__._multi_query_cache[cache_key]
            
            # 새로운 프롬프트 (법률 전문 질의 재작성)
            num_variations = max_queries - 1  # 원본 제외한 변형 개수
            
            prompt = f"""당신은 법률 분야 전문 질의 재작성(Multi-Query) 생성기입니다.  

지금부터 사용자의 원본 질문을 **서로 다른 관점·법률 용어·쟁점 표현·조문 방식**으로 다양하게 변형해 생성하세요.

아래 규칙을 따르십시오:

[생성 규칙]

1. 원문의 의미는 유지하되, 서로 다른 방식(용어·문장구조·법률 개념)으로 표현할 것

2. 법률 용어(조문, 법률명, 법적 표현 등)를 포함한 변형 1개 이상 생성

3. 실무에서 자주 쓰는 질문 형태로 변형 1개 이상 생성

4. 너무 포괄적이거나 너무 좁은 의미로 변형하지 말 것

5. 한 줄에 하나씩 출력할 것

6. 질문만 출력하고 설명은 금지

[원본 질문]
{query}

[출력 형태]
재작성:
- 질문1
- 질문2
- 질문3
{'- 질문4' if num_variations >= 4 else ''}{'- 질문5' if num_variations >= 5 else ''}

총 {num_variations}개의 변형된 질문을 생성하세요."""
            
            # LLM 호출
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
            elif hasattr(self.llm, '__call__'):
                response = self.llm(prompt)
            else:
                response = str(self.llm)
            
            if isinstance(response, str):
                llm_output = response
            elif hasattr(response, 'content'):
                llm_output = response.content
            else:
                llm_output = str(response)
            
            # 응답 파싱 (새로운 프롬프트 형식에 맞게)
            queries = []
            skip_patterns = [
                "재작성:", "재작성", "각 줄에", "하나씩", "질문:", "유형:", "원본 질문:",
                "요구사항:", "다음 질문을", "다음 법률 질문을", "출력 형태", "생성 규칙",
                "당신은", "법률 분야", "지금부터", "아래 규칙", "원본 질문", "총", "개의 변형"
            ]
            
            in_reformatted_section = False
            for line in llm_output.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # "재작성:" 섹션 시작 확인
                if "재작성" in line and ":" in line:
                    in_reformatted_section = True
                    continue
                
                # 프롬프트 텍스트 스킵
                if any(pattern in line for pattern in skip_patterns):
                    continue
                
                # "- 질문" 형식 또는 번호 패턴 제거
                if line.startswith('-'):
                    line = line[1:].strip()
                line = line.lstrip('0123456789.-) ')
                
                if line and not line.startswith('#') and len(line) > 5:
                    queries.append(line)
            
            # 원본 질문을 첫 번째로 포함
            result_queries = [query] + queries[:max_queries - 1]
            result_queries = result_queries[:max_queries]
            
            if not result_queries:
                result_queries = [query]
            
            # 캐시 저장
            if len(self.__class__._multi_query_cache) >= 200:
                oldest_key = next(iter(self.__class__._multi_query_cache))
                del self.__class__._multi_query_cache[oldest_key]
            self.__class__._multi_query_cache[cache_key] = result_queries
            
            self.logger.info(
                f"✅ [MULTI-QUERY] Generated {len(result_queries)} queries "
                f"(original + {len(result_queries) - 1} variations)"
            )
            
            return result_queries
            
        except Exception as e:
            self.logger.warning(f"⚠️ [MULTI-QUERY] LLM generation failed: {e}, using original query")
            return [query]
    
    def _get_max_queries_by_complexity(self, complexity: str) -> int:
        """복잡도에 따른 최대 쿼리 수"""
        complexity_map = {
            "simple": 2,
            "moderate": 3,
            "complex": 4
        }
        return complexity_map.get(complexity, 3)
    
    def _apply_improvements(
        self,
        optimization_result: Dict[str, Any],
        improvements: List[str]
    ) -> Dict[str, Any]:
        """개선 제안 적용"""
        # 간단한 개선 적용 (규칙 기반)
        semantic_query = optimization_result.get("semantic_query", "")
        
        # 법률 전문 용어 추가 제안이 있으면
        if any("법률 전문" in imp or "용어" in imp for imp in improvements):
            # 법률 용어 패턴 추가 (간단한 예시)
            legal_terms = ["법률", "조문", "규정"]
            for term in legal_terms:
                if term not in semantic_query:
                    semantic_query = f"{semantic_query} {term}"
                    break
        
        optimization_result["semantic_query"] = semantic_query
        return optimization_result

