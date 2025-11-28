# -*- coding: utf-8 -*-
"""
Multi-Query Search Agent Node
멀티 질의 생성 + 에이전트 기반 검색 노드
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

try:
    from lawfirm_langgraph.core.workflow.state.state_definitions import LegalWorkflowState
except ImportError:
    from core.workflow.state.state_definitions import LegalWorkflowState

# LangChain imports
try:
    from langchain.tools import tool
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock for when LangChain is not available
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = get_logger(__name__)


class MultiQuerySearchAgentNode:
    """멀티 질의 생성 + 에이전트 기반 검색 노드"""
    
    def __init__(self, workflow_instance, logger_instance=None):
        self.workflow = workflow_instance
        self.logger = logger_instance or logger
        
        # 검색 엔진 초기화 (지연 초기화)
        self.semantic_search = None
        self.keyword_search = None
        self.hybrid_query_processor = None
        
        # 에이전트 초기화 (지연 초기화)
        self.agentic_agent = None
        self.search_tools = []
        self.llm = None
    
    def _initialize_search_engines(self):
        """검색 엔진 초기화 (지연 초기화)"""
        if self.semantic_search is None:
            try:
                from core.search.engines.semantic_search_engine_v2 import SemanticSearchEngineV2
                self.semantic_search = SemanticSearchEngineV2()
                self.logger.debug("✅ SemanticSearchEngineV2 initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize SemanticSearchEngineV2: {e}")
        
        if self.keyword_search is None:
            try:
                from core.search.connectors.legal_data_connector_v2 import LegalDataConnectorV2
                self.keyword_search = LegalDataConnectorV2()
                self.logger.debug("✅ LegalDataConnectorV2 initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize LegalDataConnectorV2: {e}")
        
        if self.hybrid_query_processor is None and self.workflow:
            self.hybrid_query_processor = getattr(self.workflow, 'hybrid_query_processor', None)
    
    def _create_postgresql_keyword_search_tool(self):
        """PostgreSQL 키워드 검색 도구"""
        @tool
        def search_postgresql_keywords(query: str, limit: int = 10) -> str:
            '''
            PostgreSQL 키워드 검색: 법령명, 조문, 판례명 등 정확한 키워드 검색
            
            사용 시기:
            - 정확한 법령명이나 조문번호가 포함된 질문
            - 판례명이나 사건번호가 포함된 질문
            - 특정 키워드로 정확히 매칭해야 하는 경우
            
            Args:
                query: 검색 쿼리 (예: "민법 제750조", "대법원 2020다12345")
                limit: 최대 결과 수 (기본값: 10)
            
            Returns:
                JSON 형식의 검색 결과
            '''
            try:
                if not self.keyword_search:
                    self._initialize_search_engines()
                
                if not self.keyword_search:
                    return json.dumps({
                        "success": False,
                        "error": "Keyword search engine not available",
                        "search_type": "postgresql_keyword"
                    })
                
                results = self.keyword_search.search_documents(query, limit=limit, force_fts=True)
                return json.dumps({
                    "success": True,
                    "search_type": "postgresql_keyword",
                    "query": query,
                    "results": results,
                    "count": len(results)
                }, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"❌ [POSTGRESQL-KEYWORD] Error: {e}", exc_info=True)
                return json.dumps({
                    "success": False,
                    "error": str(e),
                    "search_type": "postgresql_keyword"
                })
        
        return search_postgresql_keywords
    
    def _create_vector_index_search_tool(self):
        """벡터 인덱스 검색 도구"""
        @tool
        def search_vector_index(query: str, limit: int = 10) -> str:
            '''
            벡터 의미 검색: 질문의 의미를 이해하여 유사한 법률 문서 검색
            
            사용 시기:
            - 의미 기반 검색이 필요한 경우
            - 키워드가 명확하지 않지만 의도를 이해해야 하는 경우
            - 유사한 법률 개념을 찾아야 하는 경우
            
            Args:
                query: 검색 쿼리 (예: "계약 해지 사유에 대해 알려주세요")
                limit: 최대 결과 수 (기본값: 10)
            
            Returns:
                JSON 형식의 검색 결과
            '''
            try:
                if not self.semantic_search:
                    self._initialize_search_engines()
                
                if not self.semantic_search:
                    return json.dumps({
                        "success": False,
                        "error": "Vector search engine not available",
                        "search_type": "vector_semantic"
                    })
                
                results = self.semantic_search.search(query, k=limit)
                return json.dumps({
                    "success": True,
                    "search_type": "vector_semantic",
                    "query": query,
                    "results": results,
                    "count": len(results)
                }, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"❌ [VECTOR-INDEX] Error: {e}", exc_info=True)
                return json.dumps({
                    "success": False,
                    "error": str(e),
                    "search_type": "vector_semantic"
                })
        
        return search_vector_index
    
    def _create_hybrid_search_tool(self):
        """하이브리드 검색 도구"""
        @tool
        def search_hybrid(query: str, limit: int = 10) -> str:
            '''
            하이브리드 검색: 키워드 검색과 벡터 검색을 모두 수행하고 결과를 병합
            
            사용 시기:
            - 정확성과 포괄성을 모두 필요로 하는 경우
            - 복잡한 법률 질문
            - 여러 관점에서 검색이 필요한 경우
            
            Args:
                query: 검색 쿼리
                limit: 최대 결과 수 (기본값: 10)
            
            Returns:
                JSON 형식의 통합 검색 결과
            '''
            try:
                if not self.keyword_search or not self.semantic_search:
                    self._initialize_search_engines()
                
                if not self.keyword_search or not self.semantic_search:
                    return json.dumps({
                        "success": False,
                        "error": "Search engines not available",
                        "search_type": "hybrid"
                    })
                
                # 병렬 검색
                with ThreadPoolExecutor(max_workers=2) as executor:
                    keyword_future = executor.submit(
                        self.keyword_search.search_documents, query, limit=limit, force_fts=True
                    )
                    vector_future = executor.submit(
                        self.semantic_search.search, query, k=limit
                    )
                    
                    keyword_results = keyword_future.result(timeout=10.0)
                    vector_results = vector_future.result(timeout=10.0)
                
                # 결과 병합 및 리랭킹
                merged_results = self._merge_and_rerank(
                    keyword_results, vector_results, limit
                )
                
                return json.dumps({
                    "success": True,
                    "search_type": "hybrid",
                    "query": query,
                    "keyword_count": len(keyword_results),
                    "vector_count": len(vector_results),
                    "results": merged_results,
                    "count": len(merged_results)
                }, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"❌ [HYBRID] Error: {e}", exc_info=True)
                return json.dumps({
                    "success": False,
                    "error": str(e),
                    "search_type": "hybrid"
                })
        
        return search_hybrid
    
    def _create_multi_query_search_tool(self):
        """멀티 질의 검색 도구 (핵심 기능)"""
        @tool
        def search_multi_query(original_query: str, max_queries: int = 3, limit_per_query: int = 5) -> str:
            '''
            멀티 질의 검색: 원본 질문을 여러 관점의 하위 질문으로 분해하여 각각 검색
            
            사용 시기:
            - 복잡하고 다면적인 법률 질문
            - 여러 관점에서 검색이 필요한 경우
            - 단일 검색으로는 부족한 경우
            
            작동 방식:
            1. 원본 질문을 여러 하위 질문으로 분해
            2. 각 하위 질문에 대해 키워드 검색과 벡터 검색을 모두 수행
            3. 모든 결과를 통합하여 리랭킹
            
            Args:
                original_query: 원본 검색 쿼리
                max_queries: 생성할 최대 하위 질문 수 (기본값: 3)
                limit_per_query: 각 질문당 최대 결과 수 (기본값: 5)
            
            Returns:
                JSON 형식의 통합 검색 결과
            '''
            try:
                if not self.keyword_search or not self.semantic_search:
                    self._initialize_search_engines()
                
                if not self.keyword_search or not self.semantic_search:
                    return json.dumps({
                        "success": False,
                        "error": "Search engines not available",
                        "search_type": "multi_query"
                    })
                
                # 1. 멀티 질의 생성
                multi_queries = self._generate_multi_queries(original_query, max_queries)
                
                self.logger.info(f"🔍 [MULTI-QUERY] Generated {len(multi_queries)} sub-queries: {multi_queries}")
                
                # 2. 각 하위 질문에 대해 병렬 검색
                all_results = []
                seen_doc_ids = set()
                
                with ThreadPoolExecutor(max_workers=min(len(multi_queries) * 2, 10)) as executor:
                    futures = []
                    
                    for sub_query in multi_queries:
                        # 각 하위 질문에 대해 키워드 검색과 벡터 검색 모두 수행
                        keyword_future = executor.submit(
                            self.keyword_search.search_documents,
                            sub_query, limit=limit_per_query, force_fts=True
                        )
                        vector_future = executor.submit(
                            self.semantic_search.search,
                            sub_query, k=limit_per_query
                        )
                        futures.append(("keyword", sub_query, keyword_future))
                        futures.append(("vector", sub_query, vector_future))
                    
                    # 결과 수집
                    for search_type, sub_query, future in futures:
                        try:
                            results = future.result(timeout=10.0)
                            for result in results:
                                # 중복 제거
                                doc_id = self._get_doc_id(result)
                                if doc_id and doc_id not in seen_doc_ids:
                                    seen_doc_ids.add(doc_id)
                                    result["sub_query"] = sub_query
                                    result["search_type"] = search_type
                                    result["original_query"] = original_query
                                    all_results.append(result)
                        except Exception as e:
                            self.logger.warning(f"⚠️ [MULTI-QUERY] Search failed for '{sub_query}': {e}")
                
                # 3. 결과 리랭킹
                ranked_results = self._rerank_multi_query_results(all_results, original_query)
                
                return json.dumps({
                    "success": True,
                    "search_type": "multi_query",
                    "original_query": original_query,
                    "sub_queries": multi_queries,
                    "sub_query_count": len(multi_queries),
                    "total_results": len(ranked_results),
                    "results": ranked_results[:limit_per_query * max_queries],
                    "count": len(ranked_results)
                }, ensure_ascii=False)
                
            except Exception as e:
                self.logger.error(f"❌ [MULTI-QUERY] Error: {e}", exc_info=True)
                return json.dumps({
                    "success": False,
                    "error": str(e),
                    "search_type": "multi_query"
                })
        
        return search_multi_query
    
    def _generate_multi_queries(self, query: str, max_queries: int = 3) -> List[str]:
        """멀티 질의 생성"""
        try:
            # 기존 HybridQueryProcessor 활용
            if self.hybrid_query_processor:
                query_info = {
                    "query": query,
                    "search_query": query,
                    "query_type": "general",
                    "extracted_keywords": [],
                    "legal_field": None,
                    "complexity": "moderate",
                    "is_retry": False
                }
                
                optimized_queries, _ = self.hybrid_query_processor.process_query_hybrid(
                    query=query_info["query"],
                    search_query=query_info["search_query"],
                    query_type=query_info["query_type"],
                    extracted_keywords=query_info["extracted_keywords"],
                    legal_field=query_info["legal_field"],
                    complexity=query_info["complexity"],
                    is_retry=query_info["is_retry"]
                )
                
                multi_queries = optimized_queries.get("multi_queries", [query])
                if len(multi_queries) > max_queries:
                    multi_queries = multi_queries[:max_queries]
                return multi_queries
            else:
                # 폴백: 워크플로우의 멀티 질의 생성 메서드 사용
                if self.workflow and hasattr(self.workflow, '_generate_multi_queries_with_llm'):
                    return self.workflow._generate_multi_queries_with_llm(
                        query=query,
                        query_type="general",
                        max_queries=max_queries
                    )
                else:
                    # 최종 폴백: 간단한 질의 분해
                    return self._generate_simple_multi_queries(query, max_queries)
        except Exception as e:
            self.logger.warning(f"⚠️ [MULTI-QUERY] Failed to generate multi-queries: {e}, using simple method")
            return self._generate_simple_multi_queries(query, max_queries)
    
    def _generate_simple_multi_queries(self, query: str, max_queries: int = 3) -> List[str]:
        """간단한 멀티 질의 생성 (폴백)"""
        queries = [query]
        
        # 질문 유형에 따른 분해
        if "해지" in query:
            queries.append(query.replace("해지", "해지 사유"))
            queries.append(query.replace("해지", "해지 절차"))
        elif "요건" in query:
            queries.append(query.replace("요건", "성립 요건"))
            queries.append(query.replace("요건", "효력 요건"))
        elif "효과" in query:
            queries.append(query.replace("효과", "법적 효과"))
            queries.append(query.replace("효과", "실제 효과"))
        elif "계약" in query:
            queries.append(query + " 사유")
            queries.append(query + " 절차")
        
        return queries[:max_queries]
    
    def _get_doc_id(self, result: Dict[str, Any]) -> Optional[str]:
        """문서 ID 추출"""
        if isinstance(result, dict):
            metadata = result.get("metadata", {})
            return (metadata.get("id") or 
                   metadata.get("chunk_id") or 
                   metadata.get("source_id") or
                   result.get("id") or
                   result.get("source", ""))
        return None
    
    def _merge_and_rerank(self, keyword_results: List[Dict], vector_results: List[Dict], limit: int) -> List[Dict]:
        """키워드 검색과 벡터 검색 결과 병합 및 리랭킹"""
        seen_ids = set()
        merged = []
        
        for result in keyword_results + vector_results:
            doc_id = self._get_doc_id(result)
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                # 키워드 검색 결과에 가중치 부여
                if result in keyword_results:
                    current_score = result.get("relevance_score", 0.0) or result.get("score", 0.0)
                    result["relevance_score"] = current_score * 1.2
                merged.append(result)
        
        # relevance_score 기준 정렬
        merged.sort(key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0), reverse=True)
        return merged[:limit]
    
    def _rerank_multi_query_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """멀티 질의 결과 리랭킹"""
        for result in results:
            base_score = result.get("relevance_score", 0.0) or result.get("score", 0.0)
            # 원본 질문과의 매칭도 추가 점수
            text = str(result.get("text", "") or result.get("content", ""))
            if original_query in text:
                base_score *= 1.3
            result["final_score"] = base_score
        
        # 최종 점수 기준 정렬
        results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return results
    
    def _initialize_agent(self):
        """에이전트 초기화"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("❌ LangChain not available. Cannot initialize agent.")
            return False
        
        if not self.llm:
            if self.workflow:
                self.llm = getattr(self.workflow, 'llm', None)
            if not self.llm:
                self.logger.error("❌ LLM not available. Cannot initialize agent.")
                return False
        
        # 검색 도구 생성
        self.search_tools = [
            self._create_postgresql_keyword_search_tool(),
            self._create_vector_index_search_tool(),
            self._create_hybrid_search_tool(),
            self._create_multi_query_search_tool()
        ]
        
        # 에이전트 프롬프트
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 전문 법률 검색 에이전트입니다. 사용자의 법률 질문에 대해 가장 효과적인 검색 전략을 수립하고 실행합니다.

당신의 역할:
1. 사용자의 법률 질문을 분석하여 적절한 검색 전략 수립
2. 다음 검색 도구 중에서 적절한 것을 선택하여 사용:
   - search_postgresql_keywords: 정확한 키워드, 법령명, 조문번호 검색
   - search_vector_index: 의미 기반 유사 문서 검색
   - search_hybrid: 키워드 + 벡터 통합 검색
   - search_multi_query: 복잡한 질문을 여러 하위 질문으로 분해하여 검색 (가장 강력함)

검색 전략 가이드:
- **단순하고 명확한 질문** (예: "민법 제750조") → search_postgresql_keywords
- **의미 기반 검색이 필요한 질문** (예: "계약 해지 사유") → search_vector_index
- **정확성과 포괄성 모두 필요** → search_hybrid
- **복잡하고 다면적인 질문** (예: "계약 해지에 대해 알려주세요") → search_multi_query (권장)

중요 원칙:
- 복잡한 질문은 search_multi_query를 우선 사용
- 단일 검색으로 부족하면 여러 도구를 순차적으로 사용 가능
- 불필요한 중복 검색은 피함
- 검색 결과가 부족하면 다른 도구로 재검색
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        try:
            agent = create_openai_tools_agent(self.llm, self.search_tools, agent_prompt)
            self.agentic_agent = AgentExecutor(
                agent=agent,
                tools=self.search_tools,
                verbose=True,
                max_iterations=3,
                max_execution_time=30,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            self.logger.debug("✅ Multi-Query Search Agent initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agent: {e}", exc_info=True)
            return False
    
    def execute(self, state: LegalWorkflowState) -> LegalWorkflowState:
        """에이전트 실행"""
        try:
            start_time = time.time()
            query = state.get("query", "") or (state.get("input", {}) or {}).get("query", "")
            
            if not query:
                self.logger.error("❌ [MULTI-QUERY-AGENT] No query found in state")
                state.setdefault("search", {})["results"] = []
                state.setdefault("retrieved_docs", [])
                return state
            
            # 에이전트 초기화 (지연 초기화)
            if self.agentic_agent is None:
                if not self._initialize_agent():
                    # 에이전트 초기화 실패 시 폴백
                    self.logger.warning("⚠️ [MULTI-QUERY-AGENT] Agent initialization failed, using direct multi-query search")
                    return self._execute_direct_multi_query(state, query)
            
            # 에이전트 실행
            try:
                result = self.agentic_agent.invoke({"input": query})
                
                # 결과 파싱 및 state 업데이트
                search_results = self._parse_agent_results(result)
                
                # retrieved_docs 형식으로 변환
                retrieved_docs = self._convert_to_retrieved_docs(search_results)
                
                state["retrieved_docs"] = retrieved_docs
                state["search"] = {
                    "results": search_results,
                    "total_results": len(search_results)
                }
                
                processing_time = time.time() - start_time
                self.logger.info(f"✅ [MULTI-QUERY-AGENT] Completed in {processing_time:.2f}s, {len(retrieved_docs)} docs")
                
            except Exception as e:
                self.logger.warning(f"⚠️ [MULTI-QUERY-AGENT] Agent execution failed: {e}, using direct multi-query search")
                return self._execute_direct_multi_query(state, query)
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ [MULTI-QUERY-AGENT] Error: {e}", exc_info=True)
            state.setdefault("search", {})["results"] = []
            state.setdefault("retrieved_docs", [])
            return state
    
    def _get_source_types_from_query_type(self, query_type: Optional[str]) -> Optional[List[str]]:
        """
        질의 타입에 따라 검색할 문서 타입 결정
        
        Args:
            query_type: 질의 타입 (law_inquiry, precedent_search, general_question 등)
        
        Returns:
            검색할 문서 타입 리스트 (None이면 모든 타입 검색)
        """
        if not query_type:
            return None
        
        query_type_lower = query_type.lower()
        
        # 질의 타입별 문서 타입 매핑
        type_mapping = {
            "law_inquiry": ["statute_article"],  # 법령 질의 → 법령 조문만 검색
            "precedent_search": ["precedent_content"],  # 판례 검색 → 판례만 검색
            "general_question": None,  # 일반 질의 → 모든 타입 검색
            "legal_advice": None,  # 법률 조언 → 모든 타입 검색
        }
        
        source_types = type_mapping.get(query_type_lower)
        
        if source_types:
            self.logger.info(f"🔍 [SEARCH TYPE FILTER] 질의 타입 '{query_type}' → 문서 타입: {source_types}")
        else:
            self.logger.info(f"🔍 [SEARCH TYPE FILTER] 질의 타입 '{query_type}' → 모든 타입 검색")
        
        return source_types
    
    def _search_keywords_with_type_filter(
        self, 
        query: str, 
        source_types: Optional[List[str]], 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        타입 필터링을 적용한 키워드 검색
        
        Args:
            query: 검색 쿼리
            source_types: 검색할 문서 타입 리스트 (None이면 모든 타입)
            limit: 최대 결과 수
        
        Returns:
            검색 결과 리스트
        """
        if not self.keyword_search:
            return []
        
        # source_types가 지정된 경우 해당 타입만 검색
        if source_types:
            results = []
            for doc_type in source_types:
                if doc_type == "statute_article":
                    # 법령 조문 검색
                    statute_results = self.keyword_search.search_statutes_fts(query, limit=limit)
                    results.extend(statute_results)
                elif doc_type == "precedent_content":
                    # 판례 검색
                    case_results = self.keyword_search.search_cases_fts(query, limit=limit)
                    results.extend(case_results)
            
            # 중복 제거 및 정렬
            seen_ids = set()
            unique_results = []
            for doc in results:
                doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("document_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(doc)
            
            # relevance_score 기준 정렬
            unique_results.sort(
                key=lambda x: x.get("relevance_score", 0.0) or x.get("score", 0.0) or 0.0,
                reverse=True
            )
            
            return unique_results[:limit]
        else:
            # 모든 타입 검색
            return self.keyword_search.search_documents(query, limit=limit, force_fts=True)
    
    def _execute_direct_multi_query(self, state: LegalWorkflowState, query: str) -> LegalWorkflowState:
        """에이전트 없이 직접 멀티 질의 검색 실행 (폴백)"""
        try:
            self._initialize_search_engines()
            
            if not self.keyword_search or not self.semantic_search:
                self.logger.error("❌ [MULTI-QUERY] Search engines not available")
                state.setdefault("search", {})["results"] = []
                state.setdefault("retrieved_docs", [])
                return state
            
            # 🔥 개선: 질의 타입에 따라 검색할 문서 타입 결정
            query_type = None
            if self.workflow:
                # workflow에서 질의 타입 가져오기
                query_type_raw = self.workflow._get_state_value(state, "query_type", "")
                if query_type_raw:
                    if hasattr(query_type_raw, 'value'):
                        query_type = query_type_raw.value
                    else:
                        query_type = str(query_type_raw)
            
            source_types = self._get_source_types_from_query_type(query_type)
            
            # 멀티 질의 생성
            multi_queries = self._generate_multi_queries(query, max_queries=3)
            
            # 각 질문에 대해 검색
            all_results = []
            seen_doc_ids = set()
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for sub_query in multi_queries:
                    # 🔥 개선: source_types에 따라 키워드 검색도 필터링
                    # source_types가 지정된 경우 해당 타입만 검색
                    keyword_future = executor.submit(
                        self._search_keywords_with_type_filter,
                        sub_query, source_types, limit=5
                    )
                    # 🔥 개선: source_types 파라미터 전달
                    vector_future = executor.submit(
                        self.semantic_search.search,
                        sub_query, k=5, source_types=source_types
                    )
                    futures.append(("keyword", sub_query, keyword_future))
                    futures.append(("vector", sub_query, vector_future))
                
                for search_type, sub_query, future in futures:
                    try:
                        results = future.result(timeout=10.0)
                        for result in results:
                            doc_id = self._get_doc_id(result)
                            if doc_id and doc_id not in seen_doc_ids:
                                seen_doc_ids.add(doc_id)
                                result["sub_query"] = sub_query
                                result["search_type"] = search_type
                                result["original_query"] = query
                                all_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"⚠️ [MULTI-QUERY] Search failed for '{sub_query}': {e}")
            
            # 리랭킹
            ranked_results = self._rerank_multi_query_results(all_results, query)
            retrieved_docs = self._convert_to_retrieved_docs(ranked_results)
            
            # 🔥 LangGraph state 업데이트: 직접 설정 (LangGraph는 반환된 state를 병합함)
            # 최상위 레벨에 저장
            state["retrieved_docs"] = retrieved_docs
            state["semantic_results"] = ranked_results
            state["semantic_count"] = len(ranked_results)
            
            # 🔥 개선: search 그룹에도 저장 (State Reduction 손실 방지)
            if "search" not in state:
                state["search"] = {}
            state["search"]["retrieved_docs"] = retrieved_docs
            state["search"]["semantic_results"] = ranked_results
            state["search"]["semantic_count"] = len(ranked_results)
            
            # common 그룹에도 저장
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["retrieved_docs"] = retrieved_docs
            state["common"]["search"]["semantic_results"] = ranked_results
            state["common"]["search"]["semantic_count"] = len(ranked_results)
            
            # search 그룹에도 저장 (여러 위치에 저장하여 안전성 확보)
            if "search" not in state:
                state["search"] = {}
            state["search"]["results"] = ranked_results
            state["search"]["total_results"] = len(ranked_results)
            state["search"]["semantic_results"] = ranked_results
            state["search"]["semantic_count"] = len(ranked_results)
            
            # common 그룹에도 저장 (복구를 위해)
            if "common" not in state:
                state["common"] = {}
            if "search" not in state["common"]:
                state["common"]["search"] = {}
            state["common"]["search"]["semantic_results"] = ranked_results
            state["common"]["search"]["semantic_count"] = len(ranked_results)
            
            # 🔥 디버그: state 저장 확인
            self.logger.info(f"✅ [MULTI-QUERY] Direct search completed, {len(retrieved_docs)} docs")
            self.logger.info(f"📥 [MULTI-QUERY] State 저장 확인 - semantic_results: {len(state.get('semantic_results', []))}, search.results: {len(state.get('search', {}).get('results', []))}, search.semantic_results: {len(state.get('search', {}).get('semantic_results', []))}")
            return state
            
        except Exception as e:
            self.logger.error(f"❌ [MULTI-QUERY] Direct search error: {e}", exc_info=True)
            state.setdefault("search", {})["results"] = []
            state.setdefault("retrieved_docs", [])
            return state
    
    def _parse_agent_results(self, agent_result: Dict) -> List[Dict]:
        """에이전트 결과 파싱"""
        search_results = []
        
        if "intermediate_steps" in agent_result:
            for step in agent_result["intermediate_steps"]:
                action, observation = step
                tool_name = action.tool if hasattr(action, 'tool') else str(action)
                
                if tool_name in ["search_postgresql_keywords", "search_vector_index", 
                                "search_hybrid", "search_multi_query"]:
                    try:
                        tool_result = json.loads(observation)
                        if tool_result.get("success") and tool_result.get("results"):
                            search_results.extend(tool_result["results"])
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to parse tool result: {e}")
        
        # 중복 제거
        seen_ids = set()
        unique_results = []
        for result in search_results:
            doc_id = self._get_doc_id(result)
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        return unique_results
    
    def _convert_to_retrieved_docs(self, search_results: List[Dict]) -> List[Dict]:
        """검색 결과를 retrieved_docs 형식으로 변환"""
        retrieved_docs = []
        for result in search_results:
            doc = {
                "text": result.get("text", "") or result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "source": result.get("source", ""),
                "relevance_score": result.get("relevance_score", 0.0) or result.get("score", 0.0),
                "search_type": result.get("search_type", "unknown"),
                "sub_query": result.get("sub_query", ""),
                "original_query": result.get("original_query", "")
            }
            retrieved_docs.append(doc)
        
        return retrieved_docs

