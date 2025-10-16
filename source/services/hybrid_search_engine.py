"""
하이브리드 검색 엔진
정확한 매칭과 의미적 검색을 결합한 통합 검색 시스템
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from services.exact_search_engine import ExactSearchEngine
from services.semantic_search_engine import SemanticSearchEngine
from services.result_merger import ResultMerger, ResultRanker

logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """하이브리드 검색 엔진"""
    
    def __init__(self, 
                 db_path: str = "data/lawfirm.db",
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss",
                 metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json"):
        
        self.exact_search = ExactSearchEngine(db_path)
        self.semantic_search = SemanticSearchEngine(model_name, index_path, metadata_path)
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()
        
        # 검색 설정
        self.search_config = {
            "exact_search_weight": 0.6,
            "semantic_search_weight": 0.4,
            "max_results": 50,
            "semantic_threshold": 0.3,
            "diversity_max_per_type": 5
        }
    
    def search(self, 
               query: str, 
               search_types: List[str] = None,
               max_results: int = None,
               include_exact: bool = True,
               include_semantic: bool = True) -> Dict[str, Any]:
        """하이브리드 검색 실행"""
        try:
            logger.info(f"Starting hybrid search for query: '{query}'")
            
            if max_results is None:
                max_results = self.search_config["max_results"]
            
            if search_types is None:
                search_types = ["law", "precedent", "constitutional", "assembly_law"]
            
            # 검색 결과 수집
            exact_results = {}
            semantic_results = []
            
            # 정확한 매칭 검색
            if include_exact:
                exact_results = self._execute_exact_search(query, search_types)
            
            # 의미적 검색
            if include_semantic:
                semantic_results = self._execute_semantic_search(query, search_types)
            
            # 결과 통합
            merged_results = self.result_merger.merge_results(
                exact_results, semantic_results, query
            )
            
            # 결과 랭킹
            ranked_results = self.result_ranker.rank_results(merged_results, query)
            
            # 다양성 필터 적용
            filtered_results = self.result_ranker.apply_diversity_filter(
                ranked_results, self.search_config["diversity_max_per_type"]
            )
            
            # 최종 결과 제한
            final_results = filtered_results[:max_results]
            
            # 검색 통계
            search_stats = self._generate_search_stats(
                exact_results, semantic_results, final_results
            )
            
            result = {
                "query": query,
                "results": final_results,
                "total_results": len(final_results),
                "search_stats": search_stats,
                "search_config": self.search_config
            }
            
            logger.info(f"Hybrid search completed: {len(final_results)} results")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_stats": {}
            }
    
    def _execute_exact_search(self, query: str, search_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """정확한 매칭 검색 실행"""
        try:
            logger.info("Executing exact search")
            
            # 쿼리 파싱
            parsed_query = self.exact_search.parse_query(query)
            
            exact_results = {}
            
            for search_type in search_types:
                if search_type == "law":
                    results = self.exact_search.search_laws(
                        query=parsed_query["raw_query"],
                        law_name=parsed_query["law_name"],
                        article_number=parsed_query["article_number"]
                    )
                    exact_results["law"] = results
                    
                elif search_type == "precedent":
                    results = self.exact_search.search_precedents(
                        query=parsed_query["raw_query"],
                        case_number=parsed_query["case_number"],
                        court_name=parsed_query["court_name"]
                    )
                    exact_results["precedent"] = results
                    
                elif search_type == "constitutional":
                    results = self.exact_search.search_constitutional_decisions(
                        query=parsed_query["raw_query"],
                        case_number=parsed_query["case_number"]
                    )
                    exact_results["constitutional"] = results
                    
                elif search_type == "assembly_law":
                    results = self.exact_search.search_assembly_laws(
                        query=parsed_query["raw_query"],
                        law_name=parsed_query["law_name"],
                        article_number=parsed_query["article_number"]
                    )
                    exact_results["assembly_law"] = results
            
            logger.info(f"Exact search completed: {sum(len(r) for r in exact_results.values())} results")
            return exact_results
            
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
            return {}
    
    def _execute_semantic_search(self, query: str, search_types: List[str]) -> List[Dict[str, Any]]:
        """의미적 검색 실행"""
        try:
            logger.info("Executing semantic search")
            
            semantic_results = []
            
            # 전체 의미적 검색
            all_semantic_results = self.semantic_search.search(
                query, 
                k=self.search_config["max_results"] * 2,
                threshold=self.search_config["semantic_threshold"]
            )
            
            # 타입별 필터링
            for result in all_semantic_results:
                if result.get("type") in search_types:
                    semantic_results.append(result)
            
            logger.info(f"Semantic search completed: {len(semantic_results)} results")
            return semantic_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _generate_search_stats(self, 
                              exact_results: Dict[str, List[Dict[str, Any]]],
                              semantic_results: List[Dict[str, Any]],
                              final_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """검색 통계 생성"""
        try:
            stats = {
                "exact_search": {
                    "total_results": sum(len(r) for r in exact_results.values()),
                    "by_type": {k: len(v) for k, v in exact_results.items()}
                },
                "semantic_search": {
                    "total_results": len(semantic_results),
                    "by_type": {}
                },
                "final_results": {
                    "total_results": len(final_results),
                    "by_type": {}
                }
            }
            
            # 의미적 검색 타입별 통계
            for result in semantic_results:
                doc_type = result.get("type", "unknown")
                stats["semantic_search"]["by_type"][doc_type] = \
                    stats["semantic_search"]["by_type"].get(doc_type, 0) + 1
            
            # 최종 결과 타입별 통계
            for result in final_results:
                doc_type = result.get("doc_type", "unknown")
                stats["final_results"]["by_type"][doc_type] = \
                    stats["final_results"]["by_type"].get(doc_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate search stats: {e}")
            return {}
    
    def search_laws_only(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """법령만 검색"""
        result = self.search(query, search_types=["law"], max_results=max_results)
        return result["results"]
    
    def search_precedents_only(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """판례만 검색"""
        result = self.search(query, search_types=["precedent"], max_results=max_results)
        return result["results"]
    
    def search_constitutional_only(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """헌재결정례만 검색"""
        result = self.search(query, search_types=["constitutional"], max_results=max_results)
        return result["results"]
    
    def get_similar_documents(self, doc_id: str, doc_type: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """유사 문서 검색"""
        try:
            logger.info(f"Searching similar documents for {doc_id}")
            
            similar_docs = self.semantic_search.get_similar_documents(doc_id, k)
            
            # 타입 필터링
            if doc_type:
                similar_docs = [doc for doc in similar_docs if doc.get("type") == doc_type]
            
            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Similar documents search failed: {e}")
            return []
    
    def build_index(self, documents: List[Dict[str, Any]]) -> bool:
        """벡터 인덱스 구축"""
        try:
            logger.info(f"Building index for {len(documents)} documents")
            
            success = self.semantic_search.build_index(documents)
            
            if success:
                logger.info("Index built successfully")
            else:
                logger.error("Index building failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 엔진 통계 정보"""
        try:
            stats = {
                "exact_search": {
                    "database_path": self.exact_search.db_path,
                    "tables": ["laws", "precedents", "constitutional_decisions"]
                },
                "semantic_search": self.semantic_search.get_index_stats(),
                "search_config": self.search_config
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}
    
    def update_search_config(self, config: Dict[str, Any]):
        """검색 설정 업데이트"""
        try:
            self.search_config.update(config)
            logger.info("Search config updated")
        except Exception as e:
            logger.error(f"Failed to update search config: {e}")
    
    def test_search(self, test_queries: List[str]) -> Dict[str, Any]:
        """검색 시스템 테스트"""
        try:
            logger.info(f"Testing search system with {len(test_queries)} queries")
            
            test_results = {}
            
            for query in test_queries:
                result = self.search(query, max_results=10)
                test_results[query] = {
                    "total_results": result["total_results"],
                    "search_stats": result["search_stats"],
                    "success": "error" not in result
                }
            
            logger.info("Search system test completed")
            return test_results
            
        except Exception as e:
            logger.error(f"Search system test failed: {e}")
            return {}
