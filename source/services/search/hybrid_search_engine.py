# -*- coding: utf-8 -*-
"""
하이브리드 검색 엔진
정확한 매칭과 의미적 검색을 결합한 통합 검색 시스템
질문 유형별 가중치를 적용한 지능형 검색
병렬 처리, 캐싱, 결과 제한을 통한 성능 최적화
"""

import logging
import time
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
# from .exact_search_engine import ExactSearchEngine
from .semantic_search_engine import SemanticSearchEngine
# from .result_merger import ResultMerger, ResultRanker
# from .question_classifier import QuestionClassifier, QuestionType, QuestionClassification
from .precedent_search_engine import PrecedentSearchEngine, PrecedentSearchResult

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    source_type: str
    metadata: Dict[str, Any]
    search_method: str

@dataclass
class OptimizedSearchConfig:
    """최적화된 검색 설정"""
    max_results_per_type: int = 5  # 각 검색 타입별 최대 결과 수
    parallel_search: bool = True    # 병렬 검색 사용
    cache_enabled: bool = True     # 캐싱 사용
    timeout_seconds: float = 3.0   # 검색 타임아웃
    min_score_threshold: float = 0.3  # 최소 점수 임계값

class SearchCache:
    """검색 결과 캐시"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self.lock = threading.RLock()

    def _generate_key(self, query: str, question_type: str, max_results: int) -> str:
        """캐시 키 생성"""
        key_string = f"{query.lower().strip()}:{question_type}:{max_results}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, query: str, question_type: str, max_results: int) -> Optional[List[SearchResult]]:
        """캐시에서 결과 가져오기"""
        with self.lock:
            key = self._generate_key(query, question_type, max_results)
            if key in self.cache:
                results, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return results
                else:
                    del self.cache[key]
            return None

    def put(self, query: str, question_type: str, max_results: int, results: List[SearchResult]):
        """캐시에 결과 저장"""
        with self.lock:
            key = self._generate_key(query, question_type, max_results)

            # 캐시 크기 제한
            if len(self.cache) >= self.max_size:
                # 가장 오래된 항목 제거
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (results, time.time())

class HybridSearchEngine:
    """하이브리드 검색 엔진"""

    def __init__(self,
                 db_path: str = "data/lawfirm.db",
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 index_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.faiss",
                 metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta/ml_enhanced_faiss_index.json",
                 precedent_index_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.faiss",
                 precedent_metadata_path: str = "data/embeddings/ml_enhanced_ko_sroberta_precedents/ml_enhanced_faiss_index.json",
                 config: Optional[OptimizedSearchConfig] = None):

        self.config = config or OptimizedSearchConfig()

        self.exact_search = ExactSearchEngine(db_path)
        self.semantic_search = SemanticSearchEngine(
            model_name=model_name,
            index_path=index_path,
            metadata_path=metadata_path
        )
        self.question_classifier = QuestionClassifier()
        self.precedent_search = PrecedentSearchEngine(
            db_path=db_path,
            vector_index_path=precedent_index_path,
            vector_metadata_path=precedent_metadata_path
        )
        self.result_merger = ResultMerger()
        self.result_ranker = ResultRanker()

        # 캐시 초기화
        self.cache = SearchCache() if self.config.cache_enabled else None

        # 스레드 풀 초기화
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 성능 통계
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'parallel_searches': 0,
            'avg_search_time': 0.0,
            'total_search_time': 0.0
        }

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
        """하이브리드 검색 실행 (캐싱 지원)"""
        start_time = time.time()
        self.stats['total_searches'] += 1

        try:
            logger.info(f"Starting hybrid search for query: '{query}'")

            if max_results is None:
                max_results = self.search_config["max_results"]

            if search_types is None:
                search_types = ["law", "precedent", "constitutional", "assembly_law"]

            # 캐시 확인
            cache_key = f"{query}:{':'.join(search_types)}:{max_results}"
            if self.cache:
                cached_results = self.cache.get(query, cache_key, max_results)
                if cached_results:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Cache hit for query: '{query}'")
                    return self._format_cached_results(cached_results, query)

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
                exact_results, semantic_results
            )

            # 결과 랭킹
            ranked_results = self.result_ranker.rank_results(merged_results)

            # 다양성 필터 적용
            filtered_results = self.result_ranker.apply_diversity_filter(
                ranked_results, self.search_config["diversity_max_per_type"]
            )

            # 최종 결과 제한
            final_results = filtered_results[:max_results]

            # MergedResult를 Dict로 변환
            final_results_dict = []
            for result in final_results:
                if hasattr(result, 'text'):
                    final_results_dict.append({
                        'text': result.text,
                        'score': result.score,
                        'source': result.source,
                        'metadata': result.metadata
                    })
                else:
                    final_results_dict.append(result)

            # 캐시 저장
            if self.cache:
                search_results = [SearchResult(
                    content=r.get('text', str(r)),
                    score=r.get('score', 1.0),
                    source_type=r.get('source', 'unknown'),
                    metadata=r.get('metadata', {}),
                    search_method='hybrid'
                ) for r in final_results_dict]
                self.cache.put(query, cache_key, max_results, search_results)

            # 통계 업데이트
            search_time = time.time() - start_time
            self.stats['total_search_time'] += search_time
            self.stats['avg_search_time'] = (
                self.stats['total_search_time'] / self.stats['total_searches']
            )

            # 검색 통계
            search_stats = self._generate_search_stats(
                exact_results, semantic_results, final_results_dict
            )
            search_stats['processing_time'] = search_time
            search_stats['cache_hit'] = False

            result = {
                "query": query,
                "results": final_results_dict,
                "total_results": len(final_results_dict),
                "search_stats": search_stats,
                "search_config": self.search_config
            }

            logger.info(f"Hybrid search completed: {len(final_results_dict)} results")
            return result

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_stats": {
                    "processing_time": time.time() - start_time,
                    "cache_hit": False
                }
            }

    def _format_cached_results(self, cached_results: List[SearchResult], query: str) -> Dict[str, Any]:
        """캐시된 결과를 표준 형식으로 변환"""
        final_results_dict = []
        for result in cached_results:
            final_results_dict.append({
                'text': result.content,
                'score': result.score,
                'source': result.source_type,
                'metadata': result.metadata
            })

        return {
            "query": query,
            "results": final_results_dict,
            "total_results": len(final_results_dict),
            "search_stats": {
                "processing_time": 0.0,
                "cache_hit": True
            },
            "search_config": self.search_config
        }

    def search_with_question_type(self,
                                 query: str,
                                 question_type: Optional[str] = None,
                                 max_results: int = None) -> Dict[str, Any]:
        """
        질문 유형을 고려한 지능형 검색

        Args:
            query: 검색 쿼리
            question_type: 질문 분류 결과 (None이면 자동 분류)
            max_results: 최대 결과 수

        Returns:
            Dict[str, Any]: 검색 결과
        """
        try:
            logger.info(f"Starting intelligent search for query: '{query}'")

            # 질문 분류
            if question_type is None:
                question_type = self.question_classifier.classify_question(query)

            logger.info(f"Question classified as: {question_type.question_type.value} "
                       f"(law_weight={question_type.law_weight}, precedent_weight={question_type.precedent_weight})")

            if max_results is None:
                max_results = self.search_config["max_results"]

            # 법률 검색
            law_results = self._search_laws_with_weight(query, question_type.law_weight, max_results)

            # 판례 검색 (민사 판례 우선)
            precedent_results = self._search_precedents_with_weight(query, question_type.precedent_weight, max_results)

            # 결과 통합 및 재랭킹
            merged_results = self._merge_and_rerank_results(law_results, precedent_results, question_type)

            # 최종 결과 제한
            final_results = merged_results[:max_results]

            # 검색 통계
            search_stats = self._generate_intelligent_search_stats(
                law_results, precedent_results, final_results, question_type
            )

            result = {
                "query": query,
                "question_type": question_type.question_type.value,
                "question_classification": {
                    "type": question_type.question_type.value,
                    "law_weight": question_type.law_weight,
                    "precedent_weight": question_type.precedent_weight,
                    "confidence": question_type.confidence,
                    "keywords": question_type.keywords
                },
                "results": final_results,
                "law_results": law_results,
                "precedent_results": precedent_results,
                "total_results": len(final_results),
                "search_stats": search_stats
            }

            logger.info(f"Intelligent search completed: {len(final_results)} results")
            return result

        except Exception as e:
            logger.error(f"Intelligent search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "error": str(e),
                "search_stats": {}
            }

    def _search_laws_with_weight(self, query: str, weight: float, max_results: int) -> List[Dict[str, Any]]:
        """가중치를 적용한 법률 검색"""
        try:
            if weight <= 0:
                return []

            # 정확한 매칭 검색
            exact_results = self._execute_exact_search(query, ["law", "assembly_law"])
            law_results = exact_results.get("law", []) + exact_results.get("assembly_law", [])

            # 의미적 검색
            semantic_results = self._execute_semantic_search(query, ["law", "assembly_law"])

            # 결과 통합
            all_law_results = law_results + semantic_results

            # 가중치 적용
            for result in all_law_results:
                original_score = result.get("similarity", 0.5)
                result["similarity"] = original_score * weight
                result["search_type"] = "law"

            # 정렬 및 제한
            sorted_results = sorted(all_law_results, key=lambda x: x.get("similarity", 0), reverse=True)
            return sorted_results[:max_results]

        except Exception as e:
            logger.error(f"Error searching laws with weight: {e}")
            return []

    def _search_precedents_with_weight(self, query: str, weight: float, max_results: int) -> List[Dict[str, Any]]:
        """가중치를 적용한 판례 검색"""
        try:
            if weight <= 0:
                return []

            # 판례 검색 엔진 사용
            precedent_results = self.precedent_search.search_precedents(
                query=query,
                category='civil',  # 민사 판례 우선
                top_k=max_results,
                search_type='hybrid'
            )

            # PrecedentSearchResult를 Dict로 변환
            dict_results = []
            for result in precedent_results:
                dict_result = {
                    'text': result.summary or result.case_name,
                    'similarity': result.similarity,
                    'score': result.similarity,
                    'type': 'precedent',
                    'source': f"{result.court} {result.case_number}",
                    'metadata': {
                        'case_id': result.case_id,
                        'case_name': result.case_name,
                        'case_number': result.case_number,
                        'court': result.court,
                        'decision_date': result.decision_date,
                        'category': result.category,
                        'field': result.field,
                        'summary': result.summary
                    },
                    'search_type': result.search_type,
                    'case_id': result.case_id,
                    'case_name': result.case_name,
                    'case_number': result.case_number,
                    'court': result.court
                }
                dict_results.append(dict_result)

            # 가중치 적용
            for result in dict_results:
                original_score = result.get("similarity", 0.5)
                result["similarity"] = original_score * weight
                result["search_type"] = "precedent"

            # 정렬 및 제한
            sorted_results = sorted(dict_results, key=lambda x: x.get("similarity", 0), reverse=True)
            return sorted_results[:max_results]

        except Exception as e:
            logger.error(f"Error searching precedents with weight: {e}")
            return []

    def _merge_and_rerank_results(self,
                                 law_results: List[Dict[str, Any]],
                                 precedent_results: List[Dict[str, Any]],
                                 question_type: str) -> List[Dict[str, Any]]:
        """결과 통합 및 재랭킹"""
        try:
            # 모든 결과 통합
            all_results = law_results + precedent_results

            # 중복 제거 (ID 기반)
            unique_results = {}
            for result in all_results:
                result_id = result.get("case_id") or result.get("article_id") or result.get("id")
                if result_id and result_id not in unique_results:
                    unique_results[result_id] = result
                elif result_id and result_id in unique_results:
                    # 더 높은 점수 유지
                    if result.get("similarity", 0) > unique_results[result_id].get("similarity", 0):
                        unique_results[result_id] = result

            # 질문 유형별 추가 가중치 적용
            for result in unique_results.values():
                result_type = result.get("type", "unknown")

                if result_type == "law" and question_type.question_type == QuestionType.LAW_INQUIRY:
                    result["similarity"] *= 1.2  # 법률 조회 질문에 법률 결과 가중치 증가
                elif result_type == "precedent" and question_type.question_type == QuestionType.PRECEDENT_SEARCH:
                    result["similarity"] *= 1.2  # 판례 검색 질문에 판례 결과 가중치 증가
                elif result_type == "law" and question_type.question_type == QuestionType.LEGAL_ADVICE:
                    result["similarity"] *= 1.1  # 법적 조언 질문에 법률 결과 약간 가중치 증가
                elif result_type == "precedent" and question_type.question_type == QuestionType.LEGAL_ADVICE:
                    result["similarity"] *= 1.1  # 법적 조언 질문에 판례 결과 약간 가중치 증가

            # 최종 정렬
            final_results = sorted(unique_results.values(), key=lambda x: x.get("similarity", 0), reverse=True)

            return final_results

        except Exception as e:
            logger.error(f"Error merging and reranking results: {e}")
            return law_results + precedent_results

    def _generate_intelligent_search_stats(self,
                                         law_results: List[Dict[str, Any]],
                                         precedent_results: List[Dict[str, Any]],
                                         final_results: List[Dict[str, Any]],
                                         question_type: str) -> Dict[str, Any]:
        """지능형 검색 통계 생성"""
        try:
            stats = {
                "question_classification": {
                    "type": question_type.question_type.value,
                    "law_weight": question_type.law_weight,
                    "precedent_weight": question_type.precedent_weight,
                    "confidence": question_type.confidence
                },
                "law_search": {
                    "total_results": len(law_results),
                    "avg_similarity": sum(r.get("similarity", 0) for r in law_results) / len(law_results) if law_results else 0
                },
                "precedent_search": {
                    "total_results": len(precedent_results),
                    "avg_similarity": sum(r.get("similarity", 0) for r in precedent_results) / len(precedent_results) if precedent_results else 0
                },
                "final_results": {
                    "total_results": len(final_results),
                    "law_count": len([r for r in final_results if r.get("type") == "law"]),
                    "precedent_count": len([r for r in final_results if r.get("type") == "precedent"]),
                    "avg_similarity": sum(r.get("similarity", 0) for r in final_results) / len(final_results) if final_results else 0
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Error generating intelligent search stats: {e}")
            return {}

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
                k=self.search_config["max_results"] * 2
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
