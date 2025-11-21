# -*- coding: utf-8 -*-
"""
검색 품질 평가 스크립트
개선 전후 성능 비교 및 MLflow 추적
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

# 프로젝트 경로 설정
script_dir = Path(__file__).parent
scripts_dir = script_dir.parent
tests_dir = scripts_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

# sys.path 설정
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# UTF-8 인코딩 설정 (Windows PowerShell 호환)
_original_stdout = sys.stdout
_original_stderr = sys.stderr

if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    if hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (ValueError, AttributeError):
            pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# SafeStreamHandler 클래스 정의
class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream, original_stdout_ref=None):
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
    
    def _get_safe_stream(self):
        """안전한 스트림 반환"""
        streams_to_try = []
        if self.stream and hasattr(self.stream, 'write'):
            streams_to_try.append(self.stream)
        if self._original_stdout and hasattr(self._original_stdout, 'write'):
            streams_to_try.append(self._original_stdout)
        
        for stream in streams_to_try:
            try:
                if hasattr(stream, 'buffer'):
                    try:
                        buffer = stream.buffer
                        if buffer is not None:
                            return stream
                    except (ValueError, AttributeError):
                        continue
                else:
                    return stream
            except (ValueError, AttributeError):
                continue
        
        return None
    
    def emit(self, record):
        """안전한 로그 출력 (버퍼 분리 오류 방지)"""
        try:
            msg = self.format(record) + self.terminator
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    safe_stream.write(msg)
                    try:
                        safe_stream.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError) as e:
                    if "detached" not in str(e).lower():
                        pass
        except Exception:
            pass

# 로깅 설정
logger = get_logger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

# SafeStreamHandler 사용
safe_handler = SafeStreamHandler(sys.stdout, _original_stdout)
safe_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
safe_handler.setFormatter(formatter)
logger.addHandler(safe_handler)

try:
    from lawfirm_langgraph.core.search.mlflow_tracker import SearchQualityTracker
    MLFLOW_AVAILABLE = True
except ImportError as e:
    try:
        from core.search.mlflow_tracker import SearchQualityTracker
        MLFLOW_AVAILABLE = True
    except ImportError as e2:
        logger.warning(f"MLflow tracker import error: {e}, {e2}")
        MLFLOW_AVAILABLE = False

try:
    from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
    from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Primary import error: {e}")
    try:
        from config.langgraph_config import LangGraphConfig
        from core.workflow.workflow_service import LangGraphWorkflowService
        IMPORTS_AVAILABLE = True
    except ImportError as e2:
        logger.error(f"All import attempts failed: {e}, {e2}")
        IMPORTS_AVAILABLE = False


# 테스트 쿼리 세트
TEST_QUERIES = {
    "statute_article": [
        "민법 제1조의 내용은 무엇인가요?",
        "형법 제250조 살인죄에 대해 알려주세요",
        "상법 제1조의 적용 범위는?",
        "노동기본법 제3조의 내용은?",
        "가족법 제777조 이혼 사유에 대해 설명해주세요",
        "행정법 제1조의 원칙은?",
        "헌법 제10조 인간의 존엄과 가치에 대해 알려주세요",
        "민사소송법 제1조의 목적은?",
        "형사소송법 제1조의 목적은?",
        "공정거래법 제1조의 목적은?",
        "소비자기본법 제1조의 목적은?",
        "개인정보보호법 제1조의 목적은?",
        "정보통신망법 제1조의 목적은?",
        "부동산등기법 제1조의 목적은?",
        "상속세법 제1조의 목적은?",
        "소득세법 제1조의 목적은?",
        "법인세법 제1조의 목적은?",
        "부가가치세법 제1조의 목적은?",
        "관세법 제1조의 목적은?",
        "조세특례제한법 제1조의 목적은?"
    ],
    "precedent": [
        "계약 해지 사유에 대한 대법원 판례를 알려주세요",
        "손해배상 책임에 대한 판례를 찾아주세요",
        "불법행위 성립 요건에 대한 판례는?",
        "계약 위반에 대한 대법원 판결을 알려주세요",
        "부당이득 반환에 대한 판례를 찾아주세요",
        "소유권 이전에 대한 판례는?",
        "임대차 보증금 반환에 대한 판례를 알려주세요",
        "근로계약 해지에 대한 판례는?",
        "명예훼손에 대한 판례를 찾아주세요",
        "배임죄에 대한 대법원 판결은?",
        "횡령죄에 대한 판례를 알려주세요",
        "사기죄에 대한 판례는?",
        "강도죄에 대한 판례를 찾아주세요",
        "절도죄에 대한 판례는?",
        "상해죄에 대한 판례를 알려주세요",
        "과실치사상죄에 대한 판례는?",
        "교통사고 배상에 대한 판례를 찾아주세요",
        "의료사고 배상에 대한 판례는?",
        "제조물 책임에 대한 판례를 알려주세요",
        "환경오염 배상에 대한 판례는?"
    ],
    "procedure": [
        "민사소송 절차는 어떻게 진행되나요?",
        "형사소송 절차를 설명해주세요",
        "가사소송 절차는 어떻게 되나요?",
        "행정소송 절차를 알려주세요",
        "조정 절차는 어떻게 진행되나요?",
        "중재 절차를 설명해주세요",
        "집행 절차는 어떻게 되나요?",
        "가압류 절차를 알려주세요",
        "가처분 절차는 어떻게 진행되나요?",
        "경매 절차를 설명해주세요",
        "공탁 절차는 어떻게 되나요?",
        "상속 절차를 알려주세요",
        "이혼 절차는 어떻게 진행되나요?",
        "입양 절차를 설명해주세요",
        "성년후견 절차는 어떻게 되나요?",
        "파산 절차를 알려주세요",
        "회생 절차는 어떻게 진행되나요?",
        "화해 절차를 설명해주세요",
        "조정 신청 절차는 어떻게 되나요?",
        "재심 절차를 알려주세요"
    ],
    "general_question": [
        "계약서 작성 시 주의사항은?",
        "임대차 계약에서 중요한 사항은?",
        "근로계약서 필수 조항은?",
        "이혼 시 재산 분할은 어떻게 되나요?",
        "상속 순위는 어떻게 되나요?",
        "손해배상 청구 방법은?",
        "명예훼손이 성립하는 조건은?",
        "개인정보 유출 시 대응 방법은?",
        "온라인 거래 시 소비자 권리는?",
        "부동산 매매 시 주의사항은?",
        "전세 보증금 반환 청구 방법은?",
        "근로자의 휴가권은?",
        "해고 사유는 무엇인가요?",
        "임금 체불 시 대응 방법은?",
        "산업재해 인정 요건은?",
        "의료사고 발생 시 대응은?",
        "교통사고 처리 절차는?",
        "범죄 신고 방법은?",
        "증인 출석 의무는?",
        "변호사 선임 시 고려사항은?"
    ]
}


def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    """진행 바 포맷팅"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


class SearchQualityEvaluator:
    """검색 품질 평가기"""
    
    def __init__(self, enable_improvements: bool = True):
        """
        초기화
        
        Args:
            enable_improvements: 개선 기능 활성화 여부
        """
        if not IMPORTS_AVAILABLE:
            raise ImportError("Required imports not available. Please check your environment.")
        
        self.enable_improvements = enable_improvements
        
        # Config 로드
        try:
            self.config = LangGraphConfig.from_env()
        except Exception:
            self.config = LangGraphConfig()
        
        # 개선 기능 설정
        if enable_improvements:
            self.config.enable_search_improvements = True
        else:
            self.config.enable_search_improvements = False
        
        # 체크포인트 비활성화 (테스트 모드)
        self.config.enable_checkpoint = False
        
        self.workflow_service = LangGraphWorkflowService(config=self.config)
        
        if MLFLOW_AVAILABLE:
            # mlflow/mlruns를 기본값으로 사용
            import os
            from pathlib import Path
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                project_root = Path(__file__).resolve().parent.parent.parent.parent
                default_mlruns = project_root / "mlflow" / "mlruns"
                if default_mlruns.exists():
                    tracking_uri = f"file:///{str(default_mlruns).replace(chr(92), '/')}"
                else:
                    tracking_uri = getattr(self.config, 'mlflow_tracking_uri', None)
            
            self.tracker = SearchQualityTracker(
                experiment_name="search_quality_improvement",
                tracking_uri=tracking_uri
            )
        else:
            self.tracker = None
            logger.warning("MLflow not available. Tracking will be disabled.")
    
    def calculate_precision_at_k(
        self,
        results: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """Precision@K 계산"""
        if not results or k == 0:
            return 0.0
        
        top_k = results[:k]
        relevant_count = sum(
            1 for doc in top_k
            if doc.get("id") in relevant_doc_ids or 
            any(rid in str(doc.get("source", "")) for rid in relevant_doc_ids)
        )
        
        return relevant_count / min(k, len(top_k))
    
    def calculate_recall_at_k(
        self,
        results: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 10
    ) -> float:
        """Recall@K 계산"""
        if not relevant_doc_ids:
            return 0.0
        
        top_k = results[:k]
        retrieved_relevant = sum(
            1 for doc in top_k
            if doc.get("id") in relevant_doc_ids or
            any(rid in str(doc.get("source", "")) for rid in relevant_doc_ids)
        )
        
        return retrieved_relevant / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    
    def calculate_ndcg_at_k(
        self,
        results: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 10
    ) -> float:
        """NDCG@K 계산 (간단한 버전)"""
        if not results or not relevant_doc_ids or k == 0:
            return 0.0
        
        try:
            import math
            top_k = results[:k]
            dcg = 0.0
            
            for i, doc in enumerate(top_k):
                is_relevant = (
                    doc.get("id") in relevant_doc_ids or
                    any(rid in str(doc.get("source", "")) for rid in relevant_doc_ids)
                )
                if is_relevant:
                    dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Ideal DCG 계산
            ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_doc_ids), k)))
            
            return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        except Exception:
            # Fallback: 간단한 관련성 점수
            top_k = results[:k]
            relevant_count = sum(
                1 for doc in top_k
                if doc.get("id") in relevant_doc_ids or
                any(rid in str(doc.get("source", "")) for rid in relevant_doc_ids)
            )
            return relevant_count / min(k, len(relevant_doc_ids)) if relevant_doc_ids else 0.0
    
    def calculate_keyword_coverage(
        self,
        results: List[Dict[str, Any]],
        query: str,
        extracted_keywords: Optional[List[str]] = None
    ) -> float:
        """Keyword Coverage 계산"""
        if not results:
            return 0.0
        
        keywords = extracted_keywords or query.split()
        query_words = set(kw.lower() for kw in keywords if len(kw) >= 2)
        
        if not query_words:
            return 0.0
        
        matched_keywords = set()
        for result in results[:10]:  # 상위 10개만 확인
            text = (result.get("text", "") or result.get("content", "")).lower()
            for keyword in query_words:
                if keyword in text:
                    matched_keywords.add(keyword)
        
        return len(matched_keywords) / len(query_words) if query_words else 0.0
    
    def calculate_diversity_score(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """다양성 점수 계산"""
        if not results:
            return 0.0
        
        # 출처 다양성
        sources = [result.get("source", "") for result in results]
        unique_sources = len(set(sources))
        source_diversity = unique_sources / len(results) if results else 0.0
        
        # 타입 다양성
        types = [result.get("type", "") for result in results]
        unique_types = len(set(types))
        type_diversity = unique_types / len(results) if results else 0.0
        
        return (source_diversity + type_diversity) / 2.0
    
    async def evaluate_query_async(
        self,
        query: str,
        query_type: str = "general_question",
        relevant_doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """단일 쿼리 평가 (비동기)"""
        start_time = time.time()
        
        try:
            # 검색 실행 (비동기)
            result = await self.workflow_service.process_query(
                query=query,
                session_id=f"eval_{int(time.time())}",
                enable_checkpoint=False
            )
            
            # 검색 결과 추출 (여러 경로 시도)
            retrieved_docs = result.get("retrieved_docs", [])
            
            # retrieved_docs가 비어있으면 다른 경로에서 시도
            if not retrieved_docs:
                # search 그룹에서 추출 시도
                if "search" in result and isinstance(result["search"], dict):
                    search_group = result["search"]
                    retrieved_docs = search_group.get("retrieved_docs", [])
                    if not retrieved_docs:
                        retrieved_docs = search_group.get("merged_documents", [])
                    if not retrieved_docs:
                        # semantic_results와 keyword_results 결합
                        semantic = search_group.get("semantic_results", [])
                        keyword = search_group.get("keyword_results", [])
                        if semantic or keyword:
                            combined = []
                            if isinstance(semantic, list):
                                combined.extend(semantic)
                            if isinstance(keyword, list):
                                combined.extend(keyword)
                            # 중복 제거
                            seen_ids = set()
                            for doc in combined:
                                doc_id = doc.get("id") or doc.get("content_id") or str(doc.get("content", ""))[:100]
                                if doc_id not in seen_ids:
                                    seen_ids.add(doc_id)
                                    retrieved_docs.append(doc)
            
            # common 그룹에서도 시도
            if not retrieved_docs:
                if "common" in result and isinstance(result["common"], dict):
                    common = result["common"]
                    if "search" in common and isinstance(common["search"], dict):
                        search = common["search"]
                        retrieved_docs = search.get("retrieved_docs", [])
            
            # 최상위 레벨에서도 시도
            if not retrieved_docs:
                semantic_results = result.get("semantic_results", [])
                keyword_results = result.get("keyword_results", [])
                if semantic_results or keyword_results:
                    combined = []
                    if isinstance(semantic_results, list):
                        combined.extend(semantic_results)
                    if isinstance(keyword_results, list):
                        combined.extend(keyword_results)
                    # 중복 제거
                    seen_ids = set()
                    for doc in combined:
                        doc_id = doc.get("id") or doc.get("content_id") or str(doc.get("content", ""))[:100]
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            retrieved_docs.append(doc)
            
            # extracted_keywords 추출 (여러 경로 시도)
            extracted_keywords = result.get("extracted_keywords", [])
            if not extracted_keywords:
                # metadata에서 추출 시도
                metadata = result.get("metadata", {})
                if isinstance(metadata, dict):
                    extracted_keywords = metadata.get("extracted_keywords", [])
            if not extracted_keywords:
                # search 그룹에서 추출 시도
                if "search" in result and isinstance(result["search"], dict):
                    extracted_keywords = result["search"].get("extracted_keywords", [])
            if not extracted_keywords:
                # common 그룹에서 추출 시도
                if "common" in result and isinstance(result["common"], dict):
                    common = result["common"]
                    if "search" in common and isinstance(common["search"], dict):
                        extracted_keywords = common["search"].get("extracted_keywords", [])
            
            elapsed_time = time.time() - start_time
            
            # 메트릭 계산
            metrics = {
                "query": query,
                "query_type": query_type,
                "result_count": len(retrieved_docs),
                "response_time": elapsed_time,
                "precision_at_5": self.calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids or [], k=5
                ),
                "precision_at_10": self.calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids or [], k=10
                ),
                "recall_at_5": self.calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids or [], k=5
                ),
                "recall_at_10": self.calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids or [], k=10
                ),
                "ndcg_at_10": self.calculate_ndcg_at_k(
                    retrieved_docs, relevant_doc_ids or [], k=10
                ),
                "keyword_coverage": self.calculate_keyword_coverage(
                    retrieved_docs, query, extracted_keywords
                ),
                "diversity_score": self.calculate_diversity_score(retrieved_docs),
                "avg_relevance": sum(
                    doc.get("relevance_score", doc.get("similarity", 0.0))
                    for doc in retrieved_docs[:10]
                ) / min(10, len(retrieved_docs)) if retrieved_docs else 0.0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Query evaluation failed: {e}")
            return {
                "query": query,
                "query_type": query_type,
                "error": str(e),
                "result_count": 0,
                "response_time": time.time() - start_time
            }
    
    def evaluate_query(
        self,
        query: str,
        query_type: str = "general_question",
        relevant_doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """단일 쿼리 평가 (동기 래퍼)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.evaluate_query_async(query, query_type, relevant_doc_ids)
        )
    
    async def evaluate_batch_async(
        self,
        test_queries: List[Dict[str, str]],
        experiment_name: str = "batch_evaluation"
    ) -> Dict[str, Any]:
        """배치 평가"""
        total_queries = len(test_queries)
        logger.info("=" * 60)
        logger.info(f"Starting batch evaluation: {total_queries} queries")
        logger.info("=" * 60)
        
        # 진행 상황 추적 변수
        start_time = time.time()
        query_times = []
        all_metrics = []
        failed_queries = []
        
        for i, test_query in enumerate(test_queries):
            query = test_query.get("query", "")
            query_type = test_query.get("type", "general_question")
            relevant_doc_ids = test_query.get("relevant_doc_ids", [])
            
            # 진행률 계산
            progress_pct = ((i + 1) / total_queries) * 100
            
            # 경과 시간 및 예상 남은 시간 계산
            elapsed_time = time.time() - start_time
            if i > 0:
                avg_time_per_query = elapsed_time / i
                remaining_queries = total_queries - (i + 1)
                estimated_remaining = avg_time_per_query * remaining_queries
            else:
                estimated_remaining = 0
            
            # 진행 상황 표시
            logger.info("")
            logger.info("-" * 60)
            progress_bar = format_progress_bar(i + 1, total_queries)
            logger.info(f"{progress_bar} {i+1}/{total_queries} ({progress_pct:.1f}%)")
            logger.info(f"[경과 시간] {elapsed_time:.1f}초")
            if estimated_remaining > 0:
                logger.info(f"[예상 남은 시간] {estimated_remaining:.1f}초")
            logger.info("-" * 60)
            logger.info(f"Evaluating query {i+1}/{total_queries}: {query[:50]}...")
            logger.info(f"Type: {query_type}")
            
            query_start_time = time.time()
            
            metrics = await self.evaluate_query_async(query, query_type, relevant_doc_ids)
            
            query_elapsed = time.time() - query_start_time
            query_times.append(query_elapsed)
            
            if "error" not in metrics:
                logger.info(f"  ✓ Success ({query_elapsed:.2f}초)")
                logger.info(f"  - Result count: {metrics.get('result_count', 0)}")
                logger.info(f"  - Response time: {metrics.get('response_time', 0):.2f}s")
                logger.info(f"  - Precision@5: {metrics.get('precision_at_5', 0):.3f}")
                all_metrics.append(metrics)
            else:
                logger.error(f"  ✗ Error: {metrics.get('error')} ({query_elapsed:.2f}초)")
                failed_queries.append({"query": query, "error": metrics.get("error")})
        
        # 최종 요약
        total_time = time.time() - start_time
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Batch evaluation completed!")
        logger.info("=" * 60)
        logger.info(f"총 소요 시간: {total_time:.2f}초")
        logger.info(f"평균 쿼리 처리 시간: {avg_query_time:.2f}초")
        logger.info(f"처리된 쿼리 수: {len(query_times)}/{total_queries}")
        logger.info(f"성공한 쿼리: {len(all_metrics)}/{total_queries}")
        logger.info(f"실패한 쿼리: {len(failed_queries)}/{total_queries}")
        if query_times:
            logger.info(f"최소 처리 시간: {min(query_times):.2f}초")
            logger.info(f"최대 처리 시간: {max(query_times):.2f}초")
        logger.info("=" * 60)
        
        # 평균 메트릭 계산
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if key not in ["query", "query_type"] and isinstance(all_metrics[0][key], (int, float)):
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                        avg_metrics[f"min_{key}"] = min(values)
                        avg_metrics[f"max_{key}"] = max(values)
            
            # MLflow 추적
            if MLFLOW_AVAILABLE and self.tracker:
                async def get_results_async(query: str, query_type: str):
                    result = await self.workflow_service.process_query(
                        query=query,
                        session_id=f"mlflow_{int(time.time())}",
                        enable_checkpoint=False
                    )
                    retrieved_docs = result.get("retrieved_docs", [])
                    if not retrieved_docs:
                        if "search" in result and isinstance(result["search"], dict):
                            search_group = result["search"]
                            retrieved_docs = search_group.get("retrieved_docs", [])
                            if not retrieved_docs:
                                retrieved_docs = search_group.get("merged_documents", [])
                            if not retrieved_docs:
                                semantic = search_group.get("semantic_results", [])
                                keyword = search_group.get("keyword_results", [])
                                if semantic or keyword:
                                    combined = []
                                    if isinstance(semantic, list):
                                        combined.extend(semantic)
                                    if isinstance(keyword, list):
                                        combined.extend(keyword)
                                    seen_ids = set()
                                    for doc in combined:
                                        doc_id = doc.get("id") or doc.get("content_id") or str(doc.get("content", ""))[:100]
                                        if doc_id not in seen_ids:
                                            seen_ids.add(doc_id)
                                            retrieved_docs.append(doc)
                    return retrieved_docs
                
                await self.tracker.track_batch_experiment(
                    test_queries=test_queries,
                    feature_name=experiment_name,
                    params={
                        "enable_improvements": self.enable_improvements,
                        "test_queries_count": len(test_queries),
                        "failed_queries_count": len(failed_queries)
                    },
                    results_func=get_results_async
                )
            
            return {
                "experiment_name": experiment_name,
                "enable_improvements": self.enable_improvements,
                "total_queries": len(test_queries),
                "successful_queries": len(all_metrics),
                "failed_queries": len(failed_queries),
                "total_time_seconds": total_time,
                "avg_query_time_seconds": avg_query_time,
                "min_query_time_seconds": min(query_times) if query_times else 0,
                "max_query_time_seconds": max(query_times) if query_times else 0,
                "average_metrics": avg_metrics,
                "detailed_metrics": all_metrics,
                "failed_queries_list": failed_queries
            }
        else:
            return {
                "experiment_name": experiment_name,
                "error": "All queries failed",
                "failed_queries": failed_queries
            }
    
    def evaluate_batch(
        self,
        test_queries: List[Dict[str, str]],
        experiment_name: str = "batch_evaluation"
    ) -> Dict[str, Any]:
        """배치 평가 (동기 래퍼)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.evaluate_batch_async(test_queries, experiment_name)
        )


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search Quality Evaluation")
    parser.add_argument(
        "--enable-improvements",
        action="store_true",
        default=True,
        help="Enable search improvements"
    )
    parser.add_argument(
        "--disable-improvements",
        action="store_true",
        help="Disable search improvements"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["statute_article", "precedent", "procedure", "general_question", "all"],
        default="all",
        help="Query type to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/search_quality_evaluation.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    enable_improvements = args.enable_improvements and not args.disable_improvements
    
    evaluator = SearchQualityEvaluator(enable_improvements=enable_improvements)
    
    # 테스트 쿼리 선택
    if args.query_type == "all":
        test_queries = []
        for qtype, queries in TEST_QUERIES.items():
            test_queries.extend([
                {"query": q, "type": qtype}
                for q in queries
            ])
    else:
        test_queries = [
            {"query": q, "type": args.query_type}
            for q in TEST_QUERIES.get(args.query_type, [])
        ]
    
    experiment_name = f"search_quality_{'with' if enable_improvements else 'without'}_improvements"
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Search Quality Evaluation")
    logger.info("=" * 60)
    logger.info(f"Total queries: {len(test_queries)}")
    logger.info(f"Improvements: {'Enabled' if enable_improvements else 'Disabled'}")
    logger.info(f"Query type: {args.query_type}")
    logger.info("=" * 60)
    logger.info("")
    
    main_start_time = time.time()
    results = evaluator.evaluate_batch(test_queries, experiment_name)
    main_total_time = time.time() - main_start_time
    
    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total execution time: {main_total_time:.2f}초")
    if "total_time_seconds" in results:
        logger.info(f"Evaluation time: {results.get('total_time_seconds', 0):.2f}초")
    logger.info(f"Successful queries: {results.get('successful_queries', 0)}/{results.get('total_queries', 0)}")
    logger.info(f"Failed queries: {results.get('failed_queries', 0)}/{results.get('total_queries', 0)}")
    logger.info("")
    logger.info("Average metrics:")
    avg_metrics = results.get('average_metrics', {})
    for key, value in avg_metrics.items():
        if isinstance(value, float):
            logger.info(f"  - {key}: {value:.4f}")
        else:
            logger.info(f"  - {key}: {value}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()

