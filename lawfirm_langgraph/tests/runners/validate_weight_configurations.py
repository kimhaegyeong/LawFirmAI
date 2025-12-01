# -*- coding: utf-8 -*-
"""
가중치 설정 검증 및 최적화 스크립트

다양한 가중치 조합을 테스트하고 평가 메트릭을 수집하여 최적의 가중치를 찾습니다.

Usage:
    python lawfirm_langgraph/tests/runners/validate_weight_configurations.py
    python lawfirm_langgraph/tests/runners/validate_weight_configurations.py --query-type law_inquiry
    python lawfirm_langgraph/tests/runners/validate_weight_configurations.py --quick  # 빠른 테스트 (적은 조합)
"""

import sys
import os
import json
import re
import asyncio
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger

# 프로젝트 루트 경로
script_dir = Path(__file__).parent
runners_dir = script_dir.parent
tests_dir = runners_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent
sys.path.insert(0, str(lawfirm_langgraph_dir))
sys.path.insert(0, str(project_root))

# 환경 변수 로드
try:
    from utils.env_loader import ensure_env_loaded
    ensure_env_loaded(project_root)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = get_logger(__name__)

# MLflow 통합
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


@dataclass
class WeightConfig:
    """가중치 설정"""
    name: str
    hybrid_law: Dict[str, float]
    hybrid_case: Dict[str, float]
    hybrid_general: Dict[str, float]
    doc_type_boost: Dict[str, float]
    quality_weight: float
    keyword_adjustment: float


@dataclass
class EvaluationMetrics:
    """평가 메트릭"""
    # 검색 관련 메트릭
    avg_relevance_score: float = 0.0
    min_relevance_score: float = 0.0
    max_relevance_score: float = 0.0
    keyword_coverage: float = 0.0
    
    # 문서 활용 메트릭
    retrieved_docs_count: int = 0
    used_docs_count: int = 0
    document_utilization_rate: float = 0.0  # used_docs / retrieved_docs
    
    # 답변 품질 메트릭
    answer_length: int = 0
    answer_quality_score: float = 0.0  # 0-100
    has_sources: bool = False
    source_count: int = 0
    
    # 소스 관련성 메트릭
    source_relevance_avg: float = 0.0
    source_coverage: float = 0.0  # 답변이 소스에 기반하는 정도
    
    # 성능 메트릭
    total_time: float = 0.0
    search_time: float = 0.0
    generation_time: float = 0.0
    
    # 종합 점수
    overall_score: float = 0.0  # 가중 평균으로 계산


@dataclass
class TestResult:
    """테스트 결과"""
    config: WeightConfig
    query: str
    query_type: str
    metrics: EvaluationMetrics
    timestamp: str
    success: bool
    error: Optional[str] = None


class WeightConfigGenerator:
    """가중치 조합 생성기"""
    
    @staticmethod
    def generate_weight_combinations(query_type: str = "all", quick: bool = False) -> List[WeightConfig]:
        """
        다양한 가중치 조합 생성
        
        Args:
            query_type: 테스트할 질문 유형 ("law_inquiry", "precedent_search", "general", "all")
            quick: 빠른 테스트 모드 (적은 조합)
        """
        combinations = []
        
        if quick:
            # 빠른 테스트: 핵심 조합만
            law_semantic_values = [0.3, 0.4, 0.45, 0.5]
            case_semantic_values = [0.6, 0.65, 0.7, 0.75]
            general_semantic_values = [0.5]
        else:
            # 전체 테스트: 더 많은 조합
            law_semantic_values = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            case_semantic_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            general_semantic_values = [0.4, 0.45, 0.5, 0.55, 0.6]
        
        config_id = 0
        
        # 법령 조회 가중치 조합
        if query_type in ["law_inquiry", "all"]:
            for law_semantic in law_semantic_values:
                law_keyword = 1.0 - law_semantic
                config_id += 1
                combinations.append(WeightConfig(
                    name=f"law_sem{law_semantic:.2f}_kw{law_keyword:.2f}",
                    hybrid_law={"semantic": law_semantic, "keyword": law_keyword},
                    hybrid_case={"semantic": 0.65, "keyword": 0.35},  # 기본값
                    hybrid_general={"semantic": 0.5, "keyword": 0.5},  # 기본값
                    doc_type_boost={"statute": 1.2, "case": 1.15},
                    quality_weight=0.2,
                    keyword_adjustment=1.8
                ))
        
        # 판례 검색 가중치 조합
        if query_type in ["precedent_search", "all"]:
            for case_semantic in case_semantic_values:
                case_keyword = 1.0 - case_semantic
                config_id += 1
                combinations.append(WeightConfig(
                    name=f"case_sem{case_semantic:.2f}_kw{case_keyword:.2f}",
                    hybrid_law={"semantic": 0.45, "keyword": 0.55},  # 기본값
                    hybrid_case={"semantic": case_semantic, "keyword": case_keyword},
                    hybrid_general={"semantic": 0.5, "keyword": 0.5},  # 기본값
                    doc_type_boost={"statute": 1.2, "case": 1.15},
                    quality_weight=0.2,
                    keyword_adjustment=1.8
                ))
        
        # 일반 질문 가중치 조합
        if query_type in ["general", "all"]:
            for general_semantic in general_semantic_values:
                general_keyword = 1.0 - general_semantic
                config_id += 1
                combinations.append(WeightConfig(
                    name=f"general_sem{general_semantic:.2f}_kw{general_keyword:.2f}",
                    hybrid_law={"semantic": 0.45, "keyword": 0.55},  # 기본값
                    hybrid_case={"semantic": 0.65, "keyword": 0.35},  # 기본값
                    hybrid_general={"semantic": general_semantic, "keyword": general_keyword},
                    doc_type_boost={"statute": 1.2, "case": 1.15},
                    quality_weight=0.2,
                    keyword_adjustment=1.8
                ))
        
        # 현재 설정 추가 (베이스라인)
        combinations.insert(0, WeightConfig(
            name="current_baseline",
            hybrid_law={"semantic": 0.45, "keyword": 0.55},
            hybrid_case={"semantic": 0.65, "keyword": 0.35},
            hybrid_general={"semantic": 0.5, "keyword": 0.5},
            doc_type_boost={"statute": 1.2, "case": 1.15},
            quality_weight=0.2,
            keyword_adjustment=1.8
        ))
        
        return combinations


class TestQuerySet:
    """테스트 쿼리 세트"""
    
    @staticmethod
    def get_queries(query_type: str = "all") -> Dict[str, List[str]]:
        """
        질문 유형별 테스트 쿼리 반환
        
        Args:
            query_type: 질문 유형 ("law_inquiry", "precedent_search", "general", "all")
        """
        queries = {
            "law_inquiry": [
                # 민법 조문 조회 (10개)
                "민법 제750조 손해배상에 대해 설명해주세요",
                "계약 위약금에 대해 설명해주세요",
                "민법 제103조 불공정한 법률행위에 대해 설명해주세요",
                "민법 제563조 매매계약의 해제에 대해 설명해주세요",
                "민법 제105조 사기·강박에 의한 의사표시에 대해 알려주세요",
                "민법 제110조 대리권의 범위에 대해 설명해주세요",
                "민법 제213조 소유권의 내용에 대해 알려주세요",
                "민법 제618조 임대차의 의의에 대해 설명해주세요",
                "민법 제543조 계약의 해제에 대해 알려주세요",
                "민법 제390조 채무불이행에 대해 설명해주세요",
                # 민사법 개념 조회 (8개)
                "손해배상의 범위는 어떻게 결정되나요?",
                "계약 해지 사유에는 어떤 것들이 있나요?",
                "불법행위가 성립하려면 어떤 요건이 필요한가요?",
                "명예훼손이 성립하려면 어떤 조건이 필요한가요?",
                "임대차 계약의 효력은 무엇인가요?",
                "계약 위약금의 법적 효력은 무엇인가요?",
                "소유권 이전의 요건은 어떻게 되나요?",
                "채권 양도의 제한사항을 알려주세요",
                # 민사법 절차 조회 (7개)
                "손해배상 청구 절차는 어떻게 되나요?",
                "계약 해지 절차를 설명해주세요",
                "임대차 계약 해지 절차는 무엇인가요?",
                "명예훼손 고소 절차를 알려주세요",
                "소유권 이전 등기 절차는 어떻게 되나요?",
                "계약 위약금 청구 절차를 설명해주세요",
                "채권 추심 절차는 무엇인가요?"
            ],
            "precedent_search": [
                # 특정 사건 검색 (10개)
                "계약 해지 관련 판례를 찾아주세요",
                "손해배상 청구 사례를 알려주세요",
                "임대차 계약 해지 판례를 알려주세요",
                "계약 위약금 관련 판례를 찾아주세요",
                "명예훼손 판례를 찾아주세요",
                "계약 해석 관련 판례를 찾아주세요",
                "소유권 이전 판례를 알려주세요",
                "채권 양도 무효 판례를 찾아주세요",
                "불법행위 손해배상 판례를 알려주세요",
                "계약 체결 무효 판례를 찾아주세요",
                # 유사 사례 검색 (8개)
                "계약 해지 사유가 불명확한 경우 판례를 찾아주세요",
                "손해배상 범위 산정 관련 판례를 알려주세요",
                "임대차 계약 해지 사유 판례를 알려주세요",
                "계약 위약금 과다 감액 판례를 찾아주세요",
                "명예훼손 공연성 요건 판례를 알려주세요",
                "소유권 이전 등기 관련 판례를 찾아주세요",
                "채권 추심 관련 판례를 알려주세요",
                "불법행위 인과관계 판례를 찾아주세요",
                # 법원별 판례 검색 (7개)
                "대법원 계약 해지 판례를 찾아주세요",
                "고등법원 손해배상 판례를 알려주세요",
                "지방법원 임대차 판례를 찾아주세요",
                "대법원 임대차 판례를 알려주세요",
                "고등법원 계약 위약금 판례를 찾아주세요",
                "대법원 명예훼손 판례를 알려주세요",
                "지방법원 소유권 이전 판례를 찾아주세요"
            ],
            "general": [
                # 민사법 자문 (7개) - 정보 조회 및 교육 관련 질문 제외
                "민사법 자문이 필요합니다",
                "계약서 작성 시 주의사항을 알려주세요",
                "민사법 용어를 설명해주세요",
                "민사 소송 절차에 대해 안내해주세요",
                "민사법 상담이 필요합니다",
                "계약 분쟁 해결 방법을 알려주세요",
                "손해배상 청구 방법을 설명해주세요"
            ]
        }
        
        if query_type == "all":
            return queries
        elif query_type in queries:
            return {query_type: queries[query_type]}
        else:
            return {query_type: queries.get("general", [])}


class WeightConfigUpdater:
    """가중치 설정 업데이트"""
    
    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.original_content = None
    
    def backup(self):
        """원본 설정 백업"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_file}")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.original_content = f.read()
            if not self.original_content:
                raise ValueError(f"설정 파일이 비어있습니다: {self.config_file}")
        except Exception as e:
            raise ValueError(f"설정 파일을 읽는 중 오류 발생: {e}")
    
    def update(self, config: WeightConfig):
        """가중치 설정 업데이트 - 숫자 값만 교체하는 안전한 방식"""
        if not self.original_content:
            self.backup()
        
        if not self.original_content:
            raise ValueError(f"설정 파일을 읽을 수 없습니다: {self.config_file}")
        
        content = self.original_content
        
        # 숫자 값만 정확하게 교체 (들여쓰기 유지)
        # hybrid_law semantic
        content = re.sub(
            r'"hybrid_law":\s*\{\s*"semantic":\s*[\d.]+',
            f'"hybrid_law": {{"semantic": {config.hybrid_law["semantic"]}',
            content
        )
        # hybrid_law keyword
        content = re.sub(
            r'"hybrid_law":\s*\{\s*"semantic":\s*[\d.]+\s*,\s*"keyword":\s*[\d.]+',
            f'"hybrid_law": {{"semantic": {config.hybrid_law["semantic"]}, "keyword": {config.hybrid_law["keyword"]}',
            content
        )
        
        # hybrid_case semantic
        content = re.sub(
            r'"hybrid_case":\s*\{\s*"semantic":\s*[\d.]+',
            f'"hybrid_case": {{"semantic": {config.hybrid_case["semantic"]}',
            content
        )
        # hybrid_case keyword
        content = re.sub(
            r'"hybrid_case":\s*\{\s*"semantic":\s*[\d.]+\s*,\s*"keyword":\s*[\d.]+',
            f'"hybrid_case": {{"semantic": {config.hybrid_case["semantic"]}, "keyword": {config.hybrid_case["keyword"]}',
            content
        )
        
        # hybrid_general semantic
        content = re.sub(
            r'"hybrid_general":\s*\{\s*"semantic":\s*[\d.]+',
            f'"hybrid_general": {{"semantic": {config.hybrid_general["semantic"]}',
            content
        )
        # hybrid_general keyword
        content = re.sub(
            r'"hybrid_general":\s*\{\s*"semantic":\s*[\d.]+\s*,\s*"keyword":\s*[\d.]+',
            f'"hybrid_general": {{"semantic": {config.hybrid_general["semantic"]}, "keyword": {config.hybrid_general["keyword"]}',
            content
        )
        
        # doc_type_boost statute
        content = re.sub(
            r'"doc_type_boost":\s*\{\s*"statute":\s*[\d.]+',
            f'"doc_type_boost": {{"statute": {config.doc_type_boost["statute"]}',
            content
        )
        # doc_type_boost case
        content = re.sub(
            r'"doc_type_boost":\s*\{\s*"statute":\s*[\d.]+\s*,\s*"case":\s*[\d.]+',
            f'"doc_type_boost": {{"statute": {config.doc_type_boost["statute"]}, "case": {config.doc_type_boost["case"]}',
            content
        )
        
        # quality_weight
        content = re.sub(
            r'"quality_weight":\s*[\d.]+',
            f'"quality_weight": {config.quality_weight}',
            content
        )
        
        # keyword_adjustment
        content = re.sub(
            r'"keyword_adjustment":\s*[\d.]+',
            f'"keyword_adjustment": {config.keyword_adjustment}',
            content
        )
        
        # 파일 저장
        with open(self.config_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def restore(self):
        """원본 설정 복원"""
        if self.original_content:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(self.original_content)


class QueryTestRunner:
    """쿼리 테스트 실행기"""
    
    def __init__(self):
        # run_query_test 함수를 직접 import
        try:
            script_dir = Path(__file__).parent
            run_query_test_path = script_dir / "run_query_test.py"
            if not run_query_test_path.exists():
                # 같은 디렉토리에 없으면 runners 디렉토리에서 찾기
                run_query_test_path = script_dir / "run_query_test.py"
            
            if not run_query_test_path.exists():
                raise FileNotFoundError(f"run_query_test.py를 찾을 수 없습니다: {run_query_test_path}")
            
            # 모듈을 동적으로 import
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_query_test", run_query_test_path)
            run_query_test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(run_query_test_module)
            
            self.run_query_test_func = run_query_test_module.run_query_test
            self._extract_and_normalize_answer = run_query_test_module._extract_and_normalize_answer
            self._evaluate_answer_quality = run_query_test_module._evaluate_answer_quality
            
        except Exception as e:
            logger.error(f"run_query_test 모듈 로드 실패: {e}")
            raise
    
    async def run_test(self, query: str) -> Tuple[Dict[str, Any], str]:
        """테스트 실행 - result 딕셔너리와 로그 출력 반환"""
        try:
            # run_query_test 함수 직접 호출
            result = await self.run_query_test_func(query, enable_profiling=False, enable_memory_monitoring=False)
            
            # 로그 핸들러에서 출력 캡처
            if isinstance(result, dict):
                result_keys = list(result.keys())
                # 큰 데이터는 필요시에만 유지 (메모리 절약)
                if "retrieved_docs" in result and len(result["retrieved_docs"]) > 20:
                    # 너무 많은 문서는 샘플만 유지
                    result["retrieved_docs"] = result["retrieved_docs"][:20]
            else:
                result_keys = []
            output_str = f"Query: {query}\nResult keys: {result_keys}"
            
            return result, output_str
            
        except Exception as e:
            logger.error(f"테스트 실행 실패: {e}")
            return {}, str(e)
        finally:
            # 테스트 완료 후 즉시 메모리 정리
            gc.collect()


class MetricsExtractor:
    """메트릭 추출기"""
    
    @staticmethod
    def extract_metrics_from_result(result: Dict[str, Any], query: str) -> EvaluationMetrics:
        """result 딕셔너리에서 직접 메트릭 추출"""
        metrics = EvaluationMetrics()
        
        if not isinstance(result, dict):
            logger.warning(f"   ⚠️  result가 딕셔너리가 아닙니다: {type(result)}")
            return metrics
        
        # 답변 정보
        answer = result.get("answer", "")
        if isinstance(answer, str):
            metrics.answer_length = len(answer)
        else:
            # answer가 딕셔너리일 수 있음
            answer_str = str(answer) if answer else ""
            metrics.answer_length = len(answer_str)
        
        # 답변 품질 평가
        retrieved_docs = result.get("retrieved_docs", [])
        sources = result.get("sources", [])
        
        # 간단한 품질 점수 계산
        answer_quality_score = 0.0
        if answer and metrics.answer_length > 0:
            answer_quality_score += 25  # 답변 존재
        if metrics.answer_length >= 100:  # MIN_ANSWER_LENGTH
            answer_quality_score += 25  # 최소 길이 충족
        if not any(pattern in str(answer) for pattern in ["죄송합니다", "오류가 발생했습니다", "시스템 오류"]):
            answer_quality_score += 25  # 오류 메시지 없음
        if len(retrieved_docs) > 0 or len(sources) > 0:
            answer_quality_score += 25  # 참고자료 존재
        
        metrics.answer_quality_score = answer_quality_score
        
        # 문서 활용 메트릭
        metrics.retrieved_docs_count = len(retrieved_docs) if retrieved_docs else 0
        metrics.used_docs_count = len(sources) if sources else 0
        metrics.source_count = metrics.used_docs_count
        metrics.has_sources = metrics.source_count > 0
        
        if metrics.retrieved_docs_count > 0:
            metrics.document_utilization_rate = metrics.used_docs_count / metrics.retrieved_docs_count
        else:
            metrics.document_utilization_rate = 0.0
        
        # 검색 관련성 점수 (retrieved_docs의 score에서 계산)
        if retrieved_docs:
            scores = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    score = doc.get("relevance_score") or doc.get("score") or doc.get("similarity_score")
                    if score is not None:
                        try:
                            scores.append(float(score))
                        except (ValueError, TypeError):
                            pass
            
            if scores:
                metrics.avg_relevance_score = sum(scores) / len(scores)
                metrics.min_relevance_score = min(scores)
                metrics.max_relevance_score = max(scores)
        
        # 성능 메트릭
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            metrics.total_time = float(processing_time)
        
        # 종합 점수 계산
        metrics.overall_score = MetricsExtractor._calculate_overall_score(metrics)
        
        return metrics
    
    @staticmethod
    def extract_metrics(output: str, query: str) -> EvaluationMetrics:
        """출력에서 메트릭 추출"""
        metrics = EvaluationMetrics()
        
        if not output or len(output.strip()) == 0:
            logger.warning("   ⚠️  출력이 비어있습니다. 메트릭 추출 불가.")
            return metrics
        
        # 검색 관련 메트릭
        # run_query_test.py는 Avg Relevance를 직접 출력하지 않으므로, retrieved_docs의 score에서 계산
        # 또는 검색 결과에서 추출 시도
        avg_match = re.search(r'Avg Relevance: ([\d.]+)', output)
        if avg_match:
            metrics.avg_relevance_score = float(avg_match.group(1))
        else:
            # retrieved_docs의 score에서 평균 계산 시도
            score_matches = re.findall(r'score=([\d.]+)', output)
            if score_matches:
                scores = [float(s) for s in score_matches]
                metrics.avg_relevance_score = sum(scores) / len(scores) if scores else 0.0
                metrics.min_relevance_score = min(scores) if scores else 0.0
                metrics.max_relevance_score = max(scores) if scores else 0.0
            else:
                # 유사도 점수 분포에서 추출 시도
                avg_score_match = re.search(r'평균=([\d.]+)', output)
                if avg_score_match:
                    metrics.avg_relevance_score = float(avg_score_match.group(1))
        
        min_match = re.search(r'Min: ([\d.]+)', output)
        if min_match:
            metrics.min_relevance_score = float(min_match.group(1))
        
        max_match = re.search(r'Max: ([\d.]+)', output)
        if max_match:
            metrics.max_relevance_score = float(max_match.group(1))
        
        keyword_match = re.search(r'Keyword Coverage: ([\d.]+)', output)
        if keyword_match:
            metrics.keyword_coverage = float(keyword_match.group(1))
        
        # 문서 활용 메트릭
        # run_query_test.py 형식: "🔍 검색된 참고자료 (retrieved_docs) ({count}개):"
        retrieved_match = re.search(r'검색된 참고자료.*?\((\d+)개\)', output)
        if not retrieved_match:
            retrieved_match = re.search(r'검색된 문서.*?(\d+)개', output)
        if retrieved_match:
            metrics.retrieved_docs_count = int(retrieved_match.group(1))
        
        # sources 개수
        sources_match = re.search(r'소스 \(sources\)\s*\((\d+)개\)', output)
        if sources_match:
            metrics.used_docs_count = int(sources_match.group(1))
            metrics.source_count = metrics.used_docs_count
            metrics.has_sources = metrics.source_count > 0
        
        # 실제 사용 문서 수 (sources가 없으면 retrieved_docs 사용)
        if metrics.retrieved_docs_count > 0:
            if metrics.used_docs_count == 0:
                # sources가 없으면 retrieved_docs를 사용된 것으로 간주
                metrics.used_docs_count = metrics.retrieved_docs_count
            metrics.document_utilization_rate = metrics.used_docs_count / metrics.retrieved_docs_count
        
        # 답변 품질 메트릭
        # run_query_test.py 형식: "📝 답변 ({length}자)"
        answer_length_match = re.search(r'답변\s*\((\d+)자\)', output)
        if answer_length_match:
            metrics.answer_length = int(answer_length_match.group(1))
        
        # run_query_test.py 형식: "품질 점수: {score}/100"
        quality_match = re.search(r'품질 점수:\s*(\d+)/100', output)
        if quality_match:
            metrics.answer_quality_score = float(quality_match.group(1))
        
        # 참고자료 개수
        if not metrics.has_sources:
            source_match = re.search(r'참고자료 존재.*?(\d+)개', output)
            if source_match:
                metrics.source_count = int(source_match.group(1))
                metrics.has_sources = metrics.source_count > 0
        
        # 성능 메트릭
        # run_query_test.py 형식: "⏱️  처리 시간: {time}초"
        time_match = re.search(r'처리 시간:\s*([\d.]+)초', output)
        if not time_match:
            time_match = re.search(r'총 소요 시간.*?([\d.]+)초', output)
        if time_match:
            metrics.total_time = float(time_match.group(1))
        
        # 종합 점수 계산
        metrics.overall_score = MetricsExtractor._calculate_overall_score(metrics)
        
        return metrics
    
    @staticmethod
    def _calculate_overall_score(metrics: EvaluationMetrics) -> float:
        """
        종합 점수 계산 (가중 평균)
        
        가중치:
        - 답변 품질: 30%
        - 문서 활용률: 25%
        - 소스 관련성: 20%
        - 검색 점수: 15%
        - 성능: 10%
        """
        # 답변 품질 점수 (0-100) → 0-1 정규화
        quality_score = metrics.answer_quality_score / 100.0
        
        # 문서 활용률 (0-1)
        utilization_score = metrics.document_utilization_rate
        
        # 소스 관련성 (소스가 있으면 1.0, 없으면 0.0)
        source_score = 1.0 if metrics.has_sources else 0.0
        
        # 검색 점수 (0-1 정규화, avg_relevance_score가 이미 0-1 범위라고 가정)
        search_score = metrics.avg_relevance_score
        
        # 성능 점수 (빠를수록 높음, 10초 기준으로 정규화)
        performance_score = max(0.0, 1.0 - (metrics.total_time / 10.0))
        
        # 가중 평균
        overall = (
            0.30 * quality_score +
            0.25 * utilization_score +
            0.20 * source_score +
            0.15 * search_score +
            0.10 * performance_score
        )
        
        return overall


class MLflowTracker:
    """MLflow 추적기"""
    
    def __init__(self, experiment_name: str = "weight_validation"):
        self.experiment_name = experiment_name
        self.mlflow_available = MLFLOW_AVAILABLE
        self.parent_run_id = None
        
        if not self.mlflow_available:
            logger.warning("MLflow not available. Tracking disabled.")
            return
        
        try:
            # MLflow 설정 (SQLite 백엔드 사용)
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                mlflow_db_path = project_root / "mlflow" / "mlflow.db"
                mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)
                tracking_uri = f"sqlite:///{str(mlflow_db_path).replace(os.sep, '/')}"
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            # 부모 run 시작 (전체 검증 세션)
            run_name = f"weight_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name, nested=False)
            self.parent_run_id = mlflow.active_run().info.run_id
            
            logger.info(f"✅ MLflow 실험 시작: {experiment_name} (run_id: {self.parent_run_id})")
            
        except Exception as e:
            logger.warning(f"MLflow 초기화 실패: {e}. Tracking disabled.")
            self.mlflow_available = False
    
    def log_config_run(self, config: WeightConfig, query: str, query_type: str, 
                      metrics: EvaluationMetrics, success: bool) -> Optional[str]:
        """개별 가중치 설정 테스트 결과 로깅"""
        if not self.mlflow_available:
            return None
        
        try:
            run_name = f"{config.name}_{query_type}_{datetime.now().strftime('%H%M%S')}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                # 태그 설정
                mlflow.set_tags({
                    "config_name": config.name,
                    "query_type": query_type,
                    "query": query[:100],  # 쿼리 일부만 (너무 길면 잘림)
                    "success": str(success)
                })
                
                # 파라미터 로깅 (가중치 설정)
                mlflow.log_params({
                    "hybrid_law_semantic": config.hybrid_law["semantic"],
                    "hybrid_law_keyword": config.hybrid_law["keyword"],
                    "hybrid_case_semantic": config.hybrid_case["semantic"],
                    "hybrid_case_keyword": config.hybrid_case["keyword"],
                    "hybrid_general_semantic": config.hybrid_general["semantic"],
                    "hybrid_general_keyword": config.hybrid_general["keyword"],
                    "doc_type_boost_statute": config.doc_type_boost["statute"],
                    "doc_type_boost_case": config.doc_type_boost["case"],
                    "quality_weight": config.quality_weight,
                    "keyword_adjustment": config.keyword_adjustment
                })
                
                # 메트릭 로깅
                if success:
                    mlflow.log_metrics({
                        "overall_score": metrics.overall_score,
                        "answer_quality_score": metrics.answer_quality_score,
                        "document_utilization_rate": metrics.document_utilization_rate,
                        "avg_relevance_score": metrics.avg_relevance_score,
                        "keyword_coverage": metrics.keyword_coverage,
                        "answer_length": float(metrics.answer_length),
                        "source_count": float(metrics.source_count),
                        "retrieved_docs_count": float(metrics.retrieved_docs_count),
                        "used_docs_count": float(metrics.used_docs_count),
                        "total_time": metrics.total_time
                    })
                
                run_id = mlflow.active_run().info.run_id
                return run_id
        
        except Exception as e:
            logger.warning(f"MLflow 로깅 실패: {e}")
            return None
    
    def log_summary(self, analysis: Dict[str, Any], best_config_name: str):
        """검증 결과 요약 로깅"""
        if not self.mlflow_available:
            return
        
        try:
            # 부모 run에 요약 메트릭 로깅
            if "best_config" in analysis:
                best = analysis["best_config"]
                mlflow.log_metrics({
                    "best_overall_score": best["metrics"]["avg_score"],
                    "best_median_score": best["metrics"]["median_score"],
                    "best_min_score": best["metrics"]["min_score"],
                    "best_max_score": best["metrics"]["max_score"],
                    "best_std_dev": best["metrics"]["std_dev"]
                })
            
            if "summary" in analysis:
                mlflow.log_metrics({
                    "total_tests": float(analysis.get("total_tests", 0)),
                    "successful_tests": float(analysis.get("successful_tests", 0))
                })
            
            # 최적 설정을 태그로 저장
            mlflow.set_tags({
                "best_config": best_config_name,
                "validation_completed": "true"
            })
            
        except Exception as e:
            logger.warning(f"MLflow 요약 로깅 실패: {e}")
    
    def log_artifacts(self, output_file: Path):
        """아티팩트 로깅"""
        if not self.mlflow_available:
            return
        
        try:
            if output_file.exists():
                mlflow.log_artifact(str(output_file), "validation_results")
                logger.info(f"✅ MLflow 아티팩트 로깅: {output_file}")
        except Exception as e:
            logger.warning(f"MLflow 아티팩트 로깅 실패: {e}")
    
    def end_run(self):
        """MLflow run 종료"""
        if self.mlflow_available:
            try:
                mlflow.end_run()
                logger.info(f"✅ MLflow run 종료: {self.parent_run_id}")
            except Exception as e:
                logger.warning(f"MLflow run 종료 실패: {e}")


class WeightValidationRunner:
    """가중치 검증 실행기"""
    
    def __init__(self, quick: bool = False, use_mlflow: bool = True, max_workers: int = None):
        self.quick = quick
        # 병렬 처리 워커 수 설정 (기본값: CPU 코어 수)
        if max_workers is None:
            import multiprocessing
            self.max_workers = min(multiprocessing.cpu_count(), 4)  # 최대 4개로 제한 (메모리 고려)
        else:
            self.max_workers = max_workers
        
        # 설정 파일 경로 확인 및 수정
        # 여러 경로 시도
        possible_paths = [
            lawfirm_langgraph_dir / "core" / "search" / "processors" / "search_result_processor.py",
            project_root / "lawfirm_langgraph" / "core" / "search" / "processors" / "search_result_processor.py",
            Path(__file__).parent.parent.parent / "core" / "search" / "processors" / "search_result_processor.py"
        ]
        
        config_file = None
        for path in possible_paths:
            if path.exists():
                config_file = path.resolve()
                break
        
        if not config_file:
            raise FileNotFoundError(
                "설정 파일을 찾을 수 없습니다. 시도한 경로:\n" + 
                "\n".join([f"  - {p}" for p in possible_paths])
            )
        
        self.config_file = config_file
        self.config_updater = WeightConfigUpdater(self.config_file)
        self.test_runner = QueryTestRunner()
        self.metrics_extractor = MetricsExtractor()
        self.mlflow_tracker = MLflowTracker() if use_mlflow and MLFLOW_AVAILABLE else None
        
        logger.info(f"✅ 병렬 처리 워커 수: {self.max_workers}")
    
    async def _run_single_test(self, config: WeightConfig, query: str, q_type: str, 
                               current_test: int, total_tests: int) -> TestResult:
        """단일 테스트 실행 (병렬 처리용)"""
        try:
            # 테스트 실행 (result 딕셔너리 반환)
            result, output_str = await self.test_runner.run_test(query)
            
            # result에서 직접 메트릭 추출
            metrics = self.metrics_extractor.extract_metrics_from_result(result, query)
            
            test_result = TestResult(
                config=config,
                query=query,
                query_type=q_type,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            # MLflow 로깅
            if self.mlflow_tracker:
                self.mlflow_tracker.log_config_run(
                    config, query, q_type, metrics, True
                )
            
            # 메모리 정리
            del result, output_str, metrics
            gc.collect()
            
            return test_result
            
        except Exception as e:
            logger.error(f"   ❌ 테스트 실패: {e}")
            test_result = TestResult(
                config=config,
                query=query,
                query_type=q_type,
                metrics=EvaluationMetrics(),
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e)
            )
            
            # MLflow 로깅 (실패한 경우도)
            if self.mlflow_tracker:
                self.mlflow_tracker.log_config_run(
                    config, query, q_type, EvaluationMetrics(), False
                )
            
            gc.collect()
            return test_result
    
    async def run_validation(self, query_type: str = "all") -> List[TestResult]:
        """검증 실행 (병렬 처리 및 가비지 컬렉션 적용)"""
        import time
        
        start_time = time.time()
        logger.info("🚀 가중치 검증 시작")
        
        # 가비지 컬렉션 설정 최적화
        gc.set_threshold(700, 10, 10)  # 더 적극적인 GC
        
        # 가중치 조합 생성
        configs = WeightConfigGenerator.generate_weight_combinations(query_type, self.quick)
        queries_dict = TestQuerySet.get_queries(query_type)
        
        total_configs = len(configs)
        total_queries = sum(len(q) for q in queries_dict.values())
        total_tests = total_configs * total_queries
        
        logger.info(f"   총 테스트 수: {total_tests}개 (가중치 조합: {total_configs}개 × 쿼리: {total_queries}개)")
        logger.info(f"   병렬 처리 워커 수: {self.max_workers}")
        
        # 원본 설정 백업
        self.config_updater.backup()
        
        all_results = []
        
        try:
            current_test = 0
            
            for config in configs:
                # 가중치 설정 업데이트
                self.config_updater.update(config)
                
                # 각 질문 유형별 쿼리 테스트
                for q_type, queries in queries_dict.items():
                    # 병렬 처리: 배치 단위로 실행
                    batch_size = self.max_workers
                    for batch_start in range(0, len(queries), batch_size):
                        batch_queries = queries[batch_start:batch_start + batch_size]
                        
                        # 배치 내 병렬 실행
                        tasks = []
                        for query in batch_queries:
                            current_test += 1
                            task = self._run_single_test(
                                config, query, q_type, current_test, total_tests
                            )
                            tasks.append(task)
                        
                        # 병렬 실행 및 결과 수집
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # 결과 처리
                        for result in batch_results:
                            if isinstance(result, Exception):
                                logger.error(f"   ❌ 배치 실행 중 오류: {result}")
                                # 실패한 테스트 결과 생성
                                all_results.append(TestResult(
                                    config=config,
                                    query="unknown",
                                    query_type=q_type,
                                    metrics=EvaluationMetrics(),
                                    timestamp=datetime.now().isoformat(),
                                    success=False,
                                    error=str(result)
                                ))
                            else:
                                all_results.append(result)
                        
                        # 진행률 표시
                        progress = (current_test / total_tests) * 100
                        elapsed_time = time.time() - start_time
                        avg_time_per_test = elapsed_time / current_test if current_test > 0 else 0
                        remaining_tests = total_tests - current_test
                        estimated_remaining = (remaining_tests * avg_time_per_test) / 60 if avg_time_per_test > 0 else 0
                        
                        logger.info(f"📊 진행률: {current_test}/{total_tests} ({progress:.1f}%) | 예상 남은 시간: {estimated_remaining:.1f}분")
                        
                        # 배치 완료 후 메모리 정리
                        del tasks, batch_results
                        gc.collect()
                
                # 설정별 테스트 완료 후 메모리 정리
                gc.collect()
        
        finally:
            # 원본 설정 복원
            self.config_updater.restore()
            
            # 최종 통계
            total_elapsed = time.time() - start_time
            successful_tests = sum(1 for r in all_results if r.success)
            
            logger.info(f"\n✅ 검증 완료: {len(all_results)}/{total_tests} 완료 (성공: {successful_tests}개, 소요 시간: {total_elapsed/60:.1f}분)")
            logger.info("   원본 설정 복원 완료")
            
            # MLflow run 종료
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run()
            
            # 최종 메모리 정리
            gc.collect()
        
        return all_results


class ResultsAnalyzer:
    """결과 분석기"""
    
    @staticmethod
    def analyze_results(results: List[TestResult]) -> Dict[str, Any]:
        """결과 분석"""
        # 성공한 결과만 필터링
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "성공한 테스트 결과가 없습니다."}
        
        # 설정별 그룹화
        config_groups = {}
        for result in successful_results:
            config_name = result.config.name
            if config_name not in config_groups:
                config_groups[config_name] = []
            config_groups[config_name].append(result)
        
        # 설정별 평균 점수 계산
        config_scores = {}
        for config_name, config_results in config_groups.items():
            scores = [r.metrics.overall_score for r in config_results]
            config_scores[config_name] = {
                "avg_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "test_count": len(scores)
            }
        
        # 최고 성능 설정 찾기
        best_config = max(config_scores.items(), key=lambda x: x[1]["avg_score"])
        
        # 질문 유형별 분석
        query_type_analysis = {}
        for q_type in ["law_inquiry", "precedent_search", "general"]:
            type_results = [r for r in successful_results if r.query_type == q_type]
            if type_results:
                type_scores = [r.metrics.overall_score for r in type_results]
                query_type_analysis[q_type] = {
                    "avg_score": statistics.mean(type_scores),
                    "test_count": len(type_scores)
                }
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "config_scores": config_scores,
            "best_config": {
                "name": best_config[0],
                "metrics": best_config[1]
            },
            "query_type_analysis": query_type_analysis,
            "summary": {
                "best_overall_score": best_config[1]["avg_score"],
                "config_count": len(config_scores)
            }
        }


class ReportGenerator:
    """리포트 생성기"""
    
    @staticmethod
    def generate_report(analysis: Dict[str, Any], results: List[TestResult], output_file: Path):
        """리포트 생성"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "detailed_results": [asdict(r) for r in results],
            "recommendations": ReportGenerator._generate_recommendations(analysis)
        }
        
        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 리포트 생성
        text_report_file = output_file.with_suffix('.txt')
        ReportGenerator._generate_text_report(analysis, text_report_file)
        
        return report
    
    @staticmethod
    def _generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if "best_config" in analysis:
            best = analysis["best_config"]
            recommendations.append(
                f"최적 가중치 설정: {best['name']} (평균 점수: {best['metrics']['avg_score']:.3f})"
            )
        
        if "query_type_analysis" in analysis:
            for q_type, metrics in analysis["query_type_analysis"].items():
                recommendations.append(
                    f"{q_type} 질문 유형 평균 점수: {metrics['avg_score']:.3f}"
                )
        
        return recommendations
    
    @staticmethod
    def _generate_text_report(analysis: Dict[str, Any], output_file: Path):
        """텍스트 리포트 생성"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("가중치 검증 결과 리포트\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if "best_config" in analysis:
                best = analysis["best_config"]
                f.write("최적 가중치 설정:\n")
                f.write(f"  이름: {best['name']}\n")
                f.write(f"  평균 점수: {best['metrics']['avg_score']:.3f}\n")
                f.write(f"  중앙값: {best['metrics']['median_score']:.3f}\n")
                f.write(f"  표준편차: {best['metrics']['std_dev']:.3f}\n")
                f.write(f"  테스트 수: {best['metrics']['test_count']}\n\n")
            
            if "config_scores" in analysis:
                f.write("설정별 점수 (상위 10개):\n")
                sorted_configs = sorted(
                    analysis["config_scores"].items(),
                    key=lambda x: x[1]["avg_score"],
                    reverse=True
                )[:10]
                
                for config_name, metrics in sorted_configs:
                    f.write(f"  {config_name}: {metrics['avg_score']:.3f} "
                           f"(중앙값: {metrics['median_score']:.3f}, "
                           f"표준편차: {metrics['std_dev']:.3f})\n")
                f.write("\n")
            
            if "query_type_analysis" in analysis:
                f.write("질문 유형별 분석:\n")
                for q_type, metrics in analysis["query_type_analysis"].items():
                    f.write(f"  {q_type}: 평균 {metrics['avg_score']:.3f} "
                           f"({metrics['test_count']}개 테스트)\n")
                f.write("\n")


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="가중치 설정 검증 및 최적화")
    parser.add_argument("--query-type", choices=["law_inquiry", "precedent_search", "general", "all"],
                       default="all", help="테스트할 질문 유형")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트 모드 (적은 조합)")
    parser.add_argument("--output-dir", type=str, default=None, help="결과 저장 디렉토리")
    parser.add_argument("--max-workers", type=int, default=None, 
                       help="병렬 처리 워커 수 (기본값: CPU 코어 수, 최대 4)")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "logs" / "test" / "weight_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 검증 실행
    runner = WeightValidationRunner(quick=args.quick, max_workers=args.max_workers)
    results = await runner.run_validation(args.query_type)
    
    # 결과 분석
    analyzer = ResultsAnalyzer()
    analysis = analyzer.analyze_results(results)
    
    # 리포트 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"weight_validation_{timestamp}.json"
    
    report_generator = ReportGenerator()
    report = report_generator.generate_report(analysis, results, output_file)
    
    # MLflow 요약 로깅
    best_config_name = analysis.get("best_config", {}).get("name", "unknown")
    if runner.mlflow_tracker:
        runner.mlflow_tracker.log_summary(analysis, best_config_name)
        runner.mlflow_tracker.log_artifacts(output_file)
        runner.mlflow_tracker.log_artifacts(output_file.with_suffix('.txt'))
        if runner.mlflow_tracker.parent_run_id:
            logger.info(f"\n📊 MLflow 실험 ID: {runner.mlflow_tracker.parent_run_id}")
            logger.info(f"   MLflow UI: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
    
    # 결과 출력
    logger.info("\n" + "="*80)
    logger.info("검증 결과 요약")
    logger.info("="*80)
    
    if "best_config" in analysis:
        best = analysis["best_config"]
        logger.info(f"\n✅ 최적 가중치 설정: {best['name']}")
        logger.info(f"   평균 점수: {best['metrics']['avg_score']:.3f}")
        logger.info(f"   중앙값: {best['metrics']['median_score']:.3f}")
        logger.info(f"   표준편차: {best['metrics']['std_dev']:.3f}")
    
    logger.info(f"\n📊 결과 저장: {output_file}")
    logger.info(f"📄 텍스트 리포트: {output_file.with_suffix('.txt')}")
    
    if "recommendations" in report:
        logger.info("\n💡 권장사항:")
        for rec in report["recommendations"]:
            logger.info(f"   - {rec}")


if __name__ == "__main__":
    asyncio.run(main())

