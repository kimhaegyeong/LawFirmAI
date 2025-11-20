# -*- coding: utf-8 -*-
"""
LangGraph 질의 테스트 스크립트 (개선 버전)

Usage:
    python lawfirm_langgraph/tests/scripts/run_query_test.py "질의 내용"
    python lawfirm_langgraph/tests/scripts/run_query_test.py 0  # 기본 질의 선택
    $env:TEST_QUERY='질의내용'; python run_query_test.py  # 환경 변수 사용
"""

import sys
import io
import os

# python-dotenv 경고 억제 (가장 먼저 실행)
# stderr를 완전히 리다이렉트하지 않고, warnings만 필터링
_original_stderr = sys.stderr
# stderr 리다이렉트 제거 - 로깅 오류 방지
# try:
#     # Windows와 Unix 모두 지원
#     if sys.platform == 'win32':
#         sys.stderr = open('nul', 'w', encoding='utf-8', errors='replace')
#     else:
#         sys.stderr = open('/dev/null', 'w', encoding='utf-8', errors='replace')
# except Exception:
#     # 실패 시 원본 stderr 유지
#     pass

# warnings 모듈도 필터링
import warnings  # noqa: E402
warnings.filterwarnings('ignore', message='.*python-dotenv.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*python-dotenv.*')
warnings.filterwarnings('ignore', category=Warning)

import asyncio  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402
import cProfile  # noqa: E402
import pstats  # noqa: E402
import tracemalloc  # noqa: E402
try:
    import psutil  # noqa: E402
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# os는 이미 import되어 있으므로 재import 불필요

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

# 테스트 모드: 스트리밍 비활성화하여 generate_answer_final 사용
# API에서는 USE_STREAMING_MODE=true로 설정하여 generate_answer_stream 사용
os.environ['USE_STREAMING_MODE'] = 'false'

# 프로젝트 경로 설정
# 스크립트 위치: lawfirm_langgraph/tests/scripts/run_query_test.py
try:
    script_dir = Path(__file__).parent
except NameError:
    # __file__이 없는 경우 (예: exec로 실행된 경우)
    script_dir = Path.cwd() / "lawfirm_langgraph" / "tests" / "scripts"
tests_dir = script_dir.parent
lawfirm_langgraph_dir = tests_dir.parent
project_root = lawfirm_langgraph_dir.parent

# sys.path 설정 (순환 import 방지)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(lawfirm_langgraph_dir) not in sys.path:
    sys.path.insert(0, str(lawfirm_langgraph_dir))

# 환경 변수 로드 (프로젝트 루트 .env 파일 사용)
try:
    from utils.env_loader import ensure_env_loaded, load_all_env_files
    # 프로젝트 루트의 .env 파일을 명시적으로 로드
    ensure_env_loaded(project_root)
    loaded_files = load_all_env_files(project_root)
    if loaded_files:
        print(f"✅ 환경 변수 로드 완료: {len(loaded_files)}개 .env 파일")
    else:
        print("⚠️  .env 파일을 찾을 수 없습니다. 환경 변수만 사용합니다.")
except ImportError:
    # python-dotenv 직접 사용 (fallback)
    try:
        from dotenv import load_dotenv
        # 프로젝트 루트 .env 파일 로드
        root_env = project_root / ".env"
        if root_env.exists():
            load_dotenv(dotenv_path=str(root_env), override=False)
            print(f"✅ 환경 변수 로드 완료: {root_env}")
        else:
            print(f"⚠️  .env 파일을 찾을 수 없습니다: {root_env}")
    except ImportError:
        print("⚠️  python-dotenv가 설치되지 않았습니다. .env 파일을 로드할 수 없습니다.")
        print("   설치: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  환경 변수 로드 중 오류 발생: {e}")

# 상수 정의
MIN_ANSWER_LENGTH = 100
ERROR_PATTERNS = [
    "죄송합니다",
    "오류가 발생했습니다",
    "시스템 오류",
    "입력값에 문제가 있습니다",
    "답변을 생성하는 중 오류가 발생했습니다"
]
MAX_PROCESSING_TIME_WARNING = 300

# SafeStreamHandler 클래스 정의
class SafeStreamHandler(logging.StreamHandler):
    """버퍼 분리 오류를 방지하는 안전한 스트림 핸들러"""
    
    def __init__(self, stream, original_stdout_ref=None):
        super().__init__(stream)
        self._original_stdout = original_stdout_ref
        self._fallback_stream = None
    
    def _get_safe_stream(self):
        """안전한 스트림 반환"""
        streams_to_try = []
        if self.stream and hasattr(self.stream, 'write'):
            streams_to_try.append(self.stream)
        if self._original_stdout is not None and hasattr(self._original_stdout, 'write'):
            streams_to_try.append(self._original_stdout)
        if sys.stdout and hasattr(sys.stdout, 'write'):
            streams_to_try.append(sys.stdout)
        if sys.stderr and hasattr(sys.stderr, 'write'):
            streams_to_try.append(sys.stderr)
        
        for stream in streams_to_try:
            try:
                if hasattr(stream, 'buffer') or hasattr(stream, 'write'):
                    return stream
            except (ValueError, AttributeError, OSError):
                continue
        return None
    
    def _is_stream_valid(self, stream):
        """스트림이 유효한지 확인"""
        if stream is None:
            return False
        try:
            if hasattr(stream, 'buffer'):
                buffer = stream.buffer
                if buffer is None:
                    return False
                if hasattr(buffer, 'raw'):
                    raw = buffer.raw
                    if raw is None:
                        return False
            if not hasattr(stream, 'write'):
                return False
            return True
        except (ValueError, AttributeError, OSError):
            return False
    
    def emit(self, record):
        """안전한 로그 출력 (버퍼 분리 오류 방지)"""
        try:
            msg = self.format(record) + self.terminator
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    if hasattr(safe_stream, 'buffer'):
                        try:
                            buffer = safe_stream.buffer
                            if buffer is None:
                                raise ValueError("Buffer is None")
                        except (ValueError, AttributeError):
                            if hasattr(safe_stream, 'write'):
                                safe_stream.write(msg)
                                return
                            else:
                                raise ValueError("No write method")
                    else:
                        safe_stream.write(msg)
                    
                    try:
                        safe_stream.flush()
                    except (ValueError, AttributeError, OSError):
                        pass
                    return
                except (ValueError, AttributeError, OSError) as e:
                    if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                        pass
                    else:
                        pass
            
            try:
                if sys.stderr and hasattr(sys.stderr, 'write'):
                    try:
                        sys.stderr.write(msg)
                        try:
                            sys.stderr.flush()
                        except (ValueError, AttributeError, OSError):
                            pass
                        return
                    except (ValueError, AttributeError, OSError) as e:
                        if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                            pass
            except (ValueError, AttributeError, OSError):
                pass
        except Exception:
            pass
    
    def flush(self):
        """안전한 flush (오류 무시)"""
        try:
            safe_stream = self._get_safe_stream()
            if safe_stream is not None:
                try:
                    safe_stream.flush()
                except (ValueError, AttributeError, OSError):
                    pass
        except (ValueError, AttributeError, OSError):
            pass

# 로깅 설정 (SafeStreamHandler 사용)
def setup_logging(log_level: str = "DEBUG", log_file: str = None):
    """로깅 설정 (Windows PowerShell 호환)
    
    Args:
        log_level: 로그 레벨 (INFO, DEBUG, WARNING, ERROR)
        log_file: 로그 파일 경로 (None이면 자동 생성)
    """
    logger = logging.getLogger("lawfirm_langgraph.tests")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    
    # 로그 파일 경로 설정
    if log_file is None:
        # 환경 변수에서 로그 디렉토리 확인
        log_dir = os.getenv("TEST_LOG_DIR", str(project_root / "logs" / "test"))
        os.makedirs(log_dir, exist_ok=True)
        
        # 타임스탬프 기반 로그 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_query_test_{timestamp}.log")
    
    # 로그 파일 핸들러 추가
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"📝 로그 파일: {log_file}")
    except Exception as e:
        logger.warning(f"⚠️  로그 파일 생성 실패: {e} (콘솔 로그만 사용)")
    
    # 콘솔 핸들러 생성
    try:
        base_handler = logging.StreamHandler(_original_stdout)
    except (ValueError, AttributeError):
        try:
            base_handler = logging.StreamHandler(sys.stdout)
        except (ValueError, AttributeError):
            base_handler = logging.StreamHandler(sys.stderr)
    
    # SafeStreamHandler로 교체
    safe_handler = SafeStreamHandler(base_handler.stream, _original_stdout)
    safe_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    safe_handler.setFormatter(formatter)
    logger.addHandler(safe_handler)
    
    return logger

# 로그 파일 경로 설정 (환경 변수 또는 자동 생성)
log_file_path = os.getenv("TEST_LOG_FILE", None)
logger = setup_logging(
    log_level=os.getenv("TEST_LOG_LEVEL", "DEBUG"),
    log_file=log_file_path
)


def get_query_from_args() -> str:
    """명령줄 인자에서 질의 추출"""
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    
    test_query = os.getenv('TEST_QUERY')
    if test_query and test_query.strip():
        return test_query.strip()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        if arg in ['-f', '--file']:
            if len(sys.argv) > 2:
                file_path = sys.argv[2]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
                except Exception as e:
                    logger.error(f"파일 읽기 오류: {e}")
                    return default_queries[1]
            else:
                logger.error("파일 경로를 지정해주세요")
                return default_queries[1]
        
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                return default_queries[idx]
        
        return " ".join(sys.argv[1:])
    
    return default_queries[1]


def _setup_mlflow_config():
    """MLflow 설정 초기화"""
    if not os.getenv('USE_MLFLOW_INDEX'):
        os.environ['USE_MLFLOW_INDEX'] = 'true'
        logger.info("   📌 USE_MLFLOW_INDEX=true 설정됨")
    
    if not os.getenv('MLFLOW_TRACKING_URI'):
        mlflow_uri = str(project_root / "mlflow" / "mlruns")
        os.environ['MLFLOW_TRACKING_URI'] = f"file:///{mlflow_uri.replace(chr(92), '/')}"
        logger.info("   📌 MLFLOW_TRACKING_URI 설정됨")
    
    if not os.getenv('MLFLOW_RUN_ID'):
        logger.info("   📌 MLFLOW_RUN_ID 비어있음 - 프로덕션 run 자동 조회 예정")
    else:
        logger.info(f"   📌 MLFLOW_RUN_ID={os.getenv('MLFLOW_RUN_ID')} 설정됨")


def _check_mlflow_index(config_obj):
    """MLflow 인덱스 설정 확인"""
    if config_obj.use_mlflow_index:
        logger.info(f"   ✅ MLflow 인덱스 사용: run_id={config_obj.mlflow_run_id or '자동 조회'}")
        
        try:
            from scripts.rag.mlflow_manager import MLflowFAISSManager
            mlflow_manager = MLflowFAISSManager()
            if mlflow_manager.is_local_filesystem:
                logger.info(f"   ✅ 로컬 파일 시스템 모드: {mlflow_manager.local_base_path}")
                
                run_id = config_obj.mlflow_run_id or mlflow_manager.get_production_run()
                if run_id:
                    run_info = mlflow_manager.client.get_run(run_id)
                    tags = run_info.data.tags if hasattr(run_info.data, 'tags') else {}
                    version_name = tags.get('version', None)
                    
                    if version_name:
                        vector_store_path = project_root / "data" / "vector_store" / version_name
                        index_path = vector_store_path / "index.faiss"
                        if index_path.exists():
                            logger.info(f"   ✅ data/vector_store 인덱스 존재: {index_path}")
                        else:
                            logger.info(f"   ℹ️  data/vector_store 인덱스 없음: {index_path}")
                        
                        artifacts_path = mlflow_manager._get_local_artifact_path(run_id, "faiss_index")
                        mlflow_index_path = artifacts_path / "index.faiss"
                        if mlflow_index_path.exists():
                            logger.info(f"   ✅ MLflow 로컬 경로 인덱스 존재: {mlflow_index_path}")
                        else:
                            logger.info(f"   ℹ️  MLflow 로컬 경로 인덱스 없음: {mlflow_index_path}")
            else:
                logger.info(f"   🌐 원격 서버 모드: {mlflow_manager.tracking_uri}")
        except Exception as e:
            logger.debug(f"   MLflow 매니저 확인 실패: {e}")
    else:
        logger.info("   ℹ️  MLflow 인덱스 미사용 (DB 기반 인덱스 사용)")


def _extract_and_normalize_answer(result):
    """답변 추출 및 정규화"""
    answer_raw = result.get("answer", "")
    
    try:
        from lawfirm_langgraph.core.workflow.utils.workflow_utils import WorkflowUtils
    except ImportError:
        try:
            from core.workflow.utils.workflow_utils import WorkflowUtils
        except ImportError:
            WorkflowUtils = None
    
    if WorkflowUtils:
        answer = WorkflowUtils.normalize_answer(answer_raw)
    else:
        if isinstance(answer_raw, dict):
            answer = answer_raw.get("content", answer_raw.get("text", str(answer_raw)))
        else:
            answer = str(answer_raw) if answer_raw else ""
        answer = answer.strip() if isinstance(answer, str) else ""
    
    return answer


def _analyze_retrieved_docs(retrieved_docs):
    """retrieved_docs 분석 및 통계 수집"""
    type_counts = {}
    statute_articles = []
    version_counts = {}
    scores = []
    
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            if doc_type == "statute_article":
                statute_articles.append(doc)
            
            version_id = doc.get("embedding_version_id") or doc.get("metadata", {}).get("embedding_version_id")
            if version_id:
                version_counts[version_id] = version_counts.get(version_id, 0) + 1
            
            score = doc.get("score") or doc.get("similarity") or doc.get("relevance_score")
            if score is not None:
                scores.append(float(score))
    
    return type_counts, statute_articles, version_counts, scores


def _log_retrieved_docs(retrieved_docs):
    """retrieved_docs 로깅"""
    if not retrieved_docs:
        logger.warning("\n⚠️  검색된 참고자료 (retrieved_docs)가 없습니다!")
        logger.warning("   - 데이터베이스/벡터스토어에서 검색이 수행되지 않았거나")
        logger.warning("   - 검색 결과가 없을 수 있습니다.")
        return
    
    logger.info(f"\n🔍 검색된 참고자료 (retrieved_docs) ({len(retrieved_docs)}개):")
    
    type_counts, statute_articles, version_counts, scores = _analyze_retrieved_docs(retrieved_docs)
    
    logger.info(f"   타입 분포: {type_counts}")
    if statute_articles:
        logger.info(f"   statute_article 타입 문서: {len(statute_articles)}개")
    
    if version_counts:
        logger.info(f"   📊 Embedding 버전 분포: {version_counts}")
    else:
        logger.warning("   ⚠️  검색 결과에 embedding_version_id가 없습니다!")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        logger.info(f"   📊 유사도 점수 분포: 평균={avg_score:.3f}, 최대={max_score:.3f}, 최소={min_score:.3f}")
    
    for i, doc in enumerate(retrieved_docs[:10], 1):
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id") or doc.get("_id") or f"doc_{i}"
            doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
            title = doc.get("title") or doc.get("name") or doc.get("content", "")[:50] or "제목 없음"
            search_type = doc.get("search_type") or doc.get("search_method") or "unknown"
            logger.info(f"   {i}. [{doc_type}] {title} (ID: {doc_id}, 검색방법: {search_type})")
            
            if doc_type == "statute_article":
                statute_name = doc.get("statute_name") or doc.get("law_name") or doc.get("metadata", {}).get("statute_name") or doc.get("metadata", {}).get("law_name")
                article_no = doc.get("article_no") or doc.get("article_number") or doc.get("metadata", {}).get("article_no") or doc.get("metadata", {}).get("article_number")
                clause_no = doc.get("clause_no") or doc.get("metadata", {}).get("clause_no")
                item_no = doc.get("item_no") or doc.get("metadata", {}).get("item_no")
                logger.info(f"      - statute_name: {statute_name}")
                logger.info(f"      - article_no: {article_no}")
                logger.info(f"      - clause_no: {clause_no}")
                logger.info(f"      - item_no: {item_no}")
            
            if doc.get("score"):
                logger.info(f"      - 점수: {doc.get('score'):.4f}")
            
            version_id = doc.get("embedding_version_id") or doc.get("metadata", {}).get("embedding_version_id")
            if version_id:
                logger.info(f"      - embedding_version_id: {version_id}")
            
            if doc.get("metadata") and doc_type != "statute_article":
                logger.info(f"      - 메타데이터: {doc.get('metadata')}")
        else:
            logger.info(f"   {i}. {str(doc)[:100]}")
    
    if len(retrieved_docs) > 10:
        logger.info(f"   ... (총 {len(retrieved_docs)}개)")


def _log_performance_metrics(service):
    """성능 메트릭 로깅"""
    logger.info("\n" + "="*80)
    logger.info("📊 분류 성능 메트릭 (최적화 결과)")
    logger.info("="*80)
    
    try:
        if hasattr(service, 'workflow') and hasattr(service.workflow, 'stats'):
            stats = service.workflow.stats
            if stats:
                unified_calls = stats.get('unified_classification_calls', 0)
                unified_llm_calls = stats.get('unified_classification_llm_calls', 0)
                avg_unified_time = stats.get('avg_unified_classification_time', 0.0)
                total_unified_time = stats.get('total_unified_classification_time', 0.0)
                
                cache_hits = stats.get('complexity_cache_hits', 0)
                cache_misses = stats.get('complexity_cache_misses', 0)
                total_cache_requests = cache_hits + cache_misses
                cache_hit_rate = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
                
                fallback_count = stats.get('complexity_fallback_count', 0)
                
                logger.info("\n✅ 통합 분류 (단일 프롬프트):")
                logger.info(f"   - 총 호출: {unified_calls}회")
                logger.info(f"   - LLM 호출: {unified_llm_calls}회 (목표: 1회/쿼리)")
                logger.info(f"   - 평균 처리 시간: {avg_unified_time:.3f}초")
                logger.info(f"   - 총 처리 시간: {total_unified_time:.3f}초")
                
                if unified_calls > 0:
                    llm_calls_per_query = unified_llm_calls / unified_calls
                    logger.info(f"   - LLM 호출/쿼리: {llm_calls_per_query:.2f}회 (목표: 1.0회)")
                    if llm_calls_per_query > 1.5:
                        logger.warning("   ⚠️  LLM 호출이 예상보다 많습니다! (목표: 1회)")
                
                logger.info("\n💾 캐시 성능:")
                logger.info(f"   - 캐시 히트: {cache_hits}회")
                logger.info(f"   - 캐시 미스: {cache_misses}회")
                logger.info(f"   - 캐시 히트율: {cache_hit_rate:.1f}%")
                if cache_hit_rate < 50 and total_cache_requests > 5:
                    logger.warning("   ⚠️  캐시 히트율이 낮습니다. 캐시 전략을 검토하세요.")
                
                logger.info("\n🔄 폴백 사용:")
                logger.info(f"   - 폴백 호출: {fallback_count}회")
                if fallback_count > 0:
                    fallback_rate = (fallback_count / unified_calls * 100) if unified_calls > 0 else 0
                    logger.info(f"   - 폴백 비율: {fallback_rate:.1f}%")
                    if fallback_rate > 10:
                        logger.warning("   ⚠️  폴백 비율이 높습니다. LLM 호출 실패 원인을 확인하세요.")
                    
                    fallback_reasons = stats.get('fallback_reasons', {})
                    if fallback_reasons:
                        logger.info("\n   📋 폴백 원인 분석:")
                        for reason, count in sorted(fallback_reasons.items(), key=lambda x: x[1], reverse=True):
                            reason_rate = (count / fallback_count * 100) if fallback_count > 0 else 0
                            logger.info(f"      - {reason}: {count}회 ({reason_rate:.1f}%)")
                            if reason in ["LLM timeout", "Network error", "Rate limit"]:
                                logger.warning(f"         ⚠️  {reason} - 재시도 메커니즘 고려 필요")
                
                if unified_calls > 0:
                    logger.info("\n📈 개선 효과 (체인 방식 대비):")
                    old_llm_calls = unified_calls * 4
                    new_llm_calls = unified_llm_calls
                    reduction = ((old_llm_calls - new_llm_calls) / old_llm_calls * 100) if old_llm_calls > 0 else 0
                    logger.info(f"   - 기존 LLM 호출 (예상): {old_llm_calls}회")
                    logger.info(f"   - 현재 LLM 호출: {new_llm_calls}회")
                    logger.info(f"   - LLM 호출 감소: {reduction:.1f}%")
                    if reduction >= 70:
                        logger.info("   ✅ 목표 달성! (75% 감소 목표)")
                    elif reduction >= 50:
                        logger.warning("   ⚠️  개선되었지만 목표에 미달 (75% 목표)")
                    else:
                        logger.warning("   ⚠️  개선 효과가 낮습니다. 원인 확인 필요")
            else:
                logger.warning("   ⚠️  통계가 활성화되지 않았습니다.")
        else:
            logger.warning("   ⚠️  통계 정보를 가져올 수 없습니다.")
    except Exception as e:
        logger.warning(f"   ⚠️  성능 메트릭 출력 실패: {e}")
    
    logger.info("\n" + "="*80)


def _evaluate_answer_quality(answer, answer_length, answer_is_valid, has_error_message, retrieved_docs, sources):
    """답변 품질 평가"""
    logger.info("\n" + "="*80)
    logger.info("📊 답변 품질 종합 평가")
    logger.info("="*80)
    
    answer_quality_score = 0
    quality_checks = []
    
    if answer and answer_length > 0:
        answer_quality_score += 25
        quality_checks.append("✅ 답변 존재")
    else:
        quality_checks.append("❌ 답변 없음")
    
    if answer_is_valid:
        answer_quality_score += 25
        quality_checks.append(f"✅ 최소 길이 충족 ({answer_length}자 >= {MIN_ANSWER_LENGTH}자)")
    else:
        quality_checks.append(f"⚠️  최소 길이 미달 ({answer_length}자 < {MIN_ANSWER_LENGTH}자)")
    
    if not has_error_message:
        answer_quality_score += 25
        quality_checks.append("✅ 오류 메시지 없음")
    else:
        quality_checks.append("❌ 오류 메시지 포함")
    
    has_sources = len(retrieved_docs) > 0 or len(sources) > 0
    if has_sources:
        answer_quality_score += 25
        quality_checks.append(f"✅ 참고자료 존재 ({len(retrieved_docs)}개 retrieved_docs, {len(sources)}개 sources)")
    else:
        quality_checks.append("⚠️  참고자료 없음")
    
    logger.info(f"\n   품질 점수: {answer_quality_score}/100")
    for check in quality_checks:
        logger.info(f"   {check}")
    
    if answer_quality_score >= 100:
        quality_grade = "🟢 우수"
    elif answer_quality_score >= 75:
        quality_grade = "🟡 양호"
    elif answer_quality_score >= 50:
        quality_grade = "🟠 보통"
    else:
        quality_grade = "🔴 불량"
    
    logger.info(f"\n   종합 평가: {quality_grade}")
    
    if answer_quality_score < 75:
        logger.warning("\n⚠️  답변 품질이 기준 미만입니다!")
        logger.warning("   다음 사항을 확인하세요:")
        if not answer or answer_length == 0:
            logger.warning("   - 답변이 생성되지 않았습니다")
        if not answer_is_valid:
            logger.warning(f"   - 답변이 너무 짧습니다 (최소 {MIN_ANSWER_LENGTH}자 필요)")
        if has_error_message:
            logger.warning("   - 답변에 오류 메시지가 포함되어 있습니다")
        if not has_sources:
            logger.warning("   - 참고자료가 없어 답변의 신뢰성이 낮을 수 있습니다")
    
    return answer_quality_score


async def run_query_test(query: str, enable_profiling: bool = False, enable_memory_monitoring: bool = False):
    """질의 테스트 실행
    
    Args:
        query: 테스트할 질의
        enable_profiling: 프로파일링 활성화 여부
        enable_memory_monitoring: 메모리 모니터링 활성화 여부
    """
    logger.info("\n" + "="*80)
    logger.info("LangGraph 질의 테스트")
    logger.info("="*80)
    logger.info(f"\n📋 질의: {query}\n")
    
    # 프로파일링 설정
    profiler = None
    profile_file = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_file = str(project_root / "logs" / "test" / f"profile_{timestamp}.prof")
        os.makedirs(os.path.dirname(profile_file), exist_ok=True)
        logger.info(f"📊 프로파일링 활성화: {profile_file}")
    
    # 메모리 모니터링 설정
    memory_snapshots = []
    if enable_memory_monitoring:
        tracemalloc.start()
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append(("초기", initial_memory))
            logger.info(f"💾 메모리 모니터링 활성화: 초기 메모리 {initial_memory:.2f} MB")
        else:
            logger.warning("💾 psutil이 설치되지 않아 프로세스 메모리 모니터링을 사용할 수 없습니다. tracemalloc만 사용합니다.")
            memory_snapshots.append(("초기", 0))
    
    # 로그 파일 경로 출력 (환경 변수로 설정된 경우)
    log_file_path = os.getenv("TEST_LOG_FILE", None)
    if log_file_path:
        logger.info(f"📝 로그 파일: {log_file_path}")
    else:
        # 자동 생성된 로그 파일 경로 찾기
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(f"📝 로그 파일: {handler.baseFilename}")
                break
    
    try:
        # python-dotenv 경고 억제를 위한 환경 변수 설정
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        
        # Import (순환 import 방지를 위해 함수 내부에서 수행)
        # sys.path가 올바르게 설정되어 있으므로 직접 import 가능
        try:
            from lawfirm_langgraph.config.langgraph_config import LangGraphConfig
        except ImportError:
            # Fallback: 상대 경로
            sys.path.insert(0, str(lawfirm_langgraph_dir))
            from config.langgraph_config import LangGraphConfig
        
        try:
            from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService
        except ImportError:
            # Fallback: 상대 경로
            sys.path.insert(0, str(lawfirm_langgraph_dir))
            from core.workflow.workflow_service import LangGraphWorkflowService
        
        logger.info("1️⃣  설정 로드 중...")
        _setup_mlflow_config()
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False
        logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   ✅ 체크포인트: {config.enable_checkpoint}")
        
        from lawfirm_langgraph.core.utils.config import Config
        config_obj = Config()
        _check_mlflow_index(config_obj)
        
        # 서비스 초기화
        logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        logger.info("   ✅ 서비스 초기화 완료")
        
        # 질의 처리
        logger.info("\n3️⃣  질의 처리 중...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        # 메모리 스냅샷 (처리 전)
        if enable_memory_monitoring:
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(("처리 전", memory_before))
            current, peak = tracemalloc.get_traced_memory()
            traced_before = current / 1024 / 1024
            if PSUTIL_AVAILABLE:
                logger.info(f"   💾 메모리 (처리 전): {memory_before:.2f} MB (traced: {traced_before:.2f} MB)")
            else:
                logger.info(f"   💾 메모리 (처리 전): traced: {traced_before:.2f} MB")
        
        try:
            result = await service.process_query(
                query=query,
                session_id="query_test",
                enable_checkpoint=False,
                use_astream_events=True
            )
        except asyncio.CancelledError:
            logger.warning("\n⚠️  작업이 취소되었습니다 (CancelledError)")
            if enable_profiling and profiler:
                profiler.disable()
            if enable_memory_monitoring:
                tracemalloc.stop()
            raise
        except KeyboardInterrupt:
            logger.warning("\n⚠️  사용자에 의해 중단되었습니다 (KeyboardInterrupt)")
            if enable_profiling and profiler:
                profiler.disable()
            if enable_memory_monitoring:
                tracemalloc.stop()
            raise
        finally:
            # 메모리 스냅샷 (처리 후) - 항상 실행되도록 finally 블록에 배치
            if enable_memory_monitoring:
                try:
                    if PSUTIL_AVAILABLE:
                        process = psutil.Process(os.getpid())
                        memory_after = process.memory_info().rss / 1024 / 1024
                        memory_snapshots.append(("처리 후", memory_after))
                    current, peak = tracemalloc.get_traced_memory()
                    traced_after = current / 1024 / 1024
                    traced_peak = peak / 1024 / 1024
                    if PSUTIL_AVAILABLE:
                        logger.info(f"   💾 메모리 (처리 후): {memory_after:.2f} MB (traced: {traced_after:.2f} MB, peak: {traced_peak:.2f} MB)")
                    else:
                        logger.info(f"   💾 메모리 (처리 후): traced: {traced_after:.2f} MB, peak: {traced_peak:.2f} MB")
                except Exception as e:
                    logger.warning(f"   ⚠️  메모리 스냅샷 저장 실패: {e}")
        
        # 결과 출력
        logger.info("\n4️⃣  결과:")
        logger.info("="*80)
        
        answer_raw = result.get("answer", "")
        answer = _extract_and_normalize_answer(result)
        answer_length = len(answer) if isinstance(answer, str) else 0
        answer_is_valid = answer_length >= MIN_ANSWER_LENGTH
        has_error_message = any(pattern in answer for pattern in ERROR_PATTERNS) if isinstance(answer, str) else False
        
        if answer and answer_length > 0:
            quality_status = "✅" if answer_is_valid else "⚠️"
            logger.info(f"\n📝 답변 ({answer_length}자) {quality_status}:")
            logger.info("-" * 80)
            logger.info(str(answer))
            
            if not answer_is_valid:
                logger.warning(f"\n⚠️  답변이 너무 짧습니다! (최소 {MIN_ANSWER_LENGTH}자 필요, 현재 {answer_length}자)")
                logger.warning("   가능한 원인:")
                logger.warning("   1. 답변 생성 중 오류 발생")
                logger.warning("   2. 검색 결과가 부족하여 답변 생성 실패")
                logger.warning("   3. LLM 응답이 제대로 처리되지 않음")
            
            if has_error_message:
                logger.warning("\n⚠️  답변에 오류 메시지가 포함되어 있습니다!")
                logger.warning("   답변이 정상적으로 생성되지 않았을 수 있습니다.")
        else:
            logger.error("\n❌ 답변이 없습니다!")
            logger.error("   가능한 원인:")
            logger.error("   1. 워크플로우 실행 중 오류 발생")
            logger.error("   2. 답변 생성 노드가 실행되지 않음")
            logger.error("   3. state에서 answer가 손실됨")
            
            errors = result.get("errors", [])
            if errors:
                logger.error(f"\n   발견된 오류 ({len(errors)}개):")
                for i, error in enumerate(errors[:5], 1):
                    logger.error(f"   {i}. {error}")
        
        retrieved_docs = result.get("retrieved_docs", [])
        sources = result.get("sources", [])
        _log_retrieved_docs(retrieved_docs)
        
        if sources:
            logger.info(f"\n📚 소스 (sources) ({len(sources)}개):")
            for i, source in enumerate(sources[:10], 1):
                if isinstance(source, dict):
                    source_id = source.get("id") or source.get("doc_id") or source.get("_id") or f"source_{i}"
                    source_name = source.get("name") or source.get("title") or source.get("content", "")[:50] or "제목 없음"
                    logger.info(f"   {i}. {source_name} (ID: {source_id})")
                else:
                    logger.info(f"   {i}. {source}")
            if len(sources) > 10:
                logger.info(f"   ... (총 {len(sources)}개)")
        else:
            logger.warning("\n⚠️  소스 (sources)가 없습니다!")
            if retrieved_docs:
                logger.warning(f"   - retrieved_docs는 {len(retrieved_docs)}개 있지만 sources로 변환되지 않았습니다.")
                logger.warning("   - prepare_final_response_part에서 sources 생성 과정을 확인하세요.")
            else:
                logger.warning("   - retrieved_docs도 없어 sources를 생성할 수 없습니다.")
        
        # sources_detail
        sources_detail = result.get("sources_detail", [])
        if sources_detail:
            logger.info(f"\n📋 소스 상세 (sources_detail) ({len(sources_detail)}개):")
            for i, detail in enumerate(sources_detail[:5], 1):
                if isinstance(detail, dict):
                    name = detail.get("name") or detail.get("title") or "제목 없음"
                    doc_id = detail.get("id") or detail.get("doc_id") or f"detail_{i}"
                    source_type = detail.get("type") or detail.get("source_type") or "unknown"
                    logger.info(f"   {i}. [{source_type}] {name} (ID: {doc_id})")
                else:
                    logger.info(f"   {i}. {detail}")
            if len(sources_detail) > 5:
                logger.info(f"   ... (총 {len(sources_detail)}개)")
        
        # 법률 참조
        legal_references = result.get("legal_references", [])
        if legal_references:
            logger.info(f"\n⚖️  법률 참조 ({len(legal_references)}개):")
            for i, ref in enumerate(legal_references[:5], 1):
                logger.info(f"   {i}. {ref}")
            if len(legal_references) > 5:
                logger.info(f"   ... (총 {len(legal_references)}개)")
        else:
            logger.warning("\n⚠️  법률 참조 (legal_references)가 없습니다!")
            if retrieved_docs:
                # statute_article 타입 문서 확인
                statute_articles = [doc for doc in retrieved_docs if isinstance(doc, dict) and (doc.get("type") == "statute_article" or doc.get("source_type") == "statute_article" or doc.get("metadata", {}).get("source_type") == "statute_article")]
                if statute_articles:
                    logger.warning(f"   - retrieved_docs에 statute_article 타입 문서가 {len(statute_articles)}개 있지만 legal_references로 변환되지 않았습니다.")
                    logger.info("\n   statute_article 문서 샘플 (처음 3개):")
                    for i, doc in enumerate(statute_articles[:3], 1):
                        logger.info(f"   {i}. type: {doc.get('type')}, statute_name: {doc.get('statute_name')}, law_name: {doc.get('law_name')}, article_no: {doc.get('article_no')}, metadata: {doc.get('metadata', {})}")
                else:
                    logger.warning("   - retrieved_docs에 statute_article 타입 문서가 없습니다.")
                    logger.info("\n   retrieved_docs 타입 분포:")
                    type_counts = {}
                    for doc in retrieved_docs:
                        if isinstance(doc, dict):
                            doc_type = doc.get("type") or doc.get("source_type") or doc.get("metadata", {}).get("source_type", "unknown")
                            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    for doc_type, count in type_counts.items():
                        logger.info(f"      - {doc_type}: {count}개")
        
        # 관련 질문 (related_questions)
        related_questions = result.get("metadata", {}).get("related_questions", [])
        if related_questions:
            logger.info(f"\n❓ 관련 질문 (related_questions) ({len(related_questions)}개):")
            for i, question in enumerate(related_questions[:5], 1):
                logger.info(f"   {i}. {question}")
            if len(related_questions) > 5:
                logger.info(f"   ... (총 {len(related_questions)}개)")
        else:
            logger.warning("\n⚠️  관련 질문 (related_questions)가 없습니다!")
            logger.warning("   가능한 원인:")
            logger.warning("   1. phase_info에 suggested_questions가 없을 수 있습니다.")
            logger.warning("   2. conversation_flow_tracker가 초기화되지 않았을 수 있습니다.")
            logger.warning("   3. metadata에 저장되지 않았을 수 있습니다.")
        
        metadata = result.get("metadata", {})
        if metadata:
            logger.info("\n📊 메타데이터:")
            for key, value in list(metadata.items())[:10]:
                if key == "related_questions":
                    logger.info(f"   {key}: {value} ({len(value) if isinstance(value, list) else 'N/A'}개)")
                else:
                    logger.info(f"   {key}: {value}")
        
        # 신뢰도
        confidence = result.get("confidence", 0.0)
        if confidence:
            logger.info(f"\n🎯 신뢰도: {confidence:.2f}")
        
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
        answer_quality_score = _evaluate_answer_quality(
            answer, answer_length, answer_is_valid, has_error_message, retrieved_docs, sources
        )
        
        # 디버깅: retrieved_docs와 sources 관계 분석
        logger.info("\n" + "="*80)
        logger.info("🔍 디버깅 정보:")
        logger.info("="*80)
        
        if retrieved_docs and not sources:
            logger.warning("⚠️  retrieved_docs는 있지만 sources가 없습니다!")
            logger.warning("   가능한 원인:")
            logger.warning("   1. prepare_final_response_part가 실행되지 않았을 수 있습니다.")
            logger.warning("   2. retrieved_docs의 형식이 sources 생성 로직과 맞지 않을 수 있습니다.")
            logger.warning("   3. source_type이 없거나 인식되지 않는 형식일 수 있습니다.")
            logger.info("\n   retrieved_docs 샘플 (처음 3개):")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                logger.info(f"   {i}. {doc}")
        elif not retrieved_docs and not sources:
            logger.warning("⚠️  retrieved_docs와 sources 모두 없습니다!")
            logger.warning("   가능한 원인:")
            logger.warning("   1. 검색이 수행되지 않았을 수 있습니다 (direct_answer 노드 사용).")
            logger.warning("   2. 검색 결과가 없을 수 있습니다.")
            logger.warning("   3. retrieved_docs가 state에서 손실되었을 수 있습니다.")
        elif retrieved_docs and sources:
            logger.info(f"✅ retrieved_docs ({len(retrieved_docs)}개) → sources ({len(sources)}개) 변환 성공")
            if len(retrieved_docs) > len(sources):
                logger.warning("   ⚠️  일부 retrieved_docs가 sources로 변환되지 않았습니다.")
                logger.warning(f"   ({len(retrieved_docs) - len(sources)}개 누락)")
        
        # legal_references 디버깅
        if retrieved_docs and not legal_references:
            statute_articles = [doc for doc in retrieved_docs if isinstance(doc, dict) and (doc.get("type") == "statute_article" or doc.get("source_type") == "statute_article" or doc.get("metadata", {}).get("source_type") == "statute_article")]
            if statute_articles:
                logger.warning(f"\n⚠️  retrieved_docs에 statute_article 타입 문서가 {len(statute_articles)}개 있지만 legal_references로 변환되지 않았습니다!")
                logger.warning("   가능한 원인:")
                logger.warning("   1. prepare_final_response_part가 실행되지 않았을 수 있습니다.")
                logger.warning("   2. statute_name이나 article_no 필드가 없을 수 있습니다.")
                logger.warning("   3. legal_references 생성 로직이 실행되지 않았을 수 있습니다.")
                logger.info("\n   statute_article 문서 상세 (처음 3개):")
                for i, doc in enumerate(statute_articles[:3], 1):
                    logger.info(f"   {i}. 전체 구조:")
                    logger.info(f"      {doc}")
        
        if not related_questions:
            logger.warning("\n⚠️  related_questions가 없습니다!")
            logger.warning("   가능한 원인:")
            logger.warning("   1. phase_info에 suggested_questions가 없을 수 있습니다.")
            logger.warning("   2. conversation_flow_tracker가 초기화되지 않았을 수 있습니다.")
            logger.warning("   3. metadata에 저장되지 않았을 수 있습니다.")
            # phase_info 확인
            if "phase_info" in result:
                phase_info = result.get("phase_info", {})
                logger.info("\n   phase_info 확인:")
                logger.info(f"      phase_info keys: {list(phase_info.keys()) if isinstance(phase_info, dict) else 'N/A'}")
                if isinstance(phase_info, dict) and "phase2" in phase_info:
                    phase2 = phase_info.get("phase2", {})
                    if isinstance(phase2, dict) and "flow_tracking_info" in phase2:
                        flow_tracking = phase2.get("flow_tracking_info", {})
                        if isinstance(flow_tracking, dict) and "suggested_questions" in flow_tracking:
                            suggested_questions = flow_tracking.get("suggested_questions", [])
                            logger.info(f"      suggested_questions in phase_info: {len(suggested_questions)}개")
                        else:
                            logger.warning("      suggested_questions가 phase_info에 없습니다.")
        
        # needs_search 확인
        needs_search = result.get("needs_search", True)
        logger.info(f"\n   needs_search: {needs_search}")
        if not needs_search:
            logger.info("   → direct_answer 노드가 사용되어 검색이 수행되지 않았을 수 있습니다.")
        
        _log_performance_metrics(service)
        
        # 프로파일링 결과 저장 및 출력
        if enable_profiling and profiler:
            try:
                profiler.disable()
                profiler.dump_stats(profile_file)
                logger.info("\n" + "="*80)
                logger.info("📊 프로파일링 결과")
                logger.info("="*80)
                logger.info(f"프로파일링 결과 저장: {profile_file}")
                
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                logger.info("\n상위 20개 함수 (cumulative time):")
                stats.print_stats(20)
                
                logger.info("\n상위 20개 함수 (tottime):")
                stats.sort_stats('tottime')
                stats.print_stats(20)
            except Exception as e:
                logger.error(f"⚠️  프로파일링 결과 저장 실패: {e}")
        
        # 메모리 모니터링 결과 출력
        if enable_memory_monitoring:
            try:
                logger.info("\n" + "="*80)
                logger.info("💾 메모리 사용량 모니터링 결과")
                logger.info("="*80)
                
                if PSUTIL_AVAILABLE:
                    for label, memory_mb in memory_snapshots:
                        if memory_mb > 0:
                            logger.info(f"   {label}: {memory_mb:.2f} MB")
                    
                    if len(memory_snapshots) >= 2 and memory_snapshots[0][1] > 0:
                        initial_memory = memory_snapshots[0][1]
                        final_memory = memory_snapshots[-1][1]
                        memory_increase = final_memory - initial_memory
                        logger.info(f"\n   프로세스 메모리 증가량: {memory_increase:.2f} MB ({memory_increase / initial_memory * 100:.1f}%)")
                
                # tracemalloc 상세 정보
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                logger.info("\n   Python 메모리 할당 상위 10개 (tracemalloc):")
                total_size = 0
                for index, stat in enumerate(top_stats[:10], 1):
                    total_size += stat.size
                    logger.info(f"   {index}. {stat}")
                
                logger.info(f"\n   총 추적된 Python 메모리: {total_size / 1024 / 1024:.2f} MB")
            except Exception as e:
                logger.error(f"⚠️  메모리 모니터링 결과 출력 실패: {e}")
            finally:
                try:
                    tracemalloc.stop()
                except Exception:
                    pass
        
        # 최종 검증 및 요약
        test_passed = True
        critical_issues = []
        warnings = []
        
        # 1. 답변 존재 및 품질 확인 (강화된 검증)
        if not answer or answer_length == 0:
            test_passed = False
            critical_issues.append("답변이 없습니다 (0자)")
            # 상세 디버깅 정보
            logger.error("\n   📋 답변 없음 상세 분석:")
            logger.error(f"      - answer 타입: {type(answer_raw).__name__}")
            logger.error(f"      - answer_raw 값: {repr(answer_raw[:200]) if answer_raw else 'None'}")
            logger.error(f"      - result['answer'] 존재: {'answer' in result}")
            logger.error(f"      - result keys: {list(result.keys())[:20]}")
            if "errors" in result:
                logger.error(f"      - errors: {result.get('errors', [])[:5]}")
            if "processing_steps" in result:
                logger.error(f"      - 마지막 processing_steps: {result.get('processing_steps', [])[-3:]}")
        elif not answer_is_valid:
            test_passed = False
            critical_issues.append(f"답변이 너무 짧습니다 ({answer_length}자 < {MIN_ANSWER_LENGTH}자)")
            # 상세 디버깅 정보
            logger.warning("\n   📋 답변 짧음 상세 분석:")
            logger.warning(f"      - 답변 내용 (처음 200자): {answer[:200]}")
            logger.warning(f"      - 답변 길이: {answer_length}자")
            logger.warning(f"      - 최소 요구 길이: {MIN_ANSWER_LENGTH}자")
        elif has_error_message:
            test_passed = False
            critical_issues.append("답변에 오류 메시지가 포함되어 있습니다")
            # 상세 디버깅 정보
            logger.error("\n   📋 오류 메시지 상세 분석:")
            logger.error(f"      - 답변 내용: {answer[:500]}")
            for pattern in ERROR_PATTERNS:
                if pattern in answer:
                    logger.error(f"      - 발견된 패턴: '{pattern}'")
        
        # 2. 워크플로우 실행 확인 (강화된 로깅)
        errors = result.get("errors", [])
        if errors:
            test_passed = False
            critical_issues.append(f"워크플로우 실행 중 {len(errors)}개 오류 발생")
            # 상세 디버깅 정보
            logger.error("\n   📋 워크플로우 오류 상세:")
            for i, error in enumerate(errors[:10], 1):
                logger.error(f"      {i}. {error}")
            if len(errors) > 10:
                logger.error(f"      ... (총 {len(errors)}개 오류, 처음 10개만 표시)")
        
        if processing_time > MAX_PROCESSING_TIME_WARNING:
            warnings.append(f"처리 시간이 매우 깁니다 ({processing_time:.2f}초)")
            logger.warning(f"⚠️  처리 시간이 매우 깁니다 ({processing_time:.2f}초)")
        
        # 4. 검색 결과 확인 (로깅 개선)
        if not retrieved_docs and not sources:
            warnings.append("검색 결과가 없습니다")
            logger.warning("\n   📋 검색 결과 없음 상세 분석:")
            logger.warning(f"      - needs_search: {result.get('needs_search', 'N/A')}")
            logger.warning(f"      - query_type: {result.get('query_type', 'N/A')}")
            logger.warning(f"      - complexity_level: {result.get('complexity_level', 'N/A')}")
            if "metadata" in result:
                metadata = result.get("metadata", {})
                logger.warning(f"      - metadata keys: {list(metadata.keys())[:10]}")
        
        # 5. State 구조 디버깅 정보 (오류 발생 시)
        if not test_passed or warnings:
            logger.info("\n   📋 State 구조 디버깅 정보:")
            logger.info(f"      - result keys: {list(result.keys())}")
            logger.info(f"      - answer 존재: {'answer' in result}")
            logger.info(f"      - retrieved_docs 존재: {'retrieved_docs' in result}")
            logger.info(f"      - sources 존재: {'sources' in result}")
            logger.info(f"      - errors 존재: {'errors' in result}")
            logger.info(f"      - metadata 존재: {'metadata' in result}")
            if "metadata" in result:
                metadata = result.get("metadata", {})
                logger.info(f"      - metadata keys: {list(metadata.keys())[:15]}")
        
        # 최종 결과 출력
        if test_passed and not warnings:
            logger.info("✅ 테스트 완료! (모든 검증 통과)")
        elif test_passed and warnings:
            logger.warning("⚠️  테스트 완료! (경고 사항 있음)")
            logger.warning("\n   경고 사항:")
            for i, warning in enumerate(warnings, 1):
                logger.warning(f"   {i}. {warning}")
        else:
            logger.error("❌ 테스트 실패! (중요 문제 발견)")
            logger.error("\n   발견된 문제:")
            for i, issue in enumerate(critical_issues, 1):
                logger.error(f"   {i}. {issue}")
            if warnings:
                logger.warning("\n   추가 경고:")
                for i, warning in enumerate(warnings, 1):
                    logger.warning(f"   {i}. {warning}")
        
        logger.info("="*80)
        
        return result
        
    except asyncio.CancelledError:
        logger.warning("\n⚠️  작업이 취소되었습니다 (CancelledError)")
        if enable_profiling and profiler:
            try:
                profiler.disable()
                if profile_file:
                    profiler.dump_stats(profile_file)
                    logger.info(f"📊 프로파일링 결과 저장: {profile_file}")
            except Exception:
                pass
        if enable_memory_monitoring:
            try:
                tracemalloc.stop()
            except Exception:
                pass
        raise
    except KeyboardInterrupt:
        logger.warning("\n⚠️  사용자에 의해 중단되었습니다 (KeyboardInterrupt)")
        if enable_profiling and profiler:
            try:
                profiler.disable()
                if profile_file:
                    profiler.dump_stats(profile_file)
                    logger.info(f"📊 프로파일링 결과 저장: {profile_file}")
            except Exception:
                pass
        if enable_memory_monitoring:
            try:
                tracemalloc.stop()
            except Exception:
                pass
        raise
    except ImportError as e:
        logger.error(f"\n❌ Import 오류: {e}")
        logger.error("\n필요한 패키지가 설치되어 있는지 확인하세요.")
        logger.error(f"   프로젝트 루트: {project_root}")
        logger.error(f"   lawfirm_langgraph 디렉토리: {lawfirm_langgraph_dir}")
        import sys
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {type(e).__name__}: {e}", exc_info=True)
        
        # 프로파일링 및 메모리 모니터링 정리
        if enable_profiling and profiler:
            try:
                profiler.disable()
                if profile_file:
                    profiler.dump_stats(profile_file)
                    logger.info(f"📊 프로파일링 결과 저장: {profile_file}")
            except Exception:
                pass
        if enable_memory_monitoring:
            try:
                tracemalloc.stop()
            except Exception:
                pass
        
        # 상세 디버깅 정보 출력
        logger.error("\n📋 오류 상세 분석:")
        logger.error(f"   - 오류 타입: {type(e).__name__}")
        logger.error(f"   - 오류 메시지: {str(e)}")
        
        # State 정보 (가능한 경우)
        try:
            if 'result' in locals():
                logger.error(f"   - result 타입: {type(result).__name__}")
                if isinstance(result, dict):
                    logger.error(f"   - result keys: {list(result.keys())[:20]}")
                    if "answer" in result:
                        logger.error(f"   - answer 존재: {bool(result.get('answer'))}")
                        logger.error(f"   - answer 길이: {len(str(result.get('answer', '')))}")
                    if "errors" in result:
                        logger.error(f"   - errors: {result.get('errors', [])[:5]}")
        except Exception:
            pass
        
        # 서비스 정보 (가능한 경우)
        try:
            if 'service' in locals():
                logger.error(f"   - service 타입: {type(service).__name__}")
                if hasattr(service, 'workflow'):
                    logger.error(f"   - workflow 존재: {service.workflow is not None}")
        except Exception:
            pass
        
        import sys
        sys.exit(1)


def main():
    """메인 실행 함수"""
    try:
        # 프로파일링 및 메모리 모니터링 옵션 확인
        enable_profiling = os.getenv("ENABLE_PROFILING", "false").lower() in ("true", "1", "yes")
        enable_memory_monitoring = os.getenv("ENABLE_MEMORY_MONITORING", "false").lower() in ("true", "1", "yes")
        
        query = get_query_from_args()
        
        if not query:
            logger.error("질의를 입력해주세요.")
            logger.info("\n사용법:")
            logger.info("  python run_query_test.py \"질의 내용\"")
            logger.info("  python run_query_test.py 0  # 기본 질의 선택")
            logger.info("  $env:TEST_QUERY='질의내용'; python run_query_test.py")
            logger.info("\n프로파일링 및 메모리 모니터링:")
            logger.info("  $env:ENABLE_PROFILING='true'; python run_query_test.py \"질의 내용\"")
            logger.info("  $env:ENABLE_MEMORY_MONITORING='true'; python run_query_test.py \"질의 내용\"")
            return 1
        
        if enable_profiling:
            logger.info("📊 프로파일링 모드 활성화")
        if enable_memory_monitoring:
            logger.info("💾 메모리 모니터링 모드 활성화")
        
        asyncio.run(run_query_test(query, enable_profiling, enable_memory_monitoring))
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except asyncio.CancelledError:
        logger.warning("\n\n⚠️  작업이 취소되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

