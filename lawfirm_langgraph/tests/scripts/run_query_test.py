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
import warnings
warnings.filterwarnings('ignore', message='.*python-dotenv.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*python-dotenv.*')
warnings.filterwarnings('ignore', category=Warning)

import asyncio
import logging
from pathlib import Path
from datetime import datetime

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

# 로깅 설정 (SafeStreamHandler 사용)
def setup_logging(log_level: str = "INFO"):
    """로깅 설정 (Windows PowerShell 호환)"""
    logger = logging.getLogger("lawfirm_langgraph.tests")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    
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
                        # 버퍼 분리 오류 방지를 위한 추가 검증
                        if hasattr(safe_stream, 'buffer'):
                            try:
                                buffer = safe_stream.buffer
                                if buffer is None:
                                    raise ValueError("Buffer is None")
                            except (ValueError, AttributeError):
                                # buffer가 분리된 경우, 직접 write 시도
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
                        # 버퍼 분리 오류인 경우 무시하고 계속 진행
                        if "detached" in str(e).lower() or "raw stream" in str(e).lower():
                            pass
                        else:
                            pass
                
                # Fallback: stderr 사용
                try:
                    if sys.stderr and hasattr(sys.stderr, 'write'):
                        # stderr도 버퍼 분리 오류가 있을 수 있으므로 안전하게 처리
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
                # 모든 예외 무시 (로깅 실패가 전체 프로그램을 중단시키지 않도록)
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

logger = setup_logging(os.getenv("TEST_LOG_LEVEL", "INFO"))


def get_query_from_args() -> str:
    """명령줄 인자에서 질의 추출"""
    default_queries = [
        "계약서 작성 시 주의할 사항은 무엇인가요?",
        "민법 제750조 손해배상에 대해 설명해주세요",
        "임대차 계약 해지 시 주의사항은 무엇인가요?",
    ]
    
    # 1. 환경 변수
    test_query = os.getenv('TEST_QUERY')
    if test_query and test_query.strip():
        return test_query.strip()
    
    # 2. 명령줄 인자
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        
        # 파일 옵션
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
        
        # 숫자 선택
        if arg.isdigit():
            idx = int(arg)
            if 0 <= idx < len(default_queries):
                return default_queries[idx]
        
        # 직접 질의
        return " ".join(sys.argv[1:])
    
    # 기본 질의
    return default_queries[1]


async def run_query_test(query: str):
    """질의 테스트 실행"""
    logger.info("\n" + "="*80)
    logger.info("LangGraph 질의 테스트")
    logger.info("="*80)
    logger.info(f"\n📋 질의: {query}\n")
    
    try:
        # python-dotenv 경고 억제를 위한 환경 변수 설정
        import os
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
        
        # 설정 로드
        logger.info("1️⃣  설정 로드 중...")
        
        # MLflow 인덱스 사용 설정 (환경 변수 우선, 없으면 기본값)
        if not os.getenv('USE_MLFLOW_INDEX'):
            os.environ['USE_MLFLOW_INDEX'] = 'true'
            logger.info("   📌 USE_MLFLOW_INDEX=true 설정됨")
        
        if not os.getenv('MLFLOW_TRACKING_URI'):
            # MLflow tracking URI 설정
            mlflow_uri = str(project_root / "mlflow" / "mlruns")
            os.environ['MLFLOW_TRACKING_URI'] = f"file:///{mlflow_uri.replace(chr(92), '/')}"
            logger.info(f"   📌 MLFLOW_TRACKING_URI 설정됨")
        
        # MLFLOW_RUN_ID가 없으면 프로덕션 run 자동 조회 (비워두면 자동)
        if not os.getenv('MLFLOW_RUN_ID'):
            logger.info("   📌 MLFLOW_RUN_ID 비어있음 - 프로덕션 run 자동 조회 예정")
        else:
            logger.info(f"   📌 MLFLOW_RUN_ID={os.getenv('MLFLOW_RUN_ID')} 설정됨")
        
        config = LangGraphConfig.from_env()
        config.enable_checkpoint = False  # 테스트 모드
        logger.info(f"   ✅ LangGraph 활성화: {config.langgraph_enabled}")
        logger.info(f"   ✅ 체크포인트: {config.enable_checkpoint}")
        
        # MLflow 인덱스 설정 확인
        from lawfirm_langgraph.core.utils.config import Config
        config_obj = Config()
        if config_obj.use_mlflow_index:
            logger.info(f"   ✅ MLflow 인덱스 사용: run_id={config_obj.mlflow_run_id or '자동 조회'}")
        else:
            logger.info(f"   ℹ️  MLflow 인덱스 미사용 (DB 기반 인덱스 사용)")
        
        # 서비스 초기화
        logger.info("\n2️⃣  LangGraphWorkflowService 초기화 중...")
        service = LangGraphWorkflowService(config)
        logger.info("   ✅ 서비스 초기화 완료")
        
        # 질의 처리
        logger.info("\n3️⃣  질의 처리 중...")
        logger.info("   (이 작업은 몇 초에서 몇 분이 걸릴 수 있습니다)")
        
        result = await service.process_query(
            query=query,
            session_id="query_test",
            enable_checkpoint=False,
            use_astream_events=True
        )
        
        # 결과 출력
        logger.info("\n4️⃣  결과:")
        logger.info("="*80)
        
        # 답변
        answer = result.get("answer", "")
        if isinstance(answer, dict):
            answer = answer.get("content", answer.get("text", str(answer)))
        
        if answer:
            logger.info(f"\n📝 답변 ({len(str(answer))}자):")
            logger.info("-" * 80)
            logger.info(str(answer))
        else:
            logger.warning("<답변 없음>")
        
        # retrieved_docs (데이터베이스/벡터스토어에서 검색한 참고자료)
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            logger.info(f"\n🔍 검색된 참고자료 (retrieved_docs) ({len(retrieved_docs)}개):")
            
            # 타입별 분포 확인
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
                    
                    # 버전 정보 수집
                    version_id = doc.get("embedding_version_id") or doc.get("metadata", {}).get("embedding_version_id")
                    if version_id:
                        version_counts[version_id] = version_counts.get(version_id, 0) + 1
                    
                    # 유사도 점수 수집
                    score = doc.get("score") or doc.get("similarity") or doc.get("relevance_score")
                    if score is not None:
                        scores.append(float(score))
            
            logger.info(f"   타입 분포: {type_counts}")
            if statute_articles:
                logger.info(f"   statute_article 타입 문서: {len(statute_articles)}개")
            
            # 버전 분포 출력
            if version_counts:
                logger.info(f"   📊 Embedding 버전 분포: {version_counts}")
            else:
                logger.warning("   ⚠️  검색 결과에 embedding_version_id가 없습니다!")
            
            # 유사도 점수 분포 분석
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
                    
                    # statute_article 타입 문서의 경우 상세 정보 출력
                    if doc_type == "statute_article":
                        statute_name = doc.get("statute_name") or doc.get("law_name") or doc.get("metadata", {}).get("statute_name") or doc.get("metadata", {}).get("law_name")
                        article_no = doc.get("article_no") or doc.get("article_number") or doc.get("metadata", {}).get("article_no") or doc.get("metadata", {}).get("article_number")
                        clause_no = doc.get("clause_no") or doc.get("metadata", {}).get("clause_no")
                        item_no = doc.get("item_no") or doc.get("metadata", {}).get("item_no")
                        logger.info(f"      - statute_name: {statute_name}")
                        logger.info(f"      - article_no: {article_no}")
                        logger.info(f"      - clause_no: {clause_no}")
                        logger.info(f"      - item_no: {item_no}")
                    
                    # 상세 정보 (선택적)
                    if doc.get("score"):
                        logger.info(f"      - 점수: {doc.get('score'):.4f}")
                    
                    # 버전 정보 출력
                    version_id = doc.get("embedding_version_id") or doc.get("metadata", {}).get("embedding_version_id")
                    if version_id:
                        logger.info(f"      - embedding_version_id: {version_id}")
                    
                    if doc.get("metadata") and doc_type != "statute_article":
                        logger.info(f"      - 메타데이터: {doc.get('metadata')}")
                else:
                    logger.info(f"   {i}. {str(doc)[:100]}")
            if len(retrieved_docs) > 10:
                logger.info(f"   ... (총 {len(retrieved_docs)}개)")
        else:
            logger.warning("\n⚠️  검색된 참고자료 (retrieved_docs)가 없습니다!")
            logger.warning("   - 데이터베이스/벡터스토어에서 검색이 수행되지 않았거나")
            logger.warning("   - 검색 결과가 없을 수 있습니다.")
        
        # 소스 (retrieved_docs에서 변환된 sources)
        sources = result.get("sources", [])
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
        
        # 메타데이터
        metadata = result.get("metadata", {})
        if metadata:
            logger.info(f"\n📊 메타데이터:")
            for key, value in list(metadata.items())[:10]:
                if key == "related_questions":
                    logger.info(f"   {key}: {value} ({len(value) if isinstance(value, list) else 'N/A'}개)")
                else:
                    logger.info(f"   {key}: {value}")
        
        # 신뢰도
        confidence = result.get("confidence", 0.0)
        if confidence:
            logger.info(f"\n🎯 신뢰도: {confidence:.2f}")
        
        # 처리 시간
        processing_time = result.get("processing_time", 0.0)
        if processing_time:
            logger.info(f"\n⏱️  처리 시간: {processing_time:.2f}초")
        
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
                logger.warning(f"   ⚠️  일부 retrieved_docs가 sources로 변환되지 않았습니다.")
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
        
        # related_questions 디버깅
        if not related_questions:
            logger.warning(f"\n⚠️  related_questions가 없습니다!")
            logger.warning("   가능한 원인:")
            logger.warning("   1. phase_info에 suggested_questions가 없을 수 있습니다.")
            logger.warning("   2. conversation_flow_tracker가 초기화되지 않았을 수 있습니다.")
            logger.warning("   3. metadata에 저장되지 않았을 수 있습니다.")
            # phase_info 확인
            if "phase_info" in result:
                phase_info = result.get("phase_info", {})
                logger.info(f"\n   phase_info 확인:")
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
        
        logger.info("\n" + "="*80)
        logger.info("✅ 테스트 완료!")
        logger.info("="*80)
        
        return result
        
    except ImportError as e:
        logger.error(f"\n❌ Import 오류: {e}")
        logger.error("\n필요한 패키지가 설치되어 있는지 확인하세요.")
        logger.error(f"   프로젝트 루트: {project_root}")
        logger.error(f"   lawfirm_langgraph 디렉토리: {lawfirm_langgraph_dir}")
        import sys
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {type(e).__name__}: {e}", exc_info=True)
        import sys
        sys.exit(1)


def main():
    """메인 실행 함수"""
    try:
        # stderr 복원 (모듈 import 후) - 이미 리다이렉트하지 않으므로 불필요
        # global _original_stderr
        # try:
        #     if hasattr(sys.stderr, 'close'):
        #         sys.stderr.close()
        # except:
        #     pass
        # sys.stderr = _original_stderr
        
        query = get_query_from_args()
        
        if not query:
            logger.error("질의를 입력해주세요.")
            logger.info("\n사용법:")
            logger.info("  python run_query_test.py \"질의 내용\"")
            logger.info("  python run_query_test.py 0  # 기본 질의 선택")
            logger.info("  $env:TEST_QUERY='질의내용'; python run_query_test.py")
            return 1
        
        result = asyncio.run(run_query_test(query))
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"\n\n❌ 테스트 실패: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

