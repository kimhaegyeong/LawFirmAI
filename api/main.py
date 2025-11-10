"""
FastAPI 메인 애플리케이션
"""
import sys
import logging
import os
from pathlib import Path

# HuggingFace 로깅 비활성화 (가장 먼저 실행)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# HuggingFace 관련 로거 비활성화
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# 로그 레벨 환경 변수 읽기 (기본값: INFO)
log_level_str = os.getenv("LOG_LEVEL", "info").upper()
log_level_map = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
log_level = log_level_map.get(log_level_str, logging.INFO)

# 디버그: 로그 레벨 확인
print(f"[DEBUG] LOG_LEVEL environment variable: {os.getenv('LOG_LEVEL', 'not set')}")
print(f"[DEBUG] Parsed log level: {log_level_str} -> {log_level} ({logging.getLevelName(log_level)})")

# Windows multiprocessing과 호환되는 로깅 설정 (가장 먼저 실행)
if sys.platform == "win32":
    # Windows에서 multiprocessing 사용 시 로깅 에러 방지
    # force=True로 기존 핸들러를 제거하고 새로 설정
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    # 로깅 에러를 무시하도록 설정
    logging.raiseExceptions = False
else:
    # Windows가 아닌 경우에도 로깅 설정
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )

# 루트 로거 레벨 설정 (모든 로거에 적용)
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.disabled = False  # 명시적으로 활성화

# 모든 핸들러의 레벨도 설정
for handler in root_logger.handlers:
    handler.setLevel(log_level)

# 핸들러가 없으면 추가
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(handler)

# 주요 로거들의 레벨도 명시적으로 설정
logging.getLogger("api").setLevel(log_level)
logging.getLogger("api").disabled = False
logging.getLogger("api.services").setLevel(log_level)
logging.getLogger("api.services").disabled = False
logging.getLogger("api.services.chat_service").setLevel(log_level)
logging.getLogger("api.services.chat_service").disabled = False
logging.getLogger("lawfirm_langgraph").setLevel(log_level)
logging.getLogger("lawfirm_langgraph").disabled = False

# 로깅이 비활성화되지 않도록 보호
logging.disable(logging.NOTSET)  # 모든 로깅 활성화

# 강제로 stdout에 출력 (로깅이 작동하지 않을 경우를 대비)
import sys
sys.stdout.write(f"[DEBUG] Root logger level set to: {logging.getLevelName(root_logger.level)}\n")
sys.stdout.write(f"[DEBUG] Root logger disabled: {root_logger.disabled}\n")
sys.stdout.write(f"[DEBUG] All handlers configured with level: {logging.getLevelName(log_level)}\n")
sys.stdout.write(f"[DEBUG] Number of handlers: {len(root_logger.handlers)}\n")
sys.stdout.flush()

# 로깅 테스트 (모듈 레벨에서) - 로깅이 완전히 설정된 후에만 출력
# 모듈 import 시점에는 로깅이 완전히 설정되지 않을 수 있으므로
# startup 이벤트에서 테스트하도록 변경

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# lawfirm_langgraph 디렉토리를 sys.path에 추가 (core 모듈 import를 위해)
lawfirm_langgraph_path = project_root / "lawfirm_langgraph"
if lawfirm_langgraph_path.exists():
    sys.path.insert(0, str(lawfirm_langgraph_path))

# 환경 변수 로드 (중앙 집중식 로더 사용)
try:
    from utils.env_loader import load_all_env_files
    load_all_env_files(project_root)
except ImportError as e:
    print(f"⚠️  Failed to load environment variables: {e}")
    print("   Make sure utils/env_loader.py exists in the project root")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 환경 변수 로드 후에만 import (순서 중요!)
# routers를 import하면 chat_service가 초기화되므로, 환경 변수를 먼저 로드해야 함
from api.config import api_config
from api.middleware.logging import setup_logging

# 라우터는 환경 변수 로드 후에 import
from api.routers import chat, session, history, feedback, health, auth

# FastAPI 앱 생성
# 프로덕션 환경에서 API 문서 비활성화
docs_url = None if not api_config.debug else "/docs"
redoc_url = None if not api_config.debug else "/redoc"

app = FastAPI(
    title="LawFirmAI API",
    description="법률 AI 어시스턴트 API 서버",
    version="1.0.0",
    docs_url=docs_url,
    redoc_url=redoc_url
)

# CORS 설정
# 참고자료: https://fastapi.tiangolo.com/tutorial/cors/
cors_origins = api_config.get_cors_origins()

# 디버깅: 원본 값 확인
print(f"[CORS Debug] Raw cors_origins from config: {api_config.cors_origins} (type: {type(api_config.cors_origins)})", flush=True)
print(f"[CORS Debug] Parsed cors_origins: {cors_origins} (type: {type(cors_origins)})", flush=True)

# CORS origins가 올바른 리스트인지 확인
if not isinstance(cors_origins, list):
    print(f"[CORS Debug] WARNING: cors_origins is not a list! Using default.", flush=True)
    # 기본값으로 설정
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

# 빈 리스트 체크
if not cors_origins:
    print(f"[CORS Debug] WARNING: cors_origins is empty! Using default.", flush=True)
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

# 각 origin이 문자열인지 확인하고 정리
cors_origins = [str(origin).strip() for origin in cors_origins if origin and str(origin).strip()]

# 필수 origin 추가: http://localhost:3000는 항상 포함되어야 함
required_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
for origin in required_origins:
    if origin not in cors_origins:
        print(f"[CORS Debug] Adding required origin: {origin}", flush=True)
        cors_origins.append(origin)

# 개발 환경에서 추가 origin 자동 추가
if api_config.debug:
    # 개발 환경에서 자주 사용되는 origin 추가
    additional_origins = [
        "http://0.0.0.0:3000",
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:5173",
    ]
    for origin in additional_origins:
        if origin not in cors_origins:
            cors_origins.append(origin)

# 와일드카드 처리: allow_credentials=True일 때는 "*" 사용 불가
# 프로덕션 환경에서 와일드카드 사용 금지
allow_credentials = True
if "*" in cors_origins:
    if api_config.debug:
        # 개발 환경에서만 와일드카드 허용 (credentials는 False)
        allow_credentials = False
        logger.warning("개발 환경에서 CORS 와일드카드(*) 사용 중. allow_credentials가 False로 설정됩니다.")
    else:
        # 프로덕션에서는 와일드카드 제거
        logger.warning("프로덕션 환경에서 CORS 와일드카드(*) 사용은 보안상 위험합니다. 제거합니다.")
        cors_origins = [origin for origin in cors_origins if origin != "*"]
        if not cors_origins:
            cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
            logger.warning("CORS origins가 비어있어 기본값을 사용합니다.")

# 최종 확인 및 출력
print(f"[CORS Debug] Final cors_origins: {cors_origins}", flush=True)
print(f"[CORS Debug] allow_credentials: {allow_credentials}", flush=True)

# CORS 설정 로깅
import logging
logger = logging.getLogger(__name__)
if api_config.debug:
    logger.info(f"CORS 설정 완료: origins={cors_origins}, credentials={allow_credentials}")
else:
    logger.info(f"CORS 설정 완료: {len(cors_origins)} origins configured")

# FastAPI CORSMiddleware 추가 (가장 먼저 추가되어야 함)
# allow_credentials=True일 때는 allow_methods=["*"]를 사용할 수 없으므로 명시적으로 메서드 지정
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"],  # 모든 헤더 노출
    max_age=600,  # preflight 캐시 시간
)

# CORS 헤더를 명시적으로 추가하는 미들웨어 (백업)
# CORSMiddleware가 작동하지 않는 경우를 대비
# 미들웨어는 역순으로 실행되므로, 이 미들웨어는 CORSMiddleware 이후에 실행됨
@app.middleware("http")
async def add_cors_headers_middleware(request: Request, call_next):
    """CORS 헤더를 명시적으로 추가하는 미들웨어"""
    origin = request.headers.get("origin")
    
    # 일반 요청 처리
    response = await call_next(request)
    
    # CORS 헤더가 이미 있는지 확인
    has_cors_header = "Access-Control-Allow-Origin" in response.headers
    
    # CORS 헤더 추가 (없는 경우에만)
    if origin:
        if origin in cors_origins:
            # CORS 헤더가 없거나 다른 origin으로 설정된 경우 덮어쓰기
            if not has_cors_header or response.headers.get("Access-Control-Allow-Origin") != origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Expose-Headers"] = "*"
                print(f"[CORS Debug] Added CORS headers for origin: {origin}", flush=True)
        else:
            print(f"[CORS Debug] Origin {origin} not in allowed origins: {cors_origins}", flush=True)
    elif not has_cors_header:
        # origin이 없어도 CORS 헤더가 없으면 기본값 추가 (개발 환경)
        if api_config.debug:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "false"
            print(f"[CORS Debug] Added default CORS headers (debug mode)", flush=True)
    
    # OPTIONS 요청에 대한 추가 헤더 (CORSMiddleware가 처리했지만, 백업으로 추가)
    if request.method == "OPTIONS" and origin and origin in cors_origins:
        if "Access-Control-Allow-Methods" not in response.headers:
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD"
        if "Access-Control-Allow-Headers" not in response.headers:
            response.headers["Access-Control-Allow-Headers"] = "*"
        if "Access-Control-Max-Age" not in response.headers:
            response.headers["Access-Control-Max-Age"] = "600"
    
    return response

# 로깅 설정
setup_logging(app)

# 보안 헤더 미들웨어 추가
from api.middleware.security_headers import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)

# Rate Limiting 설정
from api.middleware.rate_limit import limiter, is_rate_limit_enabled, create_rate_limit_response
from slowapi.errors import RateLimitExceeded

if is_rate_limit_enabled():
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, create_rate_limit_response)
    logger.info("Rate Limiting이 활성화되었습니다.")
else:
    logger.info("Rate Limiting이 비활성화되었습니다.")

# CSRF 보호 설정
from api.middleware.csrf import setup_csrf_protection
setup_csrf_protection(app)

# FastAPI startup 이벤트에서 로깅 설정 강화
# uvicorn이 app을 import할 때 실행됨
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 로깅 설정 강화"""
    # HuggingFace 로깅 비활성화 (가장 먼저 실행)
    try:
        from lawfirm_langgraph.core.utils.safe_logging import disable_external_logging
        disable_external_logging()
    except ImportError:
        # fallback: 직접 비활성화
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
        logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # 로깅 설정을 다시 강제로 적용
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.disabled = False
    
    # 핸들러 확인 및 추가
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout 
        for h in root_logger.handlers
    )
    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(handler)
    
    # 모든 핸들러의 레벨 설정
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    # 로깅 보호
    logging.disable(logging.NOTSET)
    
    # 주요 로거들 활성화
    logging.getLogger("api").setLevel(log_level)
    logging.getLogger("api").disabled = False
    logging.getLogger("api.services").setLevel(log_level)
    logging.getLogger("api.services").disabled = False
    logging.getLogger("api.services.chat_service").setLevel(log_level)
    logging.getLogger("api.services.chat_service").disabled = False
    
    # lawfirm_langgraph 로거 레벨 설정 (환경 변수 LOG_LEVEL 반영)
    logging.getLogger("lawfirm_langgraph").setLevel(log_level)
    logging.getLogger("lawfirm_langgraph").disabled = False
    
    # lawfirm_langgraph 하위 로거들도 동일한 레벨로 설정
    for logger_name in ["lawfirm_langgraph.core", 
                        "lawfirm_langgraph.config", "lawfirm_langgraph.core.agents",
                        "lawfirm_langgraph.core.services", "lawfirm_langgraph.core.utils"]:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).disabled = False
    
    # 로깅 테스트
    test_logger = logging.getLogger("api.startup")
    test_logger.setLevel(log_level)
    test_logger.disabled = False
    test_logger.propagate = True
    
    print(f"[DEBUG] Startup event - Root logger level: {logging.getLevelName(root_logger.level)}")
    print(f"[DEBUG] Startup event - Root logger disabled: {root_logger.disabled}")
    print(f"[DEBUG] Startup event - Number of handlers: {len(root_logger.handlers)}")
    test_logger.info("✅ Startup event - Logging configured and enabled!")
    
    # ChatService 초기화하여 로그 확인
    try:
        from api.services.chat_service import get_chat_service
        test_logger.info("Initializing ChatService during startup to verify logging...")
        chat_service = get_chat_service()
        if chat_service.is_available():
            test_logger.info("✅ ChatService initialized successfully during startup")
        else:
            test_logger.warning("⚠️  ChatService initialized but workflow service is not available")
    except Exception as e:
        test_logger.error(f"Failed to initialize ChatService during startup: {e}", exc_info=True)

# 라우터 등록
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(session.router, prefix="/api/v1", tags=["session"])
app.include_router(history.router, prefix="/api/v1", tags=["history"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "LawFirmAI API",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    # 로깅 설정을 다시 강제로 적용 (uvicorn 실행 전)
    # uvicorn이 로깅 설정을 변경할 수 있으므로 다시 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.disabled = False
    
    # 핸들러 확인 및 추가
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout 
        for h in root_logger.handlers
    )
    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(handler)
    
    # 모든 핸들러의 레벨 설정
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    # 로깅 보호
    logging.disable(logging.NOTSET)
    
    print(f"[DEBUG] Before uvicorn.run - Root logger level: {logging.getLevelName(root_logger.level)}")
    print(f"[DEBUG] Before uvicorn.run - Root logger disabled: {root_logger.disabled}")
    print(f"[DEBUG] Before uvicorn.run - Number of handlers: {len(root_logger.handlers)}")
    
    # Windows에서 reload 사용 시 문제가 있을 수 있으므로, 조건부로 설정
    use_reload = api_config.debug
    
    # Windows 환경 감지 및 reload 설정 조정
    if sys.platform == "win32" and use_reload:
        # Windows에서 reload를 사용할 때는 reload-dir을 명시적으로 설정
        # api 디렉토리만 감시하여 multiprocessing 문제 최소화
        reload_dirs = [
            str(project_root / "api"),
        ]
        # lawfirm_langgraph는 제외 (너무 많은 파일 변경 감지 방지)
        reload_exclude = [
            "**/lawfirm_langgraph/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.git/**",
            "**/node_modules/**",
            "**/frontend/**",
            "**/data/**",
            "**/scripts/**",
        ]
        
        print(f"[INFO] Windows detected - Using reload with explicit directories")
        print(f"[INFO] Reload delay: 0.25s (to improve stability on Windows)")
    else:
        reload_dirs = None
        reload_exclude = None
    
    # 로그 레벨 환경 변수 읽기 (uvicorn용, 기본값: info)
    uvicorn_log_level = os.getenv("LOG_LEVEL", "info").lower()
    # uvicorn은 소문자만 지원
    valid_uvicorn_levels = ["critical", "error", "warning", "info", "debug", "trace"]
    if uvicorn_log_level not in valid_uvicorn_levels:
        uvicorn_log_level = "info"
    
    # 서버 시작 메시지
    print("\n" + "="*50)
    print("🚀 LawFirmAI API 서버 시작 중...")
    print(f"   Host: {api_config.api_host}")
    print(f"   Port: {api_config.api_port}")
    print(f"   Log Level: {uvicorn_log_level}")
    print(f"   Python Log Level: {logging.getLevelName(log_level)}")
    print(f"   CORS Origins: {cors_origins}")
    print(f"   Reload: {use_reload}")
    if sys.platform == "win32":
        print(f"   Platform: Windows")
    print("="*50 + "\n", flush=True)
    
    # uvicorn 실행 설정
    uvicorn_config = {
        "app": "api.main:app",
        "host": api_config.api_host,
        "port": api_config.api_port,
        "log_level": uvicorn_log_level,
        "reload": use_reload,
        # Python logging을 uvicorn이 변경하지 않도록 설정
        "use_colors": False,  # 색상 출력 비활성화 (로깅 간섭 방지)
    }
    
    # Windows에서 reload 사용 시 추가 옵션 설정
    if sys.platform == "win32" and use_reload and reload_dirs:
        uvicorn_config["reload_dirs"] = reload_dirs
        uvicorn_config["reload_excludes"] = reload_exclude
        # Windows에서 안정성을 위해 reload-delay 추가
        uvicorn_config["reload_delay"] = 0.25
    
    # uvicorn 실행 전에 로깅 테스트
    print(f"[DEBUG] Testing logging before uvicorn.run()...")
    print(f"[DEBUG] Root logger handlers: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        print(f"[DEBUG] Handler {i}: {type(handler).__name__}, level: {logging.getLevelName(handler.level)}")
    
    # 직접 로깅 테스트
    test_logger = logging.getLogger("api.test")
    test_logger.setLevel(log_level)
    test_logger.disabled = False
    test_logger.propagate = True
    
    # 직접 핸들러에 출력 테스트
    print(f"[DEBUG] Test logger level: {logging.getLevelName(test_logger.level)}")
    print(f"[DEBUG] Test logger disabled: {test_logger.disabled}")
    print(f"[DEBUG] Test logger handlers: {len(test_logger.handlers)}")
    print(f"[DEBUG] Test logger propagate: {test_logger.propagate}")
    
    # 로깅 테스트
    test_logger.info("✅ Test log before uvicorn.run() - This should be visible!")
    
    # 직접 핸들러를 통해 출력 테스트
    if root_logger.handlers:
        for handler in root_logger.handlers:
            try:
                handler.emit(logging.LogRecord(
                    name="api.test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg="✅ Direct handler test - This should be visible!",
                    args=(),
                    exc_info=None
                ))
            except Exception as e:
                print(f"[DEBUG] Handler emit failed: {e}")
    else:
        print("[DEBUG] No handlers found! Creating new handler...")
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(handler)
        test_logger.info("✅ Test log after adding handler - This should be visible!")
    
    uvicorn.run(**uvicorn_config)
