# -*- coding: utf-8 -*-
"""
LawFirmAI - 프로덕션 실행 스크립트
개선된 UX/UI를 가진 최종 프로덕션 애플리케이션 실행
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

def setup_environment():
    """환경 설정"""
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/production_startup.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Production environment setup completed")
    
    return logger

def check_dependencies():
    """의존성 확인"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        "app_final_production.py",
        "static/production.css",
        "static/manifest.json",
        "components/production_ux.py",
        "components/advanced_features.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("All required files are present")
    return True

def create_startup_script():
    """시작 스크립트 생성"""
    script_content = '''#!/bin/bash
# LawFirmAI Production Startup Script

echo "Starting LawFirmAI Production Application..."

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="7860"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

# 로그 디렉토리 생성
mkdir -p logs

# 프로덕션 앱 실행
echo "Launching production interface..."
python gradio/app_final_production.py

echo "LawFirmAI Production Application started successfully!"
echo "Access the application at: http://localhost:7860"
'''
    
    with open("start_production.sh", "w") as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod("start_production.sh", 0o755)
    
    logger = logging.getLogger(__name__)
    logger.info("Production startup script created")

def create_windows_startup_script():
    """Windows 시작 스크립트 생성"""
    script_content = '''@echo off
REM LawFirmAI Production Startup Script for Windows

echo Starting LawFirmAI Production Application...

REM 환경 변수 설정
set PYTHONPATH=%PYTHONPATH%;%CD%
set GRADIO_SERVER_NAME=0.0.0.0
set GRADIO_SERVER_PORT=7860
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM 로그 디렉토리 생성
if not exist logs mkdir logs

REM 프로덕션 앱 실행
echo Launching production interface...
python gradio/app_final_production.py

echo LawFirmAI Production Application started successfully!
echo Access the application at: http://localhost:7860
pause
'''
    
    with open("start_production.bat", "w") as f:
        f.write(script_content)
    
    logger = logging.getLogger(__name__)
    logger.info("Windows production startup script created")

def create_docker_compose():
    """Docker Compose 파일 생성"""
    docker_compose_content = '''version: '3.8'

services:
  lawfirm-ai-production:
    build:
      context: .
      dockerfile: gradio/Dockerfile.production
    ports:
      - "7860:7860"
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
      - HF_HUB_DISABLE_SYMLINKS_WARNING=1
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
    
    with open("docker-compose.production.yml", "w") as f:
        f.write(docker_compose_content)
    
    logger = logging.getLogger(__name__)
    logger.info("Docker Compose file created")

def create_production_dockerfile():
    """프로덕션용 Dockerfile 생성"""
    dockerfile_content = '''# LawFirmAI Production Dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로덕션용 Gradio 의존성 설치
COPY gradio/requirements.txt gradio/
RUN pip install --no-cache-dir -r gradio/requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p logs

# 포트 노출
EXPOSE 7860

# 헬스체크 엔드포인트 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:7860/health || exit 1

# 프로덕션 앱 실행
CMD ["python", "gradio/app_final_production.py"]
'''
    
    dockerfile_path = Path("gradio/Dockerfile.production")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    
    logger = logging.getLogger(__name__)
    logger.info("Production Dockerfile created")

def create_health_check_endpoint():
    """헬스체크 엔드포인트 추가"""
    health_check_content = '''# -*- coding: utf-8 -*-
"""
LawFirmAI - 헬스체크 엔드포인트
프로덕션 환경을 위한 상태 확인 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil
import time
from datetime import datetime

app = FastAPI(title="LawFirmAI Health Check")

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # 시스템 상태 확인
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 메모리 사용량이 90% 이상이면 경고
        if memory.percent > 90:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # 디스크 사용량이 90% 이상이면 경고
        if disk.percent > 90:
            raise HTTPException(status_code=503, detail="High disk usage")
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "service": "LawFirmAI Production"
        })
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/ready")
async def readiness_check():
    """레디니스 체크 엔드포인트"""
    try:
        # 애플리케이션 준비 상태 확인
        # 여기에 실제 애플리케이션 상태 확인 로직 추가
        
        return JSONResponse({
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "service": "LawFirmAI Production"
        })
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''
    
    with open("gradio/health_check.py", "w") as f:
        f.write(health_check_content)
    
    logger = logging.getLogger(__name__)
    logger.info("Health check endpoint created")

def create_production_readme():
    """프로덕션 README 생성"""
    readme_content = '''# LawFirmAI Production Application

## 🚀 프로덕션 배포용 LawFirmAI 애플리케이션

### ✨ 주요 개선사항

#### 🎨 UX/UI 개선
- **사용자 친화적 인터페이스**: 복잡한 탭 구조를 단순화하여 직관적인 사용자 경험 제공
- **모바일 최적화**: 반응형 디자인으로 모든 디바이스에서 최적화된 경험
- **사용자 온보딩**: 신규 사용자를 위한 단계별 가이드 및 튜토리얼
- **지능형 질문 제안**: 사용자 유형과 대화 맥락에 따른 맞춤형 질문 제안

#### 🔧 기술적 개선
- **에러 처리 개선**: 사용자 친화적인 에러 메시지 및 복구 방안 제시
- **문서 분석 UI**: 드래그 앤 드롭, 진행률 표시, 결과 시각화
- **피드백 시스템**: 사용자 만족도 수집 및 개선사항 반영
- **성능 최적화**: 로딩 상태 개선 및 메모리 사용량 최적화

### 🛠️ 설치 및 실행

#### 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt
pip install -r gradio/requirements.txt

# 프로덕션 앱 실행
python gradio/app_final_production.py
```

#### Docker 실행
```bash
# Docker Compose로 실행
docker-compose -f docker-compose.production.yml up -d

# 또는 Docker로 직접 실행
docker build -f gradio/Dockerfile.production -t lawfirm-ai-production .
docker run -p 7860:7860 lawfirm-ai-production
```

#### 시작 스크립트 사용
```bash
# Linux/Mac
./start_production.sh

# Windows
start_production.bat
```

### 📱 주요 기능

#### 💬 스마트 채팅
- 사용자 유형별 맞춤형 답변
- 대화 맥락 유지 및 연속 상담
- 지능형 질문 제안 시스템

#### 📄 문서 분석
- 계약서, 법률 문서, 판례 분석
- 위험 요소 탐지 및 개선 제안
- 실시간 분석 진행률 표시

#### 🎯 개인화
- 사용자 프로필 기반 맞춤 서비스
- 관심 분야별 우선 정보 제공
- 전문성 수준에 따른 답변 조정

#### 📊 피드백 시스템
- 답변 만족도 평가
- 개선사항 제안 수집
- 지속적인 서비스 품질 향상

### 🔍 모니터링

#### 헬스체크
- **상태 확인**: `http://localhost:7860/health`
- **준비 상태**: `http://localhost:7860/ready`

#### 로그 확인
```bash
# 애플리케이션 로그
tail -f logs/production_app.log

# 시작 로그
tail -f logs/production_startup.log
```

### 🚀 배포 가이드

#### HuggingFace Spaces 배포
1. `gradio/app_final_production.py`를 메인 앱으로 설정
2. `gradio/requirements.txt`에 의존성 추가
3. `gradio/static/` 폴더에 정적 파일 포함
4. Spaces에서 자동 배포

#### 클라우드 배포
1. Docker 이미지 빌드
2. 클라우드 플랫폼에 배포
3. 환경 변수 설정
4. 헬스체크 엔드포인트 설정

### 📈 성능 최적화

#### 메모리 관리
- 지연 로딩으로 초기 메모리 사용량 최적화
- 불필요한 컴포넌트 자동 정리
- 메모리 사용량 모니터링

#### 응답 속도
- 캐싱 시스템으로 반복 질의 최적화
- 비동기 처리로 동시 요청 처리
- 로딩 상태 표시로 사용자 경험 개선

### 🔒 보안 고려사항

#### 입력 검증
- 사용자 입력 검증 및 필터링
- 파일 업로드 보안 검사
- SQL 인젝션 방지

#### 데이터 보호
- 개인정보 암호화 저장
- 세션 데이터 보안 관리
- 로그 데이터 익명화

### 📞 지원 및 문의

#### 문제 해결
- 로그 파일 확인
- 헬스체크 엔드포인트 상태 확인
- 의존성 버전 확인

#### 피드백
- 애플리케이션 내 피드백 시스템 활용
- GitHub Issues를 통한 버그 리포트
- 기능 요청 및 개선 제안

### 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**LawFirmAI Production Team**  
법률 AI 어시스턴트 프로덕션 버전
'''
    
    with open("README.production.md", "w") as f:
        f.write(readme_content)
    
    logger = logging.getLogger(__name__)
    logger.info("Production README created")

def main():
    """메인 함수"""
    logger = setup_environment()
    
    logger.info("Setting up LawFirmAI Production Environment...")
    
    # 의존성 확인
    if not check_dependencies():
        logger.error("Dependency check failed. Please check missing files.")
        return False
    
    # 시작 스크립트 생성
    create_startup_script()
    create_windows_startup_script()
    
    # Docker 설정
    create_docker_compose()
    create_production_dockerfile()
    
    # 헬스체크 엔드포인트
    create_health_check_endpoint()
    
    # 문서화 (Windows 환경에서는 건너뛰기)
    try:
        create_production_readme()
    except UnicodeEncodeError:
        logger.warning("Skipping README creation due to Unicode encoding issues on Windows")
    
    logger.info("Production environment setup completed successfully!")
    logger.info("You can now run the production application using:")
    logger.info("   - python gradio/app_final_production.py")
    logger.info("   - ./start_production.sh (Linux/Mac)")
    logger.info("   - start_production.bat (Windows)")
    logger.info("   - docker-compose -f docker-compose.production.yml up -d")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("LawFirmAI Production setup completed successfully!")
    else:
        print("LawFirmAI Production setup failed!")
        sys.exit(1)
