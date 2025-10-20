# LawFirmAI - Gradio Interface

법률 AI 어시스턴트의 Gradio 웹 인터페이스입니다.

## 기능

- 법률 관련 질문 답변
- 판례 검색 및 분석
- 법령 해설
- 계약서 분석
- Q&A 서비스

## 실행 방법

### 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python app.py
```

### Docker를 사용한 실행

```bash
# Docker 이미지 빌드
docker build -t lawfirm-gradio .

# 컨테이너 실행
docker run -p 7860:7860 lawfirm-gradio
```

### Docker Compose를 사용한 실행

```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down
```

## 접속

애플리케이션 실행 후 http://localhost:7860 에서 접속할 수 있습니다.

## 환경 변수

`.env.example` 파일을 참고하여 환경 변수를 설정하세요.

## 로그 확인

### 실시간 로그 모니터링
```bash
# Windows PowerShell
Get-Content logs\gradio_app.log -Wait -Tail 50

# Windows CMD
type logs\gradio_app.log

# Linux/Mac
tail -f logs/gradio_app.log
```

### 로그 레벨 설정
```bash
# DEBUG 레벨로 실행 (더 자세한 로그)
# Windows
set LOG_LEVEL=DEBUG
python app.py

# PowerShell
$env:LOG_LEVEL="DEBUG"
python app.py

# Linux/Mac
export LOG_LEVEL=DEBUG
python app.py
```

### 로그 파일 위치
- **메인 로그**: `logs/gradio_app.log`
- **콘솔 출력**: 실시간 로그 확인

## 개발

개발 환경에서는 `DEBUG=true`로 설정하여 디버그 모드를 활성화할 수 있습니다.

### 디버깅 팁
- 로그 레벨을 DEBUG로 설정하여 상세한 디버깅 정보 확인
- 특정 모듈의 로그만 필터링: `grep "ChatService" logs/gradio_app.log`
- 에러 로그만 확인: `grep "ERROR" logs/gradio_app.log`
