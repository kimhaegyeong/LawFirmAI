# LawFirmAI - FastAPI Backend

법률 AI 어시스턴트의 FastAPI 백엔드 서버입니다.

## 기능

- RESTful API 제공
- 법률 관련 질문 처리
- 판례 검색 API
- 법령 해설 API
- 계약서 분석 API
- Q&A API

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 실행 방법

### 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python main.py
```

### Docker를 사용한 실행

```bash
# Docker 이미지 빌드
docker build -t lawfirm-api .

# 컨테이너 실행
docker run -p 8000:8000 lawfirm-api
```

### Docker Compose를 사용한 실행

```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down
```

## 환경 변수

`.env.example` 파일을 참고하여 환경 변수를 설정하세요.

## 개발

개발 환경에서는 `DEBUG=true`로 설정하여 디버그 모드를 활성화할 수 있습니다.

## 헬스체크

서버 상태 확인: http://localhost:8000/health
