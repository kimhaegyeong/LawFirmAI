# LawFirmAI API Server

FastAPI 기반 법률 AI 어시스턴트 백엔드 서버

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
cp .env.example .env
```

### 3. 서버 실행

```bash
python -m api.main
```

또는 Windows 배치 파일 사용:

```bash
start_server.bat
```

## 프로젝트 구조

```
api/
├── main.py              # FastAPI 메인 애플리케이션
├── config.py            # 설정 파일
├── routers/            # API 라우터
├── schemas/             # Pydantic 스키마
├── services/            # 비즈니스 로직
├── middleware/          # 미들웨어
├── database/            # 데이터베이스 모델
├── utils/               # 유틸리티
└── test/                # 테스트 코드
```

## 관련 문서

상세한 문서는 `docs/07_api/` 폴더를 참조하세요:

- [API 서버 가이드](../docs/07_api/README.md) - API 서버 상세 가이드
- [시작 가이드](../docs/07_api/START_GUIDE.md) - API 서버 시작 가이드
- [보안 감사 가이드](../docs/07_api/SECURITY_AUDIT.md) - 보안 감사 가이드
- [보안 점검 계획서](../docs/07_api/SECURITY_CHECKLIST.md) - 보안 점검 계획서
