# ⚖️ LawFirmAI - 법률 AI 어시스턴트

법률 관련 질문에 답변해드리는 AI 어시스턴트입니다. 판례, 법령, Q&A 데이터베이스를 기반으로 정확한 법률 정보를 제공합니다.

## 🚀 주요 기능

- **판례 검색**: 법원 판례 검색 및 분석
- **법령 해설**: 법령 조문 해석 및 설명  
- **계약서 분석**: 계약서 검토 및 위험 요소 분석
- **Q&A**: 자주 묻는 법률 질문 답변
- **RAG 기반 답변**: 검색 증강 생성으로 정확한 답변 제공

## 🛠️ 기술 스택

### AI/ML
- **KoGPT-2**: 한국어 생성 모델
- **Sentence-BERT**: 텍스트 임베딩 모델
- **ChromaDB**: 벡터 데이터베이스

### Backend
- **FastAPI**: RESTful API 서버
- **SQLite**: 관계형 데이터베이스
- **Pydantic**: 데이터 검증

### Frontend
- **Gradio**: 웹 인터페이스
- **HuggingFace Spaces**: 배포 플랫폼

## 📁 프로젝트 구조

```
LawFirmAI/
├── app.py                    # Gradio 메인 애플리케이션
├── main.py                   # FastAPI 메인 애플리케이션
├── requirements.txt          # Python 의존성
├── Dockerfile               # Docker 컨테이너 설정
├── docker-compose.yml       # 로컬 개발 환경
├── env.example              # 환경 변수 템플릿
├── .gitignore               # Git 무시 파일
├── README.md                # 프로젝트 문서
├── source/                  # Core Modules
│   ├── models/              # AI 모델 관련
│   ├── services/            # 비즈니스 로직
│   ├── data/                # 데이터 처리
│   ├── api/                 # API 관련
│   └── utils/               # 유틸리티
├── data/                    # 데이터 파일
├── tests/                   # 테스트 코드
├── docs/                    # 문서
└── scripts/                 # 유틸리티 스크립트
```

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/LawFirmAI.git
cd LawFirmAI
```

### 2. 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

```bash
# 환경 변수 파일 복사
copy env.example .env

# .env 파일 편집하여 설정값 수정
```

### 5. 애플리케이션 실행

```bash
# Gradio 인터페이스 실행
python app.py

# 또는 FastAPI 서버 실행
python main.py
```

### 6. 접속

- **Gradio 인터페이스**: http://localhost:7860
- **FastAPI 서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## 🐳 Docker 사용

### Docker Compose로 실행

```bash
# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

### Docker로 직접 실행

```bash
# 이미지 빌드
docker build -t lawfirm-ai .

# 컨테이너 실행
docker run -p 7860:7860 -p 8000:8000 lawfirm-ai
```

## 📊 벤치마킹 결과

### AI 모델 성능 비교

| 지표 | KoBART | KoGPT-2 | 승자 |
|------|--------|---------|------|
| 모델 크기 | 472.5 MB | 477.5 MB | KoBART |
| 메모리 사용량 | 400.8 MB | 748.3 MB | KoBART |
| 추론 속도 | 13.18초 | 8.34초 | **KoGPT-2** |
| 응답 품질 | 낮음 | 보통 | **KoGPT-2** |

### 벡터 스토어 성능 비교

| 지표 | FAISS | ChromaDB | 승자 |
|------|-------|----------|------|
| 안정성 | 오류 발생 | 정상 동작 | **ChromaDB** |
| 검색 속도 | 측정 불가 | 0.17초 | **ChromaDB** |
| QPS | 측정 불가 | 5.82 | **ChromaDB** |

## 🔧 개발

### 개발 환경 설정

```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 코드 포맷팅
black source/
isort source/

# 린팅
flake8 source/
mypy source/

# 테스트 실행
pytest tests/
```

### 코드 스타일

- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **문서화**: 모든 클래스와 함수에 docstring 작성
- **테스트**: 핵심 기능에 대한 단위 테스트 작성

## 📚 API 문서

### 주요 엔드포인트

- `POST /api/v1/chat` - 채팅 메시지 처리
- `GET /api/v1/health` - 헬스체크
- `GET /docs` - API 문서 (Swagger UI)

### 사용 예제

```python
import requests

# 채팅 요청
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "message": "계약서에서 주의해야 할 조항은 무엇인가요?",
        "context": "부동산 매매계약"
    }
)

result = response.json()
print(result["response"])
```

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
