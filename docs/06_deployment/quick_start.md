# 빠른 시작 가이드

LawFirmAI를 빠르게 시작하는 방법에 대한 가이드입니다.

## 1. 저장소 클론

```bash
git clone https://github.com/your-username/LawFirmAI.git
cd LawFirmAI
```

## 2. 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate
```

## 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
# Google AI API 키 설정 (필수)
GOOGLE_API_KEY="your_google_key"

# LangGraph 설정 (lawfirm_langgraph/.env)
GOOGLE_API_KEY="your_google_key"
OLLAMA_BASE_URL="http://localhost:11434"

# API 서버 설정 (api/.env)
API_HOST="0.0.0.0"
API_PORT=8000
DEBUG=false

# React 프론트엔드 설정 (frontend/.env)
VITE_API_BASE_URL="http://localhost:8000"
```

자세한 설정은 각 디렉토리의 README를 참조하세요.

## 4. 데이터 수집

```bash
# 전체 데이터 수집 및 벡터DB 구축
python scripts/data_processing/run_data_pipeline.py --mode full --oc your_email_id

# 특정 데이터 타입만 수집
python scripts/data_processing/run_data_pipeline.py --mode laws --oc your_email_id --query "민법"
python scripts/data_processing/run_data_pipeline.py --mode precedents --oc your_email_id --query "계약 해지"

# 벡터DB 구축만 실행
python scripts/data_processing/run_data_pipeline.py --mode build
```

## 5. 애플리케이션 실행

### React 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

### FastAPI 서버 실행

```bash
# FastAPI 서버 실행
cd api
pip install -r requirements.txt
python main.py
```

또는:

```bash
# 프로젝트 루트에서 실행
cd api
pip install -r requirements.txt
python -m api.main
```

## 6. 접속

- **React 프론트엔드**: http://localhost:3000
- **FastAPI 서버**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

## Docker 사용

### React 프론트엔드 실행

```bash
cd frontend
docker-compose up -d
```

### FastAPI 서버 실행

```bash
cd api
docker-compose up -d
```

## 개발 환경 설정

```bash
# API 서버 의존성 설치
cd api
pip install -r requirements.txt

# React 프론트엔드 의존성 설치
cd ../frontend
npm install

# 코드 포맷팅
black api/ lawfirm_langgraph/
isort api/ lawfirm_langgraph/

# 린팅
flake8 api/ lawfirm_langgraph/
mypy api/ lawfirm_langgraph/

# 테스트 실행
pytest tests/
```

## 코드 스타일

- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **문서화**: 모든 클래스와 함수에 docstring 작성
- **테스트**: 핵심 기능에 대한 단위 테스트 작성

