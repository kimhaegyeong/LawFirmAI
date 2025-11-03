# LawFirmAI 가상환경 사용 가이드

이 프로젝트는 **별도의 가상환경**을 사용하는 두 개의 주요 디렉토리로 구성되어 있습니다.

## 📁 디렉토리별 가상환경

### 1. `lawfirm_langgraph/` - LangGraph 워크플로우
**용도**: LangGraph 기반 법률 AI 워크플로우 실행

**의존성**:
- LangGraph v1.0
- LangChain v1.0
- Google Gemini (LangChain Google GenAI)
- FAISS 벡터 검색
- Sentence Transformers

**설정 가이드**: [`lawfirm_langgraph/VENV_SETUP.md`](lawfirm_langgraph/VENV_SETUP.md)

### 2. `scripts/` - 데이터 수집 및 처리 스크립트
**용도**: 데이터 수집, 전처리, ML 훈련, 벡터 임베딩 생성

**의존성**:
- PyTorch & Transformers
- Playwright (웹 스크래핑)
- FAISS & Sentence Transformers
- Pandas & NumPy

**설정 가이드**: [`scripts/VENV_SETUP.md`](scripts/VENV_SETUP.md)

## 🚀 빠른 시작

### Windows (PowerShell)

#### LangGraph 가상환경
```powershell
# lawfirm_langgraph 디렉토리로 이동
cd lawfirm_langgraph

# 가상환경 활성화 (자동 생성 포함)
.\activate_venv.ps1

# 또는 수동으로
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Scripts 가상환경
```powershell
# scripts 디렉토리로 이동
cd scripts

# 가상환경 활성화 (자동 생성 포함)
.\activate_venv.ps1

# 또는 수동으로
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install  # Playwright 브라우저 설치
```

### Windows (CMD)

#### LangGraph 가상환경
```cmd
cd lawfirm_langgraph
activate_venv.bat
```

#### Scripts 가상환경
```cmd
cd scripts
activate_venv.bat
```

### Linux/macOS

#### LangGraph 가상환경
```bash
cd lawfirm_langgraph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Scripts 가상환경
```bash
cd scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

## 📝 사용 예시

### LangGraph 워크플로우 실행
```powershell
# lawfirm_langgraph 가상환경 활성화 상태에서
cd lawfirm_langgraph

# LangGraph 개발 서버 실행
langgraph dev

# Streamlit 앱 실행
streamlit run streamlit/app.py
```

### Scripts 실행
```powershell
# scripts 가상환경 활성화 상태에서
cd scripts

# 데이터 수집
python data_collection/assembly/collect_laws.py --sample 100

# 벡터 임베딩 생성
python ml_training/vector_embedding/build_ml_enhanced_vector_db.py

# 모델 평가
python ml_training/model_training/evaluate_legal_model.py
```

## ⚠️ 중요 사항

1. **각 디렉토리는 독립적인 가상환경 사용**: 서로 다른 의존성이 필요하므로 별도 가상환경이 필요합니다.

2. **가상환경 전환**: 한 작업을 마치고 다른 작업을 할 때는 가상환경을 비활성화한 후 해당 디렉토리의 가상환경을 활성화하세요.

3. **.gitignore**: 두 디렉토리의 `.venv` 폴더는 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

4. **Python 버전**: Python 3.9 이상을 권장합니다.

## 🔧 문제 해결

### PowerShell 실행 정책 오류
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 가상환경이 보이지 않는 경우
```powershell
# 숨겨진 파일 표시
Get-ChildItem -Force
```

### 패키지 설치 오류
```powershell
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 클리어 후 재설치
pip cache purge
pip install -r requirements.txt
```

## 📚 추가 문서

- [`lawfirm_langgraph/VENV_SETUP.md`](lawfirm_langgraph/VENV_SETUP.md) - LangGraph 가상환경 상세 가이드
- [`scripts/VENV_SETUP.md`](scripts/VENV_SETUP.md) - Scripts 가상환경 상세 가이드
