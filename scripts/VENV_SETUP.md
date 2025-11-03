# Scripts Directory 가상환경 설정 가이드

`scripts` 디렉토리는 데이터 수집, 처리, ML 훈련 등을 위한 별도 가상환경을 사용합니다.

## 가상환경 생성 및 활성화

### Windows (PowerShell)

```powershell
# scripts 디렉토리로 이동
cd scripts

# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt

# Playwright 브라우저 설치 (데이터 수집용)
playwright install
```

### Windows (CMD)

```cmd
# scripts 디렉토리로 이동
cd scripts

# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate.bat

# 의존성 설치
pip install -r requirements.txt

# Playwright 브라우저 설치
playwright install
```

### Linux/macOS

```bash
# scripts 디렉토리로 이동
cd scripts

# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# Playwright 브라우저 설치
playwright install
```

## 가상환경 비활성화

### Windows (PowerShell)

```powershell
deactivate
```

### Windows (CMD)

```cmd
deactivate
```

### Linux/macOS

```bash
deactivate
```

## 주요 의존성

- **PyTorch**: ML 모델 훈련 및 평가
- **Transformers**: 사전 훈련된 모델 사용
- **Playwright**: 웹 스크래핑 및 자동화
- **FAISS**: 벡터 검색 및 임베딩
- **Sentence Transformers**: 한국어 임베딩 생성
- **Pandas/NumPy**: 데이터 처리

## 스크립트 실행 예시

가상환경이 활성화된 상태에서:

```powershell
# 데이터 수집
python data_collection/assembly/collect_laws.py --sample 100

# 데이터 전처리
python data_processing/preprocessing/preprocess_raw_data.py

# 벡터 임베딩 생성
python ml_training/vector_embedding/build_ml_enhanced_vector_db.py

# 모델 평가
python ml_training/model_training/evaluate_legal_model.py

# 데이터베이스 마이그레이션
python database/migrate_database_schema.py
```

## 문제 해결

### PowerShell 실행 정책 오류

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CUDA 지원 PyTorch 설치 (GPU 사용 시)

```powershell
# CPU 버전은 기본 requirements.txt에 포함됨
# GPU 버전이 필요한 경우:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Playwright 설치 문제

```powershell
# Playwright 재설치
playwright install chromium
playwright install firefox
playwright install webkit
```
