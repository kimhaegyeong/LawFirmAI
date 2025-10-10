# LawFirmAI 프로젝트 구조

## 개선된 프로젝트 구조

```
LawFirmAI/
├── app.py                    # Gradio 메인 애플리케이션
├── main.py                   # FastAPI 메인 애플리케이션
├── requirements.txt          # Python 의존성
├── requirements-dev.txt      # 개발용 의존성
├── Dockerfile               # Docker 컨테이너 설정
├── docker-compose.yml       # 로컬 개발 환경
├── .env.example             # 환경 변수 템플릿
├── .gitignore               # Git 무시 파일
├── README.md                # 프로젝트 문서
├── pyproject.toml           # 프로젝트 설정 (PEP 518)
├── source/                  # Core Modules (메인 소스 코드)
│   ├── __init__.py
│   ├── models/              # AI 모델 관련
│   │   ├── __init__.py
│   │   ├── kobart_model.py
│   │   ├── sentence_bert.py
│   │   └── model_manager.py
│   ├── services/            # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   ├── rag_service.py
│   │   ├── search_service.py
│   │   └── analysis_service.py
│   ├── data/                # 데이터 처리
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── vector_store.py
│   │   └── data_processor.py
│   ├── api/                 # API 관련
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   ├── middleware.py
│   │   └── schemas.py
│   └── utils/               # 유틸리티
│       ├── __init__.py
│       ├── config.py
│       └── logger.py
├── data/                    # 데이터 파일
│   ├── raw/                 # 원본 데이터
│   │   ├── constitutional_decisions/  # 헌재결정례 원본 데이터
│   │   │   ├── yearly_2025_*/         # 연도별 수집 데이터
│   │   │   └── checkpoint.json         # 수집 체크포인트
│   │   ├── precedents/                # 판례 원본 데이터
│   │   ├── laws/                      # 법령 원본 데이터
│   │   └── legal_terms/               # 법률 용어 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   ├── embeddings/          # 벡터 임베딩
│   └── models/              # 모델 파일
├── tests/                   # 테스트 코드
│   ├── __init__.py
│   ├── unit/                # 단위 테스트
│   ├── integration/         # 통합 테스트
│   └── fixtures/            # 테스트 데이터
├── scripts/                 # 유틸리티 스크립트
│   ├── setup.py            # 환경 설정
│   ├── benchmark_models.py
│   ├── benchmark_vector_stores.py
│   ├── constitutional_decision/  # 헌재결정례 수집 스크립트
│   │   ├── collect_by_date.py     # 날짜 기반 수집 메인 스크립트
│   │   ├── date_based_collector.py # 날짜 기반 수집 클래스
│   │   ├── constitutional_collector.py # 기존 수집 클래스
│   │   └── collect_constitutional_decisions.py # 기존 수집 스크립트
│   ├── precedent/           # 판례 수집 스크립트
│   ├── legal_term/         # 법률 용어 수집 스크립트
│   └── collect_all_data.py # 전체 데이터 수집 스크립트
├── docs/                    # 문서
│   ├── api/                 # API 문서
│   ├── architecture/        # 아키텍처 문서
│   ├── development/         # 개발 문서
│   ├── deployment/          # 배포 문서
│   └── user_guide/          # 사용자 가이드
├── logs/                    # 로그 파일
├── model_cache/             # 모델 캐시
└── benchmark_results/       # 벤치마크 결과
```

## 구조 개선 사항

### 1. 모듈화 개선
- **models/**: AI 모델 관련 코드를 별도 모듈로 분리
- **services/**: 비즈니스 로직을 서비스별로 분리
- **data/**: 데이터 처리 관련 코드 통합

### 2. 문서 구조 개선
- **api/**: API 관련 문서
- **architecture/**: 시스템 아키텍처 문서
- **development/**: 개발 관련 문서
- **deployment/**: 배포 관련 문서
- **user_guide/**: 사용자 가이드

### 3. 테스트 구조 개선
- **unit/**: 단위 테스트
- **integration/**: 통합 테스트
- **fixtures/**: 테스트 데이터

### 4. 설정 파일 추가
- **pyproject.toml**: 프로젝트 설정 (PEP 518)
- **logs/**: 로그 파일 디렉토리
- **model_cache/**: 모델 캐시 디렉토리

## 모듈 의존성

```
app.py (Gradio) ──┐
                  ├── source/services/chat_service.py
main.py (FastAPI) ─┘
                  ├── source/api/endpoints.py
                  ├── source/api/middleware.py
                  └── source/api/schemas.py

source/services/chat_service.py ──┐
                                  ├── source/models/model_manager.py
source/services/rag_service.py ───┤
source/services/search_service.py ─┤
source/services/analysis_service.py┘
                                  ├── source/data/database.py
                                  ├── source/data/vector_store.py
                                  └── source/data/data_processor.py

source/models/model_manager.py ────┐
                                  ├── source/models/kobart_model.py
                                  └── source/models/sentence_bert.py
```

## 파일 명명 규칙

### Python 파일
- **모듈**: `snake_case.py` (예: `chat_service.py`)
- **클래스**: `PascalCase` (예: `ChatService`)
- **함수/변수**: `snake_case` (예: `process_message`)
- **상수**: `UPPER_SNAKE_CASE` (예: `MAX_RETRY_COUNT`)

### 디렉토리
- **모듈 디렉토리**: `snake_case` (예: `data_processing`)
- **문서 디렉토리**: `snake_case` (예: `user_guide`)

## Import 규칙

```python
# 표준 라이브러리
import os
import sys
from typing import List, Dict, Optional

# 서드파티 라이브러리
import torch
import numpy as np
from fastapi import FastAPI
from transformers import AutoTokenizer

# 로컬 모듈
from source.models.kobart_model import KoBARTModel
from source.services.chat_service import ChatService
from source.utils.config import Config
```

## 환경 변수 관리

```python
# .env.example
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
DATABASE_URL=sqlite:///./data/lawfirm.db
MODEL_PATH=./models
DEVICE=cpu
LOG_LEVEL=INFO
```

## 테스트 구조

```
tests/
├── __init__.py
├── unit/
│   ├── test_models/
│   ├── test_services/
│   └── test_data/
├── integration/
│   ├── test_api/
│   └── test_rag/
└── fixtures/
    ├── sample_documents.json
    └── test_embeddings.pkl
```

이 구조는 확장성, 유지보수성, 그리고 개발 효율성을 고려하여 설계되었습니다.
