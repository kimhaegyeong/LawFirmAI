# 프로젝트 구조 개편 마이그레이션 가이드

**날짜**: 2025-10-16  
**목적**: LawFirmAI 프로젝트 구조 개편 및 정리

## 개편 배경

기존 프로젝트 구조에서 다음과 같은 문제점들이 발견되어 전면적인 구조 개편을 진행했습니다:

1. **문서와 실제 구조 불일치**: 문서에 명시된 구조와 실제 파일 구조가 상당히 다름
2. **스크립트 디렉토리 과도한 세분화**: 12개의 하위 디렉토리로 인한 관리 복잡성
3. **데이터 파일 중복**: `gradio/data/lawfirm.db`와 메인 데이터베이스 중복
4. **임시 파일 혼재**: PID 파일과 리포트 파일이 프로젝트 루트에 위치

## 주요 변경사항

### 1. 스크립트 디렉토리 통합

#### 기존 구조 → 새로운 구조

**데이터 수집 스크립트 통합**:
```
scripts/assembly/ → scripts/data_collection/assembly/
scripts/collection/ → scripts/data_collection/qa_generation/
scripts/precedent/ → scripts/data_collection/precedent/
scripts/constitutional_decision/ → scripts/data_collection/constitutional/
scripts/legal_interpretation/ → scripts/data_collection/legal_interpretation/
scripts/administrative_appeal/ → scripts/data_collection/administrative_appeal/
scripts/legal_term/ → scripts/data_collection/legal_term/
```

**공통 유틸리티 통합**:
```
scripts/assembly/assembly_collector.py → scripts/data_collection/common/
scripts/assembly/assembly_logger.py → scripts/data_collection/common/
scripts/assembly/checkpoint_manager.py → scripts/data_collection/common/
scripts/assembly/common_utils.py → scripts/data_collection/common/
```

**데이터 처리 스크립트 통합**:
```
scripts/assembly/preprocess*.py → scripts/data_processing/preprocessing/
scripts/assembly/enhanced*.py → scripts/data_processing/preprocessing/
scripts/assembly/fast_preprocess*.py → scripts/data_processing/preprocessing/
scripts/assembly/parsers/ → scripts/data_processing/parsers/
scripts/assembly/validate*.py → scripts/data_processing/validation/
scripts/assembly/verify*.py → scripts/data_processing/validation/
scripts/assembly/check*.py → scripts/data_processing/validation/
```

**ML 및 벡터 임베딩 통합**:
```
scripts/assembly/ml_*.py → scripts/ml_training/model_training/
scripts/assembly/train_ml_model.py → scripts/ml_training/model_training/
scripts/assembly/prepare_training_data.py → scripts/ml_training/training_data/
scripts/assembly/optimized_prepare_training_data.py → scripts/ml_training/training_data/
scripts/model_training/* → scripts/ml_training/model_training/
scripts/vector_embedding/* → scripts/ml_training/vector_embedding/
```

### 2. 런타임 파일 정리

```
gradio_server.pid → runtime/gradio_server.pid
law_parsing_quality_report.txt → reports/law_parsing_quality_report.txt
```

### 3. 데이터 중복 제거

```
gradio/data/lawfirm.db (삭제) - 메인 data/lawfirm.db 사용
```

### 4. .gitignore 업데이트

다음 항목들이 추가되었습니다:
```
# Runtime files
runtime/
*.pid
```

## 새로운 디렉토리 구조

```
scripts/
├── data_collection/             # 데이터 수집
│   ├── assembly/                # Assembly 수집
│   ├── precedent/               # 판례 수집
│   ├── constitutional/          # 헌재결정례 수집
│   ├── legal_interpretation/    # 법령해석례 수집
│   ├── administrative_appeal/   # 행정심판례 수집
│   ├── legal_term/              # 법률용어 수집
│   ├── qa_generation/           # QA 데이터 생성
│   └── common/                  # 공통 유틸리티
├── data_processing/             # 데이터 전처리
│   ├── parsers/                 # 법률 문서 파서
│   ├── preprocessing/           # 전처리 파이프라인
│   ├── validation/              # 데이터 검증
│   └── utilities/               # 처리 유틸리티
├── ml_training/                 # ML 및 벡터 임베딩
│   ├── model_training/          # 모델 훈련
│   ├── vector_embedding/        # 벡터 임베딩 생성
│   └── training_data/           # 훈련 데이터 준비
├── analysis/                    # 데이터 분석 (기존 유지)
├── benchmarking/                # 성능 벤치마킹 (기존 유지)
├── database/                    # 데이터베이스 관리 (기존 유지)
├── monitoring/                  # 모니터링 스크립트 (기존 유지)
└── tests/                       # 테스트 스크립트 (기존 유지)
```

## 마이그레이션 후 작업 필요사항

### 1. Import 경로 업데이트

기존 스크립트에서 다른 스크립트를 import하는 경우, 경로를 업데이트해야 합니다:

**기존**:
```python
from scripts.assembly.assembly_collector import AssemblyCollector
from scripts.collection.generate_qa_dataset import generate_qa
```

**새로운**:
```python
from scripts.data_collection.common.assembly_collector import AssemblyCollector
from scripts.data_collection.qa_generation.generate_qa_dataset import generate_qa
```

### 2. 배치 파일 및 스크립트 업데이트

다음 파일들에서 경로 참조를 업데이트해야 합니다:
- `monitoring/` 디렉토리의 스크립트들
- 기타 시작 스크립트들

### 3. 문서 업데이트

다음 문서들이 업데이트되었습니다:
- `docs/01_project_overview/project_overview.md` - 프로젝트 구조 섹션
- `.gitignore` - 런타임 파일 제외 규칙 추가

## 롤백 가이드

만약 문제가 발생하여 롤백이 필요한 경우:

1. **Git을 사용하는 경우**: 이전 커밋으로 되돌리기
2. **수동 롤백**: 이 문서의 "기존 구조 → 새로운 구조" 매핑을 역순으로 적용

## 검증 체크리스트

마이그레이션 완료 후 다음 사항들을 확인하세요:

- [ ] 모든 스크립트가 새로운 위치에서 정상 실행되는지 확인
- [ ] Gradio 애플리케이션이 정상적으로 시작되는지 확인
- [ ] 데이터베이스 연결이 정상적으로 작동하는지 확인
- [ ] Import 에러가 없는지 확인
- [ ] 문서의 구조 설명이 실제 구조와 일치하는지 확인

## 향후 개선사항

1. **자동화 스크립트**: 구조 정리를 위한 자동화 스크립트 개발
2. **CI/CD 통합**: 구조 변경 시 자동 검증 파이프라인 구축
3. **문서 동기화**: 실제 구조와 문서 간 자동 동기화 시스템

---

*이 문서는 2025-10-16 프로젝트 구조 개편 작업의 완전한 기록입니다.*
