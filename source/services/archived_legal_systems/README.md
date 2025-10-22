# Archived Legal Restriction Systems

이 디렉토리는 중복된 법률 제한 시스템들을 아카이브한 곳입니다.

## 아카이브된 시스템들

### 1. legal_restriction_system.py
- **버전**: 기본 버전
- **상태**: 아카이브됨
- **이유**: improved_legal_restriction_system.py로 대체됨
- **기능**: 기본적인 법률 제한 규칙

### 2. multi_stage_validation_system.py
- **버전**: 다단계 검증 시스템
- **상태**: 백업용으로 유지
- **이유**: ML 통합 시스템의 백업으로 사용
- **기능**: 5단계 검증 (키워드 → 패턴 → 맥락 → 의도 → 최종 결정)

### 3. improved_multi_stage_validation_system.py
- **버전**: 개선된 다단계 검증 시스템
- **상태**: ML 통합 시스템에 통합됨
- **이유**: ml_integrated_validation_system.py에 통합됨
- **기능**: Edge Cases 특별 처리, Gemini 모더레이션 통합

## 현재 활성 시스템

### ml_integrated_validation_system.py
- **버전**: 최신 ML 통합 버전
- **상태**: 활성 사용 중
- **기능**: 
  - ML 패턴 학습과 자동 튜닝
  - 기존 시스템과 ML 예측의 조합
  - BERT 분류기, Simple 분류기 통합
  - Boundary Referee, LLM Referee 지원
  - 지속적 학습 및 피드백 수집

## 마이그레이션 가이드

기존 시스템에서 ML 통합 시스템으로 마이그레이션:

```python
# 기존 방식
from .improved_legal_restriction_system import ImprovedLegalRestrictionSystem
from .multi_stage_validation_system import MultiStageValidationSystem

# 새로운 방식
from .ml_integrated_validation_system import MLIntegratedValidationSystem

# 사용법
ml_system = MLIntegratedValidationSystem()
result = ml_system.validate(query="민법 제750조에 대해서 설명해줘")
```

## 호환성

ML 통합 시스템은 기존 시스템들과 호환되며, 백업 시스템으로 기존 시스템들을 유지합니다.
