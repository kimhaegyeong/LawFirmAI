# 재임베딩 문서 인덱스

## 개요

재임베딩 관련 문서는 중복을 제거하고 유지보수 참고 자료로 활용할 수 있도록 정리되었습니다.

## 문서 구조

```
docs/02_data/embedding/
├── re_embedding_complete_report.md          # 최종 완료 보고서
├── re_embedding_optimization_guide.md       # 최적화 가이드
├── re_embedding_troubleshooting.md         # 문제 해결 가이드
└── skipped_documents_analysis.md            # 건너뛴 문서 분석

docs/reference/re_embedding/
└── README.md                                 # 참고 자료 인덱스
```

## 문서별 용도

### 1. 재임베딩 완료 보고서
**파일**: `re_embedding_complete_report.md`

**용도**:
- 재임베딩 작업의 전체 결과 확인
- 최종 처리 결과 및 성능 개선 효과 확인
- 다음 단계 작업 계획 수립

**대상**: 프로젝트 관리자, 개발자

### 2. 재임베딩 최적화 가이드
**파일**: `re_embedding_optimization_guide.md`

**용도**:
- 재임베딩 성능 최적화 방법 학습
- 시스템 사양에 맞는 파라미터 설정
- 성능 벤치마크 참고

**대상**: 성능 최적화 담당자, 개발자

### 3. 재임베딩 문제 해결 가이드
**파일**: `re_embedding_troubleshooting.md`

**용도**:
- 재임베딩 중 발생한 문제 해결 방법 참고
- 유사한 문제 발생 시 대응 가이드
- 예방 가이드 활용

**대상**: 유지보수 담당자, 개발자

### 4. 건너뛴 문서 분석
**파일**: `skipped_documents_analysis.md`

**용도**:
- 건너뛴 문서 원인 분석
- 데이터 품질 관리
- 원본 데이터 개선 방안 수립

**대상**: 데이터 품질 관리자, 개발자

## 빠른 참조

### 재임베딩 실행
```bash
python scripts/migrations/re_embed_existing_data_optimized.py \
    --db data/lawfirm_v2.db \
    --chunking-strategy dynamic \
    --version-id 5
```

### 진행 상황 모니터링
```bash
python scripts/monitor_re_embedding_progress.py \
    --db data/lawfirm_v2.db \
    --version-id 5
```

### 성능 확인
```bash
python scripts/check_re_embedding_performance.py \
    --db data/lawfirm_v2.db \
    --version-id 5
```

## 유지보수 체크리스트

### 재임베딩 전
- [ ] 원본 테이블에 데이터가 있는지 확인
- [ ] 시스템 사양 확인 (CPU, 메모리)
- [ ] 디스크 공간 확인
- [ ] 데이터베이스 백업

### 재임베딩 중
- [ ] 진행 상황 모니터링
- [ ] 성능 확인
- [ ] 메모리 사용량 확인
- [ ] 오류 로그 확인

### 재임베딩 후
- [ ] 처리 결과 확인
- [ ] 건너뛴 문서 분석
- [ ] FAISS 인덱스 빌드
- [ ] 성능 테스트

## 관련 스크립트

### 모니터링 스크립트
- `scripts/monitor_re_embedding_progress.py`: 진행 상황 모니터링
- `scripts/check_re_embedding_performance.py`: 성능 확인
- `scripts/monitor_re_embedding_speed.py`: 실시간 속도 모니터링

### 유틸리티 스크립트
- `scripts/check_system_specs.py`: 시스템 사양 확인
- `scripts/check_pytorch_threads.py`: PyTorch 스레드 설정 확인
- `scripts/check_statute_article_status.py`: statute_article 처리 상태 확인

## 문서 업데이트 이력

- **2025-11-15**: 문서 통합 및 정리 완료
  - 14개 재임베딩 관련 문서 → 4개 통합 문서
  - 3개 건너뛴 문서 관련 문서 → 1개 통합 문서
  - 중복 문서 삭제 및 구조화

## 관련 문서

- [벡터 임베딩 가이드](embedding_guide.md)
- [FAISS 버전 관리 가이드](faiss_version_management_guide.md)
- [버전 관리 사용법](version_management_guide.md)

