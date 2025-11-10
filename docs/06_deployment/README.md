# 배포 (Deployment)

이 섹션은 LawFirmAI 애플리케이션 배포 및 운영에 대한 문서를 포함합니다.

## 📋 주요 문서

### 🎯 최종 배포 가이드 (권장)
- **[최종 배포 가이드](DEPLOYMENT_FINAL.md)** ⭐ - **이 문서부터 시작하세요!**
  - 전체 배포 과정을 한 문서에 통합
  - AWS 배포, 데이터베이스 설정, CI/CD, 모니터링 등 모든 내용 포함
  - 단계별 배포 절차 및 문제 해결 가이드

### 📚 상세 가이드

#### 배포 관련
- [배포 체크리스트](DEPLOYMENT_CHECKLIST.md) - 배포 전 확인사항
- [AWS 빠른 시작](QUICK_START_AWS.md) - 빠른 배포를 위한 단계별 가이드

#### 데이터베이스 관련
- [PostgreSQL 마이그레이션 계획](POSTGRESQL_MIGRATION_PLAN.md) - PostgreSQL 마이그레이션 계획
- [PostgreSQL 설정 가이드](POSTGRESQL_SETUP_GUIDE.md) - PostgreSQL 설정 방법
- [데이터베이스 마이그레이션 가이드](DATABASE_MIGRATION_GUIDE.md) - 데이터 마이그레이션 방법

#### 보안 및 최적화
- [Nginx 보안 가이드](NGINX_SECURITY.md) - Nginx 보안 설정
- [프리 티어 최적화 가이드](FREE_TIER_OPTIMIZATION.md) - 프리 티어 최적화 방법

## 🚀 빠른 시작

### AWS 배포 (권장)

1. **[최종 배포 가이드](DEPLOYMENT_FINAL.md)** 읽기
2. [배포 체크리스트](DEPLOYMENT_CHECKLIST.md) 확인
3. [AWS 빠른 시작](QUICK_START_AWS.md) 따라하기

### 환경별 데이터베이스

| 환경 | 데이터베이스 | 설정 |
|------|------------|------|
| **로컬 개발** | SQLite | `DATABASE_URL=sqlite:///./data/lawfirm.db` |
| **개발 서버** | PostgreSQL | `DATABASE_URL=postgresql://user:pass@postgres:5432/db` |
| **운영 서버** | PostgreSQL | `DATABASE_URL=postgresql://user:pass@postgres:5432/db` |

### 배포 옵션

| 옵션 | 비용 | 권장 용도 |
|------|------|----------|
| **프리 티어** | $0/월 (12개월) | 테스트, 학습 |
| **프로덕션** | $70-150/월 | 실제 서비스 |
| **고가용성** | $200-300/월 | 대규모 서비스 |

## 🔗 관련 섹션

- [01_Getting_Started](../01_getting_started/README.md): 프로젝트 개요
- [07_API](../07_api/README.md): API 문서
- [10_Technical_Reference](../10_technical_reference/README.md): 기술 참고 문서
