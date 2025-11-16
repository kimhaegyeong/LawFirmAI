# 건너뛴 문서 분석 보고서

## 개요

재임베딩 작업 중 건너뛴 문서 465개에 대한 상세 분석 결과입니다.

## 건너뛴 문서 현황

### 전체 통계
- **전체 원본 문서**: 14,306개
- **처리된 문서**: 13,841개 (96.8%)
- **건너뛴 문서**: 465개 (3.2%)
  - **statute_article**: 465개 (모두 건너뜀)

### 건너뛴 문서 상세 분석

#### 1. 텍스트 없음: 171개
- 원본 테이블에 텍스트가 없거나 빈 문자열
- 처리 불가 (정상적으로 건너뛰어야 함)

#### 2. 텍스트 너무 짧음 (<50자): 294개
- 원본 텍스트는 있지만 50자 미만
- 예: 14자, 29자, 43자, 48자, 49자 등
- `min_chars=300` 조건을 만족하지 못함
- fallback 로직에서도 50자 미만은 처리하지 않음

#### 3. 처리 가능하지만 건너뜀: 0개
- 모든 처리 가능한 문서는 이미 처리됨

## 원인 분석

### 문제 1: min_chars 조건이 너무 높음
- `statute_article`의 `min_chars=300` (config.py에서 설정)
- 실제 많은 statute_article이 300자 미만
- fallback 로직에서도 50자 미만은 처리하지 않음

### 문제 2: 조건 로직 오류 (해결됨)
- 기존 코드: `if len(text_stripped) >= min_chars or len(text_stripped) >= 50:`
- 이 조건은 논리적으로 `len(text_stripped) >= 50`과 같음
- 하지만 `min_chars=300`이므로 50자 이상 300자 미만인 문서도 처리되지 않음

**해결**: fallback 로직에서 최소 50자 이상이면 무조건 청크 생성
```python
# min_chars가 300이어도, 50자 이상이면 최소한 1개 청크는 생성
if len(text_stripped) >= 50:
```

## 해결 방법

### 수정 사항
**파일**: `scripts/utils/text_chunker.py`

**변경 내용**: fallback 로직에서 최소 50자 이상이면 무조건 청크 생성

```python
def chunk_statute(sentences, min_chars=200, max_chars=1200, overlap_ratio=0.2):
    chunks = []
    articles = split_statute_sentences_into_articles(sentences)
    
    # Fallback: article이 생성되지 않으면 원본 텍스트를 직접 청킹
    if not articles:
        full_text = "\n".join(sentences)
        if not full_text or not full_text.strip():
            return chunks
        
        # min_chars가 300이어도, 50자 이상이면 최소한 1개 청크는 생성
        text_stripped = full_text.strip()
        if len(text_stripped) >= 50:
            # 청크 생성 로직
            # ...
        return chunks
```

### 예상 효과
- 50자 이상 300자 미만인 문서도 처리됨
- 약 294개 문서 추가 처리 가능 (실제로는 모두 50자 미만이어서 처리 불가)

## 실제 분석 결과

### 처리되지 않은 문서 상세
- **50자 이상인 문서**: 0개
- **50자 미만인 문서**: 294개
- **텍스트 없음**: 171개

### 결론
**건너뛴 문서 465개 중:**
- **171개**: 텍스트 없음 (처리 불가, 정상)
- **294개**: 텍스트 너무 짧음 (<50자) (처리 불가, 정상)

**모든 건너뛴 문서는 처리 불가능한 문서로 정상 동작합니다.**

## 권장 사항

### 1. 원본 데이터 품질 개선
- 텍스트가 없는 문서는 원본 데이터 수집 단계에서 제외
- 너무 짧은 문서(50자 미만)는 별도 처리 또는 제외 고려

### 2. 청킹 전략 조정
- 50자 미만 문서도 처리하려면 `min_chars` 조건 완화 고려
- 단, 너무 짧은 문서는 검색 품질에 부정적 영향 가능

### 3. 모니터링
- 재임베딩 시 건너뛴 문서 수 모니터링
- 건너뛴 문서 비율이 비정상적으로 높으면 원인 조사

## 관련 문서

- [완료 보고서](./re_embedding_complete_report.md): 전체 작업 결과
- [문제 해결 가이드](./re_embedding_troubleshooting.md): 발생한 문제 및 해결 방법

