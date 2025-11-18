# Scripts Utils 모듈

analysis 스크립트에서 사용하는 공통 유틸리티 모듈입니다.

## 모듈 구조

```
scripts/utils/
├── __init__.py              # 모든 유틸리티 통합 export
├── path_utils.py            # 프로젝트 경로 설정
├── file_utils.py            # JSON 파일 처리
├── text_utils.py            # 텍스트 처리 (키워드 추출 등)
├── log_analyzer.py          # 로그 분석
└── report_utils.py          # 리포트 생성
```

## 사용 방법

### 1. 프로젝트 경로 설정 (path_utils)

```python
from scripts.utils.path_utils import setup_project_path, get_project_root

# 프로젝트 루트를 sys.path에 추가
project_root = setup_project_path()

# 프로젝트 루트 경로만 가져오기 (sys.path 설정 없이)
project_root = get_project_root()
```

**사용 예제:**
```python
from scripts.utils.path_utils import setup_project_path

# 스크립트 시작 부분에 추가
setup_project_path()

# 이제 프로젝트 모듈을 import할 수 있습니다
from lawfirm_langgraph.core.search.engines import SemanticSearchEngineV2
```

### 2. JSON 파일 처리 (file_utils)

```python
from scripts.utils.file_utils import load_json_file, save_json_file, load_json_files

# 단일 파일 로드
data = load_json_file('data/embeddings/metadata.json')

# 단일 파일 저장
save_json_file(data, 'data/embeddings/metadata_improved.json')

# 디렉토리 내 모든 JSON 파일 로드
all_data = load_json_files('data/processed/laws')
```

**사용 예제:**
```python
from scripts.utils.file_utils import load_json_file, save_json_file

# 메타데이터 로드
metadata = load_json_file('data/embeddings/metadata.json')

# 데이터 수정
metadata['updated_at'] = '2024-01-01'

# 저장 (디렉토리 자동 생성)
save_json_file(metadata, 'data/embeddings/metadata_updated.json')
```

### 3. 텍스트 처리 (text_utils)

```python
from scripts.utils.text_utils import (
    extract_keywords,
    normalize_text,
    remove_special_chars,
    extract_legal_terms,
    calculate_text_similarity
)

# 키워드 추출
keywords = extract_keywords("계약 해지 사유에 대해 알려주세요")
# 결과: ['계약', '해지', '사유']

# 텍스트 정규화
normalized = normalize_text("  여러   공백이    있는   텍스트  ")
# 결과: "여러 공백이 있는 텍스트"

# 특수문자 제거
cleaned = remove_special_chars("테스트!@#$텍스트", keep_spaces=True)
# 결과: "테스트텍스트"

# 법률 용어 추출
terms = extract_legal_terms("민법 제123조에 따라 계약법 규칙을 적용합니다")
# 결과: {'민법', '계약법', '규칙'}

# 텍스트 유사도 계산
similarity = calculate_text_similarity("계약 해지", "계약 해지 사유")
# 결과: 0.67 (Jaccard 유사도)
```

**사용 예제:**
```python
from scripts.utils.text_utils import extract_keywords, calculate_text_similarity

# 검색 쿼리에서 키워드 추출
query = "전세금 반환 보증에 대해 알려주세요"
keywords = extract_keywords(query)
# ['전세금', '반환', '보증']

# 검색 결과와 쿼리 유사도 계산
for result in search_results:
    similarity = calculate_text_similarity(query, result['text'])
    result['similarity'] = similarity
```

### 4. 로그 분석 (log_analyzer)

```python
from scripts.utils.log_analyzer import (
    analyze_sources_conversion_logs,
    analyze_legal_references_logs,
    analyze_answer_length_logs,
    analyze_context_usage_logs,
    identify_improvements
)

# 로그 파일 읽기
with open('logs/lawfirm_ai.log', 'r', encoding='utf-8') as f:
    log_content = f.read()

# Sources 변환 로그 분석
sources_analysis = analyze_sources_conversion_logs(log_content)

# Legal References 로그 분석
legal_analysis = analyze_legal_references_logs(log_content)

# 답변 길이 로그 분석
length_analysis = analyze_answer_length_logs(log_content)

# Context Usage 로그 분석
context_analysis = analyze_context_usage_logs(log_content)

# 전체 분석 결과
analysis_results = {
    "sources": sources_analysis,
    "legal_references": legal_analysis,
    "answer_length": length_analysis,
    "context_usage": context_analysis
}

# 개선 사항 식별
improvements = identify_improvements(analysis_results)
```

**사용 예제:**
```python
from scripts.utils.log_analyzer import (
    analyze_sources_conversion_logs,
    identify_improvements
)

# 로그 분석
log_content = read_log_file()
sources_analysis = analyze_sources_conversion_logs(log_content)

# 개선 사항 확인
analysis_results = {"sources": sources_analysis}
improvements = identify_improvements(analysis_results)

for improvement in improvements:
    print(f"[{improvement['priority']}] {improvement['category']}")
    print(f"  현재: {improvement['current']}")
    print(f"  목표: {improvement['target']}")
```

### 5. 리포트 생성 (report_utils)

```python
from scripts.utils.report_utils import (
    print_section_header,
    print_subsection_header,
    print_metrics,
    print_table,
    print_improvements,
    print_summary,
    save_text_report,
    generate_markdown_report
)

# 섹션 헤더 출력
print_section_header("분석 결과")

# 서브섹션 헤더 출력
print_subsection_header("품질 메트릭")

# 메트릭 출력
print_metrics({
    "평균 점수": 0.85,
    "총 개수": 100
})

# 테이블 출력
data = [
    {"이름": "항목1", "값": 10},
    {"이름": "항목2", "값": 20}
]
print_table(data, headers=["이름", "값"])

# 개선 사항 출력
print_improvements(improvements)

# 요약 정보 출력
print_summary({
    "총 문서": 1000,
    "유효 문서": 950,
    "품질 점수": 0.95
})

# 텍스트 리포트 저장
save_text_report("리포트 내용", Path("reports/report.txt"))

# 마크다운 리포트 생성
sections = [
    {
        "title": "요약",
        "type": "metrics",
        "data": {"평균": 0.85, "총계": 100}
    },
    {
        "title": "상세 결과",
        "type": "table",
        "data": {
            "headers": ["항목", "값"],
            "rows": [["항목1", "10"], ["항목2", "20"]]
        }
    }
]
markdown = generate_markdown_report(
    title="분석 리포트",
    sections=sections,
    output_path=Path("reports/analysis.md")
)
```

**사용 예제:**
```python
from scripts.utils.report_utils import (
    print_section_header,
    print_metrics,
    print_improvements,
    generate_markdown_report
)
from pathlib import Path

# 콘솔 출력
print_section_header("참조 자료 품질 분석")
print_metrics({
    "평균 유사도": 0.75,
    "고품질 결과": 50,
    "저품질 결과": 10
})
print_improvements(improvements)

# 마크다운 리포트 생성
sections = [
    {
        "title": "품질 메트릭",
        "type": "metrics",
        "data": analysis['quality_metrics']
    },
    {
        "title": "개선 사항",
        "type": "list",
        "data": {
            "items": [imp['description'] for imp in improvements]
        }
    }
]
generate_markdown_report(
    title="참조 자료 품질 분석 리포트",
    sections=sections,
    output_path=Path("reports/quality_analysis.md")
)
```

## 통합 사용 예제

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
분석 스크립트 예제
"""

from pathlib import Path
from scripts.utils.path_utils import setup_project_path
from scripts.utils.file_utils import load_json_file, save_json_file
from scripts.utils.text_utils import extract_keywords
from scripts.utils.report_utils import (
    print_section_header,
    print_metrics,
    generate_markdown_report
)

# 1. 프로젝트 경로 설정
setup_project_path()

# 2. 데이터 로드
data = load_json_file('data/embeddings/metadata.json')

# 3. 텍스트 처리
query = "계약 해지 사유에 대해 알려주세요"
keywords = extract_keywords(query)

# 4. 분석 수행
# ... 분석 로직 ...

# 5. 결과 출력
print_section_header("분석 결과")
print_metrics({
    "키워드 수": len(keywords),
    "매칭 결과": len(results)
})

# 6. 리포트 생성
sections = [
    {
        "title": "요약",
        "type": "metrics",
        "data": analysis_metrics
    }
]
generate_markdown_report(
    title="분석 리포트",
    sections=sections,
    output_path=Path("reports/analysis.md")
)

# 7. 결과 저장
save_json_file(analysis_results, 'data/analysis_results.json')
```

## 테스트

단위 테스트는 `scripts/tests/` 디렉토리에 있습니다:

```bash
# 모든 유틸리티 테스트 실행
pytest scripts/tests/test_utils_*.py -v

# 특정 모듈 테스트
pytest scripts/tests/test_utils_file.py -v
```

## 주의사항

1. **프로젝트 경로 설정**: 모든 스크립트 시작 부분에 `setup_project_path()`를 호출하세요.
2. **파일 인코딩**: 모든 파일은 UTF-8 인코딩을 사용합니다.
3. **에러 처리**: `file_utils`의 함수들은 파일이 없을 때 `FileNotFoundError`를 발생시킵니다.
4. **경로**: 상대 경로는 프로젝트 루트 기준입니다.

## 기여

새로운 유틸리티 함수를 추가할 때:
1. 해당 모듈 파일에 함수 추가
2. `__init__.py`에 export 추가
3. 단위 테스트 작성
4. 이 README에 사용 예제 추가

