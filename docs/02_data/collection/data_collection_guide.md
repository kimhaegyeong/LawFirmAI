# 데이터 수집 가이드

LawFirmAI의 데이터 수집 시스템에 대한 상세 가이드입니다.

## 국가법령정보센터 LAW OPEN API 연동

LawFirmAI는 국가법령정보센터의 LAW OPEN API를 통해 법률 데이터를 수집합니다.

## 국회 법률정보시스템 웹 스크래핑

국회 법률정보시스템(https://likms.assembly.go.kr/law)을 사용합니다.

### Assembly 시스템으로 데이터 수집

```bash
# Assembly 시스템으로 법률 수집
python scripts/data_collection/assembly/collect_laws.py --sample 100

# Assembly 시스템으로 판례 수집
python scripts/data_collection/assembly/collect_precedents.py --sample 50

# 분야별 판례 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --category civil --sample 20
python scripts/data_collection/assembly/collect_precedents_by_category.py --category criminal --sample 20
python scripts/data_collection/assembly/collect_precedents_by_category.py --category family --sample 20

# 모든 분야 한번에 수집
python scripts/data_collection/assembly/collect_precedents_by_category.py --all-categories --sample 10

# 특정 페이지부터 수집
python scripts/data_collection/assembly/collect_laws.py --sample 50 --start-page 5 --no-resume
python scripts/data_collection/assembly/collect_precedents.py --sample 30 --start-page 3 --no-resume
```

### 지원 데이터 유형

- **법령**: 주요 법령 20개 (민법, 상법, 형법 등) - 모든 조문 및 개정이력 포함
- **판례**: 판례 5,000건 (최근 5년간)
- **헌재결정례**: 1,000건 (최근 5년간)
- **법령해석례**: 2,000건 (최근 3년간)
- **행정규칙**: 1,000건 (주요 부처별)
- **자치법규**: 500건 (주요 지자체별)
- **위원회결정문**: 500건 (주요 위원회별)
- **행정심판례**: 1,000건 (최근 3년간)
- **조약**: 100건 (주요 조약)

### 데이터 수집 실행

```bash
# 새로운 분리된 데이터 파이프라인 (권장)
python scripts/data_processing/run_data_pipeline.py --mode full --oc your_email_id

# 데이터 수집만 실행
python scripts/data_processing/run_data_pipeline.py --mode collect --oc your_email_id

# 벡터DB 구축만 실행
python scripts/data_processing/run_data_pipeline.py --mode build

# 개별 데이터 타입별 수집
python scripts/data_processing/run_data_pipeline.py --mode laws --oc your_email_id --query "민법"
python scripts/data_processing/run_data_pipeline.py --mode precedents --oc your_email_id --query "계약 해지"
python scripts/data_processing/run_data_pipeline.py --mode constitutional --oc your_email_id --query "헌법"
python scripts/data_processing/run_data_pipeline.py --mode interpretations --oc your_email_id --query "법령해석"
python scripts/data_processing/run_data_pipeline.py --mode administrative --oc your_email_id --query "행정규칙"
python scripts/data_processing/run_data_pipeline.py --mode local --oc your_email_id --query "자치법규"

# 여러 데이터 타입 동시 수집
python scripts/data_processing/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional
python scripts/data_processing/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "민법"
```

### API 설정

1. [국가법령정보센터 LAW OPEN API](https://open.law.go.kr/LSO/openApi/guideList.do)에서 OC 파라미터 발급
2. 환경변수 설정:
   ```bash
   export LAW_OPEN_API_OC='your_email_id_here'
   ```

### 사용 예시

**1. 전체 데이터 수집 및 벡터DB 구축**
```bash
# 모든 데이터 타입 수집 + 벡터DB 구축
python scripts/data_processing/run_data_pipeline.py --mode full --oc your_email_id
```

**2. 특정 데이터 타입만 수집**
```bash
# 법령 데이터만 수집
python scripts/data_processing/run_data_pipeline.py --mode laws --oc your_email_id --query "민법" --display 50

# 판례 데이터만 수집
python scripts/data_processing/run_data_pipeline.py --mode precedents --oc your_email_id --query "손해배상" --display 100
```

**3. 여러 데이터 타입 동시 수집**
```bash
# 법령, 판례, 헌재결정례 동시 수집
python scripts/data_processing/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents constitutional

# 특정 쿼리로 여러 타입 수집
python scripts/data_processing/run_data_pipeline.py --mode all_types --oc your_email_id --types laws precedents --query "계약"
```

**4. 데이터 수집과 벡터DB 구축 분리**
```bash
# 1단계: 데이터 수집만
python scripts/data_processing/run_data_pipeline.py --mode collect --oc your_email_id

# 2단계: 벡터DB 구축만
python scripts/data_processing/run_data_pipeline.py --mode build
```

## Q&A 데이터셋 생성

법령/판례 데이터를 기반으로 자동으로 Q&A 데이터셋을 생성합니다.

### Q&A 생성 실행

```bash
# 기본 Q&A 데이터셋 생성
python scripts/data_collection/qa_generation/generate_qa_dataset.py

# 향상된 Q&A 데이터셋 생성 (더 많은 패턴)
python scripts/data_collection/qa_generation/enhanced_generate_qa_dataset.py

# 대규모 Q&A 데이터셋 생성 (최대 규모)
python scripts/data_collection/qa_generation/large_scale_generate_qa_dataset.py

# LLM 기반 Q&A 데이터셋 생성 (자연스러운 질문-답변)
python scripts/data_collection/qa_generation/generate_qa_with_llm.py

# LLM 기반 생성 옵션 지정
python scripts/data_collection/qa_generation/generate_qa_with_llm.py \
  --model qwen2.5:7b \
  --data-type laws precedents \
  --output data/qa_dataset/llm_generated \
  --target 1000 \
  --max-items 20
```

### 생성 결과

**템플릿 기반 생성**
- **총 Q&A 쌍 수**: 2,709개
- **평균 품질 점수**: 93.5%
- **고품질 비율**: 99.96%
- **데이터 소스**: 법령 및 판례 데이터

**LLM 기반 생성**
- **총 Q&A 쌍 수**: 계속 증가 중
- **질문 유형**: 다양한 패턴
- **자연스러움**: 법률 실무 중심 질문 생성

### 생성된 파일

**템플릿 기반 파일**
- `data/qa_dataset/large_scale_qa_dataset.json` - 전체 데이터셋
- `data/qa_dataset/large_scale_qa_dataset_high_quality.json` - 고품질 데이터셋
- `data/qa_dataset/large_scale_qa_dataset_statistics.json` - 통계 정보
- `docs/qa_dataset_quality_report.md` - 품질 보고서

**LLM 기반 파일**
- `data/qa_dataset/llm_generated/llm_qa_dataset.json` - LLM 생성 전체 데이터셋
- `data/qa_dataset/llm_generated/llm_qa_dataset_high_quality.json` - 고품질 데이터셋
- `data/qa_dataset/llm_generated/llm_qa_dataset_statistics.json` - 통계 정보
- `docs/llm_qa_dataset_quality_report.md` - LLM 품질 보고서

### Q&A 유형

**템플릿 기반 유형**
- **법령 정의 Q&A**: 법률의 목적과 정의에 관한 질문
- **조문 내용 Q&A**: 특정 조문의 내용과 의미
- **조문 제목 Q&A**: 조문의 제목과 주제
- **키워드 기반 Q&A**: 법률 용어와 개념 설명
- **판례 쟁점 Q&A**: 사건의 핵심 쟁점과 문제
- **판결 내용 Q&A**: 법원의 판단과 결론

**LLM 기반 유형**
- **개념 설명**: "~란 무엇인가요?"
- **실제 적용**: "~한 경우 어떻게 해야 하나요?"
- **요건/효과**: "~의 요건은 무엇인가요?"
- **비교/차이**: "~와 ~의 차이는 무엇인가요?"
- **절차**: "~하려면 어떤 절차를 거쳐야 하나요?"
- **예시**: "~의 구체적인 예시를 들어주세요"
- **주의사항**: "~할 때 주의할 점은 무엇인가요?"
- **적용 범위**: "~이 적용되는 대상은 무엇인가요?"
- **목적**: "~의 목적은 무엇인가요?"
- **법적 근거**: "~의 법적 근거는 무엇인가요?"
- **실무 적용**: "실무에서 ~는 어떻게 적용되나요?"
- **예외 사항**: "~의 예외 사항은 무엇인가요?"

