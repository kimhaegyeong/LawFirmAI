# 법률 데이터 텍스트 청킹 전략 개발 문서

## 📋 문서 개요

본 문서는 LawFirmAI 프로젝트에서 실제 수집된 법률 데이터의 구조를 분석하여 효과적인 검색과 RAG(Retrieval-Augmented Generation) 성능을 위한 텍스트 청킹 전략을 정의합니다.

### 📊 실제 수집 데이터 현황
- **법령 데이터**: 20개 (민법, 상법, 형법 등)
- **판례 데이터**: 7,699건 (90개 배치 파일)
- **헌재결정례**: 2,000건 (40개 배치 파일)
- **법령해석례**: 158건 (4개 배치 파일)

---

## 🎯 청킹 전략 개요

### 1.1 청킹의 목적
- **의미적 일관성 보장**: 법률 조문의 완전한 의미를 유지
- **검색 정확도 향상**: 관련성 높은 문서 조각 검색
- **컨텍스트 보존**: 법리 추론에 필요한 맥락 유지
- **처리 효율성**: 대용량 데이터의 효율적 처리

### 1.2 법률 데이터의 특성
- **구조화된 텍스트**: 조문, 항목, 호목으로 계층화
- **의미적 밀도**: 짧은 문장에 많은 법적 의미 포함
- **상호 참조성**: 다른 조문과의 연관성 중요
- **용어의 정확성**: 법률 용어의 정확한 해석 필요

---

## 🔧 청킹 방법론

### 2.1 다층 청킹 전략 (Multi-Level Chunking)

#### 2.1.1 계층적 청킹 구조
```
Level 1: 문서 단위 (Document Level)
├── Level 2: 조문 단위 (Article Level)
│   ├── Level 3: 항목 단위 (Paragraph Level)
│   │   └── Level 4: 의미 단위 (Semantic Level)
│   └── Level 3: 부칙 단위 (Supplementary Level)
└── Level 2: 부칙 단위 (Supplementary Level)
```

#### 2.1.2 각 레벨별 특성
- **Level 1**: 전체 법령의 맥락과 목적
- **Level 2**: 개별 조문의 완전한 의미
- **Level 3**: 조문 내 세부 규정
- **Level 4**: 최소 의미 단위

### 2.2 데이터 유형별 청킹 전략

#### 2.2.1 법령 데이터 청킹 (실제 데이터 구조 기반)

##### A. 법령 구조 분석
```json
{
  "basic_info": {
    "name": "민법",
    "id": "001706",
    "category": "기본법"
  },
  "current_text": {
    "법령": {
      "개정문": {
        "개정문내용": ["조문 내용 배열"]
      },
      "부칙": {
        "부칙단위": [{"부칙키": "...", "부칙내용": "..."}]
      }
    }
  }
}
```

##### B. 조문 중심 청킹 (Article-Centric Chunking)
```python
def law_article_chunking(law_data: Dict) -> List[Dict]:
    """
    실제 법령 데이터 구조를 기반으로 한 조문 중심 청킹
    
    청킹 규칙:
    1. 개정문내용 배열에서 조문별 분할
    2. "제X조" 패턴으로 조문 경계 식별
    3. 조문 제목과 본문을 함께 포함
    4. 부칙은 별도 청크로 분리
    """
    
    chunks = []
    current_text = law_data.get('current_text', {})
    
    # 개정문 내용 처리
    if '법령' in current_text and '개정문' in current_text['법령']:
        amendment_content = current_text['법령']['개정문']['개정문내용']
        
        for content_array in amendment_content:
            for content_item in content_array:
                # 조문 패턴 매칭
                article_pattern = r'제(\d+)조[^제]*?(?=제\d+조|$)'
                
                for match in re.finditer(article_pattern, content_item, re.DOTALL):
                    article_num = match.group(1)
                    article_content = match.group(0).strip()
                    
                    chunk = {
                        'type': 'article',
                        'law_name': law_data['basic_info']['name'],
                        'law_id': law_data['basic_info']['id'],
                        'article_number': int(article_num),
                        'content': article_content,
                        'title': extract_article_title(article_content),
                        'size': len(article_content),
                        'source': 'amendment'
                    }
                    chunks.append(chunk)
    
    # 부칙 처리
    if '법령' in current_text and '부칙' in current_text['법령']:
        supplementary_rules = current_text['법령']['부칙']['부칙단위']
        
        for rule in supplementary_rules:
            if '부칙내용' in rule:
                chunk = {
                    'type': 'supplementary_rule',
                    'law_name': law_data['basic_info']['name'],
                    'law_id': law_data['basic_info']['id'],
                    'rule_key': rule.get('부칙키', ''),
                    'content': rule['부칙내용'],
                    'size': len(rule['부칙내용']),
                    'source': 'supplementary'
                }
                chunks.append(chunk)
    
    return chunks
```

##### B. 항목별 청킹 (Paragraph-Level Chunking)
```python
def paragraph_level_chunking(article_text: str) -> List[Dict]:
    """
    조문 내 항목별 청킹 전략
    
    청킹 규칙:
    1. "제X항" 패턴으로 항목 경계 식별
    2. 각 항목을 독립적인 청크로 처리
    3. 항목 간 연관성 고려한 오버랩 적용
    4. 호목은 상위 항목과 함께 포함
    """
    
    # 항목 패턴 매칭
    paragraph_pattern = r'제(\d+)항\s*([^제]*?)(?=제\d+항|$)'
    
    chunks = []
    for match in re.finditer(paragraph_pattern, article_text, re.DOTALL):
        para_num = match.group(1)
        para_content = match.group(2).strip()
        
        # 호목 포함 여부 확인
        sub_items = extract_sub_items(para_content)
        
        chunk = {
            'type': 'paragraph',
            'paragraph_number': int(para_num),
            'content': para_content,
            'sub_items': sub_items,
            'size': len(para_content)
        }
        chunks.append(chunk)
    
    return chunks
```

#### 2.2.2 판례 데이터 청킹 (실제 데이터 구조 기반)

##### A. 판례 구조 분석
```json
{
  "metadata": {
    "category": "형사",
    "count": 952,
    "batch_id": "20250925_111703"
  },
  "precedents": [
    {
      "id": "14",
      "사건번호": "2020도12563",
      "사건종류명": "형사",
      "선고일자": "2022.10.27",
      "법원명": "대법원",
      "사건명": "금융실명거래및비밀보장에관한법률위반방조..."
    }
  ]
}
```

##### B. 사건별 청킹 (Case-Level Chunking)
```python
def precedent_case_chunking(precedent_data: Dict) -> List[Dict]:
    """
    실제 판례 데이터 구조를 기반으로 한 사건별 청킹
    
    청킹 규칙:
    1. 사건번호로 사건 경계 식별
    2. 사건명에서 쟁점 추출
    3. 각 사건을 독립적인 청크로 처리
    4. 법원, 사건종류 등 메타데이터 포함
    """
    
    chunks = []
    
    # 사건 기본 정보 청크
    basic_chunk = {
        'type': 'case_basic_info',
        'case_id': precedent_data.get('id', ''),
        'case_number': precedent_data.get('사건번호', ''),
        'case_type': precedent_data.get('사건종류명', ''),
        'court_name': precedent_data.get('법원명', ''),
        'decision_date': precedent_data.get('선고일자', ''),
        'case_name': precedent_data.get('사건명', ''),
        'content': f"사건번호: {precedent_data.get('사건번호', '')}\n"
                  f"사건명: {precedent_data.get('사건명', '')}\n"
                  f"법원: {precedent_data.get('법원명', '')}\n"
                  f"선고일자: {precedent_data.get('선고일자', '')}",
        'size': len(precedent_data.get('사건명', ''))
    }
    chunks.append(basic_chunk)
    
    # 사건명에서 쟁점 추출하여 별도 청크 생성
    case_name = precedent_data.get('사건명', '')
    if case_name and len(case_name) > 50:  # 긴 사건명의 경우
        issue_chunk = {
            'type': 'case_issue',
            'case_id': precedent_data.get('id', ''),
            'case_number': precedent_data.get('사건번호', ''),
            'content': case_name,
            'extracted_issues': extract_legal_issues_from_case_name(case_name),
            'size': len(case_name)
        }
        chunks.append(issue_chunk)
    
    return chunks

def extract_legal_issues_from_case_name(case_name: str) -> List[str]:
    """
    사건명에서 법적 쟁점 추출
    """
    issues = []
    
    # 법률 위반 관련 키워드 추출
    violation_patterns = [
        r'([가-힣]+법[가-힣]*위반)',
        r'([가-힣]+죄)',
        r'([가-힣]+행위)',
        r'([가-힣]+방조)',
        r'([가-힣]+공모)'
    ]
    
    for pattern in violation_patterns:
        matches = re.findall(pattern, case_name)
        issues.extend(matches)
    
    return list(set(issues))
```

##### B. 법리 중심 청킹 (Legal Reasoning Chunking)
```python
def legal_reasoning_chunking(reasoning_text: str) -> List[Dict]:
    """
    판결이유의 법리 중심 청킹 전략
    
    청킹 규칙:
    1. 법리별로 의미 단위 분할
    2. 판례 인용과 법령 조항을 함께 포함
    3. 논리적 순서 유지
    4. 법률 용어의 정확한 맥락 보존
    """
    
    # 법리 단위 분할
    legal_units = split_by_legal_reasoning(reasoning_text)
    
    chunks = []
    for i, unit in enumerate(legal_units):
        chunk = {
            'type': 'legal_reasoning',
            'unit_number': i + 1,
            'content': unit,
            'legal_terms': extract_legal_terms(unit),
            'case_citations': extract_case_citations(unit),
            'law_citations': extract_law_citations(unit),
            'size': len(unit)
        }
        chunks.append(chunk)
    
    return chunks
```

#### 2.2.3 헌재결정례 청킹 (실제 데이터 구조 기반)

##### A. 헌재결정례 구조 분석
```json
{
  "metadata": {
    "category": "constitutional_decisions_0",
    "count": 50,
    "batch_size": 50
  },
  "data": [
    {
      "basic_info": {
        "사건번호": "2017헌바323",
        "종국일자": "0",
        "사건명": ""
      },
      "detail_info": {
        "DetcService": {
          "사건번호": "2017헌바323",
          "심판대상조문": "",
          "판시사항": "",
          "결정요지": "",
          "사건종류명": "헌바"
        }
      }
    }
  ]
}
```

##### B. 쟁점별 청킹 (Issue-Based Chunking)
```python
def constitutional_issue_chunking(decision_data: Dict) -> List[Dict]:
    """
    실제 헌재결정례 데이터 구조를 기반으로 한 쟁점별 청킹
    
    청킹 규칙:
    1. 사건번호로 사건 경계 식별
    2. 심판대상조문, 판시사항, 결정요지로 구조화
    3. 각 구조 요소를 독립적인 청크로 처리
    4. 헌법적 쟁점 중심으로 분할
    """
    
    chunks = []
    
    # 기본 정보 청크
    basic_info = decision_data.get('basic_info', {})
    detail_info = decision_data.get('detail_info', {}).get('DetcService', {})
    
    # 사건 기본 정보
    basic_chunk = {
        'type': 'constitutional_basic_info',
        'case_number': basic_info.get('사건번호', ''),
        'case_name': basic_info.get('사건명', ''),
        'decision_date': basic_info.get('종국일자', ''),
        'case_type': detail_info.get('사건종류명', ''),
        'content': f"사건번호: {basic_info.get('사건번호', '')}\n"
                  f"사건명: {basic_info.get('사건명', '')}\n"
                  f"종국일자: {basic_info.get('종국일자', '')}\n"
                  f"사건종류: {detail_info.get('사건종류명', '')}",
        'size': len(basic_info.get('사건명', ''))
    }
    chunks.append(basic_chunk)
    
    # 심판대상조문 청크
    if detail_info.get('심판대상조문'):
        target_article_chunk = {
            'type': 'constitutional_target_article',
            'case_number': basic_info.get('사건번호', ''),
            'content': detail_info['심판대상조문'],
            'size': len(detail_info['심판대상조문'])
        }
        chunks.append(target_article_chunk)
    
    # 판시사항 청크
    if detail_info.get('판시사항'):
        holding_chunk = {
            'type': 'constitutional_holding',
            'case_number': basic_info.get('사건번호', ''),
            'content': detail_info['판시사항'],
            'size': len(detail_info['판시사항'])
        }
        chunks.append(holding_chunk)
    
    # 결정요지 청크
    if detail_info.get('결정요지'):
        decision_summary_chunk = {
            'type': 'constitutional_decision_summary',
            'case_number': basic_info.get('사건번호', ''),
            'content': detail_info['결정요지'],
            'size': len(detail_info['결정요지'])
        }
        chunks.append(decision_summary_chunk)
    
    return chunks
```

#### 2.2.4 법령해석례 청킹 (실제 데이터 구조 기반)

##### A. 법령해석례 구조 분석
```json
{
  "metadata": {
    "category": "기타",
    "count": 158,
    "batch_id": "20250925_145952"
  },
  "interpretations": [
    {
      "id": "1",
      "안건명": "1959년 12월 31일 이전에 퇴직한 군인의 퇴직급여금 지급에 관한특별법 시행령 제4조제2항 및 3항",
      "질의기관명": "국방부",
      "회신기관명": "법제처",
      "질의요지": "재직기간 계산 방법에 대한 질의",
      "회답": "현역병 복무연한을 공제한 후 전투근무기간을 3배로 계산...",
      "이유": "상세한 해석 이유..."
    }
  ]
}
```

##### B. 질의-회답 중심 청킹 (Q&A-Based Chunking)
```python
def legal_interpretation_chunking(interpretation_data: Dict) -> List[Dict]:
    """
    실제 법령해석례 데이터 구조를 기반으로 한 질의-회답 중심 청킹
    
    청킹 규칙:
    1. 질의요지와 회답을 별도 청크로 분리
    2. 해석 이유를 의미 단위로 분할
    3. 관련 법령 조문 추출하여 별도 청크 생성
    4. 기관 정보를 메타데이터로 포함
    """
    
    chunks = []
    
    # 기본 정보 청크
    basic_chunk = {
        'type': 'interpretation_basic_info',
        'case_id': interpretation_data.get('id', ''),
        'case_name': interpretation_data.get('안건명', ''),
        'inquiry_agency': interpretation_data.get('질의기관명', ''),
        'reply_agency': interpretation_data.get('회신기관명', ''),
        'content': f"안건명: {interpretation_data.get('안건명', '')}\n"
                  f"질의기관: {interpretation_data.get('질의기관명', '')}\n"
                  f"회신기관: {interpretation_data.get('회신기관명', '')}",
        'size': len(interpretation_data.get('안건명', ''))
    }
    chunks.append(basic_chunk)
    
    # 질의요지 청크
    if interpretation_data.get('질의요지'):
        question_chunk = {
            'type': 'interpretation_question',
            'case_id': interpretation_data.get('id', ''),
            'content': interpretation_data['질의요지'],
            'size': len(interpretation_data['질의요지'])
        }
        chunks.append(question_chunk)
    
    # 회답 청크
    if interpretation_data.get('회답'):
        answer_chunk = {
            'type': 'interpretation_answer',
            'case_id': interpretation_data.get('id', ''),
            'content': interpretation_data['회답'],
            'size': len(interpretation_data['회답'])
        }
        chunks.append(answer_chunk)
    
    # 해석 이유 청크 (긴 경우 의미 단위로 분할)
    if interpretation_data.get('이유'):
        reason_content = interpretation_data['이유']
        if len(reason_content) > 1000:  # 긴 해석 이유의 경우 분할
            reason_chunks = split_interpretation_reason(reason_content)
            for i, reason_chunk in enumerate(reason_chunks):
                chunk = {
                    'type': 'interpretation_reason',
                    'case_id': interpretation_data.get('id', ''),
                    'part_number': i + 1,
                    'content': reason_chunk,
                    'size': len(reason_chunk)
                }
                chunks.append(chunk)
        else:
            reason_chunk = {
                'type': 'interpretation_reason',
                'case_id': interpretation_data.get('id', ''),
                'content': reason_content,
                'size': len(reason_content)
            }
            chunks.append(reason_chunk)
    
    return chunks

def split_interpretation_reason(reason_text: str) -> List[str]:
    """
    해석 이유를 의미 단위로 분할
    """
    # 문단 단위로 분할 (빈 줄 기준)
    paragraphs = [p.strip() for p in reason_text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > 800:  # 800자 제한
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### 2.3 의미적 청킹 전략 (Semantic Chunking)

#### 2.3.1 문장 경계 인식 청킹
```python
def sentence_boundary_chunking(text: str, max_size: int = 500) -> List[Dict]:
    """
    문장 경계를 인식한 의미적 청킹
    
    청킹 규칙:
    1. 문장 단위로 분할 (마침표, 물음표, 느낌표)
    2. 최대 크기 내에서 문장 단위로 조합
    3. 문장 간 의미적 연관성 고려
    4. 법률 문장의 특수성 반영
    """
    
    # 문장 분할 (법률 문장 특성 고려)
    sentences = split_legal_sentences(text)
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # 크기 초과 시 현재 청크 저장
        if current_size + sentence_size > max_size and current_chunk:
            chunks.append(create_chunk(current_chunk))
            current_chunk = sentence
            current_size = sentence_size
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_size += sentence_size
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(create_chunk(current_chunk))
    
    return chunks
```

#### 2.3.2 의미적 유사도 기반 청킹
```python
def semantic_similarity_chunking(text: str, similarity_threshold: float = 0.7) -> List[Dict]:
    """
    의미적 유사도를 기반한 청킹
    
    청킹 규칙:
    1. 문장 간 의미적 유사도 계산
    2. 유사도가 높은 문장들을 그룹화
    3. 그룹 내에서 최적 크기로 조정
    4. 의미적 일관성 보장
    """
    
    # 문장 분할
    sentences = split_legal_sentences(text)
    
    # 의미적 유사도 계산
    similarities = calculate_sentence_similarities(sentences)
    
    # 유사도 기반 그룹화
    groups = group_by_similarity(sentences, similarities, similarity_threshold)
    
    # 그룹을 청크로 변환
    chunks = []
    for group in groups:
        chunk_content = " ".join(group)
        chunk = {
            'type': 'semantic_group',
            'content': chunk_content,
            'group_size': len(group),
            'coherence_score': calculate_coherence_score(group),
            'size': len(chunk_content)
        }
        chunks.append(chunk)
    
    return chunks
```

---

## 📏 청킹 크기 최적화

### 3.1 실제 데이터 기반 청킹 크기 분포

#### 3.1.1 법령 데이터 청킹 크기 분석
- **조문 단위**: 200-800자 (평균 400자)
- **항목 단위**: 100-400자 (평균 200자)
- **부칙 단위**: 300-1500자 (평균 600자)

#### 3.1.2 판례 데이터 청킹 크기 분석
- **사건 기본정보**: 100-300자 (평균 150자)
- **사건명 (쟁점)**: 50-200자 (평균 100자)
- **긴 사건명**: 200-500자 (평균 300자)

#### 3.1.3 헌재결정례 청킹 크기 분석
- **기본정보**: 100-200자 (평균 150자)
- **심판대상조문**: 50-300자 (평균 150자)
- **판시사항**: 100-500자 (평균 250자)
- **결정요지**: 200-1000자 (평균 400자)

#### 3.1.4 법령해석례 청킹 크기 분석
- **기본정보**: 100-300자 (평균 200자)
- **질의요지**: 200-800자 (평균 400자)
- **회답**: 100-500자 (평균 250자)
- **해석이유**: 500-3000자 (평균 1200자)

#### 3.1.5 권장 청킹 크기 분포
- **소형 청크 (100-300자)**: 사건 기본정보, 조문 제목
- **중형 청크 (300-800자)**: 조문 본문, 판시사항, 질의요지
- **대형 청크 (800-2000자)**: 해석이유, 긴 조문, 부칙

### 3.2 동적 크기 조정

#### 3.2.1 내용 기반 크기 조정
```python
def dynamic_size_adjustment(text: str, base_size: int = 800) -> int:
    """
    텍스트 내용에 따른 동적 크기 조정
    
    조정 요소:
    1. 법률 용어 밀도
    2. 문장 복잡도
    3. 의미적 밀도
    4. 구조적 복잡도
    """
    
    # 법률 용어 밀도 계산
    legal_term_density = calculate_legal_term_density(text)
    
    # 문장 복잡도 계산
    sentence_complexity = calculate_sentence_complexity(text)
    
    # 의미적 밀도 계산
    semantic_density = calculate_semantic_density(text)
    
    # 크기 조정 계수 계산
    adjustment_factor = (
        legal_term_density * 0.3 +
        sentence_complexity * 0.3 +
        semantic_density * 0.4
    )
    
    # 최종 크기 계산
    adjusted_size = int(base_size * adjustment_factor)
    
    # 최소/최대 크기 제한
    return max(200, min(3000, adjusted_size))
```

#### 3.2.2 오버랩 전략

##### A. 법령 데이터 오버랩 (10-15%)
```python
def law_overlap_strategy(chunk_size: int) -> int:
    """
    법령 데이터 오버랩 전략
    
    오버랩 요소:
    1. 조문 간 연관성
    2. 법리 추론 연속성
    3. 용어 정의 일관성
    """
    return int(chunk_size * 0.12)  # 12% 오버랩
```

##### B. 판례 데이터 오버랩 (15-20%)
```python
def precedent_overlap_strategy(chunk_size: int) -> int:
    """
    판례 데이터 오버랩 전략
    
    오버랩 요소:
    1. 법리 추론 연속성
    2. 판례 인용 맥락
    3. 쟁점 간 연관성
    """
    return int(chunk_size * 0.18)  # 18% 오버랩
```

---

## 🔍 품질 보장 전략

### 4.1 청킹 품질 검증

#### 4.1.1 구조적 완성도 검증
```python
def validate_structural_completeness(chunk: Dict) -> bool:
    """
    청킹의 구조적 완성도 검증
    
    검증 요소:
    1. 문법적 완성성
    2. 의미적 일관성
    3. 법률 구조 준수
    4. 용어 정확성
    """
    
    # 문법적 완성성 검증
    if not is_grammatically_complete(chunk['content']):
        return False
    
    # 의미적 일관성 검증
    if not has_semantic_coherence(chunk['content']):
        return False
    
    # 법률 구조 준수 검증
    if not follows_legal_structure(chunk['content']):
        return False
    
    # 용어 정확성 검증
    if not has_accurate_legal_terms(chunk['content']):
        return False
    
    return True
```

#### 4.1.2 의미적 품질 검증
```python
def validate_semantic_quality(chunk: Dict) -> float:
    """
    청킹의 의미적 품질 검증
    
    품질 지표:
    1. 의미적 일관성 (0.0-1.0)
    2. 법률 용어 포함도 (0.0-1.0)
    3. 맥락 보존도 (0.0-1.0)
    4. 검색 적합성 (0.0-1.0)
    """
    
    # 의미적 일관성 점수
    coherence_score = calculate_semantic_coherence(chunk['content'])
    
    # 법률 용어 포함도 점수
    term_coverage_score = calculate_legal_term_coverage(chunk['content'])
    
    # 맥락 보존도 점수
    context_preservation_score = calculate_context_preservation(chunk['content'])
    
    # 검색 적합성 점수
    searchability_score = calculate_searchability(chunk['content'])
    
    # 종합 품질 점수
    quality_score = (
        coherence_score * 0.3 +
        term_coverage_score * 0.25 +
        context_preservation_score * 0.25 +
        searchability_score * 0.2
    )
    
    return quality_score
```

### 4.2 청킹 통계 및 모니터링

#### 4.2.1 청킹 통계 생성
```python
def generate_chunking_statistics(chunks: List[Dict]) -> Dict:
    """
    청킹 통계 생성
    
    통계 항목:
    1. 기본 통계 (개수, 크기 분포)
    2. 품질 통계 (품질 점수 분포)
    3. 효율성 통계 (처리 시간, 메모리 사용량)
    4. 검색 성능 통계 (검색 정확도, 응답 시간)
    """
    
    # 기본 통계
    basic_stats = {
        'total_chunks': len(chunks),
        'avg_chunk_size': np.mean([c['size'] for c in chunks]),
        'min_chunk_size': min([c['size'] for c in chunks]),
        'max_chunk_size': max([c['size'] for c in chunks]),
        'size_std': np.std([c['size'] for c in chunks])
    }
    
    # 품질 통계
    quality_scores = [validate_semantic_quality(c) for c in chunks]
    quality_stats = {
        'avg_quality_score': np.mean(quality_scores),
        'min_quality_score': min(quality_scores),
        'max_quality_score': max(quality_scores),
        'quality_std': np.std(quality_scores)
    }
    
    # 분포 통계
    size_distribution = get_size_distribution(chunks)
    quality_distribution = get_quality_distribution(chunks)
    
    return {
        'basic_stats': basic_stats,
        'quality_stats': quality_stats,
        'size_distribution': size_distribution,
        'quality_distribution': quality_distribution
    }
```

---

## 🎯 청킹 전략 선택 가이드

### 5.1 데이터 유형별 권장 전략

#### 5.1.1 법령 데이터
- **주 전략**: 조문 중심 청킹
- **보조 전략**: 항목별 청킹
- **크기**: 500-1500자
- **오버랩**: 10-15%

#### 5.1.2 판례 데이터
- **주 전략**: 사건별 청킹
- **보조 전략**: 법리 중심 청킹
- **크기**: 800-2000자
- **오버랩**: 15-20%

#### 5.1.3 헌재결정례
- **주 전략**: 쟁점별 청킹
- **보조 전략**: 의미적 청킹
- **크기**: 1000-2500자
- **오버랩**: 20-25%

### 5.2 사용 목적별 권장 전략

#### 5.2.1 정확한 검색 (Precise Search)
- **전략**: 소형 청크 + 의미적 청킹
- **크기**: 200-500자
- **특징**: 높은 정확도, 낮은 맥락

#### 5.2.2 맥락 보존 검색 (Contextual Search)
- **전략**: 중형 청크 + 구조적 청킹
- **크기**: 500-1200자
- **특징**: 균형잡힌 정확도와 맥락

#### 5.2.3 종합적 이해 (Comprehensive Understanding)
- **전략**: 대형 청크 + 계층적 청킹
- **크기**: 1200-3000자
- **특징**: 높은 맥락, 낮은 정확도

---

## 📊 성능 지표 및 평가

### 6.1 실제 데이터 기반 청킹 품질 지표

#### 6.1.1 예상 청킹 통계 (실제 데이터 기준)
- **총 청크 수**: 약 15,000-20,000개
  - 법령 데이터: 2,000-3,000개 청크
  - 판례 데이터: 8,000-10,000개 청크
  - 헌재결정례: 3,000-4,000개 청크
  - 법령해석례: 2,000-3,000개 청크

#### 6.1.2 구조적 품질 목표
- **문법적 완성성**: 95% 이상
- **의미적 일관성**: 90% 이상
- **법률 구조 준수**: 85% 이상
- **메타데이터 완성도**: 98% 이상

#### 6.1.3 검색 성능 목표
- **검색 정확도**: 80% 이상
- **응답 시간**: 1초 이내
- **컨텍스트 품질**: 75% 이상
- **관련성 점수**: 0.7 이상

### 6.2 실제 데이터 기반 최적화 목표

#### 6.2.1 청킹 효율성
- **처리 속도**: 500청크/분 이상 (실제 데이터 크기 고려)
- **메모리 사용량**: 4GB 이하 (7,699개 판례 + 2,000개 헌재결정례)
- **저장 공간**: 5GB 이하 (압축 기준)

#### 6.2.2 검색 성능 개선 목표
- **정확도 향상**: 15-20%
- **응답 속도 향상**: 30-40%
- **컨텍스트 품질 향상**: 25-30%
- **법률 용어 매칭 정확도**: 90% 이상

---

## 🔧 실제 데이터 기반 구현 가이드

### 7.1 데이터 처리 파이프라인

#### 7.1.1 법령 데이터 처리 순서
```python
def process_law_data():
    """
    법령 데이터 처리 파이프라인
    """
    # 1. 법령 파일 로드 (20개 파일)
    law_files = glob.glob('data/raw/laws/*.json')
    
    for law_file in law_files:
        # 2. JSON 데이터 로드
        with open(law_file, 'r', encoding='utf-8') as f:
            law_data = json.load(f)
        
        # 3. 조문 중심 청킹
        chunks = law_article_chunking(law_data)
        
        # 4. 청크 저장
        save_chunks(chunks, f'data/processed/laws/{law_data["basic_info"]["name"]}_chunks.json')
```

#### 7.1.2 판례 데이터 처리 순서
```python
def process_precedent_data():
    """
    판례 데이터 처리 파이프라인 (90개 배치 파일)
    """
    # 1. 판례 배치 파일 로드
    precedent_files = glob.glob('data/raw/precedents/batch_*.json')
    
    all_chunks = []
    for batch_file in precedent_files:
        # 2. 배치 데이터 로드
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # 3. 각 판례별 청킹
        for precedent in batch_data['precedents']:
            chunks = precedent_case_chunking(precedent)
            all_chunks.extend(chunks)
    
    # 4. 통합 청크 저장
    save_chunks(all_chunks, 'data/processed/precedents/all_precedent_chunks.json')
```

#### 7.1.3 헌재결정례 데이터 처리 순서
```python
def process_constitutional_data():
    """
    헌재결정례 데이터 처리 파이프라인 (40개 배치 파일)
    """
    constitutional_files = glob.glob('data/raw/constitutional_decisions/batch_*.json')
    
    all_chunks = []
    for batch_file in constitutional_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        for decision in batch_data['data']:
            chunks = constitutional_issue_chunking(decision)
            all_chunks.extend(chunks)
    
    save_chunks(all_chunks, 'data/processed/constitutional/all_constitutional_chunks.json')
```

#### 7.1.4 법령해석례 데이터 처리 순서
```python
def process_interpretation_data():
    """
    법령해석례 데이터 처리 파이프라인 (4개 배치 파일)
    """
    interpretation_files = glob.glob('data/raw/legal_interpretations/batches/batch_*.json')
    
    all_chunks = []
    for batch_file in interpretation_files:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        for interpretation in batch_data['interpretations']:
            chunks = legal_interpretation_chunking(interpretation)
            all_chunks.extend(chunks)
    
    save_chunks(all_chunks, 'data/processed/interpretations/all_interpretation_chunks.json')
```

### 7.2 메모리 관리 전략

#### 7.2.1 배치 처리 방식
- **법령 데이터**: 파일별 순차 처리 (메모리 사용량: 100MB 이하)
- **판례 데이터**: 배치별 처리 (메모리 사용량: 200MB 이하)
- **헌재결정례**: 배치별 처리 (메모리 사용량: 150MB 이하)
- **법령해석례**: 배치별 처리 (메모리 사용량: 100MB 이하)

#### 7.2.2 청크 캐싱 전략
```python
class ChunkCache:
    """
    청크 캐싱 관리 클래스
    """
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_chunk(self, chunk_id):
        if chunk_id in self.cache:
            self.access_count[chunk_id] += 1
            return self.cache[chunk_id]
        return None
    
    def add_chunk(self, chunk_id, chunk_data):
        if len(self.cache) >= self.max_size:
            # LRU 방식으로 오래된 청크 제거
            self._remove_oldest()
        
        self.cache[chunk_id] = chunk_data
        self.access_count[chunk_id] = 1
```

### 7.3 성능 최적화

#### 7.3.1 병렬 처리
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_data_parallel():
    """
    병렬 처리를 통한 성능 최적화
    """
    # CPU 코어 수에 따른 워커 수 설정
    num_workers = min(mp.cpu_count(), 4)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 법령 데이터 병렬 처리
        law_files = glob.glob('data/raw/laws/*.json')
        law_futures = [executor.submit(process_single_law, f) for f in law_files]
        
        # 판례 데이터 병렬 처리
        precedent_files = glob.glob('data/raw/precedents/batch_*.json')
        precedent_futures = [executor.submit(process_single_precedent_batch, f) for f in precedent_files]
        
        # 결과 수집
        law_results = [f.result() for f in law_futures]
        precedent_results = [f.result() for f in precedent_futures]
```

#### 7.3.2 인덱싱 전략
```python
def build_chunk_index():
    """
    청크 인덱스 구축
    """
    index = {
        'by_type': {},  # 청크 타입별 인덱스
        'by_law': {},   # 법령별 인덱스
        'by_case': {},  # 사건별 인덱스
        'by_keyword': {} # 키워드별 인덱스
    }
    
    # 모든 청크 파일 로드 및 인덱싱
    chunk_files = glob.glob('data/processed/*/all_*_chunks.json')
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        for chunk in chunks:
            # 타입별 인덱스
            chunk_type = chunk.get('type', 'unknown')
            if chunk_type not in index['by_type']:
                index['by_type'][chunk_type] = []
            index['by_type'][chunk_type].append(chunk['id'])
            
            # 법령별 인덱스
            if 'law_name' in chunk:
                law_name = chunk['law_name']
                if law_name not in index['by_law']:
                    index['by_law'][law_name] = []
                index['by_law'][law_name].append(chunk['id'])
    
    # 인덱스 저장
    with open('data/processed/chunk_index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
```

### 7.2 품질 관리

#### 7.2.1 검증 프로세스
- **자동 검증**: 청킹 품질 자동 검증
- **수동 검토**: 법률 전문가 검토
- **지속적 모니터링**: 실시간 품질 모니터링

#### 7.2.2 개선 프로세스
- **피드백 수집**: 사용자 피드백 수집
- **성능 분석**: 정기적 성능 분석
- **전략 조정**: 데이터 특성에 따른 전략 조정

---

## 📚 참고 자료

### 8.1 관련 논문
- "Legal Text Chunking for Information Retrieval" (2023)
- "Semantic Chunking for Legal Documents" (2022)
- "Multi-Level Text Segmentation for Legal AI" (2023)

### 8.2 기술 문서
- Sentence-BERT Documentation
- FAISS Indexing Guide
- Legal NLP Best Practices

### 8.3 도구 및 라이브러리
- **텍스트 처리**: spaCy, NLTK
- **의미적 분석**: Sentence-Transformers
- **벡터 검색**: FAISS, ChromaDB
- **품질 평가**: BLEU, ROUGE, BERTScore

---

*본 문서는 LawFirmAI 프로젝트의 텍스트 청킹 전략을 정의하며, 프로젝트 진행에 따라 지속적으로 업데이트됩니다.*
