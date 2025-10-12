<!-- 03f53c05-a041-49e2-b281-de0d1533aec2 3081e6ca-f5bd-4a95-8124-7eb1f5db5129 -->
# Law Data Preprocessing Pipeline

## Overview

Create a preprocessing pipeline to transform raw Assembly law data (HTML + text) into clean, structured, searchable format for database storage and vector embedding.

## Current Data Structure

**Raw Law Data (JSON):**

```json
{
  "law_name": "수의사법 시행규칙",
  "law_content": "[시행 2025.10.2.] ... 제1조(목적) ...",
  "content_html": "<html>...</html>",
  "row_number": "7571",
  "category": "법령",
  "law_type": "부령",
  "promulgation_number": "농림축산식품부령 제725호",
  "promulgation_date": "2025.7.1.",
  "enforcement_date": "2025.10.2.",
  "amendment_type": "일부개정",
  "cont_id": "...",
  "cont_sid": "...",
  "detail_url": "...",
  "collected_at": "2025-10-10T..."
}
```

**Data Location:** `data/raw/assembly/law/20251010/`

- 127 page files (law_page_XXX.json)
- Each page contains 10 laws
- Total: ~1,270 laws collected

## Preprocessing Pipeline

### Phase 1: HTML Parser

**File:** `scripts/assembly/parsers/html_parser.py`

**Tasks:**

1. Parse HTML content to extract clean text
2. Remove navigation elements, scripts, styles
3. Extract article structure from HTML
4. Preserve formatting for legal articles

**Implementation:**

```python
from bs4 import BeautifulSoup
import re

class LawHTMLParser:
    def parse_html(self, html_content: str) -> dict:
        """Parse HTML and extract structured content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        return {
            'clean_text': self._extract_clean_text(soup),
            'articles': self._extract_articles(soup),
            'metadata': self._extract_html_metadata(soup)
        }
    
    def _extract_articles(self, soup):
        """Extract individual articles (제1조, 제2조 etc)"""
        # Find article patterns: 제N조(title)
        pass
```

### Phase 2: Article Structure Parser

**File:** `scripts/assembly/parsers/article_parser.py`

**Tasks:**

1. Parse article numbers (제1조, 제2조 etc)
2. Extract article titles (괄호 안 내용)
3. Parse sub-articles (항, 호, 목)
4. Extract article content and structure

**Article Structure:**

```python
{
  'article_number': '제1조',
  'article_title': '목적',
  'article_content': '이 규칙은 ...',
  'sub_articles': [
    {'type': '항', 'number': 1, 'content': '...'},
    {'type': '호', 'number': 1, 'content': '...'}
  ],
  'references': ['수의사법', '같은 법 시행령']
}
```

### Phase 3: Metadata Extractor

**File:** `scripts/assembly/parsers/metadata_extractor.py`

**Tasks:**

1. Extract enforcement date from text
2. Parse amendment history
3. Extract legal references
4. Identify related laws
5. Extract ministry/department info

**Metadata Structure:**

```python
{
  'enforcement_info': {
    'date': '2025.10.2.',
    'text': '[시행 2025.10.2.]'
  },
  'amendment_info': {
    'number': '농림축산식품부령 제725호',
    'date': '2025.7.1.',
    'type': '일부개정'
  },
  'references': [
    {'type': 'parent_law', 'name': '수의사법'},
    {'type': 'enforcement_decree', 'name': '수의사법 시행령'}
  ],
  'ministry': '농림축산식품부'
}
```

### Phase 4: Text Cleaner & Normalizer

**File:** `scripts/assembly/parsers/text_normalizer.py`

**Tasks:**

1. Remove duplicate whitespace
2. Normalize legal terminology
3. Convert special characters
4. Standardize date formats
5. Clean up formatting artifacts

**Normalization Rules:**

```python
{
  'date_formats': ['YYYY.M.D.', 'YYYY년 M월 D일'],
  'article_patterns': ['제N조', '제N항', '제N호'],
  'legal_terms': {
    '같은 법': 'parent_law_reference',
    '이 법': 'self_reference'
  }
}
```

### Phase 5: Searchable Text Generator

**File:** `scripts/assembly/parsers/searchable_text_generator.py`

**Tasks:**

1. Create full-text search field
2. Generate article-level search text
3. Extract keywords and terms
4. Create search-optimized summaries

**Output:**

```python
{
  'full_text': 'Complete law text for full-text search',
  'article_texts': ['제1조 내용', '제2조 내용', ...],
  'keywords': ['수의사', '면허', '시험', ...],
  'summary': 'Law summary (first 500 chars)',
  'search_metadata': {
    'total_articles': 50,
    'total_words': 5000,
    'key_terms': ['면허증', '국가시험', ...]
  }
}
```

### Phase 6: Database Preparation

**File:** `scripts/assembly/preprocess_laws.py`

**Main preprocessing script that:**

1. Loads raw JSON files
2. Applies all parsers
3. Validates processed data
4. Generates clean JSON for database import

**Output Structure:**

```python
{
  'law_id': 'assembly_law_7571',
  'source': 'assembly',
  'law_name': '수의사법 시행규칙',
  'law_type': '부령',
  'category': '법령',
  
  # Original metadata
  'promulgation_number': '농림축산식품부령 제725호',
  'promulgation_date': '2025-07-01',
  'enforcement_date': '2025-10-02',
  'amendment_type': '일부개정',
  
  # Extracted metadata
  'ministry': '농림축산식품부',
  'parent_law': '수의사법',
  'related_laws': ['수의사법 시행령'],
  
  # Parsed content
  'articles': [...],  # Structured articles
  'full_text': '...',  # Clean full text
  'searchable_text': '...',  # Search-optimized text
  'keywords': [...],
  
  # Original data
  'raw_content': '...',  # Original law_content
  'content_html': '...',  # Original HTML
  
  # Processing metadata
  'processed_at': '2025-10-10T...',
  'processing_version': '1.0',
  'data_quality': {
    'article_count': 50,
    'has_html': true,
    'has_articles': true,
    'completeness_score': 0.95
  }
}
```

## Implementation Steps

### Step 1: Create Parser Modules

**Directory:** `scripts/assembly/parsers/`

Files to create:

- `__init__.py`
- `html_parser.py`
- `article_parser.py`
- `metadata_extractor.py`
- `text_normalizer.py`
- `searchable_text_generator.py`

### Step 2: Main Preprocessing Script

**File:** `scripts/assembly/preprocess_laws.py`

```python
#!/usr/bin/env python3
"""
법률 데이터 전처리 스크립트

Usage:
  python preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
  python preprocess_laws.py --validate  # 검증만 수행
"""

import argparse
from pathlib import Path
import json
from parsers import (
    LawHTMLParser,
    ArticleParser,
    MetadataExtractor,
    TextNormalizer,
    SearchableTextGenerator
)

def preprocess_law_file(input_file: Path, output_dir: Path):
    """단일 법률 파일 전처리"""
    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_laws = []
    
    for law in data.get('laws', []):
        # Apply parsers
        html_parsed = html_parser.parse_html(law['content_html'])
        articles = article_parser.parse_articles(law['law_content'])
        metadata = metadata_extractor.extract(law)
        clean_text = text_normalizer.normalize(law['law_content'])
        searchable = search_generator.generate(clean_text, articles)
        
        # Combine results
        processed = {
            'law_id': f"assembly_law_{law['row_number']}",
            'source': 'assembly',
            **law,  # Original fields
            **metadata,  # Extracted metadata
            'articles': articles,
            'full_text': clean_text,
            'searchable_text': searchable['full_text'],
            'keywords': searchable['keywords'],
            'processed_at': datetime.now().isoformat()
        }
        
        processed_laws.append(processed)
    
    # Save processed data
    output_file = output_dir / input_file.name.replace('law_page_', 'processed_law_')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_laws, f, ensure_ascii=False, indent=2)
```

### Step 3: Validation & Quality Check

**File:** `scripts/assembly/validate_processed_laws.py`

**Validation checks:**

1. All required fields present
2. Article numbers sequential
3. No duplicate articles
4. Valid date formats
5. Keyword extraction quality
6. Text completeness

### Step 4: Database Import Script

**File:** `scripts/assembly/import_laws_to_db.py`

**Tasks:**

1. Read processed JSON files
2. Insert into assembly_laws table
3. Create full-text search indices
4. Generate statistics report

## Data Quality Metrics

Track for each law:

- Article count
- Text length
- Keyword count
- Reference count
- Completeness score (0-1)
- Processing errors

## Output Organization

```
data/
├── raw/assembly/law/20251010/          # Original data
├── processed/assembly/law/20251010/    # Processed data
│   ├── processed_law_001.json
│   ├── processed_law_002.json
│   └── ...
├── quality_reports/
│   ├── preprocessing_report_20251010.json
│   └── validation_errors.json
└── statistics/
    └── law_statistics_20251010.json
```

## Testing Strategy

1. Unit tests for each parser
2. Integration test with sample laws
3. Quality validation on full dataset
4. Performance benchmarking

## Success Criteria

- 100% of laws successfully parsed
- 95%+ completeness score
- All articles extracted correctly
- Valid searchable text generated
- Database-ready format

## Next Steps After Preprocessing

1. Import to database (assembly_laws table)
2. Generate vector embeddings
3. Build FAISS index
4. Update hybrid search
5. Integrate with RAG service

### To-dos

- [ ] Create web scraping infrastructure (AssemblyClient, models, logging)
- [ ] Implement law collection script with checkpoint support
- [ ] Implement precedent collection script with checkpoint support
- [ ] Create Assembly-specific database tables and update DatabaseManager
- [ ] Create Assembly data preprocessing pipeline
- [ ] Build FAISS vector index for Assembly data
- [ ] Update vector store to handle Assembly documents
- [ ] Extend hybrid search to query both legacy and Assembly data
- [ ] Update RAG service to include Assembly data in context retrieval
- [ ] Create unit and integration tests for Assembly modules
- [ ] Validate data quality and generate reports
- [ ] Update README and create Assembly-specific documentation