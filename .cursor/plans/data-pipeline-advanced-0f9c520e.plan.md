<!-- 0f9c520e-cde8-4420-876f-029ffb4531a0 f8ad2c59-27b0-4a0e-9aaf-2f01af48a414 -->
# Precedent Data Incremental Processing Pipeline

## Overview

Implement a complete automated pipeline for precedent data processing with category-based handling (civil, criminal, family), following the same architecture as the law_only pipeline.

## Implementation Steps

### 1. Update AutoDataDetector for Precedent Categories

**File**: `scripts/data_processing/auto_data_detector.py`

- Add precedent category patterns to `data_patterns`:
  ```python
  'precedent_civil': {
      'file_pattern': r'precedent_civil_page_\d+_\d+_\d+\.json',
      'directory_pattern': r'\d{8}/civil',
      'metadata_key': 'category',
      'expected_value': 'civil'
  },
  'precedent_criminal': {...},
  'precedent_family': {...}
  ```

- Update `base_paths` to include:
  ```python
  'precedent_civil': 'data/raw/assembly/precedent',
  'precedent_criminal': 'data/raw/assembly/precedent',
  'precedent_family': 'data/raw/assembly/precedent'
  ```

- Modify `detect_new_data_sources()` to handle nested category folders (date/category structure)
- Update `classify_data_type()` to recognize precedent patterns and metadata

### 2. Create Precedent Preprocessor

**New File**: `scripts/data_processing/precedent_preprocessor.py`

Create a dedicated preprocessor for precedent data structure:

- Parse precedent-specific fields:
  - `case_name`, `case_number`, `decision_date`
  - `field`, `court`, `detail_url`
  - `structured_content` (case_info, legal_sections, parties)

- Extract legal sections:
  - 판시사항 (points at issue)
  - 판결요지 (decision summary)
  - 참조조문 (referenced statutes)
  - 참조판례 (referenced cases)
  - 주문 (disposition)
  - 이유 (reasoning)

- Output format: JSON with processed precedent data
- Support category-specific processing (civil, criminal, family)

### 3. Create Incremental Precedent Preprocessor

**New File**: `scripts/data_processing/incremental_precedent_preprocessor.py`

Similar to `incremental_preprocessor.py` but for precedents:

- Initialize with precedent-specific paths
- Support category parameter (civil/criminal/family)
- Use `PrecedentPreprocessor` for data processing
- Handle precedent data structure in `_process_assembly_precedent_data()`
- Generate output: `data/processed/assembly/precedent/{category}/{date}/ml_enhanced_precedent_{category}_page_*.json`

### 4. Update Database Schema

**File**: `source/data/database.py`

Add new tables for precedent data:

```python
CREATE TABLE precedent_cases (
    case_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,  -- civil, criminal, family
    case_name TEXT NOT NULL,
    case_number TEXT NOT NULL,
    decision_date TEXT,
    field TEXT,
    court TEXT,
    detail_url TEXT,
    full_text TEXT,
    searchable_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE precedent_sections (
    section_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    section_type TEXT NOT NULL,  -- 판시사항, 판결요지, etc.
    section_content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES precedent_cases(case_id)
);

CREATE TABLE precedent_parties (
    party_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    party_type TEXT NOT NULL,  -- plaintiff, defendant
    party_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (case_id) REFERENCES precedent_cases(case_id)
);
```

Add indices:

```python
CREATE INDEX idx_precedent_cases_category ON precedent_cases(category);
CREATE INDEX idx_precedent_cases_date ON precedent_cases(decision_date);
CREATE INDEX idx_precedent_sections_case_id ON precedent_sections(case_id);
```

Add FTS5 tables for full-text search:

```python
CREATE VIRTUAL TABLE fts_precedent_cases USING fts5(...);
CREATE VIRTUAL TABLE fts_precedent_sections USING fts5(...);
```

### 5. Create Precedent DB Importer

**New File**: `scripts/data_processing/utilities/import_precedents_to_db.py`

Based on `import_laws_to_db.py`:

- Import processed precedent JSON files
- Support incremental mode (check existing cases, update if needed)
- Handle category-specific imports
- Populate precedent_cases, precedent_sections, precedent_parties tables
- Update FTS indices
- Generate import statistics and reports

### 6. Create Incremental Vector Builder for Precedents

**New File**: `scripts/ml_training/vector_embedding/incremental_precedent_vector_builder.py`

Similar to `incremental_vector_builder.py`:

- Use same ko-sroberta-multitask model (768 dimensions)
- Process precedent sections as chunks
- Separate FAISS index: `data/embeddings/ml_enhanced_ko_sroberta_precedents/`
- Support category-specific embeddings
- Track embedding status in processed_files table

### 7. Update Pipeline Orchestrator

**File**: `scripts/data_processing/auto_pipeline_orchestrator.py`

Add precedent pipeline support:

- Add precedent-specific configuration
- Support `--data-type precedent_civil`, `precedent_criminal`, etc.
- Integrate all precedent processing steps:

  1. Precedent data detection
  2. Precedent preprocessing
  3. Precedent vector embedding
  4. Precedent DB import

### 8. Configuration Updates

**File**: `config/pipeline_config.yaml`

Add precedent configurations:

```yaml
data_types:
  precedent_civil:
    raw_pattern: "precedent_civil_page_\\d+_\\d+_\\d+_\\d+\\.json"
    processed_subdir: "precedent/civil"
    category: "civil"
  precedent_criminal:
    raw_pattern: "precedent_criminal_page_\\d+_\\d+_\\d+_\\d+\\.json"
    processed_subdir: "precedent/criminal"
    category: "criminal"
  precedent_family:
    raw_pattern: "precedent_family_page_\\d+_\\d+_\\d+_\\d+\\.json"
    processed_subdir: "precedent/family"
    category: "family"

paths:
  precedent_embeddings: "data/embeddings/ml_enhanced_ko_sroberta_precedents"
```

### 9. Documentation

Update documentation files:

- `docs/03_data_processing/incremental_pipeline_guide.md`: Add precedent pipeline usage
- `docs/10_technical_reference/database_schema.md`: Document precedent tables
- `docs/01_project_overview/project_overview.md`: Add precedent pipeline statistics
- `README.md`: Update with precedent processing capabilities

### 10. Testing

Create test script: `scripts/tests/test_precedent_pipeline.py`

Test scenarios:

- Precedent data detection by category
- Precedent preprocessing accuracy
- Database import with incremental mode
- Vector embedding generation
- Full pipeline execution for each category

## File Structure

```
data/
├── raw/assembly/precedent/
│   └── {date}/
│       ├── civil/*.json
│       ├── criminal/*.json
│       └── family/*.json
├── processed/assembly/precedent/
│   └── {category}/{date}/ml_enhanced_*.json
└── embeddings/
    └── ml_enhanced_ko_sroberta_precedents/

scripts/
├── data_processing/
│   ├── precedent_preprocessor.py (new)
│   ├── incremental_precedent_preprocessor.py (new)
│   └── utilities/import_precedents_to_db.py (new)
└── ml_training/vector_embedding/
    └── incremental_precedent_vector_builder.py (new)
```

## Key Considerations

- Maintain consistency with law_only pipeline architecture
- Reuse checkpoint and error recovery mechanisms
- Support parallel processing of different categories
- Preserve original precedent metadata and structure
- Enable category-specific querying and filtering

### To-dos

- [ ] Update AutoDataDetector to support precedent categories (civil, criminal, family)
- [ ] Create PrecedentPreprocessor for precedent-specific data structure
- [ ] Create IncrementalPrecedentPreprocessor for category-based processing
- [ ] Add precedent_cases, precedent_sections, precedent_parties tables to database
- [ ] Create import_precedents_to_db.py with incremental mode support
- [ ] Create incremental_precedent_vector_builder.py for precedent embeddings
- [ ] Update AutoPipelineOrchestrator to support precedent pipeline
- [ ] Add precedent configurations to pipeline_config.yaml
- [ ] Update documentation with precedent pipeline information
- [ ] Create test_precedent_pipeline.py for validation