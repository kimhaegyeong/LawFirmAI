<!-- 0f9c520e-cde8-4420-876f-029ffb4531a0 c9af841e-e093-4089-b142-8e68fbbd636d -->
# Data Deduplication and Quality Improvement Plan

## Overview

Implement a comprehensive data quality improvement system with 4 phases: (1) Parsing quality improvement with validation layers, (2) Advanced duplicate detection, (3) Database schema enhancements with data migration, (4) Automated cleanup and monitoring systems.

## Phase 1: Parsing Quality Improvement System (Week 1-2)

### 1.1 Create Data Quality Validator

**File**: `scripts/data_processing/quality/data_quality_validator.py` (new)

Implement quality validation system:

- `DataQualityValidator` class with methods:
- `validate_parsing_quality()`: Check article count consistency, title extraction accuracy, article number continuity
- `calculate_quality_score()`: Compute quality score (0.0-1.0) based on multiple metrics
- `suggest_improvements()`: Generate improvement suggestions for low-quality data
- Quality metrics:
- Article count consistency (compare with expected structure)
- Title extraction completeness
- Article number sequence validation
- Legal structure completeness check

### 1.2 Enhance ML Parser with Validation Layer

**File**: `scripts/ml_training/model_training/ml_enhanced_parser.py`

Improve existing ML parser:

- Add `validate_parsed_result()` method to check parsing output
- Implement `fix_missing_articles()` to recover missing articles
- Add `remove_duplicate_articles()` to eliminate duplicates within same law
- Integrate quality validator to tag each parsed result with quality score
- Lower ML threshold from 0.5 to 0.4 for better recall, then validate results

### 1.3 Create Hybrid Parser with Auto-Correction

**File**: `scripts/data_processing/quality/hybrid_parser.py` (new)

Implement hybrid parsing approach:

- `HybridArticleParser` class that:
- Runs both rule-based and ML parsers
- Compares results and selects best output based on quality metrics
- Applies post-processing corrections
- Falls back to manual review for very low quality scores (<0.6)
- Integration with existing `LawPreprocessor` in `scripts/data_processing/preprocessing/preprocess_laws.py`

### 1.4 Update Preprocessing Pipeline

**File**: `scripts/data_processing/preprocessing/preprocess_laws.py`

Modify `_process_single_law()` method:

- Replace direct article parser call with hybrid parser
- Add quality validation step after parsing
- Store quality score in processed data
- Flag low-quality results for review

## Phase 2: Advanced Duplicate Detection System (Week 3-4)

### 2.1 Create Multi-Level Duplicate Detector

**File**: `scripts/data_processing/quality/duplicate_detector.py` (new)

Implement `AdvancedDuplicateDetector` class:

- `detect_file_level_duplicates()`: Compare file hash, size, name similarity
- `detect_content_level_duplicates()`: Use TF-IDF + Cosine similarity for content comparison
- `detect_semantic_duplicates()`: Use existing vector embeddings to find semantic duplicates
- Threshold: 0.95+ similarity = exact duplicate, 0.85-0.95 = near duplicate, <0.85 = unique

### 2.2 Create Intelligent Duplicate Resolver

**File**: `scripts/data_processing/quality/duplicate_resolver.py` (new)

Implement `IntelligentDuplicateResolver` class:

- `resolve_duplicates()`: Select best version based on quality score
- `merge_metadata()`: Combine metadata from duplicate versions
- `create_version_history()`: Track different versions of same law
- Resolution strategy: Keep highest quality score version, archive others

### 2.3 Create Duplicate Detection Pipeline

**File**: `scripts/data_processing/run_duplicate_detection.py` (new)

Standalone script to:

- Scan all processed data for duplicates
- Generate duplicate groups report
- Optionally auto-resolve duplicates based on quality scores
- Export results to `reports/duplicate_detection_report.json`

## Phase 3: Database Schema Enhancement (Week 4-5)

### 3.1 Create Schema Migration Script

**File**: `scripts/database/migrate_schema_v2.py` (new)

Implement database migration:

- Add new columns to `assembly_laws`:
- `law_name_hash TEXT UNIQUE`: MD5 hash of normalized law name
- `content_hash TEXT UNIQUE`: SHA256 hash of full content
- `quality_score REAL DEFAULT 0.0`: Parsing quality score
- `duplicate_group_id TEXT`: ID linking duplicate laws
- `is_primary_version BOOLEAN DEFAULT TRUE`: Primary version flag
- `version_number INTEGER DEFAULT 1`: Version number
- Create new `duplicate_groups` table:
- `group_id TEXT PRIMARY KEY`
- `group_type TEXT`: 'file', 'content', or 'semantic'
- `primary_law_id TEXT`
- `duplicate_law_ids TEXT`: JSON array
- `resolution_strategy TEXT`
- `created_at TIMESTAMP`
- Add constraints and indices for performance

### 3.2 Migrate Existing Data

**File**: `scripts/database/migrate_existing_data.py` (new)

Data migration script:

- Calculate hashes for all existing laws
- Compute quality scores retroactively
- Detect and group duplicates in existing data
- Update `duplicate_group_id` and `is_primary_version` fields
- Backup original data before migration

### 3.3 Update Database Manager

**File**: `source/data/database.py`

Update `_create_tables()` method:

- Apply schema changes from migration
- Add new indices for efficient duplicate lookup
- Add constraints for data integrity

### 3.4 Update Import Script

**File**: `scripts/data_processing/utilities/import_laws_to_db.py`

Modify `AssemblyLawImporter` class:

- Add duplicate check before insert using content_hash
- Update existing record if content_hash matches but quality_score is higher
- Auto-assign `duplicate_group_id` if duplicate detected
- Log duplicate handling actions

## Phase 4: Automated Cleanup and Monitoring (Week 5-6)

### 4.1 Create Automated Data Cleaner

**File**: `scripts/data_processing/quality/automated_cleaner.py` (new)

Implement `AutomatedDataCleaner` class:

- `run_daily_cleanup()`: Daily cleanup routine
- Detect new duplicates in recent data
- Recalculate quality scores
- Archive low-quality data (<0.5 score)
- Merge duplicate data
- `run_weekly_optimization()`: Weekly optimization
- Rebuild database indices
- Rebuild vector indices
- Update statistics
- Clean up temporary files
- `generate_cleanup_report()`: Generate detailed cleanup report

### 4.2 Create Real-Time Quality Monitor

**File**: `scripts/data_processing/quality/quality_monitor.py` (new)

Implement `RealTimeQualityMonitor` class:

- `monitor_data_quality()`: Monitor quality metrics
- Track average quality score trends
- Track duplicate rate
- Detect anomalies in parsing results
- `alert_on_quality_drop()`: Send alerts when quality drops below threshold
- `generate_quality_dashboard()`: Export metrics for visualization

### 4.3 Create Scheduled Task Scripts

**Files**:

- `scripts/data_processing/quality/daily_cleanup.py` (new)
- `scripts/data_processing/quality/weekly_optimization.py` (new)

Create executable scripts for:

- Daily cleanup task (can be scheduled via cron/Task Scheduler)
- Weekly optimization task
- Include logging and error handling

### 4.4 Create Quality Reporting Dashboard

**File**: `scripts/data_processing/quality/generate_quality_report.py` (new)

Generate comprehensive quality report:

- Overall quality statistics
- Duplicate detection summary
- Parsing quality trends
- Data cleanliness metrics
- Export to `reports/data_quality_report.html` and `.json`

## Integration and Testing

### Update Auto Pipeline Orchestrator

**File**: `scripts/data_processing/auto_pipeline_orchestrator.py`

Integrate quality checks:

- Add quality validation step after preprocessing
- Add duplicate detection before database import
- Log quality metrics in pipeline report

### Create End-to-End Test

**File**: `scripts/tests/test_quality_improvement.py` (new)

Test complete workflow:

- Test parsing quality improvement
- Test duplicate detection accuracy
- Test database migration
- Test automated cleanup
- Validate data integrity after all operations

## Documentation Updates

### Update Development Documentation

**Files**:

- `docs/06_models_performance/data_quality_improvement_guide.md` (new)
- `README.md`

Document:

- Quality improvement system architecture
- Duplicate detection algorithm details
- Database schema changes
- How to run quality checks
- How to interpret quality reports

## Expected Outcomes

- Parsing quality: 93.1% problem rate → <5% problem rate
- Duplicate data: 20-30% reduction in database size
- Search accuracy: Improved due to higher quality data
- System performance: Better due to reduced data volume
- Data integrity: Guaranteed through constraints and validation

### To-dos

- [x] Create DataQualityValidator class with validation methods and quality scoring
- [x] Enhance ML parser with validation layer and auto-correction features
- [x] Create HybridArticleParser that combines rule-based and ML parsing
- [x] Update preprocessing pipeline to use hybrid parser with quality validation
- [x] Create AdvancedDuplicateDetector with multi-level detection algorithms
- [x] Create IntelligentDuplicateResolver to handle duplicate resolution
- [x] Create standalone duplicate detection pipeline script
- [x] Create database schema migration script with new columns and tables
- [x] Create data migration script to update existing records
- [x] Update DatabaseManager with new schema and constraints
- [x] Update import script with duplicate checking logic
- [x] Create AutomatedDataCleaner with daily and weekly routines
- [x] Create scheduled task scripts for daily/weekly execution