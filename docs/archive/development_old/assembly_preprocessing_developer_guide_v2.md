# Assembly Law Data Preprocessing - Developer Guide v2.0

## Overview
This guide provides comprehensive documentation for developers working with the Assembly Law Data Preprocessing Pipeline v2.0, including the new ProcessingManager system, memory optimization features, and enhanced error handling.

## Architecture Overview

### Core Components
```
preprocess_laws.py (Main Orchestrator)
├── ProcessingManager (State Management)
├── LawPreprocessor (Processing Engine)
├── Parser Modules
│   ├── HTMLParser
│   ├── ArticleParser
│   ├── MetadataExtractor
│   ├── TextNormalizer
│   └── SearchableTextGenerator
└── Memory Management System
```

### Processing Flow
1. **Initialization** → ProcessingManager setup
2. **File Discovery** → Status checking and filtering
3. **Processing** → Parser application and validation
4. **State Update** → Database status tracking
5. **Cleanup** → Memory management and cleanup

## ProcessingManager Class

### Purpose
Manages processing state using SQLite database, provides resume capability, and tracks processing metrics.

### Key Methods

#### `__init__(output_dir: Path)`
Initializes ProcessingManager with SQLite database connection.

```python
processing_manager = ProcessingManager(Path("data/processed/assembly/law/20251012"))
```

#### `is_processed(input_file: Path) -> bool`
Checks if a file has been successfully processed.

```python
if processing_manager.is_processed(law_file):
    logger.info(f"Skipping {law_file.name} - already processed")
    continue
```

#### `mark_processing(input_file: Path)`
Marks a file as currently being processed.

```python
processing_manager.mark_processing(input_file)
```

#### `mark_completed(input_file: Path, laws_processed: int, processing_time: float)`
Marks a file as successfully completed.

```python
processing_manager.mark_completed(
    input_file, 
    laws_count, 
    processing_time_seconds
)
```

#### `mark_failed(input_file: Path, error_message: str)`
Marks a file as failed with error details.

```python
processing_manager.mark_failed(input_file, str(exception))
```

#### `get_summary() -> Dict[str, Any]`
Returns comprehensive processing statistics.

```python
summary = processing_manager.get_summary()
print(f"Total files: {summary['total_files']}")
print(f"Completed: {summary['by_status']['completed']['count']}")
```

#### `get_failed_files() -> List[Dict[str, Any]]`
Returns list of failed files with error details.

```python
failed_files = processing_manager.get_failed_files()
for failed in failed_files:
    print(f"Failed: {failed['input_file']} - {failed['error_message']}")
```

#### `reset_failed() -> int`
Resets all failed files for retry.

```python
count = processing_manager.reset_failed()
print(f"Reset {count} failed files for retry")
```

### Database Schema

```sql
CREATE TABLE processing_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_file TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    status TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_checksum TEXT,
    file_size INTEGER,
    laws_processed INTEGER DEFAULT 0,
    processing_time_seconds REAL,
    error_message TEXT,
    UNIQUE(input_file, output_dir)
);

CREATE INDEX idx_status ON processing_status(status);
CREATE INDEX idx_input_file ON processing_status(input_file);
```

## Memory Management System

### Memory Monitoring Functions

#### `get_memory_usage() -> Dict[str, float]`
Returns current process memory usage.

```python
memory = get_memory_usage()
print(f"RSS: {memory['rss_mb']:.1f}MB")
print(f"VMS: {memory['vms_mb']:.1f}MB")
```

#### `get_system_memory_info() -> Dict[str, float]`
Returns system-wide memory information.

```python
system_memory = get_system_memory_info()
print(f"Total: {system_memory['total_gb']:.1f}GB")
print(f"Available: {system_memory['available_gb']:.1f}GB")
print(f"Used: {system_memory['percent']:.1f}%")
```

#### `check_memory_safety(threshold_percent: float) -> bool`
Checks if system memory usage is within safe limits.

```python
if not check_memory_safety(85.0):
    logger.critical("System memory usage too high!")
    sys.exit(1)
```

#### `force_exit_on_memory_limit(threshold_percent: float)`
Forces program exit if memory usage exceeds threshold.

```python
force_exit_on_memory_limit(90.0)  # Exit if >90% memory used
```

### Memory Optimization Functions

#### `aggressive_garbage_collection()`
Performs multiple garbage collection cycles.

```python
aggressive_garbage_collection()
```

#### `cleanup_large_objects(*objects)`
Explicitly deletes large objects and forces garbage collection.

```python
cleanup_large_objects(processed_law, raw_data)
```

#### `monitor_memory_and_cleanup(threshold_mb: float)`
Monitors memory and performs cleanup if needed.

```python
monitor_memory_and_cleanup(1000.0)  # Cleanup if >1GB
```

## Enhanced LawPreprocessor Class

### Constructor
```python
def __init__(
    self, 
    enable_legal_analysis: bool = True, 
    max_memory_mb: int = 2048, 
    processing_manager: Optional['ProcessingManager'] = None
):
```

### Key Methods

#### `preprocess_law_file(input_file: Path, output_dir: Path) -> Dict[str, Any]`
Enhanced file processing with state management.

```python
# Mark as processing
if self.processing_manager:
    self.processing_manager.mark_processing(input_file)

try:
    # Process file
    result = self._process_file_content(input_file)
    
    # Mark as completed
    if self.processing_manager:
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_manager.mark_completed(
            input_file, 
            result['laws_processed'], 
            processing_time
        )
    
    return result
    
except Exception as e:
    # Mark as failed
    if self.processing_manager:
        self.processing_manager.mark_failed(input_file, str(e))
    raise
```

#### `preprocess_directory_parallel(input_dir: Path, output_dir: Path, max_workers: Optional[int] = None)`
Parallel processing with memory constraints.

```python
# Initialize ProcessingManager
processing_manager = ProcessingManager(date_output_dir)
self.processing_manager = processing_manager

# Filter files using ProcessingManager
unprocessed_files = []
for json_file in json_files:
    if not processing_manager.is_processed(json_file):
        unprocessed_files.append(json_file)

# Process with parallel workers
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks...
```

## Command Line Interface

### Basic Usage
```bash
python scripts/assembly/preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
```

### Advanced Options

#### Parallel Processing
```bash
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --parallel \
    --max-workers 4 \
    --max-memory 2048 \
    --memory-threshold 80.0
```

#### Processing Management
```bash
# Show processing summary
python scripts/assembly/preprocess_laws.py \
    --output data/processed/assembly/law \
    --show-summary

# Reset failed files
python scripts/assembly/preprocess_laws.py \
    --output data/processed/assembly/law \
    --reset-failed
```

#### Memory-Safe Processing
```bash
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --max-memory 1024 \
    --memory-threshold 85.0 \
    --max-workers 1
```

### All Available Options
- `--input`: Input directory path
- `--output`: Output directory path
- `--parallel`: Enable parallel processing
- `--max-workers`: Maximum worker processes
- `--max-memory`: Maximum memory usage in MB
- `--memory-threshold`: System memory threshold percentage
- `--reset-failed`: Reset failed files for reprocessing
- `--show-summary`: Show processing summary and exit
- `--enable-legal-analysis`: Enable legal analysis (default)
- `--disable-legal-analysis`: Disable legal analysis
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Error Handling and Recovery

### Automatic Retry Mechanism
```python
# Failed files are automatically marked for retry
failed_files = processing_manager.get_failed_files()
if failed_files:
    logger.info(f"Found {len(failed_files)} failed files")
    # Reset and retry
    processing_manager.reset_failed()
```

### Error Types and Handling

#### Memory Errors
```python
try:
    # Process large file
    process_large_file()
except MemoryError:
    logger.error("Memory error - forcing cleanup")
    aggressive_garbage_collection()
    # Retry with smaller chunks
```

#### File Processing Errors
```python
try:
    # Process file
    result = process_file(file_path)
except Exception as e:
    # Log error and mark as failed
    logger.error(f"Error processing {file_path}: {e}")
    processing_manager.mark_failed(file_path, str(e))
    continue  # Skip to next file
```

#### System Resource Errors
```python
# Check system memory before processing
if not check_memory_safety(85.0):
    logger.critical("System memory too high - aborting")
    sys.exit(1)
```

## Performance Optimization

### Memory Optimization Strategies

#### File-by-File Processing
```python
# Process one file at a time to minimize memory usage
for file_path in file_list:
    process_file(file_path)
    cleanup_large_objects(result)
    aggressive_garbage_collection()
```

#### Aggressive Cleanup
```python
# Clean up after each law
for law in laws:
    processed_law = process_law(law)
    save_law(processed_law)
    cleanup_large_objects(processed_law)
    
    # Periodic cleanup
    if i % 5 == 0:
        aggressive_garbage_collection()
```

#### Memory Monitoring
```python
# Monitor memory usage
memory = get_memory_usage()
if memory['rss_mb'] > max_memory_mb:
    logger.warning(f"Memory usage high: {memory['rss_mb']:.1f}MB")
    aggressive_garbage_collection()
```

### Parallel Processing Optimization

#### Worker Count Calculation
```python
# Calculate optimal worker count based on memory
if PSUTIL_AVAILABLE:
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    memory_based_workers = max(1, int(available_memory_gb / 2))
    max_workers = min(mp.cpu_count(), total_files, memory_based_workers, 6)
```

#### Memory-Safe Workers
```python
# Each worker has memory limits
def process_file_worker(json_file_path: str, output_dir_path: str, enable_legal_analysis: bool = False):
    try:
        # Check memory safety
        force_exit_on_memory_limit(70.0)
        
        # Process with memory limits
        preprocessor = LawPreprocessor(
            enable_legal_analysis=enable_legal_analysis, 
            max_memory_mb=1024
        )
        
        result = preprocessor.preprocess_law_file(json_file, output_dir)
        
        # Cleanup
        cleanup_large_objects(preprocessor)
        return result
        
    except Exception as e:
        aggressive_garbage_collection()
        return {'file_name': Path(json_file_path).name, 'error': str(e)}
```

## Testing and Validation

### Unit Testing
```python
import pytest
from scripts.assembly.preprocess_laws import ProcessingManager

def test_processing_manager():
    # Test ProcessingManager functionality
    manager = ProcessingManager(Path("test_output"))
    
    # Test file processing status
    assert not manager.is_processed(Path("test_file.json"))
    
    # Test marking as processing
    manager.mark_processing(Path("test_file.json"))
    
    # Test marking as completed
    manager.mark_completed(Path("test_file.json"), 10, 5.5)
    
    # Test summary
    summary = manager.get_summary()
    assert summary['total_files'] == 1
    assert summary['by_status']['completed']['count'] == 1
```

### Integration Testing
```python
def test_full_processing_pipeline():
    # Test complete processing workflow
    input_dir = Path("test_data/input")
    output_dir = Path("test_data/output")
    
    preprocessor = LawPreprocessor()
    result = preprocessor.preprocess_directory(input_dir, output_dir)
    
    assert result['success_count'] > 0
    assert result['error_count'] == 0
```

### Performance Testing
```python
def test_memory_usage():
    # Test memory usage during processing
    initial_memory = get_memory_usage()
    
    # Process large file
    process_large_file()
    
    final_memory = get_memory_usage()
    memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
    
    # Should not exceed reasonable limits
    assert memory_increase < 1000  # Less than 1GB increase
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Error: Memory usage too high (CRITICAL: System memory usage is 80.4% (threshold: 75.0%))
# Solution: Increase memory threshold or reduce memory usage
python scripts/assembly/preprocess_laws.py \
    --max-memory 256 \
    --memory-threshold 95.0 \
    --max-workers 1

# Alternative: Disable legal analysis to reduce memory usage
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --disable-legal-analysis \
    --max-memory 256 \
    --memory-threshold 90.0 \
    --max-workers 1
```

#### System Memory Threshold Issues
```bash
# Error: "Forcing program exit due to excessive memory usage"
# This occurs when system memory usage exceeds the threshold
# Solutions:

# 1. Increase memory threshold (recommended for high-memory systems)
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --memory-threshold 95.0 \
    --max-memory 512 \
    --max-workers 1

# 2. Reduce memory usage
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --disable-legal-analysis \
    --max-memory 128 \
    --memory-threshold 85.0 \
    --max-workers 1
```

#### Processing Failures
```bash
# Check failed files
python scripts/assembly/preprocess_laws.py \
    --output data/processed/assembly/law \
    --show-summary

# Reset failed files
python scripts/assembly/preprocess_laws.py \
    --output data/processed/assembly/law \
    --reset-failed
```

#### Database Issues
```python
# Check database integrity
import sqlite3
conn = sqlite3.connect("data/processed/assembly/law/20251012/processing_status.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM processing_status")
count = cursor.fetchone()[0]
print(f"Total records: {count}")
```

### Debug Mode
```bash
# Enable debug logging
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --log-level DEBUG
```

## Best Practices

### Memory Management
1. Always use memory limits for large datasets
2. Monitor memory usage during processing
3. Use aggressive cleanup after processing large files
4. Set appropriate memory thresholds

### Memory Threshold Configuration
The `--memory-threshold` parameter controls when the program will force exit due to high system memory usage. Choose appropriate values based on your system:

#### High-Memory Systems (32GB+ RAM)
```bash
# Recommended settings for high-memory systems
--memory-threshold 90.0  # Allow up to 90% system memory usage
--max-memory 2048        # Use up to 2GB per process
--max-workers 4          # Use multiple workers
```

#### Medium-Memory Systems (16-32GB RAM)
```bash
# Recommended settings for medium-memory systems
--memory-threshold 85.0  # Allow up to 85% system memory usage
--max-memory 1024        # Use up to 1GB per process
--max-workers 2          # Use limited workers
```

#### Low-Memory Systems (16GB RAM or less)
```bash
# Recommended settings for low-memory systems
--memory-threshold 95.0  # Allow up to 95% system memory usage
--max-memory 256         # Use up to 256MB per process
--max-workers 1          # Use single worker
--disable-legal-analysis  # Disable memory-intensive features
```

### Common Memory Threshold Issues

#### Issue: "CRITICAL: System memory usage is 80.4% (threshold: 75.0%)"
**Cause**: The system memory threshold is set too low for the current system state.

**Solutions**:
1. **Increase threshold** (recommended):
   ```bash
   --memory-threshold 90.0  # or 95.0 for high-memory systems
   ```

2. **Reduce memory usage**:
   ```bash
   --max-memory 256 --disable-legal-analysis --max-workers 1
   ```

3. **Check system memory**:
   ```bash
   # Windows
   wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:table
   
   # Linux
   free -h
   ```

### Error Handling
1. Always mark files as failed when errors occur
2. Use try-catch blocks around file processing
3. Log detailed error information
4. Implement retry mechanisms for transient failures

### Performance
1. Use parallel processing for large datasets
2. Calculate optimal worker count based on system resources
3. Process files individually to minimize memory usage
4. Use database for state management

### Monitoring
1. Enable comprehensive logging
2. Track processing statistics
3. Monitor system resources
4. Generate processing summaries

## Migration from v1.0

### Key Changes
1. **ProcessingManager**: New state management system
2. **Memory Management**: Enhanced memory monitoring and cleanup
3. **Parallel Processing**: Improved parallel processing with memory constraints
4. **Error Recovery**: Automatic retry and resume capabilities
5. **Command Line**: New options for processing management

### Migration Steps
1. Update command line scripts to use new options
2. Implement ProcessingManager in custom scripts
3. Add memory monitoring to existing code
4. Update error handling to use new mechanisms
5. Test with new processing management features

The Assembly Law Data Preprocessing Pipeline v2.0 provides enterprise-grade reliability, performance, and monitoring capabilities for production use.
