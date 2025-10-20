# Assembly Law Data Preprocessing - Developer Guide v3.0

## Overview
This guide provides comprehensive documentation for developers working with the Assembly Law Data Preprocessing Pipeline v3.0, featuring **sequential processing only** for improved memory management, simplified architecture, and enhanced reliability.

## Key Changes in v3.0

### 🔄 **Sequential Processing Only**
- **Removed**: All parallel processing capabilities
- **Simplified**: Single-threaded processing for predictable memory usage
- **Improved**: Memory management and stability
- **Enhanced**: Error handling and debugging capabilities

### 🧠 **Simplified Memory Management**
- **Removed**: Complex memory monitoring and aggressive cleanup
- **Simplified**: Basic memory checks and garbage collection
- **Optimized**: Memory usage patterns for sequential processing
- **Enhanced**: Predictable memory consumption

## Architecture Overview

### Core Components
```
preprocess_laws.py (Main Orchestrator)
├── ProcessingManager (State Management)
├── LawPreprocessor (Sequential Processing Engine)
├── Parser Modules
│   ├── HTMLParser
│   ├── ArticleParser
│   ├── MetadataExtractor
│   ├── TextNormalizer
│   └── SearchableTextGenerator
└── Simplified Memory Management System
```

### Processing Flow
1. **Initialization** → ProcessingManager setup
2. **File Discovery** → Status checking and filtering
3. **Sequential Processing** → One file at a time processing
4. **State Update** → Database status tracking
5. **Simple Cleanup** → Basic memory management

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

## Simplified Memory Management System

### Memory Monitoring Functions

#### `simple_memory_check() -> bool`
Simple memory check for sequential processing.

```python
if not simple_memory_check():
    logger.warning("High memory usage detected")
    return False
return True
```

#### `simple_garbage_collection()`
Simple garbage collection.

```python
simple_garbage_collection()  # Performs basic garbage collection
```

#### `simple_memory_monitor()`
Simple memory monitoring for sequential processing.

```python
simple_memory_monitor()  # Monitors memory and cleans up if needed
```

#### `simple_log_memory(stage: str)`
Simple memory logging for sequential processing.

```python
simple_log_memory("file_processing_start")  # Logs current memory usage
```

### Memory Optimization Functions

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

## Enhanced LawPreprocessor Class

### Constructor
```python
def __init__(
    self, 
    enable_legal_analysis: bool = True, 
    max_memory_mb: int = 2048, 
    max_memory_gb: float = 10.0,
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

#### `preprocess_directory(input_dir: Path, output_dir: Path) -> Dict[str, Any]`
Sequential processing with simplified memory management.

```python
# Initialize ProcessingManager
processing_manager = ProcessingManager(date_output_dir)
self.processing_manager = processing_manager

# Filter files using ProcessingManager
unprocessed_files = []
for json_file in json_files:
    if not processing_manager.is_processed(json_file):
        unprocessed_files.append(json_file)

# Process files sequentially
for i, json_file in enumerate(unprocessed_files, 1):
    try:
        logger.info(f"[{i}/{len(unprocessed_files)}] Processing file: {json_file.name}")
        file_result = self.preprocess_law_file(json_file, date_output_dir)
        # Simple cleanup after each file
        simple_garbage_collection()
    except Exception as e:
        logger.error(f"Error processing {json_file.name}: {e}")
        continue
```

## Command Line Interface

### Basic Usage
```bash
python scripts/assembly/preprocess_laws.py --input data/raw/assembly/law/20251010 --output data/processed/assembly/law
```

### Advanced Options

#### Sequential Processing (Default)
```bash
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --max-memory 1024 \
    --memory-threshold 85.0
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
    --max-memory 512 \
    --memory-threshold 90.0
```

### All Available Options
- `--input`: Input directory path
- `--output`: Output directory path
- `--max-memory`: Maximum memory usage in MB
- `--max-memory-gb`: Maximum system memory usage in GB
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
    logger.error("Memory error - performing cleanup")
    simple_garbage_collection()
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
if not simple_memory_check():
    logger.warning("System memory usage high")
    simple_garbage_collection()
```

## Performance Optimization

### Memory Optimization Strategies

#### File-by-File Processing
```python
# Process one file at a time to minimize memory usage
for file_path in file_list:
    process_file(file_path)
    simple_garbage_collection()  # Simple cleanup after each file
```

#### Simple Cleanup
```python
# Clean up after each law
for law in laws:
    processed_law = process_law(law)
    save_law(processed_law)
    del processed_law  # Simple cleanup
    
    # Periodic cleanup
    if i % 5 == 0:
        simple_garbage_collection()
```

#### Memory Monitoring
```python
# Monitor memory usage
if PSUTIL_AVAILABLE:
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > 2000:  # 2GB threshold
        logger.warning(f"Memory usage: {memory_mb:.1f}MB")
        simple_garbage_collection()
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
# Error: High memory usage
# Solution: Use sequential processing with memory limits
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --max-memory 512 \
    --memory-threshold 90.0

# Alternative: Disable legal analysis to reduce memory usage
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --disable-legal-analysis \
    --max-memory 256 \
    --memory-threshold 95.0
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
1. Use sequential processing for predictable memory usage
2. Monitor memory usage during processing
3. Use simple cleanup after processing files
4. Set appropriate memory thresholds

### Memory Threshold Configuration
The `--memory-threshold` parameter controls when the program will log warnings about high system memory usage. Choose appropriate values based on your system:

#### High-Memory Systems (32GB+ RAM)
```bash
# Recommended settings for high-memory systems
--memory-threshold 90.0  # Allow up to 90% system memory usage
--max-memory 2048        # Use up to 2GB per process
```

#### Medium-Memory Systems (16-32GB RAM)
```bash
# Recommended settings for medium-memory systems
--memory-threshold 85.0  # Allow up to 85% system memory usage
--max-memory 1024        # Use up to 1GB per process
```

#### Low-Memory Systems (16GB RAM or less)
```bash
# Recommended settings for low-memory systems
--memory-threshold 95.0  # Allow up to 95% system memory usage
--max-memory 256         # Use up to 256MB per process
--disable-legal-analysis  # Disable memory-intensive features
```

### Error Handling
1. Always mark files as failed when errors occur
2. Use try-catch blocks around file processing
3. Log detailed error information
4. Implement retry mechanisms for transient failures

### Performance
1. Use sequential processing for stable memory usage
2. Process files individually to minimize memory usage
3. Use database for state management
4. Monitor processing progress

### Monitoring
1. Enable comprehensive logging
2. Track processing statistics
3. Monitor system resources
4. Generate processing summaries

## Migration from v2.0

### Key Changes
1. **Sequential Processing Only**: Removed all parallel processing capabilities
2. **Simplified Memory Management**: Replaced complex memory monitoring with simple checks
3. **Removed Options**: `--parallel`, `--max-workers` options removed
4. **Enhanced Stability**: More predictable memory usage and processing behavior

### Migration Steps
1. Remove `--parallel` and `--max-workers` from command line scripts
2. Update memory management code to use simplified functions
3. Test with sequential processing
4. Update monitoring code to use new memory functions
5. Verify processing stability and memory usage

### Command Line Changes
```bash
# v2.0 (Old)
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --parallel \
    --max-workers 4 \
    --max-memory 2048

# v3.0 (New)
python scripts/assembly/preprocess_laws.py \
    --input data/raw/assembly/law/20251010 \
    --output data/processed/assembly/law \
    --max-memory 2048
```

## Version History

### v3.0 (Current)
- **Removed**: All parallel processing capabilities
- **Simplified**: Memory management system
- **Enhanced**: Sequential processing stability
- **Improved**: Memory usage predictability
- **Optimized**: Error handling and debugging

### v2.0 (Previous)
- Added ProcessingManager for state tracking
- Implemented parallel processing with memory constraints
- Enhanced error handling and recovery
- Added comprehensive monitoring and reporting

### v1.0 (Initial)
- Basic preprocessing pipeline
- Parser modules implementation
- Database import functionality
- Basic error handling

## FAQ (Frequently Asked Questions)

### Q: Why was parallel processing removed?
**A**: Parallel processing was removed to improve memory management and processing stability. Sequential processing provides more predictable memory usage and easier debugging.

### Q: Will sequential processing be slower?
**A**: While sequential processing may be slower for very large datasets, it provides:
- More predictable memory usage
- Better error handling and debugging
- More stable processing
- Easier resource management

### Q: How can I optimize processing speed?
**A**: 
1. Use `--disable-legal-analysis` to reduce processing overhead
2. Process smaller batches of files
3. Use faster storage (SSD)
4. Ensure sufficient RAM for the dataset

### Q: What's the recommended memory threshold for my system?
**A**: 
- **32GB+ RAM**: `--memory-threshold 90.0`
- **16-32GB RAM**: `--memory-threshold 85.0`
- **16GB RAM or less**: `--memory-threshold 95.0`

### Q: Can I still resume processing after interruption?
**A**: Yes! The ProcessingManager system still provides full resume capability. Simply run the same command again and it will skip already processed files.

### Q: How do I check processing status?
**A**: Use `--show-summary` option:
```bash
python scripts/assembly/preprocess_laws.py --output data/processed/assembly/law --show-summary
```

### Q: How do I retry failed files?
**A**: Use `--reset-failed` option:
```bash
python scripts/assembly/preprocess_laws.py --output data/processed/assembly/law --reset-failed
```

The Assembly Law Data Preprocessing Pipeline v3.0 provides enhanced stability, predictable memory usage, and simplified architecture for reliable production use.
