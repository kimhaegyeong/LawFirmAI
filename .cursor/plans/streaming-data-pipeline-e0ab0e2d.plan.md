<!-- e0ab0e2d-95c5-458c-a361-5964a0c99d40 7c48d076-9cd1-490a-b369-027e0b13b9bf -->
# Streaming Data Processing Pipeline Implementation

## Overview

Create a lightweight, file system watcher-based streaming pipeline that automatically processes new raw data files, generates embeddings, and updates the vector store with minimal resource usage.

## Architecture

### Core Components

1. **File System Watcher** - Monitors `data/raw` directory for new JSON files
2. **Sequential Processor** - Single-threaded processing pipeline
3. **Incremental Vector Store** - Smart embedding update system
4. **State Manager** - Tracks processed files and rebuild decisions

## Implementation Plan

### Phase 1: File System Watcher (scripts/streaming/file_watcher.py)

Create a lightweight file watcher that:

- Monitors `data/raw` and subdirectories recursively
- Detects new `.json` files using `watchdog` library
- Waits for file write completion before processing
- Maintains a processed files registry to avoid duplicates
- Emits events to the processing pipeline

Key features:

- Debouncing to handle rapid file writes
- Cross-platform compatibility (Windows/Linux)
- Graceful shutdown with cleanup
- PID file management per project standards

### Phase 2: Sequential Data Processor (scripts/streaming/sequential_processor.py)

Build a single-worker processor that:

- Receives file paths from the watcher
- Validates file format and data type
- Applies existing preprocessing logic from `source/data/data_processor.py`
- Generates embeddings using `source/data/vector_store.py`
- Tracks processing statistics (success/failure counts)

Processing flow:

```
New File → Validate → Preprocess → Generate Embeddings → Update Index → Log
```

### Phase 3: Smart Vector Store Manager (scripts/streaming/smart_vector_manager.py)

Implement intelligent index management:

- Track total documents and new additions
- Calculate percentage of new data vs existing
- If new data < 20%: Incremental add to existing FAISS index
- If new data >= 20%: Trigger full rebuild
- Maintain backup of previous index before rebuild
- Atomic index updates (write to temp, then swap)

Features:

- Read existing index from `data/embeddings/ml_enhanced_ko_sroberta/`
- Append new embeddings with `faiss.IndexFlatL2.add()`
- Metadata synchronization with `metadata.json`
- Checkpoint every 50 documents processed

### Phase 4: State and Checkpoint Manager (scripts/streaming/state_manager.py)

Create persistent state tracking:

- Store processed file hashes in `data/streaming_state.json`
- Track document counts and embedding statistics
- Save checkpoints after each successful processing
- Enable resume after interruption
- Maintain processing history for audit

State structure:

```json
{
  "processed_files": {"file_path": "hash"},
  "total_documents": 12345,
  "last_processed": "2025-01-15T10:30:00",
  "rebuild_threshold": 0.2,
  "last_rebuild": "2025-01-10T08:00:00"
}
```

### Phase 5: Main Streaming Pipeline (scripts/streaming/streaming_pipeline.py)

Integrate all components:

- Initialize file watcher on startup
- Set up signal handlers (SIGINT, SIGTERM)
- Process files sequentially as they arrive
- Update state after each file
- Log all operations to `logs/streaming_pipeline.log`
- Generate periodic statistics reports

Command-line interface:

```bash
python scripts/streaming/streaming_pipeline.py --start
python scripts/streaming/streaming_pipeline.py --stop
python scripts/streaming/streaming_pipeline.py --status
```

### Phase 6: Integration with Existing System

Modify existing components:

1. **source/data/vector_store.py**: Add `add_documents_incremental()` method
2. **source/data/data_processor.py**: Add `process_single_file()` method for streaming
3. **docs/development/streaming_pipeline_guide.md**: Create usage documentation

## File Structure

```
scripts/
├── streaming/
│   ├── __init__.py
│   ├── file_watcher.py          # File system monitoring
│   ├── sequential_processor.py   # Single-threaded processing
│   ├── smart_vector_manager.py   # Index update logic
│   ├── state_manager.py          # State persistence
│   ├── streaming_pipeline.py     # Main orchestrator
│   └── README.md                 # Usage guide
├── streaming_pipeline.pid        # PID file (gitignored)
└── stop_streaming.py             # Graceful shutdown script

data/
├── streaming_state.json          # Processing state
└── streaming_checkpoint.json     # Recovery checkpoint

logs/
└── streaming_pipeline.log        # Pipeline logs
```

## Key Implementation Details

### Dependencies

- `watchdog>=3.0.0` - File system monitoring
- Existing: `faiss-cpu`, `sentence-transformers`, `numpy`

### Resource Limits

- Single worker thread (no multiprocessing)
- Process one file at a time
- Batch size: 32 documents for embedding generation
- Memory limit: Monitor and warn if >4GB usage

### Error Handling

- Retry failed files up to 3 times with exponential backoff
- Move failed files to `data/raw/failed/` with error log
- Continue processing other files on individual failures
- Email/log alerts for repeated failures

### Testing Strategy

1. Unit tests for each component
2. Integration test with mock file creation
3. Load test with 100 sequential files
4. Recovery test (kill and restart)

## Success Criteria

- Detects new files within 1 second
- Processes files sequentially without race conditions
- Correctly performs incremental updates (<20% new data)
- Successfully triggers rebuild (>=20% new data)
- Maintains state across restarts
- Uses <2GB RAM during operation
- Processes 10 files/minute on average

## Future Enhancements (Out of Scope)

- Multi-worker parallel processing
- Message queue integration (Redis)
- Real-time monitoring dashboard
- Auto-scaling based on queue depth
- Distributed processing across multiple machines

### To-dos

- [ ] Install watchdog library and verify existing dependencies (faiss-cpu, sentence-transformers)
- [ ] Implement file_watcher.py with watchdog monitoring for data/raw directory
- [ ] Implement state_manager.py for tracking processed files and checkpoint management
- [ ] Implement sequential_processor.py for single-threaded file processing
- [ ] Implement smart_vector_manager.py with 20% threshold logic for incremental vs rebuild
- [ ] Add add_documents_incremental() method to source/data/vector_store.py
- [ ] Add process_single_file() method to source/data/data_processor.py for streaming
- [ ] Implement streaming_pipeline.py as main orchestrator with CLI interface
- [ ] Create stop_streaming.py for graceful shutdown using PID file
- [ ] Implement retry logic, failed file handling, and error logging across all components
- [ ] Write docs/development/streaming_pipeline_guide.md with usage examples and troubleshooting
- [ ] Test end-to-end pipeline with mock files and verify incremental/rebuild logic