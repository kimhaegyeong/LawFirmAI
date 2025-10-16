# 체크포인트 및 Graceful Shutdown 시스템 구현 보고서

**작성일**: 2025-10-15  
**작성자**: LawFirmAI 개발팀  
**버전**: v1.0

---

## 📋 개요

벡터 임베딩 생성 과정에서 발생할 수 있는 중단 상황에 대비하여 체크포인트 시스템과 Graceful Shutdown 기능을 구현한 보고서입니다.

### 문제 상황
- **장시간 처리**: 벡터 임베딩 생성에 15-20시간 소요
- **중단 위험**: 시스템 재부팅, 네트워크 문제, 사용자 중단 등
- **재시작 비효율**: 중단 시 처음부터 다시 시작해야 함
- **진행률 불투명**: 중단 시 어디까지 진행되었는지 알 수 없음

---

## 🚀 구현된 솔루션

### 1. 체크포인트 관리 시스템

#### CheckpointManager 클래스
```python
class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = self._load_checkpoint()
    
    def save_checkpoint(self, completed_chunks: List[int], total_chunks: int):
        """체크포인트 저장"""
        checkpoint_data = {
            'completed_chunks': completed_chunks,
            'total_chunks': total_chunks,
            'start_time': self.checkpoint_data.get('start_time', time.time()),
            'last_update': time.time()
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def get_remaining_chunks(self, total_chunks: int) -> List[int]:
        """남은 청크 목록 반환"""
        completed = set(self.checkpoint_data.get('completed_chunks', []))
        return [i for i in range(total_chunks) if i not in completed]
    
    def get_progress_info(self) -> Dict[str, Any]:
        """진행 상황 정보 반환"""
        completed = len(self.checkpoint_data.get('completed_chunks', []))
        total = self.checkpoint_data.get('total_chunks', 0)
        
        progress_info = {
            'completed_chunks': completed,
            'total_chunks': total,
            'progress_percentage': (completed / max(total, 1)) * 100 if total > 0 else 0
        }
        
        if start_time:
            elapsed_time = time.time() - start_time
            progress_info['elapsed_time'] = elapsed_time
            if completed > 0:
                avg_time_per_chunk = elapsed_time / completed
                remaining_chunks = total - completed
                progress_info['estimated_remaining_time'] = avg_time_per_chunk * remaining_chunks
        
        return progress_info
```

#### 주요 기능
- **자동 저장**: 매 10개 청크마다 진행 상황 저장
- **진행률 계산**: 완료된 청크 수와 전체 청크 수 기반 진행률
- **예상 시간**: 평균 처리 시간 기반 남은 시간 추정
- **재시작 지원**: 완료된 청크 제외하고 남은 청크만 처리

### 2. Graceful Shutdown 시스템

#### 시그널 핸들러 설정
```python
def _setup_signal_handlers(self):
    """시그널 핸들러 설정 (Graceful shutdown)"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True
    
    # Windows와 Unix 모두 지원
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)
```

#### 안전한 종료 처리
```python
for chunk_idx in tqdm(chunk_indices, desc="Creating embeddings"):
    # Graceful shutdown 확인
    if self.shutdown_requested:
        logger.info("Graceful shutdown requested. Saving checkpoint and exiting...")
        checkpoint_manager.save_checkpoint(completed_chunks, total_chunks)
        logger.info("Checkpoint saved. You can resume later with --resume flag.")
        return self.stats
    
    # 청크 처리 로직
    try:
        self.vector_store.add_documents(texts, metadatas)
        completed_chunks.append(chunk_idx)
        
        # 체크포인트 저장 (매 10개 청크마다)
        if len(completed_chunks) % 10 == 0:
            checkpoint_manager.save_checkpoint(completed_chunks, total_chunks)
            
    except Exception as e:
        # 에러 처리
        pass
```

#### 지원되는 시그널
- **SIGTERM**: 시스템 종료 신호
- **SIGINT**: Ctrl+C 인터럽트
- **SIGBREAK**: Windows 전용 브레이크 신호

### 3. 재시작 로직

#### 자동 재시작 감지
```python
# 재시작 확인
if resume and checkpoint_manager.is_resume_needed():
    progress_info = checkpoint_manager.get_progress_info()
    logger.info(f"Resuming from checkpoint: {progress_info['completed_chunks']}/{progress_info['total_chunks']} chunks completed")
    logger.info(f"Progress: {progress_info['progress_percentage']:.1f}%")
    if 'estimated_remaining_time' in progress_info:
        remaining_hours = progress_info['estimated_remaining_time'] / 3600
        logger.info(f"Estimated remaining time: {remaining_hours:.1f} hours")
```

#### 남은 청크 처리
```python
# 재시작 시 남은 청크만 처리
if resume and checkpoint_manager.is_resume_needed():
    remaining_chunks = checkpoint_manager.get_remaining_chunks(total_chunks)
    logger.info(f"Processing {len(remaining_chunks)} remaining chunks out of {total_chunks}")
    chunk_indices = remaining_chunks
else:
    chunk_indices = list(range(total_chunks))
```

---

## 📊 사용법 및 명령어

### 기본 사용법
```bash
# 일반 실행 (체크포인트 자동 감지)
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/ml_enhanced_ko_sroberta \
    --batch-size 20 \
    --chunk-size 200 \
    --log-level INFO
```

### 재시작 옵션
```bash
# 명시적 재시작 (체크포인트에서 이어서)
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/ml_enhanced_ko_sroberta \
    --batch-size 20 \
    --chunk-size 200 \
    --log-level INFO \
    --resume
```

### 처음부터 시작
```bash
# 체크포인트 무시하고 처음부터 시작
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py \
    --input data/processed/assembly/law/20251013_ml \
    --output data/embeddings/ml_enhanced_ko_sroberta \
    --batch-size 20 \
    --chunk-size 200 \
    --log-level INFO \
    --no-resume
```

---

## 🔧 체크포인트 파일 구조

### embedding_checkpoint.json
```json
{
  "completed_chunks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "total_chunks": 780,
  "start_time": 1697364074.123,
  "last_update": 1697364500.456
}
```

#### 필드 설명
- **completed_chunks**: 완료된 청크 인덱스 목록
- **total_chunks**: 전체 청크 수
- **start_time**: 작업 시작 시간 (Unix timestamp)
- **last_update**: 마지막 체크포인트 저장 시간

---

## 📈 성능 및 안전성 개선

### 처리 효율성
| 항목 | 개선 전 | 개선 후 | 개선율 |
|------|---------|---------|--------|
| 중단 시 재시작 | 처음부터 | 중단 지점부터 | **100% 효율** |
| 진행률 추적 | 불가능 | 실시간 | **완전 추적** |
| 예상 완료 시간 | 불가능 | 실시간 계산 | **투명성 확보** |
| 안전한 종료 | 강제 종료 | Graceful | **데이터 보호** |

### 안전성 향상
- **데이터 무결성**: 현재 청크 완료 후 체크포인트 저장
- **메모리 관리**: 정기적인 메모리 정리 및 가비지 컬렉션
- **에러 복구**: 개별 청크 에러 시 전체 작업 중단 방지
- **시그널 처리**: 다양한 종료 시그널에 대한 안전한 대응

---

## 🎯 사용 시나리오

### 시나리오 1: 사용자 중단
```bash
# 작업 시작
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py --input ... --output ...

# Ctrl+C로 중단
^C
# 출력: Graceful shutdown requested. Saving checkpoint and exiting...
# 출력: Checkpoint saved. You can resume later with --resume flag.

# 재시작
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py --input ... --output ... --resume
# 출력: Resuming from checkpoint: 150/780 chunks completed
# 출력: Progress: 19.2%
# 출력: Estimated remaining time: 12.5 hours
```

### 시나리오 2: 시스템 재부팅
```bash
# 시스템 재부팅 후
python scripts/build_ml_enhanced_vector_db_cpu_optimized.py --input ... --output ...
# 자동으로 체크포인트 감지 및 재시작
# 출력: Resuming from checkpoint: 300/780 chunks completed
# 출력: Progress: 38.5%
```

### 시나리오 3: 에러 발생
```bash
# 개별 청크 에러 발생 시
# 출력: Error creating embeddings for chunk 45: Memory error
# 출력: Continuing with next chunk...

# 전체 작업은 계속 진행
# 체크포인트는 정상적으로 저장됨
```

---

## 🔍 모니터링 및 디버깅

### 로그 메시지
```
2025-10-15 17:01:32,468 - __main__ - INFO - Creating embeddings for 155819 documents...
2025-10-15 17:01:45,727 - __main__ - INFO - Checkpoint saved: 10/780 chunks completed
2025-10-15 17:02:11,490 - __main__ - INFO - Checkpoint saved: 20/780 chunks completed
2025-10-15 17:02:37,785 - __main__ - INFO - Checkpoint saved: 30/780 chunks completed
```

### 체크포인트 파일 모니터링
```bash
# 체크포인트 파일 확인
cat data/embeddings/ml_enhanced_ko_sroberta/embedding_checkpoint.json

# 진행률 확인
python -c "
import json
with open('data/embeddings/ml_enhanced_ko_sroberta/embedding_checkpoint.json', 'r') as f:
    data = json.load(f)
    completed = len(data['completed_chunks'])
    total = data['total_chunks']
    print(f'Progress: {completed}/{total} ({completed/total*100:.1f}%)')
"
```

---

## 🚀 향후 개선 계획

### 단기 개선 (1주)
1. **진행률 시각화**: 실시간 진행률 바 및 ETA 표시
2. **체크포인트 압축**: 대용량 체크포인트 파일 압축
3. **병렬 처리**: 여러 청크 동시 처리 지원

### 중기 개선 (1개월)
1. **웹 대시보드**: 웹 기반 진행률 모니터링
2. **알림 시스템**: 완료 시 이메일/Slack 알림
3. **성능 분석**: 청크별 처리 시간 분석

### 장기 개선 (3개월)
1. **분산 처리**: 여러 머신에서 병렬 처리
2. **클라우드 통합**: AWS/Azure 클라우드 지원
3. **자동 스케일링**: 리소스에 따른 자동 확장

---

## 📝 결론

체크포인트 및 Graceful Shutdown 시스템을 통해 다음과 같은 성과를 달성했습니다:

### ✅ 달성된 목표
- **중단 안전성**: 언제든지 안전하게 중단 가능
- **재시작 효율성**: 중단된 지점부터 이어서 작업
- **진행률 투명성**: 실시간 진행 상황 및 예상 완료 시간
- **데이터 보호**: Graceful shutdown으로 데이터 무결성 보장

### 🔍 핵심 성과
1. **체크포인트 시스템**: 매 10개 청크마다 자동 저장
2. **Graceful Shutdown**: 다양한 시그널에 대한 안전한 대응
3. **재시작 로직**: 완료된 청크 제외하고 효율적 재시작
4. **진행률 추적**: 실시간 진행률 및 예상 완료 시간 계산

### 🚀 시스템 안정성
- **장시간 처리**: 15-20시간 작업도 안전하게 처리
- **중단 복구**: 시스템 재부팅, 네트워크 문제 등에 대응
- **사용자 편의**: 간단한 명령어로 재시작 가능
- **데이터 무결성**: 체크포인트 기반 안전한 데이터 관리

이제 벡터 임베딩 생성이 중단되어도 안전하게 재시작할 수 있으며, 사용자는 언제든지 작업을 중단하고 나중에 이어서 진행할 수 있습니다.

---

**보고서 상태**: ✅ 완료  
**다음 업데이트**: 웹 대시보드 구현 후
