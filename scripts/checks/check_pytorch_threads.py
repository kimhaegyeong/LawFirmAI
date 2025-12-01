#!/usr/bin/env python3
"""PyTorch 스레드 설정 확인 스크립트"""
import torch
import os

print("=" * 60)
print("PyTorch 스레드 설정 확인")
print("=" * 60)

print(f"\n[시스템 정보]")
print(f"  CPU 코어 수: {os.cpu_count()}")

print(f"\n[PyTorch 스레드 설정]")
print(f"  현재 스레드 수: {torch.get_num_threads()}")
print(f"  현재 인터럽트 스레드 수: {torch.get_num_interop_threads()}")

print(f"\n[환경 변수]")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '설정 안 됨')}")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '설정 안 됨')}")
print(f"  NUMEXPR_NUM_THREADS: {os.environ.get('NUMEXPR_NUM_THREADS', '설정 안 됨')}")

# SentenceEmbedder 초기화하여 실제 설정 확인
print(f"\n[SentenceEmbedder 초기화 후]")
try:
    import sys
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    
    from scripts.utils.embeddings import SentenceEmbedder
    
    # 임베더 초기화 (스레드 설정이 적용됨)
    embedder = SentenceEmbedder()
    
    print(f"  PyTorch 스레드 수: {torch.get_num_threads()}")
    print(f"  PyTorch 인터럽트 스레드 수: {torch.get_num_interop_threads()}")
    print(f"  환경 변수 MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '설정 안 됨')}")
    print(f"  환경 변수 OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '설정 안 됨')}")
    
    if torch.get_num_threads() == os.cpu_count():
        print(f"\n  ✅ 최적화 적용됨: {torch.get_num_threads()} 스레드 사용 중")
    else:
        print(f"\n  ⚠️  최적화 미적용: {torch.get_num_threads()} 스레드 (권장: {os.cpu_count()})")
        
except Exception as e:
    print(f"  ❌ 오류 발생: {e}")

print("\n" + "=" * 60)

