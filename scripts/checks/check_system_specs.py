#!/usr/bin/env python3
"""시스템 사양 확인 스크립트"""
import os
import sys
import psutil
import torch

print("=" * 60)
print("시스템 사양 확인")
print("=" * 60)

# CPU 정보
print(f"\n[CPU 정보]")
print(f"  논리적 코어 수: {os.cpu_count()}")
try:
    print(f"  물리적 코어 수: {psutil.cpu_count(logical=False)}")
except:
    print(f"  물리적 코어 수: 확인 불가")
cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
print(f"  CPU 사용률 (코어별): {[f'{p:.1f}%' for p in cpu_percent]}")
print(f"  평균 CPU 사용률: {sum(cpu_percent)/len(cpu_percent):.1f}%")

# 메모리 정보
mem = psutil.virtual_memory()
print(f"\n[메모리 정보]")
print(f"  전체 메모리: {mem.total / 1024**3:.2f} GB")
print(f"  사용 가능 메모리: {mem.available / 1024**3:.2f} GB")
print(f"  사용 중 메모리: {mem.used / 1024**3:.2f} GB ({mem.percent:.1f}%)")
print(f"  여유 메모리: {mem.free / 1024**3:.2f} GB")

# GPU 정보
print(f"\n[GPU 정보]")
print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 버전: {torch.version.cuda}")
    print(f"  GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    총 메모리: {props.total_memory / 1024**3:.2f} GB")
        try:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    할당된 메모리: {allocated:.2f} GB")
            print(f"    예약된 메모리: {reserved:.2f} GB")
        except:
            pass
else:
    print(f"  GPU를 사용할 수 없습니다 (CPU 모드)")

# PyTorch 정보
print(f"\n[PyTorch 정보]")
print(f"  PyTorch 버전: {torch.__version__}")
print(f"  현재 스레드 수: {torch.get_num_threads()}")
print(f"  현재 인터럽트 스레드 수: {torch.get_num_interop_threads()}")

# 현재 Python 프로세스 정보
print(f"\n[현재 Python 프로세스]")
try:
    current_process = psutil.Process()
    print(f"  프로세스 ID: {current_process.pid}")
    print(f"  메모리 사용: {current_process.memory_info().rss / 1024**3:.2f} GB")
    print(f"  CPU 사용률: {current_process.cpu_percent(interval=1):.1f}%")
except:
    pass

# 재임베딩 프로세스 확인
print(f"\n[실행 중인 재임베딩 프로세스]")
python_processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
    try:
        if 'python' in proc.info['name'].lower():
            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else ''
            if 're_embed' in cmdline:
                python_processes.append({
                    'pid': proc.info['pid'],
                    'memory_mb': proc.info['memory_info'].rss / 1024**2,
                    'cpu': proc.cpu_percent(interval=0.1)
                })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if python_processes:
    for proc in python_processes:
        print(f"  PID {proc['pid']}: 메모리 {proc['memory_mb']:.0f} MB, CPU {proc['cpu']:.1f}%")
else:
    print(f"  실행 중인 재임베딩 프로세스 없음")

# 최적화 권장사항
print(f"\n[최적화 권장사항]")
available_gb = mem.available / 1024**3
cpu_cores = os.cpu_count()

if available_gb > 16:
    print(f"  ✅ 메모리 여유: {available_gb:.1f} GB")
    print(f"     → doc_batch_size: 500 권장")
    print(f"     → embedding_batch_size: 2048 권장")
elif available_gb > 8:
    print(f"  ⚠️  메모리 보통: {available_gb:.1f} GB")
    print(f"     → doc_batch_size: 200-300 권장")
    print(f"     → embedding_batch_size: 512-1024 권장")
else:
    print(f"  ⚠️  메모리 부족: {available_gb:.1f} GB")
    print(f"     → doc_batch_size: 100 권장")
    print(f"     → embedding_batch_size: 256 권장")

if cpu_cores >= 8:
    print(f"  ✅ 멀티코어 CPU: {cpu_cores} 코어")
    print(f"     → 멀티프로세싱 활용 가능")
    print(f"     → PyTorch 스레드 수: {cpu_cores} 권장")
elif cpu_cores >= 4:
    print(f"  ⚠️  중간 코어 수: {cpu_cores} 코어")
    print(f"     → 제한적 멀티프로세싱 가능")
    print(f"     → PyTorch 스레드 수: {cpu_cores} 권장")
else:
    print(f"  ⚠️  낮은 코어 수: {cpu_cores} 코어")
    print(f"     → 멀티프로세싱 제한적")
    print(f"     → PyTorch 스레드 수: {cpu_cores} 권장")

if not torch.cuda.is_available():
    print(f"  ⚠️  GPU 없음: CPU만 사용 가능")
    print(f"     → 배치 크기는 메모리에 맞게 조정 필요")

print("\n" + "=" * 60)

