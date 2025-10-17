#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
메모리 사용량 분석 스크립트
"""

import sys
import os
import psutil
import gc
from pathlib import Path
import time
import tracemalloc

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def analyze_memory_usage():
    """현재 메모리 사용량 분석"""
    print("=== 메모리 사용량 분석 ===")
    
    # 시스템 메모리 정보
    memory = psutil.virtual_memory()
    print(f"시스템 총 메모리: {memory.total / (1024**3):.2f} GB")
    print(f"시스템 사용 메모리: {memory.used / (1024**3):.2f} GB")
    print(f"시스템 사용률: {memory.percent:.1f}%")
    print(f"시스템 가용 메모리: {memory.available / (1024**3):.2f} GB")
    
    # 현재 프로세스 메모리 정보
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"\n현재 프로세스 메모리:")
    print(f"  RSS (실제 메모리): {process_memory.rss / (1024**2):.2f} MB")
    print(f"  VMS (가상 메모리): {process_memory.vms / (1024**2):.2f} MB")
    
    # Python 메모리 추적 시작
    tracemalloc.start()
    
    print(f"\n=== 모델 로딩 전 메모리 상태 ===")
    print(f"Python 메모리 사용량: {process_memory.rss / (1024**2):.2f} MB")
    
    return process_memory.rss

def test_model_loading_memory():
    """모델 로딩 시 메모리 사용량 테스트"""
    print("\n=== 모델 로딩 메모리 테스트 ===")
    
    try:
        # Sentence-BERT 모델 로딩 테스트
        from sentence_transformers import SentenceTransformer
        
        print("Sentence-BERT 모델 로딩 중...")
        start_memory = psutil.Process().memory_info().rss / (1024**2)
        
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        
        end_memory = psutil.Process().memory_info().rss / (1024**2)
        model_memory = end_memory - start_memory
        
        print(f"Sentence-BERT 모델 메모리 사용량: {model_memory:.2f} MB")
        
        # 벡터 스토어 로딩 테스트
        print("\n벡터 스토어 로딩 중...")
        start_memory = psutil.Process().memory_info().rss / (1024**2)
        
        from source.data.vector_store import LegalVectorStore
        vector_store = LegalVectorStore()
        vector_store.load_index("data/embeddings/ml_enhanced_ko_sroberta_precedents")
        
        end_memory = psutil.Process().memory_info().rss / (1024**2)
        vector_memory = end_memory - start_memory
        
        print(f"벡터 스토어 메모리 사용량: {vector_memory:.2f} MB")
        
        # 전체 메모리 사용량
        total_memory = psutil.Process().memory_info().rss / (1024**2)
        print(f"전체 메모리 사용량: {total_memory:.2f} MB")
        
        return {
            'sentence_bert': model_memory,
            'vector_store': vector_memory,
            'total': total_memory
        }
        
    except Exception as e:
        print(f"모델 로딩 테스트 실패: {e}")
        return None

def analyze_memory_patterns():
    """메모리 사용 패턴 분석"""
    print("\n=== 메모리 사용 패턴 분석 ===")
    
    # 가비지 컬렉션 전후 메모리 비교
    gc.collect()
    before_gc = psutil.Process().memory_info().rss / (1024**2)
    
    # 더 강력한 가비지 컬렉션
    collected = gc.collect()
    after_gc = psutil.Process().memory_info().rss / (1024**2)
    
    print(f"가비지 컬렉션 전: {before_gc:.2f} MB")
    print(f"가비지 컬렉션 후: {after_gc:.2f} MB")
    print(f"정리된 메모리: {before_gc - after_gc:.2f} MB")
    print(f"수집된 객체 수: {collected}")
    
    # 메모리 추적 정보
    if tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print(f"\n메모리 사용량 상위 5개:")
        for index, stat in enumerate(top_stats[:5], 1):
            print(f"{index}. {stat}")
    
    tracemalloc.stop()

if __name__ == "__main__":
    # 기본 메모리 분석
    initial_memory = analyze_memory_usage()
    
    # 모델 로딩 메모리 테스트
    model_memory = test_model_loading_memory()
    
    # 메모리 패턴 분석
    analyze_memory_patterns()
    
    print(f"\n=== 메모리 분석 완료 ===")
    if model_memory:
        print(f"최적화 대상:")
        print(f"  - Sentence-BERT 모델: {model_memory['sentence_bert']:.2f} MB")
        print(f"  - 벡터 스토어: {model_memory['vector_store']:.2f} MB")
        print(f"  - 총 메모리 사용량: {model_memory['total']:.2f} MB")
