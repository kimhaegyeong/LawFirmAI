#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_clustering_ground_truth.py 진행 상황 확인 스크립트
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def format_time(seconds: float) -> str:
    """시간 포맷팅"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}분 {secs}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}시간 {minutes}분 {secs}초"

def check_progress(checkpoint_dir: str = None):
    """진행 상황 확인"""
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
    else:
        checkpoint_path = Path("data/evaluation/checkpoints")
    
    checkpoint_file = checkpoint_path / "ground_truth_checkpoint.json"
    embeddings_file = checkpoint_path / "embeddings.npy"
    labels_file = checkpoint_path / "labels.npy"
    
    print("=" * 80)
    print("Ground Truth 생성 진행 상황 확인")
    print("=" * 80)
    print()
    
    if not checkpoint_file.exists():
        print("❌ 체크포인트 파일이 없습니다.")
        print(f"   경로: {checkpoint_file}")
        print()
        print("가능한 상황:")
        print("  1. 스크립트가 아직 시작하지 않았습니다")
        print("  2. 스크립트가 완료되어 체크포인트가 정리되었습니다")
        print("  3. 체크포인트가 다른 위치에 저장되었습니다")
        return
    
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        stage = checkpoint_data.get("stage", "unknown")
        saved_at = checkpoint_data.get("saved_at", "N/A")
        data = checkpoint_data.get("data", {})
        
        print(f"✅ 체크포인트 발견!")
        print(f"   저장 시각: {saved_at}")
        print()
        
        print("현재 단계:")
        stage_names = {
            "embeddings_extracting": "임베딩 생성 중",
            "embeddings_extracted": "임베딩 추출 완료",
            "optimal_clusters_found": "최적 클러스터 수 찾기 완료",
            "clustering_completed": "클러스터링 완료",
            "ground_truth_completed": "Ground Truth 생성 완료"
        }
        
        stage_display = stage_names.get(stage, stage)
        print(f"   [{stage}] {stage_display}")
        print()
        
        if stage == "embeddings_extracting":
            progress = data.get("progress", 0)
            total_batches = data.get("total_batches", 0)
            completed = data.get("completed", False)
            
            if total_batches > 0:
                progress_pct = (progress / total_batches * 100) if total_batches > 0 else 0
                print(f"   진행률: {progress}/{total_batches} 배치 ({progress_pct:.1f}%)")
                if completed:
                    print("   ✅ 임베딩 생성 완료")
                else:
                    print("   ⏳ 임베딩 생성 중...")
            print()
        
        if embeddings_file.exists():
            import numpy as np
            try:
                embeddings = np.load(embeddings_file, mmap_mode='r')
                print(f"✅ 임베딩 파일 존재")
                print(f"   크기: {embeddings.shape[0]:,}개 문서, {embeddings.shape[1]}차원")
                print(f"   파일 크기: {embeddings_file.stat().st_size / (1024**2):.2f} MB")
            except Exception as e:
                print(f"⚠️  임베딩 파일 로드 실패: {e}")
        else:
            print("❌ 임베딩 파일 없음")
        
        print()
        
        if labels_file.exists():
            import numpy as np
            try:
                labels = np.load(labels_file, mmap_mode='r')
                print(f"✅ 클러스터 레이블 파일 존재")
                print(f"   크기: {len(labels):,}개 레이블")
                unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"   클러스터 수: {unique_labels}개")
                if -1 in labels:
                    noise_count = list(labels).count(-1)
                    print(f"   노이즈 포인트: {noise_count:,}개")
                print(f"   파일 크기: {labels_file.stat().st_size / (1024**2):.2f} MB")
            except Exception as e:
                print(f"⚠️  레이블 파일 로드 실패: {e}")
        else:
            print("❌ 클러스터 레이블 파일 없음")
        
        print()
        print("상세 정보:")
        for key, value in data.items():
            if key not in ["progress", "total_batches", "completed"]:
                print(f"   {key}: {value}")
        
        print()
        print("=" * 80)
        
        if stage == "ground_truth_completed":
            print("✅ 작업이 완료되었습니다!")
            total_entries = data.get("total_entries", 0)
            print(f"   생성된 Ground Truth 항목: {total_entries:,}개")
        else:
            print("⏳ 작업이 진행 중입니다.")
            print()
            print("다음 단계:")
            if stage == "embeddings_extracting" or stage == "embeddings_extracted":
                print("  → 클러스터링 수행")
            elif stage == "optimal_clusters_found":
                print("  → 클러스터링 수행")
            elif stage == "clustering_completed":
                print("  → Ground Truth 생성")
        
    except Exception as e:
        print(f"❌ 체크포인트 파일 읽기 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground Truth 생성 진행 상황 확인")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="체크포인트 디렉토리 경로 (기본값: data/evaluation/checkpoints)"
    )
    
    args = parser.parse_args()
    check_progress(args.checkpoint_dir)

