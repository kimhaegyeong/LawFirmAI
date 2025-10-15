#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
배치 전처리 스크립트 - 특정 데이터 유형만 처리

사용법:
    python scripts/batch_preprocess.py --data-type laws
    python scripts/batch_preprocess.py --data-type precedents --dry-run
    python scripts/batch_preprocess.py --data-type all --enable-normalization
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess_raw_data import RawDataPreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Raw 데이터 배치 전처리")
    parser.add_argument("--data-type", required=True, 
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       help="전처리할 데이터 유형")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="법률 용어 정규화 활성화")
    parser.add_argument("--output-dir", default="data/processed",
                       help="출력 디렉토리")
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 처리 없이 계획만 출력")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir != "data/processed":
        print(f"출력 디렉토리: {args.output_dir}")
    
    pipeline = RawDataPreprocessingPipeline(args.enable_normalization)
    
    if args.dry_run:
        print("=== 드라이런 모드 ===")
        pipeline.dry_run(args.data_type)
    else:
        if args.data_type == "all":
            print("=== 전체 전처리 시작 ===")
            pipeline.run_full_preprocessing()
        else:
            print(f"=== {args.data_type} 데이터 전처리 시작 ===")
            pipeline.process_specific_type(args.data_type)

if __name__ == "__main__":
    main()
