#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ë°°ì¹˜ ?„ì²˜ë¦??¤í¬ë¦½íŠ¸ - ?¹ì • ?°ì´??? í˜•ë§?ì²˜ë¦¬

?¬ìš©ë²?
    python scripts/batch_preprocess.py --data-type laws
    python scripts/batch_preprocess.py --data-type precedents --dry-run
    python scripts/batch_preprocess.py --data-type all --enable-normalization
"""

import argparse
import sys
from pathlib import Path

# ?„ë¡œ?íŠ¸ ë£¨íŠ¸ë¥?Python ê²½ë¡œ??ì¶”ê?
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess_raw_data import RawDataPreprocessingPipeline

def main():
    parser = argparse.ArgumentParser(description="Raw ?°ì´??ë°°ì¹˜ ?„ì²˜ë¦?)
    parser.add_argument("--data-type", required=True, 
                       choices=["laws", "precedents", "constitutional", "interpretations", "terms", "all"],
                       help="?„ì²˜ë¦¬í•  ?°ì´??? í˜•")
    parser.add_argument("--enable-normalization", action="store_true", default=True,
                       help="ë²•ë¥  ?©ì–´ ?•ê·œ???œì„±??)
    parser.add_argument("--output-dir", default="data/processed",
                       help="ì¶œë ¥ ?”ë ‰? ë¦¬")
    parser.add_argument("--dry-run", action="store_true",
                       help="?¤ì œ ì²˜ë¦¬ ?†ì´ ê³„íšë§?ì¶œë ¥")
    
    args = parser.parse_args()
    
    # ì¶œë ¥ ?”ë ‰? ë¦¬ ?¤ì •
    if args.output_dir != "data/processed":
        print(f"ì¶œë ¥ ?”ë ‰? ë¦¬: {args.output_dir}")
    
    pipeline = RawDataPreprocessingPipeline(args.enable_normalization)
    
    if args.dry_run:
        print("=== ?œë¼?´ëŸ° ëª¨ë“œ ===")
        pipeline.dry_run(args.data_type)
    else:
        if args.data_type == "all":
            print("=== ?„ì²´ ?„ì²˜ë¦??œì‘ ===")
            pipeline.run_full_preprocessing()
        else:
            print(f"=== {args.data_type} ?°ì´???„ì²˜ë¦??œì‘ ===")
            pipeline.process_specific_type(args.data_type)

if __name__ == "__main__":
    main()
