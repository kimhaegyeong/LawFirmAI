#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

raw_dir = Path('data/raw')
data_types = ['laws', 'precedents', 'constitutional_decisions', 'legal_interpretations', 
              'administrative_rules', 'local_ordinances', 'administrative_appeals', 
              'committee_decisions', 'treaties']

print("Raw 데이터 파일 현황:")
print("=" * 50)

total_files = 0
for dtype in data_types:
    subdir = raw_dir / dtype
    if subdir.exists():
        json_files = list(subdir.rglob('*.json'))
        count = len(json_files)
        total_files += count
        print(f"{dtype}: {count}개")
        if count > 0 and count <= 5:
            for f in json_files[:3]:
                print(f"  - {f.name}")

print(f"\n총 JSON 파일: {total_files}개")

