#!/usr/bin/env python3
"""
Check demo files for articles
"""

import json
import os

def check_demo_files():
    """Check demo files for articles"""
    demo_dir = "../../data/processed/assembly/law/individual_laws_demo"
    files = os.listdir(demo_dir)
    
    found = False
    for file in files[:10]:
        try:
            with open(f"{demo_dir}/{file}", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('total_articles', 0) > 0:
                print(f"File: {file}")
                print(f"Law name: {data.get('law_name')}")
                print(f"Total articles: {data.get('total_articles')}")
                print(f"Main articles: {len(data.get('main_articles', []))}")
                print(f"Supplementary articles: {len(data.get('supplementary_articles', []))}")
                print(f"Parsing status: {data.get('parsing_status')}")
                print(f"Is valid: {data.get('is_valid')}")
                print("---")
                found = True
                break
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not found:
        print("No files with articles found in first 10 files")
        print(f"Total files: {len(files)}")

if __name__ == "__main__":
    check_demo_files()
