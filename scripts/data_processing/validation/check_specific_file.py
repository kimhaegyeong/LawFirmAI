#!/usr/bin/env python3
"""
Check specific demo file details
"""

import json
import os

def check_specific_file():
    """Check specific demo file details"""
    demo_dir = "../../data/processed/assembly/law/individual_laws_demo"
    files = os.listdir(demo_dir)
    
    # Find a file with articles
    target_file = None
    for f in files:
        if '1963년도국가공무원연금법' in f:
            target_file = f
            break
    
    if not target_file:
        print("Target file not found")
        return
    
    print(f"Target file: {target_file}")
    
    try:
        with open(f"{demo_dir}/{target_file}", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Law ID: {data.get('law_id')}")
        print(f"Law Name: {data.get('law_name')}")
        print(f"Law Type: {data.get('law_type')}")
        print(f"Category: {data.get('category')}")
        print(f"Total Articles: {data.get('total_articles')}")
        print(f"Main Articles: {len(data.get('main_articles', []))}")
        print(f"Supplementary Articles: {len(data.get('supplementary_articles', []))}")
        print(f"Parsing Status: {data.get('parsing_status')}")
        print(f"Is Valid: {data.get('is_valid')}")
        print(f"Validation Errors: {data.get('validation_errors', [])}")
        
        print("\nFirst Article:")
        if data.get('main_articles'):
            article = data['main_articles'][0]
            print(f"  Article Number: {article.get('article_number')}")
            print(f"  Article Title: {article.get('article_title')}")
            print(f"  Content Length: {len(article.get('article_content', ''))}")
            print(f"  Sub Articles: {len(article.get('sub_articles', []))}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_specific_file()
