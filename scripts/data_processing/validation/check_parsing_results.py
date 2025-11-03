#!/usr/bin/env python3
"""
Check parsing results for paragraph structure
"""

import json
import os
from pathlib import Path

def check_paragraph_parsing():
    """Check if paragraphs are properly parsed"""
    print("Checking Paragraph Parsing Results")
    print("=" * 50)
    
    test_dir = Path("../../data/processed/assembly/law/test_improved_parsing")
    
    if not test_dir.exists():
        print("Test directory not found")
        return
    
    files = list(test_dir.glob("*.json"))
    print(f"Total files: {len(files)}")
    
    files_with_paragraphs = []
    total_paragraphs = 0
    
    for file_path in files[:50]:  # Check first 50 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            main_articles = data.get('main_articles', [])
            for article in main_articles:
                sub_articles = article.get('sub_articles', [])
                if sub_articles:
                    files_with_paragraphs.append({
                        'file': file_path.name,
                        'law_name': data.get('law_name', ''),
                        'article': article.get('article_number', ''),
                        'paragraphs': len(sub_articles)
                    })
                    total_paragraphs += len(sub_articles)
                    
        except Exception as e:
            continue
    
    print(f"\nFiles with paragraphs: {len(files_with_paragraphs)}")
    print(f"Total paragraphs found: {total_paragraphs}")
    
    print("\n=== Sample Results ===")
    for i, result in enumerate(files_with_paragraphs[:10], 1):
        print(f"{i}. {result['law_name'][:30]}...")
        print(f"   Article: {result['article']}")
        print(f"   Paragraphs: {result['paragraphs']}")
        print(f"   File: {result['file']}")
        print()
    
    # Check specific law that we know has paragraphs
    target_file = None
    for file_path in files:
        if 'ì¡°ì„ ?¬ë‹¨?€?¹ë ¹' in file_path.name:
            target_file = file_path
            break
    
    if target_file:
        print("=== Detailed Check: ì¡°ì„ ?¬ë‹¨?€?¹ë ¹ ===")
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Law: {data.get('law_name')}")
        print(f"Total articles: {data.get('total_articles')}")
        print(f"Main articles: {len(data.get('main_articles', []))}")
        
        for article in data.get('main_articles', []):
            print(f"\nArticle: {article.get('article_number')}")
            print(f"Title: {article.get('article_title')}")
            print(f"Sub-articles: {len(article.get('sub_articles', []))}")
            
            for sub_article in article.get('sub_articles', []):
                print(f"  {sub_article.get('type')} {sub_article.get('number')}: {sub_article.get('content', '')[:50]}...")

if __name__ == "__main__":
    check_paragraph_parsing()
