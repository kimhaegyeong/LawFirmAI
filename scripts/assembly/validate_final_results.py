#!/usr/bin/env python3
"""
Final validation of improved preprocessing results
"""

import json
import os
from pathlib import Path

def validate_final_results():
    """Validate the final preprocessing results"""
    print("Final Validation of Improved Preprocessing Results")
    print("=" * 60)
    
    output_dir = Path("../../data/processed/assembly/law/improved_individual_laws")
    
    if not output_dir.exists():
        print("Output directory not found")
        return
    
    files = list(output_dir.glob("*.json"))
    print(f"Total individual law files: {len(files)}")
    
    # Statistics
    files_with_articles = 0
    files_with_paragraphs = 0
    total_articles = 0
    total_paragraphs = 0
    
    # Sample files for detailed analysis
    sample_files = []
    
    for file_path in files[:100]:  # Check first 100 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            articles = data.get('main_articles', [])
            if articles:
                files_with_articles += 1
                total_articles += len(articles)
                
                # Check for paragraphs
                has_paragraphs = False
                for article in articles:
                    sub_articles = article.get('sub_articles', [])
                    if sub_articles:
                        has_paragraphs = True
                        total_paragraphs += len(sub_articles)
                
                if has_paragraphs:
                    files_with_paragraphs += 1
                    if len(sample_files) < 5:  # Collect sample files
                        sample_files.append((file_path.name, data))
                        
        except Exception as e:
            continue
    
    print(f"\n=== Statistics ===")
    print(f"Files with articles: {files_with_articles}")
    print(f"Files with paragraphs: {files_with_paragraphs}")
    print(f"Total articles: {total_articles}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Average articles per file: {total_articles/files_with_articles:.1f}" if files_with_articles > 0 else "N/A")
    print(f"Average paragraphs per file: {total_paragraphs/files_with_paragraphs:.1f}" if files_with_paragraphs > 0 else "N/A")
    
    print(f"\n=== Sample Files with Paragraphs ===")
    for i, (filename, data) in enumerate(sample_files, 1):
        print(f"\n{i}. {data.get('law_name', 'Unknown')[:50]}...")
        print(f"   File: {filename}")
        print(f"   Total articles: {data.get('total_articles', 0)}")
        print(f"   Main articles: {len(data.get('main_articles', []))}")
        print(f"   Supplementary articles: {len(data.get('supplementary_articles', []))}")
        
        # Show articles with paragraphs
        for article in data.get('main_articles', [])[:3]:  # Show first 3 articles
            sub_articles = article.get('sub_articles', [])
            if sub_articles:
                print(f"   - {article.get('article_number')}: {len(sub_articles)} paragraphs")
                for sub_article in sub_articles[:2]:  # Show first 2 paragraphs
                    print(f"     * {sub_article.get('type')} {sub_article.get('number')}: {sub_article.get('content', '')[:30]}...")

if __name__ == "__main__":
    validate_final_results()
