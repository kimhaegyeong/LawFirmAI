#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?�처리된 법률 ?�이?�에??article_title???�는 케?�스 검??
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_missing_titles():
    """article_title???�는 케?�스 분석"""
    print("Analyzing Missing Article Titles")
    print("=" * 50)
    
    processed_dir = Path("data/processed/assembly/law/1013/20251013")
    
    if not processed_dir.exists():
        print(f"Processed directory not found: {processed_dir}")
        return
    
    # ?�계 ?�집
    stats = {
        'total_files': 0,
        'total_articles': 0,
        'articles_without_title': 0,
        'files_with_issues': 0,
        'missing_title_patterns': defaultdict(int),
        'problematic_files': []
    }
    
    # JSON ?�일??처리
    json_files = list(processed_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith('metadata_')]
    
    print(f"Found {len(json_files)} law files to analyze")
    print()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stats['total_files'] += 1
            articles = data.get('articles', [])
            stats['total_articles'] += len(articles)
            
            file_issues = []
            
            for article in articles:
                article_title = article.get('article_title', '')
                article_number = article.get('article_number', '')
                article_content = article.get('article_content', '')
                
                if not article_title.strip():
                    stats['articles_without_title'] += 1
                    
                    # ?�턴 분석
                    if '()' in article_content:
                        stats['missing_title_patterns']['empty_parentheses'] += 1
                    elif '?? in article_content and '�? in article_content:
                        stats['missing_title_patterns']['article_reference'] += 1
                    elif len(article_content) < 50:
                        stats['missing_title_patterns']['short_content'] += 1
                    else:
                        stats['missing_title_patterns']['other'] += 1
                    
                    file_issues.append({
                        'article_number': article_number,
                        'content_preview': article_content[:100] + '...' if len(article_content) > 100 else article_content
                    })
            
            if file_issues:
                stats['files_with_issues'] += 1
                stats['problematic_files'].append({
                    'file': json_file.name,
                    'law_name': data.get('law_name', 'Unknown'),
                    'issues': file_issues
                })
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # 결과 출력
    print("=== ANALYSIS RESULTS ===")
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Total articles: {stats['total_articles']}")
    print(f"Articles without title: {stats['articles_without_title']}")
    print(f"Files with issues: {stats['files_with_issues']}")
    print()
    
    print("=== MISSING TITLE PATTERNS ===")
    for pattern, count in stats['missing_title_patterns'].items():
        print(f"{pattern}: {count}")
    print()
    
    print("=== PROBLEMATIC FILES (First 10) ===")
    for i, file_info in enumerate(stats['problematic_files'][:10]):
        print(f"\n{i+1}. {file_info['file']}")
        print(f"   Law: {file_info['law_name']}")
        print(f"   Issues: {len(file_info['issues'])}")
        for issue in file_info['issues'][:3]:  # Show first 3 issues
            print(f"     - {issue['article_number']}: {issue['content_preview']}")
    
    return stats

if __name__ == "__main__":
    analyze_missing_titles()
