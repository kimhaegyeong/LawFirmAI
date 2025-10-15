#!/usr/bin/env python3
"""
Find files with articles
"""

import json
import os

def find_files_with_articles():
    """Find files with articles"""
    demo_dir = "../../data/processed/assembly/law/individual_laws_demo"
    files = os.listdir(demo_dir)
    
    files_with_articles = []
    
    for file in files[:20]:  # Check first 20 files
        try:
            with open(f"{demo_dir}/{file}", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_articles = data.get('total_articles', 0)
            if total_articles > 0:
                files_with_articles.append((file, total_articles))
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    print(f"Files with articles: {len(files_with_articles)}")
    
    if files_with_articles:
        # Show first file with articles
        file_name, article_count = files_with_articles[0]
        print(f"First file with articles: {file_name}")
        print(f"Article count: {article_count}")
        
        # Load and show details
        try:
            with open(f"{demo_dir}/{file_name}", 'r', encoding='utf-8') as f:
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
            
            print("\nFirst Article:")
            if data.get('main_articles'):
                article = data['main_articles'][0]
                print(f"  Article Number: {article.get('article_number')}")
                print(f"  Article Title: {article.get('article_title')}")
                print(f"  Content Length: {len(article.get('article_content', ''))}")
                print(f"  Sub Articles: {len(article.get('sub_articles', []))}")
                
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("No files with articles found in first 20 files")

if __name__ == "__main__":
    find_files_with_articles()
