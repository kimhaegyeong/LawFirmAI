#!/usr/bin/env python3
"""
Verify control character removal in clean data
"""

import json
import os
from pathlib import Path

def verify_control_character_removal():
    """Verify that control characters have been removed"""
    print("Verifying Control Character Removal")
    print("=" * 40)
    
    output_dir = Path("../../data/processed/assembly/law/clean_individual_laws")
    
    if not output_dir.exists():
        print("Output directory not found")
        return
    
    files = list(output_dir.glob("*.json"))
    print(f"Total files: {len(files)}")
    
    # Check first 10 files for control characters
    files_with_control_chars = 0
    total_checked = 0
    
    for file_path in files[:10]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_checked += 1
            
            # Check main articles
            for article in data.get('main_articles', []):
                content = article.get('article_content', '')
                if '\\n' in content or '\\t' in content or '\\r' in content:
                    files_with_control_chars += 1
                    print(f"Found control chars in: {file_path.name}")
                    print(f"  Article: {article.get('article_number')}")
                    print(f"  Content preview: {content[:100]}...")
                    break
                
                # Check sub-articles
                for sub_article in article.get('sub_articles', []):
                    sub_content = sub_article.get('content', '')
                    if '\\n' in sub_content or '\\t' in sub_content or '\\r' in sub_content:
                        files_with_control_chars += 1
                        print(f"Found control chars in: {file_path.name}")
                        print(f"  Sub-article: {sub_article.get('type')} {sub_article.get('number')}")
                        print(f"  Content preview: {sub_content[:100]}...")
                        break
                        
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue
    
    print(f"\n=== Results ===")
    print(f"Files checked: {total_checked}")
    print(f"Files with control characters: {files_with_control_chars}")
    print(f"Clean files: {total_checked - files_with_control_chars}")
    print(f"Clean rate: {((total_checked - files_with_control_chars) / total_checked * 100):.1f}%")
    
    # Show sample clean content
    print(f"\n=== Sample Clean Content ===")
    for file_path in files[:3]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\nFile: {file_path.name}")
            print(f"Law: {data.get('law_name', 'Unknown')[:50]}...")
            
            for article in data.get('main_articles', [])[:2]:  # Show first 2 articles
                content = article.get('article_content', '')
                print(f"  {article.get('article_number')}: {content[:80]}...")
                print(f"    Has \\n: {'\\n' in content}")
                print(f"    Has \\t: {'\\t' in content}")
                
        except Exception as e:
            continue

if __name__ == "__main__":
    verify_control_character_removal()
