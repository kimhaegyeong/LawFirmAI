#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
sys.path.append('scripts/assembly/parsers')
from article_parser import ArticleParser

def debug_html_parsing():
    """Debug HTML parsing to see what's happening"""
    # Load actual data
    with open('data/raw/assembly/law/2025101201/law_page_364_080246.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find 가축 및 축산물 이력관리에 관한 법률
    target_law = None
    for law in data['laws']:
        if '가축 및 축산물 이력관리에 관한 법률' in law.get('law_name', ''):
            target_law = law
            break

    if target_law:
        print('Found target law:', target_law['law_name'])
        
        # Test HTML parsing
        parser = ArticleParser()
        html_content = target_law.get('content_html', '')
        
        if html_content:
            print('HTML content length:', len(html_content))
            print('First 500 chars of HTML:', html_content[:500])
            
            print('\nTesting HTML parsing...')
            articles = parser.parse_articles('', html_content)
            print(f'Found {len(articles)} articles from HTML')
            
            # List all articles found
            for i, article in enumerate(articles):
                article_number = article.get('article_number', '')
                article_title = article.get('article_title', '')
                print(f'Article {i+1}: {article_number} - {article_title}')
                
                # Check if this is 제2조
                if '제2조' in article_number:
                    print(f'  Found 제2조! Sub-articles: {len(article.get("sub_articles", []))}')
                    for sub in article.get('sub_articles', []):
                        print(f'    {sub.get("type")} {sub.get("number")}: {sub.get("content", "")[:50]}...')
        else:
            print('No HTML content found')
    else:
        print('Target law not found')

if __name__ == "__main__":
    debug_html_parsing()
