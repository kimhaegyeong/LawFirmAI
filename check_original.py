#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re

def check_original_text():
    """Check original text of 제2조"""
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
        law_content = target_law.get('law_content', '')
        
        # Find 제2조 content
        lines = law_content.split('\n')
        for i, line in enumerate(lines):
            if '제2조' in line and '정의' in line:
                print(f'Found 제2조 at line {i+1}')
                print(f'Line content: {line}')
                
                # Show next few lines
                for j in range(i, min(i+10, len(lines))):
                    print(f'Line {j+1}: {lines[j]}')
                break
        
        # Also search for numbered items
        print('\nSearching for numbered items...')
        numbered_items = re.findall(r'\d+\.\s*"[^"]+"', law_content)
        print(f'Found {len(numbered_items)} numbered items with quotes:')
        for item in numbered_items[:5]:  # Show first 5
            print(f'  {item}')
            
        # Search for numbered items without quotes
        numbered_items_no_quotes = re.findall(r'\d+\.\s*[^0-9]+', law_content)
        print(f'\nFound {len(numbered_items_no_quotes)} numbered items without quotes:')
        for item in numbered_items_no_quotes[:5]:  # Show first 5
            print(f'  {item[:100]}...')
    else:
        print('Target law not found')

if __name__ == "__main__":
    check_original_text()
