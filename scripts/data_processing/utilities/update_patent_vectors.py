#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?πÌóà ?êÎ? Î≤°ÌÑ∞ ?ÑÎ≤†???ÖÎç∞?¥Ìä∏
"""

import sys
sys.path.append('source')
from source.data.vector_store import LegalVectorStore
import json
from pathlib import Path
from tqdm import tqdm

def main():
    # Î≤°ÌÑ∞ ?§ÌÜ†??Ï¥àÍ∏∞??
    vector_store = LegalVectorStore(
        model_name='jhgan/ko-sroberta-multitask',
        dimension=768,
        index_type='flat'
    )

    # Í∏∞Ï°¥ ?∏Îç±??Î°úÎìú
    vector_store.load_index('data/embeddings/ml_enhanced_ko_sroberta_precedents')

    print(f'Current index size: {vector_store.index.ntotal}')

    # ?ÑÏ≤òÎ¶¨Îêú ?πÌóà ?êÎ? ?åÏùº Ï≤òÎ¶¨
    patent_dir = Path('data/processed/assembly/precedent/patent')
    files = list(patent_dir.rglob('*_processed.json'))

    print(f'Found {len(files)} processed patent files')

    total_added = 0
    
    for file_path in tqdm(files, desc="Processing patent files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'items' in data:
                items = data['items']
                
                # Î≤°ÌÑ∞ ?ÑÎ≤†?©Ïóê Ï∂îÍ???Î¨∏ÏÑú Ï§ÄÎπ?
                texts = []
                metadatas = []
                for item in items:
                    text = item.get('searchable_text', item.get('full_text', ''))
                    if not text.strip():
                        continue
                        
                    metadata = {
                        'case_id': item.get('case_id', ''),
                        'case_name': item.get('case_name', ''),
                        'category': item.get('category', 'patent'),
                        'court': item.get('court', ''),
                        'decision_date': item.get('decision_date', ''),
                        'field': item.get('field', ''),
                        'case_number': item.get('case_number', '')
                    }
                    texts.append(text)
                    metadatas.append(metadata)
                
                if texts:
                    # Î≤°ÌÑ∞ ?ÑÎ≤†?©Ïóê Ï∂îÍ?
                    success = vector_store.add_documents(texts, metadatas)
                    if success:
                        total_added += len(texts)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # ?∏Îç±???Ä??
    vector_store.save_index('data/embeddings/ml_enhanced_ko_sroberta_precedents')
    
    print(f'Final index size: {vector_store.index.ntotal}')
    print(f'Total documents added: {total_added}')

if __name__ == "__main__":
    main()
