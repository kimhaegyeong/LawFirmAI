#!/usr/bin/env python3
"""
Test preprocessing with control character removal
"""

import sys
import os
from pathlib import Path

# Add the parsers directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from assembly.preprocess_laws import LawPreprocessor

def test_preprocessing_control_chars():
    """Test preprocessing with control character removal"""
    print("Testing Preprocessing with Control Character Removal")
    print("=" * 55)
    
    # Create test data with control characters
    test_law_data = {
        "law_id": "test_001",
        "law_name": "테스트 법률",
        "law_content": """제1조(목적)
이 법은 테스트를 위한 법률이다.

제2조(정의)
① "테스트"란 이 법에서 정하는 것을 말한다.
② "데이터"란 처리할 정보를 말한다."""
    }
    
    # Initialize preprocessor
    preprocessor = LawPreprocessor()
    
    # Test the _clean_law_content method directly
    print("Testing _clean_law_content method:")
    print("-" * 35)
    
    original_content = test_law_data["law_content"]
    cleaned_content = preprocessor._clean_law_content(original_content)
    
    print(f"Original content length: {len(original_content)}")
    print(f"Cleaned content length: {len(cleaned_content)}")
    print(f"Contains \\n: {'\\n' in cleaned_content}")
    print(f"Contains actual newlines: {'\n' in cleaned_content}")
    print()
    
    print("Original content:")
    print(repr(original_content))
    print()
    
    print("Cleaned content:")
    print(repr(cleaned_content))
    print()
    
    # Test full preprocessing
    print("Testing full preprocessing:")
    print("-" * 25)
    
    processed_data = preprocessor._process_single_law(test_law_data)
    
    if processed_data:
        print(f"Processing successful: {processed_data.get('law_name')}")
        
        # Check main articles
        main_articles = processed_data.get('main_articles', [])
        print(f"Main articles count: {len(main_articles)}")
        
        for i, article in enumerate(main_articles[:2]):
            content = article.get('article_content', '')
            print(f"\nArticle {i+1}: {article.get('article_number')}")
            print(f"Contains \\n: {'\\n' in content}")
            print(f"Contains actual newlines: {'\n' in content}")
            print(f"Content preview: {repr(content[:100])}")
    else:
        print("Processing failed")

if __name__ == "__main__":
    test_preprocessing_control_chars()
