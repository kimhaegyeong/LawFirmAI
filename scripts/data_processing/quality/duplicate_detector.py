#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Duplicate Detector

This module provides multi-level duplicate detection for law data,
including file-level, content-level, and semantic duplicate detection.

Usage:
    detector = AdvancedDuplicateDetector()
    file_duplicates = detector.detect_file_level_duplicates(file_paths)
    content_duplicates = detector.detect_content_level_duplicates(laws)
    semantic_duplicates = detector.detect_semantic_duplicates(laws)
"""

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json

# Import required libraries
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from difflib import SequenceMatcher
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class DuplicateGroup:
    """Duplicate group information"""
    group_id: str
    group_type: str  # 'file', 'content', 'semantic'
    primary_item: Any
    duplicate_items: List[Any]
    similarity_scores: Dict[str, float]
    confidence: float
    created_at: str


class AdvancedDuplicateDetector:
    """Advanced duplicate detection system with multi-level detection algorithms"""
    
    def __init__(self):
        """Initialize the duplicate detector"""
        self.logger = logging.getLogger(__name__)
        
        # Similarity thresholds
        self.exact_duplicate_threshold = 0.95
        self.near_duplicate_threshold = 0.85
        self.semantic_similarity_threshold = 0.80
        
        # Initialize TF-IDF vectorizer for content comparison
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # Korean stop words not implemented
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
        else:
            self.tfidf_vectorizer = None
            self.logger.warning("TF-IDF vectorizer not available. Content-level detection will be limited.")
    
    def detect_file_level_duplicates(self, file_paths: List[Path]) -> List[List[Path]]:
        """
        Detect file-level duplicates based on hash, size, and name similarity
        
        Args:
            file_paths: List of file paths to check
            
        Returns:
            List[List[Path]]: Groups of duplicate files
        """
        try:
            self.logger.info(f"Detecting file-level duplicates for {len(file_paths)} files")
            
            # Calculate file hashes and metadata
            file_metadata = {}
            for file_path in file_paths:
                if file_path.exists():
                    metadata = self._get_file_metadata(file_path)
                    file_metadata[file_path] = metadata
            
            # Group by exact hash matches
            hash_groups = {}
            for file_path, metadata in file_metadata.items():
                file_hash = metadata['hash']
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(file_path)
            
            # Find exact duplicates
            exact_duplicates = []
            for file_hash, files in hash_groups.items():
                if len(files) > 1:
                    exact_duplicates.append(files)
            
            # Find near duplicates by size and name similarity
            near_duplicates = self._find_near_duplicate_files(file_metadata)
            
            # Combine results
            all_duplicates = exact_duplicates + near_duplicates
            
            self.logger.info(f"Found {len(exact_duplicates)} exact duplicate groups and {len(near_duplicates)} near duplicate groups")
            return all_duplicates
            
        except Exception as e:
            self.logger.error(f"Error in file-level duplicate detection: {e}")
            return []
    
    def detect_content_level_duplicates(self, laws: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Detect content-level duplicates using TF-IDF and cosine similarity
        
        Args:
            laws: List of law data dictionaries
            
        Returns:
            List[List[Dict[str, Any]]]: Groups of duplicate laws
        """
        try:
            self.logger.info(f"Detecting content-level duplicates for {len(laws)} laws")
            
            if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
                self.logger.warning("Using basic text similarity for content detection")
                return self._detect_content_duplicates_basic(laws)
            
            # Extract text content for comparison
            law_texts = []
            law_indices = []
            
            for i, law in enumerate(laws):
                text = self._extract_law_text(law)
                if text and len(text.strip()) > 10:  # Skip empty or very short texts
                    law_texts.append(text)
                    law_indices.append(i)
            
            if len(law_texts) < 2:
                return []
            
            # Create TF-IDF matrix
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(law_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find duplicate groups
                duplicate_groups = []
                processed_indices = set()
                
                for i in range(len(similarity_matrix)):
                    if i in processed_indices:
                        continue
                    
                    # Find similar laws
                    similar_indices = []
                    for j in range(i + 1, len(similarity_matrix)):
                        if j not in processed_indices and similarity_matrix[i][j] >= self.near_duplicate_threshold:
                            similar_indices.append(j)
                    
                    if similar_indices:
                        # Create duplicate group
                        group_laws = [laws[law_indices[i]]]
                        for j in similar_indices:
                            group_laws.append(laws[law_indices[j]])
                            processed_indices.add(j)
                        
                        duplicate_groups.append(group_laws)
                        processed_indices.add(i)
                
                self.logger.info(f"Found {len(duplicate_groups)} content-level duplicate groups")
                return duplicate_groups
                
            except Exception as e:
                self.logger.error(f"Error in TF-IDF processing: {e}")
                return self._detect_content_duplicates_basic(laws)
            
        except Exception as e:
            self.logger.error(f"Error in content-level duplicate detection: {e}")
            return []
    
    def detect_semantic_duplicates(self, laws: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Detect semantic duplicates using vector embeddings
        
        Args:
            laws: List of law data dictionaries
            
        Returns:
            List[List[Dict[str, Any]]]: Groups of semantically similar laws
        """
        try:
            self.logger.info(f"Detecting semantic duplicates for {len(laws)} laws")
            
            # This would require integration with the existing vector store
            # For now, implement a basic version using text similarity
            
            duplicate_groups = []
            processed_indices = set()
            
            for i, law1 in enumerate(laws):
                if i in processed_indices:
                    continue
                
                similar_laws = [law1]
                law1_text = self._extract_law_text(law1)
                
                for j, law2 in enumerate(laws[i+1:], i+1):
                    if j in processed_indices:
                        continue
                    
                    law2_text = self._extract_law_text(law2)
                    
                    # Calculate semantic similarity using various methods
                    similarity = self._calculate_semantic_similarity(law1_text, law2_text)
                    
                    if similarity >= self.semantic_similarity_threshold:
                        similar_laws.append(law2)
                        processed_indices.add(j)
                
                if len(similar_laws) > 1:
                    duplicate_groups.append(similar_laws)
                    processed_indices.add(i)
            
            self.logger.info(f"Found {len(duplicate_groups)} semantic duplicate groups")
            return duplicate_groups
            
        except Exception as e:
            self.logger.error(f"Error in semantic duplicate detection: {e}")
            return []
    
    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get file metadata including hash, size, and name"""
        try:
            # Calculate file hash
            file_hash = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    file_hash.update(chunk)
            
            # Get file stats
            stat = file_path.stat()
            
            return {
                'hash': file_hash.hexdigest(),
                'size': stat.st_size,
                'name': file_path.name,
                'stem': file_path.stem,
                'suffix': file_path.suffix,
                'modified_time': stat.st_mtime
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file metadata for {file_path}: {e}")
            return {'hash': '', 'size': 0, 'name': '', 'stem': '', 'suffix': '', 'modified_time': 0}
    
    def _find_near_duplicate_files(self, file_metadata: Dict[Path, Dict[str, Any]]) -> List[List[Path]]:
        """Find near duplicate files based on size and name similarity"""
        near_duplicates = []
        processed_files = set()
        
        files = list(file_metadata.keys())
        
        for i, file1 in enumerate(files):
            if file1 in processed_files:
                continue
            
            similar_files = [file1]
            metadata1 = file_metadata[file1]
            
            for j, file2 in enumerate(files[i+1:], i+1):
                if file2 in processed_files:
                    continue
                
                metadata2 = file_metadata[file2]
                
                # Check size similarity (within 10% difference)
                size_diff = abs(metadata1['size'] - metadata2['size'])
                size_ratio = size_diff / max(metadata1['size'], metadata2['size'], 1)
                
                # Check name similarity
                name_similarity = SequenceMatcher(None, metadata1['name'], metadata2['name']).ratio()
                
                if size_ratio < 0.1 and name_similarity > 0.8:
                    similar_files.append(file2)
                    processed_files.add(file2)
            
            if len(similar_files) > 1:
                near_duplicates.append(similar_files)
                processed_files.add(file1)
        
        return near_duplicates
    
    def _extract_law_text(self, law: Dict[str, Any]) -> str:
        """Extract text content from law data"""
        text_parts = []
        
        # Add law name
        if law.get('law_name'):
            text_parts.append(law['law_name'])
        
        # Add articles content
        articles = law.get('articles', [])
        for article in articles:
            if isinstance(article, dict):
                # Add article title
                if article.get('article_title'):
                    text_parts.append(article['article_title'])
                
                # Add article content
                content = article.get('content', '') or article.get('text', '')
                if content:
                    text_parts.append(content)
        
        # Add full text if available
        if law.get('full_text'):
            text_parts.append(law['full_text'])
        
        return ' '.join(text_parts)
    
    def _detect_content_duplicates_basic(self, laws: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Basic content duplicate detection without TF-IDF"""
        duplicate_groups = []
        processed_indices = set()
        
        for i, law1 in enumerate(laws):
            if i in processed_indices:
                continue
            
            similar_laws = [law1]
            law1_text = self._extract_law_text(law1)
            
            for j, law2 in enumerate(laws[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                law2_text = self._extract_law_text(law2)
                
                # Calculate text similarity
                similarity = SequenceMatcher(None, law1_text, law2_text).ratio()
                
                if similarity >= self.near_duplicate_threshold:
                    similar_laws.append(law2)
                    processed_indices.add(j)
            
            if len(similar_laws) > 1:
                duplicate_groups.append(similar_laws)
                processed_indices.add(i)
        
        return duplicate_groups
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            # Basic implementation using text similarity
            # In a full implementation, this would use vector embeddings
            
            # Normalize texts
            text1_norm = self._normalize_text(text1)
            text2_norm = self._normalize_text(text2)
            
            # Calculate similarity using multiple methods
            similarity_scores = []
            
            # Character-level similarity
            char_similarity = SequenceMatcher(None, text1_norm, text2_norm).ratio()
            similarity_scores.append(char_similarity)
            
            # Word-level similarity
            words1 = set(text1_norm.split())
            words2 = set(text2_norm.split())
            if words1 or words2:
                word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                similarity_scores.append(word_similarity)
            
            # Keyword similarity (extract legal keywords)
            keywords1 = self._extract_legal_keywords(text1_norm)
            keywords2 = self._extract_legal_keywords(text2_norm)
            if keywords1 or keywords2:
                keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
                similarity_scores.append(keyword_similarity)
            
            # Return weighted average
            if similarity_scores:
                return sum(similarity_scores) / len(similarity_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Korean characters
        text = re.sub(r'[^\w\s가-??', ' ', text)
        
        return text.strip()
    
    def _extract_legal_keywords(self, text: str) -> Set[str]:
        """Extract legal keywords from text"""
        # Basic legal keyword patterns
        legal_patterns = [
            r'�?s*�?,
            r'�?s*�?,
            r'??s*�?,
            r'부\s*�?,
            r'??s*??,
            r'�?s*??,
            r'�?s*??,
            r'??s*지',
            r'??s*\d+\s*�?,
            r'??s*\d+\s*??,
            r'??s*\d+\s*??
        ]
        
        keywords = set()
        for pattern in legal_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
        
        return keywords
    
    def create_duplicate_group(self, group_type: str, items: List[Any], similarity_scores: Dict[str, float]) -> DuplicateGroup:
        """Create a duplicate group object"""
        group_id = f"{group_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(items))}"
        
        # Calculate confidence based on similarity scores
        if similarity_scores:
            confidence = sum(similarity_scores.values()) / len(similarity_scores)
        else:
            confidence = 0.0
        
        return DuplicateGroup(
            group_id=group_id,
            group_type=group_type,
            primary_item=items[0] if items else None,
            duplicate_items=items[1:] if len(items) > 1 else [],
            similarity_scores=similarity_scores,
            confidence=confidence,
            created_at=datetime.now().isoformat()
        )
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics and capabilities"""
        return {
            'exact_duplicate_threshold': self.exact_duplicate_threshold,
            'near_duplicate_threshold': self.near_duplicate_threshold,
            'semantic_similarity_threshold': self.semantic_similarity_threshold,
            'tfidf_available': SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None,
            'detection_methods': ['file_level', 'content_level', 'semantic'],
            'supported_formats': ['json', 'text', 'html']
        }


def detect_duplicates_comprehensive(items: List[Any], item_type: str = 'laws') -> Dict[str, List[List[Any]]]:
    """
    Comprehensive duplicate detection for any type of items
    
    Args:
        items: List of items to check for duplicates
        item_type: Type of items ('laws', 'files', etc.)
        
    Returns:
        Dict[str, List[List[Any]]]: Dictionary with different types of duplicates
    """
    detector = AdvancedDuplicateDetector()
    
    results = {}
    
    if item_type == 'laws':
        # Content-level detection
        content_duplicates = detector.detect_content_level_duplicates(items)
        results['content_duplicates'] = content_duplicates
        
        # Semantic detection
        semantic_duplicates = detector.detect_semantic_duplicates(items)
        results['semantic_duplicates'] = semantic_duplicates
        
    elif item_type == 'files':
        # File-level detection
        file_duplicates = detector.detect_file_level_duplicates(items)
        results['file_duplicates'] = file_duplicates
    
    return results


if __name__ == "__main__":
    # Test the duplicate detector
    test_laws = [
        {
            'law_name': '민법',
            'articles': [
                {'article_number': '1', 'article_title': '목적', 'content': '??법�? 민사??관??기본법이??'},
                {'article_number': '2', 'article_title': '?�용', 'content': '민법?� 민사??관?�여 ?�용?�다.'}
            ]
        },
        {
            'law_name': '민법',  # Duplicate name
            'articles': [
                {'article_number': '1', 'article_title': '목적', 'content': '??법�? 민사??관??기본법이??'},  # Duplicate content
                {'article_number': '2', 'article_title': '?�용', 'content': '민법?� 민사??관?�여 ?�용?�다.'}
            ]
        }
    ]
    
    detector = AdvancedDuplicateDetector()
    duplicates = detector.detect_content_level_duplicates(test_laws)
    
    print(f"Found {len(duplicates)} duplicate groups:")
    for i, group in enumerate(duplicates):
        print(f"Group {i+1}: {len(group)} laws")
        for law in group:
            print(f"  - {law['law_name']}")

