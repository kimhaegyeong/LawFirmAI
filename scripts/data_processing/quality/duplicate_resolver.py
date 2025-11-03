#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Duplicate Resolver

This module provides intelligent duplicate resolution strategies,
including quality-based selection, metadata merging, and version history tracking.

Usage:
    resolver = IntelligentDuplicateResolver()
    resolved_items = resolver.resolve_duplicates(duplicate_groups)
    merged_metadata = resolver.merge_metadata(duplicate_items)
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResolutionStrategy:
    """Duplicate resolution strategy configuration"""
    name: str
    description: str
    quality_weight: float = 0.4
    completeness_weight: float = 0.3
    recency_weight: float = 0.2
    uniqueness_weight: float = 0.1
    auto_resolve: bool = True
    require_manual_review: bool = False


@dataclass
class ResolutionResult:
    """Result of duplicate resolution"""
    primary_item: Any
    archived_items: List[Any]
    merged_metadata: Dict[str, Any]
    version_history: Dict[str, Any]
    resolution_strategy: str
    confidence_score: float
    resolution_timestamp: str
    manual_review_required: bool


class IntelligentDuplicateResolver:
    """Intelligent duplicate resolver with multiple resolution strategies"""
    
    def __init__(self):
        """Initialize the duplicate resolver"""
        self.logger = logging.getLogger(__name__)
        
        # Default resolution strategies
        self.strategies = {
            'quality_based': ResolutionStrategy(
                name='quality_based',
                description='Select item with highest quality score',
                quality_weight=0.6,
                completeness_weight=0.3,
                recency_weight=0.1
            ),
            'completeness_based': ResolutionStrategy(
                name='completeness_based',
                description='Select item with most complete data',
                quality_weight=0.2,
                completeness_weight=0.6,
                recency_weight=0.2
            ),
            'recency_based': ResolutionStrategy(
                name='recency_based',
                description='Select most recent item',
                quality_weight=0.2,
                completeness_weight=0.2,
                recency_weight=0.6
            ),
            'conservative': ResolutionStrategy(
                name='conservative',
                description='Conservative approach requiring manual review',
                quality_weight=0.3,
                completeness_weight=0.3,
                recency_weight=0.2,
                uniqueness_weight=0.2,
                auto_resolve=False,
                require_manual_review=True
            )
        }
    
    def resolve_duplicates(self, duplicate_groups: List[List[Any]], strategy_name: str = 'quality_based') -> List[ResolutionResult]:
        """
        Resolve duplicate groups using specified strategy
        
        Args:
            duplicate_groups: List of duplicate groups
            strategy_name: Name of resolution strategy to use
            
        Returns:
            List[ResolutionResult]: Resolution results for each group
        """
        try:
            self.logger.info(f"Resolving {len(duplicate_groups)} duplicate groups using {strategy_name} strategy")
            
            strategy = self.strategies.get(strategy_name, self.strategies['quality_based'])
            results = []
            
            for group in duplicate_groups:
                if len(group) < 2:
                    continue
                
                result = self._resolve_single_group(group, strategy)
                results.append(result)
            
            self.logger.info(f"Resolved {len(results)} duplicate groups")
            return results
            
        except Exception as e:
            self.logger.error(f"Error resolving duplicates: {e}")
            return []
    
    def _resolve_single_group(self, group: List[Any], strategy: ResolutionStrategy) -> ResolutionResult:
        """Resolve a single duplicate group"""
        try:
            # Calculate scores for each item in the group
            item_scores = []
            for item in group:
                score = self._calculate_item_score(item, strategy)
                item_scores.append((item, score))
            
            # Sort by score (highest first)
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select primary item
            primary_item = item_scores[0][0]
            archived_items = [item for item, score in item_scores[1:]]
            
            # Merge metadata
            merged_metadata = self.merge_metadata(group)
            
            # Create version history
            version_history = self.create_version_history(group, primary_item)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(item_scores, strategy)
            
            # Determine if manual review is required
            manual_review_required = (
                strategy.require_manual_review or
                confidence_score < 0.7 or
                len(group) > 5  # Large groups need review
            )
            
            return ResolutionResult(
                primary_item=primary_item,
                archived_items=archived_items,
                merged_metadata=merged_metadata,
                version_history=version_history,
                resolution_strategy=strategy.name,
                confidence_score=confidence_score,
                resolution_timestamp=datetime.now().isoformat(),
                manual_review_required=manual_review_required
            )
            
        except Exception as e:
            self.logger.error(f"Error resolving single group: {e}")
            # Fallback: use first item as primary
            return ResolutionResult(
                primary_item=group[0],
                archived_items=group[1:],
                merged_metadata={},
                version_history={},
                resolution_strategy=strategy.name,
                confidence_score=0.0,
                resolution_timestamp=datetime.now().isoformat(),
                manual_review_required=True
            )
    
    def _calculate_item_score(self, item: Any, strategy: ResolutionStrategy) -> float:
        """Calculate composite score for an item based on strategy"""
        try:
            scores = {}
            
            # Quality score
            quality_score = self._get_quality_score(item)
            scores['quality'] = quality_score
            
            # Completeness score
            completeness_score = self._get_completeness_score(item)
            scores['completeness'] = completeness_score
            
            # Recency score
            recency_score = self._get_recency_score(item)
            scores['recency'] = recency_score
            
            # Uniqueness score
            uniqueness_score = self._get_uniqueness_score(item)
            scores['uniqueness'] = uniqueness_score
            
            # Calculate weighted composite score
            composite_score = (
                scores['quality'] * strategy.quality_weight +
                scores['completeness'] * strategy.completeness_weight +
                scores['recency'] * strategy.recency_weight +
                scores['uniqueness'] * strategy.uniqueness_weight
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Error calculating item score: {e}")
            return 0.0
    
    def _get_quality_score(self, item: Any) -> float:
        """Get quality score for an item"""
        if isinstance(item, dict):
            # Check for explicit quality score
            if 'quality_score' in item:
                return float(item['quality_score'])
            
            # Calculate quality based on available data
            quality_indicators = 0
            total_indicators = 0
            
            # Check for required fields
            required_fields = ['law_name', 'articles']
            for field in required_fields:
                total_indicators += 1
                if field in item and item[field]:
                    quality_indicators += 1
            
            # Check article quality
            articles = item.get('articles', [])
            if articles:
                total_indicators += 1
                articles_with_titles = sum(1 for article in articles 
                                        if isinstance(article, dict) and article.get('article_title'))
                if articles_with_titles / len(articles) > 0.7:
                    quality_indicators += 1
            
            return quality_indicators / total_indicators if total_indicators > 0 else 0.0
        
        return 0.0
    
    def _get_completeness_score(self, item: Any) -> float:
        """Get completeness score for an item"""
        if isinstance(item, dict):
            completeness_indicators = 0
            total_indicators = 0
            
            # Check for various data fields
            fields_to_check = [
                'law_name', 'law_type', 'category', 'promulgation_number',
                'promulgation_date', 'enforcement_date', 'ministry', 'articles'
            ]
            
            for field in fields_to_check:
                total_indicators += 1
                if field in item and item[field]:
                    completeness_indicators += 1
            
            # Check article completeness
            articles = item.get('articles', [])
            if articles:
                total_indicators += 1
                complete_articles = sum(1 for article in articles 
                                     if isinstance(article, dict) and 
                                     article.get('article_number') and 
                                     article.get('content'))
                if complete_articles / len(articles) > 0.8:
                    completeness_indicators += 1
            
            return completeness_indicators / total_indicators if total_indicators > 0 else 0.0
        
        return 0.0
    
    def _get_recency_score(self, item: Any) -> float:
        """Get recency score for an item"""
        if isinstance(item, dict):
            # Check for processing timestamp
            timestamp_fields = ['processing_timestamp', 'created_at', 'updated_at']
            for field in timestamp_fields:
                if field in item and item[field]:
                    try:
                        timestamp = datetime.fromisoformat(item[field].replace('Z', '+00:00'))
                        # Calculate recency (newer = higher score)
                        days_old = (datetime.now() - timestamp).days
                        return max(0.0, 1.0 - (days_old / 365.0))  # Decay over a year
                    except (ValueError, TypeError):
                        continue
            
            # Check for enforcement date
            if 'enforcement_date' in item and item['enforcement_date']:
                try:
                    # Assume enforcement date is in YYYY-MM-DD format
                    enforcement_date = datetime.strptime(item['enforcement_date'], '%Y-%m-%d')
                    days_old = (datetime.now() - enforcement_date).days
                    return max(0.0, 1.0 - (days_old / 3650.0))  # Decay over 10 years
                except (ValueError, TypeError):
                    pass
        
        return 0.5  # Default neutral score
    
    def _get_uniqueness_score(self, item: Any) -> float:
        """Get uniqueness score for an item"""
        if isinstance(item, dict):
            # Items with unique identifiers get higher scores
            unique_fields = ['law_id', 'promulgation_number']
            for field in unique_fields:
                if field in item and item[field]:
                    return 1.0
            
            # Items with more unique content get higher scores
            articles = item.get('articles', [])
            if articles:
                unique_content_ratio = len(set(str(article) for article in articles)) / len(articles)
                return unique_content_ratio
        
        return 0.0
    
    def _calculate_confidence_score(self, item_scores: List[Tuple[Any, float]], strategy: ResolutionStrategy) -> float:
        """Calculate confidence score for the resolution"""
        if len(item_scores) < 2:
            return 1.0
        
        # Calculate score difference between primary and secondary items
        primary_score = item_scores[0][1]
        secondary_score = item_scores[1][1]
        
        if primary_score == 0:
            return 0.0
        
        score_ratio = secondary_score / primary_score
        confidence = 1.0 - score_ratio
        
        # Adjust based on group size (larger groups are less confident)
        group_size_factor = max(0.5, 1.0 - (len(item_scores) - 2) * 0.1)
        
        return confidence * group_size_factor
    
    def merge_metadata(self, items: List[Any]) -> Dict[str, Any]:
        """
        Merge metadata from multiple items
        
        Args:
            items: List of items to merge metadata from
            
        Returns:
            Dict[str, Any]: Merged metadata
        """
        try:
            if not items:
                return {}
            
            merged = {}
            
            # Collect all unique values for each field
            field_values = {}
            for item in items:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if value:  # Only include non-empty values
                            if key not in field_values:
                                field_values[key] = []
                            field_values[key].append(value)
            
            # Merge fields using different strategies
            for field, values in field_values.items():
                if field == 'articles':
                    # Merge articles, removing duplicates
                    merged[field] = self._merge_articles(values)
                elif field in ['quality_score', 'parsing_version']:
                    # Use the highest quality score or latest version
                    merged[field] = max(values) if isinstance(values[0], (int, float)) else values[-1]
                elif field in ['processing_timestamp', 'created_at', 'updated_at']:
                    # Use the most recent timestamp
                    merged[field] = self._get_latest_timestamp(values)
                elif field in ['law_id', 'promulgation_number']:
                    # Use the first non-empty value
                    merged[field] = values[0]
                else:
                    # For other fields, use the most complete value
                    merged[field] = self._select_most_complete_value(values)
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging metadata: {e}")
            return {}
    
    def _merge_articles(self, article_lists: List[List[Any]]) -> List[Any]:
        """Merge article lists, removing duplicates"""
        try:
            all_articles = []
            seen_articles = set()
            
            for article_list in article_lists:
                if isinstance(article_list, list):
                    for article in article_list:
                        if isinstance(article, dict):
                            # Create a hash for duplicate detection
                            article_hash = hash(str(article.get('article_number', '')) + str(article.get('content', '')))
                            if article_hash not in seen_articles:
                                seen_articles.add(article_hash)
                                all_articles.append(article)
            
            return all_articles
            
        except Exception as e:
            self.logger.error(f"Error merging articles: {e}")
            return []
    
    def _get_latest_timestamp(self, timestamps: List[str]) -> str:
        """Get the latest timestamp from a list"""
        try:
            latest = None
            latest_dt = None
            
            for timestamp in timestamps:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                        latest = timestamp
                except (ValueError, TypeError):
                    continue
            
            return latest or timestamps[0] if timestamps else ""
            
        except Exception as e:
            self.logger.error(f"Error getting latest timestamp: {e}")
            return timestamps[0] if timestamps else ""
    
    def _select_most_complete_value(self, values: List[Any]) -> Any:
        """Select the most complete value from a list"""
        if not values:
            return None
        
        # Return the longest non-empty string, or the first value
        string_values = [str(v) for v in values if v]
        if string_values:
            return max(string_values, key=len)
        
        return values[0]
    
    def create_version_history(self, items: List[Any], primary_item: Any) -> Dict[str, Any]:
        """
        Create version history for duplicate items
        
        Args:
            items: List of all items in the duplicate group
            primary_item: The selected primary item
            
        Returns:
            Dict[str, Any]: Version history information
        """
        try:
            history = {
                'total_versions': len(items),
                'primary_version': self._get_item_identifier(primary_item),
                'version_details': [],
                'created_at': datetime.now().isoformat()
            }
            
            for i, item in enumerate(items):
                version_info = {
                    'version_number': i + 1,
                    'is_primary': item == primary_item,
                    'identifier': self._get_item_identifier(item),
                    'quality_score': self._get_quality_score(item),
                    'completeness_score': self._get_completeness_score(item),
                    'recency_score': self._get_recency_score(item),
                    'timestamp': self._get_item_timestamp(item)
                }
                history['version_details'].append(version_info)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error creating version history: {e}")
            return {'total_versions': len(items), 'error': str(e)}
    
    def _get_item_identifier(self, item: Any) -> str:
        """Get unique identifier for an item"""
        if isinstance(item, dict):
            # Try different identifier fields
            for field in ['law_id', 'promulgation_number', 'law_name']:
                if field in item and item[field]:
                    return str(item[field])
        
        return str(hash(str(item)))
    
    def _get_item_timestamp(self, item: Any) -> str:
        """Get timestamp for an item"""
        if isinstance(item, dict):
            timestamp_fields = ['processing_timestamp', 'created_at', 'updated_at']
            for field in timestamp_fields:
                if field in item and item[field]:
                    return str(item[field])
        
        return datetime.now().isoformat()
    
    def add_resolution_strategy(self, strategy: ResolutionStrategy):
        """Add a custom resolution strategy"""
        self.strategies[strategy.name] = strategy
        self.logger.info(f"Added resolution strategy: {strategy.name}")
    
    def get_resolution_strategies(self) -> Dict[str, ResolutionStrategy]:
        """Get all available resolution strategies"""
        return self.strategies.copy()
    
    def export_resolution_report(self, results: List[ResolutionResult], output_path: str):
        """Export resolution report to file"""
        try:
            report = {
                'resolution_summary': {
                    'total_groups_resolved': len(results),
                    'groups_requiring_review': sum(1 for r in results if r.manual_review_required),
                    'average_confidence': sum(r.confidence_score for r in results) / len(results) if results else 0,
                    'generated_at': datetime.now().isoformat()
                },
                'resolution_details': [asdict(result) for result in results]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Resolution report exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting resolution report: {e}")


def resolve_duplicates_intelligent(duplicate_groups: List[List[Any]], strategy: str = 'quality_based') -> List[ResolutionResult]:
    """
    Convenience function to resolve duplicates intelligently
    
    Args:
        duplicate_groups: List of duplicate groups
        strategy: Resolution strategy name
        
    Returns:
        List[ResolutionResult]: Resolution results
    """
    resolver = IntelligentDuplicateResolver()
    return resolver.resolve_duplicates(duplicate_groups, strategy)


if __name__ == "__main__":
    # Test the duplicate resolver
    test_duplicates = [
        [
            {
                'law_name': '민법',
                'quality_score': 0.8,
                'articles': [{'article_number': '1', 'content': '민사 기본�?}],
                'processing_timestamp': '2025-01-01T00:00:00'
            },
            {
                'law_name': '민법',
                'quality_score': 0.6,
                'articles': [{'article_number': '1', 'content': '민사 기본�?}],
                'processing_timestamp': '2025-01-02T00:00:00'
            }
        ]
    ]
    
    resolver = IntelligentDuplicateResolver()
    results = resolver.resolve_duplicates(test_duplicates, 'quality_based')
    
    print(f"Resolved {len(results)} duplicate groups:")
    for result in results:
        print(f"Primary: {result.primary_item['law_name']}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Manual review required: {result.manual_review_required}")

