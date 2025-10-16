"""
Assembly Law Data Parsers

This module contains parsers for preprocessing raw Assembly law data
into clean, structured, searchable format for database storage and vector embedding.
Includes comprehensive legal analysis components for Korean law structure.
"""

from .html_parser import LawHTMLParser
from .article_parser import ArticleParser
from .metadata_extractor import MetadataExtractor
from .text_normalizer import TextNormalizer

# Legal analysis components (optional imports)
try:
    from .version_detector import DataVersionDetector
    from .version_parsers import VersionParserRegistry
    from .legal_hierarchy_classifier import LegalHierarchyClassifier
    from .legal_field_classifier import LegalFieldClassifier
    from .legal_structure_parser import LegalStructureParser
    from .comprehensive_legal_analyzer import ComprehensiveLegalAnalyzer
    
    LEGAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    LEGAL_ANALYSIS_AVAILABLE = False
    print(f"Legal analysis components not available: {e}")

__all__ = [
    'LawHTMLParser',
    'ArticleParser', 
    'MetadataExtractor',
    'TextNormalizer',
    'LEGAL_ANALYSIS_AVAILABLE'
]

# Add legal analysis components to __all__ if available
if LEGAL_ANALYSIS_AVAILABLE:
    __all__.extend([
        'DataVersionDetector',
        'VersionParserRegistry',
        'LegalHierarchyClassifier',
        'LegalFieldClassifier',
        'LegalStructureParser',
        'ComprehensiveLegalAnalyzer'
    ])
