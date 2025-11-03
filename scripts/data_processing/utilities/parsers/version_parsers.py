"""
Version-Specific Parsers for Assembly Law Data

This module provides version-specific parsers for different data formats
collected at different times.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VersionParserRegistry:
    """버전�??�서 ?�록 �?관�?""
    
    def __init__(self):
        """Initialize parser registry"""
        self.parsers = {
            'v1.0': V1_0Parser(),
            'v1.1': V1_1Parser(),
            'v1.2': V1_2Parser()
        }
        self.default_version = 'v1.2'
    
    def get_parser(self, version: str):
        """
        버전???�당?�는 ?�서 반환
        
        Args:
            version (str): Version identifier
            
        Returns:
            Parser instance for the specified version
        """
        return self.parsers.get(version, self.parsers[self.default_version])
    
    def get_supported_versions(self) -> List[str]:
        """
        지?�되??버전 목록 반환
        
        Returns:
            List[str]: List of supported versions
        """
        return list(self.parsers.keys())
    
    def register_parser(self, version: str, parser):
        """
        ?�로??버전 ?�서 ?�록
        
        Args:
            version (str): Version identifier
            parser: Parser instance
        """
        self.parsers[version] = parser
        logger.info(f"Registered parser for version {version}")


class BaseVersionParser:
    """기본 버전 ?�서 ?�래??""
    
    def parse(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Raw ?�이???�싱 (기본 구현)
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            
        Returns:
            Dict[str, Any]: Parsed data
        """
        return {
            'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
            'law_name': raw_data.get('law_name', ''),
            'law_content': raw_data.get('law_content', ''),
            'content_html': raw_data.get('content_html', ''),
            'parsing_version': self.get_version(),
            'parsed_at': datetime.now().isoformat()
        }
    
    def get_version(self) -> str:
        """?�서 버전 반환"""
        return 'base'


class V1_0Parser(BaseVersionParser):
    """버전 1.0 ?�이???�서 (기본 ?�식)"""
    
    def parse(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        v1.0 ?�식???�이???�싱
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            
        Returns:
            Dict[str, Any]: Parsed data
        """
        try:
            parsed_data = {
                'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
                'law_name': raw_data.get('law_name', ''),
                'law_content': raw_data.get('law_content', ''),
                'content_html': raw_data.get('content_html', ''),
                
                # v1.0 기본 메�??�이??
                'basic_metadata': {
                    'category': raw_data.get('category', ''),
                    'law_type': raw_data.get('law_type', ''),
                    'row_number': raw_data.get('row_number', '')
                },
                
                # 버전 ?�보
                'parsing_version': 'v1.0',
                'parsed_at': datetime.now().isoformat(),
                
                # v1.0?�서??공포 ?�보가 ?�으므�?�?값으�??�정
                'promulgation_info': {
                    'number': '',
                    'date': '',
                    'enforcement_date': '',
                    'amendment_type': ''
                },
                
                # ?�집 ?�보???�으므�?�?값으�??�정
                'collection_info': {
                    'cont_id': '',
                    'cont_sid': '',
                    'detail_url': '',
                    'collected_at': ''
                }
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing v1.0 data: {e}")
            return self._get_error_result(raw_data, 'v1.0', str(e))
    
    def get_version(self) -> str:
        """?�서 버전 반환"""
        return 'v1.0'


class V1_1Parser(BaseVersionParser):
    """버전 1.1 ?�이???�서 (공포 ?�보 추�?)"""
    
    def parse(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        v1.1 ?�식???�이???�싱
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            
        Returns:
            Dict[str, Any]: Parsed data
        """
        try:
            parsed_data = {
                'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
                'law_name': raw_data.get('law_name', ''),
                'law_content': raw_data.get('law_content', ''),
                'content_html': raw_data.get('content_html', ''),
                
                # v1.1 공포 ?�보
                'promulgation_info': {
                    'number': raw_data.get('promulgation_number', ''),
                    'date': raw_data.get('promulgation_date', ''),
                    'enforcement_date': raw_data.get('enforcement_date', ''),
                    'amendment_type': ''  # v1.1?�서???�정 ?�보 ?�음
                },
                
                # 기본 메�??�이??
                'basic_metadata': {
                    'category': raw_data.get('category', ''),
                    'law_type': raw_data.get('law_type', ''),
                    'row_number': raw_data.get('row_number', '')
                },
                
                # v1.1?�서???�집 ?�보가 ?�으므�?�?값으�??�정
                'collection_info': {
                    'cont_id': '',
                    'cont_sid': '',
                    'detail_url': '',
                    'collected_at': ''
                },
                
                # 버전 ?�보
                'parsing_version': 'v1.1',
                'parsed_at': datetime.now().isoformat()
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing v1.1 data: {e}")
            return self._get_error_result(raw_data, 'v1.1', str(e))
    
    def get_version(self) -> str:
        """?�서 버전 반환"""
        return 'v1.1'


class V1_2Parser(BaseVersionParser):
    """버전 1.2 ?�이???�서 (?�재 구현, 모든 ?�보 ?�함)"""
    
    def parse(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        v1.2 ?�식???�이???�싱
        
        Args:
            raw_data (Dict[str, Any]): Raw law data
            
        Returns:
            Dict[str, Any]: Parsed data
        """
        try:
            parsed_data = {
                'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
                'law_name': raw_data.get('law_name', ''),
                'law_content': raw_data.get('law_content', ''),
                'content_html': raw_data.get('content_html', ''),
                
                # v1.2 ?�전??공포 ?�보
                'promulgation_info': {
                    'number': raw_data.get('promulgation_number', ''),
                    'date': raw_data.get('promulgation_date', ''),
                    'enforcement_date': raw_data.get('enforcement_date', ''),
                    'amendment_type': raw_data.get('amendment_type', '')
                },
                
                # v1.2 ?�집 ?�보
                'collection_info': {
                    'cont_id': raw_data.get('cont_id', ''),
                    'cont_sid': raw_data.get('cont_sid', ''),
                    'detail_url': raw_data.get('detail_url', ''),
                    'collected_at': raw_data.get('collected_at', '')
                },
                
                # 기본 메�??�이??
                'basic_metadata': {
                    'category': raw_data.get('category', ''),
                    'law_type': raw_data.get('law_type', ''),
                    'row_number': raw_data.get('row_number', '')
                },
                
                # 버전 ?�보
                'parsing_version': 'v1.2',
                'parsed_at': datetime.now().isoformat()
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing v1.2 data: {e}")
            return self._get_error_result(raw_data, 'v1.2', str(e))
    
    def get_version(self) -> str:
        """?�서 버전 반환"""
        return 'v1.2'
    
    def _get_error_result(self, raw_data: Dict[str, Any], version: str, error: str) -> Dict[str, Any]:
        """
        ?�러 발생 ??기본 결과 반환
        
        Args:
            raw_data (Dict[str, Any]): Raw data
            version (str): Version identifier
            error (str): Error message
            
        Returns:
            Dict[str, Any]: Error result
        """
        return {
            'law_id': f"assembly_law_{raw_data.get('row_number', 'unknown')}",
            'law_name': raw_data.get('law_name', ''),
            'law_content': raw_data.get('law_content', ''),
            'content_html': raw_data.get('content_html', ''),
            'parsing_version': version,
            'parsed_at': datetime.now().isoformat(),
            'parsing_error': error,
            'promulgation_info': {
                'number': '',
                'date': '',
                'enforcement_date': '',
                'amendment_type': ''
            },
            'collection_info': {
                'cont_id': '',
                'cont_sid': '',
                'detail_url': '',
                'collected_at': ''
            },
            'basic_metadata': {
                'category': '',
                'law_type': '',
                'row_number': ''
            }
        }


class VersionCompatibilityChecker:
    """버전 ?�환??검?�기"""
    
    def __init__(self):
        """Initialize compatibility checker"""
        self.compatibility_matrix = {
            'v1.0': {'v1.1': True, 'v1.2': True},
            'v1.1': {'v1.2': True},
            'v1.2': {}
        }
    
    def is_compatible(self, from_version: str, to_version: str) -> bool:
        """
        버전�??�환???�인
        
        Args:
            from_version (str): Source version
            to_version (str): Target version
            
        Returns:
            bool: Compatibility status
        """
        if from_version == to_version:
            return True
        
        return self.compatibility_matrix.get(from_version, {}).get(to_version, False)
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """
        마이그레?�션 경로 반환
        
        Args:
            from_version (str): Source version
            to_version (str): Target version
            
        Returns:
            List[str]: Migration path
        """
        if from_version == to_version:
            return [from_version]
        
        if self.is_compatible(from_version, to_version):
            return [from_version, to_version]
        
        # 직접 ?�환?��? ?�는 경우 중간 버전???�한 경로 찾기
        path = [from_version]
        current_version = from_version
        
        while current_version != to_version:
            next_versions = self.compatibility_matrix.get(current_version, {})
            if not next_versions:
                return []  # 경로 ?�음
            
            # 가??가까운 ?�음 버전 ?�택
            next_version = min(next_versions.keys())
            path.append(next_version)
            current_version = next_version
        
        return path
