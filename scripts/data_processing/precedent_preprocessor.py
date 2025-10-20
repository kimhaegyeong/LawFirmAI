#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
판례 데이터 전처리기

판례 데이터의 구조화된 내용을 파싱하고 정리하는 전용 전처리기입니다.
법률 데이터와는 다른 구조를 가지는 판례 데이터를 처리합니다.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class PrecedentPreprocessor:
    """판례 데이터 전처리기 클래스"""
    
    def __init__(self, enable_term_normalization: bool = True):
        """
        판례 전처리기 초기화
        
        Args:
            enable_term_normalization: 법률 용어 정규화 활성화
        """
        self.enable_term_normalization = enable_term_normalization
        
        # 판례 섹션 타입 정의
        self.section_types = {
            '판시사항': 'points_at_issue',
            '판결요지': 'decision_summary', 
            '참조조문': 'referenced_statutes',
            '참조판례': 'referenced_cases',
            '주문': 'disposition',
            '이유': 'reasoning'
        }
        
        logger.info("PrecedentPreprocessor initialized")
    
    def process_precedent_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        판례 데이터 전처리
        
        Args:
            raw_data: 원본 판례 데이터
            
        Returns:
            Dict[str, Any]: 전처리된 판례 데이터
        """
        try:
            metadata = raw_data.get('metadata', {})
            items = raw_data.get('items', [])
            
            processed_cases = []
            
            for item in items:
                processed_case = self._process_single_case(item, metadata)
                if processed_case:
                    processed_cases.append(processed_case)
            
            return {
                'metadata': {
                    'data_type': 'precedent',
                    'category': metadata.get('category', 'unknown'),
                    'processed_at': datetime.now().isoformat(),
                    'total_cases': len(processed_cases),
                    'processing_version': '1.0'
                },
                'cases': processed_cases
            }
            
        except Exception as e:
            logger.error(f"Error processing precedent data: {e}")
            return {'cases': []}
    
    def _process_single_case(self, case_item: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        단일 판례 케이스 처리
        
        Args:
            case_item: 판례 케이스 데이터
            metadata: 메타데이터
            
        Returns:
            Optional[Dict[str, Any]]: 처리된 케이스 데이터
        """
        try:
            # 기본 케이스 정보 추출
            case_id = self._generate_case_id(case_item)
            case_name = case_item.get('case_name', '')
            case_number = case_item.get('case_number', '')
            decision_date = case_item.get('decision_date', '')
            field = case_item.get('field', '')
            court = case_item.get('court', '')
            detail_url = case_item.get('detail_url', '')
            
            # 구조화된 내용 파싱
            structured_content = case_item.get('structured_content', {})
            case_info = structured_content.get('case_info', {})
            legal_sections = structured_content.get('legal_sections', {})
            parties = structured_content.get('parties', {})
            
            # 법률 섹션 추출
            sections = self._extract_legal_sections(legal_sections)
            
            # 당사자 정보 추출
            party_info = self._extract_party_info(parties)
            
            # 전체 텍스트 생성
            full_text = self._generate_full_text(case_item, legal_sections, parties)
            
            # 검색용 텍스트 생성
            searchable_text = self._generate_searchable_text(case_name, case_number, sections)
            
            processed_case = {
                'case_id': case_id,
                'category': metadata.get('category', field),
                'case_name': case_name,
                'case_number': case_number,
                'decision_date': decision_date,
                'field': field,
                'court': court,
                'detail_url': detail_url,
                'full_text': full_text,
                'searchable_text': searchable_text,
                'sections': sections,
                'parties': party_info,
                'case_info': case_info,
                'data_quality': {
                    'parsing_quality_score': self._calculate_quality_score(sections),
                    'content_length': len(full_text),
                    'section_count': len(sections)
                },
                'processed_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            return processed_case
            
        except Exception as e:
            logger.error(f"Error processing single case: {e}")
            return None
    
    def _generate_case_id(self, case_item: Dict[str, Any]) -> str:
        """케이스 ID 생성"""
        case_number = case_item.get('case_number', '')
        case_name = case_item.get('case_name', '')
        
        if case_number:
            return f"case_{case_number}"
        elif case_name:
            clean_name = re.sub(r'[^\w가-힣]', '_', case_name)
            return f"case_{clean_name}"
        else:
            return f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _extract_legal_sections(self, legal_sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        법률 섹션 추출
        
        Args:
            legal_sections: 구조화된 법률 섹션 데이터
            
        Returns:
            List[Dict[str, Any]]: 추출된 섹션 목록
        """
        sections = []
        
        for korean_name, english_key in self.section_types.items():
            content = legal_sections.get(korean_name, '')
            if content and content.strip():
                section = {
                    'section_type': english_key,
                    'section_type_korean': korean_name,
                    'section_content': self._clean_content(content),
                    'section_length': len(content),
                    'has_content': True
                }
                sections.append(section)
            else:
                # 빈 섹션도 포함 (구조적 일관성)
                section = {
                    'section_type': english_key,
                    'section_type_korean': korean_name,
                    'section_content': '',
                    'section_length': 0,
                    'has_content': False
                }
                sections.append(section)
        
        return sections
    
    def _extract_party_info(self, parties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        당사자 정보 추출
        
        Args:
            parties: 당사자 데이터
            
        Returns:
            List[Dict[str, Any]]: 당사자 정보 목록
        """
        party_list = []
        
        # 원고 정보
        plaintiff = parties.get('plaintiff', '')
        if plaintiff and plaintiff.strip():
            party_list.append({
                'party_type': 'plaintiff',
                'party_type_korean': '원고',
                'party_content': self._clean_content(plaintiff),
                'party_length': len(plaintiff)
            })
        
        # 피고 정보
        defendant = parties.get('defendant', '')
        if defendant and defendant.strip():
            party_list.append({
                'party_type': 'defendant',
                'party_type_korean': '피고',
                'party_content': self._clean_content(defendant),
                'party_length': len(defendant)
            })
        
        return party_list
    
    def _generate_full_text(self, case_item: Dict[str, Any], legal_sections: Dict[str, Any], parties: Dict[str, Any]) -> str:
        """전체 텍스트 생성"""
        text_parts = []
        
        # 케이스 기본 정보
        case_name = case_item.get('case_name', '')
        case_number = case_item.get('case_number', '')
        decision_date = case_item.get('decision_date', '')
        court = case_item.get('court', '')
        
        if case_name:
            text_parts.append(f"사건명: {case_name}")
        if case_number:
            text_parts.append(f"사건번호: {case_number}")
        if decision_date:
            text_parts.append(f"선고일: {decision_date}")
        if court:
            text_parts.append(f"법원: {court}")
        
        # 법률 섹션 내용
        for korean_name, english_key in self.section_types.items():
            content = legal_sections.get(korean_name, '')
            if content and content.strip():
                text_parts.append(f"{korean_name}: {content}")
        
        # 당사자 정보
        plaintiff = parties.get('plaintiff', '')
        defendant = parties.get('defendant', '')
        
        if plaintiff:
            text_parts.append(f"원고: {plaintiff}")
        if defendant:
            text_parts.append(f"피고: {defendant}")
        
        return '\n'.join(text_parts)
    
    def _generate_searchable_text(self, case_name: str, case_number: str, sections: List[Dict[str, Any]]) -> str:
        """검색용 텍스트 생성"""
        search_parts = [case_name, case_number]
        
        for section in sections:
            if section['has_content']:
                search_parts.append(section['section_content'])
        
        return ' '.join(search_parts)
    
    def _clean_content(self, content: str) -> str:
        """내용 정리"""
        if not content:
            return ""
        
        # HTML 태그 제거
        cleaned = re.sub(r'<[^>]+>', '', content)
        
        # 연속된 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 제어 문자 제거
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned.strip()
    
    def _calculate_quality_score(self, sections: List[Dict[str, Any]]) -> float:
        """파싱 품질 점수 계산"""
        if not sections:
            return 0.0
        
        total_sections = len(sections)
        filled_sections = sum(1 for section in sections if section['has_content'])
        
        # 기본 점수: 채워진 섹션 비율
        base_score = filled_sections / total_sections
        
        # 내용 길이 보너스
        total_length = sum(section['section_length'] for section in sections)
        length_bonus = min(total_length / 10000, 0.2)  # 최대 0.2 보너스
        
        return min(base_score + length_bonus, 1.0)


def main():
    """테스트용 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="판례 데이터 전처리기")
    parser.add_argument('--input-file', required=True, help='입력 파일 경로')
    parser.add_argument('--output-file', help='출력 파일 경로')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 전처리기 초기화
        preprocessor = PrecedentPreprocessor()
        
        # 파일 읽기
        with open(args.input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 전처리 실행
        processed_data = preprocessor.process_precedent_data(raw_data)
        
        # 결과 저장
        output_file = args.output_file or f"processed_{Path(args.input_file).name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Preprocessing completed. Output saved to: {output_file}")
        logger.info(f"Processed {processed_data['metadata']['total_cases']} cases")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
