
# -*- coding: utf-8 -*-
"""
법률 ?�어 추출 ?�스??
기존 ?�이?�에??법률 ?�어�??�동?�로 추출?�는 ?�스??
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import glob

logger = logging.getLogger(__name__)


class LegalTermExtractor:
    """법률 ?�어 추출�?""
    
    def __init__(self):
        """초기??""
        self.logger = logging.getLogger(__name__)
        
        # 법률 ?�어 ?�턴 ?�의
        self.legal_term_patterns = {
            'legal_concepts': r'[가-??{2,6}(?:�?�?책임|?�해|계약|?�송|처벌|?�재)',
            'legal_actions': r'[가-??{2,4}(?:배상|보상|�?��|?�기|?��?|?�반|침해|처리)',
            'legal_procedures': r'[가-??{2,5}(?:?�차|?�청|?�리|?�결|??��|?�고|조정|?�해)',
            'legal_entities': r'[가-??{2,4}(?:법원|검??변?�사|?�고???�고|?�고|증인|감정??',
            'legal_documents': r'[가-??{2,5}(?:?�장|?��???증거|?�결??조서|진술??진술조서)',
            'legal_penalties': r'[가-??{2,4}(?:?�벌|벌금|징역|금고|집행?�예|?�고?�예|보호관�?',
            'legal_rights': r'[가-??{2,5}(?:?�유�??�유�?지?�권|?�세�??�?�권|질권|?�치�?',
            'legal_relationships': r'[가-??{2,4}(?:?�인|?�혼|?�속|?�육|친권|?�육�?면접교섭�?',
            'legal_obligations': r'[가-??{2,4}(?:채무|채권|?�행|불이???�약|?�해|과실|고의)',
            'legal_crimes': r'[가-??{2,5}(?:?�인|강도|?�도|?�기|?�령|배임|?�물|강간|?�폭??'
        }
        
        # 법률 조문 ?�턴
        self.law_article_patterns = [
            r'??d+�?,  # ??50�?
            r'??d+??,  # ????
            r'??d+??,  # ????
            r'법률\s*??d+??,  # 법률 ??0883??
            r'?�행??s*??d+�?,  # ?�행????�?
            r'?�행규칙\s*??d+�?  # ?�행규칙 ??�?
        ]
        
        # ?��? ?�턴
        self.precedent_patterns = [
            r'\d{4}[가?�다?�마바사?�자차카?�?�하]\d+',  # 2023??2345
            r'?�법원\s*\d{4}\.\d+\.\d+',  # ?�법원 2023.1.1
            r'고등법원\s*\d{4}\.\d+\.\d+',  # 고등법원 2023.1.1
            r'지방법??s*\d{4}\.\d+\.\d+'  # 지방법??2023.1.1
        ]
        
        # 추출???�어 ?�??
        self.extracted_terms = defaultdict(list)
        self.term_frequencies = Counter()
        self.term_contexts = defaultdict(list)
        
    def extract_text_from_json(self, data: Dict) -> str:
        """JSON ?�이?�에???�스??추출"""
        text_parts = []
        
        if isinstance(data, dict):
            # 법령 ?�이??처리
            if 'laws' in data:
                for law in data['laws']:
                    if 'articles' in law:
                        for article in law['articles']:
                            if 'article_content' in article:
                                text_parts.append(article['article_content'])
                            if 'article_title' in article:
                                text_parts.append(article['article_title'])
            
            # ?��? ?�이??처리
            elif 'cases' in data:
                for case in data['cases']:
                    if 'full_text' in case:
                        text_parts.append(case['full_text'])
                    if 'searchable_text' in case:
                        text_parts.append(case['searchable_text'])
                    if 'case_name' in case:
                        text_parts.append(case['case_name'])
            
            # 기�? ?�스???�드??
            text_fields = ['content', 'text', 'title', 'description', 'summary']
            for field in text_fields:
                if field in data and isinstance(data[field], str):
                    text_parts.append(data[field])
        
        return ' '.join(text_parts)
    
    def extract_terms_from_text(self, text: str, file_path: str = "") -> Dict[str, List[str]]:
        """?�스?�에??법률 ?�어 추출"""
        extracted = defaultdict(list)
        
        for pattern_name, pattern in self.legal_term_patterns.items():
            matches = re.findall(pattern, text)
            extracted[pattern_name].extend(matches)
            
            # ?�어 빈도 �?컨텍?�트 ?�??
            for match in matches:
                self.term_frequencies[match] += 1
                if file_path:
                    self.term_contexts[match].append(file_path)
        
        return extracted
    
    def extract_law_articles(self, text: str) -> List[str]:
        """법률 조문 추출"""
        articles = []
        for pattern in self.law_article_patterns:
            matches = re.findall(pattern, text)
            articles.extend(matches)
        return articles
    
    def extract_precedents(self, text: str) -> List[str]:
        """?��? 추출"""
        precedents = []
        for pattern in self.precedent_patterns:
            matches = re.findall(pattern, text)
            precedents.extend(matches)
        return precedents
    
    def process_file(self, file_path: str) -> Dict:
        """?�일 ?�일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ?�스??추출
            text = self.extract_text_from_json(data)
            
            if not text.strip():
                return {}
            
            # ?�어 추출
            extracted_terms = self.extract_terms_from_text(text, file_path)
            
            # 법률 조문 �??��? 추출
            law_articles = self.extract_law_articles(text)
            precedents = self.extract_precedents(text)
            
            result = {
                'file_path': file_path,
                'extracted_terms': extracted_terms,
                'law_articles': law_articles,
                'precedents': precedents,
                'text_length': len(text),
                'term_count': sum(len(terms) for terms in extracted_terms.values())
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {}
    
    def process_directory(self, directory_path: str, min_frequency: int = 5) -> Dict:
        """?�렉?�리 ?�체 처리"""
        self.logger.info(f"Processing directory: {directory_path}")
        
        # JSON ?�일 찾기
        json_files = glob.glob(f"{directory_path}/**/*.json", recursive=True)
        self.logger.info(f"Found {len(json_files)} JSON files")
        
        processed_files = 0
        total_terms = 0
        
        for file_path in json_files:
            result = self.process_file(file_path)
            if result:
                processed_files += 1
                total_terms += result['term_count']
                
                # 결과 ?�합
                for pattern_name, terms in result['extracted_terms'].items():
                    self.extracted_terms[pattern_name].extend(terms)
        
        # 빈도 기반 ?�터�?
        filtered_terms = self._filter_terms_by_frequency(min_frequency)
        
        # 결과 ?�리
        final_result = {
            'processed_files': processed_files,
            'total_files': len(json_files),
            'total_terms_extracted': total_terms,
            'filtered_terms': filtered_terms,
            'term_frequencies': dict(self.term_frequencies.most_common(100)),
            'extraction_summary': self._generate_summary()
        }
        
        self.logger.info(f"Processing completed: {processed_files}/{len(json_files)} files processed")
        self.logger.info(f"Total terms extracted: {total_terms}")
        self.logger.info(f"Terms after filtering: {sum(len(terms) for terms in filtered_terms.values())}")
        
        return final_result
    
    def _filter_terms_by_frequency(self, min_frequency: int) -> Dict[str, List[str]]:
        """빈도 기반 ?�어 ?�터�?""
        filtered_terms = defaultdict(list)
        
        for pattern_name, terms in self.extracted_terms.items():
            # 빈도가 ?��? ?�어�??�택
            frequent_terms = [term for term in terms 
                            if self.term_frequencies[term] >= min_frequency]
            
            # 중복 ?�거 �??�렬
            unique_terms = sorted(list(set(frequent_terms)))
            filtered_terms[pattern_name] = unique_terms
        
        return filtered_terms
    
    def _generate_summary(self) -> Dict:
        """추출 결과 ?�약 ?�성"""
        summary = {}
        
        for pattern_name, terms in self.extracted_terms.items():
            unique_terms = list(set(terms))
            summary[pattern_name] = {
                'total_count': len(terms),
                'unique_count': len(unique_terms),
                'top_terms': sorted(unique_terms, 
                                   key=lambda x: self.term_frequencies[x], 
                                   reverse=True)[:10]
            }
        
        return summary
    
    def save_results(self, output_path: str, results: Dict):
        """결과 ?�??""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def main():
    """메인 ?�행 ?�수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='법률 ?�어 추출�?)
    parser.add_argument('--input_dir', type=str, required=True,
                       help='?�력 ?�렉?�리 경로')
    parser.add_argument('--output_file', type=str, 
                       default='data/extracted_terms/raw_extracted_terms.json',
                       help='출력 ?�일 경로')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='최소 빈도 (기본�? 5)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 ?�벨')
    
    args = parser.parse_args()
    
    # 로깅 ?�정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ?�어 추출 ?�행
    extractor = LegalTermExtractor()
    results = extractor.process_directory(args.input_dir, args.min_frequency)
    
    # 결과 ?�??
    extractor.save_results(args.output_file, results)
    
    # 결과 출력
    print("\n=== 법률 ?�어 추출 결과 ===")
    print(f"처리???�일: {results['processed_files']}/{results['total_files']}")
    print(f"추출??�??�어 ?? {results['total_terms_extracted']}")
    print(f"?�터�????�어 ?? {sum(len(terms) for terms in results['filtered_terms'].values())}")
    
    print("\n=== ?�턴�??�어 ??===")
    for pattern_name, terms in results['filtered_terms'].items():
        print(f"{pattern_name}: {len(terms)}�?)
    
    print("\n=== ?�위 빈도 ?�어 (?�위 10�? ===")
    for term, freq in list(results['term_frequencies'].items())[:10]:
        print(f"{term}: {freq}??)


if __name__ == "__main__":
    main()
