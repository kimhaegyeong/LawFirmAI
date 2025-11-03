#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
?�중 법률 문서 ?�싱 ?�질 검???�크립트
규칙 기반 ?�서?� ML 강화 ?�서??결과�?비교 분석
"""

import json
import sys
import os
from pathlib import Path
import logging
import random
from typing import Dict, List, Any, Tuple

# Windows 콘솔?�서 UTF-8 ?�코???�정
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# ?�서 모듈 경로 추�?
sys.path.append(str(Path(__file__).parent / 'parsers'))

from ml_enhanced_parser import MLEnhancedArticleParser
from parsers.improved_article_parser import ImprovedArticleParser

# 로깅 ?�정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LawParsingQualityChecker:
    """법률 문서 ?�싱 ?�질 검???�래??""
    
    def __init__(self):
        self.rule_parser = ImprovedArticleParser()
        self.ml_parser = MLEnhancedArticleParser()
        
    def load_raw_law_data(self, sample_size: int = 20) -> List[Dict[str, Any]]:
        """?�본 법률 ?�이???�플 로드"""
        raw_files = []
        
        # ?�본 ?�이???�렉?�리??
        raw_dirs = [
            "data/raw/assembly/law/20251010",
            "data/raw/assembly/law/20251011", 
            "data/raw/assembly/law/20251012",
            "data/raw/assembly/law/2025101201"
        ]
        
        for raw_dir in raw_dirs:
            if Path(raw_dir).exists():
                files = list(Path(raw_dir).glob("*.json"))
                raw_files.extend(files)
        
        # ?�덤 ?�플�?
        if len(raw_files) > sample_size:
            raw_files = random.sample(raw_files, sample_size)
        
        logger.info(f"로드???�본 ?�일 ?? {len(raw_files)}")
        
        laws_data = []
        for file_path in raw_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'laws' in data:
                    for law in data['laws']:
                        if law.get('law_content'):
                            laws_data.append({
                                'file_path': str(file_path),
                                'law_name': law.get('law_name', 'Unknown'),
                                'law_content': law['law_content'],
                                'law_type': law.get('law_type', 'Unknown'),
                                'cont_id': law.get('cont_id', 'Unknown')
                            })
            except Exception as e:
                logger.warning(f"?�일 로드 ?�패 {file_path}: {e}")
                continue
        
        logger.info(f"추출??법률 문서 ?? {len(laws_data)}")
        return laws_data
    
    def compare_parsing_results(self, law_data: Dict[str, Any]) -> Dict[str, Any]:
        """?�일 법률 문서???�싱 결과 비교"""
        
        law_content = law_data['law_content']
        
        # 규칙 기반 ?�서 결과
        try:
            rule_result = self.rule_parser.parse_law_document(law_content)
        except Exception as e:
            logger.error(f"규칙 기반 ?�싱 ?�패 {law_data['law_name']}: {e}")
            rule_result = {'total_articles': 0, 'all_articles': []}
        
        # ML 강화 ?�서 결과
        try:
            ml_result = self.ml_parser.parse_law_document(law_content)
        except Exception as e:
            logger.error(f"ML 강화 ?�싱 ?�패 {law_data['law_name']}: {e}")
            ml_result = {'total_articles': 0, 'all_articles': []}
        
        # 비교 분석
        comparison = {
            'law_name': law_data['law_name'],
            'law_type': law_data['law_type'],
            'cont_id': law_data['cont_id'],
            'rule_based': {
                'total_articles': rule_result['total_articles'],
                'articles_with_titles': sum(1 for a in rule_result['all_articles'] if a.get('article_title')),
                'total_paragraphs': sum(len(a.get('sub_articles', [])) for a in rule_result['all_articles']),
                'article_numbers': [a['article_number'] for a in rule_result['all_articles']]
            },
            'ml_enhanced': {
                'total_articles': ml_result['total_articles'],
                'articles_with_titles': sum(1 for a in ml_result['all_articles'] if a.get('article_title')),
                'total_paragraphs': sum(len(a.get('sub_articles', [])) for a in ml_result['all_articles']),
                'article_numbers': [a['article_number'] for a in ml_result['all_articles']]
            }
        }
        
        # 차이??분석
        comparison['differences'] = {
            'article_count_diff': ml_result['total_articles'] - rule_result['total_articles'],
            'title_coverage_diff': comparison['ml_enhanced']['articles_with_titles'] - comparison['rule_based']['articles_with_titles'],
            'paragraph_count_diff': comparison['ml_enhanced']['total_paragraphs'] - comparison['rule_based']['total_paragraphs'],
            'missing_articles': set(comparison['rule_based']['article_numbers']) - set(comparison['ml_enhanced']['article_numbers']),
            'extra_articles': set(comparison['ml_enhanced']['article_numbers']) - set(comparison['rule_based']['article_numbers'])
        }
        
        return comparison
    
    def analyze_parsing_quality(self, sample_size: int = 20) -> Dict[str, Any]:
        """?�싱 ?�질 종합 분석"""
        
        logger.info("?�본 법률 ?�이??로드 �?..")
        laws_data = self.load_raw_law_data(sample_size)
        
        if not laws_data:
            logger.error("로드??법률 ?�이?��? ?�습?�다.")
            return {}
        
        logger.info("?�싱 결과 비교 분석 �?..")
        comparisons = []
        
        for i, law_data in enumerate(laws_data):
            logger.info(f"처리 �?({i+1}/{len(laws_data)}): {law_data['law_name']}")
            
            try:
                comparison = self.compare_parsing_results(law_data)
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"비교 분석 ?�패 {law_data['law_name']}: {e}")
                continue
        
        # 종합 ?�계
        total_laws = len(comparisons)
        if total_laws == 0:
            return {}
        
        # ?�계 계산
        stats = {
            'total_laws_analyzed': total_laws,
            'article_count_differences': [],
            'title_coverage_differences': [],
            'paragraph_count_differences': [],
            'laws_with_missing_articles': 0,
            'laws_with_extra_articles': 0,
            'laws_with_issues': 0,
            'problematic_laws': []
        }
        
        for comp in comparisons:
            diff = comp['differences']
            
            stats['article_count_differences'].append(diff['article_count_diff'])
            stats['title_coverage_differences'].append(diff['title_coverage_diff'])
            stats['paragraph_count_differences'].append(diff['paragraph_count_diff'])
            
            if diff['missing_articles']:
                stats['laws_with_missing_articles'] += 1
                stats['laws_with_issues'] += 1
                stats['problematic_laws'].append({
                    'law_name': comp['law_name'],
                    'issue_type': 'missing_articles',
                    'missing': list(diff['missing_articles']),
                    'rule_count': comp['rule_based']['total_articles'],
                    'ml_count': comp['ml_enhanced']['total_articles']
                })
            
            if diff['extra_articles']:
                stats['laws_with_extra_articles'] += 1
                if comp['law_name'] not in [p['law_name'] for p in stats['problematic_laws']]:
                    stats['laws_with_issues'] += 1
                    stats['problematic_laws'].append({
                        'law_name': comp['law_name'],
                        'issue_type': 'extra_articles',
                        'extra': list(diff['extra_articles']),
                        'rule_count': comp['rule_based']['total_articles'],
                        'ml_count': comp['ml_enhanced']['total_articles']
                    })
        
        # ?�균 계산
        stats['avg_article_count_diff'] = sum(stats['article_count_differences']) / total_laws
        stats['avg_title_coverage_diff'] = sum(stats['title_coverage_differences']) / total_laws
        stats['avg_paragraph_count_diff'] = sum(stats['paragraph_count_differences']) / total_laws
        
        return {
            'statistics': stats,
            'detailed_comparisons': comparisons
        }
    
    def generate_quality_report(self, analysis_result: Dict[str, Any]) -> str:
        """?�질 분석 리포???�성"""
        
        if not analysis_result:
            return "분석 결과가 ?�습?�다."
        
        stats = analysis_result['statistics']
        comparisons = analysis_result['detailed_comparisons']
        
        report = []
        report.append("=" * 80)
        report.append("법률 문서 ?�싱 ?�질 분석 리포??)
        report.append("=" * 80)
        report.append("")
        
        # ?�체 ?�계
        report.append("1. ?�체 ?�계")
        report.append("-" * 40)
        report.append(f"분석??법률 문서 ?? {stats['total_laws_analyzed']}")
        report.append(f"문제가 ?�는 법률 문서 ?? {stats['laws_with_issues']}")
        report.append(f"문제 비율: {stats['laws_with_issues']/stats['total_laws_analyzed']*100:.1f}%")
        report.append("")
        
        # ?�균 차이
        report.append("2. ?�균 차이 (ML 강화 - 규칙 기반)")
        report.append("-" * 40)
        report.append(f"조문 ??차이: {stats['avg_article_count_diff']:+.2f}")
        report.append(f"?�목 추출 차이: {stats['avg_title_coverage_diff']:+.2f}")
        report.append(f"????차이: {stats['avg_paragraph_count_diff']:+.2f}")
        report.append("")
        
        # 문제 ?�형�??�계
        report.append("3. 문제 ?�형�??�계")
        report.append("-" * 40)
        report.append(f"?�락??조문???�는 법률: {stats['laws_with_missing_articles']}�?)
        report.append(f"추�???조문???�는 법률: {stats['laws_with_extra_articles']}�?)
        report.append("")
        
        # 문제가 ?�는 법률 목록
        if stats['problematic_laws']:
            report.append("4. 문제가 ?�는 법률 목록")
            report.append("-" * 40)
            
            for i, problem in enumerate(stats['problematic_laws'], 1):
                report.append(f"{i}. {problem['law_name']}")
                report.append(f"   문제 ?�형: {problem['issue_type']}")
                if problem['issue_type'] == 'missing_articles':
                    report.append(f"   ?�락??조문: {problem['missing']}")
                else:
                    report.append(f"   추�???조문: {problem['extra']}")
                report.append(f"   규칙 기반 조문 ?? {problem['rule_count']}")
                report.append(f"   ML 강화 조문 ?? {problem['ml_count']}")
                report.append("")
        
        # ?�세 비교 결과 (처음 5개만)
        report.append("5. ?�세 비교 결과 (처음 5�?")
        report.append("-" * 40)
        
        for i, comp in enumerate(comparisons[:5]):
            report.append(f"{i+1}. {comp['law_name']}")
            report.append(f"   규칙 기반: {comp['rule_based']['total_articles']}�?조문, {comp['rule_based']['articles_with_titles']}�??�목")
            report.append(f"   ML 강화: {comp['ml_enhanced']['total_articles']}�?조문, {comp['ml_enhanced']['articles_with_titles']}�??�목")
            
            diff = comp['differences']
            if diff['article_count_diff'] != 0:
                report.append(f"   조문 ??차이: {diff['article_count_diff']:+d}")
            if diff['missing_articles']:
                report.append(f"   ?�락??조문: {list(diff['missing_articles'])}")
            if diff['extra_articles']:
                report.append(f"   추�???조문: {list(diff['extra_articles'])}")
            report.append("")
        
        # 결론
        report.append("6. 결론 �?권장?�항")
        report.append("-" * 40)
        
        if stats['laws_with_issues'] == 0:
            report.append("[OK] 모든 법률 문서가 ?�바르게 ?�싱?�었?�니??")
        elif stats['laws_with_issues'] / stats['total_laws_analyzed'] < 0.1:
            report.append("[WARNING] ?�수??법률 문서?�서 ?�싱 문제가 발견?�었?�니??")
        else:
            report.append("[ERROR] ?�당?�의 법률 문서?�서 ?�싱 문제가 발견?�었?�니??")
        
        if stats['avg_article_count_diff'] < 0:
            report.append("- ML 강화 ?�서가 조문??과도?�게 ?�터링하�??�습?�다.")
            report.append("- ML 모델???�계값을 ??��거나 규칙 기반 ?�터링을 강화?�야 ?�니??")
        
        if stats['laws_with_missing_articles'] > 0:
            report.append("- ?��? 조문???�락?�는 문제가 ?�습?�다.")
            report.append("- 조문 경계 감�? 로직??개선?�야 ?�니??")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """메인 ?�수"""
    print("법률 문서 ?�싱 ?�질 검??)
    print("=" * 50)
    
    checker = LawParsingQualityChecker()
    
    # ?�질 분석 ?�행
    print("1. ?�싱 ?�질 분석 �?..")
    analysis_result = checker.analyze_parsing_quality(sample_size=30)
    
    if not analysis_result:
        print("분석???�이?��? ?�습?�다.")
        return
    
    # 리포???�성
    print("2. 분석 리포???�성 �?..")
    report = checker.generate_quality_report(analysis_result)
    
    # 리포??출력
    print("\n" + report)
    
    # 리포???�??
    report_path = "law_parsing_quality_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n?�질 분석 리포?��? ?�?�되?�습?�다: {report_path}")
    
    # 간단???�약
    stats = analysis_result['statistics']
    print(f"\n=== ?�약 ===")
    print(f"분석??법률 문서: {stats['total_laws_analyzed']}�?)
    print(f"문제가 ?�는 문서: {stats['laws_with_issues']}�?({stats['laws_with_issues']/stats['total_laws_analyzed']*100:.1f}%)")
    print(f"?�균 조문 ??차이: {stats['avg_article_count_diff']:+.2f}")


if __name__ == "__main__":
    main()
