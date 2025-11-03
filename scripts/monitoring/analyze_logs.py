#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
법률 ?�집 로그 분석 ?�구

?�집??로그 ?�일??분석?�여 ?�능 리포?��? ?�성?�니??
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class LogAnalyzer:
    """로그 분석�?""
    
    def __init__(self, log_dir: str = "data/raw/assembly"):
        self.log_dir = Path(log_dir)
        self.analysis_results = {}
    
    def analyze_collection_logs(self, date: str = None) -> Dict[str, Any]:
        """?�집 로그 분석"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        law_dir = self.log_dir / "law" / date
        
        if not law_dir.exists():
            print(f"No data found for date: {date}")
            return {}
        
        # ?�이지 ?�일??분석
        page_files = list(law_dir.glob("law_page_*.json"))
        
        if not page_files:
            print(f"No page files found in {law_dir}")
            return {}
        
        print(f"Analyzing {len(page_files)} page files...")
        
        # ?�이???�집
        pages_data = []
        laws_data = []
        
        for page_file in page_files:
            try:
                with open(page_file, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                
                page_info = page_data.get('page_info', {})
                laws = page_data.get('laws', [])
                
                # ?�이지 ?�보 ?�집
                pages_data.append({
                    'page_number': page_info.get('page_number', 'unknown'),
                    'laws_count': page_info.get('laws_count', 0),
                    'saved_at': page_info.get('saved_at', ''),
                    'processing_time': page_info.get('processing_time', 0),
                    'file_name': page_file.name
                })
                
                # 법률 ?�보 ?�집
                for law in laws:
                    laws_data.append({
                        'cont_id': law.get('cont_id', ''),
                        'law_name': law.get('law_name', ''),
                        'law_type': law.get('law_type', ''),
                        'category': law.get('category', ''),
                        'page_number': page_info.get('page_number', 'unknown'),
                        'collected_at': law.get('collected_at', '')
                    })
                
            except Exception as e:
                print(f"Error reading {page_file}: {e}")
                continue
        
        # ?�이?�프?�임 ?�성
        pages_df = pd.DataFrame(pages_data)
        laws_df = pd.DataFrame(laws_data)
        
        # 분석 ?�행
        analysis = self._perform_analysis(pages_df, laws_df)
        
        self.analysis_results[date] = analysis
        return analysis
    
    def _perform_analysis(self, pages_df: pd.DataFrame, laws_df: pd.DataFrame) -> Dict[str, Any]:
        """분석 ?�행"""
        analysis = {
            'summary': {},
            'performance': {},
            'content': {},
            'timeline': {}
        }
        
        # 기본 ?�계
        analysis['summary'] = {
            'total_pages': len(pages_df),
            'total_laws': len(laws_df),
            'avg_laws_per_page': laws_df.groupby('page_number').size().mean(),
            'total_processing_time': pages_df['processing_time'].sum(),
            'avg_processing_time': pages_df['processing_time'].mean(),
            'min_processing_time': pages_df['processing_time'].min(),
            'max_processing_time': pages_df['processing_time'].max()
        }
        
        # ?�능 분석
        analysis['performance'] = {
            'throughput_laws_per_minute': len(laws_df) / (pages_df['processing_time'].sum() / 60) if pages_df['processing_time'].sum() > 0 else 0,
            'processing_time_trend': pages_df['processing_time'].tolist(),
            'laws_per_page_trend': pages_df['laws_count'].tolist(),
            'efficiency_score': len(laws_df) / pages_df['processing_time'].sum() if pages_df['processing_time'].sum() > 0 else 0
        }
        
        # 콘텐�?분석
        analysis['content'] = {
            'law_types': laws_df['law_type'].value_counts().to_dict(),
            'categories': laws_df['category'].value_counts().to_dict(),
            'unique_laws': laws_df['law_name'].nunique(),
            'duplicate_laws': len(laws_df) - laws_df['law_name'].nunique()
        }
        
        # ?�?�라??분석
        if not pages_df.empty:
            pages_df['saved_at'] = pd.to_datetime(pages_df['saved_at'])
            analysis['timeline'] = {
                'start_time': pages_df['saved_at'].min().isoformat(),
                'end_time': pages_df['saved_at'].max().isoformat(),
                'duration_minutes': (pages_df['saved_at'].max() - pages_df['saved_at'].min()).total_seconds() / 60,
                'pages_per_minute': len(pages_df) / ((pages_df['saved_at'].max() - pages_df['saved_at'].min()).total_seconds() / 60) if pages_df['saved_at'].max() != pages_df['saved_at'].min() else 0
            }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], output_file: str = None) -> str:
        """분석 리포???�성"""
        if not analysis:
            return "No analysis data available"
        
        report = []
        report.append("# 법률 ?�집 ?�능 분석 리포??)
        report.append(f"?�성 ?�간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # ?�약 ?�보
        summary = analysis.get('summary', {})
        report.append("## ?�� ?�약 ?�보")
        report.append(f"- �??�이지 ?? {summary.get('total_pages', 0)}")
        report.append(f"- �?법률 ?? {summary.get('total_laws', 0)}")
        report.append(f"- ?�이지???�균 법률 ?? {summary.get('avg_laws_per_page', 0):.2f}")
        report.append(f"- �?처리 ?�간: {summary.get('total_processing_time', 0):.2f}�?)
        report.append(f"- ?�균 처리 ?�간: {summary.get('avg_processing_time', 0):.2f}�?)
        report.append("")
        
        # ?�능 ?�보
        performance = analysis.get('performance', {})
        report.append("## ???�능 분석")
        report.append(f"- 처리?? {performance.get('throughput_laws_per_minute', 0):.2f} 법률/�?)
        report.append(f"- ?�율???�수: {performance.get('efficiency_score', 0):.2f}")
        report.append("")
        
        # 콘텐�??�보
        content = analysis.get('content', {})
        report.append("## ?�� 콘텐�?분석")
        report.append(f"- 고유 법률 ?? {content.get('unique_laws', 0)}")
        report.append(f"- 중복 법률 ?? {content.get('duplicate_laws', 0)}")
        report.append("")
        
        # 법률 ?�형�?분포
        if content.get('law_types'):
            report.append("### 법률 ?�형�?분포")
            for law_type, count in content['law_types'].items():
                report.append(f"- {law_type}: {count}�?)
            report.append("")
        
        # ?�?�라???�보
        timeline = analysis.get('timeline', {})
        if timeline:
            report.append("## ???�?�라??분석")
            report.append(f"- ?�작 ?�간: {timeline.get('start_time', 'N/A')}")
            report.append(f"- 종료 ?�간: {timeline.get('end_time', 'N/A')}")
            report.append(f"- �??�요 ?�간: {timeline.get('duration_minutes', 0):.2f}�?)
            report.append(f"- 분당 ?�이지 ?? {timeline.get('pages_per_minute', 0):.2f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # ?�일 ?�??
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        
        return report_text
    
    def compare_runs(self, date1: str, date2: str) -> str:
        """???�행 결과 비교"""
        analysis1 = self.analyze_collection_logs(date1)
        analysis2 = self.analyze_collection_logs(date2)
        
        if not analysis1 or not analysis2:
            return "Cannot compare - insufficient data"
        
        comparison = []
        comparison.append("# 법률 ?�집 ?�능 비교 리포??)
        comparison.append(f"비교 ?�?? {date1} vs {date2}")
        comparison.append(f"?�성 ?�간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        comparison.append("")
        
        # ?�능 비교
        perf1 = analysis1.get('performance', {})
        perf2 = analysis2.get('performance', {})
        
        comparison.append("## ???�능 비교")
        throughput1 = perf1.get('throughput_laws_per_minute', 0)
        throughput2 = perf2.get('throughput_laws_per_minute', 0)
        throughput_change = ((throughput2 - throughput1) / throughput1 * 100) if throughput1 > 0 else 0
        
        comparison.append(f"- 처리??변?? {throughput1:.2f} ??{throughput2:.2f} 법률/�?({throughput_change:+.1f}%)")
        
        # ?�약 비교
        summary1 = analysis1.get('summary', {})
        summary2 = analysis2.get('summary', {})
        
        comparison.append("## ?�� ?�집??비교")
        comparison.append(f"- �?법률 ?? {summary1.get('total_laws', 0)} ??{summary2.get('total_laws', 0)}")
        comparison.append(f"- ?�균 처리 ?�간: {summary1.get('avg_processing_time', 0):.2f} ??{summary2.get('avg_processing_time', 0):.2f}�?)
        
        return "\n".join(comparison)


def main():
    """메인 ?�수"""
    parser = argparse.ArgumentParser(description='법률 ?�집 로그 분석 ?�구')
    parser.add_argument('--date', type=str, help='분석???�짜 (YYYYMMDD ?�식)')
    parser.add_argument('--output', type=str, help='출력 ?�일 경로')
    parser.add_argument('--compare', nargs=2, metavar=('DATE1', 'DATE2'), 
                       help='???�짜???�행 결과 비교')
    parser.add_argument('--log-dir', type=str, default='data/raw/assembly',
                       help='로그 ?�렉?�리 경로')
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    
    if args.compare:
        # 비교 분석
        result = analyzer.compare_runs(args.compare[0], args.compare[1])
        print(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Comparison report saved to: {args.output}")
    
    else:
        # ?�일 분석
        analysis = analyzer.analyze_collection_logs(args.date)
        
        if analysis:
            report = analyzer.generate_report(analysis, args.output)
            print(report)
        else:
            print("No analysis data available")


if __name__ == "__main__":
    main()
