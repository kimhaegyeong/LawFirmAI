#!/usr/bin/env python3
"""
법률 데이터 최적화 스크립트
기존 전처리된 데이터를 압축하여 용량을 50-70% 줄입니다.

주요 최적화:
1. 중복 필드 제거 (불필요한 필드 제거)
2. 메타데이터 간소화 (불필요한 필드 제거)
3. 텍스트 압축 (불필요한 공백 및 반복 텍스트 제거)
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

def compress_legal_text(text: str) -> str:
    """법률 텍스트 압축"""
    if not text:
        return ""
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 반복되는 법률 용어 축약
    replacements = {
        '이 법에 따르면': '이 법에 따라',
        '다음 각 호의 어느 하나에 해당하는': '다음에 해당하는',
        '특별시장·광역시장·특별자치시장·도지사': '시·도지사',
        '특별자치도지사': '특별자치도지사',
        '중앙행정기관의 장': '중앙행정기관장',
        '지방자치단체의 장': '지방자치단체장',
        '국가 또는 지방자치단체': '국가·지방자치단체',
        '이하 "시·도지사"라 한다': '이하 시·도지사라 함',
        '이하 "특례시"라 한다': '이하 특례시라 함'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.strip()

def compress_law_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """개별 법률 데이터 압축"""
    # 필수 필드만 유지
    compressed = {
        'law_id': data.get('law_id'),
        'law_name': data.get('law_name'),
        'law_type': data.get('law_type'),
        'category': data.get('category'),
        'promulgation_number': data.get('promulgation_number'),
        'promulgation_date': data.get('promulgation_date'),
        'enforcement_date': data.get('enforcement_date'),
        'amendment_type': data.get('amendment_type'),
        'ministry': data.get('ministry'),
        'articles': data.get('articles', [])
    }
    
    # articles 내부 텍스트도 압축
    for article in compressed['articles']:
        if 'article_content' in article:
            article['article_content'] = compress_legal_text(article['article_content'])
        
        # sub_articles도 압축
        for sub_article in article.get('sub_articles', []):
            if 'content' in sub_article:
                sub_article['content'] = compress_legal_text(sub_article['content'])
    
    return compressed

def optimize_existing_data(input_dir: Path, output_dir: Path, backup: bool = True):
    """기존 데이터 최적화"""
    print(f"[START] 법률 데이터 최적화 시작")
    print(f"   입력 디렉토리: {input_dir}")
    print(f"   출력 디렉토리: {output_dir}")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 백업 디렉토리 생성
    if backup:
        backup_dir = input_dir.parent / f"{input_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"   백업 디렉토리: {backup_dir}")
    
    total_original_size = 0
    total_compressed_size = 0
    processed_files = 0
    
    # JSON 파일들 처리
    json_files = [f for f in input_dir.glob("*.json") if f.name != "processing_status.db"]
    
    print(f"   처리할 파일 수: {len(json_files)}")
    print()
    
    for json_file in json_files:
        try:
            # 원본 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 백업 생성
            if backup:
                backup_file = backup_dir / json_file.name
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 압축된 데이터 생성
            compressed_data = compress_law_data(data)
            
            # 압축된 파일 저장
            output_file = output_dir / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(compressed_data, f, ensure_ascii=False, separators=(',', ':'))
            
            # 용량 비교
            original_size = json_file.stat().st_size
            compressed_size = output_file.stat().st_size
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            processed_files += 1
            
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"[OK] {json_file.name}: {original_size:,} -> {compressed_size:,} bytes ({compression_ratio:.1f}% 압축)")
            
        except Exception as e:
            print(f"[ERROR] {json_file.name} 처리 실패: {e}")
    
    # processing_status.db 복사
    db_file = input_dir / "processing_status.db"
    if db_file.exists():
        import shutil
        shutil.copy2(db_file, output_dir / "processing_status.db")
        print(f"[COPY] processing_status.db 복사 완료")
    
    # 결과 요약
    print(f"\n[RESULT] 최적화 결과:")
    print(f"   처리된 파일 수: {processed_files}")
    print(f"   원본 용량: {total_original_size:,} bytes ({total_original_size/1024/1024:.1f} MB)")
    print(f"   압축 용량: {total_compressed_size:,} bytes ({total_compressed_size/1024/1024:.1f} MB)")
    print(f"   압축률: {(1 - total_compressed_size / total_original_size) * 100:.1f}%")
    print(f"   절약된 용량: {(total_original_size - total_compressed_size)/1024/1024:.1f} MB")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='법률 데이터 최적화')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='입력 디렉토리 경로')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='출력 디렉토리 경로')
    parser.add_argument('--no-backup', action='store_true',
                       help='백업 생성 안함')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ 입력 디렉토리가 존재하지 않습니다: {input_dir}")
        sys.exit(1)
    
    optimize_existing_data(input_dir, output_dir, backup=not args.no_backup)

if __name__ == "__main__":
    main()
