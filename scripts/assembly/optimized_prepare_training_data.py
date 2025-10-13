#!/usr/bin/env python3
"""
최적화된 훈련 데이터 준비 스크립트
원본 법률 데이터와 처리된 데이터를 매칭하여 훈련 데이터 생성 (성능 최적화 버전)
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedTrainingDataPreparer:
    """최적화된 훈련 데이터 준비 클래스"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """
        초기화
        
        Args:
            raw_data_dir: 원본 데이터 디렉토리
            processed_data_dir: 처리된 데이터 디렉토리
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # 원본 데이터 인덱스 캐시
        self.raw_content_index = {}
        self._build_raw_content_index()
    
    def _build_raw_content_index(self):
        """원본 데이터 인덱스 구축 (한 번만 실행)"""
        logger.info("Building raw content index...")
        start_time = time.time()
        
        raw_files = list(self.raw_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(raw_files)} raw files")
        
        for i, raw_file in enumerate(raw_files):
            if i % 100 == 0:
                logger.info(f"Indexing raw file {i+1}/{len(raw_files)}: {raw_file.name}")
            
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and 'laws' in raw_data:
                    for law in raw_data['laws']:
                        law_id = law.get('cont_id')  # 원본 데이터에서는 cont_id 사용
                        law_name = law.get('law_name', '')  # 법률명도 저장
                        if law_id and law_name:
                            # cont_id와 법률명 모두로 인덱싱
                            self.raw_content_index[law_id] = law.get('law_content', '')
                            self.raw_content_index[law_name] = law.get('law_content', '')
                elif isinstance(raw_data, list):
                    for law in raw_data:
                        law_id = law.get('cont_id')  # 원본 데이터에서는 cont_id 사용
                        law_name = law.get('law_name', '')  # 법률명도 저장
                        if law_id and law_name:
                            # cont_id와 법률명 모두로 인덱싱
                            self.raw_content_index[law_id] = law.get('law_content', '')
                            self.raw_content_index[law_name] = law.get('law_content', '')
                            
            except Exception as e:
                logger.warning(f"Error reading {raw_file}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"Raw content index built in {elapsed_time:.2f} seconds")
        logger.info(f"Indexed {len(self.raw_content_index)} laws")
    
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """훈련 데이터 준비 (최적화된 버전)"""
        training_samples = []
        
        # 처리된 데이터 파일들 찾기
        processed_files = list(self.processed_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(processed_files)} processed files")
        
        processed_count = 0
        success_count = 0
        start_time = time.time()
        
        for i, processed_file in enumerate(processed_files):
            processed_count += 1
            
            # 진행 상황 표시 (더 자주)
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(processed_files) - i) / rate if rate > 0 else 0
                print(f"Processing file {i+1}/{len(processed_files)}: {processed_file.name}")
                print(f"  Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} minutes")
            
            try:
                # 처리된 데이터 로드
                with open(processed_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                if 'articles' not in processed_data:
                    continue
                
                # 원본 데이터 찾기 (인덱스 사용)
                law_id = processed_data.get('law_id', '')
                law_name = processed_data.get('law_name', '')
                
                # 법률명으로 먼저 찾기
                raw_content = self.raw_content_index.get(law_name)
                if not raw_content:
                    # law_id로 찾기 시도
                    raw_content = self.raw_content_index.get(law_id)
                
                if not raw_content:
                    continue
                
                # 각 조문에 대해 훈련 샘플 생성
                for article in processed_data['articles']:
                    sample = self._create_training_sample(article, raw_content)
                    if sample:
                        training_samples.append(sample)
                        success_count += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {processed_file}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessed {processed_count} files, generated {success_count} samples")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Rate: {processed_count/elapsed_time:.1f} files/sec")
        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def _create_training_sample(self, article: Dict[str, Any], raw_content: str) -> Optional[Dict[str, Any]]:
        """훈련 샘플 생성 (최적화된 버전)"""
        article_number = article.get('article_number', '')
        article_title = article.get('article_title', '')
        
        if not article_number:
            return None
        
        # 조문 위치 찾기 (정규식 최적화)
        position = self._find_article_position_optimized(raw_content, article_number)
        if position == -1:
            return None
        
        # 특성 추출 (최적화된 버전)
        features = self._extract_features_optimized(raw_content, position, article_number)
        
        # 레이블 결정
        label = 'real_article' if article_title else 'reference'
        
        return {
            'features': features,
            'label': label,
            'article_number': article_number,
            'article_title': article_title,
            'position': position,
            'raw_content': raw_content[:500]  # 디버깅용으로 더 적게 저장
        }
    
    def _find_article_position_optimized(self, content: str, article_number: str) -> int:
        """조문 위치 찾기 (최적화된 버전)"""
        # 정확한 조문 번호로 검색 (정규식 컴파일)
        pattern = re.compile(re.escape(article_number))
        match = pattern.search(content)
        return match.start() if match else -1
    
    def _extract_features_optimized(self, content: str, position: int, article_number: str) -> Dict[str, Any]:
        """특성 추출 (최적화된 버전)"""
        features = {}
        
        # 1. 위치 기반 특성
        features['position_ratio'] = position / len(content) if len(content) > 0 else 0
        features['is_at_start'] = 1 if position < 200 else 0
        features['is_at_end'] = 1 if position > len(content) * 0.8 else 0
        
        # 2. 문맥 기반 특성 (더 작은 윈도우 사용)
        context_before = content[max(0, position - 100):position]
        context_after = content[position:min(len(content), position + 100)]
        
        # 문장 끝 패턴 (컴파일된 정규식 사용)
        sentence_end_pattern = re.compile(r'[.!?]\s*$')
        features['has_sentence_end'] = 1 if sentence_end_pattern.search(context_before) else 0
        
        # 조문 참조 패턴 (컴파일된 정규식 사용)
        reference_patterns = [
            re.compile(r'제\d+조에\s*따라'),
            re.compile(r'제\d+조제\d+항'),
            re.compile(r'제\d+조의\d+'),
            re.compile(r'제\d+조.*?에\s*의하여'),
            re.compile(r'제\d+조.*?에\s*따라'),
        ]
        
        features['has_reference_pattern'] = 0
        for pattern in reference_patterns:
            if pattern.search(context_before):
                features['has_reference_pattern'] = 1
                break
        
        # 3. 조문 번호 특성
        article_num_match = re.search(r'\d+', article_number)
        article_num = int(article_num_match.group()) if article_num_match else 0
        features['article_number'] = article_num
        features['is_supplementary'] = 1 if '부칙' in article_number else 0
        
        # 4. 텍스트 길이 특성
        features['context_before_length'] = len(context_before)
        features['context_after_length'] = len(context_after)
        
        # 5. 조문 제목 유무 (컴파일된 정규식 사용)
        title_pattern = re.compile(r'제\d+조\s*\(([^)]+)\)')
        title_match = title_pattern.search(context_after)
        features['has_title'] = 1 if title_match else 0
        
        # 6. 특수 문자 패턴
        features['has_parentheses'] = 1 if '(' in context_after[:30] else 0
        features['has_quotes'] = 1 if '"' in context_after[:30] or "'" in context_after[:30] else 0
        
        # 7. 법률 용어 패턴 (더 적은 용어로)
        legal_terms = ['법률', '법령', '규정', '조항', '항', '호', '목']
        features['legal_term_count'] = sum(1 for term in legal_terms if term in context_after[:50])
        
        # 8. 숫자 패턴 (컴파일된 정규식 사용)
        number_pattern = re.compile(r'\d+')
        features['number_count'] = len(number_pattern.findall(context_after[:50]))
        
        # 9. 조문 내용 길이 (다음 조문까지의 거리)
        next_article_pattern = re.compile(r'제\d+조')
        next_article_match = next_article_pattern.search(content[position + 1:])
        if next_article_match:
            features['article_length'] = next_article_match.start()
        else:
            features['article_length'] = len(content) - position
        
        # 10. 문맥 밀도 (조문 참조 빈도)
        article_ref_pattern = re.compile(r'제\d+조')
        article_refs_in_context = len(article_ref_pattern.findall(context_before))
        features['reference_density'] = article_refs_in_context / max(len(context_before), 1) * 1000
        
        return features
    
    def save_training_data(self, training_samples: List[Dict[str, Any]], output_file: str):
        """훈련 데이터 저장"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved to {output_file}")
    
    def save_training_data_batch(self, training_samples: List[Dict[str, Any]], output_file: str, batch_size: int = 1000):
        """배치로 훈련 데이터 저장 (메모리 효율적)"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 배치로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            
            for i, sample in enumerate(training_samples):
                if i > 0:
                    f.write(',\n')
                
                json.dump(sample, f, ensure_ascii=False, indent=2)
                
                # 메모리 정리
                if i % batch_size == 0:
                    f.flush()
            
            f.write('\n]')
        
        logger.info(f"Training data saved to {output_file} in batches")


def main():
    """메인 함수"""
    start_time = time.time()
    
    # 데이터 준비기 생성
    preparer = OptimizedTrainingDataPreparer(
        raw_data_dir="data/raw/assembly/law/2025101201",
        processed_data_dir="data/processed/assembly/law"
    )
    
    # 훈련 데이터 준비
    training_samples = preparer.prepare_training_data()
    
    if len(training_samples) == 0:
        logger.error("No training samples generated")
        return
    
    # 훈련 데이터 저장
    output_file = "data/training/article_classification_training_data.json"
    preparer.save_training_data_batch(training_samples, output_file)
    
    # 통계 출력
    real_articles = sum(1 for sample in training_samples if sample['label'] == 'real_article')
    references = sum(1 for sample in training_samples if sample['label'] == 'reference')
    
    total_time = time.time() - start_time
    
    print(f"\n=== Training Data Statistics ===")
    print(f"Total samples: {len(training_samples)}")
    print(f"Real articles: {real_articles}")
    print(f"References: {references}")
    print(f"Real article ratio: {real_articles/len(training_samples):.3f}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(training_samples):.4f} seconds")


if __name__ == "__main__":
    main()
