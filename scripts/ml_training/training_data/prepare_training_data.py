#!/usr/bin/env python3
"""
훈련 데이터 준비 스크립트
원본 법률 데이터와 처리된 데이터를 매칭하여 훈련 데이터 생성
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparer:
    """훈련 데이터 준비 클래스"""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """
        초기화
        
        Args:
            raw_data_dir: 원본 데이터 디렉토리
            processed_data_dir: 처리된 데이터 디렉토리
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
    def prepare_training_data(self) -> List[Dict[str, Any]]:
        """훈련 데이터 준비"""
        training_samples = []
        
        # 처리된 데이터 파일들 찾기
        processed_files = list(self.processed_data_dir.glob("**/*.json"))
        logger.info(f"Found {len(processed_files)} processed files")
        
        processed_count = 0
        success_count = 0
        
        for i, processed_file in enumerate(processed_files):
            processed_count += 1
            
            # 진행 상황 표시
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(processed_files)}: {processed_file.name}")
            
            try:
                # 처리된 데이터 로드
                with open(processed_file, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                
                if 'articles' not in processed_data:
                    continue
                
                # 원본 데이터 찾기
                raw_content = self._find_raw_content(processed_data.get('law_id', ''))
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
        
        print(f"\nProcessed {processed_count} files, generated {success_count} samples")
        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def _find_raw_content(self, law_id: str) -> Optional[str]:
        """원본 법률 내용 찾기"""
        if not law_id:
            return None
            
        # 원본 데이터 파일들에서 해당 법률 찾기
        raw_files = list(self.raw_data_dir.glob("**/*.json"))
        
        # 캐시를 사용하여 성능 개선
        if not hasattr(self, '_raw_content_cache'):
            self._raw_content_cache = {}
        
        if law_id in self._raw_content_cache:
            return self._raw_content_cache[law_id]
        
        for raw_file in raw_files:
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                if isinstance(raw_data, dict) and 'laws' in raw_data:
                    for law in raw_data['laws']:
                        if law.get('law_id') == law_id:
                            content = law.get('law_content', '')
                            self._raw_content_cache[law_id] = content
                            return content
                elif isinstance(raw_data, list):
                    for law in raw_data:
                        if law.get('law_id') == law_id:
                            content = law.get('law_content', '')
                            self._raw_content_cache[law_id] = content
                            return content
                            
            except Exception as e:
                logger.warning(f"Error reading {raw_file}: {e}")
                continue
        
        # 찾지 못한 경우 캐시에 None 저장
        self._raw_content_cache[law_id] = None
        return None
    
    def _create_training_sample(self, article: Dict[str, Any], raw_content: str) -> Optional[Dict[str, Any]]:
        """훈련 샘플 생성"""
        article_number = article.get('article_number', '')
        article_title = article.get('article_title', '')
        
        if not article_number:
            return None
        
        # 조문 위치 찾기
        position = self._find_article_position(raw_content, article_number)
        if position == -1:
            return None
        
        # 특성 추출
        features = self._extract_features(raw_content, position, article_number)
        
        # 레이블 결정
        label = 'real_article' if article_title else 'reference'
        
        return {
            'features': features,
            'label': label,
            'article_number': article_number,
            'article_title': article_title,
            'position': position,
            'raw_content': raw_content[:1000]  # 디버깅용으로 일부만 저장
        }
    
    def _find_article_position(self, content: str, article_number: str) -> int:
        """조문 위치 찾기"""
        # 정확한 조문 번호로 검색
        pattern = re.escape(article_number)
        match = re.search(pattern, content)
        return match.start() if match else -1
    
    def _extract_features(self, content: str, position: int, article_number: str) -> Dict[str, Any]:
        """특성 추출"""
        features = {}
        
        # 1. 위치 기반 특성
        features['position_ratio'] = position / len(content) if len(content) > 0 else 0
        features['is_at_start'] = 1 if position < 200 else 0
        features['is_at_end'] = 1 if position > len(content) * 0.8 else 0
        
        # 2. 문맥 기반 특성
        context_before = content[max(0, position - 200):position]
        context_after = content[position:min(len(content), position + 200)]
        
        # 문장 끝 패턴
        features['has_sentence_end'] = 1 if re.search(r'[.!?]\s*$', context_before) else 0
        
        # 조문 참조 패턴
        reference_patterns = [
            r'제\d+조에\s*따라',
            r'제\d+조제\d+항',
            r'제\d+조의\d+',
            r'제\d+조.*?에\s*의하여',
            r'제\d+조.*?에\s*따라',
        ]
        
        features['has_reference_pattern'] = 0
        for pattern in reference_patterns:
            if re.search(pattern, context_before):
                features['has_reference_pattern'] = 1
                break
        
        # 3. 조문 번호 특성
        article_num = int(re.search(r'\d+', article_number).group()) if re.search(r'\d+', article_number) else 0
        features['article_number'] = article_num
        features['is_supplementary'] = 1 if '부칙' in article_number else 0
        
        # 4. 텍스트 길이 특성
        features['context_before_length'] = len(context_before)
        features['context_after_length'] = len(context_after)
        
        # 5. 조문 제목 유무
        title_match = re.search(r'제\d+조\s*\(([^)]+)\)', context_after)
        features['has_title'] = 1 if title_match else 0
        
        # 6. 특수 문자 패턴
        features['has_parentheses'] = 1 if '(' in context_after[:50] else 0
        features['has_quotes'] = 1 if '"' in context_after[:50] or "'" in context_after[:50] else 0
        
        # 7. 법률 용어 패턴
        legal_terms = [
            '법률', '법령', '규정', '조항', '항', '호', '목',
            '시행', '공포', '개정', '폐지', '제정'
        ]
        
        features['legal_term_count'] = sum(1 for term in legal_terms if term in context_after[:100])
        
        # 8. 숫자 패턴
        features['number_count'] = len(re.findall(r'\d+', context_after[:100]))
        
        # 9. 조문 내용 길이 (다음 조문까지의 거리)
        next_article_match = re.search(r'제\d+조', content[position + 1:])
        if next_article_match:
            features['article_length'] = next_article_match.start()
        else:
            features['article_length'] = len(content) - position
        
        # 10. 문맥 밀도 (조문 참조 빈도)
        article_refs_in_context = len(re.findall(r'제\d+조', context_before))
        features['reference_density'] = article_refs_in_context / max(len(context_before), 1) * 1000
        
        return features
    
    def save_training_data(self, training_samples: List[Dict[str, Any]], output_file: str):
        """훈련 데이터 저장"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training data saved to {output_file}")


def main():
    """메인 함수"""
    # 데이터 준비기 생성
    preparer = TrainingDataPreparer(
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
    preparer.save_training_data(training_samples, output_file)
    
    # 통계 출력
    real_articles = sum(1 for sample in training_samples if sample['label'] == 'real_article')
    references = sum(1 for sample in training_samples if sample['label'] == 'reference')
    
    print(f"\n=== Training Data Statistics ===")
    print(f"Total samples: {len(training_samples)}")
    print(f"Real articles: {real_articles}")
    print(f"References: {references}")
    print(f"Real article ratio: {real_articles/len(training_samples):.3f}")


if __name__ == "__main__":
    main()
