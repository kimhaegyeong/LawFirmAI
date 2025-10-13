#!/usr/bin/env python3
"""
머신러닝 기반 조문 분류기
실제 법률 문서 패턴을 학습하여 조문 참조와 실제 조문을 구분
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleMLClassifier:
    """머신러닝 기반 조문 분류기"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        초기화
        
        Args:
            model_type: 사용할 모델 타입 ("random_forest", "gradient_boosting")
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = []
        
        # 모델 선택
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def extract_features(self, content: str, position: int, article_number: str) -> Dict[str, Any]:
        """
        조문의 특성을 추출
        
        Args:
            content: 전체 문서 내용
            position: 조문 위치
            article_number: 조문 번호
            
        Returns:
            추출된 특성 딕셔너리
        """
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
    
    def prepare_training_data(self, data_dir: str) -> Tuple[List[Dict], List[str]]:
        """
        훈련 데이터 준비
        
        Args:
            data_dir: 데이터 디렉토리 경로
            
        Returns:
            특성 리스트와 레이블 리스트
        """
        features_list = []
        labels = []
        
        data_path = Path(data_dir)
        json_files = list(data_path.glob("**/*.json"))
        
        logger.info(f"Found {len(json_files)} JSON files for training")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'articles' not in data:
                    continue
                
                # 원본 법률 내용 가져오기 (raw 데이터에서)
                law_content = self._get_raw_law_content(data.get('law_id', ''))
                if not law_content:
                    continue
                
                for article in data['articles']:
                    article_number = article.get('article_number', '')
                    article_title = article.get('article_title', '')
                    
                    # 조문 위치 찾기
                    position = self._find_article_position(law_content, article_number)
                    if position == -1:
                        continue
                    
                    # 특성 추출
                    features = self.extract_features(law_content, position, article_number)
                    
                    # 레이블 결정 (제목이 있으면 실제 조문, 없으면 참조)
                    label = 'real_article' if article_title else 'reference'
                    
                    features_list.append(features)
                    labels.append(label)
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Prepared {len(features_list)} training samples")
        return features_list, labels
    
    def _get_raw_law_content(self, law_id: str) -> str:
        """원본 법률 내용 가져오기"""
        # 실제 구현에서는 raw 데이터에서 해당 법률 내용을 찾아야 함
        # 여기서는 간단한 예시로 구현
        return ""
    
    def _find_article_position(self, content: str, article_number: str) -> int:
        """조문 위치 찾기"""
        pattern = re.escape(article_number)
        match = re.search(pattern, content)
        return match.start() if match else -1
    
    def train(self, features_list: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """
        모델 훈련
        
        Args:
            features_list: 특성 리스트
            labels: 레이블 리스트
            
        Returns:
            훈련 결과 딕셔너리
        """
        # 특성을 DataFrame으로 변환
        df = pd.DataFrame(features_list)
        
        # 텍스트 특성 추출
        text_features = []
        for features in features_list:
            # 문맥 텍스트 특성 (간단한 예시)
            text_features.append(f"article_{features.get('article_number', 0)}")
        
        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(max_features=1000)
        text_matrix = self.vectorizer.fit_transform(text_features)
        
        # 수치 특성과 텍스트 특성 결합
        numeric_features = df.drop(['article_number'], axis=1, errors='ignore')
        combined_features = np.hstack([numeric_features.values, text_matrix.toarray()])
        
        # 레이블 인코딩
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 모델 훈련
        self.model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test)
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, combined_features, encoded_labels, cv=5)
        
        # 특성 중요도
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        
        # 결과 반환
        results = {
            'accuracy': self.model.score(X_test, y_test),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred, target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None
        }
        
        logger.info(f"Model training completed. Accuracy: {results['accuracy']:.3f}")
        logger.info(f"Cross-validation score: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
        
        return results
    
    def predict(self, content: str, position: int, article_number: str) -> Tuple[str, float]:
        """
        조문 분류 예측
        
        Args:
            content: 전체 문서 내용
            position: 조문 위치
            article_number: 조문 번호
            
        Returns:
            예측된 클래스와 신뢰도
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # 특성 추출
        features = self.extract_features(content, position, article_number)
        
        # DataFrame으로 변환
        df = pd.DataFrame([features])
        
        # 텍스트 특성
        text_feature = f"article_{features.get('article_number', 0)}"
        
        # TF-IDF 변환
        text_matrix = self.vectorizer.transform([text_feature])
        
        # 수치 특성과 텍스트 특성 결합
        numeric_features = df.drop(['article_number'], axis=1, errors='ignore')
        combined_features = np.hstack([numeric_features.values, text_matrix.toarray()])
        
        # 예측
        prediction = self.model.predict(combined_features)[0]
        confidence = self.model.predict_proba(combined_features)[0].max()
        
        # 레이블 디코딩
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_class, confidence
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")


def main():
    """메인 함수"""
    # 분류기 생성
    classifier = ArticleMLClassifier(model_type="random_forest")
    
    # 훈련 데이터 준비
    data_dir = "data/processed/assembly/law"
    features_list, labels = classifier.prepare_training_data(data_dir)
    
    if len(features_list) == 0:
        logger.error("No training data found")
        return
    
    # 모델 훈련
    results = classifier.train(features_list, labels)
    
    # 결과 출력
    print("\n=== Training Results ===")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # 모델 저장
    model_path = "models/article_classifier.pkl"
    Path("models").mkdir(exist_ok=True)
    classifier.save_model(model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

