#!/usr/bin/env python3
"""
머신?�닝 기반 조문 분류�?
?�제 법률 문서 ?�턴???�습?�여 조문 참조?� ?�제 조문??구분
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

# 로깅 ?�정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleMLClassifier:
    """머신?�닝 기반 조문 분류�?""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        초기??
        
        Args:
            model_type: ?�용??모델 ?�??("random_forest", "gradient_boosting")
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.feature_names = []
        
        # 모델 ?�택
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
        조문???�성??추출
        
        Args:
            content: ?�체 문서 ?�용
            position: 조문 ?�치
            article_number: 조문 번호
            
        Returns:
            추출???�성 ?�셔?�리
        """
        features = {}
        
        # 1. ?�치 기반 ?�성
        features['position_ratio'] = position / len(content) if len(content) > 0 else 0
        features['is_at_start'] = 1 if position < 200 else 0
        features['is_at_end'] = 1 if position > len(content) * 0.8 else 0
        
        # 2. 문맥 기반 ?�성
        context_before = content[max(0, position - 200):position]
        context_after = content[position:min(len(content), position + 200)]
        
        # 문장 ???�턴
        features['has_sentence_end'] = 1 if re.search(r'[.!?]\s*$', context_before) else 0
        
        # 조문 참조 ?�턴
        reference_patterns = [
            r'??d+조에\s*?�라',
            r'??d+조제\d+??,
            r'??d+조의\d+',
            r'??d+�?*???s*?�하??,
            r'??d+�?*???s*?�라',
        ]
        
        features['has_reference_pattern'] = 0
        for pattern in reference_patterns:
            if re.search(pattern, context_before):
                features['has_reference_pattern'] = 1
                break
        
        # 3. 조문 번호 ?�성
        article_num = int(re.search(r'\d+', article_number).group()) if re.search(r'\d+', article_number) else 0
        features['article_number'] = article_num
        features['is_supplementary'] = 1 if '부�? in article_number else 0
        
        # 4. ?�스??길이 ?�성
        features['context_before_length'] = len(context_before)
        features['context_after_length'] = len(context_after)
        
        # 5. 조문 ?�목 ?�무
        title_match = re.search(r'??d+�?s*\(([^)]+)\)', context_after)
        features['has_title'] = 1 if title_match else 0
        
        # 6. ?�수 문자 ?�턴
        features['has_parentheses'] = 1 if '(' in context_after[:50] else 0
        features['has_quotes'] = 1 if '"' in context_after[:50] or "'" in context_after[:50] else 0
        
        # 7. 법률 ?�어 ?�턴
        legal_terms = [
            '법률', '법령', '규정', '조항', '??, '??, '�?,
            '?�행', '공포', '개정', '?��?', '?�정'
        ]
        
        features['legal_term_count'] = sum(1 for term in legal_terms if term in context_after[:100])
        
        # 8. ?�자 ?�턴
        features['number_count'] = len(re.findall(r'\d+', context_after[:100]))
        
        # 9. 조문 ?�용 길이 (?�음 조문까�???거리)
        next_article_match = re.search(r'??d+�?, content[position + 1:])
        if next_article_match:
            features['article_length'] = next_article_match.start()
        else:
            features['article_length'] = len(content) - position
        
        # 10. 문맥 밀??(조문 참조 빈도)
        article_refs_in_context = len(re.findall(r'??d+�?, context_before))
        features['reference_density'] = article_refs_in_context / max(len(context_before), 1) * 1000
        
        return features
    
    def prepare_training_data(self, data_dir: str) -> Tuple[List[Dict], List[str]]:
        """
        ?�련 ?�이??준�?
        
        Args:
            data_dir: ?�이???�렉?�리 경로
            
        Returns:
            ?�성 리스?��? ?�이�?리스??
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
                
                # ?�본 법률 ?�용 가?�오�?(raw ?�이?�에??
                law_content = self._get_raw_law_content(data.get('law_id', ''))
                if not law_content:
                    continue
                
                for article in data['articles']:
                    article_number = article.get('article_number', '')
                    article_title = article.get('article_title', '')
                    
                    # 조문 ?�치 찾기
                    position = self._find_article_position(law_content, article_number)
                    if position == -1:
                        continue
                    
                    # ?�성 추출
                    features = self.extract_features(law_content, position, article_number)
                    
                    # ?�이�?결정 (?�목???�으�??�제 조문, ?�으�?참조)
                    label = 'real_article' if article_title else 'reference'
                    
                    features_list.append(features)
                    labels.append(label)
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        logger.info(f"Prepared {len(features_list)} training samples")
        return features_list, labels
    
    def _get_raw_law_content(self, law_id: str) -> str:
        """?�본 법률 ?�용 가?�오�?""
        # ?�제 구현?�서??raw ?�이?�에???�당 법률 ?�용??찾아????
        # ?�기?�는 간단???�시�?구현
        return ""
    
    def _find_article_position(self, content: str, article_number: str) -> int:
        """조문 ?�치 찾기"""
        pattern = re.escape(article_number)
        match = re.search(pattern, content)
        return match.start() if match else -1
    
    def train(self, features_list: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """
        모델 ?�련
        
        Args:
            features_list: ?�성 리스??
            labels: ?�이�?리스??
            
        Returns:
            ?�련 결과 ?�셔?�리
        """
        # ?�성??DataFrame?�로 변??
        df = pd.DataFrame(features_list)
        
        # ?�스???�성 추출
        text_features = []
        for features in features_list:
            # 문맥 ?�스???�성 (간단???�시)
            text_features.append(f"article_{features.get('article_number', 0)}")
        
        # TF-IDF 벡터??
        self.vectorizer = TfidfVectorizer(max_features=1000)
        text_matrix = self.vectorizer.fit_transform(text_features)
        
        # ?�치 ?�성�??�스???�성 결합
        numeric_features = df.drop(['article_number'], axis=1, errors='ignore')
        combined_features = np.hstack([numeric_features.values, text_matrix.toarray()])
        
        # ?�이�??�코??
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # ?�련/?�스??분할
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 모델 ?�련
        self.model.fit(X_train, y_train)
        
        # ?�측 �??��?
        y_pred = self.model.predict(X_test)
        
        # 교차 검�?
        cv_scores = cross_val_score(self.model, combined_features, encoded_labels, cv=5)
        
        # ?�성 중요??
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
        조문 분류 ?�측
        
        Args:
            content: ?�체 문서 ?�용
            position: 조문 ?�치
            article_number: 조문 번호
            
        Returns:
            ?�측???�래?��? ?�뢰??
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # ?�성 추출
        features = self.extract_features(content, position, article_number)
        
        # DataFrame?�로 변??
        df = pd.DataFrame([features])
        
        # ?�스???�성
        text_feature = f"article_{features.get('article_number', 0)}"
        
        # TF-IDF 변??
        text_matrix = self.vectorizer.transform([text_feature])
        
        # ?�치 ?�성�??�스???�성 결합
        numeric_features = df.drop(['article_number'], axis=1, errors='ignore')
        combined_features = np.hstack([numeric_features.values, text_matrix.toarray()])
        
        # ?�측
        prediction = self.model.predict(combined_features)[0]
        confidence = self.model.predict_proba(combined_features)[0].max()
        
        # ?�이�??�코??
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_class, confidence
    
    def save_model(self, filepath: str):
        """모델 ?�??""
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
    """메인 ?�수"""
    # 분류�??�성
    classifier = ArticleMLClassifier(model_type="random_forest")
    
    # ?�련 ?�이??준�?
    data_dir = "data/processed/assembly/law"
    features_list, labels = classifier.prepare_training_data(data_dir)
    
    if len(features_list) == 0:
        logger.error("No training data found")
        return
    
    # 모델 ?�련
    results = classifier.train(features_list, labels)
    
    # 결과 출력
    print("\n=== Training Results ===")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # 모델 ?�??
    model_path = "models/article_classifier.pkl"
    Path("models").mkdir(exist_ok=True)
    classifier.save_model(model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()

