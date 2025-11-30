# -*- coding: utf-8 -*-
"""
감정 및 의도 분석기
사용자 감정과 의도를 파악하여 적절한 응답 톤 결정
"""

import re
import os
try:
    from lawfirm_langgraph.core.utils.logger import get_logger
except ImportError:
    from core.utils.logger import get_logger
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from lawfirm_langgraph.core.conversation.conversation_manager import ConversationContext
except ImportError:
    from core.conversation.conversation_manager import ConversationContext

logger = get_logger(__name__)

# Transformers 모델 지원 여부 확인
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EmotionType(Enum):
    """감정 유형"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    URGENT = "urgent"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


class IntentType(Enum):
    """의도 유형"""
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    THANKS = "thanks"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    GENERAL = "general"


class UrgencyLevel(Enum):
    """긴급도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmotionAnalysis:
    """감정 분석 결과"""
    primary_emotion: EmotionType
    emotion_scores: Dict[str, float]
    confidence: float
    intensity: float
    reasoning: str


@dataclass
class IntentAnalysis:
    """의도 분석 결과"""
    primary_intent: IntentType
    intent_scores: Dict[str, float]
    confidence: float
    urgency_level: UrgencyLevel
    reasoning: str


@dataclass
class ResponseTone:
    """응답 톤"""
    tone_type: str
    empathy_level: float
    formality_level: float
    urgency_response: bool
    explanation_depth: str


class EmotionIntentAnalyzer:
    """감정 및 의도 분석기 (하이브리드: 패턴 기반 + KoBERT)"""
    
    def __init__(self, use_ml_model: Optional[bool] = None):
        """
        감정 및 의도 분석기 초기화
        
        Args:
            use_ml_model: ML 모델 사용 여부 (None이면 환경 변수 확인)
        """
        self.logger = get_logger(__name__)
        
        # ML 모델 사용 여부 결정 (환경 변수 또는 파라미터)
        if use_ml_model is None:
            use_ml_model = os.getenv("USE_ML_EMOTION_ANALYZER", "true").lower() == "true"
        
        self.use_ml_model = use_ml_model and TRANSFORMERS_AVAILABLE
        self.ml_model = None
        self.ml_tokenizer = None
        self.ml_model_name = os.getenv("EMOTION_ML_MODEL", "monologg/kobert")
        
        # ML 모델 지연 로딩 (필요 시에만)
        if self.use_ml_model:
            self._ml_model_loaded = False
        else:
            self._ml_model_loaded = False
            if not TRANSFORMERS_AVAILABLE:
                self.logger.debug("Transformers not available, using pattern-based only")
            else:
                self.logger.debug("ML model disabled, using pattern-based only")
        
        # 감정 키워드 패턴
        self.emotion_patterns = {
            EmotionType.POSITIVE: [
                r"감사|고마워|좋아|훌륭|완벽|만족|기쁘|행복|좋은",
                r"도움|도움이|해결|해결됨|이해|이해됨|명확|명확해"
            ],
            EmotionType.NEGATIVE: [
                r"안좋|나쁘|문제|문제가|어려워|힘들|복잡|복잡해",
                r"이해안|모르겠|모르겠어|알겠|알겠어|어떻게|어떡해"
            ],
            EmotionType.URGENT: [
                r"급해|급함|빨리|빨리빨리|즉시|당장|지금|지금당장",
                r"긴급|긴급해|시급|시급해|마감|마감이|데드라인",
                r"오늘|내일|내일까지|오늘까지|시간|시간이|시간없"
            ],
            EmotionType.ANXIOUS: [
                r"걱정|걱정돼|불안|불안해|무서워|두려워|걱정스러워",
                r"어떻게|어떡해|어떻게해|어떻게해야|어떻게하죠",
                r"괜찮|괜찮아|괜찮을까|문제없|문제없을까"
            ],
            EmotionType.ANGRY: [
                r"화나|화났|짜증|짜증나|답답|답답해|속상|속상해",
                r"이상해|이상하|말이안|말이안돼|말이안되|말이안됨",
                r"왜|왜그래|왜이래|왜이런|왜이런거야"
            ],
            EmotionType.SATISFIED: [
                r"만족|만족해|좋아|좋네|훌륭|완벽|완벽해|완료|완료됨",
                r"해결|해결됨|해결됐|이해|이해됨|이해됐|명확|명확해"
            ],
            EmotionType.CONFUSED: [
                r"헷갈|헷갈려|혼란|혼란스러워|모르겠|모르겠어|알겠",
                r"이해안|이해안돼|이해안되|이해안됨|복잡|복잡해",
                r"어떻게|어떡해|어떻게해|어떻게해야|어떻게하죠"
            ]
        }
        
        # 의도 키워드 패턴
        self.intent_patterns = {
            IntentType.QUESTION: [
                r"무엇|뭐|어떤|어떻게|언제|어디서|왜|어떤|어떤것",
                r"알려주|가르쳐|설명해|말해|알고싶|궁금|궁금해"
            ],
            IntentType.REQUEST: [
                r"해주|해줘|도와|도와줘|부탁|부탁해|요청|요청해",
                r"작성해|만들어|준비해|처리해|해결해|진행해"
            ],
            IntentType.COMPLAINT: [
                r"문제|문제가|이상해|이상하|말이안|말이안돼|말이안되",
                r"왜|왜그래|왜이래|왜이런|왜이런거야|이상|이상해"
            ],
            IntentType.THANKS: [
                r"감사|고마워|고맙|고맙습니다|감사합니다|고마워요",
                r"도움|도움이|도움이됐|도움이되었|도움이되었어"
            ],
            IntentType.CLARIFICATION: [
                r"정확히|정확한|구체적|구체적으로|자세히|자세한",
                r"예시|예를|예를들어|즉|다시|다시한번|다시한번더"
            ],
            IntentType.FOLLOW_UP: [
                r"추가|추가로|또|또한|그리고|그리고|더|더자세히",
                r"이어서|계속|계속해서|다음|다음으로|그다음"
            ],
            IntentType.EMERGENCY: [
                r"긴급|긴급해|급해|급함|즉시|당장|지금|지금당장",
                r"마감|마감이|데드라인|시간|시간이|시간없|오늘|내일"
            ]
        }
        
        # 긴급도 평가 패턴
        self.urgency_patterns = {
            UrgencyLevel.CRITICAL: [
                r"지금당장|즉시|당장|긴급|긴급해|마감|마감이|데드라인",
                r"오늘까지|내일까지|시간없|시간이|시간이없"
            ],
            UrgencyLevel.HIGH: [
                r"빨리|빨리빨리|급해|급함|시급|시급해|오늘|내일",
                r"가능한빨리|최대한빨리|빨리빨리|어서|어서빨리"
            ],
            UrgencyLevel.MEDIUM: [
                r"가능하면|가능한|시간되면|시간날때|여유있을때",
                r"나중에|나중에라도|언젠가|언젠가는|기회되면"
            ],
            UrgencyLevel.LOW: [
                r"천천히|여유있게|시간있을때|시간날때|나중에",
                r"급하지|급하지않|급하지않아|급하지않음"
            ]
        }
        
        # 응답 톤 패턴
        self.response_tone_patterns = {
            "empathetic": {
                "keywords": ["걱정", "불안", "어려워", "힘들", "문제"],
                "empathy_level": 0.8,
                "formality_level": 0.6
            },
            "professional": {
                "keywords": ["법률", "법령", "조문", "판례", "법원"],
                "empathy_level": 0.3,
                "formality_level": 0.9
            },
            "supportive": {
                "keywords": ["도움", "해결", "이해", "명확", "설명"],
                "empathy_level": 0.7,
                "formality_level": 0.5
            },
            "urgent": {
                "keywords": ["급해", "긴급", "빨리", "즉시", "당장"],
                "empathy_level": 0.4,
                "formality_level": 0.7
            },
            "casual": {
                "keywords": ["궁금", "알고싶", "좋아", "만족", "감사"],
                "empathy_level": 0.6,
                "formality_level": 0.3
            }
        }
        
        self.logger.info(f"EmotionIntentAnalyzer initialized (ML model: {'enabled' if self.use_ml_model else 'disabled'})")
    
    def _load_ml_model(self):
        """KoBERT 모델 지연 로딩 (ModelCacheManager 사용)"""
        if self._ml_model_loaded:
            return
        
        if not self.use_ml_model:
            return
        
        try:
            self.logger.debug(f"Loading ML model: {self.ml_model_name}")
            
            # ModelCacheManager를 통한 모델 로드
            try:
                from lawfirm_langgraph.core.shared.utils.model_cache_manager import get_model_cache_manager
                cache_manager = get_model_cache_manager()
                
                # GPU 자동 감지
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                model_dict = cache_manager.get_transformers_model(
                    self.ml_model_name,
                    model_type="AutoModelForSequenceClassification",
                    device=device
                )
                
                if model_dict:
                    self.ml_model = model_dict["model"]
                    self.ml_tokenizer = model_dict["tokenizer"]
                    self._ml_model_loaded = True
                    self.logger.info(f"✅ ML model loaded via cache manager: {self.ml_model_name} (device: {device})")
                    return
                else:
                    self.logger.warning("Failed to load model via cache manager, falling back to direct load")
            except ImportError:
                self.logger.debug("ModelCacheManager not available, using direct load")
            except Exception as e:
                self.logger.warning(f"Cache manager failed: {e}, falling back to direct load")
            
            # 폴백: 직접 로드
            self.ml_tokenizer = AutoTokenizer.from_pretrained(
                self.ml_model_name,
                cache_dir=os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                trust_remote_code=True  # KoBERT 등 커스텀 코드 모델 지원
            )
            self.ml_model = AutoModelForSequenceClassification.from_pretrained(
                self.ml_model_name,
                cache_dir=os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                trust_remote_code=True  # KoBERT 등 커스텀 코드 모델 지원
            )
            
            # 평가 모드로 설정
            self.ml_model.eval()
            
            # GPU 사용 가능 시 GPU로 이동
            if torch.cuda.is_available():
                self.ml_model = self.ml_model.cuda()
                self.logger.debug("ML model loaded on GPU (direct)")
            else:
                self.logger.debug("ML model loaded on CPU (direct)")
            
            self._ml_model_loaded = True
            self.logger.info(f"✅ ML model loaded: {self.ml_model_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load ML model: {e}, using pattern-based only")
            self.use_ml_model = False
            self._ml_model_loaded = False
    
    def analyze_emotion(self, text: str) -> EmotionAnalysis:
        """
        감정 분석 (하이브리드: 패턴 기반 + KoBERT)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            EmotionAnalysis: 감정 분석 결과
        """
        try:
            # 1단계: 패턴 기반 분석 (빠른 폴백)
            pattern_result = self._analyze_emotion_pattern(text)
            
            # 2단계: 신뢰도가 낮으면 ML 모델 사용
            if pattern_result.confidence < 0.6 and self.use_ml_model:
                try:
                    ml_result = self._analyze_emotion_ml(text)
                    if ml_result:
                        # 두 결과 결합 (가중 평균)
                        return self._combine_emotion_results(pattern_result, ml_result)
                except Exception as e:
                    self.logger.debug(f"ML emotion analysis failed: {e}, using pattern result")
            
            return pattern_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing emotion: {e}")
            return EmotionAnalysis(
                primary_emotion=EmotionType.NEUTRAL,
                emotion_scores={EmotionType.NEUTRAL.value: 1.0},
                confidence=0.0,
                intensity=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
    def analyze_emotion_batch(self, texts: List[str], batch_size: int = 8) -> List[EmotionAnalysis]:
        """
        감정 분석 배치 처리
        
        Args:
            texts: 분석할 텍스트 리스트
            batch_size: 배치 크기 (ML 모델 사용 시)
            
        Returns:
            List[EmotionAnalysis]: 감정 분석 결과 리스트
        """
        if not texts:
            return []
        
        # 패턴 기반 분석 (모든 텍스트에 대해 빠르게 처리)
        pattern_results = [self._analyze_emotion_pattern(text) for text in texts]
        
        # ML 모델이 필요한 텍스트만 필터링
        ml_texts = []
        ml_indices = []
        for i, (text, pattern_result) in enumerate(zip(texts, pattern_results)):
            if pattern_result.confidence < 0.6 and self.use_ml_model:
                ml_texts.append(text)
                ml_indices.append(i)
        
        # ML 모델 배치 처리
        if ml_texts and self.use_ml_model:
            try:
                ml_results = self._analyze_emotion_ml_batch(ml_texts, batch_size)
                
                # 결과 병합
                for idx, ml_result in zip(ml_indices, ml_results):
                    if ml_result:
                        pattern_results[idx] = self._combine_emotion_results(
                            pattern_results[idx], ml_result
                        )
            except Exception as e:
                self.logger.warning(f"ML batch analysis failed: {e}, using pattern results only")
        
        return pattern_results
    
    def _analyze_emotion_pattern(self, text: str) -> EmotionAnalysis:
        """패턴 기반 감정 분석 (기존 방식)"""
        text_lower = text.lower()
        emotion_scores = {}
        
        # 각 감정 유형별 점수 계산
        for emotion_type, patterns in self.emotion_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.1
            
            # 패턴 길이에 따른 가중치
            if score > 0:
                score += 0.1
            
            emotion_scores[emotion_type.value] = min(1.0, score)
        
        # 기본 감정 설정 (점수가 모두 낮으면 중립)
        if not any(score > 0.1 for score in emotion_scores.values()):
            emotion_scores[EmotionType.NEUTRAL.value] = 0.5
        
        # 주요 감정 결정
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion_type = EmotionType(primary_emotion[0])
        
        # 신뢰도 계산
        confidence = primary_emotion[1]
        
        # 강도 계산
        intensity = self._calculate_emotion_intensity(text_lower)
        
        # 추론 과정 생성
        reasoning = self._generate_emotion_reasoning(text, primary_emotion_type, emotion_scores)
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion_type,
            emotion_scores=emotion_scores,
            confidence=confidence,
            intensity=intensity,
            reasoning=reasoning
        )
    
    def _analyze_emotion_ml(self, text: str) -> Optional[EmotionAnalysis]:
        """ML 모델 기반 감정 분석 (KoBERT)"""
        if not self._ml_model_loaded:
            self._load_ml_model()
        
        if not self.ml_model or not self.ml_tokenizer:
            return None
        
        try:
            # 텍스트 토큰화
            inputs = self.ml_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            # GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # 결과를 CPU로 이동 후 numpy 변환
            probs = probabilities.cpu().numpy()[0]
            
            # KoBERT는 일반적인 분류 모델이므로, 감정 점수로 매핑
            # 실제로는 파인튜닝된 모델이 필요하지만, 여기서는 기본 모델 사용
            # 감정 점수 초기화
            emotion_scores = {
                EmotionType.POSITIVE.value: float(probs[0]) if len(probs) > 0 else 0.0,
                EmotionType.NEGATIVE.value: float(probs[1]) if len(probs) > 1 else 0.0,
                EmotionType.NEUTRAL.value: float(probs[2]) if len(probs) > 2 else 0.5,
            }
            
            # 나머지 감정 유형은 기본값
            for emotion_type in EmotionType:
                if emotion_type.value not in emotion_scores:
                    emotion_scores[emotion_type.value] = 0.0
            
            # 주요 감정 결정
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            primary_emotion_type = EmotionType(primary_emotion[0])
            
            # 신뢰도 계산
            confidence = primary_emotion[1]
            
            # 강도 계산 (ML 모델 출력 기반)
            intensity = min(1.0, confidence * 1.2)
            
            # 추론 과정 생성
            reasoning = f"ML 모델 분석 (KoBERT): {primary_emotion_type.value} (신뢰도: {confidence:.2f})"
            
            return EmotionAnalysis(
                primary_emotion=primary_emotion_type,
                emotion_scores=emotion_scores,
                confidence=confidence,
                intensity=intensity,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.warning(f"ML emotion analysis error: {e}")
            return None
    
    def _analyze_emotion_ml_batch(self, texts: List[str], batch_size: int = 8) -> List[Optional[EmotionAnalysis]]:
        """ML 모델 기반 감정 분석 배치 처리"""
        if not self._ml_model_loaded:
            self._load_ml_model()
        
        if not self.ml_model or not self.ml_tokenizer:
            return [None] * len(texts)
        
        results = []
        
        try:
            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 배치 토큰화
                inputs = self.ml_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True
                )
                
                # GPU로 이동
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 배치 추론
                with torch.no_grad():
                    outputs = self.ml_model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                
                # 결과 처리
                probs_batch = probabilities.cpu().numpy()
                
                for probs in probs_batch:
                    # 감정 점수 초기화
                    emotion_scores = {
                        EmotionType.POSITIVE.value: float(probs[0]) if len(probs) > 0 else 0.0,
                        EmotionType.NEGATIVE.value: float(probs[1]) if len(probs) > 1 else 0.0,
                        EmotionType.NEUTRAL.value: float(probs[2]) if len(probs) > 2 else 0.5,
                    }
                    
                    # 나머지 감정 유형은 기본값
                    for emotion_type in EmotionType:
                        if emotion_type.value not in emotion_scores:
                            emotion_scores[emotion_type.value] = 0.0
                    
                    # 주요 감정 결정
                    primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    primary_emotion_type = EmotionType(primary_emotion[0])
                    
                    # 신뢰도 계산
                    confidence = primary_emotion[1]
                    
                    # 강도 계산
                    intensity = min(1.0, confidence * 1.2)
                    
                    # 추론 과정 생성
                    reasoning = f"ML 모델 분석 (KoBERT, 배치): {primary_emotion_type.value} (신뢰도: {confidence:.2f})"
                    
                    results.append(EmotionAnalysis(
                        primary_emotion=primary_emotion_type,
                        emotion_scores=emotion_scores,
                        confidence=confidence,
                        intensity=intensity,
                        reasoning=reasoning
                    ))
            
            return results
            
        except Exception as e:
            self.logger.warning(f"ML batch emotion analysis error: {e}")
            return [None] * len(texts)
    
    def _combine_emotion_results(self, pattern_result: EmotionAnalysis, ml_result: EmotionAnalysis) -> EmotionAnalysis:
        """패턴 기반과 ML 모델 결과 결합"""
        # 가중 평균 (패턴 0.4, ML 0.6)
        pattern_weight = 0.4
        ml_weight = 0.6
        
        combined_scores = {}
        for emotion_type in EmotionType:
            pattern_score = pattern_result.emotion_scores.get(emotion_type.value, 0.0)
            ml_score = ml_result.emotion_scores.get(emotion_type.value, 0.0)
            combined_scores[emotion_type.value] = pattern_score * pattern_weight + ml_score * ml_weight
        
        # 주요 감정 결정
        primary_emotion = max(combined_scores.items(), key=lambda x: x[1])
        primary_emotion_type = EmotionType(primary_emotion[0])
        
        # 신뢰도 계산 (두 결과의 평균)
        confidence = (pattern_result.confidence * pattern_weight + ml_result.confidence * ml_weight)
        
        # 강도 계산
        intensity = (pattern_result.intensity * pattern_weight + ml_result.intensity * ml_weight)
        
        # 추론 과정 생성
        reasoning = f"하이브리드 분석: 패턴({pattern_result.primary_emotion.value}, {pattern_result.confidence:.2f}) + ML({ml_result.primary_emotion.value}, {ml_result.confidence:.2f})"
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion_type,
            emotion_scores=combined_scores,
            confidence=confidence,
            intensity=intensity,
            reasoning=reasoning
        )
    
    def analyze_intent(self, text: str, context: Optional[ConversationContext] = None) -> IntentAnalysis:
        """
        의도 분석 (하이브리드: 패턴 기반 + KoBERT)
        
        Args:
            text: 분석할 텍스트
            context: 대화 맥락 (선택사항)
            
        Returns:
            IntentAnalysis: 의도 분석 결과
        """
        try:
            # 1단계: 패턴 기반 분석
            pattern_result = self._analyze_intent_pattern(text, context)
            
            # 2단계: 신뢰도가 낮으면 ML 모델 사용
            if pattern_result.confidence < 0.6 and self.use_ml_model:
                try:
                    ml_result = self._analyze_intent_ml(text)
                    if ml_result:
                        # 두 결과 결합
                        return self._combine_intent_results(pattern_result, ml_result)
                except Exception as e:
                    self.logger.debug(f"ML intent analysis failed: {e}, using pattern result")
            
            return pattern_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            return IntentAnalysis(
                primary_intent=IntentType.GENERAL,
                intent_scores={IntentType.GENERAL.value: 1.0},
                confidence=0.0,
                urgency_level=UrgencyLevel.LOW,
                reasoning=f"Error: {str(e)}"
            )
    
    def analyze_intent_batch(self, texts: List[str], contexts: Optional[List[Optional[ConversationContext]]] = None, batch_size: int = 8) -> List[IntentAnalysis]:
        """
        의도 분석 배치 처리
        
        Args:
            texts: 분석할 텍스트 리스트
            contexts: 대화 맥락 리스트 (선택사항)
            batch_size: 배치 크기 (ML 모델 사용 시)
            
        Returns:
            List[IntentAnalysis]: 의도 분석 결과 리스트
        """
        if not texts:
            return []
        
        if contexts is None:
            contexts = [None] * len(texts)
        
        # 패턴 기반 분석
        pattern_results = [
            self._analyze_intent_pattern(text, context)
            for text, context in zip(texts, contexts)
        ]
        
        # ML 모델이 필요한 텍스트만 필터링
        ml_texts = []
        ml_indices = []
        for i, (text, pattern_result) in enumerate(zip(texts, pattern_results)):
            if pattern_result.confidence < 0.6 and self.use_ml_model:
                ml_texts.append(text)
                ml_indices.append(i)
        
        # ML 모델 배치 처리
        if ml_texts and self.use_ml_model:
            try:
                ml_results = self._analyze_intent_ml_batch(ml_texts, batch_size)
                
                # 결과 병합
                for idx, ml_result in zip(ml_indices, ml_results):
                    if ml_result:
                        pattern_results[idx] = self._combine_intent_results(
                            pattern_results[idx], ml_result
                        )
            except Exception as e:
                self.logger.warning(f"ML batch intent analysis failed: {e}, using pattern results only")
        
        return pattern_results
    
    def _analyze_intent_pattern(self, text: str, context: Optional[ConversationContext] = None) -> IntentAnalysis:
        """패턴 기반 의도 분석 (기존 방식)"""
        text_lower = text.lower()
        intent_scores = {}
        
        # 각 의도 유형별 점수 계산
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.1
            
            # 패턴 길이에 따른 가중치
            if score > 0:
                score += 0.1
            
            intent_scores[intent_type.value] = min(1.0, score)
        
        # 맥락 기반 의도 조정
        if context:
            intent_scores = self._adjust_intent_with_context(intent_scores, context)
        
        # 기본 의도 설정 (점수가 모두 낮으면 일반)
        if not any(score > 0.1 for score in intent_scores.values()):
            intent_scores[IntentType.GENERAL.value] = 0.5
        
        # 주요 의도 결정
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        primary_intent_type = IntentType(primary_intent[0])
        
        # 신뢰도 계산
        confidence = primary_intent[1]
        
        # 긴급도 평가
        urgency_level = self.assess_urgency(text, {})
        
        # 추론 과정 생성
        reasoning = self._generate_intent_reasoning(text, primary_intent_type, intent_scores)
        
        return IntentAnalysis(
            primary_intent=primary_intent_type,
            intent_scores=intent_scores,
            confidence=confidence,
            urgency_level=urgency_level,
            reasoning=reasoning
        )
    
    def _analyze_intent_ml(self, text: str) -> Optional[IntentAnalysis]:
        """ML 모델 기반 의도 분석 (KoBERT)"""
        if not self._ml_model_loaded:
            self._load_ml_model()
        
        if not self.ml_model or not self.ml_tokenizer:
            return None
        
        try:
            # 텍스트 토큰화
            inputs = self.ml_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            
            # GPU로 이동
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.ml_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # 결과를 CPU로 이동 후 numpy 변환
            probs = probabilities.cpu().numpy()[0]
            
            # 의도 점수 초기화 (기본 모델이므로 일반적인 매핑)
            intent_scores = {
                IntentType.QUESTION.value: float(probs[0]) if len(probs) > 0 else 0.0,
                IntentType.REQUEST.value: float(probs[1]) if len(probs) > 1 else 0.0,
                IntentType.GENERAL.value: float(probs[2]) if len(probs) > 2 else 0.5,
            }
            
            # 나머지 의도 유형은 기본값
            for intent_type in IntentType:
                if intent_type.value not in intent_scores:
                    intent_scores[intent_type.value] = 0.0
            
            # 주요 의도 결정
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            primary_intent_type = IntentType(primary_intent[0])
            
            # 신뢰도 계산
            confidence = primary_intent[1]
            
            # 긴급도 평가 (패턴 기반 사용)
            urgency_level = self.assess_urgency(text, {})
            
            # 추론 과정 생성
            reasoning = f"ML 모델 분석 (KoBERT): {primary_intent_type.value} (신뢰도: {confidence:.2f})"
            
            return IntentAnalysis(
                primary_intent=primary_intent_type,
                intent_scores=intent_scores,
                confidence=confidence,
                urgency_level=urgency_level,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.warning(f"ML intent analysis error: {e}")
            return None
    
    def _combine_intent_results(self, pattern_result: IntentAnalysis, ml_result: IntentAnalysis) -> IntentAnalysis:
        """패턴 기반과 ML 모델 결과 결합"""
        # 가중 평균 (패턴 0.4, ML 0.6)
        pattern_weight = 0.4
        ml_weight = 0.6
        
        combined_scores = {}
        for intent_type in IntentType:
            pattern_score = pattern_result.intent_scores.get(intent_type.value, 0.0)
            ml_score = ml_result.intent_scores.get(intent_type.value, 0.0)
            combined_scores[intent_type.value] = pattern_score * pattern_weight + ml_score * ml_weight
        
        # 주요 의도 결정
        primary_intent = max(combined_scores.items(), key=lambda x: x[1])
        primary_intent_type = IntentType(primary_intent[0])
        
        # 신뢰도 계산
        confidence = (pattern_result.confidence * pattern_weight + ml_result.confidence * ml_weight)
        
        # 긴급도는 패턴 결과 사용 (더 정확)
        urgency_level = pattern_result.urgency_level
        
        # 추론 과정 생성
        reasoning = f"하이브리드 분석: 패턴({pattern_result.primary_intent.value}, {pattern_result.confidence:.2f}) + ML({ml_result.primary_intent.value}, {ml_result.confidence:.2f})"
        
        return IntentAnalysis(
            primary_intent=primary_intent_type,
            intent_scores=combined_scores,
            confidence=confidence,
            urgency_level=urgency_level,
            reasoning=reasoning
        )
    
    def get_contextual_response_tone(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                     user_profile: Optional[Dict] = None) -> ResponseTone:
        """
        맥락적 응답 톤 결정
        
        Args:
            emotion: 감정 분석 결과
            intent: 의도 분석 결과
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            ResponseTone: 응답 톤
        """
        try:
            # 기본 톤 결정
            tone_type = self._determine_base_tone(emotion, intent)
            
            # 공감 수준 계산
            empathy_level = self._calculate_empathy_level(emotion, intent)
            
            # 격식 수준 계산
            formality_level = self._calculate_formality_level(emotion, intent, user_profile)
            
            # 긴급 응답 여부
            urgency_response = intent.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]
            
            # 설명 깊이 결정
            explanation_depth = self._determine_explanation_depth(emotion, intent, user_profile)
            
            return ResponseTone(
                tone_type=tone_type,
                empathy_level=empathy_level,
                formality_level=formality_level,
                urgency_response=urgency_response,
                explanation_depth=explanation_depth
            )
            
        except Exception as e:
            self.logger.error(f"Error determining response tone: {e}")
            return ResponseTone(
                tone_type="professional",
                empathy_level=0.5,
                formality_level=0.7,
                urgency_response=False,
                explanation_depth="medium"
            )
    
    def assess_urgency(self, text: str, emotion: Dict[str, float]) -> UrgencyLevel:
        """
        긴급도 평가
        
        Args:
            text: 분석할 텍스트
            emotion: 감정 점수
            
        Returns:
            UrgencyLevel: 긴급도 수준
        """
        try:
            text_lower = text.lower()
            
            # 긴급도 패턴 매칭
            for urgency_level, patterns in self.urgency_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return urgency_level
            
            # 감정 기반 긴급도 조정
            if emotion.get(EmotionType.URGENT.value, 0) > 0.5:
                return UrgencyLevel.HIGH
            elif emotion.get(EmotionType.ANXIOUS.value, 0) > 0.7:
                return UrgencyLevel.MEDIUM
            
            return UrgencyLevel.LOW
            
        except Exception as e:
            self.logger.error(f"Error assessing urgency: {e}")
            return UrgencyLevel.LOW
    
    def _calculate_emotion_intensity(self, text_lower: str) -> float:
        """감정 강도 계산"""
        try:
            intensity_indicators = [
                r"정말|정말로|진짜|진짜로|완전|완전히|너무|너무나",
                r"매우|엄청|엄청나|대단|대단히|극도로|극도로",
                r"!!+|!!!+|!!!+",  # 느낌표 반복
                r"ㅠㅠ|ㅜㅜ|ㅠㅠㅠ|ㅜㅜㅜ",  # 울음 표시
                r"ㅋㅋ|ㅎㅎ|ㅋㅋㅋ|ㅎㅎㅎ"  # 웃음 표시
            ]
            
            intensity = 0.0
            for pattern in intensity_indicators:
                matches = re.findall(pattern, text_lower)
                intensity += len(matches) * 0.1
            
            return min(1.0, intensity)
            
        except Exception as e:
            self.logger.error(f"Error calculating emotion intensity: {e}")
            return 0.0
    
    def _adjust_intent_with_context(self, intent_scores: Dict[str, float], 
                                   context: ConversationContext) -> Dict[str, float]:
        """맥락을 고려한 의도 점수 조정"""
        try:
            # 최근 대화에서 의도 패턴 분석
            if context.turns:
                recent_turns = context.turns[-3:]  # 최근 3턴
                
                # 후속 질문 패턴 감지
                if len(recent_turns) > 1:
                    last_turn = recent_turns[-1]
                    if any(keyword in last_turn.user_query.lower() 
                          for keyword in ["추가", "더", "또한", "그리고"]):
                        intent_scores[IntentType.FOLLOW_UP.value] += 0.2
                
                # 명확화 요청 패턴 감지
                if any(keyword in context.turns[-1].user_query.lower() 
                      for keyword in ["정확히", "구체적으로", "자세히", "예시"]):
                    intent_scores[IntentType.CLARIFICATION.value] += 0.2
            
            return intent_scores
            
        except Exception as e:
            self.logger.error(f"Error adjusting intent with context: {e}")
            return intent_scores
    
    def _determine_base_tone(self, emotion: EmotionAnalysis, intent: IntentAnalysis) -> str:
        """기본 톤 결정"""
        try:
            # 긴급한 경우
            if intent.urgency_level in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
                return "urgent"
            
            # 감사 표현인 경우
            if intent.primary_intent == IntentType.THANKS:
                return "supportive"
            
            # 불만/화남인 경우
            if emotion.primary_emotion in [EmotionType.ANGRY, EmotionType.NEGATIVE]:
                return "empathetic"
            
            # 불안/걱정인 경우
            if emotion.primary_emotion in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                return "empathetic"
            
            # 법률 관련 질문인 경우
            if intent.primary_intent == IntentType.QUESTION:
                return "professional"
            
            # 기본값
            return "professional"
            
        except Exception as e:
            self.logger.error(f"Error determining base tone: {e}")
            return "professional"
    
    def _calculate_empathy_level(self, emotion: EmotionAnalysis, intent: IntentAnalysis) -> float:
        """공감 수준 계산"""
        try:
            empathy_level = 0.5  # 기본값
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.ANXIOUS, EmotionType.CONFUSED]:
                empathy_level += 0.3
            elif emotion.primary_emotion in [EmotionType.ANGRY, EmotionType.NEGATIVE]:
                empathy_level += 0.2
            elif emotion.primary_emotion == EmotionType.POSITIVE:
                empathy_level += 0.1
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.COMPLAINT:
                empathy_level += 0.2
            elif intent.primary_intent == IntentType.THANKS:
                empathy_level += 0.1
            
            return min(1.0, empathy_level)
            
        except Exception as e:
            self.logger.error(f"Error calculating empathy level: {e}")
            return 0.5
    
    def _calculate_formality_level(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                  user_profile: Optional[Dict]) -> float:
        """격식 수준 계산"""
        try:
            formality_level = 0.7  # 기본값 (법률 상담이므로 격식적)
            
            # 사용자 프로필에 따른 조정
            if user_profile:
                expertise_level = user_profile.get("expertise_level", "beginner")
                if expertise_level == "expert":
                    formality_level += 0.2
                elif expertise_level == "beginner":
                    formality_level -= 0.1
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.POSITIVE, EmotionType.SATISFIED]:
                formality_level -= 0.1
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.THANKS:
                formality_level -= 0.1
            
            return max(0.0, min(1.0, formality_level))
            
        except Exception as e:
            self.logger.error(f"Error calculating formality level: {e}")
            return 0.7
    
    def _determine_explanation_depth(self, emotion: EmotionAnalysis, intent: IntentAnalysis, 
                                    user_profile: Optional[Dict]) -> str:
        """설명 깊이 결정"""
        try:
            # 사용자 프로필에 따른 조정
            if user_profile:
                detail_level = user_profile.get("preferred_detail_level", "medium")
                if detail_level == "detailed":
                    return "detailed"
                elif detail_level == "simple":
                    return "simple"
            
            # 의도에 따른 조정
            if intent.primary_intent == IntentType.CLARIFICATION:
                return "detailed"
            elif intent.primary_intent == IntentType.EMERGENCY:
                return "simple"
            
            # 감정에 따른 조정
            if emotion.primary_emotion in [EmotionType.CONFUSED, EmotionType.ANXIOUS]:
                return "detailed"
            elif emotion.primary_emotion == EmotionType.URGENT:
                return "simple"
            
            return "medium"
            
        except Exception as e:
            self.logger.error(f"Error determining explanation depth: {e}")
            return "medium"
    
    def _generate_emotion_reasoning(self, text: str, primary_emotion: EmotionType, 
                                   emotion_scores: Dict[str, float]) -> str:
        """감정 추론 과정 생성"""
        try:
            reasoning_parts = []
            
            # 주요 감정 설명
            emotion_descriptions = {
                EmotionType.POSITIVE: "긍정적 감정",
                EmotionType.NEGATIVE: "부정적 감정",
                EmotionType.NEUTRAL: "중립적 감정",
                EmotionType.URGENT: "긴급한 감정",
                EmotionType.ANXIOUS: "불안한 감정",
                EmotionType.ANGRY: "화난 감정",
                EmotionType.SATISFIED: "만족한 감정",
                EmotionType.CONFUSED: "혼란스러운 감정"
            }
            
            reasoning_parts.append(f"주요 감정: {emotion_descriptions.get(primary_emotion, '알 수 없음')}")
            
            # 높은 점수의 감정들
            high_score_emotions = [emotion for emotion, score in emotion_scores.items() 
                                  if score > 0.3 and emotion != primary_emotion.value]
            if high_score_emotions:
                reasoning_parts.append(f"기타 감정: {', '.join(high_score_emotions)}")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating emotion reasoning: {e}")
            return f"Error: {str(e)}"
    
    def _generate_intent_reasoning(self, text: str, primary_intent: IntentType, 
                                  intent_scores: Dict[str, float]) -> str:
        """의도 추론 과정 생성"""
        try:
            reasoning_parts = []
            
            # 주요 의도 설명
            intent_descriptions = {
                IntentType.QUESTION: "질문 의도",
                IntentType.REQUEST: "요청 의도",
                IntentType.COMPLAINT: "불만 의도",
                IntentType.THANKS: "감사 의도",
                IntentType.CLARIFICATION: "명확화 의도",
                IntentType.FOLLOW_UP: "후속 질문 의도",
                IntentType.EMERGENCY: "긴급 의도",
                IntentType.GENERAL: "일반 의도"
            }
            
            reasoning_parts.append(f"주요 의도: {intent_descriptions.get(primary_intent, '알 수 없음')}")
            
            # 높은 점수의 의도들
            high_score_intents = [intent for intent, score in intent_scores.items() 
                                 if score > 0.3 and intent != primary_intent.value]
            if high_score_intents:
                reasoning_parts.append(f"기타 의도: {', '.join(high_score_intents)}")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating intent reasoning: {e}")
            return f"Error: {str(e)}"


# 테스트 함수
def test_emotion_intent_analyzer():
    """감정 및 의도 분석기 테스트"""
    analyzer = EmotionIntentAnalyzer()
    
    print("=== 감정 및 의도 분석기 테스트 ===")
    
    # 테스트 텍스트들
    test_texts = [
        "손해배상 청구 방법을 알려주세요",
        "급해요! 오늘까지 답변해주세요!",
        "감사합니다. 정말 도움이 되었어요",
        "이해가 안 돼요. 더 자세히 설명해주세요",
        "왜 이런 문제가 생겼나요? 정말 화나네요",
        "추가로 궁금한 것이 있어요",
        "정확히 말하면 어떤 절차인가요?"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 텍스트: {text}")
        
        # 감정 분석
        emotion_result = analyzer.analyze_emotion(text)
        print(f"   감정: {emotion_result.primary_emotion.value} (신뢰도: {emotion_result.confidence:.2f})")
        print(f"   강도: {emotion_result.intensity:.2f}")
        print(f"   추론: {emotion_result.reasoning}")
        
        # 의도 분석
        intent_result = analyzer.analyze_intent(text)
        print(f"   의도: {intent_result.primary_intent.value} (신뢰도: {intent_result.confidence:.2f})")
        print(f"   긴급도: {intent_result.urgency_level.value}")
        print(f"   추론: {intent_result.reasoning}")
        
        # 응답 톤 결정
        response_tone = analyzer.get_contextual_response_tone(emotion_result, intent_result)
        print(f"   응답 톤: {response_tone.tone_type}")
        print(f"   공감 수준: {response_tone.empathy_level:.2f}")
        print(f"   격식 수준: {response_tone.formality_level:.2f}")
        print(f"   긴급 응답: {response_tone.urgency_response}")
        print(f"   설명 깊이: {response_tone.explanation_depth}")
        
        print("-" * 50)


if __name__ == "__main__":
    test_emotion_intent_analyzer()
