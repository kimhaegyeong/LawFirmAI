# 기술 스택별 규칙

## 1. Pydantic V2 규칙 (CRITICAL)

**모든 Pydantic 모델은 반드시 Pydantic V2 스타일을 사용해야 합니다.**

### 원칙
1. **Validator 사용**
   - `@validator` (V1) 사용 금지
   - `@field_validator` (V2) 사용 필수
   - 모든 validator는 `@classmethod` 데코레이터와 함께 사용

2. **Import 규칙**
   ```python
   # ❌ 나쁜 예: Pydantic V1 스타일
   from pydantic import BaseModel, Field, validator
   
   # ✅ 좋은 예: Pydantic V2 스타일
   from pydantic import BaseModel, Field, field_validator
   ```

3. **Validator 작성 규칙**
   ```python
   from pydantic import BaseModel, Field, field_validator
   from typing import Optional
   import re
   
   class ExampleModel(BaseModel):
       """예제 모델"""
       session_id: Optional[str] = Field(None, description="세션 ID")
       message: str = Field(..., min_length=1, max_length=10000)
       
       # ❌ 나쁜 예: Pydantic V1 스타일
       @validator('session_id')
       def validate_session_id(cls, v):
           if v:
               uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
               if not re.match(uuid_pattern, v, re.IGNORECASE):
                   raise ValueError('유효하지 않은 세션 ID 형식입니다')
           return v
       
       # ✅ 좋은 예: Pydantic V2 스타일
       @field_validator('session_id')
       @classmethod
       def validate_session_id(cls, v):
           if v:
               uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
               if not re.match(uuid_pattern, v, re.IGNORECASE):
                   raise ValueError('유효하지 않은 세션 ID 형식입니다')
           return v
       
       @field_validator('message')
       @classmethod
       def validate_message(cls, v):
           if not v or not v.strip():
               raise ValueError('메시지는 비어있을 수 없습니다')
           return v.strip()
   ```

4. **Config 클래스 규칙**
   ```python
   from pydantic import BaseModel, ConfigDict
   
   class ExampleModel(BaseModel):
       """예제 모델"""
       # ❌ 나쁜 예: Pydantic V1 스타일
       class Config:
           from_attributes = True
           extra = "ignore"
       
       # ✅ 좋은 예: Pydantic V2 스타일
       model_config = ConfigDict(
           from_attributes=True,
           extra="ignore"
       )
   ```

5. **금지 사항**
   - `@validator` 사용 금지
   - `from pydantic import validator` 사용 금지
   - Pydantic V1 스타일의 Config 클래스 사용 금지 (가능한 경우)

6. **마이그레이션 체크리스트**
   - [ ] 모든 `@validator`를 `@field_validator`로 변경했는가?
   - [ ] 모든 validator 함수에 `@classmethod` 데코레이터를 추가했는가?
   - [ ] import 문에서 `validator`를 `field_validator`로 변경했는가?
   - [ ] Config 클래스를 `model_config`로 변경했는가? (가능한 경우)

## 2. FastAPI 규칙
```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from lawfirm_langgraph.core.workflow.workflow_service import LangGraphWorkflowService

app = FastAPI(title="LawFirmAI API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]

workflow_service = LangGraphWorkflowService()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        result = await workflow_service.process_query(request.message, request.session_id)
        return ChatResponse(
            response=result["answer"],
            confidence=result.get("confidence", 0.9),
            sources=result.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 3. React 규칙
```typescript
// frontend/src/App.tsx
import React, { useState } from 'react';
import { ChatService } from './services/chatService';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const chatService = new ChatService();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await chatService.processMessage(message);
    setResponse(result);
  };

  return (
    <div className="app">
      <h1>법률 AI 어시스턴트</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="법률 관련 질문을 입력하세요"
        />
        <button type="submit">전송</button>
      </form>
      {response && <div className="response">{response}</div>}
    </div>
  );
}

export default App;
```

## 4. AI 모델 규칙
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델 로딩"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """응답 생성"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
```

## 5. 한국어 불용어 처리 규칙 (CRITICAL)

**한국어 텍스트 처리 시 KoNLPy를 우선적으로 사용하여 불용어를 처리합니다.**

### 원칙
1. **KoNLPy 우선 사용**
   - 불용어 처리는 KoNLPy의 형태소 분석을 우선적으로 사용
   - KoNLPy가 사용 불가능한 경우에만 폴백(fallback) 방법 사용
   - 형태소 분석을 통해 조사와 어미를 자동으로 제거

2. **KoNLPy 초기화 패턴**
   ```python
   # ✅ 좋은 예: KoNLPy 우선 사용 패턴
   class TextProcessor:
       def __init__(self):
           self._okt = None
           try:
               from konlpy.tag import Okt
               self._okt = Okt()
               self.logger.debug("KoNLPy Okt initialized successfully")
           except ImportError:
               self.logger.debug("KoNLPy not available, will use fallback method")
           except Exception as e:
               self.logger.warning(f"Error initializing KoNLPy: {e}, will use fallback method")
   ```

3. **불용어 처리 구현 규칙**
   ```python
   from typing import List, Set, Optional
   from konlpy.tag import Okt
   
   class KoreanStopwordProcessor:
       """한국어 불용어 처리 클래스"""
       
       # 기본 불용어 목록 (법률 도메인 특화)
       BASIC_STOPWORDS: Set[str] = {
           # 기본 조사
           '에', '대해', '설명해주세요', '설명', '의', '을', '를', '이', '가', '는', '은',
           '으로', '로', '에서', '에게', '한테', '께', '와', '과', '하고', '그리고',
           # 질문/요청 표현
           '어떻게', '무엇', '언제', '어디', '어떤', '무엇인가', '요청', '질문',
           # 법률 도메인 불용어
           '법률', '법', '조문', '항', '호', '목', '단', '절', '장', '편'
       }
       
       def __init__(self):
           self._okt: Optional[Okt] = None
           try:
               from konlpy.tag import Okt
               self._okt = Okt()
           except (ImportError, Exception):
               self._okt = None
       
       def remove_stopwords(self, text: str) -> List[str]:
           """불용어 제거 (KoNLPy 우선 사용)"""
           if self._okt:
               # KoNLPy 형태소 분석 사용
               morphs = self._okt.morphs(text)
               # 조사, 어미 제거 (명사, 동사, 형용사만 추출)
               pos_tags = self._okt.pos(text)
               filtered = [
                   word for word, pos in pos_tags
                   if pos in ['Noun', 'Verb', 'Adjective'] 
                   and word not in self.BASIC_STOPWORDS
                   and len(word) >= 2
               ]
               return filtered
           else:
               # 폴백: 정규식 기반 처리
               import re
               words = re.findall(r'[가-힣]+', text)
               return [
                   w for w in words
                   if w not in self.BASIC_STOPWORDS and len(w) >= 2
               ]
   ```

4. **형태소 분석 우선 사용**
   ```python
   # ✅ 좋은 예: KoNLPy 형태소 분석 우선
   def extract_keywords(self, query: str) -> List[str]:
       if self._okt:
           # KoNLPy 형태소 분석 사용
           pos_tags = self._okt.pos(query)
           keywords = [
               word for word, pos in pos_tags
               if pos in ['Noun', 'Verb', 'Adjective']
               and word not in self.BASIC_STOPWORDS
               and len(word) >= 2
           ]
           return keywords
       else:
           # 폴백: 정규식 기반 처리
           return self._fallback_extraction(query)
   ```

5. **금지 사항**
   - 수동 불용어 리스트만 사용하는 방식 금지 (KoNLPy 사용 가능한 경우)
   - KoNLPy를 사용하지 않고 정규식만 사용하는 방식 금지
   - KoNLPy 초기화 실패 시 에러를 발생시키는 방식 금지 (폴백 필수)

6. **예외 처리**
   ```python
   # ✅ 좋은 예: 안전한 KoNLPy 사용
   try:
       from konlpy.tag import Okt
       okt = Okt()
       result = okt.pos(text)
   except ImportError:
       # KoNLPy 미설치 시 폴백
       result = self._fallback_method(text)
   except Exception as e:
       # 기타 오류 시 폴백
       logger.warning(f"KoNLPy error: {e}, using fallback")
       result = self._fallback_method(text)
   ```

7. **의존성 관리**
   - `requirements.txt`에 `konlpy>=0.6.0` 포함 권장
   - KoNLPy는 선택적 의존성으로 처리 (ImportError 처리 필수)
   - KoNLPy 미설치 시에도 동작하도록 폴백 구현 필수

## 6. Batch File (.bat) 규칙
```batch
@echo off
REM All REM comments in batch files must be written in English only
REM This ensures compatibility across different Windows locales and prevents encoding issues
REM Example of correct usage:
REM Check if virtual environment exists
REM Activate virtual environment
REM Set environment variables

REM Incorrect examples (DO NOT USE):
REM 가상 환경 확인
REM 환경 변수 설정

REM Environment variables (defaults)
if not defined API_HOST set "API_HOST=0.0.0.0"
if not defined API_PORT set "API_PORT=8000"
if not defined LOG_LEVEL set "LOG_LEVEL=info"

REM Start server
echo [INFO] Starting server...
python -m uvicorn api.main:app --host %API_HOST% --port %API_PORT%
```

**Batch File 규칙 요약:**
- **REM 주석**: 반드시 영어로만 작성
- **이유**: 다양한 Windows 로케일에서 호환성 보장 및 인코딩 문제 방지
- **echo 메시지**: 사용자에게 표시되는 메시지는 한글 사용 가능 (인코딩 설정 필요 시)
- **변수명**: 영어 사용 권장

