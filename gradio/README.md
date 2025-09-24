# LawFirmAI - Gradio Interface

법률 AI 어시스턴트의 Gradio 웹 인터페이스입니다.

## 기능

- 법률 관련 질문 답변
- 판례 검색 및 분석
- 법령 해설
- 계약서 분석
- Q&A 서비스

## 실행 방법

### 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python app.py
```

### Docker를 사용한 실행

```bash
# Docker 이미지 빌드
docker build -t lawfirm-gradio .

# 컨테이너 실행
docker run -p 7860:7860 lawfirm-gradio
```

### Docker Compose를 사용한 실행

```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down
```

## 접속

애플리케이션 실행 후 http://localhost:7860 에서 접속할 수 있습니다.

## 환경 변수

`.env.example` 파일을 참고하여 환경 변수를 설정하세요.

## 개발

개발 환경에서는 `DEBUG=true`로 설정하여 디버그 모드를 활성화할 수 있습니다.
