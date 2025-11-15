# LawFirmAI Frontend

React 기반 법률 AI 어시스턴트 프론트엔드 애플리케이션

## 기술 스택

- **React 18+** with TypeScript
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Axios** - HTTP 클라이언트
- **Lucide React** - 아이콘
- **React Router** - 라우팅 (필요시)

## 설치 및 실행

### 1. 의존성 설치

```bash
cd frontend
npm install
```

### 2. 개발 서버 실행

```bash
npm run dev
```

애플리케이션이 http://localhost:3000 에서 실행됩니다.

**참고**: 
- 서버가 시작되면 브라우저가 자동으로 열립니다
- 터미널에 표시된 URL로 직접 접속할 수도 있습니다
- `index.html`은 프로젝트 루트(`frontend/`)에 있어야 합니다

### 빌드

```bash
npm run build
```

빌드된 파일은 `dist` 디렉토리에 생성됩니다.

### 미리보기

```bash
npm run preview
```

## 환경 변수

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=법률 AI 어시스턴트
VITE_APP_VERSION=1.0.0
```

## 프로젝트 구조

```
frontend/
├── src/
│   ├── components/      # React 컴포넌트
│   │   ├── chat/       # 채팅 관련 컴포넌트
│   │   ├── common/     # 공통 컴포넌트
│   │   ├── layout/     # 레이아웃 컴포넌트
│   │   ├── sidebar/    # 사이드바 컴포넌트
│   │   └── welcome/    # 웰컴 스크린 컴포넌트
│   ├── hooks/          # React 훅
│   ├── services/       # API 서비스
│   ├── types/          # TypeScript 타입 정의
│   ├── utils/          # 유틸리티 함수
│   ├── App.tsx         # 메인 App 컴포넌트
│   └── main.tsx        # 진입점
├── public/             # 정적 파일
└── package.json        # 프로젝트 설정
```

## 주요 기능

### 채팅 기능
- ✅ 실시간 채팅 인터페이스
- ✅ 파일 업로드 (이미지, PDF, 문서)
- ✅ 스트리밍 답변 지원
- ✅ 메시지 복사 기능
- ✅ 타임스탬프 표시

### 세션 관리
- ✅ 새 상담 시작
- ✅ 세션 목록 조회
- ✅ 세션 이름 변경
- ✅ 세션 복사 및 내보내기
- ✅ 세션 삭제

### 히스토리 관리
- ✅ 날짜별 그룹화 (오늘, 어제, 지난 7일, 지난 30일, 이전)
- ✅ 실시간 검색 (debounce 적용)
- ✅ 카테고리 필터 (전체, 계약/합의, 분쟁/소송, 노동/근로, 부동산, 형사, 기타)
- ✅ 정렬 옵션 (최근 활동순, 생성일순, 제목순, 메시지 많은 순)

### 참고자료 시스템
- ✅ 법령/판례 출처 표시
- ✅ 접이식 참고자료 섹션
- ✅ 원문 보기 모달
- ✅ 인용 하이라이트 ([1], [2] 등)

### 연관 질문 추천
- ✅ 답변 완료 후 관련 질문 3개 제안
- ✅ 원클릭 질문 전송

### 피드백 시스템
- ✅ 답변 평가 (좋아요/싫어요)
- ✅ 피드백 API 연동

### 설정
- ✅ 테마 선택 (라이트/다크/자동)
- ✅ 글꼴 크기 설정
- ✅ 답변 속도 설정
- ✅ 데이터 관리 (내보내기/삭제)

## API 연동

백엔드 API는 `http://localhost:8000`에서 실행되어야 합니다.
프록시 설정은 `vite.config.ts`에 포함되어 있습니다.

## 개발 가이드

### 컴포넌트 작성 규칙

- 컴포넌트는 함수형 컴포넌트로 작성
- TypeScript 타입 정의 필수
- Tailwind CSS 클래스 사용
- 재사용 가능한 컴포넌트는 `components/common`에 배치

### API 서비스 작성 규칙

- 모든 API 호출은 `services` 디렉토리에 배치
- `api.ts`의 Axios 인스턴스 사용
- 에러 처리는 `extractApiError` 함수 사용

### 상태 관리

- React Hooks (useState, useContext, useReducer) 사용
- 커스텀 훅은 `hooks` 디렉토리에 배치

## 주요 컴포넌트

### ChatMessage
사용자 및 AI 메시지를 표시하는 컴포넌트. 참고자료, 연관 질문, 피드백 기능 포함.

### ChatInput
채팅 입력 영역. 멀티라인 텍스트 입력, 파일 업로드, Enter 전송 지원.

### SessionList
세션 목록을 날짜별로 그룹화하여 표시. 검색, 필터, 정렬 기능 포함.

### WelcomeScreen
빈 화면 상태에서 표시되는 웰컴 스크린. 빠른 질문 카드 제공.

## API 연동 예제

### 채팅 메시지 전송

```typescript
import { sendChatMessage } from './services/chatService';

const response = await sendChatMessage({
  message: '계약서 검토 요청',
  session_id: 'session-id',
  enable_checkpoint: true,
});
```

### 세션 목록 조회

```typescript
import { getSessions } from './services/sessionService';

const sessions = await getSessions({
  category: '계약/합의',
  search: '계약서',
  page: 1,
  page_size: 10,
  sort_by: 'updated_at',
  sort_order: 'desc',
});
```

### 스트리밍 채팅

```typescript
import { sendStreamingChatMessage } from './services/chatService';

for await (const chunk of sendStreamingChatMessage({
  message: '질문',
  session_id: 'session-id',
})) {
  // 실시간으로 청크 처리
  console.log(chunk);
}
```

## 문제 해결

### CORS 에러
백엔드 서버의 CORS 설정을 확인하세요. 개발 환경에서는 Vite 프록시가 자동으로 처리합니다.

### API 연결 실패
1. 백엔드 서버가 실행 중인지 확인
2. `VITE_API_BASE_URL` 환경 변수 확인
3. 브라우저 개발자 도구의 네트워크 탭 확인

### 빌드 에러
1. TypeScript 타입 에러 확인: `npm run build`
2. 의존성 재설치: `rm -rf node_modules && npm install`
3. 캐시 삭제: `rm -rf dist && npm run build`

## 추가 자료

- [프론트엔드 개발 가이드](../docs/01_getting_started/frontend_guide.md)
- [API 문서](../docs/07_api/API_Documentation.md)
- [프로젝트 구조](../docs/01_getting_started/project_structure.md)

## 라이선스

이 프로젝트는 LawFirmAI 프로젝트의 일부입니다.

