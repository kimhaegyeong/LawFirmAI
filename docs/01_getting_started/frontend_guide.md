# 프론트엔드 개발 가이드

LawFirmAI는 React 기반의 모던 프론트엔드를 제공합니다.

## 기술 스택

- **React 18+** with TypeScript
- **Vite** - 빠른 빌드 도구
- **Tailwind CSS** - 유틸리티 기반 CSS 프레임워크
- **Axios** - HTTP 클라이언트
- **Lucide React** - 아이콘 라이브러리

## 프로젝트 구조

```
frontend/
├── src/
│   ├── components/         # React 컴포넌트
│   │   ├── chat/          # 채팅 관련 컴포넌트
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ChatHistory.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   ├── ReferencesSection.tsx
│   │   │   └── RelatedQuestions.tsx
│   │   ├── common/        # 공통 컴포넌트
│   │   │   ├── LoadingSpinner.tsx
│   │   │   ├── Modal.tsx
│   │   │   └── FileAttachment.tsx
│   │   ├── layout/        # 레이아웃 컴포넌트
│   │   │   ├── MainLayout.tsx
│   │   │   ├── Header.tsx
│   │   │   └── Sidebar.tsx
│   │   ├── sidebar/       # 사이드바 컴포넌트
│   │   │   ├── SidebarContent.tsx
│   │   │   ├── SessionList.tsx
│   │   │   ├── SessionItem.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   ├── FilterPanel.tsx
│   │   │   └── SettingsPanel.tsx
│   │   └── welcome/        # 웰컴 스크린
│   │       └── WelcomeScreen.tsx
│   ├── hooks/             # React 커스텀 훅
│   │   ├── useChat.ts
│   │   ├── useSession.ts
│   │   ├── useHistory.ts
│   │   └── useStreaming.ts
│   ├── services/          # API 서비스 레이어
│   │   ├── api.ts
│   │   ├── chatService.ts
│   │   ├── sessionService.ts
│   │   ├── historyService.ts
│   │   └── feedbackService.ts
│   ├── types/             # TypeScript 타입 정의
│   │   ├── chat.ts
│   │   ├── session.ts
│   │   ├── history.ts
│   │   ├── feedback.ts
│   │   └── api.ts
│   ├── utils/             # 유틸리티 함수
│   │   ├── dateUtils.ts
│   │   ├── fileUtils.ts
│   │   └── constants.ts
│   ├── App.tsx            # 메인 App 컴포넌트
│   └── main.tsx           # 진입점
├── public/                # 정적 파일
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## 설치 및 실행

### 1. 의존성 설치

```bash
cd frontend
npm install
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=법률 AI 어시스턴트
VITE_APP_VERSION=1.0.0
```

### 3. 개발 서버 실행

```bash
npm run dev
```

애플리케이션이 http://localhost:3000 에서 실행됩니다.

### 4. 빌드

```bash
npm run build
```

빌드된 파일은 `dist` 디렉토리에 생성됩니다.

## 주요 기능

자세한 UI/UX 기능 명세는 [UI 디자인 문서](ui_design.md)를 참조하세요.

주요 기능:
- **채팅 인터페이스**: 실시간 메시지 표시, 파일 첨부, 스트리밍 답변
- **세션 관리**: 세션 생성, 조회, 이름 변경, 삭제
- **히스토리 관리**: 날짜별 그룹화, 검색, 필터링
- **참고자료 시스템**: 법령/판례 출처 표시, 원문 보기
- **연관 질문 추천**: 답변 기반 관련 질문 제안
- **피드백 시스템**: 답변 평가 (좋아요/싫어요)

## 개발 가이드

### 컴포넌트 작성 규칙

1. **함수형 컴포넌트 사용**
   ```typescript
   interface ComponentProps {
     prop1: string;
     prop2?: number;
   }
   
   export function Component({ prop1, prop2 }: ComponentProps) {
     return <div>{prop1}</div>;
   }
   ```

2. **TypeScript 타입 정의 필수**
   - 모든 props에 타입 정의
   - 함수 반환 타입 명시

3. **Tailwind CSS 클래스 사용**
   - 인라인 스타일 지양
   - 유틸리티 클래스 활용

4. **재사용 가능한 컴포넌트 분리**
   - 공통 컴포넌트는 `components/common`에 배치
   - 특정 기능 컴포넌트는 해당 폴더에 배치

### API 서비스 작성 규칙

1. **서비스 파일 구조**
   ```typescript
   import { api, extractApiError } from './api';
   import type { RequestType, ResponseType } from '../types';
   
   export async function serviceFunction(
     request: RequestType
   ): Promise<ResponseType> {
     try {
       const response = await api.post<ResponseType>('/api/endpoint', request);
       return response.data;
     } catch (error) {
       throw extractApiError(error);
     }
   }
   ```

2. **에러 처리**
   - 모든 API 호출은 try-catch로 감싸기
   - `extractApiError` 함수로 일관된 에러 처리

3. **타입 안정성**
   - 요청/응답 타입 명시
   - 타입 정의는 `types` 디렉토리에 분리

### 커스텀 훅 작성 규칙

1. **훅 네이밍**
   - `use` 접두사 사용
   - 명확한 기능명 사용 (예: `useChat`, `useSession`)

2. **상태 관리**
   ```typescript
   export function useCustomHook() {
     const [state, setState] = useState<StateType>(initialState);
     const [isLoading, setIsLoading] = useState(false);
     const [error, setError] = useState<Error | null>(null);
     
     const action = useCallback(async () => {
       setIsLoading(true);
       setError(null);
       try {
         // 비즈니스 로직
       } catch (err) {
         setError(err);
       } finally {
         setIsLoading(false);
       }
     }, []);
     
     return { state, isLoading, error, action };
   }
   ```

3. **의존성 관리**
   - `useCallback`, `useMemo`로 최적화
   - 의존성 배열 명시

## API 연동

### 백엔드 API 엔드포인트

프론트엔드는 다음 API 엔드포인트를 사용합니다:

- `GET /api/sessions` - 세션 목록 조회
- `POST /api/sessions` - 새 세션 생성
- `GET /api/sessions/{session_id}` - 세션 상세 조회
- `PUT /api/sessions/{session_id}` - 세션 업데이트
- `DELETE /api/sessions/{session_id}` - 세션 삭제
- `POST /api/sessions/{session_id}/generate-title` - 제목 생성
- `POST /api/chat` - 채팅 메시지 전송
- `POST /api/chat/stream` - 스트리밍 채팅
- `GET /api/history` - 히스토리 조회
- `POST /api/history/export` - 히스토리 내보내기
- `POST /api/feedback` - 피드백 전송

### 프록시 설정

개발 환경에서 CORS 문제를 피하기 위해 Vite 프록시를 사용합니다:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 스타일링 가이드

### Tailwind CSS 사용

1. **유틸리티 클래스 우선 사용**
   ```tsx
   <div className="flex items-center gap-2 p-4 bg-white rounded-lg shadow">
     <span className="text-blue-600 font-medium">텍스트</span>
   </div>
   ```

2. **반응형 디자인**
   ```tsx
   <div className="w-full md:w-1/2 lg:w-1/3">
     {/* 콘텐츠 */}
   </div>
   ```

3. **상태별 스타일**
   ```tsx
   <button className={`
     px-4 py-2 rounded-lg
     ${isActive ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}
     hover:bg-blue-700 transition-colors
   `}>
     버튼
   </button>
   ```

## 성능 최적화

### 코드 스플리팅
- React.lazy를 사용한 지연 로딩
- 라우트별 코드 스플리팅

### 메모이제이션
- `useMemo`로 계산 결과 캐싱
- `useCallback`으로 함수 재생성 방지

### 이미지 최적화
- WebP 형식 사용
- 지연 로딩 적용

## 배포

### 빌드

```bash
npm run build
```

### 프로덕션 빌드 확인

```bash
npm run preview
```

### 환경 변수

프로덕션 환경에서는 `.env.production` 파일을 사용하거나 환경 변수를 설정하세요:

```env
VITE_API_BASE_URL=https://api.lawfirmai.com
VITE_APP_NAME=법률 AI 어시스턴트
VITE_APP_VERSION=1.0.0
```

## 문제 해결

### CORS 에러
- 백엔드 CORS 설정 확인
- Vite 프록시 설정 확인

### API 연결 실패
- 백엔드 서버가 실행 중인지 확인
- `VITE_API_BASE_URL` 환경 변수 확인

### 빌드 에러
- TypeScript 타입 에러 확인
- 의존성 버전 충돌 확인

## 참고 자료

- [React 공식 문서](https://react.dev/)
- [Vite 공식 문서](https://vitejs.dev/)
- [Tailwind CSS 공식 문서](https://tailwindcss.com/)
- [TypeScript 공식 문서](https://www.typescriptlang.org/)

