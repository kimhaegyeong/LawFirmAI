# 프론트엔드 정적 테스트 및 보안 검사 수정 체크리스트

## 검사 결과 요약

- **ESLint**: 162개 문제 (104개 오류, 58개 경고)
- **TypeScript**: 50개 이상 타입 오류
- **npm audit**: 2개 중간 심각도 취약점

## 수정할 사항 목록

### 1. 사용하지 않는 변수/함수/Import 제거

#### 1.1 App.tsx
- [ ] 13:10 - `LoginPage` import 제거 (사용하지 않음)
- [ ] 38:11 - `user` 변수 제거 또는 사용
- [ ] 47:10 - `messageBuffers` 변수 제거 또는 사용

#### 1.2 ChatInput.tsx
- [ ] 7:26 - `formatFileSize` import 제거 또는 사용

#### 1.3 ChatMessage.tsx
- [ ] 4:65 - `ChevronDown` import 제거
- [ ] 146:11 - `node` 매개변수를 `_node`로 변경
- [ ] 175:12 - `node` 매개변수를 `_node`로 변경
- [ ] 178:12 - `node` 매개변수를 `_node`로 변경
- [ ] 182:12 - `node` 매개변수를 `_node`로 변경
- [ ] 185:12 - `node` 매개변수를 `_node`로 변경
- [ ] 188:12 - `node` 매개변수를 `_node`로 변경
- [ ] 192:16 - `node` 매개변수를 `_node`로 변경
- [ ] 195:12 - `node` 매개변수를 `_node`로 변경
- [ ] 199:20 - `node` 매개변수를 `_node`로 변경
- [ ] 203:15 - `node` 매개변수를 `_node`로 변경
- [ ] 208:12 - `node` 매개변수를 `_node`로 변경
- [ ] 211:12 - `node` 매개변수를 `_node`로 변경
- [ ] 215:12 - `node` 매개변수를 `_node`로 변경
- [ ] 219:11 - `node` 매개변수를 `_node`로 변경

#### 1.4 DocumentSidebar.tsx
- [ ] 28:3 - `metadata` 매개변수를 `_metadata`로 변경

#### 1.5 ErrorMessage.tsx
- [ ] 4:8 - `React` import 제거

#### 1.6 ErrorBoundary.tsx
- [ ] 4:8 - `React` import 제거

#### 1.7 FileAttachment.tsx
- [ ] 4:23 - `FileImage` import 제거 또는 사용

#### 1.8 Toast.tsx
- [ ] 4:8 - `React` import 제거

#### 1.9 SessionList.tsx
- [ ] 241:28 - `user` 변수 제거 또는 사용

#### 1.10 useHistory.ts
- [ ] 7:10 - `groupSessionsByDate` import 제거 또는 사용

#### 1.11 useSession.ts
- [ ] 4:33 - `useEffect` import 제거
- [ ] 14:8 - `logger` 변수 제거 또는 사용

#### 1.12 useTypingEffect.ts
- [ ] 67:11 - `bufferIncrease` 변수 제거 또는 사용

#### 1.13 chatService.ts
- [ ] 249:9 - `hasReceivedData` 변수 제거 또는 사용

#### 1.14 vite.config.ts
- [ ] 24:48 - `res` 매개변수를 `_res`로 변경
- [ ] 33:48 - `res` 매개변수를 `_res`로 변경
- [ ] 52:35 - `res` 매개변수를 `_res`로 변경

### 2. any 타입 구체화

#### 2.1 App.tsx
- [ ] 515:33 - `any` 타입을 구체적인 타입으로 변경
- [ ] 745:33 - `any` 타입을 구체적인 타입으로 변경

#### 2.2 ChatMessage.tsx
- [ ] 122:47 - `any` 타입을 구체적인 타입으로 변경
- [ ] 129:35 - `any` 타입을 구체적인 타입으로 변경
- [ ] 146:52 - `any` 타입을 구체적인 타입으로 변경
- [ ] 175:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 178:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 182:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 185:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 188:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 192:34 - `any` 타입을 구체적인 타입으로 변경
- [ ] 195:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 199:38 - `any` 타입을 구체적인 타입으로 변경
- [ ] 203:33 - `any` 타입을 구체적인 타입으로 변경
- [ ] 208:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 211:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 215:30 - `any` 타입을 구체적인 타입으로 변경
- [ ] 219:29 - `any` 타입을 구체적인 타입으로 변경

#### 2.3 DocumentSidebar.tsx
- [ ] 17:14 - `any` 타입을 구체적인 타입으로 변경

#### 2.4 ReferencesModalContent.tsx
- [ ] 981:48 - `any` 타입을 구체적인 타입으로 변경

#### 2.5 api.ts
- [ ] 91:43 - `any` 타입을 구체적인 타입으로 변경
- [ ] 212:40 - `any` 타입을 구체적인 타입으로 변경
- [ ] 216:48 - `any` 타입을 구체적인 타입으로 변경
- [ ] 237:20 - `any` 타입을 구체적인 타입으로 변경
- [ ] 238:20 - `any` 타입을 구체적인 타입으로 변경
- [ ] 247:22 - `any` 타입을 구체적인 타입으로 변경
- [ ] 252:20 - `any` 타입을 구체적인 타입으로 변경

#### 2.6 chatService.ts
- [ ] 545:19 - `any` 타입을 구체적인 타입으로 변경
- [ ] 556:23 - `any` 타입을 구체적인 타입으로 변경

#### 2.7 types/api.ts
- [ ] 5:34 - `any` 타입을 구체적인 타입으로 변경

#### 2.8 types/chat.ts
- [ ] 59:20 - `any` 타입을 구체적인 타입으로 변경
- [ ] 74:28 - `any` 타입을 구체적인 타입으로 변경
- [ ] 113:20 - `any` 타입을 구체적인 타입으로 변경

#### 2.9 types/history.ts
- [ ] 11:29 - `any` 타입을 구체적인 타입으로 변경

#### 2.10 types/session.ts
- [ ] 12:29 - `any` 타입을 구체적인 타입으로 변경

#### 2.11 utils/dateUtils.ts
- [ ] 10:13 - `any` 타입을 구체적인 타입으로 변경
- [ ] 110:47 - `any` 타입을 구체적인 타입으로 변경
- [ ] 111:36 - `any` 타입을 구체적인 타입으로 변경

#### 2.12 utils/streamParser.ts
- [ ] 25:20 - `any` 타입을 구체적인 타입으로 변경

### 3. React Hooks 규칙 위반 수정

#### 3.1 ChatMessage.tsx
- [ ] 79:73 - `useTypingEffect` Hook을 조건문 밖으로 이동
- [ ] 89:27 - `useMemo` Hook을 조건문 밖으로 이동
- [ ] 97:29 - `useMemo` Hook을 조건문 밖으로 이동
- [ ] 120:30 - `useMemo` Hook을 조건문 밖으로 이동

### 4. 접근성 문제 수정

#### 4.1 ChatMessage.tsx
- [ ] 183:7 - 헤딩 요소에 내용 추가
- [ ] 186:7 - 헤딩 요소에 내용 추가
- [ ] 189:7 - 헤딩 요소에 내용 추가

#### 4.2 ReferencesModalContent.tsx
- [ ] 388:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가 (`onKeyDown`, `role`, `tabIndex`)
- [ ] 464:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가
- [ ] 546:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가
- [ ] 621:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가
- [ ] 694:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가

#### 4.3 Modal.tsx
- [ ] 61:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가
- [ ] 65:7 - 클릭 이벤트에 키보드 이벤트 핸들러 추가

#### 4.4 SessionItem.tsx
- [ ] 75:5 - 클릭 이벤트에 키보드 이벤트 핸들러 추가

### 5. 보안 문제 수정

#### 5.1 ReferencesModalContent.tsx
- [ ] 234:26 - 안전하지 않은 정규식 수정 (ReDoS 방지)
- [ ] 248:15 - 불필요한 이스케이프 문자 제거 (`\(` → `(`)
- [ ] 248:25 - 불필요한 이스케이프 문자 제거 (`\)` → `)`)
- [ ] 297:14 - 불필요한 이스케이프 문자 제거 (`\-` → `-`)
- [ ] 297:26 - 불필요한 이스케이프 문자 제거 (`\-` → `-`)

#### 5.2 ChatMessage.tsx
- [ ] 108:23 - 비리터럴 정규식 사용 검토 및 안전성 확인

#### 5.3 Object Injection 경고 (58개)
- [ ] App.tsx - 동적 객체 접근 검토 및 입력 검증 추가
- [ ] ChatMessage.tsx - 동적 객체 접근 검토
- [ ] DocumentSidebar.tsx - 동적 객체 접근 검토
- [ ] LoadingSpinner.tsx - 동적 객체 접근 검토
- [ ] Modal.tsx - 동적 객체 접근 검토
- [ ] Toast.tsx - 동적 객체 접근 검토
- [ ] SessionList.tsx - 동적 객체 접근 검토
- [ ] dateUtils.ts - 동적 객체 접근 검토
- [ ] ReferencesModalContent.tsx - 동적 객체 접근 검토

### 6. TypeScript 타입 오류 수정

#### 6.1 App.tsx
- [ ] 365:21 - `undefined` 가능성 처리 추가
- [ ] 374:17 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 379:24 - `undefined` 가능성 처리 추가
- [ ] 381:23 - `undefined` 가능성 처리 추가
- [ ] 445:19 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 499:17 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 503:24 - `undefined` 가능성 처리 추가
- [ ] 504:64 - `undefined` 가능성 처리 추가
- [ ] 541:19 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 544:26 - `undefined` 가능성 처리 추가
- [ ] 548:55 - `undefined` 가능성 처리 추가
- [ ] 555:40 - `undefined` 가능성 처리 추가
- [ ] 619:17 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 623:24 - `undefined` 가능성 처리 추가
- [ ] 628:46 - `undefined` 가능성 처리 추가
- [ ] 696:27 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 699:34 - `undefined` 가능성 처리 추가
- [ ] 709:48 - `undefined` 가능성 처리 추가
- [ ] 763:17 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 827:11 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 831:18 - `undefined` 가능성 처리 추가
- [ ] 932:13 - `ChatMessage` 타입의 `id` 필드 필수값 처리
- [ ] 935:20 - `undefined` 가능성 처리 추가
- [ ] 943:45 - `undefined` 가능성 처리 추가

#### 6.2 LoginPage.tsx
- [ ] 21:7 - `setError` 함수 정의 추가
- [ ] 37:11 - `setError` 함수 정의 추가

#### 6.3 ReferenceDetailView.tsx
- [ ] 267:23 - `LegalReferenceDetail` 타입에 `url` 속성 추가 또는 타입 수정
- [ ] 270:33 - `LegalReferenceDetail` 타입에 `url` 속성 추가 또는 타입 수정

#### 6.4 ReferencesSection.tsx
- [ ] 85:26 - 타입에 `link` 속성 추가 또는 타입 가드 사용
- [ ] 87:35 - 타입에 `link` 속성 추가 또는 타입 가드 사용

#### 6.5 ErrorBoundary.tsx
- [ ] 34:3 - `override` 키워드 추가
- [ ] 42:3 - `override` 키워드 추가

#### 6.6 SessionItem.tsx
- [ ] 31:13 - 모든 코드 경로에서 반환값 확인

#### 6.7 chatService.ts
- [ ] 513:47 - `undefined` 가능성 처리 추가

#### 6.8 types/error.ts
- [ ] 49:43 - `undefined` 가능성 처리 추가

### 7. 기타 문제 수정

#### 7.1 api.ts
- [ ] 116:11 - case 블록 내 변수 선언을 블록으로 감싸기
- [ ] 117:11 - case 블록 내 변수 선언을 블록으로 감싸기

#### 7.2 DeleteAccountModal.tsx
- [ ] 90:65 - 따옴표 이스케이프 처리 (`"` → `&quot;` 또는 `&ldquo;`)
- [ ] 90:80 - 따옴표 이스케이프 처리

### 8. npm 보안 취약점 수정

#### 8.1 의존성 업데이트
- [ ] esbuild 취약점 수정 (vite 업데이트 필요)
  - 현재: esbuild <=0.24.2 (중간 심각도)
  - 수정: `npm audit fix --force` 실행 (vite@7.2.2로 업데이트, breaking change 주의)

## 우선순위별 수정 가이드

### 우선순위 1: 즉시 수정 필요 (에러)
1. 사용하지 않는 변수/함수 제거
2. React Hooks 규칙 위반 수정
3. TypeScript 타입 오류 수정
4. 접근성 문제 수정 (키보드 이벤트)

### 우선순위 2: 점진적 수정 (경고)
1. any 타입 구체화
2. Object Injection 경고 검토
3. 보안 정규식 수정

### 우선순위 3: 의존성 업데이트
1. npm 보안 취약점 수정 (breaking change 주의)

## 수정 방법 예시

### 사용하지 않는 변수 제거
```typescript
// ❌ 나쁜 예
import { LoginPage } from './components/auth/LoginPage';
const unusedVar = 'test';

// ✅ 좋은 예
// import 제거 또는 변수 사용
```

### any 타입 구체화
```typescript
// ❌ 나쁜 예
function processData(data: any) {
  return data.value;
}

// ✅ 좋은 예
interface Data {
  value: string;
}
function processData(data: Data) {
  return data.value;
}
```

### React Hooks 규칙 준수
```typescript
// ❌ 나쁜 예
function Component() {
  if (condition) {
    return null;
  }
  const [state, setState] = useState(0);
  return <div>{state}</div>;
}

// ✅ 좋은 예
function Component() {
  const [state, setState] = useState(0);
  if (condition) {
    return null;
  }
  return <div>{state}</div>;
}
```

### 접근성 개선
```typescript
// ❌ 나쁜 예
<div onClick={handleClick}>Click me</div>

// ✅ 좋은 예
<div
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick();
    }
  }}
  role="button"
  tabIndex={0}
>
  Click me
</div>
```

### TypeScript 타입 안전성
```typescript
// ❌ 나쁜 예
const value = obj[key]; // obj[key]가 undefined일 수 있음

// ✅ 좋은 예
const value = obj[key];
if (value !== undefined) {
  // 사용
}
// 또는
const value = obj[key] ?? defaultValue;
```

## 검증 방법

수정 후 다음 명령어로 검증:

```bash
# ESLint 검사
npm run lint

# TypeScript 타입 체크
npm run type-check

# 보안 감사
npm run security:audit

# 전체 검사
npm run security:check
```

## 참고

- 자동 수정 가능한 항목: `npm run lint:fix`
- 특정 규칙 비활성화: `// eslint-disable-next-line rule-name`
- 상세한 가이드는 `frontend_static_analysis_and_security.md` 참조

