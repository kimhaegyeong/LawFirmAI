# 프론트엔드 정적 분석 및 보안 가이드

## 개요

이 문서는 LawFirmAI 프로젝트의 React 프론트엔드 코드에 대한 정적 분석(Static Analysis)과 보안 코딩 점검을 위한 설정 및 사용 가이드를 제공합니다.

## 목차

1. [설정 개요](#설정-개요)
2. [ESLint 설정](#eslint-설정)
3. [TypeScript Strict 모드](#typescript-strict-모드)
4. [보안 플러그인](#보안-플러그인)
5. [사용 방법](#사용-방법)
6. [CI/CD 통합](#cicd-통합)
7. [주요 검사 항목](#주요-검사-항목)
8. [문제 해결 가이드](#문제-해결-가이드)

## 설정 개요

프론트엔드 정적 분석 및 보안 검사를 위해 다음 도구들이 설정되어 있습니다:

- **ESLint**: JavaScript/TypeScript 코드 품질 및 스타일 검사
- **TypeScript**: 타입 안전성 검사
- **eslint-plugin-security**: 보안 취약점 검사
- **eslint-plugin-jsx-a11y**: 접근성 검사
- **npm audit**: 의존성 보안 검사

## ESLint 설정

### 설정 파일 위치

```
frontend/.eslintrc.json
```

### 주요 설정 내용

#### Extends (확장 규칙)

```json
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react-hooks/recommended",
    "plugin:react/recommended",
    "plugin:react/jsx-runtime",
    "plugin:jsx-a11y/recommended"
  ]
}
```

#### Plugins (플러그인)

- `react-refresh`: React Fast Refresh 지원
- `@typescript-eslint`: TypeScript 규칙
- `react`: React 규칙
- `security`: 보안 규칙
- `jsx-a11y`: 접근성 규칙

#### 주요 규칙

##### TypeScript 규칙

- `@typescript-eslint/no-unused-vars`: 사용하지 않는 변수 검사
- `@typescript-eslint/no-explicit-any`: `any` 타입 사용 금지

##### React 규칙

- `react/prop-types`: PropTypes 검사 (off - TypeScript 사용)
- `react/react-in-jsx-scope`: React import 불필요 (off - React 17+)
- `react-refresh/only-export-components`: 컴포넌트만 export 검사

##### 보안 규칙

- `security/detect-object-injection`: 객체 주입 공격 검사
- `security/detect-eval-with-expression`: eval 사용 검사 (error)
- `security/detect-unsafe-regex`: 안전하지 않은 정규식 검사 (error)
- `security/detect-new-buffer`: Buffer 생성 검사 (error)
- `security/detect-disable-mustache-escape`: Mustache 이스케이프 비활성화 검사 (error)
- `security/detect-pseudoRandomBytes`: 의사 난수 생성 검사 (error)
- `security/detect-non-literal-regexp`: 비리터럴 정규식 검사 (warn)
- `security/detect-non-literal-require`: 비리터럴 require 검사 (warn)
- `security/detect-possible-timing-attacks`: 타이밍 공격 가능성 검사 (warn)

##### 접근성 규칙

- `jsx-a11y/click-events-have-key-events`: 클릭 이벤트에 키보드 이벤트 필요
- `jsx-a11y/no-static-element-interactions`: 정적 요소의 상호작용 검사
- `jsx-a11y/heading-has-content`: 헤딩 요소에 내용 필요

## TypeScript Strict 모드

### 설정 파일 위치

```
frontend/tsconfig.json
```

### Strict 모드 옵션

```json
{
  "compilerOptions": {
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitReturns": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true
  }
}
```

#### 옵션 설명

- **strict**: 모든 strict 타입 체크 옵션 활성화
- **noUnusedLocals**: 사용하지 않는 지역 변수 검사
- **noUnusedParameters**: 사용하지 않는 매개변수 검사
- **noFallthroughCasesInSwitch**: switch 문의 fallthrough 케이스 검사
- **noImplicitReturns**: 모든 코드 경로에서 반환값 확인
- **noUncheckedIndexedAccess**: 배열/객체 인덱스 접근 시 undefined 가능성 고려
- **noImplicitOverride**: override 키워드 명시적 사용

## 보안 플러그인

### eslint-plugin-security

보안 취약점을 검사하는 플러그인입니다.

#### 주요 검사 항목

1. **Object Injection**: 동적 객체 속성 접근 시 주입 공격 가능성
2. **Eval 사용**: `eval()` 함수 사용 검사
3. **Unsafe Regex**: ReDoS(Regular Expression Denial of Service) 공격 가능성
4. **Buffer 생성**: 안전하지 않은 Buffer 생성 검사
5. **CSRF 보호**: CSRF 토큰 검증 누락 검사
6. **타이밍 공격**: 타이밍 공격 가능성 검사

### eslint-plugin-jsx-a11y

React 접근성 검사를 위한 플러그인입니다.

#### 주요 검사 항목

1. **키보드 접근성**: 클릭 이벤트에 키보드 이벤트 필요
2. **ARIA 속성**: 접근성을 위한 ARIA 속성 검사
3. **시맨틱 HTML**: 적절한 HTML 요소 사용 검사
4. **포커스 관리**: 포커스 가능한 요소 검사

## 사용 방법

### 로컬에서 실행

#### ESLint 검사

```bash
cd frontend
npm run lint
```

#### 자동 수정

```bash
npm run lint:fix
```

#### TypeScript 타입 체크

```bash
npm run type-check
```

#### 보안 감사

```bash
# npm 보안 감사
npm run security:audit

# 자동 수정 가능한 보안 이슈 수정
npm run security:audit:fix

# 전체 보안 검사 (audit + lint)
npm run security:check

# 보안 테스트
npm run test:security
```

### package.json 스크립트

```json
{
  "scripts": {
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix",
    "type-check": "tsc --noEmit",
    "security:audit": "npm audit",
    "security:audit:fix": "npm audit fix",
    "security:check": "npm audit && npm run lint",
    "test:security": "npm run security:audit && npm run lint"
  }
}
```

## CI/CD 통합

### GitHub Actions 워크플로우

#### 파일 위치

```
.github/workflows/security-check.yml
```

#### 워크플로우 내용

```yaml
name: Security Check

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      
      - name: Run ESLint
        run: |
          cd frontend
          npm run lint
      
      - name: Run TypeScript type check
        run: |
          cd frontend
          npm run type-check
      
      - name: Run npm audit
        run: |
          cd frontend
          npm audit --audit-level=moderate
        continue-on-error: true
      
      - name: Security check summary
        if: always()
        run: |
          echo "Security checks completed"
```

#### 실행 시점

- `push`: 모든 브랜치에 push 시
- `pull_request`: Pull Request 생성/업데이트 시

## 주요 검사 항목

### 1. 코드 품질

#### 사용하지 않는 변수/함수

```typescript
// ❌ 나쁜 예
const unusedVar = 'test';
const unusedFunction = () => {};

// ✅ 좋은 예
const usedVar = 'test';
console.log(usedVar);
```

#### any 타입 사용

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

### 2. 보안 취약점

#### Object Injection

```typescript
// ❌ 나쁜 예 (경고 발생)
const obj: Record<string, any> = {};
const key = userInput; // 사용자 입력
obj[key] = value;

// ✅ 좋은 예
const obj: Record<string, any> = {};
const key = sanitizeInput(userInput); // 입력 검증
obj[key] = value;
```

#### Eval 사용

```typescript
// ❌ 나쁜 예 (에러 발생)
const code = userInput;
eval(code);

// ✅ 좋은 예
// eval 사용 금지, 대안 사용
```

#### Unsafe Regex

```typescript
// ❌ 나쁜 예 (에러 발생)
const regex = new RegExp(`(${userInput})+$`);

// ✅ 좋은 예
const sanitized = escapeRegex(userInput);
const regex = new RegExp(`(${sanitized})+$`);
```

### 3. 접근성

#### 키보드 접근성

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

#### 시맨틱 HTML

```typescript
// ❌ 나쁜 예
<div className="heading">Title</div>

// ✅ 좋은 예
<h1>Title</h1>
```

### 4. React Hooks 규칙

```typescript
// ❌ 나쁜 예
function Component() {
  if (condition) {
    return null;
  }
  const [state, setState] = useState(0); // 조건부 Hook
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

## 문제 해결 가이드

### 일반적인 문제

#### 1. 사용하지 않는 변수

**문제**: `'variable' is defined but never used`

**해결 방법**:
- 변수 제거
- 변수명 앞에 `_` 추가 (예: `_unusedVar`)
- ESLint 설정에서 `argsIgnorePattern: "^_"` 사용

#### 2. any 타입 사용

**문제**: `Unexpected any. Specify a different type`

**해결 방법**:
- 구체적인 타입 정의
- `unknown` 타입 사용 후 타입 가드 사용
- 제네릭 타입 사용

#### 3. Object Injection 경고

**문제**: `Generic Object Injection Sink`

**해결 방법**:
- 사용자 입력 검증 및 sanitization
- 허용된 키 목록 사용
- Map 자료구조 사용 고려

#### 4. 접근성 오류

**문제**: `click-events-have-key-events`, `no-static-element-interactions`

**해결 방법**:
- 키보드 이벤트 핸들러 추가
- `role` 속성 추가
- `tabIndex` 속성 추가
- 적절한 HTML 요소 사용 (button, a 등)

#### 5. React Hooks 규칙 위반

**문제**: `React Hook is called conditionally`

**해결 방법**:
- 모든 Hook을 컴포넌트 최상단에서 호출
- 조건부 렌더링은 Hook 호출 후에 수행

### 자동 수정

대부분의 스타일 관련 문제는 자동으로 수정할 수 있습니다:

```bash
npm run lint:fix
```

### 특정 규칙 비활성화

필요한 경우 특정 라인에서 규칙을 비활성화할 수 있습니다:

```typescript
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const data: any = getData();

// 또는 특정 규칙만 비활성화
// eslint-disable-next-line security/detect-object-injection
obj[key] = value;
```

## 검사 결과 예시

### 현재 검사 결과

```
✖ 162 problems (104 errors, 58 warnings)
```

#### 주요 오류 유형

1. **사용하지 않는 변수/함수**: 30+ 건
2. **any 타입 사용**: 20+ 건
3. **접근성 문제**: 15+ 건
4. **React Hooks 규칙**: 5+ 건
5. **보안 경고**: 58+ 건

#### 우선순위별 수정 가이드

1. **에러 (Error)**: 즉시 수정 필요
   - 사용하지 않는 변수 제거
   - any 타입 구체화
   - React Hooks 규칙 준수

2. **경고 (Warning)**: 점진적 수정
   - Object Injection 검토
   - 접근성 개선
   - 정규식 안전성 검토

## 모범 사례

### 1. 타입 안전성

```typescript
// ✅ 좋은 예: 명시적 타입 정의
interface User {
  id: string;
  name: string;
  email: string;
}

function getUser(id: string): Promise<User> {
  return api.get(`/users/${id}`);
}
```

### 2. 입력 검증

```typescript
// ✅ 좋은 예: 입력 검증
function sanitizeInput(input: string): string {
  return input.trim().replace(/[<>]/g, '');
}

function processUserInput(input: string) {
  const sanitized = sanitizeInput(input);
  // 처리 로직
}
```

### 3. 접근성 고려

```typescript
// ✅ 좋은 예: 접근성 고려
<button
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick();
    }
  }}
  aria-label="Submit form"
>
  Submit
</button>
```

### 4. 보안 고려사항

```typescript
// ✅ 좋은 예: 안전한 정규식
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const userInput = getUserInput();
const safePattern = escapeRegex(userInput);
const regex = new RegExp(safePattern);
```

## 참고 자료

- [ESLint 공식 문서](https://eslint.org/)
- [TypeScript 공식 문서](https://www.typescriptlang.org/)
- [eslint-plugin-security](https://github.com/nodesecurity/eslint-plugin-security)
- [eslint-plugin-jsx-a11y](https://github.com/jsx-eslint/eslint-plugin-jsx-a11y)
- [React 보안 가이드](https://reactjs.org/docs/security.html)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)


