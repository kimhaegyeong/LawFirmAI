# 문제 해결 가이드

## npm run dev 실행 시 문제

### 1. 포트 3000이 이미 사용 중인 경우

**증상**: `Error: listen EADDRINUSE: address already in use :::3000`

**해결 방법**:
```bash
# Windows에서 포트 사용 중인 프로세스 찾기
netstat -ano | findstr :3000

# 프로세스 종료 (PID는 위 명령어에서 확인)
taskkill /PID [PID번호] /F

# 또는 다른 포트 사용
# vite.config.ts에서 port 변경
```

### 2. node_modules가 설치되지 않은 경우

**증상**: `Cannot find module` 오류

**해결 방법**:
```bash
cd frontend
rm -rf node_modules package-lock.json  # Windows: rmdir /s node_modules
npm install
```

### 3. Vite 서버가 시작되지 않는 경우

**증상**: 브라우저에서 localhost:3000에 접속할 수 없음

**해결 방법**:

1. **index.html 위치 확인**
   - `index.html`은 프로젝트 루트(`frontend/`)에 있어야 합니다
   - `public/index.html`에 있으면 프로젝트 루트로 이동하세요

2. **vite.config.ts 확인**
   ```typescript
   server: {
     host: '0.0.0.0',  // 모든 네트워크 인터페이스에서 접속 허용
     port: 3000,
     open: true,       // 브라우저 자동 열기
   }
   ```

3. **터미널 출력 확인**
   - `npm run dev` 실행 후 터미널에 URL이 표시되는지 확인
   - 예: `Local:   http://localhost:3000/`
   - 오류 메시지가 있는지 확인

### 4. TypeScript 컴파일 오류

**증상**: TypeScript 타입 오류로 빌드 실패

**해결 방법**:
```bash
# 타입 체크만 실행
npx tsc --noEmit

# 타입 오류 수정 후 다시 실행
npm run dev
```

### 5. 의존성 버전 충돌

**증상**: 모듈을 찾을 수 없거나 버전 충돌

**해결 방법**:
```bash
# package-lock.json 삭제 후 재설치
rm package-lock.json
npm install

# 또는 특정 패키지 재설치
npm install react@latest react-dom@latest
```

### 6. 브라우저 캐시 문제

**증상**: 변경사항이 반영되지 않음

**해결 방법**:
- 브라우저 하드 리프레시: `Ctrl + Shift + R` (Windows) 또는 `Cmd + Shift + R` (Mac)
- 브라우저 개발자 도구에서 캐시 비활성화

### 7. CORS 오류

**증상**: API 호출 시 CORS 오류

**해결 방법**:
- `vite.config.ts`의 proxy 설정 확인
- 백엔드 서버의 CORS 설정 확인
- 백엔드 서버가 실행 중인지 확인 (`http://localhost:8000`)

### 8. 환경 변수 문제

**증상**: API 연결 실패

**해결 방법**:
1. `.env` 파일이 `frontend/` 디렉토리에 있는지 확인
2. 환경 변수 형식 확인:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```
3. Vite는 `VITE_` 접두사가 필요합니다

## 일반적인 디버깅 단계

1. **터미널 출력 확인**
   ```bash
   npm run dev
   ```
   - 오류 메시지 확인
   - 서버 URL 확인

2. **브라우저 콘솔 확인**
   - F12로 개발자 도구 열기
   - Console 탭에서 오류 확인

3. **네트워크 탭 확인**
   - 개발자 도구의 Network 탭에서 요청 실패 확인

4. **포트 확인**
   ```bash
   # Windows
   netstat -ano | findstr :3000
   
   # Linux/Mac
   lsof -i :3000
   ```

5. **로그 확인**
   - Vite 서버 로그 확인
   - 브라우저 콘솔 로그 확인

## 추가 도움말

문제가 지속되면 다음을 확인하세요:

1. Node.js 버전: `node --version` (v18 이상 권장)
2. npm 버전: `npm --version`
3. 의존성 설치 완료 여부
4. 파일 권한 문제 (특히 Windows)

