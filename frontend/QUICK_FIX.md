# 빠른 문제 해결

## npm run dev 실행 방법

### 1. 현재 실행 중인 프로세스 종료
터미널에서 `Ctrl + C`로 실행 중인 프로세스를 종료하세요.

### 2. 서버 재시작
```bash
cd frontend
npm run dev
```

### 3. 확인 사항

#### 서버가 시작되는지 확인
터미널에 다음과 같은 메시지가 표시되어야 합니다:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

#### 브라우저 자동 열림
`open: true` 설정으로 브라우저가 자동으로 열립니다.

#### 수동 접속
브라우저가 자동으로 열리지 않으면 다음 URL로 접속:
- http://localhost:3000
- http://127.0.0.1:3000

### 4. 여전히 문제가 있는 경우

#### 포트 3000이 사용 중인 경우
다른 포트 사용:
```bash
# vite.config.ts에서 port: 3000을 다른 포트로 변경
# 또는
npm run dev -- --port 3001
```

#### 파일 권한 문제 (Windows)
관리자 권한으로 PowerShell/CMD 실행 후:
```bash
cd frontend
npm run dev
```

#### 캐시 삭제 후 재시작
```bash
cd frontend
rm -rf node_modules/.vite  # Windows: rmdir /s node_modules\.vite
npm run dev
```

### 5. 백엔드 서버 확인
React 프론트엔드는 백엔드 API(`http://localhost:8000`)에 연결합니다.
백엔드 서버가 실행 중인지 확인하세요:
```bash
# 새 터미널에서
cd api
python main.py
```

