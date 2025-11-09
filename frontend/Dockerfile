# Frontend Dockerfile
FROM node:18-alpine as builder

WORKDIR /app

# 의존성 설치
COPY frontend/package*.json ./
RUN npm ci

# 소스 코드 복사 및 빌드
COPY frontend/ ./
RUN npm run build

# Nginx를 사용한 프로덕션 서버
FROM nginx:alpine

# 빌드된 파일 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# robots.txt 복사 (있는 경우)
COPY frontend/public/robots.txt /usr/share/nginx/html/robots.txt 2>/dev/null || true

# Nginx 설정 복사
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

