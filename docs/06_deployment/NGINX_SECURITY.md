# Nginx 보안 설정 가이드

## 개요

이 가이드는 LawFirmAI의 Nginx 설정에서 보안을 강화하는 방법을 설명합니다.

## 보안 개선 사항

### 1. Security Headers

**추가된 보안 헤더:**
- `X-Frame-Options: DENY` - 클릭재킹 방지
- `X-Content-Type-Options: nosniff` - MIME 타입 스니핑 방지
- `X-XSS-Protection: 1; mode=block` - XSS 공격 방지
- `Referrer-Policy: strict-origin-when-cross-origin` - 리퍼러 정보 제한
- `Permissions-Policy` - 브라우저 기능 제한
- `Strict-Transport-Security` - HTTPS 강제 (SSL 설정 시)

### 2. Rate Limiting

**설정:**
- API 엔드포인트: 분당 100 요청 (burst 20)
- 일반 요청: 분당 200 요청 (burst 50)
- 연결 제한: IP당 10개 동시 연결

**Rate limiting zone 정의:**
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=general_limit:10m rate=200r/m;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
```

### 3. 요청 크기 제한

```nginx
client_max_body_size 10M;
```

- 업로드 파일 크기 제한: 10MB
- DoS 공격 방지

### 4. 타임아웃 설정

```nginx
client_body_timeout 10s;
client_header_timeout 10s;
keepalive_timeout 65s;
send_timeout 10s;
```

- 리소스 고갈 공격 방지
- 연결 타임아웃 관리

### 5. 버퍼 크기 제한

```nginx
client_body_buffer_size 128k;
client_header_buffer_size 1k;
large_client_header_buffers 4 4k;
```

- 메모리 사용량 제한
- 버퍼 오버플로우 공격 방지

### 6. 서버 정보 숨김

```nginx
server_tokens off;
```

- Nginx 버전 정보 숨김
- 정보 수집 공격 방지

### 7. 숨겨진 파일 접근 차단

```nginx
location ~ /\. {
    deny all;
    access_log off;
    log_not_found off;
}
```

- `.htaccess`, `.env` 등 숨겨진 파일 접근 차단
- 민감한 정보 노출 방지

### 8. HTTP 메서드 제한

```nginx
if ($request_method !~ ^(GET|HEAD|POST|PUT|DELETE|OPTIONS|PATCH)$) {
    return 405;
}
```

- 허용되지 않은 HTTP 메서드 차단
- 불필요한 요청 차단

### 9. 프록시 보안 설정

```nginx
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Forwarded-Host $host;
proxy_set_header X-Forwarded-Port $server_port;
```

- 실제 클라이언트 IP 전달
- HTTPS 정보 전달
- 프록시 보안 강화

### 10. 로깅 설정

```nginx
access_log /var/log/nginx/access.log;
error_log /var/log/nginx/error.log warn;
```

- 접근 로그 기록
- 보안 사고 분석에 활용

## 프로덕션 환경 설정

### 1. SSL/TLS 설정

HTTPS를 사용하는 경우:

```nginx
# HTTP에서 HTTPS로 리다이렉트
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS 서버
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL 인증서 설정
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    # SSL 보안 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HSTS 헤더
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    
    # 나머지 설정...
}
```

### 2. Content Security Policy (CSP)

필요에 따라 CSP 헤더 추가:

```nginx
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' http://api:8000;" always;
```

**주의:** CSP는 애플리케이션에 따라 조정이 필요합니다.

### 3. Rate Limiting 조정

트래픽에 따라 Rate limiting 값 조정:

```nginx
# 높은 트래픽 환경
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=500r/m;

# 낮은 트래픽 환경
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=50r/m;
```

## 보안 체크리스트

### 필수 설정
- [ ] Security headers 추가
- [ ] Rate limiting 설정
- [ ] 요청 크기 제한
- [ ] 타임아웃 설정
- [ ] 서버 정보 숨김
- [ ] 숨겨진 파일 접근 차단
- [ ] HTTP 메서드 제한

### 선택적 설정
- [ ] SSL/TLS 설정 (HTTPS)
- [ ] Content Security Policy
- [ ] 접근 로그 분석
- [ ] Fail2ban 설정 (추가 보안)

## 보안 모니터링

### 1. 로그 모니터링

```bash
# 접근 로그 확인
tail -f /var/log/nginx/access.log

# 에러 로그 확인
tail -f /var/log/nginx/error.log

# 의심스러운 요청 확인
grep -i "suspicious" /var/log/nginx/access.log
```

### 2. Rate Limiting 모니터링

```bash
# Rate limit 초과 요청 확인
grep "503" /var/log/nginx/access.log
```

### 3. 보안 헤더 확인

```bash
# 보안 헤더 확인
curl -I https://your-domain.com | grep -i "x-frame\|x-content\|x-xss\|strict-transport"
```

## 문제 해결

### 1. Rate Limiting 오작동

**증상:** 정상 사용자가 차단됨

**해결책:**
- Rate limiting 값 증가
- Whitelist 추가 (필요 시)

### 2. CSP 오작동

**증상:** 페이지가 제대로 로드되지 않음

**해결책:**
- CSP 정책 완화
- 브라우저 콘솔에서 CSP 에러 확인

### 3. 프록시 타임아웃

**증상:** API 요청이 타임아웃됨

**해결책:**
- 프록시 타임아웃 값 증가
- 스트리밍 엔드포인트는 별도 설정

## 크롤링 방지

### 1. robots.txt 제공

`frontend/public/robots.txt` 파일을 통해 크롤러 제어:

```
User-agent: *
Disallow: /api/
Disallow: /admin/
Disallow: /internal/
Disallow: /private/
Disallow: /test/
Disallow: /debug/
```

### 2. User-Agent 기반 차단

**차단되는 크롤러:**
- bot, crawler, spider, scraper
- curl, wget
- python, java, perl, ruby, php, node
- scrapy, requests, urllib, mechanize
- selenium, headless, phantom, casper, ghost, webdriver

**허용되는 검색 엔진:**
- Googlebot
- Bingbot
- Slurp (Yahoo)
- DuckDuckBot
- Baiduspider
- YandexBot
- Sogou
- Exabot
- facebot
- ia_archiver

### 3. API 엔드포인트 크롤링 차단

API 엔드포인트(`/api/*`)는 모든 크롤러 차단:
- 검색 엔진 포함 모든 크롤러 차단
- User-Agent가 없는 요청 차단

### 4. Rate Limiting

크롤러 전용 Rate limiting:
- 크롤러: 분당 10 요청 (burst 5)
- 일반 사용자: 분당 200 요청 (burst 50)

### 5. 관리자 경로 차단

다음 경로는 모든 크롤러 차단:
- `/admin/`
- `/internal/`
- `/private/`
- `/test/`
- `/debug/`

## 크롤링 방지 체크리스트

- [ ] robots.txt 파일 생성 및 배포
- [ ] User-Agent 기반 차단 설정
- [ ] API 엔드포인트 크롤링 차단
- [ ] Rate limiting 설정
- [ ] 관리자 경로 크롤러 차단
- [ ] 크롤러 접근 로그 모니터링

## 참고 자료

- [Nginx 보안 가이드](https://nginx.org/en/docs/http/ngx_http_core_module.html)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Mozilla Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)
- [robots.txt 규칙](https://www.robotstxt.org/)

---

**보안은 지속적인 프로세스입니다. 정기적으로 보안 설정을 검토하고 업데이트하세요.**

