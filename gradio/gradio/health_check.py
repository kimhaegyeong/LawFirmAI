# -*- coding: utf-8 -*-
"""
LawFirmAI - 헬스체크 엔드포인트
프로덕션 환경을 위한 상태 확인 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil
import time
from datetime import datetime

app = FastAPI(title="LawFirmAI Health Check")

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        # 시스템 상태 확인
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 메모리 사용량이 90% 이상이면 경고
        if memory.percent > 90:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # 디스크 사용량이 90% 이상이면 경고
        if disk.percent > 90:
            raise HTTPException(status_code=503, detail="High disk usage")
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "service": "LawFirmAI Production"
        })
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/ready")
async def readiness_check():
    """레디니스 체크 엔드포인트"""
    try:
        # 애플리케이션 준비 상태 확인
        # 여기에 실제 애플리케이션 상태 확인 로직 추가
        
        return JSONResponse({
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "service": "LawFirmAI Production"
        })
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
