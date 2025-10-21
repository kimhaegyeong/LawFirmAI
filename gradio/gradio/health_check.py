# -*- coding: utf-8 -*-
"""
LawFirmAI - �ｺüũ ��������Ʈ
���δ��� ȯ���� ���� ���� Ȯ�� API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import psutil
import time
from datetime import datetime

app = FastAPI(title="LawFirmAI Health Check")

@app.get("/health")
async def health_check():
    """�ｺüũ ��������Ʈ"""
    try:
        # �ý��� ���� Ȯ��
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # �޸� ��뷮�� 90% �̻��̸� ���
        if memory.percent > 90:
            raise HTTPException(status_code=503, detail="High memory usage")
        
        # ��ũ ��뷮�� 90% �̻��̸� ���
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
    """����Ͻ� üũ ��������Ʈ"""
    try:
        # ���ø����̼� �غ� ���� Ȯ��
        # ���⿡ ���� ���ø����̼� ���� Ȯ�� ���� �߰�
        
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
