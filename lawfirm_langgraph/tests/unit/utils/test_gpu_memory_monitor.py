# -*- coding: utf-8 -*-
"""
GPU Memory Monitor 테스트
core/utils/gpu_memory_monitor.py 단위 테스트
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from lawfirm_langgraph.core.utils.gpu_memory_monitor import GPUMemoryMonitor


class TestGPUMemoryMonitor:
    """GPUMemoryMonitor 테스트"""
    
    def test_init(self, tmp_path):
        """초기화 테스트"""
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        assert monitor.log_file == log_file
        assert monitor.monitoring is False
        assert monitor.memory_history == []
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.os.makedirs')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.logging.basicConfig')
    def test_setup_logging(self, mock_basic_config, mock_makedirs, tmp_path):
        """로깅 설정 테스트"""
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        mock_makedirs.assert_called_once()
        mock_basic_config.assert_called_once()
        assert monitor.logger is not None
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    def test_get_system_memory_info(self, mock_virtual_memory, tmp_path):
        """시스템 메모리 정보 수집 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        info = monitor.get_system_memory_info()
        
        assert info["total_gb"] == 16.0
        assert info["available_gb"] == 8.0
        assert info["used_gb"] == 8.0
        assert info["percentage"] == 50.0
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.get_device_properties')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.memory_allocated')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.memory_reserved')
    def test_get_gpu_memory_info_with_cuda(self, mock_reserved, mock_allocated, mock_props, mock_is_available, tmp_path):
        """GPU 메모리 정보 수집 테스트 (CUDA 사용 가능)"""
        mock_is_available.return_value = True
        mock_props.return_value.total_memory = 8 * 1024**3
        mock_allocated.return_value = 4 * 1024**3
        mock_reserved.return_value = 5 * 1024**3
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        info = monitor.get_gpu_memory_info()
        
        assert info is not None
        assert info["total_gb"] == 8.0
        assert info["allocated_gb"] == 4.0
        assert info["cached_gb"] == 5.0
        assert info["free_gb"] == 4.0
        assert info["utilization_percent"] == 50.0
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_get_gpu_memory_info_without_cuda(self, mock_is_available, tmp_path):
        """GPU 메모리 정보 수집 테스트 (CUDA 사용 불가)"""
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        info = monitor.get_gpu_memory_info()
        
        assert info is None
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_get_memory_status(self, mock_is_available, mock_virtual_memory, tmp_path):
        """전체 메모리 상태 수집 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        status = monitor.get_memory_status()
        
        assert "timestamp" in status
        assert "system_memory" in status
        assert "gpu_memory" in status
        assert len(monitor.memory_history) == 1
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_memory_history_limit(self, mock_is_available, mock_virtual_memory, tmp_path):
        """메모리 히스토리 제한 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        for _ in range(150):
            monitor.get_memory_status()
        
        assert len(monitor.memory_history) == 100
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_log_memory_status(self, mock_is_available, mock_virtual_memory, tmp_path):
        """메모리 상태 로깅 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        with patch.object(monitor.logger, 'info') as mock_info:
            monitor.log_memory_status()
            assert mock_info.called
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.time.sleep')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_stop_monitoring(self, mock_is_available, mock_virtual_memory, mock_sleep, tmp_path):
        """모니터링 중지 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        monitor.monitoring = True
        
        with patch.object(monitor.logger, 'info') as mock_info:
            monitor.stop_monitoring()
            assert monitor.monitoring is False
            assert mock_info.called
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_get_memory_summary_empty(self, mock_is_available, mock_virtual_memory, tmp_path):
        """메모리 요약 테스트 (데이터 없음)"""
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        summary = monitor.get_memory_summary()
        
        assert "error" in summary
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_get_memory_summary(self, mock_is_available, mock_virtual_memory, tmp_path):
        """메모리 요약 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        for _ in range(5):
            monitor.get_memory_status()
        
        summary = monitor.get_memory_summary()
        
        assert "monitoring_duration" in summary
        assert "latest_status" in summary
        assert "average_system_usage" in summary
        assert "average_gpu_usage" in summary
        assert "peak_system_usage" in summary
        assert "peak_gpu_usage" in summary
    
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.psutil.virtual_memory')
    @patch('lawfirm_langgraph.core.utils.gpu_memory_monitor.torch.cuda.is_available')
    def test_save_memory_report(self, mock_is_available, mock_virtual_memory, tmp_path):
        """메모리 보고서 저장 테스트"""
        mock_memory = Mock()
        mock_memory.total = 16 * 1024**3
        mock_memory.available = 8 * 1024**3
        mock_memory.used = 8 * 1024**3
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_is_available.return_value = False
        
        log_file = str(tmp_path / "gpu_memory.log")
        report_file = str(tmp_path / "memory_report.json")
        monitor = GPUMemoryMonitor(log_file=log_file)
        
        monitor.get_memory_status()
        
        with patch.object(monitor.logger, 'info') as mock_info:
            monitor.save_memory_report(report_file)
            assert mock_info.called
        
        assert os.path.exists(report_file)

