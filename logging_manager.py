# logging_manager.py
import logging
from typing import Dict, Optional, Any, Union
import os
import sys
import json
from datetime import datetime

class LoggingManager:
    """
    시스템 전체의 로깅을 관리하는 싱글톤 클래스
    """
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = LoggingManager()
        return cls._instance
    
    def setup(self, default_level: int = logging.INFO, log_format: Optional[str] = None, 
              log_file: Optional[str] = None, console_output: bool = True,
              json_format: bool = False):
        """
        전역 로깅 설정 초기화
        
        Args:
            default_level (int): 기본 로깅 레벨
            log_format (str, optional): 로그 포맷 문자열
            log_file (str, optional): 로그 파일 경로. 지정 시 파일 로깅 활성화
            console_output (bool): 콘솔 출력 여부
            json_format (bool): JSON 형식의 로그 출력 여부
        """
        log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(default_level)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 콘솔 출력 핸들러
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 파일 로깅 핸들러
        if log_file:
            # 로그 디렉토리 생성
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        특정 이름의 로거 가져오기
        
        Args:
            name (str): 로거 이름(일반적으로 모듈 또는 클래스 이름)
            
        Returns:
            logging.Logger: 요청한 이름의 로거 인스턴스
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]
    
    def set_level(self, name: str, level: int):
        """
        특정 로거의 로깅 레벨 설정
        
        Args:
            name (str): 로거 이름
            level (int): 설정할 로깅 레벨
        """
        logger = self.get_logger(name)
        logger.setLevel(level)
    
    def log_structured(self, logger_or_name: Union[logging.Logger, str], level: int, 
                       message: str, **kwargs):
        """
        구조화된 로깅 수행
        
        Args:
            logger_or_name: 로거 인스턴스 또는 로거 이름
            level: 로깅 레벨
            message: 로그 메시지
            **kwargs: 추가 구조화 데이터
        """
        if isinstance(logger_or_name, str):
            logger = self.get_logger(logger_or_name)
        else:
            logger = logger_or_name
            
        if not logger.isEnabledFor(level):
            return
            
        log_data = {"message": message, "timestamp": datetime.now().isoformat()}
        log_data.update(kwargs)
        
        # 옵션에 따라 JSON 형식 또는 텍스트 형식으로 로깅
        try:
            if kwargs.get('json_format', False):
                logger.log(level, json.dumps(log_data))
            else:
                # 간단한 구조화 로깅: key=value 형식
                extra_str = " ".join([f"{k}={v}" for k, v in kwargs.items() if k != 'json_format'])
                logger.log(level, f"{message} {extra_str}")
        except Exception as e:
            # 로깅 자체에서 오류가 발생하지 않도록 보호
            fallback_logger = logging.getLogger("logging_manager")
            fallback_logger.error(f"Logging error: {str(e)}")
            fallback_logger.log(level, message)
    
    def create_batch_logger(self, name: str) -> 'BatchLogger':
        """
        배치 로깅을 위한 로거 생성
        
        Args:
            name (str): 로거 이름
            
        Returns:
            BatchLogger: 배치 로깅 인터페이스
        """
        return BatchLogger(self.get_logger(name))
    
    def is_debug_enabled(self) -> bool:
        """디버그 로깅이 활성화되어 있는지 확인"""
        return logging.getLogger().getEffectiveLevel() <= logging.DEBUG


class BatchLogger:
    """
    여러 로그 항목을 모았다가 한 번에 처리하는 로거
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.logs = []
        
    def add(self, level: int, message: str, **kwargs):
        """배치에 로그 항목 추가"""
        self.logs.append((level, message, kwargs))
        
    def flush(self):
        """누적된 모든 로그 항목 처리"""
        for level, message, kwargs in self.logs:
            if self.logger.isEnabledFor(level):
                extra_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])
                self.logger.log(level, f"{message} {extra_str}")
        self.logs.clear()
        
    def __enter__(self):
        """컨텍스트 관리자 진입"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 자동으로 로그 처리"""
        self.flush()