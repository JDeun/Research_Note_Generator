# error_handler.py
from typing import Dict, Type, Callable, Any, Optional, Union
import traceback
import sys
import json
from logging_manager import LoggingManager

class ErrorHandler:
    """
    예외 처리 및 표준화된 오류 응답 생성을 담당하는 싱글톤 클래스
    """
    _instance = None
    _error_handlers: Dict[Type[Exception], Callable] = {}
    _logger = None
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = ErrorHandler()
        return cls._instance
    
    def __init__(self):
        """초기화"""
        # 로거 설정
        self._logger = LoggingManager.get_instance().get_logger("error_handler")
        
        # 기본 예외 핸들러 등록
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """기본 예외 핸들러 등록"""
        self.register_handler(FileNotFoundError, self._handle_file_not_found)
        self.register_handler(PermissionError, self._handle_permission_error)
        self.register_handler(ValueError, self._handle_value_error)
        self.register_handler(KeyError, self._handle_key_error)
        self.register_handler(ConnectionError, self._handle_connection_error)
        self.register_handler(TimeoutError, self._handle_timeout_error)
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """
        특정 예외 유형에 대한 핸들러 등록
        
        Args:
            exception_type (Type[Exception]): 처리할 예외 유형
            handler (Callable): 예외 처리 함수
        """
        self._error_handlers[exception_type] = handler
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None, 
                     log_level: int = None) -> Dict[str, Any]:
        """
        예외 처리 및 표준화된 응답 생성
        
        Args:
            error (Exception): 처리할 예외
            context (Dict, optional): 예외 발생 컨텍스트 정보
            log_level (int, optional): 로깅 레벨 (기본: Error)
            
        Returns:
            Dict[str, Any]: 표준화된 오류 응답
        """
        error_type = type(error)
        context = context or {}
        if log_level is None:
            import logging
            log_level = logging.ERROR
        
        # 스택 추적 및 오류 정보 수집
        exc_info = sys.exc_info()
        tb_info = ""
        if exc_info[2]:
            tb_info = "".join(traceback.format_tb(exc_info[2]))
        
        # 로깅
        self._logger.log(
            log_level, 
            f"Error occurred: {str(error)}", 
            exc_info=True, 
            extra={**context, "error_type": error_type.__name__, "traceback": tb_info}
        )
        
        # 등록된 핸들러가 있으면 사용
        if error_type in self._error_handlers:
            return self._error_handlers[error_type](error, context)
            
        # 부모 클래스에 대한 핸들러 찾기
        for exc_class, handler in self._error_handlers.items():
            if issubclass(error_type, exc_class):
                return handler(error, context)
        
        # 기본 응답
        return self._create_error_response(error, error_type, context)
    
    def with_error_handling(self, func, *args, default_return=None, context=None, **kwargs):
        """
        함수를 오류 처리 래퍼로 감싸기
        
        Args:
            func: 실행할 함수
            *args: 함수에 전달할 위치 인자
            default_return: 오류 발생 시 반환할 기본값
            context: 오류 컨텍스트 정보
            **kwargs: 함수에 전달할 키워드 인자
            
        Returns:
            함수의 반환값 또는 오류 발생 시 default_return
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context)
            return default_return
    
    def retry_with_backoff(self, func, *args, max_retries=3, initial_delay=1.0, 
                          backoff_factor=2.0, exceptions_to_retry=(Exception,), 
                          context=None, **kwargs):
        """
        지수 백오프 방식으로 함수 재시도
        
        Args:
            func: 실행할 함수
            *args: 함수에 전달할 위치 인자
            max_retries (int): 최대 재시도 횟수
            initial_delay (float): 첫 재시도 전 대기 시간(초)
            backoff_factor (float): 대기 시간 증가 계수
            exceptions_to_retry (tuple): 재시도할 예외 유형들
            context (dict): 오류 컨텍스트 정보
            **kwargs: 함수에 전달할 키워드 인자
            
        Returns:
            함수의 성공적인 반환값
            
        Raises:
            Exception: 최대 재시도 후에도 실패 시 마지막 예외 발생
        """
        import time
        import random
        
        context = context or {}
        retry_count = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except exceptions_to_retry as e:
                retry_count += 1
                context["retry_count"] = retry_count
                
                if retry_count >= max_retries:
                    # 모든 재시도 실패 시 최종 오류 처리
                    context["max_retries_reached"] = True
                    self.handle_error(e, context)
                    raise
                
                # 지수 백오프 + 지터 적용
                jitter = random.uniform(0, 0.1 * delay)
                wait_time = delay + jitter
                
                self._logger.warning(
                    f"Attempt {retry_count}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                
                time.sleep(wait_time)
                delay *= backoff_factor

    def _create_error_response(self, error: Exception, error_type: Type[Exception], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """표준화된 오류 응답 생성"""
        from error_message_repo import ErrorMessageRepo
        
        # 기본 오류 응답
        response = {
            "success": False,
            "error": str(error),
            "error_type": error_type.__name__
        }
        
        # 오류 코드 추가 (있는 경우)
        if hasattr(error, 'status_code'):
            response["status_code"] = error.status_code
        
        # 컨텍스트에서 안전하게 포함할 수 있는 정보 추가
        safe_context = {}
        for key, value in context.items():
            # 민감한 정보(API 키 등)는 제외
            if not self._is_sensitive_key(key):
                safe_context[key] = value
                
        if safe_context:
            response["context"] = safe_context
            
        # 오류 메시지 저장소에서 사용자 친화적 메시지 가져오기 시도
        try:
            repo = ErrorMessageRepo.get_instance()
            friendly_message = repo.get_message_for_exception(error_type)
            if friendly_message:
                response["friendly_message"] = friendly_message
        except Exception:
            # 오류 메시지 저장소 접근 실패 시 무시
            pass
            
        return response
    
    def _is_sensitive_key(self, key: str) -> bool:
        """키가 민감한 정보를 나타내는지 확인"""
        sensitive_patterns = ['key', 'token', 'secret', 'password', 'credential', 'auth']
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in sensitive_patterns)
    
    # 기본 예외 핸들러 구현
    def _handle_file_not_found(self, error: FileNotFoundError, context: Dict[str, Any]) -> Dict[str, Any]:
        """파일 찾을 수 없음 오류 처리"""
        path = str(error).split("'")[1] if "'" in str(error) else str(error)
        return {
            "success": False,
            "error_type": "FILE_NOT_FOUND",
            "error": f"파일을 찾을 수 없습니다: {path}",
            "details": str(error),
            "path": path
        }
    
    def _handle_permission_error(self, error: PermissionError, context: Dict[str, Any]) -> Dict[str, Any]:
        """권한 오류 처리"""
        return {
            "success": False,
            "error_type": "PERMISSION_DENIED",
            "error": "필요한 권한이 없습니다",
            "details": str(error)
        }
    
    def _handle_value_error(self, error: ValueError, context: Dict[str, Any]) -> Dict[str, Any]:
        """값 오류 처리"""
        return {
            "success": False,
            "error_type": "INVALID_VALUE",
            "error": "잘못된 값이 제공되었습니다",
            "details": str(error)
        }
    
    def _handle_key_error(self, error: KeyError, context: Dict[str, Any]) -> Dict[str, Any]:
        """키 오류 처리"""
        key = str(error).strip("'")
        return {
            "success": False,
            "error_type": "MISSING_KEY",
            "error": f"필수 키가 누락되었습니다: {key}",
            "details": str(error),
            "missing_key": key
        }
    
    def _handle_connection_error(self, error: ConnectionError, context: Dict[str, Any]) -> Dict[str, Any]:
        """연결 오류 처리"""
        return {
            "success": False,
            "error_type": "CONNECTION_ERROR",
            "error": "서버 연결에 실패했습니다",
            "details": str(error)
        }
    
    def _handle_timeout_error(self, error: TimeoutError, context: Dict[str, Any]) -> Dict[str, Any]:
        """시간 초과 오류 처리"""
        return {
            "success": False,
            "error_type": "TIMEOUT",
            "error": "작업 시간이 초과되었습니다",
            "details": str(error)
        }