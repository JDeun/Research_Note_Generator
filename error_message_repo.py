# error_message_repo.py
from typing import Dict, Any, Type, Optional
import json
import os
from logging_manager import LoggingManager

class ErrorMessageRepo:
    """
    오류 메시지를 중앙 관리하는 싱글톤 클래스
    """
    _instance = None
    _messages: Dict[str, str] = {}
    _exception_messages: Dict[str, str] = {}
    _logger = None
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = ErrorMessageRepo()
        return cls._instance
    
    def __init__(self):
        """초기화 및 기본 메시지 로드"""
        self._logger = LoggingManager.get_instance().get_logger("error_message_repo")
        self._initialize_messages()
    
    def _initialize_messages(self):
        """기본 오류 메시지 초기화"""
        # 기본 시스템 오류 메시지
        self._messages.update({
            # 파일 관련 오류
            "file_not_found": "파일을 찾을 수 없습니다: {path}",
            "file_permission_denied": "파일 접근 권한이 없습니다: {path}",
            "file_read_error": "파일 읽기 오류: {path}",
            "file_write_error": "파일 쓰기 오류: {path}",
            
            # API 관련 오류
            "api_error": "API 호출 오류: {status_code} - {message}",
            "api_timeout": "API 응답 시간 초과",
            "api_rate_limit": "API 요청 한도 초과. {retry_after}초 후 다시 시도하세요.",
            "api_auth_error": "API 인증 오류. API 키를 확인하세요.",
            
            # 파싱 관련 오류
            "parsing_error": "{parser} 파서에서 오류 발생: {message}",
            "json_parse_error": "JSON 파싱 오류: {message}",
            "xml_parse_error": "XML 파싱 오류: {message}",
            
            # 모델 관련 오류
            "model_init_error": "{model_name} 모델 초기화 실패: {reason}",
            "model_inference_error": "모델 추론 중 오류 발생: {message}",
            "model_not_found": "요청한 모델을 찾을 수 없습니다: {model_name}",
            
            # 데이터 관련 오류
            "invalid_data": "잘못된 데이터 형식: {message}",
            "missing_required_field": "필수 필드가 누락되었습니다: {field_name}",
            "validation_error": "데이터 유효성 검사 실패: {message}",
            
            # 시스템 관련 오류
            "memory_error": "메모리 부족으로 작업을 완료할 수 없습니다",
            "timeout_error": "작업 시간이 초과되었습니다",
            "concurrent_access_error": "동시 접근으로 인한 충돌이 발생했습니다",
            
            # 네트워크 관련 오류
            "connection_error": "서버 연결에 실패했습니다: {host}",
            "dns_resolution_error": "호스트명을 해석할 수 없습니다: {host}",
            "ssl_error": "보안 연결 오류: {message}",
            
            # 일반 오류
            "unknown_error": "알 수 없는 오류가 발생했습니다",
            "not_implemented": "요청한 기능이 구현되지 않았습니다: {feature}"
        })
        
        # 예외 유형별 메시지 맵핑
        self._exception_messages = {
            "FileNotFoundError": "file_not_found",
            "PermissionError": "file_permission_denied",
            "IOError": "file_read_error",
            "JSONDecodeError": "json_parse_error",
            "ConnectionError": "connection_error",
            "TimeoutError": "timeout_error",
            "ValueError": "invalid_data",
            "KeyError": "missing_required_field",
            "MemoryError": "memory_error",
            "NotImplementedError": "not_implemented"
        }
        
        # 확장: 외부 JSON 파일에서 메시지 로드
        self._load_messages_from_file()
    
    def _load_messages_from_file(self, file_path: str = None):
        """외부 JSON 파일에서 오류 메시지 로드"""
        # 기본 경로 설정
        if file_path is None:
            file_path = os.environ.get("ERROR_MESSAGES_PATH", "error_messages.json")
        
        if not os.path.exists(file_path):
            self._logger.debug(f"오류 메시지 파일을 찾을 수 없습니다: {file_path}")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                
            # 메시지 업데이트
            self._messages.update(messages.get('messages', {}))
            self._exception_messages.update(messages.get('exception_mappings', {}))
            
            self._logger.info(f"외부 오류 메시지 파일에서 {len(messages.get('messages', {}))}개의 메시지를 로드했습니다.")
            
        except Exception as e:
            self._logger.error(f"오류 메시지 파일 로드 중 오류 발생: {str(e)}")
    
    def get_message(self, key: str, **kwargs) -> str:
        """
        키에 해당하는 오류 메시지 템플릿 가져와서 변수 대체
        
        Args:
            key (str): 메시지 키
            **kwargs: 템플릿 변수
            
        Returns:
            str: 포맷팅된 오류 메시지
        """
        template = self._messages.get(key, f"알 수 없는 오류 ({key})")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self._logger.warning(f"메시지 포맷팅 중 누락된 키: {str(e)}")
            return template  # 변수가 누락된 경우 템플릿 그대로 반환
    
    def get_message_for_exception(self, exception_type: Type[Exception]) -> Optional[str]:
        """
        예외 유형에 해당하는 오류 메시지 키 가져오기
        
        Args:
            exception_type: 예외 유형
            
        Returns:
            Optional[str]: 오류 메시지 키 또는 None
        """
        exception_name = exception_type.__name__
        message_key = self._exception_messages.get(exception_name)
        
        if not message_key:
            # 부모 클래스에 대한 메시지 검색
            for exc_name, key in self._exception_messages.items():
                try:
                    exc_class = eval(exc_name)  # 예외 클래스 이름으로 클래스 가져오기
                    if issubclass(exception_type, exc_class):
                        return key
                except (NameError, TypeError):
                    pass
            
            return None
            
        return message_key
    
    def add_message(self, key: str, template: str):
        """
        새 오류 메시지 템플릿 추가
        
        Args:
            key (str): 메시지 키
            template (str): 메시지 템플릿
        """
        self._messages[key] = template
        self._logger.debug(f"오류 메시지 추가: {key}")
    
    def add_exception_mapping(self, exception_name: str, message_key: str):
        """
        예외 유형과 메시지 키 매핑 추가
        
        Args:
            exception_name (str): 예외 클래스 이름
            message_key (str): 메시지 키
        """
        self._exception_messages[exception_name] = message_key
        self._logger.debug(f"예외 매핑 추가: {exception_name} -> {message_key}")
    
    def save_to_file(self, file_path: str = "error_messages.json"):
        """
        현재 메시지와 매핑을 JSON 파일로 저장
        
        Args:
            file_path (str): 저장할 파일 경로
        """
        try:
            data = {
                "messages": self._messages,
                "exception_mappings": self._exception_messages
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self._logger.info(f"오류 메시지를 파일에 저장했습니다: {file_path}")
            return True
            
        except Exception as e:
            self._logger.error(f"오류 메시지 파일 저장 중 오류 발생: {str(e)}")
            return False