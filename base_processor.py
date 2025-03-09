# base_processor.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import time
import random
from datetime import datetime
from pathlib import Path

from logging_manager import LoggingManager
from error_handler import ErrorHandler

class BaseProcessor(ABC):
    """
    모든 파일 처리기의 기본 추상 클래스
    """
    
    def __init__(self, model_name: str, auto_optimize: bool = True):
        """
        기본 처리기 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
        """
        self.model_name = model_name.lower()
        self.auto_optimize = auto_optimize
        
        # 공유 인스턴스
        self.logger = LoggingManager.get_instance().get_logger(self.__class__.__name__)
        self.error_handler = ErrorHandler.get_instance()
        
        # 처리 상태
        self.processor_ready = False
        self.last_call_time = 0
        self.api_call_count = 0
        
        # 모델 초기화
        self.llm = self._initialize_model()
        if self.llm:
            self.processor_ready = True
        
    def _initialize_model(self) -> Any:
        """
        LLM 모델 초기화
        
        Returns:
            Any: 초기화된 LLM 모델 인스턴스 또는 None
        """
        try:
            # 추상 메서드를 통해 구체적인 모델 초기화는 하위 클래스에서 처리
            llm = self._setup_model()
            self.logger.info(f"{self.model_name} 모델 초기화 성공")
            return llm
        except Exception as e:
            self.error_handler.handle_error(
                e, {"model_name": self.model_name, "processor": self.__class__.__name__}
            )
            self.logger.error(f"{self.model_name} 모델 초기화 실패: {str(e)}")
            return None
    
    def process_file(self, file_path: str, category: str = None) -> Dict[str, Any]:
        """
        단일 파일 처리
        
        Args:
            file_path (str): 처리할 파일 경로
            category (str, optional): 파일 카테고리 ('research' 또는 'reference')
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.processor_ready:
            error_msg = f"{self.__class__.__name__} 처리기가 준비되지 않았습니다"
            self.logger.error(error_msg)
            return self._create_error_result(file_path, error_msg)
        
        try:
            # 시작 시간 기록
            start_time = time.time()
            self.logger.info(f"파일 처리 시작: {file_path}")
            
            # 파일 존재 확인
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                error_msg = f"파일이 존재하지 않습니다: {file_path}"
                self.logger.error(error_msg)
                return self._create_error_result(file_path, error_msg)
            
            # 카테고리가 제공되지 않은 경우 자동 감지
            if category is None:
                from file_type_detector import FileTypeDetector
                category = FileTypeDetector.get_instance()._determine_category(file_path)
            
            # 실제 파일 처리 로직 (하위 클래스에서 구현)
            result = self._process_file_internal(file_path)
            
            # 카테고리, 처리 시간 및 타임스탬프 추가
            if result and 'error' not in result:
                result['category'] = category
                result['processed_at'] = datetime.now().isoformat()
                result['processing_time'] = time.time() - start_time
            
            self.logger.info(f"파일 처리 완료: {file_path} ({time.time() - start_time:.2f}초)")
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"파일 처리 중 오류 발생: {error_msg}")
            self.error_handler.handle_error(
                e, {"file_path": file_path, "processor": self.__class__.__name__}
            )
            return self._create_error_result(file_path, error_msg)
    
    def _create_error_result(self, file_path: str, error_msg: str) -> Dict[str, Any]:
        """
        에러 발생 시 표준화된 결과 생성
        
        Args:
            file_path (str): 처리하려던 파일 경로
            error_msg (str): 오류 메시지
            
        Returns:
            Dict[str, Any]: 표준화된 에러 결과
        """
        file_info = {
            'file_path': file_path,
            'file_name': Path(file_path).name
        }
        
        return {
            'file_info': file_info,
            'error': error_msg,
            'processed_at': datetime.now().isoformat()
        }
    
    def _manage_rate_limit(self, operation_name: str = "API 호출"):
        """
        API 호출 속도 관리
        
        Args:
            operation_name (str): 작업 이름 (로깅용)
        """
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        # 호출 간격이 너무 짧으면 대기
        min_interval = 0.5  # 최소 0.5초 간격
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            self.logger.debug(f"{operation_name} 속도 제한: {wait_time:.2f}초 대기")
            time.sleep(wait_time)
        
        # 호출 추적
        self.api_call_count += 1
        self.last_call_time = time.time()
    
    def retry_with_backoff(self, func, *args, max_retries=3, initial_delay=1.0, 
                          backoff_factor=2.0, exceptions_to_retry=(Exception,), 
                          operation_name="작업", **kwargs):
        """
        지수 백오프 방식으로 함수 재시도
        
        Args:
            func: 실행할 함수
            *args: 함수 인자
            max_retries (int): 최대 재시도 횟수
            initial_delay (float): 초기 대기 시간
            backoff_factor (float): 백오프 계수
            exceptions_to_retry (tuple): 재시도할 예외 유형
            operation_name (str): 작업 이름 (로깅용)
            **kwargs: 함수 키워드 인자
            
        Returns:
            Any: 함수 반환값
            
        Raises:
            Exception: 최대 재시도 후에도 실패한 경우
        """
        retry_count = 0
        delay = initial_delay
        
        while True:
            try:
                # API 호출 속도 관리
                self._manage_rate_limit(operation_name)
                return func(*args, **kwargs)
                
            except exceptions_to_retry as e:
                retry_count += 1
                
                if retry_count >= max_retries:
                    self.logger.error(f"{operation_name} 최대 재시도 횟수 초과: {str(e)}")
                    raise
                
                # 지수 백오프 + 지터 적용
                jitter = random.uniform(0, 0.1 * delay)
                wait_time = delay + jitter
                
                # 오류 유형에 따라 대기 시간 조정
                if "rate limit" in str(e).lower() or "429" in str(e):
                    wait_time *= 2  # 속도 제한 오류는 더 오래 대기
                    self.logger.warning(f"API 속도 제한 감지. {wait_time:.1f}초 대기 후 재시도 ({retry_count}/{max_retries})")
                else:
                    self.logger.warning(f"{operation_name} 실패: {str(e)}. {wait_time:.1f}초 대기 후 재시도 ({retry_count}/{max_retries})")
                
                time.sleep(wait_time)
                delay *= backoff_factor
    
    @abstractmethod
    def _setup_model(self) -> Any:
        """
        LLM 모델 설정 (하위 클래스에서 구현)
        
        Returns:
            Any: 초기화된 LLM 모델
        """
        pass
    
    @abstractmethod
    def _process_file_internal(self, file_path: str) -> Dict[str, Any]:
        """
        파일 처리 내부 로직 (하위 클래스에서 구현)
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        pass
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        파일 메타데이터 추출
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        try:
            path = Path(file_path)
            stats = path.stat()
            
            return {
                "file_path": str(file_path),
                "file_name": path.name,
                "file_size": stats.st_size,
                "created_time": datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                "modified_time": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                "extension": path.suffix.lower()
            }
        except Exception as e:
            self.logger.error(f"메타데이터 추출 오류: {str(e)}")
            return {
                "file_path": str(file_path),
                "file_name": Path(file_path).name,
                "error": f"메타데이터 추출 실패: {str(e)}"
            }