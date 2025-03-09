# processor_factory.py
from typing import Dict, Any, Optional, Type, List
import importlib

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from base_processor import BaseProcessor
from file_type_detector import FileTypeDetector

class ProcessorFactory:
    """
    파일 유형에 따른 적절한 프로세서 인스턴스를 생성하는 팩토리 클래스
    """
    # 싱글톤 인스턴스
    _instance = None
    
    # 프로세서 클래스 매핑
    _processor_class_map: Dict[str, Type[BaseProcessor]] = {}
    
    # 생성된 프로세서 인스턴스 캐싱
    _processor_instances: Dict[str, Dict[str, BaseProcessor]] = {}
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = ProcessorFactory()
        return cls._instance
    
    def __init__(self):
        """초기화"""
        self.logger = LoggingManager.get_instance().get_logger("processor_factory")
        self.error_handler = ErrorHandler.get_instance()
        
        # 프로세서 클래스 매핑 초기화
        self._initialize_processor_classes()
    
    def _initialize_processor_classes(self):
        """프로세서 클래스 매핑 초기화"""
        # 기본 프로세서 매핑 설정
        # 실제 구현 시에는 동적으로 로드하거나 설정 파일에서 가져올 수 있음
        try:
            # 이미지 프로세서
            from image_processor import ImageProcessor
            self._processor_class_map["image"] = ImageProcessor
            
            # 코드 프로세서
            from code_processor import CodeProcessor
            self._processor_class_map["code"] = CodeProcessor
            
            # 문서 프로세서
            from document_processor import DocumentProcessor
            self._processor_class_map["document"] = DocumentProcessor
            
            # 일기 프로세서
            from diary_processor import DiaryProcessor
            self._processor_class_map["diary"] = DiaryProcessor
            
            self.logger.info(f"프로세서 클래스 매핑 초기화 완료: {len(self._processor_class_map)}개")
            
        except ImportError as e:
            self.logger.error(f"프로세서 클래스 로드 실패: {str(e)}")
    
    def register_processor(self, file_type: str, processor_class: Type[BaseProcessor]):
        """
        새 프로세서 클래스 등록
        
        Args:
            file_type (str): 파일 유형
            processor_class (Type[BaseProcessor]): 프로세서 클래스
        """
        if not issubclass(processor_class, BaseProcessor):
            self.logger.error(f"등록 실패: {processor_class.__name__}은(는) BaseProcessor의 하위 클래스가 아닙니다.")
            return
            
        self._processor_class_map[file_type] = processor_class
        
        # 기존에 캐싱된 인스턴스가 있으면 제거 (새 클래스 적용을 위해)
        if file_type in self._processor_instances:
            self._processor_instances[file_type] = {}
            
        self.logger.info(f"프로세서 등록 완료: {file_type} -> {processor_class.__name__}")
    
    def get_processor(self, file_type: str, model_name: str = "claude", 
                      auto_optimize: bool = True) -> Optional[BaseProcessor]:
        """
        파일 유형에 맞는 프로세서 인스턴스 가져오기
        
        Args:
            file_type (str): 파일 유형
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
            
        Returns:
            Optional[BaseProcessor]: 프로세서 인스턴스 또는 None
        """
        # 파일 유형 표준화
        file_type = file_type.lower()
        model_name = model_name.lower()
        
        # 프로세서 클래스 확인
        if file_type not in self._processor_class_map:
            self.logger.error(f"지원되지 않는 파일 유형: {file_type}")
            return None
        
        # 캐시에서 인스턴스 검색
        processor_key = f"{model_name}_{auto_optimize}"
        if file_type in self._processor_instances and processor_key in self._processor_instances[file_type]:
            return self._processor_instances[file_type][processor_key]
        
        # 새 인스턴스 생성
        try:
            processor_class = self._processor_class_map[file_type]
            processor = processor_class(model_name=model_name, auto_optimize=auto_optimize)
            
            # 캐시에 저장
            if file_type not in self._processor_instances:
                self._processor_instances[file_type] = {}
            self._processor_instances[file_type][processor_key] = processor
            
            self.logger.info(f"{file_type} 프로세서 인스턴스 생성 완료: {model_name} 모델")
            return processor
            
        except Exception as e:
            self.error_handler.handle_error(
                e, {
                    "file_type": file_type, 
                    "model_name": model_name,
                    "operation": "processor_creation"
                }
            )
            self.logger.error(f"프로세서 인스턴스 생성 실패: {str(e)}")
            return None
    
    def get_processor_for_file(self, file_path: str, model_name: str = "claude",
                              auto_optimize: bool = True,
                              diary_pattern: str = None) -> Optional[BaseProcessor]:
        """
        파일 경로에 맞는 프로세서 인스턴스 가져오기
        
        Args:
            file_path (str): 파일 경로
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
            diary_pattern (str, optional): 일기 파일 판별을 위한 정규식 패턴
            
        Returns:
            Optional[BaseProcessor]: 프로세서 인스턴스 또는 None
        """
        try:
            # 파일 유형 감지
            detector = FileTypeDetector.get_instance()
            file_info = detector.detect_file_type(file_path, diary_pattern)
            file_type = file_info['type']
            
            # 유형에 맞는 프로세서 가져오기
            return self.get_processor(file_type, model_name, auto_optimize)
            
        except Exception as e:
            self.error_handler.handle_error(
                e, {
                    "file_path": file_path, 
                    "model_name": model_name,
                    "operation": "processor_selection"
                }
            )
            self.logger.error(f"파일에 맞는 프로세서 가져오기 실패: {str(e)}")
            return None
    
    def get_all_processors(self, model_name: str = "claude",
                         auto_optimize: bool = True) -> Dict[str, BaseProcessor]:
        """
        모든 유형의 프로세서 인스턴스 가져오기
        
        Args:
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
            
        Returns:
            Dict[str, BaseProcessor]: 파일 유형별 프로세서 인스턴스
        """
        processors = {}
        
        for file_type in self._processor_class_map.keys():
            processor = self.get_processor(file_type, model_name, auto_optimize)
            if processor:
                processors[file_type] = processor
        
        return processors
    
    def clear_cache(self, file_type: str = None):
        """
        프로세서 인스턴스 캐시 정리
        
        Args:
            file_type (str, optional): 특정 파일 유형의 캐시만 정리. None이면 모든 캐시 정리
        """
        if file_type is None:
            # 모든 캐시 정리
            self._processor_instances = {}
            self.logger.info("모든 프로세서 인스턴스 캐시가 정리되었습니다.")
        elif file_type in self._processor_instances:
            # 특정 유형만 정리
            self._processor_instances[file_type] = {}
            self.logger.info(f"{file_type} 프로세서 인스턴스 캐시가 정리되었습니다.")