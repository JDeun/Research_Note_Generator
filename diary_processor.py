# diary_processor.py
import os
import re
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from base_processor import BaseProcessor

class DiaryProcessor(BaseProcessor):
    """일기 파일 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "claude", auto_optimize: bool = True):
        """
        DiaryProcessor 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
        """
        super().__init__(model_name, auto_optimize)
        
        # 일기 처리기 초기화
        self.diary_handler = DiaryHandler()
        
        # 모든 일기 항목 저장소
        self.all_diaries = {}
    
    def _setup_model(self) -> Any:
        """
        모델 설정 (BaseProcessor 추상 메서드 구현)
        
        Returns:
            Any: 초기화된 모델 인스턴스 (일기 처리에는 LLM이 필요 없음)
        """
        # 일기 처리에는 LLM 모델이 필요 없으므로 더미 객체 반환
        return object()
    
    def _process_file_internal(self, file_path: str) -> Dict[str, Any]:
        """
        일기 파일 처리 내부 로직 (BaseProcessor 추상 메서드 구현)
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        return self.process_diary(file_path)
    
    def process_diary(self, file_path: str) -> Dict[str, Any]:
        """
        일기 파일 처리
        
        Args:
            file_path (str): 일기 파일 경로 (YYMMDD_summary.txt 형식)
            
        Returns:
            Dict[str, Any]: 처리 결과
            {
                "날짜1": "내용1",
                "날짜2": "내용2",
                ...
            }
            또는 에러 발생 시:
            {
                "error": "에러 메시지"
            }
        """
        try:
            # 일기 항목 처리
            diary_entry = self.diary_handler.process_file(file_path)
            
            # 오류 확인
            if "error" in diary_entry:
                return diary_entry
            
            # 전체 일기 항목에 추가
            self.all_diaries.update(diary_entry)
            
            return diary_entry
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "diary_processing"}
            )
            self.logger.error(f"일기 파일 처리 중 오류 발생: {error_detail['error']}")
            return {"error": str(e)}
    
    def get_all_diaries(self) -> Dict[str, str]:
        """
        모든 일기 데이터 반환
        
        Returns:
            Dict[str, str]: 날짜를 키로, 내용을 값으로 하는 딕셔너리
        """
        return self.all_diaries


class DiaryHandler:
    """일기 파일 처리를 위한 핸들러 클래스"""
    
    def __init__(self):
        """DiaryHandler 초기화"""
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("diary_handler")
        self.error_handler = ErrorHandler.get_instance()
        
        # 일기 파일 패턴
        self.file_pattern = r'(\d{6})_summary\.txt'
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        일기 파일 처리
        
        Args:
            file_path (str): 일기 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과 또는 에러 정보
        """
        try:
            # 파일 존재 확인
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"일기 파일을 찾을 수 없습니다: {file_path}")
            
            # 파일명에서 날짜 추출
            date_match = self._extract_date_from_filename(path.name)
            if not date_match:
                return {"error": f"파일명 형식 불일치: {path.name} (YYMMDD_summary.txt 형식이어야 함)"}
            
            # 날짜 추출
            date_str = date_match.group(1)
            
            # 파일 내용 읽기
            content = self._read_file_content(file_path)
            if not content:
                return {"error": f"파일 내용이 비어있습니다: {file_path}"}
            
            # 결과 반환 (날짜를 키로, 내용을 값으로)
            return {date_str: content}
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "diary_file_processing"}
            )
            self.logger.error(f"일기 파일 처리 실패: {error_detail['error']}")
            return {"error": str(e)}
    
    def _extract_date_from_filename(self, filename: str) -> Optional[re.Match]:
        """
        파일명에서 날짜 추출
        
        Args:
            filename (str): 파일명
            
        Returns:
            Optional[re.Match]: 매치 객체 또는 None
        """
        return re.match(self.file_pattern, filename)
    
    def _read_file_content(self, file_path: str) -> str:
        """
        파일 내용 읽기
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            str: 파일 내용
        """
        try:
            # 여러 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read().strip()
                except UnicodeDecodeError:
                    continue
            
            # 모든 인코딩 시도 실패
            raise ValueError(f"지원되지 않는 파일 인코딩: {file_path}")
            
        except Exception as e:
            self.logger.error(f"파일 읽기 실패: {str(e)}")
            raise IOError(f"파일을 읽을 수 없습니다: {str(e)}")
    
    def format_date(self, date_str: str) -> str:
        """
        날짜 형식 변환 (YYMMDD -> YYYY-MM-DD)
        
        Args:
            date_str (str): 6자리 날짜 문자열
            
        Returns:
            str: 변환된 날짜 문자열
        """
        try:
            if len(date_str) != 6 or not date_str.isdigit():
                return date_str
                
            yy = int(date_str[:2])
            mm = int(date_str[2:4])
            dd = int(date_str[4:6])
            
            # 2000년대로 가정
            yyyy = 2000 + yy
            
            # 유효한 날짜인지 확인
            try:
                datetime(yyyy, mm, dd)
            except ValueError:
                return date_str
                
            return f"{yyyy}-{mm:02d}-{dd:02d}"
            
        except Exception as e:
            self.logger.warning(f"날짜 형식 변환 실패: {str(e)}")
            return date_str
    
    def is_diary_file(self, file_path: str) -> bool:
        """
        파일이 일기 파일인지 확인
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            bool: 일기 파일 여부
        """
        filename = Path(file_path).name
        return bool(self._extract_date_from_filename(filename))