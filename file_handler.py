# file_handler.py
from typing import Dict, List, Optional, Any
import re
import os
from pathlib import Path

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from file_type_detector import FileTypeDetector

class FileHandler:
    """파일 처리 및 분류를 담당하는 클래스"""
    
    def __init__(self, diary_pattern: str = r'\d{6}_summary\.txt'):
        """
        FileHandler 초기화
        
        Args:
            diary_pattern (str, optional): 일기 파일 인식을 위한 정규 표현식 패턴.
                                          기본값은 'YYMMDD_summary.txt' 형식.
        """
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("file_handler")
        self.error_handler = ErrorHandler.get_instance()
        
        # 일기 파일 패턴 - 사용자 정의 가능
        self.diary_pattern = diary_pattern
        
        # 파일 유형 감지기
        self.type_detector = FileTypeDetector.get_instance()

    def get_files(self, root_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        지정된 경로의 모든 파일을 유형별로 분류

        Args:
            root_path (str): 검색할 루트 디렉토리 경로
            
        Returns:
            Dict[str, List[Dict[str, str]]]: 파일 유형별 경로와 카테고리 정보
            {
                'image': [{'path': str, 'category': str}, ...],
                'code': [{'path': str, 'category': str}, ...],
                'document': [{'path': str, 'category': str}, ...],
                'diary': [{'path': str, 'category': str}, ...]
            }
        """
        # 경로 표준화
        root_path = Path(root_path)
        
        # 경로가 없으면 생성
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"입력 디렉토리를 생성했습니다: {root_path}")
            
        # 분류 결과 초기화
        classified_files = {
            'image': [],
            'code': [],
            'document': [],
            'diary': []
        }
        
        try:
            self.logger.info(f"파일 검색 시작: {root_path}")
            file_counts = {'total': 0, 'classified': 0}
            
            # 재귀적으로 모든 파일 탐색
            for file_path in self._find_files(root_path):
                file_counts['total'] += 1
                
                # 숨김 파일 제외
                if file_path.name.startswith('.'):
                    continue
                
                # 파일 분류
                file_info = self._classify_file(file_path)
                if file_info:
                    file_type = file_info['type']
                    classified_files[file_type].append(file_info)
                    file_counts['classified'] += 1
            
            # 분류 결과 로깅
            self.logger.info(f"총 파일 수: {file_counts['total']}, 분류된 파일 수: {file_counts['classified']}")
            for file_type, files in classified_files.items():
                self.logger.info(f"{file_type} 파일 수: {len(files)}")
                
            return classified_files
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"root_path": str(root_path), "operation": "file_classification"}
            )
            self.logger.error(f"파일 분류 중 오류 발생: {error_detail['error']}")
            raise

    def _find_files(self, root_path: Path) -> List[Path]:
        """
        지정된 경로에서 모든 파일 찾기 (재귀적)
        
        Args:
            root_path (Path): 검색할 루트 디렉토리 경로
            
        Returns:
            List[Path]: 찾은 모든 파일의 경로 목록
        """
        files = []
        try:
            # Path.rglob을 사용하여 모든 파일 찾기
            for file_path in root_path.rglob('*'):
                if file_path.is_file():
                    files.append(file_path)
        except Exception as e:
            self.logger.error(f"파일 검색 중 오류 발생: {str(e)}")
            
        return files

    def _classify_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        단일 파일 분류
        
        Args:
            file_path (Path): 분류할 파일 경로
            
        Returns:
            Optional[Dict[str, str]]: 분류된 파일 정보 또는 None
            {
                'path': str,        # 파일 경로
                'category': str,    # 파일 카테고리 (research 또는 reference)
                'type': str         # 파일 유형 (image, code, document, diary)
            }
        """
        try:
            # 파일 유형 감지
            file_info = self.type_detector.detect_file_type(str(file_path), self.diary_pattern)
            file_type = file_info['type']
            
            # 지원되는 유형인지 확인
            if file_type in ['image', 'code', 'document', 'diary']:
                return {
                    'path': str(file_path),
                    'category': file_info['category'],
                    'type': file_type
                }
            else:
                self.logger.debug(f"지원되지 않는 파일 유형: {file_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"파일 분류 중 오류 발생 ({file_path}): {str(e)}")
            return None

    def validate_path(self, path: str) -> bool:
        """
        입력된 경로가 유효한지 검증

        Args:
            path (str): 검증할 경로

        Returns:
            bool: 경로 유효성 여부
        """
        if not path or not isinstance(path, str):
            return False
            
        try:
            path_obj = Path(path)
            return path_obj.exists()
        except Exception:
            return False

    def get_file_info(self, file_path: str) -> Dict[str, str]:
        """
        개별 파일의 정보를 추출

        Args:
            file_path (str): 파일 경로

        Returns:
            Dict[str, str]: 파일 정보
            {
                'name': str,          # 파일명
                'extension': str,     # 확장자
                'type': str,          # 파일 유형 (image/code/document/diary)
                'category': str       # 연구/참고자료 구분
            }
        """
        try:
            path = Path(file_path)
            file_info = self.type_detector.detect_file_type(file_path, self.diary_pattern)
            
            return {
                'name': path.name,
                'extension': path.suffix.lower(),
                'type': file_info['type'],
                'category': file_info['category']
            }
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "get_file_info"}
            )
            self.logger.error(f"파일 정보 추출 중 오류 발생: {error_detail['error']}")
            
            # 기본 정보 반환
            return {
                'name': Path(file_path).name,
                'extension': Path(file_path).suffix.lower(),
                'type': 'unknown',
                'category': 'unknown'
            }

    def is_diary_file(self, file_path: str) -> bool:
        """
        파일이 일기 파일인지 확인
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            bool: 일기 파일 여부
        """
        file_name = Path(file_path).name
        return bool(re.match(self.diary_pattern, file_name))
    
    def organize_files(self, root_path: str, organize_by: str = 'type') -> Dict[str, int]:
        """
        파일 정리 (선택적 기능)
        
        Args:
            root_path (str): 정리할 루트 디렉토리 경로
            organize_by (str): 정리 기준 ('type' 또는 'category')
            
        Returns:
            Dict[str, int]: 정리된 파일 수 통계
        """
        if organize_by not in ['type', 'category']:
            raise ValueError(f"지원되지 않는 정리 기준: {organize_by}")
            
        stats = {'moved': 0, 'skipped': 0, 'error': 0}
        root_path_obj = Path(root_path)
        
        # 파일 분류
        classified_files = self.get_files(root_path)
        
        try:
            for file_type, files in classified_files.items():
                for file_info in files:
                    source_path = Path(file_info['path'])
                    
                    # 정리 기준에 따라 대상 디렉토리 결정
                    if organize_by == 'type':
                        target_dir = root_path_obj / file_type
                    else:  # 'category'
                        target_dir = root_path_obj / file_info['category']
                    
                    # 대상 디렉토리 생성
                    target_dir.mkdir(exist_ok=True)
                    
                    # 대상 파일 경로
                    target_path = target_dir / source_path.name
                    
                    try:
                        # 이미 대상 경로에 있으면 건너뛰기
                        if source_path.parent.samefile(target_dir):
                            stats['skipped'] += 1
                            continue
                            
                        # 파일 이동
                        source_path.rename(target_path)
                        stats['moved'] += 1
                        self.logger.debug(f"파일 이동: {source_path} -> {target_path}")
                        
                    except Exception as e:
                        stats['error'] += 1
                        self.logger.warning(f"파일 이동 중 오류 발생 ({source_path}): {str(e)}")
            
            self.logger.info(f"파일 정리 완료: 이동 {stats['moved']}개, 건너뜀 {stats['skipped']}개, 오류 {stats['error']}개")
            return stats
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"root_path": str(root_path), "organize_by": organize_by}
            )
            self.logger.error(f"파일 정리 중 오류 발생: {error_detail['error']}")
            raise