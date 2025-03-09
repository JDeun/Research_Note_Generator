# file_type_detector.py
import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# python-magic 라이브러리 조건부 임포트
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic 라이브러리를 찾을 수 없습니다. 확장자 기반 파일 감지만 사용됩니다.")

from logging_manager import LoggingManager
from error_handler import ErrorHandler
class FileTypeDetector:
    """
    파일 유형을 감지하는 클래스
    """
    # 싱글톤 인스턴스
    _instance = None
    
    # 로거
    _logger = None
    
    # 확장자별 파일 유형 매핑
    _extension_type_map: Dict[str, str] = {}
    
    # 파일 유형별 하위 유형
    _file_subtypes: Dict[str, List[str]] = {
        "image": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"],
        "code": ["py", "js", "java", "cpp", "c", "cs", "php", "go", "ts", "rb", "swift", "kt", "sql", "html", "css", "ipynb"],
        "document": ["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md", "rtf", "csv", "json", "xml"],
        "diary": []  # 일기 파일은 파일명 패턴으로 구분
    }
    
    # MIME 타입별 파일 유형
    _mime_type_map: Dict[str, str] = {
        "image/": "image",
        "text/": "document",
        "application/pdf": "document",
        "application/msword": "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
        "application/vnd.ms-excel": "document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "document",
        "application/vnd.ms-powerpoint": "document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "document",
        "text/plain": "document",
        "text/markdown": "document",
        "application/json": "document",
        "text/csv": "document",
        "application/xml": "document",
        "text/xml": "document",
        "application/x-python-code": "code",
        "text/x-python": "code",
        "application/javascript": "code",
        "text/javascript": "code",
        "text/x-java-source": "code",
        "text/x-c++": "code",
        "text/x-c": "code",
        "text/html": "code",
        "text/css": "code",
        "application/x-ipynb+json": "code"
    }
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = FileTypeDetector()
        return cls._instance
    
    def __init__(self):
        """초기화"""
        self._logger = LoggingManager.get_instance().get_logger("file_type_detector")
        self._initialize_extension_map()
        
        # python-magic 라이브러리 사용 가능 여부 확인
        self._magic_available = self._check_magic_availability()
        if not self._magic_available:
            self._logger.warning("python-magic 라이브러리를 사용할 수 없습니다. 확장자 기반 탐지만 사용됩니다.")
    
    def _check_magic_availability(self) -> bool:
        """python-magic 라이브러리 사용 가능 여부 확인"""
        return MAGIC_AVAILABLE
    
    def _initialize_extension_map(self):
        """확장자별 파일 유형 매핑 초기화"""
        # 각 유형별 확장자 매핑
        for file_type, extensions in self._file_subtypes.items():
            for ext in extensions:
                self._extension_type_map[f".{ext}"] = file_type
        
        self._logger.debug(f"초기화된 파일 확장자 매핑: {len(self._extension_type_map)}개")
    
    def add_extension_mapping(self, extension: str, file_type: str):
        """
        확장자 매핑 추가
        
        Args:
            extension (str): 파일 확장자 (점 포함 또는 미포함)
            file_type (str): 파일 유형 ("image", "code", "document", "diary" 중 하나)
        """
        if not extension.startswith("."):
            extension = f".{extension}"
            
        self._extension_type_map[extension] = file_type
        
        # 하위 유형 목록에도 추가
        ext_without_dot = extension[1:]
        if ext_without_dot not in self._file_subtypes.get(file_type, []):
            if file_type in self._file_subtypes:
                self._file_subtypes[file_type].append(ext_without_dot)
    
    def detect_file_type(self, file_path: str, diary_pattern: str = None) -> Dict[str, str]:
        """
        파일 유형 및 카테고리 감지
        
        Args:
            file_path (str): 파일 경로
            diary_pattern (str, optional): 일기 파일 판별을 위한 정규식 패턴
            
        Returns:
            Dict[str, str]: 파일 유형 정보
            {
                'type': str,       # 파일 유형 ('image', 'code', 'document', 'diary')
                'subtype': str,    # 파일 하위 유형 (확장자, 포맷 등)
                'category': str,   # 파일 카테고리 ('research' 또는 'reference')
                'mime_type': str   # MIME 타입 (가능한 경우)
            }
        """
        path = Path(file_path)
        file_name = path.name
        extension = path.suffix.lower()
        
        # 기본값 설정
        result = {
            'type': 'unknown',
            'subtype': extension[1:] if extension else '',
            'category': self._determine_category(file_path),
            'mime_type': ''
        }
        
        # 1. 일기 파일 패턴 확인 (우선순위 높음)
        if diary_pattern and re.match(diary_pattern, file_name):
            result['type'] = 'diary'
            return result
        
        # 2. 확장자 기반 감지
        if extension in self._extension_type_map:
            result['type'] = self._extension_type_map[extension]
            return result
        
        # 3. MIME 타입 기반 감지 (python-magic 사용)
        if self._magic_available:
            try:
                mime_type = magic.from_file(file_path, mime=True)
                result['mime_type'] = mime_type
                
                # MIME 타입 맵에서 직접 찾기
                if mime_type in self._mime_type_map:
                    result['type'] = self._mime_type_map[mime_type]
                    return result
                
                # MIME 타입 접두사 검사
                for mime_prefix, file_type in self._mime_type_map.items():
                    if mime_prefix.endswith('/'):
                        if mime_type.startswith(mime_prefix):
                            result['type'] = file_type
                            return result
                    elif mime_type == mime_prefix:
                        result['type'] = file_type
                        return result
            except Exception as e:
                self._logger.warning(f"MIME 타입 감지 실패: {str(e)}")
        
        # 4. mimetypes 모듈 사용 시도
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                result['mime_type'] = mime_type
                
                # MIME 타입 접두사 검사
                for mime_prefix, file_type in self._mime_type_map.items():
                    if mime_prefix.endswith('/'):
                        if mime_type.startswith(mime_prefix):
                            result['type'] = file_type
                            return result
                    elif mime_type == mime_prefix:
                        result['type'] = file_type
                        return result
        except Exception as e:
            self._logger.warning(f"mimetypes 모듈 사용 실패: {str(e)}")
        
        # 찾지 못한 경우 확장자를 기준으로 추측
        if extension:
            for file_type, extensions in self._file_subtypes.items():
                if extension[1:] in extensions:
                    result['type'] = file_type
                    return result
        
        return result
    
    def get_programming_language(self, file_path: str) -> Optional[str]:
        """
        코드 파일의 프로그래밍 언어 감지
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            Optional[str]: 감지된 프로그래밍 언어 또는 None
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # 확장자별 프로그래밍 언어 매핑
        language_map = {
            # 프로그래밍 언어
            '.py': 'Python',
            '.js': 'JavaScript',
            '.jsx': 'React/JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'React/TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.r': 'R',
            '.scala': 'Scala',
            '.m': 'Objective-C/MATLAB',
            '.lua': 'Lua',
            '.pl': 'Perl',
            '.sh': 'Shell Script',
            '.bash': 'Bash Script',
            '.ps1': 'PowerShell',
            '.vb': 'Visual Basic',
            '.f90': 'Fortran',
            '.dart': 'Dart',
            '.elm': 'Elm',
            
            # 웹 관련
            '.html': 'HTML',
            '.htm': 'HTML',
            '.xhtml': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.less': 'Less',
            '.vue': 'Vue.js',
            '.svelte': 'Svelte',
            
            # 마크업/마크다운
            '.xml': 'XML',
            '.md': 'Markdown',
            '.rst': 'reStructuredText',
            '.tex': 'LaTeX',
            
            # 데이터/설정 파일
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Configuration',
            '.conf': 'Configuration',
            
            # 데이터베이스
            '.sql': 'SQL',
            '.psql': 'PostgreSQL',
            '.plsql': 'PL/SQL',
            
            # 기타
            '.proto': 'Protocol Buffers',
            '.graphql': 'GraphQL',
            '.cmake': 'CMake',
            '.gradle': 'Gradle',
            '.dockerfile': 'Dockerfile',
            '.tf': 'Terraform',
            '.sol': 'Solidity',
            '.ipynb': 'Jupyter Notebook'
        }
        
        if extension in language_map:
            return language_map[extension]
        
        # Jupyter Notebook 파일 내용 확인
        if extension == '.ipynb':
            return 'Jupyter Notebook'
        
        # 파일 내용 기반 검사 (첫 몇 줄만)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = ''.join([f.readline() for _ in range(10)])
                
                # 셔뱅으로 언어 감지
                if first_lines.startswith('#!/usr/bin/env python'):
                    return 'Python'
                elif first_lines.startswith('#!/usr/bin/env node'):
                    return 'JavaScript'
                elif first_lines.startswith('#!/bin/bash'):
                    return 'Bash Script'
                
                # 특정 패턴으로 언어 추측
                if 'import' in first_lines and ('from' in first_lines and 'import' in first_lines.split('from')[1]):
                    return 'Python'
                elif 'func main()' in first_lines and '{' in first_lines:
                    return 'Go'
                elif 'public static void main' in first_lines:
                    return 'Java'
                elif 'using namespace std' in first_lines:
                    return 'C++'
                elif 'package ' in first_lines and first_lines.strip().startswith('package '):
                    return 'Java or Kotlin'
                elif 'function ' in first_lines and '{' in first_lines:
                    return 'JavaScript'
                elif '<html' in first_lines.lower():
                    return 'HTML'
                elif '@media' in first_lines or '{' in first_lines and ':' in first_lines:
                    return 'CSS'
        except Exception as e:
            self._logger.debug(f"파일 내용 기반 언어 감지 실패: {str(e)}")
            
        return "Unknown"
    
    def _determine_category(self, file_path: str) -> str:
        """
        파일 경로를 기반으로 카테고리(research/reference) 결정
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            str: 'research' 또는 'reference'
        """
        try:
            # Path 객체 사용
            path = Path(file_path)
            
            # 부모 디렉토리 검사
            for parent in path.parents:
                if parent.name.lower() == 'reference':
                    return 'reference'
            
            # 경로에 'reference' 디렉토리가 포함되어 있는지 확인 (대소문자 무관)
            # 이는 Path.parents가 때로는 전체 경로를 완전히 포착하지 못할 때 백업 확인
            if 'reference' in str(path).lower().split(os.path.sep):
                return 'reference'
        
        except Exception as e:
            self._logger.error(f"카테고리 결정 중 오류 발생: {str(e)}")
        
        # 기본값으로 'research' 반환
        return 'research'