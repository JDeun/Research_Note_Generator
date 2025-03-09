# code_processor.py
import re
import json
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
import ast
import sys
import importlib.metadata
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from base_processor import BaseProcessor
from config import CODE_PROCESSOR_MODELS, TEMPERATURES, CODE_PROCESSOR_LANGUAGE, LLM_API_KEY
from ProcessorPrompt import CODE_PROCESSOR_PROMPT

class CodeProcessor(BaseProcessor):
    """코드 파일 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "claude", auto_optimize: bool = True):
        """
        CodeProcessor 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
        """
        super().__init__(model_name, auto_optimize)
        
        # 코드 분석기 초기화
        self.code_analyzer = CodeAnalyzer(model_name)
    
    def _setup_model(self) -> Any:
        """
        모델 설정 (BaseProcessor 추상 메서드 구현)
        
        Returns:
            Any: 초기화된 모델 인스턴스
        """
        if self.model_name not in CODE_PROCESSOR_MODELS:
            self.logger.error(f"지원하지 않는 모델: {self.model_name}")
            return None

        api_model, model_class = CODE_PROCESSOR_MODELS[self.model_name]
        api_key = LLM_API_KEY.get(self.model_name)

        if not api_key:
            self.logger.error(f"API 키 누락: {self.model_name} API 키를 .env에 설정하세요.")
            return None

        try:
            return model_class(api_key=api_key, model=api_model, temperature=TEMPERATURES["code"])
        except Exception as e:
            raise Exception(f"{self.model_name} 모델 초기화 실패: {str(e)}")
    
    def _process_file_internal(self, file_path: str) -> Dict[str, Any]:
        """
        코드 파일 처리 내부 로직 (BaseProcessor 추상 메서드 구현)
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        return self.code_analyzer.process_code(file_path)


class CodeAnalyzer:
    """코드 분석 및 처리를 담당하는 클래스"""
    
    def __init__(self, selected_model: str):
        """
        코드 분석기 초기화
        
        Args:
            selected_model (str): 사용할 모델명 ('openai', 'gemini', 'claude', 'groq')
        """
        self.selected_model = selected_model.lower()
        self.llm = None
        
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("code_analyzer")
        self.error_handler = ErrorHandler.get_instance()
        
        # 프롬프트 설정
        self.system_prompt = CODE_PROCESSOR_PROMPT["system"]
        
        # 언어 감지기 설정
        self.language_detector = ProgrammingLanguageDetector()
        
        # 모듈 추출기 설정
        self.import_extractor = ImportExtractor()
        
        # LLM 모델 설정
        self._setup_llm()
    
    def _setup_llm(self):
        """LLM 모델 설정"""
        if self.selected_model not in CODE_PROCESSOR_MODELS:
            self.logger.error(f"지원하지 않는 모델: {self.selected_model}")
            self.llm = None
            return

        api_model, model_class = CODE_PROCESSOR_MODELS[self.selected_model]
        api_key = LLM_API_KEY.get(self.selected_model)

        if not api_key:
            self.logger.error(f"API 키 누락: {self.selected_model} API 키를 .env에 설정하세요.")
            self.llm = None
            return

        try:
            self.llm = model_class(api_key=api_key, model=api_model, temperature=TEMPERATURES["code"])
            self.logger.info(f"{self.selected_model} 모델 초기화 성공")
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"model_name": self.selected_model, "operation": "llm_initialization"}
            )
            self.logger.error(f"{self.selected_model} 모델 초기화 실패: {error_detail['error']}")
            self.llm = None
    
    def process_code(self, file_path: str) -> Dict[str, Any]:
        """
        코드 파일 처리
        
        Args:
            file_path (str): 분석할 코드 파일 경로
            
        Returns:
            Dict[str, Any]: {
                'file_info': {
                    'file_path': str,           # 전체 파일 경로
                    'file_name': str,           # 파일명
                    'creation_time': str,       # 생성 일시 (YYYY-MM-DD HH:MM:SS)
                    'modification_time': str,   # 수정 일시 (YYYY-MM-DD HH:MM:SS)
                    'language': str             # 프로그래밍 언어
                },
                'code_info': {
                    'imports': List[Dict],      # 임포트 정보 리스트
                    'raw_code': str,            # 코드 전체 내용
                    'analysis': {               # LLM 분석 결과
                        'purpose': str,         # 코드의 주요 목적
                        'logic': str            # 주요 알고리즘/로직 설명
                    }
                }
            }
        """
        try:
            # 파일 존재 확인
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"코드 파일을 찾을 수 없습니다: {file_path}")
            
            # 코드 내용 읽기
            code_content = self._read_code(file_path)
            
            # 메타데이터 추출
            metadata = self._extract_metadata(file_path)
            
            # 임포트 추출
            imports = self.import_extractor.extract_imports(code_content, metadata['language'])
            
            # 코드 분석
            analysis = self._generate_analysis(code_content)
            
            return {
                'file_info': {
                    'file_path': file_path,
                    'file_name': metadata['file_name'],
                    'creation_time': metadata['creation_time'],
                    'modification_time': metadata['modification_time'],
                    'language': metadata['language']
                },
                'code_info': {
                    'imports': imports,
                    'raw_code': code_content,
                    'analysis': analysis
                }
            }
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "code_processing"}
            )
            self.logger.error(f"코드 처리 중 오류 발생: {error_detail['error']}")
            raise
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
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
            
            # 언어 감지
            language = self.language_detector.detect_language(path)
            
            return {
                'file_name': path.name,
                'creation_time': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modification_time': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'language': language
            }
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "metadata_extraction"}
            )
            self.logger.error(f"메타데이터 추출 실패: {error_detail['error']}")
            return {
                'file_name': Path(file_path).name,
                'creation_time': 'N/A',
                'modification_time': 'N/A',
                'language': 'N/A'
            }
    
    def _read_code(self, file_path: str) -> str:
        """
        코드 파일 읽기
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            str: 파일 내용
        """
        path = Path(file_path)
        
        try:
            # Jupyter 노트북인 경우 특별 처리
            if path.suffix.lower() == '.ipynb':
                return self._read_notebook(file_path)
            
            # 일반 텍스트 파일
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
                
        except UnicodeDecodeError:
            # UTF-8로 읽을 수 없는 경우 다른 인코딩 시도
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise IOError(f"파일을 읽을 수 없습니다: {str(e)}")
        except Exception as e:
            raise IOError(f"파일을 읽을 수 없습니다: {str(e)}")
    
    def _read_notebook(self, file_path: str) -> str:
        """
        주피터 노트북 파일 읽기
        
        Args:
            file_path (str): 노트북 파일 경로
            
        Returns:
            str: 추출된 코드 내용
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            code_cells = []
            
            # 노트북 구조 확인
            if 'cells' not in notebook:
                raise ValueError("유효하지 않은 Jupyter 노트북 형식")
            
            # 코드 셀 추출
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = cell.get('source', [])
                    
                    # 소스 형식 처리
                    if isinstance(source, list):
                        # 빈 셀이나 매직 커맨드(%로 시작하는 라인) 제외
                        code = ''.join(line for line in source 
                                    if line.strip() and not line.strip().startswith('%'))
                        if code.strip():  # 빈 코드가 아닌 경우만 추가
                            code_cells.append(code)
                    elif isinstance(source, str) and source.strip():
                        if not source.strip().startswith('%'):
                            code_cells.append(source)
            
            # 코드가 너무 길 경우 앞부분의 중요한 부분만 사용
            combined_code = '\n\n'.join(code_cells)
            max_length = 5000  # 적절한 길이로 제한
            if len(combined_code) > max_length:
                # 첫 부분만 사용하고 축약 표시
                return combined_code[:max_length] + "\n\n# ... (코드가 너무 길어 축약됨)"
            
            return combined_code
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "notebook_reading"}
            )
            self.logger.error(f"노트북 파일 읽기 실패: {error_detail['error']}")
            return "# 노트북 파일을 읽을 수 없습니다."
    
    def _generate_analysis(self, code_content: str) -> Dict[str, str]:
        """
        LLM을 사용하여 코드 분석
        
        Args:
            code_content (str): 분석할 코드 내용
            
        Returns:
            Dict[str, str]: 분석 결과
        """
        if not self.llm:
            return {"error": "모델 초기화 실패"}
        
        # 코드 내용이 너무 길면 간략화
        if len(code_content) > 8000:
            truncated_code = code_content[:8000] + "\n\n# ... (코드가 너무 길어 축약됨)"
            self.logger.warning(f"코드가 너무 길어 축약됨: {len(code_content)} -> 8000 자")
            code_content = truncated_code
        
        # 프롬프트 생성
        prompt = CODE_PROCESSOR_PROMPT["user"].format(code_content=code_content)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            # 모델에 따라 토큰 사용량 추적 여부 결정
            if self.selected_model in ['openai', 'gemini']:
                with get_openai_callback() as cb:
                    response = self.llm.invoke(messages)
                    self.logger.info(f"{self.selected_model} 토큰 사용량: {cb.total_tokens}")
            else:
                response = self.llm.invoke(messages)
            
            # 응답 텍스트 파싱
            return self._parse_analysis_response(response.content)
                
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"operation": "code_analysis"}
            )
            self.logger.error(f"{self.selected_model} 분석 생성 실패: {error_detail['error']}")
            return {"error": str(e)}
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, str]:
        """
        LLM 응답 파싱하여 구조화된 분석 결과 생성
        
        Args:
            analysis_text (str): LLM 응답 텍스트
            
        Returns:
            Dict[str, str]: 구조화된 분석 결과
        """
        try:
            # 세 가지 부분을 구분하는 데 정규식 사용
            # 1. 목적과 기능
            purpose_pattern = r"1\.\s+(.*?)(?=2\.|\Z)"
            purpose_match = re.search(purpose_pattern, analysis_text, re.DOTALL)
            purpose_part = purpose_match.group(1).strip() if purpose_match else ""
            
            # 2. 로직과 알고리즘
            logic_pattern = r"2\.\s+(.*?)(?=3\.|\Z)"
            logic_match = re.search(logic_pattern, analysis_text, re.DOTALL)
            logic_part = logic_match.group(1).strip() if logic_match else ""
            
            # 3. Mermaid 다이어그램 추출
            mermaid_pattern = r"```mermaid\s*(.*?)```"
            mermaid_match = re.search(mermaid_pattern, analysis_text, re.DOTALL)
            mermaid_part = mermaid_match.group(1).strip() if mermaid_match else "flowchart LR\n    A[분석 실패] --> B[다이어그램을 생성할 수 없습니다.]"
            
            return {
                "purpose": purpose_part or "목적 분석을 추출할 수 없습니다.",
                "logic": logic_part or "로직 분석을 추출할 수 없습니다.",
                "flowchart": mermaid_part
            }
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"operation": "response_parsing"}
            )
            self.logger.error(f"응답 파싱 실패: {error_detail['error']}")
            return {
                "purpose": analysis_text[:500] + ("..." if len(analysis_text) > 500 else ""),
                "logic": "상세 로직 분석을 추출할 수 없습니다.",
                "flowchart": "flowchart LR\n    A[오류] --> B[응답 파싱 실패]"
            }


class ProgrammingLanguageDetector:
    """프로그래밍 언어 감지 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = LoggingManager.get_instance().get_logger("language_detector")
        self.error_handler = ErrorHandler.get_instance()
        
        # 확장자별 언어 매핑 (config.py에서 가져옴)
        self.extension_language_map = CODE_PROCESSOR_LANGUAGE
    
    def detect_language(self, file_path: Path) -> str:
        """
        파일 확장자 및 내용을 기반으로 프로그래밍 언어 감지
        
        Args:
            file_path (Path): 파일 경로
            
        Returns:
            str: 감지된 프로그래밍 언어
        """
        try:
            # 확장자 기반 감지
            extension = file_path.suffix.lower()
            if extension in self.extension_language_map:
                return self.extension_language_map[extension]
            
            # 확장자로 감지 못하면 파일 내용 분석
            return self._detect_from_content(file_path)
            
        except Exception as e:
            self.logger.warning(f"언어 감지 실패: {str(e)}")
            return "Unknown"
    
    def _detect_from_content(self, file_path: Path) -> str:
        """
        파일 내용을 기반으로 프로그래밍 언어 감지
        
        Args:
            file_path (Path): 파일 경로
            
        Returns:
            str: 감지된 프로그래밍 언어
        """
        try:
            # 파일의 첫 1000자만 읽기
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000)
            
            # 언어별 특징 패턴
            patterns = {
                'Python': [r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(', r'class\s+\w+\s*:'],
                'JavaScript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*=', r'import\s+.*from'],
                'Java': [r'public\s+class', r'public\s+static\s+void\s+main', r'import\s+java\.'],
                'C++': [r'#include\s*<\w+>', r'using\s+namespace\s+std', r'int\s+main\s*\('],
                'C#': [r'using\s+System', r'namespace\s+\w+', r'public\s+class'],
                'PHP': [r'<\?php', r'\$\w+\s*='],
                'HTML': [r'<!DOCTYPE\s+html>', r'<html>', r'<head>', r'<body>'],
                'CSS': [r'\w+\s*{', r'\w+\s*:\s*\w+']
            }
            
            # 각 언어별로 패턴 매칭 점수 계산
            scores = {lang: 0 for lang in patterns.keys()}
            
            for lang, patterns_list in patterns.items():
                for pattern in patterns_list:
                    if re.search(pattern, content, re.IGNORECASE):
                        scores[lang] += 1
            
            # 가장 높은 점수의 언어 선택
            if scores:
                best_match = max(scores.items(), key=lambda x: x[1])
                if best_match[1] > 0:
                    return best_match[0]
            
            return "Unknown"
            
        except Exception as e:
            self.logger.warning(f"파일 내용 기반 언어 감지 실패: {str(e)}")
            return "Unknown"


class ImportExtractor:
    """코드에서 임포트 정보를 추출하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = LoggingManager.get_instance().get_logger("import_extractor")
        self.error_handler = ErrorHandler.get_instance()
        
        # 언어별 임포트 추출 핸들러 등록
        self.extraction_handlers = {
            'Python': self._extract_python_imports,
            'Jupyter Notebook': self._extract_python_imports,
            'JavaScript': self._extract_javascript_imports,
            'TypeScript': self._extract_javascript_imports,
            'Java': self._extract_java_imports,
            'PHP': self._extract_php_imports
        }
    
    def extract_imports(self, code_content: str, language: str) -> List[Dict[str, str]]:
        """
        코드에서 임포트된 모듈/라이브러리 추출 및 버전 확인
        
        Args:
            code_content (str): 코드 내용
            language (str): 프로그래밍 언어
            
        Returns:
            List[Dict[str, str]]: 각 임포트에 대한 정보
            [
                {
                    'name': 'package_name',
                    'version': 'version_number',
                    'type': 'pip/built-in'
                },
                ...
            ]
        """
        try:
            # 해당 언어에 대한 핸들러가 있으면 사용
            handler = self.extraction_handlers.get(language)
            if handler:
                return handler(code_content)
            
            # 지원하지 않는 언어는 빈 목록 반환
            self.logger.warning(f"임포트 추출을 지원하지 않는 언어: {language}")
            return []
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"language": language, "operation": "import_extraction"}
            )
            self.logger.error(f"임포트 추출 실패: {error_detail['error']}")
            return []
    
    def _extract_python_imports(self, code_content: str) -> List[Dict[str, str]]:
        """Python 코드에서 임포트 추출"""
        imports = []
        imported_modules = set()
        
        try:
            # 주피터 노트북의 경우 여러 코드 블록으로 나뉘어 있을 수 있음
            code_blocks = code_content.split('\n\n')
            
            for code_block in code_blocks:
                if not code_block.strip():  # 빈 코드 블록 무시
                    continue
                    
                try:
                    tree = ast.parse(code_block)
                    
                    # 임포트 구문 추출
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                imported_modules.add(name.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:  # from ... import ... 구문
                                imported_modules.add(node.module.split('.')[0])
                            else:  # from . import ... 구문 (상대 임포트)
                                for name in node.names:
                                    imported_modules.add(name.name.split('.')[0])
                except SyntaxError:
                    # 문법 오류가 있는 코드 블록은 건너뛰기
                    continue
                except Exception:
                    # 기타 오류가 있는 코드 블록은 건너뛰기
                    continue
            
            # 버전 정보 추가
            for module_name in sorted(imported_modules):
                module_info = {'name': module_name}
                
                try:
                    # 빌트인 모듈 확인
                    if module_name in sys.stdlib_module_names:
                        module_info['version'] = sys.version.split()[0]
                        module_info['type'] = 'built-in'
                    else:
                        # pip 패키지 버전 확인
                        try:
                            module_info['version'] = importlib.metadata.version(module_name)
                            module_info['type'] = 'pip'
                        except importlib.metadata.PackageNotFoundError:
                            # 일부 특수한 케이스 처리 (예: tensorflow-gpu -> tensorflow)
                            alternate_name = module_name.split('-')[0]
                            try:
                                module_info['version'] = importlib.metadata.version(alternate_name)
                                module_info['type'] = 'pip'
                            except importlib.metadata.PackageNotFoundError:
                                module_info['version'] = 'unknown'
                                module_info['type'] = 'unknown'
                except Exception:
                    module_info['version'] = 'unknown'
                    module_info['type'] = 'unknown'
                
                imports.append(module_info)
            
        except Exception as e:
            self.logger.error(f"Python 임포트 추출 실패: {str(e)}")
        
        return imports
    
    def _extract_javascript_imports(self, code_content: str) -> List[Dict[str, str]]:
        """JavaScript/TypeScript 코드에서 임포트 추출"""
        imports = []
        package_names = set()
        
        try:
            # import/require 구문 매칭
            import_patterns = [
                r'import\s+{[^}]+}\s+from\s+[\'"]([^\'"]+)[\'"]',   # import { something } from 'package'
                r'import\s+\w+\s+from\s+[\'"]([^\'"]+)[\'"]',       # import something from 'package'
                r'import\s+[\'"]([^\'"]+)[\'"]',                    # import 'package'
                r'require\([\'"]([^\'"]+)[\'"]\)'                   # require('package')
            ]
            
            for pattern in import_patterns:
                matches = re.finditer(pattern, code_content)
                for match in matches:
                    package = match.group(1)
                    if package:
                        # 스코프 패키지(@로 시작) 또는 첫 번째 경로 부분을 패키지명으로 사용
                        if package.startswith('@'):
                            parts = package.split('/')[:2]
                            package_name = '/'.join(parts)
                        else:
                            # 상대 경로 임포트 건너뛰기
                            if package.startswith('.'):
                                continue
                            package_name = package.split('/')[0]
                            
                        package_names.add(package_name)
            
            # 결과 구성
            for package_name in sorted(package_names):
                imports.append({
                    'name': package_name,
                    'version': 'unknown',  # npm 패키지 버전은 코드 분석만으로는 알 수 없음
                    'type': 'npm'
                })
                
        except Exception as e:
            self.logger.error(f"JavaScript 임포트 추출 실패: {str(e)}")
        
        return imports
    
    def _extract_java_imports(self, code_content: str) -> List[Dict[str, str]]:
        """Java 코드에서 임포트 추출"""
        imports = []
        package_names = set()
        
        try:
            # import 구문 매칭
            import_pattern = r'import\s+([^;]+);'
            matches = re.finditer(import_pattern, code_content)
            
            for match in matches:
                import_path = match.group(1).strip()
                if import_path:
                    # 최상위 패키지 이름 추출 (java.util.List -> java)
                    parts = import_path.split('.')
                    if parts and parts[0] not in ('java', 'javax'):
                        # 자바 기본 패키지가 아닌 외부 패키지만 추가
                        package_names.add(parts[0])
            
            # 기본 자바 패키지 추가
            if re.search(r'import\s+java\.', code_content):
                package_names.add('java')
            if re.search(r'import\s+javax\.', code_content):
                package_names.add('javax')
            
            # 결과 구성
            for package_name in sorted(package_names):
                imports.append({
                    'name': package_name,
                    'version': 'unknown',
                    'type': 'java-package'
                })
                
        except Exception as e:
            self.logger.error(f"Java 임포트 추출 실패: {str(e)}")
        
        return imports
    
    def _extract_php_imports(self, code_content: str) -> List[Dict[str, str]]:
        """PHP 코드에서 임포트 추출"""
        imports = []
        package_names = set()
        
        try:
            # use, require, include 구문 매칭
            patterns = [
                r'use\s+([^;]+);',                      # use Namespace\Class;
                r'require(?:_once)?\s*\(?\s*[\'"]([^\'"]+)[\'"]\)?',  # require('file.php')
                r'include(?:_once)?\s*\(?\s*[\'"]([^\'"]+)[\'"]\)?',  # include('file.php')
                r'composer\s+require\s+([^\s]+)'        # composer require package/name
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, code_content)
                for match in matches:
                    name = match.group(1).strip()
                    if name:
                        # Namespace 처리
                        if '\\' in name:
                            parts = name.split('\\')
                            # 첫 부분만 패키지명으로 사용
                            package_names.add(parts[0])
                        else:
                            package_names.add(name)
            
            # composer.json 문자열 탐지
            composer_pattern = r'"require"\s*:\s*{([^}]+)}'
            composer_match = re.search(composer_pattern, code_content)
            if composer_match:
                require_block = composer_match.group(1)
                pkg_pattern = r'"([^"]+)"\s*:'
                for pkg_match in re.finditer(pkg_pattern, require_block):
                    package_names.add(pkg_match.group(1))
            
            # 결과 구성
            for package_name in sorted(package_names):
                imports.append({
                    'name': package_name,
                    'version': 'unknown',
                    'type': 'php-package'
                })
                
        except Exception as e:
            self.logger.error(f"PHP 임포트 추출 실패: {str(e)}")
        
        return imports