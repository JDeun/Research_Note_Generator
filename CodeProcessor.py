import re
import json
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import logging
from pathlib import Path
import ast
import sys
import importlib.metadata
import dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

from config import CODE_PROCESSOR_MODELS, TEMPERATURES

# .env 파일 로드
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Names
CHATGPT = CODE_PROCESSOR_MODELS["chatgpt"]
GEMINI = CODE_PROCESSOR_MODELS["gemini"]
CLAUDE = CODE_PROCESSOR_MODELS["claude"]
GROQ = CODE_PROCESSOR_MODELS["groq"]

# LLM Settings
TEMPERATURE = TEMPERATURES["code"]

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    def __init__(self, selected_model: str):
        """코드 분석기 초기화
        
        Args:
            selected_model (str): 사용할 모델명 ('openai', 'gemini', 'claude')
        """
        self.selected_model = selected_model.lower()
        self.llm = None
        self.system_prompt = """당신은 상세한 코드 분석과 문서화를 수행하는 전문 분석가입니다.
주어진 코드를 체계적으로 분석하여 코드의 목적, 로직, 데이터 흐름을 명확하게 설명해야 합니다.
설명은 기술적으로 정확하면서도 이해하기 쉽게 한국어로 작성해주세요."""
        self._setup_llm()

    def _setup_llm(self):
        """LLM 모델 설정"""
        model_configs = {
            'openai': (ChatOpenAI, {'api_key': OPENAI_API_KEY, 'model': CHATGPT}),
            'gemini': (ChatGoogleGenerativeAI, {'api_key': GOOGLE_API_KEY, 'model': GEMINI}),
            'claude': (ChatAnthropic, {'api_key': ANTHROPIC_API_KEY, 'model': CLAUDE}),
            'groq': (ChatGroq, {'api_key': GROQ_API_KEY, 'model': GROQ})
        }
        
        if self.selected_model in model_configs:
            model_class, config = model_configs[self.selected_model]
            try:
                self.llm = model_class(temperature=TEMPERATURE, **config)
                logger.info(f"{self.selected_model} 모델 초기화 성공")
            except Exception as e:
                logger.error(f"{self.selected_model} 모델 초기화 실패: {str(e)}")
                self.llm = None
        else:
            logger.error(f"지원하지 않는 모델: {self.selected_model}")
            self.llm = None

    def process_code(self, file_path: str) -> Dict[str, Any]:
        """코드 파일 처리
        
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
            code_content = self._read_code(file_path)
            metadata = self._extract_metadata(file_path)
            imports = self._extract_imports(code_content, metadata['language'])
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
            logger.error(f"코드 처리 중 오류 발생: {str(e)}")
            raise

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """파일 메타데이터 추출
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        try:
            path = Path(file_path)
            stats = path.stat()
            
            return {
                'file_name': path.name,
                'creation_time': datetime.fromtimestamp(stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modification_time': datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'language': self._detect_language(path)
            }
        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {str(e)}")
            return {
                'file_name': 'N/A',
                'creation_time': 'N/A',
                'modification_time': 'N/A',
                'language': 'N/A'
            }

    def _detect_language(self, path: Path) -> str:
        """파일 확장자를 기반으로 프로그래밍 언어 감지
        
        Args:
            path (Path): 파일 경로
            
        Returns:
            str: 감지된 프로그래밍 언어
        """
        extension_map = {
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
        return extension_map.get(path.suffix.lower(), 'Unknown')

    def _read_code(self, file_path: str) -> str:
        """코드 파일 읽기"""
        if file_path.lower().endswith('.ipynb'):
            return self._read_notebook(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _read_notebook(self, file_path: str) -> str:
        """주피터 노트북 파일 읽기"""
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        code_cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = cell.get('source', [])
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
        max_length = 2000  # 적절한 길이로 제한
        if len(combined_code) > max_length:
            # 첫 번째 코드 셀과 임포트문 포함
            return combined_code[:max_length] + "\n\n# ... (이하 생략)"
        
        return combined_code

    def _extract_imports(self, code_content: str, language: str) -> List[Dict[str, str]]:
        """코드에서 임포트된 모듈/라이브러리 추출 및 버전 확인
        
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
        imports = []
        imported_modules = set()
        
        if language in ['Python', 'Jupyter Notebook']:
            try:
                # 주피터 노트북의 경우 여러 코드 블록으로 나뉘어 있을 수 있음
                code_blocks = code_content.split('\n\n') if language == 'Jupyter Notebook' else [code_content]
                
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
                        logger.warning("구문 오류가 있는 코드 블록을 건너뜁니다.")
                        continue
                    except Exception as e:
                        logger.error(f"코드 블록 파싱 중 오류 발생: {str(e)}")
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
                    except Exception as e:
                        logger.warning(f"버전 확인 실패 - {module_name}: {str(e)}")
                        module_info['version'] = 'unknown'
                        module_info['type'] = 'unknown'
                    
                    imports.append(module_info)
                
                logger.info(f"추출된 임포트 정보: {imports}")
                
            except Exception as e:
                logger.error(f"Python/Jupyter 임포트 추출 실패: {str(e)}")
        
        elif language == 'JavaScript':
            try:
                # import/require 구문 매칭
                import_pattern = r'(?:import\s+{[^}]+}|import\s+\w+)\s+from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\)'
                matches = re.finditer(import_pattern, code_content)
                
                package_names = set()
                for match in matches:
                    # import나 require에서 패키지명 추출
                    package = match.group(1) or match.group(2)
                    if package:
                        # 스코프 패키지(@로 시작) 또는 첫 번째 경로 부분을 패키지명으로 사용
                        if package.startswith('@'):
                            parts = package.split('/')[:2]
                            package_name = '/'.join(parts)
                        else:
                            package_name = package.split('/')[0]
                        package_names.add(package_name)
                
                # 버전 정보는 알 수 없으므로 unknown으로 설정
                for package_name in sorted(package_names):
                    imports.append({
                        'name': package_name,
                        'version': 'unknown',
                        'type': 'npm'
                    })
                    
            except Exception as e:
                logger.error(f"JavaScript 임포트 추출 실패: {str(e)}")
        
        return imports

    def _generate_analysis(self, code_content: str) -> Dict[str, str]:
        """LLM을 사용하여 코드 분석
        
        Args:
            code_content (str): 분석할 코드 내용
            
        Returns:
            Dict[str, str]: 분석 결과
        """
        if not self.llm:
            return {"error": "모델 초기화 실패"}
                
        prompt = f"""다음 코드를 분석하여 세 가지 관점에서 설명해주세요:

    1. 코드의 주요 목적과 핵심 기능
    - 이 코드가 무엇을 하는지
    - 어떤 문제를 해결하는지
    - 주요 기능과 특징

    2. 데이터 처리 로직과 알고리즘
    - 주요 데이터 처리 단계
    - 사용된 알고리즘이나 특별한 기법
    - 핵심 함수와 메서드의 역할

    3. 데이터 흐름도 (Mermaid 형식)
    - 입력 데이터부터 출력까지의 과정
    - 주요 함수들 간의 데이터 흐름
    - 중간 처리 과정과 변환 단계

    코드:
```
{code_content}
```
각 섹션을 명확히 구분하여 작성해주세요. 
데이터 흐름도는 반드시 "```mermaid"로 시작하고 "```"로 끝나야 합니다:
flowchart LR
    A[입력] --> B[처리1]
    B --> C[처리2]
    C --> D[출력]

각 단계별로 실제 데이터 타입과 변환 과정을 포함해주세요.
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            if self.selected_model in ['openai', 'gemini']:
                with get_openai_callback() as cb:
                    response = self.llm.invoke([
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=prompt)
                    ])
                    logger.info(f"{self.selected_model} 토큰 사용량: {cb.total_tokens}")
            else:
                response = self.llm.invoke([
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt)
                ])
            
            # 응답 텍스트 파싱
            analysis_text = response.content
            
            try:
                # 목적과 기능 추출
                purpose_part = analysis_text.split("2.")[0].replace("1.", "").strip()
                
                # 로직과 알고리즘 추출
                logic_part = analysis_text.split("2.")[1].split("3.")[0].strip()
                
                # Mermaid 다이어그램 추출
                mermaid_pattern = r"```mermaid\n(.*?)```"
                mermaid_match = re.search(mermaid_pattern, analysis_text, re.DOTALL)
                mermaid_part = mermaid_match.group(1).strip() if mermaid_match else "flowchart LR\n    A[분석 실패] --> B[다이어그램을 생성할 수 없습니다.]"
                
                return {
                    "purpose": purpose_part,
                    "logic": logic_part,
                    "flowchart": mermaid_part
                }
                
            except Exception as e:
                logger.error(f"응답 파싱 실패: {str(e)}")
                return {
                    "purpose": analysis_text,
                    "logic": "상세 로직 분석을 추출할 수 없습니다.",
                    "flowchart": "flowchart LR\n    A[오류] --> B[응답 파싱 실패]"
                }
                
        except Exception as e:
            error_msg = f"Error code: {getattr(e, 'status_code', 'Unknown')} - {str(e)}"
            logger.error(f"{self.selected_model} 분석 생성 실패: {error_msg}")
            return {"error": error_msg}
def _extract_code_preview(code: str, max_lines: int = 20) -> str:
    """코드 미리보기 추출 (주석 제외, 최대 라인수 제한)
    
    Args:
        code (str): 전체 코드
        max_lines (int): 최대 표시할 라인 수
    
    Returns:
        str: 주석이 제외된 코드 미리보기
    """
    # 코드를 라인 단위로 분리
    lines = code.split('\n')
    
    # 주석이 아닌 라인만 필터링
    code_lines = []
    for line in lines:
        # 빈 줄 무시
        stripped = line.strip()
        if not stripped:
            continue
        # 주석 줄 무시
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # 인라인 주석 제거
        if '#' in line:
            line = line[:line.index('#')].rstrip()
        if line.strip():
            code_lines.append(line)
    
    # 최대 라인수만큼 반환
    preview_lines = code_lines[:max_lines]
    
    # 최대 라인수를 초과하는 경우 표시
    if len(code_lines) > max_lines:
        preview_lines.append('...')
    
    return '\n'.join(preview_lines)

def main():
    """메인 실행 함수"""
    valid_models = ['openai', 'gemini', 'claude', 'groq']
    
    # 모델 선택
    while True:
        selected_model = input("\n사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
        if selected_model in valid_models:
            break
        print("올바른 모델명을 입력해주세요.")
    
    analyzer = CodeAnalyzer(selected_model)
    
    if not analyzer.llm:
        print(f"\n{selected_model.upper()} 모델 초기화에 실패했습니다.")
        return
    
    # 코드 파일 처리
    while True:
        file_path = input("\n처리할 코드 파일 경로를 입력하세요 (종료하려면 엔터): ").strip()
        if not file_path:
            break
            
        try:
            result = analyzer.process_code(file_path)
            file_info = result['file_info']
            code_info = result['code_info']
            
            print("\n" + "="*50)
            print("코드 분석 결과")
            print("="*50)
            
            print("\n1. 파일 정보")
            print("-"*50)
            print(f"파일명: {file_info['file_name']}")
            print(f"파일 경로: {file_info['file_path']}")
            print(f"생성 일시: {file_info['creation_time']}")
            print(f"최종 수정 일시: {file_info['modification_time']}")
            print(f"프로그래밍 언어: {file_info['language']}")
            
            print("\n2. 임포트된 모듈/라이브러리")
            print("-"*50)
            if code_info['imports']:
                print(f"{'모듈명':<20} {'버전':<15} {'유형':<10}")
                print("-"*45)
                for imp in code_info['imports']:
                    print(f"{imp['name']:<20} {imp['version']:<15} {imp['type']:<10}")
            else:
                print("임포트된 모듈이 없습니다.")
            
            print("\n3. 코드 미리보기 (주석 제외 첫 20줄)")
            print("-"*50)
            preview = _extract_code_preview(code_info['raw_code'])
            print(preview)
            
            print(f"\n4. {selected_model.upper()} 모델의 코드 분석")
            print("-"*50)
            if "error" in code_info['analysis']:
                print(f"분석 실패: {code_info['analysis']['error']}")
            else:
                print(f"[코드의 주요 목적과 기능]")
                print(code_info['analysis']['purpose'])
                print(f"\n[주요 알고리즘과 로직]")
                print(code_info['analysis']['logic'])
                print(f"\n[데이터 처리 흐름도]")
                print("```mermaid")
                print(code_info['analysis']['flowchart'])
                print("```")
            
            print("\n" + "="*50)
                
        except Exception as e:
            print(f"\n코드 처리 중 오류가 발생했습니다: {str(e)}")
            continue
            
    print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()