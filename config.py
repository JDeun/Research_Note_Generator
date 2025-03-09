# config.py
import os
from typing import Dict, Tuple, Any, Type
import dotenv

# .env 파일 로드
dotenv.load_dotenv()

# LLM API 키 설정
LLM_API_KEY = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GOOGLE_API_KEY"),
    "claude": os.getenv("ANTHROPIC_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY")
}

# 동적 임포트 함수
def get_model_classes():
    """
    필요한 LLM 모델 클래스 동적 임포트
    
    Returns:
        Dict[str, Type]: 모델 이름과 클래스 매핑
    """
    model_classes = {}
    
    try:
        from langchain_openai import ChatOpenAI
        model_classes["openai"] = ChatOpenAI
    except ImportError:
        pass
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_classes["gemini"] = ChatGoogleGenerativeAI
    except ImportError:
        pass
    
    try:
        from langchain_anthropic import ChatAnthropic
        model_classes["claude"] = ChatAnthropic
    except ImportError:
        pass
    
    try:
        from langchain_groq import ChatGroq
        model_classes["groq"] = ChatGroq
    except ImportError:
        pass
    
    return model_classes

# 모델 클래스 로드
MODEL_CLASSES = get_model_classes()

# Models for each processor
IMAGE_PROCESSOR_MODELS = {
    "openai": ("gpt-4o-mini", MODEL_CLASSES.get("openai")),
    "gemini": ("gemini-2.0-flash-lite", MODEL_CLASSES.get("gemini")),
    "claude": ("claude-3-5-sonnet-20241022", MODEL_CLASSES.get("claude")),
    "groq": ("llama-3.2-90b-vision-preview", MODEL_CLASSES.get("groq"))
}

CODE_PROCESSOR_MODELS = {
    "openai": ("gpt-4o-mini", MODEL_CLASSES.get("openai")),
    "gemini": ("gemini-2.0-flash-lite", MODEL_CLASSES.get("gemini")),
    "claude": ("claude-3-5-sonnet-20241022", MODEL_CLASSES.get("claude")),
    "groq": ("llama-3.3-70b-versatile", MODEL_CLASSES.get("groq"))
}

DOCUMENT_PROCESSOR_MODELS = {
    "openai": ("gpt-4o-mini", MODEL_CLASSES.get("openai")),
    "gemini": ("gemini-2.0-flash-lite", MODEL_CLASSES.get("gemini")),
    "claude": ("claude-3-5-sonnet-20241022", MODEL_CLASSES.get("claude")),
    "groq": ("llama-3.2-90b-vision-preview", MODEL_CLASSES.get("groq"))
}

RESEARCH_NOTE_GENERATOR_MODELS = {
    "openai": ("gpt-4o-mini", MODEL_CLASSES.get("openai")),
    "gemini": ("gemini-2.0-flash-lite", MODEL_CLASSES.get("gemini")),
    "claude": ("claude-3-5-sonnet-20241022", MODEL_CLASSES.get("claude")),
    "groq": ("llama-3.3-70b-versatile", MODEL_CLASSES.get("groq"))
}

# LLM Settings for each processor  
TEMPERATURES = {
    "image": 0.0,
    "code": 0.0,
    "document": 0.0,
    "research_note": 0.4
}

# 프로그래밍 언어 감지를 위한 확장자 매핑
CODE_PROCESSOR_LANGUAGE = {
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

# 파일 확장자별 유형 매핑
FILE_EXTENSION = {
    # 이미지 파일
    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
    '.bmp': 'image', '.tiff': 'image', '.webp': 'image',
    
    # 코드 파일
    '.py': 'code', '.js': 'code', '.java': 'code', '.cpp': 'code',
    '.c': 'code', '.cs': 'code', '.php': 'code', '.go': 'code',
    '.ts': 'code', '.jsx': 'code', '.tsx': 'code', '.html': 'code',
    '.css': 'code', '.sql': 'code', '.r': 'code', '.ipynb': 'code',
    
    # 문서 파일
    '.pdf': 'document', '.docx': 'document', '.doc': 'document',
    '.pptx': 'document', '.ppt': 'document', '.xlsx': 'document',
    '.xls': 'document', '.txt': 'document', '.md': 'document',
    '.rtf': 'document', '.csv': 'document', '.json': 'document',
    '.xml': 'document'
}