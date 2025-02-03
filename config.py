import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

dotenv.load_dotenv()

LLM_API_KEY = {
    "chatgpt": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GOOGLE_API_KEY"),
    "claude": os.getenv("ANTHROPIC_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY")
}

# Models for each processor
IMAGE_PROCESSOR_MODELS = {
    "chatgpt": ("gpt-4o-mini", ChatOpenAI),
    "gemini": ("gemini-1.5-flash-8b", ChatGoogleGenerativeAI),
    "claude": ("claude-3-5-sonnet-20241022", ChatAnthropic),
    "groq": ("llama-3.2-90b-vision-preview", ChatGroq)
}

CODE_PROCESSOR_MODELS = {
    "chatgpt": ("gpt-4o-mini", ChatOpenAI),
    "gemini": ("gemini-1.5-flash-8b", ChatGoogleGenerativeAI),
    "claude": ("claude-3-haiku-20240307", ChatAnthropic),
    "groq": ("llama-3.3-70b-versatile", ChatGroq)
}

DOCUMENT_PROCESSOR_MODELS = {
    "chatgpt": ("gpt-4o-mini", ChatOpenAI),
    "gemini": ("gemini-1.5-flash-8b", ChatGoogleGenerativeAI),
    "claude": ("claude-3-haiku-20240307", ChatAnthropic),
    "groq": ("llama-3.3-70b-versatile", ChatGroq)
}

# LLM Settings for each processor  
TEMPERATURES = {
    "image": 0.0,
    "code": 0.0,
    "document": 0.0
}

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