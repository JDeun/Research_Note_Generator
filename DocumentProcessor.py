import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import dotenv
from config import DOCUMENT_PROCESSOR_MODELS, TEMPERATURES

# langchain & community
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain

# 여러 모델
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# 토큰 사용량 모니터링 (옵션)
from langchain_community.callbacks import get_openai_callback

# docling + python-pptx
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    from pptx import Presentation
    PYTHON_PPTX_AVAILABLE = True
except ImportError:
    PYTHON_PPTX_AVAILABLE = False

# .env 파일 로드
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Names
CHATGPT = DOCUMENT_PROCESSOR_MODELS.get("chatgpt")
GEMINI = DOCUMENT_PROCESSOR_MODELS.get("gemini")
CLAUDE = DOCUMENT_PROCESSOR_MODELS.get("claude")
GROQ = DOCUMENT_PROCESSOR_MODELS.get("groq")

# LLM Settings
TEMPERATURE = TEMPERATURES["document"]
SHORT_DOC_THRESHOLD = 3000  # 짧은 문서 기준 (문자 수)

class DocumentAnalyzer:
    """
    (1) 문서 로딩 (PPTX → docling -> fallback python-pptx)
    (2) 길이 짧으면 direct 요약, 길면 SummarizeChain
    (3) 최종으로 title, author, purpose, summary, caption 추출
    """
    def __init__(self, selected_model: str = "openai"):
        self.selected_model = selected_model.lower()
        self.llm = None
        self._setup_llm()

    def _setup_llm(self):
        model_configs = {
            'openai': {
                'class': ChatOpenAI,
                'kwargs': {
                    'api_key': OPENAI_API_KEY,
                    'temperature': TEMPERATURE,
                    'model': CHATGPT,
                }
            },
            'anthropic': {
                'class': ChatAnthropic,
                'kwargs': {
                    'api_key': ANTHROPIC_API_KEY,
                    'temperature': TEMPERATURE,
                    'model': CLAUDE,
                }
            },
            'google': {
                'class': ChatGoogleGenerativeAI,
                'kwargs': {
                    'api_key': GOOGLE_API_KEY,
                    'temperature': TEMPERATURE,
                    'model': GEMINI,
                }
            },
            'groq': {
                'class': ChatGroq,
                'kwargs': {
                    'api_key': GROQ_API_KEY,
                    'temperature': TEMPERATURE,
                    'model': GROQ,
                }
            }
        }
        config = model_configs.get(self.selected_model)
        if not config:
            print(f"[에러] 지원하지 않는 모델: {self.selected_model}")
            return
        klass = config["class"]
        try:
            self.llm = klass(**config["kwargs"])
        except Exception as e:
            print(f"[에러] LLM 초기화 실패: {str(e)}")

    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        for fp in file_paths:
            try:
                result = self.process_single_document(fp)
                results.append(result)
            except Exception as e:
                results.append({"file_path": fp, "error": str(e)})
        return results

    def process_single_document(self, file_path: str) -> Dict[str, Any]:
        metadata = self._extract_metadata(file_path)
        docs = self._load_document(file_path)
        if not docs:
            return {
                "file_info": metadata,
                "error": "문서에서 텍스트를 추출하지 못했습니다."
            }

        # 문서 전체 텍스트
        full_text = "\n".join(d.page_content for d in docs)
        doc_length = len(full_text)

        # 짧으면 직접 요약, 길면 SummarizeChain
        if doc_length < SHORT_DOC_THRESHOLD:
            summary_text = self._direct_llm_summary(full_text)
        else:
            splitted_docs = self._split_docs(docs)
            summary_text = self._map_reduce_summary(splitted_docs)

        # JSON 분석 (title, author, purpose, summary, caption)
        info_json = self._extract_info_json(summary_text)

        # 구조화된 텍스트
        structured_text = self._structure_text(docs)

        return {
            "file_info": metadata,
            "doc_length": doc_length,
            "map_reduce_summary": summary_text,
            "analysis_result": info_json,
            "text_content": structured_text
        }

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        p = Path(file_path)
        try:
            st = p.stat()
            return {
                "file_path": file_path,
                "file_name": p.name,
                "file_size": st.st_size,
                "created_time": str(datetime.fromtimestamp(st.st_ctime)),
                "modified_time": str(datetime.fromtimestamp(st.st_mtime)),
                "file_type": self._detect_file_type(p)
            }
        except:
            return {
                "file_path": file_path,
                "file_name": p.name,
                "file_type": self._detect_file_type(p),
                "file_size": None,
                "created_time": None,
                "modified_time": None
            }

    def _detect_file_type(self, p: Path) -> str:
        ext = p.suffix.lower()
        if ext == ".pdf":
            return "PDF"
        elif ext == ".pptx":
            return "PPTX"
        elif ext == ".xlsx":
            return "XLSX"
        elif ext == ".docx":
            return "DOCX"
        else:
            return "Unknown"

    def _load_document(self, file_path: str) -> List[Document]:
        """문서 로딩 (PPTX만 docling -> python-pptx fallback, 나머지는 기존 로더)"""
        ftype = self._detect_file_type(Path(file_path))

        try:
            if ftype == "PDF":
                loader = PyPDFLoader(file_path)
                return loader.load()
            elif ftype == "XLSX":
                loader = UnstructuredExcelLoader(file_path)
                return loader.load()
            elif ftype == "DOCX":
                loader = Docx2txtLoader(file_path)
                return loader.load()
            elif ftype == "PPTX":
                # 1) docling 시도
                docs = self._try_docling_pptx(file_path)
                if docs and len(docs) > 0:
                    return docs
                # 2) fallback to python-pptx
                text = self._fallback_python_pptx(file_path)
                if not text:
                    return []
                # Document로 감싸기
                return [Document(page_content=text, metadata={"source": file_path})]
            else:
                raise ValueError("지원하지 않는 파일 형식")
        except Exception as e:
            print(f"[에러] 문서 로딩 실패: {str(e)}")
            return []

    def _try_docling_pptx(self, file_path: str) -> List[Document]:
        """docling으로 pptx 처리 시도, 실패하면 빈 리스트 반환"""
        if not DOCLING_AVAILABLE:
            print("[docling] not installed, skipping docling process.")
            return []
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(file_path)
            md_text = result.document.export_to_markdown()
            if not md_text.strip():
                print("[docling] 변환 결과가 비어있음. fallback 예정")
                return []
            # docling 결과를 Document로
            return [Document(page_content=md_text, metadata={"source": file_path})]
        except Exception as e:
            print(f"[docling] 변환 실패: {str(e)}")
            return []

    def _fallback_python_pptx(self, file_path: str) -> str:
        """python-pptx로 pptx 처리. 실패하면 빈 문자열 반환"""
        if not PYTHON_PPTX_AVAILABLE:
            print("[python-pptx] not installed, cannot fallback.")
            return ""
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            slides_text = []
            for slide in prs.slides:
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_texts.append(shape.text)
                slides_text.append("\n".join(slide_texts))
            return "\n\n".join(slides_text)
        except Exception as e:
            print(f"[python-pptx] 변환 실패: {str(e)}")
            return ""

    def _split_docs(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100
        )
        splitted = []
        for d in docs:
            sub_texts = splitter.split_text(d.page_content)
            for t in sub_texts:
                splitted.append(Document(page_content=t, metadata=d.metadata))
        return splitted

    def _map_reduce_summary(self, splitted_docs: List[Document]) -> str:
        """긴 문서: Map-Reduce SummarizeChain"""
        if not self.llm:
            return "[LLM 초기화 실패]"
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        try:
            if self.selected_model == "openai" and get_openai_callback:
                with get_openai_callback() as cb:
                    result = chain.run(splitted_docs)
            else:
                result = chain.run(splitted_docs)
            return result
        except Exception as e:
            return f"[map_reduce 요약 실패: {str(e)}]"

    def _direct_llm_summary(self, full_text: str) -> str:
        """짧은 문서는 LLM에 통째로 전달하여 요약"""
        if not self.llm:
            return "[LLM 초기화 실패]"
        system_msg = SystemMessage(content="당신은 전문 문서 분석가입니다. 한국어로 작성하세요.")
        user_msg = HumanMessage(content=f"""아래 문서를 자세히 요약해 주세요 (3~5문단):
\"\"\"
{full_text}
\"\"\"""")
        try:
            resp = self.llm.invoke([system_msg, user_msg])
            return resp.content.strip()
        except Exception as e:
            return f"[direct_llm_summary 실패: {str(e)}]"

    def _extract_info_json(self, final_summary: str) -> Dict[str, Any]:
        """
        최종 요약결과 → title, author, purpose, summary, caption 등 JSON 반환
        """
        if not self.llm:
            return {"error": "LLM 미초기화"}
        if final_summary.startswith("[map_reduce 요약 실패") or final_summary.startswith("[direct_llm_summary 실패"):
            return {"error": final_summary}

        system_prompt = SystemMessage(content=(
            "당신은 전문 문서 분석가입니다. "
            "주어진 문서 요약을 토대로, 문서에서 유추 가능한 핵심 정보를 분석하고 title/author/purpose/summary/caption을 추출해 주세요. "
            "반드시 JSON 형식이어야 합니다."
        ))
        user_prompt = f"""
아래는 문서 전체 요약입니다:

\"\"\"
{final_summary}
\"\"\"

이 요약을 바탕으로 JSON을 만들어 주세요. 
키는 반드시 아래와 같이 5개:
1) "title": 문서의 제목(혹은 추정 주제)
2) "author": 작성자/발행인/발행처 (명시돼 있지 않다면 추론, 대략적 주제)
3) "purpose": 작성 의도나 목적, 배경
4) "summary": 문서 내용 전반에 대한 3~5문단 정도의 자세한 요약
5) "caption": 문서 내용 전체를 쉽게 파악할 수 있는 3~4문장 정도의 간략한 설명

예시:
{{
  "title": "...",
  "author": "...",
  "purpose": "...",
  "summary": "...",
  "caption": "..."
}}

주의 사항:
- JSON 키는 반드시 위와 같은 이름으로 사용해야 합니다.
- JSON의 모든 내용은 반드시 한국어로 작성하되, 고유명사, 이름, 기술용어 등의 꼭 필요한 경우에는 외래어나 영어를 사용합니다.
- 존재하지 않는 정보는 무리하게 만들지 마세요. 주어진 문서를 기반으로 추론 가능한 경우에만 작성해 주세요.
"""
        user_msg = HumanMessage(content=user_prompt)
        try:
            resp = self.llm.invoke([system_prompt, user_msg])
            content = resp.content.strip()

            match = re.search(r'(\{[\s\S]+\})', content)
            if match:
                json_part = match.group(1)
                data = json.loads(json_part)
                return {
                    "title": data.get("title", ""),
                    "author": data.get("author", ""),
                    "purpose": data.get("purpose", ""),
                    "summary": data.get("summary", ""),
                    "caption": data.get("caption", "")
                }
            else:
                return {
                    "error": "JSON 파싱 실패",
                    "raw_response": content
                }
        except Exception as e:
            return {"error": f"문서 정보 추출 실패: {str(e)}"}

    def _structure_text(self, docs: List[Document]) -> Dict[str, Any]:
        """페이지(슬라이드 등)별 200자 프리뷰"""
        pages = []
        for i, d in enumerate(docs, start=1):
            preview = d.page_content
            if len(preview) > 200:
                preview = preview[:200] + "..."
            pages.append({"page_index": i, "text_preview": preview})
        return {"pages": pages}


def main():
    print("=== Document Analyzer ===")
    model = input("사용할 모델 (openai/anthropic/google/groq) [기본 openai]: ").strip().lower()
    if model not in ["openai", "anthropic", "google", "groq"]:
        model = "openai"

    analyzer = DocumentAnalyzer(selected_model=model)

    file_paths_input = input("\n분석할 문서 경로(세미콜론 ';'로 구분): ").strip()
    if not file_paths_input:
        print("문서 경로가 입력되지 않았습니다. 종료.")
        return

    file_paths = [p.strip() for p in file_paths_input.split(";")]
    results = analyzer.process_documents(file_paths)

    print("\n=== 분석 결과 ===\n")
    for r in results:
        if "error" in r:
            print(f"[에러 발생] {r['file_path']}: {r['error']}\n")
            continue

        finfo = r["file_info"]
        doc_len = r["doc_length"]
        summary_text = r["map_reduce_summary"]
        analysis_json = r["analysis_result"]
        text_preview = r["text_content"]

        print(f"--- 파일: {finfo.get('file_name')} (길이: {doc_len} chars) ---")
        print(f"    크기(byte): {finfo.get('file_size')}, 생성일: {finfo.get('created_time')}, 수정일: {finfo.get('modified_time')}")

        # 첫 페이지(슬라이드) 미리보기
        if text_preview["pages"]:
            first_page = text_preview["pages"][0]
            print("\n[본문 프리뷰]")
            print(f" page_index={first_page['page_index']}, text={first_page['text_preview']}")
        else:
            print("\n[본문 없음]")

        # SummarizeChain / direct 요약 결과
        print("\n(1) 최종 요약 결과:")
        if summary_text.startswith("[map_reduce 요약 실패") or summary_text.startswith("[direct_llm_summary 실패"):
            print(summary_text)
        else:
            print(summary_text[:300] + ("..." if len(summary_text) > 300 else ""))

        # JSON(제목/발행처/목적/긴요약/캡션)
        print("\n(2) JSON 분석 결과:")
        if "error" in analysis_json:
            print("   [오류]", analysis_json["error"])
        else:
            print(json.dumps(analysis_json, ensure_ascii=False, indent=2))

        print("-" * 60)

    print("\n=== 프로그램 종료 ===")


if __name__ == "__main__":
    main()
