import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging

from file_handler import FileHandler
from parallel_processor import ParallelProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_markdown(file_type: str, result: Dict[str, Any], output_dir: str):
    """분석 결과를 마크다운 파일로 저장
    
    Args:
        file_type (str): 파일 유형 (image/code/document)
        result (Dict[str, Any]): 분석 결과
        output_dir (str): 출력 디렉토리 경로
    """
    try:
        # 파일명 추출
        file_name = result.get('file_info', {}).get('file_name', 'unknown')
        if file_name == 'unknown':
            file_path = result.get('file_info', {}).get('file_path', '')
            if file_path:
                file_name = Path(file_path).name
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir) / file_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 마크다운 파일 생성
        md_path = output_path / f"{file_name.split('.')[0]}.md"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            # 헤더 정보
            f.write(f"# {file_name}\n\n")
            f.write(f"## 기본 정보\n")
            f.write(f"- 파일 유형: {file_type}\n")
            f.write(f"- 카테고리: {result.get('category', 'N/A')}\n")
            f.write(f"- 처리 시간: {result.get('processed_at', 'N/A')}\n\n")
            
            # 파일 유형별 상세 정보
            if file_type == 'image':
                metadata = result.get('metadata', {})
                f.write("## 이미지 메타데이터\n")
                f.write(f"- 촬영 시간: {metadata.get('DateTimeOriginal', 'N/A')}\n")
                gps_info = metadata.get('GPSInfo', {})
                if 'coordinates' in gps_info:
                    f.write(f"- 위치: {gps_info.get('address', 'N/A')}\n")
                f.write(f"\n## 이미지 분석\n")
                for model, caption in result.get('captions', {}).items():
                    f.write(f"\n### {model} 분석 결과\n{caption}\n")
                
            elif file_type == 'code':
                f.write("## 코드 분석\n")
                f.write(f"- 프로그래밍 언어: {result.get('file_info', {}).get('language', 'N/A')}\n\n")
                f.write("### 임포트 정보\n")
                for imp in result.get('code_info', {}).get('imports', []):
                    f.write(f"- {imp['name']} (버전: {imp['version']}, 유형: {imp['type']})\n")
                analysis = result.get('code_info', {}).get('analysis', {})
                f.write("\n### 코드 목적\n")
                f.write(f"{analysis.get('purpose', 'N/A')}\n")
                f.write("\n### 주요 로직\n")
                f.write(f"{analysis.get('logic', 'N/A')}\n")
                f.write("\n### 데이터 흐름도\n")
                f.write("```mermaid\n")
                f.write(analysis.get('flowchart', 'graph TD\nA[분석 실패]'))
                f.write("\n```\n")
                
            elif file_type == 'document':
                # 기본 정보
                f.write("## 문서 분석\n")
                f.write(f"- 문서 길이: {result.get('doc_length', 'N/A')} 문자\n\n")
                
                # 전체 텍스트 내용
                f.write("## 원본 텍스트\n")
                f.write("```\n")
                f.write(result.get('final_text_or_summary', 'N/A'))
                f.write("\n```\n\n")
                
                # 표 구조
                if 'tables' in result and result['tables']:
                    f.write("## 표 구조\n")
                    for table in result['tables']:
                        f.write(f"\n### {table.get('title', '표')}\n")
                        # 헤더 작성
                        headers = table.get('headers', [])
                        if headers:
                            f.write('| ' + ' | '.join(headers) + ' |\n')
                            f.write('|' + '---|' * len(headers) + '\n')
                            # 데이터 행 작성
                            for row in table.get('rows', []):
                                f.write('| ' + ' | '.join(row) + ' |\n')
                        f.write('\n')
                
                # 페이지별 미리보기
                f.write("## 페이지 구성\n")
                for page in result.get('text_content', {}).get('pages', []):
                    f.write(f"### 페이지 {page.get('page_index', 'N/A')}\n")
                    f.write("```\n")
                    f.write(page.get('text_preview', 'N/A'))
                    f.write("\n```\n\n")
                
                # LLM 분석 결과
                f.write("## LLM 분석 결과\n")
                analysis = result.get('analysis_result', {})
                f.write(f"### 제목\n{analysis.get('title', 'N/A')}\n\n")
                f.write(f"### 작성자\n{analysis.get('author', 'N/A')}\n\n")
                f.write(f"### 목적\n{analysis.get('purpose', 'N/A')}\n\n")
                f.write(f"### 요약\n{analysis.get('summary', 'N/A')}\n\n")
                f.write(f"### 핵심 내용\n{analysis.get('caption', 'N/A')}\n")
            
        logger.info(f"마크다운 파일 생성 완료: {md_path}")
        
    except Exception as e:
        logger.error(f"마크다운 파일 생성 실패: {str(e)}")
        
def table_to_markdown(table: Dict) -> str:
    """표 데이터를 마크다운 형식으로 변환
    
    Args:
        table (Dict): 표 데이터
        {
            'title': str,
            'headers': List[str],
            'rows': List[List[str]]
        }
    
    Returns:
        str: 마크다운 형식의 표
    """
    md = []
    
    # 표 제목
    if table.get('title'):
        md.append(f"\n### {table['title']}\n")
    
    # 헤더 행
    headers = table.get('headers', [])
    if headers:
        md.append('| ' + ' | '.join(headers) + ' |')
        md.append('|' + '---|' * len(headers))
        
        # 데이터 행
        for row in table.get('rows', []):
            md.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    
    md.append('\n')
    return '\n'.join(md)

def format_section(section_name: str, content: Any) -> str:
    """섹션 포맷팅 헬퍼 함수
    
    Args:
        section_name (str): 섹션 이름
        content (Any): 섹션 내용
    
    Returns:
        str: 포맷팅된 마크다운 섹션
    """
    if not content:
        return ""
        
    result = [f"## {section_name}\n"]
    
    if isinstance(content, str):
        result.append(content + "\n")
    elif isinstance(content, dict):
        for key, value in content.items():
            result.append(f"- {key}: {value}\n")
    elif isinstance(content, list):
        for item in content:
            result.append(f"- {item}\n")
            
    result.append("\n")
    return "".join(result)

def main():
    """테스트 실행 함수"""
    try:
        # 입력 받기
        root_path = input("처리할 폴더 경로를 입력하세요: ").strip()
        if not os.path.exists(root_path):
            print("유효하지 않은 경로입니다.")
            return
            
        model_name = input("사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
        if model_name not in ['openai', 'gemini', 'claude', 'groq']:
            print("유효하지 않은 모델명입니다.")
            return
        
        # 결과 저장 디렉토리 설정
        output_dir = Path(root_path) / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        
        # 파일 분류
        file_handler = FileHandler()
        classified_files = file_handler.get_files(root_path)
        
        # 병렬 처리
        processor = ParallelProcessor(model_name)
        results = processor.process_files(classified_files)
        
        # 결과 출력 및 마크다운 생성
        print("\n처리 결과:")
        for file_type, type_results in results.items():
            success = len([r for r in type_results if r is not None])
            total = len(type_results)
            print(f"\n{file_type.upper()} 파일 처리 결과: {success}/{total} 성공")
            
            # 각 파일별 마크다운 생성
            for result in type_results:
                if result:
                    create_markdown(file_type, result, str(output_dir))
        
        print(f"\n분석 결과가 저장된 경로: {output_dir}")
        
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()