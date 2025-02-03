# DocumentProcessor와 CodeProcessor를 테스트하는 코드입니다.

from DocumentProcessor import DocumentAnalyzer
from CodeProcessor import CodeAnalyzer

def main():
    # 사용할 모델을 설정 (예: "gemini")
    document_analyzer = DocumentAnalyzer("gemini")
    code_analyzer = CodeAnalyzer("gemini")
    
    # 테스트할 문서 경로를 입력받아 문서 로더 결과 확인
    doc_file_path = input("테스트할 문서의 경로를 입력하세요: ")
    docs = document_analyzer._load_document(doc_file_path)
    
    print(f"\n문서 로더가 반환한 Document 객체 수: {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Document {i} ---")
        # 추출된 텍스트의 앞부분 5000자를 출력합니다.
        print(doc.page_content[:5000])
    
    # 테스트할 코드 파일 경로를 입력받아 코드 분석 결과 확인
    code_file_path = input("\n테스트할 코드 파일의 경로를 입력하세요: ")
    # CodeAnalyzer는 process_code 메서드를 통해 분석 결과를 반환한다고 가정합니다.
    code_result = code_analyzer.process_code(code_file_path)
    
    print("\n--- 코드 분석 결과 ---")
    print(code_result)

if __name__ == "__main__":
    main()
