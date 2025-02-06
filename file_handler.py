import os
from typing import Dict, List
import logging
from pathlib import Path
from config import FILE_EXTENSION

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileHandler:
    """파일 처리 및 분류를 담당하는 클래스"""
    
    def __init__(self):
        # 파일 확장자별 타입 매핑
        self.type_mapping = FILE_EXTENSION
    
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
                'document': [{'path': str, 'category': str}, ...]
            }
        """
        if not os.path.exists(root_path):
            logger.error(f"존재하지 않는 경로입니다: {root_path}")
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {root_path}")
            
        classified_files = {
            'image': [],
            'code': [],
            'document': []
        }
        
        try:
            logger.info(f"파일 검색 시작: {root_path}")
            for root, _, files in os.walk(root_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 숨김 파일 제외
                    if file.startswith('.'):
                        continue
                        
                    extension = Path(file).suffix.lower()
                    
                    # reference 폴더 체크
                    is_reference = 'reference' in Path(file_path).parts
                    
                    if extension in self.type_mapping:
                        file_info = {
                            'path': file_path,
                            'category': 'reference' if is_reference else 'research'
                        }
                        file_type = self.type_mapping[extension]
                        classified_files[file_type].append(file_info)
                        logger.debug(f"파일 분류됨: {file_path} -> {file_type}")
            
            # 분류 결과 로깅
            for file_type, files in classified_files.items():
                logger.info(f"{file_type} 파일 수: {len(files)}")
                
            return classified_files
            
        except Exception as e:
            logger.error(f"파일 분류 중 오류 발생: {str(e)}")
            raise

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
        return os.path.exists(path)

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
                'type': str,          # 파일 유형
                'category': str       # 연구/참고자료 구분
            }
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        return {
            'name': path.name,
            'extension': extension,
            'type': self.type_mapping.get(extension, 'unknown'),
            'category': 'reference' if 'reference' in path.parts else 'research'
        }

if __name__ == "__main__":
    # 테스트 코드
    try:
        handler = FileHandler()
        
        # 경로 입력 받기
        root_path = input("처리할 폴더 경로를 입력하세요: ").strip()
        
        # 파일 분류 실행
        result = handler.get_files(root_path)
        
        # 결과 출력
        print("\n분류 결과:")
        for file_type, files in result.items():
            print(f"\n{file_type.upper()} 파일 ({len(files)}개):")
            for file_info in files:
                print(f"- 경로: {file_info['path']}")
                print(f"  카테고리: {file_info['category']}")
                
    except Exception as e:
        print(f"오류 발생: {str(e)}")