import os
from typing import Dict, List
import logging
from pathlib import Path
import re
from config import FILE_EXTENSION

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileHandler:
    """파일 처리 및 분류를 담당하는 클래스"""
    
    def __init__(self, diary_pattern: str = r'\d{6}_summary\.txt'):
        """
        FileHandler 초기화
        
        Args:
            diary_pattern (str, optional): 일기 파일 인식을 위한 정규 표현식 패턴.
                                          기본값은 'YYMMDD_summary.txt' 형식.
        """
        # 파일 확장자별 타입 매핑
        self.type_mapping = FILE_EXTENSION
        # 일기 파일 패턴 - 사용자 정의 가능
        self.diary_pattern = diary_pattern

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
        if not os.path.exists(root_path):
            # 경로가 없으면 생성
            os.makedirs(root_path, exist_ok=True)
            logger.info(f"입력 디렉토리를 생성했습니다: {root_path}")
            
        classified_files = {
            'image': [],
            'code': [],
            'document': [],
            'diary': []
        }
        
        try:
            logger.info(f"파일 검색 시작: {root_path}")
            file_counts = {'total': 0, 'classified': 0}
            
            for root, _, files in os.walk(root_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_counts['total'] += 1
                    
                    # 숨김 파일 제외
                    if file.startswith('.'):
                        continue
                    
                    # 일기 파일 확인 (설정된 패턴 사용)
                    if re.match(self.diary_pattern, file):
                        category = self._get_category(file_path)
                        file_info = {
                            'path': file_path,
                            'category': category
                        }
                        classified_files['diary'].append(file_info)
                        logger.debug(f"파일 분류됨: {file_path} -> diary ({category})")
                        file_counts['classified'] += 1
                        continue
                    
                    # 확장자 기반 파일 분류
                    extension = Path(file).suffix.lower()
                    if extension in self.type_mapping:
                        file_type = self.type_mapping[extension]
                        category = self._get_category(file_path)
                        file_info = {
                            'path': file_path,
                            'category': category
                        }
                        classified_files[file_type].append(file_info)
                        logger.debug(f"파일 분류됨: {file_path} -> {file_type} ({category})")
                        file_counts['classified'] += 1
            
            # 분류 결과 로깅
            logger.info(f"총 파일 수: {file_counts['total']}, 분류된 파일 수: {file_counts['classified']}")
            for file_type, files in classified_files.items():
                logger.info(f"{file_type} 파일 수: {len(files)}")
                
            return classified_files
            
        except Exception as e:
            logger.error(f"파일 분류 중 오류 발생: {str(e)}")
            raise

    def _get_category(self, file_path: str) -> str:
        """
        파일 경로를 기반으로 카테고리 결정
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            str: 카테고리 ('reference' 또는 'research')
        """
        path = Path(file_path)
        
        # 정확한 'reference' 폴더 인식
        for parent in path.parents:
            if parent.name.lower() == 'reference':
                return 'reference'
        
        # 그 외는 모두 연구자료로 분류
        return 'research'

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
                'type': str,          # 파일 유형 (image/code/document/diary)
                'category': str       # 연구/참고자료 구분
            }
        """
        path = Path(file_path)
        file_name = path.name
        extension = path.suffix.lower()
        
        # 파일 유형 결정
        if re.match(self.diary_pattern, file_name):
            file_type = 'diary'
        else:
            file_type = self.type_mapping.get(extension, 'unknown')
        
        return {
            'name': file_name,
            'extension': extension,
            'type': file_type,
            'category': self._get_category(file_path)
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
        total_files = sum(len(files) for files in result.values())
        print(f"총 {total_files}개 파일이 분류되었습니다.")
        
        for file_type, files in result.items():
            print(f"\n{file_type.upper()} 파일 ({len(files)}개):")
            
            # 카테고리별 분류
            categories = {}
            for file_info in files:
                category = file_info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(file_info['path'])
            
            # 카테고리별 출력
            for category, paths in categories.items():
                print(f"  {category.upper()} ({len(paths)}개):")
                for path in paths[:5]:  # 최대 5개만 출력
                    print(f"    - {os.path.basename(path)}")
                if len(paths) > 5:
                    print(f"    ... 외 {len(paths) - 5}개")
                
    except Exception as e:
        print(f"오류 발생: {str(e)}")