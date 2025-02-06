from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import logging
from datetime import datetime

from ImageProcessor import ImageAnalyzer
from CodeProcessor import CodeAnalyzer
from DocumentProcessor import DocumentAnalyzer
from file_handler import FileHandler
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelProcessor:
    """분류된 파일들의 병렬 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "claude", max_workers: int = None):
        """
        Args:
            model_name (str): 사용할 LLM 모델명 (openai/gemini/claude/groq)
            max_workers (int, optional): 최대 스레드 수
        """
        self.model_name = model_name
        self.max_workers = max_workers
        
        # 프로세서 초기화
        logger.info(f"{model_name} 모델로 프로세서 초기화 중...")
        try:
            self.image_processor = ImageAnalyzer(model_name)
            self.code_processor = CodeAnalyzer(model_name)
            self.doc_processor = DocumentAnalyzer(model_name)
            logger.info("프로세서 초기화 완료")
        except Exception as e:
            logger.error(f"프로세서 초기화 실패: {str(e)}")
            raise
    
    def process_files(self, classified_files: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        분류된 파일들을 병렬로 처리

        Args:
            classified_files: FileHandler로부터 받은 분류된 파일 목록
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 각 파일 유형별 처리 결과
            {
                'image': [처리결과1, 처리결과2, ...],
                'code': [처리결과1, 처리결과2, ...],
                'document': [처리결과1, 처리결과2, ...]
            }
        """
        results = {
            'image': [],
            'code': [],
            'document': []
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 이미지 처리
            if classified_files['image']:
                logger.info(f"이미지 파일 처리 시작: {len(classified_files['image'])}개")
                futures = []
                for file_info in classified_files['image']:
                    future = executor.submit(self._process_image, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results['image'].append(result)
                    except Exception as e:
                        logger.error(f"이미지 처리 실패: {str(e)}")
            
            # 코드 처리
            if classified_files['code']:
                logger.info(f"코드 파일 처리 시작: {len(classified_files['code'])}개")
                futures = []
                for file_info in classified_files['code']:
                    future = executor.submit(self._process_code, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results['code'].append(result)
                    except Exception as e:
                        logger.error(f"코드 처리 실패: {str(e)}")
            
            # 문서 처리
            if classified_files['document']:
                logger.info(f"문서 파일 처리 시작: {len(classified_files['document'])}개")
                futures = []
                for file_info in classified_files['document']:
                    future = executor.submit(self._process_document, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results['document'].append(result)
                    except Exception as e:
                        logger.error(f"문서 처리 실패: {str(e)}")
        
        return results

    def _process_image(self, file_info: Dict[str, str]) -> Dict[str, Any]:
        """이미지 파일 처리"""
        try:
            result = self.image_processor._process_single_image(file_info['path'])
            if result:
                result['category'] = file_info['category']
                result['processed_at'] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"이미지 처리 오류 ({file_info['path']}): {str(e)}")
            return None

    def _process_code(self, file_info: Dict[str, str]) -> Dict[str, Any]:
        """코드 파일 처리"""
        try:
            result = self.code_processor.process_code(file_info['path'])
            if result:
                result['category'] = file_info['category']
                result['processed_at'] = datetime.now().isoformat()
            return result
        except Exception as e:
            logger.error(f"코드 처리 오류 ({file_info['path']}): {str(e)}")
            return None

    def _process_document(self, file_info: Dict[str, str]) -> Dict[str, Any]:
            """문서 파일 처리"""
            try:
                # DocumentProcessor의 모든 return 값을 유지
                result = self.doc_processor.process_single_document(file_info['path'])
                if result:
                    result['category'] = file_info['category']
                    result['processed_at'] = datetime.now().isoformat()
                    # file_info가 없는 경우 추가
                    if 'file_info' not in result:
                        result['file_info'] = {
                            'file_name': Path(file_info['path']).name,
                            'file_path': file_info['path']
                        }
                return result
            except Exception as e:
                logger.error(f"문서 처리 오류 ({file_info['path']}): {str(e)}")
                return None

if __name__ == "__main__":
    # 테스트 코드
    try:
        # 파일 경로 입력
        root_path = input("처리할 폴더 경로를 입력하세요: ").strip()
        model_name = input("사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
        
        # 파일 분류
        file_handler = FileHandler()
        classified_files = file_handler.get_files(root_path)
        
        # 병렬 처리
        processor = ParallelProcessor(model_name)
        results = processor.process_files(classified_files)
        
        # 결과 출력
        print("\n처리 결과:")
        for file_type, type_results in results.items():
            success = len([r for r in type_results if r is not None])
            total = len(type_results)
            print(f"\n{file_type.upper()} 파일 처리 결과: {success}/{total} 성공")
            
            if type_results:
                print("\n처리된 파일:")
                for result in type_results:
                    if result:
                        print(f"- 파일: {result.get('file_info', {}).get('file_name', 'N/A')}")
                        print(f"  카테고리: {result.get('category', 'N/A')}")
                        print(f"  처리 시간: {result.get('processed_at', 'N/A')}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")