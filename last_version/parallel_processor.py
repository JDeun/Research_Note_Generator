from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ImageProcessor import ImageAnalyzer
from CodeProcessor import CodeAnalyzer
from DocumentProcessor import DocumentAnalyzer
from diary_processor import DiaryProcessor
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
            self.diary_processor = DiaryProcessor()
            logger.info("프로세서 초기화 완료")
        except Exception as e:
            logger.error(f"프로세서 초기화 실패: {str(e)}")
            raise

    def process_files(self, classified_files: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
        """
        분류된 파일들을 병렬로 처리하고 결과를 표준화된 형식으로 반환
        
        Args:
            classified_files (Dict[str, List[Dict[str, str]]]): 분류된 파일 정보
                
        Returns:
            Dict[str, Any]: 표준화된 처리 결과
        """
        raw_results = {
            'image': [],
            'code': [],
            'document': [],
            'diary': {}  # 원본 날짜-내용 매핑을 위한 딕셔너리
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 이미지 처리
            if classified_files.get('image', []):
                logger.info(f"이미지 파일 처리 시작: {len(classified_files['image'])}개")
                futures = []
                for file_info in classified_files['image']:
                    logger.debug(f"이미지 파일 처리 대기 중: {file_info['path']} (카테고리: {file_info['category']})")
                    future = executor.submit(self._process_image, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            # 카테고리 정보 로깅
                            file_path = result.get('file_info', {}).get('file_path', 'unknown')
                            category = result.get('category', 'unknown')
                            logger.debug(f"이미지 파일 처리 완료: {file_path} (카테고리: {category})")
                            raw_results['image'].append(result)
                    except Exception as e:
                        logger.error(f"이미지 처리 실패: {str(e)}")
            
            # 코드 처리
            if classified_files.get('code', []):
                logger.info(f"코드 파일 처리 시작: {len(classified_files['code'])}개")
                futures = []
                for file_info in classified_files['code']:
                    logger.debug(f"코드 파일 처리 대기 중: {file_info['path']} (카테고리: {file_info['category']})")
                    future = executor.submit(self._process_code, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            # 카테고리 정보 로깅
                            file_path = result.get('file_info', {}).get('file_path', 'unknown')
                            category = result.get('category', 'unknown')
                            logger.debug(f"코드 파일 처리 완료: {file_path} (카테고리: {category})")
                            raw_results['code'].append(result)
                    except Exception as e:
                        logger.error(f"코드 처리 실패: {str(e)}")
            
            # 문서 처리
            if classified_files.get('document', []):
                logger.info(f"문서 파일 처리 시작: {len(classified_files['document'])}개")
                futures = []
                for file_info in classified_files['document']:
                    logger.debug(f"문서 파일 처리 대기 중: {file_info['path']} (카테고리: {file_info['category']})")
                    future = executor.submit(self._process_document, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            # 카테고리 정보 로깅
                            file_path = result.get('file_info', {}).get('file_path', 'unknown')
                            category = result.get('category', 'unknown')
                            logger.debug(f"문서 파일 처리 완료: {file_path} (카테고리: {category})")
                            raw_results['document'].append(result)
                    except Exception as e:
                        logger.error(f"문서 처리 실패: {str(e)}")
            
            # 일기 처리
            if classified_files.get('diary', []):
                logger.info(f"일기 파일 처리 시작: {len(classified_files['diary'])}개")
                futures = []
                for file_info in classified_files['diary']:
                    logger.debug(f"일기 파일 처리 대기 중: {file_info['path']} (카테고리: {file_info['category']})")
                    future = executor.submit(self._process_diary, file_info)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        diary_result = future.result()
                        if diary_result and "error" not in diary_result:
                            # 날짜-내용 매핑을 raw_results['diary'] 딕셔너리에 병합
                            raw_results['diary'].update(diary_result)
                            logger.debug(f"일기 파일 처리 완료: {list(diary_result.keys())}")
                    except Exception as e:
                        logger.error(f"일기 처리 실패: {str(e)}")
        
        # 결과 표준화
        standardized_results = self._standardize_results(raw_results)
        
        # 최종 결과 카테고리 확인 로깅
        logger.info("파일 처리 및 결과 표준화 완료")
        logger.info(f"이미지: {len(standardized_results['image'])}개")
        for img in standardized_results['image']:
            file_name = img.get('file_info', {}).get('file_name', 'unknown')
            category = img.get('category', 'unknown')
            logger.debug(f"표준화된 이미지: {file_name} (카테고리: {category})")
        
        logger.info(f"코드: {len(standardized_results['code'])}개")
        for code in standardized_results['code']:
            file_name = code.get('file_info', {}).get('file_name', 'unknown')
            category = code.get('category', 'unknown')
            logger.debug(f"표준화된 코드: {file_name} (카테고리: {category})")
        
        logger.info(f"문서: {len(standardized_results['document'])}개")
        for doc in standardized_results['document']:
            file_name = doc.get('file_info', {}).get('file_name', 'unknown')
            category = doc.get('category', 'unknown')
            logger.debug(f"표준화된 문서: {file_name} (카테고리: {category})")
        
        logger.info(f"일기: {len(standardized_results['diary_entries'])}개")
        
        return standardized_results

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
            
    def _process_diary(self, file_info: Dict[str, str]) -> Dict[str, str]:
        """
        일기 파일 처리
        
        Args:
            file_info (Dict[str, str]): 파일 정보
            
        Returns:
            Dict[str, str]: {날짜: 내용} 형태의 딕셔너리
        """
        try:
            # DiaryProcessor를 통해 일기 처리 (날짜-내용 매핑 반환)
            result = self.diary_processor.process_diary(file_info['path'])
            return result
        except Exception as e:
            logger.error(f"일기 처리 오류 ({file_info['path']}): {str(e)}")
            return {"error": str(e)}

    def _standardize_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        처리 결과를 표준화된 형식으로 변환
        
        Args:
            raw_results (Dict[str, Any]): 원본 처리 결과
            
        Returns:
            Dict[str, Any]: 표준화된 처리 결과
        """
        standardized = {
            'image': [],
            'code': [],
            'document': [],
            'diary_entries': []  # 날짜-내용 딕셔너리에서 리스트 형식으로 변경
        }
        
        # 이미지 결과 표준화
        for img_result in raw_results['image']:
            if not img_result or "error" in img_result:
                continue
            
            # 메타데이터에서 날짜와 위치 정보 추출
            metadata = img_result.get('metadata', {})
            date_time = metadata.get('DateTimeOriginal', 'N/A')
            location = metadata.get('GPSInfo', {}).get('address', 'N/A')
            
            # 날짜 형식 표준화 시도
            formatted_date = None
            if date_time and date_time != 'N/A':
                try:
                    # YYYY:MM:DD HH:MM:SS 형식을 YYYY-MM-DD로 변환
                    date_part = date_time.split()[0].replace(":", "-")
                    formatted_date = date_part
                except (IndexError, AttributeError):
                    formatted_date = None
            
            # 캡션 중 사용 가능한 것 선택
            caption = ''
            captions = img_result.get('captions', {})
            for model, caption_text in captions.items():
                if caption_text and not caption_text.startswith('⚠️'):
                    caption = caption_text
                    break
            
            # 표준화된 이미지 결과
            standardized_img = {
                'file_info': img_result.get('file_info', {}),
                'metadata': metadata,
                'captions': captions,
                'category': img_result.get('category', 'Unknown'),  # 카테고리 정보 유지
                'processed_at': img_result.get('processed_at', datetime.now().isoformat()),
                'formatted_date': formatted_date,
                'location': location,
                'caption': caption  # 가장 적합한 캡션 추가
            }
            
            standardized['image'].append(standardized_img)
        
        # 코드 결과 표준화
        for code_result in raw_results['code']:
            if not code_result or "error" in code_result:
                continue
            
            file_info = code_result.get('file_info', {})
            code_info = code_result.get('code_info', {})
            analysis = code_info.get('analysis', {})
            
            # 표준화된 코드 결과
            standardized_code = {
                'file_info': file_info,
                'code_info': code_info,
                'category': code_result.get('category', 'Unknown'),  # 카테고리 정보 유지
                'processed_at': code_result.get('processed_at', datetime.now().isoformat()),
                'language': file_info.get('language', 'Unknown'),
                'purpose': analysis.get('purpose', ''),
                'logic': analysis.get('logic', '')
            }
            
            standardized['code'].append(standardized_code)
        
        # 문서 결과 표준화
        for doc_result in raw_results['document']:
            if not doc_result or "error" in doc_result:
                continue
            
            file_info = doc_result.get('file_info', {})
            analysis_result = doc_result.get('analysis_result', {})
            
            # 표준화된 문서 결과
            standardized_doc = {
                'file_info': file_info,
                'analysis_result': analysis_result,
                'category': doc_result.get('category', 'Unknown'),  # 카테고리 정보 유지
                'processed_at': doc_result.get('processed_at', datetime.now().isoformat()),
                'title': analysis_result.get('title', file_info.get('file_name', 'Unknown')),
                'summary': analysis_result.get('summary', ''),
                'purpose': analysis_result.get('purpose', '')
            }
            
            standardized['document'].append(standardized_doc)
        
        # 일기 결과 표준화 (딕셔너리에서 정렬된 리스트로 변환)
        sorted_entries = []
        for date, content in raw_results['diary'].items():
            # 날짜 형식 표준화
            formatted_date = self._format_date(date)
            
            sorted_entries.append({
                'date': formatted_date,         # 표준화된 날짜 (YYYY-MM-DD)
                'content': content,             # 일기 내용
                'original_date': date,          # 원본 날짜 형식 보존
                'category': 'research'          # 일기는 항상 연구 자료(직접 작성)로 분류
            })
        
        # 날짜 기준 정렬
        sorted_entries.sort(key=lambda x: x['date'])
        standardized['diary_entries'] = sorted_entries
        
        return standardized
    
    def _format_date(self, date_str: str) -> str:
        """
        다양한 날짜 형식을 YYYY-MM-DD 형식으로 표준화
        
        Args:
            date_str (str): 변환할 날짜 문자열
            
        Returns:
            str: 표준화된 날짜 문자열
        """
        # YYMMDD 형식 처리
        if len(date_str) == 6 and date_str.isdigit():
            year = int("20" + date_str[:2])  # 2000년대로 가정
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            return f"{year}-{month:02d}-{day:02d}"
        
        # 이미 YYYY-MM-DD 형식이면 그대로 반환
        elif len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
            return date_str
        
        # YYYY:MM:DD 형식 처리
        elif len(date_str) >= 10 and date_str[4] == ':' and date_str[7] == ':':
            return date_str[:10].replace(":", "-")
        
        # 기타 형식은 원본 반환
        return date_str

    def get_all_diary_entries(self) -> Dict[str, str]:
        """
        모든 일기 항목 반환
        
        Returns:
            Dict[str, str]: 모든 일기 항목 (날짜: 내용)
        """
        return self.diary_processor.get_all_diaries()

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
        
        # 이미지, 코드, 문서 결과 출력
        for file_type in ['image', 'code', 'document']:
            type_results = results[file_type]
            success = len([r for r in type_results if r is not None])
            total = len(type_results)
            print(f"\n{file_type.upper()} 파일 처리 결과: {success}/{total} 성공")
            
            if type_results:
                print("\n처리된 파일:")
                for result in type_results:
                    if result:
                        print(f"- 파일: {result.get('file_info', {}).get('file_name', 'N/A')}")
                        print(f"  카테고리: {result.get('category', 'N/A')}")
        
        # 일기 결과 출력 (수정된 부분)
        diary_entries = results['diary_entries']
        print(f"\nDIARY 파일 처리 결과: {len(diary_entries)} 항목")
        
        if diary_entries:
            print("\n처리된 일기:")
            for entry in diary_entries:
                date = entry.get('date', 'N/A')
                content = entry.get('content', '')
                preview = content[:50] + "..." if len(content) > 50 else content
                print(f"- 날짜: {date}")
                print(f"  내용: {preview}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")