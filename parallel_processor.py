# parallel_processor.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
from pathlib import Path
import traceback

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from processor_factory import ProcessorFactory
from file_type_detector import FileTypeDetector

class ParallelProcessor:
    """분류된 파일들의 병렬 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "claude", max_workers: int = None):
        """
        ParallelProcessor 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명 (openai/gemini/claude/groq)
            max_workers (int, optional): 최대 스레드 수 (None이면 CPU 코어 수 * 5)
        """
        self.model_name = model_name
        self.max_workers = max_workers
        
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("parallel_processor")
        self.error_handler = ErrorHandler.get_instance()
        
        # 프로세서 팩토리
        self.processor_factory = ProcessorFactory.get_instance()
        
        # 파일 유형 감지기
        self.type_detector = FileTypeDetector.get_instance()
        
        # 프로세서 초기화
        self.logger.info(f"{model_name} 모델로 프로세서 초기화 중...")
        try:
            self.processors = self.processor_factory.get_all_processors(model_name)
            self.logger.info("프로세서 초기화 완료")
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"model_name": model_name, "operation": "processor_initialization"}
            )
            self.logger.error(f"프로세서 초기화 실패: {error_detail['error']}")
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
        
        # 처리 시작 시간
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 파일 유형에 대한 처리 Future 리스트
            futures = []
            
            # 이미지 처리
            if classified_files.get('image', []):
                futures.extend(self._submit_tasks(executor, 'image', classified_files['image']))
            
            # 코드 처리
            if classified_files.get('code', []):
                futures.extend(self._submit_tasks(executor, 'code', classified_files['code']))
            
            # 문서 처리
            if classified_files.get('document', []):
                futures.extend(self._submit_tasks(executor, 'document', classified_files['document']))
            
            # 일기 처리
            if classified_files.get('diary', []):
                futures.extend(self._submit_tasks(executor, 'diary', classified_files['diary']))
            
            # 모든 Future 완료 대기 및 결과 수집
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        file_type, processed_result = result
                        
                        # 특수 케이스: 일기 처리 결과는 딕셔너리 병합
                        if file_type == 'diary' and isinstance(processed_result, dict) and 'error' not in processed_result:
                            raw_results['diary'].update(processed_result)
                        # 일반적인 경우: 리스트에 결과 추가
                        elif file_type in raw_results and file_type != 'diary':
                            raw_results[file_type].append(processed_result)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.logger.error(f"작업 완료 처리 중 오류 발생: {str(e)}\n{tb}")
        
        # 결과 표준화
        standardized_results = self._standardize_results(raw_results)
        
        # 처리 완료 로깅
        elapsed_time = time.time() - start_time
        total_processed = (
            len(standardized_results['image']) + 
            len(standardized_results['code']) + 
            len(standardized_results['document']) + 
            len(standardized_results['diary_entries'])
        )
        
        self.logger.info(f"파일 처리 및 결과 표준화 완료: {total_processed}개 항목, {elapsed_time:.2f}초 소요")
        self.logger.info(f"이미지: {len(standardized_results['image'])}개")
        self.logger.info(f"코드: {len(standardized_results['code'])}개")
        self.logger.info(f"문서: {len(standardized_results['document'])}개")
        self.logger.info(f"일기: {len(standardized_results['diary_entries'])}개")
        
        return standardized_results

    def _submit_tasks(self, executor, file_type: str, files: List[Dict[str, str]]) -> List:
        """
        특정 파일 유형에 대한 모든 태스크 제출
        
        Args:
            executor: ThreadPoolExecutor 인스턴스
            file_type (str): 파일 유형
            files (List[Dict[str, str]]): 처리할 파일 목록
            
        Returns:
            List: 제출된 Future 객체 리스트
        """
        futures = []
        
        self.logger.info(f"{file_type} 파일 처리 시작: {len(files)}개")
        
        # 프로세서 가져오기
        processor = self.processors.get(file_type)
        if not processor:
            self.logger.error(f"{file_type} 프로세서를 찾을 수 없습니다.")
            return futures
        
        # 각 파일에 대한 태스크 제출
        for file_info in files:
            file_path = file_info['path']
            category = file_info['category']
            
            self.logger.debug(f"{file_type} 파일 처리 대기 중: {file_path} (카테고리: {category})")
            
            # 처리 방법 선택
            if file_type == 'diary':
                future = executor.submit(self._process_diary, file_info)
            else:
                future = executor.submit(self._process_file, file_type, file_info)
                
            futures.append(future)
            
        return futures

    def _process_file(self, file_type: str, file_info: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
        """
        일반 파일 처리 (이미지, 코드, 문서)
        
        Args:
            file_type (str): 파일 유형
            file_info (Dict[str, str]): 파일 정보
            
        Returns:
            Tuple[str, Dict[str, Any]]: (파일 유형, 처리 결과)
        """
        file_path = file_info['path']
        category = file_info['category']
        
        try:
            processor = self.processors.get(file_type)
            if not processor:
                error_msg = f"{file_type} 프로세서를 찾을 수 없습니다."
                self.logger.error(error_msg)
                return file_type, {'error': error_msg, 'file_path': file_path}
            
            result = processor.process_file(file_path, category)
            
            # 카테고리 정보 로깅
            self.logger.debug(f"{file_type} 파일 처리 완료: {file_path} (카테고리: {category})")
            
            return file_type, result
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "file_type": file_type, "category": category}
            )
            self.logger.error(f"{file_type} 처리 오류: {error_detail['error']}")
            
            return file_type, {
                'error': str(e),
                'file_path': file_path,
                'category': category
            }

    def _process_diary(self, file_info: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        """
        일기 파일 처리
        
        Args:
            file_info (Dict[str, str]): 파일 정보
            
        Returns:
            Tuple[str, Dict[str, str]]: ('diary', {날짜: 내용} 형태의 딕셔너리)
        """
        file_path = file_info['path']
        category = file_info['category']
        
        try:
            processor = self.processors.get('diary')
            if not processor:
                error_msg = "일기 프로세서를 찾을 수 없습니다."
                self.logger.error(error_msg)
                return 'diary', {'error': error_msg}
            
            # DiaryProcessor를 통해 일기 처리 (날짜-내용 매핑 반환)
            result = processor.process_diary(file_path)
            self.logger.debug(f"일기 파일 처리 완료: {file_path}")
            
            return 'diary', result
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "file_type": "diary", "category": category}
            )
            self.logger.error(f"일기 처리 오류: {error_detail['error']}")
            
            return 'diary', {"error": str(e)}

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
        
        try:
            # 이미지 결과 표준화
            standardized['image'] = self._standardize_image_results(raw_results['image'])
            
            # 코드 결과 표준화
            standardized['code'] = self._standardize_code_results(raw_results['code'])
            
            # 문서 결과 표준화
            standardized['document'] = self._standardize_document_results(raw_results['document'])
            
            # 일기 결과 표준화 (딕셔너리에서 정렬된 리스트로 변환)
            standardized['diary_entries'] = self._standardize_diary_results(raw_results['diary'])
            
            return standardized
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"operation": "result_standardization"}
            )
            self.logger.error(f"결과 표준화 중 오류 발생: {error_detail['error']}")
            
            # 오류 발생 시 빈 결과 반환
            return standardized

    def _standardize_image_results(self, image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """이미지 결과 표준화"""
        standardized_images = []
        
        for img_result in image_results:
            if not img_result or "error" in img_result:
                continue
            
            # 메타데이터에서 날짜와 위치 정보 추출
            metadata = img_result.get('metadata', {})
            date_time = metadata.get('DateTimeOriginal', 'N/A')
            location = metadata.get('GPSInfo', {}).get('address', 'N/A')
            
            # 날짜 형식 표준화 시도
            formatted_date = self._standardize_date(date_time)
            
            # 캡션 중 사용 가능한 것 선택
            caption = self._extract_best_caption(img_result.get('captions', {}))
            
            # 표준화된 이미지 결과
            standardized_img = {
                'file_info': img_result.get('file_info', {}),
                'metadata': metadata,
                'captions': img_result.get('captions', {}),
                'category': img_result.get('category', 'Unknown'),
                'processed_at': img_result.get('processed_at', datetime.now().isoformat()),
                'formatted_date': formatted_date,
                'location': location,
                'caption': caption
            }
            
            standardized_images.append(standardized_img)
            
        return standardized_images

    def _standardize_code_results(self, code_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """코드 결과 표준화"""
        standardized_codes = []
        
        for code_result in code_results:
            if not code_result or "error" in code_result:
                continue
            
            file_info = code_result.get('file_info', {})
            code_info = code_result.get('code_info', {})
            analysis = code_info.get('analysis', {})
            
            # 표준화된 코드 결과
            standardized_code = {
                'file_info': file_info,
                'code_info': code_info,
                'category': code_result.get('category', 'Unknown'),
                'processed_at': code_result.get('processed_at', datetime.now().isoformat()),
                'language': file_info.get('language', 'Unknown'),
                'purpose': analysis.get('purpose', ''),
                'logic': analysis.get('logic', '')
            }
            
            standardized_codes.append(standardized_code)
            
        return standardized_codes

    def _standardize_document_results(self, document_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 결과 표준화"""
        standardized_docs = []
        
        for doc_result in document_results:
            if not doc_result or "error" in doc_result:
                continue
            
            file_info = doc_result.get('file_info', {})
            analysis_result = doc_result.get('analysis_result', {})
            
            # 표준화된 문서 결과
            standardized_doc = {
                'file_info': file_info,
                'analysis_result': analysis_result,
                'category': doc_result.get('category', 'Unknown'),
                'processed_at': doc_result.get('processed_at', datetime.now().isoformat()),
                'title': analysis_result.get('title', file_info.get('file_name', 'Unknown')),
                'summary': analysis_result.get('summary', ''),
                'purpose': analysis_result.get('purpose', '')
            }
            
            standardized_docs.append(standardized_doc)
            
        return standardized_docs

    def _standardize_diary_results(self, diary_dict: Dict[str, str]) -> List[Dict[str, Any]]:
        """일기 결과 표준화 (딕셔너리에서 리스트로 변환)"""
        sorted_entries = []
        
        for date, content in diary_dict.items():
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
        
        return sorted_entries

    def _standardize_date(self, date_str: str) -> Optional[str]:
        """날짜 문자열 표준화"""
        if not date_str or date_str == 'N/A':
            return None
            
        try:
            # YYYY:MM:DD HH:MM:SS 형식을 YYYY-MM-DD로 변환
            date_part = date_str.split()[0].replace(":", "-")
            return date_part
        except (IndexError, AttributeError):
            return None

    def _extract_best_caption(self, captions: Dict[str, str]) -> str:
        """여러 캡션 중 가장 적합한 것 선택"""
        if not captions:
            return ''
            
        # 우선 순위에 따른 모델 선택
        priority_models = ['claude', 'gemini', 'openai', 'groq']
        
        for model in priority_models:
            if model in captions and captions[model] and not captions[model].startswith('⚠️'):
                return captions[model]
        
        # 우선 순위 모델이 없으면 첫 번째 유효한 캡션 사용
        for caption_text in captions.values():
            if caption_text and not caption_text.startswith('⚠️'):
                return caption_text
                
        return next(iter(captions.values()), '')

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
        if 'diary' in self.processors and hasattr(self.processors['diary'], 'get_all_diaries'):
            return self.processors['diary'].get_all_diaries()
        return {}
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        단일 파일 처리 (유형 자동 감지)
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 파일 유형 및 카테고리 감지
            file_info = self.type_detector.detect_file_type(file_path)
            file_type = file_info['type']
            category = file_info['category']
            
            # 지원되는 유형인지 확인
            if file_type not in ['image', 'code', 'document', 'diary']:
                error_msg = f"지원되지 않는 파일 유형: {file_type}"
                self.logger.error(error_msg)
                return {'error': error_msg, 'file_path': file_path}
            
            # 적절한 프로세서로 처리
            processor = self.processors.get(file_type)
            if not processor:
                error_msg = f"{file_type} 프로세서를 찾을 수 없습니다."
                self.logger.error(error_msg)
                return {'error': error_msg, 'file_path': file_path}
            
            # 처리 수행
            if file_type == 'diary':
                # 일기는 특별 처리
                diary_dict = processor.process_diary(file_path)
                
                # 결과를 단일 항목 리스트로 변환
                diary_entries = []
                for date, content in diary_dict.items():
                    if date != 'error':
                        formatted_date = self._format_date(date)
                        diary_entries.append({
                            'date': formatted_date,
                            'content': content,
                            'original_date': date,
                            'category': 'research'
                        })
                
                # 에러 체크
                if 'error' in diary_dict:
                    return {'error': diary_dict['error'], 'file_path': file_path}
                    
                return {'diary_entries': diary_entries}
            else:
                # 일반 파일 처리
                result = processor.process_file(file_path, category)
                
                # 결과 표준화
                if file_type == 'image':
                    std_result = self._standardize_image_results([result])
                    return {'image': std_result} if std_result else {'error': '이미지 표준화 실패', 'file_path': file_path}
                elif file_type == 'code':
                    std_result = self._standardize_code_results([result])
                    return {'code': std_result} if std_result else {'error': '코드 표준화 실패', 'file_path': file_path}
                elif file_type == 'document':
                    std_result = self._standardize_document_results([result])
                    return {'document': std_result} if std_result else {'error': '문서 표준화 실패', 'file_path': file_path}
                
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "single_file_processing"}
            )
            self.logger.error(f"단일 파일 처리 중 오류 발생: {error_detail['error']}")
            
            return {'error': str(e), 'file_path': file_path}