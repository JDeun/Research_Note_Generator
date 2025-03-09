# image_processor.py
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pathlib import Path
import base64
import io
import ssl
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from base_processor import BaseProcessor
from config import IMAGE_PROCESSOR_MODELS, TEMPERATURES, LLM_API_KEY
from ProcessorPrompt import IMAGE_PROCESSOR_PROMPT

class ImageProcessor(BaseProcessor):
    """이미지 파일 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "claude", auto_optimize: bool = True):
        """
        ImageProcessor 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명
            auto_optimize (bool): 자동 최적화 사용 여부
        """
        super().__init__(model_name, auto_optimize)
        
        # 이미지 분석기 초기화
        self.image_analyzer = ImageAnalyzer(model_name)
    
    def _setup_model(self) -> Any:
        """
        모델 설정 (BaseProcessor 추상 메서드 구현)
        
        Returns:
            Any: 초기화된 모델 인스턴스
        """
        if self.model_name not in IMAGE_PROCESSOR_MODELS:
            self.logger.error(f"지원하지 않는 모델: {self.model_name}")
            return None

        api_model, model_class = IMAGE_PROCESSOR_MODELS[self.model_name]
        api_key = LLM_API_KEY.get(self.model_name)

        if not api_key:
            self.logger.error(f"API 키 누락: {self.model_name} API 키를 .env에 설정하세요.")
            return None

        try:
            return model_class(api_key=api_key, model=api_model, temperature=TEMPERATURES["image"])
        except Exception as e:
            raise Exception(f"{self.model_name} 모델 초기화 실패: {str(e)}")
    
    def _process_file_internal(self, file_path: str) -> Dict[str, Any]:
        """
        이미지 파일 처리 내부 로직 (BaseProcessor 추상 메서드 구현)
        
        Args:
            file_path (str): 처리할 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        return self.image_analyzer.process_single_image(file_path)


class ImageAnalyzer:
    """이미지 분석 및 처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str):
        """
        ImageAnalyzer 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명
        """
        self.model_name = model_name.lower()
        self.llm = None
        
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("image_analyzer")
        self.error_handler = ErrorHandler.get_instance()
        
        # 프롬프트 설정
        self.system_prompt = IMAGE_PROCESSOR_PROMPT["system"]
        
        # 지오코더 및 LLM 모델 설정
        self._setup_geolocator()
        self._setup_llm()
    
    def _setup_geolocator(self):
        """지오코더 설정"""
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self.geolocator = Nominatim(user_agent="image_processor", ssl_context=ctx)
            self.logger.debug("지오코더 초기화 성공")
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"operation": "geolocator_setup"}
            )
            self.logger.error(f"지오코더 초기화 실패: {error_detail['error']}")
            self.geolocator = None
    
    def _setup_llm(self):
        """LLM 모델 설정"""
        if self.model_name not in IMAGE_PROCESSOR_MODELS:
            self.logger.error(f"지원하지 않는 모델: {self.model_name}")
            self.llm = None
            return

        api_model, model_class = IMAGE_PROCESSOR_MODELS[self.model_name]
        api_key = LLM_API_KEY.get(self.model_name)

        if not api_key:
            self.logger.error(f"API 키 누락: {self.model_name} API 키를 .env에 설정하세요.")
            self.llm = None
            return

        try:
            self.llm = model_class(api_key=api_key, model=api_model, temperature=TEMPERATURES["image"])
            self.logger.info(f"{self.model_name} 모델 초기화 성공")
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"model_name": self.model_name, "operation": "llm_initialization"}
            )
            self.logger.error(f"{self.model_name} 모델 초기화 실패: {error_detail['error']}")
            self.llm = None
    
    def process_single_image(self, file_path: str) -> Dict[str, Any]:
        """
        단일 이미지 처리
        
        Args:
            file_path (str): 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 파일 존재 확인
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {file_path}")
            
            # 파일 정보 설정
            file_info = {
                'file_path': str(file_path),
                'file_name': path.name
            }
            
            # 메타데이터 추출
            metadata = self._extract_metadata(file_path)
            
            # 이미지가 처리 가능한 형식인지 확인
            self._validate_image(file_path)
            
            # 이미지 인코딩
            base64_image = self._encode_image(file_path)
            
            # 캡션 생성
            captions = self._generate_captions(base64_image, metadata)
            
            return {
                'file_info': file_info,
                'metadata': metadata,
                'captions': captions
            }
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "operation": "image_processing"}
            )
            self.logger.error(f"이미지 처리 오류: {error_detail['error']}")
            
            return {
                'file_info': {
                    'file_path': str(file_path),
                    'file_name': Path(file_path).name
                },
                'error': str(e)
            }
    
    def _validate_image(self, image_path: str):
        """
        이미지가 유효한지 확인
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Raises:
            ValueError: 이미지가 유효하지 않은 경우
        """
        try:
            with Image.open(image_path) as img:
                # 최소 크기 확인
                if img.width < 10 or img.height < 10:
                    raise ValueError(f"이미지 크기가 너무 작습니다: {img.width}x{img.height}")
                
                # 지원되는 형식인지 확인
                if img.format not in ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']:
                    self.logger.warning(f"일반적이지 않은 이미지 형식: {img.format}")
        except Exception as e:
            if not isinstance(e, ValueError):
                raise ValueError(f"유효하지 않은 이미지 파일: {str(e)}")
            raise
    
    def _extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 메타데이터 추출
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if not exif:
                    self.logger.warning("EXIF 데이터가 없습니다.")
                    return {'DateTimeOriginal': 'N/A', 'GPSInfo': {'address': 'N/A'}}
                    
                metadata = {}
                self.logger.debug(f"추출된 EXIF 태그: {[TAGS.get(tag_id, tag_id) for tag_id in exif.keys()]}")
                
                # GPS 태그 확인
                gps_info_tag = None
                for tag_id, name in TAGS.items():
                    if name == 'GPSInfo':
                        gps_info_tag = tag_id
                        break
                
                # 필요한 태그 처리
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        gps_data = {}
                        for t in value:
                            gps_tag = GPSTAGS.get(t, t)
                            gps_data[gps_tag] = value[t]
                        metadata[tag] = self._process_gps_data(gps_data)
                    elif tag == "DateTimeOriginal":
                        metadata[tag] = str(value)
                
                # 필수 필드가 없는 경우 기본값 설정
                if 'DateTimeOriginal' not in metadata:
                    metadata['DateTimeOriginal'] = 'N/A'
                if 'GPSInfo' not in metadata or not metadata['GPSInfo']:
                    metadata['GPSInfo'] = {'address': 'N/A'}
                elif 'address' not in metadata['GPSInfo']:
                    metadata['GPSInfo']['address'] = 'N/A'
                    
                return metadata
                
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"image_path": image_path, "operation": "metadata_extraction"}
            )
            self.logger.error(f"메타데이터 추출 실패: {error_detail['error']}")
            return {'DateTimeOriginal': 'N/A', 'GPSInfo': {'address': 'N/A'}}
    
    def _process_gps_data(self, gps_data: Dict) -> Dict[str, Any]:
        """
        GPS 데이터 처리 및 위치 정보 획득
        
        Args:
            gps_data (Dict): GPS EXIF 데이터
            
        Returns:
            Dict[str, Any]: 처리된 GPS 정보
        """
        if not gps_data:
            return {'address': 'N/A'}
            
        try:
            lat = self._convert_to_degrees(gps_data.get('GPSLatitude', [0,0,0]))
            lon = self._convert_to_degrees(gps_data.get('GPSLongitude', [0,0,0]))
            
            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
            lon_ref = gps_data.get('GPSLongitudeRef', 'E')
            
            if lat_ref != 'N':
                lat = -lat
            if lon_ref != 'E':
                lon = -lon
                
            self.logger.debug(f"변환된 좌표: {lat}, {lon}")
            
            # 지오코더가 초기화되지 않은 경우 좌표만 반환
            if not self.geolocator:
                return {
                    'coordinates': {'latitude': lat, 'longitude': lon},
                    'address': 'N/A'
                }
            
            # 위치 정보 조회
            try:
                location = self.geolocator.reverse(f"{lat}, {lon}", timeout=5)
                if location:
                    return {
                        'coordinates': {'latitude': lat, 'longitude': lon},
                        'address': location.address,
                        'raw': location.raw['address']
                    }
            except GeocoderTimedOut:
                self.logger.warning("위치 정보 조회 시간 초과")
            except Exception as e:
                self.logger.warning(f"위치 정보 조회 실패: {str(e)}")
            
            # 조회 실패 시 좌표만 반환
            return {
                'coordinates': {'latitude': lat, 'longitude': lon},
                'address': 'N/A'
            }
            
        except Exception as e:
            self.logger.error(f"GPS 데이터 처리 실패: {str(e)}")
            return {'address': 'N/A'}
    
    def _convert_to_degrees(self, value) -> float:
        """
        GPS 좌표를 도(degree) 단위로 변환
        
        Args:
            value: GPS 좌표 값 (도, 분, 초)
            
        Returns:
            float: 변환된 좌표 값
        """
        if not value:
            return 0.0
        return float(value[0]) + float(value[1])/60.0 + float(value[2])/3600.0
    
    def _encode_image(self, image_path: str) -> str:
        """
        이미지를 base64로 인코딩
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            str: base64 인코딩된 이미지
        """
        try:
            with Image.open(image_path) as img:
                # RGBA to RGB 변환
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # 이미지 리사이즈 (최대 크기 제한)
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.LANCZOS)
                
                # JPEG로 변환 및 인코딩
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"image_path": image_path, "operation": "image_encoding"}
            )
            self.logger.error(f"이미지 인코딩 실패: {error_detail['error']}")
            raise ValueError(f"이미지 인코딩 실패: {str(e)}")
    
    def _generate_captions(self, base64_image: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        선택된 LLM 모델을 사용하여 이미지 캡션 생성
        
        Args:
            base64_image (str): base64 인코딩된 이미지
            metadata (Dict[str, Any]): 이미지 메타데이터
            
        Returns:
            Dict[str, str]: 모델별 캡션
        """
        if not self.llm:
            return {self.model_name: "❌ 모델 초기화 실패"}

        # 프롬프트 생성
        prompt = self._create_prompt(metadata)

        # 메시지 구성 (모델별 차이 처리)
        messages = self._create_model_messages(prompt, base64_image)

        # 결과 초기화
        result = {}
        
        try:
            # LLM 호출 (재시도 및 오류 처리 포함)
            response = self._call_llm_with_retry(messages)
            result[self.model_name] = response.content
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"model_name": self.model_name, "operation": "caption_generation"}
            )
            self.logger.error(f"{self.model_name} 캡션 생성 실패: {error_detail['error']}")
            result[self.model_name] = "⚠️ 이미지 분석이 불가능합니다. 다시 시도해주세요."

        return result
    
    def _create_prompt(self, metadata: Dict[str, Any]) -> str:
        """
        캡션 생성을 위한 프롬프트 생성
        
        Args:
            metadata (Dict[str, Any]): 이미지 메타데이터
            
        Returns:
            str: 생성된 프롬프트
        """
        date_time = metadata.get('DateTimeOriginal', 'N/A')
        location = metadata.get('GPSInfo', {}).get('address', 'N/A')

        user_prompt = IMAGE_PROCESSOR_PROMPT.get("user", "이미지에 대해 설명해주세요.")
        return user_prompt.format(date_time=date_time, location=location)
    
    def _create_model_messages(self, prompt: str, base64_image: str) -> List:
        """
        모델별 메시지 형식 생성
        
        Args:
            prompt (str): 프롬프트 텍스트
            base64_image (str): base64 인코딩된 이미지
            
        Returns:
            List: 메시지 리스트
        """
        # Groq 모델의 경우 시스템 메시지를 따로 설정하지 않음
        if self.model_name == 'groq':
            return [
                HumanMessage(content=[
                    {"type": "text", "text": self.system_prompt + "\n\n" + prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
        else:
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
    
    def _call_llm_with_retry(self, messages: List, max_retries: int = 3) -> Any:
        """
        재시도 로직이 적용된 LLM 호출
        
        Args:
            messages (List): 메시지 리스트
            max_retries (int): 최대 재시도 횟수
            
        Returns:
            Any: LLM 응답
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # OpenAI, Gemini 모델은 토큰 사용량을 확인함
                if self.model_name in ['openai', 'gemini']:
                    with get_openai_callback() as cb:
                        response = self.llm.invoke(messages)
                        self.logger.info(f"{self.model_name} 토큰 사용량: {cb.total_tokens}")
                        return response
                else:
                    return self.llm.invoke(messages)
            except Exception as e:
                last_error = e
                retry_count += 1
                
                # 지수 백오프 (1초, 2초, 4초...)
                wait_time = 2 ** (retry_count - 1)
                self.logger.warning(f"캡션 생성 실패 (시도 {retry_count}/{max_retries}): {str(e)}. {wait_time}초 후 재시도")
                
                import time
                time.sleep(wait_time)
        
        # 모든 재시도 실패 시 마지막 오류 발생
        raise last_error or Exception("알 수 없는 오류로 캡션 생성 실패")