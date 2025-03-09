import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pathlib import Path
import base64
import io
import logging
import ssl
from typing import Dict, Any, List, Optional
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
import dotenv
from config import IMAGE_PROCESSOR_MODELS, TEMPERATURES, LLM_API_KEY
from ProcessorPrompt import IMAGE_PROCESSOR_PROMPT

# .env 파일 로드
dotenv.load_dotenv()

# LLM Settings
TEMPERATURE = TEMPERATURES["image"]

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self, selected_model: str):
        """이미지 처리기 초기화
        
        :param selected_model: 사용할 모델명 ('openai', 'gemini', 'claude', 'groq')
        """
        self.selected_model = selected_model.lower()
        self.llm = None  # 선택된 모델의 LLM 인스턴스
        self.system_prompt = IMAGE_PROCESSOR_PROMPT["system"]
        self._setup_geolocator()
        self._setup_llm()
        
    def _setup_geolocator(self):
        """지오코더 설정"""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.geolocator = Nominatim(user_agent="image_processor", ssl_context=ctx)
        
    def _setup_llm(self):
        """LLM 모델 설정 (config.py 기반, API 키 예외 처리 추가)"""

        if self.selected_model not in IMAGE_PROCESSOR_MODELS:
            logger.error(f"지원하지 않는 모델: {self.selected_model}")
            self.llm = None
            return

        api_model, model_class = IMAGE_PROCESSOR_MODELS[self.selected_model]
        api_key = LLM_API_KEY.get(self.selected_model)

        if not api_key:
            logger.error(f"API 키 누락: {self.selected_model} API 키를 .env에 설정하세요.")
            self.llm = None
            return

        try:
            self.llm = model_class(api_key=api_key, model=api_model, temperature=TEMPERATURE)
            logger.info(f"{self.selected_model} 모델 초기화 성공")
        except Exception as e:
            logger.error(f"{self.selected_model} 모델 초기화 실패: {str(e)}")
            self.llm = None

    def _process_single_image(self, file_path: str) -> Dict[str, Any]:
        """단일 이미지 처리
        
        Args:
            file_path (str): 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
            {
                'file_info': {
                    'file_path': str,
                    'file_name': str
                },
                'metadata': {
                    'DateTimeOriginal': str,
                    'GPSInfo': {
                        'coordinates': {
                            'latitude': float,
                            'longitude': float
                        },
                        'address': str,
                        'raw': dict
                    }
                },
                'captions': {
                    'model_name': str
                }
            }
        """
        try:
            # 파일 정보 설정
            file_info = {
                'file_path': file_path,
                'file_name': Path(file_path).name
            }
            
            # 메타데이터 추출
            metadata = self._extract_metadata(file_path)
            
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
            logger.error(f"이미지 처리 오류 ({file_path}): {str(e)}")
            return {
                'file_info': {
                    'file_path': file_path,
                    'file_name': Path(file_path).name
                },
                'error': str(e)
            }

    def _extract_metadata(self, image_path: str) -> Dict[str, Any]:
        """이미지 메타데이터 추출"""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if not exif:
                    logger.warning("EXIF 데이터가 없습니다.")
                    return {'DateTimeOriginal': 'N/A', 'GPSInfo': {'address': 'N/A'}}
                    
                metadata = {}
                logger.info(f"추출된 EXIF 태그: {[TAGS.get(tag_id, tag_id) for tag_id in exif.keys()]}")
                
                # GPS 태그 확인
                gps_info_tag = None
                for tag_id, name in TAGS.items():
                    if name == 'GPSInfo':
                        gps_info_tag = tag_id
                        break
                
                if gps_info_tag and gps_info_tag in exif:
                    logger.info("GPS 정보가 발견되었습니다.")
                else:
                    logger.warning("GPS 정보가 없습니다.")
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        logger.info(f"GPS 데이터 발견: {value}")
                        gps_data = {}
                        for t in value:
                            gps_tag = GPSTAGS.get(t, t)
                            gps_data[gps_tag] = value[t]
                        logger.info(f"변환된 GPS 데이터: {gps_data}")
                        metadata[tag] = self._process_gps_data(gps_data)
                    elif tag == "DateTimeOriginal":
                        metadata[tag] = str(value)
                        logger.info(f"촬영 시간 발견: {value}")
                
                # 필수 필드가 없는 경우 기본값 설정
                if 'DateTimeOriginal' not in metadata:
                    metadata['DateTimeOriginal'] = 'N/A'
                if 'GPSInfo' not in metadata or not metadata['GPSInfo']:
                    metadata['GPSInfo'] = {'address': 'N/A'}
                elif 'address' not in metadata['GPSInfo']:
                    metadata['GPSInfo']['address'] = 'N/A'
                    
                logger.info(f"최종 추출된 메타데이터: {metadata}")
                return metadata
                
        except Exception as e:
            logger.error(f"메타데이터 추출 실패: {str(e)}")
            return {'DateTimeOriginal': 'N/A', 'GPSInfo': {'address': 'N/A'}}

    def _process_gps_data(self, gps_data: Dict) -> Dict[str, Any]:
        """GPS 데이터 처리 및 위치 정보 획득"""
        try:
            logger.info("GPS 데이터 처리 시작")
            
            if not gps_data:
                logger.warning("GPS 데이터가 비어있습니다.")
                return {'address': 'N/A'}
                
            logger.info(f"GPS 좌표 변환 시작: Latitude: {gps_data.get('GPSLatitude')}, Longitude: {gps_data.get('GPSLongitude')}")
            
            lat = self._convert_to_degrees(gps_data.get('GPSLatitude', [0,0,0]))
            lon = self._convert_to_degrees(gps_data.get('GPSLongitude', [0,0,0]))
            
            lat_ref = gps_data.get('GPSLatitudeRef', 'N')
            lon_ref = gps_data.get('GPSLongitudeRef', 'E')
            
            if lat_ref != 'N':
                lat = -lat
            if lon_ref != 'E':
                lon = -lon
                
            logger.info(f"변환된 좌표: {lat}, {lon}")
            logger.info(f"좌표 참조값: Lat Ref: {lat_ref}, Lon Ref: {lon_ref}")
            
            try:
                location = self.geolocator.reverse(f"{lat}, {lon}")
                if location:
                    logger.info(f"위치 정보 조회 성공: {location.address}")
                    return {
                        'coordinates': {'latitude': lat, 'longitude': lon},
                        'address': location.address,
                        'raw': location.raw['address']
                    }
                else:
                    logger.warning("위치 정보를 찾을 수 없습니다.")
            except GeocoderTimedOut:
                logger.error("위치 정보 조회 시간 초과")
            except Exception as e:
                logger.error(f"위치 정보 조회 실패: {str(e)}")
            
        except Exception as e:
            logger.error(f"GPS 데이터 처리 실패: {str(e)}")
            
        return {'address': 'N/A'}

    def _convert_to_degrees(self, value) -> float:
        """GPS 좌표를 도(degree) 단위로 변환"""
        if not value:
            return 0.0
        return float(value[0]) + float(value[1])/60.0 + float(value[2])/3600.0

    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with Image.open(image_path) as img:
            # RGBA to RGB 변환
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # 이미지 리사이즈
            img.thumbnail((512, 512), Image.LANCZOS)
            
            # JPEG로 변환 및 인코딩
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _generate_captions(self, base64_image: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """선택된 LLM 모델을 사용하여 이미지 캡션 생성"""
        if not self.llm:
            return {self.selected_model: "❌ 모델 초기화 실패"}

        prompt = self._create_prompt(metadata)

        # Groq 모델의 경우 시스템 메시지를 따로 설정하지 않음
        if self.selected_model == 'groq':
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": self.system_prompt + "\n\n" + prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
        else:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]

        result = {}
        try:
            # OpenAI, Gemini 모델은 토큰 사용량을 확인함
            if self.selected_model in ['openai', 'gemini']:
                with get_openai_callback() as cb:
                    response = self.llm.invoke(messages)
                    logger.info(f"{self.selected_model} 토큰 사용량: {cb.total_tokens}")
            else:
                response = self.llm.invoke(messages)

            result[self.selected_model] = response.content
        except Exception as e:
            error_msg = f"Error code: {getattr(e, 'status_code', 'Unknown')} - {str(e)}"
            logger.error(f"{self.selected_model} 캡션 생성 실패: {error_msg}")
            result[self.selected_model] = "⚠️ 이미지 분석이 불가능합니다. 다시 시도해주세요."

        return result


    def _create_prompt(self, metadata: Dict[str, Any]) -> str:
        """캡션 생성을 위한 프롬프트 생성"""
        date_time = metadata.get('DateTimeOriginal', 'N/A')
        location = metadata.get('GPSInfo', {}).get('address', 'N/A')

        system_prompt = IMAGE_PROCESSOR_PROMPT.get("system", "이미지 분석을 수행하세요.")
        user_prompt = IMAGE_PROCESSOR_PROMPT.get("user", "이미지에 대해 설명해주세요.")

        return f"{system_prompt}\n\n{user_prompt.format(date_time=date_time, location=location)}"


def main():
    """메인 실행 함수"""
    valid_models = IMAGE_PROCESSOR_MODELS.keys()
    
    # 모델 선택
    while True:
        selected_model = input("\n사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
        if selected_model in valid_models:
            break
        print("올바른 모델명을 입력해주세요.")
    
    analyzer = ImageAnalyzer(selected_model)
    
    if not analyzer.llm:
        print(f"\n{selected_model.upper()} 모델 초기화에 실패했습니다.")
        return
    
    # 이미지 처리
    while True:
        image_path = input("\n처리할 이미지 경로를 입력하세요 (종료하려면 엔터): ").strip()
        if not image_path:
            break
            
        try:
            result = analyzer._process_single_image(image_path)
            caption = result['captions'].get(selected_model)
            metadata = result['metadata']
            
            print("\n처리 결과:")
            print(f"1. 이미지 경로: {image_path}")
            print("\n2. 메타데이터:")
            print(f"   - 생성 일시: {metadata.get('DateTimeOriginal', 'N/A')}")
            print(f"   - 촬영 위치: {metadata.get('GPSInfo', {}).get('address', 'N/A')}")
            print(f"\n3. {selected_model.upper()} 모델의 이미지 분석 결과:")
            print(f"   {caption}")
                
        except Exception as e:
            print(f"\n이미지 처리 중 오류가 발생했습니다: {str(e)}")
            continue
            
    print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()