import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import base64
import io
import logging
import ssl
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback
import dotenv
from config import IMAGE_PROCESSOR_MODELS, TEMPERATURES

# .env 파일 로드
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Names
CHATGPT = IMAGE_PROCESSOR_MODELS["chatgpt"]
GEMINI = IMAGE_PROCESSOR_MODELS["gemini"]
CLAUDE = IMAGE_PROCESSOR_MODELS["claude"]
GROQ = IMAGE_PROCESSOR_MODELS["groq"]

# LLM Settings
TEMPERATURE = TEMPERATURES["image"]

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, selected_model: str):
        """이미지 처리기 초기화
        
        :param selected_model: 사용할 모델명 ('openai', 'gemini', 'claude', 'groq')
        """
        self.selected_model = selected_model.lower()
        self.llm = None  # 선택된 모델의 LLM 인스턴스
        self.system_prompt = "당신은 전문적인 이미지 분석가입니다. 사용자가 제시하는 이미지와 메타데이터를 바탕으로 객관적이고 통찰력 있는 분석을 제공하는 것이 당신의 역할입니다."
        self._setup_geolocator()
        self._setup_llms()
        
    def _setup_geolocator(self):
        """지오코더 설정"""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.geolocator = Nominatim(user_agent="image_processor", ssl_context=ctx)
        
    def _setup_llms(self):
        """LLM 모델 설정"""
        model_configs = {
            'openai': (ChatOpenAI, {'api_key': OPENAI_API_KEY, 'model': CHATGPT}),
            'gemini': (ChatGoogleGenerativeAI, {'api_key': GOOGLE_API_KEY, 'model': GEMINI}),
            'claude': (ChatAnthropic, {'api_key': ANTHROPIC_API_KEY, 'model': CLAUDE}),
            'groq': (ChatGroq, {'api_key': GROQ_API_KEY, 'model': GROQ})
        }
        
        # 선택된 모델만 초기화
        if self.selected_model in model_configs:
            model_class, config = model_configs[self.selected_model]
            try:
                self.llm = model_class(temperature=TEMPERATURE, **config)
                logger.info(f"{self.selected_model} 모델 초기화 성공")
            except Exception as e:
                logger.error(f"{self.selected_model} 모델 초기화 실패: {str(e)}")
                self.llm = None
        else:
            logger.error(f"지원하지 않는 모델: {self.selected_model}")
            self.llm = None

    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """단일 이미지 처리"""
        try:
            metadata = self._extract_metadata(image_path)
            base64_image = self._encode_image(image_path)
            captions = self._generate_captions(base64_image, metadata)
            
            return {
                'image_path': image_path,
                'metadata': metadata,
                'captions': captions
            }
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {str(e)}")
            raise

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
            return {self.selected_model: "모델 초기화 실패"}
            
        prompt = self._create_prompt(metadata)
        
        # Groq는 시스템 메시지를 지원하지 않으므로 별도 처리
        if self.selected_model == 'groq':
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": self.system_prompt + "\n\n" + prompt},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
        else:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
        
        result = {}
        try:
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
            result[self.selected_model] = f"캡션 생성 실패: {error_msg}"
                
        return result

    def _create_prompt(self, metadata: Dict[str, Any]) -> str:
        """캡션 생성을 위한 프롬프트 생성"""
        date_time = metadata.get('DateTimeOriginal', 'N/A')
        location = metadata.get('GPSInfo', {}).get('address', 'N/A')
        
        return f"""이 이미지를 한국어로 하여, 시각장애인에게 설명하는 것처럼 설명해주세요. 간결하고 명확하게 다음 요소를 중심으로 묘사해주세요."

주요 내용과 구도: 이미지에서 가장 중요한 요소와 그 배치
분위기와 특징: 색감, 조명, 감정적 분위기 등
시간 및 장소: 촬영된 시각과 배경의 특징
구조 및 외형: 이미지 속 사물이나 인물의 형태, 배치, 색상 등
행동과 상황: 등장하는 인물/대상의 감정, 동작, 행동, 맥락

참고 정보:
- 촬영 시간: {date_time}
- 위치: {location}"""

def main():
    """메인 실행 함수"""
    valid_models = ['openai', 'gemini', 'claude', 'groq']
    
    # 모델 선택
    while True:
        selected_model = input("\n사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
        if selected_model in valid_models:
            break
        print("올바른 모델명을 입력해주세요.")
    
    processor = ImageProcessor(selected_model)
    
    if not processor.llm:
        print(f"\n{selected_model.upper()} 모델 초기화에 실패했습니다.")
        return
    
    # 이미지 처리
    while True:
        image_path = input("\n처리할 이미지 경로를 입력하세요 (종료하려면 엔터): ").strip()
        if not image_path:
            break
            
        try:
            result = processor._process_single_image(image_path)
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