"""
말투 (tone)
격식 있는: 공식적인 상황이나 문서에서 사용되는 예의 바르고 정중한 말투
친근한: 편안하고 따뜻한 느낌을 주는 부드러운 말투
딱딱한: 감정을 드러내지 않고 건조하게 사실만 전달하는 사무적인 말투
유머러스한: 우스갯소리나 농담을 섞어 즐거운 분위기를 만드는 말투
진지한: 무게감 있고 신중하게 자신의 의견을 전달하는 말투
긍정적인: 낙관적이고 희망적인 메시지를 전달하는 밝은 말투
부정적인: 비관적이고 불만스러운 감정을 드러내는 어두운 말투
설득적인: 상대방을 이해시키고 동의를 얻기 위해 논리적으로 말하는 말투
비판적인: 문제점을 지적하고 개선을 요구하는 날카로운 말투
중립적인: 어느 한쪽으로 치우치지 않고 객관적인 사실만 전달하는 말투
"""

"""
어조 (voice)
분석적인: 논리적이고 체계적으로 정보를 분석하고 해석하는 어조
설명적인: 복잡한 내용을 쉽고 명확하게 전달하는 어조
비판적인: 문제점을 지적하고 개선 방향을 제시하는 어조
주장하는: 자신의 의견을 강하게 피력하고 설득하는 어조
질문하는: 호기심을 가지고 질문을 통해 정보를 얻고 탐구하는 어조
사색적인: 깊이 생각하고 성찰하는 차분하고 조용한 어조
열정적인: 강한 의지와 흥분을 표현하는 활기찬 어조
객관적인: 개인적인 감정이나 편견 없이 사실을 전달하는 어조
감성적인: 감정과 느낌을 풍부하게 표현하는 섬세한 어조
창의적인: 독창적인 아이디어를 제시하고 새로운 관점을 제시하는 어조
"""

"""
페르소나 정보를 JSON 파일로부터 로드하는 모듈
"""
import json
import os
import logging
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON 파일 경로 (환경 변수 또는 기본값)
JSON_PATH = os.environ.get("PERSONA_JSON_PATH", "personas.json")

def load_personas_from_json(file_path=JSON_PATH) -> List[Dict[str, Any]]:
    """
    JSON 파일에서 페르소나 목록 로드
    
    Args:
        file_path (str): 로드할 JSON 파일 경로
    
    Returns:
        list: 페르소나 목록
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"JSON 파일이 존재하지 않습니다: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_list = json.load(f)
        logger.info(f"{len(loaded_list)}개의 페르소나를 {file_path}에서 로드했습니다.")
        return loaded_list
    except Exception as e:
        logger.error(f"JSON 로드 실패: {str(e)}")
        return []

# 모듈 로드 시 바로 JSON에서 페르소나 로드
PERSONA_LIST = load_personas_from_json()

# 테스트 코드
if __name__ == "__main__":
    # 현재 로드된 페르소나 목록 출력
    if PERSONA_LIST:
        print(f"총 {len(PERSONA_LIST)}개의 페르소나 로드됨:")
        for persona in PERSONA_LIST:
            print(f"- {persona['name']} ({persona['gender']}, {persona['birth_date']})")
    else:
        print(f"페르소나를 로드하지 못했습니다. JSON 파일({JSON_PATH})이 있는지 확인하세요.")