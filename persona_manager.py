# persona_manager.py
"""
페르소나 정보를 관리하는 모듈
"""
import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from logging_manager import LoggingManager
from error_handler import ErrorHandler

class PersonaManager:
    """페르소나 관리 클래스"""
    
    # 싱글톤 인스턴스
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        if cls._instance is None:
            cls._instance = PersonaManager()
        return cls._instance
    
    def __init__(self):
        """초기화"""
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("persona_manager")
        self.error_handler = ErrorHandler.get_instance()
        
        # JSON 파일 경로 (환경 변수 또는 기본값)
        self.json_path = os.environ.get("PERSONA_JSON_PATH", "personas.json")
        
        # 페르소나 목록
        self.personas = []
        
        # 페르소나 로드
        self._load_personas()
    
    def _load_personas(self):
        """JSON 파일에서 페르소나 목록 로드"""
        try:
            if not os.path.exists(self.json_path):
                self.logger.warning(f"JSON 파일이 존재하지 않습니다: {self.json_path}")
                self._create_default_personas()
                return
                
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.personas = json.load(f)
                
            self.logger.info(f"{len(self.personas)}개의 페르소나를 {self.json_path}에서 로드했습니다.")
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"json_path": self.json_path, "operation": "load_personas"}
            )
            self.logger.error(f"JSON 로드 실패: {error_detail['error']}")
            self._create_default_personas()
    
    def _create_default_personas(self):
        """기본 페르소나 생성"""
        self.personas = [
            {
                "name": "나준채",
                "gender": "male",
                "birth_date": "1980-01-29",
                "tone": "격식 있는",
                "voice": "분석적인"
            }
        ]
        
        self.logger.info("기본 페르소나를 생성했습니다.")
        
        # 파일 저장
        self.save_personas()
    
    def get_all_personas(self) -> List[Dict[str, Any]]:
        """
        모든 페르소나 목록 반환
        
        Returns:
            List[Dict[str, Any]]: 페르소나 목록
        """
        return self.personas
    
    def get_persona_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        이름으로 페르소나 찾기
        
        Args:
            name (str): 페르소나 이름
            
        Returns:
            Optional[Dict[str, Any]]: 페르소나 정보 또는 None
        """
        for persona in self.personas:
            if persona['name'] == name:
                return persona
        return None
    
    def add_persona(self, persona: Dict[str, Any]) -> bool:
        """
        새 페르소나 추가
        
        Args:
            persona (Dict[str, Any]): 추가할 페르소나 정보
            
        Returns:
            bool: 성공 여부
        """
        # 필수 필드 확인
        required_fields = ['name', 'tone', 'voice', 'gender', 'birth_date']
        for field in required_fields:
            if field not in persona:
                self.logger.error(f"페르소나 추가 실패: 필수 필드 '{field}'가 누락되었습니다.")
                return False
        
        # 이름 중복 확인
        if self.get_persona_by_name(persona['name']):
            self.logger.error(f"페르소나 추가 실패: 이미 '{persona['name']}' 이름의 페르소나가 존재합니다.")
            return False
        
        # 페르소나 추가
        self.personas.append(persona)
        
        # 파일 저장
        self.save_personas()
        
        self.logger.info(f"새 페르소나 추가됨: {persona['name']}")
        return True
    
    def update_persona(self, name: str, updated_data: Dict[str, Any]) -> bool:
        """
        페르소나 정보 업데이트
        
        Args:
            name (str): 업데이트할 페르소나 이름
            updated_data (Dict[str, Any]): 업데이트할 데이터
            
        Returns:
            bool: 성공 여부
        """
        for i, persona in enumerate(self.personas):
            if persona['name'] == name:
                # 이름 변경 시 중복 확인
                if 'name' in updated_data and updated_data['name'] != name:
                    if self.get_persona_by_name(updated_data['name']):
                        self.logger.error(f"페르소나 업데이트 실패: 이미 '{updated_data['name']}' 이름의 페르소나가 존재합니다.")
                        return False
                
                # 데이터 업데이트
                self.personas[i].update(updated_data)
                
                # 파일 저장
                self.save_personas()
                
                self.logger.info(f"페르소나 업데이트됨: {name} -> {self.personas[i]['name']}")
                return True
        
        self.logger.error(f"페르소나 업데이트 실패: '{name}' 이름의 페르소나를 찾을 수 없습니다.")
        return False
    
    def delete_persona(self, name: str) -> bool:
        """
        페르소나 삭제
        
        Args:
            name (str): 삭제할 페르소나 이름
            
        Returns:
            bool: 성공 여부
        """
        for i, persona in enumerate(self.personas):
            if persona['name'] == name:
                del self.personas[i]
                
                # 페르소나가 모두 삭제되면 기본 페르소나 생성
                if not self.personas:
                    self._create_default_personas()
                else:
                    # 파일 저장
                    self.save_personas()
                
                self.logger.info(f"페르소나 삭제됨: {name}")
                return True
        
        self.logger.error(f"페르소나 삭제 실패: '{name}' 이름의 페르소나를 찾을 수 없습니다.")
        return False
    
    def save_personas(self) -> bool:
        """
        페르소나 목록을 JSON 파일로 저장
        
        Returns:
            bool: 성공 여부
        """
        try:
            # 디렉토리 확인 및 생성
            json_path = Path(self.json_path)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 파일 저장
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.personas, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"페르소나 목록을 {self.json_path}에 저장했습니다.")
            return True
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"json_path": self.json_path, "operation": "save_personas"}
            )
            self.logger.error(f"페르소나 저장 실패: {error_detail['error']}")
            return False


# 모듈 레벨에서 사용할 PERSONA_LIST 변수 정의
PERSONA_LIST = PersonaManager.get_instance().get_all_personas()