# research_note_generator.py
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
from pathlib import Path

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

from logging_manager import LoggingManager
from error_handler import ErrorHandler
from config import RESEARCH_NOTE_GENERATOR_MODELS, TEMPERATURES, LLM_API_KEY
from ProcessorPrompt import RESEARCH_NOTE_GENERATOR_PROMPT, RESEARCH_NOTE_TITLE_GENERATOR_PROMPT

print("research_note_generator.py 로드 중...")

class ResearchNoteGenerator:
    """
    표준화된 분석 결과를 기반으로 연구노트를 생성하는 클래스
    """
    
    def __init__(self, model_name: str):
        """
        ResearchNoteGenerator 초기화
        
        Args:
            model_name (str): 사용할 LLM 모델명 (openai/gemini/claude/groq)
        """
        self.model_name = model_name.lower()
        self.temperature = TEMPERATURES.get('research_note', 0.4)
        self.llm = None
        
        # 로거 및 에러 핸들러 설정
        self.logger = LoggingManager.get_instance().get_logger("research_note_generator")
        self.error_handler = ErrorHandler.get_instance()
        
        # 캐시 저장소 초기화
        self.title_cache = {}
        
        # LLM 모델 설정
        self._setup_llm()
    
    def _setup_llm(self):
        """LLM 모델 설정"""
        if self.model_name not in RESEARCH_NOTE_GENERATOR_MODELS:
            self.logger.error(f"지원하지 않는 모델: {self.model_name}")
            self.llm = None
            return

        api_model, model_class = RESEARCH_NOTE_GENERATOR_MODELS[self.model_name]
        api_key = LLM_API_KEY.get(self.model_name)

        if not api_key:
            self.logger.error(f"API 키 누락: {self.model_name} API 키를 .env에 설정하세요.")
            self.llm = None
            return

        try:
            self.llm = model_class(api_key=api_key, model=api_model, temperature=self.temperature)
            self.logger.info(f"{self.model_name} 모델 초기화 성공")
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"model_name": self.model_name, "operation": "llm_initialization"}
            )
            self.logger.error(f"{self.model_name} 모델 초기화 실패: {error_detail['error']}")
            self.llm = None

    def _estimate_token_count(self, results):
        """결과 데이터의 대략적인 토큰 수 계산"""
        total_chars = 0
        
        # 시스템 프롬프트와 기본 사용자 프롬프트 문자 수 계산 (대략적으로)
        system_prompt_chars = len(RESEARCH_NOTE_GENERATOR_PROMPT["system"]) 
        user_prompt_chars = len(RESEARCH_NOTE_GENERATOR_PROMPT["user"])
        total_chars += system_prompt_chars + user_prompt_chars
        
        # 결과 데이터 문자 수 계산
        for result in results:
            # 파일 경로
            total_chars += len(str(result.get('path', '')))
            
            # 파일 분석 결과
            if 'content' in result:
                total_chars += len(str(result.get('content', '')))
            if 'brief' in result:
                total_chars += len(str(result.get('brief', '')))
            if 'summary' in result:
                total_chars += len(str(result.get('summary', '')))
            # 기타 필요한 필드 추가
        
        # 대략적으로 4자당 1토큰으로 계산 (영어 기준, 한글은 더 많을 수 있음)
        estimated_tokens = total_chars // 3
        
        return estimated_tokens

    def generate_research_note(self,
                          results: Dict[str, Any],
                          persona_name: Optional[str] = None,
                          title: Optional[str] = None,
                          focus_category: Optional[str] = None) -> Dict[str, Any]:
        """
        연구노트 생성
        
        Args:
            results (Dict[str, Any]): ParallelProcessor의 표준화된 결과
            persona_name (str, optional): 적용할 페르소나 이름, None이면 랜덤 선택
            title (str, optional): 연구노트 제목, None이면 자동 생성
            focus_category (str, optional): 중점을 둘 카테고리 ('reference' 또는 'research'), None이면 균형있게
            
        Returns:
            Dict[str, Any]: 생성된 연구노트 정보
            {
                'title': str,                # 제목
                'content': str,              # 연구노트 내용
                'applied_persona': Dict,     # 적용된 페르소나 정보
                'generated_at': str          # 생성 일시
            }
        """
        if not self.llm:
            error_msg = "LLM 모델이 초기화되지 않았습니다"
            self.logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # 페르소나 선택 (지정되지 않은 경우 랜덤)
            persona = self._select_persona(persona_name)
            self.logger.info(f"선택된 페르소나: {persona['name']} (어조: {persona['tone']}, 음성: {persona['voice']})")
            
            # 시간 범위 추출
            time_range = self._extract_time_range(results)
            
            # 제목이 없는 경우 자동 생성
            if not title:
                title = self._generate_title(results, time_range)
            
            # GROQ 모델 사용 시 토큰 제한 체크
            if self.model_name.lower() == "groq":
                # 토큰 예측 및 제한 체크 수행
                token_check_result = self._check_groq_token_limit(results, persona, title, time_range, focus_category)
                
                # 토큰 제한 초과 시 오류 반환
                if "error" in token_check_result:
                    return token_check_result
            
            # 연구노트 생성을 위한 프롬프트 생성
            prompt = self._create_prompt(results, persona, title, time_range, focus_category)
            
            # LLM 호출
            system_message = SystemMessage(content=prompt["system"])
            user_message = HumanMessage(content=prompt["user"])
            
            if self.model_name.lower() in ['openai', 'gemini']:
                with get_openai_callback() as cb:
                    response = self.llm.invoke([system_message, user_message])
                    self.logger.info(f"{self.model_name} 토큰 사용량: {cb.total_tokens}")
            else:
                response = self.llm.invoke([system_message, user_message])
            
            # 결과 검증 및 후처리
            content = self._validate_category_descriptions(response.content)
            
            # 결과 구성
            result = {
                'title': title,
                'content': content,
                'applied_persona': persona,
                'generated_at': datetime.now().isoformat(),
            }
            
            return result
        
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"persona_name": persona_name, "focus_category": focus_category}
            )
            self.logger.error(f"연구노트 생성 실패: {error_detail['error']}")
            return {"error": str(e)}
        
    def _check_groq_token_limit(self, results, persona, title, time_range, focus_category):
        """GROQ 모델의 토큰 제한 체크"""
        try:
            # 프롬프트 생성 (토큰 예측용)
            temp_prompt = self._create_prompt(results, persona, title, time_range, focus_category)
            
            # 프롬프트 문자 수 계산
            prompt_chars = len(temp_prompt["system"]) + len(temp_prompt["user"])
            
            # 결과 데이터 문자 수 계산
            data_chars = 0
            for item in results.get('items', []):
                # 주요 필드의 문자 수 합산
                for key in ['content', 'brief', 'summary', 'caption']:
                    if key in item:
                        data_chars += len(str(item.get(key, '')))
            
            total_chars = prompt_chars + data_chars
            
            # 대략적인 토큰 수 계산 (한글 기준 약 3자당 1토큰)
            estimated_tokens = total_chars // 3
            
            # 모델별 토큰 제한
            token_limits = {
                "llama-3.3-70b-versatile": 6000,
                "llama-3.2-90b-vision-preview": 7000,
                "llama-3.1-8b": 4000,
                "default": 4000
            }
            
            # 모델 이름 가져오기 및 토큰 제한 확인
            model_name = getattr(self.llm, 'model_name', "default")
            token_limit = token_limits.get(model_name, token_limits["default"])
            
            # 토큰 제한 초과 체크
            if estimated_tokens > token_limit:
                error_msg = f"GROQ 모델 '{model_name}'의 토큰 제한({token_limit})을 초과했습니다. 예상 토큰: {estimated_tokens}"
                self.logger.error(error_msg)
                return {
                    "error": error_msg, 
                    "token_limit_exceeded": True, 
                    "estimated_tokens": estimated_tokens,
                    "token_limit": token_limit
                }
            
            # 토큰 사용량 로깅
            self.logger.info(f"GROQ 모델 토큰 예측: {estimated_tokens}/{token_limit}")
            return {"success": True}
            
        except Exception as e:
            self.logger.warning(f"토큰 제한 체크 중 오류 발생: {str(e)}")
            # 토큰 체크에 실패해도 계속 진행 (실패 시 제한 체크만 건너뜀)
            return {"success": True}
    
    def _validate_category_descriptions(self, content: str) -> str:
        """
        생성된 연구 노트 내용에서 카테고리와 서술 방식의 일치 여부 검증
        
        Args:
            content (str): 생성된 연구 노트 내용
            
        Returns:
            str: 검증 및 수정된 내용
        """
        lines = content.split('\n')
        corrected_lines = []
        
        current_category = None
        for i, line in enumerate(lines):
            # 카테고리 감지
            if "카테고리:" in line and "research" in line.lower():
                current_category = "research"
            elif "카테고리:" in line and "reference" in line.lower():
                current_category = "reference"
                
            # 연구 자료에서 '분석했다' 표현 수정
            if current_category == "research" and "분석했다" in line:
                line = line.replace("분석했다", "개발했다")
                
            # 참고 자료에서 '개발했다' 표현 수정
            if current_category == "reference" and "개발했다" in line:
                line = line.replace("개발했다", "분석했다")
                
            corrected_lines.append(line)
        
        return '\n'.join(corrected_lines)

    def _select_persona(self, persona_name: Optional[str] = None) -> Dict[str, Any]:
        """지정된 이름 또는 랜덤으로 페르소나 선택"""
        # 실제 적용 시에는 페르소나 목록을 별도 모듈에서 가져옴
        from persona_manager import PERSONA_LIST
        
        if persona_name:
            for persona in PERSONA_LIST:
                if persona["name"] == persona_name:
                    return persona
            self.logger.warning(f"'{persona_name}' 페르소나를 찾을 수 없습니다. 랜덤 선택합니다.")
        
        # 랜덤 선택
        return random.choice(PERSONA_LIST)
    
    def _extract_time_range(self, results: Dict[str, Any]) -> Dict[str, str]:
        """결과에서 시간 범위 추출"""
        dates = []
        
        # 일기에서 날짜 추출
        if results.get('diary_entries'):
            for entry in results['diary_entries']:
                date = entry.get('date')
                if date and date != "N/A":
                    dates.append(date)
        
        # 이미지 메타데이터에서 날짜 추출
        if results.get('image'):
            for img_result in results['image']:
                date = img_result.get('formatted_date')
                if date and date != "N/A":
                    dates.append(date)
        
        # 날짜 정렬 및 범위 추출
        if dates:
            dates.sort()
            return {
                "start": dates[0],
                "end": dates[-1]
            }
        
        # 날짜를 찾을 수 없는 경우 현재 날짜 반환
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "start": today,
            "end": today
        }
    
    def _generate_title(self, results: Dict[str, Any], time_range: Dict[str, str]) -> str:
        """연구 노트 제목 생성"""
        if not self.llm:
            # LLM이 없을 경우 기본 제목 반환
            return f"연구 노트: {time_range['start']} ~ {time_range['end']}"
        
        # 간단한 컨텍스트 요약 생성
        summary = self._create_context_summary(results)
        
        # 캐시 키 생성 (요약 + 시간 범위)
        cache_key = f"{summary}_{time_range['start']}_{time_range['end']}"
        
        # 캐시에서 제목 확인
        if cache_key in self.title_cache:
            self.logger.debug("캐시에서 제목 로드됨")
            return self.title_cache[cache_key]
        
        try:
            # 제목 생성 프롬프트 변수 포맷팅 적용
            system_content = RESEARCH_NOTE_TITLE_GENERATOR_PROMPT["system"]
            user_content = RESEARCH_NOTE_TITLE_GENERATOR_PROMPT["user"].format(
                            summary=summary,
                            time_range_start=time_range['start'],
                            time_range_end=time_range['end']
                        )
            
            # 제목 생성 프롬프트
            system_message = SystemMessage(content=system_content)
            user_message = HumanMessage(content=user_content)
            
            # LLM 호출을 에러 핸들링과 함께 수행
            try:
                response = self.llm.invoke([system_message, user_message])
                title = response.content.strip().replace('"', '').replace("'", "")
                
                # 제목이 너무 길면 잘라내기
                if len(title) > 50:
                    title = title[:47] + "..."
                
                # 캐시에 저장
                self.title_cache[cache_key] = title
                
                return title
            except Exception as e:
                error_detail = self.error_handler.handle_error(
                    e, {"operation": "title_generation", "summary": summary}
                )
                self.logger.error(f"제목 생성 중 오류 발생: {error_detail['error']}")
                return f"연구 노트: {time_range['start']} ~ {time_range['end']}"
                
        except Exception as e:
            self.logger.error(f"제목 생성 실패: {str(e)}")
            # 기본 제목 생성
            return f"연구 노트: {time_range['start']} ~ {time_range['end']}"
    
    def _create_context_summary(self, results: Dict[str, Any]) -> str:
        """결과에서 간단한 컨텍스트 요약 생성"""
        summary = ""
        
        # 일기 요약
        if results.get('diary_entries'):
            diary_count = len(results['diary_entries'])
            summary += f"{diary_count}개의 일기, "
        
        # 코드 파일 요약
        if results.get('code'):
            code_count = len(results['code'])
            languages = set(code.get('language', 'Unknown') for code in results['code'])
            languages = [lang for lang in languages if lang != 'Unknown']
            if languages:
                summary += f"{code_count}개의 코드 파일 ({', '.join(languages)}), "
            else:
                summary += f"{code_count}개의 코드 파일, "
        
        # 문서 요약
        if results.get('document'):
            doc_count = len(results['document'])
            summary += f"{doc_count}개의 문서, "
        
        # 이미지 요약
        if results.get('image'):
            img_count = len(results['image'])
            summary += f"{img_count}개의 이미지, "
        
        # 끝 쉼표 제거
        summary = summary.rstrip(", ")
        return summary if summary else "연구 자료"
    
    def _create_prompt(self, results: Dict[str, Any], persona: Dict[str, Any], 
                  title: str, time_range: Dict[str, str], 
                  focus_category: Optional[str]) -> Dict[str, str]:
        """표준화된 결과와 페르소나를 기반으로 LLM 프롬프트 생성"""
        # 현재 날짜 및 시간
        now = datetime.now().strftime("%Y-%m-%d")
        
        # 시스템 프롬프트 구성
        system_prompt = self._create_system_prompt(persona)
        
        # 사용자 프롬프트 구성 - 기본 정보
        user_prompt = self._create_base_user_prompt(title, now, time_range)
        
        # 각 데이터 유형별 내용 추가
        user_prompt += self._add_diary_content(results.get('diary_entries', []))
        user_prompt += self._add_code_content(results.get('code', []))
        user_prompt += self._add_document_content(results.get('document', []))
        user_prompt += self._add_image_content(results.get('image', []))
        
        # 중점 카테고리 안내 추가
        if focus_category:
            user_prompt += f"\n중점 카테고리: {focus_category} (이 카테고리에 중점을 두고 연구 노트 작성)\n\n"
        
        # 연구 노트 작성 지침 추가
        user_prompt += self._add_writing_guidelines(persona)
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _create_system_prompt(self, persona: Dict[str, Any]) -> str:
        """페르소나 기반 시스템 프롬프트 생성"""
        return RESEARCH_NOTE_GENERATOR_PROMPT["system"].format(
            persona_name=persona.get('name', '카이로스랩 연구자'),
            persona_gender=persona.get('gender', 'male'),
            persona_birth_date=persona.get('birth_date', '1990-01-01'),
            persona_tone=persona.get('tone', '격식 있는'),
            persona_voice=persona.get('voice', '분석적인')
        )
    
    def _create_base_user_prompt(self, title: str, now: str, time_range: Dict[str, str]) -> str:
        """기본 사용자 프롬프트 생성"""
        return RESEARCH_NOTE_GENERATOR_PROMPT["user"].format(
            title=title,
            now=now,
            time_range_start=time_range['start'],
            time_range_end=time_range['end']
        )
    
    def _add_diary_content(self, diary_entries: List[Dict[str, Any]]) -> str:
        """일기 데이터 프롬프트 추가"""
        if not diary_entries:
            return ""
            
        prompt = "## 일기 데이터\n"
        for entry in diary_entries:
            # 일기는 기본적으로 연구 자료로 간주 (직접 작성한 것이므로)
            prompt += f"- 날짜: {entry.get('date', 'N/A')}\n  내용: {entry.get('content', 'N/A')}\n  카테고리: 연구 자료 (research)\n\n"
        return prompt
    
    def _add_code_content(self, code_results: List[Dict[str, Any]]) -> str:
        """코드 분석 결과 프롬프트 추가"""
        if not code_results:
            return ""
            
        prompt = "## 코드 분석 결과\n"
        for code in code_results:
            file_info = code.get('file_info', {})
            
            # 파일 경로 기반 강제 분류
            file_path = file_info.get('file_path', '')
            original_category = code.get('category', 'Unknown')
            corrected_category = self._verify_category(file_path, original_category)
            
            # 카테고리 변경 여부 로깅
            if original_category != corrected_category:
                self.logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
                              f"원래: {original_category}, 수정: {corrected_category}")
            
            # 강제 서술 템플릿 적용
            file_name = file_info.get('file_name', 'Unknown')
            language = code.get('language', 'Unknown')
            purpose = code.get('purpose', 'N/A')
            logic = code.get('logic', 'N/A')
            
            if corrected_category == "reference":
                category_desc = "참고 자료 (reference)"
                action_desc = f"나는 이 {language} 파일을 분석했다. 이 코드는 {purpose}"
            else:
                category_desc = "연구 자료 (research)"
                action_desc = f"나는 이 {language} 파일을 개발했다. 이 코드는 {purpose}"
            
            prompt += f"- 파일: {file_name} ({language})\n"
            prompt += f"  설명: {action_desc}\n"
            prompt += f"  목적: {purpose}\n"
            prompt += f"  로직: {logic}\n"
            prompt += f"  카테고리: {category_desc}\n\n"
        
        return prompt
    
    def _add_document_content(self, document_results: List[Dict[str, Any]]) -> str:
        """문서 분석 결과 프롬프트 추가"""
        if not document_results:
            return ""
            
        prompt = "## 문서 분석 결과\n"
        for doc in document_results:
            file_info = doc.get('file_info', {})
            
            # 파일 경로 기반 강제 분류
            file_path = file_info.get('file_path', '')
            original_category = doc.get('category', 'Unknown')
            corrected_category = self._verify_category(file_path, original_category)
            
            # 카테고리 변경 여부 로깅
            if original_category != corrected_category:
                self.logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
                              f"원래: {original_category}, 수정: {corrected_category}")
            
            category_desc = "참고 자료 (reference)" if corrected_category == "reference" else "연구 자료 (research)"
            
            prompt += f"- 파일: {file_info.get('file_name', 'Unknown')}\n"
            prompt += f"  제목: {doc.get('title', 'Unknown')}\n"
            prompt += f"  목적: {doc.get('purpose', 'N/A')}\n"
            prompt += f"  요약: {doc.get('summary', 'N/A')}\n"
            prompt += f"  카테고리: {category_desc}\n\n"
        
        return prompt
    
    def _add_image_content(self, image_results: List[Dict[str, Any]]) -> str:
        """이미지 분석 결과 프롬프트 추가"""
        if not image_results:
            return ""
            
        prompt = "## 이미지 분석 결과\n"
        for img in image_results:
            file_info = img.get('file_info', {})
            
            # 파일 경로 기반 강제 분류
            file_path = file_info.get('file_path', '')
            original_category = img.get('category', 'Unknown')
            corrected_category = self._verify_category(file_path, original_category)
            
            # 카테고리 변경 여부 로깅
            if original_category != corrected_category:
                self.logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
                              f"원래: {original_category}, 수정: {corrected_category}")
            
            category_desc = "참고 자료 (reference)" if corrected_category == "reference" else "연구 자료 (research)"
            
            prompt += f"- 파일: {file_info.get('file_name', 'Unknown')}\n"
            prompt += f"  촬영 시간: {img.get('formatted_date', 'N/A')}\n"
            prompt += f"  위치: {img.get('location', 'N/A')}\n"
            prompt += f"  설명: {img.get('caption', 'N/A')}\n"
            prompt += f"  카테고리: {category_desc}\n\n"
        
        return prompt
    
    def _add_writing_guidelines(self, persona: Dict[str, Any]) -> str:
        """연구 노트 작성 지침 추가"""
        return RESEARCH_NOTE_GENERATOR_PROMPT["guideline"].format(
            persona_birth_date=persona.get('birth_date', '1990-01-01'),
            persona_tone=persona['tone'],
            persona_voice=persona['voice']
        )
    
    def _verify_category(self, file_path: str, original_category: str) -> str:
        """
        파일 경로를 기반으로 카테고리 확인 및 수정
        
        Args:
            file_path (str): 파일 경로
            original_category (str): 원래 카테고리
            
        Returns:
            str: 검증된 카테고리 ('reference' 또는 'research')
        """
        try:
            # 파일 경로가 비어있으면 원래 카테고리 반환
            if not file_path:
                self.logger.warning(f"파일 경로가 비어있어 원래 카테고리 유지: {original_category}")
                return original_category
                
            # Path 객체 사용으로 OS 독립적 경로 처리
            path = Path(file_path)
            
            # 부모 디렉토리 검사
            for parent in path.parents:
                if parent.name.lower() == 'reference':
                    self.logger.debug(f"파일이 reference 폴더에 있습니다: {file_path}")
                    return "reference"
            
            # 경로에 'reference' 디렉토리가 포함되어 있는지 확인 (대소문자 무관)
            # Path.parents가 때로는 전체 경로를 완전히 포착하지 못할 때의 백업 확인
            path_str = str(path).lower()
            if '/reference/' in path_str or '\\reference\\' in path_str:
                self.logger.debug(f"파일이 reference 폴더에 있습니다 (문자열 검사): {file_path}")
                return "reference"
                
            self.logger.debug(f"파일이 reference 폴더에 없습니다: {file_path}")
            return "research"
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"file_path": file_path, "original_category": original_category}
            )
            self.logger.error(f"카테고리 검증 중 오류 발생: {error_detail['error']}")
            # 오류 발생 시 원래 카테고리 사용
            return original_category
    
    def save_research_note(self, note: Dict[str, Any], output_path: str) -> str:
        """
        생성된 연구 노트를 마크다운 파일로 저장
        
        Args:
            note (Dict[str, Any]): 생성된 연구 노트 정보
            output_path (str): 저장할 파일 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 출력 디렉토리 생성
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 마크다운 형식으로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                # 제목
                f.write(f"# {note['title']}\n\n")
                
                # 페르소나 정보
                persona = note.get('applied_persona', {})
                f.write(f"작성자: {persona.get('name', 'N/A')}\n")
                f.write(f"작성 일시: {note['generated_at']}\n\n")
                
                # 본문 내용
                f.write(f"{note['content']}\n")
            
            self.logger.info(f"연구 노트 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            error_detail = self.error_handler.handle_error(
                e, {"output_path": output_path}
            )
            self.logger.error(f"연구 노트 저장 실패: {error_detail['error']}")
            return ""