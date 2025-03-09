import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks import get_openai_callback

from config import RESEARCH_NOTE_GENERATOR_MODELS, TEMPERATURES, LLM_API_KEY
from persona_manager import PERSONA_LIST
from ProcessorPrompt import RESEARCH_NOTE_GENERATOR_PROMPT, RESEARCH_NOTE_TITLE_GENERATOR_PROMPT

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # LLM 모델 설정
        self._setup_llm()
    
    def _setup_llm(self):
        """LLM 모델 설정"""
        if self.model_name not in RESEARCH_NOTE_GENERATOR_MODELS:
            logger.error(f"지원하지 않는 모델: {self.model_name}")
            self.llm = None
            return

        api_model, model_class = RESEARCH_NOTE_GENERATOR_MODELS[self.model_name]
        api_key = LLM_API_KEY.get(self.model_name)

        if not api_key:
            logger.error(f"API 키 누락: {self.model_name} API 키를 .env에 설정하세요.")
            self.llm = None
            return

        try:
            self.llm = model_class(api_key=api_key, model=api_model, temperature=self.temperature)
            logger.info(f"{self.model_name} 모델 초기화 성공")
        except Exception as e:
            logger.error(f"{self.model_name} 모델 초기화 실패: {str(e)}")
            self.llm = None
    
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
            logger.error("LLM 모델이 초기화되지 않았습니다")
            return {"error": "LLM 모델 초기화 실패"}
        
        # 페르소나 선택 (지정되지 않은 경우 랜덤)
        persona = self._select_persona(persona_name)
        logger.info(f"선택된 페르소나: {persona['name']} (어조: {persona['tone']}, 음성: {persona['voice']})")
        
        # 시간 범위 추출
        time_range = self._extract_time_range(results)
        
        # 제목이 없는 경우 자동 생성
        if not title:
            title = self._generate_title(results, time_range)
        
        # 연구노트 생성을 위한 프롬프트 생성
        prompt = self._create_prompt(results, persona, title, time_range, focus_category)
        
        try:
            # LLM 호출
            system_message = SystemMessage(content=prompt["system"])
            user_message = HumanMessage(content=prompt["user"])
            
            if self.model_name in ['openai', 'gemini']:
                with get_openai_callback() as cb:
                    response = self.llm.invoke([system_message, user_message])
                    logger.info(f"{self.model_name} 토큰 사용량: {cb.total_tokens}")
            else:
                response = self.llm.invoke([system_message, user_message])
            
            # 결과 구성
            result = {
                'title': title,
                'content': response.content,
                'applied_persona': persona,
                'generated_at': datetime.now().isoformat(),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"연구노트 생성 실패: {str(e)}")
            return {"error": str(e)}
    
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
        if persona_name:
            for persona in PERSONA_LIST:
                if persona["name"] == persona_name:
                    return persona
            logger.warning(f"'{persona_name}' 페르소나를 찾을 수 없습니다. 랜덤 선택합니다.")
        
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
            
            response = self.llm.invoke([system_message, user_message])
            title = response.content.strip().replace('"', '').replace("'", "")
            
            # 제목이 너무 길면 잘라내기
            if len(title) > 50:
                title = title[:47] + "..."
            
            return title
            
        except Exception as e:
            logger.error(f"제목 생성 실패: {str(e)}")
            # 기본 제목 생성
            return f"연구 노트: {time_range['start']} ~ {time_range['end']}"
    
    def _create_prompt(self, results: Dict[str, Any], persona: Dict[str, Any], 
                  title: str, time_range: Dict[str, str], 
                  focus_category: Optional[str]) -> Dict[str, str]:
        """표준화된 결과와 페르소나를 기반으로 LLM 프롬프트 생성"""
        # 현재 날짜 및 시간
        now = datetime.now().strftime("%Y-%m-%d")
        
        # 시스템 프롬프트 구성 (ProcessorPrompt.py에서 가져옴)
        system_prompt = RESEARCH_NOTE_GENERATOR_PROMPT["system"].format(
            persona_name=persona['name'],
            persona_birth_date=persona.get('birth_date', '1990-01-01'),
            persona_tone=persona['tone'],
            persona_voice=persona['voice']
        )

        # 사용자 프롬프트 구성 (ProcessorPrompt.py에서 가져옴)
        user_prompt = RESEARCH_NOTE_GENERATOR_PROMPT["user"].format(
            title=title,
            now=now,
            time_range_start=time_range['start'],
            time_range_end=time_range['end']
        )
        
        # 일기 데이터 추가
        if results.get('diary_entries'):
            user_prompt += "## 일기 데이터\n"
            for entry in results['diary_entries']:
                # 일기는 기본적으로 연구 자료로 간주 (직접 작성한 것이므로)
                user_prompt += f"- 날짜: {entry.get('date', 'N/A')}\n  내용: {entry.get('content', 'N/A')}\n  카테고리: 연구 자료 (research)\n\n"
        
        # 코드 분석 결과 추가 - 카테고리 일관성 검사 및 강제 분류 추가
        if results.get('code'):
            user_prompt += "## 코드 분석 결과\n"
            for code in results['code']:
                file_info = code.get('file_info', {})
                
                # 파일 경로 기반 강제 분류
                file_path = file_info.get('file_path', '')
                original_category = code.get('category', 'Unknown')
                corrected_category = self._verify_category(file_path, original_category)
                
                # 카테고리 변경 여부 로깅
                if original_category != corrected_category:
                    logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
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
                
                user_prompt += f"- 파일: {file_name} ({language})\n"
                user_prompt += f"  설명: {action_desc}\n"
                user_prompt += f"  목적: {purpose}\n"
                user_prompt += f"  로직: {logic}\n"
                user_prompt += f"  카테고리: {category_desc}\n\n"
        
        # 문서 분석 결과 추가 - 카테고리 일관성 검사 및 강제 분류 추가
        if results.get('document'):
            user_prompt += "## 문서 분석 결과\n"
            for doc in results['document']:
                file_info = doc.get('file_info', {})
                
                # 파일 경로 기반 강제 분류
                file_path = file_info.get('file_path', '')
                original_category = doc.get('category', 'Unknown')
                corrected_category = self._verify_category(file_path, original_category)
                
                # 카테고리 변경 여부 로깅
                if original_category != corrected_category:
                    logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
                                f"원래: {original_category}, 수정: {corrected_category}")
                
                category_desc = "참고 자료 (reference)" if corrected_category == "reference" else "연구 자료 (research)"
                
                user_prompt += f"- 파일: {file_info.get('file_name', 'Unknown')}\n"
                user_prompt += f"  제목: {doc.get('title', 'Unknown')}\n"
                user_prompt += f"  목적: {doc.get('purpose', 'N/A')}\n"
                user_prompt += f"  요약: {doc.get('summary', 'N/A')}\n"
                user_prompt += f"  카테고리: {category_desc}\n\n"
        
        # 이미지 분석 결과 추가 - 카테고리 일관성 검사 및 강제 분류 추가
        if results.get('image'):
            user_prompt += "## 이미지 분석 결과\n"
            for img in results['image']:
                file_info = img.get('file_info', {})
                
                # 파일 경로 기반 강제 분류
                file_path = file_info.get('file_path', '')
                original_category = img.get('category', 'Unknown')
                corrected_category = self._verify_category(file_path, original_category)
                
                # 카테고리 변경 여부 로깅
                if original_category != corrected_category:
                    logger.warning(f"카테고리 불일치 감지 및 수정: {file_info.get('file_name', 'Unknown')} - "
                                f"원래: {original_category}, 수정: {corrected_category}")
                
                category_desc = "참고 자료 (reference)" if corrected_category == "reference" else "연구 자료 (research)"
                
                user_prompt += f"- 파일: {file_info.get('file_name', 'Unknown')}\n"
                user_prompt += f"  촬영 시간: {img.get('formatted_date', 'N/A')}\n"
                user_prompt += f"  위치: {img.get('location', 'N/A')}\n"
                user_prompt += f"  설명: {img.get('caption', 'N/A')}\n"
                user_prompt += f"  카테고리: {category_desc}\n\n"
        
        # 중점 카테고리 안내
        if focus_category:
            user_prompt += f"\n중점 카테고리: {focus_category} (이 카테고리에 중점을 두고 연구 노트 작성)\n\n"
        
        # 연구 노트 작성 지침 추가
        user_prompt += RESEARCH_NOTE_GENERATOR_PROMPT["guideline"].format(
            persona_birth_date=persona.get('birth_date', '1990-01-01'),
            persona_tone=persona['tone'],
            persona_voice=persona['voice']
        )
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
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
                logger.warning(f"파일 경로가 비어있어 원래 카테고리 유지: {original_category}")
                return original_category
                
            # Windows와 Unix 경로 구분자 처리
            normalized_path = file_path.replace('\\', '/')
            logger.debug(f"카테고리 검증 중: {normalized_path}")
            
            # 'reference' 폴더 포함 여부 확인
            if '/reference/' in normalized_path or '\\reference\\' in file_path:
                logger.debug(f"파일이 reference 폴더에 있습니다: {file_path}")
                return "reference"
            else:
                logger.debug(f"파일이 reference 폴더에 없습니다: {file_path}")
                return "research"
        except Exception as e:
            logger.error(f"카테고리 검증 중 오류 발생: {str(e)}")
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
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
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
            
            logger.info(f"연구 노트 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"연구 노트 저장 실패: {str(e)}")
            return ""