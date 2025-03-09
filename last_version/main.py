import os
import logging
import argparse
from datetime import datetime

from file_handler import FileHandler
from parallel_processor import ParallelProcessor
from ResearchNoteGenerator import ResearchNoteGenerator
from persona_manager import PERSONA_LIST

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_files(input_path: str, model_name: str) -> dict:
    """파일 처리 프로세스"""
    logger.info(f"입력 폴더 처리 시작: {input_path}")
    
    # 파일 분류
    logger.info("파일 분류 중...")
    file_handler = FileHandler()
    classified_files = file_handler.get_files(input_path)
    
    # 분류 결과 출력
    total_files = sum(len(files) for files in classified_files.values())
    logger.info(f"총 {total_files}개 파일이 분류되었습니다:")
    for file_type, files in classified_files.items():
        logger.info(f"  - {file_type.upper()}: {len(files)}개")
    
    # 파일 내용 처리
    logger.info(f"{model_name.upper()} 모델로 파일 처리 중...")
    processor = ParallelProcessor(model_name)
    results = processor.process_files(classified_files)
    
    # 처리 결과 출력
    logger.info("처리 완료된 파일:")
    logger.info(f"  - 이미지: {len(results['image'])}개")
    logger.info(f"  - 코드: {len(results['code'])}개")
    logger.info(f"  - 문서: {len(results['document'])}개")
    logger.info(f"  - 일기: {len(results['diary_entries'])}개")
    
    return results

def generate_research_notes(results: dict, model_name: str, output_dir: str):
    """모든 페르소나에 대해 연구 노트 생성"""
    logger.info("모든 페르소나에 대한 연구 노트 생성 시작...")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 연구 노트 생성기 초기화
    generator = ResearchNoteGenerator(model_name)
    
    # 타임스탬프 생성 (파일명 구분용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 각 페르소나별로 연구 노트 생성
    successful_notes = 0
    for persona in PERSONA_LIST:
        persona_name = persona['name']
        logger.info(f"{persona_name} 페르소나의 연구 노트 생성 중...")
        
        try:
            # 연구 노트 생성
            note = generator.generate_research_note(
                results=results,
                persona_name=persona_name
            )
            
            if "error" in note:
                logger.error(f"{persona_name} 연구 노트 생성 실패: {note['error']}")
                continue
            
            # 파일명 생성 (페르소나 이름 포함)
            safe_name = persona_name.replace(' ', '_')
            md_path = os.path.join(output_dir, f"research_note_{safe_name}_{timestamp}.md")
            
            # 연구 노트 저장
            saved_path = generator.save_research_note(note, md_path)
            
            if saved_path:
                logger.info(f"{persona_name} 연구 노트 저장 완료: {saved_path}")
                successful_notes += 1
            else:
                logger.error(f"{persona_name} 연구 노트 저장 실패")
                
        except Exception as e:
            logger.error(f"{persona_name} 연구 노트 생성 중 오류 발생: {str(e)}")
    
    logger.info(f"연구 노트 생성 완료: 총 {successful_notes}/{len(PERSONA_LIST)}개 생성됨")
    return successful_notes

def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='연구 노트 생성 시스템')
    
    parser.add_argument('--input', '-i', type=str, help='입력 폴더 경로')
    parser.add_argument('--output', '-o', type=str, default='./output', help='출력 폴더 경로')
    parser.add_argument('--model', '-m', type=str, choices=['openai', 'gemini', 'claude', 'groq'], help='사용할 LLM 모델')
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 인터랙티브 모드 또는 커맨드 라인 모드 결정
    interactive_mode = not (args.input and args.model)
    
    if interactive_mode:
        print("\n====== 연구 노트 생성 시스템 ======")
        
        # 모델 선택
        valid_models = ["openai", "gemini", "claude", "groq"]
        while True:
            model_name = input("\n사용할 모델을 선택하세요 (openai/gemini/claude/groq): ").strip().lower()
            if model_name in valid_models:
                break
            print("올바른 모델명을 입력하세요.")
        
        # 입력 폴더 경로
        input_path = input("\n처리할 폴더 경로를 입력하세요: ").strip()
        
        # 출력 폴더 경로
        output_dir = input("\n결과 저장 디렉토리 (기본값: ./output): ").strip() or "./output"
        
    else:
        # 커맨드 라인 모드
        model_name = args.model
        input_path = args.input
        output_dir = args.output
    
    # 파일 처리
    print(f"\n{input_path} 폴더의 파일 처리 중...")
    results = process_files(input_path, model_name)
    
    # 모든 페르소나에 대해 연구노트 생성
    print("\n모든 페르소나에 대한 연구 노트 생성 중...")
    successful_notes = generate_research_notes(results, model_name, output_dir)
    
    # 최종 결과 출력
    if successful_notes > 0:
        print(f"\n✅ {len(PERSONA_LIST)}명의 페르소나 중 {successful_notes}명에 대한 연구 노트가 생성되었습니다.")
        print(f"저장 위치: {os.path.abspath(output_dir)}")
    else:
        print("\n❌ 연구 노트 생성에 실패했습니다. 로그를 확인하세요.")

if __name__ == "__main__":
    main()