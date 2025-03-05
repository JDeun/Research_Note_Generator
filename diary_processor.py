# diary_processor.py
import os
import re
import logging
from typing import Dict, Any
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiaryProcessor:
    """
    개인 일기/메모 처리 클래스
    - YYMMDD_summary.txt 형식의 파일 처리
    - 날짜별 활동 내용을 딕셔너리로 저장
    """
    
    def __init__(self):
        """DiaryProcessor 초기화"""
        self.diaries = {}  # 날짜를 키로, 내용을 값으로 저장
    
    def process_diary(self, file_path: str) -> Dict[str, Any]:
        """
        일기 파일 처리
        
        Args:
            file_path (str): 일기 파일 경로 (YYMMDD_summary.txt 형식)
            
        Returns:
            Dict[str, Any]: 처리 결과
            {
                "날짜1": "내용1",
                "날짜2": "내용2",
                ...
            }
            또는 에러 발생 시:
            {
                "error": "에러 메시지"
            }
        """
        try:
            # 파일명에서 날짜 추출
            file_name = os.path.basename(file_path)
            date_match = re.match(r'(\d{6})_summary\.txt', file_name)
            
            if not date_match:
                logger.warning(f"일기 파일 형식이 맞지 않습니다: {file_name}")
                return {
                    "error": "파일명 형식 불일치"
                }
            
            date_str = date_match.group(1)
            
            # 파일 내용 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 날짜를 키로, 내용을 값으로 저장
            self.diaries[date_str] = content
            
            logger.info(f"일기 파일 처리 완료: {file_name}")
            
            # 처리된 일기 데이터 반환
            return {date_str: content}
            
        except Exception as e:
            logger.error(f"일기 파일 처리 중 오류 발생: {str(e)}")
            return {
                "error": str(e)
            }
    
    def get_all_diaries(self) -> Dict[str, str]:
        """
        모든 일기 데이터 반환
        
        Returns:
            Dict[str, str]: 날짜를 키로, 내용을 값으로 하는 딕셔너리
        """
        return self.diaries

# 테스트 코드
if __name__ == "__main__":
    processor = DiaryProcessor()
    
    # 테스트용 경로 입력
    test_dir = input("일기 파일이 있는 디렉토리 경로를 입력하세요: ").strip()
    
    if not os.path.exists(test_dir):
        print(f"경로가 존재하지 않습니다: {test_dir}")
    else:
        count = 0
        all_diaries = {}
        
        for file in os.listdir(test_dir):
            if re.match(r'\d{6}_summary\.txt', file):
                file_path = os.path.join(test_dir, file)
                result = processor.process_diary(file_path)
                
                if "error" in result:
                    print(f"오류 ({file}): {result['error']}")
                else:
                    # 결과를 all_diaries에 병합
                    all_diaries.update(result)
                    count += 1
        
        if count == 0:
            print("일기 파일을 찾을 수 없습니다. 파일명은 'YYMMDD_summary.txt' 형식이어야 합니다.")
        else:
            print(f"\n총 {count}개의 일기 파일을 처리했습니다.")
            print("\n처리된 일기 데이터:")
            for date, content in all_diaries.items():
                print(f"\n{date}: {content[:50]}..." if len(content) > 50 else f"\n{date}: {content}")
            
            # 전체 일기 데이터 가져오기
            all_data = processor.get_all_diaries()
            print(f"\n전체 일기 항목 수: {len(all_data)}")