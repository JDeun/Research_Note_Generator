import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, 
    QLineEdit, QMessageBox, QProgressBar, QGroupBox, QHBoxLayout, QRadioButton
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from parallel_processor import ParallelProcessor
from ResearchNoteGenerator import ResearchNoteGenerator
from concurrent.futures import ThreadPoolExecutor
from config import RESEARCH_NOTE_GENERATOR_MODELS, CODE_PROCESSOR_MODELS, IMAGE_PROCESSOR_MODELS
from persona import PERSONA_LIST

# 기존 코드에서 필요한 클래스 및 함수 임포트 (예: ParallelProcessor, ResearchNoteGenerator)

class Worker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, target, args=None, kwargs=None):
      super().__init__()
      self.target = target
      self.args = args if args else []
      self.kwargs = kwargs if kwargs else {}
    
    def run(self):
        try:
            result = self.target(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Research Note Generator")

        # 레이아웃 구성
        main_layout = QVBoxLayout()

        # 파일 선택 그룹
        file_group = QGroupBox("파일 선택")
        file_layout = QHBoxLayout()
        self.file_path_label = QLineEdit()
        self.file_path_label.setReadOnly(True)
        self.select_file_button = QPushButton("파일 선택")
        self.select_file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.select_file_button)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # 모델 선택 그룹
        model_group = QGroupBox("모델 선택")
        model_layout = QHBoxLayout()
        
        self.model_label = QLabel("분석 모델:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(CODE_PROCESSOR_MODELS.keys()))
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # 페르소나 선택 그룹
        persona_group = QGroupBox("페르소나 선택")
        persona_layout = QHBoxLayout()
        self.persona_combo = QComboBox()
        self.persona_combo.addItems([persona['name'] for persona in PERSONA_LIST])
        persona_layout.addWidget(self.persona_combo)
        persona_group.setLayout(persona_layout)
        main_layout.addWidget(persona_group)

        # 분석 버튼
        self.analyze_button = QPushButton("분석 시작")
        self.analyze_button.clicked.connect(self.start_analysis)
        main_layout.addWidget(self.analyze_button)
        
        # 연구 노트 생성 버튼
        self.generate_note_button = QPushButton("연구노트 생성")
        self.generate_note_button.clicked.connect(self.generate_research_note)
        self.generate_note_button.setEnabled(False)
        main_layout.addWidget(self.generate_note_button)
        
        # 연구 노트 출력 영역
        self.note_output_area = QTextEdit()
        main_layout.addWidget(self.note_output_area)
        
        # 프로그래스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 메인 위젯 설정
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 쓰레드 설정
        self.analysis_thread = QThread()
        self.generate_note_thread = QThread()
        
        self.worker = None
        self.results = None

    def select_file(self):
        """파일 선택 다이얼로그"""
        file_path, _ = QFileDialog.getOpenFileName(self, "파일 선택")
        if file_path:
            self.file_path_label.setText(file_path)

    def start_analysis(self):
        """분석 시작"""
        file_path = self.file_path_label.text()
        model_name = self.model_combo.currentText()

        if not file_path:
            QMessageBox.warning(self, "경고", "파일을 선택하세요.")
            return
            
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = Worker(self._analyze_files, args=(file_path, model_name))
        self.worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.progress.connect(self.progress_bar.setValue)

        self.analysis_thread.start()

    def _analyze_files(self, file_path, model_name):
        """실제 파일 분석 작업을 수행하는 메서드"""
        parallel_processor = ParallelProcessor(model_name)
        result = parallel_processor.process_files([file_path])
        return result

    def analysis_finished(self, result):
        """분석 완료 처리"""
        self.results = result
        self.progress_bar.setValue(100)
        self.analyze_button.setEnabled(True)
        self.generate_note_button.setEnabled(True)
        self.analysis_thread.quit()
        self.analysis_thread.wait()

    def analysis_error(self, message):
        """분석 중 오류 처리"""
        QMessageBox.critical(self, "오류", f"분석 중 오류 발생: {message}")
        self.analyze_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.analysis_thread.quit()
        self.analysis_thread.wait()

    def generate_research_note(self):
        """연구 노트 생성"""
        if not self.results:
            QMessageBox.warning(self, "경고", "분석을 먼저 진행하세요.")
            return
        
        model_name = self.model_combo.currentText()
        persona_name = self.persona_combo.currentText()
        
        self.worker = Worker(self._generate_note, args=(self.results, model_name, persona_name))
        self.worker.moveToThread(self.generate_note_thread)
        self.generate_note_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.generate_note_finished)
        self.worker.error.connect(self.generate_note_error)
        self.generate_note_thread.start()

    def _generate_note(self, results, model_name, persona_name):
        research_note_generator = ResearchNoteGenerator(model_name)
        result = research_note_generator.generate_research_note(results, persona_name)
        return result

    def generate_note_finished(self, result):
      """연구 노트 생성 완료 처리"""
      self.note_output_area.setText(result['content'])
      self.generate_note_button.setEnabled(True)
      self.generate_note_thread.quit()
      self.generate_note_thread.wait()
        

    def generate_note_error(self, message):
        """연구 노트 생성 중 오류 처리"""
        QMessageBox.critical(self, "오류", f"연구 노트 생성 중 오류 발생: {message}")
        self.generate_note_button.setEnabled(True)
        self.generate_note_thread.quit()
        self.generate_note_thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
