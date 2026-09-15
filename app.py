import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtCore, QtGui, QtWidgets

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None


OLLAMA_URL = "http://127.0.0.1:11434"
PREFERRED_MODEL = "gemma4:31b-cloud"
PREVIEW_SIZE = (760, 520)


class CaptionCameraApp(QtWidgets.QMainWindow):
    models_loaded = QtCore.Signal(list)
    models_failed = QtCore.Signal(str)
    caption_ready = QtCore.Signal(str)
    caption_failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ollama Caption Camera")
        self.resize(980, 760)
        self.setMinimumSize(820, 650)

        self.camera = None
        self.camera_running = False
        self.latest_frame = None
        self.captured_frame = None
        self.model_names = []
        self._tts_engine = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_camera)

        self.model_combo = None
        self.selected_model = ""
        self.status_label = None
        self.preview_label = None
        self.caption_edit = None

        self._build_ui()
        self.models_loaded.connect(self._set_models)
        self.models_failed.connect(lambda message: self._show_error(f"Could not reach Ollama: {message}"))
        self.caption_ready.connect(self._caption_ready)
        self.caption_failed.connect(self._caption_failed)
        self.refresh_models()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        title = QtWidgets.QLabel("Ollama Caption Camera")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        main_layout.addWidget(header)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        start_btn = QtWidgets.QPushButton("Start camera")
        start_btn.clicked.connect(self.start_camera)
        capture_btn = QtWidgets.QPushButton("Capture frame")
        capture_btn.clicked.connect(self.capture_frame)
        save_btn = QtWidgets.QPushButton("Save image")
        save_btn.clicked.connect(self.save_image)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)

        model_label = QtWidgets.QLabel("Model:")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(260)
        self.model_combo.setEnabled(True)
        self.model_combo.setEditable(False)
        self.model_combo.currentTextChanged.connect(self._model_selected)

        controls_layout.addWidget(start_btn)
        controls_layout.addWidget(capture_btn)
        controls_layout.addWidget(save_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(refresh_btn)
        main_layout.addWidget(controls)

        preview_group = QtWidgets.QGroupBox("Camera preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        self.preview_label = QtWidgets.QLabel("Camera is off")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 420)
        self.preview_label.setStyleSheet("background: #1e1e1e; border-radius: 10px;")
        preview_layout.addWidget(self.preview_label)
        main_layout.addWidget(preview_group)

        caption_group = QtWidgets.QGroupBox("Caption")
        caption_layout = QtWidgets.QVBoxLayout(caption_group)

        speak_row = QtWidgets.QHBoxLayout()
        speak_row.addStretch()
        speak_btn = QtWidgets.QPushButton("🔊 Speak")
        speak_btn.clicked.connect(self.speak_caption)
        speak_row.addWidget(speak_btn)
        caption_layout.addLayout(speak_row)

        self.caption_edit = QtWidgets.QPlainTextEdit()
        self.caption_edit.setReadOnly(True)
        self.caption_edit.setPlaceholderText("Generated caption will appear here.")
        self.caption_edit.setMinimumHeight(110)
        caption_layout.addWidget(self.caption_edit)

        generate_btn = QtWidgets.QPushButton("Generate caption")
        generate_btn.clicked.connect(self.generate_caption)
        generate_btn.setFixedWidth(170)
        generate_row = QtWidgets.QHBoxLayout()
        generate_row.addStretch()
        generate_row.addWidget(generate_btn)
        caption_layout.addLayout(generate_row)

        main_layout.addWidget(caption_group)
        self.statusBar().showMessage("Ready")

    def refresh_models(self) -> None:
        self.status_label.setText("Loading Ollama models...")

        def load() -> None:
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
                    payload = json.load(response)
                models = [item["name"] for item in payload.get("models", [])]
                self.models_loaded.emit(models)
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                self.models_failed.emit(str(error))

        threading.Thread(target=load, daemon=True).start()

    def _set_models(self, models: list[str]) -> None:
        self.model_names = models
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)

        if PREFERRED_MODEL in models:
            preferred_index = models.index(PREFERRED_MODEL)
            self.model_combo.setCurrentIndex(preferred_index)
        elif models:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)
        self.selected_model = self.model_combo.currentText().strip()

        self.status_label.setText(f"{len(models)} model(s) available")
        self.statusBar().showMessage(f"{len(models)} model(s) available")

    def _model_selected(self, model: str) -> None:
        self.selected_model = model.strip()
        if self.selected_model:
            self.status_label.setText(f"Selected: {self.selected_model}")
            self.statusBar().showMessage(f"Selected model: {self.selected_model}")

    def start_camera(self) -> None:
        if self.camera_running:
            return
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self._show_error("Could not open the default camera.")
            return
        self.camera_running = True
        self.status_label.setText("Camera running")
        self.statusBar().showMessage("Camera running")
        self.timer.start(30)

    def update_camera(self) -> None:
        if not self.camera_running or self.camera is None:
            return
        success, frame = self.camera.read()
        if success:
            self.latest_frame = frame
            self._show_frame(frame)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        qt_image = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def capture_frame(self) -> None:
        if self.latest_frame is None:
            self._show_error("Start the camera before capturing a frame.")
            return
        self.captured_frame = self.latest_frame.copy()
        self._show_frame(self.captured_frame)
        self.status_label.setText("Frame captured")
        self.statusBar().showMessage("Frame captured")

    def save_image(self) -> None:
        if self.captured_frame is None:
            self._show_error("Capture a frame before saving.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save captured image",
            "captured_frame.jpg",
            "JPEG (*.jpg);;PNG (*.png)",
        )
        if path:
            cv2.imwrite(path, self.captured_frame)
            self.status_label.setText(f"Saved {Path(path).name}")
            self.statusBar().showMessage(f"Saved {Path(path).name}")

    def generate_caption(self) -> None:
        if self.captured_frame is None:
            self._show_error("Capture a frame before generating a caption.")
            return

        model = self.selected_model or self.model_combo.currentText().strip()
        if not model:
            self._show_error("Choose an Ollama model first.")
            return

        success, encoded = cv2.imencode(".jpg", self.captured_frame)
        if not success:
            self._show_error("Could not encode the captured frame.")
            return

        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self._set_caption("Generating caption...")
        self.status_label.setText(f"Asking {model}...")
        self.statusBar().showMessage(f"Asking {model}...")

        def generate() -> None:
            request_data = json.dumps(
                {
                    "model": model,
                    "prompt": "Describe this image in one concise, natural sentence.",
                    "images": [image_data],
                    "stream": False,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=request_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                caption = result.get("response", "").strip() or "The model returned an empty caption."
                self.caption_ready.emit(caption)
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                self.caption_failed.emit(str(error))

        threading.Thread(target=generate, daemon=True).start()

    def _set_caption(self, text: str) -> None:
        self.caption_edit.setReadOnly(False)
        self.caption_edit.setPlainText(text)
        self.caption_edit.setReadOnly(True)

    def _caption_ready(self, caption: str) -> None:
        self._set_caption(caption)
        self.status_label.setText("Caption ready")
        self.statusBar().showMessage("Caption ready")

    def _caption_failed(self, error: str) -> None:
        self._set_caption(f"Caption failed: {error}")
        self.status_label.setText("Caption failed")
        self.statusBar().showMessage("Caption failed")

    def speak_caption(self) -> None:
        text = self.caption_edit.toPlainText().strip()
        if not text:
            self._show_error("Generate a caption before speaking it.")
            return

        try:
            if pyttsx3 is not None:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                self.status_label.setText("Caption spoken")
                self.statusBar().showMessage("Caption spoken")
                return

            from win32com.client import Dispatch

            speaker = Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            self.status_label.setText("Caption spoken")
            self.statusBar().showMessage("Caption spoken")
        except Exception as error:  # pragma: no cover
            self._show_error(f"Text-to-speech is unavailable: {error}")

    def _show_error(self, message: str) -> None:
        self.status_label.setText("Error")
        self.statusBar().showMessage("Error")
        QtWidgets.QMessageBox.critical(self, "Ollama Caption Camera", message)

    def closeEvent(self, event) -> None:
        self.camera_running = False
        self.timer.stop()

        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass
            self._tts_engine = None

        self.latest_frame = None
        self.captured_frame = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = CaptionCameraApp()
    window.show()
    app.exec_()
