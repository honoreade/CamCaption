import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
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
    speech_finished = QtCore.Signal(str)
    speech_failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CamCaption")
        self.resize(980, 760)
        self.setMinimumSize(820, 650)

        self.camera = None
        self.camera_running = False
        self.latest_frame = None
        self.captured_frame = None
        self.model_names = []
        self._tts_engine = None
        self._speech_thread = None
        self._speech_active = False
        self._caption_request_active = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_camera)

        self.model_combo = None
        self.selected_model = ""
        self.status_label = None
        self.preview_label = None
        self.preview_group = None
        self.captured_preview_label = None
        self.caption_edit = None
        self.caption_edit_container = None
        self.header_widget = None
        self.controls_widget = None
        self.caption_group = None
        self.main_layout = None
        self.fullscreen_splitter = None
        self._fullscreen_layout = False
        self._layout_transition = None
        self.result_layout = None
        self.fullscreen_shortcut = None
        self.exit_fullscreen_shortcut = None

        self._build_ui()
        self.fullscreen_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self.exit_fullscreen_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        self.exit_fullscreen_shortcut.activated.connect(self._exit_fullscreen)
        self.models_loaded.connect(self._set_models)
        self.models_failed.connect(lambda message: self._show_error(f"Could not reach Ollama: {message}"))
        self.caption_ready.connect(self._caption_ready)
        self.caption_failed.connect(self._caption_failed)
        self.speech_finished.connect(self._speech_finished)
        self.speech_failed.connect(self._speech_failed)
        self.refresh_models()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        self.main_layout = main_layout
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        header = QtWidgets.QWidget()
        self.header_widget = header
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        title = QtWidgets.QLabel("CamCaption")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        main_layout.addWidget(header)

        controls = QtWidgets.QWidget()
        self.controls_widget = controls
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        self.start_btn = QtWidgets.QPushButton("Start camera")
        self.start_btn.clicked.connect(self.toggle_camera)
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

        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(capture_btn)
        controls_layout.addWidget(save_btn)
        controls_layout.addStretch()
        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(refresh_btn)
        main_layout.addWidget(controls)

        self.preview_group = QtWidgets.QGroupBox("Camera preview")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_group)
        self.preview_label = QtWidgets.QLabel("Camera is off")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 420)
        self.preview_label.setStyleSheet("background: #1e1e1e; border-radius: 10px;")
        preview_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.preview_group)

        self.caption_group = QtWidgets.QGroupBox("Caption")
        self.caption_group.installEventFilter(self)
        caption_layout = QtWidgets.QVBoxLayout(self.caption_group)

        self.speak_btn = QtWidgets.QToolButton(self.caption_group)
        self.speak_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaVolume))
        self.speak_btn.setIconSize(QtCore.QSize(20, 20))
        self.speak_btn.setFixedSize(42, 34)
        self.speak_btn.setToolTip("Speak caption")
        self.speak_btn.setAccessibleName("Speak caption")
        self.speak_btn.clicked.connect(self.speak_caption)
        self.speak_btn.setVisible(False)

        result_row = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
        self.result_layout = result_row
        result_row.setSpacing(12)

        self.caption_edit_container = QtWidgets.QFrame()
        caption_edit_layout = QtWidgets.QGridLayout(self.caption_edit_container)
        caption_edit_layout.setContentsMargins(0, 0, 0, 0)
        self.caption_edit = QtWidgets.QPlainTextEdit()
        self.caption_edit.setReadOnly(True)
        self.caption_edit.setMinimumSize(0, 0)
        self.caption_edit.setMaximumHeight(220)
        self.caption_edit.setMaximumWidth(600)
        self.caption_edit.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        caption_edit_layout.addWidget(self.caption_edit, 0, 0)
        self.copy_btn = QtWidgets.QToolButton(self.caption_edit_container)
        self.copy_btn.setText("⧉")
        self.copy_btn.setToolTip("Copy caption")
        self.copy_btn.setAccessibleName("Copy caption")
        self.copy_btn.setFixedSize(38, 34)
        self.copy_btn.clicked.connect(self.copy_caption)
        self.copy_btn.setVisible(False)
        caption_edit_layout.addWidget(self.copy_btn, 0, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
        self.caption_edit_container.setVisible(False)
        result_row.addWidget(self.caption_edit_container, 1, QtCore.Qt.AlignVCenter)

        self.captured_preview_label = QtWidgets.QLabel("Captured preview")
        self.captured_preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.captured_preview_label.setMinimumSize(180, 110)
        self.captured_preview_label.setMaximumSize(320, 190)
        self.captured_preview_label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.captured_preview_label.setStyleSheet("background: #1e1e1e; border-radius: 8px;")
        result_row.addWidget(self.captured_preview_label, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        caption_layout.addLayout(result_row, 1)

        generate_btn = QtWidgets.QPushButton("Generate caption")
        self.generate_btn = generate_btn
        generate_btn.clicked.connect(self.generate_caption)
        generate_btn.setFixedWidth(170)
        generate_row = QtWidgets.QHBoxLayout()
        generate_row.addStretch()
        generate_row.addWidget(generate_btn)
        caption_layout.addLayout(generate_row)

        main_layout.addWidget(self.caption_group)
        self.statusBar().showMessage("Ready")

        self.fullscreen_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.fullscreen_splitter.setChildrenCollapsible(False)
        self.fullscreen_splitter.setHandleWidth(8)
        self.fullscreen_splitter.addWidget(self.preview_group)
        self.fullscreen_splitter.addWidget(self.caption_group)
        self.fullscreen_splitter.setStretchFactor(0, 3)
        self.fullscreen_splitter.setStretchFactor(1, 2)
        self.fullscreen_splitter.hide()
        main_layout.addWidget(self.preview_group)
        main_layout.addWidget(self.caption_group)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview_group is None:
            return

        fullscreen_layout = self.isMaximized() or self.isFullScreen()
        if fullscreen_layout != self._fullscreen_layout:
            self._set_responsive_layout(fullscreen_layout)
        self._update_preview_height(fullscreen_layout)

        if self.latest_frame is not None:
            self._show_frame(self.latest_frame)
        if self.captured_frame is not None:
            self._show_captured_preview()
        self._position_speaker_button()
        self._position_copy_button()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.caption_group and event.type() == QtCore.QEvent.Resize:
            if self.captured_frame is not None:
                QtCore.QTimer.singleShot(0, self._show_captured_preview)
        return super().eventFilter(watched, event)

    def _position_speaker_button(self) -> None:
        if self.caption_edit_container is None or self.speak_btn is None:
            return
        box_top_left = self.caption_edit.mapTo(self.caption_group, QtCore.QPoint(0, 0))
        self.speak_btn.move(
            box_top_left.x() + self.caption_edit.width() - self.speak_btn.width() // 2,
            box_top_left.y() - self.speak_btn.height() // 2,
        )
        self.speak_btn.raise_()

    def _position_copy_button(self) -> None:
        if self.copy_btn is not None:
            self.copy_btn.raise_()

    def _update_preview_height(self, expanded_layout: bool) -> None:
        if expanded_layout:
            self.preview_group.setMaximumHeight(16777215)
            return

        available_height = self.centralWidget().height()
        preview_height = max(300, min(520, int(available_height * 0.48)))
        self.preview_label.setMinimumHeight(0)
        self.preview_group.setMaximumHeight(preview_height)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange and self.preview_group is not None:
            QtCore.QTimer.singleShot(0, self._sync_fullscreen_layout)

    def _sync_fullscreen_layout(self) -> None:
        fullscreen_layout = self.isMaximized() or self.isFullScreen()
        if fullscreen_layout != self._fullscreen_layout:
            self._set_responsive_layout(fullscreen_layout)

    def _set_responsive_layout(self, fullscreen_layout: bool) -> None:
        self._fullscreen_layout = fullscreen_layout

        self.main_layout.removeWidget(self.header_widget)
        self.main_layout.removeWidget(self.controls_widget)
        self.main_layout.removeWidget(self.preview_group)
        self.main_layout.removeWidget(self.caption_group)
        self.main_layout.removeWidget(self.fullscreen_splitter)

        if fullscreen_layout:
            self.result_layout.setDirection(QtWidgets.QBoxLayout.TopToBottom)
            self._set_result_order(preview_first=True)
            self._set_caption_result_size_mode(expanded=True)
            self.fullscreen_splitter.addWidget(self.preview_group)
            self.fullscreen_splitter.addWidget(self.caption_group)
            self.main_layout.addWidget(self.header_widget)
            self.main_layout.addWidget(self.controls_widget)
            self.main_layout.addWidget(self.fullscreen_splitter, 1)
            self.preview_label.setMinimumHeight(0)
            self.preview_group.setMaximumHeight(16777215)
            self.preview_group.setMinimumWidth(0)
            self.caption_group.setMinimumWidth(0)
            self.fullscreen_splitter.show()
            self._apply_caption_panel_visibility()
            QtCore.QTimer.singleShot(0, self._set_fullscreen_split_sizes)
            QtCore.QTimer.singleShot(0, self._apply_caption_panel_visibility)
        else:
            self.fullscreen_splitter.hide()
            self.caption_group.show()
            self.result_layout.setDirection(QtWidgets.QBoxLayout.LeftToRight)
            self._set_result_order(preview_first=False)
            self._set_caption_result_size_mode(expanded=False)
            self.main_layout.addWidget(self.header_widget)
            self.main_layout.addWidget(self.controls_widget)
            self.main_layout.addWidget(self.preview_group)
            self.main_layout.addWidget(self.caption_group)
            self.preview_label.setMinimumHeight(0)

        self._animate_layout_transition()
        if self.captured_frame is not None:
            QtCore.QTimer.singleShot(0, self._show_captured_preview)
            QtCore.QTimer.singleShot(80, self._show_captured_preview)

    def _set_result_order(self, preview_first: bool) -> None:
        self.result_layout.removeWidget(self.caption_edit_container)
        self.result_layout.removeWidget(self.captured_preview_label)
        if preview_first:
            self.result_layout.addWidget(self.captured_preview_label, 1)
            self.result_layout.addWidget(self.caption_edit_container, 1)
        else:
            self.result_layout.addWidget(self.caption_edit_container, 1, QtCore.Qt.AlignVCenter)
            self.result_layout.addWidget(
                self.captured_preview_label,
                0,
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
            )

    def _set_caption_result_size_mode(self, expanded: bool) -> None:
        if expanded:
            self.captured_preview_label.setMinimumSize(0, 0)
            self.captured_preview_label.setMaximumSize(16777215, 16777215)
            self.captured_preview_label.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self.caption_edit.setMaximumHeight(16777215)
            self.caption_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self.result_layout.setStretch(0, 1)
            self.result_layout.setStretch(1, 1)
        else:
            self.captured_preview_label.setMinimumSize(180, 110)
            self.captured_preview_label.setMaximumSize(320, 190)
            self.captured_preview_label.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred,
            )
            self.caption_edit.setMaximumHeight(220)
            self.caption_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred,
                QtWidgets.QSizePolicy.Preferred,
            )
            self.result_layout.setStretch(0, 1)
            self.result_layout.setStretch(1, 0)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()

    def _set_fullscreen_split_sizes(self) -> None:
        if not self._fullscreen_layout or self.fullscreen_splitter is None:
            return
        available_width = self.fullscreen_splitter.width()
        if self._fullscreen_layout and self.captured_frame is None:
            self.fullscreen_splitter.setSizes([available_width, 0])
        else:
            self.fullscreen_splitter.setSizes([available_width * 7 // 10, available_width * 3 // 10])
        QtCore.QTimer.singleShot(0, self._position_speaker_button)

    def _apply_caption_panel_visibility(self) -> None:
        if not self._fullscreen_layout or self.captured_frame is not None:
            self.caption_group.show()
        else:
            self.caption_group.hide()
            self._set_fullscreen_split_sizes()

    def _animate_layout_transition(self) -> None:
        effect = QtWidgets.QGraphicsOpacityEffect(self.centralWidget())
        self.centralWidget().setGraphicsEffect(effect)
        animation = QtCore.QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(220)
        animation.setStartValue(0.72)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        animation.finished.connect(lambda: self.centralWidget().setGraphicsEffect(None))
        self._layout_transition = animation
        animation.start()

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
        self.camera = None
        first_frame = None
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            candidate = cv2.VideoCapture(0, backend)
            if not candidate.isOpened():
                candidate.release()
                continue
            success, frame = candidate.read()
            if success:
                self.camera = candidate
                first_frame = cv2.flip(frame, 1)
                break
            candidate.release()

        if self.camera is None or first_frame is None:
            self._show_error("Could not open the default camera.")
            return
        self.latest_frame = first_frame
        self.camera_running = True
        self.start_btn.setText("Stop camera")
        self.status_label.setText("Camera running")
        self.statusBar().showMessage("Camera running")
        self._show_frame(first_frame)
        self.timer.start(30)

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def stop_camera(self) -> None:
        self.camera_running = False
        self.timer.stop()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        self.start_btn.setText("Start camera")
        self.status_label.setText("Camera stopped")
        self.statusBar().showMessage("Camera stopped")

    def update_camera(self) -> None:
        if not self.camera_running or self.camera is None:
            return
        success, frame = self.camera.read()
        if success:
            self.latest_frame = cv2.flip(frame, 1)
            self._show_frame(self.latest_frame)

    def _show_frame(self, frame) -> None:
        self.preview_label.setPixmap(self._pixmap_for_frame(frame, self.preview_label.size()))

    def _show_captured_preview(self) -> None:
        if self.captured_frame is not None and self.captured_preview_label is not None:
            self.captured_preview_label.setPixmap(
                self._pixmap_for_frame(
                    self.captured_frame,
                    self.captured_preview_label.size(),
                    fit_height=self._fullscreen_layout,
                )
            )

    def _pixmap_for_frame(self, frame, target_size, caption="", fill=False, fit_height=False) -> QtGui.QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        qt_image = QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        if target_size.width() <= 0 or target_size.height() <= 0:
            return pixmap
        if fit_height:
            aspect_ratio = pixmap.width() / pixmap.height()
            height = target_size.height()
            width = max(1, round(height * aspect_ratio))
            if width > target_size.width():
                width = target_size.width()
                height = max(1, round(width / aspect_ratio))
            preview_pixmap = pixmap.scaled(
                QtCore.QSize(width, height),
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        else:
            scale_mode = QtCore.Qt.KeepAspectRatioByExpanding if fill else QtCore.Qt.KeepAspectRatio
            preview_pixmap = pixmap.scaled(target_size, scale_mode, QtCore.Qt.SmoothTransformation)
        if fill and preview_pixmap.size() != target_size:
            crop_x = max(0, (preview_pixmap.width() - target_size.width()) // 2)
            crop_y = max(0, (preview_pixmap.height() - target_size.height()) // 2)
            preview_pixmap = preview_pixmap.copy(
                crop_x,
                crop_y,
                target_size.width(),
                target_size.height(),
            )
        if caption:
            painter = QtGui.QPainter(preview_pixmap)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
            caption_rect = QtCore.QRect(
                0,
                max(0, preview_pixmap.height() - 104),
                preview_pixmap.width(),
                104,
            )
            painter.fillRect(caption_rect, QtGui.QColor(0, 0, 0, 185))
            painter.setPen(QtGui.QColor("white"))
            painter.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold))
            text_rect = caption_rect.adjusted(18, 12, -18, -12)
            painter.drawText(text_rect, QtCore.Qt.TextWordWrap | QtCore.Qt.AlignVCenter, caption)
            painter.end()
        return preview_pixmap

    def capture_frame(self) -> None:
        if self.latest_frame is None:
            self._show_error("Start the camera before capturing a frame.")
            return
        self.captured_frame = self.latest_frame.copy()
        self.caption_edit.clear()
        self.caption_edit_container.setVisible(False)
        self.speak_btn.setVisible(False)
        self.copy_btn.setVisible(False)
        self._show_captured_preview()
        if self._fullscreen_layout:
            self._reveal_fullscreen_caption()
        self.status_label.setText("Frame captured")
        self.statusBar().showMessage("Frame captured")

    def _reveal_fullscreen_caption(self) -> None:
        if self.fullscreen_splitter is None:
            return
        self.caption_group.show()
        effect = QtWidgets.QGraphicsOpacityEffect(self.caption_group)
        self.caption_group.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        animation = QtCore.QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(280)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        animation.finished.connect(lambda: self.caption_group.setGraphicsEffect(None))
        self._layout_transition = animation
        animation.start()
        available_width = self.fullscreen_splitter.width()
        self.fullscreen_splitter.setSizes([available_width * 7 // 10, available_width * 3 // 10])
        QtCore.QTimer.singleShot(0, self._position_speaker_button)

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
        if self._caption_request_active:
            self.statusBar().showMessage("Caption request already running")
            return
        self._caption_request_active = True
        self.generate_btn.setEnabled(False)
        self.caption_edit.clear()
        self.caption_edit_container.setVisible(False)
        self.speak_btn.setVisible(False)
        self.copy_btn.setVisible(False)
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
        self.caption_edit_container.setVisible(True)
        self.speak_btn.setVisible(True)
        self.copy_btn.setVisible(True)
        self._position_speaker_button()
        self._position_copy_button()
        QtCore.QTimer.singleShot(0, self._position_speaker_button)
        QtCore.QTimer.singleShot(0, self._position_copy_button)

    def _caption_ready(self, caption: str) -> None:
        self._caption_request_active = False
        self.generate_btn.setEnabled(True)
        self._set_caption(caption)
        self.status_label.setText("Caption ready")
        self.statusBar().showMessage("Caption ready")

    def _caption_failed(self, error: str) -> None:
        self._caption_request_active = False
        self.generate_btn.setEnabled(True)
        self._set_caption(f"Caption failed: {error}")
        self.status_label.setText("Caption failed")
        self.statusBar().showMessage("Caption failed")

    def speak_caption(self) -> None:
        text = self.caption_edit.toPlainText().strip()
        if not text:
            self._show_error("Generate a caption before speaking it.")
            return
        if self._speech_active:
            self.statusBar().showMessage("Caption is already being spoken")
            return

        self._speech_active = True
        self.speak_btn.setEnabled(False)
        self.status_label.setText("Speaking caption...")
        self.statusBar().showMessage("Speaking caption...")

        def speak() -> None:
            try:
                if pyttsx3 is not None:
                    self._tts_engine = pyttsx3.init()
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                else:
                    from win32com.client import Dispatch

                    Dispatch("SAPI.SpVoice").Speak(text)
                self.speech_finished.emit("Caption spoken")
            except Exception as error:  # pragma: no cover
                self.speech_failed.emit(f"Text-to-speech is unavailable: {error}")

        self._speech_thread = threading.Thread(target=speak, daemon=True)
        self._speech_thread.start()

    def _speech_finished(self, message: str) -> None:
        self._speech_active = False
        self.speak_btn.setEnabled(True)
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def _speech_failed(self, message: str) -> None:
        self._speech_active = False
        self.speak_btn.setEnabled(True)
        self._show_error(message)

    def copy_caption(self) -> None:
        text = self.caption_edit.toPlainText().strip()
        if text:
            QtWidgets.QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Caption copied")

    def _show_error(self, message: str) -> None:
        self.status_label.setText("Error")
        self.statusBar().showMessage(message)

    def closeEvent(self, event) -> None:
        self.stop_camera()

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
