import base64
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None


OLLAMA_URL = "http://127.0.0.1:11434"
PREFERRED_MODEL = "gemma4:31b-cloud"
PREVIEW_SIZE = (760, 520)


class CaptionCameraApp:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Ollama Caption Camera")
        self.root.geometry("980x760")
        self.root.minsize(820, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.camera: cv2.VideoCapture | None = None
        self.camera_running = False
        self.latest_frame = None
        self.captured_frame = None
        self.preview_image = None
        self.model_names: list[str] = []
        self._camera_after_id = None
        self._tts_engine = None

        self.model_var = ctk.StringVar(value=PREFERRED_MODEL)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.root.after(100, self.refresh_models)

    def _build_ui(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, corner_radius=14)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="Ollama Caption Camera", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w", padx=18, pady=14
        )
        ctk.CTkLabel(header, textvariable=self.status_var).grid(row=0, column=1, padx=(18, 18), sticky="e")

        body = ctk.CTkFrame(self.root, corner_radius=14)
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        preview_panel = ctk.CTkFrame(body, corner_radius=14)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)
        preview_panel.grid_columnconfigure(0, weight=1)
        preview_panel.grid_rowconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(preview_panel, text="Camera is off", height=480)
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        controls = ctk.CTkFrame(body, corner_radius=12)
        controls.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        controls.grid_columnconfigure(3, weight=1)
        ctk.CTkButton(controls, text="Start camera", command=self.start_camera, width=120).grid(row=0, column=0, padx=(18, 8), pady=12)
        ctk.CTkButton(controls, text="Capture frame", command=self.capture_frame, width=130).grid(row=0, column=1, padx=(0, 8), pady=12)
        ctk.CTkButton(controls, text="Save image", command=self.save_image, width=110).grid(row=0, column=2, padx=(0, 18), pady=12)
        ctk.CTkLabel(controls, text="Model:").grid(row=0, column=3, sticky="e", padx=(0, 8), pady=12)
        self.model_combo = ctk.CTkComboBox(controls, variable=self.model_var, values=[], state="readonly", width=260)
        self.model_combo.grid(row=0, column=4, sticky="e", padx=(0, 8), pady=12)
        ctk.CTkButton(controls, text="Refresh", command=self.refresh_models, width=90).grid(row=0, column=5, padx=(0, 18), pady=12)

        caption_panel = ctk.CTkFrame(body, corner_radius=14)
        caption_panel.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        caption_panel.grid_columnconfigure(0, weight=1)
        header_frame = ctk.CTkFrame(caption_panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        header_frame.grid_columnconfigure(0, weight=1)
        self.speak_button = ctk.CTkButton(header_frame, text="🔊 Speak", command=self.speak_caption, width=110)
        self.speak_button.grid(row=0, column=1, sticky="e")
        self.caption_text = ctk.CTkTextbox(caption_panel, height=5, wrap="word")
        self.caption_text.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 10))
        self.caption_text.configure(state="disabled")
        ctk.CTkButton(caption_panel, text="Generate caption", command=self.generate_caption, width=170).grid(
            row=2, column=0, sticky="e", padx=10, pady=(0, 12)
        )

    def refresh_models(self) -> None:
        self.status_var.set("Loading Ollama models...")

        def load() -> None:
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
                    payload = json.load(response)
                models = [item["name"] for item in payload.get("models", [])]
                self.root.after(0, lambda: self._set_models(models))
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                self.root.after(0, lambda: self._show_error(f"Could not reach Ollama: {error}"))

        threading.Thread(target=load, daemon=True).start()

    def _set_models(self, models: list[str]) -> None:
        self.model_names = models
        self.model_combo.configure(values=models)
        if PREFERRED_MODEL in models:
            self.model_var.set(PREFERRED_MODEL)
        elif models and self.model_var.get() not in models:
            self.model_var.set(models[0])
        self.status_var.set(f"{len(models)} model(s) available")

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
        self.status_var.set("Camera running")
        self._read_camera_frame()

    def _read_camera_frame(self) -> None:
        if not self.camera_running or self.camera is None:
            return
        success, frame = self.camera.read()
        if success:
            self.latest_frame = frame
            self._show_frame(frame)
        self._camera_after_id = self.root.after(30, self._read_camera_frame)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.preview_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.preview_label.configure(image=self.preview_image, text="")

    def capture_frame(self) -> None:
        if self.latest_frame is None:
            self._show_error("Start the camera before capturing a frame.")
            return
        self.captured_frame = self.latest_frame.copy()
        self._show_frame(self.captured_frame)
        self.status_var.set("Frame captured")

    def save_image(self) -> None:
        if self.captured_frame is None:
            self._show_error("Capture a frame before saving.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
        )
        if path:
            cv2.imwrite(path, self.captured_frame)
            self.status_var.set(f"Saved {Path(path).name}")

    def generate_caption(self) -> None:
        if self.captured_frame is None:
            self._show_error("Capture a frame before generating a caption.")
            return
        model = self.model_var.get().strip()
        if not model:
            self._show_error("Choose an Ollama model first.")
            return

        success, encoded = cv2.imencode(".jpg", self.captured_frame)
        if not success:
            self._show_error("Could not encode the captured frame.")
            return
        image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self._set_caption("Generating caption...")
        self.status_var.set(f"Asking {model}...")

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
                self.root.after(0, lambda caption=caption: self._caption_ready(caption))
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                message = str(error)
                self.root.after(0, lambda: self._caption_failed(message))

        threading.Thread(target=generate, daemon=True).start()

    def _set_caption(self, text: str) -> None:
        self.caption_text.configure(state="normal")
        self.caption_text.delete("1.0", tk.END)
        self.caption_text.insert("1.0", text)
        self.caption_text.configure(state="disabled")

    def _caption_ready(self, caption: str) -> None:
        self._set_caption(caption)
        self.status_var.set("Caption ready")

    def _caption_failed(self, error: str) -> None:
        self._set_caption(f"Caption failed: {error}")
        self.status_var.set("Caption failed")

    def speak_caption(self) -> None:
        text = self.caption_text.get("1.0", tk.END).strip()
        if not text:
            self._show_error("Generate a caption before speaking it.")
            return

        try:
            if pyttsx3 is not None:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                self.status_var.set("Caption spoken")
                return

            import win32com.client

            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            self.status_var.set("Caption spoken")
        except Exception as error:  # pragma: no cover
            self._show_error(f"Text-to-speech is unavailable: {error}")

    def _show_error(self, message: str) -> None:
        self.status_var.set("Error")
        messagebox.showerror("Ollama Caption Camera", message)

    def close(self) -> None:
        self.camera_running = False

        if self._camera_after_id is not None:
            try:
                self.root.after_cancel(self._camera_after_id)
            except Exception:
                pass
            self._camera_after_id = None

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
        self.root.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    app_root = ctk.CTk()
    CaptionCameraApp(app_root)
    app_root.mainloop()