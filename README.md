# CamCaption

A QtPy desktop application that previews a webcam, captures a frame, and asks an Ollama vision model for a concise image caption.

## Features

- QtPy interface using the PySide6 backend.
- Live camera preview with Windows camera-backend fallback.
- Horizontally mirrored camera frames for a natural-facing preview.
- Capture and save JPEG or PNG images.
- Ollama model selector populated from the local Ollama API.
- Defaults to `gemma4:31b-cloud` when that model is installed.
- Caption text shown in a dedicated text area.
- Speaker icon for reading captions aloud without blocking the interface.
- Copy-caption icon inside the caption text area.
- Start/stop camera control and duplicate-request protection.
- Preserved captured preview while switching layouts.

## Responsive Layout

Normal windowed mode keeps the camera and caption panels vertically arranged. The caption panel places the caption text and captured image side by side.

Maximized mode uses a horizontal camera/caption split. The camera receives approximately 70% of the width, while the caption panel receives approximately 30%.

True fullscreen starts camera-first before capture. After a frame is captured, the caption panel appears with an image-above-caption arrangement. The captured image and caption each receive half of the caption panel's available height.

Use `F11` to enter or leave fullscreen and `Esc` to exit fullscreen.

## Setup

1. Install Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   The interface uses QtPy with the PySide6 Qt backend.

2. Install and start Ollama. Pull a vision-capable model if needed:

   ```powershell
   ollama pull gemma4:31b-cloud
   ```

3. Run the app:

   ```powershell
   python app.py
   ```

## Usage

1. Start the camera.
2. Capture a frame.
3. Choose an installed Ollama model.
4. Select **Generate caption**.
5. Use the speaker icon to hear the caption or the copy icon to copy it.

The selected Ollama model must support image input. Ollama must be running locally at `http://127.0.0.1:11434`.