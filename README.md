# Ollama Caption Camera

A QtPy desktop app that captures a webcam frame and asks a local Ollama vision model for a concise caption.

## Setup

1. Install Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   The interface uses QtPy with the PySide6 Qt backend.

2. Install and start Ollama, then pull a vision model if needed. The app prefers `gemma4:31b-cloud` when it is installed:

   ```powershell
   ollama pull gemma4:31b-cloud
   ```

3. Run the app:

   ```powershell
   python app.py
   ```

4. Optional: click the speaker button in the top-right of the caption panel to read the generated caption aloud.

The model selector is populated from Ollama at startup. The selected model must support image input for captioning.