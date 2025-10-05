# EmoVoice — Phase 0

MVP skeleton for an emotionally aware voice assistant. Phase 0: microphone input -> ASR -> simple emotion detector -> prompt LLM -> expressive TTS.

How to run:
1. Create virtualenv, activate:
   - PowerShell: .venv\Scripts\Activate.ps1
   - Cmd: .venv\Scripts\activate.bat
2. Install system deps (ffmpeg) and Python libs (see requirements.txt)
3. pip install -r requirements.txt
4. python src\io\run_demo.py
