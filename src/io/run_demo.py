import soundfile as sf
import sounddevice as sd
import whisper
from pathlib import Path
import numpy as np
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.encoders.text_emotion_quick import predict as predict_text_emotion
from src.state_engine.simple_state import SimpleStateEngine
from src.state_engine.emotion_to_vad import emotion_probs_to_vad

OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)
AUDIO_PATH = OUT / "sample.wav"

# Initialize emotion state engine (persistent)
state_engine = SimpleStateEngine(
    baseline={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
    decay_rate=0.6,
    reactivity=0.7
)

def record(seconds=4, sr=16000):
    print(f"\n🎙️ Recording for {seconds}s... speak now")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    audio = np.squeeze(audio)
    sf.write(str(AUDIO_PATH), audio, sr, subtype="PCM_16")
    return str(AUDIO_PATH)

def transcribe(path, model_name="small"):
    model = whisper.load_model(model_name)
    result = model.transcribe(path)
    text = result["text"].strip()
    print("📝 Transcript:", text)
    return text

def run_emotion_pipeline(text):
    # Step 1: emotion probabilities
    emotion_probs = predict_text_emotion(text)

    # Step 2: emotion → VAD
    vad_input = emotion_probs_to_vad(emotion_probs)

    # Step 3: update internal emotion state
    state_engine.update(vad_input)

    # Step 4: read current mood
    mood = state_engine.get_state()

    print("\n🎭 Emotion → VAD input:")
    for k, v in vad_input.items():
        print(f"  {k}: {v:.3f}")

    print("\n🧠 Current INTERNAL emotion state:")
    for k, v in mood.items():
        print(f"  {k}: {v:.3f}")

    return mood

if __name__ == "__main__":
    audio_path = record(seconds=4)
    text = transcribe(audio_path)

    if text:
        run_emotion_pipeline(text)
    else:
        print("No speech detected.")
