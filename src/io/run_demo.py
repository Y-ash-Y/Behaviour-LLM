# src/io/run_demo.py
import soundfile as sf
import sounddevice as sd
import whisper
from pathlib import Path
import numpy as np
import sys

# Ensure project root is in path so we can import encoders module
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import the quick text-emotion predictor
try:
    from src.encoders.text_emotion_quick import predict as predict_text_emotion
except Exception:
    # fallback if package-like import doesn't work
    try:
        from encoders.text_emotion_quick import predict as predict_text_emotion
    except Exception as e:
        raise ImportError("Could not import text_emotion_quick.predict. "
                          "Check that src/encoders/text_emotion_quick.py exists.") from e

OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)
AUDIO_PATH = OUT / "sample.wav"

def record(seconds=4, sr=16000):
    print(f"Recording for {seconds}s... speak now")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    audio = np.squeeze(audio)
    sf.write(str(AUDIO_PATH), audio, sr, subtype='PCM_16')
    print(f"Saved to {AUDIO_PATH}")
    return str(AUDIO_PATH)

def transcribe(path, model_name='small'):
    print('Loading Whisper model (this may take a while on first run)...')
    model = whisper.load_model(model_name)  # small/medium/large
    print('Transcribing...')
    result = model.transcribe(path)
    transcript = result.get('text', '').strip()
    print('Transcript:', repr(transcript))
    return transcript

def analyze_text_emotion(text):
    print('Running text-emotion predictor (this will download a small HF model if not cached)...')
    scores = predict_text_emotion(text)
    # pretty print top 5
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    print('Emotion scores (top -> bottom):')
    for label, prob in sorted_scores:
        print(f"  {label:>12}: {prob:.3f}")
    return scores

if __name__ == '__main__':
    p = record(seconds=4)
    transcript = transcribe(p, model_name='small')
    if transcript:
        analyze_text_emotion(transcript)
    else:
        print("No transcript produced.")
