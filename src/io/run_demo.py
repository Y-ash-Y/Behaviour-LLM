import soundfile as sf
import sounddevice as sd
import whisper
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[3] / 'data'
OUT.mkdir(parents=True, exist_ok=True)
AUDIO_PATH = OUT / 'sample.wav'

def record(seconds=4, sr=16000):
    print(f'Recording for {seconds}s... speak now')
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    audio = np.squeeze(audio)
    sf.write(str(AUDIO_PATH), audio, sr, subtype='PCM_16')
    print(f'Saved to {AUDIO_PATH}')
    return str(AUDIO_PATH)

def transcribe(path, model_name='small'):
    print('Loading Whisper model (this may take a while)...')
    model = whisper.load_model(model_name)  # small/medium/large
    print('Transcribing...')
    result = model.transcribe(path)
    print('Transcript:', result['text'])
    return result

if __name__ == '__main__':
    p = record(seconds=4)
    transcribe(p, model_name='small')
