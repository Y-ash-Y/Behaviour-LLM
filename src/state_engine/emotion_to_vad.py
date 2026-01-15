# src/state_engine/emotion_to_vad.py

# Mapping from discrete emotions to Valence-Arousal-Dominance
# Values are in [-1, 1]
EMOTION_VAD_MAP = {
    "joy":        {"valence": 0.8,  "arousal": 0.6,  "dominance": 0.4},
    "love":       {"valence": 0.9,  "arousal": 0.5,  "dominance": 0.3},
    "surprise":   {"valence": 0.2,  "arousal": 0.8,  "dominance": 0.1},

    "sadness":    {"valence": -0.7, "arousal": -0.4, "dominance": -0.5},
    "fear":       {"valence": -0.8, "arousal": 0.6,  "dominance": -0.7},
    "anger":      {"valence": -0.6, "arousal": 0.8,  "dominance": 0.6},
    "disgust":    {"valence": -0.7, "arousal": 0.3,  "dominance": 0.2},

    "neutral":    {"valence": 0.0,  "arousal": 0.0,  "dominance": 0.0},
}

def emotion_probs_to_vad(emotion_probs):
    """
    Convert emotion probability dict → weighted VAD vector
    """
    vad = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

    for emotion, prob in emotion_probs.items():
        if emotion not in EMOTION_VAD_MAP:
            continue
        for k in vad:
            vad[k] += prob * EMOTION_VAD_MAP[emotion][k]

    return vad
