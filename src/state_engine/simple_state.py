import math
import time

class SimpleStateEngine:
    def __init__(self, baseline=None, decay_rate=0.8, reactivity=0.5):
        # valence, arousal, dominance in [-1,1]
        self.state = baseline or {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        self.baseline = baseline or {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
        self.decay_rate = decay_rate
        self.reactivity = reactivity
        self.last_t = time.time()

    def decay(self, dt):
        for k in self.state:
            # exponential decay towards baseline
            self.state[k] += (self.baseline[k] - self.state[k]) * (1 - math.exp(-self.decay_rate * dt))

    def update(self, input_vad):
        t = time.time()
        dt = max(1e-6, t - self.last_t)
        self.decay(dt)
        for k in self.state:
            self.state[k] += self.reactivity * input_vad.get(k, 0.0)
            self.state[k] = max(-1.0, min(1.0, self.state[k]))
        self.last_t = t

    def get_state(self):
        return self.state.copy()
