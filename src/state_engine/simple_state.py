import math
import time

class SimpleStateEngine:
    """
    Persistent emotional state with decay.
    State = {valence, arousal, dominance} in [-1, 1]
    """

    def __init__(self, baseline=None, decay_rate=0.8, reactivity=0.6):
        self.baseline = baseline or {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        self.state = self.baseline.copy()
        self.decay_rate = decay_rate
        self.reactivity = reactivity
        self.last_time = time.time()

    def _decay(self, dt):
        for k in self.state:
            self.state[k] += (self.baseline[k] - self.state[k]) * (
                1 - math.exp(-self.decay_rate * dt)
            )

    def update(self, vad_input):
        now = time.time()
        dt = max(1e-6, now - self.last_time)

        self._decay(dt)

        for k in self.state:
            self.state[k] += self.reactivity * vad_input.get(k, 0.0)
            self.state[k] = max(-1.0, min(1.0, self.state[k]))

        self.last_time = now

    def get_state(self):
        return self.state.copy()
