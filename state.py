import numpy as np


class State(object):
    def __init__(self, length, stack=3):
        self.length = length
        self.stack = stack
        self.state = np.zeros((9, 9, self.length * self.stack), dtype=np.uint8)

    def update(self, new_state):
        self.state[..., :-self.length] = self.state[..., self.length:]
        self.state[..., -self.length:] = new_state

    def get_state(self):
        return np.array(self.state)
