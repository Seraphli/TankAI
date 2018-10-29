from nn import NN
from replay import Replay
import numpy as np
from util import init_logger


class QLearn(object):
    def __init__(self):
        self.logger = init_logger('tank')
        self.epsilon = 0.1
        self.replay = Replay(int(1e4))
        self.nn = NN(self.replay.sample)

    def get_action(self, state, a_mask):
        if np.random.random() < self.epsilon:
            q = np.ones(9)
        else:
            q = self.nn.predict(state)[0]

        _a = q * a_mask
        a_index, = np.where(_a == _a.max())
        a_index = np.random.choice(a_index)
        return a_index
