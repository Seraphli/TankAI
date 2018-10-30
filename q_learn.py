from nn import NN
from replay import Replay
import numpy as np
from util import init_logger


class QLearn(object):
    def __init__(self, logger_name='tank', replay_size=int(1e4), summary=True):
        self.logger = init_logger(logger_name)
        self.epsilon = 0.1
        self.replay = Replay(replay_size)
        self.nn = NN(self.replay.sample_fn, summary)

    def get_action(self, state, a_mask):
        random = np.random.random() < self.epsilon
        if random:
            q = np.ones(9)
        else:
            q = self.nn.predict(state)[0] + 0.1

        _a = q * a_mask
        a_index, = np.where(_a == _a.max())
        a_index = np.random.choice(a_index)
        return a_index, {'q': q, 'random': random}
