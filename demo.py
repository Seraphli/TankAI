from tank.env import Env
from q_learn import QLearn as AI
from state import State
import numpy as np

state = [[State(8, 3), State(8, 3)],
         [State(8, 3), State(8, 3)]]
ai = AI()
env = Env()
for _ in range(int(1e3)):
    ai.logger.debug(f'=== Game start ===')
    end = env.reset()
    s = [[0, 0], [0, 0]]
    a = [[0, 0], [0, 0]]
    r = [[0, 0], [0, 0]]
    t = [[0, 0], [0, 0]]
    s_ = [[0, 0], [0, 0]]
    for p in [0, 1]:
        for i in [0, 1]:
            if end[p * 2 + i + 1]:
                continue
            _s = env.get_state(p, i)
            ai.logger.debug(f'side:{p} index:{i} state {_s}')
            state[p][i].update(_s)
    game_end = False
    while not game_end:
        for p in [0, 1]:
            for i in [0, 1]:
                if end[p * 2 + i + 1]:
                    continue
                s[p][i] = state[p][i].get_state()
                _state = np.reshape(s[p][i], (1, *s[p][i].shape))
                a_mask = env.get_action(p, i)
                ai.logger.debug(f'side:{p} index:{i} a_mask {a_mask}')
                a[p][i] = ai.get_action(_state, a_mask)
                ai.logger.debug(f'side:{p} index:{i} a {a[p][i]}')
                env.take_action(p, i, a[p][i])
        end = env.step()

        for p in [0, 1]:
            for i in [0, 1]:
                if t[p][i] == 0:
                    if end[p * 2 + i + 1]:
                        t[p][i] = 1
                    _s = env.get_state(p, i)
                    ai.logger.debug(f'side:{p} index:{i} state {_s}')
                    state[p][i].update(_s)
                    s_[p][i] = state[p][i].get_state()
                    r[p][i] = env.get_reward(p, i)
                    ai.logger.debug(f'side:{p} index:{i} r {r[p][i]}')
                    ai.logger.debug(f'side:{p} index:{i} t {t[p][i]}')
                    ai.replay.add(s[p][i], a[p][i], r[p][i], t[p][i], s_[p][i])

        ai.logger.debug(f'r {r}')
        ai.logger.debug(f't {t}')
        ai.logger.debug(f'step {env.game.step_count}')
        game_end = end[0]

    ai.logger.info(f'step {env.game.step_count}')
    ai.logger.debug(f'base {env.game.map[4, 0]}, {env.game.map[4, 8]}')
    ai.logger.debug(f'=== Game end ===')
    if len(ai.replay) > 32 * 20:
        ai.nn.train()
