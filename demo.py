from tank.env import Env
from q_learn import QLearn as AI
from state import State
import numpy as np


def test_score(ai):
    ai.logger.debug(f'=== Test start ===')
    old_ai = AI('tank_old_ai', 5, False)
    old_ai.nn.load('./model')
    test_env = Env()
    score = [0, 0]
    for g_id in range(100):
        ai.logger.debug(f'=== Game start ===')
        end = test_env.reset()
        ai.logger.debug(f'game map {test_env.game.map}')
        state = [[State(8, 3), State(8, 3)],
                 [State(8, 3), State(8, 3)]]
        s = [[0, 0], [0, 0]]
        a = [[0, 0], [0, 0]]
        r = [[0, 0], [0, 0]]
        t = [[0, 0], [0, 0]]
        s_ = [[0, 0], [0, 0]]
        for p in [0, 1]:
            for i in [0, 1]:
                if end[p * 2 + i + 1]:
                    continue
                _s = test_env.get_state(p, i)
                # ai.logger.debug(f'side:{p} index:{i} state {_s}')
                state[p][i].update(_s)
        game_end = False
        while not game_end:
            for p in [0, 1]:
                for i in [0, 1]:
                    if end[p * 2 + i + 1]:
                        continue
                    s[p][i] = state[p][i].get_state()
                    _state = np.reshape(s[p][i], (1, *s[p][i].shape))
                    a_mask = test_env.get_action(p, i)
                    ai.logger.debug(f'side:{p} index:{i} a_mask {a_mask}')
                    if p == 0:
                        a[p][i] = ai.get_action(_state, a_mask)
                    else:
                        a[p][i] = old_ai.get_action(_state, a_mask)
                    ai.logger.debug(f'side:{p} index:{i} a {a[p][i]}')
                    test_env.take_action(p, i, a[p][i])
            end = test_env.step()
            ai.logger.debug(f'game map {test_env.game.map}')

            for p in [0, 1]:
                for i in [0, 1]:
                    if t[p][i] == 0:
                        if end[p * 2 + i + 1]:
                            t[p][i] = 1
                        _s = test_env.get_state(p, i)
                        # ai.logger.debug(f'side:{p} index:{i} state {_s}')
                        state[p][i].update(_s)
                        s_[p][i] = state[p][i].get_state()
                        r[p][i] = test_env.get_reward(p, i)
                        ai.logger.debug(f'side:{p} index:{i} r {r[p][i]}')
                        ai.logger.debug(f'side:{p} index:{i} t {t[p][i]}')

            ai.logger.debug(f'r {r}')
            ai.logger.debug(f't {t}')
            ai.logger.debug(f'step {test_env.game.step_count}')
            game_end = end[0]

        ai.logger.debug(f'step {test_env.game.step_count}')
        ai.logger.debug(f'base {test_env.game.map[4, 0]}, '
                        f'{test_env.game.map[4, 8]}')
        ai.logger.debug(f'=== Game end ===')
        ai.logger.debug(f'Game num {g_id}')
        winner = test_env.get_winner()
        if winner >= 0:
            score[winner] += 1
    old_ai.nn.close()
    ai.logger.debug(f'=== Test end ===')
    ai.logger.info(f'Test score: {score}')
    return score[0] > score[1]


def main():
    saved = False
    ai = AI()
    env = Env()
    for g_id in range(int(1e5)):
        ai.logger.debug(f'=== Game start ===')
        end = env.reset()
        ai.logger.debug(f'game map {env.game.map}')
        state = [[State(8, 3), State(8, 3)],
                 [State(8, 3), State(8, 3)]]
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
                # ai.logger.debug(f'side:{p} index:{i} state {_s}')
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
            ai.logger.debug(f'game map {env.game.map}')

            for p in [0, 1]:
                for i in [0, 1]:
                    if t[p][i] == 0:
                        if end[p * 2 + i + 1]:
                            t[p][i] = 1
                        _s = env.get_state(p, i)
                        # ai.logger.debug(f'side:{p} index:{i} state {_s}')
                        state[p][i].update(_s)
                        s_[p][i] = state[p][i].get_state()
                        r[p][i] = env.get_reward(p, i)
                        ai.logger.debug(f'side:{p} index:{i} r {r[p][i]}')
                        ai.logger.debug(f'side:{p} index:{i} t {t[p][i]}')
                        ai.replay.add(s[p][i], a[p][i], r[p][i],
                                      t[p][i], s_[p][i])

            ai.logger.debug(f'r {r}')
            ai.logger.debug(f't {t}')
            ai.logger.debug(f'step {env.game.step_count}')
            game_end = end[0]

        ai.logger.debug(f'step {env.game.step_count}')
        ai.logger.debug(f'base {env.game.map[4, 0]}, {env.game.map[4, 8]}')
        ai.logger.debug(f'=== Game end ===')
        if g_id > 0 and g_id % 200 == 0:
            ai.logger.info(f'Game num {g_id}')

        if len(ai.replay) > 32 * 20:
            for i in range(5):
                ai.nn.train()
            if g_id % 2 == 0:
                ai.nn.update_param()
            if g_id % 20 == 0:
                if saved:
                    # Test if model is better than before
                    if test_score(ai):
                        ai.nn.save('./model/model.ckpt')
                else:
                    ai.nn.save('./model/model.ckpt')
                    saved = True


if __name__ == '__main__':
    main()
