from tank.env import Env
import numpy as np

env = Env()
t = env.reset()
end = False
while not end:
    for p in [0, 1]:
        for i in [0, 1]:
            if t[p * 2 + i + 1]:
                continue
            s = env.get_state(p, i)
            a = np.ones(9)
            a_mask = env.get_action(p, i)
            _a = a * a_mask
            a_index, = np.where(_a == _a.max())
            a_index = np.random.choice(a_index)
            env.take_action(p, i, a_index)
    t = env.step()
    rs = []
    for p in [0, 1]:
        for i in [0, 1]:
            rs.append(env.get_reward(p, i))
    print(rs, t, env.game.step_count)
    end = t[0]

print(env.game.map[4, 0], env.game.map[4, 8])
