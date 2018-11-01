def random_rollout(args):
    try:
        import random
        import zlib
        import pickle
        import numpy as np

        uid, env, action_list, limit = args
        env = pickle.loads(zlib.decompress(env))

        for a in action_list:
            side = a[0]
            env.take_side_action(a[0], a[1])
            env.attempt_step()

        node_side = side
        side = 1 - side
        descendants = env.get_side_action(side)

        # Rollout
        while env.get_side_action(side):
            # while state is non-terminal
            actions = env.get_side_action(side)
            action = random.choice(actions)
            env.take_side_action(side, action)
            env.attempt_step()
            side = 1 - side

        winner = env.get_winner()
        if winner >= 0:
            if winner == node_side:
                z = 1
            else:
                z = -1
        else:
            z = 0

        return {
            'uid': uid, 'z': z, 'action_list': action_list,
            'descendants': descendants
        }
    except Exception as e:
        import traceback
        import socket
        raise Exception(socket.gethostname(), traceback.format_exc())
