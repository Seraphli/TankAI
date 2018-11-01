from tank.env import Env
from mcts_parallel.mcts import MCTSP as AI
from util import init_logger


def main():
    logger = init_logger('mcts')
    ai = [AI(), AI()]
    ai[0].setup(0, 1000, logger)
    ai[1].setup(1, 200, logger)
    env = Env()
    logger.debug(f'=== Game start ===')
    end = env.reset()
    logger.debug(f'game map {env.game.map}')

    for p in [0, 1]:
        ai[p].reset_root(env.get_side_action(p))

    game_end = False
    while not game_end:
        a = [-1, -1]
        for p in [0, 1]:
            node = ai[p].uct(env)
            a[p] = node.action
            ai[p].root = node
        for p in [0, 1]:
            env.take_side_action(p, a[p])
            logger.debug(f'game side {p}, action {a[p]}')
        end = env.step()
        for p in [0, 1]:
            if a[1 - p] in ai[p].root.action_child_map:
                ai[p].root = ai[p].root.select_child_by_action(a[1 - p])
                ai[p].root.action = 'root'
                ai[p].root.up = None
            else:
                ai[p].reset_root(env.get_side_action(p))
        logger.debug(f'game map {env.game.map}')
        logger.debug(f'step {env.game.step_count}')
        game_end = end[0]

    rewards = []
    for p in [0, 1]:
        for index in [0, 1]:
            rewards.append(env.get_reward(p, index))

    logger.info(f'rewards {rewards}')

    logger.info(f'step {env.game.step_count}')
    logger.debug(f'base {env.game.map[4, 0]}, {env.game.map[4, 8]}')
    logger.debug(f'=== Game end ===')
    logger.info(f'winner {env.get_winner()}')
    env.save_replay('.')


if __name__ == '__main__':
    main()
