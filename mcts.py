from mcts_node import MCTSNode, TreeManager
import random


class MCTS(object):
    def __init__(self, side, search_size, logger):
        self.tree_manager = TreeManager()
        self.tree_manager.c_uct = 2
        self.actual_side = side
        self.side = 1 - side
        self.search_size = search_size
        self.logger = logger

    def init(self, actions, env):
        self.root = MCTSNode(self.tree_manager, self.side,
                             descendant=actions)
        self.env = env

    def uct(self):
        self.logger.debug(f'=== UCT start === Step {self.env.game.step_count}')
        self.logger.debug(f'game map {self.env.game.map}')
        for i in range(self.search_size):
            self.logger.debug('=== Search start ===')
            node = self.root
            env = self.env.clone()
            player_move = 2
            self.logger.debug(f'game map {env.game.map}')
            self.logger.debug(f'real side {self.actual_side}')
            self.logger.debug(f'step {env.game.step_count}')

            self.logger.debug('== Select ==')
            # Select
            while node.untried_actions == [] and node.children != []:
                # node is fully expanded and non-terminal
                node = node.select()
                env.take_side_action(node.side, node.action)
                self.logger.debug(f'Select side {node.side}, '
                                  f'action {node.action}')
                player_move -= 1
                if player_move == 0:
                    env.step()
                    self.logger.debug('Select env step')
                    player_move = 2

            self.logger.debug('== Expand ==')
            # Expand
            if node.untried_actions:
                # if we can expand (i.e. state/node is non-terminal)
                action = random.choice(node.untried_actions)
                env.take_side_action(1 - node.side, action)
                self.logger.debug(f'Expand side {1 - node.side}, '
                                  f'action {action}')
                player_move -= 1
                if player_move == 0:
                    env.step()
                    self.logger.debug('Expand env step')
                    player_move = 2
                node = node.add_node(1 - node.side, action,
                                     env.get_side_action(node.side))
                self.logger.debug(f'Expand add node, side {node.side}, '
                                  f'action {node.action}')
            side = 1 - node.side

            roll_out = 0
            self.logger.debug('== Rollout ==')
            # Rollout
            while env.get_side_action(side):
                # while state is non-terminal
                actions = env.get_side_action(side)
                action = random.choice(actions)
                env.take_side_action(side, action)
                self.logger.debug(f'Rollout side {side}, action {action}')
                side = 1 - side
                player_move -= 1
                if player_move == 0:
                    env.step()
                    self.logger.debug('Rollout env step')
                    roll_out += 1
                    player_move = 2

            winner = env.get_winner()
            self.logger.debug(f'winner {winner}')
            if winner >= 0:
                if winner == node.side:
                    z = 1
                else:
                    z = -1
            else:
                z = 0

            self.logger.debug(f'z {z}')

            # Backpropagate
            while node is not None:
                node.update(z, 0)
                self.logger.debug(f'side {node.side}, '
                                  f'action {node.action}, z {z}')
                node = node.parent
                z *= -1

            self.logger.debug('=== Search end ===')

        sorted_nodes = self.root.sorted_child()

        self.logger.info(f'=== Side {self.actual_side} ===')
        _count = 0
        for node in sorted_nodes:
            if _count < 3:
                self.logger.info(node.str())
                _count += 1
            assert node.check_vls()

        selected = sorted_nodes[0]
        self.root = selected
        return selected.action
