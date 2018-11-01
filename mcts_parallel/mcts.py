from concurrency.waldorf import WaldorfPool as Pool
from tqdm import tqdm
from mcts_node import MCTSNode, TreeManager
from mcts_parallel.rollout import random_rollout


class MCTSP(object):
    def __init__(self):
        import threading

        self.pool = Pool('tankai', 0)
        self.pool.setup()
        self.pool.reg_task([random_rollout])
        self.backup_lock = threading.Lock()
        self.select_lock = threading.Lock()
        self.log_info = {}

    def setup(self, side, search_size, logger):
        self.tree_manager = TreeManager()
        self.tree_manager.c_uct = 2
        self.actual_side = side
        self.side = 1 - side
        self.search_size = search_size
        self.logger = logger

    def reset_root(self, actions):
        self.root = MCTSNode(self.tree_manager, self.side,
                             descendant=actions)

    def select(self, node):
        self.select_lock.acquire()
        action_list = []
        # Select
        while node.untried_actions == [] and node.children:
            # node is fully expanded and non-terminal
            node = node.select()
            action_list.append((node.side, node.action))
        self.select_lock.release()
        return action_list, node

    def _backup(self, node, flag, result):
        z = result['z']

        # backpropagate from the expanded node and work back to the root node
        while node is not None:
            if (node.is_leaf() and flag) or node.is_root():
                # node is add in the front of the backup
                # or node is the root, which won't apply vl
                node.update(z, 0, {})
            else:
                node.update(z, 3, {})
            node = node.parent
            z *= -1

        self.backup_lock.acquire()
        self.jobs.pop(str(result['action_list']))
        self.pbar.update()
        self.backup_lock.release()

    def backup(self, result):
        self.select_lock.acquire()
        uid = result['uid']
        node, flag = self.backup_dict[uid]
        if flag:
            # add child and descend tree
            side, action = result['action_list'][-1]
            node = node.add_node(1 - node.side, action,
                                 result['descendants'])

        self._backup(node, flag, result)
        self.backup_dict.pop(uid, None)
        self.select_lock.release()

    def uct(self, env):
        import sys
        import pickle
        import zlib
        import random
        import time
        import uuid

        self.info = {}
        self.info['results'] = []
        self.jobs = {}
        self.backup_dict = {}
        self.pbar = tqdm(total=self.search_size, file=sys.stdout)
        _env = zlib.compress(pickle.dumps(env, -1))

        for i in range(self.search_size):
            wait = True
            while wait:
                node = self.root
                untried_flag = False
                action_list, node = self.select(node)

                if node.untried_actions:
                    untried_flag = True

                # Expand
                if untried_flag:
                    # if we can expand (i.e. state/node is non-terminal)
                    action = random.choice(node.untried_actions)
                    action_list.append((1 - node.side, action))

                if str(action_list) in self.jobs:
                    time.sleep(0.01)
                else:
                    self.backup_lock.acquire()
                    self.jobs[str(action_list)] = 1
                    tmp_node = self.root
                    if untried_flag:
                        for a in action_list[:-1]:
                            tmp_node = tmp_node.select_child_by_action(a[1])
                            tmp_node.apply_vl(3)
                    else:
                        for a in action_list:
                            tmp_node = tmp_node.select_child_by_action(a[1])
                            tmp_node.apply_vl(3)
                    self.backup_lock.release()
                    wait = False

            uid = str(uuid.uuid4())
            self.backup_dict[uid] = [node, untried_flag]
            self.pool.apply_async(
                random_rollout,
                (uid, _env, action_list, 0),
                self.backup)

        self.pool.join()
        self.pbar.close()

        sorted_nodes = self.root.sorted_child()
        _count = 0
        for node in sorted_nodes:
            if _count < 3:
                self.logger.info(node.str())
                _count += 1

        selected = sorted_nodes[0]
        return selected
