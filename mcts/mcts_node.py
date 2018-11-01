from mcts.tree_node import TreeNode
from math import sqrt, log


class TreeManager(object):
    def __init__(self):
        self.root = None
        self.c_uct = 0
        self.mix_param = 0.5


class WN(object):
    def __init__(self, update_rule):
        self._N = 0
        self._W = 0
        self._vl_N = 0
        self._vl_W = 0
        self.update_rule = update_rule

    def apply_vl(self, vl):
        self._vl_N += vl
        self._vl_W -= vl

    def update(self, v, vl):
        self.update_rule(v, vl)

    @property
    def N(self):
        return self._N + self._vl_N

    @property
    def pure_N(self):
        return self._N

    @property
    def W(self):
        return self._W + self._vl_W

    @property
    def Q(self):
        if self._N == 0:
            return 0
        return self.W / self.N

    def check_vls(self):
        return self._vl_N == 0 and self._vl_W == 0


class AvgWN(WN):
    def __init__(self):
        super(AvgWN, self).__init__(self.update_rule)

    def update_rule(self, v, vl):
        self._N += 1
        self._W += v
        self._vl_N -= vl
        self._vl_W += vl


class MaxWN(WN):
    def __init__(self):
        super(MaxWN, self).__init__(self.update_rule)
        self._max_v = 0

    def update_rule(self, v, vl):
        if self._N == 0:
            self._N = 1
            self._W = v
            self._max_v = v
            return
        self._vl_N -= vl
        self._vl_W += vl
        if self._max_v < v:
            self._max_v = v
        self._N += 1
        self._W = self._max_v * self._N


class MCTSNode(TreeNode):
    def __init__(self, manager, side, action="root", descendant=None):
        super(MCTSNode, self).__init__(name=action)
        self.manager = manager
        self.side = side
        self.action = action
        self.untried_actions = descendant
        self.size = len(descendant)
        self.wn_r = AvgWN()
        self._N = 0
        self._W = 0
        self.action_child_map = {}
        self.vls = [0, 0]
        self.info = {}
        self.node_class = MCTSNode

    def sort_key(self, c):
        return c.Q, c.N

    def set_c_uct(self, c_uct):
        self.c_uct = c_uct

    def add_node(self, side, action, descendant):
        node = MCTSNode(self.manager, side, action, descendant)
        self.action_child_map[action] = node
        node = self.add_child(child=node)
        self.untried_actions.remove(action)
        return node

    def select(self):
        """Use the UCB1 formula to select a children node."""
        cond = lambda c: c.value
        sorted_children = sorted(self.children, key=cond)
        return sorted_children[-1]

    def select_child_by_action(self, action):
        return self.action_child_map[action]

    def update(self, v, vl, info=None):
        """Update this node."""
        if info:
            self.info = info
        self.wn_r.update(v, vl)

    def apply_vl(self, vl):
        self.wn_r.apply_vl(vl)

    def sorted_child(self):
        return sorted(self.children, key=self.sort_key,
                      reverse=True)

    @property
    def parent(self):
        return self.up

    @property
    def N(self):
        return self.wn_r.pure_N

    @property
    def W(self):
        return self.wn_r.W

    @property
    def Q(self):
        return self.wn_r.Q

    @property
    def U(self):
        if self.is_root():
            return 0
        return self.manager.c_uct * sqrt(log(self.parent.N) / self.N + 1)

    @property
    def value(self):
        return self.Q + self.U

    def str(self):
        return '[A: {0:>8}, N: {1:>3}, Q: {2:.3f}, U: {3:.3f}, V: {4:.3f}]'. \
            format(str(self.action), self.N, self.Q, self.U, self.value)

    def check_vls(self):
        return self.wn_r.check_vls()
