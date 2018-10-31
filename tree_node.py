class TreeNode(object):
    def __init__(self, name=''):
        self.name = name
        self.up = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.up = self
        return child

    def is_root(self):
        return self.up is None
