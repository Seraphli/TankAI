class ConPool(object):
    def __init__(self, name, limit):
        self.name = name
        self.limit = limit

    def setup(self):
        pass

    def reg_task(self, tasks):
        pass

    def apply(self, func, args, callback):
        pass

    def apply_async(self, func, args, callback):
        pass

    def map(self, func, iterable):
        pass

    def join(self):
        pass
