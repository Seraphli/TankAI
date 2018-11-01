from concurrency import ConPool
from multiprocessing import Pool


class MPPool(ConPool):
    def setup(self):
        self._pool = Pool(self.limit)

    def apply(self, func, args, callback):
        result = self._pool.apply(func, args)
        callback(result)

    def apply_async(self, func, args, callback):
        self._pool.apply_async(func, args, callback=callback)

    def map(self, func, iterable):
        self._pool.map(func, iterable)

    def join(self):
        self._pool.join()
