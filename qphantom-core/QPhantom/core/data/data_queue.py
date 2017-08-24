from threading import Thread
from queue import Queue, Full, Empty

class DataQueue(object):
    def __init__(self, gen_func, capacity=1024, num_worker=4):
        """
        Args:
            sess: tensorflow session
            gen_func: is function which return a generator that produce data batch
            dtypes: numpy dtypes
            shapes: shape without batch_dim
        """
        self.num_worker = num_worker
        self.capacity = capacity
        self.data_queue = Queue(maxsize=self.capacity)
        self.stop_flag = Queue(maxsize=1)
        self.worker_flag = Queue(maxsize=num_worker)
        self.gen_func = gen_func
        def fill_up():
            for d in self.gen_func():
                while True:
                    try:
                        if not self.stop_flag.empty():
                            break
                        self.data_queue.put(d, block=False, timeout=0.1)
                        break
                    except Full as e:
                        pass
                if not self.stop_flag.empty():
                    break
            self.worker_flag.put(1)
        self.threads = [Thread(target=fill_up) for i in range(num_worker)]

    @staticmethod
    def background(iter, capacity=1024):
        with DataQueue(lambda: iter, capacity, num_worker=1) as q:
            for v in q.buffer():
                yield v

    def start(self):
        for th in self.threads:
            th.start()

    def close(self):
        self.stop_flag.put(True)
        for th in self.threads:
            th.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def buffer(self, n=None):
        if n is not None:
            for i in range(n):
                while True:
                    try:
                        yield self.data_queue.get(timeout=0.1)
                        break
                    except Empty as e:
                        if self.worker_flag.qsize() == self.num_worker:
                            break
                if self.data_queue.empty() and self.worker_flag.qsize() == self.num_worker:
                    break
        else:
            while not self.data_queue.empty() or self.worker_flag.qsize() < self.num_worker:
                try:
                    yield self.data_queue.get(timeout=0.1)
                except Empty as e:
                    pass
            while not self.worker_flag.empty():
                self.worker_flag.get()

