from threading import Thread
from queue import Queue, Full

class DataQueue(object):
    def __init__(self, gen_func, capacity=1024, num_worker=4):
        """
        Args:
            sess: tensorflow session
            gen_func: is function which return a generator that produce data batch
            dtypes: numpy dtypes
            shapes: shape without batch_dim
        """
        self.capacity = capacity
        self.data_queue = Queue(maxsize=self.capacity)
        self.stop_flag = Queue(maxsize=1)
        def fill_up():
            for d in gen_func():
                try:
                    if not self.stop_flag.empty():
                        break
                    self.data_queue.put(d, block=False, timeout=0.2)
                except Full as e:
                    pass
        self.threads = [Thread(target=fill_up) for i in range(num_worker)]

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

    def buffer(self, n):
        for i in range(n):
            yield self.data_queue.get()
