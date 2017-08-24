class ResultBagging(object):
    def __init__(self, k=None):
        '''
        :param k:  using last k
        '''
        self.k = k
        self.stack = []
        self.current = None

    def add(self, result):
        self.stack.append(result)
        if self.current is None:
            self.current = result
        else:
            if self.k is None or len(self.stack) <= self.k:
                self.current = self.current + result
            else:
                self.current = self.current + result - self.stack[-self.k-1]
        return self.result()

    def bag(self, result):
        return self.add(result)

    def __k(self):
        return min(self.k, len(self.stack)) if self.k is not None else len(self.stack)

    def result(self):
        return (self.current / self.__k()) if self.current is not None else None
