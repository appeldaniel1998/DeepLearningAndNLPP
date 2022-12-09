import random

import numpy as np


class BatchGenerator:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.batches = []
        self.contained = []
        random.seed(0)
        for i in range(len(data)):
            self.contained.append(i)

    def getRandomBatch(self):
        size = self.batch_size
        batch = []
        while size > 0 or len(self.contained) > len(self.data) - self.batch_size:
            num = random.choice(self.contained)
            self.contained.remove(num)
            batch.append(self.data[num - 1])
            size -= 1

        batch = np.array(batch)

        return batch


if __name__ == '__main__':
    lst = [x for x in range(0, 500)]
    bg = BatchGenerator(50, lst)
    print(bg.getRandomBatch())
    print(bg.getRandomBatch())
