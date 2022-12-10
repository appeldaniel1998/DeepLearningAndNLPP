import random

import numpy as np


class BatchGenerator:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data.values.tolist()  # TODO: convert to python list
        self.batches = []
        self.contained = []
        random.seed(0)
        for i in range(len(data)):
            self.contained.append(i)

    def getRandomBatch(self):
        batch = []

        leftToBatch = len(self.contained)
        lastBatch = leftToBatch < self.batch_size

        for _ in range(self.batch_size):
            if not lastBatch:
                num = random.choice(self.contained)
                self.contained.remove(num)
                batch.append(self.data[num - 1].copy())
            else:
                pass

        batch = np.array(batch)

        return batch


if __name__ == '__main__':
    lst = [x for x in range(0, 500)]
    bg = BatchGenerator(50, lst)
    print(bg.getRandomBatch())
    print(bg.getRandomBatch())
