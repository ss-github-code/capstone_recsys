# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Original codes are from
# https://github.com/kang205/SASRec/blob/master/sampler.py
import numpy as np
from multiprocessing import Process, Queue


def random_neq(left, right, s):
    t = np.random.randint(left, right)
    while t in s:
        t = np.random.randint(left, right)
    return t


def sample_function_valid(
    dataset, usernum, itemnum, batch_size, maxlen, result_queue, num_neg
):
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
    """

    def sample(idx_arr, dataset):

        user_train = dataset.user_train
        user_valid = dataset.user_valid

        user = idx_arr
        while len(user_train[idx_arr]) < 1 or len(user_valid[idx_arr]) < 1:
            idx_arr += 1
            if idx_arr > usernum:
                idx_arr = 1
            user = idx_arr
        idx_arr += 1
        if idx_arr > usernum:
            idx_arr = 1
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        ts = set(user_train[user])
        item_idx = [user_valid[user][0]]
        for _ in range(num_neg):
            item_idx.append(random_neq(1, itemnum + 1, ts))

        return (user, seq, item_idx, idx_arr)

    idx = 1
    while True:
        one_batch = []
        for i in range(batch_size):
            user, seq, item_idx, idx = sample(idx, dataset)
            one_batch.append((user, seq, item_idx))
        result_queue.put(zip(*one_batch))

def sample_function_test(
    dataset, usernum, itemnum, batch_size, maxlen, result_queue, num_neg
):
    """Batch sampler that creates a sequence of negative items based on the
    original sequence of items (positive) that the user has interacted with.

    Args:
        user_train (dict): dictionary of training exampled for each user
        usernum (int): number of users
        itemnum (int): number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        result_queue (multiprocessing.Queue): queue for storing sample results
        seed (int): seed for random generator
    """

    def sample(idx_arr, dataset):

        user_train = dataset.user_train
        user_valid = dataset.user_valid
        user_test = dataset.user_test

        user = idx_arr
        while len(user_train[idx_arr]) < 1 or len(user_test[idx_arr]) < 1:
            idx_arr += 1
            if idx_arr > usernum:
                idx_arr = 1
            user = idx_arr
        idx_arr += 1
        if idx_arr > usernum:
            idx_arr = 1

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = user_valid[user][0]
        idx -= 1
        for i in reversed(user_train[user]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        ts = set(user_train[user])
        ts.add(user_valid[user][0])
        item_idx = [user_test[user][0]]
        for _ in range(num_neg):
            item_idx.append(random_neq(1, itemnum + 1, ts))

        return (user, seq, item_idx, idx_arr)

    idx = 1
    while True:
        one_batch = []
        for i in range(batch_size):
            user, seq, item_idx, idx = sample(idx, dataset)
            one_batch.append((user, seq, item_idx))
        result_queue.put(zip(*one_batch))

class WarpSamplerValidTest(object):
    """Sampler object that creates an iterator for feeding batch data while testing.

    Attributes:
        User: dict, all the users (keys) with items as values
        usernum: integer, total number of users
        itemnum: integer, total number of items
        batch_size (int): batch size
        maxlen (int): maximum input sequence length
        n_workers (int): number of workers for parallel execution
    """

    def __init__(self, dataset, usernum, itemnum, valid_flag, batch_size=64, maxlen=10, n_workers=1, num_neg=50):
        self.result_queue = Queue(maxsize=n_workers * 20)
        self.processors = []
        sample_function = sample_function_valid if valid_flag == True else sample_function_test
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        dataset,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        num_neg,
                    ),
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
