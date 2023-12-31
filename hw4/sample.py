"""
Sampling utils
"""

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from collections import Counter

def sample(M: int, p: List[float], sorted: bool = False, rand_seed: int = None) -> List[int]:
    """
    Sampling a series of integers from a distribution
    :param M: sampling number
    :param p: a vector representing a specific distribution of [0, ..., len(p)-1]
    :param sorted: whether sort the returned result
    :param rand_seed: random seed
    :return: return M indices sampled from distribution p
    """
    if M <= 0:
        raise ValueError('M must be an positive integer')
    F = np.cumsum(p)
    indices = []
    np.random.seed(rand_seed)
    for i in range(M):
        n = np.random.rand()
        F_and_n = np.append(F, n)
        F_and_n.sort()
        idx = F_and_n.tolist().index(n)
        indices.append(idx)
    if sorted:
        indices.sort()
    return indices


if __name__ == '__main__':
    p = [0.1, 0.2, 0.7]
    p = np.ones(10) / 10
    p = np.cumsum(p)
    p = p / np.sum(p)
    # p = [1]
    M = int(100)

    indices = sample(M, p, rand_seed=1234)
    counts = Counter(indices)
    # print(counts)
    keys = list(counts.keys())
    vals = list(counts.values())
    plt.bar(keys, vals)
    for k, v in zip(keys, vals):
        plt.text(k, v + 1, v, ha='center', va='bottom')
    plt.xticks(range(1, len(p) + 1))
    plt.title('Total sampling number: {}'.format(M))
    plt.xlabel('index')
    plt.ylabel('count')
    plt.savefig('sample_bar.png', dpi=400)
    plt.show()

