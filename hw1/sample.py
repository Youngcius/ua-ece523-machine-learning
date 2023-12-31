import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from collections import Counter


def sample(M: int, p: List[float], sorted: bool = False) -> Union[int, List[int]]:
    """
    sampling a series of integers from a distribution.
    :param M: sampling number
    :param p: a vector representing a sepific distribution
    :param sorted: whether sort the returned result
    :return: return M indices sampled from distribution p
    """
    if M <= 0:
        raise ValueError('M must be an positive integer')
    F = np.cumsum(p)
    indices = []
    for i in range(M):
        n = np.random.rand()
        F_and_n = np.append(F, n)
        F_and_n.sort()
        idx = F_and_n.tolist().index(n) + 1
        indices.append(idx)
    if sorted:
        indices.sort()
    return indices


if __name__ == '__main__':
    p = [0.1, 0.2, 0.7]
    M = int(1e4)

    np.random.seed(1234)
    indices = sample(M, p)
    counts = Counter(indices)
    # print(counts)
    keys = list(counts.keys())
    vals = list(counts.values())
    plt.bar(keys, vals)
    for k, v in zip(keys, vals):
        plt.text(k, v+1, v, ha='center', va='bottom')
    plt.xticks(range(1, len(p)+1))
    plt.title('Total sampling number: {}'.format(M))
    plt.xlabel('index')
    plt.ylabel('count')
    # plt.savefig('sample_bar.png',dpi=400)
    plt.show()
