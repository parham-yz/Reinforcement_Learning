import numba
from environment import *
import numpy as np
import sys
sys.path.append("..")

de = 0


class Agent():
    def __init__(self) -> None:
        self.probebilities_cache = np.array([[-1, 0]], dtype=np.float32)

    def deside(self, state):
        bestAct = (0, None)
        expected_prob = 0
        n = 0

        if np.sum(np.where(state == 0, 1, 0)) == 1:
            # its the last move
            return int(np.where(state == 0)[0]), int(np.where(state == 0)[1])

        # searchin on all possible states after tow moves
        for i, j, sPrime in expand(state, 2):
            if checkWin(sPrime):
                return i, j

            for _, _, sZegonde in expand(sPrime, 1):
                expected_prob += self.evaluateStateProbebility(sZegonde)
                n += 1

            expected_prob /= n
            if bestAct[0] <= expected_prob:
                bestAct = (expected_prob, i, j)

        return bestAct[1:]

    def evaluateStateProbebility(self, state):
        # evalProb warpper

        # if len(self.probebilities_cache)-len(np.unique(self.probebilities_cache[:, 0])) > 0:
        #     raise Exception('not uinque!')

        res, self.probebilities_cache = evalProb(
            state, self.probebilities_cache)
        return res


@numba.njit()
def evalProb(state, cache):
    if len(cache) % 1000 == 0:
        print('number of evaluated states: ', len(cache))

    hashOfState = hashState(state)

    isFound, value = _find(cache, hashOfState)
    if isFound:
        return value, cache

    if not np.any(state == 0):
        cache = _pushToCache(cache, (hashOfState, 0.0))
        return 0.0, cache
    elif checkWin(state) == 2:
        cache = _pushToCache(cache, (hashOfState, 1.0))
        return 1.0, cache
    elif checkWin(state) == 1:
        cache = _pushToCache(cache, (hashOfState, 0.0))
        return 0.0, cache

    elif np.sum(np.where(state == 0, 1, 0)) == 1:
        # there is only one empty position left

        probs_on_acts = []
        for _, _, sPrime in expand(state, 2):

            value, cache = evalProb(sPrime, cache)
            probs_on_acts.append(value)
        cache = _pushToCache(cache, (hashOfState, max(probs_on_acts)))
        return max(probs_on_acts), cache

    ######
    expected_prob = 0.0
    n = 0
    probs_on_acts = []

    # searchin on all possible states after tow moves
    for _, _, sPrime in expand(state, 2):
        if checkWin(sPrime) == 2:
            cache = _pushToCache(cache, (hashOfState, 1.0))
            return 1.0, cache

        for _, _, sZegonde in expand(sPrime, 1):

            value, cache = evalProb(sZegonde, cache)
            expected_prob += value
            n += 1

        expected_prob /= n
        probs_on_acts.append(expected_prob)

    cache = _pushToCache(cache, (hashOfState, max(probs_on_acts)))
    return max(probs_on_acts), cache


@numba.njit()
def _pushToCache(cache, entry):

    temp = np.array(entry, dtype=np.float32).reshape((1, 2))
    cache = np.concatenate((cache, temp), axis=0)
    return cache
    # print(np.array(entry, dtype=np.float32).reshape((1, 2)))


@ numba.njit()
def _find(cache, key):
    indeces = np.where(cache[:, 0] == key)[0]
    if len(indeces) == 0:
        return False, 0

    return True, cache[indeces[0], 1]


# cache = np.array([[-1, 0]], dtype=np.float32)
# cache = _pushToCache(cache, (2, 0.4))
# a = _find(cache, 2)
# print(a)
# cache = _pushToCache(cache, (3, 0.45))
# cache = _pushToCache(cache, (4, 0.455))
# a = _find(cache, 3)
# print(a)
# cache = _pushToCache(cache, (5, 0.46))


# a = _find(cache, 4)
# print(a)
# a = _find(cache, 5)
# print(a)
