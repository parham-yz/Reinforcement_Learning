import numpy as np
from numpy.core.numeric import Inf
from environment import *
import numba


class Agent():
    def __init__(self, mark, knowlage=None) -> None:
        self.afterStateValues = {} if knowlage is None else knowlage
        self.l = 1
        self.alpha = 0.4
        self.initialValue = 0
        self.mark = mark
        self.prevAfterState = None
        self.recentReward = 0
        self.epsilon = 0.1
        self.isGameTerminated = False
        self.new_xps = {}
        self.score = 0

    def deside(self, state):
        if self.prevAfterState is not None:
            if self.isGameTerminated:
                self.learn(
                    [self.prevAfterState, None, self.recentReward])
            else:
                self.learn(
                    [self.prevAfterState, state, self.recentReward])
            self.recentReward = 0

        if self.isGameTerminated:
            self.prevAfterState = None
            self.recentReward = 0
            return (-1, -1)

        i, j = self.__argmaxOfAfterStateValues(state)

        self.prevAfterState = state.copy()
        self.prevAfterState[i, j] = self.mark

        if random.random() > self.epsilon:
            return i, j
        else:
            Is, Js = np.where(state == 0)
            # print(len(Is))
            index = random.randint(0, len(Is)-1)
        return Is[index], Js[index]

    def giveReward(self, r, isTerminated=False):
        self.recentReward = r
        self.isGameTerminated = isTerminated

    def setEpsilon(self, e):
        self.epsilon = e

    def reset(self):
        self.isGameTerminated = False

    def learn(self, xp):

        f, sP, r = xp

        if sP is None:
            new_value = 0 + r
        else:
            new_value = self.l * self.__maxOfAfterStateValues(sP) + r

        f_hash = hashState(f)
        if f_hash in self.afterStateValues:

            self.new_xps[f_hash] = self.alpha * \
                (new_value - self.afterStateValues[f_hash])
        else:
            self.new_xps[f_hash] = self.initialValue + \
                self.l * new_value

        self.alpha = 0.95 * self.alpha if self.alpha > 0.05 else 0.05

    def syncKnowlage(self):
        for f_hash in self.new_xps.keys():
            if f_hash in self.afterStateValues:
                self.afterStateValues[f_hash] += self.new_xps[f_hash]
            else:
                self.afterStateValues[f_hash] = self.initialValue + \
                    self.new_xps[f_hash]

        self.new_xps.clear()

    def __maxOfAfterStateValues(self, state):

        max_vf = -Inf
        for _, _, f in expand(state, self.mark):
            value = self.afterStateValues[hashState(f)] if hashState(
                f) in self.afterStateValues else self.initialValue

            if value > max_vf:
                max_vf = value

        if max_vf == -Inf:
            raise Exception('making decition on a terminal sate ',
                            self.isGameTerminated, self.prevAfterState)

        return max_vf

    def __argmaxOfAfterStateValues(self, state):

        max_case = [-1, -1, -Inf]
        for i, j, f in expand(state, self.mark):
            value = self.afterStateValues[hashState(f)] if hashState(
                f) in self.afterStateValues else self.initialValue*(1+random.random())

            if max_case[2] < value and state[i, j] == 0:
                max_case = [i, j, value]

        return max_case[0], max_case[1]
