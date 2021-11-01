# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
from scipy.stats import skellam, poisson
import tqdm


# %%
class jacksCarRentals:
    def __init__(self):
        self.carsInEachLocation = [10, 10]
        self.isTerminated = False

    def interact(self, numberOfCarsToMove):
        if self.isTerminated:
            return 0, True

        self.carsInEachLocation[0] -= numberOfCarsToMove
        self.carsInEachLocation[1] += numberOfCarsToMove

        requestedCars = np.random.poisson([3, 4])
        returnedCars = np.random.poisson([3, 2])

        self.carsInEachLocation += returnedCars - requestedCars
        self.carsInEachLocation = np.where(
            self.carsInEachLocation > 20, 20, self.carsInEachLocation)
        if any(self.carsInEachLocation < 0):
            self. isTerminated = True

        reward = np.sum(requestedCars) * 10 - numberOfCarsToMove*2
        return reward, self.isTerminated


# %%

class Agent:
    def __init__(self):
        self.vValues = np.zeros((20, 20))
        self.policy = np.zeros((20, 20))

    def __probOfReq(self, requestedCars, diffrence):
        p1 = poisson(3+2)
        p2 = poisson(3+4)
        sk = skellam(3+2, 3+4)
        return p1.pmf(requestedCars+diffrence)*p2.pmf(requestedCars)/sk.pmf(diffrence)

    # def expectedReward(self,s,a,sP):
    #     dif = np.sum(sP) - np.sum(s)

    #     if s[1]+a > 20:
    #         dif -= s[1]+a - 20

    #     expectedReq = 0
    #     for req in range(2,12):
    #         ret = dif + req
    #         if ret <0 :
    #             continue

    #         expectedReq += self.__probOfReq(req,dif)*req

    #     return expectedReq*10 - a*2

    def evaluatePolicy(self, iterations=10):
        Diff1 = skellam(3, 3)
        Diff2 = skellam(2, 4)

        for _ in range(iterations):
            for i in range(self.vValues.shape[0]):
                for j in range(self.vValues.shape[1]):
                    a = self.policy[i, j]
                    expReward = 70 - 2*a

                    A = Diff1.pmf(np.arange(20)-i+a)
                    A[19] = 1-Diff1.cdf(19-i+a)
                    B = Diff2.pmf(np.arange(20)-j-a)
                    B[19] = 1-Diff2.cdf(19-j-a)

                    probs = A.reshape(-1, 1) @ B.reshape(1, -1)

                    self.vValues[i, j] = np.sum(
                        self.vValues * probs) * 0.9 + expReward

    def policyItration(self):
        Diff1 = skellam(3, 3)
        Diff2 = skellam(2, 4)

        for i in range(self.vValues.shape[0]):
            for j in range(self.vValues.shape[1]):

                neighborValues = []
                vs = []
                for a in range(0, 5):
                    iP, jP = i-a, j+a
                    if iP < 0 or jP > 20:
                        continue

                    A = Diff1.pmf(np.arange(20)-i+a)
                    A[19] = 1-Diff1.cdf(19-i+a)
                    B = Diff2.pmf(np.arange(20)-j-a)
                    B[19] = 1-Diff2.cdf(19-j-a)
                    probs = A.reshape(-1, 1) @ B.reshape(1, -1)
                    vs.append(np.sum(self.vValues * probs))

                self.policy[i, j] = np.argmax(vs)


# %%
ag = Agent()
for _ in tqdm.trange(5):
    ag.evaluatePolicy()
    ag.policyItration()


# %%
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid([range(20)], [range(20)])
ha.plot_surface(X, Y, ag.vValues)
plt.show()

# %% [markdown]
#

# %%
np.concatenate([np.zeros([2]), [2]])


# %%
