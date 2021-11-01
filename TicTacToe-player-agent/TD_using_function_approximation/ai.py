from torch._C import dtype
from environment import *
import numpy as np
import numba
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")


def toOneHot(board):
    if type(board) != torch.Tensor:
        board = torch.tensor(board)
    res = torch.empty([3, board.shape[0], board.shape[1]], dtype=torch.float32)
    res[0] = torch.where(board == 0, 1, 0)
    res[1] = torch.where(board == 1, 1, 0)
    res[2] = torch.where(board == 2, 1, 0)
    return torch.unsqueeze(res, 0)


class dummyAgent():
    def __init__(self, mark, boardSize) -> None:
        self.boardSize = boardSize
        self.score = 0
        self.mark = mark
        self.isGameTerminated = False
        self.decay = 1

    def deside(self, state):
        if self.isGameTerminated:
            return (-1, -1)

        Is, Js = np.where(state == 0)
        index = random.randint(0, len(Is)-1)
        return Is[index], Js[index]

    def giveReward(self, r, isTerminated=False):
        self.isGameTerminated = isTerminated

    def setEpsilon(self, e):
        pass

    def reset(self):
        self.isGameTerminated = False


class Agent():
    def __init__(self, mark, boardSize) -> None:
        self.l = 0.9
        self.mark = mark
        self.prevAfterState = None
        self.recentReward = 0
        self.epsilon = 0.1
        self.isGameTerminated = False
        self.score = 0

        self.brain = Brain(boardSize, 0.001)
        self.batchSize = 16
        self.new_xps = {
            'states': [toOneHot(torch.zeros(boardSize))]*(self.batchSize*2),
            'new_values': [torch.tensor([0], dtype=torch.float32)]*(self.batchSize*2)
        }
        self.valueAble_xps = {
            'states': [toOneHot(torch.zeros(boardSize))]*(self.batchSize*2),
            'new_values': [torch.tensor([0], dtype=torch.float32)]*(self.batchSize*2)
        }
        self.num_untraindXps = -self.batchSize*4

    def deside(self, state):
        if self.prevAfterState is not None:
            if self.isGameTerminated:
                self.learn(
                    [self.prevAfterState, None, self.recentReward])
            else:
                self.learn(
                    [self.prevAfterState, state, self.recentReward])
            self.recentReward = 0

    ########
        if self.isGameTerminated:
            self.prevAfterState = None
            self.recentReward = 0
            return (-1, -1)

        action, _ = self.__maxOfAfterStateValues(state)

        self.prevAfterState = state.copy()
        self.prevAfterState[action[0], action[1]] = self.mark

        if random.random() > self.epsilon:
            return action
        else:
            Is, Js = np.where(state == 0)
            # print(len(Is))
            index = random.randint(0, len(Is)-1)
        return Is[index], Js[index]

    def learn(self, xp):
        f, sP, r = xp
        if sP is None:
            new_value = torch.tensor([0 + r], dtype=torch.float32)
        else:
            _, value = self.__maxOfAfterStateValues(sP)
            new_value = self.l * value + r

        if new_value >= 0.9 or new_value <= -0.9:
            self.valueAble_xps['states'] = self.valueAble_xps['states'][1:]
            self.valueAble_xps['new_values'] = self.valueAble_xps['new_values'][1:]
            self.valueAble_xps['states'].append(toOneHot(f))
            self.valueAble_xps['new_values'].append(new_value.clone().detach())

        else:
            self.new_xps['states'] = self.new_xps['states'][1:]
            self.new_xps['new_values'] = self.new_xps['new_values'][1:]
            self.new_xps['states'].append(toOneHot(f))
            self.new_xps['new_values'].append(new_value.clone().detach())
        self.num_untraindXps += 1

        if self.num_untraindXps >= self.batchSize/2:
            self.num_untraindXps = 0

    # chosing random batch
            indeices = torch.randint(self.batchSize*4, (self.batchSize,))
            x = torch.cat(self.new_xps['states'] +
                          self.valueAble_xps['states'])[indeices]
            y = torch.cat(self.new_xps['new_values'] +
                          self.valueAble_xps['new_values'])[indeices]

            self.de = x, y

            pred = self.brain(x).view(-1)
            loss = self.brain.lossFunc(pred, y)
            self.brain.learingHistory.append(float(loss))

            self.brain.opt.zero_grad()
            loss.backward()
            self.brain.opt.step()

    def __maxOfAfterStateValues(self, state):
        actions = []
        xs = []

        for i, j, f in expand(state, self.mark):
            xs.append(toOneHot(f))
            actions.append((i, j))

        pred = self.brain(torch.cat(xs, 0))
        indx = torch.argmax(pred)
        return actions[indx], pred[indx]

    def giveReward(self, r, isTerminated=False):
        self.recentReward = r
        self.isGameTerminated = isTerminated

    def setEpsilon(self, e):
        self.epsilon = e

    def reset(self):
        self.isGameTerminated = False


class Brain(nn.Module):
    def __init__(self, boardSize, alpha) -> None:
        super().__init__()
        self.bs = boardSize

        self.learingHistory = []
        self.cnv = nn.Conv2d(in_channels=3, out_channels=8,
                             kernel_size=3, padding=(1, 1))
        self.lin1 = nn.Linear(self.bs[0]*self.bs[1]*8, 8)
        self.lin2 = nn.Linear(8, 1)
        self.lossFunc = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, board):
        x = board
        x = F.relu(self.cnv(board))
        # x = F.max_pool2d(x, [2, 2])
        x = x.view(-1, self.bs[0]*self.bs[1]*8)
        x = F.relu(self.lin1(x))
        return self.lin2(x)


if __name__ == "__main__":
    env = Board([4, 4])
    jimi = Agent(2, env.boardMap.shape)
    jimi.setEpsilon(0.1)
    jimi.reset()
    env.reset()

    for _ in range(5):
        running = True
        while running:
            action = jimi.deside(env.boardMap)
            env.put(2, action)

            if env.howIsTheWinner() == 2:
                jimi.giveReward(1, True)
                print('\nWinner is 2')
                break
            elif env.isBoardFull():
                jimi.giveReward(0, True)
                break

            env.humanTurn()

            if env.howIsTheWinner() == 1:
                jimi.giveReward(-1, True)
                print('\nWinner is 1')
                break
            elif env.isBoardFull():
                jimi.giveReward(0, True)
                break

        jimi.deside(env.boardMap)

        jimi.reset()
        env.reset()
