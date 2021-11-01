import numpy as np
import numba
import random


class Board:
    def __init__(self, board_size) -> None:
        self.boardMap = np.zeros(board_size, dtype=np.uint8)

    def reset(self):
        self.boardMap = np.zeros(self.boardMap.shape, dtype=np.uint8)

    def put(self, mark, pos):
        if self.boardMap[pos[0], pos[1]] != 0:
            raise Exception(f'{pos[0]+1},{pos[1]+1} is taken before!')

        self.boardMap[pos[0], pos[1]] = mark

    def humanTurn(self):
        print(self.boardMap)
        s = input('your move: ')
        pos = [int(temp)-1 for temp in s.split(',')]
        self.put(1, pos)

    def howIsTheWinner(self):
        return checkWin(self.boardMap)

    def isBoardFull(self):
        return np.where(self.boardMap == 0, 1, 0).sum() == 0

    def populateBoard(self, ratio):

        n = int(ratio*(self.boardMap.size)/2)
        counter1 = 0
        counter2 = 0

        while(counter1 != n or counter2 != n):
            for i in range(self.boardMap.shape[0]):
                for j in range(self.boardMap.shape[1]):
                    if self.boardMap[i, j] != 0:
                        continue

                    rand = random.random()
                    if rand < (ratio/2) and counter1 < n:
                        self.boardMap[i, j] = 1

                        winInOneAct = False
                        for _, _, sPrime in expand(self.boardMap, 1):
                            winInOneAct = winInOneAct or checkWin(sPrime) == 1

                        if checkWin(self.boardMap) == 0 and not winInOneAct:
                            counter1 += 1
                        else:
                            self.boardMap[i, j] = 0

                    elif rand < (ratio/2)*2 and counter2 < n:
                        self.boardMap[i, j] = 2

                        winInOneAct = False
                        for _, _, sPrime in expand(self.boardMap, 2):
                            winInOneAct = winInOneAct or checkWin(sPrime) == 2

                        if checkWin(self.boardMap) == 0 and not winInOneAct:
                            counter2 += 1
                        else:
                            self.boardMap[i, j] = 0


@numba.njit()
def checkWin(state) -> int:
    counter1 = 0
    counter2 = 0

    for i in range(state.shape[0]):
        for j in range(state.shape[1]):

            if state[i, j] == 1:
                counter1 += 1
            else:
                counter1 = 0

            if state[i, j] == 2:
                counter2 += 1
            else:
                counter2 = 0

            if counter1 == 3:
                return 1
            elif counter2 == 3:
                return 2

        counter1, counter2 = 0, 0

    counter1, counter2 = 0, 0
    state = state.T

    for i in range(state.shape[0]):
        for j in range(state.shape[1]):

            if state[i, j] == 1:
                counter1 += 1
            else:
                counter1 = 0

            if state[i, j] == 2:
                counter2 += 1
            else:
                counter2 = 0

            if counter1 == 3:
                return 1
            elif counter2 == 3:
                return 2

        counter1, counter2 = 0, 0

    return 0


@numba.njit()
def hashState(board) -> int:
    res = 0
    for cell in board.reshape(-1):
        res = cell + res << 2
    return res


@numba.njit()
def expand(ref, mark):
    res = ref.copy()

    for i in range(ref.shape[0]):
        for j in range(ref.shape[1]):

            if res[i, j] == 0:
                res[i, j] = mark
                yield i, j, res
                res[i, j] = 0
