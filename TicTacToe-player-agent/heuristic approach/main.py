from ai_heuristic import *
from environment import *
import sys
sys.path.append("..")


board = Board([4, 4])
board.populateBoard(0.5)


jimi = Agent()
print(board.boardMap)
while True:
    action = jimi.deside(board.boardMap)
    print()
    print("chance of agent's winning: ",
          jimi.evaluateStateProbebility(board.boardMap))
    board.put(2, action)
    if checkWin(board.boardMap) != 0:
        break
    board.humanTurn()
    if checkWin(board.boardMap) != 0:
        break


print('winner is :', checkWin(board.boardMap))
