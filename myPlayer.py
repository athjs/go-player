# -*- coding: utf-8 -*-
"""This is the file you have to modify for the tournament. Your default AI player must be called by this module, in the
myPlayer class.

Right now, this class contains the copy of the randomPlayer. But you have to change this!
"""

import time
import Goban
from random import choice
from playerInterface import *
import torch
from palluat_pereira_pedro import GoCNN

WHITE = 0
BLACK = 1


class myPlayer(PlayerInterface):
    """Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    """

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.model = GoCNN()
        self.model.load_state_dict(torch.load("./final_go_model.pth"))
        self.model.eval()

    def getPlayerName(self):
        return "GoGoDanceur"

    def evaluate_board(self, board, color):
        white, black = 0, 0
        for line in range(8):
            for column in range(8):
                if board[Goban.Board.flatten((line, column))] == color:
                    if self._mycolor == WHITE:
                        white += 1
                    else:
                        black += 1
        return (black - white) if color == BLACK else (white - black)

    def alphaBeta(
        self, board, depth: int, alpha: float, beta: float, isMaximating: bool
    ):
        if depth == 0 or self._board.is_game_over():
            return self.evaluate_board(self._board, self._mycolor), None

        moves = board.legal_moves()
        if isMaximating:
            bestMove = None
            maxEval = float("-inf")
            for move in moves:
                board.push(move)
                eval, _ = self.alphaBeta(board, depth - 1, alpha, beta, False)
                board.pop()
                if maxEval < eval:
                    maxEval = eval
                    bestMove = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return maxEval, bestMove

        else:
            bestMove = None
            minEval = float("inf")
            for move in moves:
                board.push(move)
                eval, _ = self.alphaBeta(board, depth - 1, alpha, beta, False)
                board.pop()
                if minEval > eval:
                    minEval = eval
                    bestMove = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return minEval, bestMove

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        _, bestMove = self.alphaBeta(self._board, 2, float("-inf"), float("inf"), True)
        if bestMove == None:
            moves = self._board.legal_moves()
            bestMove = choice(moves)
        self._board.push(bestMove)
        # New here: allows to consider internal representations of moves
        print("I am playing ", self._board.move_to_str(bestMove))
        print("My current board :")
        self._board.prettyPrint()
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(bestMove)

    def playOpponentMove(self, move):
        print("Opponent played ", move)  # New here
        # the board needs an internal represetation to push the move.  Not a string
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
