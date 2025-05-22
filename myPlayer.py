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
from palluat_pereira_pedro import GoCNN, position_predict

WHITE = 2  # Correction: doit correspondre aux constantes de Goban
BLACK = 1


def board_to_tensor(board: Goban.Board, current_color: int):
    size = 8
    black_plane = torch.zeros(size, size, dtype=torch.float32)
    white_plane = torch.zeros(size, size, dtype=torch.float32)

    for y in range(size):
        for x in range(size):
            flat_index = y * size + x
            stone = board._board[flat_index]
            if stone == Goban.Board._BLACK:  # Utiliser les constantes de Goban
                black_plane[y][x] = 1.0
            elif stone == Goban.Board._WHITE:
                white_plane[y][x] = 1.0

    return torch.stack([black_plane, white_plane]).unsqueeze(0)

class myPlayer(PlayerInterface):
    """Example of a random player for the go. The only tricky part is to be able to handle
    the internal representation of moves given by legal_moves() and used by push() and
    to translate them to the GO-move strings "A1", ..., "J8", "PASS". Easy!

    """

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.maxDepth = 3
        self.model = GoCNN()
        self.model.load_state_dict(torch.load("best_go_model.pth"))
        self.model.eval()

    def getPlayerName(self):
        return "GoGoDanceur"

    def evaluate_board(self, board: Goban.Board, color: int) -> float:
        input_tensor = board_to_tensor(board, color)
        with torch.no_grad():
            prediction = self.model(input_tensor).item()
        return prediction

    def alphaBeta(
        self, board, depth: int, alpha: float, beta: float, isMaximating: bool
    ):
        if depth == 0 or board.is_game_over():
            score = self.evaluate_board(board, self._mycolor)
            if self._mycolor == Goban.Board._WHITE:
                score = 1.0 - score
            return score, None

        moves = board.legal_moves()
        if not moves:
            return self.evaluate_board(board, self._mycolor), None
            
        bestMove = moves[0]  
        
        if isMaximating:
            maxEval = float("-inf")
            for move in moves:
                if not board.push(move):  
                    continue
                eval_score, _ = self.alphaBeta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > maxEval:
                    maxEval = eval_score
                    bestMove = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return maxEval, bestMove

        else:
            minEval = float("inf")
            for move in moves:
                if not board.push(move): 
                    continue
                eval_score, _ = self.alphaBeta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < minEval:
                    minEval = eval_score
                    bestMove = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return minEval, bestMove

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        
        moves = self._board.legal_moves()
        if(len(moves))<30:
            self.maxDepth=4
        else:
            self.maxDepth=3
        _, bestMove = self.alphaBeta(
            self._board, self.maxDepth, float("-inf"), float("inf"), True
        )
        
        print(f"Best move found: {bestMove}")
        
        if bestMove is None:
            print("No best move found, choosing randomly")
            moves = self._board.legal_moves()
            if moves:
                bestMove = choice(moves)
            else:
                return "PASS"
        if not self._board.push(bestMove):
            print("Best move was illegal, choosing randomly")
            moves = self._board.legal_moves()
            if moves:
                bestMove = choice(moves)
                self._board.push(bestMove)
            else:
                return "PASS"
        
        print("I am playing ", self._board.move_to_str(bestMove))
        print("My current board :")
        self._board.prettyPrint()
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
