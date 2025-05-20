import Goban

WHITE = 0
BLACK = 1


class Heuristic:

    def __init__(self, board, mycolor) -> None:
        self._board = board
        self._mycolor = mycolor

    def compute_heuristic(self, color) -> int:
        white, black = 0, 0
        for line in range(8):
            for column in range(8):
                if self._board[Goban.Board.flatten((line, column))] == color:
                    if color == WHITE:
                        white += 1
                    else:
                        black += 1
        return (black - white) if color == BLACK else (white - black)

    def alpha_beta(self) -> int:
        self.compute_heuristic(self._mycolor)

        return 1
