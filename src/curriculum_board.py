import numpy as np
import random


def make_curriculum_board(width: int, height: int, puyo_colors: int, level: int) -> np.ndarray:
    # 階段積みのボードを作成
    board = np.zeros((width, height), dtype=int)
    if level == 0:
        return board
    colors = []  # width - 1 の長さのリスト。隣合う要素が異なる色になるようにする
    now_color = 0
    for i in range(width - 1):
        color = now_color
        while color == now_color:
            color = random.randint(1, puyo_colors)
        colors.append(color)
        now_color = color
    basis = [[0] * 4 for _ in range(width)]
    for i in range(min(level, 3)):
        for j in range(width - 1):
            basis[j][i] = colors[j]
    if level > 3:
        for j in range(width - 2):
            basis[j+1][3] = colors[j]
    a = b = 0
    while b - a < 3:
        a = random.randint(0, width - 1)
        b = random.randint(0, width - 1)
    for i in range(a, b):
        for j in range(4):
            board[i][height - 1 - j] = basis[i][j]
    if random.randint(0, 1):
        board = np.flip(board, axis=0)
    return board


if __name__ == '__main__':
    print(make_curriculum_board(6, 14, 4, 4))
