import numpy as np
import random


class PuyoGame:

    def __init__(self, width: int, height: int, puyo_colors: int) -> None:
        self.width = width
        self.height = height
        self.puyo_colors = puyo_colors
        self.board = np.zeros((width, height), dtype=int)
        self.current_puyo = (random.randint(1, puyo_colors),
                             random.randint(1, puyo_colors))
        self.next_puyo = (random.randint(1, puyo_colors),
                          random.randint(1, puyo_colors))
        self.score = 0
        self.is_game_over = False

    def drop(self, action: int) -> None:
        """
        actionが0から2*width-1の範囲のときは縦にぷよを落とす
        actionが2*widthから4*width-3の範囲のときは横向きでぷよを落とす
        """
        puyo = self.current_puyo
        self.current_puyo = self.next_puyo
        self.next_puyo = (random.randint(1, self.puyo_colors),
                          random.randint(1, self.puyo_colors))
        assert 0 <= action < 4 * self.width - 2
        if action < 2 * self.width:
            if action < self.width:
                x = action  # 0からwidth-1
            else:
                x = action - self.width  # 0からwidth-1
                puyo = (puyo[1], puyo[0])
            y = 0
            while y < self.height and self.board[x][y] == 0:
                y += 1
            self.board[x][y - 1] = puyo[0]
            self.board[x][y - 2] = puyo[1]
        else:
            if action < 3*self.width-1:
                x = action - 2*self.width
            else:
                x = action - 3*self.width + 1
                puyo = (puyo[1], puyo[0])
            y = 0
            while y < self.height and self.board[x][y] == 0:
                y += 1
            self.board[x][y - 1] = puyo[0]
            x += 1
            y = 0
            while y < self.height and self.board[x][y] == 0:
                y += 1
            self.board[x][y - 1] = puyo[1]
        erase_count_list = [self.erase()]
        while erase_count_list[-1] > 0:
            erase_count_list.append(self.erase())
        self.is_game_over = self.game_over()
        return erase_count_list

    def erase(self) -> None:
        """
        浮いているぷよを落とす
        つながっている同色のぷよを消す
        """
        new_board = np.zeros((self.width, self.height), dtype=int)
        for i in range(self.width):
            puyo_list = []
            for j in range(self.height-1, -1, -1):
                if self.board[i][j] != 0:
                    puyo_list.append(self.board[i][j])
            for j in range(len(puyo_list)):
                new_board[i][self.height - j - 1] = puyo_list[j]
        self.board = new_board

        def dfs(x, y, color):
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return 0
            if visited[x][y] or self.board[x][y] != color:
                return 0
            visited[x][y] = True
            group.append((x, y))
            return 1 + dfs(x - 1, y, color) + dfs(x + 1, y, color) + dfs(x, y - 1, color) + dfs(x, y + 1, color)
        erase_list = []
        visited = np.zeros((self.width, self.height), dtype=bool)
        for i in range(self.width):
            for j in range(self.height):
                group = []
                if self.board[i][j] == 0 or visited[i][j]:
                    continue
                if dfs(i, j, self.board[i][j]) >= 4:
                    erase_list.append(group)
        erase_count = 0
        for group in erase_list:
            for x, y in group:
                erase_count += 1
                self.board[x][y] = 0
        return erase_count

    def game_over(self) -> bool:
        for i in range(self.width):
            if self.board[i][0] != 0 or self.board[i][1] != 0:
                self.game_over = True
                return True
        return False

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        boards = []
        for i in range(self.puyo_colors+1):
            boards.append((self.board == i).astype(int))
        decorded_next_puyo = np.zeros((2, 2, self.puyo_colors+1), dtype=int)
        decorded_next_puyo[0][0][self.current_puyo[0]] = 1
        decorded_next_puyo[0][1][self.current_puyo[1]] = 1
        decorded_next_puyo[1][0][self.next_puyo[0]] = 1
        decorded_next_puyo[1][1][self.next_puyo[1]] = 1
        return np.array(boards), decorded_next_puyo

    def __str__(self):
        board_str = ""
        for i in range(self.height):
            for j in range(self.width):
                board_str += str(self.board[j][i])
            board_str += "\n"
        return board_str


if __name__ == "__main__":
    game = PuyoGame(6, 14, 4)
    print(game.board)
    print(game.current_puyo)
    print(game.next_puyo)
    print(game.score)
    print(game.game_over)
    print(game)
    for i in range(100):
        act = random.randint(0, 21)
        print(act)
        print(game.current_puyo)
        result = game.drop(act)
        print(game)
        print(game.get_state())
        if game.is_game_over:
            break
        if len(result) > 1:
            print(result)
