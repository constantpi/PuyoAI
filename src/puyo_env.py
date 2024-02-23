from puyo_game import PuyoGame

import numpy as np
import torch


class PuyoEnv:
    def __init__(self, width: int, height: int, puyo_colors: int) -> None:
        self.game = PuyoGame(width, height, puyo_colors)
        self.width = width
        self.height = height
        self.puyo_colors = puyo_colors
        self.action_space = 4 * width - 2
        self.board_shape = (width, height, puyo_colors+1)
        self.next_puyo_shape = (2, 2, puyo_colors+1)

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        erase_count_list = self.game.drop(action)
        board, puyo = self.game.get_state()
        board = torch.tensor(board, dtype=torch.float32)
        puyo = torch.tensor(puyo, dtype=torch.float32)

        reward = 5
        for i in range(len(erase_count_list) - 1):
            reward += (2 ** (i + 1)) * erase_count_list[i]
        reward = min(1, reward/100)
        done = self.game.is_game_over
        if done:
            reward = -1
        return (board, puyo), reward, done, {}

    def reset(self) -> np.ndarray:
        self.game = PuyoGame(self.width, self.height, self.puyo_colors)
        return self.game.board

    def render(self) -> None:
        print(self.game)


if __name__ == '__main__':
    env = PuyoEnv(6, 12, 4)
    env.reset()
    env.render()
    done = False
    while not done:
        action = np.random.randint(0, env.action_space)
        (board, puyo), reward, done, _ = env.step(action)
        print(board)
        print(puyo)
        print(reward)
        print(done)
        env.render()
        print()
