from puyo_game import PuyoGame

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class PuyoEnv:
    def __init__(self, width: int, height: int, puyo_colors: int) -> None:
        self.game = PuyoGame(width, height, puyo_colors)
        self.width = width
        self.height = height
        self.puyo_colors = puyo_colors
        self.action_space = 4 * width - 2
        self.board_shape = (width, height, puyo_colors+1)
        self.next_puyo_shape = (2, 2, puyo_colors+1)

    def step(self, action: int, record=False) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        erase_count_list = self.game.drop(action, record)
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
        return board, puyo, reward, done, {}

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.game = PuyoGame(self.width, self.height, self.puyo_colors)
        return self.game.get_state()

    def render(self) -> None:
        print(self.game)

    def save_image(self, num=0) -> int:
        board_history = self.game.board_history
        colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 30)
        for board, puyo, next_puyo, rensa in board_history:
            img = Image.new('RGB', (self.width*60+100, (self.height)*60), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            print(board, board.shape)
            for i in range(self.width):
                for j in range(1, self.height):
                    color = board[i][j]
                    draw.rectangle((i*60, (j-1)*60, (i+1)*60, j*60), fill=(255, 255, 255), outline=(0, 0, 0))
                    if color != 0:
                        draw.ellipse((i*60+10, (j-1)*60+10, (i+1)*60-10, j*60-10), fill=colors[color], outline=(0, 0, 0))
            if rensa:
                draw.text((self.width*60+5, 10), f'連鎖: {rensa}', fill=(0, 0, 0), font=font)

            for i in range(2):
                for j in range(2):
                    color = puyo[j]
                    x = self.width*60 + 30
                    y = 100 + i*200 + j*60
                    # 縦に並べる
                    draw.rectangle((x, y, x+60, y+60), fill=(255, 255, 255), outline=(0, 0, 0))
                    if color != 0:
                        draw.ellipse((x+10, y+10, x+50, y+50), fill=colors[color], outline=(0, 0, 0))
                puyo = next_puyo
            img.save(f'./images/{num}.png')
            num += 1
            print(num)


if __name__ == '__main__':
    env = PuyoEnv(6, 14, 4)
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
