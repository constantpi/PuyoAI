from puyo_env import PuyoEnv
from q_network import CNNQNetwork

import torch
import argparse
import glob
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--model', type=str, default='model_*.pth')
args = parser.parse_args()
n_episodes = args.n
model_name = args.model
print(n_episodes, "回のプレイします")

env = PuyoEnv(width=6, height=14, puyo_colors=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = CNNQNetwork(board_shape=env.board_shape, next_puyo_shape=env.next_puyo_shape, n_action=env.action_space).to(device)
# 指定されていなければmodel_ の中で最も大きい数字のものを選ぶ
path = '/root/main/models/' + model_name
model_list = glob.glob(path)
model_list.sort()
model_path = model_list[-1]
print('model_path:', model_path)
net.load_state_dict(torch.load(model_path))

step = 0
model_cnt = 0
colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for episode in range(n_episodes):
    board, puyo = env.reset()
    board = torch.tensor(board, dtype=torch.float32)
    puyo = torch.tensor(puyo, dtype=torch.float32)
    done = False
    total_reward = 0
    img = Image.new('RGB', (6*60+100, 13*60), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    while not done:
        # 行動を選択
        action = net.act(board.float().to(device), puyo.float().to(device), 0.0)
        # 環境中で実際に行動
        next_board, next_puyo, reward, done, _ = env.step(action)
        total_reward += reward
        board = next_board
        puyo = next_puyo
        step += 1
        env.render()
        # boardの画像を作成
        for i in range(6):
            for j in range(13):
                color = 0
                for c in range(1, 5):
                    if board[c][i][j] == 1:
                        color = c
                if color == 0:
                    draw.rectangle((i*60, j*60, (i+1)*60, (j+1)*60), fill=(255, 255, 255), outline=(0, 0, 0))
                else:
                    draw.rectangle((i*60, j*60, (i+1)*60, (j+1)*60), fill=(255, 255, 255), outline=(0, 0, 0))
                    # draw.ellipse((i*60+10, j*60+10, (i+1)*60-10, (j+1)*60-10), fill=(255, 255, 255), outline=(0, 0, 0))
                    draw.ellipse((i*60+15, j*60+15, (i+1)*60-15, (j+1)*60-15), fill=colors[color], outline=(0, 0, 0))
        img.save(f'/root/main/src/images/{step}.png')

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
