from puyo_env import PuyoEnv
from q_network import CNNQNetwork

import torch
import argparse
import glob
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=int, default=0)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--model', type=str, default='model_*.pth')
args = parser.parse_args()
level = args.level
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
for episode in range(n_episodes):
    board, puyo = env.reset(level=level)
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
        next_board, next_puyo, reward, done, _ = env.step(action, record=True)
        total_reward += reward
        board = next_board
        puyo = next_puyo
        step += 1
        env.render()

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
env.save_image()
