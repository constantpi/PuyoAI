from puyo_env import PuyoEnv
from replay_buffer import PrioritizedReplayBuffer
from q_network import CNNQNetwork

import torch
from torch import nn, optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1)
args = parser.parse_args()
n_episodes = args.n
print(n_episodes, "回のプレイします")

env = PuyoEnv(width=6, height=14, puyo_colors=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = CNNQNetwork(board_shape=env.board_shape, next_puyo_shape=env.next_puyo_shape, n_action=env.action_space).to(device)
net.load_state_dict(torch.load('/root/main/models/model_60.pth'))


step = 0
model_cnt = 0
for episode in range(n_episodes):
    board, puyo = env.reset()
    board = torch.tensor(board, dtype=torch.float32)
    puyo = torch.tensor(puyo, dtype=torch.float32)
    done = False
    total_reward = 0

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

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
