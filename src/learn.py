from puyo_env import PuyoEnv
from replay_buffer import PrioritizedReplayBuffer
from q_network import CNNQNetwork

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import random
from datetime import datetime
import pickle


def update(batch_size: int, beta: float) -> float:
    board, puyo, action, reward, next_board, next_puyo, done, indices, weights = replay_buffer.sample(batch_size, beta)
    board, puyo, action, reward, next_board, next_puyo, done, weights \
        = board.float().to(device), puyo.float().to(device), action.to(device), reward.to(device), next_board.float().to(device), next_puyo.float().to(device), done.to(device), weights.to(device)

    # 　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(board, puyo).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN.
        # >> 演習: Double DQNのターゲット価値の計算を実装してみましょう
        # ① 現在のQ関数でgreedyに行動を選択し,
        greedy_action_next = torch.argmax(net(next_board, next_puyo), dim=1).unsqueeze(1)
        # ②　対応する価値はターゲットネットワークのものを参照します.
        q_values_next = target_net(next_board, next_puyo).gather(1, greedy_action_next).squeeze(1)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = (1 - gamma) * reward + gamma * q_values_next * (1 - done)

    # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
    optimizer.zero_grad()
    loss = (weights * loss_func(q_values, target_q_values)).mean()
    loss.backward()
    optimizer.step()

    # 　TD誤差に基づいて, サンプルされた経験の優先度を更新します.
    replay_buffer.update_priorities(indices, (target_q_values - q_values).abs().detach().cpu().numpy())

    return loss.item()


"""
    環境の宣言
"""
env = PuyoEnv(width=6, height=14, puyo_colors=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    リプレイバッファの宣言
"""
buffer_size = 100000  # 　リプレイバッファに入る経験の最大数
initial_buffer_size = 3000  # 学習を開始する最低限の経験の数
replay_buffer = PrioritizedReplayBuffer(buffer_size)


"""
    ネットワークの宣言
"""
net = CNNQNetwork(board_shape=env.board_shape, next_puyo_shape=env.next_puyo_shape, n_action=env.action_space).to(device)
target_net = CNNQNetwork(board_shape=env.board_shape, next_puyo_shape=env.next_puyo_shape, n_action=env.action_space).to(device)
target_update_interval = 2000  # 学習安定化のために用いるターゲットネットワークの同期間隔


"""
    オプティマイザとロス関数の宣言
"""
optimizer = optim.Adam(net.parameters(), lr=1e-4)  # オプティマイザはAdam
loss_func = nn.SmoothL1Loss(reduction='none')  # ロスはSmoothL1loss（別名Huber loss）


"""
    Prioritized Experience Replayのためのパラメータβ
"""
beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000


def beta_func(step: int) -> float:  # beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
    return min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))


"""
    探索のためのパラメータε
"""
epsilon_begin = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
# epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす


def epsilon_func(step: int) -> float:
    return max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))


"""
    その他のハイパーパラメータ
"""
gamma = 0.7  # 　割引率
batch_size = 32
n_episodes = 30000  # 学習を行うエピソード数
curriculum = False  # カリキュラム学習を行うかどうか
"""
    学習の実行
"""
step = 0
total_reward_list = []
mean_reward_list = []
model_cnt = 0
max_level = 4
level = max_level
file_name = datetime.now().strftime('%Y%m%d%H%M%S')+".txt"
for episode in range(n_episodes):
    board, puyo = env.reset(level=level)
    board = torch.tensor(board, dtype=torch.float32)
    puyo = torch.tensor(puyo, dtype=torch.float32)
    done = False
    total_reward = 0

    while not done:
        # ε-greedyで行動を選択
        action = net.act(board.float().to(device), puyo.float().to(device),
                         epsilon_func(step))
        # 環境中で実際に行動
        next_board, next_puyo, reward, done, _ = env.step(action)
        total_reward += reward

        # リプレイバッファに経験を蓄積
        replay_buffer.push([board, puyo, action, reward, next_board, next_puyo, done])
        board = next_board
        puyo = next_puyo

        # ネットワークを更新
        if len(replay_buffer) > initial_buffer_size:
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())
            # ネットワークの重みを保存
            if model_cnt % 20 == 0:
                torch.save(net.state_dict(), f'/root/main/models/model_{model_cnt}.pth')
            model_cnt += 1

        step += 1

    print('Episode: {},  Step: {}, Level: {}, Reward: {}'.format(episode + 1, step + 1, level, total_reward))
    total_reward_list.append(total_reward)
    if not curriculum:
        level = 0
    elif total_reward > 2.5:
        level -= 1
        if level < 0:
            level = random.randint(0, max_level)
    elif total_reward < 1.5:
        level = min(max_level, level + 1)
    if len(total_reward_list) >= 100:
        mean_reward = sum(total_reward_list[:100]) / 100
        mean_reward_list.append(mean_reward)
        total_reward_list = total_reward_list[100:]
        plt.plot(mean_reward_list)
        plt.xlabel('100 Episode')
        plt.ylabel('Mean Reward')
        plt.savefig('mean_reward.png')
        plt.clf()
    with open(file_name, 'wb') as f:
        pickle.dump(mean_reward_list, f)
