from puyo_env import PuyoEnv
from replay_buffer import PrioritizedReplayBuffer
from q_network import CNNQNetwork

import torch
from torch import nn, optim


def update(batch_size, beta):
    obs, action, reward, next_obs, done, indices, weights = replay_buffer.sample(batch_size, beta)
    obs, action, reward, next_obs, done, weights \
        = obs.float().to(device), action.to(device), reward.to(device), next_obs.float().to(device), done.to(device), weights.to(device)

    # 　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
    q_values = net(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # 目標値の計算なので勾配を追跡しない
    with torch.no_grad():
        # Double DQN.
        # >> 演習: Double DQNのターゲット価値の計算を実装してみましょう
        # ① 現在のQ関数でgreedyに行動を選択し,
        greedy_action_next = torch.argmax(net.forward(next_obs), dim=1).unsqueeze(1)
        # print(net.forward(next_obs))
        # print(greedy_action_next)
        # ②　対応する価値はターゲットネットワークのものを参照します.
        q_values_next = target_net.forward(next_obs).gather(1, greedy_action_next).squeeze(1)
        # print(target_net.forward(next_obs))
        # print(q_values_next)

    # ベルマン方程式に基づき, 更新先の価値を計算します.
    # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
    target_q_values = (1-gamma)*reward + gamma * q_values_next * (1 - done)
    # print("target_q_values",q_values)

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
print(env.action_space)
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
# beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
def beta_func(step): return min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))


"""
    探索のためのパラメータε
"""
epsilon_begin = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
# epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
def epsilon_func(step): return max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))


"""
    その他のハイパーパラメータ
"""
gamma = 0.99  # 　割引率
batch_size = 32
n_episodes = 30000  # 学習を行うエピソード数


"""
    学習の実行
"""

step = 0
for episode in range(n_episodes):
    board, puyo = env.reset()
    # obs = torch.tensor(obs, dtype=torch.float32)
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
        # replay_buffer.push([obs, action, reward, next_obs, done])
        replay_buffer.push([board, puyo, action, reward, next_board, next_puyo, done])
        # obs = next_obs
        board = next_board
        puyo = next_puyo

        # ネットワークを更新
        if len(replay_buffer) > initial_buffer_size:
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())

        step += 1

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
