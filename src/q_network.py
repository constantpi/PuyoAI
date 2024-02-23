"""
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します.
"""
import random
import torch
import torch.nn as nn


class CNNQNetwork(nn.Module):
    def __init__(self, board_shape: tuple[int, int, int], next_puyo_shape: tuple[int, int, int], n_action: int) -> None:
        super(CNNQNetwork, self).__init__()
        self.board_shape = board_shape  # (width, height, puyo_colors+1) : (6, 14, 5)
        self.next_puyo_shape = next_puyo_shape  # (2, 2, puyo_colors+1) : (2, 2, 5)
        self.n_action = n_action  # 行動の数:22
        # Dueling Networkでも, 畳込み部分は共有する
        self.conv_layers = nn.Sequential(
            nn.Conv2d(board_shape[2], 32, kernel_size=3, stride=1, padding=1),  # (6, 14, 5) -> (6, 14, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (6, 14, 32) -> (6, 14, 64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (6, 14, 64) -> (6, 14, 64)
            nn.ReLU()
        )
        cnn_out_size = board_shape[0] * board_shape[1] * 64

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            nn.Linear(3136 + next_puyo_shape[0] * next_puyo_shape[1] * next_puyo_shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_action)
        )

    def forward(self, board: torch.Tensor, next_puyo: torch.Tensor) -> torch.Tensor:
        feature = self.conv_layers(board)
        feature = feature.view(feature.size(0), -1)  # 　Flatten. (B, C, H, W) -> (B, C*H*W)

        state_values = self.fc_state(feature)

        next_puyo = next_puyo.view(next_puyo.size(0), -1)  # Flatten.
        advantage_input = torch.cat([feature, next_puyo], dim=1)
        advantage = self.fc_advantage(advantage_input)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        # tanhを適用して出力
        return torch.tanh(action_values)

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します.
    def act(self, board: torch.Tensor, next_puyo: torch.Tensor, epsilon: float) -> int:
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(board, next_puyo)).item()  # unsqueeze(0)を追加する必要がある？
        return action
