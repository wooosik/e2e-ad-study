import torch
import torch.nn as nn
import torch.nn.functional as F


class SupercomboModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 간단한 CNN 백본 정의 (입력: 12채널 128x256, 출력: feature_dim 차원 벡터)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [32, 32x64]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [64, 8x16]
            nn.Flatten(),
        )
        # Flatten 후 임베딩 차원 계산 (임의로 64*8*16=8192라 가정)
        self.feat_dim = 64 * 8 * 16
        self.fc_embed = nn.Sequential(nn.Linear(self.feat_dim, 512), nn.ReLU())
        # GRU 정의 (입력 512차원, 은닉 512차원)
        self.rnn = nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        # 출력 헤드 예시
        self.head_steer = nn.Linear(512, 1)
        self.head_trajectory = nn.Linear(512, 20)

    def forward(self, img_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass for a batch of image sequences."""
        if img_sequence.dim() == 5:
            b, t, c, h, w = img_sequence.shape
            img_sequence = img_sequence.view(b * t, c, h, w)
            feat = self.cnn(img_sequence)
            feat = self.fc_embed(feat)
            feat = feat.view(b, t, -1)
        else:
            feat = self.cnn(img_sequence)
            feat = self.fc_embed(feat)
            feat = feat.unsqueeze(1)

        rnn_out, _ = self.rnn(feat)
        final_feat = rnn_out[:, -1, :]
        steer_pred = self.head_steer(final_feat)
        traj_pred = self.head_trajectory(final_feat)
        return steer_pred, traj_pred


if __name__ == "__main__":
    model = SupercomboModel()
    print(model)
