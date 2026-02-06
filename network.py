import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        # Branch 1: Motion Branch (Processes Fx, Fy, Tx, Ty)
        self.motion_branch = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Branch 2: Compliance Branch (Processes Fz, Tz, and indicator I)
        self.compliance_branch = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # 輸出的解耦
        self.mu_v_xy = nn.Linear(128, 2)  # 輸出 vx, vy
        self.mu_k_z = nn.Linear(128, 1)  # 輸出 kz

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # 假設 state 順序: [Fx, Fy, Tx, Ty, Fz, Tz, I]
        s_motion = state[:, [0, 1, 2, 3]]  # Fx, Fy, Tx, Ty
        s_compliance = state[:, [4, 5, 6]]  # Fz, Tz, I

        # 物理角色解耦處理
        feat_motion = self.motion_branch(s_motion)
        feat_compliance = self.compliance_branch(s_compliance)

        # 輸出解耦
        v_xy = T.tanh(self.mu_v_xy(feat_motion))
        k_z = T.tanh(self.mu_k_z(feat_compliance))

        return T.cat([v_xy, k_z], dim=1)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        # 與 Actor 相同的結構化輸入處理
        self.motion_branch = nn.Linear(4, 128)
        self.compliance_branch = nn.Linear(3, 128)

        # Action 分解輸入
        self.action_v_xy = nn.Linear(2, 64)
        self.action_k_z = nn.Linear(1, 64)

        self.q = nn.Linear(128 + 128 + 64 + 64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        s_motion = state[:, [0, 1, 2, 3]]
        s_compliance = state[:, [4, 5, 6]]
        a_v_xy = action[:, [0, 1]]
        a_k_z = action[:, [2]]

        # 分支特徵提取
        m_feat = F.relu(self.motion_branch(s_motion))
        c_feat = F.relu(self.compliance_branch(s_compliance))
        av_feat = F.relu(self.action_v_xy(a_v_xy))
        ak_feat = F.relu(self.action_k_z(a_k_z))

        # 結構化融合
        combined = T.cat([m_feat, c_feat, av_feat, ak_feat], dim=1)
        return self.q(combined)