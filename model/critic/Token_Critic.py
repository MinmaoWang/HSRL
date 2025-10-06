import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization


class Token_Critic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_hidden_dims
        - critic_dropout_rate
        '''
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128], 
                            help='Specify a list of hidden layer sizes for the critic MLP')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2, 
                            help='Dropout rate in deep layers')
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim

        # 核心 MLP，用于 state -> value
        self.net = DNN(
            self.state_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True
        )

    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict 可以有两种模式:
            1) {'state_emb': (B, state_dim)} -> 旧逻辑，输出 slate-level value
            2) {'context_list': [context_0, context_1, ..., context_L]} -> 新增逻辑，输出 token-level value
        '''
        reg = get_regularization(self.net)

        # === 新模式：token-level value ===
        # context_list = feed_dict['context_list']      # list of (B, d)
        # vs = []
        # for context in context_list:
        #     v = self.net(context).view(-1)            # 每一步输出 (B,)
        #     vs.append(v)
        # # 堆叠成 (B, L+1)
        # v_seq = torch.stack(vs, dim=1)
        # return {
        #     'v_seq': v_seq,                           # (B, L+1)
        #     'reg': reg
        # }
        context_tensor = feed_dict['context_list']  # (B, L+1, d_model)
        B, L_plus_1, d_model = context_tensor.shape

        # reshape 成 (B*(L+1), d_model)
        context_flat = context_tensor.view(B * L_plus_1, d_model)

        # 喂给 MLP
        v_flat = self.net(context_flat).view(B, L_plus_1)  # 输出 (B, L+1)

        return {
            'v_seq': v_flat,   # (B, L+1)
            'reg': reg
        }
