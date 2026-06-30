import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization


class Token_Critic(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128],
                            help='Specify a list of hidden layer sizes for the critic MLP')
        parser.add_argument('--critic_dropout_rate', type=float, default=0.2,
                            help='Dropout rate in deep layers')
        return parser

    def __init__(self, args, environment, policy):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net = DNN(
            self.state_dim,
            args.critic_hidden_dims,
            1,
            dropout_rate=args.critic_dropout_rate,
            do_batch_norm=True
        )

    def forward(self, feed_dict):
        reg = get_regularization(self.net)
        context_tensor = feed_dict['context_list']
        B, L_plus_1, d_model = context_tensor.shape
        context_flat = context_tensor.view(B * L_plus_1, d_model)
        v_flat = self.net(context_flat).view(B, L_plus_1)
        return {
            'v_seq': v_flat,
            'reg': reg
        }
