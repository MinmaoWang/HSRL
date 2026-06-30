# model/policy/SIDPolicy.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn

from utils import get_regularization


class SIDPolicy_credit(nn.Module):
    """
    Hierarchical residual SID policy.
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--sasrec_n_layer', type=int, default=2)
        parser.add_argument('--sasrec_d_model', type=int, default=64)
        parser.add_argument('--sasrec_d_forward', type=int, default=128)
        parser.add_argument('--sasrec_n_head', type=int, default=4)
        parser.add_argument('--sasrec_dropout', type=float, default=0.1)
        parser.add_argument('--sid_levels', type=int, default=3)
        parser.add_argument('--sid_vocab_sizes', type=int, default=256,
                            help='每一层的vocab大小，将自动扩展为 [v, v, v]')
        parser.add_argument('--sid_temp', type=float, default=1.0,
                            help='softmax 温度，>1 更平缓，<1 更尖锐')
        return parser

    def __init__(self, args, environment):
        super().__init__()
        self.n_layer = args.sasrec_n_layer
        self.d_model = args.sasrec_d_model
        self.n_head = args.sasrec_n_head
        self.dropout = args.sasrec_dropout
        self.sid_temp = float(getattr(args, 'sid_temp', 1.0))

        self.n_item = environment.action_space['item_id'][1]
        self.item_dim = environment.action_space['item_feature'][1]
        self.maxlen = environment.observation_space['history'][1]

        self.state_dim = self.d_model
        self.action_dim = self.d_model

        self.item_map = nn.Linear(self.item_dim, self.d_model)
        self.pos_emb = nn.Embedding(self.maxlen, self.d_model)
        self.emb_drop = nn.Dropout(self.dropout)
        self.emb_norm = nn.LayerNorm(self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=args.sasrec_d_forward,
            nhead=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layer)

        self.register_buffer('pos_idx', torch.arange(self.maxlen, dtype=torch.long), persistent=False)
        full = torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.register_buffer('attn_mask_full', ~full, persistent=False)

        self.sid_levels = int(args.sid_levels)
        self.sid_vocab_sizes = [args.sid_vocab_sizes] * self.sid_levels

        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, v) for v in self.sid_vocab_sizes])
        self.sid_token_embeds = nn.ModuleList([
            nn.Embedding(v, self.d_model) for v in self.sid_vocab_sizes
        ])
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

    def encode_state(self, feed_dict):
        hist = feed_dict['history_features']
        B, H, _ = hist.shape

        pos = self.pos_emb(self.pos_idx[:H])
        pos = pos.unsqueeze(0).expand(B, H, -1)

        x = self.item_map(hist)
        x = self.emb_norm(self.emb_drop(x + pos))

        attn_mask = self.attn_mask_full[:H, :H]
        out_seq = self.transformer(x, mask=attn_mask)
        state = out_seq[:, -1, :]
        return {'output_seq': out_seq, 'state_emb': state}

    def forward(self, feed_dict):
        enc = self.encode_state(feed_dict)
        context = enc['state_emb']

        sid_logits = []
        context_list = [context]
        tau = self.sid_temp if self.sid_temp is not None else 1.0

        for l in range(self.sid_levels):
            logits_l = self.sid_heads[l](context)
            sid_logits.append(logits_l)

            probs_l = torch.softmax(logits_l / tau, dim=-1)
            emb_tbl = self.sid_token_embeds[l].weight
            exp_emb = probs_l @ emb_tbl

            context = self.sid_res_norms[l](context - exp_emb)
            context_list.append(context)

        context_tensor = torch.stack(context_list, dim=1)
        reg = get_regularization(self.item_map, self.transformer)
        return {
            'sid_logits': sid_logits,
            'context_list': context_tensor,
            'seq_emb': enc['output_seq'],
            'reg': reg
        }
