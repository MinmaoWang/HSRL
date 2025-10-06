# model/policy/SIDPolicy.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn

from utils import get_regularization

class SIDPolicy_credit(nn.Module):
    """
    层级残差版 SID：
    forward 返回
      - 'sid_logits': [ (B,V1), (B,V2), ... ]   # 逐层条件（残差）得到
      - 'state_emb' : (B, d_model)
      - 'seq_emb'   : (B, H, d_model)
      - 'reg'       : scalar
    残差流程（可微）：
      context_0 = state_emb
      for l in 1..L:
        logits_l = head_l(context_{l-1})                 # (B, V_l)
        probs_l  = softmax(logits_l / tau)               # (B, V_l)
        e_l      = probs_l @ token_emb_l.weight          # (B, d_model)
        context_l = LayerNorm(context_{l-1} - e_l)       # 残差，供下一层使用
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--sasrec_n_layer', type=int, default=2)
        parser.add_argument('--sasrec_d_model', type=int, default=64)
        parser.add_argument('--sasrec_d_forward', type=int, default=128)
        parser.add_argument('--sasrec_n_head', type=int, default=4)
        parser.add_argument('--sasrec_dropout', type=float, default=0.1)
        parser.add_argument('--sid_levels', type=int, default=3)
            # 修改这里：直接输入一个整数，比如 64
        parser.add_argument('--sid_vocab_sizes', type=int, default=256,
                            help='每一层的vocab大小，将自动扩展为 [v, v, v]')
        parser.add_argument('--sid_temp', type=float, default=1.0, help='softmax 温度，>1 更平缓，<1 更尖锐')
        return parser

    def __init__(self, args, environment):
        super().__init__()
        # === 基本超参 ===
        self.n_layer   = args.sasrec_n_layer
        self.d_model   = args.sasrec_d_model
        self.n_head    = args.sasrec_n_head
        self.dropout   = args.sasrec_dropout
        self.sid_temp  = float(getattr(args, 'sid_temp', 1.0))

        # === 空间信息 ===
        self.n_item    = environment.action_space['item_id'][1]
        self.item_dim  = environment.action_space['item_feature'][1]
        self.maxlen    = environment.observation_space['history'][1]

        # 兼容旧字段
        self.state_dim  = self.d_model
        self.action_dim = self.d_model

        # === 编码器（SASRec 风格）===
        self.item_map  = nn.Linear(self.item_dim, self.d_model)
        self.pos_emb   = nn.Embedding(self.maxlen, self.d_model)
        self.emb_drop  = nn.Dropout(self.dropout)
        self.emb_norm  = nn.LayerNorm(self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            dim_feedforward=args.sasrec_d_forward,
            nhead=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.n_layer)

        # 注册 buffer：位置索引 + 全局下三角 mask
        self.register_buffer('pos_idx', torch.arange(self.maxlen, dtype=torch.long), persistent=False)
        full = torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.register_buffer('attn_mask_full', ~full, persistent=False)

        # === SID 层级 ===
        self.sid_levels = int(args.sid_levels)
        

        self.sid_vocab_sizes = [args.sid_vocab_sizes] * self.sid_levels

        # 每层分类头：context -> logits_l
        self.sid_heads = nn.ModuleList([nn.Linear(self.d_model, v) for v in self.sid_vocab_sizes])

        # 每层 codebook（token embedding）
        self.sid_token_embeds = nn.ModuleList([
            nn.Embedding(v, self.d_model) for v in self.sid_vocab_sizes
        ])

        # 每层一个 LayerNorm，用于 residual 稳定
        self.sid_res_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.sid_levels)])

    def encode_state(self, feed_dict):
        """
        入: feed_dict['history_features'] -> (B, H, item_dim)
        出: {'output_seq': (B, H, d_model), 'state_emb': (B, d_model)}
        """
        hist = feed_dict['history_features']               # (B, H, item_dim)
        B, H, _ = hist.shape

        pos = self.pos_emb(self.pos_idx[:H])               # (H, d_model) → (B, H, d_model)
        pos = pos.unsqueeze(0).expand(B, H, -1)

        x   = self.item_map(hist)                          # (B, H, d_model)
        x   = self.emb_norm(self.emb_drop(x + pos))

        attn_mask = self.attn_mask_full[:H, :H]            # (H, H)
        out_seq   = self.transformer(x, mask=attn_mask)    # (B, H, d_model)
        state     = out_seq[:, -1, :]                      # (B, d_model)
        return {'output_seq': out_seq, 'state_emb': state}


    def forward(self, feed_dict):
        enc = self.encode_state(feed_dict)
        context = enc['state_emb']  # (B, d_model)

        sid_logits = []
        context_list = [context]  # 保存初始 context

        tau = self.sid_temp if self.sid_temp is not None else 1.0

        for l in range(self.sid_levels):
            logits_l = self.sid_heads[l](context)  # (B, V_l)
            sid_logits.append(logits_l)

            probs_l = torch.softmax(logits_l / tau, dim=-1)

            emb_tbl = self.sid_token_embeds[l].weight  # (V_l, d_model)
            exp_emb = probs_l @ emb_tbl  # (B, d_model)

            # 残差更新
            context = self.sid_res_norms[l](context - exp_emb)

            # 保存每一步的 context
            context_list.append(context)

        # === 新增 ===
        # 变成 tensor (B, L+1, d_model)，方便直接存到 buffer
        context_tensor = torch.stack(context_list, dim=1)

        reg = get_regularization(self.item_map, self.transformer)
        return {
            'sid_logits': sid_logits,           # list[(B, V_l)]
            'context_list': context_tensor,     # tensor (B, L+1, d_model)
            # 'state_emb' : enc['state_emb'],
            'seq_emb': enc['output_seq'],
            'reg': reg
        }


