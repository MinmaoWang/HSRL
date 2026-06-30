# model/agents/A2C_SID.py
import copy
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model.agents.BaseRLAgent import BaseRLAgent


class A2C_SID_credit_train(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8,
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4,
                            help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4,
                            help='learning rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4,
                            help='weight decay for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4,
                            help='weight decay for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01,
                            help='target-network soft update coefficient')
        parser.add_argument('--advantage_bias', type=float, default=0,
                            help='advantage bias')
        parser.add_argument('--entropy_coef', type=float, default=0.1,
                            help='entropy coefficient')
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size

        self.actor = facade.actor
        self.critic = facade.critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay
        )

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        self.token_weight = nn.Parameter(torch.ones(3 + 1, device=self.device) / (3 + 1))

        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as f:
                f.write(f"{args}\n")

    def action_before_train(self):
        super().action_before_train()
        self.training_history['entropy_loss'] = []
        self.training_history['advantage'] = []

    @torch.no_grad()
    def run_episode_step(self, *episode_args):
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=True)
        next_observation, reward, done, info = self.facade.env_step(policy_output)
        if do_buffer_update:
            self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        critic_loss, actor_loss, entropy_loss, advantage = self.get_a2c_loss(
            observation, policy_output, reward, done_mask, next_observation
        )
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "step_loss": (
                self.training_history['actor_loss'][-1],
                self.training_history['critic_loss'][-1],
                self.training_history['entropy_loss'][-1],
                self.training_history['advantage'][-1],
            )
        }

    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation,
                     do_actor_update=True, do_critic_update=True):
        with torch.no_grad():
            next_po = self.facade.apply_policy(
                next_observation, self.actor_target,
                epsilon=0.0, do_explore=False
            )
            V_sp_out = self.critic_target({'context_list': next_po['context_list']})
            V_sp_seq = V_sp_out['v_seq']
            B = V_sp_seq.shape[0]
            token_weight = torch.softmax(self.token_weight, dim=0).unsqueeze(0).expand(B, -1)
            V_sp_weighted = (V_sp_seq * token_weight).sum(dim=1)
            not_done = (~done_mask.bool()).to(V_sp_weighted.dtype)
            Q_s = reward + self.gamma * (not_done * V_sp_weighted)

        cur_po = self.facade.apply_policy(
            observation, self.actor,
            epsilon=0.0, do_explore=False
        )

        V_s_out = self.critic({'context_list': cur_po['context_list']})
        V_s_seq = V_s_out['v_seq']
        V_s_weighted = (V_s_seq * token_weight).sum(dim=1)
        value_loss = F.mse_loss(V_s_weighted, Q_s)

        if do_critic_update and self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        sid_tokens = policy_output['sid_tokens']
        sid_logits_list = cur_po['sid_logits']
        B, K, L = sid_tokens.shape

        sid_tokens_flat = sid_tokens.view(B * K, L)
        level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
        level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]

        nll_slot = 0.0
        entropy_raw = 0.0
        for l in range(L):
            probs_l_flat = level_probs_flat[l]
            z_l = sid_tokens_flat[:, l].view(-1, 1)
            logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
            nll_slot = nll_slot + (-logp_l)
            entropy_raw = entropy_raw + \
                (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()

        nll_per_sample = nll_slot.view(B, K).mean(dim=1)

        with torch.no_grad():
            advantage = torch.clamp(Q_s - V_s_weighted, -1, 1).view(-1)

        actor_loss = (nll_per_sample * (advantage + self.advantage_bias)).mean()
        total_actor = actor_loss + self.entropy_coef * entropy_raw

        if do_actor_update and self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
            total_actor.backward()
            self.actor_optimizer.step()

        return value_loss.detach(), actor_loss.detach(), entropy_raw.detach(), advantage.mean().detach()

    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")

    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)


class A2C_SID_rl4rs(A2C_SID_credit_train):
    pass
