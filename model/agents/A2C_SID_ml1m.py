# model/agents/A2C_SID.py
import copy
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model.agents.BaseRLAgent import BaseRLAgent

class A2C_SID_credit_train_pub(BaseRLAgent):
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
                            help='decay rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        parser.add_argument('--token_lr', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                    help='behaviorvise loss coefficient') 
        
        return parser


    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size

        self.actor = facade.actor
        self.critic = facade.critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=args.actor_lr,  weight_decay=args.actor_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef

        # self.token_weight = nn.Parameter(torch.ones(3+1) / (3+1))  # 初始化为均匀
        self.token_weight = nn.Parameter(torch.ones(3+1, device=self.device) / (3+1))
        self.token_optimizer = torch.optim.Adam([self.token_weight], lr=args.token_lr)
        
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as f:
                f.write(f"{args}\n")
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr, 
                                                         weight_decay=args.behavior_decay)

    def action_before_train(self):
        super().action_before_train()
        self.training_history['entropy_loss'] = []
        self.training_history['advantage'] = []
        self.training_history["behavior_loss"] = []

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
        behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # soft update targets
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
                self.training_history['behavior_loss'][-1]
            )
        }
            
  

    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation,
                    do_actor_update=True, do_critic_update=True):
        """
        - Critic：token-level 价值计算，通过 context_list 输出 V_seq
        - Actor：对整张 slate 的语义 token 做 NLL（(B,K,L)），并加熵正则
        """
        # ---- Critic Target ----
        with torch.no_grad():
            # 获取下一状态的 token-level context
            next_po = self.facade.apply_policy(
                next_observation, self.actor_target,
                epsilon=0.0, do_explore=False
            )

            # ★ 修改点 1: 使用 context_list 传入 Token Critic
            V_sp_out = self.critic_target({'context_list': next_po['context_list']})
            V_sp_seq = V_sp_out['v_seq']  # (B, L+1)
            B = V_sp_seq.shape[0]

            # # ★ 修改点 2: token_weight 扩展为 (B, L+1)，适配 batch
            # token_weight = torch.tensor(
            #     self.token_weight, device=V_sp_seq.device, dtype=V_sp_seq.dtype
            # ).unsqueeze(0).expand(B, -1)  # (B, L+1)
            
            token_weight = torch.softmax(self.token_weight, dim=0).unsqueeze(0).expand(B, -1)

            # ★ 修改点 3: TD 目标同样使用 token-level 加权 value
            V_sp_weighted = (V_sp_seq * token_weight).sum(dim=1)  # (B,)
            Q_s = reward + self.gamma * (done_mask * V_sp_weighted)

        # ---- 当前状态 Critic ----
        cur_po = self.facade.apply_policy(
            observation, self.actor,
            epsilon=0.0, do_explore=False
        )

        V_s_out = self.critic({'context_list': cur_po['context_list']})
        V_s_seq = V_s_out['v_seq']  # (B, L+1)

        # ★ 修改点 4: 当前状态的 token-level 加权
        V_s_weighted = (V_s_seq * token_weight).sum(dim=1)  # (B,)

        # Critic Loss
        value_loss = F.mse_loss(V_s_weighted, Q_s)

        if do_critic_update and self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        # ---- Actor 部分 ----
        sid_tokens = policy_output['sid_tokens']  # (B, K, L)
        sid_logits_list = cur_po['sid_logits']    # list[(B, V_l)]
        B, K, L = sid_tokens.shape

        # flatten (B,K,*) -> (B*K,*)
        sid_tokens_flat = sid_tokens.view(B * K, L)  # (B*K, L)
        level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
        level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]

        # === NLL 计算 ===
        nll_slot = 0.0
        entropy_raw = 0.0
        for l in range(L):
            probs_l_flat = level_probs_flat[l]  # (B*K, V_l)
            z_l = sid_tokens_flat[:, l].view(-1, 1)  # (B*K, 1)
            logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
            nll_slot = nll_slot + (-logp_l)

            # 熵项
            entropy_raw = entropy_raw + \
                (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()

        # (B*K,) -> (B,K) -> (B,)
        nll_per_sample = nll_slot.view(B, K).mean(dim=1)

        # ---- Advantage ----
        # ★ 修改点 5: 使用 token-level 加权的 V_s 计算 Advantage
        with torch.no_grad():
            advantage = torch.clamp(Q_s - V_s_weighted, -1, 1).view(-1)  # (B,)

        # Actor Loss
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

    def get_behavior_loss(self, observation, policy_output, next_observation, do_update = True):
        observation, exposure, feedback = self.facade.extract_behavior_data(observation, policy_output, next_observation)
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
        
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss
    
    def test(self):
        """
        评测（严格复用训练时的交互与统计）：
        - 加载权重
        - epsilon 按 exploration_scheduler 路径（含 schedule_timesteps==0 兜底）
        - 复用 run_episode_step（最后一个位置参数传 True，从而写入 buffer），但不做 step_train()
        - 定期调用 self.log_iteration(i)，得到与训练完全一致的 episode report
        """
        import torch
        from time import time
        from tqdm import tqdm

        # 兜底：有些 Facade 在纯测试没初始化 buffer 相关字段
        if not hasattr(self.facade, "current_buffer_size"):
            self.facade.current_buffer_size = 0
        if not hasattr(self.facade, "update_buffer"):
            def _no_buffer(*args, **kwargs): return None
            self.facade.update_buffer = _no_buffer

        # 加载权重
        try:
            self.load()
            print("[TEST] Loaded trained weights from save_path prefix.")
        except Exception as e:
            print(f"[TEST] WARN: load() failed or weights not found, continue. Detail: {e}")

        # eval + 关梯度
        self.actor.eval()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.eval()
        torch.set_grad_enabled(False)

        # 训练前置动作（若 facade 用得到）
        try:
            self.action_before_train()
        except Exception as e:
            print(f"[TEST] WARN: action_before_train() failed (safe to ignore): {e}")

        # 环境 reset（与训练同样的 batch）
        print("[TEST] Reset environment for evaluation.")
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})

        # epsilon 取值函数：对 schedule_timesteps==0 做兜底
        def get_eps(t):
            sch = getattr(self, "exploration_scheduler", None)
            if sch is None:
                return 0.0
            st = getattr(sch, "schedule_timesteps", None)
            if st is None or st == 0:
                return getattr(sch, "final_p", 0.0)
            try:
                return sch.value(t)
            except ZeroDivisionError:
                return getattr(sch, "final_p", 0.0)

        total_steps = self.n_iter[-1] if isinstance(self.n_iter, (list, tuple)) else int(self.n_iter)
        start_time = time(); last_t = start_time
        print("[TEST] Start rollout (mirror train: scheduler epsilon, buffer on, no optimization).")
        for i in tqdm(range(total_steps)):
            eps = get_eps(i)
            # ★ 用位置参数，最后一个 True 表示“写入 buffer”，与训练保持一致
            observation = self.run_episode_step(i, eps, observation, True)

            if i % self.check_episode == 0:
                now = time()
                print(f"[TEST] step {i} | dt={now - last_t:.2f}s | total={now - start_time:.2f}s")
                # 与训练相同的输出： step: i @ episode report: {...} @ step loss: {...}
                print(self.log_iteration(i))
                last_t = now

        if hasattr(self.facade, "stop_env"):
            try: self.facade.stop_env()
            except: pass
        torch.set_grad_enabled(True)
        print("[TEST] Evaluation finished.")
