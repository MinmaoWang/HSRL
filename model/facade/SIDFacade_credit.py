import torch
import torch.nn.functional as F
import numpy as np
import utils
import pickle


class SIDFacade_credit():
    """
    Facade for SID-based policies.
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--slate_size', type=int, default=6,
                            help='slate size for actions')
        parser.add_argument('--buffer_size', type=int, default=10000,
                            help='replay buffer size')
        parser.add_argument('--start_timestamp', type=int, default=1000,
                            help='start timestamp for buffer sampling')
        parser.add_argument('--noise_var', type=float, default=0,
                            help='noise magnitude for action embedding sampling')
        parser.add_argument('--q_laplace_smoothness', type=float, default=0.5,
                            help='critic smoothness scalar for actors')
        parser.add_argument('--topk_rate', type=float, default=1.,
                            help='rate choosing topk rather than categorical sampling for items')
        parser.add_argument('--empty_start_rate', type=float, default=0,
                            help='probability of starting an episode from empty history')
        parser.add_argument('--item2sid', type=str, default='dataset/rl4rs/sid_index_item2sid.pkl',
                            help='item to semantic id mapping')
        return parser

    def __init__(self, args, environment, actor, critic):
        super().__init__()
        self.device = args.device
        self.env = environment
        self.actor = actor
        self.critic = critic

        self.slate_size = args.slate_size
        self.noise_var = args.noise_var
        self.noise_decay = args.noise_var / args.n_iter[-1]
        self.q_laplace_smoothness = args.q_laplace_smoothness
        self.topk_rate = args.topk_rate
        self.empty_start_rate = args.empty_start_rate

        print(f"Note!! item2sid path is {args.item2sid}")
        if args.item2sid is not None:
            with open(args.item2sid, "rb") as f:
                self.item2sid = pickle.load(f)
        else:
            self.item2sid = None

        self.n_item = self.env.action_space['item_id'][1]
        self.candidate_iids = np.arange(1, self.n_item + 1)
        self.candidate_features = torch.FloatTensor(
            self.env.reader.get_item_list_meta(self.candidate_iids)
        ).to(self.device)
        self.candidate_iids = torch.tensor(self.candidate_iids).to(self.device)

        self.buffer_size = args.buffer_size
        self.start_timestamp = args.start_timestamp

    def initialize_train(self):
        self.buffer = {
            "user_profile": torch.zeros(self.buffer_size, self.env.reader.portrait_len),
            "history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "next_history": torch.zeros(self.buffer_size, self.env.reader.max_seq_len).to(torch.long),
            "context_list": torch.zeros(
                self.buffer_size,
                self.actor.sid_levels + 1,
                self.actor.state_dim,
                dtype=torch.float32
            ),
            "action": torch.zeros(self.buffer_size, self.slate_size, dtype=torch.long),
            "reward": torch.zeros(self.buffer_size),
            "feedback": torch.zeros(self.buffer_size, self.slate_size),
            "done": torch.zeros(self.buffer_size, dtype=torch.bool),
            "sid_tokens": torch.zeros(
                self.buffer_size,
                self.slate_size,
                self.actor.sid_levels,
                dtype=torch.long
            )
        }
        for k, v in self.buffer.items():
            self.buffer[k] = v.to(self.device)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False

    def reset_env(self, initial_params={"batch_size": 1}):
        initial_params['empty_history'] = True if np.random.rand() < self.empty_start_rate else False
        return self.env.reset(initial_params)

    def env_step(self, policy_output):
        action_dict = {
            'action': policy_output['action'],
            'action_features': policy_output['action_features']
        }
        observation, reward, done, info = self.env.step(action_dict)
        return observation, reward, done, info

    def stop_env(self):
        self.env.stop()

    def get_episode_report(self, n_recent=10):
        recent_rewards = self.env.reward_history[-10:]
        recent_steps = self.env.step_history[-10:]
        return {
            'average_total_reward': np.mean(recent_rewards),
            'reward_variance': np.var(recent_rewards),
            'max_total_reward': np.max(recent_rewards),
            'min_total_reward': np.min(recent_rewards),
            'average_n_step': np.mean(recent_steps),
            'max_n_step': np.max(recent_steps),
            'min_n_step': np.min(recent_steps),
            'buffer_size': self.current_buffer_size
        }

    def apply_policy(self, observation, policy_model, epsilon=0.0,
                     do_explore=False, do_softmax=True):
        feed_dict = observation
        out_dict = policy_model(feed_dict)
        assert 'sid_logits' in out_dict, "SIDPolicy must output 'sid_logits'"
        sid_logits_list = out_dict['sid_logits']
        B = sid_logits_list[0].size(0)
        L = len(sid_logits_list)

        if 'candidate_ids' in feed_dict:
            cand_ids = feed_dict['candidate_ids']
            if isinstance(cand_ids, torch.Tensor) and cand_ids.dim() == 1:
                cand_ids = cand_ids.unsqueeze(0).repeat(B, 1)
        else:
            cand_ids = self.candidate_iids.unsqueeze(0).repeat(B, 1)

        sid_table = getattr(self, "_sid_table", None)
        if (sid_table is None) or (sid_table.size(1) != L):
            table = torch.zeros(self.n_item + 1, L, dtype=torch.long)
            for iid, sid in self.item2sid.items():
                tt = tuple(sid)
                if len(tt) >= L:
                    table[int(iid), :] = torch.tensor(tt[:L], dtype=torch.long)
            self._sid_table = table.to(self.device)
            sid_table = self._sid_table

        cand_sid = sid_table[cand_ids]
        level_probs = [torch.softmax(lg, dim=-1) for lg in sid_logits_list]
        candidate_prob = torch.ones_like(cand_ids, dtype=level_probs[0].dtype)

        for l in range(L):
            idx_l = cand_sid[..., l]
            pl = level_probs[l].gather(1, idx_l)
            candidate_prob = candidate_prob * pl

        candidate_prob = candidate_prob / (candidate_prob.sum(dim=1, keepdim=True) + 1e-12)

        if do_explore and epsilon > 0:
            candidate_prob = (1 - epsilon) * candidate_prob + epsilon * (1.0 / candidate_prob.size(1))

        if np.random.rand() >= self.topk_rate:
            action, indices = utils.sample_categorical_action(
                candidate_prob, cand_ids, self.slate_size,
                with_replacement=False, batch_wise=True, return_idx=True
            )
        else:
            _, indices = torch.topk(candidate_prob, k=self.slate_size, dim=1)
            action = torch.gather(cand_ids, 1, indices).detach()

        out_dict['action'] = action
        out_dict['action_features'] = self.candidate_features[action - 1]
        out_dict['action_prob'] = torch.gather(candidate_prob, 1, indices)
        out_dict['candidate_prob'] = candidate_prob
        out_dict['sid_tokens'] = sid_table[action]
        return out_dict

    def sample_buffer(self, batch_size):
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)
        U, H, N, CON, SID, A, R, F, D = self.read_buffer(indices)
        observation = {"user_profile": U, "history_features": H}
        policy_output = {"context_list": CON, "action": A, "sid_tokens": SID}
        reward = R
        done_mask = D
        next_observation = {"user_profile": U, "history_features": N, "previous_feedback": F}
        return observation, policy_output, reward, done_mask, next_observation

    def sample_raw_data(self, batch_size):
        batch = self.env.sample_user(batch_size, with_feedback=True)
        observation = {"user_profile": batch["user_profile"], "history_features": batch["history_features"]}
        exposure = {"ids": batch["exposed_items"], "features": batch["exposed_item_features"]}
        user_feedback = torch.FloatTensor(batch["feedback"])
        return observation, exposure, user_feedback

    def extract_behavior_data(self, observation, policy_output, next_observation):
        observation = {
            "user_profile": observation["user_profile"],
            "history_features": observation["history_features"]
        }
        exposed_items = policy_output["action"]
        exposure = {
            "ids": exposed_items,
            "features": self.candidate_features[exposed_items - 1]
        }
        user_feedback = next_observation["previous_feedback"]
        return observation, exposure, user_feedback

    def update_buffer(self, observation, policy_output, reward, done_mask, next_observation, info):
        if self.buffer_head + reward.shape[0] >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                      [i for i in range(reward.shape[0] - tail)]
        else:
            indices = [self.buffer_head + i for i in range(reward.shape[0])]

        self.buffer["user_profile"][indices] = observation['user_profile']
        self.buffer["history"][indices] = observation['history']
        self.buffer["next_history"][indices] = next_observation['history']
        self.buffer["context_list"][indices] = policy_output['context_list']
        self.buffer["action"][indices] = policy_output['action']
        self.buffer["sid_tokens"][indices] = policy_output['sid_tokens']
        self.buffer["reward"][indices] = reward
        self.buffer["feedback"][indices] = info['response']
        self.buffer["done"][indices] = done_mask

        self.buffer_head = (self.buffer_head + reward.shape[0]) % self.buffer_size
        self.n_stream_record += reward.shape[0]
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        if self.n_stream_record >= self.start_timestamp:
            self.is_training_available = True

    def read_buffer(self, indices):
        U = self.buffer["user_profile"][indices]
        H = self.candidate_features[self.buffer["history"][indices] - 1]
        N = self.candidate_features[self.buffer["next_history"][indices] - 1]
        CON = self.buffer["context_list"][indices]
        SID = self.buffer["sid_tokens"][indices]
        A = self.buffer["action"][indices]
        R = self.buffer["reward"][indices]
        F = self.buffer["feedback"][indices]
        D = self.buffer["done"][indices]
        return U, H, N, CON, SID, A, R, F, D
