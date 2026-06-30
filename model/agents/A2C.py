import copy
import torch
import torch.nn.functional as F

from model.agents.BaseRLAgent import BaseRLAgent


class A2C(BaseRLAgent):
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

        return {"step_loss": (
            self.training_history['actor_loss'][-1],
            self.training_history['critic_loss'][-1],
            self.training_history['entropy_loss'][-1],
            self.training_history['advantage'][-1],
        )}

    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation,
                     do_actor_update=True, do_critic_update=True):
        with torch.no_grad():
            next_po = self.facade.apply_policy(next_observation, self.actor_target, epsilon=0.0, do_explore=False)
            V_sp = self.facade.apply_critic(next_observation, next_po, self.critic_target)['q'].view(-1)
            not_done = (~done_mask.bool()).to(V_sp.dtype)
            Q_s = reward + self.gamma * (not_done * V_sp)

        cur_value = self.facade.apply_critic(observation, policy_output, self.critic)['q'].view(-1)
        critic_loss = F.mse_loss(cur_value, Q_s.detach())

        if do_critic_update and self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        cur_po = self.facade.apply_policy(observation, self.actor, epsilon=0.0, do_explore=False)
        action_prob = cur_po.get('action_prob')
        if action_prob is None:
            actor_loss = -self.facade.apply_critic(observation, cur_po, self.critic)['q'].view(-1).mean()
            entropy_loss = torch.zeros((), device=self.device)
        else:
            log_prob = torch.log(action_prob + 1e-12).mean(dim=1)
            with torch.no_grad():
                advantage = torch.clamp(Q_s - cur_value, -1, 1)
            actor_loss = -(log_prob * (advantage + self.advantage_bias)).mean()
            entropy_loss = (action_prob * torch.log(action_prob + 1e-12)).sum(dim=1).mean()
        total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

        if do_actor_update and self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()

        with torch.no_grad():
            advantage = torch.clamp(Q_s - cur_value, -1, 1)
        return critic_loss.detach(), actor_loss.detach(), entropy_loss.detach(), advantage.mean().detach()

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
