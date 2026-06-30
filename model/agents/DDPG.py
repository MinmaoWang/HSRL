import copy
import torch
import torch.nn.functional as F

from model.agents.BaseRLAgent import BaseRLAgent


class DDPG(BaseRLAgent):
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
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size

        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_decay
        )

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay
        )

        self.tau = args.target_mitigate_coef
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as f:
                f.write(f"{args}\n")

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
        critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "step_loss": (
                self.training_history['actor_loss'][-1],
                self.training_history['critic_loss'][-1],
            )
        }

    def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation,
                      do_actor_update=True, do_critic_update=True):
        with torch.no_grad():
            next_policy_output = self.facade.apply_policy(
                next_observation, self.actor_target, epsilon=0.0, do_explore=False
            )
            target_q = self.facade.apply_critic(
                next_observation, next_policy_output, self.critic_target
            )['q'].view(-1)
            not_done = (~done_mask.bool()).to(target_q.dtype)
            target_q = reward + self.gamma * (not_done * target_q)

        current_q = self.facade.apply_critic(observation, policy_output, self.critic)['q'].view(-1)
        critic_loss = F.mse_loss(current_q, target_q.detach())

        if do_critic_update and self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        current_policy_output = self.facade.apply_policy(
            observation, self.actor, epsilon=0.0, do_explore=False
        )
        actor_q = self.facade.apply_critic(observation, current_policy_output, self.critic)['q'].view(-1)
        actor_loss = -actor_q.mean()

        if do_actor_update and self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return critic_loss.detach(), actor_loss.detach()

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
