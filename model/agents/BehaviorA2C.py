import torch
import torch.nn.functional as F

from model.agents.A2C import A2C


class BehaviorA2C(A2C):
    @staticmethod
    def parse_model_args(parser):
        parser = A2C.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001,
                            help='behavior loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003,
                            help='behavior optimizer decay')
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.actor_behavior_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.behavior_lr, weight_decay=args.behavior_decay
        )

    def action_before_train(self):
        super().action_before_train()
        self.training_history['behavior_loss'] = []

    def get_behavior_loss(self, observation, policy_output, next_observation, do_update=True):
        observation, exposure, feedback = self.facade.extract_behavior_data(
            observation, policy_output, next_observation
        )
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax=False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss

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
            self.training_history['behavior_loss'][-1],
        )}
