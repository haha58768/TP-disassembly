import os
import numpy as np
import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import NStepReplayBuffer
from collections import deque


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=256, fc2_dims=256,
                 batch_size=256, n_step=1):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.n_step = n_step

        self.memory = NStepReplayBuffer(max_size, input_dims, n_actions, n_step, gamma)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)

        # Forward pass through structured Multi-branch Network
        mu = self.actor.forward(state).to(self.actor.device)

        noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        mu_prime = mu + noise

        self.actor.train()
        return T.clamp(mu_prime, -1.0, 1.0).cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones).to(self.actor.device)

        # Target values with n-step return
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[dones] = 0.0
        critic_value_ = critic_value_.view(-1)
        target = rewards + (self.gamma ** self.n_step) * critic_value_
        target = target.view(self.batch_size, 1)

        # Critic Update
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # Actor Update (Policy Gradient)
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for name in critic_params:
            critic_params[name] = tau * critic_params[name].clone() + \
                                  (1 - tau) * target_critic_state_dict[
                                      name].clone() if 'target_critic_state_dict' in locals() else \
                tau * critic_params[name].clone() + (1 - tau) * target_critic_params[name].clone()

        for name in actor_params:
            actor_params[name] = tau * actor_params[name].clone() + \
                                 (1 - tau) * target_actor_params[name].clone()

        self.target_critic.load_state_dict(critic_params)
        self.target_actor.load_state_dict(actor_params)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()