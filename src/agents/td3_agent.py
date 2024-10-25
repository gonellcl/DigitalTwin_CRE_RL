import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# Define Actor Network for variable prediction
class Actor(nn.Module):
    def __init__(self, state_dim, max_rent, max_lease_length):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)  # Predict rent amount, lease length, and vacancy rate
        self.max_rent = max_rent
        self.max_lease_length = max_lease_length

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Scale outputs to ensure they are within valid ranges
        rent_amount = torch.relu(x[:, 0]) * self.max_rent  # Ensure non-negative rent, scale to max_rent
        lease_length = torch.relu(x[:, 1]) * self.max_lease_length  # Scale to max_lease_length
        vacancy_rate = torch.sigmoid(x[:, 2])  # Sigmoid for values between 0 and 1

        return torch.stack([rent_amount, lease_length, vacancy_rate], dim=1)


# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output the Q-value


class TD3Agent:
    def __init__(self, state_dim, max_rent, max_lease_length, memory_size=10000, batch_size=32, gamma=0.99, tau=0.005,
                 lr=0.001, policy_noise=0.2, noise_clip=0.5, policy_delay=2, epsilon=0.7):
        self.state_dim = state_dim
        self.action_dim = 3  # Predicting 3 variables: rent, lease length, vacancy rate
        self.max_rent = max_rent
        self.max_lease_length = max_lease_length
        self.memory = deque(maxlen=memory_size)  # Replay buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploratory_actions = 0  # Track exploratory actions
        self.total_actions = 0  # Track total actions
        self.epsilon = epsilon

        # Initialize actor and critic networks
        self.actor = Actor(state_dim, max_rent, max_lease_length)
        self.actor_target = Actor(state_dim, max_rent, max_lease_length)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, self.action_dim)
        self.critic_1_target = Critic(state_dim, self.action_dim)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_dim, self.action_dim)
        self.critic_2_target = Critic(state_dim, self.action_dim)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        # Sync target networks with the main networks
        self.update_target_networks(soft_update=False)

        self.total_it = 0  # For delayed policy updates

    def act(self, state):
        """
        Choose an action based on epsilon-greedy strategy and actor model prediction.
        """
        self.total_actions += 1

        # Exploration: with probability epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            self.exploratory_actions += 1
            return self.random_action()

        # Exploitation: with probability (1 - epsilon), use the actor model to predict action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy()

        # Split the action into components: rent_amount, lease_length, vacancy_rate
        rent_amount, lease_length, vacancy_rate = action

        # Ensure lease length is an integer and within a valid range
        lease_length = max(1, round(lease_length))

        return rent_amount, lease_length, vacancy_rate

    def random_action(self):
        """
        Generate a random action for exploration.
        """
        rent_amount = np.random.uniform(0, self.max_rent)  # Random rent amount
        lease_length = np.random.randint(1, self.max_lease_length + 1)  # Random lease length
        vacancy_rate = np.random.uniform(0, 1)  # Random vacancy rate
        return rent_amount, lease_length, vacancy_rate

    def get_exploration_ratio(self):
        """
        Calculate the ratio of exploratory actions to total actions.
        """
        if self.total_actions > 0:
            return self.exploratory_actions / self.total_actions
        return 0.0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def compute_reward(self, rent_amount, lease_length, vacancy_rate, done):
        """
        Compute the reward for the agent's action based on the environment state and action taken.

        Args:
        - rent_amount (float): The rent amount proposed by the agent.
        - lease_length (int): The lease length proposed by the agent.
        - vacancy_rate (float): The current vacancy rate in the environment.
        - done (bool): Whether the episode is done.

        Returns:
        - reward (float): The computed reward for the agent's action.
        """
        reward = 0

        # Condition 1: High vacancy rate and long lease length
        if vacancy_rate > 0.5 and lease_length >= 10:
            reward = 100  # Significant reward for reducing vacancy with a long lease

        # Condition 2: Rent amount too high relative to vacancy rate
        elif vacancy_rate > 0.5 and rent_amount > (self.max_rent * 0.7):  # Example threshold
            reward = 0  # No reward for high rent when vacancy is high

        # Condition 3: Suboptimal combination of vacancy rate, lease length, and rent amount
        elif vacancy_rate > 0.3 and lease_length < 5 and rent_amount > (self.max_rent * 0.5):
            reward = 25  # Moderate reward for suboptimal but not entirely negative behavior

        # Existing penalties/rewards
        # Penalty for high vacancy rates
        if vacancy_rate > 0.3:
            reward -= 10 * (vacancy_rate - 0.3)  # Higher penalty for higher vacancy

        # Reward for maximizing rent amount (incentivize higher rent)
        reward += 5 * rent_amount

        # Penalty for excessive lease lengths (e.g., beyond 10 years)
        if lease_length > 10:
            reward -= 5 * (lease_length - 10)  # Penalize longer leases

        # Penalty for tenants consistently renewing without breaks
        if done and np.random.rand() > 0.7:  # 70% chance to terminate the lease
            reward -= 20  # Encourage tenant vacating

        return reward

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Add noise to the next action (target policy smoothing)
        noise = torch.clamp(torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip)
        next_actions = self.actor_target(next_states) + noise
        next_actions[:, 0] = torch.clamp(next_actions[:, 0], 0, self.max_rent)  # Rent amount
        next_actions[:, 1] = torch.clamp(next_actions[:, 1], 1, self.max_lease_length)  # Lease length
        next_actions[:, 2] = torch.clamp(next_actions[:, 2], 0, 1)  # Vacancy rate

        # Compute target Q-values
        target_q1 = self.critic_1_target(next_states, next_actions)
        target_q2 = self.critic_2_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        # Optimize Critic 1
        current_q1 = self.critic_1(states, actions)
        loss_critic_1 = nn.MSELoss()(current_q1, target_q.detach())
        self.critic_1_optimizer.zero_grad()
        loss_critic_1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2
        current_q2 = self.critic_2(states, actions)
        loss_critic_2 = nn.MSELoss()(current_q2, target_q.detach())
        self.critic_2_optimizer.zero_grad()
        loss_critic_2.backward()
        self.critic_2_optimizer.step()

        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update of target networks
            self.update_target_networks(soft_update=True)

        self.total_it += 1

    def update_target_networks(self, soft_update=True):
        """Soft or hard update of the target networks."""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data) if soft_update else target_param.data.copy_(
                param.data)

        for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data) if soft_update else target_param.data.copy_(
                param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data) if soft_update else target_param.data.copy_(
                param.data)

    def save(self, filename):
        """Save the model weights."""
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic_2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        """Load the model weights."""
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic1.pth"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic2.pth"))
        self.update_target_networks(soft_update=False)
