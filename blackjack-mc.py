"""
Balckjack Monte Carlo version

In order to play with Monte Carlo (MC) method using this code,
prepare Python environment (assuming python3-venv) and run:

```
python3 -m venv venv

. venv/bin/activate

pip install gymnasium

python blackjack-mc.py
```
"""

from collections import defaultdict, deque
from pprint import pprint

import numpy as np
import gymnasium as gym

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from utils import plot_blackjack_values, plot_policy


# Create the Blackjack environment
env = gym.make('Blackjack-v1', render_mode=None)


# Monte Carlo Control with Exploring Starts
class MonteCarloControl:
    def __init__(self, env, gamma=1.0, epsilon=0.1, limit=None):
        self.env = env
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy strategy factor
        self.Q = defaultdict(
            lambda: np.zeros(env.action_space.n)
        )  # state-action value
        self.returns = defaultdict(
            lambda: deque([], maxlen=limit)
        )  # tracks returns for each state-action pair
        self.policy = defaultdict(
            lambda: np.random.choice(self.env.action_space.n)
        )  # exploring starts

    def generate_episode(self):
        """Generate an episode by following the stored policy."""
        episode = []
        state = self.env.reset()[0]
        while True:
            # Use epsilon-greedy policy to explore
            action = int(
                self.policy[state]
                # if state in self.policy
                if np.random.random() >= self.epsilon
                else np.random.choice(self.env.action_space.n)
            )
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def update_policy(self):
        """Update the policy to be greedy with respect to Q."""
        for state in self.Q:
            # Select the action with the maximum Q-value for the state
            best_action = int(np.argmax(self.Q[state]))
            self.policy[state] = best_action

    def train(self, episodes=50000):
        """Train the policy using Monte Carlo control."""
        for i in range(episodes):
            episode = self.generate_episode()  # policy evaluation
            G = 0  # initialize return
            visited = set()  # keep track of visited state-action pairs

            # Process episode in reverse for calculating returns
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    # Implement statistics here
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(
                        self.returns[(state, action)]
                    )
            self.update_policy()  # policy improvement

            # Optional: print progress
            if (i + 1) % 1000 == 0:
                print(f"Episode {i + 1}/{episodes} completed.")

        print("Training completed!")


# Monte Carlo Control with Exploring Starts (incremental implementation)
class MonteCarloControlIncremental:
    def __init__(self, env, gamma=1.0, epsilon=0.1, progress_steps=1000):
        self.env = env
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy strategy factor
        self.Q = defaultdict(
            lambda: np.zeros(env.action_space.n, dtype=float)
        )  # state-action value
        self.N = defaultdict(
            lambda: np.zeros(env.action_space.n, dtype=int)
        )  # tracks returns for each state-action pair
        self.policy = defaultdict(
            lambda: np.random.choice(self.env.action_space.n)
        )  # exploring starts
        self.progress_steps = progress_steps

    def generate_episode(self):
        """Generate an episode by following the stored policy."""
        episode = []
        state = self.env.reset()[0]
        while True:
            # Use epsilon-greedy policy to explore
            action = int(
                self.policy[state]
                # if state in self.policy
                if np.random.random() >= self.epsilon
                else np.random.choice(self.env.action_space.n)
            )
            next_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def update_policy(self):
        """Update the policy to be greedy with respect to Q."""
        for state in self.Q:
            # Select the action with the maximum Q-value for the state
            best_action = int(np.argmax(self.Q[state]))
            self.policy[state] = best_action

    def train(self, episodes=50000):
        """Train the policy using Monte Carlo control."""
        for i in range(episodes):
            episode = self.generate_episode()  # policy evaluation
            G = 0  # initialize return
            visited = set()  # keep track of visited state-action pairs

            # Process episode in reverse for calculating returns
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    # Implement statistics here
                    self.N[state][action] += 1  # increment count
                    alpha = 1.0 / self.N[state][action]  # step-size
                    self.Q[state][action] += alpha * (
                        G - self.Q[state][action]
                    )  # incremental update
            self.update_policy()  # policy improvement

            # Optional: print progress
            if (i + 1) % self.progress_steps == 0:
                print(f"Episode {i + 1}/{episodes} completed.")

        print("Training completed!")


# Test the trained policy
def test_policy(
    env: gym.Env,
    agents: list[MonteCarloControl],
    episodes: int = 100,
):
    batch_rewards = []
    policies = []
    Qs = []
    for agent in agents:
        policy = agent.policy
        total_rewards = 0
        for _ in range(episodes):
            state = env.reset()[0]
            while True:
                action = policy[state]
                state, reward, done, _, _ = env.step(action)
                if done:
                    total_rewards += reward
                    break
        batch_rewards.append(total_rewards)
        policies.append(policy)
        Qs.append(agent.Q)
    print("DEBUG: Q-functions")
    pprint([dict(Q) for Q in Qs])
    # plot_blackjack_values(
    #     Qs, title="Learned Q-functions for Blackjack"
    # )  # visualize Q-function
    print("DEBUG: policies")
    pprint([dict(policy) for policy in policies])
    plot_policy(
        policies, title="Learned Policies for Blackjack"
    )  # visualize policy
    for i, rewards in enumerate(batch_rewards):
        print(
            f"[{i:02d}] Average reward over {episodes} episodes: {rewards / episodes:.6f}"
        )


agents = []
instances = 5

# Train multiple agents with different initialization
for k in range(instances):
    # Initialize and train
    print(f"Start training instance {k:02d}")
    mc_control = MonteCarloControlIncremental(
        env, epsilon=0.3, progress_steps=10_000
    )
    mc_control.train(episodes=10_000_000)
    agents.append(mc_control)

# Test the trained policy
test_policy(env, agents, episodes=1000)
