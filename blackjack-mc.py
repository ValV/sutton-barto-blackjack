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
        )

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
            episode = self.generate_episode()
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
            self.update_policy()

            # Optional: print progress
            if (i + 1) % 1000 == 0:
                print(f"Episode {i + 1}/{episodes} completed.")

        print("Training completed!")


# Test the trained policy
def test_policy(env, agent, episodes=100):
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
    print("DEBUG: Q-function")
    pprint(dict(agent.Q))
    print("DEBUG: policy")
    pprint(dict(policy))
    print(
        f"Average reward over {episodes} episodes: {total_rewards / episodes:.2f}"
    )


# Initialize and train
mc_control = MonteCarloControl(env, epsilon=0.2)
mc_control.train(episodes=100000)

# Test the trained policy
test_policy(env, mc_control, episodes=1000)
