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


def plot_blackjack_values(Qs: list[defaultdict], title: str):
    """Plot action-value functions for usable and non-usable ace states."""
    if plt is None:
        # Do nothing if matplotlib is not installed
        return

    x_vals = range(1, 11)  # dealer's visible card (1-10)
    y_vals = range(4, 22)  # player's hand value (4-21)

    # Initialize grids for usable and non-usable ace states
    hit_values_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)
    stand_values_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)
    hit_values_non_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)
    stand_values_non_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)

    # Populate the grids based on Q-values
    for Q in Qs:
        for player_sum in y_vals:
            for dealer_card in x_vals:
                for usable_ace in [True, False]:
                    state = (player_sum, dealer_card, usable_ace)
                    if state in Q:
                        hit_value = Q[state][1]  # Q-value for 'hit'
                        stand_value = Q[state][0]  # Q-value for 'stand'
                        idx_y = player_sum - 4
                        idx_x = dealer_card - 1

                        if usable_ace:
                            hit_values_usable[idx_y, idx_x] = hit_value
                            stand_values_usable[idx_y, idx_x] = stand_value
                        else:
                            hit_values_non_usable[idx_y, idx_x] = hit_value
                            stand_values_non_usable[idx_y, idx_x] = stand_value
        hit_values_usable /= len(Qs)
        hit_values_non_usable /= len(Qs)
        stand_values_usable /= len(Qs)
        stand_values_non_usable /= len(Qs)

    # Plot the values
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    # Set the window title
    try:
        fig.canvas.manager.set_window_title("Blackjack Values Visualization")
    except Exception:
        ...
    plot_action_values(
        axes[0],
        x_vals,
        y_vals,
        stand_values_usable,
        hit_values_usable,
        "Usable Ace",
    )
    plot_action_values(
        axes[1],
        x_vals,
        y_vals,
        stand_values_non_usable,
        hit_values_non_usable,
        "No Usable Ace",
    )
    fig.suptitle(title, fontsize=20)
    plt.show()


def plot_action_values(ax, x_vals, y_vals, stand_values, hit_values, title):
    """Helper function to plot action values."""
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Dealer's Visible Card", fontsize=14)
    ax.set_ylabel("Player's Hand Value", fontsize=14)

    # Plot "stand" values above the line
    for y_idx, y_val in enumerate(y_vals):
        ax.plot(
            x_vals,
            stand_values[y_idx, :],
            'bo-',
            label="Stand" if y_idx == 0 else "",
        )

    # Plot "hit" values below the line
    for y_idx, y_val in enumerate(y_vals):
        ax.plot(
            x_vals,
            hit_values[y_idx, :],
            'ro-',
            label="Hit" if y_idx == 0 else "",
        )

    ax.legend(fontsize=12)
    ax.grid(True)


def plot_policy(policies: list[defaultdict], title: str):
    """Plot the policy as a grid showing the chosen action for each state."""
    x_vals = range(1, 11)  # Dealer's visible card (1-10)
    y_vals = range(4, 22)  # Player's hand value (4-21)

    # Initialize grids for usable and non-usable ace states
    policy_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)
    policy_non_usable = np.zeros((len(y_vals), len(x_vals)), dtype=float)

    # Populate the grids with the action chosen by the policy
    for policy in policies:
        for player_sum in y_vals:
            for dealer_card in x_vals:
                for usable_ace in [True, False]:
                    state = (player_sum, dealer_card, usable_ace)
                    if state in policy:
                        action = policy[state]  # Chosen action
                        idx_y = player_sum - 4
                        idx_x = dealer_card - 1

                        if usable_ace:
                            policy_usable[idx_y, idx_x] += action
                        else:
                            policy_non_usable[idx_y, idx_x] += action
        policy_usable /= len(policies)
        policy_non_usable /= len(policies)

    # Plot the policy grids
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Set the window title
    try:
        fig.canvas.manager.set_window_title("Blackjack Policy Visualization")
    except Exception:
        ...
    plot_policy_grid(axes[0], x_vals, y_vals, policy_usable, "Usable Ace")
    plot_policy_grid(
        axes[1], x_vals, y_vals, policy_non_usable, "No Usable Ace"
    )
    fig.suptitle(title, fontsize=20)
    plt.show()


def plot_policy_grid(ax, x_vals, y_vals, policy_grid, title):
    """Helper function to plot a single policy grid."""
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Dealer's Visible Card", fontsize=14)
    ax.set_ylabel("Player's Hand Value", fontsize=14)

    # Adjust extent to center grid squares on ticks
    extent = [
        min(x_vals) - 0.5,
        max(x_vals) + 0.5,
        min(y_vals) - 0.5,
        max(y_vals) + 0.5,
    ]

    # Display the policy as a grid
    cax = ax.imshow(
        policy_grid,
        cmap="coolwarm",
        origin="lower",
        extent=extent,
    )

    # Add colorbar for clarity
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label("Action (0 = Stand, 1 = Hit)", fontsize=12)

    # Configure axes
    ax.set_xticks(x_vals)
    ax.set_yticks(y_vals)
    ax.grid(visible=True, color="black", linewidth=0.5)


agents = []
iterations = 5

# Train multiple agents with different initialization
for _ in range(iterations):
    # Initialize and train
    mc_control = MonteCarloControl(env, epsilon=0.4)
    mc_control.train(episodes=1000000)
    agents.append(mc_control)

# Test the trained policy
test_policy(env, agents, episodes=1000)
