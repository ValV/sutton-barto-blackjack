"""
Balckjack Cross Entropy version

In order to play with Cross Entropy (CE) method using this code,
prepare Python environment (assuming python3-venv) and run:

```
python3 -m venv venv

. venv/bin/activate

pip install gymnasium

python blackjack-ce.py
```
"""

import pickle

from collections import defaultdict, deque
from math import ceil, log10
from multiprocessing import Lock, Manager, Process
from multiprocessing.synchronize import Lock as MPL
from pprint import pprint
from typing import Any, Dict, List, Optional, SupportsFloat, SupportsInt, Tuple

import numpy as np
import gymnasium as gym

from utils import plot_blackjack_values, plot_policy, xprint


PlayerSum = SupportsInt
DealerCard = SupportsInt
UsableAce = SupportsInt
State = Tuple[PlayerSum, DealerCard, UsableAce]
Action = SupportsInt
Reward = SupportsFloat

# Create the Blackjack environment
env = gym.make('Blackjack-v1', sab=True, render_mode=None)


# Cross Entropy Control (policy-based approach)
class CrossEntropyControl:
    def __init__(
        self,
        env: gym.Env,
        elite_fraction: float = 0.2,
        alpha: float = 0.0,
        gamma: float = 1.0,
        progress_steps: int = 10,
        thread_num: Optional[int] = None,
    ):
        """
        Initialize the Cross-Entropy Control class.

        :param env: the Gymnasium environment.
        :param elite_fraction: fraction of top-performing episodes to consider for updates.
        :param gamma: discount factor for cumulative rewards.
        """
        self.env = env
        self.elite_fraction = elite_fraction
        self.alpha = alpha
        self.gamma = gamma

        # Initialize the policy as a tabular distribution (equal probability for actions)
        self._policy: np.ndarray = (
            np.ones((32, 11, 2, 2)) / 2
        )  # player sum, dealer card, usable ace, action probabilities

        self.progress_steps = progress_steps
        self.thread_num = thread_num

    @property
    def policy(self) -> Dict[State, Action]:
        actions: np.ndarray = self._policy.argmax(axis=-1)
        return defaultdict(
            int,
            {
                (player_sum, dealer_card, usable_ace): int(
                    actions[player_sum, dealer_card, usable_ace]
                )
                for player_sum, dealer_card, usable_ace in np.ndindex(
                    actions.shape
                )
            },
        )

    @policy.setter
    def policy(self, policy: Dict[State, Action]):
        for (player_sum, dealer_card, usable_ace), action in policy.items():
            self._policy[player_sum, dealer_card, usable_ace] = (
                0  # reset all probabilities
            )
            self._policy[player_sum, dealer_card, usable_ace, action] = (
                1  # set for selected action
            )

    def generate_episode(self) -> List[Tuple[State, Action, Reward]]:
        """
        Generate a single episode following the current policy.
        :return: a list of (state, action, reward) tuples for the episode.
        """
        state: State = self.env.reset()[0]
        trajectory: List[Tuple[State, Action, Reward]] = []
        done: bool = False

        while not done:
            player_sum, dealer_card, usable_ace = state
            probs: np.ndarray = self._policy[
                player_sum, dealer_card, usable_ace
            ]  # type: ignore
            action: Action = np.random.choice([0, 1], p=probs)
            state_next, reward, done, _, _ = self.env.step(action)
            trajectory.append((state, action, reward))
            state = state_next

        return trajectory

    def evaluate_episodes(
        self, episodes: List[List[Tuple[State, Action, Reward]]]
    ) -> np.ndarray:
        """
        Compute the cumulative discounted reward for each episode.
        :param episodes: list of episodes where each episode is a list of (state, action, reward).
        :return: a numpy array of cumulative rewards for the episodes.
        """
        returns = []
        for episode in episodes:
            G = 0
            for t, (_, _, reward) in enumerate(episode):
                G += (self.gamma**t) * reward
            returns.append(G)
        return np.array(returns)

    def update_policy(
        self, elite_episodes: List[List[Tuple[State, Action, Reward]]]
    ) -> None:
        """
        Update the policy based on the elite episodes.
        :param elite_episodes: list of top-performing episodes based on cumulative rewards.
        """
        policy_update = np.zeros_like(self._policy)
        action_counts = np.zeros_like(self._policy)

        for i, episode in enumerate(elite_episodes):
            for state, action, _ in episode:
                player_sum, dealer_card, usable_ace = state
                policy_update[player_sum, dealer_card, usable_ace, action] += 1  # type: ignore
                action_counts[player_sum, dealer_card, usable_ace, action] += 1  # type: ignore

        # Normalize to get probabilities
        action_counts = action_counts.sum(axis=-1)
        mask = action_counts > 0
        policy_update[mask] /= action_counts[mask][..., None]
        policy_update[~mask] = self._policy[
            ~mask
        ]  # retain old probabilities where no updates occurred

        self._policy = (
            self.alpha * self._policy + (1 - self.alpha) * policy_update
        )

    def train(
        self,
        episodes: int = 50000,
        batch_size: int = 100,
        lock: Optional[MPL] = None,
    ) -> None:
        """
        Train the policy using the Cross-Entropy Method.
        :param episodes: total number of episodes to generate.
        :param batch_size: number of episodes per training batch.
        """
        num_iterations = episodes // batch_size
        prefix = (
            f"[{self.thread_num:02d}] " if self.thread_num is not None else ""
        )
        xprint(
            lock,
            prefix + f"Trainig for {num_iterations} iterations"
            f" from {episodes} episodes"
            f" with batch = {batch_size}.",
        )
        digits = ceil(log10(num_iterations + 1))

        for iteration in range(num_iterations):
            # Generate a batch of episodes
            batch_episodes = [
                self.generate_episode() for _ in range(batch_size)
            ]

            # Evaluate episodes and compute rewards
            rewards = self.evaluate_episodes(batch_episodes)

            # Select elite episodes
            elite_threshold = np.percentile(
                rewards, 100 * (1 - self.elite_fraction)
            )
            elite_episodes = [
                ep
                for ep, r in zip(batch_episodes, rewards)
                if r >= elite_threshold
            ]

            # Update policy
            self.update_policy(elite_episodes)  # policy improvement

            # Optional: print progress
            if (iteration + 1) % self.progress_steps == 0:
                xprint(
                    lock,
                    prefix
                    + f"Iteration {iteration + 1:{digits}d}/{num_iterations}"
                    f" completed: elite threshold = {elite_threshold}.",
                )

        xprint(lock, prefix + "Training completed!")

    def evaluate_policy(self, num_episodes: int = 1000) -> float:
        """
        Evaluate the learned policy over a number of episodes.
        :param num_episodes: number of evaluation episodes.
        :return: average cumulative reward of the policy.
        """
        total_reward: float = 0.0
        for _ in range(num_episodes):
            episode: List = self.generate_episode()
            total_reward += sum(reward for (_, _, reward) in episode)
        return total_reward / num_episodes


# Test the trained policy
def test_policy(
    env: gym.Env,
    agents: list[CrossEntropyControl],
    episodes: int = 100,
):
    batch_rewards = []
    policies = []
    # Qs = []
    for agent in agents:
        policy = agent.policy
        total_rewards = 0
        for _ in range(episodes):
            state: State = env.reset()[0]
            while True:
                action = policy[state]
                state, reward, done, _, _ = env.step(action)
                if done:
                    total_rewards += reward
                    break
        batch_rewards.append(total_rewards)
        policies.append(policy)
        # Qs.append(agent.Q)
    # print("DEBUG: Q-functions")
    # pprint([dict(Q) for Q in Qs])
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
processes = []


def run_train_process(instance, lock, context):
    env_process = gym.make('Blackjack-v1', sab=True, render_mode=None)
    ce_control = CrossEntropyControl(
        env_process,
        elite_fraction=0.2,
        alpha=0.95,
        progress_steps=10_000,
        thread_num=instance,
    )
    ce_control.train(episodes=10_000_000, lock=lock)
    serialized_control = pickle.dumps(ce_control)
    context.append((instance, serialized_control))


with Manager() as manager:
    context = manager.list()
    lock = Lock()

    # Train multiple agents with different initialization
    for k in range(instances):
        # Initialize and train
        print(f"Initialize training: instance {k:02d}")
        process = Process(target=run_train_process, args=(k, lock, context))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    for instance, serialized_control in context:
        ce_control = pickle.loads(serialized_control)
        agents.append(ce_control)

print("All processes have finished.")

# Test the trained policy
test_policy(env, agents, episodes=1000)
