from collections import defaultdict, deque
from threading import Lock
from multiprocessing.synchronize import Lock as MPL

import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


lock = Lock()


def tprint(*args, **kwargs):
    """
    Thread-safe print function.
    Ensures that prints from different threads do not mangle each other.
    """
    with lock:
        print(*args, **kwargs)


def xprint(lock, *args, **kwargs):
    if isinstance(lock, MPL):
        with lock:
            print(*args, **kwargs)
    else:
        print(lock, *args, **kwargs)


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
