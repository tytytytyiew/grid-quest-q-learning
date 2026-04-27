import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import os

GRID_SIZE = 8
ACTIONS = ['↑', '↓', '←', '→']

ACTION_MAP = {
    '↑': (-1, 0),
    '↓': (1, 0),
    '←': (0, -1),
    '→': (0, 1)
}

TREASURE_POS = (7, 7)
WALLS = [(1, 1), (1, 2), (2, 1), (3, 3), (3, 4), (4, 3),
         (5, 5), (5, 6), (6, 5), (2, 5), (2, 6), (4, 1)]

EPISODES = 800
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


def get_reward(row, col):
    if (row, col) == TREASURE_POS:
        return 100
    elif (row, col) in WALLS:
        return -20
    return -0.5


def is_valid_position(row, col):
    if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
        return False
    if (row, col) in WALLS:
        return False
    return True


def get_next_position(row, col, action):
    dr, dc = ACTION_MAP[action]
    nr, nc = row + dr, col + dc
    if is_valid_position(nr, nc):
        return nr, nc
    return row, col


def plot_training_results(rewards_history):
    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title('Награда агента во время обучения')

    plt.plot(rewards_history, alpha=0.4)

    window = 30
    if len(rewards_history) >= window:
        smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), smoothed, 'r', linewidth=2)

    plt.xlabel('Эпизоды')
    plt.ylabel('Награда')
    plt.grid()

    plt.savefig('static/training_plot.png')
    plt.close()


def train():
    q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
    rewards_history = []
    epsilon = EPSILON

    for _ in range(EPISODES):
        state = (0, 0)
        total_reward = 0

        for _ in range(200):
            if np.random.random() < epsilon:
                action_idx = np.random.randint(4)
            else:
                action_idx = np.argmax(q_table[state])

            action = ACTIONS[action_idx]

            nr, nc = get_next_position(state[0], state[1], action)
            next_state = (nr, nc)

            reward = get_reward(nr, nc)
            total_reward += reward

            best_next = np.max(q_table[next_state])
            q_table[state][action_idx] += ALPHA * (reward + GAMMA * best_next - q_table[state][action_idx])

            state = next_state

            if state == TREASURE_POS:
                break

        rewards_history.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    with open('q_model.pkl', 'wb') as f:
        pickle.dump({
            'q_table': dict(q_table),
            'grid_size': GRID_SIZE,
            'treasure_pos': TREASURE_POS,
            'walls': WALLS,
            'actions': ACTIONS
        }, f)

    plot_training_results(rewards_history)


if __name__ == "__main__":
    train()