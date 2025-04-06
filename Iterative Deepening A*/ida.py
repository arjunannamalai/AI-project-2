"""
Assignment 2: Search and Optimization

Team Members
CS25S002 - J ARJUN ANNAMALAI    
CS24M106 - LOGESH .V

-------------------------------------
Iterative Deepening A* using Frozen Lake
-------------------------------------
"""

import gymnasium as gym
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import heapq  # not used here but kept for consistency with other files
from io import BytesIO
import imageio

# Define constants
hole_positions = {3, 5, 11, 12}
goal_position = 15

def get_next_state(current_state, action):
    """
    Get the next state given current state and action.
    Returns None if the next state is a hole.
    """
    row = current_state // 4
    col = current_state % 4
    new_row, new_col = row, col

    if action == 0:  # Left
        new_col = max(col - 1, 0)
    elif action == 1:  # Down
        new_row = min(row + 1, 3)
    elif action == 2:  # Right
        new_col = min(col + 1, 3)
    elif action == 3:  # Up
        new_row = max(row - 1, 0)

    new_state = new_row * 4 + new_col
    return None if new_state in hole_positions else new_state

def heuristic(state):
    """Manhattan distance heuristic from current state to goal."""
    row, col = state // 4, state % 4
    goal_row, goal_col = goal_position // 4, goal_position % 4
    return abs(row - goal_row) + abs(col - goal_col)

def ida_star(start_state, timeout=600):
    """
    Iterative Deepening A* (IDA*) algorithm to find an optimal path.
    Returns: path (list of actions), time taken, nodes expanded.
    """
    start_time = time.time()
    threshold = heuristic(start_state)
    nodes_expanded = 0

    path = [start_state]
    actions_path = []

    def search(path, g, threshold):
        nonlocal nodes_expanded
        current_state = path[-1]
        f = g + heuristic(current_state)
        if f > threshold:
            return f
        if current_state == goal_position:
            return "FOUND"
        minimum = float('inf')
        for action in range(4):
            new_state = get_next_state(current_state, action)
            if new_state is None:
                continue
            # Avoid cycles
            if new_state in path:
                continue
            nodes_expanded += 1
            path.append(new_state)
            actions_path.append(action)
            t = search(path, g + 1, threshold)
            if t == "FOUND":
                return "FOUND"
            if t < minimum:
                minimum = t
            path.pop()
            actions_path.pop()
        return minimum

    while True:
        t = search(path, 0, threshold)
        if t == "FOUND":
            return actions_path, time.time() - start_time, nodes_expanded
        if t == float('inf') or time.time() - start_time > timeout:
            return None, time.time() - start_time, nodes_expanded
        threshold = t

def run(episodes, is_training=True, render=False):
    """Run multiple episodes with the Iterative Deepening A* algorithm."""
    # Use rgb_array to capture rendered frames if rendering is enabled
    env = gym.make('FrozenLake-v1',
                   map_name="4x4",
                   is_slippery=True,
                   render_mode="rgb_array" if render else None)

    # Performance tracking
    total_rewards = []
    total_steps = []
    convergence_points = []
    times = []
    nodes_expanded_list = []

    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        start_time = time.time()
        state = env.reset()[0]
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0

        # List to record frames for this episode
        frames_episode = []

        # Capture initial frame if rendering
        if render:
            frame = env.render()
            frames_episode.append(frame)
            time.sleep(0.5)

        while not terminated and not truncated:
            # Check timeout (10 minutes)
            if time.time() - start_time > 600:
                print("Episode timed out")
                truncated = True
                break

            # Use IDA* to find a path
            path, path_time, nodes_expanded = ida_star(state)

            if not path:
                print("No path found")
                break

            # Execute the first action in the found path
            action = path[0]
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Capture frame after the step if rendering
            if render:
                frame = env.render()
                frames_episode.append(frame)
                time.sleep(0.5)

            episode_reward += reward
            state = next_state
            steps += 1

            if terminated:
                if state == goal_position:
                    print(f"Goal reached in {steps} steps!")
                else:
                    print("Fell in a hole!")
                break

        # Save episode GIF if rendering is enabled
        if render and frames_episode:
            gif_filename = f'ida_episode_{episode+1}_simulation.gif'
            imageio.mimsave(gif_filename, frames_episode, duration=0.5)
            print(f"Saved {gif_filename}")

        # Record performance for overall metrics
        episode_time = time.time() - start_time
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        times.append(episode_time)
        nodes_expanded_list.append(nodes_expanded)

        if terminated and state == goal_position:
            convergence_points.append(steps)
        else:
            convergence_points.append(None)

        print(f"Episode {episode+1} - Reward: {episode_reward}, Steps: {steps}, Time: {episode_time:.2f}s")

    env.close()

    # Plotting overall performance
    plt.figure(figsize=(12, 8))

    # Plot 1: Time per episode
    plt.subplot(2, 2, 1)
    plt.plot(range(1, episodes+1), times)
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.title('Time Taken per Episode')

    # Plot 2: Steps per episode
    plt.subplot(2, 2, 2)
    plt.plot(range(1, episodes+1), total_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')

    # Plot 3: Rewards per episode
    plt.subplot(2, 2, 3)
    plt.plot(range(1, episodes+1), total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')

    # Plot 4: Nodes expanded per episode
    plt.subplot(2, 2, 4)
    plt.plot(range(1, episodes+1), nodes_expanded_list)
    plt.xlabel('Episode')
    plt.ylabel('Nodes Expanded')
    plt.title('Nodes Expanded per Episode')

    plt.tight_layout()
    plt.savefig('ida_performance.png')
    plt.show()

    # Save performance data
    performance = {
        'rewards': total_rewards,
        'steps': total_steps,
        'times': times,
        'convergence_points': convergence_points,
        'nodes_expanded': nodes_expanded_list
    }
    with open('ida_performance.pkl', 'wb') as f:
        pickle.dump(performance, f)

    # Print results summary
    successful = sum(1 for r in total_rewards if r == 1)
    print("\nPerformance Summary:")
    print(f"Successful episodes: {successful}/{episodes} ({successful/episodes*10:.1f}%)")

    if successful > 0:
        avg_steps = sum(s for s, r in zip(total_steps, total_rewards) if r == 1) / successful
        avg_time = sum(t for t, r in zip(times, total_rewards) if r == 1) / successful
        avg_nodes = sum(n for n, r in zip(nodes_expanded_list, total_rewards) if r == 1) / successful
        print(f"Average steps on success: {avg_steps:.2f}")
        print(f"Average time on success: {avg_time:.2f} seconds")
        print(f"Average nodes expanded on success: {avg_nodes:.2f}")

    print(f"Average time per episode: {np.mean(times):.2f} seconds")
    print(f"Average nodes expanded per episode: {np.mean(nodes_expanded_list):.2f}")

if __name__ == '__main__':
    run(episodes=10, is_training=True, render=True)