"""
Assignment 2: Search and Optimization

Team Members
CS25S002 - J ARJUN ANNAMALAI    
CS24M106 - LOGESH .V

-------------------------------------
Branch and Bound using Frozen Lake
-------------------------------------
"""

import gymnasium as gym
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import heapq
from io import BytesIO
import imageio

# Define constants
hole_positions = {3, 5, 11, 12}
goal_position = 15

def get_next_state(current_state, action):
    """
    Get the next state given current state and action
    Returns None if the next state is a hole
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

class Node:
    """Node class for Branch and Bound algorithm"""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        
    def __lt__(self, other):
        # For priority queue ordering - prioritize lower cost
        return self.path_cost < other.path_cost

def branch_and_bound(start_state, timeout=600):
    """
    Branch and Bound algorithm to find optimal path
    Returns: path, time_taken, nodes_expanded
    """
    start_time = time.time()
    
    # Use priority queue for frontier
    frontier = []
    heapq.heappush(frontier, (0, Node(start_state)))
    
    # Track explored states and their costs
    explored = {}  # state -> cost
    
    # Track metrics
    nodes_expanded = 0
    
    while frontier:
        # Check timeout
        if time.time() - start_time > timeout:
            return None, time.time() - start_time, nodes_expanded
        
        # Get node with lowest cost
        _, current_node = heapq.heappop(frontier)
        current_state = current_node.state
        
        # Check if goal reached
        if current_state == goal_position:
            # Reconstruct path
            path = []
            node = current_node
            while node.parent:
                path.append(node.action)
                node = node.parent
            path.reverse()
            return path, time.time() - start_time, nodes_expanded
        
        # Skip if we've seen this state with lower cost
        if current_state in explored and explored[current_state] <= current_node.path_cost:
            continue
        
        # Mark as explored
        explored[current_state] = current_node.path_cost
        nodes_expanded += 1
        
        # Expand node - try all actions
        for action in range(4):
            new_state = get_next_state(current_state, action)
            if new_state is not None:
                # Calculate cost (add 1 for each step)
                path_cost = current_node.path_cost + 1
                
                # Skip if we've seen this state with lower cost
                if new_state in explored and explored[new_state] <= path_cost:
                    continue
                
                # Create new node
                child = Node(
                    state=new_state,
                    parent=current_node,
                    action=action,
                    path_cost=path_cost
                )
                
                # Add to frontier
                heapq.heappush(frontier, (path_cost, child))
    
    # No path found
    return None, time.time() - start_time, nodes_expanded

def run(episodes, is_training=True, render=False):
    """Run multiple episodes with Branch and Bound algorithm"""
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
        
        # Create list to record frames for this episode
        frames_episode = []
        
        # Capture initial frame if rendering
        if render:
            frame = env.render()
            frames_episode.append(frame)
            time.sleep(0.5)
        
        while not terminated and not truncated:
            # Check timeout
            if time.time() - start_time > 600:  # 10 minutes
                print("Episode timed out")
                truncated = True
                break
            
            # Use Branch and Bound to find path
            path, path_time, nodes_expanded = branch_and_bound(state)
            
            if not path:
                print("No path found")
                break
            
            # Take action from path
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
        
        # Save episode GIF if rendering enabled
        if render and frames_episode:
            gif_filename = f'episode_{episode+1}_simulation.gif'
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
    
    # Plotting
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
    plt.savefig('branch_and_bound_performance.png')
    plt.show()
    
    # Save performance data
    performance = {
        'rewards': total_rewards,
        'steps': total_steps,
        'times': times,
        'convergence_points': convergence_points,
        'nodes_expanded': nodes_expanded_list
    }
    
    with open('bnb_performance.pkl', 'wb') as f:
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