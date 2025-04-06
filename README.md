# Assignment 2: Search and Optimization

**Team Members**  
CS25S002 - J ARJUN ANNAMALAI  
CS24M106 - LOGESH .V

---

## 1. Branch and Bound on Frozen Lake

This project uses a Branch and Bound algorithm to solve the FrozenLake environment from Gymnasium. During each episode, the algorithm tracks various performance metrics. If rendering is enabled, it also generates a GIF simulation for each episode.

### Features
- **Performance Metrics:** Records performance data such as time, steps, and nodes expanded.
- **Visualization:** Produces a performance plot saved as `branch_and_bound_performance.png`.
- **Simulation GIFs:** Generates a GIF for each episode (e.g., `episode_X_simulation.gif`).

### Requirements
- **Python 3.6+**
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- `matplotlib`
- `imageio`
- `numpy`

### Installation
Install the dependencies via pip:
```bash
pip install gymnasium matplotlib imageio numpy
```

---

## 2. Iterative Deepening A* for Frozen Lake

This project implements the Iterative Deepening A* (IDA*) algorithm to solve the FrozenLake environment from Gymnasium. The algorithm tracks performance metrics and, optionally, generates a GIF simulation for each episode along with an overall performance plot.

### Requirements
- **Python 3.6+**
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- `matplotlib`
- `imageio`
- `numpy`

### Installation
Install the required packages using pip:
```bash
pip install gymnasium matplotlib imageio numpy
```

### How to Run
1. Open your terminal and navigate to the Iterative Deepening A* folder:
    ```bash
    cd "/Users/arjun/Desktop/AI Project 2/Iterative Deepening A*"
    ```
2. Execute the script:
    ```bash
    python3 ida.py
    ```

After running, the following outputs are generated:
- A performance plot (`ida_performance.png`)
- A pickle file with detailed performance data (`ida_performance.pkl`)
- Episode simulation GIFs (if `render=True`)

---

## 3. TSP Hill Climbing Solver

This script implements the Hill Climbing algorithm for solving the Traveling Salesman Problem (TSP). It reads a `.tsp` file, executes multiple experiments (default is 5 runs), and generates visual outputs for each run.

### Features
- **Parsing TSP Files:** Reads city coordinates from a specified problem file.
- **Route Evaluation:** Computes total route distance using Euclidean distance.
- **Visualization:** Saves route plots for each run.
- **GIF Creation:** Combines snapshots into an animated GIF showcasing the progress of improvements.

### Requirements
- **Python 3.x**
- `matplotlib`
- `imageio`
- `numpy` (if indirectly used)

### Installation
Install the required libraries:
```bash
pip install matplotlib imageio
```

---

## 4. TSP Simulated Annealing Solver

This project implements a Simulated Annealing (SA) algorithm to solve the Traveling Salesman Problem (TSP). It processes a TSP file (e.g., `eil76.tsp`) and performs multiple experiments, generating route plots and animated GIFs to visualize the solution progress.

### Features
- **TSP File Parsing:** Loads city coordinates from a TSP file.
- **Route Evaluation:** Calculates the total route distance using Euclidean distance.
- **Visualization:** Saves computed route plots for each SA run.
- **Animation:** Creates animated GIFs from snapshots captured during the running of the algorithm.
- **Multiple Experiments:** Supports several runs to compare best cost and runtime.

### Requirements
- **Python 3.x**
- `matplotlib`
- `imageio`

### Installation
Install the required packages:
```bash
pip install matplotlib imageio
```

### How to Run
1. Ensure the TSP problem file (e.g., `eil76.tsp`) is in the `ant_colony_opt_TSP/problems` folder.
2. Navigate to the `Simulated Annealing` folder:
    ```bash
    cd "/Users/arjun/Desktop/AI Project 2/Simulated Annealing"
    ```
3. Run the solver:
    ```bash
    python simulatedannealing.py
    ```

---

Happy coding and experimentation!