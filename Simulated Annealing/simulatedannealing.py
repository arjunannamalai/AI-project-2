"""
Assignment 2: Search and Optimization

Team Members
CS24S002 - J ARJUN ANNAMALAI
CS24M106 - LOGESH .V
-------------------------------------
TSP Simulated Annealing Solver
-------------------------------------
This script implements the Simulated Annealing algorithm for the Traveling Salesman Problem (TSP).
It reads a .tsp file from the 'problems' folder of the cloned MicheleCattaneo/ant_colony_opt_TSP 
repository (e.g., eil76.tsp, ch130.tsp, d198.tsp, etc.) and runs multiple simulated annealing experiments.
"""

# Importing required libraries
import os
import math
import random
import time
import shutil   # To remove snapshot folders between runs
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Parsing the TSP file to get city coordinates
def parse_tsp_file(filepath):
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(__file__), filepath)
    cities = []
    reading = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("NODE_COORD_SECTION"):
                reading = True
                continue
            if line.upper().startswith("EOF"):
                break
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    cities.append((float(parts[1]), float(parts[2])))
    return cities

# Calculating total route distance using Euclidean distance 
def total_distance(route, cities):
    dist = 0.0
    n = len(route)
    for i in range(n):
        x1, y1 = cities[route[i]]
        x2, y2 = cities[route[(i + 1) % n]]
        dist += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

# Plotting the TSP route and saving the image
def plot_route(cities, route, filename, title="TSP Route"):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    route_coords = [cities[i] for i in route] + [cities[route[0]]]
    xs, ys = zip(*route_coords)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(filename)
    plt.close()

# Creating a GIF from saved snapshots
def create_gif(snapshot_folder, gif_filename, duration=0.3):
    if not os.path.exists(snapshot_folder):
        print("Snapshot folder does not exist:", snapshot_folder)
        return
    images = [os.path.join(snapshot_folder, f) for f in sorted(os.listdir(snapshot_folder))
              if f.endswith(".png") and f.startswith("iter_")]
    if not images:
        print("No snapshots found in:", snapshot_folder)
        return
    frames = [imageio.imread(img) for img in images]
    out_dir = os.path.dirname(gif_filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    imageio.mimsave(gif_filename, frames, duration=duration)
    print("GIF saved at:", gif_filename)

# Simulated Annealing algorithm for TSP
def simulated_annealing(cities, max_iterations=10000, time_limit=600,
                        initial_temp=1000, cooling_rate=0.995, snapshot_callback=None):
    start_time = time.time()
    n = len(cities)
    # Initialize with a random route
    current_route = list(range(n))
    random.shuffle(current_route)
    current_cost = total_distance(current_route, cities)
    best_route = current_route[:]
    best_cost = current_cost

    temperature = initial_temp
    iteration = 0

    # Capture the initial route once.
    if snapshot_callback:
        snapshot_callback(current_route, 0)

    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        # Generate a neighbor (swap two cities)
        neighbor = current_route[:]
        i, j = random.sample(range(n), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = total_distance(neighbor, cities)
        delta = neighbor_cost - current_cost

        # Decide whether to accept the neighbor
        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
            current_route = neighbor
            current_cost = neighbor_cost
            # Update best if improved
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost
                if snapshot_callback:
                    snapshot_callback(current_route, iteration)
        # Cool down the temperature
        temperature *= cooling_rate
        iteration += 1

    runtime = time.time() - start_time
    return best_route, best_cost, iteration, runtime

# Running multiple Simulated Annealing experiments
def run_experiments(tsp_file="ant_colony_opt_TSP/problems/eil76.tsp", num_runs=5,
                    max_iterations=10000, time_limit=600, initial_temp=1000, cooling_rate=0.995,
                    plots_dir="results/plots_SA", gifs_dir="results/gifs_SA",
                    export_gif_per_run=True):
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if export_gif_per_run and not os.path.exists(gifs_dir):
        os.makedirs(gifs_dir)

    cities = parse_tsp_file(tsp_file)
    print(f"Parsing TSP file: {tsp_file}")
    print(f"Number of cities: {len(cities)}\n")
    print("Running Simulated Annealing for TSP...\n")
    best_costs = []
    run_times = []

    for run in range(1, num_runs + 1):
        snapshot_folder = None
        if export_gif_per_run:
            snapshot_folder = os.path.join(plots_dir, f"sa_run_{run}_snapshots")
            if os.path.exists(snapshot_folder):
                shutil.rmtree(snapshot_folder)
            os.makedirs(snapshot_folder)

        def snapshot_callback(route, iteration, snap_folder=snapshot_folder, run_num=run):
            if snap_folder is None:
                return
            snap_path = os.path.join(snap_folder, f"iter_{iteration:05d}.png")
            plot_route(cities, route, snap_path, title=f"Run {run_num} - Iteration {iteration}")

        current_callback = snapshot_callback if snapshot_folder else None
        route, cost, total_iter, runtime = simulated_annealing(cities, max_iterations, time_limit,
                                                                initial_temp, cooling_rate,
                                                                current_callback)
        best_costs.append(cost)
        run_times.append(runtime)
        print(f"Run {run}: Best Cost = {cost:.2f}, Total Iterations = {total_iter}, Time = {runtime:.2f}s")
        plot_filename = os.path.join(plots_dir, f"sa_run_{run}_route.png")
        plot_route(cities, route, plot_filename, title=f"Run {run}: Best Cost = {cost:.2f}")

        if export_gif_per_run:
            gif_path = os.path.join(gifs_dir, f"sa_run_{run}_animation.gif")
            create_gif(snapshot_folder, gif_path, duration=0.3)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_runs + 1), best_costs, marker='o')
    plt.title("Best Cost per Run (SA)")
    plt.xlabel("Run")
    plt.ylabel("Cost")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_runs + 1), run_times, marker='o', color="orange")
    plt.title("Run Time per Experiment (SA)")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    summary_path = os.path.join(plots_dir, "sa_performance_summary.png")
    plt.savefig(summary_path)
    plt.show()

    print(f"\nAverage Best Cost over {num_runs} SA runs: {sum(best_costs) / num_runs:.2f}")
    print(f"Average Time over {num_runs} SA runs: {sum(run_times) / num_runs:.2f}s")

# Running a snapshot experiment to generate a single animated GIF for SA progress
def run_snapshot_experiment(tsp_file="ant_colony_opt_TSP/problems/eil76.tsp",
                            max_iterations=10000, time_limit=600, initial_temp=1000, cooling_rate=0.995,
                            plots_dir="results/plots_SA", gifs_dir="results/gifs_SA"):
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(gifs_dir):
        os.makedirs(gifs_dir)
    snapshot_folder = os.path.join(plots_dir, "sa_snapshots")
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

    cities = parse_tsp_file(tsp_file)
    print("Starting snapshot-enabled Simulated Annealing for generating GIF...\n")
    
    def snapshot_callback(route, iteration):
        snap_path = os.path.join(snapshot_folder, f"iter_{iteration:05d}.png")
        plot_route(cities, route, snap_path, title=f"Iteration {iteration}")

    route, cost, total_iter, runtime = simulated_annealing(cities, max_iterations, time_limit,
                                                            initial_temp, cooling_rate, snapshot_callback)
    final_plot = os.path.join(plots_dir, "sa_final_route.png")
    plot_route(cities, route, final_plot, title="Final Route (SA Snapshot)")
    print(f"Snapshot Experiment Results:\n Best Cost = {cost:.2f},\n Total Iterations = {total_iter},\n Time = {runtime:.2f}s\n")
    gif_path = os.path.join(gifs_dir, "simulated_annealing_animation.gif")
    create_gif(snapshot_folder, gif_path, duration=0.3)

# Main function
if __name__ == "__main__":
    random.seed(42)
    run_experiments(tsp_file="ant_colony_opt_TSP/problems/eil76.tsp", num_runs=5,
                    max_iterations=10000, time_limit=600, initial_temp=1000, cooling_rate=0.995,
                    plots_dir="results/plots_SA", gifs_dir="results/gifs_SA", export_gif_per_run=True)
    run_snapshot_experiment(tsp_file="ant_colony_opt_TSP/problems/eil76.tsp",
                            max_iterations=10000, time_limit=600, initial_temp=1000, cooling_rate=0.995,
                            plots_dir="results/plots_SA", gifs_dir="results/gifs_SA")