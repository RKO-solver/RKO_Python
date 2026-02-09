import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from typing import List, Tuple

class HistoryPlotter:
    """
    Class responsible for parsing execution logs and plotting convergence history.
    """

    @staticmethod
    def parse_log_file(file_path: str) -> List[Tuple[str, float, float, int]]:
        """
        Parses the log file to extract the history of best solutions found.

        Args:
            file_path (str): The path to the log file.

        Returns:
            List[Tuple[str, float, float, int]]: A list of tuples containing
            (metaheuristic_name, fitness, time, run_number).
        """
        data = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []

        # Regex to match log formats in RKO.py, e.g.:
        # "SA 0 NEW BEST: 123.45 - BEST: 100.0 - Time: 10.0s - 50"
        # or "SA 0 NEW BEST: 123.45 - Time: 10.0s - 50"
        pattern = re.compile(
            r"(?P<metaheuristic>.*?) NEW BEST: (?P<fitness>[-]?[\d\.]+)(?: - BEST: [\d\.]+)? - Time: (?P<time>[\d\.]+)s"
        )

        run_count = 1
        last_time = 0.0

        for line in lines:
            if "NEW BEST" in line:
                match = pattern.search(line)
                if match:
                    meta_name = match.group("metaheuristic").strip()
                    fitness = float(match.group("fitness"))
                    time_val = float(match.group("time"))

                    # Detect new run if time decreases compared to the last entry
                    if data and time_val < last_time:
                        run_count += 1
                    
                    data.append((meta_name, fitness, time_val, run_count))
                    last_time = time_val

        return data

    @staticmethod
    def plot_convergence(
        file_path: str, 
        run_number: int = 1, 
        title: str = "Convergence History", 
        x_label: str = "Time (s)", 
        y_label: str = "Cost",
        skip_pool: bool = True
    ) -> plt.Figure:
        """
        Plots the convergence history for a specific run.

        Args:
            file_path (str): Path to the log file.
            run_number (int): The run number to plot (1-indexed).
            title (str): Title of the plot.
            x_label (str): Label for the X-axis.
            y_label (str): Label for the Y-axis.
            skip_pool (bool): Whether to exclude 'pool' metaheuristic entries.

        Returns:
            plt.Figure: The matplotlib figure object.
        """
        history = HistoryPlotter.parse_log_file(file_path)
        
        if not history:
            print("No data extracted from log file.")
            # Return empty figure to avoid crash on .show()
            return plt.figure()

        df = pd.DataFrame(history, columns=['metaheuristic', 'fitness', 'time', 'run'])
        df = df[df['run'] == run_number]
        
        if df.empty:
            print(f"No data found for run {run_number}.")
            return plt.figure()

        if skip_pool:
            df = df[df['metaheuristic'] != 'pool']

        fig = plt.figure(figsize=(10, 6))
        
        # Plot the trajectory of the best cost
        plt.plot(df['time'], df['fitness'], linestyle='--', color='blue', alpha=0.7, label='Best Cost')

        metaheuristics = df['metaheuristic'].unique()
        # Use get_cmap to avoid linting errors with direct attribute access
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(metaheuristics)))

        for mh, color in zip(metaheuristics, colors):
            subset = df[df['metaheuristic'] == mh]
            plt.scatter(
                subset['time'], 
                subset['fitness'], 
                color=color, 
                s=100, 
                label=mh, 
                zorder=5, 
                edgecolors='k'
            )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        return fig
