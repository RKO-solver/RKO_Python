# Travelling Salesperson Problem (TSP) Tutorial

This tutorial demonstrates how to solve the Travelling Salesperson Problem (TSP) using the **RKO** framework. TSP is a classic combinatorial optimization problem where the goal is to find the shortest possible route that visits every city exactly once and returns to the origin city.

In our implementation, the environment handles generating a random TSP instance, calculating distances, and decoding random keys into a valid tour.

## 1. Creating the Environment

Below is the complete implementation of the `TSPProblem` environment. It extends the `RKOEnvAbstract` class, defining the problem's specifics, such as city generation, random key decoding, and the cost function.

```python
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from rko import RKO, RKOEnvAbstract, check_env, FileLogger, HistoryPlotter

class TSPProblem(RKOEnvAbstract):
    """
    An implementation of the Traveling Salesperson Problem (TSP) environment for the RKO solver.
    This class generates a random instance upon initialization.
    """
    def __init__(self, num_cities: int = 20):
        super().__init__()
        print(f"Generating a random TSP instance with {num_cities} cities.")

        self.num_cities = num_cities
        self.instance_name = f"TSP_{num_cities}_cities"
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}

        self.save_q_learning_report = False

        # Generate city coordinates and the distance matrix
        self.cities = self._generate_cities(num_cities)
        self.distance_matrix = self._calculate_distance_matrix()

        # Solution size represents the number of randomly generated keys
        self.tam_solution = self.num_cities

        # Customizing parameters for metaheuristics (example)
        self.BRKGA_parameters = {'p': [100, 50], 'pe': [0.20, 0.15], 'pm': [0.05], 'rhoe': [0.70]}
        self.SA_parameters = {'SAmax': [10, 5], 'alphaSA': [0.5, 0.7], 'betaMin': [0.01, 0.03], 'betaMax': [0.05, 0.1], 'T0': [10]}
        self.ILS_parameters = {'betaMin': [0.10, 0.5], 'betaMax': [0.20, 0.15]}
        self.VNS_parameters = {'kMax': [5, 3], 'betaMin': [0.05, 0.1]}
        self.PSO_parameters = {'PSize': [100, 50], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]}
        self.GA_parameters = {'sizePop': [100, 50], 'probCros': [0.98], 'probMut': [0.005, 0.01]}
        self.LNS_parameters = {'betaMin': [0.10], 'betaMax': [0.30], 'TO': [100], 'alphaLNS': [0.95, 0.9]}

    def _generate_cities(self, num_cities: int) -> np.ndarray:
        """Generates random (x, y) coordinates for each city in a 100x100 grid."""
        return np.random.rand(num_cities, 2) * 100 

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Computes the Euclidean distance between every pair of cities."""
        num_cities = len(self.cities)
        dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i, num_cities):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def decoder(self, keys: np.ndarray) -> list[int]:
        """
        Decodes a random-key vector into a TSP tour.
        The tour is determined by the sorted order of the keys.
        """
        tour = np.argsort(keys)
        return tour.tolist()

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        """
        Calculates the total distance of a given TSP tour.
        The RKO framework will minimize this value.
        """
        total_distance = 0
        num_cities_in_tour = len(solution)
        for i in range(num_cities_in_tour):
            from_city = solution[i]
            to_city = solution[(i + 1) % num_cities_in_tour]
            total_distance += self.distance_matrix[from_city, to_city]
        
        return total_distance
```

### Decoding Strategy 
In the `decoder` method, `np.argsort()` takes the randomly generated variables (in `[0, 1)`) and returns the indices representing their sorted order. This creates a valid permutation of cities automatically.

## 2. Running the RKO Solver

Once the environment is set up, you can instantiate the `RKO` solver and initiate the search. This setup uses multiple metaheuristics concurrently.

```python
if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Instantiate the problem environment (50 cities).
    env = TSPProblem(num_cities=50)
    check_env(env)  # Verify the environment implementation is valid!
    
    # 2. Setup the logger
    logger = FileLogger(os.path.join(current_directory, 'results.txt'), reset=True)

    # 3. Instantiate the RKO solver.
    solver = RKO(env=env, logger=logger)
    
    # 4. Run the solver for 10 seconds with selected metaheuristics
    final_cost, final_solution, time_to_best = solver.solve(
        time_total=10, 
        runs=1,
        vns=1, 
        ils=1,
        sa=1
    )
    
    solution = env.decoder(final_solution)
    
    # 5. Output and logging
    HistoryPlotter.plot_convergence(os.path.join(current_directory, 'results.txt'), run_number=1, title="TSP Convergence").show()

    print("\n" + "="*30)
    print("      FINAL RESULTS      ")
    print("="*30)
    print(f"Instance Name: {env.instance_name}")
    print(f"Best Tour Cost Found: {final_cost:.4f}")
    print(f"Time to Find Best Solution: {time_to_best}s")
    print(f"Best Tour (City Sequence): {solution}")
```

This will run VNS, ILS and SA in parallel, sharing solutions between all of them. The `HistoryPlotter` can be used to generate beautiful convergence graphs visualizing the search trajectory for the best value.
