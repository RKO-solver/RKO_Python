import numpy as np
import os
import sys
import random
from abc import ABC, abstractmethod
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.append(project_root_directory)
from RKO import RKO
from Environment import RKOEnvAbstract, check_env
import matplotlib.pyplot as plt
from LogStrategy import FileLogger
from Plots import HistoryPlotter

class TSPProblem(RKOEnvAbstract):
    """
    An implementation of the Traveling Salesperson Problem (TSP) environment for the RKO solver.
    This class generates a random instance upon initialization.
    """
    def __init__(self, num_cities: int = 20):
        super().__init__() # Initialize the abstract base class
        print(f"Generating a random TSP instance with {num_cities} cities.")

        self.num_cities = num_cities
        self.instance_name = f"TSP_{num_cities}_cities"
        self.LS_type: str = 'Best'
        self.dict_best: dict = {} # No known optimal for random instances

        self.save_q_learning_report = False

        # Generate city coordinates and the distance matrix
        self.cities = self._generate_cities(num_cities)
        self.distance_matrix = self._calculate_distance_matrix()

        # --- Set required attributes from the abstract class ---
        self.tam_solution = self.num_cities

        # You can customize the parameters for each metaheuristic here
        self.BRKGA_parameters = {
            'p': [100, 50],          
            'pe': [0.20, 0.15],      
            'pm': [0.05],        
            'rhoe': [0.70]       
        }

        self.SA_parameters = {
            'SAmax': [10, 5],     
            'alphaSA': [0.5, 0.7],  
            'betaMin': [0.01, 0.03],   
            'betaMax': [0.05, 0.1],   
            'T0': [10]      
        }

        
        self.ILS_parameters = {
            'betaMin': [0.10,0.5],   
            'betaMax': [0.20,0.15]    
        }
       

        self.VNS_parameters = {
            'kMax': [5,3],         
            'betaMin': [0.05, 0.1]    
        }

        self.PSO_parameters = {
            'PSize': [100,50],     
            'c1': [2.05],     
            'c2': [2.05],        
            'w': [0.73]         
        }

        
        self.GA_parameters = {
            'sizePop': [100,50],    
            'probCros': [0.98],  
            'probMut': [0.005, 0.01]   
        }

       
        self.LNS_parameters = {
            'betaMin': [0.10],   
            'betaMax': [0.30],  
            'TO': [100],       
            'alphaLNS': [0.95,0.9] 
        }

    def _generate_cities(self, num_cities: int) -> np.ndarray:
        """Generates random (x, y) coordinates for each city."""
        return np.random.rand(num_cities, 2) * 100 # Cities in a 100x100 grid

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
        Decodes a random-key vector into a TSP tour (a permutation of cities).
        The tour is determined by the sorted order of the keys.
        """
        # np.argsort returns the indices that would sort an array, creating a permutation.
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
            # Connect to the next city, wrapping around to the start from the last city
            to_city = solution[(i + 1) % num_cities_in_tour]
            total_distance += self.distance_matrix[from_city, to_city]
        
        return total_distance
    
    def plot_tour(self, tour: list[int], cost: float):
        """
        Plots the cities and the final tour found by the solver.
        """
        # Add the first city to the end of the tour to close the loop for plotting
        tour_to_plot = tour + [tour[0]]
        
        # Extract the X and Y coordinates in the tour's order
        ordered_cities = self.cities[tour_to_plot, :]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Panel 1: Scatter plot of the cities ---
        ax1.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', zorder=5)
        for i, city in enumerate(self.cities):
            ax1.text(city[0], city[1] + 1, str(i), fontsize=9)
        ax1.set_title(f'Scatter Plot of {self.num_cities} Cities')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.grid(True)
        
        # --- Panel 2: Cities with the best tour found ---
        ax2.scatter(self.cities[:, 0], self.cities[:, 1], c='blue', zorder=5)
        ax2.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'r-')
        for i, city in enumerate(self.cities):
            ax2.text(city[0], city[1] + 1, str(i), fontsize=9)
        ax2.set_title(f'Best Tour Found (Cost: {cost:.2f})')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    # 1. Instantiate the problem environment.
    #    This will automatically generate a new TSP instance with 50 cities.
    env = TSPProblem(num_cities=50)
    check_env(env)  # Verify the environment implementation
    
    # 2. Setup the logger
    logger = FileLogger(os.path.join(current_directory, 'results.txt'), reset=True)

    # 3. Instantiate the RKO solver, passing the environment.
    solver = RKO(
        env=env, 
        logger=logger
    )
    
    final_cost, final_solution, time_to_best = solver.solve(
        time_total=10, 
        runs=1,
        vns=1, 
        ils=1,
        sa=1
    )
    
    solution = env.decoder(final_solution)
    
    # Plot the convergence
    HistoryPlotter.plot_convergence(os.path.join(current_directory, 'results.txt'), run_number=1, title="TSP Convergence").show()

    env.plot_tour(solution, final_cost)
    print("\n" + "="*30)
    print("      FINAL RESULTS      ")
    print("="*30)
    print(f"Instance Name: {env.instance_name}")
    print(f"Best Tour Cost Found: {final_cost:.4f}")
    print(f"Time to Find Best Solution: {time_to_best}s")
    print(f"Best Tour (City Sequence): {solution}")