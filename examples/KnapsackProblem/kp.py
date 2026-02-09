import numpy as np
import os
import sys

# Standard import assumed after package installation
from rko import RKO, RKOEnvAbstract, FileLogger, HistoryPlotter

class KnapsackProblem(RKOEnvAbstract):
    """
    An implementation of the Knapsack Problem environment for the RKO solver.
    """
    def __init__(self, instance_path: str):
        super().__init__() # Initialize the abstract base class
        print(f"Loading Knapsack Problem instance from: {instance_path}")

        self.instance_name = instance_path.split('/')[-1]
        self.LS_type: str = 'Best' # Options: 'Best' or 'First'
        self.dict_best: dict = {"Best": [149]}
        self._load_data(instance_path)

        # --- Set required attributes from the abstract class ---
        self.tam_solution = self.n_items
        self.save_q_learning_report = False
        
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

    def _load_data(self, instance_path: str):
        """
        Loads the knapsack problem data from a text file.
        """
        with open(instance_path, 'r') as f:
            lines = f.readlines()
            # First line: number of items and capacity
            self.n_items, self.capacity = map(int, lines[0].strip().split())
            
            self.profits = []
            self.weights = []
            
            # Subsequent lines: profit and weight for each item
            for line in lines[1:]:
                if line.strip(): # Ensure the line is not empty
                    p, w = map(int, line.strip().split())
                    self.profits.append(p)
                    self.weights.append(w)

    def decoder(self, keys: np.ndarray) -> list[int]:
        """
        Decodes a random-key vector into a knapsack solution.
        An item is included if its corresponding key is > 0.5.
        """
        # A solution is a binary list where 1 means the item is in the knapsack
        solution = [1 if key > 0.5 else 0 for key in keys]
        return solution

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        """
        Calculates the cost of the knapsack solution.
        Since this is a maximization problem, the cost is the negative of the total profit.
        A penalty is applied for exceeding the knapsack's capacity.
        """
        total_profit = 0
        total_weight = 0
        for i, item_included in enumerate(solution):
            if item_included:
                total_profit += self.profits[i]
                total_weight += self.weights[i]

        # Apply a heavy penalty for infeasible solutions (exceeding capacity)
        if total_weight > self.capacity:
            penalty = 100000 * (total_weight - self.capacity)
            total_profit -= penalty
            
        # The RKO framework assumes a minimization problem by default,
        # so we return the negative of the profit.
        return -total_profit

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    env = KnapsackProblem(os.path.join(current_directory,'kp50.txt'))
    logger = FileLogger(os.path.join(current_directory,'results.txt'), reset=True)
    solver = RKO(env, logger=logger)
    solver.solve(time_total=30, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, runs=2)
    
    # Plot convergence for the first run
    HistoryPlotter.plot_convergence(os.path.join(current_directory, 'results.txt'), run_number=1).show()
