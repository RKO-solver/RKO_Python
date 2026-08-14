import numpy as np
import os
import sys
# Ensure the 'src' directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rko import RKO, DualLogger, HistoryPlotter

class KnapsackProblem:
    """
    An implementation of the Knapsack Problem environment for the RKO solver.
    """
    n_items: int
    capacity: int
    profits: list[int]
    weights: list[int]
    tam_solution: int
    LS_type: str
    instance_name: str

    def __init__(self, instance_path: str):
        print(f"Loading Knapsack Problem instance from: {instance_path}")

        self.instance_name = os.path.basename(instance_path)
        self.LS_type = 'Best' # Options: 'Best' or 'First'
        self.dict_best: dict = {"Best": [-149]}
        self._load_data(instance_path)

        # --- Set required attributes ---
        self.tam_solution = self.n_items
        
        self.BRKGA_parameters = {
            'p': [10000, 5000],
            'pe': [0.20, 0.15],      
            'pm': [0.05],        
            'rhoe': [0.70]       
        }

        self.SA_parameters = {
            'SAmax': [1000, 500],
            'alphaSA': [0.99, 0.9, 0.95],  
            'betaMin': [0.01, 0.03],   
            'betaMax': [0.05, 0.1],   
            'T0': [100000]
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
            'PSize': [10000,5000],
            'c1': [2.05],     
            'c2': [2.05],        
            'w': [0.73]         
        }

        
        self.GA_parameters = {
            'sizePop': [1000,500],    
            'probCros': [0.98],  
            'probMut': [0.005, 0.01]   
        }

        
        self.LNS_parameters = {
            'betaMin': [0.10],   
            'betaMax': [0.30],  
            'TO': [10000],       
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

    def decoder(self, keys: list[float]) -> list[int]:
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
            if item_included == 1:
                total_profit += self.profits[i]
                total_weight += self.weights[i]

        # Apply a heavy penalty for infeasible solutions (exceeding capacity)
        if total_weight > self.capacity:
            penalty = 100000 * (total_weight - self.capacity)
            total_profit -= penalty
            
        # The RKO framework assumes a minimization problem by default,
        # so we return the negative of the profit.
        return float(-total_profit)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    env = KnapsackProblem(os.path.join(current_directory,'kp50.txt'))
    solver = RKO(env, logger="none", log_filepath=os.path.join(current_directory,'results.txt'))
    solp = 0
    solc = 0

    temp = 0
    temc = 0
    for i in range(10):
        ret1 = solver.solve(time_total=3, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, runs=1, plot=True, backend='cpp')
        ret2 = solver.solve(time_total=3, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, runs=1, plot=True, backend='python')
        print(i,ret1[0], ret2[0])
        solc += ret1[0]
        solp += ret2[0]
        temc += ret1[2]
        temp += ret2[2]

    print(temc/10, solc/10)

    print(temp/10, solp/10)
