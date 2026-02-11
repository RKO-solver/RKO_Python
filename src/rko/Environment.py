from abc import ABC, abstractmethod
import numpy as np

class RKOEnvAbstract(ABC):
    """
    Abstract Base Class for creating a problem environment compatible with the RKO solver.

    This class serves as a template and enforces the implementation of essential
    methods and attributes required by the RKO framework. To solve a new problem,
    create a new class that inherits from this one, implement all abstract methods,
    and define all required attributes.
    """
    def __init__(self):
        # --- Required Attributes ---

        # The length of the random-key vector used to represent a solution.
        # This must be an integer.
        self.tam_solution: int = 0

        # The maximum execution time in seconds for the solver.
        self.max_time: int = 200

        # The local search strategy: 'Best' for best-improvement or 'First' for first-improvement.
        self.LS_type: str = 'Best'

        # A dictionary mapping instance names to their known best (or optimal) solution values.
        # Used for logging progress (GAP) and for early stopping if the optimal is found.
        # Example: {"instance_1": 123.45, "instance_2": 678.90}
        self.dict_best: dict = {}
        
        # The name of the current problem instance being solved.
        self.instance_name: str = "default_instance"
        
        # --- Metaheuristic Parameter Configuration ---
        # For each metaheuristic, define a dictionary where keys are parameter names
        # and values are lists of possible numerical values.
        # - For OFFLINE (static) tuning, provide a list with a single value for each parameter.
        # - For ONLINE (dynamic) tuning with Q-Learning, provide a list with multiple values.

        self.BRKGA_parameters: dict = {
            'p': [100],      # Population size
            'pe': [0.20],    # Elite set fraction
            'pm': [0.10],    # Mutant set fraction
            'rhoe': [0.70]   # Elite parent inheritance probability
        }
        self.SA_parameters: dict = {
            'SAmax': [50],     # Iterations per temperature
            'alphaSA': [0.99], # Cooling rate
            'betaMin': [0.05], # Min shaking intensity
            'betaMax': [0.25], # Max shaking intensity
            'T0': [10000]      # Initial temperature
        }
        self.ILS_parameters: dict = {
            'betaMin': [0.10],
            'betaMax': [0.20]
        }
        self.VNS_parameters: dict = {
            'kMax': [5],
            'betaMin': [0.05]
        }
        self.PSO_parameters: dict = {
            'PSize': [100],    # Swarm size
            'c1': [2.05],      # Cognitive coefficient
            'c2': [2.05],      # Social coefficient
            'w': [0.73]        # Inertia weight
        }
        self.GA_parameters: dict = {
            'sizePop': [100],
            'probCros': [0.98],
            'probMut': [0.005]
        }
        self.LNS_parameters: dict = {
            'betaMin': [0.10],
            'betaMax': [0.30],
            'TO': [1000],
            'alphaLNS': [0.95]
        }



    @abstractmethod
    def decoder(self, keys: np.ndarray):
        """
        Translates a random-key vector into a feasible solution for the problem.

        This is the most critical method to implement for a new problem. It defines
        the mapping from the continuous search space [0,1)^n to the discrete
        solution space of the problem.

        Args:
            keys (np.ndarray): A NumPy array of floats in [0, 1), representing a solution.

        Returns:
            any: A representation of a feasible solution for the problem (e.g., a list, a custom object).
        """
        pass

    @abstractmethod
    def cost(self, solution, final_solution: bool = False) -> float:
        """
        Calculates the objective function value (cost) of a decoded solution.

        The RKO framework is designed to MINIMIZE this value. If you are solving a
        maximization problem, you should return the negative of the objective value
        (e.g., return -total_profit).

        Args:
            solution: The feasible solution returned by the `decoder` method.
            final_solution (bool): An optional flag that can be used to trigger
                                   additional actions (like plotting) for the final solution.

        Returns:
            float: The numerical cost of the solution.
        """
        pass

def check_env(env_instance: RKOEnvAbstract):
    """
    Verifies that a given environment instance correctly implements the RKOEnvAbstract interface.

    This function checks for the presence and correct types of all required
    attributes and methods. It raises informative errors if any part of the
    implementation is missing or incorrect.

    Args:
        env_instance (RKOEnvAbstract): An instance of a class designed to be an RKO environment.

    Raises:
        AssertionError: If the instance does not inherit from RKOEnvAbstract.
        AttributeError: If a required attribute or method is missing.
        TypeError: If an attribute has the wrong type.
        ValueError: If an attribute has an invalid value or a parameter dictionary is incorrectly structured.
    """
    print("--- Starting RKO Environment Check ---")
    
    # Check if the class inherits from the abstract base class
    if not isinstance(env_instance, RKOEnvAbstract):
        raise AssertionError("The provided environment instance does not inherit from 'RKOEnvAbstract'.")
    print("✅ Inheritance from RKOEnvAbstract is confirmed.")

    # List of required attributes and their expected types
    required_attrs = {
        'tam_solution': int,
        'max_time': (int, float),
        'LS_type': str,
        'dict_best': dict,
        'instance_name': str,
        'save_q_learning_report': bool
    }

    # List of required metaheuristic parameter dictionaries
    param_dicts = [
        'BRKGA_parameters', 'SA_parameters', 'ILS_parameters', 'VNS_parameters',
        'PSO_parameters', 'GA_parameters', 'LNS_parameters'
    ]

    # Check for required attributes and their types
    for attr, expected_type in required_attrs.items():
        if not hasattr(env_instance, attr):
            raise AttributeError(f"Environment missing required attribute: '{attr}'")
        if not isinstance(getattr(env_instance, attr), expected_type):
            raise TypeError(f"Attribute '{attr}' has incorrect type. Expected {expected_type}, got {type(getattr(env_instance, attr))}.")
    print("✅ Basic attributes are present and have the correct types.")

    # Check specific attribute values
    if getattr(env_instance, 'tam_solution') <= 0:
        raise ValueError("Attribute 'tam_solution' must be a positive integer.")
    if getattr(env_instance, 'LS_type') not in ['Best', 'First']:
        raise ValueError("Attribute 'LS_type' must be either 'Best' or 'First'.")
    print("✅ Attribute values are valid.")

    # Check for parameter dictionaries
    for param_dict_name in param_dicts:
        if not hasattr(env_instance, param_dict_name):
            raise AttributeError(f"Environment missing required parameter dictionary: '{param_dict_name}'")
        param_dict = getattr(env_instance, param_dict_name)
        if not isinstance(param_dict, dict):
            raise TypeError(f"'{param_dict_name}' must be a dictionary.")
        for key, value in param_dict.items():
            if not isinstance(value, list) or not all(isinstance(v, (int, float)) for v in value):
                raise ValueError(f"Invalid format in '{param_dict_name}'. Value for key '{key}' must be a list of numbers.")
    print("✅ Metaheuristic parameter dictionaries are correctly structured.")

    # Check for required methods
    required_methods = ['decoder', 'cost']
    for method in required_methods:
        if not hasattr(env_instance, method) or not callable(getattr(env_instance, method)):
            # This check is somewhat redundant if inheritance is enforced, but good for clarity.
            raise AttributeError(f"Environment missing required method: '{method}'")
    print("✅ Required methods ('decoder', 'cost') are implemented.")

    print("\n🎉 --- Environment Check Passed Successfully! --- 🎉")
