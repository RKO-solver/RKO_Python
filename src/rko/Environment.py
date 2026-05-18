"""
Utility functions for RKO Environment validation.
"""

def check_env(env_instance, silent=False, time_total=None, brkga=0, ms=0, sa=0, vns=0, ils=0, lns=0, pso=0, ga=0, restart=1, runs=1):
    """
    Verifies that a given environment instance correctly implements the RKO environment interface.
    If the verification passes and silent is False, it prints a beautifully formatted summary of the test configuration.
    If it fails, it raises a clear and informative error detailing exactly what was missing or incorrect.

    Args:
        env_instance (object): The environment instance to validate.
        silent (bool): If True, suppresses printing the verification summary upon success.
        time_total (float, optional): Total execution time limit in seconds.
        brkga (int): Number of parallel BRKGA instances requested.
        ms (int): Number of parallel MultiStart instances requested.
        sa (int): Number of parallel Simulated Annealing instances requested.
        vns (int): Number of parallel VNS instances requested.
        ils (int): Number of parallel ILS instances requested.
        lns (int): Number of parallel LNS instances requested.
        pso (int): Number of parallel PSO instances requested.
        ga (int): Number of parallel GA instances requested.
        restart (float): Fraction of total time for each restart cycle.
        runs (int): Number of times to repeat the experiment.

    Raises:
        TypeError, ValueError, AttributeError: If the validation fails.
    """
    missing_elements = []
    
    # 1. Required basic attributes and their expected types
    required_attrs = {
        'tam_solution': (int, "Solution dimension / random-key length (int)"),
        'LS_type': (str, "Local Search exploration strategy ('Best' or 'First')"),
        'dict_best': (dict, "Best known objective bound dictionary (dict)"),
        'instance_name': (str, "Problem instance identifier string (str)")
    }
    
    for attr, (expected_type, desc) in required_attrs.items():
        if not hasattr(env_instance, attr):
            missing_elements.append(f"[ERROR] Missing Attribute: '{attr}' ({desc})")
        else:
            val = getattr(env_instance, attr)
            if not isinstance(val, expected_type):
                missing_elements.append(f"[ERROR] Type Mismatch: Attribute '{attr}' must be of type {expected_type.__name__}, got {type(val).__name__}")

    # 2. Specific value constraints
    if not missing_elements:
        if env_instance.tam_solution <= 0:
            missing_elements.append(f"[ERROR] Invalid Value: 'tam_solution' must be a positive integer greater than zero (got {env_instance.tam_solution})")
        if env_instance.LS_type not in ('Best', 'First'):
            missing_elements.append(f"[ERROR] Invalid Value: 'LS_type' must be either 'Best' or 'First' (got '{env_instance.LS_type}')")

    # 3. Dynamic check of metaheuristic parameter dictionaries (ONLY those requested)
    requested_solvers = {
        'BRKGA': (brkga, 'BRKGA_parameters'),
        'SA': (sa, 'SA_parameters'),
        'ILS': (ils, 'ILS_parameters'),
        'VNS': (vns, 'VNS_parameters'),
        'PSO': (pso, 'PSO_parameters'),
        'GA': (ga, 'GA_parameters'),
        'LNS': (lns, 'LNS_parameters'),
        'MS': (ms, 'MS_parameters')
    }
    
    for name, (count, attr_name) in requested_solvers.items():
        if count > 0:
            if name == 'MS':
                continue  # MultiStart relies strictly on local search / doesn't require extra parameters
            if not hasattr(env_instance, attr_name):
                missing_elements.append(f"[ERROR] Missing Parameter Dictionary: '{attr_name}' (required because {name} is active with count={count})")
            else:
                val = getattr(env_instance, attr_name)
                if not isinstance(val, dict):
                    missing_elements.append(f"[ERROR] Type Mismatch: Parameter '{attr_name}' must be of type dict, got {type(val).__name__}")
                else:
                    # Validate that all values are lists of numbers
                    for k, v in val.items():
                        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                            missing_elements.append(f"[ERROR] Format Exception: '{attr_name}' key '{k}' must be a list of numeric values (e.g. [100, 50])")

    # 4. Required methods
    required_methods = ['decoder', 'cost']
    for method in required_methods:
        if not hasattr(env_instance, method) or not callable(getattr(env_instance, method)):
            missing_elements.append(f"[ERROR] Missing or Non-Callable Method: '{method}'")

    # If any validation item failed, compile a clear, professional error in English and raise it
    if missing_elements:
        error_msg = "\n" + "=" * 70 + "\n"
        error_msg += "               RKO ENVIRONMENT VALIDATION ERROR                       \n"
        error_msg += "=" * 70 + "\n"
        error_msg += "The provided environment instance does not conform to the RKO Solver\n"
        error_msg += "specification for this execution. The following components are missing or invalid:\n\n"
        error_msg += "\n".join(missing_elements)
        error_msg += "\n\n"
        error_msg += "Please ensure the environment class defines all required solver interface\n"
        error_msg += "attributes and callable methods.\n"
        error_msg += "=" * 70 + "\n"
        raise ValueError(error_msg)

    # 5. Output a stunning and clean summary reinforcing the configuration (ASCII safe)
    if not silent:
        print("\n" + "=" * 60)
        print("  RKO ENVIRONMENT VERIFIED SUCCESSFULLY!  ".center(60))
        print("=" * 60)
        print(f" * Instance Name:       {env_instance.instance_name}")
        print(f" * Key Size:            {env_instance.tam_solution} random keys")
        print(f" * Local Search (LS):   {env_instance.LS_type}")
        print(f" * Total Time Limit:    {time_total} seconds")
        print(f" * Runs / Restarts:     Runs={runs}, Restart Cycle Fraction={restart}")
        print(" * Active Metaheuristics:")
        
        any_active = False
        for name, (count, attr_name) in requested_solvers.items():
            if count > 0:
                any_active = True
                if name == 'MS':
                    print(f"    > {name:<6} [x{count}]: (No parameters required)")
                else:
                    params = getattr(env_instance, attr_name)
                    param_desc = ", ".join(f"{k}={v}" for k, v in params.items())
                    meta_name = attr_name.replace('_parameters', '').upper()
                    print(f"    > {meta_name:<6} [x{count}]: {param_desc}")
        if not any_active:
            print("    > (No metaheuristics selected to run!)")
        print("=" * 60 + "\n")
