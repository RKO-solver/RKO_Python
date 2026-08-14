"""
Utility functions for RKO Environment validation.
"""


def check_env_py(env_instance, silent=False, time_total=None, brkga=0, ms=0, sa=0, vns=0, ils=0, lns=0, pso=0, ga=0, restart=1, runs=1):
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


def check_env_cpp(env_instance, silent=False, time_total=None, brkga=0, ms=0, sa=0, vns=0, ils=0, lns=0, pso=0, ga=0, restart=1, runs=1):
    """
    Verifies that a given environment instance correctly implements the RKO environment interface
    AND is fully compatible with the RKO Native V1 C++ compiler.
    """
    # 1. Run standard Python interface validation first
    check_env_py(
        env_instance,
        silent=True,
        time_total=time_total,
        brkga=brkga,
        ms=ms,
        sa=sa,
        vns=vns,
        ils=ils,
        lns=lns,
        pso=pso,
        ga=ga,
        restart=restart,
        runs=runs,
    )

    # 2. Run static V1 AST analysis
    from .compiler import analyze_environment
    report = analyze_environment(env_instance)

    if not report["supported"]:
        from pathlib import Path

        class_name = report.get("class_name", type(env_instance).__name__)
        source_file = report.get("source_file", "")

        diag_blocks = []
        for diag in report["diagnostics"]:
            code = diag.get("code", "V1-ERROR")
            msg = diag.get("message", "")
            span = diag.get("span", {})
            file_path = span.get("filename") or source_file
            line_num = span.get("line")
            col_num = span.get("column")
            adaptation = diag.get("adaptation")

            location_str = f"{file_path}"
            if line_num:
                location_str += f" (Line {line_num}"
                if col_num:
                    location_str += f", Col {col_num}"
                location_str += ")"

            snippet = ""
            if file_path and Path(file_path).is_file() and line_num:
                try:
                    file_lines = Path(file_path).read_text(encoding="utf-8").splitlines()
                    if 1 <= line_num <= len(file_lines):
                        code_text = file_lines[line_num - 1]
                        snippet = f"\n    Source Line {line_num}:\n      >  {code_text.strip()}"
                except Exception:
                    pass

            block = f"  [ERROR {code}]\n  File:    {location_str}\n  Message: {msg}{snippet}"
            if adaptation:
                block += f"\n  Suggestion: {adaptation}"

            # Provide specific actionable fix advice for common errors
            if "np.ndarray" in msg or "ndarray" in msg or "RKO-NATIVE-TYPE-002" in code:
                block += (
                    "\n\n  HOW TO FIX THIS:\n"
                    "    The C++ compiler requires explicit 1D/2D typed vectors (e.g. list[float] or list[int])\n"
                    "    instead of dynamic NumPy arrays.\n"
                    "    1. Change the class annotation to list[float] (or list[int]).\n"
                    "    2. When passing a NumPy array to __init__, call .tolist(), e.g.:\n"
                    "       self.my_data = my_array.tolist()\n"
                )
            elif "RKO-NATIVE-EFFECT" in code or "mutation" in msg.lower():
                block += (
                    "\n\n  HOW TO FIX THIS:\n"
                    "    Attributes on 'self' are frozen in the C++ decoder after __init__.\n"
                    "    Store temporary search state in local variables (e.g. is_allocated = [False] * N)\n"
                    "    rather than modifying self.<attribute> inside decoder() or cost().\n"
                )
            diag_blocks.append(block)

        error_msg = "\n" + "=" * 75 + "\n"
        error_msg += f"      RKO NATIVE C++20 COMPILER VALIDATION ERROR in class '{class_name}'      \n"
        error_msg += "=" * 75 + "\n"
        error_msg += "The environment cannot be compiled to C++20 because it violates Native V1 rules.\n\n"
        error_msg += "\n\n".join(diag_blocks)
        error_msg += "\n\n" + "=" * 75 + "\n"
        raise ValueError(error_msg)

    # 3. Output summary if not silent
    if not silent:
        print("\n" + "=" * 60)
        print("  RKO NATIVE C++20 ENVIRONMENT VERIFIED SUCCESSFULLY!  ".center(60))
        print("=" * 60)
        print(f" * Instance Name:       {env_instance.instance_name}")
        print(f" * Class Name:          {report.get('class_name', type(env_instance).__name__)}")
        print(f" * Key Size:            {env_instance.tam_solution} random keys")
        print(f" * V1 Compiler Status:  COMPATIBLE (C++20 Ready)")
        print(f" * Total Time Limit:    {time_total} seconds")
        print(f" * Runs / Restarts:     Runs={runs}, Restart Cycle Fraction={restart}")
        print(" * Active Metaheuristics (C++ OpenMP Threads):")

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

        any_active = False
        for name, (count, attr_name) in requested_solvers.items():
            if count > 0:
                any_active = True
                print(f"    > {name:<6} [x{count}]: (Native C++ Parallel Thread)")
        if not any_active:
            print("    > MultiStart [x1]: (Default Native C++ Parallel Thread)")
        print("=" * 60 + "\n")


def check_env(env_instance, silent=False, time_total=None, brkga=0, ms=0, sa=0, vns=0, ils=0, lns=0, pso=0, ga=0, restart=1, runs=1, backend="python"):
    """
    Unified environment checker for RKO environments supporting 'python' and 'cpp' backends.
    """
    backend_choice = backend.lower().strip() if isinstance(backend, str) else "python"
    if backend_choice == "python":
        return check_env_py(
            env_instance,
            silent=silent,
            time_total=time_total,
            brkga=brkga,
            ms=ms,
            sa=sa,
            vns=vns,
            ils=ils,
            lns=lns,
            pso=pso,
            ga=ga,
            restart=restart,
            runs=runs,
        )
    elif backend_choice in ("cpp", "c++", "native"):
        return check_env_cpp(
            env_instance,
            silent=silent,
            time_total=time_total,
            brkga=brkga,
            ms=ms,
            sa=sa,
            vns=vns,
            ils=ils,
            lns=lns,
            pso=pso,
            ga=ga,
            restart=restart,
            runs=runs,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Expected 'python' or 'cpp'.")
