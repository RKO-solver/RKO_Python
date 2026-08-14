# RKO - Random-Key Optimizer (Python Parallel Framework)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

The **Random-Key Optimizer (RKO)** is a high-performance, parallel metaheuristic framework designed for combinatorial optimization problems. By mapping discrete, high-dimensional search spaces onto a continuous unit hypercube $[0, 1)^n$, RKO detaches optimization algorithms from problem-specific logic.

RKO orchestrates up to **8 parallel metaheuristic workers** using Python's native multiprocessing. Workers cooperate in real time by feeding and pulling elite solutions from a thread-safe, shared **Solution Pool**, accelerating convergence and improving optimization robustness.

---

## 💎 Core Features

- **8 Concurrent Metaheuristics:** Run **BRKGA, Multi-Start, SA, VNS, ILS, LNS, PSO, and GA** concurrently.
- **Online hyperparameters tuning via Q-learning**
- **Collaborative Search:** Thread-safe `SolutionPool` sharing elite solutions to pull workers out of local minima.
- **Automatic Result Structuring:** Automatically groups execution runs, creating a dedicated timestamped folder (`results_{instance}_{timestamp}/`) containing:
  - `results.txt`: Scientific summary containing aggregated averages, standard deviations, lists of costs, and computational times for academic reporting.
  - `logs.txt`: Chronological trace of improvements (`NEW BEST`), optimized and clean for parser engines.
  - `run_x.png`: Convergence trajectory plot generated automatically for each trial.

---

## 📦 Installation

```bash
pip install rko
```

---

## ⚙️ Solver Configuration

### RKO Class Initializer

```python
from rko import RKO

rko_solver = RKO(
    env=my_environment,
    best_possible=None,     # early-stop threshold
    logger="dual",          # logger strategy or custom LogStrategy instance
    log_filepath="./results/results.txt" # Base directory for structured outputs
)
```

### Solve Method Parameters

```python
final_cost, final_solution, time_to_best = rko_solver.solve(
    time_total=30,      # Total time limit in seconds
    brkga=1,            # Number of parallel BRKGA instances
    ms=1,               # Number of parallel Multi-Start instances
    sa=1,               # Number of parallel SA instances
    vns=1,              # Number of parallel VNS instances
    ils=1,              # Number of parallel ILS instances
    lns=1,              # Number of parallel LNS instances
    pso=1,              # Number of parallel PSO instances
    ga=1,               # Number of parallel GA instances
    restart=1.0,        # Fraction of total time for each restart cycle
    runs=10,            # Number of independent trials
    plot=True           # Enable automatic convergence plotting (generates .png charts)
)
```

---

## 📊 Extensible Logging Engine

The RKO logging subsystem is built on the **Strategy Pattern**, allowing developers to customize logging behavior by passing a subclass of `LogStrategy` directly to the `logger` parameter in the `RKO` constructor.

### Built-in Logger Strategies

- **`"dual"` (TerminalLogger with filepath):** Prints logs to stdout and saves structured records to the dynamic results directory.
- **`"terminal"` (TerminalLogger):** Outputs logs exclusively to stdout.
- **`"file"` (FileLogger):** Saves logs exclusively to the specified disk path.

### Custom Logging Strategy Example

You can inherit from `LogStrategy` to route logs to databases, dashboards, or external analytical tools:

```python
from rko.LogStrategy import LogStrategy

class DatabaseLogger(LogStrategy):
    def __init__(self, db_connection):
        self.db = db_connection

    def log(self, *args, **kwargs):
        msg = " ".join(str(arg) for arg in args)
        # Custom logic to write to database
        self.db.insert_log(msg)
```

## 🛠️ Environment Interface & Validation

To adapt RKO to your problem, subclass `RKOEnvAbstract` and implement `decoder` and `cost`.

```python
,
import numpy as np

class RKOEnv():
    def __init__(self):
        # Required attributes
        self.tam_solution: int = 50           # Dimension of the random key vector
        self.LS_type: str = 'Best'            # Local Search strategy ('Best' or 'First')
        self.dict_best: dict = {}             # Shared best dictionary
        self.instance_name: str = "kp50.txt"  # Name of your problem instance

        # Configurations for Metaheuristics
        self.BRKGA_parameters = {'p': [1000, 2000], 'pe': [0.2, 0.15], 'pm': [0.05], 'rhoe': [0.7]}
        self.SA_parameters = {'SAmax': [100], 'alphaSA': [0.99, 0.9], 'betaMin': [0.01], 'betaMax': [0.05], 'T0': [10000]}
        self.ILS_parameters = {'betaMin': [0.1], 'betaMax': [0.2]}
        self.VNS_parameters = {'kMax': [8,7,6,5], 'betaMin': [0.05]}
        self.LNS_parameters = {'betaMin': [0.1], 'betaMax': [0.3], 'TO': [1000], 'alphaLNS': [0.99, 0.95, 0.9]}
        self.PSO_parameters = {'PSize': [1000], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]}
        self.GA_parameters = {'sizePop': [1000], 'probCros': [0.98], 'probMut': [0.005]}

  
    def decoder(self, keys: np.ndarray):
        """Maps random keys [0, 1) into a problem-specific discrete representation."""
        pass

  
    def cost(self, solution, final_solution: bool = False) -> float:
        """Returns objective cost (RKO minimizes this; return negated value for maximization)."""
        pass
```

### Static and Functional Verification via `check_env`

RKO provides a built-in diagnostic utility `check_env` to validate custom environments before runtime. It performs the following verification passes:

1. **Attribute & Type Verification:** Verifies the presence and types of all core variables (`tam_solution`, `LS_type`, `dict_best`, `instance_name`) and metaheuristic parameter dictionaries.
2. **Functional Dry-Run:** Invokes the custom `decoder` and `cost` methods using mock random keys to verify execution safety.
3. **Mathematical Compliance:** Checks that the returned cost is a scalar float and that the decoder successfully processes vectors of dimensions defined by `tam_solution`.
4. **Actionable Diagnostics:** If a check fails, the utility prints detailed stack traces and configuration hints to assist the developer in resolving interface mismatches or bugs.

```python
from rko.Environment import check_env

my_env = MyProblemEnv()
if check_env(my_env):
    print("Environment successfully validated.")
```

### Experimental native compiler V1

An initialized environment whose hot path belongs to the documented native
subset can be compiled into the original C++ RKO's `Problem.h`:

```python
from rko.compiler import analyze_environment, optimize_environment

report = analyze_environment(my_env)
if report["supported"]:
    result = optimize_environment(my_env, "native_run", max_time_seconds=10)
    print(result.solution, result.python_cost)
else:
    print(report["diagnostics"])
```

The complete language contract, supported operations and adaptation rules are
in [Native Decoder Language V1](docs/design/native_decoder_language_v1.md).
Windows builds and tests invoke `g++` through WSL.

---

## 👥 Maintainers

- **Felipe Silvestre Cardoso Roberto** - [LinkedIn](https://www.linkedin.com/in/felipesilvestrecr/)
- **João Victor Assaoka Ribeiro** - [LinkedIn](https://www.linkedin.com/in/assaoka/)
